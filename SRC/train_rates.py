# scripts/train_rates.py
import argparse, os, time
import torch
from torch.amp import GradScaler, autocast

from itoformer.utils.config import load_config
from itoformer.training.datamodules_rates import RatesDataModule
from itoformer.utils.csvlog import append_row, Stopwatch

from itoformer.models.itoformer import ItoFormer
from itoformer.models.heads.term_structure import TermStructureHead

# ---------- HJM helpers ----------
def ito_curvature_penalty(h: torch.Tensor) -> torch.Tensor:
    # h: [B, T, d]
    if h is None or h.ndim != 3 or h.size(1) < 3:
        return h.new_tensor(0.0) if isinstance(h, torch.Tensor) else torch.tensor(0.0)
    curv = h[:, 2:, :] - 2*h[:, 1:-1, :] + h[:, :-2, :]
    return curv.pow(2).mean()

def _yields_to_discounts(y: torch.Tensor, tenors: torch.Tensor) -> torch.Tensor:
    # y: [B, L, N]; tenors: [N]
    return torch.exp(-y * tenors.view(1, 1, -1))

def _discounts_to_forwards(P: torch.Tensor, tenors: torch.Tensor) -> torch.Tensor:
    dlnP = -torch.diff(torch.log(P + 1e-12), dim=2)          # [B, L, N-1]
    dT   = torch.diff(tenors).view(1, 1, -1).clamp_min(1e-6) # [1,1,N-1]
    return dlnP / dT

def _trapz_suffix_integral(sig: torch.Tensor, tenors: torch.Tensor) -> torch.Tensor:
    dT  = torch.diff(tenors).view(1, -1)                     # [1, N-1]
    mid = 0.5 * (sig[:, :-1] + sig[:, 1:]) * dT              # [B, N-1]
    integ = torch.zeros_like(sig)
    integ[:, :-1] = torch.flip(torch.cumsum(torch.flip(mid, dims=[1]), dim=1), dims=[1])
    integ[:, -1] = 0.0
    return integ

def hjm_drift_penalty_from_window(
    xb: torch.Tensor, pred_next_yields: torch.Tensor, tenors_years: torch.Tensor,
    n_tenors: int, lookback: int, eps: float
):
    """
    xb: [B, L, N] (first N features are the yields per tenor)
    pred_next_yields: [B, N]
    returns:
      pen (scalar), viol (scalar), gap_per_tenor ([N-1]) = mean |μ̂(τ)-α̂(τ)| across batch
    """
    B, L, N = xb.shape
    Lw = min(int(lookback), L)

    Y = xb[:, -Lw:, :N]                       # [B, Lw, N]
    Ynext = pred_next_yields.unsqueeze(1)     # [B, 1, N]
    Yseq  = torch.cat([Y, Ynext], dim=1)      # [B, Lw+1, N]

    Pseq = _yields_to_discounts(Yseq, tenors_years.to(Yseq.device))
    Fseq = _discounts_to_forwards(Pseq, tenors_years.to(Yseq.device))     # [B, Lw+1, N-1]
    dF_t = torch.diff(Fseq, dim=1)                                        # [B, Lw, N-1]

    mu_hat  = dF_t.mean(dim=1)                                            # [B, N-1]
    sig_hat = dF_t.std(dim=1).clamp_min(1e-8)                             # [B, N-1]

    tau_fwd = tenors_years[1:]
    integ = _trapz_suffix_integral(sig_hat, tau_fwd.to(sig_hat.device))   # [B, N-1]
    alpha_hat = sig_hat * integ                                           # [B, N-1]

    diff = mu_hat - alpha_hat                                             # [B, N-1]
    pen  = diff.pow(2).mean()
    viol = (diff.abs() > eps).float().mean()
    # per-tenor mean absolute gap across batch
    gap_per_tenor = diff.abs().mean(dim=0)                                # [N-1]
    return pen, viol, gap_per_tenor

# ---------- Model factory ----------
def make_model(cfg, n_tenors: int):
    # Ensure ItoFormer knows the input shape (A=1, F=N)
    cfg["model"]["n_assets"] = 1
    cfg["model"]["in_features"] = n_tenors
    cfg.setdefault("data", {}).setdefault("features_per_asset", n_tenors)

    def head_factory(name: str):
        if name == "term_structure":
            d_model = int(cfg["model"]["encoder"]["d_model"])
            return TermStructureHead(d_model, n_tenors)
        return None

    return ItoFormer.build_from_config(cfg, head_factory)

def _f(x, default=0.0):
    try:
        if isinstance(x, torch.Tensor):
            # handle CUDA/AMP scalars robustly
            if x.numel() == 1:
                return float(x.detach().item())
            else:
                return float(x.detach().mean().item())
        return float(x)
    except Exception:
        return float(default)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Data
    dm = RatesDataModule(cfg); dm.setup()
    sample_x, sample_y = dm.train_ds[0]          # X: [L, N], Y: [N]
    n_tenors = int(sample_y.shape[-1])

    device = "cuda" if (cfg.get("device","cpu") == "cuda" and torch.cuda.is_available()) else "cpu"

    # CSV logging
    log_cfg = cfg.get("logging", {})
    csv_enabled = bool(log_cfg.get("csv", False))
    csv_dir = os.path.normpath(log_cfg.get(
        "csv_dir",
        os.path.join(cfg["paths"]["outputs_root"], cfg["experiment_name"], "logs"),
    ))
    os.makedirs(csv_dir, exist_ok=True)
    train_csv = os.path.join(csv_dir, "train.csv")
    val_csv   = os.path.join(csv_dir, "val_summary.csv")
    train_header = [
        "epoch","split","loss_total","loss_pred_mse","loss_hjm_drift","loss_ito",
        "mse","mae","rmse","hjm_violation_rate","lr","grad_norm","wall_time_s"
    ]
    val_header = ["epoch","mse","mae","rmse","loss_hjm_drift","hjm_violation_rate","loss_ito"]

    # Diagnostics CSV for per-tenor gaps
    hjm_diag_path = cfg.get("eval",{}).get("outputs",{}).get(
        "hjm_csv",
        os.path.join(cfg["paths"]["outputs_root"], cfg["experiment_name"], "diagnostics", "hjm_drift.csv")
    )
    os.makedirs(os.path.dirname(hjm_diag_path), exist_ok=True)
    hjm_diag_header = ["epoch","split","tenor_idx","mean_abs_gap"]

    # AMP
    use_amp   = bool(cfg["train"]["amp"]["enabled"]) and (device == "cuda")
    precision = str(cfg["train"]["amp"].get("precision","bf16")).lower()
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler    = GradScaler(device if device=="cuda" else "cpu", enabled=use_amp)

    # Model / Optim
    model = make_model(cfg, n_tenors).to(device)
    opt   = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["optimizer"]["lr"]),
        betas=tuple(cfg["train"]["optimizer"].get("betas",[0.9,0.99])),
        weight_decay=float(cfg["train"]["optimizer"].get("weight_decay",1e-4)),
    )
    epochs    = int(cfg["train"].get("epochs", 1))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    # HJM + Ito weights
    hjm_enabled = bool(cfg["losses"]["no_arbitrage"].get("enabled", True))
    hjm_w       = float(cfg["losses"]["no_arbitrage"].get("hjm_drift_weight", 0.0)) if hjm_enabled else 0.0
    hjm_eps     = float(cfg["losses"]["no_arbitrage"].get("hjm_eps", 5e-4))
    ito_enabled = bool(cfg["losses"]["ito_consistency"].get("enabled", False))
    ito_w       = float(cfg["losses"]["ito_consistency"].get("weight", 0.0)) if ito_enabled else 0.0

    # Tenor grid for HJM
    tenors_years = torch.as_tensor(cfg["data"]["rates"]["tenors_years"], dtype=torch.float32, device=device)
    lookback     = int(cfg["data"]["rates"].get("lookback_for_hjm", 16))

    timer = Stopwatch()
    for ep in range(1, epochs+1):
        model.train()
        agg = dict(mse=0.0, mae=0.0, rmse=0.0, hjm=0.0, hjmvr=0.0, ito=0.0, grad=0.0, n=0)
        t0 = time.time()

        for xb, yb in dm.train_loader():
            # xb: [B, L, N] -> add asset axis A=1 to match ItoFormer [B,T,A,F]
            xb = xb.to(device); yb = yb.to(device)
            xb4 = xb.unsqueeze(2)  # [B, L, 1, N]

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out  = model(xb4)
                pred = out["term_structure"] if isinstance(out, dict) else out  # [B, N]

                mse  = (pred - yb).pow(2).mean()
                mae  = (pred - yb).abs().mean()
                rmse = mse.sqrt()
                loss = mse

                # HJM penalty + violation rate (+ per-tenor gaps)
                if hjm_enabled and hjm_w > 0.0:
                    hjm_pen, hjm_vr, gap_per_tenor = hjm_drift_penalty_from_window(
                        xb, pred, tenors_years, n_tenors=n_tenors, lookback=lookback, eps=hjm_eps
                    )
                    loss = loss + hjm_w * hjm_pen
                else:
                    hjm_pen = torch.tensor(0.0, device=device)
                    hjm_vr  = torch.tensor(0.0, device=device)
                    gap_per_tenor = None

                # Ito curvature (fixed 11/2/2025 at 1:25pm)
                ito_pen = torch.tensor(0.0, device=device)
                if ito_enabled and ito_w > 0.0:
                    ito_pen = ito_curvature_penalty(model.last_hidden)
                    loss = loss + ito_w * ito_pen

                # sanity check for 1st batch of epoch
                if ep == 1 and agg["n"] == 0 and (ito_enabled and ito_w > 0.0):
                    lh = model.last_hidden
                    lh_shape = None if lh is None else tuple(lh.shape)
                    ito_val = float(ito_pen.detach().item()) if isinstance(ito_pen, torch.Tensor) else float(ito_pen)
                    print(f"[sanity][train] last_hidden={lh_shape} ito_pen={ito_val}", flush=True)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                g = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                g = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            agg["mse"]  += float(mse.item())
            agg["mae"]  += float(mae.item())
            agg["rmse"] += float(rmse.item())
            agg["hjm"]  += _f(hjm_pen)
            agg["hjmvr"]+= _f(hjm_vr)
            agg["ito"]  += _f(ito_pen)
            agg["grad"] += float(g if isinstance(g, float) else g.item())
            agg["n"]    += 1

            # Per-tenor diagnostics (TRAIN): one row per tenor
            if csv_enabled and gap_per_tenor is not None:
                gt = gap_per_tenor.detach().float().cpu().tolist()  # length N-1
                for ti, gap in enumerate(gt):
                    append_row(hjm_diag_path, hjm_diag_header, {
                        "epoch": ep, "split": "train", "tenor_idx": ti+1, "mean_abs_gap": gap
                    })

        n = max(1, agg["n"])
        if csv_enabled:
            append_row(train_csv, train_header, {
                "epoch": ep, "split": "train",
                "loss_total": (agg["mse"]/n) + hjm_w*(agg["hjm"]/n) + ito_w*(agg["ito"]/n),
                "loss_pred_mse": agg["mse"]/n,
                "loss_hjm_drift": agg["hjm"]/n,
                "loss_ito": agg["ito"]/n,
                "mse": agg["mse"]/n, "mae": agg["mae"]/n, "rmse": agg["rmse"]/n,
                "hjm_violation_rate": agg["hjmvr"]/n,
                "lr": opt.param_groups[0]["lr"],
                "grad_norm": agg["grad"]/n,
                "wall_time_s": time.time() - t0,
            })

        # -------- Validation --------
        model.eval()
        v = dict(mse=0.0, mae=0.0, rmse=0.0, hjm=0.0, hjmvr=0.0, ito=0.0, n=0)
        with torch.no_grad():
            for xb, yb in dm.val_loader():
                xb = xb.to(device); yb = yb.to(device)
                xb4 = xb.unsqueeze(2)  # [B, L, 1, N]

                out  = model(xb4)
                pred = out["term_structure"] if isinstance(out, dict) else out

                mse  = (pred - yb).pow(2).mean()
                mae  = (pred - yb).abs().mean()
                rmse = mse.sqrt()

                if hjm_enabled and hjm_w > 0.0:
                    hjm_pen, hjm_vr, gap_per_tenor = hjm_drift_penalty_from_window(
                        xb, pred, tenors_years, n_tenors=n_tenors, lookback=lookback, eps=hjm_eps
                    )
                else:
                    hjm_pen = torch.tensor(0.0, device=device)
                    hjm_vr  = torch.tensor(0.0, device=device)
                    gap_per_tenor = None

                ito_pen = torch.tensor(0.0, device=device)
                if ito_enabled and ito_w > 0.0:
                    ito_pen = ito_curvature_penalty(model.last_hidden)

                # sanity check again
                if ep == 1 and v["n"] == 0 and (ito_enabled and ito_w > 0.0):
                    lh = model.last_hidden
                    lh_shape = None if lh is None else tuple(lh.shape)
                    ito_val = float(ito_pen.detach().item()) if isinstance(ito_pen, torch.Tensor) else float(ito_pen)
                    print(f"[sanity][val] last_hidden={lh_shape} ito_pen={ito_val}", flush=True)


                v["mse"]  += float(mse.item())
                v["mae"]  += float(mae.item())
                v["rmse"] += float(rmse.item())
                v["hjm"]  += _f(hjm_pen)
                v["hjmvr"]+= _f(hjm_vr)
                v["ito"]  += _f(ito_pen)
                v["n"]    += 1

                # Per-tenor diagnostics (VAL)
                if csv_enabled and gap_per_tenor is not None:
                    gv = gap_per_tenor.detach().float().cpu().tolist()
                    for ti, gap in enumerate(gv):
                        append_row(hjm_diag_path, hjm_diag_header, {
                            "epoch": ep, "split": "val", "tenor_idx": ti+1, "mean_abs_gap": gap
                        })

        nv = max(1, v["n"])
        if csv_enabled:
            append_row(val_csv, val_header, {
                "epoch": ep,
                "mse": v["mse"]/nv, "mae": v["mae"]/nv, "rmse": v["rmse"]/nv,
                "loss_hjm_drift": v["hjm"]/nv, "hjm_violation_rate": v["hjmvr"]/nv, "loss_ito": v["ito"]/nv
            })

        print(f"Epoch {ep}: {time.time() - t0:.1f} seconds", flush=True)



    print("OK: rates trainer finished.")

if __name__ == "__main__":
    main()
