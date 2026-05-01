# scripts/train_options_svi.py
# SVI trainer using ItoFormer + SVIHead
# - Loss: (vega-weighted) IV RMSE + no-arb (butterfly + calendar) + param L2
# - Per-step warmup+cosine LR, mixed-precision (optional)
# - Checkpointing: best (by val loss) + last
# - Logs: logs/train.csv
# - Eval tables: eval/svi_params.csv, diagnostics/no_arbitrage_svi.csv

import argparse, math, os, time, json, random
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from itoformer.utils.config import load_config
from itoformer.training.datamodules_options import OptionsSVIDataModule
from itoformer.utils.csvlog import append_row, Stopwatch

from itoformer.models.itoformer import ItoFormer
from itoformer.models.heads.options_svi import (
    SVIHead,
    svi_iv_from_params,
    svi_total_variance,
    butterfly_penalty,
    butterfly_violation_rate,
)

# help peek script
def _csv_clean(v):
    # Write only scalars as-is; safely JSON-encode anything structured so commas are quoted.
    import json
    if v is None or isinstance(v, (int, float, str)):
        return v
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic kernels (safer reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# LR Scheduler (warmup + cosine)
# -----------------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Per-step linear warmup to base LR, then cosine decay to 0 by total_steps.
    Call .step() AFTER optimizer.step() at EACH TRAINING STEP.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # scheduler's epoch is actually "step" here
        out = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                # linear warmup 0 -> base_lr
                scale = step / float(self.warmup_steps)
            else:
                # cosine from base_lr -> 0
                progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            out.append(base_lr * scale)
        return out


# -----------------------
# Utility helpers
# -----------------------
def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            g = p.grad.data.norm(2)
            total += float(g.item() ** 2)
    return math.sqrt(total)

def make_optimizer(cfg, params):
    opt_cfg = cfg["train"]["optimizer"]
    return AdamW(
        params,
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.99])),
        weight_decay=float(opt_cfg.get("weight_decay", 1.0e-4)),
    )

def make_scheduler(cfg, optimizer, steps_per_epoch: int):
    sch_cfg = cfg["train"].get("scheduler")
    if not isinstance(sch_cfg, dict):
        return None, 0
    name = sch_cfg.get("name", "").lower()
    if name in ("cosine", "warmup_cosine", "cosine_warmup"):
        warmup_steps = int(sch_cfg.get("warmup_steps", 500))
        total_epochs = int(cfg["train"]["epochs"])
        total_steps = int(sch_cfg.get("total_steps", total_epochs * max(1, steps_per_epoch)))
        return WarmupCosineLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps), total_steps
    return None, 0

def to_device_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def make_head(name: str, cfg: dict):
    if name == "svi":
        d_model = int(cfg["model"]["encoder"]["d_model"])
        return SVIHead(d_model=d_model)
    return None


# -----------------------
# Log-moneyness helper
# -----------------------
def strikes_to_log_moneyness(sl: Dict, device: torch.device) -> Tuple[torch.Tensor, float]:
    """
    Returns (k, T_years). If spot is missing, center by median strike.
    """
    K = to_device_tensor(sl["strike"], device)
    dte = float(sl.get("dte", float("nan")))
    T = max(dte / 365.0, 1e-6) if (dte == dte and dte > 0) else 30.0 / 365.0

    S = sl.get("spot", float("nan"))
    r = float(sl.get("rate", 0.0))
    q = float(sl.get("div", 0.0))

    if not (S == S):  # NaN spot
        Kmed = torch.median(K)
        k = torch.log(torch.clamp(K / Kmed, min=1e-8))
        return k, T

    # forward: F = S * e^{(r - q) T}
    F = torch.as_tensor(S, dtype=K.dtype, device=K.device) * torch.exp(
        torch.as_tensor((r - q) * T, dtype=K.dtype, device=K.device)
    )
    k = torch.log(torch.clamp(K / F, min=1e-8))
    return k, T


# -----------------------
# Weighting & regularization helpers
# -----------------------
def vega_style_weights_from_k(k: torch.Tensor, k_scale: float = 1.0, p: float = 2.0) -> torch.Tensor:
    """
    Proxy for vega weighting that upweights near-the-money quotes and downweights wings:
    w(k) = 1 / (1 + (|k| / k_scale)^p)
    """
    return 1.0 / (1.0 + (torch.abs(k) / max(1e-6, k_scale)) ** p)

def param_l2_penalty(params_tuple, bounds: dict, weights: Dict[str, float]):
    """
    L2 penalty around the midpoints of the configured bounds; weights per-parameter.
    """
    a, b, rho, m, sigma = params_tuple
    mids = {
        "a":     0.5 * (bounds["a"][0] + bounds["a"][1]),
        "b":     0.5 * (bounds["b"][0] + bounds["b"][1]),
        "rho":   0.5 * (bounds["rho"][0] + bounds["rho"][1]),
        "m":     0.5 * (bounds["m"][0] + bounds["m"][1]),
        "sigma": 0.5 * (bounds["sigma"][0] + bounds["sigma"][1]),
    }
    loss = 0.0
    loss += float(weights.get("a", 0.0))     * torch.mean((a     - mids["a"])     ** 2)
    loss += float(weights.get("b", 0.0))     * torch.mean((b     - mids["b"])     ** 2)
    loss += float(weights.get("rho", 0.0))   * torch.mean((rho   - mids["rho"])   ** 2)
    loss += float(weights.get("m", 0.0))     * torch.mean((m     - mids["m"])     ** 2)
    loss += float(weights.get("sigma", 0.0)) * torch.mean((sigma - mids["sigma"]) ** 2)
    return loss


# -----------------------
# Core per-slice losses
# -----------------------
def compute_slice_losses(params_tuple, sl: Dict, device, vega_cfg: Dict | None):
    """
    params_tuple: (a, b, rho, m, sigma) — scalars for this slice
    sl: dict with strike/iv/dte (+ optional spot/rate/div)
    returns a dict with rmse, possibly weighted_rmse, and diagnostics
    """
    a, b, rho, m, sigma = params_tuple
    k, T_years = strikes_to_log_moneyness(sl, device)
    iv_true = to_device_tensor(sl["iv"], device)

    # svi_iv expects dte in days; we have years
    iv_hat = svi_iv_from_params(k, T_years * 365.0, a, b, rho, m, sigma)

    # Unweighted RMSE (for logging)
    rmse = torch.sqrt(torch.mean((iv_hat - iv_true) ** 2) + 1e-12)

    # Vega-style weighting on |k|
    if vega_cfg and bool(vega_cfg.get("enabled", True)):
        k_scale = float(vega_cfg.get("k_scale", 1.0))
        p = float(vega_cfg.get("power", 2.0))
        w = vega_style_weights_from_k(k, k_scale=k_scale, p=p).detach()
        # weighted MSE -> weighted RMSE
        mse_w = torch.sum(w * (iv_hat - iv_true) ** 2) / (torch.sum(w) + 1e-12)
        weighted_rmse = torch.sqrt(mse_w + 1e-12)
    else:
        weighted_rmse = rmse

    k_sorted = torch.sort(k).values
    w_sorted = svi_total_variance(k_sorted, a, b, rho, m, sigma)
    bfly_pen = butterfly_penalty(k_sorted, w_sorted)
    bfly_rate = butterfly_violation_rate(k_sorted, w_sorted)

    return {
        "rmse": rmse,
        "weighted_rmse": weighted_rmse,
        "bfly_pen": bfly_pen,
        "bfly_rate": bfly_rate,
        "k_sorted": k_sorted,
        "w_sorted": w_sorted,
        "T_years": T_years,
        "params": (a, b, rho, m, sigma),  # for calendar alignment
        "k_min": float(k_sorted.min().detach().cpu()),
        "k_max": float(k_sorted.max().detach().cpu()),
        "k_len": int(k_sorted.numel()),
    }

def calendar_pair_metrics_aligned(short_row: Dict, long_row: Dict, device: torch.device):
    """
    Recompute w on a common k-grid inside the overlap of the two maturities
    to avoid spurious calendar violations from mismatched strikes.
    """
    a1, b1, r1, m1, s1 = short_row["params"]
    a2, b2, r2, m2, s2 = long_row["params"]

    lo = max(short_row["k_min"], long_row["k_min"])
    hi = min(short_row["k_max"], long_row["k_max"])
    if not (hi > lo):
        z = torch.zeros((), device=device)
        return z, z

    n_align = max(8, min(short_row["k_len"], long_row["k_len"]))  # at least 8 points
    k_grid = torch.linspace(lo, hi, steps=n_align, device=device)

    w_short = svi_total_variance(k_grid, a1, b1, r1, m1, s1)
    w_long  = svi_total_variance(k_grid, a2, b2, r2, m2, s2)

    pen = torch.relu(w_short - w_long).mean()
    rate = (w_short > w_long).float().mean()
    return pen, rate


# -----------------------
# Param bound saturation (for logging)
# -----------------------
def bound_hit_rates(params_dict: Dict[str, torch.Tensor],
                    bounds: Dict[str, Tuple[float, float]] | None,
                    tol_frac: float = 0.005) -> Dict[str, float]:
    """
    Fractions within tol of lower/upper bound for each param.
    Safe if bounds is None or missing keys (we just skip those keys).
    """
    out: Dict[str, float] = {}
    if not bounds:
        return out
    for name in ("a", "b", "rho", "m", "sigma"):
        if name not in params_dict or name not in bounds:
            continue
        lo, hi = bounds[name]
        rng = max(1e-8, hi - lo)
        tol = tol_frac * rng
        x = params_dict[name]
        hit_lo = (x <= (lo + tol)).float().mean()
        hit_hi = (x >= (hi - tol)).float().mean()
        out[f"hit_{name}_lo"] = float(hit_lo.detach().cpu())
        out[f"hit_{name}_hi"] = float(hit_hi.detach().cpu())
    return out

# -----------------------
# Forward wrapper
# -----------------------
def forward_svi(model, xb):
    # ItoFormer expects [B, T, A, F]; add A=1
    xb = xb.unsqueeze(2)
    out = model(xb)
    return out["svi"] if isinstance(out, dict) and "svi" in out else out


# -----------------------
# Train / Eval epoch
# -----------------------
def train_or_eval_epoch(model, loader, optimizer, scheduler, cfg, epoch, split, device, paths, scaler=None, svi_bounds=None):
    is_train = (split == "train")
    model.train(is_train)
    sw = Stopwatch()

    # weights
    noarb = cfg["model"]["heads"]["svi"]["noarb"]
    w_pred = float(cfg["train"].get("weights", {}).get("pred", 1.0))
    w_bfly = float(noarb.get("bfly_weight", 1.0))
    w_cal  = float(noarb.get("cal_weight", 1.0))

    # vega-style weighting config
    vega_cfg = cfg["train"].get("vega_weighting", {"enabled": True, "k_scale": 1.0, "power": 2.0})

    # parameter L2 reg weights
    l2_cfg = cfg["train"].get("param_l2", {"a":0.0,"b":0.0,"rho":0.0,"m":0.0,"sigma":0.0})
    use_amp = bool(cfg["train"].get("amp", {}).get("enabled", False)) and (device.type == "cuda")

    total_rmse = total_wr = total_bfly = total_cal = 0.0
    total_l2 = 0.0
    total_loss = 0.0
    batches = 0
    last_grad = ""
    # bound hits (average across steps)
    agg_hits = defaultdict(float)

    for xb, yb, slices in loader:
        batches += 1
        step_sw = Stopwatch()
        xb = xb.to(device=device, dtype=torch.float32)

        with autocast(device_type="cuda", enabled=use_amp):
            params = forward_svi(model, xb)  # dict of [B] tensors

            # track bounds hit rates
            hits = bound_hit_rates(params, svi_bounds or {}, tol_frac=float(cfg["train"].get("bound_tol_frac", 0.005)))
            for k, v in hits.items():
                agg_hits[k] += v

            batch_wr, batch_rmse, batch_bfly = [], [], []
            by_date = defaultdict(list)

            for i, sl in enumerate(slices):
                per = compute_slice_losses(
                    (params["a"][i], params["b"][i], params["rho"][i], params["m"][i], params["sigma"][i]),
                    sl, device, vega_cfg
                )
                batch_rmse.append(per["rmse"])
                batch_wr.append(per["weighted_rmse"])
                batch_bfly.append(per["bfly_pen"])
                by_date[str(sl.get("date", ""))].append(per)

            # calendar penalties with aligned k-grid
            batch_cal = []
            for _date, rows in by_date.items():
                rows = sorted(rows, key=lambda r: r["T_years"])
                for j in range(len(rows) - 1):
                    pen, _rate = calendar_pair_metrics_aligned(rows[j], rows[j+1], device)
                    batch_cal.append(pen)

            rmse_mean = torch.stack(batch_rmse).mean() if batch_rmse else torch.zeros((), device=device)
            wr_mean   = torch.stack(batch_wr).mean()   if batch_wr   else rmse_mean
            bfly_mean = torch.stack(batch_bfly).mean() if batch_bfly else torch.zeros((), device=device)
            cal_mean  = torch.stack(batch_cal).mean()  if batch_cal  else torch.zeros((), device=device)

            # param L2
            l2 = param_l2_penalty(
                (params["a"], params["b"], params["rho"], params["m"], params["sigma"]),
                svi_bounds or {}, l2_cfg
            )

            loss = (w_pred * wr_mean) + (w_bfly * bfly_mean) + (w_cal * cal_mean) + l2

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
                optimizer.step()

            # Per-step scheduler (AFTER optimizer.step())
            if scheduler is not None:
                scheduler.step()

            last_grad = f"{grad_norm(model.parameters()):.6f}"
            if batches % 20 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch} | {split}] Batch {batches} | Loss={float(loss):.4f} | LR={lr_now:.6g}", flush=True)

        total_rmse += float(rmse_mean.detach().cpu())
        total_wr   += float(wr_mean.detach().cpu())
        total_bfly += float(bfly_mean.detach().cpu())
        total_cal  += float(cal_mean.detach().cpu())
        total_l2   += float(l2.detach().cpu())
        total_loss += float(loss.detach().cpu())

    # average bound hits
    mean_hits = {k: (v / max(1, batches)) for k, v in agg_hits.items()}

    fixed_hit_cols = {f"hit_{p}_{side}": 0.0
                      for p in ("a","b","rho","m","sigma")
                      for side in ("lo","hi")}
    fixed_hit_cols.update(mean_hits)

    row = {
        "epoch": epoch, "split": split,
        "loss_total": total_loss / max(1, batches),
        "loss_pred_wr": total_wr / max(1, batches),
        "loss_pred_rmse": total_rmse / max(1, batches),
        "loss_noarb_bfly": total_bfly / max(1, batches),
        "loss_noarb_cal":  total_cal / max(1, batches),
        "loss_param_l2": total_l2 / max(1, batches),
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": last_grad,
        "batches": batches,
        "wall_time_s": sw.seconds(),
    }

    row.update(fixed_hit_cols)
    row = {k: _csv_clean(v) for k, v in row.items()}

    header = list(row.keys())
    append_row(paths["train_csv"], header=header, row=row)

    return row["loss_total"]  # for checkpoint selection


# -----------------------
# Eval table writers
# -----------------------
def write_eval_tables(model, val_loader, cfg, epoch, device, paths):
    model.eval()
    all_by_date = defaultdict(list)

    with torch.no_grad():
        for xb, yb, slices in val_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            params = forward_svi(model, xb)

            for i, sl in enumerate(slices):
                a = params["a"][i]; b = params["b"][i]; rho = params["rho"][i]; m = params["m"][i]; sigma = params["sigma"][i]

                # --- same preprocessing as train ---
                k, T_years = strikes_to_log_moneyness(sl, device)
                iv_true = to_device_tensor(sl["iv"], device)

                iv_hat = svi_iv_from_params(k, T_years * 365.0, a, b, rho, m, sigma)
                rmse = torch.sqrt(torch.mean((iv_hat - iv_true) ** 2) + 1e-12).item()

                k_sorted = torch.sort(k).values
                w_sorted = svi_total_variance(k_sorted, a, b, rho, m, sigma)
                bfly_rate = butterfly_violation_rate(k_sorted, w_sorted).item()

                # --- write params table row ---
                date_str = str(sl.get("date", ""))[:10]
                exp_str  = str(sl.get("expiry", ""))[:10]
                append_row(
                    paths["svi_params_csv"],
                    header=["epoch","date","expiry","dte","a_hat","b_hat","rho_hat","m_hat","sigma_hat","surf_iv_rmse"],
                    row={"epoch": epoch, "date": date_str, "expiry": exp_str, "dte": float(T_years * 365.0),
                         "a_hat": a.item(), "b_hat": b.item(), "rho_hat": rho.item(), "m_hat": m.item(), "sigma_hat": sigma.item(),
                         "surf_iv_rmse": rmse},
                )

                # --- stash for calendar alignment ---
                all_by_date[date_str].append({
                    "expiry": exp_str,
                    "T_years": float(T_years),
                    "w_sorted": w_sorted.detach(),
                    "k_sorted": k_sorted.detach(),
                    "k_min": float(k_sorted.min().detach().cpu()),
                    "k_max": float(k_sorted.max().detach().cpu()),
                    "k_len": int(k_sorted.numel()),
                    "bfly_rate": float(bfly_rate),
                    "bfly_pen_mean": float(butterfly_penalty(k_sorted, w_sorted).detach().cpu()),
                    "params": (a.detach(), b.detach(), rho.detach(), m.detach(), sigma.detach()),
                })

        # --- calendar diagnostics on aligned k-grid ---
        for date_str, rows in all_by_date.items():
            rows = sorted(rows, key=lambda r: r["T_years"])
            for j in range(len(rows)):
                cal_rate = ""
                cal_pen  = ""
                if j < len(rows) - 1:
                    pen, rate = calendar_pair_metrics_aligned(rows[j], rows[j+1], device)
                    cal_rate = float(rate.detach().cpu())
                    cal_pen  = float(pen.detach().cpu())

                append_row(
                    paths["noarb_csv"],
                    header=["epoch","date","expiry","dte","bfly_violation_rate","calendar_violation_rate","bfly_penalty_mean","calendar_penalty_mean"],
                    row={"epoch": epoch, "date": date_str, "expiry": rows[j]["expiry"], "dte": rows[j]["T_years"] * 365.0,
                         "bfly_violation_rate": rows[j]["bfly_rate"],
                         "calendar_violation_rate": cal_rate,
                         "bfly_penalty_mean": rows[j]["bfly_pen_mean"],
                         "calendar_penalty_mean": cal_pen},
                )


# -----------------------
# Run meta (optional)
# -----------------------
def write_run_meta(cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    meta = {
        "experiment_name": cfg.get("experiment_name", "options_svi_spx"),
        "config_path": "",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": cfg.get("device", "cuda"),
        "dtype": cfg.get("dtype", "fp16"),
        "seeds": cfg.get("seeds", {}),
        "results_index": {
            "train_csv": f"{outdir}/logs/train.csv",
            "svi_params_csv": f"{outdir}/eval/svi_params.csv",
            "noarb_csv": f"{outdir}/diagnostics/no_arbitrage_svi.csv",
        },
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# -----------------------
# Checkpoint helpers
# -----------------------
def save_ckpt(path, model, optimizer, epoch, cfg, extra=None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "extra": extra or {},
    }
    torch.save(ckpt, path)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    # knobs (optional CLI overrides)
    ap.add_argument("--ito_weight", type=float, default=None)        # reserved
    ap.add_argument("--martingale_weight", type=float, default=None) # reserved
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Reproducibility
    seed = int(args.seed if args.seed is not None else cfg.get("seeds", {}).get("train", 1337))
    set_seed(seed)

    # CLI overrides
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.ito_weight is not None:
        cfg["model"]["heads"]["svi"]["ito_consistency_weight"] = float(args.ito_weight)
    if args.martingale_weight is not None:
        cfg["model"]["heads"]["svi"]["martingale_weight"] = float(args.martingale_weight)

    # Device / AMP
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda") else "cpu")
    use_amp = bool(cfg["train"].get("amp", {}).get("enabled", False)) and (device.type == "cuda")
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # Data
    dm = OptionsSVIDataModule(cfg); dm.setup()
    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()

    # Align model dims
    sample_x, _, _ = dm.train_ds[0]
    cfg["model"]["in_features"] = int(cfg["model"].get("in_features", sample_x.shape[-1]))
    cfg["model"]["n_assets"]    = int(cfg["model"].get("n_assets", 1))

    # Model + SVI head
    model = ItoFormer.build_from_config(cfg, lambda name: make_head(name, cfg)).to(device)
    # Get SVI bounds for reg/logging
    svi_head = None
    heads_attr = getattr(model, "heads", None)
    if heads_attr is not None: 
        try:
            if "svi" in heads_attr:
                svi_head = heads_attr["svi"]
        except Exception:
            svi_head = None

    svi_bounds = None
    if svi_head is not None and hasattr(svi_head, "bounds"):
        svi_bounds = getattr(svi_head, "bounds", None)

    if not isinstance(svi_bounds, dict) or not all(k in svi_bounds for k in ("a","b","rho","m","sigma")):
        print("[WARN] SVI bounds not found or incomplete; bound-hit telemetry will be empty.")
    else:
        print("[INFO] SVI bounds:", {k: tuple(map(float, v)) for k, v in svi_bounds.items()})

    # Optimizer / Scheduler
    optimizer = make_optimizer(cfg, model.parameters())
    scheduler, total_steps = make_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    # Paths
    outdir = cfg["artifacts"]["outdir"]
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "eval"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "diagnostics"), exist_ok=True)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    paths = {
        "train_csv": cfg["logging"]["csv_train"],
        "svi_params_csv": cfg["eval"]["outputs"]["svi_params_csv"],
        "noarb_csv": cfg["eval"]["outputs"]["noarb_csv"],
    }

    write_run_meta(cfg, outdir)

    # Train
    E = int(cfg["train"]["epochs"])
    best_val = float("inf")
    for epoch in range(1, E + 1):
        print(f"\n===== [EPOCH {epoch}/{E}] =====", flush=True)
        tr_loss = train_or_eval_epoch(model, train_loader, optimizer, scheduler, cfg, epoch, "train", device, paths, scaler, svi_bounds)
        val_loss = train_or_eval_epoch(model, val_loader,   optimizer, scheduler, cfg, epoch, "val",   device, paths, None, svi_bounds)

        # Checkpoints
        save_ckpt(os.path.join(ckpt_dir, "last.pt"), model, optimizer, epoch, cfg,
                  extra={"train_loss": tr_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, optimizer, epoch, cfg,
                      extra={"train_loss": tr_loss, "val_loss": val_loss})
            print(f"[CKPT] New best @ epoch {epoch}: val_loss={val_loss:.6f}", flush=True)

        # Eval tables (params + no-arb diagnostics)
        write_eval_tables(model, val_loader, cfg, epoch, device, paths)

if __name__ == "__main__":
    main()
