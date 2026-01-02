# scripts/train_options_nodes.py
from __future__ import annotations
import argparse, os, sys
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# --- AMP (new + old API compatible) ---
try:
    from torch import amp
    autocast = amp.autocast
    GradScaler = amp.GradScaler
except Exception:  # older PyTorch fallback
    from torch.cuda.amp import autocast, GradScaler  # type: ignore

# Project imports (assumes PYTHONPATH points to ./src)
from itoformer.utils.csvlog import append_row, Stopwatch
from itoformer.utils.config import load_config
from itoformer.training.datamodules_options import OptionsNodesDataModule
from itoformer.models.itoformer import ItoFormer
from itoformer.models.heads.options_node import OptionsNodeHead

# --- Structural losses (no-arb, Itô, martingale) ---
# Guarded imports so training won’t crash if a file is absent.
try:
    from itoformer.losses.no_arbitrage import (
        butterfly_convexity_loss,    # expected signature: (strike, iv, dte, cp) -> penalty
        calendar_monotonicity_loss,  # expected signature: (strike, iv, dte, cp) -> penalty
    )
except Exception:
    def butterfly_convexity_loss(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return torch.tensor(0.0, device=device)
    def calendar_monotonicity_loss(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return torch.tensor(0.0, device=device)

try:
    from itoformer.losses.ito import ito_consistency_loss  # signature discovered during run
except Exception:
    def ito_consistency_loss(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return torch.tensor(0.0, device=device)

try:
    from itoformer.losses.martingale import martingale_drift_loss  # flexible signature
except Exception:
    def martingale_drift_loss(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return torch.tensor(0.0, device=device)


# ---------------- heads ----------------
def make_head(name: str, cfg: dict):
    """Construct enabled heads from YAML."""
    if name != "options_node":
        return None
    targets = cfg["data"]["targets"]["predict"]
    out_dims = {t: 1 for t in targets}
    d_model = int(cfg["model"]["encoder"]["d_model"])
    return OptionsNodeHead(d_model, out_dims)


def _extract_preds(out, head_key: str = "options_node"):
    """
    Handle both cases:
      - multiple heads: out is dict with key 'options_node'
      - single head: out is directly the dict of targets {iv, delta, ...}
    """
    if isinstance(out, dict) and head_key in out:
        return out[head_key]
    return out  # already the dict of targets


# ---------- utilities to get batch meta safely ----------
_WARNED_META_MISSING = False

def _maybe_warn_once(msg: str):
    global _WARNED_META_MISSING
    if not _WARNED_META_MISSING:
        print(f"[WARN] {msg}")
        _WARNED_META_MISSING = True


def _last_step(vec: torch.Tensor) -> torch.Tensor:
    """Normalize pred shape to [B]: accept [B], [B,T], or [B,T,1] and take last time step."""
    if vec.dim() == 3:       # [B, T, 1]
        return vec[:, -1, :].squeeze(-1)
    elif vec.dim() == 2:     # [B, T]
        return vec[:, -1]
    return vec.squeeze()     # [B] or scalar


def _extract_meta_from_batch(
    dm: OptionsNodesDataModule,
    xb,
    extra: Optional[object],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Try to get (strike, dte, cp) as float tensors on the same device as xb.
    """
    device = xb.device

    # 1) extra dict path
    if isinstance(extra, dict):
        s = extra.get("strike", None)
        t = extra.get("dte", None)
        c = extra.get("cp", None)
        if s is not None and t is not None and c is not None:
            return torch.as_tensor(s, device=device, dtype=torch.float32), \
                   torch.as_tensor(t, device=device, dtype=torch.float32), \
                   torch.as_tensor(c, device=device, dtype=torch.float32)

    # 2) object with attributes or mapping-like
    for k in ("strike", "dte", "cp"):
        if hasattr(extra, k):
            pass
        else:
            break
    else:
        try:
            return torch.as_tensor(getattr(extra, "strike"), device=device, dtype=torch.float32), \
                   torch.as_tensor(getattr(extra, "dte"), device=device, dtype=torch.float32), \
                   torch.as_tensor(getattr(extra, "cp"), device=device, dtype=torch.float32)
        except Exception:
            pass

    # 3) feature_names path (xb: [B, T, F] or [B, F])
    feat_names = getattr(dm, "feature_names", None)
    if feat_names is not None:
        try:
            def find_idx(name):
                try:
                    return feat_names.index(name)
                except ValueError:
                    lower = [f.lower() for f in feat_names]
                    aliases = {
                        "cp": ["cp", "call_put", "callput"],
                        "strike": ["strike", "k"],
                        "dte": ["dte", "days_to_expiry", "days_to_maturity", "ttm"],
                    }
                    for cand in aliases[name]:
                        if cand in lower:
                            return lower.index(cand)
                    return None

            i_strike = find_idx("strike")
            i_dte    = find_idx("dte")
            i_cp     = find_idx("cp")

            if None not in (i_strike, i_dte, i_cp):
                if xb.dim() == 3:
                    xlast = xb[:, -1, :]  # [B, F]
                else:
                    xlast = xb            # [B, F]
                strike = xlast[:, i_strike]
                dte    = xlast[:, i_dte]
                cp     = xlast[:, i_cp]
                return strike, dte, cp
        except Exception:
            pass

    return None, None, None


def _compute_structural_losses_with_meta(
    preds: Dict[str, torch.Tensor],
    xb: torch.Tensor,
    meta_extra: Optional[object],
    dm: OptionsNodesDataModule,
    device: str
) -> Dict[str, torch.Tensor]:
    import numpy as np
    import inspect

    out = {
        "loss_noarb_bfly": torch.tensor(0.0, device=device),
        "loss_noarb_cal":  torch.tensor(0.0, device=device),
        "loss_ito":        torch.tensor(0.0, device=device),
        "loss_martingale": torch.tensor(0.0, device=device),
    }

    def _ensure_tensor(val):
        if isinstance(val, torch.Tensor):
            return val.to(device=device, dtype=torch.float32)
        try:
            return torch.tensor(float(val), device=device)
        except Exception:
            return torch.tensor(0.0, device=device)

    # ---------- meta + preds ----------
    strike, dte, cp = _extract_meta_from_batch(dm, xb, meta_extra)

    iv_seq = preds.get("iv", None)
    if isinstance(iv_seq, torch.Tensor):
        iv_seq = iv_seq.to(device=device, dtype=torch.float32)
        if iv_seq.dim() == 3 and iv_seq.size(-1) == 1:
            iv_seq = iv_seq.squeeze(-1)  # [B,T,1] -> [B,T]
    else:
        iv_seq = None

    # ---------- no-arb caller ----------
    def _call_noarb(func, strike_t, iv_t, dte_t, cp_t):
        s = strike_t.to(device=device, dtype=torch.float32).view(-1)
        v = iv_t.to(device=device, dtype=torch.float32).view(-1)
        t = dte_t.to(device=device, dtype=torch.float32).view(-1)
        c = cp_t.to(device=device, dtype=torch.float32).view(-1)
        try:
            return func(s, v, t, c)
        except Exception:
            pass
        try:
            return func(strike=s, iv=v, dte=t, cp=c)
        except Exception:
            pass
        X = torch.column_stack((s, v, t, c))
        try:
            return func(X)
        except Exception:
            pass
        Xnp = X.detach().cpu().numpy()
        try:
            return func(Xnp)
        except Exception:
            pass
        payload_np = {"strike": Xnp[:, 0], "iv": Xnp[:, 1], "dte": Xnp[:, 2], "cp": Xnp[:, 3]}
        try:
            return func(payload_np)
        except Exception as e:
            _maybe_warn_once(f"No-arbitrage penalties skipped (adaptive call failed: {e}). Setting to 0.")
            return torch.tensor(0.0, device=device)

    # ---------- no-arb penalties ----------
    iv_last = (iv_seq[:, -1] if isinstance(iv_seq, torch.Tensor) else None)
    if (strike is not None) and (dte is not None) and (cp is not None) and (iv_last is not None):
        try:
            val_b = _call_noarb(butterfly_convexity_loss, strike, iv_last, dte, cp)
            val_c = _call_noarb(calendar_monotonicity_loss, strike, iv_last, dte, cp)
            val_b = torch.as_tensor(val_b, device=device, dtype=torch.float32)
            val_c = torch.as_tensor(val_c, device=device, dtype=torch.float32)
            out["loss_noarb_bfly"] = torch.nan_to_num(val_b, nan=0.0, posinf=0.0, neginf=0.0).mean()
            out["loss_noarb_cal"]  = torch.nan_to_num(val_c, nan=0.0, posinf=0.0, neginf=0.0).mean()
        except Exception as e:
            _maybe_warn_once(f"No-arbitrage penalties skipped (reason: {e}). Setting to 0.")
    else:
        _maybe_warn_once("Batch meta (strike/dte/cp) not found; no-arbitrage losses set to 0 for this run.")

    # ---------- helpers ----------
    def _find_spot_index(dm_obj) -> int:
        names = getattr(dm_obj, "feature_cols", None) or getattr(dm_obj, "x_cols", None)
        if isinstance(names, (list, tuple)):
            low = [str(n).lower() for n in names]
            for key in ("spx", "spot", "underlying", "price", "s"):
                for i, nm in enumerate(low):
                    if key in nm:
                        return i
        return 0

    # ---------- Itô terms (per-timestep) ----------
    try:
        X = xb
        if X.dim() == 2:
            X = X.unsqueeze(1)
        X = X.to(device=device, dtype=torch.float32)

        idx = _find_spot_index(dm)
        s = X[..., idx]
        B, T = s.shape

        f_raw = iv_seq if isinstance(iv_seq, torch.Tensor) else None
        if f_raw is None or f_raw.dim() == 0:
            f = torch.zeros(B, T, device=device, dtype=torch.float32)
            _maybe_warn_once("Itô: iv_seq not found; using zeros.")
        else:
            if f_raw.dim() == 1:
                f = f_raw.view(B, 1).expand(B, T).contiguous()
            elif f_raw.dim() == 2:
                Tp = f_raw.size(1)
                if Tp == T:
                    f = f_raw
                elif Tp == 1:
                    f = f_raw.expand(B, T).contiguous()
                    _maybe_warn_once(f"Itô: iv_seq has T=1; tiling to T={T}.")
                else:
                    f = f_raw[:, -1:].expand(B, T).contiguous()
                    _maybe_warn_once(f"Itô: iv_seq T={Tp} != spot T={T}; tiling last step.")
            else:
                f_tmp = f_raw.squeeze(-1) if f_raw.size(-1) == 1 else f_raw
                if f_tmp.dim() == 2 and f_tmp.size(1) == T:
                    f = f_tmp
                elif f_tmp.dim() == 2 and f_tmp.size(1) == 1:
                    f = f_tmp.expand(B, T).contiguous()
                    _maybe_warn_once(f"Itô: iv_seq squeezed to T=1; tiling to T={T}.")
                else:
                    f = torch.zeros(B, T, device=device, dtype=torch.float32)
                    _maybe_warn_once("Itô: unexpected iv_seq shape; using zeros.")

        # _____________central differences for smoother f_x; forward/backward at edges____
                # --- central differences for smoother f_x; forward/backward at edges ---
        eps = 1e-6

        # Optional normalization: compute on log-price for scale consistency
        s_log  = torch.log(s.clamp_min(1e-6))
        # Choose which coordinate to use for derivatives:
        #s_used = s        # raw price (original behavior)
        s_used = s_log      # log-price (recommended)

        # f_x: central where possible, forward/backward at edges
        f_x = torch.zeros_like(f)
        if T >= 3:
            # central difference
            num = (f[:, 2:] - f[:, :-2])
            den = (s_used[:, 2:] - s_used[:, :-2]).abs() + eps
            f_x[:, 1:-1] = num / den
            # edges
            f_x[:, 0]  = (f[:, 1]  - f[:, 0])  / ((s_used[:, 1]  - s_used[:, 0]).abs()  + eps)
            f_x[:, -1] = (f[:, -1] - f[:, -2]) / ((s_used[:, -1] - s_used[:, -2]).abs() + eps)
        elif T == 2:
            f_x[:, 1] = (f[:, 1] - f[:, 0]) / ((s_used[:, 1] - s_used[:, 0]).abs() + eps)
        # else T == 1 -> f_x stays zeros

        # f_xx: use change in f_x over a two-step span
        f_xx = torch.zeros_like(f)
        if T >= 3:
            s_span2 = (s_used[:, 2:] - s_used[:, :-2]).abs() + eps
            f_xx[:, 2:] = 2.0 * (f_x[:, 2:] - f_x[:, 1:-1]) / s_span2

        # ΔS_used and (ΔS_used)^2 (match coordinate of s_used for internal consistency)
        delta_x  = torch.zeros_like(f)
        delta_qv = torch.zeros_like(f)
        if T >= 2:
            dx = s_used[:, 1:] - s_used[:, :-1]
            delta_x[:, 1:]  = dx
            delta_qv[:, 1:] = dx * dx

        for tns in (f_x, f_xx, delta_x, delta_qv):
            tns.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        # Full-path mask, skipping the first two indices where f_xx/ΔS_used are undefined/unstable
        mask = torch.zeros(B, T, device=device, dtype=torch.float32)
        if T >= 3:
            mask[:, 2:] = 1.0
        elif T == 2:
            mask[:, 1:] = 1.0

        out["loss_ito"] = ito_consistency_loss(
            f, f_x, f_xx, delta_x, delta_qv,
            mask=mask, reduction="mean", weight=1.0
        )

    except Exception as e:
        _maybe_warn_once(f"Itô consistency loss skipped (reason: {e}). Setting to 0.")

    # ---------- Martingale drift ----------
    try:
        X = xb
        if X.dim() == 2:
            X = X.unsqueeze(1)
        X = X.to(device=device, dtype=torch.float32)

        idx = _find_spot_index(dm)
        s_path = X[..., idx]
        B, T = s_path.shape

        if T >= 2:
            dS = s_path[:, 1:] - s_path[:, :-1]
            ret = dS / s_path[:, :-1].clamp_min(1e-6)
            ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            ret = torch.zeros(B, 0, device=device, dtype=torch.float32)

        ret_full = torch.zeros(B, T, device=device, dtype=torch.float32)
        if T >= 2:
            ret_full[:, 1:] = ret

        mask = torch.zeros(B, T, device=device, dtype=torch.float32)
        if T >= 2:
            mask[:, 1:] = 1.0

        try:
            val_mart = martingale_drift_loss(ret_full, mask=mask)
        except TypeError:
            val_mart = martingale_drift_loss(ret_full)

        out["loss_martingale"] = torch.nan_to_num(
            torch.as_tensor(val_mart, device=device, dtype=torch.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        ).mean()

    except Exception as e:
        _maybe_warn_once(f"Martingale loss skipped (reason: {e}). Setting to 0.")
        out["loss_martingale"] = torch.tensor(0.0, device=device)

    return out

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs from YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    # optional CUDA speed nudge on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------- Data ----------------
    dm = OptionsNodesDataModule(cfg)
    dm.setup()

    # Infer dims from data and cfg
    in_features = int(cfg["model"].get("in_features", dm.train_ds[0][0].shape[-1]))
    n_assets    = int(cfg["model"].get("n_assets", 1))
    d_model     = int(cfg["model"]["encoder"]["d_model"])

    device = "cuda" if (cfg.get("device", "cpu") == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[DEBUG] Inferred in_features={in_features}, n_targets={len(cfg['data']['targets']['predict'])}, d_model={d_model}, device={device}")

    # Ensure model config is consistent with datamodule output
    cfg["model"]["in_features"] = in_features
    cfg["model"]["n_assets"]    = n_assets

    # ---------------- Model ----------------
    model = ItoFormer.build_from_config(cfg, lambda name: make_head(name, cfg)).to(device)

    # ---------------- Optim/AMP ----------------
    lr  = float(cfg["train"]["optimizer"]["lr"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    use_amp   = bool(cfg["train"]["amp"]["enabled"]) and (device == "cuda")
    precision = str(cfg["train"]["amp"].get("precision", "fp16")).lower()
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler    = GradScaler("cuda" if device == "cuda" else "cpu", enabled=use_amp)

    # Targets / bookkeeping
    target_names = list(cfg["data"]["targets"]["predict"])
    name_to_idx  = {n: i for i, n in enumerate(target_names)}
    epochs       = int(cfg["train"].get("epochs", 1))

    # ---------------- CSV paths ----------------
    csv_enabled = bool(cfg.get("logging", {}).get("csv", False))
    csv_dir     = os.path.normpath(
        cfg.get("logging", {}).get("csv_dir",
            os.path.join(cfg["paths"]["outputs_root"], cfg["experiment_name"], "logs"))
    )
    train_csv = os.path.join(csv_dir, "train.csv")
    val_csv   = os.path.join(csv_dir, "val_summary.csv")

    train_header = [
        "epoch", "split",
        "loss_total",
        "loss_pred_mse",
        "loss_noarb_bfly",
        "loss_noarb_cal",
        "loss_ito",
        "loss_martingale",
        "mse", "mae", "rmse",
        "lr", "grad_norm",
        "wall_time_s",
    ]

    timer = Stopwatch()

    # - - - - read structural loss weights from YAML
    def _get(cfg, *keys, default=None):
        cur = cfg
        for k in keys:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return cur if cur != {} else (default if default is not None else {})

    na_cfg   = _get(cfg, "losses", "no_arbitrage", default={})
    ito_cfg  = _get(cfg, "losses", "ito_consistency", default={})
    mart_cfg = _get(cfg, "losses", "martingale", default={})

    BF_W = float(na_cfg.get("butterfly_weight", 0.0)) if na_cfg.get("enabled", False) else 0.0
    CAL_W= float(na_cfg.get("calendar_weight",  0.0)) if na_cfg.get("enabled", False) else 0.0
    ITO_W= float(ito_cfg.get("weight",          0.0)) if ito_cfg.get("enabled", False) else 0.0
    MAR_W= float(mart_cfg.get("weight",         0.0)) if mart_cfg.get("enabled", False) else 0.0

    # ---------------- Train ----------------
    model.train()
    for ep in range(1, epochs + 1):
        running_mse  = 0.0
        running_mae  = 0.0
        running_bfly = 0.0
        running_cal  = 0.0
        running_ito  = 0.0
        running_mart = 0.0
        grad_norm_ep = 0.0
        n_batches    = 0

        for batch in dm.train_loader():
            # Accept (xb, yb) or (xb, yb, meta)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
                extra  = batch[2] if len(batch) >= 3 else None
            else:
                xb, yb, extra = batch, None, None  # unlikely, but safe

            # xb: [B, T, F] → ItoFormer expects [B, T, A, F] with A=1
            xb = xb.to(device).unsqueeze(2)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out   = model(xb)
                preds = _extract_preds(out, "options_node")

                # - - debug shape of IV output once - - -
                if not hasattr(_extract_preds, "_shapes_logged"):
                    print("[DEBUG] options_node.iv shape:", preds["iv"].shape)
                    _extract_preds._shapes_logged = True

                # --- prediction loss ---
                loss = 0.0
                batch_mse = 0.0
                batch_mae = 0.0
                for name in target_names:
                    p = preds[name]
                    p = _last_step(p)  # [B]
                    y = yb[:, name_to_idx[name]]
                    mse_i = (p - y).pow(2).mean()
                    mae_i = (p - y).abs().mean()
                    loss   = loss + mse_i
                    batch_mse += float(mse_i.item())
                    batch_mae += float(mae_i.item())

                n_t = max(1, len(target_names))
                loss      = loss / n_t
                batch_mse = batch_mse / n_t
                batch_mae = batch_mae / n_t

                # --- structural losses (best-effort) ---
                struct = _compute_structural_losses_with_meta(
                    preds=preds, xb=xb.squeeze(2), meta_extra=extra, dm=dm, device=device
                )
                # Combine using YAML weights (NaN Safe)
                if any(w > 0.0 for w in (BF_W, CAL_W, ITO_W, MAR_W)):
                    bfly = torch.nan_to_num(struct["loss_noarb_bfly"], nan=0.0, posinf=0.0, neginf=0.0)
                    cal  = torch.nan_to_num(struct["loss_noarb_cal"],  nan=0.0, posinf=0.0, neginf=0.0)
                    ito  = torch.nan_to_num(struct["loss_ito"],        nan=0.0, posinf=0.0, neginf=0.0)
                    mart = torch.nan_to_num(struct["loss_martingale"], nan=0.0, posinf=0.0, neginf=0.0)

                    ito_scale = min(1.0, ep / 10.0)
                    loss = loss + BF_W*bfly + CAL_W*cal + ITO_W*ito_scale*ito + MAR_W*mart
                    #loss = loss + BF_W * bfly + CAL_W * cal + ITO_W * ito + MAR_W * mart

            # AMP-safe grad norm + step
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
                opt.step()

            running_mse  += batch_mse
            running_mae  += batch_mae
            running_bfly += float(struct["loss_noarb_bfly"].detach().item())
            running_cal  += float(struct["loss_noarb_cal"].detach().item())
            running_ito  += float(struct["loss_ito"].detach().item())
            running_mart += float(struct["loss_martingale"].detach().item())
            grad_norm_ep += float(total_norm.detach().item()) if hasattr(total_norm, "item") else float(total_norm)
            n_batches    += 1

        # epoch aggregates
        denom     = max(1, n_batches)
        train_mse = running_mse / denom
        train_mae = running_mae / denom
        train_rmse= train_mse ** 0.5
        avg_bfly  = running_bfly / denom
        avg_cal   = running_cal  / denom
        avg_ito   = running_ito  / denom
        avg_mart  = running_mart / denom
        avg_grad  = grad_norm_ep / denom
        curr_lr   = next(iter(opt.param_groups))["lr"]
        wall_s    = timer.seconds()

        print(f"[Epoch {ep}/{epochs}] train_mse={train_mse:.6f}")

        if csv_enabled:
            append_row(
                train_csv,
                train_header,
                dict(
                    epoch=ep, split="train",
                    loss_total=train_mse,          # equals pred MSE unless you combine losses above
                    loss_pred_mse=train_mse,
                    loss_noarb_bfly=avg_bfly,
                    loss_noarb_cal=avg_cal,
                    loss_ito=avg_ito,
                    loss_martingale=avg_mart,
                    mse=train_mse, mae=train_mae, rmse=train_rmse,
                    lr=curr_lr, grad_norm=avg_grad,
                    wall_time_s=wall_s,
                ),
            )

    # ---------------- Validation (aggregate) ----------------
    model.eval()
    with torch.no_grad():
        val_mse_acc, val_mae_acc, val_n = 0.0, 0.0, 0
        for batch in dm.val_loader():
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                xb, yb = batch, None  # safety
            xb = xb.to(device).unsqueeze(2)
            yb = yb.to(device)
            out   = model(xb)
            preds = _extract_preds(out, "options_node")

            loss_mse = 0.0
            loss_mae = 0.0
            for name in target_names:
                p = _last_step(preds[name])
                y = yb[:, name_to_idx[name]]
                loss_mse += (p - y).pow(2).mean()
                loss_mae += (p - y).abs().mean()

            n_t = max(1, len(target_names))
            val_mse_acc += float((loss_mse / n_t).item())
            val_mae_acc += float((loss_mae / n_t).item())
            val_n += 1

        if val_n > 0:
            val_mse  = val_mse_acc / val_n
            val_mae  = val_mae_acc / val_n
            val_rmse = val_mse ** 0.5
            print(f"[VAL] mse={val_mse:.6f}")

            if csv_enabled:
                append_row(
                    val_csv,
                    ["epoch","mse","mae","rmse","n_batches"],
                    dict(epoch=epochs, mse=val_mse, mae=val_mae, rmse=val_rmse, n_batches=val_n),
                )

    print("OK: options_nodes training ran.")


if __name__ == "__main__":
    # Ensure local src is visible if someone forgets PYTHONPATH
    if "PYTHONPATH" not in os.environ:
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    main()