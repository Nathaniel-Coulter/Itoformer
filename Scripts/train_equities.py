# scripts/train_equities.py
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import amp
import torch.nn as nn
from torch.utils.data import DataLoader

from itoformer.models.itoformer import ItoFormer, ItoConfig
from itoformer.models.heads.forecast import ForecastHead
from itoformer.losses.martingale import martingale_drift_loss
from itoformer.models.heads.martingale import MartingaleHead
from itoformer.encoders.encoders import (
    EncPatchTST, EnciTransformer, CrossLite, EncPointwise
)


# You wrote these already:
from itoformer.training.datasets import build_equities_dataloaders  # -> (train_loader, val_loader, meta)
# Optional: if you have a separate bias builder, import it; else None
# from itoformer.training.tokenization import build_attention_biases  # example placeholder

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_mart: float = 0.10,
    lambda_ito: float = 0.05,
) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        if len(batch) == 3:
            x, y, bias = batch
            bias = bias.to(device)
        else:
            x, y = batch
            bias = None
        x = x.to(device)
        y = y.to(device)

        pred = model(x, bias=bias)
        if pred.ndim == 3 and y.ndim == 3 and pred.size(1) == 1 and y.size(1) > 1:
            y = y[:, -1:, :]
        l_pred = mse_loss(pred, y)
        l_mart, l_ito = compute_aux_losses(model, x, pred, bias)
        loss = l_pred + lambda_mart * l_mart + lambda_ito * l_ito

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

def compute_aux_losses(
    model: nn.Module,
    x: torch.Tensor,
    pred: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (L_mart, L_ito). If Q-head isn't wired, L_mart = 0.
    """
    device = x.device

    # ---- Martingale loss (only if the model has a Q/martingale head) ----
    if hasattr(model, "martingale_head") and (model.martingale_head is not None):
        # requires ItoFormer to set `self.last_hidden` in forward()
        if hasattr(model, "last_hidden") and (model.last_hidden is not None):
            q_pred = model.martingale_head(model.last_hidden)  # [B,T,A,1] or [B,T,1]
            # squeeze the last dim if present so martingale_drift_loss sees [B,T,A] / [B,T]
            if q_pred.size(-1) == 1:
                q_pred = q_pred.squeeze(-1)
            l_mart = martingale_drift_loss(q_pred)  # your implemented loss
        else:
            l_mart = torch.tensor(0.0, device=device)
    else:
        l_mart = torch.tensor(0.0, device=device)

    # ---- Itô penalty placeholder (wire later with realized quad var) ----
    l_ito = torch.tensor(0.0, device=device)

    return l_mart, l_ito

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float = 1.0,
    lambda_mart: float = 0.10,
    lambda_ito: float = 0.05,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        if len(batch) == 3:
            x, y, bias = batch
            bias = bias.to(device)
        else:
            x, y = batch
            bias = None
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = model(x, bias=bias)
                if pred.ndim == 3 and y.ndim == 3 and pred.size(1) == 1 and y.size(1) > 1:
                    y = y[:, -1:, :]
                l_pred = mse_loss(pred, y)
                l_mart, l_ito = compute_aux_losses(model, x, pred, bias)
                loss = l_pred + lambda_mart * l_mart + lambda_ito * l_ito

            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        else:
            pred = model(x, bias=bias)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            l_pred = mse_loss(pred, y)
            l_mart, l_ito = compute_aux_losses(model, x, pred, bias)
            loss = l_pred + lambda_mart * l_mart + lambda_ito * l_ito

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

class EncoderForecastModel(nn.Module):
    """
    Generic wrapper: takes X:[B,T,A,F] from the dataloader and adapts it
    for a baseline encoder that expects either:
      - 'VL'   -> [B, V, L]           (iTransformer / Pointwise)
      - 'VPPL' -> [B, V, P, P_len]    (PatchTST / CrossLite)
    Then pools over time (mean/last) if needed and forecasts with ForecastHead.
    """
    def __init__(self, encoder: nn.Module, d_model: int, n_assets: int,
                 expect: str, patch_len: int = 16, pool: str = "mean"):
        super().__init__()
        assert expect in ("VL", "VPPL")
        assert pool in ("mean", "last")
        self.encoder = encoder
        self.expect = expect
        self.patch_len = patch_len
        self.pool = pool
        self.head = ForecastHead(d_model=d_model, n_assets=n_assets)

    def _select_feature(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,A,F] -> [B,T,A]; pick the first feature (or mean if >1)
        if x.size(-1) == 1:
            return x[..., 0]
        return x.mean(dim=-1)

    def _to_VL(self, xTA: torch.Tensor) -> torch.Tensor:
        # [B,T,A] -> [B,V,L] with V=A, L=T
        return xTA.permute(0, 2, 1).contiguous()

    def _to_VPPL(self, xTA: torch.Tensor) -> torch.Tensor:
        # [B,T,A] -> [B,V,P,P_len] (no overlap; drop tail if not divisible)
        B, T, A = xTA.shape
        P_len = self.patch_len
        P = T // P_len
        if P == 0:
            raise RuntimeError(f"Sequence length T={T} shorter than patch_len={P_len}.")
        xTA = xTA[:, :P * P_len]                   # [B, P*P_len, A]
        xTA = xTA.view(B, P, P_len, A)             # [B, P, P_len, A]
        xVPP = xTA.permute(0, 3, 1, 2).contiguous()# [B, V(=A), P, P_len]
        return xVPP

    def forward(self, x, bias=None):
        # x: [B,T,A,F]  → pick feature → reshape per encoder
        x = self._select_feature(x)                 # [B,T,A]
        if self.expect == "VL":
            xin = self._to_VL(x)                    # [B,V,L]
        else:
            xin = self._to_VPPL(x)                  # [B,V,P,P_len]

        h = self.encoder(xin)                       # typically [B,V,D] or a container
        # Unwrap common container returns
        if isinstance(h, (tuple, list)):
            h = h[0]
        elif isinstance(h, dict):
            h = next(iter(h.values()))

        # If an encoder ever returns [B,T,A,D], pool over time
        if h.dim() == 4:
            h = h[:, -1] if self.pool == "last" else h.mean(dim=1)  # -> [B,A,D]
        elif h.dim() == 3:
            pass                                                    # [B,A,D] or [B,V,D]
        else:
            raise RuntimeError(f"Encoder returned unexpected shape {tuple(h.shape)}")

        # Forecast
        yhat = self.head(h)                          # usually [B,A] or [B,A,A]
        if isinstance(yhat, (tuple, list)):
            yhat = yhat[0]

        # If head returns [B,A,A], take diagonal per asset; else expect [B,A]
        if yhat.dim() == 3 and yhat.size(1) == yhat.size(2):
            B, A, _ = yhat.shape
            yhat = torch.stack([yhat[:, a, a] for a in range(A)], dim=1)
        elif yhat.dim() != 2:
            raise RuntimeError(f"Head returned unexpected shape {tuple(yhat.shape)}")

        return yhat.unsqueeze(1)                     # [B,1,A] to match exporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/etfs", help="Folder with ETF CSVs")
    parser.add_argument("--assets", type=str, nargs="+",
                        default=["SPY","QQQ","IWM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ","EFA","EEM"])
    parser.add_argument("--features-per-asset", type=int, default=1, help="e.g., returns only (=1) or returns+vol etc.")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--outdir", type=str, default="outputs/ito_equities")
    parser.add_argument("--lambda_mart", type=float, default=0.10,
                        help="Weight for martingale/no-drift penalty")
    parser.add_argument("--lambda_ito", type=float, default=0.05,
                        help="Weight for Ito quadratic-variation consistency")

    parser.add_argument("--encoder", type=str,
        default="itoformer",
        choices=["itoformer", "patchtst", "itransformer", "crosslite", "pointwise"])
    parser.add_argument("--patch-len", type=int, default=16, help="for patch-based encoders")
    parser.add_argument("--time-to-chan", type=int, default=64, help="for iTransformer")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean","last"],
        help="temporal pooling for non-ItôFormer encoders")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Dataloaders (must return X:[B,T,A,F], Y:[B,T,A], optional bias) ----
    train_loader, val_loader, meta = build_equities_dataloaders(
        data_root=Path(args.data_root),
        assets=args.assets,
        batch_size=args.batch_size,
    )
    n_assets = len(args.assets)
    in_features = args.features_per_asset

    # --- REPLACE the "Model" block in main() with this factory ---

    # ---- Model ----
    max_len = meta.get("max_len", 2048) if isinstance(meta, dict) else 2048

    # base config dict for saving later
    save_cfg = {
        "encoder": args.encoder,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "max_len": max_len,
    }
    if args.encoder in ("patchtst", "crosslite"):
        save_cfg.update({"patch_len": args.patch_len})
    if args.encoder == "itransformer":
        save_cfg.update({"time_to_chan": args.time_to_chan})
    if args.encoder == "itoformer":
        save_cfg.update({
            "in_features": in_features,
            "ff_mult": 2.0,
            "attn_dropout": 0.05,
            "ffn_dropout": 0.05,
            "resid_dropout": 0.05,
            "use_rmsnorm": True,
            "use_bias": True,
            "pos_embed": True,
        })

    # --- build the model according to encoder flag ---
    if args.encoder == "itoformer":
        cfg = ItoConfig(
            n_assets=n_assets,
            in_features=in_features,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            ff_mult=2.0,
            attn_dropout=0.05,
            ffn_dropout=0.05,
            resid_dropout=0.05,
            use_rmsnorm=True,
            use_bias=True,
            pos_embed=True,
            max_len=max_len,
        )
        head = ForecastHead(d_model=cfg.d_model, n_assets=n_assets)
        model = ItoFormer(cfg, head=head).to(device)

        # Optional: only ItôFormer uses martingale head
        # model.martingale_head = MartingaleHead(
        #     d_model=cfg.d_model,
        #     out_channels=n_assets,
        # ).to(device)

    else:
        # Build the chosen baseline encoder and wrap it with the correct expectations
        if args.encoder == "patchtst":
            enc = EncPatchTST(
                patch_len=args.patch_len,
                d_model=args.d_model,
                n_heads=args.heads,
                depth=args.layers,
                ffn=1024,
                dropout=0.10,
            )
            expect = "VPPL"  # expects [B, V, P, P_len]

        elif args.encoder == "itransformer":
            enc = EnciTransformer(
                L=max_len,                   # sequence length
                d_model=args.d_model,
                n_heads=args.heads,
                depth=args.layers,
                ffn=1024,
                dropout=0.10,
                time_to_chan=args.time_to_chan,
            )
            expect = "VL"  # expects [B, V, L]

        elif args.encoder == "crosslite":
            enc = CrossLite(
                patch_len=args.patch_len,
                d_model=args.d_model,
                n_heads_time=args.heads,
                depth_time=args.layers,
                ffn=1024,
                dropout=0.10,
                n_heads_cross=max(1, args.heads // 2),
                depth_cross=1,
                enable_cross=True,
            )
            expect = "VPPL"  # expects [B, V, P, P_len]

        elif args.encoder == "pointwise":
            enc = EncPointwise(
                in_dim=max_len,              # sequence length
                d_model=args.d_model,
                n_heads=args.heads,
                depth=args.layers,
                ffn=1024,
                dropout=0.10,
            )
            expect = "VL"  # expects [B, V, L]

        else:
            raise ValueError(f"Unknown encoder: {args.encoder}")

        # move model to device
        enc = enc.to(device)
        model = EncoderForecastModel(
            enc,
            d_model=args.d_model,
            n_assets=n_assets,
            expect=expect,
            patch_len=args.patch_len,
            pool=args.pool,
        ).to(device)

        # >>> ADD THESE THREE LINES <<<
        # model.martingale_head = MartingaleHead(
        #     d_model=args.d_model,
        #     out_channels=n_assets,  # one discounted series per asset
        # ).to(device)
        # >>> END ADD <<<

    # ---- Optimizer & scaler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    scaler = amp.GradScaler("cuda", enabled=args.use_amp)

    best_val = float("inf")

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            grad_clip=args.grad_clip,
            lambda_mart=args.lambda_mart,
            lambda_ito=args.lambda_ito,
        )
        val_loss = evaluate(
            model,
            val_loader,
            device,
            lambda_mart=args.lambda_mart,
            lambda_ito=args.lambda_ito,
        )

        print(f"[Epoch {epoch:03d}] train={train_loss:.6f}  val={val_loss:.6f}")

        # Save the best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": save_cfg,
            }
            torch.save(ckpt, outdir / "best.pt")

    # ------------------------------------------------------------------
    # Export last-epoch predictions on validation set (apples-to-apples)
    # ------------------------------------------------------------------
    from contextlib import nullcontext
    import numpy as np
    import pandas as pd

    model.eval()
    preds, targs = [], []

    amp_ctx = (
        amp.autocast("cuda", dtype=torch.float16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, y, bias = batch
                bias = bias.to(device)
            else:
                x, y = batch
                bias = None

            x = x.to(device)
            y = y.to(device)

            with amp_ctx:
                p = model(x, bias=bias)
                if isinstance(p, (tuple, list)):
                    p = p[0]

            preds.append(p.detach().cpu())
            targs.append(y.detach().cpu())

    # Flatten for CSV export
    preds = torch.cat(preds, dim=0).numpy()  # [B_total, T, A]
    targs = torch.cat(targs, dim=0).numpy()
    A = preds.shape[2]

    preds_2d = preds.reshape(-1, A)
    targs_2d = targs.reshape(-1, A)

    pd.DataFrame(preds_2d, columns=args.assets).to_csv(
        outdir / "val_predictions.csv", index=False
    )
    pd.DataFrame(targs_2d, columns=args.assets).to_csv(
        outdir / "val_targets.csv", index=False
    )
    print(f"Saved predictions/targets to {outdir}")


if __name__ == "__main__":
    main()
