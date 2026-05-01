from __future__ import annotations
import argparse
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

# =========================
# Basic forecast error metrics
# =========================
def mse(y, p): return np.mean((y - p) ** 2)
def mae(y, p): return np.mean(np.abs(y - p))
def rmse(y, p): return np.sqrt(mse(y, p))
def hedging_error(y, p): return np.std(y - p)

def _safe_std(x):
    s = np.std(x)
    return s if s > 1e-12 else 1e-12

def sharpe_of_errors(y, p):
    e = y - p
    return np.mean(e) / _safe_std(e)

def sortino_of_errors(y, p):
    e = y - p
    downside = np.std(np.minimum(e, 0.0))
    downside = downside if downside > 1e-12 else 1e-12
    return np.mean(e) / downside

# =========================
# Portfolio metrics from predictions (no retrain)
# =========================
def build_positions(pred: np.ndarray, policy: str = "proportional", cap: float = 1.0) -> np.ndarray:
    """
    pred: [N, A] predictions (assumed returns)
    Returns positions W in [-cap, cap] with same shape.
    """
    if policy == "proportional":
        W = np.clip(pred, -cap, cap)
    elif policy == "sign":
        W = np.sign(pred)
        if cap is not None:
            W = np.clip(W, -cap, cap)
    else:
        raise ValueError(f"Unknown policy '{policy}'")
    return W

def portfolio_turnover(W: np.ndarray) -> float:
    """
    Average L1 change in weights per period, averaged across assets.
    W: [N, A]
    """
    if len(W) < 2:
        return 0.0
    dW = np.abs(W[1:] - W[:-1])          # [N-1, A]
    return float(np.mean(dW))            # scalar

def pnl_from_positions(W: np.ndarray, y: np.ndarray, tcost_bps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-asset PnL and equal-weight portfolio PnL, net of transaction costs.
    W: [N, A], y: [N, A], tcost_bps: cost per unit turnover (bps)
    Returns: (pnl_asset [N,A], pnl_portfolio [N])
    """
    pnl_asset_gross = W * y              # [N, A]
    # apply transaction cost on each rebalance step t>0
    if tcost_bps and len(W) > 1:
        dW = np.abs(W[1:] - W[:-1])      # [N-1, A]
        # cost in returns = (bps/1e4) * change in position magnitude
        cost = (tcost_bps / 1e4) * dW
        # subtract cost at t (align): cost_t applies at t for the rebalance from t-1->t
        pnl_asset = pnl_asset_gross.copy()
        pnl_asset[1:] -= cost
    else:
        pnl_asset = pnl_asset_gross

    pnl_port = np.mean(pnl_asset, axis=1)  # equal-weight across assets
    return pnl_asset, pnl_port

def sharpe(x: np.ndarray, ann_factor: float = 252.0) -> float:
    mu = np.mean(x)
    sd = _safe_std(x)
    sr = mu / sd
    return float(np.sqrt(ann_factor) * sr)

def sortino(x: np.ndarray, ann_factor: float = 252.0) -> float:
    downside = np.std(np.minimum(x, 0.0))
    downside = downside if downside > 1e-12 else 1e-12
    s = np.mean(x) / downside
    return float(np.sqrt(ann_factor) * s)

# =========================
# HAC/Newey-West for DM test
# =========================
def newey_west_var(d, max_lag=None):
    d = np.asarray(d).reshape(-1)
    T = len(d)
    d_centered = d - d.mean()
    gamma0 = np.dot(d_centered, d_centered) / T
    if max_lag is None:
        max_lag = int(np.floor(T ** (1/3)))
    s = gamma0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        cov = np.dot(d_centered[k:], d_centered[:-k]) / T
        s += 2.0 * w * cov
    return s / T

def dm_test_hac(*args):
    """
    HAC-robust Diebold–Mariano test.
    Accepts either:
      dm_test_hac(y, p1, p2)   -> builds error series from y - p
      dm_test_hac(e1, e2)      -> uses error series directly
    Returns (dm_stat, p_value).
    """
    if len(args) == 3:
        y, p1, p2 = args
        e1 = np.asarray(y) - np.asarray(p1)
        e2 = np.asarray(y) - np.asarray(p2)
    elif len(args) == 2:
        e1, e2 = args
        e1 = np.asarray(e1)
        e2 = np.asarray(e2)
    else:
        raise TypeError("dm_test_hac() expects either (y, p1, p2) or (e1, e2)")

    # align lengths defensively
    n = min(len(e1), len(e2))
    e1 = e1[:n].reshape(-1)
    e2 = e2[:n].reshape(-1)

    d = e1**2 - e2**2
    T = len(d)
    if T < 5:
        return np.nan, np.nan

    var_mean = newey_west_var(d)
    if var_mean <= 0 or not np.isfinite(var_mean):
        return np.nan, np.nan

    dm = d.mean() / np.sqrt(var_mean)
    p = 2 * (1 - norm.cdf(abs(dm)))
    return dm, p

# =========================
# I/O helpers
# =========================
def _to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.values.astype(np.float64)

def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a[:, None] if a.ndim == 1 else a

def _align_lengths(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(y), len(p))
    return y[:n], p[:n]

def load_neural(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    P = _to_numpy(_load_csv(folder / "val_predictions.csv"))
    Y = _to_numpy(_load_csv(folder / "val_targets.csv"))
    # Ensure 2D
    P = _ensure_2d(P)
    Y = _ensure_2d(Y)
    # Align lengths robustly: if one is longer (common: Y >> P), take the LAST n rows
    n = min(P.shape[0], Y.shape[0])
    if P.shape[0] != Y.shape[0]:
        P = P[-n:, :]
        Y = Y[-n:, :]
    return Y, P

_CLASSICAL_TAGS = {
    "arima": re.compile(r"_arima_"),
    "garch": re.compile(r"_garch_"),
    "har":   re.compile(r"_har_"),
    "kalman":re.compile(r"_kalman_"),
}

def _guess_pred_col(df: pd.DataFrame) -> str | None:
    for c in ["yhat", "pred", "prediction", "forecast"]:
        if c in df.columns: return c
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    return numeric_cols[0] if len(numeric_cols) == 1 else None

def _guess_target_col(df: pd.DataFrame) -> str | None:
    for c in ["y", "target", "ret", "returns"]:
        if c in df.columns: return c
    return None

def load_classical(folder: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict] = {}
    all_files = list(folder.glob("*.csv"))
    if not all_files:
        return out
    by_tag = {tag: [] for tag in _CLASSICAL_TAGS}
    for f in all_files:
        name = f.name.lower()
        for tag, pat in _CLASSICAL_TAGS.items():
            if pat.search(name):
                by_tag[tag].append(f); break
    for tag, files in by_tag.items():
        if not files: continue
        assets, preds, targets = [], [], []
        for f in sorted(files):
            df = _load_csv(f)
            pred_col = _guess_pred_col(df)
            y_col = _guess_target_col(df)
            if pred_col is None or y_col is None:
                warnings.warn(f"[{tag}] Missing pred/target in {f.name}; skipping.")
                continue
            p_vec = df[pred_col].values.astype(np.float64)
            y_vec = df[y_col].values.astype(np.float64)
            y_vec, p_vec = _align_lengths(y_vec, p_vec)
            m = re.match(r"([A-Z]+)_", f.stem.upper())
            asset = m.group(1) if m else f.stem.upper()
            assets.append(asset); preds.append(p_vec); targets.append(y_vec)
        if preds and targets:
            minN = min(map(len, targets))
            Y = np.stack([t[:minN] for t in targets], axis=1)
            P = np.stack([p[:minN] for p in preds], axis=1)
            out[tag] = {"y": Y, "p": P, "assets": assets}
    return out

# =========================
# Metric aggregation
# =========================
def compute_metrics_table(
    y: np.ndarray, p: np.ndarray, model_name: str, assets: list[str],
    policy: str, cap: float, tcost_bps: float, ann_factor: float
) -> tuple[pd.DataFrame, pd.Series]:
    """
    y, p: [N, A]
    Returns per-asset dataframe and overall (averaged) series.
    """
    N, A = y.shape
    W = build_positions(p, policy=policy, cap=cap)     # [N, A]
    t_over = portfolio_turnover(W)

    # --- Align returns to the prediction horizon ---
    # W: [N, A] from predictions (validation horizon)
    # y: [M, A] from full data (train+val). We need y for the same N rows as W.
    if isinstance(W, pd.DataFrame) and isinstance(y, pd.DataFrame):
        # If both have a DatetimeIndex and overlap, align on index
        if isinstance(W.index, pd.DatetimeIndex) and isinstance(y.index, pd.DatetimeIndex):
            common_idx = W.index.intersection(y.index)
            if len(common_idx) > 0:
                W = W.loc[common_idx]
                y = y.loc[common_idx]
            else:
                # fallback: tail-align by length
                N = len(W)
                y = y.iloc[-N:]
        else:
            # fallback: tail-align by length
            N = len(W)
            y = y.iloc[-N:]
    else:
        # numpy fallback
        N = W.shape[0]
        y = y[-N:, :]

    # After alignment, enforce same shape
    assert W.shape == y.shape, f"Shape mismatch after alignment: W{W.shape} vs y{y.shape}"




    pnl_asset, pnl_port = pnl_from_positions(W, y, tcost_bps=tcost_bps)

    rows = []
    for a in range(A):
        ya, pa = y[:, a], p[:, a]
        Wa = W[:, a]
        pnl_a = pnl_asset[:, a]
        rows.append({
            "model": model_name,
            "asset": assets[a] if assets and a < len(assets) else f"A{a}",
            "MSE": mse(ya, pa),
            "MAE": mae(ya, pa),
            "RMSE": rmse(ya, pa),
            "HedgingError": hedging_error(ya, pa),
            "Sharpe(errors)": sharpe_of_errors(ya, pa),
            "Sortino(errors)": sortino_of_errors(ya, pa),
            "Turnover": portfolio_turnover(Wa[:, None]),    # per-asset turnover
            "Sharpe(PnL)": sharpe(pnl_a, ann_factor=ann_factor),
            "Sortino(PnL)": sortino(pnl_a, ann_factor=ann_factor),
            "N": len(ya),
        })
    per_asset = pd.DataFrame(rows)

    overall = pd.Series({
        "model": model_name,
        "MSE": float(per_asset["MSE"].mean()),
        "MAE": float(per_asset["MAE"].mean()),
        "RMSE": float(per_asset["RMSE"].mean()),
        "HedgingError": float(per_asset["HedgingError"].mean()),
        "Sharpe(errors)": float(per_asset["Sharpe(errors)"].mean()),
        "Sortino(errors)": float(per_asset["Sortino(errors)"].mean()),
        "Turnover": float(t_over),                                      # portfolio turnover
        "Sharpe(PnL)": sharpe(pnl_port, ann_factor=ann_factor),         # portfolio sharpe
        "Sortino(PnL)": sortino(pnl_port, ann_factor=ann_factor),       # portfolio sortino
        "N_avg": float(per_asset["N"].mean()),
    })
    return per_asset, overall

def dm_vs_itoformer(ito_y: np.ndarray,
                    ito_p: np.ndarray,
                    y: np.ndarray,
                    p: np.ndarray,
                    tag: str,
                    asset_cols: list[str]) -> pd.DataFrame:
    """
    Compare each model's errors to ItôFormer's via HAC-robust DM test.
    Inputs are aligned arrays over the common validation window:
      - ito_y, ito_p: [N, A] (ItôFormer targets/preds)
      - y, p:         [N, A] (other model targets/preds)
    Returns a dataframe with DM statistic and p-value per asset.
    """
    rows = []
    N, A = y.shape
    assert ito_y.shape == (N, A) and ito_p.shape == (N, A), "ItôFormer shapes must match model shapes"

    for a in range(A):
        # build error series
        e_model = y[:, a] - p[:, a]
        e_ito   = ito_y[:, a] - ito_p[:, a]

        # clean NaNs/Infs defensively
        mask = np.isfinite(e_model) & np.isfinite(e_ito)
        if mask.sum() < 5:
            dm_stat, pval = np.nan, np.nan
        else:
            dm_stat, pval = dm_test_hac(e_model[mask], e_ito[mask])

        rows.append({
            "model": tag,
            "asset": asset_cols[a],
            "dm_stat": dm_stat,
            "dm_pvalue": pval,
        })

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=str, default="outputs")
    ap.add_argument("--ito-dir", type=str, default="itoformer_equities")
    ap.add_argument("--neural-dirs", nargs="+",
                    default=["patchtst_equities", "itransformer_equities", "crosslite_equities", "pointwise_equities"])
    ap.add_argument("--classical-dir", type=str, default="baselines_equities")

    # New knobs for portfolio metrics
    ap.add_argument("--policy", type=str, default="proportional", choices=["proportional", "sign"])
    ap.add_argument("--cap", type=float, default=1.0, help="Max absolute position for proportional/sign policies")
    ap.add_argument("--tcost-bps", type=float, default=0.0, help="Transaction cost per unit turnover (bps)")
    ap.add_argument("--ann-factor", type=float, default=252.0, help="Annualization factor for Sharpe/Sortino")

    args = ap.parse_args()

    root = Path(args.outputs_root)

    # ---- Load ItôFormer (reference) ----
    ito_y, ito_p = load_neural(root / args.ito_dir)
    A = ito_y.shape[1]
    try:
        asset_cols = list(pd.read_csv(root / args.ito_dir / "val_targets.csv").columns)
    except Exception:
        asset_cols = [f"A{i}" for i in range(A)]

    per_asset_rows, overall_rows = [], []

    # Score ItôFormer
    pa, ov = compute_metrics_table(
        ito_y, ito_p, "itoformer", asset_cols,
        policy=args.policy, cap=args.cap, tcost_bps=args.tcost_bps, ann_factor=args.ann_factor
    )
    per_asset_rows.append(pa); overall_rows.append(ov)

    # ---- NEURAL baselines ----
    for d in args.neural_dirs:
        folder = root / d
        if not (folder / "val_predictions.csv").exists():
            warnings.warn(f"Skipping {d}: missing val_predictions.csv")
            continue
        y, p = load_neural(folder)
        n = min(len(ito_y), len(y))
        y, p = y[:n, :A], p[:n, :A]
        pa, ov = compute_metrics_table(
            y, p, d.replace("_equities",""), asset_cols,
            policy=args.policy, cap=args.cap, tcost_bps=args.tcost_bps, ann_factor=args.ann_factor
        )
        per_asset_rows.append(pa); overall_rows.append(ov)

    # ---- CLASSICAL baselines ----
    classical_map = load_classical(root / args.classical_dir)
    for tag, pack in classical_map.items():
        y, p = pack["y"], pack["p"]
        n = min(len(ito_y), len(y))
        aN = min(A, y.shape[1])
        y, p = y[:n, :aN], p[:n, :aN]
        assets = (pack.get("assets") or asset_cols)[:aN]
        pa, ov = compute_metrics_table(
            y, p, tag, assets,
            policy=args.policy, cap=args.cap, tcost_bps=args.tcost_bps, ann_factor=args.ann_factor
        )
        per_asset_rows.append(pa); overall_rows.append(ov)

    # ---- Write metrics tables ----
    per_asset_df = pd.concat(per_asset_rows, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df[
        ["model","MSE","MAE","RMSE","HedgingError","Sharpe(errors)","Sortino(errors)",
         "Turnover","Sharpe(PnL)","Sortino(PnL)","N_avg"]
    ]

    per_asset_df.to_csv(root / "metrics_by_asset.csv", index=False)
    overall_df.to_csv(root / "metrics_all.csv", index=False)
    print("Saved:", root / "metrics_by_asset.csv")
    print("Saved:", root / "metrics_all.csv")

    # ---- DM vs ItôFormer ----
    dm_frames = []
    for d in args.neural_dirs:
        folder = root / d
        if not (folder / "val_predictions.csv").exists():
            continue
        y, p = load_neural(folder)
        n = min(len(ito_y), len(y))
        y, p = y[:n, :A], p[:n, :A]
        dm_frames.append(dm_vs_itoformer(ito_y[:n,:A], ito_p[:n,:A], y, p, d.replace("_equities",""), asset_cols))

    for tag, pack in classical_map.items():
        y, p = pack["y"], pack["p"]
        n = min(len(ito_y), len(y))
        aN = min(A, y.shape[1])
        dm_frames.append(dm_vs_itoformer(ito_y[:n,:aN], ito_p[:n,:aN], y[:n,:aN], p[:n,:aN], tag, asset_cols[:aN]))

    if dm_frames:
        dm_df = pd.concat(dm_frames, ignore_index=True)
        dm_df.to_csv(root / "dm_vs_itoformer.csv", index=False)
        print("Saved:", root / "dm_vs_itoformer.csv")
    else:
        print("No DM comparisons written (not enough overlapping data).")

if __name__ == "__main__":
    main()
