# scripts/plot_all_models_with_classical.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/snowden/Desktop/quant_portfolio_scaffold")
OUT = ROOT / "outputs"
NEURAL_ALL = OUT / "metrics_all.csv"
NEURAL_BYASSET = OUT / "metrics_by_asset.csv"
ITO_VAL_TARGETS = OUT / "itoformer_equities" / "val_targets.csv"
CLASSICAL_DIR = OUT / "baselines_equities"
FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def tidy_model(m: str) -> str:
    m = str(m).replace("_equities","")
    if m == "har": m = "har_rv"
    return m

def _safe_std(x):
    s = np.std(x)
    return s if s > 1e-12 else 1e-12

def mse(y, p): return float(np.mean((y - p) ** 2))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(mse(y, p)))
def hedging_error(y, p): return float(np.std(y - p))
def sharpe(x, ann_factor=252.0):
    mu, sd = np.mean(x), _safe_std(x)
    return float(np.sqrt(ann_factor) * (mu / sd))
def sortino(x, ann_factor=252.0):
    dn = np.std(np.minimum(x, 0.0))
    dn = dn if dn > 1e-12 else 1e-12
    return float(np.sqrt(ann_factor) * (np.mean(x) / dn))

def build_positions(pred, policy="proportional", cap=1.0):
    if policy == "proportional":
        return np.clip(pred, -cap, cap)
    elif policy == "sign":
        W = np.sign(pred)
        return np.clip(W, -cap, cap)
    else:
        raise ValueError(policy)

def turnover(W):
    if len(W) < 2: return 0.0
    return float(np.mean(np.abs(W[1:] - W[:-1])))

def pnl_from_positions(W, y, tcost_bps=0.0):
    pnl_asset_gross = W * y
    if tcost_bps and len(W) > 1:
        dW = np.abs(W[1:] - W[:-1])
        cost = (tcost_bps / 1e4) * dW
        pnl_asset = pnl_asset_gross.copy()
        pnl_asset[1:] -= cost
    else:
        pnl_asset = pnl_asset_gross
    pnl_port = np.mean(pnl_asset, axis=1)
    return pnl_asset, pnl_port

def order_models(names, target_order):
    seen = list(dict.fromkeys(names))
    ordered = [m for m in target_order if m in seen]
    ordered += [m for m in seen if m not in ordered]
    return ordered

# Choose a sensible prediction column from a classical CSV
def guess_pred_series(df: pd.DataFrame):
    # prefer explicit names
    for c in ["yhat","pred","prediction","forecast","mean","mu","fit"]:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return df[c].values.astype(np.float64)
    # else: the numeric column that is NOT an obvious target
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    targetish = {"y","target","ret","returns","actual"}
    candidates = [c for c in num_cols if c.lower() not in targetish]
    if len(candidates) == 1:
        return df[candidates[0]].values.astype(np.float64)
    # fallback: last numeric
    if num_cols:
        return df[num_cols[-1]].values.astype(np.float64)
    return None

# ---------- load neural metrics ----------
overall = pd.read_csv(NEURAL_ALL)
by_asset = pd.read_csv(NEURAL_BYASSET)
overall["model"] = overall["model"].map(tidy_model)
by_asset["model"] = by_asset["model"].map(tidy_model)

# ---------- load targets (validation) for alignment ----------
val_targets_df = pd.read_csv(ITO_VAL_TARGETS)    # shape [N_val, A]
asset_cols = list(val_targets_df.columns)
Y_val = val_targets_df.values.astype(np.float64)
N_val, A = Y_val.shape
ticker_to_col = {a: i for i, a in enumerate(asset_cols)}

# ---------- sweep classical files, compute metrics ----------
pattern_map = {
    "arima": re.compile(r"_arima_"),
    "garch": re.compile(r"_garch_"),
    "har_rv": re.compile(r"_har_"),
    "kalman": re.compile(r"_kalman_"),
}
class_rows, class_overall = [], []
used, skipped = [], []

if CLASSICAL_DIR.exists():
    files = sorted(CLASSICAL_DIR.glob("*.csv"))
    for f in files:
        name = f.name.lower()
        # determine model tag
        tag = None
        for k, pat in pattern_map.items():
            if pat.search(name):
                tag = k; break
        if tag is None:
            continue

        # extract asset ticker from filename prefix
        m = re.match(r"([A-Za-z]+)_", f.name)
        asset = m.group(1).upper() if m else None
        if asset is None or asset not in ticker_to_col:
            skipped.append((f.name, "asset_not_found_in_val_targets"))
            continue
        a_idx = ticker_to_col[asset]
        y = Y_val[:, a_idx]  # validation target for this asset

        df = pd.read_csv(f)
        p = guess_pred_series(df)
        if p is None or len(p) < 3:
            skipped.append((f.name, "no_numeric_pred_col"))
            continue

        # tail align to validation length
        n = min(len(y), len(p))
        y_ = y[-n:]
        p_ = p[-n:]

        # metrics
        W = build_positions(p_[:, None], policy="proportional", cap=1.0)[:, 0:1]
        pnl_a, pnl_port = pnl_from_positions(W, y_[:, None], tcost_bps=0.0)

        class_rows.append({
            "model": tag,
            "asset": asset,
            "MSE": mse(y_, p_),
            "MAE": mae(y_, p_),
            "RMSE": rmse(y_, p_),
            "HedgingError": hedging_error(y_, p_),
            "Sharpe(errors)": (np.mean(y_-p_) / _safe_std(y_-p_)),
            "Sortino(errors)": (np.mean(y_-p_) / _safe_std(np.minimum(y_-p_, 0.0))),
            "Turnover": turnover(W),  # per-asset turnover (1D)
            "Sharpe(PnL)": sharpe(pnl_a[:,0]),
            "Sortino(PnL)": sortino(pnl_a[:,0]),
            "N": float(n),
        })
        used.append(f.name)

# aggregate classical overall (average across assets per model)
if class_rows:
    class_by_asset = pd.DataFrame(class_rows)
    classical_overall = (
        class_by_asset.groupby("model")[["MSE","MAE","RMSE","HedgingError",
                                         "Sharpe(errors)","Sortino(errors)",
                                         "Turnover","Sharpe(PnL)","Sortino(PnL)","N"]]
        .mean()
        .reset_index()
    )
else:
    class_by_asset = pd.DataFrame(columns=by_asset.columns)
    classical_overall = pd.DataFrame(columns=overall.columns)

print("[classical_used]", len(used), "files")
for s in used[:12]: print("  ✓", s)
if len(used) > 12: print("  ...", len(used)-12, "more")
print("[classical_skipped]", len(skipped))
for s, why in skipped[:12]: print("  ✗", s, "->", why)
if len(skipped) > 12: print("  ...", len(skipped)-12, "more")

# ---------- combine neural + classical in memory (no overwrite) ----------
all_overall = pd.concat([overall, classical_overall], ignore_index=True)
all_by_asset = pd.concat([by_asset, class_by_asset], ignore_index=True)

TARGET_ORDER = [
    "itoformer", "patchtst", "itransformer", "crosslite", "pointwise",
    "arima", "garch", "har_rv", "kalman",
]
present_models = order_models(all_overall["model"].tolist(), TARGET_ORDER)
present_models_by_asset = order_models(all_by_asset["model"].tolist(), TARGET_ORDER)

# ============ FIGURE 1: Pareto RMSE vs Sharpe(PnL) ============
plt.figure(figsize=(11,6))
for m in present_models:
    row = all_overall[all_overall["model"]==m].head(1)
    if row.empty: continue
    x = row["RMSE"].values[0]
    y = row["Sharpe(PnL)"].values[0]
    if np.isfinite(x) and np.isfinite(y):
        plt.scatter(x, y, s=90)
        plt.annotate(m, (x, y), xytext=(6,6), textcoords="offset points")
plt.xlabel("RMSE (lower is better)")
plt.ylabel("Sharpe(PnL) (higher is better)")
plt.title("Overall Performance — RMSE vs Portfolio Sharpe (all models)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "ALL_pareto_rmse_vs_sharpe.png", dpi=150)
plt.close()

# ============ FIGURE 2: Sharpe(PnL) bar ============
rows = []
for m in present_models:
    r = all_overall[all_overall["model"]==m].head(1)
    if not r.empty:
        rows.append((m, r["Sharpe(PnL)"].values[0]))
if rows:
    labels, vals = zip(*rows)
    plt.figure(figsize=(12,6))
    plt.bar(labels, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Sharpe(PnL)")
    plt.title("Portfolio Sharpe by Model (ItôFormer, neural, classical)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "ALL_sharpe_bar.png", dpi=150)
    plt.close()

# ============ FIGURE 3: Per-asset MSE boxplots ============
box_data, box_labels = [], []
for m in present_models_by_asset:
    vals = all_by_asset.loc[all_by_asset["model"]==m, "MSE"].dropna().values
    if len(vals):
        box_data.append(vals)
        box_labels.append(m)
if box_data:
    plt.figure(figsize=(12,7))
    # matplotlib ≥3.9 prefers tick_labels
    try:
        plt.boxplot(box_data, tick_labels=box_labels, showfliers=False)
    except TypeError:
        plt.boxplot(box_data, labels=box_labels, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Per-Asset MSE")
    plt.title("Per-Asset MSE Distributions by Model (all models)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "ALL_mse_box_by_model.png", dpi=150)
    plt.close()

# ============ FIGURE 4: DM % p<0.05 vs ItôFormer ============
dm_path = OUT / "dm_vs_itoformer.csv"
if dm_path.exists():
    dm = pd.read_csv(dm_path)
    dm["model"] = dm["model"].map(tidy_model)
    dm = dm[dm["model"]!="itoformer"].copy()
    share = (dm.assign(sig=(dm["dm_pvalue"]<0.05))
               .groupby("model")["sig"].mean())
    # Add any classical models we computed (if they exist in DM file)
    ix = [m for m in TARGET_ORDER if m in share.index] + \
         [m for m in share.index if m not in TARGET_ORDER]
    share = share.loc[ix]
    if not share.empty:
        plt.figure(figsize=(12,5))
        plt.bar(share.index, 100*share.values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("% assets with p < 0.05 (vs ItôFormer)")
        plt.title("DM significance share (HAC-robust)")
        plt.ylim(0,100)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        for i,v in enumerate(share.values):
            plt.text(i, 100*v + 1.0, f"{100*v:.0f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGDIR / "ALL_dm_sigshare.png", dpi=150)
        plt.close()

print("Wrote figures to:", FIGDIR)
