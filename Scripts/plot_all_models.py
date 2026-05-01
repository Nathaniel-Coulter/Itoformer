# scripts/plot_all_models.py
# Plots ALL models (neural + classical) when present in outputs/*.csv.
# Figures:
#   1) Pareto scatter: RMSE vs Sharpe(PnL) (all models)
#   2) Sharpe(PnL) bar (all models)
#   3) Per-asset MSE boxplots (all models)
#   4) % assets with DM p<0.05 vs ItôFormer (all models that appear in dm file)
#
# Writes to: outputs/figures/

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/snowden/Desktop/quant_portfolio_scaffold")
overall_path = ROOT / "outputs/metrics_all.csv"
by_asset_path = ROOT / "outputs/metrics_by_asset.csv"
dm_path = ROOT / "outputs/dm_vs_itoformer.csv"
figdir = ROOT / "outputs/figures"
figdir.mkdir(parents=True, exist_ok=True)

# Desired display order (if present)
TARGET_ORDER = [
    "itoformer", "patchtst", "itransformer", "crosslite", "pointwise",
    "arima", "garch", "har_rv", "kalman",
]

def tidy_model(m: str) -> str:
    m = str(m).replace("_equities", "")
    if m == "har": m = "har_rv"
    return m

# ---- Load
overall = pd.read_csv(overall_path)
by_asset = pd.read_csv(by_asset_path)
overall["model"] = overall["model"].map(tidy_model)
by_asset["model"] = by_asset["model"].map(tidy_model)

# Helper: order models by TARGET_ORDER then by Sharpe if ties/absent
def order_models(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["model"].tolist()))
    # keep TARGET_ORDER first (intersection), then any extras
    ordered = [m for m in TARGET_ORDER if m in present]
    ordered += [m for m in present if m not in ordered]
    return ordered

present_models_overall = order_models(overall)
present_models_by_asset = order_models(by_asset)

# =========================
# Figure 1: Pareto (RMSE vs Sharpe)
# =========================
plt.figure(figsize=(11, 6))
plotted, skipped = [], []
for m in present_models_overall:
    row = overall.loc[overall["model"] == m].head(1)
    if row.empty: 
        skipped.append((m, "not in metrics_all")); 
        continue
    rmse = row["RMSE"].values[0]
    sr = row["Sharpe(PnL)"].values[0]
    if np.isfinite(rmse) and np.isfinite(sr):
        plt.scatter(rmse, sr, s=90)
        plt.annotate(m, (rmse, sr), xytext=(6, 6), textcoords="offset points")
        plotted.append(m)
    else:
        skipped.append((m, "RMSE/Sharpe NaN"))

plt.xlabel("RMSE (lower is better)")
plt.ylabel("Sharpe(PnL) (higher is better)")
plt.title("Overall Performance — RMSE vs Portfolio Sharpe (ALL models present)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(figdir / "pareto_rmse_vs_sharpe_all.png", dpi=150)
plt.close()
print("[Pareto] plotted:", plotted)
print("[Pareto] skipped:", skipped)

# =========================
# Figure 2: Sharpe(PnL) bar — all models available
# =========================
bars = []
for m in present_models_overall:
    row = overall.loc[overall["model"] == m].head(1)
    if row.empty: continue
    val = row["Sharpe(PnL)"].values[0]
    bars.append((m, val))

if bars:
    labels, vals = zip(*bars)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Sharpe(PnL)")
    plt.title("Portfolio Sharpe by Model (ALL models present)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "sharpe_bar_all_models.png", dpi=150)
    plt.close()
    print("[Sharpe Bar] models:", labels)
else:
    print("[Sharpe Bar] nothing to plot (no models in metrics_all).")

# =========================
# Figure 3: Per-asset MSE boxplots — all models available
# =========================
box_data, box_labels = [], []
for m in present_models_by_asset:
    vals = by_asset.loc[by_asset["model"] == m, "MSE"].dropna().values
    if len(vals) > 0:
        box_data.append(vals)
        box_labels.append(m)

if box_data:
    plt.figure(figsize=(12, 7))
    plt.boxplot(box_data, labels=box_labels, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Per-Asset MSE")
    plt.title("Per-Asset MSE Distributions by Model (ALL models present)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "mse_box_by_model_all.png", dpi=150)
    plt.close()
    print("[MSE Box] models:", box_labels)
else:
    print("[MSE Box] nothing to plot (no per-asset rows).")

# =========================
# Figure 4: % assets with DM p<0.05 vs ItôFormer — all models
# =========================
if dm_path.exists():
    dm = pd.read_csv(dm_path)
    dm["model"] = dm["model"].map(tidy_model)
    # Some scripts may label the comparison target as "itoformer" in the model column — drop it
    dm = dm[dm["model"] != "itoformer"].copy()

    # Compute percent of assets with p<0.05 per model
    sig = dm.assign(sig=(dm["dm_pvalue"] < 0.05))
    share = sig.groupby("model")["sig"].mean().reindex(TARGET_ORDER).dropna()
    # Also include any extra models not in TARGET_ORDER
    extras = sig.groupby("model")["sig"].mean().drop(index=share.index, errors="ignore")
    share = pd.concat([share, extras])

    if not share.empty:
        plt.figure(figsize=(12, 5))
        plt.bar(share.index, 100.0 * share.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("% assets with p < 0.05 (vs ItôFormer)")
        plt.title("Diebold–Mariano Significance Share by Model (HAC-robust)")
        plt.ylim(0, 100)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(share.values):
            plt.text(i, 100.0*v + 1.0, f"{100.0*v:.0f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(figdir / "dm_sigshare_all_models.png", dpi=150)
        plt.close()
        print("[DM %<0.05] models:", list(share.index))
    else:
        print("[DM %<0.05] nothing to plot (dm_vs_itoformer.csv has no usable rows).")
else:
    print(f"[DM %<0.05] file missing: {dm_path}")

print("Wrote figures to:", figdir)
