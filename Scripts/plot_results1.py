# scripts/plot_results.py
# Creates three visuals that include ALL models (ItoFormer, neural baselines, and classical baselines):
# 1) Pareto scatter: RMSE vs Sharpe(PnL)
# 2) Sharpe(PnL) bar chart
# 3) Per-asset MSE boxplots
#
# Outputs:
#   outputs/figures/pareto_rmse_vs_sharpe.png
#   outputs/figures/sharpe_bar_all_models.png
#   outputs/figures/mse_box_by_model.png

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------- PATHS (edit if your repo differs) ---------
ROOT = Path("/home/snowden/Desktop/quant_portfolio_scaffold")
overall_path = ROOT / "outputs/metrics_all.csv"
by_asset_path = ROOT / "outputs/metrics_by_asset.csv"
figdir = ROOT / "outputs/figures"
figdir.mkdir(parents=True, exist_ok=True)

# --------- LOAD ---------
overall = pd.read_csv(overall_path)
by_asset = pd.read_csv(by_asset_path)

def tidy_model(m: str) -> str:
    m = str(m)
    m = m.replace("_equities", "")
    # Optional rename for clarity in plots
    if m == "har": m = "har_rv"
    return m

overall["model"] = overall["model"].map(tidy_model)
by_asset["model"] = by_asset["model"].map(tidy_model)

# Sort overall by Sharpe(PnL) descending for consistent ordering across figures
overall_sorted = overall.sort_values("Sharpe(PnL)", ascending=False).reset_index(drop=True)

# --------- FIGURE 1: Pareto-style scatter (RMSE vs Sharpe(PnL)) ---------
plt.figure(figsize=(10, 6))
x = overall_sorted["RMSE"].values
y = overall_sorted["Sharpe(PnL)"].values
labels = overall_sorted["model"].values

plt.scatter(x, y, s=80)
for xi, yi, lab in zip(x, y, labels):
    # nudge text so it doesn't overlap the marker
    plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 6))

plt.xlabel("RMSE (lower is better)")
plt.ylabel("Sharpe(PnL) (higher is better)")
plt.title("Overall Performance — RMSE vs Portfolio Sharpe (all models)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(figdir / "pareto_rmse_vs_sharpe.png", dpi=150)
plt.close()

# --------- FIGURE 2: Sharpe(PnL) bar chart ---------
plt.figure(figsize=(12, 6))
plt.bar(overall_sorted["model"].values, overall_sorted["Sharpe(PnL)"].values)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Sharpe(PnL)")
plt.title("Portfolio Sharpe by Model")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(figdir / "sharpe_bar_all_models.png", dpi=150)
plt.close()

# --------- FIGURE 3: Per-asset MSE distributions (boxplots) ---------
plt.figure(figsize=(12, 7))
order = list(overall_sorted["model"].values)  # use same order as Sharpe chart
data, labels_ordered = [], []
for m in order:
    vals = by_asset.loc[by_asset["model"] == m, "MSE"].dropna().values
    if len(vals) > 0:
        data.append(vals)
        labels_ordered.append(m)
# add any models present in per-asset that weren't in overall (defensive)
for m in by_asset["model"].unique():
    if m not in labels_ordered:
        vals = by_asset.loc[by_asset["model"] == m, "MSE"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels_ordered.append(m)

plt.boxplot(data, labels=labels_ordered, showfliers=False)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Per-Asset MSE")
plt.title("Per-Asset MSE Distributions by Model")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(figdir / "mse_box_by_model.png", dpi=150)
plt.close()

print("Wrote:")
print(" -", figdir / "pareto_rmse_vs_sharpe.png")
print(" -", figdir / "sharpe_bar_all_models.png")
print(" -", figdir / "mse_box_by_model.png")
