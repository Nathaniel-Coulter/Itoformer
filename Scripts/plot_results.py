# scripts/plot_results.py
# Usage (from repo root):
#   PYTHONPATH=$PWD/src:$PYTHONPATH python scripts/plot_results.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths to your CSVs
root = Path("outputs")
all_path = root / "metrics_all.csv"
by_asset_path = root / "metrics_by_asset.csv"
dm_path = root / "dm_vs_itoformer.csv"

# Where to save figures
figdir = root / "figures"
figdir.mkdir(parents=True, exist_ok=True)

# ---- Load ----
metrics_all = pd.read_csv(all_path)
metrics_by_asset = pd.read_csv(by_asset_path)
dm = pd.read_csv(dm_path)

def _norm_name(x: str) -> str:
    return str(x).replace("_equities", "").strip().lower()

metrics_all["model"] = metrics_all["model"].map(_norm_name)
metrics_by_asset["model"] = metrics_by_asset["model"].map(_norm_name)
dm["model"] = dm["model"].map(_norm_name)

# Model display order if present
ordered_models = [
    "itoformer", "patchtst", "itransformer", "crosslite", "pointwise",
    "arima", "garch", "har", "kalman",
]

# =========================
# Figure 1: Portfolio Sharpe by model
# =========================
present_models_all = [m for m in ordered_models if m in metrics_all["model"].unique()]

df1 = metrics_all.loc[metrics_all["model"].isin(present_models_all), ["model", "Sharpe(PnL)"]].copy()
df1 = df1.sort_values("Sharpe(PnL)", ascending=False)

plt.figure(figsize=(9, 5))
plt.bar(df1["model"], df1["Sharpe(PnL)"])
plt.ylabel("Sharpe (PnL)")
plt.xlabel("Model")
plt.title("Portfolio Sharpe by Model (Validation)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(figdir / "sharpe_by_model.png", dpi=200)
plt.close()

# =========================
# Figure 2: Per-asset MSE heatmap
# =========================
present_models_asset = [m for m in ordered_models if m in metrics_by_asset["model"].unique()]
heat = (
    metrics_by_asset[metrics_by_asset["model"].isin(present_models_asset)]
    .pivot_table(index="asset", columns="model", values="MSE", aggfunc="mean")
    .reindex(columns=present_models_asset)
)

assets = list(heat.index)
plt.figure(figsize=(max(8, len(present_models_asset)*1.1), max(6, len(assets)*0.35)))
im = plt.imshow(heat.values, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.yticks(range(len(assets)), assets)
plt.xticks(range(len(present_models_asset)), present_models_asset, rotation=30, ha="right")
plt.xlabel("Model")
plt.ylabel("Asset")
plt.title("Per-Asset MSE Heatmap")
plt.tight_layout()
plt.savefig(figdir / "mse_heatmap_by_asset.png", dpi=200)
plt.close()

# =========================
# Figure 3: DM (mean) vs ItôFormer + % significant
# =========================
dm_clean = dm.dropna(subset=["dm_stat", "dm_pvalue"])
dm_models = [m for m in dm_clean["model"].unique() if m != "itoformer"]
dm_models = [m for m in ordered_models if m in dm_models]

rows = []
for m in dm_models:
    sub = dm_clean[dm_clean["model"] == m]
    if len(sub) == 0:
        continue
    mean_dm = sub["dm_stat"].mean()
    sig_rate = float((sub["dm_pvalue"] < 0.05).mean())  # fraction of assets with p<0.05
    rows.append((m, mean_dm, sig_rate))

dm_summary = pd.DataFrame(rows, columns=["model", "mean_dm", "sig_rate"]).sort_values("mean_dm", ascending=False)

plt.figure(figsize=(9, 5))
plt.bar(dm_summary["model"], dm_summary["mean_dm"])
for i, r in dm_summary.reset_index(drop=True).iterrows():
    plt.text(i, r["mean_dm"], f"{int(round(r['sig_rate']*100))}% sig", ha="center", va="bottom", fontsize=9)
plt.axhline(0.0, linestyle="--")
plt.ylabel("Average DM Statistic (vs ItôFormer)")
plt.xlabel("Model")
plt.title("Diebold–Mariano Mean Statistic vs ItôFormer (HAC-robust)\nAnnotation: % assets with p<0.05")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(figdir / "dm_mean_by_model.png", dpi=200)
plt.close()

# =========================
# Figure 4: Turnover vs Sharpe (PnL)
# =========================
df4 = metrics_all.loc[metrics_all["model"].isin(present_models_all), ["model", "Turnover", "Sharpe(PnL)"]].copy()

plt.figure(figsize=(7, 6))
plt.scatter(df4["Turnover"], df4["Sharpe(PnL)"])
for _, r in df4.iterrows():
    # label each point with the model name (slightly nudged)
    plt.text(r["Turnover"], r["Sharpe(PnL)"], r["model"], fontsize=9, ha="left", va="bottom")
plt.xlabel("Turnover")
plt.ylabel("Sharpe (PnL)")
plt.title("Turnover vs Sharpe (PnL) by Model")
plt.tight_layout()
plt.savefig(figdir / "turnover_vs_sharpe.png", dpi=200)
plt.close()

print("Saved figures to:", figdir.resolve())
