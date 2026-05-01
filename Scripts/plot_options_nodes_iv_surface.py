# notepad .\scripts\plot_options_nodes_iv_surface.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")
DATA = ROOT / "data" / "options" / "spx_latest.csv"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df["expiration"] = pd.to_datetime(df["expiration"])

# choose latest date with enough observations
latest_date = df["date"].max()
day = df[df["date"] == latest_date].copy()

# keep clean IV rows
day = day.dropna(subset=["strike", "dte", "iv"])
day = day[(day["iv"] > 0) & (day["dte"] > 0)]

# limit to useful strike range to avoid extreme artifacts
lo, hi = day["strike"].quantile([0.02, 0.98])
day = day[(day["strike"] >= lo) & (day["strike"] <= hi)]

# pivot to strike x dte grid
surf = day.pivot_table(index="strike", columns="dte", values="iv", aggfunc="mean")
surf = surf.dropna(axis=0, how="all").dropna(axis=1, how="all")

# fill small gaps for plotting only
surf = surf.interpolate(axis=0).interpolate(axis=1).ffill().bfill()

X, Y = np.meshgrid(surf.columns.astype(float), surf.index.astype(float))
Z = surf.values

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.95)

ax.set_title(f"SPX Implied Volatility Surface ({latest_date.date()})")
ax.set_xlabel("Days to Expiration")
ax.set_ylabel("Strike")
ax.set_zlabel("Implied Volatility")

plt.tight_layout()
out_path = OUT / "options_nodes_market_iv_surface_3d.png"
plt.savefig(out_path, dpi=300)
print(f"Saved: {out_path}")
plt.show()