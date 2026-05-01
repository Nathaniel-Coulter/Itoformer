# plot_svi_surface_fit.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")

RAW = ROOT / "data" / "options" / "spx_latest.csv"
SPOT = ROOT / "data" / "options" / "spx.csv"
PARAMS = ROOT / "outputs" / "options_svi_spx" / "eval" / "svi_params.csv"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(RAW)
spot = pd.read_csv(SPOT)
params = pd.read_csv(PARAMS)

raw["date"] = pd.to_datetime(raw["date"])
raw["expiration"] = pd.to_datetime(raw["expiration"])
spot["date"] = pd.to_datetime(spot["date"])
spot["close"] = pd.to_numeric(spot["close"], errors="coerce")

params["date"] = pd.to_datetime(params["date"])
params["expiry"] = pd.to_datetime(params["expiry"])

raw = raw.merge(
    spot[["date", "close"]].rename(columns={"close": "spx"}),
    on="date",
    how="left",
)

raw["strike"] = pd.to_numeric(raw["strike"], errors="coerce")
raw["iv"] = pd.to_numeric(raw["iv"], errors="coerce")
raw["spx"] = pd.to_numeric(raw["spx"], errors="coerce")

# Your data needs this adjustment
raw["strike_adj"] = raw["strike"] * 10.0
raw["k"] = np.log(raw["strike_adj"] / raw["spx"])

final_epoch = params["epoch"].max()
p_final = params[params["epoch"] == final_epoch].copy()

# Pick the date with the most final-epoch SVI slices
date = p_final["date"].value_counts().idxmax()
p_date = p_final[p_final["date"] == date].copy()

market = raw[raw["date"] == date].copy()
market = market.dropna(subset=["k", "iv", "spx", "expiration"])
market = market[(market["iv"] > 0) & np.isfinite(market["k"])]

# Restrict log-moneyness for readability
lo, hi = market["k"].quantile([0.02, 0.98])
market = market[(market["k"] >= lo) & (market["k"] <= hi)]

def svi_iv(k, dte, a, b, rho, m, sigma):
    T = max(float(dte) / 365.0, 1e-6)
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / T)

# ------------------------------------------------------------
# 1. Best single slice: market vs fitted SVI curve
# ------------------------------------------------------------
best = p_date.sort_values("surf_iv_rmse").iloc[0]

slice_df = market[market["expiration"] == best["expiry"]].copy()
if slice_df.empty:
    raise ValueError(f"No market IV rows for date={date.date()} expiry={best['expiry'].date()}")

k_grid = np.linspace(slice_df["k"].min(), slice_df["k"].max(), 250)
iv_fit = svi_iv(
    k_grid,
    best["dte"],
    best["a_hat"],
    best["b_hat"],
    best["rho_hat"],
    best["m_hat"],
    best["sigma_hat"],
)

plt.figure(figsize=(10, 6))
plt.scatter(slice_df["k"], slice_df["iv"], s=18, alpha=0.65, label="Market IV nodes")
plt.plot(k_grid, iv_fit, linewidth=2.5, label="Fitted SVI slice")
plt.xlabel(r"Log-moneyness $k=\log(K/S)$")
plt.ylabel("Implied volatility")
plt.title(f"SVI Slice Fit: {date.date()} expiry {best['expiry'].date()} ({int(best['dte'])} DTE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_slice = OUT / "svi_market_vs_fitted_slice.png"
plt.savefig(out_slice, dpi=300)
print(f"Saved: {out_slice}")
plt.show()

# ------------------------------------------------------------
# 2. Full fitted SVI 3D surface
# ------------------------------------------------------------
k_min, k_max = market["k"].quantile([0.03, 0.97])
k_grid = np.linspace(k_min, k_max, 120)

surface_rows = []
for _, r in p_date.iterrows():
    dte = float(r["dte"])
    if not np.isfinite(dte) or dte <= 0:
        continue

    iv_vals = svi_iv(
        k_grid,
        dte,
        float(r["a_hat"]),
        float(r["b_hat"]),
        float(r["rho_hat"]),
        float(r["m_hat"]),
        float(r["sigma_hat"]),
    )

    for k, iv in zip(k_grid, iv_vals):
        surface_rows.append((k, dte, iv))

surf = pd.DataFrame(surface_rows, columns=["k", "dte", "iv_fit"])

K = surf["k"].values.reshape(-1, len(k_grid))
DTE = surf["dte"].values.reshape(-1, len(k_grid))
IV = surf["iv_fit"].values.reshape(-1, len(k_grid))

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(K, DTE, IV, linewidth=0, antialiased=True, alpha=0.9)

ax.set_title(f"Fitted SVI Implied Volatility Surface ({date.date()})")
ax.set_xlabel(r"Log-moneyness $k$")
ax.set_ylabel("Days to Expiration")
ax.set_zlabel("Fitted Implied Volatility")
plt.tight_layout()

out_surface = OUT / "svi_fitted_surface_3d.png"
plt.savefig(out_surface, dpi=300)
print(f"Saved: {out_surface}")
plt.show()

# ------------------------------------------------------------
# 3. Market IV 3D scatter
# ------------------------------------------------------------
market_3d = market.copy()
market_3d["dte"] = (market_3d["expiration"] - market_3d["date"]).dt.days
market_3d = market_3d.dropna(subset=["k", "dte", "iv"])
market_3d = market_3d[(market_3d["dte"] > 0) & (market_3d["iv"] > 0)]

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    market_3d["k"],
    market_3d["dte"],
    market_3d["iv"],
    s=10,
    alpha=0.45,
)

ax.set_title(f"Observed Market IV Nodes ({date.date()})")
ax.set_xlabel(r"Log-moneyness $k$")
ax.set_ylabel("Days to Expiration")
ax.set_zlabel("Market Implied Volatility")
plt.tight_layout()

out_market = OUT / "svi_market_surface_3d.png"
plt.savefig(out_market, dpi=300)
print(f"Saved: {out_market}")
plt.show()

print("\nSelected date:", date.date())
print("Final epoch:", final_epoch)
print("Market rows:", len(market_3d))
print("SVI slices:", len(p_date))
print("Best slice:")
print(best)