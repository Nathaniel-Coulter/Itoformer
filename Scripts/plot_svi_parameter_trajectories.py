# plot_svi_parameter_trajectories.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")
CSV = ROOT / "outputs" / "options_svi_spx" / "eval" / "svi_params.csv"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

for c in ["epoch", "a_hat", "b_hat", "rho_hat", "m_hat", "sigma_hat"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Average across date-expiry slices per epoch
g = df.groupby("epoch", as_index=False)[
    ["a_hat", "b_hat", "rho_hat", "m_hat", "sigma_hat"]
].mean()

# Normalize each parameter so they are comparable on one chart
g_norm = g.copy()

for col in ["a_hat", "b_hat", "rho_hat", "m_hat", "sigma_hat"]:
    base = abs(g[col].iloc[0]) + 1e-8
    g_norm[col] = g[col] / base

plt.figure(figsize=(11, 6))

plt.plot(g_norm["epoch"], g_norm["a_hat"], label=r"$a$")
plt.plot(g_norm["epoch"], g_norm["b_hat"], label=r"$b$")
plt.plot(g_norm["epoch"], g_norm["rho_hat"], label=r"$\rho$")
plt.plot(g_norm["epoch"], g_norm["m_hat"], label=r"$m$")
plt.plot(g_norm["epoch"], g_norm["sigma_hat"], label=r"$\sigma$")

plt.xlabel("Epoch")
plt.ylabel("Normalized Mean SVI Parameter Value")
plt.title("Normalized SVI Parameter Trajectories Across Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_path = OUT / "svi_parameter_trajectories_normalized.png"
plt.savefig(out_path, dpi=300)
print(f"Saved: {out_path}")
plt.show()