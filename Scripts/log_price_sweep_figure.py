import os
import matplotlib.pyplot as plt

alpha = [0.00, 0.05, 0.075, 0.09, 0.095, 0.10, 0.105, 0.11, 0.125, 0.15]

val_mse = [
    0.142738,  # 0.00
    0.136218,  # 0.05
    0.135953,  # 0.075
    0.135520,  # 0.09
    0.141903,  # 0.095
    0.134880,  # 0.10
    0.135583,  # 0.105
    0.137851,  # 0.11
    0.136231,  # 0.125
    0.140596   # 0.15
]

# Identify minimum
min_idx = val_mse.index(min(val_mse))
alpha_min = alpha[min_idx]
mse_min = val_mse[min_idx]

# Output directory
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

# Plot
plt.figure(figsize=(6.5, 4.5))

plt.plot(
    alpha,
    val_mse,
    marker='o',
    linewidth=2,
    markersize=6,
    alpha=0.85
)

# Highlight minimum
plt.scatter(
    alpha_min,
    mse_min,
    color='red',
    zorder=5
)

plt.axvline(alpha_min, linestyle='--', color='red', alpha=0.6)

# Annotation
plt.annotate(
    r"$\alpha = 0.10$ (min)",
    xy=(alpha_min, mse_min),
    xytext=(alpha_min + 0.01, mse_min + 0.0006),
    arrowprops=dict(arrowstyle="->", color='red'),
    fontsize=10
)

plt.xlabel("Log-price scaling $\\alpha$")
plt.ylabel("Validation MSE (epoch 50)")
plt.title("Log-Price Scaling Sweep: Options Nodes")

plt.grid(alpha=0.25)

# Save
out_path = os.path.join(out_dir, "log_price_sweep_options_nodes.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved figure to: {out_path}")