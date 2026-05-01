# scripts/options_nodes_loss_decay_test.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")
TEST_ROOT = ROOT / "tests" / "options_nodes"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

runs = {
    "100 epochs": TEST_ROOT / "100 EPOCHS 0010 log price" / "train.csv",
    "200 epochs": TEST_ROOT / "200 EPOCHS 0010 log price" / "train.csv",
    "400 epochs": TEST_ROOT / "400 EPOCHS 0010 log price" / "train.csv",
    "1000 epochs": TEST_ROOT / "1000 EPOCHS 0010 log price" / "train.csv",
}

plt.figure(figsize=(10, 6))

for label, path in runs.items():
    if not path.exists():
        continue

    df = pd.read_csv(path)

    # ✅ only train
    df = df[df["split"] == "train"].copy()

    # ✅ group by epoch (THIS IS THE BIG FIX)
    df = df.groupby("epoch", as_index=False)["loss_ito"].mean()

    # ✅ sort
    df = df.sort_values("epoch")

    plt.plot(df["epoch"], df["loss_ito"], label=label)
    
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel(r"Itô consistency loss $\mathcal{L}_{Itô}$")
plt.title("Options Nodes: Itô Loss Decay Across Training Horizons")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

out_path = OUT_DIR / "options_nodes_ito_loss_decay.png"
plt.savefig(out_path, dpi=300)
print(f"\nSaved figure to: {out_path}")

plt.show()