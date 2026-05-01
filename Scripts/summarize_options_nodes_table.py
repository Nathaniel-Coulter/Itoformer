# scripts/summarize_options_nodes_table.py
import pandas as pd
from pathlib import Path

base = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")
train_path = base / "outputs" / "options_nodes_spx_win" / "logs" / "train.csv"
val_path = base / "outputs" / "options_nodes_spx_win" / "logs" / "val_summary.csv"

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)

# Clean numeric columns
for df in [train, val]:
    for c in df.columns:
        if c != "split":
            df[c] = pd.to_numeric(df[c], errors="coerce")

# Use final available epoch
last_train_epoch = train["epoch"].max()
last_val_epoch = val["epoch"].max()

train_last = train[train["epoch"] == last_train_epoch].copy()
val_last = val[val["epoch"] == last_val_epoch].copy()

summary = {
    "train_epoch": last_train_epoch,
    "val_epoch": last_val_epoch,

    "train_mse": train_last["mse"].mean(),
    "train_mae": train_last["mae"].mean(),
    "train_rmse": train_last["rmse"].mean(),

    "val_mse": val_last["mse"].mean(),
    "val_mae": val_last["mae"].mean(),
    "val_rmse": val_last["rmse"].mean(),

    "loss_noarb_bfly": train_last["loss_noarb_bfly"].mean(),
    "loss_noarb_cal": train_last["loss_noarb_cal"].mean(),
    "loss_ito": train_last["loss_ito"].mean(),
    "loss_martingale": train_last["loss_martingale"].mean(),
}

print("\n=== Options Nodes Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n=== LaTeX row ===")
print(
    f"Options Nodes & "
    f"{summary['val_mse']:.6f} & "
    f"{summary['val_mae']:.6f} & "
    f"{summary['val_rmse']:.6f} & "
    f"{summary['loss_noarb_bfly']:.6g} & "
    f"{summary['loss_noarb_cal']:.6g} & "
    f"{summary['loss_ito']:.6g} \\\\"
)