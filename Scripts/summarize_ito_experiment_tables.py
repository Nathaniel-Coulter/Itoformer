# scripts/summarize_ito_experiment_tables.py
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")

RUNS = {
    "Options Nodes": BASE / "outputs" / "options_nodes_spx_win",
    "Options SVI": BASE / "outputs" / "options_svi_spx",
    "Rates HJM": BASE / "outputs" / "rates_hjm",
    "Rates HJM + Ito": BASE / "outputs" / "rates_hjm_ito",
}

def read_csv_safe(path):
    if not path.exists():
        print(f"[MISSING] {path}")
        return None
    df = pd.read_csv(path)
    for c in df.columns:
        if c != "split" and c != "date" and c != "expiry":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def mean_col(df, col):
    return df[col].mean() if df is not None and col in df.columns else float("nan")

def final_epoch(df):
    if df is None or "epoch" not in df.columns:
        return None, None
    ep = df["epoch"].max()
    return ep, df[df["epoch"] == ep].copy()

for name, run_dir in RUNS.items():
    train = read_csv_safe(run_dir / "logs" / "train.csv")
    train_epoch, train_last = final_epoch(train)

    # default summary
    summary = {
        "train_epoch": train_epoch,
        "val_epoch": float("nan"),
        "val_mse": float("nan"),
        "val_mae": float("nan"),
        "val_rmse": float("nan"),
        "train_mse": mean_col(train_last, "mse"),
        "train_mae": mean_col(train_last, "mae"),
        "train_rmse": mean_col(train_last, "rmse"),
        "loss_noarb_bfly": mean_col(train_last, "loss_noarb_bfly"),
        "loss_noarb_cal": mean_col(train_last, "loss_noarb_cal"),
        "loss_hjm_drift": mean_col(train_last, "loss_hjm_drift"),
        "hjm_violation_rate": mean_col(train_last, "hjm_violation_rate"),
        "loss_ito": mean_col(train_last, "loss_ito"),
        "loss_martingale": mean_col(train_last, "loss_martingale"),
        "bfly_violation_rate": float("nan"),
        "calendar_violation_rate": float("nan"),
    }

    if name == "Options SVI":
        svi_params = read_csv_safe(run_dir / "eval" / "svi_params.csv")
        noarb = read_csv_safe(run_dir / "diagnostics" / "no_arbitrage_svi.csv")

        ep_params, params_last = final_epoch(svi_params)
        ep_noarb, noarb_last = final_epoch(noarb)

        summary["val_epoch"] = ep_params
        summary["val_rmse"] = mean_col(params_last, "surf_iv_rmse")
        summary["loss_noarb_bfly"] = mean_col(noarb_last, "bfly_penalty_mean")
        summary["loss_noarb_cal"] = mean_col(noarb_last, "calendar_penalty_mean")
        summary["bfly_violation_rate"] = mean_col(noarb_last, "bfly_violation_rate")
        summary["calendar_violation_rate"] = mean_col(noarb_last, "calendar_violation_rate")

    else:
        val = read_csv_safe(run_dir / "logs" / "val_summary.csv")
        val_epoch, val_last = final_epoch(val)

        summary["val_epoch"] = val_epoch
        summary["val_mse"] = mean_col(val_last, "mse")
        summary["val_mae"] = mean_col(val_last, "mae")
        summary["val_rmse"] = mean_col(val_last, "rmse")

    print(f"\n=== {name} Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nLaTeX row:")
    print(
        f"{name} & "
        f"{summary['val_mse']:.6g} & "
        f"{summary['val_mae']:.6g} & "
        f"{summary['val_rmse']:.6g} & "
        f"{summary['loss_noarb_bfly']:.6g} & "
        f"{summary['loss_noarb_cal']:.6g} & "
        f"{summary['bfly_violation_rate']:.6g} & "
        f"{summary['calendar_violation_rate']:.6g} & "
        f"{summary['loss_hjm_drift']:.6g} & "
        f"{summary['hjm_violation_rate']:.6g} & "
        f"{summary['loss_ito']:.6g} \\\\"
    )