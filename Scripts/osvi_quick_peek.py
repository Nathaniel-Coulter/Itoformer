# scripts/osvi_quick_peek.py
# options_svi quick peek

import os
import pandas as pd
import numpy as np

ROOT = r"C:/Users/admin/Desktop/quant_portfolio_scaffold/outputs/options_svi_spx"  # adjust if you changed outdir

def read_csv_safe(path: str) -> pd.DataFrame:
    """
    Tries fast C engine first. If the file contains a few malformed
    legacy lines (extra commas), fall back to the Python engine and
    skip bad lines.
    """
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", low_memory=False)

# --- load ---
train_path  = os.path.join(ROOT, "logs/train.csv")
params_path = os.path.join(ROOT, "eval/svi_params.csv")
noarb_path  = os.path.join(ROOT, "diagnostics/no_arbitrage_svi.csv")

train  = read_csv_safe(train_path)
params = read_csv_safe(params_path)
noarb  = read_csv_safe(noarb_path)

# --- coerce numeric columns we expect to summarize ---
num_like_cols = [
    "loss_total", "loss_pred_wr", "loss_pred_rmse", "loss_pred_mse",
    "loss_noarb_bfly", "loss_noarb_cal", "loss_param_l2",
    "lr", "batches", "wall_time_s",
    "hit_a_lo","hit_a_hi","hit_b_lo","hit_b_hi","hit_rho_lo","hit_rho_hi","hit_m_lo","hit_m_hi","hit_sigma_lo","hit_sigma_hi",
]
for c in num_like_cols:
    if c in train.columns:
        train[c] = pd.to_numeric(train[c], errors="coerce")

for c in ["a_hat","b_hat","rho_hat","m_hat","sigma_hat","surf_iv_rmse","dte"]:
    if c in params.columns:
        params[c] = pd.to_numeric(params[c], errors="coerce")

for c in ["bfly_violation_rate","calendar_violation_rate","bfly_penalty_mean","calendar_penalty_mean","dte"]:
    if c in noarb.columns:
        noarb[c] = pd.to_numeric(noarb[c], errors="coerce")

# --- prints ---
print("TRAIN rows (tail):\n", train.tail(), "\n")
print("PARAMS head:\n", params.head(), "\n")
print("NO-ARB head:\n", noarb.head(), "\n")

# --- param sanity (handles missing columns gracefully) ---
def safe_mean(series, default=np.nan):
    return float(series.mean()) if len(series) else default

has = params.columns
a_ok     = safe_mean(params["a_hat"]    >= 0) if "a_hat"    in has else np.nan
b_ok     = safe_mean(params["b_hat"]    >  0) if "b_hat"    in has else np.nan
rho_ok   = safe_mean(params["rho_hat"].abs() < 1) if "rho_hat" in has else np.nan
sigma_ok = safe_mean(params["sigma_hat"]>  0) if "sigma_hat" in has else np.nan
print("Param ranges:", a_ok, b_ok, rho_ok, sigma_ok)

if "surf_iv_rmse" in params.columns:
    print("RMSE quantiles:\n", params["surf_iv_rmse"].quantile([0, .25, .5, .75, 1]))
else:
    print("RMSE quantiles:\n  <surf_iv_rmse not found>")

if "bfly_violation_rate" in noarb.columns:
    print("Butterfly viol rate mean:", noarb["bfly_violation_rate"].mean())
else:
    print("Butterfly viol rate mean: <column not found>")

if "calendar_violation_rate" in noarb.columns:
    print("Calendar viol rate mean:", noarb["calendar_violation_rate"].mean())
else:
    print("Calendar viol rate mean: <column not found>")

# --- expiries per date ---
date_col = "date" if "date" in params.columns else None
exp_col = "expiry" if "expiry" in params.columns else None
if date_col and exp_col:
    print("Expiries per date (top 5):\n",
          params.groupby(date_col)[exp_col].nunique().sort_values(ascending=False).head())
else:
    print("Expiries per date: <date/expiry columns not found>")

# --- optional: show last VAL row for quick reference ---
if {"epoch","split","loss_total"}.issubset(train.columns):
    last_val = train[train["split"]=="val"].tail(1)
    if not last_val.empty:
        print("\nLast VAL row:\n", last_val.to_string(index=False))
