# scripts/evaluate.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acov_kernel
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm


def load_method_panel(path: Path) -> pd.DataFrame:
    """
    Expect per-asset per-horizon outputs in tidy format, e.g.
    columns: ['date','asset','horizon','y_true','y_pred','strategy_ret']
    One file per method or a directory of files. Merge into panel.
    """
    if path.is_dir():
        dfs = []
        for f in sorted(path.glob("*.csv")):
            df = pd.read_csv(f, parse_dates=["date"])
            df["method"] = f.stem
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path, parse_dates=["date"])
        if "method" not in df.columns:
            df["method"] = path.stem
        return df


def sharpe(returns: np.ndarray, eps: float = 1e-12) -> float:
    mu = np.nanmean(returns)
    sd = np.nanstd(returns, ddof=1)
    return mu / (sd + eps)


def sortino(returns: np.ndarray, eps: float = 1e-12) -> float:
    mu = np.nanmean(returns)
    downside = returns.copy()
    downside[downside > 0] = 0.0
    dd = np.sqrt(np.nanmean(downside**2))
    return mu / (dd + eps)


def max_drawdown(returns: np.ndarray) -> float:
    # cumulative to equity curve
    curve = np.cumprod(1.0 + np.nan_to_num(returns))
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    return np.min(dd)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> Tuple[float, float]:
    """
    Diebold-Mariano test for predictive accuracy difference:
    returns (DM_stat, p_value)
    """
    d = loss_a - loss_b
    T = d.shape[0]
    d_mean = d.mean()
    # Newey-West estimator of long-run variance with lag = h-1
    # Use statsmodels HAC on a constant regression
    X = np.ones((T, 1))
    model = sm.OLS(d, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(1, h-1)})
    se = np.sqrt(model.cov_params()[0, 0])
    stat = d_mean / (se + 1e-12)
    # two-sided p-value from normal approx
    p = 2 * (1 - sm.distributions.norm.cdf(abs(stat)))
    return stat, p


def hac_se(series: np.ndarray, lags: int = 5) -> float:
    """
    HAC (Newey-West) SE of the mean of a series.
    """
    X = np.ones((series.shape[0], 1))
    model = sm.OLS(series, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return np.sqrt(model.cov_params()[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=str, required=True,
                    help="Folder with baseline CSVs (per-method or merged).")
    ap.add_argument("--itoformer_dir", type=str, required=True,
                    help="Folder with ItoFormer CSVs.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output CSV for summary metrics.")
    args = ap.parse_args()

    base_panel = load_method_panel(Path(args.baseline_dir))
    ito_panel  = load_method_panel(Path(args.itoformer_dir))

    panel = pd.concat([base_panel, ito_panel], ignore_index=True)

    # Compute per-(method, asset, horizon) metrics
    rows = []
    for (method, asset, horizon), g in panel.groupby(["method", "asset", "horizon"]):
        # Forecast accuracy
        if {"y_true", "y_pred"}.issubset(g.columns):
            err = (g["y_pred"] - g["y_true"]).to_numpy()
            mse = np.nanmean(err**2)
            mae = np.nanmean(np.abs(err))
        else:
            mse = np.nan
            mae = np.nan

        # Strategy metrics
        if "strategy_ret" in g.columns:
            r = g["strategy_ret"].to_numpy()
            sr = sharpe(r)
            so = sortino(r)
            dd = max_drawdown(r)
            hac = hac_se(r, lags=5)
        else:
            sr = so = dd = hac = np.nan

        rows.append({
            "method": method,
            "asset": asset,
            "horizon": horizon,
            "MSE": mse,
            "MAE": mae,
            "Sharpe": sr,
            "Sortino": so,
            "MaxDD": dd,
            "HAC_SE_ret_mean": hac,
        })

    summary = pd.DataFrame(rows)

    # Optional: DM tests against a chosen benchmark per (asset, horizon)
    # Here we compare everything to 'PatchTST' if present
    dm_rows = []
    bench = "PatchTST"
    for (asset, horizon), g in panel.groupby(["asset", "horizon"]):
        if bench not in g["method"].unique():
            continue
        g_bench = g[g["method"] == bench]
        if not {"date", "y_true", "y_pred"}.issubset(g_bench.columns):
            continue
        # Merge on date for alignment
        for method in g["method"].unique():
            if method == bench:
                continue
            g_m = g[g["method"] == method]
            if not {"date", "y_true", "y_pred"}.issubset(g_m.columns):
                continue
            mg = pd.merge(
                g_bench[["date", "y_true", "y_pred"]],
                g_m[["date", "y_true", "y_pred"]],
                on="date",
                suffixes=("_bench", "_meth"),
            ).sort_values("date")
            if len(mg) < 20:
                continue
            # squared error loss
            la = (mg["y_pred_meth"] - mg["y_true_meth"])**2
            lb = (mg["y_pred_bench"] - mg["y_true_bench"])**2
            stat, p = dm_test(la.to_numpy(), lb.to_numpy(), h=int(horizon))
            dm_rows.append({
                "asset": asset,
                "horizon": horizon,
                "method": method,
                "benchmark": bench,
                "DM_stat": stat,
                "DM_p": p,
            })

    dm_df = pd.DataFrame(dm_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # write both
    summary.to_csv(out_path, index=False)
    if not dm_df.empty:
        dm_path = out_path.with_name(out_path.stem + "_dm.csv")
        dm_df.to_csv(dm_path, index=False)

    print(f"Saved summary to {out_path}")
    if not dm_df.empty:
        print(f"Saved DM comparisons to {dm_path}")


if __name__ == "__main__":
    main()
