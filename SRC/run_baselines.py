# scripts/run_baselines.py
import argparse
from pathlib import Path
import pandas as pd

from itoformer.training.preprocessing import to_returns, align_and_dropna
from itoformer.baselines.arima import ARIMABaseline
from itoformer.baselines.har_rv import HAR_RV
from itoformer.baselines.garch import GARCH11
from itoformer.baselines.kalman import KalmanBaseline

# ---- tiny helpers we used to import -----------------------------------------
def simple_walk_forward_index(index: pd.Index, train_window: int, h: int, step: int):
    """
    Yield (train_index_slice, test_index_slice) as pandas Index objects for
    a rolling/expanding walk-forward split with fixed training window length.
    """
    n = len(index)
    for t in range(train_window, n - h + 1, step):
        train_idx = index[t - train_window : t]
        test_idx  = index[t : t + h]
        yield (train_idx, test_idx)

def slice_by_index(df: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    return df.loc[idx]

# ---- robust CSV loader for ETF/stock price columns --------------------------
def load_asset_series(asset: str, root: Path) -> pd.DataFrame:
    """
    Robust loader for ETFs/stocks CSVs that may use different date and price column names,
    and may contain stray header/text rows (e.g., 'SPY') in the price column.
    Returns a DataFrame indexed by datetime with a single numeric 'px' column.
    """
    p = root / f"{asset}.csv"
    df = pd.read_csv(p)

    # 1) Find/parse date column
    date_candidates = ["date", "Date", "DATE", "timestamp", "Timestamp"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        # try first column if it parses as dates
        first = df.columns[0]
        try:
            pd.to_datetime(df[first])
            date_col = first
        except Exception:
            raise ValueError(f"No parseable date column in {p}; cols={list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # 2) Choose a price column
    preferred = [
        "adj_close", "Adj Close", "adjclose",
        "close", "Close",
        "PX_LAST", "px_last",
        "price", "Price",
    ]

    def numeric_score(s: pd.Series) -> int:
        # count numeric (non-NaN) after coercion; strips commas
        coerced = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
        return int(coerced.notna().sum())

    price_col = None
    for c in preferred:
        if c in df.columns and numeric_score(df[c]) > 5:
            price_col = c
            break

    if price_col is None:
        # fallback: pick the most numeric-looking column
        scores = sorted(
            ((c, numeric_score(df[c])) for c in df.columns),
            key=lambda t: t[1],
            reverse=True,
        )
        if scores and scores[0][1] > 5:
            price_col = scores[0][0]

    if price_col is None:
        raise ValueError(f"No usable price column in {p}; cols={list(df.columns)}")

    # 3) Coerce to float and drop junk rows
    px = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce").dropna()
    if px.empty:
        raise ValueError(f"Price series is empty after cleaning in {p}")

    return px.to_frame("px")

# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/etfs")
    ap.add_argument("--outputs_root", type=str, default="outputs/baselines_equities")
    ap.add_argument("--assets", type=str, nargs="+",
                    default=["SPY","QQQ","IWM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ","EFA","EEM"])
    ap.add_argument("--train_window", type=int, default=252)
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.outputs_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for asset in args.assets:
        df = load_asset_series(asset, data_root)
        df["ret"] = to_returns(df["px"], log=True)
        df = align_and_dropna(df)

        # Prepare rolling indices
        wf = list(simple_walk_forward_index(df.index, args.train_window, args.h, args.step))
        if not wf:
            print(f"[{asset}] Not enough data for train_window={args.train_window}")
            continue

        # --- ARIMA on returns
        arima = ARIMABaseline(order=(1,0,1))
        preds = []
        for train_idx, test_idx in wf:
            train = slice_by_index(df[["ret"]].dropna(), train_idx)
            if len(train) < 10:
                continue
            arima.fit(train, target_col="ret")
            fc = arima.predict(args.h)
            fc.index = test_idx
            preds.append(fc)
        if preds:
            out = pd.concat(preds).rename(columns={"yhat": "pred"})
            out["asset"] = asset
            out["model"] = "ARIMA(1,0,1)"
            out.to_csv(out_root / f"{asset}_arima_h{args.h}.csv", index=True)

        # --- GARCH on returns (variance forecast; mean return ≈ 0 baseline)
        garch = GARCH11(dist="t", mean="Zero")
        preds = []
        for train_idx, test_idx in wf:
            train = slice_by_index(df[["ret"]].dropna(), train_idx)
            if len(train) < 50:
                continue
            garch.fit(train, target_col="ret")
            fc = garch.predict(args.h, variance=True)
            tmp = pd.DataFrame({"pred": 0.0}, index=test_idx)
            if isinstance(fc, pd.DataFrame) and "var" in fc.columns:
                tmp["var_pred"] = fc["var"].values[: len(test_idx)]
            preds.append(tmp)
        if preds:
            out = pd.concat(preds)
            out["asset"] = asset
            out["model"] = "GARCH11-t"
            out.to_csv(out_root / f"{asset}_garch_h{args.h}.csv", index=True)

        # --- HAR-RV (toy proxy uses |ret|; replace with true RV/IV when available)
        har = HAR_RV()
        preds = []
        rv_df = df["ret"].abs().to_frame("rv")
        wf_rv = list(simple_walk_forward_index(rv_df.index, args.train_window, args.h, args.step))
        for train_idx, test_idx in wf_rv:
            train = rv_df.loc[train_idx]
            if len(train) < 50:
                continue
            har.fit(train, target_col="rv")
            fc = har.predict(args.h)              # returns yhat column
            fc.index = test_idx
            preds.append(fc.rename(columns={"yhat": "pred"}))
        if preds:
            out = pd.concat(preds)
            out["asset"] = asset
            out["model"] = "HAR-RV"
            out.to_csv(out_root / f"{asset}_har_h{args.h}.csv", index=True)

        # --- Kalman local level on price → convert to return forecast (simple baseline)
        kal = KalmanBaseline(model="local_level")
        preds = []
        for train_idx, test_idx in wf:
            train = slice_by_index(df[["px"]].dropna(), train_idx)
            if len(train) < 50:
                continue
            kal.fit(train, target_col="px")
            _ = kal.predict(args.h)               # we’ll keep pred=0.0 for return mean baseline
            preds.append(pd.DataFrame({"pred": 0.0}, index=test_idx))
        if preds:
            out = pd.concat(preds)
            out["asset"] = asset
            out["model"] = "KalmanLocalLevel"
            out.to_csv(out_root / f"{asset}_kalman_h{args.h}.csv", index=True)

        print(f"[{asset}] done.")

if __name__ == "__main__":
    main()
