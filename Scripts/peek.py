import pandas as pd, numpy as np, torch, os
assets = ["SPY","QQQ","IWM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ","EFA","EEM"]
outdir = "outputs/ito_equities"

pred = pd.read_csv(f"{outdir}/val_predictions.csv")
targ = pd.read_csv(f"{outdir}/val_targets.csv")

# 1) Shapes/columns
print("pred shape:", pred.shape, "targ shape:", targ.shape)
print("columns ok:", list(pred.columns) == assets == list(targ.columns))

# 2) NaNs / infs
print("pred NaNs:", pred.isna().sum().sum(), "targ NaNs:", targ.isna().sum().sum())
print("pred finite:", np.isfinite(pred.to_numpy()).all(), "targ finite:", np.isfinite(targ.to_numpy()).all())

# 3) Basic metrics
mse = ((pred - targ)**2).mean().mean()
mae = (pred - targ).abs().mean().mean()
print("Val MSE (CSV):", float(mse), "MAE:", float(mae))

# 4) Rough scale check (returns should be ~few bps to a few %)
print("pred abs 99th pct:", float(pred.abs().stack().quantile(0.99)))
print("targ abs 99th pct:", float(targ.abs().stack().quantile(0.99)))

# 5) Peek rows
print(pred.head(3))
print(targ.head(3))

# 6) Check the checkpoint has the pieces we expect
ckpt = torch.load(os.path.join(outdir, "best.pt"), map_location="cpu")
keys = ckpt["model_state"].keys()
print("has martingale_head:", any(k.startswith("martingale_head.proj") for k in keys))
print("has input_proj:", any(k.startswith("input_proj.") for k in keys))
print("n_blocks:",
      len([k for k in keys if k.startswith("blocks.0.")]))
