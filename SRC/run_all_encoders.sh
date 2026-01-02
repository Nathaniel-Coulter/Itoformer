#!/bin/bash
set -e  # stop if any command fails

export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "=== Running PatchTST ==="
python scripts/train_equities.py \
  --encoder patchtst --patch-len 16 \
  --epochs 50 --batch-size 64 --lr 5e-4 \
  --outdir outputs/patchtst_equities

echo "=== Running iTransformer ==="
python scripts/train_equities.py \
  --encoder itransformer --time-to-chan 64 \
  --epochs 50 --batch-size 64 --lr 5e-4 \
  --outdir outputs/itransformer_equities

echo "=== Running CrossLite ==="
python scripts/train_equities.py \
  --encoder crosslite --patch-len 8 \
  --epochs 50 --batch-size 64 --lr 5e-4 \
  --outdir outputs/crosslite_equities

echo "=== Running Pointwise ==="
python scripts/train_equities.py \
  --encoder pointwise \
  --epochs 50 --batch-size 64 --lr 5e-4 \
  --outdir outputs/pointwise_equities

echo "=== All runs completed successfully. ==="
