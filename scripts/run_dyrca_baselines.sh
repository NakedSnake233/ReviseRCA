#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/dyrca/bin/python}"
INDEX="${INDEX:-artifacts/dygraph_rca_full/index.csv}"
DATA_ROOT="${DATA_ROOT:-data/experiments}"
OUT_DIR="${OUT_DIR:-artifacts/baselines/dyrca_full}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ADAPTER_MODE="${ADAPTER_MODE:-strict}"

PYTHONPATH="baselines:src:." "$PYTHON" -m dyrca_baselines.run_baselines \
  --index "$INDEX" \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --adapter-mode "$ADAPTER_MODE" \
  "$@"
