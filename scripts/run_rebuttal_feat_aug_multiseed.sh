#!/usr/bin/env bash
# Feature-augmented baseline (dynamic evidence as features, NO patches) across 5 seeds.
set -o pipefail
cd /home/user/dyrca
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate dyrca
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

INDEX="artifacts/dygraph_rca_full/index.csv"
OUT="artifacts/rebuttal_feat_aug"
SEEDS=(42 123 456 789 2024)
mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  run_dir="$OUT/s${seed}"
  if [ -f "$run_dir/metrics_overall.csv" ]; then echo "[skip] seed=$seed"; continue; fi
  echo "[run] feat_aug seed=$seed @ $(date +%H:%M:%S)"
  python -u scripts/run_rebuttal_feat_aug.py \
    --index "$INDEX" --data-root data/experiments \
    --seed "$seed" --out-dir "$run_dir" 2>&1 | tail -2
done
echo "=== feat_aug multi-seed complete @ $(date +%H:%M:%S) ==="