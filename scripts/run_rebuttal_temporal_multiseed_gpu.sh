#!/usr/bin/env bash
# Run EvolveGCN-A baseline across seeds on GPU (small model, GPU idle).
set -o pipefail
cd /home/user/dyrca
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate dyrca
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="0"

INDEX="artifacts/dygraph_rca_full/index.csv"
OUT="artifacts/rebuttal_temporal"
SEEDS=(42 123 456 789 2024)
mkdir -p "$OUT"

for seed in "${SEEDS[@]}"; do
  run_dir="$OUT/evolvegcn_a_s${seed}"
  if [ -f "$run_dir/metrics_overall.csv" ]; then
    echo "[skip] seed=$seed (exists)"; continue
  fi
  echo "[run] EvolveGCN-A seed=$seed @ $(date +%H:%M:%S)"
  python -u scripts/run_rebuttal_temporal_gnn.py \
    --index "$INDEX" --data-root data/experiments \
    --seed "$seed" --split random --arch evolvegcn_a \
    --out-dir "$run_dir" 2>&1 | tail -2
done
echo "=== temporal multi-seed (GPU) complete @ $(date +%H:%M:%S) ==="