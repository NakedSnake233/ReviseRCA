#!/usr/bin/env bash
# Run free-form LLM diagnosis ablation across remaining seeds (gpt-5.4-mini).
set -o pipefail
cd /home/user/dyrca
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate dyrca
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

INDEX="artifacts/dygraph_rca_full/index.csv"
OUT="artifacts/rebuttal_freeform"
BASE="https://subapi.buaanlsde.cn/v1"
KEY="sk-f7c7646bbdfe3c0c62bf47af70e2d59e6421581456a715b799a097e0cc408870"
MODEL="gpt-5.4-mini"
SEEDS=(123 456 789 2024)
mkdir -p "$OUT"

for seed in "${SEEDS[@]}"; do
  run_dir="$OUT/test_s${seed}"
  if [ -f "$run_dir/metrics_overall.csv" ]; then echo "[skip] seed=$seed"; continue; fi
  echo "[run] free-form seed=$seed @ $(date +%H:%M:%S)"
  python -u scripts/run_rebuttal_freeform_llm.py \
    --index "$INDEX" --data-root data/experiments \
    --out-dir "$run_dir" --split-seed "$seed" \
    --base-url "$BASE" --api-key "$KEY" --model "$MODEL" 2>&1 | tail -3
done
echo "=== free-form multi-seed complete @ $(date +%H:%M:%S) ==="