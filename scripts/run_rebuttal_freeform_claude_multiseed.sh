#!/usr/bin/env bash
# Free-form LLM diagnosis ablation with Claude (claude-3-5-haiku) across 5 seeds.
set -o pipefail
cd /home/user/dyrca
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate dyrca
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

INDEX="artifacts/dygraph_rca_full/index.csv"
OUT="artifacts/rebuttal_claude/freeform"
BASE="https://subapi.buaanlsde.cn/v1"
KEY="sk-5f2295785f05a6a3e76c793f6350fd3990dd5f33007897b9b5286fbe33d9c664"
MODEL="claude-3-5-haiku-20241022"
SEEDS=(42 123 456 789 2024)
mkdir -p "$OUT"

for seed in "${SEEDS[@]}"; do
  run_dir="$OUT/test_s${seed}"
  if [ -f "$run_dir/metrics_overall.csv" ]; then echo "[skip] seed=$seed"; continue; fi
  echo "[run] free-form(claude) seed=$seed @ $(date +%H:%M:%S)"
  python -u scripts/run_rebuttal_freeform_llm.py \
    --index "$INDEX" --data-root data/experiments \
    --out-dir "$run_dir" --split-seed "$seed" \
    --base-url "$BASE" --api-key "$KEY" --model "$MODEL" 2>&1 | tail -3
done
echo "=== free-form(claude) multi-seed complete @ $(date +%H:%M:%S) ==="