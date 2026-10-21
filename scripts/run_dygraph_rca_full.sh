#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data/experiments}"
OUT_ROOT="${OUT_ROOT:-artifacts/dygraph_rca_full}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-random}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-6}"
LLM_RERANK_CONCURRENCY="${LLM_RERANK_CONCURRENCY:-6}"

INDEX="$OUT_ROOT/index.csv"
PLANS_DIR="$OUT_ROOT/plans"
LLM_PLANS_DIR="${LLM_PLANS_DIR:-$OUT_ROOT/llm_plans}"
MODEL_DATA="$OUT_ROOT/model_data"
RESULTS_DIR="$OUT_ROOT/results"
REPORT="$OUT_ROOT/report.md"

mkdir -p "$OUT_ROOT" "$PLANS_DIR" "$MODEL_DATA" "$RESULTS_DIR"

"$PYTHON" -m dyrca_rca.data.index \
  --data-root "$DATA_ROOT" \
  --out "$INDEX"

"$PYTHON" -m dyrca_rca.planner.rule_planner \
  --index "$INDEX" \
  --out-dir "$PLANS_DIR"

if [ -n "${DYRCA_LLM_API_KEY:-}" ] && [ -n "${DYRCA_LLM_BASE_URL:-}" ]; then
  "$PYTHON" -m dyrca_rca.planner.llm_planner \
    --index "$INDEX" \
    --out-dir "$LLM_PLANS_DIR" \
    --cache-dir "$LLM_PLANS_DIR" \
    --fallback rule \
    --concurrency "$LLM_CONCURRENCY"
fi

variants=(static_pre naive_during rule_node rule_edge rule_full)
if [ -d "$LLM_PLANS_DIR" ] && find "$LLM_PLANS_DIR" -name '*.json' -print -quit | grep -q .; then
  variants+=(llm_full)
fi

for variant in "${variants[@]}"; do
  "$PYTHON" -m dyrca_rca.dataset.materialize \
    --index "$INDEX" \
    --variant "$variant" \
    --out-dir "$MODEL_DATA" \
    --llm-plan-dir "$LLM_PLANS_DIR"
done

for variant in "${variants[@]}"; do
  for model in weighted_graphsage edge_aware_gat; do
    run_dir="$RESULTS_DIR/${variant}_${model}"
    "$PYTHON" -m dyrca_rca.experiments.run_experiment \
      --data-root "$DATA_ROOT" \
      --index "$INDEX" \
      --graph-variant "$variant" \
      --model "$model" \
      --split "$SPLIT" \
      --seed "$SEED" \
      --llm-plan-dir "$LLM_PLANS_DIR" \
      --out-dir "$run_dir"

    if [ -f "$run_dir/predictions.csv" ]; then
      if [ -n "${DYRCA_LLM_API_KEY:-}" ] && [ -n "${DYRCA_LLM_BASE_URL:-}" ]; then
        "$PYTHON" -m dyrca_rca.rerank.llm_reranker \
          --predictions "$run_dir/predictions.csv" \
          --data-root "$DATA_ROOT" \
          --graphs-root "$MODEL_DATA/$variant" \
          --out-dir "$run_dir/rerank_llm" \
          --top-k 5 \
          --fallback rule \
          --concurrency "$LLM_RERANK_CONCURRENCY"
      else
        "$PYTHON" -m dyrca_rca.rerank.rule_reranker \
          --predictions "$run_dir/predictions.csv" \
          --data-root "$DATA_ROOT" \
          --graphs-root "$MODEL_DATA/$variant" \
          --out-dir "$run_dir/rerank" \
          --top-k 5
      fi
    fi
  done
done

GRAPH_UPDATE_EVAL_DIR="$RESULTS_DIR/graph_update_eval"
GRAPH_UPDATE_PLANS_DIR="$PLANS_DIR"
GRAPH_UPDATE_GRAPHS_DIR="$MODEL_DATA/rule_full"
if printf '%s\n' "${variants[@]}" | grep -qx "llm_full"; then
  GRAPH_UPDATE_EVAL_DIR="$RESULTS_DIR/graph_update_eval_llm"
  GRAPH_UPDATE_PLANS_DIR="$LLM_PLANS_DIR"
  GRAPH_UPDATE_GRAPHS_DIR="$MODEL_DATA/llm_full"
fi

"$PYTHON" -m dyrca_rca.experiments.evaluate_graph_updates \
  --index "$INDEX" \
  --plans-dir "$GRAPH_UPDATE_PLANS_DIR" \
  --graphs-dir "$GRAPH_UPDATE_GRAPHS_DIR" \
  --out-dir "$GRAPH_UPDATE_EVAL_DIR"

"$PYTHON" -m dyrca_rca.reporting.make_report \
  --index "$INDEX" \
  --results-root "$RESULTS_DIR" \
  --graph-update-dir "$GRAPH_UPDATE_EVAL_DIR" \
  --baselines-root "artifacts/baselines" \
  --out "$REPORT" \
  --title "Dynamic Graph RCA Full Report"

echo "Full run complete: $REPORT"
