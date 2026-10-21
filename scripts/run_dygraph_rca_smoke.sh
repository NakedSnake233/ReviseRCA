#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data/experiments}"
OUT_ROOT="${OUT_ROOT:-artifacts/dygraph_rca_smoke}"
LIMIT="${LIMIT:-10}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-random}"

INDEX="$OUT_ROOT/index.csv"
PLANS_DIR="$OUT_ROOT/plans"
MODEL_DATA="$OUT_ROOT/model_data"
RESULTS_DIR="$OUT_ROOT/results"
REPORT="$OUT_ROOT/report.md"

mkdir -p "$OUT_ROOT" "$PLANS_DIR" "$MODEL_DATA" "$RESULTS_DIR"

"$PYTHON" -m dyrca_rca.data.index \
  --data-root "$DATA_ROOT" \
  --out "$INDEX" \
  --limit "$LIMIT"

"$PYTHON" -m dyrca_rca.planner.rule_planner \
  --index "$INDEX" \
  --out-dir "$PLANS_DIR" \
  --limit "$LIMIT"

for variant in static_pre rule_full; do
  "$PYTHON" -m dyrca_rca.dataset.materialize \
    --index "$INDEX" \
    --variant "$variant" \
    --out-dir "$MODEL_DATA" \
    --limit "$LIMIT"

  "$PYTHON" -m dyrca_rca.experiments.run_experiment \
    --data-root "$DATA_ROOT" \
    --index "$INDEX" \
    --graph-variant "$variant" \
    --model weighted_graphsage \
    --split "$SPLIT" \
    --seed "$SEED" \
    --limit "$LIMIT" \
    --out-dir "$RESULTS_DIR/${variant}_weighted_graphsage"
done

"$PYTHON" -m dyrca_rca.experiments.evaluate_graph_updates \
  --index "$INDEX" \
  --plans-dir "$PLANS_DIR" \
  --graphs-dir "$MODEL_DATA/rule_full" \
  --out-dir "$RESULTS_DIR/graph_update_eval" \
  --limit "$LIMIT"

if [ -f "$RESULTS_DIR/rule_full_weighted_graphsage/predictions.csv" ]; then
  "$PYTHON" -m dyrca_rca.rerank.rule_reranker \
    --predictions "$RESULTS_DIR/rule_full_weighted_graphsage/predictions.csv" \
    --data-root "$DATA_ROOT" \
    --graphs-root "$MODEL_DATA/rule_full" \
    --out-dir "$RESULTS_DIR/rule_full_weighted_graphsage/rerank" \
    --top-k 5 \
    --limit "$LIMIT"
fi

"$PYTHON" -m dyrca_rca.reporting.make_report \
  --index "$INDEX" \
  --results-root "$RESULTS_DIR" \
  --graph-update-dir "$RESULTS_DIR/graph_update_eval" \
  --out "$REPORT" \
  --title "Dynamic Graph RCA Smoke Report"

echo "Smoke run complete: $REPORT"
