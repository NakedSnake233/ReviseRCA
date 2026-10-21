#!/usr/bin/env bash
# Materialize and evaluate the complete TrainTicket release after collection.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${1:-$PROJECT_DIR/data/experiments_trainticket_v2_20260928}"
VIEW_ROOT="${2:-$PROJECT_DIR/data/experiments_trainticket_release_20260928}"
PYTHON="${PYTHON:-python3}"
SUMMARY_FILE="${SUMMARY_FILE:-$RAW_ROOT/batch_summary.json}"
DATASET_ID="dyrca-v1-trainticket"
SELECTION="$PROJECT_DIR/artifacts/trainticket_regression_selection_20260928.json"
HARNESS="$PROJECT_DIR/artifacts/harness_temporal_trainticket_20260928"
BASELINES="$PROJECT_DIR/artifacts/dynamic_baselines_temporal_trainticket_20260928"
SPLIT_POLICY="${SPLIT_POLICY:-service_group}"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_DIR/scripts/audit_trainticket_collection.py" \
  --root "$RAW_ROOT" --summary "$SUMMARY_FILE" \
  --out "$PROJECT_DIR/artifacts/trainticket_collection_audit_final_20260928.json"
AUDIT_JSON="$PROJECT_DIR/artifacts/trainticket_collection_audit_final_20260928.json"
read -r OBSERVED COMPLETE_TELEMETRY EXPECTED <<EOF
$($PYTHON - "$AUDIT_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["observed_total"], d["cases_with_complete_telemetry"], d["expected_total"])
PY
)
EOF
if [ "$OBSERVED" -ne "$EXPECTED" ] || [ "$COMPLETE_TELEMETRY" -ne "$EXPECTED" ]; then
  echo "TrainTicket collection is incomplete: observed=$OBSERVED telemetry_complete=$COMPLETE_TELEMETRY expected=$EXPECTED" >&2
  exit 1
fi
"$PYTHON" "$PROJECT_DIR/scripts/build_complete_case_view.py" \
  --root "$RAW_ROOT" --out "$VIEW_ROOT" --namespace trainticket --summary "$SUMMARY_FILE"

count=$(find "$VIEW_ROOT" -mindepth 1 -maxdepth 1 -type l | wc -l)
if [ "$count" -lt 20 ]; then
  echo "too few complete TrainTicket cases: $count" >&2
  exit 1
fi

"$PYTHON" "$PROJECT_DIR/scripts/make_regression_selection.py" \
  --root "$VIEW_ROOT" --out "$SELECTION" --namespace trainticket

"$PYTHON" -m dyrca_rca.experiments.optimize_harness \
  --data-root "$VIEW_ROOT" --out-dir "$HARNESS" --limit "$count" \
  --max-rows 2000 --seed 42 --planner rule --system trainticket \
  --dataset-id "$DATASET_ID" --tasks t1 t2 t3 t4 \
  --regression-selection "$SELECTION" --split-policy "$SPLIT_POLICY"

"$PYTHON" -m dyrca_rca.experiments.run_dynamic_baselines \
  --data-root "$VIEW_ROOT" --release-root "$HARNESS" \
  --out-dir "$BASELINES" --limit "$count" --seed 42 --planner rule \
  --system trainticket --dataset-id "$DATASET_ID"

echo "TrainTicket release complete: $BASELINES"
