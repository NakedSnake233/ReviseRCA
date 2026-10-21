#!/usr/bin/env bash
# Complete the post-collection TOSEM pipeline for one frozen application.
# The script refuses to score an active/incomplete batch and never mutates raw
# experiment directories; all outputs are written below artifacts/.
set -euo pipefail

APP="${1:?usage: $0 boutique|trainticket [limit]}"
LIMIT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$APP" in
  boutique) DATA="$ROOT/data/experiments_tosem_candidate_boutique_20260930" ;;
  trainticket) DATA="$ROOT/data/experiments_tosem_candidate_trainticket_20260930" ;;
  *) echo "unknown application: $APP" >&2; exit 2 ;;
esac
AUDIT="$ROOT/artifacts/verified_revision_v1/paper_backfill_live_20260930/$APP"
RELEASE="$ROOT/artifacts/verified_revision_v1/tosem_release_20260930/$APP"
RANKING="$ROOT/artifacts/verified_revision_v1/ranking_20260930/$APP"
ABLATION="$ROOT/artifacts/verified_revision_v1/ablation_20260930/$APP"
LATENCY="$ROOT/artifacts/verified_revision_v1/latency_20260930/$APP"
LUNA="$ROOT/artifacts/verified_revision_v1/luna_agent_20260930/$APP"
DOWNSTREAM="$ROOT/artifacts/verified_revision_v1/downstream_20260930/$APP"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$DATA/batch_summary.json" ]]; then
  "$PYTHON" - "$DATA/batch_summary.json" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
if data.get("pending") or data.get("failed"):
    raise SystemExit("formal batch is not complete or contains failures")
PY
else
  echo "missing batch summary: $DATA" >&2
  exit 3
fi

mkdir -p "$AUDIT"
"$PYTHON" -m dyrca_rca.experiments.run_tosem_backfill \
  --data-root "$DATA" --out-dir "$AUDIT" --seed 42 --planner rule

PREP_ARGS=(--data-root "$DATA" --audit "$AUDIT/collection_audit.jsonl" \
  --out-dir "$RELEASE" --seed 42 --planner rule --system "$APP" \
  --dataset-id "tosem-20260930-$APP" --max-rows 10000)
if [[ -n "$LIMIT" ]]; then PREP_ARGS+=(--limit "$LIMIT"); fi
"$PYTHON" -m dyrca_rca.experiments.prepare_tosem_release "${PREP_ARGS[@]}"

RUN_LIMIT="$($PYTHON - "$RELEASE/candidate_release.json" "$LIMIT" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(sys.argv[2]) if sys.argv[2] else int(config["verified_cases"]))
PY
)"

"$PYTHON" -m dyrca_rca.experiments.optimize_harness \
  --data-root "$RELEASE/verified_data" --out-dir "$RELEASE" \
  --limit "$RUN_LIMIT" --max-rows 10000 --seed 42 --planner rule \
  --split-policy config_group --dataset-id "tosem-20260930-$APP" \
  --system "$APP" --tasks t1 t2 t3 t4 \
  --old-checkpoint "$ROOT/artifacts/interactive_route_training/model.pt" \
  --regression-selection "$RELEASE/regression_selection.json"

mkdir -p "$RANKING"
"$PYTHON" -m dyrca_rca.experiments.run_dynamic_baselines \
  --data-root "$RELEASE/verified_data" --release-root "$RELEASE" \
  --out-dir "$RANKING" --limit "$RUN_LIMIT" --seed 42 --planner rule \
  --system "$APP" --dataset-id "tosem-20260930-$APP" \
  --controls-root "$DATA" --controls-audit "$AUDIT/collection_audit.jsonl"

mkdir -p "$ABLATION"
"$PYTHON" -m dyrca_rca.experiments.run_tosem_ablation \
  --data-root "$RELEASE/verified_data" --release-root "$RELEASE" \
  --out-dir "$ABLATION" --seed 42 --planner rule

mkdir -p "$LATENCY"
"$PYTHON" -m dyrca_rca.experiments.run_tosem_latency \
  --data-root "$RELEASE/verified_data" --release-root "$RELEASE" \
  --out-dir "$LATENCY" --system "$APP" \
  --dataset-id "tosem-20260930-$APP" --limit 10 --max-rows 10000 \
  --warmup 5 --repeats 20 --seed 42 --planner rule

AGENT_ARGS=()
if [[ -n "${DYRCA_LLM_API_KEY:-}" ]]; then
  bash "$ROOT/scripts/run_tosem_luna_agent.sh" "$APP" "${LUNA_LIMIT:-36}"
  AGENT_ARGS+=(--agent-dir "$LUNA")
else
  echo "DYRCA_LLM_API_KEY is not set; leaving E8 Luna artifact pending (no credential is read from disk)." >&2
fi

mkdir -p "$DOWNSTREAM"
"$PYTHON" -m dyrca_rca.experiments.run_tosem_downstream \
  --data-root "$DATA" --audit "$AUDIT/collection_audit.jsonl" \
  --ranking-dir "$RANKING" --ablation-dir "$ABLATION" \
  --latency-dir "$LATENCY" \
  --out-dir "$DOWNSTREAM" --seed 42 --planner rule "${AGENT_ARGS[@]}"
echo "TOSEM pipeline complete for $APP; inspect $DOWNSTREAM"
