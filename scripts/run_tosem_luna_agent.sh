#!/usr/bin/env bash
# Run the H0/H1/H2 simulated SRE condition on a verified TOSEM release.
# The caller supplies DYRCA_LLM_API_KEY in the environment; this script never
# writes or echoes credentials.
set -euo pipefail

APP="${1:?usage: $0 boutique|trainticket [limit]}"
LIMIT="${2:-36}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE="$ROOT/artifacts/verified_revision_v1/tosem_release_20260930/$APP"
OUT="$ROOT/artifacts/verified_revision_v1/luna_agent_20260930/$APP"
DATA="$RELEASE/verified_data"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DYRCA_LLM_BASE_URL="${DYRCA_LLM_BASE_URL:-https://opencode.ai/zen/go/v1}"
export DYRCA_LLM_MODEL="${DYRCA_LLM_MODEL:-gpt-5.6-luna}"
if [[ -z "${DYRCA_LLM_API_KEY:-}" ]]; then
  echo "DYRCA_LLM_API_KEY must be supplied in the environment; no credential is read from disk" >&2
  exit 2
fi
if [[ ! -f "$RELEASE/temporal_v3.pt" ]]; then
  echo "missing temporal-v3 checkpoint; run the formal candidate pipeline first" >&2
  exit 3
fi

"$PYTHON" - "$RELEASE/protocol.json" "$RELEASE/temporal_v3.pt" "$DATA" <<'PY'
import json, sys
path, checkpoint, data = sys.argv[1:]
obj = json.load(open(path, encoding="utf-8"))
obj.update({"checkpoint": checkpoint, "data_root": data, "agent_model": "gpt-5.6-luna",
            "human_role": "simulated_sre_agent", "human_participants": 0})
json.dump(obj, open(path, "w", encoding="utf-8"), sort_keys=True, indent=2)
open(path, "a", encoding="utf-8").write("\n")
PY

rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"
"$PYTHON" -m dyrca_rca.experiments.simulate_sre_agent \
  --data-root "$DATA" --pilot-dir "$RELEASE" --out-dir "$OUT" \
  --system "$APP" --dataset-id "tosem-20260930-$APP" \
  --limit "$LIMIT" --max-rows 10000 --seed 42 --planner llm \
  --oracle-prob 0.5 --defect-prob 0.5 --allow-external-evidence
echo "Luna simulated-agent artifact: $OUT"
