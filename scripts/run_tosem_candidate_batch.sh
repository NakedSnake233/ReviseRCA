#!/usr/bin/env bash
# Run one frozen TOSEM candidate collection with isolated observability ports.
# The scheduler itself owns the experiment records; this wrapper only owns
# port-forwards and makes the long batch resumable through --resume-summary.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: $0 CONFIG [scheduler args ...]}"
shift
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

case "$CONFIG_ABS" in
  *trainticket*) NAMESPACE="trainticket"; BASE_PORT="${BASE_PORT:-9091}" ;;
  *)             NAMESPACE="boutique"; BASE_PORT="${BASE_PORT:-9090}" ;;
esac
PROM_PORT="$BASE_PORT"
PORT_OFFSET="$((BASE_PORT - 9090))"
LOKI_PORT="$((3100 + PORT_OFFSET))"
TEMPO_PORT="$((3200 + PORT_OFFSET))"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/artifacts/tosem_candidate_supervisor_$(basename "$CONFIG_ABS" .yaml)}"
mkdir -p "$LOG_DIR"

if ! kubectl get pods -n "$NAMESPACE" -l app --no-headers >/dev/null 2>&1; then
  echo "preflight failed: cannot read namespace $NAMESPACE" >&2
  exit 2
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  rm -f "${RUN_CONFIG:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

forward() {
  local svc="$1" local_port="$2" remote_port="$3" label="$4"
  kubectl port-forward -n observability "svc/${svc}" "${local_port}:${remote_port}" \
    >"$LOG_DIR/${label}.port-forward.log" 2>&1 &
  PIDS+=("$!")
}
forward prom-kube-prometheus-stack-prometheus "$PROM_PORT" 9090 prometheus
forward loki "$LOKI_PORT" 3100 loki
forward tempo "$TEMPO_PORT" 3200 tempo
sleep 3

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PROM_URL="http://127.0.0.1:${PROM_PORT}"
export LOKI_URL="http://127.0.0.1:${LOKI_PORT}"
export TEMPO_URL="http://127.0.0.1:${TEMPO_PORT}"
RUN_CONFIG="$(mktemp "${TMPDIR:-/tmp}/tosem-config.XXXXXX.yaml")"
"$PYTHON" - "$CONFIG_ABS" "$RUN_CONFIG" "$PROM_URL" "$LOKI_URL" "$TEMPO_URL" <<'PY'
import sys, yaml
source, dest, prom, loki, tempo = sys.argv[1:]
with open(source, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
config.setdefault("observability", {}).update({"prom_url": prom, "loki_url": loki, "tempo_url": tempo})
with open(dest, "w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
PY
echo "namespace=$NAMESPACE prom=$PROM_URL loki=$LOKI_URL tempo=$TEMPO_URL" | tee "$LOG_DIR/ports.txt"
set +e
"$PYTHON" -m runner.scheduler --config "$RUN_CONFIG" "$@" 2>&1 | tee "$LOG_DIR/scheduler.log"
status=${PIPESTATUS[0]}
exit "$status"
