#!/usr/bin/env bash
# Run a TrainTicket scheduler batch with observability port-forwards owned by
# the same supervisor.  This makes a long collection resumable/monitorable
# without requiring a second terminal.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PROJECT_DIR/config/experiment_config_trainticket_v2.yaml}"
shift || true
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/artifacts/trainticket_collection_v2_20260928}"
mkdir -p "$LOG_DIR"

# Fail fast when the caller has no cluster connection; otherwise the Python
# health loop would spend five minutes treating an empty kubectl response as
# an unhealthy application and hide the real authorization/network error.
if ! kubectl get pods -n trainticket -l app --no-headers >/dev/null 2>&1; then
    echo "TrainTicket preflight failed: kubectl cannot read namespace trainticket" >&2
    exit 2
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

forward() {
    local service="$1" port="$2" log="$3"
    kubectl port-forward -n observability "svc/${service}" "${port}:${port}" \
        >"$LOG_DIR/${log}.port-forward.log" 2>&1 &
    PIDS+=("$!")
}

forward prom-kube-prometheus-stack-prometheus 9090 prometheus
forward loki 3100 loki
forward tempo 3200 tempo
sleep 3

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
set +e
"$PYTHON" -m runner.scheduler --config "$CONFIG" "$@" \
    2>&1 | tee "$LOG_DIR/scheduler.log"
status=${PIPESTATUS[0]}
exit "$status"
