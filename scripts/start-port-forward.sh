#!/bin/bash
# ============================================================
# Start kubectl port-forward for observability APIs
# Run this in a SEPARATE terminal before running experiments.
# The experiment controller (Python) connects via these local ports.
# ============================================================
set -euo pipefail

NAMESPACE="observability"
PIDS=()

cleanup() {
    echo ""
    echo "Stopping port forwarding..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    echo "Done."
}
trap cleanup EXIT INT TERM

# Wait for services to exist
echo "Checking services..."
for svc in "prom-kube-prometheus-stack-prometheus:9090" "loki:3100" "tempo:3200"; do
    name="${svc%%:*}"
    port="${svc##*:}"
    if ! kubectl get svc "$name" -n "$NAMESPACE" &>/dev/null; then
        echo "WARNING: Service $name not found in namespace $NAMESPACE"
        echo "  Some port-forwarding will fail. Run deploy-all.sh first."
    fi
done

echo ""
echo "Starting port forwarding..."
echo "-------------------------------------------------------"

# Prometheus
kubectl port-forward -n "$NAMESPACE" svc/prom-kube-prometheus-stack-prometheus 9090:9090 >/dev/null 2>&1 &
PIDS+=($!)
echo "  Prometheus  → http://localhost:9090  (PID $!)"
sleep 1

# Loki
kubectl port-forward -n "$NAMESPACE" svc/loki 3100:3100 >/dev/null 2>&1 &
PIDS+=($!)
echo "  Loki        → http://localhost:3100  (PID $!)"
sleep 1

# Tempo
kubectl port-forward -n "$NAMESPACE" svc/tempo 3200:3200 >/dev/null 2>&1 &
PIDS+=($!)
echo "  Tempo       → http://localhost:3200  (PID $!)"
sleep 1

echo "-------------------------------------------------------"
echo ""
echo "Port forwarding active. Press Ctrl+C to stop."
echo ""
echo "  Also available via NodePort (no port-forward needed):"
echo "    Grafana  → http://localhost:30000  (admin/admin)"
echo "    Locust   → http://localhost:30001"
echo "    Boutique → http://localhost:30080"
echo ""
echo "  Verify APIs:"
echo "    curl -s http://localhost:9090/api/v1/query?query=up | python3 -m json.tool"
echo "    curl -s http://localhost:3100/loki/api/v1/labels"
echo "    curl -s http://localhost:3200/api/echo"
echo ""

wait
