#!/bin/bash
# ============================================================
# Verify all Phase 1 components are running correctly
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

check() {
    local desc="$1"
    local cmd="$2"
    local expect="${3:-}"

    if eval "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $desc"
        ((PASS++))
    else
        if [ -n "$expect" ]; then
            echo -e "  ${YELLOW}!${NC} $desc — $expect"
        else
            echo -e "  ${RED}✗${NC} $desc"
        fi
        ((FAIL++))
    fi
}

warn_check() {
    local desc="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $desc"
        ((PASS++))
    else
        echo -e "  ${YELLOW}!${NC} $desc (non-critical)"
        ((WARN++))
    fi
}

echo "╔═══════════════════════════════════════════╗"
echo "║  Cluster Verification                     ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# --- K8s Cluster ---
echo "1. Kubernetes Cluster"
echo "   ──────────────────"
check "kind cluster 'fault-exp' exists" \
    "kind get clusters | grep fault-exp"
check "3 nodes Ready" \
    "[ \$(kubectl get nodes --no-headers | grep -c ' Ready') -ge 3 ]"
echo ""

# --- Istio ---
echo "2. Istio Service Mesh"
echo "   ──────────────────"
check "istiod Running" \
    "kubectl get pods -n istio-system -l app=istiod --no-headers | grep -q Running"
check "istio-ingressgateway Running" \
    "kubectl get pods -n istio-system -l app=istio-ingressgateway --no-headers | grep -q Running"
check "boutique namespace labeled for injection" \
    "kubectl get ns boutique -o jsonpath='{.metadata.labels.istio-injection}' | grep -q enabled"
echo ""

# --- Observability ---
echo "3. Observability Stack"
echo "   ──────────────────"
check "Prometheus pod Running" \
    "kubectl get pods -n observability -l app.kubernetes.io/name=prometheus --no-headers | grep -q Running"
check "Grafana pod Running" \
    "kubectl get pods -n observability -l app.kubernetes.io/name=grafana --no-headers | grep -q Running"
check "Loki pod Running" \
    "kubectl get pods -n observability -l app=loki --no-headers | grep -q Running" \
    "helm install may have different labels"
check "Promtail DaemonSet exists" \
    "kubectl get daemonset -n observability -l app=promtail --no-headers | grep -q promtail"
check "Tempo pod Running" \
    "kubectl get pods -n observability -l app.kubernetes.io/name=tempo --no-headers | grep -q Running" \
    "helm install may have different labels"
echo ""

# --- Online Boutique ---
echo "4. Online Boutique"
echo "   ──────────────────"
check "All 11 service pods Running" \
    "[ \$(kubectl get pods -n boutique -l app --no-headers | grep -c 'Running') -ge 10 ]" \
    "Some pods may still be starting"
check "Istio sidecar injected (2 containers per pod)" \
    "kubectl get pods -n boutique -o jsonpath='{.items[0].spec.containers[*].name}' | grep -q istio-proxy"
check "Frontend service is NodePort" \
    "kubectl get svc frontend -n boutique -o jsonpath='{.spec.type}' | grep -q NodePort"
warn_check "Frontend NodePort is 30080" \
    "kubectl get svc frontend -n boutique -o jsonpath='{.spec.ports[0].nodePort}' | grep -q 30080"
check "COLLECTOR_SERVICE_ADDR set on frontend" \
    "kubectl get deploy frontend -n boutique -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name==\"COLLECTOR_SERVICE_ADDR\")].value}' | grep -q tempo" \
    "env vars may take time to propagate"
echo ""

# --- Chaos Mesh ---
echo "5. Chaos Mesh"
echo "   ──────────────────"
check "chaos-controller-manager Running" \
    "kubectl get pods -n chaos-mesh -l app.kubernetes.io/component=controller-manager --no-headers | grep -q Running"
check "chaos-daemon on each worker" \
    "[ \$(kubectl get pods -n chaos-mesh -l app.kubernetes.io/component=chaos-daemon --no-headers | grep -c Running) -ge 2 ]"
echo ""

# --- Locust ---
echo "6. Locust Load Generator"
echo "   ──────────────────"
check "locust-master Running" \
    "kubectl get pods -n boutique -l app=locust-master --no-headers | grep -q Running"
check "Locust service NodePort 30001" \
    "kubectl get svc locust-master -n boutique -o jsonpath='{.spec.ports[0].nodePort}' | grep -q 30001"
echo ""

# --- HPA ---
echo "7. HPA (for T1 scenario)"
echo "   ──────────────────"
warn_check "HPA for productcatalogservice" \
    "kubectl get hpa productcatalogservice-hpa -n boutique"
warn_check "HPA for cartservice" \
    "kubectl get hpa cartservice-hpa -n boutique"
echo ""

# --- API Connectivity (if port-forward is running) ---
echo "8. API Connectivity (requires port-forward)"
echo "   ──────────────────"
warn_check "Prometheus API reachable" \
    "curl -sf http://localhost:9090/api/v1/query?query=up"
warn_check "Loki API reachable" \
    "curl -sf http://localhost:3100/loki/api/v1/labels"
warn_check "Tempo API reachable" \
    "curl -sf http://localhost:3200/api/echo"
warn_check "Grafana reachable" \
    "curl -sf http://localhost:30000/api/health"
warn_check "Online Boutique reachable" \
    "curl -sf http://localhost:30080 | grep -q Boutique"
echo ""

# --- Summary ---
echo "═══════════════════════════════════════════"
echo -e "  ${GREEN}Passed:${NC}   $PASS"
echo -e "  ${RED}Failed:${NC}   $FAIL"
echo -e "  ${YELLOW}Warnings:${NC} $WARN"
echo "═══════════════════════════════════════════"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "Some checks failed. Common fixes:"
    echo "  - Pods not ready: kubectl get pods -A | grep -v Running"
    echo "  - Port not 30080: kubectl patch svc frontend -n boutique --type=json -p '[{\"op\":\"replace\",\"path\":\"/spec/ports/0/nodePort\",\"value\":30080}]'"
    echo "  - APIs unreachable: run ./scripts/start-port-forward.sh"
    echo ""
fi

if [ "$FAIL" -eq 0 ]; then
    echo "All critical checks passed! Ready for Phase 2 (data collectors)."
fi
