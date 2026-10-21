#!/bin/bash
# ============================================================
# Phase 1: One-click infrastructure deployment
# Dynamic Topology Fault Dataset Collection System
# ============================================================
# Prerequisites: docker, kind, kubectl, helm, istioctl
# Recommended: 8GB+ RAM, 4+ CPU cores
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================
# Prerequisites Check
# ============================================================
check_prerequisites() {
    info "Checking prerequisites..."

    local missing=()
    for tool in docker kind kubectl helm istioctl; do
        if ! command -v "$tool" &>/dev/null; then
            missing+=("$tool")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        err "Missing tools: ${missing[*]}"
        echo ""
        echo "Install instructions:"
        echo "  kind:      https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
        echo "  kubectl:   https://kubernetes.io/docs/tasks/tools/"
        echo "  helm:      https://helm.sh/docs/intro/install/"
        echo "  istioctl:  curl -L https://istio.io/downloadIstio | sh -"
        exit 1
    fi

    # Check Docker is running
    if ! docker info &>/dev/null; then
        err "Docker is not running. Please start Docker first."
        exit 1
    fi

    # Check Python
    if ! command -v python3 &>/dev/null; then
        warn "python3 not found. Install it for the experiment controller."
    fi

    info "All prerequisites found ✓"
}

# ============================================================
# Step 0: Create kind cluster
# ============================================================
create_cluster() {
    info "=========================================="
    info "Step 0: Creating kind cluster"
    info "=========================================="

    if kind get clusters 2>/dev/null | grep -q "fault-exp"; then
        warn "Cluster 'fault-exp' already exists, skipping creation."
    else
        kind create cluster --config "$SCRIPT_DIR/kind-config.yaml"
        info "Cluster created ✓"
    fi

    kubectl config use-context kind-fault-exp

    # Load containerd socket path for Chaos Mesh
    for node in kind-fault-exp-worker kind-fault-exp-worker2; do
        docker exec "$node" mkdir -p /run/containerd 2>/dev/null || true
    done

    info "Nodes:"
    kubectl get nodes
}

# ============================================================
# Step 1: Install Istio
# ============================================================
install_istio() {
    info "=========================================="
    info "Step 1: Installing Istio"
    info "=========================================="

    if kubectl get namespace istio-system &>/dev/null 2>&1 && \
       kubectl get pods -n istio-system -l app=istiod --no-headers 2>/dev/null | grep -q "Running"; then
        warn "Istio already installed and running, skipping."
    else
        istioctl install --set profile=demo -y
        info "Istio installed ✓"
    fi

    # Create boutique namespace with sidecar injection
    kubectl create namespace boutique --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace boutique istio-injection=enabled --overwrite

    info "Istio status:"
    kubectl get pods -n istio-system
}

# ============================================================
# Step 2: Deploy Observability Stack
# ============================================================
deploy_observability() {
    info "=========================================="
    info "Step 2: Deploying Observability Stack"
    info "=========================================="

    kubectl create namespace observability --dry-run=client -o yaml | kubectl apply -f -

    # Add Helm repositories
    info "Adding Helm repositories..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
    helm repo update

    # --- Prometheus + Grafana ---
    if helm status prom -n observability &>/dev/null; then
        warn "Prometheus stack already installed, upgrading..."
        helm upgrade prom prometheus-community/kube-prometheus-stack \
            -n observability \
            -f "$SCRIPT_DIR/values-prometheus.yaml" \
            --timeout 5m || warn "Prometheus upgrade had issues (may still work)"
    else
        info "Installing Prometheus + Grafana..."
        helm install prom prometheus-community/kube-prometheus-stack \
            -n observability \
            -f "$SCRIPT_DIR/values-prometheus.yaml" \
            --timeout 5m
        info "Prometheus + Grafana installed ✓"
    fi

    # --- Loki + Promtail ---
    if helm status loki -n observability &>/dev/null; then
        warn "Loki already installed, skipping."
    else
        info "Installing Loki + Promtail..."
        helm install loki grafana/loki-stack \
            -n observability \
            --set loki.persistence.enabled=true \
            --set loki.persistence.size=10Gi \
            --set grafana.enabled=false \
            --set prometheus.enabled=false \
            --timeout 5m
        info "Loki + Promtail installed ✓"
    fi

    # --- Tempo ---
    if helm status tempo -n observability &>/dev/null; then
        warn "Tempo already installed, skipping."
    else
        info "Installing Tempo..."
        helm install tempo grafana/tempo \
            -n observability \
            -f "$SCRIPT_DIR/values-tempo.yaml" \
            --timeout 5m
        info "Tempo installed ✓"
    fi

    # Wait for observability pods
    info "Waiting for observability pods to be ready..."
    kubectl wait --for=condition=ready pod -l "app.kubernetes.io/instance in (prom,loki,tempo)" \
        -n observability --timeout=300s 2>/dev/null || {
        warn "Some observability pods not ready yet. Continuing..."
        kubectl get pods -n observability
    }

    info "Observability Stack:"
    kubectl get pods -n observability
}

# ============================================================
# Step 3: Deploy Online Boutique
# ============================================================
deploy_boutique() {
    info "=========================================="
    info "Step 3: Deploying Online Boutique"
    info "=========================================="

    local boutique_installed=false

    # Try Helm first
    helm repo add online-boutique https://googlecloudplatform.github.io/microservices-demo 2>/dev/null || true
    helm repo update

    if helm status boutique -n boutique &>/dev/null; then
        warn "Online Boutique already installed, skipping."
        boutique_installed=true
    else
        info "Installing Online Boutique via Helm..."
        if helm install boutique online-boutique/microservices-demo \
            -n boutique \
            -f "$SCRIPT_DIR/values-boutique.yaml" \
            --timeout 5m 2>/dev/null; then
            info "Helm install successful ✓"
            boutique_installed=true
        else
            warn "Helm install failed, falling back to raw manifests..."
            helm uninstall boutique -n boutique 2>/dev/null || true

            # Download and apply raw manifests
            local manifest_url="https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/v0.8.1/release/kubernetes-manifests.yaml"
            info "Downloading manifests from $manifest_url ..."
            kubectl apply -n boutique -f "$manifest_url"
            boutique_installed=true
        fi
    fi

    if [ "$boutique_installed" = true ]; then
        # Patch frontend service to NodePort
        info "Patching frontend service to NodePort:30080..."
        kubectl patch svc frontend -n boutique \
            -p '{"spec":{"type":"NodePort"}}' 2>/dev/null || true

        # Try setting nodePort (may already be set)
        kubectl patch svc frontend -n boutique --type=json \
            -p='[{"op":"replace","path":"/spec/ports/0/nodePort","value":30080}]' 2>/dev/null || true

        # Configure OTel tracing: point all services to Tempo
        info "Configuring OTel tracing (COLLECTOR_SERVICE_ADDR → Tempo)..."
        local deployments
        deployments=$(kubectl get deployments -n boutique -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
        for deploy in $deployments; do
            kubectl set env "deployment/$deploy" \
                COLLECTOR_SERVICE_ADDR=tempo.observability:4317 \
                -n boutique 2>/dev/null || true
        done
        info "OTel tracing configured for: $deployments"
    fi

    # Wait for all pods
    info "Waiting for Online Boutique pods..."
    kubectl wait --for=condition=ready pod -l app -n boutique --timeout=300s 2>/dev/null || {
        warn "Some pods not ready yet. Check with: kubectl get pods -n boutique"
        kubectl get pods -n boutique
    }

    info "Online Boutique:"
    kubectl get pods -n boutique
}

# ============================================================
# Step 4: Install Chaos Mesh
# ============================================================
install_chaos_mesh() {
    info "=========================================="
    info "Step 4: Installing Chaos Mesh"
    info "=========================================="

    helm repo add chaos-mesh https://charts.chaos-mesh.org 2>/dev/null || true
    helm repo update

    kubectl create namespace chaos-mesh --dry-run=client -o yaml | kubectl apply -f -

    if helm status chaos-mesh -n chaos-mesh &>/dev/null; then
        warn "Chaos Mesh already installed, skipping."
    else
        info "Installing Chaos Mesh..."
        helm install chaos-mesh chaos-mesh/chaos-mesh \
            -n chaos-mesh \
            --set chaosDaemon.runtime=containerd \
            --set chaosDaemon.socketPath=/run/containerd/containerd.sock \
            --set dashboard.create=true \
            --set dashboard.securityMode=false \
            --timeout 5m
        info "Chaos Mesh installed ✓"
    fi

    info "Waiting for Chaos Mesh pods..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=chaos-mesh \
        -n chaos-mesh --timeout=300s 2>/dev/null || {
        warn "Some Chaos Mesh pods not ready yet."
        kubectl get pods -n chaos-mesh
    }

    info "Chaos Mesh:"
    kubectl get pods -n chaos-mesh
}

# ============================================================
# Step 5: Deploy Locust
# ============================================================
deploy_locust() {
    info "=========================================="
    info "Step 5: Deploying Locust load generator"
    info "=========================================="

    kubectl apply -f "$SCRIPT_DIR/locust/deployment.yaml"

    info "Waiting for Locust..."
    kubectl wait --for=condition=ready pod -l app=locust-master \
        -n boutique --timeout=120s 2>/dev/null || {
        warn "Locust pod not ready. Check: kubectl logs -l app=locust-master -n boutique"
    }

    info "Locust:"
    kubectl get pods -n boutique -l app=locust-master
}

# ============================================================
# Step 6: Apply HPA configs
# ============================================================
apply_hpa() {
    info "=========================================="
    info "Step 6: Applying HPA configs (for T1 scenario)"
    info "=========================================="

    if [ -f "$SCRIPT_DIR/hpa-configs.yaml" ]; then
        kubectl apply -f "$SCRIPT_DIR/hpa-configs.yaml"
        info "HPA configs applied ✓"
        kubectl get hpa -n boutique
    else
        warn "hpa-configs.yaml not found, skipping HPA setup."
    fi
}

# ============================================================
# Final Summary
# ============================================================
print_summary() {
    echo ""
    info "=========================================="
    info "  Phase 1 Deployment Complete!"
    info "=========================================="
    echo ""
    echo "  Component Status:"
    echo "  ─────────────────────────────────────────"
    kubectl get pods -A -o wide 2>/dev/null | grep -E "NAME|istio-system|observability|boutique|chaos-mesh" || true
    echo ""
    echo "  Access Points:"
    echo "  ─────────────────────────────────────────"
    echo "  Grafana:        http://localhost:30000  (admin/admin)"
    echo "  Locust UI:      http://localhost:30001"
    echo "  Boutique:       http://localhost:30080"
    echo ""
    echo "  Next Steps:"
    echo "  ─────────────────────────────────────────"
    echo "  1. Start port forwarding:"
    echo "     $ ./scripts/start-port-forward.sh"
    echo ""
    echo "  2. Verify the cluster:"
    echo "     $ ./infra/verify-cluster.sh"
    echo ""
    echo "  3. Start load generation:"
    echo "     Open http://localhost:30001 → Start with 10 users"
    echo ""
    echo "  4. Check observability in Grafana:"
    echo "     Open http://localhost:30000"
    echo "     → Explore → Prometheus: query 'istio_requests_total'"
    echo "     → Explore → Loki: query '{namespace=\"boutique\"}'"
    echo "     → Explore → Tempo: search service='frontend'"
    echo ""
}

# ============================================================
# Main
# ============================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Dynamic Topology Fault Dataset Collection   ║"
    echo "║  Phase 1: Infrastructure Deployment          ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""

    check_prerequisites
    create_cluster
    install_istio
    deploy_observability
    deploy_boutique
    install_chaos_mesh
    deploy_locust
    apply_hpa

    print_summary
}

main "$@"
