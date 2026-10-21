#!/bin/bash
# ============================================================
# Deploy TrainTicket microservice application
# ============================================================
# Prerequisites: kind cluster running, Istio installed, observability stack up
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
MANIFEST_SRC="/tmp/train-ticket/deployment/kubernetes-manifests/k8s-with-istio"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

NAMESPACE="trainticket"

# ============================================================
# Step 0: Clone TrainTicket repo if needed
# ============================================================
ensure_manifests() {
    if [ ! -d "$MANIFEST_SRC" ]; then
        info "Cloning TrainTicket repository..."
        git clone --depth 1 https://github.com/FudanSELab/train-ticket.git /tmp/train-ticket
    fi
    if [ ! -d "$MANIFEST_SRC" ]; then
        err "Manifest directory not found after clone. Check /tmp/train-ticket"
        exit 1
    fi
    info "Manifests ready ✓"
}

# ============================================================
# Step 1: Create namespace with Istio injection
# ============================================================
create_namespace() {
    info "Creating namespace '$NAMESPACE' with Istio sidecar injection..."
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace "$NAMESPACE" istio-injection=enabled --overwrite
    info "Namespace $NAMESPACE ready ✓"
}

# ============================================================
# Step 2: Patch manifests and apply
# ============================================================
deploy_services() {
    info "Deploying TrainTicket services..."

    # Prepare patched manifests with namespace
    local patch_dir="$SCRIPT_DIR/.manifests"
    mkdir -p "$patch_dir"

    for part in part1 part2 part3; do
        local src="$MANIFEST_SRC/ts-deployment-${part}.yml"
        local dst="$patch_dir/ts-deployment-${part}.yml"

        if [ ! -f "$src" ]; then
            warn "  $src not found, skipping"
            continue
        fi

        # Add namespace to all resources using kubectl dry-run approach
        # Simple approach: sed add namespace to metadata
        python3 - "$src" "$dst" "$NAMESPACE" <<'PYEOF'
import sys, re

src_file, dst_file, namespace = sys.argv[1], sys.argv[2], sys.argv[3]

with open(src_file) as f:
    content = f.read()

# Remove nodeName directives (kind doesn't have named nodes)
content = re.sub(r'\s+nodeName:\s+\S+\n', '\n', content)

# Change nodePort for ts-ui-dashboard to 30082
content = content.replace('nodePort: 32677', 'nodePort: 30082')

# Fix MongoDB image version (TrainTicket uses old driver incompatible with mongo:5+)
content = re.sub(r'image:\s+mongo\s*$', 'image: mongo:4.4', content, flags=re.MULTILINE)
content = re.sub(r"image:\s+'mongo'\s*$", "image: 'mongo:4.4'", content, flags=re.MULTILINE)

# Add namespace to metadata sections
# Pattern: after "metadata:\n  name: xxx" add "  namespace: trainticket"
lines = content.split('\n')
output = []
i = 0
while i < len(lines):
    output.append(lines[i])
    # Detect "metadata:" followed by "  name:" (resource-level metadata)
    if lines[i].strip() == 'metadata:' and i + 1 < len(lines):
        # Check if next line is name: (not labels:)
        next_stripped = lines[i + 1].strip()
        if next_stripped.startswith('name:'):
            # Check indentation of metadata line
            indent = len(lines[i]) - len(lines[i].lstrip())
            # Only add namespace to top-level resources (indent 0-2)
            if indent <= 2 and 'namespace' not in content[max(0, content.find(lines[i])-50):content.find(lines[i])+200].split('metadata:')[0] if 'metadata:' in content[content.find(lines[i]):content.find(lines[i])+200] else True:
                output.append(' ' * indent + '  namespace: ' + namespace)
    i += 1

with open(dst_file, 'w') as f:
    f.write('\n'.join(output))
print(f"  Patched: {dst_file}")
PYEOF
    done

    # Apply in order: part1 (databases) -> part2 (services) -> part3 (UI)
    for part in part1 part2 part3; do
        local manifest="$patch_dir/ts-deployment-${part}.yml"
        if [ -f "$manifest" ]; then
            info "  Applying ts-deployment-${part}.yll..."
            kubectl apply -f "$manifest" 2>&1 | tail -1
        fi
    done

    info "All manifests applied ✓"
}

# ============================================================
# Step 3: Wait for pods
# ============================================================
wait_for_pods() {
    info "Waiting for all $NAMESPACE pods to be Ready (this may take 3-5 minutes)..."
    local timeout=600
    local waited=0
    while [ $waited -lt $timeout ]; do
        local total running
        total=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
        running=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | grep -c "Running" || echo "0")
        if [ "$total" -gt 0 ] && [ "$running" -eq "$total" ]; then
            info "All $running pods Running ✓"
            return 0
        fi
        info "  $running/$total pods Running ($waited/${timeout}s)..."
        sleep 10
        waited=$((waited + 10))
    done
    warn "Timeout. Pod status:"
    kubectl get pods -n "$NAMESPACE" | grep -v Running || true
}

# ============================================================
# Step 4: Configure OTel tracing
# ============================================================
configure_tracing() {
    info "Configuring OTel tracing (COLLECTOR_SERVICE_ADDR → Tempo)..."
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
    local count=0
    for deploy in $deployments; do
        # Skip MongoDB and MySQL deployments
        if echo "$deploy" | grep -qE "mongo|mysql"; then
            continue
        fi
        kubectl set env "deployment/$deploy" \
            COLLECTOR_SERVICE_ADDR=tempo.observability:4317 \
            -n "$NAMESPACE" 2>/dev/null || true
        count=$((count + 1))
    done
    info "OTel tracing configured for $count services ✓"

    # Wait for rolling update
    info "Waiting for pods to stabilize after env update..."
    sleep 30
    wait_for_pods
}

# ============================================================
# Step 5: Apply HPA configs
# ============================================================
apply_hpa() {
    info "Applying HPA configs for T1 targets..."
    if [ -f "$SCRIPT_DIR/hpa-trainticket.yaml" ]; then
        kubectl apply -f "$SCRIPT_DIR/hpa-trainticket.yaml"
        info "HPA configs applied ✓"
    else
        warn "hpa-trainticket.yaml not found, skipping HPA setup."
    fi
}

# ============================================================
# Summary
# ============================================================
print_summary() {
    echo ""
    info "=========================================="
    info "  TrainTicket Deployment Complete!"
    info "=========================================="
    echo ""
    kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null | head -5
    echo "  ..."
    local total
    total=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    echo "  Total: $total pods"
    echo ""
    echo "  TrainTicket UI: http://localhost:30082"
    echo ""
}

# ============================================================
# Main
# ============================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Deploying TrainTicket Microservices         ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""

    ensure_manifests
    create_namespace
    deploy_services
    wait_for_pods
    configure_tracing
    apply_hpa
    print_summary
}

main "$@"
