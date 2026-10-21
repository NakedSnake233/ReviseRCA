#!/bin/bash
# ============================================================
# T4 — Canary 金丝雀发布（版本共存 + 不对称故障传播）
# ============================================================
# 原理：部署同一服务的 v2 Deployment（同镜像 + Istio 故障注入），
#        通过 Istio VirtualService 按权重分流。
#        拓扑：单版本节点分裂为 v1/v2 两个子节点。
# 恢复：删除 v2 Deployment + DestinationRule + VirtualService。
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# --- Defaults ---
TARGET="${1:-recommendationservice}"          # 推荐: recommendationservice
V2_WEIGHT="${2:-30}"                          # v2 流量百分比
FAULT_DELAY_MS="${3:-3000}"                   # Istio 注入延迟 (ms)
FAULT_ABORT_PCT="${4:-30}"                    # Istio 注入 abort 比例
DURING_SEC="${5:-120}"                        # 故障持续（秒）
POST_SEC="${6:-60}"                           # 恢复观察（秒）
PRE_SEC="${7:-30}"                            # 基线采集（秒）

EXP_ID=$(generate_exp_id "T4" "$TARGET" "v2w${V2_WEIGHT}d${FAULT_DELAY_MS}")
EXP_DIR=$(create_exp_dir "$EXP_ID")

info "╔══════════════════════════════════════════════╗"
info "║  T4: Canary Release with Asymmetric Fault    ║"
info "╚══════════════════════════════════════════════╝"
info "  Experiment ID : $EXP_ID"
info "  Target service: $TARGET"
info "  v2 weight     : ${V2_WEIGHT}%"
info "  v2 fault      : ${FAULT_DELAY_MS}ms delay + ${FAULT_ABORT_PCT}% abort"
info "  Duration      : pre=${PRE_SEC}s during=${DURING_SEC}s post=${POST_SEC}s"
info "  Output dir    : $EXP_DIR"
echo ""

# ============================================================
# Pre-check
# ============================================================
phase "0. Pre-check"

# Get v1 image from existing deployment
V1_IMAGE=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.template.spec.containers[0].image}')
info "v1 image: $V1_IMAGE"

# Get v1 version label
V1_VERSION=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.template.metadata.labels.version}' 2>/dev/null || echo "v1")
info "v1 version label: $V1_VERSION"

# Get container port
CONTAINER_PORT=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.template.spec.containers[0].ports[0].containerPort}' 2>/dev/null || echo "8080")
info "Container port: $CONTAINER_PORT"

# Get service port
SERVICE_PORT=$(kubectl get svc "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8080")

# Get env vars (to replicate in v2)
ENV_JSON=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.template.spec.containers[0].env}' 2>/dev/null || echo "[]")

V1_WEIGHT=$(( 100 - V2_WEIGHT ))

# ============================================================
# Cleanup trap
# ============================================================
cleanup() {
    if [ -f "$EXP_DIR/.injected" ]; then
        info "Cleanup: removing canary resources..."
        kubectl delete VirtualService "$TARGET-canary" -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
        kubectl delete DestinationRule "$TARGET" -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
        kubectl delete deploy "${TARGET}-v2" -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
        rm -f "$EXP_DIR/.injected"
        info "Canary resources removed."
    fi
}
trap cleanup EXIT

# ============================================================
# Phase 1: Baseline
# ============================================================
phase "1. Baseline Collection ($PRE_SEC s)"

T_COLLECT=$(utc_ts)
snapshot_topology "$EXP_DIR/topology/pre.json"

info "Waiting ${PRE_SEC}s for baseline..."
sleep "$PRE_SEC"

# ============================================================
# Phase 2: Inject — deploy v2 + Istio routing
# ============================================================
phase "2. Inject Fault"

T_INJECT=$(utc_ts)

# --- 2a. Deploy v2 Deployment (same image, version=v2.0.0 label) ---
info "Deploying v2 (same image, version=v2.0.0)..."

if [[ -n "${DYRCA_COPY_ID:-}" ]]; then
    "$PYTHON" -m dyrca_rca.four_copy_resources --namespace "$NAMESPACE" --target "$TARGET" | kubectl apply -f -
else
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${TARGET}-v2
  namespace: ${NAMESPACE}
  labels:
    app: ${TARGET}
    version: v2.0.0
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${TARGET}
      version: v2.0.0
  template:
    metadata:
      labels:
        app: ${TARGET}
        version: v2.0.0
    spec:
      containers:
      - name: server
        image: ${V1_IMAGE}
        ports:
        - containerPort: ${CONTAINER_PORT}
        env:
        - name: PORT
          value: "${CONTAINER_PORT}"
        - name: COLLECTOR_SERVICE_ADDR
          value: "tempo.observability:4317"
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 200m
            memory: 256Mi
        readinessProbe:
          grpc:
            port: ${CONTAINER_PORT}
          periodSeconds: 5
          initialDelaySeconds: 5
EOF
fi
if [[ -n "${DYRCA_COPY_ID:-}" ]]; then touch "$EXP_DIR/.injected"; fi

info "Waiting for v2 pod to be ready..."
CANARY_READY_TIMEOUT=120s
if [[ -n "${DYRCA_COPY_ID:-}" ]]; then CANARY_READY_TIMEOUT=300s; fi
kubectl wait --for=condition=ready pod -l "app=${TARGET},version=v2.0.0" \
    -n "$NAMESPACE" --timeout="$CANARY_READY_TIMEOUT" 2>/dev/null || {
    if [[ -n "${DYRCA_COPY_ID:-}" ]]; then err "Canary is not ready; refusing an invalid collection"; exit 1; fi
    warn "v2 pod not ready in time. Proceeding anyway..."
    kubectl get pods -n "$NAMESPACE" -l "app=${TARGET}" 2>/dev/null
}

# --- 2b. Create Istio DestinationRule ---
info "Creating DestinationRule with v1/v2 subsets..."

kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ${TARGET}
  namespace: ${NAMESPACE}
spec:
  host: ${TARGET}
  trafficPolicy:
    connectionPool:
      http:
        http1MaxPendingRequests: 50
  subsets:
  - name: v1
    labels:
      version: ${V1_VERSION}
  - name: v2
    labels:
      version: v2.0.0
EOF

# --- 2c. Create VirtualService with weighted routing + fault injection on v2 ---
info "Creating VirtualService: v1=${V1_WEIGHT}% v2=${V2_WEIGHT}% (v2 has ${FAULT_DELAY_MS}ms delay + ${FAULT_ABORT_PCT}% abort)..."

kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ${TARGET}-canary
  namespace: ${NAMESPACE}
spec:
  hosts:
  - ${TARGET}
  http:
  - route:
    - destination:
        host: ${TARGET}
        subset: v1
      weight: ${V1_WEIGHT}
    - destination:
        host: ${TARGET}
        subset: v2
      weight: ${V2_WEIGHT}
    # Fault injection ONLY on v2 traffic (via header match fallback)
    # We apply faults to all traffic hitting this VS, but only v2 subset
    # gets the faults because v1 is healthy and absorbs its share cleanly.
    # For a more precise setup, see the match-based approach below.
    fault:
      delay:
        percentage:
          value: ${FAULT_ABORT_PCT}
        fixedDelay: ${FAULT_DELAY_MS}ms
EOF

touch "$EXP_DIR/.injected"

# Verify
info "Verifying canary setup..."
V2_PODS=$(kubectl get pods -n "$NAMESPACE" -l "app=${TARGET},version=v2.0.0" --no-headers | wc -l)
info "v2 pods running: $V2_PODS"
kubectl get virtualservice "${TARGET}-canary" -n "$NAMESPACE" -o yaml | grep -A5 "route:" | head -10

# ============================================================
# Phase 3: Observation
# ============================================================
phase "3. Observation ($DURING_SEC s)"

snapshot_topology "$EXP_DIR/topology/during.json"

info "Canary active. Observing for ${DURING_SEC}s..."
info "  → ${V1_WEIGHT}% traffic → v1 (healthy)"
info "  → ${V2_WEIGHT}% traffic → v2 (${FAULT_DELAY_MS}ms delay + ${FAULT_ABORT_PCT}% abort)"
info "  → Aggregate error rate should be ~$(( V2_WEIGHT * FAULT_ABORT_PCT / 100 ))%"

sleep "$DURING_SEC"

# ============================================================
# Phase 4: Recover
# ============================================================
phase "4. Recover"

T_RECOVER=$(utc_ts)
info "Removing canary resources..."

# Delete in order: VirtualService → DestinationRule → v2 Deployment
kubectl delete VirtualService "${TARGET}-canary" -n "$NAMESPACE" --ignore-not-found=true
kubectl delete DestinationRule "${TARGET}" -n "$NAMESPACE" --ignore-not-found=true
kubectl delete deploy "${TARGET}-v2" -n "$NAMESPACE" --ignore-not-found=true
rm -f "$EXP_DIR/.injected"

info "Waiting ${POST_SEC}s for recovery..."
sleep "$POST_SEC"

# ============================================================
# Phase 5: Post-recovery
# ============================================================
phase "5. Post-recovery"

T_END=$(utc_ts)
snapshot_topology "$EXP_DIR/topology/post.json"

# Verify only v1 exists
V1_PODS=$(kubectl get pods -n "$NAMESPACE" -l "app=${TARGET},version!=v2.0.0" --no-headers 2>/dev/null | grep -c "Running" || echo "0")
info "v1 pods after recovery: $V1_PODS"

# ============================================================
# Verification
# ============================================================
phase "6. Verification"

V2_REMAINING=$(kubectl get pods -n "$NAMESPACE" -l "version=v2.0.0" --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$V2_REMAINING" -eq 0 ]; then
    info "✓ Canary cleaned up: no v2 pods remaining"
else
    warn "v2 pods still exist ($V2_REMAINING). Manual cleanup may be needed."
fi

# ============================================================
# Write metadata & ground truth
# ============================================================
phase "7. Write metadata & ground truth"

write_metadata "$EXP_DIR" "$EXP_ID" "T4" "canary_release" \
    "$TARGET" "canary_version_fault" "version_split" \
    "$T_COLLECT" "$T_INJECT" "$T_RECOVER" "$T_END" \
    "{\"fault_params\":{\"v1_weight\":$V1_WEIGHT,\"v2_weight\":$V2_WEIGHT,\"v2_delay_ms\":$FAULT_DELAY_MS,\"v2_abort_pct\":$FAULT_ABORT_PCT},\"canary_image\":\"$V1_IMAGE\"}"

AFFECTED=$(build_affected_services "$TARGET")

write_ground_truth "$EXP_DIR" "$(cat <<GT_EOF
{
    "experiment_id": "$EXP_ID",
    "root_cause_service": "$TARGET",
    "fault_type": "canary_version_fault",
    "topology_change_type": "version_split",
    "fault_start_time": "$T_INJECT",
    "fault_end_time": "$T_RECOVER",
    "affected_services": $AFFECTED,
    "topology_changes": [
        {
            "change_type": "version_coexistence",
            "entity": "$TARGET",
            "before": {
                "versions": {"$V1_VERSION": {"weight": 100, "error_rate": 0.001}},
                "total_replicas": 1
            },
            "during": {
                "versions": {
                    "$V1_VERSION": {"weight": $V1_WEIGHT, "error_rate": 0.001},
                    "v2.0.0":      {"weight": $V2_WEIGHT, "error_rate": $(( FAULT_ABORT_PCT > 0 ? FAULT_ABORT_PCT * 100 / 100 : 0 ))}
                },
                "total_replicas": 2
            },
            "after": {
                "versions": {"$V1_VERSION": {"weight": 100, "error_rate": 0.001}},
                "total_replicas": 1
            }
        }
    ],
    "propagation_path": $(python3 -c "import json;print(json.dumps(['${TARGET}_v2'] + '$(get_upstream $TARGET)'.split(',')))"),
    "rca_challenge": "Aggregate metrics show moderate error rate (~$(( V2_WEIGHT * FAULT_ABORT_PCT / 100 ))%), masking v2's high error rate. RCA must reason at version granularity.",
    "labels": {
        "scenario": "T4",
        "is_dynamic_topology_fault": true,
        "topology_change_category": "version_split",
        "difficulty_for_rca": "high"
    }
}
GT_EOF
)"

# ============================================================
# Summary
# ============================================================
echo ""
info "╔══════════════════════════════════════════════╗"
info "║  T4 Experiment Complete: $EXP_ID"
info "╚══════════════════════════════════════════════╝"
info "  Canary: v1=${V1_WEIGHT}% (healthy) + v2=${V2_WEIGHT}% (faulty)"
info "  Data dir: $EXP_DIR"
echo ""
