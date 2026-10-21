#!/bin/bash
# ============================================================
# T1 — HPA Elastic Scale-Out/Scale-In (TrainTicket)
# ============================================================
# Principle: Inject CPU stress via Chaos Mesh StressChaos -> trigger HPA
#            -> replicas scale from 1 to N -> topology node count increases
# Recovery: Delete StressChaos -> HPA scales back -> wait for cooldown
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_trainticket.sh
source "$SCRIPT_DIR/common_trainticket.sh"

# --- Defaults ---
TARGET="${1:-ts-travel-service}"             # Recommended: ts-travel-service, ts-order-service
CPU_WORKERS="${2:-4}"                        # Stress threads
CPU_LOAD="${3:-80}"                          # Stress percentage (0-100)
DURING_SEC="${4:-90}"                        # Fault duration (seconds)
POST_SEC="${5:-90}"                          # Recovery observation time (seconds)
PRE_SEC="${6:-30}"                           # Pre-fault baseline collection time

EXP_ID=$(generate_exp_id "T1" "$TARGET" "stress${CPU_WORKERS}w${CPU_LOAD}p")
EXP_DIR=$(create_exp_dir "$EXP_ID")
STRESS_CREATED=false
cleanup() {
    if [[ "$STRESS_CREATED" == true ]]; then
        kubectl --request-timeout=10s delete StressChaos "$EXP_ID" -n "$NAMESPACE" \
            --ignore-not-found=true --wait=false >/dev/null 2>&1 || return 0
    fi
    touch "$EXP_DIR/fault_cleanup.done"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

info "=================================================================="
info "  T1: HPA Elastic Scale-Out/Scale-In (TrainTicket)"
info "=================================================================="
info "  Experiment ID : $EXP_ID"
info "  Target service: $TARGET"
info "  CPU stress    : workers=$CPU_WORKERS load=${CPU_LOAD}%"
info "  Duration      : pre=${PRE_SEC}s during=${DURING_SEC}s post=${POST_SEC}s"
info "  Output dir    : $EXP_DIR"
echo ""

# ============================================================
# Pre-check
# ============================================================
phase "0. Pre-check"

if ! kubectl get hpa -n "$NAMESPACE" -o name 2>/dev/null | grep -q "$TARGET"; then
    warn "No HPA found for $TARGET, creating one..."
    HPA_MAX=10
    [[ -n "${DYRCA_COPY_ID:-}" ]] && HPA_MAX=3
    kubectl autoscale deployment "$TARGET" -n "$NAMESPACE" \
        --cpu-percent=50 --min=1 --max="$HPA_MAX"
    sleep 5
fi

HPA_NAME=$(kubectl get hpa -n "$NAMESPACE" -o jsonpath="{.items[?(@.spec.scaleTargetRef.name=='$TARGET')].metadata.name}")
info "HPA: $HPA_NAME"

# Reset the target to the common one-replica baseline.  A previous canary or
# stress run can leave the HPA at its maximum, in which case a new T1 case
# cannot demonstrate scale-out.  The requested fault parameters remain fixed;
# this only removes cross-case state.
CURRENT_SPEC=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
if [ "${CURRENT_SPEC:-0}" -gt 1 ]; then
    info "Resetting $TARGET replicas: $CURRENT_SPEC -> 1"
    kubectl scale deployment "$TARGET" -n "$NAMESPACE" --replicas=1 >/dev/null
    wait_healthy 120 || true
fi

# Use desired replicas rather than readyReplicas.  HPA updates the desired
# count before all replacement pods become Ready; the desired topology is the
# observable state that the experiment is designed to capture.
REPLICAS_BEFORE=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
info "Initial replicas: $REPLICAS_BEFORE"

# ============================================================
# Phase 1: Baseline (pre-fault)
# ============================================================
phase "1. Baseline Collection ($PRE_SEC s)"

T_COLLECT=$(utc_ts)
snapshot_topology "$EXP_DIR/topology/pre.json"

info "Waiting ${PRE_SEC}s for baseline data..."
sleep "$PRE_SEC"

# ============================================================
# Phase 2: Inject CPU stress -> trigger HPA scale-up
# ============================================================
phase "2. Inject Fault"

T_INJECT=$(utc_ts)
info "Injecting StressChaos on $TARGET (workers=$CPU_WORKERS, load=${CPU_LOAD}%)..."

STRESS_CREATED=true
kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: ${EXP_ID}
  namespace: ${NAMESPACE}
spec:
  mode: all
  selector:
    namespaces: ["${NAMESPACE}"]
    labelSelectors:
      app: ${TARGET}
  stressors:
    cpu:
      workers: ${CPU_WORKERS}
      load: ${CPU_LOAD}
  duration: "${DURING_SEC}s"
EOF

sleep 3
verify_chaos StressChaos "$EXP_ID"

# ============================================================
# Phase 3: Observation (during fault)
# ============================================================
phase "3. Observation ($DURING_SEC s)"

# Wait for HPA to react (typical: 30-60s)
info "Waiting for HPA to trigger scale-up..."
HPA_WAITED=0
while [ $HPA_WAITED -lt 90 ]; do
    CURRENT=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    if [ "$CURRENT" -gt "$REPLICAS_BEFORE" ]; then
        info "Scale-up detected! Replicas: $REPLICAS_BEFORE -> $CURRENT"
        break
    fi
    sleep 5
    HPA_WAITED=$((HPA_WAITED + 5))
    info "  Waiting... ($HPA_WAITED/90s) replicas=$CURRENT"
done

REPLICAS_DURING=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
info "Peak replicas during fault: $REPLICAS_DURING"

snapshot_topology "$EXP_DIR/topology/during.json"

# Wait for remaining observation time
remaining=$(( DURING_SEC - HPA_WAITED ))
if [ "$remaining" -gt 0 ]; then
    info "Observing for remaining ${remaining}s..."
    sleep "$remaining"
fi

# ============================================================
# Phase 4: Recover
# ============================================================
phase "4. Recover"

T_RECOVER=$(utc_ts)
info "Removing StressChaos..."
kubectl delete StressChaos "$EXP_ID" -n "$NAMESPACE" --ignore-not-found=true
STRESS_CREATED=false
info "Waiting for HPA scale-down (${POST_SEC}s)..."

sleep "$POST_SEC"
if [[ -n "${DYRCA_COPY_ID:-}" ]]; then
    wait_replica_recovery "$TARGET" "$REPLICAS_BEFORE" 180 || true
fi

REPLICAS_AFTER=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
info "Replicas after recovery: $REPLICAS_AFTER"

# ============================================================
# Phase 5: Post-recovery
# ============================================================
phase "5. Post-recovery"

T_END=$(utc_ts)
snapshot_topology "$EXP_DIR/topology/post.json"

# ============================================================
# Verification
# ============================================================
phase "6. Verification"

if [ "$REPLICAS_DURING" -gt "$REPLICAS_BEFORE" ]; then
    info "Topology change verified: $REPLICAS_BEFORE -> $REPLICAS_DURING -> $REPLICAS_AFTER"
else
    warn "Topology change NOT verified (replicas stayed at $REPLICAS_DURING). HPA may not have triggered."
fi

# ============================================================
# Write metadata & ground truth
# ============================================================
phase "7. Write metadata & ground truth"

write_metadata "$EXP_DIR" "$EXP_ID" "T1" "hpa_scale" \
    "$TARGET" "cpu_stress_hpa_scale" "instance_count_change" \
    "$T_COLLECT" "$T_INJECT" "$T_RECOVER" "$T_END" \
    "{\"fault_params\":{\"cpu_workers\":$CPU_WORKERS,\"cpu_load\":$CPU_LOAD},\"replicas\":{\"before\":$REPLICAS_BEFORE,\"during\":$REPLICAS_DURING,\"after\":$REPLICAS_AFTER}}"

AFFECTED=$(build_affected_services "$TARGET")
write_ground_truth "$EXP_DIR" "$(cat <<GT_EOF
{
    "experiment_id": "$EXP_ID",
    "root_cause_service": "$TARGET",
    "fault_type": "cpu_stress_induced_hpa_scale",
    "topology_change_type": "instance_count_change",
    "fault_start_time": "$T_INJECT",
    "fault_end_time": "$T_RECOVER",
    "affected_services": $AFFECTED,
    "topology_changes": [
        {
            "change_type": "replica_count",
            "entity": "$TARGET",
            "before": {"replicas": $REPLICAS_BEFORE},
            "during": {"replicas": $REPLICAS_DURING},
            "after":  {"replicas": $REPLICAS_AFTER}
        }
    ],
    "propagation_path": $(python3 -c "import json; print(json.dumps(['$TARGET'] + '$(get_upstream $TARGET)'.split(',')))"),
    "labels": {
        "scenario": "T1",
        "is_dynamic_topology_fault": true,
        "topology_change_category": "node_count_change",
        "difficulty_for_rca": "medium"
    }
}
GT_EOF
)"

# ============================================================
# Summary
# ============================================================
echo ""
info "=================================================================="
info "  T1 Experiment Complete: $EXP_ID"
info "=================================================================="
info "  Replicas: $REPLICAS_BEFORE -> $REPLICAS_DURING -> $REPLICAS_AFTER"
info "  Data dir: $EXP_DIR"
info "  Timeline:"
info "    collect=$T_COLLECT  inject=$T_INJECT"
info "    recover=$T_RECOVER  end=$T_END"
echo ""
echo "  Files:"
ls -la "$EXP_DIR/" | tail -n +2 || true
echo ""
