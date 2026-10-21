#!/bin/bash
# ============================================================
# T2 — Pod Rebuild / Node Drift (TrainTicket)
# ============================================================
# Principle: Kill target Pod via Chaos Mesh PodChaos (pod-kill),
#            Deployment controller auto-rebuilds, possibly on a different node.
#            Topology change: Pod identity changes (name, IP), possibly node changes.
# Recovery: Pod auto-rebuilds (Deployment controller), wait for Ready.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_trainticket.sh
source "$SCRIPT_DIR/common_trainticket.sh"

# --- Defaults ---
TARGET="${1:-ts-order-service}"               # Recommended: ts-order-service, ts-travel-service
KILL_COUNT="${2:-1}"                          # Kill count (10s interval between kills)
DURING_SEC="${3:-120}"                        # Observation time (seconds)
POST_SEC="${4:-60}"                           # Recovery observation (seconds)
PRE_SEC="${5:-30}"                            # Baseline collection (seconds)

EXP_ID=$(generate_exp_id "T2" "$TARGET" "kill${KILL_COUNT}x")
EXP_DIR=$(create_exp_dir "$EXP_ID")

info "=================================================================="
info "  T2: Pod Rebuild / Node Drift (TrainTicket)"
info "=================================================================="
info "  Experiment ID : $EXP_ID"
info "  Target service: $TARGET"
info "  Kill count    : $KILL_COUNT"
info "  Duration      : pre=${PRE_SEC}s during=${DURING_SEC}s post=${POST_SEC}s"
info "  Output dir    : $EXP_DIR"
echo ""

# ============================================================
# Pre-check: record original pod identity
# ============================================================
phase "0. Pre-check"

ORIG_POD=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
ORIG_NODE=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo "")
ORIG_IP=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].status.podIP}' 2>/dev/null || echo "")

if [ -z "$ORIG_POD" ]; then
    err "No pod found for app=$TARGET in namespace $NAMESPACE"
    exit 1
fi

info "Original pod identity:"
info "  Name : $ORIG_POD"
info "  Node : $ORIG_NODE"
info "  IP   : $ORIG_IP"

# Record replica count
ORIG_REPLICAS=$(kubectl get deploy "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
info "  Replicas: $ORIG_REPLICAS"

# ============================================================
# Phase 1: Baseline
# ============================================================
phase "1. Baseline Collection ($PRE_SEC s)"

T_COLLECT=$(utc_ts)
snapshot_topology "$EXP_DIR/topology/pre.json"

info "Waiting ${PRE_SEC}s for baseline..."
sleep "$PRE_SEC"

# ============================================================
# Phase 2: Inject -- kill pod(s)
# ============================================================
phase "2. Inject Fault"

T_INJECT=$(utc_ts)

for i in $(seq 1 "$KILL_COUNT"); do
    info "Kill $i/$KILL_COUNT: applying PodChaos on $TARGET..."

    # Apply PodChaos (immediate pod-kill, no duration needed)
    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: ${EXP_ID}-kill${i}
  namespace: ${NAMESPACE}
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces: ["${NAMESPACE}"]
    labelSelectors:
      app: ${TARGET}
  gracePeriod: 5
EOF

    # PodChaos pod-kill is immediate, delete the resource right after
    sleep 5
    kubectl delete PodChaos "${EXP_ID}-kill${i}" -n "$NAMESPACE" --ignore-not-found=true

    # Wait briefly between kills if multiple
    if [ "$i" -lt "$KILL_COUNT" ]; then
        info "  Waiting 10s before next kill..."
        sleep 10
    fi
done

# Wait for new pod to be scheduled and start running
info "Waiting for replacement pod to be scheduled..."
sleep 10

# ============================================================
# Phase 3: Observation
# ============================================================
phase "3. Observation ($DURING_SEC s)"

# Capture new pod identity
NEW_POD=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "(pending)")
NEW_NODE=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo "(pending)")
NEW_IP=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].status.podIP}' 2>/dev/null || echo "(pending)")

info "New pod identity:"
info "  Name : $NEW_POD"
info "  Node : $NEW_NODE"
info "  IP   : $NEW_IP"

# Check if node changed
if [ "$NEW_NODE" != "$ORIG_NODE" ] && [ "$NEW_NODE" != "(pending)" ]; then
    info "Node drift detected: $ORIG_NODE -> $NEW_NODE"
    DRIFTED=true
else
    info "  Pod stayed on same node (or still scheduling)"
    DRIFTED=false
fi

# Check if pod name changed (it should always change after kill)
if [ "$NEW_POD" != "$ORIG_POD" ]; then
    info "Pod identity changed: $ORIG_POD -> $NEW_POD"
else
    warn "Pod name unchanged (may still be restarting)"
fi

snapshot_topology "$EXP_DIR/topology/during.json"

info "Observing for ${DURING_SEC}s..."
sleep "$DURING_SEC"

# ============================================================
# Phase 4: Recover (automatic -- Deployment controller rebuilds)
# ============================================================
phase "4. Recover (automatic rebuild)"

T_RECOVER=$(utc_ts)
info "Waiting for pod to fully stabilize..."
wait_healthy 90

T_END=$(utc_ts)

# Capture final pod identity
FINAL_POD=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
FINAL_NODE=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo "")
FINAL_IP=$(kubectl get pods -n "$NAMESPACE" -l "app=$TARGET" \
    -o jsonpath='{.items[0].status.podIP}' 2>/dev/null || echo "")

info "Final pod identity:"
info "  Name : $FINAL_POD"
info "  Node : $FINAL_NODE"
info "  IP   : $FINAL_IP"

# Wait for post observation
info "Waiting ${POST_SEC}s for post-recovery observation..."
sleep "$POST_SEC"

# ============================================================
# Phase 5: Post-recovery snapshot
# ============================================================
phase "5. Post-recovery"

snapshot_topology "$EXP_DIR/topology/post.json"

# ============================================================
# Verification
# ============================================================
phase "6. Verification"

VERIFIED=false
if [ "$FINAL_POD" != "$ORIG_POD" ]; then
    info "Pod identity changed: $ORIG_POD -> $FINAL_POD"
    VERIFIED=true
fi
if [ "$FINAL_IP" != "$ORIG_IP" ]; then
    info "Pod IP changed: $ORIG_IP -> $FINAL_IP"
    VERIFIED=true
fi
if [ "$FINAL_NODE" != "$ORIG_NODE" ]; then
    info "Node changed (drift): $ORIG_NODE -> $FINAL_NODE"
    VERIFIED=true
fi

if [ "$VERIFIED" = false ]; then
    warn "No topology change detected. Pod may not have been killed successfully."
fi

# ============================================================
# Write metadata & ground truth
# ============================================================
phase "7. Write metadata & ground truth"

write_metadata "$EXP_DIR" "$EXP_ID" "T2" "pod_drift" \
    "$TARGET" "pod_kill_reschedule" "instance_drift" \
    "$T_COLLECT" "$T_INJECT" "$T_RECOVER" "$T_END" \
    "{\"kill_count\":$KILL_COUNT,\"pod_before\":{\"name\":\"$ORIG_POD\",\"node\":\"$ORIG_NODE\",\"ip\":\"$ORIG_IP\"},\"pod_after\":{\"name\":\"$FINAL_POD\",\"node\":\"$FINAL_NODE\",\"ip\":\"$FINAL_IP\"},\"node_drifted\":$DRIFTED}"

AFFECTED=$(build_affected_services "$TARGET")

write_ground_truth "$EXP_DIR" "$(cat <<GT_EOF
{
    "experiment_id": "$EXP_ID",
    "root_cause_service": "$TARGET",
    "fault_type": "pod_kill_reschedule",
    "topology_change_type": "instance_drift",
    "fault_start_time": "$T_INJECT",
    "fault_end_time": "$T_RECOVER",
    "affected_services": $AFFECTED,
    "topology_changes": [
        {
            "change_type": "pod_replacement",
            "entity": "$TARGET",
            "before": {
                "pod_name": "$ORIG_POD",
                "node": "$ORIG_NODE",
                "pod_ip": "$ORIG_IP"
            },
            "after": {
                "pod_name": "$FINAL_POD",
                "node": "$FINAL_NODE",
                "pod_ip": "$FINAL_IP"
            },
            "node_drifted": $DRIFTED
        }
    ],
    "propagation_path": $(python3 -c "import json;print(json.dumps(['$TARGET'] + '$(get_upstream $TARGET)'.split(',')))"),
    "rca_challenge": "Service recovers automatically but pod identity (name, IP, possibly node) changes. RCA must detect the transient disruption window and correlate with topology change.",
    "labels": {
        "scenario": "T2",
        "is_dynamic_topology_fault": true,
        "topology_change_category": "instance_drift",
        "difficulty_for_rca": "low"
    }
}
GT_EOF
)"

# ============================================================
# Summary
# ============================================================
echo ""
info "=================================================================="
info "  T2 Experiment Complete: $EXP_ID"
info "=================================================================="
info "  Pod: $ORIG_POD -> $FINAL_POD"
info "  Node: $ORIG_NODE -> $FINAL_NODE (drifted=$DRIFTED)"
info "  IP: $ORIG_IP -> $FINAL_IP"
info "  Data dir: $EXP_DIR"
echo ""
