#!/bin/bash
# ============================================================
# T3 — Service Selector Label Poisoning (TrainTicket)
# ============================================================
# Principle: kubectl patch modifies the Service selector, causing traffic
#            to route to the wrong Pods. Topology dependency edge endpoint
#            is tampered with.
# Recovery: Restore original selector, wait for Endpoints to rebuild.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_trainticket.sh
source "$SCRIPT_DIR/common_trainticket.sh"

# --- Defaults ---
# Variant A: selector redirected to another service (edge_redirect)
# Variant B: selector set to empty label (edge_removal)
TARGET="${1:-ts-basic-service}"               # Service to poison
VARIANT="${2:-redirect}"                      # redirect | empty
POISON_TARGET="${3:-ts-station-service}"      # redirect variant: redirect to this service's Pods
DURING_SEC="${4:-120}"                        # Fault duration (seconds)
POST_SEC="${5:-60}"                           # Recovery observation (seconds)
PRE_SEC="${6:-30}"                            # Baseline collection (seconds)

EXP_ID=$(generate_exp_id "T3" "$TARGET" "${VARIANT}")
EXP_DIR=$(create_exp_dir "$EXP_ID")

info "=================================================================="
info "  T3: Service Selector Label Poisoning (TrainTicket)"
info "=================================================================="
info "  Experiment ID : $EXP_ID"
info "  Target service: $TARGET"
info "  Variant       : $VARIANT"
[ "$VARIANT" = "redirect" ] && info "  Redirect to   : $POISON_TARGET"
info "  Duration      : pre=${PRE_SEC}s during=${DURING_SEC}s post=${POST_SEC}s"
info "  Output dir    : $EXP_DIR"
echo ""

# ============================================================
# Pre-check: save original selector
# ============================================================
phase "0. Pre-check"

# Save the ORIGINAL selector for restoration
ORIG_SELECTOR_JSON=$(kubectl get svc "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.selector}' 2>/dev/null)
ORIG_SELECTOR_APP=$(echo "$ORIG_SELECTOR_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('app',''))" 2>/dev/null)

if [ -z "$ORIG_SELECTOR_APP" ]; then
    err "Cannot determine current selector for svc/$TARGET. Aborting."
    exit 1
fi

info "Original selector: app=$ORIG_SELECTOR_APP"

# Record original endpoints
ORIG_ENDPOINTS=$(kubectl get endpoints "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null || echo "")
info "Original endpoints: $ORIG_ENDPOINTS"

# Compute the poisoned selector
if [ "$VARIANT" = "redirect" ]; then
    POISON_SELECTOR_JSON="{\"app\":\"$POISON_TARGET\"}"
    TOPO_CHANGE="edge_redirect"
    FAULT_DESC="selector_redirected_to_${POISON_TARGET}"
elif [ "$VARIANT" = "empty" ]; then
    POISON_SELECTOR_JSON="{}"
    TOPO_CHANGE="edge_removal"
    FAULT_DESC="selector_emptied_no_endpoints"
else
    err "Unknown variant: $VARIANT (use 'redirect' or 'empty')"
    exit 1
fi

# ============================================================
# Cleanup trap: always restore selector on exit
# ============================================================
cleanup() {
    if [ -f "$EXP_DIR/.injected" ]; then
        info "Cleanup: restoring original selector..."
        kubectl patch svc "$TARGET" -n "$NAMESPACE" \
            -p "{\"spec\":{\"selector\":{\"app\":\"$ORIG_SELECTOR_APP\"}}}" 2>/dev/null || true
        rm -f "$EXP_DIR/.injected"
        sleep 5
        info "Selector restored."
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
# Phase 2: Inject -- patch Service selector
# ============================================================
phase "2. Inject Fault"

T_INJECT=$(utc_ts)
info "Patching svc/$TARGET selector: app=$ORIG_SELECTOR_APP -> $(echo $POISON_SELECTOR_JSON | python3 -c 'import json,sys;print(json.load(sys.stdin))')"

kubectl patch svc "$TARGET" -n "$NAMESPACE" \
    -p "{\"spec\":{\"selector\":$POISON_SELECTOR_JSON}}"

touch "$EXP_DIR/.injected"

# Wait for Endpoints to update
info "Waiting for Endpoints to update..."
sleep 10

# Verify: check new endpoints
NEW_ENDPOINTS=$(kubectl get endpoints "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null || echo "(none)")

if [ "$VARIANT" = "empty" ]; then
    if [ -z "$NEW_ENDPOINTS" ]; then
        info "No endpoints (expected for 'empty' variant)"
    else
        warn "Endpoints still exist: $NEW_ENDPOINTS (may take more time)"
    fi
elif [ "$VARIANT" = "redirect" ]; then
    # Verify the new endpoints belong to POISON_TARGET
    POISON_POD_IPS=$(kubectl get pods -n "$NAMESPACE" -l "app=$POISON_TARGET" \
        -o jsonpath='{.items[*].status.podIP}' 2>/dev/null || echo "")
    info "New endpoints   : $NEW_ENDPOINTS"
    info "$POISON_TARGET IPs: $POISON_POD_IPS"
    if echo "$NEW_ENDPOINTS" | grep -q "$(echo $POISON_POD_IPS | awk '{print $1}')"; then
        info "Endpoints now point to $POISON_TARGET pods"
    else
        warn "Endpoints don't match expected $POISON_TARGET pods. They may still be updating."
    fi
fi

# ============================================================
# Phase 3: Observation
# ============================================================
phase "3. Observation ($DURING_SEC s)"

snapshot_topology "$EXP_DIR/topology/during.json"

info "Fault active. Observing for ${DURING_SEC}s..."
info "  -> Upstream services will see errors calling $TARGET"
info "  -> $TARGET pods are orphaned (no traffic)"
[ "$VARIANT" = "redirect" ] && info "  -> $POISON_TARGET pods receive unexpected HTTP calls"

sleep "$DURING_SEC"

# ============================================================
# Phase 4: Recover
# ============================================================
phase "4. Recover"

T_RECOVER=$(utc_ts)
info "Restoring original selector: app=$ORIG_SELECTOR_APP"
kubectl patch svc "$TARGET" -n "$NAMESPACE" \
    -p "{\"spec\":{\"selector\":{\"app\":\"$ORIG_SELECTOR_APP\"}}}"
rm -f "$EXP_DIR/.injected"

info "Waiting for Endpoints to rebuild..."
sleep 15

RESTORED_ENDPOINTS=$(kubectl get endpoints "$TARGET" -n "$NAMESPACE" \
    -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null || echo "")
info "Restored endpoints: $RESTORED_ENDPOINTS"

info "Waiting ${POST_SEC}s for recovery observation..."
sleep "$POST_SEC"

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

if [ "$RESTORED_ENDPOINTS" = "$ORIG_ENDPOINTS" ] || \
   echo "$RESTORED_ENDPOINTS" | grep -q "$(echo $ORIG_ENDPOINTS | awk '{print $1}')"; then
    info "Topology restored: endpoints match original"
else
    warn "Endpoints differ from original (pods may have restarted): original=$ORIG_ENDPOINTS restored=$RESTORED_ENDPOINTS"
fi

# ============================================================
# Write metadata & ground truth
# ============================================================
phase "7. Write metadata & ground truth"

write_metadata "$EXP_DIR" "$EXP_ID" "T3" "selector_poison" \
    "$TARGET" "$FAULT_DESC" "$TOPO_CHANGE" \
    "$T_COLLECT" "$T_INJECT" "$T_RECOVER" "$T_END" \
    "{\"original_selector\":{\"app\":\"$ORIG_SELECTOR_APP\"},\"injected_selector\":$POISON_SELECTOR_JSON}"

AFFECTED=$(build_affected_services "$TARGET")

# Build topology_changes detail
if [ "$VARIANT" = "redirect" ]; then
    EDGE_DETAIL="{\"change_type\":\"edge_redirect\",\"from_services\":\"$(get_upstream $TARGET)\",\"to_declared\":\"$TARGET\",\"to_actual\":\"$POISON_TARGET\",\"before_healthy\":true,\"after_healthy\":false}"
else
    EDGE_DETAIL="{\"change_type\":\"edge_removal\",\"from_services\":\"$(get_upstream $TARGET)\",\"to_declared\":\"$TARGET\",\"to_actual\":\"(none)\",\"before_healthy\":true,\"after_healthy\":false}"
fi

write_ground_truth "$EXP_DIR" "$(cat <<GT_EOF
{
    "experiment_id": "$EXP_ID",
    "root_cause_service": "$TARGET",
    "fault_type": "$FAULT_DESC",
    "topology_change_type": "$TOPO_CHANGE",
    "fault_start_time": "$T_INJECT",
    "fault_end_time": "$T_RECOVER",
    "affected_services": $AFFECTED,
    "topology_changes": [$EDGE_DETAIL],
    "propagation_path": $(python3 -c "import json;print(json.dumps(['${TARGET}_selector'] + '$(get_upstream $TARGET)'.split(',')))"),
    "rca_challenge": "Root cause is a dependency edge misconfiguration, not a service failure. Static topology looks normal; only runtime trace analysis reveals destination mismatch.",
    "labels": {
        "scenario": "T3",
        "is_dynamic_topology_fault": true,
        "topology_change_category": "edge_change",
        "difficulty_for_rca": "high"
    }
}
GT_EOF
)"

# ============================================================
# Summary
# ============================================================
echo ""
info "=================================================================="
info "  T3 Experiment Complete: $EXP_ID"
info "=================================================================="
info "  Selector: app=$ORIG_SELECTOR_APP -> $(echo $POISON_SELECTOR_JSON | tr -d '"{}') -> app=$ORIG_SELECTOR_APP"
info "  Endpoints: $ORIG_ENDPOINTS -> $NEW_ENDPOINTS -> $RESTORED_ENDPOINTS"
info "  Data dir: $EXP_DIR"
echo ""
