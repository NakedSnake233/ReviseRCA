#!/usr/bin/env bash
# Counterfactual dynamic-topology controls.
#
# The original T1--T4 scripts change the same service that is written as the
# root cause.  That is useful for exercising graph updates, but it also makes
# the root cause recoverable from the changed-service set alone.  This script
# breaks that shortcut: the root-cause service receives a network delay (no
# pod/service topology change), while an unrelated decoy service receives the
# dynamic topology perturbation.
#
# Modes:
#   scale    add a short-lived extra deployment for one decoy service
#   drift    kill one decoy pod and let its deployment recreate it
#   redirect redirect one decoy Service to another decoy's pods
#   split    add a second-version deployment for one decoy service
#   multi    add extra deployments for two decoy services
#   benign   dynamic decoy change without a root-cause fault
#
# Positional arguments:
#   target mode decoy decoy2 during_sec post_sec pre_sec delay_ms

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-boutique}"
PLATFORM="$NAMESPACE"
case "$PLATFORM" in
    boutique-c[1-4]) PLATFORM=boutique ;;
    trainticket-c[1-4]) PLATFORM=trainticket ;;
esac
if [[ "$PLATFORM" == "trainticket" ]]; then
    # shellcheck source=common_trainticket.sh
    source "$SCRIPT_DIR/common_trainticket.sh"
else
    # shellcheck source=common.sh
    source "$SCRIPT_DIR/common.sh"
fi

TARGET="${1:?target is required}"
MODE="${2:-scale}"
DECOY="${3:-auto}"
DECOY2="${4:-auto}"
DURING_SEC="${5:-60}"
POST_SEC="${6:-30}"
PRE_SEC="${7:-30}"
DELAY_MS="${8:-150}"

case "$MODE" in
    scale)    SCENARIO_ID="C1"; TOPOLOGY_TYPE="counterfactual_node_addition" ;;
    drift)    SCENARIO_ID="C2"; TOPOLOGY_TYPE="counterfactual_instance_drift" ;;
    redirect) SCENARIO_ID="C3"; TOPOLOGY_TYPE="counterfactual_edge_redirect" ;;
    split)    SCENARIO_ID="C4"; TOPOLOGY_TYPE="counterfactual_version_split" ;;
    multi)    SCENARIO_ID="C5"; TOPOLOGY_TYPE="counterfactual_multi_change" ;;
    benign)   SCENARIO_ID="C6"; TOPOLOGY_TYPE="benign_dynamic_change" ;;
    *) echo "Unknown control mode: $MODE" >&2; exit 2 ;;
esac

case "$PLATFORM:$TARGET" in
  # adservice is part of the canonical Online Boutique graph, but is not
  # deployed in the current kind snapshot; use only live deployments here.
  boutique:productcatalogservice) DEFAULT_DECOY="emailservice"; DEFAULT_DECOY2="shippingservice" ;;
  boutique:cartservice)           DEFAULT_DECOY="recommendationservice"; DEFAULT_DECOY2="emailservice" ;;
  boutique:checkoutservice)       DEFAULT_DECOY="recommendationservice"; DEFAULT_DECOY2="emailservice" ;;
  boutique:recommendationservice) DEFAULT_DECOY="shippingservice"; DEFAULT_DECOY2="emailservice" ;;
  boutique:currencyservice)       DEFAULT_DECOY="emailservice"; DEFAULT_DECOY2="shippingservice" ;;
  trainticket:ts-travel-service)       DEFAULT_DECOY="ts-contacts-service"; DEFAULT_DECOY2="ts-config-service" ;;
  trainticket:ts-travel2-service)      DEFAULT_DECOY="ts-route-plan-service"; DEFAULT_DECOY2="ts-config-service" ;;
  trainticket:ts-order-service)        DEFAULT_DECOY="ts-food-map-service"; DEFAULT_DECOY2="ts-route-plan-service" ;;
  trainticket:ts-order-other-service)  DEFAULT_DECOY="ts-route-plan-service"; DEFAULT_DECOY2="ts-food-map-service" ;;
  trainticket:ts-preserve-service)     DEFAULT_DECOY="ts-auth-service"; DEFAULT_DECOY2="ts-config-service" ;;
  *)
    # Fallback for a manually selected target: use the first two deployments
    # that are not the target.  The fixed mappings above are preferred because
    # they avoid a direct target/decoy call-graph relationship.
    mapfile -t _SERVICES < <(kubectl get deploy -n "$NAMESPACE" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort)
    DEFAULT_DECOY=""
    DEFAULT_DECOY2=""
    for _svc in "${_SERVICES[@]}"; do
        [[ "$_svc" == "$TARGET" ]] && continue
        [[ "$_svc" == *mongo* || "$_svc" == *mysql* || "$_svc" == redis-* ]] && continue
        if [[ -z "$DEFAULT_DECOY" ]]; then DEFAULT_DECOY="$_svc"; continue; fi
        DEFAULT_DECOY2="$_svc"; break
    done
    ;;
esac

[[ "$DECOY" == auto || -z "$DECOY" ]] && DECOY="$DEFAULT_DECOY"
[[ "$DECOY2" == auto || -z "$DECOY2" ]] && DECOY2="$DEFAULT_DECOY2"
[[ -n "$DECOY" && "$DECOY" != "$TARGET" ]] || { echo "Invalid decoy=$DECOY" >&2; exit 2; }
[[ -n "$DECOY2" && "$DECOY2" != "$TARGET" && "$DECOY2" != "$DECOY" ]] || {
    [[ "$MODE" == redirect || "$MODE" == multi ]] && { echo "Invalid second decoy=$DECOY2" >&2; exit 2; } || true
}

EXP_ID="$(generate_exp_id "$SCENARIO_ID" "$TARGET" "${MODE}-${DECOY}")"
EXP_DIR="$(create_exp_dir "$EXP_ID")"
SHORT_HASH="$(printf '%s' "$EXP_ID" | sha256sum | cut -c1-10)"
ROOT_CHAOS="cf-${SCENARIO_ID,,}-${SHORT_HASH}-root"
DECOY_CHAOS="cf-${SCENARIO_ID,,}-${SHORT_HASH}-decoy"
DECOY_CHAOS2="cf-${SCENARIO_ID,,}-${SHORT_HASH}-decoy2"
EXTRA1="${DECOY}-cf-${SCENARIO_ID,,}-${SHORT_HASH}"
EXTRA2="${DECOY2}-cf-${SCENARIO_ID,,}-${SHORT_HASH}"

info "=================================================================="
info "  Counterfactual control: $EXP_ID"
info "  namespace=$NAMESPACE target=$TARGET mode=$MODE decoy=$DECOY decoy2=$DECOY2"
info "  duration: pre=${PRE_SEC}s during=${DURING_SEC}s post=${POST_SEC}s"
info "  output: $EXP_DIR"
# Keep the scheduler's stable output contract used by the original T1--T4
# collectors so exporter/verification are never skipped for a valid case.
info "  Experiment ID : $EXP_ID"
info "  Output dir    : $EXP_DIR"
info "=================================================================="

ORIG_SELECTOR_APP=""
EXTRA1_CREATED=false
EXTRA2_CREATED=false
REDIRECTED=false
ROOT_CREATED=false

cleanup() {
    set +e
    [[ "$ROOT_CREATED" == true ]] && kubectl delete networkchaos "$ROOT_CHAOS" -n "$NAMESPACE" --ignore-not-found=true --wait=false >/dev/null 2>&1
    kubectl delete podchaos "$DECOY_CHAOS" -n "$NAMESPACE" --ignore-not-found=true --wait=false >/dev/null 2>&1
    kubectl delete podchaos "$DECOY_CHAOS2" -n "$NAMESPACE" --ignore-not-found=true --wait=false >/dev/null 2>&1
    if [[ "$REDIRECTED" == true && -n "$ORIG_SELECTOR_APP" ]]; then
        kubectl patch svc "$DECOY" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"$ORIG_SELECTOR_APP\"}}}" >/dev/null 2>&1
    fi
    [[ "$EXTRA1_CREATED" == true ]] && kubectl delete deployment "$EXTRA1" -n "$NAMESPACE" --ignore-not-found=true --wait=false >/dev/null 2>&1
    [[ "$EXTRA2_CREATED" == true ]] && kubectl delete deployment "$EXTRA2" -n "$NAMESPACE" --ignore-not-found=true --wait=false >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

deployment_image_and_port() {
    local dep="$1"
    IMAGE="$(kubectl get deploy "$dep" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')"
    CONTAINER_NAME="$(kubectl get deploy "$dep" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].name}')"
    [[ -n "$CONTAINER_NAME" ]] || CONTAINER_NAME="server"
}

create_extra_deployment() {
    local base="$1" name="$2" version="$3"
    info "Creating temporary topology-only deployment $name from $base"
    python3 - "$NAMESPACE" "$base" "$name" "$version" <<'PY' | kubectl apply -f -
import json, subprocess, sys
ns, base, name, version = sys.argv[1:]
obj = json.loads(subprocess.check_output(
    ["kubectl", "get", "deployment", base, "-n", ns, "-o", "json"], text=True))
meta = obj.get("metadata", {})
spec = obj.get("spec", {})
template = spec.get("template", {})
labels = dict(template.get("metadata", {}).get("labels", {}) or {})
app = labels.get("app", base)
labels.update({"app": app, "version": version, "dyrca-control": "true"})
template.setdefault("metadata", {})["labels"] = labels
# The copied deployments often use a long startup/readiness delay (up to
# several minutes).  A control deployment only supplies a topology node; it
# must become Ready quickly enough for the during snapshot and should not
# change application behavior through a second health-check lifecycle.
for container in template.get("spec", {}).get("containers", []):
    for probe in ("readinessProbe", "livenessProbe", "startupProbe"):
        container.pop(probe, None)
spec["replicas"] = 1
spec["selector"] = {"matchLabels": {"app": app, "version": version}}
spec["template"] = template
clean_meta = {"name": name, "namespace": ns, "labels": {"app": app, "dyrca-control": "true"}}
out = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": clean_meta, "spec": spec}
print(json.dumps(out))
PY
    kubectl rollout status deployment "$name" -n "$NAMESPACE" --timeout=120s
}

apply_root_fault() {
    [[ "$MODE" == benign ]] && return 0
    info "Injecting topology-neutral NetworkChaos on target $TARGET"
    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: ${ROOT_CHAOS}
  namespace: ${NAMESPACE}
spec:
  action: delay
  mode: all
  selector:
    namespaces: ["${NAMESPACE}"]
    labelSelectors:
      app: "${TARGET}"
  target:
    mode: all
    selector:
      namespaces: ["${NAMESPACE}"]
      expressionSelectors:
      - key: app
        operator: NotIn
        values: ["loadgenerator"]
  delay:
    latency: "${DELAY_MS}ms"
    jitter: "10ms"
  direction: both
  duration: "${DURING_SEC}s"
EOF
    ROOT_CREATED=true
    sleep 3
}

apply_decoy_change() {
    case "$MODE" in
      scale)
        create_extra_deployment "$DECOY" "$EXTRA1" "cf-scale"
        EXTRA1_CREATED=true
        ;;
      drift)
        info "Killing one pod of unrelated decoy $DECOY"
        kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: ${DECOY_CHAOS}
  namespace: ${NAMESPACE}
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces: ["${NAMESPACE}"]
    labelSelectors:
      app: "${DECOY}"
  gracePeriod: 0
EOF
        sleep 8
        kubectl delete podchaos "$DECOY_CHAOS" -n "$NAMESPACE" --ignore-not-found=true --wait=false >/dev/null 2>&1
        ;;
      redirect)
        ORIG_SELECTOR_APP="$(kubectl get svc "$DECOY" -n "$NAMESPACE" -o jsonpath='{.spec.selector.app}')"
        [[ -n "$ORIG_SELECTOR_APP" ]] || { echo "Cannot read selector for $DECOY" >&2; return 1; }
        info "Redirecting unrelated service $DECOY to decoy $DECOY2"
        kubectl patch svc "$DECOY" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"$DECOY2\"}}}"
        REDIRECTED=true
        sleep 10
        ;;
      split)
        create_extra_deployment "$DECOY" "$EXTRA1" "v2.0.0"
        EXTRA1_CREATED=true
        ;;
      multi)
        create_extra_deployment "$DECOY" "$EXTRA1" "cf-multi-a"
        EXTRA1_CREATED=true
        create_extra_deployment "$DECOY2" "$EXTRA2" "cf-multi-b"
        EXTRA2_CREATED=true
        ;;
      benign)
        create_extra_deployment "$DECOY" "$EXTRA1" "cf-benign"
        EXTRA1_CREATED=true
        ;;
    esac
}

wait_temporary_cleanup() {
    if [[ "$EXTRA1_CREATED" == true ]]; then
        kubectl wait --for=delete "deployment/$EXTRA1" -n "$NAMESPACE" --timeout=60s >/dev/null 2>&1 || true
    fi
    if [[ "$EXTRA2_CREATED" == true ]]; then
        kubectl wait --for=delete "deployment/$EXTRA2" -n "$NAMESPACE" --timeout=60s >/dev/null 2>&1 || true
    fi
    sleep 5
}

service_changed_between_snapshots() {
    local service="$1"
    python3 - "$EXP_DIR/topology/pre.json" "$EXP_DIR/topology/during.json" "$service" <<'PY'
import json, sys
pre_path, during_path, service = sys.argv[1:]

def signature(snapshot):
    pods = [p for p in snapshot.get("pods", []) if p.get("service") == service]
    pod_sig = sorted((p.get("name"), p.get("version"), p.get("node"), p.get("ip"), p.get("status")) for p in pods)
    deployments = [d for d in snapshot.get("deployments", []) if d.get("name") == service]
    dep_sig = sorted((d.get("name"), d.get("replicas"), d.get("readyReplicas"), d.get("version")) for d in deployments)
    endpoints = [s for s in snapshot.get("services", []) if s.get("name") == service]
    ep_sig = sorted((s.get("name"), tuple(sorted(s.get("endpoint_ips", [])))) for s in endpoints)
    return pod_sig, dep_sig, ep_sig

pre = json.load(open(pre_path, encoding="utf-8"))
during = json.load(open(during_path, encoding="utf-8"))
print("1" if signature(pre) != signature(during) else "0")
PY
}

phase "1. Baseline"
T_COLLECT="$(utc_ts)"
snapshot_topology "$EXP_DIR/topology/pre.json"
sleep "$PRE_SEC"

phase "2. Inject target-neutral fault and unrelated topology change"
T_INJECT="$(utc_ts)"
apply_root_fault
apply_decoy_change

phase "3. During observation"
snapshot_topology "$EXP_DIR/topology/during.json"
# Enforce the counterfactual invariant at collection time. A background HPA
# or an indirect side effect that changes the labelled target invalidates the
# case instead of silently reintroducing the shortcut into the dataset.
TARGET_CHANGED="$(service_changed_between_snapshots "$TARGET")"
DECOY_CHANGED="$(service_changed_between_snapshots "$DECOY")"
if [[ "$TARGET_CHANGED" == "1" || "$DECOY_CHANGED" != "1" ]]; then
    warn "Invalid counterfactual: target_changed=$TARGET_CHANGED decoy_changed=$DECOY_CHANGED"
    cleanup
    wait_temporary_cleanup
    exit 3
fi
sleep "$DURING_SEC"

phase "4. Recovery"
T_RECOVER="$(utc_ts)"
# Cleanup is idempotent and also runs from the EXIT trap.
cleanup
wait_temporary_cleanup
ROOT_CREATED=false
EXTRA1_CREATED=false
EXTRA2_CREATED=false
REDIRECTED=false
sleep "$POST_SEC"

phase "5. Post snapshot"
T_END="$(utc_ts)"
snapshot_topology "$EXP_DIR/topology/post.json"

if [[ "$MODE" == benign ]]; then
    FAULT_TYPE="benign_dynamic_control"
    ROOT_CAUSE="__benign__"
    AFFECTED='[]'
    IS_BENIGN=true
else
    FAULT_TYPE="network_delay_target_with_decoy_${MODE}"
    ROOT_CAUSE="$TARGET"
    AFFECTED="$(build_affected_services "$TARGET")"
    IS_BENIGN=false
fi

write_metadata "$EXP_DIR" "$EXP_ID" "$SCENARIO_ID" "counterfactual_${MODE}" \
    "$TARGET" "$FAULT_TYPE" "$TOPOLOGY_TYPE" \
    "$T_COLLECT" "$T_INJECT" "$T_RECOVER" "$T_END" \
    "{\"control_family\":\"decoy_dynamic\",\"control_mode\":\"$MODE\",\"counterfactual\":true,\"is_benign\":$IS_BENIGN,\"topology_target_service\":\"$DECOY\",\"topology_target_service_2\":\"$DECOY2\",\"root_fault_topology_neutral\":true,\"fault_params\":{\"delay_ms\":$DELAY_MS}}"

write_ground_truth "$EXP_DIR" "$(cat <<GT_EOF
{
  "experiment_id": "$EXP_ID",
  "root_cause_service": "$ROOT_CAUSE",
  "fault_type": "$FAULT_TYPE",
  "topology_change_type": "$TOPOLOGY_TYPE",
  "fault_start_time": "$T_INJECT",
  "fault_end_time": "$T_RECOVER",
  "affected_services": $AFFECTED,
  "topology_changes": [
    {"change_type": "$TOPOLOGY_TYPE", "entity": "$DECOY", "decoy": true}
  ],
  "labels": {
    "scenario": "$SCENARIO_ID",
    "is_dynamic_topology_fault": true,
    "counterfactual_control": true,
    "is_benign": $IS_BENIGN,
    "label_type": "counterfactual"
  }
}
GT_EOF
)"

info "Counterfactual case complete: $EXP_DIR"
info "  root_cause=$ROOT_CAUSE topology_decoy=$DECOY mode=$MODE"
