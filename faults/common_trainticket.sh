#!/bin/bash
# ============================================================
# faults/common_trainticket.sh — Shared functions for TrainTicket fault injection
# ============================================================
# Sourced by each T{1,2,3,4}_*_trainticket.sh script.
# Provides: experiment ID, topology snapshot, metadata, ground truth
# ============================================================

# Keep TrainTicket output isolated from the Online Boutique collection.  The
# scheduler passes an absolute DATA_DIR for batch runs; this default also makes
# direct invocations safe instead of silently mixing the two systems.
DATA_DIR="${DATA_DIR:-/home/user/dyrca/data/experiments_trainticket}"
NAMESPACE="${NAMESPACE:-trainticket}"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"

# --- Colours ---
info()  { echo -e "\033[0;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[0;31m[ERROR]\033[0m $*"; }
phase() { echo -e "\n\033[1;36m===== $1 =====\033[0m"; }

# --- Timestamp helpers ---
utc_ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
epoch_s() { date +%s; }

# --- Experiment ID ---
# Format: {scenario}_{target}_{variant}_run{NNN}
generate_exp_id() {
    local scenario="$1" target="$2" variant="${3:-default}"
    local run_id candidate base retry=0
    # A three-digit random suffix caused silent directory overwrites in the
    # earlier 400-case batch.  Prefer the scheduler's deterministic task ID;
    # direct calls use nanoseconds, and an existence check remains as a guard.
    run_id="${DYRCA_RUN_ID:-$(date -u +%Y%m%d%H%M%S%N)}"
    # IDs are also used as Kubernetes object names by T1/T2/T4.
    run_id=$(printf '%s' "$run_id" | sed 's/[^[:alnum:]-]/-/g' | tr '[:upper:]' '[:lower:]')
    base="${scenario}-${target}-${variant}-run${run_id}"
    candidate="$base"
    while [[ -e "$DATA_DIR/$candidate" ]]; do
        retry=$((retry + 1))
        candidate="${base}-retry${retry}"
    done
    echo "$candidate" | tr '[:upper:]' '[:lower:]'
}

# --- Create output directory tree ---
create_exp_dir() {
    local exp_id="$1"
    local d="$DATA_DIR/$exp_id"
    mkdir -p "$d"/{topology,metrics,logs,traces,k8s_events}
    echo "$d"
}

# --- Topology snapshot (Python for JSON reliability) ---
# Usage: snapshot_topology <output.json>
snapshot_topology() {
    local out="$1"
    python3 - "$out" <<'PYEOF'
import json, os, subprocess, sys
out = sys.argv[1]
namespace = os.environ.get("NAMESPACE", "trainticket")

def kube_json(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return json.loads(r.stdout) if r.stdout.strip() else {}

pods_raw   = kube_json(["kubectl","get","pods","-n",namespace,"-o","json"])
dep_raw    = kube_json(["kubectl","get","deployments","-n",namespace,"-o","json"])
ep_raw     = kube_json(["kubectl","get","endpoints","-n",namespace,"-o","json"])
hpa_raw    = kube_json(["kubectl","get","hpa","-n",namespace,"-o","json"])

pods = []
for p in pods_raw.get("items", []):
    pods.append({
        "name":    p["metadata"]["name"],
        "service": p["metadata"].get("labels", {}).get("app", "unknown"),
        "version": p["metadata"].get("labels", {}).get("version", "unknown"),
        "node":    p.get("spec", {}).get("nodeName", "unknown"),
        "ip":      p.get("status", {}).get("podIP", "unknown"),
        "status":  p.get("status", {}).get("phase", "unknown"),
    })

deployments = []
for d in dep_raw.get("items", []):
    deployments.append({
        "name":           d["metadata"]["name"],
        "replicas":       d["spec"]["replicas"],
        "readyReplicas":  d.get("status", {}).get("readyReplicas", 0),
        "version":        d["metadata"].get("labels", {}).get("version",
                           d["spec"]["template"]["metadata"].get("labels",{}).get("version","unknown")),
    })

services = []
for e in ep_raw.get("items", []):
    ips = []
    for s in e.get("subsets", []):
        for a in s.get("addresses", []):
            ips.append(a.get("ip", ""))
    services.append({"name": e["metadata"]["name"], "endpoint_ips": ips})

hpas = []
for h in hpa_raw.get("items", []):
    hpas.append({
        "name":             h["metadata"]["name"],
        "target":           h["spec"]["scaleTargetRef"]["name"],
        "currentReplicas":  h.get("status", {}).get("currentReplicas", 0),
        "desiredReplicas":  h.get("status", {}).get("desiredReplicas", 0),
    })

result = {
    "timestamp":    subprocess.getoutput("date -u +%Y-%m-%dT%H:%M:%SZ"),
    "pods":         pods,
    "deployments":  deployments,
    "services":     services,
    "hpas":         hpas,
}
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"  Snapshot saved: {out}  ({len(pods)} pods, {len(deployments)} deploys)")
PYEOF
}

# --- Write metadata.json ---
# Args: exp_dir, exp_id, scenario_id, scenario_name, target, fault_type,
#        topo_change_type, t_collect, t_inject, t_recover, t_end, [extra_json]
write_metadata() {
    local exp_dir="$1"   ; shift
    local exp_id="$1"    ; shift
    local scenario_id="$1"; shift
    local scenario_name="$1"; shift
    local target="$1"    ; shift
    local fault_type="$1"; shift
    local topo_chg="$1"  ; shift
    local t_collect="$1" ; shift
    local t_inject="$1"  ; shift
    local t_recover="$1" ; shift
    local t_end="$1"     ; shift
    local extra="${*:-\{\}}"

    python3 - "$exp_dir/metadata.json" "$exp_id" "$scenario_id" "$scenario_name" \
                          "$target" "$fault_type" "$topo_chg" \
                          "$t_collect" "$t_inject" "$t_recover" "$t_end" "$extra" <<PYEOF
import json, os, sys
out      = sys.argv[1]
meta = {
    "experiment_id":        sys.argv[2],
    "scenario_id":          sys.argv[3],
    "scenario_name":        sys.argv[4],
    "target_service":       sys.argv[5],
    "fault_type":           sys.argv[6],
    "topology_change_type": sys.argv[7],
    "time_window": {
        "t_collect": sys.argv[8],
        "t_inject":  sys.argv[9],
        "t_recover": sys.argv[10],
        "t_end":     sys.argv[11],
    },
    "environment": {
        "namespace": os.environ.get("NAMESPACE", "trainticket"),
        "cluster":   os.environ.get("DYRCA_CLUSTER", "kind-fault-exp"),
        "copy_id":   os.environ.get("DYRCA_COPY_ID", ""),
    },
}
extra = json.loads(sys.argv[12])
meta.update(extra)
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata saved: {out}")
PYEOF
}

# --- Write ground_truth.json ---
# Args: exp_dir  followed by an inline JSON string
write_ground_truth() {
    local exp_dir="$1"
    local gt_json="$2"
    python3 -c "import json,sys; json.dump(json.loads(sys.argv[1]),open(sys.argv[2]+'/ground_truth.json','w'),indent=2)" \
        "$gt_json" "$exp_dir"
    info "Ground truth saved: $exp_dir/ground_truth.json"
}

# --- Get upstream / downstream services (TrainTicket call graph) ---
get_upstream() {
    local svc="$1"
    case "$svc" in
        ts-station-service)       echo "ts-basic-service,ts-order-service,ts-order-other-service,ts-preserve-service,ts-preserve-other-service,ts-rebook-service,ts-food-service,ts-travel-plan-service,ts-route-plan-service,ts-preserve-service" ;;
        ts-basic-service)         echo "ts-ticketinfo-service" ;;
        ts-ticketinfo-service)    echo "ts-travel-service,ts-travel2-service,ts-preserve-service,ts-preserve-other-service,ts-rebook-service,ts-travel-plan-service" ;;
        ts-travel-service)        echo "ts-preserve-service,ts-rebook-service,ts-food-service,ts-route-plan-service,ts-travel-plan-service,ts-seat-service" ;;
        ts-travel2-service)       echo "ts-preserve-other-service,ts-rebook-service,ts-route-plan-service,ts-travel-plan-service,ts-seat-service" ;;
        ts-order-service)         echo "ts-travel-service,ts-preserve-service,ts-rebook-service,ts-security-service,ts-execute-service,ts-inside-payment-service,ts-voucher-service,ts-cancel-service" ;;
        ts-order-other-service)   echo "ts-travel2-service,ts-preserve-other-service,ts-rebook-service,ts-security-service,ts-execute-service,ts-inside-payment-service,ts-voucher-service,ts-cancel-service" ;;
        ts-seat-service)          echo "ts-travel-service,ts-travel2-service,ts-preserve-service,ts-preserve-other-service,ts-rebook-service,ts-travel-plan-service" ;;
        ts-preserve-service)      echo "" ;;
        ts-preserve-other-service) echo "" ;;
        ts-food-service)          echo "ts-preserve-service,ts-preserve-other-service,ts-rebook-service" ;;
        ts-consign-service)       echo "ts-preserve-service,ts-preserve-other-service,ts-rebook-service" ;;
        *) echo "" ;;
    esac
}

get_downstream() {
    local svc="$1"
    case "$svc" in
        ts-basic-service)         echo "ts-station-service,ts-train-service,ts-route-service,ts-price-service" ;;
        ts-ticketinfo-service)    echo "ts-basic-service" ;;
        ts-travel-service)        echo "ts-ticketinfo-service,ts-order-service,ts-train-service,ts-route-service,ts-seat-service" ;;
        ts-travel2-service)       echo "ts-ticketinfo-service,ts-order-other-service,ts-train-service,ts-route-service,ts-seat-service" ;;
        ts-preserve-service)      echo "ts-ticketinfo-service,ts-seat-service,ts-notification-service,ts-user-service,ts-assurance-service,ts-station-service,ts-security-service,ts-travel-service,ts-contacts-service,ts-order-service,ts-food-service,ts-consign-service" ;;
        ts-preserve-other-service) echo "ts-ticketinfo-service,ts-seat-service,ts-notification-service,ts-user-service,ts-assurance-service,ts-station-service,ts-security-service,ts-travel2-service,ts-contacts-service,ts-order-other-service,ts-food-service,ts-consign-service" ;;
        ts-order-service)         echo "ts-station-service" ;;
        ts-order-other-service)   echo "ts-station-service" ;;
        ts-seat-service)          echo "ts-travel-service,ts-order-service,ts-train-service,ts-config-service" ;;
        *) echo "" ;;
    esac
}

# --- Build affected_services list for ground truth ---
# Args: root_cause_service
# Outputs JSON array of affected service objects
build_affected_services() {
    local root="$1"
    local upstream downstream
    upstream=$(get_upstream "$root")
    downstream=$(get_downstream "$root")

    python3 - "$root" "$upstream" "$downstream" <<PYEOF
import json, sys
root = sys.argv[1]
upstream = sys.argv[2].split(",") if sys.argv[2] else []
downstream = sys.argv[3].split(",") if sys.argv[3] else []

services = [{"service": root, "role": "root_cause", "impact_level": "critical"}]
for svc in downstream:
    services.append({"service": svc, "role": "propagated", "impact_level": "warning"})
for svc in upstream:
    if svc not in [s["service"] for s in services]:
        services.append({"service": svc, "role": "propagated", "impact_level": "warning"})
print(json.dumps(services))
PYEOF
}

# --- Wait for all trainticket pods to be Running ---
wait_healthy() {
    local timeout="${1:-120}"
    local waited=0
    info "Waiting for all trainticket pods to be Running (max ${timeout}s)..."
    while [ $waited -lt $timeout ]; do
        local nr
        nr=$(kubectl get pods -n "$NAMESPACE" -l app --no-headers 2>/dev/null \
             | grep -v "Running" | grep -c . || true)
        nr=$(echo "$nr" | tr -d '[:space:]')
        if [ "$nr" = "0" ] || [ -z "$nr" ]; then
            info "All pods healthy"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    warn "Some pods still not Running after ${timeout}s"
    kubectl get pods -n "$NAMESPACE" | grep -v Running || true
    return 1
}

# --- Verify a Chaos Mesh resource exists and is running ---
wait_replica_recovery() {
    local target="$1" expected="$2" timeout="${3:-180}" started=$SECONDS
    local desired ready values
    while (( SECONDS - started < timeout )); do
        values=$(kubectl --request-timeout=10s get deployment "$target" -n "$NAMESPACE" \
            -o jsonpath='{.spec.replicas} {.status.readyReplicas}' 2>/dev/null || true)
        read -r desired ready <<< "$values"
        if [[ "$desired" == "$expected" && "$ready" == "$expected" ]]; then
            info "Replica recovery verified for $target: $expected"
            return 0
        fi
        sleep 5
    done
    warn "Replica recovery is not yet verified for $target"
    return 1
}

verify_chaos() {
    local kind="$1" name="$2" ns="${3:-${NAMESPACE:-trainticket}}"
    info "Verifying $kind/$name ..."
    if kubectl get "$kind" "$name" -n "$ns" &>/dev/null; then
        info "  $kind/$name exists"
        return 0
    else
        err "  $kind/$name NOT found"
        return 1
    fi
}
