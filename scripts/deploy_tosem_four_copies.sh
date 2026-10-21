#!/usr/bin/env bash
# Deploy four isolated copies of both applications in a dedicated Kind cluster.
# The existing kind-fault-exp cluster and its formal collection batches are not
# touched.  Re-running this script is idempotent at the namespace/release level.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="${DYRCA_COPY_CLUSTER:-tosem-copies}"
KIND_NODE_IMAGE="${DYRCA_KIND_NODE_IMAGE:-dyrca/kind-node:v1.30.0-nofile}"
KUBECONFIG_PATH="${DYRCA_COPY_KUBECONFIG:-${PROJECT_DIR}/artifacts/tosem_four_copies/kubeconfig}"
WORK_DIR="${PROJECT_DIR}/artifacts/tosem_four_copies/manifests"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"
mkdir -p "$(dirname "$KUBECONFIG_PATH")" "$WORK_DIR"
exec 9>"$PROJECT_DIR/artifacts/tosem_four_copies/collection.lock"
flock -n 9 || { echo "A dedicated-copy collector is already running" >&2; exit 2; }

if [[ "$CLUSTER" != "tosem-copies" ]]; then
  echo "This deployment script only supports the declared cluster name tosem-copies" >&2
  exit 2
fi

if ! kind get clusters | grep -qx "$CLUSTER"; then
  inotify_limit="$(cat /proc/sys/fs/inotify/max_user_instances)"
  if (( inotify_limit < 1024 )); then
    echo "Host fs.inotify.max_user_instances=$inotify_limit is too low for four copies; set it to at least 8192 before deployment" >&2
    exit 2
  fi
  kind create cluster --name "$CLUSTER" --image "$KIND_NODE_IMAGE" --config "$PROJECT_DIR/config/tosem_four_copy_kind.yaml"
fi
kind export kubeconfig --name "$CLUSTER" --kubeconfig "$KUBECONFIG_PATH" >/dev/null
export KUBECONFIG="$KUBECONFIG_PATH"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

k() { kubectl "$@"; }
h() { helm "$@" --kubeconfig "$KUBECONFIG_PATH"; }

wait_ns_deployments() {
  local ns="$1"
  local timeout="${2:-1200s}"
  k wait --for=condition=available deployment --all -n "$ns" --timeout="$timeout" >/dev/null || {
    echo "[WARN] deployment readiness timeout in $ns" >&2
    k get deploy -n "$ns" -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas
    return 1
  }
}

install_istio() {
  if ! k get ns istio-system >/dev/null 2>&1; then
    istioctl install --kubeconfig "$KUBECONFIG_PATH" --context "kind-${CLUSTER}" --set profile=minimal -y
  fi
  k wait --for=condition=available deployment/istiod -n istio-system --timeout=600s >/dev/null
  "$PYTHON" "$PROJECT_DIR/scripts/configure_tosem_four_copy_telemetry.py" \
    --kubeconfig "$KUBECONFIG_PATH"
}

install_chaos_mesh() {
  k create namespace chaos-mesh --dry-run=client -o yaml | k apply -f - >/dev/null
  h repo add chaos-mesh https://charts.chaos-mesh.org >/dev/null 2>&1 || true
  h repo update >/dev/null
  h upgrade --install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh \
    --set chaosDaemon.runtime=containerd \
    --set chaosDaemon.socketPath=/run/containerd/containerd.sock \
    --set dashboard.create=true --set dashboard.securityMode=false --timeout 10m
  k wait --for=condition=available deployment --all -n chaos-mesh --timeout=600s >/dev/null
}

install_metrics_server() {
  local manifest="$WORK_DIR/metrics-server-v0.7.2.yaml"
  "$PYTHON" - "$manifest" <<'PY'
import pathlib
import sys
import requests
import yaml

path = pathlib.Path(sys.argv[1])
if not path.exists():
    response = requests.get("https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml", timeout=30)
    response.raise_for_status()
    docs = list(yaml.safe_load_all(response.text))
    for obj in docs:
        if obj and obj.get("kind") == "Deployment":
            container = obj["spec"]["template"]["spec"]["containers"][0]
            container.setdefault("args", []).append("--kubelet-insecure-tls")
            container["resources"] = {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"memory": "256Mi"}}
    path.write_text(yaml.safe_dump_all(docs))
PY
  k apply -f "$manifest" >/dev/null
  k wait --for=condition=available deployment/metrics-server -n kube-system --timeout=300s >/dev/null
}

install_observability() {
  k create namespace observability --dry-run=client -o yaml | k apply -f - >/dev/null
  h repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
  h repo add grafana https://grafana.github.io/helm-charts >/dev/null 2>&1 || true
  h repo update >/dev/null
  h upgrade --install prom prometheus-community/kube-prometheus-stack \
    -n observability -f "$PROJECT_DIR/config/tosem_four_copy_prometheus.yaml" --timeout 10m
  h upgrade --install loki grafana/loki-stack -n observability \
    --set loki.persistence.enabled=false --set grafana.enabled=false \
    --set prometheus.enabled=false --timeout 10m
  configure_promtail
  h upgrade --install tempo grafana/tempo -n observability \
    --set persistence.enabled=false --set metricsGenerator.enabled=false \
    --set resources.requests.cpu=100m --set resources.requests.memory=256Mi \
    --timeout 10m
  k wait --for=condition=available deployment --all -n observability --timeout=900s >/dev/null || true
}

configure_promtail() {
  local secret="$WORK_DIR/promtail.yaml"
  local encoded=""
  for _ in $(seq 1 60); do
    encoded="$(k get secret loki-promtail -n observability \
      -o jsonpath='{.data.promtail\.yaml}' 2>/dev/null || true)"
    if [[ -n "$encoded" ]]; then
      printf '%s' "$encoded" | base64 -d > "$secret"
      break
    fi
    sleep 2
  done
  [[ -s "$secret" ]] || { echo "Promtail Secret was not ready" >&2; return 1; }
  python3 - "$secret" <<'PY'
import pathlib
import sys
import yaml

path = pathlib.Path(sys.argv[1])
data = yaml.safe_load(path.read_text()) or {}
data["scrape_configs"] = [{
    "job_name": "dyrca-copies",
    "pipeline_stages": [{"cri": {}}],
    "static_configs": [{
        "targets": ["localhost"],
        "labels": {
            "job": "dyrca-copies",
            "namespace": namespace,
            "__path__": "/var/log/pods/%s_*/*/*.log" % namespace,
        },
    } for namespace in [
        "boutique-c1", "boutique-c2", "boutique-c3", "boutique-c4",
        "trainticket-c1", "trainticket-c2", "trainticket-c3", "trainticket-c4",
    ]],
}]
path.write_text(yaml.safe_dump(data, default_flow_style=False))
PY
  k create secret generic loki-promtail -n observability \
    --from-file=promtail.yaml="$secret" --dry-run=client -o yaml | k apply -f - >/dev/null
  k rollout restart daemonset/loki-promtail -n observability >/dev/null
  k rollout status daemonset/loki-promtail -n observability --timeout=60s >/dev/null || true
}

prepare_namespace() {
  local ns="$1" copy_id="$2"
  k create namespace "$ns" --dry-run=client -o yaml | k apply -f - >/dev/null
  local injection=enabled
  if [[ "$ns" == trainticket-* ]]; then
    injection="${DYRCA_TRAINTICKET_ISTIO:-disabled}"
  fi
  k label namespace "$ns" istio-injection="$injection" dyrca-copy-id="$copy_id" \
    dyrca-system="${ns%-c[1-4]}" --overwrite >/dev/null
}

patch_workloads() {
  local ns="$1" copy_id="$2"
  local deployments
  deployments="$(k get deploy -n "$ns" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')"
  while IFS= read -r deploy; do
    [[ -z "$deploy" ]] && continue
    k patch deploy "$deploy" -n "$ns" --type=merge \
      -p "{\"spec\":{\"template\":{\"metadata\":{\"labels\":{\"dyrca_copy_id\":\"$copy_id\"}}}}}" >/dev/null
    k set env deploy/"$deploy" -n "$ns" COLLECTOR_SERVICE_ADDR=tempo.observability:4317 >/dev/null || true
  done <<< "$deployments"
  local services
  services="$(k get svc -n "$ns" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')"
  while IFS= read -r service; do
    [[ -z "$service" ]] && continue
    k patch svc "$service" -n "$ns" --type=merge -p '{"spec":{"type":"ClusterIP"}}' >/dev/null || true
  done <<< "$services"
}

deploy_boutique_copy() {
  local idx="$1" ns="boutique-c$1" copy_id="boutique-c$1"
  prepare_namespace "$ns" "$copy_id"
  if ! h upgrade --install "boutique-c$idx" online-boutique/microservices-demo \
      -n "$ns" -f "$PROJECT_DIR/config/tosem_four_copy_boutique_values.yaml" --timeout 10m; then
    # The upstream chart index was removed; the pinned release manifest is the
    # same fallback used by infra/deploy-all.sh.
    k apply -n "$ns" -f \
      https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/v0.8.1/release/kubernetes-manifests.yaml
  fi
  sed -e "s/namespace: boutique/namespace: $ns/g" \
      -e 's#locustio/locust:2.29#locustio/locust:2.32.6#g' \
      "$PROJECT_DIR/infra/locust/deployment.yaml" \
    > "$WORK_DIR/${ns}-locust.yaml"
  k apply -f "$WORK_DIR/${ns}-locust.yaml" >/dev/null
  patch_workloads "$ns" "$copy_id"
  wait_ns_deployments "$ns" 1200s
  k scale deployment --all --replicas=0 -n "$ns" >/dev/null
  k wait --for=delete pod --all -n "$ns" --timeout=180s >/dev/null
}

deploy_trainticket_copy() {
  local idx="$1" ns="trainticket-c$1" copy_id="trainticket-c$1"
  prepare_namespace "$ns" "$copy_id"
  for part in part1 part2 part3; do
    sed -e "s/namespace: trainticket/namespace: $ns/g" \
        -e 's/image: mongo$/image: mongo:4.4/g' \
        -e '/^[[:space:]]*nodePort:/d' \
        "$PROJECT_DIR/infra/trainticket/.manifests/ts-deployment-${part}.yml" \
        > "$WORK_DIR/${ns}-${part}.yaml"
    k apply -f "$WORK_DIR/${ns}-${part}.yaml" >/dev/null
  done
  sed -e "s/namespace: trainticket/namespace: $ns/g" \
      -e 's/maxReplicas: 10/maxReplicas: 3/g' \
      -e '/^[[:space:]]*nodePort:/d' \
      "$PROJECT_DIR/infra/trainticket/hpa-trainticket.yaml" > "$WORK_DIR/${ns}-hpa.yaml"
  k apply -f "$WORK_DIR/${ns}-hpa.yaml" >/dev/null
  sed "s/namespace: trainticket/namespace: $ns/g" \
      "$PROJECT_DIR/infra/trainticket/locust/deployment.yaml" \
      > "$WORK_DIR/${ns}-locust.yaml"
  k apply -f "$WORK_DIR/${ns}-locust.yaml" >/dev/null
  patch_workloads "$ns" "$copy_id"
  # If a prior attempt created pods with Istio sidecars, recycle only those
  # pods so the disabled-injection policy takes effect on a retry.
  local sidecar_pods
  sidecar_pods="$(k get pods -n "$ns" -o json | python3 -c '
import json, sys
items = json.load(sys.stdin).get("items", [])
for pod in items:
    if len(pod.get("spec", {}).get("containers", [])) > 1:
        print(pod["metadata"]["name"])
')"
  if [[ -n "$sidecar_pods" ]]; then
    while IFS= read -r pod; do
      k delete pod "$pod" -n "$ns" --wait=false >/dev/null
    done <<< "$sidecar_pods"
  fi
  "$PYTHON" "$PROJECT_DIR/scripts/inject_tosem_trainticket_targets.py" \
    --kubeconfig "$KUBECONFIG_PATH" --namespace "$ns"
  wait_ns_deployments "$ns" 1800s
  k scale deployment --all --replicas=0 -n "$ns" >/dev/null
  k wait --for=delete pod --all -n "$ns" --timeout=180s >/dev/null
}

main() {
  docker update --memory 14g --memory-swap 14g \
    tosem-copies-worker tosem-copies-worker2 tosem-copies-worker3 \
    tosem-copies-worker4 tosem-copies-worker5 >/dev/null
  docker update --memory 3g --memory-swap 3g tosem-copies-control-plane >/dev/null
  install_istio
  install_observability
  install_chaos_mesh
  install_metrics_server
  h repo add online-boutique https://googlecloudplatform.github.io/microservices-demo >/dev/null 2>&1 || true
  h repo update >/dev/null 2>&1 || true
  for idx in 1 2 3 4; do
    deploy_boutique_copy "$idx"
  done
  for idx in 1 2 3 4; do
    deploy_trainticket_copy "$idx"
  done
  "$PYTHON" "$PROJECT_DIR/scripts/reset_tosem_copy_state.py" --kubeconfig "$KUBECONFIG_PATH"
  echo "[DONE] Four isolated copies per system are prepared in standby in cluster $CLUSTER"
  k get pods -A --no-headers | wc -l | awk '{print "pods=" $1}'
  k get ns -l dyrca-copy-id
}

main "$@"
