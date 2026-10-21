"""Topology and trace signal extraction for rule-based graph planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


KNOWN_EDGES = [
    ("frontend", "productcatalogservice"),
    ("frontend", "cartservice"),
    ("frontend", "recommendationservice"),
    ("frontend", "currencyservice"),
    ("frontend", "adservice"),
    ("frontend", "checkoutservice"),
    ("recommendationservice", "productcatalogservice"),
    ("checkoutservice", "cartservice"),
    ("checkoutservice", "productcatalogservice"),
    ("checkoutservice", "shippingservice"),
    ("checkoutservice", "currencyservice"),
    ("checkoutservice", "paymentservice"),
    ("checkoutservice", "emailservice"),
]


def read_json(path: Path, required: bool = False) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def pod_node_id(pod_name: str) -> str:
    return f"pod:{pod_name}"


def service_name_from_metadata(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("target_service", "") or "")


def pods_by_service(snapshot: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    for pod in snapshot.get("pods", []) or []:
        if not isinstance(pod, dict):
            continue
        service = str(pod.get("service") or pod.get("labels", {}).get("app", "") or "")
        if not service:
            continue
        result.setdefault(service, []).append(pod)
    for service in result:
        result[service] = sorted(result[service], key=lambda p: str(p.get("name", "")))
    return result


def pods_by_ip(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for pod in snapshot.get("pods", []) or []:
        if isinstance(pod, dict) and pod.get("ip"):
            result[str(pod["ip"])] = pod
    return result


def service_endpoint_ips(snapshot: Dict[str, Any], service: str) -> List[str]:
    for svc in snapshot.get("services", []) or []:
        if isinstance(svc, dict) and svc.get("name") == service:
            return [str(ip) for ip in svc.get("endpoint_ips", []) or [] if ip]
    return []


def deployment_replicas(snapshot: Dict[str, Any], service: str) -> Optional[int]:
    for dep in snapshot.get("deployments", []) or []:
        if isinstance(dep, dict) and dep.get("name") == service:
            try:
                return int(dep.get("replicas", 0) or 0)
            except (TypeError, ValueError):
                return None
    return None


def new_pods_for_service(pre: Dict[str, Any], during: Dict[str, Any],
                         service: str) -> List[Dict[str, Any]]:
    pre_names = {str(p.get("name", "")) for p in pods_by_service(pre).get(service, [])}
    return [
        pod for pod in pods_by_service(during).get(service, [])
        if str(pod.get("name", "")) not in pre_names
    ]


def removed_pods_for_service(pre: Dict[str, Any], during: Dict[str, Any],
                             service: str) -> List[Dict[str, Any]]:
    during_names = {str(p.get("name", "")) for p in pods_by_service(during).get(service, [])}
    return [
        pod for pod in pods_by_service(pre).get(service, [])
        if str(pod.get("name", "")) not in during_names
    ]


def detect_node_scale(pre: Dict[str, Any], during: Dict[str, Any],
                      service: str) -> Dict[str, Any]:
    pre_pods = pods_by_service(pre).get(service, [])
    during_pods = pods_by_service(during).get(service, [])
    pre_replicas = deployment_replicas(pre, service)
    during_replicas = deployment_replicas(during, service)
    if pre_replicas is None:
        pre_replicas = len(pre_pods)
    if during_replicas is None:
        during_replicas = len(during_pods)
    added = new_pods_for_service(pre, during, service)
    return {
        "matched": during_replicas > pre_replicas or len(added) > 0,
        "service": service,
        "pre_replicas": pre_replicas,
        "during_replicas": during_replicas,
        "added_pods": added,
        "during_pods": during_pods,
    }


def detect_node_drift(pre: Dict[str, Any], during: Dict[str, Any],
                      service: str) -> Dict[str, Any]:
    removed = removed_pods_for_service(pre, during, service)
    added = new_pods_for_service(pre, during, service)
    pairs = []
    for idx, old_pod in enumerate(removed):
        if idx < len(added):
            pairs.append((old_pod, added[idx]))
    return {
        "matched": bool(pairs),
        "service": service,
        "pairs": pairs,
    }


def endpoint_backend_service(snapshot: Dict[str, Any], service: str) -> List[Tuple[str, Dict[str, Any]]]:
    by_ip = pods_by_ip(snapshot)
    result = []
    for ip in service_endpoint_ips(snapshot, service):
        pod = by_ip.get(ip)
        if pod is not None:
            result.append((str(pod.get("service", "")), pod))
    return result


def detect_edge_redirect(pre: Dict[str, Any], during: Dict[str, Any],
                         service: str) -> Dict[str, Any]:
    pre_backends = endpoint_backend_service(pre, service)
    during_backends = endpoint_backend_service(during, service)
    wrong = [(owner, pod) for owner, pod in during_backends if owner and owner != service]
    old_service_pods = [pod for owner, pod in pre_backends if owner == service]
    return {
        "matched": bool(wrong),
        "service": service,
        "wrong_backends": wrong,
        "old_service_pods": old_service_pods,
    }


def detect_edge_removal(during: Dict[str, Any], service: str) -> Dict[str, Any]:
    endpoint_ips = service_endpoint_ips(during, service)
    reachable = endpoint_backend_service(during, service)
    return {
        "matched": len(endpoint_ips) == 0 or len(reachable) == 0,
        "service": service,
        "endpoint_ips": endpoint_ips,
        "reachable_backends": reachable,
    }


def detect_version_split(during: Dict[str, Any], metadata: Dict[str, Any],
                         service: str) -> Dict[str, Any]:
    children = set()
    for pod in during.get("pods", []) or []:
        if not isinstance(pod, dict):
            continue
        pod_service = str(pod.get("service", "") or "")
        pod_name = str(pod.get("name", "") or "")
        version = str(pod.get("version", "") or "")
        labels = pod.get("labels", {}) if isinstance(pod.get("labels"), dict) else {}
        label_version = str(labels.get("version", "") or "")
        if pod_service == f"{service}-v2" or pod_name.startswith(f"{service}-v2"):
            children.add(f"{service}:v2")
        if pod_service == service and version not in {"", "unknown", "v1", "latest"}:
            children.add(f"{service}:{version}")
        if label_version and label_version not in {"unknown", "v1", "latest"} and service in pod_service:
            children.add(f"{service}:{label_version}")

    for dep in during.get("deployments", []) or []:
        if not isinstance(dep, dict):
            continue
        name = str(dep.get("name", "") or "")
        version = str(dep.get("version", "") or "")
        labels = dep.get("labels", {}) if isinstance(dep.get("labels"), dict) else {}
        template_labels = dep.get("template_labels", {}) if isinstance(dep.get("template_labels"), dict) else {}
        label_version = str(labels.get("version") or template_labels.get("version") or "")
        if name == f"{service}-v2":
            children.add(f"{service}:v2")
        if name == service and version not in {"", "unknown", "v1", "latest"}:
            children.add(f"{service}:{version}")
        if name.startswith(service) and label_version not in {"", "unknown", "v1", "latest"}:
            children.add(f"{service}:{label_version}")

    fault_params = metadata.get("fault_params", {}) if isinstance(metadata.get("fault_params"), dict) else {}
    if any(key in fault_params for key in ("v2_weight", "v1_weight", "v2_delay_ms", "v2_abort_pct")):
        children.add(f"{service}:v2")
    if children and f"{service}:v1" not in children:
        children.add(f"{service}:v1")

    return {
        "matched": bool(children),
        "service": service,
        "children": sorted(children),
        "route_weights": route_weights_from_metadata(metadata),
    }


def route_weights_from_metadata(metadata: Dict[str, Any]) -> Dict[str, float]:
    fault_params = metadata.get("fault_params", {}) if isinstance(metadata.get("fault_params"), dict) else {}
    weights = {}
    if "v1_weight" in fault_params:
        weights["v1"] = _pct_to_ratio(fault_params.get("v1_weight"))
    if "v2_weight" in fault_params:
        weights["v2"] = _pct_to_ratio(fault_params.get("v2_weight"))
    if "v1" not in weights and "v2" in weights:
        weights["v1"] = max(0.0, 1.0 - weights["v2"])
    if "v2" not in weights and "v1" in weights:
        weights["v2"] = max(0.0, 1.0 - weights["v1"])
    return weights


def _pct_to_ratio(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric / 100.0 if numeric > 1.0 else numeric


def upstream_services(service: str, pre_graph: Optional[Dict[str, Any]] = None) -> List[str]:
    upstreams = []
    if pre_graph:
        for edge in pre_graph.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", edge.get("src", "")) or "")
            target = str(edge.get("target", edge.get("dst", "")) or "")
            if target == service and source:
                upstreams.append(source)
    for source, target in KNOWN_EDGES:
        if target == service:
            upstreams.append(source)
    return sorted(dict.fromkeys(upstreams))


def trace_mismatch_edges(trace_summary_path: Path) -> List[Dict[str, str]]:
    """Extract declared-vs-observed destination mismatches from compact traces.

    Real trace summaries in this repo do not always carry declared destination
    fields, so this function also supports synthetic/test columns:
    declared_dst|declared_service and observed_dst|actual_dst|destination_service.
    """
    if not trace_summary_path.exists():
        return []
    result = []
    with trace_summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            declared = _first_present(row, ["declared_dst", "declared_service", "declared_target"])
            observed = _first_present(row, ["observed_dst", "actual_dst", "destination_service", "observed_service"])
            source = _first_present(row, ["source", "source_service", "client_service", "service"])
            if declared and observed and declared != observed:
                result.append({
                    "source": source or declared,
                    "declared": declared,
                    "observed": observed,
                })
    return result


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def scenario_hint(metadata: Dict[str, Any], topology_diff: Dict[str, Any]) -> Set[str]:
    hints = set()
    for key in ("scenario_id", "scenario_name", "fault_type", "topology_change_type"):
        value = str(metadata.get(key, "") or "").lower()
        if value:
            hints.add(value)
    annotation = topology_diff.get("annotation", {}) if isinstance(topology_diff.get("annotation"), dict) else {}
    for value in annotation.get("detected_types", []) or []:
        hints.add(str(value).lower())
    if annotation.get("topology_change_type"):
        hints.add(str(annotation.get("topology_change_type")).lower())
    return hints
