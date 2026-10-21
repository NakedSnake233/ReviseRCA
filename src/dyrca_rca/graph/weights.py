"""Edge weight calculation for dynamic RCA graphs."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dyrca_rca.features.metrics import load_metric_features
from dyrca_rca.features.topology import load_topology_features
from dyrca_rca.features.traces import load_trace_features
from dyrca_rca.graph.executor import normalize_graph
from dyrca_rca.planner.signals import (
    endpoint_backend_service,
    read_json,
    route_weights_from_metadata,
    service_endpoint_ips,
)


WEIGHT_COEFFICIENTS = {
    "dependency_weight": 0.30,
    "anomaly_weight": 0.35,
    "topology_confidence": 0.25,
    "change_weight": 0.10,
}


def apply_edge_weights(experiment_dir: str, graph: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate graph edges with fixed component and final weights."""
    exp_dir = Path(experiment_dir)
    metadata = read_json(exp_dir / "metadata.json", required=False)
    during = read_json(exp_dir / "topology" / "during.json", required=False)
    metrics = load_metric_features(str(exp_dir), metadata=metadata)
    traces = load_trace_features(str(exp_dir))
    topo = load_topology_features(str(exp_dir))
    normalized = normalize_graph(graph)
    weighted = {
        "attrs": deepcopy(normalized.get("attrs", {})),
        "nodes": deepcopy(normalized.get("nodes", [])),
        "edges": [],
    }
    weighted["attrs"]["weight_coefficients"] = dict(WEIGHT_COEFFICIENTS)
    weighted["attrs"]["feature_status"] = {
        "missing_metric_files": metrics.get("missing_files", []),
        "trace_missing": traces.get("missing", True),
    }

    node_service = _node_service_lookup(weighted["nodes"], during)
    target_service = str(metadata.get("target_service", "") or "")
    route_weights = route_weights_from_metadata(metadata)
    change_type = str(metadata.get("topology_change_type", "") or "")

    for edge in normalized.get("edges", []):
        edge_out = deepcopy(edge)
        attrs = edge_out.setdefault("attrs", {})
        source = edge_out.get("source", "")
        target = edge_out.get("target", "")
        source_service = _service_for_node(source, node_service)
        target_service_for_edge = _service_for_node(target, node_service)

        dependency = _dependency_weight(
            source_service,
            target_service_for_edge,
            target,
            attrs,
            metrics,
            route_weights,
            target_service,
        )
        anomaly = _anomaly_weight(source_service, target_service_for_edge, metrics, traces)
        topology_conf = _topology_confidence(source_service, target_service_for_edge, target, attrs, during)
        change = _change_weight(edge_out, attrs, change_type, topo)
        final = (
            WEIGHT_COEFFICIENTS["dependency_weight"] * dependency
            + WEIGHT_COEFFICIENTS["anomaly_weight"] * anomaly
            + WEIGHT_COEFFICIENTS["topology_confidence"] * topology_conf
            + WEIGHT_COEFFICIENTS["change_weight"] * change
        )

        attrs["dependency_weight"] = _round(dependency)
        attrs["anomaly_weight"] = _round(anomaly)
        attrs["topology_confidence"] = _round(topology_conf)
        attrs["change_weight"] = _round(change)
        attrs["final_weight"] = _round(final)
        attrs["weight_coefficients"] = dict(WEIGHT_COEFFICIENTS)
        edge_out["weight"] = _round(final)
        weighted["edges"].append(edge_out)

    if "warnings" in graph:
        weighted["warnings"] = list(graph.get("warnings", []))
    return weighted


def _dependency_weight(source_service: str, target_service: str, target_node: str,
                       attrs: Dict[str, Any], metrics: Dict[str, Any],
                       route_weights: Dict[str, float],
                       metadata_target_service: str) -> float:
    route_weight = attrs.get("route_weight")
    if route_weight is None and ":" in target_node:
        base, version = target_node.rsplit(":", 1)
        if base == metadata_target_service:
            route_weight = route_weights.get(version)
    if route_weight is not None:
        return _clamp(_float(route_weight, 0.0))

    edge_key = f"{source_service}->{target_service}"
    edge_stats = metrics.get("edges", {}).get(edge_key, {})
    if edge_stats:
        return _clamp(_float(edge_stats.get("normalized_request_rate"), 0.0))

    if target_node.startswith("pod:") and attrs.get("replica_count"):
        replicas = max(_float(attrs.get("replica_count"), 1.0), 1.0)
        return _clamp(1.0 / replicas)

    if "dependency_weight" in attrs:
        return _clamp(_float(attrs.get("dependency_weight"), 0.0))
    return _clamp(_float(attrs.get("weight"), 1.0))


def _anomaly_weight(source_service: str, target_service: str,
                    metrics: Dict[str, Any], traces: Dict[str, Any]) -> float:
    components = []
    edge_key = f"{source_service}->{target_service}"
    metric_edge = metrics.get("edges", {}).get(edge_key, {})
    trace_edge = traces.get("edges", {}).get(edge_key, {})
    service_metrics = metrics.get("services", {}).get(target_service, {})
    service_traces = traces.get("services", {}).get(target_service, {})

    if metric_edge:
        components.append(_clamp(_float(metric_edge.get("error_rate"), 0.0)))
        flags = metric_edge.get("response_flags", {}).get("sum", 0.0) if isinstance(metric_edge.get("response_flags"), dict) else 0.0
        req = metric_edge.get("request_rate", {}).get("sum", 0.0) if isinstance(metric_edge.get("request_rate"), dict) else 0.0
        components.append(_clamp(flags / req if req else 0.0))
    if service_metrics:
        p99 = service_metrics.get("latency_p99", {}).get("avg", 0.0) if isinstance(service_metrics.get("latency_p99"), dict) else 0.0
        components.append(_clamp(_float(p99) / 1000.0))
    if trace_edge:
        components.append(_clamp(_float(trace_edge.get("error_rate"), 0.0)))
        components.append(_clamp(_float(trace_edge.get("p95_duration"), 0.0) / 1000.0))
    elif service_traces:
        components.append(_clamp(_float(service_traces.get("error_rate"), 0.0)))
        components.append(_clamp(_float(service_traces.get("p95_duration"), 0.0) / 1000.0))

    if not components:
        return 0.0
    return sum(components) / len(components)


def _topology_confidence(source_service: str, target_service: str, target_node: str,
                         attrs: Dict[str, Any], during: Dict[str, Any]) -> float:
    if attrs.get("trace_mismatch") or attrs.get("redirected_from"):
        return 0.2
    if target_node.startswith("blackhole:"):
        return 1.0
    if target_service and during:
        endpoint_ips = service_endpoint_ips(during, target_service)
        if endpoint_ips == []:
            return 0.1
        backends = endpoint_backend_service(during, target_service)
        if backends and any(owner != target_service for owner, _ in backends):
            return 0.1
    elif not target_service or not during:
        return 0.5
    if "topology_confidence" in attrs:
        return _clamp(_float(attrs.get("topology_confidence"), 1.0))
    return 1.0


def _change_weight(edge: Dict[str, Any], attrs: Dict[str, Any],
                   change_type: str, topo: Dict[str, Any]) -> float:
    op = str(attrs.get("op", "") or "")
    edge_type = str(edge.get("edge_type", "") or "")
    if op or attrs.get("trace_mismatch") or attrs.get("reweighted") or edge_type in {"blackhole", "version_split", "trace_mismatch"}:
        return 1.0
    if change_type:
        return 0.3
    topo_change = topo.get("topo_change", {})
    if any(_float(value, 0.0) > 0 for value in topo_change.values()):
        return 0.3
    return 0.0


def _node_service_lookup(nodes: List[Dict[str, Any]], during: Dict[str, Any]) -> Dict[str, str]:
    lookup = {}
    for node in nodes:
        node_id = str(node.get("id", "") or "")
        attrs = node.get("attrs", {}) if isinstance(node.get("attrs"), dict) else {}
        if attrs.get("service"):
            lookup[node_id] = str(attrs["service"])
        elif node_id.startswith("pod:"):
            lookup[node_id] = _pod_service_from_snapshot(node_id[4:], during)
        elif ":" in node_id:
            lookup[node_id] = node_id.split(":", 1)[0]
        else:
            lookup[node_id] = node_id
    return lookup


def _pod_service_from_snapshot(pod_name: str, snapshot: Dict[str, Any]) -> str:
    for pod in snapshot.get("pods", []) or []:
        if isinstance(pod, dict) and pod.get("name") == pod_name:
            service = str(pod.get("service", "") or "")
            if service.endswith("-v2"):
                return service[:-3]
            return service
    if pod_name:
        return pod_name.rsplit("-", 2)[0]
    return ""


def _service_for_node(node_id: str, lookup: Dict[str, str]) -> str:
    if node_id in lookup:
        return lookup[node_id]
    if node_id.startswith("blackhole:"):
        return node_id.split(":", 1)[1]
    if node_id.startswith("pod:"):
        return node_id[4:].rsplit("-", 2)[0]
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return node_id


def _read_json(path: str) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: str, data: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply fixed edge weights to a graph.")
    parser.add_argument("--experiment-dir", required=True,
                        help="Experiment directory with metrics/traces/topology.")
    parser.add_argument("--graph", required=True,
                        help="Input graph JSON path.")
    parser.add_argument("--out", required=True,
                        help="Output weighted graph JSON path.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    graph = _read_json(args.graph)
    weighted = apply_edge_weights(args.experiment_dir, graph)
    _write_json(args.out, weighted)
    print(f"weighted {len(weighted.get('edges', []))} edges -> {args.out}")
    return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
