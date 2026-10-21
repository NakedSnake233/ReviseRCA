"""Topology feature extraction for graph update experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


TOPO_CHANGE_FIELDS = [
    "node_added",
    "node_removed",
    "edge_added",
    "edge_removed",
    "replica_delta_max",
    "version_split",
    "change_score",
]


def load_topology_features(experiment_dir: str) -> Dict[str, Any]:
    exp_dir = Path(experiment_dir)
    pre = _read_json(exp_dir / "topology" / "pre.json")
    during = _read_json(exp_dir / "topology" / "during.json")
    topology_diff = _read_json(exp_dir / "topology_diff.json")
    return {
        "topo_change": topo_change_vector(topology_diff, pre, during),
        "node_change_mask": node_change_mask(pre, during),
        "edge_change_mask": edge_change_mask(pre, during),
    }


def topo_change_vector(topology_diff: Dict[str, Any], pre: Dict[str, Any],
                       during: Dict[str, Any]) -> Dict[str, float]:
    comp = topology_diff.get("comparisons", {}).get("pre_vs_during", {}) if topology_diff else {}
    k8s = comp.get("k8s_layer", {}) if isinstance(comp, dict) else {}
    graph = comp.get("graph_layer", {}) if isinstance(comp, dict) else {}
    pod_changes = k8s.get("pod_changes", {}) if isinstance(k8s, dict) else {}
    endpoint_changes = k8s.get("endpoint_changes", {}) if isinstance(k8s, dict) else {}
    dep_changes = k8s.get("deployment_changes", {}) if isinstance(k8s, dict) else {}
    edge_diff = graph.get("edge_diff", {}) if isinstance(graph, dict) else {}
    annotation = topology_diff.get("annotation", {}) if isinstance(topology_diff.get("annotation"), dict) else {}

    replica_delta = 0.0
    for change in pod_changes.get("count_changes", {}).values() if isinstance(pod_changes.get("count_changes"), dict) else []:
        try:
            replica_delta = max(replica_delta, abs(float(change.get("delta", 0))))
        except (TypeError, ValueError):
            pass
    version_split = 0.0
    for dep in during.get("deployments", []) or []:
        if isinstance(dep, dict):
            name = str(dep.get("name", ""))
            version = str(dep.get("version", ""))
            if name.endswith("-v2") or version not in {"", "unknown", "v1", "latest"}:
                version_split = 1.0

    return {
        "node_added": float(len(pod_changes.get("added", []) or [])),
        "node_removed": float(len(pod_changes.get("removed", []) or [])),
        "edge_added": float(len(edge_diff.get("added", []) or [])),
        "edge_removed": float(
            len(edge_diff.get("removed", []) or [])
            + len(endpoint_changes.get("changed_services", {}) or {})
        ),
        "replica_delta_max": replica_delta,
        "version_split": version_split,
        "change_score": _float(annotation.get("change_score"), 0.0),
    }


def node_change_mask(pre: Dict[str, Any], during: Dict[str, Any]) -> Dict[str, bool]:
    before = _pods_by_name(pre)
    after = _pods_by_name(during)
    mask = {}
    for name in sorted(set(before) | set(after)):
        mask[f"pod:{name}"] = before.get(name) != after.get(name)
    before_deps = _deployments_by_name(pre)
    after_deps = _deployments_by_name(during)
    for name in sorted(set(before_deps) | set(after_deps)):
        b = before_deps.get(name, {})
        a = after_deps.get(name, {})
        mask[name] = b.get("replicas") != a.get("replicas") or b.get("version") != a.get("version")
    return mask


def edge_change_mask(pre: Dict[str, Any], during: Dict[str, Any]) -> Dict[str, bool]:
    before = _service_endpoints(pre)
    after = _service_endpoints(during)
    mask = {}
    for service in sorted(set(before) | set(after)):
        mask[f"{service}->endpoints"] = before.get(service, set()) != after.get(service, set())
    return mask


def _service_endpoints(snapshot: Dict[str, Any]) -> Dict[str, Set[str]]:
    result = {}
    for svc in snapshot.get("services", []) or []:
        if isinstance(svc, dict) and svc.get("name"):
            result[str(svc["name"])] = set(str(ip) for ip in svc.get("endpoint_ips", []) or [])
    return result


def _pods_by_name(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(pod.get("name", "")): pod
        for pod in snapshot.get("pods", []) or []
        if isinstance(pod, dict) and pod.get("name")
    }


def _deployments_by_name(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(dep.get("name", "")): dep
        for dep in snapshot.get("deployments", []) or []
        if isinstance(dep, dict) and dep.get("name")
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
