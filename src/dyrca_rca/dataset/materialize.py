"""Materialize dynamic RCA graphs into small-model tensor artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from dyrca_rca.dataset.labels import root_node_index, root_service_from_row
from dyrca_rca.graph.executor import execute_plan, normalize_graph
from dyrca_rca.graph.update_ops import GraphUpdatePlan
from dyrca_rca.graph.weights import apply_edge_weights
from dyrca_rca.planner.rule_planner import plan_experiment as rule_plan_experiment
from dyrca_rca.planner.signals import KNOWN_EDGES, read_json


GRAPH_VARIANTS = [
    "static_pre",
    "naive_during",
    "rule_node",
    "rule_edge",
    "rule_full",
    "llm_full",
]

NODE_FEATURE_NAMES = [
    "is_service",
    "is_pod",
    "is_version",
    "is_blackhole",
    "replicas",
    "ready_replicas",
    "removed",
    "split_parent",
    "split_child",
    "endpoint_count",
    "in_degree",
    "out_degree",
    "endpoint_out_degree",
    "dependency_in_degree",
    "max_incident_final_weight",
    "max_incident_change_weight",
    "min_incident_topology_confidence",
    "incident_update_count",
    "has_trace_mismatch_edge",
    "has_blackhole_edge",
]

EDGE_FEATURE_NAMES = [
    "dependency_weight",
    "anomaly_weight",
    "topology_confidence",
    "change_weight",
    "final_weight",
]

INTERACTIVE_FEATURE_PROTOCOL = "interactive-topology-trace-v2-missingness"
INTERACTIVE_NODE_FEATURE_NAMES = NODE_FEATURE_NAMES + [
    "trace_anomaly", "trace_observed", "trace_status_coverage",
    "trace_log_sample_count", "trace_partial",
]
# Models use the last column as the message weight.
INTERACTIVE_EDGE_FEATURE_NAMES = EDGE_FEATURE_NAMES[:-1] + [
    "anomaly_observed", "trace_status_coverage", "trace_partial",
] + EDGE_FEATURE_NAMES[-1:]


def feature_names(protocol):
    from dyrca_rca.interaction.temporal import PROTOCOL, FEATURES
    if protocol == PROTOCOL:
        return INTERACTIVE_NODE_FEATURE_NAMES + FEATURES, INTERACTIVE_EDGE_FEATURE_NAMES
    if protocol == INTERACTIVE_FEATURE_PROTOCOL:
        return INTERACTIVE_NODE_FEATURE_NAMES, INTERACTIVE_EDGE_FEATURE_NAMES
    return NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


def read_index(path: str) -> pd.DataFrame:
    """Read an index artifact, preferring parquet but tolerating CSV/JSONL."""
    index_path = Path(path)
    suffix = index_path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(index_path)
        except Exception:
            # Test and lightweight environments may lack pyarrow. If the file is
            # actually CSV/JSONL content with a parquet suffix, accept it.
            try:
                return pd.read_csv(index_path)
            except Exception:
                return pd.read_json(index_path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(index_path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(index_path, lines=True)
    raise ValueError(f"unsupported index format: {suffix}")


def build_graph_from_topology(snapshot: Dict[str, Any], phase: str = "",
                              dependency_edges=None) -> Dict[str, Any]:
    """Build a small dynamic graph from one topology snapshot."""
    nodes = []
    edges = []
    seen_nodes = set()

    pods = [pod for pod in snapshot.get("pods", []) or [] if isinstance(pod, dict)]
    pods_by_ip = {str(pod.get("ip", "")): pod for pod in pods if pod.get("ip")}
    pods_by_service = {}
    for pod in pods:
        service = str(pod.get("service", "") or "")
        if service:
            pods_by_service.setdefault(service, []).append(pod)

    for dep in snapshot.get("deployments", []) or []:
        if not isinstance(dep, dict) or not dep.get("name"):
            continue
        service = str(dep["name"])
        service_pods = pods_by_service.get(service, [])
        attrs = {
            "type": "service",
            "phase": phase,
            "service": service,
            "replicas": _float(dep.get("replicas"), len(service_pods)),
            "ready_replicas": _float(dep.get("readyReplicas"), 0.0),
            "version": dep.get("version", "unknown"),
            "endpoint_count": 0.0,
        }
        _add_node(nodes, seen_nodes, service, attrs)

    for pod in pods:
        pod_name = str(pod.get("name", "") or "")
        if not pod_name:
            continue
        service = str(pod.get("service", "") or "")
        if service and service not in seen_nodes:
            _add_node(nodes, seen_nodes, service, {
                "type": "service",
                "phase": phase,
                "service": service,
                "replicas": len(pods_by_service.get(service, [])),
                "ready_replicas": len(pods_by_service.get(service, [])),
                "version": "unknown",
                "endpoint_count": 0.0,
            })
        _add_node(nodes, seen_nodes, f"pod:{pod_name}", {
            "type": "pod",
            "phase": phase,
            "service": service,
            "ip": pod.get("ip", ""),
            "node": pod.get("node", ""),
            "version": pod.get("version", "unknown"),
            "status": pod.get("status", ""),
        })

    service_endpoint_counts = {}
    for svc in snapshot.get("services", []) or []:
        if not isinstance(svc, dict) or not svc.get("name"):
            continue
        service = str(svc["name"])
        endpoint_ips = [str(ip) for ip in svc.get("endpoint_ips", []) or [] if ip]
        service_endpoint_counts[service] = len(endpoint_ips)
        if service not in seen_nodes:
            _add_node(nodes, seen_nodes, service, {
                "type": "service",
                "phase": phase,
                "service": service,
                "endpoint_count": len(endpoint_ips),
            })
        for ip in endpoint_ips:
            pod = pods_by_ip.get(ip)
            if not pod:
                continue
            pod_id = f"pod:{pod.get('name')}"
            edges.append({
                "source": service,
                "target": pod_id,
                "edge_type": "endpoint",
                "weight": 1.0,
                "attrs": {
                    "endpoint_ip": ip,
                    "dependency_weight": 1.0 / max(len(endpoint_ips), 1),
                },
            })

    for node in nodes:
        service = node["attrs"].get("service")
        if node["attrs"].get("type") == "service" and service in service_endpoint_counts:
            node["attrs"]["endpoint_count"] = service_endpoint_counts[service]

    services = {node["id"] for node in nodes if node["attrs"].get("type") == "service"}
    for source, target in (KNOWN_EDGES if dependency_edges is None else dependency_edges):
        if source in services and target in services:
            edges.append({
                "source": source,
                "target": target,
                "edge_type": "dependency",
                "weight": 1.0,
                "attrs": {},
            })

    return {
        "attrs": {"phase": phase},
        "nodes": nodes,
        "edges": edges,
    }


def materialize_index(index_path: str, variant: str, out_dir: str,
                      limit: Optional[int] = None,
                      llm_plan_dir: str = "artifacts/llm_plans",
                      emit_pyg: bool = False) -> List[str]:
    """Materialize a whole index for one graph variant."""
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"unknown graph variant: {variant}")
    frame = read_index(index_path)
    if limit is not None:
        frame = frame.head(limit)
    outputs = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        error_message = row_dict.get("error_message")
        if isinstance(error_message, str) and error_message.strip():
            continue
        data_path = row_dict.get("data_path")
        if data_path is None or (isinstance(data_path, float) and pd.isna(data_path)):
            continue
        artifact = materialize_experiment(
            row_dict,
            variant=variant,
            out_dir=out_dir,
            llm_plan_dir=llm_plan_dir,
            emit_pyg=emit_pyg,
        )
        outputs.append(artifact)
    return outputs


def materialize_experiment(row: Dict[str, Any], variant: str, out_dir: str,
                           llm_plan_dir: str = "artifacts/llm_plans",
                           emit_pyg: bool = False) -> str:
    exp_dir = Path(str(row["data_path"]))
    experiment_id = str(row.get("experiment_id") or exp_dir.name)
    graph = build_graph_variant(exp_dir, experiment_id, variant, llm_plan_dir=llm_plan_dir)
    tensor_data = graph_to_model_tensors(graph, row, emit_pyg=emit_pyg)
    out_path = Path(out_dir) / variant / f"{experiment_id}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_pt(out_path, tensor_data)
    return str(out_path)


def build_graph_variant(exp_dir: Path, experiment_id: str, variant: str,
                        llm_plan_dir: str = "artifacts/llm_plans") -> Dict[str, Any]:
    pre = read_json(exp_dir / "topology" / "pre.json", required=True)
    during = read_json(exp_dir / "topology" / "during.json", required=True)
    g_pre = build_graph_from_topology(pre, phase="pre")
    if variant == "static_pre":
        graph = g_pre
    elif variant == "naive_during":
        graph = build_graph_from_topology(during, phase="during")
    elif variant in {"rule_node", "rule_edge", "rule_full"}:
        plan = rule_plan_experiment(str(exp_dir), g_pre=g_pre)
        graph = _apply_plan_variant(g_pre, plan, variant)
    elif variant == "llm_full":
        plan = _load_cached_llm_plan(experiment_id, llm_plan_dir)
        if plan is None:
            plan = rule_plan_experiment(str(exp_dir), g_pre=g_pre)
            graph = execute_plan(g_pre, plan).graph
            graph.setdefault("attrs", {})["llm_plan_missing"] = True
        else:
            graph = execute_plan(g_pre, plan).graph
            graph.setdefault("attrs", {})["llm_plan_missing"] = False
    else:
        raise ValueError(f"unknown graph variant: {variant}")
    graph.setdefault("attrs", {})["graph_variant"] = variant
    graph.setdefault("attrs", {})["experiment_id"] = experiment_id
    return apply_edge_weights(str(exp_dir), graph)


def graph_to_model_tensors(graph: Dict[str, Any], row: Dict[str, Any],
                           emit_pyg: bool = False) -> Dict[str, Any]:
    normalized = normalize_graph(graph)
    node_names, edge_names = feature_names(graph.get("attrs", {}).get("feature_protocol"))
    nodes = normalized["nodes"]
    edges = normalized["edges"]
    node_ids = [node["id"] for node in nodes]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    incident_features = _incident_node_features(nodes, edges)
    edge_pairs = []
    edge_attr_rows = []
    edge_types = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_to_idx or target not in node_to_idx:
            continue
        attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs"), dict) else {}
        attr_row = [_float(attrs.get(name), 0.0) for name in edge_names]
        edge_type = str(edge.get("edge_type", "dependency"))
        edge_pairs.append([node_to_idx[source], node_to_idx[target]])
        edge_attr_rows.append(attr_row)
        edge_types.append(edge_type)
        # RCA evidence can live on outgoing endpoint/update edges. Add reverse
        # message edges so service candidates can receive that signal.
        edge_pairs.append([node_to_idx[target], node_to_idx[source]])
        edge_attr_rows.append(list(attr_row))
        edge_types.append(f"{edge_type}:reverse")

    x_rows = [_node_features(node, incident_features.get(node["id"], {})) for node in nodes]
    for node, values in zip(nodes, x_rows):
        values.extend(_float(node.get("attrs", {}).get(name), 0.0)
                      for name in node_names[len(NODE_FEATURE_NAMES):])
    label = root_node_index(nodes, root_service_from_row(row))
    metadata = {
        "experiment_id": str(row.get("experiment_id", "")),
        "scenario_id": str(row.get("scenario_id", "")),
        "target_service": str(row.get("target_service", "")),
        "graph_variant": graph.get("attrs", {}).get("graph_variant", ""),
        "node_feature_names": list(node_names),
        "edge_feature_names": list(edge_names),
    }

    data = {
        "x": _tensor(x_rows, dtype="float"),
        "edge_index": _edge_index_tensor(edge_pairs),
        "edge_attr": _tensor(edge_attr_rows, dtype="float"),
        "node_ids": node_ids,
        "edge_types": edge_types,
        "label": _label_tensor(label),
        "metadata": metadata,
        "graph": graph,
    }
    if hasattr(data["edge_attr"], "reshape"):
        data["edge_attr"] = data["edge_attr"].reshape(-1, len(edge_names))
    pyg_data = _maybe_pyg_data(data) if emit_pyg else None
    if pyg_data is not None:
        data["pyg_data"] = pyg_data
    return data


def _apply_plan_variant(g_pre: Dict[str, Any], plan: GraphUpdatePlan,
                        variant: str) -> Dict[str, Any]:
    if variant == "rule_full":
        filtered = plan
    elif variant == "rule_node":
        filtered = GraphUpdatePlan(
            experiment_id=plan.experiment_id,
            change_type=plan.change_type,
            confidence=plan.confidence,
            causal_role=plan.causal_role,
            node_ops=plan.node_ops,
            edge_ops=[],
            weight_ops=[],
            state_transfer=plan.state_transfer,
            evidence=plan.evidence,
        )
    elif variant == "rule_edge":
        filtered = GraphUpdatePlan(
            experiment_id=plan.experiment_id,
            change_type=plan.change_type,
            confidence=plan.confidence,
            causal_role=plan.causal_role,
            node_ops=[],
            edge_ops=plan.edge_ops,
            weight_ops=plan.weight_ops,
            state_transfer=plan.state_transfer,
            evidence=plan.evidence,
        )
    else:
        raise ValueError(f"not a rule variant: {variant}")
    return execute_plan(g_pre, filtered).graph


def _load_cached_llm_plan(experiment_id: str, llm_plan_dir: str) -> Optional[GraphUpdatePlan]:
    path = Path(llm_plan_dir) / f"{experiment_id}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return GraphUpdatePlan.from_dict(data)


def _incident_node_features(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    features = {
        str(node.get("id", "")): {
            "in_degree": 0.0,
            "out_degree": 0.0,
            "endpoint_out_degree": 0.0,
            "dependency_in_degree": 0.0,
            "max_incident_final_weight": 0.0,
            "max_incident_change_weight": 0.0,
            "min_incident_topology_confidence": 1.0,
            "incident_update_count": 0.0,
            "has_trace_mismatch_edge": 0.0,
            "has_blackhole_edge": 0.0,
        }
        for node in nodes
    }
    for edge in edges:
        source = str(edge.get("source", "") or "")
        target = str(edge.get("target", "") or "")
        if source not in features or target not in features:
            continue
        edge_type = str(edge.get("edge_type", "dependency") or "dependency")
        attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs"), dict) else {}
        features[source]["out_degree"] += 1.0
        features[target]["in_degree"] += 1.0
        if edge_type == "endpoint":
            features[source]["endpoint_out_degree"] += 1.0
        if edge_type == "dependency":
            features[target]["dependency_in_degree"] += 1.0
        final_weight = _float(attrs.get("final_weight", edge.get("weight", 0.0)), 0.0)
        change_weight = _float(attrs.get("change_weight"), 0.0)
        topology_confidence = _float(attrs.get("topology_confidence"), 1.0)
        changed = any(
            attrs.get(key)
            for key in (
                "op",
                "reweighted",
                "split_parent_edge",
                "redirected_to",
                "redirected_from",
                "endpoint_replacement",
                "trace_mismatch",
                "blackhole",
            )
        )
        for node_id in (source, target):
            item = features[node_id]
            item["max_incident_final_weight"] = max(item["max_incident_final_weight"], final_weight)
            item["max_incident_change_weight"] = max(item["max_incident_change_weight"], change_weight)
            item["min_incident_topology_confidence"] = min(item["min_incident_topology_confidence"], topology_confidence)
            if changed:
                item["incident_update_count"] += 1.0
            if attrs.get("trace_mismatch") or edge_type == "trace_mismatch":
                item["has_trace_mismatch_edge"] = 1.0
            if attrs.get("blackhole") or edge_type == "blackhole" or str(target).startswith("blackhole:"):
                item["has_blackhole_edge"] = 1.0
    return features


def _node_features(node: Dict[str, Any], incident: Dict[str, float]) -> List[float]:
    attrs = node.get("attrs", {}) if isinstance(node.get("attrs"), dict) else {}
    node_id = str(node.get("id", "") or "")
    node_type = str(attrs.get("type", "") or "")
    return [
        1.0 if node_type == "service" or (not node_id.startswith("pod:") and ":" not in node_id and not node_id.startswith("blackhole:")) else 0.0,
        1.0 if node_type == "pod" or node_id.startswith("pod:") else 0.0,
        1.0 if attrs.get("split_child") or (":" in node_id and not node_id.startswith("blackhole:")) else 0.0,
        1.0 if attrs.get("blackhole") or node_id.startswith("blackhole:") else 0.0,
        _float(attrs.get("replicas"), 0.0),
        _float(attrs.get("ready_replicas"), 0.0),
        1.0 if attrs.get("removed") else 0.0,
        1.0 if attrs.get("split_parent") else 0.0,
        1.0 if attrs.get("split_child") else 0.0,
        _float(attrs.get("endpoint_count"), 0.0),
        _float(incident.get("in_degree"), 0.0),
        _float(incident.get("out_degree"), 0.0),
        _float(incident.get("endpoint_out_degree"), 0.0),
        _float(incident.get("dependency_in_degree"), 0.0),
        _float(incident.get("max_incident_final_weight"), 0.0),
        _float(incident.get("max_incident_change_weight"), 0.0),
        _float(incident.get("min_incident_topology_confidence"), 1.0),
        _float(incident.get("incident_update_count"), 0.0),
        _float(incident.get("has_trace_mismatch_edge"), 0.0),
        _float(incident.get("has_blackhole_edge"), 0.0),
    ]


def _add_node(nodes: List[Dict[str, Any]], seen: set, node_id: str,
              attrs: Dict[str, Any]) -> None:
    if not node_id or node_id in seen:
        return
    seen.add(node_id)
    nodes.append({"id": node_id, "attrs": dict(attrs)})


def _tensor(values: List[List[float]], dtype: str = "float") -> Any:
    try:
        import torch
    except Exception:
        return values
    if dtype == "long":
        return torch.tensor(values, dtype=torch.long)
    return torch.tensor(values, dtype=torch.float32)


def _edge_index_tensor(edge_pairs: List[List[int]]) -> Any:
    try:
        import torch
    except Exception:
        return [list(pair) for pair in zip(*edge_pairs)] if edge_pairs else [[], []]
    if not edge_pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()


def _label_tensor(label: int) -> Any:
    try:
        import torch
    except Exception:
        return int(label)
    return torch.tensor(int(label), dtype=torch.long)


def _maybe_pyg_data(data: Dict[str, Any]) -> Any:
    try:
        from torch_geometric.data import Data
    except Exception:
        return None
    return Data(
        x=data["x"],
        edge_index=data["edge_index"],
        edge_attr=data["edge_attr"],
        y=data["label"],
    )


def _save_pt(path: Path, data: Dict[str, Any]) -> None:
    try:
        import torch
    except Exception:
        with path.open("wb") as f:
            pickle.dump(data, f)
        return
    torch.save(data, str(path))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize RCA graph tensors.")
    parser.add_argument("--index", required=True,
                        help="Index parquet/csv/jsonl path.")
    parser.add_argument("--variant", required=True, choices=GRAPH_VARIANTS,
                        help="Graph variant to materialize.")
    parser.add_argument("--out-dir", default="artifacts/model_data",
                        help="Output root for .pt artifacts.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of indexed experiments to process.")
    parser.add_argument("--llm-plan-dir", default="artifacts/llm_plans",
                        help="Cached LLM plan directory for llm_full.")
    parser.add_argument("--emit-pyg", action="store_true",
                        help="Include PyG Data object when torch_geometric is installed.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    outputs = materialize_index(
        args.index,
        variant=args.variant,
        out_dir=args.out_dir,
        limit=args.limit,
        llm_plan_dir=args.llm_plan_dir,
        emit_pyg=args.emit_pyg,
    )
    print(f"materialized {len(outputs)} {args.variant} graphs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
