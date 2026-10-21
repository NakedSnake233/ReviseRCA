"""Evidence pack generation for candidate RCA reranking."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from dyrca_rca.features.logs import load_log_features
from dyrca_rca.features.metrics import load_metric_features
from dyrca_rca.features.traces import load_trace_features
from dyrca_rca.features.topology import load_topology_features
from dyrca_rca.graph.executor import normalize_graph
from dyrca_rca.planner.signals import read_json


def build_evidence_pack(prediction_row: Dict[str, Any],
                        experiment_dir: str,
                        graph_artifact: Dict[str, Any],
                        top_k: int = 5) -> Dict[str, Any]:
    """Build compact evidence for top-k predicted candidates."""
    exp_dir = Path(experiment_dir)
    metadata = read_json(exp_dir / "metadata.json", required=False)
    graph = normalize_graph(graph_artifact)
    traces = load_trace_features(str(exp_dir))
    metrics = load_metric_features(str(exp_dir), metadata=metadata)
    logs = load_log_features(str(exp_dir), include_all=True)
    topology = load_topology_features(str(exp_dir))
    ranked = _ranked_nodes(prediction_row)[:top_k]
    score_map = _score_map(prediction_row, ranked)
    candidates = []
    for rank, node_id in enumerate(ranked, start=1):
        service = _service_for_node(node_id, graph)
        evidence = _candidate_evidence(
            node_id=node_id,
            service=service,
            model_score=score_map.get(node_id, 0.0),
            graph=graph,
            traces=traces,
            metrics=metrics,
            logs=logs,
            topology=topology,
            metadata=metadata,
            original_rank=rank,
        )
        candidates.append(evidence)
    return {
        "experiment_id": str(prediction_row.get("experiment_id") or metadata.get("experiment_id", "")),
        "scenario_id": str(prediction_row.get("scenario_id") or metadata.get("scenario_id", "")),
        "scenario_name": str(metadata.get("scenario_name", "")),
        "target_service": str(prediction_row.get("target_service") or metadata.get("target_service", "")),
        "top_k": top_k,
        "candidates": candidates,
    }


def build_evidence_packs(predictions_csv: str, experiment_root_or_map: Any,
                         graph_root_or_map: Any, top_k: int = 5) -> List[Dict[str, Any]]:
    """Build evidence packs for rows in a predictions CSV.

    `experiment_root_or_map` can be a data-root path or `{experiment_id: exp_dir}`.
    `graph_root_or_map` can be a materialized-graph root/file map or a graph dict.
    """
    frame = pd.read_csv(predictions_csv)
    packs = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        exp_id = str(row_dict.get("experiment_id", ""))
        exp_dir = _resolve_experiment_dir(experiment_root_or_map, exp_id)
        graph = _resolve_graph(graph_root_or_map, exp_id)
        packs.append(build_evidence_pack(row_dict, exp_dir, graph, top_k=top_k))
    return packs


def _candidate_evidence(node_id: str, service: str, model_score: float,
                        graph: Dict[str, Any], traces: Dict[str, Any],
                        metrics: Dict[str, Any], logs: Dict[str, Any],
                        topology: Dict[str, Any], metadata: Dict[str, Any],
                        original_rank: int) -> Dict[str, Any]:
    refs = []
    refs.extend(_topology_evidence(node_id, service, graph, topology))
    refs.extend(_trace_evidence(node_id, service, graph, traces))
    refs.extend(_metric_evidence(service, metrics))
    refs.extend(_log_evidence(service, logs))
    refs.extend(_temporal_evidence(service, metadata, topology))
    refs.extend(_counter_evidence(node_id, service, graph, traces, metrics, logs))
    return {
        "candidate": node_id,
        "service": service,
        "model_score": float(model_score),
        "original_rank": original_rank,
        "topology_evidence": [ev for ev in refs if ev["type"] == "topology"],
        "trace_evidence": [ev for ev in refs if ev["type"] == "trace"],
        "metric_evidence": [ev for ev in refs if ev["type"] == "metric"],
        "log_evidence": [ev for ev in refs if ev["type"] == "log"],
        "temporal_order_evidence": [ev for ev in refs if ev["type"] == "temporal_order"],
        "counter_evidence": [ev for ev in refs if ev["type"] == "counter"],
        "evidence_refs": refs,
    }


def _topology_evidence(node_id: str, service: str, graph: Dict[str, Any],
                       topology: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = []
    node_attrs = _node_attrs(graph, node_id)
    if node_attrs:
        if node_attrs.get("split_child") or ":" in node_id:
            refs.append(_ref("topology", node_id, "version split node", {"attrs": node_attrs}))
        if node_attrs.get("blackhole") or node_id.startswith("blackhole:"):
            refs.append(_ref("topology", node_id, "blackhole endpoint", {"attrs": node_attrs}))
        if node_attrs.get("aligned_from") or node_attrs.get("aligned_to"):
            refs.append(_ref("topology", node_id, "node alignment", {"attrs": node_attrs}))
    for edge in graph.get("edges", []):
        attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs"), dict) else {}
        if edge.get("source") == node_id or edge.get("target") == node_id or _service_for_edge(edge, graph) == service:
            if attrs.get("trace_mismatch") or attrs.get("redirected_from"):
                refs.append(_ref("topology", node_id, "redirect or trace mismatch edge", {"edge": edge}))
            if edge.get("edge_type") in {"blackhole", "version_split"}:
                refs.append(_ref("topology", node_id, f"{edge.get('edge_type')} edge", {"edge": edge}))
    mask = topology.get("node_change_mask", {})
    if mask.get(node_id) or mask.get(service):
        refs.append(_ref("topology", node_id, "node changed in topology mask", {"node": node_id}))
    return refs


def _trace_evidence(node_id: str, service: str, graph: Dict[str, Any],
                    traces: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = []
    for mismatch in traces.get("route_mismatch_candidates", []) or []:
        if service in {mismatch.get("declared"), mismatch.get("observed"), mismatch.get("source")}:
            refs.append(_ref("trace", node_id, "declared vs observed route mismatch", mismatch))
    stats = traces.get("services", {}).get(service, {})
    if stats:
        refs.append(_ref("trace", node_id, "service trace stats", stats))
        if float(stats.get("error_rate", 0.0) or 0.0) > 0 or float(stats.get("p95_duration", 0.0) or 0.0) > 500:
            refs.append(_ref("trace", node_id, "abnormal trace signal", stats))
    for key, edge_stats in traces.get("edges", {}).items():
        if service in key.split("->"):
            refs.append(_ref("trace", node_id, "edge trace stats", edge_stats))
    return refs


def _metric_evidence(service: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = []
    stats = metrics.get("services", {}).get(service, {})
    if stats:
        refs.append(_ref("metric", service, "service metric stats", stats))
    for key, edge_stats in metrics.get("edges", {}).items():
        if service in key.split("->"):
            refs.append(_ref("metric", service, "edge metric stats", edge_stats))
    return refs


def _log_evidence(service: str, logs: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats = logs.get("services", {}).get(service, {})
    if not stats:
        return []
    return [_ref("log", service, "service log counts", stats)]


def _temporal_evidence(service: str, metadata: Dict[str, Any],
                       topology: Dict[str, Any]) -> List[Dict[str, Any]]:
    tw = metadata.get("time_window", {}) if isinstance(metadata.get("time_window"), dict) else {}
    if not tw:
        return []
    return [_ref("temporal_order", service, "fault window ordering", {
        "t_collect": tw.get("t_collect", ""),
        "t_inject": tw.get("t_inject", ""),
        "t_recover": tw.get("t_recover", ""),
        "topo_change": topology.get("topo_change", {}),
    })]


def _counter_evidence(node_id: str, service: str, graph: Dict[str, Any],
                      traces: Dict[str, Any], metrics: Dict[str, Any],
                      logs: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = []
    if not traces.get("services", {}).get(service) and not metrics.get("services", {}).get(service):
        refs.append(_ref("counter", node_id, "no direct telemetry evidence", {"service": service}))
    log_stats = logs.get("services", {}).get(service, {})
    if log_stats and int(log_stats.get("error_count", 0) or 0) == 0:
        refs.append(_ref("counter", node_id, "no service error logs", log_stats))
    return refs


def _ranked_nodes(row: Dict[str, Any]) -> List[str]:
    value = row.get("ranked_node_ids") or row.get("ranked_node_ids_str") or ""
    if isinstance(value, list):
        return [str(v) for v in value]
    return [token for token in str(value).split("|") if token]


def _score_map(row: Dict[str, Any], ranked: List[str]) -> Dict[str, float]:
    if row.get("scores"):
        try:
            parsed = json.loads(row["scores"]) if isinstance(row["scores"], str) else row["scores"]
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
        except Exception:
            pass
    top_score = _float(row.get("top_score"), 1.0)
    return {node_id: top_score / float(idx + 1) for idx, node_id in enumerate(ranked)}


def _node_attrs(graph: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    for node in graph.get("nodes", []):
        if node.get("id") == node_id:
            attrs = node.get("attrs", {})
            return attrs if isinstance(attrs, dict) else {}
    return {}


def _service_for_node(node_id: str, graph: Dict[str, Any]) -> str:
    attrs = _node_attrs(graph, node_id)
    if attrs.get("service"):
        return str(attrs["service"])
    if node_id.startswith("blackhole:"):
        return node_id.split(":", 1)[1]
    if node_id.startswith("pod:"):
        return node_id[4:].rsplit("-", 2)[0]
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return node_id


def _service_for_edge(edge: Dict[str, Any], graph: Dict[str, Any]) -> str:
    return _service_for_node(str(edge.get("target", "")), graph)


def _ref(kind: str, candidate: str, summary: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(_safe_json(detail), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    ref_id = f"{kind}:{candidate}:{digest}"
    return {
        "id": ref_id,
        "type": kind,
        "summary": summary,
        "detail": _safe_json(detail),
    }


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _safe_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_safe_json(v) for v in value]
        return str(value)


def _resolve_experiment_dir(root_or_map: Any, experiment_id: str) -> str:
    if isinstance(root_or_map, dict):
        return str(root_or_map[experiment_id])
    return str(Path(str(root_or_map)) / experiment_id)


def _resolve_graph(root_or_map: Any, experiment_id: str) -> Dict[str, Any]:
    if isinstance(root_or_map, dict):
        graph = root_or_map.get(experiment_id, root_or_map)
        if isinstance(graph, dict) and "nodes" in graph:
            return graph
        return graph
    path = Path(str(root_or_map))
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    for candidate in path.glob(f"**/{experiment_id}.pt"):
        try:
            import torch
            item = torch.load(str(candidate), map_location="cpu")
            return item.get("graph", item)
        except Exception:
            continue
    for candidate in path.glob(f"**/{experiment_id}.json"):
        with candidate.open(encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"graph artifact not found for {experiment_id} under {path}")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
