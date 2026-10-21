"""Quality metrics for graph update plans and executed graph artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from dyrca_rca.dataset.materialize import build_graph_from_topology
from dyrca_rca.graph.executor import execute_plan, normalize_graph
from dyrca_rca.graph.update_ops import (
    ADD_BLACKHOLE_NODE,
    ADD_EDGE,
    ADD_NODE,
    ADD_TRACE_MISMATCH_EDGE,
    ALIGN_NODE,
    GraphUpdatePlan,
    REDIRECT_EDGE,
    REMOVE_NODE,
    REWEIGHT_EDGE,
    SPLIT_EDGE,
    SPLIT_NODE,
)
from dyrca_rca.planner.signals import (
    detect_edge_redirect,
    detect_edge_removal,
    detect_node_drift,
    detect_node_scale,
    detect_version_split,
    pod_node_id,
    read_json,
    route_weights_from_metadata,
    service_name_from_metadata,
    trace_mismatch_edges,
)


GRAPH_UPDATE_METRIC_COLUMNS = [
    "experiment_id",
    "scenario_id",
    "scenario_name",
    "target_service",
    "change_type",
    "plan_found",
    "graph_found",
    "operation_validity_rate",
    "plan_execution_success_rate",
    "node_add_remove_f1",
    "node_alignment_accuracy",
    "edge_redirect_recall",
    "blackhole_recall",
    "version_split_recall",
    "edge_weight_mae",
    "trace_mismatch_recall",
    "error_message",
]

AGGREGATE_COLUMNS = [
    "scenario_id",
    "scenario_name",
    "count",
    "operation_validity_rate",
    "plan_execution_success_rate",
    "node_add_remove_f1",
    "node_alignment_accuracy",
    "edge_redirect_recall",
    "blackhole_recall",
    "version_split_recall",
    "edge_weight_mae",
    "trace_mismatch_recall",
]

NUMERIC_METRICS = [
    "operation_validity_rate",
    "plan_execution_success_rate",
    "node_add_remove_f1",
    "node_alignment_accuracy",
    "edge_redirect_recall",
    "blackhole_recall",
    "version_split_recall",
    "edge_weight_mae",
    "trace_mismatch_recall",
]


def evaluate_graph_update_rows(rows: Iterable[Dict[str, Any]], plans_dir: str,
                               graphs_dir: Optional[str] = None,
                               limit: Optional[int] = None) -> pd.DataFrame:
    """Evaluate graph update plans for index rows."""
    out_rows = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        out_rows.append(evaluate_graph_update(row, plans_dir, graphs_dir=graphs_dir))
    return pd.DataFrame(out_rows, columns=GRAPH_UPDATE_METRIC_COLUMNS)


def evaluate_graph_update(row: Dict[str, Any], plans_dir: str,
                          graphs_dir: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate one experiment's plan against topology-derived expectations."""
    experiment_id = _str(row.get("experiment_id")) or Path(_str(row.get("data_path"))).name
    base = {
        "experiment_id": experiment_id,
        "scenario_id": _str(row.get("scenario_id")),
        "scenario_name": _str(row.get("scenario_name")),
        "target_service": _str(row.get("target_service")),
        "change_type": _str(row.get("topology_change_type")),
        "plan_found": False,
        "graph_found": False,
        "operation_validity_rate": 0.0,
        "plan_execution_success_rate": 0.0,
        "node_add_remove_f1": None,
        "node_alignment_accuracy": None,
        "edge_redirect_recall": None,
        "blackhole_recall": None,
        "version_split_recall": None,
        "edge_weight_mae": None,
        "trace_mismatch_recall": None,
        "error_message": "",
    }

    exp_dir = Path(_str(row.get("data_path")))
    if not exp_dir.exists():
        base["error_message"] = "missing experiment directory"
        return base

    try:
        expected = derive_expected_operations(exp_dir, row)
    except Exception as exc:
        base["error_message"] = "could not derive expected operations: %s" % exc
        return base

    base.update({
        "target_service": expected["target_service"],
        "change_type": expected["change_type"],
    })

    plan_path = find_artifact(plans_dir, experiment_id) if plans_dir else None
    graph_path = find_artifact(graphs_dir, experiment_id) if graphs_dir else None
    graph = _read_graph_artifact(graph_path) if graph_path else {}
    base["graph_found"] = bool(graph_path)

    if not plan_path:
        _fill_missing_plan_expected_metrics(base, expected)
        base["error_message"] = "missing plan artifact"
        return base

    try:
        plan = GraphUpdatePlan.from_dict(_read_json(plan_path))
    except Exception as exc:
        _fill_missing_plan_expected_metrics(base, expected)
        base["plan_found"] = True
        base["error_message"] = "could not load plan: %s" % exc
        return base

    base["plan_found"] = True
    base["operation_validity_rate"] = operation_validity_rate(plan)
    base["plan_execution_success_rate"] = plan_execution_success(plan, exp_dir)
    base["node_add_remove_f1"] = node_add_remove_f1(plan, expected)
    base["node_alignment_accuracy"] = node_alignment_accuracy(plan, expected)
    base["edge_redirect_recall"] = edge_redirect_recall(plan, expected, graph)
    base["blackhole_recall"] = blackhole_recall(plan, expected, graph)
    base["version_split_recall"] = version_split_recall(plan, expected, graph)
    base["edge_weight_mae"] = edge_weight_mae(plan, expected, graph)
    base["trace_mismatch_recall"] = trace_mismatch_recall(plan, expected, graph)
    return base


def derive_expected_operations(exp_dir: Path, row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Derive expected graph update events from metadata and topology snapshots."""
    row = row or {}
    metadata = read_json(exp_dir / "metadata.json", required=True)
    topology_diff = read_json(exp_dir / "topology_diff.json", required=False)
    pre = read_json(exp_dir / "topology" / "pre.json", required=True)
    during = read_json(exp_dir / "topology" / "during.json", required=True)
    target_service = service_name_from_metadata(metadata) or _str(row.get("target_service"))

    scale = detect_node_scale(pre, during, target_service) if target_service else {"matched": False}
    drift = detect_node_drift(pre, during, target_service) if target_service else {"matched": False}
    redirect = detect_edge_redirect(pre, during, target_service) if target_service else {"matched": False}
    removal = detect_edge_removal(during, target_service) if target_service else {"matched": False}
    split = detect_version_split(during, metadata, target_service) if target_service else {"matched": False}
    trace_mismatches = trace_mismatch_edges(exp_dir / "traces" / "trace_summary.csv")

    change_type = (
        _str(metadata.get("topology_change_type"))
        or _str(topology_diff.get("topology_change_type"))
        or _str(row.get("topology_change_type"))
    )

    add_nodes = set()
    remove_nodes = set()
    if scale.get("matched"):
        add_nodes.update(pod_node_id(_str(pod.get("name"))) for pod in scale.get("added_pods", []))

    alignment_pairs = set()
    if drift.get("matched"):
        for old_pod, new_pod in drift.get("pairs", []):
            alignment_pairs.add((
                pod_node_id(_str(old_pod.get("name"))),
                pod_node_id(_str(new_pod.get("name"))),
            ))

    redirect_targets = set()
    if redirect.get("matched"):
        for _, pod in redirect.get("wrong_backends", []):
            redirect_targets.add(pod_node_id(_str(pod.get("name"))))

    version_children = set(split.get("children", [])) if split.get("matched") else set()
    expected_trace = set(
        (_str(item.get("source")), _str(item.get("observed")))
        for item in trace_mismatches
        if item.get("observed")
    )

    return {
        "experiment_id": _str(metadata.get("experiment_id")) or exp_dir.name,
        "scenario_id": _str(metadata.get("scenario_id")) or _str(row.get("scenario_id")),
        "scenario_name": _str(metadata.get("scenario_name")) or _str(row.get("scenario_name")),
        "target_service": target_service,
        "change_type": change_type,
        "add_nodes": add_nodes,
        "remove_nodes": remove_nodes,
        "alignment_pairs": alignment_pairs,
        "redirect_targets": redirect_targets,
        "blackhole_expected": bool(removal.get("matched")),
        "version_children": version_children,
        "route_weights": route_weights_from_metadata(metadata),
        "trace_mismatch_edges": expected_trace,
        "node_drift_expected": bool(drift.get("matched")),
    }


def operation_validity_rate(plan: GraphUpdatePlan) -> float:
    """Return valid operation count divided by total operations."""
    ops = list(plan.node_ops) + list(plan.edge_ops) + list(plan.weight_ops)
    if not ops:
        return 1.0 if not plan.validate_basic() else 0.0
    valid = sum(1 for op in ops if not op.validate_basic())
    return valid / float(len(ops))


def plan_execution_success(plan: GraphUpdatePlan, exp_dir: Path) -> float:
    """Return 1.0 when executing the plan on pre topology has no warnings."""
    try:
        pre = read_json(exp_dir / "topology" / "pre.json", required=True)
        g_pre = build_graph_from_topology(pre, phase="pre")
        result = execute_plan(g_pre, plan)
    except Exception:
        return 0.0
    return 1.0 if not result.warnings else 0.0


def node_add_remove_f1(plan: GraphUpdatePlan, expected: Dict[str, Any]) -> Optional[float]:
    """F1 over ADD_NODE and REMOVE_NODE ops.

    Pod drift is scored by alignment accuracy, so raw add/remove F1 is skipped
    for drift scenarios to avoid penalizing identity-preserving plans.
    """
    if expected.get("node_drift_expected"):
        return None
    expected_set = set(expected.get("add_nodes", set())) | set(expected.get("remove_nodes", set()))
    if not expected_set:
        return None
    predicted = set()
    for op in plan.node_ops:
        if op.op == ADD_NODE and op.node_id:
            predicted.add(op.node_id)
        elif op.op == REMOVE_NODE and op.node_id:
            predicted.add(op.node_id)
    return f1_score(expected_set, predicted)


def node_alignment_accuracy(plan: GraphUpdatePlan, expected: Dict[str, Any]) -> Optional[float]:
    expected_pairs = set(expected.get("alignment_pairs", set()))
    if not expected_pairs:
        return None
    predicted = set(
        (op.old_node, op.new_node)
        for op in plan.node_ops
        if op.op == ALIGN_NODE and op.old_node and op.new_node
    )
    return len(expected_pairs & predicted) / float(len(expected_pairs))


def edge_redirect_recall(plan: GraphUpdatePlan, expected: Dict[str, Any],
                         graph: Optional[Dict[str, Any]] = None) -> Optional[float]:
    expected_targets = set(expected.get("redirect_targets", set()))
    if not expected_targets:
        return None
    predicted = set(
        op.new_target for op in plan.edge_ops
        if op.op == REDIRECT_EDGE and op.new_target
    )
    predicted.update(_graph_targets_with_attr(graph, "redirected_from"))
    return recall(expected_targets, predicted)


def blackhole_recall(plan: GraphUpdatePlan, expected: Dict[str, Any],
                     graph: Optional[Dict[str, Any]] = None) -> Optional[float]:
    if not expected.get("blackhole_expected"):
        return None
    service = expected.get("target_service")
    blackhole_node = "blackhole:%s" % service
    predicted = any(op.op == ADD_BLACKHOLE_NODE and op.service == service for op in plan.node_ops)
    predicted = predicted or any(
        op.op == ADD_EDGE and op.target == blackhole_node for op in plan.edge_ops
    )
    if not predicted and graph:
        normalized = normalize_graph(graph)
        predicted = any(node["id"] == blackhole_node for node in normalized["nodes"])
    return 1.0 if predicted else 0.0


def version_split_recall(plan: GraphUpdatePlan, expected: Dict[str, Any],
                         graph: Optional[Dict[str, Any]] = None) -> Optional[float]:
    expected_children = set(expected.get("version_children", set()))
    if not expected_children:
        return None
    predicted = set()
    for op in plan.node_ops:
        if op.op == SPLIT_NODE:
            predicted.update(op.children)
            if op.version and op.parent:
                predicted.add("%s:%s" % (op.parent, op.version))
    if graph:
        normalized = normalize_graph(graph)
        predicted.update(
            node["id"] for node in normalized["nodes"]
            if node["id"] in expected_children or node["attrs"].get("split_child")
        )
    return recall(expected_children, predicted)


def edge_weight_mae(plan: GraphUpdatePlan, expected: Dict[str, Any],
                    graph: Optional[Dict[str, Any]] = None) -> Optional[float]:
    expected_weights = dict(expected.get("route_weights", {}))
    if not expected_weights:
        return None
    predicted = predicted_route_weights(plan, graph=graph)
    if not predicted:
        return 1.0
    errors = []
    for version, expected_value in expected_weights.items():
        pred_value = predicted.get(version, 0.0)
        errors.append(abs(float(expected_value) - float(pred_value)))
    return sum(errors) / float(len(errors)) if errors else None


def predicted_route_weights(plan: GraphUpdatePlan,
                            graph: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    weights = {}
    for op in plan.weight_ops:
        if op.op != REWEIGHT_EDGE:
            continue
        version = _version_from_target(op.target)
        if version:
            weights[version] = _op_weight(op)
    for op in plan.edge_ops:
        if op.op != SPLIT_EDGE:
            continue
        version = _version_from_target(op.new_target or op.new_source)
        if version:
            weights[version] = _edge_op_weight(op)
    if graph:
        normalized = normalize_graph(graph)
        for edge in normalized["edges"]:
            version = _version_from_target(edge.get("target"))
            if version:
                weights[version] = _edge_weight(edge)
    return {key: value for key, value in weights.items() if value is not None}


def trace_mismatch_recall(plan: GraphUpdatePlan, expected: Dict[str, Any],
                          graph: Optional[Dict[str, Any]] = None) -> Optional[float]:
    expected_edges = set(expected.get("trace_mismatch_edges", set()))
    if not expected_edges:
        return None
    predicted = set(
        (op.source, op.target)
        for op in plan.edge_ops
        if op.op == ADD_TRACE_MISMATCH_EDGE and op.source and op.target
    )
    if graph:
        normalized = normalize_graph(graph)
        for edge in normalized["edges"]:
            if edge["attrs"].get("trace_mismatch") or edge.get("edge_type") == "trace_mismatch":
                predicted.add((edge["source"], edge["target"]))
    return recall(expected_edges, predicted)


def f1_score(expected: Set[Any], predicted: Set[Any]) -> float:
    if not expected and not predicted:
        return 1.0
    if not predicted:
        return 0.0
    tp = len(expected & predicted)
    precision = tp / float(len(predicted)) if predicted else 0.0
    rec = tp / float(len(expected)) if expected else 0.0
    if precision + rec == 0.0:
        return 0.0
    return 2.0 * precision * rec / (precision + rec)


def recall(expected: Set[Any], predicted: Set[Any]) -> float:
    if not expected:
        return 1.0
    return len(expected & predicted) / float(len(expected))


def aggregate_by_scenario(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-experiment graph update metrics by scenario."""
    if frame.empty:
        return pd.DataFrame(columns=AGGREGATE_COLUMNS)
    rows = []
    group_frame = frame.copy()
    group_frame["scenario_id"] = group_frame["scenario_id"].fillna("unknown")
    group_frame["scenario_name"] = group_frame["scenario_name"].fillna("")
    grouped = group_frame.groupby(["scenario_id", "scenario_name"])
    for (scenario_id, scenario_name), group in grouped:
        out = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "count": int(len(group)),
        }
        for metric in NUMERIC_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            out[metric] = float(values.mean()) if len(values) else None
        rows.append(out)
    return pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)


def find_artifact(directory: Optional[str], experiment_id: str) -> Optional[Path]:
    """Find a JSON artifact for an experiment id in a flat or nested directory."""
    if not directory:
        return None
    root = Path(directory)
    if not root.exists():
        return None
    candidates = [
        root / ("%s.json" % experiment_id),
        root / ("%s_updated.json" % experiment_id),
        root / ("%s.graph.json" % experiment_id),
        root / ("%s.pt" % experiment_id),
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.rglob("%s.json" % experiment_id))
    if matches:
        return matches[0]
    matches = sorted(root.rglob("%s.pt" % experiment_id))
    return matches[0] if matches else None


def _fill_missing_plan_expected_metrics(row: Dict[str, Any], expected: Dict[str, Any]) -> None:
    if expected.get("add_nodes") or expected.get("remove_nodes"):
        row["node_add_remove_f1"] = 0.0
    if expected.get("alignment_pairs"):
        row["node_alignment_accuracy"] = 0.0
    if expected.get("redirect_targets"):
        row["edge_redirect_recall"] = 0.0
    if expected.get("blackhole_expected"):
        row["blackhole_recall"] = 0.0
    if expected.get("version_children"):
        row["version_split_recall"] = 0.0
    if expected.get("route_weights"):
        row["edge_weight_mae"] = 1.0
    if expected.get("trace_mismatch_edges"):
        row["trace_mismatch_recall"] = 0.0


def _graph_targets_with_attr(graph: Optional[Dict[str, Any]], attr_name: str) -> Set[str]:
    if not graph:
        return set()
    normalized = normalize_graph(graph)
    return set(
        edge["target"] for edge in normalized["edges"]
        if edge["attrs"].get(attr_name)
    )


def _version_from_target(target: Any) -> str:
    text = _str(target)
    if ":" not in text:
        return ""
    return text.rsplit(":", 1)[-1]


def _op_weight(op: Any) -> Optional[float]:
    if getattr(op, "weight", None) is not None:
        return float(op.weight)
    attrs = getattr(op, "attrs", {}) or {}
    return _float_first(attrs, ["final_weight", "route_weight", "weight", "dependency_weight"])


def _edge_op_weight(op: Any) -> Optional[float]:
    if getattr(op, "weight", None) is not None:
        return float(op.weight)
    return _float_first(getattr(op, "attrs", {}) or {}, ["route_weight", "final_weight", "weight"])


def _edge_weight(edge: Dict[str, Any]) -> Optional[float]:
    attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs"), dict) else {}
    value = _float_first(attrs, ["route_weight", "final_weight", "weight"])
    if value is not None:
        return value
    try:
        return float(edge.get("weight"))
    except (TypeError, ValueError):
        return None


def _float_first(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in mapping:
            continue
        try:
            return float(mapping[key])
        except (TypeError, ValueError):
            continue
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_graph_artifact(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".pt":
        try:
            import torch
            item = torch.load(str(path), map_location="cpu")
        except Exception:
            with Path(path).open("rb") as f:
                item = pickle.load(f)
        if isinstance(item, dict):
            graph = item.get("graph", item)
            return graph if isinstance(graph, dict) else {}
    return {}


def _str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
