"""Executor for dynamic graph update plans."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dyrca_rca.graph.update_ops import (
    ADD_BLACKHOLE_NODE,
    ADD_EDGE,
    ADD_NODE,
    ADD_TRACE_MISMATCH_EDGE,
    ALIGN_NODE,
    EdgeOp,
    GraphUpdatePlan,
    KEEP_NODE,
    MERGE_NODE,
    NodeOp,
    REDIRECT_EDGE,
    REMOVE_EDGE,
    REMOVE_NODE,
    REPLACE_ENDPOINT_EDGE,
    REWEIGHT_EDGE,
    SPLIT_EDGE,
    SPLIT_NODE,
    WeightOp,
)


@dataclass
class GraphExecutionResult:
    """Result of applying a graph update plan."""

    graph: Dict[str, Any]
    warnings: List[str]


class GraphExecutor:
    """Apply validated graph update operations to a small JSON graph."""

    def __init__(self, graph: Dict[str, Any]):
        normalized = normalize_graph(graph)
        self.graph_attrs = dict(normalized.get("attrs", {}))
        self.nodes = {}
        self.node_order = []
        self.edges = []
        self.warnings = []

        for node in normalized["nodes"]:
            self._insert_node(node["id"], node.get("attrs", {}), warn_on_existing=False)
        for edge in normalized["edges"]:
            self.edges.append(edge)

    def apply_plan(self, plan: GraphUpdatePlan) -> GraphExecutionResult:
        self.warnings.extend(plan.validate_basic())
        self.graph_attrs["last_update_plan"] = {
            "experiment_id": plan.experiment_id,
            "change_type": plan.change_type,
            "confidence": plan.confidence,
            "causal_role": plan.causal_role,
            "evidence": deepcopy(plan.evidence),
        }

        for op in plan.node_ops:
            self.apply_node_op(op)
        for op in plan.edge_ops:
            self.apply_edge_op(op)
        for op in plan.weight_ops:
            self.apply_weight_op(op)

        if plan.state_transfer:
            self.graph_attrs["state_transfer"] = deepcopy(plan.state_transfer)

        return GraphExecutionResult(graph=self.to_dict(), warnings=list(self.warnings))

    def apply_node_op(self, op: NodeOp) -> None:
        if op.validate_basic():
            self._warn(f"skipped invalid node op {op.op}: {'; '.join(op.validate_basic())}")
            return
        if op.op == ADD_NODE:
            self._insert_node(op.node_id, op.attrs)
        elif op.op == REMOVE_NODE:
            self._remove_node(op.node_id)
        elif op.op == KEEP_NODE:
            self._require_node(op.node_id, op.op)
        elif op.op == ALIGN_NODE:
            self._align_node(op)
        elif op.op == SPLIT_NODE:
            self._split_node(op)
        elif op.op == MERGE_NODE:
            self._merge_node(op)
        elif op.op == ADD_BLACKHOLE_NODE:
            self._add_blackhole_node(op.service, op.attrs)
        else:
            self._warn(f"unsupported node op: {op.op}")

    def apply_edge_op(self, op: EdgeOp) -> None:
        if op.validate_basic():
            self._warn(f"skipped invalid edge op {op.op}: {'; '.join(op.validate_basic())}")
            return
        if op.op == ADD_EDGE:
            self._add_edge(op.source, op.target, op.edge_type, op.weight, op.attrs)
        elif op.op == REMOVE_EDGE:
            self._remove_edge(op.source, op.target)
        elif op.op == REDIRECT_EDGE:
            self._redirect_edge(op, mode="redirect")
        elif op.op == REPLACE_ENDPOINT_EDGE:
            self._redirect_edge(op, mode="replace_endpoint")
        elif op.op == SPLIT_EDGE:
            self._split_edge(op)
        elif op.op == ADD_TRACE_MISMATCH_EDGE:
            attrs = dict(op.attrs)
            attrs.setdefault("trace_mismatch", True)
            self._add_edge(op.source, op.target, op.edge_type, op.weight, attrs)
        else:
            self._warn(f"unsupported edge op: {op.op}")

    def apply_weight_op(self, op: WeightOp) -> None:
        if op.validate_basic():
            self._warn(f"skipped invalid weight op {op.op}: {'; '.join(op.validate_basic())}")
            return
        if op.op == REWEIGHT_EDGE:
            self._reweight_edge(op)
        else:
            self._warn(f"unsupported weight op: {op.op}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attrs": deepcopy(self.graph_attrs),
            "nodes": [
                {"id": node_id, "attrs": deepcopy(self.nodes[node_id]["attrs"])}
                for node_id in self.node_order
                if node_id in self.nodes
            ],
            "edges": deepcopy(self.edges),
            "warnings": list(self.warnings),
        }

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _require_node(self, node_id: str, op_name: str) -> bool:
        if node_id not in self.nodes:
            self._warn(f"{op_name} references missing node: {node_id}")
            return False
        return True

    def _insert_node(self, node_id: str, attrs: Optional[Dict[str, Any]] = None,
                     warn_on_existing: bool = True) -> None:
        if not node_id:
            self._warn("cannot add node with empty id")
            return
        attrs = dict(attrs or {})
        if node_id in self.nodes:
            self.nodes[node_id]["attrs"].update(attrs)
            if warn_on_existing:
                self._warn(f"ADD_NODE updated existing node attrs: {node_id}")
            return
        self.nodes[node_id] = {"id": node_id, "attrs": attrs}
        self.node_order.append(node_id)

    def _remove_node(self, node_id: str) -> None:
        if not self._require_node(node_id, REMOVE_NODE):
            return
        self.nodes[node_id]["attrs"]["removed"] = True
        for edge in self.edges:
            if edge["source"] == node_id or edge["target"] == node_id:
                edge["attrs"]["active"] = False
                edge["attrs"]["removed_with_node"] = node_id

    def _align_node(self, op: NodeOp) -> None:
        old_exists = self._require_node(op.old_node, ALIGN_NODE)
        if not old_exists:
            return
        self._insert_node(op.new_node, op.attrs, warn_on_existing=False)
        self.nodes[op.old_node]["attrs"]["aligned_to"] = op.new_node
        self.nodes[op.new_node]["attrs"]["aligned_from"] = op.old_node
        attrs = {
            "alignment": True,
            "change_event": True,
            "op": ALIGN_NODE,
        }
        attrs.update(op.attrs)
        self._add_edge(op.old_node, op.new_node, "change_event", 1.0, attrs)

    def _split_node(self, op: NodeOp) -> None:
        if not self._require_node(op.parent, SPLIT_NODE):
            return
        children = list(op.children)
        if op.version and not children:
            children = [f"{op.parent}:{op.version}"]
        for child in children:
            child_attrs = dict(op.attrs)
            child_attrs.setdefault("parent", op.parent)
            child_attrs.setdefault("split_child", True)
            if ":" in child:
                child_attrs.setdefault("version", child.rsplit(":", 1)[-1])
            self._insert_node(child, child_attrs, warn_on_existing=False)
            self._add_edge(op.parent, child, "version_split", 1.0, {
                "op": SPLIT_NODE,
                "parent": op.parent,
                "child": child,
            })
        self.nodes[op.parent]["attrs"]["split_parent"] = True

    def _merge_node(self, op: NodeOp) -> None:
        if not self._require_node(op.node_id, MERGE_NODE):
            return
        self._insert_node(op.parent, {}, warn_on_existing=False)
        self.nodes[op.node_id]["attrs"]["merged_into"] = op.parent
        self._add_edge(op.node_id, op.parent, "merge_event", 1.0, {
            "op": MERGE_NODE,
            "merged_node": op.node_id,
            "parent": op.parent,
        })

    def _add_blackhole_node(self, service: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        if not self._require_node(service, ADD_BLACKHOLE_NODE):
            return
        node_id = f"blackhole:{service}"
        node_attrs = {"blackhole": True, "service": service}
        node_attrs.update(dict(attrs or {}))
        self._insert_node(node_id, node_attrs, warn_on_existing=False)
        self._add_edge(service, node_id, "blackhole", 1.0, {
            "op": ADD_BLACKHOLE_NODE,
            "topology_confidence": 1.0,
        })

    def _add_edge(self, source: str, target: str, edge_type: str = "dependency",
                  weight: Optional[float] = None,
                  attrs: Optional[Dict[str, Any]] = None) -> None:
        if not self._require_node(source, "ADD_EDGE source"):
            return
        if not self._require_node(target, "ADD_EDGE target"):
            return
        attrs = dict(attrs or {})
        existing = self._find_edge(source, target, edge_type=edge_type)
        if existing is not None:
            existing["attrs"].update(attrs)
            if weight is not None:
                existing["weight"] = weight
                existing["attrs"]["weight"] = weight
            return
        if weight is None:
            weight = float(attrs.get("weight", 1.0) or 1.0)
        attrs.setdefault("topology_confidence", attrs.get("confidence", 1.0))
        self.edges.append({
            "source": source,
            "target": target,
            "edge_type": edge_type or "dependency",
            "weight": float(weight),
            "attrs": attrs,
        })

    def _remove_edge(self, source: str, target: str) -> None:
        edge = self._find_edge(source, target)
        if edge is None:
            self._warn(f"REMOVE_EDGE references missing edge: {source}->{target}")
            return
        edge["attrs"]["active"] = False
        edge["attrs"]["removed"] = True
        edge["attrs"]["topology_confidence"] = 0.0

    def _redirect_edge(self, op: EdgeOp, mode: str) -> None:
        old_source = op.old_source
        old_target = op.old_target
        new_source = op.new_source or old_source
        new_target = op.new_target or old_target
        old_edge = self._find_edge(old_source, old_target)
        if old_edge is None:
            self._warn(f"{op.op} references missing old edge: {old_source}->{old_target}")
        else:
            old_edge["attrs"]["topology_confidence"] = min(
                float(old_edge["attrs"].get("topology_confidence", 1.0) or 1.0),
                0.1,
            )
            old_edge["attrs"]["redirected_to"] = f"{new_source}->{new_target}"
            old_edge["attrs"]["active"] = False if mode == "replace_endpoint" else old_edge["attrs"].get("active", True)
        attrs = dict(op.attrs)
        attrs.setdefault("op", op.op)
        attrs.setdefault("redirected_from", f"{old_source}->{old_target}")
        attrs.setdefault("topology_confidence", 1.0)
        if mode == "replace_endpoint":
            attrs.setdefault("endpoint_replacement", True)
        self._add_edge(new_source, new_target, op.edge_type, op.weight, attrs)

    def _split_edge(self, op: EdgeOp) -> None:
        source = op.source
        target = op.target
        new_source = op.new_source or source
        new_target = op.new_target or target
        old_edge = self._find_edge(source, target)
        if old_edge is None:
            self._warn(f"SPLIT_EDGE references missing edge: {source}->{target}")
        else:
            old_edge["attrs"]["split_parent_edge"] = True
        attrs = dict(op.attrs)
        attrs.setdefault("op", SPLIT_EDGE)
        attrs.setdefault("split_from", f"{source}->{target}")
        self._add_edge(new_source, new_target, op.edge_type, op.weight, attrs)

    def _reweight_edge(self, op: WeightOp) -> None:
        edge = self._find_edge(op.source, op.target)
        if edge is None:
            self._warn(f"REWEIGHT_EDGE references missing edge: {op.source}->{op.target}")
            return
        attrs = dict(op.attrs)
        edge["attrs"].update(attrs)
        new_weight = op.weight
        if new_weight is None and "weight" in attrs:
            try:
                new_weight = float(attrs["weight"])
            except (TypeError, ValueError):
                new_weight = None
        if new_weight is not None:
            edge["weight"] = float(new_weight)
            edge["attrs"]["weight"] = float(new_weight)
        edge["attrs"]["reweighted"] = True

    def _find_edge(self, source: str, target: str,
                   edge_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        for edge in self.edges:
            if edge["source"] == source and edge["target"] == target:
                if edge_type is None or edge.get("edge_type") == edge_type:
                    return edge
        return None


def normalize_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize supported graph JSON variants into nodes/edges lists."""
    nodes = []
    raw_nodes = graph.get("nodes", [])
    if isinstance(raw_nodes, dict):
        iterable = []
        for node_id, value in raw_nodes.items():
            attrs = value if isinstance(value, dict) else {}
            attrs = dict(attrs)
            attrs.pop("id", None)
            iterable.append({"id": str(node_id), "attrs": attrs})
        raw_nodes = iterable
    for raw in raw_nodes:
        if isinstance(raw, str):
            nodes.append({"id": raw, "attrs": {}})
        elif isinstance(raw, dict):
            node_id = str(raw.get("id", raw.get("name", "")))
            attrs = dict(raw.get("attrs", {})) if isinstance(raw.get("attrs"), dict) else {}
            for key, value in raw.items():
                if key not in {"id", "name", "attrs"}:
                    attrs.setdefault(key, value)
            if node_id:
                nodes.append({"id": node_id, "attrs": attrs})

    edges = []
    for raw in graph.get("edges", []):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", raw.get("src", "")))
        target = str(raw.get("target", raw.get("dst", "")))
        if not source or not target:
            continue
        attrs = dict(raw.get("attrs", {})) if isinstance(raw.get("attrs"), dict) else {}
        for key, value in raw.items():
            if key not in {"source", "src", "target", "dst", "edge_type", "type", "weight", "attrs"}:
                attrs.setdefault(key, value)
        edge_type = str(raw.get("edge_type", raw.get("type", "dependency")))
        try:
            weight = float(raw.get("weight", attrs.get("weight", 1.0)) or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        attrs.setdefault("topology_confidence", attrs.get("confidence", 1.0))
        edges.append({
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "weight": weight,
            "attrs": attrs,
        })

    return {"attrs": dict(graph.get("attrs", {})), "nodes": nodes, "edges": edges}


def execute_plan(graph: Dict[str, Any], plan: GraphUpdatePlan) -> GraphExecutionResult:
    """Apply a graph update plan to a graph dictionary."""
    return GraphExecutor(graph).apply_plan(plan)


def _read_json(path: str) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a graph update plan.")
    parser.add_argument("--graph", required=True, help="Input graph JSON file.")
    parser.add_argument("--plan", required=True, help="GraphUpdatePlan JSON file.")
    parser.add_argument("--out", required=True, help="Output updated graph JSON file.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    graph = _read_json(args.graph)
    plan = GraphUpdatePlan.from_dict(_read_json(args.plan))
    result = execute_plan(graph, plan)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result.graph, f, indent=2, sort_keys=True)
    if result.warnings:
        print(f"applied plan with {len(result.warnings)} warnings -> {args.out}")
    else:
        print(f"applied plan -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
