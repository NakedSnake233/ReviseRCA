"""Operation schemas for dynamic graph update plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ADD_NODE = "ADD_NODE"
REMOVE_NODE = "REMOVE_NODE"
KEEP_NODE = "KEEP_NODE"
ALIGN_NODE = "ALIGN_NODE"
SPLIT_NODE = "SPLIT_NODE"
MERGE_NODE = "MERGE_NODE"
ADD_EDGE = "ADD_EDGE"
REMOVE_EDGE = "REMOVE_EDGE"
REDIRECT_EDGE = "REDIRECT_EDGE"
SPLIT_EDGE = "SPLIT_EDGE"
REWEIGHT_EDGE = "REWEIGHT_EDGE"
ADD_TRACE_MISMATCH_EDGE = "ADD_TRACE_MISMATCH_EDGE"
ADD_BLACKHOLE_NODE = "ADD_BLACKHOLE_NODE"
REPLACE_ENDPOINT_EDGE = "REPLACE_ENDPOINT_EDGE"

NODE_OP_TYPES = {
    ADD_NODE,
    REMOVE_NODE,
    KEEP_NODE,
    ALIGN_NODE,
    SPLIT_NODE,
    MERGE_NODE,
    ADD_BLACKHOLE_NODE,
}

EDGE_OP_TYPES = {
    ADD_EDGE,
    REMOVE_EDGE,
    REDIRECT_EDGE,
    SPLIT_EDGE,
    ADD_TRACE_MISMATCH_EDGE,
    REPLACE_ENDPOINT_EDGE,
}

WEIGHT_OP_TYPES = {REWEIGHT_EDGE}

ALL_OP_TYPES = NODE_OP_TYPES | EDGE_OP_TYPES | WEIGHT_OP_TYPES


def normalize_op_type(value: Any) -> str:
    """Return a canonical operation type string."""
    if value is None:
        return ""
    return str(value).strip().upper()


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class NodeOp:
    """Node-level update operation."""

    op: str
    node_id: str = ""
    old_node: str = ""
    new_node: str = ""
    parent: str = ""
    children: List[str] = field(default_factory=list)
    service: str = ""
    version: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeOp":
        return cls(
            op=normalize_op_type(data.get("op", data.get("type"))),
            node_id=str(data.get("node_id", data.get("id", ""))),
            old_node=str(data.get("old_node", "")),
            new_node=str(data.get("new_node", "")),
            parent=str(data.get("parent", data.get("parent_node", ""))),
            children=[str(v) for v in _as_list(data.get("children", data.get("versions")))],
            service=str(data.get("service", "")),
            version=str(data.get("version", "")),
            attrs=dict(_as_dict(data.get("attrs"))),
            evidence=_as_list(data.get("evidence")),
        )

    def validate_basic(self) -> List[str]:
        warnings = []
        if self.op not in NODE_OP_TYPES:
            warnings.append(f"invalid node op type: {self.op or '<missing>'}")
            return warnings
        if self.op in {ADD_NODE, REMOVE_NODE, KEEP_NODE} and not self.node_id:
            warnings.append(f"{self.op} requires node_id")
        if self.op == ALIGN_NODE and (not self.old_node or not self.new_node):
            warnings.append("ALIGN_NODE requires old_node and new_node")
        if self.op == SPLIT_NODE and not self.parent:
            warnings.append("SPLIT_NODE requires parent")
        if self.op == SPLIT_NODE and not self.children and not self.version:
            warnings.append("SPLIT_NODE requires children or version")
        if self.op == MERGE_NODE and (not self.node_id or not self.parent):
            warnings.append("MERGE_NODE requires node_id and parent")
        if self.op == ADD_BLACKHOLE_NODE and not self.service:
            warnings.append("ADD_BLACKHOLE_NODE requires service")
        return warnings


@dataclass
class EdgeOp:
    """Edge-level update operation."""

    op: str
    source: str = ""
    target: str = ""
    old_source: str = ""
    old_target: str = ""
    new_source: str = ""
    new_target: str = ""
    service: str = ""
    edge_type: str = "dependency"
    weight: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeOp":
        weight = data.get("weight")
        if weight is not None:
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = None
        return cls(
            op=normalize_op_type(data.get("op", data.get("type"))),
            source=str(data.get("source", data.get("src", ""))),
            target=str(data.get("target", data.get("dst", ""))),
            old_source=str(data.get("old_source", data.get("source", data.get("src", "")))),
            old_target=str(data.get("old_target", data.get("target", data.get("dst", "")))),
            new_source=str(data.get("new_source", "")),
            new_target=str(data.get("new_target", "")),
            service=str(data.get("service", "")),
            edge_type=str(data.get("edge_type", data.get("type_name", "dependency"))),
            weight=weight,
            attrs=dict(_as_dict(data.get("attrs"))),
            evidence=_as_list(data.get("evidence")),
        )

    def validate_basic(self) -> List[str]:
        warnings = []
        if self.op not in EDGE_OP_TYPES:
            warnings.append(f"invalid edge op type: {self.op or '<missing>'}")
            return warnings
        if self.op in {ADD_EDGE, REMOVE_EDGE, ADD_TRACE_MISMATCH_EDGE}:
            if not self.source or not self.target:
                warnings.append(f"{self.op} requires source and target")
        if self.op in {REDIRECT_EDGE, REPLACE_ENDPOINT_EDGE}:
            if not self.old_source or not self.old_target:
                warnings.append(f"{self.op} requires old_source and old_target")
            if not self.new_target and not self.new_source:
                warnings.append(f"{self.op} requires new_target or new_source")
        if self.op == SPLIT_EDGE:
            if not self.source or not self.target:
                warnings.append("SPLIT_EDGE requires source and target")
            if not self.new_target and not self.new_source:
                warnings.append("SPLIT_EDGE requires new_target or new_source")
        return warnings


@dataclass
class WeightOp:
    """Weight-level update operation."""

    op: str
    source: str = ""
    target: str = ""
    weight: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightOp":
        weight = data.get("weight")
        if weight is not None:
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = None
        return cls(
            op=normalize_op_type(data.get("op", data.get("type"))),
            source=str(data.get("source", data.get("src", ""))),
            target=str(data.get("target", data.get("dst", ""))),
            weight=weight,
            attrs=dict(_as_dict(data.get("attrs"))),
            evidence=_as_list(data.get("evidence")),
        )

    def validate_basic(self) -> List[str]:
        warnings = []
        if self.op not in WEIGHT_OP_TYPES:
            warnings.append(f"invalid weight op type: {self.op or '<missing>'}")
            return warnings
        if not self.source or not self.target:
            warnings.append("REWEIGHT_EDGE requires source and target")
        if self.weight is None and "weight" not in self.attrs:
            warnings.append("REWEIGHT_EDGE requires weight or attrs.weight")
        return warnings


@dataclass
class GraphUpdatePlan:
    """Complete graph update plan for one experiment."""

    experiment_id: str
    change_type: str
    confidence: float
    causal_role: str
    node_ops: List[NodeOp] = field(default_factory=list)
    edge_ops: List[EdgeOp] = field(default_factory=list)
    weight_ops: List[WeightOp] = field(default_factory=list)
    state_transfer: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphUpdatePlan":
        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            change_type=str(data.get("change_type", "")),
            confidence=confidence,
            causal_role=str(data.get("causal_role", "")),
            node_ops=_parse_node_ops(data.get("node_ops")),
            edge_ops=_parse_edge_ops(data.get("edge_ops")),
            weight_ops=_parse_weight_ops(data.get("weight_ops")),
            state_transfer=dict(_as_dict(data.get("state_transfer"))),
            evidence=_as_list(data.get("evidence")),
        )

    def validate_basic(self) -> List[str]:
        warnings = []
        if not self.experiment_id:
            warnings.append("GraphUpdatePlan requires experiment_id")
        if self.confidence < 0.0 or self.confidence > 1.0:
            warnings.append("GraphUpdatePlan confidence should be in [0, 1]")
        for op in self.node_ops:
            warnings.extend(op.validate_basic())
        for op in self.edge_ops:
            warnings.extend(op.validate_basic())
        for op in self.weight_ops:
            warnings.extend(op.validate_basic())
        return warnings


def _parse_node_ops(value: Any) -> List[NodeOp]:
    ops = []
    for item in _as_list(value):
        if isinstance(item, dict):
            ops.append(NodeOp.from_dict(item))
        else:
            ops.append(NodeOp(op="", attrs={"invalid_raw_op": item}))
    return ops


def _parse_edge_ops(value: Any) -> List[EdgeOp]:
    ops = []
    for item in _as_list(value):
        if isinstance(item, dict):
            ops.append(EdgeOp.from_dict(item))
        else:
            ops.append(EdgeOp(op="", attrs={"invalid_raw_op": item}))
    return ops


def _parse_weight_ops(value: Any) -> List[WeightOp]:
    ops = []
    for item in _as_list(value):
        if isinstance(item, dict):
            ops.append(WeightOp.from_dict(item))
        else:
            ops.append(WeightOp(op="", attrs={"invalid_raw_op": item}))
    return ops
