"""Validation helpers for LLM-produced GraphUpdatePlan JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dyrca_rca.graph.update_ops import (
    EDGE_OP_TYPES,
    NODE_OP_TYPES,
    WEIGHT_OP_TYPES,
    GraphUpdatePlan,
)


FORBIDDEN_TOP_LEVEL_KEYS = {
    "hyperparameters",
    "hpo",
    "automl",
    "model_params",
    "model_parameters",
    "training_config",
    "search_space",
    "parameter_sweep",
}


@dataclass
class PlanValidationResult:
    valid: bool
    plan: Optional[GraphUpdatePlan]
    errors: List[str]
    raw_plan: Optional[Dict[str, Any]] = None


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse an object from an LLM response, tolerating accidental fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data


def validate_plan_dict(data: Dict[str, Any]) -> PlanValidationResult:
    """Validate a GraphUpdatePlan dictionary and coerce it to dataclasses."""
    data = dict(data)
    if "state_transfer" in data and not isinstance(data.get("state_transfer"), dict):
        data["state_transfer"] = {}
    if "evidence" in data and not isinstance(data.get("evidence"), list):
        data["evidence"] = [data["evidence"]]
    errors = []
    required = [
        "experiment_id",
        "change_type",
        "confidence",
        "causal_role",
        "node_ops",
        "edge_ops",
        "weight_ops",
        "state_transfer",
    ]
    for key in required:
        if key not in data:
            errors.append(f"missing required key: {key}")
    for key in FORBIDDEN_TOP_LEVEL_KEYS:
        if key in data:
            errors.append(f"forbidden model-parameter key: {key}")
    if "node_ops" in data and not isinstance(data.get("node_ops"), list):
        errors.append("node_ops must be a list")
    if "edge_ops" in data and not isinstance(data.get("edge_ops"), list):
        errors.append("edge_ops must be a list")
    if "weight_ops" in data and not isinstance(data.get("weight_ops"), list):
        errors.append("weight_ops must be a list")
    _validate_op_types(data.get("node_ops", []), NODE_OP_TYPES, "node_ops", errors)
    _validate_op_types(data.get("edge_ops", []), EDGE_OP_TYPES, "edge_ops", errors)
    _validate_op_types(data.get("weight_ops", []), WEIGHT_OP_TYPES, "weight_ops", errors)

    plan = None
    if not errors:
        plan = GraphUpdatePlan.from_dict(data)
        errors.extend(plan.validate_basic())

    return PlanValidationResult(
        valid=not errors,
        plan=plan if not errors else None,
        errors=errors,
        raw_plan=data,
    )


def validate_plan_json(text: str) -> PlanValidationResult:
    """Parse and validate GraphUpdatePlan JSON text."""
    try:
        data = parse_json_object(text)
    except Exception as exc:
        return PlanValidationResult(
            valid=False,
            plan=None,
            errors=[f"invalid json: {exc}"],
            raw_plan=None,
        )
    return validate_plan_dict(data)


def _validate_op_types(items: Any, allowed: set, field_name: str,
                       errors: List[str]) -> None:
    if not isinstance(items, list):
        return
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"{field_name}[{idx}] must be an object")
            continue
        op = str(item.get("op", item.get("type", ""))).upper()
        if op not in allowed:
            errors.append(f"{field_name}[{idx}] invalid op: {op or '<missing>'}")
