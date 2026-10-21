"""Validated local corrections with explicit preconditions and provenance."""
from copy import deepcopy

from jsonschema import Draft4Validator

from dyrca_rca.interaction.observation import canonical, digest


def _schema(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


PATCH_SCHEMA = _schema({
    "collection": {"enum": ["services", "pods", "deployments"]},
    "entity": {"type": "string", "minLength": 1},
    "field": {"type": "string"},
    "expected": {}, "value": {},
})
TRACE_SCHEMA = _schema({
    "expected": {"type": "object"}, "value": {"type": "object"},
})
EDIT_SCHEMA = _schema({
    "section": {"enum": ["node_ops", "edge_ops", "weight_ops"]},
    "action": {"enum": ["add", "remove", "replace"]},
    "operation_id": {"type": "string"}, "value": {"type": "object"},
})
FIELDS = {
    "services": {"endpoint_ips": {"type": "array", "uniqueItems": True,
                                  "items": {"type": "string", "minLength": 1}}},
    "pods": {name: {"type": "string"} for name in ("service", "version", "status", "ip", "node")},
    "deployments": {"replicas": {"type": "integer", "minimum": 0},
                    "readyReplicas": {"type": "integer", "minimum": 0},
                    "version": {"type": "string"}},
}
GRANULAR_OPS = {"patch_topology", "correct_trace", "edit_plan"}


def validate_correction(event, observation):
    op, payload = event["operation"], event["payload"]
    if op == "edit_plan":
        Draft4Validator(EDIT_SCHEMA).validate(payload)
        if event["target_id"] != "graph_plan":
            raise ValueError("edit_plan target must be graph_plan")
        if payload["action"] == "add":
            if payload["operation_id"]:
                raise ValueError("add requires empty operation_id")
        elif not payload["operation_id"]:
            raise ValueError("remove/replace requires operation_id")
        if payload["action"] == "remove":
            if payload["value"]:
                raise ValueError("remove requires empty value")
        else:
            from dyrca_rca.interaction.pipeline import PLAN_SCHEMA
            Draft4Validator(PLAN_SCHEMA["properties"][payload["section"]]["items"]).validate(payload["value"])
            if not payload["value"].get("evidence"):
                raise ValueError("local graph edits require explicit evidence references")
        return
    facts = [e for e in observation["evidence"] if e["id"] == event["target_id"]]
    if len(facts) != 1:
        raise ValueError("unknown evidence id")
    fact = facts[0]
    if op == "patch_topology":
        Draft4Validator(PATCH_SCHEMA).validate(payload)
        if fact["kind"] != "topology":
            raise ValueError("patch_topology requires topology evidence")
        field_schema = FIELDS[payload["collection"]].get(payload["field"])
        if field_schema is None:
            raise ValueError("unsupported topology field")
        Draft4Validator(field_schema).validate(payload["value"])
        rows = [r for r in fact["data"][payload["collection"]] if r.get("name") == payload["entity"]]
        if len(rows) != 1:
            raise ValueError("topology entity must identify exactly one row")
    else:
        Draft4Validator(TRACE_SCHEMA).validate(payload)
        keys = ["span_count", "status_count", "error_count"] if fact["kind"] == "trace_service" else ["count"]
        if fact["kind"] not in ("trace_service", "trace_route"):
            raise ValueError("correct_trace requires trace aggregate")
        counts = _schema({k: {"type": "integer", "minimum": 0} for k in keys})
        Draft4Validator(counts).validate(payload["value"])
        Draft4Validator(counts).validate(payload["expected"])
        v = payload["value"]
        if fact["kind"] == "trace_service" and not v["error_count"] <= v["status_count"] <= v["span_count"]:
            raise ValueError("trace counts must satisfy error <= status <= span")


def apply_evidence_correction(view, event):
    validate_correction(event, view)
    fact = next(e for e in view["evidence"] if e["id"] == event["target_id"])
    p = event["payload"]
    if event["operation"] == "patch_topology":
        row = next(r for r in fact["data"][p["collection"]] if r.get("name") == p["entity"])
        if canonical(row.get(p["field"])) != canonical(p["expected"]):
            raise ValueError("topology correction precondition conflict")
        row[p["field"]] = deepcopy(p["value"])
    else:
        if any(fact["data"].get(k) != v for k, v in p["expected"].items()):
            raise ValueError("trace correction precondition conflict")
        fact["data"].update(deepcopy(p["value"]))
    # IDs remain source references; effective values explicitly carry provenance.
    fact.setdefault("correction_ids", []).append(event["event_id"])


def operation_id(section, operation):
    return "op:" + digest([section, operation])


def edit_plan(plan, events):
    plan = deepcopy(plan)
    for event in events:
        p = event["payload"]
        ops = plan[p["section"]]
        if p["action"] == "add":
            if p["value"] in ops:
                raise ValueError("duplicate graph operation")
            ops.append(deepcopy(p["value"]))
        else:
            matches = [i for i, op in enumerate(ops) if operation_id(p["section"], op) == p["operation_id"]]
            if len(matches) != 1:
                raise ValueError("graph operation precondition conflict")
            index = matches[0]
            if p["action"] == "remove":
                ops.pop(index)
            else:
                ops[index] = deepcopy(p["value"])
    return plan
