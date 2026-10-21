"""Replay effective observations into graphs and frozen-model predictions."""
from __future__ import annotations

from copy import deepcopy
from math import log1p

from jsonschema import Draft4Validator

from dyrca_rca.dataset.materialize import (build_graph_from_topology, graph_to_model_tensors,
                                          INTERACTIVE_FEATURE_PROTOCOL)
from dyrca_rca.graph.executor import GraphExecutor
from dyrca_rca.graph.update_ops import GraphUpdatePlan
from dyrca_rca.graph.weights import WEIGHT_COEFFICIENTS
from dyrca_rca.interaction.observation import canonical, epoch
from dyrca_rca.interaction.state import effective_state, preview_session
from dyrca_rca.interaction.granular import edit_plan, operation_id
from dyrca_rca.planner.plan_validator import validate_plan_dict, parse_json_object


def _operation_schema(kind):
    fields = {
        "node": ("node_id", "old_node", "new_node", "parent", "service", "version"),
        "edge": ("source", "target", "old_source", "old_target", "new_source", "new_target", "service", "edge_type"),
        "weight": ("source", "target"),
    }[kind]
    properties = {name: {"type": "string"} for name in fields}
    properties.update({
        "op": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "attrs": {"type": "object", "additionalProperties": False, "properties": {
            "service": {"type": "string"}, "version": {"type": "string"},
            "route_weight": {"type": "number", "minimum": 0, "maximum": 1},
        }},
    })
    if kind == "node":
        properties["children"] = {"type": "array", "items": {"type": "string"}}
    else:
        properties["weight"] = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
    return {"type": "object", "additionalProperties": False,
            "required": ["op"], "properties": properties}


PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["experiment_id", "change_type", "confidence", "causal_role",
                 "node_ops", "edge_ops", "weight_ops", "state_transfer", "evidence"],
    "properties": {
        "experiment_id": {"type": "string"}, "change_type": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "causal_role": {"enum": ["unknown", "hypothesis"]},
        "node_ops": {"type": "array", "items": _operation_schema("node")},
        "edge_ops": {"type": "array", "items": _operation_schema("edge")},
        "weight_ops": {"type": "array", "items": _operation_schema("weight")},
        "state_transfer": {"type": "object", "maxProperties": 0},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}


def base_graph(view):
    snapshots = [e for e in view["evidence"] if e["kind"] == "topology"]
    if not snapshots:
        raise ValueError("correction leaves no usable topology")
    latest = max(snapshots, key=lambda e: epoch(e["data"]["timestamp"]))
    graph = build_graph_from_topology(latest["data"], phase="observed",
                                      dependency_edges=view["dependency_edges"])
    if not graph["nodes"]:
        raise ValueError("empty observed graph")
    graph["attrs"] = {"snapshot_ref": latest["id"], "graph_variant": "interactive_v1"}
    return graph, latest


def rule_plan(view):
    graph, latest = base_graph(view)
    plan = {"experiment_id": view["incident_id"], "change_type": "observed_update",
            "confidence": 1.0, "causal_role": "unknown", "node_ops": [],
            "edge_ops": [], "weight_ops": [], "state_transfer": {}, "evidence": [latest["id"]]}
    for svc in latest["data"]["services"]:
        # Missing endpoint data is unknown, not proof of an empty endpoint set.
        if svc.get("name") and svc.get("endpoint_ips") == []:
            plan["node_ops"].append({"op": "ADD_BLACKHOLE_NODE", "service": svc["name"]})
    nodes = {n["id"] for n in graph["nodes"]}
    pairs = {(e["source"], e["target"]) for e in graph["edges"]}
    for fact in view["evidence"]:
        if fact["kind"] != "trace_route":
            continue
        if fact["data"]["count"] <= 0:
            continue
        s, t = fact["data"]["source"], fact["data"]["target"]
        if s in nodes and t in nodes and (s, t) not in pairs:
            plan["edge_ops"].append({"op": "ADD_EDGE", "source": s, "target": t,
                                      "edge_type": "dependency", "evidence": [fact["id"]]})
            plan["evidence"].append(fact["id"])
            pairs.add((s, t))
    return plan


def validate_and_execute(view, data):
    Draft4Validator(PLAN_SCHEMA).validate(data)
    canonical(data)
    if data["experiment_id"] != view["incident_id"]:
        raise ValueError("plan incident mismatch")
    active = {e["id"] for e in view["evidence"]}
    refs = list(data["evidence"])
    for key in ("node_ops", "edge_ops", "weight_ops"):
        for op in data[key]:
            refs.extend(op.get("evidence", []))
    if any(ref not in active for ref in refs):
        raise ValueError("plan references excluded or unknown evidence")
    if any(data[k] for k in ("node_ops", "edge_ops", "weight_ops")) and not refs:
        raise ValueError("graph changes require evidence references")
    for op in data["weight_ops"]:
        if "route_weight" not in op.get("attrs", {}):
            raise ValueError("weight corrections require attrs.route_weight, not a final model weight")
        if op.get("weight") is not None and op["weight"] != op["attrs"]["route_weight"]:
            raise ValueError("weight and attrs.route_weight disagree")
    validation = validate_plan_dict(data)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    graph, _ = base_graph(view)
    # Executor edge edits address endpoints rather than (endpoints, type).
    # Reject ambiguous edits instead of silently editing the first typed edge.
    plan = validation.plan
    executor = GraphExecutor(graph)
    executor.apply_plan(GraphUpdatePlan(
        experiment_id=plan.experiment_id, change_type=plan.change_type,
        confidence=plan.confidence, causal_role=plan.causal_role,
        node_ops=plan.node_ops, evidence=plan.evidence,
    ))
    # Check against the evolving trial graph, including edges added earlier in
    # the same transaction, not just the initial graph.
    for op in plan.edge_ops + plan.weight_ops:
        if op.op != "ADD_EDGE":
            s = getattr(op, "old_source", "") or op.source
            t = getattr(op, "old_target", "") or op.target
            if sum(e["source"] == s and e["target"] == t for e in executor.edges) > 1:
                raise ValueError("ambiguous edge endpoints")
        if op in plan.edge_ops:
            executor.apply_edge_op(op)
        else:
            executor.apply_weight_op(op)
    if executor.warnings:
        raise ValueError("graph execution rejected: " + "; ".join(executor.warnings))
    graph = executor.to_dict()
    # Legacy executor preserves tombstones for history. Exclude them from the
    # active inference graph; the plan itself retains the audit history.
    graph["nodes"] = [n for n in graph["nodes"] if not n.get("attrs", {}).get("removed")]
    ids = {n["id"] for n in graph["nodes"]}
    graph["edges"] = [e for e in graph["edges"] if e["source"] in ids and e["target"] in ids
                      and e.get("attrs", {}).get("active", True)
                      and not e.get("attrs", {}).get("removed")]
    if not ids:
        raise ValueError("plan removed every node")
    graph["nodes"].sort(key=lambda n: n["id"])
    graph["edges"].sort(key=lambda e: (e["source"], e["target"], e["edge_type"]))
    return _weight_observed_graph(graph, view)


def _weight_observed_graph(graph, view):
    """Encode observed trace rates and missingness using only active evidence."""
    stats = {e["data"]["service"]: e["data"] for e in view["evidence"] if e["kind"] == "trace_service"}
    nodes = {n["id"]: n.get("attrs", {}) for n in graph["nodes"]}
    trace_state = view["modalities"]["traces"]
    partial = trace_state["status"] == "truncated" or trace_state.get("unknown_timestamp_rows", 0) > 0
    for node_id, attrs in nodes.items():
        st = stats.get(attrs.get("service", node_id), {})
        total, count = st.get("span_count", 0), st.get("status_count", 0)
        attrs["trace_anomaly"] = st.get("error_count", 0) / count if count else None
        attrs["trace_observed"] = float(count > 0)
        attrs["trace_status_coverage"] = count / total if total else 0.0
        attrs["trace_log_sample_count"] = log1p(count)
        attrs["trace_partial"] = float(partial or count < total)
    for edge in graph["edges"]:
        attrs = edge.setdefault("attrs", {})
        target = nodes[edge["target"]]
        attrs["dependency_weight"] = float(attrs.get("route_weight", attrs.get("dependency_weight", 1.0)))
        attrs["anomaly_weight"] = target["trace_anomaly"]
        attrs["anomaly_observed"] = target["trace_observed"]
        attrs["trace_status_coverage"] = target["trace_status_coverage"]
        attrs["trace_partial"] = target["trace_partial"]
        attrs["topology_confidence"] = 1.0
        attrs["change_weight"] = float(bool(attrs.get("op")))
        available = [k for k in WEIGHT_COEFFICIENTS if attrs[k] is not None]
        attrs["final_weight"] = round(sum(WEIGHT_COEFFICIENTS[k] * attrs[k] for k in available)
                                      / sum(WEIGHT_COEFFICIENTS[k] for k in available), 8)
        edge["weight"] = attrs["final_weight"]
    graph["attrs"]["feature_protocol"] = INTERACTIVE_FEATURE_PROTOCOL
    return graph


def resolve_plan(session, planner="rule", llm_client=None, mode="committed", feedback_id=None):
    if mode == "preview":
        session = preview_session(session, feedback_id)
    elif mode != "committed" or feedback_id is not None:
        raise ValueError("invalid replay mode or feedback_id")
    view, override = effective_state(session)
    edits = view.pop("plan_edits", [])
    if edits:
        # Build the base proposal first; correction failures must not be caught
        # as LLM failures and silently fall back to an uncorrected graph.
        _, plan, _, _, reason = _resolve_base(view, override, planner, llm_client)
        plan = edit_plan(plan, edits)
        return view, plan, validate_and_execute(view, plan), "human", reason
    return _resolve_base(view, override, planner, llm_client)


def _resolve_base(view, override, planner, llm_client):
    if override is not None:
        return view, override, validate_and_execute(view, override), "human", None
    fallback_reason = None
    if planner == "llm":
        try:
            if llm_client is None:
                raise ValueError("no LLM client supplied")
            prompt = ("Propose an observed dependency graph update, not a proven root cause. "
                      "Return JSON matching this schema. Cite only active evidence IDs.\n"
                      + canonical(PLAN_SCHEMA) + "\nObservation:\n" + canonical(view))
            plan = parse_json_object(llm_client(prompt))
            return view, plan, validate_and_execute(view, plan), "llm", None
        except Exception as exc:
            # Do not persist arbitrary backend error messages (credentials/URLs).
            fallback_reason = type(exc).__name__
    elif planner != "rule":
        raise ValueError("unknown planner")
    plan = rule_plan(view)
    return view, plan, validate_and_execute(view, plan), "rule", fallback_reason


def replay(session, model=None, model_name=None, planner="rule", llm_client=None,
           mode="committed", feedback_id=None, feature_protocol=None, mask_temporal=False):
    view, plan, graph, source, reason = resolve_plan(session, planner, llm_client, mode, feedback_id)
    from dyrca_rca.interaction.temporal import PROTOCOL, enrich_graph, diagnosis_support
    protocol = feature_protocol or getattr(model, "feature_protocol", INTERACTIVE_FEATURE_PROTOCOL)
    if protocol == PROTOCOL:
        graph = enrich_graph(graph, view, mask_temporal)
    elif protocol != INTERACTIVE_FEATURE_PROTOCOL or mask_temporal:
        raise ValueError("unsupported interactive feature protocol")
    item = graph_to_model_tensors(graph, {"experiment_id": view["incident_id"]})
    # Inference inputs contain no evaluation labels, even placeholder labels.
    item.pop("label", None)
    item["metadata"] = {"experiment_id": view["incident_id"], "feature_protocol": graph["attrs"]["feature_protocol"],
                        "node_feature_names": item["metadata"]["node_feature_names"],
                        "edge_feature_names": item["metadata"]["edge_feature_names"]}
    prediction = None
    if model is not None:
        from dyrca_rca.experiments.inference import predict_observed
        prediction = predict_observed(model, item, model_name)
    evidence_pack = _pack(view, graph, prediction)
    return {"incident_id": view["incident_id"], "revision": session["revision"],
            "harness": diagnosis_support(view, graph, prediction) if protocol == PROTOCOL else None,
            "plan_operations": [{"operation_id": operation_id(section, op),
                                 "section": section, "operation": deepcopy(op)}
                                for section in ("node_ops", "edge_ops", "weight_ops") for op in plan[section]],
            "state_revision": session["state_revision"], "mode": mode, "feedback_id": feedback_id,
            "observation_version": session["observation"]["observation_version"],
            "plan_source": source, "fallback_reason": reason, "plan": plan,
            "graph": graph, "prediction": prediction, "evidence_pack": evidence_pack,
            "diagnosis_status": "predicted" if prediction else "graph_only_no_checkpoint"}, item


def _pack(view, graph, prediction):
    candidates = []
    if prediction is not None:
        for rank, node_id in enumerate(prediction["ranked_node_ids"], 1):
            refs = [e["id"] for e in view["evidence"] if e["id"] == graph["attrs"]["snapshot_ref"]
                    or e["data"].get("service") == node_id
                    or node_id in (e["data"].get("source"), e["data"].get("target"))]
            candidates.append({"candidate": node_id, "original_rank": rank,
                               "model_score": prediction["scores"][node_id], "evidence_refs": refs})
    return {"incident_id": view["incident_id"], "candidates": candidates,
            "evidence": deepcopy(view["evidence"]), "modalities": deepcopy(view["modalities"])}
