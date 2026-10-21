"""Immutable revisions produced by validated, replayable correction events."""
from __future__ import annotations

from copy import deepcopy

from jsonschema import Draft4Validator

from dyrca_rca.interaction.observation import canonical, digest
from dyrca_rca.interaction.granular import GRANULAR_OPS, validate_correction, apply_evidence_correction

FEEDBACK_PROTOCOL = "granular-feedback-v1"


EVENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["event_id", "incident_id", "base_revision", "operation", "target_id",
                 "payload", "reason", "actor", "scope"],
    "properties": {
        "event_id": {"type": "string", "minLength": 1},
        "incident_id": {"type": "string", "minLength": 1},
        "base_revision": {"type": "integer", "minimum": 0},
        "operation": {"enum": ["exclude_evidence", "restore_evidence", "replace_plan", "retract_event",
                                "accept_feedback", "reject_feedback", "retract_feedback", "supersede_feedback",
                                "patch_topology", "correct_trace", "edit_plan"]},
        "event_kind": {"enum": ["feedback", "decision"]},
        "target_id": {"type": "string", "minLength": 1},
        "payload": {"type": "object"},
        "reason": {"type": "string", "minLength": 1},
        "actor": {"type": "string", "minLength": 1},
        "scope": {"type": "object", "additionalProperties": False,
                  "required": ["system_id", "as_of"], "properties": {
                      "system_id": {"type": "string"}, "as_of": {"type": "string"}}},
        "verification_status": {"enum": ["proposed", "verified"]},
    },
}


def new_session(observation):
    return {"incident_id": observation["incident_id"], "revision": 0,
            "state_revision": 0, "audit_seq": 0, "interaction_state_version": 2,
            "observation": deepcopy(observation), "events": [], "outcomes": {}}


FEEDBACK_OPS = {"exclude_evidence", "restore_evidence", "replace_plan"} | GRANULAR_OPS


def _require_v2(session):
    if session.get("interaction_state_version") != 2:
        raise ValueError("v1 session is read-only; create a new v2 observation baseline")


def feedback_statuses(session):
    """Derive lifecycle from immutable commands and system execution receipts."""
    _require_v2(session)
    statuses = {}
    for event in session["events"]:
        op, eid = event["operation"], event["event_id"]
        outcome = session["outcomes"][eid]
        if op in FEEDBACK_OPS:
            statuses[eid] = "proposed"
        if outcome["execution_status"] != "succeeded":
            continue
        if op in FEEDBACK_OPS:
            statuses[eid] = "accepted"
        else:
            statuses[event["target_id"]] = {
                "accept_feedback": "accepted", "reject_feedback": "rejected",
                "retract_feedback": "retracted", "retract_event": "retracted",
                "supersede_feedback": "superseded"}[op]
        for old_id in outcome.get("superseded_ids", []):
            statuses[old_id] = "superseded"
    return statuses


def effective_state(session):
    statuses = feedback_statuses(session)
    excluded = set()
    plan = None
    corrections = []
    edits = []
    # Apply in acceptance order, which may differ from proposal order.
    ordered = []
    by_id = {e["event_id"]: e for e in session["events"]}
    for command in session["events"]:
        if session["outcomes"][command["event_id"]]["execution_status"] != "succeeded":
            continue
        if command["operation"] in FEEDBACK_OPS:
            ordered.append(command)
        elif command["operation"] == "accept_feedback":
            ordered.append(by_id[command["target_id"]])
    for event in ordered:
        if statuses.get(event["event_id"]) != "accepted":
            continue
        op = event["operation"]
        if op == "exclude_evidence":
            excluded.add(event["target_id"])
        elif op == "restore_evidence":
            excluded.discard(event["target_id"])
        elif op == "replace_plan":
            plan = deepcopy(event["payload"])
        elif op == "edit_plan":
            edits.append(deepcopy(event))
        elif op in GRANULAR_OPS:
            corrections.append(event)
    view = deepcopy(session["observation"])
    view["evidence"] = [e for e in view["evidence"] if e["id"] not in excluded]
    for event in corrections:
        apply_evidence_correction(view, event)
    if edits:
        view["plan_edits"] = edits
    return view, plan


def apply_event(session, event, validate_state):
    """Return an immutable audit update, committing formal state on success.

    validate_state must recompute/validate the effective graph; it must not write
    external state. Failed explicit decisions return an audit-only receipt;
    invalid commands and failed compatibility submissions raise. The caller's
    session is never mutated.
    """
    _require_v2(session)
    Draft4Validator(EVENT_SCHEMA).validate(event)
    canonical(event)  # Reject NaN/Infinity, including inside arbitrary payloads.
    for old in session["events"]:
        if old["event_id"] == event["event_id"]:
            if canonical(old) != canonical(event):
                raise ValueError("event_id already used with different content")
            return deepcopy(session)
    view = session["observation"]
    if event["incident_id"] != session["incident_id"]:
        raise ValueError("incident mismatch")
    if event["base_revision"] != session["state_revision"]:
        raise ValueError("stale base_revision")
    if event["scope"] != {"system_id": view["system_id"], "as_of": view["as_of"]}:
        raise ValueError("event scope does not match observation")
    op = event["operation"]
    is_feedback = op in FEEDBACK_OPS
    if event.get("event_kind", "feedback" if is_feedback else "decision") != ("feedback" if is_feedback else "decision"):
        raise ValueError("event_kind does not match operation")
    if op not in GRANULAR_OPS | {"replace_plan"} and event["payload"]:
        raise ValueError("this operation requires an empty payload")
    if op in GRANULAR_OPS:
        validate_correction(event, view)
    elif event["operation"] in ("exclude_evidence", "restore_evidence"):
        if event["target_id"] not in {e["id"] for e in view["evidence"]}:
            raise ValueError("unknown evidence id")
    elif event["operation"] == "replace_plan":
        if event["target_id"] != "graph_plan":
            raise ValueError("replace_plan target must be graph_plan")
    else:
        targets = [e for e in session["events"] if e["event_id"] == event["target_id"]]
        if not targets or targets[0]["operation"] not in FEEDBACK_OPS:
            raise ValueError("decision must target existing feedback")
        expected = "proposed" if op in ("accept_feedback", "reject_feedback") else "accepted"
        if feedback_statuses(session)[event["target_id"]] != expected:
            raise ValueError("invalid feedback lifecycle transition")
        if event.get("verification_status") == "proposed":
            raise ValueError("decision cannot be marked proposed")
    # New feedback always starts proposed. Legacy verified input is an explicit
    # local-operator submit-and-accept shorthand, validated atomically below.
    proposed = is_feedback and (event.get("event_kind") == "feedback" or event.get("verification_status") != "verified")
    candidate = deepcopy(session)
    candidate["events"].append(deepcopy(event))
    candidate["audit_seq"] += 1
    receipt = {"execution_status": "pending" if proposed else "succeeded"}
    candidate["outcomes"][event["event_id"]] = receipt
    if proposed or op == "reject_feedback":
        return candidate
    target = event if is_feedback else targets[0]
    if target["operation"] == "replace_plan" and (is_feedback or op == "accept_feedback"):
        statuses = feedback_statuses(session)
        receipt["superseded_ids"] = [e["event_id"] for e in session["events"]
                                     if e["operation"] == "replace_plan" and statuses[e["event_id"]] == "accepted"]
    candidate["state_revision"] += 1
    candidate["revision"] = candidate["state_revision"]
    try:
        validate_state(candidate)
    except Exception as exc:
        if is_feedback or op == "retract_event":
            raise
        candidate["state_revision"] = candidate["revision"] = session["state_revision"]
        candidate["outcomes"][event["event_id"]] = {
            "execution_status": "failed", "error_type": type(exc).__name__}
    return candidate


def propose_event(session, feedback):
    if feedback.get("operation") not in FEEDBACK_OPS:
        raise ValueError("expected feedback operation")
    return apply_event(session, dict(feedback, event_kind="feedback"), lambda _: None)


def commit_decision(session, decision, validate_state):
    if decision.get("operation") in FEEDBACK_OPS:
        raise ValueError("expected decision operation")
    return apply_event(session, dict(decision, event_kind="decision"), validate_state)


def preview_session(session, feedback_id):
    """Return an in-memory hypothetical acceptance, never a persisted commit."""
    if feedback_statuses(session).get(feedback_id) != "proposed":
        raise ValueError("preview requires proposed feedback")
    feedback = next(e for e in session["events"] if e["event_id"] == feedback_id)
    decision = dict(feedback, event_id="preview:" + digest(session), event_kind="decision",
                    operation="accept_feedback", target_id=feedback_id, payload={},
                    base_revision=session["state_revision"], verification_status="verified")
    return commit_decision(session, decision, lambda _: None)


def session_key(session, model_version, planner):
    from dyrca_rca.dataset.materialize import INTERACTIVE_FEATURE_PROTOCOL
    return digest({"pipeline": "interactive-v2-state-semantics", "feedback_protocol": FEEDBACK_PROTOCOL,
                   "feature_protocol": INTERACTIVE_FEATURE_PROTOCOL, "session": session,
                   "model": model_version, "planner": planner})
