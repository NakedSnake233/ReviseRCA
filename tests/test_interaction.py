"""Invariant tests for observable state, atomic corrections and frozen replay."""
import copy
import json
from pathlib import Path

import pytest
import torch

from dyrca_rca.dataset.materialize import (INTERACTIVE_NODE_FEATURE_NAMES as NODE_FEATURE_NAMES,
                                          INTERACTIVE_EDGE_FEATURE_NAMES as EDGE_FEATURE_NAMES)
from dyrca_rca.experiments.inference import save_checkpoint, load_checkpoint
from dyrca_rca.experiments.run_interactive import parse_args, run, PROTOCOL
from dyrca_rca.experiments.train import create_model
from dyrca_rca.interaction.observation import load_observation, canonical
from dyrca_rca.interaction.pipeline import replay, rule_plan
from dyrca_rca.interaction.state import (new_session, apply_event, effective_state, session_key,
                                         propose_event, commit_decision, feedback_statuses)
from dyrca_rca.models.config import set_seed
from dyrca_rca.rerank.llm_reranker import build_rerank_prompt


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tmp_path(tmpdir):
    # The project's default Python environment still uses pytest 3.x.
    return Path(str(tmpdir))


@pytest.fixture
def experiment(tmp_path):
    exp = tmp_path / "data" / "t3-secret-target"
    write(exp / "metadata.json", {"environment": {"namespace": "boutique"},
                                   "target_service": "SECRET_TARGET", "scenario_name": "SECRET_SCENARIO"})
    write(exp / "ground_truth.json", {"root_cause_service": "SECRET_LABEL"})
    snapshot = {
        "timestamp": "2026-01-01T00:00:00Z",
        "pods": [{"name": "a-pod", "service": "a", "ip": "1", "version": "v1"},
                 {"name": "b-pod", "service": "b", "ip": "2", "version": "v1"}],
        "services": [{"name": "a", "endpoint_ips": ["1"]}, {"name": "b", "endpoint_ips": ["2"]}],
        "deployments": [{"name": "a", "replicas": 1}, {"name": "b", "replicas": 1}],
        "root_cause_service": "SECRET_TOPOLOGY_LABEL",
    }
    write(exp / "topology" / "pre.json", snapshot)
    during = copy.deepcopy(snapshot)
    during["timestamp"] = "2026-01-01T00:01:00Z"
    during["services"][0]["endpoint_ips"] = []
    write(exp / "topology" / "during.json", during)
    write(exp / "topology" / "post.json", {"secret": "FUTURE_POST"})
    traces = exp / "traces" / "trace_summary.csv"
    traces.parent.mkdir()
    traces.write_text(
        "trace_id,span_id,parent_span_id,service,start_time_unix_nano,duration_nano,status_code\n"
        "t,1,,a,1767225610000000000,1000000,2\n"
        "t,2,1,b,1767225610000100000,1000000,0\n"
        "future,3,,FUTURE_SERVICE,1767225800000000000,1000000,2\n"
        "unknown,4,,UNKNOWN_TIME,,1000000,2\n", encoding="utf-8")
    return exp


def observe(exp, **kwargs):
    return load_observation(exp, "dataset", "boutique", dependencies={"boutique": [["a", "b"]]}, **kwargs)


def event(session, operation, target_id, payload=None, event_id="event-1"):
    view = session["observation"]
    return {"event_id": event_id, "incident_id": session["incident_id"],
            "base_revision": session["revision"], "operation": operation, "target_id": target_id,
            "payload": payload or {}, "reason": "verified against the incident observation",
            "actor": "engineer", "verification_status": "verified",
            "scope": {"system_id": view["system_id"], "as_of": view["as_of"]}}


def validate(session):
    replay(session)


def test_proposal_preview_accept_and_retract(experiment):
    original = new_session(observe(experiment))
    ref = next(e["id"] for e in original["observation"]["evidence"]
               if e["kind"] == "topology" and e["data"]["phase"] == "during")
    proposed = propose_event(original, event(original, "exclude_evidence", ref))
    assert proposed["audit_seq"] == 1 and proposed["state_revision"] == 0
    assert replay(proposed)[0] == replay(original)[0]
    preview, _ = replay(proposed, mode="preview", feedback_id="event-1")
    assert preview["graph"] != replay(original)[0]["graph"]
    accepted = commit_decision(proposed, event(proposed, "accept_feedback", "event-1", event_id="accept"), validate)
    assert accepted["state_revision"] == 1 and accepted["audit_seq"] == 2
    assert replay(accepted)[0]["graph"] == preview["graph"]
    retracted = commit_decision(accepted, event(accepted, "retract_feedback", "event-1", event_id="undo"), validate)
    assert replay(retracted)[0]["graph"] == replay(original)[0]["graph"]
    with pytest.raises(ValueError, match="lifecycle"):
        commit_decision(retracted, event(retracted, "accept_feedback", "event-1", event_id="again"), validate)


def test_rejection_and_failed_accept_are_audit_only(experiment):
    original = new_session(observe(experiment))
    plan = rule_plan(original["observation"])
    plan["edge_ops"] = [{"op": "REMOVE_EDGE", "source": "absent", "target": "a"}]
    proposed = propose_event(original, event(original, "replace_plan", "graph_plan", plan))
    failed = commit_decision(proposed, event(proposed, "accept_feedback", "event-1", event_id="bad"), validate)
    assert failed["outcomes"]["bad"]["execution_status"] == "failed"
    assert failed["state_revision"] == 0 and failed["audit_seq"] == 2
    assert feedback_statuses(failed)["event-1"] == "proposed"
    rejected = commit_decision(failed, event(failed, "reject_feedback", "event-1", event_id="reject"), validate)
    assert replay(rejected)[0] == replay(original)[0]
    assert rejected["audit_seq"] == 3
    with pytest.raises(ValueError, match="lifecycle"):
        commit_decision(rejected, event(rejected, "accept_feedback", "event-1", event_id="again"), validate)


def test_plans_supersede_atomically_and_v1_is_not_reinterpreted(experiment):
    session = new_session(observe(experiment))
    plan = rule_plan(session["observation"])
    first = apply_event(session, event(session, "replace_plan", "graph_plan", plan), validate)
    second = apply_event(first, event(first, "replace_plan", "graph_plan", plan, event_id="second"), validate)
    assert feedback_statuses(second) == {"event-1": "superseded", "second": "accepted"}
    restored = commit_decision(second, event(second, "retract_feedback", "second", event_id="undo"), validate)
    assert effective_state(restored)[1] is None  # Superseded plans do not revive.
    legacy = dict(session)
    legacy.pop("interaction_state_version")
    with pytest.raises(ValueError, match="v1 session"):
        replay(legacy)


def test_proposed_legacy_restore_does_not_take_effect(experiment):
    session = new_session(observe(experiment))
    ref = session["observation"]["evidence"][0]["id"]
    excluded = apply_event(session, event(session, "exclude_evidence", ref), validate)
    restore = event(excluded, "restore_evidence", ref, event_id="restore")
    restore["verification_status"] = "proposed"
    proposed = apply_event(excluded, restore, validate)
    assert effective_state(proposed) == effective_state(excluded)
    stale = event(session, "accept_feedback", "restore", event_id="stale")
    with pytest.raises(ValueError, match="stale"):
        commit_decision(proposed, stale, validate)


def test_feedback_replays_in_acceptance_order(experiment):
    session = new_session(observe(experiment))
    ref = session["observation"]["evidence"][0]["id"]
    session = propose_event(session, event(session, "exclude_evidence", ref, event_id="exclude"))
    session = propose_event(session, event(session, "restore_evidence", ref, event_id="restore"))
    session = commit_decision(session, event(session, "accept_feedback", "restore", event_id="accept-restore"), validate)
    session = commit_decision(session, event(session, "accept_feedback", "exclude", event_id="accept-exclude"), validate)
    assert ref not in {e["id"] for e in effective_state(session)[0]["evidence"]}
    before = canonical(session)
    decision = event(session, "retract_feedback", "exclude", event_id="failed-retract")

    def failed_replay(_):
        raise ValueError("inference unavailable")

    failed = commit_decision(session, decision, failed_replay)
    assert canonical(session) == before
    assert effective_state(failed) == effective_state(session)
    assert failed["state_revision"] == session["state_revision"]
    assert failed["outcomes"]["failed-retract"]["execution_status"] == "failed"


def test_runner_proposal_resume_preserves_current(experiment, tmp_path):
    out = tmp_path / "output"
    args = parse_args(["--data-root", str(experiment.parent), "--out-dir", str(out), "--limit", "1"])
    run(args)
    manifest = json.loads((out / "manifest.jsonl").read_text())
    result_path = Path(manifest["result_path"])
    root = result_path.parent.parent
    session = json.loads(result_path.with_name("session.json").read_text())
    pointer = (root / "current.json").read_text()
    ref = session["observation"]["evidence"][0]["id"]
    proposal = dict(event(session, "exclude_evidence", ref), event_kind="feedback")
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(canonical(proposal) + "\n")
    args.feedback = str(feedback)
    run(args)
    run(args)
    assert (root / "current.json").read_text() == pointer
    audited = json.loads((root / "audit.json").read_text())
    assert audited["audit_seq"] == 1 and audited["state_revision"] == 0
    assert json.loads((root / "preview.json").read_text())["mode"] == "preview"
    decision = dict(event(audited, "accept_feedback", "event-1", event_id="accept"), event_kind="decision")
    feedback.write_text(canonical(proposal) + "\n" + canonical(decision) + "\n")
    run(args)
    assert json.loads((root / "current.json").read_text())["revision"] == 1


def test_observation_hides_labels_and_future_data(experiment):
    view = observe(experiment)
    serialized = canonical(view)
    for secret in ("SECRET", "FUTURE", "UNKNOWN_TIME", experiment.name, "target_service", "root_cause"):
        assert secret not in serialized
    assert view["modalities"]["traces"]["eligible_rows"] == 2
    assert view["modalities"]["traces"]["unknown_timestamp_rows"] == 1
    earlier = observe(experiment, as_of="2026-01-01T00:00:05Z")
    assert len(earlier["evidence"]) == 1
    assert earlier["evidence"][0]["data"]["phase"] == "pre"
    limited = observe(experiment, max_rows=1)
    assert limited["modalities"]["traces"]["status"] == "truncated"
    assert limited["modalities"]["traces"]["rows_scanned"] == 1


def test_missing_endpoints_do_not_imply_blackhole(experiment):
    path = experiment / "topology" / "during.json"
    snapshot = json.loads(path.read_text())
    del snapshot["services"][0]["endpoint_ips"]
    write(path, snapshot)
    result, _ = replay(new_session(observe(experiment)))
    assert not any(n["id"].startswith("blackhole:") for n in result["graph"]["nodes"])


def test_exclusion_propagates_to_graph_and_evidence_and_can_be_retracted(experiment):
    original = new_session(observe(experiment))
    baseline, _ = replay(original)
    during = next(e["id"] for e in original["observation"]["evidence"]
                  if e["kind"] == "topology" and e["data"]["phase"] == "during")
    correction = event(original, "exclude_evidence", during)
    corrected = apply_event(original, correction, validate)
    result, tensor = replay(corrected)
    assert any(n["id"] == "blackhole:a" for n in baseline["graph"]["nodes"])
    assert not any(n["id"] == "blackhole:a" for n in result["graph"]["nodes"])
    assert during not in canonical(result["evidence_pack"])
    assert during not in canonical(result["plan"])
    assert "label" not in tensor
    assert original["revision"] == 0
    assert session_key(original, "model", "rule") != session_key(corrected, "model", "rule")
    assert apply_event(corrected, correction, validate) == corrected  # Retry deduplicates before stale check.
    restored = apply_event(corrected, event(corrected, "retract_event", correction["event_id"], event_id="undo"), validate)
    after, _ = replay(restored)
    assert after["graph"] == baseline["graph"]


def test_stale_conflicting_and_invalid_changes_leave_session_untouched(experiment):
    session = new_session(observe(experiment))
    first_ref = session["observation"]["evidence"][0]["id"]
    change = event(session, "exclude_evidence", first_ref)
    changed = apply_event(session, change, validate)
    wrong = copy.deepcopy(change)
    wrong["reason"] = "different payload"
    with pytest.raises(ValueError, match="different content"):
        apply_event(changed, wrong, validate)
    stale = event(session, "restore_evidence", first_ref, event_id="stale")
    with pytest.raises(ValueError, match="stale"):
        apply_event(changed, stale, validate)
    plan = rule_plan(session["observation"])
    plan["edge_ops"] = [{"op": "REMOVE_EDGE", "source": "absent", "target": "a"}]
    before = canonical(session)
    with pytest.raises(ValueError, match="graph execution rejected"):
        apply_event(session, event(session, "replace_plan", "graph_plan", plan), validate)
    assert canonical(session) == before


def test_route_weight_reaches_tensors_and_frozen_model(experiment, tmp_path):
    session = new_session(observe(experiment))
    set_seed(42)
    model = create_model("weighted_graphsage", len(NODE_FEATURE_NAMES), len(EDGE_FEATURE_NAMES))
    # Random initialized checkpoint is a test fixture, not a quality evaluation.
    checkpoint = tmp_path / "fixture.pt"
    save_checkpoint(model, checkpoint, "weighted_graphsage", PROTOCOL, {"fixture_only": True})
    frozen, name = load_checkpoint(checkpoint, PROTOCOL)
    weights = {k: v.clone() for k, v in frozen.state_dict().items()}
    baseline, before = replay(session, frozen, name)
    plan = rule_plan(session["observation"])
    plan["weight_ops"] = [{"op": "REWEIGHT_EDGE", "source": "a", "target": "b",
                           "weight": 0.2, "attrs": {"route_weight": 0.2}}]
    changed = apply_event(session, event(session, "replace_plan", "graph_plan", plan), validate)
    result, after = replay(changed, frozen, name)
    edge = next(e for e in result["graph"]["edges"] if (e["source"], e["target"]) == ("a", "b"))
    assert edge["attrs"]["dependency_weight"] == 0.2
    assert edge["weight"] == edge["attrs"]["final_weight"]
    assert not torch.equal(before["edge_attr"], after["edge_attr"])
    assert result["prediction"]["scores"] != baseline["prediction"]["scores"]
    assert all(torch.equal(weights[k], v) for k, v in frozen.state_dict().items())
    assert replay(changed, frozen, name)[0] == result
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_checkpoint(checkpoint, "legacy")


def test_llm_fallback_uses_effective_view_and_validates_schema(experiment):
    original = new_session(observe(experiment))
    ref = next(e["id"] for e in original["observation"]["evidence"] if e["kind"] == "trace_service")
    session = apply_event(original, event(original, "exclude_evidence", ref), validate)
    captured = []

    def invalid(prompt):
        captured.append(prompt)
        return '{"unknown": true}'

    result, _ = replay(session, planner="llm", llm_client=invalid)
    expected, _ = replay(session)
    assert result["graph"] == expected["graph"]
    assert result["plan_source"] == "rule"
    assert result["fallback_reason"]
    assert ref not in captured[0]
    assert "SECRET" not in captured[0]
    plan = rule_plan(effective_state(session)[0])
    valid, _ = replay(session, planner="llm", llm_client=lambda _: canonical(plan))
    assert valid["plan_source"] == "llm"


def test_runner_smoke_resume_and_event_retry(experiment, tmp_path):
    out = tmp_path / "output"
    args = parse_args(["--data-root", str(experiment.parent), "--out-dir", str(out),
                       "--limit", "1", "--seed", "42", "--planner", "rule"])
    run(args)
    manifest = json.loads((out / "manifest.jsonl").read_text())
    baseline_path = Path(manifest["result_path"])
    session = json.loads(baseline_path.with_name("session.json").read_text())
    ref = next(e["id"] for e in session["observation"]["evidence"] if e["kind"] == "trace_service")
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(canonical(event(session, "exclude_evidence", ref)) + "\n")
    args.feedback = str(feedback)
    run(args)
    manifest2 = json.loads((out / "manifest.jsonl").read_text())
    assert manifest2["result_path"] != manifest["result_path"]
    assert baseline_path.exists()
    result = Path(manifest2["result_path"]).read_text()
    assert json.loads(result)["revision"] == 1
    run(args)
    assert Path(manifest2["result_path"]).read_text() == result
    assert json.loads((out / "manifest.jsonl").read_text()) == manifest2


def test_legacy_prompt_removes_cached_labels_and_disabled_target_prior():
    prompt = build_rerank_prompt({"original_label_node_id": "SECRET_LABEL",
                                  "target_service": "SECRET_TARGET",
                                  "candidates": [{"root_cause_service": "SECRET_NESTED"}]}, use_target_prior=False)
    assert "SECRET" not in prompt
    assert "target_service" not in prompt


def test_unknown_evidence_and_nonfinite_plans_are_not_committed(experiment):
    session = new_session(observe(experiment))
    plan = rule_plan(session["observation"])
    plan["evidence"].append("invented-ref")
    with pytest.raises(ValueError, match="unknown evidence"):
        apply_event(session, event(session, "replace_plan", "graph_plan", plan), validate)
    plan["evidence"].pop()
    plan["confidence"] = float("nan")
    with pytest.raises(ValueError):
        apply_event(session, event(session, "replace_plan", "graph_plan", plan), validate)
    assert session["revision"] == 0


def test_ambiguous_typed_edge_added_in_same_plan_is_rejected(experiment):
    session = new_session(observe(experiment))
    plan = rule_plan(session["observation"])
    plan["edge_ops"] = [
        {"op": "ADD_EDGE", "source": "a", "target": "b", "edge_type": "other"},
        {"op": "REMOVE_EDGE", "source": "a", "target": "b"},
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        apply_event(session, event(session, "replace_plan", "graph_plan", plan), validate)


def test_train_split_rejects_overlap_before_training(experiment, tmp_path):
    split_path = tmp_path / "split.json"
    write(split_path, {"train": [experiment.name], "val": [experiment.name], "test": []})
    args = parse_args(["--data-root", str(experiment.parent), "--out-dir", str(tmp_path / "train"),
                       "--limit", "1", "--train-split", str(split_path)])
    with pytest.raises(ValueError, match="overlap"):
        run(args)
    assert not (tmp_path / "train" / "model.pt").exists()


def test_trace_exclusion_changes_features_and_restore_recovers_them(experiment):
    original = new_session(observe(experiment))
    ref = next(e["id"] for e in original["observation"]["evidence"]
               if e["kind"] == "trace_service" and e["data"]["service"] == "a")
    baseline, before = replay(original)
    changed = apply_event(original, event(original, "exclude_evidence", ref), validate)
    result, after = replay(changed)
    assert not torch.equal(before["edge_attr"], after["edge_attr"])
    assert ref not in canonical(result["evidence_pack"])
    restored = apply_event(changed, event(changed, "restore_evidence", ref, event_id="restore"), validate)
    assert replay(restored)[0]["graph"] == baseline["graph"]


def test_observation_does_not_read_raw_trace_as_missing_summary_fallback(experiment):
    (experiment / "traces" / "trace_summary.csv").unlink()
    (experiment / "traces" / "traces_raw.json").write_text("INVALID RAW DATA MUST NOT BE READ")
    view = observe(experiment)
    assert view["modalities"]["traces"]["status"] == "missing_summary"
    assert all(e["kind"] == "topology" for e in view["evidence"])


@pytest.mark.parametrize("model_name", ["weighted_graphsage", "edge_aware_gat"])
def test_missing_trace_is_distinct_from_observed_zero_in_model(experiment, model_name):
    session = new_session(observe(experiment))
    ref = next(e["id"] for e in session["observation"]["evidence"]
               if e["kind"] == "trace_service" and e["data"]["service"] == "b")
    set_seed(42)
    model = create_model(model_name, len(NODE_FEATURE_NAMES), len(EDGE_FEATURE_NAMES))
    baseline, healthy = replay(session, model, model_name)
    excluded = apply_event(session, event(session, "exclude_evidence", ref), validate)
    result, missing = replay(excluded, model, model_name)
    i = healthy["node_ids"].index("b")
    anomaly = NODE_FEATURE_NAMES.index("trace_anomaly")
    mask = NODE_FEATURE_NAMES.index("trace_observed")
    assert healthy["x"][i, anomaly] == missing["x"][i, anomaly] == 0
    assert healthy["x"][i, mask] == 1 and missing["x"][i, mask] == 0
    edge = next(e for e in result["graph"]["edges"] if e["source"] == "a" and e["target"] == "b")
    assert edge["attrs"]["anomaly_weight"] is None
    assert edge["attrs"]["anomaly_observed"] == 0
    assert edge["weight"] == round((0.30 + 0.25) / 0.65, 8)
    assert torch.isfinite(missing["x"]).all() and torch.isfinite(missing["edge_attr"]).all()
    assert baseline["prediction"]["scores"] != result["prediction"]["scores"]
    restored = apply_event(excluded, event(excluded, "restore_evidence", ref, event_id="restore"), validate)
    assert replay(restored, model, model_name)[0]["prediction"] == baseline["prediction"]


@pytest.mark.parametrize("condition", ["missing", "empty", "outside_window", "unknown_status"])
def test_no_usable_status_never_means_observed_healthy(experiment, condition):
    path = experiment / "traces" / "trace_summary.csv"
    if condition == "missing":
        path.unlink()
    elif condition == "empty":
        path.write_text(path.read_text().splitlines()[0] + "\n")
    elif condition == "unknown_status":
        lines = path.read_text().splitlines()
        path.write_text(lines[0] + "\n" + "\n".join(line.rsplit(",", 1)[0] + "," for line in lines[1:]) + "\n")
    kwargs = {"start_at": "2026-01-01T00:00:30Z"} if condition == "outside_window" else {}
    result, item = replay(new_session(observe(experiment, **kwargs)))
    assert not item["x"][:, NODE_FEATURE_NAMES.index("trace_observed")].any()
    assert all(e["attrs"]["anomaly_weight"] is None for e in result["graph"]["edges"])
    assert torch.isfinite(item["edge_attr"]).all()


def test_truncated_trace_retains_observed_rate_and_partial_mask(experiment):
    result, item = replay(new_session(observe(experiment, max_rows=1)))
    i = item["node_ids"].index("a")
    assert item["x"][i, NODE_FEATURE_NAMES.index("trace_observed")] == 1
    assert item["x"][i, NODE_FEATURE_NAMES.index("trace_anomaly")] == 1
    assert item["x"][i, NODE_FEATURE_NAMES.index("trace_partial")] == 1
    assert item["edge_attr"].shape[1] == len(EDGE_FEATURE_NAMES)
    assert item["metadata"]["edge_feature_names"][-1] == "final_weight"


def test_missingness_protocol_rejects_old_checkpoint(experiment, tmp_path):
    from dyrca_rca.dataset.materialize import NODE_FEATURE_NAMES as OLD_NODES, EDGE_FEATURE_NAMES as OLD_EDGES
    model = create_model("weighted_graphsage", len(OLD_NODES), len(OLD_EDGES))
    path = tmp_path / "old.pt"
    save_checkpoint(model, path, "weighted_graphsage", "interactive-topology-trace-v1")
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_checkpoint(path, PROTOCOL)


def test_unknown_status_does_not_dilute_error_rate(experiment):
    path = experiment / "traces" / "trace_summary.csv"
    with path.open("a") as stream:
        stream.write("t,extra,,a,1767225610000000000,1000000,\n")
    result, item = replay(new_session(observe(experiment)))
    node = next(n["attrs"] for n in result["graph"]["nodes"] if n["id"] == "a")
    assert node["trace_anomaly"] == 1.0
    assert node["trace_status_coverage"] == 0.5
    assert node["trace_partial"] == 1.0


def test_feature_schema_keeps_legacy_dimensions_and_empty_edges():
    from dyrca_rca.dataset.materialize import graph_to_model_tensors, NODE_FEATURE_NAMES as OLD_NODES
    graph = {"attrs": {}, "nodes": [{"id": "a", "attrs": {"type": "service"}}], "edges": []}
    legacy = graph_to_model_tensors(graph, {})
    assert legacy["x"].shape == (1, len(OLD_NODES))
    assert legacy["edge_attr"].shape == (0, 5)
    graph["attrs"]["feature_protocol"] = PROTOCOL
    current = graph_to_model_tensors(graph, {})
    assert current["x"].shape == (1, len(NODE_FEATURE_NAMES))
    assert current["edge_attr"].shape == (0, len(EDGE_FEATURE_NAMES))


def test_local_topology_patch_preserves_other_rows_and_is_reversible(experiment):
    original = new_session(observe(experiment))
    fact = next(e for e in original["observation"]["evidence"]
                if e["kind"] == "topology" and e["data"]["phase"] == "during")
    payload = {"collection": "services", "entity": "a", "field": "endpoint_ips",
               "expected": [], "value": ["1"]}
    proposed = propose_event(original, event(original, "patch_topology", fact["id"], payload))
    assert replay(proposed)[0] == replay(original)[0]
    preview, _ = replay(proposed, mode="preview", feedback_id="event-1")
    accepted = commit_decision(proposed, event(proposed, "accept_feedback", "event-1", event_id="accept"), validate)
    assert replay(accepted)[0]["graph"] == preview["graph"]
    changed = next(e for e in effective_state(accepted)[0]["evidence"] if e["id"] == fact["id"])
    assert changed["data"]["services"][1] == fact["data"]["services"][1]
    assert changed["correction_ids"] == ["event-1"]
    assert not any(n["id"] == "blackhole:a" for n in preview["graph"]["nodes"])
    assert fact["data"]["services"][0]["endpoint_ips"] == []
    undo = commit_decision(accepted, event(accepted, "retract_feedback", "event-1", event_id="undo"), validate)
    assert replay(undo)[0]["graph"] == replay(original)[0]["graph"]
    conflicting = propose_event(accepted, event(accepted, "patch_topology", fact["id"], payload, event_id="conflict"))
    failed = commit_decision(conflicting, event(conflicting, "accept_feedback", "conflict", event_id="fail"), validate)
    assert failed["outcomes"]["fail"]["execution_status"] == "failed"
    assert failed["state_revision"] == accepted["state_revision"]


def test_local_trace_correction_changes_only_target_aggregate(experiment):
    session = new_session(observe(experiment))
    fact = next(e for e in session["observation"]["evidence"] if e["kind"] == "trace_service" and e["data"]["service"] == "a")
    counts = {k: fact["data"][k] for k in ("span_count", "status_count", "error_count")}
    payload = {"expected": counts, "value": dict(counts, error_count=0)}
    updated = apply_event(session, event(session, "correct_trace", fact["id"], payload), validate)
    before = {e["id"]: e for e in effective_state(session)[0]["evidence"]}
    after = {e["id"]: e for e in effective_state(updated)[0]["evidence"]}
    assert all(after[k] == v for k, v in before.items() if k != fact["id"])
    assert not torch.equal(replay(session)[1]["x"], replay(updated)[1]["x"])
    payload["value"]["error_count"] = 100
    with pytest.raises(ValueError, match="trace counts"):
        propose_event(session, event(session, "correct_trace", fact["id"], payload))


def test_local_plan_remove_add_replace_and_conflict(experiment):
    session = new_session(observe(experiment))
    baseline, _ = replay(session)
    entry = baseline["plan_operations"][0]
    remove = {"section": entry["section"], "action": "remove", "operation_id": entry["operation_id"], "value": {}}
    changed = apply_event(session, event(session, "edit_plan", "graph_plan", remove), validate)
    assert not replay(changed)[0]["plan"]["node_ops"]
    ref = baseline["plan"]["evidence"][0]
    weight = {"op": "REWEIGHT_EDGE", "source": "a", "target": "b", "weight": 0.2,
              "attrs": {"route_weight": 0.2}, "evidence": [ref]}
    add = {"section": "weight_ops", "action": "add", "operation_id": "", "value": weight}
    changed = apply_event(changed, event(changed, "edit_plan", "graph_plan", add, event_id="weight"), validate)
    output, _ = replay(changed)
    entry = next(e for e in output["plan_operations"] if e["section"] == "weight_ops")
    replace = {"section": "weight_ops", "action": "replace", "operation_id": entry["operation_id"],
               "value": dict(weight, weight=0.4, attrs={"route_weight": 0.4})}
    changed = apply_event(changed, event(changed, "edit_plan", "graph_plan", replace, event_id="replace"), validate)
    assert replay(changed)[0]["plan"]["weight_ops"][0]["weight"] == 0.4
    # Retracting an operation still referenced by a later correction fails atomically.
    failed = commit_decision(changed, event(changed, "retract_feedback", "weight", event_id="undo"), validate)
    assert failed["outcomes"]["undo"]["execution_status"] == "failed"
    assert replay(failed)[0] == replay(changed)[0]


def test_local_patch_rejects_arbitrary_fields(experiment):
    session = new_session(observe(experiment))
    fact = next(e for e in session["observation"]["evidence"] if e["kind"] == "topology")
    payload = {"collection": "services", "entity": "a", "field": "root_cause",
               "expected": None, "value": "a"}
    with pytest.raises(ValueError, match="unsupported"):
        propose_event(session, event(session, "patch_topology", fact["id"], payload))
