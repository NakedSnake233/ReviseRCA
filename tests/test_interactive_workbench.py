"""Task forms, durable decisions, explanations and frozen evaluation guards."""
import copy
import json

import pytest

from test_interaction import experiment, tmp_path, observe
from dyrca_rca.interaction.state import new_session, apply_event
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.tasks import task_feedback
from dyrca_rca.interaction.workbench import Workbench
from dyrca_rca.interaction.explain import compare_results
from dyrca_rca.experiments.evaluate_interactive import rank_metrics


def command(view, action, revision=0, **kwargs):
    return dict(incident_id=view["incident_id"], action=action, base_revision=revision,
                event_id=action, actor="operator", reason="snapshot review", **kwargs)


@pytest.mark.parametrize("task", ["T2", "T3"])
def test_task_preview_commit_resume_and_retract(experiment, tmp_path, task):
    view = observe(experiment)
    root = tmp_path / "workbench"
    wb = Workbench(root, [view])
    baseline = wb.snapshot(view["incident_id"])
    proposed = wb.command(command(view, "propose", task=task, service="a", values={"endpoint_ips": ["1"]}))
    assert proposed["current"] == baseline["current"]
    preview = wb.command(command(view, "preview", target_id="propose"))
    assert preview["comparison"]["nodes"]["removed"]
    accepted = wb.command(command(view, "accept_feedback", target_id="propose"))
    assert accepted["current"]["graph"] == preview["preview"]["graph"]
    assert accepted["history"][0]["comparison"] == preview["comparison"]
    resumed = Workbench(root, [view])
    assert resumed.snapshot(view["incident_id"]) == accepted
    with pytest.raises(ValueError, match="版本"):
        resumed.command(command(view, "retract_feedback", target_id="propose"))
    restored = resumed.command(command(view, "retract_feedback", revision=1, target_id="propose"))
    assert restored["current"]["graph"] == baseline["current"]["graph"]


def test_t4_checks_observed_versions_and_conserves_proportions(experiment):
    view = observe(experiment)
    for fact in view["evidence"]:
        if fact["kind"] == "topology":
            fact["data"]["pods"][0]["version"] = "v1"
            fact["data"]["pods"][1].update(service="a", version="v2")
    session = new_session(view)
    result = replay(session)[0]
    values = {"versions": ["v1", "v2"], "weights": [0.8, 0.2]}
    feedback = task_feedback(session, result, "T4", "a", values, "split", "operator", "observed versions")
    feedback.pop("event_kind")
    feedback["verification_status"] = "verified"
    changed = apply_event(session, feedback, lambda s: replay(s))
    graph = replay(changed)[0]["graph"]
    assert {"a:v1", "a:v2"} <= {n["id"] for n in graph["nodes"]}
    edges = [e for e in graph["edges"] if e["edge_type"] == "version_split"]
    assert sum(e["attrs"]["route_weight"] for e in edges) == 1
    assert compare_results(result, replay(changed)[0])["nodes"]["added"]
    for bad in ({"versions": ["v1", "invented"], "weights": [0.5, 0.5]},
                {"versions": ["v1", "v2"], "weights": [0.8, 0.8]}):
        with pytest.raises(ValueError):
            task_feedback(session, result, "T4", "a", bad, "bad", "operator", "bad")


def test_failed_decision_is_durable_and_preview_has_no_audit_effect(experiment, tmp_path):
    view = observe(experiment)
    wb = Workbench(tmp_path / "wb", [view])
    wb.command(command(view, "propose", task="T3", service="a", values={"endpoint_ips": ["1"]}))
    before = wb.snapshot(view["incident_id"])
    original_replay = wb.replay
    def fail(*args, **kwargs):
        raise ValueError("model execution failed")
    wb.replay = fail
    failed = wb.command(command(view, "accept_feedback", target_id="propose"))
    assert failed["current"] == before["current"]
    assert failed["session"]["outcomes"]["accept_feedback"]["execution_status"] == "failed"
    wb.replay = original_replay
    wb.command(command(view, "preview", target_id="propose"))
    assert wb.snapshot(view["incident_id"]) == failed


def test_evaluation_missing_labels_and_absent_candidates():
    prediction = {"ranked_node_ids": ["a", "b"]}
    assert rank_metrics(prediction, "b")["mrr"] == 0.5
    assert rank_metrics(prediction, "absent")["mrr"] == 0
    assert rank_metrics(prediction, None)["mrr"] is None


def test_evaluator_runs_frozen_and_rejects_training_overlap(experiment, tmp_path):
    from dyrca_rca.experiments.evaluate_interactive import evaluate
    from dyrca_rca.experiments.serve_interactive import parser
    from dyrca_rca.experiments.train import create_model
    from dyrca_rca.experiments.inference import save_checkpoint
    from dyrca_rca.dataset.materialize import INTERACTIVE_NODE_FEATURE_NAMES, INTERACTIVE_EDGE_FEATURE_NAMES
    from dyrca_rca.experiments.run_interactive import PROTOCOL
    from dyrca_rca.interaction.observation import digest
    checkpoint = tmp_path / "fixture.pt"
    split = {"train": ["train-fixture"], "val": ["val-fixture"], "test": [experiment.name]}
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"split": split}))
    model = create_model("weighted_graphsage", len(INTERACTIVE_NODE_FEATURE_NAMES), len(INTERACTIVE_EDGE_FEATURE_NAMES))
    save_checkpoint(model, checkpoint, "weighted_graphsage", PROTOCOL, {"split_hash": digest(split), "fixture_only": True})
    args = parser().parse_args(["--data-root", str(experiment.parent), "--out-dir", str(tmp_path / "evaluation"),
                                "--checkpoint", str(checkpoint), "--limit", "1"])
    args.training_manifest, args.workbench_dir, args.simulate_feedback = str(training), None, False
    report = evaluate(args)
    assert report["weights_unchanged"]
    assert {r["group"] for r in report["summary"]} == {"static_pre", "automatic_rule"}
    split["train"] = [experiment.name]
    training.write_text(json.dumps({"split": split}))
    save_checkpoint(model, checkpoint, "weighted_graphsage", PROTOCOL, {"split_hash": digest(split)})
    with pytest.raises(ValueError, match="overlaps"):
        evaluate(args)


def test_simulated_collection_count_resume_and_source_guard(experiment, tmp_path):
    from dyrca_rca.experiments.collect_simulated_feedback import collect, simulation_values
    from dyrca_rca.experiments.evaluate_interactive import evaluate
    from dyrca_rca.experiments.serve_interactive import parser
    from dyrca_rca.experiments.train import create_model
    from dyrca_rca.experiments.inference import save_checkpoint
    from dyrca_rca.dataset.materialize import INTERACTIVE_NODE_FEATURE_NAMES, INTERACTIVE_EDGE_FEATURE_NAMES
    from dyrca_rca.experiments.run_interactive import PROTOCOL
    from dyrca_rca.interaction.observation import digest
    model = create_model("weighted_graphsage", len(INTERACTIVE_NODE_FEATURE_NAMES), len(INTERACTIVE_EDGE_FEATURE_NAMES))
    checkpoint, training = tmp_path / "fixture.pt", tmp_path / "training.json"
    split = {"train": ["train-fixture"], "val": ["val-fixture"], "test": [experiment.name]}
    training.write_text(json.dumps({"split": split}))
    save_checkpoint(model, checkpoint, "weighted_graphsage", PROTOCOL, {"split_hash": digest(split), "fixture_only": True})
    out = tmp_path / "collection"
    args = parser().parse_args(["--data-root", str(experiment.parent), "--out-dir", str(out),
                               "--checkpoint", str(checkpoint), "--limit", "1"])
    args.training_manifest = str(training)
    first = collect(copy.deepcopy(args))
    feedback_bytes = (out / "feedback.jsonl").read_bytes()
    second = collect(copy.deepcopy(args))
    assert first == second
    assert (out / "feedback.jsonl").read_bytes() == feedback_bytes
    assert first["feedback_count"] == 1 and first["audit_event_count"] == 2
    assert first["collection_weights_unchanged"]
    assert {r["group"] for r in first["evaluation"]["summary"]} == {"static_pre", "automatic_rule", "simulated_feedback"}
    args.out_dir = str(tmp_path / "wrong-source")
    args.workbench_dir, args.feedback_source, args.simulate_feedback = str(out / "workbench"), "operator", False
    with pytest.raises(ValueError, match="simulated workbench"):
        evaluate(args)
    view = observe(experiment)
    expected = simulation_values(view, "T3")
    view["ground_truth"] = {"root_cause_service": "invented"}
    assert simulation_values(view, "T3") == expected
