"""Temporal evidence isolation, masking, replay and split integrity."""
import copy
import json

import pytest
import torch

from test_interaction import experiment, tmp_path, observe, event, validate
from dyrca_rca.interaction.temporal import temporal_features, FEATURES, PROTOCOL
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session, apply_event
from dyrca_rca.dataset.materialize import feature_names
from dyrca_rca.experiments.optimize_harness import grouped_split
from dyrca_rca.experiments.inference import load_checkpoint, save_checkpoint
from dyrca_rca.experiments.train import create_model


def test_new_empty_endpoint_differs_from_stable_empty(experiment):
    view = observe(experiment)
    values, refs = temporal_features(view)
    assert values["a"]["endpoint_became_empty"] == 1
    assert values["a"]["endpoint_pair_observed"] == 1
    assert len(refs) == 2
    pre = next(e for e in view["evidence"] if e["kind"] == "topology" and e["data"]["phase"] == "pre")
    pre["data"]["services"][0]["endpoint_ips"] = []
    assert temporal_features(view)[0]["a"]["endpoint_became_empty"] == 0


def test_missing_pods_and_unknown_versions_are_not_removals(experiment):
    path = experiment / "topology" / "during.json"
    data = json.loads(path.read_text())
    data.pop("pods")
    path.write_text(json.dumps(data))
    values, _ = temporal_features(observe(experiment))
    assert values["a"]["pod_pair_observed"] == 0
    assert values["a"]["pod_removed"] == 0
    assert values["a"]["version_removed"] == 0
    data["pods"] = [{"name": "a-pod", "service": "a", "version": "unknown"}]
    path.write_text(json.dumps(data))
    values, _ = temporal_features(observe(experiment))
    assert values["a"]["pod_pair_observed"] == 1
    assert values["a"]["version_pair_observed"] == 0
    assert values["a"]["version_removed"] == 0


def test_effective_temporal_inputs_retract_and_ignore_labels_future(experiment):
    session = new_session(observe(experiment))
    result, before = replay(session, feature_protocol=PROTOCOL)
    view = copy.deepcopy(session["observation"])
    view["ground_truth"] = {"root_cause_service": "SECRET"}
    future = copy.deepcopy(next(e for e in view["evidence"] if e["kind"] == "topology"))
    future["id"] = "future"
    future["data"]["timestamp"] = "2030-01-01T00:00:00Z"
    future["data"]["services"] = []
    view["evidence"].append(future)
    assert temporal_features(view) == temporal_features(session["observation"])
    during = next(e["id"] for e in view["evidence"] if e["kind"] == "topology" and e["data"]["phase"] == "during")
    excluded = apply_event(session, event(session, "exclude_evidence", during), validate)
    after = replay(excluded, feature_protocol=PROTOCOL)[1]
    index = feature_names(PROTOCOL)[0].index("topology_pair_observed")
    assert not after["x"][:, index].any()
    assert before["x"][:, index].any()
    undo = apply_event(excluded, event(excluded, "retract_event", "event-1", event_id="undo"), validate)
    assert torch.equal(replay(undo, feature_protocol=PROTOCOL)[1]["x"], before["x"])
    masked = replay(session, feature_protocol=PROTOCOL, mask_temporal=True)[1]
    assert not masked["x"][:, -len(FEATURES):].any()
    assert torch.equal(masked["x"][:, :-len(FEATURES)], before["x"][:, :-len(FEATURES)])


def test_protocol_auto_checkpoint_and_evidence_requests(experiment, tmp_path):
    nodes, edges = feature_names(PROTOCOL)
    model = create_model("weighted_graphsage", len(nodes), len(edges))
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, "weighted_graphsage", PROTOCOL)
    loaded, name = load_checkpoint(path, "auto")
    result, item = replay(new_session(observe(experiment, max_rows=1)), loaded, name)
    assert result["graph"]["attrs"]["feature_protocol"] == PROTOCOL
    assert result["harness"]["status"] == "needs_evidence"
    assert result["harness"]["margin_is_probability"] is False
    assert torch.isfinite(item["x"]).all()
    with pytest.raises(ValueError, match="protocol mismatch"):
        replay(new_session(observe(experiment)), loaded, name,
               feature_protocol="interactive-topology-trace-v2-missingness")


def test_grouped_split_is_disjoint_and_reproducible():
    names = ["%s-service%s-config-run%s" % (task, group, run)
             for task in ("t2", "t3", "t4") for group in range(10) for run in range(3)]
    regression = names[::5]
    split = grouped_split(names, regression, 42)
    assert split == grouped_split(list(reversed(names)), regression, 42)
    groups = {part: {n.rsplit("-run", 1)[0] for n in split[part]} for part in ("train", "val", "test")}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert not set(split["train"] + split["val"] + split["test"]) & set(regression)


def test_grouped_split_supports_t1_without_changing_task_membership():
    names = ["%s-service-config%d-run%d" % (task, group, run)
             for task in ("t1", "t2", "t3", "t4")
             for group in range(10) for run in range(2)]
    split = grouped_split(names, [], 42, tasks=("t2", "t3", "t4", "t1"))
    assert set(n[:2] for n in split["train"]) == {"t1", "t2", "t3", "t4"}
    assert set(n[:2] for n in split["val"]) == {"t1", "t2", "t3", "t4"}
    assert set(n[:2] for n in split["test"]) == {"t1", "t2", "t3", "t4"}
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(names)
