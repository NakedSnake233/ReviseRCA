import copy
import csv
import json

import numpy as np
import torch
import pytest

from dyrca_rca.interaction.behavior import FEATURES, augment, load_behavior, resolve_service
from dyrca_rca.baselines.methods import CloudRangerBaseline, EadroAdapterModel, HolisticRCAAdapterModel
from test_dynamic_baselines import _case


def test_identity_resolution_never_guesses_unknown_or_ambiguous_service():
    names = {"cart", "pay"}
    assert resolve_service("cart.ns.svc.cluster.local:80", names, {}) == "cart"
    assert resolve_service("cart-pod", names, {"cart-pod": "cart"}) == "cart"
    for name in ["unknown", "cart-shadow", "pay-request", None]:
        assert resolve_service(name, names, {}) is None


def test_behavior_cutoff_missing_status_and_root_metadata_invariance(tmp_path):
    (tmp_path / "topology").mkdir(); (tmp_path / "traces").mkdir(); (tmp_path / "metrics").mkdir()
    for phase, stamp in [("pre", 2), ("during", 4)]:
        (tmp_path / "topology" / (phase + ".json")).write_text(json.dumps({"timestamp": f"1970-01-01T00:00:0{stamp}Z", "services": [{"name": "cart"}], "pods": [{"name": "pod1", "service": "cart"}]}))
    with (tmp_path / "traces/trace_summary.csv").open("w") as f:
        w = csv.writer(f);w.writerow(["service", "start_time_unix_nano", "duration_nano", "status_code"])
        w.writerows([["cart", 1000000000, 100000000, ""], ["pod1", 2100000000, 1000000000, ""],
                     ["cart", 3000000000, 2000000000, "ERROR"]])
    with (tmp_path / "metrics/latency_p99.csv").open("w") as f:
        w = csv.writer(f);w.writerow(["timestamp", "labels_json", "value"])
        w.writerows([["1970-01-01T00:00:03Z", '{"destination_workload":"unknown"}', 9999],
                     ["1970-01-01T00:00:03Z", '{"pod":"pod1"}', 100]])
    first = load_behavior(tmp_path)
    (tmp_path / "metadata.json").write_text('{"target_service":"evil"}')
    (tmp_path / "ground_truth.json").write_text('{"root_cause_service":"evil"}')
    assert first == load_behavior(tmp_path)
    values = dict(zip(FEATURES, first["values"]["cart"]))
    assert values["duration_log_change"] > 0
    assert values["during_error_available"] == 0
    assert values["metric_during_available"] == 1 and values["metric_pre_available"] == 0
    assert first["coverage"]["trace_eligible"] == 2
    assert first["coverage"]["metric_unresolved"] == 1


def test_cloudranger_inference_is_invariant_to_experiment_name():
    train = [_case(f"train-{i}", float(i % 4)) for i in range(8)]
    method = CloudRangerBaseline(seed=42).fit(train)
    a = _case("t1-rootA", 3.); b = copy.deepcopy(a);b.name = "t4-rootB"
    assert method.predict(a) == method.predict(b)


def test_neural_adapters_consume_added_behavior_channels():
    case = _case("test", 2.)
    x = torch.cat([case.x, torch.ones(3, len(FEATURES))], dim=1).requires_grad_()
    for model in [EadroAdapterModel(32, len(FEATURES)), HolisticRCAAdapterModel(32, len(FEATURES))]:
        model.eval(); scores = model(x, case.edge_index, case.edge_weight)
        assert scores.shape == (3,) and torch.isfinite(scores).all()
        grad = torch.autograd.grad(scores.sum(), x)[0]
        assert torch.any(grad[:, 43:] != 0)


def test_separation_preserves_topology_and_recomputes_derived_weights():
    attrs = {"dependency_weight": 1., "anomaly_weight": None, "topology_confidence": 1., "change_weight": 1., "final_weight": 1.}
    item = {"x": torch.ones(2, 43), "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "edge_attr": torch.tensor([[1., 0., 1., 1., 0., 0., 1., 1.]]*2),
            "node_ids": ["a", "b"], "metadata": {"node_feature_names": list(range(43))},
            "graph": {"nodes": [{"id": n, "attrs": {"service": n}} for n in ["a", "b"]],
                      "edges": [{"source": "a", "target": "b", "attrs": attrs}]}}
    behavior = {"values": {"a": [1.]*len(FEATURES)}}
    out = augment(item, behavior, True)
    assert torch.equal(out["edge_index"], item["edge_index"])
    assert torch.equal(out["x"][:, 25:43], item["x"][:, 25:43])
    assert torch.all(out["x"][:, [15, 17]] == 0) and torch.all(out["edge_attr"][:, 3] == 0)
    assert torch.all(item["x"] == 1)  # original cache untouched
    assert out["x"].shape[1] == 43+len(FEATURES)
    assert torch.all(out["x"][1, 43:] == 0)  # missing behavior stays missing


def test_checkpoint_loader_validates_feature_schema_and_protocol(tmp_path):
    from dyrca_rca.baselines.evidence import file_hash
    from dyrca_rca.dataset.materialize import feature_names
    from dyrca_rca.experiments.train import create_model
    from dyrca_rca.interaction.behavior import PROTOCOL
    from dyrca_rca.interaction.behavior_inference import load_behavior_checkpoint
    from dyrca_rca.interaction.temporal import PROTOCOL as OLD_PROTOCOL
    from dyrca_rca.models.config import FIXED_GNN_CONFIG
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps({"protocol": PROTOCOL, "fixed_config": FIXED_GNN_CONFIG.__dict__}))
    names, _ = feature_names(OLD_PROTOCOL)
    model = create_model("weighted_graphsage", len(names)+len(FEATURES), 8)
    content = {"state_dict": model.state_dict(), "node_features": names+FEATURES,
               "protocol_sha256": file_hash(p), "group": "DY-RCA+behavior-separated"}
    target = tmp_path / "checkpoint.pt";torch.save(content, target)
    _, separated, _ = load_behavior_checkpoint(target)
    assert separated
    content["node_features"] = list(reversed(content["node_features"]))
    torch.save(content, target)
    with pytest.raises(ValueError, match="schema"):
        load_behavior_checkpoint(target)
