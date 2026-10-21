import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dyrca_rca.baselines.evidence import CandidateCase, _bounded_split
from dyrca_rca.baselines.methods import (
    BAROBaseline,
    CIRCABaseline,
    CloudRangerBaseline,
    EadroAdapterModel,
    HolisticRCAAdapterModel,
    NSigmaBaseline,
    ChangeOnlyBaseline,
    NodeOnlyAdapterModel,
    ranked_prediction,
)
from dyrca_rca.experiments.run_dynamic_baselines import _score_control
from dyrca_rca.experiments import run_dynamic_baselines as runner


def _case(name, shift=0.0):
    x = torch.zeros((3, 43), dtype=torch.float32)
    x[0, 20] = 1.0 + shift
    x[1, 27] = shift
    x[2, 37] = 0.5 * shift
    return CandidateCase(
        name=name,
        incident_id="incident-" + name,
        task="T2",
        node_names=["frontend", "checkoutservice", "paymentservice"],
        x=x,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_weight=torch.ones(4, dtype=torch.float32),
        directed_edges=[("frontend", "checkoutservice", 1.0),
                        ("checkoutservice", "paymentservice", 1.0)],
        source_cache_hash="0" * 64,
    )


def test_bounded_split_retains_all_partitions():
    split = {"train": list(range(125)), "val": list(range(30)),
             "test": list(range(45)), "regression": list(range(100))}
    selected = _bounded_split(split, 12)
    assert sum(map(len, selected.values())) == 12
    assert all(selected[part] for part in split)


def test_classical_adapters_return_complete_deterministic_rankings():
    train = [_case("train-%02d" % index, shift=float(index % 4)) for index in range(12)]
    test = _case("test", shift=5.0)
    methods = [NSigmaBaseline(), BAROBaseline(), CIRCABaseline(), CloudRangerBaseline(seed=7)]
    for method in methods:
        method.fit(train)
        first = ranked_prediction(test, method.predict(test))["prediction"]
        second = ranked_prediction(test, method.predict(test))["prediction"]
        assert first == second
        assert set(first["ranked_node_ids"]) == set(test.node_names)
        assert all(np.isfinite(value) for value in first["scores"].values())


def test_neural_adapter_forward_shapes_and_gradients():
    case = _case("neural", shift=2.0)
    for model in (EadroAdapterModel(32), HolisticRCAAdapterModel(32)):
        scores = model(case.x, case.edge_index, case.edge_weight)
        assert scores.shape == (3,)
        assert bool(torch.isfinite(scores).all())
        scores.sum().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_change_only_and_node_only_adapters_are_finite_and_complete():
    train = [_case("train-%02d" % index, shift=float(index % 4)) for index in range(12)]
    change = ChangeOnlyBaseline().fit(train)
    prediction = ranked_prediction(train[0], change.predict(train[0]))["prediction"]
    assert set(prediction["ranked_node_ids"]) == set(train[0].node_names)
    model = NodeOnlyAdapterModel(43, 16)
    scores = model(train[0].x, train[0].edge_index, train[0].edge_weight)
    assert scores.shape == (3,)
    assert bool(torch.isfinite(scores).all())


@pytest.mark.parametrize("truth,hit1,hit3,mrr", [
    ("frontend", 1.0, 1.0, 1.0),
    ("paymentservice", 0.0, 1.0, 1.0 / 3),
    ("outside_inventory", 0.0, 0.0, 0.0),
])
def test_fault_control_scoring_handles_metric_label_status(tmp_path, truth, hit1, hit3, mrr):
    (tmp_path / "metadata.json").write_text(json.dumps({"is_benign": False}))
    (tmp_path / "ground_truth.json").write_text(json.dumps({"root_cause_service": truth}))
    row = {"experiment": "fault-control", "split": "control", "method": "NSigma-Dyn",
           "prediction": {"ranked_node_ids": ["frontend", "checkoutservice", "paymentservice"]}}
    frozen = json.dumps(row, sort_keys=True)
    result = _score_control(row, tmp_path)
    assert (result["hit1"], result["hit3"], result["mrr"]) == (hit1, hit3, mrr)
    assert result["label_status"] == "verified_control_target"
    assert json.dumps(row, sort_keys=True) == frozen


def test_benign_control_has_no_fault_ranking_or_required_target(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"is_benign": True}))
    row = {"experiment": "healthy-control", "split": "control", "method": "NSigma-Dyn",
           "prediction": {"ranked_node_ids": ["frontend"]}}
    result = _score_control(row, tmp_path)
    assert result["label_status"] == "benign_control"
    assert all(result[key] is None for key in ("hit1", "hit3", "mrr"))


def test_runner_freezes_predictions_and_scores_controls_without_regression(tmp_path, monkeypatch):
    raw, release, out = (tmp_path / name for name in ("raw", "release", "out"))
    raw.mkdir()
    release.mkdir()
    train, test = _case("train"), _case("test", shift=2.0)
    fault, benign = _case("fault-control", shift=3.0), _case("benign-control")
    fault.task, benign.task = "C2", "C6"
    for case, healthy in ((test, False), (fault, False), (benign, True)):
        case_dir = raw / case.name
        case_dir.mkdir()
        (case_dir / "metadata.json").write_text(json.dumps({"is_benign": healthy}))
        if not healthy:
            (case_dir / "ground_truth.json").write_text(json.dumps({"root_cause_service": "paymentservice"}))
    parts = {"train": [train], "val": [], "test": [test], "regression": []}
    cases = {case.name: case for case in (train, test)}
    evidence = SimpleNamespace(
        release_root=release, data_root=raw, system="boutique", dataset_id="bounded-fixture",
        split={part: [case.name for case in rows] for part, rows in parts.items()},
        max_rows=10, feature_names=[], edge_feature_names=[], split_file_hash="split-hash",
        report_file_hash="report-hash", cache_hashes={name: case.source_cache_hash for name, case in cases.items()},
        cases=cases, part=lambda part: parts[part],
    )
    ours = ranked_prediction(test, {"frontend": 0.0, "checkoutservice": 1.0, "paymentservice": 2.0})
    (release / "predictions.jsonl").write_text(json.dumps(dict(ours, group="temporal_v3", split="test")) + "\n")
    monkeypatch.setattr(runner, "load_release_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(runner, "_verify_upstream_checkouts", lambda: {"fixture": "bounded"})
    monkeypatch.setattr(runner, "_load_controls", lambda *a: [fault, benign])
    monkeypatch.setattr(runner, "_released_dyrca_controls", lambda *a: [
        dict(ranked_prediction(case, {"frontend": 0.0, "checkoutservice": 1.0, "paymentservice": 2.0}),
             split="control", method="DY-RCA", control_scenario=case.task)
        for case in (fault, benign)
    ])
    original_labels = runner._labels
    opened_parts = []
    def labels_after_freeze(root, names):
        assert (out / "predictions.jsonl").is_file()
        opened_parts.append(list(names))
        return original_labels(root, names)
    monkeypatch.setattr(runner, "_labels", labels_after_freeze)
    report = runner.run(SimpleNamespace(
        out_dir=str(out), data_root=str(raw), release_root=str(release), baseline=["nsigma"],
        include_validation=False, seed=42, planner="rule", limit=None, system="boutique",
        dataset_id="bounded-fixture", controls_root=str(raw), controls_audit=None, controls_limit=None,
    ))
    assert opened_parts == [["test"], []]
    assert (out / "report.json").is_file()
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert {row["label_status"] for row in rows if row["experiment"] == fault.name} == {"verified_control_target"}
    assert all(row["mrr"] is None for row in rows if row["experiment"] == benign.name)
    assert all(row["test_n"] == 1 and row["regression_n"] == 0 for row in report["main_comparison"])
    absent = [row for row in report["paired_comparisons"] if row["split"] == "regression"]
    assert len(absent) == 1 and absent[0]["n"] == 0
    assert absent[0]["hit1_difference"] is None and absent[0]["mrr_difference"] is None
    assert runner.file_hash(out / "predictions.jsonl") == report["prediction_hash"]
    text = (out / "RUN_REPORT.md").read_text()
    assert "No independent regression partition was retained" in text
    assert "| N/A | N/A |" in text
