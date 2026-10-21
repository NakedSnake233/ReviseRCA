import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dyrca_rca.data.index import build_index
from dyrca_rca.data.schema import INDEX_COLUMNS


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_index_from_fake_experiment(tmpdir):
    data_root = Path(str(tmpdir)) / "experiments"
    exp_dir = data_root / "t1-cartservice-run001"

    _write_json(exp_dir / "metadata.json", {
        "experiment_id": "t1-cartservice-run001",
        "scenario_id": "T1",
        "scenario_name": "hpa_scale",
        "target_service": "cartservice",
        "fault_type": "cpu_stress_hpa_scale",
        "topology_change_type": "instance_count_change",
        "fault_params": {"cpu_load": 60, "cpu_workers": 2},
    })
    _write_json(exp_dir / "ground_truth.json", {
        "experiment_id": "t1-cartservice-run001",
        "root_cause_service": "cartservice",
        "fault_type": "cpu_stress_induced_hpa_scale",
        "topology_change_type": "instance_count_change",
        "labels": {"scenario": "T1"},
    })
    _write_json(exp_dir / "topology_diff.json", {
        "schema": "topology_diff_v1",
        "annotation": {"verification_result": "CHANGED"},
    })
    _write_json(exp_dir / "verification_report.json", {
        "verification_result": "CHANGED",
        "topology_change_type": "node_scale",
    })
    for phase in ("pre", "during", "post"):
        _write_json(exp_dir / "topology" / f"{phase}.json", {
            "timestamp": "2026-01-01T00:00:00Z",
            "deployments": [],
            "pods": [],
            "services": [],
        })

    (exp_dir / "metrics").mkdir()
    (exp_dir / "metrics" / "cpu_usage.csv").write_text("timestamp,metric_name,labels_json,value\n")
    (exp_dir / "logs").mkdir()
    (exp_dir / "logs" / "all.jsonl").write_text("")
    (exp_dir / "traces").mkdir()
    (exp_dir / "traces" / "trace_summary.csv").write_text("trace_id\n")

    frame = build_index(data_root)

    assert list(frame.columns) == INDEX_COLUMNS
    assert len(frame) == 1
    row = frame.iloc[0].to_dict()
    assert row["experiment_id"] == "t1-cartservice-run001"
    assert row["scenario_id"] == "T1"
    assert row["scenario_name"] == "hpa_scale"
    assert row["target_service"] == "cartservice"
    assert row["params_json"] == '{"cpu_load":60,"cpu_workers":2}'
    assert row["root_cause_service"] == "cartservice"
    assert row["root_cause_type"] == "cpu_stress_induced_hpa_scale"
    assert bool(row["has_pre"]) is True
    assert bool(row["has_during"]) is True
    assert bool(row["has_post"]) is True
    assert bool(row["has_metrics"]) is True
    assert bool(row["has_logs"]) is True
    assert bool(row["has_traces"]) is True
    assert row["topology_change_type"] == "instance_count_change"
    assert row["verification_status"] == "CHANGED"
    assert row["data_path"] == str(exp_dir)
    assert row["error_message"] == ""


def test_bad_experiment_records_error_message(tmpdir):
    data_root = Path(str(tmpdir)) / "experiments"
    exp_dir = data_root / "bad-run"
    exp_dir.mkdir(parents=True)
    (exp_dir / "metadata.json").write_text("{bad json", encoding="utf-8")

    frame = build_index(data_root)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    row = frame.iloc[0].to_dict()
    assert row["experiment_id"] == "bad-run"
    assert "bad json in metadata.json" in row["error_message"]
    assert "missing required file: ground_truth.json" in row["error_message"]
