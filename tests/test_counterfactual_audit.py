import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_counterfactual_controls import audit


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _case(root: Path, name: str, *, benign: bool = False) -> Path:
    case = root / name
    _write_json(case / "metadata.json", {
        "target_service": "root-svc",
        "topology_target_service": "decoy-svc",
        "control_mode": "scale",
        "is_benign": benign,
    })
    _write_json(case / "ground_truth.json", {
        "root_cause_service": "__benign__" if benign else "root-svc",
    })
    _write_json(case / "topology_diff.json", {
        "comparisons": {"pre_vs_during": {"k8s_layer": {
            "pod_changes": {"count_changes": {"decoy-svc": {"before": 1, "after": 2}}},
            "endpoint_changes": {"changed_services": {"decoy-svc": {}}},
            "deployment_changes": {"changed_deployments": {}},
        }, "graph_layer": {"node_diff": {"added": [], "removed": [], "modified": []}}}}
    })
    _write_json(case / "verification_report.json", {"verification_result": "CHANGED"})
    for phase in ("pre", "during"):
        _write_json(case / "topology" / (phase + ".json"), {
            "services": [{"name": "decoy-svc", "endpoint_ips": ["10.0.0.1"]}],
        })
    (case / "traces").mkdir()
    (case / "metrics").mkdir()
    (case / "logs").mkdir()
    (case / "traces" / "traces_raw.json").write_text('{"data":[{"traceID":"t1"}]}')
    (case / "metrics" / "request_rate.csv").write_text("timestamp,value\n1,1\n")
    (case / "logs" / "all.jsonl").write_text('{"msg":"ok"}\n')
    return case


def test_audit_enforces_counterfactual_invariant(tmpdir):
    root = Path(str(tmpdir))
    normal = _case(root, "normal")
    benign = _case(root, "benign", benign=True)
    summary = root / "batch_summary.json"
    summary.write_text(json.dumps({"experiments": [
        {"status": "success", "output_dir": str(normal)},
        {"status": "success", "output_dir": str(benign)},
        {"status": "failed", "output_dir": str(root / "ignored")},
    ]}))

    report = audit(root, summary)

    assert report["population"] == 2
    assert report["normal_controls"] == 1
    assert report["benign_controls"] == 1
    assert report["target_in_changed_service_rate"] == 0.0
    assert report["decoy_in_changed_service_rate"] == 1.0
    assert report["verifier_changed_rate"] == 1.0
    assert report["decoy_endpoints_present_pre_during_rate"] == 1.0
    assert report["strict_operational_n"] == 2
    assert report["telemetry_complete_rate"] == 1.0
    assert report["min_trace_count"] == 1
