import json
from pathlib import Path

from dyrca_rca.experiments.run_tosem_backfill import _audit_case, _build_summary, run


def _case(root: Path) -> Path:
    case = root / "t3-demo"
    for rel in (
        "topology/pre.json", "topology/during.json", "topology/post.json",
        "metrics/request_rate.csv", "traces/trace_summary.csv",
        "traces/traces_raw.json", "logs/all.jsonl", "topology_diff.json",
        "verification_report.json",
    ):
        path = case / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".csv"):
            path.write_text("timestamp,value\n1,2\n")
        elif rel.endswith("traces_raw.json"):
            path.write_text('{"data":[{"traceID":"x"}]}')
        elif rel.endswith("verification_report.json"):
            path.write_text('{"verification_result":"CHANGED","confidence":1,"recovery_verified":true}')
        else:
            path.write_text("{}")
    base = {
        "pods": [{"name": "a", "service": "svc", "version": "v1"}],
        "services": [{"name": "svc", "endpoint_ips": ["1"]}],
    }
    (case / "topology" / "pre.json").write_text(json.dumps(base))
    during = {"pods": [{"name": "a", "service": "svc", "version": "v1"}],
              "services": [{"name": "svc", "endpoint_ips": ["2"]}]}
    (case / "topology" / "during.json").write_text(json.dumps(during))
    (case / "topology" / "post.json").write_text(json.dumps(base))
    (case / "metadata.json").write_text(json.dumps({
        "scenario_id": "T3", "target_service": "svc", "fault_type": "redirect",
        "environment": {"namespace": "boutique"},
    }))
    (case / "ground_truth.json").write_text(json.dumps({"root_cause_service": "svc"}))
    return case


def test_audit_requires_scheduler_success_and_counts_active_endpoint(tmpdir):
    root = Path(str(tmpdir))
    row = _audit_case(_case(root), {"status": "success", "verify_exit_code": 0})
    assert row["verified"] is True
    assert row["endpoint_active"] == 1
    assert row["missing_evidence"] == []


def test_run_keeps_downstream_table_values_null(tmpdir):
    tmp_path = Path(str(tmpdir))
    root = tmp_path / "raw"
    root.mkdir()
    _case(root)
    (root / "batch_summary.json").write_text(json.dumps({"experiments": [{
        "status": "success", "verify_exit_code": 0,
        "output_dir": str(root / "t3-demo"),
    }]}))
    out = tmp_path / "out"
    status = run(type("Args", (), {"data_root": str(root), "out_dir": str(out),
                                    "limit": None, "seed": 42, "planner": "rule"})())
    template = json.loads((out / "results_template.json").read_text())
    assert template["fields"]["ob.intended_n"]["value"] == 1
    assert template["tab:new_ranking"]["value"] is None
    assert status["tables"]["tab:new_ranking"]["status"] == "pending_downstream_artifact"


def test_copy_namespace_is_provenance_not_a_separate_application(tmp_path):
    case = _case(tmp_path)
    metadata_path = case / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["environment"]["namespace"] = "boutique-c3"
    metadata_path.write_text(json.dumps(metadata))
    before = metadata_path.read_bytes()
    row = _audit_case(case, {"status": "success", "verify_exit_code": 0})
    assert row["application"] == "boutique"
    assert row["source_namespace"] == "boutique-c3"
    assert row["trace_rows"] == 1
    assert metadata_path.read_bytes() == before
