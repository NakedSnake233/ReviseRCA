import json
from pathlib import Path

from runner.scheduler import BatchConfig, BatchScheduler, ExperimentTask


def _complete_case(root: Path) -> Path:
    case = root / "case-1"
    for rel in (
        "topology/pre.json", "topology/during.json", "topology/post.json",
        "metrics/request_rate.csv", "traces/traces_raw.json", "logs/all.jsonl",
        "traces/trace_summary.csv",
    ):
        path = case / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".csv"):
            path.write_text("timestamp,value\n1,2\n")
        elif rel.endswith("traces_raw.json"):
            path.write_text(json.dumps({"data": [{"traceID": "x"}]}))
        else:
            path.write_text("{}\n")
    (case / "metadata.json").write_text("{}\n")
    (case / "ground_truth.json").write_text("{}\n")
    return case


def test_resume_reuses_only_complete_success(tmpdir):
    tmp_path = Path(str(tmpdir))
    complete = _complete_case(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"experiments": [
        {"index": 1, "status": "success", "output_dir": str(complete),
         "experiment_id": "case-1", "attempts": 1},
        {"index": 2, "status": "success", "output_dir": str(tmp_path / "missing")},
    ]}))
    cfg = BatchConfig(project_dir=str(tmp_path), summary_file="summary.json")
    scheduler = BatchScheduler(cfg, resume_summary=str(summary))
    scheduler.tasks = [ExperimentTask(index=1), ExperimentTask(index=2)]
    scheduler._apply_resume()
    assert scheduler.tasks[0].status == "success"
    assert scheduler.tasks[1].status == "pending"
    assert scheduler.resumed_from == str(summary)


def test_scheduler_rejects_empty_telemetry(tmpdir):
    root = Path(str(tmpdir))
    case = root / "case"
    (case / "metrics").mkdir(parents=True)
    (case / "traces").mkdir()
    (case / "logs").mkdir()
    (case / "metrics" / "request_rate.csv").write_text("timestamp,value\n")
    (case / "traces" / "trace_summary.csv").write_text("timestamp\n")
    (case / "traces" / "traces_raw.json").write_text('{"data": []}')
    (case / "logs" / "all.jsonl").write_text("{}\n")
    assert not BatchScheduler._telemetry_complete(str(case))


def test_strict_health_requires_all_containers_ready():
    assert BatchScheduler._pod_ready("a-123 2/2 Running 0 1m")
    assert not BatchScheduler._pod_ready("a-123 1/2 Running 0 1m")
    assert not BatchScheduler._pod_ready("a-123 0/1 Unknown 0 1m")
    assert not BatchScheduler._pod_ready("a-123 malformed Running 0 1m")


def test_strict_health_refuses_fault_injection(monkeypatch):
    scheduler = BatchScheduler(BatchConfig(require_healthy=True, max_retries=0))
    task = ExperimentTask(index=1, script="faults/T1_hpa_scale.sh")
    monkeypatch.setattr(scheduler, "_wait_healthy", lambda namespace: False)
    def unexpected_fault(_task):
        raise AssertionError("fault script must not run after failed preflight")
    monkeypatch.setattr(scheduler, "_run_fault_script", unexpected_fault)
    result = scheduler._execute_task(task)
    assert result.status == "pending"
    assert "no fault injected" in result.error


def test_resume_and_telemetry_check_never_load_raw_traces(tmp_path, monkeypatch):
    case = _complete_case(tmp_path)
    original = Path.read_text
    def bounded_read(path, *args, **kwargs):
        if path.name == "traces_raw.json":
            raise AssertionError("Raw traces must not be loaded for a completeness check")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", bounded_read)
    assert BatchScheduler._resume_record_is_complete({"status": "success", "output_dir": str(case)})
    assert BatchScheduler._telemetry_complete(str(case))
    (case / "traces" / "trace_summary.csv").write_text("timestamp,value\n")
    assert not BatchScheduler._resume_record_is_complete({"status": "success", "output_dir": str(case)})
    assert not BatchScheduler._telemetry_complete(str(case))


def test_quarantined_export_is_recollected_despite_complete_evidence(tmp_path):
    case = _complete_case(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"experiments": [{"index": 1, "status": "success", "output_dir": str(case)}]}))
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"excluded_output_dirs": [str(case)]}))
    scheduler = BatchScheduler(BatchConfig(project_dir=str(tmp_path), resume_exclusions_file=str(exclusions)),
                               resume_summary=str(summary))
    scheduler.tasks = [ExperimentTask(index=1)]
    scheduler._apply_resume()
    assert scheduler.tasks[0].status == "pending"
    assert (case / "metadata.json").exists()
