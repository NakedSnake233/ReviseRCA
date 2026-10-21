import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dyrca_rca.experiments import auto_tosem_pipeline as module


def cohort(root, app, incomplete=False):
    expected = {i: {"scenario_id": "T3", "target": "svc", "params": {"variant": "empty"}, "repeat": 1}
                for i in range(1, 5)}
    for i in range(1, 5):
        ns = f"{app}-c{i}"
        directory = root / ns
        directory.mkdir(parents=True)
        record = dict(expected[i], index=i, experiment_id=f"t3-demo-{i}",
                      output_dir=str(directory / f"t3-demo-{i}"), status="success",
                      fault_exit_code=0, export_exit_code=0, verify_exit_code=0)
        (directory / "batch_summary.json").write_text(json.dumps({
            "settings": {"seed": 42, "namespace": ns, "shard_count": 4, "shard_index": i - 1},
            "pending": int(incomplete and i == 4), "failed": 0,
            "experiments": [] if incomplete and i == 4 else [record]}))
    return expected


def args(tmp_path):
    return SimpleNamespace(data_root=tmp_path / "data", out_dir=tmp_path / "jobs", seed=42,
                           planner="rule", system="all", limit=None, poll_seconds=30,
                           memory_cap_gib=8, admit_gib=32, stop_below_gib=24,
                           llm_config=tmp_path / "credential.json", paper_dir=tmp_path / "paper",
                           exclusions=tmp_path / "excluded.json", once=True)


def test_boutique_completion_does_not_wait_for_train_ticket(tmp_path):
    raw = tmp_path / "raw"
    expected = cohort(raw, "boutique")
    cohort(raw, "trainticket", incomplete=True)
    assert module.collection_readiness(raw, "boutique", expected, 42)["ready"]
    tt = module.collection_readiness(raw, "trainticket", expected, 42)
    assert not tt["ready"] and tt["complete"] == 3


def test_watcher_starts_only_the_completed_application(tmp_path, monkeypatch):
    config = args(tmp_path)
    expected = cohort(config.data_root, "boutique")
    cohort(config.data_root, "trainticket", incomplete=True)
    monkeypatch.setattr(module, "_plan", lambda *_: expected)
    watcher = module.AutoPipeline(config)
    invoked = []
    def run_application(app):
        invoked.append(app)
        watcher.status["completed_applications"].append(app)
    monkeypatch.setattr(watcher, "run_application", run_application)
    assert watcher.run() == 0
    assert invoked == ["boutique"]


def test_excluded_or_failed_export_prevents_formal_start(tmp_path):
    raw = tmp_path / "raw"
    expected = cohort(raw, "boutique")
    excluded = str(raw / "boutique-c3/t3-demo-3")
    assert not module.collection_readiness(raw, "boutique", expected, 42, {excluded})["ready"]
    path = raw / "boutique-c2/batch_summary.json"
    data = json.loads(path.read_text())
    data["experiments"][0]["export_exit_code"] = 1
    path.write_text(json.dumps(data))
    assert not module.collection_readiness(raw, "boutique", expected, 42)["ready"]


def test_changed_plan_and_wrong_shard_are_rejected(tmp_path):
    raw = tmp_path / "raw"
    expected = cohort(raw, "boutique")
    path = raw / "boutique-c2/batch_summary.json"
    data = json.loads(path.read_text())
    data["experiments"][0]["params"] = {"variant": "redirect"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen task"):
        module.collection_readiness(raw, "boutique", expected, 42)


def test_memory_admission_does_not_start_a_child(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_plan", lambda *_: {})
    watcher = module.AutoPipeline(args(tmp_path))
    monkeypatch.setattr(module, "read_memory_state", lambda: {"available_gib": 27, "total_gib": 125})
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("Memory pressure must defer execution"))
    with pytest.raises(RuntimeError, match="admission deferred"):
        watcher.stage("boutique", "ranking", lambda _: {}, ["report.json"])


def test_resume_receipt_rejects_modified_results(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_plan", lambda *_: {})
    watcher = module.AutoPipeline(args(tmp_path))
    root = watcher.out / "boutique/ranking/attempt-0001"
    root.mkdir(parents=True)
    result = root / "report.json"
    result.write_text('{"hit1":0.5}')
    module.write_json(root.parent / "receipt.json", {
        "out": str(root), "seed": 42, "outputs": {"report.json": module.sha(result)}})
    assert watcher.receipt("boutique", "ranking") == root
    result.write_text('{"hit1":1.0}')
    with pytest.raises(ValueError, match="no longer matches"):
        watcher.receipt("boutique", "ranking")


def test_missing_dataset_is_bounded_check_without_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_plan", lambda *_: {1: {}})
    config = args(tmp_path)
    config.limit = 2
    watcher = module.AutoPipeline(config)
    monkeypatch.setattr(watcher, "run_application", lambda *_: pytest.fail("An incomplete dataset must not train"))
    assert watcher.run() == 0
    assert json.loads((watcher.out / "status.json").read_text())["status"] == "bounded_check_only"


def test_raw_collection_cannot_be_used_as_job_output(tmp_path, monkeypatch):
    config = args(tmp_path)
    config.out_dir = config.data_root / "jobs"
    with pytest.raises(ValueError, match="outside raw"):
        module.AutoPipeline(config)


def test_revised_source_compiles_via_its_checked_preview(tmp_path, monkeypatch):
    paper, out = tmp_path / "paper", tmp_path / "out"
    paper.mkdir()
    out.mkdir()
    (paper / "main.pdf").write_bytes(b"old uploaded PDF")
    def build(command, cwd, timeout):
        assert command == ["sh", "build.sh", "--preview"]
        assert cwd == paper
        (paper / "main.pdf").write_bytes(b"new verified preview")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", build)
    module.compile_manuscript(paper, out, revised=True)
    assert (out / "main.pdf").read_bytes() == b"new verified preview"


def test_failed_preview_never_publishes_uploaded_old_pdf(tmp_path, monkeypatch):
    paper, out = tmp_path / "paper", tmp_path / "out"
    paper.mkdir()
    out.mkdir()
    (paper / "main.pdf").write_bytes(b"old uploaded PDF")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="compilation failed"):
        module.compile_manuscript(paper, out, revised=True)
    assert not (out / "main.pdf").exists()


def test_default_experiment_budget_preserves_ten_gib_after_worker_allocation():
    config = module.build_parser().parse_args([])
    assert config.stop_below_gib == 10
    assert config.admit_gib - config.memory_cap_gib == 10
