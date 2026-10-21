import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dyrca_rca import collection_monitor as module


def _monitor(tmp_path, monkeypatch, available=75):
    args = SimpleNamespace(out_dir=tmp_path / "monitor", collection_out_dir=tmp_path / "collection",
                           data_root=tmp_path / "data", kubeconfig=tmp_path / "kubeconfig", seed=42,
                           planner="rule", limit=None, concurrency=4, stall_minutes=90, check_only=False)
    monitor = module.CollectionMonitor(args)
    monitor.collection_lock = tmp_path / "collection.lock"
    monkeypatch.setattr(module, "read_memory_state", lambda: {"available_gib": available, "total_gib": 125})
    monkeypatch.setattr(module, "tagged_children", lambda _: [])
    monkeypatch.setattr(module, "process_matches", lambda *_: False)
    monkeypatch.setattr(monitor, "progress", lambda: {"boutique": {"remaining": 10}, "trainticket": {"remaining": 330}})
    (monitor.collection_out / "status.json").write_text(json.dumps({"status": "stopped", "active": []}))
    return monitor


def test_low_memory_defers_restart_without_mutating_cluster(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch, available=9.9)
    monkeypatch.setattr(monitor, "repair_backend", lambda: pytest.fail("Must not start infrastructure under memory pressure"))
    monkeypatch.setattr(monitor, "launch", lambda: pytest.fail("Must not restart under memory pressure"))
    assert monitor.run() == 0
    assert json.loads((monitor.out / "latest.json").read_text())["action"] == "deferred_memory"


def test_new_host_reserve_does_not_retain_twenty_eight_gib_restart_gate(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch, available=18)
    actions = []
    monkeypatch.setattr(monitor, "repair_backend", lambda: actions.append("repair"))
    monkeypatch.setattr(monitor, "launch", lambda: actions.append("launch"))
    assert monitor.policy.reserve_gib == monitor.controller.policy.reserve_gib == 10
    assert monitor.run() == 0
    assert actions == ["repair", "launch"]


def test_locked_live_run_is_not_restarted(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "process_matches", lambda *_: True)
    monkeypatch.setattr(monitor, "repair_backend", lambda: pytest.fail("A live collection must not be disturbed"))
    with monitor.collection_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert monitor.run() == 0
    assert monitor.event["action"] == "running"


def test_recovery_holds_lock_then_releases_before_launch(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)
    actions = []
    def repair():
        with monitor.collection_lock.open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions.append("repair")
    def launch():
        with monitor.collection_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions.append("launch")
        monitor.event["action"] = "restarted"
    monkeypatch.setattr(monitor, "repair_backend", repair)
    monkeypatch.setattr(monitor, "launch", launch)
    assert monitor.run() == 0
    assert actions == ["repair", "launch"]


def test_readonly_monitor_does_not_recover(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)
    monitor.args.check_only = True
    monkeypatch.setattr(monitor, "repair_backend", lambda: pytest.fail("Read-only check must not repair"))
    assert monitor.run() == 0
    assert monitor.event["action"] == "stopped_check_only"


def test_pause_marker_is_respected(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)
    (monitor.out / "PAUSE_AUTO_RESUME").touch()
    monkeypatch.setattr(monitor, "repair_backend", lambda: pytest.fail("Paused monitor must not repair"))
    assert monitor.run() == 0
    assert monitor.event["action"] == "paused"


def test_backend_failure_preserves_state_for_next_check(tmp_path, monkeypatch):
    monitor = _monitor(tmp_path, monkeypatch)
    def failing_repair():
        raise RuntimeError("Observability is unavailable")
    monkeypatch.setattr(monitor, "repair_backend", failing_repair)
    monkeypatch.setattr(monitor, "launch", lambda: pytest.fail("Failed repair must not launch"))
    assert monitor.run() == 1
    assert monitor.event["action"] == "deferred_error"
    assert monitor.event["error"] == "Observability is unavailable"


def _process(root, pid, module_name, state="S", config="/dedicated/kubeconfig", cluster="kind-tosem-copies"):
    directory = root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(f"python\0-m\0{module_name}\0".encode())
    (directory / "environ").write_bytes(f"KUBECONFIG={config}\0DYRCA_CLUSTER={cluster}\0".encode())
    (directory / "stat").write_text(f"{pid} (python) {state} 1 0 0\n")


def test_process_identity_refuses_unrelated_and_zombie_pids(tmp_path):
    _process(tmp_path, 11, "dyrca_rca.four_copy_collection")
    _process(tmp_path, 12, "unrelated")
    _process(tmp_path, 13, "dyrca_rca.four_copy_collection", state="Z")
    assert module.process_matches(11, "dyrca_rca.four_copy_collection", tmp_path)
    assert not module.process_matches(12, "dyrca_rca.four_copy_collection", tmp_path)
    assert not module.process_matches(13, "dyrca_rca.four_copy_collection", tmp_path)
    assert not module.process_matches(14, "dyrca_rca.four_copy_collection", tmp_path)


def test_orphan_discovery_is_restricted_to_dedicated_identity(tmp_path):
    _process(tmp_path, 11, "runner.scheduler")
    _process(tmp_path, 12, "runner.scheduler", cluster="kind-fault-exp")
    _process(tmp_path, 13, "runner.scheduler", config="/original/kubeconfig")
    _process(tmp_path, 14, "unrelated")
    assert module.tagged_children(Path("/dedicated/kubeconfig"), tmp_path) == [11]
