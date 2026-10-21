import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.config import get_services_for_namespace
from collector.tempo_collector import TempoCollector
from collector.loki_collector import LokiCollector
from dyrca_rca.collection_safety import MemoryPolicy, read_memory_state
from dyrca_rca.four_copy_resources import canary_deployment
from dyrca_rca.four_copy_collection import Supervisor
from runner.scheduler import BatchConfig, BatchScheduler, ExperimentTask
from verifier.topology_verifier import TopologyVerifier


def test_memory_policy_preserves_reserve_before_emergency(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 131072000 kB\nMemAvailable: 10485760 kB\n")
    state = read_memory_state(meminfo)
    assert state["available_gib"] == 10
    policy = MemoryPolicy(28, 20, 12, host_only=False)
    assert policy.state(30) == "admit"
    assert policy.state(25) == "hold"
    assert policy.state(18) == "drain"
    assert policy.state(state["available_gib"]) == "emergency"
    with pytest.raises(ValueError):
        MemoryPolicy(admit_gib=10, reserve_gib=20)


def test_memory_pressure_prevents_fault_even_when_cluster_is_healthy(monkeypatch):
    scheduler = BatchScheduler(BatchConfig(memory_min_available_gib=20, max_retries=0))
    monkeypatch.setattr("dyrca_rca.collection_safety.read_memory_state", lambda: {"available_gib": 18})
    monkeypatch.setattr(scheduler, "_wait_healthy", lambda namespace: True)
    def forbidden(_):
        pytest.fail("A fault must never be injected below the host memory reserve")
    monkeypatch.setattr(scheduler, "_run_fault_script", forbidden)
    result = scheduler._execute_task(ExperimentTask(index=1, script="faults/T1_hpa_scale.sh"))
    assert result.status == "pending"
    assert result.attempts == 1
    assert "memory protection" in result.error


def test_stop_marker_prevents_starting_next_fault(tmp_path):
    marker = tmp_path / "STOP_COLLECTION"
    marker.write_text("watchdog drain\n")
    scheduler = BatchScheduler(BatchConfig(stop_file=str(marker)))
    assert not scheduler._collection_admitted()


def _supervisor(tmp_path):
    args = SimpleNamespace(out_dir=tmp_path, kubeconfig=tmp_path / "kubeconfig",
                           data_root=tmp_path / "data", seed=42, planner="rule", limit=1,
                           concurrency=4, admit_gib=28, reserve_gib=20, emergency_gib=12,
                           node_memory_guards=True)
    return Supervisor(args)


def test_copy_activation_refuses_all_mutations_without_memory_headroom(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path)
    monkeypatch.setattr(supervisor, "pressure", lambda: {"available_gib": 25, "nodes": {}})
    def forbidden(*_, **__):
        pytest.fail("No deployment may be scaled up without admission memory")
    monkeypatch.setattr(supervisor, "kube", forbidden)
    assert not supervisor.activate("trainticket-c1")
    assert supervisor.status["active"] == []


@pytest.mark.parametrize("available,ratio", [(40, 0.2), (70, 0.6)])
def test_copy_startup_budget_is_reserved_before_scaling(tmp_path, monkeypatch, available, ratio):
    supervisor = _supervisor(tmp_path)
    monkeypatch.setattr(supervisor, "pressure", lambda: {
        "available_gib": available, "nodes": {"tosem-copies-worker": ratio}})
    def forbidden(*_, **__):
        pytest.fail("A copy must not start when its projected memory exceeds reserve or a node cap")
    monkeypatch.setattr(supervisor, "kube", forbidden)
    assert not supervisor.activate("trainticket-c1")


@pytest.mark.parametrize("monitor_failed", [False, True])
def test_watchdog_stops_on_emergency_or_unavailable_monitor(tmp_path, monkeypatch, monitor_failed):
    supervisor = _supervisor(tmp_path)
    class ImmediateEvent:
        stopped = False
        def wait(self, _seconds):
            return self.stopped
        def set(self):
            self.stopped = True
    supervisor.stop = ImmediateEvent()
    def pressure():
        if monitor_failed:
            raise RuntimeError("metrics unavailable")
        return {"available_gib": 10, "nodes": {"tosem-copies-worker": 0.4}}
    monkeypatch.setattr(supervisor, "pressure", pressure)
    supervisor.watchdog()
    assert supervisor.stop.stopped
    assert supervisor.stop_file.exists()
    assert "memory" in supervisor.stop_file.read_text()


def test_service_search_and_summary_are_copy_scoped():
    services = get_services_for_namespace("trainticket-c3")
    assert "ts-travel-service.trainticket-c3" in services
    assert all(service.endswith(".trainticket-c3") for service in services)
    assert "frontend.boutique-c2" in get_services_for_namespace("boutique-c2")
    trace = {"batches": [{"resource": {"attributes": [{
        "key": "service.name", "value": {"stringValue": services[0]}}]},
        "scopeSpans": [{"spans": [{"name": "call", "startTimeUnixNano": "1", "endTimeUnixNano": "2"}]}]}]}
    assert TempoCollector._summarize_trace(trace)[0]["service"] == services[0].rsplit(".", 1)[0]
    with pytest.raises(ValueError):
        get_services_for_namespace("unrelated-namespace")


def test_static_log_targets_retain_real_service_attribution():
    labels = {"filename": "/var/log/pods/boutique-c1_shippingservice-58859d5845-9q877_uid/istio-proxy/0.log",
              "namespace": "boutique-c1"}
    row = LokiCollector._parse_entry(1_000_000_000, "request failed", labels)
    assert row["service"] == "shippingservice"
    assert row["pod"] == "shippingservice-58859d5845-9q877"
    # Explicit discovery labels take precedence over the filename fallback.
    row = LokiCollector._parse_entry(1_000_000_000, "request failed", dict(labels, app="discovered", pod="known"))
    assert row["service"] == "discovered"
    assert row["pod"] == "known"


def test_canary_retains_memory_limits_and_instrumentation():
    source = {"spec": {"template": {"metadata": {"labels": {"app": "ts-travel-service", "version": "v1"}},
                                   "spec": {"containers": [
                                       {"name": "server", "resources": {"limits": {"memory": "500Mi"}},
                                        "env": [{"name": "COLLECTOR_SERVICE_ADDR", "value": "tempo:4317"}]},
                                       {"name": "istio-proxy", "resources": {"limits": {"memory": "1Gi"}}}]}}}}
    result = canary_deployment(source, "trainticket-c2", "ts-travel-service")
    assert result["spec"]["template"]["spec"] == source["spec"]["template"]["spec"]
    assert result["spec"]["template"]["metadata"]["labels"]["version"] == "v2.0.0"
    assert source["spec"]["template"]["metadata"]["labels"]["version"] == "v1"
    with pytest.raises(ValueError):
        canary_deployment(source, "trainticket", "ts-travel-service")


def test_canary_verification_matches_only_observed_version_split():
    verifier = object.__new__(TopologyVerifier)
    assert verifier._gt_match("version_split", "mixed", ["node_scale", "edge_split"])
    assert verifier._gt_match("version_coexistence", "mixed", ["node_scale", "edge_split"])
    assert not verifier._gt_match("version_split", "node_scale", ["node_scale"])


def test_host_policy_reserves_ten_gib_and_rejects_invalid_thresholds():
    policy = MemoryPolicy()
    assert policy.host_only
    assert policy.state(10.1) == "admit"
    assert policy.state(9.9) == "emergency"
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            MemoryPolicy(admit_gib=value)


def test_host_only_activation_ignores_old_node_gate_but_reserves_startup(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path)
    supervisor.policy = MemoryPolicy()
    supervisor.args.ready_timeout = 1
    monkeypatch.setattr(supervisor, "pressure", lambda: {
        "available_gib": 33.9, "nodes": {"tosem-copies-worker": 0.60}})
    calls = []
    monkeypatch.setattr(supervisor, "kube", lambda *a, **k: calls.append(a))
    assert not supervisor.activate("trainticket-c2")
    assert calls == []
    monkeypatch.setattr(supervisor, "pressure", lambda: {
        "available_gib": 35, "nodes": {"tosem-copies-worker": 0.60}})
    monkeypatch.setattr(supervisor, "ready", lambda _: True)
    monkeypatch.setattr(supervisor, "telemetry_ready", lambda _: True)
    assert supervisor.activate("trainticket-c2")
    assert supervisor.status["active"] == ["trainticket-c2"]


def test_host_admission_relaxation_preserves_dedicated_hard_caps(tmp_path, monkeypatch):
    from dyrca_rca.four_copy_collection import NODES, WORKERS
    supervisor = _supervisor(tmp_path)
    supervisor.policy = MemoryPolicy()
    commands = []
    def command(argv):
        commands.append(argv)
        if argv[1] == "inspect":
            return json.dumps([{"Name": name, "HostConfig": {
                "Memory": (3 if name.endswith('control-plane') else 14) * 1024 ** 3}} for name in NODES])
        return ""
    monkeypatch.setattr(supervisor, "command", command)
    supervisor.set_memory_caps()
    assert commands[0] == ["docker", "update", "--memory", "14g", "--memory-swap", "14g", *WORKERS]
    assert commands[1] == ["docker", "update", "--memory", "3g", "--memory-swap", "3g", NODES[-1]]


def test_completed_shard_resume_requires_every_matching_nonexcluded_export(tmp_path):
    supervisor = _supervisor(tmp_path)
    namespace = "trainticket-c1"
    directory = supervisor.args.data_root / namespace
    directory.mkdir(parents=True)
    expected = {i: {"scenario_id": "T3", "target": "svc", "params": {}, "repeat": i} for i in (1, 5)}
    rows = [dict(expected[i], index=i, status="success", experiment_id=f"case-{i}",
                 output_dir=str(directory / f"case-{i}"), fault_exit_code=0,
                 export_exit_code=0, verify_exit_code=0) for i in expected]
    for row in rows:
        case = Path(row['output_dir'])
        for name in ('metadata.json','ground_truth.json','topology/pre.json','topology/during.json',
                     'topology/post.json','traces/traces_raw.json','logs/all.jsonl',
                     'metrics/request_rate.csv','traces/trace_summary.csv'):
            path = case / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('timestamp,value\n1,2\n' if name.endswith('.csv') else '{}\n')
    report = {"settings": {"seed":42,"namespace":namespace,"shard_count":4,"shard_index":0},
              "experiments":rows}
    summary = directory / "batch_summary.json"
    summary.write_text(json.dumps(report))
    assert supervisor.reusable_shard(namespace, expected, None) == 2
    report['experiments'] = rows[:1]
    summary.write_text(json.dumps(report))
    assert supervisor.reusable_shard(namespace, expected, None) == 0
    assert supervisor.reusable_shard(namespace, expected, 1) == 1
    report['experiments'] = rows
    rows[1]['target'] = 'wrong'
    summary.write_text(json.dumps(report))
    assert supervisor.reusable_shard(namespace, expected, None) == 0
    rows[1]['target'] = 'svc'
    summary.write_text(json.dumps(report))
    (tmp_path/'resume_exclusions.json').write_text(json.dumps({'excluded_output_dirs':[rows[1]['output_dir']]}))
    assert supervisor.reusable_shard(namespace, expected, None) == 0
