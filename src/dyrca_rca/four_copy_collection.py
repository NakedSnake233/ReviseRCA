"""Collect deterministic shards with bounded memory and staged platforms.

The supervisor keeps only one platform active, admits copies one at a time,
and leaves applications in standby on completion or an interrupted run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import yaml

from dyrca_rca.collection_safety import GIB, MemoryPolicy, read_memory_state, resume_exclusions

WORKERS = ["tosem-copies-worker" + suffix for suffix in ("", "2", "3", "4", "5")]
NODES = WORKERS + ["tosem-copies-control-plane"]
NAMESPACES = [f"{system}-c{i}" for system in ("boutique", "trainticket") for i in range(1, 5)]


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.project = Path(__file__).resolve().parents[2]
        self.out = args.out_dir.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "logs").mkdir(exist_ok=True)
        self.env = os.environ.copy()
        self.env.update(KUBECONFIG=str(args.kubeconfig.resolve()),
                        PYTHONPATH=f"{self.project / 'src'}:{self.project}",
                        DYRCA_CLUSTER="kind-tosem-copies")
        self.policy = MemoryPolicy(args.admit_gib, args.reserve_gib, args.emergency_gib,
                                   host_only=not getattr(args, "node_memory_guards", False))
        self.stop = threading.Event()
        self.stop_file = self.out / "STOP_COLLECTION"
        self.children = []
        self.logs = []
        self.status = {"seed": args.seed, "planner": args.planner, "limit": args.limit,
                       "supervisor_pid": os.getpid(),
                       "data_root": str(args.data_root.resolve()),
                       "concurrency": args.concurrency, "active": [],
                       "completed": [], "status": "starting", "reason": "",
                       "memory_policy": vars(self.policy),
                       "worker_cap_gib": 14,
                       "control_plane_cap_gib": 3,
                       "hpa_max_replicas": 3}
        self.status["trainticket_application_cpu_request"] = "same as existing runtime CPU limit"

    def save(self):
        self.status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        temporary = self.out / "status.tmp"
        temporary.write_text(json.dumps(self.status, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.out / "status.json")

    def command(self, command, timeout=40):
        result = subprocess.run(command, env=self.env, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{command[:5]}: {result.stderr[-1200:]}")
        return result.stdout

    def kube(self, *args, timeout=40):
        return self.command(["kubectl", "--request-timeout=20s", *args], timeout=timeout)

    def standby(self, namespace):
        self.kube("scale", "deployment", "--all", "--replicas=0", "-n", namespace, timeout=90)
        print(f"standby: {namespace}", flush=True)

    def set_memory_caps(self):
        # Keep the established 73 GiB aggregate hard bound. The host-reserve
        # relaxation removes projected per-node admission, not these caps.
        self.command(["docker", "update", "--memory", "14g", "--memory-swap", "14g", *WORKERS])
        self.command(["docker", "update", "--memory", "3g", "--memory-swap", "3g", NODES[-1]])
        settings = json.loads(self.command(["docker", "inspect", *NODES]))
        for node in settings:
            expected = (3 if node["Name"].endswith("control-plane") else 14) * GIB
            if node["HostConfig"]["Memory"] != expected:
                raise RuntimeError(f"Memory cap was not applied to {node['Name']}")

    def pressure(self):
        memory = read_memory_state()
        output = self.command(["docker", "stats", "--no-stream", "--format", "{{json .}}", *NODES], timeout=15)
        nodes = [json.loads(line) for line in output.splitlines() if line.strip()]
        ratios = {node["Name"]: float(node["MemPerc"].rstrip("%")) / 100 for node in nodes}
        if set(ratios) != set(NODES):
            raise RuntimeError("Incomplete dedicated-node memory readings; admission refused")
        return {**memory, "nodes": ratios}

    def watchdog(self):
        errors = 0
        while not self.stop.wait(5):
            try:
                host = read_memory_state()
                if self.policy.state(host["available_gib"]) == "emergency":
                    self.status["reason"] = f"emergency host memory: {host['available_gib']:.1f} GiB available"
                    self.stop_file.write_text(self.status["reason"] + "\n")
                    self.stop.set()
                    return
                memory = self.pressure()
                errors = 0
                with (self.out / "memory.jsonl").open("a") as stream:
                    stream.write(json.dumps({"time": time.time(), **memory}, sort_keys=True) + "\n")
                host_state = self.policy.state(memory["available_gib"])
                # Workers must retain room for fault-created replicas. Control
                # plane pressure also halts admission before its hard cap.
                node_pressure = any(ratio >= 0.94 for ratio in memory["nodes"].values())
                if host_state in ("drain", "emergency") or node_pressure:
                    reason = f"memory protection: available={memory['available_gib']:.1f} GiB; node pressure={node_pressure}"
                    self.stop_file.write_text(reason + "\n")
                    self.status["reason"] = reason
                    if host_state == "emergency" or node_pressure:
                        self.stop.set()
                        return
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors += 1
                if errors >= 3:
                    self.status["reason"] = f"memory monitoring unavailable: {exc}"
                    self.stop_file.write_text(self.status["reason"] + "\n")
                    self.stop.set()
                    return

    def start_child(self, command, name, extra_env=None):
        log = (self.out / "logs" / f"{name}.log").open("a")
        self.logs.append(log)
        env = dict(self.env, **(extra_env or {}))
        process = subprocess.Popen(command, cwd=self.project, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(process)
        return process

    def forward_apis(self):
        services = [("prom-kube-prometheus-stack-prometheus", 9190, 9090),
                    ("loki", 3190, 3100), ("tempo", 3290, 3200)]
        for name, local, remote in services:
            self.start_child(["kubectl", "port-forward", "-n", "observability", f"svc/{name}",
                              f"{local}:{remote}"], f"port-{local}")
        for _ in range(30):
            try:
                for url in ("http://127.0.0.1:9190/-/ready", "http://127.0.0.1:3190/ready",
                            "http://127.0.0.1:3290/api/echo"):
                    requests.get(url, timeout=3).raise_for_status()
                return
            except requests.RequestException:
                if self.stop.wait(2):
                    break
        raise RuntimeError("Observability APIs are not ready")

    def ready(self, namespace):
        obj = json.loads(self.kube("get", "deployment", "-n", namespace, "-o", "json"))
        return bool(obj["items"]) and all(
            item["spec"].get("replicas", 0) > 0 and
            item.get("status", {}).get("availableReplicas", 0) >= item["spec"]["replicas"]
            for item in obj["items"])

    def telemetry_ready(self, namespace):
        now = int(time.time())
        service = "frontend" if namespace.startswith("boutique") else "ts-travel-service"
        metrics = requests.get("http://127.0.0.1:9190/api/v1/query", timeout=8,
                               params={"query": f'sum(rate(istio_requests_total{{namespace="{namespace}"}}[1m]))'}).json()
        traces = requests.get("http://127.0.0.1:3290/api/search", timeout=8,
                              params={"q": f'{{resource.service.name="{service}.{namespace}"}}',
                                      "start": now - 180, "end": now, "limit": 1}).json()
        logs = requests.get("http://127.0.0.1:3190/loki/api/v1/query_range", timeout=8,
                            params={"query": f'{{namespace="{namespace}"}}', "limit": 1,
                                    "start": str((now - 180) * 10**9), "end": str(now * 10**9)}).json()
        result = metrics.get("data", {}).get("result", [])
        return (bool(result) and float(result[0]["value"][1]) > 0 and bool(traces.get("traces"))
                and bool(logs.get("data", {}).get("result")))

    def activate(self, namespace):
        memory = self.pressure()
        system = namespace.rsplit("-c", 1)[0]
        # Budget startup before scaling, so a copy admitted just above the host
        # floor cannot immediately consume that reserve. Node budgets apply
        # only to the optional legacy per-node policy.
        host_increment, node_increment = (24.0, 5.0) if system == "trainticket" else (8.0, 1.5)
        projected_available = memory["available_gib"] - host_increment
        projected_nodes_safe = all(
            ratio * 14 + node_increment <= 14 * 0.85
            for name, ratio in memory["nodes"].items() if name in WORKERS)
        node_blocked = any(ratio >= 0.94 for ratio in memory["nodes"].values()) or (
            not self.policy.host_only and (
                any(ratio >= 0.85 for ratio in memory["nodes"].values()) or not projected_nodes_safe))
        if self.policy.state(memory["available_gib"]) != "admit" or \
                projected_available < self.policy.reserve_gib or node_blocked:
            return False
        self.kube("scale", "deployment", "--all", "--replicas=1", "-n", namespace, timeout=90)
        self.status["active"].append(namespace)
        self.save()
        deadline = time.monotonic() + self.args.ready_timeout
        while time.monotonic() < deadline and not self.stop.is_set():
            if self.ready(namespace):
                try:
                    if self.telemetry_ready(namespace):
                        print(f"ready with metrics/logs/traces: {namespace}", flush=True)
                        return True
                except (requests.RequestException, ValueError, KeyError):
                    pass
            if self.stop_file.exists():
                raise RuntimeError(self.stop_file.read_text().strip())
            self.stop.wait(10)
        if self.stop.is_set():
            raise InterruptedError(f"Protected collection interrupted while preparing {namespace}")
        raise RuntimeError(f"Readiness or telemetry preflight timed out for {namespace}")

    def scheduler_config(self, namespace):
        system = namespace.rsplit("-c", 1)[0]
        config = yaml.safe_load((self.project / f"config/tosem_candidate_{system}_20260930.yaml").read_text())
        config["settings"].update(namespace=namespace, project_dir=str(self.project), python=sys.executable,
                                  seed=self.args.seed, require_healthy=True,
                                  memory_min_available_gib=self.policy.reserve_gib,
                                  resume_exclusions_file=str(self.out / "resume_exclusions.json"),
                                  stop_file=str(self.stop_file))
        config["observability"] = {"prom_url": "http://127.0.0.1:9190", "loki_url": "http://127.0.0.1:3190",
                                   "tempo_url": "http://127.0.0.1:3290"}
        root = self.args.data_root.resolve() / namespace
        config["output"] = {"data_dir": str(root), "summary_file": str(root / "batch_summary.json"),
                            "log_file": str(root / "batch.log")}
        path = self.out / f"{namespace}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return path, root / "batch_summary.json"

    def reusable_shard(self, namespace, expected, limit):
        """Only skip warming a shard if every required frozen-plan task exists."""
        from runner.scheduler import BatchScheduler
        from dyrca_rca.experiments.index_four_copy_collection import _identity
        summary = self.args.data_root.resolve() / namespace / "batch_summary.json"
        if not summary.exists():
            return 0
        report = json.loads(summary.read_text())
        settings = report.get("settings", {})
        shard = int(namespace[-1]) - 1
        if (settings.get("seed") != self.args.seed or settings.get("namespace") != namespace or
                settings.get("shard_count") != 4 or settings.get("shard_index") != shard):
            return 0
        wanted = sorted(i for i in expected if (i - 1) % 4 == shard)
        if limit is not None:
            wanted = wanted[:limit]
        records = report.get("experiments", [])
        by_index = {row["index"]: row for row in records}
        if len(records) != len(by_index) or not wanted or any(i not in by_index for i in wanted):
            return 0
        excluded = resume_exclusions(self.out / "resume_exclusions.json")
        for index in wanted:
            row = by_index[index]
            output = Path(row.get("output_dir") or ".").resolve()
            if (_identity(row) != expected[index] or not BatchScheduler._resume_record_is_complete(row) or
                    str(output) in excluded or output.parent != summary.parent or
                    output.name != row.get("experiment_id")):
                return 0
        return len(wanted)

    def run_wave(self, system, limit, phase):
        from dyrca_rca.experiments.index_four_copy_collection import _plan
        queued = [f"{system}-c{i}" for i in range(1, 5)]
        expected = _plan(self.project / f"config/tosem_candidate_{system}_20260930.yaml", self.args.seed)
        # Resume completed shards without consuming memory on a cold startup.
        for namespace in list(queued):
            reused = self.reusable_shard(namespace, expected, limit)
            if reused:
                queued.remove(namespace)
                summary = self.args.data_root.resolve() / namespace / "batch_summary.json"
                self.status["completed"].append({"namespace": namespace, "phase": phase,
                                                  "succeeded": reused, "summary": str(summary), "reused": True})
                self.save()
                print(f"reusing complete {phase}: {namespace}", flush=True)
        running = {}
        last_admission = time.monotonic()
        while queued or running:
            if self.stop.is_set():
                raise RuntimeError(self.status.get("reason") or "Collection was interrupted")
            for namespace, (process, summary) in list(running.items()):
                if process.poll() is not None:
                    report = json.loads(summary.read_text()) if summary.exists() else {}
                    self.standby(namespace)
                    self.status["active"].remove(namespace)
                    del running[namespace]
                    if process.returncode or not report.get("total_experiments") or report.get("pending") or report.get("failed"):
                        raise RuntimeError(f"{namespace}: incomplete batch; check scheduler log and resume summary")
                    self.status["completed"].append({"namespace": namespace, "phase": phase,
                                                      "succeeded": report["succeeded"], "summary": str(summary)})
                    self.save()
            if self.stop_file.exists():
                if not running:
                    raise RuntimeError(self.stop_file.read_text().strip())
                self.stop.wait(5)
                continue
            if queued and len(running) < self.args.concurrency:
                namespace = queued[0]
                if self.activate(namespace):
                    queued.pop(0)
                    config, summary = self.scheduler_config(namespace)
                    command = [sys.executable, "-m", "runner.scheduler", "--config", str(config),
                               "--shard-index", str(int(namespace[-1]) - 1), "--shard-count", "4"]
                    if limit is not None:
                        command += ["--limit", str(limit)]
                    if self.args.scenarios:
                        command += ["--scenarios", self.args.scenarios]
                    if summary.exists():
                        command += ["--resume-summary", str(summary)]
                    process = self.start_child(command, namespace, {"DYRCA_COPY_ID": namespace})
                    running[namespace] = (process, summary)
                    last_admission = time.monotonic()
                    print(f"collecting: {namespace} ({phase}), pid={process.pid}", flush=True)
                    continue
                if not running and time.monotonic() - last_admission > 600:
                    raise RuntimeError("Memory reserve does not permit a new copy; resumable summaries are preserved")
            self.stop.wait(5)

    def cleanup(self):
        self.stop.set()
        for process in self.children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        while any(p.poll() is None for p in self.children) and time.monotonic() < deadline:
            time.sleep(0.2)
        for process in self.children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for namespace in list(self.status["active"]):
            try:
                self.standby(namespace)
                self.status["active"].remove(namespace)
            except Exception as exc:
                self.status.setdefault("cleanup_errors", []).append(f"{namespace}: {exc}")
        for log in self.logs:
            log.close()
        self.save()

    def run(self):
        lock = (self.project / "artifacts/tosem_four_copies/collection.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.kube("config", "current-context").strip() != "kind-tosem-copies":
            raise RuntimeError("Only the dedicated kind-tosem-copies context is allowed")
        inotify_limit = int(Path("/proc/sys/fs/inotify/max_user_instances").read_text().strip())
        if inotify_limit < 1024:
            raise RuntimeError("Host inotify limit is below 1024; do not restart all application copies")
        (self.out / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        self.stop_file.unlink(missing_ok=True)
        self.set_memory_caps()
        for namespace in NAMESPACES:
            self.standby(namespace)
        self.save()
        watchdog = threading.Thread(target=self.watchdog, daemon=True)
        watchdog.start()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: self.stop.set())
        try:
            self.forward_apis()
            self.status["status"] = "collecting"
            self.save()
            systems = ("boutique", "trainticket") if self.args.system == "all" else (self.args.system,)
            rounds = [("bounded", self.args.limit)] if self.args.limit else [("validation", 1), ("formal", None)]
            for phase, limit in rounds:
                for system in systems:
                    self.status.update(phase=phase, system=system)
                    self.save()
                    self.run_wave(system, limit, phase)
            self.status["status"] = "complete"
        except Exception as exc:
            self.status.update(status="stopped", reason=str(exc))
            print(f"stopped safely: {exc}", file=sys.stderr, flush=True)
            raise
        finally:
            self.cleanup()
            watchdog.join(timeout=16)
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/experiments_tosem_four_copies"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/tosem_four_copies/guarded"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=["rule"], default="rule", help="Collection does not require an LLM")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system", choices=["all", "boutique", "trainticket"], default="all")
    parser.add_argument("--scenarios")
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 3, 4], default=2,
                        help="Active copies; four fixed data shards are retained")
    parser.add_argument("--kubeconfig", type=Path, default=Path("artifacts/tosem_four_copies/kubeconfig"))
    parser.add_argument("--admit-gib", type=float, default=10)
    parser.add_argument("--reserve-gib", type=float, default=10)
    parser.add_argument("--emergency-gib", type=float, default=10)
    parser.add_argument("--node-memory-guards", action="store_true",
                        help="Opt into legacy projected per-node admission checks; hard caps always remain")
    parser.add_argument("--ready-timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    Supervisor(args).run()


if __name__ == "__main__":
    main()
