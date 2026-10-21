"""Check collection progress and safely recover the dedicated four-copy run.

Run once from cron every twenty minutes. A collection lock prevents duplicate
supervisors, and memory pressure defers recovery rather than forcing a restart.
"""
from __future__ import annotations

import argparse
import calendar
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dyrca_rca.collection_safety import MemoryPolicy, read_memory_state, resume_exclusions
from dyrca_rca.collection_contamination import update_exclusions
from dyrca_rca.four_copy_collection import NAMESPACES, NODES, Supervisor
from dyrca_rca.experiments.index_four_copy_collection import _identity, _plan
from runner.scheduler import BatchScheduler


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_matches(pid, module, proc_root=Path("/proc")):
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        argv = (proc_root / str(pid) / "cmdline").read_bytes().decode().split("\0")
        state = (proc_root / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, UnicodeError, IndexError):
        return False
    return module in argv and state != "Z"


def tagged_children(kubeconfig, proc_root=Path("/proc")):
    """Find only processes launched with the dedicated collection identity."""
    found = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / "cmdline").read_bytes().decode().split("\0")
            env = dict(entry.split("=", 1) for entry in
                       (directory / "environ").read_bytes().decode().split("\0") if "=" in entry)
        except (OSError, UnicodeError, ValueError):
            continue
        if env.get("DYRCA_CLUSTER") != "kind-tosem-copies" or \
                env.get("KUBECONFIG") != str(Path(kubeconfig).resolve()):
            continue
        modules = {"runner.scheduler", "collector.exporter", "verifier.topology_verifier"}
        is_forward = Path(argv[0]).name == "kubectl" and "port-forward" in argv
        is_fault = Path(argv[0]).name == "bash" and any("/dyrca/faults/" in arg for arg in argv[1:])
        if modules.intersection(argv) or is_forward or is_fault:
            found.append(int(directory.name))
    return found


class CollectionMonitor:
    def __init__(self, args):
        self.args = args
        self.project = Path(__file__).resolve().parents[2]
        self.out = args.out_dir.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.collection_out = args.collection_out_dir.resolve()
        self.collection_out.mkdir(parents=True, exist_ok=True)
        self.collection_lock = self.project / "artifacts/tosem_four_copies/collection.lock"
        supervisor_args = SimpleNamespace(
            out_dir=self.collection_out, kubeconfig=args.kubeconfig, data_root=args.data_root,
            seed=args.seed, planner=args.planner, limit=args.limit, concurrency=args.concurrency,
            admit_gib=10, reserve_gib=10, emergency_gib=10)
        self.controller = Supervisor(supervisor_args)
        self.policy = MemoryPolicy()
        self.event = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "time": time.time(), "interval_minutes": 20, "seed": args.seed,
                      "planner": args.planner, "data_root": str(args.data_root.resolve()),
                      "action": "checking", "repairs": []}

    def record(self):
        write_json(self.out / "latest.json", self.event)
        with (self.out / "checks.jsonl").open("a") as stream:
            stream.write(json.dumps(self.event, sort_keys=True) + "\n")
        print(json.dumps(self.event, sort_keys=True), flush=True)

    def progress(self):
        result = {}
        last_progress = 0.0
        exclusions = resume_exclusions(self.collection_out / "resume_exclusions.json")
        for system in ("boutique", "trainticket"):
            plan = _plan(self.project / f"config/tosem_candidate_{system}_20260930.yaml", self.args.seed)
            seen, succeeded, failed, quarantined = set(), 0, 0, 0
            for i in range(1, 5):
                namespace = f"{system}-c{i}"
                path = self.args.data_root.resolve() / namespace / "batch_summary.json"
                if not path.exists():
                    continue
                report = json.loads(path.read_text())
                if report.get("settings", {}).get("seed") != self.args.seed:
                    continue
                if report.get("settings", {}).get("namespace") != namespace:
                    raise ValueError(f"Unexpected summary namespace: {namespace}")
                last_progress = max(last_progress, path.stat().st_mtime)
                for row in report.get("experiments", []):
                    index = row["index"]
                    if index not in plan or (index - 1) % 4 != i - 1 or index in seen or \
                            _identity(row) != plan[index]:
                        raise ValueError(f"Task plan mismatch: {namespace}, {index}")
                    seen.add(index)
                    is_excluded = str(Path(row.get("output_dir") or ".").resolve()) in exclusions
                    succeeded += row.get("status") == "success" and not is_excluded
                    quarantined += row.get("status") == "success" and is_excluded
                    failed += row.get("status") == "failed"
            result[system] = {"expected": len(plan), "succeeded": succeeded, "failed": failed,
                              "remaining": len(plan) - succeeded, "listed": len(seen), "quarantined": quarantined}
        self.event["progress"] = result
        self.event["last_progress_time"] = last_progress
        return result

    def collection_pid(self):
        try:
            return int((self.collection_out / "supervisor.pid").read_text())
        except (OSError, ValueError):
            return None

    def is_stalled(self, status):
        # Each fault, export and readiness step is bounded separately. A long
        # cold start or retry is not mistaken for a stopped process.
        if status.get("status") != "collecting":
            return False
        history_time = self.event.get("last_progress_time", 0)
        try:
            started = calendar.timegm(time.strptime(status["updated_at"], "%Y-%m-%dT%H:%M:%SZ"))
        except (KeyError, ValueError):
            started = time.time()
        return time.time() - max(history_time, started) > self.args.stall_minutes * 60

    def stop_groups(self, pids):
        own_group = os.getpgrp()
        groups = set()
        for pid in pids:
            try:
                group = os.getpgid(pid)
                if group > 1 and group != own_group:
                    groups.add(group)
            except ProcessLookupError:
                pass
        signalled = sorted(groups)
        for group in groups:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        while groups and time.monotonic() < deadline:
            for group in list(groups):
                try:
                    os.killpg(group, 0)
                except ProcessLookupError:
                    groups.remove(group)
            if groups:
                time.sleep(0.25)
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if pids:
            self.event["repairs"].append({"terminated_orphan_groups": signalled, "processes": pids})

    def cohort_complete(self):
        for namespace in NAMESPACES:
            path = self.args.data_root.resolve() / namespace / "batch_summary.json"
            if not path.exists():
                return False
            report = json.loads(path.read_text())
            if not all(BatchScheduler._resume_record_is_complete(row) for row in report.get("experiments", [])):
                return False
        return True

    def repair_backend(self):
        controller = self.controller
        archive = self.out / "recovery" / str(int(time.time()))
        archive.mkdir(parents=True, exist_ok=True)
        for namespace in NAMESPACES:
            source = self.args.data_root.resolve() / namespace / "batch_summary.json"
            if source.exists():
                (archive / f"{namespace}-before.json").write_bytes(source.read_bytes())
        exclusions = update_exclusions(self.args.data_root, self.collection_out, [archive])
        self.event["resume_exclusion_count"] = len(exclusions["excluded_output_dirs"])
        controller.set_memory_caps()
        running = controller.command(["docker", "inspect", "--format", "{{.Name}} {{.State.Running}}", *NODES])
        stopped = [line.split()[0].lstrip("/") for line in running.splitlines() if line.split()[-1] == "false"]
        if stopped:
            if any(name not in NODES for name in stopped):
                raise RuntimeError("Refusing to restart an unrelated Docker container")
            controller.command(["docker", "start", *stopped], timeout=90)
            self.event["repairs"].append({"started_dedicated_nodes": stopped})
        inotify = int(Path("/proc/sys/fs/inotify/max_user_instances").read_text().strip())
        if inotify < 1024:
            controller.command(["docker", "run", "--rm", "--privileged", "--entrypoint", "/bin/sh",
                                "kindest/node:v1.30.0", "-c", "sysctl -w fs.inotify.max_user_instances=8192"], timeout=90)
            self.event["repairs"].append({"inotify_limit_before": inotify, "inotify_limit_after": 8192})
        if controller.kube("config", "current-context").strip() != "kind-tosem-copies":
            raise RuntimeError("Automatic recovery is restricted to kind-tosem-copies")
        # Stop only dedicated applications. Existing exports and all other
        # clusters are outside the recovery scope.
        for namespace in NAMESPACES:
            controller.standby(namespace)
        subprocess.run([sys.executable, str(self.project / "scripts/reset_tosem_copy_state.py"),
                        "--kubeconfig", str(self.args.kubeconfig.resolve()),
                        "--out-dir", str(archive)],
                       cwd=self.project, env=controller.env, check=True, timeout=600)
        nodes = json.loads(controller.kube("get", "nodes", "-o", "json"))["items"]
        ready = {node["metadata"]["name"] for node in nodes if any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in node.get("status", {}).get("conditions", []))}
        if set(NODES) != ready:
            raise RuntimeError(f"Dedicated nodes still recovering: {sorted(set(NODES) - ready)}")
        self.event["repairs"].append("standby copies, restore selectors, remove fault residues, restore HPA cap 3")
        # Recheck after cleanup; do not clear the collection stop marker until
        # the existing supervisor's own admission checks are allowed to run.
        memory = controller.pressure()
        self.event["memory_after_repair"] = memory
        if self.policy.state(memory["available_gib"]) != "admit" or (
                not self.policy.host_only and any(r >= 0.85 for r in memory["nodes"].values())):
            raise RuntimeError("Memory reserve not sufficient after recovery; restart deferred")

    def launch(self):
        command = [sys.executable, "-u", "-m", "dyrca_rca.four_copy_collection",
                   "--data-root", str(self.args.data_root.resolve()), "--out-dir", str(self.collection_out),
                   "--kubeconfig", str(self.args.kubeconfig.resolve()), "--seed", str(self.args.seed),
                   "--planner", self.args.planner, "--concurrency", str(self.args.concurrency)]
        if self.args.limit is not None:
            command += ["--limit", str(self.args.limit)]
        with (self.collection_out / "launcher.log").open("a") as log:
            process = subprocess.Popen(command, cwd=self.project, env=self.controller.env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (self.collection_out / "supervisor.pid").write_text(str(process.pid) + "\n")
        self.event.update(action="restarted", supervisor_pid=process.pid, restart_time=time.time())

    def check(self):
        progress = self.progress()
        self.event["host_memory"] = read_memory_state()
        status_path = self.collection_out / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        self.event["collection_status"] = {k: status.get(k) for k in ("status", "phase", "system", "reason", "active")}
        if (self.out / "PAUSE_AUTO_RESUME").exists():
            self.event["action"] = "paused"
            return
        if status.get("status") == "complete" and all(v["remaining"] == 0 for v in progress.values()) and self.cohort_complete():
            self.event["action"] = "complete"
            return
        lock = self.collection_lock.open("a")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pid = self.collection_pid()
                self.event["supervisor_pid"] = pid
                if not process_matches(pid, "dyrca_rca.four_copy_collection"):
                    self.event["action"] = "busy_unknown_owner"
                elif not self.is_stalled(status) or self.args.check_only:
                    self.event["action"] = "running"
                else:
                    # Request the supervisor's own cleanup first. Orphans are
                    # cleaned in a subsequent check only after the lock clears.
                    os.kill(pid, signal.SIGTERM)
                    self.event["action"] = "stalled_supervisor_stop_requested"
                return
            if self.args.check_only:
                self.event["action"] = "stopped_check_only"
                return
            if process_matches(self.collection_pid(), "dyrca_rca.four_copy_collection"):
                self.event["action"] = "supervisor_starting"
                return
            self.stop_groups(tagged_children(self.args.kubeconfig))
            if self.policy.state(self.event["host_memory"]["available_gib"]) != "admit":
                self.event["action"] = "deferred_memory"
                return
            self.repair_backend()
            # Release immediately before launching. The supervisor acquires
            # this same lock before changing any application workload.
            fcntl.flock(lock, fcntl.LOCK_UN)
            self.launch()
        finally:
            lock.close()

    def run(self):
        with (self.out / "monitor.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another collection check is running; skipping duplicate", flush=True)
                return 0
            try:
                self.check()
            except Exception as exc:
                self.event.update(action="deferred_error", error=str(exc))
            self.record()
        return 0 if self.event["action"] != "deferred_error" else 1


def build_parser():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project / "data/experiments_tosem_four_copies")
    parser.add_argument("--out-dir", type=Path, default=project / "artifacts/tosem_four_copies/auto_monitor")
    parser.add_argument("--collection-out-dir", type=Path, default=project / "artifacts/tosem_four_copies/guarded")
    parser.add_argument("--kubeconfig", type=Path, default=project / "artifacts/tosem_four_copies/kubeconfig")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=("rule",), default="rule")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), default=2,
                        help="Active copies; four fixed data shards are retained")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stall-minutes", type=int, default=90)
    return parser


if __name__ == "__main__":
    raise SystemExit(CollectionMonitor(build_parser().parse_args()).run())
