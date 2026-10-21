#!/usr/bin/env python3
"""
Batch experiment scheduler for dynamic topology fault dataset.

Usage:
    python -m runner.scheduler --config config/experiment_config.yaml
    python -m runner.scheduler --config config/experiment_config.yaml --dry-run
    python -m runner.scheduler --config config/experiment_config.yaml --scenarios T1,T3
"""
import argparse
import csv
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ── Argument order for each fault script (positional args) ──────
# To add a new fault type, add an entry here.
ARG_ORDERS = {
    "T1": ["target", "cpu_workers", "cpu_load", "during_sec", "post_sec", "pre_sec"],
    "T2": ["target", "kill_count", "during_sec", "post_sec", "pre_sec"],
    "T3": ["target", "variant", "poison_target", "during_sec", "post_sec", "pre_sec"],
    "T4": ["target", "v2_weight", "delay_ms", "abort_pct",
           "during_sec", "post_sec", "pre_sec"],
    # Counterfactual controls: target receives topology-neutral delay while
    # one or two unrelated services receive the dynamic topology change.
    "C1": ["target", "mode", "decoy", "decoy2", "during_sec", "post_sec", "pre_sec", "delay_ms"],
    "C2": ["target", "mode", "decoy", "decoy2", "during_sec", "post_sec", "pre_sec", "delay_ms"],
    "C3": ["target", "mode", "decoy", "decoy2", "during_sec", "post_sec", "pre_sec", "delay_ms"],
    "C4": ["target", "mode", "decoy", "decoy2", "during_sec", "post_sec", "pre_sec", "delay_ms"],
    "C5": ["target", "mode", "decoy", "decoy2", "during_sec", "post_sec", "pre_sec", "delay_ms"],
    "C6": ["target", "mode", "decoy", "decoy2", "during_sec", "post_sec", "pre_sec", "delay_ms"],
}

log = logging.getLogger("scheduler")


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════
@dataclass
class ScenarioConfig:
    id: str
    name: str
    description: str = ""
    script: str = ""
    targets: List[str] = field(default_factory=list)
    param_sets: List[Dict[str, Any]] = field(default_factory=lambda: [{}])
    defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    repeats: int = 3
    seed: int = 42
    randomize: bool = True
    max_retries: int = 2
    cooldown_seconds: int = 120
    python: str = "/home/user/anaconda3/envs/aris/bin/python"
    project_dir: str = "/home/user/dyrca"
    health_check_timeout: int = 180
    require_healthy: bool = False
    allow_hpa_pending: bool = False
    prom_url: str = "http://localhost:9090"
    loki_url: str = "http://localhost:3100"
    tempo_url: str = "http://localhost:3200"
    # Optional namespace override for isolated application copies.  Empty
    # preserves the historical inference from the fault-script filename.
    namespace: str = ""
    memory_min_available_gib: float = 0.0
    stop_file: str = ""
    resume_exclusions_file: str = ""
    data_dir: str = "data/experiments"
    summary_file: str = "data/experiments/batch_summary.json"
    log_file: str = "data/experiments/batch.log"
    scenarios: List[ScenarioConfig] = field(default_factory=list)

    @staticmethod
    def from_yaml(path: str) -> "BatchConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = BatchConfig()
        s = raw.get("settings", {})
        cfg.repeats = s.get("repeats", 3)
        cfg.seed = s.get("seed", cfg.seed)
        cfg.randomize = s.get("randomize", True)
        cfg.max_retries = s.get("max_retries", 2)
        cfg.cooldown_seconds = s.get("cooldown_seconds", 120)
        cfg.python = s.get("python", cfg.python)
        cfg.project_dir = s.get("project_dir", cfg.project_dir)
        cfg.health_check_timeout = s.get("health_check_timeout", 180)
        cfg.require_healthy = bool(s.get("require_healthy", False))
        cfg.allow_hpa_pending = bool(s.get("allow_hpa_pending", False))
        obs = raw.get("observability", {})
        cfg.prom_url = obs.get("prom_url", cfg.prom_url)
        cfg.loki_url = obs.get("loki_url", cfg.loki_url)
        cfg.tempo_url = obs.get("tempo_url", cfg.tempo_url)
        cfg.namespace = str(s.get("namespace", "") or "")
        cfg.memory_min_available_gib = float(s.get("memory_min_available_gib", 0.0))
        cfg.stop_file = str(s.get("stop_file", "") or "")
        cfg.resume_exclusions_file = str(s.get("resume_exclusions_file", "") or "")
        out = raw.get("output", {})
        cfg.data_dir = out.get("data_dir", cfg.data_dir)
        cfg.summary_file = out.get("summary_file", cfg.summary_file)
        cfg.log_file = out.get("log_file", cfg.log_file)
        for sc in raw.get("scenarios", []):
            cfg.scenarios.append(ScenarioConfig(
                id=sc["id"],
                name=sc.get("name", sc["id"]),
                description=sc.get("description", ""),
                script=sc.get("script", ""),
                targets=sc.get("targets", []),
                param_sets=sc.get("param_sets", [{}]),
                defaults=sc.get("defaults", {}),
            ))
        return cfg


# ═══════════════════════════════════════════════════════════════
#  Experiment Task (one unit of work)
# ═══════════════════════════════════════════════════════════════
@dataclass
class ExperimentTask:
    index: int = 0
    scenario_id: str = ""
    scenario_name: str = ""
    script: str = ""
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    repeat: int = 1
    # results (filled after execution)
    status: str = "pending"
    attempts: int = 0
    experiment_id: str = ""
    output_dir: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    fault_exit_code: int = -1
    export_exit_code: int = -1
    verify_exit_code: int = -1
    error: str = ""

    def label(self) -> str:
        p = "_".join(f"{k}{v}" for k, v in self.params.items()
                     if k not in ("during_sec", "post_sec", "pre_sec"))
        return f"{self.scenario_id}_{self.target}_{p}_r{self.repeat}"


# ═══════════════════════════════════════════════════════════════
#  Scheduler
# ═══════════════════════════════════════════════════════════════
class BatchScheduler:
    def __init__(self, config: BatchConfig, filter_scenarios: Optional[List[str]] = None,
                 dry_run: bool = False, skip_export: bool = False,
                 resume_summary: Optional[str] = None,
                 limit: Optional[int] = None,
                 shard_index: int = 0,
                 shard_count: int = 1):
        self.cfg = config
        self.dry_run = dry_run
        self.skip_export = skip_export
        self.resume_summary = resume_summary
        self.limit = limit
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.filter = set(filter_scenarios) if filter_scenarios else None
        self.tasks: List[ExperimentTask] = []
        self.batch_id = datetime.now(tz=timezone.utc).strftime("batch_%Y%m%d_%H%M%S")
        self.batch_start = ""
        self.batch_end = ""
        self.resumed_from = ""

    @staticmethod
    def _csv_has_data(path: Path) -> bool:
        """Check a header and one record without materializing telemetry."""
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            return bool(next(reader, None)) and bool(next(reader, None))

    @staticmethod
    def _resume_record_is_complete(record: Dict[str, Any]) -> bool:
        """Return whether a previously successful task has usable evidence.

        A scheduler success only means that the fault script and exporter
        returned.  Before reusing a result we also require the three topology
        snapshots and non-empty metrics, traces, and logs.  This prevents a
        partially interrupted case (or the historical topology-only cases)
        from being silently treated as complete during a resume.
        """
        if record.get("status") != "success":
            return False
        out = record.get("output_dir")
        if not out:
            return False
        root = Path(out)
        required = [
            root / "metadata.json",
            root / "ground_truth.json",
            root / "topology" / "pre.json",
            root / "topology" / "during.json",
            root / "topology" / "post.json",
            root / "metrics" / "request_rate.csv",
            root / "traces" / "traces_raw.json",
            root / "logs" / "all.jsonl",
            root / "traces" / "trace_summary.csv",
        ]
        if not all(p.is_file() and p.stat().st_size > 0 for p in required):
            return False
        # The summary is the exporter's record of actual spans. Do not load
        # the large raw trace envelope merely to decide whether to resume.
        try:
            return (BatchScheduler._csv_has_data(required[5]) and
                    BatchScheduler._csv_has_data(required[8]))
        except (OSError, ValueError, TypeError):
            return False

    def _apply_resume(self) -> None:
        """Mark complete tasks from a prior summary and leave the rest pending."""
        if not self.resume_summary:
            return
        summary_path = Path(self.resume_summary)
        if not summary_path.is_absolute():
            summary_path = Path(self.cfg.project_dir) / summary_path
        if not summary_path.is_file():
            raise FileNotFoundError(f"Resume summary not found: {summary_path}")
        with open(summary_path, encoding="utf-8") as f:
            previous = json.load(f)
        by_index = {int(item.get("index", -1)): item
                    for item in previous.get("experiments", [])}
        from dyrca_rca.collection_safety import resume_exclusions
        excluded = resume_exclusions(self.cfg.resume_exclusions_file)
        reused = 0
        for task in self.tasks:
            record = by_index.get(task.index)
            if record and str(Path(record.get("output_dir") or ".").resolve()) not in excluded and self._resume_record_is_complete(record):
                task.status = "success"
                task.attempts = int(record.get("attempts", 1) or 1)
                task.experiment_id = record.get("experiment_id", "")
                task.output_dir = record.get("output_dir", "")
                task.start_time = record.get("start_time", "")
                task.end_time = record.get("end_time", "")
                task.duration_seconds = float(record.get("duration_seconds", 0) or 0)
                task.fault_exit_code = int(record.get("fault_exit_code", 0) or 0)
                task.export_exit_code = int(record.get("export_exit_code", 0) or 0)
                task.verify_exit_code = int(record.get("verify_exit_code", 0) or 0)
                reused += 1
        self.resumed_from = str(summary_path)
        log.info("Resuming from %s: reusing %d complete tasks; %d tasks remain",
                 summary_path, reused, len(self.tasks) - reused)

    # ── expand YAML → flat task list ───────────────────────────
    def expand(self):
        idx = 0
        for sc in self.cfg.scenarios:
            if self.filter and sc.id not in self.filter:
                continue
            for target in sc.targets:
                for ps in sc.param_sets:
                    for rep in range(1, self.cfg.repeats + 1):
                        task = ExperimentTask(
                            index=idx,
                            scenario_id=sc.id,
                            scenario_name=sc.name,
                            script=sc.script,
                            target=target,
                            params={**ps},
                            repeat=rep,
                        )
                        self.tasks.append(task)
                        idx += 1
        if self.cfg.randomize:
            random.Random(self.cfg.seed).shuffle(self.tasks)
        # Re-index after shuffle
        for i, t in enumerate(self.tasks):
            t.index = i + 1
        log.info("Expanded %d experiment tasks", len(self.tasks))

    # ── build args for a fault script ──────────────────────────
    def _build_args(self, task: ExperimentTask) -> List[str]:
        order = ARG_ORDERS.get(task.scenario_id)
        if order is None:
            raise ValueError(f"Unknown scenario ID: {task.scenario_id}. "
                             f"Add it to ARG_ORDERS.")
        values = {"target": task.target}
        values.update(task.params)
        values.update(self.cfg.scenarios[
            next(i for i, s in enumerate(self.cfg.scenarios)
                 if s.id == task.scenario_id)
        ].defaults)
        return [str(values.get(arg, "")) for arg in order]

    # ── health check ───────────────────────────────────────────
    def _wait_healthy(self, namespace: str = "boutique") -> bool:
        timeout = self.cfg.health_check_timeout
        waited = 0
        while waited < timeout:
            try:
                r = subprocess.run(
                    ["kubectl", "get", "pods", "-n", namespace,
                     "-l", "app", "--no-headers"],
                    capture_output=True, text=True, timeout=15,
                )
                if r.returncode != 0:
                    detail = (r.stderr or r.stdout).strip().replace("\n", " ")
                    log.warning("  Health check kubectl failed (exit=%d): %s",
                                r.returncode, detail[-300:])
                    time.sleep(10)
                    waited += 10
                    continue
                lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
                # The Boutique loadgenerator is a separate workload and may
                # remain Unknown on an idle kind node.  It is not part of the
                # application evidence collection and should not stall every
                # counterfactual case for the full health-check timeout.
                is_boutique = namespace.startswith("boutique")
                if is_boutique and not self.cfg.require_healthy:
                    lines = [l for l in lines if not l.startswith("loadgenerator-")]
                if not lines:
                    log.warning("  Health check returned no pods; waiting...")
                    time.sleep(10)
                    waited += 10
                    continue
                if self.cfg.require_healthy and self.cfg.allow_hpa_pending:
                    # HPA can legitimately create extra replicas whose long
                    # startup probe is still pending.  The service-level gate
                    # below requires at least one available replica for every
                    # deployment and rejects any non-Running pod; pending
                    # surplus replicas are retained in the pre-snapshot.
                    deploys = self._deployment_health(namespace)
                    not_ready = sum(not self._pod_running(line) for line in lines)
                    service_not_ready = sum(not ready for ready in deploys.values())
                    not_ready += service_not_ready
                elif self.cfg.require_healthy:
                    not_ready = sum(not self._pod_ready(line) for line in lines)
                else:
                    not_ready = sum(1 for l in lines if "Running" not in l)
                min_pods = 8 if is_boutique else 20
                if not_ready == 0 and len(lines) >= min_pods:
                    return True
                log.warning("  Health check (%s): %d pods not Ready, waiting...",
                            namespace, not_ready)
            except Exception as exc:
                log.warning("  Health check failed: %s", exc)
            time.sleep(10)
            waited += 10
        log.error("  Cluster not healthy after %ds.%s", timeout,
                  " Refusing to inject." if self.cfg.require_healthy else " Proceeding anyway.")
        return False

    @staticmethod
    def _pod_ready(line: str) -> bool:
        fields = line.split()
        if len(fields) < 3 or fields[2] != "Running":
            return False
        try:
            ready, total = (int(value) for value in fields[1].split("/"))
        except (ValueError, IndexError):
            return False
        return total > 0 and ready == total

    @staticmethod
    def _pod_running(line: str) -> bool:
        fields = line.split()
        return len(fields) >= 3 and fields[2] == "Running"

    @staticmethod
    def _deployment_health(namespace: str) -> Dict[str, bool]:
        try:
            result = subprocess.run(
                ["kubectl", "get", "deploy", "-n", namespace, "-o", "json"],
                capture_output=True, text=True, timeout=15,
            )
            payload = json.loads(result.stdout or "{}")
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            return {"__kubectl_error__": False}
        health = {}
        for item in payload.get("items", []):
            name = item.get("metadata", {}).get("name", "")
            spec = int(item.get("spec", {}).get("replicas", 0) or 0)
            status = item.get("status", {}) or {}
            ready = int(status.get("readyReplicas", 0) or 0)
            available = int(status.get("availableReplicas", 0) or 0)
            if name and spec > 0:
                health[name] = ready >= 1 and available >= 1
        return health or {"__no_deployments__": False}

    # ── run one fault script ───────────────────────────────────
    def _run_fault_script(self, task: ExperimentTask) -> dict:
        script = os.path.join(self.cfg.project_dir, task.script)
        args = self._build_args(task)
        log.info("  CMD: bash %s %s", script, " ".join(args))

        # Fault scripts historically defaulted to the shared Boutique output
        # directory.  Pass the configured, absolute batch directory explicitly
        # so each system remains isolated and reruns are auditable.
        env = os.environ.copy()
        data_dir = Path(self.cfg.project_dir) / self.cfg.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        env["DATA_DIR"] = str(data_dir.resolve())
        env["DYRCA_RUN_ID"] = f"{self.batch_id}_t{task.index}_a{task.attempts or 1}"
        env["NAMESPACE"] = self.cfg.namespace or (
            "trainticket" if "trainticket" in task.script else "boutique"
        )

        r = subprocess.run(
            ["bash", script] + args,
            capture_output=True, text=True, timeout=900,
            cwd=self.cfg.project_dir,
            env=env,
        )
        # Parse experiment_id and output_dir from script output
        exp_id = ""
        out_dir = ""
        for line in (r.stdout + r.stderr).splitlines():
            m = re.search(r"Experiment ID\s*:\s*(\S+)", line)
            if m:
                exp_id = m.group(1)
            m = re.search(r"(?:Output dir|Data dir)\s*:\s*(\S+)", line)
            if m:
                out_dir = m.group(1)

        return {
            "exit_code": r.returncode,
            "stdout": r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout,
            "stderr": r.stderr[-3000:] if len(r.stderr) > 3000 else r.stderr,
            "experiment_id": exp_id,
            "output_dir": out_dir,
        }

    # ── run topology verification ─────────────────────────────
    def _run_verify(self, exp_dir: str) -> dict:
        if not exp_dir or not Path(exp_dir).exists():
            return {"exit_code": -1, "skipped": True}
        cmd = [
            self.cfg.python, "-m", "verifier.topology_verifier",
            "--experiment-dir", exp_dir,
        ]
        log.info("  VERIFY: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           cwd=self.cfg.project_dir)
        return {"exit_code": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}

    @staticmethod
    def _telemetry_complete(exp_dir: str) -> bool:
        """Guard against recording a topology-only case as a success."""
        root = Path(exp_dir) if exp_dir else None
        if root is None:
            return False
        required = (
            root / "metrics" / "request_rate.csv",
            root / "traces" / "trace_summary.csv",
            root / "traces" / "traces_raw.json",
            root / "logs" / "all.jsonl",
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            return False
        try:
            return (BatchScheduler._csv_has_data(required[0]) and
                    BatchScheduler._csv_has_data(required[1]))
        except (OSError, ValueError, TypeError):
            return False

    # ── run data exporter ──────────────────────────────────────
    def _run_export(self, exp_dir: str) -> dict:
        if self.skip_export or not exp_dir or not Path(exp_dir).exists():
            return {"exit_code": -1, "skipped": True}
        # Auto-detect namespace from experiment metadata
        namespace = "boutique"
        meta_path = Path(exp_dir) / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                namespace = meta.get("environment", {}).get("namespace", "boutique")
            except Exception:
                pass
        cmd = [
            self.cfg.python, "-m", "collector.exporter",
            "--experiment-dir", exp_dir,
            "--prom-url", self.cfg.prom_url,
            "--loki-url", self.cfg.loki_url,
            "--tempo-url", self.cfg.tempo_url,
            "--namespace", namespace,
        ]
        log.info("  EXPORT: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           cwd=self.cfg.project_dir)
        return {"exit_code": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}

    # ── execute one task (with retries) ────────────────────────
    def _collection_admitted(self) -> bool:
        if self.cfg.stop_file and Path(self.cfg.stop_file).exists():
            log.error("Collection stop marker is present: %s", self.cfg.stop_file)
            return False
        if self.cfg.memory_min_available_gib > 0:
            from dyrca_rca.collection_safety import read_memory_state
            try:
                available = read_memory_state()["available_gib"]
            except (OSError, RuntimeError, ValueError) as exc:
                log.error("Memory preflight failed: %s", exc)
                return False
            if available < self.cfg.memory_min_available_gib:
                log.error("Memory reserve reached: available %.1f GiB, required %.1f GiB",
                          available, self.cfg.memory_min_available_gib)
                return False
        return True

    def _execute_task(self, task: ExperimentTask) -> ExperimentTask:
        task.start_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        label = task.label()
        log.info("=" * 60)
        log.info("  [%d/%d] %s", task.index, len(self.tasks), label)
        log.info("=" * 60)

        for attempt in range(1, self.cfg.max_retries + 2):
            task.attempts = attempt
            log.info("  Attempt %d/%d", attempt, self.cfg.max_retries + 1)
            try:
                if not self._collection_admitted():
                    task.error = "collection memory protection stopped admission; no fault injected"
                    task.status = "pending"
                    return task
                # 1. Health check (detect namespace from script name)
                ns = self.cfg.namespace or (
                    "trainticket" if "trainticket" in task.script else "boutique"
                )
                healthy = self._wait_healthy(namespace=ns)
                if self.cfg.require_healthy and not healthy:
                    task.error = "strict cluster health preflight failed; no fault injected"
                    task.status = "pending"
                    return task

                # 2. Run fault script
                fault = self._run_fault_script(task)
                task.fault_exit_code = fault["exit_code"]
                task.experiment_id = fault.get("experiment_id", "")
                task.output_dir = fault.get("output_dir", "")

                if fault["exit_code"] != 0:
                    err = (fault["stderr"].strip() or fault["stdout"].strip())[-500:]
                    log.warning("  Fault script FAILED (exit=%d): %s",
                                fault["exit_code"], err)
                    task.error = err
                    if attempt <= self.cfg.max_retries:
                        log.info("  Retrying in %ds ...", self.cfg.cooldown_seconds)
                        time.sleep(self.cfg.cooldown_seconds)
                        continue
                    task.status = "failed"
                    break

                log.info("  Fault script OK  exp_id=%s", task.experiment_id)

                # 3. Run data exporter
                export = self._run_export(task.output_dir)
                task.export_exit_code = export.get("exit_code", -1)
                if export.get("exit_code", -1) != 0 and not export.get("skipped"):
                    log.warning("  Export had issues (exit=%d)", task.export_exit_code)
                if not self._telemetry_complete(task.output_dir):
                    task.error = "incomplete telemetry evidence after export"
                    log.warning("  Telemetry evidence incomplete; retrying case")
                    if attempt <= self.cfg.max_retries:
                        time.sleep(self.cfg.cooldown_seconds)
                        continue
                    task.status = "failed"
                    break

                # 4. Verify topology changes
                verify = self._run_verify(task.output_dir)
                task.verify_exit_code = verify.get("exit_code", -1)
                if verify.get("exit_code", -1) == 0:
                    log.info("  Topology verification OK")
                elif not verify.get("skipped"):
                    log.warning("  Topology verification had issues (exit=%d)",
                                verify.get("exit_code", -1))

                task.status = "success"
                break

            except subprocess.TimeoutExpired:
                log.error("  TIMEOUT on attempt %d", attempt)
                task.error = "subprocess timeout"
                if attempt <= self.cfg.max_retries:
                    time.sleep(self.cfg.cooldown_seconds)
                    continue
                task.status = "failed"
            except Exception as exc:
                log.error("  EXCEPTION: %s", exc)
                task.error = str(exc)
                if attempt <= self.cfg.max_retries:
                    time.sleep(self.cfg.cooldown_seconds)
                    continue
                task.status = "failed"

        task.end_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = datetime.fromisoformat(task.start_time.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(task.end_time.replace("Z", "+00:00"))
        task.duration_seconds = (t1 - t0).total_seconds()

        status_ch = "+" if task.status == "success" else "x"
        log.info("  [%s] %s  %.0fs  attempts=%d",
                 status_ch, label, task.duration_seconds, task.attempts)
        return task

    # ── run all tasks ──────────────────────────────────────────
    def run(self):
        self.batch_start = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.expand()
        if self.shard_count > 1:
            if not 0 <= self.shard_index < self.shard_count:
                raise ValueError("--shard-index must be in [0, --shard-count)")
            self.tasks = [
                task for task in self.tasks
                if (task.index - 1) % self.shard_count == self.shard_index
            ]
            log.info("Selecting shard %d/%d: %d tasks",
                     self.shard_index + 1, self.shard_count, len(self.tasks))
        if self.limit is not None:
            if self.limit < 1:
                raise ValueError("--limit must be a positive integer")
            # Limit after deterministic expansion/shuffling so a smoke run is
            # a real member of the frozen plan, rather than a new hand-picked
            # case.  The full batch uses a separate output directory.
            self.tasks = self.tasks[:self.limit]
            log.info("Limiting batch to first %d expanded tasks", len(self.tasks))
        self._apply_resume()

        if self.dry_run:
            self._dry_run()
            return

        log.info("Batch %s: %d experiments", self.batch_id, len(self.tasks))

        for task in self.tasks:
            if task.status == "success":
                log.info("  [%d/%d] Reusing completed case %s", task.index,
                         len(self.tasks), task.label())
                continue
            self._execute_task(task)
            # Persist progress after every case so a long collection remains
            # inspectable even if the supervisor is interrupted.
            self._write_summary()
            if task.status == "pending":
                log.error("Stopping batch after preflight refusal: %s", task.error)
                break
            if task.index < len(self.tasks):
                log.info("  Cooldown %ds ...", self.cfg.cooldown_seconds)
                time.sleep(self.cfg.cooldown_seconds)

        self.batch_end = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._write_summary()

    # ── dry run (just print plan) ──────────────────────────────
    def _dry_run(self):
        print(f"\nDRY RUN — {len(self.tasks)} experiments planned:\n")
        print(f"{'#':>4}  {'Scenario':<5} {'Target':<28} {'Params':<30} {'Rep':>3}")
        print("-" * 80)
        for t in self.tasks:
            ps = ", ".join(f"{k}={v}" for k, v in t.params.items()
                           if k not in ("during_sec", "post_sec", "pre_sec"))
            print(f"{t.index:>4}  {t.scenario_id:<5} {t.target:<28} {ps:<30} {t.repeat:>3}")
        print(f"\nTotal: {len(self.tasks)} experiments")
        total_min = len(self.tasks) * (30 + 90 + 120 + 60 + self.cfg.cooldown_seconds) / 60
        print(f"Estimated time: ~{total_min:.0f} minutes\n")

    # ── write batch summary ────────────────────────────────────
    def _write_summary(self):
        succeeded = [t for t in self.tasks if t.status == "success"]
        failed = [t for t in self.tasks if t.status == "failed"]
        pending = [t for t in self.tasks if t.status not in ("success", "failed")]

        summary = {
            "batch_id": self.batch_id,
            "resumed_from": self.resumed_from,
            "start_time": self.batch_start,
            "end_time": self.batch_end,
            "total_experiments": len(self.tasks),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "pending": len(pending),
            "settings": {
                "repeats": self.cfg.repeats,
                "seed": self.cfg.seed,
                "randomize": self.cfg.randomize,
                "max_retries": self.cfg.max_retries,
                "cooldown_seconds": self.cfg.cooldown_seconds,
                "require_healthy": self.cfg.require_healthy,
                "allow_hpa_pending": self.cfg.allow_hpa_pending,
                "namespace": self.cfg.namespace,
                "memory_min_available_gib": self.cfg.memory_min_available_gib,
                "stop_file": self.cfg.stop_file,
                "resume_exclusions_file": self.cfg.resume_exclusions_file,
                "shard_index": self.shard_index,
                "shard_count": self.shard_count,
            },
            "experiments": [
                {
                    "index": t.index,
                    "experiment_id": t.experiment_id,
                    "scenario_id": t.scenario_id,
                    "scenario_name": t.scenario_name,
                    "target": t.target,
                    "params": t.params,
                    "repeat": t.repeat,
                    "status": t.status,
                    "attempts": t.attempts,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                    "duration_seconds": t.duration_seconds,
                    "output_dir": t.output_dir,
                    "fault_exit_code": t.fault_exit_code,
                    "export_exit_code": t.export_exit_code,
                    "verify_exit_code": t.verify_exit_code,
                    "error": t.error[:500] if t.error else "",
                }
                for t in self.tasks
            ],
        }

        out_path = Path(self.cfg.project_dir) / self.cfg.summary_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        log.info("=" * 60)
        if self.batch_end:
            log.info("  BATCH COMPLETE")
        else:
            log.info("  BATCH PROGRESS")
        log.info("  Succeeded: %d / %d", len(succeeded), len(self.tasks))
        log.info("  Failed:    %d / %d", len(failed), len(self.tasks))
        log.info("  Pending:   %d / %d", len(pending), len(self.tasks))
        log.info("  Summary:   %s", out_path)
        log.info("=" * 60)

        # Print failed experiments
        if failed and self.batch_end:
            log.warning("Failed experiments:")
            for t in failed:
                log.warning("  %s  error: %s", t.label(), t.error[:200])


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment scheduler for dynamic topology fault dataset.")
    parser.add_argument("--config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario IDs to run (e.g. T1,T3). Default: all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan without executing")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip data export step (faster, for testing)")
    parser.add_argument("--resume-summary", default=None,
                        help="Reuse complete cases from a prior batch summary and run pending cases")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Override repeats from config")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N expanded tasks (for bounded smoke runs)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based shard index for parallel isolated copies")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Number of deterministic shards")
    args = parser.parse_args()

    cfg = BatchConfig.from_yaml(args.config)
    if args.repeats is not None:
        cfg.repeats = args.repeats

    # Setup logging → console + file
    log_dir = Path(cfg.project_dir) / Path(cfg.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.project_dir) / cfg.log_file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )

    filter_ids = args.scenarios.split(",") if args.scenarios else None
    scheduler = BatchScheduler(cfg, filter_scenarios=filter_ids,
                               dry_run=args.dry_run, skip_export=args.skip_export,
                               resume_summary=args.resume_summary, limit=args.limit,
                               shard_index=args.shard_index, shard_count=args.shard_count)
    scheduler.run()


if __name__ == "__main__":
    main()
