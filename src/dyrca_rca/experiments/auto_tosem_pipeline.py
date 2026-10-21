"""Start each frozen application's experiments as soon as its four shards finish.

This watcher is independent of the collection supervisor. One bounded CPU job
runs at a time; immutable stage receipts permit recovery without scoring a
partial cohort or mixing a failed attempt into the manuscript.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dyrca_rca.collection_safety import GIB, read_memory_state, resume_exclusions
from dyrca_rca.experiments.index_four_copy_collection import _identity, _plan

PROJECT = Path(__file__).resolve().parents[3]
MODULE = "dyrca_rca.experiments.auto_tosem_pipeline"
REVISED_PAPER = PROJECT / "ref/TOSEM_R2_revised_source_20260930"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def process_owned(pid, out, worker=False):
    if not isinstance(pid, int) or pid < 2:
        return False
    try:
        root = Path("/proc") / str(pid)
        argv = root.joinpath("cmdline").read_bytes().split(b"\0")
        state = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0]
        env = root.joinpath("environ").read_bytes().split(b"\0")
        return (MODULE.encode() in argv and state != "Z" and (b"--execute-stage" in argv) == worker and
                ("DYRCA_AUTO_EXPERIMENT_ROOT=" + str(out.resolve())).encode() in env)
    except (OSError, ValueError):
        return False


def collection_readiness(raw, app, expected, seed, exclusions=()):
    complete, pending = 0, len(expected)
    all_shards_finished = True
    for copy in range(1, 5):
        namespace = f"{app}-c{copy}"
        path = Path(raw) / namespace / "batch_summary.json"
        if not path.is_file():
            all_shards_finished = False
            continue
        report = json.loads(path.read_text())
        settings = report.get("settings", {})
        if (settings.get("seed") != seed or settings.get("namespace") != namespace or
                settings.get("shard_index") != copy - 1 or settings.get("shard_count") != 4):
            raise ValueError("Collection shard does not match the frozen seed/namespace")
        indices = {i for i in expected if (i - 1) % 4 == copy - 1}
        seen = set()
        for record in report.get("experiments", []):
            i = record["index"]
            if i in seen or i not in indices or _identity(record) != expected[i]:
                raise ValueError("Collection record does not match its frozen task")
            seen.add(i)
            if record.get("status") != "success":
                continue
            source = Path(record.get("output_dir") or ".").resolve()
            if (source.parent != (Path(raw) / namespace).resolve() or
                    source.name != record.get("experiment_id")):
                raise ValueError("Collection output escaped its namespace")
            if (record.get("status") == "success" and
                    all(record.get(k) == 0 for k in ("fault_exit_code", "export_exit_code", "verify_exit_code")) and
                    str(source) not in exclusions):
                complete += 1
                pending -= 1
        if seen != indices or report.get("pending") or report.get("failed"):
            all_shards_finished = False
    return {"ready": all_shards_finished and pending == 0, "complete": complete, "remaining": pending,
            "reason": "full exported-evidence check required before training"}


def compile_manuscript(paper_dir, out, revised=False):
    """Publish a PDF only after its own preview checks and compilation succeed."""
    paper_dir, out = Path(paper_dir), Path(out)
    command = (["sh", "build.sh", "--preview"] if revised else
               ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    result = subprocess.run(command, cwd=paper_dir, timeout=600)
    if result.returncode:
        raise RuntimeError("LaTeX compilation failed; see the retained stage log")
    import shutil
    shutil.copy2(paper_dir / "main.pdf", out / "main.pdf")


def execute_stage(spec_path, cap_gib):
    # Apply before importing torch/numpy, including the shared CPU thread budget.
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (int(cap_gib * GIB), int(cap_gib * GIB)))
    os.nice(10)
    spec = json.loads(Path(spec_path).read_text())
    operation = spec.get("operation", "module")
    if operation == "module":
        if not spec["module"].startswith("dyrca_rca.experiments."):
            raise ValueError("Unexpected stage module")
        sys.argv = [spec["module"], *spec["args"]]
        runpy.run_module(spec["module"], run_name="__main__")
    elif operation == "agent":
        from dyrca_rca.experiments.simulate_sre_agent import run
        release, prepared, out = (Path(spec[k]) for k in ("release", "prepared", "out"))
        protocol = json.loads((prepared / "protocol.json").read_text())
        split = json.loads((release / "split.json").read_text())["split"]
        allowed = set(split["test"])
        setup = out / "setup"
        setup.joinpath("collection").mkdir(parents=True)
        for name in ("observation_manifest.jsonl", "cases.jsonl", "labels.jsonl"):
            with (prepared / "collection" / name).open() as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            rows = [row for row in rows if row["experiment"] in allowed]
            (setup / "collection" / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        protocol.update(checkpoint=str(release / "temporal_v3.pt"), split=split,
                        study_population="held-out test only", human_participants=0)
        write_json(setup / "protocol.json", protocol)
        args = SimpleNamespace(data_root=protocol["data_root"], pilot_dir=str(setup),
                               out_dir=str(out / "agent"), system=protocol["system"],
                               seed=spec["seed"], planner="llm", limit=min(36, len(allowed)),
                               max_rows=10000, oracle_prob=.5, defect_prob=.5,
                               allow_external_evidence=True)
        run(args)
    elif operation == "publish":
        from dyrca_rca.experiments.publish_tosem_results import run
        run(SimpleNamespace(**spec["parameters"]))
    elif operation in ("compile", "compile_revised"):
        compile_manuscript(spec["paper_dir"], spec["out"], revised=operation == "compile_revised")
    elif operation == "state_tests":
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_interaction.py",
                                 "tests/test_temporal_harness.py", "-q",
                                 "--junitxml=" + str(Path(spec["out"]) / "state_tests.xml")], cwd=PROJECT)
        if result.returncode:
            raise RuntimeError("State-contract tests failed; no result table was marked complete")
    else:
        raise ValueError("Unknown stage operation")


class AutoPipeline:
    def __init__(self, args):
        self.args = args
        self.out = args.out_dir.resolve()
        raw = args.data_root.resolve()
        if self.out == raw or raw in self.out.parents:
            raise ValueError("Experiment output must be outside raw collection data")
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = False
        self.child = None
        self.systems = ("boutique", "trainticket") if args.system == "all" else (args.system,)
        self.expected = {app: _plan(PROJECT / f"config/tosem_candidate_{app}_20260930.yaml", args.seed)
                         for app in self.systems}
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=f"{PROJECT / 'artifacts/tosem_four_copies/runtime_deps'}:{PROJECT / 'src'}:{PROJECT}", CUDA_VISIBLE_DEVICES="",
                        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                        NUMEXPR_NUM_THREADS="1", DYRCA_AUTO_EXPERIMENT_ROOT=str(self.out))
        self.status = {"status": "waiting_collection", "seed": args.seed, "planner": args.planner,
                       "limit": args.limit, "data_root": str(args.data_root.resolve()),
                       "watcher_pid": os.getpid(), "poll_seconds": args.poll_seconds,
                       "memory_cap_gib": args.memory_cap_gib, "admit_gib": args.admit_gib,
                       "stop_below_gib": args.stop_below_gib, "completed_applications": []}

    def save(self):
        self.status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.out / "status.json", self.status)

    def credential_env(self):
        env = dict(self.env)
        if not env.get("DYRCA_LLM_API_KEY"):
            path = self.args.llm_config
            if not path.is_file():
                raise RuntimeError("The previously authorized Luna credential configuration is unavailable")
            info = path.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ValueError("Luna credential file must be owned by this user with mode 0600")
            config = json.loads(path.read_text())
            for key in ("DYRCA_LLM_API_KEY", "DYRCA_LLM_BASE_URL", "DYRCA_LLM_MODEL"):
                value = config.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError("Invalid Luna credential configuration")
                env[key] = value
        env.setdefault("DYRCA_LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
        env.setdefault("DYRCA_LLM_MODEL", "gpt-5.6-luna")
        return env

    def receipt(self, app, stage):
        path = self.out / app / stage / "receipt.json"
        if not path.is_file():
            return None
        receipt = json.loads(path.read_text())
        root = Path(receipt["out"])
        if receipt.get("seed") != self.args.seed or root.parent != path.parent:
            raise ValueError("Unexpected stage receipt")
        for name, checksum in receipt["outputs"].items():
            if not (root / name).is_file() or sha(root / name) != checksum:
                raise ValueError("Completed stage artifact no longer matches its receipt")
        return root

    def terminate_child(self):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait(timeout=15)

    def recover_child(self):
        path = self.out / "child.json"
        if path.is_file():
            child = json.loads(path.read_text())
            pid = child.get("pid")
            if process_owned(pid, self.out, worker=True):
                os.killpg(pid, signal.SIGTERM)
                for _ in range(30):
                    if not process_owned(pid, self.out, worker=True):
                        break
                    time.sleep(.5)
                if process_owned(pid, self.out, worker=True):
                    os.killpg(pid, signal.SIGKILL)

    def stage(self, app, name, factory, outputs, agent=False):
        previous = self.receipt(app, name)
        if previous:
            return previous
        memory = read_memory_state()
        if memory["available_gib"] < self.args.admit_gib:
            raise RuntimeError("Experiment admission deferred by host memory reserve")
        parent = self.out / app / name
        parent.mkdir(parents=True, exist_ok=True)
        attempt = 1 + len(list(parent.glob("attempt-*")))
        root = parent / f"attempt-{attempt:04d}"
        root.mkdir()
        spec = factory(root)
        write_json(root / "stage.json", spec)
        self.status.update(status="running_experiments", application=app, stage=name,
                           stage_out=str(root), memory_available_gib=memory["available_gib"])
        self.save()
        env = self.credential_env() if agent else dict(self.env)
        with (root / "run.log").open("a") as log:
            self.child = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute-stage",
                                           str(root / "stage.json"), "--memory-cap-gib", str(self.args.memory_cap_gib)],
                                          cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(self.out / "child.json", {"pid": self.child.pid, "app": app, "stage": name})
            try:
                while self.child.poll() is None:
                    if self.stop:
                        raise InterruptedError("Experiment watcher was stopped")
                    if read_memory_state()["available_gib"] < self.args.stop_below_gib:
                        raise RuntimeError("Experiment stopped to preserve collection memory")
                    time.sleep(5)
                if self.child.returncode:
                    raise RuntimeError(f"Stage {name} exited {self.child.returncode}; retained log: {root / 'run.log'}")
                hashes = {output: sha(root / output) for output in outputs}
                write_json(parent / "receipt.json", {"app": app, "stage": name, "seed": self.args.seed,
                                                     "out": str(root), "outputs": hashes,
                                                     "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                return root
            finally:
                self.terminate_child()
                self.child = None

    def module(self, app, name, module, parameters, outputs):
        return self.stage(app, name, lambda out: {"module": "dyrca_rca.experiments." + module,
                                                "args": ["--out-dir", str(out), *map(str, parameters)]}, outputs)

    def publish(self, app, name, artifacts):
        params = {"data_root": str(self.args.data_root.resolve()), "out_dir": None,
                  "paper_dir": str(self.args.paper_dir.resolve()), "system": app,
                  "seed": self.args.seed, "planner": self.args.planner, "limit": None,
                  "artifacts": {key: str(value) for key, value in artifacts.items()}}
        def factory(out):
            return {"operation": "publish", "parameters": dict(params, out_dir=str(out))}
        return self.stage(app, name, factory, ["backfill_manifest.json", "table_status.csv", "results.tex"])

    def compile(self, app, name):
        saved = self.out / "paper_before_backfill.pdf"
        source = self.args.paper_dir / "main.pdf"
        if source.is_file() and not saved.exists():
            import shutil
            shutil.copy2(source, saved)
        return self.stage(app, name, lambda out: {"operation": "compile", "paper_dir": str(self.args.paper_dir),
                                                "out": str(out)}, ["main.pdf"])

    def run_application(self, app):
        artifacts = {}
        common = ["--seed", self.args.seed, "--planner", "rule"]
        index = self.module(app, "01_index", "index_four_copy_collection",
                            ["--data-root", self.args.data_root, "--system", app,
                             "--plan", PROJECT / f"config/tosem_candidate_{app}_20260930.yaml",
                             "--exclusions", self.args.exclusions, "--require-complete", *common], ["collection_index.json"])
        data = index / "cases"
        artifacts["index"] = index
        audit = self.module(app, "02_audit", "run_tosem_backfill", ["--data-root", data, *common],
                            ["collection_audit.jsonl", "e0_e1_summary.json"])
        artifacts["audit"] = audit
        prepared = self.module(app, "03_prepare", "prepare_tosem_release",
                               ["--data-root", data, "--audit", audit / "collection_audit.jsonl",
                                "--system", app, "--dataset-id", f"tosem-four-copies-20260930-{app}",
                                "--max-rows", 10000, *common], ["candidate_release.json", "protocol.json"])
        artifacts["prepared"] = prepared
        self.publish(app, "04_publish_cohort", artifacts)
        count = json.loads((prepared / "candidate_release.json").read_text())["verified_cases"]
        release_data = prepared / "verified_data"
        runs = []
        combinations = [(42, 42), (42, 123), (42, 456), (42, 789), (42, 2024), (137, 42), (2026, 42)]
        for split_seed, train_seed in combinations:
            suffix = f"s{split_seed}_t{train_seed}"
            release = self.module(app, "release_" + suffix, "optimize_harness",
                                  ["--data-root", release_data, "--limit", count, "--max-rows", 10000,
                                   "--seed", train_seed, "--split-seed", split_seed, "--planner", "rule",
                                   "--split-policy", "config_group", "--system", app,
                                   "--dataset-id", f"tosem-four-copies-20260930-{app}", "--tasks", "t1", "t2", "t3", "t4",
                                   "--old-checkpoint", PROJECT / "artifacts/interactive_route_training/model.pt",
                                   "--regression-selection", prepared / "regression_selection.json", "--include-validation"],
                                  ["report.json", "split.json", "temporal_v3.pt", "predictions.jsonl"])
            ranking = self.module(app, "ranking_" + suffix, "run_dynamic_baselines",
                                  ["--data-root", release_data, "--release-root", release, "--limit", count,
                                   "--seed", train_seed, "--planner", "rule", "--system", app,
                                   "--dataset-id", f"tosem-four-copies-20260930-{app}", "--include-validation",
                                   "--controls-root", data, "--controls-audit", audit / "collection_audit.jsonl"],
                                  ["results.jsonl", "predictions.jsonl", "report.json"])
            runs.append({"split_seed": split_seed, "training_seed": train_seed,
                         "results": str(ranking / "results.jsonl"), "release": str(release)})
            if (split_seed, train_seed) == (42, 42):
                artifacts.update(release=release, ranking=ranking)
                self.publish(app, "05_publish_ranking", artifacts)
                self.compile(app, "06_compile_ranking")
            if split_seed == 42:
                ablation = self.module(app, "ablation_t" + str(train_seed), "run_tosem_ablation",
                                       ["--data-root", release_data, "--release-root", release,
                                        "--seed", train_seed, "--planner", "rule"],
                                       ["ablation_summary.json", "ablation_results.jsonl"])
            if (split_seed, train_seed) == (42, 42):
                artifacts["ablation"] = ablation
                latency = self.module(app, "07_latency", "run_tosem_latency",
                                      ["--data-root", release_data, "--release-root", release,
                                       "--system", app, "--dataset-id", f"tosem-four-copies-20260930-{app}",
                                       "--limit", 10, "--max-rows", 10000, "--warmup", 5, "--repeats", 20, *common],
                                      ["latency_summary.json", "latency_samples.jsonl"])
                artifacts["latency"] = latency
                agent = self.stage(app, "08_luna", lambda out: {"operation": "agent", "out": str(out),
                                   "release": str(release), "prepared": str(prepared), "seed": self.args.seed},
                                   ["agent/summary.json", "agent/scored.jsonl", "agent/agent_predictions.jsonl"], agent=True)
                artifacts["agent"] = agent / "agent"
                self.publish(app, "09_publish_agent", artifacts)
        manifest = self.out / app / "repetitions.json"
        write_json(manifest, {"runs": runs})
        stability = self.module(app, "10_stability", "measure_tosem_stability",
                                ["--data-root", release_data, "--run-manifest", manifest, *common],
                                ["stability_summary.json"])
        artifacts["stability"] = stability
        contract = self.stage(app, "10_contract_tests", lambda out: {"operation": "state_tests", "out": str(out)},
                              ["state_tests.xml"])
        artifacts["contract"] = contract
        downstream = self.module(app, "11_downstream", "run_tosem_downstream",
                                 ["--data-root", data, "--audit", audit / "collection_audit.jsonl",
                                  "--contract-xml", contract / "state_tests.xml",
                                  *[item for key in ("ranking", "ablation", "stability", "latency", "agent")
                                    for item in ("--" + key + "-dir", artifacts[key])], *common],
                                 ["downstream_manifest.json", "ranking_summary.json", "shortcut_summary.json"])
        artifacts["downstream"] = downstream
        self.publish(app, "12_publish_final", artifacts)
        self.compile(app, "13_compile_final")
        if (REVISED_PAPER / "pending_values.tex").is_file():
            self.module(app, "14_revised_backfill", "backfill_revised_source",
                        ["--data-root", self.args.data_root, "--experiment-root", self.out,
                         "--paper-dir", REVISED_PAPER, *common], ["result_mapping.json"])
            self.stage(app, "15_compile_revised", lambda out: {
                "operation": "compile_revised", "paper_dir": str(REVISED_PAPER), "out": str(out)},
                ["main.pdf"])
            self.module(app, "16_revision_completion", "complete_revised_experiments",
                        ["--data-root", self.args.data_root, "--experiment-root", self.out,
                         "--system", app, *common], ["completion_summary.json"])
            self.module(app, "17_coverage_completion", "audit_revised_coverage",
                        ["--data-root", self.args.data_root, "--experiment-root", self.out,
                         "--system", app, *common], ["coverage_summary.json", "control_source_records.json"])
            self.module(app, "18_supplemental_backfill", "backfill_revised_source",
                        ["--data-root", self.args.data_root, "--experiment-root", self.out,
                         "--paper-dir", REVISED_PAPER, *common], ["result_mapping.json"])
            self.stage(app, "19_compile_supplemental", lambda out: {
                "operation": "compile_revised", "paper_dir": str(REVISED_PAPER), "out": str(out)},
                ["main.pdf"])
        self.status["completed_applications"].append(app)
        self.status.update(status="experiments_and_available_tables_complete", application=app,
                           remaining_table_requirements="See backfill_manifest.json; missing evidence is never filled as zero")
        self.save()

    def run(self):
        with (self.out / "watcher.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            (self.out / "watcher.pid").write_text(str(os.getpid()) + "\n")
            self.recover_child()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: setattr(self, "stop", True))
            while not self.stop:
                delay = self.args.poll_seconds
                try:
                    exclusions = resume_exclusions(self.args.exclusions)
                    progress = {app: collection_readiness(self.args.data_root, app, self.expected[app], self.args.seed, exclusions)
                                for app in self.systems}
                    self.status["progress"] = progress
                    for app in self.systems:
                        if app in self.status["completed_applications"]:
                            continue
                        if progress[app]["ready"] and self.args.limit is None:
                            self.run_application(app)
                    if all(app in self.status["completed_applications"] for app in self.systems):
                        return 0
                    self.status["status"] = "waiting_collection" if self.args.limit is None else "bounded_check_only"
                    self.save()
                except Exception as exc:
                    self.status.update(status="retry_wait", error=str(exc), retry_seconds=1200)
                    self.save()
                    delay = 1200
                if self.args.once:
                    return 0
                for _ in range(delay):
                    if self.stop:
                        break
                    time.sleep(1)
            self.terminate_child()
            self.status["status"] = "stopped_resumable"
            self.save()
        return 0


def ensure_running(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "watcher.pid"
    pid = int(path.read_text()) if path.is_file() else 0
    if process_owned(pid, args.out_dir):
        print(json.dumps({"action": "watcher_running", "pid": pid}))
        return 0
    status_path = args.out_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        wanted = {"boutique", "trainticket"} if args.system == "all" else {args.system}
        if (status.get("seed") == args.seed and status.get("status") == "experiments_and_available_tables_complete"
                and wanted <= set(status.get("completed_applications", []))):
            print(json.dumps({"action": "pipeline_completed"}))
            return 0
    env = os.environ.copy()
    env["DYRCA_AUTO_EXPERIMENT_ROOT"] = str(args.out_dir.resolve())
    command = [sys.executable, "-u", "-m", MODULE, "--data-root", str(args.data_root),
               "--out-dir", str(args.out_dir), "--seed", str(args.seed), "--planner", args.planner,
               "--system", args.system, "--paper-dir", str(args.paper_dir), "--llm-config", str(args.llm_config),
               "--exclusions", str(args.exclusions), "--poll-seconds", str(args.poll_seconds),
               "--memory-cap-gib", str(args.memory_cap_gib), "--admit-gib", str(args.admit_gib),
               "--stop-below-gib", str(args.stop_below_gib)]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    with (args.out_dir / "watcher.log").open("a") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"action": "watcher_started", "pid": process.pid}))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/experiments_tosem_four_copies")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "artifacts/tosem_four_copies/experiments_20260930")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=("rule",), default="rule")
    parser.add_argument("--limit", type=int, help="Bounded readiness check; never publishes formal results")
    parser.add_argument("--system", choices=("boutique", "trainticket", "all"), default="all")
    parser.add_argument("--exclusions", type=Path, default=PROJECT / "artifacts/tosem_four_copies/guarded/resume_exclusions.json")
    parser.add_argument("--paper-dir", type=Path, default=PROJECT / "paper")
    parser.add_argument("--llm-config", type=Path, default=PROJECT / "artifacts/tosem_four_copies/private/luna.json")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--memory-cap-gib", type=float, default=8)
    parser.add_argument("--admit-gib", type=float, default=18)
    parser.add_argument("--stop-below-gib", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ensure-running", action="store_true")
    parser.add_argument("--execute-stage", type=Path, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.execute_stage:
        execute_stage(args.execute_stage, args.memory_cap_gib)
    elif args.ensure_running:
        raise SystemExit(ensure_running(args))
    else:
        if not (args.poll_seconds >= 1 and args.memory_cap_gib > 0 and
                args.stop_below_gib >= 10 and args.admit_gib >= args.stop_below_gib + args.memory_cap_gib):
            raise SystemExit("Invalid memory reserve or polling configuration")
        raise SystemExit(AutoPipeline(args).run())
