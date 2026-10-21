"""Index isolated collection shards without copying or loading raw telemetry.

A preview never certifies a formal cohort. Formal readiness requires all four
shards to match the frozen task plan and contain complete exported evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from runner.scheduler import BatchConfig, BatchScheduler
from dyrca_rca.collection_safety import resume_exclusions


def _identity(record):
    return {key: record[key] for key in ("scenario_id", "target", "params", "repeat")}


def _plan(path, seed):
    config = BatchConfig.from_yaml(str(path))
    config.seed = seed
    scheduler = BatchScheduler(config)
    scheduler.expand()
    return {task.index: _identity(vars(task)) for task in scheduler.tasks}


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def run(args):
    raw = Path(args.data_root).resolve()
    out = Path(args.out_dir).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("The derived index must be outside raw collection data")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    plan = Path(args.plan).resolve()
    expected = _plan(plan, args.seed)
    exclusion_path = getattr(args, "exclusions", None)
    exclusions = resume_exclusions(exclusion_path)
    records, copies, names = [], [], set()
    for copy_id in range(1, 5):
        namespace = f"{args.system}-c{copy_id}"
        source_root = raw / namespace
        summary_path = source_root / "batch_summary.json"
        expected_indices = {i for i in expected if (i - 1) % 4 == copy_id - 1}
        copy = {"namespace": namespace, "expected": len(expected_indices), "complete": 0,
                "summary": str(summary_path), "formal_ready": False}
        copies.append(copy)
        if not summary_path.exists():
            continue
        content = summary_path.read_bytes()
        report = json.loads(content)
        settings = report.get("settings", {})
        if (settings.get("seed") != args.seed or settings.get("namespace") != namespace or
                settings.get("shard_count") != 4 or settings.get("shard_index") != copy_id - 1):
            raise ValueError(f"Incompatible seed, namespace or shard settings: {namespace}")
        copy["summary_sha256"] = hashlib.sha256(content).hexdigest()
        seen = set()
        for record in report.get("experiments", []):
            index = record["index"]
            if index not in expected_indices or index in seen or _identity(record) != expected[index]:
                raise ValueError(f"Task does not match the frozen plan: {namespace}, index {index}")
            seen.add(index)
            if str(Path(record.get("output_dir") or ".").resolve()) in exclusions or not BatchScheduler._resume_record_is_complete(record):
                continue
            source = Path(record["output_dir"]).resolve()
            if source.parent != source_root.resolve() or source.name != record["experiment_id"]:
                raise ValueError(f"Case path does not belong to its shard: {namespace}, index {index}")
            if source.name in names:
                raise ValueError(f"Duplicate experiment ID across copies: {source.name}")
            names.add(source.name)
            records.append(dict(record, source_namespace=namespace))
            copy["complete"] += 1
        copy["formal_ready"] = (seen == expected_indices and copy["complete"] == len(expected_indices)
                                 and not report.get("failed") and not report.get("pending"))
    formal_ready = all(copy["formal_ready"] for copy in copies) and args.limit is None
    if args.require_complete and not formal_ready:
        raise RuntimeError("Four-copy collection is incomplete; formal experiment input was not created")
    records.sort(key=lambda record: record["index"])
    if args.limit is not None:
        records = records[:args.limit]
    out.mkdir(parents=True, exist_ok=True)
    view = out / "cases"
    view.mkdir(exist_ok=True)
    for record in records:
        source = Path(record["output_dir"])
        link = view / source.name
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        link.symlink_to(source, target_is_directory=True)
    # Refuse a stale view with additional cases from another seed or protocol.
    selected = {Path(record["output_dir"]).name for record in records}
    if {path.name for path in view.iterdir() if path.is_dir() or path.is_symlink()} != selected:
        raise ValueError("The derived view contains stale cases; use a fresh output directory")
    manifest = {"data_root": str(raw), "system": args.system, "seed": args.seed,
                "planner": args.planner, "limit": args.limit, "formal_ready": formal_ready,
                "expected_cases": len(expected), "complete_exported_cases": len(records),
                "verified_population": "pending independent collection audit",
                "plan": str(plan), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                "copies": copies, "raw_data_unchanged": True, "derived_data_root": str(view),
                "resume_exclusions_file": str(exclusion_path) if exclusion_path else None,
                "excluded_cases": len(exclusions)}
    _write(view / "batch_summary.json", {
        "total_experiments": len(expected), "succeeded": len(records), "failed": 0,
        "pending": len(expected) - len(records), "settings": {"seed": args.seed, "namespace": args.system},
        "experiments": records, "formal_ready": formal_ready,
    })
    _write(out / "collection_index.json", manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--system", choices=("boutique", "trainticket"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=("rule",), default="rule")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--exclusions", type=Path, default=Path(__file__).resolve().parents[3] / "artifacts/tosem_four_copies/guarded/resume_exclusions.json")
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps({key: result[key] for key in (
        "system", "expected_cases", "complete_exported_cases", "formal_ready", "derived_data_root")}, sort_keys=True))
