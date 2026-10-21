"""Prepare a verified, leakage-audited temporal-v3 release for TOSEM runs.

The collector output is deliberately kept immutable.  This command creates a
new directory containing symlinks to verified cases, a deterministic grouped
split, and a regression manifest.  The actual tensor cache/checkpoints are
then built by :mod:`optimize_harness` using the same observation-only loader
as the existing temporal-v3 experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from dyrca_rca.experiments.optimize_harness import grouped_split
from dyrca_rca.interaction.observation import anonymous_id


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _verified_names(audit_path: Path, application: str) -> List[str]:
    names = []
    for row in _read_jsonl(audit_path):
        if row.get("application") != application or not row.get("verified"):
            continue
        # Ranking/learning tables are defined on T1--T4 incidents. Controls
        # are retained in the raw cohort and consumed by the shortcut audit.
        if str(row.get("scenario_id", "")).upper() in {"T1", "T2", "T3", "T4"}:
            names.append(str(row["case_id"]))
    return sorted(set(names))


def _target_services(data_root: Path, names: Iterable[str]) -> Dict[str, str]:
    targets = {}
    for name in names:
        metadata = json.loads((data_root / name / "metadata.json").read_text(encoding="utf-8"))
        target = metadata.get("target_service")
        if not target:
            raise ValueError(f"missing target_service in {name}")
        targets[name] = str(target)
    return targets


def _link_verified(source_root: Path, destination_root: Path, names: Iterable[str]) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = (source_root / name).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        link = destination_root / name
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == source:
                continue
            raise FileExistsError(link)
        link.symlink_to(source, target_is_directory=True)


def run(args: argparse.Namespace) -> Dict:
    raw = Path(args.data_root).resolve()
    audit = Path(args.audit).resolve()
    out = Path(args.out_dir).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("release output must be outside raw data")
    names = _verified_names(audit, args.system)
    if args.limit is not None:
        if args.limit < 4 or args.limit > len(names):
            raise ValueError("limit must retain at least one case per split and not exceed verified cases")
        names = names[: args.limit]
    if len(names) < 20:
        raise ValueError("verified cohort is too small for the four-task grouped split")
    targets = _target_services(raw, names)
    # Controls are deliberately absent from this ranking release.  Their
    # independent labels remain available in the collection audit for E3.
    split = grouped_split(names, regression=[], seed=args.seed, tasks=("t1", "t2", "t3", "t4"))
    if set().union(*(set(split[p]) for p in ("train", "val", "test"))) != set(names):
        raise AssertionError("split does not cover the verified cohort")

    linked_root = out / "verified_data"
    _link_verified(raw, linked_root, names)
    config = {
        "release_version": "tosem-temporal-v3-20260930",
        "application": args.system,
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "planner": args.planner,
        "max_rows": args.max_rows,
        "data_root": str(linked_root),
        "source_data_root": str(raw),
        "audit": str(audit),
        "verified_cases": len(names),
        "split": split,
        "target_services": targets,
        "controls_excluded_from_ranking": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_release.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    # optimize_harness owns the canonical release split.json schema; keep this
    # pre-cache planning record separate so reruns cannot be mistaken for a
    # materialized release.
    (out / "candidate_split.json").write_text(json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (out / "regression_selection.json").write_text("[]\n", encoding="utf-8")
    records = []
    cases = []
    labels = []
    for name in names:
        partition = next(p for p in ("train", "val", "test") if name in split[p])
        incident = anonymous_id(args.dataset_id, name)
        records.append({"case_id": incident, "experiment": name,
                        "data_path": str(linked_root / name), "split": partition,
                        "system": args.system})
        metadata = json.loads((raw / name / "metadata.json").read_text(encoding="utf-8"))
        truth = json.loads((raw / name / "ground_truth.json").read_text(encoding="utf-8"))
        cases.append({"case_id": incident, "experiment": name,
                      "scenario_id": metadata.get("scenario_id"), "is_benign": False,
                      "data_path": str(linked_root / name)})
        labels.append({"case_id": incident, "root_cause_service": truth.get("root_cause_service"),
                       "experiment": name})
    (out / "verified_cases.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    # These manifests mirror the simulator's pilot contract, but explicitly
    # identify the corpus as verified TOSEM data and retain no participant data.
    (out / "collection").mkdir(exist_ok=True)
    (out / "collection" / "observation_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    (out / "collection" / "cases.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cases), encoding="utf-8")
    (out / "collection" / "labels.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    protocol = {
        "status": "verified_tosem_release",
        "system": args.system,
        "data_root": str(linked_root),
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "planner": args.planner,
        "max_rows": args.max_rows,
        "split": split,
        "case_count": len(names),
        "human_role": "simulated_sre_agent",
        "agent_model": "gpt-5.6-luna",
        "label_blind_input": True,
        "cutoff_rule": "During-snapshot timestamp; pre/during topology and completed trace-summary spans only. Snapshot time approximates availability; collector arrival times are not recorded.",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"application": args.system, "verified_cases": len(names), "split_sizes": {p: len(split[p]) for p in split}, "out_dir": str(out), "linked_root": str(linked_root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=("rule",), default="rule")
    parser.add_argument("--system", choices=("boutique", "trainticket"), required=True)
    parser.add_argument("--dataset-id", default="tosem-20260930")
    parser.add_argument("--max-rows", type=int, default=10000)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
