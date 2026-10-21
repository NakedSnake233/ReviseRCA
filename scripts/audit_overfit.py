#!/usr/bin/env python3
"""Audit split leakage and label shortcuts in a dynamic RCA release.

This is deliberately a read-only audit.  It reads metadata, labels, topology
diffs, and the frozen split, then reports target-service overlap, the
configuration-to-label mapping, and how often the directly changed service
already identifies the root cause.  The latter is a diagnostic upper bound,
not an RCA method, because it uses the audited topology-diff annotation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set


def read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def changed_services(diff: Mapping[str, Any]) -> Set[str]:
    """Extract service names whose effective topology changed pre -> during."""
    comparison = diff.get("comparisons", {}).get("pre_vs_during", {})
    result: Set[str] = set()
    k8s = comparison.get("k8s_layer", {})
    pods = k8s.get("pod_changes", {})
    result.update(str(name) for name in pods.get("count_changes", {}) if name)
    result.update(str(name) for name in pods.get("version_changed", []) if name)
    endpoints = k8s.get("endpoint_changes", {}).get("changed_services", {})
    result.update(str(name) for name in endpoints if name)
    deployments = k8s.get("deployment_changes", {}).get("changed_deployments", {})
    for name in deployments:
        if name:
            text = str(name)
            result.add(text[:-3] if text.endswith("-v2") else text)
    graph = comparison.get("graph_layer", {})
    node_diff = graph.get("node_diff", {})
    for entry in node_diff.get("modified", []):
        if isinstance(entry, dict) and entry.get("service"):
            result.add(str(entry["service"]))
    result.update(str(name) for name in node_diff.get("added", []) if name)
    result.update(str(name) for name in node_diff.get("removed", []) if name)
    return result


def service_from_metadata(case: Path) -> str:
    metadata = read_json(case / "metadata.json")
    target = metadata.get("target_service")
    if target:
        return str(target)
    truth = read_json(case / "ground_truth.json")
    return str(truth["root_cause_service"])


def config_key(name: str) -> str:
    return name.rsplit("-run", 1)[0]


def target_counts(names: Iterable[str], targets: Mapping[str, str]) -> Counter:
    return Counter(targets[name] for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    split_file = Path(args.split).resolve()
    split = read_json(split_file).get("split", read_json(split_file))
    if not isinstance(split, dict):
        raise ValueError("split must contain train/val/test/regression lists")
    parts = {part: [str(name) for name in split.get(part, [])]
             for part in ("train", "val", "test", "regression")}
    names = sorted({name for values in parts.values() for name in values})
    targets: Dict[str, str] = {}
    metadata_targets: Dict[str, str] = {}
    changed: Dict[str, Set[str]] = {}
    tasks: Dict[str, str] = {}
    for name in names:
        case = root / name
        if not (case / "metadata.json").is_file():
            raise FileNotFoundError(case / "metadata.json")
        metadata = read_json(case / "metadata.json")
        if metadata.get("environment", {}).get("namespace") != args.system:
            raise ValueError(f"case {name} is not in system {args.system}")
        metadata_targets[name] = str(metadata.get("target_service", ""))
        targets[name] = service_from_metadata(case)
        tasks[name] = name[:2].upper()
        diff_path = case / "topology_diff.json"
        changed[name] = changed_services(read_json(diff_path)) if diff_path.is_file() else set()

    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test"),
                        ("train", "regression"), ("test", "regression")):
        overlap[f"{left}_x_{right}"] = sorted(set(targets[n] for n in parts[left]) &
                                                set(targets[n] for n in parts[right]))

    config_targets: Dict[str, Set[str]] = defaultdict(set)
    for name in names:
        config_targets[config_key(name)].add(targets[name])
    deterministic = sum(len(values) == 1 for values in config_targets.values())

    changed_hit = {}
    for part, values in parts.items():
        hits = sum(targets[name] in changed[name] for name in values)
        changed_hit[part] = {
            "n": len(values),
            "target_in_changed_service_set": hits,
            "rate": hits / len(values) if values else None,
            "mean_changed_service_count": (
                sum(len(changed[name]) for name in values) / len(values) if values else None),
        }

    per_task = {}
    for task in sorted(set(tasks.values())):
        per_task[task] = {}
        for part, values in parts.items():
            selected = [n for n in values if tasks[n] == task]
            per_task[task][part] = {
                "n": len(selected),
                "targets": dict(sorted(target_counts(selected, targets).items())),
                "changed_hit_rate": (sum(targets[n] in changed[n] for n in selected) / len(selected)
                                      if selected else None),
            }

    report = {
        "schema": "overfit-audit-v1",
        "data_root": str(root),
        "split_file": str(split_file),
        "system": args.system,
        "population": len(names),
        "parts": {part: len(values) for part, values in parts.items()},
        "target_service_counts": {part: dict(sorted(target_counts(values, targets).items()))
                                   for part, values in parts.items()},
        "target_overlap": overlap,
        "configuration_groups": {
            "count": len(config_targets),
            "single_target_groups": deterministic,
            "single_target_rate": deterministic / len(config_targets) if config_targets else None,
            "examples": {key: sorted(values) for key, values in sorted(config_targets.items())[:20]},
        },
        "metadata_target_matches_ground_truth": sum(
            metadata_targets[n] == targets[n] for n in names),
        "changed_service_shortcut": changed_hit,
        "per_task": per_task,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "population": report["population"],
        "parts": report["parts"],
        "train_test_target_overlap": report["target_overlap"]["train_x_test"],
        "config_single_target_rate": report["configuration_groups"]["single_target_rate"],
        "test_changed_service_hit_rate": report["changed_service_shortcut"]["test"]["rate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
