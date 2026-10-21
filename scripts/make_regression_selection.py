#!/usr/bin/env python3
"""Create a deterministic, task-balanced regression selection.

The selection is label-blind: it uses only metadata, topology/telemetry
completeness, and stable directory ordering.  The default 34/33/33 allocation
matches the existing Online Boutique release protocol (T2/T3/T4); T1 remains
in the grouped train/validation/test partitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def complete(path: Path) -> bool:
    required = [
        path / "metadata.json", path / "ground_truth.json",
        path / "topology" / "pre.json", path / "topology" / "during.json",
        path / "topology" / "post.json", path / "metrics" / "request_rate.csv",
        path / "traces" / "trace_summary.csv", path / "traces" / "traces_raw.json",
        path / "logs" / "all.jsonl",
    ]
    if not all(p.is_file() and p.stat().st_size > 0 for p in required):
        return False
    try:
        raw = json.loads((path / "traces" / "traces_raw.json").read_text())
        return bool(raw.get("data"))
    except (OSError, ValueError, AttributeError):
        return False


def select(root: Path, namespace: str, counts: dict[str, int]) -> list[str]:
    eligible = {task: [] for task in counts}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not complete(path):
            continue
        try:
            metadata = json.loads((path / "metadata.json").read_text())
        except (OSError, ValueError):
            continue
        if metadata.get("environment", {}).get("namespace") != namespace:
            continue
        task = path.name[:2].lower()
        if task in eligible:
            eligible[task].append(path.name)
    selected = []
    for task, count in counts.items():
        if len(eligible[task]) < count:
            raise ValueError(f"{task}: need {count} complete cases, found {len(eligible[task])}")
        selected.extend(eligible[task][:count])
    return sorted(selected)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--namespace", default="trainticket")
    p.add_argument("--t2", type=int, default=34)
    p.add_argument("--t3", type=int, default=33)
    p.add_argument("--t4", type=int, default=33)
    args = p.parse_args()
    values = select(Path(args.root), args.namespace, {"t2": args.t2, "t3": args.t3, "t4": args.t4})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"count": len(values), "by_task": {task: sum(n[:2] == task for n in values)
                                                           for task in ("t2", "t3", "t4")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
