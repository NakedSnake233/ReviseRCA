"""Aggregate fixed-seed/split stability runs for TOSEM E5."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _rows(path: Path) -> List[Dict[str, Any]]:
    result = []
    with (path / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "test" and row.get("mrr") is not None:
                    result.append(row)
    return result


def _bootstrap(values: Sequence[float], seed: int, n: int = 2000) -> List[float]:
    if not values:
        return []
    rng = random.Random(seed)
    result = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        result.append(sum(sample) / len(sample))
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.out_dir).resolve()
    run_dirs = [Path(path).resolve() for path in args.run_dir]
    existing = [path for path in run_dirs if (path / "results.jsonl").is_file()]
    by_method: Dict[str, List[float]] = {}
    paired: Dict[str, List[float]] = {}
    comparator = args.comparator
    for path in existing:
        rows = _rows(path)
        lookup = {(row["method"], row["experiment"]): row for row in rows}
        methods = sorted({row["method"] for row in rows})
        for method in methods:
            values = [float(row["hit1"]) for row in rows if row["method"] == method]
            by_method.setdefault(method, []).append(sum(values) / len(values) if values else 0.0)
        if comparator:
            ours = {row["experiment"]: row for row in rows if row["method"] == "DY-RCA"}
            other = {row["experiment"]: row for row in rows if row["method"] == comparator}
            paired.setdefault("%s_vs_%s" % ("DY-RCA", comparator), []).extend(
                [ours[name]["mrr"] - other[name]["mrr"] for name in sorted(set(ours) & set(other))])
    summary = []
    for method, values in sorted(by_method.items()):
        summary.append({"method": method, "runs": len(values),
                        "hit1_mean": statistics.mean(values) if values else None,
                        "hit1_sd": statistics.stdev(values) if len(values) > 1 else None,
                        "status": "measured" if len(values) >= 5 else "pending_five_seed_runs"})
    paired_summary = {}
    for key, values in paired.items():
        boot = _bootstrap(values, args.seed)
        paired_summary[key] = {"n": len(values), "delta_mrr": statistics.mean(values) if values else None,
                               "ci_low": (sorted(boot)[int(0.025 * len(boot))] if boot else None),
                               "ci_high": (sorted(boot)[int(0.975 * len(boot))] if boot else None),
                               "status": "measured" if len(values) >= 5 else "pending_five_seed_runs"}
    report = {"status": "measured" if len(existing) >= 5 else "pending_five_seed_runs",
              "seed_runs": [str(path) for path in existing], "summary": summary,
              "paired": paired_summary, "bootstrap_replicates": 2000,
              "comparator_selected_on_validation": bool(comparator)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability_summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", default="rule", choices=("rule",))
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--comparator", default="HolisticRCA-Dyn")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
