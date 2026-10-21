#!/usr/bin/env python3
"""Audit TrainTicket collection completeness without changing raw data.

The historical TrainTicket batch was written into the shared experiment root
and used short random directory suffixes.  This audit separates cases by the
namespace recorded in ``metadata.json`` and reports both topology and
observability completeness, so a topology-only directory cannot be mistaken
for an RCA-ready incident.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED = {"T1": 100, "T2": 75, "T3": 100, "T4": 125}


def _rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def _json_count(path: Path, key: str | None = None) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if key is not None:
        value = value.get(key, []) if isinstance(value, dict) else []
    return len(value) if isinstance(value, list) else 0


def audit(root: Path, namespace: str = "trainticket", summary: Path | None = None) -> dict[str, Any]:
    allowed: set[Path] | None = None
    if summary is not None:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        allowed = {
            Path(item["output_dir"]).resolve()
            for item in payload.get("experiments", [])
            if item.get("status") == "success" and item.get("output_dir")
        }
    cases: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if not path.is_dir() or not (path / "metadata.json").is_file():
            continue
        if allowed is not None and path.resolve() not in allowed:
            continue
        try:
            metadata = json.loads((path / "metadata.json").read_text())
        except (OSError, ValueError):
            continue
        if metadata.get("environment", {}).get("namespace") != namespace:
            continue
        scenario = str(metadata.get("scenario_id", "")).upper()
        telemetry = {
            "metric_rows": sum(_rows(p) for p in (path / "metrics").glob("*.csv")),
            "trace_rows": _rows(path / "traces" / "trace_summary.csv"),
            "trace_count": _json_count(path / "traces" / "traces_raw.json", "data"),
            "log_rows": sum(1 for p in (path / "logs").glob("*.jsonl")
                             for _ in p.open(encoding="utf-8", errors="replace"))
            if (path / "logs").exists() else 0,
        }
        topology = {
            phase: (path / "topology" / f"{phase}.json").is_file()
            for phase in ("pre", "during", "post")
        }
        verification = "missing"
        verification_path = path / "verification_report.json"
        if verification_path.is_file():
            try:
                verification = str(json.loads(verification_path.read_text()).get(
                    "verification_result", "unknown"))
            except (OSError, ValueError, TypeError):
                verification = "invalid"
        ground_truth = {}
        try:
            ground_truth = json.loads((path / "ground_truth.json").read_text())
        except (OSError, ValueError, TypeError):
            pass
        cases.append({
            "experiment_id": path.name,
            "path": str(path),
            "scenario": scenario,
            "topology": topology,
            "telemetry": telemetry,
            "verification_result": verification,
            "has_root_cause_service": bool(ground_truth.get("root_cause_service")),
            "topology_complete": all(topology.values()),
            "telemetry_complete": all(telemetry[k] > 0 for k in
                                        ("metric_rows", "trace_rows", "trace_count", "log_rows")),
        })

    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted(EXPECTED):
        group = [c for c in cases if c["scenario"] == task]
        by_task[task] = {
            "expected": EXPECTED[task],
            "observed": len(group),
            "missing": max(0, EXPECTED[task] - len(group)),
            "topology_complete": sum(c["topology_complete"] for c in group),
            "telemetry_complete": sum(c["telemetry_complete"] for c in group),
            "empty_metrics": sum(c["telemetry"]["metric_rows"] == 0 for c in group),
            "empty_traces": sum(c["telemetry"]["trace_rows"] == 0 for c in group),
            "empty_logs": sum(c["telemetry"]["log_rows"] == 0 for c in group),
            "verification_results": dict(Counter(c["verification_result"] for c in group)),
            "missing_root_cause_service": sum(not c["has_root_cause_service"] for c in group),
        }
    return {
        "schema_version": "trainticket-collection-audit-v1",
        "root": str(root.resolve()),
        "namespace": namespace,
        "expected_total": sum(EXPECTED.values()),
        "observed_total": len(cases),
        "cases_with_complete_topology": sum(c["topology_complete"] for c in cases),
        "cases_with_complete_telemetry": sum(c["telemetry_complete"] for c in cases),
        "by_task": by_task,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/experiments")
    parser.add_argument("--out", required=True)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--summary", default=None,
                        help="Optional scheduler summary; include only successful output directories")
    args = parser.parse_args()
    report = audit(Path(args.root), args.namespace,
                   Path(args.summary) if args.summary else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("expected_total", "observed_total",
                       "cases_with_complete_topology",
                       "cases_with_complete_telemetry")}, ensure_ascii=False))
    for task, stats in report["by_task"].items():
        print(task, json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
