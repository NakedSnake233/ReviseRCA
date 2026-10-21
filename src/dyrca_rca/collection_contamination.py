"""Conservatively quarantine exports overlapping an unclean failed T1 run."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def overlapping_failed_stress(reports, duration_seconds=180):
    failures, successes = [], {}
    for report in reports:
        namespace = report.get("settings", {}).get("namespace")
        if report.get("settings", {}).get("seed") != 42:
            continue
        for row in report.get("experiments", []):
            output = row.get("output_dir")
            if not output:
                continue
            if row.get("status") == "success":
                successes[str(Path(output).resolve())] = (namespace, row)
            if row.get("scenario_id") == "T1" and row.get("status") == "failed" and \
                    row.get("fault_exit_code", 0) != 0 and not (Path(output) / "fault_cleanup.done").exists():
                if row.get("start_time") and row.get("end_time"):
                    failures.append((namespace, row, _time(row["start_time"]),
                                     _time(row["end_time"]) + duration_seconds))
    excluded = []
    for output, (namespace, row) in sorted(successes.items()):
        metadata = Path(output) / "metadata.json"
        if metadata.exists():
            window = json.loads(metadata.read_text()).get("time_window", {})
        else:
            window = {}
        start = window.get("t_collect") or row.get("start_time")
        end = window.get("t_end") or row.get("end_time")
        if not start or not end:
            continue
        causes = [failure["output_dir"] for ns, failure, low, high in failures
                  if ns == namespace and _time(start) <= high and _time(end) >= low]
        if causes:
            excluded.append({"output_dir": output, "namespace": namespace,
                             "reason": "observation window overlaps failed T1 without confirmed cleanup",
                             "conservative_duration_seconds": duration_seconds,
                             "failed_stress_cases": sorted(set(causes))})
    return excluded


def update_exclusions(data_root, collection_out, archive_roots=()):
    reports = []
    sources = list(Path(data_root).glob("*-c[1-4]/batch_summary.json"))
    for root in archive_roots:
        sources += list(Path(root).glob("*-before.json"))
    for path in sources:
        report = json.loads(path.read_text())
        if "experiments" in report:
            reports.append(report)
    path = Path(collection_out) / "resume_exclusions.json"
    previous = json.loads(path.read_text()).get("records", []) if path.exists() else []
    records = {row["output_dir"]: row for row in previous}
    records.update({row["output_dir"]: row for row in overlapping_failed_stress(reports)})
    rows = [records[key] for key in sorted(records)]
    result = {"excluded_output_dirs": sorted(records), "records": rows,
              "scope": "resume and formal indexing; raw exports remain unchanged",
              "policy": "conservative time overlap; recollect rather than relabel"}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
    return result
