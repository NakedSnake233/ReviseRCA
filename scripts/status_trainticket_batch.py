#!/usr/bin/env python3
"""Print deterministic progress for a Train Ticket collection batch."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/experiments_trainticket_v2_20260928/batch_summary.json")
    args = parser.parse_args()
    path = Path(args.summary)
    data = json.loads(path.read_text(encoding="utf-8"))
    experiments = data.get("experiments", [])
    by_status = Counter(item.get("status", "unknown") for item in experiments)
    by_task = {}
    for task in ("T1", "T2", "T3", "T4"):
        rows = [item for item in experiments if item.get("scenario_id") == task]
        by_task[task] = dict(Counter(item.get("status", "unknown") for item in rows))
    payload = {
        "batch_id": data.get("batch_id"),
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
        "total": len(experiments),
        "status": dict(sorted(by_status.items())),
        "by_task": by_task,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
