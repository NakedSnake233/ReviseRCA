#!/usr/bin/env python3
"""Print progress and verification counters for the two TOSEM batches."""
import argparse
import json
import statistics
from pathlib import Path


def one(path):
    summary = json.loads((Path(path) / "batch_summary.json").read_text())
    rows = summary.get("experiments", [])
    completed = [row for row in rows if row.get("status") == "success" and row.get("duration_seconds")]
    durations = [float(row["duration_seconds"]) for row in completed]
    remaining = sum(row.get("status") == "pending" for row in rows)
    median_duration = statistics.median(durations) if durations else None
    eta_hours = ((remaining * (median_duration + float(summary.get("settings", {}).get("cooldown_seconds", 0)))) / 3600.0
                 if median_duration is not None else None)
    return {
        "batch_id": summary.get("batch_id"),
        "total": summary.get("total_experiments", len(rows)),
        "succeeded": sum(row.get("status") == "success" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "pending": remaining,
        "last_index": max((row.get("index", 0) for row in rows if row.get("status") != "pending"), default=0),
        "verified_exit_zero": sum(row.get("status") == "success" and row.get("verify_exit_code") == 0 for row in rows),
        "median_completed_seconds": round(median_duration, 1) if median_duration is not None else None,
        "estimated_remaining_hours": round(eta_hours, 2) if eta_hours is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boutique", default="data/experiments_tosem_candidate_boutique_20260930")
    parser.add_argument("--trainticket", default="data/experiments_tosem_candidate_trainticket_20260930")
    args = parser.parse_args()
    for label, path in (("boutique", args.boutique), ("trainticket", args.trainticket)):
        summary = Path(path) / "batch_summary.json"
        if summary.is_file():
            print(label, json.dumps(one(path), sort_keys=True))
        else:
            print(label, json.dumps({"status": "not_started"}, sort_keys=True))


if __name__ == "__main__":
    main()
