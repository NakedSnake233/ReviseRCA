"""Bounded raw-summary audit of evidence lost by the current observation loader.

Uses collector labels only for retrospective coverage measurement, never for
model input or split selection. Does not modify the dataset or train a model.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from dyrca_rca.interaction.observation import epoch


VALID = {"0", "1", "2", "UNSET", "OK", "ERROR", "STATUS_CODE_UNSET", "STATUS_CODE_OK", "STATUS_CODE_ERROR"}


def trace_coverage(path, target, pre, cutoff, max_rows=10000):
    result = dict(scanned=0, eligible=0, status_known=0, root_spans=0,
                  root_status_known=0, root_between_snapshots=0, before_pre=0,
                  root_duration_sum_ms=0., root_duration_max_ms=0., truncated=False)
    if not path.exists():
        return dict(result, missing=True)
    with path.open() as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if i >= max_rows:
                result["truncated"] = True
                break
            result["scanned"] += 1
            try:
                start, duration = int(row["start_time_unix_nano"]) / 1e9, int(row["duration_nano"]) / 1e9
            except (ValueError, KeyError, TypeError):
                continue
            if duration < 0 or start + duration > cutoff:
                continue
            result["eligible"] += 1
            result["before_pre"] += int(start < pre)
            known = row.get("status_code", "").strip().upper() in VALID
            result["status_known"] += int(known)
            if row.get("service") == target:
                result["root_spans"] += 1
                result["root_status_known"] += int(known)
                result["root_between_snapshots"] += int(start >= pre)
                result["root_duration_sum_ms"] += 1000 * duration
                result["root_duration_max_ms"] = max(result["root_duration_max_ms"], 1000 * duration)
    return result


def metric_coverage(path, target, pre, cutoff):
    result = dict(eligible=0, service_attributed=0, root_rows=0, root_between_snapshots=0)
    if not path.exists():
        return dict(result, missing=True)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                timestamp = epoch(row["timestamp"])
                labels = json.loads(row["labels_json"])
                value = float(row["value"])
            except (KeyError, ValueError, TypeError):
                continue
            if timestamp > cutoff or not math.isfinite(value):
                continue
            result["eligible"] += 1
            service = labels.get("destination_workload") or labels.get("service") or labels.get("destination_service_name")
            if str(service).strip().lower() in ("unknown", "none", "null", "", "__unknown__"):
                service = None
            result["service_attributed"] += int(bool(service))
            if service == target:
                result["root_rows"] += 1
                result["root_between_snapshots"] += int(timestamp >= pre)
    return result


def run(args):
    root, out = Path(args.data_root).resolve(), Path(args.out_dir).resolve()
    if root == out or root in out.parents:
        raise ValueError("audit output must be outside raw data")
    out.mkdir(parents=True, exist_ok=False)
    audits = [json.loads(line) for line in Path(args.audit).read_text().splitlines() if line.strip()]
    rows = []
    names = sorted(r["case_id"] for r in audits if r["verified"] and r["scenario_id"] != "C6")
    if args.limit is not None:
        names = names[:args.limit]
    for name in names:
        case = root / name
        target = json.loads((case / "ground_truth.json").read_text())["root_cause_service"]
        pre, cutoff = [epoch(json.loads((case / "topology" / (phase + ".json")).read_text())["timestamp"]) for phase in ("pre", "during")]
        rows.append({"case": name, "family": name[:2].upper(), "trace": trace_coverage(case / "traces/trace_summary.csv", target, pre, cutoff),
                     "latency_p99": metric_coverage(case / "metrics/latency_p99.csv", target, pre, cutoff)})
    (out / "cases.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    groups = defaultdict(list)
    for r in rows:
        groups["T1-T4" if r["family"].startswith("T") else "C1-C5"].append(r)
    summary = {}
    for key, group in groups.items():
        summary[key] = {"n": len(group), "trace_eligible": sum(r["trace"]["eligible"] for r in group),
                        "trace_status_known": sum(r["trace"]["status_known"] for r in group),
                        "cases_with_root_spans": sum(r["trace"]["root_spans"] > 0 for r in group),
                        "cases_with_root_status": sum(r["trace"]["root_status_known"] > 0 for r in group),
                        "cases_with_root_spans_between_snapshots": sum(r["trace"]["root_between_snapshots"] > 0 for r in group),
                        "trace_before_pre": sum(r["trace"]["before_pre"] for r in group),
                        "trace_truncated_cases": sum(r["trace"]["truncated"] for r in group),
                        "p99_service_attributed_rows": sum(r["latency_p99"]["service_attributed"] for r in group),
                        "p99_eligible_rows": sum(r["latency_p99"]["eligible"] for r in group),
                        "cases_with_root_p99_between_snapshots": sum(r["latency_p99"]["root_between_snapshots"] > 0 for r in group)}
    report = {"application": args.system, "data_root": str(root), "seed": args.seed, "planner": args.planner,
              "limit": args.limit, "max_trace_rows": 10000, "experiment_group": "retrospective-input-coverage",
              "summary": summary, "scope": "Collector-target coverage; not evidence of realized fault or causal effect. No model/split selection."}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=["rule"], default="rule")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--system", choices=["boutique", "trainticket"], required=True)
    run(parser.parse_args())
