"""Measure a trace-summary proxy for realized target delay in control cases.

This is descriptive evidence, not the independent two-reviewer verification
required by the TOSEM protocol. It never reads traces_raw.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path


MIN_SPANS_PER_PHASE = 3


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def phase(start: float, duration: float, collect: float, inject: float, recover: float) -> str | None:
    end = start + duration
    if collect <= start and end <= inject:
        return "pre"
    if inject <= start and end <= recover:
        return "during"
    return None


def measure_case(case: Path) -> dict:
    metadata = _read(case / "metadata.json")
    target = metadata["target_service"]
    window = metadata["time_window"]
    collect, inject, recover = (_time(window[key]) for key in ("t_collect", "t_inject", "t_recover"))
    values = {"pre": [], "during": []}
    path = case / "traces" / "trace_summary.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("kind") != "SPAN_KIND_CLIENT" or target + "." not in row.get("operation", ""):
                continue
            try:
                start = int(row["start_time_unix_nano"]) / 1e9
                duration = int(row["duration_nano"]) / 1e9
            except (KeyError, ValueError, TypeError):
                continue
            if duration < 0:
                continue
            bucket = phase(start, duration, collect, inject, recover)
            if bucket:
                values[bucket].append(duration * 1000.0)
    adequate = all(len(values[key]) >= MIN_SPANS_PER_PHASE for key in values)
    pre_median = statistics.median(values["pre"]) if values["pre"] else None
    during_median = statistics.median(values["during"]) if values["during"] else None
    return {"case": case.name, "target_service": target, "mode": metadata["control_mode"],
            "is_benign": bool(metadata.get("is_benign")),
            "pre_spans": len(values["pre"]), "during_spans": len(values["during"]),
            "adequate_proxy": adequate, "pre_median_ms": pre_median,
            "during_median_ms": during_median,
            "delta_median_ms": during_median - pre_median if adequate else None,
            "trace_summary_sha256": _sha256(path)}


def run(args: argparse.Namespace) -> dict:
    raw, out = Path(args.data_root).resolve(), Path(args.out_dir).resolve()
    if out == raw or raw in out.parents or args.limit < 1 or args.planner != "rule":
        raise ValueError("use positive limit, rule planner, and output outside raw data")
    summary = _read(Path(args.summary))
    selected = sorted((item for item in summary["experiments"] if item.get("status") == "success"),
                      key=lambda item: item["experiment_id"])[:args.limit]
    rows = []
    for item in selected:
        case = Path(item["output_dir"]).resolve()
        if case.parent != raw:
            raise ValueError("case outside data root")
        metadata = _read(case / "metadata.json")
        if metadata.get("environment", {}).get("namespace") != args.system:
            raise ValueError("case application mismatch")
        rows.append(measure_case(case))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fault_effect_proxy.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    groups = {}
    for key, selected_rows in (("fault", [r for r in rows if not r["is_benign"]]),
                               ("benign", [r for r in rows if r["is_benign"]])):
        adequate = [r for r in selected_rows if r["adequate_proxy"]]
        deltas = [r["delta_median_ms"] for r in adequate]
        groups[key] = {"n": len(selected_rows), "adequate_n": len(adequate),
                       "inadequate_n": len(selected_rows) - len(adequate),
                       "median_delta_ms": statistics.median(deltas) if deltas else None,
                       "delta_over_50ms_n": sum(delta > 50.0 for delta in deltas)}
    result = {"status": "automatic_proxy_only", "system": args.system,
              "data_root": str(raw), "seed": args.seed, "planner": args.planner,
              "limit": args.limit, "minimum_client_spans_per_phase": MIN_SPANS_PER_PHASE,
              "window_rule": "client span entirely within pre or during injection interval",
              "match_rule": "SPAN_KIND_CLIENT operation contains target-service followed by dot",
              "group_summary": groups, "rows_sha256": _sha256(path),
              "limitation": "Absence of target client spans is unresolved; latency change alone does not establish mechanism or healthy decoy behavior."}
    (out / "fault_effect_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--system", required=True, choices=("boutique", "trainticket"))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", default="rule")
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
