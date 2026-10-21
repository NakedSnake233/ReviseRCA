#!/usr/bin/env python3
"""Audit counterfactual-control collection integrity.

The control collection is valid only when the labelled root-cause service is
absent from the pre->during changed-service set, the declared decoy is
observed in that set, and telemetry is complete.  This audit consumes a
scheduler batch summary so interrupted/incomplete directories are excluded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _changed_services(diff: Mapping[str, Any]) -> set[str]:
    comparison = diff.get("comparisons", {}).get("pre_vs_during", {})
    k8s = comparison.get("k8s_layer", {})
    result = set(str(x) for x in k8s.get("pod_changes", {}).get("count_changes", {}) if x)
    result.update(str(x) for x in k8s.get("endpoint_changes", {}).get("changed_services", {}) if x)
    result.update(str(x) for x in k8s.get("deployment_changes", {}).get("changed_deployments", {}) if x)
    graph = comparison.get("graph_layer", {})
    result.update(str(x) for x in graph.get("node_diff", {}).get("added", []) if x)
    result.update(str(x) for x in graph.get("node_diff", {}).get("removed", []) if x)
    for item in graph.get("node_diff", {}).get("modified", []):
        if isinstance(item, dict) and item.get("service"):
            result.add(str(item["service"]))
    return result


def audit(data_root: Path, summary: Path) -> dict[str, Any]:
    batch = _read(summary)
    rows = []
    skipped = Counter()
    for item in batch.get("experiments", []):
        if item.get("status") != "success":
            skipped["scheduler_not_success"] += 1
            continue
        case = Path(item.get("output_dir", ""))
        if not case.is_absolute():
            case = (data_root / case).resolve()
        if not (case / "metadata.json").is_file() or not (case / "topology_diff.json").is_file():
            skipped["missing_metadata_or_diff"] += 1
            continue
        metadata = _read(case / "metadata.json")
        truth = _read(case / "ground_truth.json")
        diff = _read(case / "topology_diff.json")
        verification_path = case / "verification_report.json"
        verifier_result = (_read(verification_path).get("verification_result")
                           if verification_path.is_file() else "missing")
        trace_path = case / "traces" / "traces_raw.json"
        metric_path = case / "metrics" / "request_rate.csv"
        log_path = case / "logs" / "all.jsonl"
        telemetry = all(p.is_file() and p.stat().st_size > 0 for p in (trace_path, metric_path, log_path))
        trace_count = 0
        if trace_path.is_file():
            try:
                trace_count = len(_read(trace_path).get("data", []))
            except (ValueError, OSError, TypeError):
                skipped["invalid_trace_json"] += 1
        target = str(metadata.get("target_service", ""))
        decoy = str(metadata.get("topology_target_service", ""))
        decoys = [str(value) for value in (metadata.get("topology_target_service"),
                                           metadata.get("topology_target_service_2")) if value]
        # A topology change that leaves a decoy with no endpoint cannot be
        # described as independently verified healthy traffic movement.
        endpoint_sets = {}
        for phase in ("pre", "during"):
            snapshot = case / "topology" / (phase + ".json")
            if snapshot.is_file():
                endpoint_sets[phase] = {str(s.get("name")): s.get("endpoint_ips")
                                        for s in _read(snapshot).get("services", []) if isinstance(s, dict)}
        decoy_endpoint_present = (len(endpoint_sets) == 2 and bool(decoys) and
                                  all(isinstance(endpoint_sets[phase].get(service), list)
                                          and bool(endpoint_sets[phase][service])
                                          for phase in ("pre", "during") for service in decoys))
        changed = _changed_services(diff)
        rows.append({
            "case": case.name,
            "scenario_id": str(metadata.get("scenario_id", "")),
            "control_mode": str(metadata.get("control_mode", "")),
            "target_service": target,
            "decoy_service": decoy,
            "root_cause_service": str(truth.get("root_cause_service", "")),
            "is_benign": bool(metadata.get("is_benign", False)),
            "changed_services": sorted(changed),
            "target_in_changed": target in changed if target else False,
            "decoy_in_changed": decoy in changed if decoy else False,
            "verifier_result": verifier_result,
            "verifier_changed": verifier_result == "CHANGED",
            "decoy_endpoints_present_pre_during": decoy_endpoint_present,
            "telemetry_complete": telemetry,
            "trace_count": trace_count,
        })

    def rate(key: str, rows_: list[dict[str, Any]]) -> float | None:
        return (sum(bool(r[key]) for r in rows_) / len(rows_)) if rows_ else None

    normal = [r for r in rows if not r["is_benign"]]
    benign = [r for r in rows if r["is_benign"]]
    by_mode = {}
    for mode in sorted({r["control_mode"] for r in rows}):
        selected = [r for r in rows if r["control_mode"] == mode]
        by_mode[mode] = {
            "n": len(selected),
            "target_in_changed_rate": rate("target_in_changed", selected),
            "decoy_in_changed_rate": rate("decoy_in_changed", selected),
            "telemetry_complete_rate": rate("telemetry_complete", selected),
            "min_trace_count": min((r["trace_count"] for r in selected), default=0),
        }
    return {
        "schema": "counterfactual-control-audit-v1",
        "data_root": str(data_root.resolve()),
        "summary": str(summary.resolve()),
        "population": len(rows),
        "normal_controls": len(normal),
        "benign_controls": len(benign),
        "target_in_changed_service_rate": rate("target_in_changed", normal),
        "decoy_in_changed_service_rate": rate("decoy_in_changed", rows),
        "verifier_changed_rate": rate("verifier_changed", rows),
        "decoy_endpoints_present_pre_during_rate": rate("decoy_endpoints_present_pre_during", rows),
        "strict_operational_n": sum(r["verifier_changed"] and r["decoy_endpoints_present_pre_during"]
                                    for r in rows),
        "independent_fault_and_decoy_health_review": "not_recorded",
        "telemetry_complete_rate": rate("telemetry_complete", rows),
        "min_trace_count": min((r["trace_count"] for r in rows), default=0),
        "modes": by_mode,
        "skipped": dict(skipped),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(Path(args.data_root), Path(args.summary))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "population", "normal_controls", "benign_controls",
        "target_in_changed_service_rate", "decoy_in_changed_service_rate",
        "verifier_changed_rate", "decoy_endpoints_present_pre_during_rate",
        "strict_operational_n", "telemetry_complete_rate", "min_trace_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
