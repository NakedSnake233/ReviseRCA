"""Build auditable TOSEM table inputs from a verified dynamic-data batch.

The collector is deliberately label-blind while reading the diagnostic packet:
it only summarizes files already sealed by the scheduler.  Ground truth is
opened only when the audit records are written.  Missing downstream stages
remain ``null`` in the table template; no failed or unavailable measurement is
silently converted to zero.

This module is a bounded, CPU-only runner and is intended to be run after E0
collection.  It supports the repository experiment contract flags so smoke
runs and full runs use the same code path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TABLES = {
    "tab:new_systems": "E0 deployment and observation inventory",
    "tab:new_cohorts": "E0 intended-to-verified flow",
    "tab:scenario_design": "E0 verified incident strata",
    "tab:fidelity_results": "E1 representation fidelity",
    "tab:feature_coverage": "E1 feature-family coverage",
    "tab:adapter_boundaries": "E2 adapter manifest",
    "tab:new_ranking": "E2 nine-method ranking",
    "tab:shortcut_results": "E3 change-cue strata",
    "tab:retrained_ablation": "E4 retrained ablations",
    "tab:stability_results": "E5 stability and grouped intervals",
    "tab:revision_fidelity": "E6 revision execution and fidelity",
    "tab:revision_ranking": "E6 revision ranking outcomes",
    "tab:recorded_walkthrough": "E7 recorded walkthrough",
    "tab:conditions": "E8 agent conditions",
    "tab:human_results": "E8 simulated SRE-agent outcomes",
    "tab:human_process": "E8 simulated SRE-agent process",
    "tab:runtime_results": "E9 runtime latency",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _percent(n: int, d: int) -> Optional[float]:
    return round(100.0 * n / d, 4) if d else None


def _median(values: Sequence[float]) -> Optional[float]:
    return round(float(statistics.median(values)), 4) if values else None


def _iqr(values: Sequence[float]) -> Optional[List[float]]:
    if not values:
        return None
    values = sorted(values)
    q1 = values[max(0, int(math.floor((len(values) - 1) * 0.25)))]
    q3 = values[min(len(values) - 1, int(math.ceil((len(values) - 1) * 0.75)))]
    return [round(float(q1), 4), round(float(q3), 4)]


def _case_dirs(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").is_file())


def _summary_records(root: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    summary = _load_json(root / "batch_summary.json", {}) or {}
    for item in summary.get("experiments", []):
        output = item.get("output_dir")
        if output:
            result[str(Path(output).resolve())] = item
    return result


def _required_evidence(case: Path) -> Tuple[bool, List[str]]:
    required = [
        "metadata.json", "ground_truth.json", "topology/pre.json",
        "topology/during.json", "topology/post.json", "metrics/request_rate.csv",
        "traces/trace_summary.csv", "traces/traces_raw.json", "logs/all.jsonl",
        "topology_diff.json", "verification_report.json",
    ]
    missing = [name for name in required if not (case / name).is_file() or (case / name).stat().st_size == 0]
    return not missing, missing


def _snapshot(case: Path, phase: str) -> Dict[str, Any]:
    return _load_json(case / "topology" / (phase + ".json"), {}) or {}


def _services(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("name")): row for row in snapshot.get("services", []) if row.get("name")}


def _pods(snapshot: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in snapshot.get("pods", []):
        service = row.get("service")
        if service:
            grouped[str(service)].append(row)
    return grouped


def _csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            yield from csv.DictReader(stream)
    except (OSError, UnicodeError):
        return


def _audit_case(case: Path, scheduler_record: Dict[str, Any]) -> Dict[str, Any]:
    meta = _load_json(case / "metadata.json", {}) or {}
    truth = _load_json(case / "ground_truth.json", {}) or {}
    verification = _load_json(case / "verification_report.json", {}) or {}
    pre, during, post = (_snapshot(case, phase) for phase in ("pre", "during", "post"))
    pre_s, during_s, post_s = (_services(snapshot) for snapshot in (pre, during, post))
    pre_p, during_p, post_p = (_pods(snapshot) for snapshot in (pre, during, post))
    service_ids = sorted(set(pre_s) | set(during_s) | set(post_s))
    endpoint_pairs = []
    for service in service_ids:
        before = pre_s.get(service, {}).get("endpoint_ips")
        now = during_s.get(service, {}).get("endpoint_ips")
        after = post_s.get(service, {}).get("endpoint_ips")
        if isinstance(before, list) and isinstance(now, list):
            endpoint_pairs.append({
                "service": service,
                "before": sorted(before), "during": sorted(now),
                "post": sorted(after) if isinstance(after, list) else None,
                "changed": sorted(before) != sorted(now),
                "became_empty": bool(before) and not now,
            })
    pod_services = sorted(set(pre_p) & set(during_p))
    version_pairs = []
    pod_changes = 0
    changed_services = {row["service"] for row in endpoint_pairs if row["changed"]}
    for service in pod_services:
        before = sorted((p.get("name"), p.get("version")) for p in pre_p[service])
        now = sorted((p.get("name"), p.get("version")) for p in during_p[service])
        if before != now:
            pod_changes += 1
            changed_services.add(service)
        # A rollout may expose a versioned deployment (`service-v2`) while the
        # stable pod/deployment still reports an unknown image version.  Keep
        # the logical service comparison and use any explicit version labels
        # available in either pod or deployment records.
        before_versions = {str(v) for _, v in before if v not in (None, "", "unknown")}
        during_versions = {str(v) for _, v in now if v not in (None, "", "unknown")}
        for snapshot, target in ((pre, before_versions), (during, during_versions)):
            for deployment in snapshot.get("deployments", []) if isinstance(snapshot, dict) else []:
                name = str(deployment.get("name", ""))
                if name == service or name.startswith(service + "-"):
                    version = deployment.get("version")
                    if version not in (None, "", "unknown"):
                        target.add(str(version))
        if before_versions or during_versions:
            version_pairs.append({"service": service, "before": sorted(before_versions),
                                  "during": sorted(during_versions),
                                  "changed": before_versions != during_versions})
            if before_versions != during_versions:
                changed_services.add(service)
    req_rows = list(_csv_rows(case / "metrics" / "request_rate.csv"))
    trace_row_count = sum(1 for _ in _csv_rows(case / "traces" / "trace_summary.csv"))
    request_values = []
    for row in req_rows:
        for key in ("rate", "request_rate", "value", "rps"):
            try:
                if row.get(key) not in (None, ""):
                    request_values.append(float(row[key]))
                    break
            except (TypeError, ValueError):
                continue
    ok, missing = _required_evidence(case)
    scenario_id = str(meta.get("scenario_id", "")).upper()
    # Dynamic controls/T4 legitimately create transient rollout workloads.  A
    # dynamic workload already present in the pre-snapshot, or left behind in
    # the post-snapshot, indicates cross-run contamination and is excluded from
    # the verified population rather than silently treated as evidence.
    def dynamic_names(snapshot):
        names = []
        for row in snapshot.get("deployments", []) if isinstance(snapshot, dict) else []:
            name = str(row.get("name", ""))
            if "-cf-" in name or name.endswith("-v2") or "-blackhole" in name:
                names.append(name)
        return sorted(names)
    pre_dynamic, during_dynamic, post_dynamic = map(dynamic_names, (pre, during, post))
    contamination = []
    if pre_dynamic:
        contamination.append("dynamic_workload_present_pre")
    if post_dynamic:
        contamination.append("dynamic_workload_left_post")
    if scenario_id not in {"C1", "C2", "C3", "C4", "C5", "C6", "T4"} and during_dynamic:
        contamination.append("unexpected_dynamic_workload_during")
    verified = bool(
        ok and scheduler_record.get("status") == "success"
        and scheduler_record.get("verify_exit_code") == 0
        and verification.get("verification_result") in ("CHANGED", "PARTIALLY_CHANGED")
        and not contamination
    )
    return {
        "case_id": case.name,
        "path": str(case.resolve()),
        "application": re.sub(r"-c[1-4]$", "", str(meta.get("environment", {}).get("namespace", "unknown"))),
        "source_namespace": meta.get("environment", {}).get("namespace", "unknown"),
        "scenario_id": meta.get("scenario_id", "unknown"),
        "target_service": meta.get("target_service", truth.get("root_cause_service", "")),
        "fault_type": meta.get("fault_type", ""),
        "topology_change_type": meta.get("topology_change_type", truth.get("topology_change_type", "")),
        "scheduler_status": scheduler_record.get("status", "unlisted"),
        "verified": verified,
        "missing_evidence": missing,
        "verification_result": verification.get("verification_result"),
        "verification_confidence": verification.get("confidence"),
        "recovery_verified": verification.get("recovery_verified"),
        "contamination_flags": contamination,
        "dynamic_workloads": {"pre": pre_dynamic, "during": during_dynamic, "post": post_dynamic},
        "healthy_changed_services": sorted({str(change.get("entity")) for change in truth.get("topology_changes", [])
                                             if isinstance(change, dict) and change.get("decoy") and change.get("entity")}),
        "services": len(service_ids),
        "nodes_pre": len(pre.get("pods", [])),
        "nodes_during": len(during.get("pods", [])),
        "nodes_post": len(post.get("pods", [])),
        "endpoint_pairs": len(endpoint_pairs),
        "endpoint_active": sum(row["changed"] for row in endpoint_pairs),
        "endpoint_became_empty": sum(row["became_empty"] for row in endpoint_pairs),
        "pod_service_pairs": len(pod_services),
        "pod_active": pod_changes,
        "version_pairs": len(version_pairs),
        "version_active": sum(row["changed"] for row in version_pairs),
        "changed_services": sorted(changed_services),
        "request_rows": len(req_rows),
        "trace_rows": trace_row_count,
        "trace_has_data": trace_row_count > 0,
        "request_rate_median": _median(request_values),
        "version_coverage_percent": _percent(len(version_pairs), len(pod_services)),
        "endpoint_audit": endpoint_pairs,
        "version_audit": version_pairs,
        "scenario_flags": {
            "realized_relevant_change": scenario_id in {"T1", "T2", "T3", "T4"},
            "no_major_topology_change": scenario_id in {"C1", "C2", "C3", "C4", "C5"} and not changed_services,
            "fault_plus_healthy_change": scenario_id in {"C1", "C2", "C3", "C4", "C5"} and bool(changed_services),
            "multiple_changed_services": len(changed_services) > 1,
            "counterfactual_control": bool(meta.get("counterfactual", False)),
            "benign_dynamic_change": bool(meta.get("is_benign", False)),
        },
        "hashes": {name: _sha(case / name) for name in (
            "metadata.json", "ground_truth.json", "topology/pre.json",
            "topology/during.json", "topology/post.json")
            if (case / name).is_file()},
    }


def _table_template() -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    # E0/E1 keys are populated by this runner. The remaining fields are
    # intentionally explicit nulls until their frozen downstream artifacts
    # exist, which prevents fabricated zeros from entering the manuscript.
    for key in TABLES:
        values[key] = {"status": "pending", "value": None, "source_artifact": None,
                       "denominator": None}
    return values


def _build_summary(audits: Sequence[Dict[str, Any]], application: str) -> Dict[str, Any]:
    intended = len(audits)
    verified = [row for row in audits if row["verified"]]
    usable = [row for row in audits if row["scheduler_status"] == "success"]
    scenario = Counter(row["scenario_id"] for row in verified)
    flags = {key: sum(bool(row["scenario_flags"].get(key)) for row in verified)
             for key in ("realized_relevant_change", "no_major_topology_change",
                         "counterfactual_control", "benign_dynamic_change")}
    return {
        "application": application,
        "intended_n": intended,
        "scheduler_success_n": len(usable),
        "verified_n": len(verified),
        "unverified_n": intended - len(verified),
        "missing_evidence_n": sum(bool(row["missing_evidence"]) for row in audits),
        "contamination_n": sum(bool(row.get("contamination_flags")) for row in audits),
        "scenario_counts": dict(sorted(scenario.items())),
        "scenario_flags": flags,
        "services_median": _median([row["services"] for row in verified]),
        "nodes_median_pre": _median([row["nodes_pre"] for row in verified]),
        "nodes_median_during": _median([row["nodes_during"] for row in verified]),
        "eligible_endpoint_comparisons": sum(row["endpoint_pairs"] for row in verified),
        "active_endpoint_cases": sum(row["endpoint_active"] > 0 for row in verified),
        "eligible_pod_comparisons": sum(row["pod_service_pairs"] for row in verified),
        "active_pod_cases": sum(row["pod_active"] > 0 for row in verified),
        "eligible_version_comparisons": sum(row["version_pairs"] for row in verified),
        "active_version_cases": sum(row["version_active"] > 0 for row in verified),
        "trace_rows_median": _median([row["trace_rows"] for row in verified]),
        "trace_complete_percent": _percent(sum(row["trace_has_data"] for row in verified), len(verified)),
        "request_rate_median": _median([row["request_rate_median"] for row in verified
                                         if row.get("request_rate_median") is not None]),
        "version_coverage_percent": _percent(
            sum(row.get("version_pairs", 0) for row in verified),
            sum(row.get("pod_service_pairs", 0) for row in verified)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.data_root).resolve()
    out = Path(args.out_dir).resolve()
    if root == out or root in out.parents:
        raise ValueError("out-dir must be outside raw data-root")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    summary_records = _summary_records(root)
    cases = _case_dirs(root)
    if args.limit is not None:
        cases = cases[:args.limit]
    audits = []
    for case in cases:
        record = summary_records.get(str(case.resolve()), {})
        audits.append(_audit_case(case, record))
    out.mkdir(parents=True, exist_ok=True)
    (out / "collection_audit.jsonl").write_text(
        "".join(_canonical(row) + "\n" for row in audits), encoding="utf-8")
    apps = sorted(set(row["application"] for row in audits))
    application_summaries = [_build_summary([r for r in audits if r["application"] == app], app)
                             for app in apps]
    (out / "e0_e1_summary.json").write_text(_canonical({"applications": application_summaries}) + "\n",
                                             encoding="utf-8")
    template = _table_template()
    for app_summary in application_summaries:
        app = app_summary["application"]
        prefix = "ob" if app == "boutique" else "tt"
        for field in ("intended_n", "scheduler_success_n", "verified_n", "unverified_n",
                      "missing_evidence_n", "contamination_n", "services_median", "nodes_median_pre",
                      "nodes_median_during", "trace_rows_median", "trace_complete_percent",
                      "request_rate_median", "version_coverage_percent"):
            template.setdefault("fields", {})[f"{prefix}.{field}"] = {
                "status": "measured", "value": app_summary[field],
                "source_artifact": str((out / "e0_e1_summary.json").resolve()),
                "denominator": app_summary["verified_n"],
            }
        for family, nkey, akey in (("endpoint", "eligible_endpoint_comparisons", "active_endpoint_cases"),
                                   ("pod", "eligible_pod_comparisons", "active_pod_cases"),
                                   ("version", "eligible_version_comparisons", "active_version_cases")):
            template.setdefault("fields", {})[f"{prefix}.eligible_{family}_comparisons"] = {
                "status": "measured", "value": app_summary[nkey],
                "source_artifact": str((out / "e0_e1_summary.json").resolve()),
                "denominator": app_summary["verified_n"],
            }
            template["fields"][f"{prefix}.active_{family}_cases"] = {
                "status": "measured", "value": app_summary[akey],
                "source_artifact": str((out / "e0_e1_summary.json").resolve()),
                "denominator": app_summary["verified_n"],
            }
    template["protocol"] = {"seed": args.seed, "planner": args.planner,
                             "data_root": str(root), "limit": args.limit,
                             "generated_utc": datetime.now(timezone.utc).isoformat(),
                             "agent_model": "gpt-5.6-luna", "human_role": "simulated SRE agent"}
    (out / "results_template.json").write_text(_canonical(template) + "\n", encoding="utf-8")
    status = {
        "tables": {key: {"status": "measured_e0" if key in {
            "tab:new_systems", "tab:new_cohorts", "tab:scenario_design",
            "tab:feature_coverage"} else ("pending_independent_e1_reference"
            if key == "tab:fidelity_results" else "pending_downstream_artifact"),
            "description": description} for key, description in TABLES.items()},
        "applications": application_summaries,
        "case_count": len(audits),
        "raw_data_unchanged": True,
    }
    (out / "paper_backfill_status.json").write_text(_canonical(status) + "\n", encoding="utf-8")
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", default="rule", choices=("rule", "llm"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
