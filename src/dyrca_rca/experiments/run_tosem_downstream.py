"""Aggregate auditable downstream TOSEM result artifacts.

This stage never invents a value for an unavailable experiment.  It consumes
the immutable E0 audit and the label-free/frozen predictions produced by the
baseline runner, then writes table-shaped JSON/CSV artifacts.  Missing
controls, independent references, Luna traces, or latency samples remain
explicitly pending so a manuscript cannot silently receive a zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


METHODS = ["NSigma-Dyn", "BARO-Dyn", "CIRCA-Dyn", "CloudRanger-Dyn",
           "Eadro-Dyn", "HolisticRCA-Dyn", "Change-only", "Node-only", "DY-RCA"]


def _load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 6) if values else None


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _audit_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {row["case_id"]: row for row in _jsonl(path)}


def _ranking(ranking_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if ranking_dir is None:
        return []
    return _jsonl(ranking_dir / "results.jsonl")


def _ranking_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "test":
            continue
        by_method[str(row.get("method"))].append(row)
    result = []
    for method in METHODS:
        selected = by_method.get(method, [])
        scored = [row for row in selected if row.get("mrr") is not None]
        result.append({"method": method, "n": len(selected), "scored_n": len(scored),
                       "hit1": _mean([float(row["hit1"]) for row in scored]),
                       "hit3": _mean([float(row["hit3"]) for row in scored]),
                       "mrr": _mean([float(row["mrr"]) for row in scored])})
    return {"status": "measured" if rows else "pending_ranking_artifact", "methods": result,
            "method_count": sum(bool(by_method.get(method)) for method in METHODS)}


def _shortcut_summary(rows: Sequence[Mapping[str, Any]], audits: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    wanted = {"Change-only", "Node-only", "DY-RCA"}
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method") not in wanted or row.get("split") not in {"test", "control"}:
            continue
        audit = audits.get(str(row.get("experiment")), {})
        flags = audit.get("scenario_flags", {})
        if flags.get("no_major_topology_change"):
            groups["no_major_change"].append(row)
        if flags.get("realized_relevant_change"):
            groups["fault_with_relevant_change"].append(row)
        if flags.get("fault_plus_healthy_change"):
            groups["fault_plus_benign"].append(row)
        if flags.get("multiple_changed_services"):
            groups["multiple_changed_services"].append(row)
        # A matched pair is only eligible when E0 has frozen an explicit pair
        # ID.  Never infer it from similar scenario names.
        if audit.get("matched_pair_id"):
            groups["matched_change_different_outcome"].append(row)
    result = []
    for subset, subset_rows in sorted(groups.items()):
        for method in sorted(wanted):
            method_rows = [r for r in subset_rows if r.get("method") == method]
            vals = [r for r in method_rows if r.get("mrr") is not None]
            controls = [r for r in method_rows if audits.get(str(r.get("experiment")), {}).get("scenario_flags", {}).get("benign_dynamic_change")]
            benign_eligible = []
            for row in controls:
                healthy = set(audits.get(str(row.get("experiment")), {}).get("healthy_changed_services", []))
                top1 = (row.get("prediction") or {}).get("ranked_node_ids", [None])[0]
                if healthy:
                    benign_eligible.append(top1 in healthy)
            result.append({"subset": subset, "method": method, "n": len(method_rows),
                           "n_total": len(method_rows), "scored_n": len(vals),
                           "hit1": _mean([float(r["hit1"]) for r in vals]),
                           "benign_error": (round(100.0 * _mean([float(v) for v in benign_eligible]), 6)
                                             if benign_eligible else None),
                           "benign_error_denominator": len(benign_eligible),
                           "status": "measured" if vals else "not_applicable"})
    controls = [row for row in audits.values() if row.get("scenario_flags", {}).get("benign_dynamic_change")
                and row.get("verified")]
    return {"status": "measured" if result else "pending_e2_or_control_predictions",
            "rows": result,
            "control_cases_available": len(controls),
            "limitation": ("Benign-error is computed only for verified controls scored by the same fitted methods."
                           if controls else "Benign-error is not reported until the same fitted methods score verified controls.")}


def _adapter_manifest(ranking_dir: Optional[Path]) -> Dict[str, Any]:
    if ranking_dir is None:
        return {"status": "pending_ranking_artifact"}
    manifest = _load(ranking_dir / "source_manifest.json", {}) or {}
    return {"status": "measured", "methods": METHODS,
            "source_manifest": manifest, "predictions_sha256": _sha(ranking_dir / "predictions.jsonl")
            if (ranking_dir / "predictions.jsonl").is_file() else None,
            "shared_evidence": "temporal-v3 cache, identical candidate inventory and split"}


def _write_ranking_tex(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "% Generated from ranking_summary.json; insert only after the revised-source table map is available.",
        "\\begin{tabular}{lrrr}", "\\toprule", "Method & Hit@1 (\\%) & Hit@3 (\\%) & MRR (\\%) " + r"\\",
        "\\midrule",
    ]
    for row in summary.get("methods", []):
        def fmt(key):
            value = row.get(key)
            return "--" if value is None else "%.2f" % (100.0 * float(value))
        lines.append("%s & %s & %s & %s \\\\" % (row["method"], fmt("hit1"), fmt("hit3"), fmt("mrr")))
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _agent_summary(agent_dir: Optional[Path]) -> Dict[str, Any]:
    if agent_dir is None:
        return {"status": "pending_luna_agent_artifact", "model": "gpt-5.6-luna"}
    summary = _load(agent_dir / "summary.json", {}) or {}
    if not summary:
        return {"status": "pending_luna_agent_artifact", "model": "gpt-5.6-luna"}
    return {"status": "measured_simulated_agent", "model": "gpt-5.6-luna",
            "human_participants": 0, "summary": summary,
            "interpretation": "Simulated SRE-agent result; not a human-subject estimate."}


def _contract_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"status": "pending_contract_artifact"}
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    total = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites)
    return {"status": "measured", "tests_total": total, "tests_passed": total - failures,
            "source": str(path.resolve()), "source_sha256": _sha(path)}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    raw = Path(args.data_root).resolve()
    out = Path(args.out_dir).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("output must be outside raw data")
    out.mkdir(parents=True, exist_ok=True)
    audits = _audit_index(Path(args.audit).resolve()) if args.audit else {}
    ranking_dir = Path(args.ranking_dir).resolve() if args.ranking_dir else None
    rows = _ranking(ranking_dir)
    artifacts = {
        "adapter_manifest": _adapter_manifest(ranking_dir),
        "ranking_summary": _ranking_summary(rows),
        "shortcut_summary": _shortcut_summary(rows, audits),
        "ablation_summary": _load(Path(args.ablation_dir).resolve() / "ablation_summary.json", {"status": "pending_retrained_ablation"})
        if args.ablation_dir else {"status": "pending_retrained_ablation"},
        "stability_summary": _load(Path(args.stability_dir).resolve() / "stability_summary.json", {"status": "pending_seed_split_bootstrap"})
        if args.stability_dir else {"status": "pending_seed_split_bootstrap"},
        "revision_summary": {"status": "pending_revision_pool_and_agent"},
        "human_results": _agent_summary(Path(args.agent_dir).resolve() if args.agent_dir else None),
        "human_process": _agent_summary(Path(args.agent_dir).resolve() if args.agent_dir else None),
        "walkthrough": _contract_summary(Path(args.contract_xml).resolve()),
        "latency_summary": _load(Path(args.latency_dir).resolve() / "latency_summary.json", {"status": "pending_runtime_harness"})
        if args.latency_dir else {"status": "pending_runtime_harness"},
    }
    for name, value in artifacts.items():
        (out / (name + ".json")).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_ranking_tex(out / "table13_new_ranking.tex", artifacts["ranking_summary"])
    manifest = {"data_root": str(raw), "audit": str(Path(args.audit).resolve()) if args.audit else None,
                "ranking_dir": str(ranking_dir) if ranking_dir else None, "seed": args.seed,
                "planner": args.planner, "limit": args.limit, "agent_model": "gpt-5.6-luna",
                "method_count": len(METHODS), "raw_data_unchanged": True}
    (out / "downstream_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"out_dir": str(out), "ranking": artifacts["ranking_summary"],
            "shortcut": artifacts["shortcut_summary"], "human": artifacts["human_results"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", default="rule", choices=("rule", "llm"))
    parser.add_argument("--audit")
    parser.add_argument("--ranking-dir")
    parser.add_argument("--agent-dir")
    parser.add_argument("--ablation-dir")
    parser.add_argument("--stability-dir")
    parser.add_argument("--latency-dir")
    parser.add_argument("--contract-xml", default="artifacts/verified_revision_v1/contract/state_fixture_junit_20260930.xml")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
