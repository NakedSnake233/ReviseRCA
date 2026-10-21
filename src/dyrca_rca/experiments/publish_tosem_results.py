"""Publish measured four-copy results, preserving unavailable cells and sources.

The 38-page revised-source bundle is not present here. Standalone tables carry
its stable table labels in a manifest; the available paper/main.tex imports a
separate follow-up section so old releases never acquire the new denominators.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

TABLES = {
    7: ("new_systems", "Application and observations"),
    8: ("new_cohorts", "Collection flow and frozen split"),
    9: ("scenario_design", "Realized scenario coverage"),
    10: ("fidelity_results", "Independent representation fidelity"),
    11: ("feature_coverage", "Comparable and active temporal fields"),
    12: ("adapter_boundaries", "Baseline adaptation boundaries"),
    13: ("new_ranking", "Matched dynamic-evidence service ranking"),
    14: ("shortcut_results", "Change-cue controls"),
    15: ("retrained_ablation", "Fixed-configuration retrained ablations"),
    16: ("stability_results", "Training-seed and split-seed stability"),
    17: ("revision_fidelity", "Six-condition revision fidelity"),
    18: ("revision_ranking", "Six-condition revision ranking"),
    19: ("recorded_walkthrough", "Recorded revision walkthrough"),
    20: ("conditions", "Simulated SRE-agent conditions"),
    21: ("human_results", "Simulated SRE-agent service ranking"),
    22: ("human_process", "Simulated SRE-agent correction process"),
    23: ("runtime_results", "Measured operation latency"),
}


def load(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def escape(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in str(value))


def number(value, percent=False, digits=2):
    if value is None:
        return "--"
    return f"{float(value) * (100 if percent else 1):.{digits}f}"


def table(path, app, label, caption, columns, rows):
    lines = [r"\begin{table}[tbp]", r"\caption{" + escape(caption) + "}",
             r"\label{tab:" + label + "_" + app + "}", r"\resultstable",
             r"\begin{tabularx}{\linewidth}{@{}X" + "r" * (len(columns) - 1) + "@{}}",
             r"\toprule", " & ".join(escape(c) for c in columns) + r"\\", r"\midrule"]
    lines.extend(" & ".join(escape(value) for value in row) + r"\\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    Path(path).write_text("\n".join(lines))


def run(args):
    from dyrca_rca.experiments.run_tosem_downstream import _ranking_summary, _shortcut_summary, _jsonl
    artifacts = {key: Path(path) for key, path in args.artifacts.items()}
    index = load(artifacts["index"] / "collection_index.json")
    if args.limit is not None or not index.get("formal_ready") or index.get("limit") is not None:
        raise ValueError("A partial collection cannot be published as a formal experiment")
    out = Path(args.out_dir).resolve()
    raw = Path(args.data_root).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("Result tables must be outside raw collection data")
    out.mkdir(parents=True, exist_ok=True)
    app = args.system
    prefix = "ob" if app == "boutique" else "tt"
    display = "Online Boutique" if app == "boutique" else "Train Ticket"
    statuses = {n: {"table": n, "label": "tab:" + label, "status": "pending",
                    "reason": "Required experiment evidence not available", "source": None}
                for n, (label, _) in TABLES.items()}
    fields, sources, includes = {}, {}, []

    def measured(n, columns, rows, source, caption=None, status="measured"):
        label, title = TABLES[n]
        name = f"table{n}_{label}.tex"
        table(out / name, app, label, caption or (display + ": " + title + ". New four-copy cohort."), columns, rows)
        statuses[n].update(status=status, reason="", source=str(source))
        includes.append(name)
        if Path(source).is_file():
            sources[str(Path(source).resolve())] = file_sha(source)

    audit_path = artifacts["audit"] / "collection_audit.jsonl"
    audits = _jsonl(audit_path)
    summary_path = artifacts["audit"] / "e0_e1_summary.json"
    summary = next(row for row in load(summary_path)["applications"] if row["application"] == app)
    for key, value in summary.items():
        if isinstance(value, (int, float)) or value is None:
            fields[prefix + "." + key] = {"value": value, "source": str(summary_path),
                                         "denominator": summary["verified_n"]}
    measured(7, ["Observation", "Measured value"],
             [("Logical services (median)", summary["services_median"]),
              ("Pods pre / during (median)", f"{summary['nodes_median_pre']} / {summary['nodes_median_during']}"),
              ("Trace-summary rows (median)", summary["trace_rows_median"]),
              ("Namespaces", 4)], summary_path, status="partial")
    split = load(artifacts["prepared"] / "candidate_release.json") if "prepared" in artifacts else None
    rows = [("Intended runs", summary["intended_n"]), ("Complete exported runs", summary["scheduler_success_n"]),
            ("Topology/evidence audit passed", summary["verified_n"]),
            ("Excluded by this audit", summary["unverified_n"])]
    if split:
        rows += [("Verified T1--T4 training", len(split["split"]["train"])),
                 ("Verified T1--T4 validation", len(split["split"]["val"])),
                 ("Verified T1--T4 main test", len(split["split"]["test"]))]
    measured(8, ["Cohort", "Runs"], rows, summary_path,
             display + ": collection and configuration-group split. Passing this topology/evidence audit does not establish independent causal verification.")
    measured(9, ["Scenario", "Audit-passed runs"], sorted(summary["scenario_counts"].items()), summary_path,
             display + ": task-family and counterfactual coverage. C1--C6 remain separate from T1--T4; these categories are not the manuscript's overlapping causal-scene rubric.")
    measured(11, ["Family", "Comparable objects", "Active cases"],
             [(family, summary["eligible_" + family + "_comparisons"], summary["active_" + family + "_cases"])
              for family in ("endpoint", "pod", "version")], summary_path, status="partial")
    statuses[10]["reason"] = "Independent object/reference fidelity and natural repair-opportunity audit are not available"
    if "ranking" in artifacts:
        ranking = artifacts["ranking"]
        result_path = ranking / "results.jsonl"
        result_rows = _jsonl(result_path)
        ranking_summary = _ranking_summary(result_rows)
        counts = {row["n"] for row in ranking_summary["methods"]}
        if ranking_summary["method_count"] != 9 or len(counts) != 1 or 0 in counts:
            raise ValueError("Main table requires nine methods with an identical nonempty test cohort")
        inventories = [{r["experiment"] for r in result_rows if r.get("split") == "test" and r["method"] == row["method"]}
                       for row in ranking_summary["methods"]]
        if any(names != inventories[0] for names in inventories):
            raise ValueError("Main-table methods have different test cases")
        if any(row["n"] != row["scored_n"] or row["n"] != len(inventories[i])
               for i, row in enumerate(ranking_summary["methods"])):
            raise ValueError("Main-table predictions are missing scores or contain duplicate cases")
        measured(13, ["Method", "n", "Hit@1 (%)", "Hit@3 (%)", "MRR"],
                 [(r["method"], r["n"], number(r["hit1"], True), number(r["hit3"], True), number(r["mrr"], digits=4))
                  for r in ranking_summary["methods"]], result_path,
                 display + ": six source-guided RCA baselines and three graph-update conditions share temporal-v3 evidence, cutoff, candidates and test cases. MRR uses the 0--1 scale.")
        for row in ranking_summary["methods"]:
            for key in ("hit1", "hit3", "mrr"):
                fields[f"{prefix}.ranking.{row['method']}.{key}"] = {"value": row[key], "source": str(result_path), "denominator": row["n"]}
        measured(12, ["Boundary", "Record"], [("Shared evidence", "temporal-v3"),
                 ("Source manifest", file_sha(ranking / "source_manifest.json")[:12]),
                 ("Evidence manifest", file_sha(ranking / "evidence_manifest.json")[:12]),
                 ("Predictions frozen before test labels", file_sha(ranking / "predictions.jsonl")[:12])],
                 ranking / "source_manifest.json")
        shortcuts = _shortcut_summary(result_rows, {row["case_id"]: row for row in audits})
        eligible = [r for r in shortcuts["rows"] if r["n"]]
        if eligible:
            measured(14, ["Stratum / method", "n", "Hit@1 (%)"],
                     [(r["subset"] + " / " + r["method"], r["n"], number(r["hit1"], True)) for r in eligible],
                     result_path, status="partial")
    if "ablation" in artifacts:
        source = artifacts["ablation"] / "ablation_summary.json"
        report = load(source)
        names = {"current_state_v2": "Current-state v2", "full_temporal_v3": "Full temporal v3",
                 "without_change_values": "Without change values", "without_temporal_masks": "Without temporal masks",
                 "constant_positive_edge_weights": "Constant positive edge weights"}
        measured(15, ["Condition", "n", "Hit@1 (%)", "MRR"],
                 [(names[r["variant"]], r["n"], number(r["hit1"], True), number(r["mrr"], digits=4)) for r in report["variants"]], source)
    if "stability" in artifacts:
        source = artifacts["stability"] / "stability_summary.json"
        report = load(source)
        rows = [(r["kind"] + " / " + r["method"], r["runs"],
                 number(r["mrr_mean"], digits=4) + " +/- " + number(r["mrr_sd"], digits=4)) for r in report["summary"]]
        measured(16, ["Repetition / method", "Runs", "MRR mean +/- SD"], rows, source,
                 display + ": five training seeds on a fixed split, and three split seeds with training seed 42. Comparator selected on validation: " + report["comparator"] + ". Coefficient sensitivity is pending.", status="partial")
    statuses[17]["reason"] = "The independent six-condition revision pool has not been executed by this pipeline"
    statuses[18]["reason"] = statuses[17]["reason"]
    if "agent" in artifacts:
        source = artifacts["agent"] / "summary.json"
        report = load(source)
        scored = _jsonl(artifacts["agent"] / "scored.jsonl")
        measured(20, ["Condition", "Access"], [("H0", "Evidence"), ("H1", "Evidence and read-only AI"),
                                               ("H2", "Evidence, AI and workbench edits")], source,
                 display + ": simulated SRE agents; reference access is shared probabilistically across all three conditions. No human participants.")
        rows = []
        for condition in ("H0", "H1", "H2"):
            selected = [r for r in scored if r["condition"] == condition and r.get("service_correct") is not None]
            correct = sum(bool(r["service_correct"]) for r in selected)
            llm = sum(r["agent_source"] == "llm" for r in selected)
            rows.append((condition, len(selected), correct, number(correct / len(selected) if selected else None, True), llm))
        measured(21, ["Condition", "n", "Service correct", "Hit@1 (%)", "Luna diagnoses"], rows, source,
                 display + ": held-out simulated-agent service-only outcomes. Deterministic fallback counts are retained. Mechanism/evidence correctness and human task timing are not measured.", status="partial")
        measured(22, ["Process", "Count"], [("Seeded endpoint defects", report["seeded_defect_case_count"]),
                 ("Reference accessible", report["oracle_access_case_count"]),
                 ("Repair proposed / accepted", f"{report['repair_proposed']} / {report['repair_accepted']}"),
                 ("Seeded objects corrected", report["seeded_defects_corrected"]),
                 ("Accepted unnecessary edits", report["accepted_unnecessary_edits_after_guard"]),
                 ("Luna / fallback accepted repairs", f"{report['llm_accepted_repairs']} / {report['rule_fallback_accepted_repairs']}")],
                 source, status="partial")
        candidates = sorted([r for r in scored if r["condition"] == "H2" and
                             r["seeded_defect"]["seeded_defect"] and r["oracle_access"]], key=lambda r: r["case_id"])
        if candidates:
            row = candidates[0]
            measured(19, ["Recorded item", "Value"], [("Case", row["case_id"]),
                     ("Selection", "First ID with seeded defect and reference access"),
                     ("Repair proposed / accepted", f"{row['repair_proposed']} / {row['repair_accepted']}"),
                     ("Object correct after", row["topology_object_correct_after"])],
                     artifacts["agent"] / "scored.jsonl",
                     display + ": actual API workbench session with a controlled endpoint defect. Selection does not require ranking improvement; it is not a natural GUI-user walkthrough.", status="partial")
    if "latency" in artifacts:
        source = artifacts["latency"] / "latency_summary.json"
        report = load(source)
        measured(23, ["Operation", "Samples", "p50 (ms)", "p95 (ms)"],
                 [(name, row["n"], number(row["p50_ms"]), number(row["p95_ms"])) for name, row in report["summaries"].items()],
                 source, "Observed CPU operation times while Train Ticket collection may be active. Five warm-ups and twenty measured repetitions per case; remaining interaction boundaries are pending.", status="partial")
    lines = [r"\subsubsection{" + display + "}",
             "This follow-up uses the completed four-copy collection and separately frozen models. "
             "Archived release scores elsewhere in this paper keep their original cohorts. "
             "The integrity gate verifies exported evidence and topology changes; independent causal "
             "and representation verification remains an additional requirement."]
    lines += [r"\input{generated/tosem_four_copies/" + app + "/" + name[:-4] + "}" for name in includes]
    pending = [str(n) for n in statuses if statuses[n]["status"] == "pending"]
    if pending:
        lines.append("Revised-manuscript tables with no measured artifact in this snapshot: " + ", ".join(pending) + ".")
    (out / "results.tex").write_text("\n".join(lines) + "\n")
    with (out / "table_status.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("table", "label", "status", "reason", "source"))
        writer.writeheader()
        writer.writerows(statuses.values())
    manifest = {"system": app, "seed": args.seed, "planner": args.planner, "limit": args.limit,
                "formal_collection": index, "tables": list(statuses.values()), "fields": fields,
                "source_hashes": sources, "raw_data_unchanged": True,
                "revised_source_present": False,
                "source_boundary": "Generated tables map to revised PDF labels; only the available paper/main.tex is updated"}
    (out / "backfill_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    paper = Path(args.paper_dir).resolve()
    generated = paper / "generated/tosem_four_copies"
    generated.mkdir(parents=True, exist_ok=True)
    with (paper / ".tosem_backfill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # The immutable result directory is linked as a unit; readers cannot
        # combine table rows from two attempts. Only links owned here change.
        link = generated / app
        temporary = generated / (app + ".next")
        if link.exists() and not link.is_symlink():
            raise ValueError("Refusing to replace a user-owned generated table directory")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(out, target_is_directory=True)
        temporary.replace(link)
        wrapper = [r"\subsection{Verified four-copy follow-up}"]
        wrapper += [r"\input{generated/tosem_four_copies/" + system + "/results}"
                    for system in ("boutique", "trainticket") if (generated / system / "results.tex").is_file()]
        temporary = generated / "results.next"
        temporary.write_text("\n".join(wrapper) + "\n")
        temporary.replace(generated / "results.tex")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--system", choices=("boutique", "trainticket"), required=True)
    parser.add_argument("--artifacts", type=json.loads, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=("rule",), default="rule")
    parser.add_argument("--limit", type=int)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
