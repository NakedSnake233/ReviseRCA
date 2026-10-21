#!/usr/bin/env python3
"""Aggregate frozen Online Boutique and Train Ticket baseline releases.

The script is deliberately read-only with respect to raw cases.  It accepts
two completed ``run_dynamic_baselines`` directories, verifies that the seven
methods and dataset provenance are present in each, and writes a machine
readable comparison plus a compact LaTeX table.  It must be run only after
the Train Ticket completeness audit has passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List


METHODS = [
    "NSigma-Dyn", "BARO-Dyn", "CIRCA-Dyn", "CloudRanger-Dyn",
    "Eadro-Dyn", "HolisticRCA-Dyn", "DY-RCA",
]
METRICS = ("test_hit1", "test_hit3", "test_mrr")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(path: Path, expected_system: str) -> Dict[str, object]:
    table = path / "main_comparison.csv"
    config_path = path / "config.json"
    if not table.is_file() or not config_path.is_file():
        raise ValueError("incomplete baseline release: " + str(path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    system = config.get("system", "boutique")
    if system != expected_system:
        raise ValueError("{} reports system={!r}, expected {!r}".format(path, system, expected_system))
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_method = {row.get("method"): row for row in rows}
    if set(by_method) != set(METHODS):
        raise ValueError("{} methods: expected {}, got {}".format(
            path, METHODS, sorted(by_method)))
    for method in METHODS:
        for metric in METRICS:
            try:
                float(by_method[method][metric])
            except (KeyError, TypeError, ValueError):
                raise ValueError("non-numeric {} for {} in {}".format(metric, method, path))
    return {
        "system": expected_system,
        "dataset_id": config.get("dataset_id", "dyrca-v1" if expected_system == "boutique" else "dyrca-v1-trainticket"),
        "run": str(path.resolve()),
        "run_sha256": sha256(table),
        "test_n": int(by_method[METHODS[0]]["test_n"]),
        "regression_n": int(by_method[METHODS[0]]["regression_n"]),
        "rows": by_method,
    }


def _cell(value: str, bold: bool, metric: str) -> str:
    if metric.endswith("mrr"):
        rendered = "{:.4f}".format(float(value))
    else:
        rendered = "{:.2f}".format(100.0 * float(value))
    return "\\textbf{{{}}}".format(rendered) if bold else rendered


def make_tex(releases: List[Dict[str, object]]) -> str:
    best = {}
    for release in releases:
        rows = release["rows"]
        for metric in METRICS:
            values = [float(rows[m][metric]) for m in METHODS]
            best[(release["system"], metric)] = max(values)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Leakage-controlled matched dynamic-evidence comparison on both released systems. Each dataset uses a frozen target-service-held-out split: no target service occurs in both train and main test. All seven methods receive the same per-case observations and candidate inventory within a dataset. Hit@1 and Hit@3 are percentages; MRR is unitless.}",
        r"\label{tab:dual_dataset_service_heldout}",
        r"\resultstable",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Online Boutique} & \multicolumn{3}{c}{Train Ticket}\\",
        r"Method & Hit@1 & Hit@3 & MRR & Hit@1 & Hit@3 & MRR\\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [method]
        for release in releases:
            rows = release["rows"]
            for metric in METRICS:
                value = rows[method][metric]
                cells.append(_cell(value, abs(float(value) - best[(release["system"], metric)]) < 1e-12, metric))
        lines.append(" & ".join(cells) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boutique-run", required=True)
    parser.add_argument("--trainticket-run", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-tex", required=True)
    args = parser.parse_args()
    releases = [
        load_run(Path(args.boutique_run), "boutique"),
        load_run(Path(args.trainticket_run), "trainticket"),
    ]
    payload = {
        "schema_version": "dual-dataset-baselines-v1",
        "methods": METHODS,
        "datasets": [],
    }
    for release in releases:
        item = {k: v for k, v in release.items() if k != "rows"}
        item["metrics"] = release["rows"]
        payload["datasets"].append(item)
    out_json, out_tex = Path(args.out_json), Path(args.out_tex)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_tex.write_text(make_tex(releases), encoding="utf-8")
    print(json.dumps({"datasets": [(r["system"], r["test_n"]) for r in releases],
                      "methods": len(METHODS), "out_json": str(out_json), "out_tex": str(out_tex)}))


if __name__ == "__main__":
    main()
