"""Build a lightweight index over dynamic RCA experiment directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd

from dyrca_rca.data.loaders import (
    ExperimentSources,
    has_log_files,
    has_metric_files,
    has_trace_files,
    list_experiment_dirs,
    load_experiment_sources,
)
from dyrca_rca.data.schema import (
    ExperimentIndexRecord,
    coerce_index_frame,
    stable_json,
    write_index_frame,
)


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _scenario_from_name(name: str) -> str:
    prefix = name.split("-", 1)[0].upper()
    return prefix if prefix in {"T1", "T2", "T3", "T4", "C1", "C2", "C3", "C4", "C5", "C6"} else ""


def _verification_status(sources: ExperimentSources) -> str:
    report = sources.verification_report or {}
    diff = sources.topology_diff or {}
    annotation = diff.get("annotation", {}) if isinstance(diff.get("annotation"), dict) else {}
    return _first_string(
        report.get("verification_result"),
        annotation.get("verification_result"),
    )


def _topology_change_type(sources: ExperimentSources) -> str:
    metadata = sources.metadata or {}
    ground_truth = sources.ground_truth or {}
    report = sources.verification_report or {}
    diff = sources.topology_diff or {}
    annotation = diff.get("annotation", {}) if isinstance(diff.get("annotation"), dict) else {}
    return _first_string(
        ground_truth.get("topology_change_type"),
        metadata.get("topology_change_type"),
        report.get("topology_change_type"),
        annotation.get("topology_change_type"),
    )


def build_index_record(exp_dir: Union[str, Path]) -> ExperimentIndexRecord:
    """Build one index record, preserving errors on the row."""
    exp_path = Path(exp_dir)
    try:
        sources = load_experiment_sources(exp_path)
    except Exception as exc:
        sources = ExperimentSources(exp_dir=exp_path, errors=[f"unexpected index error: {exc}"])

    metadata = sources.metadata or {}
    ground_truth = sources.ground_truth or {}
    diff = sources.topology_diff or {}
    labels = ground_truth.get("labels", {})
    if not isinstance(labels, dict):
        labels = {}

    experiment_id = _first_string(
        metadata.get("experiment_id"),
        ground_truth.get("experiment_id"),
        diff.get("experiment_id"),
        exp_path.name,
    )
    scenario_id = _first_string(
        metadata.get("scenario_id"),
        labels.get("scenario"),
        diff.get("scenario_id"),
        _scenario_from_name(exp_path.name),
    )
    target_service = _first_string(
        metadata.get("target_service"),
        diff.get("target_service"),
        ground_truth.get("root_cause_service"),
    )

    return ExperimentIndexRecord(
        experiment_id=experiment_id,
        scenario_id=scenario_id,
        scenario_name=_first_string(metadata.get("scenario_name")),
        target_service=target_service,
        params_json=stable_json(metadata.get("fault_params", {})),
        root_cause_service=_first_string(ground_truth.get("root_cause_service"), target_service),
        root_cause_type=_first_string(ground_truth.get("fault_type"), metadata.get("fault_type")),
        has_pre=sources.topology_pre is not None,
        has_during=sources.topology_during is not None,
        has_post=sources.topology_post is not None,
        has_metrics=has_metric_files(exp_path),
        has_logs=has_log_files(exp_path),
        has_traces=has_trace_files(exp_path),
        topology_change_type=_topology_change_type(sources),
        verification_status=_verification_status(sources),
        data_path=str(exp_path),
        error_message="; ".join(sources.errors),
    )


def build_index(data_root: Union[str, Path], limit: Optional[int] = None) -> pd.DataFrame:
    """Scan experiment directories and return a lightweight pandas index."""
    rows = []
    for exp_dir in list_experiment_dirs(data_root, limit=limit):
        rows.append(build_index_record(exp_dir).to_dict())
    return coerce_index_frame(rows)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DY-RCA experiment index.")
    parser.add_argument("--data-root", default="data/experiments",
                        help="Directory containing experiment subdirectories.")
    parser.add_argument("--out", required=True,
                        help="Output path ending in .parquet, .csv, .jsonl, or .ndjson.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of experiment directories to scan.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    frame = build_index(args.data_root, limit=args.limit)
    write_index_frame(frame, args.out)
    print(f"indexed {len(frame)} experiments -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
