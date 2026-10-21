"""Schemas for lightweight experiment indexing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


INDEX_COLUMNS = [
    "experiment_id",
    "scenario_id",
    "scenario_name",
    "target_service",
    "params_json",
    "root_cause_service",
    "root_cause_type",
    "has_pre",
    "has_during",
    "has_post",
    "has_metrics",
    "has_logs",
    "has_traces",
    "topology_change_type",
    "verification_status",
    "data_path",
    "error_message",
]


BOOLEAN_COLUMNS = [
    "has_pre",
    "has_during",
    "has_post",
    "has_metrics",
    "has_logs",
    "has_traces",
]


@dataclass(frozen=True)
class ExperimentIndexRecord:
    """One row in the experiment index."""

    experiment_id: str
    scenario_id: str
    scenario_name: str
    target_service: str
    params_json: str
    root_cause_service: str
    root_cause_type: str
    has_pre: bool
    has_during: bool
    has_post: bool
    has_metrics: bool
    has_logs: bool
    has_traces: bool
    topology_change_type: str
    verification_status: str
    data_path: str
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        return {col: row.get(col, "") for col in INDEX_COLUMNS}


def stable_json(value: Any) -> str:
    """Serialize metadata params with deterministic key ordering."""
    if value is None:
        value = {}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def coerce_index_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with stable columns and predictable dtypes."""
    frame = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    for column in INDEX_COLUMNS:
        if column not in frame:
            frame[column] = False if column in BOOLEAN_COLUMNS else ""
    for column in BOOLEAN_COLUMNS:
        frame[column] = frame[column].fillna(False).astype(bool)
    for column in INDEX_COLUMNS:
        if column not in BOOLEAN_COLUMNS:
            frame[column] = frame[column].fillna("").astype(str)
    return frame[INDEX_COLUMNS]


def write_index_frame(frame: pd.DataFrame, out_path: Union[str, Path]) -> None:
    """Write the index to parquet, CSV, or JSONL based on file extension."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Writing parquet indexes requires pyarrow or fastparquet. "
                "Install project requirements before using a .parquet output."
            ) from exc
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(
            f"Unsupported index output format '{suffix}'. "
            "Use .parquet, .csv, .jsonl, or .ndjson."
        )
