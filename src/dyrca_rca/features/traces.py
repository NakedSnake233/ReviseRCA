"""Trace feature extraction for experiment directories."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dyrca_rca.planner.signals import trace_mismatch_edges


def load_trace_features(experiment_dir: str) -> Dict[str, Any]:
    """Load trace features from trace_summary.csv when present."""
    trace_path = Path(experiment_dir) / "traces" / "trace_summary.csv"
    result = {
        "source": "trace_summary.csv" if trace_path.exists() else "missing",
        "services": {},
        "edges": {},
        "route_mismatch_candidates": [],
        "missing": not trace_path.exists(),
    }
    if not trace_path.exists():
        return result

    result["route_mismatch_candidates"] = trace_mismatch_edges(trace_path)
    with trace_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_span = {row.get("span_id", ""): row for row in rows if row.get("span_id")}
    durations_by_service = {}
    durations_by_edge = {}

    for row in rows:
        service = str(row.get("service", "") or row.get("destination_service", "") or "")
        duration = _duration_ms(row)
        status = str(row.get("status_code", row.get("status", "")) or "")
        if service:
            stats = result["services"].setdefault(service, {"span_count": 0, "error_count": 0})
            stats["span_count"] += 1
            if _is_error_status(status):
                stats["error_count"] += 1
            durations_by_service.setdefault(service, []).append(duration)

        source, target = _edge_from_row(row, by_span)
        if source and target and source != target:
            edge_key = f"{source}->{target}"
            edge_stats = result["edges"].setdefault(edge_key, {
                "source": source,
                "target": target,
                "span_count": 0,
                "error_count": 0,
            })
            edge_stats["span_count"] += 1
            if _is_error_status(status):
                edge_stats["error_count"] += 1
            durations_by_edge.setdefault(edge_key, []).append(duration)

    for service, values in durations_by_service.items():
        _finish_duration_stats(result["services"][service], values)
    for edge_key, values in durations_by_edge.items():
        _finish_duration_stats(result["edges"][edge_key], values)
    for stats in result["services"].values():
        stats["error_rate"] = stats["error_count"] / stats["span_count"] if stats["span_count"] else 0.0
    for stats in result["edges"].values():
        stats["error_rate"] = stats["error_count"] / stats["span_count"] if stats["span_count"] else 0.0

    return result


def _edge_from_row(row: Dict[str, str], by_span: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    source = _first(row, ["source", "source_service", "client_service"])
    target = _first(row, ["destination_service", "observed_dst", "actual_dst", "target_service"])
    if source and target:
        return source, target

    parent_id = row.get("parent_span_id", "")
    parent = by_span.get(parent_id, {})
    if parent:
        source = str(parent.get("service", "") or "")
        target = str(row.get("service", "") or "")
        if source and target:
            return source, target

    service = str(row.get("service", "") or "")
    operation = str(row.get("operation", "") or "")
    inferred = _target_from_operation(operation)
    if service and inferred:
        return service, inferred
    return "", ""


def _target_from_operation(operation: str) -> str:
    if not operation:
        return ""
    token = operation.split("/", 1)[0].split(":", 1)[0]
    token = token.split(".")[0]
    return token if token and token != operation else token


def _duration_ms(row: Dict[str, str]) -> float:
    if row.get("duration_ms"):
        return _float(row.get("duration_ms"))
    if row.get("duration_nano"):
        return _float(row.get("duration_nano")) / 1e6
    return 0.0


def _finish_duration_stats(stats: Dict[str, Any], values: List[float]) -> None:
    values = sorted(v for v in values if v is not None)
    if not values:
        stats["avg_duration"] = 0.0
        stats["p95_duration"] = 0.0
        return
    stats["avg_duration"] = sum(values) / len(values)
    idx = int((len(values) - 1) * 0.95)
    stats["p95_duration"] = values[idx]


def _is_error_status(status: str) -> bool:
    if not status:
        return False
    text = status.upper()
    return text not in {"0", "OK", "STATUS_CODE_UNSET", "STATUS_CODE_OK"} and (
        text.startswith("5") or text in {"ERROR", "STATUS_CODE_ERROR", "2"}
    )


def _first(row: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
