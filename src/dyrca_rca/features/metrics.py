"""Metric feature extraction for experiment directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRIC_FILES = [
    "request_rate.csv",
    "latency_p50.csv",
    "latency_p99.csv",
    "cpu_usage.csv",
    "replica_count.csv",
    "service_up.csv",
    "response_flags.csv",
]


def load_metric_features(experiment_dir: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load compact metric summaries by service, edge, and phase.

    Missing files are represented in the returned metadata and do not raise.
    """
    exp_dir = Path(experiment_dir)
    metrics_dir = exp_dir / "metrics"
    summary = {
        "services": {},
        "edges": {},
        "phases": {},
        "missing_files": [],
        "present_files": [],
    }
    if metadata is None:
        metadata = _read_json(exp_dir / "metadata.json")

    for name in METRIC_FILES:
        path = metrics_dir / name
        if not path.exists():
            summary["missing_files"].append(name)
            continue
        summary["present_files"].append(name)
        _consume_metric_csv(path, summary, metadata)

    _finalize_metric_summary(summary)
    return summary


def _consume_metric_csv(path: Path, summary: Dict[str, Any],
                        metadata: Dict[str, Any]) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row.get("metric_name") or path.stem
            labels = _parse_labels(row.get("labels_json", ""))
            value = _float(row.get("value"), 0.0)
            phase = _phase_for_timestamp(row.get("timestamp", ""), metadata)

            if metric_name == "request_rate":
                source = str(labels.get("source_workload", "") or "")
                service = str(labels.get("destination_workload", "") or "")
                response_code = str(labels.get("response_code", "") or "")
                if service:
                    service_stats = _service(summary, service)
                    _add_stat(service_stats, "request_rate_in", value)
                    _add_phase(summary, phase, service, "request_rate_in", value)
                if source:
                    _add_stat(_service(summary, source), "request_rate_out", value)
                if source and service:
                    edge_stats = _edge(summary, source, service)
                    _add_stat(edge_stats, "request_rate", value)
                    if response_code.startswith("5"):
                        _add_stat(edge_stats, "error_rate_numer", value)
                    _add_phase(summary, phase, f"{source}->{service}", "request_rate", value, edge=True)
            elif metric_name in {"latency_p50", "latency_p99"}:
                service = str(labels.get("destination_workload", "") or labels.get("service", "") or "")
                if service:
                    _add_stat(_service(summary, service), metric_name, value)
                    _add_phase(summary, phase, service, metric_name, value)
            elif metric_name == "cpu_usage":
                service = _service_from_pod(str(labels.get("pod", "") or labels.get("service", "") or ""))
                if service:
                    _add_stat(_service(summary, service), "cpu_usage", value)
                    _add_phase(summary, phase, service, "cpu_usage", value)
            elif metric_name == "replica_count":
                service = str(labels.get("deployment", "") or labels.get("pod", "") or "")
                if service:
                    _add_stat(_service(summary, service), "replica_count", value)
                    _add_phase(summary, phase, service, "replica_count", value)
            elif metric_name == "service_up":
                service = _service_from_pod(str(labels.get("pod", "") or labels.get("service", "") or ""))
                if service:
                    _add_stat(_service(summary, service), "service_up", value)
                    _add_phase(summary, phase, service, "service_up", value)
            elif metric_name == "response_flags":
                source = str(labels.get("source_workload", "") or "")
                service = str(labels.get("destination_workload", "") or "")
                flags = str(labels.get("response_flags", "") or "")
                if service:
                    _add_stat(_service(summary, service), "response_flags", value if flags and flags != "-" else 0.0)
                if source and service:
                    _add_stat(_edge(summary, source, service), "response_flags", value if flags and flags != "-" else 0.0)


def _finalize_metric_summary(summary: Dict[str, Any]) -> None:
    max_request = 0.0
    for stats in summary["edges"].values():
        request = stats.get("request_rate", {}).get("sum", 0.0)
        max_request = max(max_request, request)
        err_num = stats.get("error_rate_numer", {}).get("sum", 0.0)
        stats["error_rate"] = err_num / request if request > 0 else 0.0
    for stats in summary["edges"].values():
        request = stats.get("request_rate", {}).get("sum", 0.0)
        stats["normalized_request_rate"] = request / max_request if max_request > 0 else 0.0


def _service(summary: Dict[str, Any], service: str) -> Dict[str, Any]:
    return summary["services"].setdefault(service, {})


def _edge(summary: Dict[str, Any], source: str, target: str) -> Dict[str, Any]:
    return summary["edges"].setdefault(f"{source}->{target}", {"source": source, "target": target})


def _add_stat(stats: Dict[str, Any], name: str, value: float) -> None:
    slot = stats.setdefault(name, {"sum": 0.0, "count": 0, "max": None})
    slot["sum"] += value
    slot["count"] += 1
    slot["max"] = value if slot["max"] is None else max(slot["max"], value)
    slot["avg"] = slot["sum"] / slot["count"] if slot["count"] else 0.0


def _add_phase(summary: Dict[str, Any], phase: str, entity: str, metric: str,
               value: float, edge: bool = False) -> None:
    phase_stats = summary["phases"].setdefault(phase, {"services": {}, "edges": {}})
    bucket_name = "edges" if edge else "services"
    _add_stat(phase_stats[bucket_name].setdefault(entity, {}), metric, value)


def _parse_labels(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _phase_for_timestamp(timestamp: str, metadata: Dict[str, Any]) -> str:
    # Lexical comparison is sufficient for ISO-8601 UTC strings in this dataset.
    tw = metadata.get("time_window", {}) if isinstance(metadata.get("time_window"), dict) else {}
    inject = str(tw.get("t_inject", "") or "")
    recover = str(tw.get("t_recover", "") or "")
    if not timestamp or not inject:
        return "all"
    if timestamp < inject:
        return "pre"
    if recover and timestamp >= recover:
        return "post"
    return "during"


def _service_from_pod(value: str) -> str:
    if not value:
        return ""
    parts = value.rsplit("-", 2)
    return parts[0] if len(parts) >= 3 else value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
