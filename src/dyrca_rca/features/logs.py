"""Log feature extraction for experiment directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_log_features(experiment_dir: str, include_all: bool = True) -> Dict[str, Any]:
    """Load per-service error and total log counts.

    Only `errors.jsonl` and `all.jsonl` are considered. Missing files do not
    raise and produce zero counts.
    """
    logs_dir = Path(experiment_dir) / "logs"
    result = {
        "services": {},
        "missing_files": [],
        "present_files": [],
    }
    error_path = logs_dir / "errors.jsonl"
    all_path = logs_dir / "all.jsonl"

    if error_path.exists():
        result["present_files"].append("errors.jsonl")
        _consume_jsonl(error_path, result, error_only=True)
    else:
        result["missing_files"].append("errors.jsonl")

    if include_all:
        if all_path.exists():
            result["present_files"].append("all.jsonl")
            _consume_jsonl(all_path, result, error_only=False)
        else:
            result["missing_files"].append("all.jsonl")

    return result


def _consume_jsonl(path: Path, result: Dict[str, Any], error_only: bool) -> None:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            service = str(row.get("service", "") or row.get("app", "") or "")
            if not service:
                continue
            stats = result["services"].setdefault(service, {"error_count": 0, "total_count": 0})
            if error_only:
                stats["error_count"] += 1
            else:
                stats["total_count"] += 1
                level = str(row.get("level", "") or "").upper()
                if level in {"ERROR", "WARN", "FATAL"}:
                    stats["error_count"] += 1
