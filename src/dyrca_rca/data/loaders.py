"""Lightweight JSON loaders for experiment indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


JsonObject = Dict[str, Any]


@dataclass
class ExperimentSources:
    """Small files needed to index one experiment directory."""

    exp_dir: Path
    metadata: Optional[JsonObject] = None
    ground_truth: Optional[JsonObject] = None
    topology_diff: Optional[JsonObject] = None
    verification_report: Optional[JsonObject] = None
    topology_pre: Optional[JsonObject] = None
    topology_during: Optional[JsonObject] = None
    topology_post: Optional[JsonObject] = None
    errors: List[str] = field(default_factory=list)


def safe_read_json(path: Path, required: bool = False) -> Tuple[Optional[JsonObject], Optional[str]]:
    """Read a JSON object, returning an error string instead of raising."""
    if not path.exists():
        if required:
            return None, f"missing required file: {path.name}"
        return None, None

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except JSONDecodeError as exc:
        return None, f"bad json in {path.name}: line {exc.lineno} column {exc.colno}"
    except OSError as exc:
        return None, f"could not read {path.name}: {exc}"

    if not isinstance(data, dict):
        return None, f"expected object in {path.name}, got {type(data).__name__}"
    return data, None


def load_experiment_sources(exp_dir: Union[str, Path]) -> ExperimentSources:
    """Load index-time JSON sources without metrics, logs, or traces."""
    exp_path = Path(exp_dir)
    sources = ExperimentSources(exp_dir=exp_path)

    required_files = {
        "metadata": exp_path / "metadata.json",
        "ground_truth": exp_path / "ground_truth.json",
        "topology_pre": exp_path / "topology" / "pre.json",
        "topology_during": exp_path / "topology" / "during.json",
        "topology_post": exp_path / "topology" / "post.json",
    }
    optional_files = {
        "topology_diff": exp_path / "topology_diff.json",
        "verification_report": exp_path / "verification_report.json",
    }

    for attr, path in required_files.items():
        data, error = safe_read_json(path, required=True)
        setattr(sources, attr, data)
        if error:
            sources.errors.append(error)

    for attr, path in optional_files.items():
        data, error = safe_read_json(path, required=False)
        setattr(sources, attr, data)
        if error:
            sources.errors.append(error)

    return sources


def list_experiment_dirs(data_root: Union[str, Path], limit: Optional[int] = None) -> List[Path]:
    """Return sorted immediate child directories under a data root."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {root}")

    dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        dirs = dirs[:limit]
    return dirs


def has_metric_files(exp_dir: Path) -> bool:
    """Check for metric artifacts without loading them."""
    metrics_dir = exp_dir / "metrics"
    if not metrics_dir.is_dir():
        return False
    return any(metrics_dir.glob("*.csv")) or any(metrics_dir.glob("*.parquet"))


def has_log_files(exp_dir: Path) -> bool:
    """Check for log artifacts without loading them."""
    logs_dir = exp_dir / "logs"
    if not logs_dir.is_dir():
        return False
    return any(logs_dir.glob("*.jsonl"))


def has_trace_files(exp_dir: Path) -> bool:
    """Check for trace artifacts without loading them."""
    traces_dir = exp_dir / "traces"
    if not traces_dir.is_dir():
        return False
    return any((traces_dir / name).exists() for name in ("trace_summary.csv", "traces_raw.json"))
