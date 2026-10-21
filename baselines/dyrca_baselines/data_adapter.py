"""Adapt DY-RCA dynamic-topology experiments to baseline tensor inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dyrca_rca.dataset.materialize import read_index
from dyrca_rca.experiments.run_experiment import _usable_rows
from dyrca_rca.experiments.splits import make_split
from dyrca_rca.features.logs import load_log_features
from dyrca_rca.features.metrics import load_metric_features
from dyrca_rca.features.traces import load_trace_features
from dyrca_rca.planner.signals import KNOWN_EDGES, read_json


PHASES = ["pre", "during", "post"]
STRICT_PHASES = ["pre", "during"]
METRIC_NAMES = [
    "request_rate_in",
    "request_rate_out",
    "latency_p50",
    "latency_p99",
    "cpu_usage",
    "replica_count",
    "service_up",
    "response_flags",
]
TRACE_NAMES = ["span_count", "error_rate", "avg_duration", "p95_duration"]
LOG_NAMES = ["error_count", "total_count", "error_rate", "has_error"]
TOPOLOGY_NAMES = [
    "replicas_pre",
    "replicas_during",
    "replica_delta",
    "endpoint_pre",
    "endpoint_during",
    "endpoint_delta",
    "pod_added",
    "pod_removed",
    "has_v2",
]
STRICT_TOPOLOGY_NAMES = [
    "replicas_pre",
    "endpoint_pre",
    "static_in_degree",
    "static_out_degree",
]


@dataclass
class BaselineDataBundle:
    service_names: List[str]
    edge_index: torch.Tensor
    splits: Dict[str, List[Dict[str, Any]]]
    split_frames: Dict[str, pd.DataFrame]
    normalizer: Dict[str, Dict[str, np.ndarray]]
    adapter_mode: str = "strict"


class BaselineDataset(Dataset):
    def __init__(self, items: Sequence[Dict[str, Any]]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        return {
            "metrics": torch.tensor(item["metrics"], dtype=torch.float32),
            "traces": torch.tensor(item["traces"], dtype=torch.float32),
            "logs": torch.tensor(item["logs"], dtype=torch.float32),
            "topology": torch.tensor(item["topology"], dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "experiment_id": item["experiment_id"],
            "scenario_id": item["scenario_id"],
            "target_service": item["target_service"],
            "label_node_id": item["label_node_id"],
        }


def collate_baseline(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["metrics", "traces", "logs", "topology", "label"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ["experiment_id", "scenario_id", "target_service", "label_node_id"]:
        out[key] = [item[key] for item in batch]
    return out


def build_baseline_bundle(index_path: str,
                          data_root: str = "data/experiments",
                          split: str = "random",
                          seed: int = 42,
                          limit: Optional[int] = None,
                          adapter_mode: str = "strict") -> BaselineDataBundle:
    if adapter_mode not in {"strict", "leaky_dynamic"}:
        raise ValueError(f"unknown adapter_mode: {adapter_mode}")
    frame = read_index(index_path)
    if limit is not None:
        frame = frame.head(limit)
    frame = _usable_rows(frame, data_root=data_root)
    frame = frame[frame["root_cause_service"].notna()].reset_index(drop=True)
    services = discover_services(frame)
    service_to_idx = {service: idx for idx, service in enumerate(services)}
    edge_index = build_edge_index(services)
    split_frames = make_split(frame, split=split, seed=seed)

    raw_splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, split_frame in split_frames.items():
        raw_splits[split_name] = [
            build_item(row.to_dict(), services, service_to_idx, adapter_mode=adapter_mode)
            for _, row in split_frame.iterrows()
        ]
    normalizer = fit_normalizer(raw_splits.get("train", []))
    norm_splits = {
        name: [normalize_item(item, normalizer) for item in items]
        for name, items in raw_splits.items()
    }
    return BaselineDataBundle(
        service_names=services,
        edge_index=edge_index,
        splits=norm_splits,
        split_frames=split_frames,
        normalizer=normalizer,
        adapter_mode=adapter_mode,
    )


def discover_services(frame: pd.DataFrame) -> List[str]:
    services = set()
    for _, row in frame.iterrows():
        exp_dir = Path(str(row["data_path"]))
        for rel in ["topology/pre.json", "topology/during.json", "topology/post.json"]:
            snapshot = read_json(exp_dir / rel, required=False)
            for dep in snapshot.get("deployments", []) or []:
                if isinstance(dep, dict) and dep.get("name"):
                    services.add(str(dep["name"]))
            for svc in snapshot.get("services", []) or []:
                if isinstance(svc, dict) and svc.get("name"):
                    services.add(str(svc["name"]))
        if row.get("root_cause_service"):
            services.add(str(row["root_cause_service"]))
        if row.get("target_service"):
            services.add(str(row["target_service"]))
    return sorted(services)


def build_edge_index(services: Sequence[str]) -> torch.Tensor:
    service_to_idx = {service: idx for idx, service in enumerate(services)}
    edges = []
    for source, target in KNOWN_EDGES:
        if source in service_to_idx and target in service_to_idx:
            edges.append((service_to_idx[source], service_to_idx[target]))
            edges.append((service_to_idx[target], service_to_idx[source]))
    for idx in range(len(services)):
        edges.append((idx, idx))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_item(row: Dict[str, Any], services: Sequence[str],
               service_to_idx: Dict[str, int],
               adapter_mode: str = "strict") -> Dict[str, Any]:
    exp_dir = Path(str(row["data_path"]))
    metadata = read_json(exp_dir / "metadata.json", required=False)
    metrics = metric_tensor(str(exp_dir), services, metadata, adapter_mode=adapter_mode)
    traces = trace_tensor(str(exp_dir), services)
    logs = log_tensor(str(exp_dir), services)
    topology = topology_tensor(str(exp_dir), services, adapter_mode=adapter_mode)
    label_node = str(row.get("root_cause_service") or row.get("target_service") or "")
    label = service_to_idx.get(label_node, -1)
    if label < 0:
        raise ValueError(f"label service not in service universe: {label_node}")
    return {
        "experiment_id": str(row.get("experiment_id") or exp_dir.name),
        "scenario_id": str(row.get("scenario_id") or ""),
        "target_service": str(row.get("target_service") or ""),
        "label_node_id": label_node,
        "label": label,
        "metrics": metrics,
        "traces": traces,
        "logs": logs,
        "topology": topology,
    }


def metric_tensor(experiment_dir: str, services: Sequence[str],
                  metadata: Dict[str, Any],
                  adapter_mode: str = "strict") -> np.ndarray:
    features = load_metric_features(experiment_dir, metadata=metadata)
    phases = STRICT_PHASES if adapter_mode == "strict" else PHASES
    arr = np.zeros((len(services), len(phases), len(METRIC_NAMES)), dtype=np.float32)
    for service_idx, service in enumerate(services):
        for phase_idx, phase in enumerate(phases):
            stats = features.get("phases", {}).get(phase, {}).get("services", {}).get(service, {})
            for metric_idx, metric_name in enumerate(METRIC_NAMES):
                value = _stat_value(stats.get(metric_name))
                arr[service_idx, phase_idx, metric_idx] = value
    return np.nan_to_num(arr)


def trace_tensor(experiment_dir: str, services: Sequence[str]) -> np.ndarray:
    features = load_trace_features(experiment_dir)
    arr = np.zeros((len(services), len(PHASES), len(TRACE_NAMES)), dtype=np.float32)
    for service_idx, service in enumerate(services):
        stats = features.get("services", {}).get(service, {})
        values = [
            _float(stats.get("span_count")),
            _float(stats.get("error_rate")),
            _float(stats.get("avg_duration")),
            _float(stats.get("p95_duration")),
        ]
        arr[service_idx, 1, :] = values
    return np.nan_to_num(arr)


def log_tensor(experiment_dir: str, services: Sequence[str]) -> np.ndarray:
    features = load_log_features(experiment_dir)
    arr = np.zeros((len(services), len(LOG_NAMES)), dtype=np.float32)
    for service_idx, service in enumerate(services):
        stats = features.get("services", {}).get(service, {})
        error_count = _float(stats.get("error_count"))
        total_count = _float(stats.get("total_count"))
        arr[service_idx] = [
            error_count,
            total_count,
            error_count / total_count if total_count > 0 else 0.0,
            1.0 if error_count > 0 else 0.0,
        ]
    return np.nan_to_num(arr)


def topology_tensor(experiment_dir: str, services: Sequence[str],
                    adapter_mode: str = "strict") -> np.ndarray:
    exp_dir = Path(experiment_dir)
    pre = read_json(exp_dir / "topology" / "pre.json", required=False)
    if adapter_mode == "strict":
        pre_stats = _snapshot_service_stats(pre)
        degrees = _static_degrees(services)
        arr = np.zeros((len(services), len(STRICT_TOPOLOGY_NAMES)), dtype=np.float32)
        for idx, service in enumerate(services):
            pre_item = pre_stats.get(service, {})
            degree_item = degrees.get(service, {})
            arr[idx] = [
                _float(pre_item.get("replicas")),
                _float(pre_item.get("endpoint_count")),
                _float(degree_item.get("in_degree")),
                _float(degree_item.get("out_degree")),
            ]
        return np.nan_to_num(arr)

    during = read_json(exp_dir / "topology" / "during.json", required=False)
    diff = read_json(exp_dir / "topology_diff.json", required=False)
    pre_stats = _snapshot_service_stats(pre)
    during_stats = _snapshot_service_stats(during)
    changed = _changed_services(diff)
    arr = np.zeros((len(services), len(TOPOLOGY_NAMES)), dtype=np.float32)
    for idx, service in enumerate(services):
        pre_item = pre_stats.get(service, {})
        during_item = during_stats.get(service, {})
        replicas_pre = _float(pre_item.get("replicas"))
        replicas_during = _float(during_item.get("replicas"))
        endpoint_pre = _float(pre_item.get("endpoint_count"))
        endpoint_during = _float(during_item.get("endpoint_count"))
        service_changed = changed.get(service, {})
        arr[idx] = [
            replicas_pre,
            replicas_during,
            replicas_during - replicas_pre,
            endpoint_pre,
            endpoint_during,
            endpoint_during - endpoint_pre,
            _float(service_changed.get("pod_added")),
            _float(service_changed.get("pod_removed")),
            _float(during_item.get("has_v2")),
        ]
    return np.nan_to_num(arr)


def fit_normalizer(items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    normalizer: Dict[str, Dict[str, np.ndarray]] = {}
    for key in ["metrics", "traces", "logs", "topology"]:
        values = np.concatenate([item[key].reshape(-1, item[key].shape[-1]) for item in items], axis=0)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        normalizer[key] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return normalizer


def normalize_item(item: Dict[str, Any], normalizer: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    out = dict(item)
    for key, stats in normalizer.items():
        out[key] = ((item[key] - stats["mean"]) / stats["std"]).astype(np.float32)
    return out


def save_bundle_metadata(bundle: BaselineDataBundle, out_dir: str) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "service_names": bundle.service_names,
        "edge_index": bundle.edge_index.cpu().tolist(),
        "feature_names": {
            "metrics": METRIC_NAMES,
            "traces": TRACE_NAMES,
            "logs": LOG_NAMES,
            "topology": STRICT_TOPOLOGY_NAMES if bundle.adapter_mode == "strict" else TOPOLOGY_NAMES,
            "phases": STRICT_PHASES if bundle.adapter_mode == "strict" else PHASES,
        },
        "adapter_mode": bundle.adapter_mode,
        "adapter_mode_notes": {
            "strict": (
                "Uses pre/during telemetry and static pre-fault topology only. "
                "Does not expose topology_diff, post metrics, during topology deltas, "
                "pod add/remove flags, or v2 flags to baselines."
            ),
            "leaky_dynamic": (
                "Legacy diagnostic mode that exposes dynamic topology deltas and post "
                "phase metrics. Do not use as the main baseline comparison."
            ),
        },
        "split_sizes": {key: len(value) for key, value in bundle.splits.items()},
    }
    with (path / "adapted_dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _snapshot_service_stats(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for dep in snapshot.get("deployments", []) or []:
        if not isinstance(dep, dict) or not dep.get("name"):
            continue
        service = str(dep["name"])
        item = stats.setdefault(service, {"replicas": 0.0, "endpoint_count": 0.0, "has_v2": 0.0})
        item["replicas"] = _float(dep.get("replicas"))
        version = str(dep.get("version", "") or "")
        if version == "v2" or service.endswith("-v2"):
            item["has_v2"] = 1.0
    for svc in snapshot.get("services", []) or []:
        if not isinstance(svc, dict) or not svc.get("name"):
            continue
        service = str(svc["name"])
        item = stats.setdefault(service, {"replicas": 0.0, "endpoint_count": 0.0, "has_v2": 0.0})
        item["endpoint_count"] = len(svc.get("endpoint_ips", []) or [])
    for pod in snapshot.get("pods", []) or []:
        if not isinstance(pod, dict):
            continue
        service = str(pod.get("service", "") or "")
        if not service:
            continue
        item = stats.setdefault(service, {"replicas": 0.0, "endpoint_count": 0.0, "has_v2": 0.0})
        version = str(pod.get("version", "") or "")
        if version == "v2" or service.endswith("-v2"):
            item["has_v2"] = 1.0
    return stats


def _changed_services(diff: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    text = json.dumps(diff, sort_keys=True).lower()
    result: Dict[str, Dict[str, float]] = {}
    # Topology diff schemas are not fully uniform in the dataset. Use a
    # conservative text scan only for service-local binary hints.
    for token in text.replace('"', " ").replace("'", " ").split():
        if "-" not in token:
            continue
        service = token.rsplit("-", 2)[0]
        if service:
            item = result.setdefault(service, {"pod_added": 0.0, "pod_removed": 0.0})
            if "added" in text or "add" in text:
                item["pod_added"] = 1.0
            if "removed" in text or "delete" in text:
                item["pod_removed"] = 1.0
    return result


def _static_degrees(services: Sequence[str]) -> Dict[str, Dict[str, float]]:
    degrees = {service: {"in_degree": 0.0, "out_degree": 0.0} for service in services}
    service_set = set(services)
    for source, target in KNOWN_EDGES:
        if source in service_set and target in service_set:
            degrees[source]["out_degree"] += 1.0
            degrees[target]["in_degree"] += 1.0
    return degrees


def _stat_value(value: Any) -> float:
    if isinstance(value, dict):
        return _float(value.get("avg", value.get("sum", value.get("max", 0.0))))
    return _float(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
