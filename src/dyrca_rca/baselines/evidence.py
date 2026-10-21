"""Load the exact label-free evidence used by the temporal-v3 release.

The loader intentionally consumes the release cache instead of reparsing raw data.
That gives every baseline the same candidate inventory and tensorized pre/during
evidence that the released DY-RCA checkpoint saw.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from dyrca_rca.dataset.materialize import feature_names
from dyrca_rca.interaction.temporal import PROTOCOL as TEMPORAL_PROTOCOL


BASELINE_PROTOCOL = "dynamic-evidence-baselines-v1"
PARTS = ("train", "val", "test", "regression")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_indices(item: Mapping[str, Any]) -> List[int]:
    """Mirror the released candidate mask exactly."""
    result = []
    for index, node_id in enumerate(item.get("node_ids", [])):
        text = str(node_id)
        if not text.startswith("pod:") and not text.startswith("blackhole:") and ":" not in text:
            result.append(index)
    return result or list(range(len(item.get("node_ids", []))))


@dataclass
class CandidateCase:
    name: str
    incident_id: str
    task: str
    node_names: List[str]
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    directed_edges: List[Tuple[str, str, float]]
    source_cache_hash: str

    def feature_map(self) -> Dict[str, torch.Tensor]:
        return {name: self.x[index] for index, name in enumerate(self.node_names)}


@dataclass
class ReleaseEvidence:
    release_root: Path
    data_root: Path
    seed: int
    planner: str
    system: str
    dataset_id: str
    max_rows: int
    split: Dict[str, List[str]]
    cases: Dict[str, CandidateCase]
    feature_names: List[str]
    edge_feature_names: List[str]
    cache_hashes: Dict[str, str]
    split_file_hash: str
    report_file_hash: str

    def part(self, name: str) -> List[CandidateCase]:
        return [self.cases[item] for item in self.split[name]]


def load_release_evidence(release_root: str,
                          data_root: str,
                          seed: int = 42,
                          planner: str = "rule",
                          limit: int = 300,
                          system: str | None = None,
                          dataset_id: str | None = None) -> ReleaseEvidence:
    root = Path(release_root).resolve()
    raw = Path(data_root).resolve()
    split_path, report_path = root / "split.json", root / "report.json"
    if not split_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("temporal release split/report is missing")
    config = json.loads(split_path.read_text())
    report = json.loads(report_path.read_text())
    if int(config.get("seed", -1)) != int(seed):
        raise ValueError("requested seed differs from the frozen release seed")
    if config.get("planner") != planner:
        raise ValueError("requested planner differs from the frozen release planner")
    frozen_system = str(config.get("system", "boutique"))
    frozen_dataset = str(config.get("dataset_id", "dyrca-v1"))
    if system is not None and system != frozen_system:
        raise ValueError("requested system differs from the frozen release system")
    if dataset_id is not None and dataset_id != frozen_dataset:
        raise ValueError("requested dataset ID differs from the frozen release dataset")
    if Path(config.get("data_root", "")).resolve() != raw:
        raise ValueError("data root differs from the frozen release data root")
    frozen = {part: list(config["split"][part]) for part in PARTS}
    total = sum(len(values) for values in frozen.values())
    if limit < 4 or limit > total:
        raise ValueError("limit must be between four and the frozen population size")
    selected = _bounded_split(frozen, limit)

    expected_node, expected_edge = feature_names(TEMPORAL_PROTOCOL)
    cache_hashes = report.get("cache_hashes", {}).get(TEMPORAL_PROTOCOL, {})
    receipt_index = _receipt_index(root / "cache")
    cases: Dict[str, CandidateCase] = {}
    selected_names = [name for part in PARTS for name in selected[part]]
    for name in selected_names:
        expected_hash = cache_hashes.get(name)
        if not expected_hash or expected_hash not in receipt_index:
            raise ValueError("release cache receipt missing for %s" % name)
        path = receipt_index[expected_hash]
        if file_hash(path) != expected_hash:
            raise ValueError("release cache content hash mismatch for %s" % name)
        item = torch.load(str(path), map_location="cpu")
        _validate_item(item, expected_node, expected_edge, name)
        cases[name] = _candidate_case(name, item, expected_hash)

    return ReleaseEvidence(
        release_root=root,
        data_root=raw,
        seed=seed,
        planner=planner,
        system=frozen_system,
        dataset_id=frozen_dataset,
        max_rows=int(config["max_rows"]),
        split=selected,
        cases=cases,
        feature_names=list(expected_node),
        edge_feature_names=list(expected_edge),
        cache_hashes={name: cache_hashes[name] for name in selected_names},
        split_file_hash=file_hash(split_path),
        report_file_hash=file_hash(report_path),
    )


def _bounded_split(split: Mapping[str, Sequence[str]], limit: int) -> Dict[str, List[str]]:
    total = sum(len(split[part]) for part in PARTS)
    if limit == total:
        return {part: list(split[part]) for part in PARTS}
    # Allocate a deterministic, roughly proportional smoke subset while retaining
    # every partition. Full paper runs always use all 300 cases.
    raw_quota = {part: limit * len(split[part]) / float(total) for part in PARTS}
    quota = {part: max(1, int(raw_quota[part])) for part in PARTS}
    while sum(quota.values()) > limit:
        choices = [part for part in PARTS if quota[part] > 1]
        part = max(choices, key=lambda key: (quota[key] - raw_quota[key], quota[key], key))
        quota[part] -= 1
    while sum(quota.values()) < limit:
        choices = [part for part in PARTS if quota[part] < len(split[part])]
        part = max(choices, key=lambda key: (raw_quota[key] - quota[key], len(split[key]), key))
        quota[part] += 1
    return {part: list(split[part])[:quota[part]] for part in PARTS}


def _receipt_index(cache_dir: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for receipt in sorted(cache_dir.glob("*.json")):
        data = json.loads(receipt.read_text())
        tensor = receipt.with_suffix(".pt")
        if tensor.is_file():
            result[str(data.get("sha256", ""))] = tensor
    return result


def _validate_item(item: Mapping[str, Any], node_features: Sequence[str],
                   edge_features: Sequence[str], name: str) -> None:
    metadata = item.get("metadata", {})
    if "label" in item:
        raise ValueError("label leaked into cached evidence for %s" % name)
    if metadata.get("feature_protocol") != TEMPORAL_PROTOCOL:
        raise ValueError("unexpected feature protocol for %s" % name)
    if metadata.get("node_feature_names") != list(node_features):
        raise ValueError("node feature schema mismatch for %s" % name)
    if metadata.get("edge_feature_names") != list(edge_features):
        raise ValueError("edge feature schema mismatch for %s" % name)
    x, edge_index, edge_attr = item["x"], item["edge_index"], item["edge_attr"]
    if x.shape != (len(item["node_ids"]), len(node_features)):
        raise ValueError("node tensor shape mismatch for %s" % name)
    if edge_index.shape[0] != 2 or edge_attr.shape != (edge_index.shape[1], len(edge_features)):
        raise ValueError("edge tensor shape mismatch for %s" % name)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(edge_attr).all()):
        raise ValueError("non-finite evidence for %s" % name)
    if not candidate_indices(item):
        raise ValueError("no service candidates for %s" % name)


def _candidate_case(name: str, item: Mapping[str, Any], cache_hash: str) -> CandidateCase:
    keep = candidate_indices(item)
    names = [str(item["node_ids"][index]) for index in keep]
    local = {node: index for index, node in enumerate(names)}
    directed: Dict[Tuple[str, str], float] = {}
    for edge in item["graph"].get("edges", []):
        if str(edge.get("edge_type", "dependency")) != "dependency":
            continue
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in local or target not in local:
            continue
        attrs = edge.get("attrs", {}) if isinstance(edge.get("attrs"), dict) else {}
        value = attrs.get("final_weight", edge.get("weight", 1.0))
        try:
            weight = max(float(value), 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        directed[(source, target)] = max(weight, directed.get((source, target), 0.0))

    # Neural adapters use symmetric message passing, matching release tensorization.
    messages: Dict[Tuple[int, int], float] = {}
    for (source, target), weight in directed.items():
        a, b = local[source], local[target]
        messages[(a, b)] = max(weight, messages.get((a, b), 0.0))
        messages[(b, a)] = max(weight, messages.get((b, a), 0.0))
    pairs = sorted(messages)
    edge_index = (torch.tensor(pairs, dtype=torch.long).t().contiguous()
                  if pairs else torch.zeros((2, 0), dtype=torch.long))
    edge_weight = torch.tensor([messages[pair] for pair in pairs], dtype=torch.float32)
    return CandidateCase(
        name=name,
        incident_id=str(item["metadata"]["experiment_id"]),
        task=name[:2].upper(),
        node_names=names,
        x=item["x"][torch.tensor(keep, dtype=torch.long)].detach().cpu().clone(),
        edge_index=edge_index,
        edge_weight=edge_weight,
        directed_edges=[(source, target, weight)
                        for (source, target), weight in sorted(directed.items())],
        source_cache_hash=cache_hash,
    )
