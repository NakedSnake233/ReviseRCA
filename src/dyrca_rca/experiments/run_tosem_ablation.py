"""Run the five fixed-config representation/weight ablations for TOSEM E4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from dyrca_rca.dataset.materialize import feature_names
from dyrca_rca.experiments.evaluate_interactive import rank_metrics
from dyrca_rca.experiments.train import _candidate_mask, _forward, train_model
from dyrca_rca.interaction.temporal import PROTOCOL, FEATURES


CHANGE_VALUES = {
    "endpoint_added", "endpoint_removed", "endpoint_became_empty", "pod_added",
    "pod_removed", "version_added", "version_removed", "ready_drop", "ready_recovery",
    "service_appeared", "service_disappeared",
}
MASK_VALUES = {"topology_pair_observed", "endpoint_pair_observed", "pod_pair_observed",
               "version_pair_observed", "ready_pair_observed"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _receipt_index(cache: Path) -> Dict[str, Path]:
    index = {}
    for receipt in cache.glob("*.json"):
        data = json.loads(receipt.read_text())
        tensor = receipt.with_suffix(".pt")
        if tensor.is_file():
            index[str(data.get("sha256"))] = tensor
    return index


def _items(release: Path, protocol: str, split: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    report = json.loads((release / "report.json").read_text())
    hashes = report["cache_hashes"][protocol]
    receipts = _receipt_index(release / "cache")
    names = [name for part in ("train", "val", "test") for name in split[part]]
    result = {}
    for name in names:
        path = receipts[hashes[name]]
        item = torch.load(str(path), map_location="cpu")
        item.pop("label", None)
        result[name] = item
    return result


def _label(raw: Path, name: str, item: Mapping[str, Any]) -> torch.Tensor:
    truth = json.loads((raw / name / "ground_truth.json").read_text())["root_cause_service"]
    for index, node in enumerate(item["node_ids"]):
        if node == truth and _candidate_mask(item)[index]:
            return torch.tensor(index, dtype=torch.long)
    raise ValueError(f"root service not in candidates: {name} -> {truth}")


def _variant(item: Mapping[str, Any], name: str) -> Dict[str, Any]:
    copied = dict(item)
    copied["x"] = item["x"].clone()
    copied["edge_attr"] = item["edge_attr"].clone()
    names = list(item["metadata"]["node_feature_names"])
    if name == "full_temporal_v3":
        return copied
    if name == "current_state_v2":
        keep = list(range(25))
    else:
        temporal = names[25:]
        if name == "without_change_values":
            keep_temporal = [i for i, field in enumerate(temporal) if field not in CHANGE_VALUES]
        elif name == "without_temporal_masks":
            keep_temporal = [i for i, field in enumerate(temporal) if field not in MASK_VALUES]
        elif name == "constant_positive_edge_weights":
            copied["edge_attr"][:, -1] = 1.0
            return copied
        else:
            raise ValueError(name)
        keep = list(range(25)) + [25 + i for i in keep_temporal]
    copied["x"] = item["x"][:, keep]
    copied["metadata"] = dict(item["metadata"], node_feature_names=[names[i] for i in keep],
                               ablation=name)
    return copied


def run(args: argparse.Namespace) -> Dict[str, Any]:
    raw = Path(args.data_root).resolve()
    release = Path(args.release_root).resolve()
    out = Path(args.out_dir).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("output must be outside raw data")
    split_config = json.loads((release / "split.json").read_text())
    split = split_config["split"]
    v3 = _items(release, PROTOCOL, split)
    # v2 is retained in the same frozen release cache by optimize_harness.
    v2 = _items(release, "interactive-topology-trace-v2-missingness", split)
    variants = ("current_state_v2", "full_temporal_v3", "without_change_values",
                "without_temporal_masks", "constant_positive_edge_weights")
    prepared = {variant: {name: _variant((v2 if variant == "current_state_v2" else v3)[name], variant)
                          for name in v3} for variant in variants}
    predictions = []
    checkpoints = {}
    for variant in variants:
        train = []
        val = []
        for part, destination in (("train", train), ("val", val)):
            for name in split[part]:
                item = dict(prepared[variant][name])
                item["label"] = _label(raw, name, item)
                destination.append(item)
        model = train_model(train, val, "weighted_graphsage", args.seed)
        checkpoint = out / "checkpoints" / (variant + ".pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"variant": variant, "seed": args.seed,
                    "state_dict": model.state_dict(),
                    "feature_names": prepared[variant][next(iter(prepared[variant]))]["metadata"]["node_feature_names"]}, str(checkpoint))
        checkpoints[variant] = _sha(checkpoint)
        for name in split["test"]:
            item = prepared[variant][name]
            with torch.no_grad():
                scores = _forward(model, item, "weighted_graphsage")
            indices = torch.nonzero(_candidate_mask(item), as_tuple=False).view(-1).tolist()
            indices.sort(key=lambda i: (-float(scores[i]), str(item["node_ids"][i])))
            predictions.append({"variant": variant, "experiment": name,
                                "ranked_node_ids": [item["node_ids"][i] for i in indices],
                                "scores": {item["node_ids"][i]: float(scores[i]) for i in indices}})
    out.mkdir(parents=True, exist_ok=True)
    # Predictions are committed before test labels are opened.
    (out / "ablation_predictions.jsonl").write_text("".join(_canonical(row) + "\n" for row in predictions))
    results = []
    for row in predictions:
        truth = json.loads((raw / row["experiment"] / "ground_truth.json").read_text())["root_cause_service"]
        rank = row["ranked_node_ids"].index(truth) + 1 if truth in row["ranked_node_ids"] else None
        results.append(dict(row, hit1=float(rank == 1), hit3=float(rank is not None and rank <= 3),
                            mrr=1.0 / rank if rank else 0.0, rank=rank))
    (out / "ablation_results.jsonl").write_text("".join(_canonical(row) + "\n" for row in results))
    summary = []
    for variant in variants:
        rows = [row for row in results if row["variant"] == variant]
        summary.append({"variant": variant, "n": len(rows),
                        "hit1": sum(r["hit1"] for r in rows) / len(rows) if rows else None,
                        "hit3": sum(r["hit3"] for r in rows) / len(rows) if rows else None,
                        "mrr": sum(r["mrr"] for r in rows) / len(rows) if rows else None})
    report = {"status": "measured", "seed": args.seed, "split": {k: len(v) for k, v in split.items()},
              "variants": summary, "checkpoint_hashes": checkpoints,
              "full_temporal_v3_shared_with_ranking": True}
    (out / "ablation_summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", default="rule", choices=("rule",))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
