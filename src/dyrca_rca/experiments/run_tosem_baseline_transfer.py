"""Run six previously trained source-guided adapters on new control evidence.

This is a frozen cross-cohort stress test, not formal retraining on a new
TOSEM verified cohort. Labels for the control cohort are read after the
prediction file is written and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from dyrca_rca.baselines.evidence import _candidate_case, file_hash, load_release_evidence
from dyrca_rca.baselines.methods import (
    BASELINES, DISPLAY_NAMES, EadroAdapterModel, HolisticRCAAdapterModel,
    NeuralBaseline, create_classical, ranked_prediction,
)
from dyrca_rca.experiments.evaluate_interactive import rank_metrics
from dyrca_rca.interaction.observation import canonical, load_observation
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session
from dyrca_rca.interaction.temporal import PROTOCOL as TEMPORAL_PROTOCOL
from dyrca_rca.models.config import FIXED_GNN_CONFIG


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _neural_from_checkpoint(name: str, path: Path, release_split_hash: str, seed: int) -> NeuralBaseline:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    manifest = json.loads(checkpoint["manifest_json"])
    if (manifest.get("baseline") != name or manifest.get("release_split_hash") != release_split_hash
            or manifest.get("seed") != seed):
        raise ValueError("neural adapter checkpoint provenance mismatch: %s" % name)
    baseline = NeuralBaseline(name, seed=seed)
    hidden = FIXED_GNN_CONFIG.hidden_dim
    baseline.model = EadroAdapterModel(hidden) if name == "eadro" else HolisticRCAAdapterModel(hidden)
    baseline.model.load_state_dict(checkpoint["state_dict"], strict=True)
    baseline.model.eval()
    baseline.mean = checkpoint["normalizer_mean"]
    baseline.std = checkpoint["normalizer_std"]
    baseline.training_record = manifest["training_record"]
    return baseline


def _aggregate(rows: list[dict]) -> dict:
    return {"n": len(rows),
            "hit1": statistics.mean(row["hit1"] for row in rows) if rows else None,
            "hit3": statistics.mean(row["hit3"] for row in rows) if rows else None,
            "mrr": statistics.mean(row["mrr"] for row in rows) if rows else None,
            "decoy_top1_rate": statistics.mean(row["top1_is_decoy"] for row in rows) if rows else None}


def run(args: argparse.Namespace) -> dict:
    raw, out = Path(args.data_root).resolve(), Path(args.out_dir).resolve()
    if out == raw or raw in out.parents or args.limit < 1 or args.planner != "rule":
        raise ValueError("bounded positive limit, rule planner and separate output are required")
    torch.set_num_threads(1)
    source_root = Path(args.source_release).resolve()
    source_config = _read(source_root / "split.json")
    source_total = sum(len(source_config["split"][part]) for part in ("train", "val", "test", "regression"))
    source = load_release_evidence(str(source_root), source_config["data_root"],
                                   seed=args.seed, planner=args.planner, limit=source_total,
                                   system=args.system, dataset_id=source_config["dataset_id"])
    prior = Path(args.baseline_run).resolve()
    prior_config = _read(prior / "config.json")
    if (prior_config.get("release_root") != str(source_root) or prior_config.get("seed") != args.seed
            or prior_config.get("system") != args.system):
        raise ValueError("previous baseline run does not match source release")
    selected = sorted((item for item in _read(Path(args.summary))["experiments"]
                       if item.get("status") == "success"),
                      key=lambda item: item["experiment_id"])[:args.limit]
    if not selected:
        raise ValueError("no successful counterfactual cases")
    controls = []
    for item in selected:
        case = Path(item["output_dir"]).resolve()
        if case.parent != raw:
            raise ValueError("case outside data root")
        view = load_observation(case, args.dataset_id, args.system, max_rows=source.max_rows)
        _, tensor = replay(new_session(view), feature_protocol=TEMPORAL_PROTOCOL)
        candidate = _candidate_case(case.name, tensor, view["observation_version"])
        controls.append((case, candidate, view["observation_version"]))
    predictions = []
    checkpoints = {}
    for name in BASELINES:
        if name in ("eadro", "holisticrca"):
            path = prior / "checkpoints" / (name + ".pt")
            method = _neural_from_checkpoint(name, path, source.split_file_hash, args.seed)
            checkpoints[name] = file_hash(path)
        else:
            method = create_classical(name, seed=args.seed).fit(source.part("train"))
        frozen = method.state_dict() if name in ("eadro", "holisticrca") else None
        for case, candidate, view_hash in controls:
            row = ranked_prediction(candidate, method.predict(candidate))
            row.update({"method": DISPLAY_NAMES[name], "adapter": name,
                        "observation_version": view_hash, "source_cache_hash": candidate.source_cache_hash})
            predictions.append(row)
        if frozen is not None and any(not torch.equal(weight, method.state_dict()[key])
                                      for key, weight in frozen.items()):
            raise RuntimeError("neural adapter weights changed during inference")
    predictions.sort(key=lambda row: (row["method"], row["experiment"]))
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = out / "predictions.jsonl"
    prediction_path.write_text("".join(canonical(row) + "\n" for row in predictions), encoding="utf-8")
    prediction_hash = file_hash(prediction_path)

    # No new-control label or injection metadata is opened before this line.
    labels = {}
    decoys = {}
    benign = {}
    for case, _, _ in controls:
        labels[case.name] = _read(case / "ground_truth.json")["root_cause_service"]
        metadata = _read(case / "metadata.json")
        decoys[case.name] = set(filter(None, (metadata.get("topology_target_service"),
                                              metadata.get("topology_target_service_2"))))
        benign[case.name] = bool(metadata.get("is_benign"))
    scored = []
    for row in predictions:
        name = row["experiment"]
        if benign[name] != (labels[name] == "__benign__"):
            raise ValueError("benign label mismatch")
        ranking = row["prediction"]["ranked_node_ids"]
        metrics = {"hit1": None, "hit3": None, "mrr": None} if benign[name] else rank_metrics(row["prediction"], labels[name])
        scored.append({"experiment": name, "method": row["method"], "adapter": row["adapter"],
                       "is_benign": benign[name], "top1": ranking[0] if ranking else None,
                       "top1_is_decoy": float(bool(ranking) and ranking[0] in decoys[name]), **metrics})
    (out / "ranking_scores.jsonl").write_text("".join(canonical(row) + "\n" for row in scored), encoding="utf-8")
    summary = {}
    for name in BASELINES:
        method = DISPLAY_NAMES[name]
        normal = [row for row in scored if row["adapter"] == name and not row["is_benign"]]
        negative = [row for row in scored if row["adapter"] == name and row["is_benign"]]
        summary[method] = {"normal": _aggregate(normal),
                           "benign": {"n": len(negative),
                                      "decoy_top1_rate": statistics.mean(row["top1_is_decoy"] for row in negative)
                                      if negative else None}}
    report = {"status": "frozen_cross_cohort_pilot_not_formal_tosem_result",
              "system": args.system, "data_root": str(raw), "dataset_id": args.dataset_id,
              "seed": args.seed, "planner": args.planner, "limit": args.limit,
              "source_release": str(source_root), "source_split_sha256": source.split_file_hash,
              "prior_baseline_run": str(prior), "source_manifest_sha256": file_hash(prior / "source_manifest.json"),
              "checkpoint_hashes": checkpoints, "prediction_sha256_before_labels": prediction_hash,
              "prediction_rows": len(predictions), "summary": summary,
              "limitations": ["old training cohort and checkpoints; no new-cohort retraining",
                              "source-guided adapters, not upstream byte-for-byte reproductions",
                              "independent fault and healthy-decoy review not completed"]}
    (out / "report.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-release", required=True)
    p.add_argument("--baseline-run", required=True)
    p.add_argument("--system", required=True, choices=("boutique", "trainticket"))
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--planner", default="rule")
    print(canonical(run(p.parse_args())))


if __name__ == "__main__":
    main()
