"""Auditable E0/E1 and frozen-model counterfactual pilot for the TOSEM protocol.

This runner intentionally does not create formal table-13 results: the source
cohort lacks independent two-reviewer verification and the checkpoint was
trained on an earlier release. Test labels are opened only after predictions
have been written and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from dyrca_rca.experiments.evaluate_interactive import rank_metrics
from dyrca_rca.experiments.inference import load_checkpoint
from dyrca_rca.interaction.observation import anonymous_id, canonical, digest, load_observation
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session
from dyrca_rca.interaction.temporal import temporal_features


CHANGE_FIELDS = (
    "endpoint_added", "endpoint_removed", "endpoint_became_empty",
    "pod_added", "pod_removed", "version_added", "version_removed",
    "ready_drop", "ready_recovery", "service_appeared", "service_disappeared",
)
FAMILIES = {
    "endpoint_change": ("endpoint_pair_observed", ("endpoint_added", "endpoint_removed")),
    "empty_endpoint": ("endpoint_pair_observed", ("endpoint_became_empty",)),
    "pod_change": ("pod_pair_observed", ("pod_added", "pod_removed")),
    "version_change": ("version_pair_observed", ("version_added", "version_removed")),
    "readiness": ("ready_pair_observed", ("ready_drop", "ready_recovery")),
    "backend_resolution": (None, ("endpoint_backend_mismatch", "endpoint_backend_unresolved")),
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def _block_id(record: dict) -> str:
    # Correlated configurations with the same decoy change pattern cannot
    # straddle provisional train/validation/test groups.
    return digest([record["application"], record["mode"], record["decoy"],
                   record["decoy2"]])[:20]


def grouped_split(records: list[dict], seed: int) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for record in records:
        groups[_block_id(record)].append(record["case_id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    # Deterministic group-level 60/20/20 assignment; this is a pilot split.
    boundaries = (round(0.6 * len(keys)), round(0.8 * len(keys)))
    parts = {part: [] for part in ("train", "validation", "test")}
    for index, key in enumerate(keys):
        part = "train" if index < boundaries[0] else "validation" if index < boundaries[1] else "test"
        parts[part].extend(groups[key])
    return {part: sorted(names) for part, names in parts.items()}


def feature_coverage(feature_maps: dict[str, dict]) -> dict:
    result = {}
    for family, (mask, fields) in FAMILIES.items():
        eligible = active_cases = eligible_cases = 0
        for features in feature_maps.values():
            selected = [values for values in features.values()
                        if mask is None or values.get(mask, 0.0) > 0]
            if mask is None:
                selected = [values for values in selected
                            if values.get("topology_pair_observed", 0.0) > 0]
            eligible += len(selected)
            eligible_cases += bool(selected)
            active_cases += any(any(values.get(field, 0.0) > 0 for field in fields)
                                for values in selected)
        result[family] = {"eligible_object_comparisons": eligible,
                          "eligible_cases": eligible_cases, "active_cases": active_cases}
    return result


def change_ranking(candidates: list[str], features: dict[str, dict], scales: dict[str, float]) -> dict:
    scores = {service: sum(max(0.0, features.get(service, {}).get(field, 0.0)) / scales[field]
                           for field in CHANGE_FIELDS) for service in candidates}
    ranked = sorted(candidates, key=lambda service: (-scores[service], service))
    return {"ranked_node_ids": ranked, "scores": scores,
            "top_score": scores[ranked[0]] if ranked else None}


def training_scales(training_split: Path, *, system: str, max_cases: int | None = None) -> tuple[dict, dict]:
    released = _read(training_split)
    train = sorted(released["split"]["train"])
    if max_cases is not None:
        train = train[:max_cases]
    raw = Path(released["data_root"])
    if released["system"] != system:
        raise ValueError("training split application differs from pilot")
    positives = {field: [] for field in CHANGE_FIELDS}
    for name in train:
        # The scale reference depends only on topology. A one-row trace budget
        # avoids parsing large summaries while retaining the same snapshots.
        view = load_observation(raw / name, released["dataset_id"], system, max_rows=1)
        values, _ = temporal_features(view)
        for service in values.values():
            for field in CHANGE_FIELDS:
                value = service.get(field, 0.0)
                if value > 0:
                    positives[field].append(float(value))
    scales = {field: statistics.median(values) if values else 1.0
              for field, values in positives.items()}
    return scales, {"training_split": str(training_split.resolve()),
                    "training_split_sha256": _hash(training_split),
                    "reference_cases": len(train),
                    "positive_observations": {field: len(values) for field, values in positives.items()},
                    "zero_positive_scale_fallback": [field for field, values in positives.items() if not values]}


def _aggregate(rows: list[dict]) -> dict:
    return {"n": len(rows), "hit1": statistics.mean(r["hit1"] for r in rows) if rows else None,
            "hit3": statistics.mean(r["hit3"] for r in rows) if rows else None,
            "mrr": statistics.mean(r["mrr"] for r in rows) if rows else None,
            "decoy_top1_rate": statistics.mean(r["top1_is_decoy"] for r in rows) if rows else None}


def run(args: argparse.Namespace) -> dict:
    raw, out = Path(args.data_root).resolve(), Path(args.out_dir).resolve()
    if out == raw or raw in out.parents or args.limit < 1 or args.max_rows < 1:
        raise ValueError("positive limits and output outside raw data required")
    if args.planner != "rule":
        raise ValueError("this pilot supports only deterministic rule planning")
    torch.set_num_threads(1)
    batch = _read(Path(args.summary))
    selected = sorted((item for item in batch["experiments"] if item.get("status") == "success"),
                      key=lambda item: item["experiment_id"])[:args.limit]
    if not selected:
        raise ValueError("no successful control cases")
    scales, scale_manifest = training_scales(Path(args.training_split), system=args.system)
    model, model_name = load_checkpoint(args.checkpoint, "auto")
    if model.feature_protocol != "interactive-topology-trace-v3-temporal":
        raise ValueError("counterfactual pilot requires temporal-v3 checkpoint")
    model.eval()
    model_hash = _hash(Path(args.checkpoint))

    cases, verification, observations, predictions = [], [], [], []
    feature_maps = {}
    for item in selected:
        case = Path(item["output_dir"]).resolve()
        if case.parent != raw:
            raise ValueError("successful case lies outside data root: %s" % case)
        metadata = _read(case / "metadata.json")
        verifier = _read(case / "verification_report.json")
        if not metadata.get("counterfactual") or verifier.get("verification_result") not in ("CHANGED", "NOT_CHANGED"):
            raise ValueError("case lacks recognized control verification: %s" % case)
        view = load_observation(case, args.dataset_id, args.system, max_rows=args.max_rows)
        features, _ = temporal_features(view)
        result, tensor = replay(new_session(view), model, model_name)
        ranking = result["prediction"]
        candidates = ranking["ranked_node_ids"]
        if len(candidates) != len(set(candidates)):
            raise ValueError("duplicate candidate service")
        alternate = change_ranking(candidates, features, scales)
        target = str(metadata["target_service"])
        decoys = sorted(set(filter(None, (metadata.get("topology_target_service"),
                                         metadata.get("topology_target_service_2")))))
        mode = str(metadata["control_mode"])
        case_id = view["incident_id"]
        record = {"case_id": case_id, "application": args.system, "scenario_id": metadata["scenario_id"],
                  "mode": mode, "decoy": decoys[0] if decoys else "", "decoy2": decoys[1] if len(decoys) > 1 else "",
                  "configuration_id": digest([args.system, mode, target, decoys, metadata.get("fault_params")])[:20],
                  "target_service": target, "is_benign": bool(metadata.get("is_benign"))}
        record["split_block_id"] = _block_id(record)
        cases.append(record)
        changes = {service for service, values in features.items()
                   if any(values.get(field, 0.0) > 0 for field in CHANGE_FIELDS)}
        snapshots = {phase: _read(case / "topology" / (phase + ".json")) for phase in ("pre", "during")}
        endpoint_sets = {phase: {service.get("name"): service.get("endpoint_ips")
                                 for service in snapshot.get("services", [])}
                         for phase, snapshot in snapshots.items()}
        decoy_has_endpoints = all(isinstance(endpoint_sets[phase].get(decoy), list) and
                                  bool(endpoint_sets[phase][decoy])
                                  for phase in ("pre", "during") for decoy in decoys)
        verification.append({"case_id": case_id, "scheduler_status": "success",
                             "topology_verifier": verifier.get("verification_result"),
                             "target_topology_unchanged": target not in changes,
                             "decoy_topology_changed": any(decoy in changes for decoy in decoys),
                             "decoy_has_endpoints_pre_and_during": decoy_has_endpoints,
                             "independent_fault_effect_review": "not_recorded",
                             "independent_decoy_health_review": "not_recorded"})
        observations.append({"case_id": case_id, "observation_version": view["observation_version"],
                             "cutoff": view["as_of"], "trace_status": view["modalities"]["traces"],
                             "candidate_count": len(candidates), "tensor_nodes": len(tensor["node_ids"]),
                             "input_sha256": {name: _hash(case / name) for name in (
                                 "topology/pre.json", "topology/during.json", "traces/trace_summary.csv")},
                             "ingest_time_status": "unavailable"})
        feature_maps[case_id] = features
        for method, prediction in (("DY-RCA-frozen-v3", ranking), ("Change-only", alternate)):
            predictions.append({"case_id": case_id, "method": method,
                                "ranked_node_ids": prediction["ranked_node_ids"],
                                "scores": prediction["scores"], "model_hash": model_hash if method.startswith("DY-") else None,
                                "observation_version": view["observation_version"]})

    if not all(row["target_topology_unchanged"] and row["decoy_topology_changed"] for row in verification):
        raise ValueError("counterfactual topology invariant failed")
    out.mkdir(parents=True, exist_ok=True)
    _jsonl(out / "collection" / "cases.jsonl", cases)
    _jsonl(out / "collection" / "verification.jsonl", verification)
    _jsonl(out / "collection" / "observation_manifest.jsonl", observations)
    review_queue = []
    for item, check in zip(selected, verification):
        flags = []
        if check["topology_verifier"] != "CHANGED":
            flags.append("topology_verifier_disagreement")
        if not check["decoy_has_endpoints_pre_and_during"]:
            flags.append("decoy_endpoint_absence")
        review_queue.append({"case_id": check["case_id"], "source_dir": item["output_dir"],
                             "required_reviews": ["target_fault_effect", "decoy_health"],
                             "independent_reviewers_required": 2, "flags": flags})
    _jsonl(out / "collection" / "review_queue.jsonl", review_queue)
    split = grouped_split(cases, args.seed)
    _write(out / "collection" / "split_manifest.json", {"seed": args.seed, "status": "pilot_provisional",
                                                      "group_unit": "application+mode+decoys", "split": split})
    _write(out / "audit" / "feature_coverage.json", feature_coverage(feature_maps))
    _write(out / "ranking" / "change_scales.json", {"scales": scales, "manifest": scale_manifest})
    _jsonl(out / "ranking" / "predictions.jsonl", predictions)
    prediction_hash = _hash(out / "ranking" / "predictions.jsonl")

    # Test labels are read for the first time only after predictions are fixed.
    labels = {}
    label_rows = []
    for item in selected:
        case = Path(item["output_dir"]).resolve()
        truth = _read(case / "ground_truth.json")["root_cause_service"]
        case_id = anonymous_id(args.dataset_id, case.name)
        labels[case_id] = truth
        label_rows.append({"case_id": case_id, "root_cause_service": truth,
                           "label_source_sha256": _hash(case / "ground_truth.json")})
    _jsonl(out / "collection" / "labels.jsonl", label_rows)
    by_case = {record["case_id"]: record for record in cases}
    scored = []
    for row in predictions:
        record = by_case[row["case_id"]]
        ranked = row["ranked_node_ids"]
        benign = record["is_benign"]
        truth = labels[row["case_id"]]
        if benign != (truth == "__benign__"):
            raise ValueError("benign label mismatch")
        metrics = {"hit1": None, "hit3": None, "mrr": None} if benign else rank_metrics(row, truth)
        scored.append({"case_id": row["case_id"], "method": row["method"],
                       "mode": record["mode"], "is_benign": benign,
                       "top1": ranked[0] if ranked else None,
                       "top1_is_decoy": float(bool(ranked) and ranked[0] in
                                              {record["decoy"], record["decoy2"]}),
                       **metrics})
    _jsonl(out / "ranking" / "ranking_scores.jsonl", scored)
    summary = {}
    strict_ids = {row["case_id"] for row in verification
                  if row["topology_verifier"] == "CHANGED" and
                  row["decoy_has_endpoints_pre_and_during"]}
    proxy_summary = {"definition": "verifier CHANGED and nonempty decoy endpoints in pre/during; not independent health verification",
                     "eligible_cases": len(strict_ids), "methods": {}}
    for method in sorted({row["method"] for row in scored}):
        normal = [row for row in scored if row["method"] == method and not row["is_benign"]]
        benign = [row for row in scored if row["method"] == method and row["is_benign"]]
        summary[method] = {"normal": _aggregate(normal),
                           "benign": {"n": len(benign), "decoy_top1_rate":
                                      statistics.mean(row["top1_is_decoy"] for row in benign) if benign else None},
                           "by_mode": {mode: _aggregate([row for row in normal if row["mode"] == mode])
                                       for mode in sorted({row["mode"] for row in normal})}}
        proxy_summary["methods"][method] = {
            "normal": _aggregate([row for row in normal if row["case_id"] in strict_ids]),
            "benign_n": sum(row["case_id"] in strict_ids for row in benign),
        }
    _write(out / "ranking" / "summary.json", summary)
    _write(out / "shortcut" / "proxy_healthy_summary.json", proxy_summary)
    integrity = {"population": len(cases), "normal": sum(not c["is_benign"] for c in cases),
                 "benign": sum(c["is_benign"] for c in cases),
                 "target_unchanged": all(v["target_topology_unchanged"] for v in verification),
                 "decoy_changed": all(v["decoy_topology_changed"] for v in verification),
                 "split_blocks_disjoint": all(not ({_block_id(c) for c in cases if c["case_id"] in split[a]} &
                                               {_block_id(c) for c in cases if c["case_id"] in split[b]})
                                              for a, b in (("train", "validation"), ("train", "test"),
                                                           ("validation", "test"))),
                 "prediction_sha256_before_labels": prediction_hash,
                 "prediction_rows": len(predictions),
                 "label_rows": len(label_rows), "independent_review_complete": False}
    integrity["verifier_not_changed_cases"] = sum(v["topology_verifier"] != "CHANGED" for v in verification)
    integrity["decoy_missing_endpoints_cases"] = sum(not v["decoy_has_endpoints_pre_and_during"]
                                                    for v in verification)
    _write(out / "collection" / "integrity_checks.json", integrity)
    _write(out / "protocol.json", {"status": "pilot_not_formal_tosem_result",
                                    "data_root": str(raw), "dataset_id": args.dataset_id,
                                    "system": args.system, "seed": args.seed, "planner": args.planner,
                                    "limit": args.limit, "max_rows": args.max_rows,
                                    "checkpoint": str(Path(args.checkpoint).resolve()),
                                    "checkpoint_sha256": model_hash,
                                    "cutoff_rule": "during_snapshot_timestamp; no recorded t0+120 or ingest times",
                                    "candidate_policy": "same frozen model candidates for both methods",
                                    "excluded_model_inputs": ["metadata", "ground_truth", "post topology", "verification_report"],
                                    "limitations": ["old checkpoint; no retraining on this cohort",
                                                    "no two-reviewer independent fault/decoy verification",
                                                    "topology verifier and endpoint-health discrepancies are retained, not silently excluded",
                                                    "benign cases have no root label and no abstention head",
                                                    "not a nine-method formal comparison"]})
    return {"integrity": integrity, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-split", required=True)
    parser.add_argument("--system", required=True, choices=("boutique", "trainticket"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", default="rule")
    print(canonical(run(parser.parse_args())))


if __name__ == "__main__":
    main()
