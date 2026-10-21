"""Run six prespecified RCA baselines on the frozen temporal-v3 evidence.

Predictions for test and regression cases are committed to disk before either
partition's ground-truth files are opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from dyrca_rca.baselines.evidence import (
    BASELINE_PROTOCOL,
    PARTS,
    CandidateCase,
    canonical,
    file_hash,
    load_release_evidence,
    _candidate_case,
)
from dyrca_rca.baselines.methods import (
    BASELINES,
    DISPLAY_NAMES,
    NeuralBaseline,
    create_classical,
    ranked_prediction,
)
from dyrca_rca.experiments.evaluate_interactive import rank_metrics
from dyrca_rca.interaction.observation import load_observation
from dyrca_rca.interaction.state import new_session
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.temporal import PROTOCOL as TEMPORAL_PROTOCOL
from dyrca_rca.experiments.inference import load_checkpoint, predict_observed
from dyrca_rca.system_config import application_id


CLASSICAL = ("nsigma", "baro", "circa", "cloudranger", "changeonly")
NEURAL = ("eadro", "holisticrca", "nodeonly")


SOURCE_MANIFEST = {
    "rcaeval": {
        "url": "https://github.com/phamquiluan/RCAEval",
        "commit": "7600283af1ea5e2e9fff6f07124951d0e989de42",
        "methods": {
            "nsigma": "RCAEval/e2e/__init__.py: maximum StandardScaler score",
            "baro": "RCAEval/e2e/baro.py: maximum RobustScaler score",
            "circa": "RCAEval/graph_heads/rht.py: additive-noise regression residual test",
            "cloudranger": "RCAEval/e2e/cloudranger.py: relation-weighted second-order random walk",
        },
        "licenses": {
            "nsigma": "baselines/external/RCAEval/LICENSES/LICENSE-NSigma",
            "baro": "baselines/external/RCAEval/LICENSES/LICENSE-BARO",
            "circa": "baselines/external/RCAEval/LICENSES/LICENSE-CIRCA",
            "cloudranger": "baselines/external/RCAEval/LICENSES/LICENSE-DyCause",
        },
    },
    "eadro": {
        "url": "https://github.com/FeiGSSS/Eadro",
        "commit": "7d69281a80adce60067aa858729746fc6a362f53",
        "mechanism": "multi-source encoders, graph aggregation, and localization",
        "license_status": "no LICENSE file found in the pinned checkout",
    },
    "holisticrca": {
        "url": "https://github.com/baiyanquan/HolisticRCA",
        "commit": "d7d829c81eecfc1aa733ccd913f0b2d17bf1df33",
        "mechanism": "modality representation/integration and relation-graph fusion",
        "license_status": "no LICENSE file found in the pinned checkout",
    },
    "changeonly": {
        "mechanism": "training-reference-scaled temporal structural change magnitudes",
        "source": "src/dyrca_rca/baselines/methods.py:ChangeOnlyBaseline",
    },
    "nodeonly": {
        "mechanism": "two-layer fixed-width node MLP without neighbor aggregation",
        "source": "src/dyrca_rca/baselines/methods.py:NodeOnlyAdapterModel",
    },
    "adaptation_boundary": {
        "retained": "ranking heads and high-level encoder/graph-fusion mechanisms",
        "replaced": "dataset-specific readers, native time windows, graph discovery, and output candidate mapping",
        "shared_input": "frozen temporal-v3 node features, observed dependency graph, and per-case service candidates",
        "claim": "source-guided adapters, not byte-for-byte upstream reproductions",
    },
}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.out_dir).resolve()
    raw = Path(args.data_root).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("output directory must be outside raw data")
    selected = args.baseline or list(BASELINES)
    evaluation_parts = ("val", "test", "regression") if getattr(args, "include_validation", False) else ("test", "regression")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate baseline selection")
    torch.set_num_threads(1)
    evidence = load_release_evidence(
        args.release_root, args.data_root, seed=args.seed,
        planner=args.planner, limit=args.limit, system=args.system,
        dataset_id=args.dataset_id,
    )
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": BASELINE_PROTOCOL,
        "release_root": str(evidence.release_root),
        "data_root": str(evidence.data_root),
        "out_dir": str(out),
        "seed": args.seed,
        "planner": args.planner,
        "system": evidence.system,
        "dataset_id": evidence.dataset_id,
        "limit": args.limit,
        "max_rows": evidence.max_rows,
        "baselines": selected,
        "include_validation": getattr(args, "include_validation", False),
        "controls_root": str(Path(args.controls_root).resolve()) if args.controls_root else None,
        "controls_audit": str(Path(args.controls_audit).resolve()) if args.controls_audit else None,
        "controls_limit": args.controls_limit,
        "split": evidence.split,
        "split_sizes": {part: len(evidence.split[part]) for part in PARTS},
    }
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing baseline run has a different configuration")
    config_path.write_text(canonical(config) + "\n")
    print("Frozen dynamic-evidence split: " + canonical(config["split_sizes"]), flush=True)

    source_manifest = dict(SOURCE_MANIFEST)
    source_manifest["local_source_hashes"] = _source_hashes()
    source_manifest["upstream_checkout_hashes_verified"] = _verify_upstream_checkouts()
    (out / "source_manifest.json").write_text(canonical(source_manifest) + "\n")
    evidence_manifest = {
        "feature_protocol": "interactive-topology-trace-v3-temporal",
        "node_feature_names": evidence.feature_names,
        "edge_feature_names": evidence.edge_feature_names,
        "release_split_hash": evidence.split_file_hash,
        "release_report_hash": evidence.report_file_hash,
        "cache_hashes": evidence.cache_hashes,
        "candidate_policy": "exact released service mask; per-case inventory",
        "excluded_inputs": ["post topology", "injection parameters", "target_service metadata",
                            "test/regression labels before prediction"],
    }
    (out / "evidence_manifest.json").write_text(canonical(evidence_manifest) + "\n")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
    }
    (out / "runtime.json").write_text(canonical(runtime) + "\n")

    # Only training and validation labels are read before prediction, and only for
    # the supervised neural adapters.
    supervised = any(name in NEURAL for name in selected)
    train_labels = _labels(raw, evidence.split["train"]) if supervised else {}
    val_labels = _labels(raw, evidence.split["val"]) if supervised else {}
    pre_prediction_label_hashes = {
        part: {name: file_hash(raw / name / "ground_truth.json") for name in evidence.split[part]}
        for part in ("train", "val") if supervised
    }

    predictions: List[Dict[str, Any]] = []
    training_records: Dict[str, Any] = {}
    checkpoint_hashes: Dict[str, str] = {}
    control_cases = _load_controls(args, evidence) if args.controls_root else []
    train_cases, val_cases = evidence.part("train"), evidence.part("val")
    for baseline in selected:
        print("Fitting %s on shared evidence" % DISPLAY_NAMES[baseline], flush=True)
        if baseline in CLASSICAL:
            method = create_classical(baseline, seed=args.seed).fit(train_cases)
            frozen = None
        else:
            method = NeuralBaseline(baseline, seed=args.seed).fit(
                train_cases, val_cases, train_labels, val_labels,
            )
            frozen = {key: value.clone() for key, value in method.state_dict().items()}
            checkpoint = out / "checkpoints" / (baseline + ".pt")
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save({
                "manifest_json": canonical({
                    "baseline": baseline,
                    "display_name": DISPLAY_NAMES[baseline],
                    "protocol": BASELINE_PROTOCOL,
                    "seed": args.seed,
                    "training_record": method.training_record,
                    "release_split_hash": evidence.split_file_hash,
                    "train_cache_hashes": {name: evidence.cache_hashes[name]
                                           for name in evidence.split["train"]},
                    "val_cache_hashes": {name: evidence.cache_hashes[name]
                                         for name in evidence.split["val"]},
                }),
                "state_dict": method.state_dict(),
                "normalizer_mean": method.mean.detach().cpu(),
                "normalizer_std": method.std.detach().cpu(),
            }, str(checkpoint))
            training_records[baseline] = method.training_record
            checkpoint_hashes[baseline] = file_hash(checkpoint)
        for part in evaluation_parts:
            for case in evidence.part(part):
                row = ranked_prediction(case, method.predict(case))
                row.update({
                    "split": part,
                    "method": DISPLAY_NAMES[baseline],
                    "adapter": baseline,
                    "source_cache_hash": case.source_cache_hash,
                })
                predictions.append(row)
        for case in control_cases:
            row = ranked_prediction(case, method.predict(case))
            row.update({"split": "control", "method": DISPLAY_NAMES[baseline],
                        "adapter": baseline, "source_cache_hash": case.source_cache_hash,
                        "control_scenario": case.task})
            predictions.append(row)
        if frozen is not None:
            current = method.state_dict()
            if any(not torch.equal(value, current[key]) for key, value in frozen.items()):
                raise RuntimeError("%s weights changed during inference" % baseline)

    # The released method's label-free predictions are copied for the exact same
    # selected cases. This avoids rerunning or silently changing its checkpoint.
    predictions.extend(_released_predictions(evidence, evaluation_parts))
    if control_cases:
        predictions.extend(_released_dyrca_controls(evidence, control_cases))
    predictions.sort(key=lambda row: (row["split"], row["method"], row["experiment"]))
    prediction_path = out / "predictions.jsonl"
    prediction_path.write_text("".join(canonical(row) + "\n" for row in predictions))
    prediction_hash = file_hash(prediction_path)
    print("Locked %d label-free predictions (%s)" % (len(predictions), prediction_hash[:12]), flush=True)

    # Test/regression labels are first opened below this point.
    evaluation_labels = {
        part: _labels(raw, evidence.split[part]) for part in evaluation_parts
    }
    post_prediction_label_hashes = {
        part: {name: file_hash(raw / name / "ground_truth.json") for name in evidence.split[part]}
        for part in evaluation_parts
    }
    label_access = {
        "prediction_artifact_hash_before_evaluation": prediction_hash,
        "pre_prediction_partitions": sorted(pre_prediction_label_hashes),
        "pre_prediction_hashes": pre_prediction_label_hashes,
        "post_prediction_partitions": list(evaluation_parts),
        "post_prediction_hashes": post_prediction_label_hashes,
    }
    (out / "label_access.json").write_text(canonical(label_access) + "\n")

    results = []
    for row in predictions:
        if row["split"] == "control":
            case_root = Path(args.controls_root) / row["experiment"]
            results.append(_score_control(row, case_root))
        else:
            truth = evaluation_labels[row["split"]][row["experiment"]]
            results.append(dict(row, **rank_metrics(row["prediction"], truth)))
    (out / "results.jsonl").write_text("".join(canonical(row) + "\n" for row in results))
    summary = _summary(results)
    _write_csv(out / "summary.csv", summary)
    main = _main_comparison(summary, selected)
    _write_csv(out / "main_comparison.csv", main)
    paired = _paired_comparisons(results, [DISPLAY_NAMES[name] for name in selected])
    _write_csv(out / "paired_comparisons.csv", paired)

    report = {
        "config": config,
        "runtime": runtime,
        "prediction_hash": prediction_hash,
        "summary": summary,
        "main_comparison": main,
        "paired_comparisons": paired,
        "training_records": training_records,
        "checkpoint_hashes": checkpoint_hashes,
        "source_manifest_hash": file_hash(out / "source_manifest.json"),
        "evidence_manifest_hash": file_hash(out / "evidence_manifest.json"),
        "weights_unchanged_during_inference": True,
    }
    (out / "report.json").write_text(canonical(report) + "\n")
    _write_report(out, report)
    print(canonical(main), flush=True)
    return report


def _score_control(row: Mapping[str, Any], case_root: Path) -> Dict[str, Any]:
    """Score a control after its prediction has been frozen by the caller."""
    metadata = json.loads((case_root / "metadata.json").read_text())
    if metadata.get("is_benign"):
        return dict(row, label_status="benign_control", hit1=None, hit3=None, mrr=None)
    truth = json.loads((case_root / "ground_truth.json").read_text())["root_cause_service"]
    metrics = rank_metrics(row["prediction"], truth)
    # rank_metrics already supplies label_status. Merge before assigning the
    # control provenance, rather than passing the same keyword twice to dict().
    return {**row, **metrics, "label_status": "verified_control_target"}


def _labels(data_root: Path, names: Iterable[str]) -> Dict[str, str]:
    result = {}
    for name in names:
        path = data_root / name / "ground_truth.json"
        truth = json.loads(path.read_text()).get("root_cause_service")
        if not truth:
            raise ValueError("missing root-cause service for %s" % name)
        result[name] = str(truth)
    return result


def _load_controls(args: argparse.Namespace, evidence: Any) -> List[CandidateCase]:
    root = Path(args.controls_root).resolve()
    audit_rows = {}
    if args.controls_audit and Path(args.controls_audit).is_file():
        audit_rows = {row["case_id"]: row for row in
                      (json.loads(line) for line in Path(args.controls_audit).read_text().splitlines()
                       if line.strip())}
    cases = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "metadata.json").is_file():
            continue
        metadata = json.loads((path / "metadata.json").read_text())
        if application_id(metadata.get("environment", {}).get("namespace")) != evidence.system:
            continue
        if not str(metadata.get("scenario_id", "")).upper().startswith("C"):
            continue
        if audit_rows and not audit_rows.get(path.name, {}).get("verified"):
            continue
        view = load_observation(path, args.dataset_id, evidence.system, max_rows=evidence.max_rows)
        _, item = replay(new_session(view), feature_protocol=TEMPORAL_PROTOCOL)
        item.pop("label", None)
        cases.append(_candidate_case(path.name, item, view["observation_version"]))
        if args.controls_limit is not None and len(cases) >= args.controls_limit:
            break
    return cases


def _released_dyrca_controls(evidence: Any, controls: Sequence[CandidateCase]) -> List[Dict[str, Any]]:
    checkpoint = evidence.release_root / "temporal_v3.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("temporal_v3 checkpoint missing for control scoring")
    model, model_name = load_checkpoint(str(checkpoint), "auto")
    result = []
    for case in controls:
        # The predictor accepts the tensorized observation directly and never
        # sees the control label or ground-truth file.
        item = {"x": case.x, "edge_index": case.edge_index,
                "edge_attr": case.edge_weight.unsqueeze(-1),
                "node_ids": case.node_names,
                "metadata": {"experiment_id": case.incident_id,
                              "feature_protocol": TEMPORAL_PROTOCOL}}
        prediction = predict_observed(model, item, model_name)
        result.append({"experiment": case.name, "incident_id": case.incident_id,
                       "task": case.task, "split": "control", "method": "DY-RCA",
                       "adapter": "dyrca", "source_cache_hash": case.source_cache_hash,
                       "control_scenario": case.task, "prediction": prediction})
    return result


def _released_predictions(evidence: Any, parts=("test", "regression")) -> List[Dict[str, Any]]:
    path = evidence.release_root / "predictions.jsonl"
    wanted = {(part, name) for part in parts for name in evidence.split[part]}
    found: Dict[Any, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row.get("split"), row.get("experiment"))
        if row.get("group") == "temporal_v3" and key in wanted:
            case = evidence.cases[row["experiment"]]
            ranked = row["prediction"]["ranked_node_ids"]
            if set(ranked) != set(case.node_names) or len(ranked) != len(case.node_names):
                raise ValueError("released candidate inventory mismatch for %s" % case.name)
            found[key] = {
                "experiment": case.name,
                "incident_id": case.incident_id,
                "task": case.task,
                "split": row["split"],
                "method": "DY-RCA",
                "adapter": "dyrca",
                "source_cache_hash": case.source_cache_hash,
                "model_hash": row.get("model_hash"),
                "prediction": row["prediction"],
            }
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError("released DY-RCA prediction missing: %s" % missing[0])
    return [found[key] for key in sorted(found)]


def _summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    methods = sorted(set(row["method"] for row in rows))
    for part in ("test", "regression"):
        for method in methods:
            for task in ("ALL", "T1", "T2", "T3", "T4"):
                subset = [row for row in rows if row["split"] == part and row["method"] == method
                          and (task == "ALL" or row["task"] == task)]
                labeled = [row for row in subset if row["mrr"] is not None]
                result.append({
                    "split": part,
                    "method": method,
                    "task": task,
                    "n": len(subset),
                    "labeled_n": len(labeled),
                    "hit1": _mean(labeled, "hit1"),
                    "hit3": _mean(labeled, "hit3"),
                    "mrr": _mean(labeled, "mrr"),
                })
    return result


def _main_comparison(summary: Sequence[Mapping[str, Any]], selected: Sequence[str]) -> List[Dict[str, Any]]:
    order = [DISPLAY_NAMES[name] for name in selected] + ["DY-RCA"]
    lookup = {(row["split"], row["method"]): row for row in summary if row["task"] == "ALL"}
    result = []
    for method in order:
        row = {"method": method}
        for part in ("test", "regression"):
            source = lookup[(part, method)]
            row.update({
                part + "_n": source["n"],
                part + "_hit1": source["hit1"],
                part + "_hit3": source["hit3"],
                part + "_mrr": source["mrr"],
            })
        result.append(row)
    return result


def _paired_comparisons(rows: Sequence[Mapping[str, Any]], baselines: Sequence[str]) -> List[Dict[str, Any]]:
    result = []
    for part in ("test", "regression"):
        ours = {row["experiment"]: row for row in rows
                if row["split"] == part and row["method"] == "DY-RCA"}
        for baseline in baselines:
            other = {row["experiment"]: row for row in rows
                     if row["split"] == part and row["method"] == baseline}
            names = sorted(set(ours) & set(other))
            mrr_delta = [ours[name]["mrr"] - other[name]["mrr"] for name in names]
            result.append({
                "split": part,
                "baseline": baseline,
                "n": len(names),
                "dyrca_only_hit1": sum(ours[name]["hit1"] == 1 and other[name]["hit1"] == 0 for name in names),
                "baseline_only_hit1": sum(ours[name]["hit1"] == 0 and other[name]["hit1"] == 1 for name in names),
                "both_hit1": sum(ours[name]["hit1"] == 1 and other[name]["hit1"] == 1 for name in names),
                "neither_hit1": sum(ours[name]["hit1"] == 0 and other[name]["hit1"] == 0 for name in names),
                "mrr_wins": sum(value > 1e-12 for value in mrr_delta),
                "mrr_ties": sum(abs(value) <= 1e-12 for value in mrr_delta),
                "mrr_losses": sum(value < -1e-12 for value in mrr_delta),
                "hit1_difference": _mean([ours[name] for name in names], "hit1")
                                   - _mean([other[name] for name in names], "hit1") if names else None,
                "mrr_difference": sum(mrr_delta) / len(mrr_delta) if mrr_delta else None,
            })
    return result


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _source_hashes() -> Dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    paths = [
        package / "baselines" / "evidence.py",
        package / "baselines" / "methods.py",
        package / "system_config.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(package)): file_hash(path) for path in paths}


def _verify_upstream_checkouts() -> Dict[str, str]:
    # Git metadata may be intentionally read-only. The pinned values are verified
    # during setup and recorded here alongside the checked-out HEAD text when found.
    result = {}
    roots = {
        "rcaeval": Path("baselines/external/RCAEval"),
        "eadro": Path("baselines/external/Eadro"),
        "holisticrca": Path("baselines/external/HolisticRCA"),
    }
    for name, root in roots.items():
        head = root / ".git" / "HEAD"
        if not head.is_file():
            raise FileNotFoundError("upstream checkout missing: %s" % root)
        text = head.read_text().strip()
        if text.startswith("ref: "):
            ref = root / ".git" / text[5:]
            if ref.is_file():
                text = ref.read_text().strip()
            else:
                packed = root / ".git" / "packed-refs"
                mapping = {}
                for line in packed.read_text().splitlines():
                    if line and not line.startswith(("#", "^")):
                        value, key = line.split(" ", 1)
                        mapping[key] = value
                text = mapping[text[5:]]
        result[name] = text
    expected = {
        "rcaeval": SOURCE_MANIFEST["rcaeval"]["commit"],
        "eadro": SOURCE_MANIFEST["eadro"]["commit"],
        "holisticrca": SOURCE_MANIFEST["holisticrca"]["commit"],
    }
    if result != expected:
        raise ValueError("upstream checkout commit differs from source manifest")
    return result


def _write_report(out: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Dynamic-evidence baseline run",
        "",
        "All methods used the frozen temporal-v3 split, candidate sets, and cached pre/during evidence.",
        "Test and regression labels were read only after `predictions.jsonl` was written.",
        "The `-Dyn` suffix denotes a source-guided input adapter, not an unmodified upstream execution.",
        "",
        "| Method | Test Hit@1 | Test Hit@3 | Test MRR | Regression Hit@1 | Regression MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["main_comparison"]:
        display = {key: "N/A" if row[key] is None else f"{row[key]:.4f}"
                   for key in ("test_hit1", "test_hit3", "test_mrr", "regression_hit1", "regression_mrr")}
        lines.append("| {method} | {test_hit1} | {test_hit3} | {test_mrr} | "
                     "{regression_hit1} | {regression_mrr} |".format(method=row["method"], **display))
    if all(row["regression_n"] == 0 for row in report["main_comparison"]):
        lines.extend(["", "No independent regression partition was retained; its scores are N/A."])
    lines.extend([
        "",
        "Control predictions, when supplied, are label-free and are excluded from the main fault-ranking denominator.",
        "All reported fault metrics use the verified held-out cohort; causal interpretation still follows the E0 audit.",
    ])
    (out / "RUN_REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/experiments")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--release-root", default="artifacts/harness_temporal_v3_release")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-validation", action="store_true",
                        help="Freeze validation predictions for prespecified comparator selection")
    parser.add_argument("--planner", choices=["rule"], default="rule")
    parser.add_argument("--system", "--namespace", dest="system",
                        choices=["boutique", "trainticket"], default="boutique")
    parser.add_argument("--dataset-id", default="dyrca-v1")
    parser.add_argument("--controls-root", default=None,
                        help="Optional raw root containing verified C* controls for E3 scoring.")
    parser.add_argument("--controls-limit", type=int, default=None)
    parser.add_argument("--controls-audit", default=None)
    parser.add_argument("--baseline", action="append", choices=BASELINES, dest="baselines")
    args = parser.parse_args()
    args.baseline = args.baselines
    del args.baselines
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
