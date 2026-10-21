"""Budgeted shared-evidence behavior update on an already frozen mixed split.

No split/seed/model search. The six adapted RCA baselines and simple controls
receive the same 57 fields and separated edge weights. A paired DY-RCA arm
retains the old tag channels to isolate their contribution. All data are read
only; final test labels are opened after prediction files have been committed.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from dyrca_rca.baselines.evidence import _candidate_case, _receipt_index, canonical, file_hash
from dyrca_rca.baselines.methods import BASELINES, DISPLAY_NAMES, NeuralBaseline, create_classical, ranked_prediction
from dyrca_rca.experiments.evaluate_interactive import rank_metrics
from dyrca_rca.experiments.inference import predict_observed
from dyrca_rca.experiments.rebalance_shortcut import bound_split, memory_check, read_jsonl, write_json
from dyrca_rca.experiments.train import train_model, evaluate_loss
from dyrca_rca.interaction.behavior import PROTOCOL, FEATURES, augment, load_behavior
from dyrca_rca.interaction.temporal import PROTOCOL as OLD_PROTOCOL
from dyrca_rca.models.config import FIXED_GNN_CONFIG


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        name = r["experiment"]
        for stratum in [name[:2].upper(), "T1-T4" if name.startswith("t") else "C1-C5", "pooled"]:
            groups[(r["method"], stratum)].append(r)
    return [{"method": m, "stratum": s, "n": len(rs),
             **{k: sum(r[k] for r in rs)/len(rs) for k in ("hit1", "hit3", "mrr")}}
            for (m, s), rs in sorted(groups.items())]


def run(args):
    start = time.monotonic()
    raw, old, out = [Path(p).resolve() for p in (args.data_root, args.mixed_root, args.out_dir)]
    if out == raw or raw in out.parents or (out.exists() and any(out.iterdir())):
        raise ValueError("fresh output outside raw data required")
    if args.budget_seconds <= 0:
        raise ValueError("positive wall-time budget required")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    def guard(stage):
        memory_check()
        elapsed = time.monotonic()-start
        write_json(out / "status.json", {"stage": stage, "elapsed_seconds": elapsed,
                                        "budget_seconds": args.budget_seconds})
        if elapsed >= args.budget_seconds:
            raise TimeoutError("experiment wall-time budget exhausted")
        print(stage, flush=True)
    prior = json.loads((old / "protocol.json").read_text())
    if args.system != prior["application"] or args.seed != prior["seed"]:
        raise ValueError("application/seed must match the frozen mixed experiment")
    split = bound_split(prior["split"], args.limit)
    release = Path(prior["original_release"])
    source_files = [Path(__file__), Path(__file__).parents[1] / "interaction/behavior.py",
                    Path(__file__).parents[1] / "baselines/methods.py", Path(__file__).with_name("train.py")]
    protocol = {"protocol": PROTOCOL, "application": args.system, "data_root": str(raw), "seed": args.seed,
                "planner": args.planner, "limit": args.limit, "split": split,
                "frozen_mixed_protocol_sha256": file_hash(old / "protocol.json"),
                "fixed_config": FIXED_GNN_CONFIG.__dict__, "max_rows": 10000,
                "budget_seconds": args.budget_seconds, "graph_cache_path": str(out / "cache"),
                "source_hashes": {str(p): file_hash(p) for p in source_files},
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scope": "post-hoc single-seed mixed-split experiment; no independent causal adjudication",
                "primary_comparison": "all baselines and DY-RCA share behavior and separated tag encoding",
                "ablation": "DY-RCA with behavior but original tag/weight encoding",
                "missing_identity": "left unavailable; no injection-derived identity repair",
                "runtime": {"device": "cpu", "threads": 1, "torch": torch.__version__}}
    write_json(out / "protocol.json", protocol)
    cache_index = _receipt_index(release / "cache")
    for p in (old / "cache").glob("*.pt"):
        cache_index[file_hash(p)] = p
    expected_hashes = json.loads((old / "cache_manifest.json").read_text())
    (out / "cache").mkdir()
    items, coupled, cases, coverage, manifest = {}, {}, {}, [], {}
    names = sorted(set(sum(split.values(), [])))
    for i, name in enumerate(names):
        if i % 50 == 0:
            guard(f"CACHE {i}/{len(names)}")
        expected = expected_hashes[name]
        path = cache_index[expected]
        if file_hash(path) != expected:
            raise ValueError("existing cache no longer matches its receipt")
        item = torch.load(str(path), map_location="cpu")
        if "label" in item:
            raise ValueError("label-bearing inference cache")
        behavior = load_behavior(raw / name)
        write_json(out / "cache" / (name + ".json"), behavior)
        coupled[name] = augment(item, behavior, False)
        items[name] = augment(item, behavior, True)
        torch.save(items[name], out / "cache" / (name + ".pt"))
        tensor_hash = file_hash(out / "cache" / (name + ".pt"))
        cases[name] = _candidate_case(name, items[name], tensor_hash)
        inputs = [raw / name / rel for rel in ["topology/pre.json", "topology/during.json", "traces/trace_summary.csv", "metrics/latency_p99.csv"]]
        manifest[name] = {"original_cache_sha256": expected, "new_cache_sha256": tensor_hash,
                          "raw_input_sha256": {str(p): file_hash(p) for p in inputs if p.exists()}}
        coverage.append({"experiment": name, **behavior["coverage"],
                         "services_with_duration": sum(v[4] > 0 for v in behavior["values"].values()),
                         "services_with_metric": sum(v[11] > 0 for v in behavior["values"].values())})
    write_json(out / "cache_manifest.json", manifest)
    write_json(out / "coverage.json", coverage)
    labels = {p: {n: json.loads((raw/n/"ground_truth.json").read_text())["root_cause_service"] for n in split[p]} for p in ("train", "val")}
    predictions, training = [], {}
    def commit():
        (out / "predictions.jsonl").write_text("".join(canonical(r)+"\n" for r in predictions))
        write_json(out / "training.json", training)
    for method_name in BASELINES:
        guard("FIT " + method_name)
        tick = time.monotonic()
        train_cases, val_cases = ([cases[n] for n in split[p]] for p in ("train", "val"))
        if method_name in ("eadro", "holisticrca", "nodeonly"):
            method = NeuralBaseline(method_name, seed=args.seed).fit(train_cases, val_cases, labels["train"], labels["val"])
            checkpoint = out / (method_name + ".pt")
            torch.save({"state_dict": method.state_dict(), "mean": method.mean, "std": method.std,
                        "protocol_sha256": file_hash(out / "protocol.json")}, checkpoint)
            training[method_name] = {**method.training_record, "checkpoint_sha256": file_hash(checkpoint)}
        else:
            method = create_classical(method_name, seed=args.seed).fit(train_cases)
            training[method_name] = {}
        training[method_name]["fit_seconds"] = time.monotonic()-tick
        for n in split["test"]:
            row = ranked_prediction(cases[n], method.predict(cases[n]))
            row.update(method=DISPLAY_NAMES[method_name], input_group="behavior-separated", source_cache_hash=cases[n].source_cache_hash)
            predictions.append(row)
        del method
        commit()
    for group, source in [("DY-RCA+behavior", coupled), ("DY-RCA+behavior-separated", items)]:
        guard("FIT " + group)
        data = {}
        for part in ("train", "val"):
            data[part] = []
            for n in split[part]:
                item = dict(source[n]); target = labels[part][n]
                if target not in cases[n].node_names:
                    raise ValueError("training target outside candidate universe")
                item["label"] = torch.tensor(item["node_ids"].index(target), dtype=torch.long)
                data[part].append(item)
        tick = time.monotonic()
        model = train_model(data["train"], data["val"], seed=args.seed)
        checkpoint = out / (group + ".pt")
        torch.save({"state_dict": model.state_dict(), "node_features": source[n]["metadata"]["node_feature_names"],
                    "protocol_sha256": file_hash(out / "protocol.json"), "group": group}, checkpoint)
        training[group] = {"fit_seconds": time.monotonic()-tick, "fixed_config": FIXED_GNN_CONFIG.__dict__,
                           "train_loss": evaluate_loss(model, data["train"], "weighted_graphsage"),
                           "validation_loss": evaluate_loss(model, data["val"], "weighted_graphsage"),
                           "checkpoint_sha256": file_hash(checkpoint)}
        for n in split["test"]:
            predictions.append({"experiment": n, "method": group, "input_group": group,
                                "prediction": predict_observed(model, source[n], "weighted_graphsage")})
        commit()
    prediction_sha = file_hash(out / "predictions.jsonl")
    # All compared methods must retain exactly the same test cases/candidates.
    for method in {r["method"] for r in predictions}:
        selected = [r for r in predictions if r["method"] == method]
        if len(selected) != len(split["test"]) or {r["experiment"] for r in selected} != set(split["test"]):
            raise ValueError("unpaired method/case population")
        for row in selected:
            if set(row["prediction"]["ranked_node_ids"]) != set(cases[row["experiment"]].node_names):
                raise ValueError("candidate universe differs")
    rows = [dict(r, **rank_metrics(r["prediction"], json.loads((raw/r["experiment"]/"ground_truth.json").read_text())["root_cause_service"])) for r in predictions]
    (out / "results.jsonl").write_text("".join(canonical(r)+"\n" for r in rows))
    summary = summarize(rows)
    with (out / "summary.csv").open("w") as stream:
        w = csv.DictWriter(stream, fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    write_json(out / "report.json", {"protocol": protocol, "summary": summary, "training": training,
                                    "elapsed_seconds": time.monotonic()-start,
                                    "predictions_sha256_before_scoring": prediction_sha, "status": "complete_exploratory"})
    guard("COMPLETE")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mixed-root", required=True)
    p.add_argument("--system", choices=["boutique", "trainticket"], required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--planner", choices=["rule"], default="rule")
    p.add_argument("--budget-seconds", type=int, default=600)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
