"""Exploratory fixed-seed mixed-cohort test; no model or split search.

Preserve the original T1--T4 split. Assign all C1--C6 cases sharing the same
controller-declared focal service together, including repeated configurations
and decoy variants. Train with C1--C5 only; C6 has no ranking target. These
are collector-labelled controls, not independently verified causal examples.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from dyrca_rca.baselines.evidence import (
    _candidate_case, _receipt_index, canonical, file_hash,
)
from dyrca_rca.baselines.methods import (
    BASELINES, DISPLAY_NAMES, NeuralBaseline, create_classical, ranked_prediction,
)
from dyrca_rca.experiments.evaluate_interactive import rank_metrics
from dyrca_rca.experiments.inference import predict_observed, save_checkpoint
from dyrca_rca.experiments.train import train_model
from dyrca_rca.interaction.observation import load_observation
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session
from dyrca_rca.interaction.temporal import PROTOCOL
from dyrca_rca.models.config import FIXED_GNN_CONFIG


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def control_split(metadata, seed):
    """Keep related focal-service controls together across modes/decoys/copies."""
    groups = defaultdict(list)
    for name, row in sorted(metadata.items()):
        if not row.get("target_service"):
            raise ValueError("control focal-service grouping key is missing")
        groups[row["target_service"]].append(name)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("at least three independent control focal-service groups required")
    random.Random(seed).shuffle(keys)
    a = max(1, min(len(keys) - 2, int(.6 * len(keys))))
    b = max(a + 1, min(len(keys) - 1, int(.8 * len(keys))))
    return {part: sorted(n for key in chosen for n in groups[key])
            for part, chosen in (("train", keys[:a]), ("val", keys[a:b]), ("test", keys[b:]))}


def bound_split(split, limit):
    """Balanced, identifier-only smoke limit; full experiments omit the limit."""
    if limit is None:
        return {k: list(v) for k, v in split.items()}
    if limit < 3:
        raise ValueError("limit must be >= 3")
    result = {k: [] for k in split}
    while sum(map(len, result.values())) < limit:
        changed = False
        for k in result:
            if len(result[k]) < len(split[k]) and sum(map(len, result.values())) < limit:
                result[k].append(split[k][len(result[k])]); changed = True
        if not changed:
            break
    return result


def memory_check():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gib = int(values["MemAvailable"].split()[0]) / 1024 ** 2
    if gib <= 10:
        raise RuntimeError("available host memory <= 10 GiB; stopping before next stage")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        family = "T1-T4" if row["experiment"].startswith("t") else "C1-C5"
        groups[(row["training_cohort"], row["method"], family)].append(row)
        groups[(row["training_cohort"], row["method"], "pooled")].append(row)
    result = []
    for (cohort, method, family), records in sorted(groups.items()):
        item = {"training_cohort": cohort, "method": method, "test_stratum": family, "n": len(records)}
        item.update({k: sum(float(r[k]) for r in records) / len(records) for k in ("hit1", "hit3", "mrr")})
        config = defaultdict(list)
        for r in records:
            config[r["experiment"].rsplit("-run", 1)[0]].append(r)
        item["configuration_groups"] = len(config)
        item["configuration_macro_hit1"] = sum(sum(float(r["hit1"]) for r in v)/len(v) for v in config.values()) / len(config)
        result.append(item)
    return result


def run(args):
    memory_check()
    torch.set_num_threads(1)
    raw, out, release = map(lambda p: Path(p).resolve(), (args.data_root, args.out_dir, args.release_root))
    if not (Path(args.original_ranking) / "results.jsonl").is_file():
        raise FileNotFoundError("original ranking results are required for paired comparison")
    if out == raw or raw in out.parents or (out.exists() and any(out.iterdir())):
        raise ValueError("use a fresh output directory outside the raw data")
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((release / "split.json").read_text())
    report = json.loads((release / "report.json").read_text())
    audit = read_jsonl(Path(args.audit))
    metadata = {}
    for r in audit:
        if r["verified"] and r["scenario_id"].startswith("C"):
            name = r["case_id"]
            metadata[name] = json.loads((raw / name / "metadata.json").read_text())
    controls = control_split(metadata, args.seed)
    c_fault = {p: [n for n in names if not metadata[n].get("is_benign")] for p, names in controls.items()}
    split = bound_split({p: sorted(frozen["split"][p] + c_fault[p]) for p in controls}, args.limit)
    plan = {
        "protocol": "exploratory-mixed-cohort-v1", "application": args.system,
        "seed": args.seed, "planner": args.planner, "limit": args.limit,
        "data_root": str(raw), "graph_cache_path": str(out / "cache"),
        "original_release": str(release), "original_split_sha256": file_hash(release / "split.json"),
        "source_sha256": file_hash(Path(__file__)), "split": split, "control_split_including_benign": controls,
        "fixed_config": FIXED_GNN_CONFIG.__dict__,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "decision_timing": "post-hoc to original published results; frozen before new training",
        "grouping": "original T groups preserved; all C modes/repeats/decoys/copies of a focal service kept together",
        "limits": ["collector labels, no independent causal adjudication", "C6 excluded from ranking, no abstention head",
                   "one fixed seed, no selection by scores", "exploratory re-use of previously evaluated observations"],
    }
    write_json(out / "protocol.json", plan)
    print("FROZEN " + canonical({p: dict(Counter(n[:2] for n in v)) for p, v in split.items()}), flush=True)
    items, cases, hashes = {}, {}, {}
    cache_index = _receipt_index(release / "cache")
    (out / "cache").mkdir()
    all_names = sorted(set(sum(split.values(), [])))
    for index, name in enumerate(all_names):
        memory_check()
        expected = report["cache_hashes"][PROTOCOL].get(name)
        if expected:
            path = cache_index[expected]
            if file_hash(path) != expected:
                raise ValueError("original cache changed")
            item = torch.load(str(path), map_location="cpu")
        else:
            view = load_observation(raw / name, frozen["dataset_id"], args.system, max_rows=frozen["max_rows"])
            _, item = replay(new_session(view), feature_protocol=PROTOCOL)
            path = out / "cache" / (view["observation_version"] + ".pt")
            torch.save(item, str(path))
        if "label" in item:
            raise ValueError("inference cache contains label")
        items[name] = item
        hashes[name] = file_hash(path)
        cases[name] = _candidate_case(name, item, hashes[name])
        if (index + 1) % 50 == 0:
            print("CACHE %s/%s" % (index + 1, len(all_names)), flush=True)
    write_json(out / "cache_manifest.json", hashes)
    train_labels = {n: json.loads((raw / n / "ground_truth.json").read_text())["root_cause_service"] for n in split["train"]}
    val_labels = {n: json.loads((raw / n / "ground_truth.json").read_text())["root_cause_service"] for n in split["val"]}
    train_cases, val_cases = ([cases[n] for n in split[p]] for p in ("train", "val"))
    predictions, training = [], {}
    for method_name in BASELINES:
        memory_check()
        print("FIT " + method_name, flush=True)
        if method_name in ("eadro", "holisticrca", "nodeonly"):
            method = NeuralBaseline(method_name, seed=args.seed).fit(train_cases, val_cases, train_labels, val_labels)
            cp = out / (method_name + ".pt")
            torch.save({"state_dict": method.state_dict(), "mean": method.mean, "std": method.std}, str(cp))
            training[method_name] = {"checkpoint_sha256": file_hash(cp), **method.training_record}
        else:
            method = create_classical(method_name, seed=args.seed).fit(train_cases)
        for name in split["test"]:
            row = ranked_prediction(cases[name], method.predict(cases[name]))
            predictions.append(dict(row, method=DISPLAY_NAMES[method_name], training_cohort="mixed", source_cache_hash=hashes[name]))
        del method
    memory_check()
    print("FIT DY-RCA", flush=True)
    def labeled(names, labels):
        result = []
        for n in names:
            item = dict(items[n])
            if labels[n] not in cases[n].node_names:
                raise ValueError("training target missing from candidates")
            item["label"] = torch.tensor(item["node_ids"].index(labels[n]), dtype=torch.long)
            result.append(item)
        return result
    model = train_model(labeled(split["train"], train_labels), labeled(split["val"], val_labels), seed=args.seed)
    save_checkpoint(model, out / "dyrca.pt", "weighted_graphsage", PROTOCOL,
                    {"protocol_sha256": file_hash(out / "protocol.json"), "seed": args.seed})
    for n in split["test"]:
        predictions.append({"experiment": n, "method": "DY-RCA", "training_cohort": "mixed",
                            "source_cache_hash": hashes[n], "prediction": predict_observed(model, items[n], "weighted_graphsage")})
    # Commit all predictions before accessing held-out ground-truth files.
    (out / "predictions.jsonl").write_text("".join(canonical(r) + "\n" for r in predictions))
    prediction_hash = file_hash(out / "predictions.jsonl")
    rows = [dict(r, **rank_metrics(r["prediction"], json.loads((raw / r["experiment"] / "ground_truth.json").read_text())["root_cause_service"])) for r in predictions]
    old = read_jsonl(Path(args.original_ranking) / "results.jsonl")
    comparison = [dict(r, training_cohort="original_T_only") for r in old if r["experiment"] in split["test"] and r["split"] in ("test", "control")]
    if args.limit is None and len(comparison) != len(predictions):
        raise ValueError("original/mixed predictions are not paired")
    rows.extend(comparison)
    (out / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in rows))
    summary = summarize(rows)
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    write_json(out / "report.json", {"protocol": plan, "summary": summary, "training": training,
                                    "predictions_sha256_before_scoring": prediction_hash,
                                    "test_cases": len(split["test"]), "status": "complete_exploratory"})
    print("COMPLETE " + str(out), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--planner", choices=["rule"], default="rule")
    p.add_argument("--system", choices=["boutique", "trainticket"], required=True)
    p.add_argument("--release-root", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--original-ranking", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
