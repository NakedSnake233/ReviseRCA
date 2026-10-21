"""Summarize training and split repetitions separately, with grouped pairing.

The comparator is selected once from seed-42 validation configuration means.
Bootstrap resamples configuration groups in that main test only; additional
training seeds are never treated as additional independent incidents.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

COMPARATORS = ("NSigma-Dyn", "BARO-Dyn", "CIRCA-Dyn", "CloudRanger-Dyn", "Eadro-Dyn", "HolisticRCA-Dyn")
TRAIN_SEEDS = (42, 123, 456, 789, 2024)
SPLIT_SEEDS = (42, 137, 2026)


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def group_means(rows, method, partition, key):
    groups = defaultdict(list)
    for row in rows:
        if row.get("method") == method and row.get("split") == partition and row.get(key) is not None:
            groups[row["experiment"].rsplit("-run", 1)[0]].append(float(row[key]))
    return {name: statistics.mean(values) for name, values in sorted(groups.items())}


def select_comparator(rows):
    candidates = []
    expected = None
    for order, method in enumerate(COMPARATORS):
        mrr = group_means(rows, method, "val", "mrr")
        hit1 = group_means(rows, method, "val", "hit1")
        if not mrr or set(mrr) != set(hit1):
            raise ValueError("Comparator selection requires complete validation predictions for all six baselines")
        if expected is not None and set(mrr) != expected:
            raise ValueError("Validation comparator cohorts differ")
        expected = set(mrr)
        candidates.append((statistics.mean(mrr.values()), statistics.mean(hit1.values()), -order, method))
    return max(candidates)[-1]


def paired_interval(rows, comparator, seed, replicates=2000):
    ours = group_means(rows, "DY-RCA", "test", "mrr")
    other = group_means(rows, comparator, "test", "mrr")
    if not ours or set(ours) != set(other):
        raise ValueError("Grouped paired interval requires identical complete test cohorts")
    values = [ours[group] - other[group] for group in sorted(ours)]
    rng = random.Random(seed)
    means = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values)
                   for _ in range(replicates))
    return {"configuration_groups": len(values), "delta_mrr": statistics.mean(values),
            "ci_low": means[int(.025 * replicates)], "ci_high": means[int(.975 * replicates)],
            "replicates": replicates, "unit": "configuration group", "cohort": "main seed-42 test only"}


def run(args):
    sources = json.loads(Path(args.run_manifest).read_text())
    repetitions = sources["runs"]
    training = [r for r in repetitions if r["split_seed"] == 42]
    splitting = [r for r in repetitions if r["training_seed"] == 42]
    if sorted(r["training_seed"] for r in training) != sorted(TRAIN_SEEDS):
        raise ValueError("The five prespecified training seeds are not complete")
    if sorted(r["split_seed"] for r in splitting) != sorted(SPLIT_SEEDS):
        raise ValueError("The three prespecified split seeds are not complete")
    main = next(r for r in training if r["training_seed"] == 42)
    main_rows = read_rows(main["results"])
    comparator = select_comparator(main_rows)
    summaries = []
    methods = COMPARATORS + ("Change-only", "Node-only", "DY-RCA")
    for kind, selected in (("training_seed", training), ("split_seed", splitting)):
        for method in methods:
            metrics = defaultdict(list)
            case_sets = []
            for record in selected:
                rows = read_rows(record["results"])
                case_sets.append({r["experiment"] for r in rows if r.get("split") == "test" and r.get("method") == method})
                for key in ("hit1", "mrr"):
                    values = group_means(rows, method, "test", key)
                    if not values:
                        raise ValueError(f"Missing repetition metric: {method}, {key}")
                    metrics[key].append(statistics.mean(values.values()))
            if kind == "training_seed" and any(names != case_sets[0] for names in case_sets):
                raise ValueError("Training-seed repetitions must keep the test cases fixed")
            summaries.append({"kind": kind, "method": method, "runs": len(selected),
                              **{key + suffix: function(metrics[key])
                                 for key in metrics for suffix, function in
                                 (("_mean", statistics.mean), ("_sd", statistics.stdev))}})
    report = {"status": "measured", "seed": args.seed, "planner": args.planner,
              "data_root": str(Path(args.data_root).resolve()), "limit": args.limit,
              "comparator": comparator, "selection": "validation configuration-macro MRR; Hit@1 then fixed method order",
              "summary": summaries, "paired": paired_interval(main_rows, comparator, args.seed),
              "runs": repetitions, "coefficient_sensitivity": "pending; no hyperparameter search performed"}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability_summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=("rule",), default="rule")
    parser.add_argument("--limit", type=int)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
