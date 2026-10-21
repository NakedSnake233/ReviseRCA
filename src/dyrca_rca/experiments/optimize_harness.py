"""Fixed-config, leakage-audited training and evaluation of temporal RCA inputs."""
import argparse
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from dyrca_rca.dataset.labels import root_node_index
from dyrca_rca.dataset.materialize import feature_names
from dyrca_rca.experiments.train import train_model
from dyrca_rca.experiments.inference import load_checkpoint, save_checkpoint
from dyrca_rca.experiments.evaluate_interactive import rank_metrics
from dyrca_rca.experiments.run_interactive import _file_hash, PROTOCOL as V2
from dyrca_rca.interaction.observation import canonical, digest, load_observation
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session
from dyrca_rca.interaction.temporal import PROTOCOL as V3
from dyrca_rca.models.config import set_seed
from dyrca_rca.system_config import application_id


DEFAULT_TASKS = ("t2", "t3", "t4")


def grouped_split(names, regression, seed, tasks=DEFAULT_TASKS):
    """Split only identifiers, never labels; keep repeat-run configurations together."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for name in sorted(set(names) - set(regression)):
        groups[name.rsplit("-run", 1)[0]].append(name)
    split = {"train": [], "val": [], "test": [], "regression": sorted(regression)}
    for task in tasks:
        keys = sorted(k for k in groups if k.startswith(task + "-"))
        if len(keys) < 5:
            raise ValueError("at least five remaining configuration groups per task required")
        rng.shuffle(keys)
        a, b = int(len(keys) * 0.6), int(len(keys) * 0.8)
        for part, selected in (("train", keys[:a]), ("val", keys[a:b]), ("test", keys[b:])):
            split[part].extend(n for key in selected for n in groups[key])
    for part in split:
        split[part].sort()
    return split


def service_grouped_split(names, regression, target_services, seed,
                          tasks=DEFAULT_TASKS):
    """Split by target service so no service label crosses train/val/test.

    Repeat runs of a service stay together, and regression cases are kept out
    of the train/validation/test partitions.  This is intentionally a
    deterministic service-level holdout: it tests generalization to a target
    service whose name and fault configuration were absent during training.
    """
    eligible = sorted(set(names) - set(regression))
    groups = defaultdict(list)
    for name in eligible:
        service = target_services.get(name)
        if not service:
            raise ValueError("missing target service metadata for %s" % name)
        groups[str(service)].append(name)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("at least three target-service groups are required")

    # Search the small service-group assignment space so every task is present
    # in all three partitions whenever that is combinatorially possible.  The
    # objective keeps case counts close to 60/20/20; a seeded tie-break makes
    # the result reproducible without using labels as model input.
    desired = (0.60, 0.20, 0.20)
    candidates = []
    assignment_space = itertools.product(range(3), repeat=len(keys))
    for assignment in assignment_space:
        if not all(any(assignment[i] == part and
                       any(name[:2] == task for name in groups[key])
                       for i, key in enumerate(keys))
                   for part in range(3) for task in tasks):
            continue
        counts = [sum(len(groups[key]) for i, key in enumerate(keys)
                      if assignment[i] == part) for part in range(3)]
        total = float(sum(counts))
        score = sum((counts[i] / total - desired[i]) ** 2 for i in range(3))
        candidates.append((score, assignment, counts))
    if not candidates:
        # Small or highly imbalanced releases may not permit task coverage in
        # every partition.  Fall back to a seeded assignment that still keeps
        # service groups disjoint and nonempty.
        rng = random.Random(seed)
        order = list(keys)
        rng.shuffle(order)
        n_train = max(1, int(len(order) * 0.6))
        n_val = max(1, int(len(order) * 0.2))
        n_val = min(n_val, len(order) - n_train - 1)
        candidates = [(0.0, tuple(0 if key in order[:n_train] else
                                  1 if key in order[n_train:n_train + n_val] else 2
                                  for key in keys), [])]
    rng = random.Random(seed)
    best_score = min(item[0] for item in candidates)
    best = [item for item in candidates if abs(item[0] - best_score) <= 1e-12]
    selected_assignment = best[rng.randrange(len(best))][1]
    selected = {
        "train": [key for i, key in enumerate(keys) if selected_assignment[i] == 0],
        "val": [key for i, key in enumerate(keys) if selected_assignment[i] == 1],
        "test": [key for i, key in enumerate(keys) if selected_assignment[i] == 2],
    }
    split = {part: [] for part in ("train", "val", "test")}
    split["regression"] = sorted(regression)
    for part, service_keys in selected.items():
        split[part] = sorted(name for key in service_keys for name in groups[key])
    return split


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): _file_hash(p) for p in sorted(root.rglob("*.py"))}


def run(args):
    raw, out = Path(args.data_root).resolve(), Path(args.out_dir).resolve()
    if out == raw or raw in out.parents or args.limit < 1 or args.max_rows < 1:
        raise ValueError("bounded positive limits and output outside raw data required")
    # Tiny graphs suffer severe overhead from the host's many CPU threads.
    # This is a fixed execution setting shared by every comparison group.
    torch.set_num_threads(1)
    set_seed(args.seed)
    regression = json.loads(Path(args.regression_selection).read_text())
    tasks = tuple(args.tasks)
    eligible = sorted(p.name for p in raw.iterdir() if p.name[:2] in tasks
                      and (p / "metadata.json").is_file()
                      and application_id(json.loads((p / "metadata.json").read_text()).get("environment", {}).get("namespace")) == args.system)
    if len(eligible) > args.limit:
        raise ValueError("limit is below eligible population; increase explicitly to avoid a biased partial split")
    if not set(regression) <= set(eligible):
        raise ValueError("regression selection includes ineligible incidents")
    target_services = {}
    for name in eligible:
        metadata = json.loads((raw / name / "metadata.json").read_text())
        target = metadata.get("target_service")
        if not target:
            raise ValueError("missing target_service metadata for %s" % name)
        target_services[name] = str(target)
    split_seed = getattr(args, "split_seed", None)
    split_seed = args.seed if split_seed is None else split_seed
    if args.split_policy == "service_group":
        split = service_grouped_split(eligible, regression, target_services, split_seed, tasks=tasks)
    else:
        split = grouped_split(eligible, regression, split_seed, tasks=tasks)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime.json").write_text(canonical({"torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
                                               "device": "cpu"}) + "\n")
    config = {"seed": args.seed, "tasks": list(tasks), "max_rows": args.max_rows, "limit": args.limit, "planner": args.planner,
              "split_policy": args.split_policy,
              "split_seed": split_seed,
              "include_validation": getattr(args, "include_validation", False),
              "data_root": str(raw), "system": args.system, "dataset_id": args.dataset_id,
              "split": split, "v2": V2, "v3": V3,
              "old_checkpoint_hash": _file_hash(args.old_checkpoint), "source_hashes": source_hashes(),
              "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads()}
    path = out / "split.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("existing run configuration differs")
    path.write_text(canonical(config) + "\n")
    print("Frozen split: " + canonical({k: len(v) for k, v in split.items()}), flush=True)
    items = {V2: {}, V3: {}}
    cache_hashes = {V2: {}, V3: {}}
    observation_hashes = {}
    sessions = {}
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    for i, name in enumerate(eligible, 1):
        view = load_observation(raw / name, args.dataset_id, args.system, max_rows=args.max_rows)
        session = new_session(view)
        sessions[name] = session
        observation_hashes[name] = view["observation_version"]
        for protocol in items:
            key = digest([view["observation_version"], protocol, config["source_hashes"], args.planner, torch.__version__])
            cached = cache / (key + ".pt")
            receipt = cached.with_suffix(".json")
            if cached.exists():
                if not receipt.exists() or json.loads(receipt.read_text())["sha256"] != _file_hash(cached):
                    raise ValueError("cache content hash mismatch")
                item = torch.load(str(cached), map_location="cpu")
            else:
                _, item = replay(session, feature_protocol=protocol)
                torch.save(item, str(cached))
                receipt.write_text(canonical({"key": key, "sha256": _file_hash(cached)}) + "\n")
            node_names, edge_names = feature_names(protocol)
            if (item["metadata"]["node_feature_names"] != node_names
                    or item["metadata"]["edge_feature_names"] != edge_names
                    or item["x"].shape != (len(item["node_ids"]), len(node_names))
                    or item["edge_attr"].shape[1] != len(edge_names)
                    or "label" in item):
                raise ValueError("cached input schema mismatch")
            items[protocol][name] = item
            cache_hashes[protocol][name] = _file_hash(cached)
        if i % 50 == 0:
            print("Materialized %s/%s" % (i, len(eligible)), flush=True)
    model_paths = {"old_smoke_v2": Path(args.old_checkpoint)}
    training_manifest = {"train": split["train"], "val": split["val"], "test": split["test"]}
    label_hashes = {name: _file_hash(raw / name / "ground_truth.json") for name in split["train"] + split["val"]}
    for group, protocol in (("retrained_v2", V2), ("temporal_v3", V3)):
        path = out / (group + ".pt")
        metadata = {"split_hash": digest(training_manifest), "dataset_id": args.dataset_id,
                    "system": args.system, "seed": args.seed,
                    "num_train": len(split["train"]), "num_val": len(split["val"]),
                    "cpu_threads": torch.get_num_threads(),
                    "source_hashes": config["source_hashes"], "torch_version": torch.__version__,
                    "max_rows": args.max_rows, "planner": args.planner,
                    "training_label_hashes": label_hashes,
                    "input_hashes": {n: observation_hashes[n] for n in split["train"] + split["val"]},
                    "cache_hashes": {n: cache_hashes[protocol][n] for n in split["train"] + split["val"]},
                    "training_incidents": [sessions[n]["incident_id"] for n in split["train"]],
                    "validation_incidents": [sessions[n]["incident_id"] for n in split["val"]]}
        if path.exists():
            saved = json.loads(torch.load(str(path), map_location="cpu", weights_only=True)["manifest_json"])
            if saved["training_metadata"] != metadata or saved["feature_protocol"] != protocol:
                raise ValueError("cached checkpoint training provenance differs; use a fresh output directory")
        if not path.exists():
            data = {}
            for part in ("train", "val"):
                data[part] = []
                for name in split[part]:
                    item = dict(items[protocol][name])
                    truth = json.loads((raw / name / "ground_truth.json").read_text())["root_cause_service"]
                    index = root_node_index(item["graph"]["nodes"], truth)
                    if index < 0 or ":" in item["node_ids"][index]:
                        raise ValueError("training root missing from service candidates")
                    item["label"] = torch.tensor(index, dtype=torch.long)
                    data[part].append(item)
            print("Training " + group + " with fixed shared config", flush=True)
            model = train_model(data["train"], data["val"], "weighted_graphsage", args.seed)
            save_checkpoint(model, path, "weighted_graphsage", protocol, metadata)
        model_paths[group] = path
    (out / "training.json").write_text(canonical({"split": training_manifest}) + "\n")
    predictions = []
    for group, path in model_paths.items():
        protocol = V3 if group == "temporal_v3" else V2
        model, model_name = load_checkpoint(path, protocol)
        frozen = {k: v.clone() for k, v in model.state_dict().items()}
        model_hash = _file_hash(path)
        for part in (("val", "test", "regression") if config["include_validation"] else ("test", "regression")):
            for name in split[part]:
                for masked in ([False, True] if protocol == V3 else [False]):
                    result, _ = replay(sessions[name], model, model_name, mask_temporal=masked)
                    predictions.append({"experiment": name, "incident_id": sessions[name]["incident_id"],
                                        "split": part, "task": name[:2].upper(),
                                        "group": "temporal_v3_masked" if masked else group,
                                        "model_hash": model_hash, "graph_version": digest(result["graph"]),
                                        "prediction": result["prediction"], "harness": result.get("harness")})
        if any(not torch.equal(frozen[k], v) for k, v in model.state_dict().items()):
            raise RuntimeError("weights changed during inference")
    # Lock prediction artifacts before reading any test/regression label.
    (out / "predictions.jsonl").write_text("".join(canonical(r) + "\n" for r in predictions))
    rows = []
    for row in predictions:
        truth = json.loads((raw / row["experiment"] / "ground_truth.json").read_text()).get("root_cause_service")
        rows.append(dict(row, **rank_metrics(row["prediction"], truth)))
    (out / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in rows))
    summary = []
    for part in ("test", "regression"):
        for group in sorted({r["group"] for r in rows}):
            for task in ("ALL", "T1", "T2", "T3", "T4"):
                subset = [r for r in rows if r["split"] == part and r["group"] == group and (task == "ALL" or r["task"] == task)]
                labeled = [r for r in subset if r["mrr"] is not None]
                summary.append({"split": part, "group": group, "task": task, "n": len(subset), "labeled_n": len(labeled),
                                **{k: sum(r[k] for r in labeled) / len(labeled) if labeled else None for k in ("hit1", "hit3", "mrr")}})
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {"config": config, "summary": summary, "weights_unchanged": True,
              "observation_hashes": observation_hashes, "cache_hashes": cache_hashes,
              "checkpoints": {k: {"path": str(p), "hash": _file_hash(p)} for k, p in model_paths.items()}}
    (out / "report.json").write_text(canonical(report) + "\n")
    print(canonical([r for r in summary if r["task"] == "ALL"]), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/experiments")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None,
                   help="Fix the configuration split independently of the training seed")
    p.add_argument("--include-validation", action="store_true")
    p.add_argument("--planner", choices=["rule"], default="rule")
    p.add_argument("--split-policy", choices=["config_group", "service_group"], default="config_group",
                   help="hold out repeated configuration groups or target services")
    p.add_argument("--dataset-id", default="dyrca-v1")
    p.add_argument("--system", "--namespace", dest="system", choices=["boutique", "trainticket"], default="boutique")
    p.add_argument("--tasks", nargs="+", choices=["t1", "t2", "t3", "t4"],
                   default=list(DEFAULT_TASKS))
    p.add_argument("--old-checkpoint", default="artifacts/interactive_route_training/model.pt")
    p.add_argument("--regression-selection", default="artifacts/interactive_simulated_100/selection.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
