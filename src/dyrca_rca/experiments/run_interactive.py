"""Bounded offline correction runner with immutable revisions and resume."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from dyrca_rca.dataset.materialize import INTERACTIVE_FEATURE_PROTOCOL

from dyrca_rca.experiments.inference import load_checkpoint, save_checkpoint
from dyrca_rca.interaction.observation import load_observation, canonical, digest
from dyrca_rca.interaction.pipeline import replay, PLAN_SCHEMA
from dyrca_rca.interaction.state import apply_event, new_session, session_key, FEEDBACK_PROTOCOL
from dyrca_rca.models.config import set_seed


PROTOCOL = INTERACTIVE_FEATURE_PROTOCOL


def _write_json(path, data):
    path.write_text(canonical(data) + "\n", encoding="utf-8")


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _persist(root, key, session, result, item):
    destination = root / key
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".revision-", dir=str(root)))
        try:
            _write_json(temporary / "session.json", session)
            _write_json(temporary / "result.json", result)
            torch.save(item, str(temporary / "graph.pt"))
            (temporary / "events.jsonl").write_text(
                "".join(canonical(e) + "\n" for e in session["events"]), encoding="utf-8")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))
    pointer = root / ".current.tmp"
    _write_json(pointer, {"cache_key": key, "revision": session["revision"]})
    os.replace(str(pointer), str(root / "current.json"))
    return destination


def _events(path, max_events):
    result = []
    if not path:
        return result
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if len(result) >= max_events:
                raise ValueError("feedback exceeds --max-events")
            result.append(json.loads(line))
    return result


def _records(args):
    root = Path(args.data_root).resolve()
    records = []
    for exp in sorted(root.iterdir()):
        path = exp / "metadata.json"
        if not exp.is_dir() or not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        namespace = metadata.get("environment", {}).get("namespace", "unknown")
        if namespace != args.system:
            continue
        # Metadata and raw paths stay in the local manifest, never in observations.
        records.append({"path": exp, "system_id": namespace})
        if args.limit is not None and len(records) >= args.limit:
            break
    if not records:
        raise ValueError("no experiments matched --system")
    return records


def _view(record, args, dependencies):
    return load_observation(str(record["path"]), args.dataset_id, record["system_id"],
                            as_of=args.as_of, start_at=args.start_at,
                            max_rows=args.max_rows, dependencies=dependencies)


def _train(args, records, dependencies, out):
    """Train once using an explicit disjoint split; labels stay in training only."""
    from dyrca_rca.experiments.train import train_model
    from dyrca_rca.dataset.labels import root_node_index
    split = json.loads(Path(args.train_split).read_text(encoding="utf-8"))
    if set(split) != {"train", "val", "test"} or any(
        not isinstance(v, list) or not all(isinstance(x, str) for x in v) for v in split.values()
    ):
        raise ValueError("split must contain train/val/test lists of directory names")
    flattened = [v for group in split.values() for v in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("train/val/test overlap or contain duplicates")
    by_name = {r["path"].name: r for r in records}
    training = {"train": [], "val": []}
    for group in training:
        for name in split[group]:
            if name not in by_name:
                raise ValueError("split sample absent from bounded selection: " + name)
            record = by_name[name]
            view = _view(record, args, dependencies)
            result, item = replay(new_session(view))
            # This is the sole label read; replay and inference receive no labels.
            truth = json.loads((record["path"] / "ground_truth.json").read_text(encoding="utf-8"))
            label = root_node_index(result["graph"]["nodes"], truth.get("root_cause_service", ""))
            if label < 0:
                raise ValueError("training label is not in the observed graph: " + name)
            item["label"] = torch.tensor(label, dtype=torch.long)
            training[group].append(item)
    if not training["train"] or not training["val"]:
        raise ValueError("explicit nonempty train and val sets required; no test-set fallback")
    if args.planner != "rule":
        raise ValueError("v1 checkpoint training requires --planner rule")
    model = train_model(training["train"], training["val"], args.model, args.seed)
    metadata = {"dataset_id": args.dataset_id, "split_hash": digest(split),
                "seed": args.seed, "num_train": len(training["train"]), "num_val": len(training["val"]),
                "training_incidents": [i["metadata"]["experiment_id"] for i in training["train"]]}
    save_checkpoint(model, out / "model.pt", args.model, PROTOCOL, metadata)
    _write_json(out / "training.json", dict(metadata, split=split, limit=args.limit,
                                           data_root=str(Path(args.data_root).resolve()), planner=args.planner))
    return {"checkpoint": str(out / "model.pt"), "status": "trained"}


def run(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.max_rows < 1 or args.max_events < 1:
        raise ValueError("row/event limits must be positive")
    data_root = Path(args.data_root).resolve()
    out = Path(args.out_dir).resolve()
    if out == data_root or data_root in out.parents:
        raise ValueError("out-dir must be outside raw data-root")
    out.mkdir(parents=True, exist_ok=True)
    dependencies = json.loads(Path(args.system_config).read_text()) if args.system_config else None
    records = _records(args)
    set_seed(args.seed)
    # One writer per output tree. Revisions are atomic; process crashes cannot
    # publish a half-written graph or leave current.json pointing at one.
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if args.train_split:
            if args.feedback or args.checkpoint:
                raise ValueError("training cannot apply feedback or load a checkpoint")
            return _train(args, records, dependencies, out)
        model = model_name = None
        model_version = "no-checkpoint"
        active_protocol = PROTOCOL
        if args.checkpoint:
            model, model_name = load_checkpoint(args.checkpoint, "auto")
            active_protocol = model.feature_protocol
            model_version = _file_hash(args.checkpoint)
        llm = None
        planner_key = args.planner
        if args.planner == "llm":
            from dyrca_rca.planner.llm_planner import default_llm_client
            llm = default_llm_client()
            planner_key += ":" + digest({"schema": PLAN_SCHEMA,
                                        "model": getattr(llm, "model", "unavailable"),
                                        "endpoint": getattr(llm, "base_url", "")})
        feedback = _events(args.feedback, args.max_events)
        views = [_view(record, args, dependencies) for record in records]
        known = {view["incident_id"] for view in views}
        if any(event.get("incident_id") not in known for event in feedback):
            raise ValueError("feedback references an incident outside this bounded selection")
        config = {"interaction_state_version": 2, "feedback_protocol": FEEDBACK_PROTOCOL, "dataset_id": args.dataset_id, "data_root": str(data_root), "system": args.system,
                  "seed": args.seed, "planner": args.planner, "limit": args.limit, "max_rows": args.max_rows,
                  "feature_protocol": active_protocol, "model_version": model_version,
                  "experiment_group": "structured_recompute"}
        config_key = digest(config)
        _write_json(out / ("run-" + config_key + ".json"), config)
        summaries = []
        manifest = []
        for record, view in zip(records, views):
            incident_root = out / view["incident_id"] / digest(["state-v2", FEEDBACK_PROTOCOL, active_protocol, view["observation_version"], model_version, planner_key, args.seed])
            incident_root.mkdir(parents=True, exist_ok=True)
            pointer = incident_root / "current.json"
            if pointer.exists():
                key = json.loads(pointer.read_text())["cache_key"]
                session = json.loads((incident_root / key / "session.json").read_text())
                result = json.loads((incident_root / key / "result.json").read_text())
                destination = incident_root / key
            else:
                session = new_session(view)
                result, item = replay(session, model, model_name, args.planner, llm)
                key = session_key(session, model_version, planner_key)
                destination = _persist(incident_root, key, session, result, item)
            audit_path = incident_root / "audit.json"
            if audit_path.exists():
                audited = json.loads(audit_path.read_text())
                # A newer current pointer means a commit completed before an
                # interrupted audit write. Its session contains the full log.
                if audited["audit_seq"] > session["audit_seq"]:
                    session = audited
            for event in feedback:
                if event["incident_id"] != view["incident_id"]:
                    continue
                trial = {}

                def validate(candidate):
                    trial["result"], trial["item"] = replay(candidate, model, model_name, args.planner, llm)

                next_session = apply_event(session, event, validate)
                if next_session == session:
                    continue
                changed = next_session["revision"] != session["revision"]
                session = next_session
                if changed:
                    result, item = trial["result"], trial["item"]
                    key = session_key(session, model_version, planner_key)
                    destination = _persist(incident_root, key, session, result, item)
                audit_tmp = incident_root / ".audit.tmp"
                _write_json(audit_tmp, session)
                os.replace(str(audit_tmp), str(audit_path))
                if session["outcomes"][event["event_id"]]["execution_status"] == "pending":
                    try:
                        preview, _ = replay(session, model, model_name, args.planner, llm,
                                            mode="preview", feedback_id=event["event_id"])
                    except Exception as exc:
                        preview = {"mode": "preview", "feedback_id": event["event_id"],
                                   "state_revision": session["state_revision"], "error_type": type(exc).__name__}
                    preview_tmp = incident_root / ".preview.tmp"
                    _write_json(preview_tmp, preview)
                    os.replace(str(preview_tmp), str(incident_root / "preview.json"))
            manifest.append({"incident_id": view["incident_id"], "data_path": str(record["path"]),
                             "system_id": record["system_id"], "observation_version": view["observation_version"],
                             "as_of": view["as_of"], "modalities": view["modalities"], "result_path": str(destination / "result.json")})
            summaries.append({"incident_id": view["incident_id"], "revision": session["revision"],
                              "graph_version": key, "graph_cache_path": str(destination / "graph.pt"),
                              "diagnosis_status": result["diagnosis_status"], "plan_source": result["plan_source"],
                              "node_count": len(result["graph"]["nodes"]), "edge_count": len(result["graph"]["edges"]),
                              "seed": args.seed, "planner": args.planner, "limit": args.limit,
                              "data_root": str(data_root), "experiment_group": config["experiment_group"]})
        (out / "manifest.jsonl").write_text("".join(canonical(row) + "\n" for row in manifest), encoding="utf-8")
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        return {"out_dir": str(out), "incidents": len(summaries), "status": "complete"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/experiments")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-id", default="dyrca-v1")
    parser.add_argument("--system", default="boutique")
    parser.add_argument("--system-config", help="JSON mapping system IDs to dependency service pairs")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=["rule", "llm"], default="rule")
    parser.add_argument("--as-of")
    parser.add_argument("--start-at")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--feedback", help="Correction events in JSONL")
    parser.add_argument("--checkpoint", help="Frozen model trained with this observation protocol")
    parser.add_argument("--train-split", help="Train once using explicit train/val/test directory-name lists")
    parser.add_argument("--model", choices=["weighted_graphsage", "edge_aware_gat"], default="weighted_graphsage")
    return parser.parse_args(argv)


def main(argv=None):
    print(canonical(run(parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
