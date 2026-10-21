"""Feature-augmented baseline for the "dynamic evidence + baseline" experiment (R2-C1, R3-C4).

Question (Reviewer 2): is Plan2RCA's gain an *evidence advantage* (it sees G_obs/ΔT that
baselines don't) or a *method advantage* (typed executable patches)?

This variant gives the baseline the SAME dynamic evidence — but as flat node features on
the static graph G_0, with NO structural patch operations (no blackhole/version/trace-mismatch
nodes or edges). It is the most generous "dynamic evidence + baseline" condition: the GNN
sees every change indicator Plan2RCA uses, just not compiled into typed graph operations.

It reuses the main pipeline's materialization (build_graph_variant("static_pre") +
graph_to_model_tensors) and the fixed WeightedGraphSAGE training; only the node features are
augmented with dynamic-evidence indicators derived from pre/during snapshots + topology_diff.

Dynamic features appended per node (service-scoped where applicable, else 0):
  replica_delta_norm, pod_added, pod_removed, endpoint_changed, version_split, trace_mismatch
  + global: g_node_changed, g_edge_changed, g_replica_delta, g_version_split, g_change_score
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from dyrca_rca.dataset.materialize import (
    build_graph_variant, graph_to_model_tensors, read_index,
)
from dyrca_rca.planner.signals import read_json
from dyrca_rca.experiments.splits import make_split
from dyrca_rca.experiments.train import train_model
from dyrca_rca.experiments.evaluate import predict_items, evaluate_predictions
from dyrca_rca.models.config import FIXED_GNN_CONFIG


DYN_FEATURE_NAMES = [
    "replica_delta_norm", "pod_added", "pod_removed", "endpoint_changed",
    "version_split", "trace_mismatch",
    "g_node_changed", "g_edge_changed", "g_replica_delta", "g_version_split", "g_change_score",
]


def _svc_of(node_id: str) -> str:
    """Map a graph node id to its service name."""
    if node_id.startswith("pod:"):
        return node_id[4:].split("-")[0] if "-" in node_id[4:] else node_id[4:]
    # service node ids are plain service names in this graph
    return node_id


def _service_dynamic_features(exp_dir: Path) -> tuple[Dict[str, List[float]], List[float]]:
    """Per-service dynamic features + global change vector, from pre/during + topology_diff."""
    pre = read_json(exp_dir / "topology" / "pre.json", required=True)
    dur = read_json(exp_dir / "topology" / "during.json", required=True)
    td = read_json(exp_dir / "topology_diff.json", required=False) or {}

    def pods_by_service(snap):
        out: Dict[str, List[dict]] = {}
        for p in (snap.get("pods") or []):
            if not isinstance(p, dict):
                continue
            svc = str(p.get("service", "") or "")
            if svc:
                out.setdefault(svc, []).append(p)
        return out

    def deps_map(snap):
        return {str(d.get("name", "")): d for d in (snap.get("deployments") or []) if isinstance(d, dict)}

    pre_pods, dur_pods = pods_by_service(pre), pods_by_service(dur)
    pre_deps, dur_deps = deps_map(pre), deps_map(dur)
    services = set(pre_pods) | set(dur_pods) | set(pre_deps) | set(dur_deps)

    # global change indicators from topology_diff
    cmp = (td.get("comparisons") or {}).get("pre_vs_during") or {}
    k8s = cmp.get("k8s_layer") or {}
    pc = k8s.get("pod_changes") or {}
    added = pc.get("added") or []
    removed = pc.get("removed") or []
    g_node_changed = 1.0 if (added or removed) else 0.0
    g_replica_delta = float(len(added) - len(removed))
    # edge / endpoint changes
    edge_changes = cmp.get("edge_changes") or {}
    g_edge_changed = 1.0 if edge_changes.get("added") or edge_changes.get("removed") else 0.0
    # version split: any pod with v1/v2 version label in during
    g_version_split = 1.0 if any(
        str(p.get("version", "")) not in ("", "unknown", "None")
        for plist in dur_pods.values() for p in plist
    ) else 0.0
    g_change_score = min(1.0, (len(added) + len(removed)) / 6.0)
    global_vec = [g_node_changed, g_edge_changed, g_replica_delta, g_version_split, g_change_score]

    per_svc: Dict[str, List[float]] = {}
    for svc in services:
        pp, dp = pre_pods.get(svc, []), dur_pods.get(svc, [])
        pre_names = {str(p.get("name", "")) for p in pp}
        dur_names = {str(p.get("name", "")) for p in pp}
        pod_added = 1.0 if (dur_names - pre_names) else 0.0
        pod_removed = 1.0 if (pre_names - dur_names) else 0.0
        pre_rep = float((pre_deps.get(svc) or {}).get("replicas", len(pp)))
        dur_rep = float((dur_deps.get(svc) or {}).get("replicas", len(dp)))
        replica_delta_norm = (dur_rep - pre_rep) / max(1.0, pre_rep)
        endpoint_changed = 1.0 if (pod_added or pod_removed) else 0.0
        version_split = 1.0 if any(
            str(p.get("version", "")) not in ("", "unknown", "None") for p in dp
        ) else 0.0
        trace_mismatch = 0.0  # set from trace summary below if available
        per_svc[svc] = [replica_delta_norm, pod_added, pod_removed, endpoint_changed,
                        version_split, trace_mismatch]

    # trace mismatch from trace_summary.csv (declared-vs-observed route mismatch)
    try:
        ts = exp_dir / "traces" / "trace_summary.csv"
        if ts.exists():
            tdf = pd.read_csv(ts)
            if "target_service" in tdf.columns and "route_mismatch" in tdf.columns:
                for svc, grp in tdf.groupby("target_service"):
                    if svc in per_svc and bool(grp["route_mismatch"].any()):
                        per_svc[svc][5] = 1.0
    except Exception:
        pass
    return per_svc, global_vec


def build_feat_aug_item(exp_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    """Build static_pre graph tensors + append dynamic-evidence node features."""
    graph = build_graph_variant(exp_dir, str(row.get("experiment_id", exp_dir.name)),
                                "static_pre", llm_plan_dir="artifacts/dygraph_rca_full/llm_plans")
    item = graph_to_model_tensors(graph, row, emit_pyg=False)
    per_svc, global_vec = _service_dynamic_features(exp_dir)
    node_ids = item["node_ids"]
    dyn_rows = []
    for nid in node_ids:
        svc = _svc_of(str(nid))
        dyn_rows.append(per_svc.get(svc, [0.0] * 6) + global_vec)
    dyn = torch.tensor(dyn_rows, dtype=item["x"].dtype)
    item["x"] = torch.cat([item["x"], dyn], dim=1)
    item["metadata"] = item.get("metadata", {})
    item["metadata"]["feat_aug"] = True
    item["metadata"]["dyn_feature_names"] = DYN_FEATURE_NAMES
    return item


def load_items(frame: pd.DataFrame, data_root: str, cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for _, row in frame.iterrows():
        rd = row.to_dict()
        err = rd.get("error_message")
        if isinstance(err, str) and err.strip():
            continue
        dp = rd.get("data_path")
        if dp is None or (isinstance(dp, float) and pd.isna(dp)):
            continue
        exp_dir = Path(str(dp))
        if not exp_dir.exists():
            exp_dir = Path(data_root) / str(rd.get("experiment_id", ""))
        key = str(exp_dir)
        if key in cache:
            it = cache[key]
        else:
            try:
                it = build_feat_aug_item(exp_dir, rd)
            except Exception:
                it = None
            cache[key] = it
        if it is None:
            continue
        items.append(it)
    return items


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--data-root", default="data/experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    frame = read_index(args.index)
    if args.limit:
        frame = frame.head(args.limit)
    splits = make_split(frame, split="random", seed=args.seed)
    cache: Dict[str, Any] = {}
    train_items = load_items(splits["train"], args.data_root, cache)
    val_items = load_items(splits["val"], args.data_root, cache)
    test_items = load_items(splits["test"], args.data_root, cache)
    if not train_items and test_items:
        train_items = test_items
    if not test_items:
        test_items = val_items or train_items
    print(f"[feat_aug] train={len(train_items)} val={len(val_items)} test={len(test_items)} "
          f"in_dim={train_items[0]['x'].shape[1]}", flush=True)

    model = train_model(train_items, val_items=val_items, model_name="weighted_graphsage", seed=args.seed)
    preds = predict_items(model, test_items, model_name="weighted_graphsage")
    res = evaluate_predictions(preds)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res["overall_frame"].to_csv(out / "metrics_overall.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "variant": "feat_aug (static graph + dynamic-evidence node features, NO patches)",
        "seed": args.seed, "in_dim": int(train_items[0]["x"].shape[1]),
        "dyn_features": DYN_FEATURE_NAMES,
        "fixed_model_config": FIXED_GNN_CONFIG.__dict__,
        "num_test": len(test_items),
    }, indent=2, default=str), encoding="utf-8")
    m = res["overall"]
    print(f"[feat_aug] metrics: top1={m.get('top1'):.4f} mrr={m.get('mrr'):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
