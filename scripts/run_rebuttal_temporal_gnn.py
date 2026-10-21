"""Temporal/dynamic GNN baseline for rebuttal (addresses R2-C6).

Implements an EvolveGCN-O-style baseline (Pareja et al., 2020), which Reviewer 2
named alongside TGN. Unlike Plan2RCA's executable graph patches, this baseline
models the topology as a *temporal sequence* of service-level graphs
(pre-incident -> during-incident) and evolves the GCN weight matrix with a GRU.
It receives the SAME topology evidence (G_0 and G_obs) as every other method.

Variant note: we use the weight-evolution form of EvolveGCN-O (the GRU state is
the flattened GCN weight matrix; one GRU step per time step). This is a standard
simplified realization of EvolveGCN-O used when node-feature pooling is replaced
by direct weight recurrence. It is a fair dynamic-GNN baseline under the fixed
shared config (no HPO):
  - hidden 128, 2 layers, lr 3e-3, dropout 0.1, 200 epochs, early stop 30
  - candidate set = service nodes (same mask convention as the main pipeline)

Usage:
    python scripts/run_rebuttal_temporal_gnn.py \
        --index artifacts/dygraph_rca_full/index.csv \
        --data-root data/experiments \
        --seed 42 --split random \
        --out-dir artifacts/rebuttal_temporal/evolvegcn_s42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dyrca_rca.dataset.materialize import build_graph_from_topology, read_index
from dyrca_rca.graph.weights import apply_edge_weights
from dyrca_rca.graph.executor import normalize_graph
from dyrca_rca.dataset.labels import root_service_from_row
from dyrca_rca.planner.signals import read_json
from dyrca_rca.experiments.splits import make_split
from dyrca_rca.models.losses import rca_node_loss
from dyrca_rca.eval.metrics import compute_ranking_metrics
from dyrca_rca.models.config import (
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, LR, WEIGHT_DECAY, EPOCHS, EARLY_STOP_PATIENCE,
)


SERVICE_FEATURE_NAMES = [
    "replicas", "ready_replicas", "endpoint_count",
    "in_degree", "out_degree", "endpoint_out_degree",
    "sum_dep_weight", "sum_final_weight", "sum_change_weight",
    "min_topo_conf", "has_trace_mismatch", "has_change",
]


def _f(v: Any, d: float = 0.0) -> float:
    try:
        if v is None:
            return d
        return float(v)
    except (TypeError, ValueError):
        return d


def _services(norm: Dict[str, Any]) -> List[str]:
    return sorted({n["id"] for n in norm["nodes"]
                   if (n.get("attrs") or {}).get("type") == "service"})


def _features_and_adj(norm: Dict[str, Any], services: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
    svc_to_idx = {s: i for i, s in enumerate(services)}
    node_map = {n["id"]: n for n in norm["nodes"]}
    deg_in = {s: 0 for s in services}
    deg_out = {s: 0 for s in services}
    ep_out = {s: 0 for s in services}
    sdep = {s: 0.0 for s in services}
    sfin = {s: 0.0 for s in services}
    schg = {s: 0.0 for s in services}
    mtopo = {s: 1.0 for s in services}
    tmis = {s: 0.0 for s in services}
    haschg = {s: 0.0 for s in services}
    A = torch.zeros(len(services), len(services))
    for e in norm["edges"]:
        s, t = e.get("source"), e.get("target")
        if s not in svc_to_idx or t not in svc_to_idx:
            continue
        a = e.get("attrs") or {}
        etype = str(e.get("edge_type", "dependency"))
        fw = _f(a.get("final_weight"))
        cw = _f(a.get("change_weight"))
        dw = _f(a.get("dependency_weight"))
        tc = _f(a.get("topology_confidence"), 1.0)
        deg_out[s] += 1; deg_in[t] += 1
        sdep[s] += dw; sfin[s] += fw; schg[s] += cw
        if tc < mtopo[s]:
            mtopo[s] = tc
        if cw > 0:
            haschg[s] = 1.0
        if "trace" in etype or "mismatch" in etype:
            tmis[s] = 1.0
        if "endpoint" in etype or "redirect" in etype:
            ep_out[s] += 1
        i, j = svc_to_idx[s], svc_to_idx[t]
        w = _f(a.get("final_weight"), 1.0)
        A[i, j] += w
        A[j, i] += w  # symmetric for GCN propagation
    rows = []
    for s in services:
        n = node_map.get(s, {})
        na = n.get("attrs") or {}
        rows.append([
            _f(na.get("replicas")), _f(na.get("ready_replicas")), _f(na.get("endpoint_count")),
            float(deg_in[s]), float(deg_out[s]), float(ep_out[s]),
            sdep[s], sfin[s], schg[s], mtopo[s], tmis[s], haschg[s],
        ])
    return torch.tensor(rows, dtype=torch.float), A


def service_level_tensors(exp_dir: Path, row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build pre/during service-level graph sequences with a common node ordering."""
    pre_path = exp_dir / "topology" / "pre.json"
    during_path = exp_dir / "topology" / "during.json"
    if not pre_path.exists() or not during_path.exists():
        return None
    pre = read_json(pre_path, required=True)
    during = read_json(during_path, required=True)
    g_pre = apply_edge_weights(str(exp_dir), build_graph_from_topology(pre, phase="pre"))
    g_dur = apply_edge_weights(str(exp_dir), build_graph_from_topology(during, phase="during"))
    npre = normalize_graph(g_pre)
    ndur = normalize_graph(g_dur)
    services = sorted(set(_services(npre)) | set(_services(ndur)))
    if not services:
        return None
    X_pre, A_pre = _features_and_adj(npre, services)
    X_dur, A_dur = _features_and_adj(ndur, services)
    root_svc = root_service_from_row(row)
    label = {s: i for i, s in enumerate(services)}.get(root_svc, -1)
    return {
        "X_pre": X_pre, "X_dur": X_dur, "A_pre": A_pre, "A_dur": A_dur,
        "services": services, "label": label,
        "metadata": {
            "experiment_id": str(row.get("experiment_id", "")),
            "scenario_id": str(row.get("scenario_id", "")),
            "target_service": str(row.get("target_service", "")),
            "root_service": root_svc,
        },
    }


def _norm_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized adjacency D^-1/2 (A+I) D^-1/2."""
    n = A.size(0)
    Ahat = A + torch.eye(n, device=A.device, dtype=A.dtype)
    deg = Ahat.sum(dim=1).clamp_min(1e-8)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(1) * Ahat * dinv.unsqueeze(0)


class EvolveGCNO(nn.Module):
    """EvolveGCN-O (weight-evolution variant).

    For each layer, a GRUCell evolves the flattened [in_dim x out_dim] GCN weight
    matrix across the time steps (pre -> during). At step t:
        W_t = GRU(W_{t-1});  H_t = ReLU( Anorm_t @ X_t @ W_t )
    Layers are stacked; the final-step node embeddings are scored.
    Note: expensive (GRU state = full weight matrix). Prefer EvolveGCN-A on CPU.
    """
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        dims = [in_dim] + [hidden_dim] * num_layers
        self.grus = nn.ModuleList()
        self.layer_in = dims[:-1]
        self.layer_out = dims[1:]
        for li in range(num_layers):
            wsize = dims[li] * dims[li + 1]
            self.grus.append(nn.GRUCell(input_size=wsize, hidden_size=wsize))
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, X_pre: torch.Tensor, A_pre: torch.Tensor,
                X_dur: torch.Tensor, A_dur: torch.Tensor) -> torch.Tensor:
        An = [_norm_adj(A_pre), _norm_adj(A_dur)]
        H = [X_pre, X_dur]
        for li in range(self.num_layers):
            gru = self.grus[li]
            in_d, out_d = self.layer_in[li], self.layer_out[li]
            w = torch.zeros(in_d * out_d, device=X_pre.device)
            H_new = []
            for t in range(len(H)):
                w = gru(w.unsqueeze(0), w.unsqueeze(0))[0]
                W = w.view(in_d, out_d)
                H_new.append(F.relu(An[t] @ H[t] @ W))
            H = H_new
            if li < self.num_layers - 1:
                H = [F.dropout(h, p=self.dropout, training=self.training) for h in H]
        return self.scorer(H[-1]).squeeze(-1)


class EvolveGCNA(nn.Module):
    """EvolveGCN-A (hidden-state-evolution variant, Pareja et al. 2020).

    A per-node LSTMCell aggregates the GCN-propagated features across time. For
    layer l, over steps t=1..T:
        Z_t = Anorm_t @ H^(l)_t          (GCN propagation, shared structure)
        h_t, c_t = LSTMCell_l(Z_t, (h_{t-1}, c_{t-1}))   (per-node, shared weights)
    Layers are stacked; the final-step node embeddings are scored. This is the
    cheaper canonical EvolveGCN variant and runs fast on CPU.
    """
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        dims = [in_dim] + [hidden_dim] * num_layers
        self.cells = nn.ModuleList()
        self.layer_in = dims[:-1]
        self.layer_out = dims[1:]
        for li in range(num_layers):
            self.cells.append(nn.LSTMCell(self.layer_in[li], self.layer_out[li]))
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, X_pre: torch.Tensor, A_pre: torch.Tensor,
                X_dur: torch.Tensor, A_dur: torch.Tensor) -> torch.Tensor:
        An = [_norm_adj(A_pre), _norm_adj(A_dur)]
        H = [X_pre, X_dur]
        N = X_pre.size(0)
        for li in range(self.num_layers):
            cell = self.cells[li]
            out_d = self.layer_out[li]
            hx = torch.zeros(N, out_d, device=X_pre.device)
            cx = torch.zeros(N, out_d, device=X_pre.device)
            H_new = []
            for t in range(len(H)):
                Z = An[t] @ H[t]
                hx, cx = cell(Z, (hx, cx))
                H_new.append(hx)
            H = H_new
            if li < self.num_layers - 1:
                H = [F.relu(h) for h in H]
                H = [F.dropout(h, p=self.dropout, training=self.training) for h in H]
        return self.scorer(H[-1]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward(model, it, device):
    return model(it["X_pre"].to(device), it["A_pre"].to(device),
                 it["X_dur"].to(device), it["A_dur"].to(device))


def build_model(arch: str, in_dim: int) -> nn.Module:
    if arch == "evolvegcn_o":
        return EvolveGCNO(in_dim=in_dim)
    if arch == "evolvegcn_a":
        return EvolveGCNA(in_dim=in_dim)
    raise ValueError(f"unknown arch: {arch}")


def train_temporal(items, val_items, seed, device, arch="evolvegcn_a"):
    set_seed(seed)
    in_dim = items[0]["X_pre"].shape[1]
    model = build_model(arch, in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state, stale = None, None, 0
    rng = random.Random(seed)
    for _ in range(EPOCHS):
        model.train()
        order = list(items); rng.shuffle(order)
        for it in order:
            opt.zero_grad()
            scores = _forward(model, it, device)
            mask = torch.ones(scores.size(0), dtype=torch.bool, device=device)
            loss, _ = rca_node_loss(scores, torch.tensor(it["label"], device=device), mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        if val_items:
            vl = eval_loss(model, val_items, device)
            if best is None or vl < best:
                best, stale = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= EARLY_STOP_PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_loss(model, items, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for it in items:
            scores = _forward(model, it, device)
            mask = torch.ones(scores.size(0), dtype=torch.bool, device=device)
            loss, _ = rca_node_loss(scores, torch.tensor(it["label"], device=device), mask)
            losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def predict(model, items, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for it in items:
            scores = _forward(model, it, device)
            order = torch.argsort(scores, descending=True).cpu().tolist()
            ranked = [it["services"][i] for i in order]
            label_node = it["services"][it["label"]] if 0 <= it["label"] < len(it["services"]) else ""
            rank = None
            for idx, nid in enumerate(ranked, 1):
                if nid == label_node:
                    rank = idx; break
            preds.append({
                "experiment_id": it["metadata"]["experiment_id"],
                "scenario_id": it["metadata"]["scenario_id"],
                "target_service": it["metadata"]["target_service"],
                "label_node_id": label_node,
                "ranked_node_ids": ranked,
                "rank": rank,
                "top_score": float(scores[order[0]].item()) if order else 0.0,
            })
    return preds


def load_items(frame, data_root, cache):
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
            it = service_level_tensors(exp_dir, rd)
            cache[key] = it
        if it is None or it["label"] < 0:
            continue
        items.append(it)
    return items


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--data-root", default="data/experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="random", choices=["random", "leave_service_out", "leave_scenario_out"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--arch", default="evolvegcn_a", choices=["evolvegcn_a", "evolvegcn_o"])
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = read_index(args.index)
    if args.limit:
        frame = frame.head(args.limit)
    splits = make_split(frame, split=args.split, seed=args.seed)
    cache: Dict[str, Any] = {}
    train_items = load_items(splits["train"], args.data_root, cache)
    val_items = load_items(splits["val"], args.data_root, cache)
    test_items = load_items(splits["test"], args.data_root, cache)
    if not train_items and test_items:
        train_items = test_items
    if not test_items:
        test_items = val_items or train_items
    print(f"[temporal] arch={args.arch} train={len(train_items)} val={len(val_items)} test={len(test_items)} device={device}", flush=True)

    model = train_temporal(train_items, val_items, args.seed, device, arch=args.arch)
    preds = predict(model, test_items, device)
    metrics = compute_ranking_metrics(preds)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / "metrics_overall.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "model": args.arch, "seed": args.seed, "split": args.split,
        "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "lr": LR,
        "epochs": EPOCHS, "early_stop": EARLY_STOP_PATIENCE,
        "num_train": len(train_items), "num_test": len(test_items),
        "features": SERVICE_FEATURE_NAMES,
    }, indent=2), encoding="utf-8")
    print(f"[temporal] metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
