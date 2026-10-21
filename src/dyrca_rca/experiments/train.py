"""Fixed-config training loop for small RCA GNN models."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional

import torch

from dyrca_rca.models.config import FIXED_GNN_CONFIG, set_seed
from dyrca_rca.models.gat import EdgeAwareGAT
from dyrca_rca.models.graphsage import WeightedGraphSAGE
from dyrca_rca.models.losses import rca_node_loss


MODEL_NAMES = ["weighted_graphsage", "edge_aware_gat"]


def create_model(model_name: str, in_dim: int, edge_attr_dim: int = 0) -> torch.nn.Module:
    """Instantiate an explicitly requested fixed-config model."""
    if model_name == "weighted_graphsage":
        return WeightedGraphSAGE(
            in_dim=in_dim,
            hidden_dim=FIXED_GNN_CONFIG.hidden_dim,
            num_layers=FIXED_GNN_CONFIG.num_layers,
            dropout=FIXED_GNN_CONFIG.dropout,
        )
    if model_name == "edge_aware_gat":
        return EdgeAwareGAT(
            in_dim=in_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=FIXED_GNN_CONFIG.hidden_dim,
            num_layers=FIXED_GNN_CONFIG.num_layers,
            dropout=FIXED_GNN_CONFIG.dropout,
        )
    raise ValueError(f"unknown model: {model_name}")


def train_model(train_items: List[Dict[str, Any]],
                val_items: Optional[List[Dict[str, Any]]] = None,
                model_name: str = "weighted_graphsage",
                seed: int = 42) -> torch.nn.Module:
    """Train a small GNN using the fixed shared config."""
    if not train_items:
        raise ValueError("train_items is empty")
    set_seed(seed)
    sample = train_items[0]
    model = create_model(
        model_name,
        in_dim=int(sample["x"].shape[1]),
        edge_attr_dim=int(sample["edge_attr"].shape[1]) if sample["edge_attr"].dim() == 2 else 0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FIXED_GNN_CONFIG.lr,
        weight_decay=FIXED_GNN_CONFIG.weight_decay,
    )
    best_state = None
    best_val = None
    stale = 0
    rng = random.Random(seed)

    for _ in range(FIXED_GNN_CONFIG.epochs):
        model.train()
        epoch_items = list(train_items)
        rng.shuffle(epoch_items)
        for item in epoch_items:
            optimizer.zero_grad()
            scores = _forward(model, item, model_name)
            loss, _ = rca_node_loss(scores, item["label"], _candidate_mask(item))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if val_items:
            val_loss = evaluate_loss(model, val_items, model_name)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= FIXED_GNN_CONFIG.early_stop_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_loss(model: torch.nn.Module, items: List[Dict[str, Any]],
                  model_name: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for item in items:
            scores = _forward(model, item, model_name)
            loss, _ = rca_node_loss(scores, item["label"], _candidate_mask(item))
            losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def load_items(paths: Iterable[str]) -> List[Dict[str, Any]]:
    items = []
    for path in paths:
        items.append(torch.load(str(path), map_location="cpu"))
    return items


def _forward(model: torch.nn.Module, item: Dict[str, Any], model_name: str) -> torch.Tensor:
    if model_name == "weighted_graphsage":
        return model(item["x"], item["edge_index"], edge_attr=item["edge_attr"])
    return model(item["x"], item["edge_index"], edge_attr=item["edge_attr"])


def _candidate_mask(item: Dict[str, Any]) -> torch.Tensor:
    node_ids = item.get("node_ids", [])
    mask = []
    for node_id in node_ids:
        text = str(node_id)
        mask.append(
            not text.startswith("pod:")
            and not text.startswith("blackhole:")
            and ":" not in text
        )
    if not any(mask):
        mask = [True for _ in node_ids]
    return torch.tensor(mask, dtype=torch.bool)
