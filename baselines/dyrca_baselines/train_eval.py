"""Training and evaluation helpers for DY-RCA baselines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dyrca_baselines.data_adapter import BaselineDataset, collate_baseline
from dyrca_baselines.models import create_baseline_model
from dyrca_rca.eval.metrics import compute_ranking_metrics, metrics_by_scenario, metrics_overall_frame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(model_name: str,
                       splits: Dict[str, List[Dict[str, Any]]],
                       service_names: List[str],
                       edge_index: torch.Tensor,
                       out_dir: str,
                       seed: int = 42,
                       epochs: int = 80,
                       batch_size: int = 32,
                       lr: float = 1e-3,
                       weight_decay: float = 1e-4,
                       patience: int = 12,
                       gpu: bool = False) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)
    train_ds = BaselineDataset(splits["train"])
    val_ds = BaselineDataset(splits["val"])
    test_ds = BaselineDataset(splits["test"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_baseline)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_baseline)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_baseline)

    model = create_baseline_model(model_name, train_ds[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch, edge_index)
            loss = torch.nn.functional.cross_entropy(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = _eval_loss(model, val_loader, edge_index, device)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = predict(model, test_loader, edge_index, service_names, device)
    eval_result = {
        "overall": compute_ranking_metrics(predictions),
        "overall_frame": metrics_overall_frame(predictions),
        "by_scenario_frame": metrics_by_scenario(predictions),
    }

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_result["overall_frame"].to_csv(run_dir / "metrics_overall.csv", index=False)
    eval_result["by_scenario_frame"].to_csv(run_dir / "metrics_by_scenario.csv", index=False)
    _predictions_frame(predictions).to_csv(run_dir / "predictions.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    config = {
        "baseline": model_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "device": str(device),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "service_names": service_names,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return {"run_dir": str(run_dir), "metrics": eval_result["overall"], "predictions": predictions}


def predict(model: torch.nn.Module, loader: DataLoader, edge_index: torch.Tensor,
            service_names: List[str], device: torch.device) -> List[Dict[str, Any]]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            original = batch
            batch = _to_device(batch, device)
            logits = model(batch, edge_index)
            order = torch.argsort(logits, dim=1, descending=True).cpu().tolist()
            labels = batch["label"].cpu().tolist()
            scores = logits.cpu()
            for idx, ranked_indices in enumerate(order):
                ranked_node_ids = [service_names[item] for item in ranked_indices]
                label_idx = int(labels[idx])
                label_node_id = service_names[label_idx]
                rank = ranked_node_ids.index(label_node_id) + 1 if label_node_id in ranked_node_ids else None
                predictions.append({
                    "experiment_id": original["experiment_id"][idx],
                    "scenario_id": original["scenario_id"][idx],
                    "target_service": original["target_service"][idx],
                    "label_node_id": label_node_id,
                    "ranked_node_ids": ranked_node_ids,
                    "ranked_node_ids_str": "|".join(ranked_node_ids),
                    "rank": rank,
                    "top_score": float(scores[idx, ranked_indices[0]].item()),
                })
    return predictions


def _eval_loss(model: torch.nn.Module, loader: DataLoader,
               edge_index: torch.Tensor, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits = model(batch, edge_index)
            loss = torch.nn.functional.cross_entropy(logits, batch["label"])
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    for key in ["metrics", "traces", "logs", "topology", "label"]:
        out[key] = out[key].to(device)
    return out


def _predictions_frame(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pred in predictions:
        row = dict(pred)
        row["ranked_node_ids"] = row.get("ranked_node_ids_str", "|".join(row.get("ranked_node_ids", [])))
        rows.append(row)
    return pd.DataFrame(rows)
