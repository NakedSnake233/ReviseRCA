"""Evaluation helpers for trained RCA GNN models."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from dyrca_rca.eval.metrics import compute_ranking_metrics, metrics_by_scenario, metrics_overall_frame
from dyrca_rca.experiments.train import _candidate_mask, _forward


def predict_items(model: torch.nn.Module, items: List[Dict[str, Any]],
                  model_name: str) -> List[Dict[str, Any]]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for item in items:
            scores = _forward(model, item, model_name)
            mask = _candidate_mask(item).to(scores.device)
            candidate_indices = torch.nonzero(mask, as_tuple=False).view(-1)
            candidate_scores = scores[candidate_indices]
            order = torch.argsort(candidate_scores, descending=True)
            ranked_indices = candidate_indices[order].cpu().tolist()
            ranked_node_ids = [item["node_ids"][idx] for idx in ranked_indices]
            label_idx = int(item["label"].item() if hasattr(item["label"], "item") else item["label"])
            label_node_id = item["node_ids"][label_idx] if 0 <= label_idx < len(item["node_ids"]) else ""
            rank = None
            for idx, node_id in enumerate(ranked_node_ids, start=1):
                if node_id == label_node_id:
                    rank = idx
                    break
            metadata = item.get("metadata", {})
            predictions.append({
                "experiment_id": metadata.get("experiment_id", ""),
                "scenario_id": metadata.get("scenario_id", ""),
                "target_service": metadata.get("target_service", ""),
                "label_node_id": label_node_id,
                "ranked_node_ids": ranked_node_ids,
                "ranked_node_ids_str": "|".join(ranked_node_ids),
                "rank": rank,
                "scores": {item["node_ids"][i]: float(scores[i].cpu().item()) for i in ranked_indices},
                "top_score": float(candidate_scores[order[0]].cpu().item()) if order.numel() else 0.0,
            })
    return predictions


def evaluate_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "overall": compute_ranking_metrics(predictions),
        "overall_frame": metrics_overall_frame(predictions),
        "by_scenario_frame": metrics_by_scenario(predictions),
    }
