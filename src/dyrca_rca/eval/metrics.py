"""Ranking metrics for service-level RCA."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


METRIC_COLUMNS = ["top1", "top3", "top5", "mrr", "mean_rank", "count"]


def rank_of_label(ranked_node_ids: Iterable[str], label_node_id: str) -> Optional[int]:
    """Return 1-based rank of the label in a ranked list."""
    for idx, node_id in enumerate(ranked_node_ids, start=1):
        if str(node_id) == str(label_node_id):
            return idx
    return None


def prediction_rank(row: Dict[str, Any]) -> Optional[int]:
    ranked = row.get("ranked_node_ids", [])
    if isinstance(ranked, str):
        ranked = [value for value in ranked.split("|") if value]
    label = row.get("label_node_id", "")
    return rank_of_label(ranked, str(label))


def compute_ranking_metrics(predictions: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Compute Top@k, MRR, and mean rank from prediction rows."""
    rows = list(predictions)
    if not rows:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0, "mean_rank": 0.0, "count": 0}

    ranks = []
    for row in rows:
        rank = row.get("rank")
        if rank is None or rank == "":
            rank = prediction_rank(row)
        ranks.append(rank)

    count = len(ranks)
    finite = [int(rank) for rank in ranks if rank is not None and int(rank) > 0]
    top1 = sum(1 for rank in finite if rank <= 1) / float(count)
    top3 = sum(1 for rank in finite if rank <= 3) / float(count)
    top5 = sum(1 for rank in finite if rank <= 5) / float(count)
    mrr = sum(1.0 / rank for rank in finite) / float(count)
    missing_rank = max([rank for rank in finite] + [5]) + 1
    mean_rank = sum((rank if rank is not None else missing_rank) for rank in ranks) / float(count)
    return {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "count": count,
    }


def metrics_by_scenario(predictions: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Compute ranking metrics grouped by scenario_id."""
    rows = list(predictions)
    grouped = {}
    for row in rows:
        scenario = str(row.get("scenario_id", "") or "unknown")
        grouped.setdefault(scenario, []).append(row)
    out_rows = []
    for scenario, group in sorted(grouped.items()):
        metrics = compute_ranking_metrics(group)
        metrics["scenario_id"] = scenario
        out_rows.append(metrics)
    columns = ["scenario_id"] + METRIC_COLUMNS
    return pd.DataFrame(out_rows, columns=columns)


def metrics_overall_frame(predictions: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    metrics = compute_ranking_metrics(predictions)
    return pd.DataFrame([metrics], columns=METRIC_COLUMNS)
