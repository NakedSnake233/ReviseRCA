import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.eval.metrics import (
    compute_ranking_metrics,
    metrics_by_scenario,
    metrics_overall_frame,
    rank_of_label,
)


def test_rank_of_label_is_one_based():
    assert rank_of_label(["a", "b", "c"], "a") == 1
    assert rank_of_label(["a", "b", "c"], "c") == 3
    assert rank_of_label(["a", "b", "c"], "x") is None


def test_compute_ranking_metrics_topk_mrr_mean_rank():
    predictions = [
        {"label_node_id": "a", "ranked_node_ids": ["a", "b", "c"]},
        {"label_node_id": "b", "ranked_node_ids": ["a", "b", "c"]},
        {"label_node_id": "e", "ranked_node_ids": ["a", "b", "c", "d", "e"]},
        {"label_node_id": "z", "ranked_node_ids": ["a", "b", "c"]},
    ]

    metrics = compute_ranking_metrics(predictions)

    assert metrics["count"] == 4
    assert metrics["top1"] == 0.25
    assert metrics["top3"] == 0.5
    assert metrics["top5"] == 0.75
    assert round(metrics["mrr"], 6) == round((1.0 + 0.5 + 0.2) / 4.0, 6)
    assert metrics["mean_rank"] == (1 + 2 + 5 + 6) / 4.0


def test_metrics_by_scenario_groups_rows():
    predictions = [
        {"scenario_id": "T1", "label_node_id": "a", "ranked_node_ids": ["a", "b"]},
        {"scenario_id": "T1", "label_node_id": "b", "ranked_node_ids": ["a", "b"]},
        {"scenario_id": "T2", "label_node_id": "c", "ranked_node_ids": ["c", "a"]},
    ]

    frame = metrics_by_scenario(predictions)

    assert list(frame["scenario_id"]) == ["T1", "T2"]
    t1 = frame[frame["scenario_id"] == "T1"].iloc[0]
    t2 = frame[frame["scenario_id"] == "T2"].iloc[0]
    assert t1["top1"] == 0.5
    assert t2["top1"] == 1.0


def test_metrics_overall_frame_columns():
    frame = metrics_overall_frame([
        {"label_node_id": "a", "ranked_node_ids": ["a"]}
    ])

    assert list(frame.columns) == ["top1", "top3", "top5", "mrr", "mean_rank", "count"]
    assert frame.iloc[0]["top1"] == 1.0
