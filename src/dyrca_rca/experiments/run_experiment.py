"""Run fixed-config RCA graph experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from dyrca_rca.dataset.materialize import GRAPH_VARIANTS, materialize_experiment, read_index
from dyrca_rca.experiments.evaluate import evaluate_predictions, predict_items
from dyrca_rca.experiments.splits import make_split
from dyrca_rca.experiments.train import MODEL_NAMES, load_items, train_model
from dyrca_rca.models.config import FIXED_GNN_CONFIG


def run_experiment(index: str,
                   graph_variant: str,
                   model_name: str,
                   split: str,
                   seed: int,
                   limit: Optional[int],
                   out_dir: str,
                   data_root: str = "data/experiments",
                   llm_plan_dir: str = "artifacts/llm_plans") -> Dict[str, Any]:
    if graph_variant not in GRAPH_VARIANTS:
        raise ValueError(f"unknown graph variant: {graph_variant}")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {model_name}")
    frame = read_index(index)
    if limit is not None:
        frame = frame.head(limit)
    frame = _usable_rows(frame, data_root=data_root)
    splits = make_split(frame, split=split, seed=seed)

    run_dir = Path(out_dir)
    materialized_root = run_dir / "model_data"
    paths_by_split = {}
    for split_name, split_frame in splits.items():
        paths = []
        for _, row in split_frame.iterrows():
            row_dict = row.to_dict()
            path = materialize_experiment(
                row_dict,
                variant=graph_variant,
                out_dir=str(materialized_root),
                llm_plan_dir=llm_plan_dir,
            )
            paths.append(path)
        paths_by_split[split_name] = paths

    train_items = load_items(paths_by_split.get("train", []))
    val_items = load_items(paths_by_split.get("val", []))
    test_items = load_items(paths_by_split.get("test", []))
    if not train_items and test_items:
        train_items = test_items
    if not test_items:
        test_items = val_items or train_items
    model = train_model(train_items, val_items=val_items, model_name=model_name, seed=seed)
    predictions = predict_items(model, test_items, model_name=model_name)
    eval_result = evaluate_predictions(predictions)

    run_dir.mkdir(parents=True, exist_ok=True)
    eval_result["overall_frame"].to_csv(run_dir / "metrics_overall.csv", index=False)
    eval_result["by_scenario_frame"].to_csv(run_dir / "metrics_by_scenario.csv", index=False)
    _predictions_frame(predictions).to_csv(run_dir / "predictions.csv", index=False)
    config = {
        "run_id": run_dir.name,
        "index": index,
        "data_root": data_root,
        "graph_variant": graph_variant,
        "model": model_name,
        "split": split,
        "seed": seed,
        "limit": limit,
        "fixed_model_config": FIXED_GNN_CONFIG.__dict__,
        "llm_plan_dir": llm_plan_dir,
        "num_train": len(train_items),
        "num_val": len(val_items),
        "num_test": len(test_items),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return {
        "run_dir": str(run_dir),
        "metrics": eval_result["overall"],
        "config": config,
    }


def _usable_rows(frame: pd.DataFrame, data_root: str) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        item = row.to_dict()
        error = item.get("error_message")
        if isinstance(error, str) and error.strip():
            continue
        if not item.get("data_path"):
            item["data_path"] = str(Path(data_root) / str(item.get("experiment_id", "")))
        rows.append(item)
    return pd.DataFrame(rows)


def _predictions_frame(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pred in predictions:
        row = dict(pred)
        row["ranked_node_ids"] = row.pop("ranked_node_ids_str", "|".join(row.get("ranked_node_ids", [])))
        if isinstance(row.get("scores"), dict):
            row["scores"] = json.dumps(row["scores"], sort_keys=True, allow_nan=False)
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-config DY-RCA experiment.")
    parser.add_argument("--data-root", default="data/experiments")
    parser.add_argument("--index", required=True)
    parser.add_argument("--graph-variant", required=True, choices=GRAPH_VARIANTS)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--split", default="random",
                        choices=["random", "leave_service_out", "leave_scenario_out"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llm-plan-dir", default="artifacts/llm_plans")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_experiment(
        index=args.index,
        graph_variant=args.graph_variant,
        model_name=args.model,
        split=args.split,
        seed=args.seed,
        limit=args.limit,
        out_dir=args.out_dir,
        data_root=args.data_root,
        llm_plan_dir=args.llm_plan_dir,
    )
    print(f"wrote results -> {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
