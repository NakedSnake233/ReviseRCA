"""CLI for evaluating graph update quality against topology-derived targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from dyrca_rca.dataset.materialize import read_index
from dyrca_rca.eval.graph_update_metrics import (
    aggregate_by_scenario,
    evaluate_graph_update_rows,
)


def evaluate_index(index_path: str, plans_dir: str, graphs_dir: str,
                   out_dir: str, limit: Optional[int] = None) -> List[str]:
    """Evaluate graph update artifacts for an experiment index."""
    frame = read_index(index_path)
    if limit is not None:
        frame = frame.head(limit)

    rows = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        error_message = row_dict.get("error_message")
        if isinstance(error_message, str) and error_message.strip():
            continue
        rows.append(row_dict)

    metrics = evaluate_graph_update_rows(rows, plans_dir, graphs_dir=graphs_dir)
    by_scenario = aggregate_by_scenario(metrics)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "graph_update_metrics.csv"
    by_scenario_path = output_dir / "graph_update_by_scenario.csv"
    metrics.to_csv(metrics_path, index=False)
    by_scenario.to_csv(by_scenario_path, index=False)
    return [str(metrics_path), str(by_scenario_path)]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph update quality.")
    parser.add_argument("--index", required=True, help="Experiment index parquet/csv/jsonl path.")
    parser.add_argument("--plans-dir", required=True, help="Directory containing GraphUpdatePlan JSON files.")
    parser.add_argument("--graphs-dir", required=True, help="Directory containing executed graph JSON files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for graph update metrics.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max experiment rows to evaluate.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    outputs = evaluate_index(
        index_path=args.index,
        plans_dir=args.plans_dir,
        graphs_dir=args.graphs_dir,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    print("wrote graph update evaluation -> %s" % ", ".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
