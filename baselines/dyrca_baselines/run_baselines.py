"""Run Eadro and HolisticRCA baselines on the DY-RCA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from dyrca_baselines.data_adapter import build_baseline_bundle, save_bundle_metadata
from dyrca_baselines.train_eval import train_and_evaluate


BASELINES = ["eadro", "holisticrca"]


def run_baselines(index: str,
                  data_root: str,
                  out_dir: str,
                  split: str = "random",
                  seed: int = 42,
                  limit: Optional[int] = None,
                  baselines: Optional[List[str]] = None,
                  epochs: int = 80,
                  batch_size: int = 32,
                  adapter_mode: str = "strict",
                  gpu: bool = False) -> Dict[str, Any]:
    selected = baselines or list(BASELINES)
    bundle = build_baseline_bundle(
        index_path=index,
        data_root=data_root,
        split=split,
        seed=seed,
        limit=limit,
        adapter_mode=adapter_mode,
    )
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_bundle_metadata(bundle, str(root / "data"))

    rows = []
    results = {}
    for name in selected:
        if name not in BASELINES:
            raise ValueError(f"unknown baseline: {name}")
        run_dir = root / "results" / name
        result = train_and_evaluate(
            model_name=name,
            splits=bundle.splits,
            service_names=bundle.service_names,
            edge_index=bundle.edge_index,
            out_dir=str(run_dir),
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            gpu=gpu,
        )
        metrics = result["metrics"]
        results[name] = result
        rows.append({
            "baseline": name,
            "top1": metrics.get("top1", 0.0),
            "top3": metrics.get("top3", 0.0),
            "top5": metrics.get("top5", 0.0),
            "mrr": metrics.get("mrr", 0.0),
            "mean_rank": metrics.get("mean_rank", 0.0),
            "count": metrics.get("count", 0),
            "run_dir": str(run_dir),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "metrics_overall.csv", index=False)
    with (root / "config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "index": index,
            "data_root": data_root,
            "split": split,
            "seed": seed,
            "limit": limit,
            "epochs": epochs,
            "batch_size": batch_size,
            "adapter_mode": adapter_mode,
            "baselines": selected,
            "service_count": len(bundle.service_names),
            "split_sizes": {k: len(v) for k, v in bundle.splits.items()},
            "upstream_repositories": {
                "eadro": "https://github.com/FeiGSSS/Eadro.git",
                "holisticrca": "https://github.com/baiyanquan/HolisticRCA.git",
            },
        }, f, indent=2, sort_keys=True)
    write_report(root, summary)
    return {"summary": summary, "results": results}


def write_report(root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# DY-RCA Baseline Results",
        "",
        "These results are produced by isolated DY-RCA adapters for Eadro and HolisticRCA.",
        "The upstream repositories are stored under `baselines/external/` and are not modified.",
        "",
        "| baseline | Top@1 | Top@3 | Top@5 | MRR | mean_rank | count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['baseline']} | {row['top1']:.4f} | {row['top3']:.4f} | "
            f"{row['top5']:.4f} | {row['mrr']:.4f} | {row['mean_rank']:.4f} | {int(row['count'])} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- Eadro adapter: multi-source telemetry encoder + service dependency graph aggregation.",
        "- HolisticRCA adapter: modality representation learning + entity feature integration + relation graph fusion.",
        "- Strict adapter mode uses pre/during telemetry and static pre-fault topology only.",
        "- Strict adapter mode does not expose topology_diff, post metrics, during topology deltas, pod add/remove flags, or v2 flags.",
        "- Both adapters use the same DY-RCA index, random seed, labels, split, and service-level ranking metrics.",
    ])
    (root / "baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DY-RCA baseline reproductions.")
    parser.add_argument("--index", default="artifacts/dygraph_rca_full/index.csv")
    parser.add_argument("--data-root", default="data/experiments")
    parser.add_argument("--out-dir", default="artifacts/baselines/dyrca_full")
    parser.add_argument("--split", default="random",
                        choices=["random", "leave_service_out", "leave_scenario_out"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline", action="append", choices=BASELINES,
                        help="Baseline to run. Can be supplied multiple times.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--adapter-mode", default="strict", choices=["strict", "leaky_dynamic"],
                        help="Baseline input mode. strict avoids dynamic-topology leakage.")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_baselines(
        index=args.index,
        data_root=args.data_root,
        out_dir=args.out_dir,
        split=args.split,
        seed=args.seed,
        limit=args.limit,
        baselines=args.baseline,
        epochs=args.epochs,
        batch_size=args.batch_size,
        adapter_mode=args.adapter_mode,
        gpu=args.gpu,
    )
    print(f"wrote baseline results -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
