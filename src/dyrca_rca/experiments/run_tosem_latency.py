"""Measure available TOSEM E9 operation boundaries with fixed repetitions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from dyrca_rca.experiments.inference import load_checkpoint, predict_observed
from dyrca_rca.interaction.observation import load_observation
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session
from dyrca_rca.system_config import application_id


def _percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    raw = Path(args.data_root).resolve()
    release = Path(args.release_root).resolve()
    out = Path(args.out_dir).resolve()
    checkpoint = release / "temporal_v3.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.set_num_threads(1)
    model, model_name = load_checkpoint(str(checkpoint), "auto")
    cases = []
    for path in sorted(raw.iterdir()):
        if path.is_dir() and (path / "metadata.json").is_file():
            metadata = json.loads((path / "metadata.json").read_text())
            if application_id(metadata.get("environment", {}).get("namespace")) == args.system:
                cases.append(path)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("no cases for latency measurement")
    samples = {"cold_evidence_graph_ms": [], "cached_scorer_inference_ms": []}
    for case in cases:
        # Warm-up uses the same immutable case packet but is excluded from the
        # reported samples.
        view = load_observation(case, args.dataset_id, args.system, max_rows=args.max_rows)
        session = new_session(view)
        _, cached = replay(session, feature_protocol=getattr(model, "feature_protocol", None))
        for _ in range(args.warmup):
            replay(session, feature_protocol=getattr(model, "feature_protocol", None))
            predict_observed(model, cached, model_name)
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            view = load_observation(case, args.dataset_id, args.system, max_rows=args.max_rows)
            replay(new_session(view), feature_protocol=getattr(model, "feature_protocol", None))
            samples["cold_evidence_graph_ms"].append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            predict_observed(model, cached, model_name)
            samples["cached_scorer_inference_ms"].append((time.perf_counter_ns() - start) / 1e6)
    summaries = {name: {"n": len(values), "p50_ms": _percentile(values, .50),
                        "p95_ms": _percentile(values, .95),
                        "mean_ms": statistics.mean(values) if values else None}
                 for name, values in samples.items()}
    report = {"status": "partial", "system": args.system, "warmup": args.warmup,
              "repeats": args.repeats, "cases": len(cases), "summaries": summaries,
              "pending_operations": ["hypothetical_preview", "accepted_reconstruction_scoring",
                                     "failed_conflict_replay"],
              "limitation": "Only boundaries executable without an injected feedback fixture are measured."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency_summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (out / "latency_samples.jsonl").write_text("".join(json.dumps({"operation": operation, "ms": value}) + "\n"
                                                          for operation, values in samples.items() for value in values))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", default="rule", choices=("rule",))
    parser.add_argument("--system", choices=("boutique", "trainticket"), required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
