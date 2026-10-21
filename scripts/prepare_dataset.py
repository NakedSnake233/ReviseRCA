#!/usr/bin/env python3
"""
Prepare the RCA dataset: preprocess all experiments into cached .pt files.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --data-dir data/experiments --workers 4
"""
import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.loader import load_experiment
from dataset.feature_extractor import extract_features
from dataset.graph_builder import build_dynamic_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("prepare")


def process_one(exp_dir: Path) -> dict:
    """Process a single experiment, save processed.pt, return summary."""
    cache_path = exp_dir / "processed.pt"
    if cache_path.exists():
        return {"exp_id": exp_dir.name, "status": "cached"}

    try:
        import torch
        sample = load_experiment(exp_dir)
        features = extract_features(sample)
        graph = build_dynamic_graph(sample, features)

        item = {
            "node_features": features.node_features,
            "edge_features": features.edge_features,
            "topo_change": features.topo_change,
            "phase_labels": features.phase_labels,
            "change_mask": graph.change_mask,
            "edge_change_mask": graph.edge_change_mask,
            "edge_index": graph.graphs[0]["edge_index"],
            "root_cause": features.root_cause_idx,
            "change_type": features.change_type_idx,
            "exp_id": sample.exp_id,
            "scenario": sample.scenario,
        }

        torch.save(item, cache_path)
        return {"exp_id": exp_dir.name, "status": "ok",
                "shape": tuple(features.node_features.shape),
                "root_cause": sample.target_service,
                "change_type": sample.change_type}

    except Exception as exc:
        return {"exp_id": exp_dir.name, "status": "error", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Preprocess RCA dataset")
    parser.add_argument("--data-dir", default="data/experiments",
                        help="Directory with experiment dirs")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1=sequential)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    exp_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith(("t1-", "t2-", "t3-", "t4-"))
    ])

    log.info("Found %d experiments in %s", len(exp_dirs), data_dir)

    ok, cached, errors = 0, 0, 0
    t0 = time.time()

    if args.workers <= 1:
        for i, d in enumerate(exp_dirs):
            r = process_one(d)
            if r["status"] == "cached":
                cached += 1
            elif r["status"] == "ok":
                ok += 1
                log.info("[%d/%d] %s  shape=%s  rc=%s  type=%s",
                         i + 1, len(exp_dirs), r["exp_id"],
                         r.get("shape", "?"), r.get("root_cause", "?"),
                         r.get("change_type", "?"))
            else:
                errors += 1
                log.error("[%d/%d] %s  FAILED: %s",
                          i + 1, len(exp_dirs), r["exp_id"], r.get("error", "?"))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, d): d for d in exp_dirs}
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                if r["status"] == "cached":
                    cached += 1
                elif r["status"] == "ok":
                    ok += 1
                else:
                    errors += 1
                    log.error("FAILED %s: %s", r["exp_id"], r.get("error"))
                if (i + 1) % 50 == 0:
                    log.info("Progress: %d/%d (ok=%d cached=%d err=%d)",
                             i + 1, len(exp_dirs), ok, cached, errors)

    elapsed = time.time() - t0
    log.info("=" * 50)
    log.info("Done in %.1fs: processed=%d cached=%d errors=%d",
             elapsed, ok, cached, errors)
    log.info("=" * 50)

    # Print split info
    from dataset.rca_dataset import RCADataset
    for split in ["train", "val", "test"]:
        ds = RCADataset(data_dir=data_dir, split=split)
        dist = ds.get_label_distribution()
        log.info("Split '%s': %d items — scenarios=%s", split, len(ds), dist["scenarios"])


if __name__ == "__main__":
    main()
