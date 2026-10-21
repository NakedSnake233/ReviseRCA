#!/usr/bin/env python3
"""Build a read-only symlink view of complete cases for a release.

The collector may leave a metadata-bearing directory when an export or
verification step fails.  Training directly from the batch root would then
silently mix incomplete evidence with valid cases.  This utility creates a
new directory containing only cases with all required topology and telemetry
artifacts; it never copies or edits raw data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def complete(path: Path) -> bool:
    required = [
        "metadata.json", "ground_truth.json",
        "topology/pre.json", "topology/during.json", "topology/post.json",
        "metrics/request_rate.csv", "traces/trace_summary.csv",
        "traces/traces_raw.json", "logs/all.jsonl",
    ]
    if not all((path / item).is_file() and (path / item).stat().st_size > 0 for item in required):
        return False
    try:
        traces = json.loads((path / "traces/traces_raw.json").read_text())
        return bool(traces.get("data"))
    except (OSError, ValueError, AttributeError):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--namespace", default="trainticket")
    p.add_argument("--summary", default=None,
                   help="Optional scheduler summary; include only successful output directories")
    args = p.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if out == root or root in out.parents:
        raise ValueError("release view must be outside the raw root")
    if out.exists():
        if any(out.iterdir()):
            raise ValueError("release view already exists and is non-empty: " + str(out))
    else:
        out.mkdir(parents=True)
    allowed = None
    if args.summary:
        payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        allowed = {
            Path(item["output_dir"]).resolve()
            for item in payload.get("experiments", [])
            if item.get("status") == "success" and item.get("output_dir")
        }
    selected = []
    for case in sorted(root.iterdir()):
        if not case.is_dir() or not complete(case):
            continue
        if allowed is not None and case.resolve() not in allowed:
            continue
        try:
            metadata = json.loads((case / "metadata.json").read_text())
        except (OSError, ValueError):
            continue
        if metadata.get("environment", {}).get("namespace") != args.namespace:
            continue
        link = out / case.name
        link.symlink_to(case, target_is_directory=True)
        selected.append(case.name)
    manifest = {
        "schema_version": "complete-case-view-v1",
        "source_root": str(root),
        "view_root": str(out),
        "namespace": args.namespace,
        "cases": selected,
    }
    (out / "VIEW_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cases": len(selected), "view": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
