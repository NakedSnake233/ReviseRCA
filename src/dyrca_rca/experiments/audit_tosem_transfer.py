"""Check paired evidence and assemble the eight-method transfer pilot table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(pilot: Path, transfer: Path, limit: int) -> dict:
    pilot_prediction = pilot / "ranking" / "predictions.jsonl"
    transfer_prediction = transfer / "predictions.jsonl"
    pilot_integrity = _read(pilot / "collection" / "integrity_checks.json")
    transfer_report = _read(transfer / "report.json")
    if pilot_integrity["prediction_sha256_before_labels"] != _sha(pilot_prediction):
        raise ValueError("pilot prediction hash differs from locked hash")
    if transfer_report["prediction_sha256_before_labels"] != _sha(transfer_prediction):
        raise ValueError("baseline prediction hash differs from locked hash")
    anchors = {row["case_id"]: row for row in _rows(pilot_prediction)
               if row["method"] == "DY-RCA-frozen-v3"}
    rows = _rows(transfer_prediction)
    methods = sorted({row["method"] for row in rows})
    if len(anchors) != limit or len(methods) != 6 or len(rows) != 6 * limit:
        raise ValueError("case or method denominator mismatch")
    seen = set()
    for row in rows:
        key = (row["incident_id"], row["method"])
        if key in seen or row["incident_id"] not in anchors:
            raise ValueError("duplicate or unmatched baseline prediction")
        seen.add(key)
        reference = anchors[row["incident_id"]]
        if row["observation_version"] != reference["observation_version"]:
            raise ValueError("observation hash differs across methods")
        if set(row["prediction"]["ranked_node_ids"]) != set(reference["ranked_node_ids"]):
            raise ValueError("candidate inventory differs across methods")
    if len(seen) != len(methods) * len(anchors):
        raise ValueError("missing paired prediction")
    pilot_summary = _read(pilot / "ranking" / "summary.json")
    table = {**pilot_summary, **transfer_report["summary"]}
    comparison = [{"method": method, "normal_n": value["normal"]["n"],
                   "hit1": value["normal"]["hit1"], "hit3": value["normal"]["hit3"],
                   "mrr": value["normal"]["mrr"],
                   "decoy_top1_rate": value["normal"]["decoy_top1_rate"],
                   "benign_n": value["benign"]["n"],
                   "benign_decoy_top1_rate": value["benign"]["decoy_top1_rate"]}
                  for method, value in sorted(table.items())]
    if any(row["normal_n"] != 33 or row["benign_n"] != 8 for row in comparison):
        raise ValueError("unexpected normal/benign denominator")
    return {"status": "paired_frozen_transfer_pilot", "case_count": len(anchors),
            "method_count": len(comparison), "methods": [row["method"] for row in comparison],
            "shared_observation_hashes": True, "shared_candidate_inventories": True,
            "all_predictions_locked_before_scoring": True,
            "pilot_prediction_sha256": _sha(pilot_prediction),
            "transfer_prediction_sha256": _sha(transfer_prediction),
            "comparison": comparison,
            "limitations": ["not new-cohort retraining", "no independent two-reviewer E0 verification",
                            "Node-only not included in this transfer pilot",
                            "benign controls have no abstention decision"]}


def run(args: argparse.Namespace) -> dict:
    raw, out = Path(args.data_root).resolve(), Path(args.out_dir).resolve()
    if out == raw or raw in out.parents or args.limit < 1 or args.planner != "rule":
        raise ValueError("bounded positive limit, rule planner, separate output required")
    report = audit(Path(args.pilot_dir), Path(args.transfer_dir), args.limit)
    report.update({"data_root": str(raw), "seed": args.seed, "planner": args.planner, "limit": args.limit})
    out.mkdir(parents=True, exist_ok=True)
    (out / "shared_evidence_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["comparison"][0]))
        writer.writeheader()
        writer.writerows(report["comparison"])
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pilot-dir", required=True)
    p.add_argument("--transfer-dir", required=True)
    p.add_argument("--limit", type=int, default=41)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--planner", default="rule")
    print(json.dumps(run(p.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
