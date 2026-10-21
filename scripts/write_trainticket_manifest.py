#!/usr/bin/env python3
"""Write a compact provenance manifest for the Train Ticket release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_config_trainticket_v2.yaml")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config, summary, audit = map(Path, (args.config, args.summary, args.audit))
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    audit_data = json.loads(audit.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "trainticket-release-manifest-v1",
        "dataset_id": "dyrca-v1-trainticket",
        "system": "trainticket",
        "raw_root": str(Path(args.raw_root).resolve()),
        "config": str(config.resolve()),
        "config_sha256": digest(config),
        "summary": str(summary.resolve()),
        "summary_sha256": digest(summary),
        "audit": str(audit.resolve()),
        "audit_sha256": digest(audit),
        "batch_id": summary_data.get("batch_id"),
        "expected_cases": audit_data.get("expected_total"),
        "observed_cases": audit_data.get("observed_total"),
        "complete_telemetry_cases": audit_data.get("cases_with_complete_telemetry"),
        "task_counts": audit_data.get("by_task", {}),
        "workload": {"locust_users": 3, "locust_spawn_rate": 1,
                     "host": "http://ts-ui-dashboard:8080"},
        "fault_parameters_unchanged": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("dataset_id", "expected_cases", "observed_cases",
                       "complete_telemetry_cases")}))


if __name__ == "__main__":
    main()
