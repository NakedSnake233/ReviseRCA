"""Bounded, label-free observations. Never reads injection labels or post data."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import timezone
from pathlib import Path

from dateutil.parser import isoparse

from dyrca_rca.system_config import dependency_edges as configured_dependency_edges


OBSERVATION_VERSION = "observation-v1"
TOPOLOGY_FIELDS = {
    "pods": ("name", "service", "ip", "node", "version", "status"),
    "services": ("name", "endpoint_ips"),
    "deployments": ("name", "replicas", "readyReplicas", "version"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def epoch(value):
    dt = isoparse(str(value))
    if dt.tzinfo is None:
        raise ValueError("observation timestamps must include a timezone")
    return dt.astimezone(timezone.utc).timestamp()


def anonymous_id(dataset_id, experiment_id):
    return "incident-" + digest([dataset_id, experiment_id])[:24]


def evidence(kind, data):
    return {"id": kind + ":" + digest(data)[:24], "kind": kind, "data": data}


def dependency_edges(system_id, config=None):
    return configured_dependency_edges(system_id, config)


def load_observation(experiment_dir, dataset_id, system_id, as_of=None,
                     start_at=None, max_rows=50000, dependencies=None):
    """Read pre/during snapshots and a bounded trace prefix ending at as_of.

    Trace spans are eligible only after completion. Unknown timestamps are
    excluded. Metrics/logs are explicitly not consumed in this first version.
    Snapshot time approximates availability; no collector arrival times exist.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    exp = Path(experiment_dir)
    snapshots = []
    for phase in ("pre", "during"):
        path = exp / "topology" / (phase + ".json")
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        epoch(raw["timestamp"])
        clean = {"timestamp": raw["timestamp"], "phase": phase}
        clean["observed_collections"] = sorted(k for k in TOPOLOGY_FIELDS if isinstance(raw.get(k), list))
        for collection, fields in TOPOLOGY_FIELDS.items():
            clean[collection] = [
                {key: row[key] for key in fields if key in row}
                for row in raw.get(collection, []) if isinstance(row, dict)
            ]
            clean[collection].sort(key=lambda row: row.get("name", ""))
        snapshots.append(clean)
    if not snapshots:
        raise ValueError("no timestamped topology snapshots")
    cutoff = as_of or max(snapshots, key=lambda s: epoch(s["timestamp"]))["timestamp"]
    cutoff_time = epoch(cutoff)
    start_time = epoch(start_at) if start_at else float("-inf")
    if start_time > cutoff_time:
        raise ValueError("start_at is later than as_of")
    visible = [s for s in snapshots if epoch(s["timestamp"]) <= cutoff_time]
    if not visible:
        raise ValueError("no topology available at as_of")
    facts = [evidence("topology", s) for s in visible]
    trace_facts, trace_status = _read_traces(exp, start_time, cutoff_time, max_rows)
    facts.extend(trace_facts)
    view = {
        "schema_version": OBSERVATION_VERSION,
        "incident_id": anonymous_id(dataset_id, exp.name),
        "system_id": system_id,
        "as_of": cutoff,
        "start_at": start_at,
        "dependency_edges": dependency_edges(system_id, dependencies),
        "evidence": sorted(facts, key=lambda item: item["id"]),
        "modalities": {
            "traces": trace_status,
            "metrics": "not_loaded_in_v1",
            "logs": "not_loaded_in_v1",
        },
    }
    view["observation_version"] = digest(view)
    return view


def _read_traces(exp, start_time, cutoff_time, max_rows):
    path = exp / "traces" / "trace_summary.csv"
    if not path.exists():
        return [], {"status": "missing_summary", "rows_scanned": 0}
    rows = []
    scanned = unknown = 0
    truncated = False
    with path.open(newline="", encoding="utf-8") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if i >= max_rows:
                truncated = True
                break
            scanned += 1
            try:
                start = int(row["start_time_unix_nano"]) / 1e9
                duration = int(row["duration_nano"]) / 1e9
                if duration < 0:
                    raise ValueError("negative span duration")
            except (KeyError, TypeError, ValueError):
                unknown += 1
                continue
            if start >= start_time and start + duration <= cutoff_time:
                rows.append({k: row.get(k, "") for k in (
                    "trace_id", "span_id", "parent_span_id", "service", "status_code"
                )})
    by_span = {(r["trace_id"], r["span_id"]): r for r in rows if r["span_id"]}
    stats = defaultdict(lambda: {"span_count": 0, "error_count": 0, "status_count": 0})
    routes = defaultdict(int)
    for row in rows:
        service = row["service"]
        if not service:
            continue
        stats[service]["span_count"] += 1
        code = (row["status_code"] or "").strip().upper()
        # OTel UNSET is an observed non-error status, distinct from an absent
        # or unrecognized field. Error rates use only recognized status values.
        valid = code in ("0", "1", "2", "UNSET", "OK", "ERROR",
                         "STATUS_CODE_UNSET", "STATUS_CODE_OK", "STATUS_CODE_ERROR")
        stats[service]["status_count"] += int(valid)
        stats[service]["error_count"] += int(code in ("2", "ERROR", "STATUS_CODE_ERROR"))
        parent = by_span.get((row["trace_id"], row["parent_span_id"]))
        if parent and parent["service"] and parent["service"] != service:
            routes[(parent["service"], service)] += 1
    facts = [evidence("trace_service", dict(value, service=service))
             for service, value in sorted(stats.items())]
    facts.extend(evidence("trace_route", {"source": s, "target": t, "count": n})
                 for (s, t), n in sorted(routes.items()))
    return facts, {"status": "truncated" if truncated else "available",
                   "rows_scanned": scanned, "eligible_rows": len(rows),
                   "unknown_timestamp_rows": unknown}
