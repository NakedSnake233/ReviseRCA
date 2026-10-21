"""
Experiment data loader.
Loads all raw data from a single experiment directory into a unified object.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ExperimentSample:
    """All data for one experiment."""
    exp_id: str
    scenario: str                     # T1/T2/T3/T4
    target_service: str
    time_window: dict                 # {t_collect, t_inject, t_recover, t_end}

    # Raw time-series (aligned to common timestamp axis)
    metrics: pd.DataFrame = None      # [timestamp, metric, service, value]
    logs: pd.DataFrame = None         # [timestamp, service, level, message]
    traces: pd.DataFrame = None       # [trace_id, span_id, parent_id, service, operation, start, duration, status]

    # Topology snapshots
    topo_pre: dict = None
    topo_during: dict = None
    topo_post: dict = None
    topo_diff: dict = None

    # Labels
    ground_truth: dict = None
    change_type: str = "no_change"
    is_dynamic: bool = False
    _exp_dir: str = ""


def _parse_timestamp(ts_str: str) -> pd.Timestamp:
    return pd.Timestamp(ts_str)


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load all metric CSVs, parse labels_json, normalize service names."""
    frames = []
    for csv_file in sorted(metrics_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        if df.empty or "labels_json" not in df.columns:
            continue
        # Parse JSON labels
        labels = df["labels_json"].apply(json.loads)
        df = df.drop(columns=["labels_json"])

        # Extract service name from labels
        metric_name = df["metric_name"].iloc[0] if len(df) > 0 else csv_file.stem

        if metric_name == "cpu_usage":
            df["service"] = labels.apply(
                lambda l: l.get("pod", "").rsplit("-", 2)[0] if "-" in l.get("pod", "") else l.get("pod", "")
            )
        elif metric_name in ("request_rate", "response_flags"):
            df["service"] = labels.apply(lambda l: l.get("destination_workload", ""))
            df["source"] = labels.apply(lambda l: l.get("source_workload", ""))
            if metric_name == "request_rate":
                df["response_code"] = labels.apply(lambda l: l.get("response_code", ""))
            if metric_name == "response_flags":
                df["response_flags"] = labels.apply(lambda l: l.get("response_flags", ""))
        elif metric_name.startswith("latency"):
            df["service"] = labels.apply(lambda l: l.get("destination_workload", ""))
        elif metric_name == "replica_count":
            df["service"] = labels.apply(lambda l: l.get("deployment", l.get("pod", "")))
        elif metric_name == "service_up":
            df["service"] = labels.apply(lambda l: l.get("pod", "").rsplit("-", 2)[0])
        else:
            df["service"] = labels.apply(lambda l: l.get("destination_workload", l.get("pod", "")))

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_logs(logs_dir: Path) -> pd.DataFrame:
    """Load all.jsonl logs."""
    all_path = logs_dir / "all.jsonl"
    if not all_path.exists() or all_path.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "service", "level", "message"])

    df = pd.read_json(all_path, lines=True)
    if "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "service", "level", "message"]].copy()


def load_traces(traces_dir: Path) -> pd.DataFrame:
    """Load trace_summary.csv (fast) or fall back to raw JSON."""
    csv_path = traces_dir / "trace_summary.csv"
    if csv_path.exists() and csv_path.stat().st_size > 50:
        df = pd.read_csv(csv_path)
        df["start_time_unix_nano"] = pd.to_numeric(df["start_time_unix_nano"], errors="coerce")
        df["duration_nano"] = pd.to_numeric(df["duration_nano"], errors="coerce")
        df["duration_ms"] = df["duration_nano"] / 1e6
        # Convert unix nano to datetime
        df["timestamp"] = pd.to_datetime(df["start_time_unix_nano"], unit="ns", utc=True)
        df["status_code"] = df["status_code"].fillna("")
        return df

    # Fallback: parse raw JSON
    raw_path = traces_dir / "traces_raw.json"
    if not raw_path.exists() or raw_path.stat().st_size < 50:
        return pd.DataFrame(columns=["trace_id", "span_id", "parent_span_id",
                                      "service", "operation", "timestamp",
                                      "duration_ms", "status_code"])

    with open(raw_path) as f:
        raw = json.load(f)

    rows = []
    for trace in raw.get("data", []):
        for batch in trace.get("batches", []):
            res_attrs = {a["key"]: a.get("value", {}).get("stringValue", "")
                         for a in batch.get("resource", {}).get("attributes", [])}
            svc = res_attrs.get("service.name", "")
            # Strip namespace suffix
            for suffix in (".boutique", ".trainticket"):
                if svc.endswith(suffix):
                    svc = svc[:-len(suffix)]
            for scope_span in batch.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    start_ns = int(span.get("startTimeUnixNano", 0))
                    end_ns = int(span.get("endTimeUnixNano", 0))
                    dur_ms = (end_ns - start_ns) / 1e6 if end_ns and start_ns else 0
                    rows.append({
                        "trace_id": span.get("traceId", ""),
                        "span_id": span.get("spanId", ""),
                        "parent_span_id": span.get("parentSpanId", ""),
                        "service": svc,
                        "operation": span.get("name", ""),
                        "timestamp": pd.Timestamp(start_ns, unit="ns", tz="UTC") if start_ns else pd.NaT,
                        "duration_ms": dur_ms,
                        "duration_nano": end_ns - start_ns if end_ns and start_ns else 0,
                        "start_time_unix_nano": start_ns,
                        "status_code": span.get("status", {}).get("code", ""),
                        "kind": span.get("kind", ""),
                    })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_topology(topo_dir: Path, phase: str) -> dict:
    """Load a topology snapshot as {nodes, edges}."""
    path = topo_dir / f"{phase}.json"
    if not path.exists():
        return {"nodes": [], "edges": []}

    with open(path) as f:
        raw = json.load(f)

    nodes = []
    for d in raw.get("deployments", []):
        name = d["name"]
        if name in ("loadgenerator",):
            continue
        pods_for_svc = [p for p in raw.get("pods", []) if p.get("service") == name]
        ips = [p.get("ip", "") for p in pods_for_svc if p.get("ip") != "unknown"]
        versions = list(set(p.get("version", "unknown") for p in pods_for_svc))
        nodes.append({
            "name": name,
            "replicas": d.get("replicas", 0),
            "readyReplicas": d.get("readyReplicas", 0),
            "version": versions[0] if len(versions) == 1 else versions,
            "ips": ips,
            "pods": [{"name": p["name"], "node": p.get("node", ""), "ip": p.get("ip", "")}
                     for p in pods_for_svc],
        })

    # Edges from endpoints (which pods back which services)
    edges = []
    for svc in raw.get("services", []):
        name = svc["name"]
        if name in ("loadgenerator", "redis-cart"):
            continue
        ep_ips = svc.get("endpoint_ips", [])
        # Count how many pods from OTHER services are pointing here
        for node in nodes:
            if node["name"] != name:
                overlap = len(set(node["ips"]) & set(ep_ips))
                if overlap > 0:
                    edges.append({
                        "src": node["name"],
                        "dst": name,
                        "weight": overlap,
                    })

    return {"nodes": nodes, "edges": edges, "raw": raw}


def load_experiment(exp_dir: str | Path) -> ExperimentSample:
    """Load a single experiment directory."""
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {exp_dir}")

    # Metadata
    with open(exp_dir / "metadata.json") as f:
        meta = json.load(f)

    # Ground truth
    gt_path = exp_dir / "ground_truth.json"
    ground_truth = {}
    if gt_path.exists():
        with open(gt_path) as f:
            ground_truth = json.load(f)

    # Topology diff
    diff_path = exp_dir / "topology_diff.json"
    topo_diff = {}
    if diff_path.exists():
        with open(diff_path) as f:
            topo_diff = json.load(f)

    # Verification report
    vr_path = exp_dir / "verification_report.json"
    change_type = "no_change"
    if vr_path.exists():
        with open(vr_path) as f:
            vr = json.load(f)
        change_type = vr.get("topology_change_type", "no_change")

    sample = ExperimentSample(
        exp_id=meta.get("experiment_id", exp_dir.name),
        _exp_dir=str(exp_dir),
        scenario=meta.get("scenario_id", "?"),
        target_service=meta.get("target_service", ""),
        time_window=meta.get("time_window", {}),
        metrics=load_metrics(exp_dir / "metrics"),
        logs=load_logs(exp_dir / "logs"),
        traces=load_traces(exp_dir / "traces"),
        topo_pre=load_topology(exp_dir / "topology", "pre"),
        topo_during=load_topology(exp_dir / "topology", "during"),
        topo_post=load_topology(exp_dir / "topology", "post"),
        topo_diff=topo_diff,
        ground_truth=ground_truth,
        change_type=change_type,
        is_dynamic=ground_truth.get("labels", {}).get("is_dynamic_topology_fault", False),
    )

    return sample
