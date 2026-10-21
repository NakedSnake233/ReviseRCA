"""
Feature extractor.
Converts raw ExperimentSample data into structured tensors.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.config_provider import get_config_from_metadata

log = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    """Extracted features for one experiment."""
    node_features: torch.Tensor   # [T, N, F_node]
    edge_features: torch.Tensor   # [T, E, F_edge]
    topo_change: torch.Tensor     # [C]
    phase_labels: torch.Tensor    # [T]  0=pre, 1=during, 2=post
    root_cause_idx: int           # index in SERVICES
    change_type_idx: int          # index in CHANGE_TYPES


def _time_bin(timestamps: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
              n_bins: int) -> np.ndarray:
    """Assign timestamps to uniform bins. Returns bin indices [-1, n_bins)."""
    if start >= end or n_bins <= 0:
        return np.full(len(timestamps), -1, dtype=int)
    total_ns = (end - start).value
    offsets = (timestamps - start).dt.total_seconds() * 1e9
    bins = (offsets / (total_ns / n_bins)).astype(int)
    return np.clip(bins, -1, n_bins - 1).astype(int)


def _extract_node_metrics(metrics: pd.DataFrame, t_start: pd.Timestamp,
                          t_end: pd.Timestamp, n_bins: int, cfg) -> torch.Tensor:
    """Extract per-service node features into [T, N, F] tensor."""
    NUM_SERVICES = cfg.NUM_SERVICES
    SERVICE_TO_IDX = cfg.SERVICE_TO_IDX
    T, N, F = n_bins, NUM_SERVICES, 12
    result = np.zeros((T, N, F), dtype=np.float32)

    if metrics.empty:
        return torch.from_numpy(result)

    ts = metrics["timestamp"]
    bins = _time_bin(ts, t_start, t_end, T)
    valid = bins >= 0
    metrics_v = metrics[valid].copy()
    metrics_v["bin"] = bins[valid]

    for bin_idx in range(T):
        m = metrics_v[metrics_v["bin"] == bin_idx]
        if m.empty:
            continue

        for _, row in m.iterrows():
            svc = row.get("service", "")
            si = SERVICE_TO_IDX.get(svc)
            if si is None:
                continue
            metric = row.get("metric_name", "")
            val = float(row.get("value", 0) or 0)

            if metric == "cpu_usage":
                # Aggregate: take mean of all pods for this service
                cur = result[bin_idx, si, 0]
                result[bin_idx, si, 0] = max(cur, val)  # take max CPU
            elif metric == "request_rate":
                src = row.get("source", "")
                si_src = SERVICE_TO_IDX.get(src)
                if si_src == si:
                    pass
                # request_rate_in (dest = this service)
                if si_src is not None and si_src != si:
                    result[bin_idx, si, 1] += val  # request_rate_in
                # request_rate_out (src = this service)
                if src == "" or si_src == si:
                    result[bin_idx, si, 2] += val  # request_rate_out
                # error_rate
                rc = str(row.get("response_code", ""))
                if rc.startswith("5"):
                    result[bin_idx, si, 3] += val
            elif metric == "latency_p50":
                result[bin_idx, si, 4] = val
            elif metric == "latency_p99":
                result[bin_idx, si, 5] = val
            elif metric == "replica_count":
                result[bin_idx, si, 6] = val

    # Normalize error_rate to ratio
    for t in range(T):
        for s in range(N):
            total_in = result[t, s, 1]
            if total_in > 0:
                result[t, s, 3] /= total_in

    return torch.from_numpy(result)


def _extract_node_log_features(logs: pd.DataFrame, t_start: pd.Timestamp,
                                t_end: pd.Timestamp, n_bins: int,
                                node_feats: torch.Tensor, cfg) -> torch.Tensor:
    """Add log-based features (error count, total count) to node_feats."""
    if logs.empty:
        return node_feats

    SERVICE_TO_IDX = cfg.SERVICE_TO_IDX
    T = n_bins
    ts = logs["timestamp"]
    bins = _time_bin(ts, t_start, t_end, T)
    valid = bins >= 0
    logs_v = logs[valid].copy()
    logs_v["bin"] = bins[valid]

    arr = node_feats.numpy()
    for _, row in logs_v.iterrows():
        bin_idx = int(row["bin"])
        svc = row.get("service", "")
        si = SERVICE_TO_IDX.get(svc)
        if si is None:
            continue
        level = str(row.get("level", "")).upper()
        arr[bin_idx, si, 8] += 1  # log_total_count
        if level in ("ERROR", "WARN", "FATAL"):
            arr[bin_idx, si, 7] += 1  # log_error_count

    return torch.from_numpy(arr)


def _extract_node_trace_features(traces: pd.DataFrame, t_start: pd.Timestamp,
                                  t_end: pd.Timestamp, n_bins: int,
                                  node_feats: torch.Tensor, cfg) -> torch.Tensor:
    """Add trace-based features (error rate, avg duration, span count)."""
    if traces.empty:
        return node_feats

    NUM_SERVICES = cfg.NUM_SERVICES
    SERVICE_TO_IDX = cfg.SERVICE_TO_IDX
    T = n_bins
    ts_col = traces.get("timestamp")
    if ts_col is None or ts_col.isna().all():
        return node_feats

    bins = _time_bin(ts_col, t_start, t_end, T)
    valid = bins >= 0
    traces_v = traces[valid].copy()
    traces_v["bin"] = bins[valid]

    arr = node_feats.numpy()
    for _, row in traces_v.iterrows():
        bin_idx = int(row["bin"])
        svc = row.get("service", "")
        si = SERVICE_TO_IDX.get(svc)
        if si is None:
            continue
        arr[bin_idx, si, 11] += 1  # trace_span_count
        status = str(row.get("status_code", ""))
        if status and status != "0":
            arr[bin_idx, si, 9] += 1  # trace error
        dur = float(row.get("duration_ms", 0) or 0)
        arr[bin_idx, si, 10] += dur  # accumulate duration

    # Normalize: error_rate = errors / spans, avg_duration = total / spans
    for t in range(T):
        for s in range(NUM_SERVICES):
            span_count = arr[t, s, 11]
            if span_count > 0:
                arr[t, s, 9] /= span_count   # trace_error_rate
                arr[t, s, 10] /= span_count   # trace_avg_duration

    return torch.from_numpy(arr)


def _extract_edge_features(metrics: pd.DataFrame, traces: pd.DataFrame,
                            t_start: pd.Timestamp, t_end: pd.Timestamp,
                            n_bins: int, cfg) -> torch.Tensor:
    """Extract per-edge features [T, E, F_edge]."""
    NUM_EDGES = cfg.NUM_EDGES
    EDGE_TO_IDX = cfg.EDGE_TO_IDX
    SERVICES = cfg.SERVICES
    T, E, F = n_bins, NUM_EDGES, 4
    result = np.zeros((T, E, F), dtype=np.float32)

    if not metrics.empty:
        ts = metrics["timestamp"]
        bins = _time_bin(ts, t_start, t_end, T)
        valid = bins >= 0
        mv = metrics[valid].copy()
        mv["bin"] = bins[valid]

        for _, row in mv.iterrows():
            bin_idx = int(row["bin"])
            metric = row.get("metric_name", "")
            val = float(row.get("value", 0) or 0)

            if metric == "request_rate":
                src = row.get("source", "")
                dst = row.get("service", "")
                ei = EDGE_TO_IDX.get((src, dst))
                if ei is not None:
                    rc = str(row.get("response_code", ""))
                    result[bin_idx, ei, 0] += val  # call_rate
                    if rc.startswith("5"):
                        result[bin_idx, ei, 1] += val  # error_rate

            elif metric == "response_flags":
                src = row.get("source", "")
                dst = row.get("service", "")
                ei = EDGE_TO_IDX.get((src, dst))
                flags = row.get("response_flags", "-")
                if ei is not None and flags and flags != "-":
                    result[bin_idx, ei, 3] += val  # response_flags

    # Trace-based edge latency
    if not traces.empty:
        ts_col = traces.get("timestamp")
        if ts_col is not None and not ts_col.isna().all():
            bins = _time_bin(ts_col, t_start, t_end, T)
            valid = bins >= 0
            tv = traces[valid].copy()
            tv["bin"] = bins[valid]

            for _, row in tv.iterrows():
                bin_idx = int(row["bin"])
                svc = row.get("service", "")
                kind = row.get("kind", "")
                dur = float(row.get("duration_ms", 0) or 0)
                op = row.get("operation", "")
                # Client spans encode edge: svc → operation target
                if kind == "SPAN_KIND_CLIENT" and svc:
                    # Parse target service from operation name
                    for target in SERVICES:
                        if target in op:
                            ei = EDGE_TO_IDX.get((svc, target))
                            if ei is not None:
                                result[bin_idx, ei, 2] += dur
                                break

    # Normalize error_rate on edges
    for t in range(T):
        for e in range(E):
            if result[t, e, 0] > 0:
                result[t, e, 1] /= result[t, e, 0]

    return torch.from_numpy(result)


def _extract_topo_change(topo_diff: dict, verification: dict | None,
                         topo_pre: dict, topo_during: dict, cfg) -> torch.Tensor:
    """Extract topology change features [C]."""
    TOPO_CHANGE_DIM = cfg.TOPO_CHANGE_DIM
    feat = np.zeros(TOPO_CHANGE_DIM, dtype=np.float32)

    if not topo_diff:
        return torch.from_numpy(feat)

    comparisons = topo_diff.get("comparisons", {})
    pre_during = comparisons.get("pre_vs_during", {})

    # K8s layer changes
    k8s = pre_during.get("k8s_layer", {})
    pods = k8s.get("pod_changes", {})
    feat[0] = len(pods.get("added", []))          # node_added
    feat[1] = len(pods.get("removed", []))         # node_removed

    endpoints = k8s.get("endpoint_changes", {})
    changed_svcs = endpoints.get("changed_services", {})
    feat[3] = len(changed_svcs)                    # edge_removed (approx)

    # Graph layer
    graph = pre_during.get("graph_layer", {})
    node_diff = graph.get("node_diff", {})
    edge_diff = graph.get("edge_diff", {})
    feat[0] += len(node_diff.get("added", []))
    feat[1] += len(node_diff.get("removed", []))
    feat[2] = len(edge_diff.get("added", []))      # edge_added
    feat[3] = max(feat[3], len(edge_diff.get("removed", [])))

    # Replica delta — handle nested structure: changed_deployments.{name}.replicas.{before,after}
    dep_changes = k8s.get("deployment_changes", {})
    changed_deps = dep_changes.get("changed_deployments", dep_changes)
    max_delta = 0.0
    for name, change in changed_deps.items():
        if isinstance(change, dict):
            rep = change.get("replicas", change)
            if isinstance(rep, dict):
                before = rep.get("before", rep.get("replicas_before", 0))
                after = rep.get("after", rep.get("replicas_after", 0))
            else:
                before, after = 0, 0
            max_delta = max(max_delta, abs(float(after) - float(before)))
    feat[4] = max_delta                             # replica_delta_max

    # Version split detection
    if topo_during:
        for node in topo_during.get("nodes", []):
            v = node.get("version", "unknown")
            if isinstance(v, list) and len(v) > 1:
                feat[5] = 1.0                        # version_split
            elif v not in ("unknown", "v1", "latest"):
                # Check if different from pre
                for pre_node in topo_pre.get("nodes", []):
                    if pre_node["name"] == node["name"]:
                        if pre_node.get("version", "unknown") != v:
                            feat[5] = 1.0
                            break

    # Change score
    if verification:
        feat[6] = float(verification.get("change_score", 0))

    return torch.from_numpy(feat)


def _compute_phase_labels(t_start: pd.Timestamp, t_end: pd.Timestamp,
                          time_window: dict, n_bins: int) -> torch.Tensor:
    """Assign phase label to each time bin: 0=pre, 1=during, 2=post."""
    labels = np.zeros(n_bins, dtype=np.int64)
    total_ns = (t_end - t_start).value

    t_inject = pd.Timestamp(time_window.get("t_inject", time_window.get("t_collect")))
    t_recover = pd.Timestamp(time_window.get("t_recover", time_window.get("t_end")))

    inject_bin = int((t_inject - t_start).total_seconds() / (t_end - t_start).total_seconds() * n_bins)
    recover_bin = int((t_recover - t_start).total_seconds() / (t_end - t_start).total_seconds() * n_bins)

    inject_bin = max(0, min(n_bins - 1, inject_bin))
    recover_bin = max(0, min(n_bins, recover_bin))

    labels[inject_bin:recover_bin] = 1
    labels[recover_bin:] = 2

    return torch.from_numpy(labels)


def extract_features(sample, n_bins: int = None, cfg=None) -> FeatureBundle:
    """Extract all features from an ExperimentSample."""
    if cfg is None:
        cfg = get_config_from_metadata(getattr(sample, '_exp_dir', ''))
    if n_bins is None:
        n_bins = cfg.TIME_WINDOW_BINS

    SERVICE_TO_IDX = cfg.SERVICE_TO_IDX
    CHANGE_TYPE_TO_IDX = cfg.CHANGE_TYPE_TO_IDX

    tw = sample.time_window
    t_start = pd.Timestamp(tw.get("t_collect", tw.get("t_start", "")))
    t_end = pd.Timestamp(tw.get("t_end", tw.get("t_end", "")))

    if pd.isna(t_start) or pd.isna(t_end):
        # Fallback: derive from metrics
        if sample.metrics is not None and not sample.metrics.empty:
            t_start = sample.metrics["timestamp"].min()
            t_end = sample.metrics["timestamp"].max()
        else:
            raise ValueError(f"Cannot determine time window for {sample.exp_id}")

    # Node features
    node_feats = _extract_node_metrics(sample.metrics, t_start, t_end, n_bins, cfg)
    node_feats = _extract_node_log_features(sample.logs, t_start, t_end, n_bins, node_feats, cfg)
    node_feats = _extract_node_trace_features(sample.traces, t_start, t_end, n_bins, node_feats, cfg)

    # Edge features
    edge_feats = _extract_edge_features(sample.metrics, sample.traces,
                                          t_start, t_end, n_bins, cfg)

    # Topology change — load verification_report from experiment dir
    import json as _json
    vr = None
    vr_path = getattr(sample, '_exp_dir', '')
    if vr_path:
        vr_file = Path(vr_path) / "verification_report.json"
        if vr_file.exists():
            vr = _json.load(open(vr_file))
    topo_change = _extract_topo_change(sample.topo_diff, vr,
                                        sample.topo_pre, sample.topo_during, cfg)

    # Phase labels
    phase_labels = _compute_phase_labels(t_start, t_end, tw, n_bins)

    # Labels
    root_cause = sample.target_service
    rc_idx = SERVICE_TO_IDX.get(root_cause, 0)
    ct_idx = CHANGE_TYPE_TO_IDX.get(sample.change_type, 0)

    return FeatureBundle(
        node_features=node_feats,
        edge_features=edge_feats,
        topo_change=topo_change,
        phase_labels=phase_labels,
        root_cause_idx=rc_idx,
        change_type_idx=ct_idx,
    )
