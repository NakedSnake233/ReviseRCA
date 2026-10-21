"""Versioned cutoff-scoped behavior features, with label-free identity resolution.

Does not read experiment metadata, injection plans, ground truth or post data.
The original loader remains frozen. Missing identities and measurements stay
unavailable; global/unknown metrics are never assigned to a root service.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dyrca_rca.interaction.observation import epoch

PROTOCOL = "behavior-v1-cutoff-prefix10000-pre120"
FEATURES = ["duration_pre_logp95", "duration_during_logp95", "duration_log_change",
            "duration_pre_available", "duration_during_available", "during_error_rate",
            "during_error_available", "metric_pre_logp99", "metric_during_logp99",
            "metric_log_change", "metric_pre_available", "metric_during_available",
            "duration_pre_log_count", "duration_during_log_count"]
UNKNOWN = {"", "unknown", "none", "null", "__unknown__"}
KNOWN_STATUS = {"0", "1", "2", "UNSET", "OK", "ERROR", "STATUS_CODE_UNSET", "STATUS_CODE_OK", "STATUS_CODE_ERROR"}
ERROR_STATUS = {"2", "ERROR", "STATUS_CODE_ERROR"}


def resolve_service(value, services, pod_services):
    """Exact service, explicit observed Pod, or service FQDN; no fuzzy guesses."""
    value = str(value or "").strip()
    if value.lower() in UNKNOWN:
        return None
    if value in services:
        return value
    if value in pod_services:
        return pod_services[value]
    # A Service DNS name identifies its first component; no substring matching.
    if ".svc" in value:
        first = value.split(".", 1)[0]
        return first if first in services else None
    return None


def load_behavior(case_dir, max_rows=10000):
    root = Path(case_dir)
    snapshots = [json.loads((root / "topology" / (p + ".json")).read_text()) for p in ("pre", "during")]
    pre, cutoff = [epoch(s["timestamp"]) for s in snapshots]
    if cutoff < pre:
        raise ValueError("during snapshot precedes pre snapshot")
    services = {s["name"] for snap in snapshots for s in snap.get("services", [])}
    pod_candidates = defaultdict(set)
    for snap in snapshots:
        for pod in snap.get("pods", []):
            if pod.get("service") in services:
                pod_candidates[pod["name"]].add(pod["service"])
    pods = {n: next(iter(v)) for n, v in pod_candidates.items() if len(v) == 1}
    durations, metrics, status = defaultdict(lambda: [[], []]), defaultdict(lambda: [[], []]), defaultdict(lambda: [0, 0])
    coverage = dict(trace_scanned=0, trace_eligible=0, trace_unresolved=0,
                    metric_scanned=0, metric_eligible=0, metric_unresolved=0,
                    trace_truncated=False, metric_truncated=False)
    trace = root / "traces/trace_summary.csv"
    if trace.exists():
        with trace.open() as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if i >= max_rows:
                    coverage["trace_truncated"] = True; break
                coverage["trace_scanned"] += 1
                try:
                    start, elapsed = int(row["start_time_unix_nano"])/1e9, int(row["duration_nano"])/1e9
                except (KeyError, ValueError, TypeError):
                    continue
                if elapsed < 0 or start < pre-120 or start+elapsed > cutoff:
                    continue
                coverage["trace_eligible"] += 1
                service = resolve_service(row.get("service"), services, pods)
                if service is None:
                    coverage["trace_unresolved"] += 1; continue
                # A span crossing the pre boundary is during, never a pre reference.
                part = int(start + elapsed > pre)
                durations[service][part].append(elapsed*1000)
                code = row.get("status_code", "").strip().upper()
                if part == 1 and code in KNOWN_STATUS:
                    status[service][0] += 1
                    status[service][1] += int(code in ERROR_STATUS)
    metric = root / "metrics/latency_p99.csv"
    if metric.exists():
        with metric.open() as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if i >= max_rows:
                    coverage["metric_truncated"] = True; break
                coverage["metric_scanned"] += 1
                try:
                    timestamp, value = epoch(row["timestamp"]), float(row["value"])
                    labels = json.loads(row["labels_json"])
                except (ValueError, TypeError, KeyError):
                    continue
                if not math.isfinite(value) or value < 0 or not pre-120 <= timestamp <= cutoff:
                    continue
                coverage["metric_eligible"] += 1
                candidates = {resolve_service(labels.get(k), services, pods) for k in
                              ("destination_workload", "destination_service_name", "destination_service", "service", "pod")}
                candidates.discard(None)
                if len(candidates) != 1:
                    coverage["metric_unresolved"] += 1; continue
                service = next(iter(candidates))
                metrics[service][int(timestamp > pre)].append(value)
    values = {}
    for service in sorted(services):
        before, during = durations[service]
        mb, md = metrics[service]
        a, b = [math.log1p(float(np.percentile(v, 95))) if v else 0. for v in (before, during)]
        c, d = [math.log1p(float(np.median(v))) if v else 0. for v in (mb, md)]
        count, errors = status[service]
        values[service] = [a, b, b-a if before and during else 0., float(bool(before)), float(bool(during)),
                           errors/count if count else 0., float(bool(count)), c, d,
                           d-c if mb and md else 0., float(bool(mb)), float(bool(md)),
                           math.log1p(len(before)), math.log1p(len(during))]
    return {"protocol": PROTOCOL, "features": FEATURES, "pre": pre, "cutoff": cutoff,
            "values": values, "coverage": coverage}


def augment(item, behavior, separate_change=False):
    """Append the same features to all methods; optionally remove tag channels.

Topology-difference features remain observed evidence. This removes only the
operation-tag weight/count paths, not real endpoint/node changes.
"""
    result = copy.deepcopy(item)
    nodes = {n["id"]: n.get("attrs", {}) for n in result["graph"]["nodes"]}
    extra = [behavior["values"].get(nodes[n].get("service", n), [0.]*len(FEATURES)) for n in result["node_ids"]]
    result["x"] = torch.cat([result["x"], torch.tensor(extra, dtype=result["x"].dtype)], dim=1)
    result["metadata"]["node_feature_names"] = list(result["metadata"]["node_feature_names"]) + FEATURES
    result["metadata"]["behavior_protocol"] = PROTOCOL
    result["metadata"]["separate_change"] = separate_change
    result["metadata"]["feature_protocol"] = PROTOCOL + ("-separated" if separate_change else "-coupled")
    if separate_change:
        attrs = result["edge_attr"]
        if attrs.numel():
            observed = attrs[:, 4]
            attrs[:, -1] = (.30*attrs[:, 0]+.35*attrs[:, 1]*observed+.25*attrs[:, 2])/(.55+.35*observed)
            attrs[:, 3] = 0
        result["x"][:, 14:16] = 0
        result["x"][:, 17] = 0
        for j in range(result["edge_index"].shape[1]):
            for node in result["edge_index"][:, j]:
                result["x"][node, 14] = max(result["x"][node, 14], attrs[j, -1])
        for edge in result["graph"]["edges"]:
            a = edge["attrs"]
            keys = [k for k in ("dependency_weight", "anomaly_weight", "topology_confidence") if a.get(k) is not None]
            coef = {"dependency_weight": .30, "anomaly_weight": .35, "topology_confidence": .25}
            a["final_weight"] = sum(coef[k]*a[k] for k in keys)/sum(coef[k] for k in keys)
            a["change_weight"] = 0.
            edge["weight"] = a["final_weight"]
    return result
