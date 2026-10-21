"""Topology differences from effective observations, never injection metadata."""
from collections import defaultdict
from math import log1p

from dyrca_rca.interaction.observation import epoch


PROTOCOL = "interactive-topology-trace-v3-temporal"
FEATURES = [
    "topology_pair_observed", "endpoint_pair_observed", "endpoint_added", "endpoint_removed",
    "endpoint_became_empty", "pod_pair_observed", "version_pair_observed",
    "pod_added", "pod_removed", "version_added", "version_removed",
    "ready_pair_observed", "ready_drop", "ready_recovery", "endpoint_backend_mismatch",
    "endpoint_backend_unresolved", "service_appeared", "service_disappeared",
]


def temporal_features(view):
    snapshots = sorted((e for e in view["evidence"] if e["kind"] == "topology"
                        and epoch(e["data"]["timestamp"]) <= epoch(view["as_of"])),
                       key=lambda e: (epoch(e["data"]["timestamp"]), e["id"]))
    if not snapshots:
        raise ValueError("no active topology evidence")
    old, new = snapshots[0], snapshots[-1]
    pair = epoch(old["data"]["timestamp"]) < epoch(new["data"]["timestamp"])
    def index(fact):
        data = fact["data"]
        services = {s["name"]: s for s in data["services"] if s.get("name")}
        pods = defaultdict(set)
        versions = defaultdict(set)
        for p in data["pods"]:
            if p.get("service"):
                pods[p["service"]].add(p["name"])
                if p.get("version") not in (None, "", "unknown"):
                    versions[p["service"]].add(p["version"])
        ready = {d["name"]: d["readyReplicas"] for d in data["deployments"]
                 if d.get("name") and isinstance(d.get("readyReplicas"), (int, float))}
        backends = {p["ip"]: p.get("service") for p in data["pods"] if p.get("ip")}
        return services, pods, versions, ready, backends
    a, ap, av, ar, _ = index(old)
    b, bp, bv, br, backends = index(new)
    old_collections = old["data"].get("observed_collections", [])
    new_collections = new["data"].get("observed_collections", [])
    pods_known = pair and "pods" in old_collections and "pods" in new_collections
    result = {}
    for service in sorted(set(a) | set(b) | set(ap) | set(bp)):
        f = {k: 0.0 for k in FEATURES}
        f["topology_pair_observed"] = float(pair)
        ea, eb = a.get(service, {}).get("endpoint_ips"), b.get(service, {}).get("endpoint_ips")
        known = pair and isinstance(ea, list) and isinstance(eb, list)
        f["endpoint_pair_observed"] = float(known)
        f["pod_pair_observed"] = float(pods_known)
        # Unknown version labels are not a known version disappearing/appearing.
        version_known = pods_known and all(p.get("version") not in (None, "", "unknown")
                                          for snapshot in (old, new) for p in snapshot["data"]["pods"]
                                          if p.get("service") == service)
        f["version_pair_observed"] = float(version_known)
        if known:
            f["endpoint_added"] = log1p(len(set(eb) - set(ea)))
            f["endpoint_removed"] = log1p(len(set(ea) - set(eb)))
            f["endpoint_became_empty"] = float(bool(ea) and not eb)
        if pods_known:
            for key, left, right in (("pod", ap, bp), ("version", av, bv)):
                if key == "version" and not version_known:
                    continue
                f[key + "_added"] = log1p(len(right.get(service, set()) - left.get(service, set())))
                f[key + "_removed"] = log1p(len(left.get(service, set()) - right.get(service, set())))
        if pair and "services" in old_collections and "services" in new_collections:
            f["service_appeared"] = float(service not in a and service in b)
            f["service_disappeared"] = float(service in a and service not in b)
        if pair and service in ar and service in br:
            f["ready_pair_observed"] = 1.0
            f["ready_drop"] = log1p(max(0, ar[service] - br[service]))
            f["ready_recovery"] = log1p(max(0, br[service] - ar[service]))
        if isinstance(eb, list) and "pods" in new_collections:
            f["endpoint_backend_mismatch"] = sum(ip in backends and backends[ip] != service for ip in eb) / max(1, len(eb))
            f["endpoint_backend_unresolved"] = sum(ip not in backends for ip in eb) / max(1, len(eb))
        result[service] = f
    return result, list(dict.fromkeys([old["id"], new["id"]]))


def enrich_graph(graph, view, mask_temporal=False):
    features, refs = temporal_features(view)
    for node in graph["nodes"]:
        attrs = node.setdefault("attrs", {})
        service = attrs.get("service", attrs.get("parent", node["id"]))
        values = features.get(service, {k: 0.0 for k in FEATURES})
        attrs.update({k: 0.0 for k in FEATURES} if mask_temporal else values)
        attrs["temporal_evidence_refs"] = refs
    graph["attrs"]["feature_protocol"] = PROTOCOL
    graph["attrs"]["temporal_masked"] = mask_temporal
    return graph


def diagnosis_support(view, graph, prediction):
    features, refs = temporal_features(view)
    temporal_signals = [k for k in FEATURES if not k.endswith("observed")]
    candidates = []
    if prediction:
        for service in prediction["ranked_node_ids"][:5]:
            signals = {k: v for k, v in features.get(service, {}).items() if k in temporal_signals and v > 0}
            trace_refs = [e["id"] for e in view["evidence"] if e["kind"] == "trace_service"
                          and e["data"].get("service") == service]
            candidates.append({"candidate": service, "topology_changes": signals,
                               "evidence_refs": refs + trace_refs})
    ranked = prediction["ranked_node_ids"] if prediction else []
    margin = (prediction["scores"][ranked[0]] - prediction["scores"][ranked[1]]) if len(ranked) > 1 else None
    traces = view["modalities"]["traces"]
    needs = []
    if traces["status"] != "available":
        needs.append("补充当前诊断窗口的 trace 汇总；现有数据缺失或被截断")
    if not any(v["topology_pair_observed"] for v in features.values()):
        needs.append("补充截止时间之前的另一份拓扑快照以核对变化")
    return {"candidate_support": candidates, "score_margin": margin, "margin_is_probability": False,
            "needs_evidence": needs, "status": "needs_evidence" if needs else "hypothesis",
            "statement": "模型排序与可观测变化支持的候选假设，不是已验证的故障因果。"}
