#!/usr/bin/env python3
"""
Dynamic Topology Change Verification Module.

Verifies that each experiment not only injected a fault,
but that the runtime topology actually changed.

Four verification layers:
  1. K8s Resource Layer  — pod count, IP/node, endpoints, deployment versions
  2. Call Relationship   — trace-based service edges, weights, propagation paths
  3. Graph Structure     — node/edge set diff, edge weights, reachability
  4. Dataset Annotation  — auto-determine topology_change_type + change_score

Usage:
    python -m verifier.topology_verifier --experiment-dir data/experiments/exp_xxx
    python -m verifier.topology_verifier --experiment-dir data/experiments/exp_xxx --pretty

Input (from experiment directory):
    topology/pre.json          — baseline K8s snapshot
    topology/during.json       — during-fault K8s snapshot
    topology/post.json         — post-recovery K8s snapshot
    traces/trace_summary.csv   — optional, trace span data
    traces/traces_raw.json     — optional, raw trace data
    metadata.json              — experiment metadata
    ground_truth.json          — ground truth labels

Output:
    topology_diff.json         — structured multi-layer diff
    verification_report.json   — human-readable summary

topology_diff JSON schema (topology_diff_v1):
    {
      "schema": "topology_diff_v1",
      "experiment_id": "string",
      "verification_timestamp": "ISO-8601",
      "scenario_id": "string",
      "target_service": "string",
      "snapshots_loaded": ["pre", "during", "post"],
      "comparisons": {
        "pre_vs_during": {
          "k8s_layer": {
            "pod_changes": {
              "total_before": 0, "total_after": 0,
              "added": [], "removed": [],
              "rescheduled": [{"pod":"", "old_node":"", "new_node":""}],
              "ip_changed": [{"pod":"", "old_ip":"", "new_ip":""}],
              "version_changed": [{"pod":"", "old_version":"", "new_version":""}],
              "count_changes": {"svc": {"before":0, "after":0, "delta":0}}
            },
            "endpoint_changes": {
              "changed_services": {
                "svc": {"before":[], "after":[], "added":[], "removed":[],
                        "endpoint_count_before":0, "endpoint_count_after":0}
              },
              "total_services_before": 0, "total_services_after": 0
            },
            "deployment_changes": {
              "changed_deployments": {
                "deploy": {"replicas": {"before":0, "after":0},
                           "version": {"before":"", "after":""}}
              }
            }
          },
          "call_layer": {
            "added_edges": [], "removed_edges": [], "weight_changed": [],
            "total_edges_before": 0, "total_edges_after": 0
          },
          "graph_layer": {
            "node_diff": {
              "added": [], "removed": [],
              "modified": [{"service":"", "changes":[]}],
              "total_before": 0, "total_after": 0
            },
            "edge_diff": {
              "added": [], "removed": [], "weight_changed": [],
              "total_before": 0, "total_after": 0
            },
            "reachability_diff": {
              "newly_reachable": [], "newly_unreachable": [],
              "total_reachable_pairs_before": 0,
              "total_reachable_pairs_after": 0
            }
          }
        },
        "during_vs_post": { "... same structure ..." },
        "pre_vs_post":     { "... same structure ..." }
      },
      "annotation": {
        "verification_result": "CHANGED | NOT_CHANGED | PARTIALLY_CHANGED",
        "topology_change_type": "node_scale | node_drift | edge_redirect | ...",
        "detected_types": [],
        "change_score": 0.0,
        "confidence": 0.0,
        "evidence": [],
        "primary_phase_comparison": "pre_vs_during",
        "ground_truth_comparison": {
          "ground_truth_type": "", "detected_type": "", "match": false
        }
      }
    }

verification_report sample:
    {
      "experiment_id": "T1_productcatalogservice_stress4w80p_run042",
      "verification_result": "CHANGED",
      "change_score": 0.62,
      "topology_change_type": "node_scale",
      "summary": "Detected: node_scale | Evidence: productcatalogservice replicas: 1→4",
      "confidence": 0.8,
      "phases": {
        "pre_vs_during": {"changed": true, "highlights": ["..."]},
        "during_vs_post": {"changed": true, "highlights": ["..."]},
        "pre_vs_post": {"changed": false, "highlights": ["..."]}
      },
      "ground_truth_match": true,
      "recovery_verified": true,
      "recommendation": "Topology change verified and recovered. Data is suitable for dataset."
    }
"""
import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("verifier")


# ═══════════════════════════════════════════════════════════════
#  Graph model
# ═══════════════════════════════════════════════════════════════

@dataclass
class ServiceNode:
    name: str
    pods: list = field(default_factory=list)
    replica_count: int = 0
    versions: set = field(default_factory=set)
    ips: set = field(default_factory=set)
    nodes: set = field(default_factory=set)


@dataclass
class ServiceEdge:
    source: str
    target: str
    weight: int = 0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p99: float = 0.0


@dataclass
class ServiceGraph:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    def adjacency(self) -> dict:
        adj = defaultdict(set)
        for e in self.edges:
            adj[e.source].add(e.target)
        return dict(adj)

    def reachable_pairs(self) -> set:
        """All (src, dst) pairs reachable via BFS."""
        adj = self.adjacency()
        pairs = set()
        for start in self.nodes:
            visited = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                if node != start:
                    pairs.add((start, node))
                for nb in adj.get(node, set()):
                    if nb not in visited:
                        queue.append(nb)
        return pairs


# Known call graph for Online Boutique (source → targets)
KNOWN_EDGES = {
    "frontend": ["productcatalogservice", "cartservice",
                  "recommendationservice", "currencyservice",
                  "adservice", "checkoutservice"],
    "recommendationservice": ["productcatalogservice"],
    "checkoutservice": ["cartservice", "productcatalogservice",
                         "shippingservice", "currencyservice",
                         "paymentservice", "emailservice"],
}


# ═══════════════════════════════════════════════════════════════
#  TopologyVerifier
# ═══════════════════════════════════════════════════════════════

class TopologyVerifier:
    """Verifies that runtime topology actually changed during an experiment."""

    def __init__(self, experiment_dir: str):
        self.exp_dir = Path(experiment_dir)
        if not self.exp_dir.is_dir():
            raise FileNotFoundError(
                f"Experiment directory not found: {self.exp_dir}")

        self.metadata = self._load_json("metadata.json")
        self.ground_truth = self._load_json("ground_truth.json", required=False)
        self.snapshots: dict[str, dict] = {}
        self.trace_edges: dict[tuple, ServiceEdge] = {}
        self.graphs: dict[str, ServiceGraph] = {}

    # ── helpers ────────────────────────────────────────────────
    def _load_json(self, filename: str, required: bool = True) -> Optional[dict]:
        path = self.exp_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None

    # ═══════════════════════════════════════════════════════════
    #  Data loading
    # ═══════════════════════════════════════════════════════════

    def _load_data(self):
        """Load topology snapshots and trace data."""
        for phase in ("pre", "during", "post"):
            path = self.exp_dir / "topology" / f"{phase}.json"
            if path.exists():
                with open(path) as f:
                    self.snapshots[phase] = json.load(f)
                log.info("  Loaded topology/%s.json", phase)

        self._load_trace_edges()

    # ── trace edge extraction ──────────────────────────────────

    def _load_trace_edges(self):
        """Extract service-to-service call edges from trace data."""
        csv_path = self.exp_dir / "traces" / "trace_summary.csv"
        raw_path = self.exp_dir / "traces" / "traces_raw.json"
        if csv_path.exists():
            self._edges_from_csv(csv_path)
        elif raw_path.exists():
            self._edges_from_raw(raw_path)

    def _edges_from_csv(self, path: Path):
        """Extract edges from trace_summary.csv."""
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                traces: dict[str, list] = defaultdict(list)
                for row in reader:
                    traces[row.get("trace_id", "")].append(row)

            stats: dict[tuple, dict] = defaultdict(
                lambda: {"count": 0, "errors": 0, "durations": []})

            for _, spans in traces.items():
                by_id = {s.get("span_id", ""): s for s in spans}
                for s in spans:
                    pid = s.get("parent_span_id", "")
                    if pid and pid in by_id:
                        src = by_id[pid].get("service", "")
                        dst = s.get("service", "")
                        if src and dst and src != dst:
                            k = (src, dst)
                            stats[k]["count"] += 1
                            dur = float(s.get("duration_ms", 0))
                            stats[k]["durations"].append(dur)
                            if s.get("status", "") == "error":
                                stats[k]["errors"] += 1

            for (src, dst), st in stats.items():
                durs = sorted(st["durations"])
                p50 = durs[len(durs) // 2] if durs else 0
                p99 = durs[int(len(durs) * 0.99)] if len(durs) > 1 else p50
                self.trace_edges[(src, dst)] = ServiceEdge(
                    source=src, target=dst,
                    weight=st["count"],
                    error_rate=st["errors"] / max(st["count"], 1),
                    latency_p50=p50, latency_p99=p99,
                )
            log.info("  Extracted %d trace edges from CSV", len(self.trace_edges))
        except Exception as exc:
            log.warning("  Failed to parse trace_summary.csv: %s", exc)

    def _edges_from_raw(self, path: Path):
        """Extract edges from traces_raw.json (Jaeger/Tempo format)."""
        try:
            with open(path) as f:
                raw = json.load(f)

            traces_list = raw if isinstance(raw, list) else raw.get("traces", [raw])
            stats: dict[tuple, dict] = defaultdict(
                lambda: {"count": 0, "errors": 0, "durations": []})

            for trace in traces_list:
                spans = trace.get("spans", trace.get("batch", []))
                processes = trace.get("processes", {})

                # Build span → service map
                span_svc: dict[str, str] = {}
                span_err: dict[str, bool] = {}
                span_dur: dict[str, float] = {}

                for sp in spans:
                    sid = sp.get("spanID", sp.get("span_id", ""))
                    # Resolve service name
                    pid = sp.get("processID", "")
                    if processes and pid in processes:
                        svc = processes[pid].get("serviceName", pid)
                    else:
                        tags = {t["key"]: t["value"]
                                for t in sp.get("tags", [])}
                        svc = tags.get("peer.service",
                                       tags.get("service.name", pid))
                    span_svc[sid] = svc
                    span_dur[sid] = int(sp.get("duration", 0)) / 1e6  # ns→ms
                    tags = {t["key"]: t["value"] for t in sp.get("tags", [])}
                    span_err[sid] = tags.get("error") == "true"

                # Parent → child edges
                for sp in spans:
                    child_id = sp.get("spanID", sp.get("span_id", ""))
                    for ref in sp.get("references", []):
                        if ref.get("refType") == "CHILD_OF":
                            parent_id = ref.get("spanID", "")
                            if parent_id in span_svc and child_id in span_svc:
                                src = span_svc[parent_id]
                                dst = span_svc[child_id]
                                if src != dst:
                                    k = (src, dst)
                                    stats[k]["count"] += 1
                                    stats[k]["durations"].append(span_dur[child_id])
                                    if span_err.get(child_id):
                                        stats[k]["errors"] += 1

            for (src, dst), st in stats.items():
                durs = sorted(st["durations"])
                p50 = durs[len(durs) // 2] if durs else 0
                p99 = durs[int(len(durs) * 0.99)] if len(durs) > 1 else p50
                self.trace_edges[(src, dst)] = ServiceEdge(
                    source=src, target=dst,
                    weight=st["count"],
                    error_rate=st["errors"] / max(st["count"], 1),
                    latency_p50=p50, latency_p99=p99,
                )
            log.info("  Extracted %d trace edges from raw JSON",
                     len(self.trace_edges))
        except Exception as exc:
            log.warning("  Failed to parse traces_raw.json: %s", exc)

    # ═══════════════════════════════════════════════════════════
    #  Graph building
    # ═══════════════════════════════════════════════════════════

    def _build_graphs(self):
        for phase, snap in self.snapshots.items():
            self.graphs[phase] = self._build_graph(snap)
            g = self.graphs[phase]
            log.info("  %s graph: %d nodes, %d edges",
                     phase, len(g.nodes), len(g.edges))

    def _build_graph(self, snapshot: dict) -> ServiceGraph:
        graph = ServiceGraph()
        pods = snapshot.get("pods", [])
        deployments = snapshot.get("deployments", [])

        # Nodes from deployments
        for dep in deployments:
            name = dep["name"]
            dep_pods = [p for p in pods if p.get("service") == name]
            graph.nodes[name] = ServiceNode(
                name=name,
                pods=dep_pods,
                replica_count=dep.get("replicas", len(dep_pods)),
                versions={p.get("version", "unknown") for p in dep_pods},
                ips={p.get("ip", "") for p in dep_pods if p.get("ip")},
                nodes={p.get("node", "") for p in dep_pods if p.get("node")},
            )

        # Pods not covered by a deployment
        covered = set(graph.nodes)
        for pod in pods:
            svc = pod.get("service", "")
            if svc and svc not in covered:
                if svc not in graph.nodes:
                    graph.nodes[svc] = ServiceNode(name=svc)
                n = graph.nodes[svc]
                n.pods.append(pod)
                if pod.get("ip"):
                    n.ips.add(pod["ip"])
                if pod.get("version"):
                    n.versions.add(pod["version"])
                if pod.get("node"):
                    n.nodes.add(pod["node"])
                n.replica_count = len(n.pods)

        # Edges from known topology, enriched with trace data
        seen_edges: set[tuple] = set()
        for src, dsts in KNOWN_EDGES.items():
            for dst in dsts:
                if src in graph.nodes and dst in graph.nodes:
                    edge = ServiceEdge(source=src, target=dst)
                    te = self.trace_edges.get((src, dst))
                    if te:
                        edge.weight = te.weight
                        edge.error_rate = te.error_rate
                        edge.latency_p50 = te.latency_p50
                        edge.latency_p99 = te.latency_p99
                    graph.edges.append(edge)
                    seen_edges.add((src, dst))

        # Trace edges not in KNOWN_EDGES
        for (src, dst), te in self.trace_edges.items():
            if (src, dst) not in seen_edges:
                if src in graph.nodes and dst in graph.nodes:
                    graph.edges.append(te)

        return graph

    # ═══════════════════════════════════════════════════════════
    #  Layer 1 — K8s Resource Layer
    # ═══════════════════════════════════════════════════════════

    def _diff_k8s_layer(self, before: dict, after: dict) -> dict:
        return {
            "pod_changes": self._diff_pods(before, after),
            "endpoint_changes": self._diff_endpoints(before, after),
            "deployment_changes": self._diff_deployments(before, after),
        }

    def _diff_pods(self, before: dict, after: dict) -> dict:
        bp = {p["name"]: p for p in before.get("pods", [])}
        ap = {p["name"]: p for p in after.get("pods", [])}

        added = sorted(ap.keys() - bp.keys())
        removed = sorted(bp.keys() - ap.keys())
        common = bp.keys() & ap.keys()

        rescheduled, ip_changed, version_changed = [], [], []
        for name in common:
            b, a = bp[name], ap[name]
            if b.get("node") and b.get("node") != a.get("node"):
                rescheduled.append(
                    {"pod": name, "old_node": b["node"], "new_node": a["node"]})
            if b.get("ip") != a.get("ip"):
                ip_changed.append(
                    {"pod": name, "old_ip": b.get("ip"), "new_ip": a.get("ip")})
            if b.get("version") != a.get("version"):
                version_changed.append(
                    {"pod": name,
                     "old_version": b.get("version"), "new_version": a.get("version")})

        # Per-service count changes
        sc_b: dict[str, int] = defaultdict(int)
        sc_a: dict[str, int] = defaultdict(int)
        for p in before.get("pods", []):
            sc_b[p.get("service", "?")] += 1
        for p in after.get("pods", []):
            sc_a[p.get("service", "?")] += 1
        count_changes: dict[str, dict] = {}
        for svc in sc_b.keys() | sc_a.keys():
            cb, ca = sc_b[svc], sc_a[svc]
            if cb != ca:
                count_changes[svc] = {"before": cb, "after": ca, "delta": ca - cb}

        return {
            "total_before": len(bp), "total_after": len(ap),
            "added": added, "removed": removed,
            "rescheduled": rescheduled,
            "ip_changed": ip_changed,
            "version_changed": version_changed,
            "count_changes": count_changes,
        }

    def _diff_endpoints(self, before: dict, after: dict) -> dict:
        bs = {s["name"]: s for s in before.get("services", [])}
        as_ = {s["name"]: s for s in after.get("services", [])}
        changes: dict[str, dict] = {}
        for name in bs.keys() | as_.keys():
            b_ips = set(bs.get(name, {}).get("endpoint_ips", []))
            a_ips = set(as_.get(name, {}).get("endpoint_ips", []))
            if b_ips != a_ips:
                changes[name] = {
                    "before": sorted(b_ips), "after": sorted(a_ips),
                    "added": sorted(a_ips - b_ips),
                    "removed": sorted(b_ips - a_ips),
                    "endpoint_count_before": len(b_ips),
                    "endpoint_count_after": len(a_ips),
                }
        return {
            "changed_services": changes,
            "total_services_before": len(bs),
            "total_services_after": len(as_),
        }

    def _diff_deployments(self, before: dict, after: dict) -> dict:
        bd = {d["name"]: d for d in before.get("deployments", [])}
        ad = {d["name"]: d for d in after.get("deployments", [])}
        changes: dict[str, dict] = {}
        for name in bd.keys() | ad.keys():
            b, a = bd.get(name, {}), ad.get(name, {})
            r_chg = b.get("replicas", 0) != a.get("replicas", 0)
            v_chg = b.get("version") != a.get("version")
            if r_chg or v_chg:
                ch: dict[str, Any] = {}
                if r_chg:
                    ch["replicas"] = {
                        "before": b.get("replicas", 0),
                        "after": a.get("replicas", 0)}
                if v_chg:
                    ch["version"] = {
                        "before": b.get("version", "?"),
                        "after": a.get("version", "?")}
                changes[name] = ch
        return {"changed_deployments": changes}

    # ═══════════════════════════════════════════════════════════
    #  Layer 2 — Call Relationship Layer
    # ═══════════════════════════════════════════════════════════

    def _diff_call_layer(self, ga: ServiceGraph, gb: ServiceGraph) -> dict:
        ea = {(e.source, e.target): e for e in ga.edges}
        eb = {(e.source, e.target): e for e in gb.edges}
        all_keys = ea.keys() | eb.keys()

        added, removed, weight_changed = [], [], []
        for k in all_keys:
            before = ea.get(k)
            after = eb.get(k)
            if before and not after:
                removed.append({"source": k[0], "target": k[1],
                                "weight_before": before.weight})
            elif after and not before:
                added.append({"source": k[0], "target": k[1],
                              "weight_after": after.weight})
            elif before and after:
                if (before.weight != after.weight or
                        before.error_rate != after.error_rate or
                        before.latency_p99 != after.latency_p99):
                    dpct = 0.0
                    if before.weight > 0:
                        dpct = (after.weight - before.weight) / before.weight * 100
                    weight_changed.append({
                        "source": k[0], "target": k[1],
                        "weight_before": before.weight,
                        "weight_after": after.weight,
                        "weight_delta_pct": round(dpct, 1),
                        "error_rate_before": round(before.error_rate, 4),
                        "error_rate_after": round(after.error_rate, 4),
                        "latency_p99_before": round(before.latency_p99, 2),
                        "latency_p99_after": round(after.latency_p99, 2),
                    })

        return {
            "added_edges": added, "removed_edges": removed,
            "weight_changed": weight_changed,
            "total_edges_before": len(ea),
            "total_edges_after": len(eb),
        }

    # ═══════════════════════════════════════════════════════════
    #  Layer 3 — Graph Structure Layer
    # ═══════════════════════════════════════════════════════════

    def _diff_graph_layer(self, ga: ServiceGraph, gb: ServiceGraph) -> dict:
        na = set(ga.nodes)
        nb = set(gb.nodes)

        # --- node diff ---
        added_nodes = sorted(nb - na)
        removed_nodes = sorted(na - nb)
        modified_nodes = []
        for name in na & nb:
            b, a = ga.nodes[name], gb.nodes[name]
            changes = []
            if b.replica_count != a.replica_count:
                changes.append(
                    f"replicas: {b.replica_count} -> {a.replica_count}")
            if b.versions != a.versions:
                changes.append(f"versions: {sorted(b.versions)} -> {sorted(a.versions)}")
            if b.ips != a.ips:
                changes.append("ips changed")
            if b.nodes != a.nodes:
                changes.append(f"nodes: {sorted(b.nodes)} -> {sorted(a.nodes)}")
            if changes:
                modified_nodes.append({"service": name, "changes": changes})

        # --- edge diff ---
        ea = {(e.source, e.target) for e in ga.edges}
        eb = {(e.source, e.target) for e in gb.edges}
        added_edges = [f"{s}->{t}" for s, t in sorted(eb - ea)]
        removed_edges = [f"{s}->{t}" for s, t in sorted(ea - eb)]

        ea_map = {(e.source, e.target): e for e in ga.edges}
        eb_map = {(e.source, e.target): e for e in gb.edges}
        weight_changed = []
        for k in ea & eb:
            b, a = ea_map[k], eb_map[k]
            if b.weight != a.weight or b.error_rate != a.error_rate:
                weight_changed.append({
                    "edge": f"{k[0]}->{k[1]}",
                    "weight_before": b.weight, "weight_after": a.weight,
                    "error_rate_before": round(b.error_rate, 4),
                    "error_rate_after": round(a.error_rate, 4),
                })

        # --- reachability ---
        ra = ga.reachable_pairs()
        rb = gb.reachable_pairs()

        return {
            "node_diff": {
                "added": added_nodes, "removed": removed_nodes,
                "modified": modified_nodes,
                "total_before": len(na), "total_after": len(nb),
            },
            "edge_diff": {
                "added": added_edges, "removed": removed_edges,
                "weight_changed": weight_changed,
                "total_before": len(ea), "total_after": len(eb),
            },
            "reachability_diff": {
                "newly_reachable": [f"{s}->{t}" for s, t in sorted(rb - ra)],
                "newly_unreachable": [f"{s}->{t}" for s, t in sorted(ra - rb)],
                "total_reachable_pairs_before": len(ra),
                "total_reachable_pairs_after": len(rb),
            },
        }

    # ═══════════════════════════════════════════════════════════
    #  Layer 4 — Annotation (change_type + change_score)
    # ═══════════════════════════════════════════════════════════

    def _annotate(self, results: dict) -> dict:
        primary = results.get("comparisons", {}).get("pre_vs_during", {})
        evidence: list[str] = []
        change_types: list[str] = []

        k8s = primary.get("k8s_layer", {})
        pod = k8s.get("pod_changes", {})
        ep = k8s.get("endpoint_changes", {})
        graph_layer = primary.get("graph_layer", {})
        call_layer = primary.get("call_layer", {})

        # ── endpoint empty → edge_removal ──
        for svc, ch in ep.get("changed_services", {}).items():
            if ch.get("endpoint_count_after", 1) == 0:
                change_types.append("edge_removal")
                evidence.append(f"Service {svc} endpoints became empty")

        # ── endpoint redirect → edge_redirect ──
        for svc, ch in ep.get("changed_services", {}).items():
            if (ch.get("endpoint_count_after", -1) > 0 and
                    ch.get("endpoint_count_before", -1) > 0 and
                    ch.get("removed") and ch.get("added")):
                change_types.append("edge_redirect")
                evidence.append(
                    f"Service {svc} endpoints redirected: "
                    f"removed={ch['removed']}, added={ch['added']}")

        # ── pod rescheduled → node_drift ──
        if pod.get("rescheduled"):
            change_types.append("node_drift")
            evidence.append(
                f"{len(pod['rescheduled'])} pods rescheduled to different nodes")

        # ── replica count change → node_scale ──
        sig_count = {k: v for k, v in pod.get("count_changes", {}).items()
                     if abs(v["delta"]) >= 1}
        if sig_count:
            change_types.append("node_scale")
            for svc, ch in sig_count.items():
                evidence.append(
                    f"Service {svc} replicas: {ch['before']} -> {ch['after']}")

        # ── version change → edge_split ──
        for mod in graph_layer.get("node_diff", {}).get("modified", []):
            if any("versions" in c for c in mod.get("changes", [])):
                change_types.append("edge_split")
                evidence.append(
                    f"Service {mod['service']} has version changes")

        # ── edge weight shift → weight_shift ──
        sig_weight = [w for w in call_layer.get("weight_changed", [])
                      if abs(w.get("weight_delta_pct", 0)) > 20]
        if sig_weight:
            change_types.append("weight_shift")
            for w in sig_weight:
                evidence.append(
                    f"Edge {w['source']}->{w['target']} weight "
                    f"changed {w['weight_delta_pct']}%")

        # ── graph-level edge add/remove ──
        edge_d = graph_layer.get("edge_diff", {})
        if edge_d.get("added"):
            evidence.append(f"New call-graph edges: {edge_d['added']}")
        if edge_d.get("removed"):
            evidence.append(
                f"Removed call-graph edges: {edge_d['removed']}")

        # ── reachability ──
        reach = graph_layer.get("reachability_diff", {})
        if reach.get("newly_unreachable"):
            evidence.append(
                f"Newly unreachable: {reach['newly_unreachable']}")

        # ── final type ──
        unique_types = list(dict.fromkeys(change_types))  # deduplicate, preserve order
        if not unique_types:
            topo_type = "no_change"
        elif len(unique_types) == 1:
            topo_type = unique_types[0]
        else:
            topo_type = "mixed"

        # ── score ──
        score = self._compute_score(primary, graph_layer, call_layer)

        # ── ground-truth comparison ──
        gt_type = ""
        gt_match = False
        if self.ground_truth:
            gt_type = self.ground_truth.get("topology_change_type", "")
            if gt_type:
                gt_match = self._gt_match(gt_type, topo_type, unique_types)

        # ── verification result ──
        # Critical types guarantee CHANGED regardless of numeric score
        critical = {"edge_redirect", "edge_removal", "node_drift",
                    "node_scale", "edge_split"}
        if unique_types and any(t in critical for t in unique_types):
            verdict = "CHANGED"
            score = max(score, 0.30)
        elif score >= 0.30:
            verdict = "CHANGED"
        elif score >= 0.10:
            verdict = "PARTIALLY_CHANGED"
        else:
            verdict = "NOT_CHANGED"

        return {
            "verification_result": verdict,
            "topology_change_type": topo_type,
            "detected_types": unique_types,
            "change_score": round(score, 4),
            "confidence": self._confidence(primary, evidence),
            "evidence": evidence,
            "primary_phase_comparison": "pre_vs_during",
            "ground_truth_comparison": (
                {"ground_truth_type": gt_type, "detected_type": topo_type,
                 "match": gt_match}
                if self.ground_truth else None),
        }

    # ── ground-truth matching ──────────────────────────────────

    # Aliases: ground truth type → verifier detected type
    _GT_ALIASES = {
        "instance_count_change": "node_scale",
        "node_count_change": "node_scale",
        "replica_change": "node_scale",
        "edge_misconfiguration": "edge_redirect",
        "selector_poison": "edge_redirect",
        "version_coexistence": "edge_split",
        "version_split": "edge_split",
        "traffic_split": "edge_split",
        "pod_reschedule": "node_drift",
        "node_reassignment": "node_drift",
    }

    def _gt_match(self, gt_type: str, detected: str,
                  all_detected: list) -> bool:
        if gt_type == detected:
            return True
        alias = self._GT_ALIASES.get(gt_type, "")
        if alias == detected:
            return True
        if gt_type in all_detected or (alias and alias in all_detected):
            return True
        if gt_type in str(all_detected) or detected in gt_type:
            return True
        return False

    # ── score computation ──────────────────────────────────────

    def _compute_score(self, primary: dict,
                       gl: dict, cl: dict) -> float:
        """
        change_score in [0, 1].
        Weights: k8s 30%, graph 30%, call-weights 20%, reachability 20%.
        """
        k8s = primary.get("k8s_layer", {})
        pod = k8s.get("pod_changes", {})
        ep = k8s.get("endpoint_changes", {})

        # 1) K8s resource score
        total_pods = max(pod.get("total_before", 1), 1)
        pod_chg = (len(pod.get("added", [])) + len(pod.get("removed", []))
                   + len(pod.get("rescheduled", [])) + len(pod.get("ip_changed", [])))
        total_svcs = max(ep.get("total_services_before", 1), 1)
        ep_chg = len(ep.get("changed_services", {}))
        dep_chg = len(k8s.get("deployment_changes", {})
                      .get("changed_deployments", {}))
        # Endpoints are the most important K8s signal for topology change
        k8s_score = min(1.0, (
            pod_chg / total_pods * 0.3
            + ep_chg / max(total_svcs, 1) * 0.5
            + dep_chg / max(total_svcs, 1) * 0.2) * 2.5)

        # 2) Graph structure score
        nd = gl.get("node_diff", {})
        ed = gl.get("edge_diff", {})
        tn = max(nd.get("total_before", 1), 1)
        te = max(ed.get("total_before", 1), 1)
        n_chg = len(nd.get("added", [])) + len(nd.get("removed", [])) + \
            len(nd.get("modified", []))
        e_chg = len(ed.get("added", [])) + len(ed.get("removed", [])) + \
            len(ed.get("weight_changed", []))
        graph_score = min(1.0, (
            n_chg / tn * 0.5 + e_chg / max(te, 1) * 0.5) * 2)

        # 3) Call weight score
        wc = cl.get("weight_changed", [])
        if wc:
            avg_d = sum(abs(w.get("weight_delta_pct", 0)) for w in wc) / len(wc)
            wt_score = min(1.0, avg_d / 100)
        else:
            wt_score = 0.0

        # 4) Reachability score
        rd = gl.get("reachability_diff", {})
        total_pairs = max(rd.get("total_reachable_pairs_before", 1), 1)
        r_chg = len(rd.get("newly_reachable", [])) + \
            len(rd.get("newly_unreachable", []))
        reach_score = min(1.0, r_chg / total_pairs * 3)

        return (k8s_score * 0.30 + graph_score * 0.30
                + wt_score * 0.20 + reach_score * 0.20)

    def _confidence(self, primary: dict, evidence: list) -> float:
        if not evidence:
            return 0.0
        conf = 0.5
        conf += min(0.3, len(evidence) * 0.1)
        k8s = primary.get("k8s_layer", {})
        if k8s.get("pod_changes", {}).get("count_changes"):
            conf += 0.1
        if k8s.get("endpoint_changes", {}).get("changed_services"):
            conf += 0.1
        return min(1.0, conf)

    # ═══════════════════════════════════════════════════════════
    #  Phase comparison
    # ═══════════════════════════════════════════════════════════

    def _compare_phases(self, pa: str, pb: str) -> dict:
        sa = self.snapshots.get(pa)
        sb = self.snapshots.get(pb)
        if not sa or not sb:
            return {"skipped": True,
                    "reason": f"Missing snapshot: {pa} or {pb}"}
        ga = self.graphs.get(pa, ServiceGraph())
        gb = self.graphs.get(pb, ServiceGraph())
        return {
            "k8s_layer": self._diff_k8s_layer(sa, sb),
            "call_layer": self._diff_call_layer(ga, gb),
            "graph_layer": self._diff_graph_layer(ga, gb),
        }

    # ═══════════════════════════════════════════════════════════
    #  Main entry
    # ═══════════════════════════════════════════════════════════

    def verify(self) -> dict:
        """Run all verification layers and return results."""
        log.info("=" * 60)
        log.info("  Topology Verification: %s", self.exp_dir.name)
        log.info("=" * 60)

        self._load_data()
        self._build_graphs()

        results: dict[str, Any] = {
            "schema": "topology_diff_v1",
            "experiment_id": self.metadata.get("experiment_id", "unknown"),
            "verification_timestamp": datetime.now(
                tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scenario_id": self.metadata.get("scenario_id", ""),
            "target_service": self.metadata.get("target_service", ""),
            "snapshots_loaded": list(self.snapshots.keys()),
            "comparisons": {},
        }

        for pa, pb in [("pre", "during"), ("during", "post"), ("pre", "post")]:
            if pa in self.snapshots and pb in self.snapshots:
                comp = self._compare_phases(pa, pb)
                results["comparisons"][f"{pa}_vs_{pb}"] = comp
                k8s_cc = len(comp.get("k8s_layer", {})
                             .get("pod_changes", {}).get("count_changes", {}))
                log.info("  %s vs %s: k8s_count_changes=%d, call_weight_changes=%d",
                         pa, pb, k8s_cc,
                         len(comp.get("call_layer", {}).get("weight_changed", [])))

        results["annotation"] = self._annotate(results)
        self._write_results(results)
        return results

    # ═══════════════════════════════════════════════════════════
    #  Output
    # ═══════════════════════════════════════════════════════════

    def _write_results(self, results: dict):
        _make_serializable(results)
        diff_path = self.exp_dir / "topology_diff.json"
        with open(diff_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("  Saved topology_diff.json")

        report = self._build_report(results)
        report_path = self.exp_dir / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info("  Saved verification_report.json")

    def _build_report(self, results: dict) -> dict:
        ann = results.get("annotation", {})

        # Phase highlights
        phases: dict[str, Any] = {}
        for ck, comp in results.get("comparisons", {}).items():
            if comp.get("skipped"):
                continue
            hl: list[str] = []
            pod = comp.get("k8s_layer", {}).get("pod_changes", {})
            ep = comp.get("k8s_layer", {}).get("endpoint_changes", {})
            for svc, ch in pod.get("count_changes", {}).items():
                hl.append(f"{svc} replicas: {ch['before']} -> {ch['after']}")
            for r in pod.get("rescheduled", []):
                hl.append(
                    f"Pod {r['pod']} moved {r['old_node']} -> {r['new_node']}")
            for svc, ch in ep.get("changed_services", {}).items():
                hl.append(
                    f"{svc} endpoints: {ch['endpoint_count_before']}"
                    f" -> {ch['endpoint_count_after']}")
            for w in comp.get("call_layer", {}).get("weight_changed", []):
                hl.append(
                    f"Edge {w['source']}->{w['target']} weight "
                    f"{w['weight_delta_pct']}%")
            nd = comp.get("graph_layer", {}).get("node_diff", {})
            if nd.get("added"):
                hl.append(f"New nodes: {nd['added']}")
            if nd.get("removed"):
                hl.append(f"Removed nodes: {nd['removed']}")
            ed = comp.get("graph_layer", {}).get("edge_diff", {})
            if ed.get("added"):
                hl.append(f"New edges: {ed['added']}")
            if ed.get("removed"):
                hl.append(f"Removed edges: {ed['removed']}")
            phases[ck] = {"changed": None, "highlights": hl}

        # Mark primary comparison
        if "pre_vs_during" in phases:
            phases["pre_vs_during"]["changed"] = \
                ann.get("verification_result") != "NOT_CHANGED"

        # Summary text
        parts: list[str] = []
        if ann.get("detected_types"):
            parts.append(f"Detected: {', '.join(ann['detected_types'])}")
        if ann.get("evidence"):
            parts.append("Evidence: " + "; ".join(ann["evidence"][:3]))

        # Recovery check
        pre_post = results.get("comparisons", {}).get("pre_vs_post", {})
        recovery_ok = True
        if not pre_post.get("skipped"):
            pp_pod = pre_post.get("k8s_layer", {}).get("pod_changes", {})
            if pp_pod.get("count_changes") or pp_pod.get("rescheduled"):
                recovery_ok = False

        return {
            "experiment_id": results.get("experiment_id"),
            "verification_result": ann.get("verification_result", "UNKNOWN"),
            "change_score": ann.get("change_score", 0),
            "topology_change_type": ann.get("topology_change_type", "unknown"),
            "summary": (" | ".join(parts) if parts
                        else "No significant topology changes detected"),
            "confidence": ann.get("confidence", 0),
            "phases": phases,
            "ground_truth_match": (ann.get("ground_truth_comparison", {})
                                   .get("match")),
            "recovery_verified": recovery_ok,
            "recommendation": self._recommend(ann, recovery_ok),
        }

    @staticmethod
    def _recommend(ann: dict, recovery_ok: bool) -> str:
        v = ann.get("verification_result", "")
        if v == "NOT_CHANGED":
            return ("Topology did not change. Consider increasing fault "
                    "intensity or checking if the fault was properly injected.")
        if v == "CHANGED" and recovery_ok:
            return "Topology change verified and recovered. Data is suitable for dataset."
        if v == "CHANGED" and not recovery_ok:
            return ("Topology change verified but recovery may be incomplete. "
                "Check post-experiment state.")
        if v == "PARTIALLY_CHANGED":
            return ("Minor topology changes detected. May need higher fault "
                    "intensity for cleaner signal.")
        return "Review experiment parameters and cluster state."


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _make_serializable(obj):
    """Convert sets to sorted lists for JSON serialization."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, set):
                obj[k] = sorted(str(x) for x in v)
            else:
                _make_serializable(v)
    elif isinstance(obj, list):
        for item in obj:
            _make_serializable(item)


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Verify topology changes in experiment data.")
    parser.add_argument("--experiment-dir", required=True,
                        help="Path to experiment directory")
    parser.add_argument("--pretty", action="store_true",
                        help="Print verification report to stdout")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ])

    verifier = TopologyVerifier(args.experiment_dir)
    verifier.verify()

    if args.pretty:
        report_path = Path(args.experiment_dir) / "verification_report.json"
        if report_path.exists():
            print("\n" + "=" * 60)
            print("  VERIFICATION REPORT")
            print("=" * 60)
            with open(report_path) as f:
                report = json.load(f)
            print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
