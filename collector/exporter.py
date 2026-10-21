#!/usr/bin/env python3
"""
ExperimentExporter — main orchestrator.

Usage:
    python -m collector.exporter --experiment-dir /path/to/exp_DIR
    python -m collector.exporter --experiment-dir /path/to/exp_DIR --prom-url http://...

Reads metadata.json from the experiment directory, queries all
observability backends for the experiment time window, and writes
structured data to the experiment directory.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

from collector.config import PROM_URL, LOKI_URL, TEMPO_URL, NAMESPACE, get_services_for_namespace
from collector.prometheus_collector import PrometheusCollector
from collector.loki_collector import LokiCollector
from collector.tempo_collector import TempoCollector
from collector.k8s_snapshot import K8sSnapshot

# ── Known edges per application ────────────────────────────────
_BOUTIQUE_EDGES = {
    "frontend": ["productcatalogservice", "cartservice",
                  "recommendationservice", "currencyservice",
                  "adservice", "checkoutservice"],
    "recommendationservice": ["productcatalogservice"],
    "checkoutservice": ["cartservice", "productcatalogservice",
                        "shippingservice", "currencyservice",
                        "paymentservice", "emailservice"],
    "cartservice": [],
}

_TRAINTICKET_EDGES = {
    "ts-basic-service": ["ts-station-service", "ts-train-service", "ts-route-service", "ts-price-service"],
    "ts-ticketinfo-service": ["ts-basic-service"],
    "ts-travel-service": ["ts-ticketinfo-service", "ts-order-service", "ts-train-service", "ts-route-service", "ts-seat-service"],
    "ts-travel2-service": ["ts-ticketinfo-service", "ts-order-other-service", "ts-train-service", "ts-route-service", "ts-seat-service"],
    "ts-preserve-service": ["ts-ticketinfo-service", "ts-seat-service", "ts-notification-service", "ts-user-service", "ts-assurance-service", "ts-station-service", "ts-security-service", "ts-travel-service", "ts-contacts-service", "ts-order-service", "ts-food-service", "ts-consign-service"],
    "ts-preserve-other-service": ["ts-ticketinfo-service", "ts-seat-service", "ts-notification-service", "ts-user-service", "ts-assurance-service", "ts-station-service", "ts-security-service", "ts-travel2-service", "ts-contacts-service", "ts-order-other-service", "ts-food-service", "ts-consign-service"],
    "ts-order-service": ["ts-station-service"],
    "ts-order-other-service": ["ts-station-service"],
    "ts-seat-service": ["ts-travel-service", "ts-order-service", "ts-train-service", "ts-config-service"],
    "ts-security-service": ["ts-order-service", "ts-order-other-service"],
    "ts-food-service": ["ts-food-map-service", "ts-travel-service", "ts-station-service"],
    "ts-consign-service": ["ts-consign-price-service"],
    "ts-cancel-service": ["ts-notification-service", "ts-order-service", "ts-order-other-service", "ts-inside-payment-service", "ts-user-service"],
    "ts-inside-payment-service": ["ts-order-service", "ts-order-other-service", "ts-payment-service"],
    "ts-rebook-service": ["ts-ticketinfo-service", "ts-seat-service", "ts-notification-service", "ts-user-service", "ts-assurance-service", "ts-station-service", "ts-security-service", "ts-travel-service", "ts-travel2-service", "ts-contacts-service", "ts-order-service", "ts-order-other-service", "ts-food-service", "ts-consign-service"],
    "ts-route-plan-service": ["ts-travel-service", "ts-station-service", "ts-route-service"],
    "ts-auth-service": ["ts-verification-code-service"],
    "ts-user-service": ["ts-auth-service"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exporter")


class ExperimentExporter:
    """Orchestrates data export for a single experiment."""

    def __init__(self, experiment_dir: str | Path,
                 prom_url: str = PROM_URL,
                 loki_url: str = LOKI_URL,
                 tempo_url: str = TEMPO_URL,
                 namespace: str = NAMESPACE):
        self.exp_dir = Path(experiment_dir)
        if not self.exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {self.exp_dir}")

        self.namespace = namespace
        self.prom = PrometheusCollector(prom_url)
        self.loki = LokiCollector(loki_url)
        self.tempo = TempoCollector(tempo_url)
        self.k8s = K8sSnapshot(namespace)

        self.metadata = self._load_metadata()
        self.tw = self.metadata["time_window"]

        # Auto-detect namespace from metadata if available
        meta_ns = self.metadata.get("environment", {}).get("namespace", "")
        if meta_ns:
            self.namespace = meta_ns

        # Re-import collector config with correct namespace so PromQL queries use it
        import collector.config as cfg
        if self.namespace != cfg.NAMESPACE:
            cfg.NAMESPACE = self.namespace

    # ── load experiment metadata ─────────────────────────────────
    def _load_metadata(self) -> dict:
        meta_path = self.exp_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.exp_dir}")
        with open(meta_path) as f:
            return json.load(f)

    # ── connectivity check ───────────────────────────────────────
    def _check_apis(self) -> list[str]:
        """Returns list of API names that are reachable."""
        ok = []
        try:
            import requests
            r = requests.get(f"{self.prom.base_url}/api/v1/query",
                             params={"query": "up"}, timeout=5)
            if r.ok:
                ok.append("Prometheus")
        except Exception:
            pass
        try:
            import requests
            r = requests.get(f"{self.loki.base_url}/loki/api/v1/labels", timeout=5)
            if r.ok:
                ok.append("Loki")
        except Exception:
            pass
        if self.tempo.check_health():
            ok.append("Tempo")
        return ok

    # ── topology graph builder ───────────────────────────────────
    def _build_topology_graph(self, prefix: str = "") -> dict | None:
        """
        Build a node+edge graph from the K8s snapshot files already
        present in the experiment dir.
        """
        pods_file = self.exp_dir / "k8s_events" / f"{prefix}pods.json"
        eps_file = self.exp_dir / "k8s_events" / f"{prefix}endpoints.json"
        deps_file = self.exp_dir / "k8s_events" / f"{prefix}deployments.json"
        if not pods_file.exists():
            return None

        try:
            with open(pods_file) as f:
                pods = json.load(f)
            with open(eps_file) as f:
                eps = json.load(f)
            with open(deps_file) as f:
                deps = json.load(f)
        except Exception:
            return None

        # Build nodes
        nodes = {}
        for d in deps:
            name = d["name"]
            pods_for_svc = [p for p in pods if p.get("labels", {}).get("app") == name]
            nodes[name] = {
                "type": "service",
                "version": d.get("template_labels", {}).get("version", "unknown"),
                "replicas": d.get("replicas", 0),
                "readyReplicas": d.get("readyReplicas", 0),
                "image": d.get("image", ""),
                "pods": [
                    {"name": p["name"], "node": p["node"], "ip": p["ip"],
                     "phase": p["phase"]}
                    for p in pods_for_svc
                ],
            }

        # Build edges from endpoints (service → pod mapping implies who calls whom)
        # We use the well-known call graph to reconstruct edges
        edges = self._infer_edges(nodes, eps, self.namespace)

        return {
            "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "phase": prefix.rstrip("_") or "unknown",
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def _infer_edges(nodes: dict, endpoints: list[dict],
                     namespace: str = "boutique") -> list[dict]:
        """Infer service-to-service edges from known topology."""
        edge_map = _TRAINTICKET_EDGES if namespace == "trainticket" else _BOUTIQUE_EDGES
        edges = []
        for src, dsts in edge_map.items():
            for dst in dsts:
                if src in nodes and dst in nodes:
                    edges.append({
                        "source": src,
                        "target": dst,
                        "source_replicas": nodes[src].get("replicas", 0),
                        "target_replicas": nodes[dst].get("replicas", 0),
                    })
        return edges

    # ── main export pipeline ─────────────────────────────────────
    def export_all(self):
        """Run the full export pipeline."""
        exp_id = self.metadata.get("experiment_id", "unknown")
        start = self.tw["t_collect"]
        end = self.tw["t_end"]

        log.info("=" * 60)
        log.info("  ExperimentExporter")
        log.info("  ID  : %s", exp_id)
        log.info("  Dir : %s", self.exp_dir)
        log.info("  Window: %s → %s", start, end)
        log.info("=" * 60)

        # ── Step 0: Connectivity ─────────────────────────────────
        log.info("[Step 0] Checking API connectivity ...")
        apis = self._check_apis()
        for name in apis:
            log.info("  ✓ %s reachable", name)
        missing = {"Prometheus", "Loki", "Tempo"} - set(apis)
        if missing:
            log.warning("  ✗ %s NOT reachable. Run scripts/start-port-forward.sh",
                        ", ".join(missing))

        # ── Step 1: K8s snapshots ────────────────────────────────
        log.info("[Step 1] K8s object snapshots ...")
        k8s_dir = self.exp_dir / "k8s_events"
        # Take a current snapshot (post-experiment)
        self.k8s.export_all(k8s_dir, prefix="post_")

        # ── Step 2: Prometheus metrics ───────────────────────────
        log.info("[Step 2] Exporting Prometheus metrics ...")
        metrics_dir = self.exp_dir / "metrics"
        try:
            stats = self.prom.export_all(start, end, metrics_dir, namespace=self.namespace)
            log.info("  → %s", json.dumps(stats))
        except Exception as exc:
            log.error("  Prometheus export failed: %s", exc)

        # ── Step 3: Loki logs ────────────────────────────────────
        log.info("[Step 3] Exporting Loki logs ...")
        logs_dir = self.exp_dir / "logs"
        try:
            stats = self.loki.export_logs(start, end, logs_dir, namespace=self.namespace)
            log.info("  → %s", json.dumps(stats))
        except Exception as exc:
            log.error("  Loki export failed: %s", exc)

        # ── Step 4: Tempo traces ─────────────────────────────────
        log.info("[Step 4] Exporting Tempo traces ...")
        traces_dir = self.exp_dir / "traces"
        try:
            services = get_services_for_namespace(self.namespace)
            stats = self.tempo.export_traces(start, end, traces_dir, services=services)
            log.info("  → %s", json.dumps(stats))
        except Exception as exc:
            log.error("  Tempo export failed: %s", exc)

        # ── Step 5: Build topology graphs from snapshots ─────────
        log.info("[Step 5] Building topology graphs ...")
        for prefix, label in [("", "pre"), ("", "during")]:
            topo_file = self.exp_dir / "topology" / f"{label}.json"
            if topo_file.exists():
                graph = self._build_topology_graph(prefix=f"{label}_")
                if graph:
                    out = self.exp_dir / "topology" / f"{label}_graph.json"
                    with open(out, "w") as f:
                        json.dump(graph, f, indent=2)
                    log.info("  topology/%s_graph.json  nodes=%d  edges=%d",
                             label, len(graph["nodes"]), len(graph["edges"]))

        # Build from post_ snapshot
        graph = self._build_topology_graph(prefix="post_")
        if graph:
            out = self.exp_dir / "topology" / "post_graph.json"
            with open(out, "w") as f:
                json.dump(graph, f, indent=2)

        # ── Step 6: Validate outputs ─────────────────────────────
        log.info("[Step 6] Validating outputs ...")
        self._validate()

        log.info("=" * 60)
        log.info("  Export complete: %s", self.exp_dir)
        log.info("=" * 60)

    # ── validation ───────────────────────────────────────────────
    def _validate(self):
        """Check that expected output files exist."""
        required = [
            "metadata.json",
            "ground_truth.json",
        ]
        optional = [
            "metrics/all_metrics.parquet",
            "metrics/request_rate.csv",
            "logs/all.jsonl",
            "traces/traces_raw.json",
            "k8s_events/post_pods.json",
        ]
        for f in required:
            p = self.exp_dir / f
            status = "✓" if p.exists() else "✗ MISSING"
            log.info("  %s %s", status, f)
        for f in optional:
            p = self.exp_dir / f
            if p.exists():
                size = p.stat().st_size
                log.info("  ✓ %s  (%d bytes)", f, size)
            else:
                log.info("  - %s  (not found, non-critical)", f)

    # ── CLI ──────────────────────────────────────────────────────
    @staticmethod
    def main():
        parser = argparse.ArgumentParser(
            description="Export experiment data from observability backends.")
        parser.add_argument("--experiment-dir", required=True,
                            help="Path to the experiment directory (contains metadata.json)")
        parser.add_argument("--prom-url", default=PROM_URL,
                            help="Prometheus API URL")
        parser.add_argument("--loki-url", default=LOKI_URL,
                            help="Loki API URL")
        parser.add_argument("--tempo-url", default=TEMPO_URL,
                            help="Tempo API URL")
        parser.add_argument("--namespace", default=NAMESPACE,
                            help="Kubernetes namespace")
        args = parser.parse_args()

        exporter = ExperimentExporter(
            experiment_dir=args.experiment_dir,
            prom_url=args.prom_url,
            loki_url=args.loki_url,
            tempo_url=args.tempo_url,
            namespace=args.namespace,
        )
        exporter.export_all()


if __name__ == "__main__":
    ExperimentExporter.main()
