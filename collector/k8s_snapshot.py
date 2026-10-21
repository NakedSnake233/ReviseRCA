"""
Kubernetes object snapshot.
Exports pods, services, endpoints, deployments, events as JSON.
"""
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from collector.config import NAMESPACE

log = logging.getLogger(__name__)


def _kubectl_json(args: list[str]) -> dict:
    """Run kubectl and return parsed JSON."""
    cmd = ["kubectl"] + args + ["-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        log.warning("  kubectl failed: %s", result.stderr.strip()[:200])
        return {"items": []}
    return json.loads(result.stdout)


def _slim_pods(raw: dict) -> list[dict]:
    """Extract only the fields we need from pod objects."""
    pods = []
    for p in raw.get("items", []):
        containers = []
        for c in p.get("spec", {}).get("containers", []):
            containers.append({
                "name": c.get("name"),
                "image": c.get("image"),
                "ports": [str(pp.get("containerPort", "")) for pp in c.get("ports", [])],
            })
        pods.append({
            "name":           p["metadata"]["name"],
            "namespace":      p["metadata"]["namespace"],
            "labels":         p["metadata"].get("labels", {}),
            "node":           p.get("spec", {}).get("nodeName", ""),
            "ip":             p.get("status", {}).get("podIP", ""),
            "phase":          p.get("status", {}).get("phase", ""),
            "startTime":      p.get("status", {}).get("startTime", ""),
            "containers":     containers,
            "restartCount":   sum(
                cs.get("restartCount", 0)
                for cs in p.get("status", {}).get("containerStatuses", [])
            ),
        })
    return pods


def _slim_services(raw: dict) -> list[dict]:
    svcs = []
    for s in raw.get("items", []):
        ports = []
        for p in s.get("spec", {}).get("ports", []):
            ports.append({"port": p.get("port"), "targetPort": p.get("targetPort"),
                          "protocol": p.get("protocol")})
        svcs.append({
            "name":      s["metadata"]["name"],
            "namespace": s["metadata"]["namespace"],
            "type":      s.get("spec", {}).get("type", ""),
            "selector":  s.get("spec", {}).get("selector", {}),
            "ports":     ports,
            "clusterIP": s.get("spec", {}).get("clusterIP", ""),
        })
    return svcs


def _slim_endpoints(raw: dict) -> list[dict]:
    eps = []
    for e in raw.get("items", []):
        addrs = []
        for sub in e.get("subsets", []):
            for a in sub.get("addresses", []):
                addrs.append({
                    "ip": a.get("ip"),
                    "podName": a.get("podName", ""),
                    "nodeName": a.get("nodeName", ""),
                })
        eps.append({
            "name": e["metadata"]["name"],
            "addresses": addrs,
        })
    return eps


def _slim_deployments(raw: dict) -> list[dict]:
    deps = []
    for d in raw.get("items", []):
        deps.append({
            "name":           d["metadata"]["name"],
            "namespace":      d["metadata"]["namespace"],
            "replicas":       d.get("spec", {}).get("replicas", 0),
            "readyReplicas":  d.get("status", {}).get("readyReplicas", 0),
            "updatedReplicas": d.get("status", {}).get("updatedReplicas", 0),
            "labels":         d["metadata"].get("labels", {}),
            "template_labels": d.get("spec", {}).get("template", {})
                                     .get("metadata", {}).get("labels", {}),
            "image":          d.get("spec", {}).get("template", {})
                                     .get("spec", {}).get("containers", [{}])[0].get("image", ""),
        })
    return deps


class K8sSnapshot:
    """Exports Kubernetes objects from the experiment namespace."""

    def __init__(self, namespace: str = NAMESPACE):
        self.ns = namespace

    def _save(self, data, path: Path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_all(self, output_dir: Path, prefix: str = "") -> dict[str, int]:
        """
        Export all K8s objects. Returns {resource: count}.
        prefix: "pre_" / "during_" / "post_" for phased snapshots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {}

        # Pods
        raw = _kubectl_json(["get", "pods", "-n", self.ns])
        slim = _slim_pods(raw)
        self._save(slim, output_dir / f"{prefix}pods.json")
        stats["pods"] = len(slim)

        # Services
        raw = _kubectl_json(["get", "services", "-n", self.ns])
        slim = _slim_services(raw)
        self._save(slim, output_dir / f"{prefix}services.json")
        stats["services"] = len(slim)

        # Endpoints
        raw = _kubectl_json(["get", "endpoints", "-n", self.ns])
        slim = _slim_endpoints(raw)
        self._save(slim, output_dir / f"{prefix}endpoints.json")
        stats["endpoints"] = len(slim)

        # Deployments
        raw = _kubectl_json(["get", "deployments", "-n", self.ns])
        slim = _slim_deployments(raw)
        self._save(slim, output_dir / f"{prefix}deployments.json")
        stats["deployments"] = len(slim)

        # Events (recent)
        raw = _kubectl_json(["get", "events", "-n", self.ns,
                             "--sort-by=.lastTimestamp"])
        events = []
        for ev in raw.get("items", []):
            events.append({
                "type":     ev.get("type", ""),
                "reason":   ev.get("reason", ""),
                "message":  ev.get("message", ""),
                "source":   ev.get("source", {}).get("component", ""),
                "involvedObject": {
                    "kind": ev.get("involvedObject", {}).get("kind", ""),
                    "name": ev.get("involvedObject", {}).get("name", ""),
                },
                "count":           ev.get("count", 0),
                "firstTimestamp":  ev.get("firstTimestamp", ""),
                "lastTimestamp":   ev.get("lastTimestamp", ""),
            })
        self._save(events, output_dir / f"{prefix}events.json")
        stats["events"] = len(events)

        # HPA (if any)
        raw = _kubectl_json(["get", "hpa", "-n", self.ns])
        hpas = []
        for h in raw.get("items", []):
            hpas.append({
                "name":            h["metadata"]["name"],
                "target":          h["spec"]["scaleTargetRef"].get("name", ""),
                "minReplicas":     h["spec"].get("minReplicas", 0),
                "maxReplicas":     h["spec"].get("maxReplicas", 0),
                "currentReplicas": h.get("status", {}).get("currentReplicas", 0),
                "desiredReplicas": h.get("status", {}).get("desiredReplicas", 0),
            })
        self._save(hpas, output_dir / f"{prefix}hpas.json")
        stats["hpas"] = len(hpas)

        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log.info("  [K8s] %s snapshot at %s → %s", prefix or "final", ts,
                 ", ".join(f"{k}={v}" for k, v in stats.items()))

        return stats
