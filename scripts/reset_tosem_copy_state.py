#!/usr/bin/env python3
"""Recover interrupted fault objects in standby copies; preserve exported data."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from collector.config import BOUTIQUE_SERVICES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeconfig", type=Path, default=PROJECT / "artifacts/tosem_four_copies/kubeconfig")
    parser.add_argument("--namespace", choices=[f"{s}-c{i}" for s in ("boutique", "trainticket") for i in range(1, 5)])
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "artifacts/tosem_four_copies/recovery")
    args = parser.parse_args()
    env = os.environ.copy()
    env["KUBECONFIG"] = str(args.kubeconfig.resolve())

    def kube(*argv, input_text=None):
        result = subprocess.run(["kubectl", "--request-timeout=20s", *argv], env=env,
                                input=input_text, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"{argv[:4]}: {result.stderr[-1200:]}")
        return result.stdout

    if kube("config", "current-context").strip() != "kind-tosem-copies":
        raise RuntimeError("Recovery is restricted to the dedicated cluster")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    namespaces = [args.namespace] if args.namespace else [f"{s}-c{i}" for s in ("boutique", "trainticket") for i in range(1, 5)]
    tt_docs = []
    for part in ("part1", "part2", "part3"):
        tt_docs += [d for d in yaml.safe_load_all((PROJECT / f"infra/trainticket/.manifests/ts-deployment-{part}.yml").read_text()) if d]
    for namespace in namespaces:
        system = namespace.rsplit("-c", 1)[0]
        deployments = json.loads(kube("get", "deployments", "-n", namespace, "-o", "json"))["items"]
        services = json.loads(kube("get", "services", "-n", namespace, "-o", "json"))["items"]
        if any(d["spec"].get("replicas", 0) for d in deployments):
            raise RuntimeError(f"{namespace} must be in standby before recovery")
        backup = args.out_dir / f"{namespace}-before.json"
        if not backup.exists():
            backup.write_text(json.dumps({"deployments": deployments, "services": services}, indent=2) + "\n")
        if system == "boutique":
            allowed = {s.rsplit(".", 1)[0] for s in BOUTIQUE_SERVICES} | {"redis-cart", "loadgenerator", "locust-master"}
            selectors = {}
            for service in services:
                raw = service["metadata"].get("annotations", {}).get("kubectl.kubernetes.io/last-applied-configuration")
                if raw:
                    original = json.loads(raw)
                    selector = original.get("spec", {}).get("selector")
                    if selector is not None:
                        selectors[service["metadata"]["name"]] = selector
        else:
            allowed = {d["metadata"]["name"] for d in tt_docs if d["kind"] == "Deployment"} | {"locust-trainticket"}
            selectors = {d["metadata"]["name"]: d["spec"]["selector"]
                         for d in tt_docs if d["kind"] == "Service" and "selector" in d["spec"]}
        removed = []
        for deployment in deployments:
            name = deployment["metadata"]["name"]
            if name not in allowed:
                if "-cf-" not in name and not name.endswith("-v2"):
                    raise RuntimeError(f"Unexpected deployment {namespace}/{name}; recovery refused")
                kube("delete", "deployment", name, "-n", namespace, "--wait=false")
                removed.append(name)
            elif deployment["spec"]["template"].get("metadata", {}).get("labels", {}).get("version") != "v1":
                kube("patch", "deployment", name, "-n", namespace, "--type=merge", "-p",
                     json.dumps({"spec": {"template": {"metadata": {"labels": {"version": "v1"}}}}}))
        for service in services:
            name = service["metadata"]["name"]
            if name in selectors and service["spec"].get("selector") != selectors[name]:
                kube("patch", "service", name, "-n", namespace, "--type=json", "-p",
                     json.dumps([{"op": "add", "path": "/spec/selector", "value": selectors[name]}]))
        kube("delete", "stresschaos,podchaos,networkchaos,virtualservice,destinationrule", "--all",
             "-n", namespace, "--wait=false", "--ignore-not-found")
        config = yaml.safe_load((PROJECT / f"config/tosem_candidate_{system}_20260930.yaml").read_text())
        targets = next(s["targets"] for s in config["scenarios"] if s["id"] == "T1")
        existing = json.loads(kube("get", "hpa", "-n", namespace, "-o", "json"))["items"]
        hpa_names = {h["spec"]["scaleTargetRef"]["name"]: h["metadata"]["name"] for h in existing}
        for target in targets:
            hpa = {"apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler",
                   "metadata": {"name": hpa_names.get(target, target + "-hpa"), "namespace": namespace},
                   "spec": {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": target},
                            "minReplicas": 1, "maxReplicas": 3,
                            "behavior": {"scaleDown": {"stabilizationWindowSeconds": 60},
                                         "scaleUp": {"stabilizationWindowSeconds": 0}},
                            "metrics": [{"type": "Resource", "resource": {"name": "cpu", "target": {
                                "type": "Utilization", "averageUtilization": 50}}}]}}
            kube("apply", "-f", "-", input_text=yaml.safe_dump(hpa))
        print(f"{namespace}: selectors restored, residual deployments removed={removed}, HPA cap=3", flush=True)


if __name__ == "__main__":
    main()
