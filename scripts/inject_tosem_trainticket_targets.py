#!/usr/bin/env python3
"""Instrument Train Ticket application services while keeping databases sidecar-free."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(env, command, input_text=None):
    result = subprocess.run(
        command, input=input_text, text=True,
        capture_output=True, env=env,
    )
    if result.returncode:
        raise RuntimeError(f"{command[:5]} failed: {result.stderr[-1500:]}")
    return result.stdout


def target_services(config_path):
    config = yaml.safe_load(config_path.read_text())
    from collector.config import TRAINTICKET_SERVICES
    services = {service.rsplit(".", 1)[0] for service in TRAINTICKET_SERVICES}
    services.add("ts-ui-dashboard")
    services.update(target for scenario in config["scenarios"]
                    for target in scenario.get("targets", []))
    return sorted(services)


def inject_one(env, namespace, service):
    deployment = json.loads(run(env, [
        "kubectl", "get", "deployment", service, "-n", namespace, "-o", "json",
    ]))
    containers = deployment["spec"]["template"]["spec"]["containers"]
    cpu_changed = False
    if service != "ts-ui-dashboard":
        for container in containers:
            if container["name"] == "istio-proxy":
                continue
            resources = container.setdefault("resources", {})
            limit = resources.get("limits", {}).get("cpu")
            if not limit:
                raise RuntimeError(f"Missing fixed CPU limit for {namespace}/{service}")
            requests = resources.setdefault("requests", {})
            if requests.get("cpu") != limit:
                requests["cpu"] = limit
                cpu_changed = True
    if any(item["name"] == "istio-proxy" for item in containers):
        if cpu_changed:
            run(env, ["kubectl", "patch", "deployment", service, "-n", namespace,
                      "--type=merge", "-p", json.dumps({"spec": {"template": deployment["spec"]["template"]}})])
        return cpu_changed
    injected_docs = list(yaml.safe_load_all(run(env, [
        "istioctl", "kube-inject", "--kubeconfig", env["KUBECONFIG"], "-f", "-",
    ], input_text=yaml.safe_dump(deployment))))
    injected = next(doc for doc in injected_docs if isinstance(doc, dict) and
                    doc.get("kind") == "Deployment" and
                    doc.get("metadata", {}).get("name") == service)
    template = injected["spec"]["template"]
    if not any(item["name"] == "istio-proxy" for item in template["spec"]["containers"]):
        raise RuntimeError(f"istioctl did not inject {namespace}/{service}")
    patch = {"spec": {"template": template}}
    run(env, ["kubectl", "patch", "deployment", service, "-n", namespace,
              "--type=merge", "-p", json.dumps(patch)])
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument("--namespace", choices=[f"trainticket-c{i}" for i in range(1, 5)], required=True)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "config/tosem_candidate_trainticket_20260930.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    env = os.environ.copy()
    env["KUBECONFIG"] = str(args.kubeconfig.resolve())
    context = run(env, ["kubectl", "config", "current-context"]).strip()
    if context != "kind-tosem-copies":
        raise RuntimeError(f"Refusing to change unexpected context {context!r}")
    targets = target_services(args.config)
    if args.limit is not None:
        targets = targets[:args.limit]
    for service in targets:
        changed = inject_one(env, args.namespace, service)
        print(f"{args.namespace}/{service}: {'updated' if changed else 'already prepared'}", flush=True)


if __name__ == "__main__":
    main()
