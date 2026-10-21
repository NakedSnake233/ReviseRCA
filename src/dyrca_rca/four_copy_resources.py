"""Bounded canary deployments that inherit the baseline resource settings."""
import argparse
import copy
import json
import os
import subprocess


def canary_deployment(source, namespace, target, version="v2.0.0"):
    if namespace not in {f"{s}-c{i}" for s in ("boutique", "trainticket") for i in range(1, 5)}:
        raise ValueError("Only dedicated four-copy namespaces are supported")
    template = copy.deepcopy(source["spec"]["template"])
    template["metadata"].pop("creationTimestamp", None)
    template["metadata"].setdefault("labels", {}).update(app=target, version=version)
    return {"apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": {"name": target + "-v2", "namespace": namespace,
                         "labels": {"app": target, "version": version}},
            "spec": {"replicas": 1,
                     "selector": {"matchLabels": {"app": target, "version": version}},
                     "template": template}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    context = subprocess.check_output(["kubectl", "config", "current-context"], text=True).strip()
    if context != "kind-tosem-copies":
        raise RuntimeError("Only kind-tosem-copies is supported")
    source = json.loads(subprocess.check_output(
        ["kubectl", "get", "deployment", args.target, "-n", args.namespace, "-o", "json"],
        text=True, timeout=20))
    print(json.dumps(canary_deployment(source, args.namespace, args.target), sort_keys=True))


if __name__ == "__main__":
    main()
