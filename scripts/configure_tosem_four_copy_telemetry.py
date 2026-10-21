#!/usr/bin/env python3
"""Enable the same Envoy-to-Tempo tracing policy used by formal collection."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml


def kubectl(env, *args, input_text=None):
    return subprocess.run(
        ["kubectl", *args], input=input_text, text=True, check=True,
        capture_output=True, env=env,
    ).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeconfig", type=Path, required=True)
    args = parser.parse_args()
    env = os.environ.copy()
    env["KUBECONFIG"] = str(args.kubeconfig.resolve())
    context = kubectl(env, "config", "current-context").strip()
    if context != "kind-tosem-copies":
        raise RuntimeError(f"Refusing to change unexpected context {context!r}")

    obj = json.loads(kubectl(env, "get", "configmap", "istio", "-n", "istio-system", "-o", "json"))
    mesh = yaml.safe_load(obj["data"]["mesh"])
    providers = [p for p in mesh.get("extensionProviders", []) if p.get("name") != "otel-tracing"]
    providers.append({
        "name": "otel-tracing",
        "opentelemetry": {
            "service": "tempo.observability.svc.cluster.local",
            "port": 4317,
        },
    })
    mesh["extensionProviders"] = providers
    patch = {"data": {"mesh": yaml.safe_dump(mesh, default_flow_style=False)}}
    kubectl(env, "patch", "configmap", "istio", "-n", "istio-system",
            "--type=merge", "-p", json.dumps(patch))
    telemetry = {
        "apiVersion": "telemetry.istio.io/v1",
        "kind": "Telemetry",
        "metadata": {"name": "default-tracing", "namespace": "istio-system"},
        "spec": {"tracing": [{
            "providers": [{"name": "otel-tracing"}],
            "randomSamplingPercentage": 100,
        }]},
    }
    kubectl(env, "apply", "-f", "-", input_text=yaml.safe_dump(telemetry))
    print("Istio Envoy tracing to Tempo is configured for kind-tosem-copies")


if __name__ == "__main__":
    main()
