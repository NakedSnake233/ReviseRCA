import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dyrca_rca.dataset.materialize import (
    build_graph_from_topology,
    graph_to_model_tensors,
    main,
    materialize_index,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _pod(name, service, ip, version="unknown"):
    return {
        "name": name,
        "service": service,
        "ip": ip,
        "version": version,
        "node": "node-a",
        "status": "Running",
    }


def _svc(name, ips):
    return {"name": name, "endpoint_ips": ips}


def _dep(name, replicas, version="unknown"):
    return {"name": name, "replicas": replicas, "readyReplicas": replicas, "version": version}


def _snapshot(pods, services, deployments):
    return {
        "timestamp": "2026-01-01T00:01:30Z",
        "pods": pods,
        "services": services,
        "deployments": deployments,
    }


def _metadata(exp_id, scenario, name, target, change_type, params=None):
    return {
        "experiment_id": exp_id,
        "scenario_id": scenario,
        "scenario_name": name,
        "target_service": target,
        "topology_change_type": change_type,
        "fault_params": params or {},
        "time_window": {
            "t_collect": "2026-01-01T00:00:00Z",
            "t_inject": "2026-01-01T00:01:00Z",
            "t_recover": "2026-01-01T00:02:00Z",
            "t_end": "2026-01-01T00:03:00Z",
        },
    }


def _make_t1_exp(tmpdir):
    exp_dir = Path(str(tmpdir)) / "t1-cart"
    pre = _snapshot(
        [_pod("cart-1", "cartservice", "10.0.0.1"), _pod("frontend-1", "frontend", "10.0.0.9")],
        [_svc("cartservice", ["10.0.0.1"]), _svc("frontend", ["10.0.0.9"])],
        [_dep("cartservice", 1), _dep("frontend", 1)],
    )
    during = _snapshot(
        [
            _pod("cart-1", "cartservice", "10.0.0.1"),
            _pod("cart-2", "cartservice", "10.0.0.2"),
            _pod("frontend-1", "frontend", "10.0.0.9"),
        ],
        [_svc("cartservice", ["10.0.0.1", "10.0.0.2"]), _svc("frontend", ["10.0.0.9"])],
        [_dep("cartservice", 2), _dep("frontend", 1)],
    )
    meta = _metadata("t1-cart", "T1", "hpa_scale", "cartservice", "instance_count_change")
    _write_json(exp_dir / "metadata.json", meta)
    _write_json(exp_dir / "ground_truth.json", {
        "experiment_id": "t1-cart",
        "root_cause_service": "cartservice",
        "topology_change_type": "instance_count_change",
    })
    _write_json(exp_dir / "topology" / "pre.json", pre)
    _write_json(exp_dir / "topology" / "during.json", during)
    _write_json(exp_dir / "topology" / "post.json", pre)
    _write_json(exp_dir / "topology_diff.json", {
        "schema": "topology_diff_v1",
        "annotation": {"change_score": 0.5, "topology_change_type": "node_scale"},
        "comparisons": {"pre_vs_during": {"k8s_layer": {}, "graph_layer": {}}},
    })
    return exp_dir


def _make_t3_exp(tmpdir):
    exp_dir = Path(str(tmpdir)) / "t3-empty"
    pre = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.1.1"), _pod("frontend-1", "frontend", "10.0.1.9")],
        [_svc("checkoutservice", ["10.0.1.1"]), _svc("frontend", ["10.0.1.9"])],
        [_dep("checkoutservice", 1), _dep("frontend", 1)],
    )
    during = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.1.1"), _pod("frontend-1", "frontend", "10.0.1.9")],
        [_svc("checkoutservice", []), _svc("frontend", ["10.0.1.9"])],
        [_dep("checkoutservice", 1), _dep("frontend", 1)],
    )
    meta = _metadata("t3-empty", "T3", "selector_poison", "checkoutservice", "edge_removal")
    _write_json(exp_dir / "metadata.json", meta)
    _write_json(exp_dir / "ground_truth.json", {
        "experiment_id": "t3-empty",
        "root_cause_service": "checkoutservice",
        "topology_change_type": "edge_removal",
    })
    _write_json(exp_dir / "topology" / "pre.json", pre)
    _write_json(exp_dir / "topology" / "during.json", during)
    _write_json(exp_dir / "topology" / "post.json", pre)
    _write_json(exp_dir / "topology_diff.json", {
        "schema": "topology_diff_v1",
        "annotation": {"change_score": 0.5, "topology_change_type": "edge_removal"},
        "comparisons": {"pre_vs_during": {"k8s_layer": {}, "graph_layer": {}}},
    })
    return exp_dir


def _write_index(path, rows):
    frame = pd.DataFrame(rows)
    # Deliberately CSV content with a parquet extension, so tests do not need pyarrow.
    frame.to_csv(path, index=False)


def _load_pt(path):
    try:
        import torch
        return torch.load(str(path), weights_only=False)
    except TypeError:
        import torch
        return torch.load(str(path))


def test_build_graph_from_topology_has_service_pod_and_dependency_edges(tmpdir):
    exp_dir = _make_t1_exp(tmpdir)
    pre = json.loads((exp_dir / "topology" / "pre.json").read_text(encoding="utf-8"))
    graph = build_graph_from_topology(pre, phase="pre")

    node_ids = {node["id"] for node in graph["nodes"]}
    edge_pairs = {(edge["source"], edge["target"], edge["edge_type"]) for edge in graph["edges"]}
    assert "cartservice" in node_ids
    assert "pod:cart-1" in node_ids
    assert ("cartservice", "pod:cart-1", "endpoint") in edge_pairs
    assert ("frontend", "cartservice", "dependency") in edge_pairs


def test_graph_to_model_tensors_contains_label_and_shapes(tmpdir):
    exp_dir = _make_t1_exp(tmpdir)
    pre = json.loads((exp_dir / "topology" / "pre.json").read_text(encoding="utf-8"))
    graph = build_graph_from_topology(pre, phase="pre")
    data = graph_to_model_tensors(graph, {
        "experiment_id": "t1-cart",
        "scenario_id": "T1",
        "target_service": "cartservice",
        "root_cause_service": "cartservice",
    })

    assert "cartservice" in data["node_ids"]
    assert int(data["label"]) == data["node_ids"].index("cartservice")
    assert data["edge_index"].shape[0] == 2
    assert data["edge_attr"].shape[1] == 5
    assert data["metadata"]["experiment_id"] == "t1-cart"
    assert "incident_update_count" in data["metadata"]["node_feature_names"]
    assert data["x"].shape[1] == len(data["metadata"]["node_feature_names"])


def test_materialize_rule_full_outputs_pt_artifact(tmpdir):
    exp_dir = _make_t1_exp(tmpdir)
    index_path = Path(str(tmpdir)) / "index.parquet"
    out_dir = Path(str(tmpdir)) / "model_data"
    _write_index(index_path, [{
        "experiment_id": "t1-cart",
        "scenario_id": "T1",
        "target_service": "cartservice",
        "root_cause_service": "cartservice",
        "data_path": str(exp_dir),
        "error_message": "",
    }])

    outputs = materialize_index(str(index_path), "rule_full", str(out_dir), limit=5)

    assert len(outputs) == 1
    artifact = Path(outputs[0])
    assert artifact == out_dir / "rule_full" / "t1-cart.pt"
    data = _load_pt(artifact)
    assert data["metadata"]["graph_variant"] == "rule_full"
    assert "pod:cart-2" in data["node_ids"]
    assert data["edge_index"].shape[0] == 2
    assert int(data["label"]) == data["node_ids"].index("cartservice")


def test_static_and_naive_during_variants_differ_by_new_pod(tmpdir):
    exp_dir = _make_t1_exp(tmpdir)
    index_path = Path(str(tmpdir)) / "index.parquet"
    out_dir = Path(str(tmpdir)) / "model_data"
    _write_index(index_path, [{
        "experiment_id": "t1-cart",
        "scenario_id": "T1",
        "target_service": "cartservice",
        "root_cause_service": "cartservice",
        "data_path": str(exp_dir),
        "error_message": "",
    }])

    materialize_index(str(index_path), "static_pre", str(out_dir))
    materialize_index(str(index_path), "naive_during", str(out_dir))
    static = _load_pt(out_dir / "static_pre" / "t1-cart.pt")
    during = _load_pt(out_dir / "naive_during" / "t1-cart.pt")

    assert "pod:cart-2" not in static["node_ids"]
    assert "pod:cart-2" in during["node_ids"]


def test_llm_full_uses_cached_plan(tmpdir):
    exp_dir = _make_t3_exp(tmpdir)
    index_path = Path(str(tmpdir)) / "index.parquet"
    out_dir = Path(str(tmpdir)) / "model_data"
    llm_dir = Path(str(tmpdir)) / "llm_plans"
    _write_index(index_path, [{
        "experiment_id": "t3-empty",
        "scenario_id": "T3",
        "target_service": "checkoutservice",
        "root_cause_service": "checkoutservice",
        "data_path": str(exp_dir),
        "error_message": "",
    }])
    _write_json(llm_dir / "t3-empty.json", {
        "experiment_id": "t3-empty",
        "change_type": "edge_removal",
        "confidence": 0.9,
        "causal_role": "cause_of_fault",
        "node_ops": [{"op": "ADD_BLACKHOLE_NODE", "service": "checkoutservice"}],
        "edge_ops": [],
        "weight_ops": [],
        "state_transfer": {},
        "evidence": [],
    })

    outputs = materialize_index(str(index_path), "llm_full", str(out_dir), llm_plan_dir=str(llm_dir))
    data = _load_pt(outputs[0])

    assert "blackhole:checkoutservice" in data["node_ids"]
    assert data["graph"]["attrs"]["llm_plan_missing"] is False


def test_materialize_cli_limit(tmpdir):
    exp_dir = _make_t1_exp(tmpdir)
    index_path = Path(str(tmpdir)) / "index.parquet"
    out_dir = Path(str(tmpdir)) / "model_data"
    _write_index(index_path, [{
        "experiment_id": "t1-cart",
        "scenario_id": "T1",
        "target_service": "cartservice",
        "root_cause_service": "cartservice",
        "data_path": str(exp_dir),
        "error_message": "",
    }])

    assert main(["--index", str(index_path), "--variant", "rule_node", "--out-dir", str(out_dir), "--limit", "1"]) == 0
    assert (out_dir / "rule_node" / "t1-cart.pt").exists()
