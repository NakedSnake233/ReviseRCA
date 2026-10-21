import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.eval.graph_update_metrics import (
    aggregate_by_scenario,
    derive_expected_operations,
    evaluate_graph_update,
)
from dyrca_rca.experiments.evaluate_graph_updates import main


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _base_metadata(exp_id, scenario_id, scenario_name, target_service, change_type, fault_params=None):
    return {
        "experiment_id": exp_id,
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "target_service": target_service,
        "topology_change_type": change_type,
        "fault_params": fault_params or {},
    }


def _snapshot(pods, services, deployments=None):
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "pods": pods,
        "services": services,
        "deployments": deployments or [],
    }


def _pod(name, service, ip, version="unknown", node="node-a"):
    return {
        "name": name,
        "service": service,
        "ip": ip,
        "version": version,
        "node": node,
        "status": "Running",
    }


def _svc(name, ips):
    return {"name": name, "endpoint_ips": ips}


def _dep(name, replicas, version="unknown"):
    return {"name": name, "replicas": replicas, "readyReplicas": replicas, "version": version}


def _make_exp(tmpdir, exp_name, metadata, pre, during):
    exp_dir = Path(str(tmpdir)) / exp_name
    _write_json(exp_dir / "metadata.json", metadata)
    _write_json(exp_dir / "ground_truth.json", {
        "experiment_id": metadata["experiment_id"],
        "root_cause_service": metadata["target_service"],
    })
    _write_json(exp_dir / "topology" / "pre.json", pre)
    _write_json(exp_dir / "topology" / "during.json", during)
    _write_json(exp_dir / "topology" / "post.json", during)
    return exp_dir


def _row(exp_dir, metadata):
    return {
        "experiment_id": metadata["experiment_id"],
        "scenario_id": metadata["scenario_id"],
        "scenario_name": metadata["scenario_name"],
        "target_service": metadata["target_service"],
        "topology_change_type": metadata["topology_change_type"],
        "data_path": str(exp_dir),
        "error_message": "",
    }


def test_t2_alignment_accuracy_and_execution_success(tmpdir):
    metadata = _base_metadata("t2-checkout", "T2", "pod_drift", "checkoutservice", "instance_drift")
    pre = _snapshot(
        pods=[_pod("checkout-old", "checkoutservice", "10.0.1.1")],
        services=[_svc("checkoutservice", ["10.0.1.1"])],
        deployments=[_dep("checkoutservice", 1)],
    )
    during = _snapshot(
        pods=[_pod("checkout-new", "checkoutservice", "10.0.1.2")],
        services=[_svc("checkoutservice", ["10.0.1.2"])],
        deployments=[_dep("checkoutservice", 1)],
    )
    exp_dir = _make_exp(tmpdir, "t2-checkout", metadata, pre, during)
    plans_dir = Path(str(tmpdir)) / "plans"
    _write_json(plans_dir / "t2-checkout.json", {
        "experiment_id": "t2-checkout",
        "change_type": "node_drift",
        "confidence": 0.9,
        "causal_role": "topology_event",
        "node_ops": [{"op": "ALIGN_NODE", "old_node": "pod:checkout-old", "new_node": "pod:checkout-new"}],
        "edge_ops": [{
            "op": "REPLACE_ENDPOINT_EDGE",
            "old_source": "checkoutservice",
            "old_target": "pod:checkout-old",
            "new_target": "pod:checkout-new",
        }],
    })

    result = evaluate_graph_update(_row(exp_dir, metadata), str(plans_dir))

    assert result["operation_validity_rate"] == 1.0
    assert result["plan_execution_success_rate"] == 1.0
    assert result["node_alignment_accuracy"] == 1.0
    assert result["node_add_remove_f1"] is None


def test_invalid_operation_lowers_validity_and_execution_success(tmpdir):
    metadata = _base_metadata("t1-cart", "T1", "hpa_scale", "cartservice", "instance_count_change")
    pre = _snapshot(
        pods=[_pod("cart-1", "cartservice", "10.0.0.1")],
        services=[_svc("cartservice", ["10.0.0.1"])],
        deployments=[_dep("cartservice", 1)],
    )
    during = _snapshot(
        pods=[_pod("cart-1", "cartservice", "10.0.0.1"), _pod("cart-2", "cartservice", "10.0.0.2")],
        services=[_svc("cartservice", ["10.0.0.1", "10.0.0.2"])],
        deployments=[_dep("cartservice", 2)],
    )
    exp_dir = _make_exp(tmpdir, "t1-cart", metadata, pre, during)
    plans_dir = Path(str(tmpdir)) / "plans"
    _write_json(plans_dir / "t1-cart.json", {
        "experiment_id": "t1-cart",
        "change_type": "node_scale",
        "confidence": 0.9,
        "causal_role": "effect_or_mixed",
        "node_ops": [{"op": "ADD_NODE", "node_id": "pod:cart-2"}],
        "edge_ops": [{"op": "ADD_EDGE", "source": "cartservice"}],
    })

    result = evaluate_graph_update(_row(exp_dir, metadata), str(plans_dir))

    assert result["operation_validity_rate"] == 0.5
    assert result["plan_execution_success_rate"] == 0.0
    assert result["node_add_remove_f1"] == 1.0


def test_t3_redirect_blackhole_and_trace_mismatch_metrics(tmpdir):
    metadata = _base_metadata("t3-redirect", "T3", "selector_poison", "checkoutservice", "edge_redirect")
    pre = _snapshot(
        pods=[
            _pod("checkout-1", "checkoutservice", "10.0.2.1"),
            _pod("cart-1", "cartservice", "10.0.2.2"),
        ],
        services=[_svc("checkoutservice", ["10.0.2.1"])],
        deployments=[_dep("checkoutservice", 1), _dep("cartservice", 1)],
    )
    during = _snapshot(
        pods=[
            _pod("checkout-1", "checkoutservice", "10.0.2.1"),
            _pod("cart-1", "cartservice", "10.0.2.2"),
        ],
        services=[_svc("checkoutservice", ["10.0.2.2"])],
        deployments=[_dep("checkoutservice", 1), _dep("cartservice", 1)],
    )
    exp_dir = _make_exp(tmpdir, "t3-redirect", metadata, pre, during)
    trace_dir = exp_dir / "traces"
    trace_dir.mkdir()
    (trace_dir / "trace_summary.csv").write_text(
        "source,declared_dst,observed_dst\nfrontend,checkoutservice,cartservice\n",
        encoding="utf-8",
    )
    plans_dir = Path(str(tmpdir)) / "plans"
    _write_json(plans_dir / "t3-redirect.json", {
        "experiment_id": "t3-redirect",
        "change_type": "edge_redirect",
        "confidence": 0.9,
        "causal_role": "cause_of_fault",
        "edge_ops": [
            {
                "op": "REDIRECT_EDGE",
                "old_source": "checkoutservice",
                "old_target": "pod:checkout-1",
                "new_target": "pod:cart-1",
            },
            {
                "op": "ADD_TRACE_MISMATCH_EDGE",
                "source": "frontend",
                "target": "cartservice",
            },
        ],
    })

    result = evaluate_graph_update(_row(exp_dir, metadata), str(plans_dir))

    assert result["edge_redirect_recall"] == 1.0
    assert result["trace_mismatch_recall"] == 1.0

    empty_metadata = _base_metadata("t3-empty", "T3", "selector_poison", "checkoutservice", "edge_removal")
    empty_exp = _make_exp(
        tmpdir,
        "t3-empty",
        empty_metadata,
        pre,
        _snapshot(
            pods=[_pod("checkout-1", "checkoutservice", "10.0.2.1")],
            services=[_svc("checkoutservice", [])],
            deployments=[_dep("checkoutservice", 1)],
        ),
    )
    _write_json(plans_dir / "t3-empty.json", {
        "experiment_id": "t3-empty",
        "change_type": "edge_removal",
        "confidence": 0.9,
        "causal_role": "cause_of_fault",
        "node_ops": [{"op": "ADD_BLACKHOLE_NODE", "service": "checkoutservice"}],
    })
    empty_result = evaluate_graph_update(_row(empty_exp, empty_metadata), str(plans_dir))
    assert empty_result["blackhole_recall"] == 1.0


def test_t4_version_split_recall_and_route_weight_mae(tmpdir):
    metadata = _base_metadata(
        "t4-canary",
        "T4",
        "canary_release",
        "checkoutservice",
        "version_split",
        fault_params={"v1_weight": 80, "v2_weight": 20},
    )
    pre = _snapshot(
        pods=[_pod("checkout-1", "checkoutservice", "10.0.4.1", version="v1")],
        services=[_svc("checkoutservice", ["10.0.4.1"])],
        deployments=[_dep("checkoutservice", 1, version="v1")],
    )
    during = _snapshot(
        pods=[
            _pod("checkout-1", "checkoutservice", "10.0.4.1", version="v1"),
            _pod("checkout-v2-1", "checkoutservice-v2", "10.0.4.2", version="v2"),
        ],
        services=[_svc("checkoutservice", ["10.0.4.1", "10.0.4.2"])],
        deployments=[_dep("checkoutservice", 1, version="v1"), _dep("checkoutservice-v2", 1, version="v2")],
    )
    exp_dir = _make_exp(tmpdir, "t4-canary", metadata, pre, during)
    expected = derive_expected_operations(exp_dir, _row(exp_dir, metadata))
    assert expected["route_weights"] == {"v1": 0.8, "v2": 0.2}
    plans_dir = Path(str(tmpdir)) / "plans"
    _write_json(plans_dir / "t4-canary.json", {
        "experiment_id": "t4-canary",
        "change_type": "edge_split",
        "confidence": 0.9,
        "causal_role": "cause_or_mixed",
        "node_ops": [{
            "op": "SPLIT_NODE",
            "parent": "checkoutservice",
            "children": ["checkoutservice:v1", "checkoutservice:v2"],
        }],
        "weight_ops": [
            {"op": "REWEIGHT_EDGE", "source": "frontend", "target": "checkoutservice:v1", "weight": 0.8},
            {"op": "REWEIGHT_EDGE", "source": "frontend", "target": "checkoutservice:v2", "weight": 0.2},
        ],
    })

    result = evaluate_graph_update(_row(exp_dir, metadata), str(plans_dir))

    assert result["version_split_recall"] == 1.0
    assert result["edge_weight_mae"] == 0.0


def test_graph_update_eval_cli_writes_outputs(tmpdir):
    metadata = _base_metadata("t1-cli", "T1", "hpa_scale", "cartservice", "instance_count_change")
    pre = _snapshot(
        pods=[_pod("cart-1", "cartservice", "10.0.0.1")],
        services=[_svc("cartservice", ["10.0.0.1"])],
        deployments=[_dep("cartservice", 1)],
    )
    during = _snapshot(
        pods=[_pod("cart-1", "cartservice", "10.0.0.1"), _pod("cart-2", "cartservice", "10.0.0.2")],
        services=[_svc("cartservice", ["10.0.0.1", "10.0.0.2"])],
        deployments=[_dep("cartservice", 2)],
    )
    exp_dir = _make_exp(tmpdir, "t1-cli", metadata, pre, during)
    plans_dir = Path(str(tmpdir)) / "plans"
    graphs_dir = Path(str(tmpdir)) / "graphs"
    _write_json(plans_dir / "t1-cli.json", {
        "experiment_id": "t1-cli",
        "change_type": "node_scale",
        "confidence": 0.9,
        "causal_role": "effect_or_mixed",
        "node_ops": [{"op": "ADD_NODE", "node_id": "pod:cart-2"}],
    })
    graphs_dir.mkdir()
    index = Path(str(tmpdir)) / "index.csv"
    pd.DataFrame([_row(exp_dir, metadata)]).to_csv(index, index=False)
    out_dir = Path(str(tmpdir)) / "results"

    assert main([
        "--index", str(index),
        "--plans-dir", str(plans_dir),
        "--graphs-dir", str(graphs_dir),
        "--out-dir", str(out_dir),
        "--limit", "1",
    ]) == 0

    metrics_path = out_dir / "graph_update_metrics.csv"
    by_scenario_path = out_dir / "graph_update_by_scenario.csv"
    assert metrics_path.exists()
    assert by_scenario_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert metrics.loc[0, "experiment_id"] == "t1-cli"
    assert metrics.loc[0, "node_add_remove_f1"] == 1.0
    by_scenario = pd.read_csv(by_scenario_path)
    assert by_scenario.loc[0, "scenario_id"] == "T1"
