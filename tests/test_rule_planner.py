import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.planner.rule_planner import main, plan_experiment, plan_to_dict


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


def _make_exp(tmpdir, exp_name, metadata, pre, during, topology_diff=None, ground_truth=None):
    exp_dir = Path(str(tmpdir)) / exp_name
    _write_json(exp_dir / "metadata.json", metadata)
    _write_json(exp_dir / "ground_truth.json", ground_truth or {
        "experiment_id": metadata["experiment_id"],
        "root_cause_service": "conflicting-ground-truth",
        "topology_change_type": "conflicting-type",
    })
    _write_json(exp_dir / "topology" / "pre.json", pre)
    _write_json(exp_dir / "topology" / "during.json", during)
    _write_json(exp_dir / "topology" / "post.json", during)
    if topology_diff is not None:
        _write_json(exp_dir / "topology_diff.json", topology_diff)
    return exp_dir


def _ops(plan, kind):
    return [op.op for op in getattr(plan, kind)]


def test_t1_node_scale_adds_new_pods_edges_and_weights(tmpdir):
    pre = _snapshot(
        pods=[_pod("cart-1", "cartservice", "10.0.0.1")],
        services=[_svc("cartservice", ["10.0.0.1"])],
        deployments=[_dep("cartservice", 1)],
    )
    during = _snapshot(
        pods=[
            _pod("cart-1", "cartservice", "10.0.0.1"),
            _pod("cart-2", "cartservice", "10.0.0.2"),
            _pod("cart-3", "cartservice", "10.0.0.3"),
        ],
        services=[_svc("cartservice", ["10.0.0.1", "10.0.0.2", "10.0.0.3"])],
        deployments=[_dep("cartservice", 3)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t1-cart",
        _base_metadata("t1-cart", "T1", "hpa_scale", "cartservice", "instance_count_change"),
        pre,
        during,
    )

    plan = plan_experiment(str(exp_dir))

    assert plan.change_type == "node_scale"
    assert plan.causal_role == "effect_or_mixed"
    assert _ops(plan, "node_ops").count("ADD_NODE") == 2
    assert _ops(plan, "edge_ops").count("ADD_EDGE") == 2
    assert _ops(plan, "weight_ops").count("REWEIGHT_EDGE") == 3
    assert {op.node_id for op in plan.node_ops} == {"pod:cart-2", "pod:cart-3"}
    assert all(abs(op.weight - (1.0 / 3.0)) < 1e-9 for op in plan.weight_ops)


def test_t2_node_drift_aligns_old_and_new_pod(tmpdir):
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
    exp_dir = _make_exp(
        tmpdir,
        "t2-checkout",
        _base_metadata("t2-checkout", "T2", "pod_drift", "checkoutservice", "instance_drift"),
        pre,
        during,
    )

    plan = plan_experiment(str(exp_dir))

    assert plan.change_type == "node_drift"
    assert plan.causal_role == "topology_event"
    assert _ops(plan, "node_ops") == ["ALIGN_NODE"]
    assert plan.node_ops[0].old_node == "pod:checkout-old"
    assert plan.node_ops[0].new_node == "pod:checkout-new"
    assert _ops(plan, "edge_ops") == ["REPLACE_ENDPOINT_EDGE"]
    assert plan.edge_ops[0].old_target == "pod:checkout-old"
    assert plan.edge_ops[0].new_target == "pod:checkout-new"


def test_t3_redirect_edge_and_trace_mismatch(tmpdir):
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
    exp_dir = _make_exp(
        tmpdir,
        "t3-redirect",
        _base_metadata("t3-redirect", "T3", "selector_poison", "checkoutservice", "edge_redirect"),
        pre,
        during,
    )
    trace_dir = exp_dir / "traces"
    trace_dir.mkdir()
    (trace_dir / "trace_summary.csv").write_text(
        "source,declared_dst,observed_dst\nfrontend,checkoutservice,cartservice\n",
        encoding="utf-8",
    )

    plan = plan_experiment(str(exp_dir))

    assert plan.change_type == "edge_redirect"
    assert plan.causal_role == "cause_of_fault"
    assert "REDIRECT_EDGE" in _ops(plan, "edge_ops")
    assert "ADD_TRACE_MISMATCH_EDGE" in _ops(plan, "edge_ops")
    redirect = [op for op in plan.edge_ops if op.op == "REDIRECT_EDGE"][0]
    assert redirect.old_source == "checkoutservice"
    assert redirect.old_target == "pod:checkout-1"
    assert redirect.new_target == "pod:cart-1"
    mismatch = [op for op in plan.edge_ops if op.op == "ADD_TRACE_MISMATCH_EDGE"][0]
    assert mismatch.source == "frontend"
    assert mismatch.target == "cartservice"


def test_t3_empty_endpoint_adds_blackhole_and_edge(tmpdir):
    pre = _snapshot(
        pods=[_pod("checkout-1", "checkoutservice", "10.0.3.1")],
        services=[_svc("checkoutservice", ["10.0.3.1"])],
        deployments=[_dep("checkoutservice", 1)],
    )
    during = _snapshot(
        pods=[_pod("checkout-1", "checkoutservice", "10.0.3.1")],
        services=[_svc("checkoutservice", [])],
        deployments=[_dep("checkoutservice", 1)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t3-empty",
        _base_metadata("t3-empty", "T3", "selector_poison", "checkoutservice", "edge_removal"),
        pre,
        during,
    )

    plan = plan_experiment(str(exp_dir))

    assert plan.change_type == "edge_removal"
    assert plan.causal_role == "cause_of_fault"
    assert _ops(plan, "node_ops") == ["ADD_BLACKHOLE_NODE"]
    assert plan.node_ops[0].service == "checkoutservice"
    assert _ops(plan, "edge_ops") == ["ADD_EDGE"]
    assert plan.edge_ops[0].target == "blackhole:checkoutservice"


def test_t4_split_node_and_split_edges_from_metadata_weights(tmpdir):
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
    exp_dir = _make_exp(
        tmpdir,
        "t4-canary",
        _base_metadata(
            "t4-canary",
            "T4",
            "canary_release",
            "checkoutservice",
            "version_split",
            fault_params={"v1_weight": 80, "v2_weight": 20},
        ),
        pre,
        during,
    )

    plan = plan_experiment(str(exp_dir), g_pre={
        "edges": [{"source": "frontend", "target": "checkoutservice"}]
    })

    assert plan.change_type == "edge_split"
    assert plan.causal_role == "cause_or_mixed"
    assert _ops(plan, "node_ops") == ["SPLIT_NODE"]
    assert set(plan.node_ops[0].children) == {"checkoutservice:v1", "checkoutservice:v2"}
    split_edges = [op for op in plan.edge_ops if op.op == "SPLIT_EDGE"]
    assert {op.new_target for op in split_edges} >= {"checkoutservice:v1", "checkoutservice:v2"}
    weights = {op.target: op.weight for op in plan.weight_ops}
    assert weights["checkoutservice:v1"] == 0.8
    assert weights["checkoutservice:v2"] == 0.2


def test_ground_truth_is_eval_metadata_not_decision_source(tmpdir):
    pre = _snapshot(
        pods=[_pod("cart-1", "cartservice", "10.0.5.1")],
        services=[_svc("cartservice", ["10.0.5.1"])],
        deployments=[_dep("cartservice", 1)],
    )
    during = _snapshot(
        pods=[_pod("cart-1", "cartservice", "10.0.5.1"), _pod("cart-2", "cartservice", "10.0.5.2")],
        services=[_svc("cartservice", ["10.0.5.1", "10.0.5.2"])],
        deployments=[_dep("cartservice", 2)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t1-conflicting-gt",
        _base_metadata("t1-conflicting-gt", "T1", "hpa_scale", "cartservice", "instance_count_change"),
        pre,
        during,
        ground_truth={
            "experiment_id": "t1-conflicting-gt",
            "root_cause_service": "checkoutservice",
            "topology_change_type": "edge_redirect",
        },
    )

    plan = plan_experiment(str(exp_dir))
    plan_dict = plan_to_dict(plan)

    assert plan.change_type == "node_scale"
    assert plan.node_ops[0].node_id == "pod:cart-2"
    assert any(ev.get("source") == "ground_truth_eval_metadata" for ev in plan_dict["evidence"])


def test_rule_planner_cli_writes_plan(tmpdir):
    pre = _snapshot(
        pods=[_pod("checkout-1", "checkoutservice", "10.0.6.1")],
        services=[_svc("checkoutservice", ["10.0.6.1"])],
        deployments=[_dep("checkoutservice", 1)],
    )
    during = _snapshot(
        pods=[_pod("checkout-1", "checkoutservice", "10.0.6.1")],
        services=[_svc("checkoutservice", [])],
        deployments=[_dep("checkoutservice", 1)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t3-cli",
        _base_metadata("t3-cli", "T3", "selector_poison", "checkoutservice", "edge_removal"),
        pre,
        during,
    )
    out = Path(str(tmpdir)) / "plan.json"

    assert main(["--experiment-dir", str(exp_dir), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["experiment_id"] == "t3-cli"
    assert data["change_type"] == "edge_removal"
    assert data["node_ops"][0]["op"] == "ADD_BLACKHOLE_NODE"
