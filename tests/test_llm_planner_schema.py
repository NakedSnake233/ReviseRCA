import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.planner.llm_planner import build_evidence_json, main, plan_experiment_with_llm
from dyrca_rca.planner.plan_validator import validate_plan_json
from dyrca_rca.planner.prompt_templates import build_graph_update_prompt


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _metadata(exp_id, scenario, scenario_name, target, change_type, params=None):
    return {
        "experiment_id": exp_id,
        "scenario_id": scenario,
        "scenario_name": scenario_name,
        "target_service": target,
        "topology_change_type": change_type,
        "fault_params": params or {},
    }


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
        "timestamp": "2026-01-01T00:00:00Z",
        "pods": pods,
        "services": services,
        "deployments": deployments,
    }


def _make_exp(tmpdir, exp_name, metadata, pre, during):
    exp_dir = Path(str(tmpdir)) / exp_name
    _write_json(exp_dir / "metadata.json", metadata)
    _write_json(exp_dir / "ground_truth.json", {
        "experiment_id": metadata["experiment_id"],
        "root_cause_service": metadata["target_service"],
        "topology_change_type": metadata["topology_change_type"],
    })
    _write_json(exp_dir / "topology" / "pre.json", pre)
    _write_json(exp_dir / "topology" / "during.json", during)
    _write_json(exp_dir / "topology" / "post.json", during)
    _write_json(exp_dir / "topology_diff.json", {
        "schema": "topology_diff_v1",
        "annotation": {
            "verification_result": "CHANGED",
            "topology_change_type": metadata["topology_change_type"],
            "detected_types": [metadata["topology_change_type"]],
            "evidence": ["synthetic evidence"],
        },
        "comparisons": {"pre_vs_during": {}},
    })
    return exp_dir


def test_evidence_json_and_prompt_contains_required_guardrails(tmpdir):
    pre = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.0.1")],
        [_svc("checkoutservice", ["10.0.0.1"])],
        [_dep("checkoutservice", 1)],
    )
    during = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.0.1"), _pod("cart-1", "cartservice", "10.0.0.2")],
        [_svc("checkoutservice", ["10.0.0.2"])],
        [_dep("checkoutservice", 1), _dep("cartservice", 1)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t3-redirect",
        _metadata("t3-redirect", "T3", "selector_poison", "checkoutservice", "edge_redirect"),
        pre,
        during,
    )
    trace_dir = exp_dir / "traces"
    trace_dir.mkdir()
    (trace_dir / "trace_summary.csv").write_text(
        "source,declared_dst,observed_dst\nfrontend,checkoutservice,cartservice\n",
        encoding="utf-8",
    )

    evidence = build_evidence_json(str(exp_dir))
    prompt = build_graph_update_prompt(evidence)

    assert evidence["experiment_id"] == "t3-redirect"
    assert evidence["scenario_name"] == "selector_poison"
    assert evidence["target_service"] == "checkoutservice"
    assert evidence["endpoint_diff_summary"]["changed_services"][0]["service"] == "checkoutservice"
    assert evidence["trace_route_summary"]["declared_observed_mismatches"][0]["observed"] == "cartservice"
    assert "You are a graph update planner for microservice RCA." in prompt
    assert "Do not search or tune model hyperparameters." in prompt
    assert "Do NOT choose hidden_dim, learning_rate, dropout, epochs, model family, or optimizer." in prompt
    assert "Only output node_ops, edge_ops, weight_ops, state_transfer." in prompt
    assert "Use graph operations, not model parameter updates." in prompt
    assert "Use trace mismatch for selector poison." in prompt
    assert "Use version split for canary." in prompt
    assert "dependency_weight, anomaly_weight, topology_confidence, change_weight, final_weight" in prompt
    assert "ADD_BLACKHOLE_NODE creates a blackhole node, so place it in node_ops" in prompt


def test_plan_validator_accepts_t3_redirect_mock_response():
    response = json.dumps({
        "experiment_id": "t3-redirect",
        "change_type": "edge_redirect",
        "confidence": 0.91,
        "causal_role": "cause_of_fault",
        "node_ops": [],
        "edge_ops": [
            {
                "op": "REDIRECT_EDGE",
                "old_source": "checkoutservice",
                "old_target": "pod:checkout-1",
                "new_target": "pod:cart-1",
                "edge_type": "endpoint",
                "attrs": {"reason": "selector_poison"},
            },
            {
                "op": "ADD_TRACE_MISMATCH_EDGE",
                "source": "frontend",
                "target": "cartservice",
                "edge_type": "trace_mismatch",
                "attrs": {"declared_dst": "checkoutservice", "observed_dst": "cartservice"},
            },
        ],
        "weight_ops": [],
        "state_transfer": {},
        "evidence": [],
    })

    result = validate_plan_json(response)

    assert result.valid
    assert result.plan.change_type == "edge_redirect"
    assert [op.op for op in result.plan.edge_ops] == ["REDIRECT_EDGE", "ADD_TRACE_MISMATCH_EDGE"]


def test_llm_planner_uses_mock_t3_redirect_and_writes_cache(tmpdir):
    pre = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.1.1"), _pod("cart-1", "cartservice", "10.0.1.2")],
        [_svc("checkoutservice", ["10.0.1.1"])],
        [_dep("checkoutservice", 1), _dep("cartservice", 1)],
    )
    during = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.1.1"), _pod("cart-1", "cartservice", "10.0.1.2")],
        [_svc("checkoutservice", ["10.0.1.2"])],
        [_dep("checkoutservice", 1), _dep("cartservice", 1)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t3-redirect",
        _metadata("t3-redirect", "T3", "selector_poison", "checkoutservice", "edge_redirect"),
        pre,
        during,
    )
    calls = []

    def mock_llm(prompt):
        calls.append(prompt)
        return json.dumps({
            "experiment_id": "t3-redirect",
            "change_type": "edge_redirect",
            "confidence": 0.92,
            "causal_role": "cause_of_fault",
            "node_ops": [],
            "edge_ops": [
                {
                    "op": "REDIRECT_EDGE",
                    "old_source": "checkoutservice",
                    "old_target": "pod:checkout-1",
                    "new_target": "pod:cart-1",
                    "edge_type": "endpoint",
                }
            ],
            "weight_ops": [],
            "state_transfer": {},
            "evidence": [],
        })

    cache_dir = Path(str(tmpdir)) / "cache"
    plan = plan_experiment_with_llm(
        str(exp_dir),
        llm_client=mock_llm,
        cache_dir=str(cache_dir),
    )

    assert calls
    assert plan.change_type == "edge_redirect"
    assert plan.edge_ops[0].op == "REDIRECT_EDGE"
    assert (cache_dir / "t3-redirect.json").exists()


def test_llm_planner_uses_mock_t4_canary_response(tmpdir):
    pre = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.2.1", version="v1")],
        [_svc("checkoutservice", ["10.0.2.1"])],
        [_dep("checkoutservice", 1, version="v1")],
    )
    during = _snapshot(
        [
            _pod("checkout-1", "checkoutservice", "10.0.2.1", version="v1"),
            _pod("checkout-v2-1", "checkoutservice-v2", "10.0.2.2", version="v2"),
        ],
        [_svc("checkoutservice", ["10.0.2.1", "10.0.2.2"])],
        [_dep("checkoutservice", 1, version="v1"), _dep("checkoutservice-v2", 1, version="v2")],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t4-canary",
        _metadata(
            "t4-canary",
            "T4",
            "canary_release",
            "checkoutservice",
            "version_split",
            {"v1_weight": 90, "v2_weight": 10},
        ),
        pre,
        during,
    )

    def mock_llm(prompt):
        return json.dumps({
            "experiment_id": "t4-canary",
            "change_type": "edge_split",
            "confidence": 0.89,
            "causal_role": "cause_or_mixed",
            "node_ops": [
                {
                    "op": "SPLIT_NODE",
                    "parent": "checkoutservice",
                    "children": ["checkoutservice:v1", "checkoutservice:v2"],
                }
            ],
            "edge_ops": [
                {
                    "op": "SPLIT_EDGE",
                    "source": "frontend",
                    "target": "checkoutservice",
                    "new_target": "checkoutservice:v2",
                    "edge_type": "dependency",
                    "weight": 0.1,
                }
            ],
            "weight_ops": [
                {
                    "op": "REWEIGHT_EDGE",
                    "source": "frontend",
                    "target": "checkoutservice:v2",
                    "weight": 0.1,
                }
            ],
            "state_transfer": {},
            "evidence": [],
        })

    plan = plan_experiment_with_llm(
        str(exp_dir),
        llm_client=mock_llm,
        cache_dir=str(Path(str(tmpdir)) / "cache"),
    )

    assert plan.change_type == "edge_split"
    assert plan.node_ops[0].op == "SPLIT_NODE"
    assert plan.node_ops[0].children == ["checkoutservice:v1", "checkoutservice:v2"]
    assert plan.edge_ops[0].op == "SPLIT_EDGE"
    assert plan.weight_ops[0].weight == 0.1


def test_invalid_llm_output_falls_back_to_rule_planner(tmpdir):
    pre = _snapshot(
        [_pod("cart-1", "cartservice", "10.0.3.1")],
        [_svc("cartservice", ["10.0.3.1"])],
        [_dep("cartservice", 1)],
    )
    during = _snapshot(
        [_pod("cart-1", "cartservice", "10.0.3.1"), _pod("cart-2", "cartservice", "10.0.3.2")],
        [_svc("cartservice", ["10.0.3.1", "10.0.3.2"])],
        [_dep("cartservice", 2)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t1-scale",
        _metadata("t1-scale", "T1", "hpa_scale", "cartservice", "instance_count_change"),
        pre,
        during,
    )

    plan = plan_experiment_with_llm(
        str(exp_dir),
        llm_client=lambda prompt: '{"not": "a plan", "hyperparameters": {"lr": 1}}',
        cache_dir=str(Path(str(tmpdir)) / "cache"),
    )

    assert plan.change_type == "node_scale"
    assert any(ev.get("fallback_used") is True for ev in plan.evidence)


def test_llm_planner_cli_falls_back_without_network(tmpdir):
    pre = _snapshot(
        [_pod("cart-1", "cartservice", "10.0.4.1")],
        [_svc("cartservice", ["10.0.4.1"])],
        [_dep("cartservice", 1)],
    )
    during = _snapshot(
        [_pod("cart-1", "cartservice", "10.0.4.1"), _pod("cart-2", "cartservice", "10.0.4.2")],
        [_svc("cartservice", ["10.0.4.1", "10.0.4.2"])],
        [_dep("cartservice", 2)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t1-cli",
        _metadata("t1-cli", "T1", "hpa_scale", "cartservice", "instance_count_change"),
        pre,
        during,
    )
    out = Path(str(tmpdir)) / "plan.json"
    cache_dir = Path(str(tmpdir)) / "cache"

    assert main([
        "--experiment-dir", str(exp_dir),
        "--out", str(out),
        "--fallback", "rule",
        "--cache-dir", str(cache_dir),
    ]) == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["change_type"] == "node_scale"
    assert (cache_dir / "t1-cli.json").exists()
