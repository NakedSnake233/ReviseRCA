import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.graph.executor import execute_plan
from dyrca_rca.graph.update_ops import GraphUpdatePlan


def _base_graph():
    return {
        "nodes": [
            {"id": "frontend", "attrs": {"type": "service"}},
            {"id": "checkoutservice", "attrs": {"type": "service"}},
            {"id": "cartservice", "attrs": {"type": "service"}},
            {"id": "pod:checkout-old", "attrs": {"service": "checkoutservice"}},
        ],
        "edges": [
            {
                "source": "frontend",
                "target": "checkoutservice",
                "edge_type": "dependency",
                "weight": 1.0,
                "attrs": {"topology_confidence": 1.0},
            },
            {
                "source": "checkoutservice",
                "target": "cartservice",
                "edge_type": "dependency",
                "weight": 2.0,
                "attrs": {"topology_confidence": 1.0},
            },
        ],
    }


def _nodes(graph):
    return {node["id"]: node for node in graph["nodes"]}


def _edge(graph, source, target, edge_type=None):
    for edge in graph["edges"]:
        if edge["source"] == source and edge["target"] == target:
            if edge_type is None or edge["edge_type"] == edge_type:
                return edge
    raise AssertionError("missing edge: %s->%s" % (source, target))


def test_t2_old_pod_new_pod_alignment():
    plan = GraphUpdatePlan.from_dict({
        "experiment_id": "t2-checkoutservice-run001",
        "change_type": "instance_drift",
        "confidence": 0.9,
        "causal_role": "root_cause",
        "node_ops": [
            {
                "op": "ALIGN_NODE",
                "old_node": "pod:checkout-old",
                "new_node": "pod:checkout-new",
                "attrs": {"service": "checkoutservice", "reason": "pod_replacement"},
            }
        ],
    })

    result = execute_plan(_base_graph(), plan)
    graph = result.graph

    assert result.warnings == []
    assert "pod:checkout-old" in _nodes(graph)
    assert "pod:checkout-new" in _nodes(graph)
    assert _nodes(graph)["pod:checkout-old"]["attrs"]["aligned_to"] == "pod:checkout-new"
    assert _nodes(graph)["pod:checkout-new"]["attrs"]["aligned_from"] == "pod:checkout-old"
    change_edge = _edge(graph, "pod:checkout-old", "pod:checkout-new", "change_event")
    assert change_edge["attrs"]["alignment"] is True
    assert change_edge["attrs"]["change_event"] is True


def test_t3_redirect_edge_marks_old_low_confidence_and_adds_new_edge():
    plan = GraphUpdatePlan.from_dict({
        "experiment_id": "t3-checkoutservice-redirect",
        "change_type": "edge_redirect",
        "confidence": 0.95,
        "causal_role": "root_cause",
        "edge_ops": [
            {
                "op": "REDIRECT_EDGE",
                "old_source": "frontend",
                "old_target": "checkoutservice",
                "new_target": "cartservice",
                "attrs": {"selector_redirect": True},
            }
        ],
    })

    result = execute_plan(_base_graph(), plan)
    graph = result.graph

    old_edge = _edge(graph, "frontend", "checkoutservice")
    new_edge = _edge(graph, "frontend", "cartservice")
    assert old_edge["attrs"]["topology_confidence"] == 0.1
    assert old_edge["attrs"]["redirected_to"] == "frontend->cartservice"
    assert new_edge["attrs"]["redirected_from"] == "frontend->checkoutservice"
    assert new_edge["attrs"]["selector_redirect"] is True


def test_t3_empty_endpoint_adds_blackhole_node_and_edge():
    plan = GraphUpdatePlan.from_dict({
        "experiment_id": "t3-checkoutservice-empty",
        "change_type": "edge_removal",
        "confidence": 0.9,
        "causal_role": "root_cause",
        "node_ops": [
            {"op": "ADD_BLACKHOLE_NODE", "service": "checkoutservice"}
        ],
    })

    result = execute_plan(_base_graph(), plan)
    graph = result.graph

    assert "blackhole:checkoutservice" in _nodes(graph)
    assert _nodes(graph)["blackhole:checkoutservice"]["attrs"]["blackhole"] is True
    blackhole_edge = _edge(graph, "checkoutservice", "blackhole:checkoutservice", "blackhole")
    assert blackhole_edge["attrs"]["op"] == "ADD_BLACKHOLE_NODE"


def test_t4_split_node_and_split_edge():
    plan = GraphUpdatePlan.from_dict({
        "experiment_id": "t4-checkoutservice-canary",
        "change_type": "version_split",
        "confidence": 0.88,
        "causal_role": "root_cause",
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
                "attrs": {"traffic_weight": 0.1},
            }
        ],
        "weight_ops": [
            {
                "op": "REWEIGHT_EDGE",
                "source": "frontend",
                "target": "checkoutservice:v2",
                "weight": 0.1,
                "attrs": {"weight_source": "canary"},
            }
        ],
    })

    result = execute_plan(_base_graph(), plan)
    graph = result.graph
    nodes = _nodes(graph)

    assert result.warnings == []
    assert "checkoutservice" in nodes
    assert nodes["checkoutservice"]["attrs"]["split_parent"] is True
    assert nodes["checkoutservice:v1"]["attrs"]["split_child"] is True
    assert nodes["checkoutservice:v2"]["attrs"]["split_child"] is True
    assert _edge(graph, "checkoutservice", "checkoutservice:v2", "version_split")
    split_edge = _edge(graph, "frontend", "checkoutservice:v2")
    assert split_edge["attrs"]["split_from"] == "frontend->checkoutservice"
    assert split_edge["weight"] == 0.1
    assert split_edge["attrs"]["reweighted"] is True


def test_invalid_ops_are_reported_not_silently_dropped():
    plan = GraphUpdatePlan.from_dict({
        "experiment_id": "bad-plan",
        "change_type": "bad",
        "confidence": 0.5,
        "causal_role": "test",
        "edge_ops": [{"op": "ADD_EDGE", "source": "frontend"}],
        "node_ops": ["not-an-object"],
    })

    result = execute_plan(_base_graph(), plan)

    assert result.warnings
    assert "ADD_EDGE requires source and target" in "; ".join(result.warnings)
    assert "invalid node op type" in "; ".join(result.warnings)


def test_executor_cli(tmpdir):
    graph_path = Path(str(tmpdir)) / "graph.json"
    plan_path = Path(str(tmpdir)) / "plan.json"
    out_path = Path(str(tmpdir)) / "updated.json"
    graph_path.write_text(json.dumps(_base_graph()), encoding="utf-8")
    plan_path.write_text(json.dumps({
        "experiment_id": "cli",
        "change_type": "edge_removal",
        "confidence": 1.0,
        "causal_role": "root_cause",
        "node_ops": [{"op": "ADD_BLACKHOLE_NODE", "service": "checkoutservice"}],
    }), encoding="utf-8")

    from dyrca_rca.graph.executor import main

    assert main(["--graph", str(graph_path), "--plan", str(plan_path), "--out", str(out_path)]) == 0
    updated = json.loads(out_path.read_text(encoding="utf-8"))
    assert "blackhole:checkoutservice" in _nodes(updated)
