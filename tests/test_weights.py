import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.features.metrics import load_metric_features
from dyrca_rca.features.topology import load_topology_features
from dyrca_rca.features.traces import load_trace_features
from dyrca_rca.graph.weights import apply_edge_weights, main


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _metadata(exp_id, target, change_type="", params=None):
    return {
        "experiment_id": exp_id,
        "scenario_id": "T",
        "scenario_name": "synthetic",
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


def _make_exp(tmpdir, name, metadata, pre, during):
    exp_dir = Path(str(tmpdir)) / name
    _write_json(exp_dir / "metadata.json", metadata)
    _write_json(exp_dir / "topology" / "pre.json", pre)
    _write_json(exp_dir / "topology" / "during.json", during)
    _write_json(exp_dir / "topology" / "post.json", during)
    _write_json(exp_dir / "topology_diff.json", {
        "schema": "topology_diff_v1",
        "annotation": {"change_score": 0.5},
        "comparisons": {"pre_vs_during": {"k8s_layer": {}, "graph_layer": {}}},
    })
    return exp_dir


def _base_graph():
    return {
        "nodes": [
            {"id": "frontend"},
            {"id": "checkoutservice"},
            {"id": "cartservice"},
        ],
        "edges": [
            {"source": "frontend", "target": "checkoutservice", "edge_type": "dependency"}
        ],
    }


def test_missing_metrics_do_not_crash_and_record_missing_files(tmpdir):
    pre = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.0.1")],
        [_svc("checkoutservice", ["10.0.0.1"])],
        [_dep("checkoutservice", 1)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "missing-metrics",
        _metadata("missing-metrics", "checkoutservice"),
        pre,
        pre,
    )

    metrics = load_metric_features(str(exp_dir))
    weighted = apply_edge_weights(str(exp_dir), _base_graph())

    assert "request_rate.csv" in metrics["missing_files"]
    assert weighted["edges"][0]["attrs"]["dependency_weight"] == 1.0
    assert weighted["edges"][0]["attrs"]["anomaly_weight"] == 0.0
    assert weighted["edges"][0]["attrs"]["topology_confidence"] == 1.0
    assert weighted["attrs"]["feature_status"]["trace_missing"] is True


def test_t3_topology_confidence_low_for_selector_redirect(tmpdir):
    pre = _snapshot(
        [
            _pod("checkout-1", "checkoutservice", "10.0.1.1"),
            _pod("cart-1", "cartservice", "10.0.1.2"),
        ],
        [_svc("checkoutservice", ["10.0.1.1"])],
        [_dep("checkoutservice", 1), _dep("cartservice", 1)],
    )
    during = _snapshot(
        [
            _pod("checkout-1", "checkoutservice", "10.0.1.1"),
            _pod("cart-1", "cartservice", "10.0.1.2"),
        ],
        [_svc("checkoutservice", ["10.0.1.2"])],
        [_dep("checkoutservice", 1), _dep("cartservice", 1)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t3-redirect",
        _metadata("t3-redirect", "checkoutservice", "edge_redirect"),
        pre,
        during,
    )
    trace_dir = exp_dir / "traces"
    trace_dir.mkdir()
    (trace_dir / "trace_summary.csv").write_text(
        "source,declared_dst,observed_dst,duration_ms,status_code\nfrontend,checkoutservice,cartservice,50,0\n",
        encoding="utf-8",
    )

    traces = load_trace_features(str(exp_dir))
    weighted = apply_edge_weights(str(exp_dir), _base_graph())

    assert traces["route_mismatch_candidates"][0]["declared"] == "checkoutservice"
    assert weighted["edges"][0]["attrs"]["topology_confidence"] == 0.1
    assert weighted["edges"][0]["attrs"]["final_weight"] < 0.4


def test_t4_version_edge_weights_from_metadata_params(tmpdir):
    pre = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.2.1", version="v1")],
        [_svc("checkoutservice", ["10.0.2.1"])],
        [_dep("checkoutservice", 1, "v1")],
    )
    during = _snapshot(
        [
            _pod("checkout-1", "checkoutservice", "10.0.2.1", version="v1"),
            _pod("checkout-v2-1", "checkoutservice-v2", "10.0.2.2", version="v2"),
        ],
        [_svc("checkoutservice", ["10.0.2.1", "10.0.2.2"])],
        [_dep("checkoutservice", 1, "v1"), _dep("checkoutservice-v2", 1, "v2")],
    )
    exp_dir = _make_exp(
        tmpdir,
        "t4-canary",
        _metadata("t4-canary", "checkoutservice", "version_split", {"v1_weight": 90, "v2_weight": 10}),
        pre,
        during,
    )
    graph = {
        "nodes": [
            {"id": "frontend"},
            {"id": "checkoutservice"},
            {"id": "checkoutservice:v1"},
            {"id": "checkoutservice:v2"},
        ],
        "edges": [
            {"source": "frontend", "target": "checkoutservice:v1", "edge_type": "dependency"},
            {"source": "frontend", "target": "checkoutservice:v2", "edge_type": "dependency"},
        ],
    }

    topo = load_topology_features(str(exp_dir))
    weighted = apply_edge_weights(str(exp_dir), graph)
    by_target = {edge["target"]: edge for edge in weighted["edges"]}

    assert topo["topo_change"]["version_split"] == 1.0
    assert by_target["checkoutservice:v1"]["attrs"]["dependency_weight"] == 0.9
    assert by_target["checkoutservice:v2"]["attrs"]["dependency_weight"] == 0.1
    assert by_target["checkoutservice:v1"]["weight"] > by_target["checkoutservice:v2"]["weight"]


def test_weights_cli_writes_output(tmpdir):
    pre = _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.3.1")],
        [_svc("checkoutservice", ["10.0.3.1"])],
        [_dep("checkoutservice", 1)],
    )
    exp_dir = _make_exp(
        tmpdir,
        "cli",
        _metadata("cli", "checkoutservice"),
        pre,
        pre,
    )
    graph_path = Path(str(tmpdir)) / "graph.json"
    out_path = Path(str(tmpdir)) / "weighted.json"
    graph_path.write_text(json.dumps(_base_graph()), encoding="utf-8")

    assert main(["--experiment-dir", str(exp_dir), "--graph", str(graph_path), "--out", str(out_path)]) == 0
    weighted = json.loads(out_path.read_text(encoding="utf-8"))
    assert weighted["edges"][0]["attrs"]["final_weight"] == weighted["edges"][0]["weight"]
