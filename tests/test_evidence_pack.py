import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.rerank.evidence import build_evidence_pack


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _pod(name, service, ip, version="unknown"):
    return {"name": name, "service": service, "ip": ip, "version": version, "node": "n1", "status": "Running"}


def _svc(name, ips):
    return {"name": name, "endpoint_ips": ips}


def _dep(name, replicas, version="unknown"):
    return {"name": name, "replicas": replicas, "readyReplicas": replicas, "version": version}


def _snapshot(pods, services, deployments):
    return {"pods": pods, "services": services, "deployments": deployments}


def _make_exp(tmpdir):
    exp_dir = Path(str(tmpdir)) / "t3-redirect"
    _write_json(exp_dir / "metadata.json", {
        "experiment_id": "t3-redirect",
        "scenario_id": "T3",
        "scenario_name": "selector_poison",
        "target_service": "checkoutservice",
        "time_window": {
            "t_collect": "2026-01-01T00:00:00Z",
            "t_inject": "2026-01-01T00:01:00Z",
            "t_recover": "2026-01-01T00:02:00Z",
        },
    })
    _write_json(exp_dir / "topology" / "pre.json", _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.0.1"), _pod("cart-1", "cartservice", "10.0.0.2")],
        [_svc("checkoutservice", ["10.0.0.1"])],
        [_dep("checkoutservice", 1), _dep("cartservice", 1)],
    ))
    _write_json(exp_dir / "topology" / "during.json", _snapshot(
        [_pod("checkout-1", "checkoutservice", "10.0.0.1"), _pod("cart-1", "cartservice", "10.0.0.2")],
        [_svc("checkoutservice", ["10.0.0.2"])],
        [_dep("checkoutservice", 1), _dep("cartservice", 1)],
    ))
    _write_json(exp_dir / "topology" / "post.json", _snapshot([], [], []))
    trace_dir = exp_dir / "traces"
    trace_dir.mkdir()
    (trace_dir / "trace_summary.csv").write_text(
        "source,declared_dst,observed_dst,duration_ms,status_code\nfrontend,checkoutservice,cartservice,50,0\n",
        encoding="utf-8",
    )
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "errors.jsonl").write_text('{"service":"cartservice","level":"ERROR","message":"bad"}\n', encoding="utf-8")
    (logs_dir / "all.jsonl").write_text('{"service":"checkoutservice","level":"INFO","message":"ok"}\n', encoding="utf-8")
    return exp_dir


def test_evidence_pack_contains_candidate_modalities(tmpdir):
    exp_dir = _make_exp(tmpdir)
    graph = {
        "nodes": [
            {"id": "checkoutservice", "attrs": {"service": "checkoutservice", "type": "service"}},
            {"id": "cartservice", "attrs": {"service": "cartservice", "type": "service"}},
        ],
        "edges": [
            {
                "source": "checkoutservice",
                "target": "cartservice",
                "edge_type": "trace_mismatch",
                "attrs": {"trace_mismatch": True, "redirected_from": "checkoutservice->pod:checkout-1"},
            }
        ],
    }
    prediction = {
        "experiment_id": "t3-redirect",
        "scenario_id": "T3",
        "target_service": "checkoutservice",
        "label_node_id": "checkoutservice",
        "ranked_node_ids": "cartservice|checkoutservice",
        "top_score": 2.0,
    }

    pack = build_evidence_pack(prediction, str(exp_dir), graph, top_k=2)

    assert pack["experiment_id"] == "t3-redirect"
    assert len(pack["candidates"]) == 2
    cart = pack["candidates"][0]
    assert cart["candidate"] == "cartservice"
    assert cart["model_score"] == 2.0
    assert cart["topology_evidence"]
    assert cart["trace_evidence"]
    assert cart["log_evidence"]
    assert cart["temporal_order_evidence"]
    assert all("id" in ev for ev in cart["evidence_refs"])


def test_evidence_pack_top_k_limits_candidates(tmpdir):
    exp_dir = _make_exp(tmpdir)
    graph = {"nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}], "edges": []}
    prediction = {
        "experiment_id": "t3-redirect",
        "ranked_node_ids": "a|b|c",
        "label_node_id": "a",
    }

    pack = build_evidence_pack(prediction, str(exp_dir), graph, top_k=1)

    assert [c["candidate"] for c in pack["candidates"]] == ["a"]
