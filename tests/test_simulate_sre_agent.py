import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from dyrca_rca.experiments.simulate_sre_agent import (
    DIAGNOSIS_SCHEMA, _call_agent, _decision, _externalize_packet, _latest,
    _repair_plan, run, seed_endpoint_defect,
)
from dyrca_rca.interaction.observation import digest, evidence
from dyrca_rca.planner.llm_planner import ResponsesAPIClient


def sample_view():
    pre = evidence("topology", {
        "phase": "pre", "timestamp": "2026-09-29T00:00:00Z",
        "services": [{"name": "private-cart", "endpoint_ips": ["10.1.2.3"]}],
        "pods": [], "deployments": [],
    })
    during = evidence("topology", {
        "phase": "during", "timestamp": "2026-09-29T00:01:00Z",
        "services": [{"name": "private-cart", "endpoint_ips": ["10.1.2.4"]}],
        "pods": [], "deployments": [],
    })
    view = {"incident_id": "incident-opaque", "system_id": "private-system",
            "as_of": "2026-09-29T00:01:00Z", "evidence": [pre, during],
            "modalities": {"traces": {"status": "available"}}}
    view["observation_version"] = digest(view)
    return view


def test_seeded_defect_is_deterministic_and_does_not_mutate_source():
    original = sample_view()
    changed, oracle, record = seed_endpoint_defect(original, 42)
    again, _, _ = seed_endpoint_defect(original, 42)
    assert _latest(original)["data"]["services"][0]["endpoint_ips"] == ["10.1.2.4"]
    assert _latest(changed)["data"]["services"][0]["endpoint_ips"] == []
    assert changed == again
    assert oracle["endpoint_ips"] == ["10.1.2.4"]
    assert record["service"] == "private-cart"
    no_defect, _, no_defect_record = seed_endpoint_defect(original, 42, False)
    assert no_defect == original
    assert no_defect_record["seeded_defect"] is False


def test_external_payload_uses_case_local_aliases_and_restores_outputs():
    packet = {
        "application": "private-system", "as_of": "2026-09-29T00:01:00Z",
        "service_topology": [{"name": "private-cart", "endpoint_ips": []}],
        "pre_service_topology": [{"name": "private-cart", "endpoint_ips": ["10.1.2.3"]}],
        "trace_service_evidence": [{"id": "trace:private", "service": "private-cart",
                                     "error_count": 2, "status_count": 5}],
        "oracle_check": {"service": "private-cart", "endpoint_ips": ["10.1.2.4"],
                         "source": "collector_during_snapshot_proxy"},
        "eligible_evidence_ids": ["trace:private"],
        "ai_recommendation": {"ranked_services": ["private-cart"], "top_score": 0.3},
    }
    remote, restore = _externalize_packet(packet, "diagnose")
    serialized = json.dumps(remote)
    for secret in ("private-cart", "private-system", "10.1.2.3",
                   "10.1.2.4", "trace:private", "2026-09-29"):
        assert secret not in serialized
    assert restore({"service": "S001", "mechanism": "unknown",
                    "evidence_ids": ["V001"], "confidence": 0.5})["service"] == "private-cart"
    repair_remote, repair_restore = _externalize_packet(packet, "repair")
    assert repair_restore({"action": "repair", "service": "S001",
                           "endpoint_ips": ["E002"], "reason": "checked"})["endpoint_ips"] == ["10.1.2.4"]
    assert repair_remote["oracle_check"]["endpoint_ips"] == ["E002"]


def test_invalid_llm_output_uses_deterministic_fallback(monkeypatch):
    class FakeClient:
        session_id = ""

        def __call__(self, prompt):
            return '{"service":"S999","mechanism":"guess","evidence_ids":[],"confidence":0.9}'

    monkeypatch.setattr(ResponsesAPIClient, "from_env", lambda: FakeClient())
    packet = {"service_topology": [{"name": "svc", "endpoint_ips": []}],
              "pre_service_topology": [], "trace_service_evidence": [],
              "eligible_evidence_ids": [], "oracle_check": None}
    fallback = {"service": "svc", "mechanism": "fallback",
                "evidence_ids": [], "confidence": 0}
    result, source, error = _call_agent(
        "diagnose", packet, DIAGNOSIS_SCHEMA, fallback,
        SimpleNamespace(planner="llm", seed=42), "case", "H0")
    assert result == fallback
    assert source == "rule_fallback"
    assert error == "ValueError"


def test_valid_but_wrong_repair_is_rejected_not_replaced(monkeypatch):
    import dyrca_rca.experiments.simulate_sre_agent as module

    def bad_proposal(*args, **kwargs):
        return ({"action": "repair", "service": "private-cart",
                 "endpoint_ips": ["10.9.9.9"], "reason": "guess"}, "llm", None)

    monkeypatch.setattr(module, "_call_agent", bad_proposal)
    changed, oracle, _ = seed_endpoint_defect(sample_view(), 42)
    repair, source, error, raw_proposed, rejected = _repair_plan(
        changed, None, oracle, SimpleNamespace(), "case")
    assert repair["action"] == "none"
    assert source == "llm_guard_rejected"
    assert error == "unsupported_repair"
    assert raw_proposed and rejected


def test_preview_decision_omits_stale_aliases_from_repair_reason(monkeypatch):
    import dyrca_rca.experiments.simulate_sre_agent as module
    seen = {}

    def capture(kind, packet, schema, fallback, args, case_id, condition):
        seen.update(packet)
        return fallback, "rule", None

    monkeypatch.setattr(module, "_call_agent", capture)
    oracle = {"service": "private-cart", "endpoint_ips": ["10.1.2.4"],
              "source": "collector_during_snapshot_proxy"}
    repair = {"action": "repair", "service": "private-cart",
              "endpoint_ips": ["10.1.2.4"],
              "reason": "old aliases S009/E008"}
    empty_change = {"added": [], "removed": [], "changed": []}
    preview = {"preview": {"prediction": {"ranked_node_ids": ["private-cart"]}},
               "comparison": {"nodes": empty_change, "edges": empty_change,
                              "evidence": empty_change}}
    _decision(oracle, repair, preview, "private-cart", [], SimpleNamespace(), "case")
    assert "reason" not in seen["repair"]
    remote, _ = _externalize_packet(seen, "decide")
    assert remote["oracle_check"]["service"] == remote["repair"]["service"]
    assert remote["oracle_check"]["endpoint_ips"] == remote["repair"]["endpoint_ips"]
    assert remote["preview"]["after_endpoint_ips"] == remote["oracle_check"]["endpoint_ips"]


def test_llm_run_requires_explicit_external_evidence_permission(tmpdir):
    tmp_path = tmpdir
    args = argparse.Namespace(
        data_root=str(tmp_path / "raw"), out_dir=str(tmp_path / "out"),
        pilot_dir=str(tmp_path / "pilot"), system="boutique", seed=42,
        planner="llm", allow_external_evidence=False, oracle_prob=0.5,
        defect_prob=0.5, limit=1, max_rows=2000)
    with pytest.raises(ValueError, match="allow-external-evidence"):
        run(args)


def test_go_client_sends_session_header_without_httpx(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"output": [{"type": "message", "content": [
                {"type": "output_text", "text": "PONG"}]}]}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResponse()

    import requests
    monkeypatch.setitem(sys.modules, "httpx", None)
    monkeypatch.setattr(requests, "post", fake_post)
    client = ResponsesAPIClient("https://opencode.ai/zen/go/v1", "fake-secret",
                                "gpt-5.6-luna", session_id="stable-test")
    assert client("ping") == "PONG"
    assert captured["headers"]["x-opencode-session"] == "stable-test"
    assert captured["headers"]["User-Agent"] == "dyrca-research-agent/0.1"
    assert captured["url"].endswith("/responses")
