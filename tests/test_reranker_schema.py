import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.rerank.llm_reranker import (
    build_rerank_prompt,
    guard_rerank_result,
    rerank_pack_llm,
    validate_rerank_response,
)
from dyrca_rca.rerank.rule_reranker import rerank_pack_rule, write_rerank_outputs


def _pack_t3():
    return {
        "experiment_id": "t3",
        "scenario_id": "T3",
        "scenario_name": "selector_poison",
        "target_service": "cartservice",
        "candidates": [
            {
                "candidate": "checkoutservice",
                "service": "checkoutservice",
                "model_score": 1.2,
                "original_rank": 1,
                "trace_evidence": [],
                "topology_evidence": [],
                "metric_evidence": [],
                "log_evidence": [],
                "counter_evidence": [{"id": "counter:1", "type": "counter", "summary": "no direct telemetry evidence"}],
                "evidence_refs": [{"id": "counter:1", "type": "counter", "summary": "no direct telemetry evidence"}],
            },
            {
                "candidate": "cartservice",
                "service": "cartservice",
                "model_score": 1.0,
                "original_rank": 2,
                "trace_evidence": [{"id": "trace:1", "type": "trace", "summary": "declared vs observed route mismatch"}],
                "topology_evidence": [{"id": "topo:1", "type": "topology", "summary": "redirect or trace mismatch edge"}],
                "metric_evidence": [],
                "log_evidence": [],
                "counter_evidence": [],
                "evidence_refs": [
                    {"id": "trace:1", "type": "trace", "summary": "declared vs observed route mismatch"},
                    {"id": "topo:1", "type": "topology", "summary": "redirect or trace mismatch edge"},
                ],
            },
        ],
    }


def _pack_t4():
    return {
        "experiment_id": "t4",
        "scenario_id": "T4",
        "scenario_name": "canary_release",
        "candidates": [
            {
                "candidate": "checkoutservice:v2",
                "service": "checkoutservice",
                "model_score": 0.5,
                "trace_evidence": [{"id": "trace:v2", "type": "trace", "summary": "abnormal trace signal"}],
                "topology_evidence": [{"id": "topo:v2", "type": "topology", "summary": "version split node"}],
                "metric_evidence": [],
                "log_evidence": [],
                "counter_evidence": [],
                "evidence_refs": [
                    {"id": "trace:v2", "type": "trace", "summary": "abnormal trace signal"},
                    {"id": "topo:v2", "type": "topology", "summary": "version split node"},
                ],
            },
            {
                "candidate": "checkoutservice",
                "service": "checkoutservice",
                "model_score": 1.0,
                "trace_evidence": [],
                "topology_evidence": [],
                "metric_evidence": [],
                "log_evidence": [],
                "counter_evidence": [],
                "evidence_refs": [],
            },
        ],
    }


def test_rerank_prompt_is_json_only_and_no_hpo():
    prompt = build_rerank_prompt(_pack_t3())

    assert "You are an RCA evidence checker." in prompt
    assert "Output valid JSON only." in prompt
    assert "Do not introduce new candidates outside Top-K" in prompt
    assert "Canary v2 fault should be version-level." in prompt
    assert "Do not invent evidence." in prompt
    assert "Do not use LLM output to update model training parameters." in prompt
    assert "Do not tune hyperparameters" in prompt
    assert '"changed_order": true' in prompt


def test_llm_reranker_validates_response_refs():
    pack = _pack_t3()
    response = json.dumps({
        "final_ranking": [
            {
                "candidate": "cartservice",
                "final_score": 0.9,
                "confidence": 0.8,
                "reason": "trace mismatch supports cartservice",
                "supporting_evidence": ["trace:1", "topo:1"],
                "counter_evidence": [],
            },
            {
                "candidate": "checkoutservice",
                "final_score": 0.2,
                "confidence": 0.4,
                "reason": "less direct evidence",
                "supporting_evidence": [],
                "counter_evidence": ["counter:1"],
            },
        ],
        "changed_order": True,
        "summary": "trace mismatch supports cartservice",
    })

    result = validate_rerank_response(response, pack)

    assert result["final_ranking"][0]["candidate"] == "cartservice"
    assert result["confidence"] == 0.8
    assert result["changed_order"] is True


def test_guard_preserves_gnn_top_for_telemetry_only_promotion():
    pack = {
        "experiment_id": "case",
        "scenario_id": "T1",
        "scenario_name": "hpa_scale",
        "target_service": "cartservice",
        "candidates": [
            {
                "candidate": "cartservice",
                "service": "cartservice",
                "model_score": 1.0,
                "original_rank": 1,
                "metric_evidence": [{"id": "metric:cart", "type": "metric", "summary": "service metric stats"}],
                "topology_evidence": [],
                "trace_evidence": [],
                "log_evidence": [],
                "counter_evidence": [],
                "evidence_refs": [{"id": "metric:cart", "type": "metric", "summary": "service metric stats"}],
            },
            {
                "candidate": "checkoutservice",
                "service": "checkoutservice",
                "model_score": 0.5,
                "original_rank": 2,
                "metric_evidence": [{"id": "metric:checkout", "type": "metric", "summary": "service metric stats"}],
                "topology_evidence": [],
                "trace_evidence": [],
                "log_evidence": [],
                "counter_evidence": [],
                "evidence_refs": [{"id": "metric:checkout", "type": "metric", "summary": "service metric stats"}],
            },
        ],
    }
    response = {
        "final_ranking": [
            {
                "candidate": "checkoutservice",
                "final_score": 0.9,
                "confidence": 0.8,
                "reason": "higher latency",
                "supporting_evidence": ["metric:checkout"],
                "counter_evidence": [],
            },
            {
                "candidate": "cartservice",
                "final_score": 0.7,
                "confidence": 0.5,
                "reason": "some metric evidence",
                "supporting_evidence": ["metric:cart"],
                "counter_evidence": [],
            },
        ],
        "changed_order": True,
        "summary": "promote telemetry-only candidate",
    }

    result = guard_rerank_result(response, pack, use_target_prior=False)

    assert result["final_ranking"][0]["candidate"] == "cartservice"
    assert result["changed_order"] is False
    assert result["guard_applied"] is True


def test_guard_blocks_direct_t3_promotion_in_safe_mode():
    pack = _pack_t3()
    response = {
        "final_ranking": [
            {
                "candidate": "cartservice",
                "final_score": 0.9,
                "confidence": 0.8,
                "reason": "trace mismatch supports cartservice",
                "supporting_evidence": ["trace:1", "topo:1"],
                "counter_evidence": [],
            },
            {
                "candidate": "checkoutservice",
                "final_score": 0.7,
                "confidence": 0.5,
                "reason": "model top but no direct evidence",
                "supporting_evidence": [],
                "counter_evidence": ["counter:1"],
            },
        ],
        "changed_order": True,
        "summary": "trace mismatch supports promotion",
    }

    result = guard_rerank_result(response, pack, use_target_prior=False)

    assert result["final_ranking"][0]["candidate"] == "checkoutservice"
    assert result["changed_order"] is False
    assert result["promotion_allowed"] is False


def test_guard_allows_direct_t3_causal_promotion_when_enabled():
    pack = _pack_t3()
    response = {
        "final_ranking": [
            {
                "candidate": "cartservice",
                "final_score": 0.9,
                "confidence": 0.8,
                "reason": "trace mismatch supports cartservice",
                "supporting_evidence": ["trace:1", "topo:1"],
                "counter_evidence": [],
            },
            {
                "candidate": "checkoutservice",
                "final_score": 0.7,
                "confidence": 0.5,
                "reason": "model top but no direct evidence",
                "supporting_evidence": [],
                "counter_evidence": ["counter:1"],
            },
        ],
        "changed_order": True,
        "summary": "trace mismatch supports promotion",
    }

    result = guard_rerank_result(response, pack, allow_promotions=True, use_target_prior=False)

    assert result["final_ranking"][0]["candidate"] == "cartservice"
    assert result["changed_order"] is True


def test_guard_target_prior_promotes_target_service_inside_topk():
    pack = _pack_t3()
    response = {
        "final_ranking": [
            {
                "candidate": "checkoutservice",
                "final_score": 0.8,
                "confidence": 0.6,
                "reason": "model top",
                "supporting_evidence": [],
                "counter_evidence": ["counter:1"],
            },
            {
                "candidate": "cartservice",
                "final_score": 0.7,
                "confidence": 0.5,
                "reason": "target prior",
                "supporting_evidence": ["trace:1"],
                "counter_evidence": [],
            },
        ],
        "changed_order": False,
        "summary": "model order",
    }

    result = guard_rerank_result(response, pack)

    assert result["final_ranking"][0]["candidate"] == "cartservice"
    assert result["changed_order"] is True
    assert result["target_prior_used"] is True


def test_llm_reranker_rejects_invented_evidence_and_falls_back():
    pack = _pack_t3()

    result = rerank_pack_llm(
        pack,
        llm_client=lambda prompt: json.dumps({
            "final_ranking": [{
                "candidate": "checkoutservice",
                "final_score": 0.5,
                "confidence": 0.5,
                "reason": "invented",
                "supporting_evidence": ["not-real"],
                "counter_evidence": [],
            }],
            "changed_order": False,
            "summary": "invented",
        }),
    )

    assert result["source"] == "rule_reranker"
    assert result["fallback_used"] is True
    assert result["final_ranking"][0]["candidate"] == "cartservice"


def test_rule_reranker_boosts_t3_trace_mismatch():
    result = rerank_pack_rule(_pack_t3())

    assert result["final_ranking"][0]["candidate"] == "cartservice"
    assert "T3 trace mismatch" in result["explanation"]


def test_rule_reranker_boosts_t4_v2_abnormal_trace():
    result = rerank_pack_rule(_pack_t4())

    assert result["final_ranking"][0]["candidate"] == "checkoutservice:v2"
    assert "T4 v2" in result["explanation"]


def test_write_rerank_outputs_and_metric_comparison(tmpdir):
    out_dir = Path(str(tmpdir)) / "results"
    original = [
        {
            "experiment_id": "t3",
            "scenario_id": "T3",
            "target_service": "checkoutservice",
            "label_node_id": "cartservice",
            "ranked_node_ids": ["checkoutservice", "cartservice", "currencyservice"],
        }
    ]
    reranked = [{
        "experiment_id": "t3",
        "scenario_id": "T3",
        "final_ranking": [
            {
                "candidate": "cartservice",
                "final_score": 0.9,
                "confidence": 0.9,
                "reason": "trace mismatch",
                "supporting_evidence": ["trace:1"],
                "counter_evidence": [],
            },
            {
                "candidate": "checkoutservice",
                "final_score": 0.1,
                "confidence": 0.3,
                "reason": "less evidence",
                "supporting_evidence": [],
                "counter_evidence": [],
            },
        ],
        "changed_order": True,
        "summary": "trace mismatch",
        "confidence": 0.9,
        "source": "rule_reranker",
        "explanation": "trace mismatch",
        "evidence_refs": ["trace:1"],
    }]

    comparison = write_rerank_outputs(original, reranked, str(out_dir))

    assert comparison["before"]["top1"] == 0.0
    assert comparison["after"]["top1"] == 1.0
    assert (out_dir / "reranked_predictions.csv").exists()
    assert (out_dir / "rerank_explanations.jsonl").exists()
    assert (out_dir / "rerank_metrics_comparison.json").exists()


def test_write_rerank_outputs_appends_original_tail(tmpdir):
    out_dir = Path(str(tmpdir)) / "tail"
    original = [{
        "experiment_id": "x",
        "scenario_id": "T1",
        "target_service": "cartservice",
        "label_node_id": "currencyservice",
        "ranked_node_ids": ["cartservice", "checkoutservice", "currencyservice"],
    }]
    reranked = [{
        "experiment_id": "x",
        "final_ranking": [
            {"candidate": "cartservice", "final_score": 1.0, "confidence": 0.1, "reason": "", "supporting_evidence": [], "counter_evidence": []},
            {"candidate": "checkoutservice", "final_score": 0.5, "confidence": 0.1, "reason": "", "supporting_evidence": [], "counter_evidence": []},
        ],
        "changed_order": False,
        "summary": "",
        "confidence": 0.1,
        "source": "llm_reranker_safe",
    }]

    comparison = write_rerank_outputs(original, reranked, str(out_dir))
    rows = (out_dir / "reranked_predictions.csv").read_text(encoding="utf-8")

    assert "cartservice|checkoutservice|currencyservice" in rows
    assert comparison["after"]["mrr"] == comparison["before"]["mrr"]
