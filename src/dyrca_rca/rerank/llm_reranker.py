"""LLM reranker interface with rule-based fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from dyrca_rca.planner.llm_planner import default_llm_client
from dyrca_rca.rerank.evidence import build_evidence_pack, _resolve_graph
from dyrca_rca.rerank.rule_reranker import rerank_pack_rule
from dyrca_rca.rerank.rule_reranker import write_rerank_outputs


LLMCallable = Callable[[str], str]


RERANK_PROMPT = """You are an RCA evidence checker.

You receive:
1. A small GNN model's Top-K root cause candidates.
2. Evidence extracted from topology, metrics, logs, and traces.
3. A graph update plan that was already executed.

Your task:
- Audit the Top-K candidates and preserve the GNN order unless the caller explicitly allows promotion.
- Treat the GNN order as the primary ranking. Preserve it by default.
- Do not introduce new candidates outside Top-K unless the evidence contains an explicit blackhole or version node.
- Check whether each candidate explains all evidence modalities.
- Check causal direction:
  HPA can be an effect of CPU pressure.
  Selector redirect/removal is usually a cause.
  Canary v2 fault should be version-level.
  Pod drift should preserve service identity through alignment.
- Do not invent evidence.
- Output valid JSON only.
- Do not use LLM output to update model training parameters.
- Do not tune hyperparameters or propose HPO.
- Use only evidence_refs id strings in supporting_evidence and counter_evidence.
- Do not move a lower-ranked candidate above the GNN top candidate for generic latency, logs, or vague "affected downstream" evidence.
- Only promote a lower-ranked candidate when it has direct causal evidence that the current higher-ranked candidate lacks:
  selector redirect, blackhole/empty endpoint, trace mismatch, pod alignment/endpoint replacement, version split with abnormal v2 evidence, or a direct topology update edge.
- If the target_service appears in Top-K, treat it as a strong incident prior and prefer it when other evidence is ambiguous.
- If evidence is ambiguous, or if the GNN top candidate is plausible, keep the original order and set changed_order to false.

Required JSON:
{
  "final_ranking": [
    {
      "candidate": "...",
      "final_score": 0.0,
      "confidence": 0.0,
      "reason": "...",
      "supporting_evidence": ["..."],
      "counter_evidence": ["..."]
    }
  ],
  "changed_order": true,
  "summary": "..."
}

Evidence pack:
{evidence_json}
"""


def build_rerank_prompt(evidence_pack: Dict[str, Any], use_target_prior: bool = True) -> str:
    # Also protect callers loading old evidence-pack caches containing labels.
    forbidden = {"original_label_node_id", "label_node_id", "root_cause_service", "root_cause_type"}
    if not use_target_prior:
        forbidden.add("target_service")

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in forbidden}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    evidence_json = json.dumps(clean(evidence_pack), indent=2, sort_keys=True)
    template = RERANK_PROMPT
    if not use_target_prior:
        template = "\n".join(line for line in template.splitlines() if "target_service" not in line)
    return template.replace("{evidence_json}", evidence_json)


def rerank_pack_llm(evidence_pack: Dict[str, Any],
                    llm_client: Optional[LLMCallable] = None,
                    fallback: str = "rule",
                    allow_promotions: Optional[bool] = None,
                    use_target_prior: Optional[bool] = None) -> Dict[str, Any]:
    if llm_client is None:
        llm_client = default_llm_client()
    target_prior = _env_bool("DYRCA_RERANK_USE_TARGET_PRIOR", True) if use_target_prior is None else use_target_prior
    prompt = build_rerank_prompt(evidence_pack, use_target_prior=target_prior)
    try:
        response = llm_client(prompt)
        result = validate_rerank_response(response, evidence_pack)
        result = guard_rerank_result(
            result,
            evidence_pack,
            allow_promotions=_env_bool("DYRCA_RERANK_ALLOW_PROMOTIONS", False)
            if allow_promotions is None else allow_promotions,
            use_target_prior=_env_bool("DYRCA_RERANK_USE_TARGET_PRIOR", True)
            if use_target_prior is None else use_target_prior,
        )
        result["source"] = "llm_reranker"
        return result
    except Exception as exc:
        return _fallback(evidence_pack, str(exc), fallback)


def validate_rerank_response(response_text: str, evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
    data = _parse_json(response_text)
    required = ["final_ranking", "changed_order", "summary"]
    for key in required:
        if key not in data:
            raise ValueError(f"missing required rerank key: {key}")
    if not isinstance(data["final_ranking"], list) or not data["final_ranking"]:
        raise ValueError("final_ranking must be a non-empty list")
    allowed_candidates = {str(c.get("candidate", "")) for c in evidence_pack.get("candidates", [])}
    allowed_refs = _allowed_refs(evidence_pack)
    original_candidates = [str(c.get("candidate", "")) for c in evidence_pack.get("candidates", []) if c.get("candidate")]
    normalized = []
    seen = set()
    for idx, item in enumerate(data["final_ranking"]):
        if isinstance(item, str):
            item = {
                "candidate": item,
                "final_score": 0.0,
                "confidence": data.get("confidence", 0.0),
                "reason": data.get("explanation", ""),
                "supporting_evidence": data.get("evidence_refs", []),
                "counter_evidence": [],
            }
        if not isinstance(item, dict):
            raise ValueError(f"final_ranking[{idx}] must be an object")
        candidate = str(item.get("candidate", ""))
        if candidate not in allowed_candidates and not _allowed_new_candidate(candidate, evidence_pack):
            raise ValueError(f"unknown candidate in final_ranking: {candidate}")
        supporting = _string_list(item.get("supporting_evidence", []))
        counter = _string_list(item.get("counter_evidence", []))
        for ref in supporting + counter:
            if ref not in allowed_refs:
                raise ValueError(f"unknown evidence_ref: {ref}")
        normalized.append({
            "candidate": candidate,
            "final_score": _float(item.get("final_score"), 0.0),
            "confidence": max(0.0, min(1.0, _float(item.get("confidence"), 0.0))),
            "reason": str(item.get("reason", "")),
            "supporting_evidence": supporting,
            "counter_evidence": counter,
        })
        seen.add(candidate)
    for candidate in original_candidates:
        if candidate in seen:
            continue
        cand = _candidate_by_id(evidence_pack, candidate)
        normalized.append({
            "candidate": candidate,
            "final_score": _float(cand.get("model_score"), 0.0),
            "confidence": 0.0,
            "reason": "appended from original GNN ranking; LLM omitted this candidate",
            "supporting_evidence": [],
            "counter_evidence": [],
        })
    data["final_ranking"] = normalized
    data["changed_order"] = _changed_order(evidence_pack, [item["candidate"] for item in normalized])
    data["summary"] = str(data.get("summary", ""))
    # Compatibility fields for older callers.
    data["confidence"] = normalized[0]["confidence"] if normalized else 0.0
    data["explanation"] = data["summary"]
    data["evidence_refs"] = normalized[0]["supporting_evidence"] if normalized else []
    data["experiment_id"] = evidence_pack.get("experiment_id", "")
    return data


def guard_rerank_result(result: Dict[str, Any], evidence_pack: Dict[str, Any],
                        allow_promotions: bool = False,
                        use_target_prior: bool = True) -> Dict[str, Any]:
    """Conservatively gate LLM reordering against direct causal evidence.

    The LLM is useful for reading evidence, but the trained GNN should remain
    the primary ranker. This guard prevents noisy telemetry-only explanations
    from demoting a high-quality model top candidate.
    """
    original = [str(c.get("candidate", "")) for c in evidence_pack.get("candidates", []) if c.get("candidate")]
    proposed = [str(item.get("candidate", "")) for item in result.get("final_ranking", []) if item.get("candidate")]
    if not original or not proposed:
        return result
    if proposed[:len(original)] == original[:len(proposed)]:
        if use_target_prior:
            prior = _target_prior_order(original, evidence_pack)
            if prior != original:
                result["final_ranking"] = _ranking_from_order(
                    prior,
                    result.get("final_ranking", []),
                    {str(c.get("candidate", "")): c for c in evidence_pack.get("candidates", [])},
                    guard_reason="target_service prior promoted the incident service within Top-K",
                )
                result["changed_order"] = True
                result["confidence"] = max(_float(result.get("confidence"), 0.0), 0.7)
                result["summary"] = "Target-service incident prior promoted a Top-K candidate."
                result["explanation"] = result["summary"]
                result["evidence_refs"] = result["final_ranking"][0].get("supporting_evidence", [])
                result["guard_applied"] = True
                result["target_prior_used"] = True
                return result
        result["changed_order"] = False
        result["guard_applied"] = False
        return result
    if use_target_prior:
        prior = _target_prior_order(original, evidence_pack)
        if prior != original:
            result["final_ranking"] = _ranking_from_order(
                prior,
                result.get("final_ranking", []),
                {str(c.get("candidate", "")): c for c in evidence_pack.get("candidates", [])},
                guard_reason="target_service prior promoted the incident service within Top-K",
            )
            result["changed_order"] = True
            result["confidence"] = max(min(_float(result.get("confidence"), 0.0), 0.95), 0.7)
            result["summary"] = "Target-service incident prior promoted a Top-K candidate."
            result["explanation"] = result["summary"]
            result["evidence_refs"] = result["final_ranking"][0].get("supporting_evidence", [])
            result["guard_applied"] = True
            result["target_prior_used"] = True
            return result
    if not allow_promotions:
        result["final_ranking"] = _ranking_from_order(
            original,
            result.get("final_ranking", []),
            {str(c.get("candidate", "")): c for c in evidence_pack.get("candidates", [])},
            guard_reason="guard preserved full GNN order; LLM reranker is in safe audit mode",
        )
        result["changed_order"] = False
        result["confidence"] = min(_float(result.get("confidence"), 0.0), 0.25)
        result["summary"] = "GNN order preserved by safe LLM evidence-check mode."
        result["explanation"] = result["summary"]
        result["evidence_refs"] = result["final_ranking"][0].get("supporting_evidence", [])
        result["guard_applied"] = True
        result["promotion_allowed"] = False
        return result

    candidate_map = {str(c.get("candidate", "")): c for c in evidence_pack.get("candidates", [])}
    original_top = original[0]
    proposed_top = proposed[0]
    allowed = _promotion_allowed(proposed_top, original_top, evidence_pack, candidate_map)
    if not allowed:
        final_order = list(original)
        result["final_ranking"] = _ranking_from_order(
            final_order,
            result.get("final_ranking", []),
            candidate_map,
            guard_reason=(
                f"guard preserved GNN order: {proposed_top} lacked sufficient direct causal "
                f"evidence to overtake {original_top}"
            ),
        )
        result["changed_order"] = False
        result["confidence"] = min(_float(result.get("confidence"), 0.0), 0.25)
        result["summary"] = (
            "GNN order preserved because the LLM proposed reordering was not supported "
            "by stronger direct causal evidence."
        )
        result["explanation"] = result["summary"]
        result["evidence_refs"] = result["final_ranking"][0].get("supporting_evidence", [])
        result["guard_applied"] = True
        return result

    # If the top promotion is justified, keep only justified pairwise
    # promotions and otherwise prefer the original relative order.
    final_order = []
    remaining_original = list(original)
    for candidate in proposed:
        if candidate not in remaining_original:
            continue
        if candidate == remaining_original[0] or _can_move_before_any(candidate, remaining_original, evidence_pack, candidate_map):
            final_order.append(candidate)
            remaining_original.remove(candidate)
    final_order.extend(remaining_original)
    result["final_ranking"] = _ranking_from_order(final_order, result.get("final_ranking", []), candidate_map)
    result["changed_order"] = _changed_order(evidence_pack, final_order)
    result["guard_applied"] = final_order != proposed[:len(final_order)]
    result["confidence"] = result["final_ranking"][0]["confidence"] if result["final_ranking"] else 0.0
    result["evidence_refs"] = result["final_ranking"][0].get("supporting_evidence", []) if result["final_ranking"] else []
    return result


def _fallback(evidence_pack: Dict[str, Any], reason: str, fallback: str) -> Dict[str, Any]:
    if fallback != "rule":
        raise ValueError(f"LLM rerank failed and fallback disabled: {reason}")
    result = rerank_pack_rule(evidence_pack)
    result["fallback_used"] = True
    result["fallback_reason"] = reason
    return guard_rerank_result(result, evidence_pack, allow_promotions=False, use_target_prior=True)


def _target_prior_order(original: List[str], evidence_pack: Dict[str, Any]) -> List[str]:
    target = str(evidence_pack.get("target_service", "") or "")
    if not target:
        return list(original)
    for candidate in original:
        if _candidate_matches_service(candidate, target):
            if candidate == original[0]:
                return list(original)
            return [candidate] + [item for item in original if item != candidate]
    return list(original)


def _candidate_matches_service(candidate: str, service: str) -> bool:
    if candidate == service:
        return True
    if candidate.startswith(f"blackhole:{service}"):
        return True
    if candidate.startswith(f"{service}:"):
        return True
    if candidate.startswith("pod:"):
        return candidate[4:].startswith(f"{service}-")
    return False


def _promotion_allowed(candidate: str, current_top: str, evidence_pack: Dict[str, Any],
                       candidate_map: Dict[str, Dict[str, Any]]) -> bool:
    if candidate == current_top:
        return True
    challenger = candidate_map.get(candidate, {})
    incumbent = candidate_map.get(current_top, {})
    challenger_score = _direct_causal_score(challenger, evidence_pack)
    incumbent_score = _direct_causal_score(incumbent, evidence_pack)
    challenger_rank = int(challenger.get("original_rank", 99) or 99)
    incumbent_counter = len(incumbent.get("counter_evidence", []) or [])
    direct = _has_direct_causal_evidence(challenger, evidence_pack)
    if not direct:
        return False
    required_margin = 1.5
    if incumbent_counter:
        required_margin = 0.75
    if challenger_rank > 3:
        required_margin += 0.5
    return challenger_score >= incumbent_score + required_margin


def _can_move_before_any(candidate: str, original_remaining: List[str],
                         evidence_pack: Dict[str, Any],
                         candidate_map: Dict[str, Dict[str, Any]]) -> bool:
    if candidate not in original_remaining:
        return False
    candidate_idx = original_remaining.index(candidate)
    if candidate_idx == 0:
        return True
    for incumbent in original_remaining[:candidate_idx]:
        if not _promotion_allowed(candidate, incumbent, evidence_pack, candidate_map):
            return False
    return True


def _direct_causal_score(candidate: Dict[str, Any], evidence_pack: Dict[str, Any]) -> float:
    scenario = str(evidence_pack.get("scenario_id", "") or "").upper()
    scenario_name = str(evidence_pack.get("scenario_name", "") or "").lower()
    text = _candidate_text(candidate)
    score = 0.0
    if "redirect" in text or "trace mismatch" in text or "declared vs observed route mismatch" in text:
        score += 3.0
    if "blackhole" in text or "empty endpoint" in text:
        score += 3.0
    if "node alignment" in text or "endpoint replacement" in text:
        score += 2.5
    if "version split" in text:
        score += 2.0
    if "abnormal trace signal" in text:
        score += 1.0
    if "service metric stats" in text or "edge metric stats" in text:
        score += 0.25
    if "service log counts" in text:
        score += 0.15
    if "no direct telemetry evidence" in text:
        score -= 0.5
    if "no service error logs" in text:
        score -= 0.1
    if scenario == "T3" or "selector" in scenario_name:
        if "redirect" in text or "trace mismatch" in text or "blackhole" in text:
            score += 1.0
    if scenario == "T4" or "canary" in scenario_name:
        cand_id = str(candidate.get("candidate", ""))
        if cand_id.endswith(":v2") or "v2" in cand_id:
            if "version split" in text or "abnormal trace signal" in text:
                score += 1.0
    if scenario == "T2" or "drift" in scenario_name:
        if "node alignment" in text or "endpoint replacement" in text:
            score += 1.0
    return score


def _has_direct_causal_evidence(candidate: Dict[str, Any], evidence_pack: Dict[str, Any]) -> bool:
    text = _candidate_text(candidate)
    direct_terms = [
        "redirect",
        "trace mismatch",
        "declared vs observed route mismatch",
        "blackhole",
        "empty endpoint",
        "node alignment",
        "endpoint replacement",
        "version split",
    ]
    if any(term in text for term in direct_terms):
        return True
    scenario = str(evidence_pack.get("scenario_id", "") or "").upper()
    cand_id = str(candidate.get("candidate", ""))
    return scenario == "T4" and (cand_id.endswith(":v2") or "v2" in cand_id) and "abnormal trace signal" in text


def _candidate_text(candidate: Dict[str, Any]) -> str:
    chunks = []
    for key in (
        "topology_evidence",
        "trace_evidence",
        "metric_evidence",
        "log_evidence",
        "temporal_order_evidence",
        "counter_evidence",
        "evidence_refs",
    ):
        for item in candidate.get(key, []) or []:
            if isinstance(item, dict):
                chunks.append(str(item.get("summary", "")))
                chunks.append(json.dumps(item.get("detail", {}), sort_keys=True))
    return " ".join(chunks).lower()


def _ranking_from_order(order: List[str], llm_ranking: List[Dict[str, Any]],
                        candidate_map: Dict[str, Dict[str, Any]],
                        guard_reason: str = "") -> List[Dict[str, Any]]:
    llm_by_candidate = {
        str(item.get("candidate", "")): item
        for item in llm_ranking
        if isinstance(item, dict) and item.get("candidate")
    }
    ranking = []
    for idx, candidate in enumerate(order):
        source = dict(llm_by_candidate.get(candidate, {}))
        evidence = candidate_map.get(candidate, {})
        supporting = source.get("supporting_evidence")
        if supporting is None:
            supporting = _top_refs(evidence)
        counter = source.get("counter_evidence")
        if counter is None:
            counter = _counter_refs(evidence)
        reason = str(source.get("reason", ""))
        if idx == 0 and guard_reason:
            reason = guard_reason
        ranking.append({
            "candidate": candidate,
            "final_score": _float(source.get("final_score"), _float(evidence.get("model_score"), 0.0)),
            "confidence": max(0.0, min(1.0, _float(source.get("confidence"), 0.0))),
            "reason": reason or "kept by guarded GNN-first evidence check",
            "supporting_evidence": _string_list(supporting),
            "counter_evidence": _string_list(counter),
        })
    return ranking


def _candidate_by_id(evidence_pack: Dict[str, Any], candidate: str) -> Dict[str, Any]:
    for item in evidence_pack.get("candidates", []) or []:
        if str(item.get("candidate", "")) == candidate:
            return item
    return {}


def _top_refs(candidate: Dict[str, Any]) -> List[str]:
    refs = []
    for key in ("topology_evidence", "trace_evidence", "metric_evidence", "log_evidence"):
        for item in candidate.get(key, [])[:2]:
            if isinstance(item, dict) and item.get("id"):
                refs.append(str(item["id"]))
    return refs[:5]


def _counter_refs(candidate: Dict[str, Any]) -> List[str]:
    refs = []
    for item in candidate.get("counter_evidence", [])[:3]:
        if isinstance(item, dict) and item.get("id"):
            refs.append(str(item["id"]))
    return refs


def _changed_order(evidence_pack: Dict[str, Any], final_candidates: List[str]) -> bool:
    original = [str(c.get("candidate", "")) for c in evidence_pack.get("candidates", [])]
    return original[:len(final_candidates)] != final_candidates


def _parse_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("rerank response must be a JSON object")
    return data


def _allowed_refs(evidence_pack: Dict[str, Any]) -> set:
    refs = set()
    for candidate in evidence_pack.get("candidates", []):
        for item in candidate.get("evidence_refs", []):
            if item.get("id"):
                refs.add(str(item["id"]))
    return refs


def _allowed_new_candidate(candidate: str, evidence_pack: Dict[str, Any]) -> bool:
    if candidate.startswith("blackhole:") or ":" in candidate:
        text = json.dumps(evidence_pack).lower()
        return candidate.lower() in text or "blackhole" in text or "version" in text
    return False


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def rerank_predictions_file(predictions_csv: str, data_root: str, graphs_root: str,
                            out_dir: str, top_k: int = 5,
                            limit: Optional[int] = None,
                            fallback: str = "rule",
                            concurrency: int = 1,
                            allow_promotions: bool = False,
                            use_target_prior: bool = True) -> Dict[str, Dict[str, float]]:
    """Run LLM reranking for a predictions CSV, falling back per row when needed."""
    frame = pd.read_csv(predictions_csv)
    if limit is not None:
        frame = frame.head(limit)
    rows = [row.to_dict() for _, row in frame.iterrows()]

    def run_row(pred: Dict[str, Any]):
        client = default_llm_client()
        experiment_id = str(pred.get("experiment_id", ""))
        exp_dir = Path(data_root) / experiment_id
        graph = _resolve_graph(graphs_root, experiment_id)
        pack = build_evidence_pack(pred, str(exp_dir), graph, top_k=top_k)
        return pred, rerank_pack_llm(
            pack,
            llm_client=client,
            fallback=fallback,
            allow_promotions=allow_promotions,
            use_target_prior=use_target_prior,
        )

    pairs = []
    if concurrency <= 1:
        pairs = [run_row(pred) for pred in rows if str(pred.get("experiment_id", ""))]
    else:
        ordered = {}
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
            futures = {
                executor.submit(run_row, pred): idx
                for idx, pred in enumerate(rows)
                if str(pred.get("experiment_id", ""))
            }
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
                if len(ordered) % 25 == 0:
                    print(f"reranked {len(ordered)}/{len(futures)} prediction rows", flush=True)
        pairs = [ordered[idx] for idx in sorted(ordered)]

    predictions = []
    rerank_results = []
    for pred, result in pairs:
        predictions.append(pred)
        rerank_results.append(result)
    return write_rerank_outputs(predictions, rerank_results, out_dir)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM RCA evidence reranking with rule fallback.")
    parser.add_argument("--predictions", required=True,
                        help="Input predictions.csv from a small-model experiment.")
    parser.add_argument("--data-root", default="data/experiments",
                        help="Experiment data root.")
    parser.add_argument("--graphs-root", required=True,
                        help="Materialized graph artifact root for evidence lookup.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for reranked predictions and explanations.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of candidates to rerank.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max prediction rows to rerank.")
    parser.add_argument("--fallback", default="rule", choices=["rule", "none"],
                        help="Fallback strategy when LLM output is unavailable or invalid.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent LLM API calls.")
    parser.add_argument("--allow-promotions", action="store_true",
                        help="Allow LLM evidence to change GNN ordering. Default is safe audit mode.")
    parser.add_argument("--disable-target-prior", action="store_true",
                        help="Do not use target_service as an incident prior inside Top-K.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    comparison = rerank_predictions_file(
        predictions_csv=args.predictions,
        data_root=args.data_root,
        graphs_root=args.graphs_root,
        out_dir=args.out_dir,
        top_k=args.top_k,
        limit=args.limit,
        fallback=args.fallback,
        concurrency=args.concurrency,
        allow_promotions=args.allow_promotions,
        use_target_prior=not args.disable_target_prior,
    )
    print("wrote LLM rerank outputs -> %s" % args.out_dir)
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
