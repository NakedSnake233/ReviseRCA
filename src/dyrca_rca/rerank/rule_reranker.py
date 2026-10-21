"""Deterministic evidence-based reranker for RCA candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from dyrca_rca.eval.metrics import compute_ranking_metrics
from dyrca_rca.rerank.evidence import build_evidence_pack, _resolve_graph


def rerank_pack_rule(evidence_pack: Dict[str, Any]) -> Dict[str, Any]:
    scenario = str(evidence_pack.get("scenario_id", "") or "").upper()
    scenario_name = str(evidence_pack.get("scenario_name", "") or "").lower()
    scored = []
    for cand in evidence_pack.get("candidates", []):
        score = float(cand.get("model_score", 0.0) or 0.0)
        reasons = []
        trace_text = _evidence_text(cand.get("trace_evidence", []))
        topo_text = _evidence_text(cand.get("topology_evidence", []))

        if scenario == "T3" or "selector" in scenario_name:
            if "trace mismatch" in trace_text or "redirect" in topo_text:
                score += 1.0
                reasons.append("boosted T3 trace mismatch/redirect evidence")
            if "blackhole" in topo_text:
                score += 0.8
                reasons.append("boosted T3 empty-endpoint blackhole evidence")
        if scenario == "T4" or "canary" in scenario_name:
            candidate = str(cand.get("candidate", ""))
            if candidate.endswith(":v2") or "v2" in candidate:
                if "abnormal trace" in trace_text or "version split" in topo_text:
                    score += 1.0
                    reasons.append("boosted T4 v2 abnormal/version evidence")
        if scenario == "T1" or "hpa" in scenario_name or "scale" in scenario_name:
            if "effect_or_mixed" in topo_text:
                score += 0.2
                reasons.append("kept HPA scale candidate because causal_role is effect_or_mixed")
            elif "replica" in topo_text or "node changed" in topo_text:
                score += 0.3
                reasons.append("boosted HPA topology change without over-penalty")

        counter_count = len(cand.get("counter_evidence", []))
        score -= min(0.2, counter_count * 0.05)
        scored.append({
            "candidate": cand.get("candidate", ""),
            "service": cand.get("service", ""),
            "score": score,
            "reasons": reasons,
            "evidence_refs": _top_refs(cand),
            "counter_refs": _counter_refs(cand),
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    confidence = 0.0
    if scored:
        if len(scored) == 1:
            confidence = 1.0
        else:
            confidence = max(0.0, min(1.0, scored[0]["score"] - scored[1]["score"]))
    return {
        "experiment_id": evidence_pack.get("experiment_id", ""),
        "final_ranking": [
            {
                "candidate": item["candidate"],
                "final_score": item["score"],
                "confidence": confidence if idx == 0 else max(0.0, min(1.0, item["score"])),
                "reason": "; ".join(item["reasons"]) if item["reasons"] else "kept by model order and evidence checks",
                "supporting_evidence": item["evidence_refs"],
                "counter_evidence": item["counter_refs"],
            }
            for idx, item in enumerate(scored)
        ],
        "changed_order": _changed_order(evidence_pack, [item["candidate"] for item in scored]),
        "summary": "; ".join(scored[0]["reasons"]) if scored and scored[0]["reasons"] else "rule reranker used model order with evidence checks",
        "confidence": confidence,
        "explanation": "; ".join(scored[0]["reasons"]) if scored and scored[0]["reasons"] else "rule reranker used model order with evidence checks",
        "evidence_refs": scored[0]["evidence_refs"] if scored else [],
        "source": "rule_reranker",
    }


def write_rerank_outputs(original_predictions: List[Dict[str, Any]],
                         rerank_results: List[Dict[str, Any]],
                         out_dir: str) -> Dict[str, Dict[str, float]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    explanations_path = out / "rerank_explanations.jsonl"
    with explanations_path.open("w", encoding="utf-8") as f:
        for pred, result in zip(original_predictions, rerank_results):
            final = _merge_with_original_ranking(
                _ranking_candidates(result.get("final_ranking", [])),
                _original_ranking(pred),
            )
            label = pred.get("label_node_id", "")
            rank = None
            for idx, node_id in enumerate(final, start=1):
                if node_id == label:
                    rank = idx
                    break
            rows.append({
                "experiment_id": pred.get("experiment_id", ""),
                "scenario_id": pred.get("scenario_id", ""),
                "target_service": pred.get("target_service", ""),
                "label_node_id": label,
                "rank": rank,
                "ranked_node_ids": "|".join(final),
                "confidence": result.get("confidence", 0.0),
                "source": result.get("source", ""),
            })
            f.write(json.dumps(result, sort_keys=True) + "\n")
    pd.DataFrame(rows).to_csv(out / "reranked_predictions.csv", index=False)
    before = compute_ranking_metrics(original_predictions)
    after = compute_ranking_metrics(rows)
    comparison = {"before": before, "after": after}
    with (out / "rerank_metrics_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)
    return comparison


def rerank_predictions_file(predictions_csv: str, data_root: str, graphs_root: str,
                            out_dir: str, top_k: int = 5,
                            limit: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Run deterministic reranking for a predictions CSV."""
    frame = pd.read_csv(predictions_csv)
    if limit is not None:
        frame = frame.head(limit)
    predictions = []
    rerank_results = []
    for _, row in frame.iterrows():
        pred = row.to_dict()
        experiment_id = str(pred.get("experiment_id", ""))
        if not experiment_id:
            continue
        exp_dir = Path(data_root) / experiment_id
        graph = _resolve_graph(graphs_root, experiment_id)
        pack = build_evidence_pack(pred, str(exp_dir), graph, top_k=top_k)
        predictions.append(pred)
        rerank_results.append(rerank_pack_rule(pack))
    return write_rerank_outputs(predictions, rerank_results, out_dir)


def _evidence_text(items: Iterable[Dict[str, Any]]) -> str:
    return " ".join(str(item.get("summary", "")).lower() for item in items)


def _top_refs(candidate: Dict[str, Any]) -> List[str]:
    refs = []
    for key in ("topology_evidence", "trace_evidence", "metric_evidence", "log_evidence"):
        for item in candidate.get(key, [])[:2]:
            if item.get("id"):
                refs.append(item["id"])
    return refs[:5]


def _counter_refs(candidate: Dict[str, Any]) -> List[str]:
    refs = []
    for item in candidate.get("counter_evidence", [])[:3]:
        if item.get("id"):
            refs.append(item["id"])
    return refs


def _ranking_candidates(final_ranking: List[Any]) -> List[str]:
    result = []
    for item in final_ranking:
        if isinstance(item, dict):
            result.append(str(item.get("candidate", "")))
        else:
            result.append(str(item))
    return [item for item in result if item]


def _original_ranking(prediction: Dict[str, Any]) -> List[str]:
    value = prediction.get("ranked_node_ids") or prediction.get("ranked_node_ids_str") or ""
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [token for token in str(value).split("|") if token]


def _merge_with_original_ranking(reranked: List[str], original: List[str]) -> List[str]:
    merged = []
    seen = set()
    for candidate in list(reranked) + list(original):
        if not candidate or candidate in seen:
            continue
        merged.append(candidate)
        seen.add(candidate)
    return merged


def _changed_order(evidence_pack: Dict[str, Any], final_candidates: List[str]) -> bool:
    original = [str(c.get("candidate", "")) for c in evidence_pack.get("candidates", [])]
    return original[:len(final_candidates)] != final_candidates


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule-based RCA evidence reranking.")
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
    )
    print("wrote rerank outputs -> %s" % args.out_dir)
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
