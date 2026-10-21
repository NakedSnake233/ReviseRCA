"""Free-form LLM diagnosis ablation for rebuttal (addresses R2-C4).

Reviewer 2 asked for an ablation showing "the same LLM, given the same evidence
package, directly performing a ranked root cause without the patch/execution
layer." This script implements exactly that:

  - Reuses the SAME compact evidence package E that the Plan2RCA planner consumes
    (dyrca_rca.planner.llm_planner.build_evidence_json).
  - Provides the SAME candidate service set the GNN ranks over.
  - Asks the LLM to output a JSON ranked list of root-cause candidates directly
    (free-form diagnosis), with NO graph patch / executor / GNN in the loop.
  - Scores the LLM ranking against ground truth with the SAME metrics.

The result isolates the value of Plan2RCA's constrained patch architecture from
the value of using an LLM at all. If free-form diagnosis is competitive, the
constrained design's auditability is still a benefit but the accuracy benefit
shrinks; if free-form is clearly worse, the constrained architecture is
vindicated.

NOTE: requires a working LLM endpoint. The repository LLM client reads
DYRCA_LLM_API_KEY / DYRCA_LLM_BASE_URL / DYRCA_LLM_MODEL (see
ResponsesAPIClient.from_env). The cached plans were generated with a previous
key; to keep the ablation fair, set DYRCA_LLM_MODEL to the SAME model the
planner used.

Usage:
    DYRCA_LLM_API_KEY=... DYRCA_LLM_BASE_URL=... DYRCA_LLM_MODEL=gpt-5 \
    python scripts/run_rebuttal_freeform_llm.py \
        --index artifacts/dygraph_rca_full/index.csv \
        --data-root data/experiments \
        --out-dir artifacts/rebuttal_freeform/test \
        --split-seed 42 --top-k-eval 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from dyrca_rca.dataset.materialize import build_graph_from_topology, read_index
from dyrca_rca.graph.weights import apply_edge_weights
from dyrca_rca.graph.executor import normalize_graph
from dyrca_rca.planner.llm_planner import build_evidence_json
from dyrca_rca.planner.signals import read_json
from dyrca_rca.experiments.splits import make_split
from dyrca_rca.eval.metrics import compute_ranking_metrics


FREEFORM_SYSTEM = (
    "You are a microservice root-cause analyst. You are given compact observability "
    "evidence (topology change, endpoint/pod/deployment diffs, trace route summary, "
    "metric anomalies) for ONE incident, plus the list of candidate services. "
    "Diagnose the root cause directly from the evidence and return a STRICT JSON "
    "object only: {\"ranking\": [{\"service\": <str>, \"reason\": <str>}, ...]} where "
    "services are ordered from most to least likely root cause. Every candidate must "
    "appear exactly once. Do not output any text outside the JSON."
)


class OpenAIChatClient:
    """Minimal OpenAI Chat Completions client (works with OpenAI-compatible proxies)."""
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90.0,
                 max_tokens: int = 1024, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": FREEFORM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        last = None
        for attempt in range(5):
            try:
                with httpx.Client(timeout=self.timeout) as c:
                    r = c.post(url, headers=headers, json=payload)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                    last = f"HTTP {r.status_code}: {r.text[:120]}"
                    import time as _t
                    _t.sleep(min(2.0 * (2 ** attempt), 30.0))
                    continue
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # retry on transient errors
                last = str(exc)[:300]
                if attempt == 4:
                    break
                import time as _t
                _t.sleep(min(2.0 * (2 ** attempt), 30.0))
        raise RuntimeError(f"LLM call failed after retries: {last}")


def candidate_services(exp_dir: Path) -> List[str]:
    """Candidate service set = service nodes in the during snapshot (same as GNN mask)."""
    during_path = exp_dir / "topology" / "during.json"
    if not during_path.exists():
        pre = read_json(exp_dir / "topology" / "pre.json", required=True)
        g = apply_edge_weights(str(exp_dir), build_graph_from_topology(pre, phase="pre"))
    else:
        during = read_json(during_path, required=True)
        g = apply_edge_weights(str(exp_dir), build_graph_from_topology(during, phase="during"))
    norm = normalize_graph(g)
    return sorted({n["id"] for n in norm["nodes"]
                   if (n.get("attrs") or {}).get("type") == "service"})


def build_freeform_prompt(evidence: Dict[str, Any], candidates: List[str]) -> str:
    return (
        f"Candidate services: {json.dumps(candidates)}\n\n"
        f"Compact evidence JSON:\n{json.dumps(evidence)}\n\n"
        f"Return the JSON ranking now."
    )


def parse_ranking(text: str, candidates: List[str]) -> List[str]:
    """Extract the LLM ranking, validate against the candidate set, append any missing."""
    cand_set = set(candidates)
    # grab the first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return list(candidates)
    try:
        obj = json.loads(m.group(0))
    except Exception:
        # tolerant fallback: pull service-like tokens in order
        order = re.findall(r'"service"\s*:\s*"([^"]+)"', text)
        ranked = [s for s in order if s in cand_set]
        ranked += [s for s in candidates if s not in set(ranked)]
        return ranked
    items = obj.get("ranking", []) if isinstance(obj, dict) else []
    ranked: List[str] = []
    seen = set()
    for it in items:
        if isinstance(it, dict):
            s = str(it.get("service", ""))
        else:
            s = str(it)
        if s in cand_set and s not in seen:
            ranked.append(s); seen.add(s)
    for s in candidates:  # append any missing candidates (LLM omitted them)
        if s not in seen:
            ranked.append(s); seen.add(s)
    return ranked


def run_freeform_on_frame(frame: pd.DataFrame, data_root: str, client,
                          top_k_eval: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = frame.to_dict("records")
    if limit:
        rows = rows[:limit]
    preds: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        err = row.get("error_message")
        if isinstance(err, str) and err.strip():
            continue
        dp = row.get("data_path")
        if dp is None or (isinstance(dp, float) and pd.isna(dp)):
            continue
        exp_dir = Path(str(dp))
        if not exp_dir.exists():
            exp_dir = Path(data_root) / str(row.get("experiment_id", ""))
        try:
            evidence = build_evidence_json(str(exp_dir))
            candidates = candidate_services(exp_dir)
            if not candidates:
                continue
            prompt = build_freeform_prompt(evidence, candidates)
            text = client(prompt)
            ranked = parse_ranking(text, candidates)
        except Exception as exc:
            # conservative fallback: original candidate order (no diagnostic skill)
            ranked = candidate_services(exp_dir) if exp_dir.exists() else []
            preds.append({
                "experiment_id": str(row.get("experiment_id", "")),
                "scenario_id": str(row.get("scenario_id", "")),
                "target_service": str(row.get("target_service", "")),
                "label_node_id": str(row.get("root_cause_service", "")),
                "ranked_node_ids": ranked, "rank": None, "fallback": True,
                "error": str(exc)[:200],
            })
            continue
        label = str(row.get("root_cause_service", ""))
        rank = None
        for idx, s in enumerate(ranked, 1):
            if s == label:
                rank = idx; break
        preds.append({
            "experiment_id": str(row.get("experiment_id", "")),
            "scenario_id": str(row.get("scenario_id", "")),
            "target_service": str(row.get("target_service", "")),
            "label_node_id": label,
            "ranked_node_ids": ranked,
            "rank": rank,
            "fallback": False,
        })
        if (i + 1) % 10 == 0:
            print(f"  [freeform] {i+1}/{len(rows)} processed", flush=True)
    return preds


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--data-root", default="data/experiments")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--top-k-eval", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--use-test-split", action="store_true", default=True,
                   help="evaluate on the same test split as the main pipeline (default)")
    p.add_argument("--base-url", default=os.environ.get("DYRCA_LLM_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("DYRCA_LLM_API_KEY", ""))
    p.add_argument("--model", default=os.environ.get("DYRCA_LLM_MODEL", "gpt-5.4-mini"))
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args(argv)

    if not (args.base_url and args.api_key):
        print("[freeform] Need --base-url and --api-key (or DYRCA_LLM_BASE_URL / DYRCA_LLM_API_KEY).")
        return 2
    client = OpenAIChatClient(base_url=args.base_url, api_key=args.api_key, model=args.model,
                              max_tokens=args.max_tokens, temperature=args.temperature)
    print(f"[freeform] client: model={args.model} base={args.base_url}", flush=True)
    # sanity ping
    try:
        reply = client("Reply with exactly: OK")
        print(f"[freeform] sanity ping OK: {reply[:40]!r}", flush=True)
    except Exception as exc:
        print(f"[freeform] LLM client unavailable: {exc}")
        return 2

    frame = read_index(args.index)
    splits = make_split(frame, split="random", seed=args.split_seed)
    test_frame = splits["test"]
    print(f"[freeform] test set size = {len(test_frame)} (seed {args.split_seed})", flush=True)

    preds = run_freeform_on_frame(test_frame, args.data_root, client,
                                  args.top_k_eval, args.limit)
    metrics = compute_ranking_metrics(preds)
    n_fallback = sum(1 for p in preds if p.get("fallback"))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / "metrics_overall.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "ablation": "free_form_llm_diagnosis",
        "split_seed": args.split_seed,
        "n_test": len(preds), "n_fallback": n_fallback,
        "top_k_eval": args.top_k_eval,
        "note": "Same evidence + candidate set as Plan2RCA; no patch/executor/GNN.",
    }, indent=2), encoding="utf-8")
    print(f"[freeform] metrics: {json.dumps({k: round(v,4) for k,v in metrics.items()})}", flush=True)
    print(f"[freeform] fallbacks: {n_fallback}/{len(preds)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
