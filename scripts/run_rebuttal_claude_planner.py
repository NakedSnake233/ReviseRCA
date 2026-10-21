"""Generate Claude-based graph-patch plans for all experiments (rebuttal, base LLM = Claude).

Uses the dyrca planner's ResponsesAPIClient (the /v1/responses proxy endpoint works
with claude-3-5-haiku-20241022) but wraps it with retry+backoff for 429/503, which the
proxy returns under load. Low concurrency to respect rate limits.

Reuses all planner logic (prompt, schema validation, rule fallback) via
plan_experiment_with_llm; only the LLM client is swapped.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from dyrca_rca.planner.llm_planner import ResponsesAPIClient, plan_experiment_with_llm
from dyrca_rca.dataset.materialize import read_index
from dyrca_rca.planner.signals import read_json


class RetryingClient:
    """Wrap a callable LLM client with exponential backoff on transient errors."""
    def __init__(self, inner, max_attempts: int = 5, base_sleep: float = 2.0):
        self.inner = inner
        self.max_attempts = max_attempts
        self.base_sleep = base_sleep

    def __call__(self, prompt: str) -> str:
        last = None
        for attempt in range(self.max_attempts):
            try:
                return self.inner(prompt)
            except Exception as exc:
                last = exc
                msg = str(exc)
                # retry on 429 / 503 / network; give up on 4xx auth/validation
                if any(s in msg for s in ("429", "503", "502", "504", "Timeout", "timed out", "RemoteProtocol")):
                    sleep_s = self.base_sleep * (2 ** attempt) + 0.5
                    time.sleep(min(sleep_s, 30.0))
                    continue
                raise
        raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts: {last}")


def is_fallback_plan(path: Path) -> bool:
    try:
        d = json.load(open(path))
    except Exception:
        return True
    for e in d.get("evidence", []):
        if isinstance(e, dict) and e.get("fallback_used"):
            return True
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--data-root", default="data/experiments")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--force-regenerate-fallback", action="store_true", default=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = read_index(args.index)

    # delete previously-cached fallback plans so they regenerate with the LLM
    removed = 0
    for p in out_dir.glob("*.json"):
        if is_fallback_plan(p):
            p.unlink(); removed += 1
    print(f"[claude-planner] removed {removed} previously-fallback cached plans", flush=True)

    rows = []
    for _, row in frame.iterrows():
        rd = row.to_dict()
        err = rd.get("error_message")
        if isinstance(err, str) and err.strip():
            continue
        dp = rd.get("data_path")
        if dp is None or (isinstance(dp, float) and pd.isna(dp)):
            continue
        rows.append(str(dp))
    print(f"[claude-planner] {len(rows)} experiments to plan", flush=True)

    def make_client():
        return RetryingClient(ResponsesAPIClient.from_env())

    def job(exp_dir):
        client = make_client()
        try:
            plan = plan_experiment_with_llm(exp_dir, llm_client=client, fallback="rule",
                                            cache_dir=str(out_dir), use_cache=True)
            fb = any(isinstance(e, dict) and e.get("fallback_used") for e in (plan.evidence or []))
            return (exp_dir, fb, None)
        except Exception as exc:
            return (exp_dir, True, str(exc)[:160])

    done = 0; fallbacks = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(job, r): r for r in rows}
        for fut in as_completed(futs):
            exp_dir, fb, err = fut.result()
            done += 1
            if fb: fallbacks += 1
            if done % 25 == 0 or done == len(rows):
                print(f"[claude-planner] generated {done}/{len(rows)} (fallback={fallbacks})", flush=True)
    print(f"[claude-planner] DONE. total={done} fallback={fallbacks} valid_claude={done-fallbacks}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
