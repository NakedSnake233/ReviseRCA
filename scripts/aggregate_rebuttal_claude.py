"""Aggregate Claude-based rebuttal results into a unified comparison + significance report.

Combines:
  - Non-LLM variants (static_pre, naive_during, rule_full) from the original multi-seed
    sweep (these do not use LLM plans, so they are unchanged).
  - Plan2RCA llm_full using CLAUDE-generated plans (artifacts/rebuttal_claude/multiseed).
  - EvolveGCN-A temporal baseline (artifacts/rebuttal_temporal, no LLM, unchanged).
  - Free-form LLM diagnosis with CLAUDE (artifacts/rebuttal_claude/freeform, primary).
  - Free-form LLM diagnosis with gpt-5.4-mini (artifacts/rebuttal_freeform, secondary model comparison).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["top1", "top3", "top5", "mrr", "mean_rank"]


def _m(d: Path) -> dict:
    return pd.read_csv(d / "metrics_overall.csv").iloc[0].to_dict()


def collect() -> pd.DataFrame:
    rows = []
    # non-LLM variants from original sweep
    orig = pd.read_csv("artifacts/rebuttal_multiseed/seed_metrics.csv")
    for v in ["static_pre", "naive_during", "rule_full"]:
        for _, r in orig[orig["variant"] == v].iterrows():
            rows.append({"method": f"Plan2RCA({v})", "variant": v, "seed": int(r["seed"]),
                         "top1": float(r["top1"]), "top3": float(r["top3"]), "top5": float(r["top5"]),
                         "mrr": float(r["mrr"]), "mean_rank": float(r["mean_rank"])})
    # Plan2RCA llm_full with CLAUDE plans
    for d in sorted(Path("artifacts/rebuttal_claude/multiseed").glob("llm_full_*_s*")):
        m = _m(d)
        rows.append({"method": "Plan2RCA llm_full (Claude planner)", "variant": "llm_full_claude",
                     "seed": int(d.name.split("_s")[-1]), "top1": float(m["top1"]),
                     "top3": float(m["top3"]), "top5": float(m["top5"]), "mrr": float(m["mrr"]),
                     "mean_rank": float(m["mean_rank"])})
    # EvolveGCN-A (no LLM)
    for d in sorted(Path("artifacts/rebuttal_temporal").glob("evolvegcn_a_s*")):
        m = _m(d)
        rows.append({"method": "EvolveGCN-A (temporal baseline)", "variant": "evolvegcn_a",
                     "seed": int(d.name.split("_s")[-1]), "top1": float(m["top1"]),
                     "top3": float(m["top3"]), "top5": float(m["top5"]), "mrr": float(m["mrr"]),
                     "mean_rank": float(m["mean_rank"])})
    # Free-form Claude (primary)
    for d in sorted(d for d in Path("artifacts/rebuttal_claude/freeform").glob("test_s*") if d.is_dir()):
        m = _m(d)
        rows.append({"method": "Free-form LLM (Claude)", "variant": "freeform_claude",
                     "seed": int(d.name.split("_s")[-1]), "top1": float(m["top1"]),
                     "top3": float(m["top3"]), "top5": float(m["top5"]), "mrr": float(m["mrr"]),
                     "mean_rank": float(m["mean_rank"])})
    # Free-form gpt-5.4-mini (secondary, model comparison)
    for d in sorted(d for d in Path("artifacts/rebuttal_freeform").glob("test_s*") if d.is_dir()):
        m = _m(d)
        rows.append({"method": "Free-form LLM (gpt-5.4-mini)", "variant": "freeform_gpt",
                     "seed": int(d.name.split("_s")[-1]), "top1": float(m["top1"]),
                     "top3": float(m["top3"]), "top5": float(m["top5"]), "mrr": float(m["mrr"]),
                     "mean_rank": float(m["mean_rank"])})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["Plan2RCA(static_pre)", "Plan2RCA(naive_during)", "Plan2RCA(rule_full)",
             "EvolveGCN-A (temporal baseline)", "Free-form LLM (gpt-5.4-mini)",
             "Free-form LLM (Claude)", "Plan2RCA llm_full (Claude planner)"]
    for method in order:
        grp = df[df["method"] == method]
        if grp.empty:
            continue
        n = len(grp)
        for metric in METRICS:
            vals = grp[metric].astype(float).values
            mean = float(np.mean(vals)); std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            if vals.size >= 2:
                rng = np.random.RandomState(20260713)
                boots = rng.choice(vals, size=(10000, vals.size), replace=True).mean(axis=1)
                lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            else:
                lo = hi = mean
            rows.append({"method": method, "metric": metric, "n": n, "mean": round(mean, 4),
                         "std": round(std, 4), "ci95": f"[{round(lo,4)},{round(hi,4)}]"})
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, base_variant: str, target_variant: str, label: str) -> list:
    out = []
    b = df[df["variant"] == base_variant].sort_values("seed")
    t = df[df["variant"] == target_variant].sort_values("seed")
    merged = pd.merge(b[["seed", "top1", "mrr"]], t[["seed", "top1", "mrr"]], on="seed", suffixes=("_b", "_t"))
    for metric in ["top1", "mrr"]:
        a = merged[f"{metric}_b"].astype(float).values
        c = merged[f"{metric}_t"].astype(float).values
        if a.size >= 2 and not np.allclose(a, c):
            tt, tp = stats.ttest_rel(c, a)
            try:
                _, wp = stats.wilcoxon(c, a, zero_method="wilcox")
            except Exception:
                wp = float("nan")
        else:
            tt = tp = wp = float("nan")
        out.append({"comparison": label, "metric": metric, "n": int(a.size),
                    "delta": round(float(np.mean(c - a)), 4), "paired_t_p": float(tp) if not np.isnan(tp) else float("nan"),
                    "wilcoxon_p": float(wp) if not np.isnan(wp) else float("nan")})
    return out


def main() -> int:
    df = collect()
    df.to_csv("artifacts/rebuttal_claude/all_metrics.csv", index=False)
    s = summarize(df)
    s.to_csv("artifacts/rebuttal_claude/summary.csv", index=False)

    # paired tests
    pt = []
    pt += paired(df, "static_pre", "llm_full_claude", "Plan2RCA(Claude) - static")
    pt += paired(df, "static_pre", "freeform_claude", "Free-form(Claude) - static")
    pt += paired(df, "freeform_claude", "llm_full_claude", "Plan2RCA(Claude) - Free-form(Claude) [SAME MODEL]")
    pt += paired(df, "freeform_gpt", "freeform_claude", "Free-form(Claude) - Free-form(gpt) [model effect]")
    pd.DataFrame(pt).to_csv("artifacts/rebuttal_claude/significance.csv", index=False)

    print("=== CLAUDE-BASED SUMMARY (Acc@1 / MRR, n=5 seeds) ===")
    for method in s["method"].unique():
        sub = s[s["method"] == method]
        t1 = sub[sub["metric"] == "top1"].iloc[0]
        mrr = sub[sub["metric"] == "mrr"].iloc[0]
        print(f"  {method}: Acc@1={t1['mean']:.3f}±{t1['std']:.3f} {t1['ci95']}  MRR={mrr['mean']:.3f}±{mrr['std']:.3f}")
    print("\n=== Paired tests ===")
    print(pd.DataFrame(pt).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
