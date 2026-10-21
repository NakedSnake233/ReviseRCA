"""Aggregate rebuttal experiment results into a single comparison + significance report.

Combines:
  - Multi-seed fixed-config sweep (artifacts/rebuttal_multiseed/)
  - EvolveGCN-A temporal baseline (artifacts/rebuttal_temporal/)
  - (optional) free-form LLM ablation (artifacts/rebuttal_freeform/)

Outputs:
  - artifacts/rebuttal_results/summary.csv  (mean ± std, 95% CI per variant/arch)
  - artifacts/rebuttal_results/significance.csv (paired tests vs static_pre)
  - artifacts/rebuttal_results/REPORT.md (human-readable, rebuttal-ready)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["top1", "top3", "top5", "mrr", "mean_rank"]


def _read_metrics(path: Path) -> dict:
    df = pd.read_csv(path)
    return df.iloc[0].to_dict()


def collect_multiseed(root: Path) -> pd.DataFrame:
    rows = []
    sm = root / "seed_metrics.csv"
    if sm.exists():
        df = pd.read_csv(sm)
        for _, r in df.iterrows():
            rows.append({
                "method": f"Plan2RCA({r['variant']})", "variant": r["variant"],
                "seed": int(r["seed"]), "top1": float(r["top1"]), "top3": float(r["top3"]),
                "top5": float(r["top5"]), "mrr": float(r["mrr"]),
                "mean_rank": float(r["mean_rank"]),
            })
    return pd.DataFrame(rows)


def collect_temporal(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(root.glob("evolvegcn_a_s*")):
        m = d / "metrics_overall.csv"
        cfg = d / "config.json"
        if not m.exists():
            continue
        md = _read_metrics(m)
        seed = -1
        if cfg.exists():
            seed = int(json.loads(cfg.read_text()).get("seed", -1))
        rows.append({
            "method": "EvolveGCN-A (temporal baseline)", "variant": "evolvegcn_a",
            "seed": seed, "top1": float(md["top1"]), "top3": float(md["top3"]),
            "top5": float(md["top5"]), "mrr": float(md["mrr"]),
            "mean_rank": float(md["mean_rank"]),
        })
    for d in sorted(root.glob("evolvegcn_o_s*")):
        m = d / "metrics_overall.csv"
        cfg = d / "config.json"
        if not m.exists():
            continue
        md = _read_metrics(m)
        seed = int(json.loads(cfg.read_text()).get("seed", -1)) if cfg.exists() else -1
        rows.append({
            "method": "EvolveGCN-O (temporal baseline)", "variant": "evolvegcn_o",
            "seed": seed, "top1": float(md["top1"]), "top3": float(md["top3"]),
            "top5": float(md["top5"]), "mrr": float(md["mrr"]),
            "mean_rank": float(md["mean_rank"]),
        })
    return pd.DataFrame(rows)


def collect_freeform(root: Path) -> pd.DataFrame:
    """Collect per-seed free-form LLM ablation results (test_s<seed>/ dirs)."""
    rows = []
    for d in sorted(root.glob("test_s*")):
        m = d / "metrics_overall.csv"
        cfg = d / "config.json"
        if not m.exists():
            continue
        md = _read_metrics(m)
        seed = -1
        if cfg.exists():
            seed = int(json.loads(cfg.read_text()).get("split_seed", -1))
        rows.append({
            "method": "Free-form LLM diagnosis (gpt-5.4-mini)", "variant": "freeform_llm",
            "seed": seed, "top1": float(md["top1"]), "top3": float(md["top3"]),
            "top5": float(md["top5"]), "mrr": float(md["mrr"]),
            "mean_rank": float(md["mean_rank"]),
        })
    return pd.DataFrame(rows)


def summarize(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in all_df.groupby("method", sort=False):
        n = len(grp)
        for metric in METRICS:
            vals = grp[metric].astype(float).values
            mean = float(np.mean(vals)) if vals.size else float("nan")
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            if vals.size >= 2:
                rng = np.random.RandomState(20260711)
                boots = rng.choice(vals, size=(10000, vals.size), replace=True).mean(axis=1)
                lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            else:
                lo = hi = mean
            rows.append({"method": method, "metric": metric, "n_seeds": n,
                         "mean": round(mean, 4), "std": round(std, 4),
                         "ci95": f"[{round(lo,4)}, {round(hi,4)}]",
                         "min": round(float(np.min(vals)), 4) if vals.size else float("nan"),
                         "max": round(float(np.max(vals)), 4) if vals.size else float("nan")})
    return pd.DataFrame(rows)


def significance(all_df: pd.DataFrame, baseline: str = "static_pre") -> pd.DataFrame:
    rows = []
    base = all_df[all_df["variant"] == baseline]
    if base.empty:
        return pd.DataFrame()
    base = base.sort_values("seed")
    for variant, grp in all_df.groupby("variant", sort=False):
        if variant == baseline:
            continue
        cur = grp.sort_values("seed")
        merged = pd.merge(base[["seed", "top1", "mrr"]], cur[["seed", "top1", "mrr"]],
                          on="seed", suffixes=("_b", "_c"))
        for metric in ["top1", "mrr"]:
            a = merged[f"{metric}_b"].astype(float).values
            b = merged[f"{metric}_c"].astype(float).values
            if a.size >= 2 and not np.allclose(a, b):
                t, tp = stats.ttest_rel(b, a)
                try:
                    w, wp = stats.wilcoxon(b, a, zero_method="wilcox")
                except Exception:
                    w, wp = float("nan"), float("nan")
            else:
                t = tp = w = wp = float("nan")
            rows.append({
                "comparison": f"{cur['method'].iloc[0]} - Plan2RCA({baseline})",
                "metric": metric, "n_seeds": int(a.size),
                "mean_delta": round(float(np.mean(b - a)), 4) if a.size else float("nan"),
                "paired_t": round(float(t), 3) if not np.isnan(t) else float("nan"),
                "paired_t_p": float(tp) if not np.isnan(tp) else float("nan"),
                "wilcoxon_p": float(wp) if not np.isnan(wp) else float("nan"),
            })
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, sig: pd.DataFrame, out: Path) -> None:
    lines = ["# Rebuttal Supplementary Experiments — Aggregated Results\n"]
    lines.append("## Mean ± std and 95% bootstrap CI across seeds (Online Boutique dynamic benchmark)\n")
    pivot = summary.pivot(index="method", columns="metric", values=["mean", "std", "ci95"])
    # build a readable table for top1 and mrr
    lines.append("| Method | Acc@1 mean±std [95% CI] | MRR mean±std [95% CI] | n seeds |")
    lines.append("|---|---|---|---|")
    for method in summary["method"].unique():
        sub = summary[summary["method"] == method]
        n = int(sub["n_seeds"].iloc[0])
        def cell(metric):
            r = sub[sub["metric"] == metric]
            if r.empty:
                return "—"
            rr = r.iloc[0]
            return f"{rr['mean']:.3f} ± {rr['std']:.3f} {rr['ci95']}"
        lines.append(f"| {method} | {cell('top1')} | {cell('mrr')} | {n} |")
    lines.append("")
    if not sig.empty:
        lines.append("## Paired significance tests vs static-pre baseline (across seeds)\n")
        lines.append("| Comparison | Metric | Δ mean | paired t | t p-value | Wilcoxon p | n |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sig.iterrows():
            lines.append(
                f"| {r['comparison']} | {r['metric']} | {r['mean_delta']:+.3f} | "
                f"{r['paired_t']} | {r['paired_t_p']:.2e} | {r['wilcoxon_p']:.2e} | {r['n_seeds']} |"
            )
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed-dir", default="artifacts/rebuttal_multiseed")
    p.add_argument("--temporal-dir", default="artifacts/rebuttal_temporal")
    p.add_argument("--freeform-dir", default="artifacts/rebuttal_freeform")
    p.add_argument("--out-dir", default="artifacts/rebuttal_results")
    args = p.parse_args(argv)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    parts = [collect_multiseed(Path(args.multiseed_dir)),
             collect_temporal(Path(args.temporal_dir)),
             collect_freeform(Path(args.freeform_dir))]
    all_df = pd.concat([d for d in parts if not d.empty], ignore_index=True)
    if all_df.empty:
        print("no results found"); return 1
    all_df.to_csv(out / "all_metrics.csv", index=False)

    summary = summarize(all_df)
    summary.to_csv(out / "summary.csv", index=False)
    sig = significance(all_df, baseline="static_pre")
    sig.to_csv(out / "significance.csv", index=False)
    write_report(summary, sig, out / "REPORT.md")

    print("=== SUMMARY (Acc@1 / MRR) ===")
    for method in summary["method"].unique():
        sub = summary[summary["method"] == method]
        t1 = sub[sub["metric"] == "top1"].iloc[0]
        mrr = sub[sub["metric"] == "mrr"].iloc[0]
        print(f"  {method}: Acc@1={t1['mean']:.3f}±{t1['std']:.3f} {t1['ci95']}  "
              f"MRR={mrr['mean']:.3f}±{mrr['std']:.3f}  n={int(t1['n_seeds'])}")
    print(f"\nreport -> {out/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
