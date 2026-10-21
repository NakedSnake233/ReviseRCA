"""Multi-seed statistical validation for rebuttal (addresses R2-C5, R3-C1).

Re-runs the fixed-config RCA pipeline across multiple seeds for the key graph
variants, reusing cached LLM plans so no new LLM calls are needed. Then computes
mean/std, 95% bootstrap confidence intervals, and paired significance tests.

Usage:
    python -m dyrca_rca.experiments.rebuttal_multiseed \
        --index artifacts/dygraph_rca_full/index.csv \
        --data-root data/experiments \
        --llm-plan-dir artifacts/dygraph_rca_full/llm_plans \
        --out-dir artifacts/rebuttal_multiseed \
        --variants static_pre naive_during rule_full llm_full \
        --seeds 42 123 456 789 2024 \
        --model weighted_graphsage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from dyrca_rca.experiments.run_experiment import run_experiment


METRIC_COLS = ["top1", "top3", "top5", "mrr", "mean_rank"]


def _bootstrap_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.RandomState(20260711)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, grp in seed_metrics.groupby("variant"):
        n = len(grp)
        for metric in METRIC_COLS:
            vals = grp[metric].astype(float).values
            mean = float(np.mean(vals)) if vals.size else float("nan")
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            lo, hi = _bootstrap_ci(vals) if vals.size else (float("nan"), float("nan"))
            rows.append({
                "variant": variant,
                "metric": metric,
                "n_seeds": n,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "min": round(float(np.min(vals)), 4) if vals.size else float("nan"),
                "max": round(float(np.max(vals)), 4) if vals.size else float("nan"),
            })
    return pd.DataFrame(rows)


def paired_tests(seed_metrics: pd.DataFrame, baseline_variant: str = "static_pre") -> pd.DataFrame:
    """Paired test across seeds (per-seed metric) and on per-instance ranks at seed 42."""
    rows = []
    variants = [v for v in sorted(seed_metrics["variant"].unique()) if v != baseline_variant]
    base = seed_metrics[seed_metrics["variant"] == baseline_variant].sort_values("seed")
    for variant in variants:
        cur = seed_metrics[seed_metrics["variant"] == variant].sort_values("seed")
        merged = pd.merge(base, cur, on="seed", suffixes=("_base", "_cur"))
        for metric in ["top1", "mrr"]:
            a = merged[f"{metric}_base"].astype(float).values
            b = merged[f"{metric}_cur"].astype(float).values
            if a.size >= 2 and not np.allclose(a, b):
                try:
                    t_stat, t_p = stats.ttest_rel(b, a)
                except Exception:
                    t_stat, t_p = float("nan"), float("nan")
                try:
                    w_stat, w_p = stats.wilcoxon(b, a, zero_method="wilcox")
                except Exception:
                    w_stat, w_p = float("nan"), float("nan")
            else:
                t_stat = t_p = w_stat = w_p = float("nan")
            rows.append({
                "comparison": f"{variant} - {baseline_variant}",
                "metric": metric,
                "n_seeds": int(a.size),
                "mean_delta": round(float(np.mean(b - a)), 4) if a.size else float("nan"),
                "paired_t": round(float(t_stat), 4),
                "paired_t_p": float(t_p) if not np.isnan(t_p) else float("nan"),
                "wilcoxon_p": float(w_p) if not np.isnan(w_p) else float("nan"),
            })
    return pd.DataFrame(rows)


def per_instance_significance(index: str, data_root: str, llm_plan_dir: str,
                              baseline_variant: str, target_variant: str,
                              seed: int = 42, model: str = "weighted_graphsage") -> Dict[str, Any]:
    """Per-experiment paired test on rank @ seed: baseline vs target."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="rebuttal_paired_"))
    base_dir = tmp / "base"; cur_dir = tmp / "cur"
    run_experiment(index=index, graph_variant=baseline_variant, model_name=model,
                   split="random", seed=seed, limit=None, out_dir=str(base_dir),
                   data_root=data_root, llm_plan_dir=llm_plan_dir)
    run_experiment(index=index, graph_variant=target_variant, model_name=model,
                   split="random", seed=seed, limit=None, out_dir=str(cur_dir),
                   data_root=data_root, llm_plan_dir=llm_plan_dir)
    base_pred = pd.read_csv(base_dir / "predictions.csv")
    cur_pred = pd.read_csv(cur_dir / "predictions.csv")
    merged = pd.merge(base_pred[["experiment_id", "rank"]], cur_pred[["experiment_id", "rank"]],
                      on="experiment_id", suffixes=("_base", "_cur"))
    a = merged["rank_base"].astype(float).values
    b = merged["rank_cur"].astype(float).values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    result = {
        "baseline": baseline_variant, "target": target_variant, "seed": seed,
        "n_instances": int(a.size),
        "baseline_mean_rank": round(float(np.mean(a)), 4),
        "target_mean_rank": round(float(np.mean(b)), 4),
    }
    if a.size and not np.allclose(a, b):
        try:
            w_stat, w_p = stats.wilcoxon(b, a, zero_method="wilcox")
            result["wilcoxon_stat"] = float(w_stat); result["wilcoxon_p"] = float(w_p)
        except Exception as exc:
            result["wilcoxon_error"] = str(exc)
        # sign test on improved vs worsened
        improved = int(np.sum(b < a)); worsened = int(np.sum(b > a)); ties = int(np.sum(b == a))
        result["improved"] = improved; result["worsened"] = worsened; result["ties"] = ties
        try:
            # McNemar-style binomial test on improved vs worsened
            if improved + worsened > 0:
                _, sign_p = stats.binomtest(improved, improved + worsened, p=0.5, alternative="greater")
                result["sign_test_p"] = float(sign_p)
        except Exception as exc:
            result["sign_test_error"] = str(exc)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Multi-seed rebuttal validation.")
    p.add_argument("--index", required=True)
    p.add_argument("--data-root", default="data/experiments")
    p.add_argument("--llm-plan-dir", default="artifacts/dygraph_rca_full/llm_plans")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants", nargs="+", default=["static_pre", "naive_during", "rule_full", "llm_full"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    p.add_argument("--model", default="weighted_graphsage")
    p.add_argument("--skip-run", action="store_true", help="only aggregate from existing per-seed dirs")
    args = p.parse_args(argv)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    seed_rows: List[Dict[str, Any]] = []

    for variant in args.variants:
        for seed in args.seeds:
            run_dir = out / f"{variant}_{args.model}_s{seed}"
            metrics_csv = run_dir / "metrics_overall.csv"
            if args.skip_run and metrics_csv.exists():
                pass
            elif (not args.skip_run) and (not metrics_csv.exists() or True):
                print(f"[run] variant={variant} seed={seed} model={args.model}", flush=True)
                res = run_experiment(
                    index=args.index, graph_variant=variant, model_name=args.model,
                    split="random", seed=seed, limit=None, out_dir=str(run_dir),
                    data_root=args.data_root, llm_plan_dir=args.llm_plan_dir,
                )
                m = res["metrics"]
            else:
                mdf = pd.read_csv(metrics_csv); m = mdf.iloc[0].to_dict()
            mdf2 = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()
            if len(mdf2):
                row = mdf2.iloc[0].to_dict()
            else:
                row = {k: m.get(k) for k in METRIC_COLS}
            row.update({"variant": variant, "seed": seed, "model": args.model})
            seed_rows.append(row)
            (out / "seed_metrics.csv").write_text(
                pd.DataFrame(seed_rows).to_csv(index=False), encoding="utf-8"
            )
            print(f"   -> top1={row.get('top1'):.4f} mrr={row.get('mrr'):.4f}", flush=True)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(out / "seed_metrics.csv", index=False)

    summary = summarize(seed_df)
    summary.to_csv(out / "summary_multiseed.csv", index=False)
    print("\n=== Summary (mean ± std, 95% CI over seeds) ===")
    print(summary.pivot(index="variant", columns="metric",
                        values=["mean", "std"]).to_string() if "mean" in summary.columns else summary.to_string())

    if len(args.variants) > 1 and "static_pre" in args.variants:
        pt = paired_tests(seed_df, baseline_variant="static_pre")
        pt.to_csv(out / "paired_tests.csv", index=False)
        print("\n=== Paired tests vs static_pre (across seeds) ===")
        print(pt.to_string(index=False))

        # per-instance significance at seed 42 for the strongest variant
        target = "llm_full" if "llm_full" in args.variants else args.variants[-1]
        try:
            pi = per_instance_significance(args.index, args.data_root, args.llm_plan_dir,
                                           "static_pre", target, seed=42, model=args.model)
            (out / "per_instance_significance.json").write_text(
                json.dumps(pi, indent=2), encoding="utf-8")
            print("\n=== Per-instance significance @ seed 42 ===")
            print(json.dumps(pi, indent=2))
        except Exception as exc:
            print(f"[warn] per-instance significance failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
