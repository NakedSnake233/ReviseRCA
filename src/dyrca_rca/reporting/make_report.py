"""Generate markdown reports for dynamic graph RCA experiment runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dyrca_rca.dataset.materialize import read_index


def make_report(index_path: str, results_root: str, out_path: str,
                graph_update_dir: Optional[str] = None,
                title: str = "Dynamic Graph RCA Experiment Report",
                max_examples: int = 10,
                baselines_root: Optional[str] = None) -> str:
    """Build a markdown report from index, metrics, predictions, and reranker outputs."""
    index = _read_optional_index(index_path)
    result_root = Path(results_root)
    graph_update_root = Path(graph_update_dir) if graph_update_dir else result_root / "graph_update_eval"
    baseline_root = Path(baselines_root) if baselines_root else Path("artifacts/baselines")
    runs = discover_runs(result_root)
    baseline_rows = _final_comparison_rows(baseline_root, runs, result_root)

    lines = [
        f"# {title}",
        "",
        "This report is generated from fixed-configuration RCA experiments. "
        "It does not tune model hyperparameters and does not compare search spaces.",
        "",
        "## Dataset Summary",
        "",
        _dataset_summary(index),
        "",
        "## Graph Update Metrics",
        "",
        _graph_update_section(graph_update_root),
        "",
        "## RCA Metrics Overall",
        "",
        _overall_metrics_table(runs),
        "",
        "## Reranked RCA Metrics Overall",
        "",
        _reranked_overall_metrics_table(runs),
        "",
        "## Baseline Comparison",
        "",
        _baseline_comparison_section(baseline_rows, baseline_root),
        "",
        "## RCA Metrics by Scenario",
        "",
        _by_scenario_table(runs),
        "",
        "## Ablation Table",
        "",
        _ablation_table(runs),
        "",
        "## Graph Update Fixed Static-GNN Errors",
        "",
        _fixed_static_errors(runs, max_examples=max_examples),
        "",
        "## Reranker Changed Ranking",
        "",
        _reranker_changes(result_root, max_examples=max_examples),
        "",
        "## Failure Case Analysis",
        "",
        _failure_analysis(index, graph_update_root),
        "",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    out.write_text(text, encoding="utf-8")
    _write_final_comparison_csv(out.with_name("final_results_with_baselines.csv"), baseline_rows)
    return str(out)


def discover_runs(results_root: Path) -> List[Dict[str, Any]]:
    """Find result directories with metrics/config files."""
    runs = []
    if not results_root.exists():
        return runs
    for overall_path in sorted(results_root.rglob("metrics_overall.csv")):
        run_dir = overall_path.parent
        config = _read_json(run_dir / "config.json")
        overall = _read_csv(overall_path)
        by_scenario = _read_csv(run_dir / "metrics_by_scenario.csv")
        predictions = _read_csv(run_dir / "predictions.csv")
        runs.append({
            "run_dir": run_dir,
            "run_id": config.get("run_id", run_dir.name),
            "graph_variant": config.get("graph_variant", _infer_variant(run_dir.name)),
            "model": config.get("model", _infer_model(run_dir.name)),
            "split": config.get("split", ""),
            "seed": config.get("seed", ""),
            "overall": overall,
            "by_scenario": by_scenario,
            "predictions": predictions,
        })
    return runs


def _dataset_summary(index: pd.DataFrame) -> str:
    if index.empty:
        return "_No index rows available._"
    rows = [
        ["experiments", len(index)],
        ["scenarios", index["scenario_id"].nunique() if "scenario_id" in index else 0],
        ["target_services", index["target_service"].nunique() if "target_service" in index else 0],
    ]
    if "scenario_id" in index:
        scenario_counts = index["scenario_id"].fillna("unknown").value_counts().sort_index()
        rows.extend([[f"scenario_{key}", int(value)] for key, value in scenario_counts.items()])
    return _markdown_table(["field", "value"], rows)


def _graph_update_section(graph_update_root: Path) -> str:
    metrics = _read_csv(graph_update_root / "graph_update_metrics.csv")
    by_scenario = _read_csv(graph_update_root / "graph_update_by_scenario.csv")
    parts = []
    if not metrics.empty:
        columns = [
            "operation_validity_rate",
            "plan_execution_success_rate",
            "node_add_remove_f1",
            "node_alignment_accuracy",
            "edge_redirect_recall",
            "blackhole_recall",
            "version_split_recall",
            "edge_weight_mae",
            "trace_mismatch_recall",
        ]
        rows = []
        for col in columns:
            values = pd.to_numeric(metrics[col], errors="coerce").dropna() if col in metrics else []
            rows.append([col, _fmt(float(values.mean())) if len(values) else "n/a"])
        parts.append(_markdown_table(["metric", "mean"], rows))
    else:
        parts.append("_No graph update metrics found._")
    if not by_scenario.empty:
        keep = [col for col in [
            "scenario_id",
            "count",
            "operation_validity_rate",
            "plan_execution_success_rate",
            "edge_redirect_recall",
            "blackhole_recall",
            "version_split_recall",
            "trace_mismatch_recall",
        ] if col in by_scenario]
        parts.append("")
        parts.append(_frame_to_markdown(by_scenario[keep], max_rows=20))
    return "\n".join(parts)


def _overall_metrics_table(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        overall = run["overall"]
        if overall.empty:
            continue
        row = overall.iloc[0].to_dict()
        rows.append([
            run["run_id"],
            run["graph_variant"],
            run["model"],
            _fmt(row.get("top1")),
            _fmt(row.get("top3")),
            _fmt(row.get("mrr")),
            _fmt(row.get("mean_rank")),
            int(row.get("count", 0) or 0),
        ])
    if not rows:
        return "_No RCA metrics found._"
    return _markdown_table(
        ["run", "variant", "model", "Top@1", "Top@3", "MRR", "mean_rank", "count"],
        rows,
    )


def _reranked_overall_metrics_table(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        path = run["run_dir"] / "rerank_llm" / "rerank_metrics_comparison.json"
        if not path.exists():
            continue
        data = _read_json(path)
        before = data.get("before", {}) if isinstance(data.get("before"), dict) else {}
        after = data.get("after", {}) if isinstance(data.get("after"), dict) else {}
        rows.append([
            run["run_id"],
            run["graph_variant"],
            run["model"],
            _fmt(before.get("top1")),
            _fmt(after.get("top1")),
            _fmt(before.get("top3")),
            _fmt(after.get("top3")),
            _fmt(before.get("mrr")),
            _fmt(after.get("mrr")),
        ])
    if not rows:
        return "_No LLM reranker metrics found._"
    rows.sort(key=lambda row: float(row[4]) if row[4] != "n/a" else -1.0, reverse=True)
    return _markdown_table(
        ["run", "variant", "model", "GNN Top@1", "Rerank Top@1", "GNN Top@3", "Rerank Top@3", "GNN MRR", "Rerank MRR"],
        rows,
    )


def _baseline_comparison_section(rows: List[Dict[str, Any]], baseline_root: Path) -> str:
    if not rows:
        return "_No baseline result files found._"
    lines = [
        "Baseline rows use the strict DY-RCA adapters: pre/during telemetry and static pre-fault topology only. "
        "They do not receive topology diffs, post metrics, during topology deltas, pod add/remove flags, or v2 flags.",
        "",
        "### Random Split",
        "",
    ]
    random_rows = [row for row in rows if row.get("split") == "random"]
    lines.append(_comparison_rows_to_markdown(random_rows))
    generalization = [row for row in rows if row.get("split") != "random"]
    if generalization:
        lines.extend([
            "",
            "### Strict Generalization Splits",
            "",
            _comparison_rows_to_markdown(generalization),
        ])
    lines.extend([
        "",
        "Baseline artifact root: `%s`." % baseline_root,
        "A consolidated machine-readable table is written to `final_results_with_baselines.csv` next to this report.",
    ])
    return "\n".join(lines)


def _comparison_rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    ordered = sorted(rows, key=lambda row: (
        str(row.get("split", "")),
        0 if row.get("family") == "baseline" else 1,
        str(row.get("method", "")),
    ))
    table_rows = [
        [
            row.get("split", ""),
            row.get("method", ""),
            _fmt(row.get("top1")),
            _fmt(row.get("top3")),
            _fmt(row.get("top5")),
            _fmt(row.get("mrr")),
            _fmt(row.get("mean_rank")),
            int(row.get("count", 0) or 0),
        ]
        for row in ordered
    ]
    return _markdown_table(
        ["split", "method", "Top@1", "Top@3", "Top@5", "MRR", "mean_rank", "count"],
        table_rows,
    )


def _final_comparison_rows(baseline_root: Path, runs: List[Dict[str, Any]],
                           results_root: Path) -> List[Dict[str, Any]]:
    rows = []
    rows.extend(_baseline_rows(baseline_root))
    rows.extend(_main_random_rows(runs))
    rows.extend(_main_generalization_rows(results_root.parent))
    return rows


def _baseline_rows(baseline_root: Path) -> List[Dict[str, Any]]:
    configs = [
        ("random", baseline_root / "dyrca_full_strict" / "metrics_overall.csv"),
        ("leave_service_out", baseline_root / "dyrca_leave_service_out_strict" / "metrics_overall.csv"),
        ("leave_scenario_out", baseline_root / "dyrca_leave_scenario_out_strict" / "metrics_overall.csv"),
    ]
    rows = []
    for split, path in configs:
        frame = _read_csv(path)
        if frame.empty:
            continue
        for _, item in frame.iterrows():
            baseline = str(item.get("baseline", ""))
            rows.append(_metric_row(
                split=split,
                method=_baseline_display_name(baseline),
                family="baseline",
                source_path=str(path),
                item=item,
            ))
    return rows


def _main_random_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        if run.get("graph_variant") != "llm_full" or run.get("model") != "weighted_graphsage":
            continue
        overall = run.get("overall", pd.DataFrame())
        if overall.empty:
            continue
        rows.append(_metric_row(
            split="random",
            method="DY-RCA llm_full + WeightedGraphSAGE",
            family="main",
            source_path=str(run["run_dir"] / "metrics_overall.csv"),
            item=overall.iloc[0],
        ))
        rerank_path = run["run_dir"] / "rerank_llm" / "rerank_metrics_comparison.json"
        rerank = _read_json(rerank_path)
        after = rerank.get("after", {}) if isinstance(rerank.get("after"), dict) else {}
        if after:
            rows.append(_metric_row(
                split="random",
                method="DY-RCA llm_full + WeightedGraphSAGE + LLM reranker",
                family="main",
                source_path=str(rerank_path),
                item=after,
            ))
    return rows


def _main_generalization_rows(full_root: Path) -> List[Dict[str, Any]]:
    configs = [
        (
            "leave_service_out",
            full_root / "results_llm_full_weighted_graphsage_leave_service_out" / "metrics_overall.csv",
        ),
        (
            "leave_scenario_out",
            full_root / "results_llm_full_weighted_graphsage_leave_scenario_out" / "metrics_overall.csv",
        ),
    ]
    rows = []
    for split, path in configs:
        frame = _read_csv(path)
        if frame.empty:
            continue
        rows.append(_metric_row(
            split=split,
            method="DY-RCA llm_full + WeightedGraphSAGE",
            family="main",
            source_path=str(path),
            item=frame.iloc[0],
        ))
    return rows


def _metric_row(split: str, method: str, family: str,
                source_path: str, item: Any) -> Dict[str, Any]:
    return {
        "split": split,
        "method": method,
        "family": family,
        "top1": _as_float(_item_get(item, "top1")),
        "top3": _as_float(_item_get(item, "top3")),
        "top5": _as_float(_item_get(item, "top5")),
        "mrr": _as_float(_item_get(item, "mrr")),
        "mean_rank": _as_float(_item_get(item, "mean_rank")),
        "count": int(_as_float(_item_get(item, "count"), 0.0) or 0),
        "source_path": source_path,
    }


def _baseline_display_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized == "eadro":
        return "Eadro adapted baseline"
    if normalized == "holisticrca":
        return "HolisticRCA adapted baseline"
    return name


def _write_final_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _by_scenario_table(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        frame = run["by_scenario"]
        if frame.empty:
            continue
        for _, item in frame.iterrows():
            rows.append([
                run["graph_variant"],
                run["model"],
                item.get("scenario_id", ""),
                _fmt(item.get("top1")),
                _fmt(item.get("top3")),
                _fmt(item.get("mrr")),
                int(item.get("count", 0) or 0),
            ])
    if not rows:
        return "_No per-scenario RCA metrics found._"
    return _markdown_table(["variant", "model", "scenario", "Top@1", "Top@3", "MRR", "count"], rows[:80])


def _ablation_table(runs: List[Dict[str, Any]]) -> str:
    rows = []
    baseline = {}
    for run in runs:
        overall = run["overall"]
        if overall.empty:
            continue
        row = overall.iloc[0].to_dict()
        key = run["model"]
        if run["graph_variant"] == "static_pre":
            baseline[key] = float(row.get("mrr", 0.0) or 0.0)
    for run in runs:
        overall = run["overall"]
        if overall.empty:
            continue
        row = overall.iloc[0].to_dict()
        mrr = float(row.get("mrr", 0.0) or 0.0)
        base = baseline.get(run["model"])
        rows.append([
            run["graph_variant"],
            run["model"],
            _fmt(row.get("top1")),
            _fmt(row.get("top3")),
            _fmt(mrr),
            _fmt(mrr - base) if base is not None else "n/a",
        ])
    if not rows:
        return "_No ablation runs found._"
    return _markdown_table(["variant", "model", "Top@1", "Top@3", "MRR", "MRR_delta_vs_static"], rows)


def _fixed_static_errors(runs: List[Dict[str, Any]], max_examples: int) -> str:
    static_by_model = {
        run["model"]: run for run in runs
        if run["graph_variant"] == "static_pre" and not run["predictions"].empty
    }
    rows = []
    for run in runs:
        if run["graph_variant"] == "static_pre" or run["predictions"].empty:
            continue
        static = static_by_model.get(run["model"])
        if static is None:
            continue
        merged = static["predictions"].merge(
            run["predictions"],
            on="experiment_id",
            suffixes=("_static", "_updated"),
        )
        for _, item in merged.iterrows():
            static_rank = _rank_value(item.get("rank_static"))
            updated_rank = _rank_value(item.get("rank_updated"))
            if static_rank != 1 and updated_rank == 1:
                rows.append([
                    item.get("experiment_id", ""),
                    run["model"],
                    run["graph_variant"],
                    item.get("scenario_id_updated", item.get("scenario_id_static", "")),
                    item.get("label_node_id_updated", item.get("label_node_id_static", "")),
                    static_rank if static_rank is not None else "missing",
                    updated_rank,
                ])
    if not rows:
        return "_No examples found where an updated graph fixed a static-GNN Top@1 miss._"
    return _markdown_table(
        ["experiment", "model", "updated_variant", "scenario", "label", "static_rank", "updated_rank"],
        rows[:max_examples],
    )


def _reranker_changes(results_root: Path, max_examples: int) -> str:
    rows = []
    for path in sorted(results_root.rglob("rerank_explanations.jsonl")):
        if path.parent.name != "rerank_llm":
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("changed_order"):
                    continue
                final = item.get("final_ranking", [])
                top = final[0].get("candidate", "") if final and isinstance(final[0], dict) else ""
                rows.append([
                    item.get("experiment_id", ""),
                    path.parent.name,
                    top,
                    _fmt(item.get("confidence")),
                    item.get("summary", item.get("explanation", "")),
                ])
    if not rows:
        return "_No reranker order changes found._"
    return _markdown_table(["experiment", "rerank_run", "new_top", "confidence", "summary"], rows[:max_examples])


def _failure_analysis(index: pd.DataFrame, graph_update_root: Path) -> str:
    metrics = _read_csv(graph_update_root / "graph_update_metrics.csv")
    lines = []
    if index.empty:
        return "_No index available for failure-case slicing._"
    if not metrics.empty and "experiment_id" in metrics:
        merged = index.merge(metrics, on="experiment_id", suffixes=("", "_metric"), how="left")
    else:
        merged = index

    t3_empty = _filter_contains(merged, ["scenario_name", "topology_change_type"], ["empty", "edge_removal"])
    lines.append("### T3 Empty Endpoint")
    lines.append(_failure_slice_summary(
        t3_empty,
        metric_names=["blackhole_recall", "plan_execution_success_rate"],
        no_rows="No T3 empty-endpoint cases found in the indexed set.",
    ))
    lines.append("")

    t4 = _filter_contains(merged, ["scenario_id", "scenario_name", "topology_change_type", "params_json"], ["T4", "canary", "version_split", "v2_weight"])
    low_weight = []
    for _, row in t4.iterrows():
        weight = _v2_weight(row.get("params_json", ""))
        if weight is None or weight <= 0.2:
            low_weight.append(row)
    low_t4 = pd.DataFrame(low_weight)
    lines.append("### T4 Low-Weight Canary")
    lines.append(_failure_slice_summary(
        low_t4,
        metric_names=["version_split_recall", "edge_weight_mae", "plan_execution_success_rate"],
        no_rows="No low-weight T4 canary cases found in the indexed set.",
    ))
    return "\n".join(lines)


def _failure_slice_summary(frame: pd.DataFrame, metric_names: List[str], no_rows: str) -> str:
    if frame.empty:
        return no_rows
    rows = [["cases", len(frame)]]
    for metric in metric_names:
        if metric not in frame:
            rows.append([metric, "n/a"])
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        rows.append([metric, _fmt(values.mean()) if len(values) else "n/a"])
    sample_ids = [str(v) for v in frame.get("experiment_id", pd.Series([])).head(5).tolist()]
    rows.append(["sample_experiments", ", ".join(sample_ids)])
    return _markdown_table(["field", "value"], rows)


def _read_optional_index(index_path: str) -> pd.DataFrame:
    try:
        return read_index(index_path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _frame_to_markdown(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = []
    for _, row in frame.head(max_rows).iterrows():
        rows.append([_fmt(row.get(col)) for col in frame.columns])
    return _markdown_table(list(frame.columns), rows)


def _markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_No rows._"
    out = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return "\n".join(out)


def _escape_cell(value: Any) -> str:
    text = _fmt(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        if isinstance(value, str) and value.strip() == "":
            return ""
        numeric = float(value)
        if str(value).strip().replace(".", "", 1).replace("-", "", 1).isdigit():
            return f"{numeric:.4f}"
    except Exception:
        pass
    return str(value)


def _item_get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    try:
        return item.get(key)
    except Exception:
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_variant(run_name: str) -> str:
    for variant in ["static_pre", "naive_during", "rule_node", "rule_edge", "rule_full", "llm_full"]:
        if variant in run_name:
            return variant
    return ""


def _infer_model(run_name: str) -> str:
    for model in ["weighted_graphsage", "edge_aware_gat"]:
        if model in run_name:
            return model
    return ""


def _rank_value(value: Any) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filter_contains(frame: pd.DataFrame, columns: List[str], needles: List[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series([False] * len(frame))
    lowered_needles = [needle.lower() for needle in needles]
    for col in columns:
        if col not in frame:
            continue
        series = frame[col].fillna("").astype(str).str.lower()
        col_mask = pd.Series([False] * len(frame))
        for needle in lowered_needles:
            col_mask = col_mask | series.str.contains(needle, regex=False)
        mask = mask | col_mask
    return frame[mask]


def _v2_weight(params_json: Any) -> Optional[float]:
    try:
        params = json.loads(str(params_json)) if isinstance(params_json, str) else params_json
    except Exception:
        return None
    if not isinstance(params, dict):
        return None
    for key in ("v2_weight", "v2Weight", "canary_weight"):
        if key in params:
            try:
                value = float(params[key])
                return value / 100.0 if value > 1.0 else value
            except (TypeError, ValueError):
                return None
    return None


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dynamic graph RCA markdown report.")
    parser.add_argument("--index", required=True, help="Experiment index artifact.")
    parser.add_argument("--results-root", required=True, help="Root containing experiment result directories.")
    parser.add_argument("--graph-update-dir", default=None,
                        help="Directory containing graph_update_metrics.csv and graph_update_by_scenario.csv.")
    parser.add_argument("--out", required=True, help="Output markdown path.")
    parser.add_argument("--title", default="Dynamic Graph RCA Experiment Report")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--baselines-root", default="artifacts/baselines",
                        help="Root containing strict baseline metrics. Missing files are ignored.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out = make_report(
        index_path=args.index,
        results_root=args.results_root,
        graph_update_dir=args.graph_update_dir,
        out_path=args.out,
        title=args.title,
        max_examples=args.max_examples,
        baselines_root=args.baselines_root,
    )
    print(f"wrote report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
