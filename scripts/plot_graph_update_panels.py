#!/usr/bin/env python
"""Plot graph-update effect panels for the DY-RCA paper.

The script uses measured experiment outputs only. It creates three independent
figures, each with axes, data, and legend only:

1. Overall Top@1/MRR grouped bars by graph variant and model.
2. Scenario residual error bars (`1 - Top@1`) for selected graph variants.
3. Executed graph-update quality bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANT_ORDER = [
    "static_pre",
    "naive_during",
    "rule_node",
    "rule_edge",
    "rule_full",
    "llm_full",
]
VARIANT_LABELS = [
    "Static\npre",
    "Naive\nduring",
    "Rule\nnode",
    "Rule\nedge",
    "Rule\nfull",
    "LLM\nfull",
]
VARIANT_SHORT = {
    "static_pre": "Static pre",
    "naive_during": "Naive during",
    "rule_node": "Rule node",
    "rule_edge": "Rule edge",
    "rule_full": "Rule full",
    "llm_full": "LLM full",
}
COLORS = {
    "static_pre": "#8E9AAF",
    "naive_during": "#C4A15A",
    "rule_node": "#66A182",
    "rule_edge": "#3E8E7E",
    "rule_full": "#2C6E91",
    "llm_full": "#7B2CBF",
    "sage": "#2563EB",
    "sage_mrr": "#93C5FD",
    "gat": "#16A34A",
    "gat_mrr": "#86EFAC",
}


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    overall = load_overall_metrics(results_root)
    scenario = load_scenario_metrics(results_root)
    graph_update = pd.read_csv(Path(args.graph_update_by_scenario))

    plot_overall_bars(overall, out_dir)
    plot_scenario_error(scenario, out_dir)
    plot_graph_update_quality(graph_update, out_dir)
    write_manifest(out_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DY-RCA graph-update panels.")
    parser.add_argument(
        "--results-root",
        default="artifacts/dygraph_rca_full/results",
        help="Directory containing graph-variant result folders.",
    )
    parser.add_argument(
        "--graph-update-by-scenario",
        default="artifacts/dygraph_rca_full/results/graph_update_eval_llm/graph_update_by_scenario.csv",
        help="Graph update by-scenario metrics CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/dygraph_rca_full/figures/graph_update_panels",
        help="Output directory for PDF/PNG/SVG figures.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })


def load_overall_metrics(results_root: Path) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        for model in ["weighted_graphsage", "edge_aware_gat"]:
            path = results_root / f"{variant}_{model}" / "metrics_overall.csv"
            row = pd.read_csv(path).iloc[0].to_dict()
            row.update({"variant": variant, "model": model})
            rows.append(row)
    return pd.DataFrame(rows)


def load_scenario_metrics(results_root: Path) -> pd.DataFrame:
    frames = []
    for variant in ["static_pre", "naive_during", "rule_edge", "llm_full"]:
        path = results_root / f"{variant}_weighted_graphsage" / "metrics_by_scenario.csv"
        frame = pd.read_csv(path)
        frame["variant"] = variant
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_axes(ax) -> None:
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


def save_all(fig, out_dir: Path, stem: str) -> None:
    for suffix in ["pdf", "png", "svg"]:
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        print(f"wrote {path}")
    plt.close(fig)


def plot_overall_bars(overall: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(len(VARIANT_ORDER))
    width = 0.18
    series = [
        ("GraphSAGE Top@1", "weighted_graphsage", "top1", COLORS["sage"], 0),
        ("GraphSAGE MRR", "weighted_graphsage", "mrr", COLORS["sage_mrr"], 1),
        ("GAT Top@1", "edge_aware_gat", "top1", COLORS["gat"], 2),
        ("GAT MRR", "edge_aware_gat", "mrr", COLORS["gat_mrr"], 3),
    ]
    for label, model, metric, color, idx in series:
        vals = [
            float(overall[(overall.variant == variant) & (overall.model == model)].iloc[0][metric])
            for variant in VARIANT_ORDER
        ]
        ax.bar(
            x + (idx - 1.5) * width,
            vals,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANT_LABELS)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        handlelength=1.2,
        columnspacing=0.8,
    )
    clean_axes(ax)
    save_all(fig, out_dir, "fig_graph_update_overall_bars")


def plot_scenario_error(scenario: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    scenarios = ["T1", "T2", "T3", "T4"]
    selected = ["static_pre", "naive_during", "rule_edge", "llm_full"]
    x = np.arange(len(scenarios))
    width = 0.18
    for idx, variant in enumerate(selected):
        vals = []
        for scenario_id in scenarios:
            row = scenario[(scenario.scenario_id == scenario_id) & (scenario.variant == variant)]
            vals.append(1.0 - float(row.iloc[0]["top1"]))
        ax.bar(
            x + (idx - 1.5) * width,
            vals,
            width=width,
            label=VARIANT_SHORT[variant],
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_ylabel("Error rate (1 - Top@1)")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        handlelength=1.2,
        columnspacing=0.8,
    )
    clean_axes(ax)
    save_all(fig, out_dir, "fig_graph_update_scenario_error")


def plot_graph_update_quality(graph_update: pd.DataFrame, out_dir: Path) -> None:
    quality = [
        ("Validity", graph_update["operation_validity_rate"].mean()),
        ("Exec.\nsuccess", graph_update["plan_execution_success_rate"].mean()),
        ("Node F1\nT1/T4", graph_update["node_add_remove_f1"].dropna().mean()),
        ("Align acc.\nT2", graph_update["node_alignment_accuracy"].dropna().mean()),
        ("Redirect\nrecall T3", graph_update["edge_redirect_recall"].dropna().mean()),
        ("Version\nrecall T4", graph_update["version_split_recall"].dropna().mean()),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    labels = [item[0] for item in quality]
    vals = [item[1] for item in quality]
    bar_colors = [
        "#16A34A" if value >= 0.95 else "#D97706" if value >= 0.65 else "#DC2626"
        for value in vals
    ]
    x = np.arange(len(vals))
    ax.bar(
        x,
        vals,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.7,
        width=0.68,
        label="Graph update quality",
    )
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.2)
    clean_axes(ax)
    save_all(fig, out_dir, "fig_graph_update_quality")


def write_manifest(out_dir: Path) -> None:
    manifest = pd.DataFrame([
        {
            "figure": "fig_graph_update_overall_bars",
            "description": "Overall Top@1/MRR grouped bars by graph variant and model.",
        },
        {
            "figure": "fig_graph_update_scenario_error",
            "description": "Scenario residual error grouped bars for selected GraphSAGE graph variants.",
        },
        {
            "figure": "fig_graph_update_quality",
            "description": "Graph update quality metrics.",
        },
    ])
    path = out_dir / "manifest.csv"
    manifest.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
