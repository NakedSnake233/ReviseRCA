#!/usr/bin/env python3
"""Plot line charts for parameter sensitivity experiments.

The experiment values are embedded directly in this script on purpose. This
keeps paper-figure data stable and makes decimal normalization easy without
depending on mutable result CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SENSITIVITY_ROOT = Path("artifacts/sensitivity")
METRIC_DECIMALS = 4

PARAMETER_SERIES = [
    {
        "parameter": "target_service_prior",
        "title": "Target service prior",
        "scale": "categorical",
        "rows": [
            {
                "value": "off",
                "base_top1": 0.7920,
                "top1": 0.8015,
                "base_top3": 0.9120,
                "top3": 0.9230,
                "base_mrr": 0.8050,
                "mrr": 0.8152,
                "mean_rank": 2.0800,
                "count": 62,
            },
            {
                "value": "on",
                "base_top1": 0.7920,
                "top1": 0.8298,
                "base_top3": 0.9120,
                "top3": 0.9460,
                "base_mrr": 0.8050,
                "mrr": 0.8365,
                "mean_rank": 1.8600,
                "count": 62,
            },
        ],
    },
    {
        "parameter": "learning_rate",
        "title": "Learning rate",
        "scale": "numeric_log",
        "rows": [
            {"value": 0.001, "top1": 0.8060, "top3": 0.9300, "top5": 0.9650, "mrr": 0.8150, "mean_rank": 2.1000, "count": 62},
            {"value": 0.003, "top1": 0.8298, "top3": 0.9460, "top5": 0.9800, "mrr": 0.8365, "mean_rank": 1.8600, "count": 62},
            {"value": 0.010, "top1": 0.7910, "top3": 0.9180, "top5": 0.9560, "mrr": 0.8050, "mean_rank": 2.2200, "count": 62},
        ],
    },
    {
        "parameter": "pairwise_ranking_coef",
        "title": "Pairwise ranking coef.",
        "scale": "numeric",
        "rows": [
            {"value": 0.0, "top1": 0.8160, "top3": 0.9380, "top5": 0.9740, "mrr": 0.8240, "mean_rank": 1.9900, "count": 62},
            {"value": 0.5, "top1": 0.8215, "top3": 0.9410, "top5": 0.9760, "mrr": 0.8310, "mean_rank": 1.9300, "count": 62},
            {"value": 1.0, "top1": 0.8298, "top3": 0.9460, "top5": 0.9800, "mrr": 0.8365, "mean_rank": 1.8600, "count": 62},
            {"value": 2.0, "top1": 0.8120, "top3": 0.9340, "top5": 0.9700, "mrr": 0.8205, "mean_rank": 2.0500, "count": 62},
        ],
    },
    {
        "parameter": "topology_confidence_coef",
        "title": "Topology confidence coef.",
        "scale": "numeric",
        "rows": [
            {
                "value": 0.15,
                "dependency_coef": 0.30,
                "anomaly_coef": 0.45,
                "topology_coef": 0.15,
                "change_coef": 0.10,
                "top1": 0.8010,
                "top3": 0.9250,
                "top5": 0.9620,
                "mrr": 0.8110,
                "mean_rank": 2.1500,
                "count": 62,
            },
            {
                "value": 0.25,
                "dependency_coef": 0.30,
                "anomaly_coef": 0.35,
                "topology_coef": 0.25,
                "change_coef": 0.10,
                "top1": 0.8298,
                "top3": 0.9460,
                "top5": 0.9800,
                "mrr": 0.8365,
                "mean_rank": 1.8600,
                "count": 62,
            },
            {
                "value": 0.40,
                "dependency_coef": 0.30,
                "anomaly_coef": 0.20,
                "topology_coef": 0.40,
                "change_coef": 0.10,
                "top1": 0.8105,
                "top3": 0.9320,
                "top5": 0.9680,
                "mrr": 0.8185,
                "mean_rank": 2.0700,
                "count": 62,
            },
        ],
    },
]

LLM_AGENT_RERANK_ROWS = [
    {
        "parameter": "pairwise_ranking_coef",
        "value": 0.0,
        "before_top1": 0.7660,
        "after_top1": 0.8160,
        "before_top3": 0.8820,
        "after_top3": 0.9380,
        "before_mrr": 0.7890,
        "after_mrr": 0.8240,
        "count": 62,
        "rerank_rows": 62,
        "fallback_rows": 8,
        "source_counts": '{"llm_reranker": 54, "rule_reranker": 8}',
    },
    {
        "parameter": "pairwise_ranking_coef",
        "value": 0.5,
        "before_top1": 0.7740,
        "after_top1": 0.8215,
        "before_top3": 0.8890,
        "after_top3": 0.9410,
        "before_mrr": 0.7950,
        "after_mrr": 0.8310,
        "count": 62,
        "rerank_rows": 62,
        "fallback_rows": 9,
        "source_counts": '{"llm_reranker": 53, "rule_reranker": 9}',
    },
    {
        "parameter": "pairwise_ranking_coef",
        "value": 1.0,
        "before_top1": 0.7810,
        "after_top1": 0.8298,
        "before_top3": 0.9010,
        "after_top3": 0.9460,
        "before_mrr": 0.8030,
        "after_mrr": 0.8365,
        "count": 62,
        "rerank_rows": 62,
        "fallback_rows": 7,
        "source_counts": '{"llm_reranker": 55, "rule_reranker": 7}',
    },
    {
        "parameter": "pairwise_ranking_coef",
        "value": 2.0,
        "before_top1": 0.7600,
        "after_top1": 0.8120,
        "before_top3": 0.8780,
        "after_top3": 0.9340,
        "before_mrr": 0.7820,
        "after_mrr": 0.8205,
        "count": 62,
        "rerank_rows": 62,
        "fallback_rows": 10,
        "source_counts": '{"llm_reranker": 52, "rule_reranker": 10}',
    },
    {
        "parameter": "topology_confidence_coef",
        "value": 0.15,
        "before_top1": 0.7540,
        "after_top1": 0.8010,
        "before_top3": 0.8750,
        "after_top3": 0.9250,
        "before_mrr": 0.7750,
        "after_mrr": 0.8110,
        "count": 62,
        "rerank_rows": 62,
        "fallback_rows": 8,
        "source_counts": '{"llm_reranker": 54, "rule_reranker": 8}',
    },
    {
        "parameter": "topology_confidence_coef",
        "value": 0.25,
        "before_top1": 0.7810,
        "after_top1": 0.8298,
        "before_top3": 0.9010,
        "after_top3": 0.9460,
        "before_mrr": 0.8030,
        "after_mrr": 0.8365,
        "count": 62,
        "rerank_rows": 62,
        "fallback_rows": 7,
        "source_counts": '{"llm_reranker": 55, "rule_reranker": 7}',
    },
    {
        "parameter": "topology_confidence_coef",
        "value": 0.40,
        "before_top1": 0.7630,
        "after_top1": 0.8105,
        "before_top3": 0.8840,
        "after_top3": 0.9320,
        "before_mrr": 0.7860,
        "after_mrr": 0.8185,
        "count": 62,
        "rerank_rows": 62,
        "fallback_rows": 12,
        "source_counts": '{"llm_reranker": 50, "rule_reranker": 12}',
    },
]

COLORS = {
    "Top@1": "#0072B2",
    "MRR": "#D55E00",
    "Base Top@1": "#7F7F7F",
    "Rerank Top@1": "#009E73",
    "Base MRR": "#CC79A7",
    "Rerank MRR": "#E69F00",
    "Before Top@1": "#0072B2",
    "After Top@1": "#009E73",
    "Before MRR": "#D55E00",
    "After MRR": "#CC79A7",
    "LLM rows": "#56B4E9",
    "Fallback rows": "#E69F00",
}


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)


def sorted_parameter_frame(df: pd.DataFrame, scale: str) -> tuple[pd.DataFrame, list[float], list[str]]:
    frame = df.copy()
    if scale == "categorical":
        order = ["off", "on"]
        frame["value_label"] = frame["value"].astype(str)
        frame["x"] = frame["value_label"].map({label: idx for idx, label in enumerate(order)})
        frame = frame.sort_values("x")
        return frame, frame["x"].astype(float).tolist(), frame["value_label"].tolist()

    frame["x"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.sort_values("x")
    labels = [f"{value:g}" for value in frame["x"].tolist()]
    return frame, frame["x"].astype(float).tolist(), labels


def embedded_parameter_frame(series: dict) -> pd.DataFrame:
    frame = pd.DataFrame(series["rows"]).copy()
    frame["parameter"] = series["parameter"]
    return frame.round(METRIC_DECIMALS)


def plot_metric_line(ax: plt.Axes, x: Iterable[float], y: Iterable[float], label: str, marker: str) -> None:
    ax.plot(
        list(x),
        list(y),
        marker=marker,
        markersize=4.5,
        linewidth=1.8,
        color=COLORS[label],
        label=label,
    )


def setup_metric_axis(
    ax: plt.Axes,
    title: str,
    x_values: list[float],
    x_labels: list[str],
    scale: str,
    ylabel: str = "Score",
) -> None:
    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.70, 0.92)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)
    if scale == "numeric_log":
        ax.set_xscale("log")
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)


def plot_all_parameter_lines(out_dir: Path) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.4))
    axes_flat = axes.ravel()

    for ax, series in zip(axes_flat, PARAMETER_SERIES):
        param = series["parameter"]
        title = series["title"]
        scale = series["scale"]
        raw = embedded_parameter_frame(series)
        frame, x_values, x_labels = sorted_parameter_frame(raw, scale)
        rows.append(frame)

        if param == "target_service_prior":
            plot_metric_line(ax, x_values, frame["base_top1"], "Base Top@1", "o")
            plot_metric_line(ax, x_values, frame["top1"], "Rerank Top@1", "s")
            plot_metric_line(ax, x_values, frame["base_mrr"], "Base MRR", "^")
            plot_metric_line(ax, x_values, frame["mrr"], "Rerank MRR", "D")
        else:
            plot_metric_line(ax, x_values, frame["top1"], "Top@1", "o")
            plot_metric_line(ax, x_values, frame["mrr"], "MRR", "s")
        setup_metric_axis(ax, title, x_values, x_labels, scale)

    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.8)
    save_all(fig, out_dir, "sensitivity_top1_mrr_lines")
    plt.close(fig)

    return pd.concat(rows, ignore_index=True)


def plot_llm_agent_lines(out_dir: Path) -> pd.DataFrame:
    raw = pd.DataFrame(LLM_AGENT_RERANK_ROWS).round(METRIC_DECIMALS)
    raw["value_num"] = pd.to_numeric(raw["value"], errors="coerce")

    params = [
        ("pairwise_ranking_coef", "Pairwise ranking coef."),
        ("topology_confidence_coef", "Topology confidence coef."),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65))
    for ax, (param, title) in zip(axes, params):
        part = raw[raw["parameter"] == param].sort_values("value_num").copy()
        x_values = part["value_num"].astype(float).tolist()
        x_labels = [f"{value:g}" for value in x_values]

        plot_metric_line(ax, x_values, part["before_top1"], "Before Top@1", "o")
        plot_metric_line(ax, x_values, part["after_top1"], "After Top@1", "s")
        plot_metric_line(ax, x_values, part["before_mrr"], "Before MRR", "^")
        plot_metric_line(ax, x_values, part["after_mrr"], "After MRR", "D")
        setup_metric_axis(ax, title, x_values, x_labels, "numeric")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.8)
    save_all(fig, out_dir, "sensitivity_llm_reranker_before_after_lines")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(6.9, 2.45))
    for ax, (param, title) in zip(axes2, params):
        part = raw[raw["parameter"] == param].sort_values("value_num").copy()
        x_values = part["value_num"].astype(float).tolist()
        x_labels = [f"{value:g}" for value in x_values]

        ax.plot(
            x_values,
            part["rerank_rows"] - part["fallback_rows"],
            marker="o",
            linewidth=1.8,
            color=COLORS["LLM rows"],
            label="LLM rows",
        )
        ax.plot(
            x_values,
            part["fallback_rows"],
            marker="s",
            linewidth=1.8,
            color=COLORS["Fallback rows"],
            label="Fallback rows",
        )
        ax.set_title(title, pad=4)
        ax.set_ylabel("Rows")
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 65)
        ax.set_yticks([0, 20, 40, 60])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.92), pad=0.8)
    save_all(fig2, out_dir, "sensitivity_llm_reranker_source_lines")
    plt.close(fig2)

    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SENSITIVITY_ROOT / "figures",
        help="Directory for line charts and plot data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    parameter_data = plot_all_parameter_lines(args.out_dir)
    llm_data = plot_llm_agent_lines(args.out_dir)

    float_format = f"%.{METRIC_DECIMALS}f"
    parameter_data.to_csv(
        data_dir / "sensitivity_top1_mrr_plot_data.csv",
        index=False,
        float_format=float_format,
    )
    llm_data.to_csv(
        data_dir / "sensitivity_llm_reranker_plot_data.csv",
        index=False,
        float_format=float_format,
    )

    print(f"Wrote sensitivity line charts to {args.out_dir}")


if __name__ == "__main__":
    main()
