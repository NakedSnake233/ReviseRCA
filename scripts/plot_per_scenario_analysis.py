#!/usr/bin/env python3
"""Plot per-scenario RCA analysis figures with embedded paper data.

The experiment values are embedded directly in this script on purpose. This
keeps paper-figure data stable and makes decimal normalization easy without
depending on mutable result CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_DECIMALS = 4

SCENARIOS = ["T1", "T2", "T3", "T4"]
SCENARIO_LABELS = {
    "T1": "T1\nScale",
    "T2": "T2\nDrift",
    "T3": "T3\nSelector",
    "T4": "T4\nCanary",
}
# ["Eadro", "HolisticRCA", "TAMO", "Plan2RCA"]
PLOT_ROWS = [
    # Baseline comparison.
    {"scenario_id": "T1", "top1": 0.110, "mrr": 0.230, "method": "Eadro", "group": "baseline"},
    {"scenario_id": "T2", "top1": 0.0992, "mrr": 0.199, "method": "Eadro", "group": "baseline"},
    {"scenario_id": "T3", "top1": 0.141, "mrr": 0.257, "method": "Eadro", "group": "baseline"},
    {"scenario_id": "T4", "top1": 0.125, "mrr": 0.240, "method": "Eadro", "group": "baseline"},
    {"scenario_id": "T1", "top1": 0.397, "mrr": 0.446, "method": "HolisticRCA", "group": "baseline"},
    {"scenario_id": "T2", "top1": 0.405, "mrr": 0.492, "method": "HolisticRCA", "group": "baseline"},
    {"scenario_id": "T3", "top1": 0.431, "mrr": 0.501, "method": "HolisticRCA", "group": "baseline"},
    {"scenario_id": "T4", "top1": 0.378, "mrr": 0.383, "method": "HolisticRCA", "group": "baseline"},
    {"scenario_id": "T1", "top1": 0.533, "mrr": 0.593, "method": "TAMO", "group": "baseline"},
    {"scenario_id": "T2", "top1": 0.562, "mrr": 0.635, "method": "TAMO", "group": "baseline"},
    {"scenario_id": "T3", "top1": 0.498, "mrr": 0.541, "method": "TAMO", "group": "baseline"},
    {"scenario_id": "T4", "top1": 0.540, "mrr": 0.606, "method": "TAMO", "group": "baseline"},
    {"scenario_id": "T1", "top1": 0.871, "mrr": 0.890, "method": "Plan2RCA", "group": "baseline"},
    {"scenario_id": "T2", "top1": 0.803, "mrr": 0.834, "method": "Plan2RCA", "group": "baseline"},
    {"scenario_id": "T3", "top1": 0.846, "mrr": 0.877, "method": "Plan2RCA", "group": "baseline"},
    {"scenario_id": "T4", "top1": 0.799, "mrr": 0.745, "method": "Plan2RCA", "group": "baseline"},

    # Graph-variant ablation.
    {"scenario_id": "T1", "top1": 1.0000, "mrr": 1.0000, "method": "DY-RCA + Reranker", "group": "graph_variant"},
    {"scenario_id": "T2", "top1": 1.0000, "mrr": 1.0000, "method": "DY-RCA + Reranker", "group": "graph_variant"},
    {"scenario_id": "T3", "top1": 1.0000, "mrr": 1.0000, "method": "DY-RCA + Reranker", "group": "graph_variant"},
    {"scenario_id": "T4", "top1": 1.0000, "mrr": 1.0000, "method": "DY-RCA + Reranker", "group": "graph_variant"},
    {"scenario_id": "T1", "top1": 1.0000, "mrr": 1.0000, "method": "LLM Full", "group": "graph_variant"},
    {"scenario_id": "T2", "top1": 1.0000, "mrr": 1.0000, "method": "LLM Full", "group": "graph_variant"},
    {"scenario_id": "T3", "top1": 0.8667, "mrr": 0.9333, "method": "LLM Full", "group": "graph_variant"},
    {"scenario_id": "T4", "top1": 1.0000, "mrr": 1.0000, "method": "LLM Full", "group": "graph_variant"},
    {"scenario_id": "T1", "top1": 1.0000, "mrr": 1.0000, "method": "Naive", "group": "graph_variant"},
    {"scenario_id": "T2", "top1": 0.1667, "mrr": 0.3730, "method": "Naive", "group": "graph_variant"},
    {"scenario_id": "T3", "top1": 0.6667, "mrr": 0.8000, "method": "Naive", "group": "graph_variant"},
    {"scenario_id": "T4", "top1": 0.6000, "mrr": 0.7583, "method": "Naive", "group": "graph_variant"},
    {"scenario_id": "T1", "top1": 1.0000, "mrr": 1.0000, "method": "Rule Full", "group": "graph_variant"},
    {"scenario_id": "T2", "top1": 1.0000, "mrr": 1.0000, "method": "Rule Full", "group": "graph_variant"},
    {"scenario_id": "T3", "top1": 0.5333, "mrr": 0.6544, "method": "Rule Full", "group": "graph_variant"},
    {"scenario_id": "T4", "top1": 1.0000, "mrr": 1.0000, "method": "Rule Full", "group": "graph_variant"},
    {"scenario_id": "T1", "top1": 0.2000, "mrr": 0.4711, "method": "Static", "group": "graph_variant"},
    {"scenario_id": "T2", "top1": 0.2500, "mrr": 0.4398, "method": "Static", "group": "graph_variant"},
    {"scenario_id": "T3", "top1": 0.8000, "mrr": 0.8778, "method": "Static", "group": "graph_variant"},
    {"scenario_id": "T4", "top1": 0.2500, "mrr": 0.5021, "method": "Static", "group": "graph_variant"},
]

COLORS = {
    "Static": "#4D4D4D",
    "Naive": "#E69F00",
    "Rule Full": "#0072B2",
    "LLM Full": "#009E73",
    "Eadro": "#CC79A7",
    "HolisticRCA": "#56B4E9",
    "TAMO": "#0072B2",
    "Plan2RCA": "#D55E00",
    "DY-RCA": "#009E73",
    "DY-RCA + Reranker": "#D55E00",
}

HATCHES = {
    "Static": "",
    "Naive": "//",
    "Rule Full": "\\\\",
    "LLM Full": "",
    "Eadro": "xx",
    "HolisticRCA": "..",
    "TAMO": "\\\\",
    "Plan2RCA": "--",
    "DY-RCA": "",
    "DY-RCA + Reranker": "--",
}


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def collect_plot_data() -> pd.DataFrame:
    data = pd.DataFrame(PLOT_ROWS).round(METRIC_DECIMALS)
    data["scenario_order"] = data["scenario_id"].map({name: idx for idx, name in enumerate(SCENARIOS)})
    return data.sort_values(["group", "method", "scenario_order"]).drop(columns=["scenario_order"])


def _scenario_matrix(data: pd.DataFrame, methods: Sequence[str], metric: str) -> pd.DataFrame:
    return (
        data[data["method"].isin(methods)]
        .pivot_table(index="scenario_id", columns="method", values=metric, aggfunc="mean")
        .reindex(SCENARIOS)
        .reindex(columns=methods)
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)


def plot_grouped_bars(
    data: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
) -> None:
    pivot = _scenario_matrix(data, methods, metric)
    x = np.arange(len(SCENARIOS))
    width = min(0.17, 0.78 / max(len(methods), 1))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(6.7, 2.55))
    for idx, method in enumerate(methods):
        ax.bar(
            x + offsets[idx],
            pivot[method].to_numpy(dtype=float),
            width=width,
            label=method,
            color=COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.4,
            hatch=HATCHES.get(method, ""),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=min(len(methods), 5),
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
    )
    fig.tight_layout(pad=0.4)
    save_all(fig, out_dir, stem)
    plt.close(fig)


def plot_radar(
    data: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
    out_dir: Path,
    stem: str,
) -> None:
    pivot = _scenario_matrix(data, methods, metric)
    angles = np.linspace(0, 2 * np.pi, len(SCENARIOS), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(4.25, 3.75), subplot_kw={"projection": "polar"})
    for method in methods:
        values = pivot[method].fillna(0.0).to_numpy(dtype=float)
        closed_values = np.r_[values, values[0]]
        ax.plot(
            closed_angles,
            closed_values,
            label=method,
            color=COLORS.get(method, "#999999"),
            linewidth=1.7,
            marker="o",
            markersize=3,
        )
        ax.fill(closed_angles, closed_values, color=COLORS.get(method, "#999999"), alpha=0.08)

    ax.set_xticks(angles)
    ax.set_xticklabels([SCENARIO_LABELS[s].replace("\n", " ") for s in SCENARIOS])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(color="#D9D9D9", linewidth=0.6)
    ax.spines["polar"].set_color("#666666")
    ax.spines["polar"].set_linewidth(0.8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.5)
    save_all(fig, out_dir, stem)
    plt.close(fig)


def write_plot_tables(data: pd.DataFrame, out_dir: Path) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    float_format = f"%.{METRIC_DECIMALS}f"
    data.to_csv(data_dir / "per_scenario_plot_data.csv", index=False, float_format=float_format)

    summary_rows = []
    for group, group_data in data.groupby("group"):
        methods = sorted(group_data["method"].unique())
        for metric in ("top1", "mrr"):
            pivot = _scenario_matrix(group_data, methods, metric)
            pivot = pivot.reset_index().rename(columns={"scenario_id": "scenario"})
            pivot.insert(0, "metric", metric)
            pivot.insert(0, "group", group)
            summary_rows.append(pivot)
    pd.concat(summary_rows, ignore_index=True).to_csv(
        data_dir / "per_scenario_plot_tables.csv",
        index=False,
        float_format=float_format,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/dygraph_rca_full/figures/per_scenario_analysis"),
        help="Directory for figures and plot data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = collect_plot_data()
    write_plot_tables(data, args.out_dir)

    ablation_methods = ["Static", "Naive", "Rule Full", "LLM Full", "DY-RCA + Reranker"]
    baseline_methods = ["Eadro", "HolisticRCA", "TAMO", "Plan2RCA"]


    plot_grouped_bars(
        data[data["group"] == "baseline"],
        baseline_methods,
        "mrr",
        "MRR",
        args.out_dir,
        "per_scenario_mrr_baseline_bars",
    )
    plot_grouped_bars(
        data[data["group"] == "baseline"],
        baseline_methods,
        "top1",
        "Top@1",
        args.out_dir,
        "per_scenario_top1_baseline_bars",
    )
    print(f"Wrote per-scenario figures to {args.out_dir}")


if __name__ == "__main__":
    main()
