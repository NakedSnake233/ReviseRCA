#!/usr/bin/env python3
"""Plot graph update plan quality figures with embedded paper data.

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

OVERALL_METRICS = [
    "Op. validity",
    "Exec. success",
    "Node F1",
    "Alignment",
    "Redirect",
    "Version split",
    "Weight acc.",
]

RADAR_AXES = [
    "T1\nnode F1",
    "T2\nalign",
    "T3\nredirect",
    "T4\nversion",
    "T4\nweight acc.",
]

OVERALL_ROWS = [
    {
        "method": "LLM Planner",
        "Op. validity": 1.0000,
        "Exec. success": 1.0000,
        "Node F1": 0.9835,
        "Alignment": 0.9740,
        "Redirect": 1.0000,
        "Version split": 1.0000,
        "Weight acc.": 1.0000,
    },
    {
        "method": "Rule Planner",
        "Op. validity": 1.0000,
        "Exec. success": 1.0000,
        "Node F1": 0.4459,
        "Alignment": 0.9740,
        "Redirect": 0.9893,
        "Version split": 0.6667,
        "Weight acc.": 0.8533,
    },
]

RADAR_ROWS = [
    {
        "method": "Rule Planner",
        "T1\nnode F1": 1.0000,
        "T2\nalign": 1.0000,
        "T3\nredirect": 1.0000,
        "T4\nversion": 1.0000,
        "T4\nweight acc.": 1.0000,
    },
    {
        "method": "LLM Planner",
        "T1\nnode F1": 0.9791,
        "T2\nalign": 1.0000,
        "T3\nredirect": 0.9893,
        "T4\nversion": 0.6667,
        "T4\nweight acc.": 0.8533,
    },
]

COLORS = {
    "Rule Planner": "#0072B2",
    "LLM Planner": "#D55E00",
}

HATCHES = {
    "Rule Planner": "\\\\",
    "LLM Planner": "--",
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


def collect_overall() -> pd.DataFrame:
    return pd.DataFrame(OVERALL_ROWS).round(METRIC_DECIMALS)


def collect_radar() -> pd.DataFrame:
    return pd.DataFrame(RADAR_ROWS).round(METRIC_DECIMALS)


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)


def plot_overall_bars(frame: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(OVERALL_METRICS))
    methods = frame["method"].tolist()
    width = min(0.34, 0.78 / max(len(methods), 1))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(6.7, 2.55))
    for idx, method in enumerate(methods):
        row = frame[frame["method"] == method].iloc[0]
        values = [float(row[label]) for label in OVERALL_METRICS]
        ax.bar(
            x + offsets[idx],
            values,
            width=width,
            label=method,
            color=COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.4,
            hatch=HATCHES.get(method, ""),
        )

    ax.set_ylabel("Quality score")
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticks(x)
    ax.set_xticklabels(OVERALL_METRICS, rotation=25, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
    )
    fig.tight_layout(pad=0.4)
    save_all(fig, out_dir, "graph_update_plan_overall_bars")
    plt.close(fig)


def plot_radar(frame: pd.DataFrame, out_dir: Path) -> None:
    angles = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(4.25, 3.75), subplot_kw={"projection": "polar"})
    for _, row in frame.iterrows():
        method = str(row["method"])
        values = row[RADAR_AXES].astype(float).to_numpy()
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
    ax.set_xticklabels([label.replace("\n", " ") for label in RADAR_AXES])
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
    save_all(fig, out_dir, "graph_update_plan_scenario_radar")
    plt.close(fig)


def write_plot_data(overall: pd.DataFrame, radar: pd.DataFrame, out_dir: Path) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    float_format = f"%.{METRIC_DECIMALS}f"
    overall.to_csv(
        data_dir / "graph_update_plan_overall_plot_data.csv",
        index=False,
        float_format=float_format,
    )
    radar.to_csv(
        data_dir / "graph_update_plan_radar_plot_data.csv",
        index=False,
        float_format=float_format,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/dygraph_rca_full/figures/graph_update_plan_metrics"),
        help="Output directory for graph update plan metric figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall = collect_overall()
    radar = collect_radar()
    write_plot_data(overall, radar, args.out_dir)
    plot_overall_bars(overall, args.out_dir)
    plot_radar(radar, args.out_dir)
    print(f"wrote graph update plan metric figures -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
