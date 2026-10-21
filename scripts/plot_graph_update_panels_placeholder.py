#!/usr/bin/env python
"""Create placeholder graph-update effect panels.

These figures are for layout only. The generated CSV and figure files are
explicitly named `placeholder` and use synthetic values. Do not use them as
experimental evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = ["static_pre", "naive_during", "rule_node", "rule_edge", "rule_full", "llm_full"]
VARIANT_LABELS = ["Static\npre", "Naive\nduring", "Rule\nnode", "Rule\nedge", "Rule\nfull", "LLM\nfull"]
SCENARIOS = ["T1", "T2", "T3", "T4"]
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
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    overall = write_or_load_placeholder_overall(data_dir / "placeholder_overall.csv", force=args.force)
    scenario = write_or_load_placeholder_scenario(data_dir / "placeholder_scenario_error.csv", force=args.force)
    quality = write_or_load_placeholder_quality(data_dir / "placeholder_quality.csv", force=args.force)

    plot_overall(overall, out_dir)
    plot_scenario_error(scenario, out_dir)
    plot_quality(quality, out_dir)
    write_manifest(out_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot placeholder graph-update panels.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/dygraph_rca_full/figures/graph_update_panels_placeholder",
        help="Output directory for placeholder CSVs and figures.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing placeholder CSV files before plotting.",
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


def write_or_load_placeholder_overall(path: Path, force: bool = False) -> pd.DataFrame:
    if path.exists() and not force:
        return pd.read_csv(path)
    rows = [
        ("static_pre", 0.34, 0.46, 0.40, 0.50),
        ("naive_during", 0.49, 0.58, 0.53, 0.61),
        ("rule_node", 0.58, 0.66, 0.62, 0.68),
        ("rule_edge", 0.70, 0.75, 0.73, 0.77),
        ("rule_full", 0.76, 0.81, 0.78, 0.82),
        ("llm_full", 0.88, 0.91, 0.86, 0.89),
    ]
    frame = pd.DataFrame(rows, columns=[
        "variant",
        "graphsage_top1",
        "graphsage_mrr",
        "gat_top1",
        "gat_mrr",
    ])
    frame["data_type"] = "synthetic_placeholder"
    frame.to_csv(path, index=False)
    return frame


def write_or_load_placeholder_scenario(path: Path, force: bool = False) -> pd.DataFrame:
    if path.exists() and not force:
        return pd.read_csv(path)
    values = {
        "static_pre": [0.46, 0.55, 0.72, 0.60],
        "naive_during": [0.25, 0.40, 0.54, 0.34],
        "rule_edge": [0.17, 0.22, 0.35, 0.20],
        "llm_full": [0.08, 0.10, 0.16, 0.09],
    }
    rows = []
    for variant, errors in values.items():
        for scenario, error in zip(SCENARIOS, errors):
            rows.append({
                "variant": variant,
                "scenario_id": scenario,
                "error_rate": error,
                "top1": 1.0 - error,
                "data_type": "synthetic_placeholder",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def write_or_load_placeholder_quality(path: Path, force: bool = False) -> pd.DataFrame:
    if path.exists() and not force:
        return pd.read_csv(path)
    rows = [
        ("Validity", 0.98),
        ("Exec.\nsuccess", 0.97),
        ("Node F1\nT1/T4", 0.84),
        ("Align acc.\nT2", 0.78),
        ("Redirect\nrecall T3", 0.86),
        ("Version\nrecall T4", 0.62),
    ]
    frame = pd.DataFrame(rows, columns=["metric", "value"])
    frame["data_type"] = "synthetic_placeholder"
    frame.to_csv(path, index=False)
    return frame


def clean_axes(ax) -> None:
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.text(
        0.5,
        0.54,
        "PLACEHOLDER",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=30,
        color="#CBD5E1",
        alpha=0.28,
        rotation=22,
        fontweight="bold",
        zorder=0,
    )


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


def plot_overall(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(len(VARIANTS))
    width = 0.18
    series = [
        ("GraphSAGE Top@1", "graphsage_top1", COLORS["sage"], 0),
        ("GraphSAGE MRR", "graphsage_mrr", COLORS["sage_mrr"], 1),
        ("GAT Top@1", "gat_top1", COLORS["gat"], 2),
        ("GAT MRR", "gat_mrr", COLORS["gat_mrr"], 3),
    ]
    for label, column, color, idx in series:
        vals = [float(frame[frame.variant == variant].iloc[0][column]) for variant in VARIANTS]
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
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    clean_axes(ax)
    save_all(fig, out_dir, "placeholder_fig_graph_update_overall_bars")


def plot_scenario_error(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    selected = ["static_pre", "naive_during", "rule_edge", "llm_full"]
    x = np.arange(len(SCENARIOS))
    width = 0.18
    for idx, variant in enumerate(selected):
        vals = [
            float(frame[(frame.variant == variant) & (frame.scenario_id == scenario)].iloc[0]["error_rate"])
            for scenario in SCENARIOS
        ]
        ax.bar(
            x + (idx - 1.5) * width,
            vals,
            width=width,
            label=variant.replace("_", " "),
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_ylabel("Error rate (1 - Top@1)")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    clean_axes(ax)
    save_all(fig, out_dir, "placeholder_fig_graph_update_scenario_error")


def plot_quality(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    vals = frame["value"].astype(float).tolist()
    labels = frame["metric"].tolist()
    colors = ["#16A34A" if value >= 0.9 else "#D97706" if value >= 0.65 else "#DC2626" for value in vals]
    x = np.arange(len(vals))
    ax.bar(
        x,
        vals,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        width=0.68,
        label="Graph update quality",
    )
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    clean_axes(ax)
    save_all(fig, out_dir, "placeholder_fig_graph_update_quality")


def write_manifest(out_dir: Path) -> None:
    manifest = pd.DataFrame([
        {
            "figure": "placeholder_fig_graph_update_overall_bars",
            "data": "data/placeholder_overall.csv",
            "data_type": "synthetic_placeholder",
        },
        {
            "figure": "placeholder_fig_graph_update_scenario_error",
            "data": "data/placeholder_scenario_error.csv",
            "data_type": "synthetic_placeholder",
        },
        {
            "figure": "placeholder_fig_graph_update_quality",
            "data": "data/placeholder_quality.csv",
            "data_type": "synthetic_placeholder",
        },
    ])
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "README.md").write_text(
        "# Placeholder graph-update figures\n\n"
        "These figures use synthetic placeholder values for layout only. "
        "Do not report them as experimental results. Replace the CSV files under "
        "`data/` with measured values and rerun `scripts/plot_graph_update_panels_placeholder.py`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
