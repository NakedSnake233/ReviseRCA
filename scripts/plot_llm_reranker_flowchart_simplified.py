#!/usr/bin/env python3
"""Draw a simplified LLM evidence reranker flowchart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


BLUE = "#D8ECFF"
PURPLE = "#E7DBFF"
ORANGE = "#FFE4C7"
GREEN = "#DDF3DF"
GRAY = "#EEEEEE"
YELLOW = "#FFF5C2"
EDGE = "#333333"
TEXT = "#111111"


def add_box(ax, xy, wh, text, facecolor, shape="process"):
    x, y = xy
    w, h = wh
    if shape == "decision":
        patch = Polygon(
            [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)],
            closed=True,
            facecolor=facecolor,
            edgecolor=EDGE,
            linewidth=1.25,
        )
    elif shape == "terminator":
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=facecolor,
            edgecolor=EDGE,
            linewidth=1.25,
        )
    else:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=facecolor,
            edgecolor=EDGE,
            linewidth=1.25,
        )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=8.0, color=TEXT, linespacing=1.12)
    return {"x": x, "y": y, "w": w, "h": h}


def anchor(node, side):
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    if side == "top":
        return x + w / 2, y + h
    if side == "bottom":
        return x + w / 2, y
    if side == "left":
        return x, y + h / 2
    if side == "right":
        return x + w, y + h / 2
    raise ValueError(side)


def arrow(ax, src, src_side, dst, dst_side, label="", via=None):
    p1 = anchor(src, src_side)
    p2 = anchor(dst, dst_side)
    points = [p1] + list(via or []) + [p2]
    for idx in range(len(points) - 1):
        patch = FancyArrowPatch(
            points[idx],
            points[idx + 1],
            arrowstyle="-|>" if idx == len(points) - 2 else "-",
            mutation_scale=11,
            linewidth=1.05,
            color=EDGE,
            shrinkA=2,
            shrinkB=2,
        )
        ax.add_patch(patch)
    if label:
        lx, ly = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        if via:
            lx, ly = via[0]
        ax.text(lx, ly, label, fontsize=7.5, ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})


def draw(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.8, 13.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 16)
    ax.axis("off")

    nodes = {}
    nodes["start"] = add_box(ax, (3.7, 14.75), (4.6, 0.65),
                             "Start\nGNN Top-K + graph + experiment", GREEN, "terminator")
    nodes["row_loop"] = add_box(ax, (4.8, 13.45), (2.4, 1.05),
                                "More\nprediction\nrows?", YELLOW, "decision")
    nodes["end"] = add_box(ax, (8.65, 13.65), (2.45, 0.65),
                           "End\nbefore/after metrics", GREEN, "terminator")
    nodes["evidence"] = add_box(ax, (3.35, 12.15), (5.3, 0.72),
                                "Build Evidence Pack\ntopology, traces, metrics, logs", BLUE)
    nodes["llm"] = add_box(ax, (3.35, 11.0), (5.3, 0.72),
                           "LLM Evidence Audit\nmodalities + causal direction", PURPLE)
    nodes["valid"] = add_box(ax, (4.65, 9.55), (2.7, 1.05),
                             "Valid LLM JSON\nand evidence refs?", YELLOW, "decision")

    nodes["use_llm"] = add_box(ax, (1.3, 8.35), (3.35, 0.72),
                               "Use validated\nLLM ranking", PURPLE)
    nodes["fallback"] = add_box(ax, (7.35, 8.35), (3.25, 0.72),
                                "Rule-Based Fallback", GRAY)
    nodes["cand_loop"] = add_box(ax, (7.7, 7.05), (2.55, 1.05),
                                 "More Top-K\ncandidates?", YELLOW, "decision")
    nodes["score"] = add_box(ax, (7.1, 5.85), (3.75, 0.72),
                             "Score candidate\nmodel score + scenario boosts\n- bounded counter penalty", GRAY)
    nodes["sort"] = add_box(ax, (7.35, 4.75), (3.25, 0.68),
                            "Sort by rule score", GRAY)

    nodes["guard"] = add_box(ax, (3.35, 3.85), (5.3, 0.72),
                             "Safety Guard\npreserve GNN order unless promotion is justified", ORANGE)
    nodes["promotion"] = add_box(ax, (4.65, 2.45), (2.7, 1.05),
                                 "Promotion\njustified?", YELLOW, "decision")
    nodes["accept"] = add_box(ax, (0.8, 1.35), (3.35, 0.72),
                              "Accept promotion\ntarget prior or\ncausal evidence", ORANGE)
    nodes["preserve"] = add_box(ax, (7.65, 1.35), (3.35, 0.72),
                                "Preserve original GNN order", ORANGE)
    nodes["final"] = add_box(ax, (3.35, 0.65), (5.3, 0.72),
                             "Finalize reranked Top-K", GREEN)
    nodes["write"] = add_box(ax, (3.35, -0.35), (5.3, 0.72),
                             "Write outputs\nreranked_predictions.csv and explanations.jsonl", GREEN)

    arrow(ax, nodes["start"], "bottom", nodes["row_loop"], "top")
    arrow(ax, nodes["row_loop"], "right", nodes["end"], "left", "No")
    arrow(ax, nodes["row_loop"], "bottom", nodes["evidence"], "top", "Yes")
    arrow(ax, nodes["evidence"], "bottom", nodes["llm"], "top")
    arrow(ax, nodes["llm"], "bottom", nodes["valid"], "top")
    arrow(ax, nodes["valid"], "left", nodes["use_llm"], "top", "Yes",
          via=[(3.4, 10.08), (2.98, 9.2)])
    arrow(ax, nodes["valid"], "right", nodes["fallback"], "top", "No",
          via=[(8.55, 10.08), (8.98, 9.2)])

    arrow(ax, nodes["fallback"], "bottom", nodes["cand_loop"], "top")
    arrow(ax, nodes["cand_loop"], "bottom", nodes["score"], "top", "Yes")
    arrow(ax, nodes["score"], "right", nodes["cand_loop"], "right", "next",
          via=[(11.35, 6.21), (11.35, 7.58)])
    arrow(ax, nodes["cand_loop"], "left", nodes["sort"], "top", "No",
          via=[(6.95, 7.58), (6.95, 5.65), (8.98, 5.65)])
    arrow(ax, nodes["use_llm"], "bottom", nodes["guard"], "top",
          via=[(2.98, 7.9), (2.98, 4.9), (6.0, 4.9)])
    arrow(ax, nodes["sort"], "bottom", nodes["guard"], "top",
          via=[(8.98, 4.65), (6.0, 4.65)])

    arrow(ax, nodes["guard"], "bottom", nodes["promotion"], "top")
    arrow(ax, nodes["promotion"], "left", nodes["accept"], "right", "Yes")
    arrow(ax, nodes["promotion"], "right", nodes["preserve"], "top", "No",
          via=[(8.5, 2.98), (9.32, 2.2)])
    arrow(ax, nodes["accept"], "right", nodes["final"], "left")
    arrow(ax, nodes["preserve"], "left", nodes["final"], "right")
    arrow(ax, nodes["final"], "bottom", nodes["write"], "top")
    arrow(ax, nodes["write"], "left", nodes["row_loop"], "left", "next row",
          via=[(0.25, 0.01), (0.25, 13.98)])

    ax.text(6.0, 15.72, "Simplified LLM Evidence Reranker Flowchart",
            ha="center", va="center", fontsize=12.5, fontweight="bold")

    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"llm_evidence_reranker_flowchart_simplified.{suffix}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the simplified LLM evidence reranker flowchart.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/dygraph_rca_full/figures/llm_reranker_flowchart",
        help="Output directory for simplified PDF/PNG/SVG flowchart files.",
    )
    args = parser.parse_args()
    draw(Path(args.out_dir))
    print(f"wrote simplified flowchart -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
