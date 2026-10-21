#!/usr/bin/env python3
"""Draw a standard flowchart for the LLM evidence reranker pipeline."""

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


def add_box(ax, name, xy, wh, text, facecolor, shape="process"):
    x, y = xy
    w, h = wh
    if shape == "decision":
        patch = Polygon(
            [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)],
            closed=True,
            facecolor=facecolor,
            edgecolor=EDGE,
            linewidth=1.2,
        )
        ax.add_patch(patch)
    elif shape == "terminator":
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor=facecolor,
            edgecolor=EDGE,
            linewidth=1.2,
        )
        ax.add_patch(patch)
    else:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=facecolor,
            edgecolor=EDGE,
            linewidth=1.2,
        )
        ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.2,
        color=TEXT,
        linespacing=1.15,
    )
    return {"name": name, "x": x, "y": y, "w": w, "h": h}


def anchor(node, side):
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    if side == "top":
        return (x + w / 2, y + h)
    if side == "bottom":
        return (x + w / 2, y)
    if side == "left":
        return (x, y + h / 2)
    if side == "right":
        return (x + w, y + h / 2)
    raise ValueError(side)


def arrow(ax, src, src_side, dst, dst_side, label="", rad=0.0, via=None):
    p1 = anchor(src, src_side)
    p2 = anchor(dst, dst_side)
    if via:
        points = [p1] + list(via) + [p2]
        for i in range(len(points) - 1):
            end_arrow = "-|>" if i == len(points) - 2 else "-"
            patch = FancyArrowPatch(
                points[i],
                points[i + 1],
                arrowstyle=end_arrow,
                mutation_scale=11,
                linewidth=1.0,
                color=EDGE,
                shrinkA=2,
                shrinkB=2,
            )
            ax.add_patch(patch)
    else:
        patch = FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=EDGE,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)
    if label:
        lx = (p1[0] + p2[0]) / 2
        ly = (p1[1] + p2[1]) / 2
        if via:
            lx, ly = via[0]
        ax.text(
            lx,
            ly,
            label,
            fontsize=7.4,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )


def draw(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 15.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 20)
    ax.axis("off")

    nodes = {}
    nodes["start"] = add_box(
        ax,
        "start",
        (4.25, 18.35),
        (3.5, 0.72),
        "Start\npredictions.csv + experiment_dir + executed graph",
        GREEN,
        "terminator",
    )
    nodes["loop"] = add_box(ax, "loop", (4.9, 17.0), (2.2, 1.15), "More\nprediction\nrows?", YELLOW, "decision")
    nodes["evidence"] = add_box(ax, "evidence", (4.4, 15.55), (3.2, 0.95), "Build Evidence Pack\nTop-K candidates\nmodel score + rank", BLUE)
    nodes["modalities"] = add_box(ax, "modalities", (4.4, 14.25), (3.2, 0.95), "Evidence Audit Inputs\ntopology, traces, metrics,\nlogs, temporal, counter", BLUE)
    nodes["prompt"] = add_box(ax, "prompt", (4.4, 12.95), (3.2, 0.95), "Build LLM Prompt\nJSON only\nno new evidence / no HPO", PURPLE)
    nodes["api"] = add_box(ax, "api", (4.9, 11.55), (2.2, 1.15), "LLM API\navailable?", YELLOW, "decision")
    nodes["parse"] = add_box(ax, "parse", (4.4, 10.3), (3.2, 0.85), "Parse LLM JSON response", PURPLE)
    nodes["schema"] = add_box(ax, "schema", (4.9, 8.95), (2.2, 1.15), "Required\nschema\nvalid?", YELLOW, "decision")
    nodes["refs"] = add_box(ax, "refs", (4.85, 7.55), (2.3, 1.15), "Candidates and\nevidence refs\nvalid?", YELLOW, "decision")
    nodes["normalize"] = add_box(ax, "normalize", (4.4, 6.35), (3.2, 0.85), "Normalize ranking\nappend omitted Top-K", ORANGE)

    nodes["fallback"] = add_box(ax, "fallback", (8.45, 11.75), (2.65, 0.85), "Rule-Based Fallback\ntriggered per row", GRAY)
    nodes["cand_loop"] = add_box(ax, "cand_loop", (8.55, 10.3), (2.45, 1.15), "More Top-K\ncandidates\nto score?", YELLOW, "decision")
    nodes["score"] = add_box(ax, "score", (8.45, 9.0), (2.65, 0.85), "Start from\nGNN model score", GRAY)
    nodes["scenario"] = add_box(ax, "scenario", (8.55, 7.55), (2.45, 1.15), "Scenario-\nspecific\nevidence?", YELLOW, "decision")
    nodes["boost"] = add_box(ax, "boost", (8.15, 6.2), (3.05, 0.95), "Apply fixed boosts\nT3 mismatch/blackhole\nT4 v2 evidence\nT1 scale evidence", GRAY)
    nodes["penalty"] = add_box(ax, "penalty", (8.45, 4.9), (2.65, 0.85), "Apply bounded\ncounter-evidence penalty", GRAY)
    nodes["append_score"] = add_box(ax, "append_score", (8.45, 3.7), (2.65, 0.85), "Append scored candidate", GRAY)
    nodes["sort"] = add_box(ax, "sort", (8.45, 2.5), (2.65, 0.85), "Sort candidates by\nrule score", GRAY)

    nodes["guard"] = add_box(ax, "guard", (4.4, 5.15), (3.2, 0.85), "Safety Guard", ORANGE)
    nodes["prior"] = add_box(ax, "prior", (4.95, 3.85), (2.1, 1.1), "Target service\nprior applies?", YELLOW, "decision")
    nodes["changed"] = add_box(ax, "changed", (4.95, 2.55), (2.1, 1.1), "Ranking\nchanged?", YELLOW, "decision")
    nodes["allow"] = add_box(ax, "allow", (1.1, 2.55), (2.1, 1.1), "Promotions\nallowed?", YELLOW, "decision")
    nodes["direct"] = add_box(ax, "direct", (1.1, 1.15), (2.1, 1.1), "Direct causal\nevidence\nsufficient?", YELLOW, "decision")
    nodes["restore"] = add_box(ax, "restore", (4.4, 1.15), (3.2, 0.85), "Restore original GNN order", ORANGE)
    nodes["accept"] = add_box(ax, "accept", (0.55, 0.05), (3.2, 0.85), "Accept justified pairwise promotions", ORANGE)
    nodes["final"] = add_box(ax, "final", (4.4, 0.05), (3.2, 0.85), "Finalize guarded ranking", GREEN)
    nodes["write"] = add_box(ax, "write", (8.05, 0.05), (3.25, 0.85), "Write outputs\nreranked_predictions.csv\nexplanations.jsonl", GREEN)
    nodes["end"] = add_box(ax, "end", (8.45, 16.98), (2.65, 0.75), "End\nbefore/after metrics", GREEN, "terminator")

    arrow(ax, nodes["start"], "bottom", nodes["loop"], "top")
    arrow(ax, nodes["loop"], "bottom", nodes["evidence"], "top", "Yes")
    arrow(ax, nodes["loop"], "right", nodes["end"], "left", "No")
    arrow(ax, nodes["evidence"], "bottom", nodes["modalities"], "top")
    arrow(ax, nodes["modalities"], "bottom", nodes["prompt"], "top")
    arrow(ax, nodes["prompt"], "bottom", nodes["api"], "top")
    arrow(ax, nodes["api"], "bottom", nodes["parse"], "top", "Yes")
    arrow(ax, nodes["parse"], "bottom", nodes["schema"], "top")
    arrow(ax, nodes["schema"], "bottom", nodes["refs"], "top", "Yes")
    arrow(ax, nodes["refs"], "bottom", nodes["normalize"], "top", "Yes")
    arrow(ax, nodes["normalize"], "bottom", nodes["guard"], "top")

    arrow(ax, nodes["api"], "right", nodes["fallback"], "left", "No", via=[(8.0, 12.1)])
    arrow(ax, nodes["schema"], "right", nodes["fallback"], "left", "No", via=[(8.0, 9.55), (8.0, 12.1)])
    arrow(ax, nodes["refs"], "right", nodes["fallback"], "left", "No", via=[(8.0, 8.1), (8.0, 12.1)])
    arrow(ax, nodes["fallback"], "bottom", nodes["cand_loop"], "top")
    arrow(ax, nodes["cand_loop"], "bottom", nodes["score"], "top", "Yes")
    arrow(ax, nodes["score"], "bottom", nodes["scenario"], "top")
    arrow(ax, nodes["scenario"], "bottom", nodes["boost"], "top", "Yes")
    arrow(ax, nodes["scenario"], "right", nodes["penalty"], "right", "No", via=[(11.65, 8.1), (11.65, 5.35)])
    arrow(ax, nodes["boost"], "bottom", nodes["penalty"], "top")
    arrow(ax, nodes["penalty"], "bottom", nodes["append_score"], "top")
    arrow(ax, nodes["append_score"], "right", nodes["cand_loop"], "right", "next", via=[(11.55, 4.12), (11.55, 10.88)])
    arrow(ax, nodes["cand_loop"], "left", nodes["sort"], "top", "No", via=[(8.0, 10.88), (8.0, 3.55), (9.78, 3.55)])
    arrow(ax, nodes["sort"], "left", nodes["guard"], "right", via=[(8.0, 2.92), (8.0, 5.58)])

    arrow(ax, nodes["guard"], "bottom", nodes["prior"], "top")
    arrow(ax, nodes["prior"], "bottom", nodes["changed"], "top", "No")
    arrow(ax, nodes["prior"], "right", nodes["final"], "top", "Yes", via=[(7.85, 4.4), (7.85, 0.48)])
    arrow(ax, nodes["changed"], "left", nodes["allow"], "right")
    arrow(ax, nodes["changed"], "bottom", nodes["final"], "top", "No", via=[(6.0, 2.25), (6.0, 0.98)])
    arrow(ax, nodes["allow"], "bottom", nodes["direct"], "top", "Yes")
    arrow(ax, nodes["allow"], "right", nodes["restore"], "top", "No", via=[(3.65, 3.1), (3.65, 1.58)])
    arrow(ax, nodes["direct"], "right", nodes["restore"], "left", "No")
    arrow(ax, nodes["direct"], "bottom", nodes["accept"], "top", "Yes")
    arrow(ax, nodes["accept"], "right", nodes["final"], "left")
    arrow(ax, nodes["restore"], "bottom", nodes["final"], "top")
    arrow(ax, nodes["final"], "right", nodes["write"], "left")
    arrow(ax, nodes["write"], "right", nodes["loop"], "right", "next row", via=[(11.65, 0.48), (11.65, 17.58), (7.15, 17.58)])

    ax.text(
        6.0,
        19.55,
        "LLM Evidence Reranker with Schema Validation, Rule-Based Fallback, and Safety Guard",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"llm_evidence_reranker_flowchart.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the LLM evidence reranker flowchart.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/dygraph_rca_full/figures/llm_reranker_flowchart",
        help="Output directory for PDF/PNG/SVG flowchart files.",
    )
    args = parser.parse_args()
    draw(Path(args.out_dir))
    print(f"wrote flowchart -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
