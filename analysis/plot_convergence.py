#!/usr/bin/env python3
"""
Plot early-locking convergence curves.

If adjacent drift is available, produces a 2x2 figure with:
- IoU vs final
- Spearman vs final
- Fraction churn vs previous checkpoint
- L2 churn vs previous checkpoint
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _extract_method_series(entries: list[dict], method: str, top_k_key: str) -> tuple[list[int], list[float], list[float]]:
    steps = []
    ious = []
    spearmans = []
    for entry in entries:
        if entry["step"] is None:
            continue
        metrics = entry["per_topk"][method][top_k_key] if "per_topk" in entry else entry[method]
        steps.append(entry["step"])
        ious.append(metrics["iou"])
        spearmans.append(metrics["spearman"])
    return steps, ious, spearmans


def _extract_adjacent(entries: list[dict], method: str, top_k_key: str) -> tuple[list[int], list[float]]:
    steps = []
    churn = []
    for entry in entries:
        metrics = entry[method][top_k_key]
        steps.append(entry["step"])
        churn.append(metrics["churn_rate"])
    return steps, churn


def main():
    parser = argparse.ArgumentParser(description="Plot convergence curves")
    parser.add_argument("--input", type=str, default="convergence.json")
    parser.add_argument("--output", type=str, default="convergence_plot.pdf")
    parser.add_argument("--top_k_percent", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    top_k = args.top_k_percent if args.top_k_percent is not None else data["top_k_percent"]
    top_k_key = str(top_k)
    conv = data["convergence"]
    adjacent = data.get("adjacent_drift", [])

    frac_steps, frac_iou, frac_spearman = _extract_method_series(conv, "fraction", top_k_key)
    l2_steps, l2_iou, l2_spearman = _extract_method_series(conv, "l2", top_k_key)

    if not frac_steps:
        print("No entries with valid step numbers.")
        return

    frac_adj_steps, frac_churn = _extract_adjacent(adjacent, "fraction", top_k_key) if adjacent else ([], [])
    l2_adj_steps, l2_churn = _extract_adjacent(adjacent, "l2", top_k_key) if adjacent else ([], [])

    color_frac = "#D55E00"
    color_l2 = "#0072B2"

    if adjacent:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax1, ax2, ax3, ax4 = axes.flatten()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax3 = ax4 = None

    ax1.plot(frac_steps, frac_iou, "o-", color=color_frac, label="Fraction", markersize=4)
    ax1.plot(l2_steps, l2_iou, "s-", color=color_l2, label="L2", markersize=4)
    ax1.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="IoU = 0.85")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel(f"IoU with Final (top-{top_k}%)")
    ax1.set_title("Binary Mask Stability")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(frac_steps, frac_spearman, "o-", color=color_frac, label="Fraction", markersize=4)
    ax2.plot(l2_steps, l2_spearman, "s-", color=color_l2, label="L2", markersize=4)
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label=r"$\rho$ = 0.95")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel(r"Spearman $\rho$ with Final")
    ax2.set_title("Score Rank Stability")
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    if adjacent and ax3 is not None and ax4 is not None:
        ax3.plot(frac_adj_steps, frac_churn, "o-", color=color_frac, markersize=4)
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("Churn vs Previous")
        ax3.set_title("Fraction Step-to-Step Churn")
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)

        ax4.plot(l2_adj_steps, l2_churn, "s-", color=color_l2, markersize=4)
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Churn vs Previous")
        ax4.set_title("L2 Step-to-Step Churn")
        ax4.set_ylim(-0.05, 1.05)
        ax4.grid(True, alpha=0.3)

    fig.suptitle(f"Early-Locking Convergence (top-{top_k}% mask)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
