#!/usr/bin/env python3
"""
Consensus analysis across multiple score or mask files.

Typical uses:
- cross-seed stability for one task
- cross-task consensus after selecting one checkpoint per task
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from common import (
    compute_iou,
    compute_score_correlation,
    extract_layer_module,
    load_importance_input,
    save_json,
)


def parse_inputs(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        pairs.append((label.strip(), path.strip()))
    return pairs


def maybe_plot_position_frequency(position_stats: dict, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping consensus heatmap")
        return

    layers = sorted({layer for layer, _ in position_stats})
    modules = sorted({module for _, module in position_stats})
    if not layers or not modules:
        return

    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}
    module_to_col = {module: idx for idx, module in enumerate(modules)}
    matrix = np.full((len(layers), len(modules)), np.nan, dtype=float)

    for (layer, module), stats in position_stats.items():
        matrix[layer_to_row[layer], module_to_col[module]] = stats["active_frequency"]

    fig, ax = plt.subplots(figsize=(max(6, len(modules) * 1.1), max(6, len(layers) * 0.25)))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Module Type")
    ax.set_ylabel("Layer Index")
    ax.set_title("Consensus Active Frequency")
    fig.colorbar(im, ax=ax, label="Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze consensus across runs")
    parser.add_argument("--inputs", type=str, required=True, help="Comma-separated LABEL=PATH pairs")
    parser.add_argument("--method", type=str, default="l2", choices=["fraction", "l2"])
    parser.add_argument("--top_k_percent", type=float, default=20.0)
    parser.add_argument("--consensus_min_fraction", type=float, default=0.8)
    parser.add_argument("--output_dir", type=str, default="./consensus")
    parser.add_argument("--plot", action="store_true", help="Save a consensus heatmap if matplotlib is available")
    parser.add_argument("--all_modules", action="store_true",
                        help="Include all per-layer modules in position stats, not just LoRA-eligible")
    args = parser.parse_args()

    run_specs = parse_inputs(args.inputs)
    runs = []
    for label, path in run_specs:
        loaded = load_importance_input(path, method=args.method, top_k_percent=args.top_k_percent)
        loaded["label"] = label
        loaded["active_set"] = set(loaded["mask"]["active_params"])
        runs.append(loaded)

    os.makedirs(args.output_dir, exist_ok=True)

    pairwise = []
    iou_values = []
    spearman_values = []

    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            left = runs[i]
            right = runs[j]
            corr = compute_score_correlation(left["scores"], right["scores"])
            iou = compute_iou(left["active_set"], right["active_set"])
            pairwise.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "iou": iou,
                    "spearman": corr["spearman"],
                    "pearson": corr["pearson"],
                    "n_common_scores": corr["n_common"],
                }
            )
            iou_values.append(iou)
            spearman_values.append(corr["spearman"])

    per_param_scores = defaultdict(list)
    per_param_active = defaultdict(int)
    per_position_scores = defaultdict(list)
    per_position_active = defaultdict(int)

    for run in runs:
        scores = run["scores"]
        active_set = run["active_set"]
        for name, score in scores.items():
            if score is None:
                continue
            score = float(score)
            per_param_scores[name].append(score)
            if name in active_set:
                per_param_active[name] += 1

            location = extract_layer_module(name, lora_only=not args.all_modules)
            if location is not None:
                per_position_scores[location].append(score)
                if name in active_set:
                    per_position_active[location] += 1

    n_runs = len(runs)
    param_stats = {}
    for name, values in sorted(per_param_scores.items()):
        arr = np.asarray(values, dtype=float)
        freq = per_param_active[name] / n_runs
        param_stats[name] = {
            "n_present": int(arr.size),
            "mean_score": float(arr.mean()),
            "std_score": float(arr.std()),
            "cv_score": float(arr.std() / arr.mean()) if arr.mean() != 0 else None,
            "active_frequency": freq,
        }

    position_stats = {}
    for location, values in sorted(per_position_scores.items()):
        arr = np.asarray(values, dtype=float)
        position_stats[location] = {
            "n_present": int(arr.size),
            "mean_score": float(arr.mean()),
            "std_score": float(arr.std()),
            "cv_score": float(arr.std() / arr.mean()) if arr.mean() != 0 else None,
            "active_frequency": per_position_active[location] / n_runs,
        }

    consensus_active = sorted(
        name for name, stats in param_stats.items() if stats["active_frequency"] >= args.consensus_min_fraction
    )
    consensus_all_scores = {name: stats["mean_score"] for name, stats in param_stats.items()}
    consensus_mask = {
        "method": args.method,
        "top_k_percent": args.top_k_percent,
        "all_scores": consensus_all_scores,
        "active_params": consensus_active,
        "frozen_params": sorted(set(consensus_all_scores) - set(consensus_active)),
        "metadata": {
            "consensus_min_fraction": args.consensus_min_fraction,
            "n_runs": n_runs,
            "run_labels": [run["label"] for run in runs],
        },
    }

    summary = {
        "method": args.method,
        "top_k_percent": args.top_k_percent,
        "consensus_min_fraction": args.consensus_min_fraction,
        "n_runs": n_runs,
        "run_labels": [run["label"] for run in runs],
        "pairwise": pairwise,
        "pairwise_mean_iou": float(np.nanmean(iou_values)) if iou_values else None,
        "pairwise_mean_spearman": float(np.nanmean(spearman_values)) if spearman_values else None,
        "consensus_active_count": len(consensus_active),
        "position_stats": {
            f"layer_{layer}:{module}": stats for (layer, module), stats in position_stats.items()
        },
    }

    save_json(summary, os.path.join(args.output_dir, "consensus_summary.json"))
    save_json(param_stats, os.path.join(args.output_dir, "param_stats.json"))
    save_json(consensus_mask, os.path.join(args.output_dir, "consensus_mask.json"))

    print(f"Runs: {n_runs}")
    print(f"Consensus active count: {len(consensus_active)}")
    if iou_values:
        print(f"Mean pairwise IoU: {np.mean(iou_values):.4f}")
    if spearman_values:
        print(f"Mean pairwise Spearman: {np.mean(spearman_values):.4f}")

    if args.plot:
        maybe_plot_position_frequency(position_stats, os.path.join(args.output_dir, "consensus_frequency.png"))


if __name__ == "__main__":
    main()
