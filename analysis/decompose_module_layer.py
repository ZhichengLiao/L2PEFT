#!/usr/bin/env python3
"""
Decompose module-layer score structure into module-only, layer-only, and
module+layer additive components.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import load_importance_input, normalize_matrix, save_json, score_matrix


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


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if not mask.any():
        return float("nan")
    y = observed[mask]
    yhat = predicted[mask]
    sst = np.sum((y - y.mean()) ** 2)
    if sst == 0:
        return float("nan")
    sse = np.sum((y - yhat) ** 2)
    return float(1.0 - sse / sst)


def decompose(matrix: np.ndarray, layers: list[int], modules: list[str]) -> dict:
    mask = np.isfinite(matrix)
    if not mask.any():
        raise ValueError("No valid module-layer cells found")

    observed = matrix.copy()
    global_mean = float(np.nanmean(observed))
    row_means = np.nanmean(observed, axis=1)
    col_means = np.nanmean(observed, axis=0)

    module_only = np.tile(col_means, (observed.shape[0], 1))
    layer_only = np.tile(row_means.reshape(-1, 1), (1, observed.shape[1]))
    additive = layer_only + module_only - global_mean
    interaction = observed - additive

    residuals = []
    for row in range(observed.shape[0]):
        for col in range(observed.shape[1]):
            if np.isfinite(interaction[row, col]):
                residuals.append(
                    {
                        "layer": layers[row],
                        "module": modules[col],
                        "interaction_residual": float(interaction[row, col]),
                        "observed_score": float(observed[row, col]),
                    }
                )
    residuals.sort(key=lambda x: abs(x["interaction_residual"]), reverse=True)

    return {
        "global_mean": global_mean,
        "row_means": row_means.tolist(),
        "col_means": col_means.tolist(),
        "r2_module_only": _r2(observed, module_only),
        "r2_layer_only": _r2(observed, layer_only),
        "r2_additive_module_plus_layer": _r2(observed, additive),
        "interaction_l2": float(np.sqrt(np.nansum(interaction ** 2))),
        "interaction_fraction_of_signal": float(
            np.nansum(interaction ** 2) / np.nansum((observed - global_mean) ** 2)
        )
        if np.nansum((observed - global_mean) ** 2) > 0
        else float("nan"),
        "interaction_matrix": np.where(np.isfinite(interaction), interaction, np.nan).tolist(),
        "top_residual_cells": residuals[:10],
    }


def maybe_plot_heatmaps(matrix: np.ndarray, interaction: np.ndarray, layers: list[int], modules: list[str], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping decomposition heatmap")
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(modules) * 1.2), max(5, len(layers) * 0.25)))
    for ax, data, title, cmap in [
        (axes[0], matrix, "Observed Score Matrix", "viridis"),
        (axes[1], interaction, "Interaction Residual", "coolwarm"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(modules)))
        ax.set_xticklabels(modules, rotation=45, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("Module Type")
        ax.set_ylabel("Layer Index")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Decompose module-layer structure")
    parser.add_argument("--inputs", type=str, required=True, help="Comma-separated LABEL=PATH pairs")
    parser.add_argument("--method", type=str, default="l2", choices=["fraction", "l2"])
    parser.add_argument("--top_k_percent", type=float, default=20.0)
    parser.add_argument("--normalize", type=str, default="none", choices=["none", "sum1", "zscore"])
    parser.add_argument("--output_dir", type=str, default="./module_layer_decomposition")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--all_modules", action="store_true",
                        help="Include all per-layer modules (norms, etc.), not just LoRA-eligible projections")
    args = parser.parse_args()

    lora_only = not args.all_modules

    os.makedirs(args.output_dir, exist_ok=True)

    per_run = []
    matrices = []
    layers_ref = None
    modules_ref = None

    for label, path in parse_inputs(args.inputs):
        loaded = load_importance_input(path, method=args.method, top_k_percent=args.top_k_percent)
        matrix, layers, modules, _ = score_matrix(loaded["scores"], lora_only=lora_only)
        matrix = normalize_matrix(matrix, args.normalize)
        result = decompose(matrix, layers, modules)
        result["label"] = label
        result["layers"] = layers
        result["modules"] = modules
        per_run.append(result)
        if layers_ref is not None and (layers != layers_ref or modules != modules_ref):
            raise ValueError(
                "All inputs must have the same layer/module grid. "
                f"Expected layers/modules from first input, got {label} with a different layout."
            )

        matrices.append(matrix)

        if layers_ref is None:
            layers_ref = layers
            modules_ref = modules

        print(
            f"{label}: R2 module={result['r2_module_only']:.4f}, "
            f"layer={result['r2_layer_only']:.4f}, additive={result['r2_additive_module_plus_layer']:.4f}"
        )

    stacked = np.stack(matrices, axis=0)
    mean_matrix = np.nanmean(stacked, axis=0)
    aggregate = decompose(mean_matrix, layers_ref, modules_ref)
    aggregate["layers"] = layers_ref
    aggregate["modules"] = modules_ref

    summary = {
        "method": args.method,
        "normalize": args.normalize,
        "n_inputs": len(per_run),
        "per_run": per_run,
        "aggregate": aggregate,
        "mean_r2_module_only": float(np.mean([run["r2_module_only"] for run in per_run])) if per_run else None,
        "mean_r2_layer_only": float(np.mean([run["r2_layer_only"] for run in per_run])) if per_run else None,
        "mean_r2_additive": float(np.mean([run["r2_additive_module_plus_layer"] for run in per_run])) if per_run else None,
    }
    save_json(summary, os.path.join(args.output_dir, "decomposition_summary.json"))

    if args.plot and layers_ref is not None and modules_ref is not None:
        interaction = np.asarray(aggregate["interaction_matrix"], dtype=float)
        maybe_plot_heatmaps(mean_matrix, interaction, layers_ref, modules_ref, os.path.join(args.output_dir, "decomposition_heatmap.png"))


if __name__ == "__main__":
    main()
