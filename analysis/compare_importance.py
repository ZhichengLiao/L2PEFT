#!/usr/bin/env python3
"""
Generic comparison between two importance artifacts (score JSONs or masks).

Useful for:
- RL mask vs Spectrum mask
- task A vs task B ranking transfer
- fraction vs L2 comparison at a chosen checkpoint
"""

from __future__ import annotations

import argparse

import numpy as np

from common import compute_iou, compute_score_correlation, load_importance_input, save_json, score_matrix, select_topk


def topk_comparison(left_scores: dict[str, float], right_scores: dict[str, float], top_k: float) -> dict:
    common = set(left_scores.keys()) & set(right_scores.keys())
    left = select_topk(left_scores, top_k, keys=common)
    right = select_topk(right_scores, top_k, keys=common)
    return {
        "iou": compute_iou(left["names"], right["names"]),
        "overlap_count": len(left["names"] & right["names"]),
        "left_boundary": {
            "threshold": left["threshold"],
            "boundary_is_ambiguous": left["boundary_is_ambiguous"],
            "threshold_tie_count": left["threshold_tie_count"],
            "selected_from_tie": left["selected_from_tie"],
        },
        "right_boundary": {
            "threshold": right["threshold"],
            "boundary_is_ambiguous": right["boundary_is_ambiguous"],
            "threshold_tie_count": right["threshold_tie_count"],
            "selected_from_tie": right["selected_from_tie"],
        },
    }


def parse_topks(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return sorted(set(values))


def main():
    parser = argparse.ArgumentParser(description="Compare two importance artifacts")
    parser.add_argument("--left", type=str, required=True)
    parser.add_argument("--right", type=str, required=True)
    parser.add_argument("--left_method", type=str, default="l2", choices=["fraction", "l2"])
    parser.add_argument("--right_method", type=str, default="l2", choices=["fraction", "l2"])
    parser.add_argument("--top_k_percents", type=str, default="10,20,30")
    parser.add_argument("--top_k_percent", type=float, default=20.0)
    parser.add_argument("--left_label", type=str, default=None)
    parser.add_argument("--right_label", type=str, default=None)
    parser.add_argument("--output", type=str, default="importance_comparison.json")
    parser.add_argument("--all_modules", action="store_true",
                        help="Include all per-layer modules in matrix diff, not just LoRA-eligible")
    args = parser.parse_args()

    left = load_importance_input(args.left, method=args.left_method, top_k_percent=args.top_k_percent)
    right = load_importance_input(args.right, method=args.right_method, top_k_percent=args.top_k_percent)
    if args.left_label is not None:
        left["label"] = args.left_label
    if args.right_label is not None:
        right["label"] = args.right_label

    corr = compute_score_correlation(left["scores"], right["scores"])
    top_ks = parse_topks(args.top_k_percents)
    if args.top_k_percent not in top_ks:
        top_ks.append(args.top_k_percent)
        top_ks = sorted(set(top_ks))

    per_topk = {str(top_k): topk_comparison(left["scores"], right["scores"], top_k) for top_k in top_ks}

    lora_only = not args.all_modules
    left_matrix, _, _, _ = score_matrix(left["scores"], lora_only=lora_only)
    right_matrix, _, _, _ = score_matrix(right["scores"], lora_only=lora_only)
    matrix_diff_summary = None
    if left_matrix.shape == right_matrix.shape and left_matrix.size > 0:
        diff = left_matrix - right_matrix
        mask = np.isfinite(diff)
        matrix_diff_summary = {
            "mean_abs_difference": float(np.abs(diff[mask]).mean()) if mask.any() else None,
            "max_abs_difference": float(np.abs(diff[mask]).max()) if mask.any() else None,
        }

    output = {
        "left": {"label": left["label"], "path": args.left, "method": left["method"]},
        "right": {"label": right["label"], "path": args.right, "method": right["method"]},
        "score_correlation": corr,
        "per_topk": per_topk,
        "matrix_diff_summary": matrix_diff_summary,
    }

    save_json(output, args.output)
    print(
        f"{left['label']} vs {right['label']}: "
        f"Spearman={corr['spearman']:.4f}, "
        f"IoU@{args.top_k_percent}={per_topk[str(args.top_k_percent)]['iou']:.4f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
