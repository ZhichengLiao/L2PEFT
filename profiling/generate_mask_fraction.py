#!/usr/bin/env python3
"""
Generate parameter importance mask using FRACTION of changed elements.

For each parameter tensor, computes:
    score = count(|delta_ij| > threshold) / numel

where delta = checkpoint_weight - base_weight.

This measures "modification breadth" — what fraction of elements in this
parameter tensor were meaningfully updated during RL training.

Usage:
    python generate_mask_fraction.py \
        --base_model /path/to/base_model \
        --checkpoint /path/to/rl_checkpoint \
        --threshold 1e-5 \
        --top_k_percent 20 \
        --output mask_fraction.json
"""

import argparse
import gc
import sys
from datetime import datetime

import torch

from utils import load_state_dict, compute_delta, build_mask_output, save_mask


def score_fraction(
    delta_dict: dict[str, torch.Tensor],
    threshold: float,
) -> dict[str, float]:
    """Score each parameter tensor by fraction of elements exceeding threshold.

    Args:
        delta_dict: {param_name: delta_tensor} from compute_delta
        threshold: absolute threshold for counting an element as "changed"

    Returns:
        {param_name: fraction_changed} for all parameters
    """
    scores = {}
    for name in sorted(delta_dict.keys()):
        d = delta_dict[name]
        n_changed = (d.abs() > threshold).sum().item()
        n_total = d.numel()
        scores[name] = n_changed / n_total
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Generate mask by fraction of changed elements"
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model (HF format)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to RL-trained checkpoint (HF format)")
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="Absolute threshold for element change (default: 1e-5)")
    parser.add_argument("--top_k_percent", type=float, default=20.0,
                        help="Percentage of top-scoring params to mark active (default: 20)")
    parser.add_argument("--output", type=str, default="mask_fraction.json",
                        help="Output JSON path (default: mask_fraction.json)")
    args = parser.parse_args()

    # Load
    print(f"Loading base model: {args.base_model}")
    base_sd = load_state_dict(args.base_model)
    print(f"  {len(base_sd)} tensors loaded")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_sd = load_state_dict(args.checkpoint)
    print(f"  {len(ckpt_sd)} tensors loaded")

    # Delta
    print("Computing delta...")
    delta = compute_delta(ckpt_sd, base_sd)
    print(f"  {len(delta)} parameter tensors to score")
    del ckpt_sd
    gc.collect()

    # Score (base_sd not needed for fraction scoring, free it)
    del base_sd
    gc.collect()

    print(f"Scoring by fraction of elements with |delta| > {args.threshold}...")
    scores = score_fraction(delta, args.threshold)
    del delta
    gc.collect()

    # Print summary
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 by fraction changed:")
    for name, score in sorted_scores[:10]:
        print(f"  {score:.4f}  {name}")
    print(f"\nBottom 5:")
    for name, score in sorted_scores[-5:]:
        print(f"  {score:.6f}  {name}")

    n_zero = sum(1 for _, s in sorted_scores if s == 0.0)
    print(f"\nParams with zero change: {n_zero}/{len(scores)}")

    # Build and save mask
    mask = build_mask_output(
        all_scores=scores,
        top_k_percent=args.top_k_percent,
        method="fraction",
        base_path=args.base_model,
        ckpt_path=args.checkpoint,
        extra_metadata={
            "threshold": args.threshold,
            "timestamp": datetime.now().isoformat(),
        },
    )
    save_mask(mask, args.output)

    print(f"\nActive: {mask['metadata']['active_count']}/{mask['metadata']['total_params_scored']} params")
    print(f"Score threshold for active: {mask['metadata']['score_range']['active_threshold']:.6f}")


if __name__ == "__main__":
    main()
