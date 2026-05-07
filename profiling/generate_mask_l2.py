#!/usr/bin/env python3
"""
Generate parameter importance mask using relative L2 change.

For each parameter tensor, computes:
    score = ||delta||_2 / (||base||_2 + eps)

where delta = checkpoint_weight - base_weight, and ||.||_2 is the L2 norm
of the flattened tensor (= Frobenius norm for matrices).

This measures "energy concentration" — how much total change energy this
parameter received relative to its original magnitude.

Usage:
    python generate_mask_l2.py \
        --base_model /path/to/base_model \
        --checkpoint /path/to/rl_checkpoint \
        --top_k_percent 20 \
        --output mask_l2.json
"""

import argparse
import gc
import sys
from datetime import datetime

import torch

from utils import load_state_dict, compute_delta, build_mask_output, save_mask


def score_l2(
    delta_dict: dict[str, torch.Tensor],
    base_sd: dict[str, torch.Tensor],
    eps: float = 1e-10,
) -> dict[str, float]:
    """Score each parameter tensor by relative L2 change.

    Args:
        delta_dict: {param_name: delta_tensor} from compute_delta
        base_sd: {param_name: base_tensor} original model weights
        eps: small constant to avoid division by zero

    Returns:
        {param_name: relative_l2_change} for all parameters
    """
    scores = {}
    for name in sorted(delta_dict.keys()):
        delta_norm = delta_dict[name].norm(p=2).item()
        base_norm = base_sd[name].norm(p=2).item()
        scores[name] = delta_norm / (base_norm + eps)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Generate mask by relative L2 change"
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model (HF format)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to RL-trained checkpoint (HF format)")
    parser.add_argument("--top_k_percent", type=float, default=20.0,
                        help="Percentage of top-scoring params to mark active (default: 20)")
    parser.add_argument("--output", type=str, default="mask_l2.json",
                        help="Output JSON path (default: mask_l2.json)")
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

    # Score
    print("Scoring by relative L2 change...")
    scores = score_l2(delta, base_sd)
    del delta
    gc.collect()

    # Print summary
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 by relative L2 change:")
    for name, score in sorted_scores[:10]:
        print(f"  {score:.6f}  {name}")
    print(f"\nBottom 5:")
    for name, score in sorted_scores[-5:]:
        print(f"  {score:.8f}  {name}")

    n_zero = sum(1 for _, s in sorted_scores if s == 0.0)
    print(f"\nParams with zero change: {n_zero}/{len(scores)}")

    # Build and save mask
    mask = build_mask_output(
        all_scores=scores,
        top_k_percent=args.top_k_percent,
        method="l2",
        base_path=args.base_model,
        ckpt_path=args.checkpoint,
        extra_metadata={
            "eps": 1e-10,
            "timestamp": datetime.now().isoformat(),
        },
    )
    save_mask(mask, args.output)

    print(f"\nActive: {mask['metadata']['active_count']}/{mask['metadata']['total_params_scored']} params")
    print(f"Score threshold for active: {mask['metadata']['score_range']['active_threshold']:.8f}")


if __name__ == "__main__":
    main()
