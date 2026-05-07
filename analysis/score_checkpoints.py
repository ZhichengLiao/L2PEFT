#!/usr/bin/env python3
"""
Batch-score multiple checkpoints with both fraction and L2 methods.

Loads the base model once, iterates through checkpoints sequentially,
computes both scoring methods per checkpoint, saves results.

Usage:
    # Explicit list:
    python score_checkpoints.py \
        --base_model /path/to/base \
        --checkpoints step_001:/path/ckpt1,step_005:/path/ckpt5,final:/path/final \
        --output_dir ./scores

    # Auto-discover from directory:
    python score_checkpoints.py \
        --base_model /path/to/base \
        --checkpoint_dir /path/to/checkpoints \
        --output_dir ./scores
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Reuse profiling utilities
_profiling_dir = os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(__file__))), "..", "profiling")
if _profiling_dir not in sys.path:
    sys.path.insert(0, _profiling_dir)
try:
    from utils import load_state_dict, compute_delta, _SafeEncoder
except ImportError:
    raise ImportError(
        f"Cannot import profiling/utils.py. Expected at: {_profiling_dir}/utils.py\n"
        "Run this script from the project root or ensure profiling/utils.py exists."
    )


def score_fraction(delta_dict: dict[str, torch.Tensor], threshold: float) -> dict[str, float]:
    """Score each tensor by fraction of elements exceeding threshold."""
    scores = {}
    for name in sorted(delta_dict.keys()):
        d = delta_dict[name]
        scores[name] = (d.abs() > threshold).sum().item() / d.numel()
    return scores


def score_l2(delta_dict: dict[str, torch.Tensor], base_sd: dict[str, torch.Tensor]) -> dict[str, float]:
    """Score each tensor by relative L2 change."""
    scores = {}
    for name in sorted(delta_dict.keys()):
        delta_norm = delta_dict[name].norm(p=2).item()
        base_norm = base_sd[name].norm(p=2).item()
        scores[name] = delta_norm / (base_norm + 1e-10)
    return scores


def sanitize_label(label: str) -> str:
    """Sanitize label for use in filenames."""
    return re.sub(r"[^\w\-.]", "_", label)


def extract_step(label: str) -> int | None:
    """Extract step number from label like 'step_003' or 'global_step_50'."""
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else None


def parse_checkpoints(s: str) -> list[tuple[str, str]]:
    """Parse 'label:path,label:path,...' into [(label, path), ...]."""
    pairs = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, path = item.split(":", 1)
            pairs.append((label.strip(), path.strip()))
        else:
            pairs.append((os.path.basename(item), item))
    return pairs


def _is_hf_checkpoint(directory: Path) -> bool:
    """Check if a directory contains HF-format model files."""
    return (
        (directory / "model.safetensors").exists()
        or (directory / "model.safetensors.index.json").exists()
        or (directory / "pytorch_model.bin").exists()
        or (directory / "pytorch_model.bin.index.json").exists()
    )


def discover_checkpoints(checkpoint_dir: str) -> list[tuple[str, str]]:
    """Auto-discover HF-format checkpoints in a directory.

    Checks both direct subdirectories and nested actor/ subdirectories
    (veRL checkpoint format). Both formats can coexist.
    """
    root = Path(checkpoint_dir)
    pairs = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        # Direct HF checkpoint in subdir
        if _is_hf_checkpoint(subdir):
            pairs.append((subdir.name, str(subdir)))
            continue

        # veRL format: subdir/actor/ contains the HF checkpoint
        actor_dir = subdir / "actor"
        if actor_dir.is_dir() and _is_hf_checkpoint(actor_dir):
            pairs.append((subdir.name, str(actor_dir)))
            continue

        # veRL format: subdir/actor/huggingface/ contains converted HF checkpoint
        hf_dir = subdir / "actor" / "huggingface"
        if hf_dir.is_dir() and _is_hf_checkpoint(hf_dir):
            pairs.append((subdir.name, str(hf_dir)))

    # Sort by step number
    pairs.sort(key=lambda x: extract_step(x[0]) or float("inf"))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Batch-score checkpoints")
    parser.add_argument("--base_model", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoints", type=str,
                       help="Comma-separated label:path pairs")
    group.add_argument("--checkpoint_dir", type=str,
                       help="Directory to auto-discover checkpoints from")
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="Threshold for fraction scoring (default: 1e-5)")
    parser.add_argument("--output_dir", type=str, default="./scores")
    args = parser.parse_args()

    if args.checkpoints:
        checkpoints = parse_checkpoints(args.checkpoints)
    else:
        checkpoints = discover_checkpoints(args.checkpoint_dir)

    if not checkpoints:
        print("No checkpoints found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Base model: {args.base_model}")
    print(f"Checkpoints: {len(checkpoints)}")
    for label, path in checkpoints:
        print(f"  {label}: {path}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output_dir}\n")

    t0 = time.time()

    # Load base model once
    print("Loading base model...")
    base_sd = load_state_dict(args.base_model)
    print(f"  {len(base_sd)} tensors\n")

    # Process each checkpoint
    for i, (label, ckpt_path) in enumerate(checkpoints):
        step = extract_step(label)
        safe_label = sanitize_label(label)
        print(f"[{i+1}/{len(checkpoints)}] {label} (step={step}): {ckpt_path}")

        ckpt_sd = load_state_dict(ckpt_path)
        delta = compute_delta(ckpt_sd, base_sd)
        del ckpt_sd; gc.collect()

        if not delta:
            print("  WARNING: empty delta (checkpoint identical to base), skipping")
            del delta; gc.collect()
            continue

        frac_scores = score_fraction(delta, args.threshold)
        l2_scores = score_l2(delta, base_sd)
        del delta; gc.collect()

        # Summary stats
        frac_vals = sorted(frac_scores.values(), reverse=True)
        l2_vals = sorted(l2_scores.values(), reverse=True)
        print(f"  Fraction — max: {frac_vals[0]:.4f}, median: {frac_vals[len(frac_vals)//2]:.4f}")
        print(f"  L2       — max: {l2_vals[0]:.6f}, median: {l2_vals[len(l2_vals)//2]:.6f}")

        result = {
            "label": label,
            "step": step,
            "base_model": args.base_model,
            "checkpoint": ckpt_path,
            "threshold": args.threshold,
            "fraction_scores": frac_scores,
            "l2_scores": l2_scores,
            "n_params": len(frac_scores),
            "timestamp": datetime.now().isoformat(),
        }

        out_path = os.path.join(args.output_dir, f"scores_{safe_label}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, cls=_SafeEncoder)
        print(f"  Saved: {out_path}\n")

    elapsed = time.time() - t0
    print(f"Done. {len(checkpoints)} checkpoints scored in {elapsed:.1f}s")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
