"""
Profiling utilities — model loading, delta computation, mask I/O.

Adapted from /Users/a0/work/optimizers_for_rl/ModelMerging/early_identify/utils.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# JSON encoder that handles NaN / Inf / numpy types
# ---------------------------------------------------------------------------

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            if np.isnan(v):
                return None
            if np.isinf(v):
                return None
            return v
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Model loading (all tensors → fp32)
# ---------------------------------------------------------------------------

def load_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load HF checkpoint as flat state_dict on CPU in fp32.

    Supports single/sharded safetensors and pytorch_model.bin.
    """
    p = Path(model_path)

    # safetensors (preferred)
    st_index = p / "model.safetensors.index.json"
    st_single = p / "model.safetensors"
    if st_index.exists():
        return _load_sharded_safetensors(st_index)
    if st_single.exists():
        return _load_single_safetensors(st_single)

    # pytorch fallback
    pt_index = p / "pytorch_model.bin.index.json"
    pt_single = p / "pytorch_model.bin"
    if pt_index.exists():
        return _load_sharded_pytorch(pt_index)
    if pt_single.exists():
        sd = torch.load(str(pt_single), map_location="cpu", weights_only=True)
        return {k: v.float() for k, v in sd.items()}

    raise FileNotFoundError(
        f"No model files found in {model_path}. "
        "Expected model.safetensors(.index.json) or pytorch_model.bin(.index.json)."
    )


def _load_single_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    sd = load_file(str(path), device="cpu")
    return {k: v.float() for k, v in sd.items()}


def _load_sharded_safetensors(index_path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    sd = {}
    for shard_name in shard_files:
        shard_path = index_path.parent / shard_name
        print(f"  Loading shard: {shard_name}")
        shard_sd = load_file(str(shard_path), device="cpu")
        for k, v in shard_sd.items():
            sd[k] = v.float()
        del shard_sd
    return sd


def _load_sharded_pytorch(index_path: Path) -> dict[str, torch.Tensor]:
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    sd = {}
    for shard_name in shard_files:
        shard_path = index_path.parent / shard_name
        print(f"  Loading shard: {shard_name}")
        shard_sd = torch.load(str(shard_path), map_location="cpu", weights_only=True)
        for k, v in shard_sd.items():
            sd[k] = v.float()
        del shard_sd
    return sd


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_delta(
    ckpt_sd: dict[str, torch.Tensor],
    base_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute ckpt - base for all common weight/bias parameters.

    Deduplicates tie_word_embeddings (lm_head == embed_tokens).
    """
    common = sorted(set(ckpt_sd.keys()) & set(base_sd.keys()))
    only_ckpt = set(ckpt_sd.keys()) - set(base_sd.keys())
    only_base = set(base_sd.keys()) - set(ckpt_sd.keys())
    if only_ckpt:
        print(f"[WARN] Keys only in checkpoint: {sorted(only_ckpt)[:5]}{'...' if len(only_ckpt) > 5 else ''}")
    if only_base:
        print(f"[WARN] Keys only in base: {sorted(only_base)[:5]}{'...' if len(only_base) > 5 else ''}")

    delta = {}
    for key in common:
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        if ckpt_sd[key].shape != base_sd[key].shape:
            raise ValueError(f"Shape mismatch for '{key}': ckpt={ckpt_sd[key].shape}, base={base_sd[key].shape}")
        delta[key] = ckpt_sd[key] - base_sd[key]

    # Deduplicate tie_word_embeddings
    lm_key = "lm_head.weight"
    embed_key = "model.embed_tokens.weight"
    if lm_key in delta and embed_key in delta:
        if torch.equal(delta[lm_key], delta[embed_key]):
            del delta[lm_key]
            print("[INFO] Removed duplicate lm_head.weight (tie_word_embeddings)")
        else:
            print("[WARN] lm_head.weight and embed_tokens.weight differ — keeping both")

    return delta


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def save_mask(data: dict, output_path: str):
    """Save mask dict as pretty-printed JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, cls=_SafeEncoder)
    print(f"[SAVED] {output_path}")


def build_mask_output(
    all_scores: dict[str, float],
    top_k_percent: float,
    method: str,
    base_path: str,
    ckpt_path: str,
    extra_metadata: dict | None = None,
) -> dict:
    """Build the standard mask JSON from a {param_name: score} dict.

    Ranks all params by score descending, selects top_k_percent as active.
    """
    sorted_params = sorted(all_scores.items(), key=lambda x: (-x[1], x[0]))
    n_total = len(sorted_params)
    n_active = max(1, int(n_total * top_k_percent / 100.0))
    threshold = sorted_params[n_active - 1][1] if sorted_params else 0
    n_strict = sum(1 for _, score in sorted_params if score > threshold)
    n_tied = sum(1 for _, score in sorted_params if score == threshold)
    n_from_tie = n_active - n_strict

    active = [name for name, _ in sorted_params[:n_active]]
    frozen = [name for name, _ in sorted_params[n_active:]]

    result = {
        "method": method,
        "top_k_percent": top_k_percent,
        "all_scores": {name: score for name, score in sorted_params},
        "active_params": active,
        "frozen_params": frozen,
        "metadata": {
            "base_model": base_path,
            "checkpoint": ckpt_path,
            "total_params_scored": n_total,
            "active_count": n_active,
            "frozen_count": n_total - n_active,
            "score_range": {
                "max": sorted_params[0][1] if sorted_params else 0,
                "min": sorted_params[-1][1] if sorted_params else 0,
                "active_threshold": threshold,
            },
            "boundary": {
                "threshold_tie_count": n_tied,
                "selected_from_tie": n_from_tie,
                "boundary_is_ambiguous": 0 < n_from_tie < n_tied,
            },
            "dtype": "fp32",
        },
    }
    if extra_metadata:
        result["metadata"].update(extra_metadata)

    return result
