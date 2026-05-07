#!/usr/bin/env python3
"""
Prepare Selective LoRA: generate mask from scores + compute matching rank.

Takes a score checkpoint JSON (from score_checkpoints.py) and produces:
  1. A mask JSON compatible with selective_peft.py
  2. The rank needed to match a classic LoRA parameter budget

Usage:
    # Basic: generate mask and print recommended rank
    python scripts/prepare_selective_lora.py \
        --scores results/scores/scores_global_step_5_HF.json \
        --model_path /path/to/Qwen3-0.6B \
        --top_k_percent 25 \
        --classic_rank 16 \
        --output masks/mask_step5_l2_top25.json

    # Sweep multiple top-k to find best (top-k, rank) pair
    python scripts/prepare_selective_lora.py \
        --scores results/scores/scores_global_step_5_HF.json \
        --model_path /path/to/Qwen3-0.6B \
        --classic_rank 16 \
        --sweep
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


EXCLUDED_LORA_MODULE_NAMES = {
    "lm_head",
}


@dataclass(frozen=True)
class ModuleInfo:
    """LoRA-eligible module metadata inferred from the model."""

    param_name: str
    module_path: str
    module_type: str
    module_class: str
    category: str
    weight_shape: tuple[int, int]
    dim_sum: int
    source: str


def load_scores(path: str) -> dict:
    """Load score checkpoint and extract L2 scores."""
    with open(path) as f:
        data = json.load(f)

    if "l2_scores" not in data:
        raise ValueError(f"Score file missing 'l2_scores' key. Keys: {list(data.keys())}")

    return data


def _module_type(module_path: str) -> str:
    return module_path.rsplit(".", 1)[-1]


def _module_category(module_path: str) -> str:
    if ".self_attn." in module_path or ".attention." in module_path or ".attn." in module_path:
        return "attention"
    if ".mlp." in module_path or ".feed_forward." in module_path or ".ffn." in module_path:
        return "mlp"
    if module_path.endswith("lm_head"):
        return "output"
    return "other"


def _is_excluded_module(module_path: str, include_lm_head: bool) -> bool:
    if include_lm_head:
        return False
    return _module_type(module_path) in EXCLUDED_LORA_MODULE_NAMES


def _module_info_from_shape(
    param_name: str,
    shape: tuple[int, int],
    module_class: str,
    source: str,
) -> ModuleInfo:
    module_path = param_name.removesuffix(".weight")
    return ModuleInfo(
        param_name=param_name,
        module_path=module_path,
        module_type=_module_type(module_path),
        module_class=module_class,
        category=_module_category(module_path),
        weight_shape=shape,
        dim_sum=sum(shape),
        source=source,
    )


def _get_module_infos_from_empty_model(
    model_path: str,
    trust_remote_code: bool,
    include_lm_head: bool,
) -> dict[str, ModuleInfo]:
    """Infer LoRA-eligible modules from an empty HF model.

    This follows the actual architecture and therefore uses true weight shapes
    such as Qwen3's explicit head_dim instead of reconstructing them by hand.
    """
    from accelerate import init_empty_weights
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

    module_infos = {}
    for name, module in model.named_modules():
        if not name or _is_excluded_module(name, include_lm_head):
            continue

        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
            continue
        if isinstance(module, nn.Embedding):
            continue

        param_name = f"{name}.weight"
        shape = tuple(int(dim) for dim in weight.shape)
        module_infos[param_name] = _module_info_from_shape(
            param_name=param_name,
            shape=shape,
            module_class=module.__class__.__name__,
            source="empty_model",
        )

    return module_infos


def _iter_safetensor_shapes(model_path: Path):
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        for shard_name in shard_names:
            shard_path = model_path / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    yield key, tuple(int(dim) for dim in handle.get_slice(key).get_shape())
        return

    single_path = model_path / "model.safetensors"
    if single_path.exists():
        with safe_open(single_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, tuple(int(dim) for dim in handle.get_slice(key).get_shape())


def _get_module_infos_from_checkpoint_shapes(
    model_path: str,
    include_lm_head: bool,
) -> dict[str, ModuleInfo]:
    """Fallback: infer LoRA-eligible modules from checkpoint weight shapes."""
    path = Path(model_path)
    module_infos = {}

    try:
        shape_iter = list(_iter_safetensor_shapes(path))
    except FileNotFoundError:
        shape_iter = []

    if not shape_iter:
        torch_files = []
        index_path = path / "pytorch_model.bin.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            torch_files = sorted(set(index["weight_map"].values()))
        elif (path / "pytorch_model.bin").exists():
            torch_files = ["pytorch_model.bin"]

        if torch_files:
            import torch

            for filename in torch_files:
                shard = torch.load(path / filename, map_location="cpu", weights_only=True)
                for key, value in shard.items():
                    shape_iter.append((key, tuple(int(dim) for dim in value.shape)))

    for key, shape in shape_iter:
        if not key.endswith(".weight") or len(shape) != 2:
            continue
        module_path = key.removesuffix(".weight")
        if _is_excluded_module(module_path, include_lm_head):
            continue
        if module_path.endswith("embed_tokens") or ".embed_tokens" in module_path:
            continue
        module_infos[key] = _module_info_from_shape(
            param_name=key,
            shape=shape,
            module_class="checkpoint_2d_weight",
            source="checkpoint_shape",
        )

    return module_infos


def _get_module_infos_from_config_fallback(
    model_path: str,
    include_lm_head: bool,
) -> dict[str, ModuleInfo]:
    """Last-resort dense transformer fallback from config.json.

    This keeps the script usable without transformers/torch installed, but the
    empty-model path above is preferred because it is architecture-driven.
    """
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim", hidden // num_heads)
    num_layers = config["num_hidden_layers"]

    # Module type -> weight shape (out_features, in_features)
    module_dims = {
        "q_proj": (num_heads * head_dim, hidden),
        "k_proj": (num_kv_heads * head_dim, hidden),
        "v_proj": (num_kv_heads * head_dim, hidden),
        "o_proj": (hidden, num_heads * head_dim),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }

    module_infos = {}
    for layer_idx in range(num_layers):
        for mod_name, shape in module_dims.items():
            if mod_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                param = f"model.layers.{layer_idx}.self_attn.{mod_name}.weight"
            else:
                param = f"model.layers.{layer_idx}.mlp.{mod_name}.weight"
            module_infos[param] = _module_info_from_shape(
                param_name=param,
                shape=shape,
                module_class="config_linear_fallback",
                source="config_fallback",
            )

    if include_lm_head and not config.get("tie_word_embeddings", False):
        shape = (config["vocab_size"], hidden)
        module_infos["lm_head.weight"] = _module_info_from_shape(
            param_name="lm_head.weight",
            shape=shape,
            module_class="config_linear_fallback",
            source="config_fallback",
        )

    return module_infos


def get_lora_module_infos(
    model_path: str,
    trust_remote_code: bool = False,
    include_lm_head: bool = False,
) -> dict[str, ModuleInfo]:
    """Get LoRA-eligible module metadata from the model.

    Preference order:
      1. Empty HF model structure (most accurate, no weight loading)
      2. Checkpoint 2D weight shapes
      3. Dense transformer config fallback
    """
    errors = []
    for loader in (
        lambda: _get_module_infos_from_empty_model(model_path, trust_remote_code, include_lm_head),
        lambda: _get_module_infos_from_checkpoint_shapes(model_path, include_lm_head),
        lambda: _get_module_infos_from_config_fallback(model_path, include_lm_head),
    ):
        try:
            module_infos = loader()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if module_infos:
            return dict(sorted(module_infos.items()))

    raise RuntimeError(
        "Could not infer LoRA module dimensions from model. Tried empty model, checkpoint shapes, "
        f"and config fallback. Errors: {errors}"
    )



def lora_params_for_module(module_info: ModuleInfo, rank: int) -> int:
    """LoRA parameter count for a single module: rank * (shape[0] + shape[1])."""
    return module_info.dim_sum * rank


def build_mask(scores_data: dict, method: str, top_k_percent: float) -> dict:
    """Build a profiling mask from score checkpoint data.

    Converts the score_checkpoints.py output format to the mask format
    expected by selective_peft.py.
    """
    if method == "l2":
        all_scores = scores_data["l2_scores"]
    elif method == "fraction":
        all_scores = scores_data["fraction_scores"]
    else:
        raise ValueError(f"Unknown method: {method}")

    if not (0 < top_k_percent <= 100):
        raise ValueError(f"top_k_percent must be in (0, 100], got {top_k_percent}")

    # Sort by score descending
    sorted_params = sorted(all_scores.items(), key=lambda x: (-x[1], x[0]))
    n_total = len(sorted_params)
    n_active = min(n_total, max(1, int(math.ceil(n_total * top_k_percent / 100))))

    active_params = [name for name, _ in sorted_params[:n_active]]
    frozen_params = [name for name, _ in sorted_params[n_active:]]

    scores_list = [s for _, s in sorted_params]
    threshold = sorted_params[n_active - 1][1] if sorted_params and n_active <= n_total else 0.0
    n_strict = sum(1 for _, score in sorted_params if score > threshold)
    n_tied = sum(1 for _, score in sorted_params if score == threshold)
    n_from_tie = n_active - n_strict

    mask = {
        "method": method,
        "top_k_percent": top_k_percent,
        "all_scores": all_scores,
        "active_params": active_params,
        "frozen_params": frozen_params,
        "metadata": {
            "base_model": scores_data.get("base_model", "unknown"),
            "checkpoint": scores_data.get("checkpoint", "unknown"),
            "source_label": scores_data.get("label", "unknown"),
            "source_step": scores_data.get("step"),
            "total_params_scored": n_total,
            "active_count": n_active,
            "frozen_count": n_total - n_active,
            "score_range": {
                "max": scores_list[0] if scores_list else 0,
                "min": scores_list[-1] if scores_list else 0,
                "active_threshold": threshold,
            },
            "boundary": {
                "threshold_tie_count": n_tied,
                "selected_from_tie": n_from_tie,
                "boundary_is_ambiguous": 0 < n_from_tie < n_tied,
            },
        },
    }
    return mask


def compute_budget_match(
    mask: dict,
    module_infos: dict[str, ModuleInfo],
    classic_rank: int,
) -> dict:
    """Compute selective rank to match classic LoRA parameter budget.

    Returns detailed budget analysis.
    """
    # Classic LoRA: all linear modules at classic_rank
    all_linear_params = sorted(module_infos.keys())
    classic_dim_sum = sum(info.dim_sum for info in module_infos.values())
    classic_total = sum(
        lora_params_for_module(info, classic_rank)
        for info in module_infos.values()
    )

    # Selective: only active linear weight params
    active_set = set(mask["active_params"])
    selected_modules = []
    for param_name in all_linear_params:
        if param_name in active_set:
            selected_modules.append(module_infos[param_name])

    if not selected_modules:
        return {
            "error": "No LoRA-eligible modules in active set",
            "classic_total_params": classic_total,
            "selected_count": 0,
        }

    # Sum of (fan_in + fan_out) for selected modules
    selected_dim_sum = sum(info.dim_sum for info in selected_modules)

    # Exact rank to match: classic_total = selected_dim_sum * selective_rank
    exact_rank = classic_total / selected_dim_sum
    recommended_rank = max(1, round(exact_rank))
    rank_floor = int(exact_rank)
    rank_ceil = rank_floor + 1

    # Also compute for common rank values
    rank_options = sorted(set([
        rank_floor, rank_ceil,
        recommended_rank,
        # nearest powers of 2
        2 ** max(0, int(math.log2(exact_rank))),
        2 ** int(math.ceil(math.log2(max(1, exact_rank)))),
    ]))

    options = []
    for r in rank_options:
        if r < 1:
            continue
        params = selected_dim_sum * r
        options.append({
            "rank": r,
            "total_params": params,
            "vs_classic_pct": params / classic_total * 100,
            "params_M": params / 1e6,
        })

    # Module breakdown
    by_category = _summarize_modules(selected_modules, key=lambda info: info.category, mask=mask)
    by_type = _summarize_modules(selected_modules, key=lambda info: info.module_type, mask=mask)
    all_by_type = _summarize_modules(module_infos.values(), key=lambda info: info.module_type, mask=mask)

    scores = mask["all_scores"]
    active_lora_eligible = active_set & set(module_infos)
    active_score_mass = sum(scores.get(p, 0) for p in active_set)
    selected_score_mass = sum(scores.get(info.param_name, 0) for info in selected_modules)
    all_lora_score_mass = sum(scores.get(p, 0) for p in module_infos)
    active_non_lora_score_mass = active_score_mass - selected_score_mass

    return {
        "method": mask.get("method", "unknown"),
        "classic_rank": classic_rank,
        "classic_dim_sum": classic_dim_sum,
        "classic_total_params": classic_total,
        "classic_total_M": classic_total / 1e6,
        "classic_n_modules": len(all_linear_params),
        "selected_n_modules": len(selected_modules),
        "selected_module_pct_of_lora": len(selected_modules) / len(all_linear_params) * 100,
        "selected_dim_sum": selected_dim_sum,
        "selected_dim_pct_of_all": selected_dim_sum / classic_dim_sum * 100,
        "selected_avg_dim_sum": selected_dim_sum / len(selected_modules),
        "exact_matching_rank": exact_rank,
        "recommended_rank": recommended_rank,
        "rank_options": options,
        "score_mass": {
            "active_all": active_score_mass,
            "selected_lora_eligible": selected_score_mass,
            "active_non_lora": active_non_lora_score_mass,
            "all_lora_eligible": all_lora_score_mass,
        },
        "active_lora_eligible_count": len(active_lora_eligible),
        "active_non_lora_count": len(active_set) - len(active_lora_eligible),
        "by_category": by_category,
        "by_type": by_type,
        "all_by_type": all_by_type,
        "selected_modules": [asdict(info) for info in selected_modules],
    }


def _summarize_modules(modules, key, mask: dict) -> dict:
    """Summarize modules by a grouping key."""
    scores = mask["all_scores"]
    summary = defaultdict(lambda: {"count": 0, "dim_sum": 0, "score_mass": 0.0})
    for info in modules:
        group = key(info)
        summary[group]["count"] += 1
        summary[group]["dim_sum"] += info.dim_sum
        summary[group]["score_mass"] += scores.get(info.param_name, 0.0)
    return dict(sorted(summary.items()))


def _print_group_summary(title: str, summary: dict, limit: int | None = None):
    if not summary:
        return
    print(f"\n  {title}:")
    items = sorted(summary.items(), key=lambda x: (-x[1]["score_mass"], x[0]))
    if limit is not None:
        items = items[:limit]
    total_dim_sum = sum(stats["dim_sum"] for _, stats in items)
    for name, stats in items:
        dim_pct = stats["dim_sum"] / total_dim_sum * 100 if total_dim_sum else 0
        print(
            f"    {name:<16} count={stats['count']:>3d} "
            f"dim_sum={stats['dim_sum']:>8d} ({dim_pct:>5.1f}%) "
            f"score_mass={stats['score_mass']:.6f}"
        )


def print_budget_report(budget: dict, top_k: float):
    """Print a clear budget matching report."""
    if "error" in budget:
        print(f"ERROR: {budget['error']}")
        return

    print(f"\n{'='*68}")
    print(f"PARAMETER BUDGET MATCHING — top-{top_k}% {budget['method']}")
    print(f"{'='*68}")
    print(f"Classic LoRA baseline:")
    print(f"  rank={budget['classic_rank']}, all {budget['classic_n_modules']} modules"
          f" → {budget['classic_total_M']:.2f}M params")
    print(f"\nSelective LoRA:")
    print(f"  {budget['selected_n_modules']} modules selected"
          f" ({budget['selected_module_pct_of_lora']:.1f}% of LoRA-eligible modules; "
          f"{budget['active_non_lora_count']} active non-LoRA params dropped)")
    print(
        f"  selected dim-sum = {budget['selected_dim_sum']:,} "
        f"({budget['selected_dim_pct_of_all']:.1f}% of all eligible dim-sum; "
        f"avg {budget['selected_avg_dim_sum']:.1f}/module)"
    )
    print(f"  exact matching rank = {budget['exact_matching_rank']:.2f}")
    print(f"  recommended rank    = {budget['recommended_rank']}")
    mass = budget["score_mass"]
    if mass["active_all"] > 0:
        print(
            f"  projected active score mass = {mass['selected_lora_eligible']:.6f} "
            f"({mass['selected_lora_eligible'] / mass['active_all'] * 100:.1f}% of active)"
        )
    print(f"\n  Rank options:")
    for opt in budget["rank_options"]:
        marker = " ← closest" if abs(opt["vs_classic_pct"] - 100) < 2 else ""
        print(f"    rank={opt['rank']:>3d}: {opt['params_M']:.2f}M"
              f" ({opt['vs_classic_pct']:.1f}% of classic){marker}")
    _print_group_summary("Selected by category", budget["by_category"])
    _print_group_summary("Selected by module type", budget["by_type"])
    print(f"{'='*68}")


def sweep_topk(
    scores_data: dict,
    module_infos: dict[str, ModuleInfo],
    classic_rank: int,
    top_k_range: list[float],
    method: str,
):
    """Sweep top-k percentages and print summary table."""
    print(f"\n{'='*68}")
    print(f"SWEEP: Classic rank={classic_rank}"
          f" ({sum(lora_params_for_module(info, classic_rank) for info in module_infos.values()) / 1e6:.2f}M)")
    print(f"{'='*68}")
    print(f"{'top-k%':>8} {'modules':>8} {'mod%':>6} {'attn':>6} {'MLP':>6}"
          f" {'dim%':>7} {'avg_dim':>8} {'match_r':>9} {'rank':>6}"
          f" {'params':>10} {'vs classic':>11}")
    print("-" * 112)

    for top_k in top_k_range:
        mask = build_mask(scores_data, method, top_k)
        budget = compute_budget_match(mask, module_infos, classic_rank)
        if "error" in budget:
            print(f"{top_k:>7.0f}%  {'(no eligible modules)':>60}")
            continue

        exact_r = budget["exact_matching_rank"]
        # Pick nearest reasonable rank
        actual_r = budget["recommended_rank"]
        actual_params = budget["selected_dim_sum"] * actual_r
        attn = budget["by_category"].get("attention", {}).get("count", 0)
        mlp = budget["by_category"].get("mlp", {}).get("count", 0)

        print(f"{top_k:>7.0f}% {budget['selected_n_modules']:>8}"
              f" {budget['selected_module_pct_of_lora']:>5.1f}%"
              f" {attn:>6} {mlp:>6}"
              f" {budget['selected_dim_pct_of_all']:>6.1f}%"
              f" {budget['selected_avg_dim_sum']:>8.0f}"
              f" {exact_r:>9.1f} {actual_r:>6d}"
              f" {actual_params/1e6:>9.2f}M {actual_params/budget['classic_total_params']*100:>10.1f}%")

    print("-" * 112)
    print("Note: 'match rank' is exact; 'actual rank' is nearest integer.")
    print("      top-k% is over all scored tensors; mod% is over LoRA-eligible modules after projection.")
    print("      Module sizes are not uniform: selected dim-sum, not module count, drives the matching rank.\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Selective LoRA mask and compute matching rank")
    parser.add_argument("--scores", required=True, help="Path to score checkpoint JSON (from score_checkpoints.py)")
    parser.add_argument("--model_path", required=True, help="Path to base model (needs config.json)")
    parser.add_argument("--top_k_percent", type=float, default=25, help="Top-k%% of params to activate")
    parser.add_argument("--classic_rank", type=int, default=16, help="Classic LoRA rank to match against")
    parser.add_argument("--method", choices=["l2", "fraction"], default="l2", help="Score field to build the mask from")
    parser.add_argument("--output", type=str, default=None, help="Output mask JSON path")
    parser.add_argument("--sweep", action="store_true", help="Sweep top-k from 10-50%% instead of single run")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when building model skeleton",
    )
    parser.add_argument("--include_lm_head", action="store_true", help="Include lm_head in LoRA budget analysis")
    args = parser.parse_args()

    scores_data = load_scores(args.scores)
    module_infos = get_lora_module_infos(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        include_lm_head=args.include_lm_head,
    )

    print(f"Loaded scores: {scores_data.get('label', '?')}, step={scores_data.get('step', '?')}")
    sources = sorted({info.source for info in module_infos.values()})
    print(f"Model LoRA-eligible modules: {len(module_infos)} (source={','.join(sources)})")
    score_key = f"{args.method}_scores"
    if score_key not in scores_data:
        raise ValueError(f"Score file missing '{score_key}' key. Keys: {list(scores_data.keys())}")
    print(f"{args.method} scores: {len(scores_data[score_key])} params")
    _print_group_summary(
        "All eligible modules by type",
        _summarize_modules(
            module_infos.values(),
            lambda info: info.module_type,
            {"all_scores": scores_data[score_key]},
        ),
    )

    if args.sweep:
        top_k_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        for cr in [8, 16, 32, 64]:
            sweep_topk(scores_data, module_infos, cr, top_k_range, args.method)
        return

    # Single run: build mask + compute budget
    mask = build_mask(scores_data, args.method, args.top_k_percent)
    budget = compute_budget_match(mask, module_infos, args.classic_rank)
    print_budget_report(budget, args.top_k_percent)

    if "error" in budget:
        raise SystemExit(1)

    mask["metadata"]["selective_lora_budget"] = {
        "method": budget["method"],
        "classic_rank": budget["classic_rank"],
        "classic_dim_sum": budget["classic_dim_sum"],
        "classic_total_params": budget["classic_total_params"],
        "classic_total_M": budget["classic_total_M"],
        "classic_n_modules": budget["classic_n_modules"],
        "selected_n_modules": budget["selected_n_modules"],
        "selected_module_pct_of_lora": budget["selected_module_pct_of_lora"],
        "selected_dim_sum": budget["selected_dim_sum"],
        "selected_dim_pct_of_all": budget["selected_dim_pct_of_all"],
        "selected_avg_dim_sum": budget["selected_avg_dim_sum"],
        "exact_matching_rank": budget["exact_matching_rank"],
        "recommended_rank": budget["recommended_rank"],
        "rank_options": budget["rank_options"],
        "score_mass": budget["score_mass"],
        "active_lora_eligible_count": budget["active_lora_eligible_count"],
        "active_non_lora_count": budget["active_non_lora_count"],
        "by_category": budget["by_category"],
        "by_type": budget["by_type"],
        "eligible_module_source": sorted({info.source for info in module_infos.values()}),
    }
    mask["metadata"]["lora_target_modules"] = [info["module_path"] for info in budget["selected_modules"]]

    # Save mask
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(mask, f, indent=2)
        print(f"\nMask saved to: {args.output}")

        # Also print the command to run training
        suggested_rank = budget["recommended_rank"]
        print(f"\n--- Run training ---")
        print(f"MASK_PATH={args.output} LORA_RANK={suggested_rank}"
              f" bash scripts/train_grpo_selective_lora.sh")
    else:
        print("\n(No --output specified, mask not saved)")


if __name__ == "__main__":
    main()
