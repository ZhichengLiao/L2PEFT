"""
Selective PEFT utilities — load profiling mask, classify active params,
apply selective freeze, and convert to LoRA target_modules.

The profiling mask (JSON) contains param-level importance scores.
This module bridges the gap between profiling output and training config:

- Selective FFT:  freeze frozen_params via requires_grad=False
- Selective LoRA: project active_params onto LoRA-compatible space,
                  report projection loss explicitly
"""

import json
from collections import Counter

import torch
import torch.nn as nn


# ── Module types that standard LoRA can target (Linear weight matrices) ──────

# These are the projection module suffixes in typical transformer architectures.
# LoRA inserts low-rank adapters on these Linear layers' weight matrices.
LORA_ELIGIBLE_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",           # MLP
    "gate_up_proj",                                # fused MLP (some models)
}

EXCLUDED_LORA_MODULE_NAMES = {
    "lm_head",
}

# ── Mask loading ─────────────────────────────────────────────────────────────


def load_mask(mask_path: str) -> dict:
    """Load a profiling mask JSON file.

    Expected keys: active_params, frozen_params, all_scores, metadata.
    """
    with open(mask_path) as f:
        mask = json.load(f)

    required = {"active_params", "frozen_params", "all_scores"}
    missing = required - set(mask.keys())
    if missing:
        raise ValueError(f"Mask file missing keys: {missing}")

    return mask


# ── Active param classification ──────────────────────────────────────────────


def _param_to_module_and_suffix(param_name: str) -> tuple[str, str, str]:
    """Parse a param name into (module_path, module_suffix, param_type).

    Examples:
        "model.layers.0.self_attn.q_proj.weight"
        → ("model.layers.0.self_attn.q_proj", "q_proj", "weight")

        "model.layers.5.input_layernorm.weight"
        → ("model.layers.5.input_layernorm", "input_layernorm", "weight")
    """
    # Split off .weight / .bias
    parts = param_name.rsplit(".", 1)
    if len(parts) == 2 and parts[1] in ("weight", "bias"):
        module_path = parts[0]
        param_type = parts[1]
    else:
        module_path = param_name
        param_type = "unknown"

    # Extract the last segment as module suffix
    module_suffix = module_path.rsplit(".", 1)[-1]

    return module_path, module_suffix, param_type


def collect_lora_compatible_module_paths(model: nn.Module) -> set[str]:
    """Collect LoRA-compatible module paths from the actual model structure.

    Priority is to follow the loaded model instead of assuming Llama/Qwen-style
    names. We include:
      - nn.Linear
      - transformers Conv1D (GPT-style) if available
      - fallback custom linear-like modules with a 2D weight parameter

    We explicitly exclude common output heads like lm_head even though they may
    also be Linear modules.
    """
    linear_types = [nn.Linear]
    try:
        from transformers.pytorch_utils import Conv1D

        linear_types.append(Conv1D)
    except Exception:
        pass
    linear_types = tuple(linear_types)

    eligible = set()
    for name, module in model.named_modules():
        if not name:
            continue

        module_suffix = name.rsplit(".", 1)[-1]
        if module_suffix in EXCLUDED_LORA_MODULE_NAMES:
            continue

        if isinstance(module, linear_types):
            eligible.add(name)
            continue

        # Fallback for custom linear wrappers / quantized linear layers.
        weight = getattr(module, "weight", None)
        if (
            isinstance(weight, nn.Parameter)
            and weight.ndim == 2
            and not isinstance(module, nn.Embedding)
        ):
            eligible.add(name)

    return eligible


def classify_active_params(mask: dict, eligible_module_paths: set[str] | None = None) -> dict:
    """Classify active params into three categories for LoRA projection.

    Returns dict with:
        projectable_active:  list of (param_name, module_path) — LoRA-compatible
                             linear weights whose .weight is in active_params
        lora_bias_active:    list of (param_name, module_path) — active bias on a
                             LoRA-compatible linear module. These are reported as
                             active signal that standard LoRA cannot express.
        bias_only_active:    subset of lora_bias_active where the corresponding
                             weight is NOT active
        non_lora_active:     list of param_name — LayerNorm, embedding, lm_head, etc.

        score_mass:          dict with total/projectable/lora_bias/bias_only/non_lora score sums
    """
    active_set = set(mask["active_params"])
    scores = mask["all_scores"]
    eligibility_source = "model_structure" if eligible_module_paths is not None else "suffix_fallback"

    projectable_active = []   # (param_name, module_path)
    lora_bias_active = []     # (param_name, module_path)
    bias_only_active = []     # (param_name, module_path)
    non_lora_active = []      # param_name

    for param_name in mask["active_params"]:
        module_path, module_suffix, param_type = _param_to_module_and_suffix(param_name)
        is_lora_eligible = (
            module_path in eligible_module_paths
            if eligible_module_paths is not None
            else module_suffix in LORA_ELIGIBLE_SUFFIXES
        )

        if is_lora_eligible:
            if param_type == "weight":
                # This is a LoRA-compatible linear weight → projectable
                projectable_active.append((param_name, module_path))
            elif param_type == "bias":
                # Bias updates are real active signal but standard LoRA with
                # bias="none" cannot express them directly. Always report them.
                lora_bias_active.append((param_name, module_path))

                # Check if the corresponding weight is also active
                weight_name = module_path + ".weight"
                if weight_name not in active_set:
                    # Bias-only active: weight is NOT active
                    bias_only_active.append((param_name, module_path))
            else:
                non_lora_active.append(param_name)
        else:
            # Not a LoRA-eligible module (LayerNorm, embedding, etc.)
            non_lora_active.append(param_name)

    # Compute score mass
    total_score = sum(scores.values())
    proj_score = sum(scores.get(pn, 0) for pn, _ in projectable_active)
    lora_bias_score = sum(scores.get(pn, 0) for pn, _ in lora_bias_active)
    bias_score = sum(scores.get(pn, 0) for pn, _ in bias_only_active)
    non_lora_score = sum(scores.get(pn, 0) for pn in non_lora_active)

    return {
        "projectable_active": projectable_active,
        "lora_bias_active": lora_bias_active,
        "bias_only_active": bias_only_active,
        "non_lora_active": non_lora_active,
        "eligibility_source": eligibility_source,
        "eligible_module_count": len(eligible_module_paths) if eligible_module_paths is not None else None,
        "score_mass": {
            "total_active": proj_score + lora_bias_score + non_lora_score,
            "total_all_params": total_score,
            "projectable": proj_score,
            "lora_bias": lora_bias_score,
            "bias_only": bias_score,
            "non_lora": non_lora_score,
        },
    }


# ── Projection report ────────────────────────────────────────────────────────


def print_projection_report(classification: dict, mask: dict):
    """Print detailed report of active param → LoRA projection.

    Shows what gets projected, what gets dropped, and score mass breakdown.
    """
    proj = classification["projectable_active"]
    lora_bias = classification["lora_bias_active"]
    bias = classification["bias_only_active"]
    non_lora = classification["non_lora_active"]
    mass = classification["score_mass"]
    eligibility_source = classification.get("eligibility_source", "unknown")
    eligible_module_count = classification.get("eligible_module_count")

    total_active = len(proj) + len(lora_bias) + len(non_lora)
    total_params = mask["metadata"]["total_params_scored"]

    print("=" * 72)
    print("SELECTIVE PEFT — PROJECTION REPORT")
    print("=" * 72)
    print(f"Mask: top-{mask['top_k_percent']}%, method={mask['method']}")
    print(f"Total params scored: {total_params}")
    print(f"Total active: {total_active}")
    print(f"Eligibility source: {eligibility_source}")
    if eligible_module_count is not None:
        print(f"LoRA-compatible modules in model: {eligible_module_count}")
    print()

    # Count breakdown
    print("── Active Param Classification ──")
    print(f"  projectable (LoRA-compatible weights): {len(proj)}")
    print(f"  lora_bias (active bias on LoRA module): {len(lora_bias)}")
    print(f"    └─ bias_only (weight not active):     {len(bias)}")
    print(f"  non_lora (norm/embed/head/other):      {len(non_lora)}")
    print()

    # Score mass breakdown
    print("── Score Mass Breakdown ──")
    total_active_mass = mass["total_active"]
    if total_active_mass > 0:
        print(f"  projectable score mass: {mass['projectable']:.6f} "
              f"({mass['projectable']/total_active_mass*100:.1f}% of active)")
        print(f"  lora_bias score mass:   {mass['lora_bias']:.6f} "
              f"({mass['lora_bias']/total_active_mass*100:.1f}% of active)")
        print(f"  bias_only score mass:   {mass['bias_only']:.6f} "
              f"({mass['bias_only']/total_active_mass*100:.1f}% of active)")
        print(f"  non_lora score mass:    {mass['non_lora']:.6f} "
              f"({mass['non_lora']/total_active_mass*100:.1f}% of active)")
    print(f"  total active mass:      {total_active_mass:.6f}")
    print(f"  total all-params mass:  {mass['total_all_params']:.6f}")
    print()

    # Projection loss
    dropped_mass = mass["lora_bias"] + mass["non_lora"]
    dropped_count = len(lora_bias) + len(non_lora)
    print("── Projection Loss (LoRA cannot express these) ──")
    print(f"  dropped params: {dropped_count}/{total_active}")
    if total_active_mass > 0:
        print(f"  dropped score mass: {dropped_mass:.6f} "
              f"({dropped_mass/total_active_mass*100:.1f}% of active signal)")
    print()

    # List dropped params
    if lora_bias:
        print("  lora_bias_active params:")
        for pn, mp in lora_bias:
            score = mask["all_scores"].get(pn, 0)
            print(f"    {score:.6f}  {pn}")
    if bias:
        print("  bias_only_active params (subset of lora_bias_active):")
        for pn, mp in bias:
            score = mask["all_scores"].get(pn, 0)
            print(f"    {score:.6f}  {pn}")
    if non_lora:
        print("  non_lora_active params:")
        for pn in non_lora:
            score = mask["all_scores"].get(pn, 0)
            print(f"    {score:.6f}  {pn}")

    print()

    # Final LoRA target modules
    target_modules = [mp for _, mp in proj]
    module_type_counts = Counter(mp.rsplit(".", 1)[-1] for mp in target_modules)
    print("── LoRA Target Module Types ──")
    for module_type, count in sorted(module_type_counts.items()):
        print(f"  {module_type:<16} {count}")
    print()

    print(f"── LoRA Target Modules ({len(target_modules)}) ──")
    for mp in target_modules:
        pn = mp + ".weight"
        score = mask["all_scores"].get(pn, 0)
        print(f"  {score:.6f}  {mp}")
    print("=" * 72)


# ── Mask → LoRA target_modules ───────────────────────────────────────────────


def mask_to_target_modules(
    mask: dict,
    model: nn.Module | None = None,
    verbose: bool = True,
) -> list[str]:
    """Convert profiling mask to PEFT target_modules list.

    Only includes LoRA-compatible linear module paths where the .weight
    is in active_params. If a model is provided, compatibility is inferred from
    the actual module structure; otherwise we fall back to a small suffix-based
    heuristic. Prints projection report if verbose.

    Returns:
        List of full module paths, e.g.
        ["model.layers.0.self_attn.q_proj", "model.layers.3.mlp.gate_proj"]
    """
    eligible_module_paths = collect_lora_compatible_module_paths(model) if model is not None else None
    classification = classify_active_params(mask, eligible_module_paths=eligible_module_paths)

    if verbose:
        print_projection_report(classification, mask)

    target_modules = [mp for _, mp in classification["projectable_active"]]

    if not target_modules:
        raise ValueError(
            "No LoRA-compatible active params found in mask. "
            "All active params are bias/norm/embedding. "
            "Consider using Selective Freeze FFT instead."
        )

    return target_modules


# ── Selective Freeze ─────────────────────────────────────────────────────────


def apply_selective_freeze(
    model: nn.Module,
    mask: dict,
    verbose: bool = True,
    strict: bool = True,
) -> dict:
    """Apply selective freeze from a profiling mask.

    In strict mode, the mask must exactly cover the model's parameters:
    every model parameter must appear in either active_params or frozen_params,
    and every mask entry must exist in the model.

    Parameters outside active_params are frozen by default, so the function is
    fail-closed even when strict=False.

    Must be called BEFORE FSDP wrapping.

    Args:
        model: The unwrapped model
        mask: Loaded mask dict
        verbose: Print stats
        strict: Require exact mask/model coverage. Default True.

    Returns:
        dict with freeze stats: n_frozen, n_trainable, param counts
    """
    active_set = set(mask["active_params"])
    frozen_set = set(mask["frozen_params"])
    overlap = active_set & frozen_set
    if overlap:
        raise ValueError(
            f"Mask has {len(overlap)} params marked as both active and frozen "
            f"(first 5: {sorted(overlap)[:5]})"
        )

    model_param_names = {name for name, _ in model.named_parameters()}

    # Validate: check that mask param names exist in model
    mask_all = active_set | frozen_set
    missing = mask_all - model_param_names
    extra = model_param_names - mask_all
    if strict and (missing or extra):
        problems = []
        if missing:
            problems.append(
                f"{len(missing)} mask params not found in model "
                f"(first 5: {sorted(missing)[:5]})"
            )
        if extra:
            problems.append(
                f"{len(extra)} model params not covered by mask "
                f"(first 5: {sorted(extra)[:5]})"
            )
        raise ValueError(
            "Selective freeze requires exact mask/model coverage in strict mode. "
            + "; ".join(problems)
        )
    if missing:
        print(
            f"[WARN] {len(missing)} mask params not found in model "
            f"(first 5: {sorted(missing)[:5]})"
        )
    if extra and verbose:
        print(
            f"[WARN] {len(extra)} model params not covered by mask; "
            "they will be frozen by default "
            f"(first 5: {sorted(extra)[:5]})"
        )

    # Apply freeze
    n_frozen = 0
    n_trainable = 0
    frozen_numel = 0
    trainable_numel = 0

    for name, param in model.named_parameters():
        if name in active_set:
            param.requires_grad_(True)
            n_trainable += 1
            trainable_numel += param.numel()
        else:
            param.requires_grad_(False)
            n_frozen += 1
            frozen_numel += param.numel()

    stats = {
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "frozen_numel": frozen_numel,
        "trainable_numel": trainable_numel,
        "total_numel": frozen_numel + trainable_numel,
        "trainable_pct": trainable_numel / (frozen_numel + trainable_numel) * 100,
    }

    if verbose:
        print("=" * 72)
        print("SELECTIVE FREEZE — APPLIED")
        print("=" * 72)
        print(f"  Strict mode:     {strict}")
        print(f"  Frozen params:    {n_frozen} ({frozen_numel:,} elements)")
        print(f"  Trainable params: {n_trainable} ({trainable_numel:,} elements)")
        print(f"  Trainable ratio:  {stats['trainable_pct']:.1f}%")
        print(f"  Memory saved (grad+optim): ~{frozen_numel * 2 * 4 / 1e9:.2f} GB "
              "(assuming fp32 AdamW states)")
        print("=" * 72)

    return stats
