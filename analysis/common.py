#!/usr/bin/env python3
"""
Shared utilities for checkpoint / mask analysis.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np


LORA_ELIGIBLE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
)

# All per-layer module suffixes (including norms, biases, etc.)
ALL_LAYER_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
    "input_layernorm",
    "post_attention_layernorm",
    "q_norm",
    "k_norm",
)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return sorted(set(values)) if values else list(default)


def _sorted_items(scores: dict[str, float], keys: set[str] | None = None) -> list[tuple[str, float]]:
    if keys is None:
        items = list(scores.items())
    else:
        items = [(name, score) for name, score in scores.items() if name in keys]

    cleaned = []
    for name, score in items:
        if score is None:
            continue
        if isinstance(score, float) and math.isnan(score):
            continue
        cleaned.append((name, float(score)))

    return sorted(cleaned, key=lambda x: (-x[1], x[0]))


def select_topk(
    scores: dict[str, float],
    k_percent: float,
    keys: set[str] | None = None,
) -> dict:
    items = _sorted_items(scores, keys=keys)
    n_total = len(items)
    if n_total == 0:
        return {
            "names": set(),
            "selected_count": 0,
            "total_count": 0,
            "threshold": None,
            "strictly_above_threshold": 0,
            "threshold_tie_count": 0,
            "selected_from_tie": 0,
            "boundary_is_ambiguous": False,
        }

    n_select = max(1, int(n_total * k_percent / 100.0))
    threshold = items[n_select - 1][1]
    n_strict = sum(1 for _, score in items if score > threshold)
    n_tied = sum(1 for _, score in items if score == threshold)
    n_from_tie = n_select - n_strict

    return {
        "names": {name for name, _ in items[:n_select]},
        "selected_count": n_select,
        "total_count": n_total,
        "threshold": threshold,
        "strictly_above_threshold": n_strict,
        "threshold_tie_count": n_tied,
        "selected_from_tie": n_from_tie,
        "boundary_is_ambiguous": 0 < n_from_tie < n_tied,
    }


def compute_iou(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(order):
        j = i + 1
        v = values[order[i]]
        while j < len(order) and values[order[j]] == v:
            j += 1

        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman_rank_correlation(values_a: list[float], values_b: list[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")

    rank_a = _average_ranks(a)
    rank_b = _average_ranks(b)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def pearson_correlation(values_a: list[float], values_b: list[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_score_correlation(scores_a: dict[str, float], scores_b: dict[str, float]) -> dict:
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    pairs = []
    for name in common:
        if scores_a[name] is None or scores_b[name] is None:
            continue
        left = float(scores_a[name])
        right = float(scores_b[name])
        if math.isnan(left) or math.isnan(right):
            continue
        pairs.append((left, right))
    if len(pairs) < 2:
        return {"n_common": len(pairs), "spearman": float("nan"), "pearson": float("nan")}

    vec_a = [a for a, _ in pairs]
    vec_b = [b for _, b in pairs]
    return {
        "n_common": len(pairs),
        "spearman": spearman_rank_correlation(vec_a, vec_b),
        "pearson": pearson_correlation(vec_a, vec_b),
    }


def make_mask_from_scores(scores: dict[str, float], top_k_percent: float, method: str, source_path: str) -> dict:
    topk = select_topk(scores, top_k_percent)
    active = sorted(topk["names"])
    frozen = sorted(set(scores.keys()) - topk["names"])
    return {
        "method": method,
        "top_k_percent": top_k_percent,
        "all_scores": {name: float(score) for name, score in sorted(scores.items())},
        "active_params": active,
        "frozen_params": frozen,
        "metadata": {
            "source_path": source_path,
            "total_params_scored": len(scores),
            "active_count": len(active),
            "frozen_count": len(frozen),
            "active_threshold": topk["threshold"],
            "boundary_is_ambiguous": topk["boundary_is_ambiguous"],
            "threshold_tie_count": topk["threshold_tie_count"],
            "selected_from_tie": topk["selected_from_tie"],
        },
    }


def load_importance_input(path: str, method: str | None = None, top_k_percent: float = 20.0) -> dict:
    data = load_json(path)

    if {"active_params", "frozen_params", "all_scores"} <= set(data.keys()):
        return {
            "kind": "mask",
            "path": path,
            "method": data.get("method", method or "unknown"),
            "mask": data,
            "scores": data["all_scores"],
            "label": data.get("label") or Path(path).stem,
            "step": data.get("step"),
        }

    if {"fraction_scores", "l2_scores"} <= set(data.keys()):
        if method not in {"fraction", "l2"}:
            raise ValueError(f"Input {path} is a score file; choose --method fraction|l2")
        scores = data[f"{method}_scores"]
        return {
            "kind": "scores",
            "path": path,
            "method": method,
            "mask": make_mask_from_scores(scores, top_k_percent, method, path),
            "scores": scores,
            "label": data.get("label") or Path(path).stem,
            "step": data.get("step"),
        }

    raise ValueError(f"Unsupported importance file format: {path}")


def extract_layer_module(param_name: str, lora_only: bool = True) -> tuple[int, str] | None:
    if not param_name.endswith(".weight"):
        return None

    parts = param_name.split(".")
    if len(parts) < 3:
        return None

    module_type = parts[-2]
    allowed = LORA_ELIGIBLE_SUFFIXES if lora_only else ALL_LAYER_SUFFIXES
    if module_type not in allowed:
        return None

    match = _LAYER_RE.search(param_name)
    if not match:
        return None

    return int(match.group(1)), module_type


def score_matrix(
    scores: dict[str, float],
    lora_only: bool = True,
) -> tuple[np.ndarray, list[int], list[str], dict[tuple[int, str], str]]:
    cell_scores = {}
    cell_params = {}
    for param_name, score in scores.items():
        location = extract_layer_module(param_name, lora_only=lora_only)
        if location is None:
            continue
        cell_scores[location] = float(score)
        cell_params[location] = param_name

    layers = sorted({layer for layer, _ in cell_scores})
    suffix_order = ALL_LAYER_SUFFIXES if not lora_only else LORA_ELIGIBLE_SUFFIXES
    modules = [suffix for suffix in suffix_order if any(m == suffix for _, m in cell_scores)]

    matrix = np.full((len(layers), len(modules)), np.nan, dtype=float)
    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}
    module_to_col = {module: idx for idx, module in enumerate(modules)}

    for (layer, module), score in cell_scores.items():
        matrix[layer_to_row[layer], module_to_col[module]] = score

    return matrix, layers, modules, cell_params


def normalize_matrix(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(arr)
    if not mask.any():
        return arr

    observed = arr[mask]
    if mode == "none":
        return arr
    if mode == "sum1":
        denom = observed.sum()
        if denom != 0:
            arr[mask] = observed / denom
        return arr
    if mode == "zscore":
        std = observed.std()
        arr[mask] = (observed - observed.mean()) / (std if std > 0 else 1.0)
        return arr

    raise ValueError(f"Unsupported normalization mode: {mode}")
