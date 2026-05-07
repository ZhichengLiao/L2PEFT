#!/usr/bin/env python3
"""
Audit profiling-to-LoRA projection loss across masks or score files.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from common import load_importance_input, save_json

VERL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verl"))
if VERL_ROOT not in sys.path:
    sys.path.insert(0, VERL_ROOT)

from verl.utils.selective_peft import classify_active_params  # noqa: E402


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


def maybe_plot_projection(entries: list[dict], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping projection plot")
        return

    labels = [entry["label"] for entry in entries]
    proj = [entry["projectable_pct"] for entry in entries]
    dropped = [entry["dropped_pct"] for entry in entries]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.5))
    ax.bar(x, proj, label="Projectable", color="#0072B2")
    ax.bar(x, dropped, bottom=proj, label="Dropped", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Active Score Mass (%)")
    ax.set_title("Profiling to LoRA Projection Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit projection loss")
    parser.add_argument("--inputs", type=str, required=True, help="Comma-separated LABEL=PATH pairs")
    parser.add_argument("--method", type=str, default="l2", choices=["fraction", "l2"])
    parser.add_argument("--top_k_percent", type=float, default=20.0)
    parser.add_argument("--output_dir", type=str, default="./projection_audit")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entries = []
    for label, path in parse_inputs(args.inputs):
        loaded = load_importance_input(path, method=args.method, top_k_percent=args.top_k_percent)
        classification = classify_active_params(loaded["mask"])
        mass = classification["score_mass"]
        total_active_mass = mass["total_active"]
        dropped_mass = mass["lora_bias"] + mass["non_lora"]
        projectable_pct = (mass["projectable"] / total_active_mass * 100.0) if total_active_mass else 0.0
        dropped_pct = (dropped_mass / total_active_mass * 100.0) if total_active_mass else 0.0

        entries.append(
            {
                "label": label,
                "path": path,
                "method": loaded["method"],
                "active_count": len(loaded["mask"]["active_params"]),
                "projectable_count": len(classification["projectable_active"]),
                "lora_bias_count": len(classification["lora_bias_active"]),
                "bias_only_count": len(classification["bias_only_active"]),
                "non_lora_count": len(classification["non_lora_active"]),
                "projectable_score_mass": mass["projectable"],
                "lora_bias_score_mass": mass["lora_bias"],
                "non_lora_score_mass": mass["non_lora"],
                "total_active_score_mass": total_active_mass,
                "projectable_pct": projectable_pct,
                "dropped_pct": dropped_pct,
            }
        )

    summary = {
        "method": args.method,
        "top_k_percent": args.top_k_percent,
        "n_inputs": len(entries),
        "entries": entries,
        "mean_projectable_pct": float(np.nanmean([e["projectable_pct"] for e in entries])) if entries else None,
        "mean_dropped_pct": float(np.nanmean([e["dropped_pct"] for e in entries])) if entries else None,
    }
    save_json(summary, os.path.join(args.output_dir, "projection_audit.json"))

    print(f"Inputs: {len(entries)}")
    if entries:
        print(f"Mean projectable active mass: {summary['mean_projectable_pct']:.2f}%")
        print(f"Mean dropped active mass: {summary['mean_dropped_pct']:.2f}%")

    if args.plot:
        maybe_plot_projection(entries, os.path.join(args.output_dir, "projection_audit.png"))


if __name__ == "__main__":
    main()
