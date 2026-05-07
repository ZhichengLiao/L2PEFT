#!/usr/bin/env python3
"""
Compute early-locking convergence metrics.

Reads score JSONs from score_checkpoints.py and compares each intermediate
checkpoint against the final checkpoint.

Outputs:
- IoU against final at one or more top-k budgets
- Spearman / Pearson score correlation against final
- Step-to-step churn, birth, and death rates
- Boundary ambiguity diagnostics for tie-heavy score distributions
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from common import compute_iou, compute_score_correlation, parse_float_list, select_topk


def load_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _metric_entry(current_scores: dict[str, float], final_scores: dict[str, float], top_k: float) -> tuple[dict, set[str]]:
    common_keys = set(current_scores.keys()) & set(final_scores.keys())
    current_topk = select_topk(current_scores, top_k, keys=common_keys)
    final_topk = select_topk(final_scores, top_k, keys=common_keys)
    current_names = current_topk["names"]
    final_names = final_topk["names"]
    corr = compute_score_correlation(current_scores, final_scores)

    entry = {
        "iou": compute_iou(current_names, final_names),
        "spearman": corr["spearman"],
        "pearson": corr["pearson"],
        "active_count": len(current_names),
        "overlap_count": len(current_names & final_names),
        "n_common_scores": corr["n_common"],
        "boundary": {
            "threshold": current_topk["threshold"],
            "boundary_is_ambiguous": current_topk["boundary_is_ambiguous"],
            "threshold_tie_count": current_topk["threshold_tie_count"],
            "selected_from_tie": current_topk["selected_from_tie"],
        },
        "final_boundary": {
            "threshold": final_topk["threshold"],
            "boundary_is_ambiguous": final_topk["boundary_is_ambiguous"],
            "threshold_tie_count": final_topk["threshold_tie_count"],
            "selected_from_tie": final_topk["selected_from_tie"],
        },
    }
    return entry, current_names


def _adjacent_entry(prev_names: set[str], curr_names: set[str]) -> dict:
    births = curr_names - prev_names
    deaths = prev_names - curr_names
    return {
        "iou": compute_iou(prev_names, curr_names),
        "birth_count": len(births),
        "death_count": len(deaths),
        "birth_rate": len(births) / len(curr_names) if curr_names else 0.0,
        "death_rate": len(deaths) / len(prev_names) if prev_names else 0.0,
        "churn_rate": len(curr_names ^ prev_names) / len(curr_names | prev_names) if (curr_names | prev_names) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute convergence metrics vs final")
    parser.add_argument("--scores_dir", type=str, required=True, help="Directory with scores_*.json files")
    parser.add_argument("--final_label", type=str, default="final", help="Label of the final checkpoint")
    parser.add_argument("--top_k_percent", type=float, default=20.0, help="Default top-k percent")
    parser.add_argument(
        "--top_k_percents",
        type=str,
        default=None,
        help="Optional comma-separated top-k sweep, e.g. '5,10,20,30,40'",
    )
    parser.add_argument("--output", type=str, default="convergence.json")
    args = parser.parse_args()

    top_ks = parse_float_list(args.top_k_percents, default=[args.top_k_percent])
    default_top_k = args.top_k_percent
    if default_top_k not in top_ks:
        top_ks.append(default_top_k)
        top_ks = sorted(set(top_ks))

    pattern = os.path.join(args.scores_dir, "scores_*.json")
    score_files = sorted(glob.glob(pattern))
    if not score_files:
        print(f"No score files found matching {pattern}")
        sys.exit(1)

    all_data = {}
    for path in score_files:
        data = load_scores(path)
        all_data[data["label"]] = data

    if args.final_label not in all_data:
        print(f"Final label '{args.final_label}' not found. Available: {sorted(all_data.keys())}")
        sys.exit(1)

    final = all_data[args.final_label]
    final_frac = final["fraction_scores"]
    final_l2 = final["l2_scores"]

    print(f"Found {len(score_files)} score files")
    print(f"Final checkpoint: {args.final_label}")
    print(f"Top-k sweep: {top_ks}")

    intermediates = []
    for label, data in all_data.items():
        if label == args.final_label:
            continue
        if data["step"] is None:
            continue
        intermediates.append((label, data))
    intermediates.sort(key=lambda x: x[1]["step"])

    results = []
    adjacent = []
    prev_selected = {"fraction": {}, "l2": {}}
    prev_meta = None

    for label, data in intermediates:
        frac_scores = data["fraction_scores"]
        l2_scores = data["l2_scores"]

        per_topk_fraction = {}
        per_topk_l2 = {}
        current_selected = {"fraction": {}, "l2": {}}

        for top_k in top_ks:
            frac_entry, frac_names = _metric_entry(frac_scores, final_frac, top_k)
            l2_entry, l2_names = _metric_entry(l2_scores, final_l2, top_k)
            per_topk_fraction[str(top_k)] = frac_entry
            per_topk_l2[str(top_k)] = l2_entry
            current_selected["fraction"][str(top_k)] = frac_names
            current_selected["l2"][str(top_k)] = l2_names

        entry = {
            "label": label,
            "step": data["step"],
            "fraction": per_topk_fraction[str(default_top_k)],
            "l2": per_topk_l2[str(default_top_k)],
            "per_topk": {
                "fraction": per_topk_fraction,
                "l2": per_topk_l2,
            },
        }
        results.append(entry)

        if prev_meta is not None:
            drift_entry = {
                "label": label,
                "step": data["step"],
                "previous_label": prev_meta["label"],
                "previous_step": prev_meta["step"],
                "fraction": {},
                "l2": {},
            }
            for method in ("fraction", "l2"):
                for top_k in top_ks:
                    key = str(top_k)
                    drift_entry[method][key] = _adjacent_entry(prev_selected[method][key], current_selected[method][key])
            adjacent.append(drift_entry)

        prev_selected = current_selected
        prev_meta = {"label": label, "step": data["step"]}

        print(
            f"  step {data['step']:>4}: "
            f"frac IoU={entry['fraction']['iou']:.3f} rho={entry['fraction']['spearman']:.3f} | "
            f"L2 IoU={entry['l2']['iou']:.3f} rho={entry['l2']['spearman']:.3f}"
        )

    final_cross = {
        "score_correlation": compute_score_correlation(final_frac, final_l2),
        "topk_overlap": {},
    }
    for top_k in top_ks:
        frac_topk = select_topk(final_frac, top_k)
        l2_topk = select_topk(final_l2, top_k)
        final_cross["topk_overlap"][str(top_k)] = {
            "iou": compute_iou(frac_topk["names"], l2_topk["names"]),
            "fraction_boundary_is_ambiguous": frac_topk["boundary_is_ambiguous"],
            "l2_boundary_is_ambiguous": l2_topk["boundary_is_ambiguous"],
        }

    output = {
        "top_k_percent": default_top_k,
        "top_k_percents": top_ks,
        "final_label": args.final_label,
        "n_checkpoints": len(results),
        "convergence": results,
        "adjacent_drift": adjacent,
        "cross_method": final_cross,
        "metadata": {
            "scores_dir": args.scores_dir,
            "threshold": final.get("threshold"),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"\nCross-method (final @ top-{default_top_k}%): "
        f"IoU={final_cross['topk_overlap'][str(default_top_k)]['iou']:.3f}, "
        f"Spearman={final_cross['score_correlation']['spearman']:.3f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
