#!/usr/bin/env python3
"""
Join convergence metrics with trainer logs and quantify coupling with reward,
KL, and entropy dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from common import pearson_correlation, save_json, spearman_rank_correlation


def load_metrics(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def detect_column(fieldnames: list[str], explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        return explicit
    lowered = {name.lower(): name for name in fieldnames}
    for pattern in candidates:
        for name in fieldnames:
            if pattern in name.lower():
                return name
    return None


def safe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Couple convergence with training metrics")
    parser.add_argument("--convergence", type=str, required=True)
    parser.add_argument("--metrics", type=str, required=True, help="CSV or JSONL with training metrics")
    parser.add_argument("--top_k_percent", type=float, default=None, help="Choose a top-k slice from convergence.json")
    parser.add_argument("--step_field", type=str, default="step")
    parser.add_argument("--reward_field", type=str, default=None)
    parser.add_argument("--kl_field", type=str, default=None)
    parser.add_argument("--entropy_field", type=str, default=None)
    parser.add_argument("--output", type=str, default="dynamics_coupling.json")
    args = parser.parse_args()

    with open(args.convergence) as f:
        convergence = json.load(f)

    top_k = args.top_k_percent if args.top_k_percent is not None else convergence["top_k_percent"]
    top_k_key = str(top_k)

    conv_by_step = {}
    for entry in convergence["convergence"]:
        fraction = entry["per_topk"]["fraction"][top_k_key] if "per_topk" in entry else entry["fraction"]
        l2 = entry["per_topk"]["l2"][top_k_key] if "per_topk" in entry else entry["l2"]
        conv_by_step[entry["step"]] = {
            "step": entry["step"],
            "fraction_iou": fraction["iou"],
            "fraction_spearman": fraction["spearman"],
            "l2_iou": l2["iou"],
            "l2_spearman": l2["spearman"],
        }

    for entry in convergence.get("adjacent_drift", []):
        step = entry["step"]
        fraction = entry["fraction"][top_k_key]
        l2 = entry["l2"][top_k_key]
        if step in conv_by_step:
            conv_by_step[step].update(
                {
                    "fraction_birth_rate": fraction["birth_rate"],
                    "fraction_death_rate": fraction["death_rate"],
                    "fraction_churn_rate": fraction["churn_rate"],
                    "l2_birth_rate": l2["birth_rate"],
                    "l2_death_rate": l2["death_rate"],
                    "l2_churn_rate": l2["churn_rate"],
                }
            )

    metric_rows = load_metrics(args.metrics)
    if not metric_rows:
        raise ValueError("No metrics rows found")

    fieldnames = list(metric_rows[0].keys())
    reward_col = detect_column(fieldnames, args.reward_field, ["reward"])
    kl_col = detect_column(fieldnames, args.kl_field, ["kl"])
    entropy_col = detect_column(fieldnames, args.entropy_field, ["entropy"])

    joined = []
    for row in metric_rows:
        if args.step_field not in row:
            continue
        raw_step = row[args.step_field]
        if raw_step is None or raw_step == "":
            continue
        try:
            step = int(float(raw_step))
        except (ValueError, TypeError):
            continue
        if step not in conv_by_step:
            continue
        record = dict(conv_by_step[step])
        record["reward"] = safe_float(row.get(reward_col)) if reward_col else None
        record["kl"] = safe_float(row.get(kl_col)) if kl_col else None
        record["entropy"] = safe_float(row.get(entropy_col)) if entropy_col else None
        joined.append(record)

    analysis_metrics = [
        "fraction_iou",
        "fraction_spearman",
        "l2_iou",
        "l2_spearman",
        "fraction_birth_rate",
        "fraction_death_rate",
        "fraction_churn_rate",
        "l2_birth_rate",
        "l2_death_rate",
        "l2_churn_rate",
    ]
    training_metrics = ["reward", "kl", "entropy"]

    correlations = []
    for train_key in training_metrics:
        for analysis_key in analysis_metrics:
            pairs = [
                (row[train_key], row[analysis_key])
                for row in joined
                if row.get(train_key) is not None and row.get(analysis_key) is not None
            ]
            if len(pairs) < 2:
                continue
            train_values = [a for a, _ in pairs]
            analysis_values = [b for _, b in pairs]
            correlations.append(
                {
                    "training_metric": train_key,
                    "analysis_metric": analysis_key,
                    "n_points": len(pairs),
                    "pearson": pearson_correlation(train_values, analysis_values),
                    "spearman": spearman_rank_correlation(train_values, analysis_values),
                }
            )

    output = {
        "top_k_percent": top_k,
        "step_field": args.step_field,
        "detected_columns": {"reward": reward_col, "kl": kl_col, "entropy": entropy_col},
        "n_joined_rows": len(joined),
        "joined_rows": joined,
        "correlations": correlations,
    }

    save_json(output, args.output)
    print(f"Joined rows: {len(joined)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
