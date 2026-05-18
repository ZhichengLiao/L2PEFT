#!/usr/bin/env python3
"""
Preprocess nvidia/Nemotron-Cascade-RL-Math for veRL GRPO training.

The raw local cache contains one Arrow train file with columns:
problem, answer, source. This script filters long problems by raw character
length, creates a deterministic train/test split, and writes veRL-compatible
parquet files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import datasets


DATA_SOURCE = "nvidia/Nemotron-Cascade-RL-Math"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
DEFAULT_RAW_DATASET_PATH = (
    "/nfs-stor/zhengqing.gao/yuhao.wu/lzc/hf_cache/datasets/"
    "nvidia___nemotron-cascade-rl-math/default/0.0.0/"
    "fcd07b1417edc12cf4642cae7d269e4f3a84c812"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def resolve_arrow_path(raw_dataset_path: str) -> Path:
    path = Path(raw_dataset_path).expanduser().resolve()
    if path.is_file():
        return path

    arrow_path = path / "nemotron-cascade-rl-math-train.arrow"
    if arrow_path.exists():
        return arrow_path

    raise FileNotFoundError(
        "Could not find nemotron-cascade-rl-math-train.arrow. "
        f"Pass --raw_dataset_path explicitly. Checked: {path}"
    )


def require_columns(dataset: datasets.Dataset) -> None:
    required = {"problem", "answer", "source"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")


def empty_stats(args: argparse.Namespace, arrow_path: Path) -> dict[str, Any]:
    return {
        "dataset": DATA_SOURCE,
        "raw_arrow_path": str(arrow_path),
        "seed": args.seed,
        "max_problem_chars": args.max_problem_chars,
        "test_size": args.test_size,
        "raw_rows": 0,
        "kept_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "skipped_empty_problem": 0,
        "skipped_empty_answer": 0,
        "skipped_overlong_problem": 0,
        "source_counts_raw": {},
        "source_counts_kept": {},
    }


def add_count(mapping: dict[str, int], key: str) -> None:
    mapping[key] = int(mapping.get(key, 0)) + 1


def build_rows(raw_dataset: datasets.Dataset, args: argparse.Namespace, stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, example in enumerate(raw_dataset):
        stats["raw_rows"] += 1

        problem = as_text(example["problem"]).strip()
        answer = as_text(example["answer"]).strip()
        source = as_text(example["source"]).strip()

        add_count(stats["source_counts_raw"], source)

        if not problem:
            stats["skipped_empty_problem"] += 1
            continue
        if not answer:
            stats["skipped_empty_answer"] += 1
            continue

        problem_chars = len(problem)
        if problem_chars > args.max_problem_chars:
            stats["skipped_overlong_problem"] += 1
            continue

        add_count(stats["source_counts_kept"], source)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "",
                    "index": idx,
                    "source": source,
                    "problem_chars": problem_chars,
                },
            }
        )

    stats["kept_rows"] = len(rows)
    return rows


def split_rows(rows: list[dict[str, Any]], test_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_size < 0:
        raise ValueError("--test_size must be non-negative")
    if test_size >= len(rows):
        raise ValueError(f"--test_size must be smaller than kept rows ({len(rows)}), got {test_size}")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    test_rows = shuffled[:test_size]
    train_rows = shuffled[test_size:]

    for row in train_rows:
        row["extra_info"]["split"] = "train"
    for row in test_rows:
        row["extra_info"]["split"] = "test"

    return train_rows, test_rows


def save_outputs(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    output_dir: Path,
    stats: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.Dataset.from_list(train_rows)
    test_dataset = datasets.Dataset.from_list(test_rows)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    examples: dict[str, Any] = {}
    if train_rows:
        examples["train"] = train_rows[0]
    if test_rows:
        examples["test"] = test_rows[0]
    with (output_dir / "examples.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(train_rows)} train rows to {train_path}")
    print(f"Saved {len(test_rows)} test rows to {test_path}")
    print(f"Saved stats to {output_dir / 'stats.json'}")
    print(f"Saved examples to {output_dir / 'examples.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Nemotron-Cascade-RL-Math for veRL GRPO.")
    parser.add_argument(
        "--raw_dataset_path",
        default=DEFAULT_RAW_DATASET_PATH,
        help="Local cache directory or Arrow file for nvidia/Nemotron-Cascade-RL-Math.",
    )
    parser.add_argument(
        "--local_save_dir",
        default=str(PROJECT_ROOT / "data" / "NemotronCascadeMath"),
        help="Directory to save train.parquet, test.parquet, stats.json, and examples.json.",
    )
    parser.add_argument("--max_problem_chars", type=int, default=1024)
    parser.add_argument("--test_size", type=int, default=280)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrow_path = resolve_arrow_path(args.raw_dataset_path)

    print(f"Loading {DATA_SOURCE} from {arrow_path}")
    raw_dataset = datasets.Dataset.from_file(str(arrow_path))
    require_columns(raw_dataset)

    stats = empty_stats(args, arrow_path)
    rows = build_rows(raw_dataset, args, stats)
    train_rows, test_rows = split_rows(rows, args.test_size, args.seed)
    stats["train_rows"] = len(train_rows)
    stats["test_rows"] = len(test_rows)

    output_dir = Path(args.local_save_dir).expanduser().resolve()
    save_outputs(train_rows, test_rows, output_dir, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
