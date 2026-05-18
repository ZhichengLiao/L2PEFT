#!/usr/bin/env python3
"""
Preprocess a small, easier NuminaMath-CoT subset for veRL GRPO training.

The default curriculum targets weaker base models by sampling mostly shorter
word/arithmetic problems:

    gsm8k=6900, orca_math=5100, synthetic_math=2000

Rows are filtered before sampling to keep boxed, short-answer examples with
moderate problem/solution length. The output format matches veRL PPO/GRPO
parquet datasets.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import datasets
import pyarrow.parquet as pq


DATASET_ID = "AI-MO/NuminaMath-CoT"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
DEFAULT_RAW_DATASET_PATH = (
    "/nfs-stor/zhengqing.gao/yuhao.wu/lzc/hf_cache/hub/"
    "datasets--AI-MO--NuminaMath-CoT/snapshots/9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
)
DEFAULT_SOURCE_QUOTAS = "gsm8k=6900,orca_math=5100,synthetic_math=2000"
DEFAULT_ALLOWED_ANSWER_KINDS = "integer,decimal,frac,sqrt,pi,tuple_or_list,other"

SOURCE_MAP = {
    "aops_forum": "numina_aops_forum",
    "synthetic_math": "numina_synthetic_math",
    "amc_aime": "numina_amc_aime",
    "synthetic_amc": "numina_synthetic_amc",
    "cn_k12": "numina_cn_k12",
    "olympiads": "numina_olympiads",
    "orca_math": "numina_cn_k12",
    "gsm8k": "numina_cn_k12",
    "math": "numina_cn_k12",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def parse_source_quotas(value: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid quota entry {item!r}; expected source=count")
        source, count = item.split("=", 1)
        source = source.strip()
        count = int(count.strip())
        if count <= 0:
            raise ValueError(f"Quota for {source} must be positive, got {count}")
        quotas[source] = count
    if not quotas:
        raise ValueError("--source_quotas must select at least one source")
    return quotas


def parse_allowed_kinds(value: str) -> set[str]:
    kinds = {item.strip() for item in value.split(",") if item.strip()}
    if not kinds:
        raise ValueError("--allowed_answer_kinds must include at least one kind")
    return kinds


def resolve_train_files(raw_dataset_path: str) -> list[Path]:
    path = Path(raw_dataset_path).expanduser().resolve()
    if path.is_file():
        return [path]

    data_dir = path / "data"
    candidates = sorted(glob.glob(str(data_dir / "train-*.parquet")))
    if not candidates:
        candidates = sorted(glob.glob(str(path / "train-*.parquet")))
    if not candidates:
        raise FileNotFoundError(f"Could not find train parquet shards under {path}")
    return [Path(candidate) for candidate in candidates]


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None
    return string[left_brace_idx + 1 : right_brace_idx].strip()


def answer_kind(answer: str) -> str:
    normalized = answer.strip().strip("$")
    if not normalized:
        return "empty"
    if re.fullmatch(r"-?\d+", normalized):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", normalized):
        return "decimal"
    if "\\frac" in normalized:
        return "frac"
    if "\\sqrt" in normalized:
        return "sqrt"
    if "\\pi" in normalized or "π" in normalized:
        return "pi"
    if "," in normalized or (len(normalized) >= 2 and normalized[0] in "({[" and normalized[-1] in ")}]"):
        return "tuple_or_list"
    if re.search(r"[A-Za-z]", normalized):
        return "symbolic_text"
    return "other"


def empty_stats(args: argparse.Namespace, train_files: list[Path], quotas: dict[str, int]) -> dict[str, Any]:
    return {
        "dataset": DATASET_ID,
        "raw_train_files": [str(path) for path in train_files],
        "seed": args.seed,
        "source_quotas": quotas,
        "test_size": args.test_size,
        "filters": {
            "max_problem_chars": args.max_problem_chars,
            "max_solution_chars": args.max_solution_chars,
            "max_answer_chars": args.max_answer_chars,
            "allowed_answer_kinds": sorted(parse_allowed_kinds(args.allowed_answer_kinds)),
        },
        "raw_rows": 0,
        "eligible_rows_by_source": {},
        "selected_rows_by_source": {},
        "train_rows": 0,
        "test_rows": 0,
        "skipped": {},
        "answer_kinds_selected": {},
    }


def add_skip(stats: dict[str, Any], reason: str) -> None:
    skipped = stats["skipped"]
    skipped[reason] = int(skipped.get(reason, 0)) + 1


def is_eligible(
    source: str,
    problem: str,
    solution: str,
    answer: str | None,
    kind: str,
    quotas: dict[str, int],
    allowed_kinds: set[str],
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> bool:
    if source not in quotas:
        add_skip(stats, "source_not_selected")
        return False
    if not problem:
        add_skip(stats, "empty_problem")
        return False
    if answer is None or not answer:
        add_skip(stats, "missing_boxed_answer")
        return False
    if len(problem) > args.max_problem_chars:
        add_skip(stats, "overlong_problem")
        return False
    if len(solution) > args.max_solution_chars:
        add_skip(stats, "overlong_solution")
        return False
    if len(answer) > args.max_answer_chars:
        add_skip(stats, "overlong_answer")
        return False
    if kind not in allowed_kinds:
        add_skip(stats, "answer_kind_not_allowed")
        return False
    return True


def make_row(split: str, idx: int, source: str, problem: str, answer: str, kind: str, solution_chars: int) -> dict[str, Any]:
    return {
        "data_source": SOURCE_MAP.get(source, "numina_cn_k12"),
        "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "source": source,
            "problem_chars": len(problem),
            "solution_chars": solution_chars,
            "answer_kind": kind,
        },
    }


def collect_eligible_rows(
    train_files: list[Path],
    quotas: dict[str, int],
    allowed_kinds: set[str],
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_idx = 0

    for file_path in train_files:
        parquet_file = pq.ParquetFile(file_path)
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=["source", "problem", "solution"])
            for example in table.to_pylist():
                stats["raw_rows"] += 1
                source = as_text(example["source"]).strip()
                problem = as_text(example["problem"]).strip()
                solution = as_text(example["solution"]).strip()
                answer = last_boxed_only_string(solution)
                kind = answer_kind(answer or "")
                if is_eligible(source, problem, solution, answer, kind, quotas, allowed_kinds, args, stats):
                    rows_by_source[source].append(
                        make_row("", global_idx, source, problem, answer or "", kind, len(solution))
                    )
                global_idx += 1

    stats["eligible_rows_by_source"] = {source: len(rows) for source, rows in sorted(rows_by_source.items())}
    return rows_by_source


def sample_rows(
    rows_by_source: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    seed: int,
    allow_underfilled_quotas: bool,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}

    for source, quota in quotas.items():
        rows = list(rows_by_source.get(source, []))
        if len(rows) < quota and not allow_underfilled_quotas:
            raise ValueError(f"Source {source} has {len(rows)} eligible rows but quota requires {quota}")
        random.Random(f"{seed}:{source}").shuffle(rows)
        chosen = rows[: min(quota, len(rows))]
        selected.extend(chosen)
        selected_counts[source] = len(chosen)

    random.Random(seed).shuffle(selected)
    stats["selected_rows_by_source"] = selected_counts
    stats["answer_kinds_selected"] = dict(Counter(row["extra_info"]["answer_kind"] for row in selected))
    return selected


def split_rows(rows: list[dict[str, Any]], test_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_size < 0:
        raise ValueError("--test_size must be non-negative")
    if test_size >= len(rows):
        raise ValueError(f"--test_size must be smaller than selected rows ({len(rows)}), got {test_size}")

    test_rows = rows[:test_size]
    train_rows = rows[test_size:]

    for row in train_rows:
        row["extra_info"]["split"] = "train"
    for row in test_rows:
        row["extra_info"]["split"] = "test"
    return train_rows, test_rows


def save_outputs(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], output_dir: Path, stats: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.Dataset.from_list(train_rows)
    test_dataset = datasets.Dataset.from_list(test_rows)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    examples = {}
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
    parser = argparse.ArgumentParser(description="Preprocess a small NuminaMath-CoT subset for veRL GRPO.")
    parser.add_argument("--raw_dataset_path", default=DEFAULT_RAW_DATASET_PATH)
    parser.add_argument("--local_save_dir", default=str(PROJECT_ROOT / "data" / "NuminaMathSubset"))
    parser.add_argument("--source_quotas", default=DEFAULT_SOURCE_QUOTAS)
    parser.add_argument("--allowed_answer_kinds", default=DEFAULT_ALLOWED_ANSWER_KINDS)
    parser.add_argument("--max_problem_chars", type=int, default=512)
    parser.add_argument("--max_solution_chars", type=int, default=1800)
    parser.add_argument("--max_answer_chars", type=int, default=32)
    parser.add_argument("--test_size", type=int, default=280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_underfilled_quotas", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quotas = parse_source_quotas(args.source_quotas)
    allowed_kinds = parse_allowed_kinds(args.allowed_answer_kinds)
    train_files = resolve_train_files(args.raw_dataset_path)
    stats = empty_stats(args, train_files, quotas)

    print(f"Loading {DATASET_ID} train shards from {Path(args.raw_dataset_path).expanduser().resolve()}")
    rows_by_source = collect_eligible_rows(train_files, quotas, allowed_kinds, args, stats)
    selected_rows = sample_rows(rows_by_source, quotas, args.seed, args.allow_underfilled_quotas, stats)
    train_rows, test_rows = split_rows(selected_rows, args.test_size)
    stats["train_rows"] = len(train_rows)
    stats["test_rows"] = len(test_rows)

    output_dir = Path(args.local_save_dir).expanduser().resolve()
    save_outputs(train_rows, test_rows, output_dir, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
