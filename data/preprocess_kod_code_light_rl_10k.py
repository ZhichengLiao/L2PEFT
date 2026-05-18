#!/usr/bin/env python3
"""
Preprocess KodCode/kod_code-light-rl-10_k for veRL code GRPO training.

The raw cache contains one Arrow train file with Python coding tasks and pytest
style tests. This script keeps a stable subset that can be converted into a
single zero-argument assertion runner for Sandbox Fusion.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import datasets


DATA_SOURCE = "kod_code_light_rl_10k"
DEFAULT_RAW_DATASET_PATH = (
    "/nfs-stor/zhengqing.gao/yuhao.wu/lzc/hf_cache/datasets/"
    "KodCode___kod_code-light-rl-10_k/default/0.0.0/"
    "dcf78a8bbba9a613b596ce993c4921a38687dfcc"
)
ARROW_FILE_NAME = "kod_code-light-rl-10_k-train.arrow"
INSTRUCTION = (
    "Write a correct Python solution for the task. Return only one Python code block "
    "containing the implementation. Do not include tests or explanations."
)
HEAVY_IMPORT_PREFIXES = (
    "cv2",
    "django",
    "flask",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "requests",
    "scipy",
    "seaborn",
    "sklearn",
    "torch",
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

    arrow_path = path / ARROW_FILE_NAME
    if arrow_path.exists():
        return arrow_path

    raise FileNotFoundError(f"Could not find {ARROW_FILE_NAME}. Checked: {path}")


def require_columns(dataset: datasets.Dataset) -> None:
    required = {"question", "solution", "test", "test_info", "subset", "question_id", "gpt_difficulty"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")


def source_import_modules(tree: ast.Module) -> list[str]:
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


def has_heavy_import(tree: ast.Module) -> bool:
    for module in source_import_modules(tree):
        if module.startswith(HEAVY_IMPORT_PREFIXES):
            return True
    return False


def test_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test")]


def filter_reason(question: str, test_source: str) -> tuple[str | None, ast.Module | None, list[ast.FunctionDef]]:
    if not question.strip():
        return "empty_question", None, []
    if not test_source.strip():
        return "empty_test", None, []

    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return "test_syntax_error", None, []

    tests = test_functions(tree)
    if not tests:
        return "no_test_function", tree, tests
    if any(test.decorator_list for test in tests):
        return "decorated_test_function", tree, tests
    if any(test.args.args or test.args.kwonlyargs or test.args.vararg or test.args.kwarg for test in tests):
        return "test_function_with_args", tree, tests
    if has_heavy_import(tree):
        return "heavy_import", tree, tests

    return None, tree, tests


def strip_solution_from_imports(test_source: str) -> str:
    """Drop simple `from solution import ...` lines.

    The runner registers a fake `solution` module anyway. Keeping `import
    solution` is useful for the rare tests that access `solution.foo`.
    """
    output_lines: list[str] = []
    skipping_multiline_from_solution = False

    for line in test_source.splitlines():
        stripped = line.strip()
        if skipping_multiline_from_solution:
            if ")" in stripped:
                skipping_multiline_from_solution = False
            continue

        if re.match(r"^from\s+solution\s+import\b", stripped):
            if "(" in stripped and ")" not in stripped:
                skipping_multiline_from_solution = True
            continue

        output_lines.append(line)

    return "\n".join(output_lines).strip()


def build_assert_case(test_source: str, test_names: list[str]) -> str:
    cleaned_test = strip_solution_from_imports(test_source)
    calls = "\n".join(f"{name}()" for name in test_names)
    prelude = """
import sys as _kod_sys
import types as _kod_types

_kod_solution = _kod_types.ModuleType("solution")
_kod_solution.__dict__.update(globals())
_kod_sys.modules["solution"] = _kod_solution
"""
    return "\n".join(
        part.strip()
        for part in (
            textwrap.dedent(prelude),
            cleaned_test,
            "# Run converted pytest-style tests.",
            calls,
        )
        if part.strip()
    ) + "\n"


def get_function_name(example: dict[str, Any]) -> str:
    test_info = example.get("test_info") or []
    if test_info and isinstance(test_info[0], dict):
        return as_text(test_info[0].get("function_name")).strip()
    return ""


def add_count(mapping: dict[str, int], key: str) -> None:
    mapping[key] = int(mapping.get(key, 0)) + 1


def empty_stats(args: argparse.Namespace, arrow_path: Path) -> dict[str, Any]:
    return {
        "dataset": "KodCode/kod_code-light-rl-10_k",
        "data_source": DATA_SOURCE,
        "raw_arrow_path": str(arrow_path),
        "seed": args.seed,
        "test_size": args.test_size,
        "raw_rows": 0,
        "kept_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "skipped_rows": 0,
        "skip_reason_counts": {},
        "subset_counts_raw": {},
        "subset_counts_kept": {},
        "difficulty_counts_raw": {},
        "difficulty_counts_kept": {},
        "test_count_distribution_kept": {},
    }


def build_rows(raw_dataset: datasets.Dataset, args: argparse.Namespace, stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_count_distribution: Counter[int] = Counter()

    for idx, example in enumerate(raw_dataset):
        stats["raw_rows"] += 1

        question = as_text(example.get("question")).strip()
        solution = as_text(example.get("solution")).strip()
        test_source = as_text(example.get("test")).strip()
        subset = as_text(example.get("subset")).strip()
        difficulty = as_text(example.get("gpt_difficulty")).strip()
        question_id = as_text(example.get("question_id")).strip()

        add_count(stats["subset_counts_raw"], subset)
        add_count(stats["difficulty_counts_raw"], difficulty)

        reason, _tree, tests = filter_reason(question, test_source)
        if reason is not None:
            stats["skipped_rows"] += 1
            add_count(stats["skip_reason_counts"], reason)
            continue
        if not solution:
            stats["skipped_rows"] += 1
            add_count(stats["skip_reason_counts"], "empty_reference_solution")
            continue

        test_names = [test.name for test in tests]
        assert_case = build_assert_case(test_source, test_names)
        ground_truth = json.dumps({"assert_case": [assert_case]}, ensure_ascii=False)
        function_name = get_function_name(example)

        add_count(stats["subset_counts_kept"], subset)
        add_count(stats["difficulty_counts_kept"], difficulty)
        test_count_distribution[len(test_names)] += 1

        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}],
                "ability": "code",
                "reward_model": {"style": "sandbox_fusion", "ground_truth": ground_truth},
                "extra_info": {
                    "split": "",
                    "index": idx,
                    "question_id": question_id,
                    "subset": subset,
                    "difficulty": difficulty,
                    "function_name": function_name,
                    "test_count": len(test_names),
                    "question_chars": len(question),
                    "test_chars": len(test_source),
                    "solution_chars": len(solution),
                },
            }
        )

    stats["kept_rows"] = len(rows)
    stats["test_count_distribution_kept"] = {str(k): v for k, v in sorted(test_count_distribution.items())}
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
    parser = argparse.ArgumentParser(description="Preprocess KodCode light RL 10k for veRL code GRPO.")
    parser.add_argument(
        "--raw_dataset_path",
        default=DEFAULT_RAW_DATASET_PATH,
        help="Local cache directory or Arrow file for KodCode/kod_code-light-rl-10_k.",
    )
    parser.add_argument(
        "--local_save_dir",
        default=str(PROJECT_ROOT / "data" / "KodCodeLightRL10K"),
        help="Directory to save train.parquet, test.parquet, stats.json, and examples.json.",
    )
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrow_path = resolve_arrow_path(args.raw_dataset_path)

    print(f"Loading KodCode/kod_code-light-rl-10_k from {arrow_path}")
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
