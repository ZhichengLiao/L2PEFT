"""
Preprocess the NuminaMath-CoT dataset (first 20K entries) to parquet format for veRL GRPO training.

Dataset: AI-MO/NuminaMath-CoT
- 860K math problems with CoT solutions, answers in \\boxed{} format
- 9 source categories: cn_k12, synthetic_math, orca_math, olympiads, synthetic_amc,
  aops_forum, math, gsm8k, amc_aime

Usage:
    python preprocess_numina.py --local_save_dir ~/data/numina_math
    python preprocess_numina.py --local_save_dir ~/data/numina_math --local_dataset_path /path/to/cached/dataset
"""

import argparse
import json
import os
import re

import datasets


# --- Answer extraction (reuse veRL's boxed extraction logic) ---

def last_boxed_only_string(string):
    """Extract the content inside the last \\boxed{} or \\fbox{} in the string."""
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


def extract_solution(solution_str):
    """Extract the ground truth answer from the solution's \\boxed{} expression."""
    return last_boxed_only_string(solution_str)


# --- Source mapping ---

# NuminaMath source -> veRL data_source for reward routing
# All mapped to numina_* to use prime_math.compute_score()
# which handles \\boxed{} extraction + sympy equivalence checking
SOURCE_MAP = {
    "aops_forum": "numina_aops_forum",
    "synthetic_math": "numina_synthetic_math",
    "amc_aime": "numina_amc_aime",
    "synthetic_amc": "numina_synthetic_amc",
    "cn_k12": "numina_cn_k12",
    "olympiads": "numina_olympiads",
    # No direct veRL route; map to numina_cn_k12 (prime_math handles all boxed answers)
    "orca_math": "numina_cn_k12",
    "gsm8k": "numina_cn_k12",
    "math": "numina_cn_k12",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NuminaMath-CoT for veRL GRPO training")
    parser.add_argument(
        "--local_dataset_path", default=None,
        help="Local path to cached NuminaMath-CoT dataset (e.g. HF cache dir).",
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/numina_math",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--num_train", type=int, default=20000,
        help="Number of training examples to use from the dataset.",
    )

    args = parser.parse_args()

    hf_dataset_id = "AI-MO/NuminaMath-CoT"
    print(f"Loading {hf_dataset_id}...", flush=True)

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(hf_dataset_id)

    # Take first N entries from train split
    train_dataset = dataset["train"].select(range(min(args.num_train, len(dataset["train"]))))
    # Use the native test split (100 examples)
    test_dataset = dataset["test"]

    print(f"Train split: {len(train_dataset)} examples", flush=True)
    print(f"Test split: {len(test_dataset)} examples", flush=True)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Statistics
    stats = {"total": 0, "skipped_no_boxed": 0, "by_source": {}}

    def make_map_fn(split):
        def process_fn(example, idx):
            stats["total"] += 1
            source = example["source"]
            problem = example["problem"]
            solution = example["solution"]

            # Extract ground truth from \boxed{}
            ground_truth = extract_solution(solution)

            # Map source to veRL data_source
            data_source = SOURCE_MAP.get(source, "numina_cn_k12")

            # Track stats
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

            # If no boxed answer found, use empty string (will get reward=0 during eval)
            if ground_truth is None:
                stats["skipped_no_boxed"] += 1
                ground_truth = ""

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": problem + " " + instruction_following}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"split": split, "index": idx, "source": source},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    stats_train = dict(stats)
    stats = {"total": 0, "skipped_no_boxed": 0, "by_source": {}}
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Filter out entries with empty ground_truth
    train_before = len(train_dataset)
    test_before = len(test_dataset)
    train_dataset = train_dataset.filter(lambda x: x["reward_model"]["ground_truth"] != "")
    test_dataset = test_dataset.filter(lambda x: x["reward_model"]["ground_truth"] != "")

    print(f"\nTrain: {train_before} → {len(train_dataset)} (filtered {train_before - len(train_dataset)} without \\boxed{{}})")
    print(f"Test: {test_before} → {len(test_dataset)} (filtered {test_before - len(test_dataset)} without \\boxed{{}})")
    print(f"\nSource distribution (train):")
    for src, cnt in sorted(stats_train["by_source"].items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    # Save
    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Save examples for inspection
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {local_dir}/")
    print(f"  train.parquet: {len(train_dataset)} rows")
    print(f"  test.parquet: {len(test_dataset)} rows")
    print("Done!")
