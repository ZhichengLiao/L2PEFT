#!/usr/bin/env python3
"""
Preprocess Dahoas/full-hh-rlhf from the local Hugging Face cache for veRL.

Default output is the PPO/RL format expected by verl.trainer.main_ppo:

    data/hh_rlhf/train.parquet
    data/hh_rlhf/test.parquet

The raw dataset stores prompts as Anthropic-style transcripts:

    Human: ...
    Assistant: ...
    Human: ...
    Assistant:

For PPO, this script parses those transcripts into chat messages and drops the
final empty assistant marker. The actor will then generate the next assistant
message, while the reward model scores the generated response.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import datasets
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


DATA_SOURCE = "Dahoas/full-hh-rlhf"
TURN_RE = re.compile(r"(?m)^\s*(Human|Assistant):\s*")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
    "{{ message['content'] | trim }}"
    "{{ '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def iter_progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def resolve_raw_dataset_path(raw_dataset_path: str | None) -> Path:
    if raw_dataset_path:
        path = Path(raw_dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"raw dataset path does not exist: {path}")
        return path

    candidates: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub" / "datasets--Dahoas--full-hh-rlhf" / "snapshots")
    candidates.append(WORKSPACE_ROOT / "hf_cache" / "hub" / "datasets--Dahoas--full-hh-rlhf" / "snapshots")

    for snapshots_dir in candidates:
        if not snapshots_dir.exists():
            continue
        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir() and (p / "data").is_dir()]
        if snapshots:
            snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return snapshots[0]

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find cached Dahoas/full-hh-rlhf snapshot. "
        f"Pass --raw_dataset_path explicitly. Searched:\n{searched}"
    )


def find_split_parquet(raw_dataset_path: Path, split: str) -> Path:
    search_dirs = [raw_dataset_path / "data", raw_dataset_path]
    for directory in search_dirs:
        if not directory.exists():
            continue
        matches = sorted(directory.glob(f"{split}-*.parquet"))
        if matches:
            return matches[0]
        direct = directory / f"{split}.parquet"
        if direct.exists():
            return direct
    raise FileNotFoundError(f"Could not find {split} parquet under {raw_dataset_path}")


def load_raw_dataset(raw_dataset_path: Path) -> datasets.DatasetDict:
    data_files = {
        "train": str(find_split_parquet(raw_dataset_path, "train")),
        "test": str(find_split_parquet(raw_dataset_path, "test")),
    }
    return datasets.load_dataset("parquet", data_files=data_files)


def maybe_limit_split(
    split: datasets.Dataset,
    limit: int | None,
    seed: int,
    shuffle_before_limit: bool,
) -> datasets.Dataset:
    if limit is None or limit < 0 or limit >= len(split):
        return split
    if shuffle_before_limit:
        split = split.shuffle(seed=seed)
    return split.select(range(limit))


def parse_hh_prompt(prompt_text: str) -> tuple[list[dict[str, str]], str]:
    text = prompt_text.strip()
    matches = list(TURN_RE.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}], "fallback_no_markers"

    messages: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        raw_role = match.group(1)
        role = "user" if raw_role == "Human" else "assistant"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if role == "assistant" and i == len(matches) - 1 and not content:
            continue
        if not content:
            continue

        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = messages[-1]["content"].rstrip() + "\n\n" + content
        else:
            messages.append({"role": role, "content": content})

    if not messages:
        return [{"role": "user", "content": text}], "fallback_empty_parse"
    if messages[0]["role"] != "user":
        return [{"role": "user", "content": text}], "fallback_first_not_user"
    if messages[-1]["role"] != "user":
        return [{"role": "user", "content": text}], "fallback_last_not_user"
    return messages, "parsed"


def get_prompt_len(tokenizer, messages: list[dict[str, str]], chat_template: str | None) -> int:
    kwargs = {}
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs))


def load_tokenizer(args):
    if args.tokenizer_path is None:
        return None, None

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if args.chat_template == "llama3":
        chat_template = LLAMA3_CHAT_TEMPLATE
    elif args.chat_template == "none":
        chat_template = None
    elif tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat_template. Use --chat_template llama3 for Llama-3 base tokenizers, "
            "or omit --tokenizer_path to skip process-time length filtering."
        )
    else:
        chat_template = None
    return tokenizer, chat_template


def empty_stats(split: str) -> dict[str, Any]:
    return {
        "split": split,
        "raw_rows": 0,
        "kept_rows": 0,
        "skipped_empty_prompt": 0,
        "skipped_empty_chosen": 0,
        "skipped_overlong_prompt": 0,
        "chosen_equals_rejected": 0,
        "parse_notes": {},
        "turn_counts": {},
        "prompt_tokens_all": {},
        "prompt_tokens_kept": {},
    }


def add_count(mapping: dict[str, int], key: str) -> None:
    mapping[key] = int(mapping.get(key, 0)) + 1


def add_token_stats(stats: dict[str, Any], key: str, token_lens: list[int]) -> None:
    if not token_lens:
        stats[key] = {}
        return
    arr = np.asarray(token_lens, dtype=np.int64)
    stats[key] = {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def build_rl_split(
    raw_split: datasets.Dataset,
    split_name: str,
    args,
    tokenizer=None,
    chat_template: str | None = None,
) -> tuple[datasets.Dataset, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_lens_all: list[int] = []
    token_lens_kept: list[int] = []
    stats = empty_stats(split_name)

    for idx, example in enumerate(iter_progress(raw_split, f"RL {split_name}")):
        stats["raw_rows"] += 1
        prompt_text = as_text(example.get("prompt", "")).strip()
        chosen = as_text(example.get("chosen", example.get("response", ""))).strip()
        rejected = as_text(example.get("rejected", "")).strip()
        response = as_text(example.get("response", chosen)).strip()

        if not prompt_text:
            stats["skipped_empty_prompt"] += 1
            continue
        if not chosen:
            stats["skipped_empty_chosen"] += 1
            continue

        messages, parse_note = parse_hh_prompt(prompt_text)
        add_count(stats["parse_notes"], parse_note)
        add_count(stats["turn_counts"], str(len(messages)))

        prompt_len = None
        if tokenizer is not None:
            prompt_len = get_prompt_len(tokenizer, messages, chat_template)
            token_lens_all.append(prompt_len)
            if args.filter_overlong_prompts and prompt_len > args.max_prompt_length:
                stats["skipped_overlong_prompt"] += 1
                continue
            token_lens_kept.append(prompt_len)

        chosen_equals_rejected = bool(chosen and rejected and chosen == rejected)
        if chosen_equals_rejected:
            stats["chosen_equals_rejected"] += 1

        extra_info: dict[str, Any] = {
            "split": split_name,
            "index": idx,
            "source": DATA_SOURCE,
            "num_turns": len(messages),
            "parse_note": parse_note,
            "chosen_equals_rejected": chosen_equals_rejected,
        }
        if prompt_len is not None:
            extra_info["prompt_tokens"] = prompt_len
        if args.keep_reference_text:
            extra_info["reference_chosen"] = chosen
            extra_info["reference_rejected"] = rejected
            extra_info["reference_response"] = response

        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": messages,
                "ability": "alignment",
                "reward_model": {"style": "model", "ground_truth": chosen},
                "extra_info": extra_info,
            }
        )

    stats["kept_rows"] = len(rows)
    add_token_stats(stats, "prompt_tokens_all", token_lens_all)
    add_token_stats(stats, "prompt_tokens_kept", token_lens_kept)
    return datasets.Dataset.from_list(rows), stats


def build_sft_split(raw_split: datasets.Dataset, split_name: str, args) -> tuple[datasets.Dataset, dict[str, Any]]:
    rows: list[dict[str, str]] = []
    stats = empty_stats(split_name)

    for idx, example in enumerate(iter_progress(raw_split, f"SFT {split_name}")):
        stats["raw_rows"] += 1
        prompt_text = as_text(example.get("prompt", "")).strip()
        chosen = as_text(example.get("chosen", example.get("response", ""))).strip()
        rejected = as_text(example.get("rejected", "")).strip()
        if not prompt_text:
            stats["skipped_empty_prompt"] += 1
            continue
        if not chosen:
            stats["skipped_empty_chosen"] += 1
            continue
        rows.append({"prompt": prompt_text, "response": chosen, "split": split_name, "index": idx, "label": "chosen"})
        if args.sft_include_rejected and rejected:
            rows.append(
                {"prompt": prompt_text, "response": rejected, "split": split_name, "index": idx, "label": "rejected"}
            )

    stats["kept_rows"] = len(rows)
    return datasets.Dataset.from_list(rows), stats


def build_rm_split(raw_split: datasets.Dataset, split_name: str) -> tuple[datasets.Dataset, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = empty_stats(split_name)

    for idx, example in enumerate(iter_progress(raw_split, f"RM {split_name}")):
        stats["raw_rows"] += 1
        prompt_text = as_text(example.get("prompt", "")).strip()
        chosen = as_text(example.get("chosen", example.get("response", ""))).strip()
        rejected = as_text(example.get("rejected", "")).strip()
        if not prompt_text:
            stats["skipped_empty_prompt"] += 1
            continue
        if not chosen:
            stats["skipped_empty_chosen"] += 1
            continue
        if chosen and rejected and chosen == rejected:
            stats["chosen_equals_rejected"] += 1
        rows.append({"prompt": prompt_text, "chosen": chosen, "rejected": rejected, "split": split_name, "index": idx})

    stats["kept_rows"] = len(rows)
    return datasets.Dataset.from_list(rows), stats


def save_dataset_pair(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    output_dir: Path,
    stats: dict[str, Any],
    write_examples: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if write_examples:
        examples = {}
        if len(train_dataset) > 0:
            examples["train"] = train_dataset[0]
        if len(test_dataset) > 0:
            examples["test"] = test_dataset[0]
        with (output_dir / "examples.json").open("w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(train_dataset)} train rows to {train_path}")
    print(f"Saved {len(test_dataset)} test rows to {test_path}")
    print(f"Saved stats to {output_dir / 'stats.json'}")


def process_rl(raw_dataset: datasets.DatasetDict, args) -> None:
    tokenizer, chat_template = load_tokenizer(args)
    train_raw = maybe_limit_split(raw_dataset["train"], args.num_train, args.seed, args.shuffle_before_limit)
    test_raw = maybe_limit_split(raw_dataset["test"], args.num_test, args.seed, args.shuffle_before_limit)
    train_dataset, train_stats = build_rl_split(train_raw, "train", args, tokenizer, chat_template)
    test_dataset, test_stats = build_rl_split(test_raw, "test", args, tokenizer, chat_template)
    stats = {"mode": "rl", "data_source": DATA_SOURCE, "train": train_stats, "test": test_stats}
    save_dataset_pair(train_dataset, test_dataset, Path(args.local_save_dir), stats)


def process_sft(raw_dataset: datasets.DatasetDict, args) -> None:
    train_raw = maybe_limit_split(raw_dataset["train"], args.num_train, args.seed, args.shuffle_before_limit)
    test_raw = maybe_limit_split(raw_dataset["test"], args.num_test, args.seed, args.shuffle_before_limit)
    train_dataset, train_stats = build_sft_split(train_raw, "train", args)
    test_dataset, test_stats = build_sft_split(test_raw, "test", args)
    stats = {"mode": "sft", "data_source": DATA_SOURCE, "train": train_stats, "test": test_stats}
    save_dataset_pair(train_dataset, test_dataset, Path(args.local_save_dir) / "sft", stats)


def process_rm(raw_dataset: datasets.DatasetDict, args) -> None:
    train_raw = maybe_limit_split(raw_dataset["train"], args.num_train, args.seed, args.shuffle_before_limit)
    test_raw = maybe_limit_split(raw_dataset["test"], args.num_test, args.seed, args.shuffle_before_limit)
    train_dataset, train_stats = build_rm_split(train_raw, "train")
    test_dataset, test_stats = build_rm_split(test_raw, "test")
    stats = {"mode": "rm", "data_source": DATA_SOURCE, "train": train_stats, "test": test_stats}
    save_dataset_pair(train_dataset, test_dataset, Path(args.local_save_dir) / "rm", stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess local Dahoas/full-hh-rlhf for veRL PPO/SFT/RM.")
    parser.add_argument(
        "--raw_dataset_path",
        default=None,
        help="Local cached dataset snapshot dir. Defaults to HF_HOME or ../hf_cache discovery.",
    )
    parser.add_argument(
        "--local_save_dir",
        default=str(PROJECT_ROOT / "data" / "hh_rlhf"),
        help="Output directory. RL mode writes train/test directly here.",
    )
    parser.add_argument("--mode", choices=["rl", "sft", "rm", "all"], default="rl")
    parser.add_argument("--num_train", type=int, default=-1, help="Limit train rows for debugging. -1 means all.")
    parser.add_argument("--num_test", type=int, default=-1, help="Limit test rows for debugging. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_before_limit", action="store_true")
    parser.add_argument(
        "--keep_reference_text",
        action="store_true",
        help="Store chosen/rejected/response in extra_info. Useful for debugging but increases parquet size.",
    )
    parser.add_argument(
        "--sft_include_rejected",
        action="store_true",
        help="For SFT mode, include rejected responses as additional supervised examples.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Optional tokenizer path for process-time max prompt length filtering.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument(
        "--filter_overlong_prompts",
        action="store_true",
        help="Drop prompts longer than --max_prompt_length. Requires --tokenizer_path.",
    )
    parser.add_argument(
        "--chat_template",
        choices=["auto", "llama3", "none"],
        default="auto",
        help="Template used only for optional tokenizer length checks.",
    )
    parser.add_argument(
        "--allow_remote_files",
        dest="local_files_only",
        action="store_false",
        default=True,
        help="Allow tokenizer files to be fetched remotely if they are not present locally.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.filter_overlong_prompts and args.tokenizer_path is None:
        raise ValueError("--filter_overlong_prompts requires --tokenizer_path")

    raw_dataset_path = resolve_raw_dataset_path(args.raw_dataset_path)
    print(f"Loading raw dataset from {raw_dataset_path}")
    raw_dataset = load_raw_dataset(raw_dataset_path)
    print(raw_dataset)

    if args.mode in {"rl", "all"}:
        process_rl(raw_dataset, args)
    if args.mode in {"sft", "all"}:
        process_sft(raw_dataset, args)
    if args.mode in {"rm", "all"}:
        process_rm(raw_dataset, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
