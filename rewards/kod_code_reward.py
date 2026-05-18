"""Sandbox Fusion reward for KodCode code GRPO training."""

from __future__ import annotations

import os
import re
from typing import Any


CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_BLOCK_RE = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)


def extract_python_code(solution_str: str) -> str:
    """Extract the last Python fenced block, or fall back to the raw response."""
    text = "" if solution_str is None else str(solution_str)

    python_blocks = CODE_BLOCK_RE.findall(text)
    if python_blocks:
        return python_blocks[-1].strip()

    generic_blocks = GENERIC_BLOCK_RE.findall(text)
    if generic_blocks:
        block = generic_blocks[-1].strip()
        first_line, sep, rest = block.partition("\n")
        if sep and first_line.strip().lower() in {"python", "py"}:
            return rest.strip()
        return block

    return text.strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    sandbox_fusion_url: str | None = None,
    timeout: int = 10,
    memory_limit_mb: int = 1024,
    continuous: bool = False,
    **kwargs: Any,
) -> float:
    """Score generated Python code by executing converted tests in Sandbox Fusion."""
    del data_source, extra_info, kwargs

    sandbox_url = sandbox_fusion_url or os.environ.get("SANDBOX_FUSION_URL")
    if not sandbox_url:
        raise RuntimeError("SANDBOX_FUSION_URL or custom_reward_function.reward_kwargs.sandbox_fusion_url is required")

    code = extract_python_code(solution_str)
    fenced_completion = f"```python\n{code}\n```"

    try:
        from verl.utils.reward_score import sandbox_fusion

        score, _metadata = sandbox_fusion.compute_score(
            sandbox_fusion_url=sandbox_url,
            concurrent_semaphore=None,
            memory_limit_mb=memory_limit_mb,
            completion=fenced_completion,
            test_cases=ground_truth,
            continuous=continuous,
            timeout=timeout,
        )
        return float(score)
    except Exception as exc:
        print(f"kod_code_reward error: {exc}")
        return 0.0
