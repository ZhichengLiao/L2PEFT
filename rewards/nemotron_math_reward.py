"""Custom prime_math reward function for math GRPO training."""

from __future__ import annotations


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Score a generated math response against a short-answer ground truth.

    veRL calls this function with keyword arguments from the reward manager.
    The project already vendors prime_math, which handles boxed answer
    extraction, answer-marker fallback, symbolic equivalence, and numeric
    tolerance. This wrapper converts its tuple return into a dict that veRL can
    log as reward_extra_info.
    """
    del data_source, extra_info, kwargs

    try:
        from verl.utils.reward_score import prime_math

        result = prime_math.compute_score(str(solution_str), str(ground_truth))
        if isinstance(result, tuple):
            is_correct, format_correct, extracted_answer = result
        else:
            is_correct = bool(result)
            format_correct = False
            extracted_answer = ""

        return {
            "score": float(bool(is_correct)),
            "format_correct": bool(format_correct),
            "extracted_answer": "" if extracted_answer is None else str(extracted_answer),
        }
    except Exception as exc:
        print(f"prime_math_reward error: {exc}")
        return {"score": 0.0, "format_correct": False, "extracted_answer": ""}
