"""
Grading and question-selection helpers for the practice questions router.

Keeps the heavy logic (running user code, comparing outputs, dispatching
to the AI judge) out of the route definitions.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from fastapi import HTTPException, status

from app.code_runner import (
    TORCH_COLAB_MESSAGE,
    ExecutionResult,
    code_uses_torch,
    run_code,
    run_function_tests,
    torch_available,
)
from app.models import User
from app.practice.chatgpt_helpers import call_chatgpt
from app.practice.prompts import build_ai_judge_prompt
from app.questions import Question


def run_and_get_expected_output(answer_code: str) -> str:
    """Run the canonical answer code and return its stdout."""
    result = run_code(answer_code, timeout=20)
    return result.stdout.strip()


def select_question_for_difficulty(
    candidates: List[Question],
    target_difficulty: float,
    served_ids: set[int],
) -> Optional[Question]:
    """Pick a question close to target_difficulty, preferring unseen ones."""
    unseen = [q for q in candidates if q.id not in served_ids]
    pool = unseen if unseen else candidates
    if not pool:
        return None
    ranked = sorted(pool, key=lambda q: abs(q.difficulty_score - target_difficulty))
    # Explore only within a band of the target. The old flat top-3 was uniform
    # random in a 3-question subtopic — a self-reported beginner (target ~22)
    # could draw the 45-difficulty variant on the first exposure.
    band = [q for q in ranked if abs(q.difficulty_score - target_difficulty) <= 15]
    top_n = (band or ranked[:1])[:3]
    return random.choice(top_n)


def grade_submission(
    question: Question,
    user_code: str,
    user: User,
) -> Tuple[bool, str, str, List[dict]]:
    """
    Run user code and decide correctness using the appropriate strategy:
    exact stdout match, function tests, or AI judge.

    Returns: (correct, actual_output, expected_output, failed_tests)
    """
    # Torch drills grade in-process via the fork runner when torch is
    # preloaded (app startup). Refuse with the Colab-routing message only
    # when torch genuinely isn't available in this environment.
    if not torch_available() and (
        getattr(question, "primary_library", None) == "torch"
        or code_uses_torch(user_code)
        or code_uses_torch(getattr(question, "answer_code", "") or "")
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=TORCH_COLAB_MESSAGE,
        )

    user_result: ExecutionResult = run_code(user_code, timeout=20)
    actual_output = user_result.stdout.strip()

    failed_tests: List[dict] = []

    # Function tests FIRST — they are the audited grading contract (per-case
    # value equality, fixture-aware seeding). The old order let the stdout
    # branch hijack 232 function-mode questions into an exact string compare
    # against the stored expected_output — and 85 of those stored strings
    # came from an UNSEEDED CSV-era capture, unreachable by any honest
    # solution under the seeded harness (tester's id-24 argmax drill).
    if question.submission_mode == "function" and question.test_cases:
        expected_output = question.expected_output or ""
        test_results, test_execution = run_function_tests(user_code, question.test_cases)
        correct = all(result.passed for result in test_results)
        if test_execution.stdout.strip():
            actual_output = test_execution.stdout.strip()
        elif test_execution.stderr.strip():
            actual_output = test_execution.stderr.strip()
        failed_tests = [
            {"actual": result.actual, "expected": result.expected, "error": result.error}
            for result in test_results
            if not result.passed
        ]
        return correct, actual_output, expected_output, failed_tests

    if (
        question.task_type == "stdout_prediction"
        and not question.supports_visual_output
    ):
        # Recompute expected stdout by running the canonical answer under the
        # SAME harness (preamble + seed) the user's code just ran under —
        # never trust the stored CSV-era string (see above).
        expected_output = run_and_get_expected_output(question.answer_code)
        if expected_output.strip():
            correct = actual_output.strip() == expected_output.strip()
            return correct, actual_output, expected_output, failed_tests

    expected_output = question.expected_output or run_and_get_expected_output(question.answer_code)

    judge_prompt = build_ai_judge_prompt(
        question_text=question.question_text,
        expected_output=expected_output,
        user_code=user_code,
        actual_output=actual_output,
        solution_code=question.answer_code,
    )
    try:
        raw = call_chatgpt(judge_prompt, model="gpt-4o-mini", user=user).strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI judge error: {e}",
        )
    correct = "1" in raw
    return correct, actual_output, expected_output, failed_tests
