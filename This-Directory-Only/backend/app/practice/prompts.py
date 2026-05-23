"""
Prompt builders shared between practice routers.

Keeping these here prevents the AI-judge and AI-explanation prompts
from drifting apart between the inline submit-time judge and the
dedicated /ai-judge endpoint.
"""

from __future__ import annotations


def build_ai_judge_prompt(
    *,
    question_text: str,
    expected_output: str,
    user_code: str,
    actual_output: str,
    solution_code: str,
) -> str:
    """Strict-but-fair binary judge: outputs '1' or '0'."""
    return (
        "You are a strict but fair NumPy instructor checking conceptual understanding.\n\n"
        "CRITICAL: The student's output will differ from the canonical solution's output because "
        "they use different test data or print different things. Do NOT compare outputs. "
        "Judge ONLY whether the student's CODE correctly implements the algorithm or formula "
        "stated in the question.\n\n"
        "However, use the student's actual output to catch clear failures: if it shows an "
        "error, a traceback, or is completely empty/blank when output was expected, return 0. "
        "If the expected output is empty, do NOT penalize empty student output.\n\n"
        "Output ONLY the single digit 1 (correct) or 0 (incorrect). No other text.\n\n"
        "---\n"
        f"QUESTION:\n{question_text}\n\n"
        f"EXPECTED OUTPUT:\n{expected_output}\n\n"
        f"STUDENT'S CODE:\n{user_code}\n\n"
        f"STUDENT'S ACTUAL OUTPUT:\n{actual_output}\n\n"
        f"CANONICAL SOLUTION (reference only):\n{solution_code}\n"
        "---\n\n"
        "Does the student's code correctly implement the concept? Reply 1 or 0 only."
    )


def build_ai_explanation_prompt(
    *,
    question_text: str,
    expected_output: str,
    user_code: str,
    actual_output: str,
    solution_code: str,
) -> str:
    """Detailed teaching explanation; favors concept over exact code match."""
    return (
        "You are an expert NumPy and Python instructor. A student has just attempted a coding problem.\n"
        "Evaluate their approach and explain the solution clearly.\n\n"
        "IMPORTANT: Do not judge correctness based on exact code match. What matters is whether the "
        "student demonstrates the right core concept and understanding. A solution that uses different "
        "but equivalent NumPy operations, or a slightly different approach that achieves the same result, "
        "shows genuine understanding and should be recognized as such.\n\n"
        "---\n"
        f"QUESTION:\n{question_text}\n\n"
        f"EXPECTED OUTPUT:\n{expected_output}\n\n"
        f"STUDENT'S CODE:\n{user_code}\n\n"
        f"STUDENT'S OUTPUT:\n{actual_output}\n\n"
        f"CANONICAL SOLUTION:\n{solution_code}\n"
        "---\n\n"
        "Please provide:\n"
        "1. The core NumPy concept being tested\n"
        "2. A step-by-step explanation of what the canonical solution does\n"
        "3. An assessment of whether the student's approach captures the right idea "
        "(focus on understanding, not syntax)\n"
        "4. Any tips or insights worth noting about this type of problem"
    )
