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


def build_ai_tutor_system_prompt(
    *,
    question_text: str,
    expected_output: str,
    user_code: str,
    actual_output: str,
    solution_code: str,
    explanation: str = "",
    was_correct: bool | None = None,
) -> str:
    """System message for the post-answer tutor chat.

    Unlike the judge and the explanation, this one is conversational: the
    learner has already seen the verdict, the canonical solution, and the
    explanation, so the tutor's job is the follow-up questions, not a
    second lecture. The full problem context is baked into the system
    message so the client never has to resend it as fake chat turns.
    """
    if was_correct is True:
        verdict_line = "The student's answer was graded CORRECT."
    elif was_correct is False:
        verdict_line = "The student's answer was graded INCORRECT."
    else:
        verdict_line = "The grade for this attempt is not known."

    prior = (
        f"\nEXPLANATION ALREADY SHOWN TO THE STUDENT:\n{explanation}\n"
        if explanation.strip()
        else ""
    )

    return (
        "You are a patient, expert Python tutor for a spaced-practice drilling app. "
        "The student has just finished a coding drill and can now ask you anything about it.\n\n"
        "The drills are PyTorch-first (some older ones are NumPy); answer in whichever "
        "library the question and the student's code actually use, and do not push them "
        "toward a different one unless they ask.\n\n"
        "HOW TO TUTOR:\n"
        "- Answer the question they actually asked, first and directly. Do not open with a "
        "recap of the problem they just solved.\n"
        "- Be concise. A couple of short paragraphs, or a small code block, is usually right. "
        "Expand only when they ask for depth.\n"
        "- Where it helps, check understanding with ONE short question at the end — never a quiz.\n"
        "- Prefer showing the smallest runnable snippet over describing code in prose.\n"
        "- If they are chasing a misconception, name it plainly and correct it.\n"
        "- The canonical solution below is reference material for you. Quote or adapt it "
        "freely — the student has already seen it.\n"
        "- If they ask about something unrelated to Python, tensors, or this drill, answer "
        "briefly and steer back.\n"
        "- Use Markdown: fenced code blocks for code, backticks for inline identifiers.\n\n"
        "--- PROBLEM CONTEXT ---\n"
        f"{verdict_line}\n\n"
        f"QUESTION:\n{question_text}\n\n"
        f"EXPECTED OUTPUT:\n{expected_output}\n\n"
        f"STUDENT'S CODE:\n{user_code}\n\n"
        f"STUDENT'S OUTPUT:\n{actual_output}\n\n"
        f"CANONICAL SOLUTION:\n{solution_code}\n"
        f"{prior}"
        "--- END PROBLEM CONTEXT ---"
    )
