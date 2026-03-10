"""
Practice API router.

Endpoints:
  GET  /api/practice/next-question  — next question via adaptive algorithm
  POST /api/practice/submit         — submit code for grading
  POST /api/practice/feedback       — provide learning feedback
  GET  /api/practice/subtopics      — list subtopics with user stats
  POST /api/practice/run-code       — run arbitrary Python code (sandboxed)
"""

from __future__ import annotations

import logging
import random
from typing import List

from openai import OpenAI

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_db

from app.adaptive import (
    apply_feedback,
    get_target_difficulty,
    get_user_state,
    override_pending_attempt,
    record_attempt,
    save_user_state,
)
from app.auth import get_current_user
from app.code_runner import ExecutionResult, run_code, run_function_tests
from app.models import User
from app.practice_schemas import (
    AIExplanationRequest,
    AIExplanationResponse,
    AIJudgeResponse,
    CodeRunRequest,
    CodeRunResponse,
    FeedbackRequest,
    FeedbackResponse,
    LocalEvalSubmitRequest,
    NextQuestionResponse,
    OverrideAttemptRequest,
    OverrideAttemptResponse,
    SubtopicStatsResponse,
    SubmitRequest,
    SubmitResponse,
    VisualDebugRequest,
    VisualDebugResponse,
    WeightsUpdateRequest,
)
from app.prioritization import select_next_subtopic, get_subtopic_weights
from app.questions import (
    Question,
    get_question_by_id,
    get_questions_by_subtopic,
)

logger = logging.getLogger(__name__)
_latest_visual_debug_by_user: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# ChatGPT module helper
# ---------------------------------------------------------------------------

def _load_chatgpt_api_key(user: "User", db: Session | None = None) -> str:
    """
    Load the OpenAI API key for the given user.
    Checks (in order): users.openai_api_key, user_settings table, OPENAI_API_KEY env var, settings.
    """
    import os
    if user is not None and user.openai_api_key:
        return user.openai_api_key
    if user is not None and db is not None:
        try:
            row = db.execute(
                text("SELECT openai_api_key FROM user_settings WHERE user_email = :email"),
                {"email": user.email},
            ).fetchone()
            if row and row[0]:
                return row[0]
        except Exception:
            pass
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    from app.config import settings
    if settings.openai_api_key:
        return settings.openai_api_key
    raise ValueError("No OpenAI API key available.")


def _call_chatgpt(prompt: str, model: str, user: "User" = None, db: Session | None = None) -> str:
    """
    Call OpenAI using ChatGPT.py's algorithm:
      1. Try the Responses API first.
      2. Fall back to Chat Completions.
    Temperature 1 matches ChatGPT.py's default.
    """
    api_key = _load_chatgpt_api_key(user, db)
    client = OpenAI(api_key=api_key)
    try:
        resp = client.responses.create(model=model, input=prompt, temperature=1)
        answer = getattr(resp, "output_text", None) or ""
        if not answer:
            first_output = getattr(resp, "output", None)
            if isinstance(first_output, list) and first_output:
                first_content = getattr(first_output[0], "content", None)
                if isinstance(first_content, list) and first_content:
                    maybe_text = getattr(first_content[0], "text", None)
                    if isinstance(maybe_text, str):
                        answer = maybe_text
        return answer
    except Exception:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )
        return completion.choices[0].message.content or "" if completion.choices else ""


router = APIRouter(prefix="/api/practice", tags=["practice"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_question_for_difficulty(
    candidates: List[Question],
    target_difficulty: float,
    served_ids: set[int],
) -> Question | None:
    """
    Pick the question whose difficulty_score is closest to target_difficulty,
    preferring questions not yet served. If all have been served, pick from all.
    """
    unseen = [q for q in candidates if q.id not in served_ids]
    pool = unseen if unseen else candidates

    if not pool:
        return None

    # Sort by distance to target, pick the closest (with a bit of randomness
    # among the top-3 closest to add variety)
    ranked = sorted(pool, key=lambda q: abs(q.difficulty_score - target_difficulty))
    top_n = ranked[: min(3, len(ranked))]
    return random.choice(top_n)


def _run_and_get_expected_output(answer_code: str) -> str:
    """Run the canonical answer code and return its stdout."""
    result = run_code(answer_code, timeout=5)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/next-question", response_model=NextQuestionResponse)
def next_question(user: User = Depends(get_current_user)) -> NextQuestionResponse:
    """
    Return the next question for the authenticated user based on the
    adaptive difficulty algorithm and subtopic prioritization.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    # 1. Pick the subtopic
    subtopic = select_next_subtopic(user_state)
    if subtopic is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No questions available",
        )

    # 2. Determine target difficulty for this subtopic
    sub_state = user_state.get_subtopic_state(subtopic)
    target_diff = get_target_difficulty(sub_state)

    # 3. Select a question close to that difficulty
    candidates = get_questions_by_subtopic(subtopic)
    served = set(sub_state.served_question_ids)
    question = _select_question_for_difficulty(candidates, target_diff, served)

    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No questions available for subtopic '{subtopic}'",
        )

    # Mark as served
    sub_state.served_question_ids.append(question.id)
    save_user_state(user_id)

    # 4. Pre-compute expected output
    expected_output = (
        question.expected_output
        if question.supports_visual_output
        else (question.expected_output or _run_and_get_expected_output(question.answer_code))
    )

    from app.adaptive import COLD_START_TARGETS
    return NextQuestionResponse(
        question_id=question.id,
        question_text=question.question_text,
        topic=question.topic,
        subtopic=question.subtopic,
        difficulty=question.difficulty_score,
        target_difficulty=target_diff,
        expected_output=expected_output,
        solution_code=question.answer_code,
        is_cold_start=sub_state.n < len(COLD_START_TARGETS),
        subtopic_n=sub_state.n,
        p_current=sub_state.p if sub_state.n > 0 else None,
        primary_library=question.primary_library,
        task_type=question.task_type,
        expected_artifact_type=question.expected_artifact_type,
        supports_visual_output=question.supports_visual_output,
        function_name=question.function_name,
        starter_code=question.starter_code,
        test_cases=question.test_cases,
        submission_mode=question.submission_mode,
    )


@router.post("/submit", response_model=SubmitResponse)
def submit_answer(
    payload: SubmitRequest,
    user: User = Depends(get_current_user),
) -> SubmitResponse:
    """
    Execute the user's code, judge it with AI,
    and record the attempt (pending feedback).
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    question = get_question_by_id(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    # Run user code
    user_result: ExecutionResult = run_code(payload.user_code, timeout=5)
    actual_output = user_result.stdout.strip()

    # Use pre-computed expected output from CSV, fall back to running code
    expected_output = question.expected_output or _run_and_get_expected_output(question.answer_code)

    failed_tests: list[dict] = []
    if (
        question.task_type == "stdout_prediction"
        and expected_output.strip()
        and not question.supports_visual_output
    ):
        correct = actual_output.strip() == expected_output.strip()
    elif question.submission_mode == "function" and question.test_cases:
        test_results, test_execution = run_function_tests(payload.user_code, question.test_cases)
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
    else:
        # AI judge always determines correctness (no output comparison)
        judge_prompt = (
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
            f"QUESTION:\n{question.question_text}\n\n"
            f"EXPECTED OUTPUT:\n{expected_output}\n\n"
            f"STUDENT'S CODE:\n{payload.user_code}\n\n"
            f"STUDENT'S ACTUAL OUTPUT:\n{actual_output}\n\n"
            f"CANONICAL SOLUTION (reference only):\n{question.answer_code}\n"
            "---\n\n"
            "Does the student's code correctly implement the concept? Reply 1 or 0 only."
        )
        try:
            raw = _call_chatgpt(judge_prompt, model="gpt-4o-mini", user=user).strip()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AI judge error: {e}",
            )
        correct = "1" in raw

    # Record the attempt (will be finalized when feedback is provided)
    record_attempt(
        user_state=user_state,
        question_id=question.id,
        subtopic=question.subtopic,
        difficulty_score=question.difficulty_score,
        correct=correct,
    )
    save_user_state(user_id)

    return SubmitResponse(
        correct=correct,
        actual_output=actual_output,
        expected_output=expected_output,
        solution_code=question.answer_code,
        failed_tests=failed_tests,
    )


@router.post("/submit-local-eval", response_model=OverrideAttemptResponse)
def submit_local_eval(
    payload: LocalEvalSubmitRequest,
    user: User = Depends(get_current_user),
) -> OverrideAttemptResponse:
    """
    Record an attempt that was graded locally in the frontend (e.g. Pyodide)
    so backend feedback can still apply to the pending attempt.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    question = get_question_by_id(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    record_attempt(
        user_state=user_state,
        question_id=question.id,
        subtopic=question.subtopic,
        difficulty_score=question.difficulty_score,
        correct=payload.correct,
    )
    save_user_state(user_id)
    return OverrideAttemptResponse(success=True)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload: FeedbackRequest,
    user: User = Depends(get_current_user),
) -> FeedbackResponse:
    """
    User provides feedback after seeing the result of their submission.
    This updates the adaptive difficulty state.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    if user_state.pending_attempt is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending attempt to provide feedback on. Submit an answer first.",
        )

    if user_state.pending_attempt.question_id != payload.question_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback question_id does not match the pending attempt.",
        )

    attempt = apply_feedback(user_state, payload.feedback)
    if attempt is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to apply feedback.",
        )
    save_user_state(user_id)

    return FeedbackResponse(
        success=True,
        target_difficulty_after=attempt.target_difficulty_after or 0.0,
        p_after=attempt.p_after or 0.0,
    )


@router.post("/visual-debug", response_model=VisualDebugResponse)
def submit_visual_debug(
    payload: VisualDebugRequest,
    user: User = Depends(get_current_user),
) -> VisualDebugResponse:
    user_id = str(user.id)
    latest = dict(payload.payload or {})
    _latest_visual_debug_by_user[user_id] = latest
    logger.info("visual_debug user=%s payload=%s", user_id, latest)
    return VisualDebugResponse(success=True, latest=latest)


@router.get("/visual-debug", response_model=VisualDebugResponse)
def get_visual_debug(
    user: User = Depends(get_current_user),
) -> VisualDebugResponse:
    user_id = str(user.id)
    latest = _latest_visual_debug_by_user.get(user_id, {})
    return VisualDebugResponse(success=True, latest=latest)


@router.post("/override", response_model=OverrideAttemptResponse)
def override_attempt(
    payload: OverrideAttemptRequest,
    user: User = Depends(get_current_user),
) -> OverrideAttemptResponse:
    """
    Override the pending attempt correctness before feedback is applied.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    updated = override_pending_attempt(
        user_state=user_state,
        question_id=payload.question_id,
        correct=payload.correct,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No matching pending attempt to override.",
        )
    save_user_state(user_id)
    return OverrideAttemptResponse(success=True)


@router.get("/subtopics", response_model=list[SubtopicStatsResponse])
def list_subtopics(user: User = Depends(get_current_user)) -> list[SubtopicStatsResponse]:
    """
    List all subtopics with the current user's adaptive stats including
    weight, learning rate, gradient (delta), baseline score, and
    correctness rate.  Results are sorted by gradient descending so the
    highest-priority subtopic comes first.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    weights_info = get_subtopic_weights(user_state)

    return [
        SubtopicStatsResponse(
            subtopic=info["subtopic"],
            topic=info["topic"],
            questions_answered=info["questions_answered"],
            current_difficulty=info["current_difficulty"],
            weight=info["weight"],
            learning_rate=info["learning_rate"],
            gradient=info["gradient"],
            baseline=info["baseline"],
            p=info["p"],
        )
        for info in weights_info
    ]


@router.put("/weights")
def update_weights(
    payload: WeightsUpdateRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """
    Persist custom per-subtopic effective weights for the authenticated user.
    These override the uniform defaults used by the subtopic selection algorithm.
    Weights are raw floats (e.g. 0.175); relative magnitudes are what matter.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    user_state.custom_weights = {k: float(v) for k, v in payload.weights.items()}
    save_user_state(user_id)
    return {"ok": True}


@router.post("/run-code", response_model=CodeRunResponse)
def run_code_endpoint(
    payload: CodeRunRequest,
    user: User = Depends(get_current_user),
) -> CodeRunResponse:
    """
    Run arbitrary Python code in a sandboxed subprocess.
    numpy is available. Timeout is 5 seconds.
    """
    result = run_code(payload.code, timeout=5)
    return CodeRunResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        success=result.success,
    )


@router.post("/ai-explanation", response_model=AIExplanationResponse)
def ai_explanation(
    payload: AIExplanationRequest,
) -> AIExplanationResponse:
    """
    Generate an AI explanation using the ChatGPT module algorithm.
    Model: gpt-4o (best quality for detailed explanations).
    Runs independently of ai-judge — no shared locks, truly parallel.
    """
    prompt = (
        "You are an expert NumPy and Python instructor. A student has just attempted a coding problem.\n"
        "Evaluate their approach and explain the solution clearly.\n\n"
        "IMPORTANT: Do not judge correctness based on exact code match. What matters is whether the "
        "student demonstrates the right core concept and understanding. A solution that uses different "
        "but equivalent NumPy operations, or a slightly different approach that achieves the same result, "
        "shows genuine understanding and should be recognized as such.\n\n"
        "---\n"
        f"QUESTION:\n{payload.question_text}\n\n"
        f"EXPECTED OUTPUT:\n{payload.expected_output}\n\n"
        f"STUDENT'S CODE:\n{payload.user_code}\n\n"
        f"STUDENT'S OUTPUT:\n{payload.actual_output}\n\n"
        f"CANONICAL SOLUTION:\n{payload.solution_code}\n"
        "---\n\n"
        "Please provide:\n"
        "1. The core NumPy concept being tested\n"
        "2. A step-by-step explanation of what the canonical solution does\n"
        "3. An assessment of whether the student's approach captures the right idea "
        "(focus on understanding, not syntax)\n"
        "4. Any tips or insights worth noting about this type of problem"
    )
    try:
        explanation = _call_chatgpt(prompt, model="gpt-4o")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI explanation error: {e}",
        )
    return AIExplanationResponse(explanation=explanation)


@router.post("/ai-judge", response_model=AIJudgeResponse)
def ai_judge(
    payload: AIExplanationRequest,
) -> AIJudgeResponse:
    """
    Binary correctness judge. Outputs '1' if the student's solution demonstrates
    the right core concept, '0' if it does not.
    Model: gpt-4o-mini (fast — should resolve before the explanation).
    Uses AI judgment instead of output comparison.
    No auth required — uses server-level OPENAI_API_KEY.
    """
    prompt = (
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
        f"QUESTION:\n{payload.question_text}\n\n"
        f"EXPECTED OUTPUT:\n{payload.expected_output}\n\n"
        f"STUDENT'S CODE:\n{payload.user_code}\n\n"
        f"STUDENT'S ACTUAL OUTPUT:\n{payload.actual_output}\n\n"
        f"CANONICAL SOLUTION (reference only):\n{payload.solution_code}\n"
        "---\n\n"
        "Does the student's code correctly implement the concept? Reply 1 or 0 only."
    )
    try:
        raw = _call_chatgpt(prompt, model="gpt-4o-mini").strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI judge error: {e}",
        )
    # Be resilient: accept any response containing "1" as correct
    verdict = "1" if "1" in raw else "0"
    return AIJudgeResponse(verdict=verdict)
