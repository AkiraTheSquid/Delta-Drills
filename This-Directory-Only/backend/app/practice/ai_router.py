"""
AI explanation/judge and arbitrary-code-runner endpoints.

Endpoints (mounted under /api/practice by the parent router):
  POST /run-code
  POST /ai-explanation
  POST /ai-judge
  POST /ai-tutor
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.code_runner import run_code
from app.practice.chatgpt_helpers import call_chatgpt, call_chatgpt_messages
from app.practice.prompts import (
    build_ai_explanation_prompt,
    build_ai_judge_prompt,
    build_ai_tutor_system_prompt,
)
from app.practice_schemas import (
    AIExplanationRequest,
    AIExplanationResponse,
    AIJudgeResponse,
    AITutorRequest,
    AITutorResponse,
    CodeRunRequest,
    CodeRunResponse,
)

router = APIRouter()

# The tutor thread is replayed by the client on every turn, so it is the one
# endpoint here whose payload grows without bound. Trim it: keep the most
# recent turns, and cap any single message. Both limits are generous for a
# real conversation about one drill and cheap insurance against a runaway tab.
TUTOR_MODEL = "gpt-4o"
TUTOR_MAX_TURNS = 24
TUTOR_MAX_CHARS = 8000


@router.post("/run-code", response_model=CodeRunResponse)
def run_code_endpoint(payload: CodeRunRequest) -> CodeRunResponse:
    """Run arbitrary Python code in a sandboxed subprocess (20s timeout)."""
    result = run_code(payload.code, timeout=20)
    return CodeRunResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        success=result.success,
    )


@router.post("/ai-explanation", response_model=AIExplanationResponse)
def ai_explanation(payload: AIExplanationRequest) -> AIExplanationResponse:
    """Generate a teaching explanation. Model: gpt-4o."""
    prompt = build_ai_explanation_prompt(
        question_text=payload.question_text,
        expected_output=payload.expected_output,
        user_code=payload.user_code,
        actual_output=payload.actual_output,
        solution_code=payload.solution_code,
    )
    try:
        explanation = call_chatgpt(prompt, model="gpt-4o")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI explanation error: {e}",
        )
    return AIExplanationResponse(explanation=explanation)


@router.post("/ai-judge", response_model=AIJudgeResponse)
def ai_judge(payload: AIExplanationRequest) -> AIJudgeResponse:
    """Binary correctness judge. Model: gpt-4o-mini."""
    prompt = build_ai_judge_prompt(
        question_text=payload.question_text,
        expected_output=payload.expected_output,
        user_code=payload.user_code,
        actual_output=payload.actual_output,
        solution_code=payload.solution_code,
    )
    try:
        raw = call_chatgpt(prompt, model="gpt-4o-mini").strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI judge error: {e}",
        )
    verdict = "1" if "1" in raw else "0"
    return AIJudgeResponse(verdict=verdict)


@router.post("/ai-tutor", response_model=AITutorResponse)
def ai_tutor(payload: AITutorRequest) -> AITutorResponse:
    """Post-answer tutor chat: one assistant turn for the thread so far."""
    if not payload.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages to answer.",
        )
    if payload.messages[-1].role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The last message must be from the student.",
        )

    system_prompt = build_ai_tutor_system_prompt(
        question_text=payload.question_text,
        expected_output=payload.expected_output,
        user_code=payload.user_code,
        actual_output=payload.actual_output,
        solution_code=payload.solution_code,
        explanation=payload.explanation,
        was_correct=payload.was_correct,
    )
    history = payload.messages[-TUTOR_MAX_TURNS:]
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.role, "content": m.content[:TUTOR_MAX_CHARS]} for m in history
    ]

    try:
        reply = call_chatgpt_messages(messages, model=TUTOR_MODEL)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI tutor error: {e}",
        )
    return AITutorResponse(reply=reply or "No reply available.")
