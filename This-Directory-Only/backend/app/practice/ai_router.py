"""
AI explanation/judge and arbitrary-code-runner endpoints.

Endpoints (mounted under /api/practice by the parent router):
  POST /run-code
  POST /ai-explanation
  POST /ai-judge
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.code_runner import run_code
from app.practice.chatgpt_helpers import call_chatgpt
from app.practice.prompts import build_ai_explanation_prompt, build_ai_judge_prompt
from app.practice_schemas import (
    AIExplanationRequest,
    AIExplanationResponse,
    AIJudgeResponse,
    CodeRunRequest,
    CodeRunResponse,
)

router = APIRouter()


@router.post("/run-code", response_model=CodeRunResponse)
def run_code_endpoint(payload: CodeRunRequest) -> CodeRunResponse:
    """Run arbitrary Python code in a sandboxed subprocess (5s timeout)."""
    result = run_code(payload.code, timeout=5)
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
