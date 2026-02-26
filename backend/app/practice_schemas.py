"""
Pydantic schemas for the practice / adaptive-learning endpoints.
"""

from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class NextQuestionResponse(BaseModel):
    question_id: int
    question_text: str
    subtopic: str
    difficulty: float
    target_difficulty: float
    expected_output: str
    solution_code: str


class SubmitRequest(BaseModel):
    question_id: int
    user_code: str


class SubmitResponse(BaseModel):
    correct: bool
    actual_output: str
    expected_output: str
    solution_code: str


class FeedbackRequest(BaseModel):
    question_id: int
    feedback: Literal["not_much", "somewhat", "a_lot"]


class FeedbackResponse(BaseModel):
    success: bool
    target_difficulty_after: float
    p_after: float = 0.0   # EWMA correctness rate (0–1) after this attempt


class OverrideAttemptRequest(BaseModel):
    question_id: int
    correct: bool = Field(default=True)


class OverrideAttemptResponse(BaseModel):
    success: bool


class SubtopicStatsResponse(BaseModel):
    subtopic: str          # full key e.g. "Numpy: Core array literacy"
    topic: str             # e.g. "Numpy" or "Einsum"
    questions_answered: int
    current_difficulty: float
    weight: float          # uniform prioritization weight (1 / total_subtopics)
    learning_rate: float   # EWMA estimate of recent performance change
    gradient: float        # delta = weight × learning_rate
    baseline: float        # running weighted-average score (0–100 scale)
    p: float               # running correctness rate (0–1)


class WeightsUpdateRequest(BaseModel):
    weights: Dict[str, float]   # { "Numpy: Core array literacy": 0.175, ... }


class CodeRunRequest(BaseModel):
    code: str


class CodeRunResponse(BaseModel):
    stdout: str
    stderr: str
    success: bool


class AIExplanationRequest(BaseModel):
    question_text: str
    solution_code: str
    user_code: str
    actual_output: str
    expected_output: str


class AIExplanationResponse(BaseModel):
    explanation: str


class AIJudgeResponse(BaseModel):
    verdict: str  # "0" = incorrect, "1" = correct
