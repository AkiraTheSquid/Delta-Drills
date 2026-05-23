"""
ARENA rating endpoint.

When a student completes an ARENA exercise from inside the unlock interstitial,
the frontend posts the elapsed-vs-target rating here. We bump the EWMA state
for every prereq subtopic of that exercise so the unlock loop reflects the
fact they actually practiced.

Endpoint (mounted under /api/practice by the parent router):
  POST /arena-rating

Request:
  {
    "exercise_title": "(1) Column-stacking",   # for logging only
    "subtopics": ["Einops: Rearrange", "Numpy: Core array literacy", ...],
    "feedback": "not_much" | "somewhat" | "a_lot",
    "correct": true,
    "elapsed_seconds": 142,                    # for logging only
    "target_seconds": 180                      # for logging only
  }

Response:
  { success: true, updated: [{subtopic, p_after, baseline_after, target_after}] }

This is a TEMP scaffold to support the ARENA-prereq pipeline before the
real concept graph + answer-grading flow lands. It synthesizes a pending
attempt per subtopic (question_id = -1, the difficulty currently targeted
for that subtopic, grade = 100 if correct) and runs the existing
apply_feedback() — so the EWMA / baseline math stays in lockstep with the
normal Delta Drills feedback path. When the real flow ships, delete this
file and the include_router line in __init__.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.adaptive import (
    AttemptRecord,
    FeedbackLevel,
    apply_feedback,
    get_user_state,
    save_user_state,
)
from app.auth import get_current_user
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter()


class ArenaRatingRequest(BaseModel):
    exercise_title: str = Field(default="", description="For logging / debugging only.")
    subtopics: List[str] = Field(..., min_length=1)
    feedback: Literal["not_much", "somewhat", "a_lot"]
    correct: bool = True
    elapsed_seconds: Optional[float] = None
    target_seconds: Optional[float] = None


class ArenaRatingUpdate(BaseModel):
    subtopic: str
    p_after: float
    baseline_after: float
    target_difficulty_after: float


class ArenaRatingResponse(BaseModel):
    success: bool
    updated: List[ArenaRatingUpdate]


@router.post("/arena-rating", response_model=ArenaRatingResponse)
def submit_arena_rating(
    payload: ArenaRatingRequest,
    user: User = Depends(get_current_user),
) -> ArenaRatingResponse:
    """Bump EWMA state for each prereq subtopic of a completed ARENA exercise."""
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    # Preserve any in-progress Delta Drills attempt — we'll restore it after.
    saved_pending = user_state.pending_attempt

    updated: List[ArenaRatingUpdate] = []
    grade = 100.0 if payload.correct else 0.0
    timestamp = datetime.now(timezone.utc).isoformat()
    fb: FeedbackLevel = payload.feedback

    for subtopic in payload.subtopics:
        sub_state = user_state.get_subtopic_state(subtopic)
        # Use the subtopic's current target difficulty (assume the ARENA
        # exercise lives roughly at the student's current level). Fall back
        # to 50 if cold-start hasn't seeded a target yet.
        difficulty = int(sub_state.target_difficulty) if sub_state.target_difficulty else 50

        user_state.pending_attempt = AttemptRecord(
            question_id=-1,
            subtopic=subtopic,
            difficulty_score=difficulty,
            grade=grade,
            correct=payload.correct,
            timestamp=timestamp,
        )
        attempt = apply_feedback(user_state, fb)
        if attempt is None:
            continue
        updated.append(
            ArenaRatingUpdate(
                subtopic=subtopic,
                p_after=sub_state.p,
                baseline_after=sub_state.baseline,
                target_difficulty_after=sub_state.target_difficulty,
            )
        )

    # Restore the original Delta Drills pending attempt (the ARENA rating
    # shouldn't clobber a half-graded question the student left open).
    user_state.pending_attempt = saved_pending
    save_user_state(user_id)

    logger.info(
        "arena_rating user=%s exercise=%r feedback=%s elapsed=%s target=%s subtopics=%d",
        user_id,
        payload.exercise_title,
        fb,
        payload.elapsed_seconds,
        payload.target_seconds,
        len(updated),
    )

    return ArenaRatingResponse(success=True, updated=updated)
