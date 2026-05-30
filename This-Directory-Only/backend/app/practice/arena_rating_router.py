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
from app import bkt_mastery
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
    # Atom ids this exercise practices (drill compositeAtomIds / tagged ARENA
    # atoms). When present, each runs a per-atom BKT update with FIRe credit to
    # the atoms it encompasses — the real per-atom mastery signal. The subtopic
    # EWMA bump above is retained only for the legacy learner-facing area score.
    atom_ids: List[str] = Field(default_factory=list)


class ArenaRatingUpdate(BaseModel):
    subtopic: str
    p_after: float
    baseline_after: float
    target_difficulty_after: float


class ArenaRatingResponse(BaseModel):
    success: bool
    updated: List[ArenaRatingUpdate]
    # atom_id -> new BKT posterior, for every atom changed (directly practiced
    # or FIRe-credited). Lets the frontend sync readiness without a re-fetch.
    atom_mastery: dict = Field(default_factory=dict)


@router.post("/arena-rating", response_model=ArenaRatingResponse)
def submit_arena_rating(
    payload: ArenaRatingRequest,
    user: User = Depends(get_current_user),
) -> ArenaRatingResponse:
    """Per-atom BKT update for a completed ARENA exercise / drill.

    EWMA is fully removed: a rating updates ONLY the per-atom BKT posteriors of
    the exercise's `atom_ids` (plus encompassing FIRe credit). `subtopics` is
    retained in the request for back-compat / logging but no longer drives any
    state — subtopic-level scores are now derived from BKT (see
    prioritization.get_subtopic_weights). An ARENA exercise that sends no
    atom_ids therefore updates nothing here; mastery is built by the bank +
    drills that DO carry atoms.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    updated: List[ArenaRatingUpdate] = []

    # Per-atom BKT update + encompassing FIRe credit — the only mastery signal.
    # One shared timestamp so every atom this exercise touches decays together.
    atom_changes: dict = {}
    now = datetime.now(timezone.utc)
    for atom_id in dict.fromkeys(payload.atom_ids):  # de-dupe, preserve order
        changed = bkt_mastery.apply_attempt(
            user_state.atom_mastery,
            user_state.atom_last_ts,
            atom_id,
            payload.correct,
            now=now,
        )
        atom_changes.update(changed)

    save_user_state(user_id)

    logger.info(
        "arena_rating user=%s exercise=%r feedback=%s elapsed=%s target=%s subtopics=%d atoms=%d→%d_changed",
        user_id,
        payload.exercise_title,
        fb,
        payload.elapsed_seconds,
        payload.target_seconds,
        len(updated),
        len(payload.atom_ids),
        len(atom_changes),
    )

    return ArenaRatingResponse(success=True, updated=updated, atom_mastery=atom_changes)
