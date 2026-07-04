"""
Feedback and visual-debug endpoints.

Endpoints (mounted under /api/practice by the parent router):
  POST /feedback
  POST /visual-debug
  GET  /visual-debug
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.adaptive import apply_feedback, get_user_state, save_user_state
from app import bkt_mastery
from app.auth import get_current_user
from app.models import User
from app.prioritization import subtopic_mastery, target_difficulty
from app.questions import get_question_by_id
from app.practice_schemas import (
    FeedbackRequest,
    FeedbackResponse,
    VisualDebugRequest,
    VisualDebugResponse,
)

logger = logging.getLogger(__name__)
_latest_visual_debug_by_user: dict[str, dict] = {}

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload: FeedbackRequest,
    user: User = Depends(get_current_user),
) -> FeedbackResponse:
    """Apply user feedback to the pending attempt and update adaptive state."""
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

    # Per-atom BKT update — the real mastery signal. Fires here (not at /submit)
    # because correctness is only final after any /override, and apply_feedback
    # finalizes the attempt. Each of the question's atom tags updates its BKT
    # posterior scaled by the tag's confidence, then FIRe-credits encompassed
    # atoms. (apply_feedback's EWMA math is now inert bookkeeping — nothing
    # reads baseline/p for any decision; see bkt_mastery.py.)
    question = get_question_by_id(attempt.question_id)
    if question is not None:
        # params carry the learner's self-reported prior so a never-practiced
        # atom's FIRST update starts from that prior (and decay regresses
        # toward it) — one wrong answer still drops a "strong" prior fast.
        user_params = bkt_mastery.params_for_level(user_state.self_reported_level)
        for tag in getattr(question, "atom_tags", []) or []:
            bkt_mastery.apply_attempt(
                user_state.atom_mastery,
                user_state.atom_last_ts,
                tag["atom_id"],
                attempt.correct,
                params=user_params,
                confidence=float(tag.get("confidence", 1.0)),
            )

    # Snapshot the subtopic's BKT mastery into the legacy baseline/p fields the
    # Statistics panel reads (frontend unchanged): 0-1 mastery → 0-100 baseline.
    # Done AFTER the BKT update so the recorded attempt reflects post-attempt
    # mastery; per-attempt baseline_after thus accrues the BKT trajectory the
    # learning-rate chart plots.
    mastery = subtopic_mastery(user_state, attempt.subtopic)
    sub_state = user_state.get_subtopic_state(attempt.subtopic)
    sub_state.baseline = mastery * 100.0
    sub_state.p = mastery
    sub_state.target_difficulty = target_difficulty(user_state, attempt.subtopic)
    attempt.baseline_after = sub_state.baseline
    attempt.p_after = sub_state.p
    attempt.target_difficulty_after = sub_state.target_difficulty

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
