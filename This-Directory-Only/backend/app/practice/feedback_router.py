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
from app.auth import get_current_user
from app.models import User
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
