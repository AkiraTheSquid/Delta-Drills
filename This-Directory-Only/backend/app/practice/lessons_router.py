"""
First-encounter exposure endpoints (lesson gate, Pass 2).

Endpoints (mounted under /api/practice by the parent router):
  GET  /exposure   — the learner's kc -> first-exposure timestamp map
  POST /exposure   — record introducing-KP completion for one or more KCs

Exposure only ever accumulates; there is no unexpose (re-reading a lesson
is always allowed client-side, but the gate never re-arms).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app import lessons
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User
from app.practice_schemas import ExposureMarkRequest, ExposureResponse

router = APIRouter()


@router.get("/exposure", response_model=ExposureResponse)
def get_exposure(user: User = Depends(get_current_user)) -> ExposureResponse:
    user_state = get_user_state(str(user.id))
    return ExposureResponse(exposed=dict(user_state.kc_exposure))


@router.post("/exposure", response_model=ExposureResponse)
def mark_exposure(
    payload: ExposureMarkRequest,
    user: User = Depends(get_current_user),
) -> ExposureResponse:
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    now = datetime.now(timezone.utc).isoformat()
    changed = False
    for kc in payload.kcs:
        # Unknown KC ids are dropped silently — a stale/renamed KC in an old
        # client must not pollute the exposure map forever.
        if kc not in user_state.kc_exposure and lessons.kc_exists(kc):
            user_state.kc_exposure[kc] = now
            changed = True
    if changed:
        save_user_state(user_id)
    return ExposureResponse(exposed=dict(user_state.kc_exposure))
