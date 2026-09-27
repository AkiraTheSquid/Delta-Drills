"""The "what have you done before?" survey — `/api/practice/survey`.

The questions and what an answer means are `app/area_survey.py`'s; this file
reads and stores them, together with the novice / intermediate / expert level
(`self_reported_level`), which the survey asks first while it can still be an
honest cold-start prior (`diagnostic.can_set_prior`).

Mounted by app/main.py, like xp_router: practice/__init__.py is over
Modulario's dependency limit.
"""
from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app import area_survey, bkt_mastery, diagnostic
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/api/practice", tags=["practice"])

# The words the learner picks → the stored level. "intermediate" is the
# no-shift default (None), the same prior the retired `PUT /self-report`
# stored as "default".
LEVELS = {"novice": "beginner", "intermediate": None, "expert": "strong"}


class SurveyRequest(BaseModel):
    answers: Dict[str, str] = {}
    level: Optional[str] = None


def _body(state) -> dict:
    level = getattr(state, "self_reported_level", None)
    return {
        "answered": area_survey.answered(state),
        "answers": dict(state.area_survey or {}),
        # Only the questions still pending: a course enabled later asks its own.
        "questions": [{"id": q["id"], "text": q["text"]} for q in area_survey.pending(state)],
        "choices": list(area_survey.ANSWERS),
        "levels": list(LEVELS),
        "level": next((k for k, v in LEVELS.items() if v == level), "intermediate"),
        "level_open": diagnostic.can_set_prior(state),
    }


@router.get("/survey")
def get_survey(user: User = Depends(get_current_user)) -> dict:
    return _body(get_user_state(str(user.id)))


@router.post("/survey")
def post_survey(payload: SurveyRequest, user: User = Depends(get_current_user)) -> dict:
    """Store the answers (a question left out = skipped, not asked again) and, while
    it is still open, the level. 400 with a sentence on a bad value."""
    user_id = str(user.id)
    state = get_user_state(user_id)
    if payload.level is not None and payload.level not in LEVELS:
        raise HTTPException(status_code=400, detail="Pick novice, intermediate or expert.")
    try:
        area_survey.save(state, payload.answers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.level is not None and diagnostic.can_set_prior(state):
        level = LEVELS[payload.level]
        state.self_reported_level = level if level in bkt_mastery.PRIOR_BY_LEVEL else None
    save_user_state(user_id)
    return _body(state)
