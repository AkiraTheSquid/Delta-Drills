"""XP = measured learning — `/api/practice/xp` and `/api/practice/xp-target`.

The numbers are `app/learning_xp.py`'s; this file only reads the learner's
time zone and stores their target. The Learner Home XP panel and the topbar
level pill both read GET /xp; the Learner Home's group view reads
GET /groups/xp, every member's summary (app/group_xp.py). Its concept list
reads GET /concept-candidates, and a concept picked from it runs through
POST /concept-route (app/concept_choice.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import concept_choice, group_xp, learning_xp
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.db import get_db
from app.models import User
from app.study_group_progress import zone_for
from app.study_groups import GroupError

# Mounted by app/main.py, not practice/__init__.py: that aggregator is over
# Modulario's dependency limit. Same `/api/practice` prefix as the rest.
router = APIRouter(prefix="/api/practice", tags=["practice"])


class XpTargetRequest(BaseModel):
    mode: str
    date: Optional[str] = None
    daily: Optional[int] = None


def _zone(tz_offset: int, tz_name: Optional[str]):
    try:
        return zone_for(tz_offset, tz_name)
    except GroupError:
        # An unrecognised zone name falls back to the offset, which every
        # browser can send; the XP panel is not worth a 400 over a zone name.
        return zone_for(tz_offset, None)


@router.get("/xp")
def xp(
    tz_offset: int = Query(0, ge=-960, le=960),
    tz_name: Optional[str] = Query(None, max_length=64),
    user: User = Depends(get_current_user),
) -> dict:
    return learning_xp.summary(get_user_state(str(user.id)), _zone(tz_offset, tz_name))


@router.post("/xp-target")
def set_xp_target(
    payload: XpTargetRequest,
    tz_offset: int = Query(0, ge=-960, le=960),
    tz_name: Optional[str] = Query(None, max_length=64),
    user: User = Depends(get_current_user),
) -> dict:
    zone = _zone(tz_offset, tz_name)
    state = get_user_state(str(user.id))
    try:
        learning_xp.set_target(state, payload.mode, payload.date, payload.daily,
                               today=datetime.now(timezone.utc).astimezone(zone).date())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_user_state(str(user.id))
    return learning_xp.summary(state, zone)


@router.get("/groups/xp")
def groups_xp(
    tz_offset: int = Query(0, ge=-960, le=960),
    tz_name: Optional[str] = Query(None, max_length=64),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Every member's `/xp` summary, for the Learner Home's group view
    (app/group_xp.py). `{group: null}` when the caller is in no group."""
    return group_xp.read_group_xp(db, user, _zone(tz_offset, tz_name))


class ConceptRouteRequest(BaseModel):
    kc: str
    served: List[int] = Field(default_factory=list)
    skip: List[int] = Field(default_factory=list)


@router.get("/concept-candidates")
def concept_candidates(keep: Optional[str] = None, user: User = Depends(get_current_user)) -> dict:
    """The Learner Home's concept list: the AI's pick, then the frontier.
    `keep` = the concept of a paused chosen block, listed whatever else."""
    return concept_choice.candidates(get_user_state(str(user.id)), keep=keep)


@router.post("/concept-route")
def concept_route(body: ConceptRouteRequest, user: User = Depends(get_current_user)) -> dict:
    """The next question on a concept the learner chose, or done — the
    practice/ready-route.js Route reads it like /ready-route. Read-only."""
    return concept_choice.plan(get_user_state(str(user.id)), body.kc, body.served, body.skip)
