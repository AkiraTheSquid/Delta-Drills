"""XP = measured learning — `/api/practice/xp` and `/api/practice/xp-target`.

The numbers are `app/learning_xp.py`'s; this file only reads the learner's
time zone and stores their target. The Learner Home XP panel and the topbar
level pill both read GET /xp.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app import learning_xp
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
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
