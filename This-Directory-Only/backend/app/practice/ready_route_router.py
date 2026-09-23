"""POST /api/practice/ready-route — the next question of a "practice until
ready" session on one ARENA exercise (app/ready_route.py).

Read-only: the route is replayed from the learner's ladder record on every
call, and the answers themselves arrive through the ordinary submit path.
POST rather than GET only because the session's served ids are a list.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app import ready_route
from app.adaptive import get_user_state
from app.auth import get_current_user
from app.models import User

router = APIRouter()


class ReadyRouteRequest(BaseModel):
    kc: str
    exercise_ids: List[int] = Field(default_factory=list)
    served: List[int] = Field(default_factory=list)
    skip: List[int] = Field(default_factory=list)


class ReadyRouteResponse(BaseModel):
    done: bool
    reason: Optional[str] = None
    target: str
    p_target: Optional[float] = None
    ready_at: Optional[float] = None
    question_id: Optional[int] = None
    kc: Optional[str] = None
    kc_title: Optional[str] = None
    rung: Optional[str] = None
    mode: Optional[str] = None
    attempt: bool = False
    p_kc: Optional[float] = None
    path: List[str] = Field(default_factory=list)
    beliefs: Dict[str, float] = Field(default_factory=dict)


@router.post("/ready-route", response_model=ReadyRouteResponse)
def post_ready_route(body: ReadyRouteRequest, user: User = Depends(get_current_user)) -> ReadyRouteResponse:
    user_state = get_user_state(str(user.id))
    return ReadyRouteResponse(**ready_route.plan(
        user_state, body.kc, body.exercise_ids, body.served, body.skip))
