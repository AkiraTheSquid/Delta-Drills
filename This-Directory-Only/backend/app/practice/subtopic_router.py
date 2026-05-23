"""
Subtopic-stats and weight-update endpoints.

Endpoints (mounted under /api/practice by the parent router):
  GET /subtopics
  PUT /weights
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User
from app.practice_schemas import SubtopicStatsResponse, WeightsUpdateRequest
from app.prioritization import get_subtopic_weights

router = APIRouter()


@router.get("/subtopics", response_model=list[SubtopicStatsResponse])
def list_subtopics(user: User = Depends(get_current_user)) -> list[SubtopicStatsResponse]:
    """List subtopics with adaptive stats, sorted by gradient descending."""
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    weights_info = get_subtopic_weights(user_state)

    return [
        SubtopicStatsResponse(
            subtopic=info["subtopic"],
            topic=info["topic"],
            questions_answered=info["questions_answered"],
            current_difficulty=info["current_difficulty"],
            weight=info["weight"],
            learning_rate=info["learning_rate"],
            gradient=info["gradient"],
            baseline=info["baseline"],
            p=info["p"],
        )
        for info in weights_info
    ]


@router.put("/weights")
def update_weights(
    payload: WeightsUpdateRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """Persist custom per-subtopic effective weights for the user."""
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    user_state.custom_weights = {k: float(v) for k, v in payload.weights.items()}
    save_user_state(user_id)
    return {"ok": True}
