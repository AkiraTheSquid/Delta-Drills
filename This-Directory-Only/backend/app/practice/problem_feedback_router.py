"""
Per-problem quality feedback endpoint.

Lets a learner flag a SPECIFIC question as broken / unclear / wrong-image / good
while they practice, with an optional one-line note. This is content-quality
feedback (is the *problem* sound?), NOT the difficulty rating that feeds the
adaptive engine (that lives in feedback_router.py).

Deliberately isolated: entries are appended to a SIBLING log file
`{user_id}.feedback.json` in the same data dir as the practice state. It never
reads or writes UserPracticeState, so a malformed note can't corrupt mastery /
attempt history. Append-only; safe to grep.

Endpoints (mounted under /api/practice by the parent router):
  POST /problem-feedback   -> append one entry
  GET  /problem-feedback   -> list this user's entries (newest first)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.adaptive import DATA_DIR
from app.auth import get_current_user
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter()

ProblemFeedbackTag = Literal["broken", "unclear", "wrong_image", "good"]


class ProblemFeedbackRequest(BaseModel):
    question_id: int
    tag: ProblemFeedbackTag
    note: str = Field(default="", max_length=2000)
    # For triage context: was the learner marked correct on this attempt?
    correct: Optional[bool] = None


class ProblemFeedbackEntry(BaseModel):
    question_id: int
    tag: ProblemFeedbackTag
    note: str = ""
    correct: Optional[bool] = None
    timestamp: str


class ProblemFeedbackResponse(BaseModel):
    success: bool
    count: int


class ProblemFeedbackListResponse(BaseModel):
    success: bool
    entries: List[ProblemFeedbackEntry]


def _log_file(user_id: str):
    safe_id = user_id.replace("/", "_").replace("..", "_")
    return DATA_DIR / f"{safe_id}.feedback.json"


def _read_entries(user_id: str) -> List[dict]:
    path = _log_file(user_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("entries") or [])
    except Exception as e:  # corrupt file shouldn't break practice
        logger.error("Failed to read problem-feedback log for %s: %s", user_id, e)
        return []


def _write_entries(user_id: str, entries: List[dict]) -> None:
    path = _log_file(user_id)
    payload = {"user_id": user_id, "entries": entries}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@router.post("/problem-feedback", response_model=ProblemFeedbackResponse)
def submit_problem_feedback(
    payload: ProblemFeedbackRequest,
    user: User = Depends(get_current_user),
) -> ProblemFeedbackResponse:
    """Append one per-problem quality flag/note to the user's sibling log."""
    user_id = str(user.id)
    entries = _read_entries(user_id)
    entry = {
        "question_id": payload.question_id,
        "tag": payload.tag,
        "note": payload.note.strip(),
        "correct": payload.correct,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entries.append(entry)
    _write_entries(user_id, entries)
    logger.info(
        "problem_feedback user=%s q=%s tag=%s note=%r",
        user_id, payload.question_id, payload.tag, entry["note"][:120],
    )
    return ProblemFeedbackResponse(success=True, count=len(entries))


@router.get("/problem-feedback", response_model=ProblemFeedbackListResponse)
def list_problem_feedback(
    user: User = Depends(get_current_user),
) -> ProblemFeedbackListResponse:
    """Return this user's per-problem feedback entries, newest first."""
    user_id = str(user.id)
    entries = list(reversed(_read_entries(user_id)))
    return ProblemFeedbackListResponse(
        success=True,
        entries=[ProblemFeedbackEntry(**e) for e in entries],
    )
