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

A submission from an allowlisted learner ALSO hands the flagged question to
Opus 5, which repairs it and writes the fix into the live bank — see
feedback_ai_improver. That runs as a background task: the model never sits in
the request path, and if it is unavailable, misconfigured or unhelpful the
feedback still lands in the log exactly as it always did.

Endpoints (mounted under /api/practice by the parent router):
  POST /problem-feedback            -> append one entry (+ maybe fire a repair)
  GET  /problem-feedback            -> list this user's entries (newest first)
  GET  /problem-feedback/revisions  -> AI repairs applied to the bank
  POST /problem-feedback/rollback   -> revert one question's AI repair
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from pydantic import BaseModel, Field

from app import feedback_ai_layer, questions
from app.adaptive import DATA_DIR
from app.auth import get_current_user
from app.models import User
# Imported by name, not as `from app.practice import ...` — this module is
# itself pulled in by app/practice/__init__.py, and going back through the
# package would make that a cycle.
from app.practice.feedback_ai_improver import (
    allowlist,
    improve_question_from_feedback,
    is_actionable_tag,
    is_enabled_for,
)

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
    # True when this submission also queued an Opus 5 repair of the question.
    # It says the work was queued, not that the bank changed — the repair can
    # still come back "no_change". Poll /problem-feedback/revisions for that.
    improvement_queued: bool = False


class RollbackRequest(BaseModel):
    question_id: int


class RevisionListResponse(BaseModel):
    success: bool
    entries: List[dict]


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
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
) -> ProblemFeedbackResponse:
    """Append one per-problem quality flag/note, and queue an AI repair for it.

    The log write is the contract; the repair is best-effort on top of it. The
    two are kept in that order deliberately — feedback is never lost because
    the improver was misconfigured.
    """
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

    email = (getattr(user, "email", "") or "").strip()
    queued = (
        is_actionable_tag(payload.tag)
        and is_enabled_for(email)
    )
    if queued:
        background_tasks.add_task(
            improve_question_from_feedback,
            payload.question_id,
            payload.tag,
            entry["note"],
            payload.correct,
            email,
        )
    return ProblemFeedbackResponse(
        success=True, count=len(entries), improvement_queued=queued,
    )


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


@router.get("/problem-feedback/revisions", response_model=RevisionListResponse)
def list_ai_revisions(
    question_id: Optional[int] = None,
    user: User = Depends(get_current_user),
) -> RevisionListResponse:
    """AI repairs applied to the bank, newest first.

    The log is global rather than per-user — a repair changes the question
    everyone sees, so hiding it behind the flagger's account would make an
    edit that shipped look like an edit that never happened. Restricted to the
    same allowlist that can trigger one.
    """
    email = (getattr(user, "email", "") or "").strip()
    if email.strip().lower() not in allowlist():
        raise HTTPException(status_code=403, detail="Not permitted")
    entries = list(reversed(feedback_ai_layer.load_revisions(question_id)))
    return RevisionListResponse(success=True, entries=entries)


@router.post("/problem-feedback/rollback", response_model=RevisionListResponse)
def rollback_ai_revision(
    payload: RollbackRequest,
    user: User = Depends(get_current_user),
) -> RevisionListResponse:
    """Revert one question to its shipped text, discarding every AI repair on it.

    This is the undo for auto-apply. It drops the question's whole record from
    the live layer rather than stepping back one revision — the layers below
    are the reviewed, batch-generated bank, which is the state worth returning
    to when a rewrite went wrong.
    """
    email = (getattr(user, "email", "") or "").strip()
    if email.strip().lower() not in allowlist():
        raise HTTPException(status_code=403, detail="Not permitted")

    entry = feedback_ai_layer.rollback(payload.question_id, actor=email)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question {payload.question_id} has no AI revision to roll back",
        )
    questions.reload_questions()
    return RevisionListResponse(success=True, entries=[entry])
