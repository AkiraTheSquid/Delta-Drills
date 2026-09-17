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

A submission from an allowlisted learner ALSO queues a repair job (see
feedback_repair_queue). The repair itself is performed by Seth's local `claude`
CLI via ops/question_repair/run_repairs.py, which pulls the queue through the
endpoints below and posts the rewrite back. The server never calls a model and
holds no model credential; if no runner is listening the job simply waits and
the feedback log is unaffected.

Endpoints (mounted under /api/practice by the parent router):
  POST /problem-feedback                    -> append one entry (+ maybe queue a repair)
  POST /lesson-feedback                     -> append one entry about a LESSON page
  GET  /lesson-feedback                     -> list this user's lesson entries
  GET  /problem-feedback                    -> list this user's entries (newest first)
  GET  /problem-feedback/revisions          -> AI repairs applied to the bank
  POST /problem-feedback/rollback           -> revert one question's AI repair
  GET  /problem-feedback/repair-queue       -> jobs waiting for the local runner
  POST /problem-feedback/repair-queue/claim -> mark one job as being worked on
  POST /problem-feedback/repair-queue/complete -> apply a rewrite / close a job

The four repair endpoints are restricted to the same allowlist that can trigger
a repair, on the plain user JWT the practice UI already issues. That is the
whole auth story on purpose — no service secret to rotate, and the runner
authenticates as the person whose feedback it is acting on.
"""

from __future__ import annotations

import fcntl
import json
import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException

from pydantic import BaseModel, Field

from app import feedback_ai_layer, feedback_repair_queue, questions
from app.adaptive import DATA_DIR
from app.auth import get_current_user
from app.models import User
# Imported by name, not as `from app.practice import ...` — this module is
# itself pulled in by app/practice/__init__.py, and going back through the
# package would make that a cycle.
from app.practice.feedback_ai_improver import (
    DEFAULT_MODEL,
    allowlist,
    apply_repair,
    enqueue_repair,
    is_actionable_tag,
    is_allowlisted,
    is_enabled_for,
    question_snapshot,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ProblemFeedbackTag = Literal["broken", "unclear", "wrong_image", "good"]


class ProblemFeedbackRequest(BaseModel):
    question_id: int
    tag: ProblemFeedbackTag
    note: str = Field(default="", max_length=5000)
    # For triage context: was the learner marked correct on this attempt?
    correct: Optional[bool] = None
    # Minted by the browser when the report is written, so a queued report that
    # is retried is recognised as the SAME report rather than filed twice.
    client_id: str = Field(default="", max_length=64)


# A lesson is prose, so its defects are different ones: nothing here maps onto
# "wrong image", and "broken" means the code in the worked example does not run.
LessonFeedbackTag = Literal["wrong", "confusing", "too_shallow", "too_verbose", "good"]


class LessonFeedbackRequest(BaseModel):
    # The KC the lesson page teaches. Not an int question id: a lesson exists
    # before any question is on screen and can gate several of them.
    kc: str = Field(default="", max_length=200)
    lesson_title: str = Field(default="", max_length=300)
    # The drill the gate was standing in front of, when there is one. Context
    # for triage only — it is NEVER the subject of the feedback, which is why
    # this endpoint does not touch the repair queue.
    question_id: Optional[int] = None
    tag: LessonFeedbackTag
    note: str = Field(default="", max_length=5000)
    client_id: str = Field(default="", max_length=64)


class LessonFeedbackEntry(BaseModel):
    kc: str = ""
    lesson_title: str = ""
    question_id: Optional[int] = None
    tag: LessonFeedbackTag
    note: str = ""
    client_id: Optional[str] = None
    timestamp: str


class LessonFeedbackResponse(BaseModel):
    success: bool
    count: int


class LessonFeedbackListResponse(BaseModel):
    success: bool
    entries: List[LessonFeedbackEntry]


class ProblemFeedbackEntry(BaseModel):
    question_id: int
    tag: ProblemFeedbackTag
    note: str = ""
    correct: Optional[bool] = None
    client_id: Optional[str] = None
    timestamp: str


class ProblemFeedbackResponse(BaseModel):
    success: bool
    count: int
    # True when this submission also queued a repair for the local runner. It
    # says the work was queued, not that the bank changed — the runner can
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


class RepairQueueResponse(BaseModel):
    success: bool
    jobs: List[dict]
    counts: Dict[str, int] = {}


class ClaimRequest(BaseModel):
    job_id: str
    runner: str = "local"


class CompleteRequest(BaseModel):
    job_id: str
    # "rewrite" applies, anything else closes the job untouched. Mirrors the
    # verdict field of REPAIR_JSON_SCHEMA.
    verdict: str = "no_change"
    rationale: str = ""
    question_text: str = ""
    starter_code: str = ""
    answer_code: str = ""
    model: str = DEFAULT_MODEL
    session_id: str = ""
    # Set by the runner when the CLI itself failed, so the job lands in `failed`
    # rather than looking like a considered "leave it alone".
    error: str = ""


class CompleteResponse(BaseModel):
    success: bool
    status: str
    applied_fields: List[str] = []
    revision: Optional[dict] = None


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
    _write_log(_log_file(user_id), user_id, entries)


def _lesson_log_file(user_id: str):
    safe_id = user_id.replace("/", "_").replace("..", "_")
    return DATA_DIR / f"{safe_id}.lesson-feedback.json"


def _read_lesson_entries(user_id: str) -> List[dict]:
    path = _lesson_log_file(user_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("entries") or [])
    except Exception as e:  # corrupt file shouldn't break a lesson
        logger.error("Failed to read lesson-feedback log for %s: %s", user_id, e)
        return []


def _write_lesson_entries(user_id: str, entries: List[dict]) -> None:
    _write_log(_lesson_log_file(user_id), user_id, entries)


def _write_log(path, user_id: str, entries: List[dict]) -> None:
    """Replace a log file atomically.

    A plain write_text truncates first, so a concurrent reader can see an empty
    or half-written file, decide the history is gone, and overwrite it. Write a
    sibling and rename: on the same filesystem os.replace is atomic, and a
    reader sees either the old file or the new one.
    """
    payload = {"user_id": user_id, "entries": entries}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def _log_lock(path):
    """Hold an exclusive lock for one read-modify-write of `path`.

    🔴 Every handler here reads the whole file, appends in memory and writes it
    back. Two requests interleaved on that sequence lose one of the two
    submissions — the second read happens before the first write, so the second
    write drops it. The lock is a sibling file so it never appears in the log
    directory listing as data.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _append_entry(path, user_id: str, read, write, entry: dict) -> List[dict]:
    """Append one entry under the lock, ignoring a replay of one already stored.

    `client_id` is minted by the browser when the report is WRITTEN, so a queued
    report that is retried carries the id it was created with. Without it a
    response lost on the way back — or a tab closed after the server committed
    but before the queue was rewritten — files the same report twice, and for an
    actionable tag that means queueing the same AI repair twice.
    """
    with _log_lock(path):
        entries = read(user_id)
        client_id = entry.get("client_id")
        if client_id and any(e.get("client_id") == client_id for e in entries):
            return entries
        entries.append(entry)
        write(user_id, entries)
        return entries


def _require_allowlisted(user: User) -> str:
    """Repair endpoints are for the people who can trigger a repair, nobody else."""
    email = (getattr(user, "email", "") or "").strip()
    if not is_allowlisted(email):
        raise HTTPException(status_code=403, detail="Not permitted")
    return email


@router.post("/problem-feedback", response_model=ProblemFeedbackResponse)
def submit_problem_feedback(
    payload: ProblemFeedbackRequest,
    user: User = Depends(get_current_user),
) -> ProblemFeedbackResponse:
    """Append one per-problem quality flag/note, and queue a repair for it.

    The log write is the contract; the repair job is best-effort on top of it.
    The two are kept in that order deliberately — feedback is never lost because
    the queue could not be written.
    """
    user_id = str(user.id)
    entry = {
        "question_id": payload.question_id,
        "tag": payload.tag,
        "note": payload.note.strip(),
        "correct": payload.correct,
        "client_id": (payload.client_id or "").strip() or None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entries = _append_entry(
        _log_file(user_id), user_id, _read_entries, _write_entries, entry,
    )
    logger.info(
        "problem_feedback user=%s q=%s tag=%s note=%r",
        user_id, payload.question_id, payload.tag, entry["note"][:120],
    )

    email = (getattr(user, "email", "") or "").strip()
    queued = False
    if is_actionable_tag(payload.tag) and is_enabled_for(email):
        try:
            queued = enqueue_repair(
                payload.question_id, payload.tag, entry["note"], payload.correct, email,
            ) is not None
        except Exception as exc:  # queueing must never fail a feedback submission
            logger.exception("problem_feedback: could not queue repair: %s", exc)
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


@router.post("/lesson-feedback", response_model=LessonFeedbackResponse)
def submit_lesson_feedback(
    payload: LessonFeedbackRequest,
    user: User = Depends(get_current_user),
) -> LessonFeedbackResponse:
    """Append one piece of feedback about a LESSON page.

    Deliberately separate from /problem-feedback rather than a flag on it. That
    endpoint's subject is an integer question id, and an actionable tag on it
    queues an AI rewrite of that question — so a note saying a worked example
    was confusing would have been filed against the drill the lesson gates and
    then used to rewrite it. Different subject, different log, no repair queue.
    """
    user_id = str(user.id)
    entry = {
        "kc": payload.kc.strip(),
        "lesson_title": payload.lesson_title.strip(),
        "question_id": payload.question_id,
        "tag": payload.tag,
        "note": payload.note.strip(),
        "client_id": (payload.client_id or "").strip() or None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entries = _append_entry(
        _lesson_log_file(user_id), user_id, _read_lesson_entries,
        _write_lesson_entries, entry,
    )
    logger.info(
        "lesson_feedback user=%s kc=%s tag=%s note=%r",
        user_id, entry["kc"], payload.tag, entry["note"][:120],
    )
    return LessonFeedbackResponse(success=True, count=len(entries))


@router.get("/lesson-feedback", response_model=LessonFeedbackListResponse)
def list_lesson_feedback(
    user: User = Depends(get_current_user),
) -> LessonFeedbackListResponse:
    """Return this user's lesson feedback entries, newest first."""
    user_id = str(user.id)
    entries = list(reversed(_read_lesson_entries(user_id)))
    return LessonFeedbackListResponse(
        success=True,
        entries=[LessonFeedbackEntry(**e) for e in entries],
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
    _require_allowlisted(user)
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
    email = _require_allowlisted(user)

    entry = feedback_ai_layer.rollback(payload.question_id, actor=email)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question {payload.question_id} has no AI revision to roll back",
        )
    questions.reload_questions()
    return RevisionListResponse(success=True, entries=[entry])


@router.get("/problem-feedback/repair-queue", response_model=RepairQueueResponse)
def list_repair_queue(
    status: str = "pending",
    user: User = Depends(get_current_user),
) -> RepairQueueResponse:
    """Jobs for the local runner.

    `status=pending` (the default) is what a runner polls: never-claimed jobs
    plus jobs whose claim went stale, each carrying a full snapshot of the
    question AS THIS SERVER CURRENTLY SERVES IT. The snapshot is the point — a
    runner on a dev checkout would otherwise repair whatever its own bank says,
    which is not necessarily what the learner saw.

    Any other value returns the raw queue for inspection, unenriched.
    """
    _require_allowlisted(user)
    if status != "pending":
        jobs = [j for j in feedback_repair_queue.load_jobs() if status in ("all", j.get("status"))]
        return RepairQueueResponse(success=True, jobs=jobs, counts=feedback_repair_queue.summary())

    enriched: List[dict] = []
    for job in feedback_repair_queue.pending_jobs():
        question = questions.get_question_by_id(int(job.get("question_id", -1)))
        if question is None:
            # The question left the bank between the flag and the repair. Close
            # the job here rather than handing the runner something it cannot act
            # on; a retired question has nothing to fix.
            feedback_repair_queue.finish(
                job["job_id"], status=feedback_repair_queue.SKIPPED,
                error="question is no longer in the bank",
            )
            continue
        enriched.append({**job, "question": question_snapshot(question)})
    return RepairQueueResponse(
        success=True, jobs=enriched, counts=feedback_repair_queue.summary(),
    )


@router.post("/problem-feedback/repair-queue/claim", response_model=RepairQueueResponse)
def claim_repair_job(
    payload: ClaimRequest,
    user: User = Depends(get_current_user),
) -> RepairQueueResponse:
    """Mark a job as being worked on so a second runner skips it.

    Advisory, not a lock: the claim expires (feedback_repair_queue
    .STALE_CLAIM_SECONDS) so a runner that dies mid-job does not strand the
    flag forever.
    """
    _require_allowlisted(user)
    job = feedback_repair_queue.claim(payload.job_id, runner=payload.runner)
    if job is None:
        raise HTTPException(status_code=409, detail="Job is unknown or already finished")
    return RepairQueueResponse(success=True, jobs=[job])


@router.post("/problem-feedback/repair-queue/complete", response_model=CompleteResponse)
def complete_repair_job(
    payload: CompleteRequest,
    user: User = Depends(get_current_user),
) -> CompleteResponse:
    """Close a job, applying the rewrite if there is one that survives the gates.

    The gates run HERE, again, on the live question — not only in the runner.
    This endpoint is reachable with nothing but an allowlisted token, so it has
    to assume the payload came from somewhere other than the runner.
    """
    email = _require_allowlisted(user)
    job = feedback_repair_queue.get_job(payload.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")

    if payload.error:
        feedback_repair_queue.finish(
            payload.job_id, status=feedback_repair_queue.FAILED,
            error=payload.error, model=payload.model, session_id=payload.session_id,
        )
        return CompleteResponse(success=True, status=feedback_repair_queue.FAILED)

    repair: Dict[str, Any] = {
        "verdict": payload.verdict,
        "rationale": payload.rationale,
        "question_text": payload.question_text,
        "starter_code": payload.starter_code,
        "answer_code": payload.answer_code,
    }
    revision = apply_repair(
        int(job["question_id"]),
        repair,
        tag=str(job.get("tag", "")),
        trigger={
            "job_id": payload.job_id,
            "user_email": job.get("user_email", email),
            "tag": job.get("tag"),
            "note": job.get("note"),
            "correct": job.get("correct"),
            "flagged_at": job.get("created_at"),
            "completed_by": email,
        },
        model=payload.model,
        session_id=payload.session_id,
    )

    status = feedback_repair_queue.DONE if revision else feedback_repair_queue.SKIPPED
    feedback_repair_queue.finish(
        payload.job_id,
        status=status,
        rationale=payload.rationale,
        model=payload.model,
        session_id=payload.session_id,
        applied_fields=(revision or {}).get("fields", []),
    )
    return CompleteResponse(
        success=True,
        status=status,
        applied_fields=(revision or {}).get("fields", []),
        revision=revision,
    )
