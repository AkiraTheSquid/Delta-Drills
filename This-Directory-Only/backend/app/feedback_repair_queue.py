"""
Queue of question repairs waiting for the local Claude Code runner.

The backend no longer calls a model itself. When an allowlisted learner flags a
question, the flag is turned into a JOB here and that is the end of the request.
The work is done on Seth's own machine by `ops/question_repair/run_repairs.py`,
which drives the `claude` CLI under his local login — no API key, no server-side
model credential, and the repair is visible in his terminal while it happens.

One file, JSONL, beside the override layer and the revision log:

  ai_feedback_queue.jsonl   one record per flagged question. Rewritten whole on
                            every status change, so the file is always the
                            current state of the queue rather than a history —
                            the history is ai_feedback_revisions.jsonl.

Job lifecycle:

  pending  -> queued by the feedback endpoint, nothing has looked at it yet
  running  -> a runner has claimed it (claim is advisory: it stops two runners
              on the same machine racing, it is not a lock)
  done     -> a repair was applied to the bank
  skipped  -> the runner looked and decided nothing should change (no_change
              verdict, or every proposed field failed a gate)
  failed   -> the runner could not finish (CLI error, timeout, bad output)

`skipped` and `failed` are deliberately distinct: skipped means the loop worked
and the answer was "leave it alone", failed means the loop broke and the flag
still deserves a human.

Stdlib plus feedback_ai_layer (for its cross-process file lock) and nothing
else — same reason as that module: this has to be importable from the questions
bank and from a standalone runner script without dragging FastAPI in.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from app.feedback_ai_layer import locked

logger = logging.getLogger(__name__)

QUEUE_FILENAME = "ai_feedback_queue.jsonl"

PENDING = "pending"
RUNNING = "running"
DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"

OPEN_STATUSES = (PENDING, RUNNING)
TERMINAL_STATUSES = (DONE, SKIPPED, FAILED)

# A claim older than this is treated as abandoned and handed back to the next
# runner. A repair is one CLI call plus a grading-harness verification; twenty
# minutes means the runner died, not that it is still thinking.
STALE_CLAIM_SECONDS = 20 * 60

_write_lock = Lock()


def queue_dir() -> Path:
    """Same directory as the override layer — see feedback_ai_layer.feedback_ai_dir."""
    configured = os.environ.get("DELTA_FEEDBACK_AI_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[3] / "This-Directory-Only" / "chatgpt").resolve()


def queue_path() -> Path:
    return queue_dir() / QUEUE_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jobs() -> List[dict]:
    """Every job, oldest first. A malformed file reads as an empty queue."""
    path = queue_path()
    if not path.exists():
        return []
    jobs: List[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return []
    return jobs


def _write_jobs(jobs: List[dict]) -> None:
    path = queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _age_seconds(stamp: str) -> float:
    try:
        then = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


def enqueue(
    *,
    question_id: int,
    tag: str,
    note: str,
    correct: Optional[bool],
    user_email: str,
) -> dict:
    """Queue one flagged question for repair.

    Re-flagging a question that is already PENDING replaces the waiting job
    rather than adding a second one: the newest note is the best description of
    what is wrong, and two runners repairing the same question from two notes
    would each overwrite the other's override.

    A job already RUNNING is left alone, and the new flag queues beside it. The
    runner holding it is mid-session and will come back to close it by id;
    deleting it under the runner throws that work away and — in remote mode —
    fails the completion with a 404 for a repair that was already paid for.
    """
    job = {
        "job_id": uuid.uuid4().hex,
        "question_id": int(question_id),
        "tag": tag,
        "note": note,
        "correct": correct,
        "user_email": user_email,
        "status": PENDING,
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _write_lock, locked(queue_path()):
        jobs = [
            existing
            for existing in load_jobs()
            if not (
                int(existing.get("question_id", -1)) == int(question_id)
                and existing.get("status") == PENDING
            )
        ]
        jobs.append(job)
        _write_jobs(jobs)
    logger.info(
        "repair_queue enqueued job=%s q=%s tag=%s", job["job_id"], question_id, tag,
    )
    return job


def pending_jobs() -> List[dict]:
    """Jobs a runner should pick up: never claimed, or claimed and abandoned."""
    out = []
    for job in load_jobs():
        status = job.get("status")
        if status == PENDING:
            out.append(job)
        elif status == RUNNING and _age_seconds(job.get("updated_at", "")) > STALE_CLAIM_SECONDS:
            out.append(job)
    return out


def get_job(job_id: str) -> Optional[dict]:
    for job in load_jobs():
        if job.get("job_id") == job_id:
            return job
    return None


def update_job(job_id: str, **fields) -> Optional[dict]:
    """Merge `fields` into one job and rewrite the queue. None if unknown id."""
    with _write_lock, locked(queue_path()):
        return _update_job_locked(job_id, **fields)


def _update_job_locked(job_id: str, **fields) -> Optional[dict]:
    """update_job's body, for callers that already hold the queue lock."""
    jobs = load_jobs()
    updated: Optional[dict] = None
    for job in jobs:
        if job.get("job_id") == job_id:
            job.update(fields)
            job["updated_at"] = _now()
            updated = job
            break
    if updated is None:
        return None
    _write_jobs(jobs)
    return updated


def claim(job_id: str, *, runner: str) -> Optional[dict]:
    """Take exclusive ownership of a job. None if someone else already has it.

    The test and the write happen under one lock. Read-then-write would let two
    runners that polled the same pending job both succeed, and they would then
    repair the same question twice and each overwrite the other's override —
    the second one silently, because both writes report success.

    A RUNNING job may only be taken over once its claim has gone stale, which is
    how a runner that died mid-job hands the work back.
    """
    with _write_lock, locked(queue_path()):
        job = next((j for j in load_jobs() if j.get("job_id") == job_id), None)
        if job is None:
            return None
        status = job.get("status")
        if status == RUNNING and _age_seconds(job.get("updated_at", "")) <= STALE_CLAIM_SECONDS:
            return None
        if status not in (PENDING, RUNNING):
            return None
        return _update_job_locked(job_id, status=RUNNING, runner=runner, claimed_at=_now())


def finish(job_id: str, *, status: str, **fields) -> Optional[dict]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"{status!r} is not a terminal job status")
    return update_job(job_id, status=status, finished_at=_now(), **fields)


def prune(keep: int = 200) -> int:
    """Drop the oldest terminal jobs, keeping the queue file readable.

    Open jobs are never pruned regardless of count — losing one would silently
    drop a learner's report. Returns how many records were removed.
    """
    with _write_lock, locked(queue_path()):
        jobs = load_jobs()
        terminal = [j for j in jobs if j.get("status") in TERMINAL_STATUSES]
        if len(terminal) <= keep:
            return 0
        drop = {id(j) for j in terminal[: len(terminal) - keep]}
        remaining = [j for j in jobs if id(j) not in drop]
        _write_jobs(remaining)
    return len(jobs) - len(remaining)


def summary() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for job in load_jobs():
        status = str(job.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts
