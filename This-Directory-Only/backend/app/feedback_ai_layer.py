"""
Storage for the AI question-repair override layer.

When an allowlisted learner flags a question from the practice UI, Opus 5 is
asked to repair it (see practice/feedback_ai_improver.py) and the repair lands
here as one more override layer on top of the question bank — the same
mechanism every earlier quality pass used (see questions.py
::_load_function_overrides), just written at runtime instead of by a batch
script.

Two files, both JSONL, both keyed by question id:

  ai_feedback_overrides.jsonl   the LIVE layer. One record per repaired
                                question, holding only the fields that
                                changed. Rewritten whole on every write, so a
                                rollback is a line disappearing rather than a
                                tombstone accumulating.
  ai_feedback_revisions.jsonl   append-only audit log. Every apply and every
                                rollback records the full before/after, the
                                feedback that triggered it, and the model that
                                wrote it. This is what makes auto-apply
                                recoverable: nothing here is ever rewritten.

Why its own directory instead of This-Directory-Only/chatgpt/ with the batch
layers: that tree ships inside the Docker image, so a deploy would silently
revert every repair written since the last one. DELTA_FEEDBACK_AI_DIR points
at the Fly /data volume in production (see This-Directory-Only/fly.toml).

Deliberately dependency-free (stdlib only, no app imports) so questions.py can
import it during startup without a cycle.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LAYER_FILENAME = "ai_feedback_overrides.jsonl"
REVISIONS_FILENAME = "ai_feedback_revisions.jsonl"

# Only these may be rewritten by the model. Everything else about a question —
# id, curriculum placement, difficulty, test_cases, atom tags — is off limits,
# because those feed the adaptive engine and the KC lattice rather than what
# the learner reads on screen.
EDITABLE_FIELDS = ("question_text", "starter_code", "answer_code")

_write_lock = Lock()


@contextmanager
def locked(path: Path):
    """Hold an exclusive lock on `path` across PROCESSES, not just threads.

    Every file in this directory is read-modify-rewrite, and the readers and
    writers are not all in one process: the API writes a repair while the local
    runner (ops/question_repair/) claims a job from the same directory. A
    `threading.Lock` is invisible across that boundary, so two processes each
    read the old file and the second write erases the first — silently, because
    both succeed.

    The lock is a sidecar `.lock` file rather than the data file itself, so
    holding it never conflicts with the atomic tmp-file replace underneath.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.with_suffix(path.suffix + ".lock")
    with lock_file.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def feedback_ai_dir() -> Path:
    """Directory holding the live layer + the revision log.

    Defaults next to the batch override layers so a dev checkout works with no
    configuration; production overrides it onto the persistent volume.
    """
    configured = os.environ.get("DELTA_FEEDBACK_AI_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[3] / "This-Directory-Only" / "chatgpt").resolve()


def layer_path() -> Path:
    return feedback_ai_dir() / LAYER_FILENAME


def revisions_path() -> Path:
    return feedback_ai_dir() / REVISIONS_FILENAME


def load_layer() -> Dict[int, dict]:
    """Read the live layer as {question_id: override_record}.

    A malformed file yields an empty layer rather than raising: a bad line here
    must never stop the question bank from loading.
    """
    path = layer_path()
    if not path.exists():
        return {}
    records: Dict[int, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[int(record["id"])] = record
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return {}
    return records


def _write_layer(records: Dict[int, dict]) -> None:
    """Rewrite the live layer atomically, ordered by id for a readable diff."""
    path = layer_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(records[qid], ensure_ascii=False) + "\n"
        for qid in sorted(records)
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def append_revision(entry: dict) -> None:
    """Append one audit record. Append-only; never rewritten."""
    path = revisions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_revisions(question_id: Optional[int] = None) -> List[dict]:
    """Read the audit log, oldest first. Optionally filtered to one question."""
    path = revisions_path()
    if not path.exists():
        return []
    out: List[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if question_id is None or int(record.get("question_id", -1)) == question_id:
                out.append(record)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return out
    return out


def apply_override(
    question_id: int,
    changes: Dict[str, str],
    *,
    before: Dict[str, str],
    trigger: dict,
    model: str,
    rationale: str,
) -> dict:
    """Write `changes` into the live layer and log the revision.

    `changes` holds only fields that actually differ; `before` holds the same
    keys as they read at the moment of the rewrite, so the audit log alone is
    enough to reconstruct the prior text without consulting the bank.
    """
    with _write_lock, locked(layer_path()):
        records = load_layer()
        record = dict(records.get(question_id) or {})
        record["id"] = question_id
        record.update(changes)
        records[question_id] = record
        _write_layer(records)

    entry = {
        "question_id": question_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "applied",
        "model": model,
        "rationale": rationale,
        "fields": sorted(changes),
        "before": before,
        "after": changes,
        "trigger": trigger,
    }
    append_revision(entry)
    logger.info(
        "feedback_ai applied q=%s fields=%s model=%s",
        question_id, sorted(changes), model,
    )
    return entry


def rollback(question_id: int, *, actor: str) -> Optional[dict]:
    """Drop a question's AI override entirely, restoring the batch-layer text.

    Returns the audit entry, or None if the question has no live override —
    rolling back what was never applied is a no-op, not an error.
    """
    with _write_lock, locked(layer_path()):
        records = load_layer()
        removed = records.pop(question_id, None)
        if removed is None:
            return None
        _write_layer(records)

    entry = {
        "question_id": question_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "rolled_back",
        "actor": actor,
        "removed": removed,
    }
    append_revision(entry)
    logger.info("feedback_ai rolled back q=%s by=%s", question_id, actor)
    return entry
