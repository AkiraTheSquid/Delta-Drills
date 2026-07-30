"""attempt_log.py — the append-only record of what was served, predicted, and scored.

WHY THIS IS THE PIECE THAT MATTERS
----------------------------------
Every mastery model in this app is a *fitted* model, and it will be refitted.
The posteriors are therefore disposable — they can be rebuilt by replaying
evidence. What cannot be rebuilt is the evidence itself, so the log is the only
piece of the design that is genuinely hard to undo, and it is worth getting
right before the engine is wired into serving rather than after.

Today the evidence is split across two places with different shapes:

  * `adaptive.SubtopicState.history` holds `AttemptRecord`s — question id,
    difficulty, grade, correct, timestamp — keyed by subtopic;
  * `kc_graph`'s `kc_ladder[kc]["attempts"]` holds the ONLY record of which
    rung an attempt was served at.

Neither records the feature values as they stood at serve time, and neither
records what the model predicted. That has two consequences:

  1. **A refit is archaeology, not replay.** Recovering "what was this learner's
     prerequisite mastery when they attempted question 412?" means replaying the
     whole BKT chain and hoping no code path changed in between.
  2. **The model is unfalsifiable.** With no stored `predicted_p` there is no
     Brier score, no reliability curve, and therefore no way to answer "is this
     working?" other than by feel. For a system whose constants are admittedly
     un-fitted guesses, that is the difference between an instrumented
     experiment and a belief.

So: one append-only file per learner, one JSON object per line, every feature
materialised at serve time, the prediction stored beside the outcome, and a
`model_version` on every row so a change to the feature set does not invalidate
history.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not decide anything and it does not hold state. It is a recorder.
Reading it back into posteriors is `replay()`, which is deliberately a pure
function of the log plus a config — that equivalence (state == replay(log)) is
what makes the posteriors safely disposable, and `scripts/test_logistic_engine.py`
asserts it.

FORMAT
------
JSONL, one row per event, appended and never rewritten. JSONL rather than a JSON
array because an append must not require reading or rewriting the file: a
truncated write then costs one corrupt line at the tail instead of the entire
history. `iter_rows` skips unparseable lines with a warning for the same reason.
"""

from __future__ import annotations

import json
import logging
import os

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from app import logistic_engine as E

logger = logging.getLogger(__name__)

# Bumped only for a BREAKING change to the row shape. Additive fields do not
# need a bump — readers use `.get()` — but a renamed or re-meaning'd field does,
# because a replay that silently misreads an old row is worse than one that
# refuses to.
LOG_SCHEMA_VERSION = 1

# Event kinds. `lesson_view` is recorded but never scored: it is the exposure
# signal the first-encounter gate needs, and keeping it in the same stream as
# graded attempts is what lets "has this learner seen the lesson?" be answered
# from the log instead of from a second counter that can drift out of sync with
# it (which is exactly how the current `kc_exposure` / `worked_seen` split
# produced a concept marked as taught with zero worked-examples seen).
KIND_ATTEMPT = "attempt"
KIND_LESSON_VIEW = "lesson_view"
KINDS = (KIND_ATTEMPT, KIND_LESSON_VIEW)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Storage location
#
# Shares DATA_DIR with adaptive.py so both live on the same Fly volume mount
# (`/data/user_data`) and are captured by the same nightly backup. Imported
# lazily inside the function rather than at module scope to keep this module
# importable — and unit-testable — without dragging in the whole adaptive stack.
# ---------------------------------------------------------------------------


def _data_dir() -> Path:
    from app.adaptive import DATA_DIR

    return DATA_DIR


def log_path(user_id: str, base_dir: Optional[Path] = None) -> Path:
    """Path to one learner's log. Sanitised the same way `_state_file` does."""
    safe_id = str(user_id).replace("/", "_").replace("..", "_")
    root = base_dir if base_dir is not None else _data_dir()
    return Path(root) / f"{safe_id}.attempts.jsonl"


# ---------------------------------------------------------------------------
# Row
# ---------------------------------------------------------------------------


@dataclass
class AttemptRow:
    """One event. Flat on purpose — a flat row survives schema drift better than
    a nested one, and every field here is either an identifier, a materialised
    input, a prediction, or an outcome.

    `features` is the full design-matrix entry as evaluated at serve time. It is
    the single most important field: with it, a refit is a replay; without it,
    the row records that something happened but not what the model saw.
    """

    # --- identity -----------------------------------------------------------
    ts: str
    kind: str
    user_id: str
    kc: Optional[str] = None
    question_id: Optional[int] = None
    subtopic: Optional[str] = None
    atoms: List[str] = field(default_factory=list)

    # --- what was served ----------------------------------------------------
    stage: Optional[str] = None
    difficulty_score: Optional[int] = None

    # --- what the model saw and said ---------------------------------------
    features: Dict[str, float] = field(default_factory=dict)
    predicted_p: Optional[float] = None
    predicted_logit_mean: Optional[float] = None
    predicted_logit_var: Optional[float] = None

    # --- what happened ------------------------------------------------------
    correct: Optional[bool] = None
    grade: Optional[float] = None
    latency_ms: Optional[int] = None

    # --- provenance ---------------------------------------------------------
    model_version: str = E.MODEL_VERSION
    schema_version: int = LOG_SCHEMA_VERSION
    note: Optional[str] = None

    def to_json(self) -> str:
        # Compact separators: these files are appended to forever, and the
        # whitespace of an indented dump is a meaningful fraction of the bytes.
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping) -> Optional["AttemptRow"]:
        """Rebuild a row, tolerating unknown fields.

        Unknown keys are DROPPED rather than raising: a row written by a newer
        build must remain readable by an older one, or a rollback loses history.
        """
        if not isinstance(raw, Mapping) or not raw.get("ts") or raw.get("kind") not in KINDS:
            return None
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        try:
            return cls(**kwargs)  # type: ignore[arg-type]
        except TypeError:
            return None

    @property
    def is_graded(self) -> bool:
        """Does this row carry evidence the estimator may consume?

        Four conditions, all required. A lesson view is not evidence. An attempt
        with no outcome is not evidence (it is an in-flight serve). An attempt at
        an unrecognised rung is not evidence, because the stage offset would be
        wrong and a wrong offset biases ability directly.

        And — the condition this originally got wrong — an attempt with no
        recorded feature values is not evidence either. Backfilled rows have a
        valid stage and a real outcome but an empty `features` dict, so the first
        three checks passed them straight into replay. They contributed no
        information to the mean (a zero design row cannot) while still advancing
        replay's clock, so each one inflated the variance of an
        already-estimated posterior: the reconstruction came out *less*
        confident the more history was backfilled. Requiring a usable design row
        is what makes the docstring on `backfill_from_state` true.
        """
        return (
            self.kind == KIND_ATTEMPT
            and self.correct is not None
            and E.normalize_stage(self.stage) in E.GRADED_STAGES
            and bool(self.features)
        )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def append(row: AttemptRow, base_dir: Optional[Path] = None) -> None:
    """Append one row. Never rewrites, never truncates.

    Opened in "a" mode per call rather than holding a handle: appends are rare
    (one per graded question) and a long-lived handle across a Fly machine
    restart is a way to lose the tail.
    """
    path = log_path(row.user_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = row.to_json()
    with path.open("a", encoding="utf-8") as fh:
        # Heal a torn tail before appending. An interrupted write can leave a
        # fragment with no trailing newline; appending straight onto it would
        # fuse the fragment and this row into ONE malformed line, so the reader
        # would drop BOTH — turning a corrupt tail into the silent loss of a
        # good record. That is the exact failure the append-only design exists
        # to prevent, so pay one seek per append to rule it out.
        if fh.tell() > 0:
            with path.open("rb") as probe:
                probe.seek(-1, os.SEEK_END)
                if probe.read(1) != b"\n":
                    fh.write("\n")
        fh.write(line + "\n")
        # Durability over throughput. The whole value of this file is that it
        # survives; a lost tail after a machine restart would be silent and
        # unrecoverable, and one fsync per graded question is free at our rate.
        fh.flush()
        os.fsync(fh.fileno())


def record_attempt(
    user_id: str,
    kc: Optional[str],
    question_id: Optional[int],
    stage: Optional[str],
    values: Mapping[str, float],
    prediction: E.Prediction,
    correct: bool,
    *,
    subtopic: Optional[str] = None,
    atoms: Optional[Sequence[str]] = None,
    difficulty_score: Optional[int] = None,
    grade: Optional[float] = None,
    latency_ms: Optional[int] = None,
    config: E.EngineConfig = E.DEFAULT_CONFIG,
    base_dir: Optional[Path] = None,
    ts: Optional[str] = None,
) -> AttemptRow:
    """Log a graded attempt together with the prediction that preceded it."""
    row = AttemptRow(
        ts=ts or _now_iso(),
        kind=KIND_ATTEMPT,
        user_id=str(user_id),
        kc=kc,
        question_id=question_id,
        subtopic=subtopic,
        atoms=list(atoms or []),
        stage=E.normalize_stage(stage),
        difficulty_score=difficulty_score,
        features={k: float(v) for k, v in values.items()},
        predicted_p=prediction.p,
        predicted_logit_mean=prediction.logit_mean,
        predicted_logit_var=prediction.logit_var,
        correct=bool(correct),
        grade=grade,
        latency_ms=latency_ms,
        model_version=config.version,
    )
    append(row, base_dir)
    return row


def record_lesson_view(
    user_id: str,
    kc: str,
    *,
    question_id: Optional[int] = None,
    note: Optional[str] = None,
    base_dir: Optional[Path] = None,
    ts: Optional[str] = None,
) -> AttemptRow:
    """Log that a lesson screen was shown for `kc`.

    Carries no prediction and no outcome by construction — reading is not
    answering. This is the row that answers "have I already seen this lesson?",
    which is the question the current two-counter arrangement gets wrong.
    """
    row = AttemptRow(
        ts=ts or _now_iso(),
        kind=KIND_LESSON_VIEW,
        user_id=str(user_id),
        kc=kc,
        question_id=question_id,
        stage=E.STAGE_LESSON,
        note=note,
    )
    append(row, base_dir)
    return row


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def iter_rows(user_id: str, base_dir: Optional[Path] = None) -> Iterator[AttemptRow]:
    """Stream a learner's log in write order.

    A malformed line is skipped with a warning rather than aborting the read: a
    partial final line from an interrupted write must not make the preceding
    months unreadable.
    """
    path = log_path(user_id, base_dir)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("attempt_log: unparseable line %s:%d — skipped", path, lineno)
                continue
            row = AttemptRow.from_dict(raw)
            if row is None:
                logger.warning("attempt_log: unusable row %s:%d — skipped", path, lineno)
                continue
            yield row


def rows_for_kc(user_id: str, kc: str, base_dir: Optional[Path] = None) -> List[AttemptRow]:
    return [r for r in iter_rows(user_id, base_dir) if r.kc == kc]


def has_seen_lesson(user_id: str, kc: str, base_dir: Optional[Path] = None) -> bool:
    """Has this learner ever been shown the lesson for `kc`?

    One counter, one source. The bug this exists to prevent: two independent
    "has been taught" counters (`kc_exposure` and `kc_ladder[kc]["worked_seen"]`)
    that can disagree, so a lesson gets re-taught to someone who has already
    read it. As of 2026-07-30 the ladder no longer demotes anyone back to the
    lesson rung (`kc_graph._stage_from` floors demotion at `faded`), which
    closes the way the two used to drift apart — but they remain separate
    fields written by separate endpoints, so read the question through here
    rather than trusting either one.
    """
    return any(r.kind == KIND_LESSON_VIEW and r.kc == kc for r in iter_rows(user_id, base_dir))


# ---------------------------------------------------------------------------
# Replay — the property that makes posteriors disposable
# ---------------------------------------------------------------------------


def replay(
    user_id: str,
    kc: str,
    config: E.EngineConfig = E.DEFAULT_CONFIG,
    base_dir: Optional[Path] = None,
    rows: Optional[Sequence[AttemptRow]] = None,
) -> Tuple[Dict[str, E.Posterior], int]:
    """Rebuild the posteriors for one KC from the log alone.

    Returns `(posteriors, n_rows_consumed)`.

    Rows whose `model_version` differs from `config.version` are still consumed.
    That is a judgement call and worth naming: the alternative — refusing them —
    would silently discard all history on every model bump, which at our data
    volume means starting from zero forever. The stored `features` dict is the
    thing being replayed, and it is version-independent; what a version bump
    actually invalidates is the stored `predicted_p`, which replay does not use.
    Calibration analysis, which DOES use `predicted_p`, must filter by version.

    Time inflation between attempts is applied here so a replay reproduces the
    live path exactly rather than approximating it.
    """
    source = list(rows) if rows is not None else rows_for_kc(user_id, kc, base_dir)
    # Filter by KC even when rows were injected. The file-backed path already
    # scopes to one concept, so an injected path that did not would make the
    # same call mean two different things — and the natural way to use the
    # parameter (hand it a learner's whole history to avoid re-reading the file)
    # would silently reconstruct this KC out of every other concept's attempts.
    graded = [r for r in source if r.is_graded and (r.kc == kc or r.kc is None)]

    posteriors: Dict[str, E.Posterior] = {
        f.name: E.initial_posterior(f) for f in config.learned_features
    }
    prev_ts: Optional[datetime] = None
    consumed = 0

    for row in graded:
        ts = _parse_ts(row.ts)
        days = 0.0
        if ts is not None and prev_ts is not None:
            days = max((ts - prev_ts).total_seconds() / 86400.0, 0.0)
        # Same entry point the live path uses. Reimplementing inflate-then-update
        # here is how replay and live silently diverge.
        posteriors, _ = E.step(
            row.features,
            posteriors,
            bool(row.correct),
            config,
            days_elapsed=days,
            timestamp=row.ts,
        )
        prev_ts = ts or prev_ts
        consumed += 1

    return posteriors, consumed


def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp to an **aware UTC** datetime.

    Normalising to aware is not cosmetic. Legacy state and backfilled rows carry
    naive timestamps while everything this module writes carries `Z`; replay
    subtracts consecutive timestamps, and subtracting a naive from an aware
    datetime raises TypeError — which would abort the reconstruction of a
    learner's entire history at the first legacy row. A naive timestamp is
    assumed UTC, which is what every producer in this app actually writes.
    """
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


_parse_ts = parse_ts  # legacy alias


# ---------------------------------------------------------------------------
# Calibration — is any of this working?
# ---------------------------------------------------------------------------


def calibration(
    user_id: str,
    base_dir: Optional[Path] = None,
    model_version: Optional[str] = None,
    bins: int = 5,
) -> dict:
    """Brier score and a reliability table over stored predictions.

    This is the honest answer to "is the model any good", and it is available
    only because `predicted_p` was stored before the outcome was known. Compare
    `brier` against `brier_baseline` (always predicting the base rate): a model
    that cannot beat its own base rate is not yet earning its complexity.

    `model_version` filters — mixing predictions from two feature sets makes the
    number meaningless.
    """
    rows = [
        r
        for r in iter_rows(user_id, base_dir)
        if r.is_graded
        and r.predicted_p is not None
        and (model_version is None or r.model_version == model_version)
    ]
    if not rows:
        return {"n": 0, "brier": None, "brier_baseline": None, "bins": []}

    n = len(rows)
    outcomes = [1.0 if r.correct else 0.0 for r in rows]
    base_rate = sum(outcomes) / n
    brier = sum((float(r.predicted_p) - y) ** 2 for r, y in zip(rows, outcomes)) / n
    brier_baseline = sum((base_rate - y) ** 2 for y in outcomes) / n

    table: List[dict] = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        # Closed at the top of the last bin so p == 1.0 is counted somewhere.
        sel = [
            (r, y)
            for r, y in zip(rows, outcomes)
            if lo <= float(r.predicted_p) < hi or (i == bins - 1 and float(r.predicted_p) == 1.0)
        ]
        if not sel:
            continue
        table.append(
            {
                "range": [round(lo, 3), round(hi, 3)],
                "n": len(sel),
                "mean_predicted": sum(float(r.predicted_p) for r, _ in sel) / len(sel),
                "observed": sum(y for _, y in sel) / len(sel),
            }
        )

    return {
        "n": n,
        "base_rate": base_rate,
        "brier": brier,
        "brier_baseline": brier_baseline,
        "beats_baseline": brier < brier_baseline,
        "bins": table,
    }


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def backfill_from_state(
    user_id: str,
    user_state,
    base_dir: Optional[Path] = None,
    dry_run: bool = True,
) -> dict:
    """Seed the log from the existing split state (history + kc_ladder).

    The old records cannot supply what was never stored — there are no feature
    values and no predictions — so backfilled rows carry `note="backfill"`, an
    empty `features` dict, and no `predicted_p`. They are therefore **excluded
    from replay** (an empty feature dict makes `is_graded` fail on stage, and
    even where a stage survives, a zero design row carries no information) and
    excluded from calibration.

    The point is not to recover the estimator's state — that evidence is gone.
    It is to make the log the complete record of WHAT WAS SERVED, so served-id
    de-duplication and "have I seen this?" questions have one source going
    forward.

    Defaults to `dry_run=True`: this writes to a learner's permanent history and
    is not idempotent, so the caller has to ask for it explicitly.
    """
    written: List[AttemptRow] = []

    ladder = getattr(user_state, "kc_ladder", None) or {}
    for kc, row in ladder.items():
        if not isinstance(row, dict):
            continue
        for att in row.get("attempts") or []:
            if not isinstance(att, dict):
                continue
            written.append(
                AttemptRow(
                    ts=att.get("ts") or att.get("timestamp") or _now_iso(),
                    kind=KIND_ATTEMPT,
                    user_id=str(user_id),
                    kc=kc,
                    question_id=att.get("question_id"),
                    stage=E.normalize_stage(att.get("stage")),
                    correct=att.get("correct"),
                    model_version="backfill",
                    note="backfill:kc_ladder",
                )
            )

    exposure = getattr(user_state, "kc_exposure", None) or {}
    for kc, seen_at in exposure.items():
        written.append(
            AttemptRow(
                ts=seen_at if isinstance(seen_at, str) else _now_iso(),
                kind=KIND_LESSON_VIEW,
                user_id=str(user_id),
                kc=kc,
                stage=E.STAGE_LESSON,
                model_version="backfill",
                note="backfill:kc_exposure",
            )
        )

    written.sort(key=lambda r: r.ts)

    # Not idempotent by nature — appending the same history twice would double
    # every backfilled row, and the log is append-only so there is no undo. A
    # migration is exactly the kind of thing that gets run twice (a retry, a
    # second machine, a nervous operator), so refuse rather than trust.
    already = sum(1 for r in iter_rows(user_id, base_dir) if r.note and r.note.startswith("backfill"))
    if already:
        return {
            "dry_run": dry_run,
            "skipped": True,
            "reason": f"log already holds {already} backfilled rows",
            "rows": 0,
            "attempts": 0,
            "lesson_views": 0,
            "path": str(log_path(user_id, base_dir)),
        }

    if not dry_run:
        for row in written:
            append(row, base_dir)

    return {
        "dry_run": dry_run,
        "skipped": False,
        "rows": len(written),
        "attempts": sum(1 for r in written if r.kind == KIND_ATTEMPT),
        "lesson_views": sum(1 for r in written if r.kind == KIND_LESSON_VIEW),
        "path": str(log_path(user_id, base_dir)),
    }
