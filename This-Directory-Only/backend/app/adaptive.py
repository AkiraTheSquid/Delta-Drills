"""
Adaptive difficulty algorithm — v0.

Per-user, per-subtopic state that determines what difficulty level to
serve next based on running performance metrics.

Key variables per (user, subtopic):
  - n:                 number of questions answered
  - baseline:          running weighted average of score (∈ [0, 100])
  - p:                 running correctness rate (∈ [0, 1])
  - target_difficulty: what numeric difficulty to serve next (0-100 scale)
  - history:           list of past attempt records
  - last_update_ts:    ISO-8601 UTC of last EWMA update; drives time decay

Cold start (n <= 3):
  First 3 questions sample with target difficulties 25, 50, 75

For n > 3, after each problem:
  1. Decay stored baseline (→ 0) and p (→ 0.5) by half-life since last_update_ts.
  2. Feedback alpha: "not_much" -> 0.3, "somewhat" -> 0.6, "a_lot" -> 0.80.
     score(n) = grade(n) * difficulty(n) / 100
     baseline(n) = alpha * score(n) + (1 - alpha) * baseline(n-1)
     indicator = 1 if grade > 85, else 0
     p(n) = p_alpha * indicator + (1 - p_alpha) * p(n-1)
  3. target_difficulty = baseline * difficulty_multiplier(p)

Grade is binary: 100 if correct, 0 if incorrect.

Calibration status: every numeric constant in this module is a v0 default,
not literature-derived. See `papers/MASTERY_ESTIMATION_REFERENCE_v2.md` for
the 2026-05-24 source audit that established this. Tune empirically once
per-atom attempt data accumulates.

Decay rationale: Yudelson & Pavlik 2013 (survived audit) flags monotonic
mastery curves as anti-pattern; HLR (Settles & Meeder 2016) is the cleanest
deployed analog for time-decay on a mastery posterior. This module's
multiplicative half-life shrinkage is the survived-audit principle, not
the v1 deep-research doc's fabricated Beta-Bernoulli update rule.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# "unrated" is a graded attempt that was never rated, and it is deliberately NOT
# in FEEDBACK_ALPHA: the learner said nothing about how hard it felt, so it
# carries no alpha, and recording it as one of the three real answers would
# invent an opinion. It exists because one route has no felt-difficulty step at
# all — the Colab edition, where running the notebook's checker IS the submit —
# and without a level to finalize under, those attempts stayed pending forever
# and never entered anyone's history. Kept distinct on the record so a reader
# can tell "not asked" from "asked and answered". Mirrors `UNRATED` in
# Local_Deployed_Shared/practice_engine.py, the offline twin of this engine.
FeedbackLevel = Literal["not_much", "somewhat", "a_lot", "unrated"]
UNRATED: FeedbackLevel = "unrated"

# ---------------------------------------------------------------------------
# Tunable parameters — v0 calibration defaults, not literature-grounded.
# Re-derive empirically from real attempt data before claiming any of these
# are tuned. See `papers/MASTERY_ESTIMATION_REFERENCE_v2.md` for the audit
# that established these are doc-author choices, not citations.
# ---------------------------------------------------------------------------

FEEDBACK_ALPHA: Dict[FeedbackLevel, float] = {
    "not_much": 0.3,
    "somewhat": 0.6,
    "a_lot": 0.80,
}

# Separate EWMA smoothing for correctness rate (independent of feedback).
P_ALPHA = 0.3

# Cold start target difficulties for the first 3 questions in each subtopic
COLD_START_TARGETS = [25, 50, 75]

# Half-life decay (v0) — pulls baseline toward 0 and p toward P_PRIOR with this
# time constant. Yudelson & Pavlik 2013 (survived 2026-05-24 audit) flags
# monotonic mastery as anti-pattern; this is the regression mechanism.
HALF_LIFE_DAYS: float = 14.0
P_PRIOR: float = 0.5      # no-information prior for correctness rate
BASELINE_PRIOR: float = 0.0  # no-information prior for accumulated score


@dataclass
class AttemptRecord:
    """Single attempt on a question."""
    question_id: int
    subtopic: str
    difficulty_score: int
    grade: float          # 0 or 100
    correct: bool
    timestamp: str        # ISO-8601 UTC time
    feedback: Optional[FeedbackLevel] = None
    alpha: Optional[float] = None
    score: Optional[float] = None
    baseline_after: Optional[float] = None
    p_after: Optional[float] = None
    target_difficulty_after: Optional[float] = None


@dataclass
class SubtopicState:
    """Tracks adaptive state for one (user, subtopic) pair."""
    subtopic: str
    n: int = 0                          # questions answered
    baseline: float = 0.0               # running weighted average of score
    p: float = 0.5                      # running correctness rate
    target_difficulty: float = 25.0     # what difficulty to serve next
    history: List[AttemptRecord] = field(default_factory=list)
    # Track which question IDs have been served to avoid repeats
    served_question_ids: List[int] = field(default_factory=list)
    # ISO-8601 UTC timestamp of last EWMA update; None until first attempt.
    last_update_ts: Optional[str] = None


@dataclass
class UserPracticeState:
    """All adaptive state for a single user."""
    user_id: str
    subtopic_states: Dict[str, SubtopicState] = field(default_factory=dict)
    # The question currently being worked on (before feedback)
    pending_attempt: Optional[AttemptRecord] = None
    # User-defined effective weights per subtopic key (e.g. "Numpy: Core array literacy" -> 0.175)
    # Empty dict means fall back to uniform weights.
    custom_weights: Dict[str, float] = field(default_factory=dict)
    # Per-atom BKT mastery posterior P(known) ∈ [0, 1], keyed by concept-graph
    # atom id. THE prioritization/readiness signal (see bkt_mastery.py). The
    # per-subtopic EWMA above is retained only as a learner-facing area readout.
    atom_mastery: Dict[str, float] = field(default_factory=dict)
    # ISO-8601 UTC of the last update to each atom's posterior; drives decay.
    atom_last_ts: Dict[str, str] = field(default_factory=dict)
    # Self-reported experience: None | "beginner" | "strong". Seeds the BKT
    # prior for never-practiced atoms (bkt_mastery.params_for_level) so the
    # first questions start near the learner's level; evidence overrules it
    # within a couple of attempts and it never unlocks anything by itself.
    self_reported_level: Optional[str] = None
    # ALEKS-style placement diagnostic state (probe log + flags) — owned and
    # interpreted by diagnostic.py; kept as a plain JSON-able dict so the
    # posterior is always recomputed from the probe log, never persisted.
    diagnostic: Dict = field(default_factory=dict)
    # First-encounter exposure: kc_id -> ISO-8601 UTC of when the learner
    # completed the introducing KP. Drives the lesson gate (lessons.py) —
    # a question whose target KC is absent here triggers lesson-first.
    kc_exposure: Dict[str, str] = field(default_factory=dict)
    # Expertise-reversal ladder state, per KC:
    #   {kc: {"worked_seen": int, "attempts": [{"correct": bool, "stage": str,
    #                                           "ts": str}, ...]}}
    # Owned and interpreted by kc_graph.py (kc_stage / record_kc_outcome). Kept
    # separate from subtopic history because the ladder is a PER-CONCEPT
    # decision: a subtopic holds many KCs, and a learner fluent in one of them
    # must not have the scaffolding pulled out from under a different one.
    kc_ladder: Dict[str, dict] = field(default_factory=dict)

    def get_subtopic_state(self, subtopic: str) -> SubtopicState:
        if subtopic not in self.subtopic_states:
            self.subtopic_states[subtopic] = SubtopicState(subtopic=subtopic)
        return self.subtopic_states[subtopic]


# ---------------------------------------------------------------------------
# File-backed store keyed by user_id (string UUID)
# Saves to a JSON file in the data directory. For production, swap with
# Supabase or database persistence.
# ---------------------------------------------------------------------------
_user_states: Dict[str, UserPracticeState] = {}

from app.config import settings as _settings
DATA_DIR = (
    Path(_settings.user_data_dir)
    if _settings.user_data_dir
    else Path(__file__).resolve().parent.parent / "user_data"
)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _state_file(user_id: str) -> Path:
    """Return the JSON file path for a user's practice state."""
    safe_id = user_id.replace("/", "_").replace("..", "_")
    return DATA_DIR / f"{safe_id}.json"


def _save_user_state(state: UserPracticeState) -> None:
    """Persist user state to a JSON file."""
    data = {
        "user_id": state.user_id,
        "pending_attempt": asdict(state.pending_attempt) if state.pending_attempt else None,
        "custom_weights": state.custom_weights,
        "atom_mastery": state.atom_mastery,
        "atom_last_ts": state.atom_last_ts,
        "self_reported_level": state.self_reported_level,
        "diagnostic": state.diagnostic,
        "kc_exposure": state.kc_exposure,
        "kc_ladder": state.kc_ladder,
        "subtopic_states": {},
    }
    for sub_name, sub_state in state.subtopic_states.items():
        data["subtopic_states"][sub_name] = {
            "subtopic": sub_state.subtopic,
            "n": sub_state.n,
            "baseline": sub_state.baseline,
            "p": sub_state.p,
            "target_difficulty": sub_state.target_difficulty,
            "served_question_ids": sub_state.served_question_ids,
            "history": [asdict(a) for a in sub_state.history],
            "last_update_ts": sub_state.last_update_ts,
        }
    try:
        _state_file(state.user_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save user state for %s: %s", state.user_id, e)


def _load_user_state(user_id: str) -> Optional[UserPracticeState]:
    """Load user state from a JSON file, or return None if not found."""
    path = _state_file(user_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = UserPracticeState(user_id=data["user_id"])
        state.custom_weights = data.get("custom_weights") or {}
        # Additive, back-compat: older saves predate per-atom BKT state.
        state.atom_mastery = data.get("atom_mastery") or {}
        state.atom_last_ts = data.get("atom_last_ts") or {}
        state.self_reported_level = data.get("self_reported_level")
        state.diagnostic = data.get("diagnostic") or {}
        state.kc_exposure = data.get("kc_exposure") or {}
        # Additive, back-compat: saves predating the ladder simply start it
        # empty, which reads as "no worked example seen" — the correct cold
        # state, so an existing learner re-enters each concept at its example.
        state.kc_ladder = data.get("kc_ladder") or {}
        if data.get("pending_attempt"):
            pa = data["pending_attempt"]
            state.pending_attempt = AttemptRecord(
                question_id=pa["question_id"],
                subtopic=pa["subtopic"],
                difficulty_score=pa["difficulty_score"],
                grade=pa["grade"],
                correct=pa["correct"],
                timestamp=pa.get("timestamp") or "",
                feedback=pa.get("feedback"),
                alpha=pa.get("alpha"),
                score=pa.get("score"),
                baseline_after=pa.get("baseline_after"),
                p_after=pa.get("p_after"),
                target_difficulty_after=pa.get("target_difficulty_after"),
            )
        for sub_name, sub_data in data.get("subtopic_states", {}).items():
            history = []
            for a in sub_data.get("history", []):
                history.append(AttemptRecord(
                    question_id=a["question_id"],
                    subtopic=a["subtopic"],
                    difficulty_score=a["difficulty_score"],
                    grade=a["grade"],
                    correct=a["correct"],
                    timestamp=a.get("timestamp") or "",
                    feedback=a.get("feedback"),
                    alpha=a.get("alpha"),
                    score=a.get("score"),
                    baseline_after=a.get("baseline_after"),
                    p_after=a.get("p_after"),
                    target_difficulty_after=a.get("target_difficulty_after"),
                ))
            # last_update_ts: backfill from last history entry if missing on
            # older saved states (so existing users get decay starting now,
            # not retroactively).
            last_ts = sub_data.get("last_update_ts")
            if last_ts is None and history:
                last_ts = history[-1].timestamp or None
            state.subtopic_states[sub_name] = SubtopicState(
                subtopic=sub_data["subtopic"],
                n=sub_data["n"],
                baseline=sub_data["baseline"],
                p=sub_data["p"],
                target_difficulty=sub_data["target_difficulty"],
                served_question_ids=sub_data.get("served_question_ids", []),
                history=history,
                last_update_ts=last_ts,
            )
        return state
    except Exception as e:
        logger.error("Failed to load user state for %s: %s", user_id, e)
        return None


def get_user_state(user_id: str) -> UserPracticeState:
    if user_id not in _user_states:
        loaded = _load_user_state(user_id)
        if loaded:
            _user_states[user_id] = loaded
        else:
            _user_states[user_id] = UserPracticeState(user_id=user_id)
    return _user_states[user_id]


def save_user_state(user_id: str) -> None:
    """Public API to persist a user's state after changes."""
    if user_id in _user_states:
        _save_user_state(_user_states[user_id])


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


# EWMA difficulty/decay helpers removed — difficulty is BKT-driven
# (prioritization.target_difficulty) and per-atom forgetting lives in
# bkt_mastery.decay(). SubtopicState.baseline/p are now snapshots of BKT subtopic
# mastery written by feedback_router (for the Statistics panel), not EWMA.


def record_attempt(
    user_state: UserPracticeState,
    question_id: int,
    subtopic: str,
    difficulty_score: int,
    correct: bool,
) -> AttemptRecord:
    """
    Record that the user answered a question (before feedback).
    The attempt is stored as pending until feedback is provided.
    """
    grade = 100.0 if correct else 0.0
    timestamp = datetime.now(timezone.utc).isoformat()
    attempt = AttemptRecord(
        question_id=question_id,
        subtopic=subtopic,
        difficulty_score=difficulty_score,
        grade=grade,
        correct=correct,
        timestamp=timestamp,
    )
    user_state.pending_attempt = attempt
    return attempt


def apply_feedback(
    user_state: UserPracticeState,
    feedback: FeedbackLevel,
) -> Optional[AttemptRecord]:
    """
    Apply user feedback to the pending attempt and update the adaptive state.
    Returns the finalized attempt record, or None if no pending attempt.
    """
    attempt = user_state.pending_attempt
    if attempt is None:
        return None

    attempt.feedback = feedback
    attempt.alpha = FEEDBACK_ALPHA.get(feedback)

    sub_state = user_state.get_subtopic_state(attempt.subtopic)
    sub_state.n += 1
    sub_state.last_update_ts = attempt.timestamp
    sub_state.history.append(attempt)
    user_state.pending_attempt = None

    # NOTE: EWMA baseline/p are NO LONGER computed here. They (and the attempt's
    # baseline_after / p_after) are snapshotted from the per-atom BKT mastery by
    # the caller (feedback_router) AFTER the BKT update runs — so the Statistics
    # panel, which reads these same fields, plots the BKT mastery trajectory
    # (0-1 mapped to 0-100) instead of EWMA. apply_feedback now only finalizes
    # the attempt + appends history; all scoring is BKT. See
    # bkt_mastery.subtopic-mastery snapshot in feedback_router.
    return attempt


def override_pending_attempt(
    user_state: UserPracticeState,
    question_id: int,
    correct: bool = True,
) -> bool:
    """
    Override the correctness of the pending attempt before feedback is applied.
    Returns True if the pending attempt was updated.
    """
    attempt = user_state.pending_attempt
    if attempt is None:
        return False
    if attempt.question_id != question_id:
        return False
    attempt.correct = correct
    attempt.grade = 100.0 if correct else 0.0
    return True


def _clamp_difficulty(value: float) -> float:
    """Clamp target difficulty to [10, 100]. v0 floor/ceiling, calibrate empirically."""
    return max(10.0, min(100.0, value))
