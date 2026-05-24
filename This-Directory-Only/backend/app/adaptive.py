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

FeedbackLevel = Literal["not_much", "somewhat", "a_lot"]

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


def _decay_factor(last_ts: Optional[str], now: datetime, half_life_days: float = HALF_LIFE_DAYS) -> float:
    """Compute multiplicative decay factor in [0, 1] for elapsed time."""
    prev = _parse_ts(last_ts)
    if prev is None or half_life_days <= 0:
        return 1.0
    elapsed_days = max(0.0, (now - prev).total_seconds() / 86400.0)
    return 0.5 ** (elapsed_days / half_life_days)


def decayed_view(state: SubtopicState, now: Optional[datetime] = None) -> Tuple[float, float]:
    """Return (baseline, p) decayed to `now` without mutating state.

    Used by get_target_difficulty so served difficulty reflects current
    (decayed) estimate. Mirrors the apply-side decay so a read between
    attempts agrees with the next write.
    """
    now = now or datetime.now(timezone.utc)
    factor = _decay_factor(state.last_update_ts, now)
    baseline = BASELINE_PRIOR + (state.baseline - BASELINE_PRIOR) * factor
    p = P_PRIOR + (state.p - P_PRIOR) * factor
    return baseline, p


def _apply_decay(state: SubtopicState, now: datetime) -> None:
    """Mutate state to decayed values; called before each EWMA update."""
    baseline, p = decayed_view(state, now)
    state.baseline = baseline
    state.p = p


def compute_difficulty_multiplier(p: float) -> float:
    """
    Compute the difficulty multiplier from the correctness rate p.

    Piecewise curve, v0 parameters (not literature-grounded):
      breakpoint 0.85, sub-breakpoint exponent 1.8, super exponent 2.5,
      hard cap at 2.5×. Re-tune empirically.
    """
    if p <= 0.85:
        return 0.5 + 0.5 * (p / 0.85) ** 1.8
    else:
        return min(2.5, 1.0 + ((p - 0.85) / 0.15) ** 2.5)


def get_target_difficulty(state: SubtopicState) -> float:
    """
    Return the target difficulty for the next question in this subtopic.
    During cold start (n <= 3), returns predefined targets.
    After that, returns the difficulty computed from the *decayed* baseline
    and p (so stale subtopics regress toward easier difficulties).
    """
    if state.n < len(COLD_START_TARGETS):
        return COLD_START_TARGETS[state.n]
    baseline, p = decayed_view(state)
    return _clamp_difficulty(baseline * compute_difficulty_multiplier(p))


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

    alpha = FEEDBACK_ALPHA[feedback]
    attempt.feedback = feedback
    attempt.alpha = alpha

    sub_state = user_state.get_subtopic_state(attempt.subtopic)
    sub_state.n += 1

    # Decay stored baseline/p by time elapsed since last update BEFORE the EWMA
    # blend. Ensures a returning user's first attempt blends against a
    # forgetting-adjusted prior rather than a stale snapshot.
    now = _parse_ts(attempt.timestamp) or datetime.now(timezone.utc)
    _apply_decay(sub_state, now)

    # Compute score
    score = attempt.grade * attempt.difficulty_score / 100.0
    attempt.score = score

    if sub_state.n <= 3:
        # During cold start, still accumulate baseline and p but use simple averages
        if sub_state.n == 1:
            sub_state.baseline = score
            sub_state.p = 1.0 if attempt.grade > 85 else 0.0
        else:
            sub_state.baseline = alpha * score + (1 - alpha) * sub_state.baseline
            indicator = 1.0 if attempt.grade > 85 else 0.0
            sub_state.p = P_ALPHA * indicator + (1 - P_ALPHA) * sub_state.p

        # Set target difficulty for next cold start question or transition
        if sub_state.n < len(COLD_START_TARGETS):
            sub_state.target_difficulty = COLD_START_TARGETS[sub_state.n]
        else:
            # Transitioning out of cold start
            multiplier = compute_difficulty_multiplier(sub_state.p)
            sub_state.target_difficulty = _clamp_difficulty(sub_state.baseline * multiplier)
    else:
        # Full algorithm for n > 3
        sub_state.baseline = alpha * score + (1 - alpha) * sub_state.baseline
        indicator = 1.0 if attempt.grade > 85 else 0.0
        sub_state.p = P_ALPHA * indicator + (1 - P_ALPHA) * sub_state.p
        multiplier = compute_difficulty_multiplier(sub_state.p)
        sub_state.target_difficulty = _clamp_difficulty(sub_state.baseline * multiplier)

    sub_state.last_update_ts = attempt.timestamp

    attempt.baseline_after = sub_state.baseline
    attempt.p_after = sub_state.p
    attempt.target_difficulty_after = sub_state.target_difficulty

    sub_state.history.append(attempt)
    user_state.pending_attempt = None

    logger.info(
        "Feedback applied: user=%s subtopic=%s n=%d baseline=%.2f p=%.3f target=%.1f",
        user_state.user_id,
        attempt.subtopic,
        sub_state.n,
        sub_state.baseline,
        sub_state.p,
        sub_state.target_difficulty,
    )

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
