"""
practice_engine.py — Browser-loadable adaptive practice engine for Pyodide.

Self-contained Python module with:
  - Data structures (UserPracticeState, SubtopicState, AttemptRecord)
  - Offline free-practice engine (record_attempt, apply_feedback, selection)
  - Subtopic prioritization (select_next_subtopic)
  - Question selection (pick_question)
  - JSON serialization for JS interop

No file I/O, no FastAPI imports. Designed to run in Pyodide in the browser.
All external communication is via JSON strings.

Usage from JS:
  pyodide.runPython(practiceEngineSource);
  const engine = pyodide.globals.get("engine_api");
  const state = engine.init_state("user@example.com");
  const nextQ = engine.next_question(state, questionsJsonString);
  ...
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants — v0 calibration, not literature-derived. Mirror of backend
# adaptive.py; see papers/MASTERY_ESTIMATION_REFERENCE_v2.md.
# ---------------------------------------------------------------------------

FEEDBACK_ALPHA = {
    "not_much": 0.3,
    "somewhat": 0.6,
    "a_lot": 0.85,
}

COLD_START_TARGETS = [25, 50, 75]

# Staircase difficulty (offline mode) — a simple up/down step model seeded by
# the learner's self-reported level. Correct answer steps the target up, a
# wrong one steps it back down harder, so strong learners climb out of the
# easy band in a few questions and beginners never get thrown into the deep
# end. Backend mode uses the BKT-driven target instead; this only governs
# offline/Supabase free practice, which previously served a FIXED 25/50/75
# ramp with no adaptivity at all.
STAIRCASE_SEED_BY_LEVEL = {
    "beginner": 15.0,
    "strong": 70.0,
}
STAIRCASE_DEFAULT_SEED = 40.0
STAIRCASE_STEP_UP = 15.0      # on a correct answer
STAIRCASE_STEP_DOWN = 20.0    # on a wrong answer (down harder: wrong at level
                              # k is stronger evidence than right at level k)

# Half-life decay (v0) — Yudelson & Pavlik 2013 (survived audit) flags
# monotonic mastery as anti-pattern; this is the regression mechanism.
HALF_LIFE_DAYS: float = 14.0
P_PRIOR: float = 0.5
BASELINE_PRIOR: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


# EWMA decay/blend removed — offline is ungated free practice; adaptive mastery
# (BKT + forgetting) is backend-only. SubtopicState.baseline/p remain as inert
# serialization fields for state round-trip compatibility.

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    question_id: int
    subtopic: str
    difficulty_score: int
    grade: float          # 0 or 100
    correct: bool
    timestamp: str = ""   # ISO-8601 UTC; backfilled to _now_iso() on record
    feedback: Optional[str] = None
    alpha: Optional[float] = None
    score: Optional[float] = None
    baseline_after: Optional[float] = None
    p_after: Optional[float] = None
    target_difficulty_after: Optional[float] = None


@dataclass
class SubtopicState:
    subtopic: str
    n: int = 0
    baseline: float = 0.0
    p: float = 0.5
    target_difficulty: float = 25.0
    history: List[AttemptRecord] = field(default_factory=list)
    served_question_ids: List[int] = field(default_factory=list)
    last_update_ts: Optional[str] = None
    # True once target_difficulty has been seeded from the self-reported
    # level — from then on the staircase owns it.
    staircase_seeded: bool = False


@dataclass
class UserPracticeState:
    user_id: str
    custom_weights: Dict[str, float] = field(default_factory=dict)
    subtopic_states: Dict[str, SubtopicState] = field(default_factory=dict)
    pending_attempt: Optional[AttemptRecord] = None
    # Per-atom BKT mastery posterior P(known) ∈ [0, 1], keyed by atom id —
    # the prioritization/readiness signal. Mirror of backend adaptive.py;
    # see bkt_mastery.py for the model. EWMA above is now only an area readout.
    atom_mastery: Dict[str, float] = field(default_factory=dict)
    atom_last_ts: Dict[str, str] = field(default_factory=dict)
    # Self-reported experience: None | "beginner" | "strong". Seeds each
    # subtopic's staircase start; the staircase corrects it from question 1.
    self_reported_level: Optional[str] = None

    def get_subtopic_state(self, subtopic: str) -> SubtopicState:
        if subtopic not in self.subtopic_states:
            self.subtopic_states[subtopic] = SubtopicState(subtopic=subtopic)
        return self.subtopic_states[subtopic]


# ---------------------------------------------------------------------------
# Serialization (to/from JSON for JS interop and Supabase persistence)
# ---------------------------------------------------------------------------

def state_to_dict(state: UserPracticeState) -> dict:
    data = {
        "user_id": state.user_id,
        "custom_weights": state.custom_weights,
        "atom_mastery": state.atom_mastery,
        "atom_last_ts": state.atom_last_ts,
        "self_reported_level": state.self_reported_level,
        "pending_attempt": asdict(state.pending_attempt) if state.pending_attempt else None,
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
            "staircase_seeded": sub_state.staircase_seeded,
        }
    return data


def state_from_dict(data: dict) -> UserPracticeState:
    state = UserPracticeState(user_id=data["user_id"])
    state.custom_weights = data.get("custom_weights") or {}
    # Additive, back-compat: older saves predate per-atom BKT state.
    state.atom_mastery = data.get("atom_mastery") or {}
    state.atom_last_ts = data.get("atom_last_ts") or {}
    state.self_reported_level = data.get("self_reported_level")
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
        # Backfill last_update_ts from last history entry for migrated saves.
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
            # Migrated saves predate the staircase: treat a subtopic with
            # attempts as already seeded so its target isn't reset mid-run.
            staircase_seeded=sub_data.get("staircase_seeded", sub_data["n"] > 0),
        )
    return state


def state_to_json(state: UserPracticeState) -> str:
    return json.dumps(state_to_dict(state))


def state_from_json(json_str: str) -> UserPracticeState:
    return state_from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Core adaptive algorithm
# ---------------------------------------------------------------------------

def _clamp_difficulty(value: float) -> float:
    return max(10.0, min(100.0, value))


def get_target_difficulty(state: SubtopicState, level: Optional[str] = None) -> float:
    """OFFLINE free-practice difficulty: a per-subtopic STAIRCASE.

    Seeded from the self-reported level (beginner 15 / default 40 / strong 70),
    then stepped by apply_feedback: +STAIRCASE_STEP_UP on correct,
    -STAIRCASE_STEP_DOWN on wrong. Replaces the old FIXED [25, 50, 75] ramp,
    which was predetermined regardless of performance. The BKT-driven target
    lives only in backend mode (prioritization.target_difficulty)."""
    if not state.staircase_seeded:
        state.target_difficulty = STAIRCASE_SEED_BY_LEVEL.get(
            level or "", STAIRCASE_DEFAULT_SEED
        )
        state.staircase_seeded = True
    return _clamp_difficulty(state.target_difficulty)


def step_staircase(state: SubtopicState, correct: bool, level: Optional[str] = None) -> float:
    """Advance the staircase after a graded attempt and return the new target."""
    current = get_target_difficulty(state, level)  # seeds if needed
    delta = STAIRCASE_STEP_UP if correct else -STAIRCASE_STEP_DOWN
    state.target_difficulty = _clamp_difficulty(current + delta)
    return state.target_difficulty


def record_attempt(
    user_state: UserPracticeState,
    question_id: int,
    subtopic: str,
    difficulty_score: int,
    correct: bool,
) -> AttemptRecord:
    grade = 100.0 if correct else 0.0
    attempt = AttemptRecord(
        question_id=question_id,
        subtopic=subtopic,
        difficulty_score=difficulty_score,
        grade=grade,
        correct=correct,
        timestamp=_now_iso(),
    )
    user_state.pending_attempt = attempt
    return attempt


def apply_feedback(
    user_state: UserPracticeState,
    feedback: str,
) -> Optional[AttemptRecord]:
    attempt = user_state.pending_attempt
    if attempt is None:
        return None

    # OFFLINE free practice: record the attempt; no EWMA mastery blend. The
    # `feedback` level is retained on the record for history but no longer drives
    # any score (adaptivity/gating is backend-only now).
    attempt.feedback = feedback
    attempt.alpha = FEEDBACK_ALPHA.get(feedback)

    sub_state = user_state.get_subtopic_state(attempt.subtopic)
    sub_state.n += 1
    if not attempt.timestamp:
        attempt.timestamp = _now_iso()
    sub_state.last_update_ts = attempt.timestamp
    step_staircase(sub_state, attempt.correct, user_state.self_reported_level)
    attempt.target_difficulty_after = sub_state.target_difficulty

    sub_state.history.append(attempt)
    user_state.pending_attempt = None

    return attempt


def override_pending_attempt(
    user_state: UserPracticeState,
    question_id: int,
    correct: bool = True,
) -> bool:
    attempt = user_state.pending_attempt
    if attempt is None:
        return False
    if attempt.question_id != question_id:
        return False
    attempt.correct = correct
    attempt.grade = 100.0 if correct else 0.0
    return True


# ---------------------------------------------------------------------------
# Subtopic prioritization (from prioritization.py)
# ---------------------------------------------------------------------------

def select_next_subtopic(user_state: UserPracticeState, questions: list) -> Optional[str]:
    """OFFLINE free practice: pick the LEAST-served subtopic that still has an
    unserved question (respecting custom_weights>0), resetting served sets when
    everything is exhausted. No EWMA learning-rate gradient — adaptive ordering
    is backend-only (prioritization.select_next_subtopic)."""
    by_subtopic: Dict[str, list] = {}
    for q in questions:
        st = q.get("subtopic", "")
        if st:
            by_subtopic.setdefault(st, []).append(q)

    subtopics = sorted(by_subtopic.keys())
    if not subtopics:
        return None
    uniform = 1.0 / len(subtopics)

    def _eligible(skip_served: bool):
        out = []
        for st_name in subtopics:
            if user_state.custom_weights.get(st_name, uniform) <= 0:
                continue
            sub_state = user_state.get_subtopic_state(st_name)
            available = by_subtopic.get(st_name, [])
            if not available:
                continue
            if skip_served:
                served = set(sub_state.served_question_ids)
                if not [q for q in available if q["id"] not in served]:
                    continue
            out.append((st_name, sub_state.n))
        return out

    cands = _eligible(skip_served=True)
    if not cands:
        for st_name in subtopics:
            user_state.get_subtopic_state(st_name).served_question_ids.clear()
        cands = _eligible(skip_served=False)
    if not cands:
        return None
    # least-served first; alpha tiebreak for determinism
    cands.sort(key=lambda item: (item[1], item[0]))
    return cands[0][0]


# ---------------------------------------------------------------------------
# Question selection
# ---------------------------------------------------------------------------

def pick_question(user_state: UserPracticeState, questions: list) -> Optional[dict]:
    """
    Pick the next question using adaptive subtopic selection + difficulty targeting.
    questions: list of question dicts from questions.json.
    Returns a question dict or None.
    """
    subtopic = select_next_subtopic(user_state, questions)
    if subtopic is None:
        return None

    sub_state = user_state.get_subtopic_state(subtopic)
    target = get_target_difficulty(sub_state, user_state.self_reported_level)

    # Filter to this subtopic, excluding already-served
    served = set(sub_state.served_question_ids)
    candidates = [q for q in questions if q["subtopic"] == subtopic and q["id"] not in served]

    if not candidates:
        # Shouldn't happen (select_next_subtopic checks), but fallback
        candidates = [q for q in questions if q["subtopic"] == subtopic]

    if not candidates:
        return None

    # Pick closest to target difficulty
    candidates.sort(key=lambda q: abs(q["difficulty_score"] - target))
    chosen = candidates[0]

    # Mark as served
    sub_state.served_question_ids.append(chosen["id"])

    return chosen


# ---------------------------------------------------------------------------
# Public API for JS interop — all functions take/return JSON strings
# ---------------------------------------------------------------------------

class EngineAPI:
    """Stateless API for JS. All state is passed in/out as JSON."""

    def init_state(self, user_id: str) -> str:
        """Create a fresh user state. Returns JSON string."""
        state = UserPracticeState(user_id=user_id)
        return state_to_json(state)

    def next_question(self, state_json: str, questions_json: str) -> str:
        """
        Pick the next question given current state and full question bank.
        Returns JSON: {question: {...}, state: "..."}
        """
        state = state_from_json(state_json)
        questions = json.loads(questions_json)
        q = pick_question(state, questions)
        return json.dumps({
            "question": q,
            "state": state_to_json(state),
        })

    def set_custom_weights(self, state_json: str, weights_json: str) -> str:
        state = state_from_json(state_json)
        state.custom_weights = json.loads(weights_json)
        return state_to_json(state)

    def set_self_reported_level(self, state_json: str, level: str) -> str:
        """Set the self-reported experience level ("beginner" | "strong";
        anything else clears it). Re-seeds the staircase for subtopics with
        no attempts yet; subtopics already in progress keep their position."""
        state = state_from_json(state_json)
        normalized = (level or "").strip().lower()
        state.self_reported_level = (
            normalized if normalized in STAIRCASE_SEED_BY_LEVEL else None
        )
        for sub_state in state.subtopic_states.values():
            if sub_state.n == 0:
                sub_state.staircase_seeded = False
        return state_to_json(state)

    def submit_answer(self, state_json: str, question_id: int, subtopic: str,
                      difficulty_score: int, correct: bool) -> str:
        """
        Record an attempt (before feedback). Returns updated state JSON.
        """
        state = state_from_json(state_json)
        record_attempt(state, question_id, subtopic, difficulty_score, correct)
        return state_to_json(state)

    def send_feedback(self, state_json: str, feedback: str) -> str:
        """
        Apply feedback to pending attempt. Returns updated state JSON.
        """
        state = state_from_json(state_json)
        apply_feedback(state, feedback)
        return state_to_json(state)

    def override_attempt(self, state_json: str, question_id: int, correct: bool = True) -> str:
        """
        Override the pending attempt correctness before feedback.
        """
        state = state_from_json(state_json)
        override_pending_attempt(state, question_id, correct)
        return state_to_json(state)


# Create singleton for JS access
engine_api = EngineAPI()
