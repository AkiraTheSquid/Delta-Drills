"""
practice_engine.py — Browser-loadable adaptive practice engine for Pyodide.

Self-contained Python module with:
  - Data structures (UserPracticeState, SubtopicState, AttemptRecord)
  - Core adaptive algorithm (compute_difficulty_multiplier, record_attempt, apply_feedback)
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
from math import exp
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEEDBACK_ALPHA = {
    "not_much": 0.3,
    "somewhat": 0.6,
    "a_lot": 0.85,
}

COLD_START_TARGETS = [25, 50, 75]

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


@dataclass
class UserPracticeState:
    user_id: str
    subtopic_states: Dict[str, SubtopicState] = field(default_factory=dict)
    pending_attempt: Optional[AttemptRecord] = None

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
        }
    return data


def state_from_dict(data: dict) -> UserPracticeState:
    state = UserPracticeState(user_id=data["user_id"])
    if data.get("pending_attempt"):
        pa = data["pending_attempt"]
        state.pending_attempt = AttemptRecord(
            question_id=pa["question_id"],
            subtopic=pa["subtopic"],
            difficulty_score=pa["difficulty_score"],
            grade=pa["grade"],
            correct=pa["correct"],
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
                feedback=a.get("feedback"),
                alpha=a.get("alpha"),
                score=a.get("score"),
                baseline_after=a.get("baseline_after"),
                p_after=a.get("p_after"),
                target_difficulty_after=a.get("target_difficulty_after"),
            ))
        state.subtopic_states[sub_name] = SubtopicState(
            subtopic=sub_data["subtopic"],
            n=sub_data["n"],
            baseline=sub_data["baseline"],
            p=sub_data["p"],
            target_difficulty=sub_data["target_difficulty"],
            served_question_ids=sub_data.get("served_question_ids", []),
            history=history,
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


def compute_difficulty_multiplier(p: float) -> float:
    if p <= 0.85:
        return 0.5 + 0.5 * (p / 0.85) ** 1.8
    else:
        return min(2.5, 1.0 + ((p - 0.85) / 0.15) ** 2.5)


def get_target_difficulty(state: SubtopicState) -> float:
    if state.n < len(COLD_START_TARGETS):
        return COLD_START_TARGETS[state.n]
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

    alpha = FEEDBACK_ALPHA[feedback]
    attempt.feedback = feedback
    attempt.alpha = alpha

    sub_state = user_state.get_subtopic_state(attempt.subtopic)
    sub_state.n += 1

    score = attempt.grade * attempt.difficulty_score / 100.0
    attempt.score = score

    if sub_state.n <= 3:
        if sub_state.n == 1:
            sub_state.baseline = score
            sub_state.p = 1.0 if attempt.grade > 85 else 0.0
        else:
            sub_state.baseline = alpha * score + (1 - alpha) * sub_state.baseline
            indicator = 1.0 if attempt.grade > 85 else 0.0
            sub_state.p = alpha * indicator + (1 - alpha) * sub_state.p

        if sub_state.n < len(COLD_START_TARGETS):
            sub_state.target_difficulty = COLD_START_TARGETS[sub_state.n]
        else:
            multiplier = compute_difficulty_multiplier(sub_state.p)
            sub_state.target_difficulty = _clamp_difficulty(sub_state.baseline * multiplier)
    else:
        sub_state.baseline = alpha * score + (1 - alpha) * sub_state.baseline
        indicator = 1.0 if attempt.grade > 85 else 0.0
        sub_state.p = alpha * indicator + (1 - alpha) * sub_state.p
        multiplier = compute_difficulty_multiplier(sub_state.p)
        sub_state.target_difficulty = _clamp_difficulty(sub_state.baseline * multiplier)

    attempt.baseline_after = sub_state.baseline
    attempt.p_after = sub_state.p
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

def _estimate_learning_rate(state: SubtopicState) -> float:
    history = state.history
    if len(history) < 2:
        return 0.5

    lambda_ = 0.3
    alpha = 1 - exp(-lambda_)

    rates = []
    for i in range(1, len(history)):
        curr = history[i]
        prev = history[i - 1]
        curr_perf = curr.baseline_after if curr.baseline_after is not None else 0.0
        prev_perf = prev.baseline_after if prev.baseline_after is not None else 0.0
        delta = curr_perf - prev_perf
        rates.append(delta)

    if not rates:
        return 0.5

    s = rates[0]
    for t in range(1, len(rates)):
        s = alpha * rates[t] + (1 - alpha) * s

    return s


def select_next_subtopic(user_state: UserPracticeState, questions: list) -> Optional[str]:
    """
    Select the subtopic to pull the next question from.
    questions: list of question dicts (from questions.json)
    """
    # Build subtopic -> question list mapping
    by_subtopic: Dict[str, list] = {}
    for q in questions:
        st = q.get("subtopic", "")
        if st:
            by_subtopic.setdefault(st, []).append(q)

    subtopics = sorted(by_subtopic.keys())
    if not subtopics:
        return None

    num_subtopics = len(subtopics)
    weight = 1.0 / num_subtopics

    gradients = []
    for st_name in subtopics:
        sub_state = user_state.get_subtopic_state(st_name)
        available = by_subtopic.get(st_name, [])
        served = set(sub_state.served_question_ids)
        remaining = [q for q in available if q["id"] not in served]
        if not remaining:
            continue

        learning_rate = _estimate_learning_rate(sub_state)
        gradient = weight * learning_rate
        gradients.append((st_name, gradient))

    if not gradients:
        # All questions served — reset and retry
        for st_name in subtopics:
            sub_state = user_state.get_subtopic_state(st_name)
            sub_state.served_question_ids.clear()
        for st_name in subtopics:
            sub_state = user_state.get_subtopic_state(st_name)
            available = by_subtopic.get(st_name, [])
            if available:
                learning_rate = _estimate_learning_rate(sub_state)
                gradients.append((st_name, weight * learning_rate))

    if not gradients:
        return None

    gradients.sort(key=lambda item: (-item[1], item[0]))
    return gradients[0][0]


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
    target = get_target_difficulty(sub_state)

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
