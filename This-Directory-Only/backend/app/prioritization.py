"""
Subtopic prioritization module.

Decides which subtopic to pull the next question from, inspired by the
gradient-based prioritization in the example code.

Each subtopic has:
  - weight  (uniform 1/num_subtopics for MVP; easy to override later)
  - learning_rate  estimated via EWMA over recent performance changes
  - gradient = weight * learning_rate

Higher gradient => higher priority => that subtopic is selected next.

"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from math import exp
from typing import Dict, List, Optional, Tuple

from app.adaptive import (
    SubtopicState,
    UserPracticeState,
    COLD_START_TARGETS,
    HALF_LIFE_DAYS,
    _parse_ts,
)
from app.questions import get_subtopics, get_questions_by_subtopic, get_topic_for_subtopic

logger = logging.getLogger(__name__)

# --- v0 calibration constants — no literature source, tune empirically ---

# Subtopics with fewer answered questions than this threshold get a boosted
# learning-rate so they are always selected before topics with real history.
# Matches the difficulty cold-start window in adaptive.py (len(COLD_START_TARGETS) = 3).
COLD_START_MIN_QUESTIONS: int = len(COLD_START_TARGETS)
# Must exceed any realistic EWMA learning rate (scores are 0-100, so deltas
# are bounded by [-100, 100]; 200 is safely above that ceiling).
COLD_START_PRIORITY_LR: float = 200.0
# A subtopic is "stale" once last_update_ts is older than half a half-life;
# stale subtopics get the cold-start priority boost so decay-driven regression
# doesn't trap them in low-gradient limbo.
STALENESS_DAYS: float = HALF_LIFE_DAYS / 2.0

def _is_stale(state: SubtopicState, now: Optional[datetime] = None) -> bool:
    """True if last_update_ts is older than STALENESS_DAYS."""
    prev = _parse_ts(state.last_update_ts)
    if prev is None:
        return False
    now = now or datetime.now(timezone.utc)
    elapsed_days = (now - prev).total_seconds() / 86400.0
    return elapsed_days >= STALENESS_DAYS


def _estimate_learning_rate(state: SubtopicState) -> float:
    """
    EWMA learning-rate estimate for a single subtopic.
    Similar to the prioritization example: computes rate of performance change
    over recent attempts.

    Returns a float. Higher = more improvement happening.
    Subtopics with no history return a moderate default (0.5).
    Stale subtopics (last_update_ts > STALENESS_DAYS old) get the cold-start
    boost so decay-driven regression resurfaces them for refresh.
    """
    # Cold-start: prioritise unexplored topics above any with real history.
    if state.n < COLD_START_MIN_QUESTIONS:
        return COLD_START_PRIORITY_LR

    # Staleness boost — forgotten subtopics should resurface.
    if _is_stale(state):
        return COLD_START_PRIORITY_LR

    history = state.history
    if len(history) < 2:
        return COLD_START_PRIORITY_LR

    # v0 EWMA rate constant (no literature source) — controls how quickly the
    # learning-rate estimate forgets old deltas. Tune empirically.
    lambda_ = 0.3
    alpha = 1 - exp(-lambda_)

    rates: List[float] = []
    for i in range(1, len(history)):
        curr = history[i]
        prev = history[i - 1]
        # "performance" is the score after each attempt
        curr_perf = curr.baseline_after if curr.baseline_after is not None else 0.0
        prev_perf = prev.baseline_after if prev.baseline_after is not None else 0.0
        delta = curr_perf - prev_perf
        # We use absolute improvement per step (not per hour since attempts are sequential)
        rates.append(delta)

    if not rates:
        return 0.5

    # EWMA
    s = rates[0]
    for t in range(1, len(rates)):
        s = alpha * rates[t] + (1 - alpha) * s

    return s


def _get_weight(user_state: UserPracticeState, st_name: str, uniform_weight: float) -> float:
    """Return the effective weight for a subtopic, using custom weights if set."""
    if user_state.custom_weights:
        return user_state.custom_weights.get(st_name, uniform_weight)
    return uniform_weight


def select_next_subtopic(user_state: UserPracticeState) -> Optional[str]:
    """
    Select the subtopic from which to pull the next question.

    Uses gradient = weight * learning_rate and selects the max.
    If custom_weights are set on the user state, they are used instead of
    uniform weights.

    Returns the subtopic name, or None if there are no subtopics available.
    """
    subtopics = get_subtopics()
    if not subtopics:
        return None

    num_subtopics = len(subtopics)
    uniform_weight = 1.0 / num_subtopics

    # Compute gradients
    gradients: List[Tuple[str, float]] = []
    for st_name in subtopics:
        sub_state = user_state.get_subtopic_state(st_name)

        # Skip subtopics where the user has answered all available questions
        available = get_questions_by_subtopic(st_name)
        served = set(sub_state.served_question_ids)
        remaining = [q for q in available if q.id not in served]
        if not remaining:
            continue

        weight = _get_weight(user_state, st_name, uniform_weight)
        if weight <= 0:
            continue
        learning_rate = _estimate_learning_rate(sub_state)
        gradient = weight * learning_rate
        gradients.append((st_name, gradient))

    if not gradients:
        for st_name in subtopics:
            sub_state = user_state.get_subtopic_state(st_name)
            sub_state.served_question_ids.clear()
        for st_name in subtopics:
            sub_state = user_state.get_subtopic_state(st_name)
            available = get_questions_by_subtopic(st_name)
            if available:
                weight = _get_weight(user_state, st_name, uniform_weight)
                if weight <= 0:
                    continue
                learning_rate = _estimate_learning_rate(sub_state)
                gradients.append((st_name, weight * learning_rate))

    if not gradients:
        return None

    gradients.sort(key=lambda item: (-item[1], item[0]))
    return gradients[0][0]


def get_subtopic_weights(user_state: UserPracticeState) -> List[Dict]:
    """
    Return all subtopics with their current prioritization info.
    Useful for debugging or displaying to the user.
    """
    subtopics = get_subtopics()
    num_subtopics = len(subtopics) if subtopics else 1
    uniform_weight = 1.0 / num_subtopics

    result = []
    for st_name in subtopics:
        sub_state = user_state.get_subtopic_state(st_name)
        weight = _get_weight(user_state, st_name, uniform_weight)
        learning_rate = _estimate_learning_rate(sub_state)
        gradient = weight * learning_rate
        result.append({
            "subtopic": st_name,
            "topic": get_topic_for_subtopic(st_name),
            "weight": weight,
            "learning_rate": learning_rate,
            "gradient": gradient,
            "questions_answered": sub_state.n,
            "current_difficulty": sub_state.target_difficulty,
            "baseline": sub_state.baseline,
            "p": sub_state.p,
        })

    return sorted(result, key=lambda r: r["gradient"], reverse=True)
