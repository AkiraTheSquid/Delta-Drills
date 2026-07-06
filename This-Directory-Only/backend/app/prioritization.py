"""
Subtopic prioritization module — BKT-driven (EWMA fully removed).

Decides which subtopic to pull the next question from, and at what difficulty,
using ONLY the per-atom Bayesian Knowledge Tracing posteriors (bkt_mastery.py).
A subtopic's mastery is the mean BKT posterior over the atoms its questions
exercise (see questions.get_atoms_for_subtopic — populated from the per-question
atom tags). The old per-subtopic EWMA gradient/learning-rate is gone.

Selection policy: WEAKEST-FIRST. priority = effective_weight * (1 - mastery).
Un-practiced atoms sit at the BKT prior (~0.10), so fresh subtopics surface
first naturally; decay regresses mastery over time, resurfacing stale ones
without a separate staleness rule.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app import bkt_mastery
from app.adaptive import UserPracticeState
from app.questions import (
    get_atoms_for_subtopic,
    get_subtopics,
    get_questions_by_subtopic,
    get_topic_for_subtopic,
)

logger = logging.getLogger(__name__)

# Difficulty target maps mastery∈[0,1] → numeric difficulty. Low mastery serves
# easy items; near-mastery serves the hardest. v0 affine map, tune empirically.
_DIFF_FLOOR = 20.0
_DIFF_SPAN = 80.0


def _get_weight(user_state: UserPracticeState, st_name: str, uniform_weight: float) -> float:
    """Return the effective weight for a subtopic, using custom weights if set."""
    if user_state.custom_weights:
        return user_state.custom_weights.get(st_name, uniform_weight)
    return uniform_weight


def question_is_unlocked(user_state: UserPracticeState, question) -> bool:
    """Unified per-atom PREREQUISITE gate for a bank question: servable iff every
    atom it is tagged with is READY (all that atom's gating prerequisites are
    mastered ≥ UNLOCK_THRESHOLD). Untagged questions are ungated. Root-atom
    questions are servable from the start (no prereqs) — the cold-start entry
    points; composite-atom questions (e.g. conv2d-module) stay locked until their
    prereq atoms are mastered."""
    tags = getattr(question, "atom_tags", None) or []
    if not tags:
        return True
    return all(
        bkt_mastery.atom_is_ready(
            t["atom_id"], user_state.atom_mastery, user_state.atom_last_ts
        )
        for t in tags
    )


def subtopic_mastery(user_state: UserPracticeState, subtopic: str) -> float:
    """Mean decay-adjusted BKT posterior over the atoms this subtopic exercises.
    Falls back to the learner's prior (self-reported level, else the BKT
    default) when the subtopic has no tagged atoms / no practice yet — so a
    self-reported beginner starts at the floor and a self-reported strong
    learner starts high, and evidence takes over from the first attempt."""
    params = bkt_mastery.params_for_level(user_state.self_reported_level)
    atoms = get_atoms_for_subtopic(subtopic)
    if not atoms:
        return params.p_init
    vals = [
        bkt_mastery.current_mastery(
            user_state.atom_mastery, user_state.atom_last_ts, a, params=params
        )
        for a in atoms
    ]
    return sum(vals) / len(vals) if vals else params.p_init


def target_difficulty(user_state: UserPracticeState, subtopic: str) -> float:
    """BKT-derived target difficulty for the next question in a subtopic.
    Scales with the learner's mastery of the subtopic's atoms."""
    m = subtopic_mastery(user_state, subtopic)
    raw = _DIFF_FLOOR + _DIFF_SPAN * m
    return max(10.0, min(100.0, raw))


def _difficulty_reach_factor(easiest_available: float, target: float) -> float:
    """Down-weight subtopics whose EASIEST unserved question sits far above the
    learner's target difficulty. Weakest-first alone is perverse for a
    self-reported beginner: every untouched subtopic ties at the prior, so the
    queue happily opens with 'the easiest matmul-backward drill' — which is
    still an advanced problem (tester hit exactly this). A subtopic is only
    servable-at-level if it actually HAS a question near the target; the
    factor fades as mastery (and thus target) rises, so advanced subtopics
    re-enter naturally. 1.0 when reachable, ~0.5 one 8-point gap out."""
    gap = max(0.0, easiest_available - target)
    return 1.0 / (1.0 + gap / 8.0)


def select_next_subtopic(user_state: UserPracticeState) -> Optional[str]:
    """Select the subtopic to pull the next question from — weakest-first by
    BKT mastery, weighted by effective (custom) weight and by whether the
    subtopic has questions reachable at the learner's target difficulty.
    Skips subtopics whose questions are all served; resets served sets if
    everything is exhausted.
    """
    subtopics = get_subtopics()
    if not subtopics:
        return None
    uniform_weight = 1.0 / len(subtopics)

    def _candidates(skip_served: bool) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for st_name in subtopics:
            available = [
                q for q in get_questions_by_subtopic(st_name)
                if question_is_unlocked(user_state, q)
            ]
            if not available:
                continue
            if skip_served:
                served = set(user_state.get_subtopic_state(st_name).served_question_ids)
                available = [q for q in available if q.id not in served]
                if not available:
                    continue
            weight = _get_weight(user_state, st_name, uniform_weight)
            if weight <= 0:
                continue
            easiest = min(q.difficulty_score for q in available)
            reach = _difficulty_reach_factor(easiest, target_difficulty(user_state, st_name))
            priority = weight * (1.0 - subtopic_mastery(user_state, st_name)) * reach
            out.append((st_name, priority, easiest))
        return out

    cands = _candidates(skip_served=True)
    if not cands:
        # Everything served — reset and retry over the full set.
        for st_name in subtopics:
            user_state.get_subtopic_state(st_name).served_question_ids.clear()
        cands = _candidates(skip_served=False)
    if not cands:
        return None

    # Highest priority (weakest, weighted, reachable) first. Ties — common on
    # fresh accounts where every subtopic sits at the same prior — resolve to
    # the subtopic with the genuinely easiest entry question, THEN alpha (the
    # old alpha-first tiebreak opened beginners on "CNN: Conv2d mechanics").
    cands.sort(key=lambda item: (-item[1], item[2], item[0]))
    return cands[0][0]


def get_subtopic_weights(user_state: UserPracticeState) -> List[Dict]:
    """All subtopics with current prioritization info, sorted weakest-first.

    `baseline`/`p` now carry the BKT subtopic mastery (0-100 and 0-1) so the
    existing frontend score readers (getArenaPrereqSubtopicScore) and the area
    readout reflect BKT, not EWMA. `gradient` is the selection priority.
    """
    subtopics = get_subtopics()
    uniform_weight = 1.0 / len(subtopics) if subtopics else 1.0

    result = []
    for st_name in subtopics:
        sub_state = user_state.get_subtopic_state(st_name)
        weight = _get_weight(user_state, st_name, uniform_weight)
        mastery = subtopic_mastery(user_state, st_name)
        priority = weight * (1.0 - mastery)
        result.append({
            "subtopic": st_name,
            "topic": get_topic_for_subtopic(st_name),
            "weight": weight,
            "learning_rate": priority,          # repurposed: BKT selection priority
            "gradient": priority,
            "questions_answered": sub_state.n,
            "current_difficulty": target_difficulty(user_state, st_name),
            "baseline": mastery * 100.0,        # BKT mastery on the 0-100 scale
            "p": mastery,                       # BKT mastery 0-1 (frontend score)
        })

    return sorted(result, key=lambda r: r["gradient"], reverse=True)
