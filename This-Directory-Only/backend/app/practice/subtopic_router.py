"""
Subtopic-stats and weight-update endpoints.

Endpoints (mounted under /api/practice by the parent router):
  GET /state
  GET /subtopics
  PUT /weights
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import bkt_mastery
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User
from app.practice_schemas import (
    PracticeStateResponse,
    SelfReportRequest,
    SelfReportResponse,
    SubtopicStateSnapshot,
    SubtopicStatsResponse,
    WeightsUpdateRequest,
)
from app.prioritization import get_subtopic_weights
from app.questions import get_topic_for_subtopic

router = APIRouter()


def _unprefix_subtopic(full_key: str) -> str:
    """Strip "{topic}: " prefix from a backend subtopic key.

    Backend `questions.py` prefixes subtopics as `f"{topic}: {raw}"` to
    keep Numpy/Einsum subtopics distinct across the bank (questions.py:556).
    The frontend `questionsBank` (loaded from the static questions.json
    export) keeps subtopics RAW. The atom-readiness bridge in
    concept-graph/atom_readiness.js builds its topicIndex from
    `questionsBank` (raw), so wire-format state keys must also be raw —
    otherwise the alias-bridge lookup misses every subtopic.

    Falls back to the raw split if the topic lookup fails (e.g. legacy
    state for a subtopic no longer in the bank).
    """
    topic = get_topic_for_subtopic(full_key)
    prefix = f"{topic}: "
    if topic and full_key.startswith(prefix):
        return full_key[len(prefix):]
    if ": " in full_key:
        return full_key.split(": ", 1)[1]
    return full_key


@router.get("/state", response_model=PracticeStateResponse)
def get_practice_state(user: User = Depends(get_current_user)) -> PracticeStateResponse:
    """Return per-subtopic adaptive snapshot for the current user.

    Consumed by the frontend (practice/adaptive.js#loadBackendAdaptiveState)
    to hydrate `adaptiveStateJson` so concept-graph/atom_readiness.js can
    bridge atoms onto real EWMA baselines in backend mode (logged-in users
    skip the Pyodide engine, so without this endpoint the bridge sees an
    empty state and all atoms return the fallback readiness).

    Subtopic keys are unprefixed on egress to match the raw-subtopic
    format the frontend questionsBank uses. See `_unprefix_subtopic`.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    return PracticeStateResponse(
        user_id=user_state.user_id,
        subtopic_states={
            _unprefix_subtopic(name): SubtopicStateSnapshot(
                subtopic=_unprefix_subtopic(s.subtopic),
                n=s.n,
                baseline=s.baseline,
                p=s.p,
                target_difficulty=s.target_difficulty,
                last_update_ts=s.last_update_ts,
            )
            for name, s in user_state.subtopic_states.items()
        },
        custom_weights=dict(user_state.custom_weights or {}),
        atom_mastery=dict(user_state.atom_mastery or {}),
        atom_last_ts=dict(user_state.atom_last_ts or {}),
        self_reported_level=user_state.self_reported_level,
    )


@router.get("/subtopics", response_model=list[SubtopicStatsResponse])
def list_subtopics(user: User = Depends(get_current_user)) -> list[SubtopicStatsResponse]:
    """List subtopics with adaptive stats, sorted by gradient descending."""
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    weights_info = get_subtopic_weights(user_state)

    return [
        SubtopicStatsResponse(
            subtopic=info["subtopic"],
            topic=info["topic"],
            questions_answered=info["questions_answered"],
            current_difficulty=info["current_difficulty"],
            weight=info["weight"],
            learning_rate=info["learning_rate"],
            gradient=info["gradient"],
            baseline=info["baseline"],
            p=info["p"],
        )
        for info in weights_info
    ]


@router.put("/weights")
def update_weights(
    payload: WeightsUpdateRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """Persist custom per-subtopic effective weights for the user."""
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    user_state.custom_weights = {k: float(v) for k, v in payload.weights.items()}
    save_user_state(user_id)
    return {"ok": True}


@router.put("/self-report", response_model=SelfReportResponse)
def update_self_report(
    payload: SelfReportRequest,
    user: User = Depends(get_current_user),
) -> SelfReportResponse:
    """Persist the learner's self-reported experience level.

    Seeds the BKT prior for never-practiced atoms (bkt_mastery.PRIOR_BY_LEVEL)
    so the first questions start near the learner's level. It is a prior, not
    evidence: mastery/unlock gates ignore it, and the first few attempts
    overrule it. "default" (or anything unrecognized) clears it.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    level = payload.level.strip().lower()
    user_state.self_reported_level = level if level in bkt_mastery.PRIOR_BY_LEVEL else None
    save_user_state(user_id)
    return SelfReportResponse(success=True, level=user_state.self_reported_level)


@router.get("/atom-gates")
def atom_gates(user: User = Depends(get_current_user)) -> dict:
    """Unified per-atom unlock state for the whole concept graph.

    Single source of truth for the frontend drill/ARENA gates (the frontend
    ships a stale v2 graph, so prereq edges must come from the backend's v3).
      - ready:    atoms whose gating prerequisites are all mastered (>= 0.85) —
                  a single-atom teaching item unlocks when its atom is ready.
      - mastered: atoms whose own posterior is >= 0.85 — a composite/ARENA item
                  unlocks when ALL its component atoms are mastered.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    ready, mastered = bkt_mastery.gate_sets(
        user_state.atom_mastery, user_state.atom_last_ts
    )
    return {
        "ready": ready,
        "mastered": mastered,
        "threshold": bkt_mastery.UNLOCK_THRESHOLD,
    }
