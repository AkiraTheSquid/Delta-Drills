"""
First-encounter exposure endpoints (lesson gate, Pass 2) and the ladder's
worked-example acknowledgement.

Endpoints (mounted under /api/practice by the parent router):
  GET  /exposure     — the learner's kc -> first-exposure timestamp map
  POST /exposure     — record introducing-KP completion for one or more KCs
  GET  /kc-estimate  — one concept's rung + interval, with no question attached
  POST /worked-seen  — record that a worked example was read for one KC

Exposure only ever accumulates; there is no unexpose (re-reading a lesson
is always allowed client-side, but the gate never re-arms).

`/worked-seen` lives here rather than in `questions_router` for two reasons:
that module is already ORANGE on the structural score, and this is the same
kind of "the learner read the teaching material" signal `/exposure` records.
The two are deliberately separate counters — exposure fires once, ever, and
gates a concept's FIRST question; worked_seen counts every time the teaching
page was actually read, which the ladder reads as "this concept has been
taught at least once" (`kc_graph._stage_from`). It can still be incremented
more than once, because a learner may re-open a lesson from the concept graph
or `?lesson=<kc>` — but the scheduler no longer forces that on them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from app import kc_graph, lessons
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User
from app.practice_schemas import (
    ExposureMarkRequest,
    ExposureResponse,
    KcEstimateResponse,
    WorkedSeenRequest,
    WorkedSeenResponse,
)
from app.prioritization import ladder_starter
from app.questions import get_question_by_id

router = APIRouter()


@router.get("/exposure", response_model=ExposureResponse)
def get_exposure(user: User = Depends(get_current_user)) -> ExposureResponse:
    user_state = get_user_state(str(user.id))
    return ExposureResponse(exposed=dict(user_state.kc_exposure))


@router.post("/exposure", response_model=ExposureResponse)
def mark_exposure(
    payload: ExposureMarkRequest,
    user: User = Depends(get_current_user),
) -> ExposureResponse:
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    now = datetime.now(timezone.utc).isoformat()
    changed = False
    for kc in payload.kcs:
        # Unknown KC ids are dropped silently — a stale/renamed KC in an old
        # client must not pollute the exposure map forever.
        if kc not in user_state.kc_exposure and lessons.exposure_key_exists(kc):
            user_state.kc_exposure[kc] = now
            changed = True
    if changed:
        save_user_state(user_id)
    return ExposureResponse(exposed=dict(user_state.kc_exposure))


@router.get("/kc-estimate", response_model=KcEstimateResponse)
def get_kc_estimate(
    kc: str,
    user: User = Depends(get_current_user),
) -> KcEstimateResponse:
    """Where one concept stands right now — no question involved.

    Read-only, and deliberately so: it must be safe for the topbar to call on
    every lesson page turn. In particular it does NOT call `note_worked_seen`,
    which would promote a concept merely for having been looked at.
    """
    user_state = get_user_state(str(user.id))
    if not kc_graph.registry_node(kc):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown concept '{kc}'",
        )
    return KcEstimateResponse(
        kc=kc,
        ladder_stage=kc_graph.kc_stage(user_state, kc),
        ladder_estimate=kc_graph.kc_estimate(user_state, kc),
    )


@router.post("/worked-seen", response_model=WorkedSeenResponse)
def mark_worked_seen(
    payload: WorkedSeenRequest,
    user: User = Depends(get_current_user),
) -> WorkedSeenResponse:
    """Acknowledge a worked example, then re-stage the question on screen.

    The client is holding a question whose starter was cut for the `worked`
    rung — i.e. not cut at all. Reading the example promotes the concept to
    `faded`, so the honest thing to return is that same question's faded
    starter, rather than making the client throw the question away and fetch
    another. Fetching another would burn a question out of the KC's pool, and
    some KCs own only two.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    if not kc_graph.registry_node(payload.kc):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown concept '{payload.kc}'",
        )
    kc_graph.note_worked_seen(user_state, payload.kc)
    save_user_state(user_id)

    stage = kc_graph.kc_stage(user_state, payload.kc)
    starter = None
    if payload.question_id is not None:
        question = get_question_by_id(payload.question_id)
        if question is not None:
            starter = ladder_starter(question, stage)
    return WorkedSeenResponse(
        ladder_stage=stage,
        ladder_estimate=kc_graph.kc_estimate(user_state, payload.kc),
        starter_code=starter,
    )
