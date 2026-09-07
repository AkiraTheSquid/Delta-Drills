"""Placement-diagnostic endpoints (graph-wide, ALEKS-style calibration).

Endpoints (mounted under /api/practice by the parent router):
  GET  /diagnostic/status
  GET  /diagnostic/plan      — the 1h / 3h / 6h picker with time estimates
  POST /diagnostic/start     — body {hours}
  POST /diagnostic/answer    — "I don't know yet" / self-rated probe results
  POST /diagnostic/finish
  POST /diagnostic/decline

Probe SELECTION + answered-probe recording live in the normal practice flow
(/next-question and /submit route through app.diagnostic when active); this
router owns the lifecycle + the no-code-attempt response path.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app import diagnostic
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User
from app.practice_schemas import (
    DiagnosticAnswerRequest,
    DiagnosticPlanResponse,
    DiagnosticStartRequest,
    DiagnosticStatusResponse,
)
from app.questions import get_question_by_id

router = APIRouter()


def _status(user_state) -> DiagnosticStatusResponse:
    d = diagnostic.get_diag(user_state)
    plan = d.get("plan")
    return DiagnosticStatusResponse(
        active=d["active"],
        completed_at=d["completed_at"],
        declined=d["declined"],
        probes_done=len(d["probes"]),
        budget=diagnostic.effective_budget(user_state),
        min_probes=diagnostic.effective_min_probes(user_state),
        # The grouped readout the existing results card draws (PyTorch /
        # Einops / Einsum), folded from the per-concept rows below.
        areas=diagnostic.display_area_estimates(user_state),
        atoms_seeded=d.get("atoms_seeded"),
        can_set_prior=diagnostic.can_set_prior(user_state),
        self_reported_level=user_state.self_reported_level,
        plan=(
            {
                **plan,
                "spent_secs": int(d.get("spent_secs") or 0),
                "remaining_secs": diagnostic.remaining_secs(user_state),
                "problem_secs_allowed": diagnostic.problem_secs_allowed(user_state),
            }
            if isinstance(plan, dict)
            else None
        ),
        kcs=diagnostic.kc_estimates(user_state),
        fast_track=list(d.get("fast_track") or []),
        edge_violations=list(d.get("edge_violations") or []),
    )


@router.get("/diagnostic/status", response_model=DiagnosticStatusResponse)
def diagnostic_status(user: User = Depends(get_current_user)) -> DiagnosticStatusResponse:
    return _status(get_user_state(str(user.id)))


@router.get("/diagnostic/plan", response_model=DiagnosticPlanResponse)
def diagnostic_plan(user: User = Depends(get_current_user)) -> DiagnosticPlanResponse:
    user_state = get_user_state(str(user.id))
    kcs = diagnostic.assessed_kcs(user_state)
    links = diagnostic._arena_links()
    return DiagnosticPlanResponse(
        options=diagnostic.plan_options(user_state),
        assessed_kcs=len(kcs),
        arena_linked_kcs=sum(1 for k in kcs if links.get(k)),
    )


@router.post("/diagnostic/start", response_model=DiagnosticStatusResponse)
def diagnostic_start(
    payload: DiagnosticStartRequest | None = None,
    user: User = Depends(get_current_user),
) -> DiagnosticStatusResponse:
    user_state = get_user_state(str(user.id))
    diagnostic.start(user_state, hours=payload.hours if payload else None)
    save_user_state(str(user.id))
    return _status(user_state)


@router.post("/diagnostic/answer", response_model=DiagnosticStatusResponse)
def diagnostic_answer(
    payload: DiagnosticAnswerRequest,
    user: User = Depends(get_current_user),
) -> DiagnosticStatusResponse:
    user_state = get_user_state(str(user.id))
    d = diagnostic.get_diag(user_state)
    if not d["active"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active placement diagnostic.",
        )
    question = get_question_by_id(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )
    # Only the problem the server served can be answered on this path — a
    # no-attempt response to an arbitrary question would be free evidence.
    #
    # 🔴 AN EMPTY `pending` IS NOT A PASS. It used to be: with nothing on
    # screen the check fell through, so any question id at all could be posted
    # as "dont_know" and be recorded as a probe the server never served —
    # evidence about a concept the learner was never asked about, straight into
    # the placement estimate.
    #
    # 🔴 AND A RETRY IS A REPLAY, NOT A SECOND RECORDING (codex, 2026-09-07).
    # The first fix let any id already in the probe log through and re-recorded
    # it — so a placement-timer "dont_know" firing late, behind a /submit that
    # had just graded the same problem correct, REPLACED the correct record
    # with dont_know, and a stale tab could re-post an old probe the same way.
    # Placement evidence seeds BKT mastery; nothing on this path may rewrite a
    # record. With nothing pending, only the most recent probe's id is accepted,
    # and it is answered with the current state, unrecorded. Everything else is
    # 409.
    pending_id = (d.get("pending") or {}).get("question_id")
    if pending_id is not None:
        if pending_id != payload.question_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="That is not the placement problem currently on screen.",
            )
    else:
        latest = d["probes"][-1]["question_id"] if d["probes"] else None
        if latest != payload.question_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="That is not the placement problem currently on screen.",
            )
        return _status(user_state)
    diagnostic.record_probe(user_state, question, payload.result, elapsed_secs=payload.elapsed_secs)
    save_user_state(str(user.id))
    return _status(user_state)


@router.post("/diagnostic/finish", response_model=DiagnosticStatusResponse)
def diagnostic_finish(user: User = Depends(get_current_user)) -> DiagnosticStatusResponse:
    user_state = get_user_state(str(user.id))
    d = diagnostic.get_diag(user_state)
    if d["completed_at"] is None and not d["probes"]:
        # Finishing with zero probes = just don't want it → treat as decline
        # (no seeding from the bare prior).
        diagnostic.decline(user_state)
    elif d["completed_at"] is None:
        diagnostic.finish(user_state)
    save_user_state(str(user.id))
    return _status(user_state)


@router.post("/diagnostic/decline", response_model=DiagnosticStatusResponse)
def diagnostic_decline(user: User = Depends(get_current_user)) -> DiagnosticStatusResponse:
    user_state = get_user_state(str(user.id))
    diagnostic.decline(user_state)
    save_user_state(str(user.id))
    return _status(user_state)
