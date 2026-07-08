"""Placement-diagnostic endpoints (ALEKS-style cold-start calibration).

Endpoints (mounted under /api/practice by the parent router):
  GET  /diagnostic/status
  POST /diagnostic/start
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
    DiagnosticStatusResponse,
)
from app.questions import get_question_by_id

router = APIRouter()


def _status(user_state) -> DiagnosticStatusResponse:
    d = diagnostic.get_diag(user_state)
    return DiagnosticStatusResponse(
        active=d["active"],
        completed_at=d["completed_at"],
        declined=d["declined"],
        probes_done=len(d["probes"]),
        budget=diagnostic.MAX_PROBES,
        min_probes=diagnostic.MIN_PROBES,
        areas=diagnostic.area_estimates(user_state),
        atoms_seeded=d.get("atoms_seeded"),
    )


@router.get("/diagnostic/status", response_model=DiagnosticStatusResponse)
def diagnostic_status(user: User = Depends(get_current_user)) -> DiagnosticStatusResponse:
    return _status(get_user_state(str(user.id)))


@router.post("/diagnostic/start", response_model=DiagnosticStatusResponse)
def diagnostic_start(user: User = Depends(get_current_user)) -> DiagnosticStatusResponse:
    user_state = get_user_state(str(user.id))
    diagnostic.start(user_state)
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
    diagnostic.record_probe(user_state, question, payload.result)
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
