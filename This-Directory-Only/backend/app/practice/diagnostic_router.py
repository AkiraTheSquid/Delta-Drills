"""Placement-diagnostic endpoints (graph-wide, ALEKS-style calibration).

Endpoints (mounted under /api/practice by the parent router):
  GET  /diagnostic/status
  GET  /diagnostic/plan      — ?areas=A,B: the area catalogue + three lengths cut to that focus
  POST /diagnostic/start     — body {minutes, areas, scope} (old clients: {hours})
  POST /diagnostic/answer    — "I don't know yet" / self-rated probe results
  POST /diagnostic/finish
  POST /diagnostic/decline

Probe SELECTION + answered-probe recording live in the normal practice flow
(/next-question and /submit route through app.diagnostic when active); this
router owns the lifecycle + the no-code-attempt response path.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app import course_mix, course_registry, diagnostic, placement_scope, practice_targets
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User
from app.practice_schemas import (
    DiagnosticAnswerRequest,
    DiagnosticPlanResponse,
    DiagnosticStartRequest,
    DiagnosticStatusResponse,
    CourseShareRequest,
    PracticeTargetRequest,
    StudyCoursesRequest,
)
from app.questions import get_question_by_id

router = APIRouter()


def _status(user_state) -> DiagnosticStatusResponse:
    d = diagnostic.get_diag(user_state)
    plan = d.get("plan")
    lo, hi = diagnostic.cap_range(user_state)
    return DiagnosticStatusResponse(
        active=d["active"],
        scope=d.get("scope", "all"),
        practice_target=user_state.practice_target,
        completed_at=d["completed_at"],
        per_problem_min_secs=lo,
        per_problem_max_secs=hi,
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
        focus_areas=d.get("areas") or None,
        plan=(
            {
                **plan,
                "spent_secs": int(d.get("spent_secs") or 0),
                "remaining_secs": diagnostic.remaining_secs(user_state),
                "problem_secs_allowed": diagnostic.pending_secs_left(user_state),
                "per_problem_min_secs": lo,
                "per_problem_max_secs": hi,
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
def diagnostic_plan(areas: str | None = None, user: User = Depends(get_current_user)) -> DiagnosticPlanResponse:
    user_state = get_user_state(str(user.id))
    plan = placement_scope.plan_options(user_state, areas)
    kcs = plan["kcs"]
    links = diagnostic._arena_links()
    return DiagnosticPlanResponse(
        options=plan["options"],
        assessed_kcs=len(kcs),
        arena_linked_kcs=sum(1 for k in kcs if links.get(k)),
        area_catalog=placement_scope.area_catalog(user_state),
        probes_to_settle=plan["probes_to_settle"],
        per_problem_min_secs=plan["per_problem_min_secs"],
        per_problem_max_secs=plan["per_problem_max_secs"],
    )


@router.post("/diagnostic/start", response_model=DiagnosticStatusResponse)
def diagnostic_start(
    payload: DiagnosticStartRequest | None = None,
    user: User = Depends(get_current_user),
) -> DiagnosticStatusResponse:
    user_state = get_user_state(str(user.id))
    if diagnostic.should_run(user_state) and diagnostic.get_diag(user_state).get("scope", "all") != (payload.scope if payload else "all"):
        raise HTTPException(status_code=409, detail="Finish the current placement before starting a different one.")
    diagnostic.start(
        user_state,
        hours=payload.hours if payload else None,
        scope=payload.scope if payload else "all",
        minutes=payload.minutes if payload else None,
        areas=payload.areas if payload else None,
    )
    save_user_state(str(user.id))
    return _status(user_state)


@router.get("/practice-target")
def practice_target(user: User = Depends(get_current_user)):
    state = get_user_state(str(user.id))
    return {"target": state.practice_target,
            "kcs": sorted(practice_targets.scope_kcs(state.practice_target) or []),
            "placement_ready": sorted(practice_targets.readiness(state))}


@router.post("/practice-target")
def set_practice_target(payload: PracticeTargetRequest, user: User = Depends(get_current_user)):
    state = get_user_state(str(user.id))
    state.practice_target = payload.target
    save_user_state(str(user.id))
    return practice_target(user)


@router.get("/course-shares")
def course_shares(user: User = Depends(get_current_user)):
    """The Courses tab: every course's enable state and mix (app/course_mix.py)
    — the share, and which turn the next drill would be."""
    state = get_user_state(str(user.id))
    return {
        "courses": [course_mix.status(state, c) for c in course_registry.COURSE_IDS],
        "enable_share": course_mix.DEFAULT_ENABLE_SHARE,
        # None until the onboarding "which courses?" question is answered.
        "study_courses": state.study_courses,
    }


@router.post("/course-shares")
def set_course_share(payload: CourseShareRequest, user: User = Depends(get_current_user)):
    """Toggle one course. Enabling sets its share to the fixed default
    (`course_mix.DEFAULT_ENABLE_SHARE`); disabling sets it to 0. No
    fine-tune slider — the Courses tab toggle is the only control."""
    if payload.course not in course_registry.COURSE_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown course '{payload.course}'")
    state = get_user_state(str(user.id))
    course_mix.toggle(state, payload.course, payload.enabled)
    save_user_state(str(user.id))
    return course_shares(user)


@router.post("/study-courses")
def set_study_courses(payload: StudyCoursesRequest, user: User = Depends(get_current_user)):
    """The onboarding "which courses do you want to study?" answer (Seth,
    2026-09-25). The courses left out go off (course_registry.course_off),
    ARENA's whole graph included; a standalone course picked is switched on
    at the Courses tab's share (unchanged if already on). ARENA's mix is
    kept when ARENA is picked and dropped with it when it is not."""
    picked = course_registry.normalize_study(payload.courses)
    if not picked:
        raise HTTPException(status_code=400, detail="Pick at least one course.")
    state = get_user_state(str(user.id))
    course_mix.set_study(state, picked)
    save_user_state(str(user.id))
    return course_shares(user)


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
