"""The learner's practice settings — target, course mix, courses studied.

Endpoints (mounted under /api/practice by the parent router):
  GET/POST /practice-target
  GET/POST /course-shares
  POST     /study-courses

Until 2026-09-26 this was `diagnostic_router.py` and also ran the placement
test (/diagnostic/status, plan, start, answer, finish, decline). Seth retired
the placement test: the Practice tab's "what have you done before?" survey
(`survey_router.py`, `app/area_survey.py`) and explore/exploit replace it.
Stored placement evidence is still READ (`app/diagnostic.py`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app import course_mix, course_registry, practice_targets
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User
from app.practice_schemas import (
    CourseShareRequest,
    PracticeTargetRequest,
    StudyCoursesRequest,
)

router = APIRouter()


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
