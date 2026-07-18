"""
Practice question endpoints: serving the next question, recording
attempts, overriding correctness.

Endpoints (mounted under /api/practice by the parent router):
  GET  /next-question
  POST /submit
  POST /submit-local-eval
  POST /override
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app import diagnostic, lessons
from app.adaptive import (
    COLD_START_TARGETS,
    get_user_state,
    override_pending_attempt,
    record_attempt,
    save_user_state,
)
from app.auth import get_current_user
from app.models import User
from app.practice.grading import (
    grade_submission,
    run_and_get_expected_output,
    select_question_for_difficulty,
)
from app.practice_schemas import (
    LocalEvalSubmitRequest,
    NextQuestionResponse,
    OverrideAttemptRequest,
    OverrideAttemptResponse,
    SubmitRequest,
    SubmitResponse,
)
from app.prioritization import question_is_unlocked, select_next_subtopic, target_difficulty
from app.questions import compose_full_solution, get_question_by_id, get_questions_by_subtopic

router = APIRouter()


def _serve_diagnostic_probe(user_id: str, user_state) -> NextQuestionResponse | None:
    """When the placement diagnostic is active, serve the next ALEKS-style
    probe (max-information item across topic areas) instead of the normal
    weakest-subtopic flow. Returns None when no informative probe remains —
    the diagnostic finishes (seeding BKT) and the caller falls through."""
    if diagnostic.should_finish(user_state):
        # Budget already met/exceeded (e.g. history credit shrank it between
        # requests) — finish now rather than serving probe N+1 "of ≤N".
        diagnostic.finish(user_state)
        save_user_state(user_id)
        return None
    question = diagnostic.select_probe(user_state)
    if question is None:
        diagnostic.finish(user_state)
        save_user_state(user_id)
        return None

    sub_state = user_state.get_subtopic_state(question.subtopic)
    if question.id not in sub_state.served_question_ids:
        sub_state.served_question_ids.append(question.id)
    save_user_state(user_id)

    expected_output = (
        question.expected_output
        if question.supports_visual_output
        else (question.expected_output or run_and_get_expected_output(question.answer_code))
    )
    return NextQuestionResponse(
        question_id=question.id,
        question_text=question.question_text,
        topic=question.topic,
        subtopic=question.subtopic,
        difficulty=question.difficulty_score,
        target_difficulty=question.difficulty_score,
        expected_output=expected_output,
        solution_code=compose_full_solution(question.starter_code, question.answer_code),
        is_cold_start=False,
        subtopic_n=sub_state.n,
        p_current=None,
        primary_library=question.primary_library,
        task_type=question.task_type,
        expected_artifact_type=question.expected_artifact_type,
        supports_visual_output=question.supports_visual_output,
        function_name=question.function_name,
        starter_code=question.starter_code,
        test_cases=question.test_cases,
        submission_mode=question.submission_mode,
        hint=question.hint,
        solution_notebook_path=question.solution_notebook_path,
        problem_notebook_path=question.problem_notebook_path,
        diagnostic_active=True,
        diagnostic_probe_index=len(diagnostic.get_diag(user_state)["probes"]) + 1,
        diagnostic_budget=diagnostic.effective_budget(user_state),
        diagnostic_area=question.topic,
    )


@router.get("/next-question", response_model=NextQuestionResponse)
def next_question(user: User = Depends(get_current_user)) -> NextQuestionResponse:
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    if diagnostic.should_run(user_state):
        probe = _serve_diagnostic_probe(user_id, user_state)
        if probe is not None:
            return probe

    subtopic = select_next_subtopic(user_state)
    if subtopic is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No questions available",
        )

    sub_state = user_state.get_subtopic_state(subtopic)
    target_diff = target_difficulty(user_state, subtopic)

    candidates = [
        q for q in get_questions_by_subtopic(subtopic)
        if question_is_unlocked(user_state, q)
    ]
    served = set(sub_state.served_question_ids)
    question = select_question_for_difficulty(candidates, target_diff, served)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No questions available for subtopic '{subtopic}'",
        )

    sub_state.served_question_ids.append(question.id)
    save_user_state(user_id)

    expected_output = (
        question.expected_output
        if question.supports_visual_output
        else (question.expected_output or run_and_get_expected_output(question.answer_code))
    )

    return NextQuestionResponse(
        question_id=question.id,
        question_text=question.question_text,
        topic=question.topic,
        subtopic=question.subtopic,
        difficulty=question.difficulty_score,
        target_difficulty=target_diff,
        expected_output=expected_output,
        solution_code=compose_full_solution(question.starter_code, question.answer_code),
        is_cold_start=sub_state.n < len(COLD_START_TARGETS),
        subtopic_n=sub_state.n,
        p_current=sub_state.p if sub_state.n > 0 else None,
        primary_library=question.primary_library,
        task_type=question.task_type,
        expected_artifact_type=question.expected_artifact_type,
        supports_visual_output=question.supports_visual_output,
        function_name=question.function_name,
        starter_code=question.starter_code,
        test_cases=question.test_cases,
        submission_mode=question.submission_mode,
        hint=question.hint,
        solution_notebook_path=question.solution_notebook_path,
        problem_notebook_path=question.problem_notebook_path,
        # Exposure guard: placement probes are never gated (the diagnostic
        # measures prior knowledge — teaching first would corrupt it), so
        # this only runs on the normal adaptive-queue path.
        lesson_gate=lessons.unexposed_target_kcs(question.id, user_state.kc_exposure),
    )


@router.post("/submit", response_model=SubmitResponse)
def submit_answer(
    payload: SubmitRequest,
    user: User = Depends(get_current_user),
) -> SubmitResponse:
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    question = get_question_by_id(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    correct, actual_output, expected_output, failed_tests = grade_submission(
        question, payload.user_code, user
    )

    if diagnostic.get_diag(user_state)["active"]:
        # Placement probe: update the diagnostic posterior directly (BKT is
        # seeded once at finish). No pending attempt / felt-difficulty step —
        # /override can still flip the latest probe's result.
        diagnostic.record_probe(
            user_state, question, "correct" if correct else "incorrect"
        )
    else:
        record_attempt(
            user_state=user_state,
            question_id=question.id,
            subtopic=question.subtopic,
            difficulty_score=question.difficulty_score,
            correct=correct,
        )
    save_user_state(user_id)

    return SubmitResponse(
        correct=correct,
        actual_output=actual_output,
        expected_output=expected_output,
        solution_code=compose_full_solution(question.starter_code, question.answer_code),
        failed_tests=failed_tests,
    )


@router.post("/submit-local-eval", response_model=OverrideAttemptResponse)
def submit_local_eval(
    payload: LocalEvalSubmitRequest,
    user: User = Depends(get_current_user),
) -> OverrideAttemptResponse:
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    question = get_question_by_id(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    if diagnostic.get_diag(user_state)["active"]:
        diagnostic.record_probe(
            user_state, question, "correct" if payload.correct else "incorrect"
        )
    else:
        record_attempt(
            user_state=user_state,
            question_id=question.id,
            subtopic=question.subtopic,
            difficulty_score=question.difficulty_score,
            correct=payload.correct,
        )
    save_user_state(user_id)
    return OverrideAttemptResponse(success=True)


@router.post("/override", response_model=OverrideAttemptResponse)
def override_attempt(
    payload: OverrideAttemptRequest,
    user: User = Depends(get_current_user),
) -> OverrideAttemptResponse:
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    if diagnostic.get_diag(user_state)["active"] or (
        diagnostic.get_diag(user_state)["probes"]
        and diagnostic.get_diag(user_state)["probes"][-1]["question_id"]
        == payload.question_id
        and user_state.pending_attempt is None
    ):
        # During (or right at the end of) placement, override flips the
        # latest probe instead of a pending attempt.
        if diagnostic.override_probe(user_state, payload.question_id, payload.correct):
            if diagnostic.get_diag(user_state)["completed_at"]:
                # The flipped probe was the finishing one — re-seed from the
                # corrected posterior (seeding only ever raises mastery).
                diagnostic.finish(user_state)
            save_user_state(user_id)
            return OverrideAttemptResponse(success=True)

    updated = override_pending_attempt(
        user_state=user_state,
        question_id=payload.question_id,
        correct=payload.correct,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No matching pending attempt to override.",
        )
    save_user_state(user_id)
    return OverrideAttemptResponse(success=True)
