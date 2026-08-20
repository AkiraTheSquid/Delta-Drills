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

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app import diagnostic, kc_graph, lessons
from app.adaptive import (
    COLD_START_TARGETS,
    UNRATED,
    get_user_state,
    override_pending_attempt,
    record_attempt,
    save_user_state,
)
from app.auth import get_current_user
from app.models import User
from app.practice.attempt_scoring import finalize_attempt, flush_stale_attempt
from app.practice.grading import (
    grade_submission,
    run_and_get_expected_output,
    select_question_for_difficulty,
)
from app.practice_schemas import (
    LocalEvalResponse,
    LocalEvalSubmitRequest,
    NextQuestionResponse,
    OverrideAttemptRequest,
    OverrideAttemptResponse,
    SubmitRequest,
    SubmitResponse,
)
from app.prioritization import (
    ladder_fields,
    ladder_starter,
    narrow_to_next_kc,
    question_is_unlocked,
    record_ladder_outcome,
    select_next_subtopic,
    question_target_difficulty,
    target_difficulty,
)
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
def next_question(
    focus_subtopic: str | None = Query(None),
    user: User = Depends(get_current_user),
) -> NextQuestionResponse:
    """`focus_subtopic` pins the queue to one subtopic (single-KC practice from
    the concept graph). It only overrides *selection*; scoring, unlock gates and
    the placement diagnostic are untouched. Unknown values fall back to the
    normal weakest-subtopic pick rather than 404ing."""
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    subtopic = None
    if focus_subtopic and get_questions_by_subtopic(focus_subtopic):
        subtopic = focus_subtopic

    # A focused request skips placement probing: the learner opened ONE concept
    # from the graph, and cross-topic probes there would look broken (and would
    # never move that concept's competency bar). The diagnostic stays active and
    # resumes on the normal queue; these attempts still count as evidence.
    if subtopic is None and diagnostic.should_run(user_state):
        probe = _serve_diagnostic_probe(user_id, user_state)
        if probe is not None:
            return probe

    if subtopic is None:
        subtopic = select_next_subtopic(user_state)
    if subtopic is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No questions available",
        )

    sub_state = user_state.get_subtopic_state(subtopic)
    candidates = [
        q for q in get_questions_by_subtopic(subtopic)
        if question_is_unlocked(user_state, q)
    ]
    # A focused request is the learner explicitly opening one concept, so honour
    # that over the queue's own idea of what comes next; on the normal path,
    # keep the served question on the same KC the graph is highlighting. The
    # concept is resolved either way and only the NARROWING is conditional: the
    # aim has to be measured on a concept on both paths, or focused practice
    # keeps the subtopic-wide average this change exists to remove.
    served = set(sub_state.served_question_ids)
    narrowed, next_kc = narrow_to_next_kc(user_state, candidates, served)
    if focus_subtopic is None:
        candidates = narrowed
    target_diff = target_difficulty(user_state, subtopic, kc=next_kc)
    question = select_question_for_difficulty(
        candidates, target_diff, served, sub_state.served_question_ids
    )
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No questions available for subtopic '{subtopic}'",
        )

    # Teach one concept, then drill THAT concept — see lessons.segment_drill.
    question = lessons.segment_drill(question, user_state.kc_exposure, served) or question

    # Report the aim on the concept actually SERVED. `next_kc` drove the pick,
    # but a question can target more than one concept and a focused pool is not
    # narrowed to one at all — and `finalize_attempt` recomputes this from the
    # answered question. Reading it the same way here is what stops the bar
    # jumping on submit and jumping back on the next load.
    target_diff = question_target_difficulty(user_state, subtopic, question.id)
    sub_state.served_question_ids.append(question.id)
    save_user_state(user_id)

    expected_output = (
        question.expected_output
        if question.supports_visual_output
        else (question.expected_output or run_and_get_expected_output(question.answer_code))
    )

    # Ladder rung for this concept, and the scaffolded starter that goes with
    # it. On the faded/partial rungs the learner is handed the canonical
    # solution with its TAIL removed (backward fading), not a blank page.
    ladder = ladder_fields(user_state, question.id)
    mastery_fields = {}
    if ladder.get("ladder_kc"):
        mastery, coverage, tier = kc_graph.kc_mastery(
            user_state, ladder["ladder_kc"]
        )
        mastery_fields = {
            "kc_mastery": mastery,
            "kc_coverage": coverage,
            "kc_tier": tier,
        }
    scaffold = ladder_starter(question, ladder.get("ladder_stage") or "")
    starter = scaffold or question.starter_code
    # Whether the promise THIS rung makes is actually on the page — blanks at
    # `faded`, an example above the problem at `partial`. See
    # lessons.rung_support and `ladder_support` in practice_schemas.
    if ladder:
        ladder["ladder_support"] = lessons.rung_support(
            question.id, ladder.get("ladder_stage") or "", scaffold
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
        starter_code=starter,
        test_cases=question.test_cases,
        submission_mode=question.submission_mode,
        hint=question.hint,
        solution_notebook_path=question.solution_notebook_path,
        problem_notebook_path=question.problem_notebook_path,
        # Exposure guard: placement probes are never gated (the diagnostic
        # measures prior knowledge — teaching first would corrupt it), so
        # this only runs on the normal adaptive-queue path.
        lesson_gate=lessons.unexposed_target_kcs(question.id, user_state.kc_exposure),
        **mastery_fields,
        **ladder,
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

    is_diagnostic = diagnostic.get_diag(user_state)["active"]
    if is_diagnostic:
        # Placement probe: update the diagnostic posterior directly (BKT is
        # seeded once at finish). No pending attempt / felt-difficulty step —
        # /override can still flip the latest probe's result.
        diagnostic.record_probe(
            user_state, question, "correct" if correct else "incorrect"
        )
    else:
        # Anything still parked is about to be overwritten by this one.
        flush_stale_attempt(user_state)
        record_attempt(
            user_state=user_state,
            question_id=question.id,
            subtopic=question.subtopic,
            difficulty_score=question.difficulty_score,
            correct=correct,
        )
        # Ladder evidence is recorded only OUTSIDE the diagnostic. A placement
        # probe measures prior knowledge on questions the learner was never
        # taught, so counting it would demote them to worked examples for
        # concepts the probe never intended to teach.
        record_ladder_outcome(user_state, question.id, correct)
    save_user_state(user_id)
    ladder = {} if is_diagnostic else ladder_fields(user_state, question.id)

    return SubmitResponse(
        correct=correct,
        actual_output=actual_output,
        expected_output=expected_output,
        solution_code=compose_full_solution(question.starter_code, question.answer_code),
        failed_tests=failed_tests,
        ladder_estimate=ladder.get("ladder_estimate"),
    )


@router.post("/submit-local-eval", response_model=LocalEvalResponse)
def submit_local_eval(
    payload: LocalEvalSubmitRequest,
    user: User = Depends(get_current_user),
) -> LocalEvalResponse:
    """A grade the client already decided, for the routes the sandbox can't run.

    Callers differ in whether anything follows, and that is what `finalize`
    says. The Colab edition and the einops fallback both post `finalize=false`:
    a felt-difficulty rating still follows, so the attempt stays pending for it
    to land on and /feedback does the scoring. Finalizing here is for a caller
    that is DONE, and closes the attempt out as UNRATED — it exists because the
    Colab edition once had no rating step, and an attempt parked with nothing
    coming for it sat pending until the next submit overwrote it: `n` never
    moved, no posterior moved, and the concept graph reported a learner who had
    done nothing. Keep it: it is the right answer for any route that grades
    without asking.
    """
    user_id = str(user.id)
    user_state = get_user_state(user_id)

    question = get_question_by_id(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    if diagnostic.get_diag(user_state)["active"]:
        # A placement probe is not a graded attempt: it updates the diagnostic
        # posterior directly and never creates a pending attempt, so there is
        # nothing here to finalize and no subtopic reading that would have
        # moved. Same rule as /submit.
        diagnostic.record_probe(
            user_state, question, "correct" if payload.correct else "incorrect"
        )
        save_user_state(user_id)
        return LocalEvalResponse(success=True, finalized=False)

    td_before = p_before = None
    if payload.finalize:
        # Read the "before" numbers first: finalize_attempt overwrites both.
        # Guarded rather than read unconditionally because get_subtopic_state
        # CREATES the row it cannot find, and a call that finalizes nothing
        # should leave no trace in stored state.
        sub_state = user_state.get_subtopic_state(question.subtopic)
        td_before = sub_state.target_difficulty
        p_before = sub_state.p

    # Same rule as /submit: an attempt left parked by an earlier request is
    # about to be overwritten, and losing it is the 2026-08-03 bug. Runs after
    # the diagnostic branch above, which returns before creating anything.
    flush_stale_attempt(user_state)
    record_attempt(
        user_state=user_state,
        question_id=question.id,
        subtopic=question.subtopic,
        difficulty_score=question.difficulty_score,
        correct=payload.correct,
    )
    record_ladder_outcome(user_state, question.id, payload.correct)

    attempt = None
    if payload.finalize:
        # UNRATED, not one of the three real levels: the learner was never
        # asked how hard it felt, so there is no opinion to record. Runs after
        # record_ladder_outcome only because the ladder reads the rung the
        # learner was sitting on; the two touch disjoint state.
        attempt = finalize_attempt(user_state, UNRATED)
    save_user_state(user_id)

    # The rung this concept sits on now that the outcome is in. `ladder_fields`
    # goes through kc_graph's READ path, so asking does not stamp an empty
    # ladder row into the learner's state, and it returns {} for a question no
    # KC claims — hence the .get()s rather than an is-tagged branch.
    ladder = ladder_fields(user_state, question.id)
    return LocalEvalResponse(
        success=True,
        finalized=attempt is not None,
        pending=user_state.pending_attempt is not None,
        target_difficulty_before=td_before,
        target_difficulty_after=attempt.target_difficulty_after if attempt else None,
        p_before=p_before,
        p_after=attempt.p_after if attempt else None,
        ladder_stage=ladder.get("ladder_stage"),
        ladder_estimate=ladder.get("ladder_estimate"),
    )


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
