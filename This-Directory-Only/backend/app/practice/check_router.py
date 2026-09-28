"""
The Run button's dry grade: which test cases pass RIGHT NOW, recorded nowhere.

Endpoints (mounted under /api/practice by the parent router):
  POST /check
  GET  /starter/{question_id}?language=javascript|java
                                   a LeetCode drill in another language: its starter
  GET  /js-starter/{question_id}   the same for JavaScript (older clients)
  POST /run-js                     JavaScript cell ▶: console output only
  POST /run-java                   Java cell ▶: compile, run a `main` if any

Seth, 2026-09-13: "when you run the code, it tells you which test cases
passed and which test cases failed, even if you didn't submit it, so that
you get immediate feedback on what you did right versus what you did wrong."

This is `grade_submission` with the verdict thrown away: the same harness
(`run_function_tests`), the same equality rules, the same stdout compare for
the stdout-graded drills, so what ▶ says and what Submit says can never
disagree. It records NOTHING — no attempt, no ladder evidence, no pending
slot — which is the whole point of pressing ▶ instead of Submit. The AI-judge
fallback is deliberately NOT here: a model call per keystroke-and-run is
neither cheap nor deterministic, so those drills answer `supported=false`
and the client shows nothing rather than a guess.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app import leetcode_java, leetcode_js
from app.auth import get_current_user
from app.code_runner import (
    TORCH_COLAB_MESSAGE,
    code_uses_torch,
    run_code,
    run_function_tests,
    torch_available,
)
from app.models import User
from app.practice.grading import TRANSLATED, run_and_get_expected_output
from app.practice_schemas import CheckRequest, CheckResponse, CodeRunRequest, CodeRunResponse
from app.questions import get_question_by_id

router = APIRouter()


@router.post("/check", response_model=CheckResponse)
def check_answer(
    payload: CheckRequest,
    user: User = Depends(get_current_user),
) -> CheckResponse:
    question = get_question_by_id(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    if payload.language in TRANSLATED:
        return _check_translated(question, payload.user_code, payload.language)

    # Same refusal as grade_submission: without torch there is no honest run.
    if not torch_available() and (
        getattr(question, "primary_library", None) == "torch"
        or code_uses_torch(payload.user_code)
        or code_uses_torch(getattr(question, "answer_code", "") or "")
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=TORCH_COLAB_MESSAGE,
        )

    # Function tests first — the audited contract, same order as grading.py.
    if question.submission_mode == "function" and question.test_cases:
        results, execution = run_function_tests(payload.user_code, question.test_cases)
        # The harness answers one row per case, in the bank's order — except
        # when the learner's code itself fails to import, where it answers ONE
        # row carrying the traceback. Zip by position and let the client pad.
        tests = [
            {
                "call": (
                    question.test_cases[i].get("call", "")
                    if i < len(question.test_cases) else ""
                ),
                "passed": r.passed,
                "actual": r.actual,
                "expected": r.expected,
                "error": r.error,
                "note": r.note,
            }
            for i, r in enumerate(results)
        ]
        return CheckResponse(
            supported=True,
            correct=all(r.passed for r in results),
            actual_output=(execution.stdout.strip() or execution.stderr.strip()),
            tests=tests,
        )

    if question.task_type == "stdout_prediction" and not question.supports_visual_output:
        expected = run_and_get_expected_output(question.answer_code)
        if expected.strip():
            run = run_code(payload.user_code, timeout=20)
            actual = run.stdout.strip()
            return CheckResponse(
                supported=True,
                correct=actual == expected.strip(),
                actual_output=actual or run.stderr.strip(),
                tests=[{
                    "call": "printed output",
                    "passed": actual == expected.strip(),
                    "actual": actual,
                    "expected": expected.strip(),
                    "error": "" if run.success else run.stderr.strip(),
                }],
            )

    # AI-judged drill: no deterministic verdict exists before Submit.
    return CheckResponse(supported=False, correct=False, actual_output="", tests=[])


def _check_translated(question, user_code: str, language: str) -> CheckResponse:
    """▶ on a LeetCode drill answered in JavaScript or Java:
    grading.grade_translated's harness, verdict discarded. `call` is the
    translated call as the learner would write it, not the Python one."""
    run_tests, _label = TRANSLATED[language]
    try:
        results, execution, cases = run_tests(user_code, question)
    except leetcode_js.Untranslatable:
        return CheckResponse(supported=False, correct=False, actual_output="", tests=[])
    tests = [
        {
            "call": cases[i]["display"] if i < len(cases) else "",
            "passed": r.passed,
            "actual": r.actual,
            "expected": r.expected,
            "error": r.error,
            "note": r.note,
        }
        for i, r in enumerate(results)
    ]
    return CheckResponse(
        supported=True,
        correct=all(r.passed for r in results),
        actual_output=(execution.stdout.strip() or execution.stderr.strip()),
        tests=tests,
    )


# language -> (runtime present?, cases (raises Untranslatable), starter, runtime name)
_STARTERS = {
    leetcode_js.LANGUAGE: (lambda: leetcode_js.node_binary() is not None, leetcode_js.js_cases,
                           leetcode_js.starter_js, "JavaScript"),
    leetcode_java.LANGUAGE: (lambda: leetcode_java.java_binary() is not None and leetcode_java.javac_binary() is not None,
                             leetcode_java.java_cases, leetcode_java.starter_java, "Java"),
}


@router.get("/starter/{question_id}")
def lang_starter(
    question_id: int,
    language: str = Query(...),
    user: User = Depends(get_current_user),
) -> dict:
    """Whether this drill can be answered in `language`, and its stub.
    Grading itself rides /check and /submit (`language: ...`)."""
    if language not in _STARTERS:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"No starters in {language!r}")
    question = get_question_by_id(question_id)
    if question is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    runtime_ok, cases, starter, name = _STARTERS[language]
    if not runtime_ok():
        return {"supported": False, "starter": "", "reason": f"The server has no {name} runtime."}
    try:
        cases(question)
        return {"supported": True, "starter": starter(question), "reason": ""}
    except leetcode_js.Untranslatable as exc:
        return {"supported": False, "starter": "", "reason": str(exc)}


@router.get("/js-starter/{question_id}")
def js_starter(question_id: int, user: User = Depends(get_current_user)) -> dict:
    """/starter?language=javascript, for a client cached before Java."""
    return lang_starter(question_id, leetcode_js.LANGUAGE, user)


@router.post("/run-js", response_model=CodeRunResponse)
def run_js(payload: CodeRunRequest, user: User = Depends(get_current_user)) -> CodeRunResponse:
    result = leetcode_js.run_js(payload.code)
    return CodeRunResponse(stdout=result.stdout, stderr=result.stderr, success=result.success)


@router.post("/run-java", response_model=CodeRunResponse)
def run_java(payload: CodeRunRequest, user: User = Depends(get_current_user)) -> CodeRunResponse:
    question = get_question_by_id(payload.question_id) if payload.question_id is not None else None
    result = leetcode_java.run_java(payload.code, question)
    return CodeRunResponse(stdout=result.stdout, stderr=result.stderr, success=result.success)
