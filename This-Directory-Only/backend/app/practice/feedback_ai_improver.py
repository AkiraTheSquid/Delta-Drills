"""
Feedback -> local Claude Code -> repaired question.

When an allowlisted learner flags a question from the practice UI, the flag
becomes a job in app/feedback_repair_queue.py and the request ends there. The
model call happens on Seth's own machine: `ops/question_repair/run_repairs.py`
drives the local `claude` CLI under his login, in a read-only sandbox, and
posts the result back. The server holds no model credential at all.

This module is the part BOTH sides share — the server, which queues jobs and
applies whatever comes back, and the runner, which imports it for the prompt and
the gates so the two can never drift apart:

  SYSTEM_PROMPT / build_prompt()  what the local session is asked to do
  REPAIR_JSON_SCHEMA             the shape it must answer in
  validated_changes()            which of its answers may touch the bank
  apply_repair()                 the write itself

It hangs off the one endpoint every exercise's quality feedback already posts to
(problem_feedback_router.py), which is why it covers the whole bank — numpy,
einsum, einops, torch, CNN, curated additions — without a per-course hook.

Three things keep an auto-applied rewrite from being a one-way door:

  Narrow surface.  Only question_text / starter_code / answer_code can change.
                   Curriculum placement, difficulty, test_cases and atom tags
                   feed the adaptive engine and the KC lattice, so the model
                   never touches them and a bad rewrite can't move a question
                   in the mastery graph.
  Gates.           Code fields must compile, answer_code only opens for a
                   `broken` flag, a rewrite that comes back identical or empty
                   is dropped, and the runner additionally re-runs a rewritten
                   answer through the real grading harness before it is sent.
  Reversibility.   Every apply logs its own before/after to an append-only
                   revision log, and any single question can be reverted to the
                   shipped text with one rollback call.

Configuration (all via env; nothing is hardcoded):
  DELTA_FEEDBACK_AI_EMAILS comma-separated allowlist. Defaults to Seth.
  DELTA_FEEDBACK_AI_DIR    where the queue, override layer and revision log are
                           written (the Fly /data volume in production).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from app import feedback_ai_layer, feedback_repair_queue, questions

logger = logging.getLogger(__name__)

# What the runner is expected to use. It may be overridden per run, and the
# model that actually answered is recorded on the revision, so this is a default
# rather than a promise.
DEFAULT_MODEL = "claude-opus-5"

# Who may trigger a rewrite. A flag from anyone else is still logged by the
# router — it just doesn't queue work or move the bank.
_DEFAULT_ALLOWLIST = "sethbgibson@gmail.com"

# `good` is praise, not a defect report; rewriting on it would churn the bank
# for questions that are already working.
_ACTIONABLE_TAGS = {"broken", "unclear", "wrong_image"}

_TAG_GUIDANCE = {
    "broken": (
        "The learner says the question is BROKEN — the prompt, the starter code or the "
        "reference answer is wrong, contradictory, or impossible as written. Find the "
        "actual defect and fix it."
    ),
    "unclear": (
        "The learner says the question is UNCLEAR — the task is probably correct but the "
        "wording leaves them guessing what is being asked, what shape the answer takes, or "
        "what the constraints are. Fix the prompt; the underlying task should stay the same."
    ),
    "wrong_image": (
        "The learner says the rendered figure or expected output does not match the prompt. "
        "Rewrite the prompt so it describes what the code actually produces, or drop the "
        "reference to the figure if the question does not need one."
    ),
}

SYSTEM_PROMPT = """\
You repair questions in Delta Drills, a practice bank for PyTorch, NumPy, einops and einsum.

You are given one question and one learner's complaint about it. Repair exactly that \
complaint. Do not restyle the question, do not change what skill it tests, do not change \
its difficulty, and do not touch anything the complaint did not raise — a question in this \
bank is tied to a knowledge-graph node by what it asks, so a drifting rewrite quietly \
breaks the learner's mastery estimate.

You are read-only. You can read and search the repository you are started in, which is the \
Delta Drills source, and that is often worth doing: the grading harness \
(This-Directory-Only/backend/app/code_runner.py) decides what a correct answer has to look \
like, and neighbouring questions in the same subtopic show the house style. You cannot edit \
any file, run any command, or reach the network. Your answer is the JSON object, and the \
tooling applies it.

Rules for the fields you may return:
- question_text: plain prose plus LaTeX ($...$) as the bank already uses. State the task, \
the shapes involved, and what to return. Never include the answer or an outline of it.
- starter_code: must be valid Python and must keep the same entry point the grader calls \
(usually `def solve(...)`). It sets up fixtures; it must not contain the solution.
- answer_code: the reference solution. Only return this if the complaint is that the \
answer itself is wrong. It will be re-run against the question's real test cases before \
it is accepted, so it must actually pass them.

Return a field only when you are changing it; return an empty string for every field you \
are leaving alone. If the complaint does not describe a real defect, or you cannot tell \
what the defect is, set verdict to "no_change" and say why in rationale.

Keep rationale to one or two sentences.\
"""

# Passed to `claude --json-schema`. The CLI validates the model's answer against
# this and hands it back on `.structured_output`, so the runner never parses
# prose. Kept as a plain dict rather than a pydantic model because the runner
# ships it straight to the CLI as JSON.
REPAIR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["rewrite", "no_change"],
            "description": "rewrite if you are changing the question, no_change otherwise",
        },
        "rationale": {
            "type": "string",
            "description": "One or two sentences on what was wrong and what you changed.",
        },
        "question_text": {"type": "string"},
        "starter_code": {"type": "string"},
        "answer_code": {"type": "string"},
    },
    "required": ["verdict", "rationale", "question_text", "starter_code", "answer_code"],
    "additionalProperties": False,
}

EDITABLE_FIELDS = feedback_ai_layer.EDITABLE_FIELDS


def allowlist() -> set[str]:
    raw = os.environ.get("DELTA_FEEDBACK_AI_EMAILS", _DEFAULT_ALLOWLIST)
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_allowlisted(email: Optional[str]) -> bool:
    return bool(email) and email.strip().lower() in allowlist()


def is_enabled_for(email: Optional[str]) -> bool:
    """True when this learner's feedback should queue a repair.

    Allowlist only. There is nothing else to check server-side any more: the
    model runs on Seth's machine, so whether the loop can actually complete
    depends on whether his runner is up, not on server configuration. A job
    queued with no runner listening simply waits.
    """
    return is_allowlisted(email)


def is_actionable_tag(tag: str) -> bool:
    """Whether this flag describes a defect worth queueing a repair for."""
    return tag in _ACTIONABLE_TAGS


def question_snapshot(question: questions.Question) -> Dict[str, object]:
    """Everything the runner needs about a question, without a bank import.

    The runner usually has the bank (it runs in the repo), but a remote runner
    talking to production over HTTP does not, and the job it is handed must
    describe the question as PRODUCTION currently serves it — not as the local
    checkout would build it.
    """
    return {
        "id": question.id,
        "topic": question.topic,
        "subtopic": question.subtopic,
        "difficulty_label": question.difficulty_label,
        "difficulty_score": question.difficulty_score,
        "submission_mode": question.submission_mode,
        "function_name": question.function_name,
        "question_text": question.question_text or "",
        "starter_code": question.starter_code or "",
        "answer_code": question.answer_code or "",
        "expected_output": question.expected_output or "",
        "test_cases": question.test_cases or [],
    }


def build_prompt(snapshot: Dict[str, object], tag: str, note: str, correct: Optional[bool]) -> str:
    """The user turn for the local session. Takes a snapshot, not a Question."""
    learner_note = (note or "").strip() or "(no note — only the flag)"
    outcome = (
        "The learner was graded CORRECT on this attempt."
        if correct
        else "The learner was graded INCORRECT on this attempt."
        if correct is False
        else "The attempt outcome was not recorded."
    )
    return f"""\
{_TAG_GUIDANCE.get(tag, "The learner flagged a problem with this question.")}

Learner's note: {learner_note}
{outcome}

--- QUESTION {snapshot.get('id')} ---
Topic: {snapshot.get('topic')}
Subtopic: {snapshot.get('subtopic')}
Difficulty: {snapshot.get('difficulty_label')} ({snapshot.get('difficulty_score')})
Submission mode: {snapshot.get('submission_mode')}
Graded entry point: {snapshot.get('function_name') or '(stdout, no function)'}

question_text:
{snapshot.get('question_text')}

starter_code:
{snapshot.get('starter_code') or "(none)"}

answer_code:
{snapshot.get('answer_code')}

expected_output:
{snapshot.get('expected_output')}
--- END QUESTION ---
"""


def validated_changes(
    repair: Dict[str, object],
    snapshot: Dict[str, object],
    tag: str,
) -> Dict[str, str]:
    """Keep only the fields that are a real, safe, actually-different change.

    Runs in the runner before anything is sent AND on the server before anything
    is written. Duplicated work on purpose — the runner's copy gives a fast, in
    terminal answer, the server's copy is the one that actually guards the bank,
    because the endpoint is reachable without going through the runner at all.
    """
    if str(repair.get("verdict", "")).strip() != "rewrite":
        return {}

    question_id = snapshot.get("id")
    current = {field: str(snapshot.get(field) or "") for field in EDITABLE_FIELDS}
    proposed = {field: str(repair.get(field) or "").strip() for field in EDITABLE_FIELDS}

    changes: Dict[str, str] = {}
    for field, value in proposed.items():
        if not value or value == current[field].strip():
            continue
        # The reference answer decides whether every future attempt is marked
        # right or wrong, so only a "this is wrong" flag may reach it.
        if field == "answer_code" and tag != "broken":
            logger.info("feedback_ai: dropped answer_code rewrite for q=%s (tag=%s)", question_id, tag)
            continue
        if field in ("starter_code", "answer_code"):
            try:
                compile(value, f"<feedback_ai q{question_id} {field}>", "exec")
            except SyntaxError as exc:
                logger.warning("feedback_ai: %s for q=%s did not compile: %s", field, question_id, exc)
                continue
        changes[field] = value
    return changes


def verify_answer_code(answer_code: str, test_cases) -> tuple:
    """Run a rewritten reference answer against the question's own test cases.

    Returns (ok, detail). This is the gate a compile check cannot stand in for:
    a syntactically perfect wrong answer marks every future learner wrong on
    that question, and nothing downstream would notice.

    Called in TWO places on purpose. The runner calls it so a failure can be fed
    back into a second attempt while the session is still cheap to redo, and
    apply_repair calls it again because the completion endpoint accepts a
    rewrite from anything holding an allowlisted token — including a runner that
    skipped verification, or a hand-rolled curl.

    A question with no test cases is unverifiable here (stdout-graded drills), so
    it passes and the compile check in validated_changes is what stands.
    """
    if not test_cases:
        return True, "no test cases to verify against"

    from app.code_runner import code_uses_torch, preload_torch, run_function_tests

    # The API preloads torch at startup so the fork runner can grade in-process;
    # a script does not, and without it the harness declines every torch drill
    # with a "open this in Colab" message that reads exactly like a failing
    # test. The bank is 100% torch, so skipping this rejects everything.
    if code_uses_torch(answer_code) and not preload_torch():
        return False, "torch is unavailable here — cannot verify a torch answer"

    try:
        results, execution = run_function_tests(answer_code, list(test_cases))
    except Exception as exc:  # a harness crash is a failed verification, not a pass
        return False, f"grading harness raised: {exc}"

    if getattr(execution, "error", ""):
        return False, f"answer failed to run: {str(execution.error)[:300]}"
    failures = [r for r in results if not r.passed]
    if failures:
        first = failures[0]
        return False, (
            f"{len(failures)}/{len(results)} test cases failed; "
            f"first expected {first.expected!r} got {first.actual!r} {first.error[:200]}"
        )
    return True, f"{len(results)}/{len(results)} test cases passed"


def enqueue_repair(
    question_id: int,
    tag: str,
    note: str,
    correct: Optional[bool],
    user_email: str,
) -> Optional[dict]:
    """Queue one flagged question for the local runner. None if not actionable."""
    if not is_actionable_tag(tag):
        return None
    return feedback_repair_queue.enqueue(
        question_id=question_id,
        tag=tag,
        note=note,
        correct=correct,
        user_email=user_email,
    )


def apply_repair(
    question_id: int,
    repair: Dict[str, object],
    *,
    tag: str,
    trigger: Dict[str, object],
    model: str,
    session_id: str = "",
) -> Optional[dict]:
    """Validate a repair against the LIVE question and write it into the bank.

    Returns the revision entry, or None when nothing survived the gates — which
    is a normal outcome, not an error: a `no_change` verdict and a rewrite that
    turned out identical both land here.
    """
    question = questions.get_question_by_id(question_id)
    if question is None:
        logger.warning("feedback_ai: question %s not in the bank — skipping", question_id)
        return None

    snapshot = question_snapshot(question)
    changes = validated_changes(repair, snapshot, tag)

    verification = ""
    if "answer_code" in changes:
        ok, verification = verify_answer_code(changes["answer_code"], snapshot.get("test_cases"))
        if not ok:
            # Drop the answer, keep any prose fix. A rewritten prompt is still
            # worth having; a reference answer that fails the question's own
            # tests is the one change that can never be worth having.
            logger.warning(
                "feedback_ai: rejected answer_code for q=%s — %s", question_id, verification,
            )
            changes.pop("answer_code")

    if not changes:
        return None

    entry = feedback_ai_layer.apply_override(
        question_id,
        changes,
        before={field: str(snapshot[field]) for field in changes},
        trigger={
            **trigger,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            # Persisted with the revision, not bolted onto the return value —
            # the audit log is what someone reads months later to find the
            # conversation that produced a question they are staring at.
            "session_id": session_id,
            "verification": verification,
        },
        model=model,
        rationale=str(repair.get("rationale", "")),
    )
    if session_id:
        entry["session_id"] = session_id

    # Rebuild the in-memory bank so the repair is live for the next draw
    # instead of waiting for a restart.
    try:
        questions.reload_questions()
    except Exception as exc:  # the layer is already on disk; a restart picks it up
        logger.exception("feedback_ai: applied q=%s but reload failed: %s", question_id, exc)
    return entry
