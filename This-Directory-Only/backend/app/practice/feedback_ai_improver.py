"""
Feedback -> Opus 5 -> repaired question.

When an allowlisted learner flags a question from the practice UI, this module
sends the question and the learner's note to Claude Opus 5 and writes the
repair straight into the live bank as an override layer (see
app/feedback_ai_layer.py). It runs behind FastAPI's BackgroundTasks, so the
learner's POST returns immediately and never waits on the model.

It hangs off the one endpoint every exercise's quality feedback already posts
to (problem_feedback_router.py), which is why it covers the whole bank —
numpy, einsum, einops, torch, CNN, curated additions — without a per-course
hook anywhere.

Three things keep an auto-applied rewrite from being a one-way door:

  Narrow surface.  Only question_text / starter_code / answer_code can change.
                   Curriculum placement, difficulty, test_cases and atom tags
                   feed the adaptive engine and the KC lattice, so the model
                   never touches them and a bad rewrite can't move a question
                   in the mastery graph.
  Gates.           Code fields must compile, answer_code only opens for a
                   `broken` flag, and a rewrite that comes back identical or
                   empty is dropped rather than written.
  Reversibility.   Every apply logs its own before/after to an append-only
                   revision log, and any single question can be reverted to
                   the shipped text with one rollback call.

Configuration (all via env; nothing is hardcoded):
  ANTHROPIC_API_KEY        required — without it the loop stays dormant and
                           feedback keeps logging as before.
  DELTA_FEEDBACK_AI_EMAILS comma-separated allowlist. Defaults to Seth.
  DELTA_FEEDBACK_AI_DIR    where the override layer + revision log are written
                           (the Fly /data volume in production).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel

from app import feedback_ai_layer, questions

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# Who may trigger a rewrite. A flag from anyone else is still logged by the
# router — it just doesn't spend a model call or move the bank.
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

Rules for the fields you may return:
- question_text: plain prose plus LaTeX ($...$) as the bank already uses. State the task, \
the shapes involved, and what to return. Never include the answer or an outline of it.
- starter_code: must be valid Python and must keep the same entry point the grader calls \
(usually `def solve(...)`). It sets up fixtures; it must not contain the solution.
- answer_code: the reference solution. Only return this if the complaint is that the \
answer itself is wrong.

Return a field only when you are changing it; return an empty string for every field you \
are leaving alone. If the complaint does not describe a real defect, or you cannot tell \
what the defect is, set verdict to "no_change" and say why in rationale.

Keep rationale to one or two sentences.\
"""


class QuestionRepair(BaseModel):
    """Structured rewrite. Empty string = leave that field untouched."""

    verdict: str  # "rewrite" | "no_change"
    rationale: str
    question_text: str
    starter_code: str
    answer_code: str


def allowlist() -> set[str]:
    raw = os.environ.get("DELTA_FEEDBACK_AI_EMAILS", _DEFAULT_ALLOWLIST)
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_enabled_for(email: Optional[str]) -> bool:
    """True when this learner's feedback should fire a repair.

    Requires the allowlist AND a usable client. Missing credentials are a
    dormant feature, not an error — practice keeps working either way.
    """
    if not email or email.strip().lower() not in allowlist():
        return False
    # Either credential the SDK accepts; never a literal in the repo.
    if not any(os.environ.get(name, "").strip() for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")):
        logger.info("feedback_ai: allowlisted user but no Anthropic credential set — skipping")
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        logger.warning("feedback_ai: anthropic package not installed — skipping")
        return False
    return True


def is_actionable_tag(tag: str) -> bool:
    """Whether this flag describes a defect worth spending a model call on."""
    return tag in _ACTIONABLE_TAGS


def _build_prompt(question: questions.Question, tag: str, note: str, correct: Optional[bool]) -> str:
    learner_note = note.strip() or "(no note — only the flag)"
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

--- QUESTION {question.id} ---
Topic: {question.topic}
Subtopic: {question.subtopic}
Difficulty: {question.difficulty_label} ({question.difficulty_score})
Submission mode: {question.submission_mode}

question_text:
{question.question_text}

starter_code:
{question.starter_code or "(none)"}

answer_code:
{question.answer_code}

expected_output:
{question.expected_output}
--- END QUESTION ---
"""


def _call_model(prompt: str) -> Optional[QuestionRepair]:
    """One structured Opus 5 call. Returns None on refusal or transport error."""
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=QuestionRepair,
        )
    except anthropic.APIError as exc:
        logger.error("feedback_ai: Anthropic call failed: %s", exc)
        return None

    # Check the refusal stop reason before touching content — on a refusal the
    # content blocks are not the rewrite.
    if response.stop_reason == "refusal":
        logger.warning("feedback_ai: model declined to rewrite the question")
        return None
    return response.parsed_output


def _validated_changes(
    repair: QuestionRepair,
    question: questions.Question,
    tag: str,
) -> Dict[str, str]:
    """Keep only the fields that are a real, safe, actually-different change."""
    current = {
        "question_text": question.question_text or "",
        "starter_code": question.starter_code or "",
        "answer_code": question.answer_code or "",
    }
    proposed = {
        "question_text": repair.question_text.strip(),
        "starter_code": repair.starter_code.strip(),
        "answer_code": repair.answer_code.strip(),
    }

    changes: Dict[str, str] = {}
    for field, value in proposed.items():
        if not value or value == current[field].strip():
            continue
        # The reference answer decides whether every future attempt is marked
        # right or wrong, so only a "this is wrong" flag may reach it.
        if field == "answer_code" and tag != "broken":
            logger.info("feedback_ai: dropped answer_code rewrite for q=%s (tag=%s)", question.id, tag)
            continue
        if field in ("starter_code", "answer_code"):
            try:
                compile(value, f"<feedback_ai q{question.id} {field}>", "exec")
            except SyntaxError as exc:
                logger.warning("feedback_ai: %s for q=%s did not compile: %s", field, question.id, exc)
                continue
        changes[field] = value
    return changes


def improve_question_from_feedback(
    question_id: int,
    tag: str,
    note: str,
    correct: Optional[bool],
    user_email: str,
) -> Optional[dict]:
    """Repair one flagged question and write the result into the live bank.

    Runs as a background task. Every failure path is a log line and a return —
    a feedback submission must never surface an error to the learner because
    the model was unavailable or unhelpful.
    """
    if tag not in _ACTIONABLE_TAGS:
        return None

    question = questions.get_question_by_id(question_id)
    if question is None:
        logger.warning("feedback_ai: question %s not in the bank — skipping", question_id)
        return None

    before = {
        "question_text": question.question_text or "",
        "starter_code": question.starter_code or "",
        "answer_code": question.answer_code or "",
    }

    try:
        repair = _call_model(_build_prompt(question, tag, note, correct))
    except Exception as exc:  # background task: never let this escape
        logger.exception("feedback_ai: unexpected failure for q=%s: %s", question_id, exc)
        return None

    if repair is None:
        return None
    if repair.verdict != "rewrite":
        logger.info("feedback_ai: no_change for q=%s — %s", question_id, repair.rationale)
        return None

    changes = _validated_changes(repair, question, tag)
    if not changes:
        logger.info("feedback_ai: nothing applicable for q=%s — %s", question_id, repair.rationale)
        return None

    entry = feedback_ai_layer.apply_override(
        question_id,
        changes,
        before={field: before[field] for field in changes},
        trigger={
            "user_email": user_email,
            "tag": tag,
            "note": note,
            "correct": correct,
            "flagged_at": datetime.now(timezone.utc).isoformat(),
        },
        model=MODEL,
        rationale=repair.rationale,
    )

    # Rebuild the in-memory bank so the repair is live for the next draw
    # instead of waiting for a restart.
    try:
        questions.reload_questions()
    except Exception as exc:  # the layer is already on disk; a restart picks it up
        logger.exception("feedback_ai: applied q=%s but reload failed: %s", question_id, exc)
    return entry
