"""Finalizing a graded attempt — the one place scoring actually happens.

WHY THIS IS ITS OWN MODULE

`record_attempt` does not score anything. It parks the attempt in
`user_state.pending_attempt` and returns, because in the normal flow the
learner is about to be asked how hard it felt, and correctness is not final
until any `/override` has had its chance. Everything that MOVES —
`sub_state.n`, the attempt's place in history, the per-atom BKT posteriors, the
subtopic mastery snapshot and the target difficulty — happens on the way out,
in `apply_feedback` plus the block that used to live inline in
`feedback_router`.

That was fine while `/feedback` was the only exit. It is not the only exit any
more: the Colab edition has no felt-difficulty step (running the notebook's
checker IS the submit), so `/submit-local-eval` parked attempts that nothing
ever came back for. They sat pending until the next submit overwrote them —
silently, because the rail advanced and the grade showed. `n` never moved, no
BKT posterior moved, the difficulty target never moved, and the concept graph
reported a learner who had done nothing.

So the exit is shared code now rather than one router's tail. A second copy of
this sequence would be worse than the bug it fixed: two places deciding what an
answer is worth, drifting apart one commit at a time.

ORDER IS LOAD-BEARING. The BKT update runs BEFORE the mastery snapshot, so the
snapshot on the attempt record reflects post-attempt mastery and the learning-
rate chart plots a trajectory rather than a lag-one copy of itself.
"""

from __future__ import annotations

from typing import Optional

from app import bkt_mastery
from app.adaptive import (
    UNRATED,
    AttemptRecord,
    FeedbackLevel,
    apply_feedback,
    nudge_difficulty_offset,
)
from app.prioritization import question_target_difficulty, subtopic_mastery
from app.questions import get_question_by_id


def flush_stale_attempt(user_state) -> Optional[AttemptRecord]:
    """Close out an attempt nothing is coming back for, as UNRATED.

    Call this at the top of any route that is about to record a NEW attempt.
    `record_attempt` overwrites `user_state.pending_attempt` outright, so an
    attempt left parked by an earlier request is not merely un-rated, it is
    gone: `n` never moves, no posterior moves, and the concept graph reports a
    learner who has been practising all week as having answered nothing. That
    is the 2026-08-03 bug, and it is reachable from anything that grades and
    then never reaches /feedback — a Skip, a closed tab, a client running half a
    deploy behind the backend it is talking to.

    Cannot double-count: this commits the PREVIOUS attempt, and the rating for
    the current one still finds its own pending record. Mirrors the flush the
    offline twin does inside its own `record_attempt`
    (Local_Deployed_Shared/practice_engine.py) — it can do it there because it
    has no BKT layer to reach up into.
    """
    if user_state.pending_attempt is None:
        return None
    return finalize_attempt(user_state, UNRATED)


def finalize_attempt(
    user_state,
    feedback: FeedbackLevel,
) -> Optional[AttemptRecord]:
    """Close out the pending attempt: history, BKT, mastery, difficulty.

    Returns the finalized attempt, or None when there was nothing pending —
    which is not an error at every call site, so the decision about what to do
    with a None is left to the caller.
    """
    attempt = apply_feedback(user_state, feedback)
    if attempt is None:
        return None

    # Per-atom BKT update — the real mastery signal. Each of the question's atom
    # tags updates its posterior scaled by that tag's confidence, then
    # FIRe-credits the atoms it encompasses.
    question = get_question_by_id(attempt.question_id)
    if question is not None:
        # params carry the learner's self-reported prior so a never-practiced
        # atom's FIRST update starts from that prior (and decay regresses toward
        # it) — one wrong answer still drops a "strong" prior fast.
        user_params = bkt_mastery.params_for_level(user_state.self_reported_level)
        for tag in getattr(question, "atom_tags", []) or []:
            bkt_mastery.apply_attempt(
                user_state.atom_mastery,
                user_state.atom_last_ts,
                tag["atom_id"],
                attempt.correct,
                params=user_params,
                confidence=float(tag.get("confidence", 1.0)),
            )

    # Snapshot the subtopic's BKT mastery into the legacy baseline/p fields the
    # Statistics panel reads (frontend unchanged): 0-1 mastery → 0-100 baseline.
    mastery = subtopic_mastery(user_state, attempt.subtopic)
    sub_state = user_state.get_subtopic_state(attempt.subtopic)
    sub_state.baseline = mastery * 100.0
    sub_state.p = mastery
    # What the learner said about how hard that felt, applied BEFORE the target
    # is recomputed — otherwise "way too easy" would only be honoured one
    # question late, which is the same as not being honoured.
    nudge_difficulty_offset(sub_state, feedback, attempt.correct)
    sub_state.target_difficulty = question_target_difficulty(
        user_state, attempt.subtopic, attempt.question_id
    )
    attempt.baseline_after = sub_state.baseline
    attempt.p_after = sub_state.p
    attempt.target_difficulty_after = sub_state.target_difficulty
    return attempt
