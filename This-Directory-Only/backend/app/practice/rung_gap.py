"""The gap dict for a rung that has nothing NEW left to put on screen.

Lifted out of prioritization.py (2026-09-20, Modulario RED at 671 LOC); its
only caller is practice/question_pick.py, on the on-screen guard. The shape it
returns is the one content_gaps.record stores and content_gaps.learner_message
reads.
"""
from __future__ import annotations

from typing import Optional

from app import kc_graph
from app.adaptive import UserPracticeState
from app.attempt_history import answered_question_ids
from app.prioritization import question_is_unlocked
from app.questions import get_question_by_id


def rung_gap(user_state: UserPracticeState, kc: Optional[str], question) -> dict:
    """The gap dict for a rung that has nothing NEW left to put on screen.

    `narrow_to_next_kc` builds the same shape inline when a rung is spent —
    every drill on it ANSWERED. This builds it for the other way a rung runs
    out, which that check cannot see: every drill already SERVED, so the picker
    can only hand back something the learner has already looked at. The caller
    (practice/questions_router) decides when that has happened; this only names
    it, in the shape `content_gaps.record` stores and `content_gaps
    .learner_message` reads.

    `kc` may be None — an untagged question has no concept and no rung, and the
    message falls back to "this concept" rather than inventing one.
    """
    if not kc:
        return {
            "kc": None,
            "kc_title": getattr(question, "subtopic", None) or "this concept",
            "stage": None,
            "seen": None,
            "answered": None,
            "total": None,
            "served_from": None,
        }
    node = kc_graph.registry_node(kc) or {}
    stage = kc_graph.kc_stage(user_state, kc)
    owned = list(kc_graph.questions_for_kc(kc))
    at_stage = set(kc_graph.questions_at_stage(owned, stage))
    if not at_stage and stage == "worked":
        # `worked` serves no rank of its own — narrow_to_next_kc falls back to
        # the concept's floor rung, narrowed to what this learner may be shown
        # (a segment's carrier stays locked until its page is read). Count THAT.
        # Counting the stage's own (empty) rank told Seth "nothing is written at
        # this rung" for torch.linalg-basics on 2026-09-20, when the truth was
        # that its one unlocked floor drill (q239) was the drill on screen.
        at_stage = {
            qid for qid in kc_graph.lowest_rung(owned)
            if (q := get_question_by_id(qid)) is not None and question_is_unlocked(user_state, q)
        }
    # `seen` counts what this rung ACTUALLY holds, and 0 is a real answer, not a
    # missing one: `einops.pattern-language` owns twelve drills and none is
    # tagged `worked`, which is why the picker had a pool of one to rotate
    # through. Falling back to `len(owned)` here would have told the learner
    # they had finished twelve lesson problems that were never written.
    # content_gaps.learner_message reads the 0 and says so.
    #
    # 🔴 SEEN IS NOT ANSWERED, and this gap is raised by the SERVED-out path:
    # the router 409s exactly when the repeat is a question the learner has NOT
    # answered. So a rung of three drills that were all skipped arrived here as
    # seen=3 and came back out of learner_message as "you have finished every
    # lesson problem, all 3 of them" — the same lie the zero branch above was
    # written to kill, one size smaller. Carry the answered count so the message
    # can say which of the two actually happened. (codex, 2026-09-07.)
    return {
        "kc": kc,
        "kc_title": node.get("title") or kc,
        "stage": stage,
        "seen": len(at_stage),
        "answered": len(at_stage & answered_question_ids(user_state)),
        "total": len(owned),
        "served_from": None,
    }
