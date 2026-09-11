"""What the learner's graded history says about each drill.

`SubtopicState.history` is the durable, untruncated record of evidence — one
entry per graded attempt, carrying the question id. Read here, not
`kc_ladder`, whose attempt list is windowed to the last 20
(`kc_graph._LADDER_WINDOW`) and so forgets that an older drill was ever solved.
Split out of `prioritization` (2026-09-11) for size only; that module
re-exports both public names.
"""
from __future__ import annotations

from typing import Dict

def _latest_outcomes(user_state) -> Dict[int, bool]:
    """question id -> whether its LATEST graded attempt was correct.

    One pass over `SubtopicState.history`, the durable untruncated record of
    evidence (see `answered_question_ids` for why history and not `kc_ladder`).
    History is appended per attempt, so the last record seen for a question is
    its latest; read across every subtopic because a question's subtopic is
    fixed but the caller does not know which one it is filed under.
    """
    latest: Dict[int, bool] = {}
    for sub_state in (getattr(user_state, "subtopic_states", None) or {}).values():
        for record in getattr(sub_state, "history", None) or ():
            qid = getattr(record, "question_id", None)
            if qid is not None:
                latest[int(qid)] = bool(getattr(record, "correct", False))
    return latest


def answered_question_ids(user_state) -> set:
    """Every question this learner has actually ANSWERED, across all subtopics.

    NOT the same set as `served_question_ids`, and the difference is the whole
    of the 2026-08-31 bug. `served` is appended the moment `/next-question`
    hands a drill over — before the learner has read it, let alone answered it.
    A skip, a reload, a double-fetch, or simply closing the tab therefore SPENT
    a drill permanently: it could never be offered again, and the rung it sat on
    counted it as done. Seth's account reached the state this exists to prevent
    on 2026-08-31 — `python.values-and-names`, the course's only root, holds two
    drills at its lowest authored rung, both were served and NEITHER was ever
    answered, and every `/next-question` for the next half hour 409'd
    "you have finished every lesson problem for this concept" (15 recorded hits
    in content-gaps.json). With that one root bricked, every other concept in
    the course stayed locked behind it.

    Evidence is what spends a drill. `SubtopicState.history` is the durable,
    untruncated record of it — one entry per graded attempt, carrying the
    question id — which is why it is read here rather than `kc_ladder`, whose
    attempt list is windowed to the last 20 (`kc_graph._LADDER_WINDOW`) and so
    forgets that an older drill was ever solved.
    """
    return set(_latest_outcomes(user_state))


def missed_question_ids(user_state) -> set:
    """Every question whose LATEST graded attempt was a miss.

    A miss spends a drill the same as a correct answer does
    (`answered_question_ids`), and on the Solo rung that spent drill is still
    owed: promotion off it needs six DISTINCT correct answers
    (solo_progress.progress), counted on the latest attempt per question, so a
    bank of exactly six drills with one miss in it could reach five at most and
    the Integrated rung never (Seth's numpy.aggregations, 2026-09-11: q497
    missed, five left, `required` 6). These are the drills that come back once
    the rung has nothing unseen left — the learner has seen the reference
    answer for each, and the retry is the evidence the gate is waiting on.
    """
    return {qid for qid, correct in _latest_outcomes(user_state).items() if not correct}
