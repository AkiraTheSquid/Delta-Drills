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
    the Integrated rung never (Seth's torch.aggregations, 2026-09-11: q497
    missed, five left, `required` 6). These are the drills that come back once
    the rung has nothing unseen left — the learner has seen the reference
    answer for each, and the retry is the evidence the gate is waiting on.
    """
    return {qid for qid, correct in _latest_outcomes(user_state).items() if not correct}


def aided_correct_question_ids(user_state) -> set:
    """Every drill whose LATEST attempt was correct BEHIND an example.

    The third kind of spent-but-owed drill, beside the misses above. Promotion
    off Solo needs six DISTINCT correct answers made WITHOUT an example
    (solo_progress.successful_questions), and the example schedule shows one
    on Solo positions 0, 2, 5 and 9 plus the drill after every miss
    (example_schedule.SCHEDULE). A drill answered correctly behind one of
    those is spent for the unseen-first order — it was answered — but it can
    never count, and unlike a miss nothing brought it back. So a nine-drill
    bank had at most six countable drills and one wrong answer anywhere made
    the gate unreachable; a six-drill bank could never clear it at all.
    Measured by replaying Seth's state (traj.py, 2026-09-18): every one of
    the course's 409s at the Solo rung was this, and a Monte Carlo of the
    schedule against the gate put the dead-lock at 48% for six drills and
    10% for twelve, before any content was too thin.

    Read from `kc_ladder`, not history: the `example` flag lives only there.
    That row is windowed to the last 20 attempts per concept, the same window
    `solo_progress` counts from, so what this forgets the gate has forgotten
    too. Latest attempt per question wins, so a drill retaken unaided leaves
    the set, whether or not the retake was right.

    Integrated drills count too: the top rung opens behind an example
    (SCHEDULE["solo"] position 0), and `kc_graph.kc_evidence_exhausted` calls
    a concept learned only when every servable drill's latest answer was
    given unaided — the aided entry drill is owed its retake, or the rung is
    spent with the concept unlearned and nothing to serve (replay 2026-09-18,
    torch.boolean-masking). Placement probes are not ladder attempts and the
    lesson rung holds no drill, so every attempt here is at a drill rung.
    """
    latest: Dict[int, dict] = {}
    for row in (getattr(user_state, "kc_ladder", None) or {}).values():
        for attempt in (row.get("attempts") if isinstance(row, dict) else None) or ():
            qid = attempt.get("question_id")
            if qid is not None:
                latest[int(qid)] = attempt
    return {qid for qid, a in latest.items() if a.get("correct") and a.get("example")}


def unaided_success_question_ids(user_state) -> set:
    """Every drill whose LATEST attempt was correct WITHOUT an example on
    screen — the learner's own evidence. Read from `kc_ladder` like
    `aided_correct_question_ids` (the `example` flag lives only there).

    This is the debt an aided answer leaves (2026-09-19): not a retake of
    that drill, but at least ONE of these on the concept before the concept
    can count as learned or its Solo rung as cleared. Without it a concept
    whose every servable drill was answered behind the entry example — a
    one-drill concept, or a legacy record — left the frontier on the
    example alone (codex, 2026-09-19).
    """
    latest: Dict[int, dict] = {}
    for row in (getattr(user_state, "kc_ladder", None) or {}).values():
        for attempt in (row.get("attempts") if isinstance(row, dict) else None) or ():
            qid = attempt.get("question_id")
            if qid is not None:
                latest[int(qid)] = attempt
    return {qid for qid, a in latest.items() if a.get("correct") and not a.get("example")}


def owed_question_ids(user_state) -> set:
    """Answered drills the concept still has a claim on: the MISSES. Spent for
    the unseen-first order, unspent for "does this concept still have work" —
    the picker's readers of `answered` (prioritization.select_next_subtopic,
    narrow_to_next_kc) subtract this set before deciding a rung is done.

    Until 2026-09-19 this also held the drills answered correctly behind an
    example (`aided_correct_question_ids`), each owed an unaided retake of
    THE SAME DRILL. Measured in replay that was 37% of every pick — the
    single largest kind — and it is the one practice condition the
    literature agrees is weakest: immediate, massed retrieval of an item the
    learner just saw answered (van Gog & Sweller 2015 and Karpicke & Aue 2015
    disagree about everything else). An aided answer now owes the CONCEPT an
    unaided success, and any fresh drill supplies it; only a miss brings a
    drill back, and only once nothing fresh is left (remediation.py).
    """
    return missed_question_ids(user_state)


# How many graded attempts, on ANY concept, must pass before an owed drill is
# handed back. Replay of Seth's state with the retakes wired in (traj.py,
# 2026-09-18): with nothing else unseen on the concept the same drill came
# straight back after every miss and every aided answer — q972 seventeen times
# in a row, q839 sixteen — which is the "same problem forever" Seth reported on
# 2026-08-28, now on a loop of one. Three is enough for the after-miss example
# and the retake to be separated by other work; it is not a spacing schedule.
RETAKE_COOLDOWN = 3


def recent_question_sequence(user_state, n: int) -> list:
    """The last `n` drills the learner answered, oldest first, across every
    subtopic — with repeats, so a caller can tell "three answers on one
    concept" from "one answer". Read from `SubtopicState.history` ordered by
    timestamp (ties keep append order), the same untruncated record
    `answered_question_ids` reads."""
    records = []
    for sub_state in (getattr(user_state, "subtopic_states", None) or {}).values():
        for i, record in enumerate(getattr(sub_state, "history", None) or ()):
            qid = getattr(record, "question_id", None)
            if qid is not None:
                records.append((getattr(record, "timestamp", None) or "", i, int(qid)))
    records.sort(key=lambda r: (str(r[0]), r[1]))
    return [qid for _, _, qid in records[-n:]] if n > 0 else []


def recent_question_ids(user_state, n: int = RETAKE_COOLDOWN) -> set:
    """The last `n` drills the learner answered, as a set."""
    return set(recent_question_sequence(user_state, n))


def retakeable_question_ids(user_state) -> set:
    """Owed drills that are not in their cooldown: the ones the queue should
    serve first. `owed_question_ids` minus `recent_question_ids`. Callers fall
    back to the full owed set when nothing else is left — a retake now beats
    a 409 — so the cooldown orders work, it never withholds it."""
    return owed_question_ids(user_state) - recent_question_ids(user_state)

