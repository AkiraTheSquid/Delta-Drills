"""Distinct, unaided evidence within the existing Solo rung (stored: partial).

New banks require six successes including two from their upper difficulty
third. Legacy thin banks retain a reachable threshold and explicitly report
missing capacity; adding content must never turn old progression into a trap.
"""
from __future__ import annotations


def successful_questions(attempts, difficulties):
    latest = {}
    for attempt in attempts:
        qid = attempt.get("question_id")
        if attempt.get("stage") == "partial" and qid in difficulties:
            latest[qid] = attempt
    return {qid for qid, a in latest.items()
            if a.get("correct") and not a.get("example")}


def progress(attempts, difficulties):
    if not difficulties:
        return None
    scores = sorted(difficulties.values())
    hard = scores[min(len(scores) - 1, 2 * len(scores) // 3)]
    passed = successful_questions(attempts, difficulties)
    harder = sum(difficulties[q] >= hard for q in passed)
    required = min(6, len(difficulties))
    hard_required = min(2, sum(s >= hard for s in scores))
    return {"correct": len(passed), "required": required,
            "hard_correct": harder, "hard_required": hard_required,
            "hard_threshold": hard, "coverage_shortfall": max(0, 6-len(scores)),
            "ready": len(passed) >= required and harder >= hard_required,
            "fraction": min(1.0, len(passed)/required, harder/hard_required)}


# How far from the learner's aim a drill may sit and still be "within reach".
# ONE number for both halves of the pick: the band walk below skips bands more
# than this far under the aim, and the difficulty picker
# (practice/grading.select_question_for_difficulty, which imports it) explores
# this far either side of the aim inside the band it is handed.
REACH = 15.0


def next_band(questions, attempts, difficulties, target=None):
    """Use the entire authored rung to define bands, never the shrinking pool.

    Two successes in an easier band lead to harder unseen work. If a band is
    exhausted after misses, advance to available work rather than repeating
    solved questions. Skips rotate within the current band through the caller.

    `target` is the aim the difficulty picker will be handed next
    (prioritization.target_difficulty) — mastery plus the learner's own
    "how much harder" rating. Bands more than REACH below it are treated as
    already cleared: the learner has asked for harder work than that band
    holds, and walking them up from the bottom regardless is what made the
    rating a dead button on this rung (Seth, 2026-09-11: three "Significantly
    harder" clicks on numpy.aggregations moved the aim from 60 to 79 and the
    next drill was 54 after 52 — the bank's next band up, not the aim's). When
    the aim clears every band the walk runs DOWN from the hardest instead:
    "harder than anything here" is answered by the hardest thing here, and the
    ceiling it hits is the rung's, which is a content fact the strip can show.
    `None` keeps the plain bottom-up walk.
    """
    if not questions:
        return questions
    passed = successful_questions(attempts, difficulties)
    # Walk the bands that still HAVE something to serve, in the aim's order.
    # Walking the authored bands and returning the whole pool when none of the
    # in-reach ones had a drill left handed the picker a MIXED list, and its
    # nearest-to-aim choice could be the drill on screen (codex, 2026-09-11).
    present = sorted({q.difficulty_score for q in questions})
    if target is not None:
        within = [s for s in present if s >= target - REACH]
        present = within or present[::-1]
    for score in present:
        successes = sum(difficulties[q] == score for q in passed)
        if successes < 2:
            return [q for q in questions if q.difficulty_score == score]
    # Every band left already has its two successes: the first in walk order.
    return [q for q in questions if q.difficulty_score == present[0]]
