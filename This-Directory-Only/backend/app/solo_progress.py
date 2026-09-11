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


def next_band(questions, attempts, difficulties):
    """Use the entire authored rung to define bands, never the shrinking pool.

    Two successes in an easier band lead to harder unseen work. If a band is
    exhausted after misses, advance to available work rather than repeating
    solved questions. Skips rotate within the current band through the caller.
    """
    if not questions:
        return questions
    passed = successful_questions(attempts, difficulties)
    scores = sorted(set(difficulties.values()))
    for score in scores:
        available = [q for q in questions if q.difficulty_score == score]
        successes = sum(difficulties[q] == score for q in passed)
        if available and successes < 2:
            return available
    return questions
