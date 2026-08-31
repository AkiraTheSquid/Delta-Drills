"""When a worked example pops up in front of a drill, and how it fades.

The ladder decides the RUNG (kc_graph.kc_stage). This decides, on the drill
rungs, whether the drill is served behind an example the learner reads first —
the worked-example effect, faded out as expertise grows. Seth, 2026-08-30:
"it shows you an example every once in a while ... and then you see the
examples less and less ... in the third stage it should really start to fade
out ... and you should really be tested before you move on."

An EXPERIMENT. The numbers here will be re-tuned from real practice, so they
live in one table and nothing reads them except `plan`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Per stored rung name (kc_graph.LADDER_STAGES): `at` is the set of positions —
# attempts made in a row at this rung, counted from 0 — on which the example is
# shown; `after_miss` shows one right after a wrong answer at this rung,
# provided that wrong answer was not itself made behind an example (so two
# misses in a row buy one example, not two). `faded` has no scheduled example
# on purpose: the learner has just read the lesson, and an example beside the
# blanks spells them out (the q484 defect). `solo` shows one on entry and
# then none — that rung is the test.
SCHEDULE: Dict[str, dict] = {
    "faded": {"at": (), "after_miss": True},
    "partial": {"at": (0, 2, 5, 9), "after_miss": False},
    "solo": {"at": (0,), "after_miss": False},
}

# How many of the learner's most recent Integrated-rung answers must have been
# made WITHOUT an example before a small-pool concept counts as learned
# (kc_graph.kc_evidence_exhausted). Two: the entry example is position 0, so
# this is "answered the two after it unaided".
UNAIDED_TO_FINISH = 2


def aided(attempt: dict) -> bool:
    """Was this attempt made behind a worked-example popup?

    Attempts recorded before 2026-08-30 carry no `example` key at all, and the
    popup did not exist then, so a missing key reads as unaided — which is what
    those rows actually were.
    """
    return bool(attempt.get("example"))


def unaided(attempts: List[dict]) -> List[dict]:
    """`attempts` with the aided ones dropped — the record of what they can do alone."""
    return [a for a in (attempts or []) if not aided(a)]


def unaided_run(attempts: List[dict]) -> List[dict]:
    """The trailing run of correct answers, counting only the unaided ones.

    An aided answer is NEUTRAL: it neither counts toward the run nor breaks it.
    Breaking on it would make this table fight the ladder — the example at Solo
    position 2 is shown by the system, not asked for, so it would reset a run
    the learner did nothing wrong in. Counting it would let a run promote on
    assistance, which is the thing the unaided rule exists to stop. A WRONG
    answer still breaks the run whether or not an example was on screen.
    """
    run = []
    for attempt in reversed(attempts or []):
        if not attempt.get("correct"):
            break
        if aided(attempt):
            continue
        run.append(attempt)
    return run


def position(attempts: List[dict], stage: str) -> int:
    """How many attempts in a row the learner has made at `stage`, ending now.

    Trailing, not total: re-entering a rung after a demotion starts again at
    0, which is what makes "an example when you come back to it" fall out of
    the same table as "an example when you first arrive".
    """
    n = 0
    for attempt in reversed(attempts or []):
        if attempt.get("stage") != stage:
            break
        n += 1
    return n


def plan(attempts: List[dict], stage: Optional[str]) -> dict:
    """Should the next drill at `stage` open with an example?

    Pure: reads the concept's ladder attempts and the rung, returns
    `{"show", "why", "position"}`. Called at serve time to decide, and again
    at record time (before the attempt is appended, like the rung itself) so
    the stored `example` flag is the same decision the learner saw.
    """
    spec = SCHEDULE.get(stage or "")
    if not spec:
        return {"show": False, "why": None, "position": 0}
    pos = position(attempts, stage)
    last = attempts[-1] if attempts else None
    if (
        spec["after_miss"]
        and last is not None
        and not last.get("correct")
        and not last.get("example")
    ):
        return {"show": True, "why": "after_miss", "position": pos}
    if pos in spec["at"]:
        return {"show": True, "why": "scheduled", "position": pos}
    return {"show": False, "why": None, "position": pos}


def unaided_finish(attempts: List[dict], stage: str = "solo") -> bool:
    """Were the last UNAIDED_TO_FINISH answers at `stage` correct AND unaided?

    The clause that keeps the entry example from being the whole test: a
    concept whose pool is spent still has to be answered without help, twice,
    on the top rung before the ladder calls it learned.
    """
    tail = [a for a in (attempts or [])[-UNAIDED_TO_FINISH:]]
    if len(tail) < UNAIDED_TO_FINISH:
        return False
    return all(
        a.get("stage") == stage and a.get("correct") and not a.get("example")
        for a in tail
    )
