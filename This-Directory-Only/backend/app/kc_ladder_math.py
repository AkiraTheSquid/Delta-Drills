"""Expertise-reversal ladder arithmetic — split out of kc_graph.py (2026-09-06,
Modulario size gate). Pure functions over a KC's attempt list plus the
calibrated thresholds; no learner state, no registry. kc_graph re-exports every
name here, so `kc_graph.LADDER_STAGES`, `kc_graph._wilson`, `kc_graph._step_down`
and friends keep working for engine_bridge, prioritization and the tests.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from app import example_schedule

# ---------------------------------------------------------------------------
# Expertise-reversal ladder
#
# A concept is met with decreasing support:
#
#   worked      — the lesson page: solved examples, read not answered. Not graded.
#   partial     — write it unaided; a worked example pops up on a schedule
#                 (example_schedule) and thins out. Displayed "Solo".
#   solo        — whole-KP problems, nothing to read first. Displayed "Integrated".
#
# `faded` (the fill-in-the-blank rung) is RETIRED (Seth, 2026-09-11: "we
# shouldn't have the faded rung anymore. it just go straight to solo, but it
# still has the rate of the lessons that show different examples"). The name
# stays in LADDER_STAGES because every attempt made there is still filed under
# it and the promotion arithmetic reads those rows back; what changed is that
# no path can LAND on it any more — everything that used to floor at `faded`
# floors at DRILL_FLOOR — and `kc_graph._STAGE_TO_RANKS` no longer serves the
# faded/guided drills for it.
#
# Which rung a learner is on is decided per KC from that KC's OWN graded
# attempts, as an interval rather than a point estimate. Promotion requires the
# Wilson LOWER bound to clear PROMOTE_LO: being probably-fine is not enough to
# remove support, you have to be confidently fine. Demotion uses the UPPER
# bound, so support returns only when the learner is confidently struggling
# rather than on one unlucky answer — except for the immediate rule below,
# which is what a learner actually expects: get it wrong, see an example again.
#
# Why an interval and not a mastery cutoff. Delta-Learning's courses ladder
# (lib/courses-scaffold.ts) originally staged on a skill estimate and had to
# abandon it: faded attempts are recorded at a fixed low difficulty, so the
# skill estimate saturated below the promotion cutoff and learners were trapped
# on scaffolded cards forever. Scoring the ladder on the KC's own attempt
# record — not a global skill number the ladder itself depresses — is what
# avoids that trap here. The bounds below are calibrated so four consecutive
# correct answers promote off `partial` (Wilson lower at 4/4 = 0.51), and four
# consecutive wrong answers bring the example schedule's support back
# (Wilson upper at 0/4 = 0.49).
#
# `worked` is entered once and never re-entered — see `_stage_from`. It is the
# teaching page rather than a drill, so demotion floors at DRILL_FLOOR, the
# lowest rung that is still a problem the learner answers.
LADDER_STAGES = ("worked", "faded", "partial", "solo")
# The lowest rung a learner can be served or demoted to. `worked` is the lesson
# and `faded` is retired, so this is `partial`.
DRILL_FLOOR = "partial"
# The rungs a learner can actually stand on today, in order.
LIVE_STAGES = ("worked", "partial", "solo")

# Wilson LOWER bound needed to climb off each rung. Calibrated against the
# bound at k/k so the pacing is legible in answers, not in probabilities:
#   partial -> solo     at 0.51 = four consecutive correct
# which is Delta-Learning's PROMOTE_TO_SOLO=4 pacing, arrived at there by trial
# on real learners. Using the lower bound rather than the point estimate means
# a lucky streak of one does not strip support. (`faded -> partial` at 0.34 was
# the other entry until the faded rung was retired.)
PROMOTE_LO = {"partial": 0.51}
DEMOTE_HI = 0.50  # Wilson UPPER; 0/4 = 0.49, so four straight wrong restore support
_LADDER_WINDOW = 20  # recent attempts the estimate rests on
# Consecutive correct answers that promote a rung on their own, regardless of
# what the window average says — see `_streak_stage`. Three, to match the
# pacing the Wilson bounds were calibrated for (lower at 3/3 = 0.438) on a
# record that has no history to outvote it.
_PROMOTE_STREAK = 3


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval. Wilson rather than the normal approximation
    because n here is tiny and p sits near the edges, where the normal interval
    runs outside [0, 1] and reports far more confidence than the data supports.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _streak_toward(run: List[dict], stage: str) -> int:
    """Trailing correct answers that count toward LEAVING `stage`.

    `_streak_stage` promotes to one rung above the LOWEST rung in the run, so a
    run that already spans the promotion it bought is aiming at the rung the
    learner is already on: the streak route is spent until a miss resets the
    run. Reporting the raw run length there draws the newly reached rung
    arriving already full — three answers made at `faded` are not progress out
    of `partial`, and the strip would promise a promotion that cannot come.
    """
    ranks = [
        LADDER_STAGES.index(a["stage"])
        for a in run
        if a.get("stage") in LADDER_STAGES
    ]
    cur = LADDER_STAGES.index(stage) if stage in LADDER_STAGES else 0
    # The LENGTH counts every answer in the run and the RUNG is read only from
    # the ones filed at a ladder stage — the same split `_streak_stage` makes,
    # deliberately. `record_kc_outcome` files an off-ladder attempt (a placement
    # probe) as `independent`, and a run carrying one promotes on three exactly
    # as it does here. Counting the two differently is how a bar comes to fill
    # against a promotion that never arrives.
    return len(run) if ranks and min(ranks) >= cur else 0


def _floored(stage: Optional[str]) -> str:
    """`stage` lifted onto a rung a learner can stand on today.

    Rows filed at `faded` before it was retired, and rows with no stage at all,
    read as DRILL_FLOOR: the arithmetic that compares "the rung they are on"
    against "the rung they earned" must never see a rung that no longer exists.
    """
    if stage in LADDER_STAGES and LADDER_STAGES.index(stage) > LADDER_STAGES.index(DRILL_FLOOR):
        return stage
    return DRILL_FLOOR


def _step_down(stage: str, floor: str) -> str:
    """One rung down from `stage`, but never below `floor`.

    `floor` is required rather than defaulting to the bottom of the ladder:
    the bottom rung is the lesson page, and a caller that lands a learner
    there by accident re-teaches a concept they have already read.
    """
    i = LADDER_STAGES.index(stage) if stage in LADDER_STAGES else 1
    return LADDER_STAGES[max(LADDER_STAGES.index(floor), i - 1)]


def _step_up(stage: str, ceiling: str = "solo") -> str:
    """One rung up from `stage`, but never above `ceiling`.

    Stepping up from the retired `faded` rung lands on DRILL_FLOOR, which is
    where a run made there would have promoted to anyway.
    """
    i = LADDER_STAGES.index(stage) if stage in LADDER_STAGES else 1
    return LADDER_STAGES[min(LADDER_STAGES.index(ceiling), i + 1)]


def _streak_stage(attempts: List[dict]) -> Optional[str]:
    """The rung earned by an unbroken run of correct answers, or None.

    WHY THE WILSON BOUND IS NOT ENOUGH ON ITS OWN.
    The interval is computed over the last `_LADDER_WINDOW` attempts, and the
    calibration note above the thresholds says three consecutive correct answers
    should promote — which is true of an EMPTY record (Wilson lower at 3/3 =
    0.438) and false of every record with history in it. A learner who met a
    concept when its drills were badly written can bank a dozen misses, fix
    nothing about their own understanding except everything, and then answer
    five in a row without moving: 7/20 has a lower bound of 0.18 against a
    `faded` bar of 0.34, so the rung does not move for another dozen questions.
    That is a real record from this app, not a hypothetical.

    The asymmetry is the giveaway. Demotion already has an immediate rule — miss
    one, drop a rung, see the support again — so the ladder listens to a single
    recent answer on the way down and to a twenty-question average on the way
    up. A learner who has clearly got it is then told, question after question,
    that they have not.

    So a run of correct answers promotes one rung on its own. It reads the rung
    the run was actually MADE at (the lowest in the run, so a streak spanning a
    promotion cannot skip the rung it just reached) and steps up once from
    there. One rung, never two: the evidence is that this rung is done, not that
    the one above it is.

    Bounded on the other side by the demotion rule, which is unchanged and still
    immediate — a streak buys one rung and one miss gives it straight back.

    ASSISTANCE HOLDS A RUNG, IT DOES NOT BUY ONE. The step up is read from the
    UNAIDED run (`example_schedule.unaided_run`): an answer given behind a
    worked-example popup is not evidence the learner can do it alone, and the
    schedule shows two of the first three drills at the Solo rung, so a raw run
    promoted them for reading. But a run that is unbroken and merely aided still
    returns the rung it was made AT, because dropping that half would demote a
    learner mid-run — the window this route exists to overrule is still poisoned
    underneath, and taking away the hold as well as the promotion means the
    examples the system chose to show cost them the rung they were standing on.
    """
    full = _correct_run(attempts)
    if len(full) < _PROMOTE_STREAK:
        return None
    held = _lowest_rung(full)
    unaided = example_schedule.unaided_run(attempts)
    if len(unaided) < _PROMOTE_STREAK:
        return held
    earned = _lowest_rung(unaided)
    if earned is None:
        return held
    return _step_up(earned)


def _correct_run(attempts: List[dict]) -> List[dict]:
    """The trailing run of correct answers, aided or not."""
    run = []
    for attempt in reversed(attempts):
        if not attempt.get("correct"):
            break
        run.append(attempt)
    return run


def _lowest_rung(run: List[dict]) -> Optional[str]:
    """The lowest ladder rung any attempt in `run` was made at."""
    ranks = [
        LADDER_STAGES.index(a.get("stage"))
        for a in run
        if a.get("stage") in LADDER_STAGES
    ]
    return LADDER_STAGES[min(ranks)] if ranks else None
