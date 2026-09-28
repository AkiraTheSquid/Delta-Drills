"""lesson_readiness.py — is the learner ready for this drill, or should the
page come back first?

Seth, 2026-09-19, after missing q650 (dtype names as strings) two days after
reading its page and never practising it: "it should have essentially shown me
the lesson again ... it needs to be based on the probability, not on some 8+
hour metric. It should ask 'are you ready for this, or does this degrade to
you solving a problem' and it should be adaptive based on the current backend
model."

THE QUESTION, IN THE ENGINE'S OWN TERMS

The logistic engine already answers "ready for a rung": `next_stage` promotes
when the LOWER credible bound of P(correct) at the next rung clears
`PROMOTE_P`, and `mastered` asks the same of the solo rung. This module asks
the same question of the drill actually about to be served — its item, its
rung, this learner, the page as it survives in memory — and of one
counterfactual: the same drill with the page re-read a moment ago.

    ready_now         = lo(P | page as it stands)   >= PROMOTE_P
    ready_after_read  = lo(P | page read just now)  >= PROMOTE_P
    show the page     = missed_since_read and not probing
                        and not ready_now and ready_after_read

Below the bar, a drill is what Seth called it: not retrieval practice but
problem solving on material that is not there, and the worked-example effect
says a novice in that state learns more from the example than from the
attempt. The page is the example. Above the bar it is expertise reversal — the
example is in the way — and the drill is served bare.

The second clause is what makes this a decision rather than a reflex. If the
page would NOT lift the learner over the bar, re-reading it is not the answer
— that learner is short of a prerequisite, which `remediation.py` handles —
and, mechanically, it is what stops a loop: the moment the page is read its
value is at its maximum, so `ready_now == ready_after_read` and the gate
cannot fire twice in a row for a re-read that did not help.

PROBE FIRST: A MISS, NOT A FORECAST, BRINGS THE PAGE BACK

Seth, 2026-09-28, shown "Refresher — you read this 18 days ago" for the einops
pattern page: "is this consistent with the FSRS model and the explore/exploit
BKT? perhaps it should do a probe before assuming that I need a lesson like
this?" It was not. The decision above reads a LOWER bound, and a long gap
widens the posterior (`inflate`), so the less the model knew the more it
re-taught — the opposite of `kc_explore`, where uncertainty is a reason to
probe. And the `lesson` feature's one-day half-life (unfitted) rates any page
read more than a few days ago at ~0 whatever FSRS says the concept's memory
is, so after day three the verdict hardly depended on the page at all.

So the forecast no longer fires on its own. The drill is served bare first:
it IS the probe, and a retrieval attempt before restudy is the better move
anyway. The page comes back only when the concept's newest answer since the
page was last read is a miss (`missed_since_read`) — evidence, not a lower
bound — and the two-clause test above still has to agree the page is what
helps. A correct answer after the miss clears it; re-reading the page resets
it, which is also what keeps the gate from firing twice for one miss.

A PROBE IS NEVER GATED. `kc_explore.probing` (a return re-probe, or a
concept not yet taught) is a measurement of what the learner brought; the page
in front of it would measure the page.

WHAT MAKES TIME ENTER

Nothing here reads a clock against a threshold. Time reaches the decision only
through the model: the `lesson` feature fades with `lesson_half_life_days`,
`recency` fades with practice, `inflate` widens the posterior while unobserved.
So the same page comes back sooner for a learner whose ability on the concept
is unproven and never for one who has demonstrated it, which is the adaptive
behaviour asked for, and every number in the chain is a model parameter the
attempt log now records per row and can therefore fit.

WHO IS NEVER GATED HERE

A page the learner never read (placed past it by the diagnostic, or reached
through an untagged path) has nothing to come back to: `read_page_for` returns
None and the concept is left alone. Unread pages are `lessons.unexposed_target_kcs`'
business and run first; a concept with a page still unread is taught, not
re-taught.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from app import engine_bridge, kc_explore, kc_graph, lessons, practice_targets
from app import logistic_engine as E


def _lower_bound(prediction: E.Prediction) -> float:
    lo, _hi = prediction.interval(E.LADDER_Z)
    return lo


def is_ready(prediction: E.Prediction) -> bool:
    """The ladder's own bar, applied to one drill: the lower credible bound of
    P(correct) clears `PROMOTE_P`."""
    return _lower_bound(prediction) >= E.PROMOTE_P


def _ts(value) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def missed_since_read(
    user_state, kc: str, read_at: datetime, now: Optional[datetime] = None
) -> bool:
    """The concept's newest answer AFTER the page was last read (and not after
    `now` — a replay or a skewed clock must not see the future) was a miss.

    False with no answer since the read: the next drill is the probe. Rows
    without a readable `ts` predate the field and count as before the read.
    Read off the ladder row (windowed to the last 20), which is ample for
    "since the last read" — a reader 20 answers past the read who is still
    missing is remediation's case, and the window still holds that miss."""
    for attempt in reversed(kc_graph.ladder_view(user_state, kc).get("attempts") or []):
        ts = _ts(attempt.get("ts"))
        if ts is not None and now is not None and ts > now:
            continue
        if ts is None or ts <= read_at:
            return False
        return not attempt.get("correct")
    return False


def readiness(
    user_state,
    kc: str,
    question_id: int,
    *,
    difficulty_score: Optional[float],
    stage: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Everything the gate decides from, for one concept of one drill.

    Returns `page` (the step fields + `read_at`, or None), `days_since_read`,
    the two predictions (`now`, `after_read`), the two verdicts, the evidence
    (`missed_since_read`, `probing`), and `lesson_needed`. `stage` defaults
    to the rung the ladder would serve this concept at; the scoring path's
    `served_stage` is the same value read back from the row it wrote.
    """
    now = now or datetime.now(timezone.utc)
    page = lessons.read_page_for(question_id, kc, getattr(user_state, "kc_exposure", None) or {})
    days = lessons.page_age_days(page, now)
    stage = stage or kc_graph.kc_stage(user_state, kc)
    current = engine_bridge.predict(
        user_state, kc, difficulty_score=difficulty_score, stage=stage, lesson_days=days,
        now=now,
    )
    after_read = engine_bridge.predict(
        user_state, kc, difficulty_score=difficulty_score, stage=stage, lesson_days=0.0,
        now=now,
    )
    ready_now = is_ready(current)
    ready_after = is_ready(after_read)
    missed = page is not None and missed_since_read(user_state, kc, page["read_at"], now)
    # Only asked when it could change the verdict: `probing` rebuilds the
    # explore posteriors.
    probe = missed and kc_explore.probing(user_state, kc)
    return {
        "kc": kc,
        "page": page,
        "days_since_read": days,
        "stage": stage,
        "now": current,
        "after_read": after_read,
        "ready_now": ready_now,
        "ready_after_read": ready_after,
        "missed_since_read": missed,
        "probing": probe,
        # A page never read is not re-taught here, and a forecast alone never
        # re-teaches one that was — see PROBE FIRST in the module note.
        "lesson_needed": missed and not probe and not ready_now and ready_after,
    }


def revisit_target_kcs(
    user_state,
    question_id: int,
    *,
    difficulty_score: Optional[float],
    now: Optional[datetime] = None,
    skip: Optional[set] = None,
) -> List[dict]:
    """Gate entries re-teaching the page of every target concept the learner
    missed since reading and the engine says the page would carry over the
    bar. Entries carry `revisit: True`,
    the ISO `read_at` the client compares its own record against, and the two
    lower bounds the decision was made on."""
    gates: List[dict] = []
    seen = set(skip or ())
    for kc in lessons.target_kcs(question_id):
        if kc in seen:
            continue
        seen.add(kc)
        info = lessons.gate_info(kc)
        if not info:
            continue
        verdict = readiness(
            user_state, kc, question_id, difficulty_score=difficulty_score, now=now
        )
        if not verdict["lesson_needed"]:
            continue
        page = verdict["page"]
        gates.append({
            **info,
            **{k: v for k, v in page.items() if k != "read_at"},
            "revisit": True,
            "read_at": page["read_at"].isoformat(),
            "ready_lo": round(_lower_bound(verdict["now"]), 3),
            "ready_lo_after_read": round(_lower_bound(verdict["after_read"]), 3),
            "ready_bar": E.PROMOTE_P,
        })
    return gates


def lesson_gate(
    user_state,
    question_id: int,
    *,
    difficulty_score: Optional[float],
    now: Optional[datetime] = None,
) -> List[dict]:
    """Every page the learner is owed before this question: the unread ones
    first (against the placement-aware exposure map), then a re-read for any
    target concept that had nothing unread and is below the bar without it."""
    gates = lessons.unexposed_target_kcs(
        question_id, practice_targets.effective_exposure(user_state)
    )
    gated = {entry["kc"] for entry in gates}
    gates.extend(
        revisit_target_kcs(
            user_state, question_id, difficulty_score=difficulty_score, now=now, skip=gated
        )
    )
    return gates
