"""One simulated learner under one arm: the shared policy and the run loop.

The policy is the same for every arm; only the belief it reads changes
(`beliefs.py`). Its shape is today's app, simplified:

  1. REVIEW   — a concept marked learned whose belief recall has fallen below
                `review_at`, every other action at most. Tuned separately from
                the gate (Seth 09-26): unlocking the next concept and keeping
                this one up are different decisions.
  2. EXPLORE  — (explore arms only) a concept not yet taught, not learned,
                whose P(known) clears the cost gate (drill/lesson minutes) and
                whose prerequisites are learned or not refuted, ranked by
                goal value x expected variance drop. Served with no lesson.
                After a break of 14+ days, concepts taught before it are
                probed again for 7 days (kc_explore's return window).
  3. EXPLOIT  — the frontier (not learned, every prerequisite learned):
                lesson first, then drills, until the belief marks it learned;
                at most DAY_CAP answers on one concept per session (the app's
                cap), so practice spaces itself across days.

LEARNED (sticky) = the belief's `gate` >= theta after at least one of the
concept's own answers: P(learned) for the BKT and hybrid beliefs, expected
recall a day out for pure FSRS (it has no learned state). Each arm is run at
several theta and compared at its best. The oracle gates on the truth itself.

METRIC: the summed practice hours at which each ARENA goal TRULY reaches
P(correct a day later) >= 0.80, checked at the end of every session.
A goal never reached counts as 1.5 x the horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sims import beliefs as B
from sims import voi
from sims.joint import JointBelief
from sims.world import Graph, Truth, minutes

SESSION_MIN = 60.0          # practice minutes per day
HORIZON_H = 150.0
MAX_DAYS = 500.0            # a policy stuck with nothing to serve ends here (goals left = penalty)
BREAK_AT_H = 40.0
REFUTED = 0.20
COST_RATIO = {"math": 1.5 / 6.0, "code": 4.0 / 8.0}
DAY_CAP = 3                 # drills on one concept per session (the app's cap)
RETURN_GAP, RETURN_WINDOW = 14.0, 7.0

ARMS = {
    # name: (belief, explore, "var" | "voi" probe score)
    "C": ("bkt", False, None),         # control: today's belief, no explore
    "B": ("bkt", True, "var"),         # Bayesian BKT x point FSRS + explore
    "BS": ("bkt_spread", True, "var"),  # B, with a spread on FSRS stability
    "BV": ("bkt", True, "voi"),        # B, probing for minutes saved (voi.py)
    "K": ("joint", True, "var"),       # one joint belief over the graph (joint.py)
    "KV": ("joint", True, "voi"),      # joint belief + minutes-saved probing
    "A0": ("fsrs", False, None),       # Bayesian FSRS alone, no explore
    "A": ("fsrs", True, "var"),        # Bayesian FSRS alone + explore
    "H": ("hybrid", True, "var"),      # Bayesian FSRS + learned state + explore
    "O": ("oracle", False, None),      # sees the truth
}


@dataclass
class Result:
    sum_h: float
    crossed: int
    goals: int
    false_decl: int
    decl: int
    probes: int
    lessons: int
    reviews: int
    drills: int


def make_belief(kind, g, truth, seed):
    rng = np.random.default_rng([seed, 7, ("bkt", "fsrs", "hybrid", "oracle", "bkt_spread", "joint").index(kind)])
    if kind == "bkt":
        return B.BKTBelief(g, rng)
    if kind == "joint":
        return JointBelief(g, rng)
    if kind == "bkt_spread":
        return B.BKTSpreadBelief(g, rng)
    if kind == "fsrs":
        return B.ParticleBelief(g, rng, hybrid=False)
    if kind == "hybrid":
        return B.ParticleBelief(g, rng, hybrid=True)
    return B.OracleBelief(g, truth)


def run(g: Graph, world: str, ltype: str, seed: int, arm: str, theta: float,
        break_days: float, review_at: float = 0.80) -> Result:
    truth = Truth(g, world, ltype, seed)
    kind, explore, scorer = ARMS[arm]
    bel = make_belief(kind, g, truth, seed)
    oracle = kind == "oracle"
    C = len(g.ids)
    gu, sl = g.guess, g.slip
    learned = np.zeros(C, bool)
    lesson_read = np.zeros(C, bool)
    own_answers = np.zeros(C, int)
    last_answer = np.full(C, -1e9)
    focus: Optional[int] = None
    today = np.zeros(C, int)                # drills per concept this session
    crossed_at = np.full(C, np.nan)
    stats = dict(false_decl=0, decl=0, probes=0, lessons=0, reviews=0, drills=0)
    prereq_lists = g.prereqs
    cost = np.array([COST_RATIO[k] for k in g.kind])
    practice_min, day = 0.0, 0.0
    took_break = break_days <= 0
    last_was_review = False
    return_until, stale = -1.0, np.zeros(C, bool)
    last_session_day = None

    def mark_learned(c, t):
        if learned[c]:
            return
        if bel.gate(c, t) >= theta and (oracle or own_answers[c] > 0):
            learned[c] = True
            if g.goal[c]:
                stats["decl"] += 1
                stats["false_decl"] += int(not truth.crossed(c, t))

    while practice_min < HORIZON_H * 60 and day < MAX_DAYS:
        t = day + 0.4
        if last_session_day is not None and explore and day - last_session_day >= RETURN_GAP:
            return_until = t + RETURN_WINDOW
            stale = (lesson_read | learned) & (last_answer < last_session_day + 1)
        last_session_day = day
        today[:] = 0
        session_end = min(practice_min + SESSION_MIN, HORIZON_H * 60)
        while practice_min < session_end:
            K = bel.K(t)
            if oracle:
                for c in np.flatnonzero(~learned):
                    mark_learned(c, t)
            due = learned & (bel.due_R(t) < review_at)
            action = None
            if due.any() and not last_was_review:
                cands = np.flatnonzero(due)
                c = cands[np.lexsort((bel.due_R(t)[cands], -g.value[cands]))[0]]
                action = ("review", c)
            if action is None and explore:
                ok_pre = learned | (K >= REFUTED)
                pre_ok = np.array([all(ok_pre[p] for p in prereq_lists[c]) for c in range(C)])
                fresh = ~learned & ~lesson_read & pre_ok & (today < DAY_CAP)
                back = stale & (today < DAY_CAP) if t < return_until else np.zeros(C, bool)
                if scorer == "voi" and back.any():
                    # Return window: re-probe what the break may have erased
                    # first, by the same rule every explore arm uses.
                    score = np.where(back, g.value * bel.info(t), -1.0)
                    c = int(np.argmax(score))
                    if score[c] > 1e-6:
                        action = ("probe", c)
                elif scorer == "voi":
                    # Probe for minutes saved (voi.py); no cost gate: the
                    # score already charges the probe's minutes.
                    cands = np.flatnonzero(fresh)
                    if len(cands):
                        pa, q = bel.p_after(cands, t)
                        sc = voi.scores(g, theta, cands, bel._p(), pa, q, learned, lesson_read)
                        j = int(np.argmax(sc))
                        if sc[j] > 0:
                            action = ("probe", int(cands[j]))
                else:
                    cand = (fresh & (K >= cost)) | back
                    if cand.any():
                        if hasattr(bel, "probe_scores"):
                            score = np.full(C, -1.0)
                            ix = np.flatnonzero(cand)
                            score[ix] = bel.probe_scores(t, ix)
                        else:
                            score = np.where(cand, g.value * bel.info(t), -1.0)
                        c = int(np.argmax(score))
                        if score[c] > 1e-6:
                            action = ("probe", c)
            if action is None:
                if focus is None or learned[focus] or today[focus] >= DAY_CAP:
                    front = [c for c in np.flatnonzero(~learned & (today < DAY_CAP))
                             if all(learned[p] for p in prereq_lists[c])]
                    focus = max(front, key=lambda c: (g.value[c], -g.depth[c], -c)) if front else None
                if focus is not None:
                    action = ("lesson", focus) if not lesson_read[focus] else ("drill", focus)
                else:
                    # Frontier used up for today (or done): review the weakest
                    # learned concept not yet answered today, else end the session.
                    spare = np.flatnonzero(learned & (today == 0))
                    if not len(spare):
                        break
                    action = ("review", spare[np.argmin(bel.due_R(t)[spare])])
            what, c = action
            last_was_review = what == "review"
            if what == "lesson":
                truth.lesson(c, t)
                bel.lesson(c, t)
                lesson_read[c] = True
                stats["lessons"] += 1
                dt = minutes(g, c, "lesson")
            else:
                probe = what == "probe"
                correct = truth.answer(c, t)
                bel.answer(c, t, correct, probe=probe and not lesson_read[c],
                           lesson_read=lesson_read[c])
                own_answers[c] += 1
                last_answer[c] = t
                if probe:
                    stale[c] = False
                    stats["probes"] += 1
                elif what == "review":
                    stats["reviews"] += 1
                else:
                    stats["drills"] += 1
                if what != "lesson":
                    today[c] += 1
                mark_learned(c, t)
                dt = minutes(g, c, "drill")
            practice_min += dt
            t += dt / 1440.0
        for c in np.flatnonzero(g.goal & np.isnan(crossed_at)):
            if truth.crossed(c, t):
                crossed_at[c] = practice_min / 60.0
        if not np.isnan(crossed_at[g.goal]).any():
            break
        day += 1.0
        if not took_break and practice_min >= BREAK_AT_H * 60:
            day += break_days
            took_break = True
    goal_h = crossed_at[g.goal]
    n_cross = int((~np.isnan(goal_h)).sum())
    total = float(np.nansum(goal_h) + (len(goal_h) - n_cross) * 1.5 * HORIZON_H)
    return Result(total, n_cross, len(goal_h), **stats)
