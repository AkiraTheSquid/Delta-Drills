"""Compiled copy (numba) of one explore arm's episode, for the planner's rollouts.

Arm P spends nearly all its time playing the rest of the course 72 times per
plan (planner.py). The Python episode does that on 71-long arrays, where
interpreter and numpy call overhead is ~all the cost. This module is the same
episode as loops: the B belief (`beliefs.BKTBelief`), the review -> explore ->
exploit policy (`sim.Episode`, "var" probe score or no explore) and the true
learner (`world.Truth`, all three worlds), line for line.

Results are identical to the Python episode (144 whole B/C runs over W1-W3,
and P end to end: same knob picks, same hours; `test_sim.py` re-checks
mid-course starts). That is checked, not guaranteed: numpy's SIMD exp / log /
pow differ from libm's in the last bit on some inputs, so a change to either
copy must re-run the test. `SIM_FAST=0` puts the planner back on Python.

`run(ep)` plays `ep` (an `Episode` on a B-family belief) from its current state
to the end and returns what `ep.run()` would, leaving `ep` untouched.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit as _njit

from sims import fsrs_vec as F
from sims import sim
from sims.beliefs import AREA_PRIOR_PSEUDO, INDIRECT_CLIP, T_ANSWER, T_LESSON, T_PROBE
from sims.world import CROSS_P, FIRE_SCALE, MAX_EVENTS, MINUTES

AGAIN, HARD, GOOD, EASY = F.AGAIN, F.HARD, F.GOOD, F.EASY
S_MIN, S_MAX, D_MIN, D_MAX = F.S_MIN, F.S_MAX, F.D_MIN, F.D_MAX
SESSION_MIN, HORIZON_MIN, MAX_DAYS = sim.SESSION_MIN, sim.HORIZON_H * 60, sim.MAX_DAYS
BREAK_AT_MIN, REFUTED, DAY_CAP = sim.BREAK_AT_H * 60, sim.REFUTED, sim.DAY_CAP
RETURN_GAP, RETURN_WINDOW = sim.RETURN_GAP, sim.RETURN_WINDOW
LOGIT_HI = 1 - 1e-9
njit = _njit(cache=True, error_model="numpy")   # float x/0 -> inf, as numpy
# action codes
NONE, REVIEW, PROBE, LESSON, DRILL = 0, 1, 2, 3, 4


# --- FSRS-6 (fsrs_vec's scalar copy) ------------------------------------------

@njit
def _cs(s):
    return min(max(s, S_MIN), S_MAX)


@njit
def _cd(d):
    return min(max(d, D_MIN), D_MAX)


@njit
def _init_d(g, w):
    return w[4] - math.exp(w[5] * (g - 1)) + 1


@njit
def _next_d(d, g, w):
    delta = -(w[6] * (g - 3))
    damped = d + (10.0 - d) * delta / 9.0
    return _cd(w[7] * _init_d(EASY, w) + (1 - w[7]) * damped)


@njit
def _R(S, tl, t, fac, dec):
    return (1 + fac * max(0.0, t - tl) / S) ** dec


@njit
def _after(S, D, tl, t, g, w, fac, dec):
    if t - tl < 1.0:
        inc = math.exp(w[17] * (g - 3 + w[18])) * S ** -w[19]
        if g >= HARD:
            inc = max(inc, 1.0)
        s = _cs(S * inc)
    elif g == AGAIN:
        r = _R(S, tl, t, fac, dec)
        long_term = w[11] * D ** -w[12] * ((S + 1) ** w[13] - 1) * math.exp((1 - r) * w[14])
        s = _cs(min(long_term, S / math.exp(w[17] * w[18])))
    else:
        r = _R(S, tl, t, fac, dec)
        hard = w[15] if g == HARD else 1.0
        s = _cs(S * (1 + math.exp(w[8]) * (11 - D) * S ** -w[9]
                     * (math.exp((1 - r) * w[10]) - 1) * hard))
    return s, _next_d(D, g, w)


@njit
def _review(S, D, tl, has, c, t, g, w, fac, dec):
    """In place on memory arrays: FSRS review of c with grade g."""
    if not has[c]:
        S[c], D[c], tl[c] = _cs(w[g - 1]), _cd(_init_d(g, w)), t
        has[c] = True
    else:
        S[c], D[c] = _after(S[c], D[c], tl[c], t, g, w, fac, dec)
        tl[c] = t


@njit
def _implicit(S, D, tl, c, t, weight, w, fac, dec):
    """In place: fractional Good review (FIRe); memory must exist."""
    if weight <= 0:
        return
    k = min(weight, 1.0)
    s0, d0, t0 = S[c], D[c], tl[c]
    r = _R(s0, t0, t, fac, dec)
    s_full, d_full = _after(s0, d0, t0, t, GOOD, w, fac, dec)
    s = _cs(s0 + k * (s_full - s0))
    d = _cd(d0 + k * (d_full - d0))
    r_new = min(max(r + k * (1 - r), 1e-9), 1.0)
    S[c], D[c], tl[c] = s, d, t - s / fac * (r_new ** (1.0 / dec) - 1.0)


@njit
def _logit(p):
    p = min(max(p, 1e-9), LOGIT_HI)
    return math.log(p / (1 - p))


@njit
def _sig(x):
    return 1 / (1 + math.exp(-min(max(x, -60.0), 60.0)))


# --- the true learner (world.Truth) ---------------------------------------------

@njit
def _own(c, t, jump, tL, tk, tS, ttl, thas, fac, dec):
    if not thas[c]:
        return 0.0
    base = (1.0 if tL[c] else 0.0) if jump[c] else tk[c]
    return base * _R(tS[c], ttl[c], t, fac, dec)


@njit
def _t_pre(c, t, pre_ptr, pre_idx, tau, jump, tL, tk, tS, ttl, thas, fac, dec):
    out = 1.0
    for j in range(pre_ptr[c], pre_ptr[c + 1]):
        out *= 1 - tau * (1 - _own(pre_idx[j], t, jump, tL, tk, tS, ttl, thas, fac, dec))
    return out


@njit
def _p_correct(c, t, luck, guess, slip, enc_ptr, enc_idx, enc_w, bad_day,
               jump, tL, tk, tS, ttl, thas, fac, dec):
    know = _own(c, t, jump, tL, tk, tS, ttl, thas, fac, dec)
    for j in range(enc_ptr[c], enc_ptr[c + 1]):
        know *= 1 - enc_w[j] * (1 - _own(enc_idx[j], t, jump, tL, tk, tS, ttl, thas, fac, dec))
    if luck and len(bad_day) > 0 and bad_day[int(t)]:
        know *= 0.75
    return guess[c] + (1 - guess[c] - slip[c]) * know


@njit
def _grow(k, r, w3):
    if w3:
        return min(1.0, r * (0.3 + k)) * (1 - k)
    return r * (1 - k)


@njit
def _draw(u, n, c):
    if n[c] >= MAX_EVENTS or n[c] >= u.shape[1]:
        raise RuntimeError("common random numbers exhausted")
    x = u[c, n[c]]
    n[c] += 1
    return x


# --- the episode ------------------------------------------------------------------

@njit
def episode(goal, value, depth, guess, slip, cost, lmin, dmin, area_ix,
            pre_ptr, pre_idx, enc_ptr, enc_idx, enc_w,
            tw, tfac, tdec, jump, w3, tau, bad_day, speed, tL, tk, tS, tD, ttl, thas, tread,
            u_ans, u_learn, n_ans, n_learn,
            bw, bfac, bdec, pL, bS, bD, btl, bhas, touched, ind, cnt_n, cnt_h,
            learned, lesson_read, own_answers, last_answer, today, crossed_at, stale,
            practice_min, day, took_break, last_was_review, return_until,
            has_lsd, last_session_day, focus, theta, review_at, break_days, explore):
    """Plays to the end on COPIES of every state array. Returns
    (crossed_at, stats[false_decl, decl, probes, lessons, reviews, drills])."""
    C = len(goal)
    tL, tk, tS, tD, ttl, thas, tread = (tL.copy(), tk.copy(), tS.copy(), tD.copy(),
                                        ttl.copy(), thas.copy(), tread.copy())
    n_ans, n_learn = n_ans.copy(), n_learn.copy()
    pL, bS, bD, btl, bhas, touched, ind = (pL.copy(), bS.copy(), bD.copy(), btl.copy(),
                                           bhas.copy(), touched.copy(), ind.copy())
    cnt_n, cnt_h = cnt_n.copy(), cnt_h.copy()
    learned, lesson_read, own_answers = learned.copy(), lesson_read.copy(), own_answers.copy()
    last_answer, today, crossed_at, stale = (last_answer.copy(), today.copy(),
                                             crossed_at.copy(), stale.copy())
    stats = np.zeros(6, np.int64)
    pe = np.empty(C)
    p = np.empty(C)
    R = np.empty(C)
    pe_dirty = True

    while not (practice_min >= HORIZON_MIN or day >= MAX_DAYS):
        # ---- session() ----
        t = day + 0.4
        if has_lsd and explore and day - last_session_day >= RETURN_GAP:
            return_until = t + RETURN_WINDOW
            for c in range(C):
                stale[c] = (lesson_read[c] or learned[c]) and last_answer[c] < last_session_day + 1
        has_lsd, last_session_day = True, day
        today[:] = 0
        session_end = min(practice_min + SESSION_MIN, HORIZON_MIN)
        while practice_min < session_end:
            # ---- the belief at t: prior_eff, p = _p(), R = _R(t) ----
            if pe_dirty:
                for c in range(C):
                    a = area_ix[c]
                    gu, sl = guess[c], slip[c]
                    neutral = gu + (1 - gu - sl) * 0.5
                    acc = (cnt_h[a] + AREA_PRIOR_PSEUDO * neutral) / (cnt_n[a] + AREA_PRIOR_PSEUDO)
                    pr = min(max((acc - gu) / (1 - gu - sl), 0.02), 0.98)
                    pe[c] = _sig(_logit(pr) + ind[c])
                pe_dirty = False
            for c in range(C):
                p[c] = pL[c] if touched[c] else pe[c]
                R[c] = _R(bS[c], btl[c], t, bfac, bdec) if bhas[c] else 1.0
            # ---- _choose(t) ----
            what, pick = NONE, -1
            if not last_was_review:
                for c in range(C):
                    if learned[c] and R[c] < review_at:
                        if (pick < 0 or value[c] > value[pick]
                                or (value[c] == value[pick] and R[c] < R[pick])):
                            pick = c
                if pick >= 0:
                    what = REVIEW
            if what == NONE and explore:
                back_on = t < return_until
                best, bs = -1, -np.inf
                for c in range(C):
                    cand = False
                    if back_on and stale[c] and today[c] < DAY_CAP:
                        cand = True
                    elif not learned[c] and not lesson_read[c] and today[c] < DAY_CAP \
                            and p[c] * R[c] >= cost[c]:
                        cand = True
                        for j in range(pre_ptr[c], pre_ptr[c + 1]):
                            q = pre_idx[j]
                            if not (learned[q] or p[q] * R[q] >= REFUTED):
                                cand = False
                                break
                    if cand:
                        Rc, pc = R[c], p[c]
                        e = (1 - slip[c]) * Rc + guess[c] * (1 - Rc)
                        q1 = pc * e + (1 - pc) * guess[c]
                        p1 = pc * e / q1
                        p0 = pc * (1 - e) / max(1 - q1, 1e-12)
                        var = Rc * Rc * pc * (1 - pc)
                        sc = value[c] * (var - Rc * Rc * (q1 * p1 * (1 - p1)
                                                          + (1 - q1) * p0 * (1 - p0)))
                    else:
                        sc = -1.0
                    if sc > bs:
                        best, bs = c, sc
                if bs > 1e-6:
                    what, pick = PROBE, best
            if what == NONE:
                if focus < 0 or learned[focus] or today[focus] >= DAY_CAP:
                    focus = -1
                    for c in range(C):
                        if learned[c] or today[c] >= DAY_CAP:
                            continue
                        ok = True
                        for j in range(pre_ptr[c], pre_ptr[c + 1]):
                            if not learned[pre_idx[j]]:
                                ok = False
                                break
                        if ok and (focus < 0 or value[c] > value[focus]
                                   or (value[c] == value[focus] and depth[c] < depth[focus])):
                            focus = c
                if focus >= 0:
                    what, pick = (DRILL if lesson_read[focus] else LESSON), focus
                else:
                    for c in range(C):
                        if learned[c] and today[c] == 0 and (pick < 0 or R[c] < R[pick]):
                            pick = c
                    if pick >= 0:
                        what = REVIEW
            if what == NONE:
                break
            c = pick
            last_was_review = what == REVIEW
            if what == LESSON:
                # truth.lesson
                pre = _t_pre(c, t, pre_ptr, pre_idx, tau, jump, tL, tk, tS, ttl, thas,
                             tfac, tdec) * speed[c]
                u = _draw(u_learn, n_learn, c)
                if jump[c]:
                    if not tL[c]:
                        if u < min(0.95, 0.6 * pre):
                            tL[c] = True
                            thas[c] = False
                            _review(tS, tD, ttl, thas, c, t, GOOD, tw, tfac, tdec)
                    elif thas[c]:
                        _implicit(tS, tD, ttl, c, t, 0.5, tw, tfac, tdec)
                else:
                    tk[c] += _grow(tk[c], min(0.95, 0.6 * pre), w3)
                    if not thas[c]:
                        _review(tS, tD, ttl, thas, c, t, GOOD, tw, tfac, tdec)
                    else:
                        _implicit(tS, tD, ttl, c, t, 0.5, tw, tfac, tdec)
                tread[c] = True
                # bel.lesson
                if not touched[c]:
                    pL[c] = pe[c]
                    touched[c] = True
                pL[c] += (1 - pL[c]) * T_LESSON
                lesson_read[c] = True
                stats[3] += 1
                dt = lmin[c]
            else:
                probe = what == PROBE
                # truth.answer
                if n_ans[c] >= MAX_EVENTS or n_ans[c] >= u_ans.shape[1]:
                    raise RuntimeError("common random numbers exhausted")
                correct = u_ans[c, n_ans[c]] < _p_correct(
                    c, t, True, guess, slip, enc_ptr, enc_idx, enc_w, bad_day,
                    jump, tL, tk, tS, ttl, thas, tfac, tdec)
                n_ans[c] += 1
                grade = GOOD if correct else AGAIN
                pre = _t_pre(c, t, pre_ptr, pre_idx, tau, jump, tL, tk, tS, ttl, thas,
                             tfac, tdec) * speed[c]
                u = _draw(u_learn, n_learn, c)
                rate = 0.15 if tread[c] else 0.04
                if jump[c]:
                    if tL[c]:
                        _review(tS, tD, ttl, thas, c, t, grade, tw, tfac, tdec)
                    elif u < min(0.95, rate * pre):
                        tL[c] = True
                        thas[c] = False
                        _review(tS, tD, ttl, thas, c, t, GOOD, tw, tfac, tdec)
                else:
                    tk[c] += _grow(tk[c], min(0.95, 2.7 * rate * pre), w3)
                    _review(tS, tD, ttl, thas, c, t, grade, tw, tfac, tdec)
                if correct:
                    for j in range(enc_ptr[c], enc_ptr[c + 1]):
                        if thas[enc_idx[j]]:
                            _implicit(tS, tD, ttl, enc_idx[j], t, enc_w[j] * FIRE_SCALE,
                                      tw, tfac, tdec)
                # bel.answer
                if not touched[c]:
                    pL[c] = pe[c]
                    touched[c] = True
                Rc = R[c]
                e = (1 - slip[c]) * Rc + guess[c] * (1 - Rc)
                if correct:
                    ll, lu = e, guess[c]
                else:
                    ll, lu = 1 - e, 1 - guess[c]
                pc = pL[c]
                pL[c] = pc * ll / (pc * ll + (1 - pc) * lu)
                _review(bS, bD, btl, bhas, c, t, grade, bw, bfac, bdec)
                if correct:
                    for j in range(enc_ptr[c], enc_ptr[c + 1]):
                        comp, wt = enc_idx[j], enc_w[j]
                        if bhas[comp]:
                            _implicit(bS, bD, btl, comp, t, wt * FIRE_SCALE, bw, bfac, bdec)
                        full = wt * math.log((1 - slip[comp]) / guess[comp])
                        d = max(0.0, min(full, INDIRECT_CLIP - ind[comp]))
                        ind[comp] += d
                        pe_dirty = True
                        frac = d / full if full > 0 else 0.0
                        if frac != 0 and touched[comp]:
                            pL[comp] = _sig(_logit(pL[comp]) + frac * full)
                lr = lesson_read[c]
                pL[c] += (1 - pL[c]) * (T_ANSWER if lr else T_PROBE)
                if probe and not lr:
                    cnt_n[area_ix[c]] += 1
                    cnt_h[area_ix[c]] += 1 if correct else 0
                    pe_dirty = True
                # episode bookkeeping
                own_answers[c] += 1
                last_answer[c] = t
                if probe:
                    stale[c] = False
                    stats[2] += 1
                elif what == REVIEW:
                    stats[4] += 1
                else:
                    stats[5] += 1
                today[c] += 1
                # mark_learned (gate = P(learned); c is touched)
                if not learned[c] and pL[c] >= theta and own_answers[c] > 0:
                    learned[c] = True
                    if goal[c]:
                        stats[1] += 1
                        if _p_correct(c, t + 1.0, False, guess, slip, enc_ptr, enc_idx, enc_w,
                                      bad_day, jump, tL, tk, tS, ttl, thas,
                                      tfac, tdec) < CROSS_P:
                            stats[0] += 1
                dt = dmin[c]
            practice_min += dt
            t += dt / 1440.0
        # ---- end of session: goal crossings ----
        all_crossed = True
        for c in range(C):
            if goal[c] and np.isnan(crossed_at[c]):
                if _p_correct(c, t + 1.0, False, guess, slip, enc_ptr, enc_idx, enc_w, bad_day,
                              jump, tL, tk, tS, ttl, thas, tfac, tdec) >= CROSS_P:
                    crossed_at[c] = practice_min / 60.0
                else:
                    all_crossed = False
        if all_crossed:
            break
        day += 1.0
        if not took_break and practice_min >= BREAK_AT_MIN:
            day += break_days
            took_break = True
    return crossed_at, stats


# --- packing Python state -----------------------------------------------------------

def _csr(lists):
    ptr = np.zeros(len(lists) + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    return ptr


def graph_args(g):
    """Graph arrays, cached on the graph."""
    a = getattr(g, "_fast", None)
    if a is None:
        enc_idx = np.array([c for row in g.enc for c, _ in row], np.int64)
        enc_w = np.array([w for row in g.enc for _, w in row], float)
        a = g._fast = (
            np.asarray(g.goal, bool), np.asarray(g.value, float), np.asarray(g.depth, np.int64),
            np.asarray(g.guess, float), np.asarray(g.slip, float),
            np.array([sim.COST_RATIO[k] for k in g.kind]),
            np.array([MINUTES[k]["lesson"] for k in g.kind]),
            np.array([MINUTES[k]["drill"] for k in g.kind]),
            np.array([sorted(set(g.area)).index(x) for x in g.area], np.int64),
            _csr(g.prereqs), np.array([p for ps in g.prereqs for p in ps], np.int64),
            _csr(g.enc), enc_idx, enc_w)
    return a


def truth_args(tr):
    C = len(tr.g.ids)
    has = np.array([m is not None for m in tr.mem])
    mem = np.array([m if m is not None else (1.0, 5.0, 0.0) for m in tr.mem], float).reshape(C, 3)
    w = np.array(tr.w, float)
    return (w, F.factor(tr.w), F.decay(tr.w), np.asarray(tr.jump, bool), tr.world == "W3",
            float(tr.tau), np.asarray(tr.bad_day, bool), tr.speed, np.asarray(tr.L, bool),
            np.asarray(tr.k, float), mem[:, 0].copy(), mem[:, 1].copy(), mem[:, 2].copy(), has,
            np.asarray(tr.lesson_read, bool), tr.u_ans, tr.u_learn,
            tr.n_ans.astype(np.int64), tr.n_learn.astype(np.int64))


def belief_args(bel):
    W = np.array(F.CONCEPT_PRIOR_WEIGHTS)
    n = np.array([bel.counts.get(a, [0, 0])[0] for a in bel.areas], float)
    h = np.array([bel.counts.get(a, [0, 0])[1] for a in bel.areas], float)
    return (W, F.factor(W), F.decay(W), bel.pL, bel.S, bel.D, bel.tl, bel.has_mem,
            bel.touched, bel.ind, n, h)


def episode_args(ep):
    lsd = ep.last_session_day
    return (ep.learned, ep.lesson_read, ep.own_answers.astype(np.int64), ep.last_answer,
            ep.today.astype(np.int64), ep.crossed_at, ep.stale,
            float(ep.practice_min), float(ep.day), bool(ep.took_break), bool(ep.last_was_review),
            float(ep.return_until), lsd is not None, float(lsd or 0.0),
            -1 if ep.focus is None else int(ep.focus))


def result(g, crossed_at, stats) -> sim.Result:
    goal_h = crossed_at[g.goal]
    n_cross = int((~np.isnan(goal_h)).sum())
    total = float(np.nansum(goal_h) + (len(goal_h) - n_cross) * 1.5 * sim.HORIZON_H)
    fd, decl, probes, lessons, reviews, drills = (int(x) for x in stats)
    return sim.Result(total, n_cross, len(goal_h), fd, decl, probes, lessons, reviews, drills)


def supported(ep) -> bool:
    return ep.bel.name == "bkt" and not ep.oracle and ep.scorer in (None, "var")


def run(ep, theta=None, review_at=None, truth=None) -> sim.Result:
    """What `ep.fork(truth).run()` would return (knobs overridable); `ep` untouched."""
    assert supported(ep), "fast engine: B-family belief, 'var' or no explore only"
    th = ep.theta if theta is None else theta
    rv = ep.review_at if review_at is None else review_at
    ca, st = episode(*graph_args(ep.g), *truth_args(truth or ep.truth), *belief_args(ep.bel),
                     *episode_args(ep), th, rv, float(ep.break_days), bool(ep.explore))
    so_far = [ep.stats[k] for k in ("false_decl", "decl", "probes", "lessons", "reviews", "drills")]
    return result(ep.g, ca, st + np.array(so_far, np.int64))
