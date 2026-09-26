"""FSRS-6 for the simulator: a scalar copy for the true learner and a numpy
copy for particle beliefs, both line-for-line ports of `app/memory_model.py`.

No app import here, so worker processes stay light. `test_sim.py` pins both
copies to `memory_model` (weights and every update rule) to 1e-9.
"""
from __future__ import annotations

import math

import numpy as np

FSRS6_DEFAULT_WEIGHTS = (
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722,
    0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425,
    0.0912, 0.0658, 0.1542,
)
_P = list(FSRS6_DEFAULT_WEIGHTS)
_P[2], _P[7], _P[8], _P[11], _P[13] = 1.5, 0.02, 1.45, 1.9286, 0.37
CONCEPT_PRIOR_WEIGHTS = tuple(_P)
del _P

AGAIN, HARD, GOOD, EASY = 1, 2, 3, 4
D_MIN, D_MAX = 1.0, 10.0
S_MIN, S_MAX = 0.001, 36500.0


def decay(w) -> float:
    return -w[20]


def factor(w) -> float:
    return 0.9 ** (1.0 / decay(w)) - 1.0


# --- scalar (true learner) ----------------------------------------------------

def _cs(s):
    return min(max(s, S_MIN), S_MAX)


def _cd(d):
    return min(max(d, D_MIN), D_MAX)


def init_s(g, w):
    return _cs(w[g - 1])


def init_d(g, w, clamp=True):
    d = w[4] - math.exp(w[5] * (g - 1)) + 1
    return _cd(d) if clamp else d


def next_d(d, g, w):
    delta = -(w[6] * (g - 3))
    damped = d + (10.0 - d) * delta / 9.0
    return _cd(w[7] * init_d(EASY, w, clamp=False) + (1 - w[7]) * damped)


def short_term_s(s, g, w):
    inc = math.exp(w[17] * (g - 3 + w[18])) * s ** -w[19]
    if g >= HARD:
        inc = max(inc, 1.0)
    return _cs(s * inc)


def recall_s(d, s, r, g, w):
    hard = w[15] if g == HARD else 1.0
    return _cs(s * (1 + math.exp(w[8]) * (11 - d) * s ** -w[9]
                    * (math.exp((1 - r) * w[10]) - 1) * hard))


def forget_s(d, s, r, w):
    long_term = w[11] * d ** -w[12] * ((s + 1) ** w[13] - 1) * math.exp((1 - r) * w[14])
    return _cs(min(long_term, s / math.exp(w[17] * w[18])))


def R(S, t_last, t, w):
    return (1 + factor(w) * max(0.0, t - t_last) / S) ** decay(w)


def elapsed_for(r, s, w):
    r = min(max(r, 1e-9), 1.0)
    return s / factor(w) * (r ** (1.0 / decay(w)) - 1.0)


def after(S, D, t_last, t, g, w):
    if t - t_last < 1.0:
        s = short_term_s(S, g, w)
    elif g == AGAIN:
        s = forget_s(D, S, R(S, t_last, t, w), w)
    else:
        s = recall_s(D, S, R(S, t_last, t, w), g, w)
    return s, next_d(D, g, w)


def review(mem, t, g, w):
    """mem = (S, D, t_last) or None → new (S, D, t_last)."""
    if mem is None:
        return (init_s(g, w), init_d(g, w), t)
    s, d = after(mem[0], mem[1], mem[2], t, g, w)
    return (s, d, t)


def implicit(mem, t, weight, w):
    if mem is None or weight <= 0:
        return mem
    k = min(weight, 1.0)
    S, D, tl = mem
    r = R(S, tl, t, w)
    s_full, d_full = after(S, D, tl, t, GOOD, w)
    s = _cs(S + k * (s_full - S))
    d = _cd(D + k * (d_full - D))
    r_new = r + k * (1 - r)
    return (s, d, t - elapsed_for(r_new, s, w))


# --- numpy (particle beliefs) -------------------------------------------------

def vR(S, t_last, t, w):
    return (1 + factor(w) * np.maximum(0.0, t - t_last) / S) ** decay(w)


def v_next_d(D, g, w):
    delta = -(w[6] * (g - 3))
    damped = D + (10.0 - D) * delta / 9.0
    return np.clip(w[7] * init_d(EASY, w, clamp=False) + (1 - w[7]) * damped, D_MIN, D_MAX)


def v_after(S, D, t_last, t, g, w):
    """Vectorised `after` for one grade over particle arrays."""
    same_day = (t - t_last) < 1.0
    inc = math.exp(w[17] * (g - 3 + w[18])) * S ** -w[19]
    if g >= HARD:
        inc = np.maximum(inc, 1.0)
    s_short = S * inc
    r = vR(S, t_last, t, w)
    if g == AGAIN:
        long_term = w[11] * D ** -w[12] * ((S + 1) ** w[13] - 1) * np.exp((1 - r) * w[14])
        s_long = np.minimum(long_term, S / math.exp(w[17] * w[18]))
    else:
        hard = w[15] if g == HARD else 1.0
        s_long = S * (1 + math.exp(w[8]) * (11 - D) * S ** -w[9]
                      * (np.exp((1 - r) * w[10]) - 1) * hard)
    s = np.clip(np.where(same_day, s_short, s_long), S_MIN, S_MAX)
    return s, v_next_d(D, g, w)


def v_elapsed_for(r, s, w):
    r = np.clip(r, 1e-9, 1.0)
    return s / factor(w) * (r ** (1.0 / decay(w)) - 1.0)


def v_implicit(S, D, t_last, t, k, w):
    """Fractional Good review of weight k (0..1) on particle arrays."""
    r = vR(S, t_last, t, w)
    s_full, d_full = v_after(S, D, t_last, t, GOOD, w)
    s = np.clip(S + k * (s_full - S), S_MIN, S_MAX)
    d = np.clip(D + k * (d_full - D), D_MIN, D_MAX)
    r_new = r + k * (1 - r)
    return s, d, t - v_elapsed_for(r_new, s, w)
