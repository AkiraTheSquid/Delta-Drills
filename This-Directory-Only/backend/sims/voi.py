"""Probe for the DECISION, not the variance (Seth 2026-09-26, "#2").

The question a probe answers is "how many practice minutes will this answer
save?". Per concept that is an optimal-stopping problem on the belief's own
model, solved exactly here by value iteration on a logit grid of p = P(learned):

  J1(p)  lesson read: drill until p >= theta.
         J1 = 0 if p >= theta, else  m + q J1(p|correct) + (1-q) J1(p|wrong)
  T(p)   teach now:  lesson minutes + J1(p after the lesson transit)
  J0(p)  lesson not read: J0 = 0 if p >= theta, else
         min( T(p),  m + q J0(p|correct) + (1-q) J0(p|wrong) )   (probe)

q = P(correct) = p(1-slip) + (1-p) guess; the posteriors are the BKT update
followed by the belief's transit (T_ANSWER after a lesson, T_PROBE before),
exactly what `BKTBelief.answer` does. Recall is taken as 1 (explore targets
concepts not yet taught).

An answer on c also moves other concepts' beliefs (area prior, joint graph);
that is worth J_i(p_i) - E_y J_i(p_i | y) minutes to concept i. J itself is a
STAIRCASE in p (one step per right answer the gate still needs), not concave,
so that difference could come out negative from the discreteness alone. The
spillover therefore reads Jc, the least concave curve above J (in p), under
which information never costs anything. The probe score is

  score(c) = [T_c(p_c) - m_c - E_y J0_c(p_c | y)]        probe c vs teach c now (exact J)
           + sum_{i != c, not learned} [Jc_i(p_i) - E_y Jc_i(p_i | y_c)]

and the policy probes the best c only when the score is positive.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from sims.beliefs import T_ANSWER, T_LESSON, T_PROBE
from sims.world import GUESS_SLIP, MINUTES

GRID = np.linspace(-9.0, 9.0, 721)          # logit p
_P = 1.0 / (1.0 + np.exp(-GRID))


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _post(p, correct, guess, slip, transit):
    e = 1.0 - slip
    if correct:
        q = p * e / (p * e + (1 - p) * guess)
    else:
        q = p * (1 - e) / (p * (1 - e) + (1 - p) * (1 - guess))
    return q + (1 - q) * transit


def _on_grid(J, p):
    return np.interp(_logit(p), GRID, J)


def _iterate(step, iters=4000, tol=1e-7):
    J = np.zeros_like(_P)
    for _ in range(iters):
        new = step(J)
        if np.max(np.abs(new - J)) < tol:
            return new
        J = new
    return J


@lru_cache(maxsize=None)
def tables(kind: str, theta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(J0, J1, T) on GRID for concepts of this kind at this gate."""
    guess, slip = GUESS_SLIP[kind]
    m, Lm = MINUTES[kind]["drill"], MINUTES[kind]["lesson"]
    done = _P >= theta
    q = _P * (1 - slip) + (1 - _P) * guess
    a1, a0 = _post(_P, True, guess, slip, T_ANSWER), _post(_P, False, guess, slip, T_ANSWER)
    J1 = _iterate(lambda J: np.where(done, 0.0, m + q * _on_grid(J, a1)
                                     + (1 - q) * _on_grid(J, a0)))
    T = Lm + _on_grid(J1, _P + (1 - _P) * T_LESSON)
    b1, b0 = _post(_P, True, guess, slip, T_PROBE), _post(_P, False, guess, slip, T_PROBE)
    J0 = _iterate(lambda J: np.where(done, 0.0, np.minimum(
        T, m + q * _on_grid(J, b1) + (1 - q) * _on_grid(J, b0))))
    return J0, J1, T


def _concave_hull(x, y):
    """Least concave majorant of (x, y), x ascending: its vertices."""
    hx, hy = [], []
    for xi, yi in zip(x, y):
        while len(hx) >= 2 and ((hy[-1] - hy[-2]) * (xi - hx[-2])
                                <= (yi - hy[-2]) * (hx[-1] - hx[-2])):
            hx.pop()
            hy.pop()
        hx.append(xi)
        hy.append(yi)
    return np.array(hx), np.array(hy)


@lru_cache(maxsize=None)
def hulls(kind: str, theta: float):
    J0, J1, _ = tables(kind, theta)
    return _concave_hull(_P, J0), _concave_hull(_P, J1)


def cost_to_go(kinds, theta, p, lesson_read):
    """Jc_i(p_i) (concave hull of J) for every concept; `p` may carry
    leading axes ([..., C])."""
    out = np.empty_like(p)
    for kind in ("math", "code"):
        mask = kinds == kind
        if not mask.any():
            continue
        (x0, y0), (x1, y1) = hulls(kind, theta)
        pm = p[..., mask]
        out[..., mask] = np.where(lesson_read[mask], np.interp(pm, x1, y1),
                                  np.interp(pm, x0, y0))
    return out


def exact_J0(kind, theta, p):
    return np.interp(_logit(p), GRID, tables(kind, theta)[0])


def teach_cost(kind, theta, p):
    return float(np.interp(_logit(p), GRID, tables(kind, theta)[2]))


def scores(g, theta, cands, p, p_after, q, learned, lesson_read):
    """Minutes saved by probing each candidate (see module docstring).

    p [C] current P(learned); p_after [k, 2, C] after a wrong / right probe
    on cands[k]; q [k] P(right)."""
    kinds = np.asarray(g.kind)
    live = ~learned
    J_now = cost_to_go(kinds, theta, p, lesson_read)                 # [C]
    J_after = cost_to_go(kinds, theta, p_after, lesson_read)         # [k, 2, C]
    EJ = (1 - q)[:, None] * J_after[:, 0] + q[:, None] * J_after[:, 1]   # [k, C]
    gain = np.where(live[None, :], J_now[None, :] - EJ, 0.0)
    out = np.empty(len(cands))
    for j, c in enumerate(cands):
        spill = gain[j].sum() - gain[j, c]
        m = MINUTES[g.kind[c]]["drill"]
        after = exact_J0(g.kind[c], theta, p_after[j, :, c])
        own = teach_cost(g.kind[c], theta, p[c]) - m - ((1 - q[j]) * after[0] + q[j] * after[1])
        out[j] = own + spill
    return out
