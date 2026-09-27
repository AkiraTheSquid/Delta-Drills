#!/usr/bin/env python3
"""Checks the simulator before any number from it is believed.

  * FSRS parity: fsrs_vec's scalar AND numpy copies reproduce
    app/memory_model.py (weights, review, implicit review) to 1e-9.
  * Common random numbers: the same learner seed gives the same truth, and
    a different arm cannot change the learner's starting state.
  * Collapsed beliefs: a particle cloud with one known particle tracks the
    point FSRS the BKT belief carries.
  * Sanity: the oracle beats the control; every run's numbers are finite.

Run: .venv/bin/python sims/test_sim.py   (from This-Directory-Only/backend)
"""
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USER_DATA_DIR", tempfile.mkdtemp(prefix="sims_test_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import memory_model as M  # noqa: E402
from sims import beliefs as Bl, fsrs_vec as F, sim, world  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


print("FSRS parity with app/memory_model.py")
check("weights identical", F.CONCEPT_PRIOR_WEIGHTS == M.CONCEPT_PRIOR_WEIGHTS)
cfg = M.DEFAULT_CONFIG
W = F.CONCEPT_PRIOR_WEIGHTS
rnd = random.Random(3)
worst = 0.0
for _ in range(300):
    m_ref, m_sim, t = None, None, 0.0
    for _ in range(12):
        t += rnd.choice([0.01, 0.3, 1.5, 4.0, 20.0, 90.0])
        if m_ref is not None and rnd.random() < 0.3:
            k = rnd.random()
            m_ref = M.implicit_review(m_ref, t, k, cfg)
            m_sim = F.implicit(m_sim, t, k, W)
        else:
            g = rnd.choice([F.AGAIN, F.GOOD])
            m_ref = M.review(m_ref, t, g, cfg)
            m_sim = F.review(m_sim, t, g, W)
        worst = max(worst, abs(m_ref.S - m_sim[0]) / m_ref.S, abs(m_ref.D - m_sim[1]),
                    abs(m_ref.t_last - m_sim[2]),
                    abs(M.retrievability(m_ref, t + 3, cfg) - F.R(m_sim[0], m_sim[2], t + 3, W)))
check("scalar review/implicit/R match", worst < 1e-9, f"worst {worst:.2e}")

S = np.exp(np.random.default_rng(1).normal(1, 2, 500))
D = np.random.default_rng(2).uniform(1, 10, 500)
tl = -np.random.default_rng(3).uniform(0, 60, 500)
worst = 0.0
for g in (F.AGAIN, F.GOOD):
    for t in (0.2, 5.0):
        vs, vd = F.v_after(S, D, tl, t, g, W)
        for i in range(0, 500, 7):
            ref = M.review(M.Memory(S[i], D[i], tl[i], 1, 0), t, g, cfg)
            worst = max(worst, abs(vs[i] - ref.S) / ref.S, abs(vd[i] - ref.D))
vs, vd, vt = F.v_implicit(S, D, tl, 5.0, 0.35, W)
for i in range(0, 500, 7):
    ref = M.implicit_review(M.Memory(S[i], D[i], tl[i], 1, 0), 5.0, 0.35, cfg)
    worst = max(worst, abs(vs[i] - ref.S) / ref.S, abs(vt[i] - ref.t_last))
check("numpy v_after / v_implicit match", worst < 1e-9, f"worst {worst:.2e}")

print("common random numbers")
G = world.load_graph()
check("graph = 27 ARENA goals + prerequisites", int(G.goal.sum()) == 27 and len(G.ids) >= 60,
      f"{int(G.goal.sum())} goals, {len(G.ids)} concepts")
a, b = world.Truth(G, "W1", "torch", 5), world.Truth(G, "W1", "torch", 5)
check("same seed → same learner", np.array_equal(a.L, b.L) and a.w == b.w
      and np.array_equal(a.u_ans, b.u_ans))
c = world.Truth(G, "W1", "torch", 6)
check("different seed → different learner", not np.array_equal(a.u_ans, c.u_ans))

print("collapsed particle belief ≈ point FSRS")
pb = Bl.ParticleBelief(G, np.random.default_rng(0), hybrid=True)
bb = Bl.BKTBelief(G, np.random.default_rng(0))
k = G.idx[next(i for i in G.ids if i.startswith("torch."))]
pb.L[k] = True
pb.S[k], pb.D[k], pb.tl[k] = F.init_s(F.GOOD, W), F.init_d(F.GOOD, W), 0.0
pb.touched[k] = True
pb.logw[k] = 0.0
bb.touched[k], bb.pL[k] = True, 1.0
bb.S[k], bb.D[k], bb.tl[k], bb.has_mem[k] = F.init_s(F.GOOD, W), F.init_d(F.GOOD, W), 0.0, True
for t, ok in ((1.0, True), (3.0, True), (9.0, False), (12.0, True)):
    pb.answer(k, t, ok, probe=False, lesson_read=True)
    bb.answer(k, t, ok, probe=False, lesson_read=True)
diff = abs(pb.due_R(20.0)[k] - bb.due_R(20.0)[k])
check("identical particles follow the point FSRS", diff < 1e-9, f"ΔR {diff:.2e}")

print("decision-value explorer (voi.py)")
from sims import voi  # noqa: E402
from sims.beliefs import BKTBelief  # noqa: E402
from sims.joint import JointBelief  # noqa: E402
for kind in ("math", "code"):
    J0, J1, T = voi.tables(kind, 0.95)
    check(f"{kind}: cost-to-go falls as P(learned) rises", np.all(np.diff(J0) <= 1e-6)
          and np.all(np.diff(J1) <= 1e-6))
    check(f"{kind}: probing option never costs more than teaching", np.all(J0 <= T + 1e-6))
    (hx, hy), _ = voi.hulls(kind, 0.95)
    Jc = np.interp(voi._P, hx, hy)
    slope = np.diff(hy) / np.diff(hx)
    check(f"{kind}: spillover curve is concave and never below J (information never hurts)",
          np.all(np.diff(slope) <= 1e-9) and np.all(Jc >= J0 - 1e-9))
for Bel in (BKTBelief, JointBelief):
    bel = Bel(G, np.random.default_rng(0))
    cands = np.arange(5)
    pa, q = bel.p_after(cands, 0.0)
    back = (1 - q)[:, None] * pa[:, 0] + q[:, None] * pa[:, 1]
    own = back[np.arange(5), cands] - (1 - back[np.arange(5), cands]) * 0  # includes transit
    others = np.delete(back, cands, 1) if Bel is JointBelief else back[:, 5:]
    check(f"{Bel.__name__}: p_after averages back to p (martingale) off the probed concept",
          np.allclose(others, np.delete(bel._p(), cands) if Bel is JointBelief else bel._p()[5:],
                      atol=0.02 if Bel is BKTBelief else 1e-9))

print("planner (#3)")
import copy as _copy  # noqa: E402
from sims import planner as Pl  # noqa: E402
tr = world.Truth(G, "W2", "torch", 5)
ep = sim.Episode(G, tr, sim.make_belief("bkt", G, tr, 5), "B", 0.95, 30.0, 0.8)
for _ in range(15):
    ep.session()
fk = ep.fork(_copy.deepcopy(tr))
check("a forked episode on a copy of its learner plays out identically",
      ep.run() == fk.run() and np.array_equal(ep.learned, fk.learned))
tr = world.Truth(G, "W1", "novice", 6)
ref = sim.Episode(G, tr, sim.make_belief("bkt", G, tr, 6), "B", 0.95, 0.0, 0.8)
for _ in range(10):
    ref.session()
snap = (ref.learned.copy(), ref.bel.pL.copy(), ref.practice_min)
e2 = ref.fork(world.Truth(G, "W1", "strong", 7))
e2.theta = 0.8
e2.run()
check("a rollout leaves the live episode untouched",
      np.array_equal(snap[0], ref.learned) and np.array_equal(snap[1], ref.bel.pL)
      and snap[2] == ref.practice_min)
for w in ("W1", "W2"):
    tr = world.Truth(G, w, "novice", 8)
    bel = sim.make_belief("bkt", G, tr, 8)
    e3 = sim.Episode(G, tr, bel, "B", 0.99, 0.0, 0.8)
    f = Pl.LearnerFilter(G, bel, 8)
    e3.observer = lambda what, c, t, y, f=f: f.lesson(c, t) if what == "lesson" else f.answer(c, t, y)
    for _ in range(30):
        e3.session()
    check(f"learner filter identifies {w} after 30 sessions", (f.p_w1() > 0.9) == (w == "W1"),
          f"P(W1)={f.p_w1():.2f}")
h = f.sample(1, np.random.default_rng(0), e3.lesson_read)[0]
check("a hypothesized learner never reuses a real seed", h.seed >= Pl.HYPO_SEED0)

print("fast engine (numba)")
from sims import fast  # noqa: E402
bad = []
for w, lt, s, arm, brk in (("W1", "novice", 11, "B", 0.0), ("W2", "strong", 12, "C", 180.0),
                           ("W3", "math", 13, "B", 30.0), ("W3", "torch", 14, "B", 180.0)):
    tr = world.Truth(G, w, lt, s)
    e4 = sim.Episode(G, tr, sim.make_belief("bkt", G, tr, s), arm, 0.95, brk, 0.8)
    for _ in range(12):                 # mid-course, so a rollout's starting state is exercised
        e4.session()
    got = fast.run(e4, 0.9, 0.7, truth=_copy.deepcopy(tr))
    e4.theta, e4.review_at = 0.9, 0.7
    if got != e4.run():
        bad.append((w, lt, arm))
check("fast.run = the Python episode, result for result (W1-W3, B and C, mid-course)", not bad, str(bad))

print("sanity runs")
res = {arm: [sim.run(G, "W1", lt, s, arm, 0.80, 0) for lt in ("novice", "torch") for s in (1, 2)]
       for arm in ("C", "O", "A", "B", "H", "BV", "K", "KV")}
med = {arm: float(np.median([r.sum_h for r in rs])) for arm, rs in res.items()}
print("   ", {k: round(v) for k, v in med.items()})
check("every result finite", all(np.isfinite(r.sum_h) for rs in res.values() for r in rs))
check("oracle beats control", med["O"] < med["C"], f'{med["O"]:.0f} < {med["C"]:.0f}')

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all passed")
