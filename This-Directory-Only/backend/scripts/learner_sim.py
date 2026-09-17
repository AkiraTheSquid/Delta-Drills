#!/usr/bin/env python3
"""learner_sim.py — Monte-Carlo learner simulator over the encompassing graph.

POMDP-flavored: a learner has a HIDDEN true skill vector `s`; the tutor holds a
BELIEF `e` updated by BKT from observed correct/incorrect responses AND by FIRe
trickle-down through the (LLM-authored) encompassing graph. The policy gates and
selects exercises from `e`. The learner answers from `s`.

The point: the tutor credits encompassed skills using the LLM's weights `w_hat`,
but the learner truly improves using the world's weights `w_true`. When they
differ (LLM mistake), the tutor can believe a skill is mastered while true skill
lags — ILLUSORY MASTERY, a hidden learning gap. We measure:

  1. FIRe value     — exercises-to-mastery WITH vs WITHOUT encompassing credit.
  2. Robustness     — illusory-mastery vs. tutor edge-weight error level.
  3. Edge impact    — which encompassing edges, if wrong, cause the most
                      illusory mastery (→ prioritize for domain-expert review).

Pure forward simulation (no state explosion). Deterministic given the seed.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
GRAPH = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v3_encompassing.json"

MASTERY = 0.95          # belief threshold the tutor treats as "mastered"
GAP = 0.6               # true-skill below this while believed-mastered = illusory
HORIZON = 5000          # time steps per learner (review happens over time)
N_LEARNERS = 10         # per profile
DECAY = 0.0001          # per-step forgetting. Sustainable mastered set ≈
                        # 0.05/DECAY ≈ 500 > 393, so the learner can reach the
                        # advanced atoms where encompassing credit originates.

# BKT-ish learner profiles: pT=learn rate, guess/slip=response noise,
# fragile=prereq-strength gates learning, decay=passive forgetting per step.
PROFILES = {
    "fast":            dict(pT=0.45, guess=0.10, slip=0.05, fragile=False),
    "fragile_prereq":  dict(pT=0.40, guess=0.10, slip=0.08, fragile=True),
    "guesser":         dict(pT=0.30, guess=0.35, slip=0.05, fragile=False),
    "slipper":         dict(pT=0.40, guess=0.10, slip=0.30, fragile=False),
}


def load_graph():
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    atoms = [c["id"] for c in g["concepts"]]
    prereqs_of = defaultdict(list)      # X -> [simpler atoms required first]
    encompassed_by = defaultdict(list)  # A(advanced) -> [(B simpler, weight, conf)]
    for e in g["prerequisite_edges"]:
        prereqs_of[e["dependent_id"]].append(e["prerequisite_id"])
        if e.get("is_encompassing"):
            encompassed_by[e["dependent_id"]].append(
                (e["prerequisite_id"], e["propagation_weight"], e.get("confidence", 0.8))
            )
    return atoms, prereqs_of, encompassed_by


def p_correct(skill, guess, slip):
    return skill * (1 - slip) + (1 - skill) * guess


def bkt_posterior(belief, correct, guess, slip):
    if correct:
        num = belief * (1 - slip)
        den = num + (1 - belief) * guess
    else:
        num = belief * slip
        den = num + (1 - belief) * (1 - guess)
    return num / den if den > 1e-12 else belief


SKIP_W = 0.4    # aggressive policy trusts encompassing edges >= this to skip review


def simulate(profile, atoms, prereqs_of, encompassed_by, w_hat, seed,
             use_encompassing=True, policy="conservative"):
    """One learner. w_hat: dict (advanced, simpler) -> tutor weight (LLM belief).
    True world uses the graph's own weights.

    policy:
      conservative — review any atom whose belief decays below mastery (direct
        evidence always required; encompassing credit is a bonus).
      aggressive   — FIRe scheduling: once an atom is learned, the tutor STOPS
        directly reviewing it if a mastered advanced atom encompasses it
        (w_hat >= SKIP_W) — it ASSUMES implicit reps keep it fresh and pins its
        belief to mastered. If that edge is wrong (w_true low), true skill
        decays unseen -> illusory mastery. This is the risk dial.
    """
    rng = random.Random(seed)
    p = PROFILES[profile]
    s = {a: 0.0 for a in atoms}     # hidden true skill
    e = {a: 0.0 for a in atoms}     # tutor belief
    learned = {a: False for a in atoms}  # has the tutor ever certified it?
    exercises = 0                   # explicit practice events (the cost)
    dep_count = defaultdict(int)
    enc_parents = defaultdict(list)  # b(simpler) -> [a(advanced) that encompass it]
    for x, prs in prereqs_of.items():
        for pr in prs:
            dep_count[pr] += 1
    for a, lst in encompassed_by.items():
        for b, w, c in lst:
            enc_parents[b].append(a)

    def unlocked(a):
        return all(e[pr] >= MASTERY for pr in prereqs_of.get(a, []))

    def fire_covered(b):
        # an advanced atom the tutor believes mastered, joined by a trusted edge
        return any(w_hat.get((a, b), 0.0) >= SKIP_W and e[a] >= MASTERY
                   for a in enc_parents.get(b, []))

    for _ in range(HORIZON):
        for a in atoms:
            s[a] *= (1 - DECAY)
            e[a] *= (1 - DECAY)
        # aggressive FIRe: pin belief of learned, implicitly-covered atoms to
        # mastered and skip their direct review (trusting the LLM edge).
        if policy == "aggressive":
            for b in atoms:
                if learned[b] and fire_covered(b):
                    e[b] = MASTERY
        cand = [a for a in atoms if e[a] < MASTERY and unlocked(a)]
        if not cand:
            continue
        a = max(cand, key=lambda x: (dep_count.get(x, 0), -e[x]))
        exercises += 1

        correct = rng.random() < p_correct(s[a], p["guess"], p["slip"])

        gate = 1.0
        if p["fragile"] and prereqs_of.get(a):
            gate = min(s[pr] for pr in prereqs_of[a]) ** 0.5
        s[a] += (1 - s[a]) * p["pT"] * gate
        for b, w_true, _ in encompassed_by.get(a, []):
            s[b] += (1 - s[b]) * p["pT"] * w_true

        e[a] = bkt_posterior(e[a], correct, p["guess"], p["slip"])
        e[a] = e[a] + (1 - e[a]) * p["pT"]
        if e[a] >= MASTERY:
            learned[a] = True
        if use_encompassing:
            for b, _, _ in encompassed_by.get(a, []):
                wh = w_hat.get((a, b), 0.0)
                e[b] = e[b] + (1 - e[b]) * p["pT"] * wh

    believed = [a for a in atoms if e[a] >= MASTERY]
    illusory = [a for a in believed if s[a] < GAP]
    return dict(exercises=exercises, coverage=len(believed),
                illusory=len(illusory))


def main():
    atoms, prereqs_of, encompassed_by = load_graph()
    n_enc = sum(len(v) for v in encompassed_by.values())
    true_w = {(a, b): w for a, lst in encompassed_by.items() for (b, w, _) in lst}
    print(f"graph: {len(atoms)} atoms, {n_enc} encompassing edges\n")

    # ---- 1. benefit/risk dial: conservative vs aggressive policy -------------
    print("1. POLICY DIAL — coverage (atoms believed fresh) and illusory (believed but NOT truly known)")
    print(f"   {'profile':16s} {'consv cov':>10s} {'consv illu':>11s} {'aggr cov':>9s} {'aggr illu':>10s}")
    for prof in PROFILES:
        cons = [simulate(prof, atoms, prereqs_of, encompassed_by, true_w, 100 + i, True, "conservative")
                for i in range(N_LEARNERS)]
        aggr = [simulate(prof, atoms, prereqs_of, encompassed_by, true_w, 100 + i, True, "aggressive")
                for i in range(N_LEARNERS)]
        cc = sum(r["coverage"] for r in cons) / len(cons)
        ci = sum(r["illusory"] for r in cons) / len(cons)
        ac = sum(r["coverage"] for r in aggr) / len(aggr)
        ai = sum(r["illusory"] for r in aggr) / len(aggr)
        print(f"   {prof:16s} {cc:10.0f} {ci:11.1f} {ac:9.0f} {ai:10.1f}")

    # ---- 2. Robustness: illusory mastery vs tutor weight-error level ----
    print("\n2. ROBUSTNESS under AGGRESSIVE policy (illusory-mastery atoms vs LLM weight error; lower=safer)")
    print(f"   {'profile':16s}" + "".join(f"{f'err={x}':>9s}" for x in (0.0, 0.2, 0.4, 0.6)))
    for prof in PROFILES:
        row = []
        for err in (0.0, 0.2, 0.4, 0.6):
            ill = []
            for i in range(N_LEARNERS):
                rng = random.Random(7000 + i)
                w_hat = {k: max(0.0, min(1.0, w + rng.gauss(0, err * (1 - c))))
                         for a, lst in encompassed_by.items() for (b, w, c) in lst
                         for k in [(a, b)]}
                ill.append(simulate(prof, atoms, prereqs_of, encompassed_by,
                                    w_hat, 7000 + i, True, "aggressive")["illusory"])
            row.append(sum(ill) / len(ill))
        print(f"   {prof:16s}" + "".join(f"{v:9.1f}" for v in row))

    # ---- 3. Edge impact: which edges, if wrong (zeroed in tutor), hurt most ----
    print("\n3. EDGE IMPACT (Δillusory if this edge's credit were spurious; review top first)")
    title = {c["id"]: c["title"] for c in json.loads(GRAPH.read_text())["concepts"]}
    base = [simulate("fragile_prereq", atoms, prereqs_of, encompassed_by, true_w, 100 + i, True, "aggressive")
            for i in range(N_LEARNERS)]
    base_ill = sum(r["illusory"] for r in base) / len(base)
    # spurious edge = tutor trusts it (w_hat=0.9) but the WORLD doesn't have it
    # (w_true=0) -> the learner gets no real implicit reps, gap opens.
    edges = sorted(true_w.items(), key=lambda kv: -kv[1])[:25]
    impacts = []
    for (a, b), w in edges:
        bad = dict(true_w); bad[(a, b)] = 0.9          # tutor still trusts it
        enc2 = {k: [(bb, (0.0 if (k, bb) == (a, b) else ww), cc) for (bb, ww, cc) in v]
                for k, v in encompassed_by.items()}    # world drops it (w_true=0)
        res = [simulate("fragile_prereq", atoms, prereqs_of, enc2, bad, 100 + i, True, "aggressive")
               for i in range(N_LEARNERS)]
        d = sum(r["illusory"] for r in res) / len(res) - base_ill
        impacts.append((d, a, b, w))
    impacts.sort(key=lambda x: -x[0])
    print(f"   baseline illusory (fragile_prereq): {base_ill:.1f}")
    for d, a, b, w in impacts[:8]:
        print(f"   +{d:4.1f}  w={w:.2f}  {title.get(a,a)[:34]:34s} ⊃ {title.get(b,b)[:30]}")


if __name__ == "__main__":
    main()
