"""Explore-then-exploit: one posterior per concept decides BOTH which concept to
probe next and when a concept is settled.

Seth, 2026-09-22: "when it started me out for the math practice for linear
algebra it could have given me a more difficult linear system, but it instead
went with how to add vectors and went from the ground up. it should have done
some exploring before doing some exploiting." And on settling: "whether it gets
categorized as settled versus not settled, the model would have it based upon
the statistical model ... rather than the model determining it according to
[a special] rule."

The greedy ladder (`kc_graph.frontier`) only serves a concept once every
prerequisite is LEARNED, so a learner with no history always starts at the
roots and grinds up. In an EXPLORE AREA (EXPLORE_PREFIXES) this module adds:

  * A belief per concept, log-odds of "known", recomputed from the ladder
    record every call (the record is the state; nothing derived is stored):
      - direct evidence: the concept's own UNAIDED answers, each a Bayes factor
        from P_GUESS / P_SLIP (an answer given behind a worked example is not
        evidence of anything the learner knows);
      - indirect evidence: a correct answer flows UP each `encompassing` edge
        of the registry, scaled by the edge weight (and multiplied along a
        chain) — solving the linear system also exercised the vector
        arithmetic inside it. A miss flows DOWN to dependents, attenuated per
        hop, as in placement_model. Indirect evidence is clipped to
        ±INDIRECT_CLIP, so it can move a concept close to the line but only
        the concept's own answer can carry it over: every concept is asked at
        least once.
  * SETTLED = P(known) >= SETTLE_P. Settled counts as learned
    (kc_graph.kc_is_learned), so the concept leaves the frontier and unlocks
    its dependents. Whether that takes one answer or five falls out of the
    prior the neighbours built plus the answers given — there is no rule that
    says "one answer is enough"; see scripts/sim_kc_explore.py for the table.
  * EXPLORABLE = not settled, not refuted (P > OUT_OF_STATE), and every
    prerequisite settled/learned or itself not refuted. Explorable concepts
    join the frontier even when their prerequisites are unlearned, and are
    ordered by expected value-weighted variance reduction (ALEKS's "probe
    near 0.5", generalised to a graph with unequal stakes). With no history
    that is a DEEP concept: its answer informs every concept it encompasses.
  * PROBING = explorable and the lesson not yet shown or read: served at
    DRILL_FLOOR (`partial`, displayed "Solo") with no lesson and no example,
    because a probe answered off an example measures the example. A refuted concept stops being a probe and
    falls back to the ordinary lesson-first ladder — exploit.

Every constant below is a calibration choice, tuned by the simulation script,
not a claim about the learner.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from app import example_schedule

# Concept areas the explore/settle model runs on. Math first: its items are
# 4-option multiple choice, which the likelihoods below are calibrated for.
EXPLORE_PREFIXES = ("math.",)

# 4-option MC: a blind guess passes 1 in 4; a knower misreads ~1 in 10.
P_GUESS = 0.25
P_SLIP = 0.10

SETTLE_P = 0.90
OUT_OF_STATE = 0.20
INDIRECT_CLIP = 1.2

# A miss is evidence against the concepts built on this one (not the ones it
# rests on), weaker per hop — a miss may be a slip.
HOP_ATTENUATION_DOWN = 0.3
INCORRECT_DOWN_SCALE = 0.5
MAX_HOPS = 3  # for misses; encompassing credit attenuates by weight instead

# Self-report nudges the starting point, as placement_model does.
PRIOR_SHIFT_BY_LEVEL = {"beginner": -0.5, None: 0.0, "strong": 0.5}

LOG_BF_CORRECT = math.log((1.0 - P_SLIP) / P_GUESS)
LOG_BF_INCORRECT = math.log(P_SLIP / (1.0 - P_GUESS))


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


SETTLE_LOGODDS = _logit(SETTLE_P)
OUT_LOGODDS = _logit(OUT_OF_STATE)


def in_area(kc: str) -> bool:
    return kc.startswith(EXPLORE_PREFIXES)


# --- graph ----------------------------------------------------------------------


def _area_graph(registry: Dict[str, dict]):
    """(kcs, encompassed, children) restricted to explore-area concepts.

    `encompassed[k]` = {ancestor: product of encompassing weights along the
    strongest chain}. Strongest chain, not a sum over chains: two routes to
    the same prerequisite are one piece of evidence, not two. No hop cap —
    weights are in (0, 1], so a long chain attenuates itself, and a cap on a
    best-weight search can drop an ancestor only a shorter, weaker path
    reaches (codex, 2026-09-22). Entries that are not prerequisites, or whose
    weight is outside (0, 1], are ignored.
    `children[k]` = {descendant: hops}."""
    kcs = [k for k in registry if in_area(k)]
    keep = set(kcs)
    encompassed: Dict[str, Dict[str, float]] = {}
    for k in kcs:
        best: Dict[str, float] = {}
        frontier = [(k, 1.0)]
        while frontier:
            node, w = frontier.pop()
            for p, ew in (registry[node].get("encompassing") or {}).items():
                try:
                    ew = float(ew)
                except (TypeError, ValueError):
                    continue
                if p not in keep or p not in registry[node]["prereqs"] or not 0.0 < ew <= 1.0:
                    continue
                pw = w * ew
                if pw > best.get(p, 0.0):
                    best[p] = pw
                    frontier.append((p, pw))
        encompassed[k] = best
    kids: Dict[str, List[str]] = {k: [] for k in kcs}
    for k in kcs:
        for p in registry[k]["prereqs"]:
            if p in keep:
                kids[p].append(k)
    children: Dict[str, Dict[str, int]] = {}
    for k in kcs:
        dist: Dict[str, int] = {}
        todo = [(c, 1) for c in kids[k]]
        while todo:
            n, d = todo.pop(0)
            if d > MAX_HOPS or (n in dist and dist[n] <= d):
                continue
            dist[n] = d
            todo.extend((c, d + 1) for c in kids[n])
        children[k] = dist
    return kcs, encompassed, children


# --- posterior -------------------------------------------------------------------


def evidence(attempts_by_kc: Dict[str, List[dict]]) -> List[tuple]:
    """[(kc, correct)] from the unaided ladder attempts, oldest first per KC."""
    out = []
    for kc, attempts in attempts_by_kc.items():
        for a in example_schedule.unaided(attempts or []):
            out.append((kc, bool(a.get("correct"))))
    return out


def logodds(registry: Dict[str, dict], ev: Iterable[tuple], level: Optional[str] = None,
            graph=None) -> Dict[str, float]:
    """Log-odds of "known" for every explore-area KC. Pure."""
    kcs, encompassed, children = graph or _area_graph(registry)
    direct = {k: 0.0 for k in kcs}
    indirect = {k: PRIOR_SHIFT_BY_LEVEL.get(level, 0.0) for k in kcs}
    for kc, correct in ev:
        if kc not in direct:
            continue
        if correct:
            direct[kc] += LOG_BF_CORRECT
            for p, w in encompassed[kc].items():
                indirect[p] += LOG_BF_CORRECT * w
        else:
            direct[kc] += LOG_BF_INCORRECT
            for c, hops in children[kc].items():
                indirect[c] += LOG_BF_INCORRECT * INCORRECT_DOWN_SCALE * HOP_ATTENUATION_DOWN ** hops
    return {k: direct[k] + max(-INDIRECT_CLIP, min(INDIRECT_CLIP, indirect[k])) for k in kcs}


def _variance(L: Dict[str, float], value: Dict[str, float]) -> float:
    total = 0.0
    for k, x in L.items():
        p = _sigmoid(x)
        total += value.get(k, 1.0) * p * (1.0 - p)
    return total


def rank(registry: Dict[str, dict], ev: List[tuple], candidates: Iterable[str],
         value: Dict[str, float], level: Optional[str] = None) -> List[str]:
    """Candidates by expected value-weighted variance reduction, best first."""
    graph = _area_graph(registry)
    L = logodds(registry, ev, level, graph)
    base = _variance(L, value)
    scored = []
    for kc in candidates:
        if kc not in L:
            continue
        p = _sigmoid(L[kc])
        p_right = p * (1.0 - P_SLIP) + (1.0 - p) * P_GUESS
        after = (p_right * _variance(logodds(registry, ev + [(kc, True)], level, graph), value)
                 + (1.0 - p_right) * _variance(logodds(registry, ev + [(kc, False)], level, graph), value))
        scored.append((base - after, kc))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [kc for _, kc in scored]


# --- learner-facing (reads kc_graph lazily: kc_graph imports this module) ----------


def _state_inputs(user_state):
    from app import kc_graph
    reg = kc_graph._registry()
    attempts = {k: (kc_graph.ladder_view(user_state, k).get("attempts") or [])
                for k in reg if in_area(k)}
    return reg, evidence(attempts), getattr(user_state, "self_reported_level", None)


def beliefs(user_state) -> Dict[str, float]:
    """P(known) per explore-area KC."""
    reg, ev, level = _state_inputs(user_state)
    return {k: _sigmoid(x) for k, x in logodds(reg, ev, level).items()}


def settled(user_state, kc: str) -> bool:
    if not in_area(kc):
        return False
    reg, ev, level = _state_inputs(user_state)
    return logodds(reg, ev, level).get(kc, -math.inf) >= SETTLE_LOGODDS


def explorable(user_state, kc: str) -> bool:
    """May be probed even though its prerequisites are not learned yet."""
    if not in_area(kc):
        return False
    from app import kc_graph
    reg, ev, level = _state_inputs(user_state)
    L = logodds(reg, ev, level)
    x = L.get(kc)
    if x is None or x >= SETTLE_LOGODDS or x <= OUT_LOGODDS:
        return False
    for p in reg[kc]["prereqs"]:
        if p in L:
            if L[p] <= OUT_LOGODDS and not kc_graph.kc_is_learned(user_state, p):
                return False
        elif not kc_graph.kc_is_learned(user_state, p):
            return False
    return True


def probing(user_state, kc: str) -> bool:
    """Serve as a diagnostic probe: no lesson first, no example on screen.
    Only while the learner has not been taught it — neither shown the lesson
    by the ladder (`worked_seen`) nor read it on their own (graph node,
    `?lesson=`) — or the answer measures the page, not what they came with."""
    from app import kc_graph, lessons, practice_targets
    if not explorable(user_state, kc):
        return False
    if int(kc_graph.ladder_view(user_state, kc).get("worked_seen") or 0):
        return False
    return not lessons.kc_lesson_read(kc, practice_targets.effective_exposure(user_state))


def reorder(user_state, ordered: List[str]) -> List[str]:
    """`ordered` with the explore-area concepts re-sorted in the slots they
    already hold: open probes first, by expected information (explore), then
    everything else in its original greedy order (exploit). Slots are kept so
    the model changes WHICH math concept comes next, never how math
    interleaves with the rest."""
    slots = [i for i, k in enumerate(ordered) if in_area(k)]
    if len(slots) < 2:
        return ordered
    from app import kc_graph
    reg, ev, level = _state_inputs(user_state)
    descendants, _depth = kc_graph._closure()
    area = [ordered[i] for i in slots]
    probes = [k for k in area if probing(user_state, k)]
    if not probes:
        return ordered
    value = {k: descendants.get(k, 0) + 1 for k in reg if in_area(k)}
    best = rank(reg, ev, probes, value, level) + [k for k in area if k not in probes]
    out = list(ordered)
    for i, kc in zip(slots, best):
        out[i] = kc
    return out
