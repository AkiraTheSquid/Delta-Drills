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
import time
import weakref
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from app import example_schedule

# Concept areas the explore/settle model runs on. Math only until 2026-09-24;
# now EVERY concept (None). The learner simulation that decided it (scratchpad
# `usim/`, reconciled with a gpt-6-astra review) found whole-graph explore
# pays ONLY together with the AREA PRIOR + COST GATE and the RETURN WINDOW
# below: probed naively at 50/50 it cost beginners 10-20%; gated, it was
# neutral with no break and 8% faster to recover after a 30-180 day break
# (15-20% for learners who already knew things). A tuple of prefixes still
# works to narrow it.
EXPLORE_PREFIXES: Optional[tuple] = None

# 4-option MC: a blind guess passes 1 in 4; a knower misreads ~1 in 10.
P_GUESS = 0.25
P_SLIP = 0.10

# Code drills (graded against hidden tests), used by the ready route
# (app/ready_route.py) over the torch/einops/cnn/raytracing concepts. Passing
# every test without the concept is rarer than guessing an MC option, but the
# ▶ button shows the tests before Submit, so not negligible; a knower fails
# more often than on MC (a shape slip, an off-by-one). Calibration choices, as
# above. Each answer's Bayes factor is further TEMPERED by app/kc_evidence.py
# (how close the item is to the ARENA problem, and how long ago it was).
P_GUESS_CODE = 0.10
P_SLIP_CODE = 0.15

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
LOG_BF_CORRECT_CODE = math.log((1.0 - P_SLIP_CODE) / P_GUESS_CODE)
LOG_BF_INCORRECT_CODE = math.log(P_SLIP_CODE / (1.0 - P_GUESS_CODE))


def likelihood(kc: str):
    """(P_GUESS, P_SLIP) for answers on `kc`: MC for math, code otherwise."""
    return (P_GUESS, P_SLIP) if kc.startswith("math.") else (P_GUESS_CODE, P_SLIP_CODE)


def _log_bf(kc: str, correct: bool) -> float:
    if kc.startswith("math."):
        return LOG_BF_CORRECT if correct else LOG_BF_INCORRECT
    return LOG_BF_CORRECT_CODE if correct else LOG_BF_INCORRECT_CODE


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


SETTLE_LOGODDS = _logit(SETTLE_P)
OUT_LOGODDS = _logit(OUT_OF_STATE)

# COST GATE. A probe replaces the lesson: right, it saves the lesson's minutes;
# wrong, the drill's minutes are spent and the lesson still comes. So a probe
# pays only while P(known) > drill minutes / lesson minutes. Minutes are the
# simulation's calibration (math MC ~1.5 of a ~6 min page, code ~4 of ~8), not
# measured from the attempt log yet. PROBE_MARGIN: how far above the break-even
# P(known) must be — see the constant's note for why it is not zero.
PROBE_COST_RATIO_MATH = 1.5 / 6.0
PROBE_COST_RATIO_CODE = 4.0 / 8.0
PROBE_MARGIN = 0.0

# AREA PRIOR. Before a concept has evidence of its own, how likely the learner
# knows it is read off their PROBE answers in the same area (`torch`, `einops`,
# `math`, ...): a probe is answered with no lesson first, so it measures what
# they came with. Guess/slip-corrected, and shrunk toward a NEUTRAL 0.5 with
# AREA_PRIOR_PSEUDO probes' weight — not toward their overall rate, which let
# one python miss block probing math the learner knew (astra, 2026-09-24).
AREA_PRIOR_PSEUDO = 2.0

# RETURN WINDOW. After a break of RETURN_GAP_DAYS with no answers, concepts
# already taught may be probed again (unaided, ranked by information) for
# RETURN_WINDOW_DAYS: their old answers have faded (kc_evidence retention), so
# one answer on a deep concept re-settles it and credits what it encompasses.
# With no break this changes nothing; re-probing taught concepts ALL the time
# was worse than only on return in the simulation.
RETURN_GAP_DAYS = 14.0
RETURN_WINDOW_DAYS = 7.0


def in_area(kc: str) -> bool:
    return EXPLORE_PREFIXES is None or kc.startswith(EXPLORE_PREFIXES)


def area_of(kc: str) -> str:
    return kc.split(".", 1)[0]


def probe_cost_ratio(kc: str) -> float:
    return PROBE_COST_RATIO_MATH if kc.startswith("math.") else PROBE_COST_RATIO_CODE


def area_known(kc: str, counts: Dict[str, Tuple[int, int]]) -> float:
    """P(known) for a concept in `kc`'s area with no evidence of its own, from
    `counts[area] = (probes, correct)`."""
    guess, slip = likelihood(kc)
    n, h = counts.get(area_of(kc), (0, 0))
    neutral = guess + (1.0 - guess - slip) * 0.5
    acc = (h + AREA_PRIOR_PSEUDO * neutral) / (n + AREA_PRIOR_PSEUDO)
    return min(0.98, max(0.02, (acc - guess) / (1.0 - guess - slip)))


def leave_out(counts: Dict[str, Tuple[int, int]], kc: str, own: Optional[Tuple[int, int]]):
    """`counts` without `kc`'s own probe responses: its posterior already holds
    them, and they must not come back a second time as its prior."""
    if not own:
        return counts
    n, h = counts.get(area_of(kc), (0, 0))
    return {**counts, area_of(kc): (max(0, n - own[0]), max(0, h - own[1]))}


def worth_probing(kc: str, x: float, counts: Dict[str, Tuple[int, int]]) -> bool:
    """The cost gate: the posterior `x` (log-odds) combined with the area prior
    clears the break-even P(known), by PROBE_MARGIN. The 1e-9 keeps a neutral
    0.5 against a 0.5 ratio from flipping on float rounding."""
    p = _sigmoid(x + _logit(area_known(kc, counts)))
    return p >= probe_cost_ratio(kc) + PROBE_MARGIN - 1e-9


def return_start(answer_days: List[float], now_day: float) -> Optional[float]:
    """When the learner came back from their latest break of RETURN_GAP_DAYS
    or more (the first answer after it, or NOW if they have not answered since),
    as days since the epoch; None when there was no such break. Pure."""
    days = sorted(answer_days)
    if not days:
        return None
    if now_day - days[-1] >= RETURN_GAP_DAYS:
        return now_day
    for i in range(len(days) - 1, 0, -1):
        if days[i] - days[i - 1] >= RETURN_GAP_DAYS:
            return days[i]
    return None


def in_return_window(answer_days: List[float], now_day: float) -> bool:
    start = return_start(answer_days, now_day)
    return start is not None and now_day - start < RETURN_WINDOW_DAYS


# --- graph ----------------------------------------------------------------------


def _area_graph(registry: Dict[str, dict], area: Optional[Iterable[str]] = None):
    """(kcs, encompassed, children) restricted to explore-area concepts, or to
    `area` when given (the ready route passes a target's prerequisite closure).

    `encompassed[k]` = {ancestor: product of encompassing weights along the
    strongest chain}. Strongest chain, not a sum over chains: two routes to
    the same prerequisite are one piece of evidence, not two. No hop cap —
    weights are in (0, 1], so a long chain attenuates itself, and a cap on a
    best-weight search can drop an ancestor only a shorter, weaker path
    reaches (codex, 2026-09-22). Entries that are not prerequisites, or whose
    weight is outside (0, 1], are ignored.
    `children[k]` = {descendant: hops}."""
    if area is None:
        kcs = [k for k in registry if in_area(k)]
    else:
        wanted = set(area)
        kcs = [k for k in registry if k in wanted]
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


def evidence(attempts_by_kc: Dict[str, List[dict]], weigh=None) -> List[tuple]:
    """[(kc, correct, weight)] from the unaided ladder attempts, oldest first
    per KC. `weigh(kc, attempt)` is app/kc_evidence.weigher (similarity to the
    ARENA problem × retention since the answer); None = every answer in full."""
    out = []
    for kc, attempts in attempts_by_kc.items():
        for a in example_schedule.unaided(attempts or []):
            w = 1.0 if weigh is None else float(weigh(kc, a))
            out.append((kc, bool(a.get("correct")), w))
    return out


def logodds(registry: Dict[str, dict], ev: Iterable[tuple], level: Optional[str] = None,
            graph=None) -> Dict[str, float]:
    """Log-odds of "known" for every KC of the graph (explore area by
    default). Pure. `ev` rows are (kc, correct) or (kc, correct, weight); the
    weight tempers that answer's Bayes factor (app/kc_evidence.py)."""
    kcs, encompassed, children = graph or _area_graph(registry)
    direct = {k: 0.0 for k in kcs}
    indirect = {k: PRIOR_SHIFT_BY_LEVEL.get(level, 0.0) for k in kcs}
    for row in ev:
        kc, correct = row[0], row[1]
        weight = row[2] if len(row) > 2 else 1.0
        if kc not in direct:
            continue
        bf = _log_bf(kc, correct) * weight
        if correct:
            direct[kc] += bf
            for p, w in encompassed[kc].items():
                indirect[p] += bf * w
        else:
            direct[kc] += bf
            for c, hops in children[kc].items():
                indirect[c] += bf * INCORRECT_DOWN_SCALE * HOP_ATTENUATION_DOWN ** hops
    return {k: direct[k] + max(-INDIRECT_CLIP, min(INDIRECT_CLIP, indirect[k])) for k in kcs}


def _variance(L: Dict[str, float], value: Dict[str, float]) -> float:
    total = 0.0
    for k, x in L.items():
        p = _sigmoid(x)
        total += value.get(k, 1.0) * p * (1.0 - p)
    return total


def rank(registry: Dict[str, dict], ev: List[tuple], candidates: Iterable[str],
         value: Dict[str, float], level: Optional[str] = None,
         area: Optional[Iterable[str]] = None,
         item_weight: Optional[Dict[str, float]] = None, graph=None) -> List[str]:
    """Candidates by expected value-weighted variance reduction, best first.

    `area` scopes the graph (default: the explore area). `item_weight[kc]` is
    the tempering weight of the item a probe of `kc` would be served as (the
    ready route probes a prerequisite with its integrated problem, 0.75)."""
    return [kc for _, kc in scored_probes(registry, ev, candidates, value, level, area, item_weight)]


def scored_probes(registry, ev, candidates, value, level=None, area=None, item_weight=None):
    """[(expected variance reduction, kc)] best first — `rank` with its scores."""
    graph = _area_graph(registry, area)
    L = logodds(registry, ev, level, graph)
    base = _variance(L, value)
    ev = list(ev)
    scored = []
    for kc in candidates:
        if kc not in L:
            continue
        w = (item_weight or {}).get(kc, 1.0)
        guess, slip = likelihood(kc)
        p = _sigmoid(L[kc])
        p_right = p * (1.0 - slip) + (1.0 - p) * guess
        after = (p_right * _variance(logodds(registry, ev + [(kc, True, w)], level, graph), value)
                 + (1.0 - p_right) * _variance(logodds(registry, ev + [(kc, False, w)], level, graph), value))
        scored.append((base - after, kc))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


# --- learner-facing (reads kc_graph lazily: kc_graph imports this module) ----------

# Whole-graph explore made every picker call rebuild the posterior over every
# concept, ~190 times per `frontier()` (0.25 s on Seth's state). The inputs are
# memoised per learner state, keyed like memory_model.memories: id() + a weakref
# that drops the entry with the state, checked against the ladder fingerprint
# (an appended answer invalidates it), the level, the explore area, and the
# hour (retention discounts old answers as time passes; an hour moves it by
# far less than one answer does).
_memo: Dict[int, tuple] = {}
_graph_memo: Optional[tuple] = None


def _default_graph(reg: Dict[str, dict]):
    global _graph_memo
    key = (id(reg), EXPLORE_PREFIXES)
    if _graph_memo is None or _graph_memo[0] is not reg or _graph_memo[1] != key:
        _graph_memo = (reg, key, _area_graph(reg))
    return _graph_memo[2]


class _Inputs:
    __slots__ = ("reg", "ev", "level", "graph", "L", "counts", "own", "answer_days", "last_day", "now_day")


def _inputs(user_state) -> _Inputs:
    from app import kc_evidence, kc_graph, memory_model
    reg = kc_graph._registry()
    level = getattr(user_state, "self_reported_level", None)
    now_day = time.time() / 86400.0
    fp = (memory_model._fingerprint(user_state), level, EXPLORE_PREFIXES, id(reg), int(now_day * 24))
    key = id(user_state)
    hit = _memo.get(key)
    if hit is not None and hit[0]() is user_state and hit[1] == fp:
        return hit[2]
    attempts = {k: (kc_graph.ladder_view(user_state, k).get("attempts") or []) for k in reg}
    graph = _default_graph(reg)
    I = _Inputs()
    I.reg, I.level, I.graph, I.now_day = reg, level, graph, now_day
    # Retention (kc_evidence.weigher) discounts an answer from long ago, so a
    # concept settled a month back reopens as a probe.
    I.ev = evidence({k: a for k, a in attempts.items() if in_area(k)}, kc_evidence.weigher(user_state))
    I.L = logodds(reg, I.ev, level, graph)
    # Probe responses per area, each counted ONCE: a question tagged with two
    # concepts wrote a row under each (codex, 2026-09-24). `own[kc]` = the
    # responses on `kc` itself, left out of its own prior (explorable).
    responses: Dict[tuple, bool] = {}
    own_keys: Dict[str, set] = {}
    days: List[float] = []
    last_day: Dict[str, float] = {}
    for k, rows in attempts.items():
        for a in rows:
            t = kc_evidence._days(a.get("ts"))
            if t is not None:
                days.append(t)
                last_day[k] = max(t, last_day.get(k, t))
            if a.get("probe") and not example_schedule.aided(a):
                rkey = (area_of(k), a.get("question_id"), a.get("ts"))
                responses[rkey] = bool(a.get("correct"))
                own_keys.setdefault(k, set()).add(rkey)
    counts: Dict[str, Tuple[int, int]] = {}
    for (ar, _q, _t), ok in responses.items():
        n, h = counts.get(ar, (0, 0))
        counts[ar] = (n + 1, h + ok)
    I.own = {k: (len(keys), sum(responses[x] for x in keys)) for k, keys in own_keys.items()}
    # The ladder keeps only each concept's last _LADDER_WINDOW answers, so on
    # its own it can FABRICATE a break (one concept's old answers trimmed
    # away between another's); the attempt log is the untrimmed record.
    for k, t, *_rest in memory_model._log_answers(user_state):
        days.append(t)
        last_day[k] = max(t, last_day.get(k, t))
    # Sorted once here: return_start() re-sorts, and Timsort on a sorted list is linear.
    I.counts, I.answer_days, I.last_day = counts, sorted(days), last_day
    try:
        ref = weakref.ref(user_state, lambda _r, k=key: _memo.pop(k, None))
    except TypeError:  # not weak-referenceable (a plain object in a test)
        return I
    _memo[key] = (ref, fp, I)
    return I


def _state_inputs(user_state):
    I = _inputs(user_state)
    return I.reg, I.ev, I.level


def beliefs(user_state) -> Dict[str, float]:
    """P(known) per explore-area KC."""
    return {k: _sigmoid(x) for k, x in _inputs(user_state).L.items()}


def _modelled(kc: str) -> bool:
    """In the explore area AND in the graph — a concept the registry does not
    know has no belief, so nothing is built for it."""
    from app import kc_graph
    return in_area(kc) and kc in kc_graph._registry()


def settled(user_state, kc: str) -> bool:
    if not _modelled(kc):
        return False
    return _inputs(user_state).L.get(kc, -math.inf) >= SETTLE_LOGODDS


def explorable(user_state, kc: str) -> bool:
    """May be probed even though its prerequisites are not learned yet: open
    (not settled, not refuted), no refuted-and-unlearned prerequisite, and
    worth a probe's minutes (the cost gate, `worth_probing`)."""
    if not _modelled(kc):
        return False
    from app import kc_graph
    I = _inputs(user_state)
    L = I.L
    x = L.get(kc)
    if x is None or x >= SETTLE_LOGODDS or x <= OUT_LOGODDS:
        return False
    if not worth_probing(kc, x, leave_out(I.counts, kc, I.own.get(kc))):
        return False
    for p in I.reg[kc]["prereqs"]:
        if p in L:
            if L[p] <= OUT_LOGODDS and not kc_graph.kc_is_learned(user_state, p):
                return False
        elif not kc_graph.kc_is_learned(user_state, p):
            return False
    return True


def taught(user_state, kc: str) -> bool:
    """Shown the lesson by the ladder (`worked_seen`) or read it on their own
    (graph node, `?lesson=`)."""
    from app import kc_graph, lessons, practice_targets
    if int(kc_graph.ladder_view(user_state, kc).get("worked_seen") or 0):
        return True
    return bool(lessons.kc_lesson_read(kc, practice_targets.effective_exposure(user_state)))


def returning(user_state) -> bool:
    """Inside the RETURN WINDOW: back from a break of RETURN_GAP_DAYS or more,
    for RETURN_WINDOW_DAYS."""
    I = _inputs(user_state)
    return in_return_window(I.answer_days, time.time() / 86400.0)


def stale(user_state, kc: str) -> bool:
    """In the return window AND last answered before the break: its evidence is
    what the break faded. A concept answered since coming back (or first met
    after it) is not stale — so each concept is re-probed once per return, and
    one taught after the return keeps its lesson (codex, 2026-09-24)."""
    I = _inputs(user_state)
    now = time.time() / 86400.0
    start = return_start(I.answer_days, now)
    if start is None or now - start >= RETURN_WINDOW_DAYS:
        return False
    last = I.last_day.get(kc)
    return last is not None and last < start


def probing(user_state, kc: str) -> bool:
    """Serve as a diagnostic probe: no lesson first, no example on screen.
    Only while the learner has not been taught it — or the answer measures the
    page, not what they came with — except in the RETURN WINDOW, where a taught
    concept whose old answers have faded is probed again (unaided, never below
    its rung: kc_graph.kc_stage) — once per return, see `stale`."""
    if not explorable(user_state, kc):
        return False
    return not taught(user_state, kc) or stale(user_state, kc)


def _prelimit(I: _Inputs, kcs: List[str], value: Dict[str, float], n: int = 10) -> List[str]:
    """The `n` candidates with the most value-weighted variance: scoring each
    costs two posterior rebuilds. An approximation — a low-variance concept
    whose answer would move many correlated neighbours can be cut here — kept
    because the simulation that chose this design used the same cut."""
    def var(k):
        p = _sigmoid(I.L[k])
        return value.get(k, 1.0) * p * (1.0 - p)
    return sorted(kcs, key=lambda k: (-var(k), k))[:n]


def _value(user_state, reg) -> Dict[str, float]:
    """What knowing each concept is worth: the frontier's own ordering value,
    (descendants + 1) × the learner's weight for it (Graph Settings), so an
    up-weighted concept is worth more to resolve."""
    from app import kc_graph, kc_prefs
    descendants, _depth = kc_graph._closure()
    return {k: (descendants.get(k, 0) + 1) * kc_prefs.weight_for(user_state, k) for k in reg if in_area(k)}


def reorder(user_state, ordered: List[str]) -> List[str]:
    """`ordered` with the explore-area concepts re-sorted in the slots they
    already hold: open probes first, by expected information (explore), then
    everything else in its original greedy order (exploit). With a narrowed
    EXPLORE_PREFIXES the other areas keep their slots. Whole-graph (the
    default since 2026-09-24) every slot is in the area, so an open probe of
    ANY area goes ahead of greedy work — as in the simulation, where a probe
    pre-empted whatever the greedy pick was. The ARENA share is applied after
    this (remediation.targets → arena_mix.order), so it still holds."""
    slots = [i for i, k in enumerate(ordered) if in_area(k)]
    if len(slots) < 2:
        return ordered
    I = _inputs(user_state)
    area = [ordered[i] for i in slots]
    probes = [k for k in area if probing(user_state, k)]
    if not probes:
        return ordered
    value = _value(user_state, I.reg)
    top = _prelimit(I, probes, value)
    best = rank(I.reg, I.ev, top, value, I.level, graph=I.graph)
    best += [k for k in probes if k not in best] + [k for k in area if k not in probes]
    out = list(ordered)
    for i, kc in zip(slots, best):
        out[i] = kc
    return out


def return_probes(user_state) -> List[str]:
    """In the RETURN WINDOW: taught concepts worth re-probing, best first (by
    expected information); [] outside it. remediation.targets serves them ahead
    of the due reviews — one answer on a deep concept re-settles what it
    encompasses, which is cheaper than reviewing each of them."""
    if not returning(user_state):
        return []
    from app import kc_graph, kc_prefs
    I = _inputs(user_state)
    cands = [k for k in I.L
             if stale(user_state, k) and not kc_prefs.is_disabled(user_state, k)
             and kc_graph.questions_for_kc(k) and taught(user_state, k) and explorable(user_state, k)]
    if not cands:
        return []
    value = _value(user_state, I.reg)
    return rank(I.reg, I.ev, _prelimit(I, cands, value), value, I.level, graph=I.graph)
