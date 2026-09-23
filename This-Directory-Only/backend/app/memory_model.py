"""memory_model.py — per-concept memory: FSRS-6 plus Fractional Implicit Repetition.

WHY THIS EXISTS
---------------
Until 2026-09-23 every concept forgot on ONE clock: `bkt_mastery.HALF_LIFE_DAYS`
and the engine's `recency_half_life_days`, both 14 days, both unsourced (the
papers/VERIFICATION_A audit said so). One scalar with one global half-life
cannot represent the spacing effect: three answers crammed into an hour and
three answers spread over three weeks leave the same belief, so the same
future. And nothing scheduled review at all — a learned concept only came back
as filler once the frontier ran dry.

Seth, 2026-09-23: "Why does it have just the same half-life always? It needs
to be much more adaptive than that." The scratchpad simulation run that day
(hierarchical graph, ACT-R learners) found FSRS with FIRe needed ~74 reviews
per learner to hold 0.90 true recall where FSRS alone needed ~135 and the
BKT-with-one-half-life structure ~150. This module is that winner.

THE MODEL
---------
Per concept, FSRS-6's state (S = stability in days, D = difficulty 1..10) and
its power forgetting curve R(t) = (1 + F·t/S)^-w20. The update rules are a
port of py-fsrs 6.3.2 (`fsrs/scheduler.py`) with its published default
weights, fitted on real review histories — flashcards, not problem practice, so
they are a starting point to be refit from `attempt_log`, not a finding. Seth
has a deep-research pass out on concept-level priors
(docs/research-question-fsrs-priors.md); its answer lands in `MemoryConfig`.

Grades: a correct unaided answer is Good, a correct answer made behind a worked
example is Hard (it happened, but with help), a miss or a timeout is Again.

Implicit repetition (Math Academy's FIRe), over `kc_registry.json`'s
`encompassing` weights, multiplied down paths:

  * a CORRECT answer on X is a fractional Good review of every concept X
    encompasses, of weight w: stability moves w of the way to what a full
    review would give (so the early-review discount in FSRS's own formula
    applies — credit arriving while a component is fresh buys almost nothing),
    and R moves w of the way to 1 through a VIRTUAL last-review time. The
    clock is never simply reset: `bkt_mastery.apply_attempt` does reset it,
    which hands a w = 0.1 edge the whole of a fresh review's recency.
  * a MISS on X is blamed on X's own step and on each component in proportion
    to w·(1 − R_c): a component that was just practised is not the likely
    culprit. Components take a partial lapse of their share (S shrinks, R is
    kept); X takes the rest.
  * a miss on X also lapses each concept that encompasses X, at w·UPWARD_LAPSE:
    if X is gone, a problem that needs X is probably gone too.

WHAT READS IT
-------------
  * `due_reviews` — learned concepts whose R has fallen below the target,
    ordered with Math Academy's compression: a concept whose problems also
    cover other due concepts goes first. `remediation.targets` interleaves
    these with the frontier.
  * `retrievability` — the engine's `recency` feature is 1 − R
    (`engine_features`, logistic-v0.4).

STATELESS
---------
Replayed from `kc_ladder[kc]["attempts"]` every time, like `remediation`, so a
replay of a learner's state reaches the same decisions and a change of weights
takes effect on the next request with no migration. The ladder keeps the last
20 attempts per concept (`kc_ladder_math._LADDER_WINDOW`); for a concept with
more, the replay starts at the 21st-newest, which UNDERSTATES its stability —
reviews come early, never late. `attempt_log` is the full record to move to
once every learner has one.
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Tuple

from app import kc_graph, kc_prefs, practice_targets

# py-fsrs 6.3.2 DEFAULT_PARAMETERS, w0..w20 (w20 = the curve's decay).
FSRS6_DEFAULT_WEIGHTS: Tuple[float, ...] = (
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722,
    0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425,
    0.0912, 0.0658, 0.1542,
)

AGAIN, HARD, GOOD = 1, 2, 3
_EASY = 4
_D_MIN, _D_MAX = 1.0, 10.0
_S_MIN, _S_MAX = 0.001, 36500.0
# Ladder rows for one question are appended one KC at a time, each with its
# own `datetime.now()`; rows for the same question this close together are one
# answer.
_SAME_EVENT_DAYS = 5.0 / 86400.0


@dataclass(frozen=True)
class MemoryConfig:
    version: str = "fsrs6-fire-v1"
    weights: Tuple[float, ...] = FSRS6_DEFAULT_WEIGHTS
    # Review when predicted recall falls to this. Seth, 2026-09-23: 0.80 —
    # Math Academy's "about a 20% chance of getting it wrong". FSRS's own
    # default is 0.90. One number for now; per-concept importance is later.
    target_retention: float = 0.80
    # Share of a miss on X passed UP to each concept encompassing X, times the
    # edge weight. 0.5 is the simulation's choice, not a measurement.
    upward_lapse: float = 0.5
    fire: bool = True

    @property
    def decay(self) -> float:
        return -self.weights[20]

    @property
    def factor(self) -> float:
        return 0.9 ** (1.0 / self.decay) - 1.0


DEFAULT_CONFIG = MemoryConfig()


@dataclass(frozen=True)
class Memory:
    S: float          # stability, days (R = 0.9 after S days)
    D: float          # difficulty, 1..10
    t_last: float     # last review, days since epoch — VIRTUAL after implicit credit
    reps: float       # explicit reviews + fractional implicit ones
    lapses: int


# --- FSRS-6 (port of py-fsrs 6.3.2 Scheduler) ------------------------------

def _clamp_s(s: float) -> float:
    return min(max(s, _S_MIN), _S_MAX)


def _clamp_d(d: float) -> float:
    return min(max(d, _D_MIN), _D_MAX)


def _init_s(g: int, w) -> float:
    return _clamp_s(w[g - 1])


def _init_d(g: int, w, clamp: bool = True) -> float:
    d = w[4] - math.exp(w[5] * (g - 1)) + 1
    return _clamp_d(d) if clamp else d


def _next_d(d: float, g: int, w) -> float:
    delta = -(w[6] * (g - 3))
    damped = d + (10.0 - d) * delta / 9.0
    return _clamp_d(w[7] * _init_d(_EASY, w, clamp=False) + (1 - w[7]) * damped)


def _short_term_s(s: float, g: int, w) -> float:
    inc = math.exp(w[17] * (g - 3 + w[18])) * s ** -w[19]
    if g >= HARD:
        inc = max(inc, 1.0)
    return _clamp_s(s * inc)


def _recall_s(d: float, s: float, r: float, g: int, w) -> float:
    hard = w[15] if g == HARD else 1.0
    return _clamp_s(s * (1 + math.exp(w[8]) * (11 - d) * s ** -w[9]
                         * (math.exp((1 - r) * w[10]) - 1) * hard))


def _forget_s(d: float, s: float, r: float, w) -> float:
    long_term = w[11] * d ** -w[12] * ((s + 1) ** w[13] - 1) * math.exp((1 - r) * w[14])
    return _clamp_s(min(long_term, s / math.exp(w[17] * w[18])))


def retrievability(mem: Memory, t: float, cfg: MemoryConfig = DEFAULT_CONFIG) -> float:
    """Predicted P(recall) at `t` (days since epoch)."""
    elapsed = max(0.0, t - mem.t_last)
    return (1 + cfg.factor * elapsed / mem.S) ** cfg.decay


def _elapsed_for(r: float, s: float, cfg: MemoryConfig) -> float:
    """Days since review at which the curve with stability `s` reads `r`."""
    r = min(max(r, 1e-9), 1.0)
    return s / cfg.factor * (r ** (1.0 / cfg.decay) - 1.0)


def _after(mem: Memory, t: float, g: int, cfg: MemoryConfig) -> Tuple[float, float]:
    """(S, D) after a full review graded `g` at `t` — py-fsrs's Review state:
    same-day reviews take the short-term rule, later ones the long-term."""
    w = cfg.weights
    if t - mem.t_last < 1.0:
        s = _short_term_s(mem.S, g, w)
    elif g == AGAIN:
        s = _forget_s(mem.D, mem.S, retrievability(mem, t, cfg), w)
    else:
        s = _recall_s(mem.D, mem.S, retrievability(mem, t, cfg), g, w)
    return s, _next_d(mem.D, g, w)


def review(mem: Optional[Memory], t: float, g: int,
           cfg: MemoryConfig = DEFAULT_CONFIG) -> Memory:
    """A full, explicit review."""
    if mem is None:
        w = cfg.weights
        return Memory(_init_s(g, w), _init_d(g, w), t, 1.0, int(g == AGAIN))
    s, d = _after(mem, t, g, cfg)
    return Memory(s, d, t, mem.reps + 1, mem.lapses + int(g == AGAIN))


def implicit_review(mem: Optional[Memory], t: float, weight: float,
                    cfg: MemoryConfig = DEFAULT_CONFIG) -> Optional[Memory]:
    """A fractional Good review of weight `weight` (FIRe credit). weight 1 is
    exactly `review(mem, t, GOOD)`; weight 0 changes nothing. A concept never
    practised has no memory to strengthen and stays None."""
    if mem is None or weight <= 0:
        return mem
    w = min(weight, 1.0)
    r = retrievability(mem, t, cfg)
    s_full, d_full = _after(mem, t, GOOD, cfg)
    s = _clamp_s(mem.S + w * (s_full - mem.S))
    d = _clamp_d(mem.D + w * (d_full - mem.D))
    r_new = r + w * (1 - r)
    return Memory(s, d, t - _elapsed_for(r_new, s, cfg), mem.reps + w, mem.lapses)


def partial_lapse(mem: Optional[Memory], t: float, share: float,
                  cfg: MemoryConfig = DEFAULT_CONFIG) -> Optional[Memory]:
    """`share` of a lapse: S moves that far toward FSRS's post-lapse stability,
    D toward Again's, and R is KEPT — nobody was shown this concept's answer."""
    if mem is None or share <= 0:
        return mem
    share = min(share, 1.0)
    r = retrievability(mem, t, cfg)
    s_f, d_f = _after(mem, t, AGAIN, cfg)
    s = _clamp_s(mem.S + share * (s_f - mem.S))
    d = _clamp_d(mem.D + share * (d_f - mem.D))
    return Memory(s, d, t - _elapsed_for(r, s, cfg), mem.reps, mem.lapses)


# --- the encompassing graph -------------------------------------------------

_graph_cache: Dict[int, Tuple[dict, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]] = {}


def _graph() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """(closure, ancestors). closure[x] = {c: weight} for everything a problem
    on x implicitly exercises, weights multiplied down each path, max over
    paths. ancestors[x] = {p: weight} is the same relation read upward: every
    concept whose problems exercise x, directly or through a chain."""
    reg = kc_graph._registry()
    hit = _graph_cache.get(id(reg))
    if hit is not None and hit[0] is reg:
        return hit[1], hit[2]
    direct = {
        k: {c: float(w) for c, w in (node.get("encompassing") or {}).items()
            if c in reg and c != k and float(w) > 0}
        for k, node in reg.items()
    }
    closure: Dict[str, Dict[str, float]] = {}
    for x in direct:
        out: Dict[str, float] = {}
        stack = [(x, 1.0, frozenset([x]))]
        while stack:
            node, acc, path = stack.pop()
            for c, w in direct.get(node, {}).items():
                if c in path:
                    continue
                v = acc * min(w, 1.0)
                if v > out.get(c, 0.0):
                    out[c] = v
                    stack.append((c, v, path | {c}))
        closure[x] = out
    ancestors: Dict[str, Dict[str, float]] = {}
    for p, comps in closure.items():
        for c, w in comps.items():
            ancestors.setdefault(c, {})[p] = w
    _graph_cache.clear()
    _graph_cache[id(reg)] = (reg, closure, ancestors)
    return closure, ancestors


# --- replay -----------------------------------------------------------------

def _to_days(ts) -> Optional[float]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0


def _now_days(now: Optional[datetime]) -> float:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.timestamp() / 86400.0


def _grade(att: Mapping) -> int:
    if not att.get("correct"):
        return AGAIN
    return HARD if att.get("example") else GOOD


def _events(user_state) -> List[Tuple[float, Dict[str, int]]]:
    """Every graded answer as (t, {kc: grade}), oldest first. One question
    tagged with several concepts wrote one ladder row per concept; those rows
    are one event, so a concept tagged directly is never also credited
    implicitly by its sibling tag."""
    ladder = getattr(user_state, "kc_ladder", None) or {}
    rows = []
    for kc, row in ladder.items():
        if not isinstance(row, dict):
            continue
        for i, att in enumerate(row.get("attempts") or []):
            if not isinstance(att, dict):
                continue
            t = _to_days(att.get("ts") or att.get("timestamp"))
            if t is None:
                continue
            rows.append((t, att.get("question_id"), kc, i, _grade(att)))
    rows.sort(key=lambda r: (r[0], str(r[1]), r[2], r[3]))
    events: List[Tuple[float, Optional[int], Dict[str, int]]] = []
    for t, qid, kc, _i, g in rows:
        if events:
            t0, q0, grades = events[-1]
            if (qid is not None and qid == q0 and t - t0 <= _SAME_EVENT_DAYS
                    and kc not in grades):
                grades[kc] = g
                continue
        events.append((t, qid, {kc: g}))
    return [(t, grades) for t, _q, grades in events]


def _apply(mems: Dict[str, Memory], t: float, grades: Dict[str, int],
           cfg: MemoryConfig) -> None:
    closure, ancestors = _graph() if cfg.fire else ({}, {})
    tagged = set(grades)
    # Everything is judged on `pre`, the state the learner walked in with, and
    # each untagged concept takes ONE combined update per answer: two tags that
    # both encompass c are one problem, not two implicit repetitions of c, and
    # the result must not depend on the order the tags are visited in.
    pre = dict(mems)
    credit: Dict[str, float] = {}
    lapse: Dict[str, float] = {}
    for x in sorted(tagged):
        if grades[x] != AGAIN:
            mems[x] = review(pre.get(x), t, grades[x], cfg)
            for c, w in closure.get(x, {}).items():
                if c not in tagged and c in pre:
                    credit[c] = max(credit.get(c, 0.0), w)
    for x in sorted(tagged):
        if grades[x] != AGAIN:
            continue
        own = pre.get(x)
        # A component a correct sibling tag just exercised is not blamed.
        comps = {c: w for c, w in closure.get(x, {}).items()
                 if c not in tagged and c in pre and c not in credit}
        blame = {x: 1.0 - retrievability(own, t, cfg) if own else 1.0}
        for c, w in comps.items():
            blame[c] = w * (1.0 - retrievability(pre[c], t, cfg))
        z = sum(blame.values())
        for c, b in blame.items():
            if c != x and z > 0:
                lapse[c] = lapse.get(c, 0.0) + b / z
        if own is None:
            mems[x] = review(None, t, AGAIN, cfg)
        else:
            share = blame[x] / z if z > 0 else 1.0
            s_f, d_f = _after(own, t, AGAIN, cfg)
            mems[x] = Memory(
                _clamp_s(own.S + share * (s_f - own.S)),
                _clamp_d(own.D + share * (d_f - own.D)),
                t, own.reps + 1, own.lapses + 1,
            )
        for p, w in ancestors.get(x, {}).items():
            if p in pre and p not in tagged and p not in credit:
                lapse[p] = lapse.get(p, 0.0) + w * cfg.upward_lapse
    for c, w in credit.items():
        mems[c] = implicit_review(pre[c], t, w, cfg)
    for c, share in lapse.items():
        mems[c] = partial_lapse(pre[c], t, min(1.0, share), cfg)


def replay_events(events, cfg: MemoryConfig = DEFAULT_CONFIG) -> Dict[str, Memory]:
    mems: Dict[str, Memory] = {}
    for t, grades in events:
        _apply(mems, t, grades, cfg)
    return mems


# Per-state memo: the picker asks several times per request. Keyed weakly on
# the state object and checked against a fingerprint of every ladder row's
# length and newest row, so an appended answer invalidates it.
_memo: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _fingerprint(user_state) -> tuple:
    ladder = getattr(user_state, "kc_ladder", None) or {}
    out = []
    for kc in sorted(ladder):
        row = ladder[kc]
        atts = row.get("attempts") if isinstance(row, dict) else None
        if atts:
            last = atts[-1] if isinstance(atts[-1], dict) else {}
            out.append((kc, len(atts), last.get("ts") or last.get("timestamp"),
                        last.get("question_id"), last.get("correct"), last.get("example")))
    return tuple(out)


def memories(user_state, cfg: MemoryConfig = DEFAULT_CONFIG) -> Dict[str, Memory]:
    """{kc: Memory} for every concept with a graded answer, replayed."""
    fp = (cfg, _fingerprint(user_state))
    try:
        hit = _memo.get(user_state)
    except TypeError:  # not weak-referenceable (a plain dict in a test)
        hit = None
    if hit is not None and hit[0] == fp:
        return hit[1]
    mems = replay_events(_events(user_state), cfg)
    try:
        _memo[user_state] = (fp, mems)
    except TypeError:
        pass
    return mems


def kc_retrievability(user_state, kc: str, now: Optional[datetime] = None,
                      cfg: MemoryConfig = DEFAULT_CONFIG,
                      exclude_latest: bool = False) -> Optional[float]:
    """Predicted recall of `kc` now, or None if it has never been answered.

    `exclude_latest`: replay WITHOUT the newest answer touching `kc` —
    directly, or implicitly through the encompassing graph. The scoring path
    needs it — the ladder row for the answer being scored is written before
    the engine scores it, and a prediction that has seen its own outcome is
    not a prediction (same rule as engine_bridge.posteriors_for's
    `exclude_latest_attempt`)."""
    if not exclude_latest:
        mem = memories(user_state, cfg).get(kc)
    else:
        closure, ancestors = _graph() if cfg.fire else ({}, {})

        def touches(grades) -> bool:
            return any(k == kc or kc in closure.get(k, {}) or kc in ancestors.get(k, {})
                       for k in grades)

        evs = _events(user_state)
        last = max((i for i, (_t, g) in enumerate(evs) if touches(g)), default=None)
        if last is None:
            return None
        mem = replay_events(evs[:last] + evs[last + 1:], cfg).get(kc)
    return None if mem is None else retrievability(mem, _now_days(now), cfg)


def recency(user_state, kc: str, now: Optional[datetime] = None,
            exclude_latest: bool = False) -> Tuple[float, Optional[float]]:
    """(the logistic engine's `recency` feature, R). The feature is 1 − R; a
    concept never answered has nothing to forget and reads 0.0. Both feature
    builders (engine_bridge, engine_features) go through here so they cannot
    drift apart."""
    recall = kc_retrievability(user_state, kc, now=now, exclude_latest=exclude_latest)
    return (0.0 if recall is None else 1.0 - recall), recall


def due_at(mem: Memory, cfg: MemoryConfig = DEFAULT_CONFIG) -> float:
    """Days since epoch at which R reaches the target."""
    return mem.t_last + _elapsed_for(cfg.target_retention, mem.S, cfg)


def _reviewable(user_state, kc: str) -> bool:
    return (
        practice_targets.includes(user_state, kc)
        and not kc_prefs.is_disabled(user_state, kc)
        and kc_graph.kc_is_learned(user_state, kc)
    )


def due_reviews(user_state, now: Optional[datetime] = None,
                cfg: MemoryConfig = DEFAULT_CONFIG) -> List[str]:
    """Learned concepts to review, best first.

    Due = learned, answered at least once, R below the target. Order is Math
    Academy's repetition compression: each learned concept scores 1 if it is
    due itself plus the implicit weight its problems carry onto due concepts,
    and the highest score goes first — so one problem on an encompassing
    concept can stand in for several reviews. A concept that is not due but
    covers due ones may be listed; that is the compression. Ties go to the
    lowest R."""
    mems = memories(user_state, cfg)
    if not mems:
        return []
    t = _now_days(now)
    rs = {k: retrievability(m, t, cfg) for k, m in mems.items()}
    learned = {k for k in mems if _reviewable(user_state, k)}
    due = {k for k in learned if rs[k] < cfg.target_retention}
    if not due:
        return []
    # Compression only pays when a problem on x actually refreshes what it
    # covers; with FIRe off it would schedule proxies that leave the due
    # concepts due.
    closure = _graph()[0] if cfg.fire else {}

    def score(x: str) -> float:
        return (1.0 if x in due else 0.0) + sum(
            w for c, w in closure.get(x, {}).items() if c in due)

    ranked = [(score(x), x) for x in learned]
    ranked = [(sc, x) for sc, x in ranked if sc > 0]
    ranked.sort(key=lambda p: (-p[0], rs[p[1]], p[1]))
    return [x for _sc, x in ranked]


def report(user_state, now: Optional[datetime] = None,
           cfg: MemoryConfig = DEFAULT_CONFIG) -> Dict[str, dict]:
    """Per-concept memory for inspection: S, D, R now, due date, reps."""
    t = _now_days(now)
    out = {}
    for kc, m in sorted(memories(user_state, cfg).items()):
        due = due_at(m, cfg)
        out[kc] = {
            "stability_days": round(m.S, 3),
            "difficulty": round(m.D, 3),
            "retrievability": round(retrievability(m, t, cfg), 4),
            "due_in_days": round(due - t, 2),
            "reps": round(m.reps, 2),
            "lapses": m.lapses,
        }
    return out
