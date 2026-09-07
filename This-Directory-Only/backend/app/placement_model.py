"""Graph-wide placement model — per-KC P(known) on the prerequisite DAG.

Seth, 2026-09-07: "I want the placement diagnostic to actually be
comprehensive. it should actually have a full estimate of the whole graph."
The old model (diagnostic.py before this date) held ONE ability number per
bank topic — four numbers for a 44-concept lattice — and smeared each over
every atom in its topic. This module holds one belief per KC and moves it the
way ALEKS moves a knowledge state (Cosyn, Uzun, Doble & Matayoshi 2021, JMP;
Doble et al. 2019, IJAIED):

  * Each KC k carries L_k = log-odds that the learner knows k. Prior from the
    learner's own atom-BKT history (their "population prior" — we have no
    population), clipped so a few probes can overrule it.
  * A probe is a Bayes-factor update with ASYMMETRIC strength. Our items are
    open-ended code, like ALEKS's: a lucky pass is rare, a careless miss is
    common, and "I don't know" is the strongest evidence of all (ALEKS's
    update multipliers are ~35 correct / 5 incorrect / 50 don't-know).
  * Evidence PROPAGATES along the surmise relation: a pass on k is evidence
    for k's prerequisites, a miss on k is evidence against k's dependents —
    attenuated per hop. Nothing flows the other way (knowing the parent says
    little about the child; failing the child says little about the parent).
  * Item selection: the KC whose probe most reduces VALUE-WEIGHTED variance
    across the whole graph, where value = coreness (descendants + 1) x the
    learner's own kc_prefs weight. That is Cosyn's "likelihood near 0.5" rule
    generalised from one item to a graph with unequal stakes.
  * Classification thresholds are ALEKS's: known above IN_STATE, unknown
    below OUT_OF_STATE, "uncertain" between — and uncertain concepts are the
    ones the learning mode fast-tracks afterwards.

TRANSFER IS NOT SETTLED HERE. ALEKS's structures are fitted on millions of
assessments; ours is hand-drawn and has one learner. So HOP_ATTENUATION_DOWN and PROP_CLIP are
deliberately conservative — propagation positions the NEXT probe, it does not
substitute for one — and `edge_violations` reports every edge the learner's
own evidence contradicts (child known, prerequisite unknown), which is the
only edge-fitting n=1 can do.

Pure functions over a probe log; no state of its own. Every constant is a v0
engineering choice.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

# --- response model ---------------------------------------------------------------
# P(correct | known) = 1 - SLIP, P(correct | unknown) = GUESS. Open-ended code:
# guessing a passing function is rare; a careless miss (typo, wrong axis) is not.
P_GUESS = 0.05
P_SLIP = 0.20
# "I don't know": almost no lucky-pass mass, and a knower rarely says it.
DK_GUESS = 0.02
DK_SLIP = 0.05

# Per-hop attenuation of propagated evidence along the prerequisite DAG.
# UP (a pass says its prerequisites are known): 0.5 = half a pass on the
# direct prerequisites, a quarter on theirs. DOWN (a miss says its dependents
# are not): weaker, because a miss may be a slip and a root has forty
# dependents — one careless error must not sink the whole course. A plain
# "incorrect" propagates down at half the strength of "I don't know".
HOP_ATTENUATION_UP = 0.5
HOP_ATTENUATION_DOWN = 0.3
INCORRECT_DOWN_SCALE = 0.5
MAX_HOPS = 3

# 🔴 NOTHING BUT A DIRECT PROBE CAN CLASSIFY A CONCEPT. Prior + everything a
# KC inherited from its neighbours is clipped, TOGETHER, to ±INDIRECT_CLIP
# log-odds, which keeps P(known) strictly inside (OUT_OF_STATE, IN_STATE)
# until the KC is probed itself. Without this, one slip on a root marked
# thirteen never-probed concepts "unknown" in simulation and the selector —
# seeing low variance — never went back for them; clipping prior and
# propagation SEPARATELY still let the two add up past the line. Indirect
# evidence positions the next probe; only a direct probe settles a concept.
# That is what makes the test comprehensive.
INDIRECT_CLIP = 1.2

# ALEKS's in-state / out-of-state cut-offs on P(known).
IN_STATE = 0.80
OUT_OF_STATE = 0.20

# The prior alone is clipped to this band before it joins the indirect ledger.
PRIOR_CLIP = 1.2
# Self-report shifts the prior by this much (log-odds) — a nudge, not a claim.
PRIOR_SHIFT_BY_LEVEL: Dict[Optional[str], float] = {
    "beginner": -0.5,
    None: 0.0,
    "strong": 0.5,
}

# --- log-odds helpers ------------------------------------------------------------


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def log_bayes_factor(result: str) -> float:
    """log P(response | known) - log P(response | unknown)."""
    if result == "correct":
        return math.log((1.0 - P_SLIP) / P_GUESS)          # ≈ +2.77
    if result == "dont_know":
        return math.log(DK_SLIP / (1.0 - DK_GUESS))        # ≈ -2.98
    return math.log(P_SLIP / (1.0 - P_GUESS))              # ≈ -1.56


def p_correct_given_belief(p_known: float) -> float:
    return p_known * (1.0 - P_SLIP) + (1.0 - p_known) * P_GUESS


# --- graph -------------------------------------------------------------------------


class Graph:
    """The prerequisite DAG restricted to the KCs under assessment, with the
    hop-distance tables propagation and selection both read."""

    def __init__(self, prereqs: Dict[str, Iterable[str]], kcs: Iterable[str]):
        self.kcs: List[str] = [k for k in kcs]
        keep = set(self.kcs)
        self.parents: Dict[str, List[str]] = {
            k: [p for p in prereqs.get(k, ()) if p in keep] for k in self.kcs
        }
        self.children: Dict[str, List[str]] = {k: [] for k in self.kcs}
        for k, ps in self.parents.items():
            for p in ps:
                self.children[p].append(k)
        # hop distance to every ancestor / descendant within MAX_HOPS
        self.ancestors: Dict[str, Dict[str, int]] = {k: self._walk(k, self.parents) for k in self.kcs}
        self.descendants: Dict[str, Dict[str, int]] = {k: self._walk(k, self.children) for k in self.kcs}

    @staticmethod
    def _walk(start: str, adj: Dict[str, List[str]]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        frontier = [(n, 1) for n in adj.get(start, ())]
        while frontier:
            n, d = frontier.pop(0)
            if n == start or d > MAX_HOPS or (n in dist and dist[n] <= d):
                continue
            dist[n] = d
            frontier.extend((m, d + 1) for m in adj.get(n, ()))
        return dist


# --- posterior -----------------------------------------------------------------------


def prior_logodds(p_known: Optional[float], level: Optional[str]) -> float:
    base = logit(p_known) if p_known is not None else 0.0
    base = max(-PRIOR_CLIP, min(PRIOR_CLIP, base))
    return base + PRIOR_SHIFT_BY_LEVEL.get(level, 0.0)


class Beliefs:
    """Log-odds per KC in two ledgers: `direct` (the KC's own probes) and
    `indirect` (prior + what flowed in from neighbours). The total clips the
    indirect ledger — see INDIRECT_CLIP."""

    __slots__ = ("direct", "indirect")

    def __init__(self, priors: Dict[str, float]):
        self.direct = {k: 0.0 for k in priors}
        self.indirect = dict(priors)

    def copy(self) -> "Beliefs":
        b = Beliefs.__new__(Beliefs)
        b.direct = dict(self.direct)
        b.indirect = dict(self.indirect)
        return b

    def logodds(self, kc: str) -> float:
        return self.direct[kc] + max(-INDIRECT_CLIP, min(INDIRECT_CLIP, self.indirect[kc]))

    def p(self, kc: str) -> float:
        return sigmoid(self.logodds(kc))

    def probs(self) -> Dict[str, float]:
        return {k: self.p(k) for k in self.direct}

    def __contains__(self, kc) -> bool:
        return kc in self.direct


def apply_probe(B: Beliefs, graph: Graph, kc: str, result: str, weight: float = 1.0) -> None:
    """Move the belief on `kc` and propagate along the DAG, in place.

    A pass flows UP to prerequisites (you cannot do the child without them);
    a miss flows DOWN to dependents (you will not do them without this)."""
    if kc not in B:
        return
    bf = log_bayes_factor(result) * weight
    B.direct[kc] += bf
    if bf > 0:
        for other, hops in graph.ancestors.get(kc, {}).items():
            B.indirect[other] += bf * (HOP_ATTENUATION_UP ** hops)
    else:
        scale = INCORRECT_DOWN_SCALE if result == "incorrect" else 1.0
        for other, hops in graph.descendants.get(kc, {}).items():
            B.indirect[other] += bf * scale * (HOP_ATTENUATION_DOWN ** hops)


def beliefs_from(priors: Dict[str, float], graph: Graph, probes: Iterable[dict]) -> Beliefs:
    B = Beliefs(priors)
    for probe in probes:
        kc = probe.get("kc")
        if kc in B:
            apply_probe(B, graph, kc, probe.get("result", "incorrect"), float(probe.get("w", 1.0)))
    return B


def posterior(
    priors: Dict[str, float],
    graph: Graph,
    probes: Iterable[dict],
) -> Dict[str, float]:
    """P(known) per KC from the prior log-odds and the probe log. Recomputed
    from scratch every call — the log is the state, nothing derived drifts."""
    return beliefs_from(priors, graph, probes).probs()


def classify(p: float) -> str:
    if p >= IN_STATE:
        return "known"
    if p <= OUT_OF_STATE:
        return "unknown"
    return "uncertain"


# --- selection ---------------------------------------------------------------------------


def weighted_variance(P: Dict[str, float], value: Dict[str, float]) -> float:
    """A KC absent from `value` counts for NOTHING — so `{kc: 1.0}` measures
    that one concept alone (the stop rule), not the whole graph."""
    return sum(value.get(k, 0.0) * p * (1.0 - p) for k, p in P.items())


def expected_variance_reduction(
    B: Beliefs,
    graph: Graph,
    kc: str,
    value: Dict[str, float],
) -> float:
    """How much value-weighted Bernoulli variance ONE probe of `kc` is expected
    to remove across the whole graph — the two outcomes weighed by their
    predictive probability, propagation included. This is the number the
    selector maximises; on a graph with equal values and no edges it reduces
    to "probe the KC nearest 0.5", i.e. ALEKS's rule."""
    P = B.probs()
    now = weighted_variance(P, value)
    p_c = p_correct_given_belief(P[kc])
    total = 0.0
    for result, prob in (("correct", p_c), ("incorrect", 1.0 - p_c)):
        B2 = B.copy()
        apply_probe(B2, graph, kc, result)
        total += prob * (now - weighted_variance(B2.probs(), value))
    return total


def rank_kcs(
    priors: Dict[str, float],
    graph: Graph,
    probes: Iterable[dict],
    value: Dict[str, float],
    candidates: Iterable[str],
) -> List[Tuple[float, str]]:
    """Candidates sorted by expected value-weighted variance reduction, best
    first. Ties break on id so two runs of one state never disagree."""
    return rank_beliefs(beliefs_from(priors, graph, probes), graph, value, candidates)


def rank_beliefs(
    B: Beliefs,
    graph: Graph,
    value: Dict[str, float],
    candidates: Iterable[str],
) -> List[Tuple[float, str]]:
    scored = [(expected_variance_reduction(B, graph, k, value), k) for k in candidates if k in B]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


# --- what the learner's evidence says about the graph itself -----------------------------


def edge_violations(P: Dict[str, float], graph: Graph) -> List[dict]:
    """Edges the learner contradicts: a dependent classified KNOWN whose
    prerequisite is classified UNKNOWN. With one learner this is a list of
    suspects, not a fit — each is either a wrong edge, an edge that is too
    strong, or a slip; the report exists so a human can look."""
    out = []
    for child, parents in graph.parents.items():
        if classify(P.get(child, 0.5)) != "known":
            continue
        for parent in parents:
            if classify(P.get(parent, 0.5)) == "unknown":
                out.append({
                    "prereq": parent,
                    "dependent": child,
                    "p_prereq": round(P[parent], 3),
                    "p_dependent": round(P[child], 3),
                })
    out.sort(key=lambda r: (r["prereq"], r["dependent"]))
    return out
