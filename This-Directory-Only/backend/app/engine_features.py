"""engine_features.py — the concept graph, expressed as model inputs.

WHERE THIS SITS
---------------
`logistic_engine.py` is deliberately pure arithmetic with no app imports: it
knows about "a feature called prereq with value -0.2" and nothing about
prerequisites. This module is the other half — it reads the KC lattice, the
crosswalk, and the BKT atom posteriors, and materialises the feature vector the
engine consumes.

The split is the point. Adding a relation to the graph means adding a function
here and a `Feature` there; it never means touching the estimator's arithmetic.
That is what "I can develop it as I go as I implement more types of edges" has
to mean structurally, rather than as an intention.

    concept graph ──▶ engine_features ──▶ logistic_engine ──▶ Prediction
    (kc_graph,          (this file:         (pure math:
     bkt_mastery)        edges -> numbers)   numbers -> P)

TWO EDGE TYPES TODAY, AND ROOM FOR MORE
---------------------------------------
  * **prerequisite** — `kc_registry.json` `prereqs`. A hard gate for sequencing
    (`kc_graph.kc_is_unlocked`, unchanged) AND, new here, a soft feature. The
    two answer different questions and we want both: the gate decides what may
    be shown, the feature decides how likely it is to go well and *why not*.
    Without the soft version there is no way to attribute a failure upward to a
    weak parent, so remediation can only ever drill the child harder.

  * **encompassing** — `arena_drillable_v1.json` edges, the channel FIRe already
    uses. Re-expressed here as a feature so its weight becomes estimable from
    the log rather than asserted by a propagation constant.

OFF-GRAPH PREREQUISITES
-----------------------
Not every prerequisite a problem leans on has a node. A tensor problem may
depend on for-loops, tuple unpacking, or a definition that no ARENA-derived
concept covers. `item_prereq_tags` handles these: tags that resolve to a KC are
scored from the learner's record, and tags that do not are RECORDED but scored
neutral. Recording them is the useful half — the log then accumulates exactly
the list of off-graph dependencies the bank actually leans on, which is the
evidence for promoting one to a real node later. Guessing a mastery value for an
entity we have never measured would be inventing data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app import bkt_mastery, kc_graph
from app import logistic_engine as E
from app.attempt_log import parse_ts

logger = logging.getLogger(__name__)


def _days_since(ts: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    """Days between `ts` and now, using the log's own timestamp parser.

    Deliberately NOT a second implementation. This module and `attempt_log` both
    need to read the same timestamps out of the same state, and two parsers with
    different naive/aware handling would make the recency feature and the replay
    clock disagree about when an attempt happened.
    """
    parsed = parse_ts(ts)
    if parsed is None:
        return None
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return max((ref - parsed).total_seconds() / 86400.0, 0.0)


# ---------------------------------------------------------------------------
# Edge readers
# ---------------------------------------------------------------------------


def prereq_mastery(user_state, kc: str) -> Dict[str, float]:
    """Mastery of each direct prerequisite of `kc`, keyed by prerequisite id.

    Direct parents only, not the transitive closure. Transitive prerequisites
    are already reflected in the direct parents' own mastery, so summing the
    closure would count a grandparent twice and let deep chains dominate the
    feature purely by being deep.

    Returns a mapping rather than two parallel lists. The parallel-list version
    dropped failed lookups from the values while keeping every id in the names,
    so one unreadable parent shifted the zip and attributed each remaining
    mastery figure to the WRONG concept — an explanation that names the wrong
    prerequisite is worse than no explanation. A dict cannot desynchronise.
    """
    node = kc_graph.registry_node(kc) or {}
    parents = [p for p in (node.get("prereqs") or []) if isinstance(p, str)]
    out: Dict[str, float] = {}
    for p in parents:
        try:
            m, _covered, _tier = kc_graph.kc_mastery(user_state, p)
        except Exception:  # noqa: BLE001 — a graph read must never fail a serve
            logger.exception("engine_features: kc_mastery failed for prereq %s", p)
            continue
        out[p] = float(m)
    return out


def encompassed_mastery(user_state, kc: str) -> Dict[str, float]:
    """Mastery of the simpler atoms this KC's atoms encompass.

    Direction matters and is easy to get backwards: `encompassed_by(advanced)`
    returns the SIMPLER atoms that receive credit when the advanced one is
    cleared. So for a KC we are about to serve, these are the sub-skills it
    subsumes — knowing them should raise P(correct), and that is the sign the
    feature carries.
    """
    row = kc_graph._crosswalk().get(kc) or {}
    atoms = [a.get("a") for a in (row.get("atoms") or []) if a.get("a")]
    params = bkt_mastery.params_for_level(getattr(user_state, "self_reported_level", None))

    simpler: Dict[str, float] = {}
    for atom in atoms:
        for simple_atom, _w in bkt_mastery.encompassed_by(atom):
            if simple_atom in simpler:
                continue
            simpler[simple_atom] = bkt_mastery.current_mastery(
                user_state.atom_mastery, user_state.atom_last_ts, simple_atom, params=params
            )
    return simpler


def item_prereq_tags(question: Mapping) -> Tuple[List[str], List[str]]:
    """Split a question's prerequisite tags into (on-graph KCs, off-graph tags).

    Reads `prereq_tags` from the question record — the field the bank uses to
    say "this tensor problem also leans on for-loops". Tags naming a real KC are
    returned first; the rest are returned separately so the caller can log them.

    Both lists are returned even when empty, because "this question declared no
    prerequisites" and "this question declared prerequisites we could not
    resolve" are different facts and the log should be able to tell them apart.
    """
    raw = question.get("prereq_tags") or question.get("prerequisite_tags") or []
    tags = [str(t).strip() for t in raw if str(t).strip()]
    registry = kc_graph._registry()
    on_graph = [t for t in tags if t in registry]
    off_graph = [t for t in tags if t not in registry]
    return on_graph, off_graph


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


class FeatureVector(dict):
    """A materialised design row, plus the provenance behind it.

    Subclasses dict so it can be handed straight to `engine.predict` and written
    straight to the log, while carrying the `sources` the numbers came from.
    Those sources are what make a prediction explainable — "this looks hard
    because `numpy.broadcasting` is weak" — instead of an unarguable number.
    """

    def __init__(self, values: Mapping[str, float], sources: Optional[Mapping] = None):
        super().__init__({k: float(v) for k, v in values.items()})
        self.sources: Dict = dict(sources or {})


def build(
    user_state,
    kc: Optional[str],
    stage: str,
    *,
    question: Optional[Mapping] = None,
    difficulty_score: Optional[float] = None,
    config: E.EngineConfig = E.DEFAULT_CONFIG,
    now: Optional[datetime] = None,
) -> FeatureVector:
    """Materialise every feature value for one (learner, KC, item, rung).

    Called at SERVE time, and the result is what gets logged — not recomputed
    after grading. Recomputing later would read post-outcome state and quietly
    make the model look better than it is.

    Every read is defensive: a missing crosswalk row, an unmapped KC, or an
    unscored question yields a neutral value rather than an exception. A serve
    that fails because the graph is incomplete is a worse failure than a serve
    made on a weaker feature vector, and the graph is known to be incomplete
    (43 of 63 KCs are `topic-proxy`).
    """
    values: Dict[str, float] = {
        # Design-matrix entry for the LEARNED ability term: this learner's
        # ability applies to this item, at unit weight.
        "ability": 1.0,
        "difficulty": E.difficulty_to_logits(difficulty_score, config),
        "stage": E.stage_offset(stage, config),
    }
    sources: Dict = {"kc": kc, "stage": E.normalize_stage(stage)}

    # Prerequisites are accumulated into ONE dict keyed by concept id, so a
    # concept named by both the graph and the item's own tags is counted once.
    # Averaging two parallel lists instead let overlap double a parent's weight:
    # with parents A=1.0, B=0.0 and an item tag also naming A, the mean became
    # 2/3 rather than 1/2 — the prediction moved on duplicated metadata rather
    # than on anything the learner did.
    prereqs: Dict[str, float] = {}

    if kc:
        prereqs.update(prereq_mastery(user_state, kc))

        encompassed = encompassed_mastery(user_state, kc)
        values["encompassing"] = E.centred_mastery(encompassed.values())
        sources["encompassed"] = encompassed

        # One clock read, used for both the feature and its provenance. Reading
        # twice let the logged explanation disagree with the number it explains.
        days = _last_practised_days(user_state, kc, now)
        values["recency"] = E.recency_value(days, config)
        sources["days_since_kc"] = days
    else:
        values["encompassing"] = 0.0
        values["recency"] = 0.0

    if question is not None:
        on_graph, off_graph = item_prereq_tags(question)
        for tag in on_graph:
            if tag in prereqs:
                continue  # already counted from the graph
            try:
                m, _c, _t = kc_graph.kc_mastery(user_state, tag)
            except Exception:  # noqa: BLE001
                logger.exception("engine_features: item prereq %s unreadable", tag)
                continue
            # Item-declared prerequisites fold into the SAME feature as the
            # graph's. They are the same kind of claim — "the learner needs this
            # to do that" — and giving them a separate weight would mean fitting
            # two coefficients for one relation on data that cannot separate them.
            prereqs[tag] = float(m)
        if on_graph:
            sources["item_prereqs"] = on_graph
        if off_graph:
            # Scored neutral, recorded loudly. This list is the backlog of
            # concepts the bank depends on and the graph does not model.
            sources["off_graph_prereqs"] = off_graph

    values["prereq"] = E.centred_mastery(prereqs.values())
    sources["prereqs"] = prereqs

    return FeatureVector(values, sources)


def _last_practised_days(user_state, kc: str, now: Optional[datetime] = None) -> Optional[float]:
    """Days since the learner last attempted anything tagged to `kc`.

    Read from the KC ladder's attempt trail, which is the only per-KC timestamp
    that exists today. Once `attempt_log` is the primary record this should read
    from there instead — noted rather than done, because doing it now would make
    this module depend on a log that is not yet being written on the live path.
    """
    row = (getattr(user_state, "kc_ladder", None) or {}).get(kc) or {}
    attempts = row.get("attempts") or []
    for att in reversed(attempts):
        if isinstance(att, dict):
            days = _days_since(att.get("ts") or att.get("timestamp"), now)
            if days is not None:
                return days
    return None


# ---------------------------------------------------------------------------
# Explore-mode prerequisite prober
# ---------------------------------------------------------------------------

# Structural confidence in an edge we did not verify. 0.85 is the vault's
# standing default for LLM-authored links; our prerequisite edges are
# hand-authored in the registry, which is better, but they have never been
# validated against learner outcomes — which is the only validation that bears
# on serving. Treated as one global constant until per-edge kappa exists.
DEFAULT_KAPPA = 0.85

# Pooling strength: how many own-attempts it takes before a KC's own record
# outweighs what its neighbourhood implies. tau=5 means a KC with 5 attempts
# splits the difference.
POOL_TAU = 5.0


def pooling_weight(n_own: int, kappa: float = DEFAULT_KAPPA, tau: float = POOL_TAU) -> float:
    """James-Stein shrinkage weight, gated by structural confidence.

        lambda = kappa * tau / (tau + n_own)

    This is Layer 3 of the design in `glicko-vs-lkt-mastery-engine.md`, and it
    is used HERE ONLY to rank probes — not to modify a stored estimate. That
    restriction is the vault's hard rule for the single-learner regime: borrowed
    strength may inform a scheduling decision, it may never become a mastery
    claim.
    """
    return float(kappa) * float(tau) / (float(tau) + max(int(n_own), 0))


def probe_priority(n_own: int, kappa: float = DEFAULT_KAPPA) -> float:
    """How much is this edge worth probing?

        priority = lambda * (1 - kappa)

    Highest for edges we lean on heavily (high lambda — little own data, so the
    graph is carrying the estimate) and trust least (low kappa). This is the
    explore criterion: it targets what the model does not know about ITSELF,
    rather than what the learner is worst at.
    """
    return pooling_weight(n_own, kappa) * (1.0 - float(kappa))


def select_probe_kc(
    user_state,
    kc: str,
    *,
    exclude: Optional[Sequence[str]] = None,
    config: E.EngineConfig = E.DEFAULT_CONFIG,
) -> Optional[dict]:
    """Pick which prerequisite to probe after repeated struggle on `kc`.

    The exploit move is "serve more of what they are failing". This is the
    explore move: when a learner keeps missing at the worked rung, the useful
    question is usually not "how do I drill this harder" but "which of the
    things this rests on is actually missing" — and the answer should be chosen
    to be INFORMATIVE, not to be easy.

    Ranking combines the two signals the design calls for:

      * **prior data on the prerequisite** — how much own evidence exists, via
        the pooling weight. A parent with no record is a better probe than one
        already well measured, because measuring it can change a belief.
      * **estimated encompassing credit** — a parent that this KC's atoms
        already subsume has probably been exercised implicitly, so the graph's
        implied mastery is partly borrowed rather than observed; that inflates
        how much a direct probe would tell us.

    Returns None when there is nothing worth probing (no parents, or all of them
    already well evidenced) — in which case the caller should fall back to the
    ordinary exploit path rather than manufacturing a probe.
    """
    node = kc_graph.registry_node(kc) or {}
    parents = [p for p in (node.get("prereqs") or []) if isinstance(p, str)]
    skip = set(exclude or ())
    ranked: List[dict] = []

    for parent in parents:
        if parent in skip:
            continue
        try:
            mastery, covered, tier = kc_graph.kc_mastery(user_state, parent)
        except Exception:  # noqa: BLE001
            logger.exception("engine_features: probe read failed for %s", parent)
            continue

        n_own = _attempt_count(user_state, parent)
        lam = pooling_weight(n_own)
        priority = probe_priority(n_own)

        # An unmeasured KC (`topic-proxy`, or low crosswalk coverage) carries a
        # number that is really its topic's average wearing a per-node label.
        # Probing it is worth more, because the current estimate is the least
        # earned. `covered` is exactly the share of the estimate that rests on
        # atoms the learner actually attempted.
        borrowed = 1.0 - float(covered)
        priority *= (1.0 + borrowed)

        # Prefer parents that are plausibly the blocker. A parent already near
        # mastery is unlikely to be what is breaking the child, so probing it
        # spends a question to confirm what we already believe.
        priority *= (1.0 - float(mastery))

        ranked.append(
            {
                "kc": parent,
                "priority": priority,
                "mastery": float(mastery),
                "covered": float(covered),
                "tier": tier,
                "n_own": n_own,
                "lambda": lam,
            }
        )

    if not ranked:
        return None
    ranked.sort(key=lambda r: r["priority"], reverse=True)
    best = ranked[0]
    # A priority of zero means every parent is fully measured and fully
    # mastered — there is genuinely nothing to learn from a probe.
    return best if best["priority"] > 0 else None


def _attempt_count(user_state, kc: str) -> int:
    row = (getattr(user_state, "kc_ladder", None) or {}).get(kc) or {}
    return len(row.get("attempts") or [])


# ---------------------------------------------------------------------------
# Convenience: the whole read path in one call
# ---------------------------------------------------------------------------


def predict_for(
    user_state,
    kc: Optional[str],
    stage: str,
    posteriors: Mapping[str, E.Posterior],
    *,
    question: Optional[Mapping] = None,
    difficulty_score: Optional[float] = None,
    config: E.EngineConfig = E.DEFAULT_CONFIG,
) -> Tuple[E.Prediction, FeatureVector]:
    """Build features and predict in one step. Returns both, because the caller
    must log the feature vector alongside the prediction — a prediction without
    its inputs is not replayable."""
    values = build(
        user_state,
        kc,
        stage,
        question=question,
        difficulty_score=difficulty_score,
        config=config,
    )
    return E.predict(values, posteriors, config), values
