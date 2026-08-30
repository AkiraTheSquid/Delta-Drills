"""bkt_mastery.py — per-atom Bayesian Knowledge Tracing with encompassing FIRe.

THE mastery + prioritization signal for Delta Drills. It replaces per-subtopic
EWMA *as the algorithm driver*; the EWMA machinery in adaptive.py, if kept at
all, is now only a learner-facing "area score" readout (and that readout is
better derived from `area_scores()` below — single source of truth).

Model
-----
Each atom (concept-graph node) carries a BKT posterior  L = P(skill known):

  * observe()  — Bayesian evidence update from a graded attempt (guess/slip),
                 followed by the learn-transit step (Corbett & Anderson 1994).
  * FIRe       — practicing (and clearing) an ADVANCED atom credits the SIMPLER
                 atoms it encompasses — Math Academy's Fractional Implicit
                 Repetitions. A single-hop implicit-transit of magnitude
                 pT * propagation_weight along each encompassing edge of the v3
                 graph. **Encompassing edges are the ONLY channel by which
                 evidence is shared between atoms** — vanilla BKT treats every
                 skill independently, so this is the necessary completion, not a
                 redundancy.
  * decay()    — vanilla BKT never forgets; the 2026-05-24 audit (Yudelson &
                 Pavlik 2013) flags monotonic mastery as an anti-pattern, so we
                 keep the production half-life regression toward L0 (p_init).

Calibration: every numeric constant here is a v0 default, NOT literature-derived
(mirrors the learner_sim.py model validated 2026-05-28). Re-fit once real
per-atom attempt data accumulates. See papers/MASTERY_ESTIMATION_REFERENCE_v2.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- v0 BKT parameters (NOT literature-derived) ------------------------------
P_INIT = 0.10            # L0: prior P(known) before any evidence
P_TRANSIT = 0.30         # T: P(unlearned -> learned) per practice
P_GUESS = 0.20           # G: P(correct | not known)
P_SLIP = 0.10            # S: P(incorrect | known)
MASTERY_THRESHOLD = 0.95  # belief at/above which an atom counts as "mastered"
UNLOCK_THRESHOLD = 0.85   # belief at/above which an atom is "cleared" for gating

# Forgetting: half-life regression of L toward p_init (mirrors adaptive.py).
HALF_LIFE_DAYS = 14.0

GRAPH_PATH = (
    Path(__file__).resolve().parent
    / "data" / "concept_graphs" / "arena_drillable_v1.json"
)


@dataclass(frozen=True)
class BKTParams:
    p_init: float = P_INIT
    p_transit: float = P_TRANSIT
    p_guess: float = P_GUESS
    p_slip: float = P_SLIP


DEFAULT_PARAMS = BKTParams()

# Self-reported experience level → prior P(known) for never-practiced atoms.
# The prior positions the difficulty search (target_difficulty, subtopic
# ordering, and the starting point of the first BKT update); it is NOT
# evidence — mastery/unlock gates (is_mastered, item_is_unlocked) keep
# DEFAULT_PARAMS so nothing unlocks from self-report alone. Both priors sit
# far below UNLOCK_THRESHOLD by construction. v0 values, calibrate like the
# rest of this module.
PRIOR_BY_LEVEL: Dict[str, float] = {
    "beginner": 0.02,   # first questions land at the difficulty floor
    "strong": 0.45,     # first questions land around difficulty ~56/100
}


def params_for_level(level: Optional[str]) -> BKTParams:
    """BKTParams with p_init seeded from a self-reported level (None/unknown
    level → DEFAULT_PARAMS)."""
    prior = PRIOR_BY_LEVEL.get(level or "")
    if prior is None:
        return DEFAULT_PARAMS
    return BKTParams(p_init=prior, p_transit=P_TRANSIT, p_guess=P_GUESS, p_slip=P_SLIP)


# --- encompassing graph index ------------------------------------------------

@lru_cache(maxsize=1)
def _encompassing_index(graph_path: str = str(GRAPH_PATH)) -> Dict[str, List[Tuple[str, float]]]:
    """advanced_atom -> [(simpler_atom, propagation_weight), ...], single hop.

    Built from the v3 graph's encompassing edges. Edge orientation:
    prerequisite_id = simpler, dependent_id = advanced (mastering the advanced
    atom implies the simpler one). Cached; the graph is immutable at runtime.
    """
    g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    idx: Dict[str, List[Tuple[str, float]]] = {}
    for e in g.get("prerequisite_edges", []):
        if not e.get("is_encompassing"):
            continue
        w = float(e.get("propagation_weight", 0.0))
        if w <= 0.0:
            continue
        adv = e["dependent_id"]       # advanced atom (the one being practiced)
        simpler = e["prerequisite_id"]  # simpler atom that receives FIRe credit
        idx.setdefault(adv, []).append((simpler, w))
    return idx


def encompassed_by(atom_id: str) -> List[Tuple[str, float]]:
    """Simpler atoms credited when `atom_id` (advanced) is practiced + cleared."""
    return _encompassing_index().get(atom_id, [])


# --- prerequisite index (UNLOCK gating) --------------------------------------

# Atoms that are infrastructure / tooling / dataset / GPU / visualization noise,
# NOT real conceptual prerequisites. They have no exercises and must never gate
# learning. Skipped when computing an atom's gating prerequisites. (Demoted
# 2026-05-30 from the untrainable-prereq audit.)
NON_GATING_ATOMS = frozenset({
    "device-to-cuda", "cuda-availability-check", "plotly-imshow", "loss-landscape",
    "tqdm-progress-bar", "mnist-dataset", "imagenet-1k-labels", "wandb-watch-gradients",
    "imagenet-normalization", "torchvision-models-pretrained", "trainer-class-pattern",
    "cifar10-dataset",
    # 2026-07-08 orphan-atom fix: untrainable prereqs (zero bank questions) that
    # permanently locked their dependents; no trainable twin exists to rewrite to.
    "tensor-item-scalar", "singular-matrix-mask-trick", "unbroadcast-pattern",
    "validation-no-grad",
    # 2026-08-30, same failure and same remedy after the ARENA content cut. Every
    # question training these two went with the concepts that taught them, and a
    # gating prereq with no trainer locks its dependents forever: as-strided
    # windowing was the whole of numpy.sliding-windows / window-stencil (Tensor.
    # unfold appears in ZERO of the 458 ARENA notebooks — einops does this job in
    # the corpus), and inf-masking was carried only by retired triangle drills.
    # The two atoms whose concepts SURVIVED the cut are not here: q103 and q198
    # were kept in the bank instead, which is the better fix where it is available.
    "as-strided-windowing", "inf-masking",
})


@lru_cache(maxsize=1)
def _prereq_index(graph_path: str = str(GRAPH_PATH)) -> Dict[str, frozenset]:
    """dependent_atom -> frozenset(prerequisite_atoms), single hop, from the v3
    graph's prerequisite_edges. NON_GATING_ATOMS are stripped so they never gate.
    Cached; the graph is immutable at runtime."""
    g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    idx: Dict[str, set] = {}
    for e in g.get("prerequisite_edges", []):
        dep = e["dependent_id"]
        pre = e["prerequisite_id"]
        if pre in NON_GATING_ATOMS:
            continue
        idx.setdefault(dep, set()).add(pre)
    return {k: frozenset(v) for k, v in idx.items()}


def prerequisites(atom_id: str) -> frozenset:
    """The gating prerequisite atoms of `atom_id` (NON_GATING already removed).
    Empty for root atoms / unknown ids."""
    return _prereq_index().get(atom_id, frozenset())


def atom_is_ready(
    atom_id: str,
    mastery: Dict[str, float],
    last_ts: Dict[str, str],
    now: Optional[datetime] = None,
    threshold: float = UNLOCK_THRESHOLD,
    params: BKTParams = DEFAULT_PARAMS,
) -> bool:
    """READY-TO-LEARN gate for a teaching item targeting `atom_id`: every gating
    prerequisite of the atom is mastered to >= threshold. Root atoms (no gating
    prereqs) are always ready — they are the entry points. This is the unified
    per-atom prerequisite gate; the atom ITSELF is not required (non-circular)."""
    return all(
        current_mastery(mastery, last_ts, p, now, params) >= threshold
        for p in prerequisites(atom_id)
    )


@lru_cache(maxsize=1)
def _all_atoms(graph_path: str = str(GRAPH_PATH)) -> Tuple[str, ...]:
    g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    return tuple(c["id"] for c in g.get("concepts", []))


def gate_sets(
    mastery: Dict[str, float],
    last_ts: Dict[str, str],
    now: Optional[datetime] = None,
    threshold: float = UNLOCK_THRESHOLD,
    params: BKTParams = DEFAULT_PARAMS,
) -> Tuple[List[str], List[str]]:
    """(ready_atoms, mastered_atoms) over every graph atom, for the unified gate.

    - mastered: decayed posterior >= threshold (the atom itself is learned).
    - ready:    all gating prerequisites of the atom are mastered (ready to LEARN
                it) — root atoms are always ready.
    A single-atom teaching item (bank Q, single drill) unlocks iff its atom is in
    `ready`; a composite/ARENA item unlocks iff all its component atoms are in
    `mastered`."""
    ready: List[str] = []
    mastered: List[str] = []
    for a in _all_atoms():
        if current_mastery(mastery, last_ts, a, now, params) >= threshold:
            mastered.append(a)
        if atom_is_ready(a, mastery, last_ts, now, threshold, params):
            ready.append(a)
    return ready, mastered


# --- core BKT ----------------------------------------------------------------

def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def observe(prior: float, correct: bool, params: BKTParams = DEFAULT_PARAMS) -> float:
    """One graded attempt → posterior + learn-transit. Returns L'.

    Transit fires ONLY on a correct attempt (same rule FIRe already uses in
    apply_attempt). Vanilla BKT adds T unconditionally — "attempting teaches
    even when you fail" — which is perverse for difficulty targeting: from a
    low prior, a WRONG answer RAISED mastery (0.10 → posterior 0.014 → +0.30
    transit → 0.31) and the next question got HARDER (target 28 → 43.9,
    caught live 2026-07-05). Gating transit on correctness makes the target
    behave like a noise-robust staircase: wrong → down, right → up."""
    L = _clamp(prior)
    g, s, t = params.p_guess, params.p_slip, params.p_transit
    if correct:
        num = L * (1 - s)
        den = num + (1 - L) * g
    else:
        num = L * s
        den = num + (1 - L) * (1 - g)
    post = num / den if den > 1e-12 else L
    if not correct:
        return post
    return post + (1 - post) * t


def implicit_transit(prior: float, gain: float) -> float:
    """A learning step of size `gain` with no direct observation (a FIRe rep)."""
    L = _clamp(prior)
    return L + (1 - L) * _clamp(gain)


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def decay(
    L: float,
    last_ts: Optional[str],
    now: Optional[datetime] = None,
    params: BKTParams = DEFAULT_PARAMS,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """Regress L toward p_init by elapsed-time half-life (forgetting)."""
    now = now or datetime.now(timezone.utc)
    prev = _parse_ts(last_ts)
    if prev is None or half_life_days <= 0:
        return L
    elapsed = max(0.0, (now - prev).total_seconds() / 86400.0)
    factor = 0.5 ** (elapsed / half_life_days)
    return params.p_init + (L - params.p_init) * factor


# --- entry point -------------------------------------------------------------

def apply_attempt(
    mastery: Dict[str, float],
    last_ts: Dict[str, str],
    atom_id: str,
    correct: bool,
    now: Optional[datetime] = None,
    params: BKTParams = DEFAULT_PARAMS,
    confidence: float = 1.0,
) -> Dict[str, float]:
    """Update the directly-practiced atom, then FIRe-credit the atoms it
    encompasses. Mutates `mastery`/`last_ts` in place; returns {atom: new_L}
    for every atom changed (for logging / frontend sync).

    Each touched atom is decayed to `now` first, so updates blend against a
    forgetting-adjusted prior (matches adaptive.py's decay-before-update order).

    `confidence` ∈ [0,1] is how sure we are this attempt exercises `atom_id`
    (a question→atom tag may be uncertain; an authored drill is ~1.0). It
    softly scales the update: the belief moves only `confidence`× of the way
    toward the full BKT posterior, and the FIRe implicit-rep magnitude scales
    by it too. confidence=1.0 → standard update; confidence=0.0 → no-op.

    FIRe credit fires ONLY on a correct attempt: a failed advanced attempt did
    not demonstrably exercise the simpler skill, so it should not inflate it.
    (This is a deliberate, more-conservative choice than learner_sim.py, which
    propagated unconditionally; the robustness finding survives either way.)
    """
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat()
    c = _clamp(confidence)
    changed: Dict[str, float] = {}
    if c <= 0.0:
        return changed

    prior = decay(mastery.get(atom_id, params.p_init), last_ts.get(atom_id), now, params)
    full = observe(prior, correct, params)
    new = prior + (full - prior) * c       # soft-apply evidence by confidence
    mastery[atom_id] = new
    last_ts[atom_id] = ts
    changed[atom_id] = new

    if correct:
        gain = params.p_transit * c        # implicit-rep magnitude, conf-scaled
        for simpler, w in encompassed_by(atom_id):
            b_prior = decay(mastery.get(simpler, params.p_init), last_ts.get(simpler), now, params)
            b_new = implicit_transit(b_prior, gain * w)
            mastery[simpler] = b_new
            last_ts[simpler] = ts
            changed[simpler] = b_new

    return changed


def current_mastery(
    mastery: Dict[str, float],
    last_ts: Dict[str, str],
    atom_id: str,
    now: Optional[datetime] = None,
    params: BKTParams = DEFAULT_PARAMS,
) -> float:
    """Decay-adjusted P(known) for an atom right now (read without mutating).

    Atoms never practiced (and never FIRe-credited) sit at p_init.
    """
    return decay(mastery.get(atom_id, params.p_init), last_ts.get(atom_id), now, params)


def is_mastered(
    mastery: Dict[str, float],
    last_ts: Dict[str, str],
    atom_id: str,
    now: Optional[datetime] = None,
    params: BKTParams = DEFAULT_PARAMS,
) -> bool:
    return current_mastery(mastery, last_ts, atom_id, now, params) >= MASTERY_THRESHOLD


def item_is_unlocked(
    required_atom_ids: List[str],
    mastery: Dict[str, float],
    last_ts: Dict[str, str],
    now: Optional[datetime] = None,
    threshold: float = UNLOCK_THRESHOLD,
    params: BKTParams = DEFAULT_PARAMS,
) -> bool:
    """Canonical UNIFIED unlock gate for any practice item — a bank question, a
    Colab drill, or an ARENA curriculum exercise. An item is unlocked iff EVERY
    atom it requires is mastered to >= `threshold` (decay-adjusted).

    `required_atom_ids` is the set of atoms the item is tagged with: for a
    single-atom drill that is its one atom; for a composite/ARENA exercise it is
    every component atom it composes. This is the single rule that replaces the
    old per-surface ad-hoc gates (drills' flat 0.50/0.70 on own atoms, ARENA's
    hand-authored subtopic prereq list). An item with NO required atoms is
    treated as unlocked (nothing to gate on).
    """
    return all(
        current_mastery(mastery, last_ts, a, now, params) >= threshold
        for a in required_atom_ids
    )


# --- learner-facing readout (NOT a driver) -----------------------------------

@lru_cache(maxsize=1)
def _atom_topic(graph_path: str = str(GRAPH_PATH)) -> Dict[str, str]:
    g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    return {c["id"]: c.get("topic") or "" for c in g.get("concepts", [])}


def area_scores(
    mastery: Dict[str, float],
    last_ts: Dict[str, str],
    now: Optional[datetime] = None,
    params: BKTParams = DEFAULT_PARAMS,
) -> Dict[str, float]:
    """Mean decay-adjusted mastery per graph topic — a derived view for the
    learner's "area score". Replaces the parallel per-subtopic EWMA readout so
    there is a single source of truth. Areas with no practiced atoms are omitted.
    """
    by_topic: Dict[str, List[float]] = {}
    topic_of = _atom_topic()
    for atom_id in mastery:
        topic = topic_of.get(atom_id) or "Other"
        by_topic.setdefault(topic, []).append(
            current_mastery(mastery, last_ts, atom_id, now, params)
        )
    return {t: sum(v) / len(v) for t, v in by_topic.items() if v}
