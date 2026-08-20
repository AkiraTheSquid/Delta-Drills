"""ALEKS/CAT-style placement diagnostic — rapid cold-start calibration.

Replaces the old "cold start = first 3 questions per subtopic" behavior with a
real adaptive placement phase, per the audited design notes in
papers/MASTERY_ESTIMATION_REFERENCE_v2.md (Q5):

  * Item selection: likelihood-near-0.5 splitting (Cosyn 2021 §1.3 — ALEKS
    picks the item that maximally splits the current state distribution).
    We operationalize it as: probe the topic AREA with the widest posterior,
    then within it the question whose predicted P(correct) is closest to 0.5.
  * Adaptive termination: posterior-precision criterion OR a question budget
    (ALEKS caps at ~30 for fatigue; our bank spans 8 areas → budget 14).
  * "I don't know" is a first-class response: strong evidence the learner sits
    below the item's difficulty, recorded without forcing a code attempt.
  * On finish, the per-area ability estimate seeds the per-atom BKT priors
    (inverse of prioritization's mastery→difficulty map), so normal practice
    starts AT the learner's level instead of staircasing up from the floor.

Model: per-area grid posterior over ability θ ∈ [0, 100] (the bank's difficulty
scale). Response model is a 1PL-with-guess/slip logistic reusing the BKT
guess/slip constants. Areas partially pool: evidence from other areas enters
each area's posterior at CROSS_AREA_WEIGHT, so a strong showing in Numpy
tightens the Einops estimate without a single Einops probe.

Every numeric constant is a v0 engineering choice (the reference doc is
explicit that ALEKS/MA publish the WHAT, not the HOW) — tune from real data.

State lives in UserPracticeState.diagnostic (a plain JSON-able dict); the
posterior is recomputed from the probe log on demand (≤ budget × 8 areas × 51
grid points — trivial), so there is no derived state to drift or migrate.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app import bkt_mastery
from app.adaptive import UserPracticeState
from app.questions import (
    get_all_questions,
    get_atoms_for_subtopic,
    get_subtopics,
    get_topic_for_subtopic,
)

# --- v0 tunables ---------------------------------------------------------------
GRID: List[float] = [float(x) for x in range(0, 101, 2)]  # ability θ support
LOGISTIC_SCALE = 10.0        # W: how fast P(correct) falls as difficulty exceeds θ
P_GUESS = bkt_mastery.P_GUESS   # reuse BKT constants for answered probes
P_SLIP = bkt_mastery.P_SLIP
DK_GUESS = 0.02              # "I don't know": near-zero lucky-pass mass
DK_SLIP = 0.05               # small chance a knower says "don't know" (fatigue)
CROSS_AREA_WEIGHT = 0.35     # partial pooling: other areas' per-probe weight…
CROSS_AREA_CAP = 3.0         # …but their TOTAL influence caps at ~3 probes'
                             # worth, so a heterogeneous profile (numpy expert,
                             # torch novice) isn't averaged flat by volume
MIN_PROBES = 6
MAX_PROBES = 14              # fatigue budget (ALEKS: 30 for a whole course)
HISTORY_WEIGHT = 0.6         # past graded practice attempts enter the posterior
                             # as discounted pseudo-probes: real evidence, but
                             # stale (decay) and gathered under practice
                             # conditions (hints, retries) rather than probes
HISTORY_PER_SUBTOPIC = 20    # most recent N per subtopic — bounds posterior
                             # cost and keeps ancient attempts from dominating
SD_STOP = 11.0               # stop early when every area's posterior SD ≤ this
INFORMATIVE_BAND = (0.25, 0.75)  # a probe is informative if P̂(correct) in band
SEED_MASTERY_FLOOR = 0.02
SEED_MASTERY_CAP = 0.92      # diagnostic may unlock (>0.85) but never "master"

# Prior over θ by self-reported level: (mean, sd) on the 0-100 scale. SDs are
# deliberately wide — self-report positions the FIRST probes, but ~3 probes of
# evidence must be able to overrule a wrong self-assessment (placement
# replaces self-report, not the other way around).
PRIOR_BY_LEVEL: Dict[Optional[str], Tuple[float, float]] = {
    "beginner": (25.0, 25.0),
    None: (40.0, 25.0),
    "strong": (62.0, 25.0),
}

# Inverse of prioritization's mastery→difficulty affine map (20 + 80·m).
_DIFF_FLOOR = 20.0
_DIFF_SPAN = 80.0


# --- state accessors -------------------------------------------------------------

def get_diag(user_state: UserPracticeState) -> dict:
    """The user's diagnostic dict (created empty on first touch)."""
    if not isinstance(getattr(user_state, "diagnostic", None), dict):
        user_state.diagnostic = {}
    d = user_state.diagnostic
    d.setdefault("active", False)
    d.setdefault("completed_at", None)
    d.setdefault("declined", False)
    d.setdefault("probes", [])
    return d


def should_run(user_state: UserPracticeState) -> bool:
    """True only after an explicit start from the Diagnostic tab.

    Placement used to auto-start for fresh accounts inside Practice. That made
    it an invisible practice mode instead of its own user-controlled flow.
    """
    return bool(get_diag(user_state)["active"])


def can_set_prior(user_state: UserPracticeState) -> bool:
    """Whether self-report can still be an honest cold-start prior.

    Once any answer or placement probe exists, changing the prior would rewrite
    the interpretation of evidence already collected. UI hides the control;
    write endpoint enforces same rule.
    """
    d = get_diag(user_state)
    if d["completed_at"] or d["probes"]:
        return False
    if getattr(user_state, "atom_mastery", None):
        return False
    if any(s.n > 0 or s.history for s in user_state.subtopic_states.values()):
        return False
    for features in getattr(user_state, "kc_posteriors", {}).values():
        if any(int(p.get("n") or 0) > 0 for p in features.values() if isinstance(p, dict)):
            return False
    return True


def start(user_state: UserPracticeState) -> dict:
    """Explicitly (re)start the diagnostic: clears the probe log, keeps BKT."""
    d = get_diag(user_state)
    d["active"] = True
    d["declined"] = False
    d["completed_at"] = None
    d["probes"] = []
    return d


def decline(user_state: UserPracticeState) -> dict:
    """Opt out — never auto-start again, no seeding."""
    d = get_diag(user_state)
    d["active"] = False
    d["declined"] = True
    return d


# --- posterior math ----------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def p_correct(theta: float, difficulty: float) -> float:
    """1PL-with-guess/slip: P(correct | θ, item difficulty)."""
    core = _sigmoid((theta - difficulty) / LOGISTIC_SCALE)
    return P_GUESS + (1.0 - P_GUESS - P_SLIP) * core


def _p_pass_dont_know(theta: float, difficulty: float) -> float:
    """Same curve under the don't-know response model (a "don't know" is a
    miss with almost no guess mass — much stronger down-evidence)."""
    core = _sigmoid((theta - difficulty) / LOGISTIC_SCALE)
    return DK_GUESS + (1.0 - DK_GUESS - DK_SLIP) * core


def _log_lik(theta: float, probe: dict) -> float:
    d = float(probe["difficulty"])
    result = probe["result"]
    if result == "dont_know":
        p = 1.0 - _p_pass_dont_know(theta, d)
    elif result == "correct":
        p = p_correct(theta, d)
    else:  # "incorrect"
        p = 1.0 - p_correct(theta, d)
    return math.log(max(p, 1e-9))


def _log_prior(theta: float, level: Optional[str]) -> float:
    mu, sd = PRIOR_BY_LEVEL.get(level, PRIOR_BY_LEVEL[None])
    z = (theta - mu) / sd
    return -0.5 * z * z


def _history_evidence(user_state: UserPracticeState) -> List[dict]:
    """Past graded practice attempts as probe-shaped evidence records.

    Existing users arrive at the diagnostic with real data — the posterior
    starts from it (at HISTORY_WEIGHT) instead of the bare self-report prior,
    so areas the learner has already practiced need few or no probes and the
    placement finishes faster. Diagnostic submits never reach subtopic
    history (questions_router routes them to record_probe instead of
    record_attempt), so these can't double-count live or past probes.

    Each record carries a recency weight "w" ∈ (0, 1]: evidence fades with the
    same half-life as BKT forgetting (bkt_mastery.HALF_LIFE_DAYS), so a stack
    of months-old wins can't resurrect knowledge the decay model has already
    written off — neither in the posterior nor in the budget credit."""
    now = datetime.now(timezone.utc)
    out = []
    for sub in user_state.subtopic_states.values():
        topic = get_topic_for_subtopic(sub.subtopic) or "Other"
        for a in sub.history[-HISTORY_PER_SUBTOPIC:]:
            w = 1.0
            try:
                age_days = (now - datetime.fromisoformat(a.timestamp)).total_seconds() / 86400.0
                if age_days > 0:
                    w = 0.5 ** (age_days / bkt_mastery.HALF_LIFE_DAYS)
            except (TypeError, ValueError):
                pass  # unparseable/missing ts: keep full weight
            if w < 0.05:
                continue  # ancient — no meaningful evidence left
            out.append({
                "topic": topic,
                "difficulty": float(a.difficulty_score),
                "result": "correct" if a.correct else "incorrect",
                "w": w,
            })
    return out


def area_posterior(user_state: UserPracticeState, area: str) -> List[float]:
    """Normalized posterior over GRID for one topic area, partially pooled:
    own-area probes at full weight, other areas' at CROSS_AREA_WEIGHT; past
    practice attempts likewise but discounted by HISTORY_WEIGHT."""
    d = get_diag(user_state)
    level = user_state.self_reported_level
    history = _history_evidence(user_state)
    n_other = sum(1 for p in d["probes"] if p.get("topic") != area)
    w_other = min(CROSS_AREA_WEIGHT, CROSS_AREA_CAP / n_other) if n_other else 0.0
    sum_w_other_hist = sum(h["w"] for h in history if h["topic"] != area)
    w_other_hist = (
        min(CROSS_AREA_WEIGHT, CROSS_AREA_CAP / sum_w_other_hist) * HISTORY_WEIGHT
        if sum_w_other_hist
        else 0.0
    )
    logs = []
    for theta in GRID:
        lp = _log_prior(theta, level)
        for probe in d["probes"]:
            w = 1.0 if probe.get("topic") == area else w_other
            lp += w * _log_lik(theta, probe)
        for h in history:
            w = (HISTORY_WEIGHT if h["topic"] == area else w_other_hist) * h["w"]
            lp += w * _log_lik(theta, h)
        logs.append(lp)
    m = max(logs)
    weights = [math.exp(v - m) for v in logs]
    total = sum(weights)
    return [w / total for w in weights]


def posterior_summary(post: List[float]) -> Tuple[float, float]:
    """(mean, sd) of a grid posterior."""
    mean = sum(p * t for p, t in zip(post, GRID))
    var = sum(p * (t - mean) ** 2 for p, t in zip(post, GRID))
    return mean, math.sqrt(max(var, 0.0))


def predicted_p_correct(post: List[float], difficulty: float) -> float:
    """Marginal predictive P(correct) for an item under a grid posterior."""
    return sum(p * p_correct(t, difficulty) for p, t in zip(post, GRID))


# --- probe selection ------------------------------------------------------------------

def _areas() -> List[str]:
    seen: Dict[str, None] = {}
    for st in get_subtopics():
        seen.setdefault(get_topic_for_subtopic(st) or "Other", None)
    return list(seen)


def _probed_ids(diag: dict) -> set:
    return {p["question_id"] for p in diag["probes"]}


def effective_budget(user_state: UserPracticeState) -> int:
    """Probe budget after crediting existing practice history.

    Each past graded attempt is worth HISTORY_WEIGHT of a probe (times its
    recency weight), so a returning learner's placement is shorter in
    proportion to the data they already have — probe selection still targets
    the widest-SD (least-known) areas first, so the remaining budget goes
    where history says least.

    The floor guarantees coverage: history piled in ONE area must not starve
    the probes needed for areas with no evidence at all (an area "has
    evidence" when its decayed history weight sums to ≥ 1 attempt's worth)."""
    history = _history_evidence(user_state)
    credit = int(HISTORY_WEIGHT * sum(h["w"] for h in history))
    w_by_area: Dict[str, float] = {}
    for h in history:
        w_by_area[h["topic"]] = w_by_area.get(h["topic"], 0.0) + h["w"]
    uncovered = sum(1 for a in _areas() if w_by_area.get(a, 0.0) < 1.0)
    return max(MIN_PROBES, uncovered, MAX_PROBES - credit)


def select_probe(user_state: UserPracticeState):
    """The next diagnostic item, ALEKS-style.

    1. Rank areas by posterior SD (widest = most to learn about the profile).
    2. In the widest area with candidates, pick the question whose predicted
       P(correct) is closest to 0.5 — the maximal split. Prefer not repeating
       the previous probe's subtopic (cheap interleaving).
    3. If NO area offers an informative item (all P̂ outside the band), return
       None — the caller finishes the diagnostic early.

    Prerequisite gating is deliberately ignored here: placement asks "what do
    you already know", it does not teach — locking probes behind unlock
    thresholds would recreate the sequential staircase this replaces.
    """
    diag = get_diag(user_state)
    probed = _probed_ids(diag)
    # Also skip anything already SERVED (probe answered or not): the frontend
    # may re-request without answering (reload, client-side question filters) —
    # re-serving the identical probe would loop it forever against a client
    # filter. Another near-0.5 item is just as informative.
    for sub in user_state.subtopic_states.values():
        probed.update(sub.served_question_ids)
    last_subtopic = diag["probes"][-1]["subtopic"] if diag["probes"] else None

    by_area: Dict[str, list] = {}
    for q in get_all_questions():
        if q.id in probed:
            continue
        by_area.setdefault(q.topic or "Other", []).append(q)

    ranked = []
    for area in _areas():
        if not by_area.get(area):
            continue
        post = area_posterior(user_state, area)
        _, sd = posterior_summary(post)
        ranked.append((sd, area, post))
    ranked.sort(key=lambda r: -r[0])

    lo, hi = INFORMATIVE_BAND
    for sd, area, post in ranked:
        best = None
        for q in by_area[area]:
            p_hat = predicted_p_correct(post, q.difficulty_score)
            if not (lo <= p_hat <= hi):
                continue
            gap = abs(p_hat - 0.5) + (0.05 if q.subtopic == last_subtopic else 0.0)
            if best is None or gap < best[0]:
                best = (gap, q)
        if best is not None:
            return best[1]
    return None


# --- recording + stopping -----------------------------------------------------------------

def record_probe(user_state: UserPracticeState, question, result: str) -> dict:
    """Log one diagnostic response ("correct" | "incorrect" | "dont_know"),
    then auto-finish if the stopping rule fires. Returns the diagnostic dict."""
    diag = get_diag(user_state)
    diag["probes"] = [p for p in diag["probes"] if p["question_id"] != question.id]
    diag["probes"].append({
        "question_id": question.id,
        "subtopic": question.subtopic,
        "topic": question.topic or "Other",
        "difficulty": question.difficulty_score,
        "result": result,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    if _should_stop(user_state):
        finish(user_state)
    return diag


def override_probe(user_state: UserPracticeState, question_id: int, correct: bool) -> bool:
    """Flip the most recent probe's result (the /override path during placement)."""
    diag = get_diag(user_state)
    if not diag["probes"] or diag["probes"][-1]["question_id"] != question_id:
        return False
    diag["probes"][-1]["result"] = "correct" if correct else "incorrect"
    return True


def effective_min_probes(user_state: UserPracticeState) -> int:
    """MIN_PROBES guards against a fluky early stop from the bare prior; a
    learner with real practice history already has evidence in the posterior,
    so don't force filler probes on them — 2 live ones suffice. Requires at
    least one fresh attempt's worth of decayed evidence."""
    total_w = sum(h["w"] for h in _history_evidence(user_state))
    return 2 if total_w >= 1.0 else MIN_PROBES


def should_finish(user_state: UserPracticeState) -> bool:
    """Public stop-check for serving paths. The budget can shrink between
    requests (history credit landing in a deploy mid-diagnostic), so callers
    must re-check BEFORE selecting a probe, not only after recording one."""
    return _should_stop(user_state)


def _should_stop(user_state: UserPracticeState) -> bool:
    diag = get_diag(user_state)
    n = len(diag["probes"])
    if n >= effective_budget(user_state):
        return True
    if n < effective_min_probes(user_state):
        return False
    for area in _areas():
        _, sd = posterior_summary(area_posterior(user_state, area))
        if sd > SD_STOP:
            return False
    return True


# --- finish: seed BKT ----------------------------------------------------------------------

def _compute_area_estimates(user_state: UserPracticeState) -> List[dict]:
    """Fresh per-area (θ̂, sd, probes) from the current posterior. finish()
    must use THIS (not area_estimates) so an /override of the finishing probe
    re-seeds from the corrected posterior, not the frozen snapshot."""
    diag = get_diag(user_state)
    out = []
    for area in _areas():
        mean, sd = posterior_summary(area_posterior(user_state, area))
        out.append({
            "topic": area,
            "theta": round(mean, 1),
            "sd": round(sd, 1),
            "probes": sum(1 for p in diag["probes"] if p["topic"] == area),
        })
    return out


def area_estimates(user_state: UserPracticeState) -> List[dict]:
    """Per-area (θ̂, sd, probes) snapshot for the UI / finish summary.

    A COMPLETED diagnostic returns the estimates frozen at finish() — later
    practice history flows into area_posterior, and recomputing would make a
    finished placement's reported result drift retroactively."""
    diag = get_diag(user_state)
    if diag["completed_at"] and diag.get("estimates"):
        return diag["estimates"]
    return _compute_area_estimates(user_state)


def _mastery_from_theta(theta: float) -> float:
    m = (theta - _DIFF_FLOOR) / _DIFF_SPAN
    return max(SEED_MASTERY_FLOOR, min(SEED_MASTERY_CAP, m))


def finish(user_state: UserPracticeState) -> dict:
    """Close the diagnostic and seed per-atom BKT mastery from the area
    estimates (inverse of the mastery→target-difficulty map), so the very next
    normal question lands at the learner's level and everything below it is
    unlocked. Atoms with practice evidence only ever RAISE; atoms whose value
    is still just a prior (self-report / default — no entry in atom_mastery)
    take the diagnostic estimate outright, so a self-reported "strong" who
    bombs placement is corrected downward too."""
    diag = get_diag(user_state)
    now = datetime.now(timezone.utc)
    estimates = _compute_area_estimates(user_state)
    theta_by_area = {e["topic"]: e["theta"] for e in estimates}

    params = bkt_mastery.params_for_level(user_state.self_reported_level)
    seeded: Dict[str, float] = {}
    for subtopic in get_subtopics():
        theta = theta_by_area.get(get_topic_for_subtopic(subtopic) or "Other")
        if theta is None:
            continue
        m = _mastery_from_theta(theta)
        for atom in get_atoms_for_subtopic(subtopic):
            if atom not in user_state.atom_mastery:
                # prior-only atom: placement estimate replaces the prior
                if atom not in seeded or m > seeded[atom]:
                    seeded[atom] = m
                continue
            current = bkt_mastery.current_mastery(
                user_state.atom_mastery, user_state.atom_last_ts, atom, now, params
            )
            if m > current and m > seeded.get(atom, 0.0):
                seeded[atom] = m
    ts = now.isoformat()
    for atom, m in seeded.items():
        user_state.atom_mastery[atom] = m
        user_state.atom_last_ts[atom] = ts

    diag["active"] = False
    diag["completed_at"] = ts
    diag["estimates"] = estimates
    diag["atoms_seeded"] = len(seeded)
    return diag
