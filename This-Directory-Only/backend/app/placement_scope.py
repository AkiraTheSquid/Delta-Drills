"""What a placement covers and how long it may run — the learner's two choices.

Seth, 2026-09-24: the placement page becomes a short sequence of questions.
First "which areas do you want to improve?" — a learner who already knows they
do not want linear algebra (or NumPy, or Python) says so and those concepts
leave the test. Then "how long do you want to spend?", and the three lengths
offered are cut from how long THAT test needs to settle: a PyTorch + Einops
placement calibrates in far less time than the whole curriculum, so the
options shrink with the focus instead of always reading 1h / 3h / 6h.

  * Areas are the registry's own `topic` field (Python, Numpy, PyTorch,
    Einops, Mathematics) — no second taxonomy to drift from the graph.
  * A run's areas live in `diagnostic["areas"]` (None = everything);
    `diagnostic.assessed_kcs` reads them through `kcs_for`.
  * Budgets are minutes on a 15-minute grid, 15 min .. 6 h. The old `hours`
    field (PLAN_HOURS in diagnostic.py) is still accepted from old clients.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app import kc_graph, kc_prefs, practice_targets

MIN_BUDGET_SECS = 15 * 60
MAX_BUDGET_SECS = 6 * 3600
BUDGET_STEP_SECS = 15 * 60
# (key, fraction of the settle time). "full" is how long the estimator needs to
# finish on evidence rather than on the clock, capped at MAX_BUDGET_SECS.
PLAN_TIERS: Tuple[Tuple[str, float], ...] = (("quick", 0.25), ("standard", 0.6), ("full", 1.0))


# --- areas -----------------------------------------------------------------------------


def kc_area(kc: str) -> str:
    return str((kc_graph._registry().get(kc) or {}).get("topic") or "Other")


def area_catalog(user_state) -> List[dict]:
    """[{key, kcs}] for every area holding a concept the learner has not
    switched off, in registry order."""
    counts: Dict[str, int] = {}
    for k in kc_graph._registry():
        if not kc_prefs.is_disabled(user_state, k):
            a = kc_area(k)
            counts[a] = counts.get(a, 0) + 1
    return [{"key": a, "kcs": n} for a, n in counts.items()]


def normalize_areas(areas, user_state=None) -> Optional[List[str]]:
    """The learner's focus as known area keys, or None for "every area".
    Unknown names are dropped; choosing nothing (or everything) is the whole
    curriculum — never a test with zero concepts in it. With `user_state`,
    "known" is `area_catalog`'s areas, so picking every area the learner is
    offered still reads as everything when a whole course (Delta Drills,
    course_registry) is switched off."""
    if not areas:
        return None
    if isinstance(areas, str):
        areas = areas.split(",")
    known = {kc_area(k) for k in kc_graph._registry()
             if user_state is None or not kc_prefs.is_disabled(user_state, k)}
    keep = [a for a in dict.fromkeys(str(x).strip() for x in areas) if a in known]
    if not keep or set(keep) == known:
        return None
    return keep


def kcs_for(user_state, scope: str, areas: Optional[List[str]]) -> List[str]:
    """Enabled registry concepts inside `scope` (practice_targets) and `areas`."""
    scope_set = practice_targets.scope_kcs(scope)
    area_set = set(areas) if areas else None
    return [k for k in kc_graph._registry() if not kc_prefs.is_disabled(user_state, k)
            and (scope_set is None or k in scope_set)
            and (area_set is None or kc_area(k) in area_set)]


# --- length ------------------------------------------------------------------------------


def normalize_minutes(minutes) -> Optional[int]:
    """A client-chosen budget in minutes, snapped to the 15-minute grid and
    clamped to [15 min, 6 h]. None when absent or unreadable."""
    try:
        m = float(minutes)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(m) or m <= 0:
        return None
    secs = round(m * 60 / BUDGET_STEP_SECS) * BUDGET_STEP_SECS
    return int(min(MAX_BUDGET_SECS, max(MIN_BUDGET_SECS, secs)) // 60)


def _tier_budgets(settle_secs: float) -> List[int]:
    """Three strictly increasing budgets on the 15-minute grid, top-down from
    the settle time. A scope too small to hold three distinct lengths loses
    its lower tiers rather than offering the same length twice."""
    full = int(math.ceil(settle_secs / BUDGET_STEP_SECS)) * BUDGET_STEP_SECS
    full = min(MAX_BUDGET_SECS, max(MIN_BUDGET_SECS, full))
    budgets: List[int] = []
    for _, frac in reversed(PLAN_TIERS):
        b = max(MIN_BUDGET_SECS, int(round(full * frac / BUDGET_STEP_SECS)) * BUDGET_STEP_SECS)
        if budgets:
            b = min(b, budgets[-1] - BUDGET_STEP_SECS)
        budgets.append(b)
    return list(reversed(budgets))


def plan_options(user_state, areas=None) -> dict:
    """The length picker for a test over `areas` (None = everything): per tier,
    how many problems it is likely to hold, how long it will probably really
    take, and what fraction of the settle-everything test it covers. Point
    estimates from the difficulty heuristic — the concept clock is the CAP,
    not the expectation.

    Always the whole-curriculum scope: a finished ray-tracing check must not
    shrink the next full placement's picker."""
    from app import diagnostic  # diagnostic imports this module

    kcs = kcs_for(user_state, "all", normalize_areas(areas, user_state))
    mean_secs = diagnostic._mean_est_secs(kcs)
    # Roughly two direct probes settle a concept (diagnostic.STOP_GAIN).
    probes_to_settle = max(1, 2 * len(kcs))
    caps = [diagnostic.kc_cap_secs(k) for k in kcs] or [diagnostic.PER_PROBLEM_SECS]
    options = []
    for (key, _), budget in zip(PLAN_TIERS, _tier_budgets(probes_to_settle * mean_secs)):
        if budget < MIN_BUDGET_SECS:
            continue
        fits = int(budget // mean_secs) if mean_secs > 0 else probes_to_settle
        probes = max(1, min(fits, probes_to_settle))
        options.append({
            "key": key,
            "hours": round(budget / 3600.0, 2),
            "minutes": budget // 60,
            "budget_secs": budget,
            "per_problem_secs": diagnostic.PER_PROBLEM_SECS,
            "est_probes": probes,
            "est_minutes": int(round(min(budget, probes * mean_secs) / 60.0)),
            "min_probes_at_cap": max(1, budget // diagnostic.PER_PROBLEM_SECS),
            "coverage": round(min(1.0, probes / probes_to_settle), 3),
        })
    return {
        "options": options,
        "kcs": kcs,
        "probes_to_settle": probes_to_settle,
        "per_problem_min_secs": min(caps),
        "per_problem_max_secs": max(caps),
    }
