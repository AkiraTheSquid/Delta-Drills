"""Per-learner concept preferences: turn a KC off, or change how eagerly the
tutor serves it.

Seth, 2026-09-06: "whenever they click on the node … on the right, the tab for
settings for that node, you can on the one hand disable the node entirely, or
decrease its representativeness according to the algorithm so it appears less
often … 0.75 makes it 25% less likely to show up, 1.5 makes it 50% more likely
… That way I can manually prioritize certain concepts I want to practice."

WHAT A PREFERENCE IS. Learner state, not curriculum: it lives in
`UserPracticeState.kc_prefs` (persisted per user) and never touches the
registry, the lattice, or anybody else's queue. It is read in exactly three
places, all through this module so the rule is stated once:

  * `kc_graph.frontier` — the frontier is ORDERED by coreness (descendant
    count) then depth. The weight scales the coreness term, so 1.5 on a node
    makes it sort as if it had 50% more dependents and 0.75 as if it had 25%
    fewer. A disabled node leaves the frontier altogether.
  * `kc_graph.kc_is_unlocked` — a disabled prerequisite is treated as
    SATISFIED. Turning a node off means "skip it", and skipping must not lock
    everything downstream of it out of reach.
  * `prioritization.question_is_unlocked` — a drill whose target KC is
    disabled is not servable, so the weakest-first fallback cannot hand it
    back after the frontier has skipped it.

Weights are clamped to [MIN_WEIGHT, MAX_WEIGHT] and NEVER zero: "off" is the
`enabled` flag alone, so the wire row can never say enabled while the queue
treats the concept as disabled. 1.0 is neutral and is stored as an absent entry
so an untouched learner's file carries nothing.
"""

from __future__ import annotations

from typing import Dict, Optional

DEFAULT_WEIGHT = 1.0
MIN_WEIGHT = 0.25
MAX_WEIGHT = 4.0
_EPS = 1e-6


def clamp_weight(value) -> float:
    try:
        w = float(value)
    except (TypeError, ValueError):
        return DEFAULT_WEIGHT
    if w != w or w in (float("inf"), float("-inf")):
        return DEFAULT_WEIGHT
    return max(MIN_WEIGHT, min(MAX_WEIGHT, w))


def _prefs(user_state) -> Dict[str, dict]:
    prefs = getattr(user_state, "kc_prefs", None)
    return prefs if isinstance(prefs, dict) else {}


def weight_for(user_state, kc: str) -> float:
    """The learner's priority multiplier for `kc`; 1.0 when unset, 0.0 when
    the concept is disabled."""
    row = _prefs(user_state).get(kc)
    if not isinstance(row, dict):
        return DEFAULT_WEIGHT
    if row.get("enabled") is False:
        return 0.0
    return clamp_weight(row.get("weight", DEFAULT_WEIGHT))


def is_disabled(user_state, kc: str) -> bool:
    row = _prefs(user_state).get(kc)
    return isinstance(row, dict) and row.get("enabled") is False


def set_pref(user_state, kc: str, *, enabled: Optional[bool] = None,
             weight: Optional[float] = None) -> dict:
    """Write one concept's preference and return the stored row.

    A row that comes back to the neutral state (enabled, weight 1.0) is REMOVED
    so the persisted file only lists concepts the learner actually changed.
    """
    if not isinstance(getattr(user_state, "kc_prefs", None), dict):
        user_state.kc_prefs = {}
    row = dict(user_state.kc_prefs.get(kc) or {})
    if enabled is not None:
        row["enabled"] = bool(enabled)
    if weight is not None:
        row["weight"] = clamp_weight(weight)
    row.setdefault("enabled", True)
    row.setdefault("weight", DEFAULT_WEIGHT)
    if row["enabled"] and abs(row["weight"] - DEFAULT_WEIGHT) < _EPS:
        user_state.kc_prefs.pop(kc, None)
        return {"enabled": True, "weight": DEFAULT_WEIGHT}
    user_state.kc_prefs[kc] = row
    return dict(row)


def pref_row(user_state, kc: str) -> dict:
    """The wire shape for one concept — always both fields, defaults filled."""
    row = _prefs(user_state).get(kc) or {}
    return {
        "enabled": row.get("enabled", True) is not False,
        "weight": clamp_weight(row.get("weight", DEFAULT_WEIGHT)),
    }
