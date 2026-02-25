"""
TEMPORARY: Staleness-review scheduler.
See REMOVE_THIS.md for how to delete this feature cleanly.

Forcing 3 consecutive questions on any subtopic not studied for 3+ days
gives the EWMA learning rate a chance to update with current performance,
catching forgetting (score drops) or retention (score holds).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.adaptive import UserPracticeState
from app.prioritization import COLD_START_MIN_QUESTIONS

# --- Tuneable constants -------------------------------------------------------

STALENESS_THRESHOLD_DAYS: float = 3.0
STALENESS_REVIEW_COUNT: int = 3

# --- In-memory review tracker -------------------------------------------------
# Keyed by user_id.  Value is (subtopic, target_n) where target_n is the
# sub_state.n value the user must reach before the review is considered done.
# Stored in memory only — resets on server restart (acceptable for a temp
# feature; the next request will simply re-detect staleness if still applicable).

_active_reviews: Dict[str, Tuple[str, int]] = {}


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _days_since_last_study(user_state: UserPracticeState, subtopic: str) -> Optional[float]:
    """
    Return the number of days since the subtopic was last studied, or None
    if it has no recorded history.
    """
    sub_state = user_state.subtopic_states.get(subtopic)
    if not sub_state or not sub_state.history:
        return None
    last_ts = sub_state.history[-1].timestamp
    try:
        last_dt = datetime.fromisoformat(last_ts)
        now = datetime.now(timezone.utc)
        return (now - last_dt).total_seconds() / 86400.0
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_staleness_override(
    user_state: UserPracticeState,
    available_subtopics: List[str],
) -> Optional[str]:
    """
    Return a subtopic that must be practiced next due to staleness, or None
    to let normal priority selection proceed.

    Call this at the top of select_next_subtopic, passing only subtopics that
    still have unserved questions (i.e. the already-filtered available list).
    """
    uid = user_state.user_id

    # --- Continue an active staleness review ----------------------------------
    if uid in _active_reviews:
        subtopic, target_n = _active_reviews[uid]
        sub_state = user_state.subtopic_states.get(subtopic)
        current_n = sub_state.n if sub_state else 0

        if current_n < target_n and subtopic in available_subtopics:
            return subtopic

        # Review complete (or subtopic ran out of questions) — clear it.
        del _active_reviews[uid]

    # --- Scan for newly stale subtopics ---------------------------------------
    # Only consider topics past cold start; cold-start topics are already
    # boosted to highest priority by the cold-start fix in prioritization.py.
    stale: List[Tuple[float, str]] = []
    for st in available_subtopics:
        sub_state = user_state.subtopic_states.get(st)
        if sub_state is None or sub_state.n < COLD_START_MIN_QUESTIONS:
            continue
        days_ago = _days_since_last_study(user_state, st)
        if days_ago is not None and days_ago >= STALENESS_THRESHOLD_DAYS:
            stale.append((days_ago, st))

    if not stale:
        return None

    # Pick the most stale subtopic to review first.
    stale.sort(reverse=True)
    _, chosen = stale[0]

    sub_state = user_state.subtopic_states.get(chosen)
    current_n = sub_state.n if sub_state else 0
    _active_reviews[uid] = (chosen, current_n + STALENESS_REVIEW_COUNT)

    return chosen
