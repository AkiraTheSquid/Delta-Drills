"""Global content-health stats — `/api/practice/kc-stats`.

The read side of `app/kc_stats.py`: per-KC aggregates over EVERY learner's
attempt log, plus the flags list (rung stalls, served-while-predicting-failure
runs). Global on purpose — this is a "which concept is broken" surface for the
Metadata tab and for audits, not a learner dashboard; per-learner reads stay on
`/kc-lattice`.

Auth is the ordinary signed-in dependency, the same trust level as the rest of
`/api/practice`. Reviewed and kept deliberately (Seth, 2026-09-01): the global
Metadata tab is FOR learners, and with a dozen learners any aggregate is nearly
individual anyway — restricting it would just hide the surface he asked for. The payload carries counts and rates only, never user ids —
`kc_stats` strips them at aggregation, so this router has nothing to leak.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import kc_stats
from app.auth import get_current_user
from app.models import User

router = APIRouter()


@router.get("/kc-stats")
def global_kc_stats(user: User = Depends(get_current_user)) -> dict:
    """Per-KC global aggregates + flags. See `kc_stats.kc_stats` for the shape."""
    return kc_stats.kc_stats()
