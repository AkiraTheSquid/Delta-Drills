"""Per-concept learner preferences — the graph's Settings tab.

Endpoints (mounted under /api/practice by the parent router):
  GET /kc-prefs            every concept the learner has changed
  PUT /kc-prefs/{kc}       {"enabled": bool?, "weight": float?} → stored row

The rule for what a preference DOES lives in `app/kc_prefs.py`; this file only
moves it on and off the wire. `/kc-lattice` carries the same row per KC under
`pref`, so the graph can draw it without a second round trip.
"""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app import kc_graph, kc_prefs
from app.adaptive import get_user_state, save_user_state
from app.auth import get_current_user
from app.models import User

router = APIRouter()


class KcPrefUpdate(BaseModel):
    enabled: Optional[bool] = None
    # Never 0: "off" is `enabled`, so a row can't say enabled while the queue
    # skips it. Out-of-range is a 422 rather than a silent clamp.
    weight: Optional[float] = Field(
        default=None, ge=kc_prefs.MIN_WEIGHT, le=kc_prefs.MAX_WEIGHT,
        allow_inf_nan=False)


class KcPrefRow(BaseModel):
    kc: str
    enabled: bool
    weight: float


class KcPrefsResponse(BaseModel):
    prefs: Dict[str, KcPrefRow]
    min_weight: float = kc_prefs.MIN_WEIGHT
    max_weight: float = kc_prefs.MAX_WEIGHT


@router.get("/kc-prefs", response_model=KcPrefsResponse)
def list_kc_prefs(user: User = Depends(get_current_user)) -> KcPrefsResponse:
    user_state = get_user_state(str(user.id))
    prefs = {
        kc: KcPrefRow(kc=kc, **kc_prefs.pref_row(user_state, kc))
        for kc in (user_state.kc_prefs or {})
    }
    return KcPrefsResponse(prefs=prefs)


@router.put("/kc-prefs/{kc}", response_model=KcPrefRow)
def update_kc_pref(
    kc: str,
    payload: KcPrefUpdate,
    user: User = Depends(get_current_user),
) -> KcPrefRow:
    if kc not in kc_graph._registry():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Unknown concept: {kc}")
    if payload.enabled is None and payload.weight is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Nothing to change: send enabled and/or weight.")
    user_id = str(user.id)
    user_state = get_user_state(user_id)
    kc_prefs.set_pref(user_state, kc, enabled=payload.enabled, weight=payload.weight)
    save_user_state(user_id)
    return KcPrefRow(kc=kc, **kc_prefs.pref_row(user_state, kc))
