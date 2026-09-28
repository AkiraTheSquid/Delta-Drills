"""Every member's measured learning — `/api/practice/groups/xp`.

Seth, 2026-09-27: joining a group changes the Learner Home. Instead of both
graphs it draws one kind at a time — Daily XP bars or knowledge toward the
course — for every member side by side, with a "Just me" view that is the
solo page. This builds that read: one `learning_xp.summary` per member, the
same payload `/xp` answers, so the client draws everybody with the one chart
code it already has.

── 🔴 ONE ZONE FOR EVERYBODY ────────────────────────────────────────────
Each member's days are cut at the VIEWER's midnight. Members in other zones
see their own page cut at theirs; on the board a late-night answer can land
on the neighbouring day. Asking every member's zone would need a stored zone
per account, which nothing keeps.

── 🔴 WHAT JOINING SHARES ─────────────────────────────────────────────────
A member's daily XP, knowledge, target and projected finish, next to the
display name they chose. `user_id` is dropped on the way out, exactly as
`groups_router._with_mastery` drops it: the invite is a link anyone can hold.

A member whose state will not load answers `xp: null` rather than taking the
endpoint out, and the cost is one state load and one replay per member,
bounded by `study_groups.MAX_MEMBERS`.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app import learning_xp, study_groups
from app.adaptive import get_user_state
from app.models import User

logger = logging.getLogger(__name__)


def _summary(user_id: str, zone) -> Optional[dict]:
    try:
        return learning_xp.summary(get_user_state(user_id), zone)
    except Exception as exc:  # pragma: no cover — one bad state must not blank the group
        logger.warning("group xp: could not read %s: %s", user_id, exc)
        return None


def read_group_xp(db: Session, user: User, zone) -> dict:
    """`{group: null}` outside a group; else the group and each member's summary,
    the caller first and the rest in join order."""
    group = study_groups.read_my_group(db, user)
    if not group:
        return {"group": None, "members": []}
    payload = study_groups.group_payload(db, group, user)
    members = []
    for m in payload["members"]:
        members.append({
            "member_id": m["member_id"],
            "display_name": m["display_name"],
            "initials": m["initials"],
            "is_you": m["member_id"] == payload["member_id"],
            "xp": _summary(m["user_id"], zone),
        })
    members.sort(key=lambda m: not m["is_you"])  # stable: the rest keep join order
    return {"group": {"group_id": payload["group_id"], "name": payload["name"]}, "members": members}
