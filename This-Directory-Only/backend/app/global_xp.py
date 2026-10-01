"""This week's XP for EVERY learner — `/api/practice/leaderboard/xp`.

Seth, 2026-09-30: "make it such that the leaderboard is across the board
for everyone, rather than being scoped to groups." The Learner Home's
week board (practice/xp-group-view.js) read only the caller's study group
(app/group_xp.py); this reads every learner who answered anything since
Monday.

── WHO IS ON IT ─────────────────────────────────────────────────────────
A learner is a candidate when their attempt log was written since the
week's Monday midnight (file mtime — no log read for the idle majority) and
the account still exists. The caller is always on it, at 0 if need be.
Everybody else needs XP > 0 this week.

── 🔴 NAMES, NEVER EMAILS ──────────────────────────────────────────────────
Accounts carry no name; the only name a learner ever chose is their study
group display name (`StudyGroupMember.display_name`, already cleaned of
emails) — and they chose it FOR THAT GROUP. So a name is shown only to a
viewer who shares a group with that learner (codex, 2026-09-30); everyone
else reads "Learner". `user_id` never leaves.

── ONE ZONE FOR EVERYBODY ──────────────────────────────────────────────────
As in group_xp.py: the week and each learner's days are cut at the
VIEWER's midnight.

Cost: one state load + one replay (~10 ms) per learner active this week;
other learners' totals are cached CACHE_S per (learner, zone, week), the
caller's own never is.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from app import learning_xp, study_groups
from app.adaptive import DATA_DIR, get_user_state
from app.models import StudyGroupMember, User

logger = logging.getLogger(__name__)

ANON = "Learner"
MAX_ROWS = 50
CACHE_S = 120
_SUFFIX = ".attempts.jsonl"
_cache: Dict[Tuple[str, str, str], Tuple[float, Optional[float]]] = {}


def week_start(today: date) -> date:
    """Monday of `today`'s week."""
    return today - timedelta(days=today.weekday())


def _active_since(since: datetime, data_dir=None) -> List[str]:
    """User ids whose attempt log was written at or after `since`."""
    root = data_dir if data_dir is not None else DATA_DIR
    cut = since.timestamp()
    out = []
    try:
        for p in root.glob(f"*{_SUFFIX}"):
            try:
                if p.stat().st_mtime >= cut:
                    out.append(p.name[: -len(_SUFFIX)])
            except OSError:
                continue
    except OSError:
        return []
    return out


def _uuids(ids: Iterable[str]) -> Dict[uuid.UUID, str]:
    out = {}
    for i in ids:
        try:
            out[uuid.UUID(i)] = i
        except ValueError:
            continue  # a test or legacy file name — never an account
    return out


def _names(db: Session, viewer: uuid.UUID, ids: Iterable[uuid.UUID]) -> Dict[uuid.UUID, str]:
    """Each learner's display name in a group they share with `viewer`
    (their most recent such group); no shared group → no name."""
    mine: Set = {g for (g,) in db.query(StudyGroupMember.group_id)
                 .filter(StudyGroupMember.user_id == viewer).all()}
    if not mine:
        return {}
    rows = (db.query(StudyGroupMember.user_id, StudyGroupMember.display_name)
            .filter(StudyGroupMember.user_id.in_(list(ids)),
                    StudyGroupMember.group_id.in_(list(mine)))
            .order_by(StudyGroupMember.joined_at).all())
    return {uid: name for uid, name in rows}  # later joins overwrite earlier


def _week_xp(user_id: str, zone, now: datetime, monday: date, cached: bool) -> Optional[float]:
    key = (user_id, str(zone), monday.isoformat())
    hit = _cache.get(key) if cached else None
    if hit and time.monotonic() - hit[0] < CACHE_S:
        return hit[1]
    value = _replay_week(user_id, zone, now, monday)
    _cache[key] = (time.monotonic(), value)
    return value


def _replay_week(user_id: str, zone, now: datetime, monday: date) -> Optional[float]:
    try:
        r = learning_xp.replay(get_user_state(user_id), zone, now)
    except Exception as exc:  # pragma: no cover — one bad state must not blank the board
        logger.warning("leaderboard: could not read %s: %s", user_id, exc)
        return None
    return round(sum(x for d, x in zip(r["days"], r["xp"]) if d >= monday), 1)


def read_board(db: Session, user: User, zone, now: Optional[datetime] = None,
               data_dir=None) -> dict:
    """`{week_start, rows}`: rows most XP first, competition-ranked (1, 1, 3)."""
    now = now or datetime.now(timezone.utc)
    monday = week_start(now.astimezone(zone).date())
    since = datetime.combine(monday, datetime.min.time(), tzinfo=zone)
    me = str(user.id)
    ids = _uuids(set(_active_since(since, data_dir)) | {me})
    known = {uid for (uid,) in db.query(User.id).filter(User.id.in_(list(ids))).all()}
    names = _names(db, user.id, known)

    rows = []
    for uid in known:
        sid = ids[uid]
        is_you = sid == me
        week = _week_xp(sid, zone, now, monday, cached=not is_you)
        if not is_you and not week:
            continue
        name = names.get(uid) or ANON
        rows.append({"display_name": name, "initials": study_groups.initials_from_name(name),
                     "is_you": is_you, "xp": week, "_id": sid})
    # Ties: you first, then a fixed order so a re-read never shuffles them.
    rows.sort(key=lambda r: (-(r["xp"] if r["xp"] is not None else -1), not r["is_you"], r["_id"]))
    for r in rows:
        del r["_id"]
    rank, prev = 0, object()
    for i, r in enumerate(rows):
        if r["xp"] != prev:
            rank = i + 1
        prev = r["xp"]
        r["rank"] = rank
    # Long boards keep the top and the caller.
    if len(rows) > MAX_ROWS:
        rows = rows[:MAX_ROWS] + [r for r in rows[MAX_ROWS:] if r["is_you"]]
    return {"week_start": monday.isoformat(), "rows": rows}
