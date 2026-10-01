"""This week's XP for EVERY learner — `/api/practice/leaderboard/xp`.

Seth, 2026-09-30: "make it such that the leaderboard is across the board
for everyone, rather than being scoped to groups." Seth, 2026-10-01: "the
leaderboard should display everyone, including their names as well as their
total XP for Monday through Sunday … by default [they] are displayed on the
leaderboard, but they can choose to change their username".

── WHO IS ON IT ─────────────────────────────────────────────────────────
Everyone who has ever answered a question (an attempt log exists) and whose
account still exists, plus the caller: this week's XP, 0 included. Only a
log written since the week's Monday midnight (file mtime) is replayed; an
older one is 0 without reading it. Accounts that never practised are left
off (360 accounts, 18 logs on 2026-10-01), and so are guest sessions
(`guest-…@guest.delta-drills.app`, throwaway) with no XP this week, and
test accounts (`test-…@test.delta-drills.app`, minted by the Account tab's
test-users.js) always — Seth, 2026-10-01: "make it such that it doesn't
add test accounts to the leaderboard". Signed in AS one, you still see
your own row.

── NAMES ────────────────────────────────────────────────────────────────
Shown to everyone, by default: the name the learner chose for the board
(`LeaderboardName`, PUT /leaderboard/name → `set_name`), else their study
group display name, else "Guest" for a guest session, else their email's
local part as
`study_groups.clean_display_name` already reduces it for a group's default
name ("ada.lovelace@x" → "ada lovelace"). 🔴 Never the address itself, and
`user_id` never leaves.

── ONE ZONE FOR EVERYBODY ──────────────────────────────────────────────────
As in group_xp.py: the week and each learner's days are cut at the
VIEWER's midnight. The week is Monday through Sunday; days not yet come
count nothing.

Cost: one state load + one replay (~10 ms) per learner active this week;
other learners' totals are cached CACHE_S per (learner, zone, week), the
caller's own never is.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, Optional, Tuple

from sqlalchemy.orm import Session

from app import learning_xp, study_groups
from app.quackback_sso import is_guest_email
from app.adaptive import DATA_DIR, get_user_state
from app.models import LeaderboardName, StudyGroupMember, User

logger = logging.getLogger(__name__)

ANON = "Learner"
GUEST = "Guest"
TEST_EMAIL_DOMAIN = "test.delta-drills.app"  # = TEST_EMAIL_DOMAIN in test-users.js
NAME_MAX = 24
MAX_ROWS = 200  # everyone today (18 logs on 2026-10-01); a bound, not a top-N
CACHE_S = 120
_SUFFIX = ".attempts.jsonl"
_cache: Dict[Tuple[str, str, str], Tuple[float, Optional[float]]] = {}


def week_start(today: date) -> date:
    """Monday of `today`'s week."""
    return today - timedelta(days=today.weekday())


def _logs(data_dir=None) -> Dict[str, float]:
    """Every attempt log's owner id → the log's mtime."""
    root = data_dir if data_dir is not None else DATA_DIR
    out = {}
    try:
        for p in root.glob(f"*{_SUFFIX}"):
            try:
                out[p.name[: -len(_SUFFIX)]] = p.stat().st_mtime
            except OSError:
                continue
    except OSError:
        return {}
    return out


def _is_test(email: Optional[str]) -> bool:
    return str(email or "").lower().strip().endswith("@" + TEST_EMAIL_DOMAIN)


def _uuids(ids: Iterable[str]) -> Dict[uuid.UUID, str]:
    out = {}
    for i in ids:
        try:
            out[uuid.UUID(i)] = i
        except ValueError:
            continue  # a test or legacy file name — never an account
    return out


def _names(db: Session, users: Dict[uuid.UUID, str]) -> Dict[uuid.UUID, str]:
    """Each learner's board name: chosen, else group, else email local part."""
    ids = list(users)
    chosen = {uid: n for uid, n in db.query(LeaderboardName.user_id, LeaderboardName.name)
              .filter(LeaderboardName.user_id.in_(ids)).all()}
    group = {uid: n for uid, n in db.query(StudyGroupMember.user_id, StudyGroupMember.display_name)
             .filter(StudyGroupMember.user_id.in_(ids)).all()}
    out = {}
    for uid, email in users.items():
        name = (chosen.get(uid) or group.get(uid)
                or (GUEST if is_guest_email(email) else study_groups.clean_display_name(email or "", fallback=ANON)))
        out[uid] = name[:40]
    return out


def set_name(db: Session, user: User, raw: Optional[str]) -> str:
    """Store the name the learner is listed under; blank resets it to the
    default. Returns the name the board will now show. Raises ValueError
    with a sentence the UI can show."""
    name = " ".join("".join(ch for ch in str(raw or "") if ch.isprintable()).split())
    if "@" in name:
        raise ValueError("Pick a name rather than an email address.")
    if len(name) > NAME_MAX:
        raise ValueError(f"Keep it to {NAME_MAX} characters.")
    row = db.get(LeaderboardName, user.id)
    if not name:
        if row is not None:
            db.delete(row)
    elif row is None:
        db.add(LeaderboardName(user_id=user.id, name=name))
    else:
        row.name = name
    db.commit()
    return _names(db, {user.id: user.email})[user.id]


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
    """`{week_start, week_end, rows}`: rows most XP first, competition-ranked
    (1, 1, 3)."""
    now = now or datetime.now(timezone.utc)
    monday = week_start(now.astimezone(zone).date())
    cut = datetime.combine(monday, datetime.min.time(), tzinfo=zone).timestamp()
    me = str(user.id)
    logs = _logs(data_dir)
    ids = _uuids(set(logs) | {me})
    users = {uid: email for uid, email in db.query(User.id, User.email).filter(User.id.in_(list(ids))).all()}
    names = _names(db, users)

    rows = []
    for uid in users:
        sid = ids[uid]
        is_you = sid == me
        if not is_you and _is_test(users[uid]):
            continue
        if is_you or logs.get(sid, 0) >= cut:
            week = _week_xp(sid, zone, now, monday, cached=not is_you)
        else:
            week = 0.0  # nothing answered since Monday: no replay needed
        if not is_you and not week and is_guest_email(users[uid]):
            continue
        name = names.get(uid) or ANON
        rows.append({"display_name": name, "initials": study_groups.initials_from_name(name),
                     "is_you": is_you, "xp": week, "_id": sid})
    # Ties: you first, then by name (then id) so a re-read never shuffles them.
    rows.sort(key=lambda r: (-(r["xp"] if r["xp"] is not None else -1), not r["is_you"],
                             r["display_name"].casefold(), r["_id"]))
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
    return {"week_start": monday.isoformat(), "week_end": (monday + timedelta(days=6)).isoformat(),
            "rows": rows}
