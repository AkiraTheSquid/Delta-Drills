"""The group's day checklists — one three-state list per person, per day.

The Groups page draws each member as a ROW with two columns: their area mastery
on the left, and on the right the checklist they wrote for the selected day
(Seth, 2026-09-02). This module is the store behind the right-hand column.

It is a port of Delta Note's sub-goal checklist (`js/subgoals/`), and the thing
being ported is the DOCUMENT: a Tiptap `{v, doc}` JSON string whose task items
carry a three-state `completion` attr ('open' | 'checked' | 'x'). The server
never looks inside it.

── 🔴 THE ROW IS KEYED BY PERSON AND DAY, NEVER BY GROUP ───────────────────
`study_group_days` has no `group_id`. A checklist is something a learner wrote,
so it follows them out of one group and into the next rather than being deleted
with the membership row. The group decides only who may READ it, and that check
lives in `read_group_day` — the one function here that answers about anybody
other than the caller.

── 🔴 THE DAY IS THE LEARNER'S LOCAL DATE, PARSED NOT DERIVED ──────────────
`parse_day` takes the `YYYY-MM-DD` the browser computed from its own clock. The
server never asks itself what day it is: for a learner eight hours away the two
answers differ for a third of every day, and a checklist written under the
server's date does not fail — it silently reads back empty, on a page that has
an empty state and therefore no way to tell the difference.

── WHAT THE PAYLOAD IS ALLOWED TO BE ───────────────────────────────────────
An opaque string, capped at `MAX_PAYLOAD_CHARS`. Text rather than JSONB because
it is round-tripped and never queried, which is also what lets the document
schema change without a migration. The cap is the only validation: this column
is written by a debounced editor on every keystroke pause, and an uncapped Text
column behind an authenticated PUT is a way to fill a database.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import StudyGroup, StudyGroupDay, StudyGroupMember, User
from app.study_groups import GroupError

# A day's checklist. Delta Note's own lists run to a few hundred characters; a
# person pasting a document into one is the case this bounds, not a person
# writing a long day.
MAX_PAYLOAD_CHARS = 20_000


def parse_day(raw: str) -> date:
    """`YYYY-MM-DD` → a date, or a refusal.

    Strict: an unparseable day must not fall back to today. A silent fallback
    writes one person's Monday into another person's Tuesday and the page shows
    a plausible, wrong list.
    """
    try:
        return date.fromisoformat(str(raw or "").strip())
    except ValueError as exc:
        raise GroupError("That is not a date this page can read.", 400) from exc


def _row(db: Session, user_id, day: date) -> Optional[StudyGroupDay]:
    return (
        db.query(StudyGroupDay)
        .filter(StudyGroupDay.user_id == user_id, StudyGroupDay.day == day)
        .first()
    )


def write_day(db: Session, user: User, day: date, payload: str) -> str:
    """Store THIS user's checklist for `day`. Returns what was stored.

    Upsert by read. The unique constraint is what makes that safe: two saves
    racing (the editor's debounce firing as a teardown flushes) both miss the
    existing row, and only one of them can write it — the loser retries against
    the row the winner just made rather than creating a second one that half the
    reads would then miss.
    """
    text = str(payload or "")
    if len(text) > MAX_PAYLOAD_CHARS:
        raise GroupError("That checklist is too long to save.", 413)

    row = _row(db, user.id, day)
    if row is None:
        row = StudyGroupDay(user_id=user.id, day=day, payload=text)
        db.add(row)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            row = _row(db, user.id, day)
            if row is None:  # pragma: no cover — the constraint says otherwise
                raise
            row.payload = text
    else:
        row.payload = text
    db.commit()
    return text


def read_group_day(db: Session, user: User, day: date) -> dict:
    """Every member of the caller's group, keyed by `member_id`, for one day.

    🔴 Keyed by `member_id`, not by `user_id`. The roster endpoint drops account
    ids on the way out on purpose (see `groups_router._with_mastery`), and a
    second endpoint that shipped them would undo that quietly — the page would
    still work, so nothing would ever notice.

    Refuses for somebody in no group: there is no such thing as "my group's
    checklists" without a group, and answering `{}` would make the page's own
    empty state indistinguishable from a bug.
    """
    membership = (
        db.query(StudyGroupMember).filter(StudyGroupMember.user_id == user.id).first()
    )
    if membership is None:
        raise GroupError("You are not in a group.", 404)
    group = db.query(StudyGroup).filter(StudyGroup.id == membership.group_id).first()
    if group is None:  # pragma: no cover — the FK says otherwise
        raise GroupError("You are not in a group.", 404)

    members = (
        db.query(StudyGroupMember)
        .filter(StudyGroupMember.group_id == group.id)
        .order_by(StudyGroupMember.joined_at.asc())
        .all()
    )
    if not members:
        return {}
    rows = (
        db.query(StudyGroupDay)
        .filter(
            StudyGroupDay.user_id.in_([m.user_id for m in members]),
            StudyGroupDay.day == day,
        )
        .all()
    )
    by_user = {str(r.user_id): r.payload or "" for r in rows}
    # Every member appears, including the ones who wrote nothing: the column has
    # to be drawn for them too, and "" is what the client reads as "no list yet".
    return {str(m.id): by_user.get(str(m.user_id), "") for m in members}
