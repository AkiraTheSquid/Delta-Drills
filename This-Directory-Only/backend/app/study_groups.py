"""Study groups — the seven database operations a group is made of.

A study group is a handful of learners practising the same curriculum who can
read each other's area mastery side by side. It is a port of Delta Note's
accountability groups (`shared/web-components/js/accountability/`, backed by
`deployed-web/supabase/migrations/00023_accountability_groups.sql`) onto this
app's own Postgres, and the rules are deliberately the same ones:

  * **Three ways in, one gate.** Start a group, paste an invite link, or pick a
    listed group out of the directory. All three arrive at `_add_member`, so a
    door added later cannot be the one that skips a check.
  * **One group per person.** `mine` is singular. The uniqueness is a database
    constraint (`study_group_members_user_id_key`), not a router check, so two
    join clicks racing cannot produce two rows.
  * **The join token is a CAPABILITY.** Anyone holding it is in the group and
    can read every member's mastery. 128 bits of entropy, rotatable by the
    owner without disturbing the roster, and never returned by the public
    directory read.
  * **A group always has a live owner.** Only the owner may rotate the token or
    change the listing, so `leave_group` hands the group to the longest-standing
    remaining member when the owner walks out. Left alone, the row would point
    at somebody who is not in the group: the controls simply stop being drawn
    for everybody, with no error and no way back.
  * **The directory answers INITIALS, never names.** `list_public_groups` is
    the only door here a non-member may open, so it returns a group's name, its
    member count and two letters per member. There is no `display_name ?? …`
    fallback anywhere on that path; if the tighter answer ever stops arriving
    the list shows question marks, which is visible, rather than quietly
    starting to show names, which would not be.

🔴 WHAT THIS MODULE DOES NOT DO: read anybody's practice state. The mastery
numbers on the group page come from `diagnostic.display_area_estimates`, which
`groups_router` calls per member. Keeping the roster store and the mastery read
apart is what lets the roster be answered from one query while the expensive
per-member read stays optional.
"""

from __future__ import annotations

import re
import secrets
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import StudyGroup, StudyGroupMember, User


class GroupError(Exception):
    """A refusal a person just earned by clicking something.

    Carries an HTTP status because every caller is a route and every one of
    them would otherwise re-derive the same mapping. A button that silently
    does nothing is worse than one that says why.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# The cap Delta Note's migration enforces, kept so the two features behave the
# same way. A group is a readout you scan in one glance; past a dozen columns
# it is a directory, which is a different feature.
MAX_MEMBERS = 12

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")


def mint_token() -> str:
    """A fresh join token: 32 hex characters, 128 bits.

    `secrets`, not `uuid4`: this string is the only thing standing between a
    stranger and everyone in the group's mastery profile.
    """
    return secrets.token_hex(16)


def is_group_token(token: str) -> bool:
    return bool(_TOKEN_RE.match(str(token or "").strip().lower()))


def initials_from_name(name: str) -> str:
    """Two letters for one member, computed HERE so the directory can show a
    member without shipping their name.

    Mirrors `initialsFromName` in Delta Note's `accountability_avatars.js`:
    parentheticals dropped, non-letters dropped, first letter of the first two
    words. `?` when there is nothing left, because an empty circle reads as a
    rendering bug.
    """
    cleaned = re.sub(r"\([^)]*\)", " ", str(name or ""))
    # Letters only: digits and punctuation make poor initials, and "3." as a
    # circle reads as a rendering fault rather than as a person.
    words = "".join(ch if ch.isalpha() else " " for ch in cleaned).split()
    letters = "".join(w[0] for w in words[:2]).upper()
    return letters or "?"


def clean_display_name(raw: str, fallback: str = "Learner") -> str:
    """What to call somebody in a group.

    🔴 Never an email address — see the note on `StudyGroupMember.display_name`.
    An address that arrives here anyway is reduced to its local part rather than
    refused, because the client's default name is derived from the account and a
    hard refusal on the join path would be a dead button.
    """
    name = " ".join(str(raw or "").split())[:120]
    if "@" in name:
        name = name.split("@")[0].replace(".", " ").replace("_", " ").replace("-", " ")
        name = " ".join(name.split())
    return name or fallback


def _member_row(member: StudyGroupMember) -> dict:
    return {
        "member_id": str(member.id),
        "user_id": str(member.user_id),
        "display_name": member.display_name,
        "initials": initials_from_name(member.display_name),
        "joined_at": member.joined_at.isoformat() if member.joined_at else None,
    }


def group_payload(db: Session, group: StudyGroup, user: User) -> dict:
    """One group as a MEMBER may see it: names, the token, the whole roster.

    The token is in here because the member is the person who hands out invites.
    `list_public_groups` builds its rows separately and never calls this.
    """
    members = (
        db.query(StudyGroupMember)
        .filter(StudyGroupMember.group_id == group.id)
        .order_by(StudyGroupMember.joined_at.asc())
        .all()
    )
    mine = next((m for m in members if str(m.user_id) == str(user.id)), None)
    return {
        "group_id": str(group.id),
        "name": group.name,
        "join_token": group.join_token,
        "member_id": str(mine.id) if mine else "",
        # Defaulted CLOSED. A row written before this column existed, or one
        # that arrived from a channel where the migration has not run, has no
        # visibility at all — and a group whose listing state is unknown must be
        # drawn as unlisted, or the bar tells somebody their group is private
        # while a directory somewhere is showing it.
        "visibility": "public" if group.visibility == "public" else "private",
        "is_owner": str(group.owner_user_id) == str(user.id),
        "members": [_member_row(m) for m in members],
    }


def read_my_group(db: Session, user: User) -> Optional[StudyGroup]:
    """The one group this user is in, or None."""
    membership = (
        db.query(StudyGroupMember).filter(StudyGroupMember.user_id == user.id).first()
    )
    if not membership:
        return None
    return db.query(StudyGroup).filter(StudyGroup.id == membership.group_id).first()


def _refuse_if_already_in_a_group(db: Session, user: User) -> None:
    if read_my_group(db, user) is not None:
        raise GroupError(
            "You are already in a group. Leave it before joining another.", 409
        )


def _add_member(db: Session, group: StudyGroup, user: User, display_name: str) -> None:
    """THE ONE GATE all three ways in pass through.

    The membership cap and the one-group rule are checked here rather than in
    each caller, and the `IntegrityError` catch is what makes the one-group rule
    real: two join clicks racing both pass the check above and only one can
    write the row.
    """
    count = (
        db.query(StudyGroupMember).filter(StudyGroupMember.group_id == group.id).count()
    )
    if count >= MAX_MEMBERS:
        raise GroupError(f"That group is full ({MAX_MEMBERS} members).", 409)
    db.add(
        StudyGroupMember(
            group_id=group.id,
            user_id=user.id,
            display_name=clean_display_name(display_name),
        )
    )
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise GroupError("You are already in a group.", 409) from exc


def create_group(db: Session, user: User, name: str, display_name: str, visibility: str) -> StudyGroup:
    _refuse_if_already_in_a_group(db, user)
    group = StudyGroup(
        name=" ".join(str(name or "").split())[:120] or "Study group",
        join_token=mint_token(),
        owner_user_id=user.id,
        visibility="public" if visibility == "public" else "private",
    )
    db.add(group)
    db.flush()
    _add_member(db, group, user, display_name)
    db.commit()
    db.refresh(group)
    return group


def join_by_token(db: Session, user: User, token: str, display_name: str) -> StudyGroup:
    token = str(token or "").strip().lower()
    if not is_group_token(token):
        raise GroupError("That does not look like an invite link.", 400)
    _refuse_if_already_in_a_group(db, user)
    group = db.query(StudyGroup).filter(StudyGroup.join_token == token).first()
    if not group:
        # Deliberately the same sentence for "no such group" and "the owner
        # replaced the link": a stranger probing tokens learns nothing from the
        # difference, and the person who was actually invited needs the same
        # next step either way.
        raise GroupError("That invite link is not valid any more.", 404)
    _add_member(db, group, user, display_name)
    db.commit()
    db.refresh(group)
    return group


def join_public_group(db: Session, user: User, group_id: str, display_name: str) -> StudyGroup:
    _refuse_if_already_in_a_group(db, user)
    group = db.query(StudyGroup).filter(StudyGroup.id == group_id).first()
    # 🔴 The visibility check is HERE, not only in the directory. A group id
    # that was public an hour ago is still an id somebody could POST; unlisting
    # a group has to actually shut this door or the toggle is decorative.
    if not group or group.visibility != "public":
        raise GroupError("That group is not open to join.", 404)
    _add_member(db, group, user, display_name)
    db.commit()
    db.refresh(group)
    return group


def leave_group(db: Session, user: User, group_id: str) -> None:
    membership = (
        db.query(StudyGroupMember)
        .filter(
            StudyGroupMember.user_id == user.id,
            StudyGroupMember.group_id == group_id,
        )
        .first()
    )
    if not membership:
        raise GroupError("You are not in that group.", 404)
    db.delete(membership)
    db.flush()
    # The last person out takes the group with them. A group row with no members
    # is unreachable — nobody holds the token, nobody can list it — so leaving it
    # behind would only accumulate rows and keep its name reserved.
    remaining = (
        db.query(StudyGroupMember)
        .filter(StudyGroupMember.group_id == group_id)
        .order_by(StudyGroupMember.joined_at.asc(), StudyGroupMember.id.asc())
        .all()
    )
    group = db.query(StudyGroup).filter(StudyGroup.id == group_id).first()
    if not remaining:
        if group:
            db.delete(group)
    elif group and str(group.owner_user_id) == str(user.id):
        # 🔴 THE OWNER LEAVING MUST HAND THE GROUP OVER. `owner_user_id` is the
        # only thing that authorises rotating the invite token and changing the
        # listing; a departed owner leaves a group whose token can never be
        # retired and whose visibility can never be changed, by anybody, ever.
        # There is no UI for that state and no error either — the buttons simply
        # stop being drawn for everybody, which reads as a bug rather than as a
        # locked door.
        #
        # Longest-standing remaining member, ties broken by id so the choice is
        # deterministic when two people joined inside the same clock tick.
        group.owner_user_id = remaining[0].user_id
    db.commit()


def _owned_group(db: Session, user: User, group_id: str) -> StudyGroup:
    group = db.query(StudyGroup).filter(StudyGroup.id == group_id).first()
    if not group:
        raise GroupError("No such group.", 404)
    if str(group.owner_user_id) != str(user.id):
        raise GroupError("Only the person who started the group can do that.", 403)
    return group


def rotate_token(db: Session, user: User, group_id: str) -> StudyGroup:
    """Retire the old invite link. The ROSTER IS UNTOUCHED — this shuts the door
    for people who have not come through it, not for people already inside."""
    group = _owned_group(db, user, group_id)
    group.join_token = mint_token()
    db.commit()
    db.refresh(group)
    return group


def set_visibility(db: Session, user: User, group_id: str, visibility: str) -> StudyGroup:
    group = _owned_group(db, user, group_id)
    group.visibility = "public" if visibility == "public" else "private"
    db.commit()
    db.refresh(group)
    return group


def set_display_name(db: Session, user: User, display_name: str) -> StudyGroup:
    membership = (
        db.query(StudyGroupMember).filter(StudyGroupMember.user_id == user.id).first()
    )
    if not membership:
        raise GroupError("You are not in a group.", 404)
    membership.display_name = clean_display_name(display_name)
    db.commit()
    group = db.query(StudyGroup).filter(StudyGroup.id == membership.group_id).first()
    if not group:
        raise GroupError("You are not in a group.", 404)
    return group


def list_public_groups(db: Session, user: User) -> list[dict]:
    """Every listed group, as a STRANGER may see it.

    🔴 No token, no display names, no user ids. Initials and a count. This is
    the only function in the module a non-member calls, and the boundary is kept
    real by there being no name on the path at all — see the module docstring.
    """
    groups = (
        db.query(StudyGroup)
        .filter(StudyGroup.visibility == "public")
        .order_by(StudyGroup.created_at.desc())
        .limit(50)
        .all()
    )
    if not groups:
        return []
    ids = [g.id for g in groups]
    members = (
        db.query(StudyGroupMember)
        .filter(StudyGroupMember.group_id.in_(ids))
        .order_by(StudyGroupMember.joined_at.asc())
        .all()
    )
    by_group: dict[str, list[StudyGroupMember]] = {}
    for member in members:
        by_group.setdefault(str(member.group_id), []).append(member)

    out: list[dict] = []
    for group in groups:
        roster = by_group.get(str(group.id), [])
        out.append(
            {
                "group_id": str(group.id),
                "name": group.name,
                "member_count": len(roster),
                "is_member": any(str(m.user_id) == str(user.id) for m in roster),
                "members": [
                    {
                        "member_id": str(m.id),
                        "initials": initials_from_name(m.display_name),
                    }
                    for m in roster
                ],
            }
        )
    return out
