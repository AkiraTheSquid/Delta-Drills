"""Study-group endpoints — `/api/practice/groups/*`.

The Groups tab: start a group, join one by invite link or out of the public
directory, and read every member's area mastery beside your own (Seth,
2026-09-02). The roster half is `app.study_groups`; this router owns the
request shapes, the refusals, and the one thing that is genuinely this file's
own — turning a roster into a page of mastery bars.

── 🔴 THE MASTERY READ IS PER MEMBER AND IT IS NOT FREE ────────────────────
`/groups/mine` answers every member's `areas` in the SAME shape
`/diagnostic/status` answers for one learner, because the Groups page draws
them with the Learner Home's own renderer (`PlacementResults.renderAreas`) and a
second shape would be a second copy of the theta→readiness map to keep in step.
Getting them costs one practice-state load and one posterior recompute per
member — bounded by `study_groups.MAX_MEMBERS`, which is why that cap exists on
this side of the feature as well as on Delta Note's.

A member whose state cannot be read yields `areas: []` rather than taking the
endpoint out. One unreadable state file must not be able to blank the group.

── 🔴 WHAT JOINING SHARES ─────────────────────────────────────────────────
Everything on this page: your display name, your area mastery, and how much of
it the placement actually measured. There is no per-member opt-out, so the
consent is asked once, on the client, in front of all three ways in — the same
single-gate shape Delta Note's `accountability_discovery.js` uses. Leaving
removes the membership row, and with it every readout of you the other members
had.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import diagnostic, study_group_days, study_groups
from app.adaptive import get_user_state
from app.auth import get_current_user
from app.db import get_db
from app.models import User
from app.study_groups import GroupError

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateGroupRequest(BaseModel):
    name: str = Field(default="Study group", max_length=120)
    display_name: str = Field(default="Learner", max_length=120)
    visibility: str = Field(default="private")


class JoinTokenRequest(BaseModel):
    token: str = Field(max_length=64)
    display_name: str = Field(default="Learner", max_length=120)


class JoinPublicRequest(BaseModel):
    group_id: str = Field(max_length=64)
    display_name: str = Field(default="Learner", max_length=120)


class GroupIdRequest(BaseModel):
    group_id: str = Field(max_length=64)


class VisibilityRequest(BaseModel):
    group_id: str = Field(max_length=64)
    visibility: str


class DisplayNameRequest(BaseModel):
    display_name: str = Field(max_length=120)


def _member_areas(user_id: str) -> list[dict]:
    """One member's area mastery, in `/diagnostic/status`'s `areas` shape.

    Swallows on purpose: this is a readout of somebody else, and a state file
    that will not load is a reason to draw that one column as unmeasured, not a
    reason for nobody in the group to see anything.
    """
    try:
        return diagnostic.display_area_estimates(get_user_state(user_id))
    except Exception as exc:  # pragma: no cover — defensive by design
        logger.warning("groups: could not read areas for %s: %s", user_id, exc)
        return []


def _with_mastery(payload: dict) -> dict:
    """Fold each member's areas into the roster payload.

    🔴 The user_id is dropped on the way out. The client needs a stable key per
    member and `member_id` already is one; a group is joined by anyone holding a
    link, and shipping account ids to it would make the invite a way of
    correlating people across the rest of the app.
    """
    members = []
    for member in payload.get("members", []):
        areas = _member_areas(member.get("user_id", ""))
        members.append(
            {
                "member_id": member["member_id"],
                "display_name": member["display_name"],
                "initials": member["initials"],
                "joined_at": member["joined_at"],
                "areas": areas,
                "probes": sum(int(a.get("probes") or 0) for a in areas),
            }
        )
    return {**payload, "members": members}


def _mine(db: Session, user: User) -> dict:
    group = study_groups.read_my_group(db, user)
    if not group:
        return {"group": None}
    return {"group": _with_mastery(study_groups.group_payload(db, group, user))}


def _answer(db: Session, user: User, group) -> dict:
    return {"group": _with_mastery(study_groups.group_payload(db, group, user))}


def _fail(exc: GroupError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("/groups/mine")
def groups_mine(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    """Your group and everyone in it, with mastery. `{"group": null}` when you
    are in none — an empty answer, not a 404: "you are in no group" is the state
    the discovery card is for, not an error."""
    return _mine(db, user)


@router.get("/groups/public")
def groups_public(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    """The listed groups. Initials and counts only — see `list_public_groups`."""
    return {"groups": study_groups.list_public_groups(db, user)}


@router.post("/groups")
def groups_create(
    body: CreateGroupRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        group = study_groups.create_group(
            db, user, body.name, body.display_name, body.visibility
        )
    except GroupError as exc:
        raise _fail(exc) from exc
    return _answer(db, user, group)


@router.post("/groups/join")
def groups_join(
    body: JoinTokenRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        group = study_groups.join_by_token(db, user, body.token, body.display_name)
    except GroupError as exc:
        raise _fail(exc) from exc
    return _answer(db, user, group)


@router.post("/groups/join-public")
def groups_join_public(
    body: JoinPublicRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        group = study_groups.join_public_group(
            db, user, body.group_id, body.display_name
        )
    except GroupError as exc:
        raise _fail(exc) from exc
    return _answer(db, user, group)


@router.post("/groups/leave")
def groups_leave(
    body: GroupIdRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        study_groups.leave_group(db, user, body.group_id)
    except GroupError as exc:
        raise _fail(exc) from exc
    return {"group": None}


@router.post("/groups/rotate-token")
def groups_rotate_token(
    body: GroupIdRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        group = study_groups.rotate_token(db, user, body.group_id)
    except GroupError as exc:
        raise _fail(exc) from exc
    return _answer(db, user, group)


@router.post("/groups/visibility")
def groups_visibility(
    body: VisibilityRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        group = study_groups.set_visibility(db, user, body.group_id, body.visibility)
    except GroupError as exc:
        raise _fail(exc) from exc
    return _answer(db, user, group)


@router.post("/groups/display-name")
def groups_display_name(
    body: DisplayNameRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        group = study_groups.set_display_name(db, user, body.display_name)
    except GroupError as exc:
        raise _fail(exc) from exc
    return _answer(db, user, group)


# ── THE DAY CHECKLISTS ──────────────────────────────────────────────────────
#
# The right-hand column of a member row: the three-state list that person wrote
# for the selected day. Deliberately its OWN pair of endpoints rather than a
# field folded into `/groups/mine`, for two reasons that pull the same way:
#
#   * Changing the day must not recompute anybody's mastery.
#     `_with_mastery` costs one practice-state load and one posterior recompute
#     PER MEMBER, and the day picker is a control a person clicks through a
#     week. Folding the checklists into the roster read would make walking back
#     seven days twelve state loads a click.
#   * The roster changes when somebody joins; a checklist changes when somebody
#     types. They are read on different occasions and they should not share a
#     cache-invalidation story.
#
# The store is `app.study_group_days`, and the day is the LEARNER'S LOCAL DATE
# as their browser named it — never derived here. See that module's header.


class DayWriteRequest(BaseModel):
    date: str = Field(max_length=32)
    # The Tiptap `{v, doc}` JSON string. Opaque to the server; the length cap
    # that actually protects the column lives in `study_group_days.write_day`,
    # and this one only keeps an absurd body out of the parser.
    payload: str = Field(default="", max_length=64_000)


@router.get("/groups/day")
def groups_day(
    date: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Every member's checklist for one day, keyed by `member_id`.

    A member who wrote nothing is present with an empty string, so the client
    can draw the column for everybody without having to tell "no row" apart
    from "not in this answer".
    """
    try:
        day = study_group_days.parse_day(date)
        entries = study_group_days.read_group_day(db, user, day)
    except GroupError as exc:
        raise _fail(exc) from exc
    return {"date": day.isoformat(), "entries": entries}


@router.put("/groups/day")
def groups_day_write(
    body: DayWriteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Store YOUR OWN checklist for one day.

    🔴 There is no member_id parameter and there must never be one. The row is
    keyed by the authenticated user, so the only checklist this endpoint can
    write is the caller's — a group is joined by anyone holding a link, and an
    endpoint that took a target would let any member rewrite anybody's day.
    """
    try:
        day = study_group_days.parse_day(body.date)
        stored = study_group_days.write_day(db, user, day, body.payload)
    except GroupError as exc:
        raise _fail(exc) from exc
    return {"date": day.isoformat(), "payload": stored}
