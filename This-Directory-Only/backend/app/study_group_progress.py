"""Group charts: answered counts, recorded BKT history, learner-owned targets.

History uses recorded posteriors, never today's score projected backwards.
Unseen concepts remain in the denominator at the model's prior; coverage and
proxy counts travel with every score. The 80/100 line is a planning reference,
not an ARENA pass rule or a replacement for prerequisite gates.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import math
from types import SimpleNamespace
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.exc import IntegrityError

from app import attempt_log, diagnostic, kc_graph
from app.adaptive import get_user_state
from app.models import StudyGroupMember, StudyGroupTarget
from app.practice.activity_router import range_counts
from app.study_groups import GroupError


def sections():
    """Use the graph's exercise-to-section mapping; keep prep separately."""
    groups = {}
    links = diagnostic._arena_links()
    for kc, node in kc_graph._registry().items():
        slugs = sorted(links.get(kc, {}).get("notebooks", []))
        area = slugs[0].replace("-", ".") if slugs else (
            "prep-python" if kc.startswith("python.") else "prep-arrays")
        groups.setdefault(area, []).append(kc)
    labels = {"prep-python": "Python prep", "prep-arrays": "Arrays & tensors prep",
              "0.0": "0.0 · Prerequisites", "0.1": "0.1 · Ray tracing",
              "0.2": "0.2 · CNNs & ResNets"}
    colors = {"prep-python": "#dfae74", "prep-arrays": "#b0b4c0",
              "0.0": "#4f9fe0", "0.1": "#bb7de8", "0.2": "#e8a765"}
    return [{"id": area, "label": labels.get(area, f"ARENA {area}"),
             "color": colors.get(area, f"hsl({(i * 137) % 360} 65% 65%)"), "kcs": kcs,
             "concepts": len(kcs)} for i, (area, kcs) in enumerate(sorted(groups.items()))]


def calendar_range(anchor: date, horizon: str):
    if horizon == "daily":
        return anchor, anchor
    if horizon == "weekly":
        start = anchor - timedelta(days=anchor.weekday())
        return start, start + timedelta(days=6)
    if horizon == "monthly":
        start = anchor.replace(day=1)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        return start, next_month - timedelta(days=1)
    raise GroupError("Choose daily, weekly, or monthly.", 400)


def zone_for(tz_offset: int, tz_name: str | None):
    try:
        return ZoneInfo(tz_name) if tz_name else timezone(timedelta(minutes=-tz_offset))
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise GroupError("That time zone is not recognized.", 400) from exc


def members_for(db, user):
    membership = db.query(StudyGroupMember).filter(StudyGroupMember.user_id == user.id).first()
    if membership is None:
        raise GroupError("You are not in a group.", 404)
    return db.query(StudyGroupMember).filter(
        StudyGroupMember.group_id == membership.group_id
    ).order_by(StudyGroupMember.joined_at.asc()).all()


def mastery_history(state, rows, areas, start, end, zone, now):
    """Daily closing estimates under the current model and concept mapping.

    The state file supplies only timestamped atom observations. Older BKT log
    events fill earlier history; nothing is inferred before the first recorded
    observation. Future days have null scores, even when a target exists there.
    """
    events = []
    for row in rows:
        if row.kind != attempt_log.KIND_BKT_UPDATE:
            continue
        stamp = attempt_log.parse_ts(row.ts)
        changes = (row.feature_sources or {}).get("bkt_changed", {})
        if stamp and isinstance(changes, dict):
            for changed in changes.values():
                if isinstance(changed, dict):
                    events.append((stamp, changed))
    for atom, value in (state.atom_mastery or {}).items():
        stamp = attempt_log.parse_ts((state.atom_last_ts or {}).get(atom))
        if stamp:
            events.append((stamp, {atom: value}))
    events.sort(key=lambda event: event[0])
    cursor = 0
    past = SimpleNamespace(atom_mastery={}, atom_last_ts={},
                           self_reported_level=state.self_reported_level)
    all_kcs = list(dict.fromkeys(kc for area in areas for kc in area["kcs"]))
    scopes = {area["id"]: area["kcs"] for area in areas}
    scopes["aggregate"] = all_kcs
    result = []
    for i in range((end - start).days + 1):
        day = start + timedelta(days=i)
        scores = {key: None for key in scopes}
        if day <= now.astimezone(zone).date():
            cutoff = min(datetime.combine(day + timedelta(days=1), time.min, zone), now)
            while cursor < len(events) and events[cursor][0] < cutoff:
                stamp, changed = events[cursor]
                for atom, value in changed.items():
                    if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1:
                        past.atom_mastery[atom] = value
                        past.atom_last_ts[atom] = stamp.isoformat()
                cursor += 1
            readings = {kc: kc_graph.kc_mastery(past, kc, now=cutoff) for kc in all_kcs}
            for key, kcs in scopes.items():
                values = [readings[kc] for kc in kcs]
                covered = sum(v[1] for v in values)
                if values and covered > 0:
                    scores[key] = {"score": round(100 * sum(v[0] for v in values) / len(values), 2),
                                   "coverage": round(100 * covered / len(values)),
                                   "concepts": len(values),
                                   "proxies": sum(v[2] != "measured" for v in values)}
        result.append({"date": day.isoformat(), "scores": scores})
    return result


def read_progress(db, user, anchor, horizon, tz_offset, tz_name):
    start, end = calendar_range(anchor, horizon)
    zone = zone_for(tz_offset, tz_name)
    members = members_for(db, user)
    areas = sections()
    now = datetime.now(timezone.utc)
    targets = db.query(StudyGroupTarget).filter(
        StudyGroupTarget.user_id.in_([m.user_id for m in members])).all()
    by_user = {}
    for target in targets:
        by_user.setdefault(str(target.user_id), {})[target.area] = {
            "date": target.day.isoformat(), "level": target.level}
    entries = {}
    for member in members:
        uid = str(member.user_id)
        # Raw attempts and account ids never cross the group API boundary.
        state = get_user_state(uid)
        entries[str(member.id)] = {
            "activity": range_counts(uid, start, end, tz_offset, now=now, tz_name=tz_name),
            "history": mastery_history(state, attempt_log.iter_rows(uid), areas, start, end, zone, now),
            "targets": by_user.get(uid, {}),
        }
    return {"start": start.isoformat(), "end": end.isoformat(), "horizon": horizon,
            "today": now.astimezone(zone).date().isoformat(), "benchmark": 80,
            "areas": [{k: v for k, v in area.items() if k != "kcs"} for area in areas],
            "entries": entries}


def write_target(db, user, area, day, level):
    members_for(db, user)
    if area not in {"aggregate", *(a["id"] for a in sections())}:
        raise GroupError("Choose one section or the aggregate to set a target.", 400)
    # No member_id argument: only the authenticated learner's target can move.
    query = db.query(StudyGroupTarget).filter(
        StudyGroupTarget.user_id == user.id, StudyGroupTarget.area == area)
    row = query.first()
    if row is None:
        row = StudyGroupTarget(user_id=user.id, area=area, day=day, level=level)
        db.add(row)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            row = query.first()
            if row is None:
                raise
    row.day, row.level = day, level
    db.commit()
    return {"area": area, "date": day.isoformat(), "level": level}
