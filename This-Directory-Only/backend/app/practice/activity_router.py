"""Per-learner weekly activity — `/api/practice/activity-week`.

The Learner Home's "how much did I practice each day" bar chart (Seth,
2026-09-01: "the amount of problems that you've done across the different
days ... Monday through Sunday ... for the current week"). One learner, one
week, seven counts.

Counts ANSWERED problems: `kind == attempt` rows with an outcome recorded.
That is looser than `AttemptRow.is_graded` on purpose — is_graded also demands
a usable feature vector because it guards what the ESTIMATOR may consume, but
a learner who answered a question worked on it whether or not the model can
refit from the row. Lesson views and in-flight serves are not answers and are
not counted.

Days are the LEARNER'S days. The log stamps UTC; the client sends its own
`Date.getTimezoneOffset()` (minutes, UTC minus local), and both "which week is
current" and each row's bucket are computed in that offset. Without it, every
evening session in the Americas lands on tomorrow's bar.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from app import attempt_log
from app.auth import get_current_user
from app.models import User

router = APIRouter()


def week_counts(
    user_id: str,
    tz_offset_minutes: int = 0,
    now: Optional[datetime] = None,
    base_dir: Optional[Path] = None,
) -> dict:
    """Answered-problem counts for the learner's current Monday-Sunday week.

    Split from the route so the test suite can feed a synthetic log and a
    frozen `now` without standing up auth.
    """
    now_utc = now or datetime.now(timezone.utc)
    offset = timedelta(minutes=tz_offset_minutes)
    local_today: date = (now_utc - offset).date()
    monday = local_today - timedelta(days=local_today.weekday())
    day_keys = [(monday + timedelta(days=i)).isoformat() for i in range(7)]
    counts = {key: 0 for key in day_keys}

    for row in attempt_log.iter_rows(user_id, base_dir):
        if row.kind != attempt_log.KIND_ATTEMPT or row.correct is None:
            continue
        ts = attempt_log.parse_ts(row.ts)
        if ts is None:
            continue
        key = (ts - offset).date().isoformat()
        if key in counts:
            counts[key] += 1

    return {
        "week_start": day_keys[0],
        "today": local_today.isoformat(),
        "days": [{"date": key, "count": counts[key]} for key in day_keys],
        "total": sum(counts.values()),
    }


@router.get("/activity-week")
def activity_week(
    # ±16h covers every real offset (UTC-12 to UTC+14); anything wilder is a
    # bogus client and gets a 422 rather than a week nobody lives in.
    tz_offset: int = Query(0, ge=-960, le=960),
    user: User = Depends(get_current_user),
) -> dict:
    """This learner's Monday-Sunday answered-problem counts, in their timezone."""
    return week_counts(str(user.id), tz_offset)
