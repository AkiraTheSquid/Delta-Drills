"""Per-learner weekly activity — `/api/practice/activity-week`.

The Learner Home's "how much did I practice each day" bar chart (Seth,
2026-09-01: "the amount of problems that you've done across the different
days ... Monday through Sunday ... for the current week"). One learner, one
week, seven counts.

Counts ANSWERED problems from TWO stores, because the app records a worked
question in two different places and neither one alone is "how much did I
practice":

  * **Practice drills** — `kind == attempt` rows in the attempt log with an
    outcome recorded. Looser than `AttemptRow.is_graded` on purpose: is_graded
    also demands a usable feature vector because it guards what the ESTIMATOR
    may consume, but a learner who answered a question worked on it whether or
    not the model can refit from the row. Lesson views and in-flight serves are
    not answers and are not counted.
  * **Placement probes** — `diagnostic.probes` in the learner's state file.
    🔴 THESE WRITE NO ATTEMPT ROW AT ALL. `diagnostic.py` appends the probe to
    its own list and the submit path deliberately creates no attempt during a
    live placement, so a 14-question placement test used to land on this chart
    as a completely empty day. Found 2026-09-01 by Seth ("I did a lot more than
    3 problems"): he had answered 3 probes and 2 drills on prod that day and the
    chart said 2. A probe is a problem he sat down and answered, so it counts.
    `dont_know` counts too — the placement treats it as a response, and a day
    that silently drops the questions he found hardest is the wrong readout.

The two are summed into each day's `count` and also reported separately as
`practice` / `placement`, so the UI can say WHY a day is higher than the number
of drills the learner remembers doing.

Days are the LEARNER'S days. The log stamps UTC; the client sends its own
`Date.getTimezoneOffset()` (minutes, UTC minus local), and both "which week is
current" and each row's bucket are computed in that offset. Without it, every
evening session in the Americas lands on tomorrow's bar.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from app import attempt_log
from app.auth import get_current_user
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter()


def _state_path(user_id: str, base_dir: Optional[Path]) -> Path:
    """The learner's practice-state JSON, beside their attempt log.

    Derived from `attempt_log.log_path` rather than `adaptive._state_file` so a
    caller-supplied `base_dir` reaches BOTH stores — otherwise the tests would
    read a synthetic attempt log and the live state file. Same directory, same
    id sanitisation, `.json` instead of `.attempts.jsonl`.
    """
    log = attempt_log.log_path(user_id, base_dir)
    return log.with_name(log.name[: -len(".attempts.jsonl")] + ".json")


def _probe_timestamps(user_id: str, base_dir: Optional[Path]) -> List[str]:
    """Every placement probe this learner has answered, as raw timestamps.

    A missing or unreadable state file yields nothing rather than raising: the
    practice half of this chart is still honest without it, and a learner who
    has never opened the placement has no diagnostic block at all.
    """
    path = _state_path(user_id, base_dir)
    if not path.exists():
        return []
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("activity_week: unreadable state file %s — probes not counted", path)
        return []
    # `null`, `[]` and `"…"` are all VALID json, so a truncated-then-rewritten
    # state file can decode to something with no .get and take the endpoint out
    # with a 500. Every read below is defensive for the same reason: this chart
    # must never be the thing that breaks the Learner Home.
    if not isinstance(state, dict):
        logger.warning("activity_week: state file %s is not an object — probes not counted", path)
        return []
    diagnostic = state.get("diagnostic")
    if not isinstance(diagnostic, dict):
        return []
    probes = diagnostic.get("probes")
    if not isinstance(probes, list):
        return []
    return [p["ts"] for p in probes if isinstance(p, dict) and p.get("ts")]


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
    practice = {key: 0 for key in day_keys}
    placement = {key: 0 for key in day_keys}

    def bucket(raw_ts, into) -> None:
        """Add one answered question to its LOCAL day, if it is in this week."""
        ts = attempt_log.parse_ts(raw_ts)
        if ts is None:
            return
        key = (ts - offset).date().isoformat()
        if key in into:
            into[key] += 1

    for row in attempt_log.iter_rows(user_id, base_dir):
        if row.kind != attempt_log.KIND_ATTEMPT or row.correct is None:
            continue
        bucket(row.ts, practice)

    for raw_ts in _probe_timestamps(user_id, base_dir):
        bucket(raw_ts, placement)

    days = [
        {
            "date": key,
            "count": practice[key] + placement[key],
            "practice": practice[key],
            "placement": placement[key],
        }
        for key in day_keys
    ]
    return {
        "week_start": day_keys[0],
        "today": local_today.isoformat(),
        "days": days,
        "total": sum(d["count"] for d in days),
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
