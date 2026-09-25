"""How much of the subscription one learner, and everyone together, may spend.

Any visitor can chat — a guest session is a real token — and on the
openai-oauth door every message is spent from ONE ChatGPT plan that Seth also
codes with. The plan's own limit is the hard wall and nobody sees it coming,
so these two soft walls sit in front of it:

  per learner   CONCEPTUAL_CHAT_PER_USER_HOUR messages in any rolling hour
  everyone      CONCEPTUAL_CHAT_DAILY_CAP messages per UTC day

🔴 IN-PROCESS MEMORY. Correct for the one Fly machine the backend runs on; a
restart forgets the counts (it only ever forgives, never over-blocks), and a
second machine would double both allowances. Move to the database before
scaling out.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone

from app.config import settings

HOUR = 3600.0
# Past this many tracked learners, drop the ones idle for an hour. Keeps the
# map bounded by the people active in the last hour, not everyone ever.
SWEEP_AT = 2000

_lock = threading.Lock()
_per_user: dict[str, deque[float]] = {}
_day = {"date": "", "count": 0}


def check_and_spend(user_id: str, now: float | None = None) -> str | None:
    """Spend one message for `user_id`. Returns a refusal reason, or None."""
    now = time.time() if now is None else now
    today = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
    per_hour = max(0, int(settings.conceptual_chat_per_user_hour))
    daily_cap = max(0, int(settings.conceptual_chat_daily_cap))
    with _lock:
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if per_hour == 0:
            return "The chat is switched off right now."
        if _day["count"] >= daily_cap:
            return "The chat has reached today's limit for everyone. It resets at 00:00 UTC."
        if len(_per_user) > SWEEP_AT:
            for uid in [u for u, s in _per_user.items() if not s or now - s[-1] >= HOUR]:
                del _per_user[uid]
        stamps = _per_user.setdefault(user_id, deque())
        while stamps and now - stamps[0] >= HOUR:
            stamps.popleft()
        if len(stamps) >= per_hour:
            wait_min = max(1, int((HOUR - (now - stamps[0])) // 60) + 1)
            return f"You've sent {per_hour} messages this hour. Try again in about {wait_min} min."
        stamps.append(now)
        _day["count"] += 1
    return None


def reset() -> None:
    """Forget every count. For checks only."""
    with _lock:
        _per_user.clear()
        _day["date"], _day["count"] = "", 0
