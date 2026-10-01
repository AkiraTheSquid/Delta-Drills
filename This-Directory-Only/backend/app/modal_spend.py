"""
What the Modal notebook kernels cost this month, and the ceiling that stops them.

Three numbers, one source of truth each:

  * Month-to-date dollars — Modal's own billing summary for the current UTC
    month (`Workspace.billing.summary`, the same metered cost the dashboard
    shows before the plan's free credits). Fetched by the kernel reaper's
    thread at most once an hour, never on a request: it is a network call.
    The bill trails real use by about an hour, so the last LAG_SECONDS of
    kernel time are priced on top of it from the usage log. That window is
    counted twice rather than missed: for a ceiling, over is the safe side.
  * Kernel-hours — the usage log, `/data/kernel_usage.jsonl` on the Fly
    volume, one line per sandbox when it closes, plus the sandboxes still
    open. Seconds a sandbox existed, not seconds it computed: that is what
    Modal bills.
  * $/kernel-hour — billed dollars over the kernel-hours the bill covers.
    The number that says what one more learner-hour costs.

At or over DD_MODAL_CAP_USD, `modal_kernel` refuses to START a sandbox.
Running ones are left to end through the idle reaper (spec D3), so a learner
mid-cell does not lose their work to the ceiling.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CAP_USD = float(os.environ.get("DD_MODAL_CAP_USD", "25"))
USAGE_LOG = Path(os.environ.get("DD_KERNEL_USAGE_LOG", "/data/kernel_usage.jsonl"))
REFRESH_SECONDS = 3600
LAG_SECONDS = 2 * 3600
# Modal sandbox rates (Sept 2026). Only the fallback when the bill has not
# been read yet, and a floor under the measured rate.
CPU_USD_PER_CORE_HOUR = 0.1419
MEM_USD_PER_GIB_HOUR = 0.024
# Below this many kernel-hours a measured rate is mostly noise.
MIN_RATE_HOURS = 0.5

CAP_MESSAGE = ("Notebook compute is paused for the month — this month's budget for "
               "running notebooks is used up. Lessons and practice still work; "
               "notebooks come back on the 1st.")

_lock = threading.Lock()
_billed: dict = {"usd": None, "fetched": 0.0, "month": ""}
_recorded: set[str] = set()


def _month_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def _month_start(ts: float) -> float:
    d = datetime.fromtimestamp(ts, timezone.utc)
    return datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp()


def request_rate(cpu: float, memory_mb: int) -> float:
    """Dollars per hour a sandbox bills at its requests, before any burst."""
    return cpu * CPU_USD_PER_CORE_HOUR + memory_mb / 1024 * MEM_USD_PER_GIB_HOUR


def record(sandbox_id: str | None, session_id: str, context: str, started: float) -> None:
    """One usage line for a sandbox that just closed. Once per sandbox: a
    session can be shut down from two paths (reaper and a failed cell), and a
    duplicate line would bill its hours twice."""
    if not sandbox_id:
        return
    with _lock:
        if sandbox_id in _recorded:
            return
        _recorded.add(sandbox_id)
    ended = time.time()
    user = session_id[1:] if session_id.startswith("u") and session_id[1:].isdigit() else None
    line = {"user": user, "session_id": session_id, "sandbox_id": sandbox_id,
            "context": context, "started": round(started, 1), "ended": round(ended, 1),
            "seconds": round(ended - started, 1)}
    try:
        with USAGE_LOG.open("a") as fh:
            fh.write(json.dumps(line) + "\n")
    except OSError as exc:
        with _lock:
            _recorded.discard(sandbox_id)  # a later shutdown may retry it
        logger.warning("modal spend: usage log %s not written: %s", USAGE_LOG, exc)


def _logged_spans(since: float) -> list[tuple[float, float]]:
    spans = []
    try:
        with USAGE_LOG.open() as fh:
            for raw in fh:
                try:
                    row = json.loads(raw)
                    started, ended = float(row["started"]), float(row["ended"])
                except (ValueError, KeyError, TypeError):
                    continue
                if ended > since:
                    spans.append((max(started, since), ended))
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("modal spend: usage log %s unreadable: %s", USAGE_LOG, exc)
    return spans


def _overlap(spans, start: float, end: float) -> float:
    return sum(max(0.0, min(e, end) - max(s, start)) for s, e in spans)


def _fetch_month_usd() -> float:
    import modal  # noqa: WPS433 — optional dependency, same as modal_kernel
    summary = modal.Workspace.from_context().billing.summary()
    return float(summary.metered_cost)


def refresh(force: bool = False) -> None:
    """Re-read the month's bill if the cached one is over an hour old or from
    last month. Failure keeps the old number (or none, in a new month) — the
    estimate covers the gap."""
    now = time.time()
    month = _month_key(now)
    with _lock:
        fresh = _billed["month"] == month and now - _billed["fetched"] < REFRESH_SECONDS
    if fresh and not force:
        return
    try:
        usd = _fetch_month_usd()
    except Exception as exc:
        logger.warning("modal spend: billing summary unavailable: %s", exc)
        with _lock:
            if _billed["month"] != month:
                _billed.update(usd=None, fetched=0.0, month=month)
        return
    with _lock:
        _billed.update(usd=usd, fetched=now, month=month)


def summary(live_starts: list[float], rate_floor: float) -> dict:
    """The month so far. `live_starts` are the wall-clock starts of the
    sandboxes open right now; `rate_floor` is what one bills at its requests."""
    now = time.time()
    month_start = _month_start(now)
    spans = _logged_spans(month_start) + [(max(s, month_start), now) for s in live_starts]
    kernel_hours = _overlap(spans, month_start, now) / 3600
    with _lock:
        billed = _billed["usd"] if _billed["month"] == _month_key(now) else None
        fetched = _billed["fetched"]
    usd_per_hour = None
    if billed is not None:
        billed_hours = _overlap(spans, month_start, fetched) / 3600
        if billed_hours >= MIN_RATE_HOURS:
            usd_per_hour = billed / billed_hours
        rate = max(rate_floor, usd_per_hour or 0.0)
        unbilled_hours = _overlap(spans, fetched - LAG_SECONDS, now) / 3600
        mtd = billed + unbilled_hours * rate
    else:
        mtd = kernel_hours * rate_floor
    return {
        "mtd_usd": round(mtd, 2),
        "billed_usd": None if billed is None else round(billed, 2),
        "cap_usd": CAP_USD,
        "kernel_hours_mtd": round(kernel_hours, 2),
        "usd_per_kernel_hour": None if usd_per_hour is None else round(usd_per_hour, 3),
        "capped": mtd >= CAP_USD,
    }


def over_cap(live_starts: list[float], rate_floor: float) -> bool:
    return summary(live_starts, rate_floor)["capped"]
