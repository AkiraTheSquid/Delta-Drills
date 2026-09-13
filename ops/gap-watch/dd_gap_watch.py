#!/usr/bin/env python3
"""dd_gap_watch.py - wake the live Claude session when a learner runs OUT of drills.

Why this exists
---------------
The queue never re-serves a question a learner has answered, so a ladder rung
can run dry while the learner is still weak on it. When that happens the app
stops, tells the learner to "ask Claude for more drills", and records the gap
in `/data/user_data/content-gaps.json` on the Fly volume (`app/content_gaps.py`).
Until now the last step was a human one: Seth read the banner, came to a
session, and pasted it in. His words on 2026-09-13:

    "any time I run out of problems, it automatically gets sent to this claude
    discussion for you to specifically create more problems. so instead of
    notifying me, you have a shell with a command that's running that's
    listening"

So this is a poll loop whose stdout is wired to the Monitor tool: every line it
prints lands in the live session as a message, and the session then writes the
drills (`/drill-gaps`). It observes and it interrupts. It decides nothing, edits
nothing, spawns nothing but `flyctl` - the same shape as Delta Note's
`ops/session-watch/`, and deliberately not the removed autofix loop: no new
Claude process, no permission-mode change, the agent it wakes is the attended
session Seth is already sitting in front of.

What fires
----------
One line per (learner, concept, rung) whose `last_seen` has ADVANCED since the
line was last printed - i.e. the learner hit the wall again. A gap printed once
is not printed again for the same hit; a learner who keeps hitting it re-nags
after RENAG_SECONDS. On arming, gaps hit inside ARM_WINDOW_SECONDS are announced
as open now (that is the wall Seth is standing at); older ones are adopted
silently - they are history, and `/drill-gaps` can still read them.

The file is fetched with `flyctl ssh console`, about two seconds a call, every
POLL_SECONDS. A failed fetch is NOT an empty file: the previous view is kept,
and after BLIND_TO_TRIP consecutive failures the watcher says it is blind rather
than staying quiet - silence must never look like "no gaps".

Usage
-----
    python3 -u ops/gap-watch/dd_gap_watch.py            # Fly volume (prod)
    python3 -u ops/gap-watch/dd_gap_watch.py --source local   # local backend

`--once` does a single pass and exits (for tests / a manual look).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE_DIR = Path(os.environ.get("DD_STATE_DIR", Path.home() / ".local/state/delta-drills"))
STATE_PATH = STATE_DIR / "gap-watch.json"   # rebound per --source in main()
LOCK_PATH = STATE_DIR / "gap-watch.lock"


def state_path(source: str) -> Path:
    """One announced-set per source. A `--source local` diagnostic run must not
    record a newer last_seen for a key the Fly watcher then treats as already
    announced - the local Seth account carries the PROD uuid, so keys collide."""
    return STATE_DIR / ("gap-watch.json" if source == "fly" else f"gap-watch-{source}.json")

FLY_APP = "delta-drills-backend"
FLY_GAPS = "/data/user_data/content-gaps.json"
FLYCTL = Path(os.environ.get("FLYCTL", Path.home() / ".fly/bin/flyctl"))
LOCAL_GAPS = REPO / "This-Directory-Only/backend/user_data/content-gaps.json"

POLL_SECONDS = 60.0
FETCH_TIMEOUT = 40
# A learner still hitting the same wall after this long gets re-announced; one
# who moved on does not (their `last_seen` stops advancing).
RENAG_SECONDS = 20 * 60.0
# On arming, a gap this recent is the wall Seth is standing at right now.
ARM_WINDOW_SECONDS = 2 * 3600.0
BLIND_TO_TRIP = 5

# `content_gaps.RUNG_LABEL` is the learner-facing name; this is the ladder name
# used in the lesson frontmatter (`faded:` / `independent:` / `integrated:`).
RUNG = {"worked": "LESSON", "faded": "FADED", "partial": "SOLO", "solo": "INTEGRATED"}


def announce(text: str) -> None:
    print(text, flush=True)


def parse_iso(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch_fly() -> dict | None:
    """The gaps file from the Fly volume, or None if the fetch FAILED.

    None and {} are different answers: {} is "no gaps recorded", None is "could
    not look", and only the first may clear anything.
    """
    env = dict(os.environ)
    if not env.get("FLY_API_TOKEN"):
        cfg = Path.home() / ".fly/config.yml"
        try:
            for line in cfg.read_text().splitlines():
                if "access_token" in line:
                    env["FLY_API_TOKEN"] = line.split(None, 1)[1].strip().strip('"')
        except OSError:
            pass
    try:
        run = subprocess.run(
            [str(FLYCTL), "ssh", "console", "-a", FLY_APP, "-C", f"cat {FLY_GAPS}"],
            capture_output=True, text=True, timeout=FETCH_TIMEOUT, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if run.returncode != 0:
        # `cat` of a file that does not exist yet is "no gaps", not a failure.
        if "No such file" in (run.stderr or "") + (run.stdout or ""):
            return {}
        return None
    return parse_gaps(run.stdout)


def fetch_local(path: Path) -> dict | None:
    if not path.exists():
        return {}
    try:
        return parse_gaps(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def parse_gaps(raw: str) -> dict | None:
    # flyctl prints its "Connecting to ..." banner on stderr, but be defensive:
    # take the first '{' onward so a stray prefix cannot turn a good file into a
    # failed fetch.
    start = raw.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(raw[start:])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def describe(key: str, row: dict) -> str:
    kc = row.get("kc") or key
    title = row.get("kc_title") or ""
    rung = RUNG.get(row.get("stage"), str(row.get("stage") or "?"))
    seen = row.get("rung_size")
    answered = row.get("rung_answered")
    total = row.get("kc_total")
    hits = row.get("hits")
    user = str(row.get("user_id") or "?")[:8]
    when = parse_iso(row.get("last_seen"))
    when_s = when.astimezone().strftime("%H:%M %Z") if when else "?"
    if seen == 0:
        state = f"NOTHING WRITTEN at this rung (concept total {total})"
    elif answered is not None and seen and answered < seen:
        state = f"seen all {seen}, answered {answered} - skipped through, not practised out (concept total {total})"
    else:
        state = f"all {seen} answered (concept total {total})"
    return (
        f"[delta-drills GAP] {kc} — “{title}”: {rung} rung dry: {state}; "
        f"hit {hits}x, last {when_s}, learner {user}. "
        f"Write drills for this concept, every rung → /drill-gaps."
    )


def load_state() -> dict:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def report(gaps: dict, fired: dict, now: datetime, arming: bool) -> list[str]:
    """Decide which gaps to announce. `fired` maps key -> {last_seen, at}; mutated."""
    lines = []
    rows = [(k, r) for k, r in gaps.items() if isinstance(r, dict)]  # {"k": null} is valid JSON
    for key, row in sorted(rows, key=lambda kv: kv[1].get("last_seen") or ""):
        last_seen = parse_iso(row.get("last_seen"))
        if last_seen is None:
            continue
        prev = fired.get(key) or {}
        prev_seen = parse_iso(prev.get("last_seen"))
        prev_at = parse_iso(prev.get("at"))
        if prev_seen is not None and last_seen <= prev_seen:
            continue  # this hit was already announced
        if prev_at is not None and (now - prev_at).total_seconds() < RENAG_SECONDS:
            continue  # same wall, within the re-nag window: one line was enough
        if arming and prev_seen is None and (now - last_seen).total_seconds() > ARM_WINDOW_SECONDS:
            # History. Adopt it quietly; /drill-gaps still lists it.
            fired[key] = {"last_seen": row.get("last_seen"), "at": now.isoformat()}
            continue
        fired[key] = {"last_seen": row.get("last_seen"), "at": now.isoformat()}
        again = " (STILL DRY)" if prev_seen is not None else ""
        lines.append(describe(key, row) + again)
    for key in [k for k in fired if k not in gaps]:
        del fired[key]  # the file was rebuilt without it; forget, don't announce
    return lines


def one_pass(run: dict, fetch, now: datetime) -> None:
    gaps = fetch()
    if gaps is None:
        run["blind"] += 1
        if run["blind"] == BLIND_TO_TRIP:
            announce(f"[delta-drills WATCHER BLIND] {run['blind']} fetches of {FLY_GAPS} failed in a row. "
                     "This is not 'no gaps'. Check flyctl / the Fly machine.")
        return
    if run["blind"] >= BLIND_TO_TRIP:
        announce("[delta-drills] watcher can see the gaps file again.")
    run["blind"] = 0

    arming = not run["armed"]
    lines = report(gaps, run["fired"], now, arming)
    for line in lines:
        announce(line)
    save_state(run["fired"])
    if arming:
        run["armed"] = True
        announce(f"[delta-drills] watching content gaps ({run['source']}) every {POLL_SECONDS:.0f}s; "
                 f"{sum(isinstance(r, dict) for r in gaps.values())} recorded, {len(lines)} open in the last {ARM_WINDOW_SECONDS / 3600:.0f}h announced above.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    ap.add_argument("--source", choices=("fly", "local"), default="fly")
    ap.add_argument("--file", type=Path, default=LOCAL_GAPS, help="gaps file for --source local")
    ap.add_argument("--fresh", action="store_true", help="forget what was already announced")
    args = ap.parse_args()

    global STATE_PATH
    STATE_PATH = state_path(args.source)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # One watcher per machine: two sessions both watching would each write the
    # same drills. "a+" not "w" - "w" truncates before the lock is attempted and
    # a refused second watcher would erase the running one's PID.
    lock = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        announce(f"[delta-drills] another session is already watching gaps ({LOCK_PATH} is held). "
                 "Not starting a second watcher.")
        return 0
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    fetch = fetch_fly if args.source == "fly" else (lambda: fetch_local(args.file))
    run = {"fired": {} if args.fresh else load_state(), "blind": 0, "armed": False, "source": args.source}
    while True:
        one_pass(run, fetch, datetime.now(timezone.utc))
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
