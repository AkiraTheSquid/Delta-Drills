#!/usr/bin/env python3
"""dd_feedback_watch.py - wake the live Claude session when a learner submits feedback.

Why this exists
---------------
A learner can flag a drill from the practice page ("Broken", "Unclear", "Wrong
image", "Good", plus a note) and a lesson from its reading column ("Wrong",
"Confusing", "Too shallow", "Too verbose", "Good"). Both land in a per-learner
log on the Fly volume (`app/practice/problem_feedback_router.py`:
`<user>.feedback.json` and `<user>.lesson-feedback.json`). Until now the last
hop was a human one: Seth wrote the note, then came to a session and pasted it.
His words on 2026-09-20:

    "set up a running shell on your side that makes it such that whenever I
    submit feedback it automatically sends it to that shell and you get
    notified and it acts as a prompt for you to continue working so that you
    can improve the coding problems or whatever else"

So this is the sibling of `ops/gap-watch/`: a poll loop whose stdout is wired
to the Monitor tool. Every line it prints lands in the live session as a
message; the session reads the note and does the work. It observes and it
interrupts. It decides nothing, edits nothing, spawns nothing but `flyctl`.

What fires
----------
One line per feedback entry not announced before, keyed by the entry's
`client_id` (minted by the browser when the report is written) with a
timestamp+question fallback for older rows. On a first-ever start (no announced
set on disk) entries filed inside ARM_WINDOW_SECONDS are announced (that is
what Seth just wrote) and older ones adopted silently - history. On a restart
every entry the announced set does not hold is announced whatever its age: the
session never saw it.

Both log files of every learner are read in ONE `flyctl ssh console` call per
poll. A failed fetch is NOT an empty log: the previous view is kept, and after
BLIND_TO_TRIP consecutive failures the watcher says it is blind rather than
staying quiet - silence must never look like "no feedback".

Each line carries the drill's own words (topic, subtopic, the first line of
the prompt) from the compiled bank, so the session can act without a lookup.

Usage
-----
    python3 -u ops/feedback-watch/dd_feedback_watch.py                 # Fly volume (prod)
    python3 -u ops/feedback-watch/dd_feedback_watch.py --source local  # local backend

`--once` does a single pass and exits (for tests / a manual look).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE_DIR = Path(os.environ.get("DD_STATE_DIR", Path.home() / ".local/state/delta-drills"))
STATE_PATH = STATE_DIR / "feedback-watch.json"   # rebound per --source in main()
LOCK_PATH = STATE_DIR / "feedback-watch.lock"


def state_path(source: str) -> Path:
    """One announced-set per source, for the same reason as gap-watch: the
    local Seth account carries the PROD uuid, so keys collide across sources."""
    return STATE_DIR / ("feedback-watch.json" if source == "fly" else f"feedback-watch-{source}.json")


FLY_APP = "delta-drills-backend"
FLY_DIR = "/data/user_data"
FLYCTL = Path(os.environ.get("FLYCTL", Path.home() / ".fly/bin/flyctl"))
LOCAL_DIR = REPO / "This-Directory-Only/backend/user_data"
BANK = REPO / "Local_Deployed_Shared/questions_structured.json"

# One shell loop on the machine prints every log with a header line, so one
# ssh round-trip covers every learner and both kinds of feedback.
# The loop ends with a sentinel header so an EMPTY directory (no logs yet) is
# a successful scan and not a failed fetch (codex, 2026-09-20).
END = "=== END"
REMOTE_CMD = (
    f"sh -c 'for f in {FLY_DIR}/*.feedback.json {FLY_DIR}/*.lesson-feedback.json; "
    "do [ -f \"$f\" ] || continue; echo \"=== $f\"; cat \"$f\"; echo; done; "
    f"echo \"{END}\"'"
)

POLL_SECONDS = 60.0
FETCH_TIMEOUT = 40
ARM_WINDOW_SECONDS = 2 * 3600.0
BLIND_TO_TRIP = 5
NOTE_MAX = 400

_bank: dict | None = None


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


def fly_env() -> dict:
    env = dict(os.environ)
    if not env.get("FLY_API_TOKEN"):
        cfg = Path.home() / ".fly/config.yml"
        try:
            for line in cfg.read_text().splitlines():
                if "access_token" in line:
                    env["FLY_API_TOKEN"] = line.split(None, 1)[1].strip().strip('"')
        except OSError:
            pass
    return env


def fetch_fly() -> dict | None:
    """{log path: entries} from the Fly volume, or None if the fetch FAILED.

    None and {} are different answers: {} is "no logs yet", None is "could not
    look", and only the first may be treated as a view of the world.
    """
    try:
        run = subprocess.run(
            [str(FLYCTL), "ssh", "console", "-a", FLY_APP, "-C", REMOTE_CMD],
            capture_output=True, text=True, timeout=FETCH_TIMEOUT, env=fly_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if run.returncode != 0:
        return None
    return parse_dump(run.stdout)


def fetch_local(directory: Path) -> dict | None:
    if not directory.exists():
        return {}
    out = {}
    try:
        for p in sorted(directory.iterdir()):
            if p.name.endswith(".feedback.json") or p.name.endswith(".lesson-feedback.json"):
                out[str(p)] = parse_log(p.read_text(encoding="utf-8"))
    except OSError:
        return None
    return out


def parse_log(raw: str) -> list:
    """The entries of one log file; a corrupt file is an empty one here (the
    backend treats it the same way), never a failed fetch."""
    start = raw.find("{")
    if start < 0:
        return []
    try:
        data = json.loads(raw[start:])
    except ValueError:
        return []
    entries = data.get("entries") if isinstance(data, dict) else None
    return [e for e in (entries or []) if isinstance(e, dict)]


def parse_dump(raw: str) -> dict | None:
    """Split the remote loop's output on its `=== <path>` header lines.

    The `=== END` sentinel is what says the loop ran to the end; without it
    (a stray flyctl banner, a dump cut off mid-file) the fetch FAILED. With it
    and no other header, the directory holds no logs yet: a real, empty view.
    """
    if not re.search(r"(?m)^=== END\s*$", raw):
        return None  # the loop never finished: banner only, or a cut-off dump
    parts = re.split(r"(?m)^=== ", raw)
    out = {}
    for chunk in parts[1:]:
        path, _, body = chunk.partition("\n")
        path = path.strip()
        if path == "END":
            continue
        out[path] = parse_log(body)
    return out


def bank_lookup(qid) -> dict:
    global _bank
    if _bank is None:
        _bank = {}
        try:
            for q in json.loads(BANK.read_text(encoding="utf-8")):
                _bank[int(q["id"])] = q
        except (OSError, ValueError, KeyError, TypeError):
            _bank = {}
    try:
        return _bank.get(int(qid)) or {}
    except (TypeError, ValueError):
        return {}


def entry_key(path: str, entry: dict) -> str:
    cid = entry.get("client_id")
    if cid:
        return f"c:{cid}"
    # Rows from before 2026-08-27 carry no client id: key them by learner (the
    # log's file name) as well, or two learners' same-second reports collapse.
    user = Path(path).name.split(".")[0]
    return f"t:{user}|{entry.get('timestamp')}|{entry.get('question_id')}|{entry.get('kc')}|{entry.get('tag')}"


def describe(path: str, entry: dict) -> str:
    user = Path(path).name.split(".")[0][:8]
    lesson = path.endswith(".lesson-feedback.json")
    when = parse_iso(entry.get("timestamp"))
    if when is None:
        when_s = "?"
    else:
        local = when.astimezone()
        # A restart can announce an old row: say the day, or "18:29" reads as today.
        fmt = "%H:%M %Z" if local.date() == datetime.now(local.tzinfo).date() else "%Y-%m-%d %H:%M %Z"
        when_s = local.strftime(fmt)
    tag = entry.get("tag") or "?"
    note = " ".join(str(entry.get("note") or "").split())
    if len(note) > NOTE_MAX:
        note = note[:NOTE_MAX] + "…"
    note_s = f' — "{note}"' if note else " — (no note)"
    if lesson:
        kc = entry.get("kc") or "?"
        title = entry.get("lesson_title") or ""
        gate = entry.get("question_id")
        gate_s = f", in front of q{gate}" if gate else ""
        return (f"[delta-drills LESSON FEEDBACK] {kc} “{title}” tagged {tag}{gate_s}"
                f"{note_s}; learner {user}, {when_s}. Fix the lesson page for this concept.")
    qid = entry.get("question_id")
    q = bank_lookup(qid)
    cur = q.get("curriculum") or {}
    ex = q.get("exercise") or {}
    where = f" ({cur.get('topic')}: {cur.get('subtopic')})" if cur.get("topic") else ""
    prompt = " ".join(str(ex.get("question_text") or "").split())[:120]
    prompt_s = f' [{prompt}]' if prompt else ""
    correct = entry.get("correct")
    outcome = {True: "answered right", False: "answered wrong", None: "not answered"}.get(correct, "?")
    return (f"[delta-drills FEEDBACK] q{qid}{where} tagged {tag}, {outcome}{note_s}{prompt_s}; "
            f"learner {user}, {when_s}. Act on it: fix the drill, its near-miss or its lesson.")


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


def report(logs: dict, seen: dict, now: datetime, arming: bool) -> list[str]:
    """Decide which entries to announce. `seen` maps entry key -> ISO time it
    was announced (or adopted); mutated."""
    lines = []
    rows = []
    for path, entries in logs.items():
        for entry in entries:
            rows.append((path, entry))
    rows.sort(key=lambda pe: pe[1].get("timestamp") or "")
    # A first-ever start (no announced set on disk) adopts anything older than
    # the arm window as history. A RESTART has a record of what was announced,
    # so every entry missing from it is one the session never saw — announce
    # it whatever its age, or feedback filed while the watcher was down for
    # more than two hours is lost (codex, 2026-09-20).
    cold_start = arming and not seen
    for path, entry in rows:
        key = entry_key(path, entry)
        if key in seen:
            continue
        when = parse_iso(entry.get("timestamp"))
        seen[key] = now.isoformat()
        if cold_start and (when is None or (now - when).total_seconds() > ARM_WINDOW_SECONDS):
            continue  # history: adopt quietly
        lines.append(describe(path, entry))
    return lines


def one_pass(run: dict, fetch, now: datetime) -> None:
    logs = fetch()
    if logs is None:
        run["blind"] += 1
        if run["blind"] == BLIND_TO_TRIP:
            announce(f"[delta-drills WATCHER BLIND] {run['blind']} fetches of the feedback logs failed in a row. "
                     "This is not 'no feedback'. Check flyctl / the Fly machine.")
        return
    if run["blind"] >= BLIND_TO_TRIP:
        announce("[delta-drills] watcher can see the feedback logs again.")
    run["blind"] = 0

    arming = not run["armed"]
    lines = report(logs, run["seen"], now, arming)
    for line in lines:
        announce(line)
    save_state(run["seen"])
    if arming:
        run["armed"] = True
        total = sum(len(v) for v in logs.values())
        announce(f"[delta-drills] watching learner feedback ({run['source']}) every {POLL_SECONDS:.0f}s; "
                 f"{total} entries in {len(logs)} logs, {len(lines)} from the last {ARM_WINDOW_SECONDS / 3600:.0f}h announced above.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    ap.add_argument("--source", choices=("fly", "local"), default="fly")
    ap.add_argument("--dir", type=Path, default=LOCAL_DIR, help="user_data dir for --source local")
    ap.add_argument("--fresh", action="store_true", help="forget what was already announced")
    args = ap.parse_args()

    global STATE_PATH
    STATE_PATH = state_path(args.source)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        announce(f"[delta-drills] another session is already watching feedback ({LOCK_PATH} is held). "
                 "Not starting a second watcher.")
        return 0
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    fetch = fetch_fly if args.source == "fly" else (lambda: fetch_local(args.dir))
    run = {"seen": {} if args.fresh else load_state(), "blind": 0, "armed": False, "source": args.source}
    while True:
        one_pass(run, fetch, datetime.now(timezone.utc))
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
