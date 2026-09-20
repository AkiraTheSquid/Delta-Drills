"""watch.py — health checks for feedback-watch

The watcher's whole job is to interrupt a live session truthfully: a line per
new feedback entry, once, nothing on a failed fetch that could read as "no
feedback", and never a second Claude process. These checks pin those
properties, plus the session-job contract (`SESSION_JOB` names the exact start
command).

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import importlib.util
import os
import re
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, '../..'))

SCRIPT = os.path.join(THIS, 'dd_feedback_watch.py')


def _load():
    spec = importlib.util.spec_from_file_location('dd_feedback_watch', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_imports():
    """The watcher imports on the SYSTEM interpreter — it runs under plain python3."""
    mod = _load()
    for name in ('main', 'one_pass', 'report', 'describe', 'parse_dump', 'parse_log', 'fetch_fly', 'fetch_local'):
        assert callable(getattr(mod, name, None)), f"dd_feedback_watch.{name} missing"


def check_public_api():
    """A failed fetch is None; a dump with headers is a view; an entry fires once."""
    mod = _load()
    assert mod.parse_dump('Connecting to host...\n') is None, "no sentinel at all must be a FAILED fetch"
    assert mod.parse_dump('Connecting...\n=== END\n') == {}, "a scan that finds no logs is an EMPTY view, not a failure"
    cut = 'Connecting...\n=== /data/user_data/u1.feedback.json\n{"entries": [{"question_id": 5, "tag": "broken", "client_id": "a"}]}\n'
    assert mod.parse_dump(cut) is None, "a dump without the END sentinel was cut off: failed fetch"
    dump = cut + '=== /data/user_data/u1.lesson-feedback.json\n{"entries": []}\n=== END\n'
    parsed = mod.parse_dump(dump)
    assert set(parsed) == {'/data/user_data/u1.feedback.json', '/data/user_data/u1.lesson-feedback.json'}, parsed
    assert parsed['/data/user_data/u1.feedback.json'][0]['client_id'] == 'a'
    assert mod.parse_log('garbage') == [], "a corrupt log is an empty one, as the backend reads it"

    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    fresh = {"question_id": 5, "tag": "broken", "client_id": "a", "note": "n",
             "timestamp": (now - timedelta(minutes=1)).isoformat()}
    old = {"question_id": 6, "tag": "good", "client_id": "b",
           "timestamp": (now - timedelta(days=3)).isoformat()}
    logs = {'/x/u1.feedback.json': [fresh, old]}
    seen = {}
    lines = mod.report(logs, seen, now, arming=True)
    assert len(lines) == 1 and 'q5' in lines[0], f"a fresh entry fires on arming, an old one is adopted: {lines}"
    assert mod.report(logs, seen, now, arming=False) == [], "the same entry must not fire twice"
    later = {"question_id": 7, "tag": "unclear", "client_id": "c", "timestamp": now.isoformat()}
    logs['/x/u1.feedback.json'].append(later)
    assert len(mod.report(logs, seen, now, arming=False)) == 1, "a new entry after arming fires"
    lesson = {"kc": "torch.linalg-basics", "lesson_title": "T", "tag": "confusing", "client_id": "d", "timestamp": now.isoformat()}
    line = mod.describe('/x/u1.lesson-feedback.json', lesson)
    assert line.startswith('[delta-drills LESSON FEEDBACK] torch.linalg-basics'), line
    legacy = {"timestamp": "t", "question_id": 1, "tag": "broken"}
    assert mod.entry_key('/x/u1.feedback.json', legacy) != mod.entry_key('/x/u2.feedback.json', legacy), "a legacy key must carry the learner"
    # a restart announces what the on-disk set does not hold, whatever its age
    stale = {"question_id": 9, "tag": "broken", "client_id": "z", "timestamp": (now - timedelta(days=2)).isoformat()}
    kept = dict(seen)
    assert len(mod.report({'/x/u1.feedback.json': [old, stale]}, kept, now, arming=True)) == 1, "a restart must announce an entry the announced set never held"


def check_invariants():
    """No model process, no backend import, session-started only, stdout flushed."""
    src = open(SCRIPT, encoding='utf-8').read()
    for forbidden in ('anthropic', 'ANTHROPIC_' + 'API_KEY', 'subprocess.Popen(["claude"', '"claude"', 'from app'):
        assert forbidden not in src, f"watcher must not carry {forbidden!r}"
    assert 'flush=True' in src, "announce must flush — Monitor reads lines as they are printed"
    assert re.search(r"BLIND_TO_TRIP\s*=\s*\d+", src), "blindness must be announced after N failed fetches"
    job = open(os.path.join(THIS, 'SESSION_JOB'), encoding='utf-8').read()
    assert 'python3 -u ops/feedback-watch/dd_feedback_watch.py' in job, "SESSION_JOB must name the start command"
    assert 'systemd' in job and 'Monitor' in job, "SESSION_JOB must say: session-started, never scheduled"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("PASS feedback-watch")
