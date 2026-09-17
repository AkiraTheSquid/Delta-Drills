"""watch.py — health checks for gap-watch

The watcher's whole job is to interrupt a live session truthfully: a line per
new gap, nothing on a failed fetch that could read as "no gaps", and never a
second Claude process. These checks pin those three properties, plus the
session-job contract (`SESSION_JOB` names the exact start command).

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import importlib.util
import os
import re
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, '../..'))

SCRIPT = os.path.join(THIS, 'dd_gap_watch.py')


def _load():
    spec = importlib.util.spec_from_file_location('dd_gap_watch', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_imports():
    """The watcher imports on the SYSTEM interpreter — it runs under plain python3."""
    mod = _load()
    for name in ('main', 'one_pass', 'report', 'describe', 'parse_gaps', 'fetch_fly', 'fetch_local'):
        assert callable(getattr(mod, name, None)), f"dd_gap_watch.{name} missing"


def check_public_api():
    """A failed fetch is None, an empty file is {} — and only {} may clear anything."""
    mod = _load()
    assert mod.parse_gaps('garbage') is None, "unparseable text must be a FAILED fetch, not an empty file"
    assert mod.parse_gaps('Connecting to host...\n{"k": {"kc": "x"}}') == {"k": {"kc": "x"}}, (
        "a banner before the JSON must not turn a good file into a failed fetch"
    )
    assert mod.parse_gaps('[]') is None, "a non-object file is not a gaps file"

    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 9, 13, tzinfo=timezone.utc)
    row = {"kc": "k", "stage": "partial", "rung_size": 1, "rung_answered": 1,
           "kc_total": 2, "hits": 1, "user_id": "u", "last_seen": (now - timedelta(minutes=1)).isoformat()}
    fired = {}
    assert len(mod.report({"a": row}, fired, now, arming=True)) == 1, "a fresh gap must fire on arming"
    assert mod.report({"a": row}, fired, now, arming=False) == [], "the same hit must not fire twice"
    old = dict(row, last_seen=(now - timedelta(days=1)).isoformat())
    assert mod.report({"b": old}, {}, now, arming=True) == [], "history is adopted silently on arming"
    assert mod.report({"bad": None, "c": row}, {}, now, arming=True) == [mod.describe("c", row)], (
        "a null row is valid JSON - skip it, never let it kill the persistent watcher"
    )
    assert mod.state_path("fly") != mod.state_path("local"), (
        "fly and local runs must not share the announced-set (a local --once would silence prod)"
    )


def check_invariants():
    """Observes and interrupts — never spawns a Claude, never lies with silence."""
    with open(SCRIPT, encoding='utf-8') as fh:
        src = fh.read()
    for banned in ('bypassPermissions', 'Popen(', 'os.system(', '"claude"', "'claude'"):
        assert banned not in src, (
            f"dd_gap_watch.py contains {banned!r} — this watcher must stay a reader that "
            "interrupts the attended session, not a launcher (Delta Note removed ops/autofix for this)"
        )
    assert re.search(r'open\(LOCK_PATH,\s*"a\+"\)', src), (
        'the lock must be opened "a+": "w" truncates before flock and a refused second '
        "watcher erases the running one's PID"
    )
    assert 'flush=True' in src, "stdout must flush per line or Monitor sees nothing until exit"
    assert 'if gaps is None' in src and 'BLIND' in src, (
        "a failed fetch must be counted and announced — silence must never read as 'no gaps'"
    )
    job = os.path.join(THIS, 'SESSION_JOB')
    assert os.path.exists(job), "SESSION_JOB marker missing — a scheduled unit would print to nobody"
    with open(job, encoding='utf-8') as fh:
        assert 'python3 -u ops/gap-watch/dd_gap_watch.py' in fh.read(), (
            "SESSION_JOB must carry the exact start command, with -u"
        )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
