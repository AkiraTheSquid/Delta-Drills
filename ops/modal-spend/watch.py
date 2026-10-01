"""watch.py — health checks for modal-spend

The check's job is to be loud truthfully: alert on either limit, never clear
on a failed read, and keep the forced-threshold test path working. These pin
the judging math and the unit wiring without touching Modal or Telegram.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import importlib.util
import os
import sys
from datetime import datetime, timezone

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, '../..'))

SCRIPT = os.path.join(THIS, 'dd_modal_spend_check.py')


def _load():
    spec = importlib.util.spec_from_file_location('dd_modal_spend_check', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_imports():
    """Importable without modal — modal is imported inside read_bill only."""
    mod = _load()
    for name in ('main', 'read_bill', 'judge', 'telegram', 'read_env_file'):
        assert callable(getattr(mod, name, None)), f"dd_modal_spend_check.{name} missing"


def check_public_api():
    """Both limits alert; the early-month floor damps pace; ok is ok."""
    mod = _load()
    mid = datetime(2026, 10, 15, tzinfo=timezone.utc)
    _, reasons = mod.judge({'yesterday_usd': 2.0, 'mtd_usd': 5.0}, mid, 1.5, 30)
    assert len(reasons) == 1 and 'yesterday' in reasons[0], reasons
    _, reasons = mod.judge({'yesterday_usd': 0.5, 'mtd_usd': 20.0}, mid, 1.5, 30)
    assert len(reasons) == 1 and 'projects' in reasons[0], reasons
    _, reasons = mod.judge({'yesterday_usd': 0.5, 'mtd_usd': 5.0}, mid, 1.5, 30)
    assert reasons == [], reasons
    first_morning = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    pace, _ = mod.judge({'yesterday_usd': 0, 'mtd_usd': 1.0}, first_morning, 1.5, 30)
    assert pace < 11, f"one $1 morning projected ${pace:.0f} — the 3-day floor is gone"


def check_invariants():
    """A failed read must not clear, and exit 2 must not read as a failure."""
    with open(SCRIPT, encoding='utf-8') as fh:
        source = fh.read()
    read_at = source.index('bill = read_bill(now)')
    unlink_at = source.index('ALERT_FILE.unlink')
    assert 'return 1' in source[read_at:unlink_at], "billing failure must return before any clear"
    with open(os.path.join(THIS, 'dd-modal-spend-check.service'), encoding='utf-8') as fh:
        unit = fh.read()
    assert 'SuccessExitStatus=0 2' in unit, "exit 2 (alert raised) would mark the unit failed"
    assert 'dd_modal_spend_check.py' in unit


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
