"""watch.py — health checks for conceptual

Pins the three things that fail silently rather than loudly here: the route
being mounted (a rename is a 404, not an import error), the endpoint needing
a signed-in (or guest) token (without it anyone with curl spends the ChatGPT
subscription), and no sampling controls reaching the openai-oauth door (the
Codex backend rejects them, so every message would error).

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
# conceptual/ lives at backend/app/conceptual/, so backend/ is two levels up.
sys.path.insert(0, os.path.join(THIS, '..', '..'))

# `mod watch` runs system python, which has no fastapi. Re-exec under the
# backend venv so the checks RUN instead of skipping.
_VENV_PY = os.path.join(THIS, '..', '..', '.venv', 'bin', 'python3')
try:
    import fastapi  # noqa: F401
except ImportError:
    # 🔴 NOT realpath: the venv's python3 symlinks to /usr/bin/python3, and it
    # is the venv PATH (pyvenv.cfg beside it) that selects site-packages.
    if os.path.exists(_VENV_PY) and not os.environ.get('CONCEPTUAL_WATCH_REEXEC'):
        os.environ['CONCEPTUAL_WATCH_REEXEC'] = '1'
        os.execv(_VENV_PY, [_VENV_PY, os.path.abspath(__file__)])
    raise


def _src(name):
    with open(os.path.join(THIS, name), encoding='utf-8') as f:
        return f.read()


def check_imports():
    from app.conceptual import router  # noqa: F401
    from app.conceptual import limits, prompts, provider  # noqa: F401


def check_public_api():
    from app.main import app
    from app.auth import get_current_user
    route = next((r for r in app.routes if getattr(r, 'path', '') == '/api/conceptual/chat'), None)
    assert route is not None, 'POST /api/conceptual/chat is not mounted in main.py'
    # The MOUNTED route's real dependency graph, not a grep of the source.
    deps, seen = list(route.dependant.dependencies), set()
    while deps:
        d = deps.pop()
        seen.add(d.call)
        deps.extend(d.dependencies)
    assert get_current_user in seen, \
        'conceptual chat must require a token: it spends a shared subscription'


def check_invariants():
    # No sampling controls on either door (openai-oauth rejects them).
    for name in ('router.py', 'provider.py'):
        assert not re.search(r'\b(temperature|top_p)\s*=', _src(name)), \
            f'{name} passes temperature/top_p; the openai-oauth door rejects them'
    # A valid body is bounded (not a memory guard: see router.py).
    from pydantic import ValidationError
    from app.conceptual.router import ChatRequest, MAX_BODY_CHARS, MAX_BODY_MESSAGES
    for bad in ({'messages': [{'role': 'user', 'content': 'x'}] * (MAX_BODY_MESSAGES + 1)},
                {'messages': [{'role': 'user', 'content': 'x' * (MAX_BODY_CHARS + 1)}]},
                {'messages': [], 'concept': 'c' * 201}):
        try:
            ChatRequest(**bad)
        except ValidationError:
            continue
        raise AssertionError('ChatRequest accepted an oversized body')
    # Deep Chat renders $ / $$ only.
    from app.conceptual.prompts import build_system_prompt
    assert '$$' in build_system_prompt(), 'system prompt must ask for $ / $$ maths'
    # Limits: per-user hour and global day both bite, and a new day forgives.
    from app.conceptual import limits
    from app.config import settings
    saved = (settings.conceptual_chat_per_user_hour, settings.conceptual_chat_daily_cap)
    try:
        settings.conceptual_chat_per_user_hour, settings.conceptual_chat_daily_cap = 2, 3
        limits.reset()
        t = 1_700_000_000.0
        assert limits.check_and_spend('a', t) is None
        assert limits.check_and_spend('a', t + 1) is None
        assert limits.check_and_spend('a', t + 2), 'per-user hour limit did not bite'
        assert limits.check_and_spend('a', t + 3601) is None, 'hour window did not roll'
        assert limits.check_and_spend('b', t + 3602), 'daily cap did not bite'
        assert limits.check_and_spend('b', t + 86400) is None, 'new UTC day did not reset'
        # A zero hourly allowance is "off", not a crash on an empty window.
        settings.conceptual_chat_per_user_hour = 0
        assert limits.check_and_spend('zero', t + 86400), 'per_hour=0 must refuse'
        settings.conceptual_chat_per_user_hour = 2
        # The per-learner map stays bounded: idle learners are swept.
        saved_sweep, limits.SWEEP_AT = limits.SWEEP_AT, 5
        settings.conceptual_chat_daily_cap = 100
        for i in range(10):
            limits.check_and_spend(f'idle{i}', t + 86400)
        limits.check_and_spend('fresh', t + 86400 + 3601)
        limits.SWEEP_AT = saved_sweep
        assert len(limits._per_user) <= 6, f'idle learners never swept ({len(limits._per_user)} kept)'
    finally:
        settings.conceptual_chat_per_user_hour, settings.conceptual_chat_daily_cap = saved
        limits.reset()


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
