"""watch.py — health checks for feedback-widget

The Quackback bubble is a third-party widget on every page, so the checks guard
two things: it can never break boot, and the Widget Secret can never ship to a
browser.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, '..'))
TAG = '<script src="feedback-widget/quackback_widget.js'


def _source():
    with open(os.path.join(HERE, 'quackback_widget.js'), encoding='utf-8') as fh:
        return fh.read()


def check_imports():
    src = _source()
    # A classic script with no imports, wrapped in an IIFE: nothing it declares
    # can collide with app.js's top-level bindings (apiFetch, authToken, ...).
    assert not re.search(r"^\s*(import|export)\s", src, re.M), "quackback_widget.js must stay a classic script"
    assert src.lstrip().startswith('/**') and '(function () {' in src and src.rstrip().endswith('})();'), (
        "quackback_widget.js is no longer wrapped in an IIFE"
    )


def check_public_api():
    with open(os.path.join(SHARED, 'index.html'), encoding='utf-8') as fh:
        html = fh.read()
    at = html.find(TAG)
    assert at != -1, "index.html does not load feedback-widget/quackback_widget.js"
    # It reads window.DDIdentity / window.apiFetch, published by app.js, and must
    # see guest-session.js's guest flag.
    for dep in ('src="app.js', 'src="guest-session.js'):
        dep_at = html.find(dep)
        assert dep_at != -1 and dep_at < at, f"quackback_widget.js must load after {dep[5:]}"


def check_invariants():
    src = _source()
    assert re.search(r"if\s*\(\s*!QUACKBACK_URL\s*\)\s*return", src), (
        "the empty-URL early return is gone; the widget would try to load from ''"
    )
    boot = src[src.find('function boot()'):]
    assert 'try {' in boot and 'catch' in boot, "boot() is no longer wrapped in try/catch"
    # Identity is signed server-side or not at all.
    for forbidden in ('SignJWT', 'jsonwebtoken', 'createHmac', 'crypto.subtle.sign', 'WIDGET_SECRET', 'wgt_'):
        assert forbidden not in src, (
            f"quackback_widget.js mentions {forbidden}. Sign the ssoToken on the "
            f"backend; the Widget Secret must never be in Local_Deployed_Shared/"
        )
    assert "Quackback('logout')" in src, "sign-out no longer calls Quackback('logout')"
    for m in re.finditer(r"Quackback\(\s*'identify'\s*,\s*([^)]*)\)", src):
        assert 'ssoToken' in m.group(1), (
            "Quackback('identify', ...) without an ssoToken. Unverified identity "
            "is rejected by Quackback; sign the token on the backend"
        )
    assert "'/api/quackback/sso'" in src, "the token no longer comes from POST /api/quackback/sso"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
