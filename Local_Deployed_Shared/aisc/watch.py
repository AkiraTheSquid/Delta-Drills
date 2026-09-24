"""watch.py — health checks for aisc

The AISC write-up on #page-learn-about-app: scoped stylesheet + lazy loader.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return f.read()


def check_imports():
    # Every script boot.js loads must exist on disk (a missing one 404s into
    # the SPA rewrite and fails silently in the browser).
    boot = _read("boot.js")
    srcs = re.findall(r'"aisc/([^"?]+)(?:\?[^"]*)?"', boot)
    assert srcs, "boot.js lists no scripts"
    for rel in srcs:
        assert os.path.exists(os.path.join(HERE, rel)), f"boot.js loads missing {rel}"


def check_public_api():
    boot = _read("boot.js")
    assert "DDAboutContentReady" in boot, (
        "boot.js must wait for the About editor's saved-copy swap"
    )
    order = [s for s in re.findall(r'"aisc/js/([^"?]+)', boot)]
    assert order.index("common.js") < min(order.index(f) for f in order if f.startswith("fig-")), (
        "common.js defines AISC and must load before every fig-*.js"
    )


def check_invariants():
    css = _read("aisc.css")
    # Strip comments, then every top-level rule must be scoped to the write-up
    # or this page; a bare `.card {` would restyle the whole app.
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    depth, sel = 0, ""
    for ch in body:
        if ch == "{":
            if depth == 0:
                s = sel.strip()
                assert ("#aisc-root" in s or "#page-learn-about-app" in s
                        or s.startswith("@keyframes")), f"unscoped top-level rule: {s!r}"
            depth += 1
            sel = ""
        elif ch == "}":
            depth -= 1
            sel = ""
        elif depth == 0:
            sel += ch


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
