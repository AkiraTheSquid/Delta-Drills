"""watch.py — health checks for aisc/js (the write-up's figures)

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "..", "..", "index.html")


def _js():
    out = {}
    for p in glob.glob(os.path.join(HERE, "*.js")):
        with open(p, encoding="utf-8") as f:
            out[os.path.basename(p)] = f.read()
    return out


def check_imports():
    js = _js()
    for name in ("engine.js", "common.js", "fig-effort.js", "fig-forget.js",
                 "fig-graphs.js", "fig-seth.js"):
        assert name in js, f"{name} missing"


def check_public_api():
    js = _js()
    assert "root.AISC =" in js["common.js"], "common.js no longer exports AISC"
    assert "delta:theme-changed" in js["common.js"], (
        "common.js must repaint figures on the app's theme event"
    )
    for name, src in js.items():
        assert not re.search(r"\bDD\.", src), f"{name} still uses the old DD global"


def check_invariants():
    # Every id a script looks up must exist in index.html with the prefix.
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()
    for name, src in _js().items():
        for i in re.findall(r'(?:getElementById|\$)\("([^"]+)"\)', src):
            assert i.startswith("aisc-"), f"{name} looks up unprefixed #{i}"
            assert f'id="{i}"' in html, f"{name} looks up #{i}, not in index.html"
        for p in re.findall(r'fetch\("([^"]+)"', src):
            assert p.startswith("aisc/data/"), f"{name} fetches {p} outside aisc/data/"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
