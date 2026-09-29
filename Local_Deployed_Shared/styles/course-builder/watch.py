"""watch.py — health checks for styles/course-builder

One stylesheet, builder.css, for #page-course-builder (course-builder/).
Checks: it exists and is linked after the courses fragments; it styles the
selectors course-builder.js renders; it uses theme tokens, not hex; and the
narrow layout keeps the graph pane in a real grid row (2026-09-29: the hidden
divider once left it auto-placed into a zero-height third row).
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(os.path.dirname(HERE))
CSS = os.path.join(HERE, "builder.css")


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def check_imports():
    assert os.path.isfile(CSS) and os.path.getsize(CSS) > 0, "builder.css missing or empty"
    html = _read(os.path.join(SHARED, "index.html"))
    mine = html.find('href="styles/course-builder/builder.css?v=')
    assert mine >= 0, "index.html does not link styles/course-builder/builder.css"
    after = html.find('href="styles/courses/responsive.css')
    assert after >= 0 and after < mine, "builder.css must load after styles/courses/responsive.css"


def check_public_api():
    css = _read(CSS)
    for sel in ("#page-course-builder", ".cb-shell", ".cb-split", ".cb-divider", ".cb-view-pane",
                ".cb-graph-host", ".cb-lesson-host", ".cb-card", ".cb-tray", ".cb-chip", ".is-full"):
        assert sel in css, f"builder.css missing {sel}"


def check_invariants():
    css = re.sub(r"/\*.*?\*/", "", _read(CSS), flags=re.DOTALL)
    hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
    assert not hexes, f"builder.css hard-codes colours {hexes[:5]} — use theme tokens"
    m = re.search(r"@media \(max-width: 820px\)\s*\{(.*?)\n\}", css, flags=re.DOTALL)
    assert m, "builder.css lost its ≤820px stacked layout"
    rows = re.findall(r"grid-template-rows:\s*([^;]+);", m.group(1))
    assert rows, "≤820px block sets no grid-template-rows"
    for r in rows:
        tracks = re.findall(r"minmax\([^)]*\)|\S+", r)
        assert len(tracks) == 2, (
            f"≤820px grid-template-rows `{r.strip()}` has {len(tracks)} tracks; the divider is "
            "display:none there, so two tracks — a third puts the graph pane in the wrong row")


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
