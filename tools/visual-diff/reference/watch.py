"""watch.py — health checks for visual-diff/reference

The mockup is the control: our content under their design. If it stops being a
faithful control, every comparison made with it is worth less than it looks.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(TOOL))


def check_imports():
    for name in ("mockup.html", "mockup.js", "mockup.css"):
        assert os.path.exists(os.path.join(HERE, name)), f"{name} is missing"

    # 🔴 A MISSING `node` RAISES, IT DOES NOT RETURN 127. subprocess.run finds
    # no executable and throws FileNotFoundError from Python, so the 127 branch
    # below never ran and a machine without node failed a check that is meant to
    # be optional. Found by codex, 2026-09-02.
    try:
        node = subprocess.run(["node", "--check", os.path.join(HERE, "mockup.js")],
                              capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return  # no node here; the syntax check is a bonus, not a requirement
    if node.returncode == 127 or "not found" in (node.stderr or ""):
        return
    assert node.returncode == 0, f"mockup.js does not parse: {node.stderr.strip()[:300]}"


def check_public_api():
    """🔴 THE MOCKUP USES THE APP'S CLASS NAMES. One role map in targets.json has
    to serve both it and the app, so a renamed class here silently empties half
    the comparison. The rail's classes now come from the app's own stylesheet
    and module, which this page links — so those two files count as sources for
    a name as much as the mockup's own do."""
    css = open(os.path.join(HERE, "mockup.css"), encoding="utf-8").read()
    html = open(os.path.join(HERE, "mockup.html"), encoding="utf-8").read()
    js = open(os.path.join(HERE, "mockup.js"), encoding="utf-8").read()
    shared = os.path.join(REPO, "Local_Deployed_Shared")
    rail_css = open(os.path.join(shared, "styles", "practice", "arena-notebook-nav.css"), encoding="utf-8").read()
    rail_js = open(os.path.join(shared, "practice", "arena-notebook-nav.js"), encoding="utf-8").read()
    markup = html + js + css + rail_css + rail_js

    targets = json.load(open(os.path.join(TOOL, "targets.json")))
    for role, selectors in targets["mockup"]["roles"].items():
        classes = re.findall(r"\.([a-z][\w-]+)", " ".join(selectors))
        for cls in classes:
            assert cls in markup, f"role {role} names .{cls}, which the mockup never uses"


def check_invariants():
    """No third-party source under this folder. ForumMagnum is GPL-3.0 and is
    read from a checkout OUTSIDE the repo."""
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ("ForumMagnum", "node_modules")]
        for name in files:
            assert not name.endswith((".tsx", ".ts")), f"their source must not live here: {name}"

    # 🔴 ONE RAIL, ONE OWNER. This page carried a second copy of the app's rail
    # until 2026-09-02, and that copy went stale the moment the app's was
    # restructured — every comparison run through here in between was measuring
    # a design nobody shipped. The page must LINK the app's rail, never restate
    # it.
    css = open(os.path.join(HERE, "mockup.css"), encoding="utf-8").read()
    js = open(os.path.join(HERE, "mockup.js"), encoding="utf-8").read()
    html = open(os.path.join(HERE, "mockup.html"), encoding="utf-8").read()
    for name, body in (("mockup.css", css), ("mockup.js", js)):
        assert ".anb-toc-row" not in body and ".anb-toc-dot" not in body, (
            f"{name} has started restating the app's rail instead of linking it"
        )
    assert "styles/practice/arena-notebook-nav.css" in html, (
        "the mockup no longer links the app's rail stylesheet"
    )
    assert "practice/arena-notebook-nav.js" in html and "ArenaNotebookNav" in js, (
        "the mockup no longer mounts the app's rail"
    )

    # It renders OUR notebooks, so it has to be able to find them.
    js = open(os.path.join(HERE, "mockup.js"), encoding="utf-8").read()
    path = re.search(r"lessons/notebooks/\$\{slug\}\.json", js)
    assert path, "the mockup no longer fetches a compiled notebook"
    assert os.path.isdir(os.path.join(REPO, "Local_Deployed_Shared", "lessons", "notebooks")), (
        "the compiled notebooks the mockup renders are missing"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS visual-diff/reference ({len(checks)} checks)")
