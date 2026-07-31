"""watch.py — health checks for panel

The panel fails in ways a syntax check does not see: a view declared in
`panel.html` but missing from `VIEWS`, an element id referenced by `panel.js`
that no longer exists, a timer path that skips `clearTimers()`. These check the
wiring between the three files.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


# Load order is a hard dependency chain, not a style choice: each file
# destructures the previous one's global at its top level, so a wrong order is
# an immediate TypeError rather than a subtle bug.
ORDER = ["notebook-index.js", "api.js", "navigate.js", "panel.js"]


def check_imports():
    """All files present, and loaded in the order the panel needs."""
    for name in ("panel.html", "panel.css", *ORDER):
        assert os.path.exists(os.path.join(HERE, name)), (
            f"missing {name}"
            + (
                " — run scripts/generate_colab_notebooks.py"
                if name == "notebook-index.js"
                else ""
            )
        )
    markup = _read("panel.html")
    seen = []
    for name in ORDER:
        at = markup.find(f'src="{name}"')
        assert at != -1, f"panel.html must load {name}"
        seen.append((at, name))
    assert seen == sorted(seen), (
        "panel.html loads scripts out of order; required: " + " → ".join(ORDER)
    )

    index = _read("notebook-index.js")
    assert "window.DD_NOTEBOOKS" in index, "notebook-index.js must define window.DD_NOTEBOOKS"


def check_public_api():
    """Every element `panel.js` reaches for by id exists in `panel.html`.

    A renamed id fails at runtime as a null dereference inside an event
    handler, which is easy to miss because the panel still renders.
    """
    markup = _read("panel.html")
    script = _read("panel.js")
    nav = _read("navigate.js")
    present = set(re.findall(r'id="([^"]+)"', markup))
    wanted = set(re.findall(r'\$\("([^"]+)"\)', script)) | set(
        re.findall(r'\bel\("([^"]+)"\)', nav)
    )
    # $(`view-${v}`) is built from VIEWS; checked separately below.
    missing = {w for w in wanted if w not in present}
    assert not missing, f"panel references ids not in panel.html: {sorted(missing)}"

    # paintNotebook builds these from a `${prefix}` template, so the loop above
    # cannot see them. Both view families must carry the full pair or the
    # notebook row silently null-dereferences on render.
    for prefix in ("p", "gate"):
        for part in ("nb-title", "nb-state"):
            assert f"{prefix}-{part}" in present, f"panel.html is missing #{prefix}-{part}"

    views = set(re.findall(r"const VIEWS = \[([^\]]+)\]", script)[0].replace('"', "").split(", "))
    declared = {i[len("view-"):] for i in present if i.startswith("view-")}
    assert views == declared, (
        f"VIEWS and the view-* sections disagree: "
        f"only in VIEWS {sorted(views - declared)}, only in HTML {sorted(declared - views)}"
    )


def check_invariants():
    """The two rules that cause wrong mastery data rather than a visible crash."""
    script = _read("panel.js")

    # A double submit logs two ladder attempts for one problem.
    assert "state.graded" in script, "grade() must be guarded by state.graded"
    grade_body = script[script.index("async function grade("):]
    assert "if (state.graded" in grade_body[:200], (
        "the state.graded guard must be the FIRST thing grade() does"
    )

    # A leaked interval auto-grades a problem the student already left.
    assert script.count("clearTimers()") >= 4, (
        "every view transition must clearTimers(); found too few calls"
    )

    api = _read("api.js")
    assert "DOMContentLoaded" not in api and "document." not in api, (
        "api.js must not touch the DOM — it is the backend contract, nothing else"
    )
    for name in ("panel.js", "navigate.js"):
        assert "fetch(" not in _read(name), f"{name} must not call fetch; go through api.js"

    # A cross-notebook jump that does not wait for the new notebook to mount
    # scrolls for a cell that does not exist yet and reports "not found",
    # which reads as a missing anchor rather than as a slow page.
    nav = _read("navigate.js")
    assert "waitForNotebook" in nav, "ensureNotebook must wait for the new notebook to mount"
    assert nav.index("navTab.navigate(") < nav.index("waitForNotebook("), (
        "navigate before waiting, not after"
    )

    # One hardcoded Colab URL and the repo/remembered-URL resolution becomes
    # advisory — the panel would send the student somewhere the settings say
    # nothing about.
    for name in ("panel.js", "navigate.js"):
        assert "colab.research.google.com" not in _read(name), (
            f"{name} must build Colab URLs through notebooks.urlFor, not inline"
        )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
