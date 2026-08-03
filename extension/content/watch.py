"""watch.py — health checks for content

This script's correctness depends on Colab internals that no test can pin down
from the shell, so these checks guard the parts that CAN be checked: that the
message contract still matches what the panel sends, that the selectors are
still referenced, and that the two rules whose violation produces silent wrong
behaviour are still in place.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "panel")

# Undocumented Colab internals. Losing a reference means every jump silently
# fails, so assert each is still mentioned.
SELECTORS = [
    "colab-scroller#notebook-main",
    "div.cell",
    "md-icon-button.header-section-toggle",
]


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def check_imports():
    script = _read(os.path.join(HERE, "colab.js"))
    assert "chrome.runtime.onMessage.addListener" in script, (
        "colab.js must register a message listener or the panel cannot reach it"
    )


def check_public_api():
    """Every `dd:` message the panel sends has a case here.

    The two files are wired only by string, so a rename on one side fails at
    runtime as an unanswered message rather than as an error.
    """
    script = _read(os.path.join(HERE, "colab.js"))
    handled = set(re.findall(r'case "(dd:[a-z-]+)"', script))
    sent = set()
    for name in ("panel.js", "api.js", "navigate.js"):
        path = os.path.join(PANEL, name)
        if os.path.exists(path):
            sent |= set(re.findall(r'type: "(dd:[a-z-]+)"', _read(path)))
    unhandled = sent - handled
    assert not unhandled, f"panel sends messages colab.js does not handle: {sorted(unhandled)}"


def check_invariants():
    script = _read(os.path.join(HERE, "colab.js"))

    for sel in SELECTORS:
        assert sel in script, f"lost the Colab selector {sel!r}"

    # A cell in a collapsed section has zero height: scrollIntoView succeeds and
    # nothing moves. Expanding first is the whole fix.
    assert "expandAbove" in script, "goto must expand collapsed sections before scrolling"
    assert script.index("expandAbove(cell)") < script.index("await scrollToCell"), (
        "expandAbove must run BEFORE the scroll, not after"
    )

    # The text fallback is the only thing that works on upstream ARENA
    # notebooks, which are nbformat 4.2 with no cell ids.
    assert "if (text)" in script, "findCell must keep its rendered-text fallback"

    # Notebook identity decides whether the panel switches notebooks before a
    # jump. Cell ids are the fragile route (Colab drops them on 4.2 files), so
    # the rendered-text routes have to stay.
    assert "DD_LESSON_ID" in script, (
        "identify must keep reading DD_LESSON_ID — it is the id-independent route"
    )
    assert "dd:dd-lesson-" in script, "identify must keep the lesson comment-marker fallback"

    # A URL carrying #scrollTo= gets stored as the notebook's address and
    # reopens mid-notebook forever after.
    assert 'location.href.split("#")' in script, (
        "identify must report a fragmentless URL — the panel stores it to reopen the notebook"
    )

    # A modal blocks the page and the extension stops receiving messages.
    for banned in ("alert(", "confirm(", "prompt("):
        assert banned not in script, f"{banned} blocks the page and kills the message channel"


def check_focus_cannot_blank_the_notebook():
    """Focus mode hides cells. The ways that goes wrong are all silent.

    * **Nothing is hidden unless a target RESOLVED to real cells.** With no
      match every cell would be tagged out-of-focus and the notebook would go
      blank — and blank is exactly what a failed load looks like. On upstream
      ARENA notebooks (nbformat 4.2, no cell ids at all) no match is the normal
      case, so this is not an edge condition.
    * **The setup cell is never hidden.** It holds the imports and
      `DD_LESSON_ID`; hidden, the answer cell dies on NameError and reads as
      broken starter code rather than as a missing prerequisite.
    * **Group membership needs a digit boundary.** A bare prefix test puts
      `dd-q12` and `dd-q123` in one group, so one problem drags another's cells
      on screen.
    * **The fragment changes without a navigation.** Opening the next problem
      only rewrites `#scrollTo=`, so without a hashchange listener the notebook
      keeps showing the previous problem.
    """
    script = _read(os.path.join(HERE, "colab_focus.js"))

    assert "inFocus > 0" in script, (
        "colab_focus.js must require at least one matching cell before hiding "
        "anything — otherwise an unmatched target blanks the notebook"
    )
    assert "dd-always-visible" in script and "dd-setup" in script, (
        "the setup cell must stay visible in focus mode, or the problem cannot run"
    )
    assert re.search(r"dd-q\(\\d\+\)\(\?:\$\|\[\^0-9\]\)", script), (
        "problemOf must match dd-q<n> with a trailing boundary, or dd-q12 and "
        "dd-q123 land in the same group"
    )
    assert 'addEventListener("hashchange"' in script, (
        "focus must re-apply on hashchange — opening the next problem changes "
        "only the fragment, which is not a navigation"
    )

    # Colab rewrites #scrollTo= to whatever cell is at the top of the viewport,
    # so the fragment is a moving target and reading it straight means scrolling
    # off the problem turns focus off. It has to be a way to CHANGE the target,
    # never a way to clear it. `apply` reads the sticky value, not the raw one.
    assert "const seen = focusTarget();" in script, (
        "apply must read the sticky focusTarget, not targetProblem — Colab "
        "rewrites the fragment while the learner scrolls, so reading it "
        "directly drops focus the moment they scroll past the problem"
    )
    # ...and unconditionally. Observing only when focus is ON lets the sticky
    # value go stale: route with focus off, scroll, re-enable, and the LAST
    # problem comes back instead of the current one.
    assert script.index("const seen = focusTarget();") < script.index("settings.focus ? seen"), (
        "focusTarget must be called before the settings.focus test, not inside "
        "it — the fragment has to be observed even while focus is disabled"
    )
    for banned in ("alert(", "confirm(", "prompt("):
        assert banned not in script, f"{banned} blocks the page and kills the message channel"


def check_css_is_opt_in():
    """Every styling rule is scoped to a class this extension adds.

    An unscoped rule restyles Colab for the student even with both toggles off,
    on every notebook they ever open — including ones that have nothing to do
    with Delta Drills. The toggle would then be a lie, and the only way back
    would be uninstalling.

    The toggle panel itself (`#dd-colab-toggle`) is the deliberate exception: it
    is the way OUT of the theme, so scoping it under the theme would make it
    disappear along with what it disables.

    `html.dd-hide-solutions` is a third scope, and the only one that is ON by
    default — solutions stay hidden until the learner has answered. That makes
    it the one class that could restyle an unrelated notebook, so every rule
    under it must ALSO name `.dd-solution`, which colab_focus.js only ever puts
    on a `dd-q<n>-solution` cell. No generated notebook, no match, no change.
    """
    css = _read(os.path.join(HERE, "colab_dd.css"))
    stripped = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for block in re.finditer(r"([^{}]+)\{[^{}]*\}", stripped):
        selectors = [s.strip() for s in block.group(1).split(",") if s.strip()]
        for selector in selectors:
            scoped = selector.startswith(
                ("html.dd-theme", "html.dd-focus", "html.dd-hide-solutions", "#dd-colab-toggle"))
            assert scoped, (
                f"unscoped CSS rule {selector!r} — it would restyle every Colab "
                f"page even with the toggles off. Scope it under html.dd-theme "
                f"or html.dd-focus"
            )
            if selector.startswith("html.dd-hide-solutions"):
                assert ".dd-solution" in selector, (
                    f"{selector!r} is under the one default-ON scope but does not "
                    f"name .dd-solution — it would restyle notebooks that have "
                    f"nothing to do with Delta Drills"
                )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_focus_cannot_blank_the_notebook, check_css_is_opt_in]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
