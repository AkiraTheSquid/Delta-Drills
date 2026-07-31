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


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
