"""watch.py — health checks for styles/practice

Practice-tab stylesheets (split from the former practice.css monolith).
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(os.path.dirname(HERE))

CSS_FILES = ["layout.css", "timer.css", "question.css", "feedback.css", "result.css", "misc.css"]


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Import checks ──────────────────────────────
# Pure CSS folder — "imports" means every split file exists, is non-empty,
# and index.html links each one (a missing <link> silently unstyles a slice
# of the practice tab).
def check_imports():
    missing = [f for f in CSS_FILES if not os.path.isfile(os.path.join(HERE, f))]
    assert not missing, f"missing stylesheet files: {missing}"
    empty = [f for f in CSS_FILES if os.path.getsize(os.path.join(HERE, f)) == 0]
    assert not empty, f"empty stylesheet files: {empty}"
    index_html = _read(os.path.join(SHARED, "index.html"))
    unlinked = [f for f in CSS_FILES if f'href="styles/practice/{f}' not in index_html]
    assert not unlinked, f"index.html missing <link> for: {unlinked}"
    # The old monolith must stay dead — a resurrected practice.css would load
    # after these files and silently win every cascade tie.
    assert not os.path.isfile(os.path.join(SHARED, "practice.css")), (
        "practice.css monolith is back — it was split into styles/practice/ (2026-07-12)"
    )
    assert 'href="practice.css' not in index_html, "index.html still links the removed practice.css"


# ── Public API checks ─────────────────────────
# The selectors practice/*.js toggles at runtime must keep existing in the
# right file — renaming one silently unstyles the element.
def check_public_api():
    expected = {
        "layout.css": [".practice-container", ".practice-split", ".practice-left"],
        "timer.css": [".session-setup", ".session-status-row", ".session-countdown", ".timer-input", "session-idle"],
        "question.css": [".question-text", ".question-imports", ".question-visual", ".cold-start-badge"],
        "feedback.css": [".result-badge", ".feedback-btn", "#practice-submit-area", ".missed-fact-row", ".practice-mode-notice"],
        "result.css": [".solution-code", ".ai-explanation-text"],
        "misc.css": [".colab-card", ".colab-card-link", ".report-btn", ".self-report-btn", ".placement-start-btn", ".practice-aids"],
    }
    for fname, selectors in expected.items():
        css = _read(os.path.join(HERE, fname))
        for sel in selectors:
            assert sel in css, f"{fname} lost required selector: {sel!r}"


# ── Invariant checks ──────────────────────────
def check_invariants():
    # No leftover modulario template markers.
    for fname in ("README.md", "watch.py"):
        first = _read(os.path.join(HERE, fname)).splitlines()[:1]
        assert first and "modulario:template" not in first[0], (
            f"{fname} still has modulario template marker on line 1"
        )
    # ID-vs-.hidden specificity fix must survive (submit row could never hide
    # while #practice-submit-area{display:flex} beat the global .hidden).
    feedback = _read(os.path.join(HERE, "feedback.css"))
    assert "#practice-submit-area.hidden" in feedback, (
        "feedback.css lost the #practice-submit-area.hidden re-assert — "
        "the submit row will stay visible after grading"
    )
    # Rigid-session page states: idle hides the split, running hides setup.
    timer = _read(os.path.join(HERE, "timer.css"))
    assert "#page-practice.session-idle .practice-split" in timer, (
        "timer.css lost the session-idle rule that hides the question split"
    )
    assert ":not(.session-idle) .session-setup" in timer, (
        "timer.css lost the rule hiding the setup panel during a session"
    )
    # The topbar is a SIBLING of the split, so the rule above does not cover it.
    # Without its own rule the setup screen shows the paused session's concept,
    # rung and difficulty for a question that is not on screen.
    assert "#page-practice.session-idle .concept-topbar" in timer, (
        "timer.css lost the session-idle rule that hides the concept topbar — "
        "the setup screen will show the previous question's concept strip"
    )


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
