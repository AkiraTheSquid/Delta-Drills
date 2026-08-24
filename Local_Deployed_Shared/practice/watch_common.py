"""watch_common.py — shared fixtures for the practice watch checks.

`watch.py` grew past the point where one file could hold every check and stay
readable, so the checks live in three modules now (this one, `watch_invariants`,
`watch_lessons`). Everything they share sits here: where the folder is, what has
to exist in it, and how to read a file. Nothing in here asserts — a helper that
fails is a failure with no check name attached to it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)

REQUIRED_JS = [
    "init.js", "dom.js", "events.js", "engine.js", "api.js",
    "runner.js", "visuals.js", "ui.js", "ai.js", "tutor.js", "mode.js",
    "adaptive.js", "questions.js", "storage.js", "timer.js",
    "bars.js", "stage-ladder.js", "diagnostic-page.js", "notebook-editor.js", "config.js",
    "notch-menu.js",  # the seam tab that proxies Pause & exit / End session to timer.js
    "placement-timer.js",  # the placement test's own fixed per-question clock
    "placement-results.js",  # the only writer of the placement results card body
    "arena-unlock-dom.js",  # injects #arena-unlock-page into #page-practice at script-eval time
    "arena-unlock.js",  # interstitial controller (consumes the stats/predicted-prereqs-temp.js scaffold)
    "kernel.js",  # persistent per-learner backend session behind notebook.js
    "notebook-view.js",  # the Notebooks tab: a whole compiled lesson on one kernel
    # Colab-style code cells (@M, 2026-08-23). Both are optional-chained from
    # runner.js, so a missing one degrades to a plain textarea rather than
    # throwing — which is exactly why they have to be asserted here instead.
    "code-highlight.js",  # tokenised <pre> overlay behind the transparent textarea
    "code-complete.js",  # name-only ghost autocomplete, accepted with Tab
    # The Practice tab's idle screen (2026-08-23).
    "readiness.js",  # % of the 63 KCs mastered, atom-sourced readings only
    "session-idle.js",  # paints the dial and proxies Continue to the real buttons
    # Basic mode (2026-08-23). styles/practice/basic-mode.css hides the felt-
    # difficulty rating; this file is what still commits the attempt to mastery
    # and still reveals Next problem once it is hidden. A missing file is a
    # silently frozen student model, so it is asserted rather than optional-
    # chained away.
    "basic-mode.js",
]
REQUIRED_DOCS = ["README.md", "RUNTIME_CONTRACT.md"]
REQUIRED_ASSETS = [
    os.path.join(SHARED, "delta_numbers.npy"),
    os.path.join(SHARED, "numbers_stacked.png"),
    os.path.join(SHARED, "questions_structured.json"),
    os.path.join(SHARED, "arena_prereqs_structured.json"),
]


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
