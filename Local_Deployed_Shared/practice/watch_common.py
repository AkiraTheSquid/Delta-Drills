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
    "bars.js", "stage-ladder.js", "notebook-editor.js", "config.js",
    "notch-menu.js",  # the seam tab that proxies Pause & exit / End session to timer.js
    "survey.js",  # the "what have you done before?" card that replaced the placement test
    "arena-unlock-dom.js",  # injects #arena-unlock-page into #page-practice at script-eval time
    "arena-unlock.js",  # interstitial controller (consumes the stats/predicted-prereqs-temp.js scaffold)
    "kernel.js",  # persistent per-learner backend session behind notebook.js
    "notebook-view.js",  # the Notebooks tab: a whole compiled lesson on one kernel
    # Colab-style code cells (@M, 2026-08-23). Both are optional-chained from
    # runner.js, so a missing one degrades to a plain textarea rather than
    # throwing — which is exactly why they have to be asserted here instead.
    "code-highlight.js",  # tokenised <pre> overlay behind the transparent textarea
    "code-complete.js",  # name-only ghost autocomplete, accepted with Tab
    # The Learner Home (idle screen): the one button back in, and measured
    # learning with its two graphs (2026-09-26).
    "session-idle.js",  # proxies Continue to the real resume/start buttons
    "xp-charts.js",  # both graphs; xp-panel.js reads its helpers at load
    "xp-target-drag.js",  # click / drag the course graph to set the target
    "xp-group-view.js",  # in a group, draws the graph column; before xp-panel.js
    "xp-panel.js",
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
