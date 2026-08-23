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
    "placement-timer.js",  # the placement test's own fixed per-question clock
    "arena-unlock-dom.js",  # injects #arena-unlock-page into #page-practice at script-eval time
    "arena-unlock.js",  # interstitial controller (consumes the stats/predicted-prereqs-temp.js scaffold)
    "kernel.js",  # persistent per-learner backend session behind notebook.js
    "notebook-view.js",  # the Notebooks tab: a whole compiled lesson on one kernel
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
