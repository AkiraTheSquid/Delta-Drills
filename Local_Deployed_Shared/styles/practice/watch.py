"""watch.py — health checks for styles/practice

Practice-tab stylesheets (split from the former practice.css monolith).
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(os.path.dirname(HERE))

CSS_FILES = ["layout.css", "timer.css", "question.css", "feedback.css", "editor.css",
             "misc.css", "stage-ladder.css", "notch-menu.css", "notebook-editor.css", "diagnostic.css",
             # The syntax-highlight overlay (@M, 2026-08-23). Its metrics are
             # SHARED with .code-editor in editor.css — a font, padding or
             # border restated in one and not the other drifts the overlay off
             # the textarea it sits behind.
             "code-highlight.css",
             # The Practice tab's idle readiness dial (2026-08-23).
             "readiness.css"]


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
        "layout.css": [".practice-container", ".practice-split", ".practice-left", ".practice-right"],
        "timer.css": [".session-setup", ".session-status-row", ".session-countdown", ".timer-input", "session-idle"],
        # .cold-start-badge left this list on 2026-08-23 — the badge, its two
        # copy blocks and their styles were deleted together.
        "question.css": [".question-text", ".question-imports", ".question-visual", ".question-number-row"],
        "feedback.css": [".result-badge", ".feedback-btn", "#practice-submit-area", ".missed-fact-row"],
        "editor.css": [".code-editor", ".output-area", ".solution-code", ".ai-explanation-text"],
        "notebook-editor.css": [".practice-notebook", ".notebook-cell", ".notebook-cell-output"],
        "misc.css": [".torch-colab-notice", ".self-report-btn", ".placement-start-btn", ".placement-next-btn", ".practice-aids"],
        # Written by practice/placement-results.js — every class it mints has
        # to keep a rule here or the results card renders as unstyled spans.
        "diagnostic.css": [
            ".placement-cta", ".placement-overall", ".placement-overall-figure",
            ".placement-overall-say", ".placement-overall-caveat", ".placement-areas",
            ".placement-areas-head", ".placement-area", ".placement-area-name",
            ".placement-area-bar", ".placement-area-pct", ".placement-area-conf",
            ".placement-area-probes", ".placement-area--unprobed",
            ".placement-results-meta", ".placement-results-empty",
        ],
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
    # The ladder moved INSIDE the split on 2026-08-23 (it is a block in the
    # question's heading card now), so the rule above covers it and this one is
    # belt and braces. Kept because the ladder has been a sibling of the split
    # before and would be again the moment anything puts a strip back above the
    # panels — and without a rule of its own the setup screen shows the paused
    # session's concept and rung for a question that is not on screen.
    assert "#page-practice.session-idle .stage-ladder" in timer, (
        "timer.css lost session-idle rule hiding ladder"
    )
    ladder = _read(os.path.join(HERE, "stage-ladder.css"))
    # ONE fill for the whole ladder (was `.stage-seg-fill`, four of them, one
    # per rung). The four bars became one continuous track cut by chevron
    # seams, so the fill class is singular now — and the seam has to be here
    # too, or the bar draws as an undivided strip with the rung names under it
    # pointing at nothing.
    assert ".stage-ladder" in ladder and ".stage-ladder-fill" in ladder
    assert ".stage-ladder-seam" in ladder, (
        "stage-ladder.css lost the chevron seam — the rung divisions would "
        "vanish from a bar whose labels still claim them"
    )
    # 🪦 THE POP-UP READING IS GONE — 2026-08-23, with the strip it hung over.
    # This used to require `--dd-ladder-gap` (the air that stopped the reading
    # reading as a bump IN the bar) and `--dd-callout-arrow-x` (the arrow aimed
    # in the box's own coordinates, because the box is clamped inside the strip
    # and its centre stops being the position described the moment that clamp
    # bites). Both variables were deleted with `.stage-ladder-callout`: the
    # ladder sits in the heading card now and the <h2> above it names the
    # concept, so the pop-up's whole job is done by the card and the fill.
    #
    # What replaces them is the pair below. The reading has to be a line under
    # the track, and the card has to be a card — a ladder with no frame around
    # it, directly under a left-aligned heading, is the layout Seth rejected.
    assert ".stage-ladder-reading" in ladder, (
        "stage-ladder.css has no .stage-ladder-reading — the percentage, the "
        "Integrated chip and the withdrawn-scaffold note are rendered into "
        "that line by stage-ladder.js and would be unstyled or invisible"
    )
    # 🔴 Checked against the RULES, not the file. Both names are still written
    # out in this file's history comments, on purpose — that is where the next
    # person reads why they went — so a bare substring test fails the moment
    # the tombstone is honest about what it is a tombstone for.
    ladder_rules = re.sub(r"/\*.*?\*/", "", ladder, flags=re.S)
    for gone in ("--dd-callout-arrow-x", ".stage-ladder-callout"):
        assert gone not in ladder_rules, (
            f"stage-ladder.css declares {gone!r} again — the floating reading "
            "is back, and it can only float over the question panel now that "
            "the ladder lives inside it"
        )
    question = _read(os.path.join(HERE, "question.css"))
    for prop in ("flex-direction: column", "border-radius"):
        assert prop in question.split(".question-number-row {", 1)[1].split("}", 1)[0], (
            f".question-number-row lost {prop!r} — it is the RECTANGLE that "
            "holds the centred concept title with the bar under it (Seth, "
            "2026-08-23), not a baseline row again"
        )
    assert "text-align: center" in question.split(".question-number {", 1)[1].split("}", 1)[0], (
        ".question-number is no longer centred — the card is built around that "
        "axis and the bar under it is full width"
    )
    # Extra information above every question, and off by default: the ladder
    # rides with Advanced mode (app.js writes `body.dd-basic-mode`).
    assert "body.dd-basic-mode .stage-ladder" in ladder, (
        "stage-ladder.css no longer hides the ladder in basic mode — it is "
        "back above every question for every learner, which is the cognitive "
        "load it was gated to avoid"
    )

    notch = _read(os.path.join(HERE, "notch-menu.css"))
    # The two audit controls (#question-id-chip, #practice-graph-jump) were
    # moved out of the question row and into this menu. They keep their
    # `.question-id-chip` class so graph-jump.js's `.is-untagged` marking still
    # lands, which means the chip's BOX comes with them unless it is overridden.
    assert ".practice-notch-item.question-id-chip" in notch, (
        "notch-menu.css no longer re-shapes the moved audit controls — they "
        "are bordered pills sitting in a menu of flat rows"
    )
    assert ".practice-notch-item.question-id-chip.is-untagged" in notch, (
        "the untagged marking is gone. `.question-id-chip.is-untagged` only "
        "says `border-style: dashed`, which draws NOTHING on a borderless "
        "menu row — and a question the tutor cannot place on its own map is "
        "the one thing this control exists to show"
    )
    assert ".practice-notch-clock" in notch, (
        "notch-menu.css lost .practice-notch-clock — the session row is hidden "
        "and this tab is the only countdown on the practice screen"
    )

    feedback = _read(os.path.join(HERE, "feedback.css"))
    assert ".problem-feedback-note:focus" in feedback


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
