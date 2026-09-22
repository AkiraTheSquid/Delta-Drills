#!/usr/bin/env python3
"""WIDTH — authored lesson code must fit the cell it is shown in.

Seth, 2026-09-22, with a screenshot of `kp-linalg-basics`' second worked
example whose comment ended mid-word at "compare with fl":

    "sometimes the code gets cut off ... the text needs to be fixed so that it
    doesn't extend past that part. And you need to check all of the lessons for
    the code cells to see if it extends past that border beyond which the user
    would not be able to see it, and they would have to scroll within the code
    cell. This also discourages things like this example, where it essentially
    tries to utilize printing the string alongside the code on the same line
    instead of breaking it up into multiple lines like it should."

Two things landed from that, and this file is the second.

The first is a RUNTIME affordance: a code cell now grows rightward when its
content does not fit, up to the room beside the column, so a learner typing a
long line can still see what they typed (styles/practice/notebook-view.css and
practice/notebook-cells.js). Its left edge never moves, so the code stays
aligned with the prose.

This file is the AUTHORING rule, and the runtime affordance is not a licence to
skip it. A worked example that only fits because the cell grew is an example
that was written too wide — the growth is there for what the LEARNER writes.
The rule is also the point: at 54 columns a three-value `print` does not fit on
one line, so it gets broken up, which is the shape Seth asked for.

🔴 54 IS MEASURED, NOT CHOSEN.

    682px   LessWrong's measure — styles/practice/notebook-view.css
    × 0.95  the lesson's zoom (`--anb-zoom`) — styles/practice/notebook.css
    − 53px  the run gutter and its divider
    − 2px   the cell's own borders
    − 28px  `.nbv-src` padding
    = 564px of code, at an advance of 10.404px for the code face at 17.29px
    = 54.2 characters

Read back out of a real browser on 2026-09-22 against the running local stack
(`.nbv-src` clientWidth 592 with 14px of padding a side; advance measured from
a 100-character probe span, not assumed from the font's metrics), because the
paper derivation had the padding wrong — notebook-view.css's `.nbv-md pre` wins
over `.nbv-src` inside a lesson and pays 14px, not the 12px `.nbv-src` asks for.

The lesson panel is the NARROWEST of the three surfaces a fence is shown on —
the Notebooks tab runs the same measure at zoom 1.25 (≈56 columns) and the
ARENA page at 1.4 (≈57) — so a fence that fits here fits everywhere, and one
number can be the rule for all three.
"""
from __future__ import annotations

from lesson_lib import FENCE_RE

MAX_CODE_COLS = 54

# What the lesson player puts on screen as a cell, by fence tag.
#
# `python starter` and `python solution` are deliberately absent: those are a
# drill's scaffold and its answer, shown in the practice editor, which is a
# different surface at a width this number was not measured against. They are
# the next pass, not this one.
DISPLAYED_FENCES = ("python", "python no-run", "python worked")

# Never rendered: it runs with the cell and is stripped before display by
# practice/notebook-cells.js, lessons/viewer.html and scripts/colab_cells.py.
# Its `assert _delta_output == '...'` lines are a transcript of what the cell
# printed and cannot be re-wrapped without changing what is asserted.
CHECK_MARKER = "\n# Hidden checks\n"


def displayed_code(text: str):
    """(tag, visible source, ordinal) for every fence a reader is shown."""
    n = 0
    for m in FENCE_RE.finditer(text or ""):
        tag = m.group(1).strip()
        if tag not in DISPLAYED_FENCES:
            continue
        n += 1
        yield tag, m.group(2).partition(CHECK_MARKER)[0], n


def check_code_width(code: str, label: str, block: int | str = 1) -> list[str]:
    """WIDTH for one fence's visible source.

    One character is one column here, which is true of every fence in the bank
    and is checked rather than assumed: a TAB is refused outright (it renders
    as anything from one column to eight, so its width is not knowable from
    the source), and a double-width or combining character has never appeared
    in authored lesson code. If one ever does, this count is the thing to fix.
    """
    problems = []
    for n, line in enumerate(code.splitlines(), 1):
        line = line.rstrip()
        if "\t" in line:
            problems.append(
                f"{label}: WIDTH — block {block} line {n} contains a TAB; its "
                f"width on screen is not knowable from the source. Use spaces."
            )
            continue
        if len(line) > MAX_CODE_COLS:
            problems.append(
                f"{label}: WIDTH — block {block} line {n} is {len(line)} columns "
                f"(max {MAX_CODE_COLS}); the last {len(line) - MAX_CODE_COLS} "
                f"need a sideways scroll to read: {line.strip()[:60]!r}"
            )
    return problems


def check_page_width(kp: dict, label: str) -> list[str]:
    """WIDTH across every fence one KP page shows.

    Four places a reader meets code on a lesson page: the concept body, the
    Watch-out band, the worked example, and the `python worked` fence that an
    applied-practice item carries.
    """
    problems = []
    for si, seg in enumerate(kp.get("segments") or []):
        for part in ("concept", "watch_out", "worked"):
            for _tag, code, n in displayed_code(seg.get(part) or ""):
                problems += check_code_width(code, f"{label}: segment {si + 1} {part}", n)
    applied = (kp.get("sections") or {}).get("Applied practice", "")
    for _tag, code, n in displayed_code(applied):
        problems += check_code_width(code, f"{label}: applied practice", n)
    return problems
