"""Focused invariants for ARENA notebook navigation and editing.

The contents tree is a plain Colab-style list (Seth, 2026-09-03) — the
proportional LessWrong rail with its dots and progress line is gone, and
these checks are what keep it gone. Since 2026-09-22 it is also a DOCKED
pane with a draggable divider, and the notebook fills everything to its
right; the hover overlay and the transparent strip that revealed it are gone.
"""

import os
import re
import subprocess

from watch_common import HERE, SHARED, read


def _live(source, css=False):
    """Source with its comments removed.

    🔴 COMMENTS FIRST, ALWAYS. Three checks in this repo have now matched their
    own documentation: the tombstone that says `.anb-toc-current` is deleted
    contains the string `.anb-toc-current`, so scanning the raw file reported
    the marker as back. A check that reads its own explanation is not a check.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    if not css:
        source = re.sub(r"(?m)^\s*//.*$", "", source)
    return source


def check_the_arena_contents_tree_is_a_plain_colab_tree():
    """ARENA keeps its plain tree, completion state, editing, and math."""
    nav = _live(read(os.path.join(HERE, "arena-notebook-nav.js")))
    view = _live(read(os.path.join(HERE, "arena-notebook.js")))
    css = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook-nav.css")), css=True)
    column = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook.css")), css=True)

    # Old proportional rail is intentionally gone: no dots, progress track,
    # section-height flex sizing, or scrolling-window marker.
    for retired in ("anb-toc-dot", "anb-toc-progress", "--anb-window", "flexBasis"):
        assert retired not in nav and retired not in css, (
            f"the retired LessWrong rail primitive {retired!r} returned"
        )

    # 🪦 The hover overlay is gone (Seth, 2026-09-22: "it won't have the
    # hover effect anymore where it hides itself until you hover your cursor
    # over it"). Its transparent reveal strip is the thing that once sat over
    # every Run button, so it must not come back by accident.
    for retired in ("anb-toc-hit", "is-hover", "--anb-gutter", "anb-toc-toggle"):
        assert retired not in nav and retired not in css, (
            f"the hover-to-reveal contents overlay is back ({retired!r}) — the "
            "pane is docked and always shown"
        )
    assert "overflow-y: auto" in css, "long notebook trees no longer scroll"

    # Current location: blue wash, blue label, bold. Completed sections: green
    # + check.
    # 🔴 ASSERTED AS TOKENS, NOT AS #e8f0fe / #174ea6. Those literals were safe
    # only while this page was pinned to the light palette in every theme, and
    # it follows `data-theme` now (2026-09-10) — a Colab-blue-on-white row
    # hardcoded here is an invisible label on the dark theme. The rule being
    # pinned is unchanged: the row the reader is on is blue and bold.
    assert "--info-rgb" in css and "var(--info)" in css and "font-weight: 700" in css, (
        "current tree row lost its blue bold treatment"
    )
    # 🪦 The ▶ direction marker is DELETED (Seth, 2026-09-03: "kind of confusing
    # to the user"). Blue-and-bold is the whole of "you are here" now.
    assert "anb-toc-current" not in nav and "anb-toc-current" not in css, (
        "the ▶ current-row marker is back — the row's blue bold already says "
        "where the reader is, and a second marker reads as a button"
    )
    assert "syncCompletion" in nav and "is-complete" in nav, (
        "the tree no longer derives section completion from run cells"
    )
    assert "has-failed" in nav and "is-stale" in nav, (
        "failed or stale cells can mark a section complete"
    )
    assert "anb-toc-check" in nav and "✓" in nav and "var(--ok)" in css, (
        "completed tree rows lost their green check"
    )

    # 🪦 The panel's title row is DELETED (Seth, 2026-09-22: "not have
    # that thing at the top left"). It restated the notebook's h1, which the
    # reading column is already showing, so the first line of a tree revealed
    # by a passing mouse was the one line in it you could not go anywhere
    # from. The name is not dropped, it MOVES: it is the rail's accessible
    # name, which is why both halves are asserted together — putting the div
    # back would read the name out twice, and dropping the label leaves a
    # screen reader with "Notebook contents" and no idea which notebook.
    assert "anb-toc-title" not in nav and "anb-toc-title" not in css, (
        "the contents panel has a visible title row again — it repeats the "
        "notebook heading two inches to its right"
    )
    assert re.search(r'setAttribute\(\s*"aria-label",\s*\n?\s*titleText', nav), (
        "the contents rail no longer names its notebook to a screen reader — "
        "the visible title is gone, so the aria-label is the only place left "
        "that says WHICH notebook these contents belong to"
    )

    # Editing uses source state, not KaTeX-mutated rendered text. Cell changes
    # persist by notebook id and every structural edit rebuilds the heading tree.
    assert "arena-nb-md-editor" in view and "_ddMarkdown" in view, (
        "ARENA text cells are no longer editable as Markdown"
    )
    assert "renderMathInElement" in view and 'left: "$$"' in view, (
        "ARENA notebook markdown no longer receives the KaTeX render pass"
    )
    for action in ("insert-code", "insert-prose", "up", "down", "convert", "delete"):
        assert f'data-cell-action="{action}"' in view, (
            f"ARENA notebook lost its {action!r} cell action"
        )
    assert "dd_arena_cells:" in view and "localStorage.setItem" in view, (
        "ARENA cell edits no longer persist per notebook"
    )
    assert "refresh({ rebuild: true })" in view, (
        "a Markdown heading or structural edit no longer rebuilds the tree"
    )
    # 🔴 ...AND THE REBUILD NEEDS SOMETHING TO REBUILD. `mount` starts by
    # calling `destroy`, which nulls `mountedPage`/`mountedHost`; a notebook
    # with fewer than two headings then returned BEFORE restoring them, so a
    # learner who wrote their second heading got no tree until they left the
    # page. The context has to be recorded above that return, not below it.
    body = nav[nav.index("const mount = (page, host, title)"):]
    body = body[: body.index("\n  };")]
    assert body.index("mountedPage = page;") < body.index("headings.length < 2"), (
        "mount() abandons its page/host before the heading-count return — "
        "refresh({ rebuild: true }) then has nothing to rebuild and a notebook "
        "that grows a second heading can never get its tree back"
    )

    # 🔴 LESSWRONG'S MEASURED NUMBERS STAY WRITTEN AS THEIR NUMBERS, and one
    # token scales them. Seth read their post at 140% zoom and wanted that
    # size; hand-multiplying 18.2 into 25.48 would have thrown away the fact
    # that 18.2 is what their page actually renders, and nothing could be
    # re-checked against them afterwards.
    # 🔴 THE SCALE ITSELF MOVED (2026-09-10). It is one set of measurements in
    # styles/practice/notebook-view.css that all three notebook surfaces share,
    # and this page contributes the ZOOM and nothing else. So the zoom is
    # asserted against this sheet and the measurements against that one; a
    # measurement reappearing HERE would be a hand-scaled copy, which is the
    # exact failure these two assertions exist to prevent.
    surface = _live(
        read(os.path.join(SHARED, "styles", "practice", "notebook-view.css")), css=True
    )
    assert "--anb-zoom:" in column, (
        "the ARENA reading surface lost its single scale token — every measured "
        "value is written as `calc(<their px> * var(--anb-zoom, 1))` so the size "
        "is one number and the provenance survives"
    )
    assert "calc(682px * var(--anb-zoom" in surface, (
        "the reading column is no longer LessWrong's 682px measure x the zoom"
    )
    assert "calc(18.2px * var(--anb-zoom" in surface and "calc(26px * var(--anb-zoom" in surface, (
        "body text is no longer their 18.2px on a 26px line x the zoom"
    )
    # 🔴 ONE COLUMN. The 99px-per-side breakout is what put code cells further
    # left than the prose above them, under the contents tree.
    assert "margin-left: -99px" not in column, (
        "code cells break out of the reading column again — prose and code have "
        "to share both edges"
    )
    assert "calc(15.08px * var(--anb-zoom" in css, (
        "the contents tree is no longer on LessWrong's measured 15.08px row x "
        "the same zoom the reading column uses"
    )

    # 🪦 EQUAL MARGINS, `--anb-nudge` and the panel sized from the centred
    # column's margin are all SUPERSEDED (Seth, 2026-09-22: "go with the collab
    # format where the code and text take up all the space on the right").
    # What replaced them is pinned in check_the_docked_contents_pane_never_
    # covers_the_notebook below.
    assert "--anb-nudge" not in column and "--anb-nudge" not in css, (
        "the centred-column nudge is back — the notebook is the whole pane "
        "right of the contents now, so there is nothing to nudge away from"
    )


def check_the_docked_contents_pane_never_covers_the_notebook():
    """🔴 ONE WIDTH TOKEN, READ ON BOTH SIDES OF THE DIVIDER.

    Seth, 2026-09-22: the contents is a docked Colab pane, "you should be able
    to drag the divider between the table of contents on the left and the code
    on the right". The pane is `position: fixed`, so nothing in layout keeps
    the notebook out from under it — only the container's margin does. Both
    read `--anb-pane-w`, which the nav writes onto the PAGE (0px when folded
    or narrow). If either side stops reading it, the pane lies over the first
    few hundred pixels of every cell, Run buttons included, and nothing LOOKS
    broken until you try to click one.
    """
    nav = _live(read(os.path.join(HERE, "arena-notebook-nav.js")))
    css = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook-nav.css")), css=True)
    column = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook.css")), css=True)

    toc = css[css.index(".anb-toc {"):]
    toc = toc[: toc.index("}")]
    assert "position: fixed" in toc and "var(--anb-pane-w" in toc, (
        "the contents pane no longer takes its width from --anb-pane-w"
    )
    container = column[column.index(".arena-notebooks-container {"):]
    container = container[: container.index("}")]
    assert "margin-left: var(--anb-pane-w" in container, (
        "the notebook no longer steps right of the docked contents pane — the "
        "pane now covers the start of every cell"
    )
    assert 'mountedPage.style.setProperty("--anb-pane-w"' in nav, (
        "the nav no longer writes the pane width onto the page, so the "
        "container's margin and the pane's width can disagree"
    )
    assert 'removeProperty("--anb-pane-w")' in nav, (
        "destroy() leaves the pane width on the page — a notebook with no "
        "contents pane would keep a blank strip where it was"
    )

    # The divider drags, and the drag cannot be lost off the element.
    assert "anb-toc-split" in nav and "cursor: col-resize" in css, (
        "the contents pane lost its draggable divider"
    )
    assert "setPointerCapture" in nav and "pointercancel" in nav, (
        "the divider drag no longer captures the pointer — a drag released "
        "over the notebook leaves the pane stuck mid-resize"
    )

    # Frozen head: the notebook's name, the ⇕ chapter switch, the book.
    for needle in ("anb-toc-name", "anb-toc-switch", "anb-toc-fold", "anb-toc-reopen"):
        assert needle in nav, f"the contents pane's frozen head lost {needle!r}"
    for needle in ("anb-toc-name", "anb-toc-switch", "anb-toc-reopen", ".is-folded"):
        assert needle in css, f"the contents pane's head is no longer styled: {needle!r}"
    assert "ArenaNotebook" in nav and ".sections()" in nav and "canOpen" in nav, (
        "the ⇕ chapter list no longer reads the compiled ARENA index, or no "
        "longer dims the sections the app cannot open"
    )

    # 🔴 NARROW = OVERLAY, and the two numbers are one number.
    m = re.search(r"const NARROW = (\d+);", nav)
    assert m, "the nav lost its NARROW breakpoint"
    fallback = re.search(r"@media \(max-width:\s*([\d.]+)px\)", css)
    assert fallback and int(m.group(1)) - 0.02 <= float(fallback.group(1)) < int(m.group(1)), (
        "the stylesheet's narrow breakpoint no longer matches NARROW in "
        "arena-notebook-nav.js — between them the pane docks with no room, "
        "or overlays with the notebook still stepped aside for it"
    )

    # Focus mode hides the pane; the notebook must take its room back.
    assert ".dd-nb-focus .arena-notebooks-container" in column, (
        "focus mode hides the contents pane but leaves its empty strip"
    )


def check_a_contents_row_can_actually_be_clicked():
    """🔴 A ROW IS A BUTTON THAT SCROLLS THE DOCUMENT.

    Seth, 2026-09-03: "it's not clickable such that when you click on one of
    the headings it takes you to that part of the page." Then it was a static
    panel painting under the hover strip; the strip is gone (2026-09-22), but
    the panel stays positioned — the divider is placed against it.

    Nothing about that is visible either: the tree renders, highlights and
    scrolls correctly, and only the jump is dead. So the stacking is pinned,
    and so is the handler on the other end of it.
    """
    css = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook-nav.css")), css=True)
    nav = _live(read(os.path.join(HERE, "arena-notebook-nav.js")))

    panel = css[css.index(".anb-toc-panel {"):]
    panel = panel[: panel.index("}")]
    assert "position: relative" in panel or "position: absolute" in panel, (
        ".anb-toc-panel is position: static again — the divider is "
        "absolutely placed against its right edge"
    )

    # The other half: a row is a real button that scrolls the document.
    assert '<button type="button" class="anb-toc-label">' in nav, (
        "a contents row is no longer a button"
    )
    assert 'addEventListener("click", onClick)' in nav, (
        "a contents row no longer carries its jump handler"
    )
    assert "window.scrollTo(" in nav and "JUMP_CLEARANCE" in nav, (
        "_jump no longer scrolls the document to the heading it names"
    )


def check_the_exercise_buttons_sit_above_the_cell_the_learner_types_in():
    """Start-timer / Drill-prerequisites goes under the QUESTION, not under
    the answer.

    Seth, 2026-09-22: "it should show up below the problem statement and
    question, rather than below any of your code cells ... sometimes it's just
    not consistent." The inconsistency was that the block was inserted after
    whichever cell the exercise's NAME was found in, and that cell is a
    different thing in each chapter — a heading in 0.1/0.2, the answer stub in
    0.0's A-I sections, a `(N)` tag above `display_soln_array_as_img(N)` in
    0.0's image ops. Only the last of those put the buttons where the question
    ends.

    So the placement is DERIVED now: walk down from the anchor over the rest of
    the statement and stop above the first cell the learner answers in. The
    einops case is the one that proves it is not simply "after the prose" —
    the target picture is drawn by a CODE cell that belongs to the question
    ("that code block would go above the buttons"), so a rule that stopped at
    the first code cell would put the buttons above the picture.
    """
    js = _live(read(os.path.join(HERE, "exercise-session.js")))
    timer = _live(read(os.path.join(HERE, "exercise-timer.js")))

    assert "_placeFor" in js and "beforebegin" in js, (
        "the exercise block is pinned to its anchor cell again — whichever "
        "cell carried the name decides where the buttons land, which is the "
        "inconsistency this rule exists to remove"
    )
    # Signature-agnostic on purpose: the walk gained a `wanted` argument on
    # 2026-09-22 and the check should follow the behaviour, not the arity.
    place = js.split("const _placeFor = (", 1)
    assert len(place) == 2, "exercise-session.js::_placeFor is gone"
    body = place[1].split("\n  };", 1)[0]
    assert "_isAnswerCell(node, table)" in body and "beforebegin" in body, (
        "_placeFor no longer stops above the learner's own cell"
    )
    # 🔴 THE PICTURE IS PART OF THE QUESTION. A code cell is skipped unless it
    # is the ANSWER; stopping at `nbv-code` would hoist the buttons above
    # `display_soln_array_as_img(N)`, which is the thing the learner is being
    # asked to reproduce.
    assert 'contains("nbv-code")' not in body, (
        "_placeFor stops at the first code cell — 0.0's target image is a code "
        "cell, so the buttons would land above the question's own picture"
    )
    # 🔴 THE WALK IS BOUNDED BY BOUNDARIES, NOT BY A SIBLING BUDGET. A count
    # spends itself on the disclosures the walk skips and on the blocks the
    # walk itself injects, so the LONGEST questions — four hints and two
    # blocks above the stub — are exactly the ones that run out and fall back
    # to the anchor, which is the bug. codex, 2026-09-22.
    assert "PLACE_LOOKAHEAD" not in js, (
        "_placeFor is bounded by a sibling budget again — a long statement "
        "exhausts it and the buttons fall back under the anchor"
    )
    assert "_ownersOf(node, table)" in body and "wanted" in body, (
        "_placeFor no longer stops at the next exercise's own cell, so a "
        "question with no answer cell of its own captures the next one's box"
    )

    answer = js.split("const _isAnswerCell = (cell, table) => {", 1)
    assert len(answer) == 2, "exercise-session.js::_isAnswerCell is gone"
    answer_body = answer[1].split("\n  };", 1)[0]
    assert "ANSWER_RE" in answer_body and "_cellKeys(cell)" in answer_body, (
        "an answer cell is recognised by only one of its two marks — ARENA "
        "writes both `# Your code here` and a bare `def`/`class` stub"
    )

    # 🔴 AND THE CLOCK HAS TO KNOW. The block now sits ABOVE its own anchor for
    # a code-cell exercise, so the anchor is the first node the end-of-exercise
    # scan meets; left in the set it ends the exercise at the answer cell and
    # focus mode hides the box the clock was started for.
    end = timer.split("const _endCell = (block) => {", 1)
    assert len(end) == 2, "exercise-timer.js::_endCell is gone"
    end_body = end[1].split("\n  };", 1)[0]
    assert "anchors.delete(source)" in end_body, (
        "_endCell can stop on the block's OWN anchor again — the exercise "
        "would end at the cell the learner types in, and focus mode would hide it"
    )


def check_the_notebook_editor_never_writes_text_the_learner_did_not_type():
    """A code cell's own decoration must be invisible to every read of it.

    `practice/notebook-code-edit.js` paints syntax colour INTO the
    contenteditable and hangs two decorations off the same tree: the inline
    completion ghost, and a span that holds a line box open on a trailing
    newline. Both sit inside the element whose `innerText` IS the learner's
    program — `arena-notebook.js` reads the cell back that way on every
    keystroke, and `_persistCells` writes what it read into localStorage.

    🔴 So neither decoration may be a real character. Drawn with `content:`
    they are in neither `innerText` nor `textContent`; drawn as text, a ghost
    the learner never accepted is spliced into their source, and the
    zero-width space is a `SyntaxError: invalid non-printable character` the
    moment the cell is run. There is nothing to see when this breaks until a
    cell fails to run, so it is pinned here.
    """
    editor = _live(read(os.path.join(HERE, "notebook-code-edit.js")))
    css = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook.css")), css=True)

    assert "​" not in editor, (
        "notebook-code-edit.js emits a literal zero-width space — it will be "
        "read back into the learner's source by arena-notebook.js and Python "
        "will refuse to parse the cell"
    )
    assert 'content: attr(data-ghost)' in css, (
        "the completion ghost is no longer generated content, so the "
        "suggestion is now part of the cell's source"
    )
    assert 'content: "\\200B"' in css, (
        "the trailing-newline line box is no longer generated content"
    )
    # Both decorations are marked, and both readers skip what is marked.
    assert editor.count('data-nb-skip="1"') >= 2, (
        "a decoration in the code cell is no longer marked data-nb-skip"
    )
    assert 'hasAttribute("data-nb-skip")' in editor, (
        "the text walker no longer skips decorations — readText and the caret "
        "walker must agree about what counts as a character"
    )

    # The cell's source is read from the DOM, never from layout.
    assert ".innerText" not in editor, (
        "notebook-code-edit.js reads a cell with innerText, which is empty "
        "for a cell inside a closed <details> — and ARENA is full of them"
    )


def check_code_on_the_arena_page_is_set_at_the_prose_size():
    """Seth, 2026-09-06: code text the same size as the paragraphs.

    One token, `--anb-code-size`, and every kind of code on the page reads
    from it — the runnable cells, the fenced blocks markdown renders, and the
    output under a cell. Three separate sizes is how the surface drifted the
    first time.
    """
    # 🔴 READ FROM THE SHARED SHEET. The rule is no longer the ARENA page's —
    # every notebook surface sets code at its own prose size now, off the same
    # token, because they are one design at three zooms (2026-09-10).
    css = _live(read(os.path.join(SHARED, "styles", "practice", "notebook-view.css")), css=True)

    # 🔴 THE EXPRESSION, NOT A TOKEN. `--anb-code-size` existed for four hours
    # on 2026-09-10 and was wrong the whole time: a custom property is
    # substituted where it is DECLARED, so one declared at :root resolved
    # against :root's zoom of 1 and every surface that set its own kept 18.2px
    # code inside 22.75px prose. So the check is that the code rule carries the
    # SAME expression `.nbv-md` sets its own font-size with — which is what
    # "code is set at the paragraph size" means — written out where it is read.
    size = "font-size: calc(18.2px * var(--anb-zoom, 1));"
    assert css.count(size) >= 2, (
        "code is no longer set with .nbv-md's own font-size expression — it is "
        "either a different number or a token frozen at the wrong zoom"
    )
    assert "--anb-code-size" not in css, (
        "the derived code-size token is back; it resolves at its declaring "
        "block's zoom, so every surface with a zoom of its own gets the root's"
    )
    for consumer in (".nbv-src code", ".nbv-md pre > code", ".nbv-out"):
        assert consumer in css, f"{consumer} no longer takes its size from the shared rule"

    # 🔴 The chip is for INLINE code in a sentence. Applied to a fenced block it
    # paints the majority of an ARENA section grey, which is the thing Seth
    # asked to be rid of.
    assert ".nbv-md :not(pre) > code" in css, (
        "the chip is back on fenced blocks"
    )
    assert ".nbv-md pre {" in css and "background" not in (
        css.split("\n.nbv-md pre {")[1].split("}")[0]
    ), "a fenced code block is filled again rather than being a plain rectangle"


def check_arena_setup_recovers_without_replaying_answers():
    """The setup queue must survive restart, failure and notebook navigation."""
    test = os.path.join(SHARED, "..", "This-Directory-Only", "scripts", "test_arena_setup.mjs")
    result = subprocess.run(["node", test], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
