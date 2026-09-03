"""Focused invariants for ARENA notebook navigation and editing.

The contents tree is a plain Colab-style list (Seth, 2026-09-03) — the
proportional LessWrong rail with its dots and progress line is gone, and
these checks are what keep it gone. The reveal-zone check is here because
that surface is invisible: it is a transparent strip that takes the mouse,
so when it is measured wrong nothing LOOKS wrong.
"""

import os
import re

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

    # Whole left gutter reveals one ordinary tree panel. JS measures the prose
    # edge because a hard-coded strip fails as viewport width changes.
    assert 'getBoundingClientRect().left' in nav and '--anb-gutter' in nav, (
        "the contents reveal zone no longer reaches the prose column's live left edge"
    )
    assert ".anb-toc-hit" in css and "inset: 0" in css, (
        "the hover target no longer fills the whole gutter"
    )
    assert ".anb-toc.is-hover .anb-toc-panel" in css, (
        "hovering the left gutter no longer reveals the contents tree"
    )
    assert "overflow-y: auto" in css, "long notebook trees no longer scroll"

    # Current location: baby-blue, bold, with a right-pointing marker at the
    # far end of the flex row. Completed sections: green + check.
    assert "#e8f0fe" in css and "#174ea6" in css and "font-weight: 700" in css, (
        "current tree row lost its Colab-blue bold treatment"
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
    assert "anb-toc-check" in nav and "✓" in nav and "#188038" in css, (
        "completed tree rows lost their green check"
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
    assert "--anb-zoom:" in column, (
        "the ARENA reading surface lost its single scale token — every measured "
        "value is written as `calc(<their px> * var(--anb-zoom, 1))` so the size "
        "is one number and the provenance survives"
    )
    assert "calc(682px * var(--anb-zoom" in column, (
        "the reading column is no longer LessWrong's 682px measure x the zoom"
    )
    assert "calc(18.2px * var(--anb-zoom" in column and "calc(26px * var(--anb-zoom" in column, (
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

    # 🔴 EQUAL MARGINS. The column is centred and the tree lives INSIDE the
    # left one — Seth, 2026-09-03: "the left sidebar is part of what I counted
    # as the margin". Reserving the tree's width as page padding is what broke
    # this the first time: it shoved the reading against the right edge, ~440px
    # of margin on one side and ~90px on the other. The panel takes the margin
    # centring leaves (`clamp(320px, 100%, ...)`) rather than the page holding
    # a strip open for it.
    assert "padding-left: calc(20px + var(--anb-toc-w))" not in column, (
        "the ARENA page reserves the contents tree's width as left padding "
        "again — that un-centres the reading column, which is the one thing "
        "about this layout Seth asked for by name"
    )
    # The panel is sized from that margin, minus the gap the column was nudged
    # over by. Whitespace-insensitive because the declaration is multi-line.
    # 🔑 `(?<!-)` or this matches the tail of `max-width:`, which is declared
    # first in the same block — the check then read a value it was not about.
    panel_w = re.search(
        r"\.anb-toc-panel\s*\{[^}]*?(?<![-\w])width:\s*([^;]+);", css, re.S
    )
    assert panel_w, ".anb-toc-panel no longer declares a width"
    flat = re.sub(r"\s+", "", panel_w.group(1))
    assert flat == "clamp(320px,calc(100%-var(--anb-nudge,100px)),var(--anb-toc-w,340px))", (
        "the contents panel no longer takes the centred column's margin less "
        f"the nudge (found {flat!r}). `min()` there obeyed a 162px gap and "
        "rendered a column of ellipses; taking the gutter WHOLE put the tree's "
        "right edge on the same pixel as the prose's left edge"
    )

    # 🔴 BOTH HALVES OF THE GAP, OR NEITHER. The column moving right without
    # the panel giving the space back just feeds the nudge to the tree.
    assert "--anb-nudge:" in column and "padding-left: calc(2 * var(--anb-nudge))" in column, (
        "the reading column no longer clears the contents tree by --anb-nudge "
        "— the two edges met on the same pixel before this"
    )


def check_the_contents_reveal_zone_never_covers_a_run_button():
    """🔴 THE HOVER STRIP IS MEASURED TO THE LEFTMOST CELL, NOT THE COLUMN.

    This is a real bug that shipped in the working tree and was caught only by
    measuring the rendered page (Chrome, 2026-09-03, 1600px): `.anb-toc-hit`
    was sized from `.nbv-cells`, whose left edge is 454px — but a code cell
    BREAKS OUT to 880px and starts at 355px, so the strip lay on top of every
    Run button on the page. `document.elementFromPoint` over a Run button
    answered `.anb-toc-hit` and not one cell in the notebook could be run.

    Nothing about that is visible: the strip is transparent, the buttons are
    still painted, and the only symptom is that clicking does nothing. So the
    shape of the measurement is pinned here rather than trusted.
    """
    nav = _live(read(os.path.join(HERE, "arena-notebook-nav.js")))

    body = nav[nav.index("const _measure = ()"):]
    body = body[: body.index("\n  };")]

    assert ".nbv-cell" in body and "Math.min(" in body, (
        "_measure no longer walks the cells for the leftmost painted edge — a "
        "gutter measured from `.nbv-cells` alone covers every breakout code "
        "cell, and with it every Run button"
    )
    assert "getClientRects().length" in body, (
        "_measure counts cells inside a closed <details>, which have no box and "
        "measure 0 — that collapses the gutter to nothing"
    )
    # 🔴 A FLOOR IS THE SAME BUG AT A NARROWER WINDOW: it claims gutter the
    # content is already using. Below 1180px the stylesheet drops the strip
    # for the toggle button instead, which is the only correct answer there.
    assert "Math.max(220" not in body, (
        "_measure floors the gutter width again — a floor wider than the "
        "content edge puts the strip back over the cells"
    )

    css = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook-nav.css")), css=True)
    # 🔴 THE FALLBACK BREAKPOINT IS WHERE THE PANEL STOPS FITTING IN THE
    # MARGIN, not where it stops fitting on screen. At 1180px the 320px floor
    # was wider than the margin below ~1410px, so the panel lay over the prose
    # — and a Run button inside an open hover panel is unreachable, because
    # moving toward it never fires mouseleave.
    fallback = re.search(
        r"@media \(max-width:\s*([\d.]+)px\)\s*\{(.*?)\n\}", css, re.S
    )
    assert fallback, "the narrow-window fallback media block is gone"
    # 🔑 Scoped to the block. Searching the whole sheet for the rule passes
    # even after the rule is moved OUT of the media query, which is the one
    # way this fallback silently stops being a fallback.
    assert ".anb-toc-hit { display: none; }" in fallback.group(2), (
        "the hover strip is no longer dropped inside the narrow-window "
        "fallback — with no room for a safe gutter it has to give way to the "
        "toggle button"
    )
    # 🔴 THE TWO BREAKPOINTS ARE ONE BREAKPOINT. `max-width: 1499px` beside
    # `min-width: 1500px` leaves 1499.5px matching NEITHER, and a viewport
    # lands there under display scaling — hover panel live, column not moved,
    # which is the overlap the fallback exists to prevent.
    column = _live(
        read(os.path.join(SHARED, "styles", "practice", "arena-notebook.css")), css=True
    )
    nudge_at = re.search(r"@media \(min-width:\s*([\d.]+)px\)", column)
    assert nudge_at, "the column's nudge is no longer bounded by a media query"
    assert float(fallback.group(1)) < float(nudge_at.group(1)) <= float(fallback.group(1)) + 0.02, (
        f"the contents fallback (max-width: {fallback.group(1)}px) and the "
        f"column nudge (min-width: {nudge_at.group(1)}px) are not complements "
        "— a fractional viewport between them matches neither, and there the "
        "panel lies over the prose with the Run button underneath it"
    )


def check_a_contents_row_can_actually_be_clicked():
    """🔴 THE PANEL MUST OUT-PAINT THE STRIP THAT REVEALS IT.

    `.anb-toc-hit` is `position: absolute; inset: 0` and fills the whole
    gutter. The panel it reveals is its SIBLING, so if the panel is left
    `position: static` it paints BELOW the strip — source order does not save
    it — and the strip eats every click on a row. Seth, 2026-09-03: "it's not
    clickable such that when you click on one of the headings it takes you to
    that part of the page."

    Nothing about that is visible either: the tree renders, highlights and
    scrolls correctly, and only the jump is dead. So the stacking is pinned,
    and so is the handler on the other end of it.
    """
    css = _live(read(os.path.join(SHARED, "styles", "practice", "arena-notebook-nav.css")), css=True)
    nav = _live(read(os.path.join(HERE, "arena-notebook-nav.js")))

    panel = css[css.index(".anb-toc-panel {"):]
    panel = panel[: panel.index("}")]
    assert "position: relative" in panel or "position: absolute" in panel, (
        ".anb-toc-panel is position: static again — it paints under "
        ".anb-toc-hit, which swallows every click on a heading"
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
