"""watch_instructor_graph.py — the instructor graph door's checks.

Split out of watch.py the way watch_front_door.py was (2026-08-24, Modulario's
700-LOC line); watch.py was already past it. Same contract as every check
there: raise AssertionError to fail. watch.py imports check_instructor_graph
back into its own namespace and keeps it in the __main__ checks list, so
`mod watch` and the explicit runner both still see it — a split must never
change WHICH checks run (a runner list has dropped checks silently before).

What this guards, 2026-08-27: the "Review the graph" door stopped drawing its
own copy of the dead ARENA atom graph and now HOSTS the real lesson graph —
it MOVES `.kg-container.kg2` out of the Knowledge Graph tab and puts it back.
Every failure mode of that move is silent. A graph left in a hidden page looks
like an empty tab; a door that quietly falls back to graph-viz.json looks like
a working screen showing the wrong curriculum. Nothing throws in either case.

And, later the same day, the door became an EDITOR: edges are selected and
inspected on the right, deleted, or reversed; a ✛ handle on the focused bubble
creates a concept or drags out an edge. Every one of those is drawn on the
LEARNER'S OWN live graph, because there is only one. So the checks below are
mostly about the way back: the ledger's reverts, the order they run in, and
the fact that nothing here may write the shared cytoscape stylesheet. An edit
that survives the exit is a curriculum change no one approved, showing up on a
learner's map with no way to trace it.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def check_instructor_graph():
    ir = _read(os.path.join(HERE, "instructor-review.js"))
    ed = _read(os.path.join(HERE, "instructor-graph-edit.js"))
    index_html = _read(os.path.join(HERE, "index.html"))
    ir_css = _read(os.path.join(HERE, "styles", "instructor-review.css"))
    lesson_graph = _read(os.path.join(HERE, "concept-graph", "lesson-graph.js"))

    # ── it reviews the graph the app TEACHES FROM ──────────────────
    # graph-viz.json is the old 205-atom ARENA export; lesson-graph.js
    # superseded it and nothing else in the app has drawn it for months.
    # Reviewing sequencing there reviews a curriculum no learner sees.
    # The FETCH, not the string: the file's header explains what it stopped
    # drawing, and a guard that forbids naming the old graph forbids saying so.
    assert 'fetch("concept-graph/graph-viz.json' not in ir, (
        "the instructor graph door must not fetch graph-viz.json — that is the "
        "dead ARENA atom graph, not the lesson graph the app teaches from"
    )
    assert 'id="ir-cy"' not in index_html, (
        "#ir-cy was this door's own canvas; hosting the real graph means there "
        "is no second cytoscape container to render into"
    )

    # ── the move, and the way back ─────────────────────────────────
    assert 'KG_SELECTOR = ".kg-container.kg2"' in ir, (
        "the hosted element must be the .kg2 CONTAINER, not the .kg2-wrap "
        "inside it — lesson-graph.js's fitWrap() looks the wrap up as "
        "'.kg2 .kg2-wrap' and the graph loses its height if the wrap moves"
    )
    assert "kgHome" in ir and "insertBefore" in ir, (
        "instructor-review.js must remember where it took the graph FROM and "
        "put it back there; appending it home is not the same node order"
    )
    assert 'closest("#page-knowledge-graph")' in ir, (
        "hostKg must refuse to take the graph from anywhere but home. "
        "concept-graph/why-graph.js borrows the same element for the landing "
        "page's maximise, and two borrowers each remember a different home — "
        "whichever releases second finds nothing to put back"
    )
    assert re.search(r'if \(name !== "graph"\) releaseKg\(\);', ir), (
        "every exit from the graph view goes through show(), so show() is "
        "where the graph is handed back"
    )
    assert "MutationObserver" in ir and 'attributeFilter: ["class"]' in ir, (
        "app.js's switchTab fires no event; the page's own `hidden` class is "
        "the only signal that the instructor navigated away mid-review, and "
        "without it the Knowledge Graph tab opens empty"
    )
    assert ir.count("releaseKg()") >= 3, (
        "release must cover all three exits: the view switch, the "
        "instructor-mode flag dropping, and a tab switch away"
    )

    # ── the borrowed cytoscape ─────────────────────────────────────
    assert "window.deltaConceptGraphCy" in lesson_graph, (
        "lesson-graph.js must export its live cytoscape instance; the "
        "instructor door edits the graph the learner is served"
    )
    assert "window.deltaConceptGraphCy" in ir, (
        "instructor-review.js must borrow the instance rather than build a "
        "second one over the same data"
    )

    # ── the editor is wired, and wired FIRST ───────────────────────
    assert os.path.exists(os.path.join(HERE, "instructor-graph-edit.js")), (
        "instructor-graph-edit.js is the editor; without it the graph door is "
        "a viewer with a disabled toolbar"
    )
    i_edit = index_html.find('src="instructor-graph-edit.js')
    i_review = index_html.find('src="instructor-review.js')
    assert i_edit != -1 and i_review != -1 and i_edit < i_review, (
        "instructor-graph-edit.js must be loaded BEFORE instructor-review.js: "
        "the door hands it the borrowed cytoscape the moment the graph lands, "
        "and both are plain scripts, so source order is load order"
    )
    # \b, because `window.DDGraphEditor` contains `window.DDGraphEdit` and a
    # substring test called a renamed export wired.
    assert re.search(r"window\.DDGraphEdit\s*=\s*api", ed), (
        "the editor must publish itself at exactly window.DDGraphEdit"
    )
    assert re.search(r"window\.DDGraphEdit\b", ir), (
        "the two halves meet at window.DDGraphEdit — the editor publishes it, "
        "the door consumes it"
    )

    # attach() answers false when it cannot build its chrome inside the
    # borrowed graph. Ignored, the toolbar keeps its buttons and none of them
    # does anything — the failure that looks like a working screen.
    assert re.search(r"if \(!editor\(\)\.attach\(\{", ir), (
        "attachEditor must act on attach()'s answer, not just call it"
    )

    # ── every edit is reversible, and reverted on the way out ──────
    # This is the whole safety story. cy belongs to the Knowledge Graph tab;
    # an edit that outlives the visit is a silent curriculum change on a
    # learner's map.
    # The CALL, at the start of a line of code — the file header explains the
    # restore contract in prose, and a guard satisfied by its own explanation
    # guards nothing.
    assert re.search(r"^\s*if \(e\.removed\) \{ e\.removed\.restore\(\); ", ed, re.M), (
        "reverting a removal has to be cytoscape's own restore() on the "
        "collection remove() returned — rebuilding the element from its data "
        "loses position, classes and the styles lesson-graph.js put on it"
    )
    assert re.search(r"for \(let i = edits\.length - 1; i >= 0; i--\) revertEdit", ed), (
        "detach must revert in REVERSE order: an edge can hang off a node "
        "staged before it, and undoing the older edit first strands the newer "
        "one's elements"
    )
    assert re.search(r"if \(editor\(\)\) editor\(\)\.detach\(\);", ir), (
        "releaseKg must detach the editor; the element handed back to the "
        "Knowledge Graph tab has to be the one that was borrowed"
    )
    detach_at = ir.find("editor().detach()")
    move_at = ir.find("kgHome.parent.insertBefore")
    assert detach_at != -1 and move_at != -1 and detach_at < move_at, (
        "detach BEFORE the DOM move: reverting reads the frame the editor's "
        "panel and ✛ handle live in, and moving first hands the learner's tab "
        "an element with instructor chrome still parented inside it"
    )
    # Anchored to the START of a line so commenting the unbind out fails the
    # check; a `//` in front of it is exactly the edit this is here to catch.
    for ev in ('"tap", "edge"', '"tap", "node"', '"pan zoom position"'):
        assert re.search(r"^\s*cy\.on\(" + re.escape(ev), ed, re.M) and re.search(
            r"^\s*cy\.removeListener\(" + re.escape(ev), ed, re.M
        ), (
            f"the editor binds {ev} on a BORROWED instance and must unbind it "
            "in detach — a handler left behind puts a ✛ handle and an "
            "inspector on the learner's own Knowledge Graph tab"
        )
    # A ✛ drag binds move/up on WINDOW, so it outlives the page under it. Exit
    # mid-drag and the next pointer event asks a null cytoscape where a node is.
    assert re.search(r"^\s*if \(dragCleanup\) dragCleanup\(\);", ed, re.M), (
        "detach must tear down an in-flight ✛ drag: its pointermove/pointerup "
        "live on window and are otherwise removed only by the drop that never "
        "came"
    )
    # An edge drawn FROM a proposed concept cannot outlive it: the orphaned
    # edit survives to the next attach(), where the replay adds an edge whose
    # source does not exist and cytoscape throws.
    assert "dependsOn" in ed and re.search(r"forEach\(\(e\) => unstage\(e, true\)\)", ed), (
        "undoing a proposed concept must cascade to every edit that references "
        "it"
    )
    assert "cy.style(" not in ed, (
        "proposals are marked with INLINE element styles, never a rule added "
        "to the shared stylesheet: the stylesheet is lesson-graph.js's and a "
        "selector added here would outlive the visit"
    )
    # Scoped to the poll itself: `if (!kgHome) return;` is also releaseKg's
    # own early return, so an unscoped test passes with the poll unguarded.
    poll = re.search(r"const attachEditor = \(\) => \{(.*?)\n  \};", ir, re.S)
    assert poll and "if (!kgHome) return;" in poll.group(1), (
        "attachEditor polls for a cytoscape that may not exist yet, and the "
        "instructor can leave inside that window — attaching afterwards arms "
        "the editor on a graph that is back home serving a learner"
    )

    # ── full-bleed, under the topbar ───────────────────────────────
    assert "#ir-graph:not(.hidden)" in ir_css, (
        "the graph view is fixed; `.hidden` (components.css) has the same "
        "specificity as a bare class, so the rule must out-specify it"
    )
    assert "inset: var(--dd-topbar-h) 0 0 0" in ir_css, (
        "the overlay covers the page and leaves the topbar — and it reads the "
        "bar's height from the one token, never a literal 44px"
    )
    for marker in ('id="ir-kg-frame"', 'id="ir-kg-exit"'):
        assert marker in index_html, f"index.html missing {marker}"
    assert "ir-kg-open" in ir and "ir-kg-open" in ir_css, (
        "a fixed overlay leaves the page under it scrollable; the body lock "
        "and the class that applies it have to ship together"
    )

    # ── the inspector, on the right ────────────────────────────────
    insp = re.search(r"\.ir-insp\s*\{(.*?)\}", ir_css, re.S)
    assert insp, "styles/instructor-review.css must style the inspector"
    assert re.search(r"right:\s*0", insp.group(1)), (
        "Seth asked for the information and the options on the RIGHT; the "
        "panel docks to that edge of the frame"
    )
    z = re.search(r"z-index:\s*(\d+)", insp.group(1))
    assert z and int(z.group(1)) > 20, (
        "the inspector must sit above the learner-model dock (z-index 20 in "
        "styles/how-it-works.css), which is opaque and cut the old flag card's "
        "send button off"
    )
    handle = re.search(r"\.ir-handle\s*\{(.*?)\}", ir_css, re.S)
    assert handle, "the ✛ handle needs a rule; it is positioned from JS but sized here"
    assert "touch-action: none" in handle.group(1), (
        "the ✛ drag is a pointer gesture we own; without touch-action a phone "
        "scrolls the page instead of drawing the edge"
    )
