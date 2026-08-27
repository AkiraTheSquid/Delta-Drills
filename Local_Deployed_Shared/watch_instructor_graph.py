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
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def check_instructor_graph():
    ir = _read(os.path.join(HERE, "instructor-review.js"))
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
        "instructor door binds the EDGE taps that file has no handler for"
    )
    assert "window.deltaConceptGraphCy" in ir, (
        "instructor-review.js must borrow the instance rather than build a "
        "second one over the same data"
    )
    assert "hosting()" in ir, (
        "the tap handlers are bound for the life of the page and must go inert "
        "while the graph is back on the Knowledge Graph tab serving a learner"
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
    for marker in ("id=\"ir-kg-frame\"", "id=\"ir-kg-exit\""):
        assert marker in index_html, f"index.html missing {marker}"
    assert "ir-kg-open" in ir and "ir-kg-open" in ir_css, (
        "a fixed overlay leaves the page under it scrollable; the body lock "
        "and the class that applies it have to ship together"
    )
    z = re.search(r"\.ir-panel\s*\{[^}]*z-index:\s*(\d+)", ir_css, re.S)
    assert z and int(z.group(1)) > 20, (
        "the flag card must sit above the learner-model dock (z-index 20 in "
        "styles/how-it-works.css), which is opaque and cut the send button off"
    )
