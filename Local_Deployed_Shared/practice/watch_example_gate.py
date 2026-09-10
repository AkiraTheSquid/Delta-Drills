"""watch_example_gate.py — no worked example is attached to a drill.

Was the ordering guard for the scheduled worked-example popup
(`example-gate.js`, 2026-08-30). The popup was DELETED 2026-09-10 and this file
is now the ratchet that keeps it deleted; watch.py imports the check and runs it
with the rest. Folder-relative, no arguments.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def code_only(source):
    """The source with its comments removed.

    Every file this guard reads carries a tombstone NAMING the thing it
    forbids — that is the point of a tombstone, and twice now a bare-name
    assertion has failed on its own gravestone. Strip `//` and block comments
    first and the checks can say plainly what may not come back.

    Crude on purpose: a `//` inside a string literal would be cut too. Nothing
    here needs one, and over-stripping can only make this guard MISS, never
    fire falsely on prose.
    """
    out, i, n = [], 0, len(source)
    while i < n:
        if source.startswith("//", i):
            i = source.find("\n", i)
            if i < 0:
                break
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


def check_no_worked_example_is_attached_to_a_drill():
    """A worked example belongs to a LESSON, never to a drill.

    Seth, on q198 (Jaccard distance), 2026-09-10: "it gave me a bunch of
    unrelated code, which is really annoying. The worked examples shouldn't be
    in the problem statements anymore … It shouldn't be either in the problem
    panel or in the runnable cells because we don't put worked examples in the
    problems anymore. They have their own lessons."

    Two paths used to do it and both are gone:

      1. `example-gate.js` — the server-scheduled popup. It took over
         `#question-text` AND called `DeltaNotebook.reset(exampleCode)`, which
         writes the example into the learner's own primary cell. That is what
         Seth was looking at: an einops broadcasting example sitting in the
         answer cell of a Jaccard problem, graded as his answer.
      2. `ladder.js::decorate` — the rail `<details class="ladder-example">`
         plus `DeltaNotebook.showExamples(...)`, which laid the example's
         fences in as runnable cells above the learner's. Dead since
         `SUPPORTED_STAGES` emptied on 2026-08-30; deleted outright now.

    The two screens that still teach — `LessonGate.maybeShow` (first contact)
    and `LadderUI.maybeShowWorked` (the `worked` rung) — are lesson screens, and
    they stay. What is asserted here is only that nothing puts an example on a
    DRILL card.

    ⚠️ Every file checked here is read through `code_only`, because each one
    carries a tombstone NAMING what it buried — `ExampleGate`, `showExamples`,
    `[data-example-cell]`. Matching the raw text makes this guard fail on its
    own gravestone, which it did twice (2026-09-10, both times on a comment I
    had just written). Strip the comments and the names can be checked plainly.
    """
    assert not os.path.exists(os.path.join(HERE, "example-gate.js")), (
        "practice/example-gate.js is back — the scheduled worked-example popup "
        "puts an example in front of a drill and into the learner's own cell")

    html = read(os.path.join(HERE, "..", "index.html"))
    assert "practice/example-gate.js" not in html, (
        "index.html still loads example-gate.js")

    events = code_only(read(os.path.join(HERE, "events.js")))
    assert "ExampleGate" not in events, (
        "events.js asks the example gate again — a drill may not open behind a "
        "worked example")
    # The lesson gates keep their order: first contact, then the `worked` rung.
    assert events.index("LessonGate.maybeShow(nextQ") < events.index(
        "LadderUI.maybeShowWorked(nextQ"), (
        "events.js asks the `worked` rung before first contact")

    ladder_js = code_only(read(os.path.join(HERE, "ladder.js")))
    for gone in ("SUPPORTED_STAGES", "_exampleHtml", "showExamples"):
        assert gone not in ladder_js, (
            f"ladder.js reaches for `{gone}` again — nothing may attach an "
            "example to a drill card, in the rail or in the cells")

    editor_js = code_only(read(os.path.join(HERE, "notebook-editor.js")))
    for gone in ("data-example-cell", "exampleCell", "showExamples", "renderExamples"):
        assert gone not in editor_js, (
            f"notebook-editor.js re-grew example cells (`{gone}`) — the only "
            "non-learner cell in the notebook is the graded solution")
