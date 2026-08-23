"""watch_notebook.py — checks for the in-app notebook view.

`notebook-view.js` renders a whole compiled lesson (up to 656 cells) against one
kernel session. It is a THIRD surface beside the practice page and the lesson
gate, and it is the only one that both runs a learner's code and records a
verdict — so the things that can rot here are quiet and expensive:

  * the view stops being loaded at all (a missing script tag is an empty tab),
  * it reads the grader's verdict line differently from the Chrome extension,
  * it records an attempt more than once per problem,
  * it falls back to the stateless prefix runner the kernel exists to replace.

None of those fail loudly in a browser. They fail as a wrong mastery estimate or
a blank page, so they are asserted here.
"""
import os
import re

from watch_common import HERE, SHARED, read

REPO = os.path.abspath(os.path.join(SHARED, ".."))


def check_the_notebook_view_is_loaded_after_what_it_calls():
    """The view is mounted, styled, and loaded after its three dependencies.

    `notebook-view.js` is an IIFE that reads `LessonGate.renderMarkdown` (in
    `lessons.js`), `LessonNotebook.runSource` (in `notebook.js`) and
    `DeltaKernel` (in `kernel.js`). It only calls them at click time, so a wrong
    load order would not throw at eval — the tab would simply render markdown as
    raw text, or refuse to run every cell as if the learner were signed out.
    """
    index_html = read(os.path.join(SHARED, "index.html"))

    assert 'id="page-notebooks"' in index_html, (
        "index.html lost the #page-notebooks mount — the tab would route to nothing"
    )
    assert 'id="notebooks-host"' in index_html, (
        "index.html lost #notebooks-host — notebook-view.js mounts into it by id"
    )
    assert 'data-tab="notebooks"' in index_html, (
        "index.html lost the Notebooks tab button — app.js routes "
        '.tab[data-tab=X] to #page-X, so the page becomes unreachable'
    )
    assert "styles/practice/notebook-view.css" in index_html, (
        "index.html no longer links notebook-view.css — the notebook renders unstyled"
    )

    view_pos = index_html.find('src="practice/notebook-view.js')
    assert view_pos != -1, 'index.html missing <script src="practice/notebook-view.js">'
    for dependency in ("practice/lessons.js", "practice/notebook.js", "practice/kernel.js"):
        pos = index_html.find(f'src="{dependency}')
        assert pos != -1, f"index.html no longer loads {dependency}"
        assert pos < view_pos, (
            f"{dependency} must load BEFORE practice/notebook-view.js — the view "
            "calls into it and would silently degrade, not throw"
        )

    # A referenced asset that is not on disk fails as a silently unstyled page
    # or an empty tab, and REQUIRED_JS in watch_common only covers the script.
    css = os.path.join(SHARED, "styles", "practice", "notebook-view.css")
    assert os.path.exists(css), (
        "styles/practice/notebook-view.css is referenced by index.html but is "
        "not on disk — the notebook would render as an unstyled wall of text"
    )
    manifest = os.path.join(SHARED, "lessons", "notebooks", "manifest.json")
    assert os.path.exists(manifest), (
        "lessons/notebooks/manifest.json is missing — notebook-view.js fetches "
        "it the moment the tab opens, so the tab is empty. Rebuild the folder: "
        "python3 scripts/compile_web_notebooks.py"
    )

    # 🔴 The tab has NO ⓘ, and neither does any other tab — all ten dots were
    # deleted on 2026-08-23 (Seth: "there's information icons literally
    # everywhere"). This used to assert the `tab.notebooks` registry key
    # existed; it asserts the tab itself instead, which is what this check was
    # ever really about. watch.py::check_infotips guards the dots staying gone.
    index_html = read(os.path.join(SHARED, "index.html"))
    assert 'data-tab="notebooks"' in index_html, (
        "index.html has no Notebooks tab — the view has no way in"
    )


def check_the_verdict_line_is_read_the_same_way_everywhere():
    """One grammar for "did this problem pass", in both readers.

    `dd_check` prints `✅ Problem <n> — k/k cases passed.` Two clients parse that
    line: the Chrome extension's `colab_focus.js` (reading a Colab output) and
    `notebook-view.js` (reading a kernel reply). They must agree exactly, and in
    particular BOTH must keep the em dash: without it the pattern matches the
    cell's own source, where `dd_check(480)` puts a bare number after the word
    Problem, and the app would record a verdict for a problem nobody ran.
    """
    view = read(os.path.join(HERE, "notebook-view.js"))
    focus_path = os.path.join(REPO, "extension", "content", "colab_focus.js")

    m = re.search(r"const VERDICT = /(.+?)/;", view)
    assert m, "notebook-view.js no longer defines a VERDICT pattern"
    verdict = m.group(1)
    assert "—" in verdict, (
        "notebook-view.js's VERDICT dropped the em dash — the pattern would match "
        "`dd_check(480)` in the cell's own source and record a verdict for a "
        "problem that was never graded"
    )
    assert r"(\d+)" in verdict, "VERDICT no longer captures the problem number"

    if not os.path.exists(focus_path):
        return  # the extension is not checked out beside the app
    focus = re.search(r"const RESULT = /(.+?)/g;", read(focus_path))
    assert focus, "colab_focus.js no longer defines a RESULT pattern"
    assert verdict == focus.group(1), (
        f"the two verdict readers have drifted: notebook-view.js reads "
        f"/{verdict}/ and colab_focus.js reads /{focus.group(1)}/ — the same "
        "grader output would be graded differently in the app and in Colab"
    )


def check_a_problem_is_recorded_once_per_visit():
    """Pressing a checker twice must not cost the learner twice.

    A learner debugging a failing drill runs its checker over and over; that is
    the loop working. If every press posted an attempt, the act of iterating
    would drive their own estimate down — the engine would be measuring
    persistence, not knowledge. So the verdict for a problem is recorded once
    per visit, and the guard has three moving parts that must stay together:
    the set, the early return, and the delete that lets a FAILED post be retried
    (a lost POST should cost one attempt, not the run the learner just did).
    """
    view = read(os.path.join(HERE, "notebook-view.js"))
    beacon = re.search(r"const _beacon = async \(.*?\n  \};", view, re.S)
    assert beacon, "notebook-view.js no longer has a _beacon"
    body = beacon.group(0)

    assert re.search(r"if \(state\.recorded\.has\(\w+\)\) return;", body), (
        "_beacon lost its already-recorded EARLY RETURN — every press of a "
        "checker would post another attempt and the engine would measure "
        "iteration rather than knowledge. (Asserted on the whole `if (…) "
        "return;`, not on the two halves: a guard whose body stopped returning "
        "reads as present and does nothing.)"
    )
    assert "state.recorded.add(" in body, "_beacon no longer marks a problem as recorded"
    assert body.index("state.recorded.add(") < body.index("recordLocalEval"), (
        "_beacon must claim the problem BEFORE awaiting the POST — two fast "
        "presses would otherwise both get past the guard"
    )
    assert "state.recorded.delete(" in body, (
        "_beacon no longer releases the problem when the POST fails — a dropped "
        "request would silently discard the attempt with no way to retry it"
    )
    assert "recorded: new Set()" in view, (
        "nothing mints a recorded set — opening a second lesson, or losing the "
        "kernel, must let a problem be recorded again"
    )
    # 🔴 The set belongs to the NOTEBOOK, not to this file. A module-global one
    # is shared by every notebook the tab renders, so a check finishing after
    # the learner opened another lesson writes into the set that lesson is
    # using — and the id it marks is silently unrecordable there.
    assert not re.search(r"^  let recorded\b", view, re.M), (
        "the recorded set is module-global again — it must live on the notebook "
        "(`state.recorded`), or a late run pollutes the next lesson's set"
    )
    # Only the graded cell may record. Any code cell could print something that
    # looks like a verdict; only `dd-q<n>-check` actually ran the grader.
    assert re.search(r'role === "check"[^\n]*_beacon|_beacon\(', view), \
        "notebook-view.js no longer calls _beacon"
    assert re.search(r'dataset\.role === "check" && !failed\) await _beacon', view), (
        "the beacon is no longer restricted to a check cell that succeeded — any "
        "cell printing a verdict-shaped line could record an attempt"
    )
    # 🔴 `PracticeAPI` is a top-level `const` in a classic script: global
    # LEXICAL scope, not a property of `window`. Reaching for it through
    # `window` reads `undefined`, and the only symptom is a console warning
    # while every verdict silently fails to reach the engine — the notebook
    # looks like it is grading and records nothing. Cost: one debugging pass on
    # 2026-08-19.
    assert "window.PracticeAPI.recordLocalEval" not in view, (
        "notebook-view.js calls recordLocalEval through `window.PracticeAPI` — "
        "that is undefined, and every recorded attempt would be swallowed by "
        "the beacon's own catch. Read it by name (`_practiceApi()`)."
    )
    assert "_practiceApi()" in body, (
        "_beacon no longer goes through _practiceApi() — the lexical/`window` "
        "distinction above is the whole reason that helper exists"
    )


def check_a_slow_run_cannot_touch_another_notebook():
    """A run belongs to the notebook it STARTED in, not the one on screen now.

    Running a cell is asynchronous and the learner is not. A checker cell can
    take seconds (it imports torch), and in those seconds they can press Back —
    which sets `current` to null, so the successful run ends in a caught
    TypeError — or open another lesson, which is worse: `current.checkerRan =
    true` then marks THAT lesson's checker as loaded, and the next kernel
    restart replays the FIRST lesson's checker source into the second lesson's
    session. Both are silent.

    The rule is mechanical, so it is checked mechanically: nothing in this file
    reads a field off the mutable `current`. A run captures the notebook it
    belongs to before its first await and uses that.
    """
    view = read(os.path.join(HERE, "notebook-view.js"))
    stray = [
        line.strip()
        for line in view.splitlines()
        if re.search(r"\bcurrent\.\w", line) and not line.lstrip().startswith(("*", "//"))
    ]
    assert not stray, (
        "notebook-view.js reads a field off the mutable `current`: "
        + " | ".join(stray)
        + " — capture the notebook in a local (`const state = current`) before "
        "the first await instead, or a cell that finishes after the learner "
        "navigates away writes into the wrong lesson"
    )
    run_cell = re.search(r"const _runCell = async \(.*?\n  \};", view, re.S)
    assert run_cell, "notebook-view.js no longer has a _runCell"
    assert "const state = current;" in run_cell.group(0), (
        "_runCell no longer captures the notebook it belongs to before running "
        "— see above; this is the capture the rest of the check assumes"
    )
    # 🔴 Holding the notebook is not enough on its own: `state.host` is
    # `#notebooks-host`, ONE element re-filled per notebook. A write through it
    # lands on whatever is on screen NOW, so the writes are gated on the
    # notebook still being the current one.
    banner = re.search(r"const _banner = .*?\n  \};", view, re.S)
    assert banner and "if (state !== current) return;" in banner.group(0), (
        "_banner no longer refuses to paint for a notebook that is not on "
        "screen — a run landing after the learner opened another lesson would "
        "write its banner into that lesson"
    )
    fresh = re.search(r"const _onFresh = async \((\w*)\)", view)
    assert fresh and fresh.group(1), (
        "_onFresh takes no argument again — it would restore the checker of "
        "whatever notebook happens to be open when the kernel restarts"
    )


def check_a_collapsed_cell_still_knows_its_own_source():
    """A cell inside a closed `<details>` must run what it says it runs.

    `innerText` is defined in terms of LAYOUT: on an element that is not being
    rendered — everything inside a collapsed `<details>`, which is every
    solution and every hints block — it returns the empty string. Running a
    solution that way executed an EMPTY program and reported "✓ ran
    successfully", after which `dd_check` below still said `solve` is not
    defined. So the source is carried on the node and only ever read back from
    the DOM as a fallback.
    """
    view = read(os.path.join(HERE, "notebook-view.js"))
    build = re.search(r"const _codeCell = .*?\n  \};", view, re.S)
    assert build, "notebook-view.js no longer has a _codeCell"
    assert "_ddSource =" in build.group(0), (
        "a code cell no longer carries its own source — a collapsed solution "
        "would run the empty string and report success"
    )
    source_of = re.search(r"const _sourceOf = .*?\n  \};", view, re.S)
    assert source_of, "notebook-view.js no longer has a _sourceOf"
    assert "_ddSource" in source_of.group(0), "_sourceOf no longer prefers the carried source"
    assert "textContent" in source_of.group(0), (
        "_sourceOf's DOM fallback is innerText-only again — that is the "
        "layout-dependent read this check exists to prevent"
    )
    assert re.search(r'addEventListener\("input"', view), (
        "nothing updates the carried source on an edit — a learner's typing "
        "would be ignored and the ORIGINAL cell would run instead"
    )


def check_the_notebook_never_falls_back_to_the_prefix_runner():
    """This surface is kernel-only, on purpose.

    `notebook.js::mount` fakes state by re-running every cell above the one
    clicked. That is fine for a six-cell lesson card and catastrophic here —
    clicking cell 600 of np-1 would run 599 cells, which is the exact cost pass
    1 was built to abolish. Signed out, the notebook READS and says why it
    cannot run; it does not quietly become O(cells) per click.
    """
    view = read(os.path.join(HERE, "notebook-view.js"))
    assert "LessonNotebook.runSource" in view, (
        "notebook-view.js no longer runs cells through LessonNotebook.runSource"
    )
    assert "LessonNotebook.mount" not in view and "_programUpTo" not in view, (
        "notebook-view.js reaches for the stateless prefix runner — a click on a "
        "late cell would re-run every cell above it"
    )
    assert "DeltaKernel" in view and "available()" in view, (
        "notebook-view.js no longer checks that a kernel is available before "
        "offering to run — signed-out learners would get a Run button that "
        "fails instead of an explanation"
    )
    notebook = read(os.path.join(HERE, "notebook.js"))
    assert "runSource" in notebook, "notebook.js no longer exports runSource"


def check_a_solution_stays_closed_until_asked():
    """The answer is one click away, never zero.

    Both the solution and the hints are compiled into the notebook as ordinary
    cells sitting directly under the problem. Rendered as plain cells they would
    simply be on screen, and the drill above them would be decoration.
    """
    view = read(os.path.join(HERE, "notebook-view.js"))
    solution = re.search(r"const _solutionCell = .*?\n  \};", view, re.S)
    assert solution, "notebook-view.js no longer has a _solutionCell"
    assert "_detailsCell" in solution.group(0), (
        "the solution cell is no longer wrapped in a <details> — the answer "
        "would be visible beside the problem it answers"
    )
    hints = re.search(r"const _hintsCell = .*?\n  \};", view, re.S)
    assert hints and "_detailsCell" in hints.group(0), (
        "the hints cell is no longer collapsed"
    )
    details = re.search(r"const _detailsCell = .*?\n  \};", view, re.S)
    assert details and "document.createElement(\"details\")" in details.group(0), (
        "_detailsCell no longer builds a <details> element"
    )
    assert "open" not in re.findall(r"el\.(\w+) = true", details.group(0)), (
        "_detailsCell opens the disclosure it just built"
    )
