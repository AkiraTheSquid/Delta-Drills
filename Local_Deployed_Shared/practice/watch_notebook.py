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

The last two checks are about the OTHER notebook on the practice page —
`notebook-editor.js`'s code cells. They live here rather than in
`watch_invariants.py` because that file is near its LOC line, and because a
cell editor is the same kind of thing this module already guards.
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


def check_the_solution_is_shown_and_the_hints_are_not():
    """The answer is on screen and runnable; a hint is still one click away.

    This check used to read `check_a_solution_stays_closed_until_asked` and
    enforced the opposite for the solution, on the reasoning that an answer on
    screen makes the drill above it decoration. Seth reversed it on 2026-08-24
    ("below your code answer it displays the actual solution that you can
    scroll down to and run"): this surface is not graded, nothing on it records
    an attempt except the line `dd_check` prints, and a closed disclosure on a
    656-cell page is not findable. The point of compiling the solution in as a
    CELL rather than as prose is that it runs.

    HINTS DID NOT MOVE, and that asymmetry is the whole check. A hint read
    before the attempt replaces the thinking it exists to prompt, and unlike
    the solution there is no "I am stuck, show me it working" that a hint
    answers better than the answer does.

    Both stay `<details>` either way, so the summary is a real collapse
    control — what changed is which one starts open.
    """
    view = read(os.path.join(HERE, "notebook-view.js"))
    solution = re.search(r"const _solutionCell = .*?\n  \};", view, re.S)
    assert solution, "notebook-view.js no longer has a _solutionCell"
    assert "_detailsCell" in solution.group(0), (
        "the solution cell is no longer wrapped in a <details> — it must keep "
        "the summary as a collapse control even though it starts open"
    )
    assert re.search(r"_detailsCell\([^)]*,\s*true\s*\)", solution.group(0), re.S), (
        "the solution disclosure no longer starts open — the answer is back to "
        "being a closed footer on a 656-cell page, which is what Seth asked to "
        "fix on 2026-08-24"
    )
    assert "_codeCell" in solution.group(0), (
        "the solution is no longer a runnable code cell — showing it as prose "
        "gives up the one thing this surface has over reading the answer"
    )
    hints = re.search(r"const _hintsCell = .*?\n  \};", view, re.S)
    assert hints and "_detailsCell" in hints.group(0), (
        "the hints cell is no longer collapsed"
    )
    assert not re.search(r"_detailsCell\([^)]*,\s*true\s*\)", hints.group(0), re.S), (
        "the hints disclosure now starts open — a hint read before the attempt "
        "replaces the thinking it was written to prompt; only the solution opens"
    )
    details = re.search(r"const _detailsCell = .*?\n  \};", view, re.S)
    assert details and "document.createElement(\"details\")" in details.group(0), (
        "_detailsCell no longer builds a <details> element"
    )
    assert re.search(r"if \(open\) el\.open = true;", details.group(0)), (
        "_detailsCell no longer opens on request — the `open` argument is what "
        "keeps the solution/hints asymmetry in ONE place instead of two builders"
    )


def check_only_one_module_patches_a_code_editors_value():
    """`editor.value = ...` fires no event, so runner.js makes it announce.

    Five modules assign a code editor's value. Nothing observes an assignment,
    so `runner.js::announceValueWrites` shadows the property on the element with
    a getter/setter over the native descriptor and dispatches
    `delta-editor-value-set`. The syntax overlay repaints on it and the cell
    re-measures its height on it.

    A SECOND module patching the same property is not a conflict, it is a silent
    breakage: `Object.defineProperty` on the instance REPLACES the descriptor,
    and the later patch reads `HTMLTextAreaElement.prototype`'s setter -- not the
    one already installed -- so its writes bypass the announcement entirely.
    Nothing throws. The overlay just keeps showing the previous question's code,
    or a long starter opens in a 96px box. One owner, many listeners.
    """
    owner = "runner.js"
    patchers = sorted(
        name for name in os.listdir(HERE)
        if name.endswith(".js")
        and re.search(r"""Object\.defineProperty\([^)]*?["']value["']""", read(os.path.join(HERE, name)))
    )
    assert patchers == [owner], (
        f"a code editor's `value` may only be patched in {owner}; found: "
        f"{patchers or 'nobody -- the announcement is gone and every overlay goes stale'}"
    )

    runner = read(os.path.join(HERE, owner))
    assert "delta-editor-value-set" in runner, (
        "runner.js patches `value` but no longer dispatches delta-editor-value-set "
        "-- the overlay and the cell height both go stale on every programmatic write"
    )
    for listener in ("code-highlight.js", "notebook-editor.js"):
        assert "delta-editor-value-set" in read(os.path.join(HERE, listener)), (
            f"{listener} stopped listening for delta-editor-value-set"
        )


def check_a_code_cell_grows_to_fit_its_own_code():
    """The cell is as tall as its code; the NOTEBOOK scrolls, not the cell.

    `resize` used to clamp at 420px, which put a scrollbar inside a box that had
    room to grow -- it hides how much code there is, and on a long starter the
    line being typed can sit under the fold. The 96px floor stays, because an
    empty cell still has to be a click target.

    The border term is guarded for the same reason it exists: `scrollHeight` is
    content+padding and EXCLUDES the border, while base.css is
    `box-sizing: border-box`, so a height assigned straight from `scrollHeight`
    leaves every cell 2px short of the content it was just measured against --
    permanently scrolling by exactly one hairline.
    """
    src = read(os.path.join(HERE, "notebook-editor.js"))
    body = re.search(r"const resize = \(editor\) => \{(.*?)\n  \};", src, re.S)
    assert body, "notebook-editor.js no longer defines `resize`"
    body = body.group(1)
    assert "Math.min" not in body, (
        "a ceiling is back in `resize` -- a code cell must grow to fit its code "
        "and let .practice-notebook do the scrolling"
    )
    assert "Math.max(96" in body, "the 96px floor is gone -- an empty cell collapses"
    assert re.search(r"scrollHeight \+ editor\.__deltaHBorders", body), (
        "the height `resize` assigns no longer ADDS the border width -- merely "
        "measuring it is not enough; with box-sizing: border-box a height taken "
        "straight from scrollHeight leaves every cell 2px short and scrolling "
        "inside itself"
    )



def check_the_solution_cell_is_never_the_learners_work():
    """The answer sits in the notebook but must never be graded or saved.

    `showSolution` appends the reference solution as a real `.notebook-cell`
    below the learner's own, so it lands under the code they typed and runs
    like any other cell (Seth, 2026-08-24: "it needs to render BELOW the code
    you typed"). That puts the answer key inside the same container every
    "what did the learner write" reader walks.

    The marker attribute is the whole defence, and the failure it prevents is
    silent in the worst way: `submissionCode()` joining the solution cell would
    POST the reference answer as the learner's own and grade it correct, which
    looks like the app working. `serialize()` would carry it into the saved
    draft, and `restore()` would lay it back out as an ordinary editable cell.
    """
    editor = read(os.path.join(HERE, "notebook-editor.js"))
    cells = re.search(r"const cells = \(\) =>.*?;", editor, re.S)
    assert cells, "notebook-editor.js no longer has a cells() accessor"
    assert ":not([data-solution-cell])" in cells.group(0), (
        "cells() no longer excludes the solution cell — submissionCode() would "
        "post the reference answer as the learner's own and grade it correct"
    )
    show = re.search(r"const showSolution = .*?\n  \};", editor, re.S)
    assert show, "notebook-editor.js no longer has a showSolution"
    assert "dataset.solutionCell" in show.group(0), (
        "the appended solution cell carries no data-solution-cell marker, so "
        "cells() cannot tell it apart from the learner's work"
    )
    assert "cellsHost.appendChild(cell)" in show.group(0), (
        "the solution is no longer appended LAST — a scratch cell added after "
        "the grade would render below the answer instead of above it"
    )
    reset = re.search(r"const reset = .*?\n  \};", editor, re.S)
    assert reset and "clearSolution()" in reset.group(0), (
        "reset() no longer clears the solution cell, so the previous question's "
        "answer stays on screen under the next question's code"
    )
    assert "showSolution" in read(os.path.join(HERE, "events.js")), (
        "nothing calls showSolution — the answer is built but never shown"
    )


def check_the_answer_only_lives_where_the_learner_can_see_it():
    """Four lifecycle rules, each one a way the answer goes wrong quietly.

    1. VISIBLE, not merely present. `#notebook-cells` stays in the DOM on
       surfaces that hide the whole right pane (a torch drill routed out to
       Colab, the Colab edition, an idle session). Appending there hides the
       left-rail fallback too — via `dd-solution-in-notebook` — so the learner
       ends up with NO answer anywhere, which is the bug this feature exists
       to fix. `renderFailedTests` makes the same test, and both must agree or
       the failures and the answer land in different columns.
    2. A correct RESUBMIT takes the answer away. Otherwise the solution to a
       question already solved sits under the learner's working code.
    3. Replacing the source clears the run. The output and its `[n]` marker
       would otherwise describe code no longer in the box.
    4. New cells go BEFORE the answer. `addCell` is reachable after a grade,
       and a plain append would bury the learner's new scratch cell under the
       solution.
    5. A RESTORED grade goes back into the notebook, on both paths. `applyResult`
       alone re-opens the left rail's copy (basic-mode.css keys it off
       `.result-incorrect`), so a reload or a session resume that stops there
       puts the answer back below the question — the layout this feature moved
       it out of.
    6. Appending is not showing. The answer lands below the fold of the pane,
       under cells as tall as the learner's own code, so the submit path has to
       scroll it into view or the screen looks unchanged by the grade.
    """
    editor = read(os.path.join(HERE, "notebook-editor.js"))
    show = re.search(r"const showSolution = .*?\n  \};", editor, re.S)
    assert show, "notebook-editor.js no longer has a showSolution"
    assert "getClientRects().length" in show.group(0), (
        "showSolution no longer checks that the notebook is VISIBLE — on a "
        "surface with a hidden right pane it appends the answer out of reach "
        "AND suppresses the rail copy, leaving the learner nothing"
    )
    ui = read(os.path.join(HERE, "ui.js"))
    failed = re.search(r"function renderFailedTests.*?\n\}", ui, re.S)
    assert failed and "getClientRects().length" in failed.group(0), (
        "renderFailedTests places the failed cases on element existence alone; "
        "it must use the same visibility test as showSolution"
    )
    for token, why in (
        ('outputOf(cell).textContent = ""',
         "a replaced solution keeps the output of the code it replaced"),
        ('execOf(cell).textContent = "[ ]"',
         "a replaced solution keeps the [n] marker of an older run"),
    ):
        assert token in show.group(0), f"showSolution: {why}"
    assert "clearSolution();" in show.group(0), (
        "showSolution refuses a hidden notebook without clearing an EARLIER "
        "answer — the suppression class outlives the visible cell and the rail "
        "fallback stays hidden with nothing to fall back to"
    )
    add = re.search(r"function addCell\(.*?\n  \}", editor, re.S)
    assert add and "insertBefore(cell, feedbackBoundary())" in add.group(0), (
        "addCell no longer inserts above the graded feedback, so a scratch "
        "cell added after a wrong grade renders below the failed cases"
    )
    boundary = re.search(r"const feedbackBoundary = .*?;", editor, re.S)
    assert boundary and "#failed-tests-block" in boundary.group(0) \
        and "[data-solution-cell]" in boundary.group(0), (
        "the insertion boundary no longer covers BOTH the failed-case block "
        "and the answer; a new cell would land between them"
    )
    events = read(os.path.join(HERE, "events.js"))
    assert "clearSolution" in events, (
        "a correct resubmission never clears the solution cell — the answer to "
        "a question the learner just got right stays on screen under their code"
    )

    # 5. Both restore paths rebuild the feedback in the notebook, and both do it
    #    AFTER the draft is back: DeltaNotebook.restore runs reset, and reset
    #    opens with clearSolution(), so an answer put back first is swept away
    #    by the code that returns the learner's own cells.
    restore = re.search(
        r"function restoreGradedFeedbackInNotebook.*?\n\}", ui, re.S)
    assert restore, (
        "ui.js no longer defines restoreGradedFeedbackInNotebook — the reload "
        "and resume paths have nothing shared to rebuild a graded review with"
    )
    assert "showSolution" in restore.group(0) and "renderFailedTests" in restore.group(0), (
        "the restore helper no longer puts BOTH halves of the review back; a "
        "restored grade must read the same as a live one"
    )
    assert "recordGradedDetail" in restore.group(0), (
        "a RESTORED review no longer re-arms the graded-detail record — rating "
        "a resumed question then saves a pendingFeedback with no failed cases, "
        "so the next reload shows a verdict with no reason for it"
    )
    assert "gradedDetailFor(q.question_id)" in events, (
        "the rating step reads the graded detail without the question-id guard; "
        "one question's failed cases can be saved next to another's verdict"
    )
    assert "scrollToSolution" in restore.group(0), (
        "a restored review appends the answer under a restored draft without "
        "scrolling to it — a long attempt puts it below the fold again, which "
        "is the invisible-answer bug arriving through the resume door"
    )
    pending = re.search(r"function applyPendingFeedbackState.*?\n\}", ui, re.S)
    assert pending and "restoreGradedFeedbackInNotebook" in pending.group(0), (
        "a reload onto a rated question repaints the verdict without the "
        "notebook feedback, so the left rail shows the answer again"
    )
    timer = read(os.path.join(HERE, "timer.js"))
    review = re.search(r"const _restoreReview = .*?\n  \};", timer, re.S)
    assert review and "restoreGradedFeedbackInNotebook(" in review.group(0), (
        "resuming a paused review no longer restores the notebook answer — "
        "this is the exact regression the Resume button showed: the solution "
        "reappears under the question instead of under the code"
    )
    draft_at = review.group(0).index("_restoreDraft")
    restore_at = review.group(0).index("restoreGradedFeedbackInNotebook")
    assert draft_at < restore_at, (
        "_restoreReview rebuilds the answer BEFORE restoring the draft; "
        "DeltaNotebook.restore -> reset -> clearSolution() then deletes it"
    )

    # 6. The submit path scrolls the pane to the answer, and does it by moving
    #    that pane rather than scrollIntoView, which drags every scrollable
    #    ancestor (the page, the left rail) along with it.
    reveal = re.search(r"const scrollToSolution = .*?\n  \};", editor, re.S)
    assert reveal, "notebook-editor.js no longer defines scrollToSolution"
    assert "scrollIntoView" not in reveal.group(0), (
        "scrollToSolution uses scrollIntoView, which scrolls the document and "
        "the left rail too; scroll .practice-notebook itself"
    )
    assert "getClientRects().length" in reveal.group(0), (
        "scrollToSolution scrolls a pane it never checked is on screen"
    )
    # `DeltaNotebook?.` and not the bare name: events.js also calls an
    # unrelated `DDColab.revealSolution`, and matching a bare identifier is how
    # an invariant quietly starts passing on the wrong call.
    assert "DeltaNotebook?.scrollToSolution" in events, (
        "submitting a wrong answer appends the solution without scrolling to "
        "it — it lands below the fold and the learner never learns it is there"
    )
