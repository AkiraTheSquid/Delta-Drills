"""watch_lessons.py — lesson-page checks (notebook, gate, ladder, clock).

Split out of `watch.py` (Modulario, 2026-08-19). These are the checks that drive
a real `node -e` probe over the lesson modules rather than reading source for a
pattern, plus the two routing checks around them. Unchanged by the split; still
run from `watch.py`'s list, which fails if one defined here is missing from it.
"""
import os
import re
import subprocess

from watch_common import HERE, SHARED, read


def check_lesson_code_can_actually_run():
    """A lesson's Run button must reach a runtime that HAS torch.

    The whole course is torch since the July dialect conversion, and Pyodide
    cannot import it. `runner.runSnippet` routes on two signals, and both of
    them silently went stale: `lessons.js` advertised the lesson's library as
    "numpy" (written before the conversion), and the source sniff was handed
    `notebook.js`'s generated harness, which packs every cell into a Python
    string literal where a line-anchored regex cannot see the import. With both
    lying, every Run button on every lesson answered with a
    ModuleNotFoundError traceback — for signed-in learners too, whose backend
    runner has torch preimported and was simply never asked.

    Neither signal announces itself when it breaks, so both are pinned here.
    """
    read = lambda name: open(os.path.join(HERE, name), encoding='utf-8').read()
    lessons = read('lessons.js')
    ctx = lessons.split('_runtimeContext = (page) => ({', 1)[-1].split('})', 1)[0]
    assert 'primary_library: "torch"' in ctx, (
        "_runtimeContext no longer declares the lesson as torch — questionIsTorch "
        "reads this field, and anything else routes torch lesson code to Pyodide, "
        "which cannot import torch"
    )
    notebook = read('notebook.js')
    call = notebook.split('DeltaRunner.runSnippet(', 1)[-1].split('});', 1)[0]
    assert 'source:' in call, (
        "the lesson notebook runs a generated harness without telling the runner "
        "what the learner actually wrote — the torch sniff then reads scaffolding"
    )
    runner = read('runner.js')
    assert 'TORCH_IMPORT.test(source || code' in runner, (
        "runSnippet sniffs torch on the program it was handed rather than on the "
        "source, so a wrapped cell can talk itself onto Pyodide"
    )


def check_colab_lesson_goes_to_the_notebook():
    """On the Colab edition the `worked` rung is READ in the notebook.

    The published notebook already carries the lesson's prose, its runnable
    blocks, the worked example, the problem, the hints and the solution, in
    that order and against a real torch runtime. A second copy in the panel put
    the reading on the left and the work on the right — the split this edition
    exists to close.

    Three things have to hold together or the rail is worse than the page it
    replaced: the href has to come from `DDColab.hrefForKc` (a concept anchor
    the generator minted, never a slug guessed here), an absent notebook has to
    fall through to the full in-panel lesson, and the panel must steer the tab
    rather than wait to be clicked.
    """
    read = lambda name: open(os.path.join(HERE, name), encoding='utf-8').read()
    lessons = read('lessons.js')
    assert 'dd.hrefForKc(page.kp.kc, page.step && page.step.exposureKey)' in lessons, (
        "the lesson rail no longer resolves its notebook through "
        "DDColab.hrefForKc — a slug guessed locally is an anchor that drifts, "
        "and without the page's exposure key a segmented KP opens all of its "
        "concepts under a topbar that says the learner is on one of them"
    )
    page_html = lessons.split('const _pageHtml = (page) => {', 1)[-1].split('\n  };', 1)[0]
    assert 'if (colabHref) return _colabPageHtml' in page_html, (
        "_pageHtml no longer routes the Colab edition to the rail, so the "
        "panel draws the lesson the notebook already contains"
    )
    assert '_colabLessonHref(page)' in page_html, (
        "the rail is chosen without asking whether THIS concept has a notebook "
        "— the ~unpublished ones must fall through to the full lesson"
    )
    assert 'DDColab.openNotebook(colabHref)' in lessons, (
        "the lesson rail never steers the notebook, so the learner is told to "
        "read something that is not on screen"
    )
    css = open(os.path.join(SHARED, 'styles', 'practice', 'colab-edition.css'),
               encoding='utf-8').read()
    assert '.lesson-colab-card' in css, (
        "colab-edition.css lost the lesson rail's styling — the card renders "
        "as unstyled prose in a panel that has no other lesson layout left"
    )




def check_a_resumed_clock_matches_the_break():
    """Resuming keeps the time left; coming back much later restarts the step.

    Two failure modes, opposite and both silent. Always restoring `remaining`
    makes the strict timer optional: pause at 00:05, come back tomorrow, get
    five seconds — so the learner learns to pause and never resume. Always
    restarting it makes pause the way to buy a fresh five minutes on any
    question that is going badly.

    The rule is therefore about the LENGTH OF THE BREAK, and it has to be
    evaluated when Resume is clicked, not when the page loaded — the resume
    panel can sit on screen for an hour, and a snapshot judged fresh at load
    would still be handing back a clock that expired while it was being read.

    Run rather than pattern-matched. The helpers are pure functions of a
    snapshot, so the actual shipped arithmetic is lifted out and exercised on
    the cases that matter, including the two that are easy to get backwards: a
    missing timestamp (an older bundle's snapshot — unknowable break, so treat
    it as long) and a clock that has gone backwards.
    """
    src = open(os.path.join(HERE, 'timer.js'), encoding='utf-8').read()
    grace = re.search(r"const RESUME_GRACE_SECS = (\d+);", src)
    assert grace, "RESUME_GRACE_SECS is gone — the resume rule has no window"
    start = src.index("  const _awaySecs = (saved) =>")
    end = src.index("  const _resumeSummary =")
    helpers = src[start:end]
    assert "_effectiveRemaining" in helpers, (
        "the resume clock is no longer computed from the snapshot's age"
    )
    assert "_effectiveRemaining(pausedState)" in src.split("const _resumeCore", 1)[-1], (
        "_resumeCore restores a raw `remaining` again — the break's length has "
        "to be read at the moment of resuming, not at page load"
    )

    probe = f"""
const RESUME_GRACE_SECS = {grace.group(1)};
{helpers}
// The clock is FROZEN for the probe. `savedAt` is an ISO string truncated to
// milliseconds and `_awaySecs` re-reads the wall clock a moment later, so a
// live `Date.now()` puts the exact boundary case a few milliseconds past the
// window and the assertion flips depending on how busy the machine is. The
// boundary is the case worth testing; it has to be testable exactly.
const NOW = Date.now();
Date.now = () => NOW;
const snap = (over) => ({{
  phase: "answer", answerSecs: 300, reviewSecs: 120, remaining: 60,
  savedAt: new Date(NOW - over * 1000).toISOString(),
}});
const eq = (got, want, why) => {{
  if (JSON.stringify(got) !== JSON.stringify(want)) {{
    console.error(`FAIL ${{why}}: got ${{JSON.stringify(got)}} want ${{JSON.stringify(want)}}`);
    process.exit(1);
  }}
}};
eq(_effectiveRemaining(snap(1)), {{secs: 60, restarted: false}},
   "straight back must keep the time left");
eq(_effectiveRemaining(snap(RESUME_GRACE_SECS)), {{secs: 60, restarted: false}},
   "the grace boundary itself must still resume");
eq(_effectiveRemaining(snap(RESUME_GRACE_SECS + 1)), {{secs: 300, restarted: true}},
   "past the window the ANSWER step restarts at its own limit");
const review = {{...snap(RESUME_GRACE_SECS + 1), phase: "review"}};
eq(_effectiveRemaining(review), {{secs: 120, restarted: true}},
   "a review-phase break must restart the REVIEW limit, not the answer one");
eq(_effectiveRemaining({{...snap(0), savedAt: null}}), {{secs: 300, restarted: true}},
   "an unknowable break must be treated as a long one, never as a fresh resume");
eq(_effectiveRemaining(snap(-99999)), {{secs: 60, restarted: false}},
   "a clock that jumped backwards must not restart the step");
"""
    proc = subprocess.run(["node", "-e", probe], capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def check_the_gate_teaches_one_concept_then_drills_it():
    """One concept per visit, and the KP's own key only at the end.

    A KP teaches up to six separate ideas and the gate used to render all of
    them back to back before handing over a single question — a learner who has
    read three things and practised one. The loop only works if three things
    hold: a visit builds ONE page, the key posted is that concept's, and the
    KC's own key (which every later gate reads as "the whole KP is done") is
    written only alongside the last one. Post it early and concepts 2 and 3 are
    credited unread and never offered again.
    """
    lessons = read(os.path.join(HERE, "lessons.js"))
    build = lessons.split("const _buildPages = (steps) => {", 1)
    assert len(build) == 2, (
        "_buildPages no longer takes steps — it is back to expanding a whole "
        "KP into pages, which is the wall of text the loop replaced"
    )
    body = build[1].split("\n  };", 1)[0]
    assert "pages.push(" in body and body.count("pages.push(") == 1, (
        "_buildPages pushes more than one page per step — a visit must owe "
        "exactly one concept"
    )
    assert "step.segmentIndex" in body, (
        "_buildPages ignores the step's concept index, so every visit would "
        "re-teach the first concept"
    )

    click = lessons.split("button.onclick = () => {", 1)[-1].split("\n        };", 1)[0]
    assert "page.step.exposureKey" in click, (
        "the continue button no longer records the concept that was read"
    )
    assert "page.lastOfKp && page.step.exposureKey !== page.kp.kc" in click, (
        "the KP's own exposure key is written without checking this is the "
        "LAST concept — that credits every remaining concept unread"
    )
    kc_write = click.index("keys.push(page.kp.kc)")
    guard = click.index("page.lastOfKp && page.step.exposureKey !== page.kp.kc")
    assert guard < kc_write, "the KP key is pushed before its guard is tested"
    assert "taught.push(page.kp.kc);" in click, (
        "no worked-example credit on this page — the drill behind it would be "
        "served on the `worked` rung with no blanks in it"
    )


def check_the_fifth_rung_is_shown_not_stored():
    """`integrated` is a display rung. It must never reach the ladder record.

    Every stored attempt is filed under one of the backend's four stage names
    and the promotion arithmetic reads them back. A fifth stored name would
    either rewrite that history or create a rung nothing can be promoted out
    of, so the ladder gets the extra section and `record_ladder_outcome` never
    sees it.
    """
    ladder_js = read(os.path.join(HERE, "stage-ladder.js"))
    assert '"integrated"' in ladder_js or "integrated:" in ladder_js, (
        "the stage ladder has no integrated section — the fifth rung is gone"
    )
    ladder = read(os.path.join(HERE, "ladder.js"))
    assert 'question.ladder_integrated ? "integrated" : stage' in ladder, (
        "ladder.js no longer reports the fifth rung to the ladder"
    )
    submit = [l for l in ladder.splitlines() if "ladder_stage" in l and "=" in l]
    for line in submit:
        assert '"integrated"' not in line, (
            f"integrated is being written into ladder_stage — {line.strip()!r}"
        )


# ── Notebook kernel ───────────────────────────
# The stateful path is the point of the default edition (cells share a
# namespace the way Colab's do), and its fallback is what keeps guests and an
# unreachable backend working. Both halves have to stay wired.
def check_the_notebook_kernel_has_a_fallback():
    notebook = read(os.path.join(HERE, "notebook.js"))
    kernel = read(os.path.join(HERE, "kernel.js"))
    index = read(os.path.join(SHARED, "index.html"))

    assert "window.DeltaKernel" in kernel, "kernel.js no longer exports window.DeltaKernel"
    for path in ("/api/practice/kernel/exec", "/api/practice/kernel/reset"):
        assert path in kernel, f"kernel.js lost the {path} endpoint"

    # Load order: notebook.js reads window.DeltaKernel, so the client has to be
    # on the page first. Both are classic scripts, so this is source order.
    k_at, n_at = index.find("practice/kernel.js"), index.find("practice/notebook.js")
    assert k_at != -1, "index.html does not load practice/kernel.js"
    assert n_at != -1, "index.html does not load practice/notebook.js"
    assert k_at < n_at, "practice/kernel.js must load before practice/notebook.js"

    # A kernel that cannot be had must degrade to the prefix-replay runner, not
    # to a dead Run button. Losing either half breaks a whole class of learner:
    # the first breaks guests, the second breaks everyone else's state.
    assert "_runOnKernel" in notebook, "notebook.js lost its kernel path"
    assert "DeltaRunner.runSnippet" in notebook, (
        "notebook.js lost the stateless fallback — guests and an unreachable "
        "backend would have no way to run a cell"
    )
    assert "_programUpTo" in notebook, (
        "notebook.js lost the prefix replay — a kernel that was evicted between "
        "two clicks would answer the next cell with a NameError"
    )
    # The server installs the cell harness only on a kernel it had to create,
    # so the client has to keep sending it with every cell.
    assert "bootstrap: HARNESS" in notebook, (
        "notebook.js no longer sends the cell harness as `bootstrap` — a "
        "re-created kernel would have no _delta_cell to call"
    )

    # The replay ENDS with the clicked cell, so the clicked cell must not have
    # already run on the way to learning the kernel was fresh. A cell that
    # appends to a list is not the same cell run twice.
    assert "skipOnFresh: index > 0" in notebook, (
        "notebook.js no longer asks the server to skip a cell it is about to "
        "replay — every non-first cell runs twice on a fresh kernel"
    )
    assert "skip_on_fresh" in kernel, (
        "kernel.js drops skipOnFresh instead of sending it — the server cannot "
        "know not to run a cell the client is about to replay"
    )

    # Failure is the backend's `success` flag. stderr is not it in either
    # direction: a DeprecationWarning is not a failure, and `sys.exit(1)`
    # prints nothing at all.
    assert "reply.ok === false" in notebook, (
        "notebook.js is inferring failure from stderr again — a warning would "
        "paint a passing cell red, and a silent exit would read as success"
    )


# ── Run all checks ────────────────────────────
