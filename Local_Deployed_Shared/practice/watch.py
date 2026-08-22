"""watch.py — health checks for practice

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.

The checks are split across three modules to stay under Modulario's LOC line:
this file keeps the file/API/threshold checks, `watch_invariants.py` holds the
invariant sweep and `watch_lessons.py` the lesson-page probes. 🔴 The runner
list below is the ONLY thing that decides what actually runs — a check that
exists and is not in it has silently stopped running twice before now, so the
guard at the bottom fails the run if any `check_*` in the sibling modules is
missing from the list.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import watch_invariants
import watch_lessons
import watch_notebook
from watch_common import (  # noqa: F401 — re-exported for anything importing watch
    HERE, SHARED, REQUIRED_JS, REQUIRED_DOCS, REQUIRED_ASSETS, read,
)
from watch_invariants import check_invariants
from watch_notebook import (
    check_a_slow_run_cannot_touch_another_notebook,
    check_a_collapsed_cell_still_knows_its_own_source,
    check_a_problem_is_recorded_once_per_visit,
    check_a_solution_stays_closed_until_asked,
    check_the_notebook_never_falls_back_to_the_prefix_runner,
    check_the_notebook_view_is_loaded_after_what_it_calls,
    check_the_verdict_line_is_read_the_same_way_everywhere,
)
from watch_lessons import (
    check_a_resumed_clock_matches_the_break,
    check_colab_lesson_goes_to_the_notebook,
    check_lesson_code_can_actually_run,
    check_the_fifth_rung_is_shown_not_stored,
    check_the_gate_teaches_one_concept_then_drills_it,
    check_the_notebook_kernel_has_a_fallback,
)



# ── Import checks ──────────────────────────────
# Folder is JS; verify required source files exist and parse-adjacent invariants
# (no Python modules to import directly).
def check_imports():
    missing = [f for f in REQUIRED_JS if not os.path.isfile(os.path.join(HERE, f))]
    assert not missing, f"missing required JS files: {missing}"
    missing_docs = [f for f in REQUIRED_DOCS if not os.path.isfile(os.path.join(HERE, f))]
    assert not missing_docs, f"missing required docs: {missing_docs}"
    missing_assets = [p for p in REQUIRED_ASSETS if not os.path.isfile(p)]
    assert not missing_assets, f"missing sibling assets: {missing_assets}"


# ── Public API checks ─────────────────────────


# fallback rendering helpers are present, and the question JSON is parseable.
def check_public_api():
    runner = read(os.path.join(HERE, "runner.js"))
    assert "buildPyodidePreamble" in runner, "runner.js missing buildPyodidePreamble"
    assert 'context: "practice-editor"' in runner and "kernel.runCell" in runner, (
        "Practice editor no longer uses persistent kernel cells"
    )
    assert "runtimeResetBtn" in runner and "DeltaKernel?.reset" in runner, (
        "Practice editor lost restart-runtime control"
    )
    notebook_editor = read(os.path.join(HERE, "notebook-editor.js"))
    for token in ("addCell", "runCell", "serialize", "restore", "submissionCode"):
        assert token in notebook_editor, f"practice notebook lost {token}"
    assert "DeltaRunner.runSnippet" in notebook_editor, (
        "per-cell Run no longer reaches shared runtime"
    )
    index = read(os.path.join(SHARED, "index.html"))
    assert 'id="notebook-cells"' in index and 'id="notebook-add-cell"' in index
    assert index.find("practice/runner.js") < index.find("practice/notebook-editor.js"), (
        "notebook-editor.js must load after runner API"
    )
    events = read(os.path.join(HERE, "events.js"))
    assert "DeltaNotebook?.submissionCode()" in events, (
        "grading ignores code outside first notebook cell"
    )
    timer = read(os.path.join(HERE, "timer.js"))
    assert "DeltaNotebook?.serialize()" in timer and "DeltaNotebook.restore" in timer, (
        "session pause/resume no longer preserves notebook cells"
    )
    assert "Array.isArray(saved.draft.cells)" in timer, (
        "saved-session parser drops multi-cell notebook draft on reload"
    )
    assert "holdClock" in timer and "releaseClock" in timer
    assert 'holdClock("problem-feedback-note")' in events
    assert 'releaseClock("problem-feedback-note")' in events
    index_html = read(os.path.join(SHARED, "index.html"))
    assert 'id="page-diagnostic"' in index_html and 'data-tab="diagnostic"' in index_html
    assert 'class="tab has-info" data-tab="diagnostic"' in index_html
    assert 'id="diagnostic-workspace-host"' in index_html
    assert "Continue diagnostic in Practice" not in index_html
    # An unfinished placement must not cost the learner the Practice tab.
    diagnostic_page = read(os.path.join(HERE, "diagnostic-page.js"))
    # These are substring checks over source, so they are spelled to match the
    # mechanism rather than a word that could survive in a comment: the guard
    # names the definition AND the call site that has to consume it.
    assert "setPracticeTabDisabled" not in diagnostic_page and (
        ".disabled = true" not in diagnostic_page
    ), (
        "an active placement must never disable the Practice tab: no :disabled "
        "style exists, so the tab looks live and silently eats the click"
    )
    assert "const diagnosticOnScreen =" in diagnostic_page and (
        "running && diagnosticOnScreen()" in diagnostic_page
    ), (
        "the practice workspace may live in #page-diagnostic only while that page "
        "is on screen — delta:practice-state-changed fires from any tab, and keying "
        "on the placement alone yanks the workspace out of a visible Practice tab"
    )
    assert 'practicePage.classList.add("hidden")' not in diagnostic_page and (
        "practicePage.hidden = true" not in diagnostic_page
    ), (
        "app.js owns page visibility; hiding #page-practice from here blanks the tab"
    )
    assert index_html.count('id="self-report-row"') == 1
    assert 'maxlength="5000"' in index_html and '<textarea class="problem-feedback-note"' in index_html
    assert "ensureArenaNumbersInPyodide" in runner, "runner.js missing ensureArenaNumbersInPyodide"

    ui_js = read(os.path.join(HERE, "ui.js"))
    assert "renderQuestionImports" in ui_js, "ui.js missing imports renderer"

    visuals = read(os.path.join(HERE, "visuals.js"))
    assert "renderQuestionVisual" in visuals, "visuals.js missing renderQuestionVisual"
    assert "renderFallbackImage" in visuals, "visuals.js missing PNG fallback (renderFallbackImage)"
    assert "getArenaNumbersPngCandidates" in visuals, "visuals.js missing PNG candidate resolver"

    unlock = read(os.path.join(HERE, "arena-unlock.js"))
    for needle in (
        "window.ArenaUnlock", "tryShow",
        # showFor is the manual-launch API consumed by the Targeted Practice
        # "Practice this problem" button. If this drops, the TP review-mode
        # action button silently no-ops with a console.warn.
        "showFor",
        "arena-unlock-page", "arena-unlock-card", "arena-unlock-continue-btn",
        ".practice-container",  # the view that gets swapped out on show
    ):
        assert needle in unlock, f"arena-unlock.js missing required symbol: {needle!r}"
    # The Colab href must fall back to currentEx.notebookPath so manual
    # launches from Targeted Practice point at the right notebook, not the
    # hardcoded 0.0 prereqs notebook used by the auto-unlock flow.
    assert re.search(r"currentEx[^\n]*notebookPath", unlock), (
        "arena-unlock.js colabHrefForUnlock lost the currentEx.notebookPath fallback "
        "— Targeted Practice will land on the wrong Colab notebook"
    )
    # The _targetedPractice flag must gate markArenaExerciseShown so manually
    # launched exercises remain re-practicable from the TP review page.
    assert "_targetedPractice" in unlock, (
        "arena-unlock.js no longer honors the _targetedPractice flag "
        "— Targeted-Practice exercises will get marked one-shot-shown"
    )
    # The unlock-page CSS lives in practice/arena-unlock.css (modular — kept
    # separate from practice.css so the unlock view can be ripped out when
    # the real concept-graph backend ships without touching the question UI).
    unlock_css = read(os.path.join(HERE, "arena-unlock.css"))
    for needle in (".arena-unlock-page", ".arena-unlock-card", ".arena-unlock-continue-btn"):
        assert needle in unlock_css, f"arena-unlock.css missing required selector: {needle!r}"
    # The unlock view used to be authored inline in index.html. It was
    # extracted into practice/arena-unlock-dom.js (script-eval-time DOM
    # injection) to keep index.html under the per-file LOC ceiling.
    # Verify the DOM module still ships the mount + the load order puts
    # it BEFORE arena-unlock-timer.js and arena-unlock.js (both query the
    # injected ids at IIFE-eval time).
    dom_mount = read(os.path.join(HERE, "arena-unlock-dom.js"))
    assert 'id="arena-unlock-page"' in dom_mount, (
        "arena-unlock-dom.js no longer injects #arena-unlock-page — the "
        "interstitial mount is the controller's hard dependency"
    )
    assert "page-practice" in dom_mount, (
        "arena-unlock-dom.js must mount into #page-practice "
        "(so the in-tab view swap with .practice-container works)"
    )
    index_html = read(os.path.join(os.path.dirname(HERE), "index.html"))
    dom_pos = index_html.find('src="practice/arena-unlock-dom.js')
    timer_pos = index_html.find('src="practice/arena-unlock-timer.js')
    ctrl_pos = index_html.find('src="practice/arena-unlock.js')
    assert dom_pos != -1, 'index.html missing <script src="practice/arena-unlock-dom.js">'
    assert timer_pos != -1, 'index.html missing <script src="practice/arena-unlock-timer.js">'
    assert ctrl_pos != -1, 'index.html missing <script src="practice/arena-unlock.js">'
    assert dom_pos < timer_pos and dom_pos < ctrl_pos, (
        "arena-unlock-dom.js must load BEFORE arena-unlock-timer.js and "
        "arena-unlock.js — those controllers query the injected ids at "
        "IIFE-eval time and would silently no-op otherwise"
    )
    events_js = read(os.path.join(HERE, "events.js"))
    assert "ArenaUnlock" in events_js, "events.js no longer routes through ArenaUnlock.tryShow"
    assert "_loadNextPracticeQuestion" in events_js, "events.js missing _loadNextPracticeQuestion helper"

    # Resumable-session lifecycle (timer.js PracticeSession) — every hook must
    # stay wired or a timer keeps running across a boundary, a stale grade
    # hijacks a new session, or pause loses the learner's current review.
    timer_js = read(os.path.join(HERE, "timer.js"))
    for method in (
        "onQuestionRendered", "pauseForGrading", "pauseForAdvance",
        "recordReviewResult", "resumeAnswerPhase", "beginReviewPhase",
        "pause", "resume", "discard", "hasSavedQuestion",
        "shouldFinishInsteadOfAdvance",
    ):
        assert method in timer_js, f"timer.js PracticeSession lost method: {method}"
    for needle in (
        "SESSION_STATE_VERSION", "getPracticeStorageKey", "pagehide",
        "questionId", "draft", "remaining", "review",
    ):
        assert needle in timer_js, f"timer.js resumable snapshot lost field/hook: {needle}"
    assert 'state.phase !== "grading"' in timer_js, (
        "timer.js beginReviewPhase lost its grading-phase guard — a stale grade "
        "after End session → Start session hijacks the new session's first question"
    )
    assert "PracticeSession.onQuestionRendered" in ui_js, (
        "ui.js renderQuestion no longer starts the session answer countdown"
    )
    init_js = read(os.path.join(HERE, "init.js"))
    assert "PracticeSession.hasSavedQuestion" in init_js, (
        "init.js no longer preserves a saved visual coding question for resume"
    )
    for hook in (
        "PracticeSession.pauseForGrading",
        "PracticeSession.pauseForAdvance",
        "PracticeSession.recordReviewResult",
        "PracticeSession.beginReviewPhase",
        "PracticeSession.shouldFinishInsteadOfAdvance",
    ):
        assert hook in events_js, f"events.js lost session hook: {hook}"
    assert "PracticeAPI.currentQuestion !== q" in events_js, (
        "events.js submit handler lost the stale-grade guard — a grade landing "
        "after Skip/End-session repaints the wrong question"
    )

    for marker in (
        'id="session-resume-panel"', 'id="session-resume-btn"',
        'id="session-discard-btn"', 'id="session-pause-btn"',
    ):
        assert marker in index_html, f"index.html missing resumable-session control: {marker}"

    questions_path = os.path.join(SHARED, "questions_structured.json")
    data = json.loads(read(questions_path))
    assert isinstance(data, list) and data, "questions_structured.json is empty or not a list"
    sample = next((q for q in data if q.get("exercise", {}).get("function_name")), None)
    assert sample is not None, "questions_structured.json has no function-backed practice questions"


# ── Invariant checks ──────────────────────────
# Structural rules that must remain true:
#  1. No leftover modulario template markers.
#  2. Every function-implementation question has a canonical_solution
#     (runtime_dependencies was dropped from the generator schema 2026-07-06).
#  3. The id-27 einsum entry has a non-null canonical_solution (was a known bug;
#     regressed once when the notebook's last cell had no trailing solution
#     markdown — now covered by the solutions-notebook fallback in
#     extract_arena_prereqs.py).
#  4. The Pyodide preamble's injected names match the doc's "Always injected" set.


def check_promotion_threshold_matches_the_backend():
    """The mark on the estimate bar must be where promotion ACTUALLY happens.

    `stage-ladder.js` fills the active section as far as the concept's interval
    has carried it toward the next rung — reach the end and the next question
    comes with less support.
    That number is a copy of `app/kc_graph.py`'s `PROMOTE_LO`, held in a second
    language because the rung is decided server-side and drawn client-side.

    A copy nobody checks is a copy that drifts, and this one drifts SILENTLY in
    the worst direction: the learner clears a mark the app drew for them and
    nothing happens, or the rung changes while the mark is still ahead of them.
    Either way the app has told them a rule it does not follow, which is worse
    than drawing no mark at all.

    ⚠️ The two speak different vocabularies. The backend's `partial` is the
    display's `example` — see STAGE_ALIASES. This check translates rather than
    comparing keys, because comparing keys is how the numbers would end up
    swapped and still "matching". They WERE swapped, until 2026-08-06: the
    display's rung names had `faded` and `worked` the wrong way round, so each
    dot described the other one's rung and this check compared the two wrong
    numbers to each other and passed. The display rung is called `example` and
    not `worked` precisely so that no key can be read in both vocabularies.

    The right long-term fix is for the ladder response to carry its own
    threshold so there is one runtime authority; codex flagged the same thing on
    2026-08-03. Until then, this.
    """
    ladder_js = read(os.path.join(HERE, "stage-ladder.js"))
    graph = os.path.join(
        HERE, "..", "..", "This-Directory-Only", "backend", "app", "kc_graph.py")
    if not os.path.exists(graph):
        return  # backend not checked out beside the frontend — nothing to compare
    backend = read(graph)

    def _floats(src, name):
        block = re.search(rf"{name}\s*=\s*\{{(.*?)\}}", src, re.S)
        assert block, f"could not find {name}"
        return {k: float(v) for k, v in re.findall(
            r'"(\w+)"\s*:\s*([0-9.]+)', block.group(1))}

    promote_lo = _floats(backend, "PROMOTE_LO")
    promote_at = {k: float(v) for k, v in re.findall(
        r"(\w+):\s*([0-9.]+),", re.search(
            r"PROMOTE_AT\s*=\s*\{(.*?)\}", ladder_js, re.S).group(1))}

    # backend rung -> the rung the display calls it
    for backend_stage, displayed in (("faded", "faded"), ("partial", "example")):
        assert backend_stage in promote_lo, f"backend lost PROMOTE_LO[{backend_stage}]"
        assert displayed in promote_at, (
            f"stage-ladder.js lost the {displayed!r} threshold — that section "
            f"would draw no progress while the backend still promotes on one"
        )
    # The OTHER route out of a rung, and the only one the learner can watch
    # move: a run of `_PROMOTE_STREAK` correct answers promotes on its own,
    # whatever the twenty-question window says. The section fills to whichever
    # route is further along, so dropping the run leaves a bar that sits still
    # through three correct answers and then jumps a whole rung with no warning
    # — the exact reading a progress bar exists to prevent. The length is READ
    # from the payload rather than hardcoded here, so this checks that the
    # backend still sends it and that the client still consults it.
    assert re.search(r"_PROMOTE_STREAK\s*=\s*\d+", backend), (
        "kc_graph lost _PROMOTE_STREAK — a run of correct answers no longer "
        "promotes and the section fill is drawing a route that is gone"
    )
    assert '"streak_needed": _PROMOTE_STREAK' in backend, (
        "kc_estimate no longer reports streak_needed — the ladder cannot know "
        "how fast a run fills a section and would guess the rate"
    )
    assert '"streak": len(run)' in backend, (
        "kc_estimate no longer reports the run of correct answers — the "
        "section fill would move only with the twenty-question average, which "
        "barely shifts on one answer"
    )
    # And the run has to be scoped to the rung the learner is standing on. A
    # raw run count survives the promotion it just bought, so the rung the
    # learner has only this second arrived at draws itself FULL — a promise of
    # a promotion that `_streak_stage` will not make, because it aims one rung
    # above the LOWEST answer in the run and that is the rung they are on.
    assert "_streak_toward(run, stage)" in backend, (
        "kc_estimate reports the raw run length again — a newly reached rung "
        "would draw full before a single answer is given on it"
    )
    # And the estimate has to say WHICH rung it describes. `setProgress` takes
    # a fresh estimate while deliberately leaving the rung on screen alone, so
    # an estimate that arrived from one rung further up, with its run scoped to
    # that rung and therefore back at zero, would drag the section on screen
    # BACKWARDS on the very answer that finished it.
    assert 'est["stage"] = stage' in backend, (
        "kc_estimate no longer says which rung it describes — a promotion "
        "arriving mid-question would empty the section it just completed"
    )
    assert "current.estStage" in ladder_js, (
        "stage-ladder.js no longer reads the rung the estimate came from — a "
        "run scoped to the next rung would be drawn as progress through this one"
    )
    assert "streak_needed" in ladder_js and "current.streakNeeded" in ladder_js, (
        "stage-ladder.js no longer fills against the run of correct answers — "
        "the bar would ignore the route the learner is actually on"
    )

    for backend_stage, displayed in (("faded", "faded"), ("partial", "example")):
        assert abs(promote_at[displayed] - promote_lo[backend_stage]) < 1e-9, (
            f"the {displayed!r} rung draws its promotion mark at "
            f"{promote_at[displayed]} but kc_graph promotes at "
            f"{promote_lo[backend_stage]} — the app is showing a rule it does "
            f"not follow"
        )
    assert len(promote_at) == 2, (
        "PROMOTE_AT gained a rung. `lesson` is left by reading and `solo` is the "
        "top of the per-concept ladder — there is no threshold above either, so "
        "a third entry means the display believes in a promotion the backend "
        "does not make"
    )


def check_one_progress_readout():
    """ONE progress readout above the practice split, and it is the ladder.

    This screen accumulated four, each a reasonable idea on its own:

      the concept strip      concept name + four rung dots
      the difficulty bar     a 0-100 track, aim as fill, problem as a tick
      the accuracy bar       EWMA over the subtopic, usually "after calibration"
      the competency bar     BKT posterior with 0.85/0.95 gate marks (KG flow)

    Top progress is ladder only. Scoped KC understanding remains below review:
    different BKT quantity, explicit concept title, tier, evidence coverage.
    """
    index_html = read(os.path.join(SHARED, "index.html"))
    assert index_html.count('id="stage-ladder"') == 1, (
        "index.html has more (or fewer) than one #stage-ladder mount — the "
        "practice page gets one progress readout"
    )
    for gone in ('id="target-difficulty"', 'id="competency-bar-container"',
                 'id="concept-topbar"'):
        assert gone not in index_html, (
            f"index.html mounts {gone} again — that is one of the four readouts "
            "the single stage ladder replaced"
        )
    assert index_html.count('id="ewma-accuracy"') == 1
    bars_js = read(os.path.join(HERE, "bars.js"))
    assert "setConceptUnderstanding" in bars_js
    assert "evidence coverage" in bars_js
    assert "Accuracy of " not in bars_js

    # Difficulty is a CAPTION. The moment it grows a track again it is the
    # second bar this check exists to prevent, whatever it is called.
    ladder_js = read(os.path.join(HERE, "stage-ladder.js"))
    assert "stage-ladder-foot" in ladder_js, (
        "stage-ladder.js no longer writes the caption — difficulty would have "
        "nowhere to be stated"
    )
    for gone in ("difficulty-bar-track", "target-difficulty-track",
                 "target-difficulty-marker"):
        assert gone not in ladder_js, (
            f"stage-ladder.js renders {gone!r} — difficulty is one clause of a "
            "text caption, not a track"
        )

    # bars.js still owns the numbers (it is called from the places that KNOW
    # them), and it must hand every one of them to the ladder. bars.js writing
    # to nothing is how the caption would freeze on the previous question.
    bars_js = read(os.path.join(HERE, "bars.js"))
    assert "StageLadder" in bars_js, (
        "bars.js no longer forwards to StageLadder — the difficulty caption "
        "would stay on whatever the last question left there"
    )
    for gone in ("DifficultyBar", "ConceptTopbar", "ewmaAccuracy"):
        assert gone not in bars_js, (
            f"bars.js calls {gone} again — that widget is deleted"
        )

    # And the caption must be REPLACED on every show(), never merged. The
    # lesson screen passes no difficulty at all — it has no problem on it — so
    # a merge left the previous question's aim standing over a page with
    # nothing to aim at. Both assignments are unconditional; the practice path
    # re-supplies them in the same synchronous render.
    for field, arg in (("problemValue", "difficulty"), ("aimValue", "target")):
        assert f"if (Number.isFinite({arg})) {field}" not in ladder_js, (
            f"stage-ladder.js only sets {field} when show() is given one — a "
            "screen that passes none inherits the last question's number"
        )
        assert f"{field} = Number.isFinite({arg})" in ladder_js, (
            f"stage-ladder.js no longer assigns {field} in show() — the "
            "difficulty caption would not follow the item on screen"
        )

    # The caption's mastery clause belongs to ONE concept (competency-bar.js
    # writes it) and the caption itself is shared, so a KC change has to drop
    # it or the next concept inherits the last one's "72% mastered".
    assert re.search(r"!==\s*current\.kc\)\s*extraNote\s*=", ladder_js), (
        "stage-ladder.js does not clear extraNote when show() changes concept "
        "— the mastery clause leaks onto the next concept's caption"
    )

    # The concept button jumps to the map. The graph exposes a FUNCTION for
    # that and it is what waits for the tab's data and layout; an event would
    # need a listener, and there is none anywhere in the app.
    assert "deltaFocusConceptGraphKc" in ladder_js, (
        "stage-ladder.js no longer calls deltaFocusConceptGraphKc — the "
        "concept button would open the knowledge graph on the wrong node"
    )

    # A graded verdict returns a fresh estimate and the fill is drawn from it.
    # Dropping it leaves the section showing where the learner stood BEFORE the
    # answer, for as long as the review is on screen.
    events_js = read(os.path.join(HERE, "events.js"))
    for update in ("setProgress(result.ladder_estimate)",
                   "setProgress(record.ladderEstimate)"):
        assert update in events_js, (
            f"events.js lost {update} — ladder fill would stay on pre-answer "
            "reading through review"
        )


# `check_difficulty_range_matches_backend` used to live here. It mirrored
# `_DIFF_FLOOR` / `_DIFF_SPAN` from `app/prioritization.py` into
# `difficulty-bar.js`'s `AIM_FLOOR` / `AIM_SPAN`, which the bar used to convert
# a mastery threshold into a position on its 0-100 track. There is no track any
# more — difficulty is stated as the two numbers themselves, in text — so the
# client holds no copy of the range and there is nothing left to drift. The
# check was deleted rather than left passing on absent constants, which is the
# failure mode `_every_check_is_registered` exists to catch.


def _every_check_is_registered(checks):
    """A check that is defined and not in the list is worse than no check.

    Twice now a check has survived a refactor as a function nobody called, and
    a watch suite that exits 0 while running less than it did is exactly the
    failure mode this folder cannot see. THIS module is scanned alongside the
    two sibling ones, because a check written here is just as easy to leave out
    of the list — and the comparison is by function IDENTITY, so a same-named
    function in another module cannot stand in for the one that was dropped."""
    registered = {id(fn) for fn in checks}
    for module in (sys.modules[__name__], watch_invariants, watch_lessons,
                   watch_notebook):
        for name in dir(module):
            fn = getattr(module, name)
            if name.startswith("check_") and callable(fn) and id(fn) not in registered:
                assert False, (
                    f"{module.__name__}.{name} is defined but not in watch.py's "
                    "checks list — it has stopped running"
                )


# ── Run all checks ───────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_promotion_threshold_matches_the_backend,
              check_one_progress_readout,
              check_lesson_code_can_actually_run,
              check_colab_lesson_goes_to_the_notebook,
              check_a_resumed_clock_matches_the_break,
              check_the_gate_teaches_one_concept_then_drills_it,
              check_the_fifth_rung_is_shown_not_stored,
              check_the_notebook_kernel_has_a_fallback,
              check_the_notebook_view_is_loaded_after_what_it_calls,
              check_the_verdict_line_is_read_the_same_way_everywhere,
              check_a_problem_is_recorded_once_per_visit,
              check_a_slow_run_cannot_touch_another_notebook,
    check_a_collapsed_cell_still_knows_its_own_source,
              check_the_notebook_never_falls_back_to_the_prefix_runner,
              check_a_solution_stays_closed_until_asked]
    _every_check_is_registered(checks)
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
