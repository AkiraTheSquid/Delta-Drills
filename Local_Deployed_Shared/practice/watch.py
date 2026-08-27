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
import watch_basic_mode
import watch_feedback
from watch_common import (  # noqa: F401 — re-exported for anything importing watch
    HERE, SHARED, REQUIRED_JS, REQUIRED_DOCS, REQUIRED_ASSETS, read,
)
from watch_invariants import (check_invariants, check_a_torch_question_never_grades_on_pyodide,
                              check_a_deleted_practice_notice_stays_deleted,
                              check_the_session_clock_is_not_the_learners_to_set)
from watch_basic_mode import check_a_hidden_rating_still_commits_the_attempt
from watch_feedback import (
    check_a_learner_can_always_report_a_broken_problem,
    check_a_lesson_can_be_reported_without_touching_the_question,
    check_feedback_that_never_left_the_browser_is_not_called_logged,
)
from watch_notebook import (
    check_a_code_cell_grows_to_fit_its_own_code,
    check_only_one_module_patches_a_code_editors_value,
    check_a_slow_run_cannot_touch_another_notebook,
    check_a_collapsed_cell_still_knows_its_own_source,
    check_a_problem_is_recorded_once_per_visit,
    check_the_solution_is_shown_and_the_hints_are_not,
    check_the_solution_cell_is_never_the_learners_work,
    check_the_answer_only_lives_where_the_learner_can_see_it,
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
    # 🔴 ONE TAB SINCE 2026-08-24. The Placement test stopped being a page and
    # a tab of its own — Seth: "the diagnostic and practice should be combined
    # into one tab, with it being called Learner Home" — so the overview card,
    # the results card and the workspace host all live inside #page-practice.
    assert 'id="page-diagnostic"' not in index_html and 'data-tab="diagnostic"' not in index_html, (
        "the Placement test is a page/tab of its own again. Two tabs sharing "
        "one editor and one PracticeAPI.currentQuestion is what the Practice "
        "tab lock existed to paper over"
    )
    assert 'id="diagnostic-workspace-host"' in index_html
    assert 'id="diagnostic-workspace-host"' in index_html.split('id="page-practice"')[1], (
        "the placement workspace host must be inside #page-practice: it is "
        "what takes the idle surface off the screen while a probe is up"
    )
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
    assert "setPracticeLock(" not in diagnostic_page, (
        "the Practice tab lock is back. There is one tab now and it cannot be "
        "locked against itself"
    )
    assert "const diagnosticOnScreen =" in diagnostic_page and (
        "running && diagnosticOnScreen()" in diagnostic_page
    ), (
        "the workspace may be hosted only while the page that owns the "
        "placement is on screen — delta:practice-state-changed fires from any "
        "tab, and keying on the placement alone hauls the workspace under a "
        "page nobody is looking at"
    )
    assert 'byId("page-practice")' in diagnostic_page and (
        'byId("page-diagnostic")' not in diagnostic_page
    ), (
        "diagnostic-page.js must address the Learner Home; every read of the "
        "deleted #page-diagnostic is silently undefined"
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
    # 2026-08-22: the difficulty caption is GONE from the strip, by request —
    # first the "this problem is rated N" half, then the aim with it. What the
    # check protects is unchanged and is the line below: difficulty may not
    # come back as a TRACK. The clause that outlived it is `setNote`'s
    # topic-level reading, which has to keep reaching the DOM or the knowledge-
    # graph flow silently loses the number it ends its loop on.
    # 🔴 THE NOTCH'S COUNTDOWN IS A MIRROR, NOT A CLOCK. `.session-status-row`
    # is hidden by CSS while timer.js keeps counting and auto-submitting, so
    # the tab on the seam is the only place the learner can see the deadline.
    # It reads #session-countdown; the moment it starts counting on its own it
    # is a SECOND clock, and the one that disagrees with the auto-submit is the
    # one on screen. There is no honest reason for a timer in this file.
    notch_js = read(os.path.join(HERE, "notch-menu.js"))
    assert "session-countdown" in notch_js, (
        "notch-menu.js no longer reads #session-countdown — the session's "
        "countdown is hidden with its row, so nothing would show the learner "
        "the deadline that auto-submits their answer"
    )
    for own_clock in ("setInterval", "setTimeout", "Date.now"):
        assert own_clock not in notch_js, (
            f"notch-menu.js calls {own_clock} — the countdown on the notch is "
            f"a mirror of timer.js's clock, and a second clock here drifts "
            f"from the one that actually auto-submits"
        )

    assert "stage-ladder-note" in ladder_js, (
        "stage-ladder.js no longer writes the callout note — setNote's "
        "topic-level mastery reading would have nowhere to be stated"
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
                   watch_notebook, watch_basic_mode, watch_feedback):
        for name in dir(module):
            fn = getattr(module, name)
            if name.startswith("check_") and callable(fn) and id(fn) not in registered:
                assert False, (
                    f"{module.__name__}.{name} is defined but not in watch.py's "
                    "checks list — it has stopped running"
                )


def check_every_placement_question_gets_the_same_clock():
    """One fixed allowance per probe, and every advance path kills it.

    The placement test runs OUTSIDE a practice session — starting it calls
    PracticeSession.finish("placement") — so none of the session timers apply
    and a probe had no limit at all until placement-timer.js. Three ways that
    silently regresses, so three assertions:

      * the allowance stops being ONE constant (per-question or
        difficulty-scaled time would make the probes incomparable, which is
        the whole point of a placement);
      * the module stops being loaded, or loads before the hooks that call it;
      * an advance path forgets to stop the clock, which is how a countdown
        expires onto the NEXT question and answers it for the learner.
    """
    timer = read(os.path.join(HERE, "placement-timer.js"))
    assert "const PLACEMENT_ANSWER_SECS = 120;" in timer, (
        "the placement allowance must stay one named constant — every probe "
        "gets the same time or the estimates are not comparable"
    )
    # No second source of truth: the only other number the clock may hold is
    # the resume floor/grace, never a per-question or per-difficulty value.
    assert "q().answer_secs" not in timer and "difficulty" not in timer, (
        "placement timing must not vary by question or difficulty"
    )
    # `PracticeAPI` is a top-level const in api.js — NOT a window property.
    # Reading it off window is undefined at runtime and silent at review time:
    # the clock simply never sees a probe. Same trap notebook-view.js documents.
    assert "window.PracticeAPI?.currentQuestion" not in timer, (
        "read PracticeAPI from script scope, not window — a top-level const is "
        "not a window property and the clock would never start"
    )
    for token in ("onQuestionRendered", "pauseForGrading", "resumeAfterFailedSubmit", "stop"):
        assert f"{token}," in timer or f"{token}:" in timer, (
            f"placement-timer.js no longer exports {token}"
        )

    index = read(os.path.join(SHARED, "index.html"))
    assert "practice/placement-timer.js" in index, "placement-timer.js is not loaded"
    # Match the SCRIPT TAG, not the bare name: both files are named in prose
    # comments elsewhere in the page, and a comment that happens to sit higher
    # would satisfy a plain substring search while the load order was wrong.
    assert index.find('src="practice/ui.js') < index.find('src="practice/placement-timer.js'), (
        "placement-timer.js must load after the modules whose hooks call it"
    )
    assert 'id="placement-timer"' in index, (
        "the countdown needs a static element: it is the anchor the info dot "
        "attaches to, and check_infotips fails without it"
    )

    ui = read(os.path.join(HERE, "ui.js"))
    assert "PlacementTimer?.onQuestionRendered()" in ui, (
        "no probe starts a clock — renderQuestion must arm the placement timer"
    )
    events = read(os.path.join(HERE, "events.js"))
    assert "PlacementTimer?.stop()" in events, (
        "_loadNextPracticeQuestion must kill the placement clock, or an expiry "
        "at 00:01 lands on the question that just loaded"
    )
    assert "PlacementTimer?.pauseForGrading()" in events, (
        "submitting must stop the placement clock while the grade is in flight"
    )
    # Pyodide cannot import torch, and the einops questions in the bank ARE
    # torch questions — routing them local made every Submit unanswerable.
    api = read(os.path.join(HERE, "api.js"))
    assert "!needsTorchRuntime(this.currentQuestion, userCode)" in api, (
        "einops questions that touch torch must grade on the backend — local "
        "Pyodide refuses them and the learner gets no verdict at all"
    )
    # A question that CANNOT run here must not re-arm a countdown: expiry
    # force-submits, the submit is refused again, and the clock loops at 00:30.
    timer = read(os.path.join(HERE, "timer.js"))
    assert "blockOnUnrunnableQuestion" in timer and "blockOnUnrunnableQuestion" in events, (
        "a blocked submit must stop the session clock, not resume it"
    )
    # A probe is timed by the placement's rule even inside a learner's session.
    assert "PlacementTimer.secondsPerQuestion()" in timer, (
        "a placement probe inside a session must use the placement allowance, "
        "not whatever answer time that session was set up with"
    )
    # One writer for the start button: two copies of the label is how it
    # flickers between two names when the page refreshes its status.
    page = read(os.path.join(HERE, "diagnostic-page.js"))
    assert "renderStartButton" in page and "renderStartButton" in events, (
        "events.js must delegate the placement start button to diagnostic-page.js"
    )
    assert "Take the placement test" not in events, (
        "the start-button label lives in diagnostic-page.js only"
    )


def check_the_placement_result_is_the_number_the_backend_seeded():
    """The results card must report the placement, and say what it doesn't know.

    Three separate failures live in this one check, all of them from
    2026-08-23.

    1. THE READINESS FIGURE IS A COPY. `placement-results.js` turns the
       backend's theta into a percentage with the same affine map that
       `diagnostic.py::_mastery_from_theta` uses to SEED per-atom BKT mastery
       at finish(). If those four constants drift apart, the card tells the
       learner a readiness the rest of the app does not act on — the quiet
       kind of wrong, because both halves keep working. Same reasoning as
       check_promotion_threshold_matches_the_backend, same remedy.

    2. `.primary` MUST NOT BE ON THE CTA. `.primary` sets `width: 100%` and
       12px/28px padding; `.placement-start-btn` overrode the padding and not
       the width, so the button rendered as a card-wide 4px-tall strip. It is
       a two-class collision, so nothing in either rule looks wrong on its
       own — this is the only place it can be caught.

    3. AN UNPROBED AREA IS NOT A MEASUREMENT. The backend returns a theta for
       every area whether or not the test ever probed it; the unprobed ones
       are the prior, propagated. Rendering those as bare percentages invents
       confidence the test never earned, and the learner plans around it. The
       renderer must keep the label and the dimmed style that separate them.
    """
    results_js = read(os.path.join(HERE, "placement-results.js"))
    index = read(os.path.join(SHARED, "index.html"))
    page = read(os.path.join(HERE, "diagnostic-page.js"))
    css = read(os.path.join(SHARED, "styles", "practice", "diagnostic.css"))

    # 1. the readiness map is a copy of the backend's seeding map
    diag = os.path.join(
        HERE, "..", "..", "This-Directory-Only", "backend", "app", "diagnostic.py")
    if os.path.exists(diag):
        backend = read(diag)
        for js_name, py_name in (("DIFF_FLOOR", "_DIFF_FLOOR"),
                                 ("DIFF_SPAN", "_DIFF_SPAN"),
                                 ("SEED_MASTERY_FLOOR", "SEED_MASTERY_FLOOR"),
                                 ("SEED_MASTERY_CAP", "SEED_MASTERY_CAP")):
            m_js = re.search(rf"^\s*const {js_name} = ([0-9.]+)\s*;", results_js, re.M)
            m_py = re.search(rf"^{py_name} = ([0-9.]+)", backend, re.M)
            assert m_js, f"placement-results.js lost its {js_name} constant"
            assert m_py, f"diagnostic.py lost {py_name}"
            assert float(m_js.group(1)) == float(m_py.group(1)), (
                f"readiness map drifted: placement-results.js {js_name}="
                f"{m_js.group(1)} but diagnostic.py {py_name}={m_py.group(1)} — "
                "the card would report a readiness the seeding does not use"
            )

    # 2. neither placement button may carry .primary, and the CTA keeps its
    #    wrapper (without it the infotip dot is a sibling flex item and drops
    #    onto its own line under the button).
    for btn_id in ("placement-start-btn", "diagnostic-practice-btn"):
        tag = re.search(rf'<button[^>]*id="{btn_id}"[^>]*>', index)
        assert tag, f"index.html lost #{btn_id}"
        classes = re.search(r'class="([^"]*)"', tag.group(0))
        assert classes and "primary" not in classes.group(1).split(), (
            f"#{btn_id} carries .primary again — width:100% plus the placement "
            "button's own padding is the full-bleed 4px-tall strip"
        )
    assert re.search(
        r'<span class="placement-cta">\s*<button[^>]*id="placement-start-btn"', index), (
        "the placement CTA lost its .placement-cta wrapper — infotips.js inserts "
        "the dot into the anchor's parentNode, so unwrapped it orphans below the button"
    )
    assert ".placement-cta" in css, "styles/practice/diagnostic.css lost .placement-cta"
    assert ".placement-cta.hidden" in css and "el.parentElement?.classList.toggle" in page, (
        "the CTA wrapper must hide WITH its button, and .hidden must be re-asserted "
        "at this file's specificity — diagnostic.css loads after components.css, so "
        "the tie goes to .placement-cta and an empty flex item keeps its gap"
    )

    # 3. one writer for the card body, and it must be reached
    for anchor in ("placement-results-meta", "placement-readiness",
                   "placement-areas", "placement-results-empty"):
        assert f'id="{anchor}"' in index, f"index.html lost the #{anchor} results anchor"
    assert "window.PlacementResults?.render(status)" in page, (
        "diagnostic-page.js must hand the status to placement-results.js — "
        "without it the card is empty for someone who finished the test"
    )
    assert "!!status.completed_at && !status.active" in page, (
        "the results card must render only for a FINISHED placement; a mid-test "
        "status carries live estimates that would read as a final result"
    )
    # 🔴 SCOPED TO THE RESULTS CARD, not to the whole file. This read
    # `"will appear here" not in index` until 2026-08-24, when e4a65d2b gave the
    # Account tab's instructor-mode hint the sentence "Review tools will appear
    # here as they are built" — copy in another tab, about another feature, that
    # turned this into a standing false failure for every session in the repo.
    # The invariant was always about ONE element: #diagnostic-results, which
    # placement-results.js fills from the status payload. A placeholder anywhere
    # else is not this bug, and a substring test over 1,700 lines of markup will
    # keep colliding with unrelated copy.
    #
    # 🔴 `\sid=` AND THE NESTING ASSERT ARE BOTH LOAD-BEARING (codex, 2026-08-24).
    # `[^>]*id="` also matches `aria-labelledby="diagnostic-results"`, which would
    # scope this to the wrong element; and a non-greedy `.*?</section>` stops at
    # the FIRST close tag, so nesting a <section> in the card would silently slice
    # off the part the placeholder is most likely to be in. A check that quietly
    # covers less than it claims is worse than one that fails — hence the assert
    # rather than a balanced-scan parser.
    start = re.search(r'<section\b[^>]*\sid="diagnostic-results"[^>]*>', index)
    assert start, "index.html lost the #diagnostic-results card"
    rest = index[start.end():]
    end = rest.find("</section>")
    assert end != -1, "#diagnostic-results is never closed"
    results_card = rest[:end]
    assert "<section" not in results_card, (
        "#diagnostic-results now nests a <section>, so this slice stops at the "
        "inner close tag and the assertion below would cover only part of the card"
    )
    assert "will appear here" not in results_card, (
        "the results placeholder is back — it showed to learners who had already "
        "finished the test, while the real numbers sat unrendered in the payload"
    )
    assert "placement-areas" not in page, (
        "placement-results.js is the only writer of the card body; a second "
        "writer is how the two halves disagree about the same placement"
    )
    assert index.find('src="practice/placement-results.js') < index.find(
        'src="practice/diagnostic-page.js'), (
        "placement-results.js must load BEFORE diagnostic-page.js, which calls it "
        "from a refresh it starts at load"
    )

    # 4. the honesty rule: an unprobed area is visibly not a measurement
    assert "not probed" in results_js and "placement-area--unprobed" in results_js, (
        "an area with zero probes must be labelled and dimmed — its theta is the "
        "prior propagated, and printing it bare invents confidence the test never earned"
    )
    assert ".placement-area--unprobed" in css, (
        "diagnostic.css lost the unprobed styling, so a prior renders identically "
        "to a measured area"
    )


# ── Run all checks ───────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_a_torch_question_never_grades_on_pyodide,
              check_promotion_threshold_matches_the_backend,
              check_one_progress_readout,
              check_lesson_code_can_actually_run,
              check_colab_lesson_goes_to_the_notebook,
              check_a_resumed_clock_matches_the_break,
              check_the_session_clock_is_not_the_learners_to_set,
              check_the_gate_teaches_one_concept_then_drills_it,
              check_the_fifth_rung_is_shown_not_stored,
              check_the_notebook_kernel_has_a_fallback,
              check_the_notebook_view_is_loaded_after_what_it_calls,
              check_the_verdict_line_is_read_the_same_way_everywhere,
              check_a_problem_is_recorded_once_per_visit,
              check_a_slow_run_cannot_touch_another_notebook,
    check_a_collapsed_cell_still_knows_its_own_source,
              check_the_notebook_never_falls_back_to_the_prefix_runner,
              check_the_solution_is_shown_and_the_hints_are_not,
              check_the_solution_cell_is_never_the_learners_work,
              check_the_answer_only_lives_where_the_learner_can_see_it,
              check_only_one_module_patches_a_code_editors_value,
              check_a_code_cell_grows_to_fit_its_own_code,
              check_every_placement_question_gets_the_same_clock,
              check_the_placement_result_is_the_number_the_backend_seeded,
              check_a_deleted_practice_notice_stays_deleted,
              check_a_hidden_rating_still_commits_the_attempt,
              check_a_learner_can_always_report_a_broken_problem,
              check_a_lesson_can_be_reported_without_touching_the_question,
              check_feedback_that_never_left_the_browser_is_not_called_logged]
    _every_check_is_registered(checks)
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
