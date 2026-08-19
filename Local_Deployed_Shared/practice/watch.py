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

    `concept-topbar.js` draws a threshold on the concept's interval — cross it
    with the left end of the bar and the next question comes with less support.
    That number is a copy of `app/kc_graph.py`'s `PROMOTE_LO`, held in a second
    language because the rung is decided server-side and drawn client-side.

    A copy nobody checks is a copy that drifts, and this one drifts SILENTLY in
    the worst direction: the learner clears a mark the app drew for them and
    nothing happens, or the rung changes while the mark is still ahead of them.
    Either way the app has told them a rule it does not follow, which is worse
    than drawing no mark at all.

    ⚠️ The two speak different vocabularies. The backend's `partial` is the
    display's `worked` — see STAGE_ALIASES. This check translates rather than
    comparing keys, because comparing keys is how the numbers would end up
    swapped and still "matching". They WERE swapped, until 2026-08-06: the
    display's rung names had `faded` and `worked` the wrong way round, so each
    dot described the other one's rung and this check compared the two wrong
    numbers to each other and passed.

    The right long-term fix is for the ladder response to carry its own
    threshold so there is one runtime authority; codex flagged the same thing on
    2026-08-03. Until then, this.
    """
    topbar = read(os.path.join(HERE, "concept-topbar.js"))
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
            r"PROMOTE_AT\s*=\s*\{(.*?)\}", topbar, re.S).group(1))}

    # backend rung -> the rung the display calls it
    for backend_stage, displayed in (("faded", "faded"), ("partial", "worked")):
        assert backend_stage in promote_lo, f"backend lost PROMOTE_LO[{backend_stage}]"
        assert displayed in promote_at, (
            f"concept-topbar.js lost the {displayed!r} threshold — the rung would "
            f"draw no promotion mark while the backend still promotes on one"
        )
        assert abs(promote_at[displayed] - promote_lo[backend_stage]) < 1e-9, (
            f"the {displayed!r} rung draws its promotion mark at "
            f"{promote_at[displayed]} but kc_graph promotes at "
            f"{promote_lo[backend_stage]} — the app is showing a rule it does "
            f"not follow"
        )
    assert len(promote_at) == 2, (
        "PROMOTE_AT gained a rung. `lesson` is left by reading, `solo` is the top "
        "of the per-concept ladder, and `integrated` is not reached by clearing a "
        "threshold on one concept — so a third entry means the display believes "
        "in a promotion the backend does not make"
    )


def check_difficulty_bar_is_one_bar():
    """ONE difficulty readout on the practice page, and its thresholds are real.

    The page drew this quantity twice: a 96px `.concept-topbar-diff-bar` in the
    concept strip whose fill was the aim and whose tick was the problem, and a
    `.target-difficulty` card further down whose readout was "Old 24.5". Two
    pictures of one number, in two visual languages, disagreeing about which
    number was which — the report was literally "there are two of them".

    So: exactly one mount, and nothing rebuilding the old segments.
    """
    index_html = read(os.path.join(SHARED, "index.html"))
    assert index_html.count('id="target-difficulty"') == 1, (
        "index.html has more (or fewer) than one #target-difficulty mount — the "
        "difficulty bar is drawn once, under the concept strip"
    )
    assert "concept-topbar-diff" not in index_html and "concept-topbar-est" not in index_html, (
        "the concept strip grew its own difficulty/estimate segment back — that "
        "is the second bar this check exists to prevent"
    )
    topbar = read(os.path.join(HERE, "concept-topbar.js"))
    for gone in ("concept-topbar-diff", "concept-topbar-est"):
        assert gone not in topbar, (
            f"concept-topbar.js renders {gone!r} again — the strip names the "
            "concept and the rung; the bar below it owns difficulty"
        )
    # The bar is drawn by bars.js and annotated by difficulty-bar.js. Neither is
    # allowed to be the only one wired up: bars.js without the readout leaves the
    # numbers frozen on the previous question, and the readout without bars.js is
    # a set of labels with no track under them.
    bars_js = read(os.path.join(HERE, "bars.js"))
    for call in ("DifficultyBar.aim(", "DifficultyBar.move(", "DifficultyBar.live(",
                 "DifficultyBar.unavailable("):
        assert call in bars_js, (
            f"bars.js no longer calls {call}…) — the track would move while the "
            "numbers beside it stayed on the last question's answer"
        )


def check_difficulty_range_matches_backend():
    """The bar's floor and span are a copy of the backend's difficulty range.

    `difficulty-bar.js` places the support floor at `AIM_FLOOR` and converts a
    promotion threshold in mastery to a point on the 0-100 track with
    `AIM_FLOOR + AIM_SPAN * bound`. Both mirror `_DIFF_FLOOR` / `_DIFF_SPAN` in
    `app/prioritization.py`, which is where `target_difficulty` is actually
    computed.

    Same reasoning as the PROMOTE_AT check below: a threshold drawn in the wrong
    place is worse than no threshold. Change the span in the backend and every
    green line on this bar quietly points at a number the queue never serves.
    """
    bar_js = read(os.path.join(HERE, "difficulty-bar.js"))
    prio = os.path.join(
        HERE, "..", "..", "This-Directory-Only", "backend", "app", "prioritization.py")
    if not os.path.exists(prio):
        return  # backend not checked out beside the frontend — nothing to compare
    backend = read(prio)

    def _const(src, name, pattern):
        m = re.search(pattern, src)
        assert m, f"could not find {name}"
        return float(m.group(1))

    for js_name, py_name in (("AIM_FLOOR", "_DIFF_FLOOR"), ("AIM_SPAN", "_DIFF_SPAN")):
        js_value = _const(bar_js, js_name, rf"const {js_name}\s*=\s*([0-9.]+)")
        py_value = _const(backend, py_name, rf"{py_name}\s*=\s*([0-9.]+)")
        assert abs(js_value - py_value) < 1e-9, (
            f"difficulty-bar.js draws {js_name}={js_value} but prioritization.py "
            f"computes with {py_name}={py_value} — the floor and the promotion "
            f"line on the bar are both off by that difference"
        )



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
              check_difficulty_bar_is_one_bar,
              check_difficulty_range_matches_backend,
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
