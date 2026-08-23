"""watch_invariants.py — the big invariant sweep for practice.

Split out of `watch.py` (Modulario, 2026-08-19): the file was over the 700-LOC
line and `check_invariants` was a third of it on its own. The check itself is
unchanged and still runs from `watch.py`'s list — see the guard at the bottom of
that file, which fails if a check defined here is not in the list.
"""
import json
import os
import re
import subprocess
import sys

from watch_common import HERE, SHARED, read


def check_the_session_clock_is_not_the_learners_to_set():
    """Fixed per-QUESTION allowances, pause/resume only, and a notch that stays.

    Four halves of one 2026-08-23 decision, and each one comes undone on its
    own without anything else here failing:

      1. The two allowances are constants in timer.js. Putting them back on the
         snapshot, or back behind an input, is how "predetermined" quietly
         stops being true — and a v2 snapshot that carries `answerSecs` again
         would resume a question under a clock this build never set.
      2. A block has no LENGTH. `shouldFinishInsteadOfAdvance` returning a
         quota comparison again reinstates a session that ends on its own,
         which is the thing pause replaced.
      3. There is no End session — not the button, not the handler, not the
         menu item. Pause is the only way out.
      4. The notch outlives the session, so it hangs off `.practice-container`
         and not off `.practice-split` (which is display:none between blocks).
         This one is pure DOM order: nothing throws when it regresses, the
         notch simply is not on the idle screen.
    """
    timer = read(os.path.join(HERE, "timer.js"))
    for name in ("ANSWER_SECS", "REVIEW_SECS"):
        assert re.search(rf"const {name} = \d+;", timer), (
            f"timer.js lost {name} — the per-question allowance is settable again"
        )
    snapshot = timer.split("const _snapshot = () => {", 1)
    assert len(snapshot) == 2, "timer.js::_snapshot is gone"
    body = snapshot[1].split("\n  };", 1)[0]
    for dead in ("answerSecs:", "reviewSecs:", "total:"):
        assert dead not in body, (
            f"_snapshot writes `{dead}` again. The clock is a constant and a "
            f"block has no length; a snapshot carrying either resumes under "
            f"rules this build does not enforce"
        )
    assert "const shouldFinishInsteadOfAdvance = () => false;" in timer, (
        "a session quota is back — `shouldFinishInsteadOfAdvance` compares "
        "again, so a block can end on its own instead of being paused"
    )

    index_html = read(os.path.join(SHARED, "index.html"))
    for dead in ('id="session-end-btn"', 'id="practice-notch-end"',
                 'id="session-question-count"', 'id="session-answer-time"',
                 'id="session-review-time"'):
        assert dead not in index_html, (
            f"{dead} is back in index.html — the learner is setting the "
            f"session again, or ending it early"
        )
    assert 'id="practice-notch-stop"' in index_html, (
        "the notch lost its square. It is the pause control that is always on "
        "screen; the menu item alone is two clicks for the only way out"
    )
    # The square is LEFT of the clock: it is the first thing in the tab after
    # the screen-reader phase span, and the clock follows it.
    tab = index_html.split('id="practice-notch-tab"', 1)[1].split("</div>", 1)[0]
    stop_at = tab.find('id="practice-notch-stop"')
    clock_at = tab.find('id="practice-notch-clock"')
    assert -1 not in (stop_at, clock_at) and stop_at < clock_at, (
        "the square is no longer to the LEFT of the clock in the notch tab"
    )
    # And the notch hangs off the container, so it survives the split going away.
    container_at = index_html.find('<div class="practice-container">')
    notch_at = index_html.find('id="practice-notch"', container_at)
    split_at = index_html.find('<div class="practice-split">', container_at)
    assert -1 not in (container_at, notch_at, split_at), "practice page lost a landmark"
    assert notch_at < split_at, (
        "#practice-notch is inside .practice-split again. The split is "
        "display:none between sessions, so the notch would vanish with it — "
        "and it has to stay, showing the allowance the next question gets"
    )

    # The idle screen reads a real number, and says so honestly when it cannot.
    idle = read(os.path.join(HERE, "session-idle.js"))
    assert "PracticeReadiness" in idle and "hasPausedSession" in idle, (
        "session-idle.js no longer reads readiness, or decides resume-vs-start "
        "from timer.js's own answer"
    )
    assert 'pctEl.textContent = "—"' in idle, (
        "the idle screen renders an unreadable readiness as a number. A "
        "registry that would not load is a claim about the network, not a 0% "
        "claim about the learner"
    )


def check_invariants():
    for fname in ("README.md", "watch.py"):
        first = read(os.path.join(HERE, fname)).splitlines()[:1]
        assert first and "modulario:template" not in first[0], (
            f"{fname} still has modulario template marker on line 1"
        )

    data = json.loads(read(os.path.join(SHARED, "arena_prereqs_structured.json")))
    bad = [
        q["id"] for q in data
        if q.get("exercise", {}).get("task_type") == "function_implementation"
        and not q.get("exercise", {}).get("canonical_solution")
    ]
    assert not bad, f"function questions missing canonical_solution: {bad}"

    id27 = next((q for q in data if q.get("id") == 27), None)
    assert id27 is not None, "missing question id 27"
    assert id27["exercise"].get("canonical_solution"), "id-27 canonical_solution still null"

    runner = read(os.path.join(HERE, "runner.js"))
    contract = read(os.path.join(HERE, "RUNTIME_CONTRACT.md"))
    # Names that the doc claims are always/conditionally injected.
    for name in ("import numpy as np", "display_array_as_img", "delta_numbers.npy"):
        assert name in runner, f"runner.js no longer injects expected name: {name!r}"
        assert name.split()[-1].split("/")[-1] in contract or name in contract, (
            f"RUNTIME_CONTRACT.md does not mention {name!r}"
        )

    # The dispatch line that forces einops-questions to Pyodide is the documented gotcha.
    assert re.search(r"questionNeedsEinops\(", runner), (
        "runner.js no longer uses questionNeedsEinops gating — update RUNTIME_CONTRACT.md"
    )

    # notebook.js recognises a silent-but-successful run by matching the exact
    # string runner.js substitutes for empty output, so it can report the cell's
    # assertion count instead. Reword it in one file only and every quiet cell
    # silently goes back to saying "no printed output".
    notebook = read(os.path.join(HERE, "notebook.js"))
    no_output = "✓ Ran successfully (no printed output)"
    assert no_output in runner, (
        f"runner.js no longer emits {no_output!r} — notebook.js matches on it"
    )
    assert no_output in notebook, (
        f"notebook.js lost its copy of {no_output!r} — silent cells will stop "
        "reporting their passed checks"
    )

    # Every cell is executed through the harness: it is what gives a bare
    # trailing expression its Jupyter-style echo, what reports the names a
    # silent cell bound, and what keeps traceback line numbers cell-relative.
    for marker in ("_delta_cell", "_delta_bound", "redirect_stdout"):
        assert marker in notebook, f"notebook.js lost the run harness piece {marker!r}"

    # The runnable/static split is the authoring format's own marker, produced
    # by md() in lessons.js and consumed here. If either half goes, every fence
    # becomes a cell or none of them do.
    lessons = read(os.path.join(HERE, "lessons.js"))
    assert 'data-fence=' in lessons, (
        "lessons.js md() no longer emits data-fence — notebook.js cannot tell a "
        "runnable ```python fence from a ```python no-run one"
    )
    assert "nb-scope" in lessons and "nb-scope" in notebook, (
        "the nb-scope contract is broken — lessons.js marks the regions whose "
        "fences validate_lessons.py executes, notebook.js only mounts inside them"
    )

    # ── concept → notebook section ────────────────────────────────
    # The Knowledge Graph routes a CONCEPT the same way ui.js routes a problem:
    # through the generated index, never by re-slugging the kc id here. A slug
    # that drifted by one character is an anchor Colab silently ignores, and the
    # failure looks like "the link does nothing" rather than like a bug.
    index = json.loads(read(os.path.join(SHARED, "lessons", "colab_notebooks.json")))
    kcs, kps = index.get("kcs") or {}, index.get("kps") or {}
    assert kcs and kps, "colab_notebooks.json lost its kcs/kps maps"
    assert set(kcs) == set(kps), (
        "colab_notebooks.json kcs and kps disagree on which concepts exist: "
        f"{sorted(set(kcs) ^ set(kps))} — regenerate with generate_colab_notebooks.py"
    )
    lesson_ids = {lesson["id"] for lesson in index.get("lessons", [])}
    orphans = sorted(kc for kc, lid in kcs.items() if lid not in lesson_ids)
    assert not orphans, f"concepts pointing at a notebook that is not in the index: {orphans}"
    bad_anchors = sorted(kc for kc, a in kps.items() if not str(a).startswith("dd-kp-"))
    assert not bad_anchors, f"concept anchors that are not dd-kp-* cells: {bad_anchors}"

    colab_mode = read(os.path.join(HERE, "colab_mode.js"))
    assert "hrefForKc" in colab_mode, (
        "colab_mode.js no longer exposes hrefForKc — the knowledge graph cannot "
        "send the tab to the section that teaches a concept"
    )
    route = read(os.path.join(SHARED, "concept-graph", "kc-colab-route.js"))
    for name in ("hrefForKc", "openNotebook", "kg-colab-link"):
        assert name in route, f"kc-colab-route.js lost {name!r}"
    graph = read(os.path.join(SHARED, "concept-graph", "lesson-graph.js"))
    assert "DDGraphColab" in graph, (
        "lesson-graph.js stopped calling window.DDGraphColab — selecting a "
        "concept will render its lesson and route nothing"
    )

    # ── an attempt has to be COUNTED, not just graded ─────────────
    # `submit_answer` parks the attempt in `pending_attempt`; `send_feedback` is
    # what increments n, steps the staircase and moves recent accuracy. Offline
    # they are two calls, and the Colab edition never makes the second one on
    # its own — there is no felt-difficulty step to hang it off. Unpaired, every
    # notebook check was overwritten by the next one and the concept graph read
    # a week of practice as nothing at all. The failure is silent by
    # construction: the UI advances, the grade shows, and only the mastery
    # numbers stay at zero.
    # The JS half is checked on CALL syntax, not on the words: the comment above
    # this code names both functions, so a presence test would keep passing on
    # the explanation of a pairing that had been deleted.
    api_js = read(os.path.join(HERE, "api.js"))
    local_eval = api_js.split("async recordLocalEval", 1)[-1].split("\n  async ", 1)[0]
    for call in ("api.submit_answer(", "api.send_feedback("):
        assert call in local_eval, (
            f"recordLocalEval no longer calls {call}…) in its offline branch — "
            "submit alone leaves the attempt pending until the next one overwrites it"
        )

    # ── the difficulty step has to reach the rail ─────────────────
    # The Colab edition's whole feedback on a verdict is the ladder moving:
    # green when the answer earned a harder next question, red when it cost one.
    # It is drawn from what `recordLocalEval` REPORTS, so the reporting is the
    # contract. Checked on call syntax and on the returned field names, because
    # the prose around this code names all of it and a word-presence test would
    # keep passing on the explanation of a wire that had been cut.
    assert re.search(r"getTargetDifficultyFromAdaptiveState\(", local_eval), (
        "recordLocalEval no longer reads the target difficulty in its offline "
        "branch — the Colab rail has nothing to draw a step from"
    )
    # BOTH of its object returns — the offline branch and the backend one. The
    # caller handles one shape, so a branch that quietly drops a field is a mode
    # in which the bar silently stops moving while the other mode still works.
    eval_returns = re.findall(r"return \{[^}]*\}", local_eval, re.S)
    assert len(eval_returns) == 2, (
        f"recordLocalEval has {len(eval_returns)} object returns, expected one "
        "per mode (offline engine + backend POST)"
    )
    for branch in eval_returns:
        for field in ("targetBefore", "targetAfter", "pBefore", "pAfter"):
            assert field in branch, (
                f"a recordLocalEval branch stopped returning {field} — the caller "
                "would have to re-read global state AFTER the await, which is "
                "exactly the state that moved during it"
            )
    assert "target_difficulty_before" in local_eval and "target_difficulty_after" in local_eval, (
        "recordLocalEval no longer maps the backend's before/after difficulty "
        "fields — backend-mode learners get no step drawn"
    )

    # `finalize: false` on every einops-fallback call. That path is a normal
    # graded submit that merely could not run server-side, so the felt-difficulty
    # step still follows and the attempt has to stay PENDING for the rating to
    # land on. Finalizing early closes it out and /feedback then has nothing to
    # apply — which surfaces to the learner as "Feedback failed" on exactly the
    # einops questions. The Colab verdict now has the same step and the same
    # requirement (see the `_rateTorchAndAdvance` block below).
    submit_answer_js = api_js.split("async submitAnswer", 1)[-1].split("\n  /**", 1)[0]
    fallback_calls = re.findall(r"this\.recordLocalEval\([^)]*\)", submit_answer_js)
    assert fallback_calls, "submitAnswer no longer records the einops-fallback attempt at all"
    for call in fallback_calls:
        assert "finalize: false" in call, (
            f"submitAnswer einops fallback calls {call} without finalize:false — "
            "the attempt finalizes early and the felt-difficulty step that "
            "follows fails with 'Feedback failed'"
        )

    events_js = read(os.path.join(HERE, "events.js"))
    events_colab = events_js.split("_rateTorchAndAdvance = async", 1)[-1]
    assert re.search(r"record = await PracticeAPI\.recordLocalEval\(", events_colab), (
        "_rateTorchAndAdvance discards recordLocalEval's return value — the "
        "difficulty step it just caused is the one thing the rail can show"
    )
    # The Colab verdict asks how hard it felt, so it must leave the attempt
    # PENDING — `finalize: !wantsRating`, where wantsRating is the review mode.
    # Hard-coding `finalize: true` here (or dropping the option, which defaults
    # to true) closes the attempt out as unrated and the rating that follows
    # 400s with nothing to apply it to.
    rate_call = events_colab.split("recordLocalEval(", 1)[-1].split(");", 1)[0]
    assert "finalize: !wantsRating" in rate_call, (
        "_rateTorchAndAdvance must post finalize:!wantsRating — the Colab "
        "review asks for a felt-difficulty rating and the attempt has to stay "
        "pending for that rating to land on"
    )
    # ...and the rating step is only offered when an attempt really is parked.
    # A placement diagnostic creates none, and an older backend does not answer
    # the field at all; showing the buttons in either case posts a /feedback
    # that fails.
    assert "record.pending === true" in events_colab, (
        "the Colab review shows the felt-difficulty buttons without checking "
        "that an attempt is pending — during a placement diagnostic there is "
        "none, and the rating fails"
    )
    assert "showFeedbackButtons()" in events_colab, (
        "the Colab review branch no longer offers the felt-difficulty rating"
    )
    assert "_drawColabDifficultyStep(q, record)" in events_colab, (
        "the Colab review branch no longer draws the difficulty step on the "
        "unrated path"
    )
    draw = events_js.split("_drawColabDifficultyStep = (", 1)[-1].split("\nconst _rateTorch", 1)[0]
    for call in ("animateTargetDifficulty(", "setTargetDifficultyFinal(", "setTargetDifficultyUnavailable("):
        assert call in draw, (
            f"_drawColabDifficultyStep lost its {call}…) call — reuse bars.js, "
            "never a second animation of the same quantity"
        )
    # ...and the shim it calls applies nothing on its own. It used to set the
    # final reading itself AND run the callback that sets the same reading,
    # which drew the caption twice per rating and stepped straight past the
    # stale-question guard above.
    bars_js = read(os.path.join(HERE, "bars.js"))
    shim = bars_js.split("function animateTargetDifficulty(", 1)[-1].split("\n}", 1)[0]
    assert "setTargetDifficultyFinal(" not in shim, (
        "animateTargetDifficulty applies the final difficulty itself again — "
        "both call sites already do, one of them behind a guard this bypasses"
    )
    assert "PracticeAPI.currentQuestion !== q" in draw, (
        "_drawColabDifficultyStep lost the stale-question guard — the tween "
        "runs for most of a second and Next problem is already on screen, so "
        "its last frame can land on the following problem's card"
    )
    # There is one readout on this screen now, so nothing has to be labelled
    # against a neighbour any more — the scope override that did that
    # (`setTargetDifficultyScope`) is gone with the bar it disambiguated.
    # What still has to hold is that the rail SHOWS the ladder. It used to be
    # `.question-meta-row`, then `.difficulty-bar`, and hiding either took the
    # difficulty readout off this deploy entirely; the same hole is one
    # `display:none` away today.
    # Comments are stripped first: the blocks in that file still name the
    # deleted widgets in prose, so a substring test over the raw file would pass
    # on the explanation alone.
    colab_css = re.sub(r"/\*.*?\*/", "", read(
        os.path.join(SHARED, "styles", "practice", "colab-edition.css")), flags=re.S)
    hidden_selectors = set()
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", colab_css):
        if "display:none" in body.replace(" ", ""):
            hidden_selectors.update(s.strip() for s in selectors.split(","))
    assert not any(s.endswith(".stage-ladder") for s in hidden_selectors), (
        "colab-edition.css hides .stage-ladder — support progress would "
        "disappear from the side-panel edition"
    )
    # And it must still be re-laid-out for a side panel. Four sections on one
    # row is a 1600px page's layout; at ~300px each rung name truncates, and a
    # rung the learner cannot read is the one thing the ladder is for.
    assert "html.dd-colab-edition .stage-ladder" in colab_css, (
        "colab-edition.css lost narrow-rail stage-ladder override"
    )

    # The Python half is checked by RUNNING it. These are the three transitions
    # the offline mastery numbers rest on, and all three fail silently: the UI
    # advances, the grade shows, and only the counters stay wrong.
    sys.path.insert(0, SHARED)
    try:
        import practice_engine as pe
    finally:
        sys.path.pop(0)
    blank = json.dumps({
        "user_id": "watch", "custom_weights": {}, "subtopic_states": {},
        "atom_mastery": {}, "atom_last_ts": {}, "self_reported_level": None,
        "pending_attempt": None,
    })
    api, sub = pe.engine_api, "Core array literacy"

    # 1. Grade-then-commit, the Colab route: four attempts, three right.
    state = blank
    for i, ok in enumerate((True, True, False, True)):
        state = api.submit_answer(state, 100 + i, sub, 50, ok)
        state = api.send_feedback(state, pe.UNRATED)
    counted = json.loads(state)
    row = counted["subtopic_states"][sub]
    assert row["n"] == 4 and len(row["history"]) == 4, (
        f"four graded attempts counted as n={row['n']} — submit_answer only "
        "parks the attempt; send_feedback is what counts it"
    )
    assert abs(row["p"] - 0.75) < 1e-9, (
        f"recent accuracy came out {row['p']} on three of four correct — p is "
        "read as a mastery number by the concept graph, so a stale default "
        "renders as a confident 50% on a subtopic the learner has aced"
    )
    assert counted["pending_attempt"] is None, "the committed attempt is still pending"

    # 1b. …and the staircase MOVES, in the direction the band is painted. The
    #     Colab rail draws green when this number goes up and red when it goes
    #     down, so a staircase that stopped stepping would not error anywhere —
    #     it would draw a bar that never moves off the same value, which reads
    #     as "the tutor ignored my answer".
    state, ladder = blank, []
    for i, ok in enumerate((True, True, False)):
        before = json.loads(state).get("subtopic_states", {}).get(sub, {}).get("target_difficulty")
        state = api.send_feedback(api.submit_answer(state, 400 + i, sub, 50, ok), pe.UNRATED)
        ladder.append((ok, before, json.loads(state)["subtopic_states"][sub]["target_difficulty"]))
    assert ladder[1][2] > ladder[1][1], (
        f"a correct answer left the target difficulty at {ladder[1]} — the rail "
        "paints that band green, so it would be claiming a step that never happened"
    )
    assert ladder[2][2] < ladder[2][1], (
        f"a miss left the target difficulty at {ladder[2]} — the rail paints that "
        "band red and it would be pointing the wrong way"
    )

    # 2. Nobody rated it: the next attempt flushes the last one, so Skip and a
    #    closed tab cost at most the attempt still on screen.
    state = blank
    for i, ok in enumerate((True, False, True)):
        state = api.submit_answer(state, 200 + i, sub, 50, ok)
    unrated = json.loads(state)["subtopic_states"][sub]
    assert unrated["n"] == 2, (
        f"n={unrated['n']} after three unrated attempts — record_attempt must "
        "flush the previous pending attempt before parking a new one"
    )
    flushed = json.loads(api.flush_pending(state))["subtopic_states"][sub]
    assert flushed["n"] == 3, "flush_pending did not count the attempt on screen"

    # 3. A real rating is recorded once, as itself — the flush must not turn the
    #    learner's answer into "unrated", nor count the attempt twice.
    state = api.send_feedback(api.submit_answer(blank, 300, sub, 50, True), "a_lot")
    rated = json.loads(state)["subtopic_states"][sub]
    assert rated["n"] == 1 and rated["history"][0]["feedback"] == "a_lot", (
        "a rated attempt must be counted once, with the level the learner gave"
    )
def check_a_torch_question_never_grades_on_pyodide():
    """Routing is RUN here, not pattern-matched.

    The substring check in `watch.py` says the guard clause exists; it cannot
    say the guard is on the right side of the branch. This lifts the two
    shipped predicates and the routing expression itself and asks the question
    that actually matters: given this question and this code, where does the
    submission go?

    The bug it exists for: `requiresLocalPyodide` was `questionNeedsEinops(q)`
    alone, so every einops question went to Pyodide — which cannot import torch,
    and the bank's einops questions are torch questions. Submit came back
    "can't run in the browser sandbox", nothing was recorded, and on a placement
    probe there was no way past the question at all.

    Both directions are asserted. Sending torch to Pyodide is the dead end;
    sending the numpy/visual einops questions to the backend would strand the
    other half, because the local preamble is what defines their helpers.
    """
    ui = read(os.path.join(HERE, "ui.js"))
    visuals = read(os.path.join(HERE, "visuals.js"))
    api = read(os.path.join(HERE, "api.js"))

    needs_torch = ui[ui.index("const TORCH_IMPORT_RE"):ui.index("const TORCH_UNAVAILABLE")]
    assert "function needsTorchRuntime" in needs_torch, (
        "needsTorchRuntime moved — the routing probe can no longer lift it"
    )
    needs_einops = visuals[
        visuals.index("function questionNeedsEinops"):visuals.index("function questionNeedsArenaArray")
    ]
    routing = re.search(
        r"const requiresLocalPyodide =\s*(.+?);", api, re.S
    )
    assert routing, "submitAnswer no longer decides where a submission grades"

    # The slice already carries `questionIsTorch`, which `needsTorchRuntime`
    # calls — lifting it again would redeclare the identifier.
    probe = f"""
{needs_torch}
{needs_einops}
const route = (question, userCode) => {{
  const requiresLocalPyodide = {routing.group(1).replace("this.currentQuestion", "question")};
  return requiresLocalPyodide ? "pyodide" : "backend";
}};
const eq = (got, want, why) => {{
  if (got !== want) {{ console.error(`FAIL ${{why}}: got ${{got}} want ${{want}}`); process.exit(1); }}
}};
const einopsTorch = {{topic: "Einops", primary_library: "einops",
  test_cases: [{{setup_code: "import torch as t"}}]}};
const einopsNumpy = {{topic: "Einops", primary_library: "einops",
  test_cases: [{{setup_code: "import numpy as np"}}]}};
eq(route(einopsTorch, "rearrange(x, 'h w -> () h w')"), "backend",
   "an einops question whose SETUP imports torch cannot grade on Pyodide");
eq(route(einopsNumpy, "import torch as t\\nt.zeros(3)"), "backend",
   "the learner reaching for torch is enough — Pyodide still cannot run it");
eq(route(einopsNumpy, "np.reshape(x, (1, 2))"), "pyodide",
   "numpy/visual einops keeps the local preamble that defines its helpers");
eq(route({{topic: "Numpy", supports_visual_output: true}}, "arr.mean()"), "pyodide",
   "a visual question still needs the local instance");
"""
    proc = subprocess.run(["node", "-e", probe], capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def check_a_deleted_practice_notice_stays_deleted():
    """Three surfaces Seth removed on 2026-08-23, and the bar that replaced one.

    Each of these grows back easily, because each one reads like an
    improvement in isolation:

      #cold-start-badge          two blocks of standing explanation above every
                                 calibration/placement question
      #practice-mode-intro       "Practice is your adaptive queue…"
      #practice-mode-notice      the floating mode-demotion banner

    They are gone from markup, JS and CSS together. Re-adding any of them from
    one side only (a stylesheet rule with no element, or an element nothing
    styles) is the state this check exists to reject — and re-adding the DOM
    writer for the notice reintroduces a banner Seth asked twice to be rid of.

    ⚠️ Deleting the notice DID cost something real, recorded here so nobody
    re-derives it as a surprise: a silent demotion to the demo pool (expired
    token, backend down) now shows up only in the console, and the practice UI
    renders identically in that state. practice/mode.js carries the full note
    and the shape of an acceptable replacement.

    The progress bar is the other half: a placement in progress has to say how
    far through it is, and `budget` is a CEILING, so the bar must never claim a
    fixed length — the count says "of at most" and a tick marks the earliest
    possible finish.
    """
    index = read(os.path.join(SHARED, "index.html"))
    mode = read(os.path.join(HERE, "mode.js"))
    page = read(os.path.join(HERE, "diagnostic-page.js"))

    # Matched on the ATTRIBUTE, never the bare name: these files carry comments
    # explaining what was deleted, and a substring check over prose fails on its
    # own tombstone. (Cost one run to learn, twice now.)
    for gone in ('id="cold-start-badge"', 'id="cold-start-label"',
                 'id="cold-start-note"', 'class="cold-start-badge',
                 'id="practice-mode-intro"', 'id="practice-mode-notice"'):
        assert gone not in index, (
            f"{gone} is back in index.html — it was deleted on 2026-08-23 "
            "(markup, JS and CSS together); read practice/mode.js first"
        )
    assert "function showPracticeModeNotice" not in mode, (
        "the mode-demotion banner is back. If a demotion needs to be visible "
        "again, put it on the session status row — mode.js says why"
    )
    # ...and the CSS half. A rule for an element that no longer exists is the
    # half-deletion this check is named for: it reads as live styling to the
    # next person and invites the markup back. (codex flagged that the check
    # claimed to cover CSS and did not.)
    styles = os.path.join(SHARED, "styles", "practice")
    for fname, sel in (("question.css", ".cold-start-badge {"),
                       ("question.css", ".cold-start-label {"),
                       ("question.css", ".cold-start-note {"),
                       ("misc.css", ".practice-mode-intro {"),
                       ("feedback.css", ".practice-mode-notice {")):
        assert sel not in read(os.path.join(styles, fname)), (
            f"{fname} still styles {sel.strip(' {')} — the element was deleted "
            "on 2026-08-23, so this rule matches nothing"
        )
    # The countdown outlived the badge that used to host it.
    assert 'id="placement-timer"' in index, (
        "#placement-timer went with the cold-start badge — the placement's "
        "fixed 2:00 clock has no anchor and every probe becomes untimed"
    )
    # ...and it is ON THE NOTCH TAB (Seth, 2026-08-23: the timer belongs on the
    # tab, not beside the concept heading). The anchor has now moved twice, so
    # what is asserted is the CURRENT home, in the markup rather than in the JS:
    # placement-timer.js reads it by id and no longer knows which row it is in.
    notch_tab = index.split('id="practice-notch-tab"', 1)
    assert len(notch_tab) == 2, "the notch tab is gone — #practice-notch-tab"
    tab_markup = notch_tab[1].split("</div>", 1)[0]
    assert 'id="placement-timer"' in tab_markup, (
        "#placement-timer left the notch tab. It is the placement's only "
        "countdown and notch-menu.js hides the session clock while it shows, "
        "so anywhere else means a probe timed off-screen"
    )
    timer_js = read(os.path.join(HERE, "placement-timer.js"))
    assert 'getElementById("cold-start-badge")' not in timer_js and (
        'getElementById("placement-timer")' in timer_js), (
        "placement-timer.js is not reading its element by id — _chip() returns "
        "null and the countdown never renders"
    )
    # 🔴 One clock on the tab. The placement runs outside a session, so
    # `_sessionOpen()` is false throughout and the session clock would sit
    # beside the probe's countdown, greyed at its idle allowance — two numbers,
    # one of them stopped.
    notch_js = read(os.path.join(HERE, "notch-menu.js"))
    assert '"placement-timer"' in notch_js, (
        "notch-menu.js no longer defers to the placement clock — the tab shows "
        "the idle session allowance next to a running probe countdown"
    )
    assert "PracticeNotch?.syncClock" in timer_js, (
        "placement-timer.js must poke the notch when it shows or hides its "
        "clock; nothing else observes this module and the session clock would "
        "not come back when the test ends"
    )

    # The progress bar: anchors present, and honest about the ceiling.
    for anchor in ("placement-progress", "placement-progress-fill",
                   "placement-progress-tick", "placement-progress-count"):
        assert f'id="{anchor}"' in index, f"index.html lost the #{anchor} anchor"
    assert "of at most" in page, (
        "the placement progress count must say 'of at most' — the test stops as "
        "soon as it is confident, so `budget` is a ceiling and not a length"
    )
    assert "min_probes" in page, (
        "the bar lost its earliest-finish tick, so it implies a run to the full "
        "budget that most placements never make"
    )
    assert 'host.classList.toggle("hidden", !show)' in page, (
        "the progress bar must hide outside an active placement — a finished "
        "test showing a part-full bar reads as unfinished"
    )
