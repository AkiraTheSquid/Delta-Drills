"""watch_invariants.py — the big invariant sweep for practice.

Split out of `watch.py` (Modulario, 2026-08-19): the file was over the 700-LOC
line and `check_invariants` was a third of it on its own. The check itself is
unchanged and still runs from `watch.py`'s list — see the guard at the bottom of
that file, which fails if a check defined here is not in the list.
"""
import json
import os
import re
import sys

from watch_common import HERE, SHARED, read


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
        "colab-edition.css hides .stage-ladder — that is the whole progress "
        "readout on this deploy, concept and rung and difficulty together"
    )
    # And it must still be re-laid-out for a side panel. Four sections on one
    # row is a 1600px page's layout; at ~300px each rung name truncates, and a
    # rung the learner cannot read is the one thing the ladder is for.
    assert "html.dd-colab-edition .stage-ladder-track" in colab_css, (
        "colab-edition.css lost the narrow-rail override for the ladder track "
        "— four sections on one row truncate to nothing in a side panel"
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


