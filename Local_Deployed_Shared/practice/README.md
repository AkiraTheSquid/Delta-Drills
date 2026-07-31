# practice

## Purpose
Practice-page frontend: loads ARENA-derived coding questions, runs the user's Python in-browser via Pyodide (or via a backend endpoint), grades the result, and renders image previews for visual exercises.

## Owns
- The practice-page UI lifecycle (init, question selection, code editor wiring, run/submit, result display).
- The Pyodide runtime contract and preamble (see `RUNTIME_CONTRACT.md`).
- Image rendering for visual exercises, including the static PNG fallback.
- Adaptive question selection, timing, and progress storage scoped to the practice flow.
- Imported-helper display for practice questions. The UI reads each starter code block and shows the imported names above the editor.

## Does NOT own
- Question authoring or curation — lives in `../arena_prereqs_structured.json` and the ARENA source dirs.
- Backend code execution — handled by the Fly.io backend behind `/api/practice/run-code`.
- AI grading prompts — system prompts live in `../function_mode_quality_*.txt` and `../chatgpt/`.
- Auth, navigation, top-level shell — handled by the host app.

## Key Files
- `init.js`: bootstraps the practice page, wires DOM and event handlers.
- `dom.js`: caches DOM element refs used across modules.
- `events.js`: top-level event handlers (run, submit, navigation).
- `engine.js`: question-selection / submission orchestration; talks to `api.js`.
- `api.js`: wrapper around `apiFetch` for `/api/practice/*` endpoints.
- `runner.js`: in-browser Pyodide runtime, `buildPyodidePreamble()`, run-button handler. **Runtime contract documented in `RUNTIME_CONTRACT.md`.**
- `visuals.js`: target-image rendering for visual exercises; multi-path `.npy` lookup, canvas rasterization, static PNG fallback.
- `ui.js`: question rendering into the editor and prompt area. Also owns `torchNeedsColab` / `applyTorchRouting` — the binary "notebook card OR editor panel" swap, never both.
- `colab_mode.js`: **which edition this deploy is, and where a question's notebook lives.** `delta-drills-colab.vercel.app` (or `?mode=colab`) is the Colab edition: a drill routes to its published lesson notebook instead of the in-page editor. Both deploys ship IDENTICAL code and ask `location.hostname` who they are, so there is no branch to keep in sync. Exposes `window.DDColab = {active, hrefFor, lessonFor, whenReady}`; must load before `ui.js`, which asks it at load time. Inert on the normal app — no fetch, no DOM change.
- `ai.js`: AI-judge submission path.
- `mode.js`: practice mode (backend vs local-pyodide) selection.
- `adaptive.js`: adaptive question difficulty/selection.
- `questions.js`: client-side question caching/normalization.
- `storage.js`: progress / streak persistence.
- `timer.js`: resumable practice session — learner sets question count + strict answer/review time per question up front (`#practice-session-setup`), then `PracticeSession` enforces them while active. `Pause & exit`, reload, or browser close writes a per-user localStorage snapshot containing the current question, block position, phase, remaining time, editor draft, and grade review; setup offers Resume/Discard on return. Answer expiry force-submits (or advances when nothing is submittable); review expiry auto-rates the default felt difficulty and advances. Hooks: `onQuestionRendered` (ui.js), `pauseForGrading`/`recordReviewResult`/`resumeAnswerPhase`/`beginReviewPhase` (events.js submit), `shouldFinishInsteadOfAdvance` (events.js `_loadNextPracticeQuestion` quota gate). Setup defaults persist in `localStorage.delta_drills_session_setup`; active session key derives from the per-user practice progress key. Styles in `../styles/practice/timer.css`.
- `bars.js`, `bars.css`: progress bars.
- `arena-unlock.js`, `arena-unlock.css`: ARENA UNLOCK INTERSTITIAL — full-viewport overlay that takes over the screen between practice questions when an ARENA exercise crosses its per-subtopic prereq thresholds. Mount point `<div id="arena-unlock-overlay">` lives at body level in `index.html` (not inside `page-practice`) so the takeover escapes any layout/overflow trap. Controller exposes `window.ArenaUnlock.tryShow(onContinue)`; `events.js#nextProblemBtn` calls it before loading the next question and passes `_loadNextPracticeQuestion` as the Continue callback. Reads its data via `window.ARENA_PREREQS_TEMP_*` (defined in `stats/predicted-prereqs-temp.js`); the underlying concept-graph data source is scaffold-only and will be replaced, but this unlock UI is permanent.
- `kc-practice.js`: SINGLE-KC PRACTICE LADDER for the concept graph's "Practice ⤢" overlay (`index.html?lesson=<kc>&embed=1`). Serves the `faded_items` → `guided_items` → `independent_items` lists that `lessons_structured.json` has always carried but nothing consumed. Three rungs of decreasing support: faded hands over a blanked solution, guided hands over the KP's authored hints (through the practice card's existing Show-hint button), independent hands over nothing. Faded items are real bank questions re-served with the KP's blanked starter, so grading/BKT/FIRe run through the normal submit path — this is NOT a parallel scoring system. Scaffold order comes from the learner's posterior using the same ERE band as `arena-unlock.js` (< 0.75 ⇒ faded first). `PracticeAPI.getNextQuestion()` calls `KcPractice.nextQuestion()` while `isActive()`; once the ladder is spent the normal adaptive queue resumes, still pinned by `window.__kcFocusSubtopics`.
- `competency-bar.js`: the mastery bar at the top of that overlay — phase heading (Lesson → Faded drills → Independent problems, driven by the tier actually served, not by the mastery band), a track with both real engine thresholds marked (0.85 unlock, 0.95 mastered), and an rAF tween on each graded attempt. Listens for `competency:feedback-update` (emitted by `events.js`), emits `competency:gate-crossed` + a same-origin `postMessage({type:"delta:kc-mastered"})` that `concept-graph/lesson-graph.js` uses to close the overlay and animate the node. Styles in `../styles/practice/misc.css`.
- `config.js`: feature flags and tunables.
- `RUNTIME_CONTRACT.md`: declarative contract of what the Pyodide preamble injects; consumed by per-question `runtime_dependencies` arrays.

## Data & External Dependencies
- **Question schema**: `arena_prereqs_structured.json` (sibling); per-entry fields include `starter_code`, `canonical_solution`, `runtime_dependencies`, `runtime_unmet_dependencies`, `supports_visual_output`, `expected_artifact_type`.
- **Notebook helpers panel**: practice questions now show clickable excerpts pulled from the source ARENA notebook, not the editable starter stub. Click a pill to expand the exact notebook line(s). For prereq exercises this includes setup lines like `arr = np.load(section_dir / "numbers.npy")` and the actual notebook import block when it matters. Array pills also render the real `numbers.npy` data from `content/ARENA_5.0-main/...` so the user can inspect the source tensor itself.
- **ARENA source data**: `../content/ARENA_5.0-main/chapter0_fundamentals/exercises/part0_prereqs/numbers.npy`, `../delta_numbers.npy` (mirrored to Pyodide FS), `../numbers_stacked.png` (static fallback).
- **Pyodide** (CDN): numpy + micropip + einops loaded at runtime.
- **Backend**: `/api/practice/run-code`, `/api/practice/visual-debug`.
- **Supabase**: progress / streaks via `../supabase-practice.js`.

## How It Works (Flow)
1. `init.js` boots the page; `engine.js` fetches the next question via `api.js`.
2. `ui.js` renders prompt + starter code into the Monaco editor.
3. If the question has `supports_visual_output`, `visuals.js` runs the canonical solution in Pyodide and rasterizes the resulting array onto a canvas. On failure, falls back to `numbers_stacked.png`.
4. User clicks Run → `runner.js` either POSTs to the backend or executes locally in Pyodide using the preamble from `buildPyodidePreamble()`.
5. Submit → `ai.js` (AI-judged) or backend grader returns pass/fail; result feeds `storage.js` and adaptive selection.

## Invariants & Constraints
- The Pyodide preamble in `runner.js` is the **single source of truth** for in-browser injected globals; `RUNTIME_CONTRACT.md` and per-question `runtime_dependencies` must stay in sync.
- Image visuals must degrade gracefully — never leave a blank canvas. Catch path in `visuals.js` must end in either rendered canvas or text note.
- `numbers_stacked.png`, `delta_numbers.npy`, and the ARENA `numbers.npy` source file must remain at served paths; multi-path candidate lookup tolerates moves but `/content/ARENA_5.0-main/...` is the canonical source path for prereq notebook data.
- Never inject runtime globals silently without updating both `RUNTIME_CONTRACT.md` and the JSON probes that derive `runtime_dependencies`.
- AI-judged grading must not rely on `assert_*` helpers being injected — those are not in the in-browser contract.
- **Subtopic naming differs by mode and BKT is keyed by whatever `question.subtopic` says.** Backend `question.subtopic` is composite (`"Numpy: Core array literacy"` — `app/questions.py` prefixes the topic so Numpy/Einops subtopics stay distinct); the local `questions.json` bank uses the bare name and keeps the composite in a separate `subtopic_key` field. Anything matching against a subtopic across both modes must accept EITHER form — that is why `window.__kcFocusSubtopics` is a list and `CompetencyBar.init()` takes an array. `window.__kcFocusSubtopic` (singular) is the composite form only, for the backend's `focus_subtopic` query param.
- **Questions with no lesson-graph KC are PARKED, not served.** `questions.js` builds `kcTaggedIds` from `lessons/qmatrix_tags.json`; `servableQuestions()` / `isKcServable()` back `getPracticeEligibleQuestions()` and `isPracticeQuestionAllowed()`. `questionsBank` itself stays complete so saved/in-flight questions still hydrate — mirror of the backend's complete-by-id / narrowed-pools split. This filter **fails OPEN on purpose**: if the q-matrix can't be fetched nothing is filtered, because an unfiltered offline queue beats an empty one and the backend enforces parking authoritatively for signed-in practice. See `docs/decision-kc-only-serving.md`.
- `buildPracticeQuestionFromBank()` in `questions.js` is the single bank-record → `currentQuestion` mapping. `api.js` and `kc-practice.js` both use it; do not re-inline it, or a faded item would grade against different fields than the same question served by the queue.

## Extension Points
- **Add a new injected global**: edit `runner.js:buildPyodidePreamble()`, update `RUNTIME_CONTRACT.md`, re-run the dependency-probe script that populates `runtime_dependencies` in the JSON.
- **Add a new question topic**: extend `questionNeedsEinops` / `questionNeedsArenaArray` in `visuals.js`, ensure the topic's data assets exist in `Local_Deployed_Shared/`, and add probes to the annotator script.
- **Add a per-question fallback image**: set `fallback_image_url` on the question; `visuals.js:getArenaNumbersPngCandidates()` will prefer it.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Run routed torch code to a runtime that has no torch** — `RESOLVED`
  - When it happens: pressing Run on any converted Einops/Einsum drill, or on a torch lesson's worked example, in backend mode.
  - Symptom: `ModuleNotFoundError: No module named 'torch'` from Run, while Submit grades the same code correctly.
  - Root cause: `runner.js` forced local Pyodide for lessons ("keep experimentation local") and for `questionNeedsEinops` (which keys on the Einops/Einsum TOPIC, not the library). Both rules predate the torch conversion and silently began covering torch code. Pyodide cannot import torch at all.
  - Prevention/fix: both rules now yield when the question or the editor contents import torch, so it runs on the backend fork runner. Outside backend mode there is no fallback, so Run says so plainly instead of surfacing the import error. When adding a "keep this local" rule, ask what happens to torch code under it.
  - Status: RESOLVED (2026-07-27).
- **Image preview blank when Pyodide errors** — `RESOLVED`
  - When it happens: visual-output question, Pyodide fetch/exec fails.
  - Symptom: empty canvas with only an error string.
  - Root cause: catch block had no fallback path.
  - Prevention/fix: `visuals.js:renderQuestionVisual` now calls `renderFallbackImage()` before showing the error note. Keep a static PNG at a served path.
  - Status: RESOLVED (2026-04-28).

- **Torch-dependent prereq questions throw `NameError: 't'`** — `ACTIVE`
  - When it happens: einops-flagged question (forced to local Pyodide by `runner.js:148`) also references `t.tensor`/`Tensor`/`assert_all_*`.
  - Symptom: run output is `NameError`. 18 of 27 prereq questions are affected; see `runtime_unmet_dependencies` per question.
  - Root cause: in-browser preamble does not inject torch or ARENA test helpers; routing forces einops questions to Pyodide regardless.
  - Prevention/fix: route questions with non-empty `runtime_unmet_dependencies` to the backend, OR inject a torch/numpy compat shim in `buildPyodidePreamble`. See `RUNTIME_CONTRACT.md` for the full unmet-dep table.
  - Status: ACTIVE.

- **Question gives away answer in starter docstring** — `ACTIVE` (design call)
  - When it happens: function-impl questions where the docstring shows expected output literally (e.g. id 9).
  - Symptom: low cognitive challenge — user can read the answer from the editor.
  - Root cause: ARENA upstream pedagogy embeds expected outputs in docstrings.
  - Prevention/fix: pedagogical trade-off; if changing, scrub docstrings via the export script and verify tests still pass.
  - Status: ACTIVE — keep as-is unless explicitly redesigning.

## Recent Changes
- 2026-07-31 (the Colab edition — a second Vercel project, same code): New `colab_mode.js`, loaded before `ui.js` (`index.html`, `ui.js?v=24`). `delta-drills-colab.vercel.app` serves the identical frontend and routes a drill to its lesson notebook; `delta-drills.vercel.app` is untouched. Deployed by `This-Directory-Only/scripts/deploy_delta_drills_colab.sh` off the `deploy-colab` branch, sharing the Fly backend, Supabase state and question bank — practice on either moves the same mastery record.
  - **The route was unreachable before this, and not because of a flag.** `torchNeedsColab` required `q.problem_notebook_path || q.solution_notebook_path`, and both are empty on all 499 bank rows; forcing `practiceMode` off `backend` would still have produced nothing. The join runs `question id → colab_notebooks.json.questions → <lesson>.ipynb`, the map `scripts/generate_colab_notebooks.py` writes beside the notebooks it publishes.
  - 🔴 **Join on the QUESTION, never on `subtopic_key`.** A lesson notebook holds only the problems its KPs reference, not every question in the subtopic. The first cut matched by subtopic and "resolved" 387 questions — including q1 → `np-3`, a notebook with no `dd-q1` cell in it. That is the dead end this route is supposed to refuse, one level finer: the editor hidden, the learner sent to a notebook to find a problem that was never in it. `audit_colab_notebook_index.py` (deploy preflight) confirms all 424 mapped questions against the notebooks PUBLISHED on GitHub, not the ones in the working tree.
  - **424 of 499 questions route; the other 75 stay on the editor.** Routing hides the editor, the aids and the submit bar, so `colabNotebookHrefFor` returning `""` is the guard, and `torchNeedsColab` keeps the pre-existing "somewhere to route TO" rule for exactly the same reason.
  - The index arrives over the network, so `ui.js` re-runs `applyTorchRouting` for the on-screen question once `DDColab.whenReady` fires; without that the first question of a session renders an editor that cannot import torch.
- 2026-07-31 (subtopic enable/disable lost its UI): The Statistics tab was deleted, and its Advanced table was the only place a learner could toggle a topic or subtopic off. The flags still live in `delta_drills_weights` and are still read by `getPracticeEligibleQuestions` / `isPracticeQuestionAllowed` in `questions.js` (and by `adaptive.js` for weighting), so anyone carrying disabled flags would have been stuck with no way to switch them back on. `getPracticeEligibleQuestions` now ignores the flags when they would empty a non-empty bank — a narrowed queue is a preference, an empty one is a dead end. The matching `api.js` error no longer tells the user to "re-enable at least one area in Statistics", which pointed at a tab that no longer exists.
- 2026-07-30 (guided practice is served; every tagged bank question reaches a node): `kc-practice.js` gained the middle rung. `guided_items` had been compiled into `lessons_structured.json` since the segmented format shipped and were rendered only by the offline `viewer.html` — the ladder went faded → independent and skipped them, so 52 authored hint sets were dead data. The queue is now faded → guided → independent (reversed above the 0.75 band, guided still in the middle), and the KP's numbered hints ride through as the question's `hint`, which the practice card's existing Show-hint button already renders. `.hint-text` gained `white-space: pre-line` so the escalation stays one hint per line, and `competency-bar.js` gained a "Guided problems" phase. Content side: an audit found 78 bank questions tagged to a KC whose KP never referenced them — dead pool. 77 are now wired in (12 as new guided items with authored hints, filling KPs that had none; 65 as independent). Referenced questions went 338 → 415 of 416 tagged. The one left out is **q387, an exact duplicate of q322** (same einops pattern, same output shape, differing only in which fixture slice the example uses) — it is still in the bank, just not taught. Cross-KP reuse remains zero: no question is practice for two different KCs. Cache-bust `kc-practice.js?v=3`, `competency-bar.js?v=4`, `misc.css?v=3`.
- 2026-07-30 (lesson notebook — cells throughout the prose, real output): `notebook.js` now mounts every runnable fence on a lesson page, not just the worked example's, so a concept is taught as a sequence of blocks the learner can run where the prose raises them. The runnable/static split is the authoring format's own marker: `md()` in `lessons.js` carries the fence info string through as `data-fence`, a plain ` ```python ` fence becomes a cell and ` ```python no-run ` stays static — the same rule `validate_lessons.py` uses, so a cell exists if and only if CI runs it. `lessons.js` marks the two regions CI executes with `.nb-scope` (concept body + worked example); a fence under "Watch out" is out of scope. Cells run through a Python harness (`_delta_cell`) that gives three things a plain concat could not: a Jupyter-style echo of a bare trailing expression, a summary of the names a silent cell bound instead of "no printed output", and cell-relative traceback line numbers. Only the clicked cell's stdout is shown — the prefix re-runs to rebuild state, and its prints already sit under its own Run buttons. Cells keep an `In [n]` counter and a run/failed border. **`validate_lessons.py` now uses a FRESH NAMESPACE PER SEGMENT**, because the lesson screen shows one segment per page: a fence depending on an earlier segment's names would pass a KP-wide namespace and hand the learner a `NameError`. All 122 segments already passed that. Pilot content: `lessons/numpy/kp-ndarray-model.md` rewritten to 3/2/2 concept cells across its three segments. Cache-bust `notebook.js?v=3`, `lessons.js?v=19`, `notebook.css?v=2`.
- 2026-07-27 (torch Run routing): `runner.js` no longer pins torch code to Pyodide. The lesson rule and the `questionNeedsEinops` rule both defer when the question or editor imports torch; guest mode gets an explicit "open it in Colab" message rather than an import error. Needed because the 159 Einops/Einsum drills are now torch while keeping their Einops/Einsum topic.
- 2026-07-27 (KC-only serving): `questions.js` now parks questions with no `target_kcs` — `loadKcTaggedIds()`, `servableQuestions()`, `isKcServable()`, wired into `getPracticeEligibleQuestions()` and `isPracticeQuestionAllowed()`. 380 servable / 75 parked (CNN, PyTorch Fundamentals, Autograd, Optimizers). `lessons/` is now the source of truth for what the tutor may teach, validated chapter by chapter; the parked questions return when their chapter's KCs are authored and validated. Cache-bust `questions.js?v=24`.
- 2026-07-24 (single-KC practice ladder + competency bar): Replaced the `?lesson=<kc>` preview dead-end (it ended on "Demo lesson complete") with the real lesson → faded → independent flow. New `kc-practice.js` + `competency-bar.js`; `questions.js` gained `buildPracticeQuestionFromBank()` (extracted from `api.js`) and the `__kcFocusSubtopics` filter; `api.js` routes through the ladder and sends `focus_subtopic` to the backend; `events.js` emits `competency:feedback-update`; `lessons.js` exports `getKpEntry()`. No session quota or timer here — this flow ends on the 0.95 mastery gate, not a question count, so `PracticeSession` stays inactive throughout. Known limits: the bar is per-SUBTOPIC (BKT's granularity), not per-concept, so KCs sharing a subtopic share a bar; and the local Pyodide engine never writes `atom_mastery`, so the graph's node recolouring only moves in backend mode.
- 2026-07-12 (rigid practice sessions): Replaced optional timed mode with mandatory session setup. Question count and per-question answer/review limits stay strict while active; answer expiry auto-submits, review expiry auto-rates "About right" + advances, and the block ends at its committed count.
- 2026-07-14 (pause/resume): Added `Pause & exit` plus Resume/Discard. Session cursor, countdown, current editor draft, and review result persist per user; reload/browser close also yields a resumable session.
- 2026-05-19 (ARENA unlock interstitial — full-viewport overlay module): Added `practice/arena-unlock.js` + `practice/arena-unlock.css` as a self-contained module. When the student clicks "Next problem" in the practice flow, `events.js` now calls `window.ArenaUnlock.tryShow(_loadNextPracticeQuestion)` first; if any ARENA 0.0 exercise has just crossed its per-subtopic prereq thresholds (AND-logic across every `{topic, subtopic, minPct}` entry — no averaging), the overlay takes over the whole viewport (`position: fixed; inset: 0; z-index: 9000;` + `body.arena-unlock-open` scroll-lock) showing: the exercise title, a "Cleared: ..." why-met recap, a code block with the exact heading (Ctrl+F target), Show hint / Show answer scaffolding buttons (placeholders for now), `Open in Colab ↗` (auto-copies heading), and `Continue to next question →` (calls the original load-next handler). The overlay DOM lives at body level in `index.html` (`<div id="arena-unlock-overlay">`), not inside `page-practice`, so the takeover is independent of any page's layout. Each unlocked exercise is shown once per student (tracked in `localStorage.arena_prereqs_temp_shown`). Underlying concept-graph data is still the temp scaffold in `stats/predicted-prereqs-temp.js`; only the labels change when the real graph ships. `watch.py` invariants assert the overlay DOM + body class + CSS selectors are wired.
- 2026-04-28: Added `RUNTIME_CONTRACT.md`, `numbers_stacked.png` fallback, `runtime_dependencies` annotations on all 27 prereq questions; fixed id-27 einsum canonical solution.
- 2026-05-13: Added an imported-helpers panel above the practice editor so the page shows the code's real imports without the source-notebook block.
- 2026-04-27: Initial doc created.
