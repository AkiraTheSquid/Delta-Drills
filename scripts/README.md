# scripts

## Purpose
- Authoring/build tooling for the first-encounter lesson course: parse KP markdown, validate it against the drill bank, compile it to the JSON the app serves.

## Owns
- The KP markdown format contract (`lesson_lib.py` parser: frontmatter, sections, single-concept segments).
- Lesson validation (`validate_lessons.py`): fences execute, faded solutions pass bank test cases, per-segment rules, registry acyclicity, coverage.
- Compilation to `lessons_structured.json` (`compile_lessons.py`) and Q-matrix generation (`build_qmatrix.py`).

## Does NOT own
- Lesson content itself (`Local_Deployed_Shared/lessons/`), runtime lesson gating (`Local_Deployed_Shared/practice/lessons.js`, backend `app/lessons.py`), or the drill bank.

## Key Files
- `lesson_lib.py`: shared parser — frontmatter, `split_sections` (ordered, repeatable), `build_segments` (one-concept segments), bank/registry loaders. HIGH fan-in: the other three scripts import it.
- `validate_lessons.py`: executes every code fence in document order, grades every faded solution against bank test_cases, enforces at-least-one worked + one-or-two faded exercises per segment, and applies `lesson_quality.py` as hard errors to pages on the new standard. Run with `--coverage` as the full gate.
- `lesson_quality.py`: the written standard for a worked example — INTRO, INTERLEAVE, PRINTS, CASES, GIVEAWAY, PROMPT_LEAK. Every rule is a defect that shipped, passed the structural validator, and was caught by a learner working the page. `strict_for(kp)` gates enforcement on the page having an `## Applied practice` section, so upgrading a page means opting into the standard rather than the standard failing 63 pages at once. Imported by both `validate_lessons.py` (errors) and `audit_ladder_pairing.py` (backlog report).
- `compile_lessons.py`: emits `lessons_structured.json` with per-KP `segments` plus legacy aggregate fields (viewer.html back-compat).
- `build_qmatrix.py`: derives `qmatrix_tags.json` question→KC tags from KP refs + hand-assigned leftovers.
- `generate_colab_notebooks.py`: compiles the nine Colab notebooks from
  `lessons_structured.json` + the question bank, and writes the
  `question -> notebook` map TWICE from one run — `extension/panel/notebook-index.js`
  (a `<script>` the MV3 side panel loads) and
  `Local_Deployed_Shared/lessons/colab_notebooks.json` (fetched by the web app).
  Both consumers navigate by that map, so it must come from the same pass that
  wrote the `.ipynb` files; `Local_Deployed_Shared/practice/watch.py` fails if
  the two copies disagree. Emits nbformat 4.5 with a deterministic
  `metadata.id` per cell (`dd-q<question_id>` for a problem, `dd-kp-<slug(kc)>`
  for a concept section) — that id is what Colab's `#scrollTo=` matches, and it
  is the only reason a link can land on a specific problem.
  **This script is the whole student-facing surface now.** Since 2026-07-31 the
  practice panel renders no problem text, no worked example and no lesson: the
  notebooks it writes carry all of it, and the panel only routes to them. The
  map it emits therefore carries a `kps` entry (`kc -> "dd-kp-<slug(kc)>"`) so
  the first-encounter gate can link to the section that teaches a concept —
  the anchor only, since `kcs` already says which notebook it is in. The
  anchor is computed here and shipped, never re-derived in JavaScript — a slug
  that drifted by one character is an anchor Colab silently ignores.
- `colab_grader.py`: the `dd_check(<problem id>)` a learner runs in the
  notebook. Not imported by anything — everything between its `embed:start` /
  `embed:end` markers is copied verbatim into one cell per notebook, because a
  notebook opened from GitHub cannot import a file that lives in this repo. It
  is a **fourth** implementation of the grading rule (`backend/app/code_runner.py`,
  the Pyodide harness in `practice/api.js`, and `validate_lessons.py` are the
  others), so `watch.py` execs it and grades the two cases that have already
  caught drift once each. Run it directly for a smoke test. It also carries the
  `np.load('/delta_numbers.npy')` rewrite the backend does in its preamble —
  without it the 24 ARENA-image einops drills cannot run in Colab at all.
- `publish_colab_notebooks.sh`: pushes those notebooks to
  `<owner>/arena-book-colab/ARENA_5.0/ch-1-foundations`. Colab can only open a
  notebook from a URL, so **regenerating is not enough — unpublished changes are
  invisible to learners.** The default repo is baked into
  `practice/colab-route.js` as `DEFAULT_REPO`.

## Data & External Dependencies
- Reads `Local_Deployed_Shared/questions_structured.json` (bank), `lessons/kc_registry.json`, `lessons/*/kp-*.md`.
- numpy (fence execution); einops fixtures need `delta_numbers.npy` (validator rewrites the Docker path to the local copy).

## How It Works (Flow)
1. Edit KP markdown → `python3 scripts/validate_lessons.py --coverage`.
2. `python3 scripts/compile_lessons.py` → `lessons_structured.json` feeds BOTH the frontend player and the backend guard maps.

## Invariants & Constraints
- Every segment starts at `## Concept` and must carry exactly one worked example plus one faded exercise whose solution passes bank tests.
- Optional `## Watch out` belongs to current segment and compiles into lesson-only content.
- Compiler extracts each segment's sole Python worked fence as `worked_example_code`; LessonGate preloads it into editor.
- Grading must mirror prod: `expected_setup_code or setup_code` re-runs before `expected_expr` (fixed 2026-07-19 — validator previously evaluated expected on solution-mutated fixtures).
- Validate BEFORE compile; compilation does not re-check code.
- Frontmatter faded/guided id lists must equal the section ids (validator-enforced).
- **A stage-2 `example` currently lives ONLY in `lessons_structured.json`, which
  `compile_lessons.py` overwrites wholesale.** The pair is authored data with no
  authoring surface yet: `lesson_lib.py::parse_kp` does not read an `example`
  marker out of the KP markdown, so `_faded_items` cannot emit one and the next
  compile silently drops every pair on the floor. Anyone converting more
  segments must teach the markdown format the marker FIRST. Same story one step
  further out — `build_qmatrix.py` derives each question's rung from the
  markdown, so promoting a question to be an example (which spends it) has to
  reach the markdown before the backend's `ladder_rank` agrees with the
  notebook about which questions are still problems.

## Extension Points
- New KP rules → add checks in `check_kp`; format changes → `lesson_lib.py` first, then both consumers.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Validator expected-setup fallback mismatch** — `RESOLVED`
  - When it happens: faded solution mutates its input (out=-style drills, e.g. q59).
  - Symptom: validator FAILs a solution that grades correct in prod.
  - Root cause: validator evaluated `expected_expr` without re-running `setup_code` when `expected_setup_code` absent; JS/backend graders re-run it.
  - Prevention/fix: `grade_against_bank` now uses `expected_setup_code or setup` — keep all three graders' semantics in lockstep.
  - Status: `RESOLVED` (2026-07-19).
- **Validator missing the runtime's numpy preamble** — `RESOLVED`
  - When it happens: a torch-dialect lesson whose bank fixture loads the ARENA image (`np.load('/delta_numbers.npy')`).
  - Symptom: `NameError: name 'np' is not defined` from the validator on a drill that grades fine in the sandbox.
  - Root cause: `grade_against_bank` seeded its namespace from the solution's own imports only, while `code_runner.CODE_PREAMBLE` always injects numpy. A torch solution imports torch, so `np` never existed.
  - Prevention/fix: the grading namespace now imports numpy first — same lockstep rule as the expected-setup fix above. If the runtime preamble grows, mirror it here.
  - Status: `RESOLVED` (2026-07-27).
- **Relative paths rejected** — `ACTIVE`
  - When it happens: `validate_lessons.py some/relative/kp.md`.
  - Symptom: "not in the subpath" parse error.
  - Root cause: `path.relative_to(REPO)` needs absolute paths.
  - Prevention/fix: pass `"$PWD/..."` absolute paths.
- **A half-built `torch` imports "successfully"** — `RESOLVED` (notebook side)
  - When it happens: a Colab session where an earlier `import torch` died partway — a `pip install` that swapped torch under the running kernel, a `torch.py` shadowing it in `/content`.
  - Symptom: `AttributeError: partially initialized module 'torch' has no attribute 'fx'`, raised from `torch/_export/utils.py` by a learner's plain `import torch as t`.
  - Root cause: two halves. Python drops only the module that RAISED, so a `torch._export` imported seconds before survives, and the next `import torch` re-runs `torch/__init__.py` right back into that stale submodule. And once a corpse sits in `sys.modules`, `import torch` does not re-execute anything — it succeeds, so `try/except` around the import sees nothing wrong and the error resurfaces from the learner's cell instead.
  - Prevention/fix: `colab_grader._dd_preflight_torch()`, called from the run-me-first checker cell. Judge the MODULE (`hasattr(torch, "fx")`), never the import statement; purge `torch` **and every `torch.*`** before retrying, or the retry reproduces the bug.
  - Status: `RESOLVED` (2026-08-13) in the generator. Republishing the notebooks is what puts it in front of a learner.

## Recent Changes
- 2026-08-13 (**a broken runtime now says so, instead of showing torch's
  internals**): `colab_grader.py`, `colab_cells.py`. A learner hit
  `AttributeError: partially initialized module 'torch' has no attribute 'fx'`
  on the bare `import torch as t` that opens every drill cell — five frames deep
  in `torch/_export/utils.py`, naming neither the cause nor the cure, and not
  even the real error: it is the echo of an import that already died. The
  checker cell is the one cell the notebook tells you to run first, so it is the
  only place this can be caught early enough to matter. `_dd_preflight_torch()`
  runs there, ahead of `_dd_install_fixtures()`.
  - **The trap is that catching the import is not enough.** `import torch` does
    not re-execute a module already in `sys.modules`, so a partially-initialised
    torch imports without raising and the failure just moves to the learner's
    cell. The preflight tests the object (`fx` and `__version__` both bound) and
    treats a corpse as a failure.
  - **Purge the submodules too.** Clearing `torch` alone leaves the stale
    `torch.*` entries that caused the re-entrancy, so the retry hits the same
    wall. One retry, then an instruction naming the two things that actually do
    it: delete the runtime, and look for a `torch.py` in `/content`.
  - Never raises — a checker that refused to load over this would take the whole
    lesson down with the runtime. Regenerating touches exactly one cell per
    notebook (`dd-checker`, hidden, `display-mode: "form"`); all 2589 other cells
    are byte-identical.
- 2026-08-06 (**the notebook can open one concept, and the compiler is two files**):
  `colab_cells.py` (new), `generate_colab_notebooks.py`, `watch.py`. The gate
  teaches a segmented KP one concept at a time and there was nowhere to send the
  learner: the segment headers carried minted ids that name nothing, so the only
  anchor was the KP's, and "Concept 2 of 3" opened all three. Each concept's
  header is now anchored `dd-seg-<kc>-<n>` and the index ships
  `"<kc>#<concept_id>" → anchor`, keyed by the exposure key the gate already
  hands the client — nothing re-derives it. **By position, not by concept id**:
  the id is authored and free to be long or edited, an anchor has 64 safe
  characters and has to survive regeneration. Multi-concept KPs only; the 32
  one-concept KPs are byte-identical and still route through `kps`. `validate`
  fails the build if the index would point at a cell the notebook does not have,
  because Colab ignores an unknown fragment in silence.
  - **The split.** `colab_cells.py` takes what a CELL is — id minting, the
    four-cell shape of a problem, the checker — and `generate_colab_notebooks.py`
    keeps the compiler: which cells a lesson turns into, in what order, and
    writing them out. 667 → 407 LOC, out of ORANGE.
  - **The extension's copy is three files**, `notebook-index.js` +
    `-questions.js` + `-concepts.js`, in load order: the first assigns
    `window.DD_NOTEBOOKS` and the others `Object.assign` onto it. It is
    generated data that only grows, and 646 lines of it were the largest thing
    in the extension. The web app's `colab_notebooks.json` is unchanged, one
    file, and gained the same `segments` map.

- 2026-08-06 (**every segment gets a stable id**): `compile_lessons.py`. `concept_id` used to come only from a KP's optional `concepts:` frontmatter, which four KPs have — so the other 27 segmented ones compiled to `""` and the per-concept lesson loop skipped them entirely. `_concept_id` falls back to position plus title slug (`s1-nesting-becomes-axes-…`). The position makes collisions impossible (two segments may legitimately share a title) and the slug keeps a stored exposure map readable. Re-titling or reordering a segment mints a new id and re-teaches that one concept once, which is the right failure: a bare index would silently credit a rewritten concept as already read.

- 2026-08-06 (**the fading hides the concept, not the argument**): `lesson_lib.py`, `lesson_quality.py`, `compile_lessons.py`, `validate_lessons.py`, `audit_ladder_pairing.py`, `../Local_Deployed_Shared/lessons/AUTHORING.md`. q67's faded starter was `return z.clamp(_____=0.0)` on the KP that teaches `clamp` — the one recall the drill existed to test was printed on the page, and the ladder promotes on it. The rule now: every symbol in the KP's `new_syntax` is blanked, everything an earlier lesson taught may stay, so the learner gets `z._____(_____=0.0)` — a method call on the tensor with one keyword argument and the value 0.0, which is all structure and no concept. `blank_new_syntax` applies it at compile time over the function body only (the grafted example-run block keeps its `t.tensor(…)` fixture); `check_fade_leak` (FADE_LEAK) verifies the OUTPUT, so anything it reports is something the rewriter could not reach. `syntax.*` entries are operators with no identifier to hide and are reported rather than rewritten — and the operator check reads code only, because q107's docstring states its problem as `a @ x = b` while its solution is `t.linalg.solve(a, b)`. 250 starters rewritten, 0 leaks left, `validate_lessons.py` PASS on all 63 pages.
- 2026-08-06 (**every faded starter now prints something**): `lesson_lib.py`, `compile_lessons.py`. All 250 hand-cut faded starters were the function and nothing else — `def solve(z): return t._____(z)` — while the question's own starter ends with a fixture and a `print(solve(fixture))`. Served the authored one, the learner ran the cell, saw no output, and had nothing to compare against the expected-output block directly above it: filling the blank right and filling it wrong looked identical. `attach_example_run` grafts the question's demo block (comments included, plus any import the faded starter is missing) onto the authored starter, and skips a starter that already calls the function at the top level. Done at COMPILE time because the same record reaches the learner by three routes that share no code — the backend's `faded` rung, the published Colab notebook, and the client-side single-KC ladder — so fixing one leaves two broken. The 7 starters whose blanks do not parse as Python were already unparseable and still are; the graft is line-based and does not care.
- 2026-08-03 (**the worked-example standard is written down and enforced**): `lesson_quality.py` turns a session of learner feedback into six named rules — INTRO (prose before the first block), INTERLEAVE (short blocks with prose between; no single block carrying the example), PRINTS (a block that asserts must also print, because `assert` is silent on success), CASES (a question whose own test cases disagree needs an example showing more than one), GIVEAWAY (the example must not run on an input the problem is graded on, and must not restate its expected output unless it is demonstrating variation), and PROMPT_LEAK (an unaided prompt must not name the call the learner has to choose).
  - **Enforcement is opt-in per page, on purpose.** `strict_for` returns True only for a KP with an `## Applied practice` section. 1 page is on the standard, 62 are not; `audit_ladder_pairing.py` prints those 218 findings (148 INTERLEAVE, 70 INTRO) as the work list without failing. A rule that turns every page red on day one gets switched off, and then it protects nothing — the same reasoning that already keeps `distance` soft.
  - **Every rule was mutation-tested against the content that motivated it**, not just against the fixed version: GIVEAWAY fires on the pre-fix q224/q482/q484 examples (all three were built from a graded input), INTERLEAVE on all three pre-fix segments, PROMPT_LEAK on the pre-override q480.
  - **PROMPT_LEAK initially passed the one question it exists to catch.** Exempting a symbol when it appeared anywhere in the starter meant q480's starter *docstring* ("True if dtype is float32") exempted `torch.float32`. The exemption now requires the starter to actually call `t.<sym>`; a docstring naming the answer is itself a leak. Evaluated over 227 independent questions, 0 currently leak.
  - `## Applied practice` is a new section: independent-rung drills the KP wrote an example for. Ids must already be in the frontmatter `independent` list and must carry a ```python worked fence — `validate_lessons.py` errors on either, because an applied item without an example is served as a solo drill while the strip promises support.
- 2026-08-03 (**stage 2 is a pair: a solved example, then the same move on different specifics**): "Worked example" was a KP-level prose block that happened to sit near a drill. It is now an explicit pair — a lesson's `faded_items` entry may carry an `example: {question_id, note_markdown}`, and `example_cells` renders that question **already solved** in the cells directly above the problem's header. A worked example is not a third kind of content; it is a problem plus its known answer, so the prompt and the canonical solution are read out of `questions_structured.json` rather than authored. The only thing an author writes is `note_markdown`: the sentence saying what carries across and what does not.
  - **The anchor is the mechanism.** Example cells are minted `dd-q<problem>-example`, naming the PROBLEM they scaffold and not the bank question they were built from, so `colab_focus.js`'s existing `problemOf` regex groups them with that problem. No extension change was needed and none should be added — `extension/content/watch.py::check_stage_two_pair_survives_focus` runs the shipped pattern against real anchors to keep it that way.
  - No `<!-- dd:… -->` marker in an example body, unlike a problem header: that marker is `colab.js`'s text fallback when Colab drops cell ids, and a substring search for `dd-q481` would hit `dd:dd-q481-example` first, routing the panel above the learner's own problem.
  - `check_examples` validates every pair at build time: the example question must exist in the bank, and it must not ALSO be served as a problem anywhere — an example question is *spent*, because handing out its full answer at stage 2 and asking for it again at stage 4 is the same question twice with the answer in between.
  - **The rail does not render the pair.** The example belongs above the problem in Colab, not in the sidebar; `practice/ladder.js::decorate` returns early on the Colab edition (except under `dd-no-notebook`, where there is no notebook to hold it).
  - Piloted on **one** pair — `np-1`, `numpy.ndarray-model`, segment *"dtype is a property of the whole block"*: **example q484, problem q486**. q484 was that segment's own faded item and its authored starter (`a._____ == b._____`) transcribed the worked example directly, so it was already spent and promoting it to the demonstration costs nothing. q486 is promoted out of `independent_items` — same concept, genuinely different move (a dtype you *force* rather than one you're handed), and harder (36 vs 24). Question count 424 → 423, which is q484 being spent. Every unconverted segment is untouched and renders exactly as before.
  - **`audit_ladder_pairing.py` picked the pair, after it rejected two others.** The first attempt (example q224 → problem q481) turned a DISTANCE finding into a COVERAGE failure, which *fails the build*: nothing demonstrates `Tensor.ndim` before q481 asks for it. The second (q224 → q480) failed the same way on `Tensor.dtype`. Both were real — a pair whose example never shows the move the problem needs is the audit's TOO FAR case, and a prose sentence in `note_markdown` is not a demonstration. Sweeping the whole course with the audit's own coverage and distance rules finds **223 segments whose existing question pool can supply a covered, distant twin**, so this is a searchable conversion and not 118 hand-authored examples. The full audit is unchanged at exit 0, 61 distance / 0 coverage / 0 blank — the pilot introduces and fixes nothing.
  - Republish with `publish_colab_notebooks.sh` — a generator change is invisible until the notebooks are pushed.
- 2026-08-02 (the setup cell stops advertising a backend): `generate_colab_notebooks.py`'s `setup_cell` emitted `DD_TOKEN = ""`, `DD_BACKEND_URL = "https://delta-drills-backend.fly.dev"` and `DD_LESSON_ID`, for a completion beacon that was never wired and cannot be on this route — Colab sandboxes a cell's rich output, so the panel learns a result by reading the line `dd_check` prints instead. Two dead variables would be harmless if they were invisible, but this is the first cell in the notebook, so every lesson opened on a URL and an instruction to paste a credential. Only `DD_LESSON_ID` remains, because `extension/content/colab.js`'s `identify` matches it as the id-independent "which notebook is this" route. The `BACKEND` constant went with them. Republish with `publish_colab_notebooks.sh` — a generator change is invisible until the notebooks are pushed.
- 2026-08-01 (`dd_check`'s output is a contract, not just prose): the summary
  line it prints — `✅ Problem 480 — 5/5 cases passed.` — is the ONLY channel
  from a notebook back to the app. Colab sandboxes a cell's rich output away
  from the page, so `extension/content/colab_focus.js` reads this text off the
  DOM and reports the grade. `watch.py` now runs `dd_check` and matches its real
  output against the pattern that file greps for; reword one without the other
  and the app silently stops recording anything done in Colab.
- 2026-07-31 (the answer is not on screen until you have answered): the solution
  cell is no longer a collapsed `display-mode: "form"` cell. Collapsing still
  printed "💡 Solution — Problem 480" under the code the learner was writing,
  and still cost a second click after the reveal. The cell is now plain, and the
  extension hides it outright (`content/colab_dd.css`) until the panel's verdict
  click says otherwise. Without the extension nothing hides it — a plain reader
  of the published repo sees the answers, the way ARENA's own notebooks do.
- 2026-07-31 (a problem you can check yourself): every generated problem now
  renders four cells instead of two — the prompt **with its expected output**,
  the starter code, `dd_check(<id>)`, and the reference solution in a collapsed
  Colab form cell. Reported as "it doesn't show you the expected output that you
  should see… there should be a code cell that tells you whether you solved it
  correctly, and if no, which cases you failed". The checker is
  `colab_grader.py` plus a per-notebook payload of that notebook's test cases,
  deflated and base64'd into the same cell: expanded JSON would be the answer
  key to every problem below it, printed at the top. Expected outputs longer
  than 24 lines are truncated (one ARENA image drill's is 7 KB of pixels).
  `publish_colab_notebooks.sh` now also ships `numbers.npy`, which the checker
  downloads on first use — moving that file breaks the einops drills.
- 2026-07-31 (notebooks are the student-facing surface): `build_index` gained a
  `kps` map — `kc -> "dd-kp-<slug(kc)>"`, the same string written onto the KP
  header cell — so `practice/lessons.js` can send a learner
  to the concept section instead of rendering the lesson in the app. Ships in
  both index outputs. No notebook content changed: the compiled notebooks
  already carried concept prose, watch-outs, multi-cell worked examples, faded /
  guided / independent problems, `<details>` hints and common mistakes, which is
  why the panel could drop all of it in one pass.
- 2026-07-31: `previews:` added to the KP frontmatter contract. `audit_lesson_syntax.py` exempts a listed symbol from "shown before it is taught" and reports it under "declared previews" instead; `validate_lessons.py` gained `check_previews`, which runs on full-corpus runs only (it needs every page to know who declares what) and rejects an entry that the page does not show, that the page also declares, or that no later page declares. Without those three checks the key would be a mute button, and a stale entry left behind by an edit would silence a real regression.
- 2026-07-30: `build_qmatrix.py` was unrunnable and nobody noticed. It aborted on the first question that appeared BOTH in a KP's `faded`/`guided`/`independent` list and in `LEFTOVER_TARGETS` — and 76 of them did, because that is exactly how a leftover retires: a KP claims the question later. All 76 agreed on the KC, so the abort was protecting nothing. A KP reference now supersedes a leftover silently (it carries the role and the page's `new_syntax`, which a hand assignment cannot), and only a DISAGREEMENT — two sources naming different KCs — is fatal. The stale committed `qmatrix_tags.json` this hid was 274 entries behind on `new_syntax` and 79 on `source`; `target_kcs` and `supporting_kcs` were correct throughout, so no question had been gated to the wrong KC. Rebuild it whenever KP refs change, or `validate_lessons.py --coverage` fails on untagged questions.
- 2026-07-27: `grade_against_bank` mirrors the runtime numpy preamble, so torch-dialect lessons using the ARENA fixture validate. `watch.py` filled in.
- 2026-07-20: Compiled `worked_example_code` supports inline optional-run lesson UI.
- 2026-07-20: Segment-specific `## Watch out`; exact one-worked/one-faded validation.
- 2026-07-19: Single-concept segment support (`build_segments`), per-segment faded enforcement, expected-setup grading fix.
- 2026-07-06: Initial doc created.
