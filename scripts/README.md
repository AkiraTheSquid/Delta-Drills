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
- `audit_graph_structure.py` + `graph_structure_baseline.json` — structural audit of both served graphs: registry/atom sanity, order-only dependencies (symbol taught earlier with NO prereq path — the prereq ratchet is order-blind to this), same-move duplicates per rung (normalized AST, strings kept), rung-difficulty inversions. Ratcheted by `watch.py::check_graph_structure_ratchet`.
- `lesson_lib.py`: shared parser — frontmatter, `split_sections` (ordered, repeatable), `build_segments` (one-concept segments), bank/registry loaders. HIGH fan-in: the other three scripts import it.
- `validate_lessons.py`: executes every code fence in document order, grades every faded solution against bank test_cases, enforces at-least-one worked + one-or-two faded exercises per segment, and applies `lesson_quality.py` as hard errors to pages on the new standard. Run with `--coverage` as the full gate.
- `lesson_quality.py`: the written standard for a worked example — INTRO, INTERLEAVE, PRINTS, CASES, GIVEAWAY, PROMPT_LEAK. Every rule is a defect that shipped, passed the structural validator, and was caught by a learner working the page. `strict_for(kp)` gates enforcement on the page having an `## Applied practice` section, so upgrading a page means opting into the standard rather than the standard failing 63 pages at once. Imported by both `validate_lessons.py` (errors) and `audit_ladder_pairing.py` (backlog report).
- `compile_lessons.py`: emits `lessons_structured.json` with per-KP `segments` plus legacy aggregate fields (viewer.html back-compat).
- `build_qmatrix.py`: derives `qmatrix_tags.json` question→KC tags from KP refs + hand-assigned leftovers.
- `compile_web_notebooks.py`: the same nine notebooks, compiled for the BROWSER. Imports `generate_colab_notebooks.build_notebook` — deliberately the same function, not a parallel implementation — and reshapes its cells into `Local_Deployed_Shared/lessons/notebooks/<id>.json` plus a `manifest.json` the in-app Notebooks tab fetches. Each cell keeps its id and gains a `role` derived from that id (`setup`, `checker`, `problem`, `code`, `check`, `hints`, `solution`, `prose`), which is what `notebook-view.js` dispatches on. Deterministic and cwd-independent: running it twice writes byte-identical files. The deploy runs it after the question-bank export, because the compiled notebooks EMBED the bank.
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
- `solution_symbols.py`: the exhaustive symbol collector. `audit_lesson_syntax.Collector` answers "what library API does this show?"; this answers "what must the learner be able to WRITE?", which is a bigger set — every call, method, attribute, operator, comprehension, import, branch and literal, named in the same vocabulary the KP pages already declare in `new_syntax:`. Resolves imports (`import math` -> `math.sqrt`, not `Tensor.sqrt`), subtracts names the chunk binds itself, reports a base that is neither bound nor imported as `undefined.<name>.<attr>`, and matches a method under any receiver spelling (`xs.append` satisfies a declared `list.append`). `unhandled_node_types()` is its own completeness check.
- `audit_solution_prereqs.py`: the guard built on it. For every drill, every symbol in the SOLUTION and in the PROBLEM (starter + prompt) must be declared by a KP whose KC sits at or before that drill's KC in the lattice. `--by-symbol`, `--kc-prefix numpy.,einops.`, `--qid`, `--surface`, `--new`, `--write-baseline`.
- `audit_arena_grounding.py`: the other guard. Every library function a KP DECLARES or a drill USES must appear in the ARENA corpus; a symbol in zero of the 458 notebooks is attention spent on something no learner will need. Reads the frozen index, so it needs neither torch nor the corpus — which is what lets the fast structural watchers run it. `--surface declarations,solution,starter`, `--by-symbol`, `--kc-prefix`, `--new`, `--write-baseline`, `--check-index`.
- `guard_checks.py`: the adapter both guards are called through, by `watch.py` here and by the five lesson-folder watchers. `run(kc_prefix)` returns the check functions to append to a folder's own list. One implementation on purpose.
- `arena_symbol_index.json`: the frozen ARENA measurement — DF per folded operation, plus the `torch.Tensor` member list the fold needs. Written by `audit_arena_frequency.py --write-index`; do not hand-edit, every number is a measurement.
- `audit_symbol_coverage.py`: the third guard. Every symbol a KP DECLARES in `new_syntax` must appear in the SOLUTION of at least two drills tagged to that KP's own KC. Solution only, because a starter that hands the learner the call is the opposite of evidence (measured: starter and prompt add coverage for zero symbols); own-concept only, because a drill tagged elsewhere never updates this concept's mastery estimate. `--kc-prefix`, `--summary`, `--new`, `--write-baseline`.
- `symbol_coverage_baseline.json`: recorded coverage debt. Unlike the other two baselines this records the drill COUNT per symbol, not just the key, so losing the one drill that was holding a symbol at 1 fails even though the symbol was already known debt.
- `arena_grounding_baseline.json`: recorded ARENA-grounding debt, same ratchet contract as the file below.
- `audit_arena_frequency.py`: what the real curriculum uses. Scans the 458 ARENA notebooks in `arena-book-colab/ARENA_5.0` (EXCLUDING `ch-1-foundations`, which is ours — measuring it would be measuring ourselves) and ranks every torch/einops operation by DOCUMENT FREQUENCY: how many notebooks you cannot read without it. Reads code cells **and fenced python inside markdown cells** — ARENA's worked solutions live in markdown, so reading only code cells measures the empty stubs. `--coverage` joins that against what our KP pages declare; `--write-index` refreshes the frozen index the guards read. Needs the torch venv: it separates a real `torch.Tensor` member from a `Path`/`str`/`dict` method of the same name.
- `solution_prereq_baseline.json`: recorded debt. `watch.py` fails only on violations NOT in it.
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

- 🔴 **Every symbol a drill's SOLUTION *or PROBLEM* uses must have a lesson at or before that drill's own concept.** Enforced by `watch.py::check_solution_prereq_ratchet` against a recorded baseline, so it fails on NEW debt only. Widened from the solution alone to all three surfaces on 2026-08-29 — a starter is not just a faded solution, at the worked rung it IS the code. Re-recording the baseline is admitting debt, not paying it, and should be argued for in the diff.
- 🔴 **Every symbol a concept declares must be drilled at least twice ON that concept.** Enforced by `watch.py::check_declared_symbols_are_drilled`, same ratchet shape, with the count recorded so debt may stay put but may not grow. The reason is the mastery models, not tidiness: BKT and the logistic engine each estimate ONE number per concept and the lattice gates on it, so a declared symbol with no drills of its own is marked learned on evidence gathered about something else. First run: **51 of 144 declared symbols under the floor, 19 of them at zero**, concentrated on the never-split blob nodes — `numpy.stack-concat-interleave` has all ten under, `numpy.random-generator` seven of ten. That measurement is what `This-Directory-Only/SPEC_NODE_SPLITTING.md` acts on.
- 🔴 **Every library function we teach or drill must appear in the ARENA corpus.** Enforced by `watch.py::check_arena_grounding_ratchet`, same ratchet shape. ARENA is what the course prepares people for, so it is the empirical test of whether a function is worth a learner's attention. The two audits ask different questions and a symbol can fail either alone: `torch.einsum` is perfectly *taught in order* and appears in zero notebooks.
- 🔴 **The ARENA index must not go stale.** `check_arena_index_is_current` re-counts the notebooks on disk against `arena_symbol_index.json`. The grounding guard reads the frozen index rather than rescanning, which is what makes it fast enough to run on every edit; the price is an artifact that can silently stop describing reality, and a guard answering from a corpus that no longer exists is worse than no guard.
- 🔴 **Both guards run from the folder being EDITED, not only from here.** `guard_checks.py` is the single implementation; `Local_Deployed_Shared/lessons/{,numpy/,einops/,einsum/,python/}watch.py` each call it, the subfolders scoped to their own KC prefix. One copy on purpose — a guard duplicated six times becomes six different guards inside a month, each drifting toward whichever was easiest to make pass.
- 🔴 **The collector may not walk past a construct.** `check_solution_symbol_coverage` fails if any AST node type in the bank has no visitor in `solution_symbols.py` and is not listed in its `STRUCTURAL` set. A blind collector reports nothing and the ratchet goes green — the worst failure a guard has.

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
- 2026-09-01: `audit_graph_structure.py` added (64 baseline findings: 41 edge-missing, 23 difficulty; 0 same-move after the string-constant fix). Wired into watch.py as a ratchet, mutation-verified.

- 2026-08-31 (**third standing guard: a concept must drill what it claims to teach**): new `audit_symbol_coverage.py` + `symbol_coverage_baseline.json`, `guard_checks.py` (`check_symbol_coverage`, added to `run()` so all four lesson watchers pick it up), `watch.py` (`check_declared_symbols_are_drilled`). The gap it closes: the two existing guards both ask whether a symbol is *legitimate* — taught in order, used by ARENA — and neither asks whether the concept claiming it ever tests it. 51 of 144 declared symbols are under a floor of two on-node drills and 19 have none at all, so it ships as a ratchet like its siblings. The baseline stores the count rather than only the key, because a key-only ratchet stays silent when a symbol already listed at 1 drops to 0. Verified by mutation, not by reading: adding an undrilled symbol to a page's `new_syntax` fires it, and untagging the single drill behind `torch.argsort#dim` fires it as a LOST regression; a wrong-family `--kc-prefix` stays quiet.
- 2026-08-30: The ARENA cut ran through these tools rather than around them. `build_qmatrix.py`'s `LEFTOVER_TARGETS` lost 38 of its 77 hand-assignments (the concept they named is retired) and gained two: q103 → `numpy.axis-reductions` and q198 → `numpy.broadcasting-rules`, kept back from the retirement because each was the only bank question training a gating BKT atom. Both guard baselines were re-recorded and **both shrank, with zero additions**: ARENA grounding 329 → 126 violations (203 removed — every ungrounded symbol that went with its concept), prerequisites 1464 → 998 (473 removed, 7 added, all seven belonging to the two questions kept back and all of the already-recorded `taught by NO lesson` class). The prerequisite guard earned its keep here: after the KCs were dropped it named exactly the six surviving drills whose solutions reached for a symbol only a retired page taught, which is the failure mode a cut of this size produces and nothing else would have found.
- 2026-08-29 (**both guards made standing, and a measurement bug that moved every ARENA number**): `guard_checks.py`, `audit_arena_grounding.py`, `arena_symbol_index.json`, `arena_grounding_baseline.json`, `audit_arena_frequency.py`, `audit_solution_prereqs.py`, `watch.py`, and the five lesson-folder `watch.py` files. Seth's ask: the prerequisite rule and the ARENA rule must both fire on the folder being edited, so nobody can add a drill and move on without them.
  - 🔴 **`cell_source` read only `cell_type == "code"`, and ARENA keeps its worked solutions in MARKDOWN.** 437 of the 458 notebooks carry fenced code inside markdown cells — 1040 blocks labelled `python` — under `<details><summary>Solution</summary>`; the code cell beside them is the empty stub the learner fills in. The audit was measuring the stubs and calling it the curriculum. Fixed: distinct library symbols 188 → **251**, symbol-notebook pairs 4258 → **5696**. Every ARENA figure published on 2026-08-28 is superseded.
  - What the fix changed in the conclusions: `einops.einsum` **61 notebooks** (was 5), `torch.einsum` still **0** — the headline finding survives and strengthens. `einops.repeat` 2 → **15** and `einops.reduce` 5 → **14**, which **kills the earlier "collapse repeat and reduce into one node" recommendation**; both are solidly used. `einops.rearrange` 33 → 42. Coverage: we teach 132 operations, ARENA uses 229, overlap 87 = 40.0% of usage mass (62.4% once the device/dtype preamble is set aside).
  - Unlabelled fences (1704 of them) are a mix of code and pasted output; a block counts only when it parses AND holds a statement that is not a bare expression. `tensor([1., 2.])` output parses fine and would otherwise inject a call nobody wrote.
  - The prerequisite ratchet now covers solution **+ starter + prompt**: 646 → **1464** known violations, +818 from the starter surface. First grounding run: **323** ungrounded — 53 declarations, 239 solution uses, 31 starter — over 88 distinct symbols, led by `torch.einsum` (70), `torch.movedim` (30), `torch.eye`, `Tensor.unfold`, `Tensor.data_ptr`. Nothing was fixed; the debt is recorded, as asked.
  - Both ratchets were proven in the failing direction (drop one baseline entry → the watcher fails; restore → green) and end to end: a synthetic drill using `torch.mvlgamma` on the first concept made `lessons/`, `lessons/numpy/` and `scripts/` all refuse at once.
  - codex: **6 correctness findings, all verified, all fixed.** (1) 🔴 An unlabelled-fence heuristic was also applied to LABELLED fences, so a `yaml` block counted as curriculum — `model: gpt-4` is a valid Python `AnnAssign`, and the corpus has 462 yaml fences. The label is trusted where there is one; the heuristic runs only where there is not. (2) 🔴 Folded operation DF took the `max` of the two spellings' counts instead of the UNION of their notebooks — right only if one spelling's notebooks nest inside the other's, which nothing guarantees. `scan` now keeps notebook identities; `--coverage` folds by union too, and its denominator moved with it. (3) The grounding guard omitted the `prompt` surface the prerequisite guard had just gained (+6 violations, 323 → 329). (4) 🔴 The guards were `assert`s and the lesson watchers have no `__debug__` refusal, so `python -O` stripped them and reported success — they raise `AssertionError` explicitly now, verified by running a watcher under `-O` and watching it still fail. (5) `--write-baseline` accepted `--qid`/`--surface`/`--kc-prefix` and would overwrite the canonical file with a subset, making the whole existing backlog read as NEW. (6) Index freshness compared notebook COUNT only, so a rewritten or renamed corpus passed; `corpus_fingerprint` (path + byte length per file, no bytes read) now moves on any of those — unit-tested for edit, rename, and that the excluded directory does not participate.
  - Declined two majors: a shared lesson-watcher bootstrap, and hashing notebook CONTENTS rather than sizes. The size fingerprint catches every failure that happens; a content edit preserving every file's exact byte count is not one.
- 2026-08-28 (**ARENA frequency audit**) — ⚠️ **every number in this entry is superseded by the 2026-08-29 entry above**; the scan was reading only code cells and missed the markdown solutions. Kept as written because the two measurement bugs it records are still worth knowing. `audit_arena_frequency.py`. The curriculum's usage distribution is savagely head-heavy — top 25 operations are 74% of all usage, top 50 are 88%, top 100 are 97.6%. We teach 132 operations and **56 of them (42%) never appear in ARENA in either spelling**. 🔴 ARENA writes `einops.einsum` (97 occurrences); `torch.einsum` appears ONCE in the whole corpus, and our einsum course teaches `torch.einsum` across 69 drills. 🔴 ARENA writes the METHOD (`x.argmax()`) where our pages declare the FUNCTION (`torch.argmax`) — half the apparent dead weight was that artifact until `operation()` folded the spellings. 🔴 **Do not rank a node by its `new_syntax`**: `numpy.boolean-masking` declares `torch.count_nonzero` (0 notebooks) while its real content, `& | ~`, is 129; `dot-matmul-patterns` declares `torch.dot` (0) while `@` is 58. Scoring by declaration alone would have deleted broadcasting and matmul.
  - Two measurement bugs worth remembering for anything that parses notebooks. Prepending an INDENTED import line makes the preamble itself unparseable and silently zeroes every cell it is added to — it reported einops as entirely absent from ARENA. And DELETING `%pip` magic lines leaves `except ImportError:` with an empty body, which killed 9.5% of all cells (the setup cell of nearly every notebook): replace a magic with `pass`, never delete it.
- 2026-08-28 (**solution -> prerequisite guard**): `solution_symbols.py`, `audit_solution_prereqs.py`, `solution_prereq_baseline.json`, `watch.py` (+2 checks). Motivated by `a.T`: eight faded drills on the course's FIRST concept required transposition, the page teaching it sat four lessons later, and nothing failed. First run: **646 violations across 282 of 511 tagged questions** — 535 plain-Python constructs no lesson owns (`syntax.arith` 162, `syntax.star-args` 87, `syntax.equality` 40), 93 library symbols, and 24 einops solutions calling `np.load` with `np` never imported (dead on the first line, left over from the numpy->torch conversion). 30 drills write a `def` on concepts that rank before `python.defining-functions`. Nothing was fixed — the debt is recorded, not paid, deliberately: the ask was for the guard.
  - `ASSUMED` is NOT honoured here. That list exempts `len`, `for`, comparison and comprehensions as "the audience already writes plain Python", which stopped being true when `py-0` started teaching them. Symbols it would have hidden are reported with an `ASSUMED` tag so the two audits' numbers can be compared (`audit_question_syntax.py` stays at 343 across three surfaces; it is unchanged and still runs).
- 2026-08-28 (**three problem types: Faded / Solo / Integrated**): `lesson_lib.py`, `compile_lessons.py`, `build_qmatrix.py`, `validate_lessons.py`, `audit_ladder_pairing.py`, `../Local_Deployed_Shared/lessons/AUTHORING.md`.
  - 🔴 **The "one or two faded exercises" ceiling is LIFTED, and it was the content-side cause of the repeats.** With three faded drills on a KC the queue ran out inside one sitting and re-served what the learner had memorised. The floor stays: at least one per segment.
  - `## Guided practice` and `## Applied practice` are retired (Guided folded into Faded). New `## Solo practice` — items may carry an optional ```python worked``` fence — and `## Integrated practice`, which never carries one. `Guided`/`Applied` remain parseable aliases for the other 62 KPs.
  - New `integrated:` frontmatter list → `kp-integrated` source tag from `build_qmatrix.py`; `compile_lessons.py` emits `solo_items` (with `worked_example_code`) and `integrated_items`.
  - 🔴 **`audit_ladder_pairing.py` COVERAGE now scores against the whole lesson page and accumulates across segments.** It used to score only `## Worked example`, but the learner reads `## Concept` and the example together, so eight legitimate drills were reported as using symbols "never shown". Nine coverage findings remain bank-wide, eight of them pre-existing in other KPs — the audit exits 1 on those, which predates this work.
- 2026-08-19 (the web edition of the notebooks): Added `compile_web_notebooks.py`. The in-app Notebooks tab needed the lessons as something a browser can fetch without a Python toolchain, and the tempting shape — a second compiler that walks the same lessons — is exactly how the two editions would come to teach different things. So it calls `build_notebook` and only reshapes the result. Parity is then asserted rather than assumed: `Local_Deployed_Shared/lessons/notebooks/watch.py` compares all 2589 cells against the published `.ipynb` files and fails on the first difference in count, id, type or source. Wired into `deploy_delta_drills.sh` immediately after `export_questions_json.py` — the notebooks carry the compiled `dd_check` grader, so a deploy that skipped this would ship a tab teaching the previous question bank with no visible symptom.
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
