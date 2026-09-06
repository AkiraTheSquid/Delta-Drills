# einops

## Purpose

- First-encounter Einops lessons for registry courses `eo-1` through `eo-3`:
  rearrange, reduce, repeat, and model-shaped tensor transformations.
- Teach patterns as named shape contracts so learners reason about axis meaning
  and order instead of memorizing reshape/transpose tuples.

## Owns

- One `kp-<slug>.md` source per `einops.*` KC in `../kc_registry.json`.
- Einops-specific concept prose, worked code, faded/guided/independent drill
  assignments, and misconception corrections.
- Dependency order from pattern grammar through merge/split, reduce/repeat,
  pooling, patches, grids, attention heads, channel groups, and temporal windows.

## Does NOT own

- Shared lesson format or KC registry: `../AUTHORING.md`, `../kc_registry.json`.
- Parsing, validation, Q-matrix generation, or compilation: repo-root `scripts/`.
- Runtime lesson rendering/grading: `../../practice/`.
- NumPy/Einsum teaching content: sibling `../numpy/`, `../einsum/` folders.
- Drill contracts and fixtures: `../../questions_structured.json`.

## Key Files

- `kp-pattern-language.md`: entry point; names axes and permutes them with
  `einops.rearrange`.
- `kp-merge-axes.md`, `kp-split-axes.md`: parenthesis grammar and factor sizes.
- `kp-reduce-model.md`, `kp-repeat-model.md`: axis deletion vs axis creation or
  stretching.
- `kp-pooling.md`, `kp-patches-space-depth.md`, `kp-grids-montage.md`: composed
  spatial transformations.
- `kp-dl-flatten-heads.md`, `kp-channel-groups-temporal.md`: model conventions
  where merge order is semantically load-bearing.

## Data & External Dependencies

- Uses PyTorch tensors (`import torch as t`) plus `einops` APIs in executable
  worked/faded code fences. einops itself is dialect-agnostic, so only the
  fixtures moved. NumPy is still importable at runtime and fixtures may use it
  to reach `/delta_numbers.npy` — that is the one sanctioned `np.` in a fence.
- Exercise IDs resolve against `../../questions_structured.json`; faded
  solutions must satisfy those exact test cases.
- `scripts/lesson_lib.py` parses Markdown segments; `validate_lessons.py`
  executes fences and grades faded solutions; `compile_lessons.py` emits
  `../lessons_structured.json`.
- Browser runner loads Einops into Pyodide when lesson/example metadata marks
  content as Einops-backed.

## How It Works (Flow)

1. Author one concept segment with one worked example, optional `## Watch out`,
   and one bank-backed faded exercise following `../AUTHORING.md`.
2. Run `python3 scripts/validate_lessons.py --coverage`; fix fence, qid, shape,
   or test-contract failures before compiling.
3. Run `python3 scripts/compile_lessons.py`; never hand-edit compiled JSON.
4. LessonGate shows teaching + worked-example explanation on left and preloads
   worked code on right for optional editing/running. Faded content remains
   practice metadata; it is not shown inside lesson screen.
5. Completing lesson records KC exposure; normal question queue resumes.

## Invariants & Constraints
- 🔴 **Every function in a drill's solution or problem must be taught by a concept at or before it, and must appear in the ARENA corpus.** Both are enforced by this folder's `watch.py` via `scripts/guard_checks.py`, scoped to `einops.`, as ratchets against `scripts/solution_prereq_baseline.json` and `scripts/arena_grounding_baseline.json` — they fail on NEW debt only. Re-recording a baseline admits debt rather than paying it.

- One concept per segment; exactly one Python worked example and one faded
  exercise per segment. One faded qid may belong to only one segment.
- `rearrange` preserves every named axis; `reduce` may delete axes and declares
  aggregation; `repeat` introduces or stretches axes. Never blur contracts.
- Parenthesis order determines memory/value order. Equal output shapes do not
  prove semantic equivalence, especially classifier flattening and head packing.
- Pattern names are space-separated words, not einsum-style concatenated letters.
- Frontmatter faded/guided lists must exactly match Markdown qid sections;
  independent IDs must exist in bank.
- Code fences execute top-to-bottom in shared namespace per KP. Reordering
  segments can break later examples.
- Some bank fixtures reference `/delta_numbers.npy`; validator rewrites path for
  local execution. Do not change source contracts to hide fixture mismatch.
- **One dialect per page, and it must match the drills the page fades into.**
  `watch.py` enforces both by parsing the fences' imports and comparing against
  each referenced question's `answer_code`. Converting a page without its
  questions (or the reverse) fails the folder check.
- Never name a variable `t` in a fence: it shadows `import torch as t` and the
  next `t.arange` call dies with `'Tensor' object has no attribute 'arange'`.
- Every fence that calls `rearrange`/`reduce`/`repeat` must import einops
  itself; fences share a namespace when executed, so a missing import only
  shows up once someone reorders segments.

## Extension Points

- New Einops KC: add registry entry/dependencies, create `kp-<slug>.md`, assign
  bank qids, update Q-matrix tags, validate with coverage, then compile.
- Splitting legacy multi-concept KP: create repeated `## Concept:` segments;
  give each one worked example and one faded qid after reading that qid's actual
  `test_cases`; update frontmatter lists.
- New runtime package need belongs in practice runner/package-loading code, not
  lesson Markdown.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Legacy multi-concept pages** — `ACTIVE`
  - When it happens: original KP tours several patterns in one Concept/Worked block.
  - Symptom: learner sees multiple transformations before applying any one idea.
  - Root cause: early one-KP/one-page format lacked atomic segments.
  - Prevention/fix: split at each new pattern rule; keep one worked/faded pair per segment.
  - Status: existing legacy files still need content-by-content conversion.
- **Correct shape, wrong ordering** — `ACTIVE`
  - When it happens: parenthesized names are reordered, e.g. `(nh d)` vs `(d nh)`.
  - Symptom: shapes/tests may look plausible while values or checkpoint semantics are permuted.
  - Root cause: treating merge order as cosmetic.
  - Prevention/fix: track one element, assert round trips, document slow/fast name order.
  - Status: permanent modeling risk; examples must make convention explicit.
- **Faded solution built from prompt assumption** — `ACTIVE`
  - When it happens: qid promoted without reading bank setup/call/expected expression.
  - Symptom: lesson validator rejects seemingly reasonable solution.
  - Root cause: prose summary omits exact graded shape/order contract.
  - Prevention/fix: inspect test cases first; let validator remain release gate.
  - Status: permanent authoring risk.

## Recent Changes
- 2026-09-06: `kp-einsum.md` (KC `einops.einsum`, lesson `es-1`) — the ONE einsum
  node the retired einsum course was replaced by: four segments (one operand /
  two operands / repeated names / batch axis) built from the einops basics
  tutorial and Rocktäschel §2.1–2.10, 8 faded + 6 solo + 3 integrated drills, the
  five 0.0 `einsum_*` exercises plus 4 variants each (q847–888). The seven KPs the
  0.0 einops exercises map to gained an `integrated:` rung holding 4 variants per
  exercise (q889–940); `lessons/arena_exercise_kcs.json` "0-0" lists them for
  the Practice buttons. Stale "like einsum" comparisons on `kp-pattern-language`
  and `kp-reduce-model` reworded: einsum now comes AFTER einops in the course.
- 2026-08-30: einops is now the END of the course, not the middle. The einsum lessons that used to sit before eo-1 are retired (ARENA writes `einops.einsum` in 61 notebooks and `torch.einsum` in none), so three pages lost a `supporting:` entry that no longer resolves: kp-pattern-language dropped `einsum.notation-model`, kp-dl-flatten-heads dropped `einsum.attention-patterns`, kp-repeat-model dropped `numpy.tile-repeat-meshgrid`. Two drills were retired with the cut because their solutions used symbols only a retired page taught: q386 (`Tensor.flip`) off kp-pooling and q393 (`torch.repeat_interleave`, 0 notebooks) off kp-split-axes. 🔴 The open gap this leaves is worth naming: `einops.einsum` is the single highest-frequency einops operation in ARENA (61 notebooks, ahead of `rearrange` at 42) and **no concept teaches it**. The next content to author here is one `einops.einsum` node, not a return of the ten torch.einsum ones — the retired drills convert mechanically, since einops spells the same pattern with spaces and full axis names.
- 2026-08-29: `watch.py` gained the two standing content guards (prerequisite order and ARENA grounding), scoped to `einops.`. Editing anything here now runs them, so a drill reaching for an untaught or ungrounded function is refused at the point it is written rather than found later. 🔴 The ARENA figures published on 2026-08-28 are superseded: the corpus scan was reading only code cells and ARENA keeps its worked solutions in markdown fences. For this folder that moved a real conclusion: `einops.repeat` is in 15 notebooks and `einops.reduce` in 14, not 2 and 5 — the earlier suggestion to collapse them into one node is withdrawn.

- 2026-07-31: q387 retired as a duplicate of q322 — same pattern string
  (`'b c (h hs) (w ws) -> (hs ws b) c h w'`), same `solve` body, differing only
  in which slice of the fixture it loaded, so completing one taught nothing
  about the other. Replaced by **q531**, wired into `kp-grids-montage`'s
  independent list: the COLUMN-major montage
  (`'(g2 g1) h w c -> (g1 h) (g2 w) c'`), which the page names as a
  misconception ("to fill column-major, swap the split order, not the names")
  and nothing in the bank asked for. Retirement goes through
  `chatgpt/function_mode_deleted_ids.json`, and the id must also leave
  `LEFTOVER_TARGETS` in `build_qmatrix.py` or the next build fails on a tag
  pointing at a question the bank no longer has.
- 2026-07-27: Converted all 11 KP pages and the 91 einops drills to the PyTorch
  dialect ARENA uses. `watch.py` filled in: enforces one dialect per page, that
  the page matches the drills it fades into, and that every einops-calling fence
  imports einops.
- 2026-07-20: Documented inline teaching + optional runnable worked-code flow;
  faded examples no longer appear in lesson screen.
- 2026-07-20: `kp-dl-flatten-heads` and `kp-repeat-model` reduced to one faded
  exercise each; extra qids retained as independent practice.
- 2026-07-19: Initial folder documentation scaffold created.
