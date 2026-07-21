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

- Uses NumPy arrays plus `einops` APIs in executable worked/faded code fences.
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

- 2026-07-20: Documented inline teaching + optional runnable worked-code flow;
  faded examples no longer appear in lesson screen.
- 2026-07-20: `kp-dl-flatten-heads` and `kp-repeat-model` reduced to one faded
  exercise each; extra qids retained as independent practice.
- 2026-07-19: Initial folder documentation scaffold created.
