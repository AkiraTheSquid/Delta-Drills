# Practice Runtime Contract

What the in-browser Pyodide runner injects before user code executes, and what each question can rely on.

The contract is declared in two places, which must stay in sync:
- `practice/runner.js` → `buildPyodidePreamble()` (the injection)
- Per-question `runtime_dependencies` / `runtime_unmet_dependencies` arrays in `arena_prereqs_structured.json` (the consumer)

## Always injected (Pyodide runtime)

| Name | Source | Notes |
|---|---|---|
| `np` | `import numpy as np` | numpy with `np.random.seed(0)` |
| `display_array_as_img` | runner stub | no-op; visual rendering is handled outside Python via `practice/visuals.js` |
| `sys`, `StringIO` | stdlib | stdout/stderr redirected to capture output |
| `math` | stdlib (importable) | not pre-imported, but available via `import math` |

## Conditionally injected

| Name | Condition | Source |
|---|---|---|
| `einops`, `einsum`, `rearrange`, `reduce`, `repeat` | `questionNeedsEinops(question)` is true | installed via micropip on first need, then `from einops import …` |
| `arr` | `questionNeedsArenaArray(question)` is true | `arr = np.load('/delta_numbers.npy')` — loaded into Pyodide's virtual FS |
| Per-question fixtures (`hwcs`, `list_of_tensors`, `img_a`, etc.) | `question.test_cases[0].setup_code` is non-empty | preamble appends `test_cases[0].setup_code` so user code sees the same fixtures the grader uses. **`expected_setup_code` is never injected** — that block constructs the canonical answer and would let `solve()` cheat. |

`questionNeedsEinops` matches `primary_library` ∈ `{einops, einops.einsum}`, `topic` ∈ `{Einops, Einsum}`, or `supports_visual_output: true`.

`questionNeedsArenaArray` matches `supports_visual_output: true` or any reference to `/delta_numbers.npy` in the question's text/code/test_cases.

## NOT injected — questions referencing these will fail in Pyodide

These appear in many ARENA-derived questions but are not currently provided by the in-browser runner:

| Name | Affected questions | Notes |
|---|---|---|
| `t` (torch alias) | 18 of 27 prereq questions | ARENA notebooks `import torch as t`. Pyodide does not have torch pre-installed; micropip cannot install it. Backend execution path (`/api/practice/run-code`) is the intended runtime for these. |
| `Tensor` | same 18 | ARENA: `from torch import Tensor` |
| `assert_all_equal`, `assert_all_close` | 16 of 27 | ARENA test helpers from `tests.py` |
| `tests.test_*` | 1 (id 27 einsum) | ARENA test runners |

A question's `runtime_unmet_dependencies` array lists which of these it relies on. If the array is non-empty, the question is expected to be routed to the backend, not Pyodide. The frontend dispatch is in `runner.js`:

```js
let useLocalPyodide =
  practiceMode !== "backend" || questionNeedsEinops(PracticeAPI?.currentQuestion);
```

Today, einops-flagged questions are forced to Pyodide even when they also need torch — so these will throw `NameError: name 't' is not defined`. Routing logic is the canonical place to fix this; see `runner.js:147`.

## Image rendering fallback

When Pyodide rendering of the canonical-solution image fails, `practice/visuals.js` falls back to displaying `numbers_stacked.png` (a static reference of the source `arr` data). Candidate paths are checked in order; first hit wins. See `getArenaNumbersPngCandidates()`.

## Adding a new question

1. Set `runtime_dependencies` honestly. Use the probes in `arena_prereqs_structured.json` (or re-run the annotator script) — don't hand-edit.
2. If the new question needs anything beyond the "Always injected" / "Conditionally injected" tables above, either:
   - extend `buildPyodidePreamble()` to inject it (and document it here), or
   - mark the question as backend-only and update `runner.js` dispatch to route it appropriately.
