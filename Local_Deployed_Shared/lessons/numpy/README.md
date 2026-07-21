# numpy

## Purpose
- The NumPy knowledge-point lessons (33 KPs across registry lessons np-1..np-4): first-encounter teaching for every NumPy KC in the drill course.

## Owns
- One `kp-<slug>.md` per NumPy KC — concept prose, worked examples, faded/guided/independent exercise assignments, misconceptions.

## Does NOT own
- Format rules and registry (`../AUTHORING.md`, `../kc_registry.json`), tooling (`scripts/` at repo root), einsum/einops content (sibling folders).

## Key Files
- `kp-ndarray-model.md`, `kp-constructors.md`, `kp-ranges.md`, `kp-dtype-astype.md`: np-1 openers — the pilot of the strict segment rhythm (one concept → one worked example → learner solves one) awaiting Seth's learner review.
- `kp-diag-triangles.md`: reference example of a 4-segment split (extract diag / trace / build diag / triangles).
- Remaining `kp-*.md`: mix of restructured (segments) and legacy single-segment files pending conversion.

## Data & External Dependencies
- Exercise ids reference `../../questions_structured.json`; segment structure is validated/compiled by `scripts/validate_lessons.py` / `compile_lessons.py`.

## How It Works (Flow)
1. Edit a KP following `../AUTHORING.md` (repeatable `## Concept:` segments).
2. Validate (`--coverage`) then compile; the app's lesson gate pages through segments.

## Invariants & Constraints
- One concept per segment; exactly ONE worked example per segment; every segment has a faded exercise whose solution passes bank tests.
- Concept prose teaches the GENERAL procedure before the example (never example-only).
- Fences execute top-to-bottom in a shared namespace per file — later segments may rely on earlier definitions, so don't reorder segments without re-validating.

## Extension Points
- Splitting a legacy KP: reuse its guided/independent bank ids as per-segment faded exercises (dump contracts with the scratchpad `dumpq.py`/`avail.py` helpers or read the bank directly); update frontmatter lists to match.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Faded starter written against assumed contract** — `ACTIVE`
  - When it happens: promoting a guided/independent qid to faded without reading its test_cases.
  - Symptom: validator FAILs the solution (e.g. q64's tuple order is (allclose, array_equal), not the "obvious" order).
  - Root cause: question text/tests differ from what the prose suggests.
  - Prevention/fix: always dump the bank question's test_cases first; let the validator be the gate.

## Recent Changes
- 2026-07-19: ~20 KPs restructured into single-concept segments; np-1 openers trimmed to one-worked-example rhythm (pilot for Seth's review).
- 2026-07-15: Initial 33 KPs authored (Pass 1).
