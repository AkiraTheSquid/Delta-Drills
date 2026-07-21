# einsum

## Purpose
- First-encounter Einsum knowledge-point lessons (registry lessons es-1..es-2): teaching `np.einsum` notation one rule at a time.

## Owns
- One `kp-<slug>.md` per Einsum KC — concept, worked example, faded/independent exercises, misconceptions.

## Does NOT own
- Format rules/registry (`../AUTHORING.md`, `../kc_registry.json`), tooling (`scripts/`), numpy/einops content (siblings).

## Key Files
- `kp-diag-trace.md`: reference example of the atomic-page split Seth wants — 6 single-concept segments (`ii->i` → `ii->` → `bii->bi` → `bii->b` → `ik,ki->i` → `ij,ji->`), one worked example + one faded exercise each. Preview it with `index.html?lesson=einsum.diag-trace`.
- `kp-reductions.md`: 4 atomic segments (one deletion → higher-rank deletion → multi-axis deletion → explicit mean division), each with concept-specific Watch out, one worked example, and one faded exercise.
- `kp-notation-model.md`: the opener (what an einsum spec string means).
- Other `kp-*.md` (matvec-matmul, outer-products, dot-frobenius, broadcast-scaling, matrix-forms, attention-patterns): mostly still original multi-concept format — NOT yet split.

## Data & External Dependencies
- Exercise ids reference `../../questions_structured.json`; validated/compiled by `scripts/`.

## How It Works (Flow)
1. Edit a KP as single-concept segments (`../AUTHORING.md`).
2. `python3 scripts/validate_lessons.py --coverage` → `compile_lessons.py`.
3. Each segment renders inline: concept + worked explanation left, complete worked code right for optional running; faded exercise stays out of lesson UI.

## Invariants & Constraints
- One einsum concept (one spec-string idea) per segment; exactly ONE worked example + ONE faded exercise per segment.
- Watch out belongs to concept's lesson screen; no faded prompt or grading appears there.
- Introduce spec rules in dependency order — later specs build on earlier ones (e.g. `bii->b` after `ii->`).
- Read letter POSITIONS not letter SETS — misconceptions must reinforce this.

## Extension Points
- Splitting a legacy einsum KP: promote its guided/independent bank ids to per-segment faded exercises; dump each qid's test_cases first (contracts vary); update frontmatter to match.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Multi-concept einsum pages** — `RESOLVED for kp-diag-trace + kp-reductions, ACTIVE elsewhere`
  - When it happens: a KP taught diagonal + trace + batched + product-diagonal on one page.
  - Symptom: too many new spec strings at once (Seth's 2026-07-20 feedback).
  - Root cause: original Pass-1 KP granularity.
  - Prevention/fix: split into atomic segments, one spec idea each; `kp-diag-trace.md` is the pattern to copy.
  - Status: `ACTIVE` for remaining legacy einsum KPs.

## Recent Changes
- 2026-07-20: `kp-reductions.md` split into 4 atomic concept/worked sequences; worked code runs optionally on right.
- 2026-07-20: `kp-diag-trace.md` rewritten into 6 atomic single-concept segments.
- 2026-07-19: Initial doc created.
