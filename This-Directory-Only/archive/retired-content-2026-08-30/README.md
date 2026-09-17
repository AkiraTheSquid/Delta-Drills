# retired-content-2026-08-30

## Purpose
The 35 knowledge-point pages the course stopped teaching on 2026-08-30, kept
whole so a concept can come back without being rewritten from memory. Nothing
here is served, compiled, or graded — this folder is the receipt for a deletion,
not a second copy of the curriculum.

## Owns
- The markdown source of every retired KP, exactly as it read when it was live.
- The evidence trail for WHY each went: see the `Recent Changes` entry below and
  `Local_Deployed_Shared/pipeline/retired_question_ids.json`.

## Does NOT own
- The live curriculum — `Local_Deployed_Shared/lessons/`.
- The retired DRILLS. Questions are not files: their ids are positional in the
  CSV sources, so they are retired by id in
  `Local_Deployed_Shared/pipeline/retired_question_ids.json` and their rows stay
  in `This-Directory-Only/csv files of problems/`.

## Key Files
- `kp-*.md`: one retired knowledge point each. The frontmatter still names a
  `kc:` that the registry no longer contains — that is expected, and is what
  `watch.py` here asserts.

## Data & External Dependencies
- `Local_Deployed_Shared/lessons/kc_registry.json` — the live graph, which must
  NOT contain any KC named here.
- `Local_Deployed_Shared/pipeline/retired_question_ids.json` — the drill half of
  the same retirement.

## How It Works (Flow)
1. A concept is retired: its KC leaves `kc_registry.json`, its page is moved
   here with `git mv`, and its drills' ids are added to the retired-ids file.
2. Everything derived is regenerated (`export_questions_json.py`,
   `build_qmatrix.py`, `compile_lessons.py`, the notebook compilers, the two
   concept-graph exports).
3. To bring a concept BACK: move its page to `Local_Deployed_Shared/lessons/`,
   re-add the KC and its prereqs to the registry, remove its ids from the
   retired-ids file, and rerun step 2. Nothing was deleted, so nothing has to be
   re-authored.

## Invariants & Constraints
- 🔴 A page in this folder must NOT have a live KC. Both halves of a retirement
  have to happen or the graph points at a page nobody can reach (and
  `validate_lessons.py --coverage` will not catch it: it only reads the live
  lessons directory).
- 🔴 Never edit a page here to "fix" it. It is a snapshot. Restore it first.
- This folder lives under `This-Directory-Only/` on purpose: it must never be
  rsynced into the Deployed worktree or shipped to Vercel.

## Extension Points
- The next retirement gets its own dated folder next to this one. Do not add to
  this one — the date is what makes the receipt readable.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **A retirement that only removes the KC** — `RESOLVED`
  - When it happens: pruning `kc_registry.json` without moving the page.
  - Symptom: `validate_lessons.py --coverage` fails with "kc not in registry",
    or worse, passes while a KP page is silently unreachable.
  - Root cause: a concept lives in three places — registry, page, drills.
  - Prevention/fix: `watch.py` in this folder asserts the pages here are absent
    from the registry, and `lessons/watch.py` holds the KC-count floor.

## Recent Changes
- 2026-08-30: Created with the ARENA content cut. 35 KCs retired — the whole
  einsum course (ARENA writes `einops.einsum` in 61 of its 458 notebooks and
  `torch.einsum` in none), all of `np-4`, and the `np-2`/`np-3` concepts whose
  own operations appear in under ~5% of the corpus. 216 drills went with them.
  What is left is the 37-concept closure of the einops nodes plus the
  high-frequency tensor literacy beneath them.
