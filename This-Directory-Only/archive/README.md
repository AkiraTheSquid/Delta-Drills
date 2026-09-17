# archive

## Purpose
Content the course has stopped teaching, kept whole. One dated subfolder per
retirement, so a concept can be brought back by moving a file rather than by
rewriting a page from memory.

## Owns
- The snapshot of every retired knowledge-point page, at the moment it was cut.
- The dated record of WHY each retirement happened (each subfolder's README).

## Does NOT own
- The live curriculum — `Local_Deployed_Shared/lessons/`.
- Retired DRILLS. Question ids are positional in the CSV sources, so drills are
  retired by id in `Local_Deployed_Shared/pipeline/retired_question_ids.json`;
  their rows stay in `This-Directory-Only/csv files of problems/`.

## Key Files
- `retired-content-2026-08-30/`: the ARENA cut — 35 concepts, 216 drills. Its
  own README carries the reasoning and the restore procedure.

## Data & External Dependencies
- `Local_Deployed_Shared/lessons/kc_registry.json` — the live graph. Nothing
  archived here may appear in it.

## How It Works (Flow)
1. A retirement creates a new dated folder here and `git mv`s the pages into it.
2. That folder's `watch.py` asserts its pages are absent from the live registry.
3. A restore is the reverse move plus a registry entry; nothing is deleted, so
   nothing has to be re-authored.

## Invariants & Constraints
- 🔴 Never edit a page inside an archive folder. It is a snapshot; restore it
  first, then edit it in `Local_Deployed_Shared/lessons/`.
- 🔴 Add a NEW dated folder per retirement rather than growing an old one — the
  date is what makes the record readable a year later.
- This tree lives under `This-Directory-Only/` on purpose: it must never be
  rsynced into the Deployed worktree or shipped to Vercel.

## Extension Points
- Copy `retired-content-2026-08-30/watch.py` into the next dated folder; its
  checks are folder-relative and need no edits beyond the path.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Half a retirement** — `RESOLVED`
  - When it happens: the KC is dropped from the registry but the page is left
    live, or the page is moved and the KC left in.
  - Symptom: `validate_lessons.py --coverage` fails on an unknown KC, or a live
    concept has no page and the lesson viewer renders nothing.
  - Root cause: a concept lives in three places — registry, page, drills.
  - Prevention/fix: each dated folder's `watch.py` cross-checks the registry.

## Recent Changes
- 2026-08-30: Created, with the ARENA content cut as its first entry.
