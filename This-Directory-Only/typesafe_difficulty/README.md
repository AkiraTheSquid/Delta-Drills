# typesafe_difficulty (data)

## Purpose
- Output directory for `../scripts/typesafe_difficulty/`: the TypeSafe (Jev) pairwise
  judgments and the per-concept difficulty ratings derived from them. Data only; the
  code and the full doc live beside the scripts.

## Owns
- `cache.jsonl`: every raw Noul answer, append-only, keyed by (model, champion, challenger, framing). The billing record — a rerun reads it instead of re-asking.
- `ratings.csv`: per concept, each drill's P(harder than the concept champion) and the champion id.
- `difficulty_scores.csv`: the rescaled 15..100 band scores that `scale.py` also writes into `../chatgpt/typesafe_difficulty_overrides.jsonl`.

## Does NOT own
- The scripts that write here — `This-Directory-Only/scripts/typesafe_difficulty/` (read its README).
- The bank the scores apply to — `This-Directory-Only/questions_full.json` via the override layer.

## Key Files
- `cache.jsonl`, `ratings.csv`, `difficulty_scores.csv` as above. All regenerable from `cache.jsonl` without the API.

## Data & External Dependencies
- Written only by `rate.py` / `scale.py`; read by `validate.py`.

## How It Works (Flow)
1. `rate.py` appends to `cache.jsonl` while it runs and rewrites `ratings.csv` at the end.
2. `scale.py` reads `ratings.csv`, writes `difficulty_scores.csv` + the override JSONL.

## Invariants & Constraints
- Never hand-edit `cache.jsonl`; delete it (or change `TYPESAFE_MODEL`) to re-ask.
- A `ratings.csv` produced by a dry run with a fake judge must be deleted, never scaled into the bank — the 2026-09-21 offline smoke test wrote one and it was removed.

## Extension Points
- None; add outputs by extending the scripts.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- **Fake-judge output mistaken for real ratings** — `RESOLVED`
  - When it happens: an offline dry run monkeypatches `Judge` and writes real-looking CSVs here.
  - Symptom: `scale.py` happily ships noise into the override layer.
  - Root cause: nothing in the CSV says which judge produced it.
  - Prevention/fix: dry runs delete their outputs before the turn ends; `cache.jsonl` absent + `ratings.csv` present is the tell.
  - Status: `RESOLVED` (procedural).

## Recent Changes
- 2026-09-21: Directory created.
