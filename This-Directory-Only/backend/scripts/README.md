# backend/scripts

## Purpose
One-shot maintenance scripts for the FastAPI backend — DB initialization and offline data migrations. Not invoked by the running app; run by hand or in deploy hooks.

## Owns
- `init_db.py`: bootstrap a fresh Postgres schema (creates pgcrypto extension, runs `Base.metadata.create_all`). Run once after a clean DB.
- `recompute_p_ewma.py`: recompute the per-subtopic EWMA correctness rate (`p`) from each user's attempt history in `../user_data/*.json`. Use after changing the EWMA alpha or fixing data corruption.

## Does NOT own
- The schema definitions themselves — those live in `../app/models.py` and `../schema.sql`.
- The running EWMA computation — handled live by `../app/adaptive.py` and `../app/practice/grading.py`.
- Question-bank / AI-pipeline scripts — those live in `../../scripts/` (one level up).

## Key Files
- `init_db.py`: `main()` runs the schema creation. Imports `app.db.Base, engine` and `app.models` so SQLAlchemy registers all tables before `create_all`.
- `recompute_p_ewma.py`: standalone script (no `main()`); top-level execution iterates `user_data/*.json`, recomputes `p` per subtopic with `P_ALPHA = 0.3`, writes back in place.

## Data & External Dependencies
- Postgres via `app.db.engine` (init_db.py only).
- `../user_data/*.json` per-user attempt files (recompute_p_ewma.py only).
- `app.models` — must be import-able for `init_db.py` to register tables.

## How It Works (Flow)
1. **Fresh DB**: `python -m scripts.init_db` (run from `backend/`). Creates pgcrypto, then all SQLAlchemy tables.
2. **EWMA migration**: `python scripts/recompute_p_ewma.py`. Reads each user's JSON, replays attempt history through the EWMA formula, writes corrected `p` values.

## Invariants & Constraints
- Scripts must be **idempotent** — running twice produces the same state. `init_db.py` uses `IF NOT EXISTS` for the extension and `create_all` (which skips existing tables); `recompute_p_ewma.py` only rewrites computed fields.
- Never delete or mutate user attempt history — only derived fields like `p`.
- The relative path `Path(__file__).resolve().parents[1] / "user_data"` in `recompute_p_ewma.py` assumes this folder is exactly one level under `backend/`. Don't move the script without updating the path.

## Extension Points
- New offline migration: add a new top-level script with a clear docstring describing what it touches and whether it's safe to re-run. Prefer idempotent-by-construction.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Recompute script reads user files in place** — `ACTIVE`
  - When it happens: `recompute_p_ewma.py` rewrites JSON files directly with no backup.
  - Symptom: a buggy formula change permanently rewrites historical `p`.
  - Root cause: no atomic write / backup step.
  - Prevention/fix: snapshot `user_data/` before running, OR add a `--dry-run` flag that prints the diff. Until then, use `cp -r user_data user_data.bak.$(date +%s)` first.
  - Status: ACTIVE.

## Recent Changes
- 2026-04-28: Doc filled in.
- 2026-04-27: Initial doc created.
