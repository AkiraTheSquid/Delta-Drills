# data

## Purpose
- Static data snapshots the AISC write-up's figures read. Parent: `../README.md`.

## Owns
- `kc_graph.json`: KC graph snapshot (82 concepts, 211 edges, lessons,
  encompassing weights, question count) for Figs. 2, 3, 5, 6.
- `seth_progress.json`: Seth's September daily aggregates (n = 1) for Fig. 7.

## Does NOT own
- The live graph (`../../lessons/kc_registry.json`) or live learner state (Fly).
- Regeneration: both are built in `~/Applications/delta-drills-aisc`
  (`scripts/build_seth_progress.py`); copy them over after rebuilding there.

## Key Files
- See Owns.

## Data & External Dependencies
- Fetched at runtime by `../js/common.js` and `../js/fig-seth.js`.

## How It Works (Flow)
1. Figures `fetch("aisc/data/<file>")` when they first become visible.

## Invariants & Constraints
- Must stay valid JSON and must NOT be listed in `.vercelignore` (a runtime
  fetch of an ignored file gets the SPA's HTML back, silently).
- `seth_progress.json` holds daily aggregates only — no answers or item ids.

## Extension Points
- New dataset → add here and fetch it with an `aisc/data/` path.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- Snapshot drifts from the live graph as concepts are added — `ACTIVE`, by design:
  the numbers on the page are the proposal's as of 2026-09-23.

## Recent Changes
- 2026-09-24: Copied from delta-drills-aisc.
