# vendor

## Purpose
- Third-party code the AISC write-up's figures need that the app does not
  already ship. Parent: `../README.md`.

## Owns
- `Cindy.js`: the CindyJS runtime behind Fig. 4 (forgetting / FSRS curves).

## Does NOT own
- Cytoscape and the dagre layout: the app's own `../../vendor/graph/` tags.
- Figure code: `../js/fig-forget.js` builds the CindyJS instance.

## Key Files
- `Cindy.js` (~550 KB): copied verbatim from `~/Applications/delta-drills-aisc/vendor/`.

## Data & External Dependencies
- None at runtime; defines the global `CindyJS`.

## How It Works (Flow)
1. `../boot.js` sets `window.cindyDontWait = true`, then loads `Cindy.js`
   first in its ordered list when `#aisc-root` first becomes visible.
2. `../js/fig-forget.js` calls `CindyJS({...})` once the figure is on screen.

## Invariants & Constraints
- Never edit `Cindy.js`; replace it wholesale from upstream.
- Never load it from a `<script>` tag in `index.html`: that would put
  ~550 KB on every visit, and the lazy load is why it is here.
- `cindyDontWait` must be set BEFORE this file runs.

## Extension Points
- New vendored library → add the file here and an entry ahead of its
  consumer in `../boot.js` SCRIPTS.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Fig. 4 blank when lazy-loaded** — `RESOLVED`
  - When it happens: Cindy.js loads after DOMContentLoaded has fired.
  - Symptom: empty canvas; controls throw "reading 'err'/'save'" in `evokeCS`.
  - Root cause: CindyJS defers every instance until DOMContentLoaded.
  - Prevention/fix: `window.cindyDontWait = true` in `../boot.js` before load.
  - Status: `RESOLVED` 2026-09-24.
- **Cindy.css not vendored** — `ACTIVE` (harmless)
  - The source repo's copy was a saved 404 page; Fig. 4 needs no CindyJS CSS.

## Recent Changes
- 2026-09-24: Initial doc; Cindy.js ported with the AISC write-up.
