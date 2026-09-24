# js

## Purpose
- The live figures of the AISC write-up (Figs. 1–7) and their shared plumbing.
  Parent context: `../README.md`.

## Owns
- Figure behaviour and the illustrative update rules they run.

## Does NOT own
- Loading order (`../boot.js`), styles (`../aisc.css`), markup (`../../index.html`).
- The production engine: `engine.js` is a PORT of backend constants for
  illustration; the real rules live in `This-Directory-Only/backend/app/`.

## Key Files
- `engine.js`: `window.DeltaEngine` — placement/BKT/FIRe/remediation constants.
- `common.js`: `window.AISC` — `css()` token reader (reads `#aisc-root`),
  `rgb01`, `onTheme` (fires on `delta:theme-changed`), `whenVisible`, `graph`
  (fetch of `aisc/data/kc_graph.json`), `lessonColor`; reveal + section nav
  (the nav is `#aisc-toc` in the app topbar).
- `field.js`: the gutter node field — a fixed canvas on the page, animated only
  while `#aisc-root` intersects (a hidden page never does) and the tab is
  visible; idle when its CSS hides it (< 1000px); still frame under reduced
  motion.
- `fig-effort.js` Fig. 1 · `fig-forget.js` Fig. 4 (SVG, draggable review handles) ·
  `fig-graphs.js` Figs. 2, 3, 5, 6 (Cytoscape + dagre) · `fig-seth.js` Fig. 7.

## Data & External Dependencies
- `../data/*.json`; global `cytoscape` (app vendor/graph).

## How It Works (Flow)
1. `common.js` builds `AISC`; each `fig-*.js` wires its controls and starts on
   `AISC.whenVisible(<its stage>)`.

## Invariants & Constraints
- Look up elements only by `aisc-`-prefixed ids.
- Fetch paths are relative to the app root: `aisc/data/…`.
- Globals are `AISC` and `DeltaEngine` only.
- SVG figures redraw from scratch; anything that must survive a redraw
  (pointer capture, keyboard focus) hangs off the `<svg>` itself — see
  `fig-forget.js`.

## Extension Points
- New figure → new `fig-*.js` + entry in `../boot.js` SCRIPTS.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- See `../README.md`.

## Recent Changes
- 2026-09-24: Ported from delta-drills-aisc; ids prefixed, `DD` → `AISC`,
  theme via app event, nav scroll without hash.
- 2026-09-24: `fig-forget.js` rewritten from CindyScript to SVG; same model,
  same readout numbers; handles keyboard-operable (←/→, Home, End).
- 2026-09-24: `field.js` added (port of the site's bg-field.js, page-scoped);
  `common.js` section nav reads `#aisc-toc` and catches clicks on the document.
