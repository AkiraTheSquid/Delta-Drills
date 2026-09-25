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
  (fetch of `aisc/data/kc_graph.json`); reveal on scroll. The topbar nav
  (`#aisc-toc`) and `#aisc-` link clicks are `../articles.js`'s since
  2026-09-25.
- `field.js`: the page's node field — a fixed canvas behind the write-up's
  cards, unmasked since 2026-09-25, animated only while `#aisc-root`
  intersects and the tab is visible; still frame under reduced motion.
  Motion is the original site's `bg-field.js` (nodes drift and link; the
  cursor links to and gently draws in the nodes near it). The cards are
  see-through wireframes, so its drawing is erased inside every card box
  (`CARDS`); nodes still drift behind them freely. Card boxes are measured
  on layout changes only (resize, ResizeObserver on `#aisc-root`, reveal
  `transitionend`, once a second), shifted by scroll. The card under the
  cursor gets `.is-hot` (lit in `--field-edge`).
- `fig-effort.js` Fig. 1 · `fig-forget.js` Fig. 4 (SVG, draggable review handles;
  implicit reps of `IMPLICIT_W` always on, tagged on the plot, keyed in the side) ·
  `fig-graphs.js` Figs. 3, 5, 6 (Cytoscape + dagre, drawn in why-graph.js's
  style) · `fig-seth.js` Fig. 7. Fig. 2 is `../../concept-graph/why-graph.js`.
- `fig-expand.js`: the "Full screen" button under every figure's graphic;
  toggles `figure.fig.is-max`, announces a window `resize` so the SVG figures
  redraw (Fig. 4 caps its height to the stage while full screen) and
  fig-graphs.js's ResizeObserver refits Cytoscape. Fig. 2's button clicks
  why-graph's `.wta-graph-btn` instead.

## Data & External Dependencies
- `../data/*.json`; global `cytoscape` (app vendor/graph).

## How It Works (Flow)
1. `common.js` builds `AISC`; each `fig-*.js` wires its controls and starts on
   `AISC.whenVisible(<its stage>)`.

## Invariants & Constraints
- Look up elements only by `aisc-`-prefixed ids.
- Fetch paths are relative to the app root: `aisc/data/…`.
- Globals are `AISC` and `DeltaEngine` only (`AISCArticles` is
  `../articles.js`'s, loaded outside this folder).
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
- 2026-09-24: Fig. 2's Cytoscape code removed (the app's why-graph.js draws it
  now) with `AISC.lessonColor`; Figs. 3/5/6 take why-graph's round-rectangle
  labelled nodes, red arrows, yellow highlight. Controls trimmed: `fig-effort.js`
  drops learner/expert sliders, Play, mechanism list, Fermi inputs;
  `fig-forget.js` drops speed/target/implicit controls (defaults kept in `S`).
  Readouts are one or two lines.
- 2026-09-25: `fig-expand.js` added (full screen per figure, Escape, Tab wrap);
  `fig-graphs.js` refits on stage resize; `fig-forget.js` plot panel back to
  `--paper-2` (it sits in a card again).
- 2026-09-25: `field.js` drawn unmasked across the page (card layout back);
  its gutter band reads `--aisc-clear` = 1180px.
- 2026-09-25: `field.js` keeps out of the (now see-through) cards: pushes
  nodes out, erases its drawing inside them. `fig-forget.js` plot panel
  unfilled, `fig-effort.js` marker ring in page paper.
- 2026-09-25: `field.js` v4: cursor drags nodes by link strength; hovered
  card `.is-hot` + pulls nodes in; title cards always pull, twice when hovered.
- 2026-09-25: Readouts back to full detail (`fig-effort.js` mechanism list,
  `fig-graphs.js` probe id / practised id / 7-line remediation log);
  `fig-forget.js` implicit reps always on with +w tags and a readout line.
- 2026-09-25: `common.js` drops its section-nav click handler and in-view
  highlight; `../articles.js` owns both now that the page is articles.
- 2026-09-25: `field.js` v5 back to a plain background (Seth: "doesn't have
  the thing where it pulls nodes in"): card push-out, hover/title pulls and
  cursor drag removed; erase-inside-cards and `.is-hot` kept.
