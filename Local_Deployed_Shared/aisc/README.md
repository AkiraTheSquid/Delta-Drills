# aisc

## Purpose
- The app's "Why this app exists" / "How this app works" page
  (`#page-learn-about-app`) is the AISC-12 proposal write-up: the argument for
  Delta Drills and the mechanisms behind it, with seven live figures.
- Ported 2026-09-24 from the standalone site `~/Applications/delta-drills-aisc`
  (Seth: replace both halves of the page with the whole site).

## Owns
- Styles for the write-up (`aisc.css`), its lazy script loader (`boot.js`),
  the figure scripts (`js/`) and their data snapshots (`data/`).

## Does NOT own
- The markup: it lives inline in `../index.html` inside `#about-page-content`
  (so the owner-only About editor, `../about-page-editor.js`, can edit it).
- The copy's source of truth: `~/Applications/delta-drills-aisc`
  (`index.html`, `content/copy.md`). Re-port from there; don't fork the prose.
- Menu routing: `../account-menu.js` (`data-lab-open` = section id to scroll to).
- Cytoscape + dagre: the app's `../vendor/graph/`.

## Key Files
- `aisc.css`: the site's stylesheet, every rule nested under `#aisc-root`. Theme
  tokens = the standalone site's own palette, on `#page-learn-about-app` (so the
  page background and the field share it): its dark paper for the app's blue and
  dark themes, its light paper for `html[data-theme="light"]`. Also the page
  layout vars (`--aisc-col` / `--aisc-wide` / `--aisc-clear` / `--aisc-fade`),
  the field's mask, and `#aisc-toc`.
- `boot.js`: waits for `DDAboutContentReady`, then loads `js/*` in order
  the first time `#aisc-root` is on screen.
- `js/`, `data/`: see those folders.

## Data & External Dependencies
- `data/kc_graph.json` (82 KCs / 211 edges snapshot), `data/seth_progress.json`
  (Seth's daily aggregates, n = 1; built by the aisc repo's
  `scripts/build_seth_progress.py`).
- Google Fonts (Instrument Serif, Newsreader, IBM Plex Mono), loaded
  non-blocking from `../index.html`; fallbacks are Georgia / monospace.

## How It Works (Flow)
1. `index.html` links `aisc/aisc.css` (last app sheet) and `aisc/boot.js` (defer,
   after vendor/graph).
2. about-page-editor.js loads any saved copy, restoring `[data-about-runtime]`
   figure shells from the static markup; that resolves `DDAboutContentReady`.
3. When the page is first shown, boot.js loads the scripts; each figure starts on
   its own `AISC.whenVisible` (IntersectionObserver).
4. App theme switches arrive as `delta:theme-changed` → figures repaint.

## Invariants & Constraints
- Every id in the write-up is `aisc-` prefixed; every in-page `href="#…"` names
  one (checked in `../watch_front_door.py`). Links scroll via JS, never set
  `location.hash`.
- Tokens must stay `#rrggbb` hex: `AISC.rgb01` parses them for colour mixing
  (Fig. 5 nodes, lesson colours).
- Never un-nest a rule out of `#aisc-root`: the site's `.card`/`.btn`/`.note`/
  `.status` would restyle the rest of the app.
- Dropped from the standalone site on purpose: its theme toggle (app owns theme),
  and presentation mode (`present.js`, global keydown). Its `bg-field.js` came
  back as `js/field.js`, scoped to this page and masked out of the column.
- Layout (Seth, 2026-09-24, after uchicagoaisafety.com/about-us): ONE column of
  paragraphs, no floating prose cards; the mask keeps the field out of the
  column, so don't reintroduce opaque tiles to hide it. Figures ARE cards
  (Seth, 2026-09-25): `--aisc-wide` (1132px, the standalone site's figure
  width), centred on the column, graphic left and controls in a 300px
  `.fig-side` column on the right (stacked under the graphic below 900px).
- Every figure has a "Full screen" button under its graphic, added by
  `js/fig-expand.js`: `figure.fig.is-max` fixes the card over the viewport,
  Escape or the button closes it, Tab wraps inside it. Fig. 2's button presses
  why-graph's own Maximize instead (the real Knowledge Graph).
- Fig. 2 is the app's own concept map: `../concept-graph/why-graph.js` mounts
  on `#wta-graph-cy` inside `.wta-graph` (styles `../styles/why-map.css`,
  unframed here). `watch_front_door.py` requires exactly one, inside
  `#aisc-fig-graph`. Its Maximize is `position: fixed` inside `#aisc-root`'s
  stacking context, hence `body.wta-max-open #aisc-root { z-index: 300 }`
  (and the same for `body.aisc-fig-max-open`).
  Figs. 3/5/6 copy that map's node/edge look (`js/fig-graphs.js`).
- The contents nav is `#aisc-toc` in the app topbar (`../index.html`,
  `.topbar-mid`), shown by a `body:has(#page-learn-about-app:not(.hidden))`
  rule; on a phone it scrolls sideways in the narrow middle cell. Not inside `#about-page-content`, so never saved.

## Extension Points
- Copy change: edit the aisc repo first, then re-port the section into
  `../index.html` with the `aisc-` prefix (ids + `href`s + JS lookups).
- New figure: add `js/fig-*.js`, list it in `boot.js` `SCRIPTS`, mark its
  `<figure>` `data-about-runtime` with an id.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Hidden page measures 0 wide** — `ACTIVE`
  - When it happens: window resized while another app page is showing.
  - Symptom: Fig. 1 / Fig. 7 redraw at the 800px fallback width.
  - Root cause: `getBoundingClientRect()` of a `display:none` page is 0.
  - Prevention/fix: resize again on the page; cosmetic only.

## Recent Changes
- 2026-09-24: Created — AISC write-up ported onto #page-learn-about-app.
- 2026-09-24: Fig. 4 moved off CindyJS onto plain SVG (`js/fig-forget.js`);
  `vendor/` removed, ~550 KB less on first open.
- 2026-09-24: XLab-style layout — one 720px column of paragraphs, cards and
  tiles unboxed, figures wider (1080px); node field in the gutters
  (`js/field.js`) with an eased mask; contents moved from the in-page sticky
  rail into the app topbar (`#aisc-toc`); hero ornament retired.
- 2026-09-24: Figures centred at column width and unboxed; controls moved
  under each graphic and cut down (Fig. 1: AGI + deploy sliders, scale; Fig. 4:
  model switch + schedule; learner/expert sliders, Play, Fermi inputs, speed,
  target and implicit-review controls removed, their defaults kept). Fig. 2 is
  the old embedded concept map (why-graph.js) with Maximize and cold/mine
  modes; Figs. 3/5/6 restyled to its look. `--aisc-wide` removed.
- 2026-09-25: Standalone site's palette (dark paper #0d1017, violet accent) on
  the whole page; figures back to cards at 1132px with controls on the right
  (the cut-down controls kept); "Full screen" button under every figure
  (`js/fig-expand.js`).
