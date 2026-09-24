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
- `aisc.css`: the site's stylesheet, every rule nested under `#aisc-root`; theme
  tokens mapped to the app's `html[data-theme]` light | dark | blue. Also the
  page layout vars (`--aisc-col` / `--aisc-wide` / `--aisc-clear` /
  `--aisc-fade` on `#page-learn-about-app`), the field's mask, and `#aisc-toc`.
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
  paragraphs, no floating cards. Figures (and the five-step `.chain`) break out
  to `--aisc-wide`; nothing else does. Don't reintroduce opaque tiles to hide
  the field: the mask keeps it out of the column.
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
