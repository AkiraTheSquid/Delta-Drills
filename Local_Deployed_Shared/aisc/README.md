# aisc

## Purpose
- The app's "Why this app exists" / "How this app works" page
  (`#page-learn-about-app`): five articles, one open at a time (2026-09-25).
  Article 1 is Seth's own pre-AISC text; 2–5 are the AISC-12 proposal
  write-up (the case, how it works, evidence, the AISC plan) with seven live
  figures.
- Ported 2026-09-24 from the standalone site `~/Applications/delta-drills-aisc`
  (Seth: replace both halves of the page with the whole site).

## Owns
- Styles for the write-up (`aisc.css`), its lazy script loader (`boot.js`),
  the figure scripts (`js/`) and their data snapshots (`data/`).

## Does NOT own
- The markup: it lives inline in `../index.html` inside `#about-page-content`
  (so the owner-only About editor, `../about-page-editor.js`, can edit it).
- The copy's source of truth: `~/Applications/delta-drills-aisc`
  (`index.html`, `content/copy.md`) for articles 2–5. Re-port from there;
  don't fork the prose. Article 1 is Seth's own words (from `../index.html`
  at 43b78681); keep them verbatim.
- Menu routing: `../account-menu.js` (`data-lab-open` = id to scroll to; it
  asks `articles.js` to open the article holding it first).
- Cytoscape + dagre: the app's `../vendor/graph/`.

## Key Files
- `aisc.css`: the site's stylesheet, every rule nested under `#aisc-root`. Theme
  tokens = the standalone site's own palette, on `#page-learn-about-app` (so the
  page background and the field share it): its dark paper for the app's blue and
  dark themes, its light paper for `html[data-theme="light"]`. Also the field
  canvas rule (`--aisc-clear` = the 1180px card measure) and `#aisc-toc`.
- `watch.py`: scoped-rule check, plus the wireframe card list in `aisc.css`
  must match `js/field.js` `CARDS`.
- `articles.js` (loaded eagerly, defer): `window.AISCArticles` — `show(id)`,
  `openFor(el)`, `current()`. Hides every `article.aisc-article` but the open
  one (`hidden` attribute), marks its link in the topbar (`#aisc-toc`),
  and handles every `#aisc-…` link: opens the
  article holding the target, then scrolls. Fires a window `resize` after a
  switch so the SVG figures redraw at their real width.
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
3. articles.js re-applies the open article (the saved copy carries stale
   `hidden` flags). Article 1 is open on load.
4. When the page is first shown, boot.js loads the scripts; each figure starts on
   its own `AISC.whenVisible` (IntersectionObserver) — so a figure in a hidden
   article starts the first time its article is opened.
5. App theme switches arrive as `delta:theme-changed` → figures repaint.

## Invariants & Constraints
- Every id in the write-up is `aisc-` prefixed; every in-page `href="#…"` names
  one (checked in `../watch_front_door.py`). Links scroll via JS, never set
  `location.hash`.
- Tokens must stay `#rrggbb` hex: `AISC.rgb01` parses them for colour mixing
  (Fig. 5 nodes, lesson colours).
- Never un-nest a rule out of `#aisc-root`: the site's `.card`/`.btn`/`.note`/
  `.status` would restyle the rest of the app.
- Dropped from the standalone site on purpose: its theme toggle (app owns theme),
  its in-page rail (contents are in the topbar), and presentation mode
  (`present.js`, global keydown). Its `bg-field.js` came back as `js/field.js`,
  scoped to this page.
- Layout = the standalone site's (Seth, 2026-09-25: "everything needs to go
  back to the card style … remove the gradient thing"): 1180px measure, every
  block of running text on an opaque card (the "solid tiles" rules, kept last
  in the `#aisc-root` block), the node field unmasked behind the whole page.
  Figures are cards at the full measure, graphic left and controls in a 300px
  `.fig-side` column on the right (stacked under the graphic below 900px). No
  fade/mask on the field. The 2026-09-24 one-column XLab layout is retired;
  only its topbar contents (`#aisc-toc`) stayed.
- The cards are see-through wireframes (Seth, 2026-09-25: "an invisible card
  there"): no fill or shadow, a thin `--wire` outline. Readability comes from
  `js/field.js`, which erases its drawing inside every card box, so it reads
  as running behind opaque cards. Since 2026-09-25 (Seth: "just a normal
  background graph") the cards push and pull no nodes; the motion is the
  original site's `bg-field.js`. A new kind of card goes in BOTH the wireframe rule
  (end of the `#aisc-root` block) and `field.js` `CARDS`; `watch.py` checks.
  Only a full-screen figure stays opaque (it covers the page).
- Hover and title cards (Seth, 2026-09-25): the card under the cursor gets
  `.is-hot` from `js/field.js`, its outline turns the cursor's colour
  (`--field-edge`). A section's title card (`.tile:has(> .kicker)`) has a
  2px outline that is always lit. Every `.tile` is centred in the column. The title markup is not
  given a class: saved About copies carry the old markup.
  `.is-hot` has to out-rank every wireframe selector: `figure.fig`, `.qs li`,
  `.modes .hand` and `.hero > div:first-child` each carry their own
  `.is-hot` variant in `aisc.css`, or the hovered figure never lights.
- Controls are cut down but the information is not (Seth, 2026-09-25): each
  figure's readout still states what its fixed defaults do, and Fig. 4 always
  runs with the fractional implicit reps on.
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
  Since 2026-09-25 it lists the articles, not sections.
- Articles (Seth, 2026-09-25): `<article class="aisc-article" id="aisc-a-…">`
  directly in `.aisc-main`; the topbar is the only switcher (the in-page
  `#aisc-articles` index of cards was removed the same day). Articles 2–5 ship `hidden` (no flash before `articles.js`). Order
  keeps figure numbers rising: why → case (Fig. 1) → how (Figs. 2–6) →
  evidence (Fig. 7) → AISC proposal (hero, status, plan, risks, lead,
  questions). Section kickers lost their 01–09 numbers. Each article ends
  in a `.art-next` link to the next.
- The About editor applies a saved copy only if it has `id="aisc-a-why"` and
  no `id="aisc-articles"`: a copy saved before the split, or while the index
  was up, would put that page back. Prod's
  saved copy (2026-09-25 14:32 UTC) predates it and is ignored; it held no
  prose edits (only runtime-written text).
- `concept-graph/why-graph.js` draws Fig. 2 when it first has a box: it
  watches the page SUBTREE for `class` and `hidden` changes, because opening
  article 3 changes nothing on the page's own class list.

## Extension Points
- Copy change: edit the aisc repo first, then re-port the section into
  `../index.html` with the `aisc-` prefix (ids + `href`s + JS lookups).
- New figure: add `js/fig-*.js`, list it in `boot.js` `SCRIPTS`, mark its
  `<figure>` `data-about-runtime` with an id.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Hidden page measures 0 wide** — `ACTIVE` (article switches are
  handled: `articles.js` fires a `resize` after each)
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
- 2026-09-25: Back to the standalone site's card layout everywhere: prose on
  opaque tiles at the 1180px measure, hero ornament back, field unmasked (the
  gutter fade is gone). Palette, topbar contents, figure cards and full screen
  kept.
- 2026-09-25: Cards made see-through wireframes; the node field now keeps
  out of them itself (pushes nodes out, erases links inside). `watch.py`
  checks the CSS and JS card lists match.
- 2026-09-25: The cursor drags the field's nodes (by link strength); the
  hovered card lights up and pulls nodes in; section title cards centred,
  2px always-lit outline, strongest pull.
- 2026-09-25: Figures light up on hover too (`figure.fig` out-ranked
  `.is-hot`). Readouts restored to the standalone site's detail (Fig. 1
  mechanism list + understanding numbers, Fig. 3 probe id + counts, Fig. 5
  practised concept id + one line per credited concept, Fig. 6 seven-line
  log). Fig. 4 shows implicit reps always: thin bars tagged +0.3, a key, and
  an implicit-reps line in the readout.
- 2026-09-25: Page split into five articles (`articles.js`), one open at a
  time, with an index of cards up top and the topbar listing articles.
  Article 1 "Why this app exists" is Seth's pre-AISC text again (the three
  markers, the three steps, the loop, the map), in the write-up's tiles and
  cards. Account menu lands on `#aisc-a-why` / `#aisc-why-use`; the editor
  ignores copies saved before the split.
- 2026-09-25: Article index cards removed (topbar switches articles); every
  tile centred in the column; `js/field.js` back to the original site's plain
  background motion (no card push-out, no hover/title pull, no cursor drag),
  still erased inside cards. Editor guard now keys on `#aisc-a-why` without
  `#aisc-articles`.
