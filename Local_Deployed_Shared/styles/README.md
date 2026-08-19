# styles

## Purpose
- Global CSS for the Delta Drills root UI. Owns the visual layer: color tokens, layout primitives, reusable component styles, and per-feature stylesheets that index.html links directly.

## Owns
- The CSS variable palette in `variables.css` (`--bg`, `--surface`, `--card`, `--text`, `--muted`, `--accent`, `--accent-dark`, `--border`, `--white`).
- Cross-cutting layout, base, component, and responsive rules.
- Per-feature stylesheets co-located here (arena, stats, courses) that style a single tab/page.

## Does NOT own
- Practice tab styling — lives at `Local_Deployed_Shared/practice.css` and `Local_Deployed_Shared/practice/bars.css` (loaded directly by index.html).
- DOM structure, behavior, or data — those are owned by the JS in the parent folder and feature subfolders.
- Backend / Python concerns.

## Key Files
- `variables.css`: design tokens (colors). Loaded first so every other stylesheet can reference them.
- `base.css`: resets and global element defaults.
- `scrollbars.css`: **base layer, not a feature.** `color-scheme: dark` on `:root` plus the `::-webkit-scrollbar` skin every scroller in the app inherits. Deliberately unscoped — a scrollbar is chrome, not content, and nothing here wants the platform's white one. The two mechanisms are not redundant: Chrome ignores `color-scheme` for any scroller a `::-webkit-scrollbar` rule matches, and `color-scheme` reaches bars the skin cannot. Must be linked before `layout.css`; `watch.py` asserts it.
- `layout.css`: page shell, container, tabs nav layout.
- `nav-drawer.css`: the hamburger menu the tab strip moves into below 900px — the toggle, the off-canvas drawer, its scrim, and `.tabs`/`.tab` restyled as a vertical list. Every rule is scoped under `body.nav-drawer-mode`, a class `../nav-drawer.js` adds, so the topbar strip is untouched at every other width (this file does not violate the "no global selector over the tabs nav" rule below). The `.tabs` grid is two columns so each tab and its ⓘ share a row. Behaviour, and the DOM move itself, live in `../nav-drawer.js`.
- `components.css`: reusable atoms (buttons, inputs, cards, hints).
- `responsive.css`: viewport-based overrides; loaded last so its rules win.
- `arena.css`: ARENA tab styling.
- `infotips.css`: the ⓘ system — the dot (`.dd-info`, plus its topbar variant `.dd-info.tab-info`) and the single reused explanation panel (`.dd-infopop`). Behaviour and copy live in `../infotips.js` and `../infotips-registry.js`.
- `stats.css`: **legacy filename** — no longer a Statistics tab stylesheet (that tab was removed 2026-07-31). Now holds only the `.stats-bar*` progress-bar vocabulary shared by the Practice page bars in `index.html`, `practice/arena-unlock.js`, and `targeted-practice/targeted-practice.js`.
- `courses/`: Courses tab — split into five fragments (`page.css`, `forkgate.css`, `detail.css`, `modal.css`, `responsive.css`) so no single file gets bloated. See `courses/README.md`.
- `practice/`: subfolder for practice-tab CSS fragments.

## Data & External Dependencies
- No runtime data. Pure CSS.
- Depends on Inter (Google Fonts) loaded by index.html.
- Logo URLs referenced from JS (e.g. `learn.arena.education/static/images/arena-logo.png` for the Courses ARENA card) are styled by `.course-card-logo` here.

## How It Works (Flow)
1. `index.html` loads stylesheets in this order: variables → base → scrollbars → layout → components → how-it-works → stats → arena → courses/page → courses/forkgate → courses/detail → courses/modal → courses/responsive → infotips → nav-drawer → responsive (then practice CSS separately).
2. The order matters — `responsive.css` and feature stylesheets must come after the tokens and base layer so their selectors override correctly.
3. Each feature CSS file scopes its rules with a feature-prefixed class (`.arena-*`, `.stats-*`, `.courses-*`, `.course-card-*`) to avoid collisions. `.stats-*` is the one prefix that no longer maps to a tab — it survives as the shared progress-bar vocabulary.

## Invariants & Constraints
- New feature stylesheets must reference colors via CSS variables from `variables.css` — never hardcode hex values. Watch enforces this on token-first files (currently every `courses/*.css` fragment); legacy `arena.css` and `stats.css` predate the rule and are exempt until refactored.
- `variables.css` must be the first stylesheet linked in `index.html`, and `responsive.css` must be linked after `components.css` and after every feature stylesheet so its overrides win.
- Feature class prefixes must stay unique per tab (e.g. `.courses-*` for the Courses page, `.arena-*` for ARENA) to keep concerns isolated.
- Do not reintroduce a global selector that overrides the tabs nav layout — that lives in `layout.css` and is fragile.

## Extension Points
- New tab/page styling: add `<feature>.css` here, link it in `index.html` after `components.css` and before `responsive.css`, prefix all selectors with `.<feature>-` or a feature-scoped wrapper class.
- New design tokens: add to `variables.css` only. Never define ad-hoc color or spacing constants in a feature file.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Hardcoded colors drift from the token palette** — `ACTIVE`
  - When it happens: a feature stylesheet inlines `#abc123` instead of `var(--accent)` because the author was iterating quickly.
  - Symptom: theme/palette changes in `variables.css` skip those rules and the UI looks inconsistent.
  - Root cause: copy-paste from prototypes that predate the variable system.
  - Prevention/fix: always reach for an existing `--*` token; add a new one to `variables.css` if none fits.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-08-18: Two new stylesheets, both about the app being read at side-panel width (the Chrome extension frames it at ~300-400px). `scrollbars.css` — linked with the base layer, before `layout.css` — replaces the platform's white scrollbar track, which on this palette was a bright stripe down the right edge of every scroller and across the bottom of every horizontal one. Note `::-webkit-scrollbar-corner`: unstyled it leaves one white square where the two bars meet, exactly at the corner of the practice rail. `nav-drawer.css` (linked after `infotips.css`, before `responsive.css`) turns the tab strip into a hamburger drawer below 900px — the strip is **moved** into it by `../nav-drawer.js`, never copied, because `app.js` holds the tab elements in NodeLists captured at eval time. In the drawer `.tabs` becomes a two-column grid, which is what puts each tab and its ⓘ on one row given the flat `tab, dot, tab, dot` markup; it only stays aligned because a tab and its dot are always hidden together. Four new tokens in `variables.css` (`--scroll-thumb`, `--scroll-thumb-hover`, `--drawer-width`, `--scrim`), all four now in `watch.py`'s `REQUIRED_TOKENS` since both files fail silently without them. `responsive.css` (`?v=2`) lost its `≤600px` topbar block: it stacked `.topbar` into a column for a tab strip that is no longer there, leaving an empty second row and breaking `.practice-container`'s `height: calc(100vh - 56px - 40px)`, which assumes a 56px bar.
- 2026-08-07: Added `infotips.css` for the ⓘ beside every tab and feature (linked before `responsive.css`, `?v=1`). Two placement rules matter to anyone touching it: `.dd-infopop` is `position: fixed` because the topbar, the practice split and the knowledge-graph pane all clip, and `.tab.has-info` drops its right padding so a tab and its dot read as one group rather than two. Also two fixes in `layout.css` (`?v=2`) that only became visible with the extra tab: `.tabs` gained `margin-right: auto` so the strip stays against the logo instead of sliding to the right edge whenever `#topbar-auth` is hidden, and `.tab` gained `white-space: nowrap; flex: 0 0 auto` — "Why This App Exists" was wrapping to four lines at 390px instead of scrolling, which `.tabs { overflow-x: auto }` was already there to handle.
- 2026-08-02 (later still): `how-it-works.css` — `html.dd-colab-edition .kg2-info { display: none }`. On that deploy the lesson is in the notebook, so the graph takes the full wrap and the docked readout names the selected concept; `kc-colab-route.js` moves `#kg-colab-link` and `#kg-maximize` out of the hidden aside into `.kg2-controls`, and the two rules under `.kg2-controls` here resize them to match the strip they land in. Note these rules live in `how-it-works.css`, not `practice/colab-edition.css` — that file owns the practice page, this one owns every `.kg2-*` selector, and splitting one component's rules across two stylesheets by deploy is how load order becomes a puzzle. Link tag `?v=18`.
- 2026-08-02 (later): `how-it-works.css` — the Knowledge Graph's idle state stopped eating a third of a side panel. `.kg2-info:has(.kg2-placeholder)` hides the (empty) header bar and tightens the body's padding, `.kg2-placeholder-more` drops the second sentence below 820px, and `.kg2-graph:not(.has-dock)` reserves 44px instead of 144/220px until a concept has been shown in the dock — a class `lesson-graph.js` adds once and never removes, because toggling it on hover-out would resize the Cytoscape canvas under the cursor. Link tag `?v=17`.
- 2026-08-02: `how-it-works.css` — the Knowledge Graph's two panes swapped sides (graph left with its docked readout, lesson right), and the Mastery ↔ Lessons toggle is styled by `#kg-colormode` instead of `.kg2-seg`. That class is *also* on every lesson segment block, so `display: inline-flex` was landing on those too and laying each concept out in a column beside its own worked example, with the pane scrolling sideways — one name, two unrelated things. Added `.kg2-colab-link` for the Colab-edition link into the notebook section that teaches the selected concept. Link tag `?v=16`.
- 2026-07-31: `stats.css` trimmed from 542 lines to 41 when the Statistics tab was deleted. Everything that styled the tab's tables, sub-tabs, graph, weight inputs, launch pills (`.stats-open-link*`) and prereq panel went with it; only `.stats-bar`, `.stats-bar-track`, `.stats-bar-fill` and `.stats-bar-value` remain, because the Practice page, `arena-unlock` and `targeted-practice` render those. Filename and `.stats-` prefix kept deliberately — renaming them would touch every consumer for no behavioural gain, so a header comment in the file explains the mismatch instead. Link tag bumped to `?v=2`.
- 2026-07-31: Renamed two courses fragments as the Courses tab collapsed to the single ARENA course — `courses/list.css` → `courses/page.css` and `courses/include.css` → `courses/forkgate.css`. `watch.py` here tracks both names in `REQUIRED_CSS`, the link-order assertions, and the token-first hex check; `index.html` link tags bumped to `?v=2`. Still five fragments, same link position between `arena.css` and `responsive.css`.
- 2026-04-29: Added `courses.css` for the new Courses tab (search input + course-card grid). Linked in `index.html` after `arena.css`.
- 2026-04-29: Extended `courses.css` with the per-course article/detail view — back button, hero block, intro paragraph, and alternating-side chapter rows with squarespace-hosted illustrations. Includes a 720px-wide responsive collapse to a single column.
- 2026-04-29: Added chapter-sections modal styling to `courses.css` (`.chapter-modal-backdrop`, `.chapter-modal`, `.chapter-modal-header/-content/-close`, `.section-item`, `.section-number/-info/-title/-desc`, plus `.course-chapter-clickable` hover/focus state and `body.modal-open` scroll lock). Section number color is themable per chapter via `--section-number-color` set inline by JS (falls back to `--accent`). Modal uses neutral `rgba(0,0,0,...)` for backdrop/shadow because they are scrim layers, not brand surfaces — no token exists for these and one would be misleading.
- 2026-04-29: Split monolithic `courses.css` (~452 LOC, YELLOW) into the `courses/` subfolder with five focused fragments (`list.css`, `include.css`, `detail.css`, `modal.css`, `responsive.css`). All five linked individually in `index.html` between `arena.css` and `responsive.css`. Parent `watch.py` updated for the new layout; `courses/watch.py` enforces the per-fragment selector ownership contract so future edits don't smear concerns across files.
- 2026-04-27: Initial doc created.
