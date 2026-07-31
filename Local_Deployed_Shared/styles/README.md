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
- `layout.css`: page shell, container, tabs nav layout.
- `components.css`: reusable atoms (buttons, inputs, cards, hints).
- `responsive.css`: viewport-based overrides; loaded last so its rules win.
- `arena.css`: ARENA tab styling.
- `stats.css`: **legacy filename** — no longer a Statistics tab stylesheet (that tab was removed 2026-07-31). Now holds only the `.stats-bar*` progress-bar vocabulary shared by the Practice page bars in `index.html`, `practice/arena-unlock.js`, and `targeted-practice/targeted-practice.js`.
- `courses/`: Courses tab — split into five fragments (`page.css`, `forkgate.css`, `detail.css`, `modal.css`, `responsive.css`) so no single file gets bloated. See `courses/README.md`.
- `practice/`: subfolder for practice-tab CSS fragments.

## Data & External Dependencies
- No runtime data. Pure CSS.
- Depends on Inter (Google Fonts) loaded by index.html.
- Logo URLs referenced from JS (e.g. `learn.arena.education/static/images/arena-logo.png` for the Courses ARENA card) are styled by `.course-card-logo` here.

## How It Works (Flow)
1. `index.html` loads stylesheets in this order: variables → base → layout → components → stats → arena → courses/list → courses/include → courses/detail → courses/modal → courses/responsive → responsive (then practice CSS separately).
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
- 2026-07-31: `stats.css` trimmed from 542 lines to 41 when the Statistics tab was deleted. Everything that styled the tab's tables, sub-tabs, graph, weight inputs, launch pills (`.stats-open-link*`) and prereq panel went with it; only `.stats-bar`, `.stats-bar-track`, `.stats-bar-fill` and `.stats-bar-value` remain, because the Practice page, `arena-unlock` and `targeted-practice` render those. Filename and `.stats-` prefix kept deliberately — renaming them would touch every consumer for no behavioural gain, so a header comment in the file explains the mismatch instead. Link tag bumped to `?v=2`.
- 2026-07-31: Renamed two courses fragments as the Courses tab collapsed to the single ARENA course — `courses/list.css` → `courses/page.css` and `courses/include.css` → `courses/forkgate.css`. `watch.py` here tracks both names in `REQUIRED_CSS`, the link-order assertions, and the token-first hex check; `index.html` link tags bumped to `?v=2`. Still five fragments, same link position between `arena.css` and `responsive.css`.
- 2026-04-29: Added `courses.css` for the new Courses tab (search input + course-card grid). Linked in `index.html` after `arena.css`.
- 2026-04-29: Extended `courses.css` with the per-course article/detail view — back button, hero block, intro paragraph, and alternating-side chapter rows with squarespace-hosted illustrations. Includes a 720px-wide responsive collapse to a single column.
- 2026-04-29: Added chapter-sections modal styling to `courses.css` (`.chapter-modal-backdrop`, `.chapter-modal`, `.chapter-modal-header/-content/-close`, `.section-item`, `.section-number/-info/-title/-desc`, plus `.course-chapter-clickable` hover/focus state and `body.modal-open` scroll lock). Section number color is themable per chapter via `--section-number-color` set inline by JS (falls back to `--accent`). Modal uses neutral `rgba(0,0,0,...)` for backdrop/shadow because they are scrim layers, not brand surfaces — no token exists for these and one would be misleading.
- 2026-04-29: Split monolithic `courses.css` (~452 LOC, YELLOW) into the `courses/` subfolder with five focused fragments (`list.css`, `include.css`, `detail.css`, `modal.css`, `responsive.css`). All five linked individually in `index.html` between `arena.css` and `responsive.css`. Parent `watch.py` updated for the new layout; `courses/watch.py` enforces the per-fragment selector ownership contract so future edits don't smear concerns across files.
- 2026-04-27: Initial doc created.
