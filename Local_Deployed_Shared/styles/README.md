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
- `stats.css`: Statistics tab styling.
- `courses.css`: Courses tab — search input + result card styling. Added 2026-04-29.
- `practice/`: subfolder for practice-tab CSS fragments.

## Data & External Dependencies
- No runtime data. Pure CSS.
- Depends on Inter (Google Fonts) loaded by index.html.
- Logo URLs referenced from JS (e.g. `learn.arena.education/static/images/arena-logo.png` for the Courses ARENA card) are styled by `.course-card-logo` here.

## How It Works (Flow)
1. `index.html` loads stylesheets in this order: variables → base → layout → components → stats → arena → courses → responsive (then practice CSS separately).
2. The order matters — `responsive.css` and feature stylesheets must come after the tokens and base layer so their selectors override correctly.
3. Each feature CSS file scopes its rules with a feature-prefixed class (`.arena-*`, `.stats-*`, `.courses-*`, `.course-card-*`) to avoid collisions.

## Invariants & Constraints
- New feature stylesheets must reference colors via CSS variables from `variables.css` — never hardcode hex values. Watch enforces this on token-first files (currently `courses.css`); legacy `arena.css` and `stats.css` predate the rule and are exempt until refactored.
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
- 2026-04-29: Added `courses.css` for the new Courses tab (search input + course-card grid). Linked in `index.html` after `arena.css`.
- 2026-04-29: Extended `courses.css` with the per-course article/detail view — back button, hero block, intro paragraph, and alternating-side chapter rows with squarespace-hosted illustrations. Includes a 720px-wide responsive collapse to a single column.
- 2026-04-29: Added chapter-sections modal styling to `courses.css` (`.chapter-modal-backdrop`, `.chapter-modal`, `.chapter-modal-header/-content/-close`, `.section-item`, `.section-number/-info/-title/-desc`, plus `.course-chapter-clickable` hover/focus state and `body.modal-open` scroll lock). Section number color is themable per chapter via `--section-number-color` set inline by JS (falls back to `--accent`). Modal uses neutral `rgba(0,0,0,...)` for backdrop/shadow because they are scrim layers, not brand surfaces — no token exists for these and one would be misleading.
- 2026-04-27: Initial doc created.
