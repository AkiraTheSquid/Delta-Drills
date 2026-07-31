# courses (styles/)

## Purpose
- All CSS for the Courses tab, split into focused fragments so no single file gets bloated. Owns the visual layer for the ARENA article (hero, source links, chapter rows), the chapter-sections modal, and the GitHub fork gate shown on the first Colab click.

## Owns
- The `.courses-*`, `.course-*`, `.chapter-modal-*`, `.section-*`, and `.fork-gate-*` selector families.
- The narrow-viewport (≤720px) overrides specific to courses-tab markup.

## Does NOT own
- The Courses tab DOM structure or behavior — that lives in `Local_Deployed_Shared/courses.js` and `Local_Deployed_Shared/courses-fork-gate.js`.
- The design token palette — defined in `../variables.css`. Every color/spacing reference here uses `var(--token)`.
- Global responsive overrides — those live in `../responsive.css` and are applied across the whole app.

## Key Files
- `page.css`: the `.courses-page` container. Small on purpose — the tab renders one course directly, so there is no list view to style.
- `forkgate.css`: the first-click GitHub fork gate — backdrop, dialog shell, numbered steps, username input, and the status/error line.
- `detail.css`: article view — hero block, the `.course-sources` link pills (ARENA site + Callum's repo), intro paragraph, alternating chapter rows, and the `.course-chapter-clickable` hover/focus state.
- `modal.css`: chapter-sections modal — backdrop, dialog shell, header/×/content layout, and `.section-item/.section-number/.section-info` rows. Section number color is themed via `--section-number-color` set inline by JS on `.chapter-modal`.
- `responsive.css`: ≤720px overrides for the courses tab. Loaded last so its rules win over the four feature fragments above.

## Data & External Dependencies
- No runtime data. Pure CSS.
- Depends on `../variables.css` tokens (`--bg`, `--surface`, `--card`, `--text`, `--muted`, `--accent`, `--accent-dark`, `--border`, `--white`).
- Inter font is loaded by `index.html`; everything here uses `font-family: inherit`.

## How It Works (Flow)
1. `index.html` links the five fragments in this order, after `../arena.css` and before `../responsive.css`:
   `page.css` → `forkgate.css` → `detail.css` → `modal.css` → `responsive.css`.
2. Each fragment scopes its rules with feature-prefixed classes — fragments don't reference each other's selectors, so individual files can be loaded in isolation for previews/tests.
3. The 720px breakpoint in `responsive.css` overrides specific selectors from `detail.css` and `forkgate.css`. It must come after them in the link order.

## Invariants & Constraints
- **No raw hex colors.** Every color uses a `var(--token)` from `../variables.css`. The only exception is `rgba(0,0,0,...)` neutral scrims in `modal.css` (backdrop and drop shadow) — those are not brand surfaces and have no token equivalent. The parent `../watch.py` enforces this with a regex on `#` literals.
- **Link order matters.** `responsive.css` must come last. The four feature fragments must come before both `responsive.css` (this folder's) and `../responsive.css` (the global one).
- **Selector prefixes stay unique.** Use `.courses-*`, `.course-*`, `.chapter-modal-*`, `.section-*`, or `.fork-gate-*`. Never introduce a generic selector that could leak into another tab.
- **Fragment ownership is enforced.** `watch.py` in this folder maps each prefix family to exactly one file and fails if a fragment defines a class another fragment owns. Renaming a fragment means updating `watch.py` here, `../watch.py`, and the `index.html` link in the same change.
- **Each fragment stays small.** The whole point of splitting is to keep individual files under ~150 LOC. If one approaches that, split further (e.g., a new `chapter.css` extracted from `detail.css`).

## Extension Points
- New courses-tab feature → add a new file here (e.g., `progress.css`), link it in `index.html` after `modal.css` and before `responsive.css`, and prefix its selectors with `.courses-` or `.course-`.
- New design token → add to `../variables.css` only. Never define ad-hoc color/spacing constants in this folder.
- Modal that's not chapter-sections → consider a generic `dialog.css` at the parent `styles/` level instead of duplicating modal scaffolding here.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Drift from token-first rule** — `ACTIVE`
  - When it happens: a quick iteration inlines `#abc123` instead of reaching for a `var(--*)`.
  - Symptom: theme changes in `variables.css` skip those rules and the UI looks inconsistent.
  - Root cause: copy-paste from prototypes that predate the variable system.
  - Prevention/fix: parent `../watch.py` runs `hex_re` against every file in this folder and fails the build. Add a token to `variables.css` if none fits.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-07-31: Courses tab reduced to the single ARENA course. `list.css` → `page.css` (only `.courses-page` survives; the search input, results grid, and course-card styling went with the deleted list view). `include.css` → `forkgate.css`, repurposed from the removed "Include course for study?" toggle to the new first-click GitHub fork gate. `detail.css` dropped `.course-back-btn` and gained `.course-sources`/`.course-source-link` pills. `responsive.css` swapped its dead `.course-card`/`.course-include-*` overrides for `.fork-gate*` ones. Fragment count is still five; `watch.py` here and `../watch.py` updated to the new names and selector families.
- 2026-04-29: Folder created. Split the monolithic `../courses.css` (~452 LOC, YELLOW) into five fragments here. Linked all five in `index.html` between `../arena.css` and `../responsive.css`. Parent `../watch.py` updated to validate the new layout, link order, and per-fragment hex-color enforcement.
