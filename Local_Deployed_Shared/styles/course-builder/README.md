# course-builder (styles/)

## Purpose
- The course builder page's look (`course-builder/`): top strip, chat pane +
  attachments tray, draggable divider, dot-grid graph pane, node card, lesson
  view, full-screen mode, narrow-screen stacking.

## Owns
- The `.cb-*` selector family and `#page-course-builder`.

## Does NOT own
- The chat's own skin — Deep Chat's shadow DOM, themed by `conceptual/conceptual_chat_theme.js`.
- The lesson prose (`.kg2-*`, `../how-it-works.css`).
- The catalog's "Add your own course" row (`../courses/course-add.css`).

## Key Files
- `builder.css`: everything, in page order (strip → split → divider → right half → card → lesson → narrow).

## Data & External Dependencies
- Tokens from `../variables.css` only, so every theme restyles it.

## How It Works (Flow)
1. `--cb-left` on `.cb-shell` (set by `course-builder.js`) sizes the chat column.
2. `.is-full` collapses the chat + divider columns to 0; `.is-dragging` blocks pointer events on both panes mid-drag.
3. `.cb-view-pane[data-view]` switches the crumb and tools between graph and lesson.
4. A container query hides the bar's secondary labels when the graph pane is under 560px.

## Invariants & Constraints
- No raw hex colours; tokens only (the Cytoscape canvas colours are set in JS, not here).
- Link after `../how-it-works.css` (lesson rules) in `index.html`.

## Extension Points
- New pane state → a `data-view` value and its show/hide pair here.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- None yet.

## Recent Changes
- 2026-09-29: Created with the course builder.
- 2026-09-29: ≤820px: two grid rows, not three — the hidden divider left the graph pane auto-placed into a zero-height row. Full screen on narrow = `0 minmax(0,1fr)`.
