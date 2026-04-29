# stats

## Purpose
Statistics page UI for Delta Drills — renders priority/learning-rate breakdowns by area and sub-area, the grade history graph, and the advanced per-topic difficulty view. Each sub-tab on the Statistics page is wired up here.

## Owns
- The four Statistics sub-tabs and their panels: **Areas**, **Graph**, **Advanced**, **Predicted course scores**.
- Tab switching logic (`showStatsPanel`) keyed off `data-stats-tab` / `data-stats-panel` pairs in `index.html`.
- Building the `statsData` model from raw subtopic rows + user-edited weights (`buildAreas` in `data.js`).
- Persisting weight + topic-enabled edits to localStorage and syncing to backend (`weights.js`).
- Rendering the Areas table and Advanced table, including expand/collapse state for area rows.
- Rendering the grade-history graph and its day/week/month range controls.

## Does NOT own
- The top-level page navigation tabs (`.tab[data-tab='statistics']`) — those live in the global page-tab system in `app.js`.
- Question data, practice engine, or scoring math beyond what `data.js` re-aggregates — see `practice_engine.py` and `practice.js`.
- Auth / Supabase wiring — see `supabase-practice.js`.

## Key Files
- `dom.js`: cached DOM references (`statsTableBody`, `statsTabs`, `statsPanels`, `graphContainer`, `graphRangeButtons`). Loaded first.
- `weights.js`: load/save user weights and topic-enabled flags; localStorage key `delta_drills_weights`.
- `data.js`: math helpers (`calcDiffMult`, smoothed correctness `p(n)`), `buildAreas()` aggregator, raw-subtopic fetch.
- `render.js`: `renderStatsTable()` (Areas panel) and `renderAdvancedTable()` (Advanced panel) — including weight inputs, topic/subtopic checkboxes, expand toggles.
- `graph.js`: `renderGraph(range)` and `initGraphControls()` — Graph panel with day/week/month ranges over `gradeSeries`.
- `predicted.js`: `renderPredictedTable()` and `buildPredictedAreas()` — Predicted course scores panel. Reads `window.ARENA_STAGE1_PROBLEMS`, groups by chapter, renders the same table shape as Areas.
- `init.js`: bootstrap — `showStatsPanel`, `loadAndRenderStats`, `initStats`, refresh scheduler. Wires tab clicks and the page-level statistics tab.

## Data & External Dependencies
- LocalStorage: `delta_drills_weights` (topic + subtopic weights, enabled flags).
- Backend: subtopic rows fetched via the practice/Supabase layer; weight sync via `weights.js`.
- DOM contracts in `index.html` (around line 372–446 and the new `data-stats-panel="predicted"` block): the four `data-stats-tab` buttons and their matching `data-stats-panel` sections must stay in sync with `init.js#showStatsPanel`.

## How It Works (Flow)
1. Page loads → `dom.js` caches references → `init.js#initStats` binds tab clicks, calls `renderStatsLoadingState`, `showStatsPanel("areas")`, `initGraphControls`, `scheduleStatsRefresh`.
2. `loadAndRenderStats()` fetches raw subtopic rows, runs `buildAreas(items, loadWeights())` → `statsData`, then calls `renderStatsTable()` and `renderAdvancedTable()`.
3. User clicks a sub-tab → `showStatsPanel(target)` toggles `.active` on the button and `.hidden` / `display` on the matching panel.
4. User edits a weight or toggles a topic → handler in `render.js` saves via `weights.js`, rebuilds via `buildAreas`, re-renders both Areas and Advanced tables.
5. User clicks the page-level Statistics tab → `init.js` resets to the Areas sub-tab and re-renders.

## Invariants & Constraints
- Every `data-stats-tab="X"` button MUST have a matching `data-stats-panel="X"` section, otherwise `showStatsPanel` will leave nothing visible. Add both when introducing a new sub-tab.
- The default sub-tab is `areas`; `init.js` hard-codes this in two places (`showStatsPanel("areas")` on init and on page-tab click).
- `buildAreas` is the single source of truth for the `statsData` shape consumed by both renderers; do not mutate `statsData` ad-hoc.
- Never write to `delta_drills_weights` directly — go through `weights.js` so the normalization (`normalizeWeights`) runs.
- Script load order matters: `dom.js` → `weights.js` → `data.js` → `render.js` → `graph.js` → `init.js` (these files share globals; they are not ES modules).

## Extension Points
- **Add a sub-tab**: add a `<button class="stats-tab" data-stats-tab="NAME">` in `index.html` (next to Areas/Graph/Advanced/Predicted) and a matching `<section class="stats-panel hidden" data-stats-panel="NAME">`. Tab switching is automatic — no `init.js` changes unless the new panel needs its own render hook.
- **Predicted course scores panel**: currently a placeholder card. To populate it, add a `renderPredictedTable()` (or similar) in a new `predicted.js` file, wire it into the load/refresh flow in `init.js`, and append a `<script src="stats/predicted.js">` in `index.html` after `render.js`.
- **New graph range**: add a `data-graph-range="..."` button in the Graph panel and extend the switch in `graph.js#renderGraph`.
- **New advanced column**: add a `<th>` in the Advanced `<thead>` and extend the row builder in `render.js#renderAdvancedTable`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- **Sub-tab added without matching panel** — `ACTIVE`
  - When it happens: someone adds a `data-stats-tab` button but forgets the `data-stats-panel` section.
  - Symptom: clicking the tab hides every other panel and shows nothing.
  - Root cause: `showStatsPanel` toggles only on the matching `data-stats-panel` attribute.
  - Prevention/fix: add both the button and the section in the same edit; grep for the new tab name in `index.html` and confirm two hits.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-04-29: Wired the "Predicted course scores" sub-tab to ARENA data. New `predicted.js` reads `window.ARENA_STAGE1_PROBLEMS`, groups by chapter, and renders the same table layout as Areas — chapters as top rows (sorted ascending by avg readiness, so the lowest-readiness chapter is rank 1), ARENA problems as expandable sub-rows. `init.js` now calls `renderPredictedTable()` on stats load and on the predicted sub-tab click.
- 2026-04-29: Added "Predicted course scores" sub-tab as a placeholder panel (button + `data-stats-panel="predicted"` section in `index.html`).
- 2026-04-27: Initial doc created.
