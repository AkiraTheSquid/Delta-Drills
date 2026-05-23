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
- `predicted-links.js`: URL helpers used by `predicted.js` — `bookHrefForNotebook` (jupyter-book static HTML), `colabUpstreamHref` (Colab via `callummcdougall/ARENA_3.0`), `vsCodeHrefFor` (`vscode://file/...` local repo path), `openLinkCell` (single-pill `<td>` wrapper for chapter/section/problem rows).
- `predicted-data.js`: data-shaping helpers — `computeProblemScore` (calls `window.computeArenaReadiness` with section fallback), `exercisesForProblem` (reads `window.ARENA_EXERCISES_BY_NOTEBOOK`), section sort/key helpers (`compareSectionLabels`, `subsectionKeyForProblem`, `sectionLabelForProblem`, `sectionNumberFromLabel`, `parseSectionPath`), and skill aggregators (`aggregateTopSkill`, `topSkillLabel`).
- `predicted-prereqs-temp.js`: **TEMPORARY SCAFFOLD — delete when the real concept graph ships.** Exposes `window.ARENA_PREREQS_TEMP_ENABLED` (kill-switch), `ARENA_PREREQS_TEMP_RESTRICT_CHAPTER`/`_PROBLEM_ID`/`_NOTEBOOK_PATH` (scope the table to ARENA 0.0 Prerequisites only), `ARENA_PREREQS_TEMP_EXERCISES` (the 26 0.0 exercises since the auto-extracted registry is `[]` for that notebook), `ARENA_PREREQS_TEMP_BY_EXERCISE` (per-exercise hardcoded `[{topic, subtopic, minPct}]` against Delta Drills topics Numpy/Einops/Einsum), and `getArenaPrereqSubtopicScore(topic, subtopic)` (reads the adaptive baseline). Built so the frontend pipeline can be tested end-to-end against a tiny ARENA slice before the concept-graph backend lands.
- `predicted.js`: `renderPredictedTable()` and `buildPredictedAreas()` — Predicted course scores panel. Reads `window.ARENA_STAGE1_PROBLEMS`, groups by chapter and subsection, renders an expandable chapter → subsection → problem → exercise tree (4 levels). Each exercise row exposes Read / Colab / VS Code / 📋 (copy heading) actions. When `window.ARENA_PREREQS_TEMP_ENABLED` is true (the temp scaffold), the renderer also (a) falls back to the temp exercise list for the 0.0 notebook (registry is empty there), (b) adds a 5th expand level per exercise — but ONLY for the 26 exercises whose titles match a key in `ARENA_PREREQS_TEMP_BY_EXERCISE` (currently only 0.0). Other chapters render exactly as before. (c) Auto-expands the 0.0 chapter/section/problem on first render so the prereq-tagged exercises are visible without hunting.
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
- Script load order matters: `dom.js` → `weights.js` → `data.js` → `render.js` → `graph.js` → `predicted-links.js` → `predicted-data.js` → `predicted-prereqs-temp.js` → `predicted.js` → `init.js` (these files share globals; they are not ES modules). `predicted-links.js`, `predicted-data.js`, and `predicted-prereqs-temp.js` must load before `predicted.js` because the renderer reads their globals during render. The temp file can be deleted in one step (remove the `<script>` tag and the `ARENA_PREREQS_TEMP_*` reads in `predicted.js`) without breaking the load chain.

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
- 2026-05-18 (un-scope: restore full curriculum): Reverted the chapter filter — the predicted-scores table now renders ALL chapters again. The temp prereq panel is still wired, but it only attaches to exercise rows whose title appears in `ARENA_PREREQS_TEMP_BY_EXERCISE` (currently only the 26 0.0 Prerequisites exercises). The 0.0 chapter/section/problem is still auto-expanded on first render so the user lands directly on the prereq-tagged exercises; other chapters stay collapsed as before. Initial commit hid chapters 1–4 — that was wrong, fixed.
- 2026-05-18 (temp prereq scaffold for ARENA 0.0): Added `stats/predicted-prereqs-temp.js` and wired it into `predicted.js`. Purpose: shake out the predicted-scores frontend pipeline end-to-end against a tiny slice of ARENA (only 0.0 Prerequisites, 26 exercises) before the real concept-graph backend lands. When `window.ARENA_PREREQS_TEMP_ENABLED` is true: (1) `buildPredictedAreas` filters problems to chapter `chapter0_fundamentals` + problem id `arena-0.0-prereqs`; (2) `getExercisesWithTempFallback` substitutes the temp exercise list because `arena/exercises.js` extracted `[]` for that notebook; (3) the renderer auto-expands chapter/section/problem on first load (one-shot, via `window._arenaPrereqsTempSeeded`); (4) each exercise leaf row gets a 5th-level ▸ toggle that opens a `stats-prereq-panel` listing the Delta Drills prereq concepts (Numpy / Einops / Einsum subtopics) with bars showing current baseline vs `minPct` target + `Practice ↗` buttons that click the page-level Practice tab. Concept-jumping is intentionally tab-level only (no topic filter passed) — kept scoped per user's "very basic" instruction. Per-exercise prereq map is hand-bucketed by canonical-solution category (`rearrange` / `repeat` / `reduce` / `indexing`). The full upstream concept graph lives in `Local_Deployed_Shared/arena_prereqs_structured.json` and is being built by another agent — once it ships, delete `predicted-prereqs-temp.js`, the `<script>` tag in `index.html`, and the `ARENA_PREREQS_TEMP_*` reads in `predicted.js`. New CSS classes appended to `styles/stats.css`: `stats-prereq-panel`, `stats-prereq-title`, `stats-prereq-item`, `stats-prereq-label`, `stats-prereq-bar`, `stats-prereq-bar-track`, `stats-prereq-bar-fill`, `stats-prereq-met`, `stats-prereq-unmet`, `stats-prereq-bar-value`, `stats-prereq-target`, `stats-prereq-jump`.
- 2026-05-16 (backtick strip): `predicted.js` now routes `data-copy-key` payloads through a new `copyKeyAttr(text)` helper that strips markdown backticks before clipboard write. Reason: Jupyter Book / Colab renders code spans (`` `nn.Module` ``) as plain `nn.Module`, so Ctrl+F for the raw title with backticks finds nothing. Helper also keeps the existing `"` → `&quot;` HTML-attr escape. All three pills (Colab, VS Code, 📋) use it.
- 2026-05-16 (URL-encoding fix): Switched all three URL builders in `predicted-links.js` from `encodeURI` to per-segment `encodeURIComponent` (new `encodePathSegments(path)` helper). `encodeURI` leaves `&`, `?`, `#` untouched, so notebooks with `&` in their filename (`0.2_CNNs_&_ResNets_exercises.ipynb`, `0.5_VAEs_&_GANs_exercises.ipynb`, `1.3.2_Function_Vectors_&_Model_Steering_exercises.ipynb`, `2.2_DQN_&_VPG_exercises.ipynb`, etc.) generated URLs where Colab's parser treated `&_ResNets…` as a query parameter and 404'd / silently failed. Per-segment encoding maps `&` → `%26` without touching `/` separators. Applies to the Read / Colab / VS Code pills.
- 2026-05-16 (even later): `Colab ↗` pills now respect a per-user GitHub-fork override. New Account form field `account-github-username` (persisted to `localStorage["account_github_username"]`) tells `predicted-links.js#arenaColabOwner()` whose ARENA fork to point Colab URLs at. When empty, falls back to upstream `callummcdougall/ARENA_3.0` (read-only). When set, opens `<username>/ARENA_3.0/blob/main/...` — combined with Colab's File → Save a copy in GitHub, a student's notebook state persists across visits (their fork is the storage layer). Predicted-scores subtitle now has a one-line tip pointing students at Account → GitHub username. No backend changes — localStorage only for now.
- 2026-05-16 (later): The `Colab ↗` and `VS Code` launch pills now carry their own `data-copy-key="<heading>"` attribute. Clicking either pill fires `navigator.clipboard.writeText(heading)` BEFORE the browser navigates the new tab — so when the user lands in the destination and hits `Ctrl+F` + `Ctrl+V` + `Enter`, the exercise heading is already on their clipboard. The 📋 button still exists as a "just copy, don't open" option and is the only element that gets the ✓ flash feedback. Verified Colab URL paths resolve at upstream `callummcdougall/ARENA_3.0` for chapters 0, 1, 2, 4 — repo layout matches our local `content/ARENA_5.0-main/` mirror.
- 2026-05-16: Split `predicted.js` (was YELLOW at 484 LOC) into three siblings: `predicted-links.js` (URL builders), `predicted-data.js` (section/sort/aggregation helpers), `predicted.js` (state + `renderPredictedTable` + expand/copy handlers). Now LIME at 374 LOC. Updated `watch.py` + index.html script load order. Each exercise row in the Predicted course scores table now exposes four actions: `Read ↗` (jupyter-book HTML), `Colab ↗` (whole upstream notebook at `callummcdougall/ARENA_3.0` — kept v3 name as canonical), `VS Code` (`vscode://file/...` opens the local repo copy), and 📋 (copies the exercise heading text so users can Ctrl+F in their destination). The ARENA top-level tab in the SPA was removed the same day — `arena/stage1.js` is still on disk but no longer loaded by `index.html`.
- 2026-04-29: Wired the "Predicted course scores" sub-tab to ARENA data. New `predicted.js` reads `window.ARENA_STAGE1_PROBLEMS`, groups by chapter, subsection, and exercise, and renders the stats-style table. `init.js` now calls `renderPredictedTable()` on stats load and on the predicted sub-tab click.
- 2026-04-29: Added "Predicted course scores" sub-tab as a placeholder panel (button + `data-stats-panel="predicted"` section in `index.html`).
- 2026-04-27: Initial doc created.
