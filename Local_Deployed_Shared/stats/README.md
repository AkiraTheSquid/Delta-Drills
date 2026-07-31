# stats

## Purpose
Leftover shared helpers from the deleted Statistics tab. This folder **no longer renders a page**. The Statistics tab was removed 2026-07-31; three files survived because other tabs read their globals, and they kept the `stats/` path so their consumers' `<script src=...>` tags stay valid. Treat this folder as a compatibility shim, not a feature.

## Owns
- Topic/subtopic **weights**: the `delta_drills_weights` localStorage contract, its normalization, and the backend weight sync (`weights.js`).
- **Colab URL resolution** for ARENA and Delta-Drills notebooks, including the per-user GitHub-fork override (`predicted-links.js`).
- The **temporary ARENA prereq scaffold** — hardcoded per-exercise prereq map + the unlock predicate the arena-unlock interstitial gates on (`predicted-prereqs-temp.js`).

## Does NOT own
- Any page or tab. There is no `#page-statistics` and no `.tab[data-tab='statistics']` anymore — `../watch.py` asserts both stay gone.
- The `.stats-bar-*` CSS in `styles/stats.css`. Those classes outlived this folder and are rendered by `index.html` (Practice page bars), `practice/arena-unlock.js`, and `targeted-practice/targeted-practice.js`.
- The adaptive engine that consumes the weights — see `practice/adaptive.js` and `practice_engine.py`.

## Key Files
- `weights.js`: load/save user weights and topic-enabled flags; localStorage key `delta_drills_weights`. **Consumers:** `practice/adaptive.js` (`buildEffectiveWeightsFromSubtopics`), `practice/questions.js` (`isSubtopicEnabled`).
- `predicted-links.js`: `colabUpstreamHref(notebookPath)` → the Colab URL for an ARENA (`callummcdougall/ARENA_3.0`) or Delta-Drills notebook, pointed at the student's fork when Account → GitHub username is set. **Consumers:** `courses.js`, `courses-fork-gate.js`, `practice/ui.js`, `practice/drills-catalog.js`, `practice/arena-unlock.js`, `targeted-practice/targeted-practice.js`.
- `predicted-prereqs-temp.js`: **TEMPORARY SCAFFOLD — delete when the real concept graph ships.** Exposes `window.ARENA_PREREQS_TEMP_ENABLED` (kill-switch), `ARENA_PREREQS_TEMP_EXERCISES` (the 26 ARENA 0.0 exercises, since the auto-extracted registry is `[]` for that notebook), `ARENA_PREREQS_TEMP_BY_EXERCISE` (per-exercise hardcoded `[{topic, subtopic, minPct}]` against Delta Drills topics Numpy/Einops/Einsum), `getArenaPrereqSubtopicScore(topic, subtopic)` (reads the adaptive baseline), and the unlock helpers `isArenaExerciseUnlocked` / `getNextUnshownUnlockedArenaExercise` / `markArenaExerciseShown` / `getArenaPrereqsForExercise`. **Consumers:** `practice/arena-unlock.js`, `practice/drills-catalog.js`, `targeted-practice/targeted-practice.js`.

## Data & External Dependencies
- LocalStorage: `delta_drills_weights` (topic + subtopic weights, enabled flags); `account_github_username` (fork override for Colab URLs).
- Backend: weight sync via `weights.js#pushWeightsToBackend`. The `/api/practice/subtopics` read that fed the old Areas table is no longer called from here; the endpoint still exists in `This-Directory-Only/backend/app/practice/subtopic_router.py`.
- No DOM contract. Nothing in this folder queries or injects elements.

## How It Works (Flow)
1. `index.html` loads `weights.js` → `predicted-links.js` → `predicted-prereqs-temp.js` as plain scripts (not ES modules); each defines top-level bindings other scripts call later.
2. `practice/adaptive.js` and `practice/questions.js` call into `weights.js` when building the adaptive queue.
3. `practice/arena-unlock.js` calls `isArenaExerciseUnlocked` / `getArenaPrereqsForExercise` to decide which ARENA exercise to surface, and `colabUpstreamHref` to build its launch link.
4. `courses.js` (via `courses-fork-gate.js`) resolves every Colab link through `colabUpstreamHref`.

## Invariants & Constraints
- Never write to `delta_drills_weights` directly — go through `weights.js` so `normalizeWeights` runs.
- Load order: these files must be evaluated **before** `courses.js` and `targeted-practice/targeted-practice.js`, which read their globals. `../watch.py` asserts both orderings.
- Do not re-add a Statistics page here. The renderers (`dom.js`, `data.js`, `render.js`, `graph.js`, `predicted-data.js`, `predicted.js`, `init.js`, `stats-dom.js`) were deleted, and `watch.py` fails if any of them reappears on disk or in `index.html`.
- `predicted-prereqs-temp.js` is still deletable in one step once the real concept graph ships — but it now has three live consumers, so port `isArenaExerciseUnlocked` and friends before removing it.

## Extension Points
- **Real concept graph lands**: replace `predicted-prereqs-temp.js` with graph-backed equivalents of `getArenaPrereqSubtopicScore` / `isArenaExerciseUnlocked`, update the three consumers, then drop the `<script>` tag and this file's entry in `watch.py`.
- **Retiring this folder**: the honest end-state is moving `weights.js` under `practice/` and `predicted-links.js` under `courses/` (or a new `links/`), then deleting `stats/`. That is a rename across ~8 consumers plus both `watch.py` files — worth doing, but as its own change.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- **Folder name lies about its contents** — `ACTIVE`
  - When it happens: anyone grepping for statistics/analytics code lands here and assumes a Statistics feature still exists.
  - Symptom: wasted search time; risk of "cleaning up" a file three other tabs depend on.
  - Root cause: the path was kept to avoid a rename across every consumer when the tab was removed.
  - Prevention/fix: read this Purpose section before touching anything here; run `mod graph <file>` to see the real fan-in. See the Extension Points note on retiring the folder.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-07-31 (Statistics tab removed): Deleted the tab and everything that only served it — `stats-dom.js`, `dom.js`, `data.js`, `render.js`, `graph.js`, `predicted-data.js`, `predicted.js`, `init.js`, plus `../stats.js` and the tab button in `index.html`. Reason: the priority / learning-rate / predicted-course-score tables leaned on assumptions the learner had no way to check, so they added cognitive load without adding signal. `predicted-links.js` lost its table-only helpers (`bookHrefForNotebook`, `vsCodeHrefFor`, `openLinkCell`, `VSCODE_LOCAL_ABS_ROOT`); `styles/stats.css` was trimmed from 542 lines to just the `.stats-bar-*` rules other tabs still render. Backend endpoints were left untouched. `watch.py` here and in `..` were rewritten to guard the removal instead of the old sub-tab/panel contract.
- 2026-05-18 (un-scope: restore full curriculum): Reverted the chapter filter — the predicted-scores table rendered ALL chapters again. The temp prereq panel only attached to exercise rows whose title appeared in `ARENA_PREREQS_TEMP_BY_EXERCISE` (the 26 0.0 Prerequisites exercises). *(Table since deleted; the prereq map itself is still live via arena-unlock.)*
- 2026-05-18 (temp prereq scaffold for ARENA 0.0): Added `predicted-prereqs-temp.js` to shake out the predicted-scores pipeline against a tiny ARENA slice before the concept-graph backend landed. Per-exercise prereq map is hand-bucketed by canonical-solution category (`rearrange` / `repeat` / `reduce` / `indexing`). The full upstream concept graph lives in `Local_Deployed_Shared/arena_prereqs_structured.json`.
- 2026-05-16 (URL-encoding fix): Switched the URL builders in `predicted-links.js` from `encodeURI` to per-segment `encodeURIComponent` (`encodePathSegments`). `encodeURI` leaves `&`, `?`, `#` untouched, so notebooks with `&` in their filename (`0.2_CNNs_&_ResNets_exercises.ipynb`, …) produced URLs Colab's parser treated as query-bearing and 404'd on. Still load-bearing for every Colab link in the app.
- 2026-05-16: `Colab ↗` links respect a per-user GitHub-fork override — Account field `account-github-username` (localStorage `account_github_username`) tells `arenaColabOwner()` / `drillsColabOwner()` whose fork to point at, falling back to upstream. Combined with Colab's File → Save a copy in GitHub, the student's fork is the storage layer.
- 2026-05-16: Split the then-YELLOW `predicted.js` into `predicted-links.js` (URL builders), `predicted-data.js` (section/sort/aggregation helpers) and `predicted.js` (render + handlers). Only `predicted-links.js` survives.
- 2026-04-29: Wired the "Predicted course scores" sub-tab to ARENA data. *(Removed 2026-07-31.)*
- 2026-04-27: Initial doc created.
