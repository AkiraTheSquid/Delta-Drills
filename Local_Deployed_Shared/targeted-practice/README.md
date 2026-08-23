# targeted-practice

## Purpose
- Lets the student search ARENA exercises by heading, queue a subset, and start a "targeted practice session" surfaced as a global banner. After the session ends, animates per-exercise readiness deltas so the student can see which queued exercises they're now ready for.

## Owns
- The "Targeted Practice" tab — search card, selected list with readiness bars, Submit → start-session, Back-to-search reset.
- The global `#tp-banner` indicator (lives outside `<main>` in `index.html` but is owned by this module's controller).
- The session lifecycle: idle → search → session-active (banner up, Practice tab in front) → review (animated before→after deltas) → idle.
- The synthetic "after" readiness model in `simulateAfter()` — placeholder until a real backend endpoint grades a targeted practice session.

## Does NOT own
- The actual practice question rendering / grading — that's the regular Practice tab (`practice/`).
- The base readiness compute — `window.computeArenaReadiness` lives in `arena/manifest.js`.
- The ARENA exercise catalog — `window.ARENA_EXERCISES_BY_NOTEBOOK` lives in `arena/exercises.js`.
- Colab/Jupyter Book/VS Code URL builders — `stats/predicted-links.js#colabUpstreamHref`.
- The `switchTab(...)` function — defined in `app.js`; this module just calls it.

## Key Files
- `targeted-practice.js`: controller (IIFE). Builds the search catalog by flattening `ARENA_EXERCISES_BY_NOTEBOOK` + the `ARENA_PREREQS_TEMP_EXERCISES` fallback for 0.0. Tracks `selected` (Map<id, exercise>), `beforeReadiness` (snapshotted at session start), `afterReadiness` (synthesized at review enter). Wires Submit/End/Back buttons and drives the bar animation.
- `targeted-practice.css`: card chrome, search input, results list, selected-list rows with embedded readiness bar, review-mode per-item action row (Colab link + Ready badge), and the global `.tp-banner` strip.

## Data & External Dependencies
- Reads `window.ARENA_EXERCISES_BY_NOTEBOOK` (arena/exercises.js).
- Reads `window.ARENA_PREREQS_TEMP_EXERCISES` + `ARENA_PREREQS_TEMP_NOTEBOOK_PATH` (stats/predicted-prereqs-temp.js) as a fallback when the registry has `[]` for a notebook.
- Reads `window.ARENA_STAGE1_PROBLEMS` (arena/manifest.js) to map notebook → problem so each selected item can show a readiness bar.
- Calls `window.computeArenaReadiness(skillWeights, fallback)` for per-problem readiness.
- Calls `colabUpstreamHref(notebookPath)` from `stats/predicted-links.js` for per-item Colab links (respects the GitHub-username override on the Account page).
- Calls `switchTab("practice" | "targeted-practice")` from `app.js` to drive the tab swap.
- Re-uses CSS classes `.stats-bar` / `.stats-bar-track` / `.stats-bar-fill` (`styles/stats.css`) and `.target-difficulty-delta` / `.target-difficulty-marker` (`practice/bars.css`) for the bar animation — same visual vocabulary as the `arena-unlock` interstitial.

## How It Works (Flow)
1. **Search**: student types ≥ 2 chars → `renderResults()` filters the flattened catalog (case-insensitive substring on title) and shows up to 40 hits with highlighted matches and a checkbox.
2. **Select**: clicking a row or its checkbox calls `toggleSelect(ex)` which mutates the `selected` Map and surgically syncs the affected row's class + checkbox without re-rendering the whole list (keeps scroll stable). Each selected item shows its current readiness as a blue `.stats-bar`.
3. **Submit (start session)**: `startSession()` snapshots `beforeReadiness` per selected exercise, fills the banner meta with the count, reveals `#tp-banner`, and calls `switchTab("practice")`. The student now drills in the regular Practice tab while the banner persists above all pages.
4. **End targeted practice (banner button)**: `endSession()` hides the banner, calls `switchTab("targeted-practice")`, and runs `enterReviewMode()` — which synthesizes `afterReadiness` via `simulateAfter(before)`, re-renders the selected list with action buttons (Colab link, Ready badge if ≥ 70), and animates each bar from `before% → after%` with the green/red delta highlight (cascaded 120ms apart).
5. **Back to search**: `resetToSearch()` clears all state (selected, before/after, banner, review-mode class), restores the search card + Submit button, and refocuses the search input.

## Invariants & Constraints
- Script load order: `targeted-practice.js` must load AFTER `arena/manifest.js`, `arena/exercises.js`, `stats/predicted-links.js`, `stats/predicted-prereqs-temp.js`, and `app.js`. Currently the last script in `index.html` after all of these.
- The global `.tp-banner` lives OUTSIDE `<main>` in `index.html` (right after `</header>`) so it persists across tab switches.
- `.tp-card { display: flex }` would otherwise override the global `.hidden { display: none }` because `targeted-practice.css` loads after `styles/components.css`. The explicit `.tp-card.hidden, .tp-submit-btn.hidden, .tp-back-btn.hidden, .tp-banner.hidden { display: none }` rule re-asserts hiding — DO NOT remove it.
- `switchTab` is a top-level `const` in `app.js`, NOT `window.switchTab`. Reference it directly as `switchTab(...)` from this module, not `window.switchTab(...)`.
- `simulateAfter()` is a placeholder for the real backend grading. Replace its body (not its signature) when the queue endpoint lands so the before→after animation reflects actual progress.
- `beforeReadiness` is captured at `startSession()` time, not at `enterReviewMode()` time, so the delta reflects the full session even if the adaptive cache has shifted during practice.

## Extension Points
- **Real grading**: replace `simulateAfter(before)` in `targeted-practice.js` with a call that fetches the per-exercise post-session readiness from the backend. Keep the same return shape (0–100 number).
- **Wire the Practice tab to actually drill the queued exercises**: today the queue is informational — the regular Practice tab still serves its normal adaptive question stream. To make the practice tab drill the queue, the practice runner would need a "targeted queue" mode that pulls from a global the targeted-practice module exposes; add a `window.TargetedPractice.getQueue()` getter and have `practice/api.js:getNextQuestion()` consume it when `sessionActive` is true. (This used to name `practice/runner.js`, which no longer exists and never chose questions anyway — the queue lives in `api.js`.)
- **More search fields**: extend `buildCatalog()` to include section/anchor in the search needle, or add a notebook filter dropdown above the search input.
- **Different Ready threshold**: change `READY_THRESHOLD` at the top of `targeted-practice.js`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **`.hidden` override silently broken by card display** — `RESOLVED`
  - When it happens: adding new card-style classes in `targeted-practice.css` with `display: flex`.
  - Symptom: `classList.add("hidden")` does nothing visually; element stays on screen.
  - Root cause: this CSS file loads after `styles/components.css`, so equal-specificity `.tp-card { display: flex }` wins over `.hidden { display: none }`.
  - Prevention/fix: any new `.tp-*` class that sets `display` must be added to the `.tp-*.hidden { display: none }` rule near the top of `targeted-practice.css`.
  - Status: `RESOLVED` — initial review-mode swap broke this way before being patched.

## Recent Changes
- 2026-08-22: `startSession()` dispatches `delta:xp` (`kind: "targeted_session"`) for the topbar XP pill. The drills themselves pay through `PracticeAPI` (see the hook block at the bottom of `../practice/api.js`); choosing a set and committing to it is the one piece of entered data on this tab that never reaches that chokepoint. Dispatched rather than calling `window.DeltaXP` directly so this file needs no load-order relationship with `../xp.js`.
- 2026-08-07: `targeted-practice-dom.js` tags the search label and the selected-list title with `data-dd-info`, so `../infotips.js` hangs a ⓘ on each — this page is injected at runtime, which is why infotips re-derives its dots from a MutationObserver instead of a one-shot pass at load. `watch.py`'s auth-only invariant now matches on class *tokens* rather than the literal `class="tab auth-only"` string: the tab picked up a `has-info` class and a purely additive change failed the assertion. It also now covers the tab's sibling ⓘ, which carries the same `data-tab` and must stay `.auth-only` or it is left hanging beside a hidden tab.
- 2026-05-20: Initial module — Targeted Practice tab with search, selected list with per-item readiness bars, Submit → review-mode in place with synthetic before→after animation, Back-to-search reset.
- 2026-05-20: Reworked submit flow — Submit now starts a session (banner + tab swap to Practice). The before→after animation moved behind a new End-targeted-practice button on the banner. Banner lives outside `<main>` as a global indicator.
