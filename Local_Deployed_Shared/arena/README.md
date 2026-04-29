# arena

## Purpose
ARENA Stage-1 problem registry for Delta Drills. Owns the canonical list of ARENA curriculum sections (chapters 0–3, 24 entries) and the UI that renders them on the ARENA tab. Other tabs (notably the Predicted course scores stats sub-tab) read this same registry as their source of truth — so when the curriculum coverage or per-section data changes, only this folder needs to change.

## Owns
- The full curriculum spec: every ARENA section across chapters 0–3, with chapter label, section label, title, notebook path, lesson page path, placeholder readiness score, prerequisite tags, and weighted skill profile.
- The hand-crafted overrides for the original 4 entries (`RICH_SECTIONS` in `manifest.js`) — preserved verbatim so existing curated data isn't lost when the registry is re-synthesized.
- The `window.ARENA_STAGE1_PROBLEMS` global — the runtime contract every consumer depends on.
- The ARENA tab UI on `page-arena`: chapter filter, problem list sidebar, detail panel with summary, prerequisite chips, weighted skill cards, launch action links, and metadata grid.

## Does NOT own
- The Predicted course scores stats sub-tab UI — that lives in `stats/predicted.js` and only *consumes* `window.ARENA_STAGE1_PROBLEMS`.
- The ARENA notebook content itself — those `.ipynb` files live under `content/ARENA_4.0-main/` after the recent root restructure.
- Per-user readiness scoring math — every `readinessScore` here is a placeholder until Delta Drills can compute user-specific scores.
- ARENA tab styling — see `styles/arena.css`.

## Key Files
- `manifest.js`: defines `ARENA_CURRICULUM`, `ARENA_CHAPTER_DEFAULTS`, `RICH_SECTIONS`, and the `buildArenaProblem` builder that assembles each entry. Exposes `window.ARENA_STAGE1_PROBLEMS`.
- `stage1.js`: ARENA tab UI. Reads `window.ARENA_STAGE1_PROBLEMS`, builds the chapter filter, problem list, and detail panel. Top-level IIFE — runs once on script load.

## Data & External Dependencies
- DOM: `#arena-problem-list`, `#arena-problem-detail`, `#arena-chapter-filter`, `#arena-problem-count` (all in `index.html` under `page-arena`).
- Globals: writes `window.ARENA_STAGE1_PROBLEMS` (manifest); reads it (stage1, plus `stats/predicted.js`).
- Notebook/lesson hrefs are relative deploy paths beginning with `ARENA_4.0-main/...`. After the root restructure, the actual files live under `content/ARENA_4.0-main/...` — the ARENA tab's external launch links may need rewriting at deploy time or a path-rewrite layer; the data shape itself is stable.

## How It Works (Flow)
1. `index.html` loads `arena/manifest.js` which registers `window.ARENA_STAGE1_PROBLEMS` (24 entries built from `ARENA_CURRICULUM` × `buildArenaProblem`).
2. `index.html` loads `arena/stage1.js`. Its IIFE checks for the four ARENA tab DOM nodes, populates the chapter filter from unique `chapterId`/`chapterLabel` pairs, and renders the problem list + detail panel for the first problem.
3. User clicks a problem card → `renderDetail` rewrites the detail HTML with summary, prereq chips, skill cards, launch links, and metadata.
4. Independently, `stats/predicted.js` reads `window.ARENA_STAGE1_PROBLEMS` lazily when the Predicted course scores sub-tab is rendered, groups by chapter, and renders the stats-style table.

## Invariants & Constraints
- `window.ARENA_STAGE1_PROBLEMS` is the public contract. Every entry MUST have `id`, `chapterId`, `chapterLabel`, `sectionLabel`, `title`, `summary`, `readinessScore`, `readinessLabel`, `readinessNote`, `prerequisiteTags`, `skillWeights`, `lessonPath`, `notebookPath`, `backupNotebookPath`, `launchPath`, `executionMode`. Removing or renaming any field breaks the ARENA detail panel and/or `stats/predicted.js`.
- `id` values must be unique across the array — `stage1.js` uses them as selection keys.
- Every `chapter` in `ARENA_CURRICULUM` must have a matching entry in `ARENA_CHAPTER_DEFAULTS`, otherwise `buildArenaProblem` crashes on `def.label`.
- `manifest.js` must be loaded BEFORE any consumer that reads `window.ARENA_STAGE1_PROBLEMS` synchronously. Lazy consumers (`stats/predicted.js`) are safe regardless.
- Hand-crafted entries in `RICH_SECTIONS` should be preserved across rebuilds — they represent curated content, not regenerable data.

## Extension Points
- **Add a new ARENA section**: append an entry to `ARENA_CURRICULUM` with `id`, `chapter`, `section`, `title`, `notebookPath`, `lessonPath`, and a `placeholderScore`. Both the ARENA tab and the Predicted course scores tab pick it up automatically.
- **Promote a synthesized entry to hand-crafted**: add the entry's `id` to `RICH_SECTIONS` with custom `summary`, `readinessScore`, `prerequisiteTags`, `skillWeights`, etc. Anything you specify there overrides the chapter default.
- **Add a new chapter**: register it in `ARENA_CHAPTER_DEFAULTS` (label + prereqs + skill weights), then add curriculum entries with that `chapter` key.
- **Replace placeholder readiness with real per-user scores**: swap `placeholderScore` and the `RICH_SECTIONS` `readinessScore` overrides with values pulled from Delta Drills skill state. Keep `labelForScore` or replace it with a richer mapping.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- **Notebook/lesson paths broken after root restructure** — `ACTIVE`
  - When it happens: ARENA folders moved from `Local_Deployed_Shared/ARENA_4.0-main/` to `Local_Deployed_Shared/content/ARENA_4.0-main/` in commit `c821f86`, but `manifest.js` paths still start with `ARENA_4.0-main/...`.
  - Symptom: clicking "Open exercise notebook file" or "Open lesson markdown" in the ARENA detail panel 404s when served from the deploy.
  - Root cause: the manifest still uses pre-restructure relative paths.
  - Prevention/fix: either (a) prefix paths with `content/` in the curriculum spec, (b) add a Vercel rewrite from `/ARENA_4.0-main/*` → `/content/ARENA_4.0-main/*`, or (c) add a build step that copies ARENA assets back to deploy root. Choice depends on `.vercelignore` strategy.
  - Status: `ACTIVE`. Predicted course scores tab is unaffected — it doesn't use the paths.

- **Missing chapter default crashes builder** — `RESOLVED` (guard not added; relying on convention)
  - When it happens: someone adds a curriculum entry with a new `chapter` key without registering it in `ARENA_CHAPTER_DEFAULTS`.
  - Symptom: `TypeError: Cannot read properties of undefined (reading 'label')` at script load — both ARENA tab and Predicted tab go silent.
  - Prevention/fix: `watch.py` checks every `chapter` in `ARENA_CURRICULUM` has a matching default, and reports the gap before runtime.
  - Status: `RESOLVED` via `watch.py` invariant.

## Recent Changes
- 2026-04-29: Expanded `manifest.js` from 4 entries to the full ARENA curriculum (24 sections across chapters 0–3). Replaced the literal-array form with `ARENA_CURRICULUM` × `buildArenaProblem`. Hand-crafted data for the original 4 entries preserved in `RICH_SECTIONS`. Both the ARENA tab and the Predicted course scores stats sub-tab now show the full curriculum.
- 2026-04-27: Initial doc created.
