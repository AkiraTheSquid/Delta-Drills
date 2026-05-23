# Delta Drills

This repo is set up with two git worktrees:

- Local dev: `/home/stellar-thread/Applications/Delta-Drills-Local` (branch: `main`)
- Production deploy: `/home/stellar-thread/Applications/Delta-Drills-Deployed` (branch: `deploy`)

The Vercel production branch is `deploy` and the public URL is:

- https://delta-drills.vercel.app

## Local development

1) Backend (FastAPI):

```
cd /home/stellar-thread/Applications/Delta-Drills-Local/backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2) Frontend (static):

```
cd /home/stellar-thread/Applications/Delta-Drills-Local
python3 serve.py 5173
```

3) Open the UI:

- http://localhost:5173/

The API base defaults to `http://localhost:8000` automatically when running on localhost — no manual configuration needed.

## Practice (adaptive learning)

- The Practice tab is available when logged in (auth token present).
- Practice uses the backend algorithm via `/api/practice/*`.
- Progress is persisted per user on the backend at `backend/user_data/`.

## Notes + fixes added

- Root UI (`/home/stellar-thread/Applications/Delta-Drills-Local/index.html`) now includes the Practice tab/page and loads `frontend/practice.css`, `frontend/practice.js`, and Pyodide. Previously the root UI did not include the Practice page, so the tab never appeared when serving from the repo root.
- Practice is auth-gated in the root UI switch logic, matching the rest of the app.
- `python3 -m http.server` is required on this system (plain `python` may not exist).
- 2026-04-29: added auth-gated `Courses` tab (`data-tab="courses"` → `#page-courses`) between Course and Papers. Page hosts two mutually-exclusive views inside the container — `#courses-list-view` (search input + filtered results) and `#courses-detail-view` (article view rendered on click). `courses.js` owns navigation: clicking a course card swaps to the detail view, the in-article "Back to courses" button swaps back. Seeded with one course — ARENA Curriculum — including a hero block, intro paragraph, and per-chapter cards (Fundamentals, Transformer Interpretability, RL, LLM Evaluations, Alignment Science, Capstone) each with its illustration sourced from `images.squarespace-cdn.com`. Styling lives at `styles/courses.css`.
- 2026-04-29: added per-course "Include course for study?" Yes/No toggle. Renders on the right side of each list card and at the top of the detail article (above the hero). State persists in `localStorage` under `delta_drills_courses_include` as `{ courseId: "yes" | "no" }`. List and detail controls mirror each other via `syncIncludeControls()`; clicks on the toggle do not propagate to the card-open handler.
- 2026-04-29: courses detail — chapter cards with a `sections` array become clickable and open a pure-information modal listing the chapter's sections (number badge + title + description). Backdrop/×/Esc all close. Per-chapter accent color is passed via inline `--section-number-color` on the modal root. Currently only Chapter 1 (Transformer Interpretability) has section data populated; other chapters render as static rows until their data lands. `openChapterModal()` / `closeChapterModal()` live in `courses.js`; styles in `styles/courses/`.
- 2026-04-29: courses CSS reorganized — the single `styles/courses.css` was split into five focused fragments under `styles/courses/` (`list.css`, `include.css`, `detail.css`, `modal.css`, `responsive.css`) so each file stays small. `index.html` now links all five between `styles/arena.css` and `styles/responsive.css`. Per-fragment selector ownership is enforced by `styles/courses/watch.py`.
- 2026-04-29: chapter-sections modal data filled out for the remaining ARENA chapters. Each chapter now has its own per-chapter accent color sourced from learn.arena.education: Ch0 Fundamentals `#DC2626`, Ch1 Interpretability `#D97706`, Ch2 RL `#059669`, Ch3 Evals `#2563EB`, Ch4 Alignment Science `#4F46E5`. Capstone Project still has no `sections` array (it's an open-ended project, not section-based) and renders as a static row. `watch.py` invariants now lock one representative section title plus the hex color per chapter.
- 2026-04-29: chapter-modal visual pass — modal background switched to `--bg` (less saturated than `--card`), section-row bubbles removed (rows blend into the modal and pick up `--surface` only on hover), section number lost its bordered rectangle, scrollbar themed to match the dark UI, and overall row density tightened.

## UI layout note

- The root UI is now the primary interface (served at `/home/stellar-thread/Applications/Delta-Drills-Local/index.html`).
- The previous `frontend/` UI has been moved into `/home/stellar-thread/Applications/Delta-Drills-Local-Backups/ui-legacy-20260215-110811/frontend/` as a snapshot.

## Troubleshooting Practice tab

- Make sure you are serving `http://localhost:5173/` from `/home/stellar-thread/Applications/Delta-Drills-Local`.
- If the tab still does not show, hard refresh and confirm you are logged in.
- For the adaptive algorithm, the backend must be running on port 8000.

## Production deploy workflow

Make changes locally in the `main` worktree, then sync to the deploy worktree:

```
cd /home/stellar-thread/Applications/Delta-Drills-Local
./scripts/sync-deploy.sh
```

Review and push production:

```
cd /home/stellar-thread/Applications/Delta-Drills-Deployed
git status
# commit any deploy-only changes if needed
# then push deploy

git push origin deploy
```

## Sync back from deploy to main

If you made deploy-only changes and want to bring them back to local:

```
cd /home/stellar-thread/Applications/Delta-Drills-Local
./scripts/sync-local.sh
```

## Notes

- The backend is not deployed on Vercel. Only the static frontend is.
- `http://localhost:8000/` returns 404 by design. Use `/health` for checks.
- Keep deploy-only tweaks in the deploy worktree to avoid polluting local dev.
