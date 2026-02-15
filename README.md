# Delta Drills

This repo is set up with two git worktrees:

- Local dev: `/home/stellar-thread/Applications/pdf-split-tool` (branch: `main`)
- Production deploy: `/home/stellar-thread/Applications/pdf-split-tool-deployed` (branch: `deploy`)

The Vercel production branch is `deploy` and the public URL is:

- https://delta-drills.vercel.app

## Local development

1) Backend (FastAPI):

```
cd /home/stellar-thread/Applications/pdf-split-tool/backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2) Frontend (static):

```
cd /home/stellar-thread/Applications/pdf-split-tool
python3 -m http.server 5173
```

3) Open the UI:

- http://localhost:5173/

In the UI Account tab, set API base to:

- http://localhost:8000

## Practice (adaptive learning)

- The Practice tab is available when logged in (auth token present).
- Practice uses the backend algorithm via `/api/practice/*`.
- Progress is persisted per user on the backend at `backend/user_data/`.

## Notes + fixes added

- Root UI (`/home/stellar-thread/Applications/pdf-split-tool/index.html`) now includes the Practice tab/page and loads `frontend/practice.css`, `frontend/practice.js`, and Pyodide. Previously the root UI did not include the Practice page, so the tab never appeared when serving from the repo root.
- Practice is auth-gated in the root UI switch logic, matching the rest of the app.
- `python3 -m http.server` is required on this system (plain `python` may not exist).

## UI layout note

- The root UI is now the primary interface (served at `/home/stellar-thread/Applications/pdf-split-tool/index.html`).
- The previous `frontend/` UI has been moved into `backups/ui-legacy-20260215-110811/frontend/` as a snapshot.

## Troubleshooting Practice tab

- Make sure you are serving `http://localhost:5173/` from `/home/stellar-thread/Applications/pdf-split-tool`.
- If the tab still does not show, hard refresh and confirm you are logged in.
- For the adaptive algorithm, the backend must be running and the API base must be set to `http://localhost:8000`.

## Production deploy workflow

Make changes locally in the `main` worktree, then sync to the deploy worktree:

```
cd /home/stellar-thread/Applications/pdf-split-tool
./scripts/sync-deploy.sh
```

Review and push production:

```
cd /home/stellar-thread/Applications/pdf-split-tool-deployed
git status
# commit any deploy-only changes if needed
# then push deploy

git push origin deploy
```

## Sync back from deploy to main

If you made deploy-only changes and want to bring them back to local:

```
cd /home/stellar-thread/Applications/pdf-split-tool
./scripts/sync-local.sh
```

## Notes

- The backend is not deployed on Vercel. Only the static frontend is.
- `http://localhost:8000/` returns 404 by design. Use `/health` for checks.
- Keep deploy-only tweaks in the deploy worktree to avoid polluting local dev.
