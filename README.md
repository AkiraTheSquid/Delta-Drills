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
python -m http.server 5173
```

3) Open the UI:

- http://localhost:5173/

In the UI Account tab, set API base to:

- http://localhost:8000

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
