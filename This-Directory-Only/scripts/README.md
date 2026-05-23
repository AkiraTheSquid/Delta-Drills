# scripts

## Purpose
- Operator entry points. One-shot commands a developer runs from the shell to deploy, sync worktrees, build artifacts, or refresh the question bank.
- Glue layer between the local environment (this `This-Directory-Only/` tree) and the shared payload that gets shipped to Vercel/Fly (`Local_Deployed_Shared/`).

## Owns
- The full deploy pipeline: build → rsync to deploy worktree → push deploy → Vercel deploy → Fly deploy. Lives in `deploy_delta_drills.sh`.
- Worktree sync helpers (`sync-deploy.sh`, `sync-local.sh`) — fast-forward merges between `main` and `deploy` worktrees.
- The Jupyter Book build step (`build_arena_book.sh`) and its venv lifecycle.
- Thin Python launchers that bootstrap `sys.path` and delegate to the real implementations under `Local_Deployed_Shared/pipeline/`.
- The split-layout invariant check (`refresh_split_layout.py`) that guards the repo-root structure (only `Local_Deployed_Shared/`, `This-Directory-Only/`, and a small allowlist of metadata files at root).
- The Delta Drills browser isolation watcher (`watch_delta_drills_dev.py`) that keeps the Chrome debug session pinned to the local app and closes stray tabs from unrelated profiles.

## Does NOT own
- Pipeline business logic (data extraction, validation, prompt rewriting). That lives in `Local_Deployed_Shared/pipeline/`. Scripts here just `runpy.run_path(...)` into it.
- Backend deployment internals (Dockerfile, fly.toml). Owned by `This-Directory-Only/backend/` + `This-Directory-Only/fly.toml`. Deploy script just calls `flyctl deploy`.
- Vercel project configuration. Owned by `Local_Deployed_Shared/vercel.json` + the Vercel project linked at `Local_Deployed_Shared/.vercel/`.
- Jupyter Book content/config. Owned by `arena-book/` at repo root.

## Key Files
- `deploy_delta_drills.sh`: main one-shot deploy. Auto-commits dirty main, exports question bank, builds Book, rsyncs shared payload to deploy worktree, pushes deploy (Vercel auto-redeploys), then deploys backend to Fly.
- `deploy_delta_drills_local.sh`: thin alias that just `exec`s `deploy_delta_drills.sh`. Kept because the shortcut name is referenced from `/usr/local/bin/deploy_delta_drills`.
- `build_arena_book.sh`: builds the Jupyter Book. Creates a venv at `arena-book/.venv` if missing, runs `jupyter-book build .`, stages output at `Local_Deployed_Shared/arena-book/` (gitignored on main, tracked on deploy).
- `sync-deploy.sh` / `sync-local.sh`: fast-forward merge between worktrees. Used outside the deploy flow when you want to align branches without doing a full deploy.
- `refresh_split_layout.py`: verifies no unexpected files exist at repo root and prunes stale root-level symlinks. Run by `deploy_delta_drills.sh` against both worktrees.
- `extract_arena_prereqs.py`: extracts structured exercises (starter code, solutions, image markers) from the ARENA prereqs notebook into `Local_Deployed_Shared/arena_prereqs_structured.json`.
- `export_questions_json.py`, `build_function_bank.py`, `build_function_mode_requests.py`, `build_function_mode_repair_requests.py`, `validate_function_bank.py`, `test_function_validator.py`: thin `runpy` wrappers that set `sys.path` to include `Local_Deployed_Shared/pipeline/` and `Local_Deployed_Shared/`, then run the matching script in `Local_Deployed_Shared/pipeline/`.
- `watch_delta_drills_dev.py`: long-running helper launched by `delta_drills_dev`. Polls Chrome's remote-debugging endpoint and closes any tab whose URL is not Delta Drills. This is the guardrail against Delta Note or any other app leaking into the MCP browser session.
- `EXTRACTION-PIPELINE.md`: design notes for the structured-exercise extraction flow.
- `STORAGE-ARCHITECTURE.txt`: notes on where artifacts land on disk.

## Data & External Dependencies
- Worktrees: `Delta-Drills-Local` (main), `Delta-Drills-Deployed` (deploy). Both expected at fixed paths under `~/Applications/`.
- External CLIs: `git`, `rsync`, `vercel`, `flyctl` (`~/.fly/bin/flyctl`), `supabase`, `python3`. Deploy script tolerates missing CLIs by skipping the relevant step with a warning.
- Vercel scope: `seth-gibsons-projects`, project `delta-drills`, alias `https://delta-drills.vercel.app`.
- Fly app: `delta-drills-backend`, deployed from `This-Directory-Only/Dockerfile` + `This-Directory-Only/fly.toml`.
- Pipeline modules: `Local_Deployed_Shared/pipeline/*.py` — these are the real implementations behind every `build_*` / `validate_*` / `export_*` wrapper here.

## How It Works (Flow)

**Main deploy flow** (`deploy_delta_drills.sh`):
1. Auto-commit any dirty state on `main` (excluding logs).
2. Run question-bank exporters: `export_questions_json.py`, `extract_arena_prereqs.py`, `refresh_split_layout.py`.
3. Push `main` to origin.
4. Best-effort `supabase db push` + functions deploy.
5. Build Jupyter Book via `build_arena_book.sh` → output in `Local_Deployed_Shared/arena-book/`.
6. `rsync -a --delete --exclude '.vercel/' Local_Deployed_Shared/ → deploy worktree's Local_Deployed_Shared/`. Auto-commit deploy worktree.
7. Push `deploy` to origin (triggers Vercel rebuild via Git integration).
8. Vercel CLI does an explicit `vercel deploy --prod` from `DEPLOY_DIR/Local_Deployed_Shared/`. Verify the alias serves 200; force-redeploy once if not.
9. Deploy backend via `flyctl deploy` against `This-Directory-Only/Dockerfile`.

**Pipeline runner flow** (`build_function_bank.py` etc.):
1. Resolve `Local_Deployed_Shared/` from the script's location.
2. Insert `Local_Deployed_Shared/pipeline/` and `Local_Deployed_Shared/` into `sys.path`.
3. `runpy.run_path` the matching module under `pipeline/` with `run_name="__main__"`.

## Invariants & Constraints
- **Never edit pipeline logic in this folder.** The Python wrappers here must stay thin runpy bootstraps. Real work lives in `Local_Deployed_Shared/pipeline/`. If you find yourself adding logic here, you're in the wrong file.
- **Never push from inside the deploy script with `--force` to `main`.** The script uses plain `git push`. If the push fails, fix locally — don't paper over.
- **`refresh_split_layout.py`'s allowlist must stay aligned with the actual root layout.** Adding a new top-level file at the repo root (outside the allowlist) will make this script raise on every deploy. Update `ALLOWED_ROOT_NAMES` only when intentionally adding a new permitted root entry.
- **The deploy script auto-commits dirty state.** Never start a deploy with WIP you don't want shipped — it will get committed and pushed. Inspect `git status` first, or stash.
- **Build script must not write anywhere outside `arena-book/_build/` and `Local_Deployed_Shared/arena-book/`.** It removes the staging dir with `rm -rf` before copying — it must never run with the wrong target path.

## Extension Points
- New pipeline step: add the real script to `Local_Deployed_Shared/pipeline/`, then add a thin runpy wrapper here mirroring the existing pattern in `build_function_bank.py`.
- New deploy target: add a `--- Step N: ...` block to `deploy_delta_drills.sh` between existing steps. Wrap external CLI calls in `command -v` guards so missing tooling degrades gracefully.
- New build artifact (similar to arena-book): write a sibling shell script that builds + stages into `Local_Deployed_Shared/<name>/`, then call it from `deploy_delta_drills.sh` before the rsync step.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Auto-commit absorbs WIP** — `ACTIVE`
  - When it happens: running `deploy_delta_drills.sh` with uncommitted experimental changes on main.
  - Symptom: a `chore: auto-commit before deploy` commit ships your WIP to production.
  - Root cause: the `auto_commit_if_dirty` helper bulk-stages anything not under `This-Directory-Only/logs/`.
  - Prevention/fix: `git status` (or stash) before deploying. If WIP must coexist, gate it behind a flag in the actual code, not by leaving it uncommitted.
  - Status: `ACTIVE`.

- **`refresh_split_layout.py` raises on unexpected root files** — `ACTIVE`
  - When it happens: dropping a stray file (notes, scratch script, IDE artifact) at the repo root.
  - Symptom: deploy aborts with `Unexpected root-level path outside split layout: ...`.
  - Root cause: by design — the split-layout invariant exists to keep root tidy.
  - Prevention/fix: put the file under `This-Directory-Only/` (local-only) or `Local_Deployed_Shared/` (ships). Only update the `ALLOWED_ROOT_NAMES` set if the file legitimately belongs at root.
  - Status: `ACTIVE`.

- **MCP Chrome picks up unrelated tabs** — `RESOLVED`
  - When it happened: `delta_drills_dev` reused a Chrome profile that already had another app open, so the remote-debug session surfaced Delta Note / other tabs instead of only Delta Drills.
  - Symptom: `chrome-devtools-mcp` landed on the wrong app or showed stale signed-in state from another workspace.
  - Root cause: profile/session bleed, not Delta Drills app code.
  - Prevention/fix: `delta_drills_dev` now uses a dedicated profile under `/tmp/delta_drills_dev_chrome_profile` by default and launches `watch_delta_drills_dev.py` to close any non-Delta-Drills tabs seen on port `9222`.
  - Status: `RESOLVED` (2026-05-13).

## Recent Changes
- 2026-04-30: Added `build_arena_book.sh` and a `Step 3c: Build the ARENA Jupyter Book` block to `deploy_delta_drills.sh`. Output staged to `Local_Deployed_Shared/arena-book/` (gitignored on main, tracked on deploy).
- 2026-04-27: Initial doc created.
