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
- `torchify_einops_einsum.py`, `torchify_np_drills.py` (+ a per-group `torchify_*_manual.py` data table), `torchify_lessons.py`, `torchify_np_pages.py`, `torchify_np_prose.py`, `verify_torch_dialect_layer.py`: the NumPy→PyTorch dialect conversion, one lesson group at a time. Each generator emits an override layer into `This-Directory-Only/chatgpt/` (gitignored, so `git add -f`) that must be registered in BOTH `pipeline/export_questions_json.py` and `backend/app/questions.py` — `pipeline/watch.py` fails if the two lists diverge. They read a `questions_base.json` snapshot of the PRE-conversion bank rather than the live one: pointed at `questions.json` they read their own output. Nothing about an expected value is authored — every one is executed, and each question is additionally cross-checked by running the numpy answer and the torch answer over identical inputs.
- `torchify_np_drills.py` is the bank half and is now **parameterized rather than forked per lesson**. A conversion group is a manual data module plus a command line: `--lessons np-4` (registry lesson ids), `--ids 405 406 …` (explicit ids, for questions no lesson tags), `--manual torchify_np4_manual` (the module beside this script holding the hand-written translations) and `--out torch_dialect_overrides_np4.jsonl` (the layer filename, written into `This-Directory-Only/chatgpt/`). Those become a `Config` dataclass that the shared translation rules read; the rules themselves do not change between groups.
- `torchify_np23_manual.py`, `torchify_np4_manual.py`, `torchify_parked_manual.py`: the `--manual` data modules. Each exports up to six tables, all optional and all read by name off the module — **data, never machinery**: `MANUAL` (id → hand-written torch `solve`, for drills whose numpy function has no torch spelling at all: `ogrid`, `nditer`, `apply_along_axis`, `argpartition`, `intersect1d`, `add.reduceat`, `r_`, ufunc `out=`/`where=`); `EXCLUDE` (id → why this drill CANNOT cross dialects, refused loudly and reported at the end of the run instead of being quietly emitted wrong); `NO_CROSSCHECK` (id → why numpy and torch cannot be run over the same inputs here, e.g. a numpy `Generator` and a `torch.Generator` draw different sequences by design, or float32-vs-float64 rounding at extreme magnitudes — the expected values are still produced by execution, only the second opinion is unavailable); `SHADOW_RENAMES` (id → `(old, new)` for a variable named `t`, which shadows `import torch as t` and makes the drill unrunnable); `CALL_PATCHES` (id → replacements inside a test case's `call` when the ASSERTION itself is numpy-only, such as `np.shares_memory(r, z)` → a storage-pointer comparison — these are the point of their drill, so they are re-spelled rather than dropped, and a patch that matches nothing raises, because a stale patch means the numpy-only assertion silently survived); `TEXT_PATCHES` (id → replacements in the prompt, for a question whose wording names a numpy API — nothing executes the prompt, so a drill telling the learner to reach for `np.nditer` fails silently, in the learner's head). Supported but so far unused by any group.
- `torchify_np_pages.py` / `torchify_np_prose.py`: the lesson half of the same pass, also `--lessons`-driven and reading their page set from the KC registry rather than a hardcoded list, so a KC that moves between lessons cannot leave a page behind in the old dialect. Pages converts fenced code only; prose renames only the symbols torch spells identically and prints everything else as a hand-review list, because prose makes claims — about dtypes, about views vs copies, about what numpy does differently — that a regex has no business rewriting. `--skip-kc` leaves a named KC in NumPy when converting its page alone would split it from drills that have no translation.
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
- **Nothing in a dialect conversion is trusted on sight.** Two rules, and neither is negotiable when adding a group. First, no expected value is ever authored: every `expected_output` and every `expected_expr` in an emitted layer is produced by EXECUTING the translated answer. Second, every translation is cross-checked by running the ORIGINAL numpy answer and the TRANSLATED torch answer over identical inputs and requiring agreement — that is what makes a hand-written rewrite safe, because a rewrite that quietly changed the meaning disagrees with numpy and the question is REFUSED rather than emitted. A question that genuinely cannot be cross-checked goes in `NO_CROSSCHECK` with a written reason, so the exemption is visible in the diff instead of being a silent skip.
- **Never edit pipeline logic in this folder.** The Python wrappers here must stay thin runpy bootstraps. Real work lives in `Local_Deployed_Shared/pipeline/`. If you find yourself adding logic here, you're in the wrong file.
- **Never push from inside the deploy script with `--force` to `main`.** The script uses plain `git push`. If the push fails, fix locally — don't paper over.
- **`refresh_split_layout.py`'s allowlist must stay aligned with the actual root layout.** Adding a new top-level file at the repo root (outside the allowlist) will make this script raise on every deploy. Update `ALLOWED_ROOT_NAMES` only when intentionally adding a new permitted root entry.
- **The deploy script auto-commits dirty state.** Never start a deploy with WIP you don't want shipped — it will get committed and pushed. Inspect `git status` first, or stash.
- **Build script must not write anywhere outside `arena-book/_build/` and `Local_Deployed_Shared/arena-book/`.** It removes the staging dir with `rm -rf` before copying — it must never run with the wrong target path.

## Extension Points
- New dialect-conversion group: do NOT copy `torchify_np_drills.py`. Snapshot the pre-conversion bank (`git show <sha>:Local_Deployed_Shared/questions.json > questions_base.json`), write a `torchify_<group>_manual.py` next to it exporting whichever of the six tables the group needs, then run the existing script with `--lessons`/`--ids`, `--manual` and `--out`. Register the emitted layer in both override-layer lists, `git add -f` it out of the gitignored `chatgpt/` directory, and verify it through the real grader with `verify_torch_dialect_layer.py`.
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
- 2026-07-28: **The three np-2/np-3 generators were renamed by `git mv`, because they are no longer np-2/np-3 specific**: `torchify_np23.py` → `torchify_np_drills.py`, `torchify_lessons_np23.py` → `torchify_np_pages.py`, `torchify_prose_np23.py` → `torchify_np_prose.py`. The drills script gained a `Config` dataclass and a `--lessons/--ids/--manual/--out` command line, so converting a group is now "write a data table and pass some flags" rather than "fork the script and edit the constants at the top" — the previous shape would have meant a fourth and a fifth copy of 800 lines of translation rules diverging from each other. `--ids` exists because the last group belongs to no lesson at all, so there are no KC tags to select it by. The tables a group may export are `MANUAL`, `EXCLUDE`, `NO_CROSSCHECK`, `SHADOW_RENAMES`, `CALL_PATCHES` and `TEXT_PATCHES` (documented under Key Files); all are optional, read off the module by name, and hold data only.
- 2026-07-28: Two more conversion passes finished the bank, which is now 448/448 PyTorch. `torchify_np4_manual.py` did np-4 "Applied patterns" — 45 of its 51 drills, the other 6 being the whole `numpy.structured-dtypes` KC, which is in `EXCLUDE` and was then retired rather than converted (a tensor is homogeneous, so record dtypes, `datetime64` and `genfromtxt` have nothing to become). `torchify_parked_manual.py` did the 17 CNN/backprop drills that no lesson tags yet, reached with `--ids`; most of them only ever carried a vestigial `import numpy as np` they never called, and the mechanical pass simply dropped it. New traps from these two: the drills that take a random generator cannot be cross-checked at all, because a numpy `Generator` and a `torch.Generator` draw different sequences by design; the log-sum-exp drill deliberately runs logits near 1e3, where numpy's float64 and torch's float32 differ by rounding alone and the cross-check reads that as a disagreement; and a `CALL_PATCH` that matches nothing now raises, because a stale patch silently leaves a numpy-only assertion (`np.shares_memory`, `isinstance(r, np.ndarray)`) grading a torch answer.
- 2026-07-28: Added the np-2/np-3 dialect conversion (`torchify_np23.py`, since renamed to `torchify_np_drills.py`, + `torchify_np23_manual.py`; `torchify_lessons_np23.py` → `torchify_np_pages.py`; `torchify_prose_np23.py` → `torchify_np_prose.py`). Roughly a quarter of those drills use a numpy function torch has no spelling for (`ogrid`, `nditer`, `apply_along_axis`, `argpartition`, `intersect1d`, ufunc `out=`/`where=`), so they are hand-translated in the `_manual` table and verified by the same numpy-vs-torch cross-check as the regex ones. Two traps worth remembering: a call expression that names `solve(...)` more than once runs it more than once, which silently double-applies an in-place drill; and the hand-written answers must keep the original demo block, or `stdout_prediction` questions export an empty stdout and the exporter keeps the stale NumPy-formatted `expected_output`.
- 2026-04-30: Added `build_arena_book.sh` and a `Step 3c: Build the ARENA Jupyter Book` block to `deploy_delta_drills.sh`. Output staged to `Local_Deployed_Shared/arena-book/` (gitignored on main, tracked on deploy).
- 2026-04-27: Initial doc created.
