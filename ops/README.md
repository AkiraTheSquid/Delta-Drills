# ops

## Purpose
- Operator tooling: things Seth runs on his own machine against Delta Drills, rather than things the app runs for a learner.
- The domain concern is the maintenance loop — work that needs his credentials, his terminal, and his judgement, and therefore cannot live on a server.

## Owns
- The local half of any loop whose other half is a queue in the backend.
- Scripts that hold a human in the loop by design: they print what they are about to do, they are safe to Ctrl-C, and they keep a history someone can read afterwards.

## Does NOT own
- Anything the deployed app depends on. Nothing under `This-Directory-Only/` or `Local_Deployed_Shared/` may import from `ops/` — the Docker image does not ship it, so such an import is a production `ImportError`. The dependency runs one way: `ops/` imports the backend.
- Build and deploy steps — those are `scripts/`, `Local_Deployed_Shared/pipeline/`, and `/usr/local/bin/deploy_delta_drills`.
- Question authoring and batch rewrites — `Local_Deployed_Shared/pipeline/`.

## Key Files
- `question_repair/`: the local Claude Code runner that repairs questions learners have flagged. See its own README.

## Data & External Dependencies
- The backend package (`This-Directory-Only/backend/app`), imported directly — these scripts run against the same code the server runs.
- The backend venv (`This-Directory-Only/backend/.venv`), which is the only interpreter here with torch.
- The `claude` CLI, authenticated as Seth. No API key is used or stored anywhere in this tree.
- Machine-local state under `~/.local/state/delta-drills/`, deliberately outside the repo.

## How It Works (Flow)
1. Something in the app records work that needs a human or a local credential — currently, a flagged question becomes a job in `app/feedback_repair_queue.py`.
2. A script here picks the job up, does the part that needs the local machine, and writes the outcome back through the backend's own code path so the server-side gates still apply.
3. Whatever happened is appended to a history log that a companion viewer renders.

## Invariants & Constraints
- **`ops/` is not deployed.** It is not in the Docker build context that ships, and the backend must never import it.
- **Local state stays out of the repo.** Run logs go to `~/.local/state/delta-drills/`, overridable per-script. A data directory inside the tree collects READMEs and health checks that have nothing to describe, and grows without bound in `git status`.
- **Nothing here holds a model API key.** The point of this folder is that the model runs under Seth's own login; a script that reads `ANTHROPIC_API_KEY` belongs somewhere else, and `app/practice/watch.py` fails if that credential path reappears in the backend.
- Scripts that need torch must run under the backend venv. They should re-exec themselves into it rather than failing halfway.

## Extension Points
- New operator loop → new subfolder here, plus a queue module in the backend if the app is what notices the work.
- The pattern to copy is `question_repair/`: a queue in the backend, a runner here, a history viewer beside it, and gates that live on the server rather than in the prompt.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **A venv re-exec that never happens** — `RESOLVED`
  - When it happens: a script tries to put itself on the backend interpreter by comparing `sys.executable` to `.venv/bin/python`.
  - Symptom: `ModuleNotFoundError: No module named 'fastapi'` (or torch) from a script that contains code to prevent exactly that.
  - Root cause: `.venv/bin/python` is a **symlink to the system interpreter**, so `Path(sys.executable).resolve()` is identical from either one and the guard always says "already there".
  - Prevention/fix: compare `sys.prefix` against the venv root, which is what actually differs. `question_repair/run_repairs.py::ensure_backend_python` is the reference.
  - Status: `RESOLVED` where it has been hit; the trap is available to every new script here.

## Recent Changes
- 2026-08-18: Folder created for `question_repair/`, the local question-repair runner. The rule worth carrying forward is the one-way dependency: `ops/` may import the backend, the backend may never import `ops/`, because this tree is not in the deployed image.
