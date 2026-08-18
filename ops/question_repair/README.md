# question_repair

## Purpose
- Repairs the questions learners flag as broken, unclear, or mismatched with their figure — using the `claude` CLI on Seth's own machine, under his own login, with no API key anywhere in the system.
- The domain concern is that a practice bank rots one question at a time. Someone hits a badly-worded drill, flags it, and that flag has to turn into a better question without a person sitting down to a triage queue.

## Owns
- The local half of the repair loop: pulling jobs, starting the Claude Code session, verifying what comes back, and sending it to be applied.
- The sandbox that session runs in.
- The runner's own history, and the viewer that renders it.

## Does NOT own
- **The gates that protect the bank.** Field allowlist, compile check, `answer_code`-only-on-`broken`, identical-rewrite rejection: those are `backend/app/practice/feedback_ai_improver.py::validated_changes`, and the server re-runs them on its own copy of the question no matter what this runner sends.
- The queue itself — `backend/app/feedback_repair_queue.py`.
- The override layer and revision log — `backend/app/feedback_ai_layer.py`.
- Where a repair sits in the question-bank layer stack — see `Local_Deployed_Shared/pipeline/README.md`.

## Key Files
- `run_repairs.py`: the runner. `--watch` to follow the queue, `--once` to drain it, `--question N --tag ... --note ...` to repair one by hand.
- `sandbox_guard.py`: the PreToolUse hook. Deny-by-default; this is what actually constrains a session started with `--dangerously-skip-permissions`.
- `history.py`: `list` / `show` / `transcript` / `queue` / `revisions`. Read this after any run that surprised you.
- `watch.py`: pins the sandbox wiring and the verification step.

## Data & External Dependencies
- The `claude` CLI, on PATH, authenticated as Seth. **No `ANTHROPIC_API_KEY` is read, set, or wanted** — a server-side credential is the thing this design removes.
- The backend package and its venv. `run_repairs.py` re-execs itself into `This-Directory-Only/backend/.venv` because verification needs torch.
- `~/.local/state/delta-drills/question-repair/repair_runs.jsonl` — the runner's history (override with `DELTA_REPAIR_RUNS_DIR`).
- `~/.claude/projects/<repo-slug>/<session-id>.jsonl` — the conversations themselves, stored by Claude Code. `history.py transcript` renders them; `claude --resume <session-id>` reopens one.
- In `--api` mode: an allowlisted user token, from `DELTA_DRILLS_TOKEN`, `--token`, or `~/.config/delta-drills/token`.

## How It Works (Flow)
1. A learner flags a question in the practice UI. The backend logs the feedback and, for an allowlisted account, queues a job. That is all the server does.
2. `run_repairs.py` pulls pending jobs — locally by reading the queue file, or from production with `--api .../api/practice`. Each job carries a snapshot of the question **as that server currently serves it**, so a dev checkout does not repair its own local text.
3. It claims the job and starts one `claude -p` session: read-only tools, the guard hook, `--json-schema` so the answer is a validated object rather than prose.
4. If the session rewrote `answer_code`, the runner re-runs it against the question's real `test_cases` through `app.code_runner`. A failure is fed back into a second attempt; a second failure drops `answer_code` and keeps whatever prose fix survived.
5. What survives is sent to be applied — through `feedback_ai_improver.apply_repair` locally, or `POST /problem-feedback/repair-queue/complete` remotely. The bank reloads in place, no restart and no deploy.
6. The whole run is appended to the history log, including the session id.

## Invariants & Constraints
- **`--dangerously-skip-permissions` and the guard hook travel together.** The flag exists so the session never blocks waiting for a human; the hook is then the only thing standing between it and the machine. `watch.py` fails if one appears without the other.
- **The guard denies by default, and checks paths as well as tool names.** A tool that did not exist when it was written is denied. So is a `Read`/`Grep`/`Glob` aimed outside the repository — a tool-name allowlist alone still leaves `Read` pointed at `~/.ssh`, and this session returns free text that gets logged, so "read-only" without a path check is a way to copy any file this user owns into a repair rationale. The learner note that seeds the prompt is attacker-supplied text, which is the exact shape of input that talks a read-only agent into reading the wrong file. `watch.py` runs the guard for real against both cases.
- **A claim is exclusive.** `feedback_repair_queue.claim` tests and writes under one lock; read-then-write would let two runners take the same job and each overwrite the other's override, the second one silently. Cross-process `flock` covers the rest — the API and this runner write the same files from different processes, where a `threading.Lock` is invisible.
- **The session is read-only, and the runner applies the change.** Never move a gate into the prompt. A gate the model can talk its way past is not a gate.
- **Never trust a rewritten reference answer without running it.** It decides whether every future attempt is graded right or wrong. The check runs **twice on purpose**: here, so a failure can be fed into a second attempt while the session is cheap to redo, and again in `apply_repair`, because `/repair-queue/complete` accepts a rewrite from anything holding an allowlisted token — a runner that skipped verification, or a hand-rolled `curl`.
- **A blank answer is a failure, not a no-op.** An empty `structured_output` means the session could not reply, and the runner raises rather than closing the job quietly.
- Run history stays outside the repo. It is machine-local and unbounded.

## Extension Points
- Repair something other than a flagged question → add a job type in `feedback_repair_queue`, and a branch in `process_job`. Keep the shape: the session proposes, the runner verifies, the server applies.
- Give the session a new capability → add the tool to `ALLOWED_TOOLS` **and** to `--tools`, and say in this README why a read-only session needed it.
- Change what the session is asked → `SYSTEM_PROMPT` / `build_prompt` in `feedback_ai_improver.py`, so the server and the runner cannot drift.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Every repair comes back as "no change"** — `RESOLVED`
  - When it happens: after tightening the sandbox.
  - Symptom: sessions cost real money, reason correctly, and return an empty verdict. Nothing ever reaches the bank, and the loop looks like a model that politely declines every question.
  - Root cause: `StructuredOutput` is a **tool**, and the guard denied it along with everything else. The answer was blocked, not withheld.
  - Prevention/fix: `StructuredOutput` is in `ALLOWED_TOOLS` and pinned by `watch.py`. The runner now raises on an empty answer and names the denied tools in the error, and `history.py show` prints `sandbox denied: ...` per attempt.
  - Status: `RESOLVED`. The general trap stands: any new CLI-internal tool hits deny-by-default first.

- **Verification rejects every torch answer** — `RESOLVED`
  - When it happens: running `run_function_tests` from a standalone script rather than the API process.
  - Symptom: `1/1 test cases failed ... This drill uses PyTorch, which the in-app sandbox can't run. Open it in Colab`. Reads as a wrong answer; is actually a missing import. The bank is 100% torch, so it rejects *everything*.
  - Root cause: the API preloads torch at startup so the fork runner can grade in-process; a script does not.
  - Prevention/fix: `verify_answer` calls `code_runner.preload_torch()` when the answer imports torch, and fails with an explicit "torch is unavailable here" instead of a test-shaped message. Pinned by `watch.py`.
  - Status: `RESOLVED`.

- **A repair applied in production is not in the repo** — `ACTIVE`
  - When it happens: `--api` mode, always.
  - Symptom: a question is repaired and live, the next deploy exports the bank from the CSVs plus the committed layers, and the static `questions.json` the browser loads still has the old text.
  - Root cause: the runtime layer lives on the Fly volume by design; the exporter cannot see it.
  - Prevention/fix: before a deploy that matters, copy `ai_feedback_overrides.jsonl` down off the volume, or promote the repairs into a batch layer. `history.py revisions` lists what would need promoting. See `Local_Deployed_Shared/pipeline/README.md`.
  - Status: `ACTIVE` — inherent to runtime repairs; no automation for it yet.

## Recent Changes
- 2026-08-18 (codex review of the above, same day): seven fixes, all in the "works until two things happen at once, or until someone skips the runner" class. The completion endpoint now re-runs the grading harness itself instead of trusting that the runner did — that was the real hole, since an allowlisted token could post a compiling-but-wrong reference answer straight to the bank. The sandbox gained path containment. `claim()` became a test-and-set under one lock, and every queue write took a cross-process `flock`, because the API and this runner are different processes and a `threading.Lock` between them is decoration. Re-flagging a question no longer deletes a job a runner is mid-session on. A completion that fails in transit no longer takes the `--watch` loop down with it, and keeps the rewrite in the run log so it can be replayed by hand. `--dry-run` stopped claiming jobs, which had been hiding them from the real runner for twenty minutes.
- 2026-08-18: Folder created. The repair loop moved off a server-side Anthropic API key and onto the local `claude` CLI: the backend now only queues, and this runner does the work under Seth's login in a read-only sandbox. Two bugs found while proving it, both recorded above — the denied `StructuredOutput` tool and the missing torch preload — plus a third worth remembering: `.venv/bin/python` is a symlink to the system interpreter, so a re-exec guard that compares resolved executable paths never fires (compare `sys.prefix` instead).
