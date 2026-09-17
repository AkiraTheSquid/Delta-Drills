# Spec — in-app execution for ARENA chapters 0–3 on E2B

Status: DRAFT, not authorized. Written 2026-07-31.

Sequenced **after** the prerequisite work. Nothing here is started.

Supersedes the execution-backend question left open in
`instructions/spec-in-app-execution-and-colab-routing.md` (SUPERSEDED) and
discharges the "ask which of three options" instruction recorded in the
2026-05-16 Jupyter Book decision.

---

## 1. What is being bought

The estimate for building this from scratch was 4–6 weeks. It decomposes into
four pieces, and a vendor only sells two of them:

| # | Piece | Solved by |
|---|---|---|
| 1 | Stateful per-learner interpreter | E2B |
| 2 | Rich output — matplotlib, HTML, streams | E2B |
| 3 | Cell UI in the browser | us (`practice/notebook.js` exists) |
| 4 | 458 notebooks as content | `arena-book` already |

### Why E2B

- **`create_code_context()` / `run_code(code, context=ctx)`.** A headless Jupyter
  server runs inside each sandbox; variables and imports persist across calls, and
  several independent contexts can coexist in one sandbox. This is the single
  largest line item in the from-scratch estimate and it is a library call.
- **`pause()` / `connect(id)`.** Pausing saves filesystem *and memory*; paused
  sandboxes are kept indefinitely with no TTL; resume is ~1 s and restores running
  processes and loaded variables. Pause costs ~4 s per GiB of RAM.

  That is the decisive property. The Modal design was expensive because a session
  billed idle time, forcing an aggressive idle timeout that threw away the
  learner's kernel. Here a learner can close the tab mid-exercise, and come back to
  a live kernel without paying for the gap.
- ~$0.10/hr at 2 vCPU. $100 one-time credit, 20 concurrent sandboxes, 1-hour max
  session on Hobby; Pro is $150/mo for 24-hour sessions and 100 concurrent.

### What it costs to choose this

- **CPU-only, permanently.** Chapter 4 (`gemma-2-27b-it`, `Qwen2.5-14B`,
  `Llama-3.1-8B`) can never move here. It stays on the learner's Colab with the
  learner's API key — which is already the decision, so this forecloses nothing
  that was planned. Modal was the only candidate that could ever absorb it.
- **A 4th vendor**, breaking the previous Vercel + Neon + Fly rule. Accepted.
- Hobby's 1-hour session cap is shorter than an ARENA exercise sitting. Verify
  whether pause/resume resets that clock before assuming Hobby is enough for
  pilot users.

---

## 2. Backend

New, roughly 600–800 lines. Lives beside the existing runner — it does **not**
replace it. `code_runner.py`'s fork runner keeps grading the 499 bank drills at
milliseconds and $0; E2B is only for ARENA chapter exercises.

### 2.1 Session store

```
exec_sessions
  user_id       FK
  sandbox_id    text        -- E2B's id, survives pause
  context_id    text        -- per-notebook code context
  notebook_id   text
  state         enum(running, paused, dead)
  last_used_at  timestamptz
  created_at    timestamptz
```

One row per learner per notebook. `context_id` is per-notebook because two ARENA
parts share no variables and giving them one namespace invites collisions that
present as impossible bugs.

### 2.2 Routes

- `POST /api/exec/session` — create-or-resume for `{notebook_id}`. Returns
  `{session_id, context_id, resumed: bool}`. Auth by the existing JWT; **never
  accept a `sandbox_id` from the client** — it is a capability, and a learner who
  can name another learner's sandbox gets their kernel.
- `POST /api/exec/run` — `{session_id, code}` → `{stdout, stderr, results[], error}`.
- `POST /api/exec/interrupt` — a runaway training loop must be stoppable without
  destroying kernel state.
- `DELETE /api/exec/session` — explicit "reset my kernel", mapping to
  `restart_code_context`.

### 2.3 Idle reaper

Where the money is. A periodic job pauses any `running` session past N minutes
idle. Start at N = 10 and measure.

⚠️ Pausing costs ~4 s per GiB, so a reaper that pauses and a learner who returns
immediately is worse than no reaper. Rate-limit pause/resume per session.

### 2.4 Sandbox template

A Dockerfile built once, not pip-installed per session — if setup happens at
session start the cost model is wrong by an order of magnitude.

- `torch` from the CPU index (mirror the existing `This-Directory-Only/Dockerfile`,
  which installs CPU torch *before* `requirements.txt` deliberately)
- `transformer_lens`, `einops`, `jaxtyping`, `datasets`
- the ARENA exercise package
- **`gpt2-small` weights baked into the image.** All 88 of chapter 1's model loads
  are `gpt2-small`; downloading it per session is the difference between a cheap
  tier and an expensive one.

---

## 3. Frontend

`Local_Deployed_Shared/practice/notebook.js` (369 lines, restored 2026-07-31) is
the starting point, not a rewrite. Its own header documents the compromise to
remove:

> "Neither runtime keeps a session between calls… Running a cell therefore
> executes every cell above it as well… The alternative — a persistent
> per-learner interpreter — is a server-side session to build, expire and
> secure, which is not worth it."

That judgement was correct when the alternative was building sessions by hand.
With `context_id` the prefix re-run goes away: each cell posts only its own
source.

Also needed:
- matplotlib PNGs and HTML reprs. `practice/visuals.js` already renders arrays to
  canvas; images are new.
- stdout streaming, or at minimum a running indicator — an ARENA training cell
  can run for minutes and a dead panel reads as a hang.

### Not using thebe

Thebe would supply the cell UI, but it speaks **Jupyter's websocket protocol**
while E2B's SDK is server-side HTTP. Bridging means either a websocket shim or
running `jupyter-server` inside the sandbox and exposing its port —
**and whether E2B can expose a port with websocket support was not verified.**
Since most of the UI already exists on disk, the REST path avoids the risk for
little extra work. Revisit only if the cell UI turns out to be the hard part.

---

## 4. Content

`arena-book` already holds the pages. What is undecided is how an ARENA exercise
enters the practice queue at all — the 499 bank questions are
`submission_mode: "function"` with test cases, and ARENA exercises are neither.
Until that representation exists there is nothing for the queue to route to, which
is the real reason this is sequenced after the prerequisite work.

---

## 5. Sequence

| # | Item | Effort |
|---|---|---|
| 1 | Sandbox template + verify `gpt2-small` loads from the image | 2–3 d |
| 2 | Session store, routes, auth | 3–4 d |
| 3 | Idle reaper + pause/resume, measured | 2 d |
| 4 | `notebook.js` against `context_id` | 3–4 d |
| 5 | Image/HTML output rendering | 2–3 d |
| 6 | ARENA exercise representation in the queue | unknown — depends on §4 |

Items 1–5: **two to three weeks**, against 4–6 from scratch. Item 6 is the
unbounded one and is a content-model question, not an execution one.

---

## 6. Open questions

- Does pause/resume reset Hobby's 1-hour session cap?
- N for the idle reaper — start at 10 minutes and measure.
- Does E2B expose ports with websockets? Only matters if §3 revisits thebe.
- §2.2's rule that the client never supplies a `sandbox_id` is the security
  boundary of this whole design. `/critic` before those routes ship.
