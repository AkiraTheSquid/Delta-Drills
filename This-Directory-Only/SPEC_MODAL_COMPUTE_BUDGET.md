# SPEC — Modal compute budget for the ARENA chapters (pass 1)

Status: SIGNED OFF 2026-09-30 — D1–D5 confirmed by Seth as written (refuse-new at cap; his account capped too).

## Goal

Let Seth (plus 1–3 other learners) work through ARENA chapters 0–4 in the web app
on Modal kernels **without the Modal bill exceeding $30/month**, and know at any
moment what a learner-hour actually costs.

Context: September's bill was $56.21 for chapters 0.0–0.1 alone. The cause, every
sandbox running its full 4 h lifetime, was fixed in e54b3734 (idle reaper, 0.25 core
/ 1 GiB requests). This spec covers what comes next.

## The full arc (only pass 1 is specced)

| Pass | What | Why |
|---|---|---|
| **1** | ch1 preload + spend meter + hard cap + daily alert | cheap, certain wins; measure before spending on GPU |
| 2 | compute class per section (cpu / gpu-small / gpu-big / api) + T4 sandboxes with a per-learner GPU-minutes cap; prefer ARENA's own gpt2-small routes in 1.3.3 / 1.4.2 | ~8 of 31 sections need a GPU; $30 buys ~40 T4-hours total |
| 3 | snapshot-on-idle spike (`Sandbox._experimental_snapshot`) | keep a learner's variables across breaks without paying idle time |
| 4 | per-learner daily kernel-minute caps | needed once there are more than ~3 learners |

Model swapping is **not** a plan item. Swapping a model is safe where the model is
only a tool (training loops, RL, eval harnesses). It breaks the lesson where the
model is the subject: IOI heads exist only in gpt2-small, SAEs/transcoders load
only on their own base model, the fine-tuned checkpoints exist for one base, and
probe or steering effects need scale. ARENA's tests also hard-code
original-model outputs. Pass 2 uses ARENA's existing small routes instead.

## Scope — pass 1

### A. Preload chapter 1 into the kernel image
- Add `transformer_lens==2.17.0`, CircuitsVis (git, `python/` subdir), and
  `eindex-callum` (already present) to the image in `modal_kernel._ensure_app`.
  ARENA's setup cell runs `%pip install` only inside `try: import transformer_lens
  / except:`, so the install skips itself, with zero ARENA edits (same trick as the
  `/root/<chapter>` checkout).
- Bake the `gpt2-small` weights and tokenizer into the image's HF cache with a
  build step that calls `HookedTransformer.from_pretrained("gpt2-small")`. This
  covers 1.1, 1.2, 1.3.3 (gpt2 half), 1.4.1, 1.4.2 (gpt2 half) and 2.4.
- Check what the 1.3.3 / 1.4.2 / 2.4 setup cells import (e.g. `sae_lens`) and add
  only the packages their CPU route needs. Gemma stays out (pass 2).

### B. Spend meter
- New `app/modal_spend.py`, which keeps `modal_kernel.py` (YELLOW, 485 LOC) from
  growing:
  - **Usage log:** `/data/kernel_usage.jsonl` gets one line per sandbox when it
    closes: `{user, session_id, context, started, ended, seconds}`. Written from
    `ModalSession.shutdown()` via a single call.
  - **Month-to-date $:** taken from the Modal billing API (the
    `Workspace.billing.report` the CLI wraps, using the same `MODAL_TOKEN_*` Fly
    already has) and cached for 1 h. The API reports full hours only, so the
    current partial hour is estimated as live sandboxes × the requested rate.
  - **$/kernel-hour:** MTD $ ÷ Σ seconds in the usage log ÷ 3600.
- `/kernel/status` gains
  `spend: {mtd_usd, cap_usd, kernel_hours_mtd, usd_per_kernel_hour, capped}`.

### C. Hard monthly ceiling
- `DD_MODAL_CAP_USD` (default **25**: $30 budget minus a buffer for billing lag).
- At or above the cap, `_reserve_and_run` refuses to **spawn** a new sandbox with
  a learner-facing sentence ("Notebook compute is paused for the month — …").
  Behaviour for sessions already running: see D3.
- Backstop outside the app: if the Modal dashboard has a workspace spend limit,
  Seth sets it to $30 by hand. (Not verified that Modal offers this; check during
  implementation.)

### D. Daily spend alert
- `ops/modal-spend/dd_modal_spend_check.py`, run by a systemd user timer on Greed
  (same pattern as `dn-egress-check.timer`).
- Reads the billing API. Alerts when **yesterday > $1.50** or **the MTD pace
  projects > $30**. The alert is a phone push via `notify_user` plus a
  SessionStart alert file, like the Delta Note egress alert, and it clears on the
  next run that comes back ok.

## Not this pass
GPU sandboxes and compute-class tags (pass 2) · snapshot-on-idle (pass 3) ·
per-learner caps (pass 4) · Gemma / Llama / Qwen weights · any ARENA notebook edit ·
model swapping · the Colab route (abandoned 09-13, stays abandoned).

## Success criteria
1. **Ch1 fast:** on a fresh prod kernel, the 1.1 setup cell runs no `%pip`, and
   `HookedTransformer.from_pretrained("gpt2-small")` returns in **< 10 s** with
   `HF_HUB_OFFLINE=1` set (proves nothing downloads).
2. **Meter:** `/kernel/status` returns the `spend` block. `usd_per_kernel_hour`
   is within ±25% of (Modal bill ÷ logged kernel-hours) computed by hand for the
   same day.
3. **Ceiling:** locally, with `DD_MODAL_CAP_USD=0.01`, a new kernel is refused
   with the sentence and an already-running kernel behaves per D3. With the cap
   unset, 0.0 and 0.1 behave exactly as today.
4. **Alert:** with the threshold forced to $0, one timer run sends the phone
   push and writes the alert file; the next run with the real threshold clears it.
5. **No regressions:** the idle reaper still closes sessions (e54b3734 test); the
   image rebuild does not break 0.0 / 0.1 cells. Codex `/critic` before deploy.

## Key decisions (signed off 2026-09-30)
- **D1 Where the weights live:** baked into the image *(pick)* vs a Modal Volume.
  The image is simpler and already content-hashed; a Volume only pays off for
  multi-GB models (pass 2).
- **D2 Source of the cap's number:** Modal billing API plus a live-hour estimate
  *(pick)* vs only the app's own usage log × rates. The API is the truth but lags
  about an hour; the estimate fills the gap.
- **D3 At the cap:** refuse new kernels and let running ones end through the idle
  reaper *(pick)* vs also terminate running ones at 110% of the cap.
- **D4 Seth's account:** capped like everyone else *(pick: the budget is the
  budget)* vs exempt (alert only).
- **D5 Where the alert runs:** Greed systemd timer *(pick: reuses the egress
  pattern)* vs a Fly cron. If Greed is off there is no alert, which is why the
  dashboard limit is the backstop.

## Touch list
- `This-Directory-Only/backend/app/modal_kernel.py`: image additions,
  one `modal_spend` call in `shutdown()`, cap check in `_reserve_and_run`
  (net ≈ +15 LOC).
- `This-Directory-Only/backend/app/modal_spend.py`: **new** (usage log, MTD
  cache, cap).
- `This-Directory-Only/backend/app/practice/kernel_router.py`: `spend` in status.
- `ops/modal-spend/dd_modal_spend_check.py` + `README.md`: **new**; systemd unit
  and timer in `~/.config/systemd/user/` (machine config, not in the repo).
- `This-Directory-Only/backend/app/README.md`: Lifetimes / spend paragraph +
  Recent Changes.
