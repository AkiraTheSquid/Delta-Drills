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

---

# PASS 2 — every section runs, or says where it can't (GPU sections)

Status: SIGNED OFF 2026-10-01 — P1 L4, P2 AUTOMATIC at the section's GPU cell (Seth: "start the gpu when it's needed and disable it when the user is not using it"), P3 5 h, P4 Volume with Seth's token, P5 banner only.

## Goal

Every ARENA section opens on a kernel that can run it, or says on open what it
needs that the app does not give. GPU time stays inside the same $30/month.

## What the sections need (survey of ARENA 527f937)

| Class | Sections | What decides it |
|---|---|---|
| **cpu** (19) | 0.0–0.5, 1.1, 1.2, 1.4.1, 1.5.1–1.5.4, 2.1, 2.2.1, 2.2.2, 2.3, 2.4, 2.5 | toy models or gpt2-small; 2.4 takes the `LOW_GPU_MEM` gpt2-small route |
| **cpu + API key** (7) | 3.1–3.5, 4.2, 4.5 | OpenRouter / OpenAI clients, no local model |
| **gpu** (6): fits one 24 GB L4 | 1.3.1 (Llama-2-13b 8-bit, Llama-3.1-8B), 1.3.2 (gpt2-xl from cell 12; gpt-j-6b only via NDIF), 1.3.3 (gemma-2-2b from cell 115 of 226), 1.3.4 (Qwen3-8B, Llama-3.1-8B), 1.4.2 (gemma from cell 78 of 199), 4.3 (R1-Distill 1.5B / 8B) | 2–13 B params in bf16/8-bit |
| **gpu-big** (2) | 4.1 (Qwen2.5-14B + LoRA, ~30 GB), 4.4 (gemma-2-27b / Qwen3-32B; a Qwen2.5-7B small route exists) | ≥ 14 B params, needs L40S/A100 |

Prices (modal.com/pricing, 2026-10-01). Sandbox GPUs bill at standard rates:
T4 $0.59/h, L4 $0.80/h, A10 $1.10/h, L40S $1.95/h, A100-80 $2.50/h. An L4 sandbox
at 0.5 core / 4 GiB is ≈ **$0.97/h**. Late September ran about $1/day of CPU kernels,
which leaves roughly **$15–20/month ≈ 15–20 L4-hours shared by everyone**.

## Known bugs this pass fixes (no new spend)
- **ch2 never installs gymnasium.** 2.1–2.3 guard the whole `%pip` line on
  `import jaxtyping`, which the image already has. Bake that line's packages:
  `gymnasium[atari,accept-rom-license,other]==0.29.0`, `pygame`. mujoco-py (the
  2.3 bonus only) stays out.
- **A hung billing call stops the reaper.** `modal_spend.refresh()` runs inline in
  the reaper loop (relayed codex flag). A stalled `billing.summary()` would keep
  idle sandboxes billing. Move the refresh onto its own thread.

## Scope — pass 2
- **A.** The two fixes above.
- **B. Compute class per section.** `app/kernel_classes.py` maps an ARENA section
  number to `cpu | api | gpu | gpu-big`. Opening a non-cpu section shows a
  one-line banner saying what it needs: "needs an API key: set
  `os.environ[...]`", "from cell N this section runs on a GPU (H h left this month)", or "needs a GPU larger
  than the app offers".
- **C. GPU kernels on gpu sections, automatic.** Each gpu section carries
  `gpu_from_cell`: the first cell that loads the big model (1.3.3 → 115,
  1.4.2 → 78, 1.3.2 → 12, the others near their top).
  - Running a cell at or past that index moves the session onto an L4 sandbox. The
    kernel restarts and setup is restored, exactly as after an idle restart. A
    banner says so and shows the GPU hours left.
  - Cells before it stay on CPU. There is no hopping back per cell: the model and
    every variable live in the GPU kernel's memory.
  - Idle 5 min → the GPU sandbox closes (CPU stays 15). The next cell spawns a new
    one, and the model reloads from the Volume.
  - The GPU image is a CUDA variant (same packages, CUDA torch). Gated weights
    (gemma, Llama, Qwen3-8B, R1-8B) are prefetched once into a Modal Volume with
    Seth's HF token, so learners never see a token.
  - At most 2 GPU sandboxes at once. Per learner, 5 GPU-hours a month, counted
    from the usage log (`gpu` field). The $25 workspace cap still applies.
- **D.** `/kernel/status` → `spend.gpu_hours_mtd` for the learner, plus the remaining GPU hours.

## Not this pass
gpu-big hardware (L40S/A100), snapshot-on-idle (pass 3), per-learner CPU caps
(pass 4), API-key storage per learner, NDIF key handling, ch3 package baking
(its `inspect_ai` guard installs at runtime and works; baking it conflicts with
ch1's `openai==1.56.1` pin).

## Success criteria
1. 2.1's setup cell runs no `%pip`, and `gym.make("CartPole-v1")` plus an atari
   env construct on a fresh kernel.
2. With `billing.summary` patched to sleep 300 s, the reaper still closes an idle
   sandbox inside `IDLE_SECONDS + REAP_INTERVAL`.
3. Opening 1.3.3 shows the gpu banner. 0.1 shows none. 3.1 shows the api banner.
4. On the test app, a GPU kernel loads gemma-2-2b from the Volume with
   `HF_HUB_OFFLINE=1`, and `torch.cuda.is_available()` is True. Leaving it idle
   for 5 min closes it. A learner at the GPU cap is refused with a sentence;
   their CPU kernel still works.
5. No regressions: 0.0 / 0.1 / 1.1 cells as in pass 1. Codex `/critic` before deploy.

## Key decisions (signed off 2026-10-01)
- **P1 GPU tier:** L4 24 GB *(pick: covers all 6 gpu sections, native bf16)* vs
  T4 16 GB ($0.59; no bf16, fails the 8B models in 1.3.1 / 1.3.4 / 4.3).
- **P2 How a GPU kernel starts:** AUTOMATIC once a cell at or past `gpu_from_cell`
  runs (Seth) vs a Switch-to-GPU button vs GPU from the moment the section opens.
- **P3 Per-learner GPU cap:** 5 h/month *(pick)* vs 3 h vs 10 h.
- **P4 Gated weights:** prefetched to a Volume with Seth's HF token *(pick)* vs
  each learner pastes their own HF token.
- **P5 gpu-big (4.1, 4.4):** banner only, pointing to the 7B route in 4.4 *(pick)*
  vs offer an L40S with a 2 h cap.

## Touch list
- `backend/app/modal_kernel.py` (already 645 LOC): image packages (ch2), `gpu=`
  passthrough only. The GPU image, Volume and cap go in a new
  `backend/app/modal_gpu.py` so modal_kernel does not grow.
- `backend/app/modal_spend.py`: refresh thread; `gpu` field and GPU-hours per user.
- `backend/app/kernel_classes.py`: **new**.
- `backend/app/practice/kernel_router.py`: `gpu` flag on run/reset; class in status.
- `Local_Deployed_Shared/practice/arena-notebook.js` (+ the css): banner; GPU flag on
  runs at or past `gpu_from_cell`.
- READMEs: `backend/app/`, `practice/`.
