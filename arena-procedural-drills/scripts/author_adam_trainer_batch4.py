#!/usr/bin/env python3
"""Author 8 standalone Colab drills for the Adam-internals + Trainer-skeleton
family of atoms (batch-4).

Atoms covered (each drill = ONE LO + ONE Bloom level, max 2 concurrent KCs):

  ema-first-moment                  — 1 drill (ex1)
  bias-correction-divide            — 1 drill (ex1)
  trainer-class-skeleton            — 1 drill (ex1)
  cross-entropy-classification-loss — 1 drill (ex1)
  argmax-accuracy-eval              — 1 drill (ex1)
  step-counter-increment            — 1 drill (ex1)
  examples-seen-step-axis           — 1 drill (ex1)
  optimizer-loop-on-tensor          — 1 drill (ex1)

These are SMALLER constituent skills that ARENA 0_3_3 (implement Adam) and
chap-3 transformer-training scaffolding both assume the learner can already
perform in isolation. Batch-3 covered the sibling skills
(ema-second-moment, momentum-buffer-update, weight-decay-l2-add,
optimizer-state-tensor-buffers, optimizer-init-params-list, zero-grad-set-none,
training-step-cycle, backward-on-scalar-loss, validation-no-grad,
train-eval-mode-branch, inplace-param-update, inference-mode-step).

Each spec is verified by re-running its solution against its test_body inside
the build venv (torch 2.12.0+cpu) before emission. Any failure aborts the build.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_adam_trainer"


# ---------------------------------------------------------------------------
# Per-atom recap blocks.
# ---------------------------------------------------------------------------

RECAP_EMA_FIRST_MOMENT = (
    "## Adam EMA first moment `m = beta1*m + (1-beta1)*g` — quick refresher\n"
    "\n"
    "Adam maintains TWO running averages per parameter. The first moment "
    "`m` is an EMA of the gradient itself; the second moment `v` (separate "
    "drill) is an EMA of the squared gradient. The first-moment recurrence:\n"
    "\n"
    "```\n"
    "m_t = beta1 * m_{t-1} + (1 - beta1) * g_t\n"
    "```\n"
    "\n"
    "**Why an EMA of `g`, not `g` itself.** The raw gradient is noisy — it "
    "swings step-to-step even on a stationary loss surface. The EMA "
    "low-pass-filters that noise. With `beta1 = 0.9` (Adam default) the "
    "effective averaging window is ~10 recent steps.\n"
    "\n"
    "**Relation to classical momentum.** Classical momentum is "
    "`b_t = mu * b_{t-1} + g_t` (no `(1-mu)` factor). Adam's first moment "
    "is the same idea but RESCALED so `m_t` stays on the same magnitude "
    "scale as `g_t` — which is what makes the bias-correction divide work "
    "cleanly.\n"
    "\n"
    "**Why we update IN PLACE via `m.copy_(...)`.** The buffer lives in the "
    "optimizer's `self.m` list; rebinding `m = beta1*m + (1-beta1)*g` "
    "would only update the local for-loop variable, not the list entry. "
    "Next step would see the stale zero buffer."
)

RECAP_BIAS_CORRECTION = (
    "## Adam bias correction `m_hat = m / (1 - beta1**t)` — quick refresher\n"
    "\n"
    "Both Adam moments start at zero. The EMA recurrence "
    "`m_t = beta1 * m_{t-1} + (1 - beta1) * g_t` therefore drags toward "
    "zero in the FIRST FEW steps even when the true gradient is far from "
    "zero. Concretely, for constant `g`:\n"
    "\n"
    "```\n"
    "m_1 = (1 - beta1) * g\n"
    "m_2 = (1 - beta1**2) * g\n"
    "...\n"
    "m_t = (1 - beta1**t) * g\n"
    "```\n"
    "\n"
    "So `m_t / g = 1 - beta1**t` — the EMA is BIASED toward zero by a "
    "factor of `(1 - beta1**t)`. The bias correction divides it out:\n"
    "\n"
    "```\n"
    "m_hat = m / (1 - beta1**t)\n"
    "v_hat = v / (1 - beta2**t)\n"
    "```\n"
    "\n"
    "**Why this matters.** Without the correction, the first few steps of "
    "Adam would take suspiciously tiny updates — `m` is small not because "
    "the gradient is small, but because the EMA hasn't warmed up. The "
    "correction makes step 1 take a full-magnitude update.\n"
    "\n"
    "**`t` is the STEP COUNTER**, not a tensor. It starts at 1 (not 0) and "
    "is incremented after each optimizer step. `beta1**t` shrinks toward 0 "
    "with `t`, so the divisor `(1 - beta1**t)` grows toward 1 — the "
    "correction fades to a no-op as training progresses."
)

RECAP_TRAINER_SKELETON = (
    "## Trainer class skeleton — quick refresher\n"
    "\n"
    "A Trainer class is a thin object that owns the train/eval loop "
    "scaffold. It separates the GENERIC plumbing (epoch loop, optimizer "
    "step cycle, eval branch) from the MODEL-SPECIFIC `_step` body. The "
    "minimal interface looks like:\n"
    "\n"
    "```\n"
    "class Trainer:\n"
    "    def __init__(self, model, optimizer, train_loader, val_loader):\n"
    "        self.model = model\n"
    "        self.optimizer = optimizer\n"
    "        self.train_loader = train_loader\n"
    "        self.val_loader = val_loader\n"
    "        self.step = 0          # global step counter\n"
    "\n"
    "    def _step(self, x, y):       # one batch → scalar loss\n"
    "        ...\n"
    "\n"
    "    def fit(self, n_epochs):\n"
    "        for epoch in range(n_epochs):\n"
    "            self.model.train()\n"
    "            for x, y in self.train_loader:\n"
    "                loss = self._step(x, y)\n"
    "                loss.backward()\n"
    "                self.optimizer.step()\n"
    "                self.optimizer.zero_grad()\n"
    "                self.step += 1\n"
    "            self.validate()\n"
    "\n"
    "    def validate(self):\n"
    "        self.model.eval()\n"
    "        with t.inference_mode():\n"
    "            ...\n"
    "```\n"
    "\n"
    "ARENA's chapter-3 training code follows exactly this shape (without "
    "PyTorch Lightning) — the same skeleton scales from a 3-line "
    "regression demo to a transformer training run."
)

RECAP_CROSS_ENTROPY = (
    "## Cross-entropy classification loss — quick refresher\n"
    "\n"
    "Cross-entropy for multi-class classification is the negative log-"
    "likelihood of the correct class under the model's softmax:\n"
    "\n"
    "```\n"
    "p_i        = softmax(logits)[i]            # probability of class i\n"
    "loss_per_x = -log(p_{y_true})              # NLL for that example\n"
    "loss       = mean(loss_per_x over batch)   # scalar batch loss\n"
    "```\n"
    "\n"
    "PyTorch packages this as `F.cross_entropy(logits, labels)` and "
    "computes it in a NUMERICALLY STABLE way via the log-sum-exp trick — "
    "you should call it instead of rolling softmax + log + index yourself.\n"
    "\n"
    "**Critical: `cross_entropy` takes LOGITS, not probabilities.** It "
    "applies softmax internally. Passing already-softmaxed values is a "
    "common silent bug — the loss still computes, but it's `-log(softmax("
    "softmax(...)))`, which gives near-uniform gradients and the model "
    "stops learning.\n"
    "\n"
    "**Shape contract.** `logits` is `(B, C)`, `labels` is `(B,)` with "
    "integer class indices in `[0, C)`. Output is a single scalar (default "
    "reduction is `'mean'`)."
)

RECAP_ARGMAX_ACCURACY = (
    "## `(logits.argmax(dim=-1) == labels).float().mean()` — quick refresher\n"
    "\n"
    "Top-1 classification accuracy in three idioms:\n"
    "\n"
    "```\n"
    "preds   = logits.argmax(dim=-1)             # (B,) predicted class\n"
    "correct = (preds == labels)                 # (B,) boolean\n"
    "acc     = correct.float().mean()            # scalar in [0, 1]\n"
    "```\n"
    "\n"
    "**Why `argmax(dim=-1)`.** `logits` is `(B, C)`. We want the index of "
    "the largest logit ALONG THE CLASS AXIS for each example. `dim=-1` "
    "is the class axis regardless of whether there are extra leading dims "
    "(e.g. `(B, T, C)` for token-level outputs).\n"
    "\n"
    "**Why `.float().mean()` and not `.sum() / len(labels)`.** Boolean "
    "tensors can't be `.mean()`-ed directly. Casting to float gives "
    "`1.0 / 0.0` per example, and `.mean()` then handles partial-last-batch "
    "sizes correctly when accumulated across batches with a weighted "
    "average.\n"
    "\n"
    "**Logits vs probabilities — argmax is the same.** Softmax is "
    "monotonic, so `argmax(logits) == argmax(softmax(logits))`. You can "
    "skip the softmax for accuracy; the predicted class is identical."
)

RECAP_STEP_COUNTER = (
    "## `self.step += 1` placement — quick refresher\n"
    "\n"
    "Every trainer / optimizer that needs a notion of TIME (logging "
    "intervals, learning-rate schedules, Adam's bias correction, wandb "
    "x-axis) carries a step counter. The canonical placement:\n"
    "\n"
    "```\n"
    "loss = self._step(x, y)\n"
    "loss.backward()\n"
    "self.optimizer.step()        # apply update\n"
    "self.optimizer.zero_grad()   # clear grads\n"
    "self.step += 1               # tick AFTER the update is committed\n"
    "if self.step % LOG_EVERY == 0:\n"
    "    self.log({'loss': loss.item(), 'step': self.step})\n"
    "```\n"
    "\n"
    "**Why AFTER optimizer.step, not before.** The step counter measures "
    "how many UPDATES have been applied to the model. Incrementing before "
    "the update would mean step 1 logs the model state from BEFORE step 1 "
    "ran. Off-by-one bugs in training graphs almost always trace back to "
    "this placement.\n"
    "\n"
    "**Why BEFORE logging.** Logging at step N should reflect the state "
    "AFTER N updates have happened. So: update → increment → log.\n"
    "\n"
    "**Adam's `self.t` is a SEPARATE counter** — internal to the "
    "optimizer, used for bias correction. The trainer's `self.step` is "
    "external. They happen to be equal if there's one trainer and one "
    "optimizer, but conceptually they're independent."
)

RECAP_EXAMPLES_SEEN = (
    "## `examples_seen = step * batch_size` — quick refresher\n"
    "\n"
    "When comparing training runs that use DIFFERENT batch sizes, plotting "
    "loss vs `step` is misleading — a run with batch_size=64 takes half "
    "the steps of a batch_size=32 run to see the same amount of data. The "
    "fair x-axis is the number of EXAMPLES the model has seen:\n"
    "\n"
    "```\n"
    "examples_seen = step * batch_size\n"
    "wandb.log({'loss': loss.item(), 'examples_seen': examples_seen})\n"
    "```\n"
    "\n"
    "Then in the wandb UI you set the x-axis to `examples_seen` and curves "
    "from different batch sizes overlay correctly.\n"
    "\n"
    "**When step IS the right axis.** If you're comparing two runs with "
    "the SAME batch size, plotting against `step` is fine — and avoids "
    "the multiplication. The distinction matters when batch size varies.\n"
    "\n"
    "**Partial last batch caveat.** `step * batch_size` slightly OVER-"
    "counts when the last batch of an epoch was partial. For most "
    "training graphs the error is < 1% and irrelevant; if you need the "
    "exact count, accumulate `batch.shape[0]` per step."
)

RECAP_OPTIMIZER_LOOP_TENSOR = (
    "## `for p in self.params: p.data -= lr * p.grad` — quick refresher\n"
    "\n"
    "The bare-minimum hand-rolled SGD step is one explicit Python loop "
    "over the parameter list:\n"
    "\n"
    "```\n"
    "class SGD:\n"
    "    def __init__(self, params, lr):\n"
    "        self.params = list(params)\n"
    "        self.lr = lr\n"
    "\n"
    "    @t.inference_mode()\n"
    "    def step(self):\n"
    "        for p in self.params:\n"
    "            if p.grad is not None:\n"
    "                p -= self.lr * p.grad         # in-place under inference_mode\n"
    "```\n"
    "\n"
    "**Why the loop is explicit.** Each parameter is a DIFFERENT tensor "
    "with a DIFFERENT shape. You cannot vectorize the update across "
    "parameters without flattening into a single contiguous buffer (which "
    "PyTorch's `_foreach_*` ops do — see `torch.optim.SGD` source for "
    "the optimized path). For a hand-rolled optimizer the explicit loop "
    "is the right form.\n"
    "\n"
    "**Why guard `if p.grad is not None`.** `zero_grad(set_to_none=True)` "
    "(the default since PyTorch 1.11) clears grads to `None`, not to "
    "zero. A parameter that was never used in the forward pass (frozen "
    "head, masked branch) has `p.grad is None` after `zero_grad`. "
    "Skipping it is correct; trying `p.grad * self.lr` would crash.\n"
    "\n"
    "**Why `p -= ...` and not `p.data -= ...`.** Under `@t.inference_mode"
    "()` the bare in-place op on a leaf is legal. Without the decorator "
    "you reach for `p.data -= ...` as the escape hatch — but ARENA "
    "prefers the decorator. Both produce identical numerical results."
)


# ---------------------------------------------------------------------------
# SPEC list. Each spec = one drill notebook.
# ---------------------------------------------------------------------------

SPECS = [

    # =========================================================
    # ema-first-moment — ex1
    # =========================================================
    {
        "atom_id": "ema-first-moment",
        "subtopic": "Optimizer: Adam EMA first moment",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EMA_FIRST_MOMENT,
        "exercise_index": 1,
        "exercise_title": "Adam m-buffer EMA update m = beta1*m + (1-beta1)*g",
        "slug": "adam-m-buffer-ema-update-m-equals-beta1-m-plus-one-minus-beta1-g",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["adam", "first-moment", "ema", "gradient-ema"],
        "kcs": [
            "ema-first-moment-recurrence",
            "buffer-copy_-mutates-state-in-place",
        ],
        "lo": (
            "Apply the Adam first-moment recurrence `m = beta1*m + "
            "(1-beta1)*g` via `buffer.copy_()` so the gradient-EMA buffer "
            "state is correctly mutated in place across steps."
        ),
        "prompt_body": (
            "Implement `ex1_ema_m_step(m_list, grad_list, beta1)`. The "
            "first-moment update from Adam.\n\n"
            "For each `(m, g)` pair drawn from `(m_list, grad_list)`:\n"
            "\n"
            "1. Compute the new value: `beta1 * m + (1 - beta1) * g`.\n"
            "2. Mutate the buffer IN PLACE: `m.copy_(...)`. Don't rebind.\n"
            "3. Append the new buffer value (by reference) to the return "
            "list.\n\n"
            "Inputs:\n"
            "- `m_list`: list of per-param first-moment buffers (mutated).\n"
            "- `grad_list`: list of per-param gradients (NOT mutated).\n"
            "- `beta1`: float in `(0, 1)` — Adam default is `0.9`.\n\n"
            "Output: list of updated `m` tensors.\n\n"
            "The test runs three steps with KNOWN gradients (including "
            "negative values — unlike the second-moment EMA, the first-"
            "moment EMA preserves sign) and verifies the buffer is "
            "in-place mutated (id and data_ptr preserved across steps)."
        ),
        "stub": (
            "def ex1_ema_m_step(m_list: list, grad_list: list, beta1: float) -> list:\n"
            '    """In-place update: m.copy_(beta1*m + (1-beta1)*g)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# One param, zero-init buffer.\n"
            "m = t.zeros(4)\n"
            "orig_id = id(m)\n"
            "orig_ptr = m.data_ptr()\n"
            "\n"
            "# === Step 1: zero buffer + g => m_1 = (1 - beta1) * g ===\n"
            "g1 = t.tensor([1.0, -2.0, 3.0, -4.0])\n"
            "beta1 = 0.9\n"
            "out1 = ex1_ema_m_step([m], [g1], beta1=beta1)\n"
            "expected1 = (1 - beta1) * g1\n"
            "assert t.allclose(out1[0], expected1), (\n"
            "    f'step 1: expected {expected1}, got {out1[0]}; '\n"
            "    f'check formula: beta1*m + (1-beta1)*g'\n"
            ")\n"
            "assert t.allclose(m, expected1), 'step 1: buffer not mutated to new value'\n"
            "assert id(m) == orig_id, 'buffer was rebound — use m.copy_(...) not m = ...'\n"
            "assert m.data_ptr() == orig_ptr, 'buffer storage reallocated'\n"
            "\n"
            "# === Step 2: same g; m approaches g monotonically (preserving sign) ===\n"
            "m_before_step2 = m.clone()\n"
            "ex1_ema_m_step([m], [g1], beta1=beta1)\n"
            "expected2 = beta1 * expected1 + (1 - beta1) * g1\n"
            "assert t.allclose(m, expected2), (\n"
            "    f'step 2: expected {expected2}, got {m}; '\n"
            "    f'this fails if step 1 did NOT mutate the buffer (rebind bug)'\n"
            ")\n"
            "# Sign preservation: m[1] and m[3] negative; m[0] and m[2] positive.\n"
            "assert m[0] > 0 and m[2] > 0, f'positive-g coords should be positive: {m}'\n"
            "assert m[1] < 0 and m[3] < 0, f'negative-g coords should be negative: {m}'\n"
            "# Magnitude growing toward |g|.\n"
            "assert (m.abs() > m_before_step2.abs()).all(), (\n"
            "    'EMA magnitude should grow toward |g| with constant g; '\n"
            "    'if it shrunk, your formula has the wrong sign'\n"
            ")\n"
            "\n"
            "# === Step 3 — still moving toward g, never overshooting ===\n"
            "ex1_ema_m_step([m], [g1], beta1=beta1)\n"
            "# m_3 = (1 - beta1**3) * g for constant g.\n"
            "expected3 = (1 - beta1 ** 3) * g1\n"
            "assert t.allclose(m, expected3, atol=1e-6), (\n"
            "    f'step 3 closed-form: expected {expected3}, got {m}'\n"
            ")\n"
            "# Magnitude is below |g|.\n"
            "assert (m.abs() < g1.abs()).all(), 'm must not exceed g in magnitude with constant g'\n"
            "\n"
            "# === Multi-param batch ===\n"
            "m_multi = [t.zeros(2), t.zeros(3, 3)]\n"
            "g_multi = [t.tensor([1.0, -1.0]), t.ones(3, 3) * 2.0]\n"
            "ex1_ema_m_step(m_multi, g_multi, beta1=0.5)\n"
            "# 0.5 * 0 + 0.5 * [1, -1] = [0.5, -0.5]\n"
            "assert t.allclose(m_multi[0], t.tensor([0.5, -0.5])), (\n"
            "    f'multi-param step: m_multi[0]={m_multi[0]}'\n"
            ")\n"
            "# 0.5 * 0 + 0.5 * 2 = 1.0 everywhere\n"
            "assert t.allclose(m_multi[1], t.ones(3, 3)), (\n"
            "    f'multi-param step: m_multi[1]={m_multi[1]}'\n"
            ")\n"
            "\n"
            "# === beta1 = 0 collapses to plain g ===\n"
            "m_fresh = t.zeros(3)\n"
            "g_fresh = t.tensor([7.0, 8.0, 9.0])\n"
            "ex1_ema_m_step([m_fresh], [g_fresh], beta1=0.0)\n"
            "assert t.allclose(m_fresh, g_fresh), 'beta1=0: m should just equal g'\n"
            "\n"
            "# === Input grad must not be mutated ===\n"
            "g_in = t.tensor([2.0, 3.0])\n"
            "g_snap = g_in.clone()\n"
            "ex1_ema_m_step([t.zeros(2)], [g_in], beta1=0.99)\n"
            "assert t.equal(g_in, g_snap), 'grad tensors must not be mutated by the EMA update'"
        ),
        "solution_body": (
            "def ex1_ema_m_step(m_list, grad_list, beta1):\n"
            "    out = []\n"
            "    for m, g in zip(m_list, grad_list):\n"
            "        m.copy_(beta1 * m + (1 - beta1) * g)\n"
            "        out.append(m)\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why this is its own atom (separate from `ema-second-"
            "moment`).** Numerically the recurrences look symmetric: "
            "`m = b1*m + (1-b1)*g` vs `v = b2*v + (1-b2)*g.pow(2)`. But "
            "they behave differently: `m` preserves the SIGN of `g` (so "
            "it captures direction); `v` is always non-negative (so it "
            "captures magnitude). Conflating them is the #2 Adam bug "
            "after the rebind bug.\n\n"
            "**What the EMA converges to.** For constant `g`, "
            "`m_t = (1 - beta1^t) * g`. As `t -> inf`, `m_t -> g`. With "
            "`beta1 = 0.9` the effective averaging window is ~10 steps — "
            "shorter than the second-moment's ~1000-step window. This is "
            "intentional: gradient DIRECTION needs to react faster than "
            "gradient MAGNITUDE.\n\n"
            "**Why we return `m` by reference.** After `m.copy_(...)`, "
            "the buffer IS the new value. Returning `m` (rather than the "
            "computed expression) means downstream code that does "
            "`theta -= lr * m_hat / (v_hat.sqrt() + eps)` reads from the "
            "freshly-mutated buffer."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # bias-correction-divide — ex1
    # =========================================================
    {
        "atom_id": "bias-correction-divide",
        "subtopic": "Optimizer: Adam bias-correction divide",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_BIAS_CORRECTION,
        "exercise_index": 1,
        "exercise_title": "bias-correct an Adam moment: m_hat = m / (1 - beta**t)",
        "slug": "bias-correct-an-adam-moment-m-hat-equals-m-over-one-minus-beta-to-the-t",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["bias-correction", "adam", "warmup", "step-counter"],
        "kcs": [
            "bias-correction-divide-by-one-minus-beta-power-t",
            "step-counter-1-based-for-bias-correction",
        ],
        "lo": (
            "Apply the Adam bias-correction divide `m_hat = m / (1 - "
            "beta**t)` to undo the zero-initialization bias of an EMA "
            "buffer at step `t`."
        ),
        "prompt_body": (
            "Implement `ex1_bias_correct(m, beta, t_step)`. The two-"
            "argument bias-correction divide from Adam.\n\n"
            "1. Compute the correction factor `1 - beta ** t_step`. This "
            "is a Python scalar (not a tensor).\n"
            "2. Return `m / correction`. Do NOT mutate `m` in place — "
            "return a new tensor.\n\n"
            "Inputs:\n"
            "- `m`: EMA buffer (tensor, any shape).\n"
            "- `beta`: float decay coefficient (e.g. 0.9 for first moment, "
            "0.999 for second moment).\n"
            "- `t_step`: int >= 1, the current step (1-based).\n\n"
            "Output: bias-corrected `m_hat`, same shape as `m`.\n\n"
            "**Critical:** `t_step` is 1-based, not 0-based. At step 1, "
            "the correction is `1 / (1 - beta)`, which is BIG — it "
            "amplifies the small warmup-EMA back to full magnitude. If "
            "you pass `t_step=0` you get a division by zero."
        ),
        "stub": (
            "def ex1_bias_correct(m: Tensor, beta: float, t_step: int) -> Tensor:\n"
            '    """Return m / (1 - beta**t_step)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Step 1 with beta=0.9 and constant g => m = 0.1 * g.\n"
            "# Bias correction should recover m_hat == g.\n"
            "g = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "beta1 = 0.9\n"
            "m_step1 = (1 - beta1) * g       # what the EMA produces at step 1\n"
            "m_hat = ex1_bias_correct(m_step1, beta1, t_step=1)\n"
            "assert t.allclose(m_hat, g, atol=1e-6), (\n"
            "    f'step 1 with constant g should recover g exactly; got {m_hat}, expected {g}; '\n"
            "    f'check formula: m / (1 - beta**t)'\n"
            ")\n"
            "\n"
            "# === Step 5 with beta=0.999 (second-moment style) — correction approaches 1 as t grows.\n"
            "beta2 = 0.999\n"
            "v = t.tensor([0.5, 1.0, 2.0])\n"
            "v_hat = ex1_bias_correct(v, beta2, t_step=5)\n"
            "expected_v_hat = v / (1 - beta2 ** 5)\n"
            "assert t.allclose(v_hat, expected_v_hat, atol=1e-6), (\n"
            "    f'step 5 with beta=0.999: expected {expected_v_hat}, got {v_hat}'\n"
            ")\n"
            "# The correction divisor at step 5 with beta=0.999 is tiny → v_hat is HUGE.\n"
            "assert (v_hat > v).all(), 'bias-corrected v should be larger than raw v early in training'\n"
            "\n"
            "# === As t -> inf, correction -> 1 → m_hat -> m ===\n"
            "m_big_t = ex1_bias_correct(t.tensor([1.0, 1.0]), beta=0.9, t_step=10000)\n"
            "assert t.allclose(m_big_t, t.tensor([1.0, 1.0]), atol=1e-6), (\n"
            "    f'large t: m_hat should approach m; got {m_big_t}'\n"
            ")\n"
            "\n"
            "# === Multi-dim shape preservation ===\n"
            "m_2d = t.randn(4, 5)\n"
            "out_2d = ex1_bias_correct(m_2d, beta=0.9, t_step=3)\n"
            "assert out_2d.shape == (4, 5)\n"
            "expected_2d = m_2d / (1 - 0.9 ** 3)\n"
            "assert t.allclose(out_2d, expected_2d, atol=1e-6)\n"
            "\n"
            "# === Input m must NOT be mutated ===\n"
            "m_in = t.tensor([1.0, 2.0, 3.0])\n"
            "m_snap = m_in.clone()\n"
            "_ = ex1_bias_correct(m_in, beta=0.9, t_step=1)\n"
            "assert t.equal(m_in, m_snap), (\n"
            "    f'input m was mutated (now {m_in}, was {m_snap}); '\n"
            "    f'this fold must be out-of-place'\n"
            ")\n"
            "\n"
            "# === Step-1 closed-form sanity: m_hat[i] should equal g[i] exactly when EMA started from 0.\n"
            "# Round-trip: run the EMA recurrence then bias-correct → recover g.\n"
            "for beta_test in [0.5, 0.9, 0.999]:\n"
            "    for step_test in [1, 2, 5, 10]:\n"
            "        # Closed form for constant g across step_test steps: m = (1 - beta**step_test) * g\n"
            "        m_synth = (1 - beta_test ** step_test) * g\n"
            "        recovered = ex1_bias_correct(m_synth, beta_test, step_test)\n"
            "        assert t.allclose(recovered, g, atol=1e-5), (\n"
            "            f'round-trip failed for beta={beta_test}, t={step_test}: got {recovered}, expected {g}'\n"
            "        )"
        ),
        "solution_body": (
            "def ex1_bias_correct(m, beta, t_step):\n"
            "    return m / (1 - beta ** t_step)"
        ),
        "solution_notes": (
            "**Why `beta ** t_step` (Python power) not `t.pow(beta, "
            "t_step)`.** `beta` is a Python float and `t_step` is a "
            "Python int — `**` evaluates on the host as a plain scalar. "
            "Wrapping it in a tensor op would create a tiny GPU "
            "synchronization for nothing. Adam's reference impl does the "
            "scalar.\n\n"
            "**Why this is its own atom.** The bias correction is a "
            "ONE-LINE divide, but it's where two-thirds of Adam-from-"
            "scratch bugs live: passing `t=0`, forgetting to update `t` "
            "between steps, applying the correction to the wrong moment "
            "(e.g. `m / (1 - beta2**t)`). Isolating it as its own drill "
            "lets you nail the placement of `t_step` and the "
            "`(1 - beta**t)` denominator before composing it into the "
            "full Adam impl.\n\n"
            "**Aside: AdamW skips bias correction.** Some literature "
            "claims AdamW doesn't bias-correct; the reference HuggingFace "
            "AdamW implementation DOES still apply it — only the weight-"
            "decay term is decoupled, not the bias correction. Don't be "
            "tricked by old blog posts."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # trainer-class-skeleton — ex1
    # =========================================================
    {
        "atom_id": "trainer-class-skeleton",
        "subtopic": "Trainer: Trainer class skeleton",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_TRAINER_SKELETON,
        "exercise_index": 1,
        "exercise_title": "minimal Trainer class: fit, validate, _step",
        "slug": "minimal-trainer-class-fit-validate-step",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["trainer", "fit-loop", "validate", "object-oriented"],
        "kcs": [
            "trainer-fit-loop-walks-epochs",
            "trainer-validate-uses-eval-mode",
        ],
        "lo": (
            "Apply the canonical Trainer-class skeleton — `__init__`, "
            "`_step`, `fit`, `validate` — so the per-batch training "
            "logic is separated from the model-specific loss computation."
        ),
        "prompt_body": (
            "Implement `Ex1Trainer`. The minimal-but-correct Trainer "
            "skeleton.\n\n"
            "1. `__init__(self, model, optimizer, train_loader, "
            "val_loader, loss_fn)`: store all five as attributes. Also "
            "initialize `self.step = 0` (global step counter, batches "
            "completed) and `self.history = {'train_loss': [], "
            "'val_loss': []}`.\n\n"
            "2. `_step(self, x, y) -> Tensor`: forward + loss only. "
            "Compute `logits = self.model(x)` then return "
            "`self.loss_fn(logits, y)`. Do NOT call `backward()` here.\n\n"
            "3. `fit(self, n_epochs)`: for each epoch:\n"
            "   - call `self.model.train()` (sets BN/dropout to train mode).\n"
            "   - iterate `(x, y) in self.train_loader`:\n"
            "     - `loss = self._step(x, y)`\n"
            "     - `loss.backward()`\n"
            "     - `self.optimizer.step()`\n"
            "     - `self.optimizer.zero_grad()`\n"
            "     - `self.step += 1`\n"
            "     - `self.history['train_loss'].append(loss.item())`\n"
            "   - call `self.validate()` at the end of each epoch.\n\n"
            "4. `validate(self)`: switch to eval mode, run inference under "
            "`t.inference_mode()`, accumulate loss over `self.val_loader`, "
            "average, append to `self.history['val_loss']`.\n\n"
            "The test uses a tiny linear regression task so you can check "
            "the loop walks the data, validates each epoch, and reduces "
            "loss across epochs."
        ),
        "stub": (
            "class Ex1Trainer:\n"
            '    """Minimal Trainer class: model + optimizer + loaders + loss."""\n'
            "\n"
            "    def __init__(self, model, optimizer, train_loader, val_loader, loss_fn):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def _step(self, x, y):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def fit(self, n_epochs: int):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def validate(self):\n"
            "        raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.utils.data import TensorDataset, DataLoader\n"
            "\n"
            "# Tiny linear regression task: y = 2x + 1 with noise.\n"
            "t.manual_seed(0)\n"
            "N = 64\n"
            "x_train = t.randn(N, 1)\n"
            "y_train = 2.0 * x_train + 1.0 + 0.05 * t.randn(N, 1)\n"
            "x_val = t.randn(16, 1)\n"
            "y_val = 2.0 * x_val + 1.0\n"
            "\n"
            "train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True)\n"
            "val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8, shuffle=False)\n"
            "\n"
            "model = t.nn.Linear(1, 1)\n"
            "opt = t.optim.SGD(model.parameters(), lr=0.1)\n"
            "loss_fn = t.nn.MSELoss()\n"
            "\n"
            "trainer = Ex1Trainer(model, opt, train_loader, val_loader, loss_fn)\n"
            "\n"
            "# === Attribute check ===\n"
            "assert trainer.model is model\n"
            "assert trainer.optimizer is opt\n"
            "assert trainer.train_loader is train_loader\n"
            "assert trainer.val_loader is val_loader\n"
            "assert trainer.loss_fn is loss_fn\n"
            "assert trainer.step == 0, f'step must start at 0, got {trainer.step}'\n"
            "assert 'train_loss' in trainer.history and 'val_loss' in trainer.history\n"
            "assert trainer.history['train_loss'] == []\n"
            "assert trainer.history['val_loss'] == []\n"
            "\n"
            "# === _step returns a SCALAR loss tensor (not a number, not yet detached) ===\n"
            "x_b, y_b = next(iter(train_loader))\n"
            "loss = trainer._step(x_b, y_b)\n"
            "assert isinstance(loss, t.Tensor), f'_step must return a tensor, got {type(loss)}'\n"
            "assert loss.ndim == 0, f'_step must return a scalar tensor, got shape {tuple(loss.shape)}'\n"
            "assert loss.requires_grad, '_step must return a grad-tracking loss (no .item() / .detach())'\n"
            "\n"
            "# === fit runs without error and steps the counter ===\n"
            "trainer.fit(n_epochs=3)\n"
            "expected_steps = 3 * len(train_loader)\n"
            "assert trainer.step == expected_steps, (\n"
            "    f'after 3 epochs of {len(train_loader)} batches, step should be {expected_steps}; got {trainer.step}'\n"
            ")\n"
            "assert len(trainer.history['train_loss']) == expected_steps, (\n"
            "    f'train_loss history length should equal step count'\n"
            ")\n"
            "assert len(trainer.history['val_loss']) == 3, (\n"
            "    f'val_loss should be logged once per epoch; got {len(trainer.history[\"val_loss\"])} entries after 3 epochs'\n"
            ")\n"
            "\n"
            "# === Loss decreases across epochs ===\n"
            "v0, v1, v2 = trainer.history['val_loss']\n"
            "assert v2 < v0, f'val loss should decrease over 3 epochs: epoch0={v0:.4f}, epoch2={v2:.4f}'\n"
            "# Model fitted close to truth.\n"
            "assert abs(model.weight.item() - 2.0) < 0.2, (\n"
            "    f'weight should approach 2.0; got {model.weight.item():.4f}'\n"
            ")\n"
            "assert abs(model.bias.item() - 1.0) < 0.2, (\n"
            "    f'bias should approach 1.0; got {model.bias.item():.4f}'\n"
            ")\n"
            "\n"
            "# === Validate uses eval mode + inference_mode (no grads tracked) ===\n"
            "# Hard to test directly; instead verify that after fit(), val_loss values are scalars (not tensors).\n"
            "for v in trainer.history['val_loss']:\n"
            "    assert isinstance(v, float), (\n"
            "        f'val_loss entries should be Python floats (use .item()); got {type(v)}'\n"
            "    )"
        ),
        "solution_body": (
            "class Ex1Trainer:\n"
            "    def __init__(self, model, optimizer, train_loader, val_loader, loss_fn):\n"
            "        self.model = model\n"
            "        self.optimizer = optimizer\n"
            "        self.train_loader = train_loader\n"
            "        self.val_loader = val_loader\n"
            "        self.loss_fn = loss_fn\n"
            "        self.step = 0\n"
            "        self.history = {'train_loss': [], 'val_loss': []}\n"
            "\n"
            "    def _step(self, x, y):\n"
            "        logits = self.model(x)\n"
            "        return self.loss_fn(logits, y)\n"
            "\n"
            "    def fit(self, n_epochs):\n"
            "        for _epoch in range(n_epochs):\n"
            "            self.model.train()\n"
            "            for x, y in self.train_loader:\n"
            "                loss = self._step(x, y)\n"
            "                loss.backward()\n"
            "                self.optimizer.step()\n"
            "                self.optimizer.zero_grad()\n"
            "                self.step += 1\n"
            "                self.history['train_loss'].append(loss.item())\n"
            "            self.validate()\n"
            "\n"
            "    def validate(self):\n"
            "        self.model.eval()\n"
            "        total = 0.0\n"
            "        count = 0\n"
            "        with t.inference_mode():\n"
            "            for x, y in self.val_loader:\n"
            "                loss = self.loss_fn(self.model(x), y)\n"
            "                total += loss.item() * x.shape[0]\n"
            "                count += x.shape[0]\n"
            "        self.history['val_loss'].append(total / count)"
        ),
        "solution_notes": (
            "**Why separate `_step` from `fit`.** The model-specific "
            "logic (forward + loss) lives in `_step`. The generic "
            "scaffolding (epoch loop, optimizer cycle, validation) lives "
            "in `fit`. Swapping models means rewriting only `_step`. "
            "PyTorch Lightning generalizes this further with "
            "`training_step` / `validation_step` hooks — same idea, more "
            "ceremony.\n\n"
            "**Why `self.history` is a dict.** As you add more metrics "
            "(accuracy, learning rate, gradient norms) the history dict "
            "just grows new keys. A `(train_losses, val_losses)` tuple "
            "would force a refactor.\n\n"
            "**Why `inference_mode()` inside `validate`.** Eval-only "
            "code shouldn't build the autograd graph (wastes memory and "
            "time). `inference_mode` is stricter than `no_grad` — it "
            "also disables version counters. Both are correct; "
            "`inference_mode` is what new PyTorch code uses.\n\n"
            "**Weighted-average val loss.** Multiplying by `x.shape[0]` "
            "(the batch size) handles the partial last batch correctly. "
            "A naive `mean(losses)` would over-weight the small partial "
            "batch."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # cross-entropy-classification-loss — ex1
    # =========================================================
    {
        "atom_id": "cross-entropy-classification-loss",
        "subtopic": "Loss: Cross-entropy classification",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_CROSS_ENTROPY,
        "exercise_index": 1,
        "exercise_title": "manual cross-entropy matches F.cross_entropy on logits",
        "slug": "manual-cross-entropy-matches-f-cross-entropy-on-logits",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["cross-entropy", "softmax", "log-softmax", "classification"],
        "kcs": [
            "cross-entropy-takes-logits-not-probs",
            "cross-entropy-equals-mean-of-neg-log-target-probs",
        ],
        "lo": (
            "Apply the manual cross-entropy decomposition "
            "`loss = -log_softmax(logits)[range(B), labels].mean()` and "
            "verify it matches `F.cross_entropy(logits, labels)`."
        ),
        "prompt_body": (
            "Implement `ex1_manual_cross_entropy(logits, labels)`. Roll "
            "your own cross-entropy WITHOUT calling `F.cross_entropy` or "
            "`F.nll_loss`. Then we verify it matches PyTorch's reference.\n\n"
            "1. `log_probs = F.log_softmax(logits, dim=-1)` — numerically "
            "stable log-softmax along the class axis. Shape `(B, C)`.\n"
            "2. Gather the LOG-PROBABILITY of the CORRECT class for each "
            "example. The clearest way: "
            "`target_log_probs = log_probs[range(len(labels)), labels]`. "
            "Shape `(B,)`.\n"
            "3. Return `-target_log_probs.mean()` — a scalar tensor.\n\n"
            "Inputs:\n"
            "- `logits`: `(B, C)` raw model outputs.\n"
            "- `labels`: `(B,)` int64 class indices in `[0, C)`.\n\n"
            "Output: scalar loss, must match `F.cross_entropy(logits, "
            "labels)` to within 1e-5.\n\n"
            "**Critical:** you receive LOGITS, not probabilities. Do NOT "
            "softmax-then-log — use `F.log_softmax` to be numerically "
            "stable. The reference is `F.cross_entropy` which expects "
            "logits."
        ),
        "stub": (
            "import torch.nn.functional as F\n"
            "\n"
            "\n"
            "def ex1_manual_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:\n"
            '    """Manual cross-entropy from logits + integer labels. Returns a scalar."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn.functional as F\n"
            "\n"
            "# === Tiny canonical case ===\n"
            "logits = t.tensor([\n"
            "    [2.0, 1.0, 0.1],   # softmax peaks at class 0\n"
            "    [0.1, 2.0, 1.0],   # softmax peaks at class 1\n"
            "    [0.5, 0.5, 0.5],   # uniform\n"
            "])\n"
            "labels = t.tensor([0, 1, 2], dtype=t.long)\n"
            "\n"
            "our_loss = ex1_manual_cross_entropy(logits, labels)\n"
            "ref_loss = F.cross_entropy(logits, labels)\n"
            "\n"
            "assert isinstance(our_loss, t.Tensor), f'must return a tensor, got {type(our_loss)}'\n"
            "assert our_loss.ndim == 0, f'must return a scalar tensor, got shape {tuple(our_loss.shape)}'\n"
            "assert t.allclose(our_loss, ref_loss, atol=1e-5), (\n"
            "    f'manual cross-entropy ({our_loss.item():.6f}) != F.cross_entropy ({ref_loss.item():.6f}); '\n"
            "    f'check you are using log_softmax + gather of target-class log-probs, then mean'\n"
            ")\n"
            "\n"
            "# === Perfect prediction: confident logit toward correct class -> ~0 loss ===\n"
            "logits_easy = t.tensor([\n"
            "    [10.0, -10.0, -10.0],\n"
            "    [-10.0, 10.0, -10.0],\n"
            "])\n"
            "labels_easy = t.tensor([0, 1], dtype=t.long)\n"
            "loss_easy = ex1_manual_cross_entropy(logits_easy, labels_easy)\n"
            "assert loss_easy.item() < 0.001, (\n"
            "    f'confident-correct predictions should give near-zero loss; got {loss_easy.item():.6f}'\n"
            ")\n"
            "\n"
            "# === Confidently wrong: huge loss ===\n"
            "labels_wrong = t.tensor([1, 0], dtype=t.long)\n"
            "loss_wrong = ex1_manual_cross_entropy(logits_easy, labels_wrong)\n"
            "assert loss_wrong.item() > 5, (\n"
            "    f'confidently wrong predictions should give large loss; got {loss_wrong.item():.4f}'\n"
            ")\n"
            "\n"
            "# === Uniform logits → loss == log(C) ===\n"
            "import math\n"
            "C = 4\n"
            "logits_unif = t.zeros(10, C)        # uniform softmax\n"
            "labels_unif = t.randint(0, C, (10,))\n"
            "loss_unif = ex1_manual_cross_entropy(logits_unif, labels_unif)\n"
            "expected_unif = math.log(C)\n"
            "assert abs(loss_unif.item() - expected_unif) < 1e-5, (\n"
            "    f'uniform logits should give log(C)={expected_unif:.4f}, got {loss_unif.item():.4f}'\n"
            ")\n"
            "\n"
            "# === Realistic randomized case at scale ===\n"
            "rng = t.Generator().manual_seed(7)\n"
            "B, C = 32, 10\n"
            "big_logits = t.randn(B, C, generator=rng) * 2.0\n"
            "big_labels = t.randint(0, C, (B,), generator=rng)\n"
            "ours = ex1_manual_cross_entropy(big_logits, big_labels)\n"
            "ref = F.cross_entropy(big_logits, big_labels)\n"
            "assert t.allclose(ours, ref, atol=1e-5), (\n"
            "    f'large-scale mismatch: ours={ours.item():.6f}, ref={ref.item():.6f}'\n"
            ")\n"
            "\n"
            "# === Loss tracks grad (so it can be backward'd) ===\n"
            "logits_grad = t.randn(4, 3, requires_grad=True)\n"
            "labels_grad = t.tensor([0, 1, 2, 0], dtype=t.long)\n"
            "loss_grad = ex1_manual_cross_entropy(logits_grad, labels_grad)\n"
            "assert loss_grad.requires_grad, 'loss must track grad through log_softmax'\n"
            "loss_grad.backward()\n"
            "assert logits_grad.grad is not None and logits_grad.grad.shape == (4, 3)"
        ),
        "solution_body": (
            "import torch.nn.functional as F\n"
            "\n"
            "\n"
            "def ex1_manual_cross_entropy(logits, labels):\n"
            "    log_probs = F.log_softmax(logits, dim=-1)\n"
            "    target_log_probs = log_probs[range(len(labels)), labels]\n"
            "    return -target_log_probs.mean()"
        ),
        "solution_notes": (
            "**Why `F.log_softmax` not `log(softmax(...))`.** Softmax "
            "involves `exp(logits)`, which overflows for large logits "
            "(e.g. 1000). `log_softmax` uses the log-sum-exp trick "
            "internally — it shifts logits by the per-row max before "
            "exponentiating, so the largest exponent is `exp(0) = 1` and "
            "the rest are smaller. This is numerically stable for any "
            "input magnitude.\n\n"
            "**Why `range(len(labels))` for the gather.** "
            "`log_probs[range(B), labels]` is fancy-indexing — it picks "
            "`log_probs[0, labels[0]], log_probs[1, labels[1]], ...`. "
            "Equivalent to `log_probs.gather(1, labels.unsqueeze(1)"
            ").squeeze(1)`, just less ceremonial. Both work.\n\n"
            "**Why this drill matters even though `F.cross_entropy` "
            "exists.** Two reasons: (1) you'll see this exact "
            "decomposition in language-modeling losses where you want to "
            "weight per-token losses before reducing; (2) `F.cross_"
            "entropy(probs, labels)` is the most common 'why isn't my "
            "model learning' bug — and you can only diagnose it if you "
            "understand what the function expects (LOGITS) vs what you "
            "accidentally passed (POST-SOFTMAX PROBS)."
        ),
        "extra_imports": ["import torch.nn.functional as F"],
    },

    # =========================================================
    # argmax-accuracy-eval — ex1
    # =========================================================
    {
        "atom_id": "argmax-accuracy-eval",
        "subtopic": "Eval: argmax accuracy",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_ARGMAX_ACCURACY,
        "exercise_index": 1,
        "exercise_title": "top-1 classification accuracy from logits",
        "slug": "top-1-classification-accuracy-from-logits",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["accuracy", "argmax", "eval-metric", "classification"],
        "kcs": [
            "argmax-along-class-dim-minus-1",
            "boolean-tensor-float-mean-accuracy",
        ],
        "lo": (
            "Apply the `(logits.argmax(dim=-1) == labels).float().mean()` "
            "accuracy pattern, including the dim=-1 axis choice and the "
            "boolean-to-float cast."
        ),
        "prompt_body": (
            "Implement `ex1_top1_accuracy(logits, labels)`. The standard "
            "classification eval metric.\n\n"
            "1. Compute `preds = logits.argmax(dim=-1)` — shape `(B,)`.\n"
            "2. Compute `correct = (preds == labels)` — shape `(B,)`, "
            "dtype bool.\n"
            "3. Return `correct.float().mean()` — a scalar tensor in "
            "`[0, 1]`.\n\n"
            "Inputs:\n"
            "- `logits`: `(B, C)` float tensor.\n"
            "- `labels`: `(B,)` int tensor with class indices in `[0, C)`.\n\n"
            "Output: scalar accuracy tensor.\n\n"
            "**Critical:** use `dim=-1` (the class axis) not `dim=0` or "
            "`dim=1` explicitly — `dim=-1` correctly handles `(B, C)` and "
            "`(B, T, C)` (token-level) shapes alike. The test verifies "
            "you got this right by passing both shapes."
        ),
        "stub": (
            "def ex1_top1_accuracy(logits: Tensor, labels: Tensor) -> Tensor:\n"
            '    """Top-1 classification accuracy: argmax along class axis, mean over examples."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Perfect accuracy: argmax aligned with labels ===\n"
            "logits = t.tensor([\n"
            "    [10.0, 0.0, 0.0],   # argmax → 0\n"
            "    [0.0, 10.0, 0.0],   # argmax → 1\n"
            "    [0.0, 0.0, 10.0],   # argmax → 2\n"
            "    [10.0, 0.0, 0.0],   # argmax → 0\n"
            "])\n"
            "labels = t.tensor([0, 1, 2, 0], dtype=t.long)\n"
            "acc = ex1_top1_accuracy(logits, labels)\n"
            "assert isinstance(acc, t.Tensor), f'must return a tensor, got {type(acc)}'\n"
            "assert acc.ndim == 0, f'must return a scalar, got shape {tuple(acc.shape)}'\n"
            "assert acc.dtype.is_floating_point, (\n"
            "    f'must return a float tensor (cast bool→float before mean); got dtype {acc.dtype}'\n"
            ")\n"
            "assert acc.item() == 1.0, f'all predictions correct → acc=1.0; got {acc.item()}'\n"
            "\n"
            "# === Zero accuracy: every prediction wrong ===\n"
            "labels_wrong = t.tensor([2, 2, 0, 2], dtype=t.long)   # 0/4 correct\n"
            "acc_wrong = ex1_top1_accuracy(logits, labels_wrong)\n"
            "assert acc_wrong.item() == 0.0, f'all wrong → acc=0.0; got {acc_wrong.item()}'\n"
            "\n"
            "# === Half right ===\n"
            "labels_half = t.tensor([0, 1, 0, 2], dtype=t.long)   # 2/4 correct\n"
            "acc_half = ex1_top1_accuracy(logits, labels_half)\n"
            "assert abs(acc_half.item() - 0.5) < 1e-7, f'2/4 correct → acc=0.5; got {acc_half.item()}'\n"
            "\n"
            "# === Bound check on random data ===\n"
            "rng = t.Generator().manual_seed(13)\n"
            "B, C = 100, 10\n"
            "random_logits = t.randn(B, C, generator=rng)\n"
            "random_labels = t.randint(0, C, (B,), generator=rng)\n"
            "acc_rand = ex1_top1_accuracy(random_logits, random_labels)\n"
            "assert 0.0 <= acc_rand.item() <= 1.0, f'acc must be in [0,1]; got {acc_rand.item()}'\n"
            "# Random predictions on 10-class labels: expected acc ≈ 0.1 (loose bound).\n"
            "assert acc_rand.item() < 0.3, f'random preds should give ~0.1 acc; got {acc_rand.item()} (>0.3 suspicious)'\n"
            "\n"
            "# === dim=-1 handles (B, T, C) token-level shapes too ===\n"
            "# This is what catches dim=0 or dim=1 mistakes.\n"
            "tok_logits = t.tensor([\n"
            "    [[10.0, 0.0], [0.0, 10.0]],   # batch 0: token 0 → class 0, token 1 → class 1\n"
            "    [[0.0, 10.0], [10.0, 0.0]],   # batch 1: token 0 → class 1, token 1 → class 0\n"
            "])  # shape (B=2, T=2, C=2)\n"
            "tok_labels = t.tensor([\n"
            "    [0, 1],\n"
            "    [1, 0],\n"
            "], dtype=t.long)  # shape (B=2, T=2)\n"
            "tok_acc = ex1_top1_accuracy(tok_logits, tok_labels)\n"
            "assert tok_acc.item() == 1.0, (\n"
            "    f'token-level (B,T,C) inputs should give 1.0 (all correct); got {tok_acc.item()}. '\n"
            "    f'Make sure you used dim=-1 (class axis), not dim=0 or dim=1.'\n"
            ")\n"
            "\n"
            "# === Verify intermediate tensor types are sensible (catches forgotten .float()) ===\n"
            "# If you returned a Long mean it would error or round; we check the result is non-integer typically.\n"
            "acc_check = ex1_top1_accuracy(\n"
            "    t.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),\n"
            "    t.tensor([0, 1, 1], dtype=t.long),   # 2/3 correct → 0.6667\n"
            ")\n"
            "assert abs(acc_check.item() - 2/3) < 1e-6, (\n"
            "    f'2/3 correct should give ~0.6667; got {acc_check.item()}; '\n"
            "    f'did you forget the .float() cast before .mean()?'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_top1_accuracy(logits, labels):\n"
            "    preds = logits.argmax(dim=-1)\n"
            "    return (preds == labels).float().mean()"
        ),
        "solution_notes": (
            "**Why bool → float → mean.** `t.tensor([True, False]).mean()` "
            "raises `RuntimeError: Can only calculate the mean of "
            "floating types`. The fix is `(preds == labels).float()` "
            "before `.mean()`. A common almost-right form is "
            "`correct.sum() / correct.numel()` — works but is less "
            "idiomatic and slightly slower (two ops vs one).\n\n"
            "**Why argmax is enough — no softmax needed.** Softmax is "
            "MONOTONIC: if `logits[a] > logits[b]` then "
            "`softmax(logits)[a] > softmax(logits)[b]`. So "
            "`argmax(logits) == argmax(softmax(logits))`. Doing the "
            "softmax first is harmless but wasteful.\n\n"
            "**Where this lives in real training code.** Inside "
            "`validate()` you accumulate accuracy over the val loader, "
            "weighted by batch size (just like loss). For top-K accuracy "
            "you replace `argmax(dim=-1)` with `topk(k, dim=-1).indices` "
            "and check whether `labels` is in that set."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # step-counter-increment — ex1
    # =========================================================
    {
        "atom_id": "step-counter-increment",
        "subtopic": "Trainer: step counter increment",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_STEP_COUNTER,
        "exercise_index": 1,
        "exercise_title": "step counter increments AFTER optimizer.step",
        "slug": "step-counter-increments-after-optimizer-step",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["step-counter", "training-loop", "logging", "order-of-ops"],
        "kcs": [
            "step-counter-increments-after-optimizer-step",
            "step-counter-starts-at-zero",
        ],
        "lo": (
            "Apply the canonical `self.step += 1` placement inside a "
            "training loop body: AFTER `optimizer.step()` and "
            "`zero_grad()`, BEFORE any logging that should reflect the "
            "post-update state."
        ),
        "prompt_body": (
            "Implement `ex1_train_one_epoch(model, optimizer, loader, "
            "loss_fn, start_step)`. ONE epoch over the loader with a "
            "correctly-placed step counter.\n\n"
            "Initialize `step = start_step`. For each `(x, y)` in "
            "`loader`:\n"
            "\n"
            "1. `loss = loss_fn(model(x), y)` — forward + loss.\n"
            "2. `loss.backward()` — compute grads.\n"
            "3. `optimizer.step()` — apply update.\n"
            "4. `optimizer.zero_grad()` — clear grads.\n"
            "5. `step += 1` — tick AFTER the update is committed.\n"
            "6. Record `(step, loss.item())` in a log list AFTER the "
            "tick — so the log entry's step value reflects the state "
            "AFTER `step` batches have been processed.\n\n"
            "Return `(final_step, log_list)`.\n\n"
            "Inputs:\n"
            "- `model`, `optimizer`, `loader`, `loss_fn`: usual.\n"
            "- `start_step`: int — the counter value BEFORE this epoch "
            "runs. (Lets you accumulate across epochs.)\n\n"
            "Output:\n"
            "- `final_step`: int — the counter after the epoch.\n"
            "- `log_list`: list of `(step, loss_float)` tuples, one per "
            "batch."
        ),
        "stub": (
            "def ex1_train_one_epoch(model, optimizer, loader, loss_fn, start_step: int) -> tuple:\n"
            '    """Train one epoch. Increment step counter AFTER each optimizer.step()."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.utils.data import TensorDataset, DataLoader\n"
            "\n"
            "t.manual_seed(0)\n"
            "x = t.randn(24, 2)\n"
            "y = (x.sum(dim=-1, keepdim=True) > 0).float()   # toy binary task\n"
            "loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)\n"
            "n_batches = len(loader)\n"
            "assert n_batches == 6\n"
            "\n"
            "model = t.nn.Linear(2, 1)\n"
            "opt = t.optim.SGD(model.parameters(), lr=0.1)\n"
            "loss_fn = t.nn.MSELoss()\n"
            "\n"
            "# === Epoch 1: start_step=0 ===\n"
            "final_step, log = ex1_train_one_epoch(model, opt, loader, loss_fn, start_step=0)\n"
            "assert final_step == n_batches, (\n"
            "    f'after 6 batches starting at step 0, final_step should be 6; got {final_step}'\n"
            ")\n"
            "assert len(log) == n_batches, f'expected {n_batches} log entries, got {len(log)}'\n"
            "\n"
            "# The FIRST log entry should be step=1 (one update has happened by then).\n"
            "first_step, first_loss = log[0]\n"
            "assert first_step == 1, (\n"
            "    f'first log entry should have step=1 (one update happened before logging); got step={first_step}. '\n"
            "    f'If you got step=0, you logged BEFORE incrementing — the canonical placement is AFTER.'\n"
            ")\n"
            "# The LAST log entry should be step=n_batches.\n"
            "last_step, last_loss = log[-1]\n"
            "assert last_step == n_batches, (\n"
            "    f'last log entry should have step={n_batches}; got step={last_step}'\n"
            ")\n"
            "# Step values are strictly increasing by 1.\n"
            "steps_only = [s for s, _ in log]\n"
            "assert steps_only == list(range(1, n_batches + 1)), (\n"
            "    f'step values should be [1, 2, ..., {n_batches}]; got {steps_only}'\n"
            ")\n"
            "# Loss values are floats (you ran .item()).\n"
            "for s, l in log:\n"
            "    assert isinstance(l, float), f'loss in log entry should be float; got {type(l)}'\n"
            "\n"
            "# === Epoch 2: start_step continues from the previous final_step ===\n"
            "next_final, next_log = ex1_train_one_epoch(model, opt, loader, loss_fn, start_step=final_step)\n"
            "assert next_final == 2 * n_batches, (\n"
            "    f'after 2 epochs of {n_batches} batches, final_step should be {2*n_batches}; got {next_final}'\n"
            ")\n"
            "next_steps_only = [s for s, _ in next_log]\n"
            "assert next_steps_only == list(range(n_batches + 1, 2 * n_batches + 1)), (\n"
            "    f'epoch 2 step values should be [{n_batches+1}, ..., {2*n_batches}]; got {next_steps_only}'\n"
            ")\n"
            "\n"
            "# === Grad gets cleared each batch (no accumulation) ===\n"
            "# After train_one_epoch, all params' .grad should be None.\n"
            "for p in model.parameters():\n"
            "    assert p.grad is None, (\n"
            "        f'after train_one_epoch, params should have .grad=None (zero_grad ran); got {p.grad}'\n"
            "    )\n"
            "\n"
            "# === Model actually updated (loss decreased across epoch 1) ===\n"
            "losses_epoch1 = [l for _, l in log]\n"
            "# Mean of first three vs last three batches.\n"
            "early = sum(losses_epoch1[:3]) / 3\n"
            "late = sum(losses_epoch1[-3:]) / 3\n"
            "assert late < early + 0.05, (\n"
            "    f'loss should not be increasing across epoch 1 (early={early:.4f}, late={late:.4f})'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_train_one_epoch(model, optimizer, loader, loss_fn, start_step):\n"
            "    step = start_step\n"
            "    log = []\n"
            "    for x, y in loader:\n"
            "        loss = loss_fn(model(x), y)\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "        optimizer.zero_grad()\n"
            "        step += 1\n"
            "        log.append((step, loss.item()))\n"
            "    return step, log"
        ),
        "solution_notes": (
            "**Why this placement matters.** Logging at step N should "
            "reflect 'N updates have been applied to the model.' If you "
            "log BEFORE incrementing, the first entry says step=0 — but "
            "an update DID happen before that line. The off-by-one ripples "
            "into every downstream graph: learning-rate schedules trigger "
            "one step late, checkpoints save the wrong state, wandb "
            "curves are misaligned across runs.\n\n"
            "**Why start_step is a parameter.** A training run typically "
            "calls `train_one_epoch` in a loop. Each epoch must continue "
            "the global counter, not reset to zero — otherwise you'd "
            "overwrite step-0 logs every epoch. Passing `start_step` "
            "explicitly makes the contract clear.\n\n"
            "**ARENA chap-3 specifically.** The transformer-training "
            "code in chap-3 follows this exact placement: `step += 1` "
            "after `optimizer.step() / optimizer.zero_grad()`, before "
            "any `wandb.log(...)` call. The Trainer pattern from "
            "trainer-class-skeleton (sibling drill) puts the same line "
            "inside `fit()`."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # examples-seen-step-axis — ex1
    # =========================================================
    {
        "atom_id": "examples-seen-step-axis",
        "subtopic": "Trainer: examples-seen step axis",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EXAMPLES_SEEN,
        "exercise_index": 1,
        "exercise_title": "compute examples_seen = step * batch_size as wandb x-axis",
        "slug": "compute-examples-seen-as-wandb-x-axis",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["examples-seen", "wandb-axis", "logging", "comparable-runs"],
        "kcs": [
            "examples-seen-equals-step-times-batch-size",
            "examples-seen-as-fair-cross-batch-size-x-axis",
        ],
        "lo": (
            "Apply `examples_seen = step * batch_size` inside a training "
            "loop so logging the same number of examples processed "
            "produces overlapping curves across different batch sizes."
        ),
        "prompt_body": (
            "Implement `ex1_log_examples_seen(loss_history, batch_size)`. "
            "Convert a list of `(step, loss)` log tuples into the "
            "wandb-friendly `(examples_seen, loss)` format.\n\n"
            "1. For each `(step, loss)` in `loss_history`, compute "
            "`examples_seen = step * batch_size`.\n"
            "2. Return a new list of `(examples_seen, loss)` tuples in "
            "the same order.\n\n"
            "Inputs:\n"
            "- `loss_history`: list of `(step, loss)` tuples — `step` is "
            "1-based int, `loss` is float.\n"
            "- `batch_size`: int — examples per batch.\n\n"
            "Output: list of `(examples_seen, loss)` tuples.\n\n"
            "Then implement `ex1_runs_match_at_examples_seen(history_a, "
            "history_b, batch_a, batch_b)`. The PAYOFF of this "
            "transformation: two runs with DIFFERENT batch sizes that "
            "have seen the same number of examples should align on the "
            "x-axis even though their step counts differ.\n\n"
            "Return `True` if the maximum `examples_seen` value matches "
            "across the two converted histories (within `batch_size` "
            "tolerance — partial last batches don't count exactly), "
            "`False` otherwise."
        ),
        "stub": (
            "def ex1_log_examples_seen(loss_history: list, batch_size: int) -> list:\n"
            '    """Convert [(step, loss)] -> [(examples_seen, loss)] where examples_seen = step * batch_size."""\n'
            "    raise NotImplementedError()\n"
            "\n"
            "\n"
            "def ex1_runs_match_at_examples_seen(history_a: list, history_b: list,\n"
            "                                    batch_a: int, batch_b: int) -> bool:\n"
            '    """Do the two runs cover ~the same examples_seen range?"""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Basic conversion ===\n"
            "history = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)]\n"
            "out = ex1_log_examples_seen(history, batch_size=32)\n"
            "expected = [(32, 1.0), (64, 0.9), (96, 0.8), (128, 0.7)]\n"
            "assert out == expected, f'expected {expected}, got {out}'\n"
            "\n"
            "# === Loss values preserved exactly (no rounding) ===\n"
            "history_f = [(1, 0.123456789), (2, 0.987654321)]\n"
            "out_f = ex1_log_examples_seen(history_f, batch_size=16)\n"
            "assert out_f[0][1] == 0.123456789 and out_f[1][1] == 0.987654321, (\n"
            "    f'loss values must be preserved exactly; got {out_f}'\n"
            ")\n"
            "\n"
            "# === Type check: examples_seen is int, loss is float ===\n"
            "for ex_seen, l in out:\n"
            "    assert isinstance(ex_seen, int), f'examples_seen must be int (step*batch_size); got {type(ex_seen)}'\n"
            "    assert isinstance(l, float), f'loss must remain float; got {type(l)}'\n"
            "\n"
            "# === Order preserved ===\n"
            "history_shuffled = [(5, 0.5), (1, 1.0), (3, 0.8)]\n"
            "out_shuffled = ex1_log_examples_seen(history_shuffled, batch_size=10)\n"
            "assert out_shuffled == [(50, 0.5), (10, 1.0), (30, 0.8)], (\n"
            "    f'order must match input; got {out_shuffled}'\n"
            ")\n"
            "\n"
            "# === Empty history ===\n"
            "assert ex1_log_examples_seen([], batch_size=64) == []\n"
            "\n"
            "# === Matching-runs payoff ===\n"
            "# Run A: batch_size=32, 10 batches → max examples_seen = 320.\n"
            "# Run B: batch_size=64,  5 batches → max examples_seen = 320.\n"
            "# Same examples seen even though step counts differ.\n"
            "hist_a = [(i, 1.0 / i) for i in range(1, 11)]\n"
            "hist_b = [(i, 1.0 / i) for i in range(1, 6)]\n"
            "assert ex1_runs_match_at_examples_seen(hist_a, hist_b, batch_a=32, batch_b=64) is True, (\n"
            "    'run A (10 steps * 32) and run B (5 steps * 64) both reach 320 examples — should match'\n"
            ")\n"
            "\n"
            "# Mismatched runs.\n"
            "hist_c = [(i, 1.0 / i) for i in range(1, 11)]   # 10 steps\n"
            "hist_d = [(i, 1.0 / i) for i in range(1, 11)]   # 10 steps\n"
            "assert ex1_runs_match_at_examples_seen(hist_c, hist_d, batch_a=32, batch_b=64) is False, (\n"
            "    'run C (10 * 32 = 320) and run D (10 * 64 = 640) should NOT match — different example counts'\n"
            ")\n"
            "\n"
            "# === Realistic scale ===\n"
            "big_history = [(s, 1.0 / (s + 1)) for s in range(1, 1001)]\n"
            "big_out = ex1_log_examples_seen(big_history, batch_size=128)\n"
            "assert len(big_out) == 1000\n"
            "assert big_out[-1][0] == 1000 * 128, f'last examples_seen wrong: {big_out[-1][0]}'\n"
            "assert big_out[0][0] == 128, f'first examples_seen wrong: {big_out[0][0]}'"
        ),
        "solution_body": (
            "def ex1_log_examples_seen(loss_history, batch_size):\n"
            "    return [(step * batch_size, loss) for step, loss in loss_history]\n"
            "\n"
            "\n"
            "def ex1_runs_match_at_examples_seen(history_a, history_b, batch_a, batch_b):\n"
            "    if not history_a or not history_b:\n"
            "        return False\n"
            "    max_a = max(s for s, _ in history_a) * batch_a\n"
            "    max_b = max(s for s, _ in history_b) * batch_b\n"
            "    tolerance = max(batch_a, batch_b)\n"
            "    return abs(max_a - max_b) <= tolerance"
        ),
        "solution_notes": (
            "**Why `step * batch_size` and not accumulating "
            "`x.shape[0]`.** For full-size batches they're identical. "
            "The accumulated form is exact when the last batch is "
            "partial (size < batch_size). For most training graphs the "
            "1% error in the final batch doesn't matter — the simpler "
            "`step * batch_size` is what ARENA's wandb integration "
            "uses.\n\n"
            "**When you'd reach for the accumulated form.** Curriculum "
            "learning, dataset streaming with variable batch sizes, or "
            "any setup where you NEED an exact example count (e.g. "
            "compute-budget comparisons across runs). For those, "
            "maintain `examples_seen += x.shape[0]` alongside `step += "
            "1` in the loop body.\n\n"
            "**Why this matters for cross-run comparison.** Compute-"
            "compare paper plots, scaling-law fits, and ablation grids "
            "all need a fair x-axis. Plotting against `step` privileges "
            "small-batch runs (they tick more steps per training-data "
            "epoch). Plotting against `examples_seen` makes runs of "
            "different batch sizes directly visually comparable — and "
            "that's exactly what wandb 'set x-axis' dropdown is for."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # optimizer-loop-on-tensor — ex1
    # =========================================================
    {
        "atom_id": "optimizer-loop-on-tensor",
        "subtopic": "Optimizer: optimizer.step loop over params",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_OPTIMIZER_LOOP_TENSOR,
        "exercise_index": 1,
        "exercise_title": "manual SGD step: for p in self.params: p -= lr * p.grad",
        "slug": "manual-sgd-step-for-p-in-self-params-p-minus-equals-lr-times-p-grad",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["sgd", "manual-step", "per-param-loop", "inference-mode"],
        "kcs": [
            "optimizer-step-explicit-for-loop-over-params",
            "guard-none-grad-from-frozen-params",
        ],
        "lo": (
            "Apply the explicit `for p in self.params: p -= self.lr * "
            "p.grad` loop inside a hand-rolled SGD optimizer's `step` "
            "method, with the `None`-grad guard for frozen / unused "
            "parameters."
        ),
        "prompt_body": (
            "Implement `Ex1ManualSGD`. A hand-rolled SGD with an "
            "explicit per-parameter loop.\n\n"
            "1. `__init__(self, params, lr)`: materialize "
            "`self.params = list(params)`, store `self.lr = lr`.\n"
            "2. `step(self)`: decorated with `@t.inference_mode()`. For "
            "each `p in self.params`:\n"
            "   - If `p.grad is None`, SKIP it (frozen or unused param).\n"
            "   - Else, update IN PLACE: `p -= self.lr * p.grad`.\n"
            "3. `zero_grad(self)`: set every `p.grad = None`.\n\n"
            "Inputs/outputs at the optimizer level match torch.optim "
            "conventions — no explicit return.\n\n"
            "The test passes a mix of TRAINABLE (`requires_grad=True`, "
            "has grad) and FROZEN (`requires_grad=False`, no grad) "
            "parameters to verify the `None`-grad guard. It also checks "
            "that the per-param loop visits EVERY param (not just the "
            "first), by giving each param a distinct gradient and "
            "verifying each moved."
        ),
        "stub": (
            "class Ex1ManualSGD:\n"
            '    """Hand-rolled SGD with explicit per-param loop."""\n'
            "\n"
            "    def __init__(self, params, lr: float):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def step(self):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def zero_grad(self):\n"
            "        raise NotImplementedError()"
        ),
        "test_body": (
            "# === Basic case: two trainable params, distinct gradients ===\n"
            "p_a = t.tensor([10.0, 20.0, 30.0], requires_grad=True)\n"
            "p_b = t.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)\n"
            "\n"
            "opt = Ex1ManualSGD([p_a, p_b], lr=0.1)\n"
            "assert isinstance(opt.params, list), 'params must be materialized to a list (handle generator inputs)'\n"
            "assert len(opt.params) == 2\n"
            "\n"
            "# Manually set grads (so we don't depend on a backward pass).\n"
            "p_a.grad = t.tensor([1.0, 2.0, 3.0])\n"
            "p_b.grad = t.tensor([[0.5, 0.5], [0.5, 0.5]])\n"
            "\n"
            "before_a = p_a.detach().clone()\n"
            "before_b = p_b.detach().clone()\n"
            "\n"
            "opt.step()    # must not raise — decorator is required\n"
            "\n"
            "# Expected: p_a = before_a - 0.1 * grad_a; same for p_b.\n"
            "expected_a = before_a - 0.1 * t.tensor([1.0, 2.0, 3.0])\n"
            "expected_b = before_b - 0.1 * t.tensor([[0.5, 0.5], [0.5, 0.5]])\n"
            "assert t.allclose(p_a.detach(), expected_a, atol=1e-6), (\n"
            "    f'p_a: expected {expected_a}, got {p_a.detach()}; '\n"
            "    f'check `p -= self.lr * p.grad`'\n"
            ")\n"
            "assert t.allclose(p_b.detach(), expected_b, atol=1e-6), (\n"
            "    f'p_b: expected {expected_b}, got {p_b.detach()}'\n"
            ")\n"
            "# Both params requires_grad still True.\n"
            "assert p_a.requires_grad and p_b.requires_grad\n"
            "\n"
            "# === None-grad guard: a frozen / unused param ===\n"
            "p_frozen = t.tensor([100.0, 100.0], requires_grad=True)\n"
            "# Don't call backward → grad is None.\n"
            "assert p_frozen.grad is None\n"
            "p_trainable = t.tensor([5.0, 5.0], requires_grad=True)\n"
            "p_trainable.grad = t.tensor([1.0, 1.0])\n"
            "\n"
            "opt2 = Ex1ManualSGD([p_frozen, p_trainable], lr=0.5)\n"
            "opt2.step()       # must NOT raise on the None-grad param\n"
            "\n"
            "# p_frozen should be UNCHANGED.\n"
            "assert t.allclose(p_frozen.detach(), t.tensor([100.0, 100.0])), (\n"
            "    f'param with None grad must be skipped; was modified to {p_frozen.detach()}; '\n"
            "    f'did you forget the `if p.grad is not None` guard?'\n"
            ")\n"
            "# p_trainable should have moved by lr * grad = 0.5 * 1.0 = 0.5.\n"
            "assert t.allclose(p_trainable.detach(), t.tensor([4.5, 4.5])), (\n"
            "    f'trainable param should have moved by lr * grad; got {p_trainable.detach()}'\n"
            ")\n"
            "\n"
            "# === zero_grad sets all grads to None ===\n"
            "opt.zero_grad()\n"
            "for p in opt.params:\n"
            "    assert p.grad is None, f'zero_grad must set .grad = None; got {p.grad}'\n"
            "\n"
            "# === Real model convergence check ===\n"
            "t.manual_seed(0)\n"
            "model = t.nn.Linear(2, 1)\n"
            "opt3 = Ex1ManualSGD(model.parameters(), lr=0.1)\n"
            "x = t.randn(32, 2)\n"
            "y = 2.0 * x[:, :1] - 3.0 * x[:, 1:2] + 1.0\n"
            "for _ in range(100):\n"
            "    loss = ((model(x) - y) ** 2).mean()\n"
            "    loss.backward()\n"
            "    opt3.step()\n"
            "    opt3.zero_grad()\n"
            "# Weight close to [2, -3], bias close to 1.\n"
            "w = model.weight.detach().flatten()\n"
            "b = model.bias.item()\n"
            "assert abs(w[0].item() - 2.0) < 0.1, f'w[0] should be ~2.0; got {w[0].item():.3f}'\n"
            "assert abs(w[1].item() + 3.0) < 0.1, f'w[1] should be ~-3.0; got {w[1].item():.3f}'\n"
            "assert abs(b - 1.0) < 0.1, f'bias should be ~1.0; got {b:.3f}'"
        ),
        "solution_body": (
            "class Ex1ManualSGD:\n"
            "    def __init__(self, params, lr):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "\n"
            "    @t.inference_mode()\n"
            "    def step(self):\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                p -= self.lr * p.grad\n"
            "\n"
            "    def zero_grad(self):\n"
            "        for p in self.params:\n"
            "            p.grad = None"
        ),
        "solution_notes": (
            "**Why the loop is explicit, not vectorized.** Each "
            "parameter is a tensor of a DIFFERENT shape (a "
            "`Linear(2,1)` has a `(1,2)` weight and a `(1,)` bias). You "
            "cannot subtract a single `lr * grad` block from all params "
            "at once without flattening — and flattening would lose the "
            "per-param shape needed for the next forward pass. The "
            "explicit `for p in self.params` is the canonical form.\n\n"
            "**Why guard `if p.grad is not None`.** Two cases produce a "
            "None grad: (1) `optimizer.zero_grad(set_to_none=True)` "
            "(the PyTorch default since 1.11) clears grads to None "
            "rather than zero; (2) a parameter that was never reached "
            "by the forward pass (e.g. a frozen feature extractor, an "
            "unused embedding row) has no `.grad` populated. Hitting "
            "`None * lr` would crash. The guard skips them cleanly.\n\n"
            "**`p -= ...` vs `p.data -= ...`.** With `@t.inference_mode"
            "()` the bare in-place update on a leaf is legal. Without "
            "the decorator you must reach for `p.data -= ...`. ARENA "
            "uses the decorator approach consistently — `torch.optim.SGD"
            "` source uses `@torch.no_grad()` which is functionally "
            "equivalent. PyTorch's optimized `_foreach_*` ops do the "
            "same thing but fused across params — they're a "
            "performance optimization of this same loop."
        ),
        "extra_imports": [],
    },
]


# ---------------------------------------------------------------------------
# Verifier — exec every (solution + test_body) in a fresh namespace.
# Aborts the whole build if any spec fails.
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    from torch import Tensor
    import einops
    from einops import rearrange, reduce, repeat

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"
        ns = {
            "t": t,
            "np": np,
            "Tensor": Tensor,
            "einops": einops,
            "rearrange": rearrange,
            "reduce": reduce,
            "repeat": repeat,
            "_dd_passed": set(),
            "__name__": "__main__",
        }
        t.manual_seed(0)
        np.random.seed(0)

        try:
            exec(spec["solution_body"], ns)
            exec(spec["test_body"], ns)
        except Exception as e:
            failed.append((tag, repr(e), traceback.format_exc()))
            continue
        passed += 1
        print(f"  [verify] {tag}: ok")

    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, err, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(err)
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[batch4] Verifying {len(SPECS)} specs against torch backend...")
    _verify_all(SPECS)

    print(f"\n[batch4] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[batch4] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
