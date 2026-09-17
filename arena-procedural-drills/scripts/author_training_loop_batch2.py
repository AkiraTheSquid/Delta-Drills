#!/usr/bin/env python3
"""Author 8 standalone Colab drills for the PyTorch training-loop family of atoms.

Atoms covered (each drill = ONE LO + ONE Bloom level, max 2 concurrent KCs):

  training-step-cycle           — 2 drills (ex1, ex2)
  zero-grad-set-none            — 1 drill  (ex1)
  backward-on-scalar-loss       — 1 drill  (ex1)
  optimizer-init-params-list    — 1 drill  (ex1)
  inplace-param-update          — 1 drill  (ex1)
  validation-no-grad            — 1 drill  (ex1)
  train-eval-mode-branch        — 1 drill  (ex1)

These are SMALLER skills that ARENA 0_3_* (optimization) and 0_2_10..12 (CNN
inference/training) assume the learner can already perform in isolation.

Each spec is verified by re-running its solution against its test_body inside
the build venv (torch 2.12.0+cpu) before emission. Any failure aborts the build.
"""
from __future__ import annotations

import sys
import textwrap
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_training_loop"

# ---------------------------------------------------------------------------
# Per-atom recap blocks. One per atom, reused across drills if the atom has
# multiple drills (currently only training-step-cycle).
# ---------------------------------------------------------------------------

RECAP_TRAINING_STEP_CYCLE = (
    "## The PyTorch training step cycle — quick refresher\n"
    "\n"
    "Every PyTorch training loop body, no matter how exotic the model, walks "
    "the same four-call cycle on each batch:\n"
    "\n"
    "```\n"
    "logits = model(x)             # 1. forward\n"
    "loss   = loss_fn(logits, y)   # 2. loss\n"
    "loss.backward()               # 3. backward → gradients into .grad\n"
    "optimizer.step()              # 4. apply update from .grad to params\n"
    "optimizer.zero_grad()         # 5. clear .grad so next batch starts fresh\n"
    "```\n"
    "\n"
    "**Order matters.** `.backward()` must come before `.step()` (no grads → "
    "no update). `.zero_grad()` must come after `.step()` (or before the next "
    "forward) — otherwise gradients from the previous batch accumulate into "
    "the next one. The default is gradient ACCUMULATION; `.zero_grad()` is "
    "what makes each batch independent."
)

RECAP_ZERO_GRAD = (
    "## `optimizer.zero_grad(set_to_none=True)` — quick refresher\n"
    "\n"
    "PyTorch accumulates gradients into `param.grad` on every `.backward()`. "
    "Without an explicit reset, the gradient from batch N would be ADDED to "
    "the one from batch N+1, producing a stale, oversized update. "
    "`optimizer.zero_grad()` is what makes mini-batch SGD actually behave like "
    "stochastic gradient descent.\n"
    "\n"
    "**`set_to_none=True` (default since PyTorch 1.7) vs `False`.** "
    "`set_to_none=True` replaces `.grad` with `None`; the next `.backward()` "
    "allocates a fresh gradient tensor. `set_to_none=False` keeps the buffer "
    "and writes zeros into it. `None` is faster (no kernel launch, no memory "
    "to clear) and lets autograd skip the addition in the next backward, but "
    "downstream code that reads `param.grad` must handle the `None` case."
)

RECAP_BACKWARD_SCALAR = (
    "## `loss.backward()` on a scalar — quick refresher\n"
    "\n"
    "`tensor.backward()` populates `.grad` on every leaf tensor with "
    "`requires_grad=True` that contributed to `tensor`. PyTorch requires the "
    "tensor to be a SCALAR (0-D / single-element) unless you pass a "
    "gradient-output tensor. Hence the universal training loop reduces the "
    "per-sample loss to a single number first — typically `.mean()` or "
    "`.sum()` — and only then calls `.backward()`.\n"
    "\n"
    "**Gradient accumulates.** Each call to `.backward()` ADDS into existing "
    "`.grad`. That's why the loop ends with `optimizer.zero_grad()` — to "
    "reset accumulation between batches.\n"
    "\n"
    "**Backward requires a graph.** If you wrap the forward in `t.no_grad()` "
    "or call `.detach()`, the resulting tensor has no `grad_fn` and "
    "`.backward()` will raise."
)

RECAP_OPTIMIZER_INIT = (
    "## `optim.SGD(params, lr=...)` — quick refresher\n"
    "\n"
    "An optimizer is constructed with an iterable of `nn.Parameter` tensors. "
    "Internally PyTorch's optimizer immediately materializes that iterable "
    "into a list — but if you ROLL YOUR OWN optimizer (as the ARENA SGD "
    "exercise does), you must do it yourself: `self.params = list(params)`. "
    "The reason: a generator can only be iterated once. If you store the "
    "generator and iterate it during `.step()`, the second call hits an "
    "empty iterator and silently does nothing.\n"
    "\n"
    "**The model.parameters() trap.** `model.parameters()` returns a "
    "generator. If your optimizer stores `self.params = params` instead of "
    "`list(params)`, the first `.step()` consumes it and every subsequent "
    "step is a no-op. Tests pass on iteration 1 and fail mysteriously on "
    "iteration 2."
)

RECAP_INPLACE_PARAM_UPDATE = (
    "## In-place parameter update — quick refresher\n"
    "\n"
    "When you hand-roll an optimizer, the parameter update MUST happen "
    "in-place. Two valid forms:\n"
    "\n"
    "```\n"
    "theta -= lr * g                # in-place op via -=\n"
    "theta.sub_(lr * g)             # in-place op via *_ method\n"
    "```\n"
    "\n"
    "**Why in-place.** A new-tensor update (`theta = theta - lr * g`) rebinds "
    "the LOCAL name to a brand-new tensor; the original `nn.Parameter` is "
    "untouched and the model still sees the old weights. The model holds a "
    "reference to the original storage; you have to mutate that storage.\n"
    "\n"
    "**Inside `@t.inference_mode()` / `@t.no_grad()`.** Optimizer `.step()` "
    "methods are decorated with `@t.no_grad()` (or `@t.inference_mode()`) so "
    "the in-place mutation doesn't get tracked by autograd. Without that, "
    "the next `.backward()` would error: 'a leaf Variable that requires grad "
    "is being used in an in-place operation.'"
)

RECAP_VALIDATION_NO_GRAD = (
    "## Validation under `torch.no_grad()` — quick refresher\n"
    "\n"
    "During validation/inference you don't need gradients — and computing "
    "them anyway wastes memory (the autograd graph is kept alive) and time "
    "(every op records its backward closure). `torch.no_grad()` is a "
    "context manager (or decorator) that disables gradient tracking for the "
    "code inside it. `torch.inference_mode()` is the newer, even-stricter "
    "version that additionally disables version counters.\n"
    "\n"
    "**Tensors produced inside `no_grad` have `requires_grad=False` and no "
    "`grad_fn`.** Calling `.backward()` on them raises. Calling `.item()` or "
    "`.numpy()` is fine.\n"
    "\n"
    "**Always pair with `model.eval()`.** `no_grad` controls autograd; "
    "`model.eval()` controls layer behavior (BatchNorm / Dropout). They are "
    "independent — you need both for honest validation."
)

RECAP_TRAIN_EVAL_MODE = (
    "## `model.train()` vs `model.eval()` — quick refresher\n"
    "\n"
    "Some layers behave DIFFERENTLY at train time vs eval time. Two big "
    "examples:\n"
    "\n"
    "- **Dropout** drops activations only when `model.training is True`. In "
    "eval mode dropout is a no-op (identity).\n"
    "- **BatchNorm** uses the current batch's mean/var in train mode and "
    "updates running statistics; in eval mode it uses the frozen running "
    "stats and updates nothing.\n"
    "\n"
    "`model.train()` and `model.eval()` flip the `self.training` flag "
    "RECURSIVELY on every submodule. Forgetting `model.eval()` at validation "
    "time is the #1 reason ARENA learners see 'my validation accuracy is "
    "wildly noisy and changes every run.'\n"
    "\n"
    "**Standalone of `no_grad`.** `model.train()/eval()` controls layer "
    "behavior. `torch.no_grad()` controls autograd. You need both for real "
    "validation; this drill isolates the train/eval flip."
)


# ---------------------------------------------------------------------------
# SPEC list. Each spec is one drill notebook. Field meanings live in
# `_emit_standalone.py`.
# ---------------------------------------------------------------------------

SPECS = [

    # =========================================================
    # training-step-cycle  —  ex1
    # =========================================================
    {
        "atom_id": "training-step-cycle",
        "subtopic": "PyTorch: Training step cycle",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_TRAINING_STEP_CYCLE,
        "exercise_index": 1,
        "exercise_title": "order the five calls of the canonical training step",
        "slug": "order-the-five-calls-of-the-canonical-training-step",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["training-loop", "step-order", "mini-batch-sgd"],
        "kcs": [
            "training-step-five-call-order",
            "training-step-zero-grad-resets-accumulation",
        ],
        "lo": (
            "Apply the canonical 5-call training-step cycle "
            "(forward → loss → backward → step → zero_grad) in the correct "
            "order against a 1-parameter model so that each step strictly "
            "decreases the loss."
        ),
        "prompt_body": (
            "Implement `ex1_train_one_param(w_init, x, y, lr, n_steps)`. "
            "Goal: fit a scalar weight `w` so that `w * x ≈ y` using vanilla "
            "SGD via the official `torch.optim.SGD` API. Concretely:\n\n"
            "1. Build `w = t.tensor([w_init], requires_grad=True)` — a "
            "1-element leaf with grad tracking.\n"
            "2. Build `optimizer = t.optim.SGD([w], lr=lr)`.\n"
            "3. Loop `n_steps` times. In each iteration do the 5 canonical "
            "calls IN ORDER:\n"
            "   - `pred = w * x`           (forward)\n"
            "   - `loss = ((pred - y) ** 2).mean()`  (loss)\n"
            "   - `loss.backward()`        (backward)\n"
            "   - `optimizer.step()`       (step)\n"
            "   - `optimizer.zero_grad()`  (zero_grad)\n"
            "4. Record `loss.item()` BEFORE `.backward()` so the test can "
            "verify the loss strictly decreases.\n"
            "5. Return `(w.detach().clone(), losses_list)`.\n\n"
            "Inputs:\n"
            "- `w_init`: float — starting weight value.\n"
            "- `x, y`: 1-D float tensors of equal length.\n"
            "- `lr`: float learning rate.\n"
            "- `n_steps`: int.\n\n"
            "Output: tuple `(w_final, losses)`. `w_final` is a detached "
            "1-element tensor. `losses` is a list of `n_steps` Python floats."
        ),
        "stub": (
            "def ex1_train_one_param(w_init: float, x: Tensor, y: Tensor,\n"
            "                        lr: float, n_steps: int) -> tuple:\n"
            '    """Fit w so that w*x ~= y. Returns (w_final, losses)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "y = t.tensor([2.0, 4.0, 6.0, 8.0])   # true w = 2.0\n"
            "w_final, losses = ex1_train_one_param(w_init=0.0, x=x, y=y, lr=0.05, n_steps=20)\n"
            "assert isinstance(losses, list), f'losses must be a list, got {type(losses)}'\n"
            "assert len(losses) == 20, f'expected 20 losses, got {len(losses)}'\n"
            "for i, lv in enumerate(losses):\n"
            "    assert isinstance(lv, float), f'losses[{i}] is {type(lv)}, must be float'\n"
            "# Monotonic decrease — proves the 5-call order is correct.\n"
            "# If zero_grad is missing or in the wrong spot, gradients\n"
            "# accumulate and loss diverges or oscillates.\n"
            "for i in range(1, len(losses)):\n"
            "    assert losses[i] <= losses[i - 1] + 1e-7, (\n"
            "        f'loss not decreasing at step {i}: {losses[i-1]:.6f} -> {losses[i]:.6f}; '\n"
            "        f'check call order (likely zero_grad missing or backward-after-step)'\n"
            "    )\n"
            "# After 20 steps with lr=0.05, w should be close to 2.0.\n"
            "assert abs(w_final.item() - 2.0) < 0.05, (\n"
            "    f'expected w ~ 2.0, got {w_final.item():.4f}; '\n"
            "    f'either step/backward order wrong or gradients not zeroed'\n"
            ")\n"
            "# First loss must equal mean((0*x - y)^2) = mean(y^2) = 30.0 exactly.\n"
            "assert abs(losses[0] - 30.0) < 1e-4, (\n"
            "    f'first loss should be 30.0 (mean of y^2 since w_init=0), got {losses[0]:.4f}; '\n"
            "    f'are you appending loss BEFORE backward()?'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_train_one_param(w_init, x, y, lr, n_steps):\n"
            "    w = t.tensor([w_init], requires_grad=True)\n"
            "    optimizer = t.optim.SGD([w], lr=lr)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        pred = w * x\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        losses.append(loss.item())   # snapshot BEFORE backward\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "        optimizer.zero_grad()\n"
            "    return w.detach().clone(), losses"
        ),
        "solution_notes": (
            "**Why snapshot the loss BEFORE backward.** `.item()` on the loss "
            "is read-only — its value is fixed at that point. Putting it "
            "after `.step()` would show the loss for the CURRENT weights "
            "AFTER the update, which still trends down but is a different "
            "(and confusing) curve.\n\n"
            "**The five calls form a single conceptual unit.** Mentally "
            "treat `forward / loss / backward / step / zero_grad` as one "
            "indivisible block. Every PyTorch training loop you'll ever "
            "write — Karpathy's, ARENA's, HuggingFace's — has this skeleton."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # training-step-cycle  —  ex2
    # =========================================================
    {
        "atom_id": "training-step-cycle",
        "subtopic": "PyTorch: Training step cycle",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_TRAINING_STEP_CYCLE,
        "exercise_index": 2,
        "exercise_title": "diagnose a buggy training step that forgets zero_grad",
        "slug": "diagnose-a-buggy-training-step-that-forgets-zero-grad",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["debug", "gradient-accumulation", "zero-grad-missing"],
        "kcs": [
            "training-step-zero-grad-resets-accumulation",
            "training-step-debug-via-loss-trajectory",
        ],
        "lo": (
            "Analyze a training loop that omits `optimizer.zero_grad()` and "
            "fix it by inserting the missing call in the correct position so "
            "the loss curve becomes monotonically decreasing."
        ),
        "prompt_body": (
            "Below is `train_buggy` — a training loop whose loss curve "
            "oscillates wildly and never gets close to zero. The bug is "
            "that it is missing `optimizer.zero_grad()`. Implement "
            "`ex2_train_fixed(w_init, x, y, lr, n_steps)` — the corrected "
            "version. Use exactly the same 5-call cycle as exercise 1, but "
            "this time return a tuple `(w_final, losses, buggy_losses)` "
            "where `buggy_losses` is what the buggy version produces on the "
            "same inputs.\n\n"
            "You must define `ex2_train_fixed` AND call the provided "
            "`train_buggy` to populate `buggy_losses`. The test then "
            "verifies your fixed version converges while the buggy one does "
            "not.\n\n"
            "Use the same problem as ex1 (`y = 2x`, lr=0.05).\n\n"
            "```python\n"
            "def train_buggy(w_init, x, y, lr, n_steps):\n"
            "    # BUG: never calls optimizer.zero_grad()\n"
            "    w = t.tensor([w_init], requires_grad=True)\n"
            "    optimizer = t.optim.SGD([w], lr=lr)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        pred = w * x\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        losses.append(loss.item())\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "        # <-- missing optimizer.zero_grad() here\n"
            "    return w.detach().clone(), losses\n"
            "```"
        ),
        "stub": (
            "def train_buggy(w_init, x, y, lr, n_steps):\n"
            "    w = t.tensor([w_init], requires_grad=True)\n"
            "    optimizer = t.optim.SGD([w], lr=lr)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        pred = w * x\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        losses.append(loss.item())\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "    return w.detach().clone(), losses\n"
            "\n"
            "\n"
            "def ex2_train_fixed(w_init: float, x: Tensor, y: Tensor,\n"
            "                    lr: float, n_steps: int) -> tuple:\n"
            '    """Return (w_final, losses, buggy_losses) — fixed + buggy for comparison."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "y = t.tensor([2.0, 4.0, 6.0, 8.0])\n"
            "w_final, losses, buggy_losses = ex2_train_fixed(0.0, x, y, lr=0.05, n_steps=15)\n"
            "assert len(losses) == 15 and len(buggy_losses) == 15\n"
            "# Fixed version converges monotonically.\n"
            "for i in range(1, 15):\n"
            "    assert losses[i] <= losses[i - 1] + 1e-7, (\n"
            "        f'fixed losses non-decreasing at step {i}: {losses[i-1]:.4f} -> {losses[i]:.4f}'\n"
            "    )\n"
            "assert abs(w_final.item() - 2.0) < 0.05, f'fixed run did not converge: w={w_final.item():.4f}'\n"
            "\n"
            "# Buggy version oscillates — many loss-INCREASES across the run,\n"
            "# because the un-zeroed gradient accumulates and overshoots,\n"
            "# then reverses sign, then overshoots back.\n"
            "buggy_increases = sum(1 for i in range(1, 15) if buggy_losses[i] > buggy_losses[i-1])\n"
            "assert buggy_increases >= 3, (\n"
            "    f'buggy version should have many loss INCREASES due to gradient accumulation '\n"
            "    f'oscillation; saw only {buggy_increases}/14. Did you accidentally also zero_grad '\n"
            "    f'in the buggy fn?'\n"
            ")\n"
            "# Buggy run is unable to converge below starting loss.\n"
            "assert buggy_losses[-1] > 1.0, (\n"
            "    f'buggy version should not reach near-zero loss; got final {buggy_losses[-1]:.6f}'\n"
            ")\n"
            "# Fixed version is dramatically better than buggy at the end.\n"
            "assert losses[-1] < buggy_losses[-1] / 100, (\n"
            "    f'fixed should be >>100x better than buggy at the end: '\n"
            "    f'fixed={losses[-1]:.6e}, buggy={buggy_losses[-1]:.4f}'\n"
            ")"
        ),
        "solution_body": (
            "def train_buggy(w_init, x, y, lr, n_steps):\n"
            "    # BUG: never calls optimizer.zero_grad()\n"
            "    w = t.tensor([w_init], requires_grad=True)\n"
            "    optimizer = t.optim.SGD([w], lr=lr)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        pred = w * x\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        losses.append(loss.item())\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "    return w.detach().clone(), losses\n"
            "\n"
            "\n"
            "def ex2_train_fixed(w_init, x, y, lr, n_steps):\n"
            "    # Run the buggy version first to capture its loss trajectory.\n"
            "    _, buggy_losses = train_buggy(w_init, x, y, lr, n_steps)\n"
            "\n"
            "    # Fixed version — adds the missing zero_grad after step.\n"
            "    w = t.tensor([w_init], requires_grad=True)\n"
            "    optimizer = t.optim.SGD([w], lr=lr)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        pred = w * x\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        losses.append(loss.item())\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "        optimizer.zero_grad()      # <-- the fix\n"
            "    return w.detach().clone(), losses, buggy_losses"
        ),
        "solution_notes": (
            "**Why the buggy version OSCILLATES instead of just stalling.** "
            "Without `zero_grad`, the gradient at step N is the SUM of "
            "every per-batch gradient computed so far. After the first "
            "step the (now oversized) accumulated gradient overshoots the "
            "minimum. At the new point on the loss surface the gradient "
            "points the OTHER way; that new gradient is added to the "
            "previous accumulated value, partially cancelling it. The "
            "result is a noisy ping-pong around the minimum — the loss "
            "neither converges to zero nor blows up to infinity, it just "
            "wanders. On harder problems with steeper curvature the same "
            "bug can outright diverge.\n\n"
            "**Diagnosis recipe.** When your training loss INCREASES at "
            "any step early in training, the first three things to check "
            "are (in order): (1) is `zero_grad` missing? (2) is the "
            "learning rate too high? (3) is the loss being aggregated "
            "correctly (e.g. summed instead of meaned across the batch)? "
            "The `zero_grad` bug has a distinctive signature: many "
            "non-monotone loss steps and a loss floor that never gets "
            "below the starting value."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # zero-grad-set-none  —  ex1
    # =========================================================
    {
        "atom_id": "zero-grad-set-none",
        "subtopic": "PyTorch: zero_grad",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_ZERO_GRAD,
        "exercise_index": 1,
        "exercise_title": "implement zero_grad with set_to_none semantics",
        "slug": "implement-zero-grad-with-set-to-none-semantics",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["zero-grad", "set-to-none", "hand-rolled-optimizer"],
        "kcs": [
            "zero-grad-set-to-none-semantics",
            "zero-grad-iterates-params-list",
        ],
        "lo": (
            "Apply the `set_to_none=True` convention by iterating a "
            "parameter list and setting each `.grad` attribute to `None`, "
            "matching PyTorch's default zero_grad behavior."
        ),
        "prompt_body": (
            "Implement `ex1_zero_grad(params)`. This is the body of a "
            "hand-rolled optimizer's `.zero_grad()` method, using the "
            "modern `set_to_none=True` convention.\n\n"
            "1. Iterate `params` (a list of `nn.Parameter` / leaf tensors "
            "with `requires_grad=True`).\n"
            "2. For each `p`, set `p.grad = None`.\n"
            "3. Do NOT use `.detach()`, do NOT use `torch.zeros_like`, do "
            "NOT call `.zero_()` on existing grad buffers. The point is the "
            "None convention.\n\n"
            "After calling `ex1_zero_grad(params)` every parameter must "
            "have `p.grad is None`. After the NEXT `.backward()`, PyTorch "
            "will allocate a fresh `.grad` tensor automatically.\n\n"
            "No return value — the function mutates the params in place."
        ),
        "stub": (
            "def ex1_zero_grad(params: list) -> None:\n"
            '    """Set every param.grad = None (set_to_none=True semantics)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build three leaf tensors with gradients populated.\n"
            "p1 = t.zeros(3, requires_grad=True)\n"
            "p2 = t.zeros(2, 2, requires_grad=True)\n"
            "p3 = t.zeros(5, requires_grad=True)\n"
            "loss = (p1.sum() + p2.sum() + p3.sum())\n"
            "loss.backward()\n"
            "# Sanity — grads are not None yet.\n"
            "for p in (p1, p2, p3):\n"
            "    assert p.grad is not None, 'precondition: grads should exist'\n"
            "\n"
            "# Call the user's zero_grad.\n"
            "ex1_zero_grad([p1, p2, p3])\n"
            "\n"
            "for i, p in enumerate((p1, p2, p3)):\n"
            "    assert p.grad is None, (\n"
            "        f'p{i+1}.grad is {p.grad!r}, expected None (set_to_none=True semantics); '\n"
            "        f'did you call p.grad.zero_() or assign torch.zeros_like(p) instead?'\n"
            "    )\n"
            "\n"
            "# A fresh backward should re-allocate.\n"
            "loss2 = p1.sum() + p2.sum() + p3.sum()\n"
            "loss2.backward()\n"
            "for i, p in enumerate((p1, p2, p3)):\n"
            "    assert p.grad is not None, f'after second backward, p{i+1}.grad should be a fresh tensor'\n"
            "    assert t.allclose(p.grad, t.ones_like(p)), (\n"
            "        f'p{i+1}.grad after re-backward should be all-ones, got {p.grad}; '\n"
            "        f'a leftover grad would have shown 2x because of accumulation'\n"
            "    )\n"
            "\n"
            "# Empty list must be a no-op (not raise).\n"
            "ex1_zero_grad([])"
        ),
        "solution_body": (
            "def ex1_zero_grad(params):\n"
            "    for p in params:\n"
            "        p.grad = None"
        ),
        "solution_notes": (
            "**Why the modern convention is `None`, not zeros.** Before "
            "PyTorch 1.7 the default was `set_to_none=False` — `.grad` was "
            "kept and zeroed. Two reasons it changed:\n\n"
            "1. Memory: a `None` grad releases the tensor; the next backward "
            "allocates fresh and PyTorch's autograd can sometimes avoid "
            "creating it at all (e.g. parameters not used in the current "
            "batch).\n"
            "2. Speed: `p.grad = None` is a pointer assignment (free). "
            "`p.grad.zero_()` launches a CUDA kernel.\n\n"
            "**Downstream consequence.** Code that reads `p.grad` AFTER a "
            "`zero_grad(set_to_none=True)` and BEFORE the next backward must "
            "handle `None` — typical pattern is `if p.grad is not None: ...`. "
            "Optimizers internally do exactly this in their `.step()` loop."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # backward-on-scalar-loss  —  ex1
    # =========================================================
    {
        "atom_id": "backward-on-scalar-loss",
        "subtopic": "PyTorch: backward()",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_BACKWARD_SCALAR,
        "exercise_index": 1,
        "exercise_title": "reduce per-sample loss to a scalar before backward",
        "slug": "reduce-per-sample-loss-to-a-scalar-before-backward",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["backward", "scalar-reduction", "mean-vs-sum"],
        "kcs": [
            "backward-requires-scalar-loss",
            "backward-populates-leaf-grad",
        ],
        "lo": (
            "Apply scalar reduction (`.mean()`) to a per-sample loss vector "
            "and call `.backward()` on the scalar so that the leaf "
            "parameter's `.grad` is populated correctly."
        ),
        "prompt_body": (
            "Implement `ex1_backward_mean(w, x, y)`. A common ARENA pattern: "
            "compute a per-sample squared-error loss, reduce to a scalar "
            "with `.mean()`, and call `.backward()`.\n\n"
            "1. Compute `per_sample_loss = (w * x - y) ** 2`. This is a "
            "vector with the same shape as `x`.\n"
            "2. Reduce to a scalar with `.mean()` → `scalar_loss`.\n"
            "3. Call `scalar_loss.backward()`.\n"
            "4. Return the tuple `(scalar_loss, per_sample_loss)`. The test "
            "verifies (a) the scalar is correct, (b) `w.grad` is correct, "
            "(c) per-sample loss has the right shape.\n\n"
            "Inputs:\n"
            "- `w`: a leaf tensor with `requires_grad=True`, shape `(1,)`.\n"
            "- `x, y`: 1-D float tensors of equal length.\n\n"
            "**Common trap.** Calling `.backward()` directly on the "
            "per-sample loss tensor (without reducing) raises "
            "`RuntimeError: grad can be implicitly created only for scalar "
            "outputs`. The scalar reduction is mandatory."
        ),
        "stub": (
            "def ex1_backward_mean(w: Tensor, x: Tensor, y: Tensor) -> tuple:\n"
            '    """Return (scalar_loss, per_sample_loss) after backward()."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "y = t.tensor([2.0, 4.0, 6.0, 8.0])\n"
            "w = t.tensor([0.0], requires_grad=True)\n"
            "scalar_loss, per_sample = ex1_backward_mean(w, x, y)\n"
            "\n"
            "# Per-sample loss shape and values.\n"
            "assert per_sample.shape == x.shape, (\n"
            "    f'per_sample_loss shape {tuple(per_sample.shape)} != x shape {tuple(x.shape)}'\n"
            ")\n"
            "expected_per_sample = t.tensor([4.0, 16.0, 36.0, 64.0])  # (0*x - y)^2\n"
            "assert t.allclose(per_sample.detach(), expected_per_sample), (\n"
            "    f'per_sample wrong: got {per_sample.detach()}, expected {expected_per_sample}'\n"
            ")\n"
            "\n"
            "# Scalar loss = mean of per_sample = 30.0.\n"
            "assert scalar_loss.dim() == 0, (\n"
            "    f'scalar_loss must be 0-D; got shape {tuple(scalar_loss.shape)}; '\n"
            "    f'did you forget to call .mean()?'\n"
            ")\n"
            "assert abs(scalar_loss.item() - 30.0) < 1e-5, (\n"
            "    f'scalar_loss = {scalar_loss.item()}, expected 30.0'\n"
            ")\n"
            "\n"
            "# Most important: w.grad must be populated by .backward().\n"
            "assert w.grad is not None, (\n"
            "    'w.grad is None — did you call .backward() on a tensor without grad_fn? '\n"
            "    'or never call .backward() at all?'\n"
            ")\n"
            "# Analytic gradient: d/dw mean((w*x - y)^2) = mean(2 * x * (w*x - y))\n"
            "# at w=0: = mean(2 * x * (-y)) = mean(-2 * x * y)\n"
            "# = -2 * (1*2 + 2*4 + 3*6 + 4*8) / 4 = -2 * (2+8+18+32)/4 = -2 * 15 = -30.0\n"
            "expected_grad = t.tensor([-30.0])\n"
            "assert t.allclose(w.grad, expected_grad), (\n"
            "    f'w.grad = {w.grad.item()}, expected {expected_grad.item()}; '\n"
            "    f'did you use .sum() instead of .mean()? sum would give -120.0'\n"
            ")\n"
            "\n"
            "# Demonstrate that backward on a NON-scalar raises.\n"
            "w2 = t.tensor([0.0], requires_grad=True)\n"
            "bad_loss = (w2 * x - y) ** 2   # vector — not reduced\n"
            "raised = False\n"
            "try:\n"
            "    bad_loss.backward()\n"
            "except RuntimeError as e:\n"
            "    raised = True\n"
            "    assert 'scalar' in str(e).lower() or 'gradient' in str(e).lower(), (\n"
            "        f'unexpected error message: {e}'\n"
            "    )\n"
            "assert raised, 'calling .backward() on a vector loss must raise RuntimeError'"
        ),
        "solution_body": (
            "def ex1_backward_mean(w, x, y):\n"
            "    per_sample_loss = (w * x - y) ** 2\n"
            "    scalar_loss = per_sample_loss.mean()\n"
            "    scalar_loss.backward()\n"
            "    return scalar_loss, per_sample_loss"
        ),
        "solution_notes": (
            "**Mean vs sum changes the gradient magnitude.** "
            "`.mean()` divides by N, so the gradient is the average per-"
            "sample gradient. `.sum()` is N× larger, which acts like an "
            "effective lr of `lr * N`. Karpathy's micrograd/nano-GPT both "
            "use `.mean()`; that's the default convention for `nn.MSELoss` "
            "and `nn.CrossEntropyLoss` too (`reduction='mean'`).\n\n"
            "**The 'implicit scalar' check is fundamental.** Internally "
            "PyTorch's `.backward()` creates a 1.0 gradient at the output "
            "of the graph and propagates from there. For a scalar that's "
            "unambiguous. For a vector PyTorch refuses to guess; you'd have "
            "to pass `gradient=t.ones_like(vec)` explicitly, which is "
            "advanced. Production training loops never do this — they "
            "always reduce."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # optimizer-init-params-list  —  ex1
    # =========================================================
    {
        "atom_id": "optimizer-init-params-list",
        "subtopic": "PyTorch: Optimizer init",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_OPTIMIZER_INIT,
        "exercise_index": 1,
        "exercise_title": "materialize a generator of params into a list at init",
        "slug": "materialize-a-generator-of-params-into-a-list-at-init",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["optimizer-init", "generator", "list-materialization"],
        "kcs": [
            "optimizer-init-list-vs-generator",
            "optimizer-init-stores-params-attribute",
        ],
        "lo": (
            "Analyze why `self.params = list(params)` is required in a "
            "hand-rolled optimizer's `__init__` and implement the fix so "
            "that the optimizer survives being passed a generator."
        ),
        "prompt_body": (
            "You are given a `BuggyOptimizer` whose `__init__` stores the "
            "raw `params` iterable (often a generator). On the second call "
            "to `.step()` the optimizer silently does nothing because the "
            "generator was already consumed.\n\n"
            "Implement `Ex1FixedOptimizer.__init__(self, params, lr)` to "
            "fix the bug. The contract:\n\n"
            "1. Materialize the iterable: `self.params = list(params)`.\n"
            "2. Store `self.lr = lr`.\n"
            "3. Provide a `.step()` method that does an in-place vanilla "
            "SGD update for every param that has a non-None `.grad`:\n"
            "   `p.data -= self.lr * p.grad`.\n"
            "4. Provide a `.zero_grad()` method that sets every "
            "`p.grad = None`.\n\n"
            "The test verifies:\n"
            "- The fixed optimizer works when given a `model.parameters()` "
            "generator (the canonical PyTorch pattern).\n"
            "- The fixed optimizer's `.step()` actually mutates params on "
            "the SECOND call (the buggy one fails this).\n"
            "- The buggy version is demonstrably broken on the second "
            "`.step()` for the same input.\n\n"
            "Decorate `.step()` with `@t.no_grad()` so the in-place "
            "mutation doesn't pollute the autograd graph."
        ),
        "stub": (
            "class BuggyOptimizer:\n"
            "    # BUG: stores generator as-is\n"
            "    def __init__(self, params, lr):\n"
            "        self.params = params   # <- never materialized to a list\n"
            "        self.lr = lr\n"
            "\n"
            "    @t.no_grad()\n"
            "    def step(self):\n"
            "        for p in self.params:           # generator: empty on 2nd call\n"
            "            if p.grad is not None:\n"
            "                p.data -= self.lr * p.grad\n"
            "\n"
            "    def zero_grad(self):\n"
            "        for p in self.params:\n"
            "            p.grad = None\n"
            "\n"
            "\n"
            "class Ex1FixedOptimizer:\n"
            '    """Optimizer that materializes `params` into a list at init."""\n'
            "\n"
            "    def __init__(self, params, lr):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    @t.no_grad()\n"
            "    def step(self):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def zero_grad(self):\n"
            "        raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a tiny model and pass its PARAMETER GENERATOR to the optimizers.\n"
            "model = t.nn.Linear(3, 1, bias=False)\n"
            "with t.no_grad():\n"
            "    model.weight.copy_(t.zeros_like(model.weight))\n"
            "\n"
            "# Sanity: model.parameters() is indeed a generator.\n"
            "import types\n"
            "params_gen = model.parameters()\n"
            "assert isinstance(params_gen, types.GeneratorType), (\n"
            "    f'model.parameters() should be a generator; got {type(params_gen)}'\n"
            "    f'  (PyTorch version mismatch?)'\n"
            ")\n"
            "\n"
            "fixed = Ex1FixedOptimizer(model.parameters(), lr=0.1)\n"
            "# After construction, the FIXED optimizer must store a LIST.\n"
            "assert isinstance(fixed.params, list), (\n"
            "    f'fixed.params must be a list, got {type(fixed.params)}; '\n"
            "    f'did you forget list(params)?'\n"
            ")\n"
            "assert len(fixed.params) == 1, f'expected 1 param, got {len(fixed.params)}'\n"
            "\n"
            "# Do two real training steps; weight should change BOTH times.\n"
            "x = t.tensor([[1.0, 2.0, 3.0]])\n"
            "y = t.tensor([[14.0]])  # true weight = [1, 2, 3]\n"
            "\n"
            "snapshots = [model.weight.detach().clone()]\n"
            "for _ in range(2):\n"
            "    loss = ((model(x) - y) ** 2).mean()\n"
            "    loss.backward()\n"
            "    fixed.step()\n"
            "    fixed.zero_grad()\n"
            "    snapshots.append(model.weight.detach().clone())\n"
            "\n"
            "# Both updates must actually move the weights.\n"
            "delta_1 = (snapshots[1] - snapshots[0]).abs().sum().item()\n"
            "delta_2 = (snapshots[2] - snapshots[1]).abs().sum().item()\n"
            "assert delta_1 > 1e-5, f'first step did not update weights: delta={delta_1}'\n"
            "assert delta_2 > 1e-5, (\n"
            "    f'second step did not update weights: delta={delta_2}; '\n"
            "    f'this is exactly the generator-consumed bug — did you wrap params in list()?'\n"
            ")\n"
            "\n"
            "# Now demonstrate the bug by running the BUGGY optimizer on a fresh model.\n"
            "model2 = t.nn.Linear(3, 1, bias=False)\n"
            "with t.no_grad():\n"
            "    model2.weight.copy_(t.zeros_like(model2.weight))\n"
            "buggy = BuggyOptimizer(model2.parameters(), lr=0.1)\n"
            "snap2 = [model2.weight.detach().clone()]\n"
            "for _ in range(2):\n"
            "    loss = ((model2(x) - y) ** 2).mean()\n"
            "    loss.backward()\n"
            "    buggy.step()\n"
            "    buggy.zero_grad()\n"
            "    snap2.append(model2.weight.detach().clone())\n"
            "delta_1b = (snap2[1] - snap2[0]).abs().sum().item()\n"
            "delta_2b = (snap2[2] - snap2[1]).abs().sum().item()\n"
            "assert delta_1b > 1e-5, 'buggy step 1 should still work (generator not yet drained)'\n"
            "assert delta_2b < 1e-9, (\n"
            "    f'buggy step 2 should be a no-op (generator consumed), but moved by {delta_2b}; '\n"
            "    f'are you sure you are testing the buggy version?'\n"
            ")"
        ),
        "solution_body": (
            "class BuggyOptimizer:\n"
            "    # BUG: stores generator as-is\n"
            "    def __init__(self, params, lr):\n"
            "        self.params = params\n"
            "        self.lr = lr\n"
            "\n"
            "    @t.no_grad()\n"
            "    def step(self):\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                p.data -= self.lr * p.grad\n"
            "\n"
            "    def zero_grad(self):\n"
            "        for p in self.params:\n"
            "            p.grad = None\n"
            "\n"
            "\n"
            "class Ex1FixedOptimizer:\n"
            "    def __init__(self, params, lr):\n"
            "        self.params = list(params)   # <-- the critical fix\n"
            "        self.lr = lr\n"
            "\n"
            "    @t.no_grad()\n"
            "    def step(self):\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                p.data -= self.lr * p.grad\n"
            "\n"
            "    def zero_grad(self):\n"
            "        for p in self.params:\n"
            "            p.grad = None"
        ),
        "solution_notes": (
            "**Why this bug is so insidious.** It survives one full step. "
            "Loss does decrease on iteration 1. The optimizer 'works' for "
            "exactly one batch. Tests that use a single-step smoke check "
            "will pass. Then every subsequent step is a silent no-op and "
            "the loss curve flatlines.\n\n"
            "**PyTorch's built-in optimizers already do this for you.** "
            "`torch.optim.SGD.__init__` calls `param_groups = list(params)` "
            "internally — that's why you never see this bug with the "
            "official API. The trap only matters for hand-rolled optimizers, "
            "which is exactly what ARENA chapter 0 part 3 asks you to "
            "write.\n\n"
            "**General Python lesson.** Any constructor that takes an "
            "`Iterable[T]` and intends to iterate it MORE THAN ONCE must "
            "materialize it: `list(it)`, `tuple(it)`, or `dict(it)`. The "
            "type annotation `Iterable` is the warning sign."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # inplace-param-update  —  ex1
    # =========================================================
    {
        "atom_id": "inplace-param-update",
        "subtopic": "PyTorch: In-place param update",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_INPLACE_PARAM_UPDATE,
        "exercise_index": 1,
        "exercise_title": "in-place vs out-of-place parameter update",
        "slug": "in-place-vs-out-of-place-parameter-update",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["in-place", "rebind", "data-attribute"],
        "kcs": [
            "inplace-update-mutates-storage",
            "out-of-place-update-rebinds-local-name",
        ],
        "lo": (
            "Analyze why `theta -= lr * g` (in-place) correctly updates a "
            "model's parameters but `theta = theta - lr * g` (out-of-place) "
            "does not, by comparing the post-update tensor identities."
        ),
        "prompt_body": (
            "Implement two functions that BOTH look like an SGD parameter "
            "update — but only ONE actually changes the model's weights. "
            "The test confirms the failure mode of the wrong one and the "
            "success of the right one.\n\n"
            "**Implement `ex1_apply_update_inplace(param, grad, lr)`:**\n"
            "- Mutate `param` IN PLACE: `param.data -= lr * grad`.\n"
            "- Return nothing.\n"
            "- After the call, the same `param` tensor still exists in its "
            "containing module — only its underlying storage is updated.\n\n"
            "**Implement `ex1_apply_update_wrong(param, grad, lr)`:**\n"
            "- Do the out-of-place version: `param = param - lr * grad` "
            "(simple Python rebind of the LOCAL name).\n"
            "- This is the bug we're isolating. Return the rebound local "
            "name so the test can compare identities.\n\n"
            "Inputs:\n"
            "- `param`: an `nn.Parameter` (or leaf tensor) wrapped by a "
            "module — the test passes a real `nn.Linear`'s `.weight`.\n"
            "- `grad`: a tensor with the same shape as `param`.\n"
            "- `lr`: float.\n\n"
            "The test then constructs a 1-layer `nn.Linear`, calls each "
            "function on its weight, and checks (a) `model.weight is` the "
            "same Python object before and after either call, (b) only the "
            "in-place version actually changed the underlying values."
        ),
        "stub": (
            "def ex1_apply_update_inplace(param: Tensor, grad: Tensor, lr: float) -> None:\n"
            '    """In-place SGD update: mutate param.data, return None."""\n'
            "    raise NotImplementedError()\n"
            "\n"
            "\n"
            "def ex1_apply_update_wrong(param: Tensor, grad: Tensor, lr: float):\n"
            '    """Out-of-place (broken) SGD update: rebind local name, return new tensor."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a tiny model with a known weight.\n"
            "model = t.nn.Linear(2, 1, bias=False)\n"
            "with t.no_grad():\n"
            "    model.weight.copy_(t.tensor([[1.0, 2.0]]))\n"
            "before_ptr = model.weight.data_ptr()\n"
            "before_id = id(model.weight)\n"
            "grad = t.tensor([[0.1, 0.2]])\n"
            "\n"
            "# === IN-PLACE update ===\n"
            "ret = ex1_apply_update_inplace(model.weight, grad, lr=1.0)\n"
            "assert ret is None, f'in-place function should return None, got {ret!r}'\n"
            "# Weight identity preserved.\n"
            "assert id(model.weight) == before_id, 'in-place must not rebind model.weight'\n"
            "# Underlying storage preserved (same data_ptr).\n"
            "assert model.weight.data_ptr() == before_ptr, (\n"
            "    'in-place must not reallocate storage'\n"
            ")\n"
            "# Values actually changed.\n"
            "expected = t.tensor([[0.9, 1.8]])  # [1, 2] - 1.0 * [0.1, 0.2]\n"
            "assert t.allclose(model.weight.data, expected), (\n"
            "    f'in-place update wrong: got {model.weight.data}, expected {expected}'\n"
            ")\n"
            "\n"
            "# === OUT-OF-PLACE (broken) update on a fresh model ===\n"
            "model2 = t.nn.Linear(2, 1, bias=False)\n"
            "with t.no_grad():\n"
            "    model2.weight.copy_(t.tensor([[1.0, 2.0]]))\n"
            "before_id2 = id(model2.weight)\n"
            "before_values2 = model2.weight.data.clone()\n"
            "\n"
            "returned = ex1_apply_update_wrong(model2.weight, grad, lr=1.0)\n"
            "# The local rebind returned a NEW tensor with the updated values.\n"
            "assert returned is not None, 'wrong-update function must return the rebound tensor'\n"
            "expected2 = t.tensor([[0.9, 1.8]])\n"
            "assert t.allclose(returned.detach(), expected2), (\n"
            "    f'returned tensor wrong: got {returned}, expected {expected2}'\n"
            ")\n"
            "# Critical: the MODEL is unchanged. This is the bug.\n"
            "assert id(model2.weight) == before_id2, 'model.weight identity should be unchanged'\n"
            "assert t.allclose(model2.weight.data, before_values2), (\n"
            "    f'model.weight should NOT have changed but did: {model2.weight.data} vs {before_values2}; '\n"
            "    f'are you sure the function only does `param = param - lr * grad`?'\n"
            ")\n"
            "# And the returned tensor is a DIFFERENT Python object.\n"
            "assert returned is not model2.weight, (\n"
            "    'returned tensor must be a new object, not the original parameter'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_apply_update_inplace(param, grad, lr):\n"
            "    # `param.data` lets us mutate the storage outside the autograd graph.\n"
            "    # Equivalent inside a no_grad/inference_mode block: `param -= lr * grad`.\n"
            "    param.data -= lr * grad\n"
            "\n"
            "\n"
            "def ex1_apply_update_wrong(param, grad, lr):\n"
            "    # This rebinds the LOCAL `param` name to a new tensor.\n"
            "    # The model still holds a reference to the original — unchanged.\n"
            "    param = param - lr * grad\n"
            "    return param"
        ),
        "solution_notes": (
            "**Why `.data` instead of `param -= lr * grad`.** Outside a "
            "`no_grad` context, the bare in-place op on a leaf with "
            "`requires_grad=True` raises: 'a leaf Variable that requires "
            "grad is being used in an in-place operation.' Real PyTorch "
            "optimizers wrap `.step()` in `@t.no_grad()` (or "
            "`@t.inference_mode()`) so they can write `param -= lr * grad` "
            "without the `.data` workaround. Using `.data` is the "
            "battle-tested escape hatch when you want to mutate weights "
            "from outside autograd's view.\n\n"
            "**The Python aliasing model is the real lesson.** In Python, "
            "`x = x + 1` rebinds the local name `x` — it does not mutate "
            "the original object. `x += 1` calls `__iadd__` which mutates "
            "in place for mutable types (lists, tensors). Tensor parameters "
            "live inside `nn.Module` containers via attribute reference; "
            "you have to mutate the storage to be seen.\n\n"
            "**ARENA SGD impl gotcha.** The solution comment literally "
            "reads `theta -= self.lr * g  # inplace operation, to modify "
            "params`. This is THE most-explained line in the chap-0 part-3 "
            "optimizer exercise. The drill here isolates the lesson."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # validation-no-grad  —  ex1
    # =========================================================
    {
        "atom_id": "validation-no-grad",
        "subtopic": "PyTorch: no_grad validation",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_VALIDATION_NO_GRAD,
        "exercise_index": 1,
        "exercise_title": "validation pass wrapped in torch.no_grad",
        "slug": "validation-pass-wrapped-in-torch-no-grad",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["validation", "no-grad", "inference"],
        "kcs": [
            "no-grad-context-manager-usage",
            "no-grad-disables-grad-fn",
        ],
        "lo": (
            "Apply `torch.no_grad()` as a context manager around an "
            "inference pass so that produced tensors have no `grad_fn` and "
            "`.backward()` cannot be called on them."
        ),
        "prompt_body": (
            "Implement `ex1_validate(model, x, y)`. A textbook validation "
            "pass.\n\n"
            "1. Open a `with t.no_grad():` block.\n"
            "2. Inside, run `logits = model(x)`.\n"
            "3. Inside, compute `loss = ((logits - y) ** 2).mean()`.\n"
            "4. Inside, extract `loss_val = loss.item()`.\n"
            "5. Return the tuple `(loss_val, logits)`. (Both must come "
            "from inside the `no_grad` block.)\n\n"
            "Inputs:\n"
            "- `model`: an `nn.Module`.\n"
            "- `x, y`: same-shape input/target tensors.\n\n"
            "**Critical behavior under no_grad.** Any tensor produced "
            "inside the block has `requires_grad=False` and `grad_fn=None`. "
            "Calling `.backward()` on `loss` (returned from inside the "
            "block) must raise `RuntimeError`. The test verifies this."
        ),
        "stub": (
            "def ex1_validate(model: t.nn.Module, x: Tensor, y: Tensor) -> tuple:\n"
            '    """Validation pass under no_grad. Returns (loss_val, logits)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# A trivial linear model.\n"
            "model = t.nn.Linear(3, 2, bias=False)\n"
            "with t.no_grad():\n"
            "    model.weight.copy_(t.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))\n"
            "x = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n"
            "y = t.tensor([[1.0, 2.0], [4.0, 5.0]])\n"
            "\n"
            "loss_val, logits = ex1_validate(model, x, y)\n"
            "\n"
            "assert isinstance(loss_val, float), (\n"
            "    f'loss_val must be a Python float (from .item()), got {type(loss_val)}'\n"
            ")\n"
            "assert abs(loss_val) < 1e-6, (\n"
            "    f'loss should be ~0 because logits exactly match y, got {loss_val}'\n"
            ")\n"
            "\n"
            "# logits must have been produced inside no_grad.\n"
            "assert logits.requires_grad is False, (\n"
            "    'logits.requires_grad must be False — was the forward inside `with t.no_grad():` ?'\n"
            ")\n"
            "assert logits.grad_fn is None, (\n"
            "    f'logits.grad_fn must be None inside no_grad; got {logits.grad_fn}'\n"
            ")\n"
            "\n"
            "# Backward on a tensor produced inside no_grad must raise.\n"
            "raised = False\n"
            "try:\n"
            "    fake_loss = logits.sum()\n"
            "    fake_loss.backward()\n"
            "except RuntimeError as e:\n"
            "    raised = True\n"
            "    assert 'grad' in str(e).lower() or 'requires_grad' in str(e).lower() or 'leaf' in str(e).lower(), (\n"
            "        f'unexpected error: {e}'\n"
            "    )\n"
            "assert raised, 'tensors from no_grad must not be backward-able'\n"
            "\n"
            "# Outside no_grad the model is still gradient-tracked.\n"
            "logits_train = model(x)\n"
            "assert logits_train.requires_grad is True, (\n"
            "    'after exiting no_grad, model output must again require grad'\n"
            ")\n"
            "assert logits_train.grad_fn is not None"
        ),
        "solution_body": (
            "def ex1_validate(model, x, y):\n"
            "    with t.no_grad():\n"
            "        logits = model(x)\n"
            "        loss = ((logits - y) ** 2).mean()\n"
            "        loss_val = loss.item()\n"
            "        return loss_val, logits"
        ),
        "solution_notes": (
            "**Why `no_grad` matters for memory.** Every op inside a "
            "gradient-tracked forward stashes the intermediate activations "
            "it needs to compute the backward. For a ResNet-50 forward at "
            "batch 32, that's hundreds of megabytes. `no_grad` skips the "
            "stashing entirely. For validation on a held-out test set this "
            "is a free 5-10× memory reduction.\n\n"
            "**`no_grad` vs `inference_mode`.** `inference_mode` is "
            "PyTorch's stricter newer version. It additionally disables "
            "version counters and forbids any later `requires_grad=True` "
            "use of the produced tensors. Faster, but slightly less "
            "ergonomic — you can't accidentally re-enter training with a "
            "tensor that was made under it. ARENA uses both interchangeably.\n\n"
            "**Not a substitute for `model.eval()`.** This drill targets "
            "only the autograd-control half of validation. The other half — "
            "switching BatchNorm/Dropout into eval mode — is a separate "
            "drill (`train-eval-mode-branch`). You need both for honest "
            "validation."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # train-eval-mode-branch  —  ex1
    # =========================================================
    {
        "atom_id": "train-eval-mode-branch",
        "subtopic": "PyTorch: train/eval mode",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_TRAIN_EVAL_MODE,
        "exercise_index": 1,
        "exercise_title": "flip train and eval mode around dropout",
        "slug": "flip-train-and-eval-mode-around-dropout",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["train-mode", "eval-mode", "dropout"],
        "kcs": [
            "model-train-eval-flips-training-flag",
            "dropout-active-only-when-training",
        ],
        "lo": (
            "Apply `model.train()` and `model.eval()` around a forward pass "
            "of a dropout-containing network so that dropout is stochastic "
            "during training and deterministic during evaluation."
        ),
        "prompt_body": (
            "Implement `ex1_train_then_eval(model, x, n_repeats)`. The "
            "function isolates the train/eval flip on a network that "
            "contains `nn.Dropout`.\n\n"
            "1. Call `model.train()`. Repeatedly forward `x` through "
            "`model` `n_repeats` times — collect every output into a list. "
            "Because dropout is active and uses fresh masks each call, "
            "consecutive outputs will DIFFER.\n"
            "2. Call `model.eval()`. Repeatedly forward `x` through "
            "`model` `n_repeats` times — collect every output. Because "
            "dropout is a no-op in eval mode, every output is IDENTICAL.\n"
            "3. Return `(train_outputs, eval_outputs, was_training_before, "
            "is_training_after)` where `was_training_before` is "
            "`model.training` BEFORE the function does anything, and "
            "`is_training_after` is `model.training` after `model.eval()`.\n\n"
            "Inputs:\n"
            "- `model`: an `nn.Module` containing at least one "
            "`nn.Dropout` layer with `p > 0`.\n"
            "- `x`: input tensor.\n"
            "- `n_repeats`: how many forward passes per mode.\n\n"
            "The test builds a small Sequential with a `nn.Dropout(p=0.5)` "
            "in the middle and checks both the variance behavior and the "
            "`model.training` flag transitions."
        ),
        "stub": (
            "def ex1_train_then_eval(model: t.nn.Module, x: Tensor, n_repeats: int) -> tuple:\n"
            '    """Forward under train/eval, return (train_outs, eval_outs, was_training, is_training_after)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a Sequential: Linear -> Dropout -> Linear.\n"
            "t.manual_seed(42)\n"
            "model = t.nn.Sequential(\n"
            "    t.nn.Linear(4, 8),\n"
            "    t.nn.Dropout(p=0.5),\n"
            "    t.nn.Linear(8, 2),\n"
            ")\n"
            "# Force the model into a known initial mode.\n"
            "model.train()                      # initial state: training\n"
            "x = t.randn(3, 4)\n"
            "\n"
            "train_outs, eval_outs, was_training, is_training_after = (\n"
            "    ex1_train_then_eval(model, x, n_repeats=5)\n"
            ")\n"
            "\n"
            "# Mode-flag bookkeeping.\n"
            "assert was_training is True, (\n"
            "    f'model was set to train mode before the call but was_training={was_training}; '\n"
            "    f'did you check model.training BEFORE calling .train()/.eval() ?'\n"
            ")\n"
            "assert is_training_after is False, (\n"
            "    f'after model.eval() the flag must be False, got {is_training_after}'\n"
            ")\n"
            "assert model.training is False, (\n"
            "    f'after the function exits, model.training should still be False, got {model.training}'\n"
            ")\n"
            "# Recursive flip — every submodule should also be in eval mode.\n"
            "for m in model.modules():\n"
            "    assert m.training is False, (\n"
            "        f'submodule {m.__class__.__name__}.training is True; '\n"
            "        f'model.eval() should propagate recursively'\n"
            "    )\n"
            "\n"
            "# Output lists.\n"
            "assert len(train_outs) == 5 and len(eval_outs) == 5\n"
            "for o in train_outs + eval_outs:\n"
            "    assert isinstance(o, Tensor)\n"
            "    assert o.shape == (3, 2), f'output shape wrong: {tuple(o.shape)}'\n"
            "\n"
            "# Train-mode dropout produces VARYING outputs across repeats.\n"
            "train_stack = t.stack(train_outs)\n"
            "train_variance_across_repeats = train_stack.std(dim=0).sum().item()\n"
            "assert train_variance_across_repeats > 0.01, (\n"
            "    f'train-mode outputs barely vary (std-sum={train_variance_across_repeats:.6f}); '\n"
            "    f'dropout should make them differ — did you forget model.train() ?'\n"
            ")\n"
            "\n"
            "# Eval-mode dropout produces IDENTICAL outputs across repeats.\n"
            "eval_stack = t.stack(eval_outs)\n"
            "eval_variance_across_repeats = eval_stack.std(dim=0).sum().item()\n"
            "assert eval_variance_across_repeats < 1e-6, (\n"
            "    f'eval-mode outputs should be identical (std-sum={eval_variance_across_repeats:.6e}); '\n"
            "    f'did you forget model.eval() ?'\n"
            ")\n"
            "# Sanity — all eval outputs equal each other.\n"
            "for i in range(1, 5):\n"
            "    assert t.allclose(eval_outs[0], eval_outs[i]), (\n"
            "        f'eval_outs[0] != eval_outs[{i}]'\n"
            "    )"
        ),
        "solution_body": (
            "def ex1_train_then_eval(model, x, n_repeats):\n"
            "    was_training = model.training       # snapshot first\n"
            "    model.train()\n"
            "    train_outs = [model(x) for _ in range(n_repeats)]\n"
            "    model.eval()\n"
            "    eval_outs = [model(x) for _ in range(n_repeats)]\n"
            "    is_training_after = model.training\n"
            "    return train_outs, eval_outs, was_training, is_training_after"
        ),
        "solution_notes": (
            "**`model.training` is the toggle.** `.train(mode=True)` and "
            "`.eval()` (which is `.train(mode=False)`) walk every "
            "submodule and set `.training = mode`. Layers consult their "
            "own `self.training` flag in `forward` and branch accordingly. "
            "Dropout, BatchNorm, RMSNorm-with-tracked-stats, and any "
            "custom layer that wants train/eval differences read this "
            "flag.\n\n"
            "**Why the flip is so common in ARENA debugging.** The most "
            "frequent ARENA training-loop bug is calling `model.eval()` "
            "during validation but forgetting to call `model.train()` "
            "again before the next epoch's training loop. The model "
            "trains on FROZEN batchnorm running stats from then on; "
            "training proceeds without error but accuracy plateaus far "
            "below what it should. Always pair them — `model.train()` "
            "at the start of the train loop, `model.eval()` at the start "
            "of validation."
        ),
        "extra_imports": [],
    },
]


# ---------------------------------------------------------------------------
# Verifier — execs every (stub-imports + solution_body + test_body) in a
# fresh namespace. Aborts the whole build if any spec fails.
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
        # Build the namespace exactly as the notebook would after running setup +
        # the SOLUTION (not the stub). Tests assert against the solution.
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
        # Reset seeds for determinism, like the notebook does.
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
    print(f"[batch2] Verifying {len(SPECS)} specs against torch backend...")
    _verify_all(SPECS)

    print(f"\n[batch2] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[batch2] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
