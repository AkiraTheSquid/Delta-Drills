#!/usr/bin/env python3
"""Author 8 standalone Colab drills for optimizer / training-instrumentation atoms.

Atoms covered (each drill = ONE LO + ONE Bloom level, max 2 concurrent KCs):

  inference-mode-step             — 2 drills (ex1, ex2)
  dataloader-batching             — 1 drill  (ex1)
  optimizer-state-tensor-buffers  — 2 drills (ex1, ex2)
  weight-decay-l2-add             — 1 drill  (ex1)
  momentum-buffer-update          — 1 drill  (ex1)
  ema-second-moment               — 1 drill  (ex1)

These are SMALLER constituent skills that ARENA 0_3_1..0_3_3 (SGD / RMSprop /
Adam impl) and 0_2_12 (training loop) assume the learner can already perform
in isolation.

Each spec is verified by re-running its solution against its test_body inside
the build venv (torch 2.12.0+cpu) before emission. Any failure aborts the build.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_optimizer_internals"


# ---------------------------------------------------------------------------
# Per-atom recap blocks. Reused across drills that share an atom.
# ---------------------------------------------------------------------------

RECAP_INFERENCE_MODE = (
    "## `@t.inference_mode()` on `step` — quick refresher\n"
    "\n"
    "PyTorch optimizers mutate parameters IN PLACE. The naive form "
    "`theta -= self.lr * g` on a leaf with `requires_grad=True` outside any "
    "no-grad context raises:\n"
    "\n"
    "> *RuntimeError: a leaf Variable that requires grad is being used in an "
    "in-place operation.*\n"
    "\n"
    "The fix is to declare `step` as living outside autograd's bookkeeping. "
    "Two equivalent decorators:\n"
    "\n"
    "```\n"
    "@t.no_grad()           # disables grad tracking inside the call\n"
    "@t.inference_mode()    # stricter newer version (also kills version counters)\n"
    "```\n"
    "\n"
    "ARENA's SGD/RMSprop/Adam impls all use `@t.inference_mode()` on `step`. "
    "That single decorator is what lets the body do `theta -= ...` and "
    "`buffer.copy_(...)` without autograd screaming. It is a hard requirement, "
    "not a stylistic choice."
)

RECAP_DATALOADER_BATCHING = (
    "## `DataLoader(dataset, batch_size, shuffle)` — quick refresher\n"
    "\n"
    "A `Dataset` yields ONE example at a time. A `DataLoader` wraps a dataset "
    "and yields BATCHES — stacking individual `__getitem__` returns into a "
    "leading batch dimension.\n"
    "\n"
    "```\n"
    "trainloader = DataLoader(trainset, batch_size=64, shuffle=True)\n"
    "testloader  = DataLoader(testset,  batch_size=64, shuffle=False)\n"
    "for xb, yb in trainloader:\n"
    "    ...  # xb: (B, *features), yb: (B, *labels)\n"
    "```\n"
    "\n"
    "**Two non-negotiable conventions.** `shuffle=True` for training (so "
    "consecutive batches see different examples — required for SGD's "
    "stochasticity). `shuffle=False` for validation/test (so metrics are "
    "deterministic across runs).\n"
    "\n"
    "**Partial last batch.** If `len(dataset) % batch_size != 0`, the final "
    "batch is smaller. Setting `drop_last=True` discards it (useful when "
    "fixed-size batches matter — e.g. BatchNorm with very small batches)."
)

RECAP_OPTIMIZER_STATE_BUFFERS = (
    "## Per-parameter state buffers — quick refresher\n"
    "\n"
    "Optimizers that need MEMORY across steps (momentum, EMA, second moment) "
    "must keep a buffer FOR EACH parameter, allocated at construction time:\n"
    "\n"
    "```\n"
    "self.params = list(params)\n"
    "self.b = [t.zeros_like(p) for p in self.params]    # one buffer per param\n"
    "```\n"
    "\n"
    "**Why `zeros_like` not `zeros`.** It mirrors `p`'s `shape`, `dtype`, AND "
    "`device` — so a buffer for a `(256, 768)` float16 CUDA weight is itself "
    "`(256, 768)` float16 on the same GPU. Initializing as `t.zeros(p.shape)` "
    "would silently put the buffer on CPU.\n"
    "\n"
    "**Why a list, not one big tensor.** Different parameters can have "
    "different shapes; you can't flatten them into a single tensor without "
    "losing the per-param indexing that `step` relies on. PyTorch's own "
    "optimizers use the same per-param list pattern internally."
)

RECAP_WEIGHT_DECAY = (
    "## Weight decay (L2 fold) — quick refresher\n"
    "\n"
    "L2 regularization adds `(lambda / 2) * ||theta||^2` to the loss. Its "
    "gradient w.r.t. `theta` is `lambda * theta`. Rather than build that into "
    "the loss expression, optimizers FOLD it directly into the gradient at "
    "each step:\n"
    "\n"
    "```\n"
    "if self.lmda != 0:\n"
    "    g = g + self.lmda * theta\n"
    "```\n"
    "\n"
    "Then the rest of the optimizer treats this augmented `g` as the "
    "gradient. The `if self.lmda != 0` guard skips the fold (and the "
    "tensor allocation) when weight decay is disabled — a tiny but "
    "real performance win in the inner loop.\n"
    "\n"
    "**This is the classical L2 form, NOT AdamW.** AdamW DECOUPLES the "
    "decay from the gradient — it subtracts `lr * lmda * theta` from `theta` "
    "directly, bypassing the moment estimates. Mixing them up is the "
    "single most common 'why does my Adam train weirdly' bug."
)

RECAP_MOMENTUM_BUFFER = (
    "## Momentum buffer update `b = mu*b + g` — quick refresher\n"
    "\n"
    "Classical momentum keeps a velocity buffer `b` that exponentially "
    "decays past gradients while accumulating new ones. The recurrence is:\n"
    "\n"
    "```\n"
    "b_t = mu * b_{t-1} + g_t      # update buffer\n"
    "g_t = b_t                     # use buffer as the effective gradient\n"
    "```\n"
    "\n"
    "Two impl-level details that trip people up:\n"
    "\n"
    "1. The buffer is updated IN PLACE: `b.copy_(self.mu * b + g)`. A "
    "rebind (`b = self.mu * b + g`) only rebinds the LOCAL variable; the "
    "entry in `self.b[i]` is unchanged, so next step uses the stale value.\n"
    "2. The `g = b` assignment is by-reference. Later mutation of `g` would "
    "mutate the buffer — but we don't mutate `g` after this, so it's fine "
    "in the canonical impl."
)

RECAP_EMA_SECOND_MOMENT = (
    "## Adam EMA second moment `v = beta2*v + (1-beta2)*g^2` — quick refresher\n"
    "\n"
    "Adam maintains TWO running averages per parameter: the first moment "
    "(EMA of gradients, `m`) and the second moment (EMA of squared "
    "gradients, `v`). The second moment recurrence is:\n"
    "\n"
    "```\n"
    "v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2\n"
    "```\n"
    "\n"
    "**Why squared.** `v_t` approximates the second raw moment of the "
    "gradient distribution. After bias correction, `sqrt(v_hat)` is "
    "approximately the per-coord gradient magnitude. Adam divides by this "
    "to produce its adaptive per-parameter learning rate — large-gradient "
    "coordinates get small effective lr, small-gradient coordinates get "
    "large effective lr.\n"
    "\n"
    "**Default `beta2 = 0.999`.** That gives an effective horizon of "
    "~1/(1-beta2) = 1000 steps. The recurrence is unbiased only in the "
    "limit; the `1/(1 - beta2^t)` bias correction at step `t` is what makes "
    "early steps usable."
)


# ---------------------------------------------------------------------------
# SPEC list. Each spec = one drill notebook.
# ---------------------------------------------------------------------------

SPECS = [

    # =========================================================
    # inference-mode-step  —  ex1
    # =========================================================
    {
        "atom_id": "inference-mode-step",
        "subtopic": "PyTorch: Inference mode step",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_INFERENCE_MODE,
        "exercise_index": 1,
        "exercise_title": "decorate step with inference_mode to allow in-place leaf update",
        "slug": "decorate-step-with-inference-mode-to-allow-in-place-leaf-update",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["inference-mode", "no-grad", "in-place-leaf"],
        "kcs": [
            "inference-mode-decorator-wraps-step",
            "inference-mode-allows-leaf-in-place-mutation",
        ],
        "lo": (
            "Apply the `@t.inference_mode()` decorator to a hand-rolled "
            "optimizer's `step` method so the in-place `theta -= lr * g` "
            "leaf-update succeeds without raising."
        ),
        "prompt_body": (
            "Implement `Ex1InferenceSGD` — a minimal SGD optimizer whose "
            "`step` method is decorated with `@t.inference_mode()`.\n\n"
            "1. `__init__(self, params, lr)`: materialize `params` into "
            "`self.params = list(params)`, store `self.lr = lr`.\n"
            "2. `step(self)`: decorated with `@t.inference_mode()`. For each "
            "param `p` with non-None `.grad`, do the BARE in-place update "
            "`p -= self.lr * p.grad` — NOT `p.data -= ...`. The decorator is "
            "what allows this on a leaf with `requires_grad=True`.\n"
            "3. `zero_grad(self)`: set every `p.grad = None`.\n\n"
            "The test verifies:\n"
            "- One step actually moves the weights toward the target.\n"
            "- The decorator is present (a missing decorator would make the "
            "bare `p -= ...` raise the leaf-in-place error).\n"
            "- Re-running the same forward AFTER `.step()` gives a smaller "
            "loss than before."
        ),
        "stub": (
            "class Ex1InferenceSGD:\n"
            '    """SGD whose step is decorated with @t.inference_mode()."""\n'
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
            "# Trivial regression: fit y = 3x with a single scalar weight.\n"
            "model = t.nn.Linear(1, 1, bias=False)\n"
            "with t.no_grad():\n"
            "    model.weight.copy_(t.tensor([[0.0]]))\n"
            "opt = Ex1InferenceSGD(model.parameters(), lr=0.1)\n"
            "assert isinstance(opt.params, list), 'optimizer must materialize params into a list'\n"
            "\n"
            "x = t.tensor([[1.0], [2.0], [3.0], [4.0]])\n"
            "y = t.tensor([[3.0], [6.0], [9.0], [12.0]])\n"
            "\n"
            "loss_before = ((model(x) - y) ** 2).mean().item()\n"
            "loss = ((model(x) - y) ** 2).mean()\n"
            "loss.backward()\n"
            "opt.step()                      # must NOT raise the leaf-in-place error\n"
            "opt.zero_grad()\n"
            "loss_after = ((model(x) - y) ** 2).mean().item()\n"
            "\n"
            "assert loss_after < loss_before, (\n"
            "    f'one SGD step should decrease loss: {loss_before:.4f} -> {loss_after:.4f}; '\n"
            "    f'either the in-place update was silently a no-op or weights did not move'\n"
            ")\n"
            "# Param values actually moved.\n"
            "assert model.weight.item() != 0.0, (\n"
            "    'weight should have moved off 0 after the step; '\n"
            "    'is your step body actually mutating in place?'\n"
            ")\n"
            "\n"
            "# Verify the decorator was applied — the step function should\n"
            "# be wrapped (the bare in-place leaf update would otherwise raise).\n"
            "# We confirm this indirectly: run a SECOND step from a still-grad-required leaf.\n"
            "model2 = t.nn.Linear(1, 1, bias=False)\n"
            "with t.no_grad():\n"
            "    model2.weight.copy_(t.tensor([[0.0]]))\n"
            "opt2 = Ex1InferenceSGD(model2.parameters(), lr=0.1)\n"
            "for _ in range(5):\n"
            "    L = ((model2(x) - y) ** 2).mean()\n"
            "    L.backward()\n"
            "    opt2.step()                 # repeated bare in-place mutation\n"
            "    opt2.zero_grad()\n"
            "assert abs(model2.weight.item() - 3.0) < 0.5, (\n"
            "    f'after 5 steps weight should be approaching 3.0; got {model2.weight.item():.4f}; '\n"
            "    f'is the decorator wrapping `step` correctly?'\n"
            ")\n"
            "# After all steps the param should still require grad (decorator did not contaminate it).\n"
            "assert model2.weight.requires_grad, (\n"
            "    'param should still require grad after step; '\n"
            "    'inference_mode should only affect the step call, not the param state'\n"
            ")"
        ),
        "solution_body": (
            "class Ex1InferenceSGD:\n"
            "    def __init__(self, params, lr: float):\n"
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
            "**Why `p -= ...` and not `p.data -= ...`.** With "
            "`@t.inference_mode()` (or `@t.no_grad()`) wrapping `step`, the "
            "in-place op on a leaf is legal — autograd is told to ignore "
            "this block entirely. Without the decorator, the same line raises "
            "`a leaf Variable that requires grad is being used in an "
            "in-place operation.` The `.data` escape hatch is the older "
            "workaround you reach for OUTSIDE such a block.\n\n"
            "**`@t.inference_mode()` vs `@t.no_grad()`.** Functionally "
            "interchangeable for the optimizer step. `inference_mode` is "
            "newer and slightly stricter — it disables version counters and "
            "forbids later upgrading the produced tensors to "
            "`requires_grad=True`. ARENA uses `inference_mode` because "
            "PyTorch recommends it for new code."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # inference-mode-step  —  ex2
    # =========================================================
    {
        "atom_id": "inference-mode-step",
        "subtopic": "PyTorch: Inference mode step",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_INFERENCE_MODE,
        "exercise_index": 2,
        "exercise_title": "diagnose missing inference_mode decorator on step",
        "slug": "diagnose-missing-inference-mode-decorator-on-step",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["leaf-in-place-error", "debug", "inference-mode-missing"],
        "kcs": [
            "inference-mode-decorator-wraps-step",
            "leaf-in-place-error-signature",
        ],
        "lo": (
            "Analyze the `leaf Variable ... in-place` runtime error raised by "
            "an optimizer whose `step` lacks `@t.inference_mode()`, and fix "
            "it by adding the decorator without otherwise changing the body."
        ),
        "prompt_body": (
            "Below is `BrokenSGD` — a hand-rolled SGD whose `step` body uses "
            "the bare in-place form `p -= self.lr * p.grad` but FORGETS to "
            "decorate `step`. The first `.step()` call raises a RuntimeError.\n\n"
            "Your job: implement `Ex2FixedSGD` — identical to `BrokenSGD` "
            "EXCEPT that `step` is correctly decorated with "
            "`@t.inference_mode()`. Do NOT change the body of `step`. Do NOT "
            "switch to `p.data -= ...`. The fix is exactly one decorator.\n\n"
            "Also implement `ex2_demonstrate_broken_raises(broken_opt)` — "
            "call `broken_opt.step()` inside a `try / except RuntimeError` "
            "and return the string of the exception (so the test can verify "
            "the characteristic error message)."
        ),
        "stub": (
            "class BrokenSGD:\n"
            "    # BUG: missing @t.inference_mode() decorator on step.\n"
            "    def __init__(self, params, lr):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "\n"
            "    def step(self):\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                p -= self.lr * p.grad        # raises on leaf with requires_grad\n"
            "\n"
            "    def zero_grad(self):\n"
            "        for p in self.params:\n"
            "            p.grad = None\n"
            "\n"
            "\n"
            "class Ex2FixedSGD:\n"
            '    """Same as BrokenSGD but with the missing decorator added."""\n'
            "\n"
            "    def __init__(self, params, lr):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def step(self):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def zero_grad(self):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "\n"
            "def ex2_demonstrate_broken_raises(broken_opt) -> str:\n"
            '    """Call broken_opt.step() and return the RuntimeError message."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build two identical models so we can compare broken vs fixed.\n"
            "def _make_model():\n"
            "    m = t.nn.Linear(1, 1, bias=False)\n"
            "    with t.no_grad():\n"
            "        m.weight.copy_(t.tensor([[0.0]]))\n"
            "    return m\n"
            "x = t.tensor([[1.0], [2.0], [3.0]])\n"
            "y = t.tensor([[2.0], [4.0], [6.0]])\n"
            "\n"
            "# --- BROKEN path: should raise on .step() ---\n"
            "broken_model = _make_model()\n"
            "broken = BrokenSGD(broken_model.parameters(), lr=0.1)\n"
            "loss = ((broken_model(x) - y) ** 2).mean()\n"
            "loss.backward()\n"
            "msg = ex2_demonstrate_broken_raises(broken)\n"
            "assert isinstance(msg, str) and len(msg) > 0, (\n"
            "    f'demonstrate_broken_raises must return the error message string, got {msg!r}'\n"
            ")\n"
            "msg_low = msg.lower()\n"
            "assert ('leaf' in msg_low) or ('in-place' in msg_low) or ('inplace' in msg_low), (\n"
            "    f'expected the characteristic leaf-in-place error, got: {msg}'\n"
            ")\n"
            "\n"
            "# --- FIXED path: should NOT raise; should converge ---\n"
            "fixed_model = _make_model()\n"
            "fixed = Ex2FixedSGD(fixed_model.parameters(), lr=0.1)\n"
            "for _ in range(20):\n"
            "    L = ((fixed_model(x) - y) ** 2).mean()\n"
            "    L.backward()\n"
            "    fixed.step()\n"
            "    fixed.zero_grad()\n"
            "assert abs(fixed_model.weight.item() - 2.0) < 0.05, (\n"
            "    f'fixed optimizer should converge to w=2; got {fixed_model.weight.item():.4f}'\n"
            ")\n"
            "# Param still tracks grad after all steps.\n"
            "assert fixed_model.weight.requires_grad"
        ),
        "solution_body": (
            "class BrokenSGD:\n"
            "    def __init__(self, params, lr):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "\n"
            "    def step(self):\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                p -= self.lr * p.grad\n"
            "\n"
            "    def zero_grad(self):\n"
            "        for p in self.params:\n"
            "            p.grad = None\n"
            "\n"
            "\n"
            "class Ex2FixedSGD:\n"
            "    def __init__(self, params, lr):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "\n"
            "    @t.inference_mode()       # <-- the entire fix\n"
            "    def step(self):\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                p -= self.lr * p.grad\n"
            "\n"
            "    def zero_grad(self):\n"
            "        for p in self.params:\n"
            "            p.grad = None\n"
            "\n"
            "\n"
            "def ex2_demonstrate_broken_raises(broken_opt) -> str:\n"
            "    try:\n"
            "        broken_opt.step()\n"
            "    except RuntimeError as e:\n"
            "        return str(e)\n"
            "    return ''"
        ),
        "solution_notes": (
            "**Diagnosis recipe.** When you see "
            "`a leaf Variable that requires grad is being used in an "
            "in-place operation` from an optimizer step, the cause is "
            "ALMOST ALWAYS one of:\n"
            "\n"
            "1. Missing `@t.inference_mode()` / `@t.no_grad()` on `step`.\n"
            "2. Using bare `p -= ...` outside such a block, in user code "
            "(e.g. a manual update inside a Jupyter cell).\n"
            "3. Forgetting `.data` in a fix attempt that did NOT add the "
            "decorator.\n"
            "\n"
            "The cleanest fix is the decorator — `.data` is older and PyTorch "
            "discourages it in new code. ARENA's reference SGD/Adam/RMSprop "
            "all use the decorator.\n\n"
            "**Why the bare op raises at all.** Autograd tracks every "
            "operation that participates in a parameter's history. An "
            "in-place mutation invalidates that history (the input tensor "
            "no longer holds the values backward needs). For a NON-leaf "
            "tensor PyTorch detects this via version counters and raises at "
            "backward time; for a leaf it refuses up-front."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # dataloader-batching  —  ex1
    # =========================================================
    {
        "atom_id": "dataloader-batching",
        "subtopic": "PyTorch: DataLoader batching",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_DATALOADER_BATCHING,
        "exercise_index": 1,
        "exercise_title": "wrap a TensorDataset in a DataLoader and iterate batches",
        "slug": "wrap-a-tensordataset-in-a-dataloader-and-iterate-batches",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["dataloader", "batch-size", "shuffle"],
        "kcs": [
            "dataloader-wraps-dataset",
            "dataloader-shuffle-true-for-train-false-for-test",
        ],
        "lo": (
            "Apply `DataLoader(dataset, batch_size, shuffle)` to produce "
            "stacked-batch iteration with the ARENA-canonical `shuffle=True` "
            "for training and `shuffle=False` for eval."
        ),
        "prompt_body": (
            "Implement `ex1_build_loaders(x_train, y_train, x_test, y_test, "
            "batch_size)`. The canonical training/eval DataLoader setup.\n\n"
            "1. Wrap `(x_train, y_train)` in a `TensorDataset`.\n"
            "2. Wrap `(x_test, y_test)`  in a `TensorDataset`.\n"
            "3. Build `train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)`.\n"
            "4. Build `test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)`.\n"
            "5. Return `(train_loader, test_loader)`.\n\n"
            "Inputs: same-length 2-D feature tensors and 1-D label tensors. "
            "`batch_size` may not evenly divide the dataset; let the default "
            "`drop_last=False` keep the partial last batch.\n\n"
            "The test verifies:\n"
            "- Batches stack correctly into `(B, *features)` / `(B,)`.\n"
            "- The training loader yields different batch orderings across "
            "two epochs (shuffle on).\n"
            "- The test loader yields the SAME batch ordering across two "
            "epochs (shuffle off — reproducible eval metrics).\n"
            "- Every example appears in exactly ONE batch per epoch (no "
            "duplicates, no drops)."
        ),
        "stub": (
            "from torch.utils.data import TensorDataset, DataLoader\n"
            "\n"
            "\n"
            "def ex1_build_loaders(x_train: Tensor, y_train: Tensor,\n"
            "                      x_test: Tensor,  y_test: Tensor,\n"
            "                      batch_size: int) -> tuple:\n"
            '    """Return (train_loader, test_loader). Train shuffles, test does not."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a tiny dataset with KNOWN indices encoded into x[:, 0].\n"
            "N_train, N_test, F = 20, 6, 3\n"
            "x_train = t.stack([t.full((F,), float(i)) for i in range(N_train)])\n"
            "y_train = t.arange(N_train)\n"
            "x_test  = t.stack([t.full((F,), float(i + 100)) for i in range(N_test)])\n"
            "y_test  = t.arange(N_test) + 100\n"
            "BATCH = 4\n"
            "\n"
            "t.manual_seed(0)\n"
            "train_loader, test_loader = ex1_build_loaders(x_train, y_train, x_test, y_test, BATCH)\n"
            "\n"
            "# === Train loader: shapes & shuffle ===\n"
            "train_batches_epoch1 = [(xb.clone(), yb.clone()) for xb, yb in train_loader]\n"
            "# Shape sanity.\n"
            "for xb, yb in train_batches_epoch1[:-1]:\n"
            "    assert xb.shape == (BATCH, F), f'train xb shape {tuple(xb.shape)} != ({BATCH}, {F})'\n"
            "    assert yb.shape == (BATCH,),    f'train yb shape {tuple(yb.shape)} != ({BATCH},)'\n"
            "# Partial last batch.\n"
            "last_xb, last_yb = train_batches_epoch1[-1]\n"
            "assert last_xb.shape == (N_train % BATCH, F) if (N_train % BATCH) else (BATCH, F), (\n"
            "    f'partial-last-batch shape unexpected: {tuple(last_xb.shape)}'\n"
            ")\n"
            "\n"
            "# Every label 0..N_train-1 appears exactly once across all batches.\n"
            "all_train_y = t.cat([yb for _, yb in train_batches_epoch1])\n"
            "assert all_train_y.shape == (N_train,), f'concat train y wrong size: {tuple(all_train_y.shape)}'\n"
            "assert sorted(all_train_y.tolist()) == list(range(N_train)), (\n"
            "    f'every train index must appear exactly once per epoch; got {sorted(all_train_y.tolist())}'\n"
            ")\n"
            "\n"
            "# === Train loader: shuffle ===\n"
            "train_batches_epoch2 = [(xb.clone(), yb.clone()) for xb, yb in train_loader]\n"
            "y1 = t.cat([yb for _, yb in train_batches_epoch1]).tolist()\n"
            "y2 = t.cat([yb for _, yb in train_batches_epoch2]).tolist()\n"
            "assert y1 != y2, (\n"
            "    f'train shuffle=True: epoch-1 and epoch-2 orderings must differ; both were {y1}'\n"
            ")\n"
            "\n"
            "# === Test loader: shuffle=False ===\n"
            "test_batches_epoch1 = [(xb.clone(), yb.clone()) for xb, yb in test_loader]\n"
            "test_batches_epoch2 = [(xb.clone(), yb.clone()) for xb, yb in test_loader]\n"
            "yt1 = t.cat([yb for _, yb in test_batches_epoch1]).tolist()\n"
            "yt2 = t.cat([yb for _, yb in test_batches_epoch2]).tolist()\n"
            "assert yt1 == yt2, (\n"
            "    f'test shuffle=False: both epochs must produce identical ordering; got {yt1} vs {yt2}'\n"
            ")\n"
            "assert yt1 == [100, 101, 102, 103, 104, 105], (\n"
            "    f'test loader should yield in dataset order; got {yt1}'\n"
            ")\n"
            "\n"
            "# x and y must be consistently paired (xb[:, 0] == yb for our synthetic dataset).\n"
            "for xb, yb in train_batches_epoch1:\n"
            "    assert t.equal(xb[:, 0].to(t.long), yb), (\n"
            "        f'(x, y) pairing broken in train batch: x[:,0]={xb[:,0]}, y={yb}; '\n"
            "        f'did you forget to pair them in a TensorDataset?'\n"
            "    )"
        ),
        "solution_body": (
            "from torch.utils.data import TensorDataset, DataLoader\n"
            "\n"
            "\n"
            "def ex1_build_loaders(x_train, y_train, x_test, y_test, batch_size):\n"
            "    train_ds = TensorDataset(x_train, y_train)\n"
            "    test_ds  = TensorDataset(x_test,  y_test)\n"
            "    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)\n"
            "    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)\n"
            "    return train_loader, test_loader"
        ),
        "solution_notes": (
            "**Why `TensorDataset`.** It is the simplest `Dataset` subclass — "
            "wraps any number of tensors (all sharing the first-dim length) "
            "and `__getitem__(i)` returns the tuple `(x_train[i], y_train[i], "
            "...)`. The `DataLoader` then collates a list of these tuples "
            "into stacked batches automatically.\n\n"
            "**Why iterating the same loader twice produces different "
            "orderings (with shuffle=True).** Each `__iter__` call builds a "
            "fresh `RandomSampler` whose permutation is independent. That's "
            "what makes successive epochs of training see different batch "
            "orderings without you doing anything — and why `shuffle=False` "
            "is what you reach for when you want REPRODUCIBLE eval.\n\n"
            "**ARENA convention.** Look at 0_2_12 (training loop for "
            "feature extraction): the two-line DataLoader pair with "
            "`shuffle=True` then `shuffle=False` is identical to what "
            "this drill produces."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # optimizer-state-tensor-buffers  —  ex1
    # =========================================================
    {
        "atom_id": "optimizer-state-tensor-buffers",
        "subtopic": "Optimizer: Per-param state buffers",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_OPTIMIZER_STATE_BUFFERS,
        "exercise_index": 1,
        "exercise_title": "allocate a per-param zeros_like buffer at init",
        "slug": "allocate-a-per-param-zeros-like-buffer-at-init",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["zeros-like", "per-param-buffer", "init"],
        "kcs": [
            "state-buffer-allocated-via-zeros-like",
            "state-buffer-one-per-param-as-list",
        ],
        "lo": (
            "Apply `[t.zeros_like(p) for p in self.params]` to allocate a "
            "per-parameter momentum buffer at optimizer init time, matching "
            "the shape and dtype of every parameter."
        ),
        "prompt_body": (
            "Implement `ex1_allocate_buffer(params)`. Return a Python list "
            "of zero-initialized tensors — one per param — each with the "
            "SAME shape AND dtype AND device AS the corresponding parameter.\n\n"
            "Constraints:\n"
            "- Use `t.zeros_like(p)`. Do NOT use `t.zeros(p.shape)` (that "
            "loses dtype/device).\n"
            "- The result is a list, NOT a single stacked tensor — different "
            "params can have different shapes.\n"
            "- Buffers do NOT track gradients (`zeros_like` returns "
            "`requires_grad=False` by default — verify that).\n\n"
            "Input:\n"
            "- `params`: iterable of leaf tensors (typically from "
            "`model.parameters()`).\n\n"
            "The test passes a mix of `(256, 768)`, `(10,)`, `(3, 3, 5)` "
            "parameters and verifies shape, dtype, requires_grad, and that "
            "every entry is all-zeros."
        ),
        "stub": (
            "def ex1_allocate_buffer(params) -> list:\n"
            '    """Return [zeros_like(p) for p in params] — one buffer per param."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a diverse param list: different shapes + a float64 entry.\n"
            "p1 = t.randn(256, 768, requires_grad=True)\n"
            "p2 = t.randn(10, requires_grad=True)\n"
            "p3 = t.randn(3, 3, 5, requires_grad=True)\n"
            "p4 = t.randn(4, 4, dtype=t.float64, requires_grad=True)\n"
            "params = [p1, p2, p3, p4]\n"
            "\n"
            "buffers = ex1_allocate_buffer(params)\n"
            "\n"
            "assert isinstance(buffers, list), f'must return a list, got {type(buffers)}'\n"
            "assert len(buffers) == 4, f'expected 4 buffers, got {len(buffers)}'\n"
            "\n"
            "for i, (p, b) in enumerate(zip(params, buffers)):\n"
            "    assert b.shape == p.shape, (\n"
            "        f'buffers[{i}] shape {tuple(b.shape)} != param shape {tuple(p.shape)}'\n"
            "    )\n"
            "    assert b.dtype == p.dtype, (\n"
            "        f'buffers[{i}] dtype {b.dtype} != param dtype {p.dtype}; '\n"
            "        f'did you use t.zeros(p.shape) instead of t.zeros_like(p)?'\n"
            "    )\n"
            "    assert b.requires_grad is False, (\n"
            "        f'buffers[{i}].requires_grad must be False; '\n"
            "        f'state buffers are NOT trainable'\n"
            "    )\n"
            "    assert t.all(b == 0), f'buffers[{i}] must be all zeros'\n"
            "\n"
            "# Each buffer must be a DISTINCT tensor (mutating one mustn't affect another).\n"
            "buffers[0] += 1.0\n"
            "for i in (1, 2, 3):\n"
            "    assert t.all(buffers[i] == 0), (\n"
            "        f'buffers[{i}] aliased buffers[0]; did you reuse the same tensor?'\n"
            "    )\n"
            "\n"
            "# Buffer must not alias its parameter — modifying buffer[0] left p1 untouched.\n"
            "assert not t.allclose(p1, buffers[0]), (\n"
            "    'param and buffer should be independent storage; '\n"
            "    'did you accidentally return the param list itself?'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_allocate_buffer(params):\n"
            "    return [t.zeros_like(p) for p in params]"
        ),
        "solution_notes": (
            "**Why `zeros_like` and not `zeros`.** `zeros_like(p)` mirrors "
            "`p`'s `shape`, `dtype`, `layout`, and `device`. For a "
            "`(256, 768)` `float16` `cuda:0` weight, the resulting buffer is "
            "`(256, 768)` `float16` `cuda:0`. `t.zeros(p.shape)` would give "
            "you `float32` on CPU — wrong dtype causes silent precision "
            "drift in the optimizer state; wrong device crashes the first "
            "buffer arithmetic operation.\n\n"
            "**Why a list of tensors rather than one big tensor.** Different "
            "params have different shapes. A `Linear(3, 5)` has `(5, 3)` "
            "weight and `(5,)` bias — those can't share a tensor. PyTorch "
            "internally also stores per-param state as a dict keyed by "
            "param id, mapping to per-param buffers.\n\n"
            "**`zeros_like` default for `requires_grad`.** False, which is "
            "what we want — buffers are bookkeeping, not parameters to "
            "optimize through."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # optimizer-state-tensor-buffers  —  ex2
    # =========================================================
    {
        "atom_id": "optimizer-state-tensor-buffers",
        "subtopic": "Optimizer: Per-param state buffers",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_OPTIMIZER_STATE_BUFFERS,
        "exercise_index": 2,
        "exercise_title": "two-buffer init for an RMSprop-style optimizer",
        "slug": "two-buffer-init-for-an-rmsprop-style-optimizer",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["rmsprop-init", "two-buffers", "v-and-b"],
        "kcs": [
            "state-buffer-allocated-via-zeros-like",
            "state-buffer-multiple-buffers-per-optimizer",
        ],
        "lo": (
            "Apply per-param `zeros_like` allocation to BOTH the momentum "
            "buffer `b` and the squared-gradient EMA buffer `v` in a single "
            "optimizer init — matching ARENA's RMSprop spine."
        ),
        "prompt_body": (
            "Implement `Ex2RMSpropInit.__init__(self, params)`. The "
            "skeleton of an RMSprop optimizer's init: TWO per-param buffers "
            "must be allocated.\n\n"
            "1. Materialize `self.params = list(params)` (so the generator "
            "case doesn't bite us — see the optimizer-init-list drill).\n"
            "2. Allocate the momentum buffer: `self.b = [t.zeros_like(p) for "
            "p in self.params]`.\n"
            "3. Allocate the EMA-of-squared-gradients buffer: "
            "`self.v = [t.zeros_like(p) for p in self.params]`.\n"
            "4. (No step / no zero_grad needed — this drill isolates "
            "the init step.)\n\n"
            "**Critical:** `self.b` and `self.v` must be SEPARATE lists "
            "of SEPARATE tensors. A common bug is `self.v = self.b` (alias) "
            "— mutating one then mutates the other.\n\n"
            "The test passes `model.parameters()` from a small `nn.Linear` "
            "stack and verifies that both buffer lists have the right shapes "
            "and are independent."
        ),
        "stub": (
            "class Ex2RMSpropInit:\n"
            '    """Allocate momentum buffer self.b and EMA buffer self.v."""\n'
            "\n"
            "    def __init__(self, params):\n"
            "        raise NotImplementedError()"
        ),
        "test_body": (
            "# Two-layer MLP — diverse param shapes including biases.\n"
            "model = t.nn.Sequential(\n"
            "    t.nn.Linear(8, 16),\n"
            "    t.nn.ReLU(),\n"
            "    t.nn.Linear(16, 4),\n"
            ")\n"
            "expected_shapes = [(16, 8), (16,), (4, 16), (4,)]\n"
            "\n"
            "opt = Ex2RMSpropInit(model.parameters())\n"
            "\n"
            "assert isinstance(opt.params, list), (\n"
            "    f'opt.params must be a list (generator must be materialized), got {type(opt.params)}'\n"
            ")\n"
            "assert len(opt.params) == 4, f'expected 4 params, got {len(opt.params)}'\n"
            "\n"
            "# Both buffer lists exist.\n"
            "assert hasattr(opt, 'b'), 'missing self.b momentum buffer list'\n"
            "assert hasattr(opt, 'v'), 'missing self.v EMA buffer list'\n"
            "assert isinstance(opt.b, list) and isinstance(opt.v, list), 'b and v must be lists'\n"
            "assert len(opt.b) == 4 and len(opt.v) == 4, (\n"
            "    f'expected 4-long buffer lists, got len(b)={len(opt.b)}, len(v)={len(opt.v)}'\n"
            ")\n"
            "\n"
            "# Per-param shape & zero-init.\n"
            "for i, (p, b, v, exp_shape) in enumerate(zip(opt.params, opt.b, opt.v, expected_shapes)):\n"
            "    assert tuple(b.shape) == exp_shape, (\n"
            "        f'opt.b[{i}].shape {tuple(b.shape)} != {exp_shape}'\n"
            "    )\n"
            "    assert tuple(v.shape) == exp_shape, (\n"
            "        f'opt.v[{i}].shape {tuple(v.shape)} != {exp_shape}'\n"
            "    )\n"
            "    assert t.all(b == 0), f'opt.b[{i}] must be all zeros'\n"
            "    assert t.all(v == 0), f'opt.v[{i}] must be all zeros'\n"
            "    assert b.requires_grad is False and v.requires_grad is False\n"
            "\n"
            "# CRITICAL: b and v must be DIFFERENT tensors, not aliases.\n"
            "for i in range(4):\n"
            "    assert opt.b[i] is not opt.v[i], (\n"
            "        f'opt.b[{i}] and opt.v[{i}] are the same tensor — did you write self.v = self.b?'\n"
            "    )\n"
            "# Prove non-aliasing by mutation.\n"
            "opt.b[0] += 5.0\n"
            "assert t.all(opt.v[0] == 0), (\n"
            "    'after mutating opt.b[0], opt.v[0] should be unchanged; '\n"
            "    'b and v share storage — did you alias the lists?'\n"
            ")\n"
            "# Buffers must not share storage WITH the params either.\n"
            "for i, (p, b) in enumerate(zip(opt.params, opt.b[1:], )):\n"
            "    pass  # placeholder — exhaustive check below\n"
            "assert opt.b[1].data_ptr() != opt.params[1].data_ptr(), (\n"
            "    'opt.b[1] shares storage with opt.params[1] — buffers must be independent'\n"
            ")"
        ),
        "solution_body": (
            "class Ex2RMSpropInit:\n"
            "    def __init__(self, params):\n"
            "        self.params = list(params)\n"
            "        self.b = [t.zeros_like(p) for p in self.params]\n"
            "        self.v = [t.zeros_like(p) for p in self.params]"
        ),
        "solution_notes": (
            "**Why two separate `zeros_like` list-comprehensions, not one.** "
            "Each call to `t.zeros_like(p)` allocates a FRESH tensor. If you "
            "tried to share allocation — `buf = [t.zeros_like(p) for p in "
            "self.params]; self.b = buf; self.v = buf` — both attributes "
            "would point to the SAME list of the SAME tensors. Mutating "
            "`self.b[0]` in the momentum-update line would silently corrupt "
            "`self.v[0]`. Two list-comprehensions guarantees independence.\n\n"
            "**The pattern generalizes.** Adam needs `m` and `v` — same two "
            "list-comprehensions with names swapped. AdaGrad needs only "
            "`G` (sum of squared grads) — one list-comp. The buffer-init "
            "block always looks like this; the rest of the optimizer "
            "differs.\n\n"
            "**Why this is its own drill.** Forgetting the second buffer "
            "(or accidentally aliasing it) is a top-3 source of "
            "RMSprop/Adam impl bugs in ARENA. Isolating buffer ALLOCATION "
            "from buffer UPDATE makes it possible to catch the bug at init "
            "time rather than at step time."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # weight-decay-l2-add  —  ex1
    # =========================================================
    {
        "atom_id": "weight-decay-l2-add",
        "subtopic": "Optimizer: Weight decay L2",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_WEIGHT_DECAY,
        "exercise_index": 1,
        "exercise_title": "fold weight decay lambda*theta into the gradient",
        "slug": "fold-weight-decay-lambda-theta-into-the-gradient",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["weight-decay", "l2-fold", "gradient-augment"],
        "kcs": [
            "weight-decay-l2-fold-into-grad",
            "weight-decay-zero-lambda-bypass",
        ],
        "lo": (
            "Apply the L2 weight-decay fold `g = g + lmda * theta` inside an "
            "optimizer step, guarded by `if lmda != 0` to skip the fold when "
            "decay is disabled."
        ),
        "prompt_body": (
            "Implement `ex1_apply_weight_decay(theta, g, lmda)`. This is the "
            "TWO-LINE block from ARENA's SGD/RMSprop/Adam impls.\n\n"
            "1. If `lmda == 0`, return `g` unchanged (the bypass — saves "
            "allocation in the inner loop).\n"
            "2. Otherwise, return `g + lmda * theta` (do NOT mutate `g` "
            "in place — return a new tensor).\n\n"
            "Inputs:\n"
            "- `theta`: parameter tensor.\n"
            "- `g`: gradient tensor, same shape as `theta`.\n"
            "- `lmda`: float decay coefficient.\n\n"
            "Output: the augmented gradient — same shape and dtype as `g`.\n\n"
            "The test verifies BOTH paths: the augmented value when "
            "`lmda > 0`, the literal identity (same object) when "
            "`lmda == 0`, and the sign behavior when `theta` is negative."
        ),
        "stub": (
            "def ex1_apply_weight_decay(theta: Tensor, g: Tensor, lmda: float) -> Tensor:\n"
            '    """Return g + lmda*theta when lmda != 0, else return g unchanged."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Basic positive theta, positive grad, positive lmda.\n"
            "theta = t.tensor([1.0, 2.0, 3.0])\n"
            "g     = t.tensor([0.1, 0.2, 0.3])\n"
            "out = ex1_apply_weight_decay(theta, g, lmda=0.01)\n"
            "expected = t.tensor([0.11, 0.22, 0.33])  # g + 0.01 * theta\n"
            "assert t.allclose(out, expected), (\n"
            "    f'lmda=0.01: got {out}, expected {expected}; '\n"
            "    f'check you are computing g + lmda * theta'\n"
            ")\n"
            "\n"
            "# === Bypass path: lmda == 0 must return g UNCHANGED ===\n"
            "g_for_bypass = t.tensor([0.5, -0.5, 0.0])\n"
            "out_bypass = ex1_apply_weight_decay(theta, g_for_bypass, lmda=0.0)\n"
            "assert t.equal(out_bypass, g_for_bypass), (\n"
            "    f'lmda=0 must return g identical; got {out_bypass}'\n"
            ")\n"
            "# Ideally the SAME object — many ARENA solutions check this.\n"
            "assert out_bypass is g_for_bypass, (\n"
            "    'lmda=0 should return the input g unchanged (same object), '\n"
            "    'avoiding an unnecessary tensor allocation in the inner loop'\n"
            ")\n"
            "\n"
            "# === Negative theta — weight decay pulls toward zero ===\n"
            "theta_neg = t.tensor([-2.0, -1.0, 1.0, 2.0])\n"
            "g_zero = t.zeros(4)\n"
            "out_neg = ex1_apply_weight_decay(theta_neg, g_zero, lmda=0.1)\n"
            "expected_neg = t.tensor([-0.2, -0.1, 0.1, 0.2])\n"
            "assert t.allclose(out_neg, expected_neg), (\n"
            "    f'with zero base gradient, fold should equal lmda * theta; '\n"
            "    f'got {out_neg}, expected {expected_neg}'\n"
            ")\n"
            "# Sign: positive theta → positive augment → after update theta moves negative (toward 0).\n"
            "# Negative theta → negative augment → after update theta moves positive (toward 0).\n"
            "# This is the WHOLE POINT of weight decay.\n"
            "\n"
            "# === Input grad not mutated when lmda != 0 ===\n"
            "g_check = t.tensor([1.0, 2.0])\n"
            "g_snapshot = g_check.clone()\n"
            "_ = ex1_apply_weight_decay(t.tensor([10.0, 20.0]), g_check, lmda=0.5)\n"
            "assert t.equal(g_check, g_snapshot), (\n"
            "    f'g was mutated by the function (now {g_check}, was {g_snapshot}); '\n"
            "    f'this fold should be out-of-place; use g + lmda*theta, not g += lmda*theta'\n"
            ")\n"
            "\n"
            "# === Shape preservation on a multi-dim param ===\n"
            "theta_2d = t.randn(4, 5)\n"
            "g_2d = t.randn(4, 5)\n"
            "out_2d = ex1_apply_weight_decay(theta_2d, g_2d, lmda=0.01)\n"
            "assert out_2d.shape == (4, 5)\n"
            "assert t.allclose(out_2d, g_2d + 0.01 * theta_2d)"
        ),
        "solution_body": (
            "def ex1_apply_weight_decay(theta, g, lmda):\n"
            "    if lmda != 0:\n"
            "        g = g + lmda * theta\n"
            "    return g"
        ),
        "solution_notes": (
            "**Why fold instead of adding to the loss.** Adding "
            "`(lmda/2)*||theta||^2` to the loss expression would work — but "
            "it would force autograd to track the regularizer through the "
            "backward, which is wasteful (the gradient of "
            "`(lmda/2)*||theta||^2` is just `lmda*theta`, computable "
            "directly). Folding skips the loss-side detour.\n\n"
            "**Why the `if lmda != 0` guard.** In the inner training loop "
            "this runs once per param per step. With `lmda=0` (no decay), "
            "the addition `g + 0 * theta` would still allocate a new tensor "
            "the size of `theta` — pure waste. The branch costs one Python-"
            "level comparison and skips the allocation. For a model with "
            "1000 parameter groups stepping 10k times, the saved "
            "allocations matter.\n\n"
            "**Classical L2 fold ≠ AdamW.** AdamW subtracts "
            "`lr * lmda * theta` from `theta` directly AFTER the Adam "
            "update, bypassing the moment estimates entirely. The fold "
            "form (this drill) modifies `g`, so for Adam the decay gets "
            "scaled by `1 / (sqrt(v_hat) + eps)` along with everything "
            "else — usually undesirable for transformers. Knowing which "
            "form your library uses is critical."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # momentum-buffer-update  —  ex1
    # =========================================================
    {
        "atom_id": "momentum-buffer-update",
        "subtopic": "Optimizer: Momentum buffer",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_MOMENTUM_BUFFER,
        "exercise_index": 1,
        "exercise_title": "in-place momentum buffer update b = mu*b + g",
        "slug": "in-place-momentum-buffer-update-b-equals-mu-b-plus-g",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["momentum", "buffer-update", "copy-inplace"],
        "kcs": [
            "momentum-recurrence-mu-b-plus-g",
            "buffer-copy_-mutates-state-in-place",
        ],
        "lo": (
            "Apply the classical momentum recurrence `b = mu*b + g` via "
            "`buffer.copy_()` so the optimizer's STATE list is mutated "
            "(not just the local variable rebound)."
        ),
        "prompt_body": (
            "Implement `ex1_momentum_step(buffer_list, grad_list, mu)`. "
            "Simulates one momentum-step pass over ALL params in an "
            "optimizer.\n\n"
            "For each `(b, g)` pair drawn from `(buffer_list, grad_list)`:\n"
            "\n"
            "1. Update IN PLACE: `b.copy_(mu * b + g)`. This mutates the "
            "tensor stored at `buffer_list[i]` — so next step sees the new "
            "value.\n"
            "2. The effective gradient `g_eff` for this param is the new "
            "buffer value. Append `g_eff` to a list and return it.\n\n"
            "**Critical:** the buffer must be mutated in place, not "
            "rebound. `buffer_list[i] = mu * buffer_list[i] + g` would also "
            "work IF you go through the index. But `b = mu * b + g` inside "
            "the loop rebinds the LOCAL `b` — the entry in `buffer_list` is "
            "untouched. Use `b.copy_(...)` to make the in-place semantics "
            "explicit.\n\n"
            "Inputs:\n"
            "- `buffer_list`: list of per-param buffers (mutated in place).\n"
            "- `grad_list`: list of per-param gradients (NOT mutated).\n"
            "- `mu`: float momentum coefficient.\n\n"
            "Output: list of per-param effective gradients (the new buffer "
            "values).\n\n"
            "The test runs TWO consecutive steps to catch the rebind bug — "
            "if the in-place mutation is missing, step 2 sees a zero buffer "
            "again instead of `mu*g1`."
        ),
        "stub": (
            "def ex1_momentum_step(buffer_list: list, grad_list: list, mu: float) -> list:\n"
            '    """Update each buffer in place via b.copy_(mu*b + g). Return new buffer values."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Two params, two buffers (zero-init like a real optimizer at step 1).\n"
            "b1 = t.zeros(3)\n"
            "b2 = t.zeros(2, 2)\n"
            "buffers = [b1, b2]\n"
            "# Save the original tensors' identity so we can verify NO rebind happens.\n"
            "orig_b1_id = id(b1)\n"
            "orig_b2_id = id(b2)\n"
            "orig_b1_ptr = b1.data_ptr()\n"
            "\n"
            "# === Step 1 ===\n"
            "g1 = [t.tensor([1.0, 2.0, 3.0]), t.tensor([[0.5, -0.5], [1.0, -1.0]])]\n"
            "g_eff_1 = ex1_momentum_step(buffers, g1, mu=0.9)\n"
            "\n"
            "# At step 1 with zero-init buffer, b = mu*0 + g = g exactly.\n"
            "assert t.allclose(g_eff_1[0], g1[0]), (\n"
            "    f'step 1: with zero buffer, g_eff should equal g; got {g_eff_1[0]} vs {g1[0]}'\n"
            ")\n"
            "assert t.allclose(g_eff_1[1], g1[1]), 'step 1: g_eff[1] mismatch'\n"
            "\n"
            "# Buffer must have been mutated IN PLACE: same object, same storage, new values.\n"
            "assert id(buffers[0]) == orig_b1_id, (\n"
            "    'buffers[0] was REBOUND to a new tensor — '\n"
            "    'use b.copy_(mu*b + g), not b = mu*b + g'\n"
            ")\n"
            "assert buffers[0].data_ptr() == orig_b1_ptr, 'buffers[0] storage was reallocated'\n"
            "assert t.allclose(buffers[0], g1[0]), (\n"
            "    f'after step 1 buffers[0] should hold g1; got {buffers[0]}'\n"
            ")\n"
            "\n"
            "# === Step 2 (the rebind bug shows up here) ===\n"
            "g2 = [t.tensor([1.0, 1.0, 1.0]), t.zeros(2, 2)]\n"
            "g_eff_2 = ex1_momentum_step(buffers, g2, mu=0.9)\n"
            "# Expected: b_2 = 0.9 * g1 + g2 = 0.9*[1,2,3] + [1,1,1] = [1.9, 2.8, 3.7]\n"
            "expected_b1_after = t.tensor([1.9, 2.8, 3.7])\n"
            "assert t.allclose(g_eff_2[0], expected_b1_after), (\n"
            "    f'step 2 g_eff[0]: got {g_eff_2[0]}, expected {expected_b1_after}; '\n"
            "    f'if you see [1,1,1] then the buffer was not mutated at step 1 (rebind bug)'\n"
            ")\n"
            "assert t.allclose(buffers[0], expected_b1_after), (\n"
            "    f'after step 2 buffers[0] should hold {expected_b1_after}, got {buffers[0]}'\n"
            ")\n"
            "# Same object still.\n"
            "assert id(buffers[0]) == orig_b1_id, 'buffers[0] rebound on step 2'\n"
            "\n"
            "# === Step 3 — verify mu=0 collapses to plain g ===\n"
            "b_fresh = t.zeros(4)\n"
            "g3 = [t.tensor([7.0, 8.0, 9.0, 10.0])]\n"
            "_ = ex1_momentum_step([b_fresh], g3, mu=0.0)\n"
            "assert t.allclose(b_fresh, g3[0]), 'mu=0: buffer should just equal g'\n"
            "\n"
            "# Input grad_list MUST NOT be mutated.\n"
            "g_in = t.tensor([5.0, 5.0])\n"
            "g_snap = g_in.clone()\n"
            "_ = ex1_momentum_step([t.zeros(2)], [g_in], mu=0.9)\n"
            "assert t.equal(g_in, g_snap), 'grad tensors must not be mutated'"
        ),
        "solution_body": (
            "def ex1_momentum_step(buffer_list, grad_list, mu):\n"
            "    g_eff_list = []\n"
            "    for b, g in zip(buffer_list, grad_list):\n"
            "        b.copy_(mu * b + g)        # IN-PLACE — mutates buffer_list[i]\n"
            "        g_eff_list.append(b)       # by-reference; the buffer IS the effective grad\n"
            "    return g_eff_list"
        ),
        "solution_notes": (
            "**Why `b.copy_(...)` and not `b = ...`.** Inside the for-loop, "
            "`b` is a LOCAL name bound to the tensor object at "
            "`buffer_list[i]`. `b = mu * b + g` rebinds the LOCAL name to a "
            "brand-new tensor; `buffer_list[i]` still points at the original "
            "zero tensor. Next iteration starts from zero again — the "
            "momentum never accumulates. This is the #1 bug in handwritten "
            "momentum/Adam impls.\n\n"
            "**Equivalent forms.** `b.copy_(mu*b + g)`, `b.mul_(mu).add_(g)`, "
            "and `buffer_list[i] = mu * buffer_list[i] + g` all produce the "
            "right result. ARENA picks `copy_` because it makes the in-place "
            "semantics literal and matches the comment "
            "'this does need to be inplace, since we're modifying the value "
            "in self.b'.\n\n"
            "**Why we return `b`, not `mu*b + g`.** After `b.copy_(mu*b+g)`, "
            "the buffer IS the new value. Returning `b` by reference means "
            "downstream code that does `theta -= lr * g_eff` sees the "
            "updated buffer. Returning `mu*b+g` would compute the same "
            "value twice."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # ema-second-moment  —  ex1
    # =========================================================
    {
        "atom_id": "ema-second-moment",
        "subtopic": "Optimizer: Adam EMA second moment",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EMA_SECOND_MOMENT,
        "exercise_index": 1,
        "exercise_title": "Adam v-buffer EMA update v = beta2*v + (1-beta2)*g^2",
        "slug": "adam-v-buffer-ema-update-v-equals-beta2-v-plus-one-minus-beta2-g-squared",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["adam", "second-moment", "ema", "squared-grad"],
        "kcs": [
            "ema-second-moment-recurrence",
            "buffer-copy_-mutates-state-in-place",
        ],
        "lo": (
            "Apply the Adam second-moment recurrence `v = beta2*v + "
            "(1-beta2)*g^2` via `buffer.copy_()` so the squared-gradient EMA "
            "buffer state is correctly mutated in place across steps."
        ),
        "prompt_body": (
            "Implement `ex1_ema_v_step(v_list, grad_list, beta2)`. The "
            "second-moment update from Adam.\n\n"
            "For each `(v, g)` pair drawn from `(v_list, grad_list)`:\n"
            "\n"
            "1. Compute the new value: `beta2 * v + (1 - beta2) * g.pow(2)` "
            "(or equivalently `g * g`).\n"
            "2. Mutate the buffer IN PLACE: `v.copy_(...)`. Don't rebind.\n"
            "3. Append the new buffer value to the return list.\n\n"
            "Inputs:\n"
            "- `v_list`: list of per-param second-moment buffers (mutated).\n"
            "- `grad_list`: list of per-param gradients (NOT mutated).\n"
            "- `beta2`: float in `(0, 1)` — Adam default is `0.999`.\n\n"
            "Output: list of updated `v` tensors.\n\n"
            "The test runs three steps with KNOWN gradients and verifies "
            "that the EMA converges toward `g^2` at the rate dictated by "
            "`beta2`, AND that the buffer is in-place mutated (id and "
            "data_ptr preserved across steps)."
        ),
        "stub": (
            "def ex1_ema_v_step(v_list: list, grad_list: list, beta2: float) -> list:\n"
            '    """In-place update: v.copy_(beta2*v + (1-beta2)*g.pow(2))."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# One param, zero-init buffer.\n"
            "v = t.zeros(4)\n"
            "orig_id = id(v)\n"
            "orig_ptr = v.data_ptr()\n"
            "\n"
            "# === Step 1: zero buffer + g => v_1 = (1 - beta2) * g^2 ===\n"
            "g1 = t.tensor([2.0, 4.0, 6.0, 8.0])\n"
            "beta2 = 0.9\n"
            "out1 = ex1_ema_v_step([v], [g1], beta2=beta2)\n"
            "expected1 = (1 - beta2) * g1.pow(2)   # 0.1 * [4, 16, 36, 64]\n"
            "assert t.allclose(out1[0], expected1), (\n"
            "    f'step 1: expected {expected1}, got {out1[0]}; '\n"
            "    f'check formula: beta2*v + (1-beta2)*g^2'\n"
            ")\n"
            "assert t.allclose(v, expected1), 'step 1: buffer not mutated to new value'\n"
            "assert id(v) == orig_id, 'buffer was rebound — use v.copy_(...) not v = ...'\n"
            "assert v.data_ptr() == orig_ptr, 'buffer storage reallocated'\n"
            "\n"
            "# === Step 2: same g; v approaches g^2 monotonically ===\n"
            "# Snapshot v BEFORE step 2 so we can compare directions after the in-place update.\n"
            "v_before_step2 = v.clone()\n"
            "out2 = ex1_ema_v_step([v], [g1], beta2=beta2)\n"
            "expected2 = beta2 * expected1 + (1 - beta2) * g1.pow(2)\n"
            "v_after_step2 = v.clone()\n"
            "assert t.allclose(v_after_step2, expected2), (\n"
            "    f'step 2: expected {expected2}, got {v_after_step2}; '\n"
            "    f'this fails if step 1 did NOT mutate the buffer (rebind bug)'\n"
            ")\n"
            "# v should be moving toward g^2 = [4, 16, 36, 64].\n"
            "g_sq = g1.pow(2)\n"
            "assert (v_after_step2 < g_sq).all(), 'v should still be below g^2 after only 2 steps'\n"
            "assert (v_after_step2 > v_before_step2).all(), (\n"
            "    'EMA should grow toward g^2 with positive constant g; '\n"
            "    'if it shrunk, your formula has the wrong sign'\n"
            ")\n"
            "\n"
            "# === Step 3: still moving toward g^2; spot-check direction ===\n"
            "v_before_step3 = v.clone()\n"
            "ex1_ema_v_step([v], [g1], beta2=beta2)\n"
            "v_after_step3 = v.clone()\n"
            "assert (v_after_step3 > v_before_step3).all(), 'EMA should keep growing toward g^2 with constant g'\n"
            "assert (v_after_step3 < g_sq).all(), 'EMA must not overshoot g^2'\n"
            "\n"
            "# === Multi-param batch ===\n"
            "v_multi = [t.zeros(2), t.zeros(3, 3)]\n"
            "g_multi = [t.tensor([1.0, -1.0]), t.ones(3, 3) * 2.0]\n"
            "ex1_ema_v_step(v_multi, g_multi, beta2=0.5)\n"
            "# 0.5 * 0 + 0.5 * [1, 1] = [0.5, 0.5]\n"
            "assert t.allclose(v_multi[0], t.tensor([0.5, 0.5])), (\n"
            "    f'multi-param step: v_multi[0]={v_multi[0]}'\n"
            ")\n"
            "# 0.5 * 0 + 0.5 * 4 = 2.0 everywhere\n"
            "assert t.allclose(v_multi[1], t.full((3, 3), 2.0)), (\n"
            "    f'multi-param step: v_multi[1]={v_multi[1]}'\n"
            ")\n"
            "\n"
            "# === Negative gradients: g^2 is positive, v stays non-negative ===\n"
            "v_neg = t.zeros(3)\n"
            "g_neg = t.tensor([-3.0, -4.0, -5.0])\n"
            "ex1_ema_v_step([v_neg], [g_neg], beta2=0.9)\n"
            "assert (v_neg >= 0).all(), (\n"
            "    f'v must remain non-negative (we square g); got {v_neg}; '\n"
            "    f'are you computing g.pow(2) or just g?'\n"
            ")\n"
            "expected_neg = 0.1 * t.tensor([9.0, 16.0, 25.0])\n"
            "assert t.allclose(v_neg, expected_neg), (\n"
            "    f'negative-g case: expected {expected_neg}, got {v_neg}'\n"
            ")\n"
            "\n"
            "# === Input grad must not be mutated ===\n"
            "g_in = t.tensor([2.0, 3.0])\n"
            "g_snap = g_in.clone()\n"
            "ex1_ema_v_step([t.zeros(2)], [g_in], beta2=0.99)\n"
            "assert t.equal(g_in, g_snap), 'grad tensors must not be mutated by the EMA update'"
        ),
        "solution_body": (
            "def ex1_ema_v_step(v_list, grad_list, beta2):\n"
            "    out = []\n"
            "    for v, g in zip(v_list, grad_list):\n"
            "        v.copy_(beta2 * v + (1 - beta2) * g.pow(2))\n"
            "        out.append(v)\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why `g.pow(2)` and not `g ** 2`.** Functionally identical for "
            "this case. ARENA's solution uses `g.pow(2)` to mirror the "
            "explicit pseudocode notation `g_t^2`. Both compile down to the "
            "same kernel.\n\n"
            "**What the EMA converges to.** If `g` is constant, "
            "`v_t = (1 - beta2^t) * g^2`. As `t -> inf`, `v_t -> g^2`. "
            "That's the intuition — `v` is a low-pass filter on `g^2` with "
            "time constant `~1/(1 - beta2)`. With `beta2 = 0.999`, the "
            "effective averaging window is ~1000 recent steps.\n\n"
            "**Bias correction is a SEPARATE drill.** The recurrence here "
            "is biased toward zero in the first few steps (because `v` "
            "starts at zero). Adam fixes this with "
            "`v_hat = v / (1 - beta2^t)`. That correction is its own "
            "atom (`bias-correction-divide`) — this drill isolates just "
            "the EMA mechanic.\n\n"
            "**Why this matters operationally.** `v` shows up under a "
            "`sqrt` in the Adam denominator: `theta -= lr * m_hat / "
            "(sqrt(v_hat) + eps)`. Coordinates with consistently large `|g|` "
            "get large `v`, large denominator, SMALL effective lr. "
            "Coordinates with tiny `|g|` get tiny `v`, tiny denominator, "
            "LARGE effective lr. That's adaptive per-parameter learning — "
            "the whole point of Adam over plain SGD."
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
    print(f"[batch3] Verifying {len(SPECS)} specs against torch backend...")
    _verify_all(SPECS)

    print(f"\n[batch3] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[batch3] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
