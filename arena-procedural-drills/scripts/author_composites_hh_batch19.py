"""Composite drills cx13..cx18 — batch-19 part3 (HH-cell, ARENA optim WD variants + eval).

Six composite procedural drills exercising 2-atom pairs from ARENA part 3 —
optimizer weight-decay variants and the eval/training-loop boundary.

cx13  weight-decay-l2-add  + momentum-buffer-update — L2 WD folded into momentum
cx14  weight-decay-l2-add  + zero-grad-set-none     — WD + zero_grad in same loop
cx15  inference-mode-step  + zero-grad-set-none     — eval pass + resume training
cx16  inference-mode-step  + inplace-param-update   — eval forward then in-place step
cx17  weight-decay-decoupled + inplace-param-update — AdamW p *= (1 - lr*lam) then step
cx18  weight-decay-l2-add  + weight-decay-decoupled — L2 vs decoupled effective update
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
]


# ===========================================================================
# cx13 — L2 weight decay folded into momentum buffer
# ===========================================================================
spec_13 = {
    "atom_ids": ["weight-decay-l2-add", "momentum-buffer-update"],
    "subtopics": _subs(["weight-decay-l2-add", "momentum-buffer-update"]),
    "primary_atom": "weight-decay-l2-add",
    "part": "part3",
    "exercise_index": 13,
    "exercise_title": "L2 weight decay folded into the SGD momentum buffer",
    "slug": "wd-l2-into-momentum-buffer",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "PyTorch's `torch.optim.SGD(momentum=mu, weight_decay=lam)` does NOT apply weight decay as a "
        "separate post-step. Instead it ADDS `lam * theta` to the gradient BEFORE the momentum "
        "buffer is updated, so the buffer carries WD-augmented gradients through every step.\n\n"
        "**Atom A — weight-decay-l2-add.** The L2 trick: `grad <- grad + lam * theta`. This is "
        "equivalent (only for vanilla SGD without momentum) to adding `(lam/2) * ||theta||^2` to "
        "the loss. With momentum, it is NOT mathematically identical to L2 regularization any more "
        "— but it is what PyTorch's SGD does, and the test cross-checks against it.\n\n"
        "**Atom B — momentum-buffer-update.** `b <- mu * b + g`. The momentum buffer accumulates "
        "the (now WD-augmented) gradient with an EMA-like decay.\n\n"
        "**Anatomy of one step.**\n"
        "```python\n"
        "for p in params:\n"
        "    g = p.grad\n"
        "    if lam != 0: g = g + lam * p          # Atom A: fold WD into the grad.\n"
        "    if mu != 0:\n"
        "        if buf[p] is None: buf[p] = g.clone()\n"
        "        else: buf[p].mul_(mu).add_(g)     # Atom B: b = mu*b + g.\n"
        "        g = buf[p]\n"
        "    p.data.add_(g, alpha=-lr)             # update theta.\n"
        "```\n\n"
        "**Why both atoms together.** Cross-checks against `torch.optim.SGD` only line up if the WD "
        "is added to `g` BEFORE the buffer update. Doing it after (or as a separate `theta *= "
        "(1 - lr*lam)` step like AdamW) gives a different trajectory."
    ),
    "prompt_body": (
        "Implement `cx13_sgd_step(params, lr, momentum, weight_decay, buffers)`:\n\n"
        "- `params`: list of `t.Tensor` with `.grad` populated and `.data` to be updated in-place.\n"
        "- `buffers`: dict mapping `id(param) -> t.Tensor or None`. Updated in-place. (Caller "
        "supplies an initial `{id(p): None for p in params}`.)\n"
        "- Steps per param:\n"
        "  1. Read `g = p.grad`.\n"
        "  2. If `weight_decay != 0`: `g = g + weight_decay * p.data` (the L2 fold; do NOT mutate "
        "`p.grad`).\n"
        "  3. If `momentum != 0`: if `buffers[id(p)] is None` initialise it to `g.clone()`, else "
        "do `buffers[id(p)].mul_(momentum).add_(g)`; then set `g = buffers[id(p)]`.\n"
        "  4. `p.data.add_(g, alpha=-lr)`.\n\n"
        "The test cross-checks the trajectory against `torch.optim.SGD(lr, momentum, weight_decay)`."
    ),
    "stub_body": (
        "def cx13_sgd_step(params, lr, momentum, weight_decay, buffers):\n"
        "    \"\"\"In-place one-step SGD with momentum + L2 weight decay.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: matches torch.optim.SGD over a 4-step trajectory.\n"
        "t.manual_seed(0)\n"
        "lr, mu, lam = 0.05, 0.9, 0.01\n"
        "x_ref = t.randn(6, requires_grad=True)\n"
        "x_mine = x_ref.detach().clone().requires_grad_(True)\n"
        "ref_opt = t.optim.SGD([x_ref], lr=lr, momentum=mu, weight_decay=lam)\n"
        "buffers = {id(x_mine): None}\n"
        "\n"
        "for step in range(4):\n"
        "    loss_ref = (x_ref * t.arange(1.0, 7.0)).pow(2).sum()\n"
        "    ref_opt.zero_grad()\n"
        "    loss_ref.backward()\n"
        "    ref_opt.step()\n"
        "\n"
        "    loss_mine = (x_mine * t.arange(1.0, 7.0)).pow(2).sum()\n"
        "    if x_mine.grad is not None:\n"
        "        x_mine.grad.zero_()\n"
        "    loss_mine.backward()\n"
        "    cx13_sgd_step([x_mine], lr=lr, momentum=mu, weight_decay=lam, buffers=buffers)\n"
        "\n"
        "    assert t.allclose(x_mine.data, x_ref.data, atol=1e-5), (\n"
        "        f'step {step}: mine={x_mine.data}, ref={x_ref.data}'\n"
        "    )\n"
        "\n"
        "# Case B: with mu=0, just gradient + WD.\n"
        "t.manual_seed(1)\n"
        "p = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "p.grad = t.tensor([0.1, 0.1, 0.1])\n"
        "before = p.data.clone()\n"
        "buf = {id(p): None}\n"
        "cx13_sgd_step([p], lr=0.1, momentum=0.0, weight_decay=0.5, buffers=buf)\n"
        "expected = before - 0.1 * (t.tensor([0.1, 0.1, 0.1]) + 0.5 * before)\n"
        "assert t.allclose(p.data, expected, atol=1e-6), f'mu=0 path: got {p.data}, want {expected}'\n"
        "\n"
        "# Case C: buffer is actually updated in-place (not replaced).\n"
        "t.manual_seed(2)\n"
        "p = t.tensor([1.0, 1.0], requires_grad=True)\n"
        "p.grad = t.tensor([0.2, 0.2])\n"
        "buf = {id(p): None}\n"
        "cx13_sgd_step([p], lr=0.01, momentum=0.9, weight_decay=0.0, buffers=buf)\n"
        "assert buf[id(p)] is not None, 'buffer not initialised on first step'\n"
        "first_buf = buf[id(p)]\n"
        "p.grad = t.tensor([0.3, 0.3])\n"
        "cx13_sgd_step([p], lr=0.01, momentum=0.9, weight_decay=0.0, buffers=buf)\n"
        "assert buf[id(p)] is first_buf, 'buffer must be updated in-place across steps, not replaced'\n"
        "expected_buf = 0.9 * t.tensor([0.2, 0.2]) + t.tensor([0.3, 0.3])\n"
        "assert t.allclose(buf[id(p)], expected_buf, atol=1e-6), (\n"
        "    f'buffer update wrong: got {buf[id(p)]}, want {expected_buf}'\n"
        ")\n"
        "\n"
        "# Case D: WD must NOT mutate p.grad (only the local `g`).\n"
        "p = t.tensor([5.0, 5.0], requires_grad=True)\n"
        "g_in = t.tensor([0.0, 0.0])\n"
        "p.grad = g_in\n"
        "buf = {id(p): None}\n"
        "cx13_sgd_step([p], lr=0.01, momentum=0.0, weight_decay=0.1, buffers=buf)\n"
        "assert t.allclose(p.grad, t.tensor([0.0, 0.0])), (\n"
        "    f'WD must not write back into p.grad; got {p.grad}'\n"
        ")"
    ),
    "solution_body": (
        "def cx13_sgd_step(params, lr, momentum, weight_decay, buffers):\n"
        "    for p in params:\n"
        "        # Atom A (weight-decay-l2-add): fold lam*theta into the grad BEFORE the buffer update.\n"
        "        g = p.grad\n"
        "        if weight_decay != 0:\n"
        "            g = g + weight_decay * p.data   # new tensor — leaves p.grad alone.\n"
        "        # Atom B (momentum-buffer-update): b <- mu*b + g, in-place.\n"
        "        if momentum != 0:\n"
        "            buf = buffers[id(p)]\n"
        "            if buf is None:\n"
        "                buffers[id(p)] = g.clone().detach()\n"
        "            else:\n"
        "                buf.mul_(momentum).add_(g)\n"
        "            g = buffers[id(p)]\n"
        "        # In-place parameter update.\n"
        "        p.data.add_(g, alpha=-lr)"
    ),
    "solution_notes": (
        "Two ordering traps: (1) folding WD AFTER the buffer update would shift the WD contribution "
        "off by one step and diverge from `torch.optim.SGD`. (2) Initialising the buffer with the "
        "WD-augmented `g.clone()` is what PyTorch does, so the first step uses `g` directly (not "
        "`mu*0 + g`). Cloning is essential — assigning `buffers[id(p)] = g` would alias the buffer "
        "to whatever tensor `g` is currently bound to, and the next step's reassignment would "
        "silently drop the running state."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["weight-decay-l2-add", "momentum-buffer-update"],
    "lo": (
        "Compose L2 weight decay (fold lam*theta into grad before the buffer touches it) with the "
        "momentum buffer update (b = mu*b + g, in-place) to match torch.optim.SGD's WD-with-momentum "
        "trajectory exactly."
    ),
}


# ===========================================================================
# cx14 — WD + zero_grad in same loop
# ===========================================================================
spec_14 = {
    "atom_ids": ["weight-decay-l2-add", "zero-grad-set-none"],
    "subtopics": _subs(["weight-decay-l2-add", "zero-grad-set-none"]),
    "primary_atom": "weight-decay-l2-add",
    "part": "part3",
    "exercise_index": 14,
    "exercise_title": "training loop with L2 WD step + set_to_none zero_grad",
    "slug": "wd-l2-plus-zero-grad-loop",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Every training step has the same outer rhythm: `forward -> backward -> step -> "
        "zero_grad`. The step folds in **L2 weight decay**, and the zero_grad uses **set_to_none "
        "semantics** so the next backward doesn't accidentally accumulate on stale data.\n\n"
        "**Atom A — weight-decay-l2-add.** `g <- g + lam * theta`, then `theta <- theta - lr * g`.\n\n"
        "**Atom B — zero-grad-set-none.** Setting `p.grad = None` (rather than `p.grad.zero_()`) "
        "is the modern PyTorch default. Why: (i) skips the memset-to-zero kernel; (ii) makes a "
        "subsequent `+=` into `p.grad` a write (allocates a fresh tensor) instead of an accumulate. "
        "After `zero_grad(set_to_none=True)`, `backward()` REASSIGNS `p.grad`, it does not "
        "in-place-add.\n\n"
        "**Anatomy of one outer step.**\n"
        "```python\n"
        "loss = f(x, target).mean()\n"
        "loss.backward()                               # populates p.grad\n"
        "for p in params:                              # the 'step'\n"
        "    g = p.grad + lam * p                      # Atom A\n"
        "    p.data.add_(g, alpha=-lr)\n"
        "for p in params: p.grad = None                # Atom B\n"
        "```\n\n"
        "**Why both atoms together.** A common bug: forgetting `zero_grad` keeps adding new grads "
        "to old ones, so the WD-augmented step keeps drifting. A second common bug: zeroing with "
        "`p.grad.zero_()` while expecting `set_to_none` semantics — your invariant 'p.grad is None "
        "at the top of each step' silently fails."
    ),
    "prompt_body": (
        "Implement `cx14_train_loop(x, target, params, lr, weight_decay, n_steps)`:\n\n"
        "On each of `n_steps` iterations:\n"
        "1. Compute `pred = x @ params[0]` (params is a single-element list `[W]`, shape `(d, k)`).\n"
        "2. `loss = ((pred - target) ** 2).mean()`.\n"
        "3. `loss.backward()`.\n"
        "4. **L2 WD step**: for `p in params`, `g = p.grad + weight_decay * p.data`, then "
        "`p.data.add_(g, alpha=-lr)` (no momentum here — vanilla SGD + WD).\n"
        "5. **set_to_none zero_grad**: for `p in params`, set `p.grad = None`.\n\n"
        "Return the FINAL loss as a float.\n\n"
        "The test checks: (a) FINAL value of `W` matches `torch.optim.SGD(lr=lr, weight_decay=lam, "
        "momentum=0)`; (b) AFTER the loop, `params[0].grad is None` (not just zero); (c) the "
        "training actually drove the loss down (sanity)."
    ),
    "stub_body": (
        "def cx14_train_loop(x, target, params, lr, weight_decay, n_steps):\n"
        "    \"\"\"Vanilla SGD+L2 WD loop using set_to_none zero_grad. Returns final loss (float).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: matches torch.optim.SGD with weight_decay over a multi-step run.\n"
        "t.manual_seed(0)\n"
        "d, k = 6, 3\n"
        "x = t.randn(20, d)\n"
        "target = t.randn(20, k)\n"
        "W_mine = t.zeros(d, k, requires_grad=True)\n"
        "W_ref  = t.zeros(d, k, requires_grad=True)\n"
        "lr, lam, N = 0.05, 0.02, 5\n"
        "\n"
        "ref_opt = t.optim.SGD([W_ref], lr=lr, weight_decay=lam, momentum=0.0)\n"
        "for _ in range(N):\n"
        "    pred = x @ W_ref\n"
        "    loss = ((pred - target) ** 2).mean()\n"
        "    ref_opt.zero_grad(set_to_none=True)\n"
        "    loss.backward()\n"
        "    ref_opt.step()\n"
        "\n"
        "final = cx14_train_loop(x, target, [W_mine], lr=lr, weight_decay=lam, n_steps=N)\n"
        "assert isinstance(final, float), f'must return float, got {type(final).__name__}'\n"
        "assert t.allclose(W_mine.data, W_ref.data, atol=1e-5), (\n"
        "    f'final W mismatch; max err={(W_mine - W_ref).abs().max().item()}'\n"
        ")\n"
        "\n"
        "# Case B: set_to_none — after the loop, p.grad is None (not a zero tensor).\n"
        "assert W_mine.grad is None, (\n"
        "    f'after cx14_train_loop, params[0].grad must be None (set_to_none semantics); '\n"
        "    f'got {type(W_mine.grad).__name__}'\n"
        ")\n"
        "\n"
        "# Case C: with WD=0, behaves like plain SGD.\n"
        "t.manual_seed(1)\n"
        "x2 = t.randn(10, 4)\n"
        "y2 = t.randn(10, 2)\n"
        "Wa = t.zeros(4, 2, requires_grad=True)\n"
        "Wb = t.zeros(4, 2, requires_grad=True)\n"
        "cx14_train_loop(x2, y2, [Wa], lr=0.1, weight_decay=0.0, n_steps=3)\n"
        "ref = t.optim.SGD([Wb], lr=0.1, weight_decay=0.0, momentum=0.0)\n"
        "for _ in range(3):\n"
        "    pred = x2 @ Wb\n"
        "    ((pred - y2) ** 2).mean().backward()\n"
        "    ref.step(); ref.zero_grad(set_to_none=True)\n"
        "assert t.allclose(Wa.data, Wb.data, atol=1e-5), 'WD=0 path must match plain SGD'\n"
        "\n"
        "# Case D: training drove loss down (sanity).\n"
        "t.manual_seed(2)\n"
        "x3 = t.randn(30, 5)\n"
        "y3 = t.randn(30, 2)\n"
        "W3 = t.zeros(5, 2, requires_grad=True)\n"
        "initial_loss = ((x3 @ W3 - y3) ** 2).mean().item()\n"
        "final_loss = cx14_train_loop(x3, y3, [W3], lr=0.05, weight_decay=0.001, n_steps=20)\n"
        "assert final_loss < initial_loss, f'loss didn\\'t decrease: {initial_loss} -> {final_loss}'"
    ),
    "solution_body": (
        "def cx14_train_loop(x, target, params, lr, weight_decay, n_steps):\n"
        "    last_loss = float('nan')\n"
        "    for _ in range(n_steps):\n"
        "        pred = x @ params[0]\n"
        "        loss = ((pred - target) ** 2).mean()\n"
        "        loss.backward()\n"
        "        # Atom A (weight-decay-l2-add): fold lam*theta into the grad, then update.\n"
        "        for p in params:\n"
        "            g = p.grad + weight_decay * p.data\n"
        "            p.data.add_(g, alpha=-lr)\n"
        "        # Atom B (zero-grad-set-none): reset by setting to None, not by zeroing in-place.\n"
        "        for p in params:\n"
        "            p.grad = None\n"
        "        last_loss = loss.item()\n"
        "    return last_loss"
    ),
    "solution_notes": (
        "Three subtleties: (1) `g = p.grad + weight_decay * p.data` makes a NEW tensor so the "
        "next iteration's backward can safely reassign `p.grad`. (2) `p.grad = None` (not "
        "`p.grad.zero_()`) is the set_to_none flavour PyTorch defaults to since 1.7 — backward "
        "reassigns the slot. (3) Computing `loss.item()` inside the loop (not just at the end) is "
        "fine; the return value is just the most recent. NB: `p.data.add_(g, alpha=-lr)` is the "
        "vectorised form of `p.data -= lr * g` and avoids building an intermediate."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["weight-decay-l2-add", "zero-grad-set-none"],
    "lo": (
        "Compose L2 weight decay (g = grad + lam*theta inside the step) with set_to_none zero_grad "
        "(p.grad = None at the end of the step) to express a full vanilla SGD+WD training loop "
        "that matches torch.optim.SGD step-for-step."
    ),
}


# ===========================================================================
# cx15 — eval pass (no-grad) then resume training (zero_grad before next step)
# ===========================================================================
spec_15 = {
    "atom_ids": ["inference-mode-step", "zero-grad-set-none"],
    "subtopics": _subs(["inference-mode-step", "zero-grad-set-none"]),
    "primary_atom": "inference-mode-step",
    "part": "part3",
    "exercise_index": 15,
    "exercise_title": "no-grad eval pass then zero_grad before resuming training",
    "slug": "no-grad-eval-then-zero-grad",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The validation-mid-training pattern needs both atoms:\n\n"
        "**Atom A — inference-mode-step.** Wrap the eval forward in `t.no_grad()` (or "
        "`t.inference_mode()`). This: (i) skips autograd-graph construction (cheaper, less RAM); "
        "(ii) is REQUIRED if your eval forward includes in-place leaf updates (BN running stats "
        "would error if grad tracking were on); (iii) leaves `p.grad` untouched — there is nothing "
        "to backward through.\n\n"
        "**Atom B — zero-grad-set-none.** AFTER the eval pass, before the NEXT training step, you "
        "still need to reset `p.grad` (which is whatever it was BEFORE the eval — possibly nonzero "
        "from a prior backward you forgot to clear). Setting `p.grad = None` makes the invariant "
        "'`p.grad is None or freshly-overwritten` at the top of each train step' hold.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "# ... train step finishes; p.grad may or may not have been zeroed ...\n"
        "with t.no_grad():\n"
        "    val_pred = model(x_val)\n"
        "    val_loss = loss_fn(val_pred, y_val).item()\n"
        "# Resume training: ensure clean grads BEFORE the next forward+backward.\n"
        "for p in params: p.grad = None\n"
        "pred = model(x_train)\n"
        "loss = loss_fn(pred, y_train); loss.backward()  # safe: grad starts from None.\n"
        "```\n\n"
        "**Why both atoms together.** A subtle bug: doing the eval pass WITHOUT `t.no_grad()` "
        "builds a graph that holds onto the eval batch's activations, blowing up RAM. A second "
        "bug: skipping the post-eval `zero_grad` because 'we're under no_grad anyway' — but "
        "no_grad doesn't touch existing `p.grad`, only stops new ones from being made."
    ),
    "prompt_body": (
        "Implement `cx15_eval_then_resume(W, x_train, y_train, x_val, y_val, lr)`:\n\n"
        "1. **One training step** with current `W.grad` (assume backward has ALREADY populated it). "
        "Update `W.data -= lr * W.grad` in-place.\n"
        "2. **Eval pass under `t.no_grad()`**: compute `val_pred = x_val @ W`, "
        "`val_loss = ((val_pred - y_val) ** 2).mean()`. Capture `val_loss.item()` as `val_loss_v`.\n"
        "3. **Zero grad with set_to_none**: set `W.grad = None`.\n"
        "4. **Resume training**: compute `pred = x_train @ W`, `loss = ((pred - y_train) ** 2).mean()`, "
        "call `loss.backward()`.\n\n"
        "Return the tuple `(val_loss_v, W.grad.clone())`.\n\n"
        "The test verifies: (a) val_pred is computed WITHOUT a grad graph; (b) `W.grad` is None "
        "between eval and resume; (c) after resume, `W.grad` is freshly populated by the new "
        "backward (not accumulated on top of the pre-step grad)."
    ),
    "stub_body": (
        "def cx15_eval_then_resume(W, x_train, y_train, x_val, y_val, lr):\n"
        "    \"\"\"Step W with current W.grad, run no_grad eval, zero_grad, then one more backward.\n"
        "\n"
        "    Returns (val_loss: float, fresh_W_grad: Tensor).\n"
        "    \"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Setup — populate W.grad with a known pre-step gradient.\n"
        "t.manual_seed(0)\n"
        "d, k = 4, 2\n"
        "W = t.randn(d, k, requires_grad=True)\n"
        "x_train = t.randn(10, d)\n"
        "y_train = t.randn(10, k)\n"
        "x_val = t.randn(8, d)\n"
        "y_val = t.randn(8, k)\n"
        "\n"
        "# Pre-step backward — populates W.grad.\n"
        "loss_pre = ((x_train @ W - y_train) ** 2).mean()\n"
        "loss_pre.backward()\n"
        "assert W.grad is not None\n"
        "pre_grad = W.grad.clone()\n"
        "W_before_step = W.data.clone()\n"
        "lr = 0.05\n"
        "\n"
        "val_loss_v, fresh_grad = cx15_eval_then_resume(W, x_train, y_train, x_val, y_val, lr)\n"
        "\n"
        "# Case A: W was updated by the pre-step grad (step happened BEFORE eval).\n"
        "expected_W_after_step = W_before_step - lr * pre_grad\n"
        "# Tolerate that the resume-backward happened AFTER the step (W is still post-step now).\n"
        "assert t.allclose(W.data, expected_W_after_step, atol=1e-6), (\n"
        "    'W.data must reflect ONE step taken with the pre-eval grad'\n"
        ")\n"
        "\n"
        "# Case B: val_loss matches a no_grad reference computed from the post-step W.\n"
        "with t.no_grad():\n"
        "    expected_val_loss = ((x_val @ W - y_val) ** 2).mean().item()\n"
        "assert abs(val_loss_v - expected_val_loss) < 1e-5, (\n"
        "    f'val_loss mismatch: got {val_loss_v}, want {expected_val_loss}'\n"
        ")\n"
        "\n"
        "# Case C: fresh_grad came from a NEW backward (not accumulation onto pre_grad).\n"
        "# Sanity 1: shape matches.\n"
        "assert fresh_grad.shape == W.shape\n"
        "# Sanity 2: the post-eval grad does NOT equal pre_grad + something-non-zero — it equals\n"
        "# the gradient of the post-step loss alone.\n"
        "with t.no_grad():\n"
        "    W_after = W.data.clone()\n"
        "ref_W = W_after.clone().requires_grad_(True)\n"
        "ref_loss = ((x_train @ ref_W - y_train) ** 2).mean()\n"
        "ref_loss.backward()\n"
        "assert t.allclose(fresh_grad, ref_W.grad, atol=1e-5), (\n"
        "    'returned grad must equal the gradient of the resume forward alone (no stale accumulation)'\n"
        ")\n"
        "# Crucial inequality: fresh_grad != pre_grad + ref_W.grad — would fire if zero_grad missing.\n"
        "wrong_grad = pre_grad + ref_W.grad\n"
        "assert not t.allclose(fresh_grad, wrong_grad, atol=1e-5), (\n"
        "    'fresh_grad looks like an accumulation of pre_grad + new — did you forget zero_grad?'\n"
        ")\n"
        "\n"
        "# Case D: W.grad is currently the fresh one (not None) because we ran backward last.\n"
        "assert W.grad is not None\n"
        "assert t.allclose(W.grad, fresh_grad), 'W.grad must be the post-resume backward result'"
    ),
    "solution_body": (
        "def cx15_eval_then_resume(W, x_train, y_train, x_val, y_val, lr):\n"
        "    # One training step using the pre-existing W.grad.\n"
        "    W.data.add_(W.grad, alpha=-lr)\n"
        "    # Atom A (inference-mode-step): eval forward under t.no_grad() — no graph built.\n"
        "    with t.no_grad():\n"
        "        val_pred = x_val @ W\n"
        "        val_loss_v = ((val_pred - y_val) ** 2).mean().item()\n"
        "    # Atom B (zero-grad-set-none): clear stale grad BEFORE the next backward.\n"
        "    W.grad = None\n"
        "    # Resume training — fresh backward populates W.grad anew.\n"
        "    pred = x_train @ W\n"
        "    loss = ((pred - y_train) ** 2).mean()\n"
        "    loss.backward()\n"
        "    return val_loss_v, W.grad.clone()"
    ),
    "solution_notes": (
        "The load-bearing ordering: STEP → EVAL (no_grad) → ZERO_GRAD → RESUME. Move the "
        "`W.grad = None` BEFORE the eval and nothing breaks (eval doesn't touch grads under "
        "no_grad). Move it AFTER the resume backward and the next train step would skip its own "
        "zero_grad, accumulating from this step's grad. The eval pass MUST be under `t.no_grad()` "
        "— the test doesn't check that directly (we capture `.item()`), but skipping it would "
        "build a graph that holds the eval batch in memory unnecessarily."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["inference-mode-step", "zero-grad-set-none"],
    "lo": (
        "Compose no-grad eval (t.no_grad() around the validation forward, no autograd graph) with "
        "set_to_none zero_grad (clear stale gradient between eval and the next train backward) so "
        "that resumed training computes a clean gradient with no accumulation."
    ),
}


# ===========================================================================
# cx16 — eval-mode forward, then in-place param update on next train batch
# ===========================================================================
spec_16 = {
    "atom_ids": ["inference-mode-step", "inplace-param-update"],
    "subtopics": _subs(["inference-mode-step", "inplace-param-update"]),
    "primary_atom": "inplace-param-update",
    "part": "part3",
    "exercise_index": 16,
    "exercise_title": "no-grad eval forward then in-place leaf update on the train batch",
    "slug": "no-grad-eval-then-inplace-step",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Mid-training validation followed by an in-place parameter update tests a subtle invariant: "
        "`p.data.add_(...)` (or any `_`-suffixed leaf write) only works when autograd grad-tracking "
        "is OFF — otherwise PyTorch raises:\n"
        "`RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`.\n\n"
        "**Atom A — inference-mode-step.** The eval forward is wrapped in `t.no_grad()` — the "
        "standard pattern. `t.inference_mode()` is the even-stricter variant.\n\n"
        "**Atom B — inplace-param-update.** The optimizer step is `p.data.add_(g, alpha=-lr)` "
        "(or equivalent in-place call on `p.data`). Going through `.data` is what makes this legal "
        "even on a leaf tensor with `requires_grad=True` — `.data` returns a view that is NOT a "
        "leaf with grad-tracking. (Alternative: `with t.no_grad(): p.add_(...)`.)\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "# Eval forward — no graph, no grads.\n"
        "with t.no_grad():\n"
        "    val_pred = model(x_val); val_loss = ((val_pred - y_val)**2).mean().item()\n"
        "# Train batch — build a graph, backward, then in-place update via .data (or under no_grad).\n"
        "pred = model(x_train); loss = ((pred - y_train)**2).mean()\n"
        "loss.backward()                            # populates p.grad\n"
        "p.data.add_(p.grad, alpha=-lr)             # Atom B: in-place leaf update via .data\n"
        "```\n\n"
        "**Why both atoms together.** Real ARENA training-vs-eval loops interleave them. If you do "
        "the inplace update WITHOUT `.data` and without `with t.no_grad():`, PyTorch throws. If "
        "you do the eval forward WITHOUT `t.no_grad()`, you build an autograd graph that you then "
        "never use."
    ),
    "prompt_body": (
        "Implement `cx16_eval_then_step(W, x_train, y_train, x_val, y_val, lr)`:\n\n"
        "1. **Eval forward under `t.no_grad()`** — compute `val_pred = x_val @ W`, "
        "`val_loss = ((val_pred - y_val) ** 2).mean().item()`. Capture as `val_loss_v`.\n"
        "2. **Train forward + backward** — compute `pred = x_train @ W`, "
        "`loss = ((pred - y_train) ** 2).mean()`, then `loss.backward()`.\n"
        "3. **In-place leaf update**: `W.data.add_(W.grad, alpha=-lr)`. (Equivalently any in-place "
        "form on `W.data`. The `_` suffix and the `.data` view are BOTH necessary on a leaf with "
        "`requires_grad=True`.)\n\n"
        "Return `(val_loss_v, W_after_data_clone)` where `W_after_data_clone = W.data.clone()`.\n\n"
        "The test verifies: (a) the in-place op was on `.data` (W's `id` doesn't change between "
        "calls); (b) `W` is still the same Python object with `requires_grad=True` after the step; "
        "(c) the eval forward result matches a no_grad reference (so it really WAS the pre-step W); "
        "(d) the post-step `W` matches a manual reference `W_before - lr * W.grad`."
    ),
    "stub_body": (
        "def cx16_eval_then_step(W, x_train, y_train, x_val, y_val, lr):\n"
        "    \"\"\"No-grad eval then in-place SGD step via W.data.add_().\n"
        "\n"
        "    Returns (val_loss: float, W_data_after: Tensor).\n"
        "    \"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "d, k = 5, 3\n"
        "W = t.randn(d, k, requires_grad=True)\n"
        "x_train = t.randn(12, d)\n"
        "y_train = t.randn(12, k)\n"
        "x_val = t.randn(8, d)\n"
        "y_val = t.randn(8, k)\n"
        "lr = 0.05\n"
        "\n"
        "W_id_before = id(W)\n"
        "W_storage_ptr_before = W.data.data_ptr()\n"
        "W_before = W.data.clone()\n"
        "\n"
        "val_loss_v, W_after = cx16_eval_then_step(W, x_train, y_train, x_val, y_val, lr)\n"
        "\n"
        "# Case A: eval result matches no_grad reference computed from pre-step W.\n"
        "with t.no_grad():\n"
        "    ref_val = ((x_val @ W_before - y_val) ** 2).mean().item()\n"
        "assert abs(val_loss_v - ref_val) < 1e-5, (\n"
        "    f'eval forward must use pre-step W; got {val_loss_v}, want {ref_val}'\n"
        ")\n"
        "\n"
        "# Case B: in-place — id(W) and storage pointer unchanged.\n"
        "assert id(W) == W_id_before, 'W object identity must not change (in-place via .data)'\n"
        "assert W.data.data_ptr() == W_storage_ptr_before, (\n"
        "    'W.data storage was reallocated — step was NOT in-place'\n"
        ")\n"
        "\n"
        "# Case C: requires_grad still True; W is still a leaf.\n"
        "assert W.requires_grad is True, 'W must still require grad after the step'\n"
        "assert W.is_leaf is True, 'W must remain a leaf tensor'\n"
        "\n"
        "# Case D: post-step W matches manual reference.\n"
        "# Compute expected by hand: run the train backward on a frozen copy.\n"
        "W_ref = W_before.clone().requires_grad_(True)\n"
        "pred_ref = x_train @ W_ref\n"
        "loss_ref = ((pred_ref - y_train) ** 2).mean()\n"
        "loss_ref.backward()\n"
        "expected_W = W_before - lr * W_ref.grad\n"
        "assert t.allclose(W.data, expected_W, atol=1e-6), (\n"
        "    f'post-step W mismatch; max err {(W.data - expected_W).abs().max().item()}'\n"
        ")\n"
        "assert t.allclose(W_after, expected_W, atol=1e-6), 'returned W_after must equal in-place result'\n"
        "\n"
        "# Case E: doing the SAME thing without .data on the leaf would raise — sanity-document.\n"
        "Wx = t.randn(3, requires_grad=True)\n"
        "Wx.grad = t.ones_like(Wx)\n"
        "raised = False\n"
        "try:\n"
        "    Wx.add_(Wx.grad, alpha=-0.1)  # NO .data, NO no_grad — PyTorch refuses.\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, (\n"
        "    'sanity: PyTorch should refuse in-place leaf update without .data or no_grad — '\n"
        "    'if this passed, the explanation of why we use .data is now wrong'\n"
        ")"
    ),
    "solution_body": (
        "def cx16_eval_then_step(W, x_train, y_train, x_val, y_val, lr):\n"
        "    # Atom A (inference-mode-step): eval under no_grad — no graph, no grads.\n"
        "    with t.no_grad():\n"
        "        val_pred = x_val @ W\n"
        "        val_loss_v = ((val_pred - y_val) ** 2).mean().item()\n"
        "    # Train forward + backward (grad mode ON — outside the no_grad block).\n"
        "    pred = x_train @ W\n"
        "    loss = ((pred - y_train) ** 2).mean()\n"
        "    loss.backward()\n"
        "    # Atom B (inplace-param-update): in-place leaf write via .data.\n"
        "    W.data.add_(W.grad, alpha=-lr)\n"
        "    return val_loss_v, W.data.clone()"
    ),
    "solution_notes": (
        "The `.data` access is the load-bearing trick: `W` itself is a leaf with `requires_grad=True`, "
        "and PyTorch's autograd refuses any in-place op on such a leaf (would corrupt the grad graph "
        "for any consumer that still references it). `W.data` returns the same underlying storage "
        "as a non-leaf view, so the in-place add is legal. The equivalent fully-explicit form is "
        "`with t.no_grad(): W.add_(W.grad, alpha=-lr)` — both work, `.data` is the ARENA convention."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["inference-mode-step", "inplace-param-update"],
    "lo": (
        "Compose no-grad eval forward (t.no_grad() block around the validation forward) with "
        "in-place leaf parameter update (W.data.add_(W.grad, alpha=-lr)) so that training resumes "
        "without rebuilding the optimizer state or alloc'ing a new W."
    ),
}


# ===========================================================================
# cx17 — AdamW: decoupled WD then step (vs L2-add)
# ===========================================================================
spec_17 = {
    "atom_ids": ["weight-decay-decoupled", "inplace-param-update"],
    "subtopics": _subs(["weight-decay-decoupled", "inplace-param-update"]),
    "primary_atom": "weight-decay-decoupled",
    "part": "part3",
    "exercise_index": 17,
    "exercise_title": "decoupled weight decay (AdamW): p *= (1 - lr*lam) then the Adam step",
    "slug": "adamw-decoupled-wd-then-step",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "AdamW (Loshchilov & Hutter 2017) fixed Adam's interaction with weight decay. The fix is "
        "subtle: **don't add `lam*theta` to the gradient**, instead **multiply theta by `(1 - "
        "lr*lam)` BEFORE the Adam step**. This 'decoupled' WD is what gives AdamW its name.\n\n"
        "**Atom A — weight-decay-decoupled.** `p *= (1 - lr * lam)`. Note the LR appears here too "
        "— it's the same effective lr-scaled decay as you'd get from L2 in vanilla SGD, BUT it "
        "applies to `theta` directly rather than via the (whitened, second-moment-rescaled) Adam "
        "update.\n\n"
        "**Atom B — inplace-param-update.** Both the WD shrink AND the Adam step are in-place on "
        "`p.data`. Done in-place because (i) the same tensor is reused across steps; (ii) it "
        "must be done outside the autograd graph (via `.data` or `t.no_grad()`).\n\n"
        "**Anatomy of one AdamW step.**\n"
        "```python\n"
        "for p in params:\n"
        "    g = p.grad\n"
        "    # Atom A: decoupled WD FIRST (before the Adam adaptive update).\n"
        "    p.data.mul_(1 - lr * weight_decay)\n"
        "    # Adam moment updates.\n"
        "    m.mul_(b1).add_(g, alpha=1 - b1)\n"
        "    v.mul_(b2).addcmul_(g, g, value=1 - b2)\n"
        "    step += 1\n"
        "    m_hat = m / (1 - b1**step); v_hat = v / (1 - b2**step)\n"
        "    # Atom B: in-place param update.\n"
        "    p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)\n"
        "```\n\n"
        "**Why both atoms together.** Run AdamW with `lam=0` → reduces exactly to Adam. Compare to "
        "Adam-with-L2-add (`g = g + lam*p`) — the trajectories DIFFER, because the L2-add gets "
        "rescaled by the Adam second-moment denom while the decoupled WD does not. This is the "
        "whole motivation for AdamW."
    ),
    "prompt_body": (
        "Implement `cx17_adamw_step(params, lr, betas, eps, weight_decay, state)` for one AdamW "
        "optimizer step.\n\n"
        "- `params`: list of `t.Tensor` with `.grad` populated.\n"
        "- `betas`: `(b1, b2)` tuple, e.g. `(0.9, 0.999)`.\n"
        "- `state`: dict `id(p) -> {'step': int, 'exp_avg': Tensor or None, 'exp_avg_sq': Tensor or None}`. "
        "Caller seeds with `{id(p): {'step': 0, 'exp_avg': None, 'exp_avg_sq': None}}`.\n\n"
        "Per param:\n"
        "1. Read `g = p.grad`.\n"
        "2. **Decoupled WD on theta**: `p.data.mul_(1 - lr * weight_decay)` (in-place). Do this "
        "BEFORE the Adam moment updates.\n"
        "3. Initialize `exp_avg` and `exp_avg_sq` to `t.zeros_like(p.data)` on the first step.\n"
        "4. `state[id(p)]['step'] += 1`; let `s = state[id(p)]['step']`.\n"
        "5. Adam moments: `m.mul_(b1).add_(g, alpha=1 - b1)`; "
        "`v.mul_(b2).addcmul_(g, g, value=1 - b2)`.\n"
        "6. Bias-correct: `m_hat = m / (1 - b1**s)`; `v_hat = v / (1 - b2**s)`.\n"
        "7. **In-place update**: `p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)` (or "
        "equivalent in-place form on `p.data`).\n\n"
        "The test cross-checks against `torch.optim.AdamW`."
    ),
    "stub_body": (
        "def cx17_adamw_step(params, lr, betas, eps, weight_decay, state):\n"
        "    \"\"\"One AdamW step with decoupled WD + in-place leaf update.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: matches torch.optim.AdamW over a multi-step trajectory.\n"
        "t.manual_seed(0)\n"
        "lr, betas, eps, lam = 0.01, (0.9, 0.999), 1e-8, 0.1\n"
        "x_ref = t.randn(8, requires_grad=True)\n"
        "x_mine = x_ref.detach().clone().requires_grad_(True)\n"
        "ref_opt = t.optim.AdamW([x_ref], lr=lr, betas=betas, eps=eps, weight_decay=lam)\n"
        "state = {id(x_mine): {'step': 0, 'exp_avg': None, 'exp_avg_sq': None}}\n"
        "\n"
        "scale = t.arange(1.0, 9.0)\n"
        "for step in range(5):\n"
        "    loss_ref = (x_ref * scale).pow(2).sum()\n"
        "    ref_opt.zero_grad()\n"
        "    loss_ref.backward()\n"
        "    ref_opt.step()\n"
        "\n"
        "    loss_mine = (x_mine * scale).pow(2).sum()\n"
        "    if x_mine.grad is not None:\n"
        "        x_mine.grad.zero_()\n"
        "    loss_mine.backward()\n"
        "    cx17_adamw_step([x_mine], lr=lr, betas=betas, eps=eps, weight_decay=lam, state=state)\n"
        "\n"
        "    assert t.allclose(x_mine.data, x_ref.data, atol=1e-5), (\n"
        "        f'step {step}: max err {(x_mine - x_ref).abs().max().item()}'\n"
        "    )\n"
        "\n"
        "# Case B: with lam=0, AdamW reduces to Adam — and it should also match torch.optim.Adam.\n"
        "t.manual_seed(1)\n"
        "x_ref2 = t.randn(5, requires_grad=True)\n"
        "x_mine2 = x_ref2.detach().clone().requires_grad_(True)\n"
        "ref_adam = t.optim.Adam([x_ref2], lr=0.01, betas=betas, eps=eps)\n"
        "state2 = {id(x_mine2): {'step': 0, 'exp_avg': None, 'exp_avg_sq': None}}\n"
        "scale2 = t.arange(1.0, 6.0)\n"
        "for _ in range(3):\n"
        "    loss = (x_ref2 * scale2).pow(2).sum()\n"
        "    ref_adam.zero_grad(); loss.backward(); ref_adam.step()\n"
        "    lm = (x_mine2 * scale2).pow(2).sum()\n"
        "    if x_mine2.grad is not None: x_mine2.grad.zero_()\n"
        "    lm.backward()\n"
        "    cx17_adamw_step([x_mine2], lr=0.01, betas=betas, eps=eps, weight_decay=0.0, state=state2)\n"
        "assert t.allclose(x_mine2.data, x_ref2.data, atol=1e-5), 'lam=0: AdamW must reduce to Adam'\n"
        "\n"
        "# Case C: decoupled WD vs L2-add — different trajectories at nonzero g.\n"
        "# (Sanity that the implementer didn't accidentally do `g += lam*p` instead of `p *= (1-lr*lam)`.)\n"
        "t.manual_seed(2)\n"
        "x_dec = t.tensor([10.0, -10.0], requires_grad=True)\n"
        "x_dec.grad = t.tensor([1.0, 1.0])\n"
        "state3 = {id(x_dec): {'step': 0, 'exp_avg': None, 'exp_avg_sq': None}}\n"
        "cx17_adamw_step([x_dec], lr=0.1, betas=betas, eps=eps, weight_decay=0.5, state=state3)\n"
        "# Reference: torch.optim.AdamW.\n"
        "x_dec_ref = t.tensor([10.0, -10.0], requires_grad=True)\n"
        "x_dec_ref.grad = t.tensor([1.0, 1.0])\n"
        "ref = t.optim.AdamW([x_dec_ref], lr=0.1, betas=betas, eps=eps, weight_decay=0.5)\n"
        "ref.step()\n"
        "assert t.allclose(x_dec.data, x_dec_ref.data, atol=1e-5), (\n"
        "    f'mismatch with torch.optim.AdamW at nonzero p; mine={x_dec.data} ref={x_dec_ref.data}'\n"
        ")\n"
        "\n"
        "# Case D: in-place — p.data storage pointer unchanged across steps.\n"
        "t.manual_seed(3)\n"
        "x_inp = t.randn(4, requires_grad=True)\n"
        "ptr_before = x_inp.data.data_ptr()\n"
        "x_inp.grad = t.randn(4)\n"
        "state4 = {id(x_inp): {'step': 0, 'exp_avg': None, 'exp_avg_sq': None}}\n"
        "cx17_adamw_step([x_inp], lr=0.01, betas=betas, eps=eps, weight_decay=0.1, state=state4)\n"
        "assert x_inp.data.data_ptr() == ptr_before, 'AdamW step must be in-place on p.data'"
    ),
    "solution_body": (
        "def cx17_adamw_step(params, lr, betas, eps, weight_decay, state):\n"
        "    b1, b2 = betas\n"
        "    for p in params:\n"
        "        g = p.grad\n"
        "        s = state[id(p)]\n"
        "        # Atom A (weight-decay-decoupled): theta <- (1 - lr*lam) * theta, BEFORE Adam.\n"
        "        if weight_decay != 0:\n"
        "            p.data.mul_(1 - lr * weight_decay)\n"
        "        # Lazy-init moments on first step.\n"
        "        if s['exp_avg'] is None:\n"
        "            s['exp_avg'] = t.zeros_like(p.data)\n"
        "            s['exp_avg_sq'] = t.zeros_like(p.data)\n"
        "        s['step'] += 1\n"
        "        step = s['step']\n"
        "        m = s['exp_avg']\n"
        "        v = s['exp_avg_sq']\n"
        "        # Adam EMA moments.\n"
        "        m.mul_(b1).add_(g, alpha=1 - b1)\n"
        "        v.mul_(b2).addcmul_(g, g, value=1 - b2)\n"
        "        # Bias-correct.\n"
        "        bc1 = 1 - b1 ** step\n"
        "        bc2 = 1 - b2 ** step\n"
        "        m_hat = m / bc1\n"
        "        v_hat = v / bc2\n"
        "        # Atom B (inplace-param-update): one in-place addcdiv on p.data.\n"
        "        denom = v_hat.sqrt().add_(eps)\n"
        "        p.data.addcdiv_(m_hat, denom, value=-lr)"
    ),
    "solution_notes": (
        "Two failure modes the cross-check catches: (1) Putting `p.data.mul_(1 - lr*lam)` AFTER "
        "the Adam step changes which `theta` value the decay applies to (post-step vs pre-step), "
        "and diverges from `torch.optim.AdamW` by one step. (2) Adding `lam*p` to `g` (the L2-add "
        "form) gives a NUMERICALLY DIFFERENT trajectory because the Adam denom (`v_hat.sqrt() + "
        "eps`) rescales the L2 contribution — exactly the bug Loshchilov & Hutter pointed out. "
        "The in-place `addcdiv_` is one fused kernel doing `p += -lr * m_hat / denom`."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 5,
    "kcs": ["weight-decay-decoupled", "inplace-param-update"],
    "lo": (
        "Compose decoupled weight decay (p *= 1 - lr*lam BEFORE the adaptive Adam step) with "
        "in-place leaf updates (mul_/addcdiv_ on p.data) to implement one AdamW step that "
        "matches torch.optim.AdamW exactly."
    ),
}


# ===========================================================================
# cx18 — L2-add vs decoupled WD: same lam, different trajectories
# ===========================================================================
spec_18 = {
    "atom_ids": ["weight-decay-l2-add", "weight-decay-decoupled"],
    "subtopics": _subs(["weight-decay-l2-add", "weight-decay-decoupled"]),
    "primary_atom": "weight-decay-l2-add",
    "part": "part3",
    "exercise_index": 18,
    "exercise_title": "L2-add WD vs decoupled WD — different effective updates at nonzero g",
    "slug": "wd-l2-vs-decoupled",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "These two WD flavours look the same in the special case `g = 0` — both pull `theta` "
        "toward zero by the same multiplicative factor — but at a nonzero gradient they apply the "
        "decay at DIFFERENT points in the pipeline, and produce different updates.\n\n"
        "**Atom A — weight-decay-l2-add.** `g <- g + lam * theta`, THEN `theta <- theta - lr * g`. "
        "Substituting: `theta_new = theta - lr * (g + lam * theta) = (1 - lr*lam) * theta - lr * g`. "
        "Here the WD shrinks theta and the gradient step are MIXED together — fine for vanilla SGD, "
        "interacts badly with Adam's adaptive denom.\n\n"
        "**Atom B — weight-decay-decoupled.** `theta <- (1 - lr * lam) * theta`, THEN "
        "`theta <- theta - lr * g`. Substituting: `theta_new = (1 - lr*lam) * theta - lr * g`. "
        "For vanilla SGD (no adaptive scaling), this is ALGEBRAICALLY IDENTICAL to L2-add. The "
        "two atoms diverge ONLY when there's an adaptive denom in the middle (Adam → AdamW).\n\n"
        "**The composite drill.** Implement BOTH update rules over a single SGD-style step. Show "
        "that with the SAME `(theta, g, lr, lam)`, L2-add and decoupled-WD produce the SAME "
        "`theta_new` (vanilla SGD case). Then change the step rule to include an adaptive denom — "
        "now they diverge."
    ),
    "prompt_body": (
        "Implement two pure-function step rules, both returning a NEW tensor (no in-place):\n\n"
        "1. `cx18_l2_add_step(theta, g, lr, lam)` — apply L2-add:\n"
        "   - `g_eff = g + lam * theta`\n"
        "   - return `theta - lr * g_eff`\n\n"
        "2. `cx18_decoupled_step(theta, g, lr, lam)` — apply decoupled WD:\n"
        "   - `theta_shrunk = (1 - lr * lam) * theta`\n"
        "   - return `theta_shrunk - lr * g`\n\n"
        "Then implement `cx18_decoupled_with_denom_step(theta, g, lr, lam, denom)` to show the "
        "interesting case: when the gradient is rescaled by a per-element `denom > 0` (mimicking "
        "Adam's `sqrt(v_hat) + eps`):\n"
        "   - `theta_shrunk = (1 - lr * lam) * theta`\n"
        "   - return `theta_shrunk - lr * (g / denom)`\n\n"
        "And `cx18_l2_add_with_denom_step(theta, g, lr, lam, denom)`:\n"
        "   - `g_eff = g + lam * theta`\n"
        "   - return `theta - lr * (g_eff / denom)`   # ← the WD ALSO gets divided by denom.\n\n"
        "The test shows: (a) without a denom (vanilla SGD), L2-add == decoupled identically; "
        "(b) WITH a denom, the two diverge — this is the AdamW vs Adam-L2 distinction."
    ),
    "stub_body": (
        "def cx18_l2_add_step(theta, g, lr, lam):\n"
        "    \"\"\"theta - lr*(g + lam*theta).\"\"\"\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx18_decoupled_step(theta, g, lr, lam):\n"
        "    \"\"\"(1 - lr*lam)*theta - lr*g.\"\"\"\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx18_decoupled_with_denom_step(theta, g, lr, lam, denom):\n"
        "    \"\"\"(1 - lr*lam)*theta - lr*(g/denom).\"\"\"\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx18_l2_add_with_denom_step(theta, g, lr, lam, denom):\n"
        "    \"\"\"theta - lr*((g + lam*theta)/denom).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: at g=0, both rules shrink theta by the same factor (1 - lr*lam).\n"
        "theta = t.tensor([2.0, -3.0, 0.5])\n"
        "g0 = t.zeros_like(theta)\n"
        "out_l2 = cx18_l2_add_step(theta, g0, lr=0.1, lam=0.5)\n"
        "out_dc = cx18_decoupled_step(theta, g0, lr=0.1, lam=0.5)\n"
        "expected_shrink = (1 - 0.1 * 0.5) * theta\n"
        "assert t.allclose(out_l2, expected_shrink, atol=1e-6)\n"
        "assert t.allclose(out_dc, expected_shrink, atol=1e-6)\n"
        "assert t.allclose(out_l2, out_dc, atol=1e-6), 'at g=0, L2 and decoupled must agree'\n"
        "\n"
        "# Case B: WITHOUT a denom (vanilla SGD), L2-add == decoupled IDENTICALLY at any g.\n"
        "t.manual_seed(0)\n"
        "theta = t.randn(5)\n"
        "g = t.randn(5)\n"
        "out_l2 = cx18_l2_add_step(theta, g, lr=0.05, lam=0.1)\n"
        "out_dc = cx18_decoupled_step(theta, g, lr=0.05, lam=0.1)\n"
        "assert t.allclose(out_l2, out_dc, atol=1e-7), (\n"
        "    'L2-add and decoupled WD must be ALGEBRAICALLY identical for vanilla SGD'\n"
        ")\n"
        "# Pin the exact value too.\n"
        "manual = (1 - 0.05 * 0.1) * theta - 0.05 * g\n"
        "assert t.allclose(out_l2, manual, atol=1e-6)\n"
        "\n"
        "# Case C: vanilla L2 matches torch.optim.SGD(momentum=0, weight_decay=lam).\n"
        "t.manual_seed(1)\n"
        "p_ref = t.tensor([1.0, -2.0, 0.5], requires_grad=True)\n"
        "p_ref.grad = t.tensor([0.3, -0.4, 0.1])\n"
        "ref = t.optim.SGD([p_ref], lr=0.07, momentum=0.0, weight_decay=0.2)\n"
        "ref.step()\n"
        "mine = cx18_l2_add_step(t.tensor([1.0, -2.0, 0.5]), t.tensor([0.3, -0.4, 0.1]), lr=0.07, lam=0.2)\n"
        "assert t.allclose(mine, p_ref.data, atol=1e-6), 'L2-add must match torch.optim.SGD WD path'\n"
        "\n"
        "# Case D: WITH an adaptive denom, L2-add and decoupled DIVERGE at nonzero g.\n"
        "t.manual_seed(2)\n"
        "theta = t.tensor([2.0, -1.0, 3.0])\n"
        "g = t.tensor([0.5, -0.5, 1.0])\n"
        "denom = t.tensor([2.0, 0.5, 4.0])  # mimics Adam's sqrt(v_hat)+eps — pure-positive.\n"
        "lr, lam = 0.1, 0.5\n"
        "out_l2_d = cx18_l2_add_with_denom_step(theta, g, lr=lr, lam=lam, denom=denom)\n"
        "out_dc_d = cx18_decoupled_with_denom_step(theta, g, lr=lr, lam=lam, denom=denom)\n"
        "assert not t.allclose(out_l2_d, out_dc_d, atol=1e-3), (\n"
        "    'with an adaptive denom, L2-add and decoupled MUST differ (that is the AdamW fix)'\n"
        ")\n"
        "# Pin both values numerically.\n"
        "expected_l2_d = theta - lr * ((g + lam * theta) / denom)\n"
        "expected_dc_d = (1 - lr * lam) * theta - lr * (g / denom)\n"
        "assert t.allclose(out_l2_d, expected_l2_d, atol=1e-6)\n"
        "assert t.allclose(out_dc_d, expected_dc_d, atol=1e-6)\n"
        "# And the decoupled version's WD MAGNITUDE is independent of the denom, while L2-add's is not.\n"
        "# Subtract the (post-denom) gradient piece -lr*(g/denom) from (theta - out_*):\n"
        "#   theta - out_dc_d - lr*(g/denom)   should equal   lr*lam*theta  (no denom dependence)\n"
        "#   theta - out_l2_d - lr*(g/denom)   should differ — denom shows up.\n"
        "wd_part_dc = theta - out_dc_d - lr * (g / denom)\n"
        "wd_part_l2 = theta - out_l2_d - lr * (g / denom)\n"
        "assert t.allclose(wd_part_dc, lr * lam * theta, atol=1e-6), (\n"
        "    'decoupled WD magnitude must be independent of the adaptive denom'\n"
        ")\n"
        "assert not t.allclose(wd_part_l2, lr * lam * theta, atol=1e-4), (\n"
        "    'L2-add WD magnitude MUST depend on denom — that is the bug AdamW fixes'\n"
        ")"
    ),
    "solution_body": (
        "def cx18_l2_add_step(theta, g, lr, lam):\n"
        "    # Atom A (weight-decay-l2-add): fold lam*theta INTO the grad.\n"
        "    g_eff = g + lam * theta\n"
        "    return theta - lr * g_eff\n"
        "\n"
        "def cx18_decoupled_step(theta, g, lr, lam):\n"
        "    # Atom B (weight-decay-decoupled): shrink theta directly, THEN apply the grad step.\n"
        "    theta_shrunk = (1 - lr * lam) * theta\n"
        "    return theta_shrunk - lr * g\n"
        "\n"
        "def cx18_decoupled_with_denom_step(theta, g, lr, lam, denom):\n"
        "    # Decay applies to theta unscaled; grad gets the adaptive rescale.\n"
        "    theta_shrunk = (1 - lr * lam) * theta\n"
        "    return theta_shrunk - lr * (g / denom)\n"
        "\n"
        "def cx18_l2_add_with_denom_step(theta, g, lr, lam, denom):\n"
        "    # Decay folded into the grad — so it ALSO gets divided by denom.\n"
        "    g_eff = g + lam * theta\n"
        "    return theta - lr * (g_eff / denom)"
    ),
    "solution_notes": (
        "The key insight: with vanilla SGD (no adaptive denom), `theta - lr*(g + lam*theta) = "
        "(1 - lr*lam)*theta - lr*g` is just algebra — same expression two ways. With Adam's "
        "denom, the algebra breaks: in L2-add the `lam*theta` rides INSIDE the division (so the "
        "effective decay rate becomes lam/denom, parameter-dependent and curvature-rescaled), but "
        "in decoupled WD the `(1 - lr*lam)*theta` shrink happens OUTSIDE — a uniform, "
        "parameter-INdependent decay rate `lam`. That's why AdamW separates them."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "kcs": ["weight-decay-l2-add", "weight-decay-decoupled"],
    "lo": (
        "Compose L2-add WD (g += lam*theta then SGD step) with decoupled WD (theta *= 1 - lr*lam "
        "then SGD step), prove they coincide for vanilla SGD, then prove they DIVERGE under an "
        "adaptive Adam-style denom — the motivation for AdamW."
    ),
}


SPECS = [spec_13, spec_14, spec_15, spec_16, spec_17, spec_18]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
