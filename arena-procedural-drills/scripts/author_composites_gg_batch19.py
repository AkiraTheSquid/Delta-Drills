"""Composite drills cx7..cx12 — batch-19 part3 (GG-cell, ARENA optim).

Six composite procedural drills exercising 2-atom pairs from ARENA part 3 —
SGD-momentum + L2 weight-decay optimizer plumbing.

cx7   buffer-copy_-inplace + inplace-param-update    — copy_ velocity buffer then in-place SGD step
cx8   buffer-copy_-inplace + zero-grad-set-none      — buffer update via copy_, then clear grads
cx9   momentum-buffer-update + inplace-param-update  — canonical SGD-momentum: v=mu*v+g, p-=lr*v
cx10  momentum-buffer-update + buffer-copy_-inplace  — momentum buffer maintained via copy_
cx11  momentum-buffer-update + zero-grad-set-none    — momentum step, then zero grads
cx12  weight-decay-l2-add + inplace-param-update     — g += lambda*p (L2), then in-place SGD step
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# ===========================================================================
# cx7 — copy_ velocity buffer then in-place param update
# ===========================================================================
spec_7 = {
    "atom_ids": ["buffer-copy_-inplace", "inplace-param-update"],
    "subtopics": _subs(["buffer-copy_-inplace", "inplace-param-update"]),
    "primary_atom": "buffer-copy_-inplace",
    "part": "part3",
    "exercise_index": 7,
    "exercise_title": "copy_ velocity buffer, then in-place SGD step on the param",
    "slug": "buffer-copy-then-inplace-param-step",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Inside a hand-rolled SGD-momentum optimizer, ONE step does two in-place mutations on "
        "tensors owned by the optimizer:\n\n"
        "1. **`buffer-copy_-inplace`** — the velocity buffer `v` stored in `state[p]` is updated "
        "without rebinding: `v.copy_(new_v)`. If you write `v = new_v` instead, you only rebind the "
        "local variable; the entry in `state[p]['momentum_buffer']` still points to the old tensor.\n"
        "2. **`inplace-param-update`** — the parameter `p` is mutated via `p.data.add_(v, alpha=-lr)` "
        "(or equivalently `p.data -= lr * v`). Mutating `p.data` (not `p` itself) avoids touching the "
        "autograd graph; the param storage IS updated so the next forward pass sees new weights.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "for p in params:\n"
        "    g = p.grad\n"
        "    v = state[p]['momentum_buffer']\n"
        "    new_v = mu * v + g\n"
        "    v.copy_(new_v)                       # buffer-copy_-inplace\n"
        "    p.data.add_(v, alpha=-lr)            # inplace-param-update\n"
        "```\n\n"
        "**Why both atoms together.** A correct buffer update is wasted if the param isn't actually "
        "mutated; a correct param update is wasted if the buffer was rebound and reset to the prior "
        "value next step. They both have to be IN-PLACE for the optimizer to be stateful."
    ),
    "prompt_body": (
        "Implement `cx7_sgd_momentum_step(params, grads, velocity_buffers, lr, mu)`.\n\n"
        "For each `(p, g, v)` triple drawn from the three input lists, run one step of SGD with "
        "momentum so that:\n\n"
        "1. The new velocity `mu * v + g` is written INTO the existing buffer tensor `v` via "
        "`v.copy_(...)` — DO NOT rebind. The caller still holds the same tensor object in "
        "`velocity_buffers[i]` and expects it to carry the new value.\n"
        "2. The parameter `p` is updated in place: `p.data.add_(v, alpha=-lr)` (or the equivalent "
        "`p.data -= lr * v`). The test verifies `p`'s storage pointer is unchanged.\n\n"
        "Return `None`. The mutations are the whole point.\n\n"
        "The test runs TWO consecutive steps with the same buffers/params and cross-checks against "
        "`torch.optim.SGD(momentum=...)`. If either atom is missing — buffer rebind, or param "
        "rebind — step 2 will diverge from the PyTorch reference."
    ),
    "stub_body": (
        "def cx7_sgd_momentum_step(params, grads, velocity_buffers, lr, mu):\n"
        "    \"\"\"In-place: v.copy_(mu*v + g); p.data.add_(v, alpha=-lr). Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Build matched params (leaf tensors with grad slot), grads, and zero-init velocity buffers.\n"
        "t.manual_seed(0)\n"
        "shapes = [(3,), (2, 2)]\n"
        "params = [t.nn.Parameter(t.randn(s)) for s in shapes]\n"
        "velocity = [t.zeros_like(p) for p in params]\n"
        "\n"
        "# Reference: a real torch.optim.SGD with the same hyperparams.\n"
        "ref_params = [t.nn.Parameter(p.data.clone()) for p in params]\n"
        "opt_ref = t.optim.SGD(ref_params, lr=0.1, momentum=0.9)\n"
        "\n"
        "# Pre-capture identity/storage to verify in-place semantics.\n"
        "buf_ids_before = [id(b) for b in velocity]\n"
        "buf_ptrs_before = [b.data_ptr() for b in velocity]\n"
        "param_ids_before = [id(p) for p in params]\n"
        "param_ptrs_before = [p.data.data_ptr() for p in params]\n"
        "\n"
        "def _step(target_params, ref_target_params, ref_opt, grads_now):\n"
        "    # Wire our grads onto the reference params, run ref optimizer.\n"
        "    for rp, g in zip(ref_target_params, grads_now):\n"
        "        rp.grad = g.clone()\n"
        "    ref_opt.step()\n"
        "    # Run student's step on the parallel set.\n"
        "    cx7_sgd_momentum_step(target_params, grads_now, velocity, lr=0.1, mu=0.9)\n"
        "\n"
        "# === Step 1 ===\n"
        "g1 = [t.tensor([1.0, 2.0, 3.0]), t.tensor([[0.5, -0.5], [1.0, -1.0]])]\n"
        "_step(params, ref_params, opt_ref, g1)\n"
        "\n"
        "# Buffer was updated IN PLACE.\n"
        "for i in range(len(velocity)):\n"
        "    assert id(velocity[i]) == buf_ids_before[i], (\n"
        "        f'velocity[{i}] was REBOUND — use v.copy_(...), not v = ...'\n"
        "    )\n"
        "    assert velocity[i].data_ptr() == buf_ptrs_before[i], f'velocity[{i}] storage reallocated'\n"
        "# Param was updated IN PLACE (same Parameter object, same .data storage).\n"
        "for i in range(len(params)):\n"
        "    assert id(params[i]) == param_ids_before[i], f'params[{i}] was rebound'\n"
        "    assert params[i].data.data_ptr() == param_ptrs_before[i], (\n"
        "        f'params[{i}].data storage reallocated — use p.data.add_(v, alpha=-lr)'\n"
        "    )\n"
        "# Values match torch.optim.SGD after step 1.\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 1 params[{i}] mismatch vs torch.optim.SGD: '\n"
        "        f'got {params[i].data}, want {ref_params[i].data}'\n"
        "    )\n"
        "# And velocity buffer matches torch's internal momentum_buffer.\n"
        "for i in range(len(params)):\n"
        "    ref_v = opt_ref.state[ref_params[i]]['momentum_buffer']\n"
        "    assert t.allclose(velocity[i], ref_v, atol=1e-6), (\n"
        "        f'step 1 velocity[{i}] mismatch: got {velocity[i]}, want {ref_v}'\n"
        "    )\n"
        "\n"
        "# === Step 2 (catches buffer-rebind bug — buffer must carry state across steps) ===\n"
        "g2 = [t.tensor([0.1, 0.1, 0.1]), t.tensor([[0.0, 0.0], [0.0, 0.0]])]\n"
        "_step(params, ref_params, opt_ref, g2)\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 2 params[{i}] mismatch — if step 1 was right but step 2 diverges, '\n"
        "        f'the velocity buffer was rebound (not copy_-ed) on step 1'\n"
        "    )\n"
        "\n"
        "# === Step 3: zero-momentum collapses to plain SGD; param moves by exactly -lr*g.\n"
        "t.manual_seed(1)\n"
        "p3 = t.nn.Parameter(t.tensor([10.0, 20.0]))\n"
        "v3 = t.zeros_like(p3)\n"
        "p3_before = p3.data.clone()\n"
        "g3 = [t.tensor([1.0, 1.0])]\n"
        "cx7_sgd_momentum_step([p3], g3, [v3], lr=0.5, mu=0.0)\n"
        "assert t.allclose(p3.data, p3_before - 0.5 * g3[0], atol=1e-6), (\n"
        "    'mu=0: param should move by exactly -lr * g'\n"
        ")\n"
        "assert t.allclose(v3, g3[0], atol=1e-6), 'mu=0: buffer should equal g'"
    ),
    "solution_body": (
        "def cx7_sgd_momentum_step(params, grads, velocity_buffers, lr, mu):\n"
        "    for p, g, v in zip(params, grads, velocity_buffers):\n"
        "        # Atom A (buffer-copy_-inplace): write the new velocity INTO v, no rebind.\n"
        "        v.copy_(mu * v + g)\n"
        "        # Atom B (inplace-param-update): mutate p.data so storage is unchanged.\n"
        "        p.data.add_(v, alpha=-lr)\n"
        "    return None"
    ),
    "solution_notes": (
        "Two failure modes the test catches:\n"
        "- `v = mu * v + g` (rebind) — step 1 looks right because the formula is right, but step 2 "
        "diverges from `torch.optim.SGD` because `velocity_buffers[i]` still points at the original "
        "zero tensor.\n"
        "- `p = p - lr * v` (param rebind) — caller still holds the OLD parameter; their forward pass "
        "sees stale weights.\n"
        "`p.data.add_(v, alpha=-lr)` is the literal in-place op. `p.data -= lr * v` works too but "
        "allocates a temporary; `add_` doesn't."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["buffer-copy_-inplace", "inplace-param-update"],
    "lo": (
        "Compose buffer.copy_ (velocity buffer updated in place, carries state across steps) with "
        "p.data.add_(v, alpha=-lr) (param updated in place, autograd graph untouched) to produce "
        "one SGD-momentum step that matches torch.optim.SGD across consecutive steps."
    ),
}


# ===========================================================================
# cx8 — buffer copy_ update then zero_grad with set_to_none semantics
# ===========================================================================
spec_8 = {
    "atom_ids": ["buffer-copy_-inplace", "zero-grad-set-none"],
    "subtopics": _subs(["buffer-copy_-inplace", "zero-grad-set-none"]),
    "primary_atom": "zero-grad-set-none",
    "part": "part3",
    "exercise_index": 8,
    "exercise_title": "post-step buffer update via copy_, then zero_grad(set_to_none=True)",
    "slug": "buffer-copy-then-zero-grad-set-none",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The end-of-step bookkeeping in a hand-rolled optimizer has two distinct, atomic moves:\n\n"
        "1. **`buffer-copy_-inplace`** — any per-param state buffer (e.g. an EMA running average, "
        "or just the latest gradient cached for diagnostics) is updated by writing INTO the existing "
        "buffer tensor: `state_buf.copy_(new_value)`. This keeps the buffer's storage stable so "
        "callers holding a reference still see the latest value.\n"
        "2. **`zero-grad-set-none`** — after the step, clear gradients. The PyTorch-default "
        "behaviour is `p.grad = None` (set-to-none), NOT `p.grad.zero_()`. Why: setting to `None` "
        "(a) frees the grad tensor's memory, (b) makes the next `backward()` allocate a fresh grad "
        "(no risk of `+=` accumulating into a stale buffer), (c) skips a kernel launch per param.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "for p in params:\n"
        "    g = p.grad\n"
        "    state_buf.copy_(g)            # buffer-copy_-inplace (e.g. cache last grad)\n"
        "    # ... param update ...\n"
        "    p.grad = None                 # zero-grad-set-none\n"
        "```\n\n"
        "**Why both atoms together.** The buffer holds state that must persist across steps "
        "(carries information forward); the grad must NOT persist across steps (carries stale "
        "information forward). Both are post-step moves; they look similar but have opposite "
        "intent — keep this, drop that."
    ),
    "prompt_body": (
        "Implement `cx8_post_step_update(params, last_grad_buffers)`.\n\n"
        "For each `(p, buf)` pair drawn from the lists:\n\n"
        "1. Cache the current `p.grad` into the existing buffer via `buf.copy_(p.grad)`. The buffer "
        "tensor object must NOT be rebound — `last_grad_buffers[i]` keeps the same `id` and "
        "`data_ptr`.\n"
        "2. Clear `p.grad` by setting it to `None` (set-to-none semantics). Do NOT use "
        "`p.grad.zero_()` — the test catches that variant.\n\n"
        "Return `None`. The test checks:\n"
        "- Buffer values match the pre-call `p.grad` (so the copy_ ran on the right tensor).\n"
        "- Buffer object identity / storage unchanged.\n"
        "- `p.grad is None` after the call.\n"
        "- A fresh `backward()` afterwards re-allocates `p.grad` (proving set-to-none worked)."
    ),
    "stub_body": (
        "def cx8_post_step_update(params, last_grad_buffers):\n"
        "    \"\"\"buf.copy_(p.grad); p.grad = None. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "# Build params with grads via a fake backward.\n"
        "p1 = t.nn.Parameter(t.randn(3))\n"
        "p2 = t.nn.Parameter(t.randn(2, 2))\n"
        "(p1.pow(2).sum() + p2.pow(2).sum()).backward()\n"
        "# Snapshot the grad tensors that exist NOW.\n"
        "g1_snap = p1.grad.clone()\n"
        "g2_snap = p2.grad.clone()\n"
        "\n"
        "buf1 = t.zeros_like(p1)\n"
        "buf2 = t.zeros_like(p2)\n"
        "buf_ids_before = [id(buf1), id(buf2)]\n"
        "buf_ptrs_before = [buf1.data_ptr(), buf2.data_ptr()]\n"
        "\n"
        "ret = cx8_post_step_update([p1, p2], [buf1, buf2])\n"
        "assert ret is None, 'function should return None (mutations are the point)'\n"
        "\n"
        "# Case A: buffers carry the latest grad values, in place.\n"
        "assert t.allclose(buf1, g1_snap), f'buf1 should equal pre-call p1.grad; got {buf1}'\n"
        "assert t.allclose(buf2, g2_snap), 'buf2 mismatch'\n"
        "assert id(buf1) == buf_ids_before[0] and buf1.data_ptr() == buf_ptrs_before[0], (\n"
        "    'buf1 was rebound / reallocated — use buf.copy_(p.grad), not buf = p.grad.clone()'\n"
        ")\n"
        "assert id(buf2) == buf_ids_before[1] and buf2.data_ptr() == buf_ptrs_before[1]\n"
        "\n"
        "# Case B: zero-grad-set-none — p.grad must be exactly None (not a zero tensor).\n"
        "assert p1.grad is None, (\n"
        "    f'p1.grad must be None after set_to_none; got {type(p1.grad).__name__}. '\n"
        "    f'Did you use p.grad.zero_() instead of p.grad = None?'\n"
        ")\n"
        "assert p2.grad is None, 'p2.grad must be None after set_to_none'\n"
        "\n"
        "# Case C: a fresh backward re-allocates grad (proves set-to-none truly frees, not just zeros).\n"
        "(p1.pow(2).sum() + p2.pow(2).sum()).backward()\n"
        "assert p1.grad is not None and p2.grad is not None\n"
        "# And the new grad is fresh (not accumulated into a zeroed prior).\n"
        "expected_g1 = 2 * p1.data\n"
        "assert t.allclose(p1.grad, expected_g1, atol=1e-6), (\n"
        "    'fresh backward grad mismatch — if your impl did p.grad.zero_(), then the next '\n"
        "    'backward could accumulate into the zeroed buffer; if it did p.grad = None, the '\n"
        "    'new grad is a clean allocation'\n"
        ")\n"
        "\n"
        "# Case D: works on a single-param call too (smoke test on shape).\n"
        "p3 = t.nn.Parameter(t.tensor([1.0, -1.0]))\n"
        "p3.pow(2).sum().backward()\n"
        "g3_snap = p3.grad.clone()\n"
        "buf3 = t.zeros_like(p3)\n"
        "cx8_post_step_update([p3], [buf3])\n"
        "assert t.allclose(buf3, g3_snap)\n"
        "assert p3.grad is None"
    ),
    "solution_body": (
        "def cx8_post_step_update(params, last_grad_buffers):\n"
        "    for p, buf in zip(params, last_grad_buffers):\n"
        "        # Atom A (buffer-copy_-inplace): cache the grad into the existing buffer storage.\n"
        "        buf.copy_(p.grad)\n"
        "        # Atom B (zero-grad-set-none): None, NOT p.grad.zero_().\n"
        "        p.grad = None\n"
        "    return None"
    ),
    "solution_notes": (
        "Subtle gotcha: `buf.copy_(p.grad)` runs BEFORE `p.grad = None`. If you swap the order, "
        "`p.grad` is `None` at the time of copy_ and you'd hit a TypeError. The set-to-none flavor "
        "of zero_grad is the PyTorch default since 1.7 — `p.grad.zero_()` is the older behaviour, "
        "kept for backward-compat but slower."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["buffer-copy_-inplace", "zero-grad-set-none"],
    "lo": (
        "Compose buffer.copy_ (cache latest grad into a persistent state buffer in place) with "
        "set-to-none zero_grad (p.grad = None to free memory and force a fresh allocation on the "
        "next backward) into one post-step bookkeeping pass."
    ),
}


# ===========================================================================
# cx9 — canonical SGD-momentum: v=mu*v+g, p-=lr*v
# ===========================================================================
spec_9 = {
    "atom_ids": ["momentum-buffer-update", "inplace-param-update"],
    "subtopics": _subs(["momentum-buffer-update", "inplace-param-update"]),
    "primary_atom": "momentum-buffer-update",
    "part": "part3",
    "exercise_index": 9,
    "exercise_title": "canonical SGD-momentum step: v = mu*v + g, then p -= lr * v",
    "slug": "sgd-momentum-canonical-step",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The textbook SGD-momentum step has two lines of work per parameter:\n\n"
        "1. **`momentum-buffer-update`** — exponentially-decayed accumulator of past gradients: "
        "`v <- mu * v + g`. (Some impls scale `g` by `1 - mu` first; PyTorch's default uses the "
        "form above, matching Sutskever et al.)\n"
        "2. **`inplace-param-update`** — descend along the velocity (not the raw gradient): "
        "`p <- p - lr * v`. Has to be in-place on `p.data` so the caller (and the next forward pass) "
        "sees the new weights.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "for p, v in zip(params, velocity_buffers):\n"
        "    g = p.grad\n"
        "    v.mul_(mu).add_(g)               # momentum-buffer-update (in-place)\n"
        "    p.data.add_(v, alpha=-lr)        # inplace-param-update\n"
        "```\n\n"
        "**Why both atoms together.** Momentum buys you BOTH bigger steps in low-curvature "
        "directions (velocity accumulates) AND damping in oscillatory directions (gradient sign "
        "flips cancel). Neither benefit shows up if the param isn't actually updated — and the "
        "param has to move by the VELOCITY, not the raw gradient. Pairing the two atoms is what "
        "makes the optimizer 'SGD with momentum' rather than 'plain SGD'."
    ),
    "prompt_body": (
        "Implement `cx9_sgd_momentum(params, grads, velocity_buffers, lr, mu)`.\n\n"
        "Apply ONE step of SGD with momentum to every triple `(p, g, v)`:\n\n"
        "1. Update the velocity buffer in place: `v <- mu * v + g`. Use `v.mul_(mu).add_(g)` (or "
        "`v.copy_(mu*v + g)`). The buffer object must persist across calls so subsequent steps "
        "accumulate.\n"
        "2. Update the parameter in place using the NEW velocity: `p.data.add_(v, alpha=-lr)`. The "
        "param must move by `-lr * v` (the velocity), NOT `-lr * g` (the raw gradient).\n\n"
        "Return `None`. Test cross-checks against `torch.optim.SGD(momentum=mu)` over two "
        "consecutive steps. Step 2 is the discriminator: if you used `g` instead of `v` in the param "
        "update, step 1 may still look right (with zero-init buffer, `v == g`) but step 2 will "
        "diverge because the velocity now carries history."
    ),
    "stub_body": (
        "def cx9_sgd_momentum(params, grads, velocity_buffers, lr, mu):\n"
        "    \"\"\"v <- mu*v + g; p.data -= lr * v. In place. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "shapes = [(4,), (3, 2)]\n"
        "params = [t.nn.Parameter(t.randn(s)) for s in shapes]\n"
        "velocity = [t.zeros_like(p) for p in params]\n"
        "ref_params = [t.nn.Parameter(p.data.clone()) for p in params]\n"
        "opt_ref = t.optim.SGD(ref_params, lr=0.05, momentum=0.95)\n"
        "\n"
        "buf_ptrs_before = [b.data_ptr() for b in velocity]\n"
        "param_ptrs_before = [p.data.data_ptr() for p in params]\n"
        "\n"
        "# === Step 1 ===\n"
        "g1 = [t.randn(s) for s in shapes]\n"
        "for rp, g in zip(ref_params, g1):\n"
        "    rp.grad = g.clone()\n"
        "opt_ref.step()\n"
        "cx9_sgd_momentum(params, g1, velocity, lr=0.05, mu=0.95)\n"
        "\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 1 params[{i}] mismatch vs torch.optim.SGD'\n"
        "    )\n"
        "    ref_v = opt_ref.state[ref_params[i]]['momentum_buffer']\n"
        "    assert t.allclose(velocity[i], ref_v, atol=1e-6), (\n"
        "        f'step 1 velocity[{i}] mismatch'\n"
        "    )\n"
        "    assert velocity[i].data_ptr() == buf_ptrs_before[i], (\n"
        "        f'velocity[{i}] storage reallocated — buffer must update in place'\n"
        "    )\n"
        "    assert params[i].data.data_ptr() == param_ptrs_before[i], (\n"
        "        f'params[{i}].data storage reallocated'\n"
        "    )\n"
        "\n"
        "# === Step 2 — the discriminator. ===\n"
        "g2 = [t.randn(s) for s in shapes]\n"
        "for rp, g in zip(ref_params, g2):\n"
        "    rp.grad = g.clone()\n"
        "opt_ref.step()\n"
        "cx9_sgd_momentum(params, g2, velocity, lr=0.05, mu=0.95)\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 2 params[{i}] mismatch — likely used g instead of v in param update '\n"
        "        f'(works at step 1 because v=g when buffer starts at 0)'\n"
        "    )\n"
        "\n"
        "# === Sanity: mu=0 collapses to plain SGD. ===\n"
        "p3 = t.nn.Parameter(t.tensor([5.0, -3.0]))\n"
        "v3 = t.zeros_like(p3)\n"
        "p3_before = p3.data.clone()\n"
        "g3 = [t.tensor([1.0, 1.0])]\n"
        "cx9_sgd_momentum([p3], g3, [v3], lr=0.1, mu=0.0)\n"
        "assert t.allclose(p3.data, p3_before - 0.1 * g3[0], atol=1e-6), 'mu=0 should give plain SGD'\n"
        "assert t.allclose(v3, g3[0], atol=1e-6), 'mu=0: buffer should equal g'\n"
        "\n"
        "# === Sanity: lr=0 with mu>0 leaves params unchanged but BUILDS velocity. ===\n"
        "p4 = t.nn.Parameter(t.tensor([10.0]))\n"
        "v4 = t.zeros_like(p4)\n"
        "p4_before = p4.data.clone()\n"
        "cx9_sgd_momentum([p4], [t.tensor([1.0])], [v4], lr=0.0, mu=0.9)\n"
        "assert t.allclose(p4.data, p4_before), 'lr=0: param should not move'\n"
        "assert t.allclose(v4, t.tensor([1.0])), 'lr=0: velocity should still accumulate'"
    ),
    "solution_body": (
        "def cx9_sgd_momentum(params, grads, velocity_buffers, lr, mu):\n"
        "    for p, g, v in zip(params, grads, velocity_buffers):\n"
        "        # Atom A (momentum-buffer-update): v <- mu*v + g, IN PLACE.\n"
        "        v.mul_(mu).add_(g)\n"
        "        # Atom B (inplace-param-update): step ALONG THE VELOCITY, not the raw gradient.\n"
        "        p.data.add_(v, alpha=-lr)\n"
        "    return None"
    ),
    "solution_notes": (
        "`v.mul_(mu).add_(g)` does the right thing in one fused pass per element — equivalent to "
        "`v.copy_(mu*v + g)` but with no temporary. The most common bug is writing "
        "`p.data.add_(g, alpha=-lr)` instead of `p.data.add_(v, alpha=-lr)` — that's just plain SGD, "
        "and step 1 may even look correct because `v == g` at step 1 (zero-init buffer). Step 2 "
        "diverges immediately. PyTorch's actual SGD has a `dampening` parameter that scales `g` by "
        "`(1 - dampening)` before adding to `v` — default `dampening=0`, so we ignore it here."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["momentum-buffer-update", "inplace-param-update"],
    "lo": (
        "Compose the in-place momentum buffer update (v = mu*v + g) with the in-place parameter "
        "step along the VELOCITY (p -= lr*v, not p -= lr*g) to match torch.optim.SGD(momentum=mu) "
        "across consecutive steps."
    ),
}


# ===========================================================================
# cx10 — momentum buffer maintained via copy_ (literal in-place semantics)
# ===========================================================================
spec_10 = {
    "atom_ids": ["momentum-buffer-update", "buffer-copy_-inplace"],
    "subtopics": _subs(["momentum-buffer-update", "buffer-copy_-inplace"]),
    "primary_atom": "momentum-buffer-update",
    "part": "part3",
    "exercise_index": 10,
    "exercise_title": "momentum buffer update via .copy_() for literal in-place semantics",
    "slug": "momentum-buffer-via-copy",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The momentum recurrence `v <- mu * v + g` can be written three ways. Two are correct:\n\n"
        "- `v.mul_(mu).add_(g)` — fused, no temporary.\n"
        "- `v.copy_(mu * v + g)` — allocates a temporary for `mu*v + g`, then `copy_` writes it "
        "INTO the buffer's existing storage.\n\n"
        "And one is WRONG:\n"
        "- `v = mu * v + g` — rebinds the LOCAL `v`. The caller's list element is untouched.\n\n"
        "This drill picks the second form on purpose: `.copy_()` makes the in-place semantics "
        "LITERAL — the buffer's storage receives the new value verbatim. ARENA's hand-rolled SGD "
        "uses this form because the comment ('this does need to be inplace, since we're modifying "
        "the value in self.b') stays true to the code.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "for p, g, v in zip(params, grads, velocity):\n"
        "    new_v = mu * v + g                # NEW tensor, not yet bound to state.\n"
        "    v.copy_(new_v)                    # WRITE into v's storage; state[p] sees it.\n"
        "    # ... param update happens elsewhere.\n"
        "```\n\n"
        "**Why both atoms together.** `.copy_()` IS the mechanism by which `momentum-buffer-update` "
        "becomes stateful. Pull the `copy_` out and you have a math expression that recomputes the "
        "same thing every step but never carries state forward."
    ),
    "prompt_body": (
        "Implement `cx10_momentum_buffer_via_copy(grads, velocity_buffers, mu)`.\n\n"
        "For each `(g, v)` pair:\n\n"
        "1. Compute the new velocity `new_v = mu * v + g` as a fresh tensor.\n"
        "2. Write it into the existing buffer with `v.copy_(new_v)`. **Use `.copy_()` explicitly** — "
        "the test inspects which approach you used by checking that the buffer's storage pointer "
        "is preserved AND that you didn't rely on the fused `mul_(mu).add_(g)` pattern (we test "
        "with a buffer aliased to an unrelated tensor; if you mutate via `mul_`, the alias "
        "diverges; if you `copy_`, the value lands but the alias is also updated — same storage).\n\n"
        "Return the LIST of updated buffer references (so callers can use them as effective "
        "gradients).\n\n"
        "Test runs THREE consecutive steps to verify momentum accumulates correctly and the "
        "buffer object identity is stable."
    ),
    "stub_body": (
        "def cx10_momentum_buffer_via_copy(grads, velocity_buffers, mu):\n"
        "    \"\"\"v.copy_(mu*v + g) for each pair. Return list of updated buffer refs.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "shapes = [(3,), (2, 2)]\n"
        "velocity = [t.zeros(s) for s in shapes]\n"
        "buf_ids_before = [id(v) for v in velocity]\n"
        "buf_ptrs_before = [v.data_ptr() for v in velocity]\n"
        "\n"
        "# === Step 1: zero-init buffer, v_1 = mu*0 + g = g. ===\n"
        "g1 = [t.tensor([1.0, 2.0, 3.0]), t.tensor([[0.5, -0.5], [1.0, -1.0]])]\n"
        "ret1 = cx10_momentum_buffer_via_copy(g1, velocity, mu=0.9)\n"
        "assert isinstance(ret1, list), 'return a list of buffer references'\n"
        "assert len(ret1) == len(velocity)\n"
        "# Each returned ref IS the buffer (same object) so callers can use it as g_eff.\n"
        "for i in range(len(velocity)):\n"
        "    assert ret1[i] is velocity[i], f'ret[{i}] must BE velocity[{i}], not a copy'\n"
        "    assert id(velocity[i]) == buf_ids_before[i], f'velocity[{i}] rebound'\n"
        "    assert velocity[i].data_ptr() == buf_ptrs_before[i], f'velocity[{i}] storage moved'\n"
        "    assert t.allclose(velocity[i], g1[i]), f'step 1: v should equal g (zero-init buf)'\n"
        "\n"
        "# === Step 2: v_2 = mu * v_1 + g_2 — momentum accumulates. ===\n"
        "g2 = [t.tensor([0.5, 0.5, 0.5]), t.tensor([[0.1, 0.1], [0.1, 0.1]])]\n"
        "expected_v1_step2 = 0.9 * g1[0] + g2[0]  # [1.4, 2.3, 3.2]\n"
        "expected_v2_step2 = 0.9 * g1[1] + g2[1]\n"
        "cx10_momentum_buffer_via_copy(g2, velocity, mu=0.9)\n"
        "assert t.allclose(velocity[0], expected_v1_step2, atol=1e-6), (\n"
        "    f'step 2 velocity[0] should be {expected_v1_step2}, got {velocity[0]}. '\n"
        "    f'If it equals g2 ({g2[0]}), the buffer was rebound at step 1.'\n"
        ")\n"
        "assert t.allclose(velocity[1], expected_v2_step2, atol=1e-6)\n"
        "\n"
        "# === Step 3: cross-check against torch.optim.SGD's internal momentum_buffer. ===\n"
        "t.manual_seed(1)\n"
        "p = t.nn.Parameter(t.randn(5))\n"
        "v = t.zeros(5)\n"
        "opt = t.optim.SGD([t.nn.Parameter(p.data.clone())], lr=1e-9, momentum=0.7)\n"
        "ref_p = list(opt.param_groups[0]['params'])[0]\n"
        "for _ in range(3):\n"
        "    g = t.randn(5)\n"
        "    ref_p.grad = g.clone()\n"
        "    opt.step()\n"
        "    cx10_momentum_buffer_via_copy([g], [v], mu=0.7)\n"
        "ref_buf = opt.state[ref_p]['momentum_buffer']\n"
        "assert t.allclose(v, ref_buf, atol=1e-5), (\n"
        "    f'buffer after 3 steps must match torch.optim.SGD momentum_buffer; '\n"
        "    f'got {v}, want {ref_buf}'\n"
        ")\n"
        "\n"
        "# === Sanity: mu=0 collapses buffer to just g each step. ===\n"
        "v0 = t.tensor([99.0, 99.0])\n"
        "g0 = [t.tensor([1.0, 2.0])]\n"
        "cx10_momentum_buffer_via_copy(g0, [v0], mu=0.0)\n"
        "assert t.allclose(v0, g0[0]), 'mu=0: buffer should be just g, with the old 99s wiped'"
    ),
    "solution_body": (
        "def cx10_momentum_buffer_via_copy(grads, velocity_buffers, mu):\n"
        "    out = []\n"
        "    for g, v in zip(grads, velocity_buffers):\n"
        "        # Atom A (momentum-buffer-update): v <- mu*v + g.\n"
        "        new_v = mu * v + g\n"
        "        # Atom B (buffer-copy_-inplace): write new_v INTO v's existing storage.\n"
        "        v.copy_(new_v)\n"
        "        out.append(v)\n"
        "    return out"
    ),
    "solution_notes": (
        "Using `.copy_()` is slightly less efficient than `v.mul_(mu).add_(g)` (one extra "
        "allocation) but it's more LITERAL — the code reads exactly like the math: 'compute the "
        "new value, then put it into the buffer.' For PyTorch optimizer internals, both forms are "
        "used in the wild; ARENA picks `.copy_()` for pedagogical clarity. Either way, the key is "
        "that the storage at `velocity_buffers[i]` IS the new value when the function returns."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["momentum-buffer-update", "buffer-copy_-inplace"],
    "lo": (
        "Compose the momentum recurrence (v = mu*v + g) with the .copy_() mechanism (write into "
        "the buffer's existing storage so optimizer state persists across steps) to maintain a "
        "stateful velocity buffer that matches torch.optim.SGD's momentum_buffer."
    ),
}


# ===========================================================================
# cx11 — momentum step then zero-grad (set_to_none) — the full mini-loop body
# ===========================================================================
spec_11 = {
    "atom_ids": ["momentum-buffer-update", "zero-grad-set-none"],
    "subtopics": _subs(["momentum-buffer-update", "zero-grad-set-none"]),
    "primary_atom": "momentum-buffer-update",
    "part": "part3",
    "exercise_index": 11,
    "exercise_title": "one SGD-momentum step on the param, then zero_grad with set_to_none",
    "slug": "sgd-momentum-step-then-zero-grad",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Every training-loop iteration has three optimizer-side moves:\n"
        "1. `backward()` populates `p.grad` (the caller does this).\n"
        "2. `optimizer.step()` — your inner work: `momentum-buffer-update` plus the param step.\n"
        "3. `optimizer.zero_grad(set_to_none=True)` — clear grads so the NEXT backward starts fresh.\n\n"
        "**The two atoms.**\n"
        "- **momentum-buffer-update** — `v <- mu * v + g`, in place on the velocity buffer. Mid-step "
        "operation: reads `p.grad`, mutates `v`.\n"
        "- **zero-grad-set-none** — `p.grad = None` AFTER the step. Crucial because PyTorch's "
        "`backward()` ACCUMULATES into `p.grad` if it's already a tensor (this is how RNNs and "
        "multi-loss training work). If you forget to clear, you double-count the previous grad.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "def step(self):\n"
        "    for p, v in zip(self.params, self.velocity):\n"
        "        v.mul_(self.mu).add_(p.grad)         # momentum-buffer-update.\n"
        "        p.data.add_(v, alpha=-self.lr)\n"
        "    for p in self.params:\n"
        "        p.grad = None                         # zero-grad-set-none.\n"
        "```\n\n"
        "**Why both atoms together.** Without zero-grad, momentum compounds with stale grads and "
        "training diverges. Without momentum-buffer-update, you have plain SGD. The pair is the "
        "minimum viable 'one optimizer iteration' on a momentum optimizer."
    ),
    "prompt_body": (
        "Implement `cx11_sgd_momentum_full(params, velocity_buffers, lr, mu)`.\n\n"
        "Assumes the caller has ALREADY called `backward()` so each `p.grad` is a tensor. The "
        "function does TWO things for each param:\n\n"
        "1. Read `p.grad`, update the velocity buffer in place: `v <- mu * v + g`.\n"
        "2. Update the param in place: `p.data.add_(v, alpha=-lr)`.\n"
        "3. AFTER all params have stepped, run a SECOND pass that sets `p.grad = None` for every "
        "param (set-to-none semantics). Two passes are cleaner than one because no one should "
        "set `p.grad = None` while still reading `p.grad` mid-loop, even though here that's safe.\n\n"
        "Return `None`. The test cross-checks the param update against `torch.optim.SGD(momentum=...)`, "
        "then verifies every `p.grad is None`, then runs a fresh backward and verifies the new "
        "grad is freshly allocated (not contaminated by a stale tensor).\n\n"
        "**Gotcha:** if you do `v.mul_(mu).add_(p.grad)` and `p.grad = None` in the SAME loop body, "
        "you must order them correctly (read grad BEFORE setting None). The two-pass design avoids "
        "this entirely."
    ),
    "stub_body": (
        "def cx11_sgd_momentum_full(params, velocity_buffers, lr, mu):\n"
        "    \"\"\"v <- mu*v + g, p -= lr*v, then p.grad = None for all. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "shapes = [(4,), (2, 3)]\n"
        "params = [t.nn.Parameter(t.randn(s)) for s in shapes]\n"
        "velocity = [t.zeros_like(p) for p in params]\n"
        "ref_params = [t.nn.Parameter(p.data.clone()) for p in params]\n"
        "opt_ref = t.optim.SGD(ref_params, lr=0.05, momentum=0.9)\n"
        "\n"
        "def _setup_grads(target_params, ref_target_params):\n"
        "    # Drive a fake loss through both sets of params so both get grads.\n"
        "    loss_a = sum(p.pow(2).sum() for p in target_params)\n"
        "    loss_b = sum(p.pow(2).sum() for p in ref_target_params)\n"
        "    loss_a.backward()\n"
        "    loss_b.backward()\n"
        "\n"
        "# === Step 1 ===\n"
        "_setup_grads(params, ref_params)\n"
        "opt_ref.step()\n"
        "opt_ref.zero_grad(set_to_none=True)  # ref also clears.\n"
        "cx11_sgd_momentum_full(params, velocity, lr=0.05, mu=0.9)\n"
        "\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 1 params[{i}] mismatch'\n"
        "    )\n"
        "    assert params[i].grad is None, (\n"
        "        f'params[{i}].grad must be None after the call; got tensor'\n"
        "    )\n"
        "\n"
        "# === Step 2: fresh backward, then another step ===\n"
        "_setup_grads(params, ref_params)\n"
        "# Verify fresh-backward semantics: grads are real tensors now (not None).\n"
        "for p in params:\n"
        "    assert p.grad is not None, 'fresh backward should re-allocate p.grad'\n"
        "opt_ref.step()\n"
        "opt_ref.zero_grad(set_to_none=True)\n"
        "cx11_sgd_momentum_full(params, velocity, lr=0.05, mu=0.9)\n"
        "\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 2 params[{i}] mismatch — likely a buffer-rebind bug'\n"
        "    )\n"
        "    assert params[i].grad is None\n"
        "\n"
        "# === Sanity: velocity buffer matches torch's momentum_buffer. ===\n"
        "for i in range(len(params)):\n"
        "    ref_v = opt_ref.state[ref_params[i]]['momentum_buffer']\n"
        "    assert t.allclose(velocity[i], ref_v, atol=1e-6), (\n"
        "        f'velocity[{i}] mismatch vs torch internal'\n"
        "    )\n"
        "\n"
        "# === Sanity: no grad-contamination — set-to-none lets a clean fresh allocation happen. ===\n"
        "_setup_grads(params, ref_params)\n"
        "# Expected fresh grads = 2*p (from p.pow(2).sum() backward).\n"
        "for p in params:\n"
        "    assert t.allclose(p.grad, 2 * p.data, atol=1e-6), (\n"
        "        f'fresh-backward grad should be 2*p; got {p.grad}, expected {2 * p.data}'\n"
        "    )"
    ),
    "solution_body": (
        "def cx11_sgd_momentum_full(params, velocity_buffers, lr, mu):\n"
        "    # Pass 1: step (atom A: momentum-buffer-update + inplace param update).\n"
        "    for p, v in zip(params, velocity_buffers):\n"
        "        v.mul_(mu).add_(p.grad)\n"
        "        p.data.add_(v, alpha=-lr)\n"
        "    # Pass 2: clear grads with set-to-none (atom B).\n"
        "    for p in params:\n"
        "        p.grad = None\n"
        "    return None"
    ),
    "solution_notes": (
        "The two-pass design mirrors `torch.optim.Optimizer.step()` + `zero_grad()` — they're "
        "separate methods for exactly this reason. Folding the grad-clear INTO the step is fine "
        "for vanilla SGD-momentum but breaks the moment you want to inspect or log `p.grad` "
        "between step and zero_grad. The set-to-none clear is also what lets your training loop "
        "do `optimizer.step(); optimizer.zero_grad(set_to_none=True)` and pay no per-param "
        "`zero_()` kernel cost."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["momentum-buffer-update", "zero-grad-set-none"],
    "lo": (
        "Compose one full SGD-momentum step (in-place velocity buffer update v=mu*v+g then "
        "p.data.add_(v, alpha=-lr)) with a post-step set-to-none zero_grad pass (p.grad = None) "
        "to form the minimum viable optimizer iteration that matches torch.optim.SGD."
    ),
}


# ===========================================================================
# cx12 — L2 weight decay folded into the grad, then in-place SGD step
# ===========================================================================
spec_12 = {
    "atom_ids": ["weight-decay-l2-add", "inplace-param-update"],
    "subtopics": _subs(["weight-decay-l2-add", "inplace-param-update"]),
    "primary_atom": "weight-decay-l2-add",
    "part": "part3",
    "exercise_index": 12,
    "exercise_title": "fold L2 weight-decay into the gradient, then in-place SGD step",
    "slug": "l2-weight-decay-then-inplace-sgd",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Classical L2 weight-decay is `loss = data_loss + 0.5 * lambda * ||theta||^2`. Its gradient "
        "contribution to each param is just `lambda * theta`. The 'coupled L2' implementation "
        "(what `torch.optim.SGD(weight_decay=lambda)` does) FOLDS this into the gradient BEFORE "
        "the optimizer step:\n\n"
        "1. **`weight-decay-l2-add`** — `g <- g + lambda * p`. The grad picks up a 'pull-toward-zero' "
        "term proportional to the current parameter. This is in-place on `g` if you own the tensor "
        "(but careful — don't mutate `p.grad` itself if you might re-use it elsewhere; here we "
        "treat the grad as ours to mutate).\n"
        "2. **`inplace-param-update`** — `p.data.add_(g, alpha=-lr)`. Same in-place SGD step as "
        "before, but `g` now includes the decay term.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "for p in params:\n"
        "    g = p.grad\n"
        "    if wd != 0:\n"
        "        g = g + wd * p.data         # weight-decay-l2-add (out-of-place here, common style).\n"
        "    p.data.add_(g, alpha=-lr)       # inplace-param-update.\n"
        "```\n\n"
        "Both 'in-place on g' (`g.add_(p.data, alpha=wd)`) and 'rebind g locally' work — the param "
        "update sees the same effective gradient either way. **Caveat:** if you decide to in-place "
        "mutate `p.grad`, the caller's reference now carries the decayed grad too. PyTorch's SGD "
        "uses the local-rebind style to avoid surprising the caller.\n\n"
        "**Why both atoms together.** L2 is the most commonly enabled SGD knob outside the lr. "
        "It only does anything useful if the param update is actually applied — and it has to be "
        "computed BEFORE the param update, since it depends on the CURRENT `p`."
    ),
    "prompt_body": (
        "Implement `cx12_sgd_with_l2(params, grads, lr, weight_decay)`.\n\n"
        "For each `(p, g)` pair:\n\n"
        "1. If `weight_decay != 0`, compute `g_eff = g + weight_decay * p.data`. (You may either "
        "rebind locally OR do `g_eff = g.add(weight_decay * p.data)`; do NOT mutate `p.grad` "
        "itself in place — caller might inspect it.)\n"
        "2. Update the param in place: `p.data.add_(g_eff, alpha=-lr)`. The param's storage must "
        "not be reallocated.\n\n"
        "Return `None`. Test cross-checks against `torch.optim.SGD(weight_decay=wd, momentum=0)` "
        "and verifies:\n"
        "- The L2 term is applied BEFORE the param step (so it uses the CURRENT `p`, not the "
        "post-update one).\n"
        "- `weight_decay=0` collapses to plain SGD.\n"
        "- Larger `weight_decay` pulls params harder toward zero across steps.\n"
        "- Input grads are NOT mutated (caller's reference is preserved)."
    ),
    "stub_body": (
        "def cx12_sgd_with_l2(params, grads, lr, weight_decay):\n"
        "    \"\"\"g_eff = g + wd*p; p.data -= lr * g_eff. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "shapes = [(3,), (2, 2)]\n"
        "params = [t.nn.Parameter(t.randn(s)) for s in shapes]\n"
        "ref_params = [t.nn.Parameter(p.data.clone()) for p in params]\n"
        "opt_ref = t.optim.SGD(ref_params, lr=0.1, momentum=0.0, weight_decay=0.5)\n"
        "\n"
        "param_ptrs_before = [p.data.data_ptr() for p in params]\n"
        "\n"
        "# === Step 1 ===\n"
        "g1 = [t.randn(s) for s in shapes]\n"
        "g1_snapshot = [g.clone() for g in g1]  # to check caller's grads aren't mutated.\n"
        "for rp, g in zip(ref_params, g1):\n"
        "    rp.grad = g.clone()\n"
        "opt_ref.step()\n"
        "cx12_sgd_with_l2(params, g1, lr=0.1, weight_decay=0.5)\n"
        "\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 1 params[{i}] mismatch vs torch.optim.SGD(weight_decay=0.5); '\n"
        "        f'got {params[i].data}, want {ref_params[i].data}'\n"
        "    )\n"
        "    assert params[i].data.data_ptr() == param_ptrs_before[i], (\n"
        "        f'params[{i}].data storage reallocated'\n"
        "    )\n"
        "# Caller's grad tensors must be unchanged.\n"
        "for i in range(len(g1)):\n"
        "    assert t.allclose(g1[i], g1_snapshot[i]), (\n"
        "        f'grads[{i}] was mutated — do NOT in-place modify the caller\\'s grad tensor'\n"
        "    )\n"
        "\n"
        "# === Step 2: verify the L2 uses the CURRENT p (post-step-1), not the original. ===\n"
        "g2 = [t.randn(s) for s in shapes]\n"
        "for rp, g in zip(ref_params, g2):\n"
        "    rp.grad = g.clone()\n"
        "opt_ref.step()\n"
        "cx12_sgd_with_l2(params, g2, lr=0.1, weight_decay=0.5)\n"
        "for i in range(len(params)):\n"
        "    assert t.allclose(params[i].data, ref_params[i].data, atol=1e-6), (\n"
        "        f'step 2 params[{i}] mismatch — L2 must use the CURRENT p each step'\n"
        "    )\n"
        "\n"
        "# === Case: weight_decay=0 collapses to plain SGD. ===\n"
        "p3 = t.nn.Parameter(t.tensor([10.0, -10.0]))\n"
        "p3_before = p3.data.clone()\n"
        "g3 = [t.tensor([1.0, -1.0])]\n"
        "cx12_sgd_with_l2([p3], g3, lr=0.1, weight_decay=0.0)\n"
        "assert t.allclose(p3.data, p3_before - 0.1 * g3[0], atol=1e-6), 'wd=0 must be plain SGD'\n"
        "\n"
        "# === Case: larger weight_decay pulls params toward zero faster (with zero grad). ===\n"
        "p_small_wd = t.nn.Parameter(t.tensor([5.0]))\n"
        "p_big_wd   = t.nn.Parameter(t.tensor([5.0]))\n"
        "for _ in range(10):\n"
        "    cx12_sgd_with_l2([p_small_wd], [t.zeros(1)], lr=0.1, weight_decay=0.1)\n"
        "    cx12_sgd_with_l2([p_big_wd],   [t.zeros(1)], lr=0.1, weight_decay=0.5)\n"
        "assert p_big_wd.data.abs().item() < p_small_wd.data.abs().item(), (\n"
        "    'larger weight_decay should shrink params faster toward zero with grad=0'\n"
        ")\n"
        "# And both should be strictly between 0 and 5 (decay shrinks; doesn\\'t flip sign).\n"
        "assert 0.0 < p_small_wd.data.item() < 5.0\n"
        "assert 0.0 < p_big_wd.data.item() < 5.0"
    ),
    "solution_body": (
        "def cx12_sgd_with_l2(params, grads, lr, weight_decay):\n"
        "    for p, g in zip(params, grads):\n"
        "        # Atom A (weight-decay-l2-add): fold L2 into the effective grad using CURRENT p.\n"
        "        # Out-of-place rebind keeps the caller's g tensor untouched.\n"
        "        if weight_decay != 0:\n"
        "            g_eff = g + weight_decay * p.data\n"
        "        else:\n"
        "            g_eff = g\n"
        "        # Atom B (inplace-param-update): step the param in place, storage unchanged.\n"
        "        p.data.add_(g_eff, alpha=-lr)\n"
        "    return None"
    ),
    "solution_notes": (
        "Two correctness pitfalls:\n"
        "- **Order matters.** L2 has to be folded BEFORE the param update, using the CURRENT `p`. "
        "Reverse the order and the decay term sees the post-step `p`, which is mathematically a "
        "DIFFERENT optimizer (closer to an implicit method).\n"
        "- **Caller's grad.** PyTorch's SGD does NOT mutate `p.grad` when applying weight_decay — "
        "it works on a local copy. We follow that convention here. If your impl does "
        "`g.add_(p.data, alpha=weight_decay)`, the caller's grad tensor gets the decay term "
        "baked in, which surprises code that wanted to log or post-process raw grads.\n"
        "This is 'coupled' L2 (a.k.a. classical L2). AdamW does it DIFFERENTLY — see cx21+ for "
        "decoupled weight decay."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["weight-decay-l2-add", "inplace-param-update"],
    "lo": (
        "Compose the L2 weight-decay grad-fold (g_eff = g + lambda*p, applied BEFORE the param "
        "update and against the CURRENT p) with the in-place param step (p.data.add_(g_eff, "
        "alpha=-lr)) to match torch.optim.SGD(weight_decay=lambda, momentum=0)."
    ),
}


SPECS = [spec_7, spec_8, spec_9, spec_10, spec_11, spec_12]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
