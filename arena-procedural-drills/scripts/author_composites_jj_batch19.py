"""Composite drills cx25..cx30 — batch-19 part3 (JJ-cell, ARENA Adam full step + variants).

Six composite procedural drills exercising 2-3-atom pairs from ARENA part 3 —
Adam optimizer + state buffers + inference-mode plumbing.

cx25  ema-first-moment + ema-second-moment + bias-correction-divide  — Adam full step
cx26  ema-second-moment + buffer-copy_-inplace                        — v stored via buffer copy_
cx27  ema-first-moment  + buffer-copy_-inplace                        — m stored via buffer copy_
cx28  ema-second-moment + inplace-param-update                        — v_hat feeds p in-place update
cx29  ema-second-moment + inference-mode-step                         — Adam state untouched in eval
cx30  ema-second-moment + optimizer-state-tensor-buffers              — v lives in per-param state dict
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# Adam composites mostly need nn / F only when comparing to torch.optim.Adam.
NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
]


# ===========================================================================
# cx25 — Adam full step: dual EMA + bias correction (cross-check vs torch.optim.Adam)
# ===========================================================================
spec_25 = {
    "atom_ids": ["ema-first-moment", "ema-second-moment", "bias-correction-divide"],
    "subtopics": _subs(["ema-first-moment", "ema-second-moment", "bias-correction-divide"]),
    "primary_atom": "ema-first-moment",
    "part": "part3",
    "exercise_index": 25,
    "exercise_title": "full Adam step: dual EMAs + bias correction + param update",
    "slug": "adam-full-step-dual-ema-bias-correction",
    "atom_recap_md": (
        "## How these three atoms compose\n\n"
        "Adam threads three operations through every parameter:\n"
        "1. **EMA of the gradient** (`m`, first moment) — `m = beta1*m + (1-beta1)*g`. "
        "Atom: `ema-first-moment`.\n"
        "2. **EMA of the squared gradient** (`v`, second moment) — "
        "`v = beta2*v + (1-beta2)*g*g`. Atom: `ema-second-moment`.\n"
        "3. **Bias correction** — both EMAs start at 0, so early steps are systematically too small. "
        "Divide by `(1 - beta**t)` to undo the warm-up bias: "
        "`m_hat = m / (1 - beta1**t)`, `v_hat = v / (1 - beta2**t)`. Atom: `bias-correction-divide`.\n\n"
        "Then the parameter update is `p -= lr * m_hat / (sqrt(v_hat) + eps)`.\n\n"
        "**Why all three together.** Either EMA alone is just a one-sided estimator — `m_hat` "
        "gives a momentum-style direction, `v_hat` gives a per-coordinate learning-rate scale. "
        "Bias correction is what makes them comparable to the true expected gradient and squared "
        "gradient at step `t`. Skip it and the first ~1/(1-beta) steps shrink the effective lr by "
        "a factor of (1 - beta**t).\n\n"
        "**Anatomy of one Adam step.**\n"
        "```python\n"
        "t_step += 1\n"
        "for p, m, v in zip(params, ms, vs):\n"
        "    g = p.grad\n"
        "    m.mul_(beta1).add_(g, alpha=1 - beta1)            # ema-first-moment.\n"
        "    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)     # ema-second-moment.\n"
        "    m_hat = m / (1 - beta1 ** t_step)                  # bias-correction-divide.\n"
        "    v_hat = v / (1 - beta2 ** t_step)                  # bias-correction-divide.\n"
        "    p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)\n"
        "```\n\n"
        "We will cross-check the output against `torch.optim.Adam` — the reference Adam in PyTorch."
    ),
    "prompt_body": (
        "Implement `cx25_adam_step(params, ms, vs, t_step, lr, beta1, beta2, eps)`.\n\n"
        "Inputs:\n"
        "- `params` — list of `t.Tensor` parameters, each with `.grad` already populated.\n"
        "- `ms`, `vs` — lists of running buffers, same shapes as the params, holding the current "
        "`m` and `v` EMAs.\n"
        "- `t_step` — the **post-increment** step counter (i.e. for the first step, pass `t_step=1`, "
        "not `0`). Used in `(1 - beta**t_step)`.\n"
        "- `lr`, `beta1`, `beta2`, `eps` — Adam hyperparameters.\n\n"
        "Required behaviour for each `(p, m, v)`:\n"
        "1. Update `m` IN-PLACE with the first-moment EMA: `m = beta1*m + (1-beta1)*g`.\n"
        "2. Update `v` IN-PLACE with the second-moment EMA: `v = beta2*v + (1-beta2)*g*g`.\n"
        "3. Compute `m_hat = m / (1 - beta1**t_step)`, `v_hat = v / (1 - beta2**t_step)` "
        "(may be NEW tensors — they're scratch).\n"
        "4. Update `p` IN-PLACE: `p -= lr * m_hat / (sqrt(v_hat) + eps)`.\n\n"
        "The function should return `None` — it mutates `params`, `ms`, `vs` in place. Run it under "
        "`t.inference_mode()` (or `t.no_grad()`) internally so the in-place leaf update on `p` is "
        "legal.\n\n"
        "The test takes 5 Adam steps with random gradients and cross-checks against "
        "`torch.optim.Adam` — your `params` must end up element-wise equal to the reference."
    ),
    "stub_body": (
        "def cx25_adam_step(params, ms, vs, t_step, lr, beta1, beta2, eps):\n"
        "    \"\"\"Apply one full Adam step in place. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import torch.optim as optim\n"
        "\n"
        "# Set up: two parameters of distinct shapes, both with grad already populated.\n"
        "def _make_setup(seed):\n"
        "    t.manual_seed(seed)\n"
        "    p1 = t.randn(3, 4, requires_grad=True)\n"
        "    p2 = t.randn(5, requires_grad=True)\n"
        "    return [p1, p2]\n"
        "\n"
        "lr, beta1, beta2, eps = 1e-2, 0.9, 0.999, 1e-8\n"
        "\n"
        "# --- Reference path: torch.optim.Adam over 5 steps ---\n"
        "ref_params = _make_setup(seed=0)\n"
        "opt = optim.Adam(ref_params, lr=lr, betas=(beta1, beta2), eps=eps)\n"
        "ref_grads_per_step = []\n"
        "for step in range(5):\n"
        "    t.manual_seed(100 + step)\n"
        "    gs = [t.randn_like(p) for p in ref_params]\n"
        "    ref_grads_per_step.append(gs)\n"
        "    opt.zero_grad()\n"
        "    for p, g in zip(ref_params, gs):\n"
        "        p.grad = g.clone()\n"
        "    opt.step()\n"
        "\n"
        "# --- Our path: cx25_adam_step over 5 steps with identical grads ---\n"
        "my_params = _make_setup(seed=0)\n"
        "ms = [t.zeros_like(p) for p in my_params]\n"
        "vs = [t.zeros_like(p) for p in my_params]\n"
        "for step in range(5):\n"
        "    gs = ref_grads_per_step[step]\n"
        "    for p, g in zip(my_params, gs):\n"
        "        p.grad = g.clone()\n"
        "    cx25_adam_step(my_params, ms, vs, t_step=step + 1, lr=lr, beta1=beta1, beta2=beta2, eps=eps)\n"
        "\n"
        "# Case A: params match torch.optim.Adam element-wise.\n"
        "for i, (mp, rp) in enumerate(zip(my_params, ref_params)):\n"
        "    err = (mp - rp).abs().max().item()\n"
        "    assert err < 1e-5, f'param[{i}] disagrees with torch.optim.Adam: max err {err:.2e}'\n"
        "\n"
        "# Case B: m and v buffers were mutated (not replaced with new tensors).\n"
        "# Re-run one extra step and confirm m, v change in place.\n"
        "m_before_id = id(ms[0])\n"
        "v_before_id = id(vs[0])\n"
        "m_before = ms[0].clone()\n"
        "v_before = vs[0].clone()\n"
        "g_extra = t.randn_like(my_params[0])\n"
        "my_params[0].grad = g_extra.clone()\n"
        "my_params[1].grad = t.zeros_like(my_params[1])\n"
        "cx25_adam_step(my_params, ms, vs, t_step=6, lr=lr, beta1=beta1, beta2=beta2, eps=eps)\n"
        "assert id(ms[0]) == m_before_id, 'ms[0] was replaced — must be updated in place'\n"
        "assert id(vs[0]) == v_before_id, 'vs[0] was replaced — must be updated in place'\n"
        "assert not t.allclose(ms[0], m_before), 'm buffer did not change after a nonzero gradient'\n"
        "assert not t.allclose(vs[0], v_before), 'v buffer did not change after a nonzero gradient'\n"
        "\n"
        "# Case C: bias correction is load-bearing — turn it OFF and the early-step direction\n"
        "# is too small. Sanity-check that step 1 with bias correction matches lr-scale -lr*sign(g).\n"
        "my_params2 = _make_setup(seed=0)\n"
        "ms2 = [t.zeros_like(p) for p in my_params2]\n"
        "vs2 = [t.zeros_like(p) for p in my_params2]\n"
        "g0 = t.ones_like(my_params2[0])\n"
        "g1 = t.ones_like(my_params2[1])\n"
        "my_params2[0].grad = g0\n"
        "my_params2[1].grad = g1\n"
        "p0_before = my_params2[0].detach().clone()\n"
        "cx25_adam_step(my_params2, ms2, vs2, t_step=1, lr=lr, beta1=beta1, beta2=beta2, eps=eps)\n"
        "# After 1 step with grad=1: m_hat=1, v_hat=1, so delta = -lr*1/(sqrt(1)+eps) ~= -lr.\n"
        "delta = my_params2[0] - p0_before\n"
        "assert t.allclose(delta, -lr * t.ones_like(delta), atol=1e-4), (\n"
        "    f'first-step delta should equal -lr (bias correction is missing or wrong); got max '\n"
        "    f'{delta.abs().max().item():.5f}'\n"
        ")"
    ),
    "solution_body": (
        "def cx25_adam_step(params, ms, vs, t_step, lr, beta1, beta2, eps):\n"
        "    # In-place leaf updates on a requires_grad=True tensor need inference_mode/no_grad.\n"
        "    with t.inference_mode():\n"
        "        for p, m, v in zip(params, ms, vs):\n"
        "            g = p.grad\n"
        "            # Atom A (ema-first-moment): m <- beta1*m + (1-beta1)*g.\n"
        "            m.mul_(beta1).add_(g, alpha=1 - beta1)\n"
        "            # Atom B (ema-second-moment): v <- beta2*v + (1-beta2)*g*g.\n"
        "            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)\n"
        "            # Atom C (bias-correction-divide): undo warm-up bias of both EMAs.\n"
        "            bc1 = 1 - beta1 ** t_step\n"
        "            bc2 = 1 - beta2 ** t_step\n"
        "            m_hat = m / bc1\n"
        "            v_hat = v / bc2\n"
        "            # In-place param update: p -= lr * m_hat / (sqrt(v_hat) + eps).\n"
        "            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)"
    ),
    "solution_notes": (
        "Cross-checking against `torch.optim.Adam` is the gold standard — PyTorch's reference Adam "
        "uses exactly this update (per-param `m`, `v`, bias-corrected via division). The `v_hat.sqrt().add_(eps)` "
        "is an in-place mutation of the scratch tensor (no aliasing risk because `v_hat` was created "
        "by the `/` op). If your delta on step 1 does NOT equal `-lr` for a unit gradient, you're "
        "almost certainly missing the bias correction — without it, the step-1 delta is "
        "`-lr * (1 - beta1) / sqrt(1 - beta2)` ≈ `-lr * 0.1 / 0.0316` ≈ `-3.16 * lr`. Counter-"
        "intuitively, MISSING bias correction makes the FIRST step way too big, not too small "
        "(because `sqrt(v_hat)` is divided by a smaller number than `m_hat`)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["ema-first-moment", "ema-second-moment", "bias-correction-divide"],
    "lo": (
        "Compose the two EMA updates (first and second moment) with bias correction to implement "
        "one full Adam step that matches torch.optim.Adam element-wise after multiple iterations."
    ),
}


# ===========================================================================
# cx26 — v state stored via buffer copy_  (ema-second-moment + buffer-copy_-inplace)
# ===========================================================================
spec_26 = {
    "atom_ids": ["ema-second-moment", "buffer-copy_-inplace"],
    "subtopics": _subs(["ema-second-moment", "buffer-copy_-inplace"]),
    "primary_atom": "ema-second-moment",
    "part": "part3",
    "exercise_index": 26,
    "exercise_title": "v buffer updated via in-place copy_ (no rebinding)",
    "slug": "ema-second-moment-via-buffer-copy",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Adam's second-moment `v` is a per-parameter buffer that *evolves* across steps. The naive "
        "way to update it is `v = beta2 * v + (1 - beta2) * g * g`, but this REBINDS the local "
        "name to a new tensor — the original buffer (registered via `register_buffer` or held in an "
        "optimizer state dict) keeps its old value forever.\n\n"
        "The fix: compute the new value out-of-place, then **copy_** it back into the same storage.\n\n"
        "**The two atoms.**\n"
        "- **ema-second-moment** — the formula `v_new = beta2 * v + (1 - beta2) * g**2`.\n"
        "- **buffer-copy_-inplace** — `v.copy_(v_new)` rather than `v = v_new`. `copy_` writes into "
        "the EXISTING tensor's storage, so any reference held elsewhere (`state_dict`, "
        "`register_buffer` entry, optimizer dict) sees the update.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "v_new = beta2 * v + (1 - beta2) * g * g     # ema-second-moment (out-of-place compute).\n"
        "v.copy_(v_new)                              # buffer-copy_-inplace (write back).\n"
        "```\n\n"
        "This composition is the *manual* pattern; `v.mul_(beta2).addcmul_(g, g, value=1-beta2)` is "
        "the more efficient fused-in-place pattern. We test the EXPLICIT compute-then-copy pattern "
        "here because that's what makes the buffer-aliasing contract visible."
    ),
    "prompt_body": (
        "Implement `cx26_update_v_via_copy(v, g, beta2)`.\n\n"
        "Required behaviour:\n"
        "1. Compute `v_new = beta2 * v + (1 - beta2) * g * g` (the second-moment EMA, atom: "
        "ema-second-moment).\n"
        "2. Write the result back into `v` using `v.copy_(v_new)` (atom: buffer-copy_-inplace).\n"
        "3. Return `None`. The caller's reference to `v` MUST be mutated in place; "
        "`id(v)` MUST be unchanged.\n\n"
        "Specifically: the test creates `v`, takes a snapshot of `id(v)`, calls "
        "`cx26_update_v_via_copy(v, g, 0.999)`, and asserts:\n"
        "- `id(v)` is still the same (you did NOT rebind).\n"
        "- `v.data_ptr()` is still the same storage (you did NOT detach / clone-and-overwrite).\n"
        "- `v` element-wise equals `0.999 * v_before + 0.001 * g**2`.\n"
        "- The function is robust to `g` having `requires_grad=True` (gradient should not flow "
        "through the buffer update — wrap with `t.no_grad()` if needed)."
    ),
    "stub_body": (
        "def cx26_update_v_via_copy(v, g, beta2):\n"
        "    \"\"\"In-place update v <- beta2*v + (1-beta2)*g**2 via copy_. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: in-place semantics — id and data_ptr preserved.\n"
        "t.manual_seed(0)\n"
        "v = t.randn(3, 4)\n"
        "g = t.randn(3, 4)\n"
        "v_id_before = id(v)\n"
        "v_ptr_before = v.data_ptr()\n"
        "v_before = v.clone()\n"
        "beta2 = 0.999\n"
        "ret = cx26_update_v_via_copy(v, g, beta2)\n"
        "assert ret is None, 'function must return None (in-place semantics)'\n"
        "assert id(v) == v_id_before, 'v was rebound — must update in place'\n"
        "assert v.data_ptr() == v_ptr_before, 'v storage changed — copy_ should preserve data_ptr'\n"
        "expected = beta2 * v_before + (1 - beta2) * g * g\n"
        "assert t.allclose(v, expected, atol=1e-7), 'v did not get the EMA-of-squared-gradient update'\n"
        "\n"
        "# Case B: aliasing — buffer held in a dict sees the update too.\n"
        "v2 = t.zeros(5)\n"
        "stash = {'v_ref': v2}\n"
        "g2 = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0])\n"
        "cx26_update_v_via_copy(v2, g2, beta2=0.5)\n"
        "# After: v = 0.5*0 + 0.5*g^2 = 0.5 * [1,4,9,16,25] = [0.5,2,4.5,8,12.5].\n"
        "assert t.allclose(stash['v_ref'], t.tensor([0.5, 2.0, 4.5, 8.0, 12.5]), atol=1e-7), (\n"
        "    'dict-aliased reference did not see the update — proves you rebound instead of copy_'\n"
        ")\n"
        "\n"
        "# Case C: works when g has requires_grad=True — no autograd graph through v.\n"
        "v3 = t.ones(4)\n"
        "g3 = t.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)\n"
        "cx26_update_v_via_copy(v3, g3, beta2=0.9)\n"
        "# v should be 0.9*1 + 0.1*g^2.\n"
        "expected3 = 0.9 * t.ones(4) + 0.1 * g3.detach() ** 2\n"
        "assert t.allclose(v3, expected3, atol=1e-7)\n"
        "# v itself must not require grad — a buffer is not a parameter.\n"
        "assert v3.requires_grad is False, 'v buffer should not require grad after the update'\n"
        "\n"
        "# Case D: multi-step EMA accumulates correctly.\n"
        "v4 = t.zeros(2)\n"
        "for step in range(3):\n"
        "    g_step = t.tensor([1.0, 2.0])\n"
        "    cx26_update_v_via_copy(v4, g_step, beta2=0.5)\n"
        "# After 3 steps with constant g and v0=0:\n"
        "# v1 = 0.5*0 + 0.5*g^2 = 0.5*g^2\n"
        "# v2 = 0.5*0.5*g^2 + 0.5*g^2 = 0.75*g^2\n"
        "# v3 = 0.5*0.75*g^2 + 0.5*g^2 = 0.875*g^2\n"
        "assert t.allclose(v4, 0.875 * t.tensor([1.0, 4.0]), atol=1e-7), (\n"
        "    f'multi-step EMA broken; got {v4.tolist()}'\n"
        ")"
    ),
    "solution_body": (
        "def cx26_update_v_via_copy(v, g, beta2):\n"
        "    # No autograd through buffer updates.\n"
        "    with t.no_grad():\n"
        "        # Atom A (ema-second-moment): compute the new value out-of-place.\n"
        "        v_new = beta2 * v + (1 - beta2) * g * g\n"
        "        # Atom B (buffer-copy_-inplace): write back into the SAME storage.\n"
        "        v.copy_(v_new)"
    ),
    "solution_notes": (
        "The semantic difference between `v = v_new` and `v.copy_(v_new)` only matters when SOMEONE "
        "ELSE holds a reference to the original `v` — `register_buffer`, an optimizer's `state` dict, "
        "or just a separate variable. `copy_` writes into the existing storage; `=` rebinds the "
        "local name. PyTorch's `register_buffer` stores the tensor identity in the module's "
        "`_buffers` dict, so a rebind would orphan the registered version."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-second-moment", "buffer-copy_-inplace"],
    "lo": (
        "Compose the second-moment EMA formula with buffer.copy_ to keep the v state in the same "
        "storage so registered buffers and dict-aliased references see the update."
    ),
}


# ===========================================================================
# cx27 — m state stored via buffer copy_  (ema-first-moment + buffer-copy_-inplace)
# ===========================================================================
spec_27 = {
    "atom_ids": ["ema-first-moment", "buffer-copy_-inplace"],
    "subtopics": _subs(["ema-first-moment", "buffer-copy_-inplace"]),
    "primary_atom": "ema-first-moment",
    "part": "part3",
    "exercise_index": 27,
    "exercise_title": "m buffer updated via in-place copy_ (no rebinding)",
    "slug": "ema-first-moment-via-buffer-copy",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Adam's first-moment `m` is a per-parameter buffer — a running EMA of the GRADIENTS (not "
        "the squared gradients; that's `v`). The same buffer-aliasing trap applies: rebinding "
        "(`m = beta1*m + ...`) orphans the registered buffer; `m.copy_(...)` writes back into the "
        "same storage so anyone holding the old reference sees the update.\n\n"
        "**The two atoms.**\n"
        "- **ema-first-moment** — `m_new = beta1 * m + (1 - beta1) * g`.\n"
        "- **buffer-copy_-inplace** — `m.copy_(m_new)`.\n\n"
        "**Why care about this pattern.** Adam state buffers live in the optimizer's `state[param]` "
        "dict, keyed by parameter identity. If you rebind, the dict still points at the old, stale "
        "tensor — the optimizer effectively starts every step from `m=0` again. The bug is silent "
        "because the formula still 'runs'; only the convergence behaviour looks subtly off."
    ),
    "prompt_body": (
        "Implement `cx27_update_m_via_copy(m, g, beta1)`.\n\n"
        "Required behaviour:\n"
        "1. Compute `m_new = beta1 * m + (1 - beta1) * g` (atom: ema-first-moment).\n"
        "2. Write back via `m.copy_(m_new)` (atom: buffer-copy_-inplace).\n"
        "3. Return `None`. `id(m)` and `m.data_ptr()` MUST be unchanged.\n\n"
        "The test checks: in-place semantics, numerical correctness over a few steps, "
        "robustness when `g.requires_grad is True`, and aliasing (a dict reference held to "
        "`m` must see the update)."
    ),
    "stub_body": (
        "def cx27_update_m_via_copy(m, g, beta1):\n"
        "    \"\"\"In-place update m <- beta1*m + (1-beta1)*g via copy_. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: in-place semantics.\n"
        "t.manual_seed(0)\n"
        "m = t.randn(3, 4)\n"
        "g = t.randn(3, 4)\n"
        "m_id_before = id(m)\n"
        "m_ptr_before = m.data_ptr()\n"
        "m_before = m.clone()\n"
        "beta1 = 0.9\n"
        "ret = cx27_update_m_via_copy(m, g, beta1)\n"
        "assert ret is None\n"
        "assert id(m) == m_id_before, 'm was rebound'\n"
        "assert m.data_ptr() == m_ptr_before, 'm storage changed — copy_ preserves data_ptr'\n"
        "expected = beta1 * m_before + (1 - beta1) * g\n"
        "assert t.allclose(m, expected, atol=1e-7)\n"
        "\n"
        "# Case B: dict-aliased reference sees the update.\n"
        "m2 = t.zeros(5)\n"
        "stash = {'m_ref': m2}\n"
        "g2 = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0])\n"
        "cx27_update_m_via_copy(m2, g2, beta1=0.5)\n"
        "# m1 = 0.5*0 + 0.5*g = [0.5, 1.0, 1.5, 2.0, 2.5].\n"
        "assert t.allclose(stash['m_ref'], t.tensor([0.5, 1.0, 1.5, 2.0, 2.5]), atol=1e-7), (\n"
        "    'aliased reference did not see the update'\n"
        ")\n"
        "\n"
        "# Case C: works with grad-requiring g.\n"
        "m3 = t.ones(4)\n"
        "g3 = t.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)\n"
        "cx27_update_m_via_copy(m3, g3, beta1=0.9)\n"
        "expected3 = 0.9 * t.ones(4) + 0.1 * g3.detach()\n"
        "assert t.allclose(m3, expected3, atol=1e-7)\n"
        "assert m3.requires_grad is False\n"
        "\n"
        "# Case D: multi-step accumulation.\n"
        "m4 = t.zeros(2)\n"
        "for step in range(3):\n"
        "    g_step = t.tensor([1.0, -1.0])\n"
        "    cx27_update_m_via_copy(m4, g_step, beta1=0.5)\n"
        "# m0=0, m1=0.5*g, m2=0.5*0.5*g+0.5*g=0.75*g, m3=0.5*0.75*g+0.5*g=0.875*g.\n"
        "assert t.allclose(m4, 0.875 * t.tensor([1.0, -1.0]), atol=1e-7), (\n"
        "    f'multi-step EMA broken; got {m4.tolist()}'\n"
        ")"
    ),
    "solution_body": (
        "def cx27_update_m_via_copy(m, g, beta1):\n"
        "    with t.no_grad():\n"
        "        # Atom A (ema-first-moment): m_new = beta1*m + (1-beta1)*g.\n"
        "        m_new = beta1 * m + (1 - beta1) * g\n"
        "        # Atom B (buffer-copy_-inplace): preserve storage identity.\n"
        "        m.copy_(m_new)"
    ),
    "solution_notes": (
        "The contrast with cx26 is just `g` vs `g*g` — the structural lesson (use `copy_` to keep "
        "the buffer aliased) is identical. In real Adam, `m` is also commonly updated via the fused "
        "`m.mul_(beta1).add_(g, alpha=1-beta1)` for one less temporary allocation."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-first-moment", "buffer-copy_-inplace"],
    "lo": (
        "Compose the first-moment EMA formula with buffer.copy_ so the m buffer stays in the same "
        "storage across update steps and registered/aliased references see each update."
    ),
}


# ===========================================================================
# cx28 — v_hat feeds in-place param update  (ema-second-moment + inplace-param-update)
# ===========================================================================
spec_28 = {
    "atom_ids": ["ema-second-moment", "inplace-param-update"],
    "subtopics": _subs(["ema-second-moment", "inplace-param-update"]),
    "primary_atom": "ema-second-moment",
    "part": "part3",
    "exercise_index": 28,
    "exercise_title": "v_hat per-coordinate scale feeds an in-place param update",
    "slug": "v-hat-feeds-inplace-param-update",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Adam's per-coordinate adaptive learning rate comes from `v_hat = v / (1 - beta2**t)`. "
        "The actual parameter step is `p -= lr * g / (sqrt(v_hat) + eps)`. Two atoms have to wire "
        "together:\n\n"
        "1. **ema-second-moment** — keep `v` as a running EMA of `g*g`, then bias-correct to "
        "`v_hat`. `v_hat` quantifies how 'noisy' each coordinate has been historically — large "
        "`v_hat` → small effective step on that coordinate.\n"
        "2. **inplace-param-update** — apply the step to the parameter IN PLACE: "
        "`p.data.addcdiv_(g, sqrt(v_hat) + eps, value=-lr)` or "
        "`p.data -= lr * g / (sqrt(v_hat) + eps)`. The in-place form is critical when `p` is "
        "wrapped as `nn.Parameter` and may have `requires_grad=True` — out-of-place rebinding "
        "would break the optimizer's hold on the tensor.\n\n"
        "**Anatomy (Adam variant: RMSProp — just the v-side, no momentum).**\n"
        "```python\n"
        "v.mul_(beta2).addcmul_(g, g, value=1 - beta2)           # ema-second-moment.\n"
        "v_hat = v / (1 - beta2 ** t_step)\n"
        "p.data.addcdiv_(g, v_hat.sqrt().add_(eps), value=-lr)   # inplace-param-update.\n"
        "```\n\n"
        "We test the **RMSProp-style** update (no momentum on the gradient — just adaptive scaling) "
        "to isolate the v-to-param wiring. The first-moment is exercised separately in cx25/cx27."
    ),
    "prompt_body": (
        "Implement `cx28_rmsprop_step(p, v, g, t_step, lr, beta2, eps)`.\n\n"
        "Inputs:\n"
        "- `p` — a `t.Tensor` parameter (may have `requires_grad=True`).\n"
        "- `v` — running second-moment buffer (same shape as `p`).\n"
        "- `g` — gradient for this step.\n"
        "- `t_step` — post-increment step counter (`1` on the first call).\n"
        "- `lr`, `beta2`, `eps` — hyperparams.\n\n"
        "Required behaviour:\n"
        "1. Update `v` IN PLACE with the second-moment EMA: `v <- beta2*v + (1-beta2)*g**2`.\n"
        "2. Compute `v_hat = v / (1 - beta2**t_step)` (may be a new scratch tensor).\n"
        "3. Update `p` IN PLACE: `p -= lr * g / (sqrt(v_hat) + eps)`. The id and data_ptr of `p` "
        "MUST be preserved; this is what `inplace-param-update` means.\n"
        "4. Return `None`.\n\n"
        "Both `v` and `p` must be mutated in place. The test confirms:\n"
        "- `id(p)`, `id(v)` unchanged.\n"
        "- `p.data_ptr()`, `v.data_ptr()` unchanged.\n"
        "- After step 1 with unit gradient and `v=0`: `v == 1 - beta2`, `v_hat == 1`, "
        "so `p_new == p_old - lr / (1 + eps)`.\n"
        "- Multi-step trajectory matches the manual formula."
    ),
    "stub_body": (
        "def cx28_rmsprop_step(p, v, g, t_step, lr, beta2, eps):\n"
        "    \"\"\"One RMSProp-style step using v_hat. Mutates p, v in place. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: in-place semantics — both p and v keep their identity.\n"
        "t.manual_seed(0)\n"
        "p = t.randn(3, 4, requires_grad=True)\n"
        "v = t.zeros_like(p)\n"
        "p_id, v_id = id(p), id(v)\n"
        "p_ptr, v_ptr = p.data_ptr(), v.data_ptr()\n"
        "p_before = p.detach().clone()\n"
        "g = t.ones_like(p)  # unit gradient.\n"
        "lr, beta2, eps = 1e-2, 0.999, 1e-8\n"
        "ret = cx28_rmsprop_step(p, v, g, t_step=1, lr=lr, beta2=beta2, eps=eps)\n"
        "assert ret is None\n"
        "assert id(p) == p_id and p.data_ptr() == p_ptr, 'p must update in place'\n"
        "assert id(v) == v_id and v.data_ptr() == v_ptr, 'v must update in place'\n"
        "\n"
        "# Case B: step-1 numerical contract.\n"
        "# v after step 1 with v0=0, g=1: v = beta2*0 + (1-beta2)*1 = (1-beta2).\n"
        "assert t.allclose(v, (1 - beta2) * t.ones_like(v), atol=1e-7), (\n"
        "    f'v after step 1 should equal (1-beta2)={1-beta2}; got {v[0,0].item()}'\n"
        ")\n"
        "# v_hat = v / (1-beta2**1) = (1-beta2) / (1-beta2) = 1 -> sqrt(v_hat)+eps ≈ 1.\n"
        "# Delta on p: -lr * g / (sqrt(v_hat)+eps) ≈ -lr.\n"
        "delta = p.detach() - p_before\n"
        "assert t.allclose(delta, -lr * t.ones_like(delta), atol=1e-4), (\n"
        "    f'step-1 delta should be ~-lr; got max abs err {(delta + lr).abs().max().item():.2e}'\n"
        ")\n"
        "\n"
        "# Case C: multi-step trajectory matches the manual formula.\n"
        "t.manual_seed(1)\n"
        "p2 = t.randn(5, requires_grad=True)\n"
        "v2 = t.zeros(5)\n"
        "p2_before = p2.detach().clone()\n"
        "v2_ref = t.zeros(5)\n"
        "p2_ref = p2_before.clone()\n"
        "for step in range(1, 4):\n"
        "    t.manual_seed(500 + step)\n"
        "    g_step = t.randn(5)\n"
        "    # My step.\n"
        "    cx28_rmsprop_step(p2, v2, g_step, t_step=step, lr=lr, beta2=beta2, eps=eps)\n"
        "    # Reference step.\n"
        "    v2_ref = beta2 * v2_ref + (1 - beta2) * g_step * g_step\n"
        "    v_hat_ref = v2_ref / (1 - beta2 ** step)\n"
        "    p2_ref = p2_ref - lr * g_step / (t.sqrt(v_hat_ref) + eps)\n"
        "assert t.allclose(v2, v2_ref, atol=1e-7), 'v trajectory disagrees with reference'\n"
        "assert t.allclose(p2.detach(), p2_ref, atol=1e-6), (\n"
        "    f'p trajectory disagrees with reference; max err {(p2.detach()-p2_ref).abs().max():.2e}'\n"
        ")\n"
        "\n"
        "# Case D: requires_grad preserved on p (in-place update should not flip it).\n"
        "assert p2.requires_grad is True, 'p must retain requires_grad after in-place update'"
    ),
    "solution_body": (
        "def cx28_rmsprop_step(p, v, g, t_step, lr, beta2, eps):\n"
        "    # In-place leaf update requires inference_mode / no_grad.\n"
        "    with t.inference_mode():\n"
        "        # Atom A (ema-second-moment): v <- beta2*v + (1-beta2)*g*g.\n"
        "        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)\n"
        "        # Bias-correct to v_hat.\n"
        "        v_hat = v / (1 - beta2 ** t_step)\n"
        "        # Atom B (inplace-param-update): p -= lr * g / (sqrt(v_hat) + eps).\n"
        "        p.data.addcdiv_(g, v_hat.sqrt().add_(eps), value=-lr)"
    ),
    "solution_notes": (
        "Using `addcdiv_` on `p.data` (not on `p` itself) keeps the autograd machinery quiet — "
        "the leaf's `.data` is unwatched. The alternative (`with t.no_grad(): p -= ...`) also "
        "works. The key contract `inplace-param-update` enforces is that `p` keeps its identity, "
        "because the optimizer's `state` dict and any external references (e.g. a `Module`'s "
        "`_parameters` dict) are keyed by tensor identity."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-second-moment", "inplace-param-update"],
    "lo": (
        "Compose v's second-moment EMA + bias correction with an in-place parameter update so the "
        "per-coordinate adaptive scaling reaches p without breaking p's tensor identity."
    ),
}


# ===========================================================================
# cx29 — Adam state untouched in eval  (ema-second-moment + inference-mode-step)
# ===========================================================================
spec_29 = {
    "atom_ids": ["ema-second-moment", "inference-mode-step"],
    "subtopics": _subs(["ema-second-moment", "inference-mode-step"]),
    "primary_atom": "inference-mode-step",
    "part": "part3",
    "exercise_index": 29,
    "exercise_title": "Adam state (v EMA) stays frozen under inference_mode",
    "slug": "adam-state-untouched-in-eval",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "An optimizer step is *defined* to mutate its state buffers (`v`, `m`, step counter) — "
        "that's how the EMAs evolve. But during **evaluation** you must NOT call `.step()` at all; "
        "the entire forward+backward pass should run under `t.inference_mode()` (or `t.no_grad()`), "
        "and `optimizer.step()` is simply not invoked. Result: `v` stays frozen.\n\n"
        "The trap students hit: they wrap a *training* step in `inference_mode` (e.g. for a "
        "validation-style 'metrics' batch) and the leaf updates silently disappear — or, worse, "
        "they call `.step()` after a backward that didn't run, leaving `p.grad is None` and the "
        "step skipped without warning.\n\n"
        "**The two atoms.**\n"
        "- **ema-second-moment** — the `v` EMA update inside `.step()`.\n"
        "- **inference-mode-step** — the eval-time discipline: `with t.inference_mode(): forward(x)` "
        "DOES NOT call `.step()`, so `v` is frozen.\n\n"
        "**Anatomy of a `MiniRMSProp` with a `step(do_update: bool)` switch.**\n"
        "```python\n"
        "class MiniRMSProp:\n"
        "    def __init__(self, params, lr, beta2, eps):\n"
        "        self.params = list(params)\n"
        "        self.v = [t.zeros_like(p) for p in self.params]\n"
        "        self.lr, self.beta2, self.eps = lr, beta2, eps\n"
        "        self.t = 0\n"
        "    @t.inference_mode()\n"
        "    def step(self):\n"
        "        self.t += 1\n"
        "        for p, v in zip(self.params, self.v):\n"
        "            g = p.grad\n"
        "            v.mul_(self.beta2).addcmul_(g, g, value=1-self.beta2)  # ema-second-moment.\n"
        "            v_hat = v / (1 - self.beta2 ** self.t)\n"
        "            p.data.addcdiv_(g, v_hat.sqrt().add_(self.eps), value=-self.lr)\n"
        "```\n\n"
        "**Why care.** `model.eval()` flips Module-level `self.training`; "
        "`t.inference_mode()` flips the autograd switch. NEITHER touches the optimizer — that's on "
        "you. If you call `.step()` inside an eval loop, the optimizer happily updates `v` and `p` "
        "using whatever stale `p.grad` happens to be lying around."
    ),
    "prompt_body": (
        "Implement `cx29_make_mini_rmsprop()` — return the `MiniRMSProp` class.\n\n"
        "Required structure:\n"
        "- `__init__(self, params, lr=1e-2, beta2=0.999, eps=1e-8)`:\n"
        "  - `self.params = list(params)`\n"
        "  - `self.v = [t.zeros_like(p) for p in self.params]`\n"
        "  - `self.lr, self.beta2, self.eps = lr, beta2, eps`\n"
        "  - `self.t = 0`  # step counter.\n"
        "- `step(self)` — runs under `t.inference_mode()`. For each `(p, v)`:\n"
        "  - Increment `self.t` (once per call, before the loop).\n"
        "  - `v.mul_(beta2).addcmul_(g, g, value=1-beta2)` (ema-second-moment).\n"
        "  - `v_hat = v / (1 - beta2 ** self.t)`.\n"
        "  - `p.data.addcdiv_(g, v_hat.sqrt().add_(eps), value=-lr)`.\n"
        "- `zero_grad(self)` — sets each `p.grad` to `None` (mirrors `torch.optim.Optimizer.zero_grad(set_to_none=True)`).\n\n"
        "The test runs two scenarios:\n"
        "1. **Training scenario** — populate `p.grad`, call `.step()` repeatedly. Confirm `v` "
        "and `p` evolve.\n"
        "2. **Eval scenario** — wrap a forward pass in `with t.inference_mode():` but DO NOT call "
        "`.step()`. Confirm `v` and `p` are byte-for-byte identical to before — Adam state is "
        "untouched.\n"
        "Then it confirms `step()` mutates `v` in place (id and data_ptr unchanged) and that "
        "step `t` increments by exactly 1 per call."
    ),
    "stub_body": (
        "def cx29_make_mini_rmsprop():\n"
        "    \"\"\"Return the MiniRMSProp class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "MiniRMSProp = cx29_make_mini_rmsprop()\n"
        "\n"
        "# Build a tiny 'model': just one Parameter.\n"
        "t.manual_seed(0)\n"
        "p = t.nn.Parameter(t.randn(3, 4))\n"
        "opt = MiniRMSProp([p], lr=1e-2, beta2=0.999, eps=1e-8)\n"
        "assert opt.t == 0\n"
        "assert len(opt.v) == 1 and opt.v[0].shape == p.shape\n"
        "assert t.allclose(opt.v[0], t.zeros_like(p))\n"
        "\n"
        "# Case A: training — .step() mutates p and v.\n"
        "p.grad = t.ones_like(p)\n"
        "v_id_before = id(opt.v[0])\n"
        "v_ptr_before = opt.v[0].data_ptr()\n"
        "p_id_before = id(p)\n"
        "p_ptr_before = p.data_ptr()\n"
        "p_before = p.detach().clone()\n"
        "opt.step()\n"
        "assert opt.t == 1, f'step counter must be 1 after one step; got {opt.t}'\n"
        "assert id(opt.v[0]) == v_id_before, 'v[0] was rebound'\n"
        "assert opt.v[0].data_ptr() == v_ptr_before, 'v[0] storage changed'\n"
        "assert id(p) == p_id_before and p.data_ptr() == p_ptr_before, 'p identity broken'\n"
        "assert not t.allclose(p.detach(), p_before), 'p should have moved after a training step'\n"
        "assert not t.allclose(opt.v[0], t.zeros_like(p)), 'v should be nonzero after step with grad=1'\n"
        "\n"
        "# Case B: EVAL — wrap a forward in inference_mode, do NOT call .step().\n"
        "v_snapshot = opt.v[0].clone()\n"
        "p_snapshot = p.detach().clone()\n"
        "t_snapshot = opt.t\n"
        "with t.inference_mode():\n"
        "    # Simulate a forward pass — read p, compute some output. Do NOT call .step().\n"
        "    out = p * 2.0\n"
        "    _ = out.sum().item()\n"
        "assert t.allclose(opt.v[0], v_snapshot), 'eval scenario must not change v (no .step() called)'\n"
        "assert t.allclose(p.detach(), p_snapshot), 'eval scenario must not change p'\n"
        "assert opt.t == t_snapshot, 'eval scenario must not increment t'\n"
        "\n"
        "# Case C: zero_grad sets grads to None.\n"
        "opt.zero_grad()\n"
        "assert p.grad is None, 'zero_grad should set p.grad to None'\n"
        "\n"
        "# Case D: multi-step ema-second-moment trajectory matches the manual recurrence.\n"
        "t.manual_seed(7)\n"
        "p2 = t.nn.Parameter(t.randn(5))\n"
        "opt2 = MiniRMSProp([p2], lr=1e-2, beta2=0.9, eps=1e-8)\n"
        "v_ref = t.zeros(5)\n"
        "for step in range(1, 4):\n"
        "    t.manual_seed(step * 11)\n"
        "    g = t.randn(5)\n"
        "    p2.grad = g.clone()\n"
        "    opt2.step()\n"
        "    v_ref = 0.9 * v_ref + 0.1 * g * g\n"
        "assert t.allclose(opt2.v[0], v_ref, atol=1e-7), 'v EMA trajectory disagrees with reference'\n"
        "assert opt2.t == 3, f'after 3 .step() calls, t should be 3; got {opt2.t}'"
    ),
    "solution_body": (
        "def cx29_make_mini_rmsprop():\n"
        "    class MiniRMSProp:\n"
        "        def __init__(self, params, lr=1e-2, beta2=0.999, eps=1e-8):\n"
        "            self.params = list(params)\n"
        "            self.v = [t.zeros_like(p) for p in self.params]\n"
        "            self.lr = lr\n"
        "            self.beta2 = beta2\n"
        "            self.eps = eps\n"
        "            self.t = 0\n"
        "\n"
        "        @t.inference_mode()\n"
        "        def step(self):\n"
        "            # Atom B (inference-mode-step): the decorator turns off autograd for the whole\n"
        "            # body — required because we mutate leaf tensors in place.\n"
        "            self.t += 1\n"
        "            for p, v in zip(self.params, self.v):\n"
        "                g = p.grad\n"
        "                # Atom A (ema-second-moment): in-place v <- beta2*v + (1-beta2)*g*g.\n"
        "                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)\n"
        "                v_hat = v / (1 - self.beta2 ** self.t)\n"
        "                p.data.addcdiv_(g, v_hat.sqrt().add_(self.eps), value=-self.lr)\n"
        "\n"
        "        def zero_grad(self):\n"
        "            for p in self.params:\n"
        "                p.grad = None\n"
        "\n"
        "    return MiniRMSProp"
    ),
    "solution_notes": (
        "Notice the test does NOT call `.step()` inside the eval `with t.inference_mode():` block "
        "— it's only the forward pass. That's the canonical pattern: the inference-mode decorator "
        "on `.step()` is for the LEAF MUTATIONS (so the in-place update on a `requires_grad=True` "
        "tensor is legal); the eval discipline (don't call `.step()` at all) is on the trainer "
        "loop. Both atoms have to be respected in their own scope."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-second-moment", "inference-mode-step"],
    "lo": (
        "Compose the in-place v EMA update (inside an @inference_mode .step) with the eval-time "
        "discipline of NOT calling .step(), so optimizer state is mutated only during training "
        "and frozen during evaluation."
    ),
}


# ===========================================================================
# cx30 — v lives in per-param state dict  (ema-second-moment + optimizer-state-tensor-buffers)
# ===========================================================================
spec_30 = {
    "atom_ids": ["ema-second-moment", "optimizer-state-tensor-buffers"],
    "subtopics": _subs(["ema-second-moment", "optimizer-state-tensor-buffers"]),
    "primary_atom": "optimizer-state-tensor-buffers",
    "part": "part3",
    "exercise_index": 30,
    "exercise_title": "v EMA stored per-param in a state dict (PyTorch optimizer convention)",
    "slug": "v-in-per-param-state-dict",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`torch.optim.Optimizer` stores per-parameter state in a `dict` keyed by parameter "
        "identity: `self.state[p] = {'step': 0, 'exp_avg': ..., 'exp_avg_sq': ...}`. The "
        "second-moment EMA `v` lives there as `exp_avg_sq`. This indirection — instead of a "
        "parallel list `self.v = [...]` like cx29 — is what makes optimizers checkpointable "
        "(`optimizer.state_dict()` serialises the WHOLE per-param state dict) and gives them "
        "the ability to lazily allocate state on first use.\n\n"
        "**The two atoms.**\n"
        "- **optimizer-state-tensor-buffers** — `self.state[p] = {'exp_avg_sq': t.zeros_like(p), ...}` "
        "(lazy or eager). State allocation happens once per parameter, then is reused.\n"
        "- **ema-second-moment** — the second-moment EMA update on whatever buffer the state "
        "dict holds: `state['exp_avg_sq'].mul_(beta2).addcmul_(g, g, value=1-beta2)`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class StatefulRMSProp:\n"
        "    def __init__(self, params, lr, beta2, eps):\n"
        "        self.params = list(params)\n"
        "        self.lr, self.beta2, self.eps = lr, beta2, eps\n"
        "        self.state = {}                              # optimizer-state-tensor-buffers.\n"
        "    @t.inference_mode()\n"
        "    def step(self):\n"
        "        for p in self.params:\n"
        "            if p not in self.state:\n"
        "                # Lazy allocation — first time we see this param.\n"
        "                self.state[p] = {'step': 0, 'exp_avg_sq': t.zeros_like(p)}\n"
        "            s = self.state[p]\n"
        "            s['step'] += 1\n"
        "            g = p.grad\n"
        "            s['exp_avg_sq'].mul_(self.beta2).addcmul_(g, g, value=1-self.beta2)  # ema-second.\n"
        "            v_hat = s['exp_avg_sq'] / (1 - self.beta2 ** s['step'])\n"
        "            p.data.addcdiv_(g, v_hat.sqrt().add_(self.eps), value=-self.lr)\n"
        "```\n\n"
        "**Why care.** Loading a checkpoint into a fresh optimizer relies on `state[p]` shape and "
        "key conventions. PyTorch's `Adam.state_dict()` returns exactly this nested dict (keyed by "
        "*index*, then reattached to params on load)."
    ),
    "prompt_body": (
        "Implement `cx30_make_stateful_rmsprop()` — return the `StatefulRMSProp` class.\n\n"
        "Required structure:\n"
        "- `__init__(self, params, lr=1e-2, beta2=0.999, eps=1e-8)`:\n"
        "  - `self.params = list(params)`, `self.lr, self.beta2, self.eps = lr, beta2, eps`.\n"
        "  - `self.state = {}` — empty dict, keyed by parameter `Tensor` identity.\n"
        "- `step(self)` — runs under `t.inference_mode()`. For each `p` in `self.params`:\n"
        "  - If `p not in self.state`, allocate the state lazily: "
        "`self.state[p] = {'step': 0, 'exp_avg_sq': t.zeros_like(p)}`.\n"
        "  - `s = self.state[p]`; `s['step'] += 1`.\n"
        "  - Update `s['exp_avg_sq']` IN PLACE: "
        "`s['exp_avg_sq'].mul_(beta2).addcmul_(g, g, value=1-beta2)` (atom: ema-second-moment).\n"
        "  - `v_hat = s['exp_avg_sq'] / (1 - beta2 ** s['step'])`.\n"
        "  - `p.data.addcdiv_(g, v_hat.sqrt().add_(eps), value=-lr)`.\n"
        "- `zero_grad(self)` — sets each `p.grad = None`.\n\n"
        "The test verifies the state-dict atom is present:\n"
        "- Before first `.step()`, `self.state == {}`.\n"
        "- After first `.step()` with one param, `len(self.state) == 1` and the key is the "
        "parameter itself (by identity).\n"
        "- `state[p]['exp_avg_sq']` is the SAME tensor across steps (id and data_ptr preserved).\n"
        "- A second parameter, added later, lazily allocates its OWN state entry on first step.\n"
        "- The state-stored `exp_avg_sq` evolves under the EMA recurrence (atom: ema-second-moment)."
    ),
    "stub_body": (
        "def cx30_make_stateful_rmsprop():\n"
        "    \"\"\"Return the StatefulRMSProp class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "StatefulRMSProp = cx30_make_stateful_rmsprop()\n"
        "\n"
        "# Case A: state is empty pre-step; populated lazily on first step.\n"
        "t.manual_seed(0)\n"
        "p = t.nn.Parameter(t.randn(3, 4))\n"
        "opt = StatefulRMSProp([p], lr=1e-2, beta2=0.9, eps=1e-8)\n"
        "assert hasattr(opt, 'state') and isinstance(opt.state, dict)\n"
        "assert len(opt.state) == 0, f'state must be empty before any .step(); got {len(opt.state)}'\n"
        "\n"
        "p.grad = t.ones_like(p)\n"
        "opt.step()\n"
        "assert len(opt.state) == 1, f'after 1 step state should have 1 entry; got {len(opt.state)}'\n"
        "assert p in opt.state, 'state must be keyed by the Parameter tensor identity'\n"
        "s = opt.state[p]\n"
        "assert isinstance(s, dict), f'state[p] must be a dict; got {type(s).__name__}'\n"
        "assert 'exp_avg_sq' in s, f\"state[p] must have key 'exp_avg_sq'; got {list(s.keys())}\"\n"
        "assert 'step' in s, f\"state[p] must have key 'step'; got {list(s.keys())}\"\n"
        "assert s['step'] == 1\n"
        "\n"
        "# Case B: exp_avg_sq is the SAME tensor across steps (in-place updates).\n"
        "v_id_before = id(s['exp_avg_sq'])\n"
        "v_ptr_before = s['exp_avg_sq'].data_ptr()\n"
        "p.grad = t.ones_like(p)\n"
        "opt.step()\n"
        "s = opt.state[p]  # re-fetch — same dict.\n"
        "assert id(s['exp_avg_sq']) == v_id_before, 'exp_avg_sq was rebound across steps'\n"
        "assert s['exp_avg_sq'].data_ptr() == v_ptr_before, 'exp_avg_sq storage changed'\n"
        "assert s['step'] == 2\n"
        "\n"
        "# Case C: numerical EMA trajectory matches manual recurrence.\n"
        "# With grad=1 across all steps, beta2=0.9: v0=0, v1=0.1, v2=0.09+0.1=0.19, v3=0.171+0.1=0.271.\n"
        "p.grad = t.ones_like(p)\n"
        "opt.step()\n"
        "assert t.allclose(opt.state[p]['exp_avg_sq'], 0.271 * t.ones_like(p), atol=1e-6), (\n"
        "    f'EMA trajectory wrong after 3 steps with g=1, beta2=0.9; '\n"
        "    f'expected 0.271, got {opt.state[p][\"exp_avg_sq\"][0,0].item():.4f}'\n"
        ")\n"
        "\n"
        "# Case D: lazy allocation — adding a NEW param later gets its own entry on first step.\n"
        "p2 = t.nn.Parameter(t.zeros(5))\n"
        "opt.params.append(p2)\n"
        "p2.grad = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0])\n"
        "p.grad = t.ones_like(p)\n"
        "opt.step()\n"
        "assert p2 in opt.state, 'new param must lazily allocate state on first step'\n"
        "assert opt.state[p2]['step'] == 1, 'new param state starts at step=1, not 4'\n"
        "# v for p2 after 1 step: 0.9*0 + 0.1*g^2 = 0.1 * [1,4,9,16,25] = [0.1,0.4,0.9,1.6,2.5].\n"
        "expected_v_p2 = t.tensor([0.1, 0.4, 0.9, 1.6, 2.5])\n"
        "assert t.allclose(opt.state[p2]['exp_avg_sq'], expected_v_p2, atol=1e-7)\n"
        "\n"
        "# Case E: zero_grad sets all grads to None.\n"
        "opt.zero_grad()\n"
        "assert p.grad is None and p2.grad is None"
    ),
    "solution_body": (
        "def cx30_make_stateful_rmsprop():\n"
        "    class StatefulRMSProp:\n"
        "        def __init__(self, params, lr=1e-2, beta2=0.999, eps=1e-8):\n"
        "            self.params = list(params)\n"
        "            self.lr = lr\n"
        "            self.beta2 = beta2\n"
        "            self.eps = eps\n"
        "            # Atom B (optimizer-state-tensor-buffers): empty per-param state dict;\n"
        "            # buffers are lazily allocated on first .step().\n"
        "            self.state = {}\n"
        "\n"
        "        @t.inference_mode()\n"
        "        def step(self):\n"
        "            for p in self.params:\n"
        "                if p not in self.state:\n"
        "                    # Lazy alloc: zeros_like(p) gives a buffer with the right shape/dtype/device.\n"
        "                    self.state[p] = {'step': 0, 'exp_avg_sq': t.zeros_like(p)}\n"
        "                s = self.state[p]\n"
        "                s['step'] += 1\n"
        "                g = p.grad\n"
        "                # Atom A (ema-second-moment): in-place EMA update on the dict-stored buffer.\n"
        "                s['exp_avg_sq'].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)\n"
        "                v_hat = s['exp_avg_sq'] / (1 - self.beta2 ** s['step'])\n"
        "                p.data.addcdiv_(g, v_hat.sqrt().add_(self.eps), value=-self.lr)\n"
        "\n"
        "        def zero_grad(self):\n"
        "            for p in self.params:\n"
        "                p.grad = None\n"
        "\n"
        "    return StatefulRMSProp"
    ),
    "solution_notes": (
        "Keying `state` by the Parameter itself (not its index) means moving / reassigning a param "
        "list does NOT lose state — the same tensor identity carries through. PyTorch's `Optimizer` "
        "does this with `param_groups[i]['params'][j]` as the canonical iteration order, but "
        "stores `state` keyed by Tensor. Lazy allocation (`if p not in self.state`) is what lets "
        "`torch.optim.Adam` start with no state on construction and grow it on demand — important "
        "because `zeros_like(p)` has to know the device/dtype, which is easier after `.to(device)` "
        "has been called."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["ema-second-moment", "optimizer-state-tensor-buffers"],
    "lo": (
        "Compose the per-parameter state dict pattern (lazy zeros_like allocation keyed by tensor "
        "identity) with the second-moment EMA update so the v buffer survives across .step() calls "
        "and across multiple parameters."
    ),
}


SPECS = [spec_25, spec_26, spec_27, spec_28, spec_29, spec_30]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
