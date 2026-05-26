"""Composite drills cx1..cx6 — batch-19 part3 (FF-cell, ARENA optim/training SGD basics).

Six composite procedural drills exercising 2-atom pairs from ARENA part 3 —
Optimizer internals (SGD).

cx1  optimizer-init-params-list + inplace-param-update    — __init__ + p.data -= lr*p.grad
cx2  optimizer-init-params-list + optimizer-state-tensor-buffers — __init__ allocates state buffers
cx3  optimizer-init-params-list + zero-grad-set-none      — __init__ + zero_grad(set_to_none=True)
cx4  optimizer-state-tensor-buffers + buffer-copy_-inplace — state buffer updated via copy_
cx5  optimizer-state-tensor-buffers + inplace-param-update — momentum SGD: buffer feeds in-place
cx6  inplace-param-update + zero-grad-set-none            — one full SGD step + zero_grad
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
# cx1 — Optimizer __init__ collects params + step() does in-place update
# ===========================================================================
spec_1 = {
    "atom_ids": ["optimizer-init-params-list", "inplace-param-update"],
    "subtopics": _subs(["optimizer-init-params-list", "inplace-param-update"]),
    "primary_atom": "optimizer-init-params-list",
    "part": "part3",
    "exercise_index": 1,
    "exercise_title": "minimal SGD optimizer: __init__ materializes params, step() updates in-place",
    "slug": "sgd-init-params-and-inplace-step",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A PyTorch optimizer is a thin object that takes an iterable of `nn.Parameter`s and a "
        "learning rate, and exposes a `.step()` method that applies one update rule (SGD, Adam, ...) "
        "in-place to those params.\n\n"
        "The TWO atoms you compose here are the minimum to build a working SGD:\n"
        "1. **optimizer-init-params-list** — in `__init__`, the `params` argument is usually a "
        "GENERATOR (returned by `model.parameters()`). Generators are single-pass, so the optimizer "
        "MUST materialize it into a list (or tuple) once, up front: `self.params = list(params)`. "
        "After that, the same list is iterated by every `.step()` call.\n"
        "2. **inplace-param-update** — inside `.step()`, the update is `p.data -= lr * p.grad` "
        "(or equivalently `p.data.sub_(p.grad, alpha=lr)`). Two crucial details:\n"
        "   - We mutate `p.data`, not `p`, so autograd doesn't track the update as a graph op.\n"
        "   - We mutate IN-PLACE (`-=`, not `p = p - ...`). A non-in-place assign would create a "
        "new tensor and the model's `p` would still point at the OLD weights.\n"
        "   - The full step is wrapped in `t.no_grad()` so autograd doesn't try to build a graph "
        "through the optimizer's own arithmetic.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class SGD:\n"
        "    def __init__(self, params, lr):\n"
        "        self.params = list(params)            # atom A: materialize generator.\n"
        "        self.lr = lr\n"
        "\n"
        "    @t.no_grad()\n"
        "    def step(self):\n"
        "        for p in self.params:\n"
        "            if p.grad is None:\n"
        "                continue\n"
        "            p.data -= self.lr * p.grad        # atom B: in-place update of .data.\n"
        "```\n\n"
        "**Why both atoms together.** Without materializing the generator, the second `.step()` "
        "call would see an empty iterator and silently do nothing. Without the in-place `-=`, the "
        "model would never actually update."
    ),
    "prompt_body": (
        "Implement `cx1_make_sgd()` — return a class `MySGD` (do NOT subclass `torch.optim.Optimizer`; "
        "build a from-scratch class) such that:\n\n"
        "- `MySGD(params, lr)`:\n"
        "  - Materializes `params` into `self.params = list(params)`.\n"
        "  - Stores `self.lr = lr`.\n"
        "- `MySGD.step(self)`:\n"
        "  - For each `p` in `self.params`: if `p.grad is not None`, do `p.data -= self.lr * p.grad`.\n"
        "  - Wrap the loop in `with t.no_grad():` (or decorate with `@t.no_grad()`).\n\n"
        "The test cross-checks against `torch.optim.SGD(..., lr=0.1, momentum=0)` on the SAME "
        "starting parameters and the SAME gradient. After one `.step()`, your params must equal "
        "PyTorch's. The test also passes a GENERATOR (not a list) and calls `.step()` TWICE to "
        "verify the second call still sees the same params."
    ),
    "stub_body": (
        "def cx1_make_sgd():\n"
        "    \"\"\"Return the MySGD class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "MySGD = cx1_make_sgd()\n"
        "assert isinstance(MySGD, type), 'cx1 must return a class'\n"
        "\n"
        "# Case A: materialize a generator (single-pass) — two steps must both update.\n"
        "t.manual_seed(0)\n"
        "p1 = t.nn.Parameter(t.randn(3, 4))\n"
        "p2 = t.nn.Parameter(t.randn(5))\n"
        "p1.grad = t.ones_like(p1)\n"
        "p2.grad = t.full_like(p2, 2.0)\n"
        "p1_before = p1.detach().clone()\n"
        "p2_before = p2.detach().clone()\n"
        "\n"
        "def param_gen():\n"
        "    yield p1\n"
        "    yield p2\n"
        "\n"
        "opt = MySGD(param_gen(), lr=0.1)\n"
        "opt.step()\n"
        "# After one step: p -= 0.1 * grad.\n"
        "assert t.allclose(p1.data, p1_before - 0.1 * t.ones_like(p1)), 'p1 not updated correctly after step 1'\n"
        "assert t.allclose(p2.data, p2_before - 0.1 * t.full_like(p2, 2.0)), 'p2 not updated correctly after step 1'\n"
        "# Now refresh grads and call step AGAIN — should update again (proves generator was materialized).\n"
        "p1.grad = t.ones_like(p1)\n"
        "p2.grad = t.full_like(p2, 2.0)\n"
        "opt.step()\n"
        "assert t.allclose(p1.data, p1_before - 0.2 * t.ones_like(p1)), 'second step did not update p1 — generator not materialized?'\n"
        "assert t.allclose(p2.data, p2_before - 0.2 * t.full_like(p2, 2.0)), 'second step did not update p2 — generator not materialized?'\n"
        "\n"
        "# Case B: cross-check vs torch.optim.SGD on identical setup.\n"
        "t.manual_seed(1)\n"
        "weight_init = t.randn(7, 3)\n"
        "grad_val = t.randn(7, 3)\n"
        "\n"
        "p_mine = t.nn.Parameter(weight_init.clone())\n"
        "p_mine.grad = grad_val.clone()\n"
        "opt_mine = MySGD([p_mine], lr=0.05)\n"
        "opt_mine.step()\n"
        "\n"
        "p_ref = t.nn.Parameter(weight_init.clone())\n"
        "p_ref.grad = grad_val.clone()\n"
        "opt_ref = t.optim.SGD([p_ref], lr=0.05, momentum=0)\n"
        "opt_ref.step()\n"
        "\n"
        "assert t.allclose(p_mine.data, p_ref.data, atol=1e-7), (\n"
        "    f'MySGD diverges from torch.optim.SGD; max err = {(p_mine.data - p_ref.data).abs().max().item()}'\n"
        ")\n"
        "\n"
        "# Case C: in-place mutation — p_mine is the SAME tensor object before/after step.\n"
        "p3 = t.nn.Parameter(t.randn(4))\n"
        "p3.grad = t.ones_like(p3)\n"
        "tensor_id_before = id(p3)\n"
        "data_id_before = id(p3.data)  # tricky: data identity may stay, but VALUES must change.\n"
        "opt = MySGD([p3], lr=0.1)\n"
        "before_clone = p3.detach().clone()\n"
        "opt.step()\n"
        "assert id(p3) == tensor_id_before, 'param tensor was replaced — must be in-place'\n"
        "assert not t.allclose(p3.data, before_clone), 'p3 was not actually updated'\n"
        "\n"
        "# Case D: p.grad is None — must be skipped, not crash.\n"
        "p4 = t.nn.Parameter(t.randn(2))\n"
        "p4.grad = None\n"
        "p5 = t.nn.Parameter(t.randn(2))\n"
        "p5.grad = t.ones_like(p5)\n"
        "p5_before = p5.detach().clone()\n"
        "opt = MySGD([p4, p5], lr=0.1)\n"
        "opt.step()  # must not crash on p4\n"
        "assert t.allclose(p5.data, p5_before - 0.1), 'p5 should still update even when p4.grad is None'"
    ),
    "solution_body": (
        "def cx1_make_sgd():\n"
        "    class MySGD:\n"
        "        def __init__(self, params, lr):\n"
        "            # Atom A (optimizer-init-params-list): materialize the generator once.\n"
        "            # If we kept the generator object, the second .step() would iterate an exhausted iterator.\n"
        "            self.params = list(params)\n"
        "            self.lr = lr\n"
        "\n"
        "        @t.no_grad()\n"
        "        def step(self):\n"
        "            for p in self.params:\n"
        "                if p.grad is None:\n"
        "                    continue\n"
        "                # Atom B (inplace-param-update): mutate .data in place; do NOT reassign p.\n"
        "                p.data -= self.lr * p.grad\n"
        "\n"
        "    return MySGD"
    ),
    "solution_notes": (
        "Three common bugs the test catches: (1) storing the generator instead of materializing — "
        "second step silently no-ops; (2) writing `p = p - lr * p.grad` instead of `p.data -= ...` "
        "— rebinds the local name without touching the model's parameter; (3) forgetting `t.no_grad()` "
        "— the optimizer math itself enters the autograd graph and you get a slow memory leak. "
        "PyTorch's `Optimizer` base class handles (1) via `param_groups`; here we do it by hand."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["optimizer-init-params-list", "inplace-param-update"],
    "lo": (
        "Compose the optimizer's __init__ (materialize the params generator into a list) with the "
        "in-place param update (p.data -= lr*p.grad inside t.no_grad()) to implement a minimal "
        "SGD optimizer that matches torch.optim.SGD."
    ),
}


# ===========================================================================
# cx2 — Optimizer __init__ allocates per-param state tensor buffers
# ===========================================================================
spec_2 = {
    "atom_ids": ["optimizer-init-params-list", "optimizer-state-tensor-buffers"],
    "subtopics": _subs(["optimizer-init-params-list", "optimizer-state-tensor-buffers"]),
    "primary_atom": "optimizer-init-params-list",
    "part": "part3",
    "exercise_index": 2,
    "exercise_title": "momentum SGD __init__: materialize params AND allocate a velocity buffer per param",
    "slug": "sgd-init-with-velocity-buffers",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "SGD with momentum needs ONE extra tensor per parameter: the **velocity** `v`, which has the "
        "same shape as the param and starts at zero. Adam needs TWO (first and second moments). "
        "These per-param tensors are called the optimizer's **state**.\n\n"
        "The atom that captures this is **optimizer-state-tensor-buffers**: at `__init__` time the "
        "optimizer allocates one `zeros_like(p)` buffer per param, and at `.step()` time it reads "
        "and writes those buffers in-place.\n\n"
        "It composes with **optimizer-init-params-list** because the allocation has to happen "
        "AFTER the param list has been materialized — you need a fixed list to align the state "
        "buffers against. ARENA's `SGD.__init__` and `Adam.__init__` both follow this exact pattern.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class SGDMomentum:\n"
        "    def __init__(self, params, lr, momentum=0.9):\n"
        "        self.params = list(params)                          # atom A.\n"
        "        self.lr = lr\n"
        "        self.momentum = momentum\n"
        "        # atom B: ONE velocity buffer per param, shaped like the param, zero-init.\n"
        "        self.velocities = [t.zeros_like(p) for p in self.params]\n"
        "```\n\n"
        "Two layout choices for the state. ARENA uses parallel lists `self.params` and "
        "`self.velocities`, indexed together. PyTorch uses a dict keyed by id(param). Both work; "
        "parallel lists are simpler when you write the optimizer by hand."
    ),
    "prompt_body": (
        "Implement `cx2_make_sgd_momentum_init()` — return a class `SGDM` such that:\n\n"
        "- `SGDM(params, lr, momentum=0.9)`:\n"
        "  - `self.params = list(params)` — materialize the generator (atom A).\n"
        "  - `self.lr = lr; self.momentum = momentum`.\n"
        "  - `self.velocities = [t.zeros_like(p) for p in self.params]` — one buffer per param, "
        "shape-aligned, zero-init (atom B).\n\n"
        "You do NOT need to implement `.step()` for this drill — the test only checks the "
        "initialisation. (A follow-up drill, cx5, wires the velocity buffer into the update rule.)\n\n"
        "The test checks: (a) `self.params` is a list (not a generator); (b) `self.velocities` is "
        "the same length as `self.params`; (c) each velocity is shape-aligned with its param; "
        "(d) each velocity is all-zero; (e) the velocities are FRESH tensors (not aliased to the "
        "params themselves)."
    ),
    "stub_body": (
        "def cx2_make_sgd_momentum_init():\n"
        "    \"\"\"Return the SGDM class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "SGDM = cx2_make_sgd_momentum_init()\n"
        "assert isinstance(SGDM, type)\n"
        "\n"
        "# Case A: materializes generator + allocates velocities.\n"
        "t.manual_seed(0)\n"
        "p1 = t.nn.Parameter(t.randn(3, 4))\n"
        "p2 = t.nn.Parameter(t.randn(5))\n"
        "p3 = t.nn.Parameter(t.randn(2, 2, 2))\n"
        "\n"
        "def gen():\n"
        "    yield p1; yield p2; yield p3\n"
        "\n"
        "opt = SGDM(gen(), lr=0.01, momentum=0.9)\n"
        "assert isinstance(opt.params, list), f'self.params must be a list, got {type(opt.params).__name__}'\n"
        "assert len(opt.params) == 3, f'expected 3 params materialized, got {len(opt.params)}'\n"
        "\n"
        "# Case B: velocities exist and align by length.\n"
        "assert hasattr(opt, 'velocities'), 'optimizer must expose self.velocities'\n"
        "assert len(opt.velocities) == 3, f'expected 3 velocity buffers, got {len(opt.velocities)}'\n"
        "\n"
        "# Case C: velocity shapes match param shapes.\n"
        "for i, (p, v) in enumerate(zip(opt.params, opt.velocities)):\n"
        "    assert v.shape == p.shape, f'velocity[{i}] shape {tuple(v.shape)} != param shape {tuple(p.shape)}'\n"
        "\n"
        "# Case D: velocities are zero.\n"
        "for i, v in enumerate(opt.velocities):\n"
        "    assert t.all(v == 0).item(), f'velocity[{i}] must start at zero'\n"
        "\n"
        "# Case E: velocities are FRESH tensors — not aliasing the params (mutating them must NOT touch p).\n"
        "for i, (p, v) in enumerate(zip(opt.params, opt.velocities)):\n"
        "    p_before = p.detach().clone()\n"
        "    v.add_(1.0)  # mutate the velocity buffer in-place.\n"
        "    assert t.allclose(p.data, p_before), f'mutating velocity[{i}] changed param[{i}] — they alias!'\n"
        "\n"
        "# Case F: hyperparams stored.\n"
        "assert opt.lr == 0.01\n"
        "assert opt.momentum == 0.9\n"
        "\n"
        "# Case G: dtype/device should also match (sanity for float32 CPU case).\n"
        "for p, v in zip(opt.params, opt.velocities):\n"
        "    assert v.dtype == p.dtype, f'velocity dtype {v.dtype} != param dtype {p.dtype}'\n"
        "    assert v.device == p.device, f'velocity device {v.device} != param device {p.device}'"
    ),
    "solution_body": (
        "def cx2_make_sgd_momentum_init():\n"
        "    class SGDM:\n"
        "        def __init__(self, params, lr, momentum=0.9):\n"
        "            # Atom A: materialize the generator FIRST so we can align state against it.\n"
        "            self.params = list(params)\n"
        "            self.lr = lr\n"
        "            self.momentum = momentum\n"
        "            # Atom B: one fresh zeros_like buffer per param. zeros_like preserves\n"
        "            # shape, dtype, and device — which is exactly what we want.\n"
        "            self.velocities = [t.zeros_like(p) for p in self.params]\n"
        "\n"
        "    return SGDM"
    ),
    "solution_notes": (
        "`t.zeros_like(p)` is doing the work of three checks at once: it picks the right shape, "
        "the right dtype, and the right device. Beginners sometimes write `t.zeros(p.shape)` and "
        "discover at the first GPU run that the velocity is on CPU and the param is on CUDA, so "
        "the `v + p` arithmetic crashes. The parallel-list layout (params + velocities indexed "
        "together) is the same layout ARENA uses in `solutions.py`."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["optimizer-init-params-list", "optimizer-state-tensor-buffers"],
    "lo": (
        "Compose the optimizer's __init__ (materialize params generator into a list) with the "
        "allocation of per-param state tensor buffers (zeros_like, shape/dtype/device-aligned) to "
        "set up the storage that a momentum-SGD step() will read and write."
    ),
}


# ===========================================================================
# cx3 — Optimizer __init__ + zero_grad(set_to_none=True)
# ===========================================================================
spec_3 = {
    "atom_ids": ["optimizer-init-params-list", "zero-grad-set-none"],
    "subtopics": _subs(["optimizer-init-params-list", "zero-grad-set-none"]),
    "primary_atom": "optimizer-init-params-list",
    "part": "part3",
    "exercise_index": 3,
    "exercise_title": "optimizer __init__ + zero_grad(set_to_none=True)",
    "slug": "optimizer-init-plus-zero-grad-set-none",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Every training step in PyTorch looks like:\n"
        "```\n"
        "loss.backward()    # accumulates grads INTO p.grad (with += !).\n"
        "optimizer.step()   # update params.\n"
        "optimizer.zero_grad()  # clear grads for the next iteration.\n"
        "```\n\n"
        "The clear is mandatory. PyTorch's `backward()` ACCUMULATES into `p.grad` — if you skip "
        "the clear, the next batch's gradient is added on top of the previous one, and you train "
        "on a running sum of gradients (silent bug — loss looks like it's going down 'differently').\n\n"
        "There are two ways to clear:\n"
        "- `p.grad.zero_()` — in-place zero. Keeps the gradient tensor allocated.\n"
        "- `p.grad = None` — drop the reference entirely. The NEXT backward call will re-allocate.\n\n"
        "**zero-grad-set-none** is the second approach. PyTorch ≥ 1.7 made `set_to_none=True` the "
        "default in `Optimizer.zero_grad()` for two reasons: (1) saves memory between optim "
        "steps; (2) makes 'I forgot to zero_grad' show up as a `NoneType` crash instead of a "
        "silent accumulation bug.\n\n"
        "**Composition with optimizer-init-params-list.** `zero_grad` iterates `self.params` — "
        "which only exists because `__init__` materialized the generator into a list. With a raw "
        "generator, the FIRST `zero_grad` call would empty the iterator and `.step()` would "
        "silently no-op.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class SGD:\n"
        "    def __init__(self, params, lr):\n"
        "        self.params = list(params)              # atom A.\n"
        "        self.lr = lr\n"
        "\n"
        "    def zero_grad(self, set_to_none=True):\n"
        "        for p in self.params:\n"
        "            if set_to_none:\n"
        "                p.grad = None                   # atom B: drop the reference.\n"
        "            else:\n"
        "                if p.grad is not None:\n"
        "                    p.grad.zero_()\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx3_make_sgd_with_zero_grad()` — return a class `MySGD2`:\n\n"
        "- `MySGD2(params, lr)`:\n"
        "  - `self.params = list(params)` (atom A).\n"
        "  - `self.lr = lr`.\n"
        "- `MySGD2.zero_grad(self, set_to_none=True)`:\n"
        "  - If `set_to_none`: for each param, set `p.grad = None` (atom B).\n"
        "  - Else: for each param with non-None grad, call `p.grad.zero_()` in place.\n\n"
        "(You do NOT need to implement `.step()` for this drill.)\n\n"
        "The test checks: (a) after `zero_grad(set_to_none=True)`, every param's `.grad` is "
        "literally `None`; (b) after `zero_grad(set_to_none=False)`, every param's `.grad` is a "
        "tensor of zeros (not None); (c) the optimizer works when constructed from a generator "
        "(materialization atom); (d) calling `zero_grad` doesn't crash when some params already "
        "have `grad=None` (idempotent)."
    ),
    "stub_body": (
        "def cx3_make_sgd_with_zero_grad():\n"
        "    \"\"\"Return the MySGD2 class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "MySGD2 = cx3_make_sgd_with_zero_grad()\n"
        "assert isinstance(MySGD2, type)\n"
        "\n"
        "# Case A: materialize generator, then zero_grad(set_to_none=True) drops every grad.\n"
        "t.manual_seed(0)\n"
        "p1 = t.nn.Parameter(t.randn(3, 4))\n"
        "p2 = t.nn.Parameter(t.randn(5))\n"
        "p3 = t.nn.Parameter(t.randn(2, 2))\n"
        "# Populate grads so we can see them be cleared.\n"
        "p1.grad = t.ones_like(p1)\n"
        "p2.grad = t.full_like(p2, 7.0)\n"
        "p3.grad = t.randn_like(p3)\n"
        "\n"
        "def gen():\n"
        "    yield p1; yield p2; yield p3\n"
        "\n"
        "opt = MySGD2(gen(), lr=0.01)\n"
        "assert isinstance(opt.params, list) and len(opt.params) == 3, 'params must be materialized'\n"
        "opt.zero_grad(set_to_none=True)\n"
        "assert p1.grad is None, 'p1.grad must be None after zero_grad(set_to_none=True)'\n"
        "assert p2.grad is None, 'p2.grad must be None'\n"
        "assert p3.grad is None, 'p3.grad must be None'\n"
        "\n"
        "# Case B: set_to_none=False zeros in place — grads are tensors of zeros, NOT None.\n"
        "p1.grad = t.ones_like(p1)\n"
        "p2.grad = t.full_like(p2, 7.0)\n"
        "p3.grad = t.randn_like(p3)\n"
        "opt.zero_grad(set_to_none=False)\n"
        "assert p1.grad is not None and t.all(p1.grad == 0).item(), 'p1.grad must be zero TENSOR (not None)'\n"
        "assert p2.grad is not None and t.all(p2.grad == 0).item(), 'p2.grad must be zero TENSOR'\n"
        "assert p3.grad is not None and t.all(p3.grad == 0).item(), 'p3.grad must be zero TENSOR'\n"
        "\n"
        "# Case C: default kwarg is set_to_none=True (matches PyTorch ≥ 1.7).\n"
        "p1.grad = t.ones_like(p1)\n"
        "opt.zero_grad()  # no arg — should default to True.\n"
        "assert p1.grad is None, 'default zero_grad() should set grads to None'\n"
        "\n"
        "# Case D: idempotent — calling zero_grad twice does not crash on already-None grads.\n"
        "opt.zero_grad(set_to_none=True)  # all grads already None.\n"
        "opt.zero_grad(set_to_none=False)  # call again from the None state.\n"
        "# After set_to_none=False from a None state, an implementation may either leave it None or\n"
        "# allocate a zero tensor — both are OK. Just must not crash.\n"
        "\n"
        "# Case E: prevents gradient accumulation (the bug zero_grad exists to fix).\n"
        "t.manual_seed(1)\n"
        "p = t.nn.Parameter(t.randn(4))\n"
        "opt2 = MySGD2([p], lr=0.1)\n"
        "# Run two 'mini-backwards' — without zero_grad, the second would accumulate onto the first.\n"
        "x = t.randn(4)\n"
        "y1 = (p * x).sum(); y1.backward()\n"
        "g1 = p.grad.clone()\n"
        "opt2.zero_grad(set_to_none=True)\n"
        "y2 = (p * x).sum(); y2.backward()\n"
        "g2 = p.grad.clone()\n"
        "# After zero_grad, g2 should equal g1 (single backward), NOT 2*g1 (accumulated).\n"
        "assert t.allclose(g1, g2), 'gradient accumulated across calls — zero_grad failed to clear'"
    ),
    "solution_body": (
        "def cx3_make_sgd_with_zero_grad():\n"
        "    class MySGD2:\n"
        "        def __init__(self, params, lr):\n"
        "            # Atom A: materialize generator into a list — without this, zero_grad\n"
        "            # would consume the iterator on the first call and step() would do nothing.\n"
        "            self.params = list(params)\n"
        "            self.lr = lr\n"
        "\n"
        "        def zero_grad(self, set_to_none=True):\n"
        "            for p in self.params:\n"
        "                if set_to_none:\n"
        "                    # Atom B: drop the reference. Next backward() will re-allocate.\n"
        "                    p.grad = None\n"
        "                else:\n"
        "                    if p.grad is not None:\n"
        "                        p.grad.zero_()  # in-place zero of the existing tensor.\n"
        "\n"
        "    return MySGD2"
    ),
    "solution_notes": (
        "Why `set_to_none=True` became the default in PyTorch 1.7+: (1) memory — the gradient "
        "tensors aren't kept alive between optim steps; (2) bug surfacing — if you forget to call "
        "`zero_grad` before the next `backward()`, the accumulation either happens (bad) or now "
        "crashes (good, because `p.grad` is None and the `+=` inside backward needs to allocate "
        "anyway). The `if p.grad is not None` guard in the `set_to_none=False` branch matters: "
        "calling `.zero_()` on `None` would crash."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["optimizer-init-params-list", "zero-grad-set-none"],
    "lo": (
        "Compose the optimizer's __init__ (materialize the params generator into self.params) "
        "with zero_grad(set_to_none=True/False) (drop the gradient reference vs in-place zero) "
        "to implement the gradient-clearing half of the canonical PyTorch training loop."
    ),
}


# ===========================================================================
# cx4 — Allocate state buffer + update it via copy_
# ===========================================================================
spec_4 = {
    "atom_ids": ["optimizer-state-tensor-buffers", "buffer-copy_-inplace"],
    "subtopics": _subs(["optimizer-state-tensor-buffers", "buffer-copy_-inplace"]),
    "primary_atom": "optimizer-state-tensor-buffers",
    "part": "part3",
    "exercise_index": 4,
    "exercise_title": "allocate velocity buffer, then update it via copy_ (in-place)",
    "slug": "velocity-buffer-update-via-copy",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "When an optimizer maintains state buffers (e.g. SGD's velocity, Adam's first/second "
        "moments), the update rule MUST mutate those buffers IN PLACE. If you write "
        "`self.velocities[i] = momentum * v + g`, you've created a NEW tensor and rebound the list "
        "slot — fine for the next step, but breaks anyone who held a reference to the OLD buffer "
        "(e.g. a `.to(device)` call, a `state_dict` snapshot, or an external param-group view).\n\n"
        "ARENA's convention (and PyTorch's): compute the new buffer value as a fresh tensor, then "
        "`buffer.copy_(new_value)` to write it back in place. `copy_` is the **broadcast-aware "
        "in-place copy** — it preserves the underlying storage of `buffer` while overwriting all "
        "of its elements with the source's values.\n\n"
        "**The two atoms.**\n"
        "- **optimizer-state-tensor-buffers** (atom A) — `self.velocities[i]` is a tensor "
        "allocated at `__init__` with the right shape/dtype/device.\n"
        "- **buffer-copy_-inplace** (atom B) — `self.velocities[i].copy_(new_velocity)` writes "
        "the next-iteration value back into the SAME tensor.\n\n"
        "**Anatomy of one momentum step.**\n"
        "```python\n"
        "for p, v in zip(self.params, self.velocities):\n"
        "    if p.grad is None: continue\n"
        "    g = p.grad\n"
        "    new_v = self.momentum * v + g          # fresh tensor.\n"
        "    v.copy_(new_v)                         # write into the SAME buffer (atom B).\n"
        "    p.data -= self.lr * v                  # use the just-updated v.\n"
        "```\n\n"
        "Equivalent in-place idioms: `v.mul_(self.momentum).add_(g)` does the same thing without "
        "the intermediate `new_v`. We use `copy_` here because it's the most general pattern — "
        "Adam needs more than one step of arithmetic before writing the buffer back."
    ),
    "prompt_body": (
        "Implement `cx4_make_sgdm_with_copy()` — return a class `SGDMomentumCopy`.\n\n"
        "- `SGDMomentumCopy(params, lr, momentum=0.9)`:\n"
        "  - `self.params = list(params)`\n"
        "  - `self.lr = lr; self.momentum = momentum`\n"
        "  - `self.velocities = [t.zeros_like(p) for p in self.params]` (atom A).\n"
        "- `SGDMomentumCopy.step(self)`:\n"
        "  - For each `(p, v)` in `zip(self.params, self.velocities)`:\n"
        "    - If `p.grad is None`, skip.\n"
        "    - Compute `new_v = self.momentum * v + p.grad` (FRESH tensor — NOT in-place).\n"
        "    - Write back with `v.copy_(new_v)` (atom B — in-place into v).\n"
        "    - Apply `p.data -= self.lr * v`.\n"
        "  - Wrap in `t.no_grad()` (or decorator).\n\n"
        "The test checks: (a) `self.velocities[i]` is the SAME tensor object after `.step()` "
        "(proves `copy_` was used, not reassignment); (b) the velocity VALUES are the correct "
        "momentum-update result; (c) the params updated correctly; (d) cross-check vs "
        "`torch.optim.SGD(..., momentum=0.9)`."
    ),
    "stub_body": (
        "def cx4_make_sgdm_with_copy():\n"
        "    \"\"\"Return the SGDMomentumCopy class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "SGDMC = cx4_make_sgdm_with_copy()\n"
        "assert isinstance(SGDMC, type)\n"
        "\n"
        "# Case A: in-place copy_ — velocity tensor identity preserved across step().\n"
        "t.manual_seed(0)\n"
        "p = t.nn.Parameter(t.randn(3, 4))\n"
        "p.grad = t.ones_like(p)\n"
        "opt = SGDMC([p], lr=0.1, momentum=0.9)\n"
        "v_obj_before = opt.velocities[0]            # capture the tensor object.\n"
        "v_storage_id = v_obj_before.data_ptr()      # capture storage pointer.\n"
        "opt.step()\n"
        "assert opt.velocities[0] is v_obj_before, (\n"
        "    'velocity tensor was REPLACED — must use buffer.copy_(new), not buffer = new'\n"
        ")\n"
        "assert opt.velocities[0].data_ptr() == v_storage_id, (\n"
        "    'velocity storage changed — copy_ should preserve storage'\n"
        ")\n"
        "\n"
        "# Case B: velocity VALUES are correct after one step.\n"
        "# v0 = 0; v1 = 0.9*0 + grad = grad = ones.\n"
        "assert t.allclose(opt.velocities[0], t.ones_like(p)), (\n"
        "    f'after step 1, velocity should equal grad (=ones); got {opt.velocities[0]}'\n"
        ")\n"
        "\n"
        "# Case C: second step — v2 = 0.9*v1 + grad = 0.9 + 1 = 1.9 (broadcast over the tensor).\n"
        "p.grad = t.ones_like(p)\n"
        "opt.step()\n"
        "assert t.allclose(opt.velocities[0], 1.9 * t.ones_like(p), atol=1e-6), (\n"
        "    'after step 2, velocity should be 1.9 (= 0.9*1 + 1)'\n"
        ")\n"
        "\n"
        "# Case D: cross-check vs torch.optim.SGD with momentum on identical setup.\n"
        "t.manual_seed(1)\n"
        "w_init = t.randn(5, 2)\n"
        "p_mine = t.nn.Parameter(w_init.clone())\n"
        "p_ref = t.nn.Parameter(w_init.clone())\n"
        "opt_mine = SGDMC([p_mine], lr=0.05, momentum=0.9)\n"
        "opt_ref = t.optim.SGD([p_ref], lr=0.05, momentum=0.9)\n"
        "# Run 3 steps with identical gradients.\n"
        "for step_i in range(3):\n"
        "    g = t.randn_like(p_mine) + step_i  # vary per step.\n"
        "    p_mine.grad = g.clone()\n"
        "    p_ref.grad = g.clone()\n"
        "    opt_mine.step()\n"
        "    opt_ref.step()\n"
        "    assert t.allclose(p_mine.data, p_ref.data, atol=1e-6), (\n"
        "        f'step {step_i}: my params diverge from torch.optim.SGD; '\n"
        "        f'max err = {(p_mine.data - p_ref.data).abs().max().item()}'\n"
        "    )"
    ),
    "solution_body": (
        "def cx4_make_sgdm_with_copy():\n"
        "    class SGDMomentumCopy:\n"
        "        def __init__(self, params, lr, momentum=0.9):\n"
        "            self.params = list(params)\n"
        "            self.lr = lr\n"
        "            self.momentum = momentum\n"
        "            # Atom A (optimizer-state-tensor-buffers): one velocity per param.\n"
        "            self.velocities = [t.zeros_like(p) for p in self.params]\n"
        "\n"
        "        @t.no_grad()\n"
        "        def step(self):\n"
        "            for p, v in zip(self.params, self.velocities):\n"
        "                if p.grad is None:\n"
        "                    continue\n"
        "                # Compute the new velocity as a FRESH tensor.\n"
        "                new_v = self.momentum * v + p.grad\n"
        "                # Atom B (buffer-copy_-inplace): write back into the SAME buffer.\n"
        "                v.copy_(new_v)\n"
        "                # Param update uses the just-updated v.\n"
        "                p.data -= self.lr * v\n"
        "\n"
        "    return SGDMomentumCopy"
    ),
    "solution_notes": (
        "Why `copy_` and not `v[...] = new_v`? Both work for the basic case, but `copy_` is the "
        "idiom PyTorch uses internally (see `torch/optim/_functional.py`) because it's "
        "broadcast-aware and handles dtype/device casts safely. The mathematically-equivalent "
        "`v.mul_(self.momentum).add_(p.grad)` is faster (no intermediate `new_v` allocation) and "
        "is what PyTorch's fused SGD uses, but `copy_` reads more clearly when the new value "
        "involves multi-step arithmetic (think Adam's `m_hat = m / (1 - beta1**step)`)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["optimizer-state-tensor-buffers", "buffer-copy_-inplace"],
    "lo": (
        "Compose per-param state-tensor-buffer allocation (zeros_like velocities at __init__) "
        "with the in-place copy_ idiom (buffer.copy_(new_value) preserves storage) to implement a "
        "momentum-SGD step whose velocity tensor identity is invariant across step calls."
    ),
}


# ===========================================================================
# cx5 — State buffer feeds in-place param mutation (momentum SGD complete)
# ===========================================================================
spec_5 = {
    "atom_ids": ["optimizer-state-tensor-buffers", "inplace-param-update"],
    "subtopics": _subs(["optimizer-state-tensor-buffers", "inplace-param-update"]),
    "primary_atom": "optimizer-state-tensor-buffers",
    "part": "part3",
    "exercise_index": 5,
    "exercise_title": "momentum SGD: velocity buffer feeds the in-place param update",
    "slug": "velocity-buffer-feeds-inplace-param-update",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Momentum SGD has TWO in-place updates per step, in this order:\n"
        "1. Update the velocity buffer (state): `v ← momentum * v + grad`.\n"
        "2. Update the parameter (in place): `p.data -= lr * v`.\n\n"
        "The composition is: the JUST-UPDATED velocity (atom A: optimizer-state-tensor-buffers) is "
        "the quantity that feeds the in-place param update (atom B: inplace-param-update). If you "
        "use the OLD velocity in the param step, you've implemented something different — closer "
        "to plain SGD with a stale velocity term.\n\n"
        "PyTorch's `torch.optim.SGD` with `momentum > 0` follows this exact order: velocity first, "
        "then param. The reference update rule (without dampening, nesterov, or weight_decay) is:\n"
        "```\n"
        "v_t = momentum * v_{t-1} + g_t\n"
        "p_t = p_{t-1} - lr * v_t                # uses v_t, not v_{t-1}.\n"
        "```\n\n"
        "**Idiomatic in-place version (what we do here).**\n"
        "```python\n"
        "@t.no_grad()\n"
        "def step(self):\n"
        "    for p, v in zip(self.params, self.velocities):\n"
        "        if p.grad is None: continue\n"
        "        v.mul_(self.momentum).add_(p.grad)   # atom A: update buffer in place.\n"
        "        p.data.add_(v, alpha=-self.lr)       # atom B: param update in place, uses NEW v.\n"
        "```\n"
        "The `add_(v, alpha=-lr)` is the in-place equivalent of `p.data -= lr * v`.\n\n"
        "**Why both atoms together.** This is the canonical 'optimizer that needs state' pattern. "
        "Same shape applies to Adam, RMSProp, Adagrad — the only thing that changes is how the "
        "buffer update is computed."
    ),
    "prompt_body": (
        "Implement `cx5_make_sgdm()` — return `SGDM` with a working momentum step.\n\n"
        "- `SGDM(params, lr, momentum=0.9)`:\n"
        "  - `self.params = list(params)`\n"
        "  - `self.lr = lr; self.momentum = momentum`\n"
        "  - `self.velocities = [t.zeros_like(p) for p in self.params]` (atom A).\n"
        "- `SGDM.step(self)`:\n"
        "  - For each `(p, v)` in `zip(self.params, self.velocities)`:\n"
        "    - If `p.grad is None`, skip.\n"
        "    - Update v IN-PLACE: `v.mul_(self.momentum).add_(p.grad)` (or equivalent `copy_`/`-=`).\n"
        "    - Update p IN-PLACE: `p.data -= self.lr * v` (or `p.data.add_(v, alpha=-self.lr)`) (atom B).\n"
        "  - Wrap in `t.no_grad()`.\n\n"
        "The test cross-checks against `torch.optim.SGD(..., momentum=0.9)` over 4 steps with "
        "varying gradients. It also checks that the velocity buffer was used in the CORRECT order "
        "(updated FIRST, then fed into the param step) by re-deriving the expected values by hand."
    ),
    "stub_body": (
        "def cx5_make_sgdm():\n"
        "    \"\"\"Return the SGDM class with momentum step().\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "SGDM = cx5_make_sgdm()\n"
        "assert isinstance(SGDM, type)\n"
        "\n"
        "# Case A: one step from zero velocity — v should = grad, p should = p_init - lr*grad.\n"
        "t.manual_seed(0)\n"
        "p = t.nn.Parameter(t.zeros(3))\n"
        "p.grad = t.tensor([1.0, 2.0, 3.0])\n"
        "opt = SGDM([p], lr=0.1, momentum=0.9)\n"
        "opt.step()\n"
        "# v = 0.9*0 + grad = [1,2,3]\n"
        "assert t.allclose(opt.velocities[0], t.tensor([1.0, 2.0, 3.0])), (\n"
        "    f'after step 1, v should equal grad; got {opt.velocities[0]}'\n"
        ")\n"
        "# p = 0 - 0.1*v = [-0.1,-0.2,-0.3]\n"
        "assert t.allclose(p.data, t.tensor([-0.1, -0.2, -0.3])), (\n"
        "    f'after step 1, p should be -0.1*grad; got {p.data}'\n"
        ")\n"
        "\n"
        "# Case B: ORDER check — the param step must use the NEW velocity, not the old one.\n"
        "# Second step with same grad: v_new = 0.9*1 + 1 = 1.9; p_new = p_old - 0.1*1.9 = -0.1 - 0.19 = -0.29\n"
        "p.grad = t.tensor([1.0, 2.0, 3.0])\n"
        "opt.step()\n"
        "assert t.allclose(opt.velocities[0], t.tensor([1.9, 3.8, 5.7]), atol=1e-6), (\n"
        "    f'after step 2, v should = 1.9*grad; got {opt.velocities[0]}'\n"
        ")\n"
        "# If implementer used OLD v ([1,2,3]) instead of NEW v ([1.9,3.8,5.7]), p would be -0.2,-0.4,-0.6.\n"
        "assert t.allclose(p.data, t.tensor([-0.29, -0.58, -0.87]), atol=1e-6), (\n"
        "    f'after step 2, p should = p_old - 0.1*NEW_v; got {p.data} '\n"
        "    f'(if you got [-0.2,-0.4,-0.6] you used the OLD velocity — order bug)'\n"
        ")\n"
        "\n"
        "# Case C: cross-check vs torch.optim.SGD(momentum=0.9) over 4 steps with varying grads.\n"
        "t.manual_seed(1)\n"
        "w_init = t.randn(6, 3)\n"
        "p_mine = t.nn.Parameter(w_init.clone())\n"
        "p_ref = t.nn.Parameter(w_init.clone())\n"
        "opt_mine = SGDM([p_mine], lr=0.05, momentum=0.9)\n"
        "opt_ref = t.optim.SGD([p_ref], lr=0.05, momentum=0.9)\n"
        "for step_i in range(4):\n"
        "    g = t.randn_like(p_mine) * (1 + step_i * 0.3)\n"
        "    p_mine.grad = g.clone()\n"
        "    p_ref.grad = g.clone()\n"
        "    opt_mine.step()\n"
        "    opt_ref.step()\n"
        "    assert t.allclose(p_mine.data, p_ref.data, atol=1e-6), (\n"
        "        f'step {step_i}: diverges from torch.optim.SGD; '\n"
        "        f'max err = {(p_mine.data - p_ref.data).abs().max().item()}'\n"
        "    )\n"
        "\n"
        "# Case D: in-place — param tensor identity preserved.\n"
        "p2 = t.nn.Parameter(t.randn(4))\n"
        "p2.grad = t.ones_like(p2)\n"
        "opt2 = SGDM([p2], lr=0.1, momentum=0.5)\n"
        "p2_obj = p2\n"
        "p2_data_ptr = p2.data.data_ptr()\n"
        "opt2.step()\n"
        "assert p2 is p2_obj, 'param object was replaced — must be in-place'\n"
        "# Velocity buffer is also still the same tensor.\n"
        "v2_obj = opt2.velocities[0]\n"
        "opt2.step()\n"
        "assert opt2.velocities[0] is v2_obj, 'velocity tensor was replaced — must be in-place'"
    ),
    "solution_body": (
        "def cx5_make_sgdm():\n"
        "    class SGDM:\n"
        "        def __init__(self, params, lr, momentum=0.9):\n"
        "            self.params = list(params)\n"
        "            self.lr = lr\n"
        "            self.momentum = momentum\n"
        "            # Atom A (optimizer-state-tensor-buffers): velocity buffers, zero-init.\n"
        "            self.velocities = [t.zeros_like(p) for p in self.params]\n"
        "\n"
        "        @t.no_grad()\n"
        "        def step(self):\n"
        "            for p, v in zip(self.params, self.velocities):\n"
        "                if p.grad is None:\n"
        "                    continue\n"
        "                # Update velocity IN PLACE: v = momentum*v + grad.\n"
        "                # mul_ + add_ is the idiomatic two-step in-place form.\n"
        "                v.mul_(self.momentum).add_(p.grad)\n"
        "                # Atom B (inplace-param-update): p.data -= lr * v (using the NEW v).\n"
        "                p.data.add_(v, alpha=-self.lr)\n"
        "\n"
        "    return SGDM"
    ),
    "solution_notes": (
        "The order matters: PyTorch's SGD computes `v_t = mu*v_{t-1} + g_t` FIRST, then "
        "`p_t = p_{t-1} - lr*v_t`. If you swap the order (use the OLD velocity in the param step), "
        "you get a step that's effectively 'plain SGD with a stale momentum kicker' — convergence "
        "looks similar early, then diverges on harder problems. The `add_(v, alpha=-self.lr)` "
        "form fuses the multiplication and subtraction into a single CUDA kernel; the equivalent "
        "`p.data -= self.lr * v` allocates a temporary `lr*v` tensor. Both are mathematically "
        "identical."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["optimizer-state-tensor-buffers", "inplace-param-update"],
    "lo": (
        "Compose per-param state-tensor-buffer state (velocity, zeros_like, in-place mul_+add_ "
        "update) with the in-place param update (p.data.add_(v, alpha=-lr) using the JUST-updated "
        "velocity) to implement a momentum-SGD step that matches torch.optim.SGD(momentum=...)."
    ),
}


# ===========================================================================
# cx6 — One full SGD iteration: step then zero_grad
# ===========================================================================
spec_6 = {
    "atom_ids": ["inplace-param-update", "zero-grad-set-none"],
    "subtopics": _subs(["inplace-param-update", "zero-grad-set-none"]),
    "primary_atom": "inplace-param-update",
    "part": "part3",
    "exercise_index": 6,
    "exercise_title": "one full SGD training iteration: step() then zero_grad(set_to_none=True)",
    "slug": "sgd-step-then-zero-grad",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "These two atoms are the two POST-backward halves of one PyTorch training iteration:\n"
        "```\n"
        "loss.backward()       # accumulates grads into p.grad.\n"
        "optimizer.step()      # atom A: in-place update of params using p.grad.\n"
        "optimizer.zero_grad() # atom B: clear p.grad so the next backward() starts fresh.\n"
        "```\n\n"
        "Both atoms touch `self.params` from the optimizer's materialized list. Both are in-place "
        "but in different senses:\n"
        "- **inplace-param-update** mutates `p.data` (the param's storage).\n"
        "- **zero-grad-set-none** rebinds `p.grad` to `None` (drops the reference; doesn't mutate "
        "the underlying tensor storage).\n\n"
        "ORDER matters: step BEFORE zero_grad. If you zero first, you've thrown away the gradient "
        "you were about to use to update params, and `.step()` silently no-ops because every "
        "`p.grad is None`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class SGD:\n"
        "    def __init__(self, params, lr):\n"
        "        self.params = list(params)\n"
        "        self.lr = lr\n"
        "\n"
        "    @t.no_grad()\n"
        "    def step(self):\n"
        "        for p in self.params:\n"
        "            if p.grad is None: continue\n"
        "            p.data -= self.lr * p.grad         # atom A.\n"
        "\n"
        "    def zero_grad(self, set_to_none=True):\n"
        "        for p in self.params:\n"
        "            if set_to_none:\n"
        "                p.grad = None                  # atom B.\n"
        "            else:\n"
        "                if p.grad is not None:\n"
        "                    p.grad.zero_()\n"
        "```\n\n"
        "**Why both atoms together.** This is the smallest 'usable' optimizer — a SGD class with "
        "`step + zero_grad` is enough to wire into any training loop and get correct (if slow) "
        "convergence. Every other optimizer (Adam, RMSProp) is this pattern with a more elaborate "
        "step rule."
    ),
    "prompt_body": (
        "Implement `cx6_make_sgd_with_step_and_zero()` — return a class `SGDFull`.\n\n"
        "- `SGDFull(params, lr)`:\n"
        "  - `self.params = list(params); self.lr = lr`\n"
        "- `SGDFull.step(self)`:\n"
        "  - For each `p` with non-None `p.grad`: `p.data -= self.lr * p.grad` (atom A).\n"
        "  - Wrap in `t.no_grad()`.\n"
        "- `SGDFull.zero_grad(self, set_to_none=True)`:\n"
        "  - If `set_to_none`: set every `p.grad = None` (atom B).\n"
        "  - Else: `p.grad.zero_()` in place for every param with a non-None grad.\n\n"
        "The test runs a small autograd-based loop: build a tiny linear model, compute a loss, "
        "call `backward()`, then `opt.step(); opt.zero_grad()`, and repeat for 5 iterations. It "
        "checks that:\n"
        "- Params end up at the same values as `torch.optim.SGD` would.\n"
        "- Between iterations, `p.grad` is `None` (proving zero_grad set_to_none worked).\n"
        "- The loss DECREASES over iterations (proving step actually moves params toward minimum).\n"
        "- Forgetting `zero_grad` would have caused gradient accumulation — we verify by NOT "
        "calling it and showing the params diverge from the reference."
    ),
    "stub_body": (
        "def cx6_make_sgd_with_step_and_zero():\n"
        "    \"\"\"Return the SGDFull class with both step() and zero_grad().\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "SGDFull = cx6_make_sgd_with_step_and_zero()\n"
        "assert isinstance(SGDFull, type)\n"
        "\n"
        "# Case A: basic correctness — single step matches torch.optim.SGD, zero_grad clears.\n"
        "t.manual_seed(0)\n"
        "p_mine = t.nn.Parameter(t.tensor([1.0, 2.0, 3.0]))\n"
        "p_ref  = t.nn.Parameter(t.tensor([1.0, 2.0, 3.0]))\n"
        "p_mine.grad = t.tensor([0.5, 0.5, 0.5])\n"
        "p_ref.grad  = t.tensor([0.5, 0.5, 0.5])\n"
        "opt_mine = SGDFull([p_mine], lr=0.1)\n"
        "opt_ref  = t.optim.SGD([p_ref], lr=0.1, momentum=0)\n"
        "opt_mine.step(); opt_mine.zero_grad()\n"
        "opt_ref.step();  opt_ref.zero_grad()\n"
        "assert t.allclose(p_mine.data, p_ref.data, atol=1e-7), (\n"
        "    f'after step+zero_grad, diverges from torch.optim.SGD'\n"
        ")\n"
        "assert p_mine.grad is None, 'zero_grad() default must set grad to None'\n"
        "assert p_ref.grad is None, '(sanity) PyTorch also sets grad to None by default'\n"
        "\n"
        "# Case B: full training loop — autograd-driven, 5 iterations, monotone-decreasing loss.\n"
        "t.manual_seed(1)\n"
        "# Tiny linear-regression-style problem: minimize ||W x - y||^2 over W.\n"
        "x = t.randn(8, 3)\n"
        "y = t.randn(8, 4)\n"
        "W_mine = t.nn.Parameter(t.randn(4, 3))\n"
        "W_ref  = t.nn.Parameter(W_mine.detach().clone())\n"
        "opt_mine = SGDFull([W_mine], lr=0.05)\n"
        "opt_ref  = t.optim.SGD([W_ref], lr=0.05, momentum=0)\n"
        "losses = []\n"
        "for it in range(5):\n"
        "    # Mine.\n"
        "    pred_mine = x @ W_mine.T\n"
        "    loss_mine = ((pred_mine - y) ** 2).sum()\n"
        "    loss_mine.backward()\n"
        "    losses.append(loss_mine.item())\n"
        "    opt_mine.step()\n"
        "    opt_mine.zero_grad()\n"
        "    # Between iterations, grad must be None.\n"
        "    assert W_mine.grad is None, f'iter {it}: grad should be None after zero_grad()'\n"
        "    # Reference.\n"
        "    pred_ref = x @ W_ref.T\n"
        "    loss_ref = ((pred_ref - y) ** 2).sum()\n"
        "    loss_ref.backward()\n"
        "    opt_ref.step()\n"
        "    opt_ref.zero_grad()\n"
        "    assert t.allclose(W_mine.data, W_ref.data, atol=1e-6), (\n"
        "        f'iter {it}: my W diverges from torch.optim.SGD; '\n"
        "        f'max err = {(W_mine.data - W_ref.data).abs().max().item()}'\n"
        "    )\n"
        "# Loss should be monotonically decreasing (lr small enough for this tiny problem).\n"
        "for i in range(1, len(losses)):\n"
        "    assert losses[i] < losses[i-1], f'loss should decrease, but losses = {losses}'\n"
        "\n"
        "# Case C: omitting zero_grad demonstrates accumulation — params would diverge.\n"
        "t.manual_seed(2)\n"
        "W_noclear = t.nn.Parameter(t.randn(2, 3))\n"
        "W_clear   = t.nn.Parameter(W_noclear.detach().clone())\n"
        "opt_noclear = SGDFull([W_noclear], lr=0.01)\n"
        "opt_clear   = SGDFull([W_clear], lr=0.01)\n"
        "x_small = t.randn(4, 3)\n"
        "y_small = t.randn(4, 2)\n"
        "for it in range(3):\n"
        "    loss_n = ((x_small @ W_noclear.T - y_small) ** 2).sum(); loss_n.backward()\n"
        "    opt_noclear.step()   # NO zero_grad — grads accumulate.\n"
        "    loss_c = ((x_small @ W_clear.T - y_small) ** 2).sum(); loss_c.backward()\n"
        "    opt_clear.step(); opt_clear.zero_grad()\n"
        "# After 3 iterations the no-zero version has accumulated grads → very different params.\n"
        "assert not t.allclose(W_noclear.data, W_clear.data, atol=1e-3), (\n"
        "    'sanity check: omitting zero_grad should cause divergence (gradient accumulation), '\n"
        "    'but the two trajectories matched — your zero_grad is suspicious'\n"
        ")\n"
        "\n"
        "# Case D: set_to_none=False clears to zero tensor (not None).\n"
        "p4 = t.nn.Parameter(t.randn(3))\n"
        "p4.grad = t.ones_like(p4)\n"
        "opt4 = SGDFull([p4], lr=0.1)\n"
        "opt4.zero_grad(set_to_none=False)\n"
        "assert p4.grad is not None and t.all(p4.grad == 0).item(), (\n"
        "    'zero_grad(set_to_none=False) must leave grad as a zero tensor, not None'\n"
        ")"
    ),
    "solution_body": (
        "def cx6_make_sgd_with_step_and_zero():\n"
        "    class SGDFull:\n"
        "        def __init__(self, params, lr):\n"
        "            self.params = list(params)\n"
        "            self.lr = lr\n"
        "\n"
        "        @t.no_grad()\n"
        "        def step(self):\n"
        "            for p in self.params:\n"
        "                if p.grad is None:\n"
        "                    continue\n"
        "                # Atom A (inplace-param-update): p.data -= lr * p.grad.\n"
        "                p.data -= self.lr * p.grad\n"
        "\n"
        "        def zero_grad(self, set_to_none=True):\n"
        "            for p in self.params:\n"
        "                if set_to_none:\n"
        "                    # Atom B (zero-grad-set-none): drop the reference.\n"
        "                    p.grad = None\n"
        "                else:\n"
        "                    if p.grad is not None:\n"
        "                        p.grad.zero_()\n"
        "\n"
        "    return SGDFull"
    ),
    "solution_notes": (
        "The two atoms are intentionally symmetric in shape but different in semantics: `step` "
        "mutates `p.data` (tensor storage), `zero_grad(set_to_none=True)` rebinds `p.grad` "
        "(reference). The training-loop ORDER `step → zero_grad` is what every PyTorch tutorial "
        "shows; swap them and step silently no-ops. The `@t.no_grad()` on `step` is the autograd-"
        "side guarantee that the optimizer's own arithmetic doesn't build a graph through "
        "`p.data` (which would be silently ignored anyway, but builds garbage for the gc to clean)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["inplace-param-update", "zero-grad-set-none"],
    "lo": (
        "Compose the in-place parameter update (p.data -= lr*p.grad inside step()) with the "
        "set-to-none gradient clear (p.grad = None inside zero_grad()) to implement the full "
        "post-backward half of a PyTorch SGD training iteration, in the canonical order."
    ),
}


SPECS = [spec_1, spec_2, spec_3, spec_4, spec_5, spec_6]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
