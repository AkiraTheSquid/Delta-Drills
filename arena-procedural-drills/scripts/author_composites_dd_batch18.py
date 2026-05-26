"""Composite drills cx19..cx24 — batch-18 (DD-cell, part2).

Six composite procedural drills exercising 2-atom pairs from the ARENA CNN
nn.Module construction machinery (ARENA part 2 — CNNs / nn.Module). The shared
anchor atom is `nn-module-subclass`; each cx pairs it with a neighbour atom
that lives in the same ARENA CNN nn.Module dataflow.

cx19  nn-module-subclass + module-composition
cx20  nn-module-subclass + nn-parameter-wrap
cx21  nn-module-subclass + module-extra-repr
cx22  nn-module-subclass + linear-affine-on-custom-tensor
cx23  nn-module-subclass + relu-elementwise-max
cx24  nn-module-subclass + batchnorm-affine-params
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


_EXTRA = ["import torch.nn as nn", "import torch.nn.functional as F"]


# ===========================================================================
# cx19 — subclass nn.Module with submodules registered in __init__
# ===========================================================================
spec_19 = {
    "atom_ids": ["nn-module-subclass", "module-composition"],
    "subtopics": _subs(["nn-module-subclass", "module-composition"]),
    "primary_atom": "nn-module-subclass",
    "part": "part2",
    "exercise_index": 19,
    "exercise_title": "subclass nn.Module and compose two registered submodules",
    "slug": "subclass-with-submodules-in-init",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Building a CNN-shaped module is two atoms in lockstep:\n"
        "- **nn-module-subclass** — write `class Foo(nn.Module):` with `__init__` calling "
        "`super().__init__()` and a `forward(self, x)` method. The base-class init is what wires up "
        "the bookkeeping dicts (`_parameters`, `_modules`, `_buffers`) PyTorch uses to walk the tree.\n"
        "- **module-composition** — assign child modules as ATTRIBUTES inside `__init__`. The custom "
        "`nn.Module.__setattr__` watches for any value that is itself an `nn.Module` and quietly "
        "registers it in `self._modules` under the attribute name. That registration is what makes "
        "`.parameters()`, `.to(device)`, `.train()`, and `.state_dict()` recurse into the children.\n\n"
        "**Anatomy.**\n"
        "1. `super().__init__()` — MUST run before any `self.<child> = ...` assignment, otherwise "
        "`_modules` does not exist yet and the auto-registration silently breaks.\n"
        "2. `self.fc1 = nn.Linear(...)` / `self.fc2 = nn.Linear(...)` — each assignment registers the "
        "child. The attribute name is the registration key.\n"
        "3. `forward(self, x)` — call the children in order. `self.fc1(x)` invokes their `__call__`, "
        "which wraps `forward` with hooks and gradient bookkeeping."
    ),
    "prompt_body": (
        "Define a class `TwoLayer(nn.Module)` and a builder `cx19_build_two_layer(in_dim, hid_dim, "
        "out_dim)` that returns an instance.\n\n"
        "`TwoLayer.__init__` must:\n"
        "1. Call `super().__init__()` first.\n"
        "2. Assign `self.fc1 = nn.Linear(in_dim, hid_dim)`.\n"
        "3. Assign `self.fc2 = nn.Linear(hid_dim, out_dim)`.\n\n"
        "`TwoLayer.forward(self, x)` must return `self.fc2(self.fc1(x))` (no activation — this drill "
        "is about REGISTRATION, not nonlinearity).\n\n"
        "Because both `fc1` and `fc2` are assigned as attributes inside `__init__`, the base "
        "`nn.Module.__setattr__` should auto-register them, and `list(model.parameters())` should "
        "yield 4 tensors (2 weights + 2 biases) in registration order."
    ),
    "stub_body": (
        "class TwoLayer(nn.Module):\n"
        "    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError\n"
        "\n"
        "\n"
        "def cx19_build_two_layer(in_dim: int, hid_dim: int, out_dim: int) -> 'TwoLayer':\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: instance is an nn.Module and has the right submodules registered.\n"
        "model = cx19_build_two_layer(4, 8, 3)\n"
        "assert isinstance(model, nn.Module), 'TwoLayer must subclass nn.Module'\n"
        "assert isinstance(model, TwoLayer), 'builder must return a TwoLayer instance'\n"
        "assert hasattr(model, 'fc1') and isinstance(model.fc1, nn.Linear), 'fc1 must be nn.Linear'\n"
        "assert hasattr(model, 'fc2') and isinstance(model.fc2, nn.Linear), 'fc2 must be nn.Linear'\n"
        "assert model.fc1.in_features == 4 and model.fc1.out_features == 8\n"
        "assert model.fc2.in_features == 8 and model.fc2.out_features == 3\n"
        "\n"
        "# Case B: AUTO-registration via __setattr__ — both children appear in _modules.\n"
        "child_names = list(model._modules.keys())\n"
        "assert child_names == ['fc1', 'fc2'], f'child registration order broken: {child_names}'\n"
        "\n"
        "# Case C: .parameters() recurses into BOTH children — 4 tensors total.\n"
        "ps = list(model.parameters())\n"
        "assert len(ps) == 4, f'expected 4 params (2 W + 2 b), got {len(ps)}'\n"
        "named = dict(model.named_parameters())\n"
        "assert set(named.keys()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}, named.keys()\n"
        "\n"
        "# Case D: forward composes fc2(fc1(x)) — shape and value match the manual chain.\n"
        "t.manual_seed(0)\n"
        "model = cx19_build_two_layer(5, 7, 2)\n"
        "x = t.randn(3, 5)\n"
        "y = model(x)\n"
        "assert tuple(y.shape) == (3, 2), f'expected (3, 2), got {tuple(y.shape)}'\n"
        "y_ref = model.fc2(model.fc1(x))\n"
        "assert t.allclose(y, y_ref, atol=1e-6), 'forward must be fc2(fc1(x)), no activation'\n"
        "\n"
        "# Case E: .to(device) and .train() reach the children — proves registration worked.\n"
        "model.eval()\n"
        "assert not model.fc1.training and not model.fc2.training, 'eval() must propagate to children'\n"
        "model.train()\n"
        "assert model.fc1.training and model.fc2.training, 'train() must propagate to children'"
    ),
    "solution_body": (
        "class TwoLayer(nn.Module):\n"
        "    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):\n"
        "        # Atom A (nn-module-subclass): super().__init__() FIRST so _modules exists\n"
        "        # before we start assigning child modules.\n"
        "        super().__init__()\n"
        "        # Atom B (module-composition): assigning nn.Module instances to attrs triggers\n"
        "        # nn.Module.__setattr__, which registers them in self._modules under the attr name.\n"
        "        self.fc1 = nn.Linear(in_dim, hid_dim)\n"
        "        self.fc2 = nn.Linear(hid_dim, out_dim)\n"
        "\n"
        "    def forward(self, x):\n"
        "        # Children are called as __call__ — that runs hooks + forward.\n"
        "        return self.fc2(self.fc1(x))\n"
        "\n"
        "\n"
        "def cx19_build_two_layer(in_dim: int, hid_dim: int, out_dim: int) -> 'TwoLayer':\n"
        "    return TwoLayer(in_dim, hid_dim, out_dim)"
    ),
    "solution_notes": (
        "**Order of operations in `__init__` is load-bearing.** If you assign `self.fc1` BEFORE "
        "`super().__init__()`, the auto-registration silently no-ops (no `_modules` dict yet) and "
        "`.parameters()` returns an empty iterator — a classic ARENA bug because forward still works."
    ),
    "extra_imports": _EXTRA,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["nn-module-subclass", "module-composition"],
    "lo": (
        "Compose nn.Module subclassing (super().__init__ + forward) with module composition "
        "(assigning child nn.Modules as attributes for auto-registration) to build a two-layer net "
        "whose .parameters() correctly recurses into both children."
    ),
}


# ===========================================================================
# cx20 — subclass + wrap raw tensor as nn.Parameter
# ===========================================================================
spec_20 = {
    "atom_ids": ["nn-module-subclass", "nn-parameter-wrap"],
    "subtopics": _subs(["nn-module-subclass", "nn-parameter-wrap"]),
    "primary_atom": "nn-module-subclass",
    "part": "part2",
    "exercise_index": 20,
    "exercise_title": "subclass nn.Module and wrap a raw tensor as nn.Parameter",
    "slug": "subclass-wrap-weight-as-parameter",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "When ARENA asks you to roll your own `Linear` / `Embedding` / `BatchNorm`, you don't get to "
        "lean on `nn.Linear` — you have to manage the **raw weight tensor** yourself. The combination:\n"
        "- **nn-module-subclass** — the usual `class Foo(nn.Module)` + `super().__init__()` + "
        "`forward` scaffold.\n"
        "- **nn-parameter-wrap** — wrap the raw tensor with `nn.Parameter(tensor)`. The bare wrapper "
        "marks the tensor as a *learnable* parameter; assigning it to an attribute (`self.weight = "
        "nn.Parameter(...)`) registers it in `self._parameters`, which is what makes it show up in "
        "`.parameters()` and get moved by `.to(device)`.\n\n"
        "**Why not just `self.weight = t.randn(...)`?** A raw tensor assigned to a module attribute "
        "is invisible to `.parameters()` (it lives in `__dict__`, not `_parameters`). The optimizer "
        "would silently never update it.\n\n"
        "**Anatomy.**\n"
        "1. `super().__init__()` first.\n"
        "2. `w = t.empty(out_dim, in_dim); nn.init.kaiming_uniform_(w, a=5**0.5)` — Kaiming-like init "
        "matching `nn.Linear`'s default scheme.\n"
        "3. `self.weight = nn.Parameter(w)` — the wrap + assign that registers the tensor.\n"
        "4. `forward(self, x)`: `return x @ self.weight.T` (no bias in this drill)."
    ),
    "prompt_body": (
        "Define a class `MyLinear(nn.Module)` and a builder `cx20_build_my_linear(in_dim, out_dim)` "
        "that returns an instance.\n\n"
        "`MyLinear.__init__(self, in_dim, out_dim)` must:\n"
        "1. Call `super().__init__()`.\n"
        "2. Create a `(out_dim, in_dim)` weight tensor (use `t.empty` + `nn.init.kaiming_uniform_(w, "
        "a=5**0.5)` so init matches `nn.Linear`).\n"
        "3. **Wrap** it as `nn.Parameter(w)` and assign to `self.weight`. Do NOT assign the raw "
        "tensor — the test verifies the type.\n\n"
        "`MyLinear.forward(self, x)` returns `x @ self.weight.T` (no bias here — cx22 does the full "
        "affine).\n\n"
        "The test checks: type is `nn.Parameter`, `requires_grad=True`, registered in "
        "`_parameters` (NOT in `__dict__`), and that `.parameters()` yields exactly one tensor."
    ),
    "stub_body": (
        "class MyLinear(nn.Module):\n"
        "    def __init__(self, in_dim: int, out_dim: int):\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError\n"
        "\n"
        "\n"
        "def cx20_build_my_linear(in_dim: int, out_dim: int) -> 'MyLinear':\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: weight is an nn.Parameter, not a raw tensor.\n"
        "m = cx20_build_my_linear(in_dim=4, out_dim=3)\n"
        "assert isinstance(m, nn.Module)\n"
        "assert hasattr(m, 'weight'), 'must expose self.weight'\n"
        "assert isinstance(m.weight, nn.Parameter), (\n"
        "    f'self.weight must be wrapped in nn.Parameter, got {type(m.weight).__name__}'\n"
        ")\n"
        "assert m.weight.requires_grad is True, 'nn.Parameter wraps with requires_grad=True by default'\n"
        "assert tuple(m.weight.shape) == (3, 4), f'expected (out_dim, in_dim) = (3, 4), got {tuple(m.weight.shape)}'\n"
        "\n"
        "# Case B: nn.Parameter assignment registers in _parameters, NOT in __dict__.\n"
        "# This is the load-bearing distinction vs a raw tensor assignment.\n"
        "assert 'weight' in m._parameters, '_parameters dict must contain weight'\n"
        "assert m._parameters['weight'] is m.weight\n"
        "assert 'weight' not in m.__dict__, 'nn.Parameter assignment should not land in __dict__'\n"
        "\n"
        "# Case C: .parameters() yields exactly the single weight tensor.\n"
        "ps = list(m.parameters())\n"
        "assert len(ps) == 1, f'expected 1 param, got {len(ps)}'\n"
        "assert ps[0] is m.weight\n"
        "\n"
        "# Case D: forward is x @ weight.T — matches a manual reference.\n"
        "t.manual_seed(0)\n"
        "m = cx20_build_my_linear(in_dim=5, out_dim=2)\n"
        "x = t.randn(7, 5)\n"
        "y = m(x)\n"
        "assert tuple(y.shape) == (7, 2), f'expected (7, 2), got {tuple(y.shape)}'\n"
        "assert t.allclose(y, x @ m.weight.T, atol=1e-6), 'forward must be x @ self.weight.T'\n"
        "\n"
        "# Case E: gradient flows into m.weight (proof it is a real learnable parameter).\n"
        "loss = m(x).sum()\n"
        "loss.backward()\n"
        "assert m.weight.grad is not None, 'gradient did not flow — weight not registered as parameter'\n"
        "assert m.weight.grad.shape == m.weight.shape\n"
        "\n"
        "# Case F: counter-example — a raw tensor (not wrapped) is invisible to .parameters().\n"
        "class _RawTensorMod(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.weight = t.zeros(3, 4)  # NOT wrapped\n"
        "raw = _RawTensorMod()\n"
        "assert list(raw.parameters()) == [], 'raw tensor must NOT appear in .parameters() — proves wrap is required'"
    ),
    "solution_body": (
        "class MyLinear(nn.Module):\n"
        "    def __init__(self, in_dim: int, out_dim: int):\n"
        "        super().__init__()\n"
        "        # Allocate the raw weight tensor and Kaiming-init it (same scheme nn.Linear uses).\n"
        "        w = t.empty(out_dim, in_dim)\n"
        "        nn.init.kaiming_uniform_(w, a=5 ** 0.5)\n"
        "        # Atom (nn-parameter-wrap): wrap the raw tensor so it gets registered in\n"
        "        # self._parameters by nn.Module.__setattr__. requires_grad=True by default.\n"
        "        self.weight = nn.Parameter(w)\n"
        "\n"
        "    def forward(self, x):\n"
        "        # (B, in_dim) @ (in_dim, out_dim) = (B, out_dim)\n"
        "        return x @ self.weight.T\n"
        "\n"
        "\n"
        "def cx20_build_my_linear(in_dim: int, out_dim: int) -> 'MyLinear':\n"
        "    return MyLinear(in_dim, out_dim)"
    ),
    "solution_notes": (
        "`nn.Parameter` is a subclass of `Tensor` that `nn.Module.__setattr__` specifically watches "
        "for — it slots into `_parameters` not `_modules`. The wrap is the contract. Without it, the "
        "tensor is a stray attribute and the optimizer never sees it."
    ),
    "extra_imports": _EXTRA,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["nn-module-subclass", "nn-parameter-wrap"],
    "lo": (
        "Compose nn.Module subclassing with nn.Parameter wrapping (raw tensor → learnable, "
        "auto-registered into _parameters) to build a weight-only Linear whose parameter is "
        "visible to .parameters() and receives gradients."
    ),
}


# ===========================================================================
# cx21 — subclass + extra_repr for a useful __repr__
# ===========================================================================
spec_21 = {
    "atom_ids": ["nn-module-subclass", "module-extra-repr"],
    "subtopics": _subs(["nn-module-subclass", "module-extra-repr"]),
    "primary_atom": "nn-module-subclass",
    "part": "part2",
    "exercise_index": 21,
    "exercise_title": "subclass nn.Module and customize __repr__ via extra_repr",
    "slug": "subclass-with-custom-extra-repr",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Default `nn.Module.__repr__()` prints `ClassName()` plus the children — which is useless if "
        "your class has non-module hyperparameters (kernel size, num features, stride). The "
        "**extra_repr** hook is the canonical override:\n"
        "- **nn-module-subclass** — the usual `class Foo(nn.Module)` skeleton.\n"
        "- **module-extra-repr** — override `def extra_repr(self) -> str` to return a comma-separated "
        "string of the hyperparameters. The base `__repr__` then renders `ClassName(<extra_repr>)`, "
        "followed (one indent in) by each registered child's own repr.\n\n"
        "Crucially you do NOT override `__repr__` directly — let the base class do the indentation "
        "and recursion. You only own the string between the parens.\n\n"
        "**Anatomy.**\n"
        "1. `super().__init__()` + store hyperparams on `self`.\n"
        "2. `def extra_repr(self) -> str: return f'in_features={self.in_features}, "
        "out_features={self.out_features}'`.\n"
        "3. `repr(m)` produces `'MyLayer(in_features=4, out_features=3)'` — same shape as the "
        "built-in `nn.Linear` repr."
    ),
    "prompt_body": (
        "Define a class `MyLayer(nn.Module)` and a builder `cx21_build_my_layer(in_features, "
        "out_features)` that returns an instance.\n\n"
        "`MyLayer.__init__` must:\n"
        "1. Call `super().__init__()`.\n"
        "2. Store `self.in_features = in_features` and `self.out_features = out_features`.\n\n"
        "`MyLayer.extra_repr(self)` must return EXACTLY the string "
        "`f'in_features={self.in_features}, out_features={self.out_features}'` — no leading/trailing "
        "spaces, no parentheses, no class name.\n\n"
        "Do NOT define `__repr__` yourself — let the base class wrap your `extra_repr()` into "
        "`'MyLayer(<extra_repr>)'`.\n\n"
        "`MyLayer.forward(self, x)` returns `x` unchanged (identity — this drill is about the repr, "
        "not the math)."
    ),
    "stub_body": (
        "class MyLayer(nn.Module):\n"
        "    def __init__(self, in_features: int, out_features: int):\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def extra_repr(self) -> str:\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError\n"
        "\n"
        "\n"
        "def cx21_build_my_layer(in_features: int, out_features: int) -> 'MyLayer':\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: instance has the stored hyperparams.\n"
        "m = cx21_build_my_layer(4, 3)\n"
        "assert isinstance(m, nn.Module)\n"
        "assert m.in_features == 4 and m.out_features == 3\n"
        "\n"
        "# Case B: extra_repr returns the right naked string.\n"
        "s = m.extra_repr()\n"
        "assert isinstance(s, str)\n"
        "assert s == 'in_features=4, out_features=3', f'extra_repr mismatch: {s!r}'\n"
        "\n"
        "# Case C: __repr__ wraps extra_repr — note we did NOT override __repr__ ourselves.\n"
        "r = repr(m)\n"
        "assert r == 'MyLayer(in_features=4, out_features=3)', f'repr mismatch: {r!r}'\n"
        "\n"
        "# Case D: forward is identity (this drill exercises the repr, not the math).\n"
        "x = t.randn(2, 4)\n"
        "y = m(x)\n"
        "assert t.equal(y, x), 'forward should be identity in this drill'\n"
        "\n"
        "# Case E: different sizes also render correctly — proves extra_repr reads from self,\n"
        "# not from a hard-coded string.\n"
        "m2 = cx21_build_my_layer(128, 256)\n"
        "assert m2.extra_repr() == 'in_features=128, out_features=256'\n"
        "assert repr(m2) == 'MyLayer(in_features=128, out_features=256)'\n"
        "\n"
        "# Case F: nesting test — when wrapped in nn.Sequential, the child repr shows our extra_repr,\n"
        "# proving we let the base class own __repr__ indentation/recursion.\n"
        "seq = nn.Sequential(cx21_build_my_layer(5, 6))\n"
        "rs = repr(seq)\n"
        "assert 'MyLayer(in_features=5, out_features=6)' in rs, f'nested repr broken: {rs!r}'"
    ),
    "solution_body": (
        "class MyLayer(nn.Module):\n"
        "    def __init__(self, in_features: int, out_features: int):\n"
        "        # Atom A (nn-module-subclass): wire up the base bookkeeping FIRST.\n"
        "        super().__init__()\n"
        "        # Plain Python attrs — these are the hyperparameters extra_repr will print.\n"
        "        self.in_features = in_features\n"
        "        self.out_features = out_features\n"
        "\n"
        "    def extra_repr(self) -> str:\n"
        "        # Atom B (module-extra-repr): own ONLY the string between the parens.\n"
        "        # nn.Module.__repr__ wraps this into f'{ClassName}({extra_repr()})' and recurses\n"
        "        # into children for us.\n"
        "        return f'in_features={self.in_features}, out_features={self.out_features}'\n"
        "\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "\n"
        "\n"
        "def cx21_build_my_layer(in_features: int, out_features: int) -> 'MyLayer':\n"
        "    return MyLayer(in_features, out_features)"
    ),
    "solution_notes": (
        "Overriding `__repr__` directly is the wrong move — you lose the recursion into children and "
        "the indentation. `extra_repr` is the surgical override slot: it composes with the base "
        "class's tree-printing machinery instead of replacing it."
    ),
    "extra_imports": _EXTRA,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["nn-module-subclass", "module-extra-repr"],
    "lo": (
        "Compose nn.Module subclassing with the extra_repr hook (return naked hyperparam string; "
        "base class wraps it into ClassName(...)) to produce a clean nn.Linear-style repr without "
        "breaking the child-recursion in the base __repr__."
    ),
}


# ===========================================================================
# cx22 — subclass + full affine: y = x @ W.T + b
# ===========================================================================
spec_22 = {
    "atom_ids": ["nn-module-subclass", "linear-affine-on-custom-tensor"],
    "subtopics": _subs(["nn-module-subclass", "linear-affine-on-custom-tensor"]),
    "primary_atom": "nn-module-subclass",
    "part": "part2",
    "exercise_index": 22,
    "exercise_title": "custom Linear: y = x @ W.T + b inside a subclass",
    "slug": "custom-linear-affine-module",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's `Linear` re-implementation is the canonical first nn.Module subclass exercise. Two "
        "atoms running together:\n"
        "- **nn-module-subclass** — `class Linear(nn.Module)` + `super().__init__()` + `forward`.\n"
        "- **linear-affine-on-custom-tensor** — the math: `y = x @ W.T + b`, where `W` has shape "
        "`(out_features, in_features)` and `b` has shape `(out_features,)`. Both are wrapped as "
        "`nn.Parameter` so they show up in `.parameters()`.\n\n"
        "**Why `W.T`?** PyTorch stores `W` as `(out_features, in_features)` (rows = output neurons). "
        "The forward computes `x @ W.T` so the `(B, in_features) @ (in_features, out_features)` "
        "matmul produces `(B, out_features)`. This is the SAME shape convention `nn.Linear` uses.\n\n"
        "**Bias broadcast.** `b` has shape `(out_features,)`. Adding to `(B, out_features)` "
        "broadcasts across batch — one bias vector per output neuron, shared across the batch.\n\n"
        "**Anatomy.**\n"
        "1. `super().__init__()` + store `in_features`, `out_features` on `self`.\n"
        "2. Kaiming-init `W` of shape `(out_features, in_features)`; uniform-init `b` of shape "
        "`(out_features,)` in `[-1/sqrt(in_features), 1/sqrt(in_features)]` (matches `nn.Linear`).\n"
        "3. Wrap both as `nn.Parameter`.\n"
        "4. `forward(x): return x @ self.weight.T + self.bias`."
    ),
    "prompt_body": (
        "Define a class `MyAffine(nn.Module)` and a builder `cx22_build_affine(in_features, "
        "out_features)`.\n\n"
        "`MyAffine.__init__` must:\n"
        "1. `super().__init__()`.\n"
        "2. Store `self.in_features`, `self.out_features`.\n"
        "3. Create `weight` of shape `(out_features, in_features)`, Kaiming-init "
        "(`nn.init.kaiming_uniform_(w, a=5**0.5)`), wrap as `nn.Parameter`, assign to `self.weight`.\n"
        "4. Create `bias` of shape `(out_features,)`, uniform-init in `[-1/sqrt(in_features), "
        "1/sqrt(in_features)]` (use `nn.init.uniform_(b, -bound, bound)` with "
        "`bound = 1 / in_features**0.5`), wrap as `nn.Parameter`, assign to `self.bias`.\n\n"
        "`MyAffine.forward(self, x)` returns `x @ self.weight.T + self.bias`.\n\n"
        "Shape contract: `x` is `(..., in_features)`. Output is `(..., out_features)` — the affine "
        "applies along the LAST axis, broadcasting across any leading batch dims."
    ),
    "stub_body": (
        "class MyAffine(nn.Module):\n"
        "    def __init__(self, in_features: int, out_features: int):\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError\n"
        "\n"
        "\n"
        "def cx22_build_affine(in_features: int, out_features: int) -> 'MyAffine':\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: both params registered and right-shaped.\n"
        "m = cx22_build_affine(in_features=4, out_features=3)\n"
        "assert isinstance(m, nn.Module)\n"
        "assert isinstance(m.weight, nn.Parameter)\n"
        "assert isinstance(m.bias, nn.Parameter)\n"
        "assert tuple(m.weight.shape) == (3, 4), f'weight shape: (out, in) = (3, 4); got {tuple(m.weight.shape)}'\n"
        "assert tuple(m.bias.shape) == (3,), f'bias shape: (out,) = (3,); got {tuple(m.bias.shape)}'\n"
        "assert m.weight.requires_grad and m.bias.requires_grad\n"
        "\n"
        "# Case B: .parameters() yields exactly weight + bias.\n"
        "ps = list(m.parameters())\n"
        "assert len(ps) == 2, f'expected 2 params, got {len(ps)}'\n"
        "named = dict(m.named_parameters())\n"
        "assert set(named.keys()) == {'weight', 'bias'}\n"
        "\n"
        "# Case C: forward shape — (B, in) -> (B, out).\n"
        "t.manual_seed(0)\n"
        "m = cx22_build_affine(in_features=5, out_features=2)\n"
        "x = t.randn(7, 5)\n"
        "y = m(x)\n"
        "assert tuple(y.shape) == (7, 2)\n"
        "expected = x @ m.weight.T + m.bias\n"
        "assert t.allclose(y, expected, atol=1e-6), 'forward must be x @ W.T + b'\n"
        "\n"
        "# Case D: bias broadcasts correctly — adding the SAME bias vector to every row.\n"
        "# Set weight to zero so y = bias broadcast across the batch.\n"
        "m_zero = cx22_build_affine(in_features=3, out_features=4)\n"
        "with t.no_grad():\n"
        "    m_zero.weight.zero_()\n"
        "    m_zero.bias.copy_(t.tensor([1.0, 2.0, 3.0, 4.0]))\n"
        "x = t.randn(5, 3)\n"
        "y = m_zero(x)\n"
        "assert tuple(y.shape) == (5, 4)\n"
        "for i in range(5):\n"
        "    assert t.allclose(y[i], t.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-6), (\n"
        "        f'row {i} should equal bias; got {y[i]}'\n"
        "    )\n"
        "\n"
        "# Case E: leading-batch broadcasting — (B1, B2, in) -> (B1, B2, out).\n"
        "m = cx22_build_affine(in_features=4, out_features=6)\n"
        "x = t.randn(2, 3, 4)\n"
        "y = m(x)\n"
        "assert tuple(y.shape) == (2, 3, 6), f'leading-batch shape: got {tuple(y.shape)}'\n"
        "expected = x @ m.weight.T + m.bias\n"
        "assert t.allclose(y, expected, atol=1e-6)\n"
        "\n"
        "# Case F: numeric vs nn.Linear with identical weights — proves the math matches.\n"
        "m = cx22_build_affine(in_features=4, out_features=3)\n"
        "ref = nn.Linear(4, 3)\n"
        "with t.no_grad():\n"
        "    ref.weight.copy_(m.weight)\n"
        "    ref.bias.copy_(m.bias)\n"
        "x = t.randn(8, 4)\n"
        "assert t.allclose(m(x), ref(x), atol=1e-6), 'MyAffine must match nn.Linear for equal params'"
    ),
    "solution_body": (
        "class MyAffine(nn.Module):\n"
        "    def __init__(self, in_features: int, out_features: int):\n"
        "        super().__init__()\n"
        "        self.in_features = in_features\n"
        "        self.out_features = out_features\n"
        "        # Weight: (out_features, in_features). Kaiming-init matches nn.Linear default.\n"
        "        w = t.empty(out_features, in_features)\n"
        "        nn.init.kaiming_uniform_(w, a=5 ** 0.5)\n"
        "        self.weight = nn.Parameter(w)\n"
        "        # Bias: uniform in [-1/sqrt(in), 1/sqrt(in)] — also matches nn.Linear default.\n"
        "        bound = 1.0 / (in_features ** 0.5)\n"
        "        b = t.empty(out_features)\n"
        "        nn.init.uniform_(b, -bound, bound)\n"
        "        self.bias = nn.Parameter(b)\n"
        "\n"
        "    def forward(self, x):\n"
        "        # Atom (linear-affine-on-custom-tensor): y = x @ W.T + b.\n"
        "        # x: (..., in_features). W.T: (in_features, out_features). y: (..., out_features).\n"
        "        # bias broadcasts on the last axis.\n"
        "        return x @ self.weight.T + self.bias\n"
        "\n"
        "\n"
        "def cx22_build_affine(in_features: int, out_features: int) -> 'MyAffine':\n"
        "    return MyAffine(in_features, out_features)"
    ),
    "solution_notes": (
        "The `.T` is the only \"trick\". Storing `W` as `(out, in)` is PyTorch convention because "
        "it lines up with how rows of a weight matrix correspond to output neurons. The matmul "
        "`x @ W.T` is what makes the shapes work; if you forget `.T`, you'll see a shape mismatch "
        "error pointing at the matmul."
    ),
    "extra_imports": _EXTRA,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["nn-module-subclass", "linear-affine-on-custom-tensor"],
    "lo": (
        "Compose nn.Module subclassing with the linear-affine math (y = x @ W.T + b, weight stored "
        "as (out, in) with Kaiming init, bias as (out,) with uniform init, both wrapped in "
        "nn.Parameter) to re-implement nn.Linear from scratch."
    ),
}


# ===========================================================================
# cx23 — subclass + ReLU as elementwise max(x, 0)
# ===========================================================================
spec_23 = {
    "atom_ids": ["nn-module-subclass", "relu-elementwise-max"],
    "subtopics": _subs(["nn-module-subclass", "relu-elementwise-max"]),
    "primary_atom": "nn-module-subclass",
    "part": "part2",
    "exercise_index": 23,
    "exercise_title": "custom ReLU as a parameterless nn.Module: max(x, 0)",
    "slug": "custom-relu-elementwise-max-module",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Even a parameterless layer is worth wrapping as an `nn.Module` — it slots cleanly into "
        "`nn.Sequential`, shows up in the repr tree, and the `.train()/.eval()` toggle reaches it.\n"
        "- **nn-module-subclass** — the usual scaffold, but with NOTHING registered in "
        "`__init__` beyond `super().__init__()` (no params, no buffers, no children).\n"
        "- **relu-elementwise-max** — the math: `relu(x) = max(x, 0)`. Implement as `t.maximum(x, "
        "t.zeros_like(x))` (explicit elementwise max). Avoid `t.relu` / `F.relu` — those would dodge "
        "the atom.\n\n"
        "**Why max(x, 0) and not `x * (x > 0)`?** The mask form loses the gradient signal at zero "
        "in a different way; `t.maximum` is the canonical PyTorch implementation pattern and matches "
        "what ARENA expects. (At exactly `x == 0` both forms have subgradient ambiguity, but `maximum` "
        "consistently picks the right branch.)\n\n"
        "**Anatomy.**\n"
        "1. `super().__init__()` — that's the entire `__init__`.\n"
        "2. `forward(self, x): return t.maximum(x, t.zeros_like(x))`.\n"
        "3. `.parameters()` returns an EMPTY iterator — no learnables. The test verifies this."
    ),
    "prompt_body": (
        "Define a class `MyReLU(nn.Module)` and a builder `cx23_build_relu()`.\n\n"
        "`MyReLU.__init__` calls `super().__init__()` and does NOTHING else — no parameters, no "
        "buffers, no children.\n\n"
        "`MyReLU.forward(self, x)` must return `t.maximum(x, t.zeros_like(x))` — the elementwise "
        "max against zero.\n\n"
        "**Forbidden shortcuts** (these dodge the atom):\n"
        "- `t.relu(x)` / `F.relu(x)` / `nn.functional.relu(x)` — would not exercise the elementwise "
        "max construction.\n"
        "- `x.clamp(min=0)` — also a shortcut. The drill is specifically about the `t.maximum` "
        "pattern.\n\n"
        "(The test cannot easily detect which built-in you used since results match — but on the "
        "honor system, write `t.maximum(...)`.)"
    ),
    "stub_body": (
        "class MyReLU(nn.Module):\n"
        "    def __init__(self):\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError\n"
        "\n"
        "\n"
        "def cx23_build_relu() -> 'MyReLU':\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: parameterless — no learnables, no buffers, no children.\n"
        "m = cx23_build_relu()\n"
        "assert isinstance(m, nn.Module)\n"
        "assert list(m.parameters()) == [], 'ReLU has no parameters — got some'\n"
        "assert list(m.buffers()) == [], 'ReLU has no buffers — got some'\n"
        "assert list(m.children()) == [], 'ReLU has no submodules — got some'\n"
        "\n"
        "# Case B: numerics on a hand-built case covering negative, zero, positive.\n"
        "x = t.tensor([-2.0, -0.1, 0.0, 0.1, 3.0])\n"
        "y = m(x)\n"
        "assert tuple(y.shape) == (5,)\n"
        "assert t.equal(y, t.tensor([0.0, 0.0, 0.0, 0.1, 3.0]))\n"
        "\n"
        "# Case C: matches the reference t.relu / F.relu (math agreement, even though the\n"
        "# implementation should use t.maximum internally).\n"
        "t.manual_seed(0)\n"
        "x = t.randn(4, 5)\n"
        "y = m(x)\n"
        "assert t.equal(y, t.relu(x))\n"
        "assert t.equal(y, F.relu(x))\n"
        "\n"
        "# Case D: shape pass-through — 4-D input goes in, same shape comes out.\n"
        "x = t.randn(2, 3, 4, 5)\n"
        "y = m(x)\n"
        "assert tuple(y.shape) == (2, 3, 4, 5)\n"
        "assert (y >= 0).all().item(), 'ReLU output must be non-negative everywhere'\n"
        "\n"
        "# Case E: gradient through the positive branch is 1, through the negative branch is 0.\n"
        "x = t.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)\n"
        "y = m(x)\n"
        "y.sum().backward()\n"
        "assert t.equal(x.grad, t.tensor([0.0, 1.0, 0.0, 1.0])), f'wrong grads: {x.grad}'\n"
        "\n"
        "# Case F: composes into nn.Sequential cleanly — slots in like nn.ReLU().\n"
        "stack = nn.Sequential(nn.Linear(4, 3), cx23_build_relu())\n"
        "x = t.randn(2, 4)\n"
        "y = stack(x)\n"
        "assert tuple(y.shape) == (2, 3)\n"
        "assert (y >= 0).all().item()"
    ),
    "solution_body": (
        "class MyReLU(nn.Module):\n"
        "    def __init__(self):\n"
        "        # Atom A (nn-module-subclass): super().__init__() is the WHOLE __init__ for\n"
        "        # parameterless layers — no params, no buffers, no children to register.\n"
        "        super().__init__()\n"
        "\n"
        "    def forward(self, x):\n"
        "        # Atom B (relu-elementwise-max): the canonical elementwise max(x, 0) form.\n"
        "        # t.maximum broadcasts the zero tensor against x and picks the larger at each cell.\n"
        "        return t.maximum(x, t.zeros_like(x))\n"
        "\n"
        "\n"
        "def cx23_build_relu() -> 'MyReLU':\n"
        "    return MyReLU()"
    ),
    "solution_notes": (
        "Wrapping ReLU as a Module (vs leaving it as a free function) lets it participate in the "
        "module tree — appears in `repr(model)`, slots into `nn.Sequential`, gets toggled by "
        "`.train()/.eval()` (irrelevant for ReLU but consistent), and shows up in `state_dict()` "
        "(empty for parameterless layers, but consistent)."
    ),
    "extra_imports": _EXTRA,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["nn-module-subclass", "relu-elementwise-max"],
    "lo": (
        "Compose nn.Module subclassing (parameterless variant — super().__init__ only) with the "
        "ReLU elementwise-max atom (t.maximum(x, zeros_like(x))) to produce an nn.ReLU-equivalent "
        "module that drops into nn.Sequential."
    ),
}


# ===========================================================================
# cx24 — subclass + BatchNorm affine params (gamma, beta as nn.Parameter)
# ===========================================================================
spec_24 = {
    "atom_ids": ["nn-module-subclass", "batchnorm-affine-params"],
    "subtopics": _subs(["nn-module-subclass", "batchnorm-affine-params"]),
    "primary_atom": "nn-module-subclass",
    "part": "part2",
    "exercise_index": 24,
    "exercise_title": "custom BatchNorm affine: gamma and beta as nn.Parameter inside a subclass",
    "slug": "custom-batchnorm-affine-params",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "BatchNorm has TWO halves: the normalization (mean/var) and the affine (gamma, beta). This "
        "drill exercises only the AFFINE half — pretend the input is already zero-mean unit-variance "
        "and just learn the per-channel scale and shift.\n"
        "- **nn-module-subclass** — `class Foo(nn.Module)` + `super().__init__()` + forward.\n"
        "- **batchnorm-affine-params** — the two learnables: `gamma` (scale) and `beta` (shift), each "
        "shape `(num_features,)`. Standard init: `gamma = ones`, `beta = zeros` (so the affine is "
        "initially identity).\n\n"
        "**Why init gamma=1, beta=0?** At init the affine `gamma * x + beta` must be the identity, "
        "so a freshly-constructed BatchNorm doesn't perturb the input distribution. Any other init "
        "would inject noise from step zero.\n\n"
        "**Shape and broadcast.** `x` is `(N, C, H, W)`. `gamma` is `(C,)`. To broadcast across "
        "`N, H, W`, reshape `gamma` to `(1, C, 1, 1)` — same for `beta`. The output is "
        "`gamma_b * x + beta_b` where `gamma_b, beta_b` are the broadcast-reshaped versions.\n\n"
        "**Anatomy.**\n"
        "1. `super().__init__()` + store `num_features`.\n"
        "2. `self.weight = nn.Parameter(t.ones(num_features))`  (gamma — convention name in PyTorch).\n"
        "3. `self.bias = nn.Parameter(t.zeros(num_features))`   (beta — convention name in PyTorch).\n"
        "4. `forward(x)`: reshape `weight` and `bias` to `(1, C, 1, 1)`, then `weight_b * x + bias_b`."
    ),
    "prompt_body": (
        "Define a class `MyBNAffine(nn.Module)` and a builder `cx24_build_bn_affine(num_features)`.\n\n"
        "`MyBNAffine.__init__(self, num_features)` must:\n"
        "1. `super().__init__()`.\n"
        "2. Store `self.num_features = num_features`.\n"
        "3. `self.weight = nn.Parameter(t.ones(num_features))`  — gamma, initialized to ALL ONES.\n"
        "4. `self.bias = nn.Parameter(t.zeros(num_features))`   — beta, initialized to ALL ZEROS.\n\n"
        "(We use the PyTorch-canonical attribute names `weight` and `bias` — the same names "
        "`nn.BatchNorm2d` uses for its affine params.)\n\n"
        "`MyBNAffine.forward(self, x)` takes a `(N, C, H, W)` tensor and applies "
        "`weight_b * x + bias_b`, where `weight_b` and `bias_b` are the affine params reshaped to "
        "`(1, C, 1, 1)` so they broadcast across batch and spatial.\n\n"
        "This drill skips the mean/var normalization — pretend the input is already normalized. The "
        "atom under test is the AFFINE half of BatchNorm specifically."
    ),
    "stub_body": (
        "class MyBNAffine(nn.Module):\n"
        "    def __init__(self, num_features: int):\n"
        "        raise NotImplementedError\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError\n"
        "\n"
        "\n"
        "def cx24_build_bn_affine(num_features: int) -> 'MyBNAffine':\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: both params exist, right shape, right init.\n"
        "m = cx24_build_bn_affine(num_features=8)\n"
        "assert isinstance(m, nn.Module)\n"
        "assert isinstance(m.weight, nn.Parameter), 'gamma must be nn.Parameter (attr name: weight)'\n"
        "assert isinstance(m.bias, nn.Parameter), 'beta must be nn.Parameter (attr name: bias)'\n"
        "assert tuple(m.weight.shape) == (8,), f'gamma shape (C,): got {tuple(m.weight.shape)}'\n"
        "assert tuple(m.bias.shape) == (8,), f'beta shape (C,): got {tuple(m.bias.shape)}'\n"
        "assert t.equal(m.weight.data, t.ones(8)), 'gamma must init to ones — got something else'\n"
        "assert t.equal(m.bias.data, t.zeros(8)), 'beta must init to zeros — got something else'\n"
        "assert m.weight.requires_grad and m.bias.requires_grad\n"
        "\n"
        "# Case B: identity at init — gamma=1, beta=0 means the affine is identity.\n"
        "t.manual_seed(0)\n"
        "x = t.randn(2, 8, 4, 4)\n"
        "y = m(x)\n"
        "assert tuple(y.shape) == (2, 8, 4, 4), f'shape preservation: got {tuple(y.shape)}'\n"
        "assert t.allclose(y, x, atol=1e-6), 'init affine must be identity (gamma=1, beta=0)'\n"
        "\n"
        "# Case C: set gamma and beta to known per-channel values and verify the broadcast.\n"
        "m = cx24_build_bn_affine(num_features=3)\n"
        "with t.no_grad():\n"
        "    m.weight.copy_(t.tensor([2.0, 0.0, -1.0]))\n"
        "    m.bias.copy_(t.tensor([10.0, 20.0, 30.0]))\n"
        "x = t.ones(1, 3, 2, 2)  # all-ones input, easy to check.\n"
        "y = m(x)\n"
        "# Channel 0: 2*1 + 10 = 12.  Channel 1: 0*1 + 20 = 20.  Channel 2: -1*1 + 30 = 29.\n"
        "expected = t.tensor([12.0, 20.0, 29.0]).reshape(1, 3, 1, 1).expand(1, 3, 2, 2)\n"
        "assert t.allclose(y, expected), f'per-channel affine broken; got {y[0, :, 0, 0]}'\n"
        "\n"
        "# Case D: per-channel agreement vs reference manual broadcast on a random input.\n"
        "t.manual_seed(0)\n"
        "m = cx24_build_bn_affine(num_features=5)\n"
        "with t.no_grad():\n"
        "    m.weight.copy_(t.randn(5))\n"
        "    m.bias.copy_(t.randn(5))\n"
        "x = t.randn(3, 5, 4, 6)\n"
        "y = m(x)\n"
        "expected = m.weight.view(1, 5, 1, 1) * x + m.bias.view(1, 5, 1, 1)\n"
        "assert t.allclose(y, expected, atol=1e-6), 'affine must broadcast as gamma_b * x + beta_b'\n"
        "\n"
        "# Case E: matches nn.BatchNorm2d's AFFINE-only behavior (with running stats disabled and\n"
        "# eval-mode so the normalization half is a no-op identity).\n"
        "m = cx24_build_bn_affine(num_features=4)\n"
        "with t.no_grad():\n"
        "    m.weight.copy_(t.tensor([1.5, 2.0, 0.5, 1.0]))\n"
        "    m.bias.copy_(t.tensor([0.1, 0.2, 0.3, 0.4]))\n"
        "ref = nn.BatchNorm2d(4, affine=True, track_running_stats=False)\n"
        "with t.no_grad():\n"
        "    ref.weight.copy_(m.weight)\n"
        "    ref.bias.copy_(m.bias)\n"
        "# Construct an input that is already normalized per-channel so the BN normalization is a\n"
        "# no-op — then the only difference between ref and m is whether the affine matches.\n"
        "# nn.BatchNorm2d in train mode uses BATCH stats; if x is mean-0 var-1 per channel within\n"
        "# the (N, H, W) slice, the normalization yields x back (modulo eps).\n"
        "N, C, H, W = 8, 4, 3, 3\n"
        "raw = t.randn(N, C, H, W)\n"
        "# Standardize each channel across the (N, H, W) axes so BN's normalization is a no-op.\n"
        "axes = (0, 2, 3)\n"
        "mean = raw.mean(dim=axes, keepdim=True)\n"
        "std = raw.std(dim=axes, keepdim=True, unbiased=False)\n"
        "x_std = (raw - mean) / (std + 1e-8)\n"
        "y_ours = m(x_std)\n"
        "y_ref = ref(x_std)\n"
        "assert t.allclose(y_ours, y_ref, atol=1e-3), 'must match nn.BatchNorm2d affine on already-normalized input'\n"
        "\n"
        "# Case F: .parameters() returns exactly 2 tensors.\n"
        "m = cx24_build_bn_affine(num_features=6)\n"
        "ps = list(m.parameters())\n"
        "assert len(ps) == 2, f'expected 2 params (weight, bias); got {len(ps)}'\n"
        "named = dict(m.named_parameters())\n"
        "assert set(named.keys()) == {'weight', 'bias'}"
    ),
    "solution_body": (
        "class MyBNAffine(nn.Module):\n"
        "    def __init__(self, num_features: int):\n"
        "        # Atom A (nn-module-subclass): scaffold first.\n"
        "        super().__init__()\n"
        "        self.num_features = num_features\n"
        "        # Atom B (batchnorm-affine-params): gamma (scale) inits to ONES so the affine is\n"
        "        # identity at construction. PyTorch attribute name is `weight` for consistency\n"
        "        # with nn.BatchNorm2d.\n"
        "        self.weight = nn.Parameter(t.ones(num_features))\n"
        "        # beta (shift) inits to ZEROS — also for identity init. Attribute name is `bias`.\n"
        "        self.bias = nn.Parameter(t.zeros(num_features))\n"
        "\n"
        "    def forward(self, x):\n"
        "        # x shape: (N, C, H, W). gamma / beta are (C,) — reshape to (1, C, 1, 1) so they\n"
        "        # broadcast across batch and spatial axes.\n"
        "        gamma_b = self.weight.view(1, -1, 1, 1)\n"
        "        beta_b = self.bias.view(1, -1, 1, 1)\n"
        "        return gamma_b * x + beta_b\n"
        "\n"
        "\n"
        "def cx24_build_bn_affine(num_features: int) -> 'MyBNAffine':\n"
        "    return MyBNAffine(num_features)"
    ),
    "solution_notes": (
        "Init choice is non-negotiable: gamma=1, beta=0 makes the affine the identity at "
        "construction, so a freshly-built BatchNorm doesn't shift the input distribution. The "
        "broadcast reshape `(C,) -> (1, C, 1, 1)` is the canonical pattern for any per-channel "
        "affine on a 4-D feature map."
    ),
    "extra_imports": _EXTRA,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["nn-module-subclass", "batchnorm-affine-params"],
    "lo": (
        "Compose nn.Module subclassing with the BatchNorm affine atom (gamma=ones, beta=zeros, both "
        "as nn.Parameter, reshaped to (1, C, 1, 1) for broadcast across batch and spatial) to "
        "implement the affine half of BatchNorm2d from scratch."
    ),
}


SPECS = [spec_19, spec_20, spec_21, spec_22, spec_23, spec_24]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
