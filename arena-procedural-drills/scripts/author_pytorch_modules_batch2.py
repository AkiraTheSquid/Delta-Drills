#!/usr/bin/env python3
"""Author Colab-native standalones for PyTorch nn.Module mechanics atoms.

Batch 2: covers four atoms that all ARENA chapter-0 CNN exercises assume:
    - nn-module-subclass    (3 exercises)
    - nn-parameter-wrap     (2 exercises)
    - module-composition    (2 exercises)
    - module-extra-repr     (1 exercise)

Each drill is one constituent skill that ARENA's `make_cnn` / Linear / Conv2d
/ BatchNorm / ResidualBlock exercises silently assume the learner already has.
Brand-new folder `prereqs_pytorch_modules/` — exercises numbered from ex1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_pytorch_modules"

RECAP_SUBCLASS = (
    "## `nn.Module` subclassing — quick refresher\n"
    "\n"
    "Every learnable building block in PyTorch is an `nn.Module` subclass. The "
    "minimal pattern is:\n"
    "\n"
    "```python\n"
    "class MyLayer(nn.Module):\n"
    "    def __init__(self, ...):\n"
    "        super().__init__()      # MUST be first — wires up _parameters / _modules dicts\n"
    "        self.weight = nn.Parameter(...)\n"
    "    def forward(self, x):\n"
    "        return ...              # never call .forward() directly — use module(x)\n"
    "```\n"
    "\n"
    "**Two non-obvious rules.**\n"
    "1. If `__init__` does anything (assigns Parameters, sub-Modules, or "
    "buffers), it MUST call `super().__init__()` first. Forgetting this raises "
    "`AttributeError: cannot assign parameter before Module.__init__() call`.\n"
    "2. Modules with no state can omit `__init__` entirely and just define "
    "`forward` (e.g. ARENA's `ReLU`). The base `nn.Module.__init__` runs "
    "implicitly.\n"
    "\n"
    "**Call convention.** Use `module(x)`, never `module.forward(x)` — the "
    "`__call__` wrapper runs hooks (pre/post forward, gradient hooks) that you "
    "lose by calling forward directly."
)

RECAP_PARAMETER = (
    "## `nn.Parameter` — quick refresher\n"
    "\n"
    "`nn.Parameter(tensor)` is a tensor subclass with one job: when assigned as "
    "an attribute of an `nn.Module`, it gets auto-registered in the module's "
    "`_parameters` dict so it shows up in `.parameters()`, `.state_dict()`, "
    "and gets moved by `.to(device)` / `.cuda()`.\n"
    "\n"
    "**Parameter vs raw tensor.** `self.w = torch.randn(3)` — invisible to the "
    "optimizer. `self.w = nn.Parameter(torch.randn(3))` — included.\n"
    "\n"
    "**Parameter vs buffer.** Both round-trip through `state_dict()`. Only "
    "Parameters are trainable (`.requires_grad=True` by default and included in "
    "`.parameters()`). Buffers are for non-learnable state — running stats, "
    "position encodings, attention masks. Register with `self.register_buffer("
    "'running_mean', torch.zeros(C))`.\n"
    "\n"
    "**Gotcha — default dtype.** `nn.Parameter(torch.tensor([1, 2, 3]))` "
    "creates an `int64` parameter, which optimizers reject. Always pass a "
    "float tensor or call `.float()` first."
)

RECAP_COMPOSITION = (
    "## Module composition — quick refresher\n"
    "\n"
    "Modules compose by assignment. Any `nn.Module` instance assigned to "
    "`self.<name>` inside `__init__` is auto-registered as a child module — "
    "visible in `.children()`, `.named_modules()`, and recursively included in "
    "`.parameters()` / `.state_dict()`.\n"
    "\n"
    "**Two composition styles.**\n"
    "1. **Named attributes** — `self.linear1 = nn.Linear(...)`, "
    "`self.linear2 = nn.Linear(...)`. Use when the forward pass branches "
    "(residual blocks, attention heads, gating).\n"
    "2. **`nn.Sequential(*modules)`** — wraps a list of Modules into a single "
    "callable that pipes the output of one into the input of the next. Use "
    "when the forward is a strict left-to-right pipeline.\n"
    "\n"
    "**Lists need `nn.ModuleList`, not `list`.** A plain Python list of "
    "Modules assigned to an attribute does NOT register the children — "
    "their parameters become invisible. Use `nn.ModuleList(...)` (registers + "
    "supports indexing) or `nn.Sequential(...)` (registers + auto-pipes)."
)

RECAP_EXTRA_REPR = (
    "## `extra_repr` — quick refresher\n"
    "\n"
    "`nn.Module.__repr__` is already implemented for you — it prints the class "
    "name and recursively the children's reprs. What it CAN'T know is which "
    "hyperparameters you stored on `self`. That's what `extra_repr` is for: "
    "override it to return a single string of `key=value` pairs that get "
    "inserted between the parentheses of the class repr.\n"
    "\n"
    "```python\n"
    "class Linear(nn.Module):\n"
    "    def __init__(self, in_f, out_f, bias=True):\n"
    "        super().__init__()\n"
    "        self.in_features, self.out_features, self.has_bias = in_f, out_f, bias\n"
    "        ...\n"
    "    def extra_repr(self):\n"
    "        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias}'\n"
    "```\n"
    "\n"
    "Then `print(Linear(3, 4))` shows `Linear(in_features=3, out_features=4, "
    "bias=True)` instead of the bare `Linear()`.\n"
    "\n"
    "**Gotcha — print the bool, not the bias tensor.** ARENA's Linear "
    "exercise calls this out: `bias={self.bias}` would dump the full tensor; "
    "`bias={self.bias is not None}` prints the intended `True`/`False`."
)


SPECS = [
    # ============================================================ nn-module-subclass / ex1
    {
        "atom_id": "nn-module-subclass",
        "subtopic": "PyTorch: nn.Module subclassing",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_SUBCLASS,
        "exercise_index": 1,
        "exercise_title": "minimal stateless Module (no __init__)",
        "slug": "minimal-stateless-module",
        "bloom_level": "Apply",
        "difficulty_num": 1,
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "keywords": ["nn.Module", "forward", "stateless", "ReLU-like"],
        "kcs": ["module-subclass-trivial-forward", "module-call-via-dunder-call"],
        "lo": (
            "Define an nn.Module subclass that has no state — only a forward "
            "method — and invoke it via __call__."
        ),
        "prompt_body": (
            "Implement `ex1_make_square_module()`. The minimal possible Module:\n\n"
            "1. Define a class `SquareLayer` that subclasses `t.nn.Module`.\n"
            "2. Do NOT write an `__init__` method — the layer has no state.\n"
            "3. Implement `forward(self, x)` returning `x ** 2` (elementwise).\n"
            "4. Return an INSTANCE of `SquareLayer` from `ex1_make_square_module()`.\n\n"
            "This mirrors ARENA's first nn.Module exercise (the trivial ReLU): "
            "when there's no state to declare, you can skip `__init__` entirely "
            "and the base `nn.Module.__init__` runs implicitly.\n\n"
            "The test calls the module via `module(x)` (NOT `module.forward(x)`) "
            "to confirm the `__call__` plumbing works without you wiring "
            "anything up."
        ),
        "stub": (
            "def ex1_make_square_module():\n"
            '    """Return an instance of a stateless nn.Module that squares its input."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "module = ex1_make_square_module()\n"
            "# Must be an nn.Module subclass instance.\n"
            "assert isinstance(module, t.nn.Module), f'expected nn.Module, got {type(module).__name__}'\n"
            "# Must NOT define __init__ (i.e. inherits the base Module.__init__).\n"
            "assert '__init__' not in type(module).__dict__, (\n"
            "    'this exercise tests the no-__init__ pattern; remove your __init__ override'\n"
            ")\n"
            "# forward must square elementwise.\n"
            "x = t.tensor([-2.0, -1.0, 0.0, 1.0, 3.0])\n"
            "y = module(x)\n"
            "expected = t.tensor([4.0, 1.0, 0.0, 1.0, 9.0])\n"
            "assert t.allclose(y, expected), f'expected {expected}, got {y}'\n"
            "# No state — parameters() must be empty.\n"
            "assert list(module.parameters()) == [], 'a stateless module should have zero parameters'\n"
            "# state_dict must be empty too.\n"
            "assert len(module.state_dict()) == 0, 'a stateless module should have an empty state_dict'\n"
            "# Calling via __call__ vs .forward must give the same result\n"
            "# (this is the only difference that matters for hooks downstream).\n"
            "x2 = t.randn(3, 4)\n"
            "assert t.allclose(module(x2), module.forward(x2)), 'module(x) and module.forward(x) must agree on output'"
        ),
        "solution_body": (
            "def ex1_make_square_module():\n"
            "    class SquareLayer(t.nn.Module):\n"
            "        def forward(self, x):\n"
            "            return x ** 2\n"
            "    return SquareLayer()"
        ),
        "solution_notes": (
            "**Why no `__init__` works.** `nn.Module.__init__` is what creates "
            "`self._parameters`, `self._modules`, `self._buffers` (the dicts "
            "auto-registration writes to). When you don't override it, Python "
            "calls the base version automatically when you instantiate the "
            "class. No state → no need to register anything → no `__init__` "
            "needed.\n\n"
            "**Why use `module(x)` not `module.forward(x)`.** The base class "
            "implements `__call__` which runs registered pre/post-forward "
            "hooks and then calls your `forward`. Calling `.forward` directly "
            "skips the hooks — fine for this trivial case, fatal once you "
            "attach gradient checkpointing, profiling, or grad hooks."
        ),
    },
    # ============================================================ nn-module-subclass / ex2
    {
        "atom_id": "nn-module-subclass",
        "subtopic": "PyTorch: nn.Module subclassing",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_SUBCLASS,
        "exercise_index": 2,
        "exercise_title": "diagnose missing super().__init__()",
        "slug": "diagnose-missing-super-init",
        "bloom_level": "Analyze",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["super", "__init__", "debugging", "AttributeError"],
        "kcs": ["module-init-super-call"],
        "lo": (
            "Diagnose the AttributeError raised when an nn.Module subclass "
            "with state forgets to call super().__init__(), then fix it."
        ),
        "prompt_body": (
            "Implement `ex2_fix_broken_scaler(scale_value)`. The exercise has "
            "two parts:\n\n"
            "**Part A — observe the bug.** A buggy class `BrokenScaler` is "
            "defined for you below the test cell (in the solution). It has "
            "`self.scale = nn.Parameter(...)` in `__init__` but forgets to "
            "call `super().__init__()` first. Attempting to instantiate it "
            "raises `AttributeError: cannot assign parameter before Module."
            "__init__() call`.\n\n"
            "**Part B — write the fix.** Define a class `FixedScaler` that:\n"
            "1. Subclasses `t.nn.Module`.\n"
            "2. In `__init__(self, scale_value)` calls `super().__init__()` "
            "FIRST.\n"
            "3. Stores `self.scale = nn.Parameter(t.tensor(float(scale_value)))`.\n"
            "4. `forward(self, x)` returns `x * self.scale`.\n\n"
            "Return an INSTANCE of `FixedScaler` initialized with `scale_value`.\n\n"
            "The test asserts: (a) the broken version really does raise, "
            "(b) your fixed version constructs cleanly, (c) the parameter is "
            "registered (`.parameters()` is non-empty), (d) forward works."
        ),
        "stub": (
            "def ex2_fix_broken_scaler(scale_value: float):\n"
            '    """Return an instance of a Module that scales by scale_value, with super().__init__() correctly placed."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Reproduce the bug — confirm the broken pattern fails.\n"
            "class BrokenScaler(t.nn.Module):\n"
            "    def __init__(self, scale_value):\n"
            "        # BUG: super().__init__() missing\n"
            "        self.scale = t.nn.Parameter(t.tensor(float(scale_value)))\n"
            "    def forward(self, x):\n"
            "        return x * self.scale\n"
            "try:\n"
            "    BrokenScaler(2.0)\n"
            "    raise RuntimeError('BrokenScaler should have raised AttributeError')\n"
            "except AttributeError as e:\n"
            "    assert 'Module.__init__' in str(e) or 'cannot assign' in str(e), (\n"
            "        f'expected an AttributeError about Module.__init__, got: {e}'\n"
            "    )\n"
            "    print(f'  observed expected error: {e}')\n"
            "\n"
            "# Now the fix.\n"
            "fixed = ex2_fix_broken_scaler(2.5)\n"
            "assert isinstance(fixed, t.nn.Module), f'expected nn.Module, got {type(fixed).__name__}'\n"
            "# Parameter must be registered (this is what super().__init__() enables).\n"
            "params = list(fixed.parameters())\n"
            "assert len(params) == 1, f'expected exactly 1 registered parameter, got {len(params)}'\n"
            "assert isinstance(params[0], t.nn.Parameter), f'registered tensor should be nn.Parameter, got {type(params[0]).__name__}'\n"
            "# Forward should scale by 2.5.\n"
            "x = t.tensor([1.0, 2.0, 4.0])\n"
            "y = fixed(x)\n"
            "expected = t.tensor([2.5, 5.0, 10.0])\n"
            "assert t.allclose(y, expected), f'expected {expected}, got {y}'\n"
            "# state_dict must round-trip the scale.\n"
            "sd = fixed.state_dict()\n"
            "assert 'scale' in sd, f'expected key \"scale\" in state_dict, got {list(sd.keys())}'\n"
            "assert sd['scale'].item() == 2.5"
        ),
        "solution_body": (
            "def ex2_fix_broken_scaler(scale_value: float):\n"
            "    class FixedScaler(t.nn.Module):\n"
            "        def __init__(self, scale_value):\n"
            "            super().__init__()\n"
            "            self.scale = t.nn.Parameter(t.tensor(float(scale_value)))\n"
            "        def forward(self, x):\n"
            "            return x * self.scale\n"
            "    return FixedScaler(scale_value)"
        ),
        "solution_notes": (
            "**Why the error message points at Module.__init__.** The "
            "`nn.Module.__setattr__` override intercepts every attribute "
            "assignment on a Module to check whether the value is a "
            "`Parameter` / `Module` / `Buffer` and route it to the right "
            "internal dict. But those dicts (`_parameters`, `_modules`, "
            "`_buffers`) don't exist until `Module.__init__` creates them. So "
            "the first `self.scale = nn.Parameter(...)` assignment can't find "
            "`self._parameters` to write into → AttributeError.\n\n"
            "**Why this only fails for stateful Modules.** The `SquareLayer` "
            "from ex1 had no attribute assignments in its (absent) `__init__`, "
            "so the missing super() call never bit. The bug ONLY surfaces "
            "when you assign a Parameter / Module / Buffer attribute."
        ),
    },
    # ============================================================ nn-module-subclass / ex3
    {
        "atom_id": "nn-module-subclass",
        "subtopic": "PyTorch: nn.Module subclassing",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_SUBCLASS,
        "exercise_index": 3,
        "exercise_title": "Linear layer from scratch",
        "slug": "linear-layer-from-scratch",
        "bloom_level": "Create",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["Linear", "forward", "matmul", "bias"],
        "kcs": ["module-forward-signature", "module-init-super-call"],
        "lo": (
            "Build an end-to-end nn.Module with __init__ + super + forward "
            "that implements a Linear layer, returning the correct shape."
        ),
        "prompt_body": (
            "Implement `MyLinear` — a from-scratch `nn.Linear` clone (the "
            "weights are passed in so we focus on the Module mechanics, NOT "
            "initialization):\n\n"
            "1. `class MyLinear(t.nn.Module)` with `__init__(self, weight, "
            "bias)`:\n"
            "   - Call `super().__init__()` first.\n"
            "   - Wrap and store the inputs as Parameters: "
            "`self.weight = nn.Parameter(weight)`, "
            "`self.bias = nn.Parameter(bias)`.\n"
            "2. `forward(self, x: Tensor) -> Tensor` computes "
            "`x @ self.weight.T + self.bias`.\n"
            "   - `x` has shape `(*batch, in_features)`.\n"
            "   - `self.weight` has shape `(out_features, in_features)` "
            "(matching `nn.Linear`'s convention).\n"
            "   - `self.bias` has shape `(out_features,)`.\n"
            "   - Output has shape `(*batch, out_features)`.\n\n"
            "The function `ex3_build_linear(weight, bias)` should return an "
            "instance of `MyLinear` built with the supplied tensors.\n\n"
            "**Don't reinvent initialization** — the test passes specific "
            "weight + bias tensors so we can check the forward output by "
            "hand."
        ),
        "stub": (
            "def ex3_build_linear(weight: Tensor, bias: Tensor):\n"
            '    """Return a MyLinear instance wrapping the given weight (out, in) and bias (out,) tensors."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-crafted weight + bias so we can check forward output exactly.\n"
            "weight = t.tensor([[1.0, 2.0, 3.0],\n"
            "                   [4.0, 5.0, 6.0]])  # (out=2, in=3)\n"
            "bias = t.tensor([0.5, -1.0])           # (out=2,)\n"
            "linear = ex3_build_linear(weight, bias)\n"
            "\n"
            "# Module shape checks.\n"
            "assert isinstance(linear, t.nn.Module), f'expected nn.Module, got {type(linear).__name__}'\n"
            "params = dict(linear.named_parameters())\n"
            "assert set(params.keys()) == {'weight', 'bias'}, f'expected weight+bias, got {set(params.keys())}'\n"
            "assert isinstance(params['weight'], t.nn.Parameter)\n"
            "assert isinstance(params['bias'], t.nn.Parameter)\n"
            "assert params['weight'].shape == (2, 3)\n"
            "assert params['bias'].shape == (2,)\n"
            "\n"
            "# Forward — single sample.\n"
            "x = t.tensor([1.0, 1.0, 1.0])     # (3,) → input row of ones\n"
            "y = linear(x)\n"
            "# expected: weight @ x + bias = [1+2+3, 4+5+6] + [0.5, -1.0] = [6.5, 14.0]\n"
            "expected = t.tensor([6.5, 14.0])\n"
            "assert y.shape == (2,), f'expected (2,), got {tuple(y.shape)}'\n"
            "assert t.allclose(y, expected), f'expected {expected}, got {y}'\n"
            "\n"
            "# Forward — batched input.\n"
            "x_batch = t.tensor([[1.0, 0.0, 0.0],\n"
            "                    [0.0, 1.0, 0.0],\n"
            "                    [0.0, 0.0, 1.0]])  # (3, 3) identity-ish rows\n"
            "y_batch = linear(x_batch)\n"
            "# Each row picks one column of weight.T → output is weight.T + bias broadcast.\n"
            "expected_batch = weight.T + bias  # (3, 2)\n"
            "assert y_batch.shape == (3, 2)\n"
            "assert t.allclose(y_batch, expected_batch), f'batched forward mismatch:\\n{y_batch}\\nvs\\n{expected_batch}'\n"
            "\n"
            "# Forward — leading-dim flexibility (B, S, in_features).\n"
            "x_3d = t.randn(4, 7, 3, generator=t.Generator().manual_seed(0))\n"
            "y_3d = linear(x_3d)\n"
            "assert y_3d.shape == (4, 7, 2), f'expected (4,7,2), got {tuple(y_3d.shape)}'"
        ),
        "solution_body": (
            "def ex3_build_linear(weight: Tensor, bias: Tensor):\n"
            "    class MyLinear(t.nn.Module):\n"
            "        def __init__(self, weight, bias):\n"
            "            super().__init__()\n"
            "            self.weight = t.nn.Parameter(weight)\n"
            "            self.bias = t.nn.Parameter(bias)\n"
            "        def forward(self, x: Tensor) -> Tensor:\n"
            "            return x @ self.weight.T + self.bias\n"
            "    return MyLinear(weight, bias)"
        ),
        "solution_notes": (
            "**Why `weight.T` not `weight`.** PyTorch's `nn.Linear` stores "
            "weight as `(out_features, in_features)` — the transpose of what "
            "you'd write in a math textbook for `y = Wx` — so that "
            "`weight @ x` lines up dimensionally for a single sample but for "
            "batched input we want `x @ weight.T` (shape `(B, in) @ (in, out)"
            " = (B, out)`). Matching this convention means your `MyLinear` "
            "can swap in for `nn.Linear` (e.g. load its `state_dict`).\n\n"
            "**Why bias broadcasts.** `self.bias` is `(out_features,)`. "
            "Adding it to a `(*batch, out_features)` tensor broadcasts over "
            "all leading batch dims for free — no `unsqueeze` needed.\n\n"
            "**Leading-dim flexibility comes from `@`.** The `@` operator "
            "treats everything but the last two axes as batch dims, so "
            "`(B, S, in) @ (in, out) → (B, S, out)` works without any "
            "reshape. This is one reason real PyTorch code prefers `@` over "
            "`torch.matmul` calls + explicit reshaping."
        ),
    },
    # ============================================================ nn-parameter-wrap / ex1
    {
        "atom_id": "nn-parameter-wrap",
        "subtopic": "PyTorch: nn.Parameter",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_PARAMETER,
        "exercise_index": 1,
        "exercise_title": "Parameter vs raw tensor — visibility test",
        "slug": "parameter-vs-raw-tensor-visibility",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["nn.Parameter", "parameters", "state_dict", "optimizer"],
        "kcs": ["parameter-wrap-tensor", "parameter-auto-register"],
        "lo": (
            "Demonstrate that wrapping a tensor with nn.Parameter is what "
            "makes it visible to .parameters(), .state_dict(), and the "
            "optimizer — a raw tensor attribute is invisible."
        ),
        "prompt_body": (
            "Implement `ex1_build_module_pair()` returning a tuple "
            "`(invisible_mod, visible_mod)`.\n\n"
            "Both Modules look almost identical — they each store a single "
            "tensor of shape `(4,)` as an attribute named `weight`. The "
            "ONLY difference:\n\n"
            "1. `InvisibleMod.weight = t.randn(4)` — raw tensor. NOT wrapped.\n"
            "2. `VisibleMod.weight = nn.Parameter(t.randn(4))` — wrapped.\n\n"
            "Use `t.manual_seed(42)` immediately before EACH `randn` call so "
            "both Modules hold the same numeric values (the test ignores the "
            "values; this just keeps reproducibility tidy).\n\n"
            "Both Modules must:\n"
            "- Subclass `t.nn.Module`.\n"
            "- Call `super().__init__()` in `__init__`.\n"
            "- Implement `forward(self, x)` returning `x * self.weight`.\n\n"
            "Return `(invisible_mod, visible_mod)` — both already instantiated.\n\n"
            "The test asserts that `invisible_mod` has ZERO entries in "
            "`.parameters()` and an EMPTY `.state_dict()`, while `visible_mod` "
            "has exactly one parameter named `weight`. Same code structure, "
            "totally different visibility — that's the whole point of "
            "`nn.Parameter`."
        ),
        "stub": (
            "def ex1_build_module_pair():\n"
            '    """Return (invisible_mod, visible_mod) — same shape, different visibility."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "invisible_mod, visible_mod = ex1_build_module_pair()\n"
            "\n"
            "# Both are nn.Modules.\n"
            "assert isinstance(invisible_mod, t.nn.Module)\n"
            "assert isinstance(visible_mod, t.nn.Module)\n"
            "\n"
            "# Invisible mod — raw tensor → registry empty.\n"
            "inv_params = list(invisible_mod.parameters())\n"
            "assert len(inv_params) == 0, (\n"
            "    f'invisible_mod should have 0 params, got {len(inv_params)} '\n"
            "    f'(did you wrap with nn.Parameter? you should NOT have, for invisible_mod)'\n"
            ")\n"
            "assert len(invisible_mod.state_dict()) == 0, 'invisible_mod state_dict should be empty'\n"
            "# Still callable.\n"
            "x = t.ones(4)\n"
            "y_inv = invisible_mod(x)\n"
            "assert y_inv.shape == (4,)\n"
            "\n"
            "# Visible mod — nn.Parameter → registered.\n"
            "vis_params = list(visible_mod.parameters())\n"
            "assert len(vis_params) == 1, (\n"
            "    f'visible_mod should have exactly 1 param, got {len(vis_params)} '\n"
            "    f'(did you wrap weight with nn.Parameter?)'\n"
            ")\n"
            "assert isinstance(vis_params[0], t.nn.Parameter)\n"
            "assert vis_params[0].shape == (4,)\n"
            "named = dict(visible_mod.named_parameters())\n"
            "assert 'weight' in named, f'expected param named \"weight\", got {list(named.keys())}'\n"
            "assert 'weight' in visible_mod.state_dict(), 'weight should round-trip via state_dict'\n"
            "y_vis = visible_mod(x)\n"
            "assert y_vis.shape == (4,)\n"
            "\n"
            "# requires_grad True by default for nn.Parameter, False (or absent) for raw tensor.\n"
            "assert vis_params[0].requires_grad, 'nn.Parameter defaults to requires_grad=True'\n"
            "\n"
            "# The smoking gun: SGD over invisible_mod has nothing to optimize.\n"
            "try:\n"
            "    t.optim.SGD(invisible_mod.parameters(), lr=0.1)\n"
            "    raise RuntimeError('SGD should reject an empty parameter list')\n"
            "except ValueError as e:\n"
            "    assert 'empty parameter list' in str(e) or 'got an empty' in str(e), (\n"
            "        f'expected ValueError about empty parameter list, got: {e}'\n"
            "    )\n"
            "    print(f'  observed expected SGD rejection: {e}')\n"
            "# But SGD over visible_mod is fine.\n"
            "opt = t.optim.SGD(visible_mod.parameters(), lr=0.1)\n"
            "assert len(opt.param_groups[0]['params']) == 1"
        ),
        "solution_body": (
            "def ex1_build_module_pair():\n"
            "    class InvisibleMod(t.nn.Module):\n"
            "        def __init__(self):\n"
            "            super().__init__()\n"
            "            t.manual_seed(42)\n"
            "            self.weight = t.randn(4)  # raw tensor — NOT a Parameter\n"
            "        def forward(self, x):\n"
            "            return x * self.weight\n"
            "    class VisibleMod(t.nn.Module):\n"
            "        def __init__(self):\n"
            "            super().__init__()\n"
            "            t.manual_seed(42)\n"
            "            self.weight = t.nn.Parameter(t.randn(4))  # wrapped → registered\n"
            "        def forward(self, x):\n"
            "            return x * self.weight\n"
            "    return InvisibleMod(), VisibleMod()"
        ),
        "solution_notes": (
            "**The mechanism.** `nn.Module.__setattr__` inspects the value of "
            "every attribute assignment. If it's a `Parameter`, it stores it "
            "in `self._parameters` (the dict `.parameters()` iterates). If "
            "it's a raw `Tensor`, it gets stored on `self.__dict__` like any "
            "Python attribute — invisible to `.parameters()`.\n\n"
            "**Why this is a footgun.** Forward still works fine with a raw "
            "tensor — you only notice the bug when training silently doesn't "
            "improve, because the optimizer never sees the weight. Always "
            "wrap learnable tensors in `nn.Parameter`.\n\n"
            "**The non-learnable case.** If you have a tensor that should "
            "round-trip via `state_dict` but NOT be trained (e.g. attention "
            "mask, running mean), use `self.register_buffer('name', tensor)` "
            "instead. That's the third option — and exactly what `ex2` "
            "explores."
        ),
    },
    # ============================================================ nn-parameter-wrap / ex2
    {
        "atom_id": "nn-parameter-wrap",
        "subtopic": "PyTorch: nn.Parameter",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_PARAMETER,
        "exercise_index": 2,
        "exercise_title": "Parameter vs buffer — pick the right registration",
        "slug": "parameter-vs-buffer",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["register_buffer", "nn.Parameter", "state_dict", "requires_grad"],
        "kcs": ["parameter-vs-buffer", "parameter-wrap-tensor"],
        "lo": (
            "Build a BatchNorm-style module that registers a learnable weight "
            "as nn.Parameter and a running statistic as a buffer, and verify "
            "the two registries differ correctly."
        ),
        "prompt_body": (
            "Implement `RunningScaler` — a stripped-down stand-in for "
            "BatchNorm that exercises the Parameter-vs-buffer choice:\n\n"
            "1. `class RunningScaler(t.nn.Module)` with `__init__(self, "
            "num_features)`:\n"
            "   - Call `super().__init__()` first.\n"
            "   - Register `weight` as a learnable `nn.Parameter` initialized "
            "to `t.ones(num_features)`.\n"
            "   - Register `running_mean` as a BUFFER (NOT a Parameter) "
            "initialized to `t.zeros(num_features)`, using "
            "`self.register_buffer('running_mean', t.zeros(num_features))`.\n"
            "2. `forward(self, x: Tensor) -> Tensor`:\n"
            "   - `x` has shape `(B, num_features)`.\n"
            "   - During training (`self.training is True`): update "
            "`self.running_mean` with the batch mean using EMA with momentum "
            "`0.1`:\n"
            "     `self.running_mean = 0.9 * self.running_mean + 0.1 * x.mean(dim=0)`\n"
            "     Wrap the update in `with t.no_grad():` so it doesn't pollute "
            "the autograd graph.\n"
            "   - Return `(x - self.running_mean) * self.weight` "
            "(broadcasts over the batch dim).\n\n"
            "Return an instance from `ex2_build_running_scaler(num_features)`.\n\n"
            "The test confirms `weight` lives in `parameters()`, "
            "`running_mean` lives in `buffers()` (NOT in `parameters()`), and "
            "BOTH round-trip via `state_dict()`."
        ),
        "stub": (
            "def ex2_build_running_scaler(num_features: int):\n"
            '    """Return a RunningScaler(num_features) instance with weight as Parameter and running_mean as buffer."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "mod = ex2_build_running_scaler(num_features=3)\n"
            "assert isinstance(mod, t.nn.Module)\n"
            "\n"
            "# weight should be in parameters() (learnable).\n"
            "named_params = dict(mod.named_parameters())\n"
            "assert set(named_params.keys()) == {'weight'}, (\n"
            "    f'expected only \"weight\" in parameters(), got {set(named_params.keys())}'\n"
            ")\n"
            "assert isinstance(named_params['weight'], t.nn.Parameter)\n"
            "assert named_params['weight'].shape == (3,)\n"
            "assert t.allclose(named_params['weight'], t.ones(3)), 'weight should init to ones'\n"
            "\n"
            "# running_mean should be in buffers() (NOT in parameters()).\n"
            "named_buffers = dict(mod.named_buffers())\n"
            "assert set(named_buffers.keys()) == {'running_mean'}, (\n"
            "    f'expected only \"running_mean\" in buffers(), got {set(named_buffers.keys())}'\n"
            ")\n"
            "assert named_buffers['running_mean'].shape == (3,)\n"
            "assert t.allclose(named_buffers['running_mean'], t.zeros(3)), 'running_mean should init to zeros'\n"
            "\n"
            "# CRITICAL distinction — running_mean must NOT appear in parameters().\n"
            "assert 'running_mean' not in named_params, (\n"
            "    'running_mean should be a buffer, not a parameter — '\n"
            "    'did you accidentally use nn.Parameter instead of register_buffer?'\n"
            ")\n"
            "\n"
            "# BUT both must round-trip via state_dict (that's why we use register_buffer over a raw attribute).\n"
            "sd = mod.state_dict()\n"
            "assert set(sd.keys()) == {'weight', 'running_mean'}, (\n"
            "    f'expected both weight + running_mean in state_dict, got {set(sd.keys())}'\n"
            ")\n"
            "\n"
            "# Forward + EMA update in training mode.\n"
            "mod.train()\n"
            "x = t.tensor([[2.0, 4.0, 6.0],\n"
            "              [4.0, 8.0, 12.0]])  # batch mean = [3, 6, 9]\n"
            "y = mod(x)\n"
            "expected_running = 0.9 * t.zeros(3) + 0.1 * t.tensor([3.0, 6.0, 9.0])\n"
            "assert t.allclose(mod.running_mean, expected_running), (\n"
            "    f'EMA wrong: got {mod.running_mean}, expected {expected_running}'\n"
            ")\n"
            "\n"
            "# In eval mode running_mean should NOT update.\n"
            "mod.eval()\n"
            "snapshot = mod.running_mean.clone()\n"
            "_ = mod(t.tensor([[100.0, 100.0, 100.0]]))\n"
            "assert t.allclose(mod.running_mean, snapshot), (\n"
            "    'running_mean changed during eval mode — did you guard the update with self.training?'\n"
            ")\n"
            "\n"
            "# requires_grad — Parameter yes, buffer no.\n"
            "assert mod.weight.requires_grad, 'Parameter should require grad'\n"
            "assert not mod.running_mean.requires_grad, 'buffer should NOT require grad'\n"
            "\n"
            "# Optimizer sees only weight (1 param), NOT running_mean.\n"
            "opt = t.optim.SGD(mod.parameters(), lr=0.1)\n"
            "assert len(opt.param_groups[0]['params']) == 1, (\n"
            "    'optimizer should see exactly 1 param (weight); running_mean must NOT be there'\n"
            ")"
        ),
        "solution_body": (
            "def ex2_build_running_scaler(num_features: int):\n"
            "    class RunningScaler(t.nn.Module):\n"
            "        def __init__(self, num_features):\n"
            "            super().__init__()\n"
            "            self.weight = t.nn.Parameter(t.ones(num_features))\n"
            "            self.register_buffer('running_mean', t.zeros(num_features))\n"
            "        def forward(self, x: Tensor) -> Tensor:\n"
            "            if self.training:\n"
            "                with t.no_grad():\n"
            "                    batch_mean = x.mean(dim=0)\n"
            "                    self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean\n"
            "            return (x - self.running_mean) * self.weight\n"
            "    return RunningScaler(num_features)"
        ),
        "solution_notes": (
            "**Three registries, not two.** `nn.Module` separates state into "
            "three internal dicts:\n"
            "- `_parameters` — learnable, in `.parameters()`, in `state_dict()`, "
            "`requires_grad=True` by default.\n"
            "- `_buffers` — non-learnable, NOT in `.parameters()`, in "
            "`state_dict()`, `requires_grad=False`.\n"
            "- `_modules` — child modules, recursively contribute their own "
            "parameters + buffers.\n\n"
            "**Why buffers and not raw tensors.** Buffers move with `.cuda()`, "
            "round-trip via `state_dict()`, get included in `.to(device)`. A "
            "raw tensor attribute survives none of that.\n\n"
            "**Why the `with t.no_grad():` guard.** Without it, the EMA "
            "update `0.9 * self.running_mean + 0.1 * batch_mean` would build "
            "an autograd graph through every training step, eventually OOMing "
            "or breaking `loss.backward()`. Buffers are non-learnable; their "
            "updates must not participate in autograd."
        ),
    },
    # ============================================================ module-composition / ex1
    {
        "atom_id": "module-composition",
        "subtopic": "PyTorch: Module composition",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_COMPOSITION,
        "exercise_index": 1,
        "exercise_title": "child Modules auto-register as attributes",
        "slug": "child-modules-auto-register",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["children", "named_modules", "composition", "attribute-registration"],
        "kcs": ["child-module-attribute-registration"],
        "lo": (
            "Compose a parent Module from two named child Modules via "
            "attribute assignment, and verify the parent's parameters() "
            "transitively includes the children's parameters."
        ),
        "prompt_body": (
            "Implement `TwoLayerMLP` — the simplest possible module-"
            "composition pattern. Subclass `t.nn.Module` and in `__init__("
            "self, in_features, hidden_features, out_features)`:\n\n"
            "1. Call `super().__init__()` first.\n"
            "2. Assign two child Modules as named attributes:\n"
            "   - `self.fc1 = t.nn.Linear(in_features, hidden_features)`\n"
            "   - `self.fc2 = t.nn.Linear(hidden_features, out_features)`\n"
            "3. `forward(self, x: Tensor) -> Tensor` computes "
            "`fc2(relu(fc1(x)))` using `t.relu` between the two linears.\n\n"
            "Return an instance from `ex1_build_two_layer_mlp(in_features, "
            "hidden_features, out_features)`.\n\n"
            "The test confirms: (a) both child Linears show up in "
            "`.children()` and `.named_modules()`, (b) the parent's "
            "`.parameters()` recursively contains all 4 tensors "
            "(fc1.weight, fc1.bias, fc2.weight, fc2.bias), (c) the forward "
            "pass produces the correct shape, (d) the children are "
            "auto-named after the attributes you assigned them to."
        ),
        "stub": (
            "def ex1_build_two_layer_mlp(in_features: int, hidden_features: int, out_features: int):\n"
            '    """Return a TwoLayerMLP instance with fc1 + relu + fc2."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "mod = ex1_build_two_layer_mlp(in_features=4, hidden_features=8, out_features=3)\n"
            "assert isinstance(mod, t.nn.Module)\n"
            "\n"
            "# Both child Linears must appear in .children().\n"
            "children = list(mod.children())\n"
            "assert len(children) == 2, f'expected 2 child modules, got {len(children)}'\n"
            "assert all(isinstance(c, t.nn.Linear) for c in children), (\n"
            "    f'expected both children to be Linear, got {[type(c).__name__ for c in children]}'\n"
            ")\n"
            "\n"
            "# named_children should match the attribute names exactly.\n"
            "named = dict(mod.named_children())\n"
            "assert set(named.keys()) == {'fc1', 'fc2'}, (\n"
            "    f'expected children named fc1+fc2, got {set(named.keys())} '\n"
            "    f'(child names come from the attribute names you assigned them to)'\n"
            ")\n"
            "\n"
            "# Recursion: parent's parameters() must include all 4 tensors.\n"
            "params = list(mod.parameters())\n"
            "assert len(params) == 4, (\n"
            "    f'expected 4 parameters (fc1.weight, fc1.bias, fc2.weight, fc2.bias), '\n"
            "    f'got {len(params)}'\n"
            ")\n"
            "named_params = dict(mod.named_parameters())\n"
            "assert set(named_params.keys()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}, (\n"
            "    f'expected dot-prefixed names, got {set(named_params.keys())}'\n"
            ")\n"
            "assert named_params['fc1.weight'].shape == (8, 4)\n"
            "assert named_params['fc2.weight'].shape == (3, 8)\n"
            "\n"
            "# Forward — shape correctness.\n"
            "x = t.randn(5, 4, generator=t.Generator().manual_seed(0))  # batch of 5\n"
            "y = mod(x)\n"
            "assert y.shape == (5, 3), f'expected (5, 3), got {tuple(y.shape)}'\n"
            "\n"
            "# Sanity: the ReLU between linears must be in the path.\n"
            "# Feed a tensor where fc1 outputs all-negative — fc2 sees all zeros if ReLU is wired.\n"
            "# Override fc1 weights so fc1(x) is guaranteed negative.\n"
            "with t.no_grad():\n"
            "    mod.fc1.weight.fill_(-1.0)\n"
            "    mod.fc1.bias.fill_(-1.0)\n"
            "    mod.fc2.weight.fill_(1.0)\n"
            "    mod.fc2.bias.fill_(0.0)\n"
            "y_neg = mod(t.ones(1, 4))\n"
            "# fc1(ones) is all -5 → ReLU zeros them → fc2(zeros) is all bias (=0).\n"
            "assert t.allclose(y_neg, t.zeros(1, 3)), (\n"
            "    f'expected ReLU to zero out negatives → fc2(zeros)=0, got {y_neg} '\n"
            "    f'(did you forget to apply relu between fc1 and fc2?)'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_build_two_layer_mlp(in_features: int, hidden_features: int, out_features: int):\n"
            "    class TwoLayerMLP(t.nn.Module):\n"
            "        def __init__(self, in_features, hidden_features, out_features):\n"
            "            super().__init__()\n"
            "            self.fc1 = t.nn.Linear(in_features, hidden_features)\n"
            "            self.fc2 = t.nn.Linear(hidden_features, out_features)\n"
            "        def forward(self, x: Tensor) -> Tensor:\n"
            "            return self.fc2(t.relu(self.fc1(x)))\n"
            "    return TwoLayerMLP(in_features, hidden_features, out_features)"
        ),
        "solution_notes": (
            "**The auto-registration mechanism.** `nn.Module.__setattr__` "
            "checks the value of every attribute assignment. If it's an "
            "`nn.Module`, it goes into `self._modules['fc1']` (named after "
            "the attribute). That's why `named_children()` returns the names "
            "you used in your code, and `named_parameters()` returns "
            "dot-prefixed paths like `fc1.weight`.\n\n"
            "**The forgotten-relu bug.** A common slip is "
            "`self.fc2(self.fc1(x))` — no nonlinearity between linears, "
            "making the whole stack equivalent to a single linear "
            "transformation. The test's negative-fc1 trick catches exactly "
            "this: with all-negative fc1 outputs, ReLU should zero them and "
            "fc2 produces all-bias; without ReLU, fc2 would produce a "
            "non-zero value.\n\n"
            "**Why `nn.ModuleList` not `list`.** If you'd written "
            "`self.layers = [Linear(...), Linear(...)]`, the children "
            "wouldn't register — `mod.parameters()` would be empty and "
            "training silently wouldn't update the layers."
        ),
    },
    # ============================================================ module-composition / ex2
    {
        "atom_id": "module-composition",
        "subtopic": "PyTorch: Module composition",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_COMPOSITION,
        "exercise_index": 2,
        "exercise_title": "rebuild the MLP with nn.Sequential",
        "slug": "rebuild-mlp-with-sequential",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["nn.Sequential", "pipeline", "composition", "state_dict-parity"],
        "kcs": ["sequential-pipeline", "child-module-attribute-registration"],
        "lo": (
            "Build the same two-layer MLP using nn.Sequential as a single "
            "self.net attribute, and confirm forward-pass parity with the "
            "named-attribute version from the previous drill."
        ),
        "prompt_body": (
            "Implement `SequentialMLP` — the exact same two-layer MLP as the "
            "previous exercise but composed via `nn.Sequential` instead of "
            "named attributes. In `__init__(self, in_features, "
            "hidden_features, out_features)`:\n\n"
            "1. Call `super().__init__()`.\n"
            "2. Build `self.net = t.nn.Sequential(linear1, relu, linear2)` "
            "where:\n"
            "   - `linear1 = t.nn.Linear(in_features, hidden_features)`\n"
            "   - `relu = t.nn.ReLU()`\n"
            "   - `linear2 = t.nn.Linear(hidden_features, out_features)`\n"
            "3. `forward(self, x: Tensor) -> Tensor` just returns `self.net(x)`.\n\n"
            "Return an instance from `ex2_build_sequential_mlp(in_features, "
            "hidden_features, out_features)`.\n\n"
            "The big idea: `Sequential` is itself a Module, so the whole "
            "pipeline lives behind a single child attribute named `net`. "
            "Compared to the named-attribute version (`fc1` + `fc2`), the "
            "names of the children change (`net.0.weight` not `fc1.weight`) "
            "but the parameter count and forward output match exactly.\n\n"
            "The test asserts: (a) `self.net` is an `nn.Sequential` of length "
            "3, (b) parameter names are now `net.0.*` / `net.2.*`, (c) "
            "forward output matches the equivalent named-attribute MLP when "
            "their parameters are copied across."
        ),
        "stub": (
            "def ex2_build_sequential_mlp(in_features: int, hidden_features: int, out_features: int):\n"
            '    """Return a SequentialMLP instance — same forward as TwoLayerMLP but composed via nn.Sequential."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "mod = ex2_build_sequential_mlp(in_features=4, hidden_features=8, out_features=3)\n"
            "assert isinstance(mod, t.nn.Module)\n"
            "\n"
            "# Top-level child should be exactly one — the Sequential named 'net'.\n"
            "named_children = dict(mod.named_children())\n"
            "assert set(named_children.keys()) == {'net'}, (\n"
            "    f'expected exactly one top-level child named \"net\", got {set(named_children.keys())}'\n"
            ")\n"
            "assert isinstance(named_children['net'], t.nn.Sequential), (\n"
            "    f'expected nn.Sequential, got {type(named_children[\"net\"]).__name__}'\n"
            ")\n"
            "assert len(named_children['net']) == 3, (\n"
            "    f'Sequential should contain 3 modules (Linear, ReLU, Linear), got {len(named_children[\"net\"])}'\n"
            ")\n"
            "\n"
            "# Element types inside the Sequential.\n"
            "seq = named_children['net']\n"
            "assert isinstance(seq[0], t.nn.Linear)\n"
            "assert isinstance(seq[1], t.nn.ReLU)\n"
            "assert isinstance(seq[2], t.nn.Linear)\n"
            "assert seq[0].in_features == 4 and seq[0].out_features == 8\n"
            "assert seq[2].in_features == 8 and seq[2].out_features == 3\n"
            "\n"
            "# Param names: Sequential children are auto-named by integer index → 'net.0.weight' etc.\n"
            "named_params = dict(mod.named_parameters())\n"
            "expected_names = {'net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias'}\n"
            "assert set(named_params.keys()) == expected_names, (\n"
            "    f'expected {expected_names}, got {set(named_params.keys())}'\n"
            ")\n"
            "\n"
            "# Forward shape check.\n"
            "x = t.randn(5, 4, generator=t.Generator().manual_seed(7))\n"
            "y = mod(x)\n"
            "assert y.shape == (5, 3), f'expected (5, 3), got {tuple(y.shape)}'\n"
            "\n"
            "# Parity check: build a reference Sequential by hand and copy params over,\n"
            "# then verify outputs agree.\n"
            "reference = t.nn.Sequential(\n"
            "    t.nn.Linear(4, 8),\n"
            "    t.nn.ReLU(),\n"
            "    t.nn.Linear(8, 3),\n"
            ")\n"
            "with t.no_grad():\n"
            "    reference[0].weight.copy_(seq[0].weight)\n"
            "    reference[0].bias.copy_(seq[0].bias)\n"
            "    reference[2].weight.copy_(seq[2].weight)\n"
            "    reference[2].bias.copy_(seq[2].bias)\n"
            "y_ref = reference(x)\n"
            "assert t.allclose(y, y_ref, atol=1e-6), (\n"
            "    f'forward mismatch vs reference Sequential — your forward should be self.net(x)'\n"
            ")"
        ),
        "solution_body": (
            "def ex2_build_sequential_mlp(in_features: int, hidden_features: int, out_features: int):\n"
            "    class SequentialMLP(t.nn.Module):\n"
            "        def __init__(self, in_features, hidden_features, out_features):\n"
            "            super().__init__()\n"
            "            self.net = t.nn.Sequential(\n"
            "                t.nn.Linear(in_features, hidden_features),\n"
            "                t.nn.ReLU(),\n"
            "                t.nn.Linear(hidden_features, out_features),\n"
            "            )\n"
            "        def forward(self, x: Tensor) -> Tensor:\n"
            "            return self.net(x)\n"
            "    return SequentialMLP(in_features, hidden_features, out_features)"
        ),
        "solution_notes": (
            "**Sequential is just a Module that auto-pipes.** "
            "`Sequential(A, B, C)(x)` is exactly `C(B(A(x)))`. It stores its "
            "children in `_modules` keyed by string index ('0', '1', '2'), "
            "which is why `named_parameters()` produces `net.0.weight` etc.\n\n"
            "**When to use Sequential vs named attributes.** Sequential is "
            "cleanest when (a) the forward is a strict left-to-right pipeline "
            "and (b) you never want to introspect a specific child by a "
            "meaningful name. The moment your forward branches (residual "
            "block, attention, gating) or you want `self.encoder` / "
            "`self.decoder` to be separately introspectable, switch back to "
            "named attributes.\n\n"
            "**ARENA's actual `make_cnn`** uses Sequential as the outer "
            "container with a long Conv2d → BN → ReLU → ... stack. Knowing "
            "this composition style is non-negotiable for that exercise."
        ),
    },
    # ============================================================ module-extra-repr / ex1
    {
        "atom_id": "module-extra-repr",
        "subtopic": "PyTorch: Module __repr__",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EXTRA_REPR,
        "exercise_index": 1,
        "exercise_title": "extra_repr for a Linear-style module",
        "slug": "extra-repr-for-linear-style-module",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["extra_repr", "__repr__", "debugging", "introspection"],
        "kcs": ["extra-repr-returns-string", "extra-repr-print-bool-not-tensor"],
        "lo": (
            "Override extra_repr on an nn.Module subclass so that print(mod) "
            "displays in_features, out_features, and a bias-presence boolean "
            "(NOT the bias tensor)."
        ),
        "prompt_body": (
            "Implement `MyReprLinear` — a Linear-style Module whose ONLY "
            "interesting feature is its `extra_repr`. In `__init__(self, "
            "in_features, out_features, bias=True)`:\n\n"
            "1. Call `super().__init__()` first.\n"
            "2. Store `self.in_features = in_features`, "
            "`self.out_features = out_features`.\n"
            "3. Store `self.weight = nn.Parameter(t.zeros(out_features, "
            "in_features))` (zeros — initialization is not the point here).\n"
            "4. If `bias` is True: store `self.bias = nn.Parameter(t.zeros("
            "out_features))`. Otherwise: `self.bias = None`.\n"
            "5. Implement `forward(self, x): return x @ self.weight.T + ("
            "self.bias if self.bias is not None else 0)`.\n"
            "6. **Implement `extra_repr(self)`** returning the string "
            "`f'in_features={self.in_features}, out_features={self."
            "out_features}, bias={self.bias is not None}'`.\n\n"
            "**Critical:** the `bias=...` part must print the BOOLEAN "
            "`True` / `False` — NOT the bias tensor (`bias={self.bias}` would "
            "dump the whole tensor into the repr, which is ARENA's explicit "
            "gotcha for this atom).\n\n"
            "Return an instance from `ex1_build_repr_linear(in_features, "
            "out_features, bias)`.\n\n"
            "The test verifies that `repr(mod)` contains the right substrings "
            "and does NOT contain the bias tensor representation."
        ),
        "stub": (
            "def ex1_build_repr_linear(in_features: int, out_features: int, bias: bool = True):\n"
            '    """Return a MyReprLinear instance with a properly-implemented extra_repr."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Case 1 — bias=True.\n"
            "mod_with_bias = ex1_build_repr_linear(in_features=3, out_features=5, bias=True)\n"
            "assert isinstance(mod_with_bias, t.nn.Module)\n"
            "r = repr(mod_with_bias)\n"
            "# Must contain the three key=value fragments.\n"
            "assert 'in_features=3' in r, f'expected in_features=3 in repr, got: {r}'\n"
            "assert 'out_features=5' in r, f'expected out_features=5 in repr, got: {r}'\n"
            "assert 'bias=True' in r, f'expected bias=True in repr, got: {r}'\n"
            "# Class name should be the wrapping one (default nn.Module __repr__ shows it).\n"
            "assert 'MyReprLinear' in r or type(mod_with_bias).__name__ in r, (\n"
            "    f'class name should be in repr, got: {r}'\n"
            ")\n"
            "# CRITICAL — bias tensor should NOT be dumped into the repr.\n"
            "# Tensor reprs always contain 'tensor(' — assert that substring is absent.\n"
            "# (Allow it inside Parameter sub-reprs if any; this only checks what extra_repr emits.)\n"
            "# We can isolate extra_repr's output directly:\n"
            "extra = mod_with_bias.extra_repr()\n"
            "assert isinstance(extra, str), f'extra_repr must return a str, got {type(extra).__name__}'\n"
            "assert 'tensor(' not in extra, (\n"
            "    f'extra_repr should print bias as a bool (True/False), not the tensor itself. Got: {extra!r}'\n"
            ")\n"
            "assert 'bias=True' in extra, f'extra_repr should contain bias=True, got: {extra!r}'\n"
            "\n"
            "# Case 2 — bias=False.\n"
            "mod_no_bias = ex1_build_repr_linear(in_features=4, out_features=2, bias=False)\n"
            "r2 = repr(mod_no_bias)\n"
            "assert 'in_features=4' in r2\n"
            "assert 'out_features=2' in r2\n"
            "assert 'bias=False' in r2, f'expected bias=False (not None, not absent), got: {r2}'\n"
            "assert mod_no_bias.bias is None, 'when bias=False, self.bias should be None'\n"
            "\n"
            "# Forward still works in both cases.\n"
            "x = t.randn(7, 3, generator=t.Generator().manual_seed(0))\n"
            "y = mod_with_bias(x)\n"
            "assert y.shape == (7, 5)\n"
            "x2 = t.randn(7, 4, generator=t.Generator().manual_seed(1))\n"
            "y2 = mod_no_bias(x2)\n"
            "assert y2.shape == (7, 2)\n"
            "\n"
            "# Sanity print of the repr so the learner sees their work.\n"
            "print(f'  with bias: {mod_with_bias}')\n"
            "print(f'  no bias:   {mod_no_bias}')"
        ),
        "solution_body": (
            "def ex1_build_repr_linear(in_features: int, out_features: int, bias: bool = True):\n"
            "    class MyReprLinear(t.nn.Module):\n"
            "        def __init__(self, in_features, out_features, bias=True):\n"
            "            super().__init__()\n"
            "            self.in_features = in_features\n"
            "            self.out_features = out_features\n"
            "            self.weight = t.nn.Parameter(t.zeros(out_features, in_features))\n"
            "            if bias:\n"
            "                self.bias = t.nn.Parameter(t.zeros(out_features))\n"
            "            else:\n"
            "                self.bias = None\n"
            "        def forward(self, x):\n"
            "            return x @ self.weight.T + (self.bias if self.bias is not None else 0)\n"
            "        def extra_repr(self):\n"
            "            return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'\n"
            "    return MyReprLinear(in_features, out_features, bias)"
        ),
        "solution_notes": (
            "**Where extra_repr fits in the print output.** "
            "`nn.Module.__repr__` produces `ClassName(<extra_repr output>)` "
            "for leaf modules, or wraps the children's reprs inside for "
            "container modules. Your `extra_repr` only controls what goes "
            "between the parens — class name and children are handled for "
            "you.\n\n"
            "**The bias-tensor-dump bug.** ARENA flags this explicitly. "
            "`f'bias={self.bias}'` formats the bias tensor via its own "
            "`__repr__`, producing output like "
            "`bias=Parameter containing: tensor([0., 0., 0., 0., 0.], "
            "requires_grad=True)` — useless for debugging Module shape "
            "issues. Always print the BOOLEAN presence flag (`self.bias is "
            "not None`), matching what `nn.Linear` itself does.\n\n"
            "**Why `is not None` not `bool(self.bias)`.** Calling "
            "`bool(tensor)` on a multi-element tensor raises "
            "`RuntimeError: Boolean value of Tensor with more than one "
            "element is ambiguous`. The `is not None` check is the safe "
            "idiom."
        ),
    },
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
