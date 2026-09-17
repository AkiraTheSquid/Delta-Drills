#!/usr/bin/env python3
"""Author 8 deepening standalones for single-drill ARENA atoms (batch E).

Each atom already has 2-3 existing notebooks. This batch adds ONE more
exercise per atom that probes a DISTINCT facet not covered by any prior
exercise in that folder. PS4 framing — one LO, one Bloom, max 2 KCs.

Subtopic key matches ex1 verbatim so the registry stays single-source.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ============================================================ ATOM 1 / 8
# nn-module-subclass — prereqs_pytorch_modules
# ex1: stateless Module + dunder-call            ex2: super().__init__() bug
# ex3: Linear from scratch (init+super+forward+matmul)
# ex4 NEW facet: self.training flag drives forward branching
#                (Dropout-style: differs train vs eval). No prior exercise
#                touches the .train()/.eval() mode toggle.
SPEC_NN_MODULE_SUBCLASS = {
    "atom_id": "nn-module-subclass",
    "subtopic": "PyTorch: nn.Module subclassing",
    "topic_folder": "prereqs_pytorch_modules",
    "atom_recap_md": (
        "## nn.Module subclassing — quick refresher\n"
        "\n"
        "An `nn.Module` subclass is just a Python class with three contracts:\n"
        "1. Call `super().__init__()` first when you have state.\n"
        "2. Implement `forward(self, ...)` — never call `.forward()` directly; "
        "use `module(x)`, which routes through `__call__`.\n"
        "3. The base class also tracks `self.training` (default `True`). "
        "`.train()` and `.eval()` flip it for the module AND all children.\n"
        "\n"
        "**This drill (ex4) vs prior exercises.** ex1 built a stateless square "
        "layer; ex2 diagnosed a missing `super().__init__()`; ex3 wrote Linear "
        "from scratch. None of them touched the `train`/`eval` mode toggle — "
        "the most common reason a `forward` reads `self.<something>` and "
        "branches at runtime. Dropout, BatchNorm, and inference-only paths all "
        "key off `self.training`."
    ),
    "exercise_index": 4,
    "exercise_title": "Dropout-style Module that branches on self.training",
    "slug": "dropout-style-module-branches-on-self-training",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["self.training", "train-eval-toggle", "dropout-style", "forward-branching"],
    "kcs": ["module-self-training-flag", "module-subclass-trivial-forward"],
    "lo": (
        "Apply the `self.training` flag inside `forward` to branch a Module "
        "between a train-time scaling path and an eval-time identity path, "
        "exactly as Dropout does."
    ),
    "prompt_body": (
        "Implement `ScalingDropoutLike` — a Module whose forward pass depends "
        "on the `self.training` flag set by `.train()`/`.eval()`.\n\n"
        "1. Subclass `t.nn.Module`. In `__init__(self, scale)` call "
        "`super().__init__()` first; store `self.scale = float(scale)` (a "
        "plain Python float, NOT a Parameter — this drill isolates the mode-"
        "branching mechanic).\n"
        "2. In `forward(self, x: Tensor) -> Tensor`:\n"
        "   - If `self.training` is `True`, return `x * self.scale` (train "
        "path — analogous to Dropout's inverse-scaling).\n"
        "   - Otherwise, return `x` unchanged (eval / inference path).\n"
        "3. Return an INSTANCE from "
        "`ex4_build_scaling_dropout_like(scale)`.\n\n"
        "The test toggles `.train()` and `.eval()` and asserts the forward "
        "output flips between the two paths. It also asserts that "
        "`.eval()` on the parent flips `self.training` to `False` "
        "(propagation through `nn.Module.train(mode=False)`).\n\n"
        "**No __call__ tricks needed.** `nn.Module.__call__` already routes "
        "to `forward` AND respects the `training` flag for you."
    ),
    "stub": (
        "def ex4_build_scaling_dropout_like(scale: float) -> 't.nn.Module':\n"
        '    """Return a Module whose forward branches on self.training."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "mod = ex4_build_scaling_dropout_like(0.5)\n"
        "assert isinstance(mod, t.nn.Module), f'expected nn.Module, got {type(mod).__name__}'\n"
        "\n"
        "# Default state: training=True (nn.Module default).\n"
        "assert mod.training is True, f'expected training=True at construction, got {mod.training}'\n"
        "\n"
        "x = t.tensor([1.0, 2.0, 4.0, 8.0])\n"
        "\n"
        "# Train path: scaled by 0.5.\n"
        "mod.train()\n"
        "assert mod.training is True\n"
        "out_train = mod(x)\n"
        "expected_train = x * 0.5\n"
        "assert t.allclose(out_train, expected_train), (\n"
        "    f'train path: got {out_train}, expected {expected_train}'\n"
        ")\n"
        "\n"
        "# Eval path: identity.\n"
        "mod.eval()\n"
        "assert mod.training is False, f'.eval() must flip training to False, got {mod.training}'\n"
        "out_eval = mod(x)\n"
        "assert t.allclose(out_eval, x), f'eval path: got {out_eval}, expected identity {x}'\n"
        "\n"
        "# Back to train.\n"
        "mod.train()\n"
        "assert mod.training is True\n"
        "out_train2 = mod(x)\n"
        "assert t.allclose(out_train2, expected_train), 'second .train() should re-enable scaling'\n"
        "\n"
        "# Different scale → different train output (sanity).\n"
        "mod2 = ex4_build_scaling_dropout_like(3.0)\n"
        "mod2.train()\n"
        "assert t.allclose(mod2(x), x * 3.0)\n"
        "mod2.eval()\n"
        "assert t.allclose(mod2(x), x)"
    ),
    "solution_body": (
        "class ScalingDropoutLike(t.nn.Module):\n"
        "    def __init__(self, scale):\n"
        "        super().__init__()\n"
        "        self.scale = float(scale)\n"
        "    def forward(self, x):\n"
        "        if self.training:\n"
        "            return x * self.scale\n"
        "        return x\n"
        "\n"
        "def ex4_build_scaling_dropout_like(scale):\n"
        "    return ScalingDropoutLike(scale)"
    ),
    "solution_notes": (
        "**`self.training` is inherited.** `nn.Module.__init__` sets "
        "`self.training = True` for you — that's why a fresh Module is in "
        "train mode by default. `.train()` and `.eval()` are also inherited "
        "methods that flip the flag recursively across children.\n\n"
        "**Why this branching pattern is everywhere.** Dropout multiplies by "
        "`1/(1-p)` during training and is the identity at eval. BatchNorm "
        "uses the running stats at eval but per-batch stats during training. "
        "Any layer that should behave differently for evaluation reads "
        "`self.training` inside `forward`.\n\n"
        "**Common pitfall.** Forgetting `model.eval()` before a validation "
        "loop silently keeps Dropout / BatchNorm in training mode — the "
        "classic source of \"my model performs worse at inference\" bugs."
    ),
}

# ============================================================ ATOM 2 / 8
# nn-parameter-wrap — prereqs_pytorch_modules
# ex1: Parameter vs raw tensor visibility    ex2: Parameter vs buffer
# ex3 NEW facet: requires_grad=False on a Parameter → frozen weight,
#                still in .parameters() / state_dict but excluded from
#                optimizer updates. None of the prior exercises touch
#                Parameter-with-grad-disabled (the freezing pattern).
SPEC_NN_PARAMETER_WRAP = {
    "atom_id": "nn-parameter-wrap",
    "subtopic": "PyTorch: nn.Parameter",
    "topic_folder": "prereqs_pytorch_modules",
    "atom_recap_md": (
        "## nn.Parameter — quick refresher\n"
        "\n"
        "`nn.Parameter` is a tensor subclass that auto-registers when assigned "
        "as a Module attribute. Two independent flags govern its behavior:\n"
        "1. **Visibility** — whether it shows up in `.parameters()` and "
        "`.state_dict()`. Controlled by wrapping in `nn.Parameter`.\n"
        "2. **Gradient tracking** — whether autograd records ops on it. "
        "Controlled by `requires_grad` (default `True` for Parameters).\n"
        "\n"
        "**This drill (ex3) vs prior.** ex1 contrasted Parameter vs raw "
        "tensor (visibility); ex2 contrasted Parameter vs buffer (the buffer "
        "is the alternative *registry*). ex3 holds Parameter constant and "
        "varies `requires_grad` — the canonical **frozen-weight** pattern "
        "(transfer learning, LoRA base weights). A frozen Parameter is still "
        "in the state dict but the optimizer sees no grad."
    ),
    "exercise_index": 3,
    "exercise_title": "frozen Parameter — requires_grad=False keeps it visible but un-trainable",
    "slug": "frozen-parameter-requires-grad-false-visible-but-untrainable",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["frozen-weight", "requires-grad", "transfer-learning", "freeze"],
    "kcs": ["parameter-wrap-tensor", "parameter-requires-grad-flag"],
    "lo": (
        "Apply `nn.Parameter(tensor, requires_grad=False)` to register a "
        "frozen weight on a Module, then verify it appears in "
        "`.parameters()` and `.state_dict()` but its `.grad` stays `None` "
        "after a backward pass."
    ),
    "prompt_body": (
        "Implement `FrozenScaler` — a Module with TWO Parameters of "
        "identical shape; one trainable, one frozen.\n\n"
        "1. Subclass `t.nn.Module`. In `__init__(self)`:\n"
        "   - Call `super().__init__()` first.\n"
        "   - `self.alpha = nn.Parameter(t.ones(3))` — trainable.\n"
        "   - `self.beta  = nn.Parameter(t.ones(3), requires_grad=False)` "
        "— frozen.\n"
        "2. `forward(self, x: Tensor) -> Tensor` returns "
        "`x * self.alpha + self.beta` (both broadcast across `x`).\n"
        "3. Return an INSTANCE from `ex3_build_frozen_scaler()`.\n\n"
        "**What this drill verifies.** After one `.backward()` call:\n"
        "- `alpha.grad` is a real tensor (trainable Parameter accumulates "
        "gradient).\n"
        "- `beta.grad` stays `None` (frozen — autograd never touched it).\n"
        "- BOTH alpha and beta appear in `module.parameters()` and "
        "`module.state_dict()` (the wrap-with-nn.Parameter visibility is "
        "INDEPENDENT of requires_grad).\n\n"
        "This is exactly how transfer learning works: freeze the backbone "
        "Parameters but keep them in the state dict so they save/load with "
        "the model."
    ),
    "stub": (
        "def ex3_build_frozen_scaler() -> 't.nn.Module':\n"
        '    """Return a Module with one trainable + one frozen Parameter."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "mod = ex3_build_frozen_scaler()\n"
        "assert isinstance(mod, t.nn.Module)\n"
        "\n"
        "# Both attrs must be nn.Parameter instances.\n"
        "assert isinstance(mod.alpha, nn.Parameter), f'alpha must be nn.Parameter, got {type(mod.alpha)}'\n"
        "assert isinstance(mod.beta, nn.Parameter),  f'beta must be nn.Parameter, got {type(mod.beta)}'\n"
        "\n"
        "# requires_grad flags split correctly.\n"
        "assert mod.alpha.requires_grad is True,  f'alpha.requires_grad must be True, got {mod.alpha.requires_grad}'\n"
        "assert mod.beta.requires_grad  is False, f'beta.requires_grad must be False, got {mod.beta.requires_grad}'\n"
        "\n"
        "# Both visible in .parameters() — wrapping is sufficient.\n"
        "params = list(mod.parameters())\n"
        "assert len(params) == 2, f'expected 2 Parameters listed, got {len(params)}'\n"
        "\n"
        "# Both visible in state_dict — saving/loading still works.\n"
        "sd_keys = set(mod.state_dict().keys())\n"
        "assert 'alpha' in sd_keys, f'alpha missing from state_dict: {sd_keys}'\n"
        "assert 'beta'  in sd_keys, f'beta missing from state_dict:  {sd_keys}'\n"
        "\n"
        "# Forward + backward — alpha gets grad, beta does NOT.\n"
        "x = t.tensor([1.0, 2.0, 3.0], requires_grad=False)\n"
        "y = mod(x).sum()\n"
        "y.backward()\n"
        "assert mod.alpha.grad is not None, 'alpha.grad must be populated after backward'\n"
        "assert t.allclose(mod.alpha.grad, t.tensor([1.0, 2.0, 3.0])), (\n"
        "    f'alpha.grad wrong: {mod.alpha.grad}'\n"
        ")\n"
        "assert mod.beta.grad is None, f'beta.grad must remain None (frozen), got {mod.beta.grad}'\n"
        "\n"
        "# Optimizer filter idiom — common transfer-learning pattern.\n"
        "trainable = [p for p in mod.parameters() if p.requires_grad]\n"
        "assert len(trainable) == 1, f'expected 1 trainable Parameter, got {len(trainable)}'"
    ),
    "solution_body": (
        "import torch.nn as nn\n"
        "\n"
        "class FrozenScaler(t.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.alpha = nn.Parameter(t.ones(3))\n"
        "        self.beta  = nn.Parameter(t.ones(3), requires_grad=False)\n"
        "    def forward(self, x):\n"
        "        return x * self.alpha + self.beta\n"
        "\n"
        "def ex3_build_frozen_scaler():\n"
        "    return FrozenScaler()"
    ),
    "solution_notes": (
        "**Two orthogonal flags.** Wrapping in `nn.Parameter` controls "
        "*visibility* (parameters / state_dict). `requires_grad` controls "
        "*autograd participation*. The frozen-weight pattern combines "
        "`Parameter` + `requires_grad=False` to get visibility without "
        "gradient.\n\n"
        "**Why not just use a buffer?** A buffer is for non-learnable state "
        "that's not even *conceptually* a parameter (running BatchNorm "
        "stats, embedding lookup tables before pretraining). A frozen "
        "Parameter is one you WILL probably unfreeze later — keeping it as "
        "a Parameter signals that intent and lets you write "
        "`p.requires_grad = True` to thaw it.\n\n"
        "**Optimizer filter.** Always pass "
        "`filter(lambda p: p.requires_grad, model.parameters())` to your "
        "optimizer constructor when you have frozen Parameters — otherwise "
        "PyTorch silently ignores them at step time but still tracks them "
        "for state-management purposes (wasted memory)."
    ),
}

# ============================================================ ATOM 3 / 8
# module-composition — prereqs_pytorch_modules
# ex1: child Modules via named attributes      ex2: nn.Sequential pipeline
# ex3 NEW facet: nn.ModuleList for VARIABLE-DEPTH stack (N layers chosen
#                at construction time). Plain Python list would NOT
#                register children — this is the canonical Pythonic
#                composition pattern, distinct from named-attr (fixed)
#                and Sequential (linear no-branching).
SPEC_MODULE_COMPOSITION = {
    "atom_id": "module-composition",
    "subtopic": "PyTorch: Module composition",
    "topic_folder": "prereqs_pytorch_modules",
    "atom_recap_md": (
        "## Module composition — quick refresher\n"
        "\n"
        "Three composition patterns map to three child-registration mechanics:\n"
        "1. **Named attributes** — `self.fc1 = Linear(...)`. Fixed shape, "
        "fully manual.\n"
        "2. **nn.Sequential** — one container, strictly linear data flow.\n"
        "3. **nn.ModuleList** — Python-list-flavored container for "
        "VARIABLE-DEPTH stacks where you keep manual control over the "
        "forward pass (skip connections, gating, conditional routing).\n"
        "\n"
        "**This drill (ex3) vs prior.** ex1 used named attributes for a "
        "fixed two-layer MLP; ex2 used Sequential for the same. ex3 makes "
        "depth a constructor argument — only `ModuleList` registers a "
        "Python-list-shaped set of children. A plain `self.layers = [...]` "
        "list LOOKS like it works but the children would be invisible to "
        "`.parameters()` and `.to(device)`."
    ),
    "exercise_index": 3,
    "exercise_title": "variable-depth MLP via nn.ModuleList",
    "slug": "variable-depth-mlp-via-modulelist",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["nn.ModuleList", "variable-depth", "list-vs-modulelist", "iter-forward"],
    "kcs": ["modulelist-registers-list-of-children", "child-module-attribute-registration"],
    "lo": (
        "Apply `nn.ModuleList` to compose an N-layer MLP whose depth is a "
        "constructor argument, then iterate over the list in `forward` "
        "while keeping all children visible to `.parameters()`."
    ),
    "prompt_body": (
        "Implement `DeepMLP` — an N-layer MLP whose depth is chosen at "
        "construction time.\n\n"
        "1. Subclass `t.nn.Module`. `__init__(self, dim, num_layers)`:\n"
        "   - `super().__init__()` first.\n"
        "   - `self.layers = t.nn.ModuleList([t.nn.Linear(dim, dim) "
        "for _ in range(num_layers)])`. (Plain Python list would NOT "
        "register the Linears as children — they'd be invisible to "
        "`.parameters()`.)\n"
        "2. `forward(self, x: Tensor) -> Tensor`:\n"
        "   - Loop over `self.layers`, applying each in sequence with "
        "`t.relu` between (NOT after the last layer).\n"
        "   - Return the final activation.\n"
        "3. Return an INSTANCE from `ex3_build_deep_mlp(dim, num_layers)`.\n\n"
        "**The test contrasts ModuleList vs plain list.** It also builds a "
        "second BROKEN class `BrokenList` that stores the same Linears in a "
        "raw `[]` instead of ModuleList, and asserts its `.parameters()` "
        "iterator is empty — driving home WHY ModuleList exists."
    ),
    "stub": (
        "def ex3_build_deep_mlp(dim: int, num_layers: int) -> 't.nn.Module':\n"
        '    """Return an N-layer Linear+ReLU MLP using nn.ModuleList."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a 3-layer MLP.\n"
        "mod = ex3_build_deep_mlp(dim=8, num_layers=3)\n"
        "assert isinstance(mod, t.nn.Module)\n"
        "assert isinstance(mod.layers, nn.ModuleList), (\n"
        "    f'mod.layers must be nn.ModuleList, got {type(mod.layers).__name__}'\n"
        ")\n"
        "assert len(mod.layers) == 3, f'expected 3 layers, got {len(mod.layers)}'\n"
        "for layer in mod.layers:\n"
        "    assert isinstance(layer, nn.Linear)\n"
        "    assert layer.in_features == 8 and layer.out_features == 8\n"
        "\n"
        "# Children visible: ModuleList registers transitively.\n"
        "# Each Linear has 2 Parameters (weight, bias) → 3 layers × 2 = 6.\n"
        "params = list(mod.parameters())\n"
        "assert len(params) == 6, f'expected 6 Parameters (3 layers × 2), got {len(params)}'\n"
        "\n"
        "# Forward — preserves shape (dim → dim → dim → dim).\n"
        "x = t.randn(2, 8, generator=t.Generator().manual_seed(0))\n"
        "y = mod(x)\n"
        "assert y.shape == (2, 8), f'output shape {tuple(y.shape)} != (2, 8)'\n"
        "assert y.dtype == t.float32\n"
        "\n"
        "# Different depth → different param count.\n"
        "mod5 = ex3_build_deep_mlp(dim=4, num_layers=5)\n"
        "assert len(list(mod5.parameters())) == 10, 'depth-5 should yield 10 Parameters'\n"
        "\n"
        "# --- Contrast with the broken plain-list version ---\n"
        "class BrokenList(t.nn.Module):\n"
        "    def __init__(self, dim, num_layers):\n"
        "        super().__init__()\n"
        "        self.layers = [t.nn.Linear(dim, dim) for _ in range(num_layers)]\n"
        "    def forward(self, x):\n"
        "        for layer in self.layers:\n"
        "            x = layer(x)\n"
        "        return x\n"
        "\n"
        "broken = BrokenList(dim=8, num_layers=3)\n"
        "broken_params = list(broken.parameters())\n"
        "assert len(broken_params) == 0, (\n"
        "    f'plain list should NOT register children — got {len(broken_params)} params. '\n"
        "    'This is exactly why nn.ModuleList exists.'\n"
        ")"
    ),
    "solution_body": (
        "import torch.nn as nn\n"
        "\n"
        "class DeepMLP(t.nn.Module):\n"
        "    def __init__(self, dim, num_layers):\n"
        "        super().__init__()\n"
        "        self.layers = nn.ModuleList(\n"
        "            [nn.Linear(dim, dim) for _ in range(num_layers)]\n"
        "        )\n"
        "    def forward(self, x):\n"
        "        for i, layer in enumerate(self.layers):\n"
        "            x = layer(x)\n"
        "            if i < len(self.layers) - 1:\n"
        "                x = t.relu(x)\n"
        "        return x\n"
        "\n"
        "def ex3_build_deep_mlp(dim, num_layers):\n"
        "    return DeepMLP(dim, num_layers)"
    ),
    "solution_notes": (
        "**Why ModuleList instead of Sequential?** Sequential locks you "
        "into linear data flow — `out = layer3(layer2(layer1(x)))`. "
        "ModuleList gives you the SAME registration semantics but you write "
        "the forward yourself, which is necessary for skip connections "
        "(`x = x + layer(x)`), depth-conditional routing, weight tying, or "
        "anything that's not strictly a pipeline.\n\n"
        "**The plain-list trap.** Python's `__setattr__` magic for "
        "`nn.Module` checks whether the assigned value is a Module / "
        "Parameter / Buffer-compatible object. A raw `[Linear(), Linear()]` "
        "list passes the isinstance check for `list`, not for any of those "
        "registry-eligible types — so the children never get registered. "
        "`.parameters()` returns nothing, `.to('cuda')` doesn't move them, "
        "`.state_dict()` is empty. **Always use `nn.ModuleList` for "
        "list-of-Module patterns.**\n\n"
        "**ModuleDict.** The dict analog exists too — "
        "`self.heads = nn.ModuleDict({'cls': ..., 'reg': ...})` for "
        "name-keyed child Modules."
    ),
}

# ============================================================ ATOM 4 / 8
# backward-fn-signature — prereqs_backprop
# ex1: log_back (unary)              ex2: negative_back + exp_back (unary)
# ex3 NEW facet: BINARY back fn — multiply_back0 and multiply_back1 with
#                the EXTENDED (grad_out, out, x, y) signature AND
#                broadcasting-aware reduction. Both prior exercises stay
#                unary; this is the first time `y` enters the signature
#                AND the first time grad shape != input shape (broadcast
#                sum).
SPEC_BACKWARD_FN_SIGNATURE = {
    "atom_id": "backward-fn-signature",
    "subtopic": "Backprop: backward fn signature",
    "topic_folder": "prereqs_backprop",
    "atom_recap_md": (
        "## backward fn signature — quick refresher\n"
        "\n"
        "The ARENA back-fn convention is uniform across unary and binary "
        "ops:\n"
        "```python\n"
        "back_fn(grad_out, out, *args, **kwargs) -> grad_in\n"
        "```\n"
        "For a BINARY op like `multiply`, the `*args` slot is `(x, y)`, and "
        "you write TWO back fns — `multiply_back0` (gradient wrt `x`) and "
        "`multiply_back1` (gradient wrt `y`).\n"
        "\n"
        "**This drill (ex3) vs prior.** ex1 wrote `log_back` (unary). ex2 "
        "wrote `negative_back` + `exp_back` (both unary). ex3 is the first "
        "BINARY back-fn drill — `y` enters the signature, AND broadcasting "
        "means the raw chain-rule output can have a different shape than "
        "the input you're differentiating against, so you have to sum over "
        "the broadcast axes."
    ),
    "exercise_index": 3,
    "exercise_title": "multiply_back0 and multiply_back1 with broadcast-reduce",
    "slug": "multiply-back-binary-with-broadcast-reduce",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["multiply-back", "binary-op", "broadcast-reduce", "grad-shape-match"],
    "kcs": ["backward-fn-signature", "back-fn-broadcast-axis-sum"],
    "lo": (
        "Apply the (grad_out, out, x, y) back-fn signature to write "
        "multiply_back0 and multiply_back1 so that each grad has the same "
        "shape as its input — summing over broadcast axes when needed."
    ),
    "prompt_body": (
        "Implement TWO backward fns for `out = x * y` with the extended "
        "binary signature.\n\n"
        "**1. `multiply_back0(grad_out, out, x, y) -> grad_x`** — gradient "
        "wrt `x`.\n"
        "   - Math: `d(x*y)/dx = y`, so raw `grad_x = grad_out * y`.\n"
        "   - If `x` and `y` broadcast (e.g. `x: (3,)`, `y: (4, 3)`), then "
        "`grad_out` has shape `(4, 3)` and the raw product has shape "
        "`(4, 3)` — but `grad_x` MUST have shape `(3,)`. Sum over the "
        "broadcast axes (the leading axes of `grad_out` that aren't in "
        "`x`).\n"
        "\n"
        "**2. `multiply_back1(grad_out, out, x, y) -> grad_y`** — mirror "
        "image. Raw `grad_y = grad_out * x`; sum over the broadcast axes "
        "of `y`.\n"
        "\n"
        "**Use this helper** (provided in the test cell):\n"
        "```python\n"
        "def unbroadcast(grad, target_shape):\n"
        "    # sum extra leading dims\n"
        "    while grad.ndim > len(target_shape):\n"
        "        grad = grad.sum(dim=0)\n"
        "    # sum dims where target was size 1\n"
        "    for i, s in enumerate(target_shape):\n"
        "        if s == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "```\n"
        "\n"
        "**Return** tensors with EXACTLY `x.shape` and `y.shape` "
        "respectively. The test verifies non-broadcasted, "
        "leading-broadcasted, and size-1-broadcasted cases."
    ),
    "stub": (
        "def multiply_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """∂L/∂x for out = x * y, broadcasting-aware."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """∂L/∂y for out = x * y, broadcasting-aware."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def unbroadcast(grad, target_shape):\n"
        "    while grad.ndim > len(target_shape):\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, s in enumerate(target_shape):\n"
        "        if s == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "\n"
        "# === Case A: no broadcasting — shapes match ===\n"
        "x = t.tensor([1.0, 2.0, 3.0])\n"
        "y = t.tensor([4.0, 5.0, 6.0])\n"
        "out = x * y\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "gx = multiply_back0(grad_out, out, x, y)\n"
        "gy = multiply_back1(grad_out, out, x, y)\n"
        "assert gx.shape == x.shape, f'gx shape {tuple(gx.shape)} != {tuple(x.shape)}'\n"
        "assert gy.shape == y.shape\n"
        "assert t.allclose(gx, y), f'gx should equal y when grad_out is ones: got {gx}'\n"
        "assert t.allclose(gy, x), f'gy should equal x when grad_out is ones: got {gy}'\n"
        "\n"
        "# === Case B: x is (3,), y is (4, 3) — broadcast leading axis ===\n"
        "x = t.tensor([1.0, 2.0, 3.0])               # (3,)\n"
        "y = t.tensor([[1.0]*3, [2.0]*3, [3.0]*3, [4.0]*3])  # (4, 3)\n"
        "out = x * y                                 # (4, 3)\n"
        "grad_out = t.ones(4, 3)\n"
        "gx = multiply_back0(grad_out, out, x, y)\n"
        "gy = multiply_back1(grad_out, out, x, y)\n"
        "assert gx.shape == x.shape, f'gx broadcasted shape {tuple(gx.shape)} != {tuple(x.shape)}'\n"
        "assert gy.shape == y.shape, f'gy shape {tuple(gy.shape)} != {tuple(y.shape)}'\n"
        "# gx[i] = sum over leading axis of y → 1+2+3+4 = 10 for every i.\n"
        "assert t.allclose(gx, t.tensor([10.0, 10.0, 10.0])), f'gx mismatch: {gx}'\n"
        "# gy[k, i] = grad_out[k,i] * x[i] = x[i].\n"
        "expected_gy = x.expand_as(y).clone()\n"
        "assert t.allclose(gy, expected_gy), f'gy mismatch: {gy}'\n"
        "\n"
        "# === Case C: x is (3, 1), y is (3, 4) — broadcast size-1 axis ===\n"
        "x = t.tensor([[1.0], [2.0], [3.0]])         # (3, 1)\n"
        "y = t.tensor([[1.0, 2.0, 3.0, 4.0]] * 3)    # (3, 4)\n"
        "out = x * y                                 # (3, 4)\n"
        "grad_out = t.ones(3, 4)\n"
        "gx = multiply_back0(grad_out, out, x, y)\n"
        "gy = multiply_back1(grad_out, out, x, y)\n"
        "assert gx.shape == x.shape, f'gx shape {tuple(gx.shape)} != (3,1)'\n"
        "assert gy.shape == y.shape, f'gy shape {tuple(gy.shape)} != (3,4)'\n"
        "# gx[i, 0] = sum over the size-1-broadcasted axis of grad_out * y = sum(y[i, :])\n"
        "#         = 1+2+3+4 = 10.\n"
        "assert t.allclose(gx, t.tensor([[10.0], [10.0], [10.0]])), f'gx mismatch: {gx}'\n"
        "\n"
        "# === Cross-check vs autograd ===\n"
        "xa = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "ya = t.tensor([[1.0]*3, [2.0]*3, [3.0]*3, [4.0]*3], requires_grad=True)\n"
        "(xa * ya).sum().backward()\n"
        "gx_ref = xa.grad\n"
        "gy_ref = ya.grad\n"
        "gx_ours = multiply_back0(t.ones_like(xa.expand_as(ya).contiguous() * ya),\n"
        "                          xa * ya, xa.detach(), ya.detach())\n"
        "gy_ours = multiply_back1(t.ones_like(xa.expand_as(ya).contiguous() * ya),\n"
        "                          xa * ya, xa.detach(), ya.detach())\n"
        "assert t.allclose(gx_ours, gx_ref, atol=1e-5), f'gx vs autograd: ours={gx_ours} ref={gx_ref}'\n"
        "assert t.allclose(gy_ours, gy_ref, atol=1e-5), f'gy vs autograd: ours={gy_ours} ref={gy_ref}'"
    ),
    "solution_body": (
        "def _unbroadcast(grad, target_shape):\n"
        "    while grad.ndim > len(target_shape):\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, s in enumerate(target_shape):\n"
        "        if s == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    raw = grad_out * y\n"
        "    return _unbroadcast(raw, x.shape)\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    raw = grad_out * x\n"
        "    return _unbroadcast(raw, y.shape)"
    ),
    "solution_notes": (
        "**Why binary ops need TWO back fns.** A binary op has two input "
        "positions in the graph (argnum 0 and 1), and each needs its own "
        "gradient calculation. The ARENA `BackwardFuncLookup` keys back fns "
        "by `(forward_fn, argnum)` to keep them separate.\n\n"
        "**Why broadcast-aware reduction is essential.** PyTorch's "
        "`out = x * y` happily broadcasts `x: (3,)` with `y: (4, 3)`. The "
        "raw chain-rule output `grad_out * y` then has shape `(4, 3)` — but "
        "`grad_x` MUST have `x.shape == (3,)` for the optimizer to apply "
        "it. Summing over the broadcast axes restores the input shape. "
        "PyTorch's autograd does this internally; we have to do it "
        "explicitly.\n\n"
        "**The `_unbroadcast` helper.** Two passes: (1) sum the leading "
        "dims that don't exist on the target, (2) sum dims where the target "
        "has size 1. Order matters — leading-dim reduction first reduces "
        "ndim; the size-1 pass then matches by position."
    ),
}

# ============================================================ ATOM 5 / 8
# register-back-fn-after-wrap — prereqs_backprop
# ex1: register ONE back fn for torch.log + dispatch
# ex2: register binary op at TWO argnums (multiply_back0/1)
# ex3 NEW facet: handle MISSING registration — get_back_func raises a
#                clear, contextual KeyError when (fwd_fn, argnum) is
#                un-registered, and the test verifies the error message
#                contains both the fn name and the argnum. Neither
#                prior exercise tests the failure path; both assume the
#                lookup succeeds.
SPEC_REGISTER_BACK_FN = {
    "atom_id": "register-back-fn-after-wrap",
    "subtopic": "Backprop: register back fn",
    "topic_folder": "prereqs_backprop",
    "atom_recap_md": (
        "## register back fn — quick refresher\n"
        "\n"
        "`BackwardFuncLookup` is a dict keyed by `(fwd_fn, argnum)`. The "
        "dispatcher resolves the back fn at backward-pass time:\n"
        "```python\n"
        "back_fn = BACK_FUNCS.get_back_func(recipe.func, argnum)\n"
        "grad_in = back_fn(grad_out, out, *args, **kwargs)\n"
        "```\n"
        "**This drill (ex3) vs prior.** ex1 wired ONE entry and dispatched "
        "it. ex2 wired a binary op (two argnums). Both assume the lookup "
        "succeeds. ex3 is the FAILURE path: an un-registered op should "
        "raise a clear, debuggable `KeyError`, not a silent `None`. The "
        "error message must name both the forward fn AND the argnum — "
        "that's how you debug \"backward through unsupported op\" stacks."
    ),
    "exercise_index": 3,
    "exercise_title": "raise contextual KeyError on missing (fwd_fn, argnum)",
    "slug": "raise-contextual-keyerror-on-missing-registration",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["error-handling", "keyerror", "missing-registration", "debugging-message"],
    "kcs": ["register-back-fn-after-wrap", "lookup-missing-raises-keyerror"],
    "lo": (
        "Apply the `(forward_fn, argnum)` lookup pattern with an explicit "
        "missing-key path: `get_back_func` raises `KeyError` whose message "
        "includes both the fn `__name__` and the argnum."
    ),
    "prompt_body": (
        "Implement `BackwardFuncLookup` with an EXPLICIT missing-key path.\n\n"
        "1. `__init__`: create an internal `self._table = {}` (dict keyed "
        "by `(fwd_fn, argnum)`).\n"
        "2. `add_back_func(fwd_fn, argnum, back_fn)`: store the entry.\n"
        "3. `get_back_func(fwd_fn, argnum)`:\n"
        "   - If `(fwd_fn, argnum)` is registered, return the back fn.\n"
        "   - Otherwise raise `KeyError` whose message contains BOTH "
        "`fwd_fn.__name__` AND the string `f'argnum={argnum}'`. The test "
        "matches both substrings in the exception text.\n"
        "\n"
        "Then implement `ex3_demo_missing_lookup(BACK_FUNCS, fwd_fn, argnum)` "
        "— a small helper that calls `BACK_FUNCS.get_back_func(fwd_fn, "
        "argnum)` inside a `try / except KeyError as e` and returns the "
        "string form of the exception (so the test can substring-match).\n\n"
        "**Why this matters.** A silent `None` from `get_back_func` "
        "produces a `'NoneType' is not callable` error one stack frame "
        "later — confusing because it doesn't say WHICH op was un-"
        "registered. An explicit KeyError with the fn name and argnum "
        "saves you 5 minutes of reading tracebacks."
    ),
    "stub": (
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        raise NotImplementedError()\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        raise NotImplementedError()\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "\n"
        "def ex3_demo_missing_lookup(BACK_FUNCS, fwd_fn, argnum) -> str:\n"
        '    """Return the str(e) of the KeyError raised when lookup fails."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "BACK_FUNCS = BackwardFuncLookup()\n"
        "\n"
        "# A registered op — succeeds.\n"
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "BACK_FUNCS.add_back_func(t.log, 0, log_back)\n"
        "fn = BACK_FUNCS.get_back_func(t.log, 0)\n"
        "x = t.tensor([1.0, 2.0, 4.0])\n"
        "out = t.log(x)\n"
        "grad_out = t.ones_like(out)\n"
        "assert t.allclose(fn(grad_out, out, x), grad_out / x), 'registered lookup must work'\n"
        "\n"
        "# Missing (fwd_fn, argnum) — argnum mismatch.\n"
        "msg = ex3_demo_missing_lookup(BACK_FUNCS, t.log, 1)\n"
        "assert 'log' in msg, f'error msg must contain fn name \"log\": {msg!r}'\n"
        "assert 'argnum=1' in msg, f'error msg must contain \"argnum=1\": {msg!r}'\n"
        "\n"
        "# Missing fwd_fn entirely.\n"
        "msg2 = ex3_demo_missing_lookup(BACK_FUNCS, t.exp, 0)\n"
        "assert 'exp' in msg2, f'error msg must contain fn name \"exp\": {msg2!r}'\n"
        "assert 'argnum=0' in msg2, f'error msg must contain \"argnum=0\": {msg2!r}'\n"
        "\n"
        "# get_back_func directly raises KeyError (not just the helper).\n"
        "try:\n"
        "    BACK_FUNCS.get_back_func(t.sin, 0)\n"
        "    raise AssertionError('get_back_func must raise KeyError for missing entry')\n"
        "except KeyError as e:\n"
        "    s = str(e)\n"
        "    assert 'sin' in s and 'argnum=0' in s, f'KeyError msg lacks context: {s!r}'\n"
        "\n"
        "# Adding a second argnum for the SAME fwd_fn works (independent keys).\n"
        "def log_back_argnum1(grad_out, out, x):\n"
        "    return t.zeros_like(grad_out)  # dummy\n"
        "BACK_FUNCS.add_back_func(t.log, 1, log_back_argnum1)\n"
        "fn1 = BACK_FUNCS.get_back_func(t.log, 1)\n"
        "assert fn1 is log_back_argnum1, 'second argnum entry must be addressable independently'"
    ),
    "solution_body": (
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self._table = {}\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        self._table[(fwd_fn, argnum)] = back_fn\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        key = (fwd_fn, argnum)\n"
        "        if key not in self._table:\n"
        "            name = getattr(fwd_fn, '__name__', repr(fwd_fn))\n"
        "            raise KeyError(\n"
        "                f'no back_fn registered for {name} at argnum={argnum}'\n"
        "            )\n"
        "        return self._table[key]\n"
        "\n"
        "def ex3_demo_missing_lookup(BACK_FUNCS, fwd_fn, argnum):\n"
        "    try:\n"
        "        BACK_FUNCS.get_back_func(fwd_fn, argnum)\n"
        "        return ''  # unreachable if registration is correct for the test\n"
        "    except KeyError as e:\n"
        "        return str(e)"
    ),
    "solution_notes": (
        "**Why an explicit error matters.** ARENA's autograd dispatcher is "
        "called for every node during `backward`. If a single op isn't "
        "registered, the first symptom you see is whatever the dispatcher "
        "does with `None`. Raising a `KeyError` with the function name and "
        "argnum gives you a one-line fix path: `add_back_func(that_fn, "
        "that_argnum, ...)`.\n\n"
        "**Why include `argnum` in the message.** A binary op has TWO "
        "entries. \"no back_fn for multiply\" is ambiguous between argnum=0 "
        "and argnum=1 — the actual fix differs (which mathematical "
        "derivative to write). Spelling out `argnum=` removes the "
        "ambiguity.\n\n"
        "**`getattr(fwd_fn, '__name__', repr(fwd_fn))`.** Some forward fns "
        "are lambdas or partials with no `__name__`. Falling back to "
        "`repr` keeps the error message printable in those cases."
    ),
}

# ============================================================ ATOM 6 / 8
# wrap-forward-fn-generic — prereqs_backprop
# ex1: shell — unbox, call, box
# ex2: extend with kwargs + is_differentiable flag
# ex3 NEW facet: handle a forward fn whose output is NOT array-like
#                (returns a Python scalar / non-tensor). The wrapper must
#                pass the result through WITHOUT boxing it into a Tensor
#                — otherwise downstream code crashes calling `.shape` on
#                an `int`. Neither prior exercise tests non-Tensor output;
#                both assume the forward returns an array.
SPEC_WRAP_FORWARD_FN = {
    "atom_id": "wrap-forward-fn-generic",
    "subtopic": "Backprop: wrap forward fn",
    "topic_folder": "prereqs_backprop",
    "atom_recap_md": (
        "## wrap_forward_fn — quick refresher\n"
        "\n"
        "`wrap_forward_fn(fwd_fn)` returns a Tensor-aware wrapper. The "
        "happy path is **unbox → call → box**: pull `.array` out of "
        "Tensor inputs, invoke the raw `fwd_fn`, wrap the array result "
        "back into a `Tensor`.\n"
        "\n"
        "**This drill (ex3) vs prior.** ex1 implemented the shell — "
        "assuming `fwd_fn` always returns a tensor. ex2 added kwargs "
        "pass-through and the `is_differentiable` short-circuit. NEITHER "
        "handled the case where `fwd_fn` returns a *non-array* — a Python "
        "int, a tuple, a bool. Boxing those into a `Tensor` either crashes "
        "(`Tensor(int)` does the wrong thing) or hides a real bug. The "
        "ARENA wrapper must pass non-array outputs through UN-BOXED."
    ),
    "exercise_index": 3,
    "exercise_title": "wrap_forward_fn passes non-array outputs through un-boxed",
    "slug": "wrap-forward-fn-passes-non-array-output-through",
    "bloom_level": "Create",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["non-tensor-output", "argmax", "scalar-return", "pass-through"],
    "kcs": ["wrap-forward-fn-generic", "non-tensor-output-pass-through"],
    "lo": (
        "Create a `wrap_forward_fn` variant that boxes array-like results "
        "as Tensor but passes non-array results (Python scalars, ints, "
        "tuples) through unchanged."
    ),
    "prompt_body": (
        "Implement `wrap_forward_fn(fwd_fn)`. The wrapper closure must:\n\n"
        "1. **Unbox** — for each positional arg, if it is a `Tensor` "
        "instance (the minimal class given below), replace with "
        "`.array`. Pass everything else through unchanged.\n"
        "2. **Call** — `result = fwd_fn(*raw_args, **kwargs)`.\n"
        "3. **Conditional box** — if `result` is a `torch.Tensor` (i.e. "
        "`isinstance(result, t.Tensor)`), return `Tensor(result)`. "
        "OTHERWISE return `result` unchanged (no boxing).\n"
        "\n"
        "**Why this matters.** Some ARENA-eligible forward fns don't "
        "return arrays:\n"
        "- `lambda x: int(x.argmax().item())` → returns `int`.\n"
        "- `lambda x: x.shape` → returns `torch.Size` (a tuple subclass, "
        "not a Tensor).\n"
        "- `lambda x: x.numel()` → returns `int`.\n"
        "\n"
        "Forcing every output through `Tensor(...)` either errors "
        "(`Tensor(some_int)` doesn't have an `.array`) or produces a "
        "0-D Tensor that breaks downstream control flow expecting a real "
        "Python int.\n"
        "\n"
        "The minimal `Tensor` class is given:\n"
        "```python\n"
        "class Tensor:\n"
        "    def __init__(self, array):\n"
        "        self.array = array\n"
        "```\n"
        "(Definitions are pasted into the solution; the test relies on "
        "your wrap_forward_fn working with this exact class.)"
    ),
    "stub": (
        "class Tensor:\n"
        "    def __init__(self, array):\n"
        "        self.array = array\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn):\n"
        '    """Return a Tensor-aware wrapper that passes non-array outputs through."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Array output: must be boxed into Tensor ===\n"
        "wrapped_log = wrap_forward_fn(t.log)\n"
        "x = Tensor(t.tensor([1.0, 2.0, 4.0]))\n"
        "out = wrapped_log(x)\n"
        "assert isinstance(out, Tensor), f'array result must be boxed, got {type(out).__name__}'\n"
        "assert t.allclose(out.array, t.log(x.array)), f'array contents wrong: {out.array}'\n"
        "\n"
        "# === Python int output: must pass through un-boxed ===\n"
        "def raw_numel(arr):\n"
        "    return int(arr.numel())\n"
        "\n"
        "wrapped_numel = wrap_forward_fn(raw_numel)\n"
        "y = Tensor(t.tensor([1.0, 2.0, 4.0, 8.0, 16.0]))\n"
        "n = wrapped_numel(y)\n"
        "assert isinstance(n, int), f'int result must pass through, got {type(n).__name__}'\n"
        "assert n == 5, f'numel of 5-element tensor must be 5, got {n}'\n"
        "\n"
        "# === argmax-as-int: typical control-flow consumer ===\n"
        "def raw_argmax_int(arr):\n"
        "    return int(arr.argmax().item())\n"
        "\n"
        "wrapped_argmax = wrap_forward_fn(raw_argmax_int)\n"
        "z = Tensor(t.tensor([0.1, 0.9, 0.2, 0.7]))\n"
        "idx = wrapped_argmax(z)\n"
        "assert isinstance(idx, int) and not isinstance(idx, bool), (\n"
        "    f'argmax result must be a plain int, got {type(idx).__name__}'\n"
        ")\n"
        "assert idx == 1, f'argmax should be 1, got {idx}'\n"
        "\n"
        "# === Tuple output: must pass through un-boxed ===\n"
        "def raw_shape(arr):\n"
        "    return tuple(arr.shape)\n"
        "\n"
        "wrapped_shape = wrap_forward_fn(raw_shape)\n"
        "w = Tensor(t.randn(3, 4, 5, generator=t.Generator().manual_seed(0)))\n"
        "shp = wrapped_shape(w)\n"
        "assert isinstance(shp, tuple), f'shape result must be tuple, got {type(shp).__name__}'\n"
        "assert shp == (3, 4, 5), f'shape mismatch: {shp}'\n"
        "\n"
        "# === Plain torch tensor input (non-Tensor wrapper) passes through unbox unchanged ===\n"
        "wrapped_add = wrap_forward_fn(lambda a, b: a + b)\n"
        "out2 = wrapped_add(Tensor(t.tensor([1.0])), 2.5)  # 2.5 is a plain float\n"
        "assert isinstance(out2, Tensor)\n"
        "assert t.allclose(out2.array, t.tensor([3.5])), f'mixed inputs failed: {out2.array}'"
    ),
    "solution_body": (
        "class Tensor:\n"
        "    def __init__(self, array):\n"
        "        self.array = array\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = [a.array if isinstance(a, Tensor) else a for a in args]\n"
        "        result = fwd_fn(*raw_args, **kwargs)\n"
        "        if isinstance(result, t.Tensor):\n"
        "            return Tensor(result)\n"
        "        return result\n"
        "    return tensor_func"
    ),
    "solution_notes": (
        "**Why an `isinstance(result, t.Tensor)` check, not "
        "`hasattr(result, 'shape')`.** A `torch.Size` has `.shape`-like "
        "behavior but is a tuple, and tuples have no `.array` semantics. "
        "Type-checking on `t.Tensor` is the precise discriminator.\n\n"
        "**Why NOT box ints into 0-D Tensors.** A Python-int output from "
        "`argmax` or `numel` is usually consumed by code like "
        "`for i in range(n):` or `xs[idx]`. A 0-D Tensor doesn't satisfy "
        "those patterns without an extra `.item()` call. Pass-through "
        "preserves the original semantics.\n\n"
        "**The is_differentiable extension from ex2 still applies.** When "
        "you combine ex2 and ex3, the order of operations is: (a) unbox; "
        "(b) call fwd_fn; (c) IF result is a Tensor AND is_differentiable "
        "AND any input had requires_grad → attach a Recipe; (d) ELSE pass "
        "result through. Non-tensor outputs are non-differentiable by "
        "construction."
    ),
}

# ============================================================ ATOM 7 / 8
# inference-mode-step — prereqs_optimizer_internals
# ex1: decorate step with @t.inference_mode()
# ex2: diagnose missing decorator (the leaf-Variable error)
# ex3 NEW facet: rewrite step using `with t.no_grad():` CONTEXT MANAGER
#                instead of the decorator. Verify functional equivalence —
#                same parameter trajectory, no leaf-in-place error. Neither
#                prior exercise uses the context-manager form; both are
#                decorator-only.
SPEC_INFERENCE_MODE_STEP = {
    "atom_id": "inference-mode-step",
    "subtopic": "PyTorch: Inference mode step",
    "topic_folder": "prereqs_optimizer_internals",
    "atom_recap_md": (
        "## inference_mode for optimizer.step — quick refresher\n"
        "\n"
        "The `theta -= lr * grad` in-place update on a leaf tensor with "
        "`requires_grad=True` raises:\n"
        "```\n"
        "RuntimeError: a leaf Variable that requires grad is being used in "
        "an in-place operation.\n"
        "```\n"
        "There are TWO standard ways to silence this safely:\n"
        "1. Decorate the step method: `@t.inference_mode()`.\n"
        "2. Wrap the body in a context manager: `with t.no_grad():` "
        "(or `with t.inference_mode():`).\n"
        "\n"
        "**This drill (ex3) vs prior.** ex1 demonstrated the decorator "
        "form. ex2 diagnosed the missing-decorator error. ex3 uses the "
        "**context-manager** form — functionally equivalent but the "
        "decision point is granularity: a context manager scopes to "
        "specific lines (e.g. you want autograd ON for some pre-step "
        "logging but OFF for the actual update)."
    ),
    "exercise_index": 3,
    "exercise_title": "step body wrapped in `with t.no_grad():` (no decorator)",
    "slug": "step-wrapped-in-no-grad-context-manager",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["no_grad", "context-manager", "inference-mode", "scoped-disable"],
    "kcs": ["inference-mode-allows-leaf-in-place-mutation", "no-grad-context-manager-equivalence"],
    "lo": (
        "Apply `with t.no_grad():` inside an optimizer's `step` to "
        "achieve the same leaf-in-place permission as `@t.inference_mode()`, "
        "and verify the parameter trajectory is identical to the "
        "decorator form."
    ),
    "prompt_body": (
        "Implement `Ex3ContextSGD` — a minimal SGD optimizer whose `step` "
        "uses a `with t.no_grad():` block instead of the "
        "`@t.inference_mode()` decorator.\n\n"
        "1. `__init__(self, params, lr)`: materialize "
        "`self.params = list(params)`, store `self.lr = lr`.\n"
        "2. `step(self)`:\n"
        "   - NO decorator.\n"
        "   - Wrap the entire body in `with t.no_grad():`.\n"
        "   - Inside the block, for each `p` with non-None `.grad`, do "
        "the BARE in-place update `p -= self.lr * p.grad` (NOT "
        "`p.data -= ...`).\n"
        "3. `zero_grad(self)`: set every `p.grad = None`.\n"
        "\n"
        "**What the test verifies.** It runs your optimizer side-by-side "
        "with a reference decorator-style optimizer for 5 steps on the "
        "same model + same loss, and asserts the parameter trajectories "
        "are IDENTICAL (allclose with atol=1e-6). The point: "
        "`with t.no_grad()` and `@t.inference_mode()` produce the same "
        "behavior for this in-place update pattern.\n\n"
        "**No decorator allowed.** The test also inspects "
        "`Ex3ContextSGD.step` for the absence of `__wrapped__` (which "
        "decorators set). A decorated step would still pass the "
        "trajectory check but fails the no-decorator structural check — "
        "the LO is specifically about the context-manager form."
    ),
    "stub": (
        "class Ex3ContextSGD:\n"
        "    def __init__(self, params, lr):\n"
        "        raise NotImplementedError()\n"
        "    def step(self):\n"
        "        raise NotImplementedError()\n"
        "    def zero_grad(self):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "# === Reference: decorator-style SGD (mirrors ex1) ===\n"
        "class _RefDecoratorSGD:\n"
        "    def __init__(self, params, lr):\n"
        "        self.params = list(params)\n"
        "        self.lr = lr\n"
        "    @t.inference_mode()\n"
        "    def step(self):\n"
        "        for p in self.params:\n"
        "            if p.grad is not None:\n"
        "                p -= self.lr * p.grad\n"
        "    def zero_grad(self):\n"
        "        for p in self.params:\n"
        "            p.grad = None\n"
        "\n"
        "# === Build TWO models with identical init, run 5 steps each ===\n"
        "def _build_model_and_data(seed=0):\n"
        "    g = t.Generator().manual_seed(seed)\n"
        "    w = t.nn.Parameter(t.randn(3, 2, generator=g))\n"
        "    b = t.nn.Parameter(t.zeros(2))\n"
        "    X  = t.randn(8, 3, generator=g)\n"
        "    Y  = t.randn(8, 2, generator=g)\n"
        "    return [w, b], X, Y\n"
        "\n"
        "def _train_5_steps(opt_cls, lr=0.05):\n"
        "    params, X, Y = _build_model_and_data(seed=42)\n"
        "    opt = opt_cls(params, lr=lr)\n"
        "    history = []\n"
        "    for _ in range(5):\n"
        "        opt.zero_grad()\n"
        "        pred = X @ params[0] + params[1]\n"
        "        loss = ((pred - Y) ** 2).mean()\n"
        "        loss.backward()\n"
        "        opt.step()\n"
        "        history.append((params[0].detach().clone(), params[1].detach().clone()))\n"
        "    return history\n"
        "\n"
        "ref_traj  = _train_5_steps(_RefDecoratorSGD)\n"
        "ours_traj = _train_5_steps(Ex3ContextSGD)\n"
        "\n"
        "assert len(ref_traj) == len(ours_traj) == 5\n"
        "for i, ((rw, rb), (ow, ob)) in enumerate(zip(ref_traj, ours_traj)):\n"
        "    assert t.allclose(rw, ow, atol=1e-6), f'step {i}: weight mismatch\\n  ref={rw}\\n  ours={ow}'\n"
        "    assert t.allclose(rb, ob, atol=1e-6), f'step {i}: bias mismatch'\n"
        "\n"
        "# === Structural check: step uses context manager, NOT decorator ===\n"
        "import inspect\n"
        "step_src = inspect.getsource(Ex3ContextSGD.step)\n"
        "assert 'with t.no_grad' in step_src or 'with torch.no_grad' in step_src, (\n"
        "    f'step body must contain a `with t.no_grad():` block. Got:\\n{step_src}'\n"
        ")\n"
        "assert '@t.inference_mode' not in step_src and '@torch.inference_mode' not in step_src, (\n"
        "    f'step must NOT use the @inference_mode decorator (drill scope is context-manager). Got:\\n{step_src}'\n"
        ")\n"
        "\n"
        "# === Sanity: step doesn't raise on a fresh leaf param ===\n"
        "p = t.nn.Parameter(t.randn(4))\n"
        "p.grad = t.ones(4)\n"
        "opt = Ex3ContextSGD([p], lr=0.1)\n"
        "before = p.detach().clone()\n"
        "opt.step()  # must not raise the leaf-in-place RuntimeError\n"
        "after = p.detach().clone()\n"
        "assert t.allclose(after, before - 0.1), f'one step should move by -lr*grad: {after} vs {before - 0.1}'"
    ),
    "solution_body": (
        "class Ex3ContextSGD:\n"
        "    def __init__(self, params, lr):\n"
        "        self.params = list(params)\n"
        "        self.lr = lr\n"
        "    def step(self):\n"
        "        with t.no_grad():\n"
        "            for p in self.params:\n"
        "                if p.grad is not None:\n"
        "                    p -= self.lr * p.grad\n"
        "    def zero_grad(self):\n"
        "        for p in self.params:\n"
        "            p.grad = None"
    ),
    "solution_notes": (
        "**Decorator vs context manager — same effect, different scope.** "
        "`@t.inference_mode()` (or `@t.no_grad()`) disables autograd for "
        "the ENTIRE function body. `with t.no_grad():` disables it only "
        "for the indented block. For an optimizer.step() that's a "
        "distinction without a difference — but if you wanted to log "
        "`loss.item()` AFTER the update (with grad re-enabled for a "
        "later backward), the context manager gives you that control.\n\n"
        "**`t.no_grad()` vs `t.inference_mode()`.** Both disable autograd. "
        "`inference_mode()` is slightly faster (skips version-counter "
        "bookkeeping) and disallows view-tracking; `no_grad` is the older, "
        "more permissive form. For optimizer steps either works — most "
        "codebases use `no_grad` out of habit, ARENA's reference solution "
        "uses `inference_mode`.\n\n"
        "**The trajectory test.** Identical seeds + identical learning "
        "rate + identical loss + same in-place update mechanic ⇒ "
        "bit-identical parameter trajectories. Any divergence would "
        "indicate the two modes aren't actually equivalent for this "
        "pattern (they are)."
    ),
}

# ============================================================ ATOM 8 / 8
# optimizer-state-tensor-buffers — prereqs_optimizer_internals
# ex1: ONE buffer (zeros_like list)
# ex2: TWO buffers (RMSprop-style b + v)
# ex3 NEW facet: Adam-style THREE buffers (m + v) plus a per-param SCALAR
#                step counter `t_step` (a 0-d tensor) + the alias-bug
#                guard. Both prior exercises use `len > 0` shaped params;
#                ex3 introduces the 0-d scalar buffer case AND the
#                `id()`-based aliasing guard.
SPEC_OPTIMIZER_STATE_BUFFERS = {
    "atom_id": "optimizer-state-tensor-buffers",
    "subtopic": "Optimizer: Per-param state buffers",
    "topic_folder": "prereqs_optimizer_internals",
    "atom_recap_md": (
        "## per-param state buffers — quick refresher\n"
        "\n"
        "Per-parameter state is a list of `zeros_like(p)` tensors, one "
        "per param. Different optimizers need different buffer counts:\n"
        "- SGD-momentum: ONE buffer (`b`).\n"
        "- RMSprop: TWO buffers (`b`, `v`).\n"
        "- Adam: THREE buffers — first moment `m`, second moment `v`, "
        "and a step counter `t_step` (a 0-d scalar tensor per param OR "
        "one global int — Adam uses the global form).\n"
        "\n"
        "**This drill (ex3) vs prior.** ex1 allocated ONE buffer. ex2 "
        "allocated TWO. ex3 allocates Adam's full state: TWO per-param "
        "tensor buffers (`m`, `v`) PLUS a global Python-int step counter "
        "`self.t = 0`. The drill also tests the **alias-bug guard**: "
        "`self.m` and `self.v` must be SEPARATE lists of SEPARATE "
        "tensors. A common mistake is "
        "`self.v = self.m` (alias) — mutating one mutates the other."
    ),
    "exercise_index": 3,
    "exercise_title": "Adam-style three-state init: m, v lists + scalar step counter",
    "slug": "adam-style-three-state-init-m-v-step-counter",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["adam-init", "step-counter", "alias-guard", "three-buffers"],
    "kcs": ["state-buffer-multiple-buffers-per-optimizer", "state-buffer-no-aliasing"],
    "lo": (
        "Apply per-param `zeros_like` allocation to BOTH Adam moment "
        "buffers `m` and `v` AND initialize a separate scalar step "
        "counter `self.t = 0`, with no aliasing between `m` and `v`."
    ),
    "prompt_body": (
        "Implement `Ex3AdamInit.__init__(self, params)`. Adam's "
        "init-time skeleton:\n\n"
        "1. Materialize `self.params = list(params)`.\n"
        "2. Allocate the first-moment list: "
        "`self.m = [t.zeros_like(p) for p in self.params]`.\n"
        "3. Allocate the second-moment list: "
        "`self.v = [t.zeros_like(p) for p in self.params]`.\n"
        "4. Initialize the SCALAR step counter `self.t = 0` (Python int "
        "— Adam increments it once per `step()` call, used in the bias-"
        "correction term).\n"
        "\n"
        "**Two failure modes the test catches.**\n"
        "- **Alias bug.** If you write `self.v = self.m`, both lists "
        "share the same underlying tensors — the test asserts "
        "`id(self.m[i]) != id(self.v[i])` AND mutates `m[0]` to verify "
        "`v[0]` is unaffected.\n"
        "- **Wrong step counter type.** Adam's step counter is a Python "
        "int (or sometimes a 0-d tensor), but NOT a list and NOT None. "
        "The test asserts `isinstance(self.t, int) and self.t == 0`.\n"
        "\n"
        "**No step() body needed.** This drill isolates the init step — "
        "the same pattern as ex1 and ex2 in this folder."
    ),
    "stub": (
        "class Ex3AdamInit:\n"
        "    def __init__(self, params):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "# === Case A: mixed param shapes ===\n"
        "p1 = t.nn.Parameter(t.randn(4, 5, generator=t.Generator().manual_seed(0)))\n"
        "p2 = t.nn.Parameter(t.randn(8,    generator=t.Generator().manual_seed(1)))\n"
        "p3 = t.nn.Parameter(t.randn(2, 3, 4, generator=t.Generator().manual_seed(2)))\n"
        "\n"
        "opt = Ex3AdamInit([p1, p2, p3])\n"
        "\n"
        "# self.params materialized as list (not generator).\n"
        "assert isinstance(opt.params, list) and len(opt.params) == 3\n"
        "\n"
        "# m and v are lists of zeros_like, matching shapes + dtypes.\n"
        "for name in ('m', 'v'):\n"
        "    bufs = getattr(opt, name)\n"
        "    assert isinstance(bufs, list), f'{name} must be a list, got {type(bufs).__name__}'\n"
        "    assert len(bufs) == 3, f'{name} must have one entry per param, got {len(bufs)}'\n"
        "    for buf, p in zip(bufs, [p1, p2, p3]):\n"
        "        assert buf.shape == p.shape, f'{name}: shape {tuple(buf.shape)} != {tuple(p.shape)}'\n"
        "        assert buf.dtype == p.dtype, f'{name}: dtype {buf.dtype} != {p.dtype}'\n"
        "        assert t.all(buf == 0), f'{name}: buffer must be zero-initialized'\n"
        "        assert buf.requires_grad is False, f'{name}: buffers must not track grad'\n"
        "\n"
        "# step counter — scalar Python int, NOT tensor, NOT list.\n"
        "assert hasattr(opt, 't'), 'opt must expose a step counter as opt.t'\n"
        "assert isinstance(opt.t, int) and not isinstance(opt.t, bool), (\n"
        "    f'opt.t must be a plain Python int, got {type(opt.t).__name__}'\n"
        ")\n"
        "assert opt.t == 0, f'step counter must start at 0, got {opt.t}'\n"
        "\n"
        "# === Alias guard: m and v must be SEPARATE tensors ===\n"
        "for i in range(3):\n"
        "    assert id(opt.m[i]) != id(opt.v[i]), (\n"
        "        f'opt.m[{i}] and opt.v[{i}] are the SAME tensor — '\n"
        "        'mutating one will mutate the other. Allocate them with '\n"
        "        'separate zeros_like calls.'\n"
        "    )\n"
        "\n"
        "# Mutate m[0]; verify v[0] is untouched.\n"
        "opt.m[0].add_(1.0)\n"
        "assert t.all(opt.v[0] == 0), (\n"
        "    f'mutating m[0] altered v[0] → buffers are aliased. v[0]:\\n{opt.v[0]}'\n"
        ")\n"
        "\n"
        "# === Case B: SINGLE param edge case ===\n"
        "p_solo = t.nn.Parameter(t.zeros(7))\n"
        "opt2 = Ex3AdamInit([p_solo])\n"
        "assert len(opt2.m) == len(opt2.v) == 1\n"
        "assert opt2.m[0].shape == (7,)\n"
        "assert opt2.t == 0\n"
        "\n"
        "# === Case C: empty params edge case ===\n"
        "opt3 = Ex3AdamInit([])\n"
        "assert opt3.m == [] and opt3.v == []\n"
        "assert opt3.t == 0"
    ),
    "solution_body": (
        "class Ex3AdamInit:\n"
        "    def __init__(self, params):\n"
        "        self.params = list(params)\n"
        "        self.m = [t.zeros_like(p) for p in self.params]\n"
        "        self.v = [t.zeros_like(p) for p in self.params]\n"
        "        self.t = 0"
    ),
    "solution_notes": (
        "**Why TWO list comprehensions.** "
        "`[t.zeros_like(p) for p in self.params]` creates fresh tensors "
        "each call. If you wrote `self.v = self.m` you'd alias the "
        "lists; if you wrote `self.v = list(self.m)` you'd alias the "
        "underlying tensors (the list is new but the references "
        "inside it are shared). Two separate comprehensions are the "
        "simplest correct allocation.\n\n"
        "**Why an int step counter, not a tensor.** Adam reads `t` only "
        "to compute the bias correction `1 - beta**t`. A Python int is "
        "cheap, easy to increment with `self.t += 1`, and never has the "
        "device / dtype / requires_grad concerns that tensor counters "
        "have. (Some implementations DO use a 0-d tensor for state-dict "
        "serialization; that's a downstream concern beyond init.)\n\n"
        "**`zeros_like(p).requires_grad`.** Default is `False`, which is "
        "exactly what we want — moment buffers are STATE, not learnable "
        "parameters. Autograd should never trace ops on them."
    ),
}

# === Emit all 8 ===
ALL_SPECS = [
    SPEC_NN_MODULE_SUBCLASS,
    SPEC_NN_PARAMETER_WRAP,
    SPEC_MODULE_COMPOSITION,
    SPEC_BACKWARD_FN_SIGNATURE,
    SPEC_REGISTER_BACK_FN,
    SPEC_WRAP_FORWARD_FN,
    SPEC_INFERENCE_MODE_STEP,
    SPEC_OPTIMIZER_STATE_BUFFERS,
]

for spec in ALL_SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
