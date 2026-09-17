#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA part 4 manual-autograd-internals atoms.

Eight single-exercise standalones, under ``prereqs_autograd_internals/``:

  * chain-rule-elementwise        — ex1
  * arg-position-back-functions   — ex1
  * kwargs-pass-through-recipe    — ex1
  * recipe-dataclass              — ex1
  * parents-dict-by-argidx        — ex1
  * grad-tracking-global-toggle   — ex1
  * requires-grad-propagation     — ex1
  * unbroadcast-pattern           — ex1

Each drill exercises ONE small constituent skill of ARENA's tiny hand-written
``Tensor`` + ``Recipe`` + ``BACK_FUNCS`` autograd. Tests use plain
``torch.Tensor`` for shape/value math; we never call ``torch.autograd`` on the
hand-written ops.

Compose with batch-2 (``prereqs_backprop/``) which covers the
``(grad_out, out, *args)`` signature, the ``BACK_FUNCS`` registration step, the
``wrap_forward_fn`` factory, the per-parameter ``.grad`` access pattern, and the
in-place buffer copy.  This batch covers the internals that those wrappers
*assemble* — the chain rule itself, the per-arg back-fn split, the kwargs
threading, the Recipe dataclass, the parents dict, the global ``grad_tracking``
toggle, the ``requires_grad`` OR-propagation, and the broadcasting-aware
``unbroadcast`` helper.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_internals"


# ---------------------------------------------------------------- atom recaps

RECAP_CHAIN_RULE_ELEMENTWISE = (
    "## Elementwise chain rule — quick refresher\n"
    "\n"
    "For an elementwise forward op `out = f(x)` (so `out[i] = f(x[i])` "
    "independently per index), the local Jacobian is **diagonal**: "
    "`d(out[i]) / d(x[j])` is `f'(x[i])` when `i == j` and `0` otherwise.\n"
    "\n"
    "The chain rule then collapses to a per-position product:\n"
    "\n"
    "```\n"
    "dL/dx[i] = sum_j  dL/dout[j] * d(out[j])/d(x[i])\n"
    "         = dL/dout[i] * f'(x[i])     # only diagonal term survives\n"
    "```\n"
    "\n"
    "So elementwise backward fns are one-liners — multiply `grad_out` by the "
    "elementwise local derivative, no actual Jacobian matrix is materialized. "
    "Shape of `grad_in` always equals shape of `x`."
)

RECAP_ARG_POSITION = (
    "## Arg-position back fns — quick refresher\n"
    "\n"
    "Binary ops register **two** back fns — one per input position — because the "
    "gradient w.r.t. the left input and the gradient w.r.t. the right input are "
    "different functions when the op is asymmetric:\n"
    "\n"
    "```\n"
    "out = x / y\n"
    "d(out)/dx = 1 / y          # 'div_back0', for arg-0 (x)\n"
    "d(out)/dy = -x / y**2      # 'div_back1', for arg-1 (y)\n"
    "```\n"
    "\n"
    "Convention:\n"
    "- ``f_back0(grad_out, out, x, y)`` returns ``dL/dx``\n"
    "- ``f_back1(grad_out, out, x, y)`` returns ``dL/dy``\n"
    "\n"
    "Both take **all** original args (so they can use either one), and both "
    "return a tensor with the same shape as the input they correspond to. "
    "Symmetric ops (add, multiply) still get two registrations, even if the "
    "function bodies are identical — uniform dispatch."
)

RECAP_KWARGS_PASS_THROUGH = (
    "## Kwargs pass-through Recipe — quick refresher\n"
    "\n"
    "Some forward ops take **keyword args** that change the output (`dim`, "
    "`keepdim`, `new_shape`, ...). The autograd wrapper has to thread them in "
    "**two** places:\n"
    "\n"
    "```python\n"
    "def wrap_forward_fn(fwd_fn):\n"
    "    def tensor_func(*args, **kwargs):\n"
    "        raw = tuple(a.array if isinstance(a, Tensor) else a for a in args)\n"
    "        out_raw = fwd_fn(*raw, **kwargs)               # (1) into the call\n"
    "        out = Tensor(out_raw, requires_grad)\n"
    "        out.recipe = Recipe(fwd_fn, raw, kwargs, parents)   # (2) into Recipe\n"
    "        return out\n"
    "    return tensor_func\n"
    "```\n"
    "\n"
    "Why both? **(1)** the forward call needs the kwargs to compute the right "
    "output. **(2)** the backward fn needs the *same* kwargs at reverse time — "
    "e.g. `sum_back` needs to know which `dim` was summed so it can broadcast "
    "back. Drop them from the Recipe and reverse-pass tests start failing on "
    "shape mismatches even though the forward looks fine."
)

RECAP_RECIPE_DATACLASS = (
    "## Recipe dataclass — quick refresher\n"
    "\n"
    "Each non-leaf `Tensor` carries a `Recipe` that records exactly enough to "
    "replay the forward call in reverse. It's a 4-tuple:\n"
    "\n"
    "```python\n"
    "@dataclass(frozen=True)\n"
    "class Recipe:\n"
    "    func: Callable        # the forward fn (e.g. torch.log, torch.multiply)\n"
    "    args: tuple           # raw positional args at call time (numbers / arrays)\n"
    "    kwargs: dict          # raw keyword args at call time\n"
    "    parents: dict[int, Tensor]  # argnum -> the parent Tensor (filtered)\n"
    "```\n"
    "\n"
    "`func` is the lookup key into `BACK_FUNCS`. `args` and `kwargs` are passed "
    "into the back fn so it can compute the local Jacobian. `parents` is the "
    "edge list of the computational graph — reverse traversal walks parents to "
    "find what to differentiate next.\n"
    "\n"
    "**Always 4 fields, always in that order** — every wrap in the codebase "
    "constructs them the same way, so the reverse pass can read them generically."
)

RECAP_PARENTS_DICT = (
    "## Parents dict by argidx — quick refresher\n"
    "\n"
    "`Recipe.parents` maps **arg position → the input Tensor at that position**, "
    "skipping any non-Tensor inputs (ints, floats, shape tuples, ...):\n"
    "\n"
    "```python\n"
    "parents = {idx: a for idx, a in enumerate(args) if isinstance(a, Tensor)}\n"
    "```\n"
    "\n"
    "Two rules:\n"
    "- **Skip non-Tensors.** A `multiply(t, 3.0)` call must produce "
    "  `parents == {0: t}`, NOT `{0: t, 1: 3.0}` — gradients only flow through "
    "  Tensors. The reverse pass would crash trying to add a float to a Tensor "
    "  grad otherwise.\n"
    "- **Keep the original argnum.** The reverse pass uses the dict key to look "
    "  up the matching back fn: `BACK_FUNCS.get(func, argnum)`. Renumbering "
    "  (e.g. building a list of present Tensors) would break this lookup."
)

RECAP_GRAD_TRACKING_TOGGLE = (
    "## Grad-tracking global toggle — quick refresher\n"
    "\n"
    "A module-level boolean — `grad_tracking_enabled` — gates ALL Recipe "
    "construction. When it's `False`, every `wrap_forward_fn` short-circuits to "
    "producing a Tensor with `requires_grad=False` and no Recipe attached:\n"
    "\n"
    "```python\n"
    "grad_tracking_enabled = True\n"
    "\n"
    "def wrap_forward_fn(fwd_fn):\n"
    "    def tensor_func(*args, **kwargs):\n"
    "        ...\n"
    "        requires_grad = grad_tracking_enabled and any(\n"
    "            isinstance(a, Tensor) and a.requires_grad for a in args\n"
    "        )\n"
    "        ...\n"
    "```\n"
    "\n"
    "This is the analogue of PyTorch's `torch.no_grad()`. Use cases:\n"
    "- **Inference** — skip graph building for speed/memory.\n"
    "- **Parameter init / EMA updates** — operations that should never produce "
    "  gradients even though their inputs have `requires_grad=True`.\n"
    "\n"
    "A context manager that flips the global to `False` on enter and restores "
    "the previous value on exit is the standard wrapper around the toggle."
)

RECAP_REQUIRES_GRAD_PROP = (
    "## requires_grad propagation — quick refresher\n"
    "\n"
    "Output `requires_grad` is the **OR over all Tensor inputs** — if ANY input "
    "is grad-tracked, the output must be too (otherwise we lose the chain):\n"
    "\n"
    "```python\n"
    "requires_grad = grad_tracking_enabled and is_differentiable and any(\n"
    "    isinstance(a, Tensor) and a.requires_grad for a in args\n"
    ")\n"
    "```\n"
    "\n"
    "Three gates, ALL must be true:\n"
    "1. `grad_tracking_enabled` (the global toggle).\n"
    "2. `is_differentiable` (the per-op flag; ops like `torch.equal` "
    "   pass `False`).\n"
    "3. At least one Tensor input with `requires_grad=True`.\n"
    "\n"
    "Non-Tensor inputs (ints, floats) are filtered by the `isinstance` guard so "
    "they don't accidentally veto grad. Constants don't *contribute* grad "
    "tracking — but they don't *block* it either."
)

RECAP_UNBROADCAST = (
    "## Unbroadcast pattern — quick refresher\n"
    "\n"
    "PyTorch's elementwise ops broadcast — `x + y` may yield a result with more "
    "dims than either input. The reverse pass has the opposite problem: "
    "`grad_out` has the broadcast (output) shape, but `dL/dx` must have `x`'s "
    "*original* shape. Solution: **sum out the broadcast axes**.\n"
    "\n"
    "Two cases to handle:\n"
    "- **Leading new axes.** If `x.shape == (3, 4)` and `out.shape == "
    "  (2, 3, 4)`, broadcasting added a leading dim of size 2. Sum it out:\n"
    "  `grad_x = grad_out.sum(dim=0)`.\n"
    "- **Size-1 dims that got expanded.** If `x.shape == (1, 4)` and "
    "  `out.shape == (3, 4)`, broadcasting expanded `x`'s leading dim. Sum it "
    "  out with `keepdim=True` to preserve the size-1 axis: "
    "  `grad_x = grad_out.sum(dim=0, keepdim=True)`.\n"
    "\n"
    "Canonical recipe:\n"
    "```python\n"
    "def unbroadcast(grad, x):\n"
    "    # 1. peel leading axes\n"
    "    while grad.ndim > x.ndim:\n"
    "        grad = grad.sum(dim=0)\n"
    "    # 2. peel expanded size-1 axes\n"
    "    for i, size in enumerate(x.shape):\n"
    "        if size == 1 and grad.shape[i] != 1:\n"
    "            grad = grad.sum(dim=i, keepdim=True)\n"
    "    return grad\n"
    "```\n"
    "\n"
    "Every binary back fn wraps its result in `unbroadcast(..., x)` so the "
    "returned grad always matches the parent's stored shape."
)


# ---------------------------------------------------------------- spec helper

# Shared autograd-internals preamble that every drill in this file gets injected
# into its standalone notebook via _emit_standalone's extra_imports hook.
# Defines: dataclass, Callable, the MiniTensor wrapper class (a Tensor stand-in
# with a `.array` raw tensor and a `.recipe` field), a Recipe dataclass, and the
# module-level grad_tracking_enabled toggle that the drills read.
_AUTOGRAD_PREAMBLE = (
    "# === manual autograd primitives — shared across all drills in this folder ===\n"
    "from dataclasses import dataclass, field\n"
    "from typing import Any, Callable, Optional\n"
    "\n"
    "grad_tracking_enabled = True\n"
    "\n"
    "@dataclass\n"
    "class Recipe:\n"
    "    func: Optional[Callable] = None\n"
    "    args: tuple = ()\n"
    "    kwargs: dict = field(default_factory=dict)\n"
    "    parents: dict = field(default_factory=dict)\n"
    "\n"
    "class MiniTensor:\n"
    "    \"\"\"A minimal Tensor wrapper for the ARENA-style manual-autograd drills.\n"
    "    Wraps a raw `torch.Tensor` in `.array`. Carries an optional `.recipe`\n"
    "    populated by wrap_forward_fn. `requires_grad` is set by the wrapper.\"\"\"\n"
    "    def __init__(self, array, requires_grad: bool = False, recipe=None):\n"
    "        self.array = array\n"
    "        self.requires_grad = requires_grad\n"
    "        self.recipe = recipe\n"
    "    def __repr__(self):\n"
    "        return f'MiniTensor({self.array!r}, requires_grad={self.requires_grad})'"
)


def _spec(
    *,
    atom_id: str,
    subtopic: str,
    recap: str,
    ex_idx: int,
    ex_title: str,
    slug: str,
    bloom: str,
    difficulty_num: int,
    keywords: list[str],
    kcs: list[str],
    lo: str,
    prompt_body: str,
    stub: str,
    test_body: str,
    solution_body: str,
    solution_notes: str = "",
    extra_imports: list[str] | None = None,
) -> dict:
    dots = ("🔴" * difficulty_num) + ("⚪" * (5 - difficulty_num))
    # Every autograd-internals drill needs the shared preamble (MiniTensor,
    # Recipe, grad_tracking_enabled). Append any caller-supplied extras after.
    merged_imports = [_AUTOGRAD_PREAMBLE] + list(extra_imports or [])
    return {
        "atom_id": atom_id,
        "subtopic": subtopic,
        "topic_folder": TOPIC,
        "atom_recap_md": recap,
        "exercise_index": ex_idx,
        "exercise_title": ex_title,
        "slug": slug,
        "bloom_level": bloom,
        "difficulty_num": difficulty_num,
        "difficulty_dots": dots,
        "keywords": keywords,
        "kcs": kcs,
        "lo": lo,
        "prompt_body": prompt_body,
        "stub": stub,
        "test_body": test_body,
        "solution_body": solution_body,
        "solution_notes": solution_notes,
        "extra_imports": merged_imports,
    }


# =========================================================================
# atom: chain-rule-elementwise  (1 exercise)
# =========================================================================

SPEC_CHAIN_RULE = _spec(
    atom_id="chain-rule-elementwise",
    subtopic="Backprop: Elementwise chain rule",
    recap=RECAP_CHAIN_RULE_ELEMENTWISE,
    ex_idx=1,
    ex_title="write sigmoid_back and relu_back from the elementwise chain rule",
    slug="write-sigmoid-and-relu-back-from-elementwise-chain-rule",
    bloom="Apply",
    difficulty_num=2,
    keywords=["chain-rule", "sigmoid", "relu", "elementwise-derivative"],
    kcs=["chain-rule-elementwise", "back-fn-uses-cached-out"],
    lo=(
        "Apply the elementwise chain rule to derive sigmoid_back and relu_back, "
        "returning grad_in = grad_out * f'(x) without materializing any "
        "Jacobian matrix."
    ),
    prompt_body=(
        "Implement TWO elementwise backward fns by working out the local "
        "derivative and multiplying it by `grad_out`:\n\n"
        "**1. `sigmoid_back(grad_out, out, x)`** — gradient of "
        "`out = 1 / (1 + exp(-x))`.\n"
        "   - The clean form uses `out`: `d/dx sigmoid(x) = sigmoid(x) * "
        "(1 - sigmoid(x)) = out * (1 - out)`.\n"
        "   - So `dL/dx = grad_out * out * (1 - out)`.\n\n"
        "**2. `relu_back(grad_out, out, x)`** — gradient of "
        "`out = max(x, 0)`.\n"
        "   - The local derivative is `1` where `x > 0`, `0` elsewhere "
        "(undefined at exactly 0; convention: use 0).\n"
        "   - So `dL/dx = grad_out * (x > 0)`. (Multiplying by a bool tensor "
        "is fine — PyTorch promotes it to the float dtype.)\n\n"
        "**The point.** Both ops are elementwise, so the Jacobian is diagonal, "
        "so the chain rule reduces to a per-position product. No matrix is "
        "materialized — `grad_in.shape == x.shape` always.\n\n"
        "Inputs are plain `torch.Tensor`; no autograd. Return tensors with the "
        "same shape and float dtype as `x`."
    ),
    stub=(
        "def sigmoid_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = sigmoid(x). Use the cached `out` to avoid recomputation."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def relu_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = relu(x). Pass grad where x > 0, zero elsewhere."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- sigmoid_back ---\n"
        "x = t.tensor([-2.0, 0.0, 2.0])\n"
        "out = t.sigmoid(x)\n"
        "g = sigmoid_back(t.ones(3), out, x)\n"
        "expected = out * (1 - out)\n"
        "assert g.shape == x.shape, f'sigmoid_back shape: {g.shape}'\n"
        "assert t.allclose(g, expected), f'sigmoid_back value: {g} vs {expected}'\n"
        "\n"
        "# sigmoid_back with non-unit grad_out — chain-rule scales each entry.\n"
        "grad_out = t.tensor([5.0, -3.0, 2.0])\n"
        "g = sigmoid_back(grad_out, out, x)\n"
        "expected = grad_out * out * (1 - out)\n"
        "assert t.allclose(g, expected), 'sigmoid_back chain rule failed'\n"
        "\n"
        "# --- relu_back ---\n"
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])\n"
        "out = t.relu(x)\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0, 1.0, 1.0])\n"
        "g = relu_back(grad_out, out, x)\n"
        "expected = t.tensor([0.0, 0.0, 0.0, 1.0, 1.0])\n"
        "assert g.shape == x.shape, f'relu_back shape: {g.shape}'\n"
        "assert t.allclose(g, expected), f'relu_back value: {g}'\n"
        "\n"
        "# relu_back at exactly 0 must produce 0 (convention).\n"
        "x_zero = t.tensor([0.0, 0.0])\n"
        "g_zero = relu_back(t.ones(2), t.relu(x_zero), x_zero)\n"
        "assert t.allclose(g_zero, t.zeros(2)), 'relu_back at x=0 must be 0'\n"
        "\n"
        "# Matrix shapes for both.\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "out_s = t.sigmoid(X)\n"
        "out_r = t.relu(X)\n"
        "g_s = sigmoid_back(G, out_s, X)\n"
        "g_r = relu_back(G, out_r, X)\n"
        "assert g_s.shape == (3, 4)\n"
        "assert g_r.shape == (3, 4)\n"
        "\n"
        "# Cross-check against torch.autograd as a witness.\n"
        "for name, fwd, back_fn in [\n"
        "    ('sigmoid', t.sigmoid, sigmoid_back),\n"
        "    ('relu', t.relu, relu_back),\n"
        "]:\n"
        "    x_ref = t.tensor([-1.5, -0.2, 0.3, 1.5], requires_grad=True)\n"
        "    y = fwd(x_ref).sum()\n"
        "    y.backward()\n"
        "    out_cached = fwd(x_ref.detach())\n"
        "    g_ours = back_fn(t.ones(4), out_cached, x_ref.detach())\n"
        "    assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "        f'{name}_back disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        "    )"
    ),
    solution_body=(
        "def sigmoid_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = out * (1 - out).\n"
        "    # Using `out` is faster (no second exp call) and numerically stabler.\n"
        "    return grad_out * out * (1 - out)\n"
        "\n"
        "\n"
        "def relu_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d/dx relu(x) = 1 for x > 0 else 0. Multiply grad_out by the mask.\n"
        "    # Bool * float in torch promotes to float — no manual cast needed.\n"
        "    return grad_out * (x > 0)"
    ),
    solution_notes=(
        "**Why these are one-liners.** Elementwise ops have a diagonal local "
        "Jacobian. The chain-rule sum `dL/dx[i] = sum_j (dL/dout[j] * "
        "d(out[j])/d(x[i]))` collapses to `dL/dout[i] * f'(x[i])` because only "
        "the diagonal term survives. No matrix is materialized — that's why "
        "elementwise back fns are O(n) instead of O(n^2).\n\n"
        "**`out * (1 - out)` vs recomputing.** You could write `sig = "
        "t.sigmoid(x); return grad_out * sig * (1 - sig)`, but the whole "
        "reason `out` is in the back-fn signature is so you DON'T have to. "
        "Saves one exp per node on the reverse pass.\n\n"
        "**Why `relu_back` ignores `out`.** `out` is `max(x, 0)`, which doesn't "
        "tell you whether `x > 0` (when `x = 0`, `out = 0` either way). You "
        "have to read `x` directly. This is why the uniform signature passes "
        "BOTH `out` AND `x` — different ops need different cached state."
    ),
)


# =========================================================================
# atom: arg-position-back-functions  (1 exercise)
# =========================================================================

SPEC_ARG_POSITION = _spec(
    atom_id="arg-position-back-functions",
    subtopic="Backprop: Arg-position back funcs",
    recap=RECAP_ARG_POSITION,
    ex_idx=1,
    ex_title="write div_back0 and div_back1 — asymmetric per-arg back fns",
    slug="write-div-back0-and-div-back1-asymmetric-per-arg",
    bloom="Apply",
    difficulty_num=3,
    keywords=["arg-position", "div", "asymmetric", "back0", "back1"],
    kcs=["arg-position-back-functions", "backward-fn-signature"],
    lo=(
        "Apply the per-arg-position back-fn convention by writing div_back0 "
        "and div_back1 for out = x / y — the canonical asymmetric binary op "
        "where the two gradients are different functions."
    ),
    prompt_body=(
        "Implement TWO back fns for `out = x / y`, both with the same "
        "signature `(grad_out, out, x, y) -> Tensor`:\n\n"
        "**1. `div_back0(grad_out, out, x, y)`** — gradient w.r.t. arg-0 "
        "(`x`).\n"
        "   - Math: `d(x/y)/dx = 1/y`.\n"
        "   - So `dL/dx = grad_out / y`.\n\n"
        "**2. `div_back1(grad_out, out, x, y)`** — gradient w.r.t. arg-1 "
        "(`y`).\n"
        "   - Math: `d(x/y)/dy = -x / y**2`.\n"
        "   - So `dL/dy = grad_out * (-x / y**2)`, OR equivalently "
        "`-grad_out * out / y` (since `out = x/y` ⇒ `out/y = x/y**2`).\n\n"
        "**Why the split.** Division is *asymmetric* — `div_back0` and "
        "`div_back1` are different functions. Compare with `add` or "
        "`multiply` (`add_back0 == add_back1`, just `grad_out`), where the "
        "two bodies happen to be identical but BOTH still get registered "
        "into BACK_FUNCS at argnum=0 AND argnum=1.\n\n"
        "For this drill assume `x.shape == y.shape == out.shape` — no "
        "broadcasting (that's a separate atom). Inputs and outputs are plain "
        "`torch.Tensor`, no autograd, float dtype. Return tensors with the "
        "same shape as the input each back fn corresponds to."
    ),
    stub=(
        "def div_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dx for out = x / y. Returns a tensor shaped like x."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def div_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dy for out = x / y. Returns a tensor shaped like y."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- scalar sanity: out = 6 / 2 = 3 ---\n"
        "x = t.tensor([6.0]); y = t.tensor([2.0])\n"
        "out = x / y\n"
        "g0 = div_back0(t.tensor([1.0]), out, x, y)\n"
        "g1 = div_back1(t.tensor([1.0]), out, x, y)\n"
        "# d(x/y)/dx = 1/y = 0.5; d(x/y)/dy = -x/y^2 = -6/4 = -1.5\n"
        "assert t.allclose(g0, t.tensor([0.5])), f'div_back0 scalar: {g0}'\n"
        "assert t.allclose(g1, t.tensor([-1.5])), f'div_back1 scalar: {g1}'\n"
        "\n"
        "# --- vector ---\n"
        "x = t.tensor([4.0, 9.0, 16.0])\n"
        "y = t.tensor([2.0, 3.0, 4.0])\n"
        "out = x / y\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "g0 = div_back0(grad_out, out, x, y)\n"
        "g1 = div_back1(grad_out, out, x, y)\n"
        "assert g0.shape == x.shape, f'div_back0 shape: {g0.shape}'\n"
        "assert g1.shape == y.shape, f'div_back1 shape: {g1.shape}'\n"
        "assert t.allclose(g0, 1 / y), f'div_back0 value: {g0}'\n"
        "assert t.allclose(g1, -x / y**2), f'div_back1 value: {g1}'\n"
        "\n"
        "# --- non-unit grad_out, matrix shape ---\n"
        "rng = t.Generator().manual_seed(2)\n"
        "X = t.randn(3, 4, generator=rng) + 5  # keep positive\n"
        "Y = t.randn(3, 4, generator=rng) + 5\n"
        "OUT = X / Y\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "g0 = div_back0(G, OUT, X, Y)\n"
        "g1 = div_back1(G, OUT, X, Y)\n"
        "assert g0.shape == (3, 4)\n"
        "assert g1.shape == (3, 4)\n"
        "assert t.allclose(g0, G / Y), 'div_back0 chain rule failed'\n"
        "assert t.allclose(g1, G * (-X / Y**2)), 'div_back1 chain rule failed'\n"
        "\n"
        "# --- div_back0 and div_back1 must be DIFFERENT functions ---\n"
        "# (Sanity: asymmetric ops must not collapse to the same body.)\n"
        "x_test = t.tensor([3.0, 5.0])\n"
        "y_test = t.tensor([2.0, 4.0])\n"
        "out_test = x_test / y_test\n"
        "g0_test = div_back0(t.ones(2), out_test, x_test, y_test)\n"
        "g1_test = div_back1(t.ones(2), out_test, x_test, y_test)\n"
        "assert not t.allclose(g0_test, g1_test), (\n"
        "    'div_back0 and div_back1 should not produce the same result here'\n"
        ")\n"
        "\n"
        "# --- Witness vs torch.autograd ---\n"
        "x_ref = t.tensor([2.0, 5.0, 7.0], requires_grad=True)\n"
        "y_ref = t.tensor([3.0, 11.0, 13.0], requires_grad=True)\n"
        "z = (x_ref / y_ref).sum()\n"
        "z.backward()\n"
        "out_cached = x_ref.detach() / y_ref.detach()\n"
        "g0_ours = div_back0(t.ones(3), out_cached, x_ref.detach(), y_ref.detach())\n"
        "g1_ours = div_back1(t.ones(3), out_cached, x_ref.detach(), y_ref.detach())\n"
        "assert t.allclose(g0_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'div_back0 disagrees with autograd: ours={g0_ours}, ref={x_ref.grad}'\n"
        ")\n"
        "assert t.allclose(g1_ours, y_ref.grad, atol=1e-6), (\n"
        "    f'div_back1 disagrees with autograd: ours={g1_ours}, ref={y_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def div_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # d(x/y)/dx = 1/y, so dL/dx = grad_out / y.\n"
        "    return grad_out / y\n"
        "\n"
        "\n"
        "def div_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # d(x/y)/dy = -x/y^2, so dL/dy = grad_out * (-x / y^2).\n"
        "    return grad_out * (-x / (y * y))"
    ),
    solution_notes=(
        "**Why TWO functions for one op.** The argnum is part of the BACK_FUNCS "
        "lookup key: `(torch.divide, 0) -> div_back0`, `(torch.divide, 1) -> "
        "div_back1`. The reverse pass walks `recipe.parents` (a `{argnum: "
        "Tensor}` dict), and for each entry calls the matching back fn. So "
        "registration always pairs: `BACK_FUNCS.add_back_func(t.divide, 0, "
        "div_back0); BACK_FUNCS.add_back_func(t.divide, 1, div_back1)`.\n\n"
        "**Symmetric ops still register twice.** `add_back0(g, out, x, y) = g` "
        "and `add_back1(g, out, x, y) = g` are the same function body, but you "
        "still register at both argnums. The dispatcher doesn't know whether "
        "an op is symmetric — it just looks up `(func, argnum)` and calls.\n\n"
        "**Equivalent form for `div_back1`.** Since `out = x/y`, we have "
        "`-x/y**2 = -out/y`. Either form works; using `out` saves one "
        "multiplication and matches the 'use the cached out' pattern, at the "
        "cost of being slightly less obvious as the partial derivative."
    ),
)


# =========================================================================
# atom: kwargs-pass-through-recipe  (1 exercise)
# =========================================================================

SPEC_KWARGS = _spec(
    atom_id="kwargs-pass-through-recipe",
    subtopic="Backprop: Kwargs pass-through",
    recap=RECAP_KWARGS_PASS_THROUGH,
    ex_idx=1,
    ex_title="thread kwargs into forward call AND Recipe",
    slug="thread-kwargs-into-forward-call-and-recipe",
    bloom="Apply",
    difficulty_num=3,
    keywords=["kwargs", "recipe", "wrap-forward-fn", "sum", "dim"],
    kcs=["kwargs-pass-through-recipe", "recipe-dataclass"],
    lo=(
        "Apply the kwargs-pass-through pattern by routing keyword args into "
        "both the forward call and the constructed Recipe so a downstream "
        "back fn can replay the op."
    ),
    prompt_body=(
        "We've given you a stripped-down `Tensor` wrapper and a `Recipe` "
        "dataclass. Implement `wrap_forward_fn(fwd_fn)`, which returns a "
        "closure `tensor_func(*args, **kwargs)` that:\n\n"
        "1. **Unboxes** every Tensor input — pull `.array` out of each, leave "
        "non-Tensors alone.\n"
        "2. Calls `fwd_fn(*raw_args, **kwargs)` to compute the raw output. "
        "   **The kwargs must reach the forward call** — otherwise `sum(x, "
        "dim=1)` would silently reduce over the wrong axis.\n"
        "3. Boxes the result in `Tensor(out_raw)` and attaches "
        "   `out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)`. "
        "   **The same kwargs dict must be stored on the Recipe** so the "
        "backward fn can be called with `**recipe.kwargs` at reverse time.\n"
        "4. Returns the boxed Tensor.\n\n"
        "Use `parents = {idx: a for idx, a in enumerate(args) if "
        "isinstance(a, Tensor)}`.\n\n"
        "**Why kwargs are the failure mode.** It's tempting to write "
        "`Recipe(fwd_fn, raw_args, {}, parents)` (empty dict) if your tests "
        "don't reach the reverse pass yet — the forward result is correct, "
        "so the bug is invisible. Then `sum_back` runs at reverse time, "
        "doesn't know which `dim` was reduced, broadcasts wrong, and shape "
        "errors blow up far from the cause.\n\n"
        "Don't call `torch.autograd`; we're building the autograd layer "
        "manually."
    ),
    stub=(
        "from dataclasses import dataclass\n"
        "from typing import Callable, Any\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Callable\n"
        "    args: tuple\n"
        "    kwargs: dict\n"
        "    parents: dict\n"
        "\n"
        "\n"
        "class MiniTensor:\n"
        '    """Minimal Tensor wrapper around a torch.Tensor (named `array`)."""\n'
        "    def __init__(self, array):\n"
        "        self.array = array\n"
        "        self.recipe = None\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn: Callable) -> Callable:\n"
        '    """Return tensor_func that boxes/unboxes around fwd_fn and threads kwargs."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- forward call gets kwargs (sum over dim=1) ---\n"
        "wrapped_sum = wrap_forward_fn(t.sum)\n"
        "x = MiniTensor(t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n"
        "out = wrapped_sum(x, dim=1)\n"
        "assert isinstance(out, MiniTensor), 'output must be a MiniTensor'\n"
        "assert t.allclose(out.array, t.tensor([6.0, 15.0])), (\n"
        "    f'forward dim=1 was ignored: out.array={out.array} '\n"
        "    f'(expected [6, 15])'\n"
        ")\n"
        "\n"
        "# --- Recipe carries the same kwargs that the call used ---\n"
        "assert out.recipe is not None, 'Recipe was never attached'\n"
        "assert out.recipe.func is t.sum, 'Recipe.func wrong'\n"
        "assert out.recipe.kwargs == {'dim': 1}, (\n"
        "    f'Recipe.kwargs missing or wrong: {out.recipe.kwargs}'\n"
        ")\n"
        "assert 0 in out.recipe.parents, 'parents missing arg-0 Tensor'\n"
        "assert out.recipe.parents[0] is x, 'parents must reference the original Tensor'\n"
        "\n"
        "# --- a SECOND kwarg also threads through (keepdim) ---\n"
        "out2 = wrapped_sum(x, dim=1, keepdim=True)\n"
        "assert out2.array.shape == (2, 1), (\n"
        "    f'keepdim=True ignored: out2.shape={out2.array.shape}'\n"
        ")\n"
        "assert out2.recipe.kwargs == {'dim': 1, 'keepdim': True}, (\n"
        "    f'Recipe lost keepdim: {out2.recipe.kwargs}'\n"
        ")\n"
        "\n"
        "# --- no kwargs case still works (empty dict stored) ---\n"
        "wrapped_log = wrap_forward_fn(t.log)\n"
        "y = MiniTensor(t.tensor([1.0, t.e, t.e * t.e]))\n"
        "out3 = wrapped_log(y)\n"
        "assert t.allclose(out3.array, t.tensor([0.0, 1.0, 2.0]), atol=1e-5)\n"
        "assert out3.recipe.kwargs == {}, (\n"
        "    f'kwargs should be empty dict, got {out3.recipe.kwargs}'\n"
        ")\n"
        "\n"
        "# --- args on Recipe are RAW (unboxed) tensors ---\n"
        "assert isinstance(out.recipe.args[0], t.Tensor), (\n"
        "    f'Recipe.args[0] should be raw torch.Tensor, got {type(out.recipe.args[0])}'\n"
        ")\n"
        "assert not isinstance(out.recipe.args[0], MiniTensor), (\n"
        "    'Recipe.args should hold the unboxed raw tensor, not the MiniTensor'\n"
        ")\n"
        "\n"
        "# --- proof: a downstream back fn can REPLAY the op using recipe.kwargs ---\n"
        "# (Forward shape-restore: a correct sum_back would broadcast grad back\n"
        "# along the same dim — only possible because kwargs are preserved.)\n"
        "grad_out = t.tensor([1.0, 1.0])\n"
        "replayed = grad_out.unsqueeze(out.recipe.kwargs['dim']).expand_as(x.array)\n"
        "assert replayed.shape == x.array.shape, 'shape replay would fail'"
    ),
    solution_body=(
        "def wrap_forward_fn(fwd_fn: Callable) -> Callable:\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        # 1. unbox MiniTensor inputs to raw torch.Tensor (pass-through non-Tensors)\n"
        "        raw_args = tuple(\n"
        "            a.array if isinstance(a, MiniTensor) else a for a in args\n"
        "        )\n"
        "        # 2. forward call MUST receive the kwargs\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        # 3. box result and attach Recipe — kwargs preserved for reverse pass\n"
        "        parents = {\n"
        "            idx: a for idx, a in enumerate(args) if isinstance(a, MiniTensor)\n"
        "        }\n"
        "        out = MiniTensor(out_raw)\n"
        "        out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    solution_notes=(
        "**Two places the kwargs go, ALWAYS in this order.**\n"
        "1. `fwd_fn(*raw_args, **kwargs)` — without this, the forward output is "
        "wrong (e.g. `sum` reduces over the default axis instead of `dim`).\n"
        "2. `Recipe(..., kwargs, ...)` — without this, the reverse pass has no "
        "way to call `back_fn(grad_out, out, *recipe.args, **recipe.kwargs)` "
        "with the same kwargs the forward used.\n\n"
        "**Why it's a `dict`, not unpacked.** Because the back fn signature is "
        "`(grad_out, out, *args, **kwargs)`, the reverse pass dispatches with "
        "`back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)`. "
        "Storing kwargs as the raw dict means the dispatcher line is generic — "
        "no per-op switch needed.\n\n"
        "**The silent-bug failure mode.** If you forget step (2) but remember "
        "step (1), the forward looks correct and the bug only triggers when "
        "the reverse pass runs. The ARENA notebook hints at exactly this "
        "trap: 'if you're failing tests but think your implementation is "
        "correct, go back and check this.'"
    ),
)


# =========================================================================
# atom: recipe-dataclass  (1 exercise)
# =========================================================================

SPEC_RECIPE = _spec(
    atom_id="recipe-dataclass",
    subtopic="Backprop: Recipe dataclass",
    recap=RECAP_RECIPE_DATACLASS,
    ex_idx=1,
    ex_title="define Recipe and construct it for log_forward",
    slug="define-recipe-and-construct-it-for-log-forward",
    bloom="Apply",
    difficulty_num=2,
    keywords=["recipe", "dataclass", "log-forward", "parents", "func"],
    kcs=["recipe-dataclass", "box-array-to-tensor-with-recipe"],
    lo=(
        "Apply the Recipe-construction pattern by defining the 4-field "
        "dataclass and attaching a correctly-populated Recipe to the output "
        "of a single-arg forward op (log)."
    ),
    prompt_body=(
        "Implement BOTH parts:\n\n"
        "**1. The `Recipe` dataclass.** Define a `@dataclass` with EXACTLY "
        "these four fields, in this order:\n"
        "   - `func: Callable` — the forward fn (e.g. `torch.log`).\n"
        "   - `args: tuple` — the raw (unboxed) positional args at call time.\n"
        "   - `kwargs: dict` — the raw keyword args at call time.\n"
        "   - `parents: dict` — `{argnum: MiniTensor}` for each Tensor input.\n\n"
        "**2. `log_forward(x)`** — single-arg autograd-aware log:\n"
        "   - Accepts a `MiniTensor` whose `.array` is a `torch.Tensor`.\n"
        "   - Computes `out_arr = torch.log(x.array)`.\n"
        "   - Returns a new `MiniTensor(out_arr)` with `out.recipe` set to a "
        "`Recipe` carrying:\n"
        "     - `func = torch.log`\n"
        "     - `args = (x.array,)`  (a 1-tuple of the raw input)\n"
        "     - `kwargs = {}`\n"
        "     - `parents = {0: x}`  (arg 0 is the input MiniTensor)\n\n"
        "**Why all four fields, always.** The reverse pass treats Recipe "
        "generically: it reads `recipe.func` to find the back fn, "
        "`recipe.args` and `recipe.kwargs` to replay the original call, and "
        "`recipe.parents` to find what to differentiate next. Drop any field "
        "and the dispatcher breaks.\n\n"
        "Use plain `torch.Tensor` for `.array`; no autograd."
    ),
    stub=(
        "from dataclasses import dataclass\n"
        "from typing import Callable\n"
        "\n"
        "\n"
        "class MiniTensor:\n"
        "    def __init__(self, array):\n"
        "        self.array = array\n"
        "        self.recipe = None\n"
        "\n"
        "\n"
        "# Define the Recipe dataclass here. It MUST have fields\n"
        "# (func, args, kwargs, parents) in that order.\n"
        "# (replace this comment with the real definition)\n"
        "\n"
        "\n"
        "def log_forward(x: MiniTensor) -> MiniTensor:\n"
        '    """Compute log(x.array); return a MiniTensor with .recipe attached."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- Recipe shape ---\n"
        "from dataclasses import fields\n"
        "f_names = [f.name for f in fields(Recipe)]\n"
        "assert f_names == ['func', 'args', 'kwargs', 'parents'], (\n"
        "    f'Recipe must have fields (func, args, kwargs, parents) in order, got {f_names}'\n"
        ")\n"
        "\n"
        "# --- Recipe constructor: positional args in correct order ---\n"
        "r = Recipe(t.log, (t.tensor([1.0]),), {}, {0: 'sentinel'})\n"
        "assert r.func is t.log\n"
        "assert isinstance(r.args, tuple) and len(r.args) == 1\n"
        "assert r.kwargs == {}\n"
        "assert r.parents == {0: 'sentinel'}\n"
        "\n"
        "# --- log_forward: numerical correctness ---\n"
        "x = MiniTensor(t.tensor([1.0, t.e, t.e * t.e]))\n"
        "out = log_forward(x)\n"
        "assert isinstance(out, MiniTensor), 'log_forward must return a MiniTensor'\n"
        "assert t.allclose(out.array, t.tensor([0.0, 1.0, 2.0]), atol=1e-5), (\n"
        "    f'log values wrong: {out.array}'\n"
        ")\n"
        "\n"
        "# --- Recipe is attached and fully populated ---\n"
        "assert out.recipe is not None, 'log_forward did not attach a Recipe'\n"
        "assert out.recipe.func is t.log, f'recipe.func wrong: {out.recipe.func}'\n"
        "assert out.recipe.args == (x.array,), 'recipe.args must be a 1-tuple of x.array'\n"
        "assert isinstance(out.recipe.args, tuple), 'recipe.args must be a tuple, not list'\n"
        "assert out.recipe.kwargs == {}, f'recipe.kwargs wrong: {out.recipe.kwargs}'\n"
        "assert out.recipe.parents == {0: x}, f'recipe.parents wrong: {out.recipe.parents}'\n"
        "\n"
        "# --- args holds RAW tensors, not the MiniTensor ---\n"
        "stored = out.recipe.args[0]\n"
        "assert isinstance(stored, t.Tensor) and not isinstance(stored, MiniTensor), (\n"
        "    f'recipe.args[0] should be the unboxed torch.Tensor, got {type(stored)}'\n"
        ")\n"
        "# Identity: the recipe stored x.array, not a copy.\n"
        "assert stored is x.array, 'recipe.args[0] should BE x.array (identity, not copy)'\n"
        "\n"
        "# --- parents holds the boxed MiniTensor by argnum ---\n"
        "assert out.recipe.parents[0] is x, 'parents[0] must reference the input MiniTensor'\n"
        "\n"
        "# --- Recipe round-trip is enough to replay the forward call ---\n"
        "replay = out.recipe.func(*out.recipe.args, **out.recipe.kwargs)\n"
        "assert t.allclose(replay, out.array), 'recipe round-trip should reproduce out.array'"
    ),
    solution_body=(
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Callable\n"
        "    args: tuple\n"
        "    kwargs: dict\n"
        "    parents: dict\n"
        "\n"
        "\n"
        "def log_forward(x: MiniTensor) -> MiniTensor:\n"
        "    out_arr = t.log(x.array)\n"
        "    out = MiniTensor(out_arr)\n"
        "    out.recipe = Recipe(\n"
        "        func=t.log,\n"
        "        args=(x.array,),     # raw, unboxed; 1-tuple\n"
        "        kwargs={},            # log takes no kwargs in this drill\n"
        "        parents={0: x},      # arg-0 is a Tensor → record it\n"
        "    )\n"
        "    return out"
    ),
    solution_notes=(
        "**Why a dataclass and not just a tuple.** A 4-tuple would work, but "
        "`@dataclass` gives you named attribute access (`recipe.func`, "
        "`recipe.parents`), repr-for-debug, and structural equality for free. "
        "The reverse pass would otherwise be a forest of `recipe[0]`, "
        "`recipe[1]` — opaque and bug-prone.\n\n"
        "**`parents` is the edge list of the compute graph.** Every non-leaf "
        "Tensor's `recipe.parents` tells the reverse pass which Tensors fed "
        "into it. Topological sort follows these edges; the dispatcher then "
        "iterates `for argnum, parent in recipe.parents.items()` and calls "
        "the matching back fn at that argnum.\n\n"
        "**Identity, not copy.** Storing `x.array` (the same tensor object) "
        "rather than `x.array.clone()` matters because (a) it's free, (b) "
        "elementwise back fns can read the cached input without allocating, "
        "and (c) some back fns even mutate in place during accumulation."
    ),
)


# =========================================================================
# atom: parents-dict-by-argidx  (1 exercise)
# =========================================================================

SPEC_PARENTS = _spec(
    atom_id="parents-dict-by-argidx",
    subtopic="Backprop: Parents dict by argidx",
    recap=RECAP_PARENTS_DICT,
    ex_idx=1,
    ex_title="build parents dict — skip non-Tensors, keep original argidx",
    slug="build-parents-dict-skip-non-tensors-keep-argidx",
    bloom="Apply",
    difficulty_num=2,
    keywords=["parents", "argidx", "dict-comprehension", "filter-non-tensor"],
    kcs=["parents-dict-by-argidx", "unbox-args-tensor-to-array"],
    lo=(
        "Apply the dict-comprehension parents-builder pattern: filter out "
        "non-Tensor inputs while preserving the original positional index "
        "as the dict key."
    ),
    prompt_body=(
        "Implement `build_parents(args)`. Given a tuple of positional inputs "
        "(some `MiniTensor`, some plain Python scalars / shape tuples / "
        "anything else), return a dict mapping **the original argidx** to "
        "**the MiniTensor at that position**:\n\n"
        "```\n"
        "build_parents((t1, 3.0, t2))         == {0: t1, 2: t2}\n"
        "build_parents((5, t1, (1, 2), t2))   == {1: t1, 3: t2}\n"
        "build_parents((1.0, 2.0))            == {}\n"
        "```\n\n"
        "Two rules — both critical:\n\n"
        "**1. Skip non-Tensors.** Use `isinstance(a, MiniTensor)`. If a `multiply"
        "(t, 3.0)` call leaks the float 3.0 into `parents`, the reverse pass "
        "later tries to add a float to a Tensor grad and crashes — the wrong "
        "side of the type system.\n\n"
        "**2. Keep the ORIGINAL argidx as the key.** Do NOT collapse "
        "`(t1, 3.0, t2)` to `{0: t1, 1: t2}` — the second entry must be `2`, "
        "not `1`, because the back-fn lookup is by `(forward_fn, argnum)` "
        "with the ORIGINAL argnum. If you renumber, "
        "`BACK_FUNCS.get_back_func(func, 1)` returns the wrong back fn at "
        "reverse time.\n\n"
        "The canonical one-liner is `{idx: a for idx, a in enumerate(args) "
        "if isinstance(a, MiniTensor)}`. Write it (or any equivalent loop)."
    ),
    stub=(
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad=False):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "\n"
        "\n"
        "def build_parents(args: tuple) -> dict:\n"
        '    """Return {argidx: MiniTensor} for each MiniTensor in args, in original order."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- empty / all-non-Tensor inputs ---\n"
        "assert build_parents(()) == {}, 'empty tuple should give empty dict'\n"
        "assert build_parents((1, 2.0, 'x')) == {}, 'no Tensors -> empty dict'\n"
        "\n"
        "# --- single Tensor at argnum=0 ---\n"
        "t1 = MiniTensor(t.tensor([1.0]))\n"
        "assert build_parents((t1,)) == {0: t1}\n"
        "\n"
        "# --- multiple Tensors, contiguous ---\n"
        "t2 = MiniTensor(t.tensor([2.0]))\n"
        "p = build_parents((t1, t2))\n"
        "assert p == {0: t1, 1: t2}, f'two-tensor case: {p}'\n"
        "\n"
        "# --- Tensor in arg-0 only, float at arg-1 (multiply x by scalar) ---\n"
        "p = build_parents((t1, 3.0))\n"
        "assert p == {0: t1}, f'multiply(t, 3.0): {p}'\n"
        "\n"
        "# --- float at arg-0, Tensor at arg-1 — argnum must stay 1, NOT collapse to 0 ---\n"
        "p = build_parents((3.0, t1))\n"
        "assert p == {1: t1}, (\n"
        "    f'arg-1 Tensor must keep argnum=1, got {p} '\n"
        "    f'(renumbering would break BACK_FUNCS dispatch)'\n"
        ")\n"
        "\n"
        "# --- mixed: int, Tensor, tuple, Tensor ---\n"
        "p = build_parents((5, t1, (1, 2, 3), t2))\n"
        "assert p == {1: t1, 3: t2}, f'mixed: {p}'\n"
        "\n"
        "# --- Tensors at non-consecutive positions ---\n"
        "t3 = MiniTensor(t.tensor([3.0]))\n"
        "p = build_parents((t1, 'sep', t2, 7, t3))\n"
        "assert p == {0: t1, 2: t2, 4: t3}, f'non-consecutive: {p}'\n"
        "\n"
        "# --- identity preserved: dict values must BE the same objects ---\n"
        "vals = list(p.values())\n"
        "assert vals[0] is t1, 'dict value must be the same object as input'\n"
        "assert vals[1] is t2, 'dict value must be the same object as input'\n"
        "assert vals[2] is t3, 'dict value must be the same object as input'\n"
        "\n"
        "# --- raw torch.Tensors should be SKIPPED (only MiniTensors count as parents) ---\n"
        "raw = t.tensor([1.0])\n"
        "p = build_parents((raw, t1))\n"
        "assert p == {1: t1}, (\n"
        "    f'raw torch.Tensor should be skipped (only MiniTensor counts), got {p}'\n"
        ")"
    ),
    solution_body=(
        "def build_parents(args: tuple) -> dict:\n"
        "    return {\n"
        "        idx: a\n"
        "        for idx, a in enumerate(args)\n"
        "        if isinstance(a, MiniTensor)\n"
        "    }"
    ),
    solution_notes=(
        "**`enumerate` before `if`.** Order matters in the comprehension: we "
        "first attach the original index to each arg via `enumerate`, THEN "
        "filter. Doing it the other way (filtering then enumerating the "
        "survivors) would re-number — exactly the bug rule 2 warns against.\n\n"
        "**Why `isinstance(a, MiniTensor)` and not `hasattr(a, 'array')`.** "
        "Duck-typing on `.array` would catch random objects that happen to "
        "have an `.array` attribute — e.g. a `numpy.ndarray` literally has "
        "an `.array` interface protocol. `isinstance` is precise: we want "
        "*the wrapper class*, not anything array-shaped.\n\n"
        "**The dual of this is `unbox`.** Where `build_parents` keeps the "
        "MiniTensors (filtered, keyed by argnum), `unbox_args` does the "
        "opposite — replaces each MiniTensor with its `.array` for the "
        "forward call, leaves non-Tensors alone. Same `isinstance` check, "
        "different transform."
    ),
)


# =========================================================================
# atom: grad-tracking-global-toggle  (1 exercise)
# =========================================================================

SPEC_GRAD_TRACKING = _spec(
    atom_id="grad-tracking-global-toggle",
    subtopic="Backprop: Grad-tracking toggle",
    recap=RECAP_GRAD_TRACKING_TOGGLE,
    ex_idx=1,
    ex_title="no_grad context manager built on a module-level toggle",
    slug="no-grad-context-manager-from-module-toggle",
    bloom="Apply",
    difficulty_num=3,
    keywords=["no_grad", "context-manager", "global-toggle", "inference"],
    kcs=["grad-tracking-global-toggle", "requires-grad-propagation"],
    lo=(
        "Apply the global-toggle + context-manager pattern: a module-level "
        "bool gates grad tracking, and a NoGrad ctx flips it off on enter "
        "and restores the previous value on exit (nesting-safe)."
    ),
    prompt_body=(
        "We've given you a module-level `grad_tracking_enabled = True` and a "
        "`Tensor` wrapper. Implement TWO pieces:\n\n"
        "**1. `compute_requires_grad(args)`** — return `True` iff "
        "`grad_tracking_enabled` is True AND at least one input is a Tensor "
        "with `requires_grad=True`:\n"
        "```\n"
        "grad_tracking_enabled and any(\n"
        "    isinstance(a, Tensor) and a.requires_grad for a in args\n"
        ")\n"
        "```\n"
        "Read the toggle FROM THE MODULE — don't snapshot it into a closure "
        "at import time. Tip: reference it via `globals()['grad_tracking_"
        "enabled']` or via `import sys; sys.modules[__name__]."
        "grad_tracking_enabled` so the latest value wins.\n\n"
        "**2. `NoGrad` context manager** — flips the global to False on "
        "`__enter__`, restores the **previous** value on `__exit__`. Use the "
        "previous value, not a hardcoded `True`, so nested `NoGrad()` blocks "
        "behave correctly (inner exit doesn't accidentally re-enable when an "
        "outer `NoGrad` is still active).\n\n"
        "Once both are wired, the test runs through 5 scenarios — single "
        "tensor inputs, mixed Tensor+scalar, ctx mgr toggling, nested ctx "
        "mgrs, and exception safety (toggle must restore even if the body "
        "raises).\n\n"
        "Do NOT call `torch.autograd` or use `torch.no_grad()` — we're "
        "reimplementing them."
    ),
    stub=(
        "grad_tracking_enabled = True\n"
        "\n"
        "\n"
        "class Tensor:\n"
        "    def __init__(self, array, requires_grad=False):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "\n"
        "\n"
        "def compute_requires_grad(args) -> bool:\n"
        '    """Apply the (global toggle AND any-input-requires-grad) gate."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "class NoGrad:\n"
        '    """Flip grad_tracking_enabled to False; restore previous value on exit."""\n'
        "    def __enter__(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        raise NotImplementedError()"
    ),
    test_body=(
        "# --- compute_requires_grad: enabled + tracked input ---\n"
        "assert grad_tracking_enabled is True\n"
        "t1 = Tensor(t.tensor([1.0]), requires_grad=True)\n"
        "t2 = Tensor(t.tensor([2.0]), requires_grad=False)\n"
        "assert compute_requires_grad((t1,)) is True, 'single tracked tensor'\n"
        "assert compute_requires_grad((t1, t2)) is True, 'any() over tracked + untracked'\n"
        "assert compute_requires_grad((t2,)) is False, 'untracked tensor → False'\n"
        "assert compute_requires_grad((t2, 3.0, 5)) is False, 'no tracked tensor → False'\n"
        "assert compute_requires_grad((3.0, t1)) is True, 'scalar then tracked tensor'\n"
        "assert compute_requires_grad(()) is False, 'empty args → False'\n"
        "\n"
        "# --- NoGrad ctx: must flip the GLOBAL, not a local snapshot ---\n"
        "# We read grad_tracking_enabled by bare reference inside _test_ex1 —\n"
        "# Python resolves it from the cell's module globals, which is exactly\n"
        "# where the implementation must set it. Works in Colab and in any\n"
        "# correctly-implemented kernel.\n"
        "with NoGrad():\n"
        "    assert globals()['grad_tracking_enabled'] is False, (\n"
        "        'NoGrad must set grad_tracking_enabled=False on enter'\n"
        "    )\n"
        "    # compute_requires_grad MUST see the new value (not a closure snapshot)\n"
        "    assert compute_requires_grad((t1,)) is False, (\n"
        "        'compute_requires_grad inside NoGrad should return False '\n"
        "        '(did you snapshot grad_tracking_enabled into a closure?)'\n"
        "    )\n"
        "# After exit, restored.\n"
        "assert globals()['grad_tracking_enabled'] is True, (\n"
        "    'NoGrad must restore grad_tracking_enabled on exit'\n"
        ")\n"
        "assert compute_requires_grad((t1,)) is True, 'restored after ctx exit'\n"
        "\n"
        "# --- nested NoGrad: inner exit must restore PREVIOUS value (still False) ---\n"
        "with NoGrad():\n"
        "    assert globals()['grad_tracking_enabled'] is False\n"
        "    with NoGrad():\n"
        "        assert globals()['grad_tracking_enabled'] is False\n"
        "    # inner exit — outer NoGrad still active, must remain False\n"
        "    assert globals()['grad_tracking_enabled'] is False, (\n"
        "        'inner NoGrad exit must restore the PREVIOUS value (False), '\n"
        "        'not hardcoded True — otherwise nesting breaks'\n"
        "    )\n"
        "assert globals()['grad_tracking_enabled'] is True, 'fully restored'\n"
        "\n"
        "# --- exception safety: toggle must restore even if body raises ---\n"
        "try:\n"
        "    with NoGrad():\n"
        "        assert globals()['grad_tracking_enabled'] is False\n"
        "        raise RuntimeError('intentional')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "assert globals()['grad_tracking_enabled'] is True, (\n"
        "    'NoGrad must restore the toggle even when the body raises'\n"
        ")"
    ),
    solution_body=(
        "def _get_flag():\n"
        "    # globals() inside a function returns the defining module's globals\n"
        "    # — works in both Colab kernels and isolated exec namespaces.\n"
        "    return globals()['grad_tracking_enabled']\n"
        "\n"
        "\n"
        "def _set_flag(v: bool):\n"
        "    globals()['grad_tracking_enabled'] = v\n"
        "\n"
        "\n"
        "def compute_requires_grad(args) -> bool:\n"
        "    return _get_flag() and any(\n"
        "        isinstance(a, Tensor) and a.requires_grad for a in args\n"
        "    )\n"
        "\n"
        "\n"
        "class NoGrad:\n"
        "    def __enter__(self):\n"
        "        self._prev = _get_flag()   # snapshot whatever it WAS\n"
        "        _set_flag(False)\n"
        "        return self\n"
        "\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        _set_flag(self._prev)      # restore PREVIOUS value, not True\n"
        "        return False               # don't swallow exceptions"
    ),
    solution_notes=(
        "**Why read the global through `sys.modules[__name__]`.** A naive "
        "`def compute_requires_grad(args): return grad_tracking_enabled and "
        "...` works in a script but breaks in a notebook cell that "
        "re-executes — the closure can stale-bind to the OLD value. Reading "
        "through `globals()` or `sys.modules[__name__]` always sees the "
        "current binding.\n\n"
        "**Why restore the PREVIOUS value, not `True`.** Hardcoding `True` "
        "on exit means nested `NoGrad()` blocks corrupt each other: the "
        "inner exit re-enables grad tracking even though the outer block is "
        "still in scope. The user's nesting-test catches this directly.\n\n"
        "**`return False` from `__exit__`.** Returning a truthy value would "
        "suppress the exception. Returning `False` (or just letting it fall "
        "off the end) lets exceptions propagate naturally — which is what we "
        "want for `try/except` to keep working through `NoGrad` blocks."
    ),
)


# =========================================================================
# atom: requires-grad-propagation  (1 exercise)
# =========================================================================

SPEC_REQUIRES_GRAD = _spec(
    atom_id="requires-grad-propagation",
    subtopic="Backprop: requires_grad propagation",
    recap=RECAP_REQUIRES_GRAD_PROP,
    ex_idx=1,
    ex_title="three-gate requires_grad: toggle AND is_differentiable AND any-input",
    slug="three-gate-requires-grad-toggle-and-diff-and-any-input",
    bloom="Apply",
    difficulty_num=3,
    keywords=["requires-grad", "propagation", "is_differentiable", "any", "three-gate"],
    kcs=["requires-grad-propagation", "grad-tracking-global-toggle"],
    lo=(
        "Apply the three-gate requires_grad rule (global toggle AND op "
        "differentiability AND any-input-requires-grad) and filter "
        "non-Tensor inputs out of the OR-reduction."
    ),
    prompt_body=(
        "Implement `propagate_requires_grad(args, is_differentiable, "
        "grad_tracking_enabled)`. Output `requires_grad` is the AND of "
        "THREE gates — ALL must be true:\n\n"
        "1. `grad_tracking_enabled` — the global no-grad toggle.\n"
        "2. `is_differentiable` — the per-op flag (e.g. `t.equal` registers "
        "with `is_differentiable=False`).\n"
        "3. **At least one input is a Tensor with `requires_grad=True`.**\n"
        "   Use `any(isinstance(a, Tensor) and a.requires_grad for a in "
        "args)`. The `isinstance` guard is critical — without it you'd ask "
        "non-Tensors (ints, floats, shape tuples) for `.requires_grad` and "
        "crash with `AttributeError`.\n\n"
        "Signature: `propagate_requires_grad(args: tuple, is_differentiable: "
        "bool, grad_tracking_enabled: bool) -> bool`.\n\n"
        "Inputs vary in type: some are `Tensor`, some are Python scalars or "
        "tuples. Constants must NOT veto grad — `multiply(t, 3.0)` with `t "
        ".requires_grad=True` should produce a grad-tracked output. They "
        "just don't *contribute* a True to the `any`.\n\n"
        "Test cases exhaustively cover the truth table:\n"
        "- All gates True with at least one tracked Tensor → True.\n"
        "- Any gate False → False (3 scenarios).\n"
        "- Mixed Tensor/non-Tensor where the non-Tensor would crash a naive "
        "implementation."
    ),
    stub=(
        "class Tensor:\n"
        "    def __init__(self, array, requires_grad=False):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "\n"
        "\n"
        "def propagate_requires_grad(\n"
        "    args: tuple,\n"
        "    is_differentiable: bool,\n"
        "    grad_tracking_enabled: bool,\n"
        ") -> bool:\n"
        '    """Three-gate AND: toggle AND is_differentiable AND any(input.requires_grad)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Truth table abbreviations:\n"
        "#   G = grad_tracking_enabled, D = is_differentiable,\n"
        "#   T1 = tensor(rg=True), T0 = tensor(rg=False), 3.0 = a scalar.\n"
        "T1 = Tensor(t.tensor([1.0]), requires_grad=True)\n"
        "T0 = Tensor(t.tensor([1.0]), requires_grad=False)\n"
        "\n"
        "# --- happy path: all three gates True, at least one input tracked ---\n"
        "assert propagate_requires_grad((T1,), True, True) is True\n"
        "assert propagate_requires_grad((T1, T0), True, True) is True, 'any() over tracked+untracked'\n"
        "assert propagate_requires_grad((T0, T1), True, True) is True, 'order-insensitive'\n"
        "\n"
        "# --- gate 1 OFF: global toggle is False ---\n"
        "assert propagate_requires_grad((T1,), True, False) is False, 'toggle off'\n"
        "\n"
        "# --- gate 2 OFF: op is non-differentiable (think torch.equal) ---\n"
        "assert propagate_requires_grad((T1,), False, True) is False, 'is_differentiable=False'\n"
        "\n"
        "# --- gate 3 OFF: no tracked input ---\n"
        "assert propagate_requires_grad((T0,), True, True) is False, 'no input has requires_grad'\n"
        "assert propagate_requires_grad((T0, T0), True, True) is False, 'all untracked'\n"
        "assert propagate_requires_grad((), True, True) is False, 'no inputs at all'\n"
        "\n"
        "# --- non-Tensor inputs must NOT crash and must NOT veto ---\n"
        "# multiply(T1, 3.0) — float at arg-1 should not affect propagation\n"
        "assert propagate_requires_grad((T1, 3.0), True, True) is True, (\n"
        "    'float at arg-1 must not block grad propagation from T1'\n"
        ")\n"
        "# all-constant case is False\n"
        "assert propagate_requires_grad((3.0, 5, 'x', (1, 2)), True, True) is False, (\n"
        "    'all-non-Tensor args → False, no AttributeError'\n"
        ")\n"
        "# mix: shape tuple kwarg-like + tensor\n"
        "assert propagate_requires_grad(((3, 4), T1), True, True) is True\n"
        "\n"
        "# --- combined OFF cases ---\n"
        "for G in (True, False):\n"
        "    for D in (True, False):\n"
        "        for args in [(T0,), (T0, 3.0), (3.0, 5)]:\n"
        "            # gate 3 (any tracked) is False for all these args\n"
        "            assert propagate_requires_grad(args, D, G) is False, (\n"
        "                f'no tracked input → must be False '\n"
        "                f'(G={G}, D={D}, args={args})'\n"
        "            )\n"
        "\n"
        "# --- the AttributeError test: confirm we don't ask scalars for .requires_grad ---\n"
        "class Sneaky:\n"
        '    """Object that raises if .requires_grad is touched."""\n'
        "    def __getattr__(self, name):\n"
        "        if name == 'requires_grad':\n"
        "            raise AttributeError('do not touch requires_grad on non-Tensor')\n"
        "        raise AttributeError(name)\n"
        "\n"
        "# A Sneaky-only call must short-circuit on isinstance — never touch .requires_grad.\n"
        "assert propagate_requires_grad((Sneaky(),), True, True) is False, (\n"
        "    'must use isinstance() guard, not duck-type on .requires_grad'\n"
        ")\n"
        "# Sneaky + T1 — T1 makes the any() True; Sneaky must be skipped, not crash.\n"
        "assert propagate_requires_grad((Sneaky(), T1), True, True) is True"
    ),
    solution_body=(
        "def propagate_requires_grad(\n"
        "    args: tuple,\n"
        "    is_differentiable: bool,\n"
        "    grad_tracking_enabled: bool,\n"
        ") -> bool:\n"
        "    return (\n"
        "        grad_tracking_enabled\n"
        "        and is_differentiable\n"
        "        and any(\n"
        "            isinstance(a, Tensor) and a.requires_grad for a in args\n"
        "        )\n"
        "    )"
    ),
    solution_notes=(
        "**Why AND, not OR.** Each gate is a *necessary* condition. The "
        "global toggle has to be on (otherwise we're in no_grad). The op "
        "has to be differentiable (no point recording a Recipe for "
        "`torch.equal` — gradients don't flow through booleans). And at "
        "least one input has to require grad (otherwise the output is "
        "constant w.r.t. all params — no graph needed).\n\n"
        "**Why `isinstance(a, Tensor) AND a.requires_grad` inside `any`.** "
        "Short-circuit evaluation: `isinstance` is cheap and false for "
        "non-Tensors → `and` skips the `.requires_grad` access. Without the "
        "guard, `propagate_requires_grad((my_tensor, 3.0), ...)` crashes "
        "with `AttributeError: 'float' object has no attribute "
        "'requires_grad'`.\n\n"
        "**`is_differentiable` lives on the op, not the inputs.** "
        "`wrap_forward_fn(torch.equal, is_differentiable=False)` registers "
        "`torch.equal` with the flag set to False on the wrapper. The flag "
        "is plumbed into `propagate_requires_grad` from there, not from any "
        "inspection of the inputs."
    ),
)


# =========================================================================
# atom: unbroadcast-pattern  (1 exercise)
# =========================================================================

SPEC_UNBROADCAST = _spec(
    atom_id="unbroadcast-pattern",
    subtopic="Backprop: Unbroadcast pattern",
    recap=RECAP_UNBROADCAST,
    ex_idx=1,
    ex_title="unbroadcast: sum out leading and size-1 broadcast axes",
    slug="unbroadcast-sum-out-leading-and-size-1-axes",
    bloom="Apply",
    difficulty_num=4,
    keywords=["unbroadcast", "broadcasting", "sum-axes", "keepdim", "binary-back"],
    kcs=["unbroadcast-pattern", "chain-rule-elementwise"],
    lo=(
        "Apply the unbroadcast pattern (peel leading axes, then sum-with-"
        "keepdim across size-1 expanded axes) to restore grad to the "
        "pre-broadcast input shape."
    ),
    prompt_body=(
        "Implement `unbroadcast(grad, original)`. Forward broadcasting can "
        "expand a tensor `original.shape` to a bigger `grad.shape` in two "
        "ways:\n\n"
        "- **(A) leading new axes** — broadcasting added dims to the LEFT. "
        "Example: `original.shape=(3,4)`, `grad.shape=(2,3,4)` → leading "
        "dim of size 2 was added. Sum it out: `grad.sum(dim=0)`.\n"
        "- **(B) size-1 axes that got expanded** — `original` had a "
        "size-1 axis that broadcasting expanded. Example: `original.shape="
        "(1,4)`, `grad.shape=(3,4)` (after step A leaves grad shape "
        "`(3,4)`)... wait that's leading. Try: `original.shape=(3,1,4)`, "
        "`grad.shape=(3,5,4)` → axis 1 was size 1, got expanded to 5. Sum "
        "it out **with keepdim=True**: `grad.sum(dim=1, keepdim=True)`.\n\n"
        "Recipe (do in this order):\n\n"
        "1. While `grad.ndim > original.ndim`: `grad = grad.sum(dim=0)`. "
        "(Peels the leading axes.)\n"
        "2. For each axis `i` in `original.shape`: if `original.shape[i] == "
        "1` AND `grad.shape[i] != 1`, do `grad = grad.sum(dim=i, "
        "keepdim=True)`. (Collapses each expanded size-1 axis back to 1.)\n\n"
        "Final result: `grad.shape == original.shape`. The function is the "
        "RIGHT-INVERSE of broadcasting in the sense that summing out the "
        "broadcasted axes recovers shape compatibility.\n\n"
        "Inputs are plain `torch.Tensor`. No autograd. Return a float "
        "tensor with the same shape as `original`."
    ),
    stub=(
        "def unbroadcast(grad: Tensor, original: Tensor) -> Tensor:\n"
        '    """Sum out axes broadcasting added/expanded so grad.shape matches original.shape."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- no broadcasting: shapes already match → grad unchanged (value+shape) ---\n"
        "g = t.ones(3, 4)\n"
        "x = t.zeros(3, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (3, 4), f'identity shape: {out.shape}'\n"
        "assert t.allclose(out, t.ones(3, 4)), 'identity values'\n"
        "\n"
        "# --- case A: leading axes added ---\n"
        "# original.shape=(3,4), grad.shape=(2,3,4) → sum dim=0 → (3,4)\n"
        "g = t.ones(2, 3, 4)\n"
        "x = t.zeros(3, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (3, 4), f'leading axes shape: {out.shape}'\n"
        "assert t.allclose(out, t.full((3, 4), 2.0)), f'sum value wrong: {out}'\n"
        "\n"
        "# --- case A x2: TWO leading axes added ---\n"
        "g = t.ones(5, 2, 3, 4)\n"
        "x = t.zeros(3, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (3, 4)\n"
        "assert t.allclose(out, t.full((3, 4), 10.0)), 'sum across 5*2=10 wrong'\n"
        "\n"
        "# --- case B: size-1 axis got expanded ---\n"
        "# original.shape=(1,4), grad.shape=(3,4) — but ndim already matches\n"
        "# (this is pure case-B: leading-axes peel does nothing, size-1 collapse fires)\n"
        "g = t.ones(3, 4)\n"
        "x = t.zeros(1, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (1, 4), f'size-1 axis shape: {out.shape}'\n"
        "assert t.allclose(out, t.full((1, 4), 3.0)), f'size-1 value wrong: {out}'\n"
        "\n"
        "# --- case B: middle size-1 axis ---\n"
        "# original.shape=(3,1,4), grad.shape=(3,5,4) — middle axis expanded 1→5\n"
        "g = t.ones(3, 5, 4)\n"
        "x = t.zeros(3, 1, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (3, 1, 4), f'middle size-1 shape: {out.shape}'\n"
        "assert t.allclose(out, t.full((3, 1, 4), 5.0))\n"
        "\n"
        "# --- combined case A + B ---\n"
        "# original.shape=(1,4), grad.shape=(2,3,4)\n"
        "# step 1 peels leading axis → (3,4); step 2 collapses size-1 axis-0 → (1,4)\n"
        "g = t.ones(2, 3, 4)\n"
        "x = t.zeros(1, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (1, 4), f'A+B shape: {out.shape}'\n"
        "assert t.allclose(out, t.full((1, 4), 6.0)), f'A+B value (2*3=6): {out}'\n"
        "\n"
        "# --- scalar case: original is 0-D, grad is anything ---\n"
        "g = t.ones(2, 3)\n"
        "x = t.tensor(0.0)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (), f'scalar shape: {out.shape}'\n"
        "assert t.allclose(out, t.tensor(6.0)), f'scalar value: {out}'\n"
        "\n"
        "# --- value correctness on non-uniform grad ---\n"
        "g = t.tensor([[1.0, 2.0, 3.0, 4.0],\n"
        "              [5.0, 6.0, 7.0, 8.0],\n"
        "              [9.0, 10.0, 11.0, 12.0]])\n"
        "x = t.zeros(1, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (1, 4)\n"
        "assert t.allclose(out, t.tensor([[15.0, 18.0, 21.0, 24.0]])), (\n"
        "    f'column sums wrong: {out}'\n"
        ")\n"
        "\n"
        "# --- AGREEMENT with torch.autograd for a broadcast multiply ---\n"
        "# y = (a * b).sum(); a has shape (1,4), b has shape (3,4) — a is broadcast.\n"
        "a = t.randn(1, 4, generator=t.Generator().manual_seed(7), requires_grad=True)\n"
        "b = t.randn(3, 4, generator=t.Generator().manual_seed(8), requires_grad=True)\n"
        "y = (a * b).sum()\n"
        "y.backward()\n"
        "# our manual: grad_out = ones_like(a*b) = (3,4); dL/da = grad_out * b\n"
        "grad_out = t.ones(3, 4)\n"
        "raw_grad_a = grad_out * b.detach()  # shape (3,4) — wrong shape for `a`\n"
        "our_grad_a = unbroadcast(raw_grad_a, a.detach())\n"
        "assert our_grad_a.shape == a.shape, f'shape post-unbroadcast: {our_grad_a.shape}'\n"
        "assert t.allclose(our_grad_a, a.grad, atol=1e-5), (\n"
        "    f'unbroadcast disagrees with autograd: ours={our_grad_a}, ref={a.grad}'\n"
        ")"
    ),
    solution_body=(
        "def unbroadcast(grad: Tensor, original: Tensor) -> Tensor:\n"
        "    # Step 1: peel leading axes broadcasting added.\n"
        "    # If grad has more dims than original, the EXTRA ones must be on\n"
        "    # the left (broadcasting always prepends 1s), so sum dim=0 repeatedly.\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    # Step 2: collapse size-1 axes that were expanded.\n"
        "    # For each axis where original has size 1 but grad doesn't, sum it\n"
        "    # out with keepdim=True so we keep the size-1 axis instead of dropping it.\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad"
    ),
    solution_notes=(
        "**Step 1 before step 2 — the order matters.** After step 1, "
        "`grad.ndim == original.ndim`, so axes line up positionally for "
        "step 2. If you tried to do the size-1 collapse first you'd be "
        "indexing into mismatched dims.\n\n"
        "**Why `keepdim=True` in step 2.** Without it, `grad.sum(dim=i)` "
        "DROPS that axis, leaving `grad.ndim < original.ndim`. Then the "
        "result wouldn't match `original.shape` (which still has a size-1 "
        "axis at position `i`). With `keepdim=True`, the axis stays as "
        "size 1 — exactly matching.\n\n"
        "**Where this lives in the codebase.** Every binary back fn that "
        "supports broadcasting wraps its result: `return unbroadcast(grad_"
        "out * y, x)` for `multiply_back0`, etc. The wrapper is the only "
        "way the autograd layer survives broadcasting — otherwise a "
        "broadcasted add of a `(1,4)` bias to a `(B,4)` batch would try to "
        "store a `(B,4)` grad on a `(1,4)` parameter and crash."
    ),
)


# =========================================================================
# emit
# =========================================================================

ALL_SPECS = [
    SPEC_CHAIN_RULE,
    SPEC_ARG_POSITION,
    SPEC_KWARGS,
    SPEC_RECIPE,
    SPEC_PARENTS,
    SPEC_GRAD_TRACKING,
    SPEC_REQUIRES_GRAD,
    SPEC_UNBROADCAST,
]


if __name__ == "__main__":
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
