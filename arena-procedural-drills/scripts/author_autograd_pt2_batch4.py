#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA part 4 manual-autograd PT 2 atoms.

Eight single-exercise standalones under ``prereqs_autograd_pt2/``:

  * log-back                      — ex1
  * multiply-back                 — ex1
  * max-back-tied-half            — ex1
  * non-diff-fn-wrap              — ex1
  * is-differentiable-flag        — ex1
  * end-grad-default-ones-like    — ex1
  * backward-func-lookup          — ex1
  * sorted-computational-graph    — ex1

Each drill exercises ONE small constituent of ARENA part 4's manual autograd.
Tests use plain ``torch.Tensor`` for shape/value math; we never call
``torch.autograd`` on the hand-written ops.

Compose with batch-2 (``prereqs_backprop/``: signature / register / wrap-fn /
param-grad / buffer-copy) and batch-3 (``prereqs_autograd_internals/``:
chain-rule / arg-position / kwargs / Recipe / parents / grad-toggle /
requires_grad / unbroadcast). This batch covers the SPECIFIC backward fns
(log_back, multiply_back, max_back with tied half-mass), the wrap-time flags
(is_differentiable, non-diff op wrap), the backward-entry-point defaults
(ones_like end-grad), and the dispatch machinery (BackwardFuncLookup, sorted
computational graph via topological sort).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_pt2"


# ---------------------------------------------------------------- atom recaps

RECAP_LOG_BACK = (
    "## log_back — quick refresher\n"
    "\n"
    "For the elementwise forward `out = log(x)`, the local derivative is "
    "`d/dx log(x) = 1/x`. Chain rule on an elementwise op collapses to a "
    "per-position product:\n"
    "\n"
    "```\n"
    "dL/dx[i] = dL/dout[i] * 1/x[i]\n"
    "        => grad_x   = grad_out / x\n"
    "```\n"
    "\n"
    "Things to notice:\n"
    "- Even though `out` is in the signature, you don't NEED it — `1/x` is "
    "the cleanest form. (You could equivalently use `out` via `exp(-out)`, "
    "but it's slower and numerically worse.)\n"
    "- Shape of `grad_x` always equals shape of `x` (no broadcasting in a "
    "single-arg op).\n"
    "- `log(0)` and `log(negative)` are domain errors at FORWARD time; "
    "`log_back` itself is only safe where the forward was — division by zero "
    "happens if `x` has a 0 anywhere."
)

RECAP_MULTIPLY_BACK = (
    "## multiply_back0 / multiply_back1 — quick refresher\n"
    "\n"
    "Binary elementwise op `out = x * y` registers TWO back fns — one per "
    "argument position — even though the local derivatives are symmetric:\n"
    "\n"
    "```\n"
    "d(x*y)/dx = y          =>  multiply_back0(grad_out, out, x, y) = grad_out * y\n"
    "d(x*y)/dy = x          =>  multiply_back1(grad_out, out, x, y) = grad_out * x\n"
    "```\n"
    "\n"
    "Then wrap the result in `unbroadcast(..., parent)` so the returned grad "
    "matches the ORIGINAL (pre-broadcast) shape of the parent.\n"
    "\n"
    "Two practical wrinkles:\n"
    "- **Scalar floats on either side.** `multiply(t, 3.0)` must work — "
    "coerce the float to a tensor (or just let torch broadcast) so the math "
    "doesn't trip on type mismatches.\n"
    "- **Symmetric registration.** Both bodies are tiny mirror images, but "
    "both still get added to `BACK_FUNCS` at argnums 0 and 1 — the dispatcher "
    "doesn't know multiply is symmetric, it just looks up `(func, argnum)`."
)

RECAP_MAX_BACK_TIED = (
    "## max_back with tied half-mass — quick refresher\n"
    "\n"
    "Forward op `out = maximum(x, y)` is elementwise: per position, pick "
    "whichever of `x[i]` or `y[i]` is larger. The derivative is **piecewise** "
    "and the natural recipe is:\n"
    "\n"
    "```\n"
    "dL/dx = grad_out * (x >  y)     # x is the winner\n"
    "dL/dy = grad_out * (x <  y)     # y is the winner\n"
    "```\n"
    "\n"
    "But what about **ties** (`x == y`)? Sending all the mass to one side is "
    "asymmetric and discontinuous; sending none is wrong (the sum of partials "
    "should equal `grad_out` for the winning value). ARENA's convention is "
    "**split the mass 50/50**:\n"
    "\n"
    "```\n"
    "bool_sum_x = (x > y) + 0.5 * (x == y)\n"
    "bool_sum_y = (x < y) + 0.5 * (x == y)\n"
    "dL/dx = unbroadcast(grad_out * bool_sum_x, x)\n"
    "dL/dy = unbroadcast(grad_out * bool_sum_y, y)\n"
    "```\n"
    "\n"
    "**Invariant.** `bool_sum_x + bool_sum_y == 1` everywhere — gradient mass "
    "is conserved across the two inputs, exactly as it should be for an "
    "operator whose forward picks ONE of the two values."
)

RECAP_NON_DIFF_FN_WRAP = (
    "## non-differentiable fn wrap — quick refresher\n"
    "\n"
    "Some forward ops have no useful gradient — e.g. `torch.eq` returns a "
    "bool, `torch.argmax` returns indices. We still want to call them on "
    "`MiniTensor`s, but we should NOT build a Recipe or set "
    "`requires_grad=True` on the output.\n"
    "\n"
    "Solution: a per-op flag plumbed into the wrapper at register time:\n"
    "\n"
    "```python\n"
    "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
    "    def tensor_func(*args, **kwargs):\n"
    "        ...\n"
    "        requires_grad = (\n"
    "            grad_tracking_enabled\n"
    "            and is_differentiable\n"
    "            and any(isinstance(a, MiniTensor) and a.requires_grad for a in args)\n"
    "        )\n"
    "        out = MiniTensor(out_arr, requires_grad)\n"
    "        if requires_grad:\n"
    "            out.recipe = Recipe(...)   # only built when grad will flow\n"
    "        return out\n"
    "    return tensor_func\n"
    "\n"
    "eq = wrap_forward_fn(torch.eq, is_differentiable=False)\n"
    "```\n"
    "\n"
    "Two effects:\n"
    "- Output of non-diff op is ALWAYS `requires_grad=False`, even if all "
    "inputs are tracked.\n"
    "- No `Recipe` is attached, so reverse pass treats the output as a leaf — "
    "the graph terminates here naturally."
)

RECAP_IS_DIFFERENTIABLE_FLAG = (
    "## is_differentiable flag — quick refresher\n"
    "\n"
    "`is_differentiable` is a **per-op** static flag, supplied at register "
    "time, NOT inferred from inputs or runtime state. It's the second gate in "
    "the three-gate `requires_grad` rule:\n"
    "\n"
    "```\n"
    "requires_grad = (\n"
    "    grad_tracking_enabled        # global toggle (no_grad context)\n"
    "    and is_differentiable        # PER-OP flag, captured in closure\n"
    "    and any(input.requires_grad for input in tensor_inputs)\n"
    ")\n"
    "```\n"
    "\n"
    "Differences from the global toggle:\n"
    "- Global toggle changes at RUNTIME (per `with no_grad():`). Per-op flag "
    "is set ONCE, at wrap-time.\n"
    "- Global toggle gates ALL ops. Per-op flag gates only this one op.\n"
    "\n"
    "Captured via closure: `wrap_forward_fn(fn, is_differentiable=False)` "
    "returns a `tensor_func` whose closure remembers the False, so EVERY "
    "subsequent call short-circuits to `requires_grad=False` for the output."
)

RECAP_END_GRAD_ONES_LIKE = (
    "## end-grad ones_like default — quick refresher\n"
    "\n"
    "PyTorch lets you call `.backward()` on a scalar with no argument:\n"
    "\n"
    "```\n"
    "loss = ...\n"
    "loss.backward()        # same as loss.backward(torch.tensor(1.0))\n"
    "```\n"
    "\n"
    "Inside `backprop(end_node, end_grad=None)` the convention is: if "
    "`end_grad` is `None`, default to `ones_like(end_node.array)` — a tensor "
    "of ones with the same shape as the end node's array.\n"
    "\n"
    "```python\n"
    "def backprop(end_node, end_grad=None):\n"
    "    end_grad_arr = (\n"
    "        torch.ones_like(end_node.array)\n"
    "        if end_grad is None\n"
    "        else end_grad.array\n"
    "    )\n"
    "    ...\n"
    "```\n"
    "\n"
    "Why ones, not zeros? Because `dL/dL = 1` — when `end_node` IS the loss, "
    "its gradient w.r.t. itself is the identity, and 1 is the multiplicative "
    "identity that lets every downstream chain-rule product reduce to the "
    "actual partial.\n"
    "\n"
    "Why `ones_like` (matching shape), not `ones(1)`? Because the end node "
    "may NOT be a scalar — for a `(B,)` per-sample loss vector you want "
    "`dL/dL == eye(B)` collapsed to a `(B,)` of ones."
)

RECAP_BACKWARD_FUNC_LOOKUP = (
    "## BackwardFuncLookup — quick refresher\n"
    "\n"
    "Central registry that maps `(forward_fn, arg_position) -> back_fn`. "
    "Tiny by design — it's just a `dict` with two methods:\n"
    "\n"
    "```python\n"
    "class BackwardFuncLookup:\n"
    "    def __init__(self):\n"
    "        self.back_funcs = {}   # (forward_fn, arg_position) -> back_fn\n"
    "\n"
    "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
    "        self.back_funcs[(forward_fn, arg_position)] = back_fn\n"
    "\n"
    "    def get_back_func(self, forward_fn, arg_position):\n"
    "        return self.back_funcs[(forward_fn, arg_position)]\n"
    "```\n"
    "\n"
    "Why a 2-key `(fn, argnum)` instead of nested dicts? Same lookup cost, "
    "but flatter — registration and dispatch are both `O(1)` one-liners.\n"
    "\n"
    "Symmetric ops still register TWICE (e.g. `add_back0` and `add_back1` "
    "with the same body). The dispatcher in `backprop` does not know which "
    "ops are symmetric — it always asks the lookup for `(func, argnum)` "
    "given the parent's argnum from `recipe.parents`."
)

RECAP_SORTED_GRAPH = (
    "## Sorted computational graph — quick refresher\n"
    "\n"
    "The reverse pass needs the nodes in an order such that **every node "
    "comes BEFORE its parents**, so by the time we pop a node off the "
    "iteration, all the gradients flowing into it have already been "
    "accumulated.\n"
    "\n"
    "Recipe (assumes you already have `topological_sort(node, get_children)` "
    "that returns descendants of `node` in DAG order where `node` is LAST):\n"
    "\n"
    "```python\n"
    "def sorted_computational_graph(tensor: MiniTensor) -> list[MiniTensor]:\n"
    "    def get_parents(t):\n"
    "        if t.recipe is None:\n"
    "            return []\n"
    "        return list(t.recipe.parents.values())\n"
    "    return topological_sort(tensor, get_parents)[::-1]\n"
    "```\n"
    "\n"
    "Two key choices:\n"
    "- **`get_parents` returns `[]` for leaves.** A leaf has no `recipe`; the "
    "traversal stops there.\n"
    "- **Reverse the result.** `topological_sort` ends with the root (the "
    "end node) LAST; the reverse pass wants the root FIRST so it can seed "
    "the grad accumulator and walk backward. Reversal flips the order — `[::-1]` "
    "is the one-liner.\n"
    "\n"
    "After reversal, `result[0] is end_node`, `result[-1]` is some leaf, and "
    "iterating in order means every node's gradient has been summed by the "
    "time the dispatcher needs it."
)


# ---------------------------------------------------------------- spec helper

# Shared autograd preamble — same shape as batch3, gives every drill access to
# Recipe + MiniTensor + the global grad_tracking_enabled toggle so the test
# bodies can construct realistic call sites.
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
# atom: log-back  (1 exercise)
# =========================================================================

SPEC_LOG_BACK = _spec(
    atom_id="log-back",
    subtopic="Backprop: log_back",
    recap=RECAP_LOG_BACK,
    ex_idx=1,
    ex_title="implement log_back from the elementwise chain rule",
    slug="implement-log-back-from-elementwise-chain-rule",
    bloom="Apply",
    difficulty_num=2,
    keywords=["log-back", "elementwise", "chain-rule", "reciprocal"],
    kcs=["log-back", "chain-rule-elementwise"],
    lo=(
        "Apply the elementwise chain rule to derive log_back: grad_x = "
        "grad_out / x, no Jacobian materialized."
    ),
    prompt_body=(
        "Implement `log_back(grad_out, out, x)` — the backward fn for the "
        "forward op `out = log(x)`.\n\n"
        "**The math.** `d/dx log(x) = 1/x`. Elementwise, the Jacobian is "
        "diagonal, so the chain rule reduces to per-position product:\n\n"
        "```\n"
        "dL/dx[i] = dL/dout[i] * (1 / x[i])\n"
        "        => grad_x   = grad_out / x\n"
        "```\n\n"
        "Signature (same uniform `(grad_out, out, *fwd_args)` shape every "
        "back fn uses):\n\n"
        "```python\n"
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    ...\n"
        "```\n\n"
        "**Why `out` is in the signature even though you won't use it.** The "
        "uniform back-fn signature lets the reverse-pass dispatcher call any "
        "back fn the same way: `back_fn(grad_out, out, *recipe.args, "
        "**recipe.kwargs)`. Some back fns (sigmoid_back) need `out`; "
        "log_back doesn't — but the signature is fixed so dispatch is "
        "generic.\n\n"
        "Inputs are plain `torch.Tensor`; no autograd. Return a tensor with "
        "the same shape and float dtype as `x`."
    ),
    stub=(
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = log(x). Returns grad_x = grad_out / x."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- scalar ---\n"
        "x = t.tensor([2.0])\n"
        "out = t.log(x)\n"
        "g = log_back(t.tensor([1.0]), out, x)\n"
        "assert t.allclose(g, t.tensor([0.5])), f'log_back scalar: {g}'\n"
        "\n"
        "# --- vector ---\n"
        "x = t.tensor([1.0, 2.0, 4.0])\n"
        "out = t.log(x)\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "g = log_back(grad_out, out, x)\n"
        "expected = t.tensor([1.0, 0.5, 0.25])\n"
        "assert g.shape == x.shape, f'shape: {g.shape}'\n"
        "assert t.allclose(g, expected), f'log_back value: {g}'\n"
        "\n"
        "# --- non-unit grad_out (chain rule scales each entry) ---\n"
        "grad_out = t.tensor([5.0, -3.0, 2.0])\n"
        "g = log_back(grad_out, out, x)\n"
        "expected = grad_out / x\n"
        "assert t.allclose(g, expected), f'log_back chain rule: {g} vs {expected}'\n"
        "\n"
        "# --- matrix shape preserved ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.rand(3, 4, generator=rng) + 0.5   # keep strictly positive\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "OUT = t.log(X)\n"
        "g = log_back(G, OUT, X)\n"
        "assert g.shape == (3, 4), f'matrix shape: {g.shape}'\n"
        "assert t.allclose(g, G / X), 'matrix chain rule failed'\n"
        "\n"
        "# --- shape MUST equal x.shape, not out.shape (here they coincide, but assert anyway) ---\n"
        "assert g.shape == X.shape, 'grad shape must match input x, not output'\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = t.tensor([0.5, 1.0, 2.5, 7.0], requires_grad=True)\n"
        "y = t.log(x_ref).sum()\n"
        "y.backward()\n"
        "out_cached = t.log(x_ref.detach())\n"
        "g_ours = log_back(t.ones(4), out_cached, x_ref.detach())\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'log_back disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d/dx log(x) = 1/x; chain rule => grad_x = grad_out * (1/x).\n"
        "    # `out` is in the signature only because the uniform back-fn\n"
        "    # signature requires it — we don't read it here.\n"
        "    return grad_out / x"
    ),
    solution_notes=(
        "**Why not `grad_out * (1/x)`.** Same answer, but `grad_out / x` is "
        "one fused op in torch's kernel scheduler — fewer intermediates, "
        "lower peak memory.\n\n"
        "**Why `out` is unused but still passed.** Every back fn in this "
        "library has the SAME signature: `(grad_out, out, *original_fwd_"
        "args, **original_fwd_kwargs)`. The dispatcher in `backprop` "
        "doesn't know which fns need `out` (sigmoid_back uses it heavily) "
        "and which don't (log_back, relu_back). Keeping the signature "
        "uniform means dispatch is one line.\n\n"
        "**Domain.** `log` is only defined for `x > 0`. If the forward "
        "succeeded, `x` is positive everywhere, so `1/x` is finite and "
        "safe. Otherwise the forward would already have produced NaNs."
    ),
)


# =========================================================================
# atom: multiply-back  (1 exercise)
# =========================================================================

SPEC_MULTIPLY_BACK = _spec(
    atom_id="multiply-back",
    subtopic="Backprop: multiply_back",
    recap=RECAP_MULTIPLY_BACK,
    ex_idx=1,
    ex_title="implement multiply_back0 / multiply_back1 with unbroadcast",
    slug="implement-multiply-back0-and-back1-with-unbroadcast",
    bloom="Apply",
    difficulty_num=3,
    keywords=["multiply-back", "binary-op", "unbroadcast", "back0", "back1"],
    kcs=["multiply-back", "arg-position-back-functions"],
    lo=(
        "Apply the per-arg-position binary back-fn pattern to write "
        "multiply_back0 and multiply_back1, wrapping each result in "
        "unbroadcast(grad, parent) so the returned grad matches the "
        "pre-broadcast input shape."
    ),
    prompt_body=(
        "Implement TWO back fns for `out = x * y`. Both must:\n"
        "1. Compute the local derivative w.r.t. the right arg "
        "(`d(x*y)/dx = y`, `d(x*y)/dy = x`).\n"
        "2. Multiply by `grad_out` (the chain rule).\n"
        "3. **Wrap the result in `unbroadcast(grad, parent)`** so it has the "
        "same shape as the parent (pre-broadcast).\n\n"
        "We've given you `unbroadcast(grad, original)` already implemented "
        "for you in the setup cell — it peels leading axes and collapses "
        "size-1 expanded axes via `sum(dim=i, keepdim=True)`.\n\n"
        "Signatures:\n\n"
        "```python\n"
        "def multiply_back0(grad_out, out, x, y) -> Tensor:   # dL/dx\n"
        "def multiply_back1(grad_out, out, x, y) -> Tensor:   # dL/dy\n"
        "```\n\n"
        "**Float-input bonus.** Either of `x` or `y` may be a Python float "
        "(`multiply(t, 3.0)` is a valid call). Coerce floats to tensors via "
        "`torch.tensor(...)` so `unbroadcast` and the broadcasting math don't "
        "trip — OR just let torch broadcast and don't call `unbroadcast` when "
        "the parent isn't a tensor.\n\n"
        "Inputs are plain `torch.Tensor` (or Python float). No autograd. "
        "Return tensors that match the corresponding parent's shape."
    ),
    stub=(
        "def unbroadcast(grad: Tensor, original: Tensor) -> Tensor:\n"
        "    # Provided helper — sums out axes broadcasting added/expanded.\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y) -> Tensor:\n"
        '    """dL/dx for out = x * y. Returns a tensor shaped like x."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y) -> Tensor:\n"
        '    """dL/dy for out = x * y. Returns a tensor shaped like y."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- vector, same shape, unit grad_out ---\n"
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y = t.tensor([5.0, 6.0, 7.0])\n"
        "out = x * y\n"
        "g0 = multiply_back0(t.ones(3), out, x, y)\n"
        "g1 = multiply_back1(t.ones(3), out, x, y)\n"
        "assert g0.shape == x.shape, f'g0 shape: {g0.shape}'\n"
        "assert g1.shape == y.shape, f'g1 shape: {g1.shape}'\n"
        "assert t.allclose(g0, y), f'multiply_back0 (=y) wrong: {g0}'\n"
        "assert t.allclose(g1, x), f'multiply_back1 (=x) wrong: {g1}'\n"
        "\n"
        "# --- non-unit grad_out ---\n"
        "grad_out = t.tensor([10.0, 100.0, 1000.0])\n"
        "g0 = multiply_back0(grad_out, out, x, y)\n"
        "g1 = multiply_back1(grad_out, out, x, y)\n"
        "assert t.allclose(g0, grad_out * y), f'chain g0: {g0}'\n"
        "assert t.allclose(g1, grad_out * x), f'chain g1: {g1}'\n"
        "\n"
        "# --- broadcasting case: x is (1,4), y is (3,4), out is (3,4) ---\n"
        "x_b = t.tensor([[1.0, 2.0, 3.0, 4.0]])           # (1,4)\n"
        "y_b = t.tensor([[5.0, 6.0, 7.0, 8.0],\n"
        "                [9.0, 10.0, 11.0, 12.0],\n"
        "                [13.0, 14.0, 15.0, 16.0]])        # (3,4)\n"
        "out_b = x_b * y_b                                  # (3,4)\n"
        "grad_out = t.ones(3, 4)\n"
        "g0_b = multiply_back0(grad_out, out_b, x_b, y_b)\n"
        "g1_b = multiply_back1(grad_out, out_b, x_b, y_b)\n"
        "assert g0_b.shape == x_b.shape, (\n"
        "    f'g0 shape should match x_b={x_b.shape}, got {g0_b.shape} '\n"
        "    f'(did you forget to call unbroadcast?)'\n"
        ")\n"
        "assert g1_b.shape == y_b.shape, f'g1 shape: {g1_b.shape}'\n"
        "# column sums of y_b: 5+9+13=27, etc.\n"
        "expected_g0 = (grad_out * y_b).sum(dim=0, keepdim=True)\n"
        "assert t.allclose(g0_b, expected_g0), f'broadcast g0: {g0_b} vs {expected_g0}'\n"
        "assert t.allclose(g1_b, grad_out * x_b.expand_as(y_b)), 'broadcast g1'\n"
        "\n"
        "# --- agreement with torch.autograd on broadcast multiply ---\n"
        "x_ref = t.randn(1, 4, generator=t.Generator().manual_seed(7), requires_grad=True)\n"
        "y_ref = t.randn(3, 4, generator=t.Generator().manual_seed(8), requires_grad=True)\n"
        "loss = (x_ref * y_ref).sum()\n"
        "loss.backward()\n"
        "out_cached = x_ref.detach() * y_ref.detach()\n"
        "g0_ours = multiply_back0(t.ones(3, 4), out_cached, x_ref.detach(), y_ref.detach())\n"
        "g1_ours = multiply_back1(t.ones(3, 4), out_cached, x_ref.detach(), y_ref.detach())\n"
        "assert t.allclose(g0_ours, x_ref.grad, atol=1e-5), (\n"
        "    f'multiply_back0 disagrees with autograd: ours={g0_ours}, ref={x_ref.grad}'\n"
        ")\n"
        "assert t.allclose(g1_ours, y_ref.grad, atol=1e-5), (\n"
        "    f'multiply_back1 disagrees with autograd: ours={g1_ours}, ref={y_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def multiply_back0(grad_out, out, x, y) -> Tensor:\n"
        "    # d(x*y)/dx = y, chain rule => grad_out * y; then collapse any\n"
        "    # broadcast axes back to x's original shape.\n"
        "    if not isinstance(y, Tensor):\n"
        "        y = t.tensor(y)\n"
        "    return unbroadcast(grad_out * y, x)\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y) -> Tensor:\n"
        "    # d(x*y)/dy = x, chain rule => grad_out * x.\n"
        "    if not isinstance(x, Tensor):\n"
        "        x = t.tensor(x)\n"
        "    return unbroadcast(grad_out * x, y)"
    ),
    solution_notes=(
        "**Why unbroadcast matters even on this simple op.** PyTorch lets "
        "`x * y` broadcast — `(1,4) * (3,4) -> (3,4)`. The reverse pass "
        "computes `grad_out * y` which has shape `(3,4)`, but the parent `x` "
        "is `(1,4)`. Without `unbroadcast`, you'd try to accumulate a "
        "`(3,4)` grad into a `(1,4)` `.grad` slot and crash.\n\n"
        "**Coercing the scalar.** `multiply(t, 3.0)` is allowed — the float "
        "isn't a `Tensor`. `multiply_back0` still needs `y` to be tensor-like "
        "for the multiplication. The `isinstance` guard handles both paths "
        "without forcing the caller to pre-coerce.\n\n"
        "**Symmetric registration in BACK_FUNCS.** Even though `multiply` "
        "is mathematically symmetric (`x*y == y*x`), we register both "
        "`(np.multiply, 0) -> multiply_back0` and `(np.multiply, 1) -> "
        "multiply_back1`. The dispatcher always looks up by `(func, argnum)` "
        "— it doesn't know which ops are symmetric."
    ),
)


# =========================================================================
# atom: max-back-tied-half  (1 exercise)
# =========================================================================

SPEC_MAX_BACK_TIED = _spec(
    atom_id="max-back-tied-half",
    subtopic="Backprop: max_back with tied half-mass",
    recap=RECAP_MAX_BACK_TIED,
    ex_idx=1,
    ex_title="maximum_back with 50/50 tie-splitting",
    slug="maximum-back-with-50-50-tie-splitting",
    bloom="Apply",
    difficulty_num=3,
    keywords=["maximum-back", "ties", "half-mass", "subgradient", "piecewise"],
    kcs=["max-back-tied-half", "unbroadcast-pattern"],
    lo=(
        "Apply the half-mass tie-splitting convention to derive "
        "maximum_back0 and maximum_back1 such that gradient mass is "
        "conserved across the two inputs at every position."
    ),
    prompt_body=(
        "Implement `maximum_back0(grad_out, out, x, y)` and "
        "`maximum_back1(grad_out, out, x, y)` for the elementwise op "
        "`out = maximum(x, y)`.\n\n"
        "Per position, three cases:\n\n"
        "1. **`x[i] > y[i]`** — `x` won. `dout/dx = 1`, `dout/dy = 0`.\n"
        "2. **`x[i] < y[i]`** — `y` won. `dout/dx = 0`, `dout/dy = 1`.\n"
        "3. **`x[i] == y[i]`** — tie. ARENA convention: **split the mass "
        "50/50** so `dout/dx = 0.5`, `dout/dy = 0.5`.\n\n"
        "Recipe:\n\n"
        "```python\n"
        "bool_sum_x = (x > y).float() + 0.5 * (x == y).float()\n"
        "bool_sum_y = (x < y).float() + 0.5 * (x == y).float()\n"
        "dL/dx = unbroadcast(grad_out * bool_sum_x, x)\n"
        "dL/dy = unbroadcast(grad_out * bool_sum_y, y)\n"
        "```\n\n"
        "**Why the tie matters.** A hardcoded `dout/dx = (x > y)` and "
        "`dout/dy = (x <= y)` would pass the strict-tie case but is "
        "asymmetric (it favours `y`). Half-mass is the only choice that "
        "(a) is symmetric in `x` and `y`, and (b) **conserves mass**: "
        "`bool_sum_x + bool_sum_y == 1` everywhere, which means "
        "`dL/dx + dL/dy == grad_out` at every position (relative to the "
        "winning input). This is the right invariant — the forward picks "
        "one value, so the backward should distribute the grad onto "
        "whoever contributed.\n\n"
        "We've also provided `unbroadcast(grad, original)` in the cell — "
        "wrap each return so broadcasting between `x` and `y` is handled.\n\n"
        "Inputs are plain `torch.Tensor`. No autograd."
    ),
    stub=(
        "def unbroadcast(grad: Tensor, original: Tensor) -> Tensor:\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "\n"
        "\n"
        "def maximum_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dx for out = maximum(x, y), with 50/50 tie-splitting."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def maximum_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dy for out = maximum(x, y), with 50/50 tie-splitting."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- strict case: x[i] > y[i] for all i ---\n"
        "x = t.tensor([5.0, 7.0, 9.0])\n"
        "y = t.tensor([1.0, 2.0, 3.0])\n"
        "out = t.maximum(x, y)\n"
        "g0 = maximum_back0(t.ones(3), out, x, y)\n"
        "g1 = maximum_back1(t.ones(3), out, x, y)\n"
        "assert t.allclose(g0, t.ones(3)), f'x-wins g0: {g0}'\n"
        "assert t.allclose(g1, t.zeros(3)), f'x-wins g1: {g1}'\n"
        "\n"
        "# --- strict case: y wins ---\n"
        "x = t.tensor([1.0, 2.0, 3.0])\n"
        "y = t.tensor([5.0, 7.0, 9.0])\n"
        "out = t.maximum(x, y)\n"
        "g0 = maximum_back0(t.ones(3), out, x, y)\n"
        "g1 = maximum_back1(t.ones(3), out, x, y)\n"
        "assert t.allclose(g0, t.zeros(3)), f'y-wins g0: {g0}'\n"
        "assert t.allclose(g1, t.ones(3)), f'y-wins g1: {g1}'\n"
        "\n"
        "# --- pure tie: x == y everywhere, mass splits 50/50 ---\n"
        "x = t.tensor([3.0, 3.0, 3.0])\n"
        "y = t.tensor([3.0, 3.0, 3.0])\n"
        "out = t.maximum(x, y)\n"
        "g0 = maximum_back0(t.ones(3), out, x, y)\n"
        "g1 = maximum_back1(t.ones(3), out, x, y)\n"
        "assert t.allclose(g0, t.full((3,), 0.5)), f'tie g0 (should be 0.5): {g0}'\n"
        "assert t.allclose(g1, t.full((3,), 0.5)), f'tie g1 (should be 0.5): {g1}'\n"
        "# Conservation: g0 + g1 == grad_out everywhere.\n"
        "assert t.allclose(g0 + g1, t.ones(3)), 'mass conservation broke at ties'\n"
        "\n"
        "# --- mixed: x wins, y wins, tie all in one tensor ---\n"
        "x = t.tensor([5.0, 1.0, 3.0])\n"
        "y = t.tensor([1.0, 5.0, 3.0])\n"
        "out = t.maximum(x, y)\n"
        "grad_out = t.tensor([10.0, 20.0, 40.0])\n"
        "g0 = maximum_back0(grad_out, out, x, y)\n"
        "g1 = maximum_back1(grad_out, out, x, y)\n"
        "assert t.allclose(g0, t.tensor([10.0, 0.0, 20.0])), f'mixed g0: {g0}'\n"
        "assert t.allclose(g1, t.tensor([0.0, 20.0, 20.0])), f'mixed g1: {g1}'\n"
        "# Conservation at every position.\n"
        "assert t.allclose(g0 + g1, grad_out), 'mass conservation across the whole tensor'\n"
        "\n"
        "# --- broadcasting: x is (1,4), y is (3,4) ---\n"
        "x_b = t.tensor([[1.0, 5.0, 3.0, 8.0]])\n"
        "y_b = t.tensor([[3.0, 5.0, 4.0, 2.0],\n"
        "                [2.0, 5.0, 1.0, 7.0],\n"
        "                [4.0, 5.0, 6.0, 6.0]])\n"
        "out_b = t.maximum(x_b, y_b)\n"
        "g0_b = maximum_back0(t.ones(3, 4), out_b, x_b, y_b)\n"
        "g1_b = maximum_back1(t.ones(3, 4), out_b, x_b, y_b)\n"
        "assert g0_b.shape == x_b.shape, f'broadcast g0 shape: {g0_b.shape}'\n"
        "assert g1_b.shape == y_b.shape, f'broadcast g1 shape: {g1_b.shape}'\n"
        "# Column-1 is all ties (x=5, y=5) → 0.5 contribution from EACH of 3 rows = 1.5 on x's slot.\n"
        "assert t.allclose(g0_b[0, 1], t.tensor(1.5)), f'tie column on x: {g0_b[0, 1]}'\n"
        "\n"
        "# --- per-position conservation pre-unbroadcast ---\n"
        "# At positions in (out_b == x_b) ∩ (out_b == y_b) we should see g0+g1 sum to grad_out,\n"
        "# even though after unbroadcast the per-position numbers shift.\n"
        "raw_g0 = t.ones(3, 4) * ((x_b > y_b).float() + 0.5 * (x_b == y_b).float())\n"
        "raw_g1 = t.ones(3, 4) * ((x_b < y_b).float() + 0.5 * (x_b == y_b).float())\n"
        "assert t.allclose(raw_g0 + raw_g1, t.ones(3, 4)), 'half-mass invariant violated'"
    ),
    solution_body=(
        "def maximum_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # x's share of the grad: 1 where x > y, 0 where x < y, 0.5 at ties.\n"
        "    bool_sum = (x > y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype)\n"
        "    return unbroadcast(grad_out * bool_sum, x)\n"
        "\n"
        "\n"
        "def maximum_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # y's share: 1 where x < y, 0 where x > y, 0.5 at ties.\n"
        "    bool_sum = (x < y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype)\n"
        "    return unbroadcast(grad_out * bool_sum, y)"
    ),
    solution_notes=(
        "**Why half-mass is the right convention.** The subgradient of "
        "`max(x, y)` at `x == y` is the whole convex hull of the two "
        "one-sided derivatives — any `λ * 1_x + (1-λ) * 1_y` for "
        "`λ ∈ [0, 1]` is valid. `λ = 0.5` is the only choice that's "
        "symmetric in `x` and `y` and treats ties as the limit of the "
        "averaging procedure.\n\n"
        "**Mass conservation as the diagnostic.** `bool_sum_x + bool_sum_y "
        "== 1` everywhere — if your implementation breaks this (e.g. uses "
        "`(x >= y)` and `(x <= y)`, which double-count ties to 1+1=2), the "
        "reverse pass over-counts. The mass-conservation test in the "
        "harness catches this directly.\n\n"
        "**`.to(grad_out.dtype)` casts.** `(x > y)` returns a `bool` "
        "tensor; multiplying it by a float still works in torch, but the "
        "intermediate `bool_sum` would have weird dtypes (mixing bool and "
        "float). Explicit `.to(grad_out.dtype)` keeps everything in the "
        "same float space and avoids surprise promotions.\n\n"
        "**ReLU as a special case.** `relu(x) = maximum(x, 0)` — so "
        "`relu_back` is `maximum_back0` with `y = 0`. The tie-splitting "
        "convention means `relu_back(0)` returns `0.5 * grad_out` "
        "(controversial but consistent with this library)."
    ),
)


# =========================================================================
# atom: non-diff-fn-wrap  (1 exercise)
# =========================================================================

SPEC_NON_DIFF = _spec(
    atom_id="non-diff-fn-wrap",
    subtopic="Backprop: non-differentiable fn wrap",
    recap=RECAP_NON_DIFF_FN_WRAP,
    ex_idx=1,
    ex_title="wrap a non-differentiable op (eq) — no Recipe, no requires_grad",
    slug="wrap-non-differentiable-op-no-recipe-no-requires-grad",
    bloom="Apply",
    difficulty_num=3,
    keywords=["non-differentiable", "eq", "argmax", "no-recipe", "wrap"],
    kcs=["non-diff-fn-wrap", "is-differentiable-flag"],
    lo=(
        "Apply the non-differentiable-op wrap path: when is_differentiable "
        "is False, return a MiniTensor whose requires_grad is False and "
        "whose recipe is None, regardless of the inputs."
    ),
    prompt_body=(
        "Implement `wrap_forward_fn(fwd_fn, is_differentiable=True)`. It "
        "must:\n\n"
        "1. Unbox MiniTensor inputs to their raw `.array`s (pass non-Tensors "
        "through unchanged).\n"
        "2. Call `fwd_fn(*raw_args, **kwargs)`.\n"
        "3. Box the result as a `MiniTensor`. Compute `requires_grad` via the "
        "**three-gate AND**: `grad_tracking_enabled AND is_differentiable "
        "AND any(input is a tracked MiniTensor)`.\n"
        "4. **Only if `requires_grad` is True**, attach a `Recipe`. Otherwise "
        "leave `recipe = None`.\n\n"
        "**Why the conditional Recipe matters.** For `eq = wrap_forward_fn"
        "(torch.eq, is_differentiable=False)`, the output is a `bool` "
        "tensor — there is no gradient to compute, and we don't want the "
        "reverse pass to try to traverse `eq`'s parents. Setting "
        "`recipe = None` makes the output behave like a leaf in the "
        "computational graph — backprop stops here.\n\n"
        "The setup cell provides `MiniTensor`, `Recipe`, and the module-level "
        "`grad_tracking_enabled = True`. After you implement "
        "`wrap_forward_fn`, the test wraps THREE ops:\n"
        "- `add = wrap_forward_fn(torch.add)` — differentiable.\n"
        "- `eq = wrap_forward_fn(torch.eq, is_differentiable=False)` — "
        "  not differentiable.\n"
        "- `argmax = wrap_forward_fn(torch.argmax, is_differentiable=False)` — "
        "  not differentiable.\n\n"
        "And checks the requires_grad / recipe state on each.\n\n"
        "Don't call `torch.autograd`."
    ),
    stub=(
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        '    """Return a tensor_func that boxes/unboxes around fwd_fn.\n'
        "\n"
        "    When is_differentiable=False, output.requires_grad is False and\n"
        "    output.recipe is None regardless of input requires_grad.\n"
        '    """\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Reset the global toggle — earlier exercises in the session may have left it off.\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "\n"
        "add    = wrap_forward_fn(t.add)\n"
        "eq     = wrap_forward_fn(t.eq, is_differentiable=False)\n"
        "argmax = wrap_forward_fn(t.argmax, is_differentiable=False)\n"
        "\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([1.0, 0.0, 3.0]), requires_grad=True)\n"
        "\n"
        "# --- differentiable op: requires_grad propagates, Recipe is built ---\n"
        "out_add = add(a, b)\n"
        "assert isinstance(out_add, MiniTensor)\n"
        "assert t.allclose(out_add.array, t.tensor([2.0, 2.0, 6.0]))\n"
        "assert out_add.requires_grad is True, 'differentiable + tracked inputs → True'\n"
        "assert out_add.recipe is not None, 'Recipe must be built when requires_grad is True'\n"
        "assert out_add.recipe.func is t.add\n"
        "assert 0 in out_add.recipe.parents and 1 in out_add.recipe.parents\n"
        "\n"
        "# --- non-differentiable op (eq): requires_grad False, recipe None ---\n"
        "out_eq = eq(a, b)\n"
        "assert isinstance(out_eq, MiniTensor)\n"
        "# torch.eq returns a 0-D bool tensor.\n"
        "assert out_eq.array.dtype == t.bool, f'eq output should be bool: {out_eq.array.dtype}'\n"
        "assert out_eq.requires_grad is False, (\n"
        "    'non-differentiable op MUST produce requires_grad=False, '\n"
        "    f'got {out_eq.requires_grad}'\n"
        ")\n"
        "assert out_eq.recipe is None, (\n"
        "    'non-differentiable op MUST leave recipe=None (so backprop stops here), '\n"
        "    f'got {out_eq.recipe}'\n"
        ")\n"
        "\n"
        "# --- non-differentiable op (argmax): same story ---\n"
        "c = MiniTensor(t.tensor([3.0, 1.0, 4.0, 1.0, 5.0]), requires_grad=True)\n"
        "out_ax = argmax(c)\n"
        "assert out_ax.array.item() == 4, f'argmax wrong: {out_ax.array}'\n"
        "assert out_ax.requires_grad is False\n"
        "assert out_ax.recipe is None\n"
        "\n"
        "# --- non-differentiable + UNtracked inputs → also False / None ---\n"
        "a2 = MiniTensor(t.tensor([1.0, 2.0]), requires_grad=False)\n"
        "b2 = MiniTensor(t.tensor([1.0, 2.0]), requires_grad=False)\n"
        "out_eq2 = eq(a2, b2)\n"
        "assert out_eq2.requires_grad is False and out_eq2.recipe is None\n"
        "\n"
        "# --- differentiable + UNtracked inputs → requires_grad False, recipe None ---\n"
        "out_add2 = add(a2, b2)\n"
        "assert out_add2.requires_grad is False, 'no tracked inputs → requires_grad False'\n"
        "assert out_add2.recipe is None, 'no tracked inputs → no recipe (skip the build)'\n"
        "\n"
        "# --- when the global toggle is OFF, even differentiable + tracked stays False ---\n"
        "globals()['grad_tracking_enabled'] = False\n"
        "try:\n"
        "    out_add3 = add(a, b)\n"
        "    assert out_add3.requires_grad is False, 'toggle OFF → False'\n"
        "    assert out_add3.recipe is None, 'toggle OFF → no recipe'\n"
        "finally:\n"
        "    globals()['grad_tracking_enabled'] = True"
    ),
    solution_body=(
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        # 1. unbox\n"
        "        raw_args = tuple(\n"
        "            a.array if isinstance(a, MiniTensor) else a for a in args\n"
        "        )\n"
        "        # 2. forward call\n"
        "        out_arr = fwd_fn(*raw_args, **kwargs)\n"
        "        # 3. three-gate AND (read the global toggle FRESH each call)\n"
        "        requires_grad = (\n"
        "            globals()['grad_tracking_enabled']\n"
        "            and is_differentiable\n"
        "            and any(\n"
        "                isinstance(a, MiniTensor) and a.requires_grad for a in args\n"
        "            )\n"
        "        )\n"
        "        out = MiniTensor(out_arr, requires_grad)\n"
        "        # 4. conditional Recipe — only when grad will flow\n"
        "        if requires_grad:\n"
        "            parents = {\n"
        "                idx: a for idx, a in enumerate(args) if isinstance(a, MiniTensor)\n"
        "            }\n"
        "            out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    solution_notes=(
        "**Why skip Recipe entirely (vs. building it with `requires_grad="
        "False`).** A node with `recipe=None` is treated by the reverse pass "
        "as a leaf — `sorted_computational_graph` stops there. If you build "
        "the Recipe anyway, the traversal continues past it, walks into "
        "non-differentiable ancestors, and either crashes (no back fn "
        "registered for `torch.eq`) or wastes work.\n\n"
        "**`is_differentiable` is a CLOSURE-captured constant.** "
        "`wrap_forward_fn(torch.eq, is_differentiable=False)` returns a "
        "`tensor_func` whose closure remembers `False`. Every subsequent "
        "call short-circuits on gate 2. You CANNOT change this at runtime "
        "without re-wrapping the op — which is the point: differentiability "
        "is a property of the op, not the call site.\n\n"
        "**Why read `grad_tracking_enabled` via `globals()`.** A naive "
        "bare reference closes over the value at function-definition time. "
        "Re-binding the module-level name (e.g. via a `NoGrad` context "
        "manager) wouldn't change the closure's value. `globals()['...']` "
        "always reads the current binding."
    ),
)


# =========================================================================
# atom: is-differentiable-flag  (1 exercise)
# =========================================================================

SPEC_IS_DIFF_FLAG = _spec(
    atom_id="is-differentiable-flag",
    subtopic="Backprop: is_differentiable flag",
    recap=RECAP_IS_DIFFERENTIABLE_FLAG,
    ex_idx=1,
    ex_title="three-gate requires_grad reading is_differentiable from closure",
    slug="three-gate-requires-grad-reading-is-differentiable-from-closure",
    bloom="Apply",
    difficulty_num=3,
    keywords=["is-differentiable", "three-gate", "closure", "per-op"],
    kcs=["is-differentiable-flag", "requires-grad-propagation"],
    lo=(
        "Apply the three-gate requires_grad rule with the is_differentiable "
        "flag closed over at wrap-time, distinguishing per-op behavior from "
        "the global runtime toggle."
    ),
    prompt_body=(
        "Implement TWO related pieces:\n\n"
        "**1. `make_check_requires_grad(is_differentiable)`** — a *factory* "
        "that captures the per-op `is_differentiable` flag in closure and "
        "returns a function `check(args) -> bool`. `check` reads "
        "`grad_tracking_enabled` from the module globals each call, ANDs all "
        "three gates, and returns the result.\n\n"
        "Pseudocode:\n\n"
        "```python\n"
        "def make_check_requires_grad(is_differentiable):\n"
        "    def check(args):\n"
        "        return (\n"
        "            globals()['grad_tracking_enabled']\n"
        "            and is_differentiable\n"
        "            and any(\n"
        "                isinstance(a, MiniTensor) and a.requires_grad for a in args\n"
        "            )\n"
        "        )\n"
        "    return check\n"
        "```\n\n"
        "**2. `set_grad_tracking(enabled: bool)`** — a tiny helper that "
        "writes `grad_tracking_enabled` in the module globals. Use "
        "`globals()['grad_tracking_enabled'] = enabled`.\n\n"
        "**The point of the closure.** `is_differentiable` is set ONCE per "
        "op (at wrap-time, via the factory). After that it's fixed — you "
        "cannot change it at call-site. By contrast, `grad_tracking_enabled` "
        "is read FRESH each call, so flipping the global instantly affects "
        "every subsequent `check`.\n\n"
        "Test exhaustively covers:\n"
        "- per-op flag stays sticky across calls.\n"
        "- runtime toggle flips behavior on/off mid-program.\n"
        "- mixed factory invocations don't cross-contaminate.\n"
        "- gate 3 (any input tracked) is unchanged.\n\n"
        "Do NOT call `torch.autograd`."
    ),
    stub=(
        "def make_check_requires_grad(is_differentiable: bool):\n"
        '    """Factory: closes over is_differentiable, returns check(args) -> bool."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def set_grad_tracking(enabled: bool):\n"
        '    """Write grad_tracking_enabled in the module globals."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Reset toggle (earlier cells may have left it off).\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "\n"
        "check_diff    = make_check_requires_grad(is_differentiable=True)\n"
        "check_nondiff = make_check_requires_grad(is_differentiable=False)\n"
        "\n"
        "T1 = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "T0 = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "\n"
        "# --- happy path: all three gates ---\n"
        "assert check_diff((T1,)) is True, 'all gates True → True'\n"
        "assert check_diff((T1, T0)) is True, 'any-tracked covers'\n"
        "assert check_diff((T0,)) is False, 'gate 3 fails (no tracked input)'\n"
        "assert check_diff(()) is False, 'empty args → False'\n"
        "\n"
        "# --- gate 2 (per-op): non-diff factory is sticky False ---\n"
        "assert check_nondiff((T1,)) is False, 'is_differentiable=False → always False'\n"
        "assert check_nondiff((T1, T1)) is False, 'still False with extra tracked inputs'\n"
        "\n"
        "# --- gate 1 (global): toggle FLIPS check_diff at runtime ---\n"
        "set_grad_tracking(False)\n"
        "assert globals()['grad_tracking_enabled'] is False, 'set_grad_tracking did not write the global'\n"
        "assert check_diff((T1,)) is False, 'toggle OFF should turn check_diff False'\n"
        "set_grad_tracking(True)\n"
        "assert globals()['grad_tracking_enabled'] is True\n"
        "assert check_diff((T1,)) is True, 'toggle ON restores check_diff True'\n"
        "\n"
        "# --- gate 2 is NOT toggled by set_grad_tracking — it's per-op, captured at factory time ---\n"
        "set_grad_tracking(True)\n"
        "assert check_nondiff((T1,)) is False, (\n"
        "    'non-diff stays False even when the global toggle is on '\n"
        "    '(is_differentiable is per-op, not global)'\n"
        ")\n"
        "\n"
        "# --- two factory invocations do NOT cross-contaminate ---\n"
        "# If you mistakenly used a module-level variable instead of closure, the second\n"
        "# call would overwrite the first.\n"
        "c1 = make_check_requires_grad(is_differentiable=True)\n"
        "c2 = make_check_requires_grad(is_differentiable=False)\n"
        "assert c1((T1,)) is True\n"
        "assert c2((T1,)) is False\n"
        "# c1 still True after c2 was made — proves the flag is per-closure, not shared.\n"
        "assert c1((T1,)) is True, 'closure capture broke — second factory leaked into first'\n"
        "\n"
        "# --- non-Tensor inputs do not crash gate 3 ---\n"
        "assert check_diff((T1, 3.0, 'x', (1, 2))) is True\n"
        "assert check_diff((3.0, 'x', (1, 2))) is False"
    ),
    solution_body=(
        "def make_check_requires_grad(is_differentiable: bool):\n"
        "    # Closure captures is_differentiable — sticky for the lifetime of `check`.\n"
        "    def check(args):\n"
        "        return (\n"
        "            globals()['grad_tracking_enabled']    # gate 1: runtime\n"
        "            and is_differentiable                  # gate 2: per-op (closure)\n"
        "            and any(                               # gate 3: any tracked input\n"
        "                isinstance(a, MiniTensor) and a.requires_grad\n"
        "                for a in args\n"
        "            )\n"
        "        )\n"
        "    return check\n"
        "\n"
        "\n"
        "def set_grad_tracking(enabled: bool):\n"
        "    globals()['grad_tracking_enabled'] = enabled"
    ),
    solution_notes=(
        "**Why a factory.** The cleanest way to make a per-op flag sticky "
        "is to bind it via closure — `is_differentiable` lives in the "
        "`check` function's enclosing scope and is unreachable from "
        "outside. That's the same mechanism `wrap_forward_fn(fn, "
        "is_differentiable=False)` uses to make the per-op flag persist "
        "across all calls to `eq` or `argmax`.\n\n"
        "**Why three gates, all ANDed.** Each is a NECESSARY condition for "
        "building a Recipe:\n"
        "- Gate 1 (global): user disabled grad tracking → no graph.\n"
        "- Gate 2 (per-op): op is fundamentally non-differentiable → no "
        "graph through this node.\n"
        "- Gate 3 (any input): inputs are all constants → output is "
        "constant → no graph needed.\n\n"
        "If ANY one fails, the output is a leaf with no Recipe, and "
        "backprop simply stops at it.\n\n"
        "**Why `globals()` not a bare name.** If `check` is defined inside "
        "the factory, a bare `grad_tracking_enabled` reference would "
        "first look in the factory's locals (not found), then the module "
        "globals — but the binding is captured by NAME, not by value. "
        "Re-assigning the module-level name via `set_grad_tracking` is "
        "still seen correctly. The explicit `globals()['...']` makes the "
        "intent obvious and works even if the function is moved into a "
        "different scope later."
    ),
)


# =========================================================================
# atom: end-grad-default-ones-like  (1 exercise)
# =========================================================================

SPEC_END_GRAD = _spec(
    atom_id="end-grad-default-ones-like",
    subtopic="Backprop: end-grad ones_like default",
    recap=RECAP_END_GRAD_ONES_LIKE,
    ex_idx=1,
    ex_title="resolve end_grad — default ones_like, else use .array",
    slug="resolve-end-grad-default-ones-like-else-use-array",
    bloom="Apply",
    difficulty_num=2,
    keywords=["end-grad", "ones-like", "backward", "default", "shape-matching"],
    kcs=["end-grad-default-ones-like", "buffer-copy-inplace"],
    lo=(
        "Apply the .backward() entry-point convention: when no end_grad is "
        "supplied, default to torch.ones_like(end_node.array); otherwise "
        "use end_grad.array as-is."
    ),
    prompt_body=(
        "Implement `resolve_end_grad(end_node, end_grad)`. This is the FIRST "
        "thing `backprop(end_node, end_grad=None)` does — figure out the seed "
        "gradient that starts the reverse pass.\n\n"
        "Rules:\n\n"
        "1. **`end_grad is None` → return `torch.ones_like(end_node.array)`.** "
        "Same shape, same dtype as `end_node.array`, all ones. This handles "
        "the common `loss.backward()` call where the user means `dL/dL = 1`.\n\n"
        "2. **`end_grad` is a `MiniTensor` → return `end_grad.array`.** "
        "Unbox; the rest of `backprop` works with raw arrays internally. The "
        "shape and dtype must match `end_node.array` — if they don't, raise "
        "`AssertionError` with a helpful message:\n\n"
        "```python\n"
        "assert end_grad.array.shape == end_node.array.shape, (\n"
        "    f'end_grad shape {tuple(end_grad.array.shape)} mismatches '\n"
        "    f'end_node shape {tuple(end_node.array.shape)}'\n"
        ")\n"
        "```\n\n"
        "**Why `ones_like` and not `ones(1)`.** The end node may be a scalar "
        "(`loss.shape == ()`), a per-sample loss vector (`(B,)`), or a "
        "structured object. `ones_like` correctly handles all of them — for a "
        "`(B,)` per-sample loss, the seed is `ones(B)`, encoding `dL/dL_i = 1` "
        "for every sample independently.\n\n"
        "**Why ones, not zeros.** `dL/dL = 1` (the identity). All downstream "
        "back fns multiply by this seed; 1 is the multiplicative identity "
        "that lets every chain-rule product reduce to the actual partial.\n\n"
        "Inputs:\n"
        "- `end_node`: a `MiniTensor` whose `.array` is a `torch.Tensor`.\n"
        "- `end_grad`: `None` OR a `MiniTensor`.\n\n"
        "Output: a raw `torch.Tensor` (NOT a `MiniTensor`) matching "
        "`end_node.array`'s shape and dtype.\n\n"
        "Do NOT call `torch.autograd`."
    ),
    stub=(
        "def resolve_end_grad(end_node: 'MiniTensor', end_grad: 'MiniTensor | None') -> Tensor:\n"
        '    """Resolve the seed gradient for backprop.\n'
        "\n"
        "    None  -> torch.ones_like(end_node.array)\n"
        "    given -> end_grad.array (must match end_node.array shape)\n"
        '    """\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- scalar end node, end_grad=None → ones_like (a 0-D 1.0) ---\n"
        "scalar = MiniTensor(t.tensor(3.5))\n"
        "g = resolve_end_grad(scalar, None)\n"
        "assert isinstance(g, t.Tensor) and not isinstance(g, MiniTensor), (\n"
        "    'must return raw torch.Tensor, not MiniTensor'\n"
        ")\n"
        "assert g.shape == (), f'scalar shape: {g.shape}'\n"
        "assert g.dtype == scalar.array.dtype, f'dtype must match: {g.dtype} vs {scalar.array.dtype}'\n"
        "assert g.item() == 1.0, f'scalar end_grad must be 1.0, got {g.item()}'\n"
        "\n"
        "# --- vector end node, end_grad=None → ones of matching shape ---\n"
        "vec = MiniTensor(t.tensor([1.0, 2.0, 3.0, 4.0]))\n"
        "g = resolve_end_grad(vec, None)\n"
        "assert g.shape == (4,)\n"
        "assert t.allclose(g, t.ones(4)), f'vec ones: {g}'\n"
        "\n"
        "# --- matrix end node, end_grad=None ---\n"
        "mat = MiniTensor(t.zeros(3, 5))\n"
        "g = resolve_end_grad(mat, None)\n"
        "assert g.shape == (3, 5)\n"
        "assert t.allclose(g, t.ones(3, 5))\n"
        "\n"
        "# --- explicit end_grad → unbox and return as-is ---\n"
        "explicit_grad = MiniTensor(t.tensor([0.5, 0.5, 0.5, 0.5]))\n"
        "g = resolve_end_grad(vec, explicit_grad)\n"
        "assert isinstance(g, t.Tensor) and not isinstance(g, MiniTensor)\n"
        "assert t.allclose(g, t.tensor([0.5, 0.5, 0.5, 0.5])), f'explicit grad: {g}'\n"
        "# IDENTITY: the returned tensor should be `explicit_grad.array` (not a copy).\n"
        "assert g is explicit_grad.array, 'unbox should be identity, not copy'\n"
        "\n"
        "# --- shape mismatch raises AssertionError ---\n"
        "wrong_shape = MiniTensor(t.zeros(2, 5))  # vec is (4,)\n"
        "try:\n"
        "    resolve_end_grad(vec, wrong_shape)\n"
        "except AssertionError as e:\n"
        "    msg = str(e)\n"
        "    # Helpful message must mention both shapes.\n"
        "    assert '(2, 5)' in msg or '2, 5' in msg, f'msg missing end_grad shape: {msg!r}'\n"
        "    assert '(4,)' in msg or '4,' in msg, f'msg missing end_node shape: {msg!r}'\n"
        "else:\n"
        "    raise AssertionError('shape mismatch should have raised AssertionError')\n"
        "\n"
        "# --- dtype preserved by ones_like ---\n"
        "fp64_node = MiniTensor(t.tensor([1.0, 2.0], dtype=t.float64))\n"
        "g = resolve_end_grad(fp64_node, None)\n"
        "assert g.dtype == t.float64, f'ones_like must preserve dtype, got {g.dtype}'\n"
        "\n"
        "# --- integer dtype (rare but legal) — still ones_like, dtype preserved ---\n"
        "int_node = MiniTensor(t.tensor([1, 2, 3], dtype=t.int32))\n"
        "g = resolve_end_grad(int_node, None)\n"
        "assert g.dtype == t.int32, f'int32 dtype preserved: {g.dtype}'\n"
        "assert t.eq(g, t.ones(3, dtype=t.int32)).all()"
    ),
    solution_body=(
        "def resolve_end_grad(end_node, end_grad):\n"
        "    if end_grad is None:\n"
        "        # default seed: dL/dL = 1, same shape & dtype as end_node.\n"
        "        return t.ones_like(end_node.array)\n"
        "    # explicit end_grad — must be a MiniTensor; check shape and unbox.\n"
        "    assert end_grad.array.shape == end_node.array.shape, (\n"
        "        f'end_grad shape {tuple(end_grad.array.shape)} mismatches '\n"
        "        f'end_node shape {tuple(end_node.array.shape)}'\n"
        "    )\n"
        "    return end_grad.array"
    ),
    solution_notes=(
        "**Why `ones_like` is the right default.** The seed of the reverse "
        "pass is `d(end_node)/d(end_node)` — the gradient of `end_node` "
        "with respect to itself. For a scalar that's `1`. For a "
        "non-scalar end node, the user is implicitly asking for the "
        "gradient of `end_node.sum()` w.r.t. each input — and the "
        "Jacobian of `sum` is `ones_like(input)`. Same answer, easier to "
        "implement.\n\n"
        "**Why this requires the user to pass `end_grad` when `end_node` "
        "is non-scalar in PyTorch.** PyTorch raises if you call "
        "`tensor.backward()` on a non-scalar without `gradient=...`. The "
        "drill convention is LOOSER — we always default to `ones_like`, "
        "which assumes 'sum reduction.' That's fine for educational use "
        "but be aware real PyTorch is stricter.\n\n"
        "**Why return raw tensor, not MiniTensor.** The rest of "
        "`backprop` works with raw `torch.Tensor` accumulators in a "
        "`dict[MiniTensor, Tensor]`. Keeping MiniTensors out of the "
        "internal accumulator dict avoids double-wrapping and confusion "
        "about which fields are populated."
    ),
)


# =========================================================================
# atom: backward-func-lookup  (1 exercise)
# =========================================================================

SPEC_BACK_LOOKUP = _spec(
    atom_id="backward-func-lookup",
    subtopic="Backprop: BackwardFuncLookup",
    recap=RECAP_BACKWARD_FUNC_LOOKUP,
    ex_idx=1,
    ex_title="implement BackwardFuncLookup with (fn, argnum) keys",
    slug="implement-backward-func-lookup-with-fn-argnum-keys",
    bloom="Apply",
    difficulty_num=2,
    keywords=["backward-func-lookup", "dispatcher", "register", "dict"],
    kcs=["backward-func-lookup", "register-back-fn-after-wrap"],
    lo=(
        "Apply the (forward_fn, arg_position) → back_fn dispatch pattern "
        "by implementing a BackwardFuncLookup class with add and get methods."
    ),
    prompt_body=(
        "Implement `BackwardFuncLookup` — the central back-fn dispatcher "
        "used by the reverse pass. It's just a `dict` with two methods:\n\n"
        "```python\n"
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self.back_funcs = {}    # (forward_fn, arg_position) -> back_fn\n"
        "\n"
        "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
        "        ...\n"
        "\n"
        "    def get_back_func(self, forward_fn, arg_position):\n"
        "        ...\n"
        "```\n\n"
        "Requirements:\n\n"
        "1. **`add_back_func(fwd, argnum, back_fn)`** — store `back_fn` keyed "
        "by the 2-tuple `(fwd, argnum)`. Overwriting an existing key is fine "
        "(makes re-registration cheap).\n\n"
        "2. **`get_back_func(fwd, argnum)`** — return the stored back fn. "
        "**On a missing key, raise a clear `KeyError`** with a message "
        "containing both the function name and the argnum so the user can "
        "diagnose missing registrations.\n\n"
        "Why the 2-key tuple, not nested dicts? Same lookup cost (`O(1)`), "
        "but flat — registration and dispatch are both one-liners. Nested "
        "would require `defaultdict(dict)` and an extra `.get` step.\n\n"
        "Why both `add` and `get` named methods, not `__setitem__` / "
        "`__getitem__`? Named methods make the call sites in the reverse "
        "pass self-documenting:\n\n"
        "```python\n"
        "back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)\n"
        "```\n\n"
        "vs the cryptic `BACK_FUNCS[(node.recipe.func, argnum)]`. Both "
        "work; the named methods are the convention."
    ),
    stub=(
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        '        """Empty registry of (forward_fn, arg_position) -> back_fn."""\n'
        "        raise NotImplementedError()\n"
        "\n"
        "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
        '        """Store back_fn at key (forward_fn, arg_position)."""\n'
        "        raise NotImplementedError()\n"
        "\n"
        "    def get_back_func(self, forward_fn, arg_position):\n"
        '        """Return the stored back_fn, or raise KeyError with diagnostic message."""\n'
        "        raise NotImplementedError()"
    ),
    test_body=(
        "# --- basic register + lookup ---\n"
        "BF = BackwardFuncLookup()\n"
        "\n"
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    return grad_out * y\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    return grad_out * x\n"
        "\n"
        "BF.add_back_func(t.log, 0, log_back)\n"
        "BF.add_back_func(t.multiply, 0, multiply_back0)\n"
        "BF.add_back_func(t.multiply, 1, multiply_back1)\n"
        "\n"
        "assert BF.get_back_func(t.log, 0) is log_back\n"
        "assert BF.get_back_func(t.multiply, 0) is multiply_back0\n"
        "assert BF.get_back_func(t.multiply, 1) is multiply_back1\n"
        "# Different argnums for the same fn return DIFFERENT back fns.\n"
        "assert BF.get_back_func(t.multiply, 0) is not BF.get_back_func(t.multiply, 1)\n"
        "\n"
        "# --- missing key raises a useful KeyError ---\n"
        "try:\n"
        "    BF.get_back_func(t.sin, 0)\n"
        "except KeyError as e:\n"
        "    msg = str(e)\n"
        "    # message must mention the fn (by name) and the argnum so user can debug\n"
        "    assert 'sin' in msg or 'torch' in msg, f'KeyError message missing fn: {msg!r}'\n"
        "    assert '0' in msg, f'KeyError message missing argnum: {msg!r}'\n"
        "else:\n"
        "    raise AssertionError('get_back_func on missing key should raise KeyError')\n"
        "\n"
        "# --- missing argnum (right fn, wrong argnum) also raises ---\n"
        "try:\n"
        "    BF.get_back_func(t.log, 5)  # log is registered at argnum 0, not 5\n"
        "except KeyError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('missing argnum should raise KeyError')\n"
        "\n"
        "# --- re-register (overwrite) is allowed ---\n"
        "def log_back_v2(grad_out, out, x):\n"
        "    return grad_out / (x + 0)\n"
        "BF.add_back_func(t.log, 0, log_back_v2)\n"
        "assert BF.get_back_func(t.log, 0) is log_back_v2, 'overwrite did not take'\n"
        "\n"
        "# --- two BackwardFuncLookup instances are independent ---\n"
        "BF2 = BackwardFuncLookup()\n"
        "try:\n"
        "    BF2.get_back_func(t.log, 0)\n"
        "except KeyError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('separate instances must not share storage')\n"
        "\n"
        "# --- dispatcher-style usage: lookup by (recipe.func, argnum) ---\n"
        "# Simulate the reverse-pass call site so we know the API works as advertised.\n"
        "x_raw = t.tensor([1.0, 2.0, 4.0])\n"
        "out_raw = t.log(x_raw)\n"
        "grad_out = t.ones(3)\n"
        "fwd = t.log\n"
        "argnum = 0\n"
        "back_fn = BF.get_back_func(fwd, argnum)\n"
        "g = back_fn(grad_out, out_raw, x_raw)\n"
        "assert t.allclose(g, grad_out / x_raw), 'dispatched back_fn produced wrong grad'\n"
        "\n"
        "# --- key tuple, not nested dict: lookup is one dict access ---\n"
        "# (peek at the internal — common implementation choice)\n"
        "assert hasattr(BF, 'back_funcs'), 'expected a `back_funcs` dict attribute'\n"
        "assert isinstance(BF.back_funcs, dict)\n"
        "assert (t.multiply, 0) in BF.back_funcs, 'expected (fn, argnum) tuple keys'"
    ),
    solution_body=(
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        # Flat dict keyed by (forward_fn, arg_position).\n"
        "        self.back_funcs = {}\n"
        "\n"
        "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
        "        self.back_funcs[(forward_fn, arg_position)] = back_fn\n"
        "\n"
        "    def get_back_func(self, forward_fn, arg_position):\n"
        "        key = (forward_fn, arg_position)\n"
        "        if key not in self.back_funcs:\n"
        "            raise KeyError(\n"
        "                f'No back_fn registered for ({forward_fn!r}, argnum={arg_position}). '\n"
        "                f'Did you forget BACK_FUNCS.add_back_func({forward_fn.__name__}, '\n"
        "                f'{arg_position}, ...)?'\n"
        "            )\n"
        "        return self.back_funcs[key]"
    ),
    solution_notes=(
        "**Why `KeyError` with a diagnostic message.** The default "
        "`self.back_funcs[key]` would also raise `KeyError`, but the "
        "message would be `KeyError: (<function torch.log>, 0)` — opaque. "
        "Spelling out 'No back_fn registered for ... Did you forget "
        "add_back_func(...)?' turns a 5-minute hunt into a one-second "
        "fix. Worth the three-line investment.\n\n"
        "**Why the flat 2-tuple key.** Conceptually we want a "
        "two-dimensional lookup `(fn, argnum) -> back_fn`. Three "
        "natural implementations:\n"
        "- Nested dict `{fn: {argnum: back_fn}}` — needs `defaultdict` + "
        "two `.get` calls.\n"
        "- Flat 2-tuple key (this one) — one dict, one access.\n"
        "- Class attribute on the back fn itself (e.g. "
        "`back_fn._for = (fn, 0)`) — clever, but the registry is global "
        "anyway, so the class isn't carrying its weight.\n\n"
        "Flat 2-tuple wins on simplicity.\n\n"
        "**Two instances → two registries.** A single global `BACK_FUNCS` "
        "is the usual setup, but having `__init__` create a per-instance "
        "dict means you can spin up a separate registry for unit tests "
        "without polluting the global. The test exercises this."
    ),
)


# =========================================================================
# atom: sorted-computational-graph  (1 exercise)
# =========================================================================

SPEC_SORTED_GRAPH = _spec(
    atom_id="sorted-computational-graph",
    subtopic="Backprop: Sorted computation graph",
    recap=RECAP_SORTED_GRAPH,
    ex_idx=1,
    ex_title="topological sort of the compute graph for the reverse pass",
    slug="topological-sort-of-the-compute-graph-for-reverse-pass",
    bloom="Apply",
    difficulty_num=4,
    keywords=["topological-sort", "reverse-order", "get-parents", "recipe-walk"],
    kcs=["sorted-computational-graph", "parents-dict-by-argidx"],
    lo=(
        "Apply topological sort over a MiniTensor's recipe-parents graph "
        "and reverse the result so the end node comes first — the order "
        "the reverse pass needs."
    ),
    prompt_body=(
        "Implement TWO pieces:\n\n"
        "**1. `topological_sort(node, get_children)`** — generic DFS-based "
        "topological sort. Returns descendants of `node` such that `node` is "
        "LAST (every parent appears before its children's dependencies). "
        "**Must raise `ValueError` on a cycle** (we're a DAG-only system; "
        "circular Recipes mean somebody's mutating during forward).\n\n"
        "Classic three-color DFS:\n"
        "- `temp` (gray) — currently on the DFS stack. Re-visiting one of "
        "these means we found a back-edge → cycle.\n"
        "- `perm` (black) — fully processed; skip.\n"
        "- Anything else (white) — not yet visited; recurse into.\n\n"
        "**2. `sorted_computational_graph(tensor)`** — apply "
        "`topological_sort` over MiniTensor's parent graph and **reverse** "
        "the result. After this call:\n"
        "- `result[0] is tensor` (the end node — where the reverse pass "
        "starts).\n"
        "- `result[-1]` is some leaf MiniTensor (`.recipe is None`).\n"
        "- Iterating in order means every node's accumulated gradient is "
        "fully summed by the time the dispatcher needs it.\n\n"
        "Use `get_parents(t) = list(t.recipe.parents.values())` if "
        "`t.recipe is not None`, else `[]`. Then "
        "`return topological_sort(tensor, get_parents)[::-1]`.\n\n"
        "**Why reverse.** `topological_sort` is designed for forward graphs "
        "(deps first). The reverse pass needs nodes with **outgoing edges "
        "resolved first** — i.e. start at the root, walk to leaves — which "
        "is the same DAG in reverse order. The `[::-1]` is the cheapest "
        "way to flip orientation.\n\n"
        "Helpers (`Recipe`, `MiniTensor`) are in the setup cell already."
    ),
    stub=(
        "def topological_sort(node, get_children):\n"
        '    """Return descendants of node in topological order (node LAST).\n'
        "    Raise ValueError on cycle.\n"
        '    """\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def sorted_computational_graph(tensor):\n"
        '    """Return MiniTensors in reverse-topological order (end node FIRST)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- topological_sort: linked list a -> b -> c ---\n"
        "class N:\n"
        "    def __init__(self, name, *children):\n"
        "        self.name = name\n"
        "        self.children = list(children)\n"
        "    def __repr__(self):\n"
        "        return f'N({self.name})'\n"
        "\n"
        "def get_children(n):\n"
        "    return n.children\n"
        "\n"
        "c = N('c')\n"
        "b = N('b', c)\n"
        "a = N('a', b)\n"
        "order = topological_sort(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "# `a` must be LAST; descendants come before.\n"
        "assert names[-1] == 'a', f'a should be last, got {names}'\n"
        "assert names.index('c') < names.index('b') < names.index('a'), (\n"
        "    f'expected c before b before a, got {names}'\n"
        ")\n"
        "\n"
        "# --- topological_sort: branching DAG ---\n"
        "#         a\n"
        "#        / \\\n"
        "#       b   c\n"
        "#        \\ /\n"
        "#         d\n"
        "d = N('d')\n"
        "b = N('b', d)\n"
        "c = N('c', d)\n"
        "a = N('a', b, c)\n"
        "order = topological_sort(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "# `d` must be first (deepest dep); a last; b and c both before a, after d.\n"
        "assert names[-1] == 'a'\n"
        "assert names.index('d') < names.index('b') < names.index('a')\n"
        "assert names.index('d') < names.index('c') < names.index('a')\n"
        "# d appears exactly once (despite being a shared descendant).\n"
        "assert names.count('d') == 1, f'd should appear once, names={names}'\n"
        "\n"
        "# --- topological_sort: cycle detection ---\n"
        "x = N('x')\n"
        "y = N('y')\n"
        "x.children = [y]\n"
        "y.children = [x]  # cycle!\n"
        "try:\n"
        "    topological_sort(x, get_children)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('cycle should have raised ValueError')\n"
        "\n"
        "# --- sorted_computational_graph: tiny diamond compute graph ---\n"
        "# leaves: a, b, c (no recipe)\n"
        "# d = a * b\n"
        "# e = log(c)\n"
        "# f = d * e\n"
        "# g = log(f)\n"
        "a = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "c = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "d = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "d.recipe = Recipe(func=t.multiply, args=(a.array, b.array), kwargs={}, parents={0: a, 1: b})\n"
        "e = MiniTensor(t.log(c.array), requires_grad=True)\n"
        "e.recipe = Recipe(func=t.log, args=(c.array,), kwargs={}, parents={0: c})\n"
        "f = MiniTensor(d.array * e.array, requires_grad=True)\n"
        "f.recipe = Recipe(func=t.multiply, args=(d.array, e.array), kwargs={}, parents={0: d, 1: e})\n"
        "g = MiniTensor(t.log(f.array), requires_grad=True)\n"
        "g.recipe = Recipe(func=t.log, args=(f.array,), kwargs={}, parents={0: f})\n"
        "\n"
        "order = sorted_computational_graph(g)\n"
        "# End node MUST be first.\n"
        "assert order[0] is g, f'first node should be g (end node), got {order[0]}'\n"
        "# All 7 nodes in the graph appear exactly once.\n"
        "assert len(order) == 7, f'expected 7 nodes, got {len(order)}: {order}'\n"
        "ids = {id(x) for x in order}\n"
        "assert ids == {id(a), id(b), id(c), id(d), id(e), id(f), id(g)}\n"
        "# Ancestor-before-descendant in REVERSE order = descendant-before-ancestor in this list.\n"
        "# i.e. every parent appears AFTER its child in `order`.\n"
        "pos = {id(x): i for i, x in enumerate(order)}\n"
        "assert pos[id(f)] < pos[id(d)], 'f (uses d) must come before d in reverse order'\n"
        "assert pos[id(f)] < pos[id(e)], 'f (uses e) must come before e in reverse order'\n"
        "assert pos[id(d)] < pos[id(a)], 'd (uses a) must come before a'\n"
        "assert pos[id(d)] < pos[id(b)], 'd (uses b) must come before b'\n"
        "assert pos[id(e)] < pos[id(c)], 'e (uses c) must come before c'\n"
        "assert pos[id(g)] < pos[id(f)], 'g (uses f) must come before f'\n"
        "# Each leaf comes AFTER all its direct consumers (DFS-order property).\n"
        "# NB: a and b are consumed by d only; c is consumed by e only. In reverse-topo,\n"
        "# leaves don't necessarily ALL come after all non-leaves — diamond DAGs can\n"
        "# interleave (e.g. order can be g, f, e, c, d, b, a where c appears before d).\n"
        "# What MUST hold is the parent-before-child invariant already asserted above.\n"
        "for leaf, consumer in [(a, d), (b, d), (c, e)]:\n"
        "    assert pos[id(consumer)] < pos[id(leaf)], (\n"
        "        f'{consumer} (consumer) must come before {leaf} (leaf) in reverse-topo'\n"
        "    )\n"
        "\n"
        "# --- single-node graph (just a leaf) ---\n"
        "lonely = MiniTensor(t.tensor([5.0]), requires_grad=True)\n"
        "order = sorted_computational_graph(lonely)\n"
        "assert order == [lonely], f'singleton graph: {order}'"
    ),
    solution_body=(
        "def topological_sort(node, get_children):\n"
        "    result = []\n"
        "    perm = set()    # fully processed nodes (black) — by id() since MiniTensor isn't hashable-by-equality\n"
        "    temp = set()    # currently on the DFS stack (gray) — cycle detector\n"
        "\n"
        "    def visit(cur):\n"
        "        cid = id(cur)\n"
        "        if cid in perm:\n"
        "            return\n"
        "        if cid in temp:\n"
        "            raise ValueError(f'Cycle detected at node {cur!r} — graph is not a DAG')\n"
        "        temp.add(cid)\n"
        "        for child in get_children(cur):\n"
        "            visit(child)\n"
        "        temp.remove(cid)\n"
        "        perm.add(cid)\n"
        "        result.append(cur)\n"
        "\n"
        "    visit(node)\n"
        "    return result\n"
        "\n"
        "\n"
        "def sorted_computational_graph(tensor):\n"
        "    def get_parents(t):\n"
        "        if t.recipe is None:\n"
        "            return []\n"
        "        return list(t.recipe.parents.values())\n"
        "    # topological_sort returns deps-first (end node LAST); reverse for the\n"
        "    # backward pass which wants end node FIRST.\n"
        "    return topological_sort(tensor, get_parents)[::-1]"
    ),
    solution_notes=(
        "**Why `id(...)` instead of the node itself for set membership.** "
        "MiniTensors compare by identity already (we didn't override "
        "`__eq__` or `__hash__`), so `set` would work — but for objects "
        "that DO have value-equality (numpy arrays, torch tensors with "
        "custom `__eq__`) you'd get false positives. `id(...)` is the "
        "safe-by-default identity key.\n\n"
        "**Why three colors, not two.** Two colors (visited / unvisited) "
        "catches re-visits but doesn't distinguish 'I've finished this "
        "subtree' from 'I'm in the middle of this subtree' — i.e. it "
        "can't detect cycles. The `temp`/`perm` split is the standard DFS "
        "topo-sort idiom: temp catches back-edges (cycles), perm avoids "
        "re-processing shared descendants in branching DAGs.\n\n"
        "**Why `[::-1]` at the end of `sorted_computational_graph`.** "
        "`topological_sort` is generic — it's also useful for forward "
        "operations (deps-first). The reverse pass needs the OPPOSITE "
        "order: end node first, leaves last. Reversing keeps the "
        "generic sort reusable. Alternative: write a "
        "`reverse_topological_sort` that builds the list in reverse "
        "order natively (slightly faster, no reversal cost; we choose "
        "clarity here)."
    ),
)


# =========================================================================
# emit
# =========================================================================

ALL_SPECS = [
    SPEC_LOG_BACK,
    SPEC_MULTIPLY_BACK,
    SPEC_MAX_BACK_TIED,
    SPEC_NON_DIFF,
    SPEC_IS_DIFF_FLAG,
    SPEC_END_GRAD,
    SPEC_BACK_LOOKUP,
    SPEC_SORTED_GRAPH,
]


if __name__ == "__main__":
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
