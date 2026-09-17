#!/usr/bin/env python3
"""Author 8 deepening (ex2) drills for ARENA part-4 autograd_pt2 atoms.

Each ex2 probes a DISTINCT facet from the existing ex1 — different cognitive
operation, different surface context, same MiniTensor+Recipe+BackwardFuncLookup
framework. ONE LO + ONE Bloom + <=2 KCs each.

Verification re-runs each spec's solution against its test_body inside the
build venv (torch 2.12.0+cpu) before any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_pt2"


# ---------------------------------------------------------------------------
# Shared autograd preamble — same shape as batch4 ex1 drills.
# ---------------------------------------------------------------------------

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
    "    def __init__(self, array, requires_grad: bool = False, recipe=None):\n"
    "        self.array = array\n"
    "        self.requires_grad = requires_grad\n"
    "        self.recipe = recipe\n"
    "    def __repr__(self):\n"
    "        return f'MiniTensor({self.array!r}, requires_grad={self.requires_grad})'"
)


# ---------------------------------------------------------------------------
# Atom recaps — trimmed to the deepening facet each ex2 probes.
# ---------------------------------------------------------------------------

RECAP_BACK_LOOKUP_DEEP = (
    "## BackwardFuncLookup in a tiny reverse pass — quick refresher\n"
    "\n"
    "The lookup is keyed by `(forward_fn, arg_position)` and dispatched at the "
    "moment the reverse pass walks a node's parents:\n"
    "\n"
    "```python\n"
    "for argnum, parent in node.recipe.parents.items():\n"
    "    back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)\n"
    "    grad_in = back_fn(grad_out, node.array, *node.recipe.args)\n"
    "    grads[parent] = grads.get(parent, 0) + grad_in\n"
    "```\n"
    "\n"
    "Two dispatch invariants that ex1's basic register-and-get tests do NOT "
    "exercise:\n"
    "- **Per-argnum independence.** `(multiply, 0)` and `(multiply, 1)` route "
    "to DIFFERENT back fns even though the forward is symmetric. The "
    "dispatcher does not know about symmetry — it always asks for "
    "`(func, argnum)`.\n"
    "- **Composition.** The lookup is called ONCE PER PARENT EDGE in the "
    "reverse pass. For an op with `K` tensor parents, dispatch happens `K` "
    "times — once per `(func, argnum)` pair."
)

RECAP_END_GRAD_DEEP = (
    "## Non-scalar end_grad with explicit shape — quick refresher\n"
    "\n"
    "ex1 covered the `end_grad=None → ones_like` default. The deeper facet: "
    "when the user passes `end_grad` EXPLICITLY, every per-position value is a "
    "weighting on the seed gradient.\n"
    "\n"
    "```python\n"
    "loss = per_sample_loss(x, y)   # shape (B,)\n"
    "weights = (y == HARD_CLASS).float()  # 1 for hard samples, 0 otherwise\n"
    "loss.backward(weights)         # only hard samples contribute\n"
    "```\n"
    "\n"
    "Inside `resolve_end_grad`, when `end_grad` is supplied, it must:\n"
    "1. Be a `MiniTensor` (not a raw tensor — the public API is consistent).\n"
    "2. Match `end_node.array.shape` element-for-element.\n"
    "3. Be unboxed to its `.array` (the rest of `backprop` works on raw "
    "tensors)."
)

RECAP_IS_DIFF_DEEP = (
    "## is_differentiable wired into wrap_forward_fn — quick refresher\n"
    "\n"
    "ex1 isolated `make_check_requires_grad(is_differentiable)` as a "
    "stand-alone factory. The ex2 facet is the **integration** path: the same "
    "flag is also the gate for ATTACHING A RECIPE inside `wrap_forward_fn`.\n"
    "\n"
    "Two distinct effects of `is_differentiable=False`, both flowing from the "
    "same flag:\n"
    "- Output `requires_grad` is forced False (the three-gate AND short-"
    "circuits on gate 2).\n"
    "- Output `recipe` is `None` — backprop treats the node as a leaf and "
    "stops there, exactly as if the user had detached.\n"
    "\n"
    "Both effects are necessary: setting `requires_grad=False` without "
    "skipping the Recipe would leave a dangling parent edge that the reverse "
    "pass might still try to walk."
)

RECAP_LOG_BACK_DEEP = (
    "## log_back composed with multiply_back — quick refresher\n"
    "\n"
    "ex1 derived `log_back` from the local chain rule `d/dx log(x) = 1/x`. "
    "The deeper facet is **composition**: backward fns aren't just stand-"
    "alone — they get chained through the reverse pass when the forward op is "
    "composed.\n"
    "\n"
    "Forward: `z = log(x * y)`. Two ops, three tensors. The reverse pass "
    "walks the graph end-first:\n"
    "\n"
    "```\n"
    "dL/dz   = 1                              # seed (ones_like)\n"
    "dL/d(x*y) = log_back(dL/dz, z, x*y)      # = 1 / (x*y)\n"
    "dL/dx   = multiply_back0(dL/d(x*y), x*y, x, y) = (1 / (x*y)) * y = 1/x\n"
    "dL/dy   = multiply_back1(dL/d(x*y), x*y, x, y) = (1 / (x*y)) * x = 1/y\n"
    "```\n"
    "\n"
    "The closed-form gradients `dL/dx = 1/x` and `dL/dy = 1/y` are what "
    "torch.autograd produces — composition of the two back fns reproduces "
    "the global chain rule."
)

RECAP_RELU_AS_MAX_DEEP = (
    "## ReLU as maximum_back with y=0 — quick refresher\n"
    "\n"
    "`relu(x) = maximum(x, 0)`. The backward fn `relu_back` is therefore "
    "`maximum_back0` evaluated with `y = 0`. The tie-splitting convention "
    "from ex1 propagates: at the kink `x == 0`, the gradient is `0.5 * "
    "grad_out` (half-mass), not `0` and not `1`.\n"
    "\n"
    "Comparison to torch:\n"
    "- `torch.nn.functional.relu` uses **`grad_out * (x > 0)`** — strict "
    "inequality → 0 at the kink.\n"
    "- Our `relu_back` (via `maximum_back0(grad_out, _, x, zeros_like(x))`) "
    "uses **`(x > 0) + 0.5 * (x == 0)`** → 0.5 at the kink.\n"
    "\n"
    "Both are valid subgradients of `max(x, 0)` at zero. The half-mass "
    "convention is symmetric and conserves gradient mass across both sides "
    "of the kink — convenient for theoretical analysis. Real frameworks pick "
    "strict inequality because the kink almost never occurs in practice with "
    "floating-point inputs."
)

RECAP_MULTIPLY_FLOAT_DEEP = (
    "## multiply_back with a Python-float operand — quick refresher\n"
    "\n"
    "Real call sites mix tensors and scalars: `out = multiply(x, 3.0)`. The "
    "forward wrapper stores `3.0` as a raw float in `recipe.args`. The "
    "reverse pass then calls `multiply_back0(grad_out, out, x, 3.0)` — `y` "
    "is a Python float, NOT a tensor.\n"
    "\n"
    "Two implementation choices that ex1's tensor-tensor tests do not "
    "exercise:\n"
    "1. **Coerce the float to a 0-D tensor** before the multiplication. "
    "`grad_out * 3.0` already works via torch's scalar broadcasting, but "
    "`unbroadcast(grad, y)` needs `y.shape` — calling `.shape` on a Python "
    "float crashes.\n"
    "2. **No backward for the float side.** Python scalars don't have "
    "`.grad`; `multiply_back1` is still defined for the API uniformity, but "
    "the dispatcher only calls it if the corresponding arg was a tracked "
    "MiniTensor. So `multiply_back1` is never invoked for the float — but "
    "it must not crash if it is."
)

RECAP_NON_DIFF_TERMINATES_DEEP = (
    "## non-differentiable wrap as a graph terminator — quick refresher\n"
    "\n"
    "ex1 verified `is_differentiable=False` produces `requires_grad=False` "
    "and `recipe=None`. The deeper consequence: when "
    "`sorted_computational_graph` walks parents, it stops at any node with "
    "`recipe=None` — including the output of a non-diff op.\n"
    "\n"
    "Concretely: if `mask = eq(a, b)` (non-diff), then `mask.recipe is None`, "
    "so even though `mask`'s downstream consumers may reference it, the "
    "reverse-pass walk treats `mask` as a leaf and does NOT recurse into "
    "`a` or `b` THROUGH `mask`.\n"
    "\n"
    "**Why this matters.** Without the terminator behavior, the reverse pass "
    "would walk into a non-diff op (`torch.eq`), find no back_fn registered "
    "for it, and crash with `KeyError`. The `recipe=None` short-circuit is "
    "what makes detach/eq/argmax safe to use mid-graph."
)

RECAP_SORTED_GRAPH_SHARED_DEEP = (
    "## sorted_computational_graph with shared subgraph — quick refresher\n"
    "\n"
    "ex1 covered a tree-like compute graph + diamond DAG. The ex2 facet is a "
    "MORE-shared subgraph: a single leaf consumed by multiple intermediate "
    "nodes, and at multiple depths.\n"
    "\n"
    "Two invariants that get exercised harder here:\n"
    "- **Each node appears EXACTLY ONCE in the sorted list**, regardless of "
    "how many edges point at it. The DFS topo-sort uses a 'permanent' marker "
    "to suppress repeats.\n"
    "- **Parent-before-child holds in reverse order across ALL edges**, "
    "including the long-range edge from end-node to leaf via multiple "
    "intermediate consumers.\n"
    "\n"
    "Without proper marker-set handling, repeated visits to the shared leaf "
    "either duplicate it in the output or recurse infinitely. The `perm` set "
    "guards both."
)


# ---------------------------------------------------------------------------
# Spec helper (mirrors batch4 _spec but with explicit ex_idx=2 default).
# ---------------------------------------------------------------------------

def _spec(
    *,
    atom_id: str,
    subtopic: str,
    recap: str,
    ex_title: str,
    slug: str,
    bloom: str,
    difficulty_num: int,
    keywords: list,
    kcs: list,
    lo: str,
    prompt_body: str,
    stub: str,
    test_body: str,
    solution_body: str,
    solution_notes: str = "",
) -> dict:
    dots = ("\U0001f534" * difficulty_num) + ("⚪" * (5 - difficulty_num))
    return {
        "atom_id": atom_id,
        "subtopic": subtopic,
        "topic_folder": TOPIC,
        "atom_recap_md": recap,
        "exercise_index": 2,
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
        "extra_imports": [_AUTOGRAD_PREAMBLE],
    }


# =========================================================================
# 1. backward-func-lookup ex2 — multi-op register + dispatch in mini reverse pass
# =========================================================================

SPEC_BACK_LOOKUP = _spec(
    atom_id="backward-func-lookup",
    subtopic="Backprop: BackwardFuncLookup",
    recap=RECAP_BACK_LOOKUP_DEEP,
    ex_title="dispatch through BackwardFuncLookup for a 2-op reverse pass",
    slug="dispatch-through-backward-func-lookup-for-2-op-reverse-pass",
    bloom="Apply",
    difficulty_num=3,
    keywords=["dispatch", "reverse-pass", "argnum", "compose"],
    kcs=["backward-func-lookup", "register-back-fn-after-wrap"],
    lo=(
        "Apply BackwardFuncLookup to dispatch the correct back fn for each "
        "(func, argnum) pair across two composed ops in a mini reverse pass."
    ),
    prompt_body=(
        "Implement `mini_reverse_pass(end_node, end_grad, BACK_FUNCS)` — a "
        "minimal reverse pass that consumes a pre-sorted graph and dispatches "
        "back fns via the lookup.\n\n"
        "Inputs:\n"
        "- `end_node`: a `MiniTensor` at the tip of a compute graph.\n"
        "- `end_grad`: a raw `torch.Tensor` (already resolved, same shape as "
        "`end_node.array`).\n"
        "- `BACK_FUNCS`: a populated `BackwardFuncLookup`.\n\n"
        "Output: a dict `grads: dict[int, torch.Tensor]` keyed by `id(leaf)`, "
        "value = accumulated gradient for that leaf.\n\n"
        "Algorithm:\n\n"
        "```python\n"
        "node_grads = {id(end_node): end_grad}\n"
        "for node in sorted_graph:                 # end node first\n"
        "    if node.recipe is None:               # leaf — record and skip\n"
        "        continue\n"
        "    grad_out = node_grads[id(node)]\n"
        "    for argnum, parent in node.recipe.parents.items():\n"
        "        back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)\n"
        "        grad_in = back_fn(grad_out, node.array, *node.recipe.args)\n"
        "        prev = node_grads.get(id(parent))\n"
        "        node_grads[id(parent)] = grad_in if prev is None else prev + grad_in\n"
        "```\n\n"
        "We've provided `sorted_computational_graph` (drops in via a tiny "
        "reverse-topological-walk) and `BackwardFuncLookup` for you. The two "
        "ops you'll be exercising are `t.log` and `t.multiply`.\n\n"
        "**The point of this drill.** It's NOT about implementing topo-sort or "
        "the math — both are upstream prerequisites you already have. It's "
        "about the LOOKUP CALL: `BACK_FUNCS.get_back_func(func, argnum)` runs "
        "ONCE PER PARENT EDGE in the reverse pass. For an op with 2 tensor "
        "parents, the lookup is invoked TWICE per node — once with each "
        "argnum."
    ),
    stub=(
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self.back_funcs = {}\n"
        "    def add_back_func(self, fwd, argnum, back_fn):\n"
        "        self.back_funcs[(fwd, argnum)] = back_fn\n"
        "    def get_back_func(self, fwd, argnum):\n"
        "        return self.back_funcs[(fwd, argnum)]\n"
        "\n"
        "\n"
        "def sorted_computational_graph(tensor):\n"
        "    \"\"\"Provided: reverse-topo sort (end node first).\"\"\"\n"
        "    result = []\n"
        "    perm = set()\n"
        "    def visit(cur):\n"
        "        if id(cur) in perm:\n"
        "            return\n"
        "        perm.add(id(cur))\n"
        "        if cur.recipe is not None:\n"
        "            for p in cur.recipe.parents.values():\n"
        "                visit(p)\n"
        "        result.append(cur)\n"
        "    visit(tensor)\n"
        "    return result[::-1]\n"
        "\n"
        "\n"
        "def mini_reverse_pass(end_node, end_grad, BACK_FUNCS):\n"
        '    """Walk sorted_computational_graph end-first, dispatching back fns."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- register two ops worth of back fns ---\n"
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    return grad_out * y if isinstance(y, t.Tensor) else grad_out * t.tensor(y)\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    return grad_out * x if isinstance(x, t.Tensor) else grad_out * t.tensor(x)\n"
        "\n"
        "BACK_FUNCS = BackwardFuncLookup()\n"
        "BACK_FUNCS.add_back_func(t.log, 0, log_back)\n"
        "BACK_FUNCS.add_back_func(t.multiply, 0, multiply_back0)\n"
        "BACK_FUNCS.add_back_func(t.multiply, 1, multiply_back1)\n"
        "\n"
        "# --- build z = log(x * y) by hand ---\n"
        "x_raw = t.tensor([2.0, 4.0, 8.0])\n"
        "y_raw = t.tensor([3.0, 5.0, 7.0])\n"
        "x = MiniTensor(x_raw, requires_grad=True)\n"
        "y = MiniTensor(y_raw, requires_grad=True)\n"
        "xy = MiniTensor(x.array * y.array, requires_grad=True)\n"
        "xy.recipe = Recipe(func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y})\n"
        "z = MiniTensor(t.log(xy.array), requires_grad=True)\n"
        "z.recipe = Recipe(func=t.log, args=(xy.array,), kwargs={}, parents={0: xy})\n"
        "\n"
        "# --- run the reverse pass ---\n"
        "end_grad = t.ones_like(z.array)\n"
        "grads = mini_reverse_pass(z, end_grad, BACK_FUNCS)\n"
        "\n"
        "# --- structural checks ---\n"
        "assert isinstance(grads, dict), f'expected dict, got {type(grads)}'\n"
        "assert id(x) in grads, 'x not in grads'\n"
        "assert id(y) in grads, 'y not in grads'\n"
        "# z is the end node — caller usually records its grad too.\n"
        "assert id(z) in grads, 'end node grad should be in grads'\n"
        "\n"
        "# --- value: closed form dL/dx = 1/x, dL/dy = 1/y ---\n"
        "expected_dx = 1.0 / x_raw\n"
        "expected_dy = 1.0 / y_raw\n"
        "assert t.allclose(grads[id(x)], expected_dx, atol=1e-6), (\n"
        "    f'dL/dx wrong: got {grads[id(x)]}, expected {expected_dx}'\n"
        ")\n"
        "assert t.allclose(grads[id(y)], expected_dy, atol=1e-6), (\n"
        "    f'dL/dy wrong: got {grads[id(y)]}, expected {expected_dy}'\n"
        ")\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = x_raw.clone().requires_grad_(True)\n"
        "y_ref = y_raw.clone().requires_grad_(True)\n"
        "loss = t.log(x_ref * y_ref).sum()\n"
        "loss.backward()\n"
        "assert t.allclose(grads[id(x)], x_ref.grad, atol=1e-6), 'disagrees with autograd on x'\n"
        "assert t.allclose(grads[id(y)], y_ref.grad, atol=1e-6), 'disagrees with autograd on y'\n"
        "\n"
        "# --- dispatch happens via the lookup (not hardcoded) ---\n"
        "# Re-register multiply_back0 to a SCALED version and verify the new\n"
        "# back fn is what runs — proves we go through BACK_FUNCS each time.\n"
        "BACK_FUNCS.add_back_func(t.multiply, 0, lambda g, o, a, b: 2.0 * (g * (b if isinstance(b, t.Tensor) else t.tensor(b))))\n"
        "grads2 = mini_reverse_pass(z, end_grad, BACK_FUNCS)\n"
        "# dL/dx is now 2 * (1/x) because multiply_back0 doubled the grad.\n"
        "assert t.allclose(grads2[id(x)], 2.0 / x_raw, atol=1e-6), (\n"
        "    f'dispatch not going through BACK_FUNCS — got {grads2[id(x)]} expected {2.0 / x_raw}'\n"
        ")"
    ),
    solution_body=(
        "def mini_reverse_pass(end_node, end_grad, BACK_FUNCS):\n"
        "    node_grads = {id(end_node): end_grad}\n"
        "    id_to_node = {id(end_node): end_node}\n"
        "    for node in sorted_computational_graph(end_node):\n"
        "        id_to_node[id(node)] = node\n"
        "        if node.recipe is None:\n"
        "            continue\n"
        "        grad_out = node_grads[id(node)]\n"
        "        for argnum, parent in node.recipe.parents.items():\n"
        "            back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)\n"
        "            grad_in = back_fn(grad_out, node.array, *node.recipe.args)\n"
        "            prev = node_grads.get(id(parent))\n"
        "            node_grads[id(parent)] = grad_in if prev is None else prev + grad_in\n"
        "    return node_grads"
    ),
    solution_notes=(
        "**Why dispatch matters even on this tiny graph.** The two `(t.multiply, "
        "argnum)` registrations point to DIFFERENT back fns even though the "
        "math is symmetric. The reverse pass would crash with a `KeyError` or "
        "produce wrong grads if the lookup keyed only on `func` and ignored "
        "`argnum`.\n\n"
        "**Why `id(parent)` keys.** Two MiniTensors with equal `.array` "
        "tensors are still SEPARATE leaves — they accumulate their own grads. "
        "Identity keys avoid accidental merging. The same pattern shows up in "
        "real frameworks: PyTorch's autograd graph keys by tensor identity, "
        "not value.\n\n"
        "**Re-register exercises the dispatch path.** Hardcoding `log_back` "
        "and `multiply_back0` inside the loop would pass the first half of "
        "the test. The 'doubled back fn' test exposes that — only an "
        "implementation that goes through `BACK_FUNCS.get_back_func(...)` "
        "every call sees the re-registration."
    ),
)


# =========================================================================
# 2. end-grad-default-ones-like ex2 — non-scalar end_grad with explicit weights
# =========================================================================

SPEC_END_GRAD = _spec(
    atom_id="end-grad-default-ones-like",
    subtopic="Backprop: end-grad ones_like default",
    recap=RECAP_END_GRAD_DEEP,
    ex_title="resolve end_grad with explicit per-sample weights on a (B,) loss",
    slug="resolve-end-grad-explicit-weights-per-sample-loss",
    bloom="Apply",
    difficulty_num=2,
    keywords=["end-grad", "per-sample-weights", "loss-vector", "unbox"],
    kcs=["end-grad-default-ones-like", "buffer-copy-inplace"],
    lo=(
        "Apply the explicit end_grad path: validate the supplied MiniTensor's "
        "shape against the end node, unbox it, and demonstrate that per-"
        "sample weighting flows through to per-leaf grads as a linear scale."
    ),
    prompt_body=(
        "Implement `weighted_seed_and_chain(end_node, end_grad, x_leaf)` — a "
        "tiny demonstration that the explicit `end_grad` is honored "
        "ELEMENTWISE in the reverse pass.\n\n"
        "Setup (we build this inside the test):\n"
        "- `x_leaf`: a `(B,)` MiniTensor leaf, `requires_grad=True`.\n"
        "- `end_node`: a `(B,)` MiniTensor where `end_node = log(x_leaf)` (we "
        "build the Recipe by hand).\n"
        "- `end_grad`: a `(B,)` MiniTensor of per-sample weights.\n\n"
        "Your function must:\n\n"
        "1. **Resolve the seed.** If `end_grad is None`, default to "
        "`t.ones_like(end_node.array)`. Otherwise unbox `end_grad.array` "
        "after asserting `end_grad.array.shape == end_node.array.shape` "
        "(raise `AssertionError` with a helpful message).\n"
        "2. **Apply one step of the reverse chain.** `end_node`'s recipe is "
        "`log`, so `dL/dx_leaf = seed * (1 / x_leaf.array)`.\n"
        "3. **Return `(seed, dL_dx_leaf)`** — both raw `torch.Tensor`.\n\n"
        "The test verifies:\n"
        "- shape and dtype propagation,\n"
        "- explicit weights produce per-position scaling on the leaf grad,\n"
        "- `end_grad=None` reproduces `ones_like` behavior,\n"
        "- shape-mismatch raises with a useful message,\n"
        "- per-sample weighting matches torch.autograd on the equivalent "
        "weighted-sum loss `(weights * log(x)).sum()`.\n\n"
        "Do NOT call `torch.autograd`."
    ),
    stub=(
        "def weighted_seed_and_chain(end_node, end_grad, x_leaf):\n"
        '    """Resolve seed, then chain one log_back step. Returns (seed, dL_dx)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- build x_leaf and end_node = log(x_leaf) by hand ---\n"
        "x_raw = t.tensor([2.0, 4.0, 8.0, 16.0])\n"
        "x_leaf = MiniTensor(x_raw, requires_grad=True)\n"
        "end_node = MiniTensor(t.log(x_raw), requires_grad=True)\n"
        "end_node.recipe = Recipe(func=t.log, args=(x_raw,), kwargs={}, parents={0: x_leaf})\n"
        "\n"
        "# --- explicit per-sample weights: emphasize hard samples (indices 2, 3) ---\n"
        "weights = MiniTensor(t.tensor([0.0, 0.0, 1.0, 1.0]))\n"
        "seed, dL_dx = weighted_seed_and_chain(end_node, weights, x_leaf)\n"
        "\n"
        "# --- seed is the unboxed weights ---\n"
        "assert isinstance(seed, t.Tensor) and not isinstance(seed, MiniTensor)\n"
        "assert t.allclose(seed, weights.array), f'seed should unbox weights: {seed}'\n"
        "assert seed is weights.array, 'unbox should be identity, not copy'\n"
        "\n"
        "# --- dL/dx_leaf = weights / x_raw, position-wise ---\n"
        "expected = t.tensor([0.0, 0.0, 1.0/8.0, 1.0/16.0])\n"
        "assert t.allclose(dL_dx, expected, atol=1e-7), f'dL/dx wrong: {dL_dx} vs {expected}'\n"
        "assert dL_dx.shape == x_raw.shape, f'leaf grad shape: {dL_dx.shape}'\n"
        "\n"
        "# --- end_grad=None falls back to ones_like ---\n"
        "seed_none, dL_dx_none = weighted_seed_and_chain(end_node, None, x_leaf)\n"
        "assert t.allclose(seed_none, t.ones(4)), f'None → ones_like: {seed_none}'\n"
        "assert t.allclose(dL_dx_none, 1.0 / x_raw, atol=1e-7), f'None chain: {dL_dx_none}'\n"
        "\n"
        "# --- dtype preserved by ones_like ---\n"
        "fp64_raw = t.tensor([1.0, 2.0], dtype=t.float64)\n"
        "fp64_leaf = MiniTensor(fp64_raw, requires_grad=True)\n"
        "fp64_end = MiniTensor(t.log(fp64_raw), requires_grad=True)\n"
        "fp64_end.recipe = Recipe(func=t.log, args=(fp64_raw,), kwargs={}, parents={0: fp64_leaf})\n"
        "seed64, _ = weighted_seed_and_chain(fp64_end, None, fp64_leaf)\n"
        "assert seed64.dtype == t.float64, f'dtype preserved: {seed64.dtype}'\n"
        "\n"
        "# --- shape mismatch raises ---\n"
        "wrong = MiniTensor(t.zeros(2, 3))\n"
        "try:\n"
        "    weighted_seed_and_chain(end_node, wrong, x_leaf)\n"
        "except AssertionError as e:\n"
        "    msg = str(e)\n"
        "    assert '(2, 3)' in msg or '2, 3' in msg, f'msg missing end_grad shape: {msg!r}'\n"
        "    assert '(4,)' in msg or '4,' in msg, f'msg missing end_node shape: {msg!r}'\n"
        "else:\n"
        "    raise AssertionError('shape mismatch should have raised')\n"
        "\n"
        "# --- per-sample weighting matches torch.autograd on weighted-sum loss ---\n"
        "x_ref = x_raw.clone().requires_grad_(True)\n"
        "w_ref = weights.array.clone()\n"
        "loss = (w_ref * t.log(x_ref)).sum()\n"
        "loss.backward()\n"
        "assert t.allclose(dL_dx, x_ref.grad, atol=1e-7), (\n"
        "    f'weighted chain disagrees with autograd: ours={dL_dx}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def weighted_seed_and_chain(end_node, end_grad, x_leaf):\n"
        "    # 1. resolve the seed.\n"
        "    if end_grad is None:\n"
        "        seed = t.ones_like(end_node.array)\n"
        "    else:\n"
        "        assert end_grad.array.shape == end_node.array.shape, (\n"
        "            f'end_grad shape {tuple(end_grad.array.shape)} mismatches '\n"
        "            f'end_node shape {tuple(end_node.array.shape)}'\n"
        "        )\n"
        "        seed = end_grad.array\n"
        "    # 2. one step of log_back: dL/dx = seed / x.\n"
        "    dL_dx = seed / x_leaf.array\n"
        "    return seed, dL_dx"
    ),
    solution_notes=(
        "**Why explicit weights are equivalent to `(weights * loss).sum()`.** "
        "Passing `end_grad=w` to `loss.backward()` is mathematically the same "
        "as calling `(w * loss).sum().backward()`. The chain rule's seed "
        "multiplies through every downstream back fn — putting it in the "
        "seed vs in a wrapping `.sum()` produces identical leaf grads.\n\n"
        "**Why unbox is identity, not copy.** `end_grad.array` is the raw "
        "tensor; we want to use it directly as the seed for the rest of "
        "backprop. Cloning would double memory for no reason. The `is` test "
        "in the harness pins this down — implementations that "
        "`return end_grad.array.clone()` would fail it.\n\n"
        "**Why ones_like preserves dtype.** `t.ones_like(x)` reads `x.dtype` "
        "by default. If you write `t.ones(end_node.array.shape)` you get "
        "`float32` regardless — silently demoting `float64` end nodes."
    ),
)


# =========================================================================
# 3. is-differentiable-flag ex2 — flag wired into full wrap_forward_fn
# =========================================================================

SPEC_IS_DIFF = _spec(
    atom_id="is-differentiable-flag",
    subtopic="Backprop: is_differentiable flag",
    recap=RECAP_IS_DIFF_DEEP,
    ex_title="wire is_differentiable into a Recipe-building wrap_forward_fn",
    slug="wire-is-differentiable-into-recipe-building-wrap-forward-fn",
    bloom="Apply",
    difficulty_num=3,
    keywords=["is-differentiable", "recipe-gate", "wrap-forward-fn", "closure"],
    kcs=["is-differentiable-flag", "non-diff-fn-wrap"],
    lo=(
        "Apply the per-op is_differentiable flag as a TWO-effect gate inside "
        "wrap_forward_fn: it forces requires_grad=False AND skips Recipe "
        "construction, both via the same closure-captured boolean."
    ),
    prompt_body=(
        "Implement `wrap_forward_fn(fwd_fn, is_differentiable=True)`. "
        "Different surface from ex1's `make_check_requires_grad` factory: "
        "this is the FULL wrapper that produces a working `tensor_func`.\n\n"
        "Requirements:\n\n"
        "1. **Unbox** MiniTensor args to their raw arrays. Non-tensor args "
        "pass through.\n"
        "2. **Forward call** `fwd_fn(*raw_args, **kwargs)`.\n"
        "3. **Three-gate AND** to compute `requires_grad`:\n"
        "   `grad_tracking_enabled AND is_differentiable AND any-tracked-input`.\n"
        "4. **Box** the result as a `MiniTensor(out_arr, requires_grad)`.\n"
        "5. **Conditional Recipe.** ATTACH a Recipe ONLY when `requires_grad` "
        "is True — both effects of `is_differentiable=False` flow from this "
        "same boolean.\n\n"
        "Verify the two effects co-occur:\n"
        "- A differentiable op (`add` wrapper) with tracked inputs produces "
        "`requires_grad=True` AND a populated `recipe`.\n"
        "- A non-differentiable op (`eq` wrapper) with tracked inputs "
        "produces `requires_grad=False` AND `recipe=None`.\n"
        "- The closure-captured flag is sticky: re-using the SAME wrapped op "
        "across many calls keeps the flag's effect consistent.\n\n"
        "Setup cell provides `MiniTensor`, `Recipe`, "
        "`grad_tracking_enabled=True`. Don't call `torch.autograd`."
    ),
    stub=(
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        '    """Return a tensor_func. is_differentiable=False forces\n'
        "    requires_grad=False AND recipe=None on every output.\n"
        '    """\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "globals()['grad_tracking_enabled'] = True\n"
        "\n"
        "add    = wrap_forward_fn(t.add)\n"
        "eq     = wrap_forward_fn(t.eq, is_differentiable=False)\n"
        "\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([1.0, 2.5, 3.0]), requires_grad=True)\n"
        "\n"
        "# --- differentiable op: BOTH effects (requires_grad True + recipe present) ---\n"
        "out_add = add(a, b)\n"
        "assert isinstance(out_add, MiniTensor)\n"
        "assert out_add.requires_grad is True, 'diff + tracked → True'\n"
        "assert out_add.recipe is not None, 'diff + tracked → recipe attached'\n"
        "assert out_add.recipe.func is t.add\n"
        "# parents dict keyed by argnum.\n"
        "assert set(out_add.recipe.parents.keys()) == {0, 1}, (\n"
        "    f'expected parents at argnums 0, 1, got {out_add.recipe.parents.keys()}'\n"
        ")\n"
        "assert out_add.recipe.parents[0] is a and out_add.recipe.parents[1] is b\n"
        "\n"
        "# --- non-differentiable op: BOTH effects (requires_grad False + recipe None) ---\n"
        "out_eq = eq(a, b)\n"
        "assert out_eq.requires_grad is False, 'non-diff → requires_grad False'\n"
        "assert out_eq.recipe is None, 'non-diff → recipe None'\n"
        "# Output of eq is bool — sanity.\n"
        "assert out_eq.array.dtype == t.bool, f'eq output dtype: {out_eq.array.dtype}'\n"
        "\n"
        "# --- closure stickiness: 5 calls in a row, flag never leaks ---\n"
        "for _ in range(5):\n"
        "    o = eq(a, b)\n"
        "    assert o.requires_grad is False and o.recipe is None, (\n"
        "        'is_differentiable closure broke across repeated calls'\n"
        "    )\n"
        "\n"
        "# --- two wrappers with DIFFERENT flags do not cross-contaminate ---\n"
        "fn_diff = wrap_forward_fn(t.add)                  # diff\n"
        "fn_nondiff = wrap_forward_fn(t.add, is_differentiable=False)   # same fwd, non-diff\n"
        "o_diff = fn_diff(a, b)\n"
        "o_nondiff = fn_nondiff(a, b)\n"
        "assert o_diff.requires_grad is True and o_diff.recipe is not None\n"
        "assert o_nondiff.requires_grad is False and o_nondiff.recipe is None\n"
        "# Forward results are equal (same fwd fn), but recipe state differs.\n"
        "assert t.allclose(o_diff.array, o_nondiff.array)\n"
        "\n"
        "# --- gate 3 (input tracking) still composes: untracked inputs → False even for diff ---\n"
        "a2 = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "b2 = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "out_untracked = add(a2, b2)\n"
        "assert out_untracked.requires_grad is False, 'no tracked inputs → False'\n"
        "assert out_untracked.recipe is None, 'no tracked inputs → no recipe'\n"
        "\n"
        "# --- gate 1 (global) propagates: toggle OFF disables every wrapper ---\n"
        "globals()['grad_tracking_enabled'] = False\n"
        "try:\n"
        "    out_off = add(a, b)\n"
        "    assert out_off.requires_grad is False and out_off.recipe is None\n"
        "finally:\n"
        "    globals()['grad_tracking_enabled'] = True"
    ),
    solution_body=(
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = tuple(a.array if isinstance(a, MiniTensor) else a for a in args)\n"
        "        out_arr = fwd_fn(*raw_args, **kwargs)\n"
        "        requires_grad = (\n"
        "            globals()['grad_tracking_enabled']\n"
        "            and is_differentiable\n"
        "            and any(isinstance(a, MiniTensor) and a.requires_grad for a in args)\n"
        "        )\n"
        "        out = MiniTensor(out_arr, requires_grad)\n"
        "        if requires_grad:\n"
        "            parents = {i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
        "            out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    solution_notes=(
        "**Why BOTH effects flow from the same flag.** The `if requires_grad:` "
        "guard around Recipe construction makes the two effects atomic. If you "
        "build the Recipe unconditionally and only set "
        "`out.requires_grad = False`, the reverse pass — which keys on "
        "`recipe is not None` — would still walk into the non-diff op's "
        "parents and crash on the missing back fn.\n\n"
        "**Two wrappers, same fwd fn, different flags.** This is uncommon in "
        "production (you'd just register `t.add` once with the right flag), "
        "but it pins down the closure semantics: each call to "
        "`wrap_forward_fn` creates a SEPARATE closure with its own captured "
        "`is_differentiable`. Reused references to the same `fwd_fn` do not "
        "share state.\n\n"
        "**Reading the global via `globals()`.** A bare "
        "`grad_tracking_enabled` reference inside `tensor_func` ALSO works "
        "(Python resolves it via LEGB → module globals at call time), but "
        "`globals()['...']` is more explicit when the wrapper is later moved "
        "into a class or another module."
    ),
)


# =========================================================================
# 4. log-back ex2 — compose log_back ∘ multiply_back via the chain rule
# =========================================================================

SPEC_LOG_BACK = _spec(
    atom_id="log-back",
    subtopic="Backprop: log_back",
    recap=RECAP_LOG_BACK_DEEP,
    ex_title="compose log_back and multiply_back through z = log(x*y)",
    slug="compose-log-back-and-multiply-back-through-log-of-x-times-y",
    bloom="Apply",
    difficulty_num=3,
    keywords=["log-back", "chain-rule", "composition", "multiply-back"],
    kcs=["log-back", "chain-rule-elementwise"],
    lo=(
        "Apply log_back composed with multiply_back across a 2-op forward "
        "z = log(x*y) and show the result matches the closed-form gradients "
        "dL/dx = 1/x and dL/dy = 1/y."
    ),
    prompt_body=(
        "Implement `compose_log_multiply_back(x, y)` — returns "
        "`(dL_dx, dL_dy)` for the forward `z = log(x * y)` summed to scalar, "
        "computed by hand-composing two back fns.\n\n"
        "The math (this is the LO):\n"
        "```\n"
        "z = log(x * y)\n"
        "L = z.sum()              # so dL/dz = ones_like(z)\n"
        "dL/d(x*y) = log_back(dL/dz, z, x*y)                   = ones / (x*y)\n"
        "dL/dx     = multiply_back0(dL/d(x*y), x*y, x, y)      = (1/(x*y)) * y  = 1/x\n"
        "dL/dy     = multiply_back1(dL/d(x*y), x*y, x, y)      = (1/(x*y)) * x  = 1/y\n"
        "```\n\n"
        "You implement THREE back fns + the composition:\n\n"
        "1. `log_back(grad_out, out, x_arg)` — `grad_out / x_arg`.\n"
        "2. `multiply_back0(grad_out, out, x_arg, y_arg)` — `grad_out * y_arg`.\n"
        "3. `multiply_back1(grad_out, out, x_arg, y_arg)` — `grad_out * x_arg`.\n"
        "4. `compose_log_multiply_back(x, y)`:\n"
        "   - Compute `xy = x * y` and `z = log(xy)`.\n"
        "   - Seed `dL_dz = t.ones_like(z)`.\n"
        "   - Chain back through `log_back` and then `multiply_back{0,1}`.\n"
        "   - Return `(dL_dx, dL_dy)` — both raw `torch.Tensor`.\n\n"
        "Composition is what gives you the simple closed form: the algebra "
        "cancels even though no individual back fn 'knows' the final answer. "
        "Verify against `torch.autograd`.\n\n"
        "Inputs strictly positive (so `log` is defined). Don't call "
        "`torch.autograd` in your implementation."
    ),
    stub=(
        "def log_back(grad_out: Tensor, out: Tensor, x_arg: Tensor) -> Tensor:\n"
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def multiply_back0(grad_out: Tensor, out: Tensor, x_arg: Tensor, y_arg: Tensor) -> Tensor:\n"
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out: Tensor, out: Tensor, x_arg: Tensor, y_arg: Tensor) -> Tensor:\n"
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def compose_log_multiply_back(x: Tensor, y: Tensor):\n"
        '    """Hand-compose log_back ∘ multiply_back for z = log(x*y). Returns (dL/dx, dL/dy)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- back-fn correctness in isolation ---\n"
        "x = t.tensor([2.0, 4.0])\n"
        "y = t.tensor([3.0, 5.0])\n"
        "xy = x * y\n"
        "z = t.log(xy)\n"
        "g_log = log_back(t.ones(2), z, xy)\n"
        "assert t.allclose(g_log, 1.0 / xy, atol=1e-7), f'log_back wrong: {g_log}'\n"
        "g0 = multiply_back0(t.ones(2), xy, x, y)\n"
        "g1 = multiply_back1(t.ones(2), xy, x, y)\n"
        "assert t.allclose(g0, y), f'multiply_back0 wrong: {g0}'\n"
        "assert t.allclose(g1, x), f'multiply_back1 wrong: {g1}'\n"
        "\n"
        "# --- composition produces closed-form 1/x, 1/y ---\n"
        "x = t.tensor([1.0, 2.0, 4.0, 8.0])\n"
        "y = t.tensor([3.0, 5.0, 7.0, 11.0])\n"
        "dL_dx, dL_dy = compose_log_multiply_back(x, y)\n"
        "assert dL_dx.shape == x.shape, f'dL/dx shape: {dL_dx.shape}'\n"
        "assert dL_dy.shape == y.shape, f'dL/dy shape: {dL_dy.shape}'\n"
        "expected_dx = 1.0 / x\n"
        "expected_dy = 1.0 / y\n"
        "assert t.allclose(dL_dx, expected_dx, atol=1e-6), (\n"
        "    f'dL/dx mismatch: got {dL_dx}, expected {expected_dx}'\n"
        ")\n"
        "assert t.allclose(dL_dy, expected_dy, atol=1e-6), (\n"
        "    f'dL/dy mismatch: got {dL_dy}, expected {expected_dy}'\n"
        ")\n"
        "\n"
        "# --- agreement with torch.autograd on the full composed loss ---\n"
        "x_ref = x.clone().requires_grad_(True)\n"
        "y_ref = y.clone().requires_grad_(True)\n"
        "loss = t.log(x_ref * y_ref).sum()\n"
        "loss.backward()\n"
        "assert t.allclose(dL_dx, x_ref.grad, atol=1e-6), 'composition disagrees with autograd on x'\n"
        "assert t.allclose(dL_dy, y_ref.grad, atol=1e-6), 'composition disagrees with autograd on y'\n"
        "\n"
        "# --- different shape to catch shape-loss bugs ---\n"
        "x2 = t.tensor([[1.0, 2.0], [4.0, 8.0]])\n"
        "y2 = t.tensor([[3.0, 5.0], [7.0, 11.0]])\n"
        "dL_dx2, dL_dy2 = compose_log_multiply_back(x2, y2)\n"
        "assert dL_dx2.shape == (2, 2)\n"
        "assert t.allclose(dL_dx2, 1.0 / x2, atol=1e-6)\n"
        "assert t.allclose(dL_dy2, 1.0 / y2, atol=1e-6)"
    ),
    solution_body=(
        "def log_back(grad_out: Tensor, out: Tensor, x_arg: Tensor) -> Tensor:\n"
        "    return grad_out / x_arg\n"
        "\n"
        "\n"
        "def multiply_back0(grad_out: Tensor, out: Tensor, x_arg: Tensor, y_arg: Tensor) -> Tensor:\n"
        "    return grad_out * y_arg\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out: Tensor, out: Tensor, x_arg: Tensor, y_arg: Tensor) -> Tensor:\n"
        "    return grad_out * x_arg\n"
        "\n"
        "\n"
        "def compose_log_multiply_back(x: Tensor, y: Tensor):\n"
        "    xy = x * y\n"
        "    z = t.log(xy)\n"
        "    # seed.\n"
        "    dL_dz = t.ones_like(z)\n"
        "    # step 1: back through log.\n"
        "    dL_dxy = log_back(dL_dz, z, xy)\n"
        "    # step 2: back through multiply (two parents → two back fns).\n"
        "    dL_dx = multiply_back0(dL_dxy, xy, x, y)\n"
        "    dL_dy = multiply_back1(dL_dxy, xy, x, y)\n"
        "    return dL_dx, dL_dy"
    ),
    solution_notes=(
        "**Why composition yields the simple closed form.** The "
        "intermediate `dL/d(x*y) = 1/(x*y)` looks ugly, but the next "
        "back fn multiplies by the OTHER factor — `y` in the case of "
        "`dL/dx` — and the algebra cancels: `(1/(x*y)) * y = 1/x`. This "
        "is the whole point of automatic differentiation: each back fn "
        "stays local and simple, but the chain of compositions reproduces "
        "the global derivative without ever materializing the Jacobian.\n\n"
        "**Why log_back doesn't read `out`.** Same observation as ex1, but "
        "the composition makes it concrete: `log_back(dL_dz, z, xy)` "
        "ignores `z` entirely. The signature carries `out` for dispatcher "
        "uniformity; the actual gradient computation only needs the input.\n\n"
        "**Why both multiply_backs run on the SAME intermediate.** "
        "`multiply` has two tensor parents, both contributing to the same "
        "output. The reverse pass dispatches both back fns with the SAME "
        "`grad_out` (here `dL_dxy`), routing each to its corresponding "
        "parent. This is the canonical pattern for any K-parent op."
    ),
)


# =========================================================================
# 5. max-back-tied-half ex2 — ReLU as maximum_back0 special case
# =========================================================================

SPEC_MAX_BACK = _spec(
    atom_id="max-back-tied-half",
    subtopic="Backprop: max_back with tied half-mass",
    recap=RECAP_RELU_AS_MAX_DEEP,
    ex_title="derive relu_back as maximum_back0 with y=0 (half-mass at the kink)",
    slug="derive-relu-back-as-maximum-back0-with-y-zero",
    bloom="Apply",
    difficulty_num=3,
    keywords=["relu-back", "subgradient", "half-mass", "kink", "specialization"],
    kcs=["max-back-tied-half", "unbroadcast-pattern"],
    lo=(
        "Apply the half-mass tie-splitting convention by specializing "
        "maximum_back0 to y=0 and verifying that the resulting relu_back "
        "produces 0.5 * grad_out at the kink x==0, in contrast to torch's "
        "strict-inequality convention."
    ),
    prompt_body=(
        "Implement `relu_back(grad_out, out, x)` as a SPECIALIZATION of "
        "`maximum_back0`. The forward is `relu(x) = maximum(x, 0)`, so the "
        "backward fn is:\n\n"
        "```\n"
        "relu_back(grad_out, out, x) = maximum_back0(grad_out, out, x, zeros_like(x))\n"
        "```\n\n"
        "Both pieces:\n\n"
        "1. **`maximum_back0(grad_out, out, x, y)`** — your batch-4 ex1 "
        "result. The half-mass rule: `bool_sum_x = (x > y) + 0.5 * (x == y)`, "
        "then return `grad_out * bool_sum_x` (no unbroadcast — shapes match "
        "in this scalar-y case).\n"
        "2. **`relu_back(grad_out, out, x)`** — single-line specialization "
        "calling `maximum_back0` with `y = t.zeros_like(x)`.\n\n"
        "**The point of this drill.** ReLU is the most-used activation in "
        "deep learning, and its backward is a one-line spec of "
        "`maximum_back0`. The tie-splitting convention propagates: at "
        "`x == 0` (the ReLU kink), `bool_sum_x = (0 > 0) + 0.5*(0 == 0) = "
        "0.5`, so `relu_back(grad_out, _, 0) == 0.5 * grad_out`. Compare:\n"
        "- Torch's `nn.functional.relu` uses **`grad_out * (x > 0)`** — "
        "strict inequality → 0 at the kink.\n"
        "- Ours uses **half-mass** → 0.5 at the kink.\n\n"
        "Both are valid subgradients. The test pins down the half-mass "
        "behavior explicitly and confirms agreement with torch in the "
        "strict-positive / strict-negative regions where both conventions "
        "coincide.\n\n"
        "Inputs are raw `torch.Tensor`. No autograd."
    ),
    stub=(
        "def maximum_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """Half-mass tie-splitting backward for out = maximum(x, y)."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def relu_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """ReLU backward as the y=0 specialization of maximum_back0."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- relu_back: strict positive → 1 * grad_out ---\n"
        "x = t.tensor([1.0, 2.0, 3.0])\n"
        "out = t.relu(x)\n"
        "g = relu_back(t.ones(3), out, x)\n"
        "assert t.allclose(g, t.ones(3)), f'positive: {g}'\n"
        "\n"
        "# --- strict negative → 0 ---\n"
        "x = t.tensor([-1.0, -2.0, -3.0])\n"
        "out = t.relu(x)\n"
        "g = relu_back(t.ones(3), out, x)\n"
        "assert t.allclose(g, t.zeros(3)), f'negative: {g}'\n"
        "\n"
        "# --- x == 0: HALF-MASS (0.5), NOT 0 ---\n"
        "x = t.zeros(4)\n"
        "out = t.relu(x)\n"
        "g = relu_back(t.ones(4), out, x)\n"
        "expected = t.full((4,), 0.5)\n"
        "assert t.allclose(g, expected), (\n"
        "    f'half-mass at kink violated: got {g}, expected 0.5 everywhere'\n"
        ")\n"
        "# Half-mass is the entire point — explicit positive check.\n"
        "assert (g == 0.5).all(), f'every kink position must be 0.5, got {g}'\n"
        "\n"
        "# --- mixed signs in one tensor ---\n"
        "x = t.tensor([-2.0, 0.0, 3.0, 0.0, -1.0])\n"
        "out = t.relu(x)\n"
        "grad_out = t.tensor([10.0, 20.0, 30.0, 40.0, 50.0])\n"
        "g = relu_back(grad_out, out, x)\n"
        "expected = t.tensor([0.0, 10.0, 30.0, 20.0, 0.0])\n"
        "assert t.allclose(g, expected), f'mixed: got {g}, expected {expected}'\n"
        "\n"
        "# --- maximum_back0 with y=0 returns the SAME result ---\n"
        "y_zero = t.zeros_like(x)\n"
        "g_via_max = maximum_back0(grad_out, out, x, y_zero)\n"
        "assert t.allclose(g, g_via_max), 'relu_back should be a specialization of maximum_back0'\n"
        "\n"
        "# --- agreement with torch.autograd on strict regions only ---\n"
        "# (Torch uses strict inequality at the kink — so we EXCLUDE x == 0.)\n"
        "x_ref = t.tensor([-2.0, 1.0, -0.5, 3.0, -1.0], requires_grad=True)\n"
        "(t.relu(x_ref)).sum().backward()\n"
        "out_cached = t.relu(x_ref.detach())\n"
        "g_ours = relu_back(t.ones(5), out_cached, x_ref.detach())\n"
        "# No kink values in this input → both conventions agree everywhere.\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'strict-region disagreement with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")\n"
        "\n"
        "# --- explicit difference at the kink: torch=0, ours=0.5 ---\n"
        "x_kink = t.tensor([0.0], requires_grad=True)\n"
        "(t.relu(x_kink)).sum().backward()\n"
        "out_kink = t.relu(x_kink.detach())\n"
        "g_kink_ours = relu_back(t.ones(1), out_kink, x_kink.detach())\n"
        "assert x_kink.grad.item() == 0.0, f'torch should be 0 at kink, got {x_kink.grad.item()}'\n"
        "assert g_kink_ours.item() == 0.5, f'ours should be 0.5 at kink, got {g_kink_ours.item()}'"
    ),
    solution_body=(
        "def maximum_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # Half-mass: 1 where x > y, 0 where x < y, 0.5 at ties.\n"
        "    bool_sum = (x > y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype)\n"
        "    return grad_out * bool_sum\n"
        "\n"
        "\n"
        "def relu_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    return maximum_back0(grad_out, out, x, t.zeros_like(x))"
    ),
    solution_notes=(
        "**Why half-mass at the kink is mathematically the 'right' answer.** "
        "The subgradient of `max(x, 0)` at `x = 0` is the interval `[0, 1]` — "
        "any value in there is a valid one-sided derivative. `0.5` is the "
        "midpoint and the only choice that's symmetric around the kink and "
        "conserves the gradient mass across both sides (the analogue of the "
        "`bool_sum_x + bool_sum_y == 1` invariant from "
        "`max_back`).\n\n"
        "**Why torch picks strict-inequality (`x > 0`) instead.** Two "
        "practical reasons: (1) `x == 0` is vanishingly rare in floating-"
        "point arithmetic, so the two conventions disagree on a measure-zero "
        "set; (2) the strict version avoids a branchy `==` comparison in a "
        "hot kernel — `(x > 0)` compiles to a single comparison op, "
        "`(x > 0) + 0.5*(x == 0)` requires two. Performance, not "
        "correctness.\n\n"
        "**Why the specialization saves work.** Without it, ReLU would "
        "need its own dedicated back fn duplicating the half-mass logic. "
        "By specializing `maximum_back0(_, _, x, zeros_like(x))`, the "
        "framework re-uses a single tested code path — a tiny but real "
        "lessons-from-software-engineering moment inside the autograd "
        "library."
    ),
)


# =========================================================================
# 6. multiply-back ex2 — Python-float operand + coercion
# =========================================================================

SPEC_MULTIPLY_BACK = _spec(
    atom_id="multiply-back",
    subtopic="Backprop: multiply_back",
    recap=RECAP_MULTIPLY_FLOAT_DEEP,
    ex_title="multiply_back with a Python float operand on one side",
    slug="multiply-back-with-python-float-operand-on-one-side",
    bloom="Apply",
    difficulty_num=3,
    keywords=["multiply-back", "scalar", "float-operand", "coerce", "unbroadcast"],
    kcs=["multiply-back", "arg-position-back-functions"],
    lo=(
        "Apply multiply_back0/back1 to a forward op where one operand is a "
        "Python float, coercing as needed and verifying the float-side "
        "backward never crashes even though it never accumulates a grad."
    ),
    prompt_body=(
        "Implement `multiply_back0(grad_out, out, x, y)` and "
        "`multiply_back1(grad_out, out, x, y)` for the forward op "
        "`out = multiply(x, y)` where EITHER `x` OR `y` may be a Python "
        "float instead of a tensor.\n\n"
        "Provided in the cell: `unbroadcast(grad, original)`.\n\n"
        "Requirements:\n\n"
        "1. **Coerce the partner operand if it's a Python float** so the "
        "multiplication produces a tensor. `grad_out * 3.0` works via "
        "scalar broadcasting, but `unbroadcast(grad, y)` calls `.shape` on "
        "`y` — needs a tensor.\n"
        "2. **`isinstance(y, Tensor)` test** is the cleanest gate: if `y` is "
        "a float, `y = t.tensor(y, dtype=grad_out.dtype)`.\n"
        "3. **Don't crash on a float in the parent slot.** Even if `y` is "
        "a float, `multiply_back1` (the back fn for the y-side) might still "
        "be called by a buggy dispatcher — it must return a sane tensor, "
        "not raise. We'll test this directly.\n"
        "4. **Wrap in `unbroadcast(grad, parent)`** for the tensor side.\n\n"
        "The drill exercises the mixed-type call: `out = multiply(x, 3.0)`. "
        "Verify `multiply_back0(grad_out, out, x, 3.0)` produces "
        "`grad_out * 3.0` reshaped to `x`'s shape, and the float-side back "
        "fn doesn't crash.\n\n"
        "Inputs raw `torch.Tensor` or Python float. No autograd."
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
        "def multiply_back0(grad_out, out, x, y) -> Tensor:\n"
        '    """dL/dx for out = x * y. y may be a Python float."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y) -> Tensor:\n"
        '    """dL/dy for out = x * y. x may be a Python float."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- tensor * float, back through the tensor side ---\n"
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y_f = 5.0\n"
        "out = x * y_f\n"
        "g0 = multiply_back0(t.ones(3), out, x, y_f)\n"
        "assert g0.shape == x.shape, f'g0 shape: {g0.shape}'\n"
        "assert t.allclose(g0, t.full((3,), 5.0)), f'g0 (=y_f) wrong: {g0}'\n"
        "\n"
        "# --- non-unit grad_out ---\n"
        "grad_out = t.tensor([10.0, 100.0, 1000.0])\n"
        "g0 = multiply_back0(grad_out, out, x, y_f)\n"
        "assert t.allclose(g0, grad_out * 5.0), f'chain g0: {g0}'\n"
        "\n"
        "# --- float * tensor (flipped argument order) ---\n"
        "x_f = 7.0\n"
        "y = t.tensor([1.0, 2.0, 4.0])\n"
        "out = x_f * y\n"
        "g1 = multiply_back1(t.ones(3), out, x_f, y)\n"
        "assert g1.shape == y.shape\n"
        "assert t.allclose(g1, t.full((3,), 7.0)), f'g1 (=x_f) wrong: {g1}'\n"
        "\n"
        "# --- float-side back fn must NOT crash when called ---\n"
        "# In a real dispatcher this back fn is skipped for non-tensor parents,\n"
        "# but it must be SAFE to call defensively.\n"
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y_f = 5.0\n"
        "out = x * y_f\n"
        "try:\n"
        "    g_float_side = multiply_back1(t.ones(3), out, x, y_f)\n"
        "    # Whatever it returns, must be a tensor — no AttributeError.\n"
        "    assert isinstance(g_float_side, t.Tensor)\n"
        "except AttributeError as e:\n"
        "    raise AssertionError(f'float-side back fn crashed: {e}')\n"
        "\n"
        "# --- broadcasting still works on the tensor side ---\n"
        "x_b = t.tensor([[1.0, 2.0, 3.0, 4.0]])  # (1,4)\n"
        "y_b = 2.0                                 # scalar\n"
        "out_b = x_b * y_b                         # (1,4)\n"
        "g0_b = multiply_back0(t.ones(3, 4), out_b, x_b, y_b)\n"
        "# Without unbroadcast we'd get (3,4); with unbroadcast we get (1,4) = x_b.shape.\n"
        "assert g0_b.shape == x_b.shape, (\n"
        "    f'expected unbroadcast to (1,4), got {g0_b.shape} '\n"
        "    f'(did you forget to call unbroadcast for the tensor side?)'\n"
        ")\n"
        "# Value: 3 rows of ones * 2 broadcast = 6 per column after unbroadcast.\n"
        "assert t.allclose(g0_b, t.full((1, 4), 6.0)), f'broadcast unbroadcast: {g0_b}'\n"
        "\n"
        "# --- agreement with torch.autograd on tensor*float ---\n"
        "x_ref = t.tensor([2.0, 3.0, 4.0], requires_grad=True)\n"
        "(x_ref * 5.0).sum().backward()\n"
        "g_ours = multiply_back0(t.ones(3), x_ref.detach() * 5.0, x_ref.detach(), 5.0)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), 'disagrees with autograd'"
    ),
    solution_body=(
        "def multiply_back0(grad_out, out, x, y) -> Tensor:\n"
        "    # y may be a Python float — coerce so unbroadcast(., y) works.\n"
        "    if not isinstance(y, t.Tensor):\n"
        "        y = t.tensor(y, dtype=grad_out.dtype)\n"
        "    return unbroadcast(grad_out * y, x)\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y) -> Tensor:\n"
        "    # x may be a Python float — coerce same way.\n"
        "    if not isinstance(x, t.Tensor):\n"
        "        x = t.tensor(x, dtype=grad_out.dtype)\n"
        "    if not isinstance(y, t.Tensor):\n"
        "        # Float-side back fn called defensively: return a 0-D tensor.\n"
        "        return t.tensor(0.0, dtype=grad_out.dtype)\n"
        "    return unbroadcast(grad_out * x, y)"
    ),
    solution_notes=(
        "**Why coerce the partner instead of skipping unbroadcast.** "
        "`grad_out * y_f` already works (torch promotes the float), but the "
        "follow-up `unbroadcast(result, y_f)` calls `y_f.shape` — Python "
        "floats have no `.shape`. The cheapest fix is to coerce `y_f` to a "
        "0-D tensor; `unbroadcast` then collapses correctly to a scalar.\n\n"
        "**Why the float-side back fn must not crash.** In production, the "
        "dispatcher skips non-tensor parents (the parents dict only "
        "contains tensor-valued parents at tracked argnums). But during "
        "development, an over-eager dispatcher might still invoke it; "
        "raising `AttributeError` here turns into a confusing crash trace "
        "30 frames deep. Returning a `0.0` scalar is the safe defensive "
        "answer — the grad is never accumulated anywhere, so the value "
        "doesn't matter.\n\n"
        "**Why match `grad_out.dtype` when coercing.** `t.tensor(3.0)` "
        "defaults to `float32`. If `grad_out` is `float64`, the silent "
        "downcast in the multiplication produces a `float32` grad that "
        "later crashes in-place accumulation. Pin the dtype to "
        "`grad_out.dtype` and the pipeline stays homogeneous."
    ),
)


# =========================================================================
# 7. non-diff-fn-wrap ex2 — graph terminator via sorted_computational_graph
# =========================================================================

SPEC_NON_DIFF = _spec(
    atom_id="non-diff-fn-wrap",
    subtopic="Backprop: non-differentiable fn wrap",
    recap=RECAP_NON_DIFF_TERMINATES_DEEP,
    ex_title="confirm sorted_computational_graph stops at a non-differentiable node",
    slug="confirm-sorted-computational-graph-stops-at-non-diff-node",
    bloom="Analyze",
    difficulty_num=3,
    keywords=["non-differentiable", "graph-terminator", "leaf", "topo-walk"],
    kcs=["non-diff-fn-wrap", "is-differentiable-flag"],
    lo=(
        "Analyze the graph-termination consequence of recipe=None: a non-"
        "differentiable op's output behaves as a leaf during reverse-pass "
        "traversal, so its tensor parents are not reachable through it."
    ),
    prompt_body=(
        "ex1 verified the wrapper's local behavior: non-diff op → "
        "`requires_grad=False` and `recipe=None`. Here we exercise the "
        "DOWNSTREAM consequence: when `sorted_computational_graph` walks "
        "parents, it stops at any node with `recipe=None`.\n\n"
        "You implement TWO things:\n\n"
        "1. **`wrap_forward_fn(fwd_fn, is_differentiable=True)`** — same as "
        "ex1 (full wrapper that builds the Recipe conditionally). Provided "
        "in the stub as a guide; you finish the body.\n"
        "2. **Build a 4-node compute graph by hand** and call "
        "`sorted_computational_graph(end_node)` (provided). Verify that the "
        "non-differentiable node terminates the walk.\n\n"
        "We've provided `sorted_computational_graph` for you. You just have "
        "to build the graph and verify the structural property in the "
        "test body — that's the Analyze-level Bloom: you're examining the "
        "GRAPH SHAPE, not deriving math.\n\n"
        "Setup cell provides `MiniTensor`, `Recipe`, `grad_tracking_enabled`. "
        "Don't call `torch.autograd`."
    ),
    stub=(
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = tuple(a.array if isinstance(a, MiniTensor) else a for a in args)\n"
        "        out_arr = fwd_fn(*raw_args, **kwargs)\n"
        "        # TODO: three-gate AND for requires_grad.\n"
        "        # TODO: build MiniTensor; attach Recipe only when requires_grad.\n"
        "        raise NotImplementedError()\n"
        "    return tensor_func\n"
        "\n"
        "\n"
        "def sorted_computational_graph(tensor):\n"
        "    \"\"\"Provided: reverse-topo sort (end node first). Stops at recipe=None.\"\"\"\n"
        "    result = []\n"
        "    perm = set()\n"
        "    def visit(cur):\n"
        "        if id(cur) in perm:\n"
        "            return\n"
        "        perm.add(id(cur))\n"
        "        if cur.recipe is not None:\n"
        "            for p in cur.recipe.parents.values():\n"
        "                visit(p)\n"
        "        result.append(cur)\n"
        "    visit(tensor)\n"
        "    return result[::-1]"
    ),
    test_body=(
        "globals()['grad_tracking_enabled'] = True\n"
        "add = wrap_forward_fn(t.add)\n"
        "eq  = wrap_forward_fn(t.eq, is_differentiable=False)\n"
        "mul = wrap_forward_fn(t.multiply)\n"
        "\n"
        "# --- build:\n"
        "#   a, b are leaves\n"
        "#   mask = eq(a, b)       <-- non-diff: graph terminates here\n"
        "#   c    = add(a, b)\n"
        "#   d    = mul(c, c)      <-- end node\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([1.0, 0.0, 3.0]), requires_grad=True)\n"
        "mask = eq(a, b)\n"
        "c = add(a, b)\n"
        "d = mul(c, c)\n"
        "\n"
        "# --- structural invariants on the WRAPPED outputs ---\n"
        "assert mask.recipe is None, 'eq output must have recipe=None (graph terminator)'\n"
        "assert mask.requires_grad is False\n"
        "assert c.recipe is not None and c.recipe.func is t.add\n"
        "assert d.recipe is not None and d.recipe.func is t.multiply\n"
        "\n"
        "# --- walk from d: only d, c, a, b reachable; mask is NOT in the graph ---\n"
        "graph = sorted_computational_graph(d)\n"
        "ids = {id(n) for n in graph}\n"
        "assert id(d) in ids and id(c) in ids and id(a) in ids and id(b) in ids\n"
        "assert id(mask) not in ids, (\n"
        "    'mask not consumed by d — should not appear; this is a sanity check'\n"
        ")\n"
        "\n"
        "# --- walk from mask: traversal STOPS at mask (it is itself a leaf) ---\n"
        "mask_graph = sorted_computational_graph(mask)\n"
        "mask_ids = {id(n) for n in mask_graph}\n"
        "# mask has recipe=None → topo walk treats it as a leaf, so only `mask` appears.\n"
        "assert mask_ids == {id(mask)}, (\n"
        "    f'non-diff output should be a graph leaf, but walk found {mask_ids} '\n"
        "    f'(expected {{id(mask)}}) — recipe-None termination broken'\n"
        ")\n"
        "assert id(a) not in mask_ids and id(b) not in mask_ids, (\n"
        "    'parents of non-diff op MUST NOT be reachable through it'\n"
        ")\n"
        "\n"
        "# --- now build a graph that CONSUMES mask — the same termination still holds ---\n"
        "# (We can multiply tensors elementwise by a bool tensor — torch promotes.)\n"
        "weighted = mul(c, mask)   # c * mask. Note mask.requires_grad=False → weighted...\n"
        "# weighted.requires_grad is True (c is tracked), but its parents dict only contains\n"
        "# MiniTensor inputs. The walk reaches mask via parents and STOPS there.\n"
        "weighted_graph = sorted_computational_graph(weighted)\n"
        "w_ids = {id(n) for n in weighted_graph}\n"
        "assert id(weighted) in w_ids and id(c) in w_ids and id(mask) in w_ids\n"
        "# Crucially: mask's parents (a, b through eq) are NOT reachable via mask.\n"
        "# But a and b ARE reachable via c. So they appear — just not THROUGH mask.\n"
        "# Verify by checking the position invariant.\n"
        "pos = {id(n): i for i, n in enumerate(weighted_graph)}\n"
        "# mask comes after weighted (since weighted's recipe lists mask as parent).\n"
        "assert pos[id(weighted)] < pos[id(mask)], 'parent before child in reverse order'\n"
        "# But mask is itself a leaf in the walk — its 'parents' (a, b through eq) are\n"
        "# not traversed THROUGH mask. They're traversed THROUGH c instead.\n"
        "# We assert by replacing c temporarily so mask is the only path to a/b:\n"
        "# easier: just verify the walk would have crashed without termination by\n"
        "# checking that mask's recipe is None.\n"
        "assert mask.recipe is None, 'final invariant: mask terminates the walk'"
    ),
    solution_body=(
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = tuple(a.array if isinstance(a, MiniTensor) else a for a in args)\n"
        "        out_arr = fwd_fn(*raw_args, **kwargs)\n"
        "        requires_grad = (\n"
        "            globals()['grad_tracking_enabled']\n"
        "            and is_differentiable\n"
        "            and any(isinstance(a, MiniTensor) and a.requires_grad for a in args)\n"
        "        )\n"
        "        out = MiniTensor(out_arr, requires_grad)\n"
        "        if requires_grad:\n"
        "            parents = {i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
        "            out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    solution_notes=(
        "**Why graph termination is necessary, not optional.** Without it, "
        "`sorted_computational_graph` recursing into `mask.recipe.parents` "
        "would walk into `(a, b)` via the eq op, then the reverse pass "
        "would look up `(t.eq, 0)` in `BACK_FUNCS` — `KeyError`. The "
        "`recipe is None` short-circuit is what makes detach / eq / argmax "
        "safe to use mid-graph.\n\n"
        "**Why this is an Analyze-level exercise.** You're not computing "
        "gradients or implementing math — you're examining the GRAPH SHAPE "
        "to confirm a structural invariant holds. The cognitive operation "
        "is verification of a system-level property: 'does this design "
        "decision (recipe=None for non-diff) prevent the failure mode it "
        "was supposed to prevent?'\n\n"
        "**Why mask's parents are still reachable via c.** Each downstream "
        "tensor knows its own parents; multiple paths to the same leaf is "
        "the diamond-DAG pattern. The point is that they're not reachable "
        "THROUGH the non-differentiable node. Without termination, the "
        "walk would erroneously go through mask AND through c, and the "
        "dispatcher would crash on the eq lookup."
    ),
)


# =========================================================================
# 8. sorted-computational-graph ex2 — shared subgraph with multi-depth reuse
# =========================================================================

SPEC_SORTED = _spec(
    atom_id="sorted-computational-graph",
    subtopic="Backprop: Sorted computation graph",
    recap=RECAP_SORTED_GRAPH_SHARED_DEEP,
    ex_title="topo-sort a multi-depth shared-leaf compute graph (each node once)",
    slug="topo-sort-multi-depth-shared-leaf-compute-graph",
    bloom="Apply",
    difficulty_num=4,
    keywords=["topo-sort", "shared-leaf", "dedup", "multi-depth-dag"],
    kcs=["sorted-computational-graph", "parents-dict-by-argidx"],
    lo=(
        "Apply topological sort over a compute graph with a leaf shared by "
        "multiple intermediate consumers at different depths, ensuring each "
        "node appears exactly once and parent-before-child holds across all "
        "edges in the reverse order."
    ),
    prompt_body=(
        "Implement `topological_sort(node, get_children)` and "
        "`sorted_computational_graph(tensor)`. Specs are the same as ex1, "
        "but the test graph here is HARDER:\n\n"
        "```\n"
        "Compute graph (read top-down — each line is a recipe):\n"
        "leaf:  a\n"
        "       b\n"
        "node:  u = a * b\n"
        "       v = log(a)       <-- a shared with u\n"
        "       w = u + v        <-- merges two ancestors of a\n"
        "       z = w * a        <-- a appears again at depth 1!\n"
        "end:   z\n"
        "```\n\n"
        "Properties this graph exercises:\n"
        "- `a` is consumed by FOUR nodes at three distinct depths "
        "(depth-3 via u, depth-2 via v, depth-1 directly into z).\n"
        "- `b` is consumed by exactly one node.\n"
        "- The DAG fans in at `w` and fans out from `a`.\n\n"
        "Your sort must:\n"
        "1. Visit each node EXACTLY ONCE (no duplicates of `a`).\n"
        "2. End node `z` is FIRST in the result.\n"
        "3. For EVERY edge `(parent → child)` in the recipe graph: "
        "`pos[child] < pos[parent]` in the result.\n"
        "4. Cycle detection still works (we'll re-test it).\n\n"
        "Implementation: same three-color DFS pattern. The shared-leaf case "
        "is what the `perm` set is FOR — it stops the second visit to `a` "
        "from duplicating it.\n\n"
        "Use `id(...)` for set membership (MiniTensors aren't hashable by "
        "value)."
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
        "# --- cycle detection still works ---\n"
        "class N:\n"
        "    def __init__(self, name, *children):\n"
        "        self.name = name\n"
        "        self.children = list(children)\n"
        "    def __repr__(self):\n"
        "        return f'N({self.name})'\n"
        "\n"
        "x = N('x'); y = N('y')\n"
        "x.children = [y]; y.children = [x]\n"
        "try:\n"
        "    topological_sort(x, lambda n: n.children)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('cycle should raise ValueError')\n"
        "\n"
        "# --- build the multi-depth shared-leaf graph ---\n"
        "a = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "u = MiniTensor(a.array * b.array, requires_grad=True)\n"
        "u.recipe = Recipe(func=t.multiply, args=(a.array, b.array), kwargs={}, parents={0: a, 1: b})\n"
        "v = MiniTensor(t.log(a.array), requires_grad=True)\n"
        "v.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "w = MiniTensor(u.array + v.array, requires_grad=True)\n"
        "w.recipe = Recipe(func=t.add, args=(u.array, v.array), kwargs={}, parents={0: u, 1: v})\n"
        "z = MiniTensor(w.array * a.array, requires_grad=True)\n"
        "z.recipe = Recipe(func=t.multiply, args=(w.array, a.array), kwargs={}, parents={0: w, 1: a})\n"
        "\n"
        "order = sorted_computational_graph(z)\n"
        "\n"
        "# --- every node appears EXACTLY ONCE (despite a being shared by u, v, z) ---\n"
        "ids = [id(n) for n in order]\n"
        "from collections import Counter\n"
        "counts = Counter(ids)\n"
        "for label, node in [('a', a), ('b', b), ('u', u), ('v', v), ('w', w), ('z', z)]:\n"
        "    assert counts[id(node)] == 1, (\n"
        "        f'{label} appears {counts[id(node)]} times in topo sort (expected 1) '\n"
        "        f'— shared-leaf dedup broke'\n"
        "    )\n"
        "\n"
        "# --- exactly 6 nodes, no extras ---\n"
        "assert len(order) == 6, f'expected 6 nodes, got {len(order)}: {[id(n) for n in order]}'\n"
        "assert set(ids) == {id(a), id(b), id(u), id(v), id(w), id(z)}\n"
        "\n"
        "# --- end node first ---\n"
        "assert order[0] is z, f'first should be z, got {order[0]}'\n"
        "\n"
        "# --- parent-before-child in reverse-topo: pos[child] < pos[parent] for every edge ---\n"
        "pos = {id(n): i for i, n in enumerate(order)}\n"
        "edges = [\n"
        "    (z, w), (z, a),   # z's parents\n"
        "    (w, u), (w, v),   # w's parents\n"
        "    (u, a), (u, b),   # u's parents\n"
        "    (v, a),           # v's parents\n"
        "]\n"
        "for child, parent in edges:\n"
        "    assert pos[id(child)] < pos[id(parent)], (\n"
        "        f'reverse-topo violated for edge ({child.recipe.func.__name__ if child.recipe else \"leaf\"})'\n"
        "        f'@{pos[id(child)]} → {parent}@{pos[id(parent)]}'\n"
        "    )\n"
        "\n"
        "# --- `a` is the deepest-shared node: should appear AFTER u, v, AND z ---\n"
        "# (z → a edge, u → a edge, v → a edge — all three say a comes later.)\n"
        "assert pos[id(a)] > pos[id(u)], 'a after u (z → w → u → a)'\n"
        "assert pos[id(a)] > pos[id(v)], 'a after v (z → w → v → a)'\n"
        "assert pos[id(a)] > pos[id(z)], 'a after z (z → a direct)'\n"
        "\n"
        "# --- singleton graph (just a leaf) still works ---\n"
        "lonely = MiniTensor(t.tensor([5.0]), requires_grad=True)\n"
        "order_lone = sorted_computational_graph(lonely)\n"
        "assert order_lone == [lonely], f'singleton: {order_lone}'"
    ),
    solution_body=(
        "def topological_sort(node, get_children):\n"
        "    result = []\n"
        "    perm = set()   # fully processed nodes — keyed by id()\n"
        "    temp = set()   # on the DFS stack — cycle detector\n"
        "\n"
        "    def visit(cur):\n"
        "        cid = id(cur)\n"
        "        if cid in perm:\n"
        "            return\n"
        "        if cid in temp:\n"
        "            raise ValueError(f'Cycle at {cur!r} — graph is not a DAG')\n"
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
        "    def get_parents(t_):\n"
        "        if t_.recipe is None:\n"
        "            return []\n"
        "        return list(t_.recipe.parents.values())\n"
        "    return topological_sort(tensor, get_parents)[::-1]"
    ),
    solution_notes=(
        "**Why the perm set is necessary (not just temp).** A two-color "
        "DFS (visited / unvisited) would either: revisit shared leaves and "
        "duplicate them in the output, OR mark them after first visit but "
        "have no way to distinguish 'finished this subtree' from "
        "'currently processing.' The perm/temp split handles BOTH the "
        "shared-leaf dedup case (perm) AND the cycle detection (temp).\n\n"
        "**Why `a` ends up at the BACK of the reverse-topo result.** `a` "
        "has three parent edges pointing INTO it (from u, v, z) — in the "
        "forward direction, `a` is the deepest dependency. After "
        "reversal, the deepest dependencies come last. The position-"
        "invariant test pins this down: `pos[a]` must exceed `pos[u]`, "
        "`pos[v]`, AND `pos[z]`.\n\n"
        "**Why id(...) and not the object itself.** MiniTensors compare by "
        "identity by default (we didn't override `__eq__`), so `set(...)` "
        "of MiniTensors works — but for safety, `id(...)` is robust to "
        "any future change in equality semantics. PyTorch's autograd "
        "graph uses identity keys for the same reason."
    ),
)


# ---------------------------------------------------------------------------
# SPECS + verify + emit
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_BACK_LOOKUP,
    SPEC_END_GRAD,
    SPEC_IS_DIFF,
    SPEC_LOG_BACK,
    SPEC_MAX_BACK,
    SPEC_MULTIPLY_BACK,
    SPEC_NON_DIFF,
    SPEC_SORTED,
]


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

        # 1. Run the autograd preamble (MiniTensor, Recipe, grad_tracking_enabled).
        for preamble in spec.get("extra_imports", []):
            try:
                exec(preamble, ns)
            except Exception as e:
                failed.append((tag, f"preamble: {e!r}", traceback.format_exc()))
                break
        else:
            # 2. Exec stub (may have NotImplementedError, but defines names).
            try:
                exec(spec["stub"], ns)
            except Exception:
                pass

            # 3. Overwrite with solution, then run tests.
            try:
                exec(spec["solution_body"], ns)
                exec(spec["test_body"], ns)
            except Exception as e:
                failed.append((tag, repr(e), traceback.format_exc()))
                continue
            passed += 1
            print(f"  [verify] {tag}: ok")
            continue
        # preamble loop broke → already recorded as failure
        continue

    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, err, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(err)
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[deepening_j_batch9] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_j_batch9] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_j_batch9] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
