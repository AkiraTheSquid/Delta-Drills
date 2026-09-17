#!/usr/bin/env python3
"""Author 8 ex3 deepening drills (batch 14, group D — prereqs_autograd_pt2).

Atoms (all 8 already have ex1+ex2 — this adds the third facet):
    - backward-func-lookup        (ex3: alias one back_fn under multiple keys; symmetric ops)
    - end-grad-default-ones-like  (ex3: 0-D scalar end-node — ones_like gives tensor(1.0))
    - is-differentiable-flag      (ex3: global grad_tracking_enabled=False overrides is_differentiable=True)
    - log-back                    (ex3: domain-edge blowup — small-x behaviour, matches torch.autograd)
    - max-back-tied-half          (ex3: mass-conservation — back0+back1 sums to grad_out for every tie pattern)
    - multiply-back               (ex3: 3-D multi-axis unbroadcast — (5,1,4) * (1,3,1))
    - non-diff-fn-wrap            (ex3: chained non-diff — argmax(eq(a,b)); both terminate the graph)
    - sorted-computational-graph  (ex3: diamond a -> {b,c} -> d — node a appears once, both paths respected)

Each ex3 hits a DISTINCT facet from ex1 AND ex2. ONE LO + ONE Bloom + <=2 KCs per drill.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_pt2"

# Shared autograd preamble — mirror author_autograd_pt2_batch4.py so MiniTensor/Recipe
# are available in every notebook's Setup cell (via extra_imports prepend).
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
# Recap blocks (each emphasises the ex3 facet, not ex1/ex2)
# ---------------------------------------------------------------------------

RECAP_BFL_ALIAS = (
    "## BackwardFuncLookup — aliasing one back_fn under multiple keys\n"
    "\n"
    "Ex1 built the class; ex2 dispatched it across a 2-op reverse pass. The deepening "
    "move exploits a property the dispatcher REQUIRES but neither earlier ex tested: "
    "the SAME `back_fn` object can be registered under multiple `(fwd, argnum)` keys.\n"
    "\n"
    "**Where this comes up.** Symmetric binary ops — `add`, `multiply` — have "
    "identical (up to operand order) back-fns at both argnums:\n"
    "\n"
    "```\n"
    "add_back(grad_out, out, x, y) = grad_out          # same for argnum 0 AND argnum 1\n"
    "BFL.add_back_func(t.add, 0, add_back)\n"
    "BFL.add_back_func(t.add, 1, add_back)             # same function object both times\n"
    "```\n"
    "\n"
    "**Both lookups must return THE SAME object** (same `id`), not a clone. This is "
    "what lets the dispatcher route either argnum to the shared implementation "
    "without paying for duplicate registration storage or risking divergence "
    "between two copies. ARENA's actual back-fn table relies on this — most "
    "single-function entries are also aliased across argnums for the symmetric ops."
)

RECAP_END_GRAD_SCALAR = (
    "## end_grad on a 0-D scalar end-node — the loss-tip case\n"
    "\n"
    "Ex1 covered the default-vs-explicit branch on a multi-element tensor; ex2 used "
    "a `(B,)` per-sample-weighted loss. The deepening move handles the MOST COMMON "
    "real case: the end-node is the loss itself — a 0-D scalar.\n"
    "\n"
    "```python\n"
    "loss = (x.array ** 2).sum()         # 0-D scalar — shape ()\n"
    "end_node = MiniTensor(loss, ...)\n"
    "\n"
    "# Default path: ones_like(scalar) -> tensor(1.0), shape ()\n"
    "end_grad = t.ones_like(end_node.array)\n"
    "assert end_grad.shape == ()\n"
    "assert end_grad.item() == 1.0\n"
    "```\n"
    "\n"
    "**Why scalar end-grad always seeds to 1.0.** Calling `loss.backward()` in real "
    "torch corresponds to `dL/dL = 1` — the identity. Our manual seed is the same: "
    "`ones_like` on a scalar gives `tensor(1.0)`, exactly the chain-rule identity.\n"
    "\n"
    "**The shape-mismatch failure mode.** If a caller PASSES an explicit `end_grad` "
    "of the wrong rank — e.g. `(B,)` against a 0-D end-node — the resolver must "
    "raise an `AssertionError`. Silently broadcasting is wrong: it would scale every "
    "leaf by `B` copies of the seed and break gradient accounting."
)

RECAP_IS_DIFF_GLOBAL_TOGGLE = (
    "## is_differentiable + grad_tracking_enabled — global toggle wins\n"
    "\n"
    "Ex1 + ex2 fixed `grad_tracking_enabled=True` and varied `is_differentiable`. "
    "The deepening move flips the OTHER gate: with `grad_tracking_enabled=False` "
    "(inside a `no_grad()`-style block), no op can produce a grad-tracked output, "
    "no matter how the per-op flag is set.\n"
    "\n"
    "```\n"
    "requires_grad = (grad_tracking_enabled       # GLOBAL — short-circuits everything\n"
    "                 AND is_differentiable        # per-op closure flag\n"
    "                 AND any(input.requires_grad))\n"
    "```\n"
    "\n"
    "The three gates are a strict AND. Setting any one to `False` zeros the output's "
    "`requires_grad` AND skips Recipe construction — exactly like setting "
    "`is_differentiable=False` in ex2.\n"
    "\n"
    "**Why the global toggle takes priority semantically.** A user writing "
    "`with no_grad():` is saying 'no autograd activity at all'. That intent must "
    "win over any per-op default. Concretely: a differentiable op like `add` with "
    "tracked inputs, called inside `no_grad`, must produce `requires_grad=False` "
    "and `recipe=None` — indistinguishable from a non-diff op's output.\n"
    "\n"
    "**Re-entry behaviour.** Flipping `grad_tracking_enabled` back to `True` "
    "restores normal behaviour on the next call — the flag is read FRESH each "
    "invocation, not captured at wrap-time (unlike `is_differentiable`)."
)

RECAP_LOG_BACK_DOMAIN_EDGE = (
    "## log_back at the domain edge — `1/x` blows up near zero\n"
    "\n"
    "Ex1 derived `log_back(grad_out, out, x) = grad_out / x` and ex2 composed it "
    "with multiply_back. The deepening move exercises the DOMAIN BEHAVIOUR: "
    "`log(x)` is only defined for `x > 0`, and `log_back` inherits a `1/x` "
    "singularity at the origin.\n"
    "\n"
    "```\n"
    "x = 1.0          ->  log_back(1, _, 1)    = 1.0           (tame)\n"
    "x = 1e-10        ->  log_back(1, _, 1e-10) = 1e10          (large)\n"
    "x = 1e-30        ->  log_back(1, _, 1e-30) = 1e30          (huge — float overflow nearby)\n"
    "x = 0.0          ->  log_back(1, _, 0)    = +inf           (division)\n"
    "x < 0            ->  forward log(x) was already nan; back is nan too\n"
    "```\n"
    "\n"
    "**The drill verifies our hand-rolled `log_back` matches `torch.autograd` "
    "BIT-FOR-BIT on the same inputs** — even at the blow-up. That's the test: "
    "real autograd doesn't 'protect' you from `1/x`; it just computes it. If "
    "your model overflows at log of a near-zero input, the fix is upstream "
    "(clamp, eps-add), not in the back-fn.\n"
    "\n"
    "**Why no `eps` in log_back.** Some libraries silently add `+ 1e-8` to the "
    "denominator. PyTorch doesn't — and our drill doesn't either. Correctness "
    "= matching the closed-form gradient exactly; numerical stability is the "
    "caller's job."
)

RECAP_MAX_BACK_MASS_CONSERVATION = (
    "## maximum_back — mass conservation across half-mass ties\n"
    "\n"
    "Ex1 derived the half-mass tie rule; ex2 specialised it to `relu_back`. The "
    "deepening move tests a STRUCTURAL INVARIANT both earlier exercises rely on "
    "but never check directly: per-position grad mass is conserved.\n"
    "\n"
    "```\n"
    "for every position i:\n"
    "    maximum_back0[i] + maximum_back1[i]  ==  grad_out[i]\n"
    "```\n"
    "\n"
    "Case-by-case proof:\n"
    "\n"
    "| condition       | bool_sum_x       | bool_sum_y       | sum   |\n"
    "|-----------------|------------------|------------------|-------|\n"
    "| `x > y`         | 1.0              | 0.0              | 1.0   |\n"
    "| `x < y`         | 0.0              | 1.0              | 1.0   |\n"
    "| `x == y` (tie)  | 0.5              | 0.5              | 1.0   |\n"
    "\n"
    "Multiply both sides by `grad_out[i]`: the sum of per-arg-position grads "
    "equals `grad_out[i]` at every position, with no leak and no double-count.\n"
    "\n"
    "**Why this matters.** Mass conservation is the autograd correctness oracle "
    "for any back-fn that splits a single gradient across multiple parents. A "
    "naive implementation that returned `grad_out * (x >= y)` for BOTH back0 "
    "and back1 would double-count at the kink (sum = `2 * grad_out`). The "
    "half-mass rule is the unique split that conserves mass AND remains "
    "symmetric across the tie."
)

RECAP_MULTIPLY_BACK_3D_BROADCAST = (
    "## multiply_back — 3-D broadcasting and multi-axis unbroadcast\n"
    "\n"
    "Ex1 covered `(1,4) * (3,4)` (single broadcast axis); ex2 covered Python-float "
    "operands. The deepening move stretches both back-fns over a 3-D broadcast "
    "where EACH side has a different broadcast axis:\n"
    "\n"
    "```\n"
    "x = randn(5, 1, 4)        # singleton dim at axis 1\n"
    "y = randn(1, 3, 1)        # singleton dims at axes 0 AND 2\n"
    "out = x * y               # shape (5, 3, 4) — broadcasts both sides\n"
    "```\n"
    "\n"
    "Working backward through the per-arg-position rule:\n"
    "\n"
    "```\n"
    "grad_x_pre = grad_out * y                          # shape (5,3,4)\n"
    "grad_x     = unbroadcast(grad_x_pre, x)            # -> (5,1,4), sums axis 1\n"
    "\n"
    "grad_y_pre = grad_out * x                          # shape (5,3,4)\n"
    "grad_y     = unbroadcast(grad_y_pre, y)            # -> (1,3,1), sums axes 0 AND 2\n"
    "```\n"
    "\n"
    "**Why `unbroadcast` walks multiple axes.** The helper iterates the broadcast "
    "rules in reverse: for each axis where `original.shape[i] == 1` but "
    "`grad.shape[i] > 1`, sum-reduce with `keepdim=True`. Multi-axis cases "
    "exercise the inner loop multiple times — easy to get wrong if the helper "
    "uses a single `dim=0` reduction instead of an axis-by-axis sweep.\n"
    "\n"
    "**Matches `torch.autograd`.** The drill compares the hand-rolled grads to "
    "`(x_t * y_t).sum().backward()` results — should match to within 1e-6."
)

RECAP_NON_DIFF_CHAIN = (
    "## Non-diff op chained into another non-diff op — graph stays isolated\n"
    "\n"
    "Ex1 wrapped a single non-diff op (`eq`); ex2 confirmed a non-diff branch is "
    "skipped during topo walk from a diff end-node. The deepening move chains TWO "
    "non-diff ops in series:\n"
    "\n"
    "```\n"
    "mask = eq(a, b)            # non-diff: requires_grad=False, recipe=None\n"
    "idx  = argmax(mask)        # non-diff: same — operates on a non-tracked input\n"
    "```\n"
    "\n"
    "The interesting consequence: BOTH outputs have `recipe=None`. The graph "
    "walk from `idx` cannot reach `mask`, `a`, or `b` — it stops at `idx` itself "
    "(graph = `[idx]`). Each non-diff output is its own isolated leaf in the "
    "autograd-graph view, regardless of how many non-diff ops produced it.\n"
    "\n"
    "**Why this is the right behaviour.** A chain of non-diff ops produces a "
    "tensor that cannot receive a meaningful grad — there's no chain rule that "
    "would let one flow. Treating it as a leaf in the graph is correct: the "
    "reverse pass simply has no work to do at it.\n"
    "\n"
    "**Contrast with ex2.** Ex2 isolated ONE non-diff branch on a graph whose "
    "main path was differentiable. Here, EVERY op is non-diff — the entire "
    "subgraph collapses into independent single-node graphs."
)

RECAP_SORT_DIAMOND = (
    "## Diamond compute graph — one shared leaf, two paths, single visit\n"
    "\n"
    "Ex1 sorted a simple linear chain; ex2 handled a multi-depth shared-leaf "
    "graph. The deepening move pins down the canonical 'diamond' DAG pattern, "
    "which appears anywhere a single tensor flows through two parallel ops "
    "and merges back:\n"
    "\n"
    "```\n"
    "        a       (leaf)\n"
    "       / \\\n"
    "      b   c     (b = log(a), c = neg(a))\n"
    "       \\ /\n"
    "        d       (d = b * c)\n"
    "```\n"
    "\n"
    "**The four invariants the sort must satisfy.**\n"
    "1. Each of `{a, b, c, d}` appears EXACTLY once.\n"
    "2. `d` is first in the reverse-topo order.\n"
    "3. `b` and `c` BOTH appear before `a` (parent-before-child holds across "
    "BOTH paths through the diamond, not just one).\n"
    "4. An unrelated leaf `z` (not consumed by `d`) is NOT visited.\n"
    "\n"
    "**Why the `perm` set is the load-bearing part.** Without `id(...)` "
    "membership de-duplication, the second visit to `a` (via `c`'s parent "
    "list) would add it again. The sort would then have either `a` "
    "duplicated OR `b`/`c` placed wrongly. The diamond is the smallest "
    "graph that exposes this bug — a simple chain doesn't.\n"
    "\n"
    "**Generalisation.** Any DAG with shared ancestry is a 'union of diamonds'. "
    "Getting the diamond right is sufficient to get arbitrary DAGs right."
)


# ---------------------------------------------------------------------------
# SPEC 1 — backward-func-lookup ex3 (aliased back_fn)
# ---------------------------------------------------------------------------

SPEC_BFL = {
    "atom_id": "backward-func-lookup",
    "subtopic": "Backprop: BackwardFuncLookup",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BFL_ALIAS,
    "exercise_index": 3,
    "exercise_title": "alias one back_fn under multiple (fwd,argnum) keys",
    "slug": "alias-one-back-fn-under-multiple-fwd-argnum-keys",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["backward-func-lookup", "alias", "symmetric-op", "shared-fn"],
    "kcs": [
        "shared-back-fn-aliasing",
        "lookup-returns-same-object",
    ],
    "lo": (
        "Apply the BackwardFuncLookup contract that the same back_fn object can "
        "be registered under multiple (fwd, argnum) keys, and verify both "
        "lookups return THE SAME object (identity, not equality)."
    ),
    "prompt_body": (
        "Implement `BackwardFuncLookup` (same `add_back_func` + `get_back_func` "
        "API as ex1) AND a registration helper `register_symmetric(BFL, fwd_fn, "
        "back_fn)` that aliases the SINGLE `back_fn` under BOTH `argnum=0` and "
        "`argnum=1` keys.\n\n"
        "Requirements:\n\n"
        "1. `BackwardFuncLookup.__init__()` — initialise an empty dict.\n"
        "2. `add_back_func(forward_fn, argnum, back_fn)` — store at "
        "`(forward_fn, argnum)`.\n"
        "3. `get_back_func(forward_fn, argnum)` — return the stored fn, raise "
        "`KeyError` if missing.\n"
        "4. `register_symmetric(BFL, fwd_fn, back_fn)` — call `add_back_func` "
        "TWICE with the SAME `back_fn` object (argnum 0, argnum 1).\n\n"
        "The test then verifies that `BFL.get_back_func(t.add, 0)` and "
        "`BFL.get_back_func(t.add, 1)` return the IDENTICAL function object "
        "(`is` check, not just `==`). It also verifies the symmetric pattern "
        "for `t.multiply` — `multiply_back` aliased under both argnums — and "
        "confirms updating the shared fn affects both lookups together "
        "(showing they're truly the same object)."
    ),
    "stub": (
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def add_back_func(self, forward_fn, argnum, back_fn):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def get_back_func(self, forward_fn, argnum):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "\n"
        "def register_symmetric(BFL, fwd_fn, back_fn):\n"
        '    """Register `back_fn` under BOTH (fwd_fn, 0) and (fwd_fn, 1)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- basic class works (smoke test) ---\n"
        "BFL = BackwardFuncLookup()\n"
        "def dummy(grad_out, out, x): return grad_out\n"
        "BFL.add_back_func(t.log, 0, dummy)\n"
        "assert BFL.get_back_func(t.log, 0) is dummy\n"
        "\n"
        "# --- KeyError on missing key ---\n"
        "raised = False\n"
        "try:\n"
        "    BFL.get_back_func(t.log, 99)\n"
        "except KeyError:\n"
        "    raised = True\n"
        "assert raised, 'missing key should raise KeyError'\n"
        "\n"
        "# --- the headline: register_symmetric aliases under BOTH argnums ---\n"
        "BFL = BackwardFuncLookup()\n"
        "def add_back(grad_out, out, x, y): return grad_out\n"
        "register_symmetric(BFL, t.add, add_back)\n"
        "fn0 = BFL.get_back_func(t.add, 0)\n"
        "fn1 = BFL.get_back_func(t.add, 1)\n"
        "assert fn0 is add_back, f'argnum 0 should be the registered fn, got {fn0}'\n"
        "assert fn1 is add_back, f'argnum 1 should be the registered fn, got {fn1}'\n"
        "assert fn0 is fn1, 'aliased entries must be the SAME OBJECT, not separate copies'\n"
        "\n"
        "# --- same pattern for multiply (symmetric structure) ---\n"
        "def multiply_back(grad_out, out, x, y): return grad_out * y\n"
        "register_symmetric(BFL, t.multiply, multiply_back)\n"
        "m0 = BFL.get_back_func(t.multiply, 0)\n"
        "m1 = BFL.get_back_func(t.multiply, 1)\n"
        "assert m0 is m1 is multiply_back, 'multiply aliasing broken'\n"
        "\n"
        "# --- aliasing is per (fwd_fn) — different ops keep their own pair ---\n"
        "assert BFL.get_back_func(t.add, 0) is add_back\n"
        "assert BFL.get_back_func(t.multiply, 0) is multiply_back\n"
        "assert BFL.get_back_func(t.add, 0) is not BFL.get_back_func(t.multiply, 0)\n"
        "\n"
        "# --- adversarial: registering DIFFERENT fns at the two argnums shouldn't break the class\n"
        "BFL2 = BackwardFuncLookup()\n"
        "def back_a(grad_out, out, x, y): return grad_out + 1\n"
        "def back_b(grad_out, out, x, y): return grad_out + 2\n"
        "BFL2.add_back_func(t.subtract, 0, back_a)\n"
        "BFL2.add_back_func(t.subtract, 1, back_b)\n"
        "assert BFL2.get_back_func(t.subtract, 0) is back_a\n"
        "assert BFL2.get_back_func(t.subtract, 1) is back_b\n"
        "assert back_a is not back_b, 'sanity: the two fns are distinct'\n"
        "\n"
        "# --- overwrite aliased entry on BOTH argnums (re-aliasing) ---\n"
        "BFL3 = BackwardFuncLookup()\n"
        "def add_back_v1(grad_out, out, x, y): return grad_out * 1.0\n"
        "def add_back_v2(grad_out, out, x, y): return grad_out * 2.0\n"
        "register_symmetric(BFL3, t.add, add_back_v1)\n"
        "assert BFL3.get_back_func(t.add, 0) is add_back_v1\n"
        "register_symmetric(BFL3, t.add, add_back_v2)\n"
        "assert BFL3.get_back_func(t.add, 0) is add_back_v2\n"
        "assert BFL3.get_back_func(t.add, 1) is add_back_v2"
    ),
    "solution_body": (
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self.back_funcs = {}\n"
        "\n"
        "    def add_back_func(self, forward_fn, argnum, back_fn):\n"
        "        self.back_funcs[(forward_fn, argnum)] = back_fn\n"
        "\n"
        "    def get_back_func(self, forward_fn, argnum):\n"
        "        key = (forward_fn, argnum)\n"
        "        if key not in self.back_funcs:\n"
        "            raise KeyError(\n"
        "                f'no back-fn registered for fwd={getattr(forward_fn, \"__name__\", forward_fn)!r} argnum={argnum}'\n"
        "            )\n"
        "        return self.back_funcs[key]\n"
        "\n"
        "\n"
        "def ex3_register_symmetric(BFL, fwd_fn, back_fn):\n"
        "    BFL.add_back_func(fwd_fn, 0, back_fn)\n"
        "    BFL.add_back_func(fwd_fn, 1, back_fn)\n"
        "    return BFL\n"
        "\n"
        "register_symmetric = ex3_register_symmetric"
    ),
    "solution_notes": (
        "**`is` not `==`.** The test uses `is` because aliasing is about IDENTITY: "
        "the dispatcher must dispatch to one function object, not two. If a user "
        "(or a buggy register helper) accidentally registered `copy.copy(back_fn)` "
        "under one argnum, `==` would still hold for function objects but `is` "
        "would fail — exactly what we want to catch.\n\n"
        "**Why aliasing instead of a single `add_symmetric(fwd, back_fn)` method.** "
        "Two flat keys keep dispatch O(1) and uniform — the reverse pass never "
        "has to check 'is this op symmetric?'. The asymmetry handling lives only "
        "at REGISTRATION time, not at LOOKUP time.\n\n"
        "**Same trick for `t.add`, `t.multiply`, bitwise `t.bitwise_and`, etc.** "
        "Anywhere both back-fns are identical (or can be expressed as one fn with "
        "argnum-agnostic logic), aliasing saves an entry and a divergence risk."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 2 — end-grad-default-ones-like ex3 (0-D scalar loss tip)
# ---------------------------------------------------------------------------

SPEC_END_GRAD = {
    "atom_id": "end-grad-default-ones-like",
    "subtopic": "Backprop: end-grad ones_like default",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_END_GRAD_SCALAR,
    "exercise_index": 3,
    "exercise_title": "resolve end_grad on a 0-D scalar loss-tip",
    "slug": "resolve-end-grad-on-zero-d-scalar-loss-tip",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["end-grad", "scalar", "0-d", "loss-tip", "ones_like"],
    "kcs": [
        "scalar-end-grad-is-ones",
        "shape-mismatch-raises",
    ],
    "lo": (
        "Apply the .backward() entry-point convention to a 0-D scalar end-node: "
        "ones_like gives tensor(1.0), an explicit 0-D end_grad is unboxed as-is, "
        "and a non-scalar explicit end_grad raises AssertionError."
    ),
    "prompt_body": (
        "Implement `ex3_resolve_scalar_end_grad(end_node, end_grad)`.\n\n"
        "Inputs:\n"
        "- `end_node`: a `MiniTensor` wrapping a 0-D (scalar) `torch.Tensor` — "
        "this is the typical loss tensor.\n"
        "- `end_grad`: either `None` OR a `MiniTensor`.\n\n"
        "Behaviour:\n\n"
        "1. If `end_grad is None` — default-path. Return "
        "`t.ones_like(end_node.array)`. For a scalar end-node this is "
        "`tensor(1.0)` with shape `()`.\n"
        "2. If `end_grad` is a `MiniTensor` — explicit path. First assert "
        "`end_grad.array.shape == end_node.array.shape` with a helpful "
        "message that names both shapes. If they match, return "
        "`end_grad.array`.\n"
        "3. Output type: raw `torch.Tensor` (NOT a MiniTensor).\n\n"
        "Constraints:\n"
        "- Use `t.ones_like` for the default, NOT `t.tensor(1.0)`. "
        "`ones_like` preserves `device` and `dtype`.\n"
        "- The assertion failure must be an `AssertionError` (not a plain "
        "`ValueError`). The tests catch that specifically.\n"
        "- Do NOT call `torch.autograd`."
    ),
    "stub": (
        "def ex3_resolve_scalar_end_grad(end_node, end_grad):\n"
        '    """Resolve end_grad for a 0-D scalar end-node. Returns raw torch.Tensor."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- default path: 0-D end-node, end_grad=None -> ones_like (scalar 1.0) ---\n"
        "x = t.tensor(2.5)            # 0-D float scalar\n"
        "end_node = MiniTensor(x, requires_grad=True)\n"
        "seed = ex3_resolve_scalar_end_grad(end_node, None)\n"
        "assert isinstance(seed, t.Tensor), f'default seed must be a torch.Tensor; got {type(seed).__name__}'\n"
        "assert seed.shape == (), f'scalar end-node -> scalar seed; got shape {seed.shape}'\n"
        "assert seed.item() == 1.0, f'ones_like(scalar) == 1.0; got {seed.item()}'\n"
        "assert seed.dtype == x.dtype, f'dtype should be preserved; got {seed.dtype} vs {x.dtype}'\n"
        "\n"
        "# --- explicit path: 0-D end_grad MiniTensor of matching shape -> unboxed ---\n"
        "explicit = MiniTensor(t.tensor(7.5))\n"
        "seed = ex3_resolve_scalar_end_grad(end_node, explicit)\n"
        "assert isinstance(seed, t.Tensor)\n"
        "assert seed.shape == ()\n"
        "assert seed.item() == 7.5, f'explicit value passes through; got {seed.item()}'\n"
        "\n"
        "# --- shape mismatch: end_grad is (B,) against a 0-D end-node -> AssertionError ---\n"
        "mismatched = MiniTensor(t.ones(3))\n"
        "raised = False\n"
        "try:\n"
        "    ex3_resolve_scalar_end_grad(end_node, mismatched)\n"
        "except AssertionError as e:\n"
        "    raised = True\n"
        "    msg = str(e)\n"
        "    # The message must name BOTH shapes for diagnosis.\n"
        "    assert 'torch.Size([])' in msg or '()' in msg or 'shape' in msg.lower(), (\n"
        "        f'assertion message should mention shapes; got {msg!r}'\n"
        "    )\n"
        "assert raised, 'shape mismatch must raise AssertionError'\n"
        "\n"
        "# --- another shape mismatch: 2-D end_grad against 0-D end-node ---\n"
        "raised = False\n"
        "try:\n"
        "    ex3_resolve_scalar_end_grad(end_node, MiniTensor(t.ones(2, 3)))\n"
        "except AssertionError:\n"
        "    raised = True\n"
        "assert raised, '(2,3) vs () must also raise'\n"
        "\n"
        "# --- preserves dtype on non-default scalar dtypes (float64) ---\n"
        "x64 = t.tensor(3.14, dtype=t.float64)\n"
        "end_node64 = MiniTensor(x64, requires_grad=True)\n"
        "seed64 = ex3_resolve_scalar_end_grad(end_node64, None)\n"
        "assert seed64.dtype == t.float64, f'dtype preserved through ones_like; got {seed64.dtype}'\n"
        "assert seed64.item() == 1.0\n"
        "\n"
        "# --- semantic check: composed with log_back at the loss tip ---\n"
        "# end_node = log(x_leaf).sum() — 0-D scalar. seed=ones_like -> 1.0. \n"
        "# Then dL/dx_leaf = seed * (1/x_leaf) — should match torch.autograd.\n"
        "x_leaf_raw = t.tensor([1.0, 2.0, 4.0], requires_grad=True)\n"
        "loss = t.log(x_leaf_raw).sum()\n"
        "loss_mt = MiniTensor(loss.detach(), requires_grad=True)\n"
        "seed = ex3_resolve_scalar_end_grad(loss_mt, None)\n"
        "assert seed.item() == 1.0\n"
        "# Chain one step (this is the integration with log_back):\n"
        "# dL/dx_leaf = seed * (1/x_leaf) — but seed is 0-D, must broadcast over (3,).\n"
        "hand_grad = seed * (1.0 / x_leaf_raw.detach())\n"
        "loss.backward()\n"
        "assert t.allclose(hand_grad, x_leaf_raw.grad, atol=1e-6), (\n"
        "    f'hand-rolled scalar seed must match torch; got hand={hand_grad} torch={x_leaf_raw.grad}'\n"
        ")"
    ),
    "solution_body": (
        "def ex3_resolve_scalar_end_grad(end_node, end_grad):\n"
        "    if end_grad is None:\n"
        "        return t.ones_like(end_node.array)\n"
        "    assert end_grad.array.shape == end_node.array.shape, (\n"
        "        f'end_grad shape {tuple(end_grad.array.shape)} != '\n"
        "        f'end_node shape {tuple(end_node.array.shape)}'\n"
        "    )\n"
        "    return end_grad.array"
    ),
    "solution_notes": (
        "**`ones_like` not `torch.ones(end_node.array.shape)`.** `ones_like` "
        "preserves `device` AND `dtype`. Hardcoding shape+dtype defeats the "
        "purpose — if `end_node` is on CUDA float64, the seed must be CUDA "
        "float64 too, otherwise the first multiplication in the reverse pass "
        "triggers a device or dtype mismatch.\n\n"
        "**Why `AssertionError` not `ValueError`.** The autograd entry point "
        "treats shape mismatch as a CALLER BUG — they passed an `end_grad` "
        "that doesn't match. `assert` is the right form: it's a contract "
        "check, not a domain error. Real PyTorch raises `RuntimeError` "
        "instead, but for the manual-autograd drill, `AssertionError` is the "
        "convention used across the rest of the chain.\n\n"
        "**Scalar broadcast in the chain.** A 0-D seed multiplied by a "
        "non-scalar local grad (e.g. `1/x` where x is (3,)) broadcasts via "
        "PyTorch's normal rules — the seed scales every element uniformly. "
        "That's exactly what `loss.backward()` does in real torch."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 3 — is-differentiable-flag ex3 (global toggle interaction)
# ---------------------------------------------------------------------------

SPEC_IS_DIFF = {
    "atom_id": "is-differentiable-flag",
    "subtopic": "Backprop: is_differentiable flag",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_IS_DIFF_GLOBAL_TOGGLE,
    "exercise_index": 3,
    "exercise_title": "global grad_tracking toggle overrides is_differentiable",
    "slug": "global-grad-tracking-toggle-overrides-is-differentiable",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["grad-tracking-enabled", "no-grad", "global-toggle", "three-gate"],
    "kcs": [
        "global-toggle-short-circuits-three-gate",
        "fresh-read-of-runtime-flag",
    ],
    "lo": (
        "Analyze the three-gate AND for requires_grad to show that "
        "grad_tracking_enabled=False forces a False output regardless of "
        "is_differentiable=True OR any tracked inputs, and re-enabling restores "
        "normal behaviour on the next call (fresh read, not closure-captured)."
    ),
    "prompt_body": (
        "Implement `wrap_forward_fn(fwd_fn, is_differentiable=True)` — same "
        "signature as ex2 — but the test will exercise the GLOBAL toggle "
        "interaction.\n\n"
        "Requirements:\n\n"
        "1. Unbox MiniTensor inputs to raw arrays.\n"
        "2. Call `fwd_fn(*raw_args, **kwargs)`.\n"
        "3. **Three-gate AND** (READ GLOBAL FRESH AT CALL TIME, not at "
        "wrap-time): `grad_tracking_enabled` AND `is_differentiable` AND "
        "`any-tracked-input`.\n"
        "4. Box the result; attach `Recipe` only when `requires_grad=True`.\n\n"
        "The fresh-read part is critical: the test will flip the global "
        "flag mid-program and re-call the SAME wrapped op — the flag must "
        "take effect IMMEDIATELY on the next call, not stay at whatever it "
        "was when `wrap_forward_fn` ran.\n\n"
        "Implementation hint: read the flag via `globals()['grad_tracking_enabled']` "
        "or simply reference it by name inside the inner `tensor_func` "
        "(Python's name resolution will look it up fresh each call).\n\n"
        "Verify:\n"
        "- Global True + is_diff True + tracked input -> requires_grad=True, "
        "recipe is built.\n"
        "- Global False + is_diff True + tracked input -> requires_grad=False, "
        "recipe=None (global short-circuits).\n"
        "- Global True + is_diff False + tracked input -> requires_grad=False, "
        "recipe=None (per-op gate short-circuits — same as ex2).\n"
        "- Toggle global False -> True between calls: behaviour flips back. "
        "Without re-wrapping. The same `add` wrapped fn produces both kinds "
        "of outputs depending on the live value."
    ),
    "stub": (
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        '    """Three-gate wrapper with FRESH read of grad_tracking_enabled."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- Make sure global is ON for the canonical path ---\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "add = wrap_forward_fn(t.add)                                  # is_diff=True\n"
        "eq  = wrap_forward_fn(t.eq, is_differentiable=False)          # is_diff=False\n"
        "\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([4.0, 5.0, 6.0]), requires_grad=True)\n"
        "\n"
        "# --- canonical path: global=T, is_diff=T, tracked input -> requires_grad=T ---\n"
        "c = add(a, b)\n"
        "assert c.requires_grad is True, 'canonical path must produce requires_grad=True'\n"
        "assert c.recipe is not None, 'canonical path must build a Recipe'\n"
        "assert c.recipe.func is t.add\n"
        "\n"
        "# --- per-op gate alone: is_diff=False blocks the Recipe (this is ex2 territory but baseline) ---\n"
        "mask = eq(a, b)\n"
        "assert mask.requires_grad is False, 'is_diff=False must block requires_grad'\n"
        "assert mask.recipe is None\n"
        "\n"
        "# === THE HEADLINE: global toggle short-circuits even is_diff=True ===\n"
        "globals()['grad_tracking_enabled'] = False\n"
        "c2 = add(a, b)\n"
        "assert c2.requires_grad is False, (\n"
        "    f'with grad_tracking_enabled=False, even is_diff=True ops must produce '\n"
        "    f'requires_grad=False; got {c2.requires_grad}. Did you capture the flag at wrap-time?'\n"
        ")\n"
        "assert c2.recipe is None, (\n"
        "    f'with grad_tracking_enabled=False, no Recipe should be built; got {c2.recipe}'\n"
        ")\n"
        "# Forward value should still be correct — global toggle does not change MATH.\n"
        "assert t.allclose(c2.array, a.array + b.array), 'forward result must still be a+b'\n"
        "\n"
        "# === FRESH READ: flip global back on; SAME wrapped `add` resumes normal behaviour ===\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "c3 = add(a, b)\n"
        "assert c3.requires_grad is True, (\n"
        "    'after toggling global back on, requires_grad must return to True — '\n"
        "    'the flag is read FRESH each call, not closure-captured at wrap-time'\n"
        ")\n"
        "assert c3.recipe is not None\n"
        "\n"
        "# --- non-tracked inputs always produce non-tracked output, regardless of flags ---\n"
        "a_const = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=False)\n"
        "b_const = MiniTensor(t.tensor([4.0, 5.0, 6.0]), requires_grad=False)\n"
        "c4 = add(a_const, b_const)\n"
        "assert c4.requires_grad is False, 'no tracked input -> no tracked output'\n"
        "assert c4.recipe is None\n"
        "\n"
        "# --- all-False matrix: every (global, is_diff, tracked) combo with at least one False -> requires_grad=False ---\n"
        "combos_false = [\n"
        "    (False, True,  True),   # global off\n"
        "    (True,  False, True),   # per-op off\n"
        "    (True,  True,  False),  # no tracked input\n"
        "    (False, False, True),\n"
        "    (False, True,  False),\n"
        "    (True,  False, False),\n"
        "    (False, False, False),\n"
        "]\n"
        "for (g, isd, tr) in combos_false:\n"
        "    globals()['grad_tracking_enabled'] = g\n"
        "    op = wrap_forward_fn(t.add, is_differentiable=isd)\n"
        "    x = MiniTensor(t.tensor([1.0]), requires_grad=tr)\n"
        "    y = MiniTensor(t.tensor([2.0]), requires_grad=tr)\n"
        "    out = op(x, y)\n"
        "    assert out.requires_grad is False, (\n"
        "        f'combo (global={g}, is_diff={isd}, tracked={tr}) must give requires_grad=False, got True'\n"
        "    )\n"
        "    assert out.recipe is None, (\n"
        "        f'combo (global={g}, is_diff={isd}, tracked={tr}) must give recipe=None, got {out.recipe}'\n"
        "    )\n"
        "\n"
        "# --- only all-True triple produces requires_grad=True ---\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "op = wrap_forward_fn(t.add, is_differentiable=True)\n"
        "x = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "out = op(x, y)\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "\n"
        "# --- leave global on for the rest of the notebook (good hygiene) ---\n"
        "globals()['grad_tracking_enabled'] = True"
    ),
    "solution_body": (
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        # Unbox MiniTensor inputs.\n"
        "        raw_args = tuple(a.array if isinstance(a, MiniTensor) else a for a in args)\n"
        "        out_arr = fwd_fn(*raw_args, **kwargs)\n"
        "        # Three-gate AND. Critical: read grad_tracking_enabled FRESH at call time\n"
        "        # (via globals()), not at wrap-time.\n"
        "        global_on = globals().get('grad_tracking_enabled', True)\n"
        "        any_tracked = any(isinstance(a, MiniTensor) and a.requires_grad for a in args)\n"
        "        requires_grad = bool(global_on and is_differentiable and any_tracked)\n"
        "        out = MiniTensor(out_arr, requires_grad=requires_grad)\n"
        "        if requires_grad:\n"
        "            parents = {i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
        "            out.recipe = Recipe(func=fwd_fn, args=raw_args, kwargs=kwargs, parents=parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    "solution_notes": (
        "**The closure trap.** If `wrap_forward_fn` captured "
        "`grad_tracking_enabled` at wrap-time (e.g. "
        "`g = grad_tracking_enabled` outside the inner fn), it would freeze "
        "to whatever value the flag had when wrapping happened. A user "
        "calling `wrap_forward_fn(t.add)` at module import (flag=True) "
        "and later doing `with no_grad(): ...` (flag flipped to False) "
        "would still see grad-tracked outputs. Fresh read fixes this.\n\n"
        "**Why not capture `is_differentiable` fresh too?** "
        "`is_differentiable` IS a per-op closure flag — it's part of the "
        "wrapped function's identity. `add` is differentiable; `eq` is "
        "not. That's a wrapping-time fact, fixed by the time the function "
        "exists. Only the GLOBAL runtime flag needs the fresh-read treatment.\n\n"
        "**`global_on AND ...` short-circuits at the first False.** Python's "
        "`and` evaluates left-to-right and stops at the first falsy value. "
        "Putting `global_on` first makes the common no-grad case (e.g. "
        "inside `with t.no_grad():`) skip the `any_tracked` scan entirely — "
        "a tiny optimisation that matches PyTorch's actual implementation."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 4 — log-back ex3 (domain-edge blowup, matches torch.autograd)
# ---------------------------------------------------------------------------

SPEC_LOG_BACK = {
    "atom_id": "log-back",
    "subtopic": "Backprop: log_back",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_LOG_BACK_DOMAIN_EDGE,
    "exercise_index": 3,
    "exercise_title": "log_back at the domain edge — matches torch.autograd",
    "slug": "log-back-domain-edge-matches-torch-autograd",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["log-back", "domain", "blowup", "torch-autograd", "comparison"],
    "kcs": [
        "log-back-no-eps-no-clamp",
        "matches-torch-autograd-at-edge",
    ],
    "lo": (
        "Analyze log_back's behaviour at small-x and zero-x inputs by computing "
        "it directly and comparing bit-for-bit against torch.autograd, confirming "
        "neither implementation adds an eps or clamps — both reproduce the 1/x "
        "blow-up faithfully."
    ),
    "prompt_body": (
        "Implement `log_back(grad_out, out, x)` AND a comparison helper "
        "`ex3_compare_with_torch(x_values, grad_out_values)` that returns "
        "the side-by-side audit of hand-rolled `log_back` against "
        "`torch.autograd`.\n\n"
        "Function 1: `log_back(grad_out, out, x)` — same as ex1: "
        "`return grad_out / x`. The `out` argument is unused (we keep it in "
        "the signature for dispatcher compatibility).\n\n"
        "Function 2: `ex3_compare_with_torch(x_values, grad_out_values)` "
        "where both inputs are 1-D `torch.Tensor` of the same length. For "
        "EACH index `i`:\n\n"
        "1. Compute `hand = log_back(grad_out_values[i:i+1], None, "
        "x_values[i:i+1])` (single-element slice).\n"
        "2. Compute the torch.autograd answer by building `x_i = "
        "x_values[i:i+1].clone().requires_grad_(True)`, then "
        "`loss = (t.log(x_i) * grad_out_values[i:i+1]).sum()`, then "
        "`loss.backward()`, and reading `x_i.grad`.\n"
        "3. Append `(hand.item(), torch_answer.item())` to a result list.\n\n"
        "Return the list of `(hand_value, torch_value)` pairs, one per "
        "input element.\n\n"
        "Constraints:\n"
        "- Don't add an `eps`. Don't `.clamp_min`. Don't `torch.where`.\n"
        "- The test will pass values like `1e-30`, `0.0` (the test handles "
        "the inf case carefully), and large values, and expects your "
        "answers to MATCH `torch.autograd` exactly (or in the inf case, "
        "both to be `inf`)."
    ),
    "stub": (
        "def log_back(grad_out, out, x):\n"
        '    """Elementwise backward for out = log(x). Returns grad_out / x."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def ex3_compare_with_torch(x_values, grad_out_values):\n"
        '    """Return list of (hand, torch) gradient pairs for each input."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- log_back smoke test (tame values) ---\n"
        "x = t.tensor([1.0, 2.0, 4.0, 8.0])\n"
        "g = log_back(t.ones(4), None, x)\n"
        "assert t.allclose(g, 1.0 / x, atol=1e-7), f'log_back basic wrong: {g}'\n"
        "\n"
        "# --- comparison helper: tame inputs all agree to ~machine precision ---\n"
        "x_tame = t.tensor([1.0, 2.0, 10.0, 100.0])\n"
        "go_tame = t.tensor([1.0, 1.0, 1.0, 1.0])\n"
        "pairs = ex3_compare_with_torch(x_tame, go_tame)\n"
        "assert len(pairs) == 4, f'one pair per input; got {len(pairs)}'\n"
        "for i, (hand, torch_ans) in enumerate(pairs):\n"
        "    assert abs(hand - torch_ans) < 1e-6, (\n"
        "        f'index {i}: hand={hand} torch={torch_ans} (tame range)'\n"
        "    )\n"
        "\n"
        "# --- domain edge: very small x. hand-rolled blows up; torch blows up the same way ---\n"
        "x_small = t.tensor([1e-3, 1e-6, 1e-10, 1e-20])\n"
        "go_small = t.tensor([1.0, 1.0, 1.0, 1.0])\n"
        "pairs = ex3_compare_with_torch(x_small, go_small)\n"
        "for i, (hand, torch_ans) in enumerate(pairs):\n"
        "    # Compare in relative terms — values may be huge.\n"
        "    expected = 1.0 / x_small[i].item()\n"
        "    rel = abs(hand - torch_ans) / max(abs(torch_ans), 1e-30)\n"
        "    assert rel < 1e-5, (\n"
        "        f'small-x mismatch at i={i}: hand={hand} torch={torch_ans} expected~{expected}'\n"
        "    )\n"
        "    # And both should match the 1/x ground truth.\n"
        "    rel_hand = abs(hand - expected) / max(abs(expected), 1e-30)\n"
        "    assert rel_hand < 1e-5, f'hand-rolled missed 1/x at i={i}: {hand} vs {expected}'\n"
        "\n"
        "# --- huge x: small grad ---\n"
        "x_big = t.tensor([1e10, 1e15])\n"
        "go_big = t.tensor([1.0, 1.0])\n"
        "pairs = ex3_compare_with_torch(x_big, go_big)\n"
        "for i, (hand, torch_ans) in enumerate(pairs):\n"
        "    assert abs(hand - torch_ans) < 1e-25 or (hand == 0.0 and torch_ans == 0.0), (\n"
        "        f'big-x mismatch at i={i}: hand={hand} torch={torch_ans}'\n"
        "    )\n"
        "\n"
        "# --- non-unit grad_out: scales the answer linearly ---\n"
        "x_mix = t.tensor([2.0, 4.0])\n"
        "go_mix = t.tensor([7.0, -3.0])\n"
        "pairs = ex3_compare_with_torch(x_mix, go_mix)\n"
        "for i, (hand, torch_ans) in enumerate(pairs):\n"
        "    expected = go_mix[i].item() / x_mix[i].item()\n"
        "    assert abs(hand - expected) < 1e-6, f'hand wrong: {hand} vs {expected}'\n"
        "    assert abs(torch_ans - expected) < 1e-6\n"
        "\n"
        "# --- x=0: both implementations produce inf (matching). We check sign for grad_out=1.0. ---\n"
        "x_zero = t.tensor([0.0])\n"
        "go_zero = t.tensor([1.0])\n"
        "pairs = ex3_compare_with_torch(x_zero, go_zero)\n"
        "hand, torch_ans = pairs[0]\n"
        "import math\n"
        "# Both should be +inf (1/0 with positive numerator).\n"
        "assert math.isinf(hand), f'hand at x=0 should be inf; got {hand}'\n"
        "assert math.isinf(torch_ans), f'torch at x=0 should be inf; got {torch_ans}'\n"
        "assert (hand > 0) == (torch_ans > 0), f'sign should agree; hand={hand} torch={torch_ans}'\n"
        "\n"
        "# --- shape: helper returns Python list, each pair Python (float, float) ---\n"
        "pairs = ex3_compare_with_torch(t.tensor([1.0, 2.0]), t.tensor([1.0, 1.0]))\n"
        "assert isinstance(pairs, list)\n"
        "assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)\n"
        "assert all(isinstance(v, float) for p in pairs for v in p)"
    ),
    "solution_body": (
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "\n"
        "\n"
        "def ex3_compare_with_torch(x_values, grad_out_values):\n"
        "    pairs = []\n"
        "    for i in range(x_values.numel()):\n"
        "        x_slice = x_values[i:i+1]\n"
        "        go_slice = grad_out_values[i:i+1]\n"
        "        hand = log_back(go_slice, None, x_slice).item()\n"
        "        x_var = x_slice.clone().detach().requires_grad_(True)\n"
        "        loss = (t.log(x_var) * go_slice).sum()\n"
        "        loss.backward()\n"
        "        torch_ans = x_var.grad.item()\n"
        "        pairs.append((hand, torch_ans))\n"
        "    return pairs"
    ),
    "solution_notes": (
        "**No eps, no clamp.** Real `torch.autograd` for `log` does NOT add "
        "a stability term. If you want eps, you add it to the FORWARD "
        "(`log(x + eps)`), which changes the derivative to `1/(x+eps)` "
        "automatically via the chain rule. Doing it inside `log_back` "
        "would silently disagree with the forward — autograd's first "
        "correctness rule is that backward matches forward's analytic "
        "derivative.\n\n"
        "**Why `.item()` for comparison.** The pairs are pairs of Python "
        "floats. At inf, two tensors comparing via `==` work but lose the "
        "is-it-actually-inf signal. `math.isinf` is the precise check.\n\n"
        "**Per-element loop, not vectorised.** We unroll because "
        "`torch.autograd` accumulates `.grad` into the leaf, so running it "
        "vectorised would mix gradients across positions. The slice + "
        "`requires_grad_(True)` per element keeps each backward isolated. "
        "Slower but correct; this is a comparison drill, not a perf drill."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 5 — max-back-tied-half ex3 (mass conservation invariant)
# ---------------------------------------------------------------------------

SPEC_MAX_BACK = {
    "atom_id": "max-back-tied-half",
    "subtopic": "Backprop: max_back with tied half-mass",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MAX_BACK_MASS_CONSERVATION,
    "exercise_index": 3,
    "exercise_title": "verify per-position mass conservation across all tie patterns",
    "slug": "verify-per-position-mass-conservation-across-ties",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["mass-conservation", "tie", "invariant", "back0", "back1"],
    "kcs": [
        "mass-conservation-invariant",
        "half-mass-symmetric-split",
    ],
    "lo": (
        "Analyze the mass-conservation invariant of half-mass tie-splitting: "
        "for every position i, maximum_back0[i] + maximum_back1[i] == "
        "grad_out[i], holds across x>y, x<y, and x==y cases."
    ),
    "prompt_body": (
        "Implement `maximum_back0(grad_out, out, x, y)`, "
        "`maximum_back1(grad_out, out, x, y)`, and a verifier "
        "`ex3_check_mass_conservation(x, y, grad_out)` that returns a dict "
        "summarising the invariant on a per-position basis.\n\n"
        "Function 1+2 — the back-fns (no unbroadcast; assume `x.shape == "
        "y.shape == grad_out.shape`):\n"
        "- `maximum_back0(grad_out, out, x, y) = grad_out * ((x > y) + "
        "0.5 * (x == y))`.\n"
        "- `maximum_back1(grad_out, out, x, y) = grad_out * ((y > x) + "
        "0.5 * (x == y))`. Note the symmetric inequality flipped to `y > x`.\n\n"
        "Function 3 — `ex3_check_mass_conservation(x, y, grad_out)`. Returns "
        "a dict with these keys:\n\n"
        "- `'g0'`: the tensor `maximum_back0(grad_out, _, x, y)`.\n"
        "- `'g1'`: the tensor `maximum_back1(grad_out, _, x, y)`.\n"
        "- `'sum'`: the elementwise sum `g0 + g1`.\n"
        "- `'conserves_mass'`: bool — True iff `t.allclose(g0 + g1, "
        "grad_out, atol=1e-6)`.\n"
        "- `'n_ties'`: int — number of positions where `x == y`.\n"
        "- `'n_x_wins'`: int — positions where `x > y`.\n"
        "- `'n_y_wins'`: int — positions where `x < y`.\n\n"
        "The verifier must work on tensors of any shape; the position counts "
        "are over `.numel()`."
    ),
    "stub": (
        "def maximum_back0(grad_out, out, x, y):\n"
        '    """grad_x for out = maximum(x, y). Half-mass tie convention."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def maximum_back1(grad_out, out, x, y):\n"
        '    """grad_y for out = maximum(x, y). Half-mass tie convention."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def ex3_check_mass_conservation(x, y, grad_out) -> dict:\n"
        '    """Audit dict of per-position back0/back1 + sum + invariant flag."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- back-fn smoke: strict inequality cases produce the textbook answer ---\n"
        "x = t.tensor([3.0, 1.0])     # 3 > 2 and 1 < 5\n"
        "y = t.tensor([2.0, 5.0])\n"
        "g = t.tensor([1.0, 1.0])\n"
        "g0 = maximum_back0(g, None, x, y)\n"
        "g1 = maximum_back1(g, None, x, y)\n"
        "assert t.allclose(g0, t.tensor([1.0, 0.0])), f'g0 wrong: {g0}'\n"
        "assert t.allclose(g1, t.tensor([0.0, 1.0])), f'g1 wrong: {g1}'\n"
        "assert t.allclose(g0 + g1, g), 'mass conservation broken even in strict case'\n"
        "\n"
        "# --- tie position: half-mass each ---\n"
        "x = t.tensor([2.0, 2.0])\n"
        "y = t.tensor([2.0, 2.0])\n"
        "g = t.tensor([4.0, 6.0])\n"
        "g0 = maximum_back0(g, None, x, y)\n"
        "g1 = maximum_back1(g, None, x, y)\n"
        "assert t.allclose(g0, t.tensor([2.0, 3.0])), f'tie g0 wrong: {g0}'\n"
        "assert t.allclose(g1, t.tensor([2.0, 3.0])), f'tie g1 wrong: {g1}'\n"
        "assert t.allclose(g0 + g1, g), 'tie mass conservation broken'\n"
        "\n"
        "# --- verifier on a mixed vector ---\n"
        "x = t.tensor([5.0, 1.0, 3.0, 7.0])    # x_wins, y_wins, tie, x_wins\n"
        "y = t.tensor([2.0, 4.0, 3.0, 6.0])\n"
        "g = t.tensor([10.0, 20.0, 30.0, 40.0])\n"
        "report = ex3_check_mass_conservation(x, y, g)\n"
        "assert isinstance(report, dict)\n"
        "for k in ('g0', 'g1', 'sum', 'conserves_mass', 'n_ties', 'n_x_wins', 'n_y_wins'):\n"
        "    assert k in report, f'missing key {k}: {report}'\n"
        "assert isinstance(report['g0'], t.Tensor)\n"
        "assert report['n_ties'] == 1, f'n_ties wrong: {report}'\n"
        "assert report['n_x_wins'] == 2, f'n_x_wins wrong: {report}'\n"
        "assert report['n_y_wins'] == 1, f'n_y_wins wrong: {report}'\n"
        "assert report['conserves_mass'] is True\n"
        "# Spot-check: g0[2]=15 (half of 30), g1[2]=15. g0[0]=10 (x wins), g1[0]=0.\n"
        "assert t.allclose(report['g0'], t.tensor([10.0, 0.0, 15.0, 40.0])), f'g0: {report[\"g0\"]}'\n"
        "assert t.allclose(report['g1'], t.tensor([0.0, 20.0, 15.0, 0.0])), f'g1: {report[\"g1\"]}'\n"
        "assert t.allclose(report['sum'], g)\n"
        "\n"
        "# --- ALL ties (every position) — sum must still equal grad_out exactly ---\n"
        "x = t.ones(100)\n"
        "y = t.ones(100)\n"
        "g = t.linspace(-50.0, 50.0, 100)\n"
        "report = ex3_check_mass_conservation(x, y, g)\n"
        "assert report['n_ties'] == 100\n"
        "assert report['n_x_wins'] == 0 and report['n_y_wins'] == 0\n"
        "assert report['conserves_mass'] is True\n"
        "assert t.allclose(report['g0'], g * 0.5)\n"
        "assert t.allclose(report['g1'], g * 0.5)\n"
        "\n"
        "# --- ZERO ties (random distinct values) — sum still equals grad_out exactly ---\n"
        "t.manual_seed(0)\n"
        "x = t.randn(50)\n"
        "y = x + 1.0    # guaranteed y > x at every position\n"
        "g = t.randn(50)\n"
        "report = ex3_check_mass_conservation(x, y, g)\n"
        "assert report['n_ties'] == 0\n"
        "assert report['n_x_wins'] == 0\n"
        "assert report['n_y_wins'] == 50\n"
        "assert report['conserves_mass'] is True\n"
        "# g0 should be all zeros, g1 all grad_out.\n"
        "assert t.allclose(report['g0'], t.zeros_like(g))\n"
        "assert t.allclose(report['g1'], g)\n"
        "\n"
        "# --- multi-dim shape ---\n"
        "x = t.tensor([[1.0, 2.0], [3.0, 3.0]])\n"
        "y = t.tensor([[2.0, 1.0], [3.0, 1.0]])\n"
        "g = t.tensor([[1.0, 1.0], [1.0, 1.0]])\n"
        "report = ex3_check_mass_conservation(x, y, g)\n"
        "assert report['g0'].shape == x.shape\n"
        "assert report['n_ties'] == 1, f'(1,1) is tied with (3,3); expected 1, got {report[\"n_ties\"]}'\n"
        "assert report['conserves_mass'] is True"
    ),
    "solution_body": (
        "def maximum_back0(grad_out, out, x, y):\n"
        "    bool_sum_x = (x > y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype)\n"
        "    return grad_out * bool_sum_x\n"
        "\n"
        "\n"
        "def maximum_back1(grad_out, out, x, y):\n"
        "    bool_sum_y = (y > x).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype)\n"
        "    return grad_out * bool_sum_y\n"
        "\n"
        "\n"
        "def ex3_check_mass_conservation(x, y, grad_out):\n"
        "    g0 = maximum_back0(grad_out, None, x, y)\n"
        "    g1 = maximum_back1(grad_out, None, x, y)\n"
        "    s = g0 + g1\n"
        "    return {\n"
        "        'g0': g0,\n"
        "        'g1': g1,\n"
        "        'sum': s,\n"
        "        'conserves_mass': bool(t.allclose(s, grad_out, atol=1e-6)),\n"
        "        'n_ties': int((x == y).sum().item()),\n"
        "        'n_x_wins': int((x > y).sum().item()),\n"
        "        'n_y_wins': int((y > x).sum().item()),\n"
        "    }"
    ),
    "solution_notes": (
        "**Mass conservation = correctness oracle.** Any back-fn that splits "
        "one upstream gradient across multiple parents must preserve total "
        "mass. If the sum exceeds `grad_out`, the loss surface is "
        "double-counted; if it falls short, gradient flow is lossy. The "
        "half-mass rule is the UNIQUE symmetric split that conserves mass "
        "at ties.\n\n"
        "**Three exhaustive cases.** At any position one of: `x > y`, "
        "`x < y`, `x == y` holds. Verify the three cases produce sum=1 "
        "(times `grad_out`): (1+0)=1, (0+1)=1, (0.5+0.5)=1. There's no "
        "fourth case to worry about.\n\n"
        "**Cast booleans before multiplication.** `(x > y)` is a "
        "`torch.bool` tensor; multiplying it directly works (auto-coerces) "
        "but adding `0.5 * (x == y)` to it requires explicit `.to(...)`. "
        "Cast to `grad_out.dtype` keeps the result dtype-correct and "
        "avoids a sneaky upcast to float64 when `grad_out` is float32."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 6 — multiply-back ex3 (3-D multi-axis unbroadcast)
# ---------------------------------------------------------------------------

SPEC_MULTIPLY_BACK = {
    "atom_id": "multiply-back",
    "subtopic": "Backprop: multiply_back",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MULTIPLY_BACK_3D_BROADCAST,
    "exercise_index": 3,
    "exercise_title": "multiply_back across 3-D multi-axis broadcasting",
    "slug": "multiply-back-three-d-multi-axis-broadcasting",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["multiply-back", "3d", "broadcast", "multi-axis", "unbroadcast"],
    "kcs": [
        "unbroadcast-multi-axis-reduce",
        "matches-torch-autograd-3d",
    ],
    "lo": (
        "Analyze multiply_back's behaviour under 3-D broadcasting where each "
        "parent has a different singleton-axis pattern, and verify the result "
        "matches torch.autograd to within 1e-6 on (5,1,4) * (1,3,1)."
    ),
    "prompt_body": (
        "Implement `multiply_back0(grad_out, out, x, y)` and "
        "`multiply_back1(grad_out, out, x, y)` for the forward op "
        "`out = x * y` where `x` and `y` are TENSORS with DIFFERENT "
        "broadcasting axes (different singleton dims).\n\n"
        "Provided in the stub: `unbroadcast(grad, original)`.\n\n"
        "Behaviour:\n"
        "1. `multiply_back0(grad_out, out, x, y) = unbroadcast(grad_out * "
        "y, x)`. The `unbroadcast` reduces axes where `x` was a singleton "
        "but the broadcast result was bigger.\n"
        "2. `multiply_back1(grad_out, out, x, y) = unbroadcast(grad_out * "
        "x, y)`. Symmetric.\n\n"
        "The test runs the canonical 3-D case `x=(5,1,4)`, `y=(1,3,1)`, "
        "`out=(5,3,4)`, and compares your back-fn outputs against "
        "`torch.autograd` on the same computation. They must agree to "
        "within 1e-6 on every element."
    ),
    "stub": (
        "def unbroadcast(grad, original):\n"
        "    # Provided helper — sums out axes broadcasting added/expanded.\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        '    """dL/dx for out = x*y, with multi-axis unbroadcast to x.shape."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        '    """dL/dy for out = x*y, with multi-axis unbroadcast to y.shape."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === THE HEADLINE: 3-D mixed-singleton broadcast ===\n"
        "t.manual_seed(0)\n"
        "x = t.randn(5, 1, 4)            # singleton at axis 1\n"
        "y = t.randn(1, 3, 1)            # singletons at axes 0 AND 2\n"
        "out = x * y                       # shape (5, 3, 4) — both broadcast\n"
        "grad_out = t.randn(5, 3, 4)\n"
        "\n"
        "g0 = multiply_back0(grad_out, out, x, y)\n"
        "g1 = multiply_back1(grad_out, out, x, y)\n"
        "assert g0.shape == x.shape, f'g0 shape: got {g0.shape}, expected {x.shape}'\n"
        "assert g1.shape == y.shape, f'g1 shape: got {g1.shape}, expected {y.shape}'\n"
        "\n"
        "# --- compare with torch.autograd ---\n"
        "x_at = x.clone().requires_grad_(True)\n"
        "y_at = y.clone().requires_grad_(True)\n"
        "loss = (x_at * y_at * grad_out).sum()\n"
        "loss.backward()\n"
        "assert t.allclose(g0, x_at.grad, atol=1e-6), (\n"
        "    f'multiply_back0 mismatch with autograd. max |diff| = {(g0 - x_at.grad).abs().max()}'\n"
        ")\n"
        "assert t.allclose(g1, y_at.grad, atol=1e-6), (\n"
        "    f'multiply_back1 mismatch with autograd. max |diff| = {(g1 - y_at.grad).abs().max()}'\n"
        ")\n"
        "\n"
        "# --- expected formula in closed form: g0 sums axis 1 (size 3) of (grad_out * y) ---\n"
        "expected_g0 = (grad_out * y).sum(dim=1, keepdim=True)\n"
        "assert t.allclose(g0, expected_g0, atol=1e-6), 'g0 != sum over broadcast axis'\n"
        "# g1 sums axes 0 (size 5) AND 2 (size 4) of (grad_out * x).\n"
        "expected_g1 = (grad_out * x).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)\n"
        "assert t.allclose(g1, expected_g1, atol=1e-6), 'g1 != sum over two broadcast axes'\n"
        "\n"
        "# --- 4-D case: (1,1,3,4) * (2,5,1,1) -> out (2,5,3,4); each side has TWO singleton axes ---\n"
        "x4 = t.randn(1, 1, 3, 4)\n"
        "y4 = t.randn(2, 5, 1, 1)\n"
        "out4 = x4 * y4\n"
        "go4 = t.randn(2, 5, 3, 4)\n"
        "g0_4 = multiply_back0(go4, out4, x4, y4)\n"
        "g1_4 = multiply_back1(go4, out4, x4, y4)\n"
        "assert g0_4.shape == x4.shape, f'4D g0 shape: {g0_4.shape}'\n"
        "assert g1_4.shape == y4.shape, f'4D g1 shape: {g1_4.shape}'\n"
        "x_at4 = x4.clone().requires_grad_(True)\n"
        "y_at4 = y4.clone().requires_grad_(True)\n"
        "(x_at4 * y_at4 * go4).sum().backward()\n"
        "assert t.allclose(g0_4, x_at4.grad, atol=1e-6)\n"
        "assert t.allclose(g1_4, y_at4.grad, atol=1e-6)\n"
        "\n"
        "# --- degenerate case: same-shape (no broadcast) — unbroadcast is identity ---\n"
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y = t.tensor([5.0, 6.0, 7.0])\n"
        "out = x * y\n"
        "g = t.ones(3)\n"
        "g0 = multiply_back0(g, out, x, y)\n"
        "g1 = multiply_back1(g, out, x, y)\n"
        "assert t.allclose(g0, y) and t.allclose(g1, x), 'same-shape case broke'\n"
        "assert g0.shape == x.shape and g1.shape == y.shape\n"
        "\n"
        "# --- adversarial: rank-mismatch broadcast (x has fewer dims than y) ---\n"
        "x_lr = t.tensor([2.0, 3.0])     # shape (2,)\n"
        "y_lr = t.tensor([[1.0, 1.0],    # shape (3, 2)\n"
        "                 [2.0, 2.0],\n"
        "                 [3.0, 3.0]])\n"
        "out_lr = x_lr * y_lr   # (3, 2) via implicit leading 1\n"
        "go_lr = t.ones(3, 2)\n"
        "g0_lr = multiply_back0(go_lr, out_lr, x_lr, y_lr)\n"
        "g1_lr = multiply_back1(go_lr, out_lr, x_lr, y_lr)\n"
        "assert g0_lr.shape == x_lr.shape, f'lower-rank g0 shape: {g0_lr.shape}'\n"
        "assert g1_lr.shape == y_lr.shape, f'lower-rank g1 shape: {g1_lr.shape}'\n"
        "x_at = x_lr.clone().requires_grad_(True)\n"
        "y_at = y_lr.clone().requires_grad_(True)\n"
        "(x_at * y_at * go_lr).sum().backward()\n"
        "assert t.allclose(g0_lr, x_at.grad, atol=1e-6)\n"
        "assert t.allclose(g1_lr, y_at.grad, atol=1e-6)"
    ),
    "solution_body": (
        "def multiply_back0(grad_out, out, x, y):\n"
        "    return unbroadcast(grad_out * y, x)\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    return unbroadcast(grad_out * x, y)"
    ),
    "solution_notes": (
        "**The unbroadcast helper is doing all the work.** Both back-fns "
        "are one-liners; the multi-axis reduction is the part that gets "
        "hard. The helper's inner loop is what handles `(1,3,1)` correctly: "
        "the rank-equal check is satisfied at the start (both grad and "
        "original are 3-D after the multiplication), then the per-axis "
        "loop sums axes 0 and 2 with `keepdim=True`.\n\n"
        "**Why `keepdim=True`.** Without it, summing axis 0 of `(5,3,4)` "
        "would give `(3,4)` and the next axis index would refer to a "
        "different axis. `keepdim=True` keeps the rank constant so the "
        "loop indices stay valid.\n\n"
        "**The rank-mismatch test exercises the OUTER `while` loop.** "
        "When `x_lr` has shape `(2,)` (rank 1) but `grad_out * y_lr` has "
        "shape `(3,2)` (rank 2), the `while grad.ndim > original.ndim` "
        "guard fires once and sums out the leading axis. The inner loop "
        "then finds no singleton-to-broadcast axes in `(2,)` and exits. "
        "Two-stage reduction — both stages must work."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 7 — non-diff-fn-wrap ex3 (chained non-diff)
# ---------------------------------------------------------------------------

SPEC_NON_DIFF = {
    "atom_id": "non-diff-fn-wrap",
    "subtopic": "Backprop: non-differentiable fn wrap",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_NON_DIFF_CHAIN,
    "exercise_index": 3,
    "exercise_title": "chained non-diff ops — argmax(eq(a,b)) stays isolated",
    "slug": "chained-non-diff-argmax-of-eq-stays-isolated",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["non-diff", "chain", "argmax", "eq", "graph-isolation"],
    "kcs": [
        "non-diff-chain-isolation",
        "recipe-none-cascades",
    ],
    "lo": (
        "Analyze the cascade behaviour of chained non-differentiable ops: "
        "argmax(eq(a, b)) produces a MiniTensor whose recipe=None, "
        "requires_grad=False, and is unreachable from any other graph walk "
        "started from a different end-node."
    ),
    "prompt_body": (
        "Implement `wrap_forward_fn(fwd_fn, is_differentiable=True)` (same "
        "contract as ex1/ex2: three-gate AND, conditional Recipe). Then "
        "use it to wrap TWO non-diff ops — `eq` and `argmax` — and build a "
        "compute graph that chains them.\n\n"
        "Provided in the stub: `sorted_computational_graph(tensor)` walks "
        "the recipe-parents graph and stops at any `recipe=None` node.\n\n"
        "Test inputs build (verbatim — your `wrap_forward_fn` must produce "
        "these structurally):\n"
        "- `a`, `b`: tracked MiniTensor leaves.\n"
        "- `mask = eq_wrap(a, b)` — first non-diff op. Recipe must be None.\n"
        "- `idx  = argmax_wrap(mask)` — second non-diff op, OPERATES ON a "
        "non-tracked input. Recipe must also be None.\n"
        "- `c = add_wrap(a, b)`, `d = mul_wrap(c, c)` — pure-diff side "
        "branch as a control.\n\n"
        "The test verifies:\n"
        "1. Both `mask.recipe is None` and `idx.recipe is None`.\n"
        "2. Walking from `idx`: only `[idx]` is found (length 1 — chain stays "
        "isolated). `mask`, `a`, `b` are NOT in the walk.\n"
        "3. Walking from `mask`: only `[mask]` (length 1).\n"
        "4. Walking from `d`: includes `[d, c, a, b]` (the diff side); "
        "EXCLUDES `mask` and `idx`.\n"
        "5. `mask.requires_grad is False` and `idx.requires_grad is False`."
    ),
    "stub": (
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        '    """Three-gate wrapper. Recipe only when requires_grad=True."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def sorted_computational_graph(tensor):\n"
        '    """Provided: reverse-topo sort, stops at recipe=None."""\n'
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
    "test_body": (
        "globals()['grad_tracking_enabled'] = True\n"
        "add_wrap = wrap_forward_fn(t.add)\n"
        "mul_wrap = wrap_forward_fn(t.multiply)\n"
        "eq_wrap = wrap_forward_fn(t.eq, is_differentiable=False)\n"
        "# argmax: torch op that returns a long tensor. is_differentiable=False.\n"
        "# Note: argmax doesn't accept bool inputs in modern torch, so we cast first.\n"
        "argmax_wrap = wrap_forward_fn(lambda z: t.argmax(z.float()), is_differentiable=False)\n"
        "\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 3.0, 5.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([1.0, 0.0, 3.0, 4.0]), requires_grad=True)\n"
        "\n"
        "# --- non-diff branch 1: mask = eq(a, b) ---\n"
        "mask = eq_wrap(a, b)\n"
        "assert mask.recipe is None, 'mask: first non-diff must have recipe=None'\n"
        "assert mask.requires_grad is False\n"
        "# Forward value should be correct (boolean tensor of equality).\n"
        "assert (mask.array == (a.array == b.array)).all()\n"
        "\n"
        "# --- non-diff branch 2: idx = argmax(mask). mask is non-tracked. ---\n"
        "idx = argmax_wrap(mask)\n"
        "assert idx.recipe is None, 'idx: second non-diff (chained) must also have recipe=None'\n"
        "assert idx.requires_grad is False\n"
        "# Forward value: argmax of [T, F, T, F] cast to bool int = first True position.\n"
        "# In torch, argmax over bool returns the first max; for [1,0,1,0] -> 0.\n"
        "assert isinstance(idx.array, t.Tensor)\n"
        "\n"
        "# --- diff side branch (control): c = add(a, b), d = mul(c, c) ---\n"
        "c = add_wrap(a, b)\n"
        "d = mul_wrap(c, c)\n"
        "assert c.recipe is not None and c.recipe.func is t.add\n"
        "assert d.recipe is not None and d.recipe.func is t.multiply\n"
        "assert c.requires_grad is True and d.requires_grad is True\n"
        "\n"
        "# === walking from idx: only [idx] reachable (chain isolated) ===\n"
        "idx_graph = sorted_computational_graph(idx)\n"
        "idx_ids = {id(n) for n in idx_graph}\n"
        "assert idx_ids == {id(idx)}, (\n"
        "    f'walk from chained non-diff output should yield only itself; got {idx_ids} '\n"
        "    f'(expected just {{id(idx)}})'\n"
        ")\n"
        "assert id(mask) not in idx_ids, 'mask must not be reachable from idx (recipe=None breaks chain)'\n"
        "assert id(a) not in idx_ids and id(b) not in idx_ids\n"
        "\n"
        "# === walking from mask: only [mask] reachable ===\n"
        "mask_graph = sorted_computational_graph(mask)\n"
        "mask_ids = {id(n) for n in mask_graph}\n"
        "assert mask_ids == {id(mask)}, f'mask walk wrong: {mask_ids}'\n"
        "\n"
        "# === walking from d (diff side): d, c, a, b reachable; mask + idx NOT ===\n"
        "d_graph = sorted_computational_graph(d)\n"
        "d_ids = {id(n) for n in d_graph}\n"
        "assert id(d) in d_ids and id(c) in d_ids and id(a) in d_ids and id(b) in d_ids, (\n"
        "    f'diff side walk missing nodes; got {d_ids}'\n"
        ")\n"
        "assert id(mask) not in d_ids, 'mask must not appear in d-walk'\n"
        "assert id(idx) not in d_ids, 'idx must not appear in d-walk'\n"
        "\n"
        "# === confirm reverse-topo ordering on d-walk ===\n"
        "# d first, leaves last; parents come AFTER children in the result list.\n"
        "assert d_graph[0] is d, f'd must be first in reverse-topo, got {type(d_graph[0]).__name__}'\n"
        "# c must precede a and b.\n"
        "d_list_ids = [id(n) for n in d_graph]\n"
        "assert d_list_ids.index(id(c)) < d_list_ids.index(id(a))\n"
        "assert d_list_ids.index(id(c)) < d_list_ids.index(id(b))\n"
        "\n"
        "# === structural property: non-diff chain length = 2, but each output is an island ===\n"
        "# Even though idx semantically depends on mask which depends on a,b, the AUTOGRAD GRAPH\n"
        "# has zero edges among them. Counted as 3 disconnected singletons.\n"
        "all_ids = idx_ids | mask_ids | d_ids\n"
        "assert id(idx) in all_ids and id(mask) in all_ids\n"
        "# but idx_ids and mask_ids and d_ids are pairwise disjoint:\n"
        "assert idx_ids.isdisjoint(mask_ids)\n"
        "assert idx_ids.isdisjoint(d_ids)\n"
        "assert mask_ids.isdisjoint(d_ids)"
    ),
    "solution_body": (
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = tuple(a.array if isinstance(a, MiniTensor) else a for a in args)\n"
        "        out_arr = fwd_fn(*raw_args, **kwargs)\n"
        "        global_on = globals().get('grad_tracking_enabled', True)\n"
        "        any_tracked = any(isinstance(a, MiniTensor) and a.requires_grad for a in args)\n"
        "        requires_grad = bool(global_on and is_differentiable and any_tracked)\n"
        "        out = MiniTensor(out_arr, requires_grad=requires_grad)\n"
        "        if requires_grad:\n"
        "            parents = {i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
        "            out.recipe = Recipe(func=fwd_fn, args=raw_args, kwargs=kwargs, parents=parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    "solution_notes": (
        "**Recipe-None cascades.** When `eq_wrap(a, b)` produces "
        "`mask.requires_grad=False`, the subsequent `argmax_wrap(mask)` "
        "call sees `mask` as a non-tracked input. Even if `argmax` had "
        "been `is_differentiable=True`, the three-gate AND would still "
        "give `requires_grad=False` (no tracked input). Chaining preserves "
        "the non-tracking property.\n\n"
        "**Per-non-diff-output is a graph singleton.** From the autograd "
        "graph's point of view, each non-diff output is its own isolated "
        "1-node DAG. The semantic dependency `idx <- mask <- a, b` is "
        "INVISIBLE to the topo walker because there are no recipes "
        "encoding those edges. This is correct: no gradient flows along "
        "those edges, so the reverse pass has no work to do.\n\n"
        "**Why this is a feature, not a limitation.** A user calling "
        "`loss.backward()` on a diff loss `d` that incidentally shares "
        "leaves with a non-diff `idx` doesn't want grad to flow through "
        "`idx`. The recipe=None mechanism makes that the default, with no "
        "conditional logic in the reverse pass."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 8 — sorted-computational-graph ex3 (diamond pattern)
# ---------------------------------------------------------------------------

SPEC_SORTED_GRAPH = {
    "atom_id": "sorted-computational-graph",
    "subtopic": "Backprop: Sorted computation graph",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_SORT_DIAMOND,
    "exercise_index": 3,
    "exercise_title": "sort a diamond DAG with one shared root",
    "slug": "sort-diamond-dag-with-shared-root",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["diamond", "dag", "topo-sort", "shared-root", "deduplication"],
    "kcs": [
        "diamond-graph-ordering",
        "perm-set-prevents-duplicates",
    ],
    "lo": (
        "Apply topological sort over a diamond compute graph "
        "a -> {b, c} -> d to produce a single-visit reverse-topo order "
        "where d is first, b and c both precede a, and unrelated nodes "
        "are not visited."
    ),
    "prompt_body": (
        "Implement `topological_sort(node, get_children)` and "
        "`sorted_computational_graph(tensor)`.\n\n"
        "Spec:\n"
        "- `topological_sort(node, get_children)` does a DFS with a "
        "`perm` set keyed by `id(child)` to avoid revisits. Returns the "
        "post-order (leaves first, end-node last).\n"
        "- `sorted_computational_graph(tensor)` calls "
        "`topological_sort` with `get_children = lambda n: list("
        "n.recipe.parents.values()) if n.recipe else []`, then REVERSES "
        "the result so the end-node is first.\n\n"
        "The test graph is a diamond:\n\n"
        "```\n"
        "    a         (leaf, single root)\n"
        "   / \\\n"
        "  b   c      (b = log(a), c = neg(a))\n"
        "   \\ /\n"
        "    d        (d = b * c, end-node)\n"
        "```\n"
        "\n"
        "Verify:\n"
        "1. Result length is exactly 4 (one entry per node: a, b, c, d).\n"
        "2. `d` is at index 0 (reverse-topo: end-node first).\n"
        "3. `a` is at index 3 (leaf comes last in reverse-topo).\n"
        "4. `b` and `c` both come before `a` and after `d`.\n"
        "5. An unrelated leaf `z` (constructed but not connected to the "
        "graph) does NOT appear in the result.\n"
        "6. The graph also handles a multi-output extension: appending a "
        "second end-node `e = b + c` (sharing b and c with d) and walking "
        "from `e` gives the right order with `e` first."
    ),
    "stub": (
        "def topological_sort(node, get_children):\n"
        '    """DFS-based topo sort; uses perm set keyed by id(node) to dedupe."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def sorted_computational_graph(tensor):\n"
        '    """Reverse-topo sort of the recipe-parents graph; end-node first."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Build the diamond by hand ===\n"
        "a = MiniTensor(t.tensor([2.0, 3.0]), requires_grad=True)\n"
        "b = MiniTensor(t.log(a.array), requires_grad=True)\n"
        "b.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "c = MiniTensor(-a.array, requires_grad=True)\n"
        "c.recipe = Recipe(func=t.neg, args=(a.array,), kwargs={}, parents={0: a})\n"
        "d = MiniTensor(b.array * c.array, requires_grad=True)\n"
        "d.recipe = Recipe(func=t.multiply, args=(b.array, c.array), kwargs={}, parents={0: b, 1: c})\n"
        "# Unrelated leaf — should NOT appear in the walk from d.\n"
        "z = MiniTensor(t.tensor([99.0]), requires_grad=True)\n"
        "\n"
        "# --- walk from d ---\n"
        "result = sorted_computational_graph(d)\n"
        "assert len(result) == 4, f'diamond has 4 unique nodes; got {len(result)}'\n"
        "ids = [id(n) for n in result]\n"
        "assert ids[0] == id(d), f'd must be first (reverse-topo); got node 0 = {type(result[0]).__name__}'\n"
        "assert ids[-1] == id(a), f'a (leaf) must be last in reverse-topo; got node -1 = {type(result[-1]).__name__}'\n"
        "\n"
        "# --- single-visit: each node appears exactly once ---\n"
        "assert len(set(ids)) == 4, f'duplicate visit detected; ids={ids}'\n"
        "\n"
        "# --- b and c both come BEFORE a (closer to d in reverse-topo) ---\n"
        "pos = {n_id: i for i, n_id in enumerate(ids)}\n"
        "assert pos[id(b)] < pos[id(a)], 'b should precede a in reverse-topo'\n"
        "assert pos[id(c)] < pos[id(a)], 'c should precede a in reverse-topo'\n"
        "# Both come AFTER d.\n"
        "assert pos[id(b)] > pos[id(d)]\n"
        "assert pos[id(c)] > pos[id(d)]\n"
        "\n"
        "# --- unrelated leaf z does NOT appear ---\n"
        "assert id(z) not in ids, 'z is disconnected; must not be visited'\n"
        "\n"
        "# === Multi-end extension: add e = b + c (also shares b, c with d) ===\n"
        "e = MiniTensor(b.array + c.array, requires_grad=True)\n"
        "e.recipe = Recipe(func=t.add, args=(b.array, c.array), kwargs={}, parents={0: b, 1: c})\n"
        "\n"
        "result_e = sorted_computational_graph(e)\n"
        "assert len(result_e) == 4, f'walk from e covers e,b,c,a -> 4 nodes; got {len(result_e)}'\n"
        "ids_e = [id(n) for n in result_e]\n"
        "assert ids_e[0] == id(e), 'e must be first in its walk'\n"
        "assert id(d) not in ids_e, 'd should not appear in walk from e'\n"
        "assert id(b) in ids_e and id(c) in ids_e and id(a) in ids_e\n"
        "pos_e = {n_id: i for i, n_id in enumerate(ids_e)}\n"
        "assert pos_e[id(b)] < pos_e[id(a)]\n"
        "assert pos_e[id(c)] < pos_e[id(a)]\n"
        "\n"
        "# === Edge case: walk from a leaf returns just [leaf] ===\n"
        "result_a = sorted_computational_graph(a)\n"
        "assert len(result_a) == 1, f'leaf walk returns just itself; got {len(result_a)}'\n"
        "assert result_a[0] is a\n"
        "\n"
        "# === All edges respected: for every (parent, child) in the diamond, pos[parent] > pos[child] ===\n"
        "edges = [(a, b), (a, c), (b, d), (c, d)]   # (parent_in_data_flow, child_in_data_flow)\n"
        "for (parent, child) in edges:\n"
        "    p, ch = pos[id(parent)], pos[id(child)]\n"
        "    assert p > ch, f'edge parent={parent} child={child} violated: pos[parent]={p} pos[child]={ch}'\n"
        "\n"
        "# === Topological sort low-level: caller-supplied get_children ===\n"
        "g = {1: [2, 3], 2: [4], 3: [4], 4: []}   # diamond on ints\n"
        "raw = topological_sort(1, lambda n: g[n])\n"
        "# raw is post-order: leaves first. Reverse to check end-first.\n"
        "raw_rev = raw[::-1]\n"
        "assert raw_rev[0] == 1, f'topo from 1 should start at 1 when reversed; got {raw_rev}'\n"
        "assert raw_rev[-1] == 4, f'4 is the deepest, should be last; got {raw_rev}'\n"
        "assert set(raw) == {1, 2, 3, 4}\n"
        "assert len(raw) == 4, 'no duplicate visits in diamond'"
    ),
    "solution_body": (
        "def topological_sort(node, get_children):\n"
        "    result = []\n"
        "    perm = set()\n"
        "    def visit(cur):\n"
        "        if id(cur) in perm:\n"
        "            return\n"
        "        perm.add(id(cur))\n"
        "        for child in get_children(cur):\n"
        "            visit(child)\n"
        "        result.append(cur)\n"
        "    visit(node)\n"
        "    return result\n"
        "\n"
        "\n"
        "def sorted_computational_graph(tensor):\n"
        "    def get_children(n):\n"
        "        if n.recipe is None:\n"
        "            return []\n"
        "        return list(n.recipe.parents.values())\n"
        "    return topological_sort(tensor, get_children)[::-1]"
    ),
    "solution_notes": (
        "**The `perm` set keyed by `id` is load-bearing.** Without "
        "deduplication, the diamond's leaf `a` would be visited twice "
        "(once via `b`'s parents, once via `c`'s parents). The post-order "
        "list would then have `a` appearing twice — breaking the "
        "exact-once invariant.\n\n"
        "**Why `id(...)` and not the tensor itself.** MiniTensors aren't "
        "hashable by value (they hold mutable `.array`). Using `id` is "
        "the standard idiom — it's stable for the object's lifetime and "
        "gives O(1) membership.\n\n"
        "**Post-order + reverse is the canonical reverse-topo trick.** "
        "DFS post-order has leaves first; reversing puts the end-node "
        "first. The reverse pass consumes this list head-to-tail, "
        "guaranteeing that by the time it processes a node, all of that "
        "node's CHILDREN (downstream consumers) have already been visited "
        "and their grads computed. That's the invariant that makes "
        "single-pass reverse-mode correct."
    ),
    "extra_imports": [_AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------

SPECS = [
    SPEC_BFL,
    SPEC_END_GRAD,
    SPEC_IS_DIFF,
    SPEC_LOG_BACK,
    SPEC_MAX_BACK,
    SPEC_MULTIPLY_BACK,
    SPEC_NON_DIFF,
    SPEC_SORTED_GRAPH,
]


def _verify_all(specs):
    import torch as t
    import numpy as np
    from torch import Tensor

    passed = 0
    failed = []
    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"
        ns = {
            "t": t,
            "np": np,
            "Tensor": Tensor,
            "_dd_passed": set(),
            "__name__": "__main__",
        }
        t.manual_seed(0)
        np.random.seed(0)
        # Run preamble first (sets up MiniTensor, Recipe).
        try:
            exec(_AUTOGRAD_PREAMBLE, ns)
        except Exception:
            pass
        # Best-effort stub exec (may fail because stub has NotImplementedError).
        try:
            exec(spec["stub"], ns)
        except Exception:
            pass
        try:
            exec(spec["solution_body"], ns)
            exec(spec["test_body"], ns)
        except Exception as e:
            failed.append((tag, repr(e), traceback.format_exc()))
            continue
        passed += 1
        print(f"  [verify] {tag}: ok")
    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, err, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(err)
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[deepening_d_batch14] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_d_batch14] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_d_batch14] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
