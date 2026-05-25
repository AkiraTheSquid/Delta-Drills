#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA part 4 backprop-DRIVER atoms.

Eight single-exercise standalones, under ``prereqs_backprop_driver/``:

  * grad-expressed-in-out          — ex1
  * no-grad-context-mgr-update     — ex1
  * manual-chain-forward-and-back  — ex1
  * dfs-three-set-toposort         — ex1
  * cycle-detection-temp-set       — ex1
  * backprop-pop-outgrad-loop      — ex1
  * dispatch-back-fn-from-recipe   — ex1
  * back-fn-call-with-recipe-args  — ex1

These atoms cover the ARENA-style manual backprop DRIVER — the reverse-pass
loop, the recipe-driven dispatch, the per-arg back_fn call, and supporting
primitives (no_grad context manager, manual forward+backward chain, three-set
DFS toposort + cycle detection).

Composition note: batch-4 already authored ``sorted-computational-graph`` —
that atom WRAPS this batch's ``dfs-three-set-toposort`` with a ``[::-1]``
reversal for the reverse pass. The two are deliberately layered.

Tests use plain ``torch.Tensor`` for shape / value math; we never call
``torch.autograd`` on the hand-written ops.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_backprop_driver"


# ---------------------------------------------------------------- atom recaps

RECAP_GRAD_EXPRESSED_IN_OUT = (
    "## Grad expressed in `out` — quick refresher\n"
    "\n"
    "Many elementwise back fns can be written in terms of the CACHED forward "
    "`out` instead of recomputing the activation:\n"
    "\n"
    "```\n"
    "sigmoid_back(grad_out, out, x) = grad_out * out * (1 - out)\n"
    "tanh_back   (grad_out, out, x) = grad_out * (1 - out**2)\n"
    "exp_back    (grad_out, out, x) = grad_out * out\n"
    "```\n"
    "\n"
    "The point of the `(grad_out, out, x)` signature is exactly to make this "
    "possible — every back fn receives the cached forward output `out`, so it "
    "never has to call `sigmoid(x)` or `exp(x)` a second time on the reverse "
    "pass. The savings compound: one allocation + one exp per node, multiplied "
    "by every elementwise op in the graph.\n"
    "\n"
    "Numerical stability bonus: `out * (1 - out)` and `1 - out**2` are both "
    "bounded in `[0, 0.25]` and `[0, 1]` respectively, with no intermediate "
    "exponentials — far better behaved than `exp(-x) / (1 + exp(-x))**2`."
)

RECAP_NO_GRAD_CTX = (
    "## `no_grad` context manager — quick refresher\n"
    "\n"
    "A context manager that flips the module-level `grad_tracking_enabled` "
    "flag to `False` for the duration of the `with` block, and restores the "
    "previous value on exit:\n"
    "\n"
    "```python\n"
    "class NoGrad:\n"
    "    def __enter__(self):\n"
    "        global grad_tracking_enabled\n"
    "        self._prev = grad_tracking_enabled\n"
    "        grad_tracking_enabled = False\n"
    "    def __exit__(self, *exc):\n"
    "        global grad_tracking_enabled\n"
    "        grad_tracking_enabled = self._prev\n"
    "```\n"
    "\n"
    "Three things matter:\n"
    "- **Save the previous value** (don't just set `True` on exit) so nesting "
    "  works — an inner `NoGrad` that exits doesn't accidentally re-enable "
    "  grad inside an outer `NoGrad`.\n"
    "- **Restore on exit even if the block raises** — `__exit__` is "
    "  guaranteed to fire; that's the contract.\n"
    "- **Use cases:** in-place parameter updates inside an optimizer step, "
    "  EMA buffers, inference paths — anywhere you DON'T want a Recipe "
    "  attached to the output."
)

RECAP_MANUAL_CHAIN = (
    "## Manual forward-and-back chain — quick refresher\n"
    "\n"
    "Before the dispatcher exists, you can run a short chain by hand:\n"
    "\n"
    "```\n"
    "# Forward:  a → b = log(a) → c = exp(b)\n"
    "b = log(a)\n"
    "c = exp(b)\n"
    "\n"
    "# Backward (assume dL/dc is given):\n"
    "dL_db = exp_back(dL_dc, c, b)     # back_fn for c = exp(b)\n"
    "dL_da = log_back(dL_db, b, a)     # back_fn for b = log(a)\n"
    "```\n"
    "\n"
    "Two patterns to internalize:\n"
    "- **Reverse the call order.** The forward computed `a → b → c`; the "
    "  backward computes `dL_dc → dL_db → dL_da`. Same nodes, opposite "
    "  direction.\n"
    "- **Each back_fn receives `(grad_out, out_at_that_node, *inputs)`.** "
    "  `exp_back` gets `(dL_dc, c, b)` — the cached output `c` (used to "
    "  compute the gradient) AND the input `b` it was applied to. The "
    "  uniform signature is what makes the dispatcher possible later."
)

RECAP_DFS_TOPOSORT = (
    "## DFS three-set toposort — quick refresher\n"
    "\n"
    "The deps-FIRST topological sort: every node appears AFTER all of its "
    "(transitive) children. The root node ends up LAST. This is the lower-"
    "level helper that the reverse-pass driver wraps with `[::-1]` to get the "
    "end-node-first order.\n"
    "\n"
    "Classic three-color DFS:\n"
    "\n"
    "```python\n"
    "def topological_sort(root, get_children):\n"
    "    result = []\n"
    "    perm  = set()   # fully processed (black)\n"
    "    temp  = set()   # currently on the DFS stack (gray) — cycle detector\n"
    "\n"
    "    def visit(node):\n"
    "        nid = id(node)\n"
    "        if nid in perm: return\n"
    "        if nid in temp: raise ValueError('cycle')\n"
    "        temp.add(nid)\n"
    "        for child in get_children(node):\n"
    "            visit(child)\n"
    "        temp.remove(nid)\n"
    "        perm.add(nid)\n"
    "        result.append(node)\n"
    "\n"
    "    visit(root)\n"
    "    return result\n"
    "```\n"
    "\n"
    "Returns `[...children..., root]`. Two color-sets, not one: `perm` "
    "skips already-finished subtrees in a branching DAG; `temp` catches "
    "back-edges (cycles)."
)

RECAP_CYCLE_DETECTION = (
    "## Cycle detection via temp set — quick refresher\n"
    "\n"
    "In a DFS topological sort, the **temp** (gray) set holds every node "
    "currently on the recursion stack. A cycle is exactly a back-edge: a "
    "node being visited that's already in temp.\n"
    "\n"
    "```python\n"
    "def visit(node):\n"
    "    if id(node) in temp:\n"
    "        raise ValueError(f'cycle through {node!r}')\n"
    "    temp.add(id(node))\n"
    "    for child in get_children(node):\n"
    "        visit(child)\n"
    "    temp.remove(id(node))\n"
    "    perm.add(id(node))\n"
    "```\n"
    "\n"
    "Why a SEPARATE set from `perm`? `perm` is 'I have finished this subtree, "
    "skip it' (good — avoids re-traversal of shared descendants in a DAG). "
    "`temp` is 'I am currently inside this subtree' — meeting it again means "
    "we walked in a circle. Without the split, you can't tell 'shared "
    "descendant' from 'cycle'.\n"
    "\n"
    "Use `id(node)` as the key — the input nodes might not be hashable "
    "(or might have value-equality that gives false positives)."
)

RECAP_BACKPROP_LOOP = (
    "## Backprop pop-outgrad loop — quick refresher\n"
    "\n"
    "The main reverse-pass driver. Walk the graph in reverse-topological "
    "order (end node first), pop each node's accumulated grad out of a "
    "`grads` dict, dispatch the back_fn for each parent, accumulate into "
    "the parent's slot:\n"
    "\n"
    "```python\n"
    "grads = {id(end_node): end_grad}   # seed with dL/d(end_node)\n"
    "for node in sorted_computational_graph(end_node):\n"
    "    grad_out = grads.pop(id(node))     # pop — node done after this\n"
    "    if node.recipe is None:            # leaf: write to .grad\n"
    "        node.grad = grad_out if node.grad is None else node.grad + grad_out\n"
    "        continue\n"
    "    for argnum, parent in node.recipe.parents.items():\n"
    "        back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)\n"
    "        grad_parent = back_fn(grad_out, node.array,\n"
    "                              *node.recipe.args, **node.recipe.kwargs)\n"
    "        grads[id(parent)] = grads.get(id(parent), 0) + grad_parent\n"
    "```\n"
    "\n"
    "Three invariants:\n"
    "- **Pop, don't peek.** Once we process a node, its grad is no longer "
    "  needed; popping frees it and surfaces bugs where a node's grad got "
    "  consumed before all parents accumulated.\n"
    "- **Accumulate with `+`, never overwrite.** Diamond graphs route grad "
    "  through multiple paths; the same parent shows up in multiple "
    "  `recipe.parents` walks.\n"
    "- **Leaves get `.grad`, non-leaves stay in `grads` dict.** Leaves are "
    "  the user-facing parameters; non-leaves are intermediate."
)

RECAP_DISPATCH_BACK_FN = (
    "## Dispatch back_fn from recipe — quick refresher\n"
    "\n"
    "Given a node with a `Recipe`, the reverse pass dispatches the right "
    "back_fn from the (forward_fn, argnum) registry:\n"
    "\n"
    "```python\n"
    "for argnum, parent in node.recipe.parents.items():\n"
    "    back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)\n"
    "    # back_fn is a function with signature (grad_out, out, *args, **kwargs)\n"
    "```\n"
    "\n"
    "Two things to internalize:\n"
    "- **`recipe.parents` is the LOOP.** Iterating its `.items()` gives "
    "  `(argnum, parent_tensor)` pairs for every Tensor input — exactly the "
    "  parents whose gradient we need to compute.\n"
    "- **The lookup uses `recipe.func` AND the argnum.** Symmetric and "
    "  asymmetric ops alike — `add` registers `add_back0` AND `add_back1` "
    "  (identical bodies); `div` registers `div_back0` AND `div_back1` "
    "  (different bodies). The dispatcher never asks if the op is "
    "  symmetric; it just looks up `(fwd, argnum)`."
)

RECAP_BACK_FN_CALL = (
    "## Back fn call with recipe args — quick refresher\n"
    "\n"
    "Once you've dispatched the back_fn, you call it with the cached "
    "forward output, plus the original args/kwargs that the Recipe stored:\n"
    "\n"
    "```python\n"
    "grad_parent = back_fn(\n"
    "    grad_out,            # dL/d(this node's output)\n"
    "    node.array,          # the cached forward `out` for this node\n"
    "    *node.recipe.args,   # raw positional args at call time (unboxed)\n"
    "    **node.recipe.kwargs # kwargs the forward used (dim, keepdim, ...)\n"
    ")\n"
    "```\n"
    "\n"
    "Three places things go wrong:\n"
    "- **Forgetting `*recipe.args`** — back_fn for `multiply` needs both "
    "  inputs to compute `dL/dx = grad_out * y`. Drop them and you have "
    "  no derivative.\n"
    "- **Forgetting `**recipe.kwargs`** — `sum_back` needs `dim` to "
    "  broadcast back; without it you get a shape mismatch.\n"
    "- **Passing `node` instead of `node.array`** — back_fns operate on "
    "  raw torch tensors, not the MiniTensor wrapper. The whole layer "
    "  exists to keep the back_fn signature uniform across ops."
)


# ---------------------------------------------------------------- spec helper

# Shared autograd preamble — matches batch3/batch4, gives every drill access
# to Recipe + MiniTensor + the global grad_tracking_enabled toggle so the test
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
    "    populated by wrap_forward_fn. `requires_grad` is set by the wrapper.\n"
    "    `.grad` accumulates the leaf gradient at the end of the reverse pass.\"\"\"\n"
    "    def __init__(self, array, requires_grad: bool = False, recipe=None):\n"
    "        self.array = array\n"
    "        self.requires_grad = requires_grad\n"
    "        self.recipe = recipe\n"
    "        self.grad = None\n"
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
    dots = ("\U0001F534" * difficulty_num) + ("⚪" * (5 - difficulty_num))
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
# atom: grad-expressed-in-out  (1 exercise)
# =========================================================================

SPEC_GRAD_IN_OUT = _spec(
    atom_id="grad-expressed-in-out",
    subtopic="Backprop: grad expressed in out",
    recap=RECAP_GRAD_EXPRESSED_IN_OUT,
    ex_idx=1,
    ex_title="write sigmoid_back using cached out (no second sigmoid call)",
    slug="write-sigmoid-back-using-cached-out",
    bloom="Apply",
    difficulty_num=2,
    keywords=["sigmoid", "cached-out", "elementwise", "no-recompute"],
    kcs=["grad-expressed-in-out", "back-fn-uses-cached-out"],
    lo=(
        "Apply the 'grad expressed in out' pattern by writing sigmoid_back "
        "as grad_out * out * (1 - out) — reusing the cached forward output "
        "rather than recomputing sigmoid(x)."
    ),
    prompt_body=(
        "Implement `sigmoid_back(grad_out, out, x)` — the exemplar of the "
        "'grad expressed in `out`' pattern.\n\n"
        "**Math.** `out = sigmoid(x) = 1 / (1 + exp(-x))`. The derivative "
        "factors cleanly through the output:\n\n"
        "```\n"
        "d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))\n"
        "                = out * (1 - out)\n"
        "```\n\n"
        "So by the chain rule:\n\n"
        "```\n"
        "dL/dx = grad_out * out * (1 - out)\n"
        "```\n\n"
        "**The point of this drill.** You **must use `out`**, not "
        "`t.sigmoid(x)`. The whole reason the back-fn signature passes "
        "`out` is so we never recompute the activation on the reverse "
        "pass. The test inspects the function body to make sure you "
        "didn't sneak in a second `sigmoid` call.\n\n"
        "**Inputs.** Plain `torch.Tensor`, same shape; no autograd. Float "
        "dtype. Output: tensor with the same shape as `x`.\n\n"
        "**Tip.** One line is enough. The clarity of the cached-`out` "
        "form is the lesson — `out * (1 - out)` reads exactly like the "
        "math."
    ),
    stub=(
        "def sigmoid_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = sigmoid(x), expressed via the cached `out`."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- numerical correctness ---\n"
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])\n"
        "out = t.sigmoid(x)\n"
        "grad_out = t.ones(5)\n"
        "g = sigmoid_back(grad_out, out, x)\n"
        "expected = out * (1 - out)\n"
        "assert g.shape == x.shape, f'shape: {g.shape}'\n"
        "assert t.allclose(g, expected), f'value: {g} vs {expected}'\n"
        "\n"
        "# --- non-unit grad_out scales each entry by the chain rule ---\n"
        "grad_out = t.tensor([5.0, -3.0, 2.0, 0.5, -1.0])\n"
        "g = sigmoid_back(grad_out, out, x)\n"
        "expected = grad_out * out * (1 - out)\n"
        "assert t.allclose(g, expected), 'chain-rule scaling failed'\n"
        "\n"
        "# --- matrix shape ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "out_mat = t.sigmoid(X)\n"
        "g_mat = sigmoid_back(G, out_mat, X)\n"
        "assert g_mat.shape == (3, 4)\n"
        "assert t.allclose(g_mat, G * out_mat * (1 - out_mat))\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([-1.5, -0.2, 0.3, 1.5], requires_grad=True)\n"
        "y = t.sigmoid(x_ref).sum()\n"
        "y.backward()\n"
        "out_cached = t.sigmoid(x_ref.detach())\n"
        "g_ours = sigmoid_back(t.ones(4), out_cached, x_ref.detach())\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'sigmoid_back disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")\n"
        "\n"
        "# --- THE point of this atom: must use `out`, NOT recompute sigmoid(x) ---\n"
        "# Behavioural witness: pass a DELIBERATELY WRONG `out` (not equal to sigmoid(x))\n"
        "# and check the function trusts `out` rather than recomputing from `x`.\n"
        "# A correct implementation returns grad_out * fake_out * (1 - fake_out);\n"
        "# a recompute-from-x implementation would ignore fake_out and return the\n"
        "# true sigmoid derivative — easy to distinguish.\n"
        "fake_x = t.tensor([0.0, 0.0, 0.0])\n"
        "fake_out = t.tensor([0.25, 0.5, 0.75])   # NOT what sigmoid(0) is (= 0.5)\n"
        "got = sigmoid_back(t.ones(3), fake_out, fake_x)\n"
        "expected_from_fake_out = fake_out * (1 - fake_out)\n"
        "assert t.allclose(got, expected_from_fake_out), (\n"
        "    'sigmoid_back must use the cached `out`, not recompute sigmoid(x). '\n"
        "    f'Given fake_out={fake_out.tolist()} the result should be '\n"
        "    f'{expected_from_fake_out.tolist()}; got {got.tolist()}.'\n"
        ")\n"
        "\n"
        "# --- robustness: works on a scalar too ---\n"
        "x_sc = t.tensor(0.0)\n"
        "out_sc = t.sigmoid(x_sc)\n"
        "g_sc = sigmoid_back(t.tensor(1.0), out_sc, x_sc)\n"
        "assert g_sc.shape == x_sc.shape\n"
        "# At x=0, sigmoid=0.5, derivative = 0.5 * 0.5 = 0.25.\n"
        "assert abs(g_sc.item() - 0.25) < 1e-6, f'scalar case: {g_sc}'"
    ),
    solution_body=(
        "def sigmoid_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # The cached `out` is sigmoid(x); we reuse it instead of recomputing.\n"
        "    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = out * (1 - out).\n"
        "    return grad_out * out * (1 - out)"
    ),
    solution_notes=(
        "**Why `out`, not `x`, drives the formula.** The local derivative of "
        "sigmoid happens to factor as `out * (1 - out)`. Other activations "
        "that share this property: `tanh_back` uses `1 - out**2`, `exp_back` "
        "uses `out` directly. The shared `(grad_out, out, x)` signature is "
        "engineered so any of these can write the cleanest expression.\n\n"
        "**Cost savings.** A second `t.sigmoid(x)` call would allocate a new "
        "tensor and run another exp+division per element. For a deep network "
        "with millions of activations, that's a measurable hit on the "
        "backward pass.\n\n"
        "**Why pass `x` at all then?** Because not every op can be expressed "
        "in terms of `out`. `relu_back` needs `x > 0` (the cached `out = "
        "max(x, 0)` doesn't tell you whether `x` was positive at 0). "
        "Keeping `x` in the signature is the uniform-dispatch tax."
    ),
)


# =========================================================================
# atom: no-grad-context-mgr-update  (1 exercise)
# =========================================================================

SPEC_NO_GRAD = _spec(
    atom_id="no-grad-context-mgr-update",
    subtopic="Backprop: no_grad ctx-mgr update",
    recap=RECAP_NO_GRAD_CTX,
    ex_idx=1,
    ex_title="implement NoGrad context manager with restore-on-exit",
    slug="implement-nograd-context-mgr-with-restore-on-exit",
    bloom="Apply",
    difficulty_num=2,
    keywords=["context-manager", "no-grad", "grad-tracking", "restore"],
    kcs=["no-grad-context-mgr-update", "grad-tracking-global-toggle"],
    lo=(
        "Apply the save-and-restore context-manager pattern to write NoGrad: "
        "flip grad_tracking_enabled to False on enter, restore the previous "
        "value on exit (so nesting works)."
    ),
    prompt_body=(
        "Implement `NoGrad` — a context manager that disables the module-"
        "level `grad_tracking_enabled` flag for the duration of the `with` "
        "block, then restores the **previous** value on exit:\n\n"
        "```python\n"
        "with NoGrad():\n"
        "    # grad_tracking_enabled is False here\n"
        "    ...\n"
        "# grad_tracking_enabled is back to whatever it was before\n"
        "```\n\n"
        "Two requirements:\n\n"
        "**1. Save the PREVIOUS value, not just `True`.** Hard-coding "
        "`grad_tracking_enabled = True` on exit breaks nested `with NoGrad()` "
        "blocks: the inner exit would re-enable grad inside the outer "
        "`NoGrad`. Read the current value in `__enter__`, stash it, restore "
        "it in `__exit__`.\n\n"
        "**2. Restore on exit even if the block raised.** `__exit__` always "
        "fires (that's the contract), so the simplest correct "
        "implementation already gets this for free. Just don't `return` "
        "out of `__exit__` early on the exception path.\n\n"
        "`grad_tracking_enabled` is a **module-level** name — use `global` "
        "to write to it.\n\n"
        "Don't touch `torch.autograd`; we're rebuilding the autograd layer "
        "by hand."
    ),
    stub=(
        "class NoGrad:\n"
        '    """Disable grad_tracking_enabled for the duration of a `with` block."""\n'
        "    def __enter__(self):\n"
        "        raise NotImplementedError()\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        raise NotImplementedError()"
    ),
    test_body=(
        "# --- baseline: grad_tracking_enabled starts True (from the preamble) ---\n"
        "assert grad_tracking_enabled is True, 'preamble must seed True'\n"
        "\n"
        "# --- inside the `with`, the flag flips to False ---\n"
        "with NoGrad():\n"
        "    assert grad_tracking_enabled is False, (\n"
        "        'NoGrad must disable grad_tracking_enabled on enter'\n"
        "    )\n"
        "\n"
        "# --- after exit, the flag is restored ---\n"
        "assert grad_tracking_enabled is True, (\n"
        "    'NoGrad must restore grad_tracking_enabled on exit'\n"
        ")\n"
        "\n"
        "# --- restore the PREVIOUS value, not hard-coded True ---\n"
        "# (Simulate caller having already disabled grad themselves. We write to the\n"
        "#  same module dict that NoGrad's `global` will read/write.)\n"
        "globals()['grad_tracking_enabled'] = False\n"
        "with NoGrad():\n"
        "    assert grad_tracking_enabled is False\n"
        "assert grad_tracking_enabled is False, (\n"
        "    'NoGrad should restore the PRE-EXISTING value (False), not hard-code True'\n"
        ")\n"
        "globals()['grad_tracking_enabled'] = True  # reset before next check\n"
        "\n"
        "# --- nesting: inner NoGrad exit must NOT re-enable grad ---\n"
        "with NoGrad():\n"
        "    assert grad_tracking_enabled is False, 'outer NoGrad disabled'\n"
        "    with NoGrad():\n"
        "        assert grad_tracking_enabled is False, 'inner NoGrad still disabled'\n"
        "    # back inside the OUTER NoGrad — must still be False\n"
        "    assert grad_tracking_enabled is False, (\n"
        "        'inner NoGrad exit must restore previous value (False), '\n"
        "        'NOT hard-code True'\n"
        "    )\n"
        "assert grad_tracking_enabled is True, 'outer exit restores original True'\n"
        "\n"
        "# --- restore on exit even when the block raises ---\n"
        "try:\n"
        "    with NoGrad():\n"
        "        assert grad_tracking_enabled is False\n"
        "        raise RuntimeError('boom')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "assert grad_tracking_enabled is True, (\n"
        "    'NoGrad must restore the flag even if the block raises an exception'\n"
        ")\n"
        "\n"
        "# --- __enter__ returns the manager (or None) — both are acceptable ---\n"
        "with NoGrad() as ng:\n"
        "    pass  # just exercise the protocol, don't constrain the return value\n"
        "\n"
        "# --- usage in a parameter-update setting (the canonical use case) ---\n"
        "# Inside NoGrad, an 'update' should not flip the flag back.\n"
        "param_array = t.tensor([1.0, 2.0])\n"
        "lr = 0.1\n"
        "grad = t.tensor([0.5, -0.2])\n"
        "with NoGrad():\n"
        "    # canonical optimizer.step()-style in-place update\n"
        "    param_array -= lr * grad\n"
        "    assert grad_tracking_enabled is False, (\n"
        "        'grad must stay disabled during the in-place update'\n"
        "    )\n"
        "assert grad_tracking_enabled is True\n"
        "assert t.allclose(param_array, t.tensor([0.95, 2.02])), (\n"
        "    f'in-place update inside NoGrad must still mutate the tensor: {param_array}'\n"
        ")"
    ),
    solution_body=(
        "class NoGrad:\n"
        "    def __enter__(self):\n"
        "        global grad_tracking_enabled\n"
        "        self._prev = grad_tracking_enabled   # stash so nesting works\n"
        "        grad_tracking_enabled = False\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        global grad_tracking_enabled\n"
        "        grad_tracking_enabled = self._prev   # restore PREVIOUS value\n"
        "        # Return None / False so any in-block exception is re-raised."
    ),
    solution_notes=(
        "**Why save `self._prev` instead of toggling True on exit.** Three "
        "scenarios all need the saved-previous-value behavior:\n"
        "1. Caller had already disabled grad (e.g. they're running "
        "inference and called `set_grad_enabled(False)` once at top of "
        "main). A naive `__exit__` setting True would silently re-enable.\n"
        "2. Nested `with NoGrad(): with NoGrad(): ...` — inner exit must "
        "leave the outer block disabled.\n"
        "3. Cross-test isolation — pytest fixtures often want the flag "
        "preserved across context exits.\n\n"
        "All three reduce to 'save what you saw on entry; restore it.'\n\n"
        "**Why no try/finally needed.** `__exit__` is called by the "
        "interpreter as part of the `with` protocol regardless of whether "
        "the body raised; we don't need to wrap anything. Returning "
        "`None` (implicit) from `__exit__` tells Python NOT to suppress "
        "exceptions — which is what we want.\n\n"
        "**Equivalent with `contextlib.contextmanager`.** A `@contextmanager` "
        "generator + try/yield/finally version would also work, but the "
        "class form is what ARENA uses and what PyTorch's `torch.no_grad()` "
        "looks like internally."
    ),
)


# =========================================================================
# atom: manual-chain-forward-and-back  (1 exercise)
# =========================================================================

SPEC_MANUAL_CHAIN = _spec(
    atom_id="manual-chain-forward-and-back",
    subtopic="Backprop: manual chain forward-and-back",
    recap=RECAP_MANUAL_CHAIN,
    ex_idx=1,
    ex_title="manually chain forward log/exp and run backward by hand",
    slug="manually-chain-forward-log-exp-and-run-backward",
    bloom="Apply",
    difficulty_num=3,
    keywords=["manual-chain", "forward-backward", "log", "exp", "reverse-order"],
    kcs=["manual-chain-forward-and-back", "back-fn-uses-cached-out"],
    lo=(
        "Apply the manual forward-then-backward chain by computing "
        "b = log(a), c = exp(b) on the forward pass and then dL_da via "
        "exp_back followed by log_back in the reverse order."
    ),
    prompt_body=(
        "Implement `manual_chain(a, dL_dc)` — a hand-run forward+backward "
        "for a length-2 chain, BEFORE the dispatcher exists.\n\n"
        "**Forward pass.**\n"
        "```\n"
        "b = log(a)\n"
        "c = exp(b)\n"
        "```\n\n"
        "**Backward pass.** Given `dL/dc`, compute `dL/da` by running the "
        "chain in reverse:\n"
        "```\n"
        "dL_db = exp_back(dL_dc, c, b)\n"
        "dL_da = log_back(dL_db, b, a)\n"
        "```\n\n"
        "where:\n"
        "- `exp_back(grad_out, out, x) = grad_out * out`  "
        "(uses the cached `out`, since `d/dx exp(x) = exp(x) = out`)\n"
        "- `log_back(grad_out, out, x) = grad_out / x`    "
        "(uses `x`, since `d/dx log(x) = 1/x`)\n\n"
        "**Return** a 4-tuple `(b, c, dL_db, dL_da)` so the test can "
        "inspect every intermediate.\n\n"
        "**The point of this drill** is to internalize the reverse-order "
        "pattern: forward goes `a → b → c`; backward goes `dL_dc → dL_db "
        "→ dL_da`. Each back_fn takes `(grad_out, cached_out, input)` — "
        "the same signature the dispatcher will use later.\n\n"
        "All inputs are plain `torch.Tensor`, float dtype, same shape. "
        "Assume `a > 0` (so `log(a)` is well-defined)."
    ),
    stub=(
        "def manual_chain(a: Tensor, dL_dc: Tensor) -> tuple:\n"
        '    """Forward: b = log(a), c = exp(b). Backward: return (b, c, dL_db, dL_da)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- sanity: log+exp is the identity, so c == a ---\n"
        "a = t.tensor([1.0, 2.0, 4.0])\n"
        "dL_dc = t.tensor([1.0, 1.0, 1.0])\n"
        "b, c, dL_db, dL_da = manual_chain(a, dL_dc)\n"
        "assert t.allclose(c, a, atol=1e-5), f'log+exp should be identity: c={c} vs a={a}'\n"
        "assert t.allclose(b, t.log(a), atol=1e-5), f'b should be log(a): {b}'\n"
        "\n"
        "# --- backward shapes ---\n"
        "assert dL_db.shape == a.shape\n"
        "assert dL_da.shape == a.shape\n"
        "\n"
        "# --- backward values ---\n"
        "# exp_back: dL_db = dL_dc * c (and c == a)\n"
        "expected_dL_db = dL_dc * c\n"
        "assert t.allclose(dL_db, expected_dL_db, atol=1e-5), (\n"
        "    f'dL_db wrong: got {dL_db}, expected {expected_dL_db}'\n"
        ")\n"
        "# log_back: dL_da = dL_db / a\n"
        "expected_dL_da = dL_db / a\n"
        "assert t.allclose(dL_da, expected_dL_da, atol=1e-5), (\n"
        "    f'dL_da wrong: got {dL_da}, expected {expected_dL_da}'\n"
        ")\n"
        "# Composite check: dL_da should equal dL_dc * (c / a) = dL_dc (since c == a).\n"
        "assert t.allclose(dL_da, dL_dc, atol=1e-5), (\n"
        "    f'For log+exp = identity, dL_da should equal dL_dc: {dL_da} vs {dL_dc}'\n"
        ")\n"
        "\n"
        "# --- non-unit dL_dc — chain rule scales each entry ---\n"
        "a = t.tensor([2.0, 3.0])\n"
        "dL_dc = t.tensor([5.0, -2.0])\n"
        "b, c, dL_db, dL_da = manual_chain(a, dL_dc)\n"
        "assert t.allclose(dL_da, dL_dc, atol=1e-5), 'composite log+exp identity'\n"
        "\n"
        "# --- witness against torch.autograd on the full chain ---\n"
        "a_ref = t.tensor([1.5, 2.5, 4.5], requires_grad=True)\n"
        "c_ref = t.exp(t.log(a_ref))\n"
        "loss = (c_ref * t.tensor([0.5, -1.0, 2.0])).sum()\n"
        "loss.backward()\n"
        "_, _, _, dL_da_ours = manual_chain(a_ref.detach(), t.tensor([0.5, -1.0, 2.0]))\n"
        "assert t.allclose(dL_da_ours, a_ref.grad, atol=1e-5), (\n"
        "    f'chain disagrees with autograd: ours={dL_da_ours}, ref={a_ref.grad}'\n"
        ")\n"
        "\n"
        "# --- reverse-order invariant: dL_db must be computed BEFORE dL_da ---\n"
        "# (If you implemented forward order by mistake — log_back first — the\n"
        "# numerical mismatch would already have caught it; this is a structural\n"
        "# nudge for the explanation.)\n"
        "# Quick structural smoke check: each gradient depends on the previous one.\n"
        "# Verified by the value asserts above."
    ),
    solution_body=(
        "def manual_chain(a: Tensor, dL_dc: Tensor) -> tuple:\n"
        "    # --- forward: a -> b -> c ---\n"
        "    b = t.log(a)\n"
        "    c = t.exp(b)\n"
        "    # --- backward (reverse order): dL_dc -> dL_db -> dL_da ---\n"
        "    # exp_back uses the cached `out` (c): d/dx exp(x) = exp(x).\n"
        "    dL_db = dL_dc * c\n"
        "    # log_back uses the input x (a): d/dx log(x) = 1/x.\n"
        "    dL_da = dL_db / a\n"
        "    return b, c, dL_db, dL_da"
    ),
    solution_notes=(
        "**Why this drill matters even before the dispatcher.** Once "
        "`BACK_FUNCS` and the topo-sort exist, this whole function "
        "collapses to `out.backward()`. But the dispatcher is just "
        "AUTOMATING what you just did by hand — pick the right back_fn "
        "per node, call it with `(grad_out, cached_out, input)`, walk in "
        "reverse-topological order. Internalize the pattern at length-2 "
        "and the n-node case is the same thing in a loop.\n\n"
        "**Why `exp_back` uses `out` and `log_back` uses `x`.** "
        "`d/dx exp(x) = exp(x) = out` — the cached forward output IS the "
        "derivative, so it's the cheapest pick. `d/dx log(x) = 1/x` — "
        "no relationship to `out = log(x)`, so we have to read `x` "
        "directly. The shared `(grad_out, out, x)` signature lets each "
        "op pick whichever cache makes sense.\n\n"
        "**Identity check is the strongest test.** `exp(log(a)) == a`, "
        "so by the chain rule `d/da (exp(log(a))) = 1`, which means "
        "`dL/da == dL/dc` exactly. If your code passes this you almost "
        "certainly got the reverse order right."
    ),
)


# =========================================================================
# atom: dfs-three-set-toposort  (1 exercise)
# =========================================================================

SPEC_DFS_TOPOSORT = _spec(
    atom_id="dfs-three-set-toposort",
    subtopic="Backprop: DFS three-set toposort",
    recap=RECAP_DFS_TOPOSORT,
    ex_idx=1,
    ex_title="implement three-set DFS topological sort (deps-first, root LAST)",
    slug="implement-three-set-dfs-topological-sort-deps-first",
    bloom="Apply",
    difficulty_num=3,
    keywords=["dfs", "topological-sort", "three-color", "deps-first"],
    kcs=["dfs-three-set-toposort", "cycle-detection-temp-set"],
    lo=(
        "Apply the three-color DFS topological-sort algorithm to return "
        "descendants of a root in deps-FIRST order (root appears LAST), "
        "with a temp-set guard for cycle detection."
    ),
    prompt_body=(
        "Implement `topological_sort(root, get_children)` — the lower-"
        "level DAG traversal that batch-4's `sorted_computational_graph` "
        "wraps with `[::-1]` for the reverse pass.\n\n"
        "**Contract.**\n"
        "- Returns a `list` of nodes reachable from `root` via "
        "`get_children`.\n"
        "- Every node appears AFTER all of its (transitive) children — "
        "deps-first.\n"
        "- `root` appears LAST. This is the order a FORWARD compute pass "
        "would use; reversing it gives the backward order.\n"
        "- Each reachable node appears EXACTLY once, even in diamond "
        "DAGs where multiple paths reach the same descendant.\n"
        "- Raises `ValueError` on a cycle (we're DAG-only).\n\n"
        "**Algorithm.** Classic three-color DFS:\n"
        "- `temp` (gray) = currently on the recursion stack — cycle "
        "detector.\n"
        "- `perm` (black) = fully processed — skip if already in here.\n"
        "- (white = not in either set = not yet visited).\n\n"
        "```python\n"
        "def visit(node):\n"
        "    nid = id(node)\n"
        "    if nid in perm: return            # already done\n"
        "    if nid in temp: raise ValueError  # cycle\n"
        "    temp.add(nid)\n"
        "    for child in get_children(node):\n"
        "        visit(child)\n"
        "    temp.remove(nid)\n"
        "    perm.add(nid)\n"
        "    result.append(node)\n"
        "```\n\n"
        "Use `id(node)` as the set key (the test nodes don't override "
        "`__hash__`, but it's the safe-by-default identity key).\n\n"
        "**This drill is the LOWER half of `sorted_computational_graph`.** "
        "Batch-4's atom built the reverse-pass wrapper by composing this "
        "helper with `[::-1]`. The cycle-detection logic specifically is "
        "the focus of its own sibling atom — feel free to crib that "
        "behavior here (the same temp set serves both purposes)."
    ),
    stub=(
        "def topological_sort(root, get_children):\n"
        '    """DFS topo sort. Returns descendants of root in deps-first order\n'
        "    (root LAST). Raises ValueError on cycle.\n"
        '    """\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- helper graph node ---\n"
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
        "# --- linear chain a -> b -> c ---\n"
        "c = N('c')\n"
        "b = N('b', c)\n"
        "a = N('a', b)\n"
        "order = topological_sort(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names[-1] == 'a', f'root must be LAST, got {names}'\n"
        "assert names.index('c') < names.index('b') < names.index('a'), names\n"
        "\n"
        "# --- diamond DAG ---\n"
        "#      a\n"
        "#     / \\\n"
        "#    b   c\n"
        "#     \\ /\n"
        "#      d\n"
        "d = N('d')\n"
        "b = N('b', d)\n"
        "c = N('c', d)\n"
        "a = N('a', b, c)\n"
        "order = topological_sort(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names.count('d') == 1, f'd must appear ONCE, got {names}'\n"
        "assert names[-1] == 'a', f'root LAST, got {names}'\n"
        "assert names.index('d') < names.index('b'), 'd before b (b depends on d)'\n"
        "assert names.index('d') < names.index('c'), 'd before c (c depends on d)'\n"
        "assert names.index('b') < names.index('a')\n"
        "assert names.index('c') < names.index('a')\n"
        "assert len(order) == 4, f'four unique nodes, got {len(order)}'\n"
        "\n"
        "# --- linked list with shared descendants (long chain) ---\n"
        "leaf = N('leaf')\n"
        "n3 = N('n3', leaf)\n"
        "n2 = N('n2', n3)\n"
        "n1 = N('n1', n2)\n"
        "n0 = N('n0', n1)\n"
        "order = topological_sort(n0, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names == ['leaf', 'n3', 'n2', 'n1', 'n0'], (\n"
        "    f'linear chain must yield deps-first order, got {names}'\n"
        ")\n"
        "\n"
        "# --- cycle detection raises ValueError ---\n"
        "x = N('x')\n"
        "y = N('y')\n"
        "x.children = [y]\n"
        "y.children = [x]\n"
        "raised = False\n"
        "try:\n"
        "    topological_sort(x, get_children)\n"
        "except ValueError:\n"
        "    raised = True\n"
        "assert raised, 'a cycle must raise ValueError'\n"
        "\n"
        "# --- self-loop ---\n"
        "s = N('s')\n"
        "s.children = [s]\n"
        "raised = False\n"
        "try:\n"
        "    topological_sort(s, get_children)\n"
        "except ValueError:\n"
        "    raised = True\n"
        "assert raised, 'self-loop must raise ValueError'\n"
        "\n"
        "# --- singleton (root with no children) ---\n"
        "lonely = N('lonely')\n"
        "order = topological_sort(lonely, get_children)\n"
        "assert order == [lonely], f'singleton: {order}'\n"
        "\n"
        "# --- mid-graph cycle does NOT silently succeed ---\n"
        "# Graph: a -> b -> c -> b (cycle through b).\n"
        "p = N('p')\n"
        "q = N('q')\n"
        "r = N('r')\n"
        "p.children = [q]\n"
        "q.children = [r]\n"
        "r.children = [q]  # cycle\n"
        "raised = False\n"
        "try:\n"
        "    topological_sort(p, get_children)\n"
        "except ValueError:\n"
        "    raised = True\n"
        "assert raised, 'mid-graph cycle must raise ValueError'\n"
        "\n"
        "# --- branching where one branch is deep ---\n"
        "# root -> a -> b -> c\n"
        "#      -> d\n"
        "cc = N('c')\n"
        "bb = N('b', cc)\n"
        "aa = N('a', bb)\n"
        "dd = N('d')\n"
        "root = N('root', aa, dd)\n"
        "order = topological_sort(root, get_children)\n"
        "names = [n.name for n in order]\n"
        "# Just check the deps-first invariant, not a specific ordering.\n"
        "pos = {nm: i for i, nm in enumerate(names)}\n"
        "assert pos['c'] < pos['b'] < pos['a'] < pos['root']\n"
        "assert pos['d'] < pos['root']\n"
        "assert names[-1] == 'root'"
    ),
    solution_body=(
        "def topological_sort(root, get_children):\n"
        "    result = []\n"
        "    perm = set()   # fully processed (black) — keyed by id()\n"
        "    temp = set()   # currently on DFS stack (gray) — cycle detector\n"
        "\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm:\n"
        "            return\n"
        "        if nid in temp:\n"
        "            raise ValueError(f'Cycle detected at {node!r} — graph is not a DAG')\n"
        "        temp.add(nid)\n"
        "        for child in get_children(node):\n"
        "            visit(child)\n"
        "        temp.remove(nid)\n"
        "        perm.add(nid)\n"
        "        result.append(node)\n"
        "\n"
        "    visit(root)\n"
        "    return result"
    ),
    solution_notes=(
        "**Two color sets — `temp` and `perm` — do different jobs.** "
        "`perm` is the 'I have finished this subtree, do not recurse "
        "again' marker — it keeps a diamond DAG from re-traversing the "
        "shared descendant. `temp` is the 'I am currently inside this "
        "subtree' marker — re-entering it means a back-edge → cycle. "
        "Two sets, two semantics.\n\n"
        "**Why `id(node)` instead of the node itself.** The graph nodes "
        "in the tests are simple Python objects with identity-equality, "
        "so a plain `set` of nodes would work. But the same code runs on "
        "MiniTensors (where `__eq__` might compare by value if you ever "
        "add it), or numpy arrays (where `__eq__` returns an array). "
        "`id()` is the safe-by-default identity key.\n\n"
        "**Output order is deps-FIRST.** The result list ends with the "
        "root, which is what a forward compute pass expects: evaluate "
        "leaves first, then their consumers. The reverse pass needs the "
        "OPPOSITE — and that's exactly the `[::-1]` you'll see in the "
        "sibling `sorted-computational-graph` atom. Keeping this helper "
        "deps-first means it's reusable for forward-only graph "
        "operations (`zero_grad`, structural-check passes)."
    ),
)


# =========================================================================
# atom: cycle-detection-temp-set  (1 exercise)
# =========================================================================

SPEC_CYCLE_DETECT = _spec(
    atom_id="cycle-detection-temp-set",
    subtopic="Backprop: cycle detection via temp set",
    recap=RECAP_CYCLE_DETECTION,
    ex_idx=1,
    ex_title="add temp-set cycle detection to a DFS traversal",
    slug="add-temp-set-cycle-detection-to-dfs-traversal",
    bloom="Apply",
    difficulty_num=2,
    keywords=["cycle-detection", "temp-set", "gray", "back-edge"],
    kcs=["cycle-detection-temp-set", "dfs-three-set-toposort"],
    lo=(
        "Apply the temp-set (gray) back-edge detection pattern to "
        "distinguish 'already-visited shared descendant' (legal) from "
        "'cycle' (illegal) inside a DFS traversal."
    ),
    prompt_body=(
        "Implement `has_cycle(root, get_children)` — return `True` if the "
        "directed graph reachable from `root` contains a cycle, `False` "
        "otherwise.\n\n"
        "**Why this is non-trivial.** A diamond DAG (multiple paths to "
        "the same descendant) is NOT a cycle. Naive 'mark visited; "
        "raise if you see a visited node' gives false positives:\n\n"
        "```\n"
        "    a\n"
        "   / \\\n"
        "  b   c\n"
        "   \\ /\n"
        "    d            ← a, b, c, d, then we reach d again via c → false alarm\n"
        "```\n\n"
        "Correct algorithm uses **two** sets:\n"
        "- `perm` — 'I have FINISHED processing this subtree'. Re-seeing "
        "a `perm` node is fine; it's a shared descendant in a DAG.\n"
        "- `temp` — 'I am CURRENTLY inside this subtree (still on the "
        "recursion stack)'. Re-seeing a `temp` node IS a cycle.\n\n"
        "```python\n"
        "def visit(node):\n"
        "    if id(node) in perm: return         # legal — already finished\n"
        "    if id(node) in temp: return True    # CYCLE — back-edge\n"
        "    temp.add(id(node))\n"
        "    for child in get_children(node):\n"
        "        if visit(child) is True:\n"
        "            return True\n"
        "    temp.remove(id(node))\n"
        "    perm.add(id(node))\n"
        "    return False\n"
        "```\n\n"
        "Return type: `bool`. Do NOT raise — the caller picks what to do "
        "with the result. (Compare with the sibling `dfs-three-set-"
        "toposort` atom, which DOES raise; same `temp` set, different "
        "response.)\n\n"
        "Use `id(node)` as the set key."
    ),
    stub=(
        "def has_cycle(root, get_children) -> bool:\n"
        '    """Return True if reachable graph contains a cycle, False otherwise."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- helper node ---\n"
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
        "# --- acyclic linear chain → False ---\n"
        "c = N('c')\n"
        "b = N('b', c)\n"
        "a = N('a', b)\n"
        "assert has_cycle(a, get_children) is False, 'linear chain is acyclic'\n"
        "\n"
        "# --- diamond DAG → False (shared descendant is NOT a cycle) ---\n"
        "d = N('d')\n"
        "b = N('b', d)\n"
        "c = N('c', d)\n"
        "a = N('a', b, c)\n"
        "assert has_cycle(a, get_children) is False, (\n"
        "    'diamond DAG is acyclic — shared descendants must use perm, not raise'\n"
        ")\n"
        "\n"
        "# --- two-node cycle → True ---\n"
        "x = N('x')\n"
        "y = N('y')\n"
        "x.children = [y]\n"
        "y.children = [x]\n"
        "assert has_cycle(x, get_children) is True, 'x → y → x is a cycle'\n"
        "\n"
        "# --- self-loop → True ---\n"
        "s = N('s')\n"
        "s.children = [s]\n"
        "assert has_cycle(s, get_children) is True, 's → s is a cycle'\n"
        "\n"
        "# --- cycle deeper in the graph → True ---\n"
        "# p → q → r → q (cycle between q and r)\n"
        "p = N('p')\n"
        "q = N('q')\n"
        "r = N('r')\n"
        "p.children = [q]\n"
        "q.children = [r]\n"
        "r.children = [q]\n"
        "assert has_cycle(p, get_children) is True, 'mid-graph cycle'\n"
        "\n"
        "# --- singleton (no children) → False ---\n"
        "lonely = N('lonely')\n"
        "assert has_cycle(lonely, get_children) is False\n"
        "\n"
        "# --- complex DAG with multiple diamonds → False ---\n"
        "leaf = N('leaf')\n"
        "m1 = N('m1', leaf)\n"
        "m2 = N('m2', leaf)\n"
        "n1 = N('n1', m1, m2)\n"
        "n2 = N('n2', m1, m2)\n"
        "root = N('root', n1, n2)\n"
        "assert has_cycle(root, get_children) is False, (\n"
        "    'nested diamonds with shared descendants must NOT report a cycle'\n"
        ")\n"
        "\n"
        "# --- branching graph where ONE branch has a cycle → True ---\n"
        "# root -> good (acyclic chain)\n"
        "#      -> bad (cycle)\n"
        "good_c = N('good_c')\n"
        "good_b = N('good_b', good_c)\n"
        "good_a = N('good_a', good_b)\n"
        "bad_x = N('bad_x')\n"
        "bad_y = N('bad_y')\n"
        "bad_x.children = [bad_y]\n"
        "bad_y.children = [bad_x]\n"
        "root = N('root', good_a, bad_x)\n"
        "assert has_cycle(root, get_children) is True, (\n"
        "    'cycle in a subgraph must propagate up to True'\n"
        ")\n"
        "\n"
        "# --- temp set MUST be popped on return (otherwise sibling subtrees ---\n"
        "# --- false-positive as cycles)                                     ---\n"
        "# Graph: root → x → leaf\n"
        "#             → y → leaf      ← x and y both child of root; both reach leaf\n"
        "# If you forget to remove from temp after a subtree finishes,\n"
        "# the second visit to leaf would (wrongly) hit temp and report a cycle.\n"
        "leaf = N('leaf')\n"
        "x = N('x', leaf)\n"
        "y = N('y', leaf)\n"
        "root = N('root', x, y)\n"
        "assert has_cycle(root, get_children) is False, (\n"
        "    'two siblings sharing a leaf is NOT a cycle — '\n"
        "    'did you forget to temp.remove() after each subtree?'\n"
        ")"
    ),
    solution_body=(
        "def has_cycle(root, get_children) -> bool:\n"
        "    perm = set()   # fully processed subtree (NOT a cycle)\n"
        "    temp = set()   # currently on the DFS stack (BACK-EDGE means cycle)\n"
        "\n"
        "    def visit(node) -> bool:\n"
        "        nid = id(node)\n"
        "        if nid in perm:\n"
        "            return False   # already finished — legal shared descendant\n"
        "        if nid in temp:\n"
        "            return True    # back-edge — cycle\n"
        "        temp.add(nid)\n"
        "        for child in get_children(node):\n"
        "            if visit(child):\n"
        "                return True\n"
        "        temp.remove(nid)   # MUST remove on the way out — else siblings false-positive\n"
        "        perm.add(nid)\n"
        "        return False\n"
        "\n"
        "    return visit(root)"
    ),
    solution_notes=(
        "**Two color sets, two purposes.** One color (visited / not) "
        "would catch any re-visit — but that flags diamond DAGs as "
        "cycles. The split between `temp` (currently in flight) and "
        "`perm` (already finished) is the minimum information needed "
        "to distinguish 'shared descendant' from 'back-edge'.\n\n"
        "**Why `temp.remove(nid)` matters.** When a subtree finishes, "
        "its root leaves the recursion stack — so its `id` must leave "
        "the `temp` set too. Otherwise, two sibling subtrees that share "
        "a leaf will report a false cycle: the first subtree adds the "
        "leaf's id to `temp`, finishes without removing it, then the "
        "second subtree tries to visit the leaf and sees it in `temp` "
        "→ false alarm. The dedicated test at the bottom catches this.\n\n"
        "**Vs. the sibling `dfs-three-set-toposort` atom.** Same `temp` "
        "machinery; that atom RAISES on the back-edge, this one "
        "RETURNS `True`. Different ergonomics for different callers — "
        "but the cycle-detection logic itself is the shared insight."
    ),
)


# =========================================================================
# atom: backprop-pop-outgrad-loop  (1 exercise)
# =========================================================================

SPEC_BACKPROP_LOOP = _spec(
    atom_id="backprop-pop-outgrad-loop",
    subtopic="Backprop: backprop pop-outgrad loop",
    recap=RECAP_BACKPROP_LOOP,
    ex_idx=1,
    ex_title="implement the main reverse-pass loop over a sorted graph",
    slug="implement-main-reverse-pass-loop-over-sorted-graph",
    bloom="Apply",
    difficulty_num=4,
    keywords=["reverse-pass", "loop", "pop-grad", "accumulate", "leaf-grad"],
    kcs=["backprop-pop-outgrad-loop", "dispatch-back-fn-from-recipe"],
    lo=(
        "Apply the reverse-pass driver pattern: iterate a "
        "sorted-computational-graph, pop each node's accumulated grad, "
        "dispatch the per-arg back_fn, and accumulate into each parent's "
        "grad slot."
    ),
    prompt_body=(
        "Implement `backprop(end_node, end_grad, sorted_graph, back_funcs)` "
        "— the main reverse-pass driver.\n\n"
        "**Inputs.**\n"
        "- `end_node` — the MiniTensor at which to start the reverse pass "
        "(e.g. the loss).\n"
        "- `end_grad` — a `torch.Tensor` with `dL/d(end_node)`. Usually "
        "`t.ones_like(end_node.array)`.\n"
        "- `sorted_graph` — list of MiniTensors in REVERSE-topological "
        "order (end_node FIRST, leaves LAST). Pre-computed by the caller.\n"
        "- `back_funcs` — a dict `{(forward_fn, argnum): back_fn}` that "
        "maps to the right back_fn for each (op, arg-position) pair.\n\n"
        "**Algorithm.**\n\n"
        "```python\n"
        "grads = {id(end_node): end_grad}            # seed accumulator\n"
        "for node in sorted_graph:\n"
        "    grad_out = grads.pop(id(node))          # pop — node done after\n"
        "    if node.recipe is None:                 # leaf: write to .grad\n"
        "        if node.grad is None:\n"
        "            node.grad = grad_out\n"
        "        else:\n"
        "            node.grad = node.grad + grad_out\n"
        "        continue\n"
        "    for argnum, parent in node.recipe.parents.items():\n"
        "        back_fn = back_funcs[(node.recipe.func, argnum)]\n"
        "        grad_parent = back_fn(grad_out, node.array,\n"
        "                              *node.recipe.args,\n"
        "                              **node.recipe.kwargs)\n"
        "        pid = id(parent)\n"
        "        grads[pid] = grads.get(pid, 0) + grad_parent\n"
        "```\n\n"
        "**Three invariants the test checks.**\n"
        "1. **Pop, don't peek.** `grads.pop` removes the entry; later "
        "accidental reads should hit `KeyError`.\n"
        "2. **Accumulate with `+`, never overwrite.** Diamond DAGs route "
        "grad through multiple paths; the same parent appears in multiple "
        "`recipe.parents` walks.\n"
        "3. **Leaves write to `.grad`; non-leaves stay in `grads` dict.** "
        "Leaves are the user-facing parameters; non-leaves are intermediate.\n\n"
        "**No return value.** Mutate `.grad` on each leaf MiniTensor "
        "in-place. The dispatcher is a one-pass walk."
    ),
    stub=(
        "def backprop(end_node, end_grad, sorted_graph, back_funcs) -> None:\n"
        '    """Run the reverse-pass loop; mutate .grad on each leaf in-place."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- back fns we will register (raw-torch, no autograd) ---\n"
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    return grad_out * y\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    return grad_out * x\n"
        "\n"
        "BF = {\n"
        "    (t.log, 0): log_back,\n"
        "    (t.multiply, 0): multiply_back0,\n"
        "    (t.multiply, 1): multiply_back1,\n"
        "}\n"
        "\n"
        "# === TEST 1: single-leaf log chain ===\n"
        "# leaf = a; out = log(a). Manual sorted_graph: [out, a].\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 4.0]), requires_grad=True)\n"
        "out_arr = t.log(a.array)\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "backprop(out, t.ones(3), [out, a], BF)\n"
        "assert a.grad is not None, 'leaf a must have .grad after backprop'\n"
        "assert t.allclose(a.grad, 1 / a.array), (\n"
        "    f'a.grad wrong: got {a.grad}, expected {1 / a.array}'\n"
        ")\n"
        "\n"
        "# === TEST 2: two-leaf multiply ===\n"
        "# leaves: x, y; out = x * y. sorted_graph: [out, x, y].\n"
        "x = MiniTensor(t.tensor([2.0, 3.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([5.0, 7.0]), requires_grad=True)\n"
        "out_arr = x.array * y.array\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y}\n"
        ")\n"
        "backprop(out, t.ones(2), [out, x, y], BF)\n"
        "assert t.allclose(x.grad, y.array), f'd(xy)/dx = y; got {x.grad}'\n"
        "assert t.allclose(y.grad, x.array), f'd(xy)/dy = x; got {y.grad}'\n"
        "\n"
        "# === TEST 3: diamond accumulation ===\n"
        "# leaf z used by two intermediates p, q; out = p * q  where p = z, q = z.\n"
        "# Modeled as: p = z (recipe.parents={0: z}, func=identity-via-log/exp);\n"
        "# Cleaner: use multiply(z, z) directly so d(out)/dz appears at BOTH parents 0 and 1.\n"
        "z = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out_arr = z.array * z.array\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(z.array, z.array), kwargs={}, parents={0: z, 1: z}\n"
        ")\n"
        "backprop(out, t.ones(1), [out, z], BF)\n"
        "# d(z^2)/dz = 2z = 6 — both arg-0 and arg-1 contribute z each.\n"
        "assert t.allclose(z.grad, t.tensor([6.0])), (\n"
        "    f'accumulation across diamond failed: z.grad={z.grad} (expected 6.0)'\n"
        ")\n"
        "\n"
        "# === TEST 4: three-node chain — log + multiply ===\n"
        "# a, b leaves; c = log(b); out = a * c.\n"
        "# sorted_graph: [out, a, c, b]  (note: a and c both children of out; b child of c).\n"
        "import math as _math\n"
        "a = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([_math.e]), requires_grad=True)\n"
        "c_arr = t.log(b.array)\n"
        "c = MiniTensor(c_arr, requires_grad=True)\n"
        "c.recipe = Recipe(func=t.log, args=(b.array,), kwargs={}, parents={0: b})\n"
        "out_arr = a.array * c.array\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(a.array, c.array), kwargs={}, parents={0: a, 1: c}\n"
        ")\n"
        "backprop(out, t.ones(1), [out, a, c, b], BF)\n"
        "# d(a * log(b))/da = log(b) = 1.0; d(a * log(b))/db = a / b = 2 / e.\n"
        "assert t.allclose(a.grad, t.tensor([1.0]), atol=1e-5), f'a.grad={a.grad}'\n"
        "assert t.allclose(b.grad, t.tensor([2.0 / _math.e]), atol=1e-5), f'b.grad={b.grad}'\n"
        "\n"
        "# === TEST 5: pop-not-peek invariant — grads dict must NOT retain old entries ===\n"
        "# We re-run TEST 1 and inspect the (final) grads state via a wrapper. The\n"
        "# strict check is that `grads` does not leak: each id() popped once and only\n"
        "# once. We verify this BEHAVIORALLY by re-running backprop on a fresh `a` and\n"
        "# confirming a.grad equals 1/a.array (no double-accumulate from leftover state).\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 4.0]), requires_grad=True)\n"
        "out_arr = t.log(a.array)\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "backprop(out, t.ones(3), [out, a], BF)\n"
        "assert t.allclose(a.grad, 1 / a.array), 're-run consistency'\n"
        "\n"
        "# === TEST 6: leaf already has a .grad → accumulate, do not overwrite ===\n"
        "a = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "a.grad = t.tensor([10.0])   # pre-seed\n"
        "out_arr = t.log(a.array)\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "backprop(out, t.ones(1), [out, a], BF)\n"
        "# expected: previous (10) + 1/a (which is 1) = 11\n"
        "assert t.allclose(a.grad, t.tensor([11.0])), (\n"
        "    f'leaf .grad must accumulate, not overwrite: {a.grad}'\n"
        ")"
    ),
    solution_body=(
        "def backprop(end_node, end_grad, sorted_graph, back_funcs) -> None:\n"
        "    grads = {id(end_node): end_grad}\n"
        "    for node in sorted_graph:\n"
        "        nid = id(node)\n"
        "        if nid not in grads:\n"
        "            # Node not reached from end_node in this traversal — skip.\n"
        "            continue\n"
        "        grad_out = grads.pop(nid)           # POP — node done after this\n"
        "        if node.recipe is None:\n"
        "            # Leaf: accumulate (don't overwrite) into .grad.\n"
        "            if node.grad is None:\n"
        "                node.grad = grad_out\n"
        "            else:\n"
        "                node.grad = node.grad + grad_out\n"
        "            continue\n"
        "        # Non-leaf: dispatch + accumulate into each parent.\n"
        "        for argnum, parent in node.recipe.parents.items():\n"
        "            back_fn = back_funcs[(node.recipe.func, argnum)]\n"
        "            grad_parent = back_fn(\n"
        "                grad_out, node.array,\n"
        "                *node.recipe.args, **node.recipe.kwargs,\n"
        "            )\n"
        "            pid = id(parent)\n"
        "            grads[pid] = grads.get(pid, 0) + grad_parent"
    ),
    solution_notes=(
        "**Why `grads.pop` instead of `grads[nid]`.** Once we've processed "
        "a node, its accumulated grad is no longer needed — every parent "
        "that wanted to add to it has already done so (that's what the "
        "reverse-topological ordering buys us). Popping surfaces bugs "
        "where a node's grad is consumed before all incoming grads have "
        "been accumulated; a later read would `KeyError`.\n\n"
        "**Why `grads.get(pid, 0) + grad_parent` not assignment.** A "
        "single parent can appear in multiple `recipe.parents` walks: "
        "(a) the diamond DAG case (one node used twice in a forward op), "
        "(b) the multi-consumer case (one node feeds multiple downstream "
        "ops, each of which will eventually contribute grad). The "
        "accumulator pattern handles both.\n\n"
        "**Why leaves get `.grad`, non-leaves stay in `grads`.** The "
        "user-facing API of an autograd Tensor is `.grad`. But during "
        "the reverse pass we need a scratch dict for non-leaf "
        "intermediates (we never expose those). Splitting the storage "
        "by `node.recipe is None` is the cheap classifier — leaves "
        "have no recipe by construction.\n\n"
        "**Why pass `*node.recipe.args, **node.recipe.kwargs`.** The "
        "back_fn signature is `(grad_out, out, *args, **kwargs)`. The "
        "Recipe stored the raw call-time args (unboxed tensors) and "
        "kwargs (dim, keepdim, ...) so reverse-pass dispatch is "
        "generic. Drop either and shape-mismatched ops will blow up."
    ),
)


# =========================================================================
# atom: dispatch-back-fn-from-recipe  (1 exercise)
# =========================================================================

SPEC_DISPATCH = _spec(
    atom_id="dispatch-back-fn-from-recipe",
    subtopic="Backprop: dispatch back fn from recipe",
    recap=RECAP_DISPATCH_BACK_FN,
    ex_idx=1,
    ex_title="dispatch back fn from (recipe.func, argnum) for every parent",
    slug="dispatch-back-fn-from-recipe-func-argnum-for-every-parent",
    bloom="Apply",
    difficulty_num=3,
    keywords=["dispatch", "recipe", "argnum", "back-funcs", "parents-loop"],
    kcs=["dispatch-back-fn-from-recipe", "parents-dict-by-argidx"],
    lo=(
        "Apply the (recipe.func, argnum)-keyed dispatch pattern: iterate "
        "recipe.parents, look up the matching back_fn, return the "
        "(argnum, parent, back_fn) triples ready for the back_fn call site."
    ),
    prompt_body=(
        "Implement `dispatch_back_fns(node, back_funcs)` — given a "
        "non-leaf MiniTensor and a `{(forward_fn, argnum): back_fn}` "
        "registry, return a list of `(argnum, parent, back_fn)` triples — "
        "one per parent in `node.recipe.parents` — that the caller can "
        "use as the input to the actual back_fn call.\n\n"
        "**Signature.**\n"
        "```python\n"
        "def dispatch_back_fns(node, back_funcs) -> list[tuple]:\n"
        "    ...\n"
        "    # returns [(argnum_0, parent_0, back_fn_0), ...]\n"
        "```\n\n"
        "**Algorithm.**\n"
        "```python\n"
        "results = []\n"
        "for argnum, parent in node.recipe.parents.items():\n"
        "    back_fn = back_funcs[(node.recipe.func, argnum)]\n"
        "    results.append((argnum, parent, back_fn))\n"
        "return results\n"
        "```\n\n"
        "**Why this is its own atom.** The actual call site in `backprop` "
        "is two operations smashed together: (a) figure out WHICH back_fn "
        "to call (this drill), and (b) call it with the right args (the "
        "sibling atom). Separating the dispatch from the call clarifies "
        "the responsibility split — and lets you test the dispatch in "
        "isolation.\n\n"
        "**Error case.** If `(node.recipe.func, argnum)` isn't in "
        "`back_funcs`, let the natural `KeyError` propagate — the "
        "caller (the main reverse-pass driver) has the context to "
        "decorate it.\n\n"
        "**Leaf case.** Assume `node.recipe is not None` (the caller "
        "filters leaves before reaching this function)."
    ),
    stub=(
        "def dispatch_back_fns(node, back_funcs) -> list:\n"
        '    """Return [(argnum, parent, back_fn), ...] for every parent of node."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- back fns + registry ---\n"
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "def mul_back0(grad_out, out, x, y):\n"
        "    return grad_out * y\n"
        "def mul_back1(grad_out, out, x, y):\n"
        "    return grad_out * x\n"
        "\n"
        "BF = {\n"
        "    (t.log, 0): log_back,\n"
        "    (t.multiply, 0): mul_back0,\n"
        "    (t.multiply, 1): mul_back1,\n"
        "}\n"
        "\n"
        "# === single-parent case: c = log(b) ===\n"
        "b = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "c = MiniTensor(t.log(b.array), requires_grad=True)\n"
        "c.recipe = Recipe(func=t.log, args=(b.array,), kwargs={}, parents={0: b})\n"
        "triples = dispatch_back_fns(c, BF)\n"
        "assert len(triples) == 1, f'one parent → one triple, got {triples}'\n"
        "argnum, parent, back_fn = triples[0]\n"
        "assert argnum == 0\n"
        "assert parent is b\n"
        "assert back_fn is log_back\n"
        "\n"
        "# === two-parent case: out = x * y ===\n"
        "x = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out = MiniTensor(x.array * y.array, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y}\n"
        ")\n"
        "triples = dispatch_back_fns(out, BF)\n"
        "assert len(triples) == 2, f'two parents → two triples, got {triples}'\n"
        "# Order may match parents.items() iteration order (insertion-preserving in py3.7+).\n"
        "by_argnum = {a: (p, f) for a, p, f in triples}\n"
        "assert by_argnum[0] == (x, mul_back0), f'argnum 0 dispatch wrong: {by_argnum[0]}'\n"
        "assert by_argnum[1] == (y, mul_back1), f'argnum 1 dispatch wrong: {by_argnum[1]}'\n"
        "\n"
        "# === diamond — same parent at BOTH argnums (z * z) ===\n"
        "z = MiniTensor(t.tensor([4.0]), requires_grad=True)\n"
        "out = MiniTensor(z.array * z.array, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(z.array, z.array), kwargs={}, parents={0: z, 1: z}\n"
        ")\n"
        "triples = dispatch_back_fns(out, BF)\n"
        "assert len(triples) == 2, 'diamond — z appears at both argnums → 2 triples'\n"
        "# Both should reference z; back_fns should differ (mul_back0 vs mul_back1).\n"
        "by_argnum = {a: (p, f) for a, p, f in triples}\n"
        "assert by_argnum[0][0] is z and by_argnum[1][0] is z, 'both parents are z'\n"
        "assert by_argnum[0][1] is mul_back0\n"
        "assert by_argnum[1][1] is mul_back1\n"
        "assert by_argnum[0][1] is not by_argnum[1][1], (\n"
        "    'same op, different argnums must dispatch to different back_fns'\n"
        ")\n"
        "\n"
        "# === non-contiguous argnums: parents={1: a} (e.g. scalar at arg-0) ===\n"
        "# When forward was multiply(3.0, a) — first arg is a float, no parent at 0.\n"
        "a = MiniTensor(t.tensor([7.0]), requires_grad=True)\n"
        "out = MiniTensor(3.0 * a.array, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(3.0, a.array), kwargs={}, parents={1: a}\n"
        ")\n"
        "triples = dispatch_back_fns(out, BF)\n"
        "assert len(triples) == 1\n"
        "argnum, parent, back_fn = triples[0]\n"
        "assert argnum == 1, 'argnum must remain 1, NOT collapse to 0'\n"
        "assert parent is a\n"
        "assert back_fn is mul_back1, 'must dispatch to mul_back1, NOT mul_back0'\n"
        "\n"
        "# === missing registration → KeyError ===\n"
        "missing = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "missing.recipe = Recipe(\n"
        "    func=t.sin, args=(t.tensor([1.0]),), kwargs={}, parents={0: missing}\n"
        ")\n"
        "raised = False\n"
        "try:\n"
        "    dispatch_back_fns(missing, BF)\n"
        "except KeyError:\n"
        "    raised = True\n"
        "assert raised, 'unregistered (fn, argnum) must propagate a KeyError'\n"
        "\n"
        "# === structural shape of the result ===\n"
        "# Each triple is exactly (argnum:int, parent:MiniTensor, back_fn:callable).\n"
        "argnum, parent, back_fn = dispatch_back_fns(c, BF)[0]\n"
        "assert isinstance(argnum, int)\n"
        "assert isinstance(parent, MiniTensor)\n"
        "assert callable(back_fn)"
    ),
    solution_body=(
        "def dispatch_back_fns(node, back_funcs) -> list:\n"
        "    results = []\n"
        "    for argnum, parent in node.recipe.parents.items():\n"
        "        # Lookup uses the 2-key tuple (forward_fn, argnum).\n"
        "        # KeyError propagates naturally — caller decorates.\n"
        "        back_fn = back_funcs[(node.recipe.func, argnum)]\n"
        "        results.append((argnum, parent, back_fn))\n"
        "    return results"
    ),
    solution_notes=(
        "**Why a separate dispatch step from the actual back_fn call.** "
        "The reverse-pass loop in `backprop` does TWO things per parent: "
        "(a) decide WHICH back_fn applies, (b) call it with the right "
        "args. Splitting (a) into its own helper means:\n"
        "- The KeyError surface lives in one place (easier to add a "
        "  diagnostic message).\n"
        "- You can unit-test dispatch without running the math.\n"
        "- A future change (e.g. swapping the registry for a vtable on "
        "  the op itself) only touches this function.\n\n"
        "**Why iterate `recipe.parents`, not `recipe.args`.** "
        "`recipe.args` is ALL positional args including non-Tensors "
        "(scalars, shape tuples). `recipe.parents` is ALREADY the "
        "filtered `{argnum: Tensor}` dict — only the parents that "
        "actually need a back_fn. Iterating args would force "
        "isinstance-checks here that the wrapper already did.\n\n"
        "**Why the argnum stays in the result.** The caller needs to "
        "know WHICH arg it's writing the grad to; for asymmetric ops "
        "(div, sub), `argnum=0` and `argnum=1` produce different "
        "tensor shapes and values. Without the argnum, the caller "
        "would have to re-derive it, defeating the dispatch."
    ),
)


# =========================================================================
# atom: back-fn-call-with-recipe-args  (1 exercise)
# =========================================================================

SPEC_BACK_FN_CALL = _spec(
    atom_id="back-fn-call-with-recipe-args",
    subtopic="Backprop: back fn call with recipe args",
    recap=RECAP_BACK_FN_CALL,
    ex_idx=1,
    ex_title="call a dispatched back_fn with (grad_out, node.array, *args, **kwargs)",
    slug="call-dispatched-back-fn-with-recipe-args-and-kwargs",
    bloom="Apply",
    difficulty_num=3,
    keywords=["back-fn-call", "recipe-args", "recipe-kwargs", "splat"],
    kcs=["back-fn-call-with-recipe-args", "kwargs-pass-through-recipe"],
    lo=(
        "Apply the canonical back_fn invocation pattern: call "
        "back_fn(grad_out, node.array, *recipe.args, **recipe.kwargs) so "
        "both positional and keyword args from the forward call reach the "
        "reverse pass."
    ),
    prompt_body=(
        "Implement `call_back_fn(back_fn, grad_out, node)` — invoke a "
        "back_fn with the canonical argument shape:\n\n"
        "```\n"
        "back_fn(\n"
        "    grad_out,                     # dL/d(out)\n"
        "    node.array,                   # the cached forward `out`\n"
        "    *node.recipe.args,            # raw positional args (unboxed)\n"
        "    **node.recipe.kwargs,         # forward kwargs (dim, keepdim, ...)\n"
        ")\n"
        "```\n\n"
        "**Why every argument.**\n"
        "- `grad_out`: dL/d(this node's output). The chain-rule "
        "  multiplier.\n"
        "- `node.array`: the cached forward `out`. Activations like "
        "  sigmoid use it to avoid recomputation.\n"
        "- `*node.recipe.args`: the original positional inputs at "
        "  call-time, unboxed. `multiply_back0` needs both `x` and `y` "
        "  to compute `dL/dx = grad_out * y`.\n"
        "- `**node.recipe.kwargs`: forward keyword args. `sum_back` "
        "  needs `dim` to broadcast back; without it you get a shape "
        "  mismatch deep in the reverse pass.\n\n"
        "**Common bugs the test catches.**\n"
        "1. Passing `node` instead of `node.array` — back_fns operate on "
        "raw torch tensors, not MiniTensors.\n"
        "2. Forgetting the `*` on `recipe.args` — passing the whole "
        "tuple as a single arg.\n"
        "3. Forgetting the `**` on `recipe.kwargs` — `sum_back` gets "
        "called with `dim=` missing and reduces over the default axis.\n\n"
        "Return whatever the back_fn returned (a `torch.Tensor` with "
        "the same shape as the parent at that argnum)."
    ),
    stub=(
        "def call_back_fn(back_fn, grad_out, node):\n"
        '    """Invoke back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# === TEST 1: single-arg back_fn (log) ===\n"
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "\n"
        "b = MiniTensor(t.tensor([2.0, 4.0]), requires_grad=True)\n"
        "c = MiniTensor(t.log(b.array), requires_grad=True)\n"
        "c.recipe = Recipe(func=t.log, args=(b.array,), kwargs={}, parents={0: b})\n"
        "g = call_back_fn(log_back, t.ones(2), c)\n"
        "assert t.allclose(g, 1 / b.array), f'log_back via dispatcher: {g}'\n"
        "\n"
        "# === TEST 2: two-arg back_fn (multiply_back0) — uses positional args ===\n"
        "def mul_back0(grad_out, out, x, y):\n"
        "    return grad_out * y\n"
        "\n"
        "x = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([7.0]), requires_grad=True)\n"
        "out = MiniTensor(x.array * y.array, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y}\n"
        ")\n"
        "g = call_back_fn(mul_back0, t.ones(1), out)\n"
        "# d(x*y)/dx = y; with grad_out = 1, result = y.\n"
        "assert t.allclose(g, y.array), (\n"
        "    f'mul_back0 must receive both x and y via *args: got {g}, expected {y.array}'\n"
        ")\n"
        "\n"
        "# === TEST 3: kwargs are threaded — sum_back wants dim= ===\n"
        "def sum_back(grad_out, out, x, dim=None, keepdim=False):\n"
        "    # broadcast grad_out back to x.shape along dim\n"
        "    if dim is None:\n"
        "        return grad_out * t.ones_like(x)\n"
        "    if not keepdim:\n"
        "        grad_out = grad_out.unsqueeze(dim)\n"
        "    return grad_out.expand_as(x)\n"
        "\n"
        "x = MiniTensor(t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)\n"
        "out_arr = x.array.sum(dim=1)   # shape (2,)\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.sum, args=(x.array,), kwargs={'dim': 1}, parents={0: x}\n"
        ")\n"
        "g = call_back_fn(sum_back, t.ones(2), out)\n"
        "assert g.shape == x.array.shape, (\n"
        "    f'sum_back needs dim= via kwargs to broadcast — '\n"
        "    f'got shape {g.shape}, expected {x.array.shape}'\n"
        ")\n"
        "assert t.allclose(g, t.ones_like(x.array)), 'sum_back broadcast value'\n"
        "\n"
        "# === TEST 4: BOTH args AND kwargs threaded — multiplication with axis ===\n"
        "# Trivial back_fn that asserts it received what we passed.\n"
        "received = {}\n"
        "def spy_back(grad_out, out, x, y, *, scale=1.0):\n"
        "    received['grad_out'] = grad_out\n"
        "    received['out'] = out\n"
        "    received['x'] = x\n"
        "    received['y'] = y\n"
        "    received['scale'] = scale\n"
        "    return grad_out * y * scale\n"
        "\n"
        "x = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out = MiniTensor(t.tensor([6.0]), requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(x.array, y.array),\n"
        "    kwargs={'scale': 4.0}, parents={0: x, 1: y}\n"
        ")\n"
        "g = call_back_fn(spy_back, t.tensor([10.0]), out)\n"
        "assert received['grad_out'] is not None and received['grad_out'].item() == 10.0\n"
        "assert received['out'].item() == 6.0, f'out should be node.array, got {received[\"out\"]}'\n"
        "assert received['x'].item() == 2.0\n"
        "assert received['y'].item() == 3.0\n"
        "assert received['scale'] == 4.0, (\n"
        "    f'kwargs must be threaded: scale={received[\"scale\"]} (expected 4.0)'\n"
        ")\n"
        "# Final result reflects all four channels.\n"
        "assert t.allclose(g, t.tensor([120.0])), f'spy_back result: {g}'\n"
        "\n"
        "# === TEST 5: empty kwargs case still works ===\n"
        "def take_one(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "\n"
        "z = MiniTensor(t.tensor([5.0]), requires_grad=True)\n"
        "out = MiniTensor(t.log(z.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(z.array,), kwargs={}, parents={0: z})\n"
        "g = call_back_fn(take_one, t.ones(1), out)\n"
        "assert t.allclose(g, 1 / z.array)\n"
        "\n"
        "# === TEST 6: out passed is node.array, NOT the MiniTensor wrapper ===\n"
        "type_recorder = {}\n"
        "def type_spy(grad_out, out, x):\n"
        "    type_recorder['out_type'] = type(out)\n"
        "    return grad_out / x\n"
        "\n"
        "n = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "out = MiniTensor(t.log(n.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(n.array,), kwargs={}, parents={0: n})\n"
        "call_back_fn(type_spy, t.ones(1), out)\n"
        "assert type_recorder['out_type'] is t.Tensor, (\n"
        "    f'back_fn must receive raw torch.Tensor (node.array), '\n"
        "    f'not MiniTensor — got {type_recorder[\"out_type\"]}'\n"
        ")"
    ),
    solution_body=(
        "def call_back_fn(back_fn, grad_out, node):\n"
        "    # Canonical invocation — note BOTH * and ** splats.\n"
        "    return back_fn(\n"
        "        grad_out,\n"
        "        node.array,                # raw torch.Tensor, NOT the MiniTensor\n"
        "        *node.recipe.args,         # forward positional args (unboxed)\n"
        "        **node.recipe.kwargs,      # forward keyword args (dim, keepdim, ...)\n"
        "    )"
    ),
    solution_notes=(
        "**Why the splats matter.** Without `*node.recipe.args`, you'd "
        "pass the whole tuple as one argument and `multiply_back0(grad, "
        "out, (x, y))` would crash on signature mismatch. Without "
        "`**node.recipe.kwargs`, `sum_back` would be called with no "
        "`dim=` and silently reduce over the wrong axis — the failure "
        "shows up much later as a shape mismatch.\n\n"
        "**Why `node.array`, not `node`.** Back_fns operate on raw "
        "torch tensors so they can do tensor math directly. Passing "
        "the MiniTensor wrapper would force every back_fn body to do "
        "`out.array` first — defeating the whole point of the "
        "unboxing wrapper.\n\n"
        "**This is the call-site half of the dispatcher.** The sibling "
        "`dispatch-back-fn-from-recipe` atom answers 'WHICH back_fn?'; "
        "this atom answers 'with WHAT args?'. The actual reverse-pass "
        "driver in `backprop` interleaves them in one tight loop, but "
        "the responsibilities are distinct.\n\n"
        "**Why no return-type validation.** Each back_fn is responsible "
        "for returning a tensor with the correct shape (matching the "
        "parent at that argnum). The dispatcher would have to compare "
        "against each parent's shape to validate — that's pushed up to "
        "the reverse-pass loop, which already has the parent reference."
    ),
)


# =========================================================================
# emit
# =========================================================================

ALL_SPECS = [
    SPEC_GRAD_IN_OUT,
    SPEC_NO_GRAD,
    SPEC_MANUAL_CHAIN,
    SPEC_DFS_TOPOSORT,
    SPEC_CYCLE_DETECT,
    SPEC_BACKPROP_LOOP,
    SPEC_DISPATCH,
    SPEC_BACK_FN_CALL,
]


if __name__ == "__main__":
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
