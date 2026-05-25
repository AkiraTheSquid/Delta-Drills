#!/usr/bin/env python3
"""Author 8 ex2 deepening drills for ARENA chap-0 backprop-DRIVER atoms.

Each ex2 probes a DISTINCT facet from the existing ex1 in the same atom —
a different cognitive operation or surface context, but the same single LO
and ≤2 KCs. Shares the MiniTensor + Recipe preamble with batch-6.

  * grad-expressed-in-out         — ex2: tanh_back via (1 - out**2)
  * no-grad-context-mgr-update    — ex2: companion EnableGrad ctx-mgr
  * manual-chain-forward-and-back — ex2: 3-step chain log -> square -> exp
  * dfs-three-set-toposort        — ex2: iterative (stack-based) DFS toposort
  * cycle-detection-temp-set      — ex2: return the cycle PATH for debugging
  * backprop-pop-outgrad-loop     — ex2: per-leaf accumulation counter
  * dispatch-back-fn-from-recipe  — ex2: friendly KeyError with op+argnum
  * back-fn-call-with-recipe-args — ex2: detect splat-bugs via spy back_fn

Verification re-execs preamble + stub + solution + test_body in a fresh
namespace per spec, asserts pass, BEFORE any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_backprop_driver"


# Shared autograd preamble — matches batch6, gives every drill access to
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


# ---------------------------------------------------------------- atom recaps

RECAP_GRAD_EXPRESSED_IN_OUT = (
    "## Grad expressed in `out` — quick refresher\n"
    "\n"
    "Some elementwise back fns can be written purely in terms of the CACHED "
    "forward `out`, avoiding a second activation call:\n"
    "\n"
    "```\n"
    "sigmoid_back(grad_out, out, x) = grad_out * out * (1 - out)\n"
    "tanh_back   (grad_out, out, x) = grad_out * (1 - out**2)\n"
    "exp_back    (grad_out, out, x) = grad_out * out\n"
    "```\n"
    "\n"
    "The `(grad_out, out, x)` signature exists exactly so any of these "
    "back_fns can pick the cheapest cache. For `tanh`, the identity "
    "`d/dx tanh(x) = 1 - tanh(x)**2 = 1 - out**2` is the key — no second "
    "`t.tanh(x)` call, no `cosh`, no division. Just a square and a subtract."
)

RECAP_NO_GRAD_CTX = (
    "## `no_grad` / `enable_grad` context managers — quick refresher\n"
    "\n"
    "Two complementary save-and-restore managers around the module-level "
    "`grad_tracking_enabled` flag:\n"
    "\n"
    "```python\n"
    "class NoGrad:      # disables grad inside the block\n"
    "    def __enter__(self):\n"
    "        global grad_tracking_enabled\n"
    "        self._prev = grad_tracking_enabled\n"
    "        grad_tracking_enabled = False\n"
    "    def __exit__(self, *exc):\n"
    "        global grad_tracking_enabled\n"
    "        grad_tracking_enabled = self._prev\n"
    "\n"
    "class EnableGrad:  # re-enables grad inside the block (cancels outer NoGrad)\n"
    "    def __enter__(self):\n"
    "        global grad_tracking_enabled\n"
    "        self._prev = grad_tracking_enabled\n"
    "        grad_tracking_enabled = True\n"
    "    def __exit__(self, *exc):\n"
    "        global grad_tracking_enabled\n"
    "        grad_tracking_enabled = self._prev\n"
    "```\n"
    "\n"
    "Both save the PREVIOUS value (so nesting works) and restore on exit "
    "(even if the block raises). The only difference is the value they "
    "write on `__enter__`. `EnableGrad` is what you wrap a backward call "
    "in when you're inside `torch.no_grad()` but still need an inner "
    "gradient (e.g. higher-order grad inside an inference loop)."
)

RECAP_MANUAL_CHAIN = (
    "## Manual forward-and-back chain — quick refresher\n"
    "\n"
    "The pattern at any chain length: compute forward in call order, then "
    "compute backward in REVERSE call order, threading `grad_out` through "
    "each back_fn. For a 3-step chain `a → b → c → d`:\n"
    "\n"
    "```\n"
    "# Forward:    a -> b = log(a) -> c = b**2 -> d = exp(c)\n"
    "b = log(a)\n"
    "c = b ** 2\n"
    "d = exp(c)\n"
    "\n"
    "# Backward (given dL/dd, want dL/da):\n"
    "dL_dc = exp_back(dL_dd, d, c)       # uses cached out (d)\n"
    "dL_db = square_back(dL_dc, c, b)    # uses input b: d/db b**2 = 2*b\n"
    "dL_da = log_back(dL_db, b, a)       # uses input a: d/da log(a) = 1/a\n"
    "```\n"
    "\n"
    "Two invariants that hold at any length:\n"
    "- **Reverse the call order.** Last forward op = first backward op.\n"
    "- **Each back_fn gets `(grad_out, cached_out_at_that_node, input)`.** "
    "Which one of `out`/`input` it actually USES depends on the op."
)

RECAP_DFS_TOPOSORT = (
    "## DFS three-set toposort — iterative form\n"
    "\n"
    "The recursive three-color DFS works great until you hit Python's "
    "default recursion limit (1000) on a deep computational graph. The "
    "iterative form uses an explicit stack and two-phase node entries:\n"
    "\n"
    "```python\n"
    "def topological_sort(root, get_children):\n"
    "    result = []\n"
    "    perm = set()\n"
    "    temp = set()\n"
    "    # Stack frames: (node, 'enter') or (node, 'exit').\n"
    "    stack = [(root, 'enter')]\n"
    "    while stack:\n"
    "        node, phase = stack.pop()\n"
    "        nid = id(node)\n"
    "        if phase == 'exit':\n"
    "            temp.discard(nid); perm.add(nid); result.append(node)\n"
    "            continue\n"
    "        if nid in perm: continue\n"
    "        if nid in temp: raise ValueError('cycle')\n"
    "        temp.add(nid)\n"
    "        stack.append((node, 'exit'))     # post-order hook\n"
    "        for child in get_children(node):\n"
    "            stack.append((child, 'enter'))\n"
    "    return result\n"
    "```\n"
    "\n"
    "Same deps-first output as the recursive version (root LAST). The "
    "two-phase trick — push an `exit` marker BEFORE the children — gives "
    "us the post-order moment for `result.append` without recursion."
)

RECAP_CYCLE_DETECTION = (
    "## Cycle PATH detection — quick refresher\n"
    "\n"
    "A boolean `has_cycle` tells you THAT a cycle exists. For debugging "
    "(or for producing actionable error messages), you usually want the "
    "ACTUAL nodes on the cycle. The same temp-set DFS can capture them:\n"
    "\n"
    "```python\n"
    "def find_cycle(root, get_children):\n"
    "    path = []          # nodes currently on the DFS stack, in order\n"
    "    on_stack = set()   # ids of those nodes (O(1) lookup)\n"
    "    perm = set()\n"
    "\n"
    "    def visit(node):\n"
    "        nid = id(node)\n"
    "        if nid in perm: return None\n"
    "        if nid in on_stack:\n"
    "            # back-edge — slice path from first appearance + this node\n"
    "            i = next(j for j, n in enumerate(path) if id(n) == nid)\n"
    "            return path[i:] + [node]\n"
    "        on_stack.add(nid); path.append(node)\n"
    "        for child in get_children(node):\n"
    "            cyc = visit(child)\n"
    "            if cyc is not None: return cyc\n"
    "        on_stack.discard(nid); path.pop()\n"
    "        perm.add(nid)\n"
    "        return None\n"
    "    return visit(root)\n"
    "```\n"
    "\n"
    "Returns `None` for a DAG; for a cycle, returns the list of nodes "
    "around the loop (closing node repeated at the end). The cost is one "
    "extra `path` list — `O(depth)` memory — and a `next()` slice once "
    "per cycle hit."
)

RECAP_BACKPROP_LOOP = (
    "## Backprop loop + per-leaf accumulation counts — quick refresher\n"
    "\n"
    "Same reverse-pass driver, instrumented: alongside the normal "
    "`.grad` accumulation, keep a parallel `{id(leaf): count}` of how "
    "many gradient contributions each leaf received. Each `+=` "
    "increments the count.\n"
    "\n"
    "```python\n"
    "counts = {}\n"
    "for node in sorted_graph:\n"
    "    grad_out = grads.pop(id(node))\n"
    "    if node.recipe is None:\n"
    "        node.grad = grad_out if node.grad is None else node.grad + grad_out\n"
    "        counts[id(node)] = counts.get(id(node), 0) + 1   # <-- count++\n"
    "        continue\n"
    "    for argnum, parent in node.recipe.parents.items():\n"
    "        back_fn = back_funcs[(node.recipe.func, argnum)]\n"
    "        grad_parent = back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)\n"
    "        grads[id(parent)] = grads.get(id(parent), 0) + grad_parent\n"
    "```\n"
    "\n"
    "**Use case.** A diamond DAG (z appearing at parents={0: z, 1: z} of "
    "`z * z`) should yield `counts[id(z)] == 1` — z is a leaf, it gets "
    "ONE merged grad after both paths feed in. A leaf consumed by `k` "
    "separate downstream operations gets `counts[id(leaf)] == k`. "
    "Counts diagnose where grad routed."
)

RECAP_DISPATCH_BACK_FN = (
    "## Dispatch back_fn with friendly KeyError — quick refresher\n"
    "\n"
    "When `(recipe.func, argnum)` isn't in the registry, the raw "
    "`back_funcs[key]` lookup raises a `KeyError` that prints the tuple "
    "but tells you nothing about WHICH op or argnum is unregistered. "
    "Wrap the dispatch with a diagnostic:\n"
    "\n"
    "```python\n"
    "def dispatch(node, back_funcs):\n"
    "    out = []\n"
    "    for argnum, parent in node.recipe.parents.items():\n"
    "        key = (node.recipe.func, argnum)\n"
    "        if key not in back_funcs:\n"
    "            fn_name = getattr(node.recipe.func, '__name__', repr(node.recipe.func))\n"
    "            raise KeyError(\n"
    "                f'No back_fn registered for ({fn_name}, argnum={argnum}). '\n"
    "                f'Add it to BACK_FUNCS via register_back_func({fn_name}, {argnum}, ...).'\n"
    "            )\n"
    "        out.append((argnum, parent, back_funcs[key]))\n"
    "    return out\n"
    "```\n"
    "\n"
    "The naming pattern matches PyTorch's autograd: error messages that "
    "tell you exactly what was missing AND how to fix it — not just "
    "`KeyError: (<built-in function sin>, 0)`."
)

RECAP_BACK_FN_CALL = (
    "## Back_fn call channels — splat-bug spy — quick refresher\n"
    "\n"
    "The canonical call:\n"
    "```\n"
    "back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)\n"
    "```\n"
    "has FOUR channels, each of which can be silently miswired:\n"
    "\n"
    "1. `grad_out` — dL/d(out); missing it = no chain-rule multiplier.\n"
    "2. `node.array` — cached forward `out`; passing `node` (the wrapper) "
    "instead breaks back_fns that do tensor math on `out`.\n"
    "3. `*recipe.args` — forgetting the `*` passes the whole tuple as "
    "a single argument; back_fns crash on signature mismatch.\n"
    "4. `**recipe.kwargs` — forgetting the `**` drops `dim`/`keepdim`/"
    "etc.; reductions silently use the wrong axis.\n"
    "\n"
    "A **spy back_fn** that records every received argument lets you "
    "diagnose which channel went wrong — recording (grad_out, out, args, "
    "kwargs) and comparing against expected, an authoring helper can "
    "report 'kwargs={} but recipe.kwargs={'dim': 1}' — pinpointing the "
    "missing `**` splat."
)


# ---------------------------------------------------------------- spec helper


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
    keywords: list,
    kcs: list,
    lo: str,
    prompt_body: str,
    stub: str,
    test_body: str,
    solution_body: str,
    solution_notes: str = "",
    extra_imports: list | None = None,
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
# atom: grad-expressed-in-out — ex2: tanh_back via (1 - out**2)
# ex1 wrote sigmoid_back; ex2 transfers the same pattern to tanh.
# =========================================================================

SPEC_GRAD_IN_OUT = _spec(
    atom_id="grad-expressed-in-out",
    subtopic="Backprop: grad expressed in out",
    recap=RECAP_GRAD_EXPRESSED_IN_OUT,
    ex_idx=2,
    ex_title="write tanh_back using cached out via (1 - out**2)",
    slug="write-tanh-back-using-cached-out",
    bloom="Apply",
    difficulty_num=2,
    keywords=["tanh", "cached-out", "elementwise", "no-recompute"],
    kcs=["grad-expressed-in-out", "back-fn-uses-cached-out"],
    lo=(
        "Apply the 'grad expressed in out' pattern to a DIFFERENT activation "
        "by writing tanh_back as grad_out * (1 - out**2), reusing the cached "
        "forward output rather than recomputing tanh(x) or any cosh."
    ),
    prompt_body=(
        "Implement `tanh_back(grad_out, out, x)` using only the cached `out` "
        "— no `t.tanh(x)` recompute, no `cosh`.\n\n"
        "**Math.** `out = tanh(x)`. The derivative collapses neatly:\n\n"
        "```\n"
        "d/dx tanh(x) = 1 - tanh(x)**2\n"
        "             = 1 - out**2\n"
        "```\n\n"
        "So by the chain rule:\n\n"
        "```\n"
        "dL/dx = grad_out * (1 - out**2)\n"
        "```\n\n"
        "**Point of this drill.** It's the same SHAPE as `sigmoid_back` "
        "but a different activation — you're TRANSFERRING the cached-out "
        "pattern, not re-inventing it. One line should do it. The test "
        "feeds a deliberately WRONG `out` to catch any recompute-from-x.\n\n"
        "**Inputs.** Plain `torch.Tensor`, same shape; float dtype. "
        "Output: tensor with the same shape as `x`."
    ),
    stub=(
        "def tanh_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = tanh(x), expressed via the cached `out`."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- numerical correctness ---\n"
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])\n"
        "out = t.tanh(x)\n"
        "grad_out = t.ones(5)\n"
        "g = tanh_back(grad_out, out, x)\n"
        "expected = 1 - out**2\n"
        "assert g.shape == x.shape\n"
        "assert t.allclose(g, expected, atol=1e-6), f'value: {g} vs {expected}'\n"
        "\n"
        "# --- non-unit grad_out scales each entry ---\n"
        "grad_out = t.tensor([5.0, -3.0, 2.0, 0.5, -1.0])\n"
        "g = tanh_back(grad_out, out, x)\n"
        "expected = grad_out * (1 - out**2)\n"
        "assert t.allclose(g, expected, atol=1e-6), 'chain-rule scaling failed'\n"
        "\n"
        "# --- matrix shape ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "out_mat = t.tanh(X)\n"
        "g_mat = tanh_back(G, out_mat, X)\n"
        "assert g_mat.shape == (3, 4)\n"
        "assert t.allclose(g_mat, G * (1 - out_mat**2), atol=1e-6)\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([-1.5, -0.2, 0.3, 1.5], requires_grad=True)\n"
        "y = t.tanh(x_ref).sum()\n"
        "y.backward()\n"
        "out_cached = t.tanh(x_ref.detach())\n"
        "g_ours = tanh_back(t.ones(4), out_cached, x_ref.detach())\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'tanh_back disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")\n"
        "\n"
        "# --- THE point: must use `out`, NOT recompute tanh(x) ---\n"
        "# Pass a deliberately wrong `out`; a correct impl uses it, a recompute\n"
        "# impl would ignore it and return the true derivative from x.\n"
        "fake_x = t.tensor([0.0, 0.0, 0.0])\n"
        "fake_out = t.tensor([0.3, 0.5, 0.8])   # NOT tanh(0) (= 0)\n"
        "got = tanh_back(t.ones(3), fake_out, fake_x)\n"
        "expected_from_fake = 1 - fake_out**2\n"
        "assert t.allclose(got, expected_from_fake, atol=1e-6), (\n"
        "    'tanh_back must use cached `out`, not recompute tanh(x). '\n"
        "    f'fake_out={fake_out.tolist()} should give {expected_from_fake.tolist()}; '\n"
        "    f'got {got.tolist()}.'\n"
        ")\n"
        "\n"
        "# --- scalar case: at x=0, tanh=0, derivative = 1 - 0 = 1 ---\n"
        "x_sc = t.tensor(0.0)\n"
        "out_sc = t.tanh(x_sc)\n"
        "g_sc = tanh_back(t.tensor(1.0), out_sc, x_sc)\n"
        "assert g_sc.shape == x_sc.shape\n"
        "assert abs(g_sc.item() - 1.0) < 1e-6, f'scalar at x=0: {g_sc}'"
    ),
    solution_body=(
        "def tanh_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d/dx tanh(x) = 1 - tanh(x)**2 = 1 - out**2.\n"
        "    return grad_out * (1 - out**2)"
    ),
    solution_notes=(
        "**Same shape, different op.** Like sigmoid_back, every input to "
        "tanh_back is already cached — you just rearrange them. Watch the "
        "table of activations sharing this pattern: sigmoid uses `out * "
        "(1 - out)`; tanh uses `1 - out**2`; exp uses `out` directly. "
        "Each is a one-liner once you remember the identity.\n\n"
        "**Why not `1 - t.tanh(x)**2`.** Bit-for-bit, with float math, "
        "`1 - t.tanh(x)**2 != 1 - out**2` if `out` and `t.tanh(x)` were "
        "computed with different rounding modes (rare, but possible on "
        "GPU). More importantly: a second tanh call is one more kernel "
        "launch per layer per backward — wasteful at scale."
    ),
)


# =========================================================================
# atom: no-grad-context-mgr-update — ex2: companion EnableGrad ctx-mgr
# ex1 wrote NoGrad; ex2 writes EnableGrad — same pattern, opposite write,
# used to re-enable grad inside an outer NoGrad block.
# =========================================================================

SPEC_NO_GRAD = _spec(
    atom_id="no-grad-context-mgr-update",
    subtopic="Backprop: no_grad ctx-mgr update",
    recap=RECAP_NO_GRAD_CTX,
    ex_idx=2,
    ex_title="implement companion EnableGrad context manager (re-enable inside NoGrad)",
    slug="implement-enable-grad-companion-context-mgr",
    bloom="Apply",
    difficulty_num=2,
    keywords=["context-manager", "enable-grad", "nested", "save-restore"],
    kcs=["no-grad-context-mgr-update", "grad-tracking-global-toggle"],
    lo=(
        "Apply the save-and-restore context-manager pattern to write "
        "EnableGrad — the companion of NoGrad that re-enables "
        "grad_tracking_enabled inside its block and restores the previous "
        "value on exit, so it can cancel an outer NoGrad temporarily."
    ),
    prompt_body=(
        "Implement `EnableGrad` — the companion context manager to "
        "`NoGrad`. It SETS `grad_tracking_enabled = True` on enter (the "
        "opposite write from `NoGrad`), saves whatever the flag was "
        "BEFORE, and restores it on exit:\n\n"
        "```python\n"
        "with NoGrad():\n"
        "    # grad disabled\n"
        "    with EnableGrad():\n"
        "        # grad RE-ENABLED here (cancels outer NoGrad)\n"
        "        ...\n"
        "    # back to disabled (outer NoGrad's previous state)\n"
        "```\n\n"
        "**Use case.** You're inside a long inference path that disabled "
        "grad globally, but you need ONE inner block to compute a "
        "gradient (e.g. computing a Hessian-vector product for a small "
        "second-order optimizer step). `EnableGrad` cancels the outer "
        "`NoGrad` for the duration of its `with` block.\n\n"
        "**Requirements.**\n"
        "- `__enter__` saves the previous value of `grad_tracking_enabled` "
        "and sets it to `True`.\n"
        "- `__exit__` restores the saved previous value — NOT hard-coded "
        "`False`.\n"
        "- Works nested inside `NoGrad`. Works nested inside another "
        "`EnableGrad`. Works on its own.\n"
        "- Restores on exit even if the block raised.\n\n"
        "`grad_tracking_enabled` is a **module-level** name — use `global` "
        "to write to it. A `NoGrad` reference implementation is already "
        "in scope from the preamble for the nesting tests; if you want "
        "to use it directly, the test re-defines it as a fresh class so "
        "the two managers compose cleanly."
    ),
    stub=(
        "class EnableGrad:\n"
        '    """Re-enable grad_tracking_enabled for the duration of a `with` block.\n'
        '\n'
        '    Saves the previous value on enter (so nesting / cancelling NoGrad works),\n'
        '    restores it on exit.\n'
        '    """\n'
        "    def __enter__(self):\n"
        "        raise NotImplementedError()\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        raise NotImplementedError()"
    ),
    test_body=(
        "# --- give the test a fresh NoGrad reference implementation so the\n"
        "#     two managers compose without relying on any specific outer state ---\n"
        "class NoGrad:\n"
        "    def __enter__(self):\n"
        "        global grad_tracking_enabled\n"
        "        self._prev = grad_tracking_enabled\n"
        "        grad_tracking_enabled = False\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        global grad_tracking_enabled\n"
        "        grad_tracking_enabled = self._prev\n"
        "\n"
        "# --- baseline: starts True from the preamble ---\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "assert grad_tracking_enabled is True\n"
        "\n"
        "# --- when grad already True, EnableGrad keeps it True; exit restores True ---\n"
        "with EnableGrad():\n"
        "    assert grad_tracking_enabled is True, (\n"
        "        'EnableGrad must set grad_tracking_enabled True'\n"
        "    )\n"
        "assert grad_tracking_enabled is True\n"
        "\n"
        "# --- THE main use case: cancel an outer NoGrad ---\n"
        "with NoGrad():\n"
        "    assert grad_tracking_enabled is False, 'outer NoGrad disabled'\n"
        "    with EnableGrad():\n"
        "        assert grad_tracking_enabled is True, (\n"
        "            'EnableGrad must cancel outer NoGrad inside its block'\n"
        "        )\n"
        "    # back outside the inner EnableGrad — must RESTORE the previous False\n"
        "    assert grad_tracking_enabled is False, (\n"
        "        'EnableGrad exit must restore the previous value (False), '\n"
        "        'NOT hard-code True or False'\n"
        "    )\n"
        "assert grad_tracking_enabled is True, 'outer NoGrad exit restores baseline True'\n"
        "\n"
        "# --- nesting two EnableGrads inside NoGrad ---\n"
        "with NoGrad():\n"
        "    with EnableGrad():\n"
        "        with EnableGrad():\n"
        "            assert grad_tracking_enabled is True\n"
        "        assert grad_tracking_enabled is True, 'still enabled (outer EnableGrad)'\n"
        "    assert grad_tracking_enabled is False, 'outermost NoGrad context'\n"
        "assert grad_tracking_enabled is True\n"
        "\n"
        "# --- restore on exception ---\n"
        "globals()['grad_tracking_enabled'] = False   # caller had disabled grad\n"
        "try:\n"
        "    with EnableGrad():\n"
        "        assert grad_tracking_enabled is True\n"
        "        raise RuntimeError('boom')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "assert grad_tracking_enabled is False, (\n"
        "    'EnableGrad must restore the previous value even if the block raised'\n"
        ")\n"
        "globals()['grad_tracking_enabled'] = True   # reset for any downstream\n"
        "\n"
        "# --- EnableGrad alone (no surrounding NoGrad) is a no-op on entry/exit value ---\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "with EnableGrad():\n"
        "    assert grad_tracking_enabled is True\n"
        "assert grad_tracking_enabled is True\n"
        "\n"
        "# --- canonical inner-grad use case: in-place tensor op INSIDE EnableGrad\n"
        "#     under an outer NoGrad still works and grad stays enabled ---\n"
        "scratch = t.tensor([1.0, 2.0])\n"
        "with NoGrad():\n"
        "    with EnableGrad():\n"
        "        # grad is enabled here — a hypothetical wrap_forward_fn would\n"
        "        # attach a Recipe; we just check the flag is right.\n"
        "        scratch = scratch * 2\n"
        "        assert grad_tracking_enabled is True\n"
        "    assert grad_tracking_enabled is False\n"
        "assert grad_tracking_enabled is True\n"
        "assert t.allclose(scratch, t.tensor([2.0, 4.0]))"
    ),
    solution_body=(
        "class EnableGrad:\n"
        "    def __enter__(self):\n"
        "        global grad_tracking_enabled\n"
        "        self._prev = grad_tracking_enabled   # stash so nesting works\n"
        "        grad_tracking_enabled = True\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
        "        global grad_tracking_enabled\n"
        "        grad_tracking_enabled = self._prev   # restore PREVIOUS value\n"
        "        # implicit return None — exceptions propagate normally."
    ),
    solution_notes=(
        "**Symmetry with NoGrad.** Same template, opposite write: NoGrad "
        "sets `False`, EnableGrad sets `True`. Both stash `self._prev` to "
        "support arbitrary nesting. This symmetry is the same one PyTorch "
        "exposes as `torch.no_grad()` and `torch.enable_grad()` — the "
        "latter exists precisely for the 'cancel an outer no_grad' case "
        "demonstrated in the test.\n\n"
        "**Why no try/finally.** The `__exit__` method is called by the "
        "interpreter regardless of whether the body raised; explicit "
        "try/finally would be redundant. Returning `None` (or anything "
        "falsy) tells Python to propagate the exception.\n\n"
        "**Cancelling vs over-riding.** The convention is that a child "
        "`EnableGrad` OVERRIDES its parent for the duration, then "
        "restores the parent's state. This matches what users expect "
        "from any `with`-based override (volume control, log level, "
        "verbosity flag) — the inner block is a temporary patch."
    ),
)


# =========================================================================
# atom: manual-chain-forward-and-back — ex2: 3-step chain log → square → exp
# ex1 was 2-step (log,exp); ex2 stretches to 3 steps with a square middle.
# =========================================================================

SPEC_MANUAL_CHAIN = _spec(
    atom_id="manual-chain-forward-and-back",
    subtopic="Backprop: manual chain forward-and-back",
    recap=RECAP_MANUAL_CHAIN,
    ex_idx=2,
    ex_title="manually chain forward log→square→exp and run backward by hand",
    slug="manually-chain-3-step-log-square-exp-backward",
    bloom="Apply",
    difficulty_num=3,
    keywords=["manual-chain", "three-step", "square", "log", "exp", "reverse-order"],
    kcs=["manual-chain-forward-and-back", "back-fn-uses-cached-out"],
    lo=(
        "Apply the manual forward-then-backward chain pattern to a "
        "3-operation chain: forward a → b=log(a) → c=b**2 → d=exp(c), "
        "then run backward in reverse via exp_back, square_back, log_back."
    ),
    prompt_body=(
        "Implement `manual_chain_3(a, dL_dd)` — a hand-run forward+backward "
        "for a length-3 chain. This is the same pattern as ex1 (log+exp), "
        "stretched by one more op — a `square` middle.\n\n"
        "**Forward pass.**\n"
        "```\n"
        "b = log(a)\n"
        "c = b ** 2\n"
        "d = exp(c)\n"
        "```\n\n"
        "**Backward pass.** Given `dL/dd`, compute `dL/da` by running "
        "the chain in REVERSE:\n"
        "```\n"
        "dL_dc = exp_back(dL_dd, d, c)         # d/dx exp(x) = exp(x) = d\n"
        "dL_db = square_back(dL_dc, c, b)      # d/db b**2  = 2*b\n"
        "dL_da = log_back(dL_db, b, a)         # d/da log(a) = 1/a\n"
        "```\n\n"
        "where:\n"
        "- `exp_back(grad_out, out, x) = grad_out * out`\n"
        "- `square_back(grad_out, out, x) = grad_out * 2 * x`\n"
        "- `log_back(grad_out, out, x) = grad_out / x`\n\n"
        "**Return** the 6-tuple `(b, c, d, dL_dc, dL_db, dL_da)` so the "
        "test can inspect every intermediate.\n\n"
        "**Point of this drill.** The 2-step ex1 makes the reverse-order "
        "pattern visible; the 3-step ex2 confirms you can extend it "
        "MECHANICALLY. Each forward op picks up its mirror back_fn in "
        "the reverse pass. Assume `a > 0`."
    ),
    stub=(
        "def manual_chain_3(a: Tensor, dL_dd: Tensor) -> tuple:\n"
        '    """Forward: b=log(a), c=b**2, d=exp(c).\n'
        '    Backward: return (b, c, d, dL_dc, dL_db, dL_da)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import math as _math\n"
        "\n"
        "# --- forward values match the math ---\n"
        "a = t.tensor([_math.e, 2.0, 4.0])   # log(e) = 1, etc.\n"
        "dL_dd = t.ones(3)\n"
        "b, c, d, dL_dc, dL_db, dL_da = manual_chain_3(a, dL_dd)\n"
        "assert t.allclose(b, t.log(a), atol=1e-5), f'b={b}'\n"
        "assert t.allclose(c, t.log(a)**2, atol=1e-5), f'c={c}'\n"
        "assert t.allclose(d, t.exp(t.log(a)**2), atol=1e-5), f'd={d}'\n"
        "\n"
        "# --- backward shapes ---\n"
        "assert dL_dc.shape == a.shape\n"
        "assert dL_db.shape == a.shape\n"
        "assert dL_da.shape == a.shape\n"
        "\n"
        "# --- backward values: chain rule by hand ---\n"
        "# dL_dc = dL_dd * d\n"
        "expected_dL_dc = dL_dd * d\n"
        "assert t.allclose(dL_dc, expected_dL_dc, atol=1e-5), (\n"
        "    f'dL_dc wrong: got {dL_dc}, expected {expected_dL_dc}'\n"
        ")\n"
        "# dL_db = dL_dc * 2 * b\n"
        "expected_dL_db = dL_dc * 2 * b\n"
        "assert t.allclose(dL_db, expected_dL_db, atol=1e-5), (\n"
        "    f'dL_db wrong: got {dL_db}, expected {expected_dL_db}'\n"
        ")\n"
        "# dL_da = dL_db / a\n"
        "expected_dL_da = dL_db / a\n"
        "assert t.allclose(dL_da, expected_dL_da, atol=1e-5), (\n"
        "    f'dL_da wrong: got {dL_da}, expected {expected_dL_da}'\n"
        ")\n"
        "\n"
        "# --- witness against torch.autograd on the full 3-step chain ---\n"
        "a_ref = t.tensor([1.5, 2.5, 4.0], requires_grad=True)\n"
        "d_ref = t.exp(t.log(a_ref) ** 2)\n"
        "loss = (d_ref * t.tensor([0.5, -1.0, 2.0])).sum()\n"
        "loss.backward()\n"
        "_, _, _, _, _, dL_da_ours = manual_chain_3(\n"
        "    a_ref.detach(), t.tensor([0.5, -1.0, 2.0])\n"
        ")\n"
        "assert t.allclose(dL_da_ours, a_ref.grad, atol=1e-5), (\n"
        "    f'3-step chain disagrees with autograd: '\n"
        "    f'ours={dL_da_ours}, ref={a_ref.grad}'\n"
        ")\n"
        "\n"
        "# --- non-unit dL_dd scales linearly through the chain ---\n"
        "a = t.tensor([2.0, 3.0])\n"
        "_, _, _, dL_dc_a, dL_db_a, dL_da_a = manual_chain_3(a, t.ones(2))\n"
        "_, _, _, dL_dc_b, dL_db_b, dL_da_b = manual_chain_3(a, t.tensor([3.0, 3.0]))\n"
        "assert t.allclose(dL_da_b, 3 * dL_da_a, atol=1e-5), (\n"
        "    f'linearity in dL_dd should give 3x scaling: '\n"
        "    f'{dL_da_b} vs 3*{dL_da_a}'\n"
        ")\n"
        "\n"
        "# --- order check: dL_dc must be computed BEFORE dL_db, BEFORE dL_da ---\n"
        "# Smoke check via dependency: dL_db should equal dL_dc * 2 * b, NOT\n"
        "# dL_dd * 2 * b (forward order bug).\n"
        "a = t.tensor([2.0])\n"
        "_, _, _, dL_dc, dL_db, _ = manual_chain_3(a, t.tensor([1.0]))\n"
        "# d here is exp(log(2)**2) ~ 1.617; dL_dc = 1.617;\n"
        "# dL_db = 1.617 * 2 * log(2) ~ 2.243 (correct);\n"
        "# Forward-order bug would give: 1.0 * 2 * log(2) ~ 1.386 (wrong).\n"
        "assert dL_db.item() > 2.0, (\n"
        "    f'dL_db {dL_db} too small — did you skip dL_dc multiplier '\n"
        "    f'(forward-order bug)?'\n"
        ")"
    ),
    solution_body=(
        "def manual_chain_3(a: Tensor, dL_dd: Tensor) -> tuple:\n"
        "    # forward: a -> b -> c -> d\n"
        "    b = t.log(a)\n"
        "    c = b ** 2\n"
        "    d = t.exp(c)\n"
        "    # backward (reverse order): dL_dd -> dL_dc -> dL_db -> dL_da\n"
        "    dL_dc = dL_dd * d            # exp_back: uses cached d\n"
        "    dL_db = dL_dc * 2 * b        # square_back: d/db b**2 = 2*b\n"
        "    dL_da = dL_db / a            # log_back: d/da log(a) = 1/a\n"
        "    return b, c, d, dL_dc, dL_db, dL_da"
    ),
    solution_notes=(
        "**The pattern extends mechanically.** Every additional forward "
        "op picks up ONE additional back step in the reverse pass, "
        "threaded through `grad_out`. Once you've internalized this at "
        "length 3, length-n is the same thing in a loop — which is "
        "exactly what `backprop` will do over a sorted_graph.\n\n"
        "**Why `square_back` uses `x` (not `out`).** `d/dx x**2 = 2x` "
        "— no relationship to `out = x**2`, so we must read `x`. "
        "Compare with `exp_back` (uses `out` because `d/dx exp(x) = "
        "out` by identity). The (`grad_out`, `out`, `x`) signature "
        "always passes both, leaving the choice to each back_fn.\n\n"
        "**Composite identity sanity.** `d = exp((log a)**2)`, so\n"
        "`d(log d)/d(log a) = 2 * log(a)` — a clean closed form that "
        "the autograd witness test confirms numerically."
    ),
)


# =========================================================================
# atom: dfs-three-set-toposort — ex2: iterative (stack-based) toposort
# ex1 was recursive 3-color DFS; ex2 swaps recursion for an explicit stack
# (avoids Python's 1000-frame default recursion limit on deep graphs).
# =========================================================================

SPEC_DFS_TOPOSORT = _spec(
    atom_id="dfs-three-set-toposort",
    subtopic="Backprop: DFS three-set toposort",
    recap=RECAP_DFS_TOPOSORT,
    ex_idx=2,
    ex_title="implement iterative (stack-based) three-set DFS toposort",
    slug="implement-iterative-stack-based-three-set-toposort",
    bloom="Apply",
    difficulty_num=4,
    keywords=["dfs", "topological-sort", "iterative", "explicit-stack", "deep-graph"],
    kcs=["dfs-three-set-toposort", "cycle-detection-temp-set"],
    lo=(
        "Apply the iterative (stack-based) variant of three-color DFS "
        "topological sort using two-phase node entries so deep "
        "computational graphs don't hit Python's recursion limit."
    ),
    prompt_body=(
        "Implement `topological_sort_iter(root, get_children)` — same "
        "contract as the recursive version from ex1, but with an "
        "EXPLICIT stack so it doesn't recurse into Python.\n\n"
        "**Why this matters.** A real-world neural-network forward graph "
        "can be hundreds of layers deep. Python's default `sys.setrecursionlimit` "
        "is 1000 frames — the recursive form throws `RecursionError` on a "
        "1024-layer DAG. Production autograd libraries (including PyTorch's "
        "C++ engine) use iterative traversal for exactly this reason.\n\n"
        "**Contract.** Same as the recursive version:\n"
        "- Returns descendants of `root` in deps-FIRST order (root LAST).\n"
        "- Each reachable node appears EXACTLY once (diamond DAGs OK).\n"
        "- Raises `ValueError` on a cycle (DAG only).\n"
        "- Key by `id(node)`.\n\n"
        "**Two-phase stack trick.** Pre-order tells you which node is "
        "ENTERING the DFS frame; post-order tells you which node has "
        "FINISHED its subtree. Recursive code gets both for free. "
        "Iterative: push an `('exit', node)` marker BEFORE pushing the "
        "children, so when we pop the `'exit'` we know the subtree is "
        "done.\n\n"
        "```python\n"
        "stack = [(root, 'enter')]\n"
        "while stack:\n"
        "    node, phase = stack.pop()\n"
        "    if phase == 'exit':       # subtree done — post-order moment\n"
        "        temp.discard(id(node)); perm.add(id(node)); result.append(node)\n"
        "        continue\n"
        "    if id(node) in perm: continue\n"
        "    if id(node) in temp: raise ValueError('cycle')\n"
        "    temp.add(id(node))\n"
        "    stack.append((node, 'exit'))           # post-order hook\n"
        "    for child in get_children(node):\n"
        "        stack.append((child, 'enter'))\n"
        "```\n\n"
        "DO NOT use `sys.setrecursionlimit` — that's the wrong fix. The "
        "test explicitly builds a graph deeper than `sys.getrecursionlimit()`."
    ),
    stub=(
        "def topological_sort_iter(root, get_children):\n"
        '    """Iterative DFS topo sort. Same contract as the recursive version:\n'
        "    descendants of root in deps-first order (root LAST), unique nodes,\n"
        "    cycle -> ValueError.\n"
        '    """\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import sys as _sys\n"
        "\n"
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
        "# --- linear chain a -> b -> c ---\n"
        "c = N('c')\n"
        "b = N('b', c)\n"
        "a = N('a', b)\n"
        "order = topological_sort_iter(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names == ['c', 'b', 'a'], f'linear chain order: {names}'\n"
        "\n"
        "# --- diamond DAG ---\n"
        "d = N('d')\n"
        "b = N('b', d)\n"
        "c = N('c', d)\n"
        "a = N('a', b, c)\n"
        "order = topological_sort_iter(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names.count('d') == 1, f'd must appear ONCE: {names}'\n"
        "assert names[-1] == 'a'\n"
        "assert names.index('d') < names.index('b')\n"
        "assert names.index('d') < names.index('c')\n"
        "assert len(order) == 4\n"
        "\n"
        "# --- THE iterative-form payoff: deep chain that would blow Python\n"
        "#     recursion (default ~1000). Build a chain ~2000 deep and assert\n"
        "#     the iterative form completes without RecursionError. ---\n"
        "depth = max(2000, _sys.getrecursionlimit() + 500)\n"
        "deep_leaf = N('leaf')\n"
        "cur = deep_leaf\n"
        "for i in range(depth):\n"
        "    cur = N(f'n{i}', cur)\n"
        "deep_root = cur\n"
        "order = topological_sort_iter(deep_root, get_children)\n"
        "assert len(order) == depth + 1, (\n"
        "    f'deep chain should yield depth+1 nodes, got {len(order)}'\n"
        ")\n"
        "assert order[0] is deep_leaf, 'leaf must be FIRST (deps-first)'\n"
        "assert order[-1] is deep_root, 'root must be LAST'\n"
        "\n"
        "# --- cycle detection still works ---\n"
        "x = N('x')\n"
        "y = N('y')\n"
        "x.children = [y]\n"
        "y.children = [x]\n"
        "raised = False\n"
        "try:\n"
        "    topological_sort_iter(x, get_children)\n"
        "except ValueError:\n"
        "    raised = True\n"
        "assert raised, 'cycle must still raise ValueError in iterative form'\n"
        "\n"
        "# --- self-loop ---\n"
        "s = N('s')\n"
        "s.children = [s]\n"
        "raised = False\n"
        "try:\n"
        "    topological_sort_iter(s, get_children)\n"
        "except ValueError:\n"
        "    raised = True\n"
        "assert raised, 'self-loop must raise ValueError'\n"
        "\n"
        "# --- singleton ---\n"
        "lonely = N('lonely')\n"
        "order = topological_sort_iter(lonely, get_children)\n"
        "assert order == [lonely]\n"
        "\n"
        "# --- branching: ensure deps-first invariant holds ---\n"
        "leaf = N('leaf')\n"
        "m1 = N('m1', leaf)\n"
        "m2 = N('m2', leaf)\n"
        "root = N('root', m1, m2)\n"
        "order = topological_sort_iter(root, get_children)\n"
        "names = [n.name for n in order]\n"
        "pos = {nm: i for i, nm in enumerate(names)}\n"
        "assert pos['leaf'] < pos['m1'] < pos['root']\n"
        "assert pos['leaf'] < pos['m2'] < pos['root']\n"
        "assert names[-1] == 'root'\n"
        "assert len(order) == 4   # leaf, m1, m2, root — each once\n"
        "\n"
        "# --- still uses id() (not value-eq) — value-equal nodes are distinct ---\n"
        "class Vn:\n"
        "    def __init__(self, name, *children):\n"
        "        self.name = name; self.children = list(children)\n"
        "    def __eq__(self, other): return self.name == getattr(other, 'name', None)\n"
        "    def __hash__(self): return hash(self.name)\n"
        "x1 = Vn('x'); x2 = Vn('x')   # value-equal but DIFFERENT objects\n"
        "rt = Vn('rt', x1, x2)\n"
        "order = topological_sort_iter(rt, lambda n: n.children)\n"
        "# id() keying should keep them DISTINCT — three nodes, not two.\n"
        "assert len(order) == 3, (\n"
        "    f'id()-keying must keep value-equal nodes distinct: {[n.name for n in order]}'\n"
        ")"
    ),
    solution_body=(
        "def topological_sort_iter(root, get_children):\n"
        "    result = []\n"
        "    perm = set()\n"
        "    temp = set()\n"
        "    # Stack frames: (node, 'enter') or (node, 'exit').\n"
        "    stack = [(root, 'enter')]\n"
        "    while stack:\n"
        "        node, phase = stack.pop()\n"
        "        nid = id(node)\n"
        "        if phase == 'exit':\n"
        "            # subtree finished — drop from temp, mark perm, emit\n"
        "            temp.discard(nid)\n"
        "            if nid not in perm:\n"
        "                perm.add(nid)\n"
        "                result.append(node)\n"
        "            continue\n"
        "        # phase == 'enter'\n"
        "        if nid in perm:\n"
        "            continue\n"
        "        if nid in temp:\n"
        "            raise ValueError(\n"
        "                f'Cycle detected at {node!r} — graph is not a DAG'\n"
        "            )\n"
        "        temp.add(nid)\n"
        "        # push our own exit marker BEFORE pushing children, so we\n"
        "        # finalize after all children finalize (LIFO stack order).\n"
        "        stack.append((node, 'exit'))\n"
        "        for child in get_children(node):\n"
        "            stack.append((child, 'enter'))\n"
        "    return result"
    ),
    solution_notes=(
        "**Why two phases.** A recursive DFS gets two natural moments: "
        "the call (pre-order) and the return (post-order). Iterative "
        "code only has 'I popped this off the stack' — so we encode the "
        "two moments as two stack entries. The `'exit'` marker pushed "
        "BEFORE the children ensures it's popped AFTER they all finish "
        "(LIFO).\n\n"
        "**Why `temp.discard(nid)` in `'exit'`.** If we just `temp.remove`, "
        "and a node was re-popped after already being in `perm` (rare edge "
        "case if children are pushed multiple times), `remove` would raise. "
        "`discard` is the no-op-on-absent variant — safer.\n\n"
        "**Why the `if nid not in perm` guard on emit.** With diamond DAGs, "
        "the same child can be queued from multiple parents. The "
        "`'enter'`-phase guard `if nid in perm: continue` rejects the "
        "re-entry, so the `'exit'` is never pushed a second time — but "
        "the guard on `'exit'` is a cheap belt-and-braces for any "
        "unusual `get_children` callback.\n\n"
        "**Sibling visitation order.** Because we push children "
        "left-to-right and pop right-to-left, the iterative version "
        "visits siblings in REVERSE order vs the recursive form. The "
        "deps-first contract is unchanged; only the order of "
        "ties between independent leaves differs."
    ),
)


# =========================================================================
# atom: cycle-detection-temp-set — ex2: return the cycle PATH for debugging
# ex1 was bool detect; ex2 returns the actual node list for actionable
# error messages.
# =========================================================================

SPEC_CYCLE_DETECT = _spec(
    atom_id="cycle-detection-temp-set",
    subtopic="Backprop: cycle detection via temp set",
    recap=RECAP_CYCLE_DETECTION,
    ex_idx=2,
    ex_title="return the cycle PATH (list of nodes around the loop) for debugging",
    slug="return-cycle-path-for-debugging",
    bloom="Apply",
    difficulty_num=3,
    keywords=["cycle-detection", "cycle-path", "debug", "back-edge", "temp-set"],
    kcs=["cycle-detection-temp-set", "dfs-three-set-toposort"],
    lo=(
        "Apply the temp-set DFS cycle-detection pattern to RETURN the "
        "actual list of nodes around the cycle (instead of just bool), "
        "by slicing the in-flight path from the first appearance of the "
        "back-edge target."
    ),
    prompt_body=(
        "Implement `find_cycle(root, get_children)` — like `has_cycle` "
        "from ex1, but instead of returning `True/False`, return either:\n"
        "- `None` — no cycle reachable from `root` (the DAG case), OR\n"
        "- A `list[node]` — the nodes around the cycle, in DFS-traversal "
        "order, with the back-edge target REPEATED at the end so the cycle "
        "is closed.\n\n"
        "**Why this is useful.** A boolean tells you 'something is wrong'. "
        "The actual cycle path lets you write an error message like "
        "`Cycle: a -> b -> c -> a` — which the user can fix.\n\n"
        "**Algorithm sketch.** Same temp-set DFS as ex1, plus an "
        "auxiliary ORDERED `path` list of the nodes currently on the "
        "stack. When you hit a back-edge (id is in `on_stack`), slice "
        "`path` from the position of the back-edge target onwards, and "
        "append the target again to make the cycle explicit:\n\n"
        "```python\n"
        "def visit(node):\n"
        "    nid = id(node)\n"
        "    if nid in perm: return None\n"
        "    if nid in on_stack:\n"
        "        i = next(j for j, n in enumerate(path) if id(n) == nid)\n"
        "        return path[i:] + [node]\n"
        "    on_stack.add(nid); path.append(node)\n"
        "    for child in get_children(node):\n"
        "        cyc = visit(child)\n"
        "        if cyc is not None: return cyc\n"
        "    on_stack.discard(nid); path.pop()\n"
        "    perm.add(nid)\n"
        "    return None\n"
        "```\n\n"
        "Sibling-subtree contamination — `on_stack.discard` AND `path.pop` "
        "must both run on the way back up — otherwise sibling subtrees "
        "will (wrongly) see leftover nodes and false-positive.\n\n"
        "**Return type.** `list[node] | None`. Don't raise."
    ),
    stub=(
        "def find_cycle(root, get_children):\n"
        '    """Return the list of nodes around the cycle (closing node\n'
        "    repeated at the end) if one exists; else None.\n"
        '    """\n'
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
        "# --- DAG cases return None ---\n"
        "c = N('c')\n"
        "b = N('b', c)\n"
        "a = N('a', b)\n"
        "assert find_cycle(a, get_children) is None, 'linear chain is DAG'\n"
        "\n"
        "d = N('d')\n"
        "b = N('b', d)\n"
        "c = N('c', d)\n"
        "a = N('a', b, c)\n"
        "assert find_cycle(a, get_children) is None, 'diamond DAG is acyclic'\n"
        "\n"
        "# --- two-node cycle: a -> b -> a ---\n"
        "a = N('a')\n"
        "b = N('b')\n"
        "a.children = [b]\n"
        "b.children = [a]\n"
        "cyc = find_cycle(a, get_children)\n"
        "assert cyc is not None, 'two-node cycle must be reported'\n"
        "names = [n.name for n in cyc]\n"
        "assert names == ['a', 'b', 'a'], f'expected closed loop a->b->a, got {names}'\n"
        "\n"
        "# --- self-loop: s -> s ---\n"
        "s = N('s')\n"
        "s.children = [s]\n"
        "cyc = find_cycle(s, get_children)\n"
        "assert cyc is not None\n"
        "names = [n.name for n in cyc]\n"
        "assert names == ['s', 's'], f'self-loop should be [s, s], got {names}'\n"
        "\n"
        "# --- 3-node cycle, mid-graph: p -> q -> r -> q ---\n"
        "p = N('p')\n"
        "q = N('q')\n"
        "r = N('r')\n"
        "p.children = [q]\n"
        "q.children = [r]\n"
        "r.children = [q]\n"
        "cyc = find_cycle(p, get_children)\n"
        "assert cyc is not None\n"
        "names = [n.name for n in cyc]\n"
        "# The cycle starts at q (back-edge target), goes through r, closes at q.\n"
        "# Path-from-q slice = [q, r], plus [q] at end = [q, r, q].\n"
        "assert names == ['q', 'r', 'q'], f'expected [q, r, q], got {names}'\n"
        "\n"
        "# --- sibling-subtree non-contamination: ---\n"
        "# Graph: root -> a -> leaf, root -> b -> leaf (shared leaf, NO cycle).\n"
        "# A buggy on_stack.discard miss would false-report a cycle here.\n"
        "leaf = N('leaf')\n"
        "ax = N('a', leaf)\n"
        "bx = N('b', leaf)\n"
        "rt = N('rt', ax, bx)\n"
        "assert find_cycle(rt, get_children) is None, (\n"
        "    'two siblings sharing a leaf must NOT report a cycle — '\n"
        "    'did you forget on_stack.discard or path.pop?'\n"
        ")\n"
        "\n"
        "# --- cycle in ONE branch only ---\n"
        "good_c = N('good_c')\n"
        "good_b = N('good_b', good_c)\n"
        "bad_x = N('bad_x'); bad_y = N('bad_y')\n"
        "bad_x.children = [bad_y]; bad_y.children = [bad_x]\n"
        "rt = N('rt', good_b, bad_x)\n"
        "cyc = find_cycle(rt, get_children)\n"
        "assert cyc is not None, 'cycle in subgraph must propagate'\n"
        "names = [n.name for n in cyc]\n"
        "# Whichever side the DFS visits first, the cycle must close.\n"
        "assert names[0] == names[-1], (\n"
        "    f'closing node must equal opening node: {names}'\n"
        ")\n"
        "assert set(names[:-1]) == {'bad_x', 'bad_y'}, (\n"
        "    f'cycle path should consist of the two cycle nodes only: {names}'\n"
        ")\n"
        "\n"
        "# --- the returned objects are the ORIGINAL nodes, not copies ---\n"
        "a = N('a')\n"
        "b = N('b')\n"
        "a.children = [b]\n"
        "b.children = [a]\n"
        "cyc = find_cycle(a, get_children)\n"
        "assert cyc[0] is a or cyc[0] is b, 'first elem is one of the cycle nodes by identity'\n"
        "assert cyc[0] is cyc[-1], 'closing node must be the SAME object as opening'"
    ),
    solution_body=(
        "def find_cycle(root, get_children):\n"
        "    path = []           # nodes currently on the DFS stack, in order\n"
        "    on_stack = set()    # ids of those nodes — O(1) lookup\n"
        "    perm = set()        # finished subtrees — safe to skip\n"
        "\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm:\n"
        "            return None            # already finished\n"
        "        if nid in on_stack:\n"
        "            # back-edge — locate the first appearance of nid in path,\n"
        "            # slice forward, and CLOSE the loop by repeating the target.\n"
        "            i = next(j for j, n in enumerate(path) if id(n) == nid)\n"
        "            return path[i:] + [node]\n"
        "        on_stack.add(nid)\n"
        "        path.append(node)\n"
        "        for child in get_children(node):\n"
        "            cyc = visit(child)\n"
        "            if cyc is not None:\n"
        "                return cyc        # propagate up — found, no further work\n"
        "        on_stack.discard(nid)     # leaving subtree — drop from stack-tracking\n"
        "        path.pop()                # ...and from the ordered path\n"
        "        perm.add(nid)\n"
        "        return None\n"
        "\n"
        "    return visit(root)"
    ),
    solution_notes=(
        "**`path` is the ordered companion of `on_stack`.** `on_stack` "
        "gives O(1) membership; `path` preserves the visit order so we "
        "can slice from the back-edge target. Both update in lock-step "
        "— add+append on enter, discard+pop on exit. The `next(j ... if "
        "id(n) == nid)` scan is at worst O(depth), which is what you "
        "pay once on the cycle-found path; the DAG case never runs it.\n\n"
        "**Why repeat the closing node.** Returning `[a, b]` for an "
        "`a → b → a` cycle is ambiguous — is it a 2-cycle or a 1-cycle? "
        "Repeating the target as `[a, b, a]` makes the closure explicit "
        "and matches how graph-theory texts print cycles.\n\n"
        "**Vs. the sibling `dfs-three-set-toposort` atom (ex1).** Same "
        "temp-set machinery; that atom RAISES on the back-edge. This "
        "one needs the path BEFORE raising — so we collect it and "
        "return it. The caller decides whether to `raise ValueError(' -> "
        "'.join(repr(n) for n in cyc))` or just log it."
    ),
)


# =========================================================================
# atom: backprop-pop-outgrad-loop — ex2: per-leaf accumulation count
# ex1 was the main reverse-pass driver; ex2 instruments it with a parallel
# {id(leaf): count} dict for diagnosing how grad routed.
# =========================================================================

SPEC_BACKPROP_LOOP = _spec(
    atom_id="backprop-pop-outgrad-loop",
    subtopic="Backprop: backprop pop-outgrad loop",
    recap=RECAP_BACKPROP_LOOP,
    ex_idx=2,
    ex_title="backprop with per-leaf accumulation-count diagnostic",
    slug="backprop-with-per-leaf-accumulation-counts",
    bloom="Apply",
    difficulty_num=4,
    keywords=["reverse-pass", "diagnostic", "accumulation-count", "diamond-dag"],
    kcs=["backprop-pop-outgrad-loop", "dispatch-back-fn-from-recipe"],
    lo=(
        "Apply the reverse-pass driver pattern while instrumenting it with "
        "a parallel per-leaf accumulation-count dict, so the caller can "
        "verify how many gradient contributions each leaf received."
    ),
    prompt_body=(
        "Implement `backprop_counted(end_node, end_grad, sorted_graph, "
        "back_funcs)` — same as the ex1 reverse-pass driver, plus return a "
        "`{id(leaf): count}` dict tracking how many `.grad` accumulations "
        "each leaf received during this backward pass.\n\n"
        "**Signature.**\n"
        "```python\n"
        "def backprop_counted(end_node, end_grad, sorted_graph, back_funcs):\n"
        "    ...\n"
        "    return counts   # {id(leaf): int}\n"
        "```\n\n"
        "**Semantics.** A leaf node is a `MiniTensor` with `recipe is "
        "None`. Each time the reverse pass reaches a leaf in `sorted_graph` "
        "and writes/adds to `.grad`, increment `counts[id(leaf)]` by 1. "
        "Non-leaves do not appear in `counts`.\n\n"
        "**Use case.** In a diamond DAG `out = z * z`, leaf `z` appears "
        "as `parents={0: z, 1: z}`. Both arg-0 and arg-1 contribute "
        "back into `grads[id(z)]` BEFORE `z` itself is popped — so when "
        "`z`'s turn comes, `.grad` is written ONCE with the merged "
        "value. Expected `counts[id(z)] == 1`. In contrast, a leaf "
        "consumed by k separate downstream operations (each its own "
        "node in sorted_graph) would accumulate k times — once per "
        "consumer's reverse step.\n\n"
        "**Algorithm.** Same as ex1 plus the counter increment:\n"
        "```python\n"
        "grads = {id(end_node): end_grad}\n"
        "counts = {}\n"
        "for node in sorted_graph:\n"
        "    if id(node) not in grads: continue\n"
        "    grad_out = grads.pop(id(node))\n"
        "    if node.recipe is None:\n"
        "        node.grad = grad_out if node.grad is None else node.grad + grad_out\n"
        "        counts[id(node)] = counts.get(id(node), 0) + 1\n"
        "        continue\n"
        "    for argnum, parent in node.recipe.parents.items():\n"
        "        bf = back_funcs[(node.recipe.func, argnum)]\n"
        "        gp = bf(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)\n"
        "        grads[id(parent)] = grads.get(id(parent), 0) + gp\n"
        "return counts\n"
        "```\n\n"
        "Same three invariants as ex1: pop-don't-peek, accumulate-don't-"
        "overwrite, leaves-get-`.grad`-non-leaves-stay-in-`grads`."
    ),
    stub=(
        "def backprop_counted(end_node, end_grad, sorted_graph, back_funcs) -> dict:\n"
        '    """Reverse-pass driver; mutate leaf .grad in-place;\n'
        "    return {id(leaf): accumulation_count}.\n"
        '    """\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import math as _math\n"
        "\n"
        "# --- back fns (raw torch, no autograd) ---\n"
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
        "# === TEST 1: single-leaf log chain → leaf accumulates ONCE ===\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 4.0]), requires_grad=True)\n"
        "out = MiniTensor(t.log(a.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "counts = backprop_counted(out, t.ones(3), [out, a], BF)\n"
        "assert t.allclose(a.grad, 1 / a.array), 'value sanity'\n"
        "assert counts == {id(a): 1}, f'expected {{id(a): 1}}, got {counts}'\n"
        "\n"
        "# === TEST 2: diamond `z * z` → z accumulates ONCE (merged before leaf turn) ===\n"
        "z = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out_arr = z.array * z.array\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(z.array, z.array), kwargs={}, parents={0: z, 1: z}\n"
        ")\n"
        "counts = backprop_counted(out, t.ones(1), [out, z], BF)\n"
        "# value: d(z**2)/dz = 2z = 6\n"
        "assert t.allclose(z.grad, t.tensor([6.0]))\n"
        "# z is a leaf that was popped once → ONE .grad write\n"
        "assert counts == {id(z): 1}, (\n"
        "    f'diamond merges INTO grads dict before leaf pop — '\n"
        "    f'.grad should be written once, got counts={counts}'\n"
        ")\n"
        "\n"
        "# === TEST 3: two-leaf chain → each leaf counted once ===\n"
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
        "counts = backprop_counted(out, t.ones(1), [out, a, c, b], BF)\n"
        "assert t.allclose(a.grad, t.tensor([1.0]), atol=1e-5)\n"
        "assert t.allclose(b.grad, t.tensor([2.0 / _math.e]), atol=1e-5)\n"
        "assert counts == {id(a): 1, id(b): 1}, f'got {counts}'\n"
        "assert id(c) not in counts, (\n"
        "    f'c is a non-leaf and must NOT appear in counts: {counts}'\n"
        ")\n"
        "\n"
        "# === TEST 4: leaf pre-seeded .grad still counted exactly once ===\n"
        "a = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "a.grad = t.tensor([10.0])\n"
        "out = MiniTensor(t.log(a.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "counts = backprop_counted(out, t.ones(1), [out, a], BF)\n"
        "# pre + new = 10 + 1 = 11\n"
        "assert t.allclose(a.grad, t.tensor([11.0])), f'accumulate, got {a.grad}'\n"
        "assert counts == {id(a): 1}, (\n"
        "    f'pre-seeded .grad is NOT counted (only THIS pass increments)'\n"
        ")\n"
        "\n"
        "# === TEST 5: non-leaf nodes NEVER appear in counts ===\n"
        "# Build a 3-node chain and confirm the middle node is absent.\n"
        "x = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "mid = MiniTensor(t.log(x.array), requires_grad=True)\n"
        "mid.recipe = Recipe(func=t.log, args=(x.array,), kwargs={}, parents={0: x})\n"
        "end = MiniTensor(t.log(mid.array), requires_grad=True)\n"
        "end.recipe = Recipe(func=t.log, args=(mid.array,), kwargs={}, parents={0: mid})\n"
        "counts = backprop_counted(end, t.ones(1), [end, mid, x], BF)\n"
        "assert id(x) in counts and counts[id(x)] == 1\n"
        "assert id(mid) not in counts, 'mid is non-leaf — must not be in counts'\n"
        "assert id(end) not in counts, 'end is non-leaf — must not be in counts'\n"
        "\n"
        "# === TEST 6: returns the COUNTS DICT (not None) ===\n"
        "a = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "out = MiniTensor(t.log(a.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "result = backprop_counted(out, t.ones(1), [out, a], BF)\n"
        "assert isinstance(result, dict), f'must return dict, got {type(result)}'\n"
        "assert all(isinstance(v, int) for v in result.values()), 'counts must be ints'"
    ),
    solution_body=(
        "def backprop_counted(end_node, end_grad, sorted_graph, back_funcs) -> dict:\n"
        "    grads = {id(end_node): end_grad}\n"
        "    counts = {}\n"
        "    for node in sorted_graph:\n"
        "        nid = id(node)\n"
        "        if nid not in grads:\n"
        "            continue\n"
        "        grad_out = grads.pop(nid)         # POP — node done after this\n"
        "        if node.recipe is None:\n"
        "            # Leaf: accumulate (don't overwrite) into .grad + bump count.\n"
        "            if node.grad is None:\n"
        "                node.grad = grad_out\n"
        "            else:\n"
        "                node.grad = node.grad + grad_out\n"
        "            counts[nid] = counts.get(nid, 0) + 1\n"
        "            continue\n"
        "        # Non-leaf: dispatch + accumulate into each parent's slot.\n"
        "        for argnum, parent in node.recipe.parents.items():\n"
        "            back_fn = back_funcs[(node.recipe.func, argnum)]\n"
        "            grad_parent = back_fn(\n"
        "                grad_out, node.array,\n"
        "                *node.recipe.args, **node.recipe.kwargs,\n"
        "            )\n"
        "            pid = id(parent)\n"
        "            grads[pid] = grads.get(pid, 0) + grad_parent\n"
        "    return counts"
    ),
    solution_notes=(
        "**Counts diagnose routing, not values.** Two graphs can produce "
        "the same numerical `.grad` via different routings — counts tell "
        "you which one actually fired. In a diamond `z*z`, count is 1 "
        "(merged in grads dict before leaf pop); in a fork "
        "`out1 = z; out2 = z; loss = f(out1, out2)` where the leaf is "
        "consumed by TWO separate non-leaf nodes, count is 2.\n\n"
        "**Why counts isn't just a length check on grads dict.** Once a "
        "leaf is popped, its `id` leaves `grads` — only the cumulative "
        "`counts` survives the pass. Useful for off-line analysis of a "
        "completed backward.\n\n"
        "**Composability with the un-instrumented driver.** Wrapping "
        "the existing `backprop` in a counter would require closure "
        "tricks; the cleaner approach is to add a counter dict to the "
        "ONE function that already knows when the increment should fire."
    ),
)


# =========================================================================
# atom: dispatch-back-fn-from-recipe — ex2: friendly KeyError with op+argnum
# ex1 returned triples and let raw KeyError propagate. ex2 wraps the lookup
# with a diagnostic message that names op + argnum + how to fix it.
# =========================================================================

SPEC_DISPATCH = _spec(
    atom_id="dispatch-back-fn-from-recipe",
    subtopic="Backprop: dispatch back fn from recipe",
    recap=RECAP_DISPATCH_BACK_FN,
    ex_idx=2,
    ex_title="dispatch with friendly KeyError naming op + argnum + fix",
    slug="dispatch-with-friendly-key-error",
    bloom="Apply",
    difficulty_num=3,
    keywords=["dispatch", "diagnostic", "key-error", "error-message"],
    kcs=["dispatch-back-fn-from-recipe", "parents-dict-by-argidx"],
    lo=(
        "Apply the (recipe.func, argnum)-keyed dispatch pattern while "
        "wrapping the registry lookup with a custom KeyError that names "
        "the forward fn AND the argnum that has no registered back_fn — "
        "so users see actionable error messages instead of opaque tuples."
    ),
    prompt_body=(
        "Implement `dispatch_back_fns_diag(node, back_funcs)` — same "
        "return shape as ex1 (`[(argnum, parent, back_fn), ...]`), but "
        "when a `(recipe.func, argnum)` key is MISSING from `back_funcs`, "
        "raise a `KeyError` whose message names:\n\n"
        "1. The forward function (use `__name__` if available, else "
        "`repr(fn)`).\n"
        "2. The argnum that's missing.\n"
        "3. A suggested fix: `'register_back_func({fn_name}, {argnum}, ...)'`.\n\n"
        "**Example message** for an unregistered `t.sin` at arg 0:\n"
        "```\n"
        "KeyError: No back_fn registered for (sin, argnum=0). Add it via "
        "register_back_func(sin, 0, ...).\n"
        "```\n\n"
        "**Why this matters.** The raw `KeyError: (<built-in function sin>, 0)` "
        "is technically correct but cryptic — users have to know what the "
        "tuple means. A human-readable message is the whole point of a "
        "diagnostic wrapper.\n\n"
        "**Algorithm.**\n"
        "```python\n"
        "for argnum, parent in node.recipe.parents.items():\n"
        "    key = (node.recipe.func, argnum)\n"
        "    if key not in back_funcs:\n"
        "        fn_name = getattr(node.recipe.func, '__name__', repr(node.recipe.func))\n"
        "        raise KeyError(\n"
        "            f'No back_fn registered for ({fn_name}, argnum={argnum}). '\n"
        "            f'Add it via register_back_func({fn_name}, {argnum}, ...).'\n"
        "        )\n"
        "    out.append((argnum, parent, back_funcs[key]))\n"
        "```\n\n"
        "**Return type.** `list[tuple]`. **Error type.** `KeyError` "
        "(not RuntimeError, not a custom class — the caller still "
        "catches `KeyError` if it wants to handle this generically)."
    ),
    stub=(
        "def dispatch_back_fns_diag(node, back_funcs) -> list:\n"
        '    """Return [(argnum, parent, back_fn), ...]. Raise KeyError with\n'
        "    a HUMAN-READABLE message if any (recipe.func, argnum) is unregistered.\n"
        '    """\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- registered back fns ---\n"
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
        "# === happy path: single-parent log node ===\n"
        "b = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "c = MiniTensor(t.log(b.array), requires_grad=True)\n"
        "c.recipe = Recipe(func=t.log, args=(b.array,), kwargs={}, parents={0: b})\n"
        "triples = dispatch_back_fns_diag(c, BF)\n"
        "assert len(triples) == 1\n"
        "argnum, parent, back_fn = triples[0]\n"
        "assert argnum == 0 and parent is b and back_fn is log_back\n"
        "\n"
        "# === happy path: two-parent multiply ===\n"
        "x = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out = MiniTensor(x.array * y.array, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y}\n"
        ")\n"
        "triples = dispatch_back_fns_diag(out, BF)\n"
        "by_argnum = {a: (p, f) for a, p, f in triples}\n"
        "assert by_argnum[0] == (x, mul_back0)\n"
        "assert by_argnum[1] == (y, mul_back1)\n"
        "\n"
        "# === unregistered op → KeyError with HUMAN-READABLE message ===\n"
        "missing = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "missing.recipe = Recipe(\n"
        "    func=t.sin, args=(t.tensor([1.0]),), kwargs={}, parents={0: missing}\n"
        ")\n"
        "raised = False\n"
        "msg = ''\n"
        "try:\n"
        "    dispatch_back_fns_diag(missing, BF)\n"
        "except KeyError as e:\n"
        "    raised = True\n"
        "    # KeyError stringifies the args; .args[0] is the readable message we set\n"
        "    msg = e.args[0] if e.args else str(e)\n"
        "assert raised, 'unregistered op must raise KeyError'\n"
        "assert 'sin' in msg, (\n"
        "    f'error message must name the forward fn (\\'sin\\'), got: {msg!r}'\n"
        ")\n"
        "assert 'argnum=0' in msg or 'argnum 0' in msg, (\n"
        "    f'error message must name argnum=0, got: {msg!r}'\n"
        ")\n"
        "assert 'register_back_func' in msg, (\n"
        "    f'error must suggest the fix register_back_func, got: {msg!r}'\n"
        ")\n"
        "\n"
        "# === multi-arg partial registration: arg 0 registered, arg 1 missing ===\n"
        "# Simulate by stripping (multiply, 1) from a copy of BF.\n"
        "partial = {(t.log, 0): log_back, (t.multiply, 0): mul_back0}\n"
        "xx = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "yy = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out2 = MiniTensor(xx.array * yy.array, requires_grad=True)\n"
        "out2.recipe = Recipe(\n"
        "    func=t.multiply, args=(xx.array, yy.array), kwargs={},\n"
        "    parents={0: xx, 1: yy}\n"
        ")\n"
        "raised = False\n"
        "msg = ''\n"
        "try:\n"
        "    dispatch_back_fns_diag(out2, partial)\n"
        "except KeyError as e:\n"
        "    raised = True\n"
        "    msg = e.args[0] if e.args else str(e)\n"
        "assert raised, 'partial registration must still raise on the missing argnum'\n"
        "assert 'multiply' in msg\n"
        "assert ('argnum=1' in msg) or ('argnum 1' in msg), (\n"
        "    f'must name argnum=1 specifically: {msg!r}'\n"
        ")\n"
        "\n"
        "# === structural shape of happy path ===\n"
        "argnum, parent, back_fn = dispatch_back_fns_diag(c, BF)[0]\n"
        "assert isinstance(argnum, int)\n"
        "assert isinstance(parent, MiniTensor)\n"
        "assert callable(back_fn)\n"
        "\n"
        "# === the raised exception is a KeyError SUBCLASS, not a different class ===\n"
        "try:\n"
        "    dispatch_back_fns_diag(missing, BF)\n"
        "except KeyError:\n"
        "    pass\n"
        "except Exception as e:\n"
        "    assert False, f'expected KeyError, got {type(e).__name__}'"
    ),
    solution_body=(
        "def dispatch_back_fns_diag(node, back_funcs) -> list:\n"
        "    out = []\n"
        "    for argnum, parent in node.recipe.parents.items():\n"
        "        key = (node.recipe.func, argnum)\n"
        "        if key not in back_funcs:\n"
        "            fn = node.recipe.func\n"
        "            fn_name = getattr(fn, '__name__', repr(fn))\n"
        "            raise KeyError(\n"
        "                f'No back_fn registered for ({fn_name}, argnum={argnum}). '\n"
        "                f'Add it via register_back_func({fn_name}, {argnum}, ...).'\n"
        "            )\n"
        "        out.append((argnum, parent, back_funcs[key]))\n"
        "    return out"
    ),
    solution_notes=(
        "**Diagnostic messages are a feature, not polish.** PyTorch's "
        "autograd raises `RuntimeError: Trying to backward through the "
        "graph a second time...` — a sentence, not a tuple. Every "
        "minute the user spends decoding `KeyError: (<built-in "
        "function sin>, 0)` is a minute they're not fixing the bug. "
        "The wrapper is the API surface where you spend a paragraph "
        "of error message in exchange for hours of user-debug time.\n\n"
        "**Why `getattr(fn, '__name__', repr(fn))`.** Built-in torch "
        "ops have `__name__`. Some wrapped/partial functions don't — "
        "the `repr` fallback keeps the message printable on any "
        "callable.\n\n"
        "**Why still `KeyError`, not a new exception class.** "
        "Subclassing KeyError would let callers catch `KeyError` to "
        "handle 'op not registered yet' uniformly with native dict "
        "misses. Don't fragment the exception hierarchy without a "
        "reason."
    ),
)


# =========================================================================
# atom: back-fn-call-with-recipe-args — ex2: spy back_fn that detects splat bugs
# ex1 was the canonical 4-channel call. ex2 builds a DEBUG helper that
# inspects which channel went wrong (missing *, missing **, wrong out).
# =========================================================================

SPEC_BACK_FN_CALL = _spec(
    atom_id="back-fn-call-with-recipe-args",
    subtopic="Backprop: back fn call with recipe args",
    recap=RECAP_BACK_FN_CALL,
    ex_idx=2,
    ex_title="diagnose back_fn call-site bugs via a recording wrapper",
    slug="diagnose-back-fn-call-site-bugs-via-recording-wrapper",
    bloom="Analyze",
    difficulty_num=3,
    keywords=["debug", "splat-bug", "back-fn-call", "spy", "recording"],
    kcs=["back-fn-call-with-recipe-args", "kwargs-pass-through-recipe"],
    lo=(
        "Analyze a back_fn invocation by writing a `call_back_fn_recording` "
        "wrapper that returns the back_fn result PLUS a dict capturing "
        "exactly what reached each of the four channels (grad_out, out, "
        "args, kwargs) — enabling diagnosis of missing *, **, or wrong "
        "`out` passes."
    ),
    prompt_body=(
        "Implement `call_back_fn_recording(back_fn, grad_out, node)` — "
        "the canonical invocation from ex1 PLUS an audit trail. Returns "
        "`(result, record)` where:\n\n"
        "- `result` is whatever the back_fn returned (a `torch.Tensor`).\n"
        "- `record` is a dict with FOUR keys:\n"
        "  * `'grad_out'` — the `grad_out` you forwarded.\n"
        "  * `'out'` — the second positional you forwarded (should be "
        "    `node.array`, NOT `node`).\n"
        "  * `'args'` — the tuple of positional args after `out` (should "
        "    be `node.recipe.args`).\n"
        "  * `'kwargs'` — the kwargs dict (should be `node.recipe.kwargs`).\n\n"
        "**How to record.** Wrap the back_fn in a one-shot closure that "
        "captures every arg it receives BEFORE forwarding to the real "
        "back_fn. The simplest way:\n\n"
        "```python\n"
        "record = {}\n"
        "def _spy(g_out, out_, *args, **kwargs):\n"
        "    record['grad_out'] = g_out\n"
        "    record['out'] = out_\n"
        "    record['args'] = args\n"
        "    record['kwargs'] = kwargs\n"
        "    return back_fn(g_out, out_, *args, **kwargs)\n"
        "result = _spy(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)\n"
        "```\n\n"
        "The recorded dict is the diagnostic surface — a caller can "
        "diff `record['kwargs']` against `node.recipe.kwargs` to see if "
        "any kwarg got dropped (missing `**` splat), check "
        "`record['out'] is node.array` to confirm the `out` channel is "
        "the raw tensor, etc.\n\n"
        "**Why a wrapper instead of just calling and returning result.** "
        "ex1 returned only the result. ex2 elevates the call site into "
        "an analysis tool — the same invocation, but every channel is "
        "now externally inspectable. Useful when porting a back_fn from "
        "one autograd to another, or when a backward pass produces "
        "unexpectedly-shaped grads."
    ),
    stub=(
        "def call_back_fn_recording(back_fn, grad_out, node) -> tuple:\n"
        '    """Invoke back_fn canonically; return (result, record_dict)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# === TEST 1: log back_fn — no kwargs, single arg ===\n"
        "def log_back(grad_out, out, x):\n"
        "    return grad_out / x\n"
        "\n"
        "b = MiniTensor(t.tensor([2.0, 4.0]), requires_grad=True)\n"
        "c = MiniTensor(t.log(b.array), requires_grad=True)\n"
        "c.recipe = Recipe(func=t.log, args=(b.array,), kwargs={}, parents={0: b})\n"
        "result, rec = call_back_fn_recording(log_back, t.ones(2), c)\n"
        "\n"
        "# value\n"
        "assert t.allclose(result, 1 / b.array)\n"
        "# record shape\n"
        "assert set(rec.keys()) == {'grad_out', 'out', 'args', 'kwargs'}, (\n"
        "    f'record must have exactly four keys, got {set(rec.keys())}'\n"
        ")\n"
        "# channel content\n"
        "assert t.equal(rec['grad_out'], t.ones(2))\n"
        "assert rec['out'] is c.array, 'out channel must be node.array (raw torch.Tensor)'\n"
        "assert isinstance(rec['args'], tuple) and len(rec['args']) == 1\n"
        "assert t.equal(rec['args'][0], b.array)\n"
        "assert rec['kwargs'] == {}\n"
        "\n"
        "# === TEST 2: multiply — two positional args via *args ===\n"
        "def mul_back0(grad_out, out, x, y):\n"
        "    return grad_out * y\n"
        "\n"
        "x = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([7.0]), requires_grad=True)\n"
        "out = MiniTensor(x.array * y.array, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y}\n"
        ")\n"
        "result, rec = call_back_fn_recording(mul_back0, t.ones(1), out)\n"
        "assert t.allclose(result, y.array), f'd(x*y)/dx = y; got {result}'\n"
        "assert len(rec['args']) == 2, (\n"
        "    f'two positional args must reach the back_fn via *recipe.args: got {rec[\"args\"]}'\n"
        ")\n"
        "assert t.equal(rec['args'][0], x.array)\n"
        "assert t.equal(rec['args'][1], y.array)\n"
        "\n"
        "# === TEST 3: kwargs — sum_back wants dim= via **recipe.kwargs ===\n"
        "def sum_back(grad_out, out, x, dim=None, keepdim=False):\n"
        "    if dim is None:\n"
        "        return grad_out * t.ones_like(x)\n"
        "    if not keepdim:\n"
        "        grad_out = grad_out.unsqueeze(dim)\n"
        "    return grad_out.expand_as(x)\n"
        "\n"
        "xt = MiniTensor(t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)\n"
        "out_arr = xt.array.sum(dim=1)\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.sum, args=(xt.array,), kwargs={'dim': 1}, parents={0: xt}\n"
        ")\n"
        "result, rec = call_back_fn_recording(sum_back, t.ones(2), out)\n"
        "assert result.shape == xt.array.shape, 'sum_back broadcast shape'\n"
        "assert rec['kwargs'] == {'dim': 1}, (\n"
        "    f'kwargs must thread through via **recipe.kwargs; got {rec[\"kwargs\"]}'\n"
        ")\n"
        "\n"
        "# === TEST 4: both args AND kwargs — make sure both channels are recorded ===\n"
        "def spy_back(grad_out, out, x, y, *, scale=1.0):\n"
        "    return grad_out * y * scale\n"
        "\n"
        "x = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out = MiniTensor(t.tensor([6.0]), requires_grad=True)\n"
        "out.recipe = Recipe(\n"
        "    func=t.multiply, args=(x.array, y.array),\n"
        "    kwargs={'scale': 4.0}, parents={0: x, 1: y}\n"
        ")\n"
        "result, rec = call_back_fn_recording(spy_back, t.tensor([10.0]), out)\n"
        "assert t.allclose(result, t.tensor([120.0]))\n"
        "assert len(rec['args']) == 2\n"
        "assert rec['kwargs'] == {'scale': 4.0}\n"
        "\n"
        "# === TEST 5: 'out' channel diagnoses MiniTensor-vs-Tensor confusion ===\n"
        "# If a caller passed `node` (the wrapper) instead of `node.array`,\n"
        "# `rec['out']` would be a MiniTensor — the diagnostic surface lets you spot it.\n"
        "n = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "out = MiniTensor(t.log(n.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(n.array,), kwargs={}, parents={0: n})\n"
        "result, rec = call_back_fn_recording(log_back, t.ones(1), out)\n"
        "assert type(rec['out']) is t.Tensor, (\n"
        "    f'rec[\\'out\\'] must be raw torch.Tensor (node.array), '\n"
        "    f'NOT MiniTensor — got {type(rec[\"out\"])}'\n"
        ")\n"
        "\n"
        "# === TEST 6: diagnostic value — diff record against recipe ===\n"
        "# A caller can re-derive 'what SHOULD have been passed' from the recipe\n"
        "# and compare against `rec` to pinpoint missing splats.\n"
        "expected_kwargs = out.recipe.kwargs\n"
        "assert rec['kwargs'] == expected_kwargs\n"
        "expected_args_len = len(out.recipe.args)\n"
        "assert len(rec['args']) == expected_args_len, (\n"
        "    f'args length differs from recipe — splat bug detected: '\n"
        "    f'recipe.args has {expected_args_len}, got {len(rec[\"args\"])}'\n"
        ")\n"
        "\n"
        "# === TEST 7: returns a TUPLE (result, dict) — order matters ===\n"
        "r = call_back_fn_recording(log_back, t.ones(1), out)\n"
        "assert isinstance(r, tuple) and len(r) == 2\n"
        "assert isinstance(r[0], t.Tensor)\n"
        "assert isinstance(r[1], dict)"
    ),
    solution_body=(
        "def call_back_fn_recording(back_fn, grad_out, node) -> tuple:\n"
        "    record = {}\n"
        "\n"
        "    def _spy(g_out, out_, *args, **kwargs):\n"
        "        # Record EVERY channel before forwarding to the real back_fn.\n"
        "        record['grad_out'] = g_out\n"
        "        record['out'] = out_\n"
        "        record['args'] = args\n"
        "        record['kwargs'] = kwargs\n"
        "        return back_fn(g_out, out_, *args, **kwargs)\n"
        "\n"
        "    # Canonical invocation — SAME shape as ex1, just through the spy.\n"
        "    result = _spy(\n"
        "        grad_out,\n"
        "        node.array,                # raw torch.Tensor, NOT the MiniTensor\n"
        "        *node.recipe.args,         # forward positional args (unboxed)\n"
        "        **node.recipe.kwargs,      # forward kwargs (dim, keepdim, ...)\n"
        "    )\n"
        "    return result, record"
    ),
    solution_notes=(
        "**The spy is a one-line lens.** It doesn't change behavior — "
        "the back_fn still computes the same gradient — but every "
        "channel becomes externally inspectable. A test that compares "
        "`record['kwargs']` against `node.recipe.kwargs` will catch a "
        "missing `**` splat immediately; otherwise the bug only "
        "manifests when the back_fn shape-mismatches downstream.\n\n"
        "**Why a closure, not a wrapper class.** A `class CallRecorder` "
        "would work but adds ceremony. The closure captures `record` "
        "by reference, mutates it on every call (here, just one), and "
        "is gone the moment the function returns — exactly the "
        "lifetime we want.\n\n"
        "**Composability with the ex1 call_back_fn.** A tidy "
        "refactoring would have `call_back_fn` (no record) and "
        "`call_back_fn_recording` (with record) share an internal "
        "`_invoke(back_fn, grad_out, node, hook=None)` helper. For the "
        "drill, the duplication is small enough that we keep them "
        "side-by-side."
    ),
)


# =========================================================================
# emit + verify
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

        # 1. exec preamble (MiniTensor + Recipe + grad_tracking_enabled)
        try:
            exec(_AUTOGRAD_PREAMBLE, ns)
        except Exception as e:
            failed.append((tag, f"preamble failed: {e!r}", traceback.format_exc()))
            continue

        # 2. exec stub (defines target function with NotImplementedError)
        try:
            exec(spec["stub"], ns)
        except Exception:
            # stub may include a forward-declared helper that fails to import — tolerate
            pass

        # 3. exec solution (OVERWRITES the stub's NotImplementedError function)
        # 4. exec test (must pass against solution)
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
    print(f"[deepening_l_batch9] Verifying {len(ALL_SPECS)} specs...")
    _verify_all(ALL_SPECS)

    print(f"\n[deepening_l_batch9] All verified — emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_l_batch9] {len(ALL_SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
