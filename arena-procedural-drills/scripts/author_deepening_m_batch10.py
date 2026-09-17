#!/usr/bin/env python3
"""Author 8 deepening (ex2) drills for ARENA part-4 prereqs_backward_fns atoms.

Each ex2 probes a DISTINCT facet from the existing ex1 — different cognitive
operation, different surface context, same MiniTensor+Recipe+grad_tracking
scaffolding. ONE LO + ONE Bloom + <=2 KCs each.

Verification re-runs each spec's solution against its test_body inside the
build venv (torch 2.12.0+cpu) before any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_backward_fns"


# ---------------------------------------------------------------------------
# Shared autograd preamble — MiniTensor + Recipe + grad_tracking_enabled.
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

RECAP_LAMBDAS_DEEP = (
    "## `add`/`sub`/`div` back lambdas in a mini dispatcher — quick refresher\n"
    "\n"
    "ex1 built the 6-entry `BACK` dict and called lambdas directly. The deeper "
    "facet is **dispatch-as-data**: the reverse pass never names a back fn — "
    "it always looks one up by `(op_name, argnum)` and calls it.\n"
    "\n"
    "```python\n"
    "for argnum, parent in node.recipe.parents.items():\n"
    "    back_fn = BACK[(node.recipe.func_name, argnum)]\n"
    "    grad_in = back_fn(grad_out, node.array, *node.recipe.args)\n"
    "    grads[parent] = grads.get(parent, 0) + grad_in\n"
    "```\n"
    "\n"
    "Two invariants:\n"
    "- **The dispatcher routes through the dict EVERY call.** If you hardcode "
    "`g + 0` inside the loop for add, swapping `BACK[('add', 0)]` to a scaled "
    "lambda silently does nothing. Real reverse passes go through the table.\n"
    "- **Symmetric ops still get two entries.** `('add', 0)` and `('add', 1)` "
    "have identical bodies. The dispatcher cannot know — it just keys by "
    "`(op, argnum)`."
)

RECAP_EXP_DEEP = (
    "## `exp_back` composed with `log_back` — quick refresher\n"
    "\n"
    "ex1 derived `exp_back = grad_out * out`. The deeper facet is what happens "
    "when you COMPOSE inverse ops in a chain.\n"
    "\n"
    "For `y = exp(log(x))` the math is `y = x` so `dy/dx = 1`. Through the "
    "chain rule applied via the back fns:\n"
    "\n"
    "```\n"
    "u = log(x)         out_u = log(x)\n"
    "y = exp(u)         out_y = exp(u) = x\n"
    "\n"
    "exp_back(g, out_y, u)  =  g * out_y  =  g * x\n"
    "log_back(g', out_u, x) =  g' / x\n"
    "\n"
    "Compose:  g_x = log_back(exp_back(g, x, log(x)), log(x), x)\n"
    "              = (g * x) / x  =  g\n"
    "```\n"
    "\n"
    "The product `out_exp / x_leaf` collapses to **1** — the back-fn pipeline "
    "produces the identity gradient, mirroring the analytic derivative."
)

RECAP_GETITEM_DEEP = (
    "## 2-D row indexing: `getitem_back` along axis 0 of `(N, D)` — quick refresher\n"
    "\n"
    "ex1 covered the 1-D case. The deeper facet is the SAME primitive "
    "(`index_add_`) on a higher-rank `x`: `out = x[idx]` where `x: (N, D)` and "
    "`idx: (K,)` returns a `(K, D)` slice of ROWS.\n"
    "\n"
    "```python\n"
    "x.shape   = (N, D)\n"
    "idx.shape = (K,)         # 1-D long tensor selecting rows\n"
    "out       = x[idx]       # shape (K, D)\n"
    "grad_out.shape = (K, D)\n"
    "\n"
    "grad_in = zeros_like(x)                   # (N, D)\n"
    "grad_in.index_add_(0, idx, grad_out)      # scatter-add full ROWS\n"
    "```\n"
    "\n"
    "Repeated indices still SUM — but now the contribution at each repeated "
    "position is a whole length-`D` row, not a scalar. The 1-D case is the "
    "`D=1` degenerate version of this."
)

RECAP_MATMUL_DEEP = (
    "## Batched `matmul_back`: `(B, m, k) @ (B, k, n)` — quick refresher\n"
    "\n"
    "ex1 covered the 2-D transpose pair. The deeper facet is what happens "
    "when there's a batch axis on the LEFT: every formula stays the same, but "
    "`.T` becomes `.transpose(-1, -2)` so it only swaps the LAST two axes.\n"
    "\n"
    "```\n"
    "x: (B, m, k),  y: (B, k, n),  out: (B, m, n)\n"
    "\n"
    "dL/dx must be (B, m, k)  →  grad_out @ y.transpose(-1, -2)\n"
    "                            : (B, m, n) @ (B, n, k) = (B, m, k)  ✓\n"
    "dL/dy must be (B, k, n)  →  x.transpose(-1, -2) @ grad_out\n"
    "                            : (B, k, m) @ (B, m, n) = (B, k, n)  ✓\n"
    "```\n"
    "\n"
    "Why `.transpose(-1, -2)` not `.T`. On a `(B, m, k)` tensor, `.T` "
    "reverses ALL axes → `(k, m, B)`, which is the wrong shape. "
    "`transpose(-1, -2)` swaps only the last two, leaving the batch alone — "
    "which is what every framework's batched matmul backward does."
)

RECAP_NEGATIVE_DEEP = (
    "## `negative_back` chained twice — quick refresher\n"
    "\n"
    "ex1 derived `negative_back = -grad_out`. The deeper facet: a graph "
    "containing TWO consecutive negations (e.g. `y = -(-x)`) must produce a "
    "leaf gradient identical to the seed.\n"
    "\n"
    "```\n"
    "u = -x         negative_back(g, _, x) = -g\n"
    "y = -u         negative_back(g, _, u) = -g\n"
    "\n"
    "g_x = negative_back(negative_back(g, _, u), _, x)\n"
    "    = -(-g)\n"
    "    = g\n"
    "```\n"
    "\n"
    "Each individual back fn flips sign; composing two flips recovers the "
    "original. This is the cleanest demonstration that back fns compose by "
    "ordinary function composition — there is no extra accumulation step "
    "between two single-parent ops in a chain."
)

RECAP_PERMUTE_DEEP = (
    "## `permute_back` round-trip invariant — quick refresher\n"
    "\n"
    "ex1 derived `inverse = argsort(dims)`. The deeper facet is the STRUCTURAL "
    "invariant this produces: for ANY valid permutation `dims`, applying "
    "`permute_back` to `grad_out = x.permute(*dims)` returns `x` itself.\n"
    "\n"
    "```\n"
    "x.permute(*dims).permute(*argsort(dims)) == x   # for every dims\n"
    "```\n"
    "\n"
    "Why this matters: if you forward then immediately backward through a "
    "permute, the gradient flowing into `x` is identical to the gradient that "
    "exited `out` — because `permute` is a pure axis shuffle with no value "
    "change, the backward is a pure inverse shuffle.\n"
    "\n"
    "Pinning the round-trip identity gives you a TEST that catches inverse-"
    "permutation bugs without needing to derive the gradient by hand."
)

RECAP_RESHAPE_DEEP = (
    "## `reshape_back` across the squeeze/unsqueeze family — quick refresher\n"
    "\n"
    "ex1 covered the canonical reshape `(2, 6) → (3, 4)`. The deeper facet: "
    "the size-1 axis manipulations — `squeeze`, `unsqueeze`, "
    "`reshape(N, 1) → (N,)` — are all special cases of `reshape_back`. None "
    "needs a separate back fn.\n"
    "\n"
    "```\n"
    "x.shape  = (N, 1)                          # column vector\n"
    "out_a    = x.reshape(N)                    # squeeze to 1-D\n"
    "out_b    = out_a.reshape(N, 1)             # unsqueeze back\n"
    "\n"
    "grad_back_a = reshape_back(g, out_a, x,     (N,))    # → (N, 1)\n"
    "grad_back_b = reshape_back(g, out_b, out_a, (N, 1))  # → (N,)\n"
    "```\n"
    "\n"
    "Both calls are `grad_out.reshape(x.shape)`. The same backward handles "
    "every view-op shape change. PyTorch's `view`, `flatten`, `squeeze`, "
    "`unsqueeze` all reduce to this primitive on the backward."
)

RECAP_SUM_DEEP = (
    "## Multi-axis `sum_back`: collapsing TWO axes — quick refresher\n"
    "\n"
    "ex1 collapsed a single dim. The deeper facet is `dim=(d0, d1)` — a "
    "multi-axis sum. The expand-broadcast pattern still works, but you must "
    "restore EACH collapsed axis as size 1 before expanding.\n"
    "\n"
    "```\n"
    "x.shape       = (2, 3, 4, 5)\n"
    "out = x.sum(dim=(0, 2))           # out.shape = (3, 5)\n"
    "grad_out.shape = (3, 5)\n"
    "\n"
    "# restore BOTH collapsed axes as size-1 (sorted ascending so indexing stays valid):\n"
    "g = grad_out.unsqueeze(0).unsqueeze(2)   # (1, 3, 1, 5)\n"
    "grad_in = g.expand(x.shape)              # (2, 3, 4, 5)\n"
    "```\n"
    "\n"
    "Why sort the dims ascending. `unsqueeze(0)` shifts every later axis by "
    "+1 — so if you insert at axis 0 first, axis 2 IN THE OUTPUT corresponds "
    "to axis 2 in the FINAL tensor (the next insertion target). Reverse "
    "order (insert at 2 first, then 0) also works but the bookkeeping is "
    "trickier — ascending is the standard convention."
)


# ---------------------------------------------------------------------------
# Spec helper.
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
# 1. add-sub-div-back-lambdas ex2 — dispatch through BACK in a 2-op chain
# =========================================================================

SPEC_LAMBDAS = _spec(
    atom_id="add-sub-div-back-lambdas",
    subtopic="Backprop: add/sub/div back as lambdas",
    recap=RECAP_LAMBDAS_DEEP,
    ex_title="dispatch through BACK for a 2-op mini reverse pass",
    slug="dispatch-through-back-for-two-op-mini-reverse-pass",
    bloom="Apply",
    difficulty_num=3,
    keywords=["dispatch", "back-dict", "reverse-pass", "argnum"],
    kcs=["arg-position-back-functions", "backward-fn-signature"],
    lo=(
        "Apply BACK-dict dispatch in a 2-op mini reverse pass: route grad_out "
        "through (op_name, argnum) lookups for `z = (x - y) / w`."
    ),
    prompt_body=(
        "Build a single dispatcher `mini_back(op_name, argnum, grad_out, out, "
        "x, y)` that looks up `BACK[(op_name, argnum)]` and calls it. Then "
        "use that dispatcher to walk the reverse pass of `z = (x - y) / w` "
        "and produce `dL/dx`, `dL/dy`, `dL/dw`.\n\n"
        "BACK contains the 6 lambdas from ex1 (`add`, `sub`, `div` x argnums "
        "0, 1). The dispatcher does NOT name any back fn — it always indexes "
        "into BACK.\n\n"
        "Algorithm for the 2-op chain `s = x - y; z = s / w` (no broadcasting):\n\n"
        "```python\n"
        "# 1. seed grad_out_z = ones_like(z)\n"
        "# 2. dispatch div argnum=0 → grad_s = mini_back('div', 0, grad_out_z, z, s, w)\n"
        "# 3. dispatch div argnum=1 → grad_w = mini_back('div', 1, grad_out_z, z, s, w)\n"
        "# 4. dispatch sub argnum=0 → grad_x = mini_back('sub', 0, grad_s, s, x, y)\n"
        "# 5. dispatch sub argnum=1 → grad_y = mini_back('sub', 1, grad_s, s, x, y)\n"
        "```\n\n"
        "Closed-form check (you'll see these in the tests):\n"
        "```\n"
        "dL/dx =  1/w           # +1 from sub * 1/w from div\n"
        "dL/dy = -1/w           # -1 from sub * 1/w from div\n"
        "dL/dw = -s / w**2      # -s/w**2 directly from div argnum=1\n"
        "```\n\n"
        "Implement `mini_back` AND `reverse_div_sub_chain(x, y, w)`. Return a "
        "dict `{'dx': ..., 'dy': ..., 'dw': ...}`. No autograd."
    ),
    stub=(
        "# BACK is provided — same 6 lambdas as ex1.\n"
        "BACK = {\n"
        "    ('add', 0): lambda g, o, x, y:  g,\n"
        "    ('add', 1): lambda g, o, x, y:  g,\n"
        "    ('sub', 0): lambda g, o, x, y:  g,\n"
        "    ('sub', 1): lambda g, o, x, y: -g,\n"
        "    ('div', 0): lambda g, o, x, y:  g / y,\n"
        "    ('div', 1): lambda g, o, x, y: -g * x / (y * y),\n"
        "}\n"
        "\n"
        "\n"
        "def mini_back(op_name: str, argnum: int, grad_out, out, x, y):\n"
        '    """Look up and call BACK[(op_name, argnum)]. No hardcoding."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def reverse_div_sub_chain(x, y, w):\n"
        '    """Walk the reverse pass of z = (x - y) / w. Return dict dx/dy/dw."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- dispatcher routes through BACK ---\n"
        "x = t.tensor([1.0, 2.0])\n"
        "y = t.tensor([3.0, 4.0])\n"
        "g = t.ones(2)\n"
        "out_add = x + y\n"
        "assert t.allclose(mini_back('add', 0, g, out_add, x, y), g)\n"
        "assert t.allclose(mini_back('add', 1, g, out_add, x, y), g)\n"
        "out_sub = x - y\n"
        "assert t.allclose(mini_back('sub', 0, g, out_sub, x, y),  g)\n"
        "assert t.allclose(mini_back('sub', 1, g, out_sub, x, y), -g)\n"
        "\n"
        "# --- dispatcher re-reads BACK each call (proves no hardcoding) ---\n"
        "saved = BACK[('add', 0)]\n"
        "BACK[('add', 0)] = lambda g, o, x, y: g * 7.0\n"
        "try:\n"
        "    res = mini_back('add', 0, g, x + y, x, y)\n"
        "    assert t.allclose(res, g * 7.0), f'dispatcher must re-read BACK: {res}'\n"
        "finally:\n"
        "    BACK[('add', 0)] = saved\n"
        "\n"
        "# --- 2-op chain: z = (x - y) / w ---\n"
        "x = t.tensor([6.0, 10.0, 14.0])\n"
        "y = t.tensor([2.0,  4.0,  6.0])\n"
        "w = t.tensor([2.0,  2.0,  4.0])\n"
        "grads = reverse_div_sub_chain(x, y, w)\n"
        "assert set(grads.keys()) == {'dx', 'dy', 'dw'}, f'keys: {grads.keys()}'\n"
        "\n"
        "# Closed-form: with grad_out_z = ones:\n"
        "expected_dx =  1.0 / w\n"
        "expected_dy = -1.0 / w\n"
        "s = x - y\n"
        "expected_dw = -s / (w * w)\n"
        "assert t.allclose(grads['dx'], expected_dx), f'dx: {grads[\"dx\"]} vs {expected_dx}'\n"
        "assert t.allclose(grads['dy'], expected_dy), f'dy: {grads[\"dy\"]} vs {expected_dy}'\n"
        "assert t.allclose(grads['dw'], expected_dw), f'dw: {grads[\"dw\"]} vs {expected_dw}'\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_r = x.clone().requires_grad_(True)\n"
        "y_r = y.clone().requires_grad_(True)\n"
        "w_r = w.clone().requires_grad_(True)\n"
        "((x_r - y_r) / w_r).sum().backward()\n"
        "assert t.allclose(grads['dx'], x_r.grad, atol=1e-6)\n"
        "assert t.allclose(grads['dy'], y_r.grad, atol=1e-6)\n"
        "assert t.allclose(grads['dw'], w_r.grad, atol=1e-6)\n"
        "\n"
        "# --- swap BACK[('div', 1)] to scaled — chain output must REFLECT the swap ---\n"
        "saved_div1 = BACK[('div', 1)]\n"
        "BACK[('div', 1)] = lambda g, o, a, b: 2.0 * (-g * a / (b * b))\n"
        "try:\n"
        "    grads2 = reverse_div_sub_chain(x, y, w)\n"
        "    assert t.allclose(grads2['dw'], 2.0 * expected_dw), (\n"
        "        f'chain did not dispatch through BACK on call to div argnum=1: '\n"
        "        f'got {grads2[\"dw\"]}, expected {2.0 * expected_dw}'\n"
        "    )\n"
        "finally:\n"
        "    BACK[('div', 1)] = saved_div1"
    ),
    solution_body=(
        "def mini_back(op_name: str, argnum: int, grad_out, out, x, y):\n"
        "    return BACK[(op_name, argnum)](grad_out, out, x, y)\n"
        "\n"
        "\n"
        "def reverse_div_sub_chain(x, y, w):\n"
        "    s = x - y\n"
        "    z = s / w\n"
        "    grad_z = t.ones_like(z)\n"
        "    grad_s = mini_back('div', 0, grad_z, z, s, w)\n"
        "    grad_w = mini_back('div', 1, grad_z, z, s, w)\n"
        "    grad_x = mini_back('sub', 0, grad_s, s, x, y)\n"
        "    grad_y = mini_back('sub', 1, grad_s, s, x, y)\n"
        "    return {'dx': grad_x, 'dy': grad_y, 'dw': grad_w}"
    ),
    solution_notes=(
        "**Why route through BACK every call.** In the test above, "
        "re-binding `BACK[('div', 1)]` to a doubled lambda must change "
        "`grads2['dw']`. If `reverse_div_sub_chain` had hardcoded "
        "`-grad_z * s / (w * w)` inline, the swap would do nothing — proof "
        "that dispatch-as-data is the right abstraction.\n\n"
        "**Why two separate dispatches for div.** `(div, 0)` and `(div, 1)` "
        "are NOT the same function — symmetric ops like `add` happen to "
        "share a body, but `div` doesn't. The dispatcher cannot collapse "
        "them; it must look up each argnum independently.\n\n"
        "**Chain shape.** `grad_s` is the intermediate gradient feeding back "
        "into the sub op. It is itself the OUTPUT of div's argnum=0 back "
        "fn — exactly how `Recipe.parents` would route it in a real "
        "reverse pass."
    ),
)


# =========================================================================
# 2. exp-back ex2 — compose with log_back: derive identity on exp(log(x))
# =========================================================================

SPEC_EXP = _spec(
    atom_id="exp-back",
    subtopic="Backprop: exp_back",
    recap=RECAP_EXP_DEEP,
    ex_title="compose log_back ∘ exp_back — recover identity on exp(log(x))",
    slug="compose-log-back-exp-back-identity-on-exp-log",
    bloom="Apply",
    difficulty_num=3,
    keywords=["compose", "chain-rule", "inverse-ops", "identity"],
    kcs=["chain-rule-elementwise", "back-fn-uses-cached-out"],
    lo=(
        "Apply two-step chain composition: compose log_back and exp_back to "
        "verify that the gradient through exp(log(x)) equals the seed grad_out."
    ),
    prompt_body=(
        "Implement TWO back fns plus the composed chain:\n\n"
        "1. `exp_back(grad_out, out, x)` — `grad_out * out` (cached out reuse).\n"
        "2. `log_back(grad_out, out, x)` — `grad_out / x`.\n"
        "3. `chain_exp_of_log(grad_out, x_leaf)` — walk the reverse pass of "
        "`y = exp(log(x_leaf))` using your two back fns. Return the leaf grad.\n\n"
        "The pipeline:\n"
        "```\n"
        "u    = log(x_leaf)                              # forward\n"
        "y    = exp(u)                                   # forward\n"
        "g_u  = exp_back(grad_out, y, u)                 # = grad_out * y\n"
        "g_x  = log_back(g_u, u, x_leaf)                 # = g_u / x_leaf\n"
        "       = grad_out * y / x_leaf\n"
        "       = grad_out                               # because y = x_leaf\n"
        "```\n\n"
        "The point: the final leaf grad must equal `grad_out` exactly — the "
        "two-step chain through inverse ops collapses to the identity. This is "
        "the cleanest demonstration that back fns COMPOSE by ordinary function "
        "composition (no extra accumulation in a single-parent chain).\n\n"
        "Tests verify:\n"
        "- both individual back fns,\n"
        "- the composed chain returns `grad_out` for positive `x_leaf` of "
        "varying values and grad_out of varying values,\n"
        "- agreement with torch.autograd on `exp(log(x)).sum()`.\n\n"
        "No autograd inside your implementation."
    ),
    stub=(
        "def exp_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """dL/dx for out = exp(x). Use cached out."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """dL/dx for out = log(x). out is unused; grad / x."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def chain_exp_of_log(grad_out: Tensor, x_leaf: Tensor) -> Tensor:\n"
        '    """Walk reverse of y = exp(log(x_leaf)). Returns dL/d(x_leaf)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- individual back fns sanity ---\n"
        "x = t.tensor([1.0, 2.0, 3.0])\n"
        "out_e = t.exp(x)\n"
        "assert t.allclose(exp_back(t.ones(3), out_e, x), out_e)\n"
        "out_l = t.log(x)\n"
        "assert t.allclose(log_back(t.ones(3), out_l, x), 1.0 / x)\n"
        "\n"
        "# --- chain: leaf grad must equal grad_out ---\n"
        "x_leaf = t.tensor([0.5, 1.0, 2.0, 4.0])\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0, 1.0])\n"
        "g_x = chain_exp_of_log(grad_out, x_leaf)\n"
        "assert g_x.shape == x_leaf.shape, f'shape: {g_x.shape}'\n"
        "assert t.allclose(g_x, grad_out, atol=1e-5), (\n"
        "    f'chain through exp(log(x)) should give grad_out, got {g_x}'\n"
        ")\n"
        "\n"
        "# --- non-unit grad_out ---\n"
        "grad_out = t.tensor([3.0, -5.0, 0.5, 2.0])\n"
        "g_x = chain_exp_of_log(grad_out, x_leaf)\n"
        "assert t.allclose(g_x, grad_out, atol=1e-5), (\n"
        "    f'non-unit grad_out through chain: {g_x} vs {grad_out}'\n"
        ")\n"
        "\n"
        "# --- varying scale of x_leaf — chain still gives identity ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "x_leaf = t.rand(20, generator=rng) * 100.0 + 0.01\n"
        "grad_out = t.randn(20, generator=rng)\n"
        "g_x = chain_exp_of_log(grad_out, x_leaf)\n"
        "assert t.allclose(g_x, grad_out, atol=1e-3), (\n"
        "    f'identity broken at scale: max diff {(g_x - grad_out).abs().max()}'\n"
        ")\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = t.tensor([0.3, 1.5, 4.7], requires_grad=True)\n"
        "y = t.exp(t.log(x_ref)).sum()\n"
        "y.backward()\n"
        "g_ours = chain_exp_of_log(t.ones(3), x_ref.detach())\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-5), (\n"
        "    f'disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def exp_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    return grad_out * out\n"
        "\n"
        "\n"
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    return grad_out / x\n"
        "\n"
        "\n"
        "def chain_exp_of_log(grad_out: Tensor, x_leaf: Tensor) -> Tensor:\n"
        "    # forward (cached for back-fn use)\n"
        "    u = t.log(x_leaf)\n"
        "    y = t.exp(u)\n"
        "    # reverse: exp_back, then log_back\n"
        "    g_u = exp_back(grad_out, y, u)\n"
        "    g_x = log_back(g_u, u, x_leaf)\n"
        "    return g_x"
    ),
    solution_notes=(
        "**Why the chain collapses to identity.** `exp_back` multiplies by "
        "`y = exp(log(x)) = x`, then `log_back` divides by `x`. The "
        "multiplications cancel: `grad_out * x / x = grad_out`. This is the "
        "back-fn-level mirror of the analytic fact that `d/dx exp(log(x)) = 1`.\n\n"
        "**Why cache `y` in the chain.** `exp_back` needs the cached `out` of "
        "the exp call. If you wrote `g_u = exp_back(grad_out, t.exp(u), u)` "
        "you'd recompute exp redundantly. Saving `y` once on the forward "
        "matches what `Recipe.args` would do in a real autograd graph.\n\n"
        "**Numerical drift.** The composition isn't bit-exact — `log(x)` "
        "followed by `exp(.)` introduces one rounding step. The `atol=1e-3` "
        "on the random-scale test accounts for it; at unit scale, `atol=1e-5` "
        "is fine."
    ),
)


# =========================================================================
# 3. getitem-back-add-at ex2 — 2-D row indexing
# =========================================================================

SPEC_GETITEM = _spec(
    atom_id="getitem-back-add-at",
    subtopic="Backprop: getitem_back via add-at",
    recap=RECAP_GETITEM_DEEP,
    ex_title="getitem_back along axis 0 of (N, D) — row scatter-add",
    slug="getitem-back-row-scatter-add-on-2d",
    bloom="Apply",
    difficulty_num=3,
    keywords=["getitem", "2d", "row-index", "index-add", "scatter-add"],
    kcs=["getitem-backward-pattern", "scatter-add-for-repeated-indices"],
    lo=(
        "Apply 2-D row scatter-add: for x: (N, D) and idx: (K,), use "
        "index_add_ along axis 0 to accumulate full-row contributions."
    ),
    prompt_body=(
        "Implement `getitem_back_rows(grad_out, out, x, idx)` for the "
        "forward op `out = x[idx]` where:\n\n"
        "- `x`     shape `(N, D)` — embedding table.\n"
        "- `idx`   shape `(K,)`   — `torch.LongTensor` of row indices.\n"
        "- `out`   shape `(K, D)` — gathered rows.\n"
        "- `grad_out` shape `(K, D)`.\n\n"
        "Derivation:\n"
        "- `out[i, :] = x[idx[i], :]`, so `d(out[i, :]) / d(x[j, :]) = I_D` "
        "if `j == idx[i]` else `0`.\n"
        "- Chain rule: `dL/dx[j, :] = sum_{i : idx[i] == j} grad_out[i, :]`.\n"
        "- Each contribution is a FULL ROW (length D), not a scalar.\n\n"
        "Implementation:\n"
        "1. `grad_in = torch.zeros_like(x)` — shape `(N, D)`.\n"
        "2. `grad_in.index_add_(0, idx, grad_out)` — scatter-adds the K rows "
        "of `grad_out` into the rows of `grad_in` selected by `idx`.\n"
        "3. Return `grad_in`.\n\n"
        "**This is the embedding-layer backward.** Token embeddings, "
        "positional embeddings, anywhere you do `x[idx]` to gather rows — "
        "the backward is this same scatter-add.\n\n"
        "No autograd. Return a `(N, D)` tensor with the same dtype as `x`."
    ),
    stub=(
        "def getitem_back_rows(grad_out: Tensor, out: Tensor, x: Tensor, idx: Tensor) -> Tensor:\n"
        '    """dL/dx for out = x[idx], where x is 2-D and idx is 1-D row indices."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- unique row indices ---\n"
        "x = t.tensor([\n"
        "    [1.0, 2.0, 3.0],\n"
        "    [4.0, 5.0, 6.0],\n"
        "    [7.0, 8.0, 9.0],\n"
        "    [10., 11., 12.],\n"
        "])  # (4, 3)\n"
        "idx = t.tensor([0, 2], dtype=t.long)\n"
        "out = x[idx]  # (2, 3)\n"
        "grad_out = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n"
        "g = getitem_back_rows(grad_out, out, x, idx)\n"
        "assert g.shape == x.shape, f'shape: {g.shape}'\n"
        "expected = t.tensor([\n"
        "    [1., 2., 3.],\n"
        "    [0., 0., 0.],\n"
        "    [4., 5., 6.],\n"
        "    [0., 0., 0.],\n"
        "])\n"
        "assert t.allclose(g, expected), f'unique: {g}'\n"
        "\n"
        "# --- repeated row index: contributions SUM as full rows ---\n"
        "idx = t.tensor([1, 1, 3], dtype=t.long)\n"
        "out = x[idx]\n"
        "grad_out = t.tensor([[1., 1., 1.], [2., 2., 2.], [10., 20., 30.]])\n"
        "g = getitem_back_rows(grad_out, out, x, idx)\n"
        "# row 1 gets [1+2, 1+2, 1+2] = [3, 3, 3]\n"
        "# row 3 gets [10, 20, 30]\n"
        "expected = t.tensor([\n"
        "    [0., 0., 0.],\n"
        "    [3., 3., 3.],\n"
        "    [0., 0., 0.],\n"
        "    [10., 20., 30.],\n"
        "])\n"
        "assert t.allclose(g, expected), f'repeated rows: {g}'\n"
        "\n"
        "# --- all-same row index ---\n"
        "idx = t.tensor([2, 2, 2, 2], dtype=t.long)\n"
        "out = x[idx]\n"
        "grad_out = t.ones(4, 3)\n"
        "g = getitem_back_rows(grad_out, out, x, idx)\n"
        "# row 2 gets 4 * [1, 1, 1] = [4, 4, 4]\n"
        "assert t.allclose(g[2], t.full((3,), 4.0)), f'all-same row 2: {g[2]}'\n"
        "assert t.allclose(g[[0, 1, 3]], t.zeros(3, 3)), f'other rows nonzero: {g}'\n"
        "\n"
        "# --- conservation: g.sum() == grad_out.sum() ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(8, 5, generator=rng)\n"
        "IDX = t.randint(0, 8, (20,), generator=rng, dtype=t.long)\n"
        "G = t.randn(20, 5, generator=rng)\n"
        "g = getitem_back_rows(G, X[IDX], X, IDX)\n"
        "assert g.shape == (8, 5)\n"
        "assert abs(g.sum().item() - G.sum().item()) < 1e-4, (\n"
        "    f'conservation broken: g.sum={g.sum()} grad_out.sum={G.sum()}'\n"
        ")\n"
        "\n"
        "# --- per-column conservation ---\n"
        "for col in range(5):\n"
        "    assert abs(g[:, col].sum().item() - G[:, col].sum().item()) < 1e-4\n"
        "\n"
        "# --- not aliased to grad_out ---\n"
        "g_in = t.ones(2, 3)\n"
        "g_out = getitem_back_rows(g_in, x[t.tensor([0, 1])], x, t.tensor([0, 1], dtype=t.long))\n"
        "assert g_out.data_ptr() != g_in.data_ptr(), 'must not alias grad_out'\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = t.randn(6, 4, requires_grad=True, generator=t.Generator().manual_seed(2))\n"
        "idx_ref = t.tensor([0, 2, 0, 5, 2], dtype=t.long)\n"
        "y = x_ref[idx_ref].sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "g_ours = getitem_back_rows(t.ones(5, 4), x_det[idx_ref], x_det, idx_ref)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'disagrees with autograd: max diff '\n"
        "    f'{(g_ours - x_ref.grad).abs().max()}'\n"
        ")"
    ),
    solution_body=(
        "def getitem_back_rows(grad_out: Tensor, out: Tensor, x: Tensor, idx: Tensor) -> Tensor:\n"
        "    grad_in = t.zeros_like(x)\n"
        "    # axis 0 scatter — each grad_out row is added to row idx[i] of grad_in.\n"
        "    grad_in.index_add_(0, idx, grad_out)\n"
        "    return grad_in"
    ),
    solution_notes=(
        "**Why `index_add_` not `scatter_add_`.** Both work, but `index_add_` "
        "with `dim=0` and 1-D `idx` is the row-scatter primitive — its "
        "semantics map 1:1 to the math here. `scatter_add_` needs an `index` "
        "tensor of the same shape as `grad_out`, which is overkill.\n\n"
        "**Why this is the embedding-layer backward.** A token embedding is "
        "exactly `x[idx]` where `x: (vocab_size, embed_dim)`. Every NLP model "
        "uses this same back fn to accumulate token-level gradients into the "
        "embedding table. Repeated tokens in a sequence get summed.\n\n"
        "**Conservation per column.** Each element of `grad_out` lands "
        "somewhere in `grad_in`, so column sums match. Useful debugging "
        "invariant when shapes get high-rank."
    ),
)


# =========================================================================
# 4. matmul-back-transpose-pair ex2 — batched matmul backward
# =========================================================================

SPEC_MATMUL = _spec(
    atom_id="matmul-back-transpose-pair",
    subtopic="Backprop: matmul_back transpose pair",
    recap=RECAP_MATMUL_DEEP,
    ex_title="batched matmul_back — transpose(-1, -2) on the OTHER input",
    slug="batched-matmul-back-transpose-last-two-axes",
    bloom="Apply",
    difficulty_num=3,
    keywords=["matmul", "batched", "transpose", "broadcasting"],
    kcs=["matmul-backward-pattern", "arg-position-back-functions"],
    lo=(
        "Apply the matmul transpose-pair pattern to batched inputs (B, m, k) "
        "@ (B, k, n) using transpose(-1, -2) on the partner input."
    ),
    prompt_body=(
        "Implement TWO back fns for batched matmul `out = x @ y` where:\n"
        "- `x` shape `(B, m, k)`\n"
        "- `y` shape `(B, k, n)`\n"
        "- `out` shape `(B, m, n)`\n\n"
        "**1. `bmm_back0(grad_out, out, x, y)`** — gradient w.r.t. `x`.\n"
        "   - Target shape `(B, m, k)`.\n"
        "   - `grad_out @ y.transpose(-1, -2)`: `(B, m, n) @ (B, n, k) = "
        "(B, m, k)`. ✓\n\n"
        "**2. `bmm_back1(grad_out, out, x, y)`** — gradient w.r.t. `y`.\n"
        "   - Target shape `(B, k, n)`.\n"
        "   - `x.transpose(-1, -2) @ grad_out`: `(B, k, m) @ (B, m, n) = "
        "(B, k, n)`. ✓\n\n"
        "**Why `.transpose(-1, -2)` and not `.T`.** On a 3-D tensor, `.T` "
        "reverses ALL axes — `(B, m, k)` would become `(k, m, B)`, which is "
        "the wrong shape AND scrambles the batch axis. `transpose(-1, -2)` "
        "swaps only the last two; the batch axis stays put.\n\n"
        "Use `@` / `t.matmul`. Return tensors with the correct shapes. No "
        "autograd. The ex1 2-D transpose-pair is the `B=1` special case."
    ),
    stub=(
        "def bmm_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dx for out = x @ y (batched). Shape (B, m, k)."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def bmm_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dy for out = x @ y (batched). Shape (B, k, n)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- small batched: (2, 3, 4) @ (2, 4, 5) = (2, 3, 5) ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "x = t.randn(2, 3, 4, generator=rng)\n"
        "y = t.randn(2, 4, 5, generator=rng)\n"
        "out = x @ y\n"
        "assert out.shape == (2, 3, 5)\n"
        "grad_out = t.randn(2, 3, 5, generator=rng)\n"
        "g0 = bmm_back0(grad_out, out, x, y)\n"
        "g1 = bmm_back1(grad_out, out, x, y)\n"
        "assert g0.shape == (2, 3, 4), f'g0 shape: {g0.shape}'\n"
        "assert g1.shape == (2, 4, 5), f'g1 shape: {g1.shape}'\n"
        "assert t.allclose(g0, grad_out @ y.transpose(-1, -2))\n"
        "assert t.allclose(g1, x.transpose(-1, -2) @ grad_out)\n"
        "\n"
        "# --- per-batch independence: results match per-batch 2-D matmul_back ---\n"
        "g0_manual = t.stack([grad_out[b] @ y[b].T for b in range(2)])\n"
        "g1_manual = t.stack([x[b].T @ grad_out[b] for b in range(2)])\n"
        "assert t.allclose(g0, g0_manual), 'per-batch g0 mismatch'\n"
        "assert t.allclose(g1, g1_manual), 'per-batch g1 mismatch'\n"
        "\n"
        "# --- larger batch ---\n"
        "x = t.randn(5, 7, 3, generator=rng)\n"
        "y = t.randn(5, 3, 8, generator=rng)\n"
        "out = x @ y\n"
        "grad_out = t.randn(5, 7, 8, generator=rng)\n"
        "g0 = bmm_back0(grad_out, out, x, y)\n"
        "g1 = bmm_back1(grad_out, out, x, y)\n"
        "assert g0.shape == (5, 7, 3)\n"
        "assert g1.shape == (5, 3, 8)\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = t.randn(3, 4, 5, requires_grad=True, generator=t.Generator().manual_seed(7))\n"
        "y_ref = t.randn(3, 5, 6, requires_grad=True, generator=t.Generator().manual_seed(8))\n"
        "z = (x_ref @ y_ref).sum()\n"
        "z.backward()\n"
        "x_det, y_det = x_ref.detach(), y_ref.detach()\n"
        "out_cached = x_det @ y_det\n"
        "g0_ours = bmm_back0(t.ones(3, 4, 6), out_cached, x_det, y_det)\n"
        "g1_ours = bmm_back1(t.ones(3, 4, 6), out_cached, x_det, y_det)\n"
        "assert t.allclose(g0_ours, x_ref.grad, atol=1e-5)\n"
        "assert t.allclose(g1_ours, y_ref.grad, atol=1e-5)\n"
        "\n"
        "# --- using .T instead of transpose(-1, -2) would crash or scramble ---\n"
        "# Sanity: confirm y.transpose(-1, -2) has shape (B, n, k), NOT (n, k, B).\n"
        "assert y.transpose(-1, -2).shape == (5, 8, 3)\n"
        "# y.T on a 3-D tensor returns shape (n, k, B) — provably wrong for our case.\n"
        "# (Just assert the right shape was used; we don't run the .T variant.)\n"
        "assert g0.shape[0] == x.shape[0], 'batch axis must be preserved'"
    ),
    solution_body=(
        "def bmm_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # (B, m, n) @ (B, n, k) = (B, m, k).\n"
        "    return grad_out @ y.transpose(-1, -2)\n"
        "\n"
        "\n"
        "def bmm_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # (B, k, m) @ (B, m, n) = (B, k, n).\n"
        "    return x.transpose(-1, -2) @ grad_out"
    ),
    solution_notes=(
        "**Why `.transpose(-1, -2)` is the batched analog of `.T`.** On a "
        "2-D tensor `.T` swaps the only two axes — but on a 3-D tensor it "
        "reverses ALL axes. `transpose(-1, -2)` is the explicit form: swap "
        "EXACTLY the matmul axes, leaving every leading axis alone. This is "
        "what `torch.matmul`'s own backward uses.\n\n"
        "**Per-batch independence is the structural invariant.** Batched "
        "matmul is just B independent 2-D matmuls run in parallel. The test "
        "that stacks per-batch results pins this down — if you accidentally "
        "summed across the batch axis, that assertion would fail.\n\n"
        "**Where this generalizes.** Attention's `Q @ K.T` and `attn @ V` are "
        "both batched matmuls — over heads, over batch, sometimes over "
        "sequence prefixes. The transpose-the-OTHER-input rule survives "
        "every such generalization."
    ),
)


# =========================================================================
# 5. negative-back ex2 — double-negate compose
# =========================================================================

SPEC_NEGATIVE = _spec(
    atom_id="negative-back",
    subtopic="Backprop: negative_back",
    recap=RECAP_NEGATIVE_DEEP,
    ex_title="compose negative_back twice — recover grad_out through y = -(-x)",
    slug="compose-negative-back-twice-roundtrip-identity",
    bloom="Apply",
    difficulty_num=2,
    keywords=["compose", "double-negate", "chain", "identity"],
    kcs=["chain-rule-elementwise", "backward-fn-signature"],
    lo=(
        "Apply two-step composition of negative_back through y = -(-x) and "
        "verify the leaf gradient equals grad_out."
    ),
    prompt_body=(
        "Implement `negative_back(grad_out, out, x)` AND "
        "`chain_double_negate(grad_out, x_leaf)` — a tiny reverse pass over "
        "`y = -(-x_leaf)`.\n\n"
        "Pipeline:\n"
        "```\n"
        "u    = -x_leaf                           # forward\n"
        "y    = -u                                # forward\n"
        "g_u  = negative_back(grad_out, y, u)     # = -grad_out\n"
        "g_x  = negative_back(g_u, u, x_leaf)     # = -(-grad_out) = grad_out\n"
        "```\n\n"
        "The point: two single-parent back fns COMPOSE by ordinary function "
        "composition — no extra dispatcher, no parent-grads accumulation. "
        "The two sign flips cancel and the leaf grad is `grad_out`.\n\n"
        "Tests verify:\n"
        "- single-step `negative_back` for scalar, vector, matrix,\n"
        "- the two-step chain returns `grad_out` exactly (no atol slack),\n"
        "- the chain works for varying shapes,\n"
        "- agreement with torch.autograd on `(-(-x)).sum()`.\n\n"
        "No autograd inside your implementation."
    ),
    stub=(
        "def negative_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """dL/dx for out = -x."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def chain_double_negate(grad_out: Tensor, x_leaf: Tensor) -> Tensor:\n"
        '    """Walk reverse of y = -(-x_leaf). Returns dL/d(x_leaf) = grad_out."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- single-step sanity ---\n"
        "x = t.tensor([1.0, -2.0, 3.0])\n"
        "g = negative_back(t.tensor([4.0, 5.0, 6.0]), -x, x)\n"
        "assert t.allclose(g, t.tensor([-4.0, -5.0, -6.0]))\n"
        "\n"
        "# --- two-step chain: must return grad_out EXACTLY ---\n"
        "x_leaf = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "grad_out = t.tensor([10.0, 20.0, 30.0, 40.0])\n"
        "g_x = chain_double_negate(grad_out, x_leaf)\n"
        "assert g_x.shape == x_leaf.shape\n"
        "# Sign flips cancel: result is EXACTLY grad_out (no floating-point drift\n"
        "# because negation is exact in floats).\n"
        "assert t.equal(g_x, grad_out), f'two flips should give grad_out: {g_x}'\n"
        "\n"
        "# --- varying shapes ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "for shape in [(5,), (3, 4), (2, 3, 4)]:\n"
        "    x_leaf = t.randn(*shape, generator=rng)\n"
        "    grad_out = t.randn(*shape, generator=rng)\n"
        "    g_x = chain_double_negate(grad_out, x_leaf)\n"
        "    assert g_x.shape == shape, f'shape: {g_x.shape} for {shape}'\n"
        "    assert t.equal(g_x, grad_out), f'identity broken at shape {shape}'\n"
        "\n"
        "# --- one flip is NOT identity (sanity: prove we actually compose two) ---\n"
        "x_leaf = t.tensor([1.0, 2.0])\n"
        "grad_out = t.tensor([3.0, 5.0])\n"
        "g_one = negative_back(grad_out, -x_leaf, x_leaf)\n"
        "assert not t.allclose(g_one, grad_out), 'one flip must not be identity'\n"
        "g_two = chain_double_negate(grad_out, x_leaf)\n"
        "assert t.allclose(g_two, grad_out), 'two flips must be identity'\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = t.tensor([0.5, -1.5, 2.7], requires_grad=True)\n"
        "y = -(-x_ref)\n"
        "y.sum().backward()\n"
        "g_ours = chain_double_negate(t.ones(3), x_ref.detach())\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-7), (\n"
        "    f'disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def negative_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    return -grad_out\n"
        "\n"
        "\n"
        "def chain_double_negate(grad_out: Tensor, x_leaf: Tensor) -> Tensor:\n"
        "    u = -x_leaf\n"
        "    y = -u\n"
        "    g_u = negative_back(grad_out, y, u)\n"
        "    g_x = negative_back(g_u, u, x_leaf)\n"
        "    return g_x"
    ),
    solution_notes=(
        "**Why this composes exactly (no drift).** Float negation flips the "
        "sign bit — it's bit-exact. Two flips return the identical bit "
        "pattern, so `t.equal(g_x, grad_out)` (not just `allclose`) passes. "
        "Compare with `chain_exp_of_log` from the exp-back ex2, which "
        "accumulates rounding error.\n\n"
        "**Why no accumulation between steps.** Each MiniTensor has ONE "
        "parent in this chain — `u` only feeds `y`, `x_leaf` only feeds "
        "`u`. The accumulation step in a real reverse pass only fires when a "
        "tensor has multiple downstream consumers. Single-parent chains "
        "are pure function composition.\n\n"
        "**Why we still write the chain explicitly.** Could we hand-wave "
        "and just return `grad_out`? Yes. But this is the SHAPE every "
        "longer reverse pass has — exec each back fn, thread the output "
        "into the next as `grad_out`. Practicing on a 2-step trivial case "
        "makes the 10-step nontrivial case mechanical."
    ),
)


# =========================================================================
# 6. permute-back-argsort ex2 — round-trip structural invariant
# =========================================================================

SPEC_PERMUTE = _spec(
    atom_id="permute-back-argsort",
    subtopic="Backprop: permute_back via argsort",
    recap=RECAP_PERMUTE_DEEP,
    ex_title="permute round-trip invariant — verify forward∘backward = identity",
    slug="permute-round-trip-forward-backward-identity",
    bloom="Analyze",
    difficulty_num=3,
    keywords=["permute", "round-trip", "invariant", "argsort", "structural"],
    kcs=["permute-backward-pattern", "inverse-permutation-via-argsort"],
    lo=(
        "Analyze the structural invariant: x.permute(*dims).permute("
        "*argsort(dims)) == x for every valid permutation, and use it to "
        "verify permute_back."
    ),
    prompt_body=(
        "Implement TWO pieces:\n\n"
        "1. **`permute_back(grad_out, out, x, dims)`** — apply the inverse "
        "permutation via `argsort`.\n\n"
        "2. **`assert_permute_round_trip(x, dims)`** — a property checker. "
        "Compute `forward = x.permute(*dims)`, then "
        "`back_to_x = permute_back(forward, forward, x, dims)`. Assert "
        "`back_to_x.shape == x.shape` AND `t.equal(back_to_x, x)` (bit-"
        "exact — permute does no arithmetic). Raise `AssertionError` with a "
        "message naming `dims` on failure.\n\n"
        "**Why analyze rather than apply.** ex1 had you derive the back fn. "
        "ex2 makes you EXAMINE the structural property that derivation "
        "produces: applying `permute(*dims)` then `permute(*argsort(dims))` "
        "is a NO-OP for ANY valid `dims`. Pinning this invariant in code "
        "gives you a property-based test that catches inverse-permutation "
        "bugs WITHOUT having to derive the gradient by hand on a case-by-"
        "case basis.\n\n"
        "The test runs `assert_permute_round_trip` against every "
        "permutation of `(0, 1, 2)` and several 4-D permutations.\n\n"
        "No autograd."
    ),
    stub=(
        "def permute_back(grad_out: Tensor, out: Tensor, x: Tensor, dims: tuple) -> Tensor:\n"
        '    """dL/dx for out = x.permute(*dims). Inverse-perm via argsort."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def assert_permute_round_trip(x: Tensor, dims: tuple) -> None:\n"
        '    """Verify x.permute(*dims) then permute_back gives x bit-exactly."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- permute_back sanity (cycle of 3) ---\n"
        "x = t.arange(24.0).reshape(2, 3, 4)\n"
        "out = x.permute(2, 0, 1)\n"
        "g = permute_back(out, out, x, (2, 0, 1))\n"
        "assert g.shape == x.shape\n"
        "assert t.equal(g, x), f'round-trip on (2,0,1) broken: max diff {(g - x).abs().max()}'\n"
        "\n"
        "# --- round-trip on every perm of (0, 1, 2) ---\n"
        "from itertools import permutations\n"
        "x3 = t.randn(2, 3, 4, generator=t.Generator().manual_seed(0))\n"
        "for dims in permutations((0, 1, 2)):\n"
        "    assert_permute_round_trip(x3, dims)\n"
        "\n"
        "# --- 4-D arbitrary permutations ---\n"
        "x4 = t.randn(2, 3, 4, 5, generator=t.Generator().manual_seed(1))\n"
        "for dims in [(3, 1, 0, 2), (2, 3, 0, 1), (0, 3, 2, 1), (1, 0, 3, 2)]:\n"
        "    assert_permute_round_trip(x4, dims)\n"
        "\n"
        "# --- identity perm trivially round-trips ---\n"
        "assert_permute_round_trip(x3, (0, 1, 2))\n"
        "assert_permute_round_trip(x4, (0, 1, 2, 3))\n"
        "\n"
        "# --- assert_permute_round_trip RAISES if permute_back is wrong ---\n"
        "# We simulate a buggy permute_back by monkey-patching it temporarily.\n"
        "good_permute_back = permute_back\n"
        "def buggy_permute_back(g, o, x, dims):\n"
        "    # WRONG: re-apply forward dims instead of inverse.\n"
        "    return g.permute(*dims)\n"
        "globals()['permute_back'] = buggy_permute_back\n"
        "try:\n"
        "    raised = False\n"
        "    try:\n"
        "        assert_permute_round_trip(x3, (2, 0, 1))\n"
        "    except AssertionError as e:\n"
        "        raised = True\n"
        "        msg = str(e)\n"
        "        # Message should mention the dims that failed.\n"
        "        assert '(2, 0, 1)' in msg or '2, 0, 1' in msg, (\n"
        "            f'AssertionError should name dims that failed: {msg!r}'\n"
        "        )\n"
        "    assert raised, 'buggy permute_back should have failed round-trip'\n"
        "finally:\n"
        "    globals()['permute_back'] = good_permute_back\n"
        "\n"
        "# --- agreement with torch.autograd (4-D, non-trivial perm) ---\n"
        "x_ref = t.randn(2, 3, 4, 5, requires_grad=True, generator=t.Generator().manual_seed(2))\n"
        "dims = (2, 0, 3, 1)\n"
        "y = x_ref.permute(*dims).sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "ones_perm = t.ones_like(x_det.permute(*dims))\n"
        "g_ours = permute_back(ones_perm, x_det.permute(*dims), x_det, dims)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), 'disagrees with autograd'"
    ),
    solution_body=(
        "def permute_back(grad_out: Tensor, out: Tensor, x: Tensor, dims: tuple) -> Tensor:\n"
        "    inverse = tuple(int(i) for i in np.argsort(dims))\n"
        "    return grad_out.permute(*inverse)\n"
        "\n"
        "\n"
        "def assert_permute_round_trip(x: Tensor, dims: tuple) -> None:\n"
        "    forward = x.permute(*dims)\n"
        "    back_to_x = permute_back(forward, forward, x, dims)\n"
        "    assert back_to_x.shape == x.shape, (\n"
        "        f'shape broke on dims={dims}: got {back_to_x.shape}, expected {x.shape}'\n"
        "    )\n"
        "    assert t.equal(back_to_x, x), (\n"
        "        f'round-trip on dims={dims} failed: max diff {(back_to_x - x).abs().max()}'\n"
        "    )"
    ),
    solution_notes=(
        "**Why bit-exact equality, not allclose.** Permute reads from "
        "storage by stride — no arithmetic — so the values are identical "
        "bit patterns. `t.equal` is the right pin; `t.allclose` would mask "
        "permutation bugs that happen to produce nearby float values.\n\n"
        "**Why this is Analyze-Bloom.** ex1 was Apply: 'compute the "
        "inverse'. ex2 is Analyze: 'identify the structural property that "
        "MAKES the inverse correct'. The round-trip invariant is what "
        "lets you trust `argsort` without rederiving on every shape — and "
        "what catches monkey-patched bugs (as the test demonstrates).\n\n"
        "**Property-based testing in autograd.** Real frameworks (PyTorch, "
        "JAX) include round-trip tests like this in their CI — exactly "
        "because they catch implementation bugs that per-case assertion "
        "tests can miss. The pattern transfers."
    ),
)


# =========================================================================
# 7. reshape-back ex2 — squeeze/unsqueeze family
# =========================================================================

SPEC_RESHAPE = _spec(
    atom_id="reshape-back",
    subtopic="Backprop: reshape_back",
    recap=RECAP_RESHAPE_DEEP,
    ex_title="reshape_back across squeeze/unsqueeze — view-op family unification",
    slug="reshape-back-squeeze-unsqueeze-family",
    bloom="Apply",
    difficulty_num=2,
    keywords=["reshape", "squeeze", "unsqueeze", "view-op", "shape-restore"],
    kcs=["reshape-backward-pattern", "backward-fn-signature"],
    lo=(
        "Apply reshape_back uniformly to the squeeze/unsqueeze family: the "
        "same back fn handles (N, 1) ↔ (N,) ↔ (N, 1) round-trips."
    ),
    prompt_body=(
        "Implement `reshape_back(grad_out, out, x, new_shape)` AND a chain "
        "`squeeze_unsqueeze_chain(grad_out, x_leaf)` that walks the reverse "
        "pass of:\n\n"
        "```\n"
        "x_leaf.shape  = (N, 1)\n"
        "u    = x_leaf.reshape(N)            # squeeze\n"
        "y    = u.reshape(N, 1)              # unsqueeze\n"
        "```\n\n"
        "The chain calls `reshape_back` TWICE, once for each step. Both "
        "calls reduce to `grad_out.reshape(x.shape)` — the same primitive "
        "handles both squeeze and unsqueeze, because they're both pure "
        "view operations.\n\n"
        "Tests verify:\n"
        "- `reshape_back` works for `(N,) → (N, 1)` and `(N, 1) → (N,)` "
        "(both `unsqueeze`-like and `squeeze`-like reshapes),\n"
        "- the two-step chain returns `grad_out` reshaped to `(N, 1)` "
        "(matching `x_leaf.shape`),\n"
        "- agreement with torch.autograd on the equivalent chain.\n\n"
        "No autograd. The point is that `reshape_back` SUBSUMES `squeeze_back` "
        "and `unsqueeze_back` — no separate back fn needed."
    ),
    stub=(
        "def reshape_back(grad_out: Tensor, out: Tensor, x: Tensor, new_shape: tuple) -> Tensor:\n"
        '    """dL/dx for out = x.reshape(new_shape). Restore x.shape."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def squeeze_unsqueeze_chain(grad_out: Tensor, x_leaf: Tensor) -> Tensor:\n"
        '    """Reverse-pass through (N,1) → reshape(N) → reshape(N,1). Returns dL/d(x_leaf)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- reshape_back: (N,) → (N, 1) (unsqueeze-like reshape on the forward) ---\n"
        "x = t.tensor([1.0, 2.0, 3.0, 4.0])           # (4,)\n"
        "out = x.reshape(4, 1)\n"
        "grad_out = t.tensor([[10.], [20.], [30.], [40.]])\n"
        "g = reshape_back(grad_out, out, x, (4, 1))\n"
        "assert g.shape == (4,), f'shape: {g.shape}'\n"
        "assert t.allclose(g, t.tensor([10., 20., 30., 40.]))\n"
        "\n"
        "# --- reshape_back: (N, 1) → (N,) (squeeze-like reshape on the forward) ---\n"
        "x = t.tensor([[1.], [2.], [3.], [4.]])       # (4, 1)\n"
        "out = x.reshape(4)\n"
        "grad_out = t.tensor([10., 20., 30., 40.])\n"
        "g = reshape_back(grad_out, out, x, (4,))\n"
        "assert g.shape == (4, 1), f'shape: {g.shape}'\n"
        "assert t.allclose(g, t.tensor([[10.], [20.], [30.], [40.]]))\n"
        "\n"
        "# --- chain: (N, 1) → (N,) → (N, 1) leaves shape and values intact ---\n"
        "x_leaf = t.tensor([[1.], [2.], [3.], [4.]])  # (4, 1)\n"
        "grad_out = t.tensor([[5.], [6.], [7.], [8.]])\n"
        "g_x = squeeze_unsqueeze_chain(grad_out, x_leaf)\n"
        "assert g_x.shape == x_leaf.shape, f'shape: {g_x.shape}'\n"
        "# Bit-exact: reshape does no arithmetic, two reshapes round-trip values.\n"
        "assert t.equal(g_x, grad_out), f'chain broken: {g_x} vs {grad_out}'\n"
        "\n"
        "# --- varying N ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "for N in [1, 5, 32]:\n"
        "    x_leaf = t.randn(N, 1, generator=rng)\n"
        "    grad_out = t.randn(N, 1, generator=rng)\n"
        "    g_x = squeeze_unsqueeze_chain(grad_out, x_leaf)\n"
        "    assert g_x.shape == (N, 1)\n"
        "    assert t.equal(g_x, grad_out), f'chain broken at N={N}'\n"
        "\n"
        "# --- 3-D case: reshape_back on (2, 1, 4) ↔ (2, 4) ---\n"
        "x3 = t.randn(2, 1, 4, generator=rng)\n"
        "out3 = x3.reshape(2, 4)\n"
        "g3 = t.randn(2, 4, generator=rng)\n"
        "g3_back = reshape_back(g3, out3, x3, (2, 4))\n"
        "assert g3_back.shape == (2, 1, 4)\n"
        "assert t.allclose(g3_back.reshape(2, 4), g3)\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = t.randn(5, 1, requires_grad=True, generator=t.Generator().manual_seed(3))\n"
        "y = x_ref.reshape(5).reshape(5, 1).sum()\n"
        "y.backward()\n"
        "g_ours = squeeze_unsqueeze_chain(t.ones(5, 1), x_ref.detach())\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'disagrees with autograd: max diff '\n"
        "    f'{(g_ours - x_ref.grad).abs().max()}'\n"
        ")"
    ),
    solution_body=(
        "def reshape_back(grad_out: Tensor, out: Tensor, x: Tensor, new_shape: tuple) -> Tensor:\n"
        "    return grad_out.reshape(x.shape)\n"
        "\n"
        "\n"
        "def squeeze_unsqueeze_chain(grad_out: Tensor, x_leaf: Tensor) -> Tensor:\n"
        "    # forward (cached for back-fn use)\n"
        "    N = x_leaf.shape[0]\n"
        "    u = x_leaf.reshape(N)             # (N,)\n"
        "    y = u.reshape(N, 1)               # (N, 1)\n"
        "    # reverse\n"
        "    g_u = reshape_back(grad_out, y, u, (N, 1))\n"
        "    g_x = reshape_back(g_u, u, x_leaf, (N,))\n"
        "    return g_x"
    ),
    solution_notes=(
        "**Why the same back fn handles squeeze AND unsqueeze.** Both are "
        "pure view operations — same data, different shape interpretation. "
        "`grad_out.reshape(x.shape)` is the universal answer; the direction "
        "doesn't matter.\n\n"
        "**Why bit-exact through the chain.** Two reshapes are pure storage "
        "re-interpretations — no arithmetic, no rounding. `t.equal` (not "
        "`allclose`) pins this down. Compare with `exp ∘ log` where "
        "rounding does creep in.\n\n"
        "**Production note.** PyTorch's autograd has separate "
        "`SqueezeBackward` / `UnsqueezeBackward` / `ViewBackward` nodes for "
        "performance reasons (no shape-checking overhead), but the math is "
        "identical. In our MiniTensor where we want minimal surface area, "
        "one `reshape_back` covers the whole family."
    ),
)


# =========================================================================
# 8. sum-back-expand-broadcast ex2 — multi-axis sum
# =========================================================================

SPEC_SUM = _spec(
    atom_id="sum-back-expand-broadcast",
    subtopic="Backprop: sum_back via expand_broadcast",
    recap=RECAP_SUM_DEEP,
    ex_title="multi-axis sum_back — collapse two axes, restore both via unsqueeze chain",
    slug="sum-back-multi-axis-restore-via-unsqueeze-chain",
    bloom="Apply",
    difficulty_num=3,
    keywords=["sum", "multi-axis", "expand", "unsqueeze", "broadcast"],
    kcs=["sum-backward-pattern", "kwargs-pass-through-recipe"],
    lo=(
        "Apply expand-broadcast to multi-axis sum: restore EACH collapsed "
        "axis as size-1 via successive unsqueeze, then expand to x.shape."
    ),
    prompt_body=(
        "Implement `sum_back_multi(grad_out, out, x, dims, keepdim=False)` "
        "for the forward op `out = x.sum(dim=dims, keepdim=keepdim)` where "
        "`dims` is a TUPLE of ints (not a single int).\n\n"
        "Derivation:\n"
        "- Each element of `x` contributes to exactly ONE output entry.\n"
        "- Backward broadcasts `grad_out` back along EVERY collapsed axis.\n\n"
        "**`keepdim=False`** — `grad_out` is missing every dim in `dims`. "
        "Restore them as size-1 axes one at a time, then expand:\n"
        "```\n"
        "g = grad_out\n"
        "for d in sorted(dims):            # ascending so inserts don't shift each other\n"
        "    g = g.unsqueeze(d)\n"
        "grad_in = g.expand(x.shape)\n"
        "```\n\n"
        "**`keepdim=True`** — `grad_out` already has size-1 at every `d in "
        "dims`. Just `expand` directly:\n"
        "```\n"
        "grad_in = grad_out.expand(x.shape)\n"
        "```\n\n"
        "**Why sort ascending.** `unsqueeze(d)` shifts later axis positions "
        "by +1 — but only ones LATER than `d`. If you process dims in "
        "ascending order, each insertion happens at the same numerical "
        "index it had in the original `dims` tuple (because earlier "
        "insertions only added axes BEFORE it, leaving its index intact "
        "AS the next axis to insert at). Descending order also works but "
        "the bookkeeping inverts.\n\n"
        "Return a `torch.Tensor` with `x.shape`. No autograd."
    ),
    stub=(
        "def sum_back_multi(grad_out: Tensor, out: Tensor, x: Tensor, dims: tuple, keepdim: bool = False) -> Tensor:\n"
        '    """dL/dx for out = x.sum(dim=dims, keepdim=keepdim). dims is a tuple."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- single-dim case (sanity that the multi-axis fn subsumes single) ---\n"
        "x = t.arange(12.0).reshape(3, 4)\n"
        "out = x.sum(dim=(0,))\n"
        "grad_out = t.tensor([1., 2., 3., 4.])\n"
        "g = sum_back_multi(grad_out, out, x, dims=(0,), keepdim=False)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, grad_out.unsqueeze(0).expand(3, 4))\n"
        "\n"
        "# --- two-axis: dims=(0, 2) on shape (2, 3, 4, 5) ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(2, 3, 4, 5, generator=rng)\n"
        "OUT = X.sum(dim=(0, 2))                # (3, 5)\n"
        "G = t.randn(3, 5, generator=rng)\n"
        "g = sum_back_multi(G, OUT, X, dims=(0, 2), keepdim=False)\n"
        "assert g.shape == X.shape\n"
        "# Manual expand reference: unsqueeze ascending.\n"
        "expected = G.unsqueeze(0).unsqueeze(2).expand(X.shape)\n"
        "assert t.allclose(g, expected)\n"
        "\n"
        "# --- spot-check: every position in the SUMMED axes gets the same value ---\n"
        "# For dims=(0, 2), positions (b, j, c, k) and (b', j, c', k) must agree.\n"
        "assert t.allclose(g[0, 1, 0, 2], g[1, 1, 3, 2]), 'broadcast along dim 0 and 2 broken'\n"
        "\n"
        "# --- three-axis: dims=(0, 1, 3) on shape (2, 3, 4, 5) ---\n"
        "OUT = X.sum(dim=(0, 1, 3))             # (4,)\n"
        "G = t.randn(4, generator=rng)\n"
        "g = sum_back_multi(G, OUT, X, dims=(0, 1, 3), keepdim=False)\n"
        "assert g.shape == X.shape\n"
        "expected = G.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(X.shape)\n"
        "assert t.allclose(g, expected)\n"
        "\n"
        "# --- keepdim=True: skip the unsqueeze loop ---\n"
        "OUT_KD = X.sum(dim=(0, 2), keepdim=True)   # (1, 3, 1, 5)\n"
        "G_KD = t.randn(1, 3, 1, 5, generator=rng)\n"
        "g_kd = sum_back_multi(G_KD, OUT_KD, X, dims=(0, 2), keepdim=True)\n"
        "assert g_kd.shape == X.shape\n"
        "assert t.allclose(g_kd, G_KD.expand(X.shape))\n"
        "\n"
        "# --- dims passed UNSORTED still works (we sort internally) ---\n"
        "OUT = X.sum(dim=(2, 0))                # same as (0, 2)\n"
        "G = t.randn(3, 5, generator=rng)\n"
        "g_unsorted = sum_back_multi(G, OUT, X, dims=(2, 0), keepdim=False)\n"
        "g_sorted = sum_back_multi(G, OUT, X, dims=(0, 2), keepdim=False)\n"
        "assert t.allclose(g_unsorted, g_sorted), 'dims order should not affect result'\n"
        "\n"
        "# --- agreement with torch.autograd ---\n"
        "x_ref = t.randn(2, 3, 4, 5, requires_grad=True, generator=t.Generator().manual_seed(5))\n"
        "y = x_ref.sum(dim=(0, 2)).sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "out_cached = x_det.sum(dim=(0, 2))\n"
        "g_ours = sum_back_multi(t.ones(3, 5), out_cached, x_det, dims=(0, 2), keepdim=False)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-5), (\n"
        "    f'disagrees with autograd: max diff '\n"
        "    f'{(g_ours - x_ref.grad).abs().max()}'\n"
        ")"
    ),
    solution_body=(
        "def sum_back_multi(grad_out: Tensor, out: Tensor, x: Tensor, dims: tuple, keepdim: bool = False) -> Tensor:\n"
        "    if keepdim:\n"
        "        return grad_out.expand(x.shape)\n"
        "    g = grad_out\n"
        "    # Ascending order: each unsqueeze leaves later indices in dims intact.\n"
        "    for d in sorted(dims):\n"
        "        g = g.unsqueeze(d)\n"
        "    return g.expand(x.shape)"
    ),
    solution_notes=(
        "**Why ascending order for unsqueeze.** Suppose `dims=(0, 2)`. After "
        "`unsqueeze(0)`, the tensor gains a leading axis: indices 1, 2, ... "
        "in the result correspond to indices 0, 1, ... in the input. Axis "
        "`2` in the FINAL output is still axis `2` in the current state — "
        "because the new axis was inserted BEFORE it. Descending order "
        "would need index correction.\n\n"
        "**Why `sorted(dims)` even if the caller already sorted.** "
        "Callers shouldn't be obliged to sort — and dispatchers don't "
        "reorder Recipe kwargs. Internal `sorted()` keeps the back fn "
        "robust to whatever the forward wrapper stored.\n\n"
        "**Where multi-axis sum_back shows up.** Layer-norm reduces over "
        "the last `K` axes simultaneously. Cross-entropy over "
        "`(batch, classes)` reduces over classes. Any reduction with "
        "`dim=tuple` uses this exact back fn — and getting the unsqueeze "
        "order wrong is a classic bug."
    ),
)


# ---------------------------------------------------------------------------
# SPECS + verify + emit
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_LAMBDAS,
    SPEC_EXP,
    SPEC_GETITEM,
    SPEC_MATMUL,
    SPEC_NEGATIVE,
    SPEC_PERMUTE,
    SPEC_RESHAPE,
    SPEC_SUM,
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
        preamble_ok = True
        for preamble in spec.get("extra_imports", []):
            try:
                exec(preamble, ns)
            except Exception as e:
                failed.append((tag, f"preamble: {e!r}", traceback.format_exc()))
                preamble_ok = False
                break
        if not preamble_ok:
            continue

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

    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, err, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(err)
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[deepening_m_batch10] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_m_batch10] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_m_batch10] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
