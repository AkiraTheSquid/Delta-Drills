#!/usr/bin/env python3
"""Author 8 deepening (ex2) standalones for prereqs_autograd_internals atoms.

Each ex2 here probes a DISTINCT facet from the existing ex1 for that atom —
different cognitive operation, different surface context. ONE LO + ONE
Bloom + <=2 KCs per drill. PS4 framing: facts + ONE exemplar in recap,
then stub + assertions; no skill-explanation prose.

All 8 atoms live in prereqs_autograd_internals/ and share the MiniTensor +
Recipe + grad_tracking_enabled scaffold — injected via the
_AUTOGRAD_PREAMBLE string (same pattern as author_autograd_internals_batch3).

Verification re-runs each spec inside the build venv (torch 2.12.0+cpu)
before any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_internals"


# -----------------------------------------------------------------------------
# Shared autograd-internals preamble. Identical wording to batch-3's so the
# notebooks compose cleanly. Injected via extra_imports.
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Atom recaps — short, PS4-style, with ONE worked exemplar each.
# -----------------------------------------------------------------------------

RECAP_ARG_POSITION = (
    "## Arg-position back fns — quick refresher\n"
    "\n"
    "Binary ops register **two** back fns — one per input position — because the "
    "gradient w.r.t. each arg is a different function of `(grad_out, x, y)`.\n"
    "\n"
    "**Worked exemplar.** For `out = x ** y` (element-wise power):\n"
    "```\n"
    "d(x**y)/dx = y * x**(y-1)             # 'pow_back0', for arg-0 (x)\n"
    "d(x**y)/dy = x**y * log(x)            # 'pow_back1', for arg-1 (y)\n"
    "```\n"
    "Both back fns take `(grad_out, out, x, y)`. `pow_back0` reuses `x` and `y`; "
    "`pow_back1` reuses the cached `out` (= `x**y`) and `log(x)`."
)

RECAP_CHAIN_RULE_ELEMENTWISE = (
    "## Elementwise chain rule — quick refresher\n"
    "\n"
    "For elementwise `out = f(x)`, the local Jacobian is diagonal and the chain "
    "rule collapses to `dL/dx[i] = dL/dout[i] * f'(x[i])`.\n"
    "\n"
    "**Worked exemplar.** `out = tanh(x)`:\n"
    "```\n"
    "tanh'(x) = 1 - tanh(x)**2 = 1 - out**2     # use the cached out\n"
    "tanh_back(grad_out, out, x) = grad_out * (1 - out**2)\n"
    "```\n"
    "Reusing `out` skips the recomputation of `tanh(x)` during backward."
)

RECAP_GRAD_TRACKING_TOGGLE = (
    "## Grad-tracking global toggle — quick refresher\n"
    "\n"
    "A module-level `grad_tracking_enabled` boolean gates all autograd. The "
    "**decorator** form wraps a whole function in an automatic disable/restore.\n"
    "\n"
    "**Worked exemplar.**\n"
    "```python\n"
    "@no_grad\n"
    "def update_ema(target, source, decay=0.99):\n"
    "    # inside this body, grad_tracking_enabled is False; restored on return\n"
    "    ...\n"
    "```\n"
    "Even if the body raises, the previous value must be restored — `try/finally`."
)

RECAP_KWARGS_PASS_THROUGH = (
    "## Kwargs pass-through Recipe — quick refresher\n"
    "\n"
    "The wrapper threads kwargs into **two** places: the forward call AND the "
    "Recipe. When the caller passes NO kwargs, the Recipe's `kwargs` must be the "
    "empty dict `{}` — NOT the function's default values.\n"
    "\n"
    "**Worked exemplar.** `t.sum` has default `dim=None, keepdim=False`. "
    "`wrapped_sum(x)` (no kwargs) → forward sees defaults, but the Recipe stores "
    "`kwargs == {}` because nothing was passed at this call site."
)

RECAP_PARENTS_DICT = (
    "## Parents dict by argidx — quick refresher\n"
    "\n"
    "`Recipe.parents` maps the arg's original index (positional) or name (kwarg) "
    "to its parent Tensor, skipping non-Tensor inputs.\n"
    "\n"
    "**Worked exemplar.** `op(t1, 3.0, kw=t2)` → "
    "`parents == {0: t1, 'kw': t2}`. Positional float skipped; kwarg Tensor "
    "keyed by its kwarg name (string), not by a positional index."
)

RECAP_RECIPE_DATACLASS = (
    "## Recipe dataclass — quick refresher\n"
    "\n"
    "A Recipe records exactly enough to replay a forward in reverse: "
    "`(func, args, kwargs, parents)` with `args` as the **unboxed raw arrays** "
    "and `parents` as a `{argidx: MiniTensor}` map.\n"
    "\n"
    "**Worked exemplar.** A binary `add_forward(x, y)`:\n"
    "```python\n"
    "out = MiniTensor(x.array + y.array)\n"
    "out.recipe = Recipe(\n"
    "    func=t.add, args=(x.array, y.array), kwargs={},\n"
    "    parents={0: x, 1: y},\n"
    ")\n"
    "```\n"
    "Both inputs unboxed in `args`; both Tensors keyed by argidx in `parents`."
)

RECAP_REQUIRES_GRAD_PROP = (
    "## requires_grad propagation — quick refresher\n"
    "\n"
    "Three-gate AND: toggle AND is_differentiable AND any-input-requires-grad. "
    "The any-input scan must include **kwargs Tensor values** too — `dim=` or "
    "`mask=tensor` style kwargs can be Tensors that should propagate grad.\n"
    "\n"
    "**Worked exemplar.** `op(x_constant, mask=tracked_tensor)` — positional `x` "
    "has `requires_grad=False`, but kwarg `mask` has `requires_grad=True`. The "
    "output must have `requires_grad=True`."
)

RECAP_UNBROADCAST = (
    "## Unbroadcast — quick refresher\n"
    "\n"
    "`unbroadcast(grad, original)` peels leading axes then collapses expanded "
    "size-1 axes so `grad.shape == original.shape`.\n"
    "\n"
    "**Worked exemplar.** `original.shape == (1, 4)`, `grad.shape == (2, 3, 1, 4)`:\n"
    "1. Peel two leading axes: sum dim=0 twice → `(1, 4)`.\n"
    "2. No size-1 collapse needed (shapes already match).\n"
    "Result: shape `(1, 4)`, values = sum of 6 broadcast copies."
)


# -----------------------------------------------------------------------------
# spec helper
# -----------------------------------------------------------------------------

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
# 1. arg-position-back-functions  — ex2
#    ex1 facet: div_back0/div_back1 (rational op, both numeric)
#    ex2 facet: pow_back0/pow_back1 (power op — both args differentiable,
#    different formula structure: polynomial vs exponential-with-log)
# =========================================================================

SPEC_ARG_POSITION = _spec(
    atom_id="arg-position-back-functions",
    subtopic="Backprop: Arg-position back funcs",
    recap=RECAP_ARG_POSITION,
    ex_idx=2,
    ex_title="write pow_back0 and pow_back1 — polynomial vs log-exponential per-arg back fns",
    slug="write-pow-back0-and-pow-back1-power-op",
    bloom="Apply",
    difficulty_num=3,
    keywords=["pow-back", "binary-op", "log-derivative", "per-arg-back-fn"],
    kcs=["arg-position-back-functions", "back-fn-uses-cached-out"],
    lo=(
        "Apply the (grad_out, out, x, y) back-fn convention to write "
        "pow_back0 (polynomial form) and pow_back1 (log-exponential form) "
        "for out = x ** y, picking the right cached value in each."
    ),
    prompt_body=(
        "Implement TWO back fns for the elementwise power op `out = x ** y` "
        "(`x > 0` so logs are defined).\n\n"
        "**1. `pow_back0(grad_out, out, x, y) -> grad_x`** — gradient w.r.t. `x`.\n"
        "   Math: `d(x**y)/dx = y * x**(y-1)`.\n"
        "   Return `grad_out * y * x ** (y - 1)`, shape == `x.shape`.\n\n"
        "**2. `pow_back1(grad_out, out, x, y) -> grad_y`** — gradient w.r.t. `y`.\n"
        "   Math: `d(x**y)/dy = x**y * log(x) = out * log(x)`.\n"
        "   Return `grad_out * out * t.log(x)`, shape == `y.shape`. "
        "Reuse the cached `out`, do not recompute `x ** y`.\n\n"
        "The test checks scalar, vector, and matrix shapes, and cross-checks "
        "against `torch.autograd` on a `requires_grad=True` ground truth."
    ),
    stub=(
        "def pow_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dx for out = x ** y (x > 0)."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def pow_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dy for out = x ** y (x > 0)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- scalar sanity: 2**3 = 8 ---\n"
        "x = t.tensor([2.0]); y = t.tensor([3.0])\n"
        "out = x ** y\n"
        "g0 = pow_back0(t.tensor([1.0]), out, x, y)\n"
        "g1 = pow_back1(t.tensor([1.0]), out, x, y)\n"
        "# d/dx (x**3) at x=2 = 3*4 = 12;  d/dy (2**y) at y=3 = 8 * ln(2)\n"
        "assert t.allclose(g0, t.tensor([12.0])), f'pow_back0 scalar: {g0}'\n"
        "assert t.allclose(g1, t.tensor([8.0 * float(t.log(t.tensor(2.0)))])), (\n"
        "    f'pow_back1 scalar: {g1}'\n"
        ")\n"
        "\n"
        "# --- vector ---\n"
        "x = t.tensor([1.5, 2.0, 4.0])\n"
        "y = t.tensor([2.0, 3.0, 0.5])\n"
        "out = x ** y\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "g0 = pow_back0(grad_out, out, x, y)\n"
        "g1 = pow_back1(grad_out, out, x, y)\n"
        "assert g0.shape == x.shape, f'pow_back0 shape: {g0.shape}'\n"
        "assert g1.shape == y.shape, f'pow_back1 shape: {g1.shape}'\n"
        "assert t.allclose(g0, y * x ** (y - 1)), f'pow_back0 vector: {g0}'\n"
        "assert t.allclose(g1, out * t.log(x)), f'pow_back1 vector: {g1}'\n"
        "\n"
        "# --- non-unit grad_out, matrix ---\n"
        "rng = t.Generator().manual_seed(7)\n"
        "X = t.rand(3, 4, generator=rng) * 2 + 0.5    # positive\n"
        "Y = t.rand(3, 4, generator=rng) * 2 + 0.5\n"
        "OUT = X ** Y\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "g0 = pow_back0(G, OUT, X, Y)\n"
        "g1 = pow_back1(G, OUT, X, Y)\n"
        "assert g0.shape == X.shape and g1.shape == Y.shape\n"
        "assert t.allclose(g0, G * Y * X ** (Y - 1)), 'pow_back0 matrix mismatch'\n"
        "assert t.allclose(g1, G * OUT * t.log(X)), 'pow_back1 matrix mismatch'\n"
        "\n"
        "# --- cross-check vs autograd ---\n"
        "xa = t.tensor([1.5, 2.0, 4.0], requires_grad=True)\n"
        "ya = t.tensor([2.0, 3.0, 0.5], requires_grad=True)\n"
        "loss = (xa ** ya).sum()\n"
        "loss.backward()\n"
        "g0_ref = xa.grad\n"
        "g1_ref = ya.grad\n"
        "out_ref = (xa ** ya).detach()\n"
        "g0_ours = pow_back0(t.ones_like(out_ref), out_ref, xa.detach(), ya.detach())\n"
        "g1_ours = pow_back1(t.ones_like(out_ref), out_ref, xa.detach(), ya.detach())\n"
        "assert t.allclose(g0_ours, g0_ref, atol=1e-5), (\n"
        "    f'pow_back0 vs autograd: ours={g0_ours} ref={g0_ref}'\n"
        ")\n"
        "assert t.allclose(g1_ours, g1_ref, atol=1e-5), (\n"
        "    f'pow_back1 vs autograd: ours={g1_ours} ref={g1_ref}'\n"
        ")"
    ),
    solution_body=(
        "def pow_back0(grad_out, out, x, y):\n"
        "    return grad_out * y * x ** (y - 1)\n"
        "\n"
        "\n"
        "def pow_back1(grad_out, out, x, y):\n"
        "    return grad_out * out * t.log(x)"
    ),
    solution_notes=(
        "**Why pow_back1 reuses `out`.** `out == x ** y` is already cached; "
        "recomputing `x ** y` inside the back fn is wasteful and risks "
        "numerical drift. The `(grad_out, out, x, y)` signature exists "
        "precisely so back fns can lean on the forward's cached result.\n\n"
        "**Why pow_back0 does NOT need `out`.** The derivative `y * x**(y-1)` "
        "is a function of the inputs alone — `out` is unused here. Both "
        "back fns receive `out` for uniform dispatch even when one doesn't "
        "need it.\n\n"
        "**Domain restriction.** `log(x)` is undefined for `x <= 0`, so this "
        "back fn assumes `x > 0` at call time. Real-world autograd "
        "implementations either restrict the domain or use the complex-log "
        "extension; for ARENA we keep `x` positive in the tests."
    ),
)


# =========================================================================
# 2. chain-rule-elementwise  — ex2
#    ex1 facet: sigmoid_back + relu_back (different cache strategies)
#    ex2 facet: tanh_back + softplus_back — tanh uses cached `out`,
#    softplus' derivative IS sigmoid (different fact-pattern coupling)
# =========================================================================

SPEC_CHAIN_RULE = _spec(
    atom_id="chain-rule-elementwise",
    subtopic="Backprop: Elementwise chain rule",
    recap=RECAP_CHAIN_RULE_ELEMENTWISE,
    ex_idx=2,
    ex_title="write tanh_back and softplus_back from the elementwise chain rule",
    slug="write-tanh-and-softplus-back-from-elementwise-chain-rule",
    bloom="Apply",
    difficulty_num=3,
    keywords=["tanh", "softplus", "chain-rule", "elementwise-derivative"],
    kcs=["chain-rule-elementwise", "back-fn-uses-cached-out"],
    lo=(
        "Apply the elementwise chain rule to derive tanh_back (uses cached "
        "out) and softplus_back (derivative is sigmoid(x)), returning "
        "grad_in = grad_out * f'(x) per-position."
    ),
    prompt_body=(
        "Implement TWO elementwise back fns. Both have signature "
        "`(grad_out, out, x) -> grad_in` with shape `grad_in.shape == x.shape`.\n\n"
        "**1. `tanh_back(grad_out, out, x) -> grad_in`** — for `out = tanh(x)`.\n"
        "   Math: `tanh'(x) = 1 - tanh(x)**2 = 1 - out**2`.\n"
        "   Return `grad_out * (1 - out ** 2)`. Reuse the cached `out`.\n\n"
        "**2. `softplus_back(grad_out, out, x) -> grad_in`** — for "
        "`out = softplus(x) = log(1 + exp(x))`.\n"
        "   Math: `softplus'(x) = sigmoid(x) = 1 / (1 + exp(-x))`.\n"
        "   Return `grad_out * t.sigmoid(x)`. Note: depends on `x`, NOT `out` — "
        "softplus' derivative happens to be a different elementary function.\n\n"
        "The test checks shape preservation, non-unit grad_out scaling, the "
        "limit behavior (`tanh_back` vanishes as `|x| -> inf`; `softplus_back` "
        "saturates to 0 for large negative `x` and to 1 for large positive `x`), "
        "and cross-checks against `torch.autograd`."
    ),
    stub=(
        "def tanh_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = tanh(x). Use the cached `out`."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def softplus_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = softplus(x). Derivative is sigmoid(x)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- tanh_back ---\n"
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])\n"
        "out = t.tanh(x)\n"
        "g = tanh_back(t.ones(5), out, x)\n"
        "expected = 1 - out ** 2\n"
        "assert g.shape == x.shape, f'tanh_back shape: {g.shape}'\n"
        "assert t.allclose(g, expected), f'tanh_back value: {g} vs {expected}'\n"
        "\n"
        "# tanh_back chain-rule scaling with non-unit grad_out.\n"
        "grad_out = t.tensor([3.0, -1.0, 2.0, 0.5, -4.0])\n"
        "g = tanh_back(grad_out, out, x)\n"
        "assert t.allclose(g, grad_out * (1 - out ** 2)), 'tanh_back chain rule failed'\n"
        "\n"
        "# Saturation: tanh_back ~ 0 at |x|=10.\n"
        "x_sat = t.tensor([-10.0, 10.0])\n"
        "out_sat = t.tanh(x_sat)\n"
        "g_sat = tanh_back(t.ones(2), out_sat, x_sat)\n"
        "assert t.all(g_sat.abs() < 1e-4), f'tanh_back must saturate to ~0 at |x|=10: {g_sat}'\n"
        "\n"
        "# --- softplus_back ---\n"
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])\n"
        "out = t.nn.functional.softplus(x)\n"
        "g = softplus_back(t.ones(5), out, x)\n"
        "expected = t.sigmoid(x)\n"
        "assert g.shape == x.shape, f'softplus_back shape: {g.shape}'\n"
        "assert t.allclose(g, expected), f'softplus_back value: {g} vs {expected}'\n"
        "\n"
        "# softplus_back at x=0 = sigmoid(0) = 0.5 EXACTLY.\n"
        "x0 = t.tensor([0.0, 0.0])\n"
        "g0 = softplus_back(t.ones(2), t.nn.functional.softplus(x0), x0)\n"
        "assert t.allclose(g0, t.full((2,), 0.5)), f'softplus_back(0) must be 0.5: {g0}'\n"
        "\n"
        "# Saturation: softplus_back -> 0 at very negative x, -> 1 at very positive x.\n"
        "x_lim = t.tensor([-10.0, 10.0])\n"
        "out_lim = t.nn.functional.softplus(x_lim)\n"
        "g_lim = softplus_back(t.ones(2), out_lim, x_lim)\n"
        "assert g_lim[0] < 1e-4, f'softplus_back(-10) must be ~0: {g_lim[0]}'\n"
        "assert g_lim[1] > 1 - 1e-4, f'softplus_back(10) must be ~1: {g_lim[1]}'\n"
        "\n"
        "# --- cross-check vs autograd ---\n"
        "xa = t.tensor([-1.0, 0.5, 2.0], requires_grad=True)\n"
        "(t.tanh(xa).sum()).backward()\n"
        "g_ref_tanh = xa.grad.clone()\n"
        "xa.grad = None\n"
        "(t.nn.functional.softplus(xa).sum()).backward()\n"
        "g_ref_sp = xa.grad.clone()\n"
        "\n"
        "x_det = xa.detach()\n"
        "g_tanh = tanh_back(t.ones(3), t.tanh(x_det), x_det)\n"
        "g_sp   = softplus_back(t.ones(3), t.nn.functional.softplus(x_det), x_det)\n"
        "assert t.allclose(g_tanh, g_ref_tanh, atol=1e-5), f'tanh vs autograd: {g_tanh} vs {g_ref_tanh}'\n"
        "assert t.allclose(g_sp,   g_ref_sp,   atol=1e-5), f'softplus vs autograd: {g_sp} vs {g_ref_sp}'"
    ),
    solution_body=(
        "def tanh_back(grad_out, out, x):\n"
        "    return grad_out * (1 - out ** 2)\n"
        "\n"
        "\n"
        "def softplus_back(grad_out, out, x):\n"
        "    return grad_out * t.sigmoid(x)"
    ),
    solution_notes=(
        "**Two cache strategies, one signature.** `tanh_back` uses the cached "
        "`out` because `tanh'(x) = 1 - tanh(x)**2` is naturally written in "
        "terms of the output. `softplus_back` uses `x` because softplus' "
        "derivative IS sigmoid — a different elementary function that "
        "isn't expressible cleanly via `out = log(1 + exp(x))`.\n\n"
        "**Why `softplus_back` ignores its `out` argument.** The uniform "
        "back-fn signature `(grad_out, out, x)` always passes `out` so the "
        "dispatcher doesn't have to know which back fn needs it. Unused "
        "arguments are normal — the back fn just doesn't reference them.\n\n"
        "**Saturation matters.** `tanh_back` vanishes at the tails, which "
        "is the source of the classic vanishing-gradient problem with "
        "deep tanh networks. `softplus_back` saturates to 1 on the positive "
        "side (like ReLU) and to 0 on the negative side, but smoothly — "
        "this is the appeal of softplus as a smooth ReLU substitute."
    ),
)


# =========================================================================
# 3. grad-tracking-global-toggle  — ex2
#    ex1 facet: NoGrad context manager (`with` form)
#    ex2 facet: decorator form (`@no_grad`) — wraps fn body, must restore
#    even on exception
# =========================================================================

SPEC_GRAD_TOGGLE = _spec(
    atom_id="grad-tracking-global-toggle",
    subtopic="Backprop: Grad-tracking toggle",
    recap=RECAP_GRAD_TRACKING_TOGGLE,
    ex_idx=2,
    ex_title="no_grad decorator built on the module-level toggle (try/finally restore)",
    slug="no-grad-decorator-built-on-module-toggle",
    bloom="Apply",
    difficulty_num=3,
    keywords=["no-grad", "decorator", "try-finally", "exception-safety"],
    kcs=["grad-tracking-global-toggle", "no-grad-decorator-restores-on-exception"],
    lo=(
        "Apply a decorator pattern over the module-level grad_tracking_enabled "
        "toggle: disable on entry, restore previous value on exit even when "
        "the decorated function raises."
    ),
    prompt_body=(
        "Implement `no_grad` as a DECORATOR (not a context manager).\n\n"
        "1. `no_grad(fn)` returns a wrapper that:\n"
        "   - Snapshots the current value of the global "
        "`grad_tracking_enabled`.\n"
        "   - Sets `grad_tracking_enabled = False` before calling `fn`.\n"
        "   - Uses `try / finally` so the previous value is restored even if "
        "`fn` raises.\n"
        "   - Returns whatever `fn` returns.\n\n"
        "**Critical: assign to the GLOBAL.** Inside the wrapper, you MUST do "
        "`global grad_tracking_enabled` before assigning. Without it, the "
        "wrapper creates a local variable and the global never flips — silent "
        "no-op bug.\n\n"
        "The test verifies: (a) inside the decorated fn the toggle is False; "
        "(b) after return the toggle is restored to its prior value; "
        "(c) if the decorated fn raises, the toggle is STILL restored; "
        "(d) nesting works (decorate inside `with NoGrad(): ...` keeps "
        "the toggle False both during and after); "
        "(e) the wrapper preserves the return value."
    ),
    stub=(
        "def no_grad(fn):\n"
        '    """Decorator: disable grad_tracking_enabled around the call, restore on exit."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Confirm initial state.\n"
        "assert globals()['grad_tracking_enabled'] is True\n"
        "\n"
        "# --- happy path: toggle off inside, restored after ---\n"
        "seen_inside = []\n"
        "\n"
        "@no_grad\n"
        "def f(x):\n"
        "    seen_inside.append(globals()['grad_tracking_enabled'])\n"
        "    return x * 2\n"
        "\n"
        "result = f(5)\n"
        "assert result == 10, f'wrapper must return fn result: {result}'\n"
        "assert seen_inside == [False], (\n"
        "    f'inside decorated fn, toggle must be False. saw {seen_inside}'\n"
        ")\n"
        "assert globals()['grad_tracking_enabled'] is True, (\n"
        "    'toggle must be restored to True after decorated fn returns'\n"
        ")\n"
        "\n"
        "# --- exception path: toggle restored even if fn raises ---\n"
        "@no_grad\n"
        "def f_raises():\n"
        "    assert globals()['grad_tracking_enabled'] is False\n"
        "    raise RuntimeError('boom')\n"
        "\n"
        "try:\n"
        "    f_raises()\n"
        "except RuntimeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('f_raises should have raised RuntimeError')\n"
        "\n"
        "assert globals()['grad_tracking_enabled'] is True, (\n"
        "    'toggle must be restored even when decorated fn raises — try/finally'\n"
        ")\n"
        "\n"
        "# --- nesting: decorated fn called from inside another decorated fn ---\n"
        "log = []\n"
        "\n"
        "@no_grad\n"
        "def outer():\n"
        "    log.append(('outer enter', globals()['grad_tracking_enabled']))\n"
        "    inner()\n"
        "    log.append(('outer after_inner', globals()['grad_tracking_enabled']))\n"
        "\n"
        "@no_grad\n"
        "def inner():\n"
        "    log.append(('inner', globals()['grad_tracking_enabled']))\n"
        "\n"
        "outer()\n"
        "assert log == [\n"
        "    ('outer enter', False),\n"
        "    ('inner', False),\n"
        "    ('outer after_inner', False),\n"
        "], f'nested no_grad must keep toggle False throughout: {log}'\n"
        "assert globals()['grad_tracking_enabled'] is True, 'fully restored after outer returns'\n"
        "\n"
        "# --- starting from False: must restore to False, not flip to True ---\n"
        "globals()['grad_tracking_enabled'] = False\n"
        "\n"
        "@no_grad\n"
        "def g():\n"
        "    return globals()['grad_tracking_enabled']\n"
        "\n"
        "assert g() is False\n"
        "assert globals()['grad_tracking_enabled'] is False, (\n"
        "    'must restore the PREVIOUS value (False), not unconditionally set True'\n"
        ")\n"
        "globals()['grad_tracking_enabled'] = True  # reset for any later cells\n"
        "\n"
        "# --- argument + kwargs pass-through ---\n"
        "@no_grad\n"
        "def h(a, b, c=10):\n"
        "    return a + b + c\n"
        "\n"
        "assert h(1, 2) == 13\n"
        "assert h(1, 2, c=5) == 8"
    ),
    solution_body=(
        "def no_grad(fn):\n"
        "    def wrapper(*args, **kwargs):\n"
        "        global grad_tracking_enabled\n"
        "        prev = grad_tracking_enabled\n"
        "        grad_tracking_enabled = False\n"
        "        try:\n"
        "            return fn(*args, **kwargs)\n"
        "        finally:\n"
        "            grad_tracking_enabled = prev\n"
        "    return wrapper"
    ),
    solution_notes=(
        "**Why `try / finally` and not `try / except`.** We don't want to "
        "swallow the exception — propagate it. We only want to guarantee the "
        "restore. `finally` runs whether the body returned, raised, or even "
        "called `sys.exit`.\n\n"
        "**Why snapshot `prev` instead of `True`.** If `no_grad` is called "
        "while the toggle is ALREADY False (nested call, or after an outer "
        "`with NoGrad():`), unconditionally restoring to True would re-enable "
        "grad tracking inside an outer no-grad scope — bug. Saving and "
        "restoring the previous value composes correctly.\n\n"
        "**The `global` keyword.** Without it, "
        "`grad_tracking_enabled = False` creates a function-local variable "
        "and the module-level toggle never changes. This is the most common "
        "bug when implementing toggle decorators in Python — the test "
        "specifically checks the global flips."
    ),
)


# =========================================================================
# 4. kwargs-pass-through-recipe  — ex2
#    ex1 facet: wrap t.sum threading dim/keepdim into forward AND Recipe
#    ex2 facet: NO kwargs at call site → Recipe.kwargs == {} (NOT default
#    values); plus a separate call WITH kwargs to show the wrapper handles
#    both consistently
# =========================================================================

SPEC_KWARGS = _spec(
    atom_id="kwargs-pass-through-recipe",
    subtopic="Backprop: Kwargs pass-through",
    recap=RECAP_KWARGS_PASS_THROUGH,
    ex_idx=2,
    ex_title="empty-kwargs case — wrap_forward_fn must record `{}` when caller passes none",
    slug="empty-kwargs-recipe-records-empty-dict",
    bloom="Apply",
    difficulty_num=3,
    keywords=["kwargs", "recipe", "empty-kwargs", "no-default-injection"],
    kcs=["kwargs-pass-through-recipe", "recipe-kwargs-faithful-to-call-site"],
    lo=(
        "Apply wrap_forward_fn so that when the caller passes NO kwargs the "
        "Recipe stores `kwargs == {}` exactly — not the forward fn's default "
        "values — while still working when kwargs ARE passed."
    ),
    prompt_body=(
        "Implement `wrap_forward_fn(fwd_fn)` so it threads kwargs faithfully:\n\n"
        "1. Unbox positional args: `raw = tuple(a.array if isinstance(a, "
        "MiniTensor) else a for a in args)`.\n"
        "2. Forward call: `out_raw = fwd_fn(*raw, **kwargs)` (kwargs may be "
        "empty — that's fine, Python accepts `**{}`).\n"
        "3. Wrap: `out = MiniTensor(out_raw)`.\n"
        "4. Attach a Recipe:\n"
        "   - `Recipe.func = fwd_fn`\n"
        "   - `Recipe.args = raw`\n"
        "   - `Recipe.kwargs = kwargs` (the literal dict passed to the "
        "wrapper, NOT the function's defaults)\n"
        "   - `Recipe.parents = {0: x}` for a unary call. (For this drill we "
        "only test unary ops — kwargs handling is the focus.)\n\n"
        "**Key invariant.** If the caller writes `wrapped_sum(x)`, the Recipe's "
        "`kwargs` must be `{}` — even though `t.sum(x)` internally uses "
        "`dim=None, keepdim=False`. Reverse-pass code keys back fns on the "
        "ACTUAL kwargs passed at the forward call site; injecting defaults "
        "would break that contract.\n\n"
        "The test calls the wrapper TWICE: once with no kwargs, once with "
        "`dim=1`, and asserts Recipe.kwargs matches the call site in each case."
    ),
    stub=(
        "def wrap_forward_fn(fwd_fn):\n"
        '    """Return a MiniTensor-aware wrapper that records kwargs faithfully."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Wrap t.sum.\n"
        "wrapped_sum = wrap_forward_fn(t.sum)\n"
        "\n"
        "# --- Case A: no kwargs ---\n"
        "x = MiniTensor(t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n"
        "out = wrapped_sum(x)\n"
        "assert isinstance(out, MiniTensor), 'output must be MiniTensor'\n"
        "# t.sum(x) with no kwargs sums the entire tensor.\n"
        "assert t.allclose(out.array, t.tensor(21.0)), (\n"
        "    f'no-kwargs forward must use default behavior (full reduction): {out.array}'\n"
        ")\n"
        "assert out.recipe is not None, 'Recipe was never attached'\n"
        "assert out.recipe.func is t.sum, f'Recipe.func wrong: {out.recipe.func}'\n"
        "assert out.recipe.kwargs == {}, (\n"
        "    f'no-kwargs case must store empty dict, got {out.recipe.kwargs!r}. '\n"
        "    'The Recipe must reflect what the CALLER passed, not the fn defaults.'\n"
        ")\n"
        "assert 'dim' not in out.recipe.kwargs, (\n"
        "    'Recipe must NOT inject the fn default dim=None into kwargs'\n"
        ")\n"
        "assert 'keepdim' not in out.recipe.kwargs, (\n"
        "    'Recipe must NOT inject the fn default keepdim=False into kwargs'\n"
        ")\n"
        "\n"
        "# --- Case B: WITH kwargs ---\n"
        "out2 = wrapped_sum(x, dim=1)\n"
        "assert t.allclose(out2.array, t.tensor([6.0, 15.0])), (\n"
        "    f'dim=1 forward wrong: {out2.array}'\n"
        ")\n"
        "assert out2.recipe.kwargs == {'dim': 1}, (\n"
        "    f'with-kwargs case must store {{\"dim\": 1}}, got {out2.recipe.kwargs!r}'\n"
        ")\n"
        "\n"
        "# --- Case C: WITH multiple kwargs ---\n"
        "out3 = wrapped_sum(x, dim=1, keepdim=True)\n"
        "assert out3.array.shape == (2, 1), f'keepdim shape wrong: {out3.array.shape}'\n"
        "assert out3.recipe.kwargs == {'dim': 1, 'keepdim': True}, (\n"
        "    f'multi-kwarg case wrong: {out3.recipe.kwargs!r}'\n"
        ")\n"
        "\n"
        "# --- Case D: Recipe.args is the UNBOXED raw torch.Tensor, not the MiniTensor ---\n"
        "assert len(out.recipe.args) == 1\n"
        "assert isinstance(out.recipe.args[0], t.Tensor), (\n"
        "    f'Recipe.args[0] must be raw torch.Tensor (unboxed), got {type(out.recipe.args[0]).__name__}'\n"
        ")\n"
        "assert t.allclose(out.recipe.args[0], x.array)\n"
        "\n"
        "# --- Case E: independence — modifying out.recipe.kwargs must NOT affect a later call ---\n"
        "out_a = wrapped_sum(x)\n"
        "out_b = wrapped_sum(x, dim=0)\n"
        "assert out_a.recipe.kwargs == {}, 'Case A snapshot must remain {}'\n"
        "assert out_b.recipe.kwargs == {'dim': 0}, 'Case B snapshot must be its own dict'\n"
        "out_b.recipe.kwargs['extra'] = 'mutation'\n"
        "out_c = wrapped_sum(x)\n"
        "assert out_c.recipe.kwargs == {}, (\n"
        "    f'mutating a prior Recipe.kwargs leaked into a new call: {out_c.recipe.kwargs!r}'\n"
        ")"
    ),
    solution_body=(
        "def wrap_forward_fn(fwd_fn):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw = tuple(a.array if isinstance(a, MiniTensor) else a for a in args)\n"
        "        out_raw = fwd_fn(*raw, **kwargs)\n"
        "        out = MiniTensor(out_raw)\n"
        "        parents = {i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
        "        out.recipe = Recipe(func=fwd_fn, args=raw, kwargs=dict(kwargs), parents=parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    solution_notes=(
        "**Why `dict(kwargs)` instead of the bare `kwargs` reference.** "
        "Python passes `**kwargs` as a fresh dict each call, so reusing the "
        "reference is usually safe — BUT later test cases mutate Recipe.kwargs "
        "to verify no leakage. Copying with `dict(kwargs)` gives each Recipe "
        "its own dict, preventing surprises if downstream code edits it.\n\n"
        "**Why the empty-kwargs case matters.** Some back fns dispatch on "
        "whether a particular kwarg was passed at all (e.g. `sum_back` "
        "differs whether `dim` was specified vs the default of summing all "
        "axes). Injecting `dim=None` into the Recipe would conflate "
        "\"caller passed dim=None explicitly\" with \"caller passed nothing\". "
        "Faithful empty-dict preservation keeps that distinction.\n\n"
        "**Forward call still uses defaults.** `fwd_fn(*raw, **{})` is "
        "identical to `fwd_fn(*raw)` — Python's `**` unpacking does nothing "
        "for an empty dict, and the fn's default values activate normally. "
        "We get default behavior on the forward AND a faithful empty-dict in "
        "the Recipe — no conflict."
    ),
)


# =========================================================================
# 5. parents-dict-by-argidx  — ex2
#    ex1 facet: positional-only — {argidx: T} skipping non-Tensors
#    ex2 facet: kwargs case — kwarg-keyed entries by NAME (string), not
#    positional index; build_parents_full(args, kwargs) returns mixed dict
# =========================================================================

SPEC_PARENTS = _spec(
    atom_id="parents-dict-by-argidx",
    subtopic="Backprop: Parents dict by argidx",
    recap=RECAP_PARENTS_DICT,
    ex_idx=2,
    ex_title="build_parents_full — extend parents dict to include kwarg Tensors keyed by name",
    slug="build-parents-full-include-kwarg-tensors-by-name",
    bloom="Apply",
    difficulty_num=3,
    keywords=["parents", "kwargs", "mixed-keys", "string-keyed"],
    kcs=["parents-dict-by-argidx", "parents-kwarg-keyed-by-name"],
    lo=(
        "Apply the parents-dict construction over BOTH positional args and "
        "kwargs: positional Tensors keyed by argidx (int), kwarg Tensors "
        "keyed by name (str). Skip non-Tensors in both."
    ),
    prompt_body=(
        "Implement `build_parents_full(args, kwargs) -> dict`.\n\n"
        "Return a single dict mixing two key types:\n"
        "1. **Positional Tensors** → keyed by their original index (int). "
        "Skip non-Tensor positions.\n"
        "2. **Kwarg Tensors** → keyed by their kwarg name (str). Skip "
        "non-Tensor kwarg values.\n\n"
        "**Worked examples.**\n"
        "```python\n"
        "build_parents_full((t1, 3.0), {})            # {0: t1}\n"
        "build_parents_full((), {'mask': t1})         # {'mask': t1}\n"
        "build_parents_full((t1,), {'mask': t2})      # {0: t1, 'mask': t2}\n"
        "build_parents_full((3.0, t1), {'a': 5, 'b': t2})   # {1: t1, 'b': t2}\n"
        "```\n\n"
        "The dict mixes int and str keys — Python permits this. The reverse "
        "pass looks up positional back fns by int and kwarg back fns by "
        "string, so the key type DOES carry information.\n\n"
        "Use `MiniTensor` (defined in the preamble) as the Tensor type to "
        "test against."
    ),
    stub=(
        "def build_parents_full(args: tuple, kwargs: dict) -> dict:\n"
        '    """{argidx: MT} for positional + {kwname: MT} for kwargs, skipping non-Tensors."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "t1 = MiniTensor(t.tensor([1.0]))\n"
        "t2 = MiniTensor(t.tensor([2.0]))\n"
        "t3 = MiniTensor(t.tensor([3.0]))\n"
        "\n"
        "# --- empty / no tensors ---\n"
        "assert build_parents_full((), {}) == {}\n"
        "assert build_parents_full((1, 2.0), {'a': 3, 'b': 'str'}) == {}\n"
        "\n"
        "# --- positional only ---\n"
        "assert build_parents_full((t1,), {}) == {0: t1}\n"
        "p = build_parents_full((t1, t2), {})\n"
        "assert p == {0: t1, 1: t2}, f'positional pair: {p}'\n"
        "\n"
        "# --- kwarg only ---\n"
        "p = build_parents_full((), {'mask': t1})\n"
        "assert p == {'mask': t1}, f'kwarg-only: {p}'\n"
        "# key must be a string, not an int.\n"
        "assert 'mask' in p and 0 not in p, (\n"
        "    f'kwarg Tensor must be keyed by name (str), not collapsed to int: {p}'\n"
        ")\n"
        "\n"
        "# --- mixed ---\n"
        "p = build_parents_full((t1,), {'mask': t2})\n"
        "assert p == {0: t1, 'mask': t2}, f'mixed pos+kwarg: {p}'\n"
        "\n"
        "# --- skip non-Tensor in BOTH ---\n"
        "p = build_parents_full((3.0, t1), {'a': 5, 'b': t2})\n"
        "assert p == {1: t1, 'b': t2}, (\n"
        "    f'non-Tensors in both pos and kwarg must be skipped: {p}'\n"
        ")\n"
        "\n"
        "# --- argnum preserved for arg-1 Tensor even when arg-0 is non-Tensor ---\n"
        "p = build_parents_full((5.0, t1), {})\n"
        "assert p == {1: t1}, (\n"
        "    f'arg-1 must keep argnum=1, not collapse to 0: {p} '\n"
        "    '(renumbering would break BACK_FUNCS dispatch)'\n"
        ")\n"
        "\n"
        "# --- multi-tensor kwargs ---\n"
        "p = build_parents_full((), {'mask': t1, 'weights': t2, 'count': 5})\n"
        "assert p == {'mask': t1, 'weights': t2}, f'multi-kwarg: {p}'\n"
        "\n"
        "# --- kwarg name does NOT shadow positional argnum ---\n"
        "# Even if a user passes kwarg key '0' as a string, str('0') != int(0).\n"
        "p = build_parents_full((t1,), {'0': t2})\n"
        "assert 0 in p and '0' in p, (\n"
        "    f'int 0 and str \"0\" must coexist as distinct keys: {p}'\n"
        ")\n"
        "assert p[0] is t1 and p['0'] is t2\n"
        "\n"
        "# --- key types are EXACTLY int and str (no float, no bytes) ---\n"
        "p = build_parents_full((t1, t2), {'a': t3})\n"
        "for k in p:\n"
        "    assert isinstance(k, (int, str)) and not isinstance(k, bool), (\n"
        "        f'key {k!r} has unexpected type {type(k).__name__}'\n"
        "    )"
    ),
    solution_body=(
        "def build_parents_full(args, kwargs):\n"
        "    parents = {i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
        "    parents.update(\n"
        "        {k: v for k, v in kwargs.items() if isinstance(v, MiniTensor)}\n"
        "    )\n"
        "    return parents"
    ),
    solution_notes=(
        "**Why mix key types in one dict.** Python dicts are heterogeneous-"
        "key by design. The reverse pass distinguishes positional from kwarg "
        "back fns by the key TYPE: `isinstance(k, int)` → positional back fn "
        "at argnum `k`; `isinstance(k, str)` → kwarg back fn for kwarg name "
        "`k`. One unified container, two lookup paths.\n\n"
        "**Why kwargs Tensors need a back fn at all.** Some torch ops take "
        "tensor-valued kwargs that are differentiable (e.g. an `index_select` "
        "with a `weights=tensor` kwarg, or attention with a `mask` that "
        "participates in the gradient). Treating them as parents lets the "
        "reverse pass route gradients back to them.\n\n"
        "**The `'0'` vs `0` test catches a subtle collapse bug.** A naive "
        "implementation might `str(i)` the positional keys to unify the dict; "
        "that would clash if a user ever passed a kwarg literally named `'0'`. "
        "Keeping int keys as int avoids the collision."
    ),
)


# =========================================================================
# 6. recipe-dataclass  — ex2
#    ex1 facet: construct Recipe for unary log_forward
#    ex2 facet: construct Recipe for BINARY add_forward — proves
#    parents = {0: x, 1: y} AND args = (x.array, y.array) for binary case
# =========================================================================

SPEC_RECIPE = _spec(
    atom_id="recipe-dataclass",
    subtopic="Backprop: Recipe dataclass",
    recap=RECAP_RECIPE_DATACLASS,
    ex_idx=2,
    ex_title="construct Recipe for binary add_forward — both parents, both unboxed args",
    slug="construct-recipe-for-binary-add-forward",
    bloom="Apply",
    difficulty_num=3,
    keywords=["recipe", "binary-op", "add-forward", "two-parents"],
    kcs=["recipe-dataclass", "recipe-args-unboxed-correctly"],
    lo=(
        "Apply the 4-field Recipe construction to a binary forward "
        "(add_forward): args = (x.array, y.array), parents = {0: x, 1: y}, "
        "with kwargs = {}."
    ),
    prompt_body=(
        "Implement `add_forward(x, y)` for `MiniTensor` inputs:\n\n"
        "1. Compute `out_raw = x.array + y.array`.\n"
        "2. Return a NEW `MiniTensor` with `.array = out_raw` and a fully "
        "populated `.recipe`:\n"
        "   - `recipe.func = t.add`\n"
        "   - `recipe.args = (x.array, y.array)` — both inputs UNBOXED. "
        "Order matters: positional argnum 0 first.\n"
        "   - `recipe.kwargs = {}`\n"
        "   - `recipe.parents = {0: x, 1: y}` — both Tensors keyed by their "
        "argnum, referring to the ORIGINAL `MiniTensor` objects (not "
        "their `.array`).\n\n"
        "**Key contrast with the unary case.** A unary `log_forward(x)` has "
        "`parents = {0: x}` — one entry. The binary case has two, both "
        "indexed. Skipping either parent would orphan a branch of the graph "
        "and break the reverse pass.\n\n"
        "The test verifies field shape, numerical correctness, AND identity: "
        "`recipe.parents[0] is x` (the exact object, not a copy) — because "
        "the reverse pass mutates `.grad` on the parent."
    ),
    stub=(
        "def add_forward(x: 'MiniTensor', y: 'MiniTensor') -> 'MiniTensor':\n"
        '    """Return MiniTensor(x.array + y.array) with a fully-populated Recipe."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "x = MiniTensor(t.tensor([1.0, 2.0, 3.0]))\n"
        "y = MiniTensor(t.tensor([10.0, 20.0, 30.0]))\n"
        "\n"
        "out = add_forward(x, y)\n"
        "\n"
        "# --- Type + numerical correctness ---\n"
        "assert isinstance(out, MiniTensor), f'must return MiniTensor, got {type(out).__name__}'\n"
        "assert t.allclose(out.array, t.tensor([11.0, 22.0, 33.0])), f'sum wrong: {out.array}'\n"
        "\n"
        "# --- Recipe attached ---\n"
        "assert out.recipe is not None, 'Recipe was never attached'\n"
        "\n"
        "# --- Recipe.func ---\n"
        "assert out.recipe.func is t.add, f'Recipe.func must be t.add, got {out.recipe.func}'\n"
        "\n"
        "# --- Recipe.args: unboxed RAW tensors, in order (x.array, y.array) ---\n"
        "assert isinstance(out.recipe.args, tuple), f'args must be tuple: {type(out.recipe.args).__name__}'\n"
        "assert len(out.recipe.args) == 2, f'args must have 2 entries: len={len(out.recipe.args)}'\n"
        "assert isinstance(out.recipe.args[0], t.Tensor), (\n"
        "    f'args[0] must be raw torch.Tensor (unboxed), got {type(out.recipe.args[0]).__name__}. '\n"
        "    'Did you forget to .array the input?'\n"
        ")\n"
        "assert isinstance(out.recipe.args[1], t.Tensor)\n"
        "assert t.allclose(out.recipe.args[0], x.array), 'args[0] must equal x.array'\n"
        "assert t.allclose(out.recipe.args[1], y.array), 'args[1] must equal y.array'\n"
        "\n"
        "# --- Recipe.kwargs: empty dict for a pure positional call ---\n"
        "assert out.recipe.kwargs == {}, f'kwargs must be {{}}, got {out.recipe.kwargs!r}'\n"
        "\n"
        "# --- Recipe.parents: BOTH Tensors keyed by argidx, by IDENTITY ---\n"
        "assert isinstance(out.recipe.parents, dict)\n"
        "assert set(out.recipe.parents.keys()) == {0, 1}, (\n"
        "    f'parents must have keys {{0, 1}} for a binary op, got {set(out.recipe.parents.keys())}'\n"
        ")\n"
        "assert out.recipe.parents[0] is x, (\n"
        "    'parents[0] must be the SAME MiniTensor object as x (identity), '\n"
        "    'not a copy or x.array. The reverse pass mutates .grad on the parent.'\n"
        ")\n"
        "assert out.recipe.parents[1] is y, 'parents[1] must be the SAME MiniTensor object as y'\n"
        "\n"
        "# --- Mismatched shapes (broadcast) still works, parents still by argidx ---\n"
        "a = MiniTensor(t.tensor([[1.0, 2.0, 3.0]]))   # (1, 3)\n"
        "b = MiniTensor(t.tensor([[10.0], [20.0]]))    # (2, 1)\n"
        "out2 = add_forward(a, b)\n"
        "assert out2.array.shape == (2, 3), f'broadcast shape wrong: {out2.array.shape}'\n"
        "assert out2.recipe.parents == {0: a, 1: b}, (\n"
        "    f'broadcast case: parents wrong: {out2.recipe.parents}'\n"
        ")\n"
        "\n"
        "# --- Independence: a new call yields a fresh Recipe ---\n"
        "out3 = add_forward(x, y)\n"
        "assert out3.recipe is not out.recipe, 'each call must produce a fresh Recipe'\n"
        "assert out3.recipe.parents[0] is x and out3.recipe.parents[1] is y"
    ),
    solution_body=(
        "def add_forward(x, y):\n"
        "    out = MiniTensor(x.array + y.array)\n"
        "    out.recipe = Recipe(\n"
        "        func=t.add,\n"
        "        args=(x.array, y.array),\n"
        "        kwargs={},\n"
        "        parents={0: x, 1: y},\n"
        "    )\n"
        "    return out"
    ),
    solution_notes=(
        "**Two parents, not one.** A binary op like `add` has two inputs that "
        "both participate in the gradient — `d(x+y)/dx = 1` AND `d(x+y)/dy = 1`. "
        "Both must appear in `parents` so the reverse pass routes the "
        "gradient back to both. Omitting `1: y` would silently drop `y`'s "
        "gradient.\n\n"
        "**`args` unboxes; `parents` does not.** `args` contains the raw "
        "tensors that get passed to the back fn — `add_back0(grad_out, out, "
        "x.array, y.array)`. `parents` keeps the MiniTensor identity so the "
        "reverse pass can write `parent.grad += grad_in`. Two different "
        "purposes, two different containers.\n\n"
        "**Identity (`is`) vs equality (`==`).** The test uses `is` because "
        "the reverse pass mutates the parent's `.grad` attribute — a copy "
        "would receive the mutation in vain. PyTorch's autograd has the same "
        "invariant: `param.grad += ...` only updates the original `param`."
    ),
)


# =========================================================================
# 7. requires-grad-propagation  — ex2
#    ex1 facet: three-gate AND over positional args
#    ex2 facet: kwargs Tensors also participate — any_input_requires_grad
#    must scan BOTH args and kwargs.values()
# =========================================================================

SPEC_REQUIRES_GRAD = _spec(
    atom_id="requires-grad-propagation",
    subtopic="Backprop: requires_grad propagation",
    recap=RECAP_REQUIRES_GRAD_PROP,
    ex_idx=2,
    ex_title="requires_grad propagation also scans kwargs Tensor values",
    slug="requires-grad-propagation-also-scans-kwargs",
    bloom="Apply",
    difficulty_num=3,
    keywords=["requires-grad", "kwargs", "three-gate", "tensor-valued-kwarg"],
    kcs=["requires-grad-propagation", "requires-grad-scan-includes-kwargs"],
    lo=(
        "Apply the three-gate AND (toggle, is_differentiable, any-input) "
        "where any-input scans BOTH positional args AND kwargs.values() for "
        "MiniTensors with requires_grad=True."
    ),
    prompt_body=(
        "Implement `propagate_requires_grad_full(args, kwargs, "
        "is_differentiable, grad_tracking_enabled) -> bool`:\n\n"
        "Return `True` IFF ALL of:\n"
        "1. `grad_tracking_enabled` is `True`.\n"
        "2. `is_differentiable` is `True`.\n"
        "3. At least one MiniTensor in `args` OR in `kwargs.values()` has "
        "`requires_grad == True`.\n\n"
        "Non-MiniTensor entries (ints, floats, lists, plain torch.Tensors) "
        "are filtered out — same as the positional case in ex1. The "
        "extension is that kwargs `.values()` go through the SAME filter "
        "AND OR-combine into the any-test.\n\n"
        "**Why this matters.** Some forward ops take Tensor-valued kwargs "
        "(`F.conv2d(x, weight=W, bias=b)` style). If `x` has "
        "`requires_grad=False` but `W` has `requires_grad=True`, the output "
        "must propagate grad — the parameter is the one being differentiated. "
        "Ignoring kwargs would drop those gradient paths.\n\n"
        "The test sweeps all 8 truth-table combinations of (toggle, "
        "is_diff, has-tracked-input) PLUS a focused set of pos-vs-kwarg "
        "any-input cases."
    ),
    stub=(
        "def propagate_requires_grad_full(\n"
        "    args: tuple,\n"
        "    kwargs: dict,\n"
        "    is_differentiable: bool,\n"
        "    grad_tracking_enabled: bool,\n"
        ") -> bool:\n"
        '    """Three-gate AND scanning both args and kwargs.values() for tracked MiniTensors."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "T1 = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "T0 = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "\n"
        "# --- happy path: all three gates True, kwarg-only tracked Tensor ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T0,), kwargs={'mask': T1},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is True, 'kwarg-only tracked Tensor must propagate'\n"
        "\n"
        "# --- positional tracked, kwarg untracked ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T1,), kwargs={'mask': T0},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is True\n"
        "\n"
        "# --- both tracked ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T1,), kwargs={'mask': T1},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is True\n"
        "\n"
        "# --- NEITHER tracked (across BOTH pos and kwarg) ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T0,), kwargs={'mask': T0},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is False\n"
        "\n"
        "# --- empty inputs ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(), kwargs={},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is False, 'no inputs at all → False'\n"
        "\n"
        "# --- non-MiniTensor kwargs values are skipped (not crashed on) ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T0,), kwargs={'dim': 1, 'keepdim': False, 'mask': T1},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is True, 'int/bool kwargs must be skipped without crashing'\n"
        "\n"
        "# --- non-MiniTensor kwargs values, no tracked anywhere ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T0,), kwargs={'dim': 1, 'keepdim': False},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is False, 'no Tensor anywhere → False'\n"
        "\n"
        "# --- gate 1 OFF: toggle is False ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T1,), kwargs={'mask': T1},\n"
        "    is_differentiable=True, grad_tracking_enabled=False,\n"
        ") is False, 'toggle off vetoes everything'\n"
        "\n"
        "# --- gate 2 OFF: non-differentiable op ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T1,), kwargs={'mask': T1},\n"
        "    is_differentiable=False, grad_tracking_enabled=True,\n"
        ") is False, 'is_differentiable=False vetoes (e.g. torch.equal)'\n"
        "\n"
        "# --- plain torch.Tensor in kwargs must NOT count as a parent ---\n"
        "# (the wrapper sees MiniTensor; raw torch.Tensor is non-input scaffold)\n"
        "raw = t.tensor([1.0])    # NOT a MiniTensor, has no requires_grad-tracked semantics here\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(T0,), kwargs={'x': raw},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is False, 'raw torch.Tensor in kwargs must be filtered out — not a MiniTensor'\n"
        "\n"
        "# --- multiple kwargs, only ONE tracked ---\n"
        "assert propagate_requires_grad_full(\n"
        "    args=(), kwargs={'a': T0, 'b': T0, 'c': T1, 'd': T0},\n"
        "    is_differentiable=True, grad_tracking_enabled=True,\n"
        ") is True, 'any() over kwargs must find the one tracked Tensor'"
    ),
    solution_body=(
        "def propagate_requires_grad_full(args, kwargs, is_differentiable, grad_tracking_enabled):\n"
        "    any_tracked = any(\n"
        "        isinstance(a, MiniTensor) and a.requires_grad for a in args\n"
        "    ) or any(\n"
        "        isinstance(v, MiniTensor) and v.requires_grad for v in kwargs.values()\n"
        "    )\n"
        "    return grad_tracking_enabled and is_differentiable and any_tracked"
    ),
    solution_notes=(
        "**Why kwargs need the same scan.** Many real autograd ops take "
        "tensor-valued kwargs that ARE differentiable: `F.linear(x, "
        "weight=W, bias=b)`, attention's `mask` weight, etc. If "
        "`requires_grad` propagation only looked at positional args, "
        "calling `F.linear(x, weight=W)` with `x.requires_grad=False` but "
        "`W.requires_grad=True` would skip building a Recipe and lose the "
        "gradient path back to `W`.\n\n"
        "**Filter on the input type, not the key.** A kwarg whose VALUE is "
        "a Tensor counts; a kwarg whose value is an int doesn't. The "
        "`isinstance(v, MiniTensor)` filter applied to `.values()` "
        "performs the same role as the positional `isinstance(a, "
        "MiniTensor)` filter — uniform semantics.\n\n"
        "**Short-circuit ordering.** Putting the cheap gates first "
        "(`grad_tracking_enabled`, `is_differentiable`) lets Python's "
        "`and` short-circuit before the linear `any()` scan over inputs. "
        "Tiny win, but it's the natural ordering — global state first, "
        "then per-op flag, then per-input traversal."
    ),
)


# =========================================================================
# 8. unbroadcast-pattern  — ex2
#    ex1 facet: leading axes case + size-1 case in ISOLATION
#    ex2 facet: COMBINED case (both leading-add and size-1-expand in one
#    call) + idempotence — calling twice == calling once
# =========================================================================

SPEC_UNBROADCAST = _spec(
    atom_id="unbroadcast-pattern",
    subtopic="Backprop: Unbroadcast pattern",
    recap=RECAP_UNBROADCAST,
    ex_idx=2,
    ex_title="unbroadcast — combined leading + size-1 case AND idempotence",
    slug="unbroadcast-combined-leading-and-size-1-idempotent",
    bloom="Apply",
    difficulty_num=3,
    keywords=["unbroadcast", "leading-axes", "size-1-axes", "idempotence"],
    kcs=["unbroadcast-pattern", "unbroadcast-is-idempotent"],
    lo=(
        "Apply unbroadcast to the combined case — broadcasting added BOTH "
        "leading axes AND expanded a size-1 dim — and verify the function "
        "is idempotent (unbroadcast(unbroadcast(g, x), x) == unbroadcast(g, x))."
    ),
    prompt_body=(
        "Implement `unbroadcast(grad: Tensor, original: Tensor) -> Tensor` "
        "returning a tensor with shape `original.shape` and values summed "
        "along the broadcast axes.\n\n"
        "Two passes (order matters):\n"
        "1. **Peel leading axes** while `grad.ndim > original.ndim`: "
        "`grad = grad.sum(dim=0)`.\n"
        "2. **Collapse size-1 axes** that got expanded: for each axis `i` "
        "where `original.shape[i] == 1` and `grad.shape[i] != 1`, "
        "`grad = grad.sum(dim=i, keepdim=True)`.\n\n"
        "**Why this drill's facet matters.** ex1 tested each pass in "
        "isolation — leading-axes-only and size-1-only. ex2 tests the "
        "COMBINED case: e.g. `original.shape == (1, 4)`, "
        "`grad.shape == (2, 3, 1, 4)`. Step 1 peels TWO leading axes "
        "(yielding `(1, 4)`); step 2 finds no size-1 expansion to undo. "
        "But if you reordered the passes, you'd try to sum a missing "
        "axis and crash — order locks in.\n\n"
        "ex2 also verifies **idempotence**: once `grad.shape == "
        "original.shape`, a second call must be a no-op (returning a "
        "tensor with the same shape and values). The function is its own "
        "fixed-point under repeated application."
    ),
    stub=(
        "def unbroadcast(grad: Tensor, original: Tensor) -> Tensor:\n"
        '    """Peel leading axes then collapse expanded size-1 axes to match original.shape."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- COMBINED case A: leading axes ADDED + size-1 axis already in original ---\n"
        "# original.shape = (1, 4), grad.shape = (2, 3, 1, 4)\n"
        "g = t.ones(2, 3, 1, 4)\n"
        "x = t.zeros(1, 4)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (1, 4), f'combined A shape: {out.shape}'\n"
        "# Sum over 2*3 = 6 ones in each (1, 4) slot.\n"
        "assert t.allclose(out, t.full((1, 4), 6.0)), f'combined A value: {out}'\n"
        "\n"
        "# --- COMBINED case B: leading axes added AND size-1 axis expanded ---\n"
        "# original.shape = (3, 1), grad.shape = (2, 3, 4) — leading dim 2 to peel\n"
        "# AND inner size-1 axis was expanded to 4.\n"
        "g = t.ones(2, 3, 4)\n"
        "x = t.zeros(3, 1)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (3, 1), f'combined B shape: {out.shape}'\n"
        "# After leading peel: (3, 4) of value 2. After size-1 collapse on axis 1: (3, 1) of value 8.\n"
        "assert t.allclose(out, t.full((3, 1), 8.0)), f'combined B value: {out}'\n"
        "\n"
        "# --- COMBINED case C: multi-axis size-1 collapse + leading peel ---\n"
        "# original.shape = (1, 4, 1), grad.shape = (5, 2, 4, 3)\n"
        "g = t.ones(5, 2, 4, 3)\n"
        "x = t.zeros(1, 4, 1)\n"
        "out = unbroadcast(g, x)\n"
        "assert out.shape == (1, 4, 1), f'combined C shape: {out.shape}'\n"
        "# After leading peel (dim=0 sum once): (2, 4, 3), values 5.\n"
        "# After axis-0 collapse (orig size 1, grad size 2): (1, 4, 3), values 10.\n"
        "# After axis-2 collapse (orig size 1, grad size 3): (1, 4, 1), values 30.\n"
        "assert t.allclose(out, t.full((1, 4, 1), 30.0)), f'combined C value: {out}'\n"
        "\n"
        "# --- IDEMPOTENCE: shapes already match → identity ---\n"
        "g_match = t.randn(3, 4, generator=t.Generator().manual_seed(0))\n"
        "x_match = t.zeros(3, 4)\n"
        "once  = unbroadcast(g_match, x_match)\n"
        "twice = unbroadcast(once,    x_match)\n"
        "assert once.shape == (3, 4)\n"
        "assert twice.shape == (3, 4)\n"
        "assert t.allclose(once, g_match), 'shape-match input must pass through unchanged'\n"
        "assert t.allclose(once, twice), (\n"
        "    'unbroadcast must be idempotent — applying it twice == applying once'\n"
        ")\n"
        "\n"
        "# --- IDEMPOTENCE after a reduction ---\n"
        "g = t.ones(2, 3, 1, 4)\n"
        "x = t.zeros(1, 4)\n"
        "once  = unbroadcast(g, x)\n"
        "twice = unbroadcast(once, x)\n"
        "assert once.shape == twice.shape == (1, 4)\n"
        "assert t.allclose(once, twice), 'idempotence after a non-trivial reduction'\n"
        "\n"
        "# --- gradient-sum correctness via autograd cross-check ---\n"
        "# x: (3, 1), y: (2, 3, 4), out = x * y. dL/dx must have shape (3, 1).\n"
        "xa = t.randn(3, 1, generator=t.Generator().manual_seed(1), requires_grad=True)\n"
        "ya = t.randn(2, 3, 4, generator=t.Generator().manual_seed(2))\n"
        "(xa * ya).sum().backward()\n"
        "expected_gx = xa.grad\n"
        "raw_g = t.ones(2, 3, 4) * ya       # dL/d(out) is ones; chain by y gives raw\n"
        "gx = unbroadcast(raw_g, xa.detach())\n"
        "assert gx.shape == (3, 1)\n"
        "assert t.allclose(gx, expected_gx, atol=1e-5), (\n"
        "    f'cross-check vs autograd: ours={gx} ref={expected_gx}'\n"
        ")"
    ),
    solution_body=(
        "def unbroadcast(grad, original):\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad"
    ),
    solution_notes=(
        "**Why the two passes in THIS order.** The leading-axes pass "
        "reduces `grad.ndim` until it matches `original.ndim`. Only after "
        "that match is reached can the size-1 pass index axes by position — "
        "`original.shape[i]` and `grad.shape[i]` must refer to the same "
        "axis. Reversing the order would index past the end (or hit the "
        "wrong axis) and either crash or silently produce a wrong shape.\n\n"
        "**Why `keepdim=True` in pass 2 but NOT pass 1.** Pass 1 is DROPPING "
        "axes (going from `ndim=4` to `ndim=2`, say) — `keepdim` would defeat "
        "the purpose. Pass 2 must PRESERVE the size-1 axis because "
        "`original.shape` has size-1 at that position; removing it would "
        "produce a shape that doesn't match `original`.\n\n"
        "**Idempotence as a sanity invariant.** If shapes already match, "
        "the while-loop body doesn't run AND the for-loop's `if size == 1 "
        "and grad.shape[i] != 1` is always False (because the shapes match). "
        "Result: `grad` is returned unchanged. Applying `unbroadcast` to a "
        "tensor that already matches its target is a no-op — which is "
        "exactly what \"idempotent\" means."
    ),
)


# -----------------------------------------------------------------------------
# Master list + verify pass + emit.
# -----------------------------------------------------------------------------

ALL_SPECS = [
    SPEC_ARG_POSITION,
    SPEC_CHAIN_RULE,
    SPEC_GRAD_TOGGLE,
    SPEC_KWARGS,
    SPEC_PARENTS,
    SPEC_RECIPE,
    SPEC_REQUIRES_GRAD,
    SPEC_UNBROADCAST,
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

        # Exec the autograd preamble first — every spec depends on
        # MiniTensor, Recipe, and the grad_tracking_enabled global.
        try:
            exec(_AUTOGRAD_PREAMBLE, ns)
        except Exception as e:
            failed.append((tag, f"preamble exec failed: {e!r}", traceback.format_exc()))
            continue

        # Best-effort stub exec (some stubs reference unbound names — tolerable).
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
    print(f"[deepening_i_batch9] Verifying {len(ALL_SPECS)} specs...")
    _verify_all(ALL_SPECS)

    print(f"\n[deepening_i_batch9] All verified — emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_i_batch9] {len(ALL_SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
