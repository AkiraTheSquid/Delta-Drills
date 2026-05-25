#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA part 4 per-op manual backward fns.

Eight single-exercise standalones, under ``prereqs_backward_fns/``:

  * negative-back                    — ex1
  * exp-back                         — ex1
  * reshape-back                     — ex1
  * permute-back-argsort             — ex1
  * sum-back-expand-broadcast        — ex1
  * add-sub-div-back-lambdas         — ex1
  * getitem-back-add-at              — ex1
  * matmul-back-transpose-pair       — ex1

Per-op manual backward functions. Drills are smaller than ARENA composites:
each one isolates a single derivation + shape pattern. Tests use plain
``torch.Tensor`` for shape/value math; we never call ``torch.autograd`` on the
hand-written backward fns themselves.

Backward signature: ``(grad_out, out, *args, **kwargs) -> grad_in``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_backward_fns"


# ---------------------------------------------------------------- atom recaps

RECAP_NEGATIVE = (
    "## `negative_back` — quick refresher\n"
    "\n"
    "`negative(x) = -x` is elementwise and linear. The local derivative is "
    "constant `-1` at every position, so the chain rule collapses to a sign "
    "flip on `grad_out`.\n"
    "\n"
    "**Worked exemplar.**\n"
    "```\n"
    "out  = -x                  # forward\n"
    "d/dx (-x) = -1             # local derivative\n"
    "grad_in = grad_out * (-1)  # chain rule\n"
    "        = -grad_out\n"
    "```\n"
    "\n"
    "Shape of `grad_in` equals shape of `x` — same as `grad_out` for this op."
)

RECAP_EXP = (
    "## `exp_back` — quick refresher\n"
    "\n"
    "`exp(x)` is elementwise with local derivative `exp(x) = out`. Because `out` "
    "is already cached in the back-fn signature, we never need to recompute "
    "`exp`.\n"
    "\n"
    "**Worked exemplar.**\n"
    "```\n"
    "out  = exp(x)              # forward\n"
    "d/dx exp(x) = exp(x) = out # local derivative — use cached out\n"
    "grad_in = grad_out * out   # chain rule\n"
    "```\n"
    "\n"
    "Same shape as `x`. Numerically cheaper than `grad_out * torch.exp(x)` "
    "because the exp was already paid for on the forward pass."
)

RECAP_RESHAPE = (
    "## `reshape_back` — quick refresher\n"
    "\n"
    "`reshape(x, new_shape)` is a pure view rearrangement — every output entry "
    "corresponds 1-to-1 with an input entry, just at a different position. The "
    "local Jacobian is a permutation matrix, so the backward fn is "
    "**reshape `grad_out` back to `x`'s original shape**.\n"
    "\n"
    "**Worked exemplar.**\n"
    "```\n"
    "x.shape         = (2, 6)\n"
    "out = x.reshape(3, 4)\n"
    "grad_out.shape  = (3, 4)\n"
    "grad_in = grad_out.reshape(2, 6)  # restore x's shape\n"
    "```\n"
    "\n"
    "The forward shape is read from `x.shape`; no kwargs needed."
)

RECAP_PERMUTE = (
    "## `permute_back` via argsort — quick refresher\n"
    "\n"
    "`permute(x, dims)` shuffles axes. To undo it, you need the **inverse "
    "permutation** — and `numpy.argsort(dims)` (or `torch.argsort` on a tensor "
    "of dims) computes it for free.\n"
    "\n"
    "**Worked exemplar.**\n"
    "```\n"
    "dims     = (2, 0, 1)               # forward axis order\n"
    "inverse  = argsort(dims) = (1, 2, 0)\n"
    "# apply forward then inverse → identity:\n"
    "x.permute(2,0,1).permute(1,2,0) == x\n"
    "grad_in = grad_out.permute(*inverse)\n"
    "```\n"
    "\n"
    "Why argsort works: `argsort(dims)[i]` is the position of axis `i` in the "
    "permuted order, which is exactly where it needs to come from to get back."
)

RECAP_SUM = (
    "## `sum_back` via expand/broadcast — quick refresher\n"
    "\n"
    "`sum(x, dim=k, keepdim=False)` collapses axis `k`. Each input entry "
    "`x[..., i, ...]` contributes to exactly ONE output entry, so the backward "
    "pass broadcasts `grad_out` back along the summed axis.\n"
    "\n"
    "**Worked exemplar.**\n"
    "```\n"
    "x.shape         = (3, 4)\n"
    "out = x.sum(dim=0)             # out.shape = (4,)\n"
    "grad_out.shape  = (4,)\n"
    "# 1. restore the collapsed axis as size-1:\n"
    "g = grad_out.unsqueeze(0)      # (1, 4)\n"
    "# 2. expand back to x's shape:\n"
    "grad_in = g.expand(3, 4)       # broadcast — no copy\n"
    "```\n"
    "\n"
    "With `keepdim=True`, the unsqueeze step is skipped — `grad_out` already "
    "has the size-1 axis."
)

RECAP_LAMBDAS = (
    "## `add` / `sub` / `div` back as lambdas — quick refresher\n"
    "\n"
    "When two backward fns are short one-liners with the same `(grad_out, out, "
    "x, y)` signature, define them inline as lambdas keyed by `(op, argnum)` — "
    "no `def` boilerplate, no per-fn name to remember.\n"
    "\n"
    "**Worked exemplar.** For `out = x - y`:\n"
    "```\n"
    "d(x - y)/dx =  1   →  sub_back0 = lambda g, o, x, y:  g\n"
    "d(x - y)/dy = -1   →  sub_back1 = lambda g, o, x, y: -g\n"
    "```\n"
    "\n"
    "Use a dict to store all 6 lambdas (3 ops × 2 argnums) so the dispatcher "
    "can look one up with `BACK[(op, argnum)](grad_out, out, x, y)`."
)

RECAP_GETITEM = (
    "## `getitem_back` via add-at — quick refresher\n"
    "\n"
    "`out = x[idx]` reads a subset of `x`. The backward fn must place "
    "`grad_out` into a zero tensor at `idx` — but plain assignment fails when "
    "`idx` repeats (you'd overwrite). Use **scatter-add** semantics: any index "
    "appearing N times accumulates N contributions.\n"
    "\n"
    "**Worked exemplar.**\n"
    "```\n"
    "x.shape  = (5,)\n"
    "idx      = [0, 2, 0]            # index 0 appears twice\n"
    "out      = x[idx]               # out.shape = (3,)\n"
    "grad_in  = zeros_like(x)\n"
    "grad_in.index_add_(0, idx, grad_out)   # accumulates at repeated idx\n"
    "# grad_in == [g[0]+g[2], 0, g[1], 0, 0]\n"
    "```\n"
    "\n"
    "Repeat indices = gradients sum. Plain `grad_in[idx] = grad_out` would "
    "drop one of the two contributions to position 0."
)

RECAP_MATMUL = (
    "## `matmul_back` transpose pair — quick refresher\n"
    "\n"
    "`out = x @ y` (with `x: (m,k)`, `y: (k,n)`, `out: (m,n)`). Both backward "
    "fns are themselves matmuls — each contracts `grad_out` with the OTHER "
    "input transposed.\n"
    "\n"
    "**Worked exemplar.** Shape-driven derivation:\n"
    "```\n"
    "dL/dx must be (m,k)  →  grad_out @ y.T   :  (m,n) @ (n,k) = (m,k)  ✓\n"
    "dL/dy must be (k,n)  →  x.T @ grad_out   :  (k,m) @ (m,n) = (k,n)  ✓\n"
    "```\n"
    "\n"
    "The transpose-on-the-other-input pattern is general: in `A @ B`, the "
    "gradient w.r.t. `A` involves `B.T`, and vice versa."
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
        "extra_imports": extra_imports or [],
    }


# =========================================================================
# atom: negative-back
# =========================================================================

SPEC_NEGATIVE = _spec(
    atom_id="negative-back",
    subtopic="Backprop: negative_back",
    recap=RECAP_NEGATIVE,
    ex_idx=1,
    ex_title="negative_back — sign flip of grad_out",
    slug="negative-back-sign-flip-of-grad-out",
    bloom="Apply",
    difficulty_num=1,
    keywords=["negative", "elementwise", "sign-flip"],
    kcs=["chain-rule-elementwise", "backward-fn-signature"],
    lo=(
        "Apply the elementwise chain rule to derive negative_back, returning "
        "grad_in = -grad_out with the same shape and dtype as x."
    ),
    prompt_body=(
        "Implement `negative_back(grad_out, out, x)` for the forward op "
        "`out = -x`.\n\n"
        "Derivation:\n"
        "- `d/dx (-x) = -1` at every position.\n"
        "- Chain rule: `dL/dx = grad_out * (-1) = -grad_out`.\n\n"
        "Return a `torch.Tensor` with the same shape as `x`. No autograd, "
        "no in-place mutation — return a new tensor."
    ),
    stub=(
        "def negative_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """dL/dx for out = -x."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- scalar ---\n"
        "x = t.tensor([3.0])\n"
        "out = -x\n"
        "g = negative_back(t.tensor([1.0]), out, x)\n"
        "assert t.allclose(g, t.tensor([-1.0])), f'scalar: {g}'\n"
        "\n"
        "# --- vector with non-unit grad_out ---\n"
        "x = t.tensor([1.0, -2.0, 3.0])\n"
        "out = -x\n"
        "grad_out = t.tensor([5.0, 7.0, -2.0])\n"
        "g = negative_back(grad_out, out, x)\n"
        "assert g.shape == x.shape, f'shape: {g.shape}'\n"
        "assert t.allclose(g, -grad_out), f'value: {g}'\n"
        "\n"
        "# --- matrix shape ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "g = negative_back(G, -X, X)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, -G)\n"
        "\n"
        "# --- not in-place: grad_out must be unchanged ---\n"
        "g_in = t.tensor([1.0, 2.0, 3.0])\n"
        "g_in_copy = g_in.clone()\n"
        "_ = negative_back(g_in, -t.tensor([0.0, 0.0, 0.0]), t.tensor([0.0, 0.0, 0.0]))\n"
        "assert t.allclose(g_in, g_in_copy), 'negative_back must not mutate grad_out'\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([1.5, -0.3, 2.7], requires_grad=True)\n"
        "y = (-x_ref).sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "g_ours = negative_back(t.ones(3), -x_det, x_det)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def negative_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d/dx (-x) = -1; chain rule collapses to grad_in = -grad_out.\n"
        "    return -grad_out"
    ),
    solution_notes=(
        "**One-liner.** Elementwise + constant derivative = pure sign flip.\n\n"
        "**Why neither `out` nor `x` is read.** The local derivative is the "
        "constant `-1` — independent of position and of the input value. Some "
        "back fns ignore `out`, some ignore `x`, some use both. The uniform "
        "signature carries all three so any back fn can be dispatched the "
        "same way."
    ),
)


# =========================================================================
# atom: exp-back
# =========================================================================

SPEC_EXP = _spec(
    atom_id="exp-back",
    subtopic="Backprop: exp_back",
    recap=RECAP_EXP,
    ex_idx=1,
    ex_title="exp_back — reuse cached out",
    slug="exp-back-reuse-cached-out",
    bloom="Apply",
    difficulty_num=2,
    keywords=["exp", "elementwise", "cached-out"],
    kcs=["chain-rule-elementwise", "back-fn-uses-cached-out"],
    lo=(
        "Apply the elementwise chain rule to derive exp_back, reusing the "
        "cached `out` tensor instead of recomputing exp(x)."
    ),
    prompt_body=(
        "Implement `exp_back(grad_out, out, x)` for the forward op "
        "`out = exp(x)`.\n\n"
        "Derivation:\n"
        "- `d/dx exp(x) = exp(x) = out`.\n"
        "- Chain rule: `dL/dx = grad_out * out`.\n\n"
        "**Use the cached `out`** — do NOT call `torch.exp(x)` inside the "
        "backward fn. The whole reason the backward signature includes `out` "
        "is to skip recomputation.\n\n"
        "Return a `torch.Tensor` with the same shape as `x`. No autograd."
    ),
    stub=(
        "def exp_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """dL/dx for out = exp(x). Use cached out — do not recompute."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- scalar ---\n"
        "x = t.tensor([0.0])\n"
        "out = t.exp(x)              # = [1.0]\n"
        "g = exp_back(t.tensor([1.0]), out, x)\n"
        "assert t.allclose(g, t.tensor([1.0])), f'at x=0: {g}'\n"
        "\n"
        "# --- vector ---\n"
        "x = t.tensor([0.0, 1.0, 2.0])\n"
        "out = t.exp(x)\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "g = exp_back(grad_out, out, x)\n"
        "expected = out                   # grad_out is all-ones, so g == out\n"
        "assert g.shape == x.shape\n"
        "assert t.allclose(g, expected), f'value: {g} vs {expected}'\n"
        "\n"
        "# --- non-unit grad_out: chain-rule scaling ---\n"
        "grad_out = t.tensor([3.0, -2.0, 5.0])\n"
        "g = exp_back(grad_out, out, x)\n"
        "assert t.allclose(g, grad_out * out), 'chain-rule scaling failed'\n"
        "\n"
        "# --- matrix ---\n"
        "rng = t.Generator().manual_seed(1)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "OUT = t.exp(X)\n"
        "g = exp_back(G, OUT, X)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, G * OUT)\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([-0.5, 0.0, 0.8, 1.5], requires_grad=True)\n"
        "y = t.exp(x_ref).sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "out_cached = t.exp(x_det)\n"
        "g_ours = exp_back(t.ones(4), out_cached, x_det)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def exp_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d/dx exp(x) = exp(x) = out. Reuse the cached out — no recompute.\n"
        "    return grad_out * out"
    ),
    solution_notes=(
        "**One multiplication, no exp call.** The forward already paid for the "
        "exponential. Reading it back from `out` saves one elementwise exp on "
        "every reverse pass.\n\n"
        "**Why not `grad_out * t.exp(x)`.** Equivalent numerically (modulo "
        "float rounding) but costs one redundant exp per node. On long chains "
        "this is the difference between O(n) and O(2n) exp evaluations."
    ),
)


# =========================================================================
# atom: reshape-back
# =========================================================================

SPEC_RESHAPE = _spec(
    atom_id="reshape-back",
    subtopic="Backprop: reshape_back",
    recap=RECAP_RESHAPE,
    ex_idx=1,
    ex_title="reshape_back — restore x's original shape",
    slug="reshape-back-restore-original-shape",
    bloom="Apply",
    difficulty_num=2,
    keywords=["reshape", "shape-restore", "view-op"],
    kcs=["reshape-backward-pattern", "backward-fn-signature"],
    lo=(
        "Apply the view-op backward pattern: reshape grad_out back to x's "
        "original shape using x.shape read from the cached input."
    ),
    prompt_body=(
        "Implement `reshape_back(grad_out, out, x, new_shape)` for the "
        "forward op `out = x.reshape(new_shape)`.\n\n"
        "Derivation:\n"
        "- Reshape is a pure permutation of the storage — every output entry "
        "corresponds to exactly one input entry.\n"
        "- The local Jacobian is a permutation matrix, so the backward fn is "
        "the inverse reshape: `grad_in = grad_out.reshape(x.shape)`.\n\n"
        "The `new_shape` argument is part of the forward signature (so the "
        "wrapper passes it through) but you don't actually need it on the "
        "backward — `x.shape` is the authoritative shape to restore.\n\n"
        "Return a `torch.Tensor` with the same shape as `x`. No autograd."
    ),
    stub=(
        "def reshape_back(grad_out: Tensor, out: Tensor, x: Tensor, new_shape: tuple) -> Tensor:\n"
        '    """dL/dx for out = x.reshape(new_shape). Restore x.shape."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- flatten + restore ---\n"
        "x = t.arange(12.0).reshape(3, 4)\n"
        "out = x.reshape(12)\n"
        "grad_out = t.arange(12.0)\n"
        "g = reshape_back(grad_out, out, x, (12,))\n"
        "assert g.shape == x.shape, f'shape: {g.shape}'\n"
        "assert t.allclose(g, grad_out.reshape(3, 4))\n"
        "\n"
        "# --- (2, 6) → (3, 4) ---\n"
        "x = t.arange(12.0).reshape(2, 6)\n"
        "out = x.reshape(3, 4)\n"
        "grad_out = t.randn(3, 4, generator=t.Generator().manual_seed(0))\n"
        "g = reshape_back(grad_out, out, x, (3, 4))\n"
        "assert g.shape == (2, 6), f'shape: {g.shape}'\n"
        "# Values must be the same memory order — reshape preserves storage.\n"
        "assert t.allclose(g.flatten(), grad_out.flatten())\n"
        "\n"
        "# --- adding a dim ---\n"
        "x = t.arange(6.0)               # (6,)\n"
        "out = x.reshape(2, 3)\n"
        "grad_out = t.tensor([[1., 2., 3.], [4., 5., 6.]])\n"
        "g = reshape_back(grad_out, out, x, (2, 3))\n"
        "assert g.shape == (6,)\n"
        "assert t.allclose(g, t.tensor([1., 2., 3., 4., 5., 6.]))\n"
        "\n"
        "# --- 3D reshape ---\n"
        "rng = t.Generator().manual_seed(2)\n"
        "X = t.randn(2, 3, 4, generator=rng)\n"
        "OUT = X.reshape(6, 4)\n"
        "G = t.randn(6, 4, generator=rng)\n"
        "g = reshape_back(G, OUT, X, (6, 4))\n"
        "assert g.shape == (2, 3, 4)\n"
        "assert t.allclose(g.reshape(6, 4), G)\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.randn(4, 5, requires_grad=True, generator=t.Generator().manual_seed(3))\n"
        "y = x_ref.reshape(20).sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "g_ours = reshape_back(t.ones(20), x_det.reshape(20), x_det, (20,))\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'disagrees with autograd: ours sum={g_ours.sum()}, ref sum={x_ref.grad.sum()}'\n"
        ")"
    ),
    solution_body=(
        "def reshape_back(grad_out: Tensor, out: Tensor, x: Tensor, new_shape: tuple) -> Tensor:\n"
        "    # Reshape is a permutation of storage; backward = inverse reshape.\n"
        "    # x.shape is the authoritative target — new_shape is unused here.\n"
        "    return grad_out.reshape(x.shape)"
    ),
    solution_notes=(
        "**Why `x.shape`, not `new_shape`.** They're inverses, but `x.shape` "
        "is direct: the target we want `grad_in` to wear. Using `new_shape` "
        "would force you to also store `x.shape` somewhere else.\n\n"
        "**View ops all look like this.** Anything that's just a storage "
        "re-interpretation (`reshape`, `view`, `flatten`, `squeeze`, "
        "`unsqueeze`) has the same shape: backward = inverse-of-the-shape-op. "
        "The values don't change, only the layout."
    ),
)


# =========================================================================
# atom: permute-back-argsort
# =========================================================================

SPEC_PERMUTE = _spec(
    atom_id="permute-back-argsort",
    subtopic="Backprop: permute_back via argsort",
    recap=RECAP_PERMUTE,
    ex_idx=1,
    ex_title="permute_back — inverse permutation via argsort",
    slug="permute-back-inverse-permutation-via-argsort",
    bloom="Apply",
    difficulty_num=3,
    keywords=["permute", "argsort", "inverse-permutation"],
    kcs=["permute-backward-pattern", "inverse-permutation-via-argsort"],
    lo=(
        "Apply argsort to compute the inverse permutation of `dims`, then "
        "permute grad_out by that inverse to restore x's axis order."
    ),
    prompt_body=(
        "Implement `permute_back(grad_out, out, x, dims)` for the forward op "
        "`out = x.permute(*dims)`.\n\n"
        "Derivation:\n"
        "- `permute` only reorders axes — no value changes.\n"
        "- The backward is `grad_out` permuted by the **inverse** of `dims`.\n"
        "- `numpy.argsort(dims)` returns that inverse: for each position `i`, "
        "it's the index at which `i` appears in `dims`.\n\n"
        "Concretely:\n"
        "1. Compute `inverse = tuple(np.argsort(dims).tolist())`.\n"
        "2. Return `grad_out.permute(*inverse)`.\n\n"
        "`dims` is a tuple of ints, e.g. `(2, 0, 1)`. Return a `torch.Tensor` "
        "with the same shape as `x`. No autograd."
    ),
    stub=(
        "def permute_back(grad_out: Tensor, out: Tensor, x: Tensor, dims: tuple) -> Tensor:\n"
        '    """dL/dx for out = x.permute(*dims). Use argsort to invert dims."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- identity permutation: (0, 1, 2) ---\n"
        "x = t.arange(24.0).reshape(2, 3, 4)\n"
        "out = x.permute(0, 1, 2)\n"
        "grad_out = t.randn(2, 3, 4, generator=t.Generator().manual_seed(0))\n"
        "g = permute_back(grad_out, out, x, (0, 1, 2))\n"
        "assert g.shape == x.shape\n"
        "assert t.allclose(g, grad_out), 'identity perm: g should equal grad_out'\n"
        "\n"
        "# --- 2D transpose: dims=(1, 0) ---\n"
        "x = t.arange(12.0).reshape(3, 4)\n"
        "out = x.permute(1, 0)                # (4, 3)\n"
        "grad_out = t.arange(12.0).reshape(4, 3)\n"
        "g = permute_back(grad_out, out, x, (1, 0))\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, grad_out.permute(1, 0))\n"
        "\n"
        "# --- 3D cycle: dims=(2, 0, 1) — inverse is (1, 2, 0) ---\n"
        "x = t.arange(24.0).reshape(2, 3, 4)\n"
        "out = x.permute(2, 0, 1)             # (4, 2, 3)\n"
        "grad_out = t.randn(4, 2, 3, generator=t.Generator().manual_seed(1))\n"
        "g = permute_back(grad_out, out, x, (2, 0, 1))\n"
        "assert g.shape == (2, 3, 4)\n"
        "# Sanity: applying forward then back to grad_out should reproduce grad_out.\n"
        "assert t.allclose(g.permute(2, 0, 1), grad_out)\n"
        "\n"
        "# --- 4D arbitrary perm: (3, 1, 0, 2) ---\n"
        "rng = t.Generator().manual_seed(2)\n"
        "X = t.randn(2, 3, 4, 5, generator=rng)\n"
        "dims = (3, 1, 0, 2)                 # forward: (5, 3, 2, 4)\n"
        "OUT = X.permute(*dims)\n"
        "G = t.randn(*OUT.shape, generator=rng)\n"
        "g = permute_back(G, OUT, X, dims)\n"
        "assert g.shape == X.shape, f'shape: {g.shape}'\n"
        "# Round-trip: permuting g forward must give G back.\n"
        "assert t.allclose(g.permute(*dims), G)\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.randn(2, 3, 4, requires_grad=True, generator=t.Generator().manual_seed(3))\n"
        "y = x_ref.permute(2, 0, 1).sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "g_ours = permute_back(t.ones(4, 2, 3), x_det.permute(2, 0, 1), x_det, (2, 0, 1))\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), 'disagrees with autograd'"
    ),
    solution_body=(
        "def permute_back(grad_out: Tensor, out: Tensor, x: Tensor, dims: tuple) -> Tensor:\n"
        "    # Inverse permutation via argsort: position of each axis in dims.\n"
        "    inverse = tuple(int(i) for i in np.argsort(dims))\n"
        "    return grad_out.permute(*inverse)"
    ),
    solution_notes=(
        "**Why argsort gives the inverse.** If `dims[k] = j`, then axis `j` of "
        "`x` lands at position `k` of `out`. The inverse must send position "
        "`j` of `grad_out` back to axis... well, the position where `j` "
        "appears in `dims`, which is exactly `argsort(dims)[j]`.\n\n"
        "**Don't hand-roll the inverse.** For a small `dims=(1,0)` it's "
        "tempting to write `(1,0)` by inspection. But the same back fn must "
        "handle 4-D, 5-D, arbitrary permutations — `argsort` works uniformly "
        "and never has off-by-one bugs."
    ),
)


# =========================================================================
# atom: sum-back-expand-broadcast
# =========================================================================

SPEC_SUM = _spec(
    atom_id="sum-back-expand-broadcast",
    subtopic="Backprop: sum_back via expand_broadcast",
    recap=RECAP_SUM,
    ex_idx=1,
    ex_title="sum_back — unsqueeze + expand back to x.shape",
    slug="sum-back-unsqueeze-expand-to-x-shape",
    bloom="Apply",
    difficulty_num=3,
    keywords=["sum", "expand", "broadcast", "keepdim"],
    kcs=["sum-backward-pattern", "kwargs-pass-through-recipe"],
    lo=(
        "Apply the expand-broadcast pattern: restore the collapsed axis (if "
        "keepdim=False) then broadcast grad_out back to x.shape."
    ),
    prompt_body=(
        "Implement `sum_back(grad_out, out, x, dim, keepdim=False)` for the "
        "forward op `out = x.sum(dim=dim, keepdim=keepdim)`.\n\n"
        "Derivation:\n"
        "- Each input entry `x[..., i, ...]` contributes to ONE output entry. "
        "Local derivative is 1 at every position.\n"
        "- Backward broadcasts `grad_out` back along the summed axis.\n\n"
        "Two cases:\n\n"
        "**`keepdim=True`** — `grad_out` already has a size-1 axis at `dim`. "
        "Just `expand` to `x.shape`:\n"
        "```\n"
        "grad_in = grad_out.expand(x.shape)\n"
        "```\n\n"
        "**`keepdim=False`** — `grad_out` is missing the axis. Unsqueeze it "
        "back to size 1, then expand:\n"
        "```\n"
        "g = grad_out.unsqueeze(dim)\n"
        "grad_in = g.expand(x.shape)\n"
        "```\n\n"
        "Return a `torch.Tensor` with the same shape as `x`. `dim` is a "
        "single int (no multi-dim sum in this drill). No autograd."
    ),
    stub=(
        "def sum_back(grad_out: Tensor, out: Tensor, x: Tensor, dim: int, keepdim: bool = False) -> Tensor:\n"
        '    """dL/dx for out = x.sum(dim=dim, keepdim=keepdim)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- sum dim=0, keepdim=False ---\n"
        "x = t.arange(12.0).reshape(3, 4)\n"
        "out = x.sum(dim=0)                  # (4,)\n"
        "grad_out = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "g = sum_back(grad_out, out, x, dim=0, keepdim=False)\n"
        "assert g.shape == (3, 4), f'shape: {g.shape}'\n"
        "# Every row of g is grad_out broadcast.\n"
        "assert t.allclose(g, grad_out.unsqueeze(0).expand(3, 4))\n"
        "\n"
        "# --- sum dim=1, keepdim=False ---\n"
        "out = x.sum(dim=1)                  # (3,)\n"
        "grad_out = t.tensor([10.0, 20.0, 30.0])\n"
        "g = sum_back(grad_out, out, x, dim=1, keepdim=False)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, grad_out.unsqueeze(1).expand(3, 4))\n"
        "# Spot-check: row 0 of g must be [10, 10, 10, 10].\n"
        "assert t.allclose(g[0], t.full((4,), 10.0))\n"
        "\n"
        "# --- sum dim=0, keepdim=True ---\n"
        "out_kd = x.sum(dim=0, keepdim=True)  # (1, 4)\n"
        "grad_out_kd = t.tensor([[1.0, 2.0, 3.0, 4.0]])\n"
        "g = sum_back(grad_out_kd, out_kd, x, dim=0, keepdim=True)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, grad_out_kd.expand(3, 4))\n"
        "\n"
        "# --- 3D ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(2, 3, 4, generator=rng)\n"
        "for d in range(3):\n"
        "    OUT = X.sum(dim=d)\n"
        "    G = t.randn(*OUT.shape, generator=rng)\n"
        "    g = sum_back(G, OUT, X, dim=d, keepdim=False)\n"
        "    assert g.shape == X.shape\n"
        "    assert t.allclose(g, G.unsqueeze(d).expand(*X.shape)), f'dim={d}'\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.randn(3, 4, requires_grad=True, generator=t.Generator().manual_seed(5))\n"
        "y = x_ref.sum(dim=1).sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "out_cached = x_det.sum(dim=1)\n"
        "g_ours = sum_back(t.ones(3), out_cached, x_det, dim=1, keepdim=False)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), 'disagrees with autograd'"
    ),
    solution_body=(
        "def sum_back(grad_out: Tensor, out: Tensor, x: Tensor, dim: int, keepdim: bool = False) -> Tensor:\n"
        "    # Restore the collapsed axis if keepdim=False, then broadcast.\n"
        "    g = grad_out if keepdim else grad_out.unsqueeze(dim)\n"
        "    return g.expand(x.shape)"
    ),
    solution_notes=(
        "**Why `expand` not `repeat`.** `expand` returns a view — no memory "
        "allocation, just a stride trick. `repeat` copies. For the reverse "
        "pass, expand is the right call: downstream ops that read the "
        "broadcast grad don't need a contiguous buffer.\n\n"
        "**Why kwargs matter here.** This is the exemplar of why `Recipe` "
        "must store `kwargs`. If the forward was `x.sum(dim=1)` and you "
        "forgot to thread `dim=1` into the recipe, this back fn doesn't "
        "know which axis to restore. The shape error would surface here, "
        "not at the forward."
    ),
)


# =========================================================================
# atom: add-sub-div-back-lambdas
# =========================================================================

SPEC_LAMBDAS = _spec(
    atom_id="add-sub-div-back-lambdas",
    subtopic="Backprop: add/sub/div back as lambdas",
    recap=RECAP_LAMBDAS,
    ex_idx=1,
    ex_title="build BACK dict — 6 lambdas for add/sub/div × arg0/arg1",
    slug="back-dict-six-lambdas-for-add-sub-div",
    bloom="Apply",
    difficulty_num=3,
    keywords=["lambda", "back-dict", "add", "sub", "div", "arg-position"],
    kcs=["arg-position-back-functions", "backward-fn-signature"],
    lo=(
        "Apply the lambda-keyed back-fn dispatch pattern: build a dict mapping "
        "(op_name, argnum) → backward lambda for add, sub, div."
    ),
    prompt_body=(
        "Build a single dict `BACK` mapping `(op_name, argnum)` to a backward "
        "lambda. Six entries total: `add`, `sub`, `div` × argnum `0`, `1`.\n\n"
        "Each lambda has signature `(grad_out, out, x, y) -> Tensor` and "
        "returns `dL/dx` for argnum=0 or `dL/dy` for argnum=1.\n\n"
        "Derivations (no broadcasting in this drill — assume `x.shape == "
        "y.shape == out.shape`):\n\n"
        "```\n"
        "out = x + y:\n"
        "  d/dx =  1   →  BACK[('add', 0)] = lambda g, o, x, y:  g\n"
        "  d/dy =  1   →  BACK[('add', 1)] = lambda g, o, x, y:  g\n"
        "\n"
        "out = x - y:\n"
        "  d/dx =  1   →  BACK[('sub', 0)] = lambda g, o, x, y:  g\n"
        "  d/dy = -1   →  BACK[('sub', 1)] = lambda g, o, x, y: -g\n"
        "\n"
        "out = x / y:\n"
        "  d/dx =  1/y      →  BACK[('div', 0)] = lambda g, o, x, y:  g / y\n"
        "  d/dy = -x/y**2   →  BACK[('div', 1)] = lambda g, o, x, y: -g * x / (y * y)\n"
        "```\n\n"
        "**Why lambdas, not `def`.** Each body is a one-liner; the def "
        "boilerplate would be louder than the math. Storing them in a dict "
        "by `(op, argnum)` also matches how the autograd dispatcher actually "
        "looks back fns up at reverse time.\n\n"
        "Build `BACK` as a module-level dict. No autograd."
    ),
    stub=(
        "# Build BACK: dict[(str, int), Callable]\n"
        "# Keys: ('add', 0), ('add', 1), ('sub', 0), ('sub', 1), ('div', 0), ('div', 1)\n"
        "# Each value is a lambda(grad_out, out, x, y) -> Tensor.\n"
        "BACK = {\n"
        "    # fill me in\n"
        "}"
    ),
    test_body=(
        "# --- key set is exactly the 6 expected pairs ---\n"
        "expected_keys = {('add', 0), ('add', 1), ('sub', 0), ('sub', 1), ('div', 0), ('div', 1)}\n"
        "assert set(BACK.keys()) == expected_keys, (\n"
        "    f'BACK keys mismatch: extra={set(BACK.keys()) - expected_keys}, '\n"
        "    f'missing={expected_keys - set(BACK.keys())}'\n"
        ")\n"
        "\n"
        "# --- all values are callable ---\n"
        "for k, v in BACK.items():\n"
        "    assert callable(v), f'BACK[{k}] is not callable'\n"
        "\n"
        "# --- add ---\n"
        "x = t.tensor([1.0, 2.0, 3.0])\n"
        "y = t.tensor([4.0, 5.0, 6.0])\n"
        "out = x + y\n"
        "g_in = t.tensor([7.0, 8.0, 9.0])\n"
        "assert t.allclose(BACK[('add', 0)](g_in, out, x, y), g_in)\n"
        "assert t.allclose(BACK[('add', 1)](g_in, out, x, y), g_in)\n"
        "\n"
        "# --- sub ---\n"
        "out = x - y\n"
        "assert t.allclose(BACK[('sub', 0)](g_in, out, x, y),  g_in)\n"
        "assert t.allclose(BACK[('sub', 1)](g_in, out, x, y), -g_in)\n"
        "\n"
        "# --- div ---\n"
        "x = t.tensor([6.0, 8.0, 10.0])\n"
        "y = t.tensor([2.0, 4.0, 5.0])\n"
        "out = x / y\n"
        "g_in = t.tensor([1.0, 1.0, 1.0])\n"
        "g0 = BACK[('div', 0)](g_in, out, x, y)\n"
        "g1 = BACK[('div', 1)](g_in, out, x, y)\n"
        "assert t.allclose(g0, 1 / y)\n"
        "assert t.allclose(g1, -x / (y * y))\n"
        "\n"
        "# --- div: non-unit grad_out, witnessed by autograd ---\n"
        "x_ref = t.tensor([3.0, 5.0, 7.0], requires_grad=True)\n"
        "y_ref = t.tensor([2.0, 4.0, 6.0], requires_grad=True)\n"
        "z = (x_ref / y_ref).sum()\n"
        "z.backward()\n"
        "x_det, y_det = x_ref.detach(), y_ref.detach()\n"
        "out_cached = x_det / y_det\n"
        "g0_ours = BACK[('div', 0)](t.ones(3), out_cached, x_det, y_det)\n"
        "g1_ours = BACK[('div', 1)](t.ones(3), out_cached, x_det, y_det)\n"
        "assert t.allclose(g0_ours, x_ref.grad, atol=1e-6)\n"
        "assert t.allclose(g1_ours, y_ref.grad, atol=1e-6)\n"
        "\n"
        "# --- asymmetric ops must produce DIFFERENT results at the two argnums ---\n"
        "x_t = t.tensor([3.0, 5.0])\n"
        "y_t = t.tensor([2.0, 4.0])\n"
        "g_t = t.ones(2)\n"
        "assert not t.allclose(\n"
        "    BACK[('sub', 0)](g_t, x_t - y_t, x_t, y_t),\n"
        "    BACK[('sub', 1)](g_t, x_t - y_t, x_t, y_t),\n"
        "), 'sub back0 and back1 must differ'\n"
        "assert not t.allclose(\n"
        "    BACK[('div', 0)](g_t, x_t / y_t, x_t, y_t),\n"
        "    BACK[('div', 1)](g_t, x_t / y_t, x_t, y_t),\n"
        "), 'div back0 and back1 must differ'"
    ),
    solution_body=(
        "BACK = {\n"
        "    ('add', 0): lambda g, o, x, y:  g,\n"
        "    ('add', 1): lambda g, o, x, y:  g,\n"
        "    ('sub', 0): lambda g, o, x, y:  g,\n"
        "    ('sub', 1): lambda g, o, x, y: -g,\n"
        "    ('div', 0): lambda g, o, x, y:  g / y,\n"
        "    ('div', 1): lambda g, o, x, y: -g * x / (y * y),\n"
        "}"
    ),
    solution_notes=(
        "**Symmetric ops still get two entries.** `('add', 0)` and `('add', 1)` "
        "have identical bodies. The dispatcher doesn't know — it just looks up "
        "`(op, argnum)` and calls. Skip the second entry and `add(x, y).backward()` "
        "fails to flow grad to `y`.\n\n"
        "**Lambdas vs `def`.** For genuinely one-line bodies, lambdas are "
        "fine. For anything with broadcasting (`unbroadcast` wrappers, "
        "conditional dim handling), promote to a named `def` — the dispatch "
        "key stays the same."
    ),
)


# =========================================================================
# atom: getitem-back-add-at
# =========================================================================

SPEC_GETITEM = _spec(
    atom_id="getitem-back-add-at",
    subtopic="Backprop: getitem_back via add-at",
    recap=RECAP_GETITEM,
    ex_idx=1,
    ex_title="getitem_back — index_add_ into zeros_like(x)",
    slug="getitem-back-index-add-into-zeros",
    bloom="Apply",
    difficulty_num=3,
    keywords=["getitem", "index", "scatter-add", "index_add"],
    kcs=["getitem-backward-pattern", "scatter-add-for-repeated-indices"],
    lo=(
        "Apply scatter-add semantics to place grad_out into a zeros tensor at "
        "idx, accumulating contributions for any repeated indices."
    ),
    prompt_body=(
        "Implement `getitem_back(grad_out, out, x, idx)` for the forward op "
        "`out = x[idx]`. For this drill `idx` is a 1-D `torch.LongTensor` and "
        "`x` is 1-D — we're indexing along axis 0.\n\n"
        "Derivation:\n"
        "- `out[i] = x[idx[i]]`, so `d(out[i])/d(x[j]) = 1` if `j == idx[i]` "
        "else `0`.\n"
        "- Chain rule: `dL/dx[j] = sum_i (grad_out[i] if idx[i] == j else 0)`.\n"
        "- **Repeated indices SUM** — index 0 appearing twice contributes "
        "twice.\n\n"
        "Implementation:\n"
        "1. Allocate `grad_in = torch.zeros_like(x)`.\n"
        "2. Use `grad_in.index_add_(0, idx, grad_out)` to accumulate at the "
        "right positions.\n"
        "3. Return `grad_in`.\n\n"
        "**Why not `grad_in[idx] = grad_out`.** With repeated indices, the "
        "last write wins — you'd drop one of the contributions. `index_add_` "
        "is the scatter-add primitive; it accumulates instead of overwriting.\n\n"
        "Return a `torch.Tensor` with the same shape as `x`. No autograd."
    ),
    stub=(
        "def getitem_back(grad_out: Tensor, out: Tensor, x: Tensor, idx: Tensor) -> Tensor:\n"
        '    """dL/dx for out = x[idx]. Scatter-add grad_out into zeros_like(x)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- unique indices ---\n"
        "x = t.tensor([10.0, 20.0, 30.0, 40.0, 50.0])\n"
        "idx = t.tensor([1, 3], dtype=t.long)\n"
        "out = x[idx]\n"
        "grad_out = t.tensor([7.0, 9.0])\n"
        "g = getitem_back(grad_out, out, x, idx)\n"
        "assert g.shape == x.shape, f'shape: {g.shape}'\n"
        "assert t.allclose(g, t.tensor([0.0, 7.0, 0.0, 9.0, 0.0]))\n"
        "\n"
        "# --- repeated indices must SUM ---\n"
        "idx = t.tensor([0, 2, 0], dtype=t.long)\n"
        "out = x[idx]\n"
        "grad_out = t.tensor([1.0, 5.0, 4.0])\n"
        "g = getitem_back(grad_out, out, x, idx)\n"
        "# pos 0 gets contributions from grad_out[0]=1 AND grad_out[2]=4 → 5.\n"
        "# pos 2 gets grad_out[1]=5.\n"
        "assert t.allclose(g, t.tensor([5.0, 0.0, 5.0, 0.0, 0.0])), f'repeated: {g}'\n"
        "\n"
        "# --- all-same index ---\n"
        "idx = t.tensor([2, 2, 2, 2], dtype=t.long)\n"
        "out = x[idx]\n"
        "grad_out = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "g = getitem_back(grad_out, out, x, idx)\n"
        "# All four contributions land on position 2 → sum is 10.\n"
        "assert t.allclose(g, t.tensor([0.0, 0.0, 10.0, 0.0, 0.0]))\n"
        "\n"
        "# --- larger random test ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.arange(20.0)\n"
        "IDX = t.randint(0, 20, (50,), generator=rng, dtype=t.long)\n"
        "G = t.randn(50, generator=rng)\n"
        "g = getitem_back(G, X[IDX], X, IDX)\n"
        "assert g.shape == X.shape\n"
        "# total gradient must equal total of G (conservation).\n"
        "assert abs(g.sum().item() - G.sum().item()) < 1e-4\n"
        "\n"
        "# --- result is a fresh tensor, not aliased to grad_out ---\n"
        "g_arr = t.tensor([1.0, 2.0])\n"
        "g_out = getitem_back(g_arr, x[t.tensor([0, 1])], x, t.tensor([0, 1], dtype=t.long))\n"
        "assert g_out.data_ptr() != g_arr.data_ptr(), 'must not alias grad_out'\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)\n"
        "idx_ref = t.tensor([0, 2, 0, 3], dtype=t.long)\n"
        "y = x_ref[idx_ref].sum()\n"
        "y.backward()\n"
        "x_det = x_ref.detach()\n"
        "g_ours = getitem_back(t.ones(4), x_det[idx_ref], x_det, idx_ref)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def getitem_back(grad_out: Tensor, out: Tensor, x: Tensor, idx: Tensor) -> Tensor:\n"
        "    grad_in = t.zeros_like(x)\n"
        "    # index_add_ accumulates — repeated entries in idx sum into the same row.\n"
        "    grad_in.index_add_(0, idx, grad_out)\n"
        "    return grad_in"
    ),
    solution_notes=(
        "**Repeated-index = summation.** This is the rule that makes scatter-"
        "add the right primitive. Whenever multiple output positions read the "
        "same input position, their gradient contributions ALL flow back and "
        "ADD. Plain assignment drops all but one.\n\n"
        "**Why `index_add_` not `scatter_add_`.** Both work for 1-D. "
        "`scatter_add_` is more general (supports arbitrary-dim scatter "
        "indices), but for the 1-D-along-axis-0 case `index_add_` reads "
        "more clearly.\n\n"
        "**Conservation.** `g.sum()` must equal `grad_out.sum()` — every unit "
        "of gradient that came in must land somewhere in `x`. Useful sanity "
        "check when debugging."
    ),
)


# =========================================================================
# atom: matmul-back-transpose-pair
# =========================================================================

SPEC_MATMUL = _spec(
    atom_id="matmul-back-transpose-pair",
    subtopic="Backprop: matmul_back transpose pair",
    recap=RECAP_MATMUL,
    ex_idx=1,
    ex_title="matmul_back — grad_out @ y.T and x.T @ grad_out",
    slug="matmul-back-transpose-pair-grad-x-grad-y",
    bloom="Apply",
    difficulty_num=3,
    keywords=["matmul", "transpose", "shape-derivation", "linear"],
    kcs=["matmul-backward-pattern", "arg-position-back-functions"],
    lo=(
        "Apply the matmul backward pair: derive dL/dx = grad_out @ y.T and "
        "dL/dy = x.T @ grad_out from output-shape requirements."
    ),
    prompt_body=(
        "Implement two back fns for `out = x @ y` (2-D matmul; "
        "`x: (m,k)`, `y: (k,n)`, `out: (m,n)`).\n\n"
        "**1. `matmul_back0(grad_out, out, x, y)`** — gradient w.r.t. `x`.\n"
        "   - `dL/dx` must have shape `(m, k)`.\n"
        "   - Only one matmul of `grad_out (m,n)` with a transposed input "
        "produces that shape: `grad_out @ y.T` is `(m,n) @ (n,k) = (m,k)`. ✓\n\n"
        "**2. `matmul_back1(grad_out, out, x, y)`** — gradient w.r.t. `y`.\n"
        "   - `dL/dy` must have shape `(k, n)`.\n"
        "   - The shape-matching matmul: `x.T @ grad_out` is `(k,m) @ (m,n) = "
        "`(k,n)`. ✓\n\n"
        "**The general pattern.** For `A @ B`, the gradient w.r.t. one factor "
        "is a matmul that contracts `grad_out` with the OTHER factor "
        "transposed. The transpose lives on the input you're NOT "
        "differentiating w.r.t.\n\n"
        "Use `t.matmul`, `@`, or `.T`. Both inputs are 2-D `torch.Tensor`. "
        "Return tensors with the correct shapes. No autograd."
    ),
    stub=(
        "def matmul_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dx for out = x @ y. Shape (m, k)."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def matmul_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        '    """dL/dy for out = x @ y. Shape (k, n)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- small (2,3) @ (3,4) = (2,4) ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "x = t.randn(2, 3, generator=rng)\n"
        "y = t.randn(3, 4, generator=rng)\n"
        "out = x @ y\n"
        "grad_out = t.randn(2, 4, generator=rng)\n"
        "g0 = matmul_back0(grad_out, out, x, y)\n"
        "g1 = matmul_back1(grad_out, out, x, y)\n"
        "assert g0.shape == (2, 3), f'g0 shape: {g0.shape}'\n"
        "assert g1.shape == (3, 4), f'g1 shape: {g1.shape}'\n"
        "assert t.allclose(g0, grad_out @ y.T)\n"
        "assert t.allclose(g1, x.T @ grad_out)\n"
        "\n"
        "# --- square: (3,3) @ (3,3) ---\n"
        "x = t.randn(3, 3, generator=rng)\n"
        "y = t.randn(3, 3, generator=rng)\n"
        "out = x @ y\n"
        "grad_out = t.randn(3, 3, generator=rng)\n"
        "g0 = matmul_back0(grad_out, out, x, y)\n"
        "g1 = matmul_back1(grad_out, out, x, y)\n"
        "assert g0.shape == (3, 3) and g1.shape == (3, 3)\n"
        "assert t.allclose(g0, grad_out @ y.T)\n"
        "assert t.allclose(g1, x.T @ grad_out)\n"
        "\n"
        "# --- larger non-square: (5,7) @ (7,3) ---\n"
        "x = t.randn(5, 7, generator=rng)\n"
        "y = t.randn(7, 3, generator=rng)\n"
        "out = x @ y\n"
        "grad_out = t.randn(5, 3, generator=rng)\n"
        "g0 = matmul_back0(grad_out, out, x, y)\n"
        "g1 = matmul_back1(grad_out, out, x, y)\n"
        "assert g0.shape == (5, 7)\n"
        "assert g1.shape == (7, 3)\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.randn(4, 5, requires_grad=True, generator=t.Generator().manual_seed(7))\n"
        "y_ref = t.randn(5, 6, requires_grad=True, generator=t.Generator().manual_seed(8))\n"
        "z = (x_ref @ y_ref).sum()\n"
        "z.backward()\n"
        "x_det, y_det = x_ref.detach(), y_ref.detach()\n"
        "out_cached = x_det @ y_det\n"
        "g0_ours = matmul_back0(t.ones(4, 6), out_cached, x_det, y_det)\n"
        "g1_ours = matmul_back1(t.ones(4, 6), out_cached, x_det, y_det)\n"
        "assert t.allclose(g0_ours, x_ref.grad, atol=1e-5), (\n"
        "    f'g0 disagrees with autograd: max diff '\n"
        "    f'{(g0_ours - x_ref.grad).abs().max()}'\n"
        ")\n"
        "assert t.allclose(g1_ours, y_ref.grad, atol=1e-5), (\n"
        "    f'g1 disagrees with autograd: max diff '\n"
        "    f'{(g1_ours - y_ref.grad).abs().max()}'\n"
        ")\n"
        "\n"
        "# --- back0 and back1 must be DIFFERENT functions ---\n"
        "x_t = t.randn(3, 3, generator=t.Generator().manual_seed(11))\n"
        "y_t = t.randn(3, 3, generator=t.Generator().manual_seed(12))\n"
        "g_t = t.randn(3, 3, generator=t.Generator().manual_seed(13))\n"
        "assert not t.allclose(\n"
        "    matmul_back0(g_t, x_t @ y_t, x_t, y_t),\n"
        "    matmul_back1(g_t, x_t @ y_t, x_t, y_t),\n"
        "), 'matmul back0 and back1 must produce different results'"
    ),
    solution_body=(
        "def matmul_back0(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # Shape (m,n) @ (n,k) = (m,k). y.T transposes y to (n,k).\n"
        "    return grad_out @ y.T\n"
        "\n"
        "\n"
        "def matmul_back1(grad_out: Tensor, out: Tensor, x: Tensor, y: Tensor) -> Tensor:\n"
        "    # Shape (k,m) @ (m,n) = (k,n). x.T transposes x to (k,m).\n"
        "    return x.T @ grad_out"
    ),
    solution_notes=(
        "**Shape-driven derivation.** When the math is unfamiliar, derive the "
        "backward by shape: grad_out is `(m,n)`, target is `(m,k)` (or "
        "`(k,n)`), there's only one matmul that fits. The transpose lives on "
        "the input you're NOT differentiating w.r.t.\n\n"
        "**Why two separate fns.** Matmul is asymmetric: `back0` involves "
        "`y.T`, `back1` involves `x.T`. They're not interchangeable. "
        "Registering at both `(matmul, 0)` and `(matmul, 1)` lets the "
        "dispatcher route grad_out through the correct one.\n\n"
        "**In real autograd,** the same pattern generalizes to batched matmul "
        "and einsum — the transpose-on-the-other-input rule is what powers "
        "every linear-layer backward in every framework."
    ),
)


# ---------------------------------------------------------------- emit all

SPECS = [
    SPEC_NEGATIVE,
    SPEC_EXP,
    SPEC_RESHAPE,
    SPEC_PERMUTE,
    SPEC_SUM,
    SPEC_LAMBDAS,
    SPEC_GETITEM,
    SPEC_MATMUL,
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
