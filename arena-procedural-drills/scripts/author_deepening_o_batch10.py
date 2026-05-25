#!/usr/bin/env python3
"""Author 8 deepening ex2 standalones for prereqs_cnn_extras + prereqs_custom_tensor.

Each ex2 exercises a DISTINCT FACET of the ex1 skill (no overlap with the ex1
LO/Bloom/KCs except the shared atom). PS4 framing: recap = facts + one
exemplar; exercise = stub + NotImplementedError + def _test_exN with >=3
invariants. One LO, one Bloom, <=2 KCs.

Atoms (8):
  prereqs_cnn_extras:
    diagonal-via-strides             ex2 — k-th off-diagonal via storage_offset
    fractional-stride-zero-insertion ex2 — predict ConvT output size from stride (no padding)
    freeze-requires-grad             ex2 — selective unfreeze of last N submodules
    matmul-2d                        ex2 — diagnose & fix shape mismatch with a transpose
    padding-amount-formula-convT     ex2 — inverse-solve p given desired H_out
  prereqs_custom_tensor (autograd preamble injected):
    arange-fancy-index-cross-entropy ex2 — full mean cross-entropy via lse - picked
    grads-dict-accumulate-parents    ex2 — propagate one node's contributions via BACK_FUNCS
    kaiming-uniform-sf-init          ex2 — Conv2d fan_in = IC * kH * kW variant
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


_CUSTOM_TENSOR_PREAMBLE = (
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
    "    \"\"\"Minimal Tensor wrapper for the ARENA-style manual-autograd drills.\n"
    "    Wraps a raw `torch.Tensor` in `.array`. Carries optional `.recipe`,\n"
    "    `.requires_grad`, and `.grad` (the accumulated gradient at leaves).\"\"\"\n"
    "    def __init__(self, array, requires_grad: bool = False, recipe=None):\n"
    "        self.array = array\n"
    "        self.requires_grad = requires_grad\n"
    "        self.recipe = recipe\n"
    "        self.grad = None\n"
    "    def __repr__(self):\n"
    "        return f'MiniTensor({self.array!r}, requires_grad={self.requires_grad})'"
)


# ──────────────────────────────────────────────────────────────────────────
# Recaps — each = ex2 facet facts + ONE exemplar. No meta-prose.
# ──────────────────────────────────────────────────────────────────────────

RECAP_DIAG_OFFSET = (
    "## k-th off-diagonal via `as_strided` + storage_offset — quick refresher\n"
    "\n"
    "The MAIN diagonal of a contiguous `(N, N)` tensor uses "
    "`as_strided(size=(N,), stride=(N+1,))` with implicit offset 0. The "
    "**k-th SUPER-diagonal** (k > 0) reuses the SAME stride `N+1` and adds "
    "the third argument `storage_offset=k`:\n"
    "\n"
    "```\n"
    "m.as_strided(size=(N - k,), stride=(N + 1,), storage_offset=k)\n"
    "```\n"
    "\n"
    "**Why `storage_offset=k`.** The k-th super-diagonal's first element is "
    "`m[0, k]`, which lives at linear offset `k` in row-major storage.\n"
    "\n"
    "**Why length `N - k`.** Each diagonal step advances both row and col by "
    "one — the super-diagonal hits the right wall first; only `N - k` steps "
    "stay inside the matrix.\n"
    "\n"
    "**Exemplar.** For a 4×4 matrix and `k=1`, the super-diagonal is "
    "`[m[0,1], m[1,2], m[2,3]]` (length 3, offset 1, stride 5)."
)

RECAP_FRAC_STRIDE_SHAPE = (
    "## Stride-S ConvT output shape (no padding) — quick refresher\n"
    "\n"
    "With `padding=0` and `output_padding=0`, a `nn.ConvTranspose2d` "
    "with kernel `K` and stride `S` maps a 1-D input of length `H_in` to:\n"
    "\n"
    "```\n"
    "H_out = (H_in - 1) * S + K\n"
    "```\n"
    "\n"
    "This is the **adjoint** of forward `Conv2d(stride=S, K)` (which goes "
    "`H_in -> (H_in - K) // S + 1`). The fractional-stride view: dilate the "
    "input by inserting `S - 1` zeros between every pair of input pixels — "
    "intermediate length `(H_in - 1) * S + 1` — then apply a stride-1 "
    "K-tap conv, adding `K - 1`.\n"
    "\n"
    "**Exemplar.** `H_in = 4, K = 3, S = 2 -> (4-1)*2 + 3 = 9`. Plug into "
    "`nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)(t.randn(1,1,4,4))` "
    "and the spatial axis is 9."
)

RECAP_PARTIAL_UNFREEZE = (
    "## Selective unfreeze of the last N submodules — quick refresher\n"
    "\n"
    "Pure 'freeze backbone, train head' is the simplest transfer-learning "
    "pattern. A finer-grained alternative: freeze EVERYTHING then "
    "**re-enable** training on only the last `n_last` submodules of the "
    "encoder. This trains a bit more of the network than head-only, but "
    "much less than full fine-tune.\n"
    "\n"
    "```\n"
    "for p in model.parameters():\n"
    "    p.requires_grad = False                              # freeze all\n"
    "children = list(model.encoder.children())                # ordered list\n"
    "for layer in children[-n_last:]:                         # last n_last submodules\n"
    "    for p in layer.parameters():\n"
    "        p.requires_grad = True                           # unfreeze\n"
    "```\n"
    "\n"
    "**Exemplar.** For `encoder = Sequential(L0, L1, L2, L3)` and "
    "`n_last = 2`, layers `L2` and `L3` regain `requires_grad=True`; "
    "`L0` and `L1` stay frozen."
)

RECAP_MATMUL_FIX = (
    "## Diagnose-and-fix matmul mismatch via transpose — quick refresher\n"
    "\n"
    "Two 2-D tensors are matmul-compatible iff their inner dims match: "
    "`(M, K) @ (K, N)`. If you receive `a: (M, K)` and `b: (N, K)`, the "
    "inner dims are `K` vs `N` — wrong unless you fix `b` first. Two "
    "rescue moves:\n"
    "\n"
    "```\n"
    "a @ b.T          # (M, K) @ (K, N) -> (M, N)\n"
    "a.T @ b          # only valid if M == N — different output\n"
    "```\n"
    "\n"
    "**Diagnosis rule.** Read both shapes; find which axis pair already "
    "agrees in size; transpose the operand that needs to swap so the "
    "matching axis lands on the inner side.\n"
    "\n"
    "**Exemplar.** `a` is `(3, 5)`, `b` is `(7, 5)`. The `5`s already "
    "agree, but they live on the WRONG sides (`a`'s last vs `b`'s last). "
    "Transpose `b` to `(5, 7)`, then `a @ b.T` gives `(3, 7)`."
)

RECAP_CONVT_INVERSE_PAD = (
    "## Inverse-solving `padding` from a desired ConvT output size — quick refresher\n"
    "\n"
    "ConvTranspose2d output shape (no `output_padding`, dilation 1):\n"
    "\n"
    "```\n"
    "H_out = (H_in - 1) * S - 2 * P + K\n"
    "```\n"
    "\n"
    "Inverting algebraically for `P` given a desired `H_out`:\n"
    "\n"
    "```\n"
    "2 * P = (H_in - 1) * S + K - H_out\n"
    "P     = ((H_in - 1) * S + K - H_out) // 2\n"
    "```\n"
    "\n"
    "**Validity.** The numerator must be `>= 0` AND even — otherwise no "
    "non-negative integer `P` produces exactly `H_out`. Return `None` "
    "when the target isn't achievable.\n"
    "\n"
    "**Exemplar.** `H_in = 4, K = 3, S = 1`. Default `P=0` gives "
    "`H_out = 6`. To force `H_out = 4`, solve "
    "`P = ((4-1)*1 + 3 - 4) // 2 = 1`. Plug back: "
    "`(4-1)*1 - 2*1 + 3 = 4`. Correct."
)

RECAP_CE_LOSS_FROM_LSE = (
    "## Full mean cross-entropy via `logsumexp - picked` — quick refresher\n"
    "\n"
    "Per-sample CE on logits `(B, C)` with integer targets `(B,)`:\n"
    "\n"
    "```\n"
    "lse_i   = log sum_c exp(logits[i, c])            # logsumexp over class axis\n"
    "picked_i = logits[i, target[i]]                  # arange fancy-index\n"
    "ce_i    = lse_i - picked_i                       # = -log softmax(logits)[i, target[i]]\n"
    "loss    = mean_i(ce_i)                           # scalar\n"
    "```\n"
    "\n"
    "Why `lse - picked` IS cross-entropy. `softmax(logits)[i, c] = "
    "exp(logits[i, c]) / sum_c' exp(logits[i, c'])`. Negate the log, "
    "expand, and the denominator becomes `lse_i`, the numerator becomes "
    "`picked_i`.\n"
    "\n"
    "**Exemplar.** `logits = [[1.0, 1.0, 1.0]]`, `target = [0]`. "
    "`lse = log(3 * e^1) = 1 + log 3 ~= 2.0986`, "
    "`picked = 1.0`, `ce ~= 1.0986 == log 3` (uniform-over-3 CE).\n"
    "\n"
    "Matches `F.cross_entropy(logits, target, reduction='mean')` to fp tol."
)

RECAP_NODE_PROPAGATE = (
    "## Propagate one node via BACK_FUNCS + accumulate — quick refresher\n"
    "\n"
    "One reverse-pass step: take `node` (a MiniTensor with `.recipe`), look "
    "up the existing `grads[node]` (the upstream gradient), and for each "
    "`(argnum, parent)` in `node.recipe.parents` look up the matching back "
    "fn in a `BACK_FUNCS: dict[(func, argnum), fn]` table and accumulate "
    "the contribution into `grads[parent]`:\n"
    "\n"
    "```\n"
    "out_grad = grads[node]\n"
    "for argnum, parent in node.recipe.parents.items():\n"
    "    back_fn      = BACK_FUNCS[(node.recipe.func, argnum)]\n"
    "    contribution = back_fn(out_grad, node.array, *node.recipe.args)\n"
    "    grads[parent] = grads.get(parent, 0) + contribution     # get-default-0 + add\n"
    "```\n"
    "\n"
    "Two pieces compose: the `accumulate_into_grads` rule from ex1 "
    "(`get(parent, 0) + g`) AND a dispatcher that picks the right per-arg "
    "back fn.\n"
    "\n"
    "**Exemplar.** For `y = a + b`, `BACK_FUNCS[(add, 0)] = identity`, "
    "`BACK_FUNCS[(add, 1)] = identity`. Reverse pass on `y` with "
    "`grads[y] = g_y` writes `grads[a] = g_y` and `grads[b] = g_y`. "
    "If `a is b` (i.e. `y = a + a`), both contributions land on `a` and "
    "the accumulate-rule sums them to `2 * g_y`."
)

RECAP_KAIMING_CONV = (
    "## Kaiming uniform SF init for Conv2d weights — quick refresher\n"
    "\n"
    "For Conv2d, `fan_in` is NOT `in_channels` alone — it's the size of "
    "the **receptive patch** that produces one output unit:\n"
    "\n"
    "```\n"
    "fan_in = in_channels * kernel_h * kernel_w\n"
    "sf     = 1 / sqrt(fan_in)\n"
    "weight ~ Uniform(-sf, +sf)              shape (OC, IC, kH, kW)\n"
    "```\n"
    "\n"
    "Same `Uniform(-sf, +sf)` recipe as `nn.Linear`; only the `fan_in` "
    "formula changes. PyTorch's `nn.Conv2d.reset_parameters` uses exactly "
    "this.\n"
    "\n"
    "**Exemplar.** `IC=3, kH=3, kW=3 -> fan_in = 27, sf = 1/sqrt(27) ~= "
    "0.192`. Weights of shape `(OC, 3, 3, 3)` are sampled on "
    "`(-0.192, 0.192)`. Empirical `std ~= sf / sqrt(3) ~= 0.111`."
)


# ──────────────────────────────────────────────────────────────────────────
# Specs.
# ──────────────────────────────────────────────────────────────────────────

SPECS = [
    # ═══════════════════════════════════════════════════════════════════════
    # diagonal-via-strides ex2 — k-th off-diagonal via storage_offset
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "diagonal-via-strides",
        "subtopic": "Numpy: Diagonal via strides",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_DIAG_OFFSET,
        "exercise_index": 2,
        "exercise_title": "k-th super-diagonal via as_strided + storage_offset",
        "slug": "k-th-super-diagonal-via-storage-offset",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["as_strided", "storage_offset", "off-diagonal", "super-diagonal"],
        "kcs": ["diagonal-stride-formula", "as-strided-storage-offset"],
        "lo": (
            "Apply `as_strided(size=(N-k,), stride=(N+1,), storage_offset=k)` "
            "to extract the k-th super-diagonal of a contiguous `(N, N)` "
            "tensor as a no-copy view, and verify against `torch.diagonal(m, "
            "offset=k)`."
        ),
        "prompt_body": (
            "Implement `ex2_kth_super_diagonal(m, k)`. Given a 2-D "
            "contiguous tensor `m` of shape `(N, N)` and integer `k` with "
            "`0 <= k < N`, return a 1-D length-`(N - k)` view that aliases "
            "the k-th super-diagonal of `m`: "
            "`[m[0, k], m[1, k+1], ..., m[N-1-k, N-1]]`.\n\n"
            "**Use `as_strided` with `storage_offset`.** The stride along "
            "the diagonal is still `N + 1` (one row down + one col right). "
            "What changes for `k > 0` is the START — the first element "
            "lives at linear offset `k` in storage:\n\n"
            "```\n"
            "m.as_strided(size=(N - k,), stride=(N + 1,), storage_offset=k)\n"
            "```\n\n"
            "**The view must alias `m`.** Writing through the returned "
            "tensor mutates the k-th super-diagonal of `m`. The test "
            "verifies with `.data_ptr()` (now offset from `m.data_ptr()` "
            "by `k * element_size`) AND with an in-place write check.\n\n"
            "**`k = 0`** must reduce to the main-diagonal case (length `N`, "
            "offset `0`).\n\n"
            "**Boundary.** Assume `m.is_contiguous()`, `m.dim() == 2`, "
            "`m.shape[0] == m.shape[1]`, and `0 <= k < N`."
        ),
        "stub": (
            "def ex2_kth_super_diagonal(m: Tensor, k: int) -> Tensor:\n"
            '    """Return the k-th super-diagonal as a strided no-copy view."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-checkable 4x4 case, k = 1.\n"
            "m = t.arange(16.0).reshape(4, 4).contiguous()\n"
            "# m[0,1]=1, m[1,2]=6, m[2,3]=11 → length 3.\n"
            "d1 = ex2_kth_super_diagonal(m, k=1)\n"
            "assert d1.shape == (3,), f'k=1 length must be N-k=3, got {tuple(d1.shape)}'\n"
            "assert t.allclose(d1, t.tensor([1.0, 6.0, 11.0])), f'k=1 values wrong: {d1.tolist()}'\n"
            "\n"
            "# k=2: length N-k=2, values m[0,2]=2, m[1,3]=7.\n"
            "d2 = ex2_kth_super_diagonal(m, k=2)\n"
            "assert d2.shape == (2,)\n"
            "assert t.allclose(d2, t.tensor([2.0, 7.0])), f'k=2 values wrong: {d2.tolist()}'\n"
            "\n"
            "# k=3: length 1, single element m[0, 3] = 3.\n"
            "d3 = ex2_kth_super_diagonal(m, k=3)\n"
            "assert d3.shape == (1,)\n"
            "assert d3.item() == 3.0\n"
            "\n"
            "# k=0 must equal the main diagonal of length N.\n"
            "d0 = ex2_kth_super_diagonal(m, k=0)\n"
            "assert d0.shape == (4,)\n"
            "assert t.allclose(d0, t.tensor([0.0, 5.0, 10.0, 15.0])), f'k=0 must be main diag: {d0}'\n"
            "\n"
            "# View — must alias m (data_ptr offset = k * element_size).\n"
            "elem = m.element_size()\n"
            "for k in [0, 1, 2, 3]:\n"
            "    v = ex2_kth_super_diagonal(m, k=k)\n"
            "    expected_ptr = m.data_ptr() + k * elem\n"
            "    assert v.data_ptr() == expected_ptr, (\n"
            "        f'k={k}: data_ptr must be m.data_ptr() + {k}*element_size; got offset '\n"
            "        f'{v.data_ptr() - m.data_ptr()} expected {k * elem}'\n"
            "    )\n"
            "\n"
            "# Write-through aliasing: mutate via view, verify m updated.\n"
            "v1 = ex2_kth_super_diagonal(m, k=1)\n"
            "v1[1] = -77.0     # was m[1, 2] = 6\n"
            "assert m[1, 2].item() == -77.0, 'view must alias — write to v1[1] should update m[1, 2]'\n"
            "m[1, 2] = 6.0     # restore\n"
            "\n"
            "# Cross-check against torch.diagonal(m, offset=k) on multiple sizes.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "for N in [2, 3, 5, 8, 16]:\n"
            "    mk = t.randn(N, N, generator=rng).contiguous()\n"
            "    for k in range(N):\n"
            "        ours = ex2_kth_super_diagonal(mk, k=k)\n"
            "        ref  = t.diagonal(mk, offset=k)\n"
            "        assert ours.shape == (N - k,) == ref.shape, (\n"
            "            f'N={N},k={k}: ours shape {tuple(ours.shape)} vs ref {tuple(ref.shape)}'\n"
            "        )\n"
            "        assert t.allclose(ours, ref, atol=1e-6), (\n"
            "            f'N={N},k={k}: ours {ours.tolist()} != t.diagonal {ref.tolist()}'\n"
            "        )\n"
            "\n"
            "# Identity matrix: only the main diagonal is all-ones; every k>0 is all zeros.\n"
            "I = t.eye(6)\n"
            "assert t.allclose(ex2_kth_super_diagonal(I, k=0), t.ones(6))\n"
            "for k in [1, 2, 3]:\n"
            "    v = ex2_kth_super_diagonal(I, k=k)\n"
            "    assert v.shape == (6 - k,)\n"
            "    assert t.allclose(v, t.zeros(6 - k)), f'I, k={k}: must be all zero'"
        ),
        "solution_body": (
            "def ex2_kth_super_diagonal(m: Tensor, k: int) -> Tensor:\n"
            "    N = m.shape[0]\n"
            "    return m.as_strided(size=(N - k,), stride=(N + 1,), storage_offset=k)"
        ),
        "solution_notes": (
            "**`storage_offset` is the third positional / kwarg arg of "
            "`as_strided`.** It's the byte offset (in elements, not bytes) "
            "from the start of the underlying storage to the first element "
            "of the view. The default is `m.storage_offset()` (i.e. start "
            "wherever `m` itself starts). Bumping it by `k` skips the "
            "first `k` row-major elements — exactly the offset of `m[0, k]`.\n\n"
            "**Why the stride is still `N + 1`.** A diagonal-step always "
            "moves down-and-right by `(1 row, 1 col) = N + 1` elements in "
            "row-major storage. The starting point shifts; the inter-"
            "element step does not.\n\n"
            "**Sub-diagonals (k < 0).** Outside this drill, but for "
            "completeness: the k-th sub-diagonal starts at `m[|k|, 0]`, "
            "which lives at offset `|k| * N`. Same stride, length "
            "`N - |k|`. `m.as_strided(size=(N - abs(k),), stride=(N + 1,), "
            "storage_offset=abs(k) * N)` handles `k < 0`.\n\n"
            "**Composability.** Now you can build the full "
            "`offdiag_view(m, k)` for any signed `k` in 3 lines — and the "
            "sum-of-all-diagonals is a `for k in range(-(N-1), N): sum(...)` "
            "loop. No copies anywhere."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # fractional-stride-zero-insertion ex2 — predict ConvT output size
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "fractional-stride-zero-insertion",
        "subtopic": "CNN: ConvT fractional-stride zero insertion",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_FRAC_STRIDE_SHAPE,
        "exercise_index": 2,
        "exercise_title": "predict stride-S ConvT output size from the shape formula",
        "slug": "predict-stride-s-convt-output-size",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["ConvTranspose2d", "fractional-stride", "output-shape", "no-padding"],
        "kcs": ["convT-stride-shape-formula", "convT-stride-zero-dilation"],
        "lo": (
            "Apply the no-padding ConvT shape formula `H_out = (H_in - 1) * "
            "S + K` to predict the spatial output size of "
            "`nn.ConvTranspose2d` for given `(H_in, K, S)` and verify "
            "against the real module."
        ),
        "prompt_body": (
            "Implement `ex2_convT_no_pad_outlen(h_in, k, s)`. Return the "
            "spatial output length of a `nn.ConvTranspose2d` with kernel "
            "size `k`, stride `s`, no padding, no output_padding, applied "
            "to a 1-D input of length `h_in`.\n\n"
            "**Formula.**\n"
            "```\n"
            "h_out = (h_in - 1) * s + k\n"
            "```\n\n"
            "**Why.** The stride-S ConvT first DILATES the input by "
            "inserting `s - 1` zeros between every pair of adjacent input "
            "pixels — giving an intermediate length `(h_in - 1) * s + 1`. "
            "Then it applies a stride-1 K-tap conv with full `(K - 1)` "
            "implicit padding on each side, which adds `K - 1` more. The "
            "two effects compose as `(h_in - 1) * s + 1 + (K - 1) = "
            "(h_in - 1) * s + K`.\n\n"
            "**Adjoint relationship.** Forward `Conv2d(stride=s, K)` "
            "maps `H -> (H - K)//s + 1` (shrinks). Its adjoint ConvT goes "
            "the other way at the same stride.\n\n"
            "The test exercises many `(h_in, k, s)` combinations and "
            "compares your prediction against the actual output shape of "
            "`nn.ConvTranspose2d` (stride-1, 2, 3 cases)."
        ),
        "stub": (
            "def ex2_convT_no_pad_outlen(h_in: int, k: int, s: int) -> int:\n"
            '    """Output length of nn.ConvTranspose2d with no padding."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "# Direct value checks.\n"
            "assert ex2_convT_no_pad_outlen(h_in=4, k=3, s=1) == 6,  '(4-1)*1 + 3 = 6'\n"
            "assert ex2_convT_no_pad_outlen(h_in=4, k=3, s=2) == 9,  '(4-1)*2 + 3 = 9'\n"
            "assert ex2_convT_no_pad_outlen(h_in=5, k=3, s=2) == 11, '(5-1)*2 + 3 = 11'\n"
            "assert ex2_convT_no_pad_outlen(h_in=8, k=4, s=2) == 18, '(8-1)*2 + 4 = 18'\n"
            "assert ex2_convT_no_pad_outlen(h_in=1, k=5, s=1) == 5,  'single pixel + 5-tap → 5'\n"
            "assert ex2_convT_no_pad_outlen(h_in=1, k=3, s=4) == 3,  'single-pixel input: stride has no effect'\n"
            "assert ex2_convT_no_pad_outlen(h_in=3, k=2, s=3) == 8,  '(3-1)*3 + 2 = 8'\n"
            "\n"
            "# Monotonicity sanity — output strictly increases with each input dim.\n"
            "assert ex2_convT_no_pad_outlen(5, 3, 2) > ex2_convT_no_pad_outlen(4, 3, 2), 'larger h_in → larger output'\n"
            "assert ex2_convT_no_pad_outlen(5, 5, 2) > ex2_convT_no_pad_outlen(5, 3, 2), 'larger k → larger output'\n"
            "assert ex2_convT_no_pad_outlen(5, 3, 3) > ex2_convT_no_pad_outlen(5, 3, 2), 'larger s → larger output'\n"
            "\n"
            "# Stride-S adjoint round-trip sanity: forward Conv2d with stride S and K, no pad:\n"
            "#   H_in → (H_in - K) // S + 1\n"
            "# Plug back into ConvT with the SAME S, K, no pad: lands at >= H_in for any divisible H_in.\n"
            "# We just check that no-padding ConvT yields the canonical 'dilate + stride-1 conv' shape.\n"
            "cases = [(4, 3, 1), (4, 3, 2), (5, 3, 2), (8, 4, 2), (16, 3, 1), (10, 5, 2), (7, 3, 2), (1, 3, 4), (3, 2, 3)]\n"
            "for h_in, k, s in cases:\n"
            "    ct = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=k, stride=s, padding=0)\n"
            "    x = t.randn(1, 1, h_in, h_in)\n"
            "    actual = ct(x).shape[-1]\n"
            "    predicted = ex2_convT_no_pad_outlen(h_in, k, s)\n"
            "    assert predicted == actual, (\n"
            "        f'(h={h_in},k={k},s={s}): predicted {predicted}, actual {actual}'\n"
            "    )"
        ),
        "solution_body": (
            "def ex2_convT_no_pad_outlen(h_in: int, k: int, s: int) -> int:\n"
            "    return (h_in - 1) * s + k"
        ),
        "solution_notes": (
            "**No-padding form is the cleanest entry into the full formula.** "
            "The full PyTorch shape rule (with `padding=P`, "
            "`output_padding=OP`, `dilation=D`) is "
            "`h_out = (h_in - 1) * s - 2 * p + d * (k - 1) + op + 1`. "
            "With `D = 1`, `P = 0`, `OP = 0`, this collapses to "
            "`(h_in - 1) * s + k`. Master this base case, then layer on "
            "padding (`- 2 * p`) and output_padding (`+ op`) as separate "
            "corrections.\n\n"
            "**Why ConvT(K=4, S=2, P=1, OP=0) does NOT clean-2× upsample.** "
            "Plug in: `(h - 1) * 2 - 2 + 4 = 2 * h`. That's clean 2× — but "
            "ONLY when `P=1`. Drop the padding and you get "
            "`(h - 1) * 2 + 4 = 2h + 2` — two extra pixels. The padding "
            "argument is what makes the upsample clean; this drill is the "
            "no-padding baseline you correct from.\n\n"
            "**Composes with `padding-amount-formula-convT` ex1.** That "
            "drill adds the `- 2 * p` correction. The combined formula "
            "covers every shape question a stride-S decoder block can ask."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # freeze-requires-grad ex2 — selective unfreeze last N submodules
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "freeze-requires-grad",
        "subtopic": "PyTorch: freeze via requires_grad=False",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_PARTIAL_UNFREEZE,
        "exercise_index": 2,
        "exercise_title": "selectively unfreeze the last N submodules of an encoder",
        "slug": "selectively-unfreeze-last-n-submodules",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["transfer-learning", "partial-fine-tune", "requires_grad", "encoder-children"],
        "kcs": ["freeze-requires-grad-false", "partial-unfreeze-last-n"],
        "lo": (
            "Apply the partial fine-tune pattern: freeze every param, then "
            "re-enable `requires_grad=True` on the parameters of the last "
            "`n_last` children of an encoder Sequential."
        ),
        "prompt_body": (
            "A toy 'pretrained' model is provided:\n\n"
            "```\n"
            "class ToyBackbone(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.Sequential(\n"
            "            nn.Linear(10, 32),     # children[0]\n"
            "            nn.ReLU(),             # children[1] — no params\n"
            "            nn.Linear(32, 16),     # children[2]\n"
            "            nn.ReLU(),             # children[3] — no params\n"
            "            nn.Linear(16, 16),     # children[4]\n"
            "        )\n"
            "        self.fc = nn.Linear(16, 5)\n"
            "    def forward(self, x):\n"
            "        return self.fc(self.encoder(x))\n"
            "```\n\n"
            "Implement `ex2_partial_unfreeze(model, n_last)` that performs "
            "**partial fine-tuning** in this order:\n\n"
            "1. **Freeze everything**: set `p.requires_grad = False` on every "
            "param of `model`.\n"
            "2. **Re-enable the last `n_last` encoder children**: take "
            "`list(model.encoder.children())[-n_last:]` and for each submodule "
            "set `p.requires_grad = True` on every param inside.\n"
            "3. Leave the head `model.fc` FROZEN — that's the difference from "
            "the ex1 'replace head' pattern.\n"
            "4. Return the list of trainable params "
            "(`[p for p in model.parameters() if p.requires_grad]`).\n\n"
            "**Indexing detail.** `list(model.encoder.children())[-n_last:]` "
            "grabs the LAST `n_last` submodules regardless of which are "
            "param-bearing. ReLU children have no params, so they "
            "contribute nothing to the trainable list — but they still "
            "count toward the slice. This matches the standard 'last 2 "
            "blocks' partial-tune pattern.\n\n"
            "**Boundary.** Assume `1 <= n_last <= len(list(model.encoder.children()))`."
        ),
        "stub": (
            "import torch.nn as nn\n"
            "\n"
            "class ToyBackbone(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.Sequential(\n"
            "            nn.Linear(10, 32),\n"
            "            nn.ReLU(),\n"
            "            nn.Linear(32, 16),\n"
            "            nn.ReLU(),\n"
            "            nn.Linear(16, 16),\n"
            "        )\n"
            "        self.fc = nn.Linear(16, 5)\n"
            "    def forward(self, x):\n"
            "        return self.fc(self.encoder(x))\n"
            "\n"
            "\n"
            "def ex2_partial_unfreeze(model, n_last: int):\n"
            '    """Freeze all, then re-enable requires_grad on the last n_last encoder children; head stays frozen."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "from torch.nn import functional as F\n"
            "\n"
            "# n_last=1 — only the LAST encoder Linear (children[4]) is trainable.\n"
            "m = ToyBackbone()\n"
            "trainable = ex2_partial_unfreeze(m, n_last=1)\n"
            "# encoder children: 0 Linear, 1 ReLU, 2 Linear, 3 ReLU, 4 Linear.\n"
            "enc_children = list(m.encoder.children())\n"
            "# children[0] and children[2] (the first two Linears) must stay frozen.\n"
            "for p in enc_children[0].parameters():\n"
            "    assert p.requires_grad is False, 'encoder[0] must be frozen at n_last=1'\n"
            "for p in enc_children[2].parameters():\n"
            "    assert p.requires_grad is False, 'encoder[2] must be frozen at n_last=1'\n"
            "# children[4] must be trainable.\n"
            "for p in enc_children[4].parameters():\n"
            "    assert p.requires_grad is True, 'encoder[4] must be trainable at n_last=1'\n"
            "# Head must stay frozen — unlike ex1.\n"
            "for p in m.fc.parameters():\n"
            "    assert p.requires_grad is False, 'fc head must remain FROZEN (different from ex1)'\n"
            "# trainable list = exactly fc-free encoder[4] (weight + bias).\n"
            "assert len(trainable) == 2, f'n_last=1 → 2 trainable tensors (Linear weight+bias), got {len(trainable)}'\n"
            "\n"
            "# n_last=3 — slice [-3:] is (children[2] Linear, children[3] ReLU, children[4] Linear).\n"
            "# Both Linears trainable; ReLU contributes 0 params; fc still frozen.\n"
            "m3 = ToyBackbone()\n"
            "trainable3 = ex2_partial_unfreeze(m3, n_last=3)\n"
            "enc3 = list(m3.encoder.children())\n"
            "for p in enc3[0].parameters():\n"
            "    assert p.requires_grad is False, 'encoder[0] still frozen at n_last=3'\n"
            "for p in enc3[2].parameters():\n"
            "    assert p.requires_grad is True, 'encoder[2] trainable at n_last=3'\n"
            "for p in enc3[4].parameters():\n"
            "    assert p.requires_grad is True, 'encoder[4] trainable at n_last=3'\n"
            "for p in m3.fc.parameters():\n"
            "    assert p.requires_grad is False, 'fc still frozen at n_last=3'\n"
            "assert len(trainable3) == 4, f'n_last=3 → 4 trainable tensors (two Linears, w+b each), got {len(trainable3)}'\n"
            "\n"
            "# n_last=5 — every encoder Linear trainable; fc still frozen.\n"
            "m5 = ToyBackbone()\n"
            "trainable5 = ex2_partial_unfreeze(m5, n_last=5)\n"
            "enc5 = list(m5.encoder.children())\n"
            "for i in [0, 2, 4]:\n"
            "    for p in enc5[i].parameters():\n"
            "        assert p.requires_grad is True, f'encoder[{i}] trainable at n_last=5'\n"
            "for p in m5.fc.parameters():\n"
            "    assert p.requires_grad is False, 'fc still frozen at n_last=5'\n"
            "assert len(trainable5) == 6, f'n_last=5 → 6 trainable tensors, got {len(trainable5)}'\n"
            "\n"
            "# Optimizer + backward step: only encoder[4] updates at n_last=1.\n"
            "m_b = ToyBackbone()\n"
            "trainable_b = ex2_partial_unfreeze(m_b, n_last=1)\n"
            "enc_b = list(m_b.encoder.children())\n"
            "frozen_w = enc_b[0].weight.detach().clone()\n"
            "frozen_w_2 = enc_b[2].weight.detach().clone()\n"
            "head_w = m_b.fc.weight.detach().clone()\n"
            "trainable_w = enc_b[4].weight.detach().clone()\n"
            "opt = t.optim.Adam(trainable_b, lr=1e-2)\n"
            "x = t.randn(4, 10)\n"
            "y = m_b(x)\n"
            "# Target shape (4,) for 5-way CE.\n"
            "loss = F.cross_entropy(y, t.tensor([0, 1, 2, 3]))\n"
            "loss.backward()\n"
            "opt.step()\n"
            "assert t.equal(enc_b[0].weight, frozen_w), 'encoder[0] mutated despite being frozen'\n"
            "assert t.equal(enc_b[2].weight, frozen_w_2), 'encoder[2] mutated despite being frozen'\n"
            "assert t.equal(m_b.fc.weight, head_w), 'fc head mutated despite being frozen'\n"
            "assert not t.equal(enc_b[4].weight, trainable_w), 'encoder[4] should have been updated'"
        ),
        "solution_body": (
            "def ex2_partial_unfreeze(model, n_last: int):\n"
            "    for p in model.parameters():\n"
            "        p.requires_grad = False\n"
            "    children = list(model.encoder.children())\n"
            "    for layer in children[-n_last:]:\n"
            "        for p in layer.parameters():\n"
            "            p.requires_grad = True\n"
            "    return [p for p in model.parameters() if p.requires_grad]"
        ),
        "solution_notes": (
            "**Why `list(model.encoder.children())[-n_last:]` and not "
            "`model.encoder[-n_last:]`.** Both work for `nn.Sequential`, "
            "but `.children()` is the generic interface across any "
            "`nn.Module` container — it'd work for a custom encoder built "
            "from named submodules too. The slice is the portable form.\n\n"
            "**Why ReLU in the slice is harmless.** "
            "`nn.ReLU.parameters()` yields nothing, so `for p in "
            "relu.parameters(): p.requires_grad = True` is a no-op. The "
            "slice can include any mix of param-bearing and parameter-less "
            "modules without affecting correctness.\n\n"
            "**Contrast with ex1.** Ex1 also replaces `model.fc` with a "
            "fresh head — new modules default to `requires_grad=True`, so "
            "the head trains. Here we LEAVE the original head in place "
            "and DON'T unfreeze it. Use case: continue using the original "
            "head (e.g. dimensionality is fine) but fine-tune deeper into "
            "the network. Common when transferring within-domain (same "
            "label space, different distribution).\n\n"
            "**Generalization to ResNet.** Replace `model.encoder` with "
            "`model.layer4` (the last residual stage) and you have the "
            "standard 'fine-tune the last block' recipe used in countless "
            "downstream-task papers."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # matmul-2d ex2 — diagnose & fix via transpose
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "matmul-2d",
        "subtopic": "Numpy: matmul 2-D",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_MATMUL_FIX,
        "exercise_index": 2,
        "exercise_title": "diagnose and fix a shape mismatch with a single transpose",
        "slug": "diagnose-and-fix-matmul-mismatch-with-transpose",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["matmul", "transpose", "shape-debug", "inner-dim-fix"],
        "kcs": ["matmul-2d-shape-rule", "transpose-to-align-inner-dim"],
        "lo": (
            "Analyze a pair of 2-D shapes that fails the matmul inner-dim "
            "rule, identify which operand needs transposing, and return "
            "the corrected product."
        ),
        "prompt_body": (
            "Implement `ex2_fix_matmul(a, b)`. Given two 2-D tensors `a` "
            "and `b` (each `(rows, cols)`), apply the following rules and "
            "return the result:\n\n"
            "1. If `a.shape[1] == b.shape[0]`, no fix needed — return `a @ b`.\n"
            "2. Else if `a.shape[1] == b.shape[1]` (i.e. `b`'s LAST dim "
            "matches `a`'s last dim), transpose `b` and return `a @ b.T`.\n"
            "3. Else if `a.shape[0] == b.shape[0]` (i.e. `a`'s FIRST dim "
            "matches `b`'s first dim), transpose `a` and return `a.T @ b`.\n"
            "4. Otherwise no single transpose fixes the mismatch — return "
            "`None`.\n\n"
            "**Rule order matters.** Check the no-fix case first, then the "
            "two single-transpose cases in the stated order. If multiple "
            "branches would fire (e.g. `a` and `b` are both square and "
            "symmetric on size), the earlier rule wins.\n\n"
            "**Examples.**\n"
            "- `a: (3, 5), b: (5, 7)` → already compatible. Return "
            "`(3, 7)` product.\n"
            "- `a: (3, 5), b: (7, 5)` → inner mismatch but `a.shape[1] == "
            "b.shape[1] == 5`. Return `a @ b.T` of shape `(3, 7)`.\n"
            "- `a: (3, 5), b: (3, 7)` → inner mismatch but `a.shape[0] == "
            "b.shape[0] == 3`. Return `a.T @ b` of shape `(5, 7)`.\n"
            "- `a: (3, 5), b: (8, 11)` → no shared axis. Return `None`.\n\n"
            "**The test uses concrete random tensors** and verifies both "
            "the chosen transpose AND the numerical equivalence to a "
            "manually-applied `t.matmul` on the correctly-shaped operands."
        ),
        "stub": (
            "def ex2_fix_matmul(a: Tensor, b: Tensor):\n"
            '    """Apply one transpose (if needed) to make a @ b shape-valid; return None if no single transpose fixes it."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# --- already compatible: no transpose needed ---\n"
            "a1 = t.randn(3, 5, generator=rng)\n"
            "b1 = t.randn(5, 7, generator=rng)\n"
            "y1 = ex2_fix_matmul(a1, b1)\n"
            "assert y1.shape == (3, 7), f'no-fix path shape: {tuple(y1.shape)}'\n"
            "assert t.allclose(y1, a1 @ b1, atol=1e-5)\n"
            "\n"
            "# --- b needs transpose (last dim of b matches last dim of a) ---\n"
            "a2 = t.randn(3, 5, generator=rng)\n"
            "b2 = t.randn(7, 5, generator=rng)\n"
            "y2 = ex2_fix_matmul(a2, b2)\n"
            "assert y2.shape == (3, 7), f'b.T path shape: {tuple(y2.shape)}'\n"
            "assert t.allclose(y2, a2 @ b2.T, atol=1e-5), 'must equal a @ b.T'\n"
            "\n"
            "# --- a needs transpose (first dim of a matches first dim of b) ---\n"
            "a3 = t.randn(3, 5, generator=rng)\n"
            "b3 = t.randn(3, 7, generator=rng)\n"
            "y3 = ex2_fix_matmul(a3, b3)\n"
            "assert y3.shape == (5, 7), f'a.T path shape: {tuple(y3.shape)}'\n"
            "assert t.allclose(y3, a3.T @ b3, atol=1e-5), 'must equal a.T @ b'\n"
            "\n"
            "# --- no single transpose works → None ---\n"
            "a4 = t.randn(3, 5, generator=rng)\n"
            "b4 = t.randn(8, 11, generator=rng)\n"
            "assert ex2_fix_matmul(a4, b4) is None, 'no compatible single-transpose pair → None'\n"
            "\n"
            "a5 = t.randn(2, 4, generator=rng)\n"
            "b5 = t.randn(7, 9, generator=rng)\n"
            "assert ex2_fix_matmul(a5, b5) is None\n"
            "\n"
            "# --- rule-order check: when both transposed cases would work, prefer b.T ---\n"
            "# a:(K, K), b:(K, K) with K matching everywhere — already-compatible branch fires.\n"
            "ak = t.randn(4, 4, generator=rng)\n"
            "bk = t.randn(4, 4, generator=rng)\n"
            "yk = ex2_fix_matmul(ak, bk)\n"
            "assert yk.shape == (4, 4)\n"
            "assert t.allclose(yk, ak @ bk, atol=1e-5), 'already-compatible branch must fire first'\n"
            "\n"
            "# --- order check when ONLY transpose branches would fire ---\n"
            "# Construct shapes where inner dims don't match but BOTH transposes are valid:\n"
            "# a:(3, 4), b:(3, 4). a.shape[0]=3=b.shape[0] (a.T branch) AND a.shape[1]=4=b.shape[1] (b.T branch).\n"
            "# The b.T branch is stated FIRST in the rules → it must win.\n"
            "a6 = t.randn(3, 4, generator=rng)\n"
            "b6 = t.randn(3, 4, generator=rng)\n"
            "y6 = ex2_fix_matmul(a6, b6)\n"
            "assert y6.shape == (3, 3), f'b.T branch must win, shape (3,3), got {tuple(y6.shape)}'\n"
            "assert t.allclose(y6, a6 @ b6.T, atol=1e-5)\n"
            "\n"
            "# --- larger random sweep cross-checks against manual logic ---\n"
            "for _ in range(10):\n"
            "    # pick shapes where exactly ONE branch fires\n"
            "    M, K, N = int(t.randint(2, 8, (1,)).item()), int(t.randint(2, 8, (1,)).item()), int(t.randint(2, 8, (1,)).item())\n"
            "    while M == N or K == M or K == N:\n"
            "        M, K, N = int(t.randint(2, 8, (1,)).item()), int(t.randint(2, 8, (1,)).item()), int(t.randint(2, 8, (1,)).item())\n"
            "    A = t.randn(M, K, generator=rng)\n"
            "    # b shape: pick (N, K) → forces b.T branch.\n"
            "    B_NK = t.randn(N, K, generator=rng)\n"
            "    out = ex2_fix_matmul(A, B_NK)\n"
            "    assert out is not None, f'(M,K,N)=({M},{K},{N}): b.T branch should fire'\n"
            "    assert out.shape == (M, N)\n"
            "    assert t.allclose(out, A @ B_NK.T, atol=1e-4)"
        ),
        "solution_body": (
            "def ex2_fix_matmul(a: Tensor, b: Tensor):\n"
            "    # Rule 1: already compatible.\n"
            "    if a.shape[1] == b.shape[0]:\n"
            "        return a @ b\n"
            "    # Rule 2: a's last dim matches b's last dim → transpose b.\n"
            "    if a.shape[1] == b.shape[1]:\n"
            "        return a @ b.T\n"
            "    # Rule 3: a's first dim matches b's first dim → transpose a.\n"
            "    if a.shape[0] == b.shape[0]:\n"
            "        return a.T @ b\n"
            "    # Rule 4: no single transpose fixes it.\n"
            "    return None"
        ),
        "solution_notes": (
            "**Why each rule exists.** Matmul needs `a.shape[1] == "
            "b.shape[0]`. There are exactly two ways a single `.T` can "
            "create that match: transpose `b` to swap its dims (covers "
            "the case `a.shape[1] == b.shape[1]`); transpose `a` (covers "
            "`a.shape[0] == b.shape[0]`). If neither shared axis exists, "
            "no single transpose works — you'd need a reshape, broadcast, "
            "or different operands.\n\n"
            "**Why rule-order matters.** When multiple rules would fire, "
            "the result depends on which you check first — and the choice "
            "is semantically meaningful. `a @ b.T` and `a.T @ b` produce "
            "DIFFERENT tensors (different shapes, different values). "
            "Documenting the priority is the only way the function "
            "behaves predictably.\n\n"
            "**This is the real debugging move.** When you get "
            "`RuntimeError: mat1 and mat2 shapes cannot be multiplied`, "
            "this is the diagnostic loop: print both shapes, find the "
            "shared axis, transpose the wrong operand. The drill is the "
            "shape-debug-as-function-call version of that workflow.\n\n"
            "**Why we don't always succeed.** `(3, 5)` and `(8, 11)` "
            "share nothing — neither transpose helps. The caller would "
            "have to look elsewhere (broadcasting, einsum with explicit "
            "contractions, or a genuine bug in their pipeline)."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # padding-amount-formula-convT ex2 — inverse-solve p
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "padding-amount-formula-convT",
        "subtopic": "CNN: ConvT padding amount formula",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_CONVT_INVERSE_PAD,
        "exercise_index": 2,
        "exercise_title": "inverse-solve `padding` from a target ConvT output size",
        "slug": "inverse-solve-convt-padding-from-target-output",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["ConvTranspose2d", "padding", "inverse", "decoder-design"],
        "kcs": ["convT-padding-subtracts-from-output", "convT-padding-inverse-solve"],
        "lo": (
            "Apply the inverted ConvT shape formula "
            "`P = ((H_in - 1) * S + K - H_out) // 2` (with validity check) "
            "to recover the `padding` arg needed to hit a target output "
            "size, and verify against `nn.ConvTranspose2d`."
        ),
        "prompt_body": (
            "Implement `ex2_convT_padding_for(h_in, k, s, h_out_target)`. "
            "Given `H_in, K, S`, and a desired output length "
            "`h_out_target`, return the non-negative integer `P` that "
            "makes `nn.ConvTranspose2d` produce exactly `h_out_target`. "
            "Return `None` if no non-negative integer `P` achieves the "
            "target.\n\n"
            "**Algebra.** Start from "
            "`H_out = (H_in - 1) * S - 2 * P + K`. Solve for `P`:\n"
            "```\n"
            "2 * P = (H_in - 1) * S + K - H_out\n"
            "P     = ((H_in - 1) * S + K - H_out) // 2\n"
            "```\n\n"
            "**Validity checks (return `None` if any fails).**\n"
            "1. `(H_in - 1) * S + K - H_out` must be `>= 0` — a negative "
            "value would mean negative padding (not a thing).\n"
            "2. The same expression must be EVEN — `H_out` is reached by "
            "subtracting `2 * P` (an even number) from "
            "`(H_in - 1) * S + K`, so the gap must be even.\n\n"
            "**Sanity check.** Plug the returned `P` back into the forward "
            "formula and confirm it hits `h_out_target`. The test does "
            "this against the actual `nn.ConvTranspose2d` module too.\n\n"
            "**Use case.** Decoder design — you've decided the spatial "
            "size at each level (e.g. 8 → 16 → 32 → 64) and need to "
            "back-out the padding arg for each transposed conv. This is "
            "the inverted form of ex1."
        ),
        "stub": (
            "def ex2_convT_padding_for(h_in: int, k: int, s: int, h_out_target: int):\n"
            '    """Return the integer padding P such that nn.ConvTranspose2d hits h_out_target, or None."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "# Direct algebra checks.\n"
            "# h_in=4, k=3, s=1, target=4 → (4-1)*1 + 3 - 4 = 2, /2 = 1 → P=1.\n"
            "assert ex2_convT_padding_for(4, 3, 1, 4) == 1\n"
            "# h_in=4, k=3, s=1, target=6 → (4-1)*1 + 3 - 6 = 0, /2 = 0 → P=0 (no padding).\n"
            "assert ex2_convT_padding_for(4, 3, 1, 6) == 0\n"
            "# h_in=4, k=3, s=1, target=2 → (4-1)*1 + 3 - 2 = 4, /2 = 2 → P=2.\n"
            "assert ex2_convT_padding_for(4, 3, 1, 2) == 2\n"
            "# h_in=8, k=4, s=2, target=16 → (8-1)*2 + 4 - 16 = 18 - 16 = 2, /2 = 1 → P=1 (canonical 2x).\n"
            "assert ex2_convT_padding_for(8, 4, 2, 16) == 1\n"
            "# h_in=5, k=3, s=2, target=9 → (5-1)*2 + 3 - 9 = 8+3-9 = 2, /2 = 1 → P=1.\n"
            "assert ex2_convT_padding_for(5, 3, 2, 9) == 1\n"
            "\n"
            "# Invalid: target too LARGE (would need negative padding).\n"
            "# h_in=4, k=3, s=1, target=99 → 99 > 6 (the no-pad max). Should return None.\n"
            "assert ex2_convT_padding_for(4, 3, 1, 99) is None, 'target > no-pad max must return None'\n"
            "# h_in=4, k=3, s=1, target=7 → 7 > 6. None.\n"
            "assert ex2_convT_padding_for(4, 3, 1, 7) is None\n"
            "\n"
            "# Invalid: parity violation (gap is odd).\n"
            "# h_in=4, k=3, s=1, target=5 → (4-1)*1 + 3 - 5 = 1 (odd). No integer P. None.\n"
            "assert ex2_convT_padding_for(4, 3, 1, 5) is None, 'odd gap must return None (no integer P)'\n"
            "# h_in=4, k=4, s=1, target=4 → (4-1)*1 + 4 - 4 = 3 (odd). None.\n"
            "assert ex2_convT_padding_for(4, 4, 1, 4) is None\n"
            "\n"
            "# Round-trip: solved P must reproduce h_out_target via the forward formula.\n"
            "for h_in, k, s in [(4, 3, 1), (5, 3, 2), (8, 4, 2), (10, 5, 2), (16, 3, 1)]:\n"
            "    for target in range(1, 20):\n"
            "        P = ex2_convT_padding_for(h_in, k, s, target)\n"
            "        if P is None:\n"
            "            continue\n"
            "        assert P >= 0, f'returned P={P} must be non-negative'\n"
            "        forward = (h_in - 1) * s - 2 * P + k\n"
            "        assert forward == target, (\n"
            "            f'round-trip fail: h_in={h_in},k={k},s={s},target={target} → P={P} '\n"
            "            f'gives h_out={forward}'\n"
            "        )\n"
            "\n"
            "# Cross-check against actual nn.ConvTranspose2d for several cases.\n"
            "for h_in, k, s, target in [(4, 3, 1, 4), (8, 4, 2, 16), (5, 3, 2, 9), (4, 3, 1, 6)]:\n"
            "    P = ex2_convT_padding_for(h_in, k, s, target)\n"
            "    assert P is not None\n"
            "    ct = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=k, stride=s, padding=P)\n"
            "    x = t.randn(1, 1, h_in, h_in)\n"
            "    actual = ct(x).shape[-1]\n"
            "    assert actual == target, (\n"
            "        f'nn.ConvTranspose2d with P={P} produced {actual}, expected {target}'\n"
            "    )\n"
            "\n"
            "# Edge: target=0 only happens if (h_in-1)*s + k is even — most don't match exactly.\n"
            "# h_in=3, k=2, s=1: (3-1)*1+2 = 4, target=0 → P=2. Edge of validity.\n"
            "P_edge = ex2_convT_padding_for(3, 2, 1, 0)\n"
            "assert P_edge == 2, f'edge target=0 case: P should be 2, got {P_edge}'"
        ),
        "solution_body": (
            "def ex2_convT_padding_for(h_in: int, k: int, s: int, h_out_target: int):\n"
            "    gap = (h_in - 1) * s + k - h_out_target\n"
            "    if gap < 0:\n"
            "        return None              # target exceeds no-pad maximum\n"
            "    if gap % 2 != 0:\n"
            "        return None              # parity violation — no integer P\n"
            "    return gap // 2"
        ),
        "solution_notes": (
            "**Why the two validity checks.** "
            "`H_out = (H_in - 1) * S + K - 2 * P`. Solving algebraically "
            "for `P` gives `gap / 2` where `gap = (H_in-1)*S + K - "
            "H_out`. Three obstructions:\n"
            "- `gap < 0` ⇒ `P < 0`, which `nn.ConvTranspose2d` rejects.\n"
            "- `gap` odd ⇒ `P` non-integer; PyTorch only accepts ints.\n"
            "- `gap == 0` is fine (means `P = 0`, no padding).\n\n"
            "When you build decoder networks, parity violations are the "
            "common reason a specific `(H_in, K, S, H_target)` doesn't "
            "fit — the standard fix is to use `output_padding=1` (or to "
            "adjust `K`), not to fight the formula.\n\n"
            "**Why this composes with `output_padding`.** The full "
            "formula adds `+ OP` to `H_out`. So if you have an off-by-one "
            "parity issue (`gap == 2*P + 1`), set `OP = 1` and the "
            "remaining `gap - 1` becomes even — and `P = (gap - 1) // 2` "
            "works. The combined `ex2 + output-padding` recipe lets you "
            "hit any spatial target.\n\n"
            "**Decoder design recipe.** For each upsample block you "
            "decide: source size, target size, kernel, stride. Run "
            "`ex2_convT_padding_for` to get `P`. If it's `None`, either "
            "bump `K` by 1, change `S`, or set `output_padding=1`. This "
            "is exactly what `torchgan` / ARENA's DCGAN decoder helper "
            "does under the hood."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # arange-fancy-index-cross-entropy ex2 — full CE via lse - picked
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "arange-fancy-index-cross-entropy",
        "subtopic": "Loss: arange fancy-index cross-entropy",
        "topic_folder": "prereqs_custom_tensor",
        "atom_recap_md": RECAP_CE_LOSS_FROM_LSE,
        "exercise_index": 2,
        "exercise_title": "full mean cross-entropy loss via logsumexp − picked",
        "slug": "full-mean-cross-entropy-via-lse-minus-picked",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["cross-entropy", "logsumexp", "arange-fancy-index", "mean-reduce"],
        "kcs": ["arange-fancy-index-cross-entropy", "logsumexp-cross-entropy"],
        "lo": (
            "Apply the per-sample CE decomposition `ce_i = lse_i - "
            "logits[i, target[i]]` and average over the batch axis to "
            "produce a scalar loss matching `F.cross_entropy(reduction="
            "'mean')`."
        ),
        "prompt_body": (
            "Implement `ex2_cross_entropy_mean(logits, target)`. Returns a "
            "scalar tensor equal to "
            "`F.cross_entropy(logits, target, reduction='mean')`.\n\n"
            "Inputs:\n"
            "- `logits`: shape `(B, C)`, float.\n"
            "- `target`: shape `(B,)`, integer class indices in `[0, C)`.\n\n"
            "**Recipe (three composable steps — all vectorized).**\n"
            "```\n"
            "lse_per_sample = t.logsumexp(logits, dim=-1)        # (B,)\n"
            "picked         = logits[t.arange(B), target]        # (B,)  — ex1 facet\n"
            "ce_per_sample  = lse_per_sample - picked            # (B,)\n"
            "loss           = ce_per_sample.mean()               # scalar\n"
            "```\n\n"
            "**Why the formula is correct.** "
            "`-log softmax(logits)[i, c] = -log(exp(logits[i, c]) / "
            "sum_c' exp(logits[i, c'])) = logsumexp_c'(logits[i, c']) - "
            "logits[i, c]`. So the standard `NLL(log_softmax(logits))[i] "
            "= lse_i - picked_i`. Mean over `i` is the canonical "
            "`reduction='mean'` form.\n\n"
            "**Forbidden:** calling `F.cross_entropy` or `F.nll_loss` "
            "directly — the drill is the build-from-pieces version. "
            "`t.logsumexp` and fancy-indexing are OK (they're the "
            "primitives we're composing).\n\n"
            "**Output is a 0-D scalar tensor**, not a Python float. "
            "`.shape == ()`."
        ),
        "stub": (
            "def ex2_cross_entropy_mean(logits: Tensor, target: Tensor) -> Tensor:\n"
            '    """Mean cross-entropy = mean(logsumexp(logits, -1) - logits[arange(B), target])."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "import math\n"
            "\n"
            "# --- hand-checkable: uniform logits → CE ≈ log(C) ---\n"
            "logits = t.tensor([[1.0, 1.0, 1.0]])\n"
            "target = t.tensor([0])\n"
            "loss = ex2_cross_entropy_mean(logits, target)\n"
            "assert loss.shape == (), f'must be 0-D scalar, got {tuple(loss.shape)}'\n"
            "assert abs(loss.item() - math.log(3)) < 1e-5, (\n"
            "    f'uniform 3-way CE should equal log(3) ≈ {math.log(3):.4f}, got {loss.item():.4f}'\n"
            ")\n"
            "\n"
            "# --- one-hot perfect prediction (very large correct logit) → near zero loss ---\n"
            "logits_perfect = t.tensor([[100.0, 0.0, 0.0]])\n"
            "loss_perfect = ex2_cross_entropy_mean(logits_perfect, t.tensor([0]))\n"
            "assert loss_perfect.item() < 1e-3, (\n"
            "    f'perfect prediction loss should be ~0, got {loss_perfect.item()}'\n"
            ")\n"
            "\n"
            "# --- one-hot WRONG prediction → loss ≈ huge_logit (large) ---\n"
            "loss_wrong = ex2_cross_entropy_mean(logits_perfect, t.tensor([1]))\n"
            "assert loss_wrong.item() > 50.0, (\n"
            "    f'totally wrong prediction loss should be ~huge_logit, got {loss_wrong.item()}'\n"
            ")\n"
            "\n"
            "# --- cross-check against F.cross_entropy(reduction='mean') on many shapes ---\n"
            "rng = t.Generator().manual_seed(0)\n"
            "for B, C in [(1, 2), (4, 10), (16, 100), (3, 5), (32, 1000)]:\n"
            "    big_logits = t.randn(B, C, generator=rng)\n"
            "    big_target = t.randint(0, C, (B,), generator=rng)\n"
            "    ours = ex2_cross_entropy_mean(big_logits, big_target)\n"
            "    ref  = F.cross_entropy(big_logits, big_target, reduction='mean')\n"
            "    assert ours.shape == ref.shape == ()\n"
            "    assert t.allclose(ours, ref, atol=1e-5), (\n"
            "        f'(B={B},C={C}): ours {ours.item():.5f} vs F.cross_entropy {ref.item():.5f}'\n"
            "    )\n"
            "\n"
            "# --- mean reduction sanity: mean of per-sample equals overall mean ---\n"
            "B, C = 8, 5\n"
            "lg = t.randn(B, C, generator=rng)\n"
            "tg = t.randint(0, C, (B,), generator=rng)\n"
            "ours_full = ex2_cross_entropy_mean(lg, tg)\n"
            "lse = t.logsumexp(lg, dim=-1)\n"
            "picked = lg[t.arange(B), tg]\n"
            "expected = (lse - picked).mean()\n"
            "assert t.allclose(ours_full, expected, atol=1e-6), 'must equal (lse - picked).mean()'\n"
            "\n"
            "# --- dtype preserved ---\n"
            "lg64 = t.randn(4, 6, generator=rng).double()\n"
            "tg64 = t.randint(0, 6, (4,), generator=rng)\n"
            "loss64 = ex2_cross_entropy_mean(lg64, tg64)\n"
            "assert loss64.dtype == t.float64, f'dtype not preserved: {loss64.dtype}'\n"
            "\n"
            "# --- numerical stability: very large logits don't overflow ---\n"
            "# (logsumexp internally subtracts the max — overflow would crash here.)\n"
            "lg_huge = t.tensor([[1000.0, 999.0, 1001.0]])\n"
            "loss_huge = ex2_cross_entropy_mean(lg_huge, t.tensor([2]))\n"
            "assert t.isfinite(loss_huge), 'logsumexp must be stable for huge logits'\n"
            "ref_huge = F.cross_entropy(lg_huge, t.tensor([2]), reduction='mean')\n"
            "assert t.allclose(loss_huge, ref_huge, atol=1e-4), 'stable lse path must match F.cross_entropy'"
        ),
        "solution_body": (
            "def ex2_cross_entropy_mean(logits: Tensor, target: Tensor) -> Tensor:\n"
            "    B = logits.shape[0]\n"
            "    lse    = t.logsumexp(logits, dim=-1)             # (B,)\n"
            "    picked = logits[t.arange(B), target]             # (B,)\n"
            "    return (lse - picked).mean()                     # scalar"
        ),
        "solution_notes": (
            "**The 3-line CE is a teaching artifact.** Real "
            "`F.cross_entropy` fuses these into a single CUDA kernel with "
            "label smoothing, ignore_index handling, weighted-mean "
            "reduction, and gradient bookkeeping. But the math is "
            "EXACTLY these three steps. When you debug a custom loss "
            "(reweighting per-sample, masking, hierarchical softmax), "
            "you'll be working at this decomposed level.\n\n"
            "**Why `t.logsumexp`, not `(logits.exp()).sum(-1).log()`.** "
            "Numerical stability. `t.logsumexp` subtracts the per-row max "
            "before the exp, preventing overflow at large logits. The "
            "test for `1000`-scale logits checks this — a naive "
            "implementation would return `inf` or `nan` and crash the "
            "comparison.\n\n"
            "**`.mean()` vs `.sum() / B`.** Both give the same scalar; "
            "`.mean()` keeps the dtype's natural epsilon. For tiny "
            "batches `(B=1, 2)` the difference is irrelevant, but at "
            "fp16 with large batches the in-kernel mean is more "
            "accurate.\n\n"
            "**Composes with the autograd preamble.** The MiniTensor "
            "scaffolding is loaded so this drill can sit next to other "
            "autograd-internals drills, but THIS exercise uses raw "
            "`torch.Tensor` for clarity — same code translates directly "
            "to `MiniTensor` if you wrap each op with its `_back` fn.\n\n"
            "**Composes with grads-dict-accumulate.** When you "
            "differentiate this loss by hand: the gradient w.r.t. "
            "`logits[i, c]` is `softmax(logits)[i, c] - (1 if c == "
            "target[i] else 0)`, divided by `B`. That's the canonical "
            "'softmax minus one-hot' gradient that every classification "
            "head uses on the backward pass."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # grads-dict-accumulate-parents ex2 — propagate one node via BACK_FUNCS
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "grads-dict-accumulate-parents",
        "subtopic": "Backprop: grads dict accumulate parents",
        "topic_folder": "prereqs_custom_tensor",
        "atom_recap_md": RECAP_NODE_PROPAGATE,
        "exercise_index": 2,
        "exercise_title": "propagate one node's contributions via BACK_FUNCS",
        "slug": "propagate-one-node-via-back-funcs",
        "bloom_level": "Apply",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["BACK_FUNCS", "dispatcher", "grads-dict", "reverse-pass", "node-propagate"],
        "kcs": ["grads-dict-accumulate-parents", "back-funcs-dispatch"],
        "lo": (
            "Apply the per-node reverse-pass step: read `grads[node]`, "
            "iterate `node.recipe.parents.items()`, dispatch the matching "
            "back fn from a `BACK_FUNCS` table, and accumulate each "
            "contribution into `grads[parent]` via `get-default-0 + add`."
        ),
        "prompt_body": (
            "Implement `propagate_node(node, grads, BACK_FUNCS)`. ONE step "
            "of the reverse pass. Mutates `grads` in place.\n\n"
            "Inputs:\n"
            "- `node`: a `MiniTensor` with `.recipe` set (a `Recipe` "
            "object). The recipe's `.func` is the forward op (`add`, "
            "`mul`, etc), `.args` are the forward args, and `.parents` is "
            "a `dict[int, MiniTensor]` mapping argnum to the parent "
            "MiniTensor.\n"
            "- `grads`: `dict[MiniTensor, torch.Tensor]`. Must already "
            "contain `node` (the upstream gradient for this node, "
            "produced by the parent of `node` in the reverse walk).\n"
            "- `BACK_FUNCS`: `dict[(func, argnum), Callable]`. Each entry "
            "is `back_fn(out_grad, node.array, *node.recipe.args) -> "
            "contribution_to_that_arg`.\n\n"
            "**Algorithm.**\n"
            "```\n"
            "out_grad = grads[node]\n"
            "for argnum, parent in node.recipe.parents.items():\n"
            "    back_fn      = BACK_FUNCS[(node.recipe.func, argnum)]\n"
            "    contribution = back_fn(out_grad, node.array, *node.recipe.args)\n"
            "    grads[parent] = grads.get(parent, 0) + contribution\n"
            "```\n\n"
            "**Two-step composition.** First step looks up the right "
            "per-arg back fn from the table (the dispatcher). Second step "
            "is the **same** accumulate rule as ex1: "
            "`grads.get(parent, 0) + contribution`. Don't overwrite, "
            "don't `+=`, don't `KeyError` on first-touch.\n\n"
            "**The `y = a + a` stress case.** When the same parent "
            "appears at two argnums, the loop visits each separately and "
            "BOTH contributions must land on the parent via the "
            "accumulator. Argnum 0 and 1 are different keys in "
            "`BACK_FUNCS`, but here they happen to be the same fn "
            "(identity for add).\n\n"
            "Returns `None`."
        ),
        "stub": (
            "def propagate_node(node, grads: dict, BACK_FUNCS: dict) -> None:\n"
            '    """One reverse-pass step: look up per-arg back fns, accumulate into parent grads."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a tiny BACK_FUNCS table over Python-level ops (add, mul, sub).\n"
            "# Each back fn takes (out_grad, out_value, *forward_args) and returns d(out)/d(arg).\n"
            "def _add_back0(grad_out, value, a, b): return grad_out                  # d(a+b)/da = 1\n"
            "def _add_back1(grad_out, value, a, b): return grad_out                  # d(a+b)/db = 1\n"
            "def _mul_back0(grad_out, value, a, b): return grad_out * b              # d(a*b)/da = b\n"
            "def _mul_back1(grad_out, value, a, b): return grad_out * a              # d(a*b)/db = a\n"
            "def _sub_back0(grad_out, value, a, b): return grad_out                  # d(a-b)/da = 1\n"
            "def _sub_back1(grad_out, value, a, b): return -grad_out                 # d(a-b)/db = -1\n"
            "\n"
            "def fake_add(a, b): return a + b\n"
            "def fake_mul(a, b): return a * b\n"
            "def fake_sub(a, b): return a - b\n"
            "\n"
            "BACK_FUNCS = {\n"
            "    (fake_add, 0): _add_back0, (fake_add, 1): _add_back1,\n"
            "    (fake_mul, 0): _mul_back0, (fake_mul, 1): _mul_back1,\n"
            "    (fake_sub, 0): _sub_back0, (fake_sub, 1): _sub_back1,\n"
            "}\n"
            "\n"
            "# --- Case 1: y = a + b. propagate_node on y should write grads[a] = grads[b] = grad_out. ---\n"
            "a = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
            "b = MiniTensor(t.tensor([4.0, 5.0, 6.0]), requires_grad=True)\n"
            "y_arr = a.array + b.array\n"
            "y = MiniTensor(y_arr, requires_grad=True,\n"
            "               recipe=Recipe(func=fake_add, args=(a.array, b.array), parents={0: a, 1: b}))\n"
            "grads = {y: t.tensor([1.0, 1.0, 1.0])}\n"
            "ret = propagate_node(y, grads, BACK_FUNCS)\n"
            "assert ret is None, 'must mutate grads in place and return None'\n"
            "assert a in grads and b in grads\n"
            "assert t.allclose(grads[a], t.tensor([1.0, 1.0, 1.0])), f'grads[a] = {grads[a]}'\n"
            "assert t.allclose(grads[b], t.tensor([1.0, 1.0, 1.0])), f'grads[b] = {grads[b]}'\n"
            "\n"
            "# --- Case 2: y = a * b. grads[a] = grad_out * b, grads[b] = grad_out * a. ---\n"
            "a2 = MiniTensor(t.tensor([2.0, 3.0]), requires_grad=True)\n"
            "b2 = MiniTensor(t.tensor([10.0, 20.0]), requires_grad=True)\n"
            "y2_arr = a2.array * b2.array\n"
            "y2 = MiniTensor(y2_arr, requires_grad=True,\n"
            "                recipe=Recipe(func=fake_mul, args=(a2.array, b2.array), parents={0: a2, 1: b2}))\n"
            "grads = {y2: t.tensor([1.0, 1.0])}\n"
            "propagate_node(y2, grads, BACK_FUNCS)\n"
            "assert t.allclose(grads[a2], t.tensor([10.0, 20.0])), f'd/da: {grads[a2]} (expected b)'\n"
            "assert t.allclose(grads[b2], t.tensor([2.0, 3.0])),  f'd/db: {grads[b2]} (expected a)'\n"
            "\n"
            "# --- Case 3: y = a - b. grads[a] = grad_out, grads[b] = -grad_out. ---\n"
            "a3 = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
            "b3 = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
            "y3 = MiniTensor(t.tensor([0.0]), requires_grad=True,\n"
            "                recipe=Recipe(func=fake_sub, args=(a3.array, b3.array), parents={0: a3, 1: b3}))\n"
            "grads = {y3: t.tensor([5.0])}\n"
            "propagate_node(y3, grads, BACK_FUNCS)\n"
            "assert t.allclose(grads[a3], t.tensor([5.0]))\n"
            "assert t.allclose(grads[b3], t.tensor([-5.0])), 'sub flips sign on arg1'\n"
            "\n"
            "# --- THE CRITICAL TEST: y = a + a. Same parent at argnum 0 and 1; contributions sum. ---\n"
            "a4 = MiniTensor(t.tensor([7.0]), requires_grad=True)\n"
            "y4_arr = a4.array + a4.array\n"
            "y4 = MiniTensor(y4_arr, requires_grad=True,\n"
            "                recipe=Recipe(func=fake_add, args=(a4.array, a4.array), parents={0: a4, 1: a4}))\n"
            "grads = {y4: t.tensor([1.0])}\n"
            "propagate_node(y4, grads, BACK_FUNCS)\n"
            "assert t.allclose(grads[a4], t.tensor([2.0])), (\n"
            "    f'y = a + a reverse-pass case: grads[a] must be 2 * grad_out = 2.0, got {grads[a4]} — '\n"
            "    'did you overwrite instead of accumulating?'\n"
            ")\n"
            "\n"
            "# --- Pre-existing parent grad: must be ADDED, not overwritten. ---\n"
            "a5 = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
            "b5 = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
            "y5 = MiniTensor(t.tensor([3.0]), requires_grad=True,\n"
            "                recipe=Recipe(func=fake_add, args=(a5.array, b5.array), parents={0: a5, 1: b5}))\n"
            "grads = {y5: t.tensor([1.0]), a5: t.tensor([100.0])}    # a5 already has a prior contribution\n"
            "propagate_node(y5, grads, BACK_FUNCS)\n"
            "assert t.allclose(grads[a5], t.tensor([101.0])), (\n"
            "    f'pre-existing grads[a5] must be added to, got {grads[a5]} (expected 101.0)'\n"
            ")\n"
            "assert t.allclose(grads[b5], t.tensor([1.0]))\n"
            "\n"
            "# --- Out-grad not mutated by the propagation. ---\n"
            "a6 = MiniTensor(t.tensor([0.0]), requires_grad=True)\n"
            "b6 = MiniTensor(t.tensor([0.0]), requires_grad=True)\n"
            "y6 = MiniTensor(t.tensor([0.0]), requires_grad=True,\n"
            "                recipe=Recipe(func=fake_add, args=(a6.array, b6.array), parents={0: a6, 1: b6}))\n"
            "out_grad_original = t.tensor([3.0])\n"
            "grads = {y6: out_grad_original}\n"
            "propagate_node(y6, grads, BACK_FUNCS)\n"
            "assert t.allclose(out_grad_original, t.tensor([3.0])), (\n"
            "    f'grads[y6] (out_grad) must not be mutated, got {out_grad_original}'\n"
            ")"
        ),
        "solution_body": (
            "def propagate_node(node, grads: dict, BACK_FUNCS: dict) -> None:\n"
            "    out_grad = grads[node]\n"
            "    for argnum, parent in node.recipe.parents.items():\n"
            "        back_fn = BACK_FUNCS[(node.recipe.func, argnum)]\n"
            "        contribution = back_fn(out_grad, node.array, *node.recipe.args)\n"
            "        grads[parent] = grads.get(parent, 0) + contribution"
        ),
        "solution_notes": (
            "**Why the table key is `(func, argnum)`.** Different forward "
            "ops have different gradient rules per argument. `add` has "
            "the same back fn for arg 0 and arg 1 (both identity). `mul` "
            "has DIFFERENT back fns: `arg 0 -> grad_out * b`, "
            "`arg 1 -> grad_out * a`. `sub` has `arg 0 -> grad_out` and "
            "`arg 1 -> -grad_out`. The `(func, argnum)` key uniquely "
            "identifies which back fn to dispatch.\n\n"
            "**Why we accumulate, not overwrite.** Two reasons:\n"
            "- A parent at TWO argnums of the SAME node (like `y = a + "
            "a`): the loop visits each separately and both contributions "
            "land on `parent` — must sum.\n"
            "- A parent visited from MULTIPLE downstream nodes (the "
            "general DAG case): each prior `propagate_node` call may have "
            "already deposited a contribution. The new one adds.\n\n"
            "**Why `back_fn(out_grad, node.array, *node.recipe.args)`.** "
            "The back fn signature is `(grad_out, value, *forward_args)`. "
            "Some back fns need the forward output value (e.g. "
            "softmax-back needs the softmax output); some need the "
            "forward inputs (e.g. mul-back needs the other operand). "
            "Passing all three covers every case the ARENA "
            "manual-autograd layer ships.\n\n"
            "**Why we look up via `node.recipe.func`, not `node.func`.** "
            "MiniTensors don't store the forward function directly — the "
            "recipe does. `node.recipe` is the link between the forward "
            "tape and the backward dispatch table.\n\n"
            "**Compose with topological sort.** The full reverse pass "
            "is: topo-sort the DAG, walk it in reverse, call "
            "`propagate_node` on each non-leaf node, then copy "
            "`grads[leaf]` into `leaf.grad` for each leaf. This drill is "
            "the per-node step — the load-bearing inner block."
        ),
        "extra_imports": [_CUSTOM_TENSOR_PREAMBLE],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # kaiming-uniform-sf-init ex2 — Conv2d fan_in variant
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "kaiming-uniform-sf-init",
        "subtopic": "Init: Kaiming uniform SF init",
        "topic_folder": "prereqs_custom_tensor",
        "atom_recap_md": RECAP_KAIMING_CONV,
        "exercise_index": 2,
        "exercise_title": "Kaiming uniform SF init for Conv2d weights (fan_in = IC * kH * kW)",
        "slug": "kaiming-uniform-sf-init-for-conv2d",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["kaiming", "uniform", "init", "conv2d", "receptive-fan-in"],
        "kcs": ["kaiming-uniform-sf-init", "conv-fan-in-formula"],
        "lo": (
            "Apply the Conv2d-specific Kaiming-uniform recipe — "
            "`fan_in = in_channels * kernel_h * kernel_w`, "
            "`sf = 1/sqrt(fan_in)`, `weight ~ Uniform(-sf, +sf)` of "
            "shape `(out_channels, in_channels, kernel_h, kernel_w)`."
        ),
        "prompt_body": (
            "Implement `kaiming_uniform_sf_conv2d(out_channels, "
            "in_channels, kernel_h, kernel_w, generator)`. The Conv2d "
            "weight initializer ARENA uses (and PyTorch's `nn.Conv2d` "
            "default):\n\n"
            "1. `fan_in = in_channels * kernel_h * kernel_w` — NOT just "
            "`in_channels`. This is the size of the receptive PATCH that "
            "produces one output unit, not just the channel dimension.\n"
            "2. `sf = 1 / sqrt(fan_in)`.\n"
            "3. Sample `(out_channels, in_channels, kernel_h, kernel_w)` "
            "floats uniformly on `(-sf, +sf)` using `generator`.\n"
            "4. Return as a `torch.Tensor` (4-D).\n\n"
            "**Contrast with ex1.** Ex1's Linear init uses `fan_in = "
            "in_features`. For Conv2d the `fan_in` formula changes — the "
            "spatial extent of the kernel matters because every spatial "
            "position participates in the summation that produces one "
            "output activation. Same `Uniform(-sf, +sf)` recipe, "
            "different `fan_in`.\n\n"
            "**Why this differs from `nn.Linear` viewed as 1x1 Conv.** A "
            "`Conv2d(IC, OC, kernel_size=1)` has `fan_in = IC * 1 * 1 = "
            "IC` — which matches `Linear(IC, OC)`. The two coincide at "
            "kernel size 1. They diverge for bigger kernels: a 3x3 conv "
            "has `fan_in = 9 * IC` — `sf` shrinks by `sqrt(9) = 3`.\n\n"
            "Hint: `t.rand(shape, generator=g)` is uniform on `[0, 1)`. "
            "To get `(-sf, +sf)`, do `(t.rand(shape, generator=g) * 2 - 1) "
            "* sf`.\n\n"
            "Output: `torch.Tensor` of shape `(out_channels, in_channels, "
            "kernel_h, kernel_w)`."
        ),
        "stub": (
            "def kaiming_uniform_sf_conv2d(\n"
            "    out_channels: int, in_channels: int, kernel_h: int, kernel_w: int,\n"
            "    generator: t.Generator,\n"
            ") -> Tensor:\n"
            '    """Sample Conv2d weight ~ Uniform(-1/sqrt(IC*kH*kW), +1/sqrt(IC*kH*kW))."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "# --- shape ---\n"
            "g = t.Generator().manual_seed(0)\n"
            "w = kaiming_uniform_sf_conv2d(4, 3, 3, 3, g)\n"
            "assert isinstance(w, t.Tensor), f'expected torch.Tensor, got {type(w).__name__}'\n"
            "assert w.shape == (4, 3, 3, 3), f'shape: {w.shape}'\n"
            "\n"
            "# --- bounds: |w| <= sf for every element ---\n"
            "fan_in = 3 * 3 * 3\n"
            "sf = 1.0 / math.sqrt(fan_in)\n"
            "assert w.abs().max().item() <= sf + 1e-6, (\n"
            "    f'all entries in (-sf, +sf); max |w| = {w.abs().max().item()} vs sf = {sf:.4f}'\n"
            ")\n"
            "\n"
            "# --- large-sample empirical std ≈ sf / sqrt(3) ---\n"
            "g2 = t.Generator().manual_seed(1)\n"
            "OC, IC, kH, kW = 200, 16, 5, 5\n"
            "w_big = kaiming_uniform_sf_conv2d(OC, IC, kH, kW, g2)   # 200*16*25 = 80k samples\n"
            "fan_in_big = IC * kH * kW\n"
            "sf_big = 1.0 / math.sqrt(fan_in_big)\n"
            "assert w_big.abs().max().item() <= sf_big + 1e-6\n"
            "expected_std = sf_big / math.sqrt(3.0)\n"
            "empirical_std = w_big.std().item()\n"
            "rel_err = abs(empirical_std - expected_std) / expected_std\n"
            "assert rel_err < 0.05, (\n"
            "    f'empirical std {empirical_std:.5f} too far from expected '\n"
            "    f'{expected_std:.5f} (rel err {rel_err:.4f}); did you use fan_in = IC alone?'\n"
            ")\n"
            "\n"
            "# --- the key distinction from ex1: fan_in MUST include kH * kW ---\n"
            "# If init mistakenly used fan_in = IC, std would be sqrt(kH*kW) times too large.\n"
            "wrong_sf = 1.0 / math.sqrt(IC)\n"
            "wrong_std = wrong_sf / math.sqrt(3.0)\n"
            "assert empirical_std < wrong_std * 0.5, (\n"
            "    f'empirical std {empirical_std} is close to the IC-only formula {wrong_std:.5f} — '\n"
            "    'fan_in must include kernel_h * kernel_w'\n"
            ")\n"
            "\n"
            "# --- generator honored: same seed → same tensor ---\n"
            "g_a = t.Generator().manual_seed(42)\n"
            "g_b = t.Generator().manual_seed(42)\n"
            "w_a = kaiming_uniform_sf_conv2d(2, 3, 3, 3, g_a)\n"
            "w_b = kaiming_uniform_sf_conv2d(2, 3, 3, 3, g_b)\n"
            "assert t.allclose(w_a, w_b), 'same seed must produce the same weight tensor'\n"
            "\n"
            "# --- different seed → different tensor ---\n"
            "g_c = t.Generator().manual_seed(99)\n"
            "w_c = kaiming_uniform_sf_conv2d(2, 3, 3, 3, g_c)\n"
            "assert not t.allclose(w_a, w_c), 'different seed should produce different tensor'\n"
            "\n"
            "# --- kernel-size scaling: a 3x3 conv has sf ~ 3x smaller than a 1x1 conv at same IC ---\n"
            "g_1 = t.Generator().manual_seed(7)\n"
            "g_3 = t.Generator().manual_seed(7)\n"
            "w_1x1 = kaiming_uniform_sf_conv2d(100, 64, 1, 1, g_1)  # fan_in = 64\n"
            "w_3x3 = kaiming_uniform_sf_conv2d(100, 64, 3, 3, g_3)  # fan_in = 576\n"
            "ratio = w_3x3.abs().max().item() / w_1x1.abs().max().item()\n"
            "# sf_3x3 / sf_1x1 = sqrt(64/576) = sqrt(1/9) = 1/3.\n"
            "assert 0.25 < ratio < 0.45, (\n"
            "    f'sf must scale as 1/sqrt(IC*kH*kW); 3x3 vs 1x1 max-abs ratio {ratio:.3f}, expected ~1/3'\n"
            ")\n"
            "\n"
            "# --- agreement with the Linear formula at kernel = 1x1 ---\n"
            "# For (kH, kW) = (1, 1), fan_in = IC, matching Linear(IC, OC). The sf should match.\n"
            "g_lin = t.Generator().manual_seed(123)\n"
            "g_1x1 = t.Generator().manual_seed(123)\n"
            "# Linear-equivalent: sf = 1/sqrt(IC)\n"
            "raw_lin = t.rand(64, 100, generator=g_lin)            # we mimic Linear init manually\n"
            "sf_lin  = 1.0 / math.sqrt(64)\n"
            "w_lin   = (raw_lin * 2 - 1) * sf_lin                  # shape (64, 100)\n"
            "w_conv1x1 = kaiming_uniform_sf_conv2d(100, 64, 1, 1, g_1x1)  # shape (100, 64, 1, 1)\n"
            "# Compare the underlying scalar distributions (max-abs and std).\n"
            "assert abs(w_lin.abs().max().item() - w_conv1x1.abs().max().item()) < 0.05, (\n"
            "    'at kernel 1x1, Conv2d and Linear inits should have the same scale'\n"
            ")"
        ),
        "solution_body": (
            "def kaiming_uniform_sf_conv2d(\n"
            "    out_channels: int, in_channels: int, kernel_h: int, kernel_w: int,\n"
            "    generator: t.Generator,\n"
            ") -> Tensor:\n"
            "    fan_in = in_channels * kernel_h * kernel_w\n"
            "    sf = fan_in ** -0.5\n"
            "    raw = t.rand(out_channels, in_channels, kernel_h, kernel_w, generator=generator)\n"
            "    return (raw * 2 - 1) * sf"
        ),
        "solution_notes": (
            "**Why `fan_in = IC * kH * kW`, not `IC`.** For a forward "
            "Conv2d, each output activation is the sum of `IC * kH * kW` "
            "weighted inputs. The Kaiming derivation argues that the "
            "input-output variance is preserved when "
            "`Var(w) * fan_in = 1` (for ReLU it's 2; for the SF form, "
            "it's whatever the constant works out to under `Uniform(-sf, "
            "sf)`). The relevant `fan_in` is the count of inputs "
            "AGGREGATED per output unit. For Conv2d that's `IC * kH * "
            "kW` — every spatial position of the kernel patch.\n\n"
            "**At kernel 1x1, Conv2d == Linear.** "
            "`fan_in = IC * 1 * 1 = IC`. Both inits sample on the same "
            "`(-1/sqrt(IC), +1/sqrt(IC))` interval. This is why 1x1 "
            "convs are sometimes called 'pointwise linear layers' — they "
            "really are linear in the channel dimension, with no spatial "
            "context.\n\n"
            "**Why bigger kernels → smaller weights.** A 3x3 conv at the "
            "same `IC` has `9x` more inputs feeding each output, so each "
            "individual weight should be `3x` smaller on average to "
            "preserve the pre-activation scale. The test asserts the "
            "1x1 vs 3x3 ratio is `~1/3 = sqrt(1/9)`.\n\n"
            "**Why this matters at training time.** Wrong `fan_in` "
            "(using `IC` alone for a `3x3` conv) gives weights `3x` too "
            "large — pre-activations explode by `~3x` per layer, "
            "saturating ReLUs and killing gradients in the first few "
            "backward passes. The empirical-std test catches this."
        ),
        "extra_imports": [_CUSTOM_TENSOR_PREAMBLE],
    },
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

        # If spec has the custom_tensor preamble in extra_imports, exec it first.
        for extra in spec.get("extra_imports", []) or []:
            try:
                exec(extra, ns)
            except Exception:
                pass

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
    print(f"[deepening_o_batch10] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_o_batch10] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")


if __name__ == "__main__":
    main()
