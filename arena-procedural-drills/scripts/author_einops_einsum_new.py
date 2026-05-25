#!/usr/bin/env python3
"""Author 4 new standalone procedural drills for the `einops-einsum` atom.

ex6..ex9 — Colab-only material that flashcards can't deliver:
  ex6: full scaled dot-product attention with mask + softmax + viz of weights
  ex7: multi-head einsum — split (b, s, h*d) -> (b, h, s, s) QK^T scores
  ex8: batched bilinear form y = x^T A x for a batch of x's, visualized as a heatmap over a 2D grid
  ex9: 3-tensor Tucker-style contraction with intermediate-shape debug prints
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone  # noqa: E402

ATOM_ID = "einops-einsum"
SUBTOPIC = "Einops: Deep Learning"
TOPIC = "prereqs_einops"

RECAP = (
    "## einops.einsum — quick refresher\n"
    "\n"
    "`einsum(*tensors, pattern)` performs sum-contraction over named indices:\n"
    "1. **Elementwise** — `'i j, i j -> i j'` multiplies pointwise (no reduction).\n"
    "2. **Matmul** — `'i k, k j -> i j'` contracts the shared `k` (sum-reduce).\n"
    "3. **Batched** — `'b i k, b k j -> b i j'` carries `b` through, contracts `k`.\n"
    "4. **Three operands** — `'i j, j k, k l -> i l'` chains two contractions; the optimizer picks pairing order.\n"
    "\n"
    "**The two rules:**\n"
    "- An index that appears on input AND output → preserved (broadcast-like).\n"
    "- An index that appears on input but NOT on output → sum-contracted."
)


SPECS = [
    # ─────────────────────────────────────────────────────────────────────────
    # ex6 — full scaled dot-product attention with mask + softmax + viz
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "scaled dot-product attention end-to-end with mask + softmax + weight heatmap",
        "slug": "scaled-dot-product-attention-with-mask-viz",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["attention", "softmax", "mask", "visualization", "pipeline"],
        "kcs": ["einsum-matmul-contraction", "einsum-attention-scores", "einsum-weighted-aggregation"],
        "lo": "Compose two einsum contractions (QK^T then weights@V) with masking and softmax to produce a full attention head output, and visualize the resulting attention weights.",
        "prompt_body": (
            "Implement `ex6_scaled_dot_product_attention(q, k, v, mask)`.\n"
            "\n"
            "Inputs:\n"
            "- `q`: `(B, T, D)` queries.\n"
            "- `k`: `(B, T, D)` keys.\n"
            "- `v`: `(B, T, D_v)` values.\n"
            "- `mask`: `(T, T)` bool tensor — `True` means \"blocked\" (set score to `-inf` before softmax).\n"
            "\n"
            "Steps:\n"
            "1. **Scores** via `einsum`: `scores[b, i, j] = sum_d q[b,i,d] * k[b,j,d] / sqrt(D)`.\n"
            "2. Apply `mask` (broadcast across batch) — fill blocked positions with `-inf`.\n"
            "3. **Weights** via `softmax` along the key axis.\n"
            "4. **Output** via a second `einsum`: `out[b, i, e] = sum_j weights[b,i,j] * v[b,j,e]`.\n"
            "\n"
            "Return `(out, weights)` where `weights` has shape `(B, T, T)` and `out` has shape `(B, T, D_v)`. Use `einops.einsum` for **both** matmul-like steps — do not use `@`, `torch.bmm`, or `matmul`.\n"
            "\n"
            "The test cell visualizes `weights[0]` (the first batch element's attention matrix) as a heatmap so you can see what the model is attending to under the mask."
        ),
        "stub": (
            "def ex6_scaled_dot_product_attention(\n"
            "    q: Tensor, k: Tensor, v: Tensor, mask: Tensor\n"
            ") -> tuple[Tensor, Tensor]:\n"
            "    \"\"\"Full scaled dot-product attention via two einsum contractions.\n"
            "    Returns (out, weights). `out`: (B, T, D_v). `weights`: (B, T, T).\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn.functional as F\n"
            "\n"
            "B, T, D, D_v = 2, 6, 8, 5\n"
            "q = t.randn(B, T, D)\n"
            "k = t.randn(B, T, D)\n"
            "v = t.randn(B, T, D_v)\n"
            "# Causal mask: True above the diagonal = blocked.\n"
            "mask = t.triu(t.ones(T, T, dtype=t.bool), diagonal=1)\n"
            "\n"
            "out, weights = ex6_scaled_dot_product_attention(q, k, v, mask)\n"
            "\n"
            "assert out.shape == (B, T, D_v), f'out: expected ({B},{T},{D_v}), got {out.shape}'\n"
            "assert weights.shape == (B, T, T), f'weights: expected ({B},{T},{T}), got {weights.shape}'\n"
            "\n"
            "# Ground truth via PyTorch primitives.\n"
            "scale = D ** 0.5\n"
            "expected_scores = (q @ k.transpose(-2, -1)) / scale\n"
            "expected_scores = expected_scores.masked_fill(mask, float('-inf'))\n"
            "expected_weights = F.softmax(expected_scores, dim=-1)\n"
            "expected_out = expected_weights @ v\n"
            "\n"
            "assert t.allclose(weights, expected_weights, atol=1e-5), 'weights differ from reference'\n"
            "assert t.allclose(out, expected_out, atol=1e-5), 'out differs from reference'\n"
            "\n"
            "# Causal sanity: weights must be zero above the diagonal.\n"
            "for i in range(T):\n"
            "    for j in range(i + 1, T):\n"
            "        assert weights[:, i, j].abs().max().item() < 1e-7, f'leak at ({i},{j})'\n"
            "# Each row of weights must sum to 1.\n"
            "row_sums = weights.sum(dim=-1)\n"
            "assert t.allclose(row_sums, t.ones_like(row_sums), atol=1e-5), 'rows not normalized'\n"
            "\n"
            "# Visualize attention weights for batch 0.\n"
            "fig, ax = plt.subplots(figsize=(4, 4))\n"
            "im = ax.imshow(weights[0].numpy(), cmap='magma', vmin=0, vmax=1)\n"
            "ax.set_title('attention weights (batch 0, causal mask)')\n"
            "ax.set_xlabel('key position j')\n"
            "ax.set_ylabel('query position i')\n"
            "plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex6_scaled_dot_product_attention(\n"
            "    q: Tensor, k: Tensor, v: Tensor, mask: Tensor\n"
            ") -> tuple[Tensor, Tensor]:\n"
            "    import torch.nn.functional as F\n"
            "    d = q.shape[-1]\n"
            "    scores = einsum(q, k, 'b i d, b j d -> b i j') / (d ** 0.5)\n"
            "    scores = scores.masked_fill(mask, float('-inf'))\n"
            "    weights = F.softmax(scores, dim=-1)\n"
            "    out = einsum(weights, v, 'b i j, b j e -> b i e')\n"
            "    return out, weights"
        ),
        "solution_notes": (
            "**Two einsums, one pipeline.** The first call contracts `d` (the QK^T inner product). The second contracts `j` (the weighted aggregation over keys/values). Notice the second einsum's pattern `'b i j, b j e -> b i e'` looks exactly like a batched matmul — that's because it **is** a batched matmul. The point of using `einsum` here is that the index names (`i` = query position, `j` = key position, `e` = value embedding dim) make the semantics readable in a way `weights @ v` does not.\n"
            "\n"
            "**Reading the heatmap.** With the causal mask, the upper triangle is exactly zero. Below the diagonal you should see one bright cell per row — that's the model assigning most of its weight to one key per query (a soft argmax). On random inputs the brightest cell tends to wander; on real trained attention you'd see structure (diagonals, induction-head stripes, etc.)."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex7 — multi-head split via einsum/rearrange + QK^T per head, debug prints
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "multi-head attention scores — split (b, s, h*d) and contract per head",
        "slug": "multi-head-scores-split-and-contract",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["multi-head", "split", "rearrange", "contraction", "debug-print"],
        "kcs": ["einsum-batched", "einsum-attention-scores", "einops-rearrange-axis-split"],
        "lo": "Split a packed (B, S, H*D) projection into per-head tensors via rearrange, then use einsum to compute (B, H, S, S) attention scores in one pattern, printing intermediate shapes.",
        "prompt_body": (
            "Implement `ex7_multihead_scores(q_packed, k_packed, h)`.\n"
            "\n"
            "Inputs:\n"
            "- `q_packed`: `(B, S, H*D)` — queries with the H heads packed into the last dim.\n"
            "- `k_packed`: `(B, S, H*D)` — keys, same packing.\n"
            "- `h`: int — number of heads (so per-head dim is `D = (H*D) / h`).\n"
            "\n"
            "Steps:\n"
            "1. Use `einops.rearrange` to split the packed dim: `(B, S, H*D) -> (B, H, S, D)` for both `q_packed` and `k_packed`. Print the shape of `q` after the split with prefix `q_split`.\n"
            "2. Use `einops.einsum` with one pattern to compute scores: `scores[b, h, i, j] = sum_d q[b,h,i,d] * k[b,h,j,d]`. Print the shape of `scores` with prefix `scores`.\n"
            "3. Return `scores` of shape `(B, H, S, S)`. Do **not** apply softmax or scale here.\n"
            "\n"
            "The test verifies the prints, the shapes, and the values against an explicit per-head reference."
        ),
        "stub": (
            "def ex7_multihead_scores(q_packed: Tensor, k_packed: Tensor, h: int) -> Tensor:\n"
            "    \"\"\"Multi-head attention scores from packed (B, S, H*D) inputs.\n"
            "    Must print `q_split shape=...` and `scores shape=...` debug lines.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import io, contextlib\n"
            "\n"
            "B, S, H, D = 2, 4, 3, 5\n"
            "q_packed = t.randn(B, S, H * D)\n"
            "k_packed = t.randn(B, S, H * D)\n"
            "\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            "    scores = ex7_multihead_scores(q_packed, k_packed, h=H)\n"
            "log = buf.getvalue()\n"
            "print(log, end='')\n"
            "\n"
            "assert scores.shape == (B, H, S, S), f'expected ({B},{H},{S},{S}), got {scores.shape}'\n"
            "\n"
            "# Ground truth: split manually, then per-head matmul.\n"
            "q_ref = q_packed.reshape(B, S, H, D).permute(0, 2, 1, 3)  # (B, H, S, D)\n"
            "k_ref = k_packed.reshape(B, S, H, D).permute(0, 2, 1, 3)\n"
            "expected = q_ref @ k_ref.transpose(-2, -1)               # (B, H, S, S)\n"
            "assert t.allclose(scores, expected, atol=1e-5), 'scores differ from per-head reference'\n"
            "\n"
            "# Debug-print contract.\n"
            "assert 'q_split' in log, f'missing `q_split` print:\\n{log}'\n"
            "assert 'scores' in log, f'missing `scores` print:\\n{log}'\n"
            "assert f'({B}, {H}, {S}, {D})' in log or f'{(B, H, S, D)}' in log, (\n"
            "    f'q_split shape not reported as ({B},{H},{S},{D}):\\n{log}'\n"
            ")\n"
            "assert f'({B}, {H}, {S}, {S})' in log or f'{(B, H, S, S)}' in log, (\n"
            "    f'scores shape not reported as ({B},{H},{S},{S}):\\n{log}'\n"
            ")\n"
            "\n"
            "# Visualize per-head scores for batch 0: H subplots.\n"
            "fig, axes = plt.subplots(1, H, figsize=(3 * H, 3))\n"
            "if H == 1:\n"
            "    axes = [axes]\n"
            "for hi in range(H):\n"
            "    im = axes[hi].imshow(scores[0, hi].numpy(), cmap='coolwarm')\n"
            "    axes[hi].set_title(f'head {hi}')\n"
            "    axes[hi].set_xlabel('key j'); axes[hi].set_ylabel('query i')\n"
            "    plt.colorbar(im, ax=axes[hi], fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex7_multihead_scores(q_packed: Tensor, k_packed: Tensor, h: int) -> Tensor:\n"
            "    q = rearrange(q_packed, 'b s (h d) -> b h s d', h=h)\n"
            "    k = rearrange(k_packed, 'b s (h d) -> b h s d', h=h)\n"
            "    print(f\"q_split shape={tuple(q.shape)}\")\n"
            "    scores = einsum(q, k, 'b h i d, b h j d -> b h i j')\n"
            "    print(f\"scores shape={tuple(scores.shape)}\")\n"
            "    return scores"
        ),
        "solution_notes": (
            "**Why rearrange + einsum, not one einsum.** einsum patterns don't split/merge axes — they only label and contract. So you need `rearrange` to do the head-split (`(h d) -> h d`) up front, then the einsum is a clean batched matmul over `d` with `b` and `h` both carried through.\n"
            "\n"
            "**Reading the per-head plots.** Each head sees the same input but applies a different score pattern (in real models, that's because Q and K were projected through different weight matrices first). Here we skipped the projection — the heads only differ because their packed-dim slices are different — so the score heatmaps will look unrelated to each other."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex8 — batched bilinear form y = x^T A x, visualized over a 2D x-grid
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "batched bilinear form y = x^T A x over a 2D grid, with heatmap viz",
        "slug": "batched-bilinear-form-with-heatmap",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["bilinear", "quadratic-form", "broadcast", "visualization"],
        "kcs": ["einsum-three-operand-contraction", "einsum-batched", "einsum-shared-tensor-broadcast"],
        "lo": "Use a single einsum pattern with three operands (x, A, x) to compute the quadratic form y = x^T A x for a batch of vectors, then visualize it as a 2D heatmap over a grid of x values.",
        "prompt_body": (
            "Implement `ex8_batched_bilinear(x, A)`.\n"
            "\n"
            "Inputs:\n"
            "- `x`: `(N, D)` — batch of N row-vectors.\n"
            "- `A`: `(D, D)` — a **shared** matrix used for every x in the batch.\n"
            "\n"
            "Output: `(N,)` — the vector of bilinear forms `y[n] = x[n] @ A @ x[n]` (a scalar per row).\n"
            "\n"
            "Use a **single** `einops.einsum` call passing three operands `(x, A, x)` and one pattern that contracts both indices of `A`. Do not reshape, transpose, or call `@` / `matmul`.\n"
            "\n"
            "The test cell additionally evaluates your function on a `49 x 49` grid of `x = (x0, x1)` in `[-2, 2]^2` with a fixed `A`, and plots the resulting scalar field as a heatmap. For a positive-definite `A` you should see concentric elliptical contours around the origin."
        ),
        "stub": (
            "def ex8_batched_bilinear(x: Tensor, A: Tensor) -> Tensor:\n"
            "    \"\"\"Compute y[n] = x[n] @ A @ x[n] for a batch of row-vectors, via one einsum.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Random correctness check.\n"
            "N, D = 7, 4\n"
            "x = t.randn(N, D)\n"
            "A = t.randn(D, D)\n"
            "y = ex8_batched_bilinear(x, A)\n"
            "assert y.shape == (N,), f'expected ({N},), got {y.shape}'\n"
            "\n"
            "# Reference: per-row x[n] @ A @ x[n].\n"
            "expected = t.stack([x[n] @ A @ x[n] for n in range(N)])\n"
            "assert t.allclose(y, expected, atol=1e-5), 'values differ from x[n] @ A @ x[n]'\n"
            "\n"
            "# Symmetric-PD case sanity (positive semi-definite A → y >= 0).\n"
            "A_pd = A.T @ A + 0.1 * t.eye(D)\n"
            "y_pd = ex8_batched_bilinear(x, A_pd)\n"
            "assert (y_pd >= -1e-6).all(), 'PSD A should produce non-negative bilinear values'\n"
            "\n"
            "# Visualize the bilinear scalar field over a 2D grid.\n"
            "G = 49\n"
            "grid = t.linspace(-2.0, 2.0, G)\n"
            "xx, yy = t.meshgrid(grid, grid, indexing='xy')\n"
            "x_grid = t.stack([xx.flatten(), yy.flatten()], dim=1)  # (G*G, 2)\n"
            "A_viz = t.tensor([[1.0, 0.6], [0.6, 2.0]])             # PSD, elliptical\n"
            "y_grid = ex8_batched_bilinear(x_grid, A_viz).reshape(G, G)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(4.2, 4))\n"
            "im = ax.imshow(\n"
            "    y_grid.numpy(),\n"
            "    extent=(-2, 2, -2, 2),\n"
            "    origin='lower',\n"
            "    cmap='viridis',\n"
            ")\n"
            "ax.contour(xx.numpy(), yy.numpy(), y_grid.numpy(), levels=10, colors='white', linewidths=0.5)\n"
            "ax.set_title('y = x^T A x over (x0, x1)')\n"
            "ax.set_xlabel('x0'); ax.set_ylabel('x1')\n"
            "plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex8_batched_bilinear(x: Tensor, A: Tensor) -> Tensor:\n"
            "    # x appears twice with different contraction indices (i, j); A contracts both.\n"
            "    # n is preserved (batch); i and j are both contracted (no shared output).\n"
            "    return einsum(x, A, x, 'n i, i j, n j -> n')"
        ),
        "solution_notes": (
            "**The trick: the same tensor twice with different indices.** einsum lets you pass the same tensor more than once — the index name on each position is what disambiguates. Here `x` appears with index `i` first and `j` second; the contraction `i j` is exactly the inner double-sum that defines `x^T A x`.\n"
            "\n"
            "**Reading the pattern.**\n"
            "- `n` appears in both `x` operands and on the output → preserved (batch axis).\n"
            "- `i` appears in the first `x` and in `A`, not on the output → contracted.\n"
            "- `j` appears in the second `x` and in `A`, not on the output → contracted.\n"
            "- `A` has no `n` → it's broadcast across the batch (shared matrix).\n"
            "\n"
            "**Reading the heatmap.** With `A = [[1, 0.6], [0.6, 2]]` (positive definite, eigenvalues both positive), the level sets are ellipses centered at the origin. The tilt comes from the off-diagonal `0.6`; the vertical squish comes from the larger eigenvalue along the y-axis (`A[1,1] = 2 > A[0,0] = 1`)."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex9 — 3-tensor Tucker-style contraction with intermediate-shape prints
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "three-tensor Tucker-style contraction with intermediate-shape debug prints",
        "slug": "three-tensor-tucker-contraction-debug",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["tucker", "three-tensor", "pairing-order", "debug-print"],
        "kcs": ["einsum-three-operand-contraction", "einsum-pairwise-vs-fused", "einsum-debug-introspection"],
        "lo": "Use a single 3-tensor einsum to project a core (P, Q, R) tensor onto three factor matrices U, V, W in one pattern, and compare its result against a hand-rolled 3-step pairwise contraction that prints each intermediate shape.",
        "prompt_body": (
            "Implement TWO functions:\n"
            "\n"
            "1. `ex9_tucker_fused(core, U, V, W)` — one-shot 3-operand einsum.\n"
            "   - `core`: `(P, Q, R)`.\n"
            "   - `U`: `(I, P)`. `V`: `(J, Q)`. `W`: `(K, R)`.\n"
            "   - Output: `(I, J, K)` where `out[i,j,k] = sum_p sum_q sum_r U[i,p] * V[j,q] * W[k,r] * core[p,q,r]`.\n"
            "   - Use ONE `einops.einsum` call with all four operands and one pattern.\n"
            "\n"
            "2. `ex9_tucker_pairwise(core, U, V, W)` — hand-rolled 3-step pairwise version, for comparison.\n"
            "   - Step 1: contract `p` between `U` and `core` → shape `(I, Q, R)`. Print: `step1 shape=...`.\n"
            "   - Step 2: contract `q` between `V` and step1 → shape `(I, J, R)`. Print: `step2 shape=...`.\n"
            "   - Step 3: contract `r` between `W` and step2 → shape `(I, J, K)`. Print: `step3 shape=...`.\n"
            "   - Each step uses a separate `einops.einsum` call.\n"
            "\n"
            "The test cell calls both and asserts they produce the **same** tensor (up to floating-point tolerance). It also greps the debug log for the three step prints."
        ),
        "stub": (
            "def ex9_tucker_fused(core: Tensor, U: Tensor, V: Tensor, W: Tensor) -> Tensor:\n"
            "    \"\"\"One-shot 3-mode Tucker reconstruction via a single 4-operand einsum.\"\"\"\n"
            "    raise NotImplementedError()\n"
            "\n"
            "\n"
            "def ex9_tucker_pairwise(core: Tensor, U: Tensor, V: Tensor, W: Tensor) -> Tensor:\n"
            "    \"\"\"Same Tucker reconstruction but as 3 pairwise einsums.\n"
            "    Must print `step1 shape=...`, `step2 shape=...`, `step3 shape=...`.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import io, contextlib\n"
            "\n"
            "P, Q, R = 3, 4, 5\n"
            "I, J, K = 6, 7, 2\n"
            "core = t.randn(P, Q, R)\n"
            "U = t.randn(I, P)\n"
            "V = t.randn(J, Q)\n"
            "W = t.randn(K, R)\n"
            "\n"
            "fused = ex9_tucker_fused(core, U, V, W)\n"
            "assert fused.shape == (I, J, K), f'fused: expected ({I},{J},{K}), got {fused.shape}'\n"
            "\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            "    pairwise = ex9_tucker_pairwise(core, U, V, W)\n"
            "log = buf.getvalue()\n"
            "print(log, end='')\n"
            "\n"
            "assert pairwise.shape == (I, J, K), f'pairwise: expected ({I},{J},{K}), got {pairwise.shape}'\n"
            "assert t.allclose(fused, pairwise, atol=1e-4), (\n"
            "    'fused vs pairwise disagree — same math should give the same answer'\n"
            ")\n"
            "\n"
            "# Independent ground truth via explicit nested matmuls.\n"
            "# m1[i,q,r] = sum_p U[i,p] * core[p,q,r]  ==  (U @ core.reshape(P, Q*R)).reshape(I,Q,R)\n"
            "m1 = (U @ core.reshape(P, Q * R)).reshape(I, Q, R)\n"
            "m2 = (V @ m1.permute(1, 0, 2).reshape(Q, I * R)).reshape(J, I, R).permute(1, 0, 2)\n"
            "expected = (W @ m2.permute(2, 0, 1).reshape(R, I * J)).reshape(K, I, J).permute(1, 2, 0)\n"
            "assert t.allclose(fused, expected, atol=1e-4), 'fused disagrees with reference'\n"
            "\n"
            "# Debug-print contract.\n"
            "for needle in ['step1', 'step2', 'step3']:\n"
            "    assert needle in log, f'missing `{needle}` debug print:\\n{log}'\n"
            "assert f'({I}, {Q}, {R})' in log, f'step1 should report ({I},{Q},{R}):\\n{log}'\n"
            "assert f'({I}, {J}, {R})' in log, f'step2 should report ({I},{J},{R}):\\n{log}'\n"
            "assert f'({I}, {J}, {K})' in log, f'step3 should report ({I},{J},{K}):\\n{log}'\n"
            "\n"
            "# Visualize the (I, J) slice at k=0 of the fused result as a heatmap.\n"
            "fig, ax = plt.subplots(figsize=(4, 4))\n"
            "im = ax.imshow(fused[..., 0].numpy(), cmap='coolwarm')\n"
            "ax.set_title(f'Tucker reconstruction[:,:,0]  shape={tuple(fused.shape)}')\n"
            "ax.set_xlabel('j'); ax.set_ylabel('i')\n"
            "plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex9_tucker_fused(core: Tensor, U: Tensor, V: Tensor, W: Tensor) -> Tensor:\n"
            "    return einsum(core, U, V, W, 'p q r, i p, j q, k r -> i j k')\n"
            "\n"
            "\n"
            "def ex9_tucker_pairwise(core: Tensor, U: Tensor, V: Tensor, W: Tensor) -> Tensor:\n"
            "    step1 = einsum(U, core, 'i p, p q r -> i q r')\n"
            "    print(f\"step1 shape={tuple(step1.shape)}\")\n"
            "    step2 = einsum(V, step1, 'j q, i q r -> i j r')\n"
            "    print(f\"step2 shape={tuple(step2.shape)}\")\n"
            "    step3 = einsum(W, step2, 'k r, i j r -> i j k')\n"
            "    print(f\"step3 shape={tuple(step3.shape)}\")\n"
            "    return step3"
        ),
        "solution_notes": (
            "**Tucker decomposition recap.** A Tucker reconstruction of a 3D tensor is a small \"core\" tensor `core ∈ ℝ^{P×Q×R}` projected back to full size by three factor matrices `U ∈ ℝ^{I×P}`, `V ∈ ℝ^{J×Q}`, `W ∈ ℝ^{K×R}`. The fused pattern `'p q r, i p, j q, k r -> i j k'` contracts `p`, `q`, `r` (all the small dims) and keeps `i`, `j`, `k` (the full dims).\n"
            "\n"
            "**Pairwise vs fused is the same math.** The fused einsum is mathematically identical to the 3-step pairwise version — `einops.einsum` delegates to `torch.einsum`, which uses `opt_einsum` to pick a contraction order automatically. Writing it out as 3 steps lets you **see** the intermediate shapes, which is the whole point of the debug prints: at each step, exactly one small index is consumed and one full index is materialized. The intermediate `(I, Q, R)` is small if `I ≪ Q*R`, large if `I ≫ Q*R` — and that choice is what `opt_einsum` agonizes over for big tensors."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
