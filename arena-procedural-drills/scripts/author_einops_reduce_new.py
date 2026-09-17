#!/usr/bin/env python3
"""Author ex6-ex9 for the einops-reduce atom.

Each new exercise is Colab-native — it does something flashcards can't:
visualize a reduction pyramid, debug a multi-step normalization pipeline,
or explore a reduce-driven algorithm whose intermediate state is the
whole point.

Run:
    python arena-procedural-drills/scripts/author_einops_reduce_new.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone  # noqa: E402

ATOM_ID = "einops-reduce"
SUBTOPIC = "Einops: Reduce"
TOPIC = "prereqs_einops"

RECAP = (
    "## einops.reduce — quick refresher\n"
    "\n"
    "`reduce(tensor, pattern, op)` collapses one or more named axes with a "
    "reduction `op` ∈ `{'mean', 'sum', 'max', 'min', 'prod'}`. Drop an axis "
    "name on the right side to reduce it; keep it inside parentheses on the "
    "left and decompose first to do windowed pooling.\n"
    "\n"
    "The exercises below stop being about *which op?* and start being about "
    "*reduce as part of a larger pipeline* — pyramid pooling, per-channel "
    "normalization, argmax-without-`torch.argmax`, top-k by repeated masked "
    "max. Each one needs visualization or print-debug to be solvable in your "
    "head."
)

SPECS = [
    # ──────────────────────────────────────────────────────────────────
    # ex6 — Spatial pyramid pooling with visualized levels
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "spatial pyramid pooling with imshow per level",
        "slug": "spatial-pyramid-pooling-imshow",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["pyramid", "pooling", "visualization", "matplotlib"],
        "kcs": ["reduce-axis-decomposition", "reduce-mean"],
        "lo": (
            "Build a spatial pyramid (1×1, 2×2, 4×4 mean pools) using "
            "einops.reduce, and visualize each pyramid level as an imshow "
            "to feel how information collapses as the grid coarsens."
        ),
        "prompt_body": (
            "Spatial Pyramid Pooling (SPP) and the pyramid pooling module in "
            "PSPNet both produce a *multi-scale* summary of a feature map by "
            "average-pooling it to several fixed grid sizes (1×1, 2×2, 4×4, …) "
            "and concatenating the results. The visualization tells you which "
            "regions of the image survived at which scale.\n"
            "\n"
            "Implement `ex6_pyramid_pool(feat, levels)`:\n"
            "1. `feat` has shape `(C, H, W)` (single image, multi-channel).\n"
            "2. `levels` is a list of grid sizes, e.g. `[1, 2, 4]`.\n"
            "3. For each `L` in `levels`, mean-pool `feat` down to "
            "`(C, L, L)` using one `reduce` call with axis decomposition. "
            "Hint: `'c (h p1) (w p2) -> c h w'` with `h=L, w=L` works when "
            "`H, W` are multiples of `L`.\n"
            "4. Plot the **channel-0** result at each level as imshow in a "
            "row of subplots, titled `'L=1'`, `'L=2'`, `'L=4'` etc.\n"
            "5. Return a list of pooled tensors, one per level.\n"
            "\n"
            "Looking at the plots, you should see the highest-energy regions "
            "of channel 0 survive even at the coarsest level."
        ),
        "stub": (
            "def ex6_pyramid_pool(feat: Tensor, levels: list[int]) -> list[Tensor]:\n"
            "    \"\"\"Multi-scale mean pool with per-level imshow of channel 0.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "C, H, W = 4, 16, 16\n"
            "# Build a feature with a hot spot in the top-left of channel 0,\n"
            "# so we can verify the coarsest pool still preserves it.\n"
            "feat = t.randn(C, H, W) * 0.1\n"
            "feat[0, :4, :4] += 5.0  # hot spot\n"
            "levels = [1, 2, 4]\n"
            "pools = ex6_pyramid_pool(feat, levels)\n"
            "assert len(pools) == len(levels), f'expected {len(levels)} pools, got {len(pools)}'\n"
            "for L, p in zip(levels, pools):\n"
            "    assert p.shape == (C, L, L), f'level {L}: shape {p.shape}'\n"
            "# Level 1: should equal the per-channel global mean.\n"
            "assert t.allclose(pools[0][:, 0, 0], feat.mean(dim=(1, 2)), atol=1e-5), \\\n"
            "    'L=1 pool should equal per-channel global mean'\n"
            "# Level 4: the top-left cell of channel 0 should be the mean of the\n"
            "# 4x4 hot-spot region (which has +5 added to ~0.1*N(0,1)).\n"
            "expected_topleft = feat[0, :4, :4].mean()\n"
            "assert t.allclose(pools[2][0, 0, 0], expected_topleft, atol=1e-5), \\\n"
            "    'L=4 pool top-left of channel 0 should equal the 4x4 sub-mean'\n"
            "# Hot spot survives at coarsest level (>> background).\n"
            "assert pools[0][0, 0, 0] > pools[0][1:, 0, 0].abs().max(), \\\n"
            "    'channel-0 hot spot should dominate at L=1'"
        ),
        "solution_body": (
            "def ex6_pyramid_pool(feat: Tensor, levels: list[int]) -> list[Tensor]:\n"
            "    import matplotlib.pyplot as plt\n"
            "    C, H, W = feat.shape\n"
            "    pools = []\n"
            "    for L in levels:\n"
            "        assert H % L == 0 and W % L == 0, f'level {L} must divide H,W'\n"
            "        p = reduce(feat, 'c (h p1) (w p2) -> c h w', 'mean', h=L, w=L)\n"
            "        pools.append(p)\n"
            "    fig, axes = plt.subplots(1, len(levels), figsize=(3 * len(levels), 3))\n"
            "    if len(levels) == 1:\n"
            "        axes = [axes]\n"
            "    for ax, L, p in zip(axes, levels, pools):\n"
            "        ax.imshow(p[0].cpu().numpy(), cmap='viridis')\n"
            "        ax.set_title(f'L={L}')\n"
            "        ax.axis('off')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    return pools"
        ),
        "solution_notes": (
            "**Why the pattern works.** `'c (h p1) (w p2) -> c h w'` with "
            "`h=L, w=L` decomposes `H = L * p1` and `W = L * p2`, then "
            "drops the `p1, p2` axes — implicitly meaning *reduce over those "
            "axes* via the `'mean'` op. One line, three jobs.\n"
            "\n"
            "**Why visualize?** A 1×1 pool is just a number per channel — fine "
            "for a classifier head, but invisible to inspection. A 4×4 pool is "
            "an image you can *look at*. The pyramid plots make it obvious "
            "that L=1 throws away spatial structure entirely while L=4 still "
            "preserves the hot spot's location."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ──────────────────────────────────────────────────────────────────
    # ex7 — Per-channel BatchNorm-style normalization with visualization
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "per-channel BN-style stats (mean, var, normalized output)",
        "slug": "per-channel-bn-stats-normalized-output",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["batchnorm", "broadcast", "keepdim", "normalization"],
        "kcs": ["reduce-mean", "reduce-keepdim-with-parens"],
        "lo": (
            "Use einops.reduce with `keepdim`-style `()` axes to compute "
            "per-channel mean+var over (batch, height, width), then broadcast "
            "back to normalize, and visualize the per-channel result."
        ),
        "prompt_body": (
            "BatchNorm in 2-D normalizes each channel independently using "
            "statistics gathered across the `(batch, height, width)` axes. "
            "Done naively with `.mean(dim=...)`, the result has the wrong "
            "shape for broadcasting back. Done with `reduce`'s `()` keepdim "
            "trick, the broadcast just works.\n"
            "\n"
            "Implement `ex7_per_channel_bn(x, eps=1e-5)`:\n"
            "1. `x` has shape `(N, C, H, W)`.\n"
            "2. Compute `mu` of shape `(1, C, 1, 1)` using one `reduce` call "
            "with `()` on the b/h/w slots. Same for `var` (use the formula "
            "`E[x²] - E[x]²` so you only do `reduce`-style ops).\n"
            "3. Return `(x - mu) / sqrt(var + eps)` — shape `(N, C, H, W)`.\n"
            "4. **Print** `mu` and `var` flattened (one row per channel), then "
            "plot the per-channel mean of the normalized output as a bar chart "
            "(should all be ≈ 0) and the per-channel variance as another bar "
            "chart (should all be ≈ 1).\n"
            "\n"
            "These two bar charts are the BatchNorm sanity check — if they're "
            "not flat at 0 and 1, your reduce broadcast is wrong."
        ),
        "stub": (
            "def ex7_per_channel_bn(x: Tensor, eps: float = 1e-5) -> Tensor:\n"
            "    \"\"\"BN-style per-channel normalize over (N, H, W); visualize output stats.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "N, C, H, W = 8, 3, 16, 16\n"
            "# Per-channel different means/scales so normalization has work to do.\n"
            "x = t.randn(N, C, H, W) * t.tensor([1.0, 3.0, 0.5]).view(1, C, 1, 1)\n"
            "x = x + t.tensor([0.0, 5.0, -2.0]).view(1, C, 1, 1)\n"
            "y = ex7_per_channel_bn(x)\n"
            "assert y.shape == x.shape, f'shape: {y.shape}'\n"
            "# Per-channel mean of output should be ~0, per-channel var ~1.\n"
            "out_mean = reduce(y, 'n c h w -> c', 'mean')\n"
            "assert t.allclose(out_mean, t.zeros(C), atol=1e-4), f'output means not zero: {out_mean}'\n"
            "# Variance check via E[y^2] - E[y]^2 (which is ~ E[y^2] since mean ~ 0).\n"
            "out_var = reduce(y ** 2, 'n c h w -> c', 'mean') - out_mean ** 2\n"
            "assert t.allclose(out_var, t.ones(C), atol=1e-3), f'output vars not one: {out_var}'\n"
            "# Sanity: the original per-channel means were NOT zero, so this is a real test.\n"
            "in_mean = reduce(x, 'n c h w -> c', 'mean')\n"
            "assert (in_mean.abs() > 0.1).any(), 'test setup: input should have non-zero per-channel mean'"
        ),
        "solution_body": (
            "def ex7_per_channel_bn(x: Tensor, eps: float = 1e-5) -> Tensor:\n"
            "    import matplotlib.pyplot as plt\n"
            "    # The () axes keep that slot in the output with size 1 — perfect for broadcast.\n"
            "    mu = reduce(x, 'n c h w -> () c () ()', 'mean')\n"
            "    mean_sq = reduce(x ** 2, 'n c h w -> () c () ()', 'mean')\n"
            "    var = mean_sq - mu ** 2\n"
            "    y = (x - mu) / t.sqrt(var + eps)\n"
            "    print('per-channel mu :', mu.flatten().tolist())\n"
            "    print('per-channel var:', var.flatten().tolist())\n"
            "    # Sanity bar charts.\n"
            "    out_mean = reduce(y, 'n c h w -> c', 'mean')\n"
            "    out_var = reduce(y ** 2, 'n c h w -> c', 'mean') - out_mean ** 2\n"
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))\n"
            "    ax1.bar(range(len(out_mean)), out_mean.cpu().numpy())\n"
            "    ax1.set_title('output mean per channel (should be ~0)')\n"
            "    ax1.axhline(0, color='k', linewidth=0.5)\n"
            "    ax2.bar(range(len(out_var)), out_var.cpu().numpy())\n"
            "    ax2.set_title('output var per channel (should be ~1)')\n"
            "    ax2.axhline(1, color='k', linewidth=0.5)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    return y"
        ),
        "solution_notes": (
            "**Why `()` in the pattern?** Writing `'n c h w -> () c () ()'` "
            "tells einops: collapse n, h, w but *keep* a length-1 slot in "
            "their positions. The result is shape `(1, C, 1, 1)`, which "
            "broadcasts against `(N, C, H, W)` with zero ceremony. Without "
            "the `()`, you'd get shape `(C,)` and need a manual `view(1, C, "
            "1, 1)` — that's the silent shape bug `()` exists to prevent.\n"
            "\n"
            "**Why E[x²] − E[x]²?** It lets you compute variance using only "
            "reduce-style ops, no `.var` call. Numerically less stable than "
            "the two-pass formula for huge tensors, but fine here and "
            "demonstrates that variance *is* just two reductions plus a "
            "subtract."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ──────────────────────────────────────────────────────────────────
    # ex8 — Argmax without torch.argmax, via reduce + arithmetic
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "argmax-via-reduce (no torch.argmax) with intermediate prints",
        "slug": "argmax-via-reduce-no-argmax",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["argmax", "boolean-mask", "reduce-max", "debug-print"],
        "kcs": ["reduce-max", "reduce-keepdim-with-parens"],
        "lo": (
            "Recover argmax indices using only einops.reduce + arithmetic + "
            "boolean masking, printing the intermediate max-mask and "
            "index-weighted tensor to see the algorithm work."
        ),
        "prompt_body": (
            "`torch.argmax` is a black box — it returns indices, you have no "
            "idea what happened. Reconstructing argmax from `reduce + max + "
            "broadcast + multiply` makes the algorithm transparent and "
            "doubles as a great `reduce`-keepdim exercise.\n"
            "\n"
            "Implement `ex8_argmax_via_reduce(x)` for a 2-D tensor `x` of "
            "shape `(R, C)`, returning a 1-D tensor of column indices (one "
            "per row), **without calling `torch.argmax`** or `.argmax`. The "
            "recipe:\n"
            "\n"
            "1. `max_per_row = reduce(x, 'r c -> r ()', 'max')` — shape `(R, "
            "1)`, broadcasts back over `x`.\n"
            "2. `is_max = (x == max_per_row)` — boolean mask, `True` where the "
            "value equals its row's max.\n"
            "3. `col_idx = torch.arange(C)` broadcast over rows.\n"
            "4. Multiply `is_max.float() * col_idx` then `reduce` with `'max'` "
            "over the column axis to pull out the largest True-position. "
            "(Why `max` and not `sum`? Ties — if two cells tie for max, you "
            "want a single index. Take the rightmost via `max`. If you'd "
            "rather match `torch.argmax`'s leftmost-tie rule, use a trick "
            "with a tiny epsilon weighted by `-col_idx`.)\n"
            "\n"
            "Print `max_per_row`, `is_max`, and the masked-index tensor "
            "before the final reduce so you can *see* the algorithm. Return "
            "the `(R,)` long-tensor of indices."
        ),
        "stub": (
            "def ex8_argmax_via_reduce(x: Tensor) -> Tensor:\n"
            "    \"\"\"Compute per-row argmax using reduce + mask + arithmetic only.\n"
            "\n"
            "    Returns a (R,) int64 tensor of column indices. On ties, returns the\n"
            "    rightmost column index (matches the 'max via masked indices' recipe;\n"
            "    note torch.argmax returns the leftmost — that's OK for this exercise).\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# 1) Unique-max case: should agree with torch.argmax exactly.\n"
            "x = t.tensor([\n"
            "    [1.0, 5.0, 3.0, 2.0],\n"
            "    [9.0, 0.0, 0.0, 1.0],\n"
            "    [0.0, 0.0, 0.0, 7.0],\n"
            "])\n"
            "idx = ex8_argmax_via_reduce(x)\n"
            "assert idx.shape == (3,), f'shape: {idx.shape}'\n"
            "assert idx.dtype in (t.int64, t.long), f'dtype should be long, got {idx.dtype}'\n"
            "expected = t.tensor([1, 0, 3])\n"
            "assert t.equal(idx, expected), f'expected {expected.tolist()}, got {idx.tolist()}'\n"
            "\n"
            "# 2) Tie case: rule is rightmost, so a tie at cols 1 and 3 should give 3.\n"
            "x_tie = t.tensor([[0.0, 5.0, 2.0, 5.0]])\n"
            "idx_tie = ex8_argmax_via_reduce(x_tie)\n"
            "assert idx_tie.item() == 3, f'tie should resolve to rightmost (3), got {idx_tie.item()}'\n"
            "\n"
            "# 3) Random sanity vs torch.argmax (where there are no ties).\n"
            "t.manual_seed(42)\n"
            "x_rand = t.randn(20, 10)\n"
            "idx_ours = ex8_argmax_via_reduce(x_rand)\n"
            "idx_ref = x_rand.argmax(dim=1)\n"
            "assert t.equal(idx_ours, idx_ref), 'should match torch.argmax when no ties'"
        ),
        "solution_body": (
            "def ex8_argmax_via_reduce(x: Tensor) -> Tensor:\n"
            "    R, C = x.shape\n"
            "    max_per_row = reduce(x, 'r c -> r ()', 'max')\n"
            "    is_max = (x == max_per_row)\n"
            "    col_idx = t.arange(C, device=x.device).expand(R, C)\n"
            "    # Mask out non-max positions with -1 so reduce-max ignores them\n"
            "    # (works because col indices are >= 0). For tied maxes, picking\n"
            "    # 'max' returns the rightmost index.\n"
            "    masked = t.where(is_max, col_idx, t.full_like(col_idx, -1))\n"
            "    print('max_per_row:\\n', max_per_row)\n"
            "    print('is_max:\\n', is_max)\n"
            "    print('masked col indices:\\n', masked)\n"
            "    idx = reduce(masked, 'r c -> r', 'max')\n"
            "    return idx.long()"
        ),
        "solution_notes": (
            "**Why the `-1` masking trick?** A boolean mask multiplied by "
            "indices works *unless* the argmax sits at column 0 — then both "
            "the true max-position and all the masked-out positions read as "
            "0, and `reduce-max` can't tell them apart. Using `-1` for "
            "masked-out positions makes the true-positive a strict winner.\n"
            "\n"
            "**Why this is in the curriculum.** Every `argmax`/`top-k` trick "
            "in attention code (relative position bias, ALiBi, sparse "
            "routing) is a variation on this recipe. Building it once by "
            "hand removes the magic.\n"
            "\n"
            "**Tie-breaking note.** `torch.argmax` returns the *leftmost* tie. "
            "Our recipe returns the *rightmost* because `reduce-max` over "
            "indices picks the largest. If you need leftmost-tie semantics, "
            "subtract `col_idx * eps` from `x` before the whole pipeline."
        ),
        "extra_imports": [],
    },
    # ──────────────────────────────────────────────────────────────────
    # ex9 — Top-k via repeated masked max
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "top-k via repeated masked max (integrative multi-step)",
        "slug": "top-k-via-repeated-masked-max",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["top-k", "iterative", "mask", "reduce-max"],
        "kcs": ["reduce-max", "reduce-keepdim-with-parens"],
        "lo": (
            "Combine reduce-max, boolean masking, and iteration to implement "
            "top-k from scratch, visualizing the mask after each iteration "
            "to see what got pulled out."
        ),
        "prompt_body": (
            "This integrates ex8's argmax recipe into a real algorithm: "
            "**top-k via repeated masked max**. The idea: find the max, "
            "record its position, mask it out, repeat `k` times. Slower than "
            "`torch.topk` for big `k`, but it's a transparent algorithm and "
            "the exact shape of code you'd write to implement custom routing "
            "(MoE top-k gating, beam search, k-WTA layers).\n"
            "\n"
            "Implement `ex9_topk_via_reduce(x, k)` for a 1-D tensor `x` of "
            "shape `(N,)`, returning `(values, indices)` both shape `(k,)`, "
            "in **descending order of value**, **without calling `torch.topk`** "
            "or `.topk`.\n"
            "\n"
            "Recipe:\n"
            "1. Start with a `mask` of all `True`, shape `(N,)`.\n"
            "2. For each iteration `i in 0..k-1`:\n"
            "   a. Use `reduce(masked_x, 'n -> ()', 'max')` (where `masked_x` "
            "has `-inf` at masked positions) to get the current max value.\n"
            "   b. Find its index by comparing `x == max_val` and taking the "
            "first match within the still-`True` mask (use the same `-1` "
            "trick from ex8).\n"
            "   c. Record value+index, then set `mask[chosen_idx] = False`.\n"
            "3. Plot `mask.float()` after each iteration as a row in a "
            "heatmap (shape `(k, N)`) — that's the visualization that makes "
            "the algorithm legible.\n"
            "\n"
            "Return `(values, indices)` as a tuple of 1-D tensors of length `k`."
        ),
        "stub": (
            "def ex9_topk_via_reduce(x: Tensor, k: int) -> tuple[Tensor, Tensor]:\n"
            "    \"\"\"Top-k via repeated masked max. Plots mask-per-iteration heatmap.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# 1) Hand case: top-3 of a small permutation.\n"
            "x = t.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])\n"
            "vals, idx = ex9_topk_via_reduce(x, k=3)\n"
            "assert vals.shape == (3,), f'values shape: {vals.shape}'\n"
            "assert idx.shape == (3,), f'indices shape: {idx.shape}'\n"
            "assert t.equal(vals, t.tensor([9.0, 6.0, 5.0])), f'values: {vals.tolist()}'\n"
            "assert t.equal(idx, t.tensor([5, 7, 4])), f'indices: {idx.tolist()}'\n"
            "\n"
            "# 2) Cross-check vs torch.topk on random data (no ties).\n"
            "t.manual_seed(7)\n"
            "x_rand = t.randn(50)\n"
            "vals_ref, idx_ref = x_rand.topk(5)\n"
            "vals_ours, idx_ours = ex9_topk_via_reduce(x_rand, k=5)\n"
            "assert t.allclose(vals_ours, vals_ref), f'value mismatch:\\n  ours={vals_ours}\\n  ref={vals_ref}'\n"
            "assert t.equal(idx_ours, idx_ref), f'index mismatch: {idx_ours.tolist()} vs {idx_ref.tolist()}'\n"
            "\n"
            "# 3) k == N should give a full sort (descending).\n"
            "x_small = t.tensor([2.0, 5.0, 1.0, 4.0])\n"
            "v_all, i_all = ex9_topk_via_reduce(x_small, k=4)\n"
            "assert t.equal(v_all, t.tensor([5.0, 4.0, 2.0, 1.0])), 'full top-k should sort'\n"
            "assert t.equal(i_all, t.tensor([1, 3, 0, 2])), 'index sort mismatch'"
        ),
        "solution_body": (
            "def ex9_topk_via_reduce(x: Tensor, k: int) -> tuple[Tensor, Tensor]:\n"
            "    import matplotlib.pyplot as plt\n"
            "    N = x.shape[0]\n"
            "    mask = t.ones(N, dtype=t.bool, device=x.device)\n"
            "    values = []\n"
            "    indices = []\n"
            "    mask_history = []\n"
            "    neg_inf = t.tensor(float('-inf'), device=x.device, dtype=x.dtype)\n"
            "    for _ in range(k):\n"
            "        masked_x = t.where(mask, x, neg_inf)\n"
            "        # reduce-max over the single axis; '() -> ()' keeps no axis.\n"
            "        max_val = reduce(masked_x, 'n -> ()', 'max').squeeze()\n"
            "        # Find first index where x == max_val AND still unmasked.\n"
            "        is_max_here = (masked_x == max_val)\n"
            "        col_idx = t.arange(N, device=x.device)\n"
            "        # Pick the *leftmost* surviving max via min over true-positions.\n"
            "        masked_idx = t.where(is_max_here, col_idx, t.full_like(col_idx, N))\n"
            "        chosen = reduce(masked_idx, 'n -> ()', 'min').squeeze().long()\n"
            "        values.append(max_val)\n"
            "        indices.append(chosen)\n"
            "        mask[chosen] = False\n"
            "        mask_history.append(mask.clone().float())\n"
            "    history = t.stack(mask_history)  # (k, N)\n"
            "    fig, ax = plt.subplots(figsize=(8, max(2, k * 0.4)))\n"
            "    ax.imshow(history.cpu().numpy(), aspect='auto', cmap='gray')\n"
            "    ax.set_xlabel('position')\n"
            "    ax.set_ylabel('iteration')\n"
            "    ax.set_title(f'mask state after each of {k} iterations (white = still eligible)')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    return t.stack(values), t.stack(indices).long()"
        ),
        "solution_notes": (
            "**Why this matches `torch.topk` exactly.** Both algorithms break "
            "ties by leftmost index (we use `min` over surviving-true index "
            "positions, which is the leftmost). And both return values in "
            "descending order because we pull the max out each iteration.\n"
            "\n"
            "**Why visualize the mask.** Each row of the heatmap is a "
            "snapshot of \"who's still eligible\". You see a single pixel "
            "flip black each iteration. This is exactly the picture in your "
            "head when you debug a routing layer that's eliminating the "
            "wrong tokens.\n"
            "\n"
            "**Performance note.** This is O(k·N). Real `topk` uses a partial "
            "sort and is O(N log k). For `k ≪ N`, the difference doesn't "
            "matter; for `k ≈ N`, prefer `sort`."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for spec in SPECS:
        path = emit_standalone(spec)
        print(f"wrote {path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
