#!/usr/bin/env python3
"""Author 4 new Colab-native exercises (ex6-ex9) for atom `as-strided-noncontig-source`.

These are NOT formula one-shots — each requires visualization, multi-step
debugging, or integrative composition that flashcards can't deliver:
  ex6 — 2-D sliding window via as_strided for image patches (matplotlib grid)
  ex7 — Memory cost comparison: same logical reshape, with/without .contiguous()
  ex8 — Strided 1-D convolution via as_strided + einsum (step-by-step debug)
  ex9 — Diagonal extraction via as_strided (stride manipulation + visualization)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

ATOM_ID = "as-strided-noncontig-source"
SUBTOPIC = "Numpy: Applied patterns and advanced"
TOPIC = "prereqs_numpy"

RECAP = (
    "## strides and non-contiguity — quick refresher\n"
    "\n"
    "**Stride** = number of *elements* (not bytes) to advance one step along an axis. "
    "A contiguous `(H, W)` float tensor has stride `(W, 1)`.\n"
    "\n"
    "**`torch.as_strided(input, size, stride)`** builds a zero-copy view at the exact "
    "(shape, stride) you specify. It bypasses safety checks — overlapping windows, "
    "out-of-bounds offsets, the works. Powerful, dangerous, and the foundation of "
    "rolling-window tricks, im2col, and stride-based broadcasting hacks.\n"
    "\n"
    "**`.contiguous()`** materializes a row-major copy if the current strides aren't "
    "already row-major. Required before `.view()`; optional but often a perf-vs-memory "
    "trade-off otherwise."
)

SPECS = [
    # ─────────────────────────────────────────────────────────────────────────
    # ex6 — 2-D sliding window via as_strided + matplotlib patch grid
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "2-D sliding-window image patches + visualize",
        "slug": "2d-sliding-window-image-patches-visualize",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["sliding-window", "as_strided", "image-patches", "visualization", "im2col"],
        "kcs": ["strides-anatomy", "as-strided-rolling-window", "stride-2d-window"],
        "lo": "Build a zero-copy 2-D sliding-window view over an image with `as_strided`, "
              "then visualize each patch as a tile in a matplotlib grid.",
        "prompt_body": (
            "Implement `ex6_image_patches(img, kh, kw)` to return a 4-D view of shape "
            "`(num_h, num_w, kh, kw)` containing every contiguous `kh × kw` patch of the "
            "2-D `img` tensor.\n"
            "\n"
            "Use `torch.as_strided` so the patches are a **zero-copy view** into the "
            "original storage — no data duplication. Hint: given `img` of shape `(H, W)` "
            "with stride `(sH, sW)`, the output shape is `(H - kh + 1, W - kw + 1, kh, kw)` "
            "and the output stride is `(sH, sW, sH, sW)`.\n"
            "\n"
            "After your test passes, the visualization cell below renders a grid of every "
            "patch as a small `imshow` tile — a debugging trick you'll reach for when "
            "checking whether your strides line up with what you intended."
        ),
        "stub": (
            "def ex6_image_patches(img: Tensor, kh: int, kw: int) -> Tensor:\n"
            "    \"\"\"Return (num_h, num_w, kh, kw) sliding-window view of img via as_strided.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a 6x6 image with a recognizable pattern.\n"
            "img = t.arange(36, dtype=t.float32).reshape(6, 6)\n"
            "patches = ex6_image_patches(img, 3, 3)\n"
            "\n"
            "# Shape check.\n"
            "assert patches.shape == (4, 4, 3, 3), f'expected (4,4,3,3), got {tuple(patches.shape)}'\n"
            "\n"
            "# Zero-copy: must share storage with img.\n"
            "assert patches.data_ptr() == img.data_ptr(), 'patches must be a view, not a copy'\n"
            "\n"
            "# Top-left patch should be img[:3, :3].\n"
            "assert t.equal(patches[0, 0], img[:3, :3]), f'top-left patch wrong:\\n{patches[0, 0]}'\n"
            "\n"
            "# Bottom-right patch should be img[3:, 3:].\n"
            "assert t.equal(patches[3, 3], img[3:, 3:]), f'bottom-right patch wrong:\\n{patches[3, 3]}'\n"
            "\n"
            "# Mutation through the view propagates back (proves view-not-copy).\n"
            "img2 = t.zeros(5, 5)\n"
            "p2 = ex6_image_patches(img2, 2, 2)\n"
            "p2[0, 0, 0, 0] = 99.0\n"
            "assert img2[0, 0].item() == 99.0, 'patch view must alias img storage'\n"
            "\n"
            "# Visualize: render every patch as a tile in a grid.\n"
            "fig, axes = plt.subplots(4, 4, figsize=(6, 6))\n"
            "for i in range(4):\n"
            "    for j in range(4):\n"
            "        axes[i, j].imshow(patches[i, j].numpy(), cmap='viridis', vmin=0, vmax=35)\n"
            "        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])\n"
            "        axes[i, j].set_title(f'({i},{j})', fontsize=7)\n"
            "fig.suptitle('Every 3x3 patch of a 6x6 image (zero-copy via as_strided)')\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
            "print(f'img stride={img.stride()}  patches stride={patches.stride()}')"
        ),
        "solution_body": (
            "def ex6_image_patches(img: Tensor, kh: int, kw: int) -> Tensor:\n"
            "    H, W = img.shape\n"
            "    sH, sW = img.stride()\n"
            "    out_h, out_w = H - kh + 1, W - kw + 1\n"
            "    return t.as_strided(\n"
            "        img,\n"
            "        size=(out_h, out_w, kh, kw),\n"
            "        stride=(sH, sW, sH, sW),\n"
            "    )"
        ),
        "solution_notes": (
            "**Why the strides repeat.** The first two axes (`out_h`, `out_w`) walk the "
            "*top-left corner* of each patch across the image — same step size as moving "
            "one row / column in the source. The last two axes (`kh`, `kw`) walk *within* "
            "a patch — also one source row / column. So `(sH, sW, sH, sW)` is exactly right.\n"
            "\n"
            "**Memory cost.** Zero. The output is a view — same storage, just a different "
            "`(size, stride)` interpretation. The materialized 4-D tensor would cost "
            "`(H-kh+1) × (W-kw+1) × kh × kw` floats, which for a 224×224 image with 7×7 "
            "patches is ~2.4M floats vs ~50K in the source. This is why `as_strided` is "
            "the secret sauce behind efficient im2col and convolution implementations."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex7 — Memory cost comparison: reshape with vs without .contiguous()
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "memory cost: contiguous() vs view comparison table",
        "slug": "memory-cost-contiguous-vs-view-comparison-table",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["contiguous", "memory-cost", "data_ptr", "element_size", "view-vs-copy"],
        "kcs": ["contiguous-fixes-view", "view-requires-contiguous", "memory-cost-of-contiguous"],
        "lo": "Quantify the storage cost of `.contiguous()` vs a strided view by measuring "
              "shared `data_ptr`, `element_size * numel`, and producing a side-by-side table.",
        "prompt_body": (
            "Implement `ex7_compare_layouts(x)`. Given a contiguous `(H, W)` tensor `x`, "
            "return a `dict` with three keys describing three logically-equivalent "
            "tensors derived from `x.T` (which is non-contiguous):\n"
            "\n"
            "```\n"
            "{\n"
            "  'view_attempt': dict with keys 'succeeded' (bool), 'error' (str|None),\n"
            "  'contig_then_view': dict with keys 'shares_storage' (bool), 'bytes' (int),\n"
            "  'reshape': dict with keys 'shares_storage' (bool), 'bytes' (int),\n"
            "}\n"
            "```\n"
            "\n"
            "- `view_attempt`: try `x.T.view(-1)`. Catch the `RuntimeError`. Record whether "
            "  it succeeded and the error message (or `None`).\n"
            "- `contig_then_view`: compute `y = x.T.contiguous().view(-1)`. Check whether "
            "  `y.data_ptr() == x.data_ptr()` (it won't) and report `y.element_size() * y.numel()`.\n"
            "- `reshape`: compute `z = x.T.reshape(-1)`. Same checks as above.\n"
            "\n"
            "The test then prints a formatted comparison table so you can *see* that "
            "`.contiguous()` and `.reshape()` on a non-contiguous source both pay a full "
            "memory copy."
        ),
        "stub": (
            "def ex7_compare_layouts(x: Tensor) -> dict:\n"
            "    \"\"\"Return a dict comparing view / contiguous+view / reshape on x.T.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12, dtype=t.float32).reshape(3, 4)  # contiguous (3, 4)\n"
            "report = ex7_compare_layouts(x)\n"
            "\n"
            "# Structure.\n"
            "assert set(report.keys()) == {'view_attempt', 'contig_then_view', 'reshape'}, \\\n"
            "    f'unexpected keys: {sorted(report.keys())}'\n"
            "\n"
            "# view should fail on non-contiguous transpose.\n"
            "assert report['view_attempt']['succeeded'] is False\n"
            "assert report['view_attempt']['error'] is not None\n"
            "assert 'contiguous' in report['view_attempt']['error'].lower() or \\\n"
            "       'non-contiguous' in report['view_attempt']['error'].lower() or \\\n"
            "       'view' in report['view_attempt']['error'].lower(), \\\n"
            "    f'expected error about contiguity/view, got: {report[\"view_attempt\"][\"error\"]}'\n"
            "\n"
            "# contig + view: must be a fresh allocation (copy).\n"
            "assert report['contig_then_view']['shares_storage'] is False\n"
            "expected_bytes = 12 * 4  # 12 float32 elements × 4 bytes\n"
            "assert report['contig_then_view']['bytes'] == expected_bytes, \\\n"
            "    f'expected {expected_bytes} bytes, got {report[\"contig_then_view\"][\"bytes\"]}'\n"
            "\n"
            "# reshape on non-contiguous source also copies.\n"
            "assert report['reshape']['shares_storage'] is False\n"
            "assert report['reshape']['bytes'] == expected_bytes\n"
            "\n"
            "# Print the comparison table.\n"
            "print(f'{\"strategy\":<22}{\"succeeded\":<12}{\"shares storage\":<18}{\"bytes\":<8}')\n"
            "print('-' * 60)\n"
            "va = report['view_attempt']\n"
            "print(f'{\"x.T.view(-1)\":<22}{str(va[\"succeeded\"]):<12}{\"n/a\":<18}{\"n/a\":<8}')\n"
            "ct = report['contig_then_view']\n"
            "print(f'{\"x.T.contiguous().view\":<22}{\"True\":<12}{str(ct[\"shares_storage\"]):<18}{ct[\"bytes\"]:<8}')\n"
            "rs = report['reshape']\n"
            "print(f'{\"x.T.reshape(-1)\":<22}{\"True\":<12}{str(rs[\"shares_storage\"]):<18}{rs[\"bytes\"]:<8}')\n"
            "print()\n"
            "print(f'(view error was: {va[\"error\"]!r})')"
        ),
        "solution_body": (
            "def ex7_compare_layouts(x: Tensor) -> dict:\n"
            "    out = {}\n"
            "\n"
            "    # 1. Try the doomed direct view.\n"
            "    try:\n"
            "        _ = x.T.view(-1)\n"
            "        out['view_attempt'] = {'succeeded': True, 'error': None}\n"
            "    except RuntimeError as e:\n"
            "        out['view_attempt'] = {'succeeded': False, 'error': str(e)}\n"
            "\n"
            "    # 2. .contiguous() then .view() — always works, always copies.\n"
            "    y = x.T.contiguous().view(-1)\n"
            "    out['contig_then_view'] = {\n"
            "        'shares_storage': y.data_ptr() == x.data_ptr(),\n"
            "        'bytes': y.element_size() * y.numel(),\n"
            "    }\n"
            "\n"
            "    # 3. .reshape() — view if possible, copy if not. On x.T it must copy.\n"
            "    z = x.T.reshape(-1)\n"
            "    out['reshape'] = {\n"
            "        'shares_storage': z.data_ptr() == x.data_ptr(),\n"
            "        'bytes': z.element_size() * z.numel(),\n"
            "    }\n"
            "\n"
            "    return out"
        ),
        "solution_notes": (
            "**Takeaway.** Every \"flatten a transpose\" pattern costs you `numel × itemsize` "
            "bytes of *new* allocation, because the row-major output simply can't share "
            "storage with column-major-ordered data. `.reshape()` is a polite wrapper that "
            "calls `.contiguous()` for you when needed — same cost, fewer try/except blocks.\n"
            "\n"
            "**When this bites you in production.** Hot inner loops that transpose+flatten "
            "tensors per step are silently allocating-and-freeing the same buffer thousands "
            "of times. Profile with `torch.profiler` and look for `aten::contiguous` calls "
            "you didn't write — they're almost always implicit from `.reshape()` on a "
            "non-contiguous source."
        ),
        "extra_imports": [],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex8 — Strided 1-D convolution via as_strided + einsum (multi-step debug)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "1-D convolution via as_strided + einsum pipeline",
        "slug": "1d-convolution-via-as-strided-einsum-pipeline",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["convolution", "as_strided", "einsum", "rolling-window", "integrative"],
        "kcs": ["as-strided-rolling-window", "einsum-axis-contract", "strided-conv-pipeline"],
        "lo": "Compose `as_strided` (build the window view) with `einsum` (contract over "
              "the window axis) to implement 1-D valid convolution end-to-end.",
        "prompt_body": (
            "Implement `ex8_conv1d_via_strided(x, kernel)` — a valid 1-D convolution "
            "(no padding, stride 1) using **only** `t.as_strided` for the windowing and "
            "**only** `t.einsum` for the summation.\n"
            "\n"
            "Given `x` of shape `(L,)` and `kernel` of shape `(K,)`, return an output of "
            "shape `(L - K + 1,)` where `out[i] = sum_j x[i+j] * kernel[j]`.\n"
            "\n"
            "**Pipeline (build this in order):**\n"
            "1. Pull `(L,)`, `(sL,)` off `x.shape` / `x.stride()`. Print them.\n"
            "2. Use `t.as_strided` to construct a windows view of shape "
            "`(L - K + 1, K)` with stride `(sL, sL)`. Print the windows shape + stride.\n"
            "3. Contract the windows with the kernel via `t.einsum('ij,j->i', windows, kernel)`. "
            "Print the output shape.\n"
            "\n"
            "The test compares against `torch.nn.functional.conv1d` for correctness.\n"
            "\n"
            "> ⚠️ **Integrative.** Three concepts in one pipeline (stride math + window view "
            "+ einsum reduction). Step through it with print statements — don't try to "
            "one-line it on the first attempt."
        ),
        "stub": (
            "def ex8_conv1d_via_strided(x: Tensor, kernel: Tensor) -> Tensor:\n"
            "    \"\"\"1-D valid convolution implemented with as_strided + einsum.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn.functional as F\n"
            "\n"
            "# Small hand-checkable case.\n"
            "x = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "kernel = t.tensor([1.0, 0.0, -1.0])  # finite-difference filter\n"
            "out = ex8_conv1d_via_strided(x, kernel)\n"
            "expected = t.tensor([1.0 - 3.0, 2.0 - 4.0, 3.0 - 5.0])  # [-2, -2, -2]\n"
            "assert out.shape == (3,), f'expected (3,), got {tuple(out.shape)}'\n"
            "assert t.allclose(out, expected), f'finite-diff mismatch: {out} vs {expected}'\n"
            "\n"
            "# Compare against torch.nn.functional.conv1d on a larger random case.\n"
            "# Note: F.conv1d does cross-correlation (same as our formula), so no flip.\n"
            "t.manual_seed(7)\n"
            "x2 = t.randn(32)\n"
            "k2 = t.randn(5)\n"
            "ours = ex8_conv1d_via_strided(x2, k2)\n"
            "ref = F.conv1d(x2.view(1, 1, -1), k2.view(1, 1, -1)).view(-1)\n"
            "assert ours.shape == ref.shape, f'shape mismatch: {ours.shape} vs {ref.shape}'\n"
            "assert t.allclose(ours, ref, atol=1e-5), f'value mismatch vs F.conv1d:\\n{ours}\\n{ref}'\n"
            "\n"
            "# Smoke check: the windows view should share storage with x (no copy).\n"
            "L, K = x.shape[0], kernel.shape[0]\n"
            "sL, = x.stride()\n"
            "windows_dbg = t.as_strided(x, size=(L - K + 1, K), stride=(sL, sL))\n"
            "assert windows_dbg.data_ptr() == x.data_ptr(), 'windows view must alias x storage'\n"
            "\n"
            "print(f'x.shape={tuple(x.shape)}  x.stride()={x.stride()}')\n"
            "print(f'windows.shape={tuple(windows_dbg.shape)}  windows.stride()={windows_dbg.stride()}')\n"
            "print(f'out.shape={tuple(out.shape)}  out={out.tolist()}')\n"
            "print(f'matches F.conv1d on len-32 input: {t.allclose(ours, ref, atol=1e-5)}')"
        ),
        "solution_body": (
            "def ex8_conv1d_via_strided(x: Tensor, kernel: Tensor) -> Tensor:\n"
            "    L = x.shape[0]\n"
            "    K = kernel.shape[0]\n"
            "    sL, = x.stride()\n"
            "    windows = t.as_strided(x, size=(L - K + 1, K), stride=(sL, sL))\n"
            "    return t.einsum('ij,j->i', windows, kernel)"
        ),
        "solution_notes": (
            "**Why this is the canonical \"convolution from scratch\" trick.** Modern "
            "conv kernels under the hood do exactly this — build a windowed view "
            "(im2col-style) and reduce via matmul/einsum. The only reason "
            "`torch.nn.functional.conv1d` is faster is the fused cuDNN kernel; the "
            "math is identical.\n"
            "\n"
            "**Subtle gotcha.** `F.conv1d` does cross-correlation by default — same "
            "formula as ours. \"True\" convolution flips the kernel: "
            "`out[i] = sum_j x[i+j] * kernel[K-1-j]`. Pass `kernel.flip(0)` if you ever "
            "need the textbook signal-processing convention."
        ),
        "extra_imports": [],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex9 — Diagonal extraction via as_strided + visualization
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "diagonal extraction via stride manipulation",
        "slug": "diagonal-extraction-via-stride-manipulation",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["diagonal", "as_strided", "stride-arithmetic", "visualization"],
        "kcs": ["strides-anatomy", "as-strided-arbitrary-view"],
        "lo": "Use stride arithmetic to extract the main diagonal of a square matrix as a "
              "zero-copy 1-D view, and visualize the access pattern.",
        "prompt_body": (
            "Implement `ex9_diagonal_via_strided(m)`. Given a square `(N, N)` tensor `m`, "
            "return a 1-D view of length `N` containing `[m[0,0], m[1,1], ..., m[N-1,N-1]]` "
            "— the main diagonal — built **using only `t.as_strided`**.\n"
            "\n"
            "Key insight: if `m` has stride `(sR, sC)`, then advancing one step along the "
            "diagonal means moving one row *and* one column — a single offset of `sR + sC` "
            "elements in storage. So the diagonal view has shape `(N,)` and stride "
            "`(sR + sC,)`.\n"
            "\n"
            "The test verifies values + zero-copy aliasing, then visualizes the access "
            "pattern: a heatmap of `m` with the diagonal cells highlighted, showing exactly "
            "which storage positions your strided view reads."
        ),
        "stub": (
            "def ex9_diagonal_via_strided(m: Tensor) -> Tensor:\n"
            "    \"\"\"Return the main diagonal of square m as a zero-copy 1-D view.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import numpy as np\n"
            "\n"
            "N = 5\n"
            "m = t.arange(N * N, dtype=t.float32).reshape(N, N)\n"
            "diag = ex9_diagonal_via_strided(m)\n"
            "\n"
            "# Shape and value checks.\n"
            "assert diag.shape == (N,), f'expected ({N},), got {tuple(diag.shape)}'\n"
            "expected = t.tensor([m[i, i].item() for i in range(N)])\n"
            "assert t.equal(diag, expected), f'diagonal mismatch: {diag} vs {expected}'\n"
            "\n"
            "# Zero-copy: must share storage with m.\n"
            "assert diag.data_ptr() == m.data_ptr(), 'diagonal must be a view, not a copy'\n"
            "\n"
            "# Stride sanity: should be sR + sC for contiguous m, that's N + 1 = 6.\n"
            "sR, sC = m.stride()\n"
            "assert diag.stride() == (sR + sC,), f'expected stride ({sR + sC},), got {diag.stride()}'\n"
            "\n"
            "# Cross-check against t.diagonal as ground truth.\n"
            "assert t.equal(diag, m.diagonal()), 'must agree with t.diagonal'\n"
            "\n"
            "# Mutation propagates back through the view.\n"
            "diag2 = ex9_diagonal_via_strided(m)\n"
            "before = m[2, 2].item()\n"
            "diag2[2] = -7.0\n"
            "assert m[2, 2].item() == -7.0, 'mutation through diag view should hit m[i,i]'\n"
            "m[2, 2] = before  # restore for the plot\n"
            "\n"
            "# Visualize: heatmap of m with diagonal cells outlined.\n"
            "fig, ax = plt.subplots(figsize=(4.5, 4.5))\n"
            "ax.imshow(m.numpy(), cmap='Blues')\n"
            "for i in range(N):\n"
            "    for j in range(N):\n"
            "        color = 'red' if i == j else 'black'\n"
            "        weight = 'bold' if i == j else 'normal'\n"
            "        ax.text(j, i, f'{int(m[i, j].item())}', ha='center', va='center',\n"
            "                color=color, fontweight=weight)\n"
            "ax.set_xticks(range(N)); ax.set_yticks(range(N))\n"
            "ax.set_title(f'diag = as_strided(m, ({N},), ({sR + sC},))  -- red cells are the view')\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(f'm.stride() = {m.stride()}')\n"
            "print(f'diag.stride() = {diag.stride()}  (= sR + sC = {sR + sC})')\n"
            "print(f'diag = {diag.tolist()}')"
        ),
        "solution_body": (
            "def ex9_diagonal_via_strided(m: Tensor) -> Tensor:\n"
            "    N = m.shape[0]\n"
            "    sR, sC = m.stride()\n"
            "    return t.as_strided(m, size=(N,), stride=(sR + sC,))"
        ),
        "solution_notes": (
            "**Why `sR + sC` is the diagonal step.** From `m[i, i]` to `m[i+1, i+1]` you "
            "move one row down (`+sR` elements in storage) *and* one column right "
            "(`+sC` elements). The diagonal view is just the source storage sampled every "
            "`sR + sC` elements.\n"
            "\n"
            "**Off-diagonals.** For the `k`-th diagonal above the main, start your view at "
            "offset `k * sC` (use `t.as_strided(m[..., k:], ...)` or pass an explicit "
            "`storage_offset`) with shape `(N - k,)` and stride `(sR + sC,)`. Below the "
            "main, swap roles. This is the entire trick behind banded-matrix routines."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        # path.parents: [atom-dir, topic-dir, drill-root, repo, ...]
        print(f"wrote {path.relative_to(path.parents[4])}")
