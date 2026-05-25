#!/usr/bin/env python3
"""Author ex6-ex9 for the einops-rearrange atom.

Each new exercise is Colab-native — it does something flashcards can't:
visualize layouts, debug a multi-step pipeline by printing shapes, or
combine several KCs in a realistic ML-adjacent flow.

Run:
    python arena-procedural-drills/scripts/author_einops_rearrange_new.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone  # noqa: E402

ATOM_ID = "einops-rearrange"
SUBTOPIC = "Einops: Rearrange"
TOPIC = "prereqs_einops"

RECAP = (
    "## einops.rearrange — quick refresher\n"
    "\n"
    "`rearrange(tensor, pattern, **axes_lengths)` is one operator with three jobs: "
    "**reorder** axes (`'h w -> w h'`), **compose** them (`'h w c -> (h w) c'`), and "
    "**decompose** them (`'(b1 b2) c -> b1 b2 c'`, with `b1=` or `b2=`). Every identifier "
    "on the right must appear on the left and vice versa.\n"
    "\n"
    "The exercises below build on that: each one runs `rearrange` inside a small pipeline "
    "where you have to *see* what the layout did — by plotting it, by printing the shape "
    "at each step, or by combining 2–3 patterns into a single ML-adjacent transformation."
)

SPECS = [
    # ──────────────────────────────────────────────────────────────────
    # ex6 — Multi-head attention split prep, with a shape-print debug
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "multi-head attention split (with shape-pipeline debug)",
        "slug": "multi-head-attention-split-shape-pipeline",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["multi-head", "attention", "axis-decomposition", "shape-debug"],
        "kcs": ["rearrange-axis-decomposition", "rearrange-axis-swap"],
        "lo": (
            "Use rearrange to split a packed (b, s, h*d) tensor into the "
            "(b, h, s, d) layout expected by scaled-dot-product attention, "
            "and verify the intermediate shapes by printing them."
        ),
        "prompt_body": (
            "In multi-head attention, the projected Q/K/V tensors arrive as "
            "`(batch, seq_len, n_heads * head_dim)`. Before the attention matmul, "
            "you need to split the last axis into `n_heads` and `head_dim`, then "
            "move `n_heads` next to `batch` so attention scores can be computed "
            "per-head in a single batched matmul.\n"
            "\n"
            "Implement `ex6_split_heads(x, n_heads)` that takes `x` of shape "
            "`(b, s, h*d)` and returns `(b, h, s, d)` using **one** `rearrange` "
            "call. Before returning, **print** the input shape, the intermediate "
            "shape if you had split first then moved (you can compute it from the "
            "input shape — you don't need a second rearrange), and the output "
            "shape. The print is the point: this is the kind of multi-step "
            "transformation you have to feel in your fingers when debugging "
            "attention bugs.\n"
            "\n"
            "Hint: the pattern `'b s (h d) -> b h s d'` does both jobs in one go "
            "when you pass `h=n_heads`."
        ),
        "stub": (
            "def ex6_split_heads(x: Tensor, n_heads: int) -> Tensor:\n"
            "    \"\"\"Reshape (b, s, h*d) -> (b, h, s, d) and print the shape pipeline.\n"
            "\n"
            "    Args:\n"
            "        x: shape (batch, seq_len, n_heads * head_dim)\n"
            "        n_heads: number of attention heads\n"
            "\n"
            "    Returns:\n"
            "        tensor of shape (batch, n_heads, seq_len, head_dim)\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "b, s, h, d = 2, 5, 4, 8\n"
            "x = t.arange(b * s * h * d).reshape(b, s, h * d).float()\n"
            "y = ex6_split_heads(x, n_heads=h)\n"
            "assert y.shape == (b, h, s, d), f'shape mismatch: {y.shape}'\n"
            "# Each (b, h, s, :) slice should contain the d contiguous values that\n"
            "# lived at x[b, s, h*d : (h+1)*d] — i.e. the head split must respect\n"
            "# the original last-axis layout (head 0 is the first d values, etc.).\n"
            "for bi in range(b):\n"
            "    for si in range(s):\n"
            "        for hi in range(h):\n"
            "            expected = x[bi, si, hi * d : (hi + 1) * d]\n"
            "            got = y[bi, hi, si]\n"
            "            assert t.equal(got, expected), (\n"
            "                f'head split mismatch at b={bi} s={si} h={hi}: '\n"
            "                f'{got.tolist()} vs {expected.tolist()}'\n"
            "            )"
        ),
        "solution_body": (
            "def ex6_split_heads(x: Tensor, n_heads: int) -> Tensor:\n"
            "    b, s, hd = x.shape\n"
            "    d = hd // n_heads\n"
            "    print(f'input         : {tuple(x.shape)}  (b, s, h*d)')\n"
            "    print(f'if split only : ({b}, {s}, {n_heads}, {d})  (b, s, h, d)')\n"
            "    print(f'after swap    : ({b}, {n_heads}, {s}, {d})  (b, h, s, d)')\n"
            "    return rearrange(x, 'b s (h d) -> b h s d', h=n_heads)"
        ),
        "solution_notes": (
            "**Why one rearrange instead of two?** `'b s (h d) -> b h s d'` fuses "
            "the *decompose* (`(h d)` on the left, `h d` on the right) with the "
            "*reorder* (the `h` axis moves between `s` and `d`). Doing it in one "
            "call avoids an extra intermediate tensor — and, more importantly, "
            "makes the *intent* legible: any reader sees the head-split AND the "
            "swap in one line.\n"
            "\n"
            "**Why print the pipeline?** Attention bugs often come from heads "
            "and seq_len being swapped, or from `head_dim` being computed wrong. "
            "Calling out the shape at each conceptual step (split → swap) makes "
            "the bug visible without a debugger."
        ),
        "extra_imports": [],
    },
    # ──────────────────────────────────────────────────────────────────
    # ex7 — NHWC ↔ NCHW with side-by-side imshow
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "NHWC ↔ NCHW round-trip with imshow",
        "slug": "nhwc-nchw-roundtrip-imshow",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["visualization", "matplotlib", "image-layout", "axis-swap"],
        "kcs": ["rearrange-axis-swap"],
        "lo": (
            "Convert an image batch between NHWC (TF / matplotlib convention) and "
            "NCHW (PyTorch convention) using rearrange, and visually verify the "
            "round-trip preserves the image."
        ),
        "prompt_body": (
            "PyTorch wants images as `(N, C, H, W)`. matplotlib's `imshow` wants "
            "them as `(H, W, C)`. Frameworks like TensorFlow keep `(N, H, W, C)`. "
            "Converting between them is one of the most common rearrange "
            "applications in real ML code — and getting it wrong silently "
            "corrupts your batch (the loss won't crash, the image just won't "
            "*look* like an image to the model).\n"
            "\n"
            "Implement `ex7_nhwc_to_nchw_and_back(img_nhwc)`:\n"
            "1. Take an image batch in NHWC layout, shape `(N, H, W, 3)`.\n"
            "2. Convert it to NCHW (`(N, 3, H, W)`) — that's `chw`.\n"
            "3. Convert `chw` back to NHWC — that's `roundtrip`.\n"
            "4. Use matplotlib to plot the **first** image of `img_nhwc` and the "
            "first image of `roundtrip` side by side, with titles 'original' "
            "and 'after NHWC→NCHW→NHWC'.\n"
            "5. Return the tuple `(chw, roundtrip)`.\n"
            "\n"
            "If the two imshow panels don't look identical, you've axis-swapped "
            "wrong — that's the visual sanity check this exercise exists for."
        ),
        "stub": (
            "def ex7_nhwc_to_nchw_and_back(img_nhwc: Tensor) -> tuple[Tensor, Tensor]:\n"
            "    \"\"\"Round-trip NHWC -> NCHW -> NHWC, plotting first-image before/after.\n"
            "\n"
            "    Args:\n"
            "        img_nhwc: shape (N, H, W, 3) float tensor in [0, 1].\n"
            "\n"
            "    Returns:\n"
            "        (chw, roundtrip) — chw is (N, 3, H, W); roundtrip is (N, H, W, 3)\n"
            "        and should equal img_nhwc exactly.\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a small batch with a recognizable pattern: a red square on\n"
            "# top-left and a blue gradient elsewhere. If a wrong rearrange shuffles\n"
            "# channels, the imshow panels will visibly differ.\n"
            "N, H, W = 3, 16, 16\n"
            "img = t.zeros(N, H, W, 3)\n"
            "img[..., :8, :8, 0] = 1.0  # red square\n"
            "img[..., :, :, 2] = t.linspace(0, 1, W).unsqueeze(0).expand(H, W)  # blue gradient\n"
            "chw, roundtrip = ex7_nhwc_to_nchw_and_back(img)\n"
            "assert chw.shape == (N, 3, H, W), f'chw shape: {chw.shape}'\n"
            "assert roundtrip.shape == (N, H, W, 3), f'roundtrip shape: {roundtrip.shape}'\n"
            "assert t.equal(roundtrip, img), 'round-trip should be bit-identical to input'\n"
            "# Channel sanity: the red square should land in channel 0 of chw.\n"
            "assert chw[0, 0, :8, :8].mean() > 0.99, 'red channel lost in NHWC->NCHW'\n"
            "assert chw[0, 1, :8, :8].mean() < 0.01, 'green channel got data it should not have'"
        ),
        "solution_body": (
            "def ex7_nhwc_to_nchw_and_back(img_nhwc: Tensor) -> tuple[Tensor, Tensor]:\n"
            "    import matplotlib.pyplot as plt\n"
            "    chw = rearrange(img_nhwc, 'n h w c -> n c h w')\n"
            "    roundtrip = rearrange(chw, 'n c h w -> n h w c')\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(6, 3))\n"
            "    axes[0].imshow(img_nhwc[0].cpu().numpy())\n"
            "    axes[0].set_title('original')\n"
            "    axes[0].axis('off')\n"
            "    axes[1].imshow(roundtrip[0].cpu().numpy())\n"
            "    axes[1].set_title('after NHWC→NCHW→NHWC')\n"
            "    axes[1].axis('off')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    return chw, roundtrip"
        ),
        "solution_notes": (
            "**Why this isn't a `.permute` review.** Naming the axes "
            "(`n h w c -> n c h w`) makes the *meaning* of the transformation "
            "visible — `permute(0, 3, 1, 2)` is the same op but you have to "
            "decode three integers to know what moved where. In a real ML "
            "pipeline with 4–5 axes, that matters.\n"
            "\n"
            "**Why imshow?** A wrong channel swap (e.g. `'n h w c -> n h c w'`) "
            "passes shape asserts but produces nonsense images. The visual "
            "sanity check is faster than chasing a silent accuracy regression."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ──────────────────────────────────────────────────────────────────
    # ex8 — Patch-and-unpatch round-trip with debug print
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "patchify ↔ unpatchify round-trip",
        "slug": "patchify-unpatchify-roundtrip",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["patching", "round-trip", "shape-debug", "ViT"],
        "kcs": ["rearrange-axis-decomposition", "rearrange-axis-composition"],
        "lo": (
            "Compose a ViT-style patchify with its inverse (unpatchify) and "
            "verify the round-trip is bit-identical, printing the shape at "
            "each of the 4 conceptual stages."
        ),
        "prompt_body": (
            "ViT-style patch embedding turns an `(N, C, H, W)` image into a "
            "sequence of flattened patches `(N, num_patches, patch_dim)`. The "
            "*inverse* — unpatchify — is what you need for things like Masked "
            "Autoencoders, segmentation decoders, or visualizing what a patch "
            "embedding actually looks like.\n"
            "\n"
            "Implement two functions, both using only `rearrange`:\n"
            "\n"
            "**`ex8_patchify(img, p)`** — input `(N, C, H, W)`, patch size `p`. "
            "Decompose `H -> (h p)` and `W -> (w p)`, then compose "
            "`h w` into a single `num_patches` axis and `c p p` into a single "
            "`patch_dim` axis. Print the shape at each of the 4 conceptual "
            "stages (input → after decompose → after num_patches compose → "
            "after patch_dim compose). Return the final `(N, h*w, c*p*p)` "
            "tensor.\n"
            "\n"
            "**`ex8_unpatchify(patches, p, h, w)`** — take `(N, h*w, c*p*p)` "
            "and reverse the whole pipeline back to `(N, C, H, W)`. One "
            "`rearrange` call is enough.\n"
            "\n"
            "The test cell composes them and asserts the round-trip is exactly "
            "equal to the original image. If you got the patch order wrong on "
            "either side, the round-trip will scramble the image."
        ),
        "stub": (
            "def ex8_patchify(img: Tensor, p: int) -> Tensor:\n"
            "    \"\"\"(N, C, H, W) -> (N, num_patches, patch_dim) with shape-stage prints.\"\"\"\n"
            "    raise NotImplementedError()\n"
            "\n"
            "\n"
            "def ex8_unpatchify(patches: Tensor, p: int, h: int, w: int) -> Tensor:\n"
            "    \"\"\"(N, h*w, c*p*p) -> (N, C, H, W). Inverse of ex8_patchify.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "N, C, H, W, p = 2, 3, 8, 8, 4\n"
            "img = t.arange(N * C * H * W).reshape(N, C, H, W).float()\n"
            "patches = ex8_patchify(img, p)\n"
            "h, w = H // p, W // p\n"
            "assert patches.shape == (N, h * w, C * p * p), (\n"
            "    f'patchify shape: {patches.shape}, expected {(N, h * w, C * p * p)}'\n"
            ")\n"
            "recon = ex8_unpatchify(patches, p, h, w)\n"
            "assert recon.shape == img.shape, f'unpatchify shape: {recon.shape}'\n"
            "assert t.equal(recon, img), 'round-trip should be bit-identical'\n"
            "# Sanity: a non-rectangular grid should also work.\n"
            "img2 = t.randn(1, 3, 8, 16)\n"
            "patches2 = ex8_patchify(img2, p=4)\n"
            "recon2 = ex8_unpatchify(patches2, p=4, h=2, w=4)\n"
            "assert t.allclose(recon2, img2), 'non-square round-trip failed'"
        ),
        "solution_body": (
            "def ex8_patchify(img: Tensor, p: int) -> Tensor:\n"
            "    N, C, H, W = img.shape\n"
            "    h, w = H // p, W // p\n"
            "    print(f'stage 0 input              : {tuple(img.shape)}  (N, C, H, W)')\n"
            "    print(f'stage 1 after H,W decompose: ({N}, {C}, {h}, {p}, {w}, {p})  (N, C, h, p, w, p)')\n"
            "    print(f'stage 2 num_patches compose: ({N}, {h * w}, {C}, {p}, {p})  (N, h*w, C, p, p)')\n"
            "    print(f'stage 3 patch_dim compose  : ({N}, {h * w}, {C * p * p})  (N, h*w, C*p*p)')\n"
            "    return rearrange(img, 'n c (h p1) (w p2) -> n (h w) (c p1 p2)', p1=p, p2=p)\n"
            "\n"
            "\n"
            "def ex8_unpatchify(patches: Tensor, p: int, h: int, w: int) -> Tensor:\n"
            "    return rearrange(\n"
            "        patches, 'n (h w) (c p1 p2) -> n c (h p1) (w p2)',\n"
            "        h=h, w=w, p1=p, p2=p,\n"
            "    )"
        ),
        "solution_notes": (
            "**Why this is the canonical round-trip test.** In ViT/MAE code, "
            "patchify is everywhere but unpatchify hides bugs that don't "
            "surface in shape — they surface in *content*. The `t.equal(recon, "
            "img)` assertion is the only way to catch e.g. a swapped `p1`/`p2` "
            "binding, or `(h w)` written as `(w h)` on the inverse.\n"
            "\n"
            "**Why the stage prints?** When you debug a real patchify bug, "
            "you'll mentally walk these four stages anyway. Doing it on screen "
            "makes the failure mode obvious: if stage-3 patch_dim isn't "
            "`C*p*p`, your last compose is wrong."
        ),
        "extra_imports": [],
    },
    # ──────────────────────────────────────────────────────────────────
    # ex9 — Edge case: patch_size doesn't divide H/W
    # ──────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "edge case: patch_size that doesn't divide H/W",
        "slug": "edge-case-non-divisible-patch-size",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["edge-case", "error-handling", "padding", "ViT"],
        "kcs": ["rearrange-axis-decomposition"],
        "lo": (
            "Discover by execution that rearrange decomposition requires exact "
            "divisibility, and write a pad-then-patchify helper that handles "
            "non-divisible spatial dimensions cleanly."
        ),
        "prompt_body": (
            "`rearrange` will *raise* on `'n c (h p1) (w p2) -> ...'` if `H` "
            "isn't a multiple of `p1` — there's no silent truncation. In real "
            "ViT-style code with non-square images (segmentation masks, "
            "medical slices, web crops), you usually want to pad up to the "
            "nearest multiple of `p` first.\n"
            "\n"
            "Implement two things:\n"
            "\n"
            "**`ex9_observe_failure(img, p)`** — call `rearrange(img, 'n c (h "
            "p1) (w p2) -> n (h w) (c p1 p2)', p1=p, p2=p)` inside a "
            "`try / except` and return the exception's message as a string "
            "(or `None` if no exception was raised). This makes the failure "
            "mode concrete instead of abstract.\n"
            "\n"
            "**`ex9_safe_patchify(img, p)`** — pad `img` of shape `(N, C, H, "
            "W)` with **zeros** on the bottom and right so the new H and W "
            "are the smallest multiples of `p` that are ≥ the originals, "
            "*then* patchify. Return the tuple `(patches, padded_h, "
            "padded_w)` so the caller knows how to unpatchify back. Use "
            "`torch.nn.functional.pad` or direct tensor assignment for the "
            "pad; use `rearrange` for the patchify.\n"
            "\n"
            "If the input H and W already divide `p`, `safe_patchify` should "
            "skip padding (return the original spatial dims)."
        ),
        "stub": (
            "def ex9_observe_failure(img: Tensor, p: int) -> str | None:\n"
            "    \"\"\"Try to patchify with a non-divisible p; return the error message.\"\"\"\n"
            "    raise NotImplementedError()\n"
            "\n"
            "\n"
            "def ex9_safe_patchify(img: Tensor, p: int) -> tuple[Tensor, int, int]:\n"
            "    \"\"\"Zero-pad to a multiple of p, then patchify. Returns (patches, H_pad, W_pad).\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn.functional as F\n"
            "\n"
            "# 1) Observe that rearrange raises on non-divisible spatial dims.\n"
            "img_bad = t.zeros(1, 3, 10, 10)  # 10 isn't a multiple of 4\n"
            "msg = ex9_observe_failure(img_bad, p=4)\n"
            "assert msg is not None, 'rearrange should raise on non-divisible H,W'\n"
            "assert isinstance(msg, str) and len(msg) > 0, f'expected error message, got {msg!r}'\n"
            "print('observed rearrange failure:', msg[:120])\n"
            "\n"
            "# 2) safe_patchify pads to a multiple of p, then patchifies.\n"
            "patches, H_pad, W_pad = ex9_safe_patchify(img_bad, p=4)\n"
            "assert H_pad == 12 and W_pad == 12, f'expected (12,12) after padding, got ({H_pad},{W_pad})'\n"
            "h, w = H_pad // 4, W_pad // 4\n"
            "assert patches.shape == (1, h * w, 3 * 4 * 4), f'shape: {patches.shape}'\n"
            "\n"
            "# 3) When H,W already divide p, no padding should happen.\n"
            "img_ok = t.randn(2, 3, 8, 8)\n"
            "patches_ok, H_ok, W_ok = ex9_safe_patchify(img_ok, p=4)\n"
            "assert (H_ok, W_ok) == (8, 8), f'no-op pad expected, got ({H_ok},{W_ok})'\n"
            "\n"
            "# 4) The padded region should be zeros — easiest check is that the\n"
            "# patches covering the bottom/right edge sum to less than a non-padded\n"
            "# equivalent. Use a constant-1 image and check the last-row patches.\n"
            "img_ones = t.ones(1, 1, 6, 6)  # pad to 8x8 -> last row+col are zero\n"
            "patches_ones, _, _ = ex9_safe_patchify(img_ones, p=4)\n"
            "# patches_ones is (1, 4, 16). The 4 patches tile the 8x8 grid in row-major:\n"
            "# patch 0 = top-left 4x4 (all ones), patch 1 = top-right (mixed: cols 4..5 ones, 6..7 zero),\n"
            "# patch 2 = bottom-left (mixed), patch 3 = bottom-right (mostly zero).\n"
            "assert patches_ones[0, 0].sum() == 16.0, 'top-left patch should be all ones'\n"
            "assert patches_ones[0, 1].sum() < 16.0, 'top-right patch should include zero-padding'\n"
            "assert patches_ones[0, 3].sum() < patches_ones[0, 0].sum(), 'bottom-right should be padded'"
        ),
        "solution_body": (
            "def ex9_observe_failure(img: Tensor, p: int) -> str | None:\n"
            "    try:\n"
            "        rearrange(img, 'n c (h p1) (w p2) -> n (h w) (c p1 p2)', p1=p, p2=p)\n"
            "        return None\n"
            "    except Exception as e:\n"
            "        return f'{type(e).__name__}: {e}'\n"
            "\n"
            "\n"
            "def ex9_safe_patchify(img: Tensor, p: int) -> tuple[Tensor, int, int]:\n"
            "    import torch.nn.functional as F\n"
            "    N, C, H, W = img.shape\n"
            "    H_pad = ((H + p - 1) // p) * p\n"
            "    W_pad = ((W + p - 1) // p) * p\n"
            "    pad_h = H_pad - H\n"
            "    pad_w = W_pad - W\n"
            "    # F.pad takes pads in reverse-axis order: (left, right, top, bottom, ...)\n"
            "    img_padded = F.pad(img, (0, pad_w, 0, pad_h), value=0.0) if (pad_h or pad_w) else img\n"
            "    patches = rearrange(\n"
            "        img_padded, 'n c (h p1) (w p2) -> n (h w) (c p1 p2)',\n"
            "        p1=p, p2=p,\n"
            "    )\n"
            "    return patches, H_pad, W_pad"
        ),
        "solution_notes": (
            "**Why this matters.** ViT papers always assume `H % p == 0`. Real "
            "datasets don't. The cheapest fix is right/bottom zero padding; "
            "more sophisticated options (reflect padding, learned padding "
            "tokens) exist but this is the engineering baseline. The unpatchify "
            "step would then crop back to `(H, W)` after reconstruction.\n"
            "\n"
            "**Why observe the failure first?** Letting students *see* the "
            "exception message — rather than reading 'rearrange requires "
            "divisibility' in a doc — locks in the failure mode. Next time "
            "they see `EinopsError: ... shape mismatch`, they'll recognize "
            "it as a divisibility issue immediately."
        ),
        "extra_imports": [],
    },
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for spec in SPECS:
        path = emit_standalone(spec)
        print(f"wrote {path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
