#!/usr/bin/env python3
"""Author 4 new standalone procedural drills for the `einops-repeat` atom.

ex6..ex9 — Colab-only material that flashcards can't deliver:
  ex6: causal-attention mask repeat-and-broadcast + heatmap viz
  ex7: 2D positional encoding tile (1D PE replicated across spatial grid) + viz
  ex8: nearest-neighbor 2x upsample pyramid (8x8 -> 16x16 -> 32x32) + viz
  ex9: grayscale-to-RGB replicate-then-shift, multi-step debug print
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone  # noqa: E402

ATOM_ID = "einops-repeat"
SUBTOPIC = "Einops: Repeat"
TOPIC = "prereqs_einops"

RECAP = (
    "## einops.repeat — quick refresher\n"
    "\n"
    "`repeat(tensor, pattern, **axes_lengths)` introduces new axes or stretches existing ones:\n"
    "1. **New axis** — `'h w -> b h w'` with `b=4` broadcasts across a new leading dim.\n"
    "2. **Stretch (nearest-neighbor)** — `'h w -> (h r) w'` with `r=2` makes each row appear twice in a contiguous block (rows 0,0,1,1,2,2,...).\n"
    "3. **Tile** — `'h w -> h (r w)'` with `r=2` concatenates two full copies side-by-side (cols 0..w-1, then 0..w-1 again).\n"
    "\n"
    "Stretch vs tile: in the composite `(a b)` the axis written **first varies slower**. `(h r)` puts source row 0 at output rows `0..r-1`; `(r h)` puts source row 0 at output rows `0, h, 2h, ...`. The new exercises lean on this distinction repeatedly."
)


SPECS = [
    # ─────────────────────────────────────────────────────────────────────────
    # ex6 — causal attention mask broadcast + viz
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "causal attention mask — broadcast (T,T) → (B,H,T,T) and visualize",
        "slug": "causal-mask-broadcast-and-visualize",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["attention", "mask", "broadcast", "visualization"],
        "kcs": ["repeat-add-axis", "repeat-multi-axis-broadcast"],
        "lo": "Use repeat to lift a 2D causal mask into the (batch, head, query, key) shape Transformer attention expects, and visualize it as a heatmap.",
        "prompt_body": (
            "Implement `ex6_broadcast_causal_mask(mask_2d, b, h)`.\n"
            "\n"
            "Input: `mask_2d` is a `(T, T)` float tensor (0.0 = keep, -inf = block) — a lower-triangular causal mask used inside one attention head. Output: `(b, h, T, T)`, where every (batch, head) slice is the **same** mask broadcast in.\n"
            "\n"
            "Use a **single** `einops.repeat` call with two new named axes. Do **not** use `unsqueeze`, `expand`, `tile`, or `torch.stack`.\n"
            "\n"
            "After your function passes its shape/value asserts, the test cell also plots `mask_2d` as a matplotlib heatmap so you can see the staircase pattern of allowed (white) vs blocked (dark) query→key positions. The plot is for you; the asserts are what grade the answer."
        ),
        "stub": (
            "def ex6_broadcast_causal_mask(mask_2d: Tensor, b: int, h: int) -> Tensor:\n"
            "    \"\"\"Broadcast a (T, T) causal mask up to (b, h, T, T) with a single einops.repeat.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "T = 6\n"
            "# Build a standard causal mask: 0.0 on-and-below diagonal, -inf above.\n"
            "mask_2d = t.zeros(T, T)\n"
            "mask_2d.masked_fill_(t.triu(t.ones(T, T, dtype=t.bool), diagonal=1), float('-inf'))\n"
            "\n"
            "b, h = 2, 4\n"
            "out = ex6_broadcast_causal_mask(mask_2d, b=b, h=h)\n"
            "\n"
            "assert out.shape == (b, h, T, T), f'expected ({b},{h},{T},{T}), got {out.shape}'\n"
            "for bi in range(b):\n"
            "    for hi in range(h):\n"
            "        assert t.equal(out[bi, hi], mask_2d), f'(b={bi},h={hi}) slice differs from mask_2d'\n"
            "# -inf locations preserved (no accidental fill).\n"
            "assert t.isinf(out).sum().item() == b * h * (T * (T - 1) // 2)\n"
            "\n"
            "# Visualize the causal mask: dark = blocked (-inf), bright = allowed (0).\n"
            "fig, ax = plt.subplots(figsize=(3.5, 3.5))\n"
            "viz = mask_2d.clone()\n"
            "viz[t.isinf(viz)] = -1.0  # remap -inf so imshow can render it\n"
            "im = ax.imshow(viz.numpy(), cmap='viridis')\n"
            "ax.set_title(f'Causal mask (T={T})')\n"
            "ax.set_xlabel('key position')\n"
            "ax.set_ylabel('query position')\n"
            "plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex6_broadcast_causal_mask(mask_2d: Tensor, b: int, h: int) -> Tensor:\n"
            "    return repeat(mask_2d, 'q k -> b h q k', b=b, h=h)"
        ),
        "solution_notes": (
            "**Reading the pattern.** Two new named axes (`b`, `h`) are bound by kwarg and inserted to the **left** of `q k`. einops broadcasts the original `(T, T)` block identically into every `(b, h)` slot — no data is copied semantically (it's a stride-0 broadcast under the hood for the PyTorch backend), so this is essentially free.\n"
            "\n"
            "**Why this is Colab-only.** A flashcard can ask \"what pattern broadcasts a 2D mask into 4D?\" but it can't show you the staircase heatmap that makes the causal structure click. Look at the plot: row `i` has bright cells only at columns `0..i` — that's exactly \"query i can attend to keys 0..i\"."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex7 — 2D positional encoding tile + viz
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "2D positional encoding — tile a 1D PE across a spatial grid",
        "slug": "2d-positional-encoding-tile",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["positional-encoding", "tile", "broadcast", "visualization"],
        "kcs": ["repeat-add-axis", "repeat-broadcast-vs-tile"],
        "lo": "Build a 2D positional encoding by repeating a 1D row encoding across the column axis (and a 1D column encoding across the row axis), then sum them.",
        "prompt_body": (
            "Implement `ex7_make_2d_pe(pe_row, pe_col)`.\n"
            "\n"
            "Input: `pe_row` is a `(H, D)` 1D positional encoding indexed by row, `pe_col` is a `(W, D)` 1D positional encoding indexed by column. Output: `(H, W, D)` — the standard \"separable\" 2D PE you sum onto a flattened image patch grid.\n"
            "\n"
            "Construct the output by **repeating** `pe_row` across the new `W` axis and `pe_col` across the new `H` axis, then summing the two `(H, W, D)` tensors. Use one `einops.repeat` per operand — no `unsqueeze`, `expand`, or `broadcast_to`.\n"
            "\n"
            "The test cell visualizes the resulting `(H, W)` map at one feature channel (`d=0`) as a heatmap. You should be able to see vertical stripes from `pe_col` and horizontal stripes from `pe_row` superposed."
        ),
        "stub": (
            "def ex7_make_2d_pe(pe_row: Tensor, pe_col: Tensor) -> Tensor:\n"
            "    \"\"\"Combine a (H, D) row PE and (W, D) col PE into a (H, W, D) 2D PE.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "H, W, D = 8, 12, 16\n"
            "# Simple sinusoidal-ish 1D PEs (we just need something visually distinct).\n"
            "rows = t.linspace(0, 3.14159, H).unsqueeze(1) * (t.arange(D).float() + 1)\n"
            "cols = t.linspace(0, 3.14159, W).unsqueeze(1) * (t.arange(D).float() + 1)\n"
            "pe_row = t.sin(rows)         # (H, D)\n"
            "pe_col = t.cos(cols)         # (W, D)\n"
            "\n"
            "out = ex7_make_2d_pe(pe_row, pe_col)\n"
            "\n"
            "assert out.shape == (H, W, D), f'expected ({H},{W},{D}), got {out.shape}'\n"
            "# Each (h, w, :) should equal pe_row[h] + pe_col[w].\n"
            "for h in [0, H // 2, H - 1]:\n"
            "    for w in [0, W // 2, W - 1]:\n"
            "        expected = pe_row[h] + pe_col[w]\n"
            "        assert t.allclose(out[h, w], expected, atol=1e-6), f'mismatch at ({h},{w})'\n"
            "\n"
            "# Visualize one feature channel as a 2D heatmap.\n"
            "fig, axes = plt.subplots(1, 3, figsize=(10, 3))\n"
            "axes[0].imshow(repeat(pe_row[:, 0], 'h -> h w', w=W).numpy(), cmap='RdBu_r')\n"
            "axes[0].set_title('pe_row[:,0] broadcast')\n"
            "axes[1].imshow(repeat(pe_col[:, 0], 'w -> h w', h=H).numpy(), cmap='RdBu_r')\n"
            "axes[1].set_title('pe_col[:,0] broadcast')\n"
            "axes[2].imshow(out[..., 0].numpy(), cmap='RdBu_r')\n"
            "axes[2].set_title('summed 2D PE (channel 0)')\n"
            "for ax in axes:\n"
            "    ax.set_xlabel('col w'); ax.set_ylabel('row h')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex7_make_2d_pe(pe_row: Tensor, pe_col: Tensor) -> Tensor:\n"
            "    row_grid = repeat(pe_row, 'h d -> h w d', w=pe_col.shape[0])\n"
            "    col_grid = repeat(pe_col, 'w d -> h w d', h=pe_row.shape[0])\n"
            "    return row_grid + col_grid"
        ),
        "solution_notes": (
            "**Why two repeats then a sum.** `pe_row` has no `w` axis; `pe_col` has no `h` axis. You can't add them directly because their shapes don't broadcast (`(H, D)` vs `(W, D)`). Repeating each one into the full `(H, W, D)` grid is the explicit, pattern-driven way to align them — torch's implicit broadcasting would require `unsqueeze`s in the right slots, which is exactly the bookkeeping einops eliminates.\n"
            "\n"
            "**The three-panel plot** shows the two stripe patterns and their sum, which is the standard separable 2D positional encoding used in vision transformers (ViT, MAE)."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex8 — NN-upsample pyramid with viz
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "nearest-neighbor upsample pyramid (8 → 16 → 32) with side-by-side viz",
        "slug": "nn-upsample-pyramid-with-viz",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["upsample", "pyramid", "stretch", "composition", "visualization"],
        "kcs": ["repeat-stretch-vs-tile", "repeat-axis-composition", "repeat-iterated-application"],
        "lo": "Compose two stretches in a single repeat pattern to perform a 2x nearest-neighbor upsample, then apply it iteratively to build an image pyramid.",
        "prompt_body": (
            "Implement `ex8_nn_upsample_pyramid(img, levels)`.\n"
            "\n"
            "Input: `img` is a `(H, W)` 2D tensor. Output: a Python list of length `levels + 1`. Element 0 is the original `img`; element `k` is `img` upsampled by `2**k` using nearest-neighbor (each source pixel becomes a `2x2` block of identical values at level 1, a `4x4` block at level 2, etc.).\n"
            "\n"
            "Build each level by calling `einops.repeat` on the **previous level** with a single pattern that stretches both axes by 2. Do not call `F.interpolate`, `kron`, or write a Python `for`-loop over individual pixels. Loop over levels is fine.\n"
            "\n"
            "The test cell plots all levels side-by-side using matplotlib `imshow` with `interpolation='nearest'` so you can confirm the staircase scaling is exact, not blurred."
        ),
        "stub": (
            "def ex8_nn_upsample_pyramid(img: Tensor, levels: int) -> list[Tensor]:\n"
            "    \"\"\"Iteratively upsample `img` by 2x per level using a single einops.repeat.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Distinctive 8x8 source so nearest-neighbor scaling is obvious in the plot.\n"
            "H = W = 8\n"
            "img = t.arange(H * W).reshape(H, W).float()\n"
            "\n"
            "levels = 2\n"
            "pyramid = ex8_nn_upsample_pyramid(img, levels=levels)\n"
            "\n"
            "assert isinstance(pyramid, list), f'expected list, got {type(pyramid)}'\n"
            "assert len(pyramid) == levels + 1, f'expected {levels + 1} levels, got {len(pyramid)}'\n"
            "assert t.equal(pyramid[0], img), 'level 0 must be the original image'\n"
            "\n"
            "for k in range(1, levels + 1):\n"
            "    factor = 2 ** k\n"
            "    expected_shape = (H * factor, W * factor)\n"
            "    assert pyramid[k].shape == expected_shape, (\n"
            "        f'level {k}: expected {expected_shape}, got {tuple(pyramid[k].shape)}'\n"
            "    )\n"
            "    # Spot-check NN-upsample correctness: every factor×factor block of pyramid[k]\n"
            "    # should equal img[i,j] for the source pixel (i,j) it came from.\n"
            "    for i in [0, H // 2, H - 1]:\n"
            "        for j in [0, W // 2, W - 1]:\n"
            "            block = pyramid[k][i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]\n"
            "            assert t.all(block == img[i, j]), f'level {k} block @ ({i},{j}) not constant'\n"
            "\n"
            "# Side-by-side pyramid visualization.\n"
            "fig, axes = plt.subplots(1, levels + 1, figsize=(3.2 * (levels + 1), 3.2))\n"
            "if levels == 0:\n"
            "    axes = [axes]\n"
            "for k, level_img in enumerate(pyramid):\n"
            "    axes[k].imshow(level_img.numpy(), cmap='viridis', interpolation='nearest')\n"
            "    axes[k].set_title(f'level {k} — {tuple(level_img.shape)}')\n"
            "    axes[k].set_xticks([]); axes[k].set_yticks([])\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex8_nn_upsample_pyramid(img: Tensor, levels: int) -> list[Tensor]:\n"
            "    out = [img]\n"
            "    for _ in range(levels):\n"
            "        prev = out[-1]\n"
            "        upsampled = repeat(prev, 'h w -> (h r1) (w r2)', r1=2, r2=2)\n"
            "        out.append(upsampled)\n"
            "    return out"
        ),
        "solution_notes": (
            "**Pattern recap.** `'h w -> (h r1) (w r2)'` with `r1=r2=2` is the canonical 2x NN-upsample. The composite `(h r1)` puts source row `i` at output rows `2i, 2i+1` — i.e., **stretch** semantics (each row appears as a contiguous 2-block), not **tile** semantics (rows interleaved). If you instead wrote `'h w -> (r1 h) (r2 w)'`, you'd get an image where each source row appears at output rows `i` and `H+i` — which is **not** nearest-neighbor upsample; it's a checkerboard-replica.\n"
            "\n"
            "**Why iterative.** Each level depends on the previous, so the function loop is structural (over levels) rather than over pixels."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex9 — grayscale → RGB replicate-then-shift with debug prints
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "grayscale → RGB replicate-then-shift with debug-print pipeline",
        "slug": "gray-to-rgb-replicate-then-shift-debug",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["channels", "debug-print", "broadcast-trap", "visualization"],
        "kcs": ["repeat-add-axis", "repeat-channel-replicate", "broadcasting-rules"],
        "lo": "Use repeat to replicate a single-channel image across an RGB channel axis, then add a per-channel shift via broadcasting — printing shape/dtype/stride at each step to verify the pipeline.",
        "prompt_body": (
            "Implement `ex9_gray_to_rgb_shifted(gray, shifts)`.\n"
            "\n"
            "Inputs:\n"
            "- `gray`: `(B, 1, H, W)` float image batch (grayscale, with explicit channel-of-1).\n"
            "- `shifts`: `(3,)` float tensor `[r, g, b]` — per-channel additive offsets.\n"
            "\n"
            "Output: `(B, 3, H, W)` float — `gray` replicated across the RGB axis, with `shifts[c]` added to every pixel of channel `c`.\n"
            "\n"
            "Constraints:\n"
            "1. Use **exactly one** `einops.repeat` to go from `(B, 1, H, W)` to `(B, 3, H, W)`. The `'1'` in the input pattern is significant — you're replacing the size-1 channel axis with size 3.\n"
            "2. After the repeat, **print** the result's `.shape`, `.dtype`, and `.stride()` to stdout in this exact format so the test can grep your debug output: `f\"after_repeat shape={tuple(r.shape)} dtype={r.dtype} stride={r.stride()}\"`.\n"
            "3. Add `shifts` to the replicated tensor via plain broadcasting (`+`). Print the same triple for the shifted tensor with prefix `after_shift`.\n"
            "\n"
            "The test cell visualizes channel-0 (red) of the first batch element as a heatmap so you can confirm the shift moved the red baseline as expected."
        ),
        "stub": (
            "def ex9_gray_to_rgb_shifted(gray: Tensor, shifts: Tensor) -> Tensor:\n"
            "    \"\"\"(B, 1, H, W) → (B, 3, H, W) with per-channel additive shift.\n"
            "    Must print `after_repeat ...` and `after_shift ...` debug lines.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import io, contextlib\n"
            "\n"
            "B, H, W = 2, 4, 5\n"
            "gray = t.linspace(0.0, 1.0, B * H * W).reshape(B, 1, H, W)\n"
            "shifts = t.tensor([0.0, 0.25, -0.5])\n"
            "\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            "    out = ex9_gray_to_rgb_shifted(gray, shifts)\n"
            "log = buf.getvalue()\n"
            "print(log, end='')  # forward to the real stdout so the student sees it too\n"
            "\n"
            "assert out.shape == (B, 3, H, W), f'expected ({B},3,{H},{W}), got {out.shape}'\n"
            "assert out.dtype == gray.dtype, f'dtype changed: {out.dtype}'\n"
            "\n"
            "# Correctness: channel c equals gray + shifts[c].\n"
            "for c in range(3):\n"
            "    expected = gray[:, 0] + shifts[c]\n"
            "    assert t.allclose(out[:, c], expected, atol=1e-6), f'channel {c} mismatch'\n"
            "\n"
            "# Debug-print contract: both prefixes must appear with shape/dtype/stride.\n"
            "assert 'after_repeat' in log, f'missing `after_repeat` print:\\n{log}'\n"
            "assert 'after_shift' in log, f'missing `after_shift` print:\\n{log}'\n"
            "for needle in ['shape=', 'dtype=', 'stride=']:\n"
            "    assert log.count(needle) >= 2, f'missing `{needle}` in at least one debug line:\\n{log}'\n"
            "\n"
            "# Visualize channel 0 of batch 0 — should look like `gray` itself (shifts[0]=0).\n"
            "fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))\n"
            "for c, name in enumerate(['R', 'G', 'B']):\n"
            "    im = axes[c].imshow(out[0, c].numpy(), cmap='gray', vmin=-0.6, vmax=1.3)\n"
            "    axes[c].set_title(f'channel {name} (shift={shifts[c].item():+.2f})')\n"
            "    axes[c].set_xticks([]); axes[c].set_yticks([])\n"
            "    plt.colorbar(im, ax=axes[c], fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex9_gray_to_rgb_shifted(gray: Tensor, shifts: Tensor) -> Tensor:\n"
            "    # Step 1: replace size-1 channel axis with size 3 via repeat.\n"
            "    r = repeat(gray, 'b 1 h w -> b c h w', c=3)\n"
            "    print(f\"after_repeat shape={tuple(r.shape)} dtype={r.dtype} stride={r.stride()}\")\n"
            "\n"
            "    # Step 2: broadcast-add the per-channel shifts. shifts is (3,); we need (1, 3, 1, 1).\n"
            "    shifted = r + shifts.view(1, 3, 1, 1)\n"
            "    print(f\"after_shift shape={tuple(shifted.shape)} dtype={shifted.dtype} stride={shifted.stride()}\")\n"
            "    return shifted"
        ),
        "solution_notes": (
            "**The size-1-axis trap.** Writing `'b c h w -> b c2 h w'` with `c=1` in the input pattern is the explicit, readable way to absorb a singleton channel axis. einops accepts the literal `1` in patterns specifically for this. Without it, you'd have to `squeeze` first, then `repeat`, then `unsqueeze` — three steps where one suffices.\n"
            "\n"
            "**Why the debug prints matter.** A stride of `0` in the channel axis after `repeat` would tell you einops returned a broadcast view (cheap); a non-zero stride means it materialized a copy. With the PyTorch backend, `einops.repeat` typically materializes via `expand` + `.contiguous()`-when-needed — the stride print lets you confirm what happened in your environment. This is the kind of multi-step pipeline introspection a flashcard can't deliver."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        # path.parents: [atom-dir, topic-dir, arena-procedural-drills, REPO]
        # We want it printed relative to REPO root.
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
