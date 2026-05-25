#!/usr/bin/env python3
"""Author Colab-native standalones (ex6-ex8) for atom `tensor-unbind`.

Each exercise exercises something flashcards cannot deliver: a visualization,
a multi-step debug pipeline, or an integrative ML-adjacent decomposition.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

ATOM_ID = "tensor-unbind"
SUBTOPIC = "Numpy: Indexing and selection"
TOPIC = "prereqs_numpy"

RECAP = (
    "## torch unbind — quick refresher\n"
    "\n"
    "`x.unbind(dim=k)` returns a tuple of `x.shape[k]` view-tensors with axis "
    "`k` removed. The result is a *Python tuple*, not a tensor — perfect for "
    "destructuring named components (`origin, direction = rays.unbind(dim=1)`) "
    "or for fanning a batched tensor into per-head / per-channel slices.\n"
    "\n"
    "**Compared to `select`.** `unbind(dim=k)[i]` ≡ `select(k, i)`. Use `select` "
    "when you want ONE slice; use `unbind` when you want ALL of them. Both "
    "return views (no copy), so writes through the view alias the source."
)


SPECS = [
    # --------------------------------------------------------------- ex6
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "batched ray cast with per-step shape debug",
        "slug": "batched-ray-cast-with-per-step-shape-debug",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["ray-tracing", "shape-debug", "two-level-unbind", "ground-plane"],
        "kcs": ["unbind-explicit-dim", "unbind-tuple-destructure", "unbind-ray-decomposition"],
        "lo": (
            "Apply two levels of unbind (rays → origin/direction → x/y/z) to "
            "solve the analytic ray-plane intersection for a whole batch, "
            "printing shapes at each step."
        ),
        "prompt_body": (
            "Implement `ex6_ray_ground_intersect(rays)`. A full batched ray "
            "cast against the ground plane `y = 0`:\n\n"
            "1. `rays` has shape `(B, 2, 3)` — row 0 of each `(2,3)` block is "
            "the origin, row 1 is the direction.\n"
            "2. First-level unbind: `origin, direction = rays.unbind(dim=1)`. "
            "**Print `origin.shape` and `direction.shape`** with descriptive "
            "labels so the caller sees the decomposition.\n"
            "3. Second-level unbind on `origin` and `direction` along the last "
            "axis to get `ox, oy, oz` and `dx, dy, dz` (each `(B,)`). **Print "
            "`oy.shape` and `dy.shape`**.\n"
            "4. Solve `origin.y + t * direction.y == 0` for `t`:\n"
            "   `t_hit = -oy / dy`. Watch for `dy == 0` (ray parallel to ground "
            "→ produces `inf` or `nan`, which the test tolerates).\n"
            "5. Compute the hit point with the parametric ray equation, return "
            "an `(B, 3)` tensor.\n\n"
            "Output: `(B, 3)` `float32` hit points. For parallel rays the row "
            "contains `inf` or `nan` (don't try to mask them).\n\n"
            "The visualization cell projects the X/Z components onto a 2-D "
            "ground-plane scatter so you can see where each ray landed."
        ),
        "stub": (
            "def ex6_ray_ground_intersect(rays: Tensor) -> Tensor:\n"
            '    """Intersect a (B, 2, 3) batch of rays with the y=0 plane."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rays = t.tensor([\n"
            "    # ray 0 — from (0, 2, 0) pointing straight down → hits (0, 0, 0) at t=2\n"
            "    [[0.0, 2.0, 0.0], [0.0, -1.0, 0.0]],\n"
            "    # ray 1 — from (1, 4, 1) pointing down-and-forward → hits (1, 0, 5) at t=4\n"
            "    [[1.0, 4.0, 1.0], [0.0, -1.0, 1.0]],\n"
            "    # ray 2 — from (-3, 3, 2) pointing down → hits (-3, 0, 2) at t=3\n"
            "    [[-3.0, 3.0, 2.0], [0.0, -1.0, 0.0]],\n"
            "])\n"
            "hits = ex6_ray_ground_intersect(rays)\n"
            "assert hits.shape == (3, 3), f'expected (3,3), got {tuple(hits.shape)}'\n"
            "assert hits.dtype == t.float32, f'expected float32, got {hits.dtype}'\n"
            "expected = t.tensor([\n"
            "    [0.0,  0.0, 0.0],\n"
            "    [1.0,  0.0, 5.0],\n"
            "    [-3.0, 0.0, 2.0],\n"
            "])\n"
            "assert t.allclose(hits, expected, atol=1e-5), f'value mismatch:\\n{hits}\\nvs\\n{expected}'\n"
            "# y-component of every hit must be 0 (we hit the ground plane).\n"
            "assert t.allclose(hits[:, 1], t.zeros(3), atol=1e-5), 'all hits must have y == 0'\n"
            "\n"
            "# Edge case — parallel ray (dy == 0) should produce inf / nan without erroring.\n"
            "parallel = t.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])  # ray traveling along +x\n"
            "p_hit = ex6_ray_ground_intersect(parallel)\n"
            "assert p_hit.shape == (1, 3)\n"
            "assert not t.isfinite(p_hit).all().item(), 'parallel ray should yield inf/nan, got finite'\n"
            "\n"
            "# --- Visualization: scatter hits on the ground plane ---\n"
            "rng = t.Generator().manual_seed(11)\n"
            "B = 100\n"
            "origins = t.stack([\n"
            "    t.linspace(-5, 5, B),\n"
            "    t.full((B,), 4.0),\n"
            "    t.linspace(-3, 3, B),\n"
            "], dim=1)\n"
            "directions = t.stack([\n"
            "    0.3 * t.randn(B, generator=rng),\n"
            "    t.full((B,), -1.0),\n"
            "    0.3 * t.randn(B, generator=rng),\n"
            "], dim=1)\n"
            "big_rays = t.stack([origins, directions], dim=1)  # (B, 2, 3)\n"
            "big_hits = ex6_ray_ground_intersect(big_rays)\n"
            "fig, ax = plt.subplots(figsize=(5, 5))\n"
            "ax.scatter(big_hits[:, 0].numpy(), big_hits[:, 2].numpy(),\n"
            "           c=range(B), cmap='viridis', s=20)\n"
            "ax.set_xlabel('hit X')\n"
            "ax.set_ylabel('hit Z')\n"
            "ax.set_title(f'ex6 ground-plane hits (B={B} rays from y=4 downward)')\n"
            "ax.set_aspect('equal')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex6_ray_ground_intersect(rays: Tensor) -> Tensor:\n"
            "    origin, direction = rays.unbind(dim=1)\n"
            "    print(f'  origin.shape={tuple(origin.shape)}  direction.shape={tuple(direction.shape)}')\n"
            "    ox, oy, oz = origin.unbind(dim=-1)\n"
            "    dx, dy, dz = direction.unbind(dim=-1)\n"
            "    print(f'  oy.shape={tuple(oy.shape)}  dy.shape={tuple(dy.shape)}')\n"
            "    t_hit = -oy / dy\n"
            "    return origin + t_hit.unsqueeze(-1) * direction"
        ),
        "solution_notes": (
            "**Two-level unbind.** The outer `unbind(dim=1)` peels `rays` "
            "`(B,2,3)` into two `(B,3)` named tensors. The inner `unbind(dim=-1)` "
            "peels each into three `(B,)` scalars, ready for elementwise "
            "arithmetic. This is dramatically clearer than `rays[:, 1, 1]` "
            "for the y-component of direction.\n\n"
            "**Why broadcast with `unsqueeze`.** `t_hit` is `(B,)`; `direction` "
            "is `(B,3)`. To multiply them elementwise we need `(B,1) * (B,3)` "
            "so broadcast lines up. `t_hit.unsqueeze(-1)` adds the trailing "
            "size-1 axis.\n\n"
            "**Parallel rays produce `inf`/`nan` — and that's fine.** A real "
            "renderer masks them out with `t.isfinite(t_hit)`. The test only "
            "asserts the divergence happens; downstream code is responsible "
            "for filtering."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # --------------------------------------------------------------- ex7
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "split heads for multi-head attention",
        "slug": "split-heads-for-multi-head-attention",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["attention", "multi-head", "rearrange", "per-head-debug"],
        "kcs": ["unbind-explicit-dim", "unbind-tuple-destructure"],
        "lo": (
            "Apply `unbind(dim=2)` to fan a `(B, S, H, D)` attention tensor "
            "into a list of `(B, S, D)` per-head tensors, transform each, and "
            "restack while printing per-head norms for debug."
        ),
        "prompt_body": (
            "Implement `ex7_apply_per_head(x, scales)`. The canonical "
            "attention-head splitting pattern (without any matmul, so we focus "
            "on the unbind/restack mechanics):\n\n"
            "1. `x` has shape `(B, S, H, D)` — batch, sequence, num_heads, "
            "head_dim.\n"
            "2. Use `x.unbind(dim=2)` to get a length-`H` tuple of `(B, S, D)` "
            "per-head tensors.\n"
            "3. For each head `h`, multiply by `scales[h]` (a scalar) and "
            "**print** `head_idx, scaled.norm()` so the caller can see the "
            "per-head magnitudes.\n"
            "4. Restack with `t.stack(scaled_heads, dim=2)` to recover the "
            "`(B, S, H, D)` shape.\n\n"
            "Inputs:\n"
            "- `x`: `(B, S, H, D)` float tensor.\n"
            "- `scales`: 1-D float tensor of length `H`.\n\n"
            "Output: `(B, S, H, D)` float tensor where head `h` is scaled by "
            "`scales[h]`.\n\n"
            "The visualization renders the per-head L2 norm bar chart from the "
            "real attention-shaped batch used in the smoke test."
        ),
        "stub": (
            "def ex7_apply_per_head(x: Tensor, scales: Tensor) -> Tensor:\n"
            '    """Unbind heads, scale each, restack."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "B, S, H, D = 2, 3, 4, 5\n"
            "x = t.ones(B, S, H, D)\n"
            "scales = t.tensor([1.0, 2.0, 0.5, 0.0])\n"
            "out = ex7_apply_per_head(x, scales)\n"
            "assert out.shape == (B, S, H, D), f'expected {(B,S,H,D)}, got {tuple(out.shape)}'\n"
            "assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "# Head h must equal scales[h] (because input was all ones).\n"
            "for h in range(H):\n"
            "    expected_val = scales[h].item()\n"
            "    actual = out[:, :, h, :]\n"
            "    assert t.allclose(actual, t.full_like(actual, expected_val)), (\n"
            "        f'head {h}: expected all {expected_val}, got\\n{actual}'\n"
            "    )\n"
            "# Head 3 scaled by 0 → must be exactly zero.\n"
            "assert t.all(out[:, :, 3, :] == 0), 'scale=0 head must produce zero output'\n"
            "\n"
            "# Realistic-shape smoke test on random data.\n"
            "rng = t.Generator().manual_seed(3)\n"
            "x_big = t.randn(2, 8, 6, 16, generator=rng)\n"
            "scales_big = t.linspace(0.5, 1.5, 6)\n"
            "out_big = ex7_apply_per_head(x_big, scales_big)\n"
            "assert out_big.shape == (2, 8, 6, 16)\n"
            "# Restacked-head norm == input-head-norm * |scale|.\n"
            "for h in range(6):\n"
            "    in_norm = x_big[:, :, h, :].norm().item()\n"
            "    out_norm = out_big[:, :, h, :].norm().item()\n"
            "    expected = in_norm * abs(scales_big[h].item())\n"
            "    assert abs(out_norm - expected) < 1e-4, (\n"
            "        f'head {h} norm wrong: got {out_norm:.4f}, expected {expected:.4f}'\n"
            "    )\n"
            "\n"
            "# --- Per-head norm bar chart ---\n"
            "per_head_norms = [out_big[:, :, h, :].norm().item() for h in range(6)]\n"
            "fig, ax = plt.subplots(figsize=(6, 3))\n"
            "ax.bar(range(6), per_head_norms, color='coral', edgecolor='black')\n"
            "ax.set_xlabel('head index')\n"
            "ax.set_ylabel('L2 norm of head slice')\n"
            "ax.set_title('ex7 per-head L2 norm after scaling')\n"
            "ax.set_xticks(range(6))\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex7_apply_per_head(x: Tensor, scales: Tensor) -> Tensor:\n"
            "    heads = x.unbind(dim=2)\n"
            "    scaled = []\n"
            "    for h, head in enumerate(heads):\n"
            "        s = head * scales[h]\n"
            "        print(f'  head {h}: norm={s.norm().item():.4f}')\n"
            "        scaled.append(s)\n"
            "    return t.stack(scaled, dim=2)"
        ),
        "solution_notes": (
            "**`unbind` + `stack` is the round-trip identity.** If you do "
            "`t.stack(x.unbind(dim=k), dim=k)`, you get `x` back. This is what "
            "lets per-head transforms compose: peel along the head axis, do "
            "anything you want with the per-head tensors, restack along the "
            "same axis.\n\n"
            "**In real attention,** you wouldn't unbind heads — you'd just "
            "broadcast or use `einsum`. But the unbind/stack pattern is "
            "indispensable when each head needs a DIFFERENT module (e.g. "
            "per-head LoRA adapters, per-head dropout masks, mixture-of-"
            "experts gating).\n\n"
            "**Why print per-head norms.** Dead heads (norm → 0) and "
            "saturating heads (norm → ∞) are the two failure modes of "
            "multi-head models. Logging per-head magnitudes during forward "
            "passes is the first-line diagnostic."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # --------------------------------------------------------------- ex8
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "RGB to grayscale with side-by-side plot",
        "slug": "rgb-to-grayscale-with-side-by-side-plot",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["channels", "grayscale", "luma", "visualization"],
        "kcs": ["unbind-explicit-dim", "unbind-tuple-destructure"],
        "lo": (
            "Apply `unbind(dim=-1)` to destructure an `(H, W, 3)` RGB image "
            "into named R/G/B channels and compute the luma-weighted grayscale "
            "conversion."
        ),
        "prompt_body": (
            "Implement `ex8_rgb_to_grayscale(img)`. The canonical channel-"
            "destructure pattern for image processing:\n\n"
            "1. `img` has shape `(H, W, 3)` — height × width × RGB channels, "
            "values in `[0, 1]`.\n"
            "2. Use `img.unbind(dim=-1)` to get `r, g, b` as three `(H, W)` "
            "tensors.\n"
            "3. Compute the ITU-R BT.601 luma:\n"
            "   `gray = 0.299 * r + 0.587 * g + 0.114 * b`\n"
            "4. Return the `(H, W)` grayscale tensor.\n\n"
            "Input: `(H, W, 3)` float tensor, values in `[0, 1]`.\n"
            "Output: `(H, W)` float tensor, values in `[0, 1]`.\n\n"
            "The visualization renders the original RGB and the grayscale "
            "result side by side so you can verify the conversion looks "
            "reasonable on a synthetic image."
        ),
        "stub": (
            "def ex8_rgb_to_grayscale(img: Tensor) -> Tensor:\n"
            '    """Convert (H, W, 3) RGB to (H, W) grayscale via BT.601 luma."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Solid red, green, blue patches → known luma values.\n"
            "red = t.zeros(2, 2, 3); red[..., 0] = 1.0\n"
            "green = t.zeros(2, 2, 3); green[..., 1] = 1.0\n"
            "blue = t.zeros(2, 2, 3); blue[..., 2] = 1.0\n"
            "assert t.allclose(ex8_rgb_to_grayscale(red),   t.full((2, 2), 0.299), atol=1e-5)\n"
            "assert t.allclose(ex8_rgb_to_grayscale(green), t.full((2, 2), 0.587), atol=1e-5)\n"
            "assert t.allclose(ex8_rgb_to_grayscale(blue),  t.full((2, 2), 0.114), atol=1e-5)\n"
            "# White → 1.0 (coefficients sum to 1).\n"
            "white = t.ones(3, 4, 3)\n"
            "g_white = ex8_rgb_to_grayscale(white)\n"
            "assert g_white.shape == (3, 4), f'expected (3,4), got {tuple(g_white.shape)}'\n"
            "assert t.allclose(g_white, t.ones(3, 4), atol=1e-5), 'white in → white out'\n"
            "# Black → 0.0.\n"
            "black = t.zeros(3, 4, 3)\n"
            "assert t.allclose(ex8_rgb_to_grayscale(black), t.zeros(3, 4), atol=1e-5), 'black in → black out'\n"
            "# Larger image just to validate it runs at scale.\n"
            "big = t.rand(64, 64, 3, generator=t.Generator().manual_seed(0))\n"
            "g_big = ex8_rgb_to_grayscale(big)\n"
            "assert g_big.shape == (64, 64)\n"
            "assert g_big.min().item() >= 0.0 and g_big.max().item() <= 1.0, 'luma must stay in [0, 1]'\n"
            "\n"
            "# --- Side-by-side visualization ---\n"
            "H, W = 64, 96\n"
            "ys = t.linspace(0, 1, H).unsqueeze(1).expand(H, W)\n"
            "xs = t.linspace(0, 1, W).unsqueeze(0).expand(H, W)\n"
            "synth = t.stack([\n"
            "    xs,                  # R increases left→right\n"
            "    ys,                  # G increases top→bottom\n"
            "    1 - 0.5 * (xs + ys), # B falls off diagonally\n"
            "], dim=-1).clamp(0, 1)\n"
            "synth_gray = ex8_rgb_to_grayscale(synth)\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))\n"
            "ax1.imshow(synth.numpy())\n"
            "ax1.set_title('original RGB')\n"
            "ax1.axis('off')\n"
            "ax2.imshow(synth_gray.numpy(), cmap='gray', vmin=0, vmax=1)\n"
            "ax2.set_title('luma grayscale (BT.601)')\n"
            "ax2.axis('off')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex8_rgb_to_grayscale(img: Tensor) -> Tensor:\n"
            "    r, g, b = img.unbind(dim=-1)\n"
            "    return 0.299 * r + 0.587 * g + 0.114 * b"
        ),
        "solution_notes": (
            "**Why luma weights are not equal.** The human eye is much more "
            "sensitive to green than to blue or red. The BT.601 coefficients "
            "(`0.299, 0.587, 0.114`) reflect that perceptual weighting — a "
            "naive `(r + g + b) / 3` produces a darker, washed-out grayscale.\n\n"
            "**Why `unbind(dim=-1)` not `img[..., 0]`.** Both work, but the "
            "destructure reads like math: `r, g, b` are named, the order is "
            "explicit, and there's no chance of accidentally writing `[..., 1]` "
            "when you meant blue. For channel-last image tensors, "
            "`unbind(dim=-1)` is the idiomatic move.\n\n"
            "**Sums to 1 → preserves brightness.** Because `0.299 + 0.587 + "
            "0.114 == 1.0`, white maps to white and black maps to black. If "
            "you weight differently the output will be biased dim or bright."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
