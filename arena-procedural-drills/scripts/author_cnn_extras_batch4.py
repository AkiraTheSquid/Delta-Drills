#!/usr/bin/env python3
"""Author 8 standalone Colab drills for CNN-extras + module-utility prereq atoms.

Targets atoms used across ARENA chapter 0 parts 2 (CNN/ResNet) + 4 (backprop,
freeze/detach). Companion to batch-3 (`author_cnn_deep_batch3.py`) — that batch
covered the core conv mechanics (channel-sum, kernel-shape, stride, 2-D
windowing, ConvT, ReLU). This batch covers the *surrounding* modules that
plug into those convs plus the two numpy/PyTorch utility atoms that recur
across the whole chapter.

Atom layout (8 exercises across 8 atoms — single-exercise each):
  batchnorm-affine-params              — gamma * normalized + beta affine step
  avgpool-reduce                       — AvgPool2d == einops.reduce('mean')
  block-group-stack                    — ResNet BlockGroup stacking pattern
  fractional-stride-zero-insertion     — ConvT stride-S inserts (S-1) zeros
  padding-amount-formula-convT         — ConvT `padding` arg → K-1-P effective
  matmul-2d                            — (M,K) @ (K,N) shape rule + @-operator
  freeze-requires-grad                 — transfer-learning freeze pattern
  diagonal-via-strides                 — as_strided trick: stride=(N+1,)

Constraints (per Doughty ACE 2024 + Maier 2021):
  - One LO + one Bloom per exercise.
  - <= 2 KCs per exercise.
  - Solution runs cleanly in backend venv (torch 2.12.0+cpu + einops 0.8.2).
  - Shape/algebra drills stay assertion-only.
  - fractional-stride-zero-insertion gets imshow viz (visually load-bearing).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

# ─────────────────────────────────────────────────────────────────────────────
# Recap snippets — one per atom.
# ─────────────────────────────────────────────────────────────────────────────

RECAP_BN_AFFINE = (
    "## BatchNorm affine params (`gamma * x̂ + beta`) — quick refresher\n"
    "\n"
    "BatchNorm2d's forward has two stages:\n"
    "\n"
    "```\n"
    "x_hat = (x - mean) / sqrt(var + eps)        # normalize: zero-mean unit-var\n"
    "y     = gamma * x_hat + beta                # affine: learnable rescale + shift\n"
    "```\n"
    "\n"
    "**Why the affine step exists.** Pure normalization would force every "
    "channel to mean-0/var-1, which removes the network's ability to learn "
    "channel-specific scale or bias. The affine params restore that capacity: "
    "`gamma` (a.k.a. `weight`) lets each channel rescale; `beta` (a.k.a. "
    "`bias`) lets each channel shift. They're learnable `nn.Parameter`s.\n"
    "\n"
    "**Initialization convention.** `gamma` initialized to **ones**, `beta` to "
    "**zeros**. That way the layer is the **identity** at step 0 — it can't "
    "hurt training and only helps once gradient signal accumulates.\n"
    "\n"
    "**Shape.** For a `(B, C, H, W)` input, both `gamma` and `beta` are "
    "1-D tensors of length `C`. They must be reshaped to `(1, C, 1, 1)` "
    "before the multiply/add so they broadcast across batch and spatial "
    "axes.\n"
    "\n"
    "**Contrast with RMSNorm.** RMSNorm has only the `gamma` scale — no "
    "`beta`. That's the architectural difference at the affine step: "
    "BatchNorm = scale + shift; RMSNorm = scale only."
)

RECAP_AVGPOOL_REDUCE = (
    "## AvgPool2d == einops.reduce(mean) — quick refresher\n"
    "\n"
    "`nn.AvgPool2d(p)` with non-overlapping windows is exactly an einops "
    "`reduce` with the `mean` op:\n"
    "\n"
    "```\n"
    "y = einops.reduce(x, 'b c (h p1) (w p2) -> b c h w', 'mean', p1=p, p2=p)\n"
    "```\n"
    "\n"
    "**Reading the einops string.**\n"
    "- `(h p1)` factors the input H-axis into `h * p1` — `h` is the output "
    "spatial axis, `p1` is the pool-window axis being reduced.\n"
    "- Same trick on width: `(w p2)`.\n"
    "- Right-hand side keeps `b c h w` — the pool axes `p1, p2` are dropped "
    "(that's the reduction).\n"
    "- `'mean'` averages over the dropped axes.\n"
    "\n"
    "**Global avg-pool** is the all-spatial special case — collapse the "
    "ENTIRE `(H, W)` into one scalar per (batch, channel):\n"
    "\n"
    "```\n"
    "y = einops.reduce(x, 'b c h w -> b c', 'mean')\n"
    "```\n"
    "\n"
    "This is exactly what ResNet does between the last conv block and the "
    "classifier — turn `(B, 512, 7, 7)` into `(B, 512)` so a `Linear` can "
    "consume it. Equivalent to `nn.AdaptiveAvgPool2d(1)` followed by "
    "`squeeze(-1).squeeze(-1)`.\n"
    "\n"
    "**Why einops is cleaner.** No need to compute `kernel_size` or `stride` "
    "explicitly — the (h p1) factor pattern declares both at once. And it "
    "trivially extends to 3-D / N-D pooling."
)

RECAP_BLOCK_GROUP = (
    "## ResNet BlockGroup stack — quick refresher\n"
    "\n"
    "A `BlockGroup` is `n_blocks` `ResidualBlock`s wired in series, with the "
    "**first** block doing any downsampling/channel-change and the rest being "
    "identity-shaped:\n"
    "\n"
    "```\n"
    "Sequential(\n"
    "    ResidualBlock(in_feats, out_feats, first_stride=first_stride),   # shape-changing\n"
    "    *[ResidualBlock(out_feats, out_feats, first_stride=1) for _ in range(n_blocks - 1)],\n"
    ")\n"
    "```\n"
    "\n"
    "**Two invariants of the pattern:**\n"
    "1. **Only the first block changes shape.** Stride > 1 or `in_feats != "
    "out_feats` happens exactly once, at the top of the group. After that, "
    "every subsequent block is `out_feats → out_feats, stride=1`.\n"
    "2. **Subsequent blocks get an identity skip.** Because `in_feats == "
    "out_feats` and stride == 1, the residual branch is a no-op — addition "
    "is well-defined without a projection.\n"
    "\n"
    "**Why this matters.** It's the canonical way to compose a deep CNN: "
    "the *group* is the unit of width/resolution change, the *block* is the "
    "unit of additive refinement. ResNet-34 has 4 BlockGroups of "
    "(3, 4, 6, 3) blocks at widths (64, 128, 256, 512); ResNet-50/101/152 "
    "use the same group structure with different block counts.\n"
    "\n"
    "**Stride convention.** The first-block stride is usually 1 for "
    "BlockGroup-0 (no downsample after the stem's MaxPool already cut "
    "resolution) and 2 for BlockGroups 1, 2, 3 — each later group halves "
    "the spatial resolution and doubles channels."
)

RECAP_FRACTIONAL_STRIDE = (
    "## ConvTranspose2d fractional stride (zero insertion) — quick refresher\n"
    "\n"
    "`nn.ConvTranspose2d` with `stride=S` doesn't physically stride the "
    "kernel — it **dilates the input** by inserting `S - 1` rows/cols of "
    "zeros *between every pair of adjacent input pixels*, then does a "
    "regular stride-1 convolution.\n"
    "\n"
    "Example with `S = 2` on a 1-D input of length 4:\n"
    "\n"
    "```\n"
    "input          : [a, b, c, d]\n"
    "after dilation : [a, 0, b, 0, c, 0, d]      # length = (4-1)*2 + 1 = 7\n"
    "```\n"
    "\n"
    "**Why call it 'fractional stride'.** The output advances by ONE pixel "
    "for every `1/S` input pixels — equivalent to a forward conv with "
    "stride `1/S`. That's where the name comes from.\n"
    "\n"
    "**Shape formula** (stride-S, no padding, no output_padding):\n"
    "\n"
    "```\n"
    "H_out = (H_in - 1) * S + K\n"
    "```\n"
    "\n"
    "Compare to a forward stride-S conv `H_out = (H_in - K) // S + 1` — they "
    "are *adjoint*: ConvT(stride=S, K) reverses the spatial shape change of "
    "Conv(stride=S, K).\n"
    "\n"
    "**Why upsampling networks use it.** GANs, U-Nets, autoencoder decoders, "
    "and stable-diffusion's VAE all use stride-2 ConvT layers to double "
    "spatial resolution at each decoder block. The zero-insertion is the "
    "mechanism that makes the output bigger than the input.\n"
    "\n"
    "**Visualization in this drill.** You'll build the zero-inserted "
    "intermediate explicitly and imshow it next to the convolved output, so "
    "you can see the dilation rather than read about it."
)

RECAP_CONVT_PADDING = (
    "## ConvTranspose2d `padding` arg — quick refresher\n"
    "\n"
    "PyTorch's `nn.ConvTranspose2d(..., padding=P)` does **not** add `P` "
    "rows of zero padding to the input. Instead it **removes** `P` rows from "
    "the output, equivalent to using effective padding `K - 1 - P` in the "
    "underlying flipped-padded conv:\n"
    "\n"
    "```\n"
    "effective_pad = K - 1 - P\n"
    "H_out         = (H_in - 1) * S - 2 * P + K\n"
    "```\n"
    "\n"
    "**Why the asymmetric meaning.** ConvT is the adjoint of forward conv. A "
    "forward conv with `padding=P` adds `P` to each side of the input — "
    "increases output size. Its adjoint subtracts that same `P` from the "
    "output of the transposed operation. Same letter, opposite sign at "
    "shape-time.\n"
    "\n"
    "**The gotcha.** Reading `nn.ConvTranspose2d(..., padding=1)` and "
    "expecting 'pad input by 1' is the most-confused PyTorch shape bug. "
    "`padding=1` actually **shrinks** the output by 2 (1 row per side) "
    "relative to `padding=0`.\n"
    "\n"
    "**The intuition.** Pair a forward `Conv2d(stride=2, K=3, padding=1)` "
    "with its inverse `ConvTranspose2d(stride=2, K=3, padding=1, "
    "output_padding=1)` — the shapes round-trip exactly. The matching "
    "`padding` args 'cancel' as adjoints.\n"
    "\n"
    "**Quick check (stride 1).** "
    "`ConvTranspose2d(IC, OC, K=3, padding=0)` on H_in=4 → H_out = 4 + 2 = 6. "
    "Same kernel with `padding=1` → H_out = 4 + 2 - 2 = 4. With `padding=2` "
    "→ H_out = 4 + 2 - 4 = 2. Each unit of `padding` peels one row off each "
    "side of the output."
)

RECAP_MATMUL_2D = (
    "## Numpy matmul 2-D — quick refresher\n"
    "\n"
    "For 2-D tensors, matrix multiplication has the shape rule:\n"
    "\n"
    "```\n"
    "(M, K) @ (K, N) → (M, N)\n"
    "```\n"
    "\n"
    "The **inner** dimensions must match (both equal `K`); they're "
    "contracted (summed) away. The **outer** dimensions become the output "
    "shape (`M` from the left, `N` from the right).\n"
    "\n"
    "**Python `@` operator.** `a @ b` dispatches to `torch.matmul(a, b)` "
    "(or `numpy.matmul`). For pure 2-D inputs this is identical to "
    "`a.matmul(b)`, `t.mm(a, b)`, and `einops.einsum(a, b, 'm k, k n -> m n')`.\n"
    "\n"
    "**Per-element formula.** Each output entry is the dot product of one "
    "row of `a` with one column of `b`:\n"
    "\n"
    "```\n"
    "out[i, j] = sum_k a[i, k] * b[k, j]\n"
    "```\n"
    "\n"
    "**Common mistakes.**\n"
    "- `(M, K) @ (N, K)` shape-errors — the second matrix needs a transpose "
    "first (`a @ b.T` gives `(M, N)`).\n"
    "- Confusing `matmul` with `mul` (elementwise) — `a * b` and `a @ b` do "
    "totally different things; the former needs broadcasting-compatible "
    "shapes, the latter needs the (K, K) inner match.\n"
    "\n"
    "**Higher dims.** For batched inputs, `(..., M, K) @ (..., K, N) → "
    "(..., M, N)` — the leading batch dims broadcast. The 2-D rule is the "
    "base case."
)

RECAP_FREEZE_REQ_GRAD = (
    "## Freeze backbone via `requires_grad = False` — quick refresher\n"
    "\n"
    "Transfer learning: take a pretrained model, *freeze* the backbone so "
    "gradients don't update it, and train only a new task-specific head. "
    "The freeze idiom:\n"
    "\n"
    "```\n"
    "for p in model.parameters():\n"
    "    p.requires_grad = False        # freeze everything\n"
    "model.fc = nn.Linear(in_features, n_classes)   # new head — defaults to requires_grad=True\n"
    "```\n"
    "\n"
    "Or all-at-once via the helper method:\n"
    "\n"
    "```\n"
    "model.requires_grad_(False)        # freeze recursively in-place\n"
    "```\n"
    "\n"
    "**What `requires_grad = False` does.** It tells autograd to skip "
    "computing gradients for that tensor during `backward()`. Forward passes "
    "still flow through the param normally — only the gradient computation "
    "is suppressed.\n"
    "\n"
    "**Critical optimizer step.** Pass ONLY the trainable params to the "
    "optimizer:\n"
    "\n"
    "```\n"
    "trainable = [p for p in model.parameters() if p.requires_grad]\n"
    "opt = t.optim.Adam(trainable, lr=...)\n"
    "```\n"
    "\n"
    "If you forget this and pass `model.parameters()`, Adam will still try "
    "to update frozen params using their (zero) gradients — wastes memory on "
    "moment buffers and produces a subtle warning.\n"
    "\n"
    "**Why new modules unfreeze automatically.** Brand-new `nn.Linear` / "
    "`nn.Conv2d` instances have `requires_grad = True` by default — only "
    "the params you explicitly froze stay frozen. So replacing `model.fc` "
    "after the freeze is the standard pattern: head trainable, backbone "
    "frozen, no extra wiring."
)

RECAP_DIAGONAL_STRIDES = (
    "## Diagonal via `as_strided` — quick refresher\n"
    "\n"
    "Given an `(N, N)` matrix `m` stored row-major (row stride = `N`, col "
    "stride = `1`), its main diagonal `[m[0,0], m[1,1], ..., m[N-1, N-1]]` "
    "lives at offsets `0, N+1, 2(N+1), ...` in memory.\n"
    "\n"
    "```\n"
    "m.as_strided(size=(N,), stride=(N + 1,))\n"
    "```\n"
    "\n"
    "**Why `N + 1`.** To walk one step along the diagonal, you advance "
    "**one row down** (`+N` elements) and **one column right** (`+1` "
    "element). Total stride per diagonal-step = `N + 1`.\n"
    "\n"
    "**Why it's cheaper than `torch.diagonal`.** `torch.diagonal` does the "
    "same thing internally but goes through extra dispatching / checks. "
    "`as_strided` is a one-line storage-header rewrite — no copy, no "
    "allocation, no dispatch.\n"
    "\n"
    "**Off-diagonals.** The same trick generalizes: the `k`-th off-diagonal "
    "starts at offset `k` (positive `k`) or `k * N` (negative `k`) and uses "
    "the same `N + 1` stride. Length shrinks to `N - |k|`.\n"
    "\n"
    "**Trace.** Sum of the diagonal — `m.as_strided(size=(N,), stride=(N+1,))"
    ".sum()` — equals `torch.trace(m)`. Same memory access pattern, no "
    "allocation.\n"
    "\n"
    "**Caveat.** Assumes contiguous row-major storage. If `m` is a transpose "
    "or other view, read `m.stride()` first and use `m.stride(0) + m.stride(1)` "
    "as the diagonal stride."
)


# ─────────────────────────────────────────────────────────────────────────────
# Exercise specs.
# ─────────────────────────────────────────────────────────────────────────────

SPECS = [
    # ═══════════════════════════════════════════════════════════════════════
    # batchnorm-affine-params (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "batchnorm-affine-params",
        "subtopic": "CNN: BatchNorm affine params",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_BN_AFFINE,
        "exercise_index": 1,
        "exercise_title": "apply BatchNorm's affine step to a normalized tensor",
        "slug": "apply-batchnorms-affine-step-to-a-normalized-tensor",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["batchnorm", "affine", "gamma-beta", "per-channel"],
        "kcs": ["bn-affine-formula", "bn-per-channel-broadcast"],
        "lo": (
            "Apply the `y = gamma * x_hat + beta` affine step of BatchNorm2d "
            "across a `(B, C, H, W)` normalized tensor by reshaping per-"
            "channel params `(C,)` to broadcast over batch and spatial axes."
        ),
        "prompt_body": (
            "Implement `ex1_bn_affine(x_hat, gamma, beta)`. Given:\n\n"
            "- `x_hat` of shape `(B, C, H, W)` — already-normalized input "
            "(mean 0, var 1 per channel).\n"
            "- `gamma` of shape `(C,)` — per-channel scale (a.k.a. `weight`).\n"
            "- `beta`  of shape `(C,)` — per-channel shift (a.k.a. `bias`).\n\n"
            "Compute and return the affine output:\n\n"
            "```\n"
            "y = gamma * x_hat + beta\n"
            "```\n\n"
            "**The catch.** `gamma` and `beta` are `(C,)` — to multiply/add "
            "against a `(B, C, H, W)` tensor you must reshape them so the "
            "`C` axis lines up. The canonical reshape is `(1, C, 1, 1)`:\n\n"
            "```\n"
            "gamma.view(1, -1, 1, 1)            # or .reshape(1, C, 1, 1)\n"
            "einops.rearrange(gamma, 'c -> 1 c 1 1')\n"
            "```\n\n"
            "Either form is fine. The point is that the per-channel "
            "parameter must broadcast across B, H, W — `(C,)` would "
            "incorrectly try to broadcast against the *last* axis (`W`).\n\n"
            "**Boundary cases.**\n"
            "- `gamma = ones, beta = zeros` → output equals input (the "
            "identity init that BatchNorm uses at step 0).\n"
            "- `gamma = zeros` → output equals `beta` broadcast everywhere "
            "(channel becomes a constant)."
        ),
        "stub": (
            "def ex1_bn_affine(x_hat: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:\n"
            '    """Apply per-channel BatchNorm affine: y = gamma * x_hat + beta."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# Identity init: gamma=ones, beta=zeros → output must equal input.\n"
            "x = t.randn(2, 4, 3, 3, generator=rng)\n"
            "gamma_id = t.ones(4)\n"
            "beta_id  = t.zeros(4)\n"
            "y_id = ex1_bn_affine(x, gamma_id, beta_id)\n"
            "assert y_id.shape == x.shape, f'shape changed: {tuple(x.shape)} -> {tuple(y_id.shape)}'\n"
            "assert y_id.dtype == x.dtype\n"
            "assert t.allclose(y_id, x, atol=1e-6), 'gamma=1,beta=0 must be identity'\n"
            "\n"
            "# Per-channel scale + shift correctness.\n"
            "x2 = t.randn(2, 3, 4, 4, generator=rng)\n"
            "gamma = t.tensor([2.0, -1.0, 0.5])\n"
            "beta  = t.tensor([0.1, 0.0, -3.0])\n"
            "y2 = ex1_bn_affine(x2, gamma, beta)\n"
            "for c in range(3):\n"
            "    expected = gamma[c] * x2[:, c, :, :] + beta[c]\n"
            "    assert t.allclose(y2[:, c, :, :], expected, atol=1e-6), (\n"
            "        f'channel {c} affine mismatch: got\\n{y2[:, c]}\\nvs\\n{expected}'\n"
            "    )\n"
            "\n"
            "# Zero-gamma collapse: every entry of a channel must equal beta[c].\n"
            "gamma_z = t.zeros(3)\n"
            "beta_z  = t.tensor([7.0, -2.0, 0.5])\n"
            "y3 = ex1_bn_affine(x2, gamma_z, beta_z)\n"
            "for c in range(3):\n"
            "    assert t.allclose(y3[:, c], t.full_like(y3[:, c], beta_z[c].item()), atol=1e-6), (\n"
            "        f'gamma=0 channel {c}: should be uniform {beta_z[c].item()}'\n"
            "    )\n"
            "\n"
            "# Cross-check against torch's nn.functional.batch_norm with identity\n"
            "# running stats — confirms our affine produces the same output.\n"
            "from torch.nn import functional as F\n"
            "B, C, H, W = 4, 5, 6, 6\n"
            "x4 = t.randn(B, C, H, W, generator=rng)\n"
            "# Normalize manually with given running_mean=0 / running_var=1 so\n"
            "# x_hat == x exactly; affine is the only transform left.\n"
            "running_mean = t.zeros(C)\n"
            "running_var  = t.ones(C)\n"
            "gamma4 = t.randn(C, generator=rng)\n"
            "beta4  = t.randn(C, generator=rng)\n"
            "ours = ex1_bn_affine(x4, gamma4, beta4)\n"
            "ref  = F.batch_norm(x4, running_mean, running_var, weight=gamma4, bias=beta4, training=False, eps=0.0)\n"
            "assert t.allclose(ours, ref, atol=1e-5), 'must match F.batch_norm affine output'\n"
            "\n"
            "# Shape sanity for non-square spatial dims (BatchNorm doesn't care).\n"
            "x5 = t.randn(1, 2, 7, 3, generator=rng)\n"
            "y5 = ex1_bn_affine(x5, t.tensor([3.0, 4.0]), t.tensor([0.0, 1.0]))\n"
            "assert y5.shape == (1, 2, 7, 3)"
        ),
        "solution_body": (
            "def ex1_bn_affine(x_hat: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:\n"
            "    g = gamma.view(1, -1, 1, 1)\n"
            "    b = beta.view(1, -1, 1, 1)\n"
            "    return g * x_hat + b"
        ),
        "solution_notes": (
            "**Why `.view(1, -1, 1, 1)`.** Adds three size-1 axes around "
            "the `C` axis so PyTorch's broadcasting matches it against "
            "`x_hat`'s channel position. Without the reshape, `(C,)` would "
            "broadcast against `x_hat`'s last axis (`W`) — silently wrong.\n\n"
            "**Equivalent rewrites.**\n"
            "- `einops.rearrange(gamma, 'c -> 1 c 1 1')` — same result, more "
            "self-documenting.\n"
            "- `gamma[None, :, None, None]` — None-indexing inserts size-1 "
            "axes; works identically but is less readable.\n"
            "- `gamma.reshape(1, -1, 1, 1)` — semantically identical to "
            "`.view` here because `gamma` is already contiguous.\n\n"
            "**Why this is its own atom.** BatchNorm, LayerNorm, GroupNorm, "
            "RMSNorm, and InstanceNorm all share this final affine step. "
            "The DIFFERENCE between these layers is the **normalization** "
            "stage — which axes they reduce over to compute mean/var. The "
            "affine stage is shared. Drilling it in isolation lets you "
            "compose any norm variant by swapping the upstream "
            "normalization while reusing this exact affine code."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # avgpool-reduce (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "avgpool-reduce",
        "subtopic": "CNN: AvgPool as reduce",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_AVGPOOL_REDUCE,
        "exercise_index": 1,
        "exercise_title": "build AvgPool2d via einops.reduce",
        "slug": "build-avgpool2d-via-einops-reduce",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["avgpool", "einops-reduce", "global-pool", "resnet-head"],
        "kcs": ["avgpool-as-reduce-mean", "global-avgpool-collapse"],
        "lo": (
            "Apply `einops.reduce('mean')` with axis-factoring to reproduce "
            "`nn.AvgPool2d` with non-overlapping windows, and verify against "
            "`F.avg_pool2d`."
        ),
        "prompt_body": (
            "Implement `ex1_avgpool_via_reduce(x, p)`. Given input "
            "`x: (B, C, H, W)` and pool size `p` (assume `H` and `W` are "
            "divisible by `p`), return a `(B, C, H // p, W // p)` tensor "
            "whose entries are the **mean** of each non-overlapping "
            "`p × p` window of `x`.\n\n"
            "**Use einops.reduce with axis factoring.** The pattern is:\n\n"
            "```\n"
            "einops.reduce(x, 'b c (h p1) (w p2) -> b c h w', 'mean', p1=p, p2=p)\n"
            "```\n\n"
            "**Read it letter-by-letter.**\n"
            "- `(h p1)` says 'factor the input H axis into `h * p1`' — "
            "`h` will appear on the output as the pooled axis; `p1` is "
            "dropped (reduced).\n"
            "- `'mean'` averages over the dropped axes.\n"
            "- Pass `p1=p, p2=p` so einops knows the factor sizes.\n\n"
            "**Boundary handling.** This drill assumes `H % p == 0` and "
            "`W % p == 0`. Real `nn.AvgPool2d` can pad to handle non-"
            "divisible sizes — out of scope here.\n\n"
            "The test compares your output to `F.avg_pool2d(x, kernel_size=p)` "
            "to fp tolerance."
        ),
        "stub": (
            "def ex1_avgpool_via_reduce(x: Tensor, p: int) -> Tensor:\n"
            '    """Non-overlapping AvgPool2d via einops.reduce mean."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# Small hand-checkable case.\n"
            "x = t.tensor([\n"
            "    [[[1.0, 2.0, 3.0, 4.0],\n"
            "      [5.0, 6.0, 7.0, 8.0],\n"
            "      [9.0, 1.0, 2.0, 3.0],\n"
            "      [4.0, 5.0, 6.0, 7.0]]]\n"
            "])  # shape (1, 1, 4, 4)\n"
            "y = ex1_avgpool_via_reduce(x, p=2)\n"
            "assert y.shape == (1, 1, 2, 2), f'expected (1,1,2,2), got {tuple(y.shape)}'\n"
            "expected = t.tensor([[[\n"
            "    [(1+2+5+6)/4, (3+4+7+8)/4],\n"
            "    [(9+1+4+5)/4, (2+3+6+7)/4],\n"
            "]]])\n"
            "assert t.allclose(y, expected, atol=1e-6), f'value mismatch:\\n{y}\\nvs\\n{expected}'\n"
            "\n"
            "# Cross-check against F.avg_pool2d on random data, various sizes.\n"
            "for B, C, H, W, p in [(2, 3, 8, 8, 2), (1, 4, 16, 16, 4), (3, 2, 12, 6, 2), (1, 1, 32, 32, 8)]:\n"
            "    xr = t.randn(B, C, H, W, generator=rng)\n"
            "    yr = ex1_avgpool_via_reduce(xr, p)\n"
            "    yref = F.avg_pool2d(xr, kernel_size=p)\n"
            "    assert yr.shape == yref.shape\n"
            "    assert t.allclose(yr, yref, atol=1e-5), (\n"
            "        f'mismatch for shape ({B},{C},{H},{W}) p={p}'\n"
            "    )\n"
            "\n"
            "# p == 1 is the identity (each window is a single pixel).\n"
            "x_id = t.randn(1, 2, 4, 4, generator=rng)\n"
            "assert t.allclose(ex1_avgpool_via_reduce(x_id, 1), x_id, atol=1e-7)\n"
            "\n"
            "# Constant input → constant output (mean of constants = constant).\n"
            "x_c = t.full((2, 3, 6, 6), 4.2)\n"
            "y_c = ex1_avgpool_via_reduce(x_c, 3)\n"
            "assert y_c.shape == (2, 3, 2, 2)\n"
            "assert t.allclose(y_c, t.full((2, 3, 2, 2), 4.2), atol=1e-6)"
        ),
        "solution_body": (
            "def ex1_avgpool_via_reduce(x: Tensor, p: int) -> Tensor:\n"
            "    return einops.reduce(\n"
            "        x,\n"
            "        'b c (h p1) (w p2) -> b c h w',\n"
            "        'mean',\n"
            "        p1=p, p2=p,\n"
            "    )"
        ),
        "solution_notes": (
            "**Why `(h p1)` and not `(p1 h)`.** Order inside the parentheses "
            "matters for einops factoring. `(h p1)` means 'rows are grouped "
            "in BLOCKS of size `p1` — `h` is the block index'. This is the "
            "non-overlapping-window semantics that matches AvgPool. `(p1 h)` "
            "would interleave (stride-style) and produce a different "
            "tensor — not what we want.\n\n"
            "**Global avg-pool variant.** Collapse the whole spatial extent: "
            "`einops.reduce(x, 'b c h w -> b c', 'mean')`. This is what "
            "ResNet does between the last BlockGroup and the classifier — "
            "turns `(B, 512, 7, 7)` into `(B, 512)` so `nn.Linear` can "
            "consume it.\n\n"
            "**Equivalence with adaptive pool.** "
            "`nn.AdaptiveAvgPool2d((1, 1))(x).squeeze(-1).squeeze(-1)` does "
            "the global-pool version. Both produce identical output; the "
            "einops form is more transparent about what's happening."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # block-group-stack (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "block-group-stack",
        "subtopic": "CNN: BlockGroup stack",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_BLOCK_GROUP,
        "exercise_index": 1,
        "exercise_title": "build a ResNet BlockGroup from toy blocks",
        "slug": "build-a-resnet-blockgroup-from-toy-blocks",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["resnet", "block-group", "sequential", "first-block-stride"],
        "kcs": ["block-group-first-stride-trick", "block-group-shape-invariant"],
        "lo": (
            "Apply the ResNet BlockGroup construction pattern — first block "
            "takes (in_feats, out_feats, first_stride), subsequent blocks "
            "take (out_feats, out_feats, stride=1) — using a toy `ResBlock` "
            "stand-in inside `nn.Sequential`."
        ),
        "prompt_body": (
            "We're going to drill the BlockGroup *stacking* pattern, "
            "isolated from the (irrelevant for this skill) actual residual "
            "math. A toy `ResBlock` is provided:\n\n"
            "```\n"
            "class ResBlock(nn.Module):\n"
            "    def __init__(self, in_feats, out_feats, first_stride=1):\n"
            "        super().__init__()\n"
            "        self.in_feats     = in_feats\n"
            "        self.out_feats    = out_feats\n"
            "        self.first_stride = first_stride\n"
            "        # toy 1x1 conv that's just a learnable channel projection\n"
            "        self.proj = nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride)\n"
            "    def forward(self, x):\n"
            "        return self.proj(x)\n"
            "```\n\n"
            "(The provided cell defines this; you don't need to write it.)\n\n"
            "Implement `ex1_make_block_group(in_feats, out_feats, n_blocks, first_stride)` "
            "that returns a `nn.Sequential` of `n_blocks` `ResBlock` "
            "instances:\n\n"
            "1. **Block 0** = `ResBlock(in_feats, out_feats, first_stride=first_stride)`.\n"
            "2. **Blocks 1..n_blocks-1** = `ResBlock(out_feats, out_feats, first_stride=1)`.\n\n"
            "Wrap them in a single `nn.Sequential(*blocks)`.\n\n"
            "**Why the asymmetry.** Only the first block changes the "
            "channel count (`in_feats → out_feats`) and applies any "
            "downsampling stride. After that, all subsequent blocks operate "
            "at the new resolution and channel count, with identity-shaped "
            "residual branches. This is the *only* pattern ResNet uses for "
            "block stacking — every CNN-family variant repeats it.\n\n"
            "**Edge case.** When `n_blocks == 1`, the group is just the "
            "single shape-changing block (no extra identity-shaped blocks)."
        ),
        "stub": (
            "import torch.nn as nn\n"
            "\n"
            "class ResBlock(nn.Module):\n"
            "    def __init__(self, in_feats, out_feats, first_stride=1):\n"
            "        super().__init__()\n"
            "        self.in_feats = in_feats\n"
            "        self.out_feats = out_feats\n"
            "        self.first_stride = first_stride\n"
            "        self.proj = nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride)\n"
            "    def forward(self, x):\n"
            "        return self.proj(x)\n"
            "\n"
            "\n"
            "def ex1_make_block_group(in_feats: int, out_feats: int, n_blocks: int, first_stride: int):\n"
            '    """Return a Sequential of n_blocks ResBlocks (first changes shape, rest are identity-shaped)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "\n"
            "# Canonical ResNet-34 stage-2 group: in=64, out=128, n_blocks=4, first_stride=2.\n"
            "group = ex1_make_block_group(in_feats=64, out_feats=128, n_blocks=4, first_stride=2)\n"
            "assert isinstance(group, nn.Sequential), 'must return nn.Sequential'\n"
            "assert len(group) == 4, f'expected 4 blocks, got {len(group)}'\n"
            "\n"
            "# Block 0: in_feats=64, out_feats=128, first_stride=2 (the only shape-changer).\n"
            "b0 = group[0]\n"
            "assert isinstance(b0, ResBlock)\n"
            "assert (b0.in_feats, b0.out_feats, b0.first_stride) == (64, 128, 2), (\n"
            "    f'block 0 wrong: got {(b0.in_feats, b0.out_feats, b0.first_stride)}'\n"
            ")\n"
            "\n"
            "# Blocks 1..3: in_feats == out_feats == 128, first_stride == 1.\n"
            "for i in [1, 2, 3]:\n"
            "    bi = group[i]\n"
            "    assert (bi.in_feats, bi.out_feats, bi.first_stride) == (128, 128, 1), (\n"
            "        f'block {i} wrong: got {(bi.in_feats, bi.out_feats, bi.first_stride)}'\n"
            "    )\n"
            "\n"
            "# Forward smoke test — shape must downsample by first_stride and lift channels.\n"
            "x = t.randn(1, 64, 16, 16)\n"
            "y = group(x)\n"
            "# (1, 64, 16, 16) → block 0 (in=64,out=128,stride=2) → (1, 128, 8, 8)\n"
            "# → 3 more identity-shaped blocks → still (1, 128, 8, 8)\n"
            "assert y.shape == (1, 128, 8, 8), f'expected (1,128,8,8), got {tuple(y.shape)}'\n"
            "\n"
            "# Edge: n_blocks == 1 → group is just the shape-changer.\n"
            "group_one = ex1_make_block_group(32, 64, n_blocks=1, first_stride=2)\n"
            "assert len(group_one) == 1\n"
            "assert (group_one[0].in_feats, group_one[0].out_feats, group_one[0].first_stride) == (32, 64, 2)\n"
            "\n"
            "# Edge: first_stride == 1 + in == out (typical stage-0 group: no downsample).\n"
            "group_stage0 = ex1_make_block_group(64, 64, n_blocks=3, first_stride=1)\n"
            "assert len(group_stage0) == 3\n"
            "x0 = t.randn(1, 64, 56, 56)\n"
            "y0 = group_stage0(x0)\n"
            "assert y0.shape == (1, 64, 56, 56), 'stage-0 group should preserve shape exactly'\n"
            "\n"
            "# Confirm that EVERY block past index 0 has matching in/out (identity-shaped).\n"
            "for n_blocks in [2, 5, 7]:\n"
            "    g = ex1_make_block_group(32, 96, n_blocks=n_blocks, first_stride=2)\n"
            "    assert len(g) == n_blocks\n"
            "    for i in range(1, n_blocks):\n"
            "        assert g[i].in_feats == g[i].out_feats == 96, f'identity invariant broken at block {i}'\n"
            "        assert g[i].first_stride == 1, f'stride invariant broken at block {i}'"
        ),
        "solution_body": (
            "def ex1_make_block_group(in_feats: int, out_feats: int, n_blocks: int, first_stride: int):\n"
            "    import torch.nn as nn\n"
            "    blocks = [ResBlock(in_feats, out_feats, first_stride=first_stride)]\n"
            "    for _ in range(n_blocks - 1):\n"
            "        blocks.append(ResBlock(out_feats, out_feats, first_stride=1))\n"
            "    return nn.Sequential(*blocks)"
        ),
        "solution_notes": (
            "**Why the splat `*blocks`.** `nn.Sequential` takes positional "
            "args for each module, not a list. `Sequential(blocks)` (no "
            "splat) would be wrong — Sequential would store the list as "
            "one element. `Sequential(*blocks)` unpacks so each ResBlock "
            "becomes its own ordered child.\n\n"
            "**Equivalent one-liner.**\n"
            "```\n"
            "return nn.Sequential(\n"
            "    ResBlock(in_feats, out_feats, first_stride),\n"
            "    *[ResBlock(out_feats, out_feats, 1) for _ in range(n_blocks - 1)],\n"
            ")\n"
            "```\n"
            "This is the exact form ARENA's official ResNet code uses.\n\n"
            "**Why first_stride is a kwarg in the real ResBlock.** The "
            "*first* conv inside the residual branch has the stride; the "
            "skip branch has its own (1×1, stride=first_stride) projection "
            "when needed. Lifting the stride to the constructor lets the "
            "block decide where to apply it. Out of scope for this drill — "
            "our toy ResBlock just exposes the constructor signature.\n\n"
            "**Why this composes with `module-composition`.** A ResNet is "
            "`Sequential(stem, *[BlockGroup(...) for _ in range(4)], head)`. "
            "BlockGroup is one level of the recursion — recognising the "
            "stacking pattern at this level makes the full assembly readable."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # fractional-stride-zero-insertion (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "fractional-stride-zero-insertion",
        "subtopic": "CNN: ConvT fractional-stride zero insertion",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_FRACTIONAL_STRIDE,
        "exercise_index": 1,
        "exercise_title": "build the zero-inserted intermediate for stride-2 ConvT",
        "slug": "build-the-zero-inserted-intermediate-for-stride-2-convt",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["ConvTranspose2d", "fractional-stride", "zero-insertion", "upsample", "visualization"],
        "kcs": ["convT-stride-zero-dilation", "convT-stride-shape-formula"],
        "lo": (
            "Analyze stride-`S` ConvTranspose2d by explicitly building the "
            "`(S-1)`-zero-dilated intermediate input and confirming a "
            "stride-1 conv on that intermediate reproduces "
            "`F.conv_transpose2d(stride=S)`."
        ),
        "prompt_body": (
            "Implement `ex1_zero_insert(x, s)`. Given input `x: (B, C, H, W)` "
            "and an integer upsample stride `s >= 1`, return the **zero-"
            "inserted** tensor of shape `(B, C, (H-1)*s + 1, (W-1)*s + 1)` "
            "where:\n\n"
            "- Original pixel `x[b, c, i, j]` lives at position "
            "`[b, c, i*s, j*s]` in the output.\n"
            "- Every other position (between original pixels) is **0**.\n\n"
            "Example for `s = 2`, 1-D input `[a, b, c, d]` (length 4):\n\n"
            "```\n"
            "input   : [a, b, c, d]\n"
            "output  : [a, 0, b, 0, c, 0, d]      # length (4-1)*2 + 1 = 7\n"
            "```\n\n"
            "**Approach.**\n"
            "1. Compute `H_out = (H - 1) * s + 1` and `W_out = (W - 1) * s + 1`.\n"
            "2. Allocate `y = t.zeros(B, C, H_out, W_out, dtype=x.dtype)`.\n"
            "3. Scatter `x` into `y` with step `s`: `y[:, :, ::s, ::s] = x`.\n\n"
            "**Edge case.** `s = 1` → output equals input (no dilation, no "
            "zeros inserted).\n\n"
            "After your implementation, the test runs a stride-1 conv on "
            "your zero-inserted output and compares to the *real* "
            "`F.conv_transpose2d(x, weight, stride=s)`. The two must match "
            "to fp tolerance — this is the equivalence the atom teaches.\n\n"
            "The visualization renders one 8×8 input feature map next to "
            "its zero-inserted 15×15 dilation so you can SEE the (s-1) "
            "zero rows/cols between every pair of original rows/cols."
        ),
        "stub": (
            "def ex1_zero_insert(x: Tensor, s: int) -> Tensor:\n"
            '    """Insert (s-1) rows/cols of zeros between every pair of pixels of x."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "# Hand-checkable 1-D case via a (1, 1, 1, 4) input and s=2.\n"
            "x = t.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1, 1, 1, 4)\n"
            "y = ex1_zero_insert(x, s=2)\n"
            "assert y.shape == (1, 1, 1, 7), f'expected (1,1,1,7), got {tuple(y.shape)}'\n"
            "expected = t.tensor([[[[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]]]])\n"
            "assert t.allclose(y, expected), f'value mismatch:\\n{y}\\nvs\\n{expected}'\n"
            "\n"
            "# 2-D case: s=2, 3x3 input → 5x5 zero-inserted.\n"
            "x2 = t.arange(1.0, 10.0).reshape(1, 1, 3, 3)\n"
            "y2 = ex1_zero_insert(x2, s=2)\n"
            "assert y2.shape == (1, 1, 5, 5)\n"
            "# Original pixels at (0,0), (0,2), (0,4), (2,0), ..., (4,4) — others zero.\n"
            "for i in range(3):\n"
            "    for j in range(3):\n"
            "        assert y2[0, 0, i*2, j*2].item() == x2[0, 0, i, j].item(), (\n"
            "            f'pixel ({i},{j}) not at ({i*2},{j*2}): got {y2[0,0,i*2,j*2].item()} expected {x2[0,0,i,j].item()}'\n"
            "        )\n"
            "# Off-grid positions must be exactly zero.\n"
            "for i in range(5):\n"
            "    for j in range(5):\n"
            "        if i % 2 == 0 and j % 2 == 0:\n"
            "            continue\n"
            "        assert y2[0, 0, i, j].item() == 0.0, f'expected 0 at ({i},{j}), got {y2[0,0,i,j].item()}'\n"
            "\n"
            "# s=1 → identity.\n"
            "x_id = t.randn(2, 3, 4, 4)\n"
            "assert t.allclose(ex1_zero_insert(x_id, s=1), x_id, atol=1e-7)\n"
            "\n"
            "# Equivalence with F.conv_transpose2d: stride-S ConvT == zero-insert(S) → stride-1 conv\n"
            "# with the flipped+swapped kernel. We test that *just the zero-insertion* is correct\n"
            "# by composing it with F.conv2d on a flipped+swapped kernel and comparing to\n"
            "# F.conv_transpose2d directly.\n"
            "rng = t.Generator().manual_seed(7)\n"
            "B, IC, OC = 2, 3, 4\n"
            "for s in [1, 2, 3]:\n"
            "    H, W, K = 5, 5, 3\n"
            "    x_in = t.randn(B, IC, H, W, generator=rng)\n"
            "    w_ct = t.randn(IC, OC, K, K, generator=rng)            # ConvT weight layout (IC, OC, K, K)\n"
            "    y_ref = F.conv_transpose2d(x_in, w_ct, stride=s)\n"
            "    # Reconstruct via: zero-insert → pad by (K-1) → stride-1 conv with flipped/swapped kernel.\n"
            "    x_di = ex1_zero_insert(x_in, s=s)\n"
            "    x_padded = F.pad(x_di, (K - 1, K - 1, K - 1, K - 1))\n"
            "    w_eq = w_ct.flip(-1).flip(-2).transpose(0, 1)           # → (OC, IC, K, K)\n"
            "    y_ours = F.conv2d(x_padded, w_eq)\n"
            "    assert y_ours.shape == y_ref.shape, f's={s}: shape mismatch {tuple(y_ours.shape)} vs {tuple(y_ref.shape)}'\n"
            "    assert t.allclose(y_ours, y_ref, atol=1e-4), (\n"
            "        f's={s}: zero-insert + padded-conv must equal F.conv_transpose2d to fp tol'\n"
            "    )\n"
            "\n"
            "# --- Visualization: see the zeros ---\n"
            "import matplotlib.pyplot as plt\n"
            "vis_in = t.randn(8, 8, generator=t.Generator().manual_seed(2))\n"
            "vis_di = ex1_zero_insert(vis_in[None, None], s=2)[0, 0]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "axes[0].imshow(vis_in.numpy(), cmap='viridis')\n"
            "axes[0].set_title(f'original 8x8 input')\n"
            "axes[1].imshow(vis_di.numpy(), cmap='viridis')\n"
            "axes[1].set_title(f'zero-inserted (s=2): 15x15\\n(every other row/col is 0)')\n"
            "for ax in axes:\n"
            "    ax.set_xticks([])\n"
            "    ax.set_yticks([])\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex1_zero_insert(x: Tensor, s: int) -> Tensor:\n"
            "    B, C, H, W = x.shape\n"
            "    H_out = (H - 1) * s + 1\n"
            "    W_out = (W - 1) * s + 1\n"
            "    y = t.zeros(B, C, H_out, W_out, dtype=x.dtype)\n"
            "    y[:, :, ::s, ::s] = x\n"
            "    return y"
        ),
        "solution_notes": (
            "**Why the shape formula is `(H-1)*s + 1`.** There are `H` "
            "input pixels and `H - 1` *gaps* between adjacent pixels. Each "
            "gap gets filled with `s - 1` zeros — that's `(H-1)*(s-1)` "
            "inserted zeros. Total length = `H + (H-1)*(s-1) = (H-1)*s + 1`. "
            "The same logic gives the width.\n\n"
            "**Why `y[..., ::s, ::s] = x` works.** Python slice `::s` is "
            "'every s-th index starting from 0'. For `s=2` on a length-7 "
            "axis, that's indices `[0, 2, 4, 6]` — exactly 4 positions, "
            "matching the 4 original pixels of a length-4 input. The "
            "remaining positions stay zero (we allocated with `t.zeros`).\n\n"
            "**Why this is the meaning of 'fractional stride'.** The "
            "downstream stride-1 conv now slides over a 2x-bigger input — "
            "so the OUTPUT advances at half the rate of the original "
            "input. That's the 'stride 1/2' interpretation: one output "
            "pixel per ½ input pixel.\n\n"
            "**Equivalence in 3 lines.** Compose this drill with the "
            "`convT-as-flipped-padded-conv` drill from batch-3 and you've "
            "built `F.conv_transpose2d` for any stride from scratch:\n"
            "```\n"
            "x_di     = zero_insert(x, s)\n"
            "x_padded = F.pad(x_di, (K-1,) * 4)\n"
            "w_eq     = w.flip(-1).flip(-2).transpose(0, 1)\n"
            "y        = F.conv2d(x_padded, w_eq)        # == F.conv_transpose2d(x, w, stride=s)\n"
            "```"
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ═══════════════════════════════════════════════════════════════════════
    # padding-amount-formula-convT (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "padding-amount-formula-convT",
        "subtopic": "CNN: ConvT padding amount formula",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_CONVT_PADDING,
        "exercise_index": 1,
        "exercise_title": "predict ConvTranspose2d output size from the padding arg",
        "slug": "predict-convtranspose2d-output-size-from-the-padding-arg",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["ConvTranspose2d", "padding", "output-shape", "adjoint"],
        "kcs": ["convT-padding-subtracts-from-output", "convT-output-shape-formula"],
        "lo": (
            "Apply `H_out = (H_in - 1) * S - 2 * P + K` to predict the spatial "
            "output size of `nn.ConvTranspose2d` for given `(H_in, K, S, P)` "
            "and verify against the real module on multiple parameter "
            "combinations."
        ),
        "prompt_body": (
            "Implement `ex1_convT_outlen(h_in, k, s, p)`. Return the spatial "
            "output length of a `nn.ConvTranspose2d` with kernel size `k`, "
            "stride `s`, padding `p`, and no output_padding, applied to a "
            "1-D input of length `h_in`.\n\n"
            "**Formula.**\n"
            "```\n"
            "h_out = (h_in - 1) * s - 2 * p + k\n"
            "```\n\n"
            "**Reading the formula.**\n"
            "- `(h_in - 1) * s + k` is the no-padding case (matches the "
            "fractional-stride dilation atom's shape).\n"
            "- `- 2 * p` is the asymmetric padding subtraction: each unit "
            "of `padding=P` PEELS one row off EACH side of the output.\n"
            "- So `padding` in ConvT2d is the *opposite* of `padding` in "
            "Conv2d (which ADDS to the output).\n\n"
            "**Mental check.** For `h_in=4, k=3, s=1, p=0` → "
            "`(4-1)*1 - 0 + 3 = 6` (output grows from 4 to 6). For the "
            "same input + kernel with `p=1` → `(4-1)*1 - 2 + 3 = 4` "
            "(output back to 4 — the +2 expansion is exactly cancelled by "
            "the -2 padding shrink).\n\n"
            "The test exercises many `(h_in, k, s, p)` combinations and "
            "compares your prediction against the actual output shape "
            "of `nn.ConvTranspose2d` (stride-1 and stride-2 cases)."
        ),
        "stub": (
            "def ex1_convT_outlen(h_in: int, k: int, s: int, p: int) -> int:\n"
            '    """Output length of nn.ConvTranspose2d for given (h_in, k, s, p)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "# Direct value checks.\n"
            "assert ex1_convT_outlen(h_in=4,  k=3, s=1, p=0) == 6,  '(4-1)*1 - 0 + 3 = 6'\n"
            "assert ex1_convT_outlen(h_in=4,  k=3, s=1, p=1) == 4,  '(4-1)*1 - 2 + 3 = 4 (padding shrinks)'\n"
            "assert ex1_convT_outlen(h_in=4,  k=3, s=1, p=2) == 2,  '(4-1)*1 - 4 + 3 = 2'\n"
            "assert ex1_convT_outlen(h_in=5,  k=3, s=2, p=0) == 11, '(5-1)*2 - 0 + 3 = 11'\n"
            "assert ex1_convT_outlen(h_in=5,  k=3, s=2, p=1) == 9,  '(5-1)*2 - 2 + 3 = 9'\n"
            "assert ex1_convT_outlen(h_in=8,  k=4, s=2, p=1) == 16, '(8-1)*2 - 2 + 4 = 16 (canonical 2x upsample)'\n"
            "assert ex1_convT_outlen(h_in=1,  k=5, s=1, p=0) == 5,  'single-pixel input + 5-tap kernel → 5'\n"
            "\n"
            "# Sign-direction trap: each +1 padding must DECREASE output by 2,\n"
            "# never increase. (This is the most-confused PyTorch shape gotcha.)\n"
            "baseline = ex1_convT_outlen(10, 3, 1, p=0)\n"
            "for p in [1, 2, 3]:\n"
            "    assert ex1_convT_outlen(10, 3, 1, p=p) == baseline - 2 * p, (\n"
            "        f'p={p}: padding must subtract 2*p from output, not add'\n"
            "    )\n"
            "\n"
            "# Cross-check against actual nn.ConvTranspose2d shapes — many combos.\n"
            "cases = [\n"
            "    (4, 3, 1, 0), (4, 3, 1, 1), (5, 3, 2, 1), (8, 4, 2, 1),\n"
            "    (16, 3, 1, 1), (10, 5, 2, 0), (7, 3, 2, 0), (3, 7, 1, 2),\n"
            "]\n"
            "for h_in, k, s, p in cases:\n"
            "    ct = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=k, stride=s, padding=p)\n"
            "    x = t.randn(1, 1, h_in, h_in)\n"
            "    actual = ct(x).shape[-1]\n"
            "    predicted = ex1_convT_outlen(h_in, k, s, p)\n"
            "    assert predicted == actual, (\n"
            "        f'(h={h_in},k={k},s={s},p={p}): predicted {predicted}, actual {actual}'\n"
            "    )\n"
            "\n"
            "# Conv2d ↔ ConvTranspose2d round-trip:\n"
            "# A stride-2 K=3 P=1 Conv2d takes H → ceil(H/2).\n"
            "# Pairing with ConvTranspose2d stride=2 K=3 P=1 (output_padding=1) recovers H exactly\n"
            "# (when H is even). We verify with the formula version: paired padding cancels.\n"
            "h = 16\n"
            "# Forward conv2d: (16 + 2*1 - 3) // 2 + 1 = 8 → halves cleanly.\n"
            "# ConvT inverse without output_padding: ex1_convT_outlen(8, 3, 2, 1) = (8-1)*2 - 2 + 3 = 15.\n"
            "# Off by 1 from h=16 — that's exactly what output_padding=1 corrects (covered elsewhere).\n"
            "assert ex1_convT_outlen(8, 3, 2, 1) == 15, 'paired ConvT lands at H-1 — output_padding closes the gap'"
        ),
        "solution_body": (
            "def ex1_convT_outlen(h_in: int, k: int, s: int, p: int) -> int:\n"
            "    return (h_in - 1) * s - 2 * p + k"
        ),
        "solution_notes": (
            "**The full PyTorch formula** (including `output_padding=OP` "
            "and `dilation=D`):\n"
            "```\n"
            "h_out = (h_in - 1) * s - 2 * p + d * (k - 1) + op + 1\n"
            "```\n"
            "With `D = 1` and `OP = 0` this reduces to our "
            "`(h_in - 1)*s - 2*p + k`. `output_padding` is the extra "
            "knob for closing the (stride-2 ⇒ shape rounds down by 1) "
            "gap when round-tripping with a forward conv.\n\n"
            "**Why `padding` subtracts (the adjoint argument).** "
            "ConvTranspose is the adjoint of Conv2d. A forward Conv2d with "
            "`padding=P` ADDS `P` zeros on each side of the input → the "
            "output grows by `2P`. The adjoint operation must reverse that "
            "shape change — so ConvT's `padding=P` SUBTRACTS `2P` from the "
            "output. Same param name, opposite shape effect, because "
            "they're adjoint.\n\n"
            "**Practical recipe.** When pairing a `Conv2d(s, p)` with its "
            "inverse `ConvTranspose2d(s, p)`, use **identical** stride and "
            "padding values. For stride > 1, also set `output_padding = "
            "stride - 1` to recover the exact input shape — otherwise the "
            "result is off by 1.\n\n"
            "**Why every upsampling network uses `ConvT(K=4, S=2, P=1)`.** "
            "Plug in: `(h_in - 1)*2 - 2 + 4 = 2 * h_in` — clean 2× "
            "upsample with no off-by-one. This is the GAN / U-Net / "
            "diffusion-VAE default."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # matmul-2d (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "matmul-2d",
        "subtopic": "Numpy: matmul 2-D",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_MATMUL_2D,
        "exercise_index": 1,
        "exercise_title": "predict matmul output shape and verify with @",
        "slug": "predict-matmul-output-shape-and-verify-with-at",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["matmul", "at-operator", "shape-rule", "inner-dim-match"],
        "kcs": ["matmul-2d-shape-rule", "at-operator-dispatches-matmul"],
        "lo": (
            "Apply the `(M, K) @ (K, N) -> (M, N)` matmul shape rule by "
            "predicting valid/invalid pairs and dispatching `@` against "
            "concrete tensors."
        ),
        "prompt_body": (
            "Implement TWO functions:\n\n"
            "**1.** `ex1_matmul_outshape(shape_a, shape_b)` — pure shape "
            "logic, no tensors. Inputs are 2-tuples (each shape is "
            "`(rows, cols)`). Return:\n"
            "- The output shape tuple `(M, N)` if the inner dimensions match.\n"
            "- `None` if they don't.\n\n"
            "```\n"
            "ex1_matmul_outshape((3, 4), (4, 5)) -> (3, 5)\n"
            "ex1_matmul_outshape((3, 4), (5, 4)) -> None    # inner dims 4 != 5\n"
            "ex1_matmul_outshape((2, 2), (2, 2)) -> (2, 2)  # square\n"
            "```\n\n"
            "**2.** `ex1_matmul(a, b)` — given two 2-D tensors, return their "
            "matrix product using the `@` operator. (This is a one-liner; "
            "the point is to dispatch via `@`, not via `t.mm` or "
            "`einops.einsum`.)\n\n"
            "**Why two functions in one drill.** The drill targets BOTH "
            "KCs: predicting the output shape WITHOUT running the op (the "
            "interview / debug skill), and dispatching the op via `@` (the "
            "idiomatic Python form). They share the same shape-rule but "
            "exercise it at different levels — one symbolic, one concrete.\n\n"
            "**No higher-dim cases.** This drill is strictly 2-D. Batched "
            "matmul has its own broadcasting rules — out of scope here."
        ),
        "stub": (
            "def ex1_matmul_outshape(shape_a: tuple, shape_b: tuple):\n"
            '    """Return output shape (M, N) or None if inner dims don\'t match."""\n'
            "    raise NotImplementedError()\n"
            "\n"
            "\n"
            "def ex1_matmul(a: Tensor, b: Tensor) -> Tensor:\n"
            '    """Matrix product of two 2-D tensors, via the @ operator."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# --- Shape predictor ---\n"
            "assert ex1_matmul_outshape((3, 4), (4, 5)) == (3, 5)\n"
            "assert ex1_matmul_outshape((1, 10), (10, 1)) == (1, 1), 'row-vec @ col-vec → scalar matrix'\n"
            "assert ex1_matmul_outshape((10, 1), (1, 10)) == (10, 10), 'col-vec @ row-vec → outer product matrix'\n"
            "assert ex1_matmul_outshape((128, 64), (64, 32)) == (128, 32)\n"
            "assert ex1_matmul_outshape((2, 2), (2, 2)) == (2, 2)\n"
            "\n"
            "# Inner-dim mismatch → None.\n"
            "assert ex1_matmul_outshape((3, 4), (5, 4)) is None, 'inner 4 != 5'\n"
            "assert ex1_matmul_outshape((10, 1), (10, 1)) is None, 'inner 1 != 10'\n"
            "assert ex1_matmul_outshape((3, 7), (8, 2)) is None\n"
            "\n"
            "# --- Concrete @ matmul ---\n"
            "rng = t.Generator().manual_seed(0)\n"
            "a = t.tensor([[1.0, 2.0], [3.0, 4.0]])\n"
            "b = t.tensor([[5.0, 6.0], [7.0, 8.0]])\n"
            "y = ex1_matmul(a, b)\n"
            "expected = t.tensor([[1*5+2*7, 1*6+2*8],\n"
            "                     [3*5+4*7, 3*6+4*8]], dtype=t.float32)\n"
            "assert t.allclose(y, expected), f'value mismatch:\\n{y}\\nvs\\n{expected}'\n"
            "\n"
            "# Identity matrix is the multiplicative identity.\n"
            "I3 = t.eye(3)\n"
            "x  = t.randn(3, 5, generator=rng)\n"
            "assert t.allclose(ex1_matmul(I3, x), x, atol=1e-6), 'I @ x must equal x'\n"
            "assert t.allclose(ex1_matmul(x, t.eye(5)), x, atol=1e-6), 'x @ I must equal x'\n"
            "\n"
            "# Cross-check against torch.matmul on random matrices, multiple shapes.\n"
            "for M, K, N in [(3, 4, 5), (10, 1, 7), (1, 128, 1), (64, 64, 64)]:\n"
            "    A = t.randn(M, K, generator=rng)\n"
            "    B = t.randn(K, N, generator=rng)\n"
            "    ours = ex1_matmul(A, B)\n"
            "    ref  = t.matmul(A, B)\n"
            "    assert ours.shape == (M, N) == ref.shape\n"
            "    assert t.allclose(ours, ref, atol=1e-4)\n"
            "\n"
            "# Shape-predictor must agree with reality.\n"
            "for shapes in [((3, 4), (4, 5)), ((10, 1), (1, 10)), ((2, 2), (2, 2))]:\n"
            "    sA, sB = shapes\n"
            "    pred = ex1_matmul_outshape(sA, sB)\n"
            "    A = t.randn(*sA, generator=rng)\n"
            "    B = t.randn(*sB, generator=rng)\n"
            "    real = tuple(ex1_matmul(A, B).shape)\n"
            "    assert pred == real, f'predictor {pred} disagrees with @ result {real}'"
        ),
        "solution_body": (
            "def ex1_matmul_outshape(shape_a: tuple, shape_b: tuple):\n"
            "    m, k1 = shape_a\n"
            "    k2, n = shape_b\n"
            "    if k1 != k2:\n"
            "        return None\n"
            "    return (m, n)\n"
            "\n"
            "\n"
            "def ex1_matmul(a: Tensor, b: Tensor) -> Tensor:\n"
            "    return a @ b"
        ),
        "solution_notes": (
            "**Why `@` is the idiomatic form.** PEP 465 added the `@` "
            "operator specifically for matrix multiplication. It "
            "dispatches via `__matmul__` to `torch.matmul` (for tensors), "
            "`numpy.matmul` (for ndarrays), or any custom implementation "
            "via `__matmul__` / `__rmatmul__`. Equivalents include "
            "`t.mm(a, b)` (2-D only), `t.matmul(a, b)` (any dims), and "
            "`einops.einsum(a, b, 'm k, k n -> m n')`.\n\n"
            "**Why the shape rule has THIS form.** The matmul definition is "
            "`out[i, j] = sum_k a[i, k] * b[k, j]`. The `k` index must "
            "RANGE over the same values on both sides (it's the same sum "
            "index) — so `a.shape[1]` and `b.shape[0]` must be equal. "
            "The free indices `i, j` survive into the output as rows and "
            "columns.\n\n"
            "**Quick mnemonic.** Write the shapes left-to-right with their "
            "inner dimensions touching: `(M, K)(K, N)`. The middle `K`s "
            "cancel like algebra — what's left is `(M, N)`.\n\n"
            "**Why this is its own atom (not subsumed by einsum-contraction).** "
            "Einsum is more general but more verbose. For pure 2-D matmul, "
            "`@` is shorter, faster to read, and what PyTorch's "
            "`nn.Linear` calls under the hood (via `F.linear → addmm`). "
            "Knowing when to reach for `@` vs `einsum` is its own skill."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # freeze-requires-grad (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "freeze-requires-grad",
        "subtopic": "PyTorch: freeze via requires_grad=False",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_FREEZE_REQ_GRAD,
        "exercise_index": 1,
        "exercise_title": "freeze a toy backbone and collect trainable params",
        "slug": "freeze-a-toy-backbone-and-collect-trainable-params",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["transfer-learning", "requires_grad", "freeze", "head-replace"],
        "kcs": ["freeze-requires-grad-false", "collect-trainable-params-for-optimizer"],
        "lo": (
            "Apply the transfer-learning freeze pattern: set "
            "`requires_grad = False` on every backbone param, replace the "
            "head, and collect the (still-trainable) head params for the "
            "optimizer."
        ),
        "prompt_body": (
            "A toy 'pretrained' model is provided:\n\n"
            "```\n"
            "class ToyBackbone(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.Sequential(\n"
            "            nn.Linear(10, 32),\n"
            "            nn.ReLU(),\n"
            "            nn.Linear(32, 16),\n"
            "        )\n"
            "        self.fc = nn.Linear(16, 1000)   # ImageNet-style head\n"
            "    def forward(self, x):\n"
            "        return self.fc(self.encoder(x))\n"
            "```\n\n"
            "Implement `ex1_freeze_and_swap_head(model, n_classes)` that "
            "does the full transfer-learning prep, in this order:\n\n"
            "1. **Freeze everything**: set `p.requires_grad = False` for "
            "every parameter currently in `model`.\n"
            "2. **Replace the head**: set `model.fc = nn.Linear(16, "
            "n_classes)`. Brand-new linear modules default to "
            "`requires_grad=True`, so this re-unfreezes the head "
            "implicitly.\n"
            "3. **Collect trainable params**: build "
            "`trainable_params = [p for p in model.parameters() if "
            "p.requires_grad]`.\n"
            "4. Return a tuple `(model, trainable_params)`.\n\n"
            "**What the test checks.**\n"
            "- All encoder params have `requires_grad == False` after the "
            "call.\n"
            "- The new head has `requires_grad == True` on both weight and "
            "bias.\n"
            "- `trainable_params` contains exactly the new head's "
            "(weight + bias) — 2 tensors, totalling `n_classes * 16 + "
            "n_classes` scalars.\n"
            "- An Adam optimizer built on `trainable_params` runs without "
            "warnings.\n"
            "- A backward pass updates the head params but leaves encoder "
            "params bit-identical."
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
            "        )\n"
            "        self.fc = nn.Linear(16, 1000)\n"
            "    def forward(self, x):\n"
            "        return self.fc(self.encoder(x))\n"
            "\n"
            "\n"
            "def ex1_freeze_and_swap_head(model, n_classes: int):\n"
            '    """Freeze backbone, swap head to (16, n_classes), return (model, trainable_params)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "from torch.nn import functional as F\n"
            "\n"
            "model = ToyBackbone()\n"
            "n_classes = 5\n"
            "\n"
            "# Snapshot encoder params before the call — they must be unchanged after backward.\n"
            "enc_before = [p.detach().clone() for p in model.encoder.parameters()]\n"
            "\n"
            "model, trainable = ex1_freeze_and_swap_head(model, n_classes)\n"
            "\n"
            "# All encoder params must be frozen.\n"
            "for p in model.encoder.parameters():\n"
            "    assert p.requires_grad is False, f'encoder param still trainable: {p.shape}'\n"
            "\n"
            "# The new head must have requires_grad=True on both weight + bias.\n"
            "assert model.fc.weight.requires_grad is True\n"
            "assert model.fc.bias.requires_grad is True\n"
            "# Head shape must be (n_classes, 16).\n"
            "assert model.fc.weight.shape == (n_classes, 16), (\n"
            "    f'fc weight wrong shape: {tuple(model.fc.weight.shape)} expected ({n_classes}, 16)'\n"
            ")\n"
            "assert model.fc.bias.shape == (n_classes,)\n"
            "\n"
            "# trainable params: exactly 2 tensors (fc.weight + fc.bias), totaling 16*n + n scalars.\n"
            "assert len(trainable) == 2, f'expected 2 trainable tensors (fc.weight, fc.bias), got {len(trainable)}'\n"
            "total_trainable = sum(p.numel() for p in trainable)\n"
            "assert total_trainable == 16 * n_classes + n_classes, (\n"
            "    f'trainable scalar count wrong: {total_trainable} vs {16*n_classes + n_classes}'\n"
            ")\n"
            "# Both trainable tensors must be the head params.\n"
            "trainable_ids = {id(p) for p in trainable}\n"
            "assert id(model.fc.weight) in trainable_ids\n"
            "assert id(model.fc.bias)  in trainable_ids\n"
            "\n"
            "# Optimizer construction must succeed.\n"
            "opt = t.optim.Adam(trainable, lr=1e-2)\n"
            "\n"
            "# Backward step: encoder params must remain bit-identical; head params must change.\n"
            "head_w_before = model.fc.weight.detach().clone()\n"
            "x = t.randn(4, 10)\n"
            "y = model(x)\n"
            "loss = F.cross_entropy(y, t.tensor([0, 1, 2, 3]))\n"
            "loss.backward()\n"
            "opt.step()\n"
            "\n"
            "# Encoder params unchanged (frozen).\n"
            "for p_now, p_then in zip(model.encoder.parameters(), enc_before):\n"
            "    assert t.equal(p_now, p_then), 'frozen encoder param mutated by backward+step'\n"
            "    assert p_now.grad is None or t.all(p_now.grad == 0), 'frozen param has nonzero grad'\n"
            "\n"
            "# Head changed.\n"
            "assert not t.equal(model.fc.weight, head_w_before), 'head weight did NOT update after step'"
        ),
        "solution_body": (
            "def ex1_freeze_and_swap_head(model, n_classes: int):\n"
            "    import torch.nn as nn\n"
            "    for p in model.parameters():\n"
            "        p.requires_grad = False\n"
            "    model.fc = nn.Linear(16, n_classes)         # new module defaults to requires_grad=True\n"
            "    trainable = [p for p in model.parameters() if p.requires_grad]\n"
            "    return model, trainable"
        ),
        "solution_notes": (
            "**Why the order matters.** If you replaced the head BEFORE "
            "freezing, the `for p in model.parameters()` loop would also "
            "freeze the brand-new head. Wrong order → no trainable params → "
            "Adam complains. Always freeze first, swap second.\n\n"
            "**Equivalent freezes.**\n"
            "- `model.requires_grad_(False)` — the in-place method, "
            "recursively visits children. Same effect.\n"
            "- `for p in model.encoder.parameters(): p.requires_grad = "
            "False` — finer-grained, freeze only a submodule. ARENA's "
            "transfer-learning exercise uses this form to freeze only "
            "everything *except* `out_layers[-1]`.\n\n"
            "**Why pass `trainable_params` (not `model.parameters()`) to "
            "Adam.** If you pass all params, Adam still allocates moment "
            "buffers for the frozen ones (wastes memory) and you'll see a "
            "`UserWarning` about gradients being None. Filtering up-front "
            "keeps the optimizer state lean.\n\n"
            "**Why `.grad` is None (not zero) for frozen params.** "
            "Autograd never *creates* a `.grad` attribute on a tensor that "
            "doesn't have `requires_grad=True` — the gradient buffer "
            "simply never gets allocated. `.grad is None` is the canonical "
            "post-condition for a frozen param after backward; you do NOT "
            "need to `zero_grad()` them."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # diagonal-via-strides (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "diagonal-via-strides",
        "subtopic": "Numpy: Diagonal via strides",
        "topic_folder": "prereqs_cnn_extras",
        "atom_recap_md": RECAP_DIAGONAL_STRIDES,
        "exercise_index": 1,
        "exercise_title": "extract diagonal of (N, N) via as_strided",
        "slug": "extract-diagonal-of-nn-via-as-strided",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["as_strided", "diagonal", "view", "no-copy"],
        "kcs": ["diagonal-stride-formula", "diagonal-as-strided-view"],
        "lo": (
            "Apply `as_strided(size=(N,), stride=(N+1,))` to extract the "
            "main diagonal of a contiguous `(N, N)` tensor as a view (no "
            "copy) and verify against `torch.diagonal`."
        ),
        "prompt_body": (
            "Implement `ex1_diagonal_via_strides(m)`. Given a 2-D tensor "
            "`m` of shape `(N, N)` that is contiguous and row-major, "
            "return a 1-D length-`N` view that aliases the main diagonal "
            "of `m`. **No copy** — the test confirms with `.data_ptr()`.\n\n"
            "**Use `as_strided`.** For a row-major `(N, N)` tensor, the "
            "row stride is `N` and the column stride is `1`. To walk one "
            "step along the main diagonal you advance one row down (`+N`) "
            "AND one column right (`+1`). So the diagonal-step stride is "
            "`N + 1`:\n\n"
            "```\n"
            "m.as_strided(size=(N,), stride=(N + 1,))\n"
            "```\n\n"
            "Don't use `torch.diagonal(m)`, `m.diag()`, or fancy indexing "
            "(`m[range(N), range(N)]`) — the drill is specifically about "
            "the strided-view trick.\n\n"
            "**The view must alias `m`.** Writing through the returned "
            "tensor mutates the diagonal of `m`. The test verifies this "
            "with both `.data_ptr()` equality and an in-place write check.\n\n"
            "**Boundary.** Assume `m.is_contiguous()` and `m.dim() == 2` "
            "and `m.shape[0] == m.shape[1]`. The test honors those."
        ),
        "stub": (
            "def ex1_diagonal_via_strides(m: Tensor) -> Tensor:\n"
            '    """Return the main diagonal of a contiguous (N, N) tensor as a strided view."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-checkable 4x4 case.\n"
            "m = t.arange(16.0).reshape(4, 4).contiguous()\n"
            "# m =\n"
            "#   [[ 0,  1,  2,  3],\n"
            "#    [ 4,  5,  6,  7],\n"
            "#    [ 8,  9, 10, 11],\n"
            "#    [12, 13, 14, 15]]\n"
            "# Diagonal = [0, 5, 10, 15].\n"
            "diag = ex1_diagonal_via_strides(m)\n"
            "assert diag.shape == (4,), f'expected (4,), got {tuple(diag.shape)}'\n"
            "assert diag.dtype == m.dtype\n"
            "expected = t.tensor([0.0, 5.0, 10.0, 15.0])\n"
            "assert t.allclose(diag, expected), f'diagonal mismatch: {diag.tolist()} vs {expected.tolist()}'\n"
            "\n"
            "# Must be a VIEW — share storage with m.\n"
            "assert diag.data_ptr() == m.data_ptr(), 'must be a view (share storage with m)'\n"
            "\n"
            "# Write-through aliasing: mutate via diag, verify m's diagonal updated.\n"
            "diag[2] = -99.0\n"
            "assert m[2, 2].item() == -99.0, 'view must alias — write to diag[2] should change m[2,2]'\n"
            "# Restore.\n"
            "diag[2] = 10.0\n"
            "\n"
            "# Cross-check against torch.diagonal on multiple sizes.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "for N in [1, 2, 3, 5, 8, 16, 64]:\n"
            "    mk = t.randn(N, N, generator=rng).contiguous()\n"
            "    ours = ex1_diagonal_via_strides(mk)\n"
            "    ref  = t.diagonal(mk)\n"
            "    assert ours.shape == (N,) == ref.shape\n"
            "    assert t.allclose(ours, ref, atol=1e-7), f'N={N}: ours != t.diagonal'\n"
            "    # Trace equivalence: sum(diag) == trace.\n"
            "    assert abs(ours.sum().item() - t.trace(mk).item()) < 1e-4\n"
            "\n"
            "# N=1: trivially the single element.\n"
            "m1 = t.tensor([[42.0]])\n"
            "d1 = ex1_diagonal_via_strides(m1)\n"
            "assert d1.shape == (1,)\n"
            "assert d1.item() == 42.0\n"
            "\n"
            "# Identity matrix → diagonal of all ones.\n"
            "I = t.eye(7)\n"
            "assert t.allclose(ex1_diagonal_via_strides(I), t.ones(7))"
        ),
        "solution_body": (
            "def ex1_diagonal_via_strides(m: Tensor) -> Tensor:\n"
            "    N = m.shape[0]\n"
            "    return m.as_strided(size=(N,), stride=(N + 1,))"
        ),
        "solution_notes": (
            "**Why `N + 1`.** For a row-major `(N, N)` matrix, adjacent "
            "elements within a row are `1` apart in memory; adjacent rows "
            "are `N` apart. A diagonal step moves down AND right "
            "simultaneously — that's `N + 1` elements forward in storage. "
            "Walk `N` such steps starting from offset 0 and you've "
            "visited `[m[0,0], m[1,1], ..., m[N-1, N-1]]`.\n\n"
            "**Why this is more general than `torch.diagonal`.** "
            "`torch.diagonal` is a high-level op that constructs the "
            "diagonal view internally — same end result, but the trick "
            "generalizes. Off-diagonals use the SAME `N + 1` stride with "
            "a different starting offset; `as_strided(..., storage_offset="
            "k)` gives the `k`-th super-diagonal. Anti-diagonal uses "
            "stride `N - 1` from offset `N - 1`. Once you've internalized "
            "the storage math, you can extract ANY contiguous-step pattern.\n\n"
            "**Why `.data_ptr()` equality matters.** It proves no copy "
            "was made. Strided views are O(1) in time and memory; "
            "`torch.diagonal` is also O(1) in modern PyTorch (it returns "
            "a view too), but historically some libraries materialized — "
            "writing the explicit `as_strided` form makes the "
            "no-copy contract obvious.\n\n"
            "**Non-contiguous caveat.** If `m` is a transpose or other "
            "view, `m.stride()` is not `(N, 1)`. The robust version is "
            "`m.as_strided(size=(N,), stride=(m.stride(0) + m.stride(1),))` "
            "— read the actual strides and sum them. Out of scope for "
            "this drill since we restricted to contiguous inputs."
        ),
    },
]


def main() -> None:
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()
