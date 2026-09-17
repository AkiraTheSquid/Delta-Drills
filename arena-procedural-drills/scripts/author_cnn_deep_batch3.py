#!/usr/bin/env python3
"""Author 8 standalone Colab drills for CNN deep-dive prereq atoms.

Targets ARENA chapter 0 part 2 (CNNs) atoms that batch-2's geometry_cnn drills
deliberately deferred — channel-axis semantics, kernel-layout convention,
stride downsample arithmetic, 2-D windowing, ConvTranspose oddities, and
ReLU placement traps. All under a new `prereqs_cnn_deep` topic folder.

Atom layout (8 exercises across 8 atoms — single-exercise each):
  conv-channel-sum               — IC summed inside einsum (b ic h w, oc ic kh kw -> b oc h w)
  conv-kernel-shape              — (OC, IC, KH, KW) layout introspection
  conv-stride-downsample         — H_out = (H_in - K) // S + 1 for stride > 1
  conv-windowing-2d              — as_strided in BOTH spatial axes
  convT-kernel-axis-swap         — ConvTranspose2d weight (IC, OC, KH, KW)
  convT-as-flipped-padded-conv   — transpose conv == flipped-kernel + padded-input conv
  relu-elementwise-max           — ReLU(x) = max(x, 0)
  no-relu-on-final-layer         — diagnose stray ReLU on classifier head

Constraints (per Doughty ACE 2024 + Maier 2021):
  - One LO + one Bloom per exercise.
  - <= 2 KCs per exercise.
  - Solution runs cleanly in backend venv (torch 2.12.0+cpu).
  - Shape-only drills stay assertion-driven; spatial drills get optional viz.
  - Each drill is a smaller composable skill — no full ARENA-conv re-do.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_cnn_deep"

# ─────────────────────────────────────────────────────────────────────────────
# Recap snippets — one per atom.
# ─────────────────────────────────────────────────────────────────────────────

RECAP_CHANNEL_SUM = (
    "## Conv2d channel-axis sum semantics — quick refresher\n"
    "\n"
    "A 2-D convolution does **three** things at once. The cleanest way to see "
    "this is the einsum form:\n"
    "\n"
    "```\n"
    "y = einops.einsum(\n"
    "    x_windows, weight,\n"
    "    'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow',\n"
    ")\n"
    "```\n"
    "\n"
    "**Read the einsum letter-by-letter:**\n"
    "- `b` (batch) — passes through.\n"
    "- `oc` (output channels) — appears only on the right of the kernel; each "
    "output channel is a separate filter.\n"
    "- `ic` (input channels) — appears on **both** inputs but NOT on the output. "
    "That's a sum: every output pixel is the sum across `IC` of "
    "`window_ic * kernel_ic`. The kernel-output gets one scalar per (oh, ow).\n"
    "- `kh`, `kw` — also contracted (sum) between window and kernel.\n"
    "- `oh`, `ow` — pass through from the windowed input.\n"
    "\n"
    "**The headline.** Convolution is a *per-output-channel* operation that "
    "sums across input channels. With `IC = 3` and `OC = 16`, you have 16 "
    "independent filters, each of which mixes the 3 RGB channels into a "
    "single scalar at every spatial location. The contraction axis is `IC` "
    "(plus the kernel spatial extent)."
)

RECAP_KERNEL_SHAPE = (
    "## Conv kernel shape `(OC, IC, KH, KW)` — quick refresher\n"
    "\n"
    "`nn.Conv2d(in_channels=IC, out_channels=OC, kernel_size=(KH, KW)).weight` "
    "has shape **`(OC, IC, KH, KW)`** — output channels first. This is the "
    "PyTorch convention and matches the einsum role of each axis:\n"
    "\n"
    "```\n"
    "weight[oc, ic, kh, kw]  →  the (kh, kw) tap of input-channel ic\n"
    "                            for output-channel oc\n"
    "```\n"
    "\n"
    "**Why `OC` is first.** A `nn.Conv2d` is a stack of `OC` independent "
    "filters, each shaped `(IC, KH, KW)`. Listing `OC` as the leading axis "
    "makes `weight[i]` directly index the `i`-th filter — convenient for "
    "visualization or per-filter analysis.\n"
    "\n"
    "**Common confusion.** `nn.ConvTranspose2d` flips this to "
    "`(IC, OC, KH, KW)` (see the `convT-kernel-axis-swap` atom). And `nn.Linear`"
    " uses `(out_features, in_features)` — same OC-first convention as Conv2d "
    "but with no spatial axes. Mixing these up at init time produces silently "
    "wrong models; mixing them at forward time produces a shape error.\n"
    "\n"
    "**Parameter count.** Total params (no bias) = `OC * IC * KH * KW`. "
    "Bias adds `OC` more. A 3×3 conv from 64→128 channels is "
    "`128 * 64 * 9 = 73,728` weights — most of a CNN's storage lives here."
)

RECAP_STRIDE_DOWNSAMPLE = (
    "## Conv stride-downsample arithmetic — quick refresher\n"
    "\n"
    "With kernel size `K`, stride `S`, padding `P = 0`, an input of length "
    "`H_in` produces:\n"
    "\n"
    "```\n"
    "H_out = (H_in - K) // S + 1\n"
    "```\n"
    "\n"
    "**Reading the formula.**\n"
    "- The first valid window starts at index 0. That accounts for the `+1`.\n"
    "- Each subsequent window advances by `S` along the input.\n"
    "- The last window must fit entirely — `H_in - K` is the start index of "
    "the *last possible* window; dividing by `S` counts how many strided "
    "positions fit in that span.\n"
    "- Floor division is essential: any partial trailing window is dropped.\n"
    "\n"
    "**The off-by-one trap.** Naively, doubling the stride should halve the "
    "output. But `H_out = H_in // S` is wrong by 1 because of the leading "
    "window. For `H_in = 32, K = 3, S = 2`: actual `H_out = (32-3)//2 + 1 = 15`, "
    "*not* 16. With same-padding `P = 1` it becomes `(32 + 2 - 3)//2 + 1 = 16`, "
    "which IS the clean half — that's why ResNet-style downsampling uses "
    "stride-2 *and* same-padding together.\n"
    "\n"
    "**For `as_strided` windowing.** With stride > 1, the window-index axis "
    "advances by `s_w * stride` instead of `s_w`. Skipping is in the *stride*, "
    "not in a separate indexing step."
)

RECAP_WINDOWING_2D = (
    "## 2-D conv windowing via `as_strided` — quick refresher\n"
    "\n"
    "The 2-D case is the 1-D case applied to **both** spatial axes. Given "
    "`x: (B, IC, H, W)` and kernel `(KH, KW)` at stride 1, you want a view "
    "of shape `(B, IC, OH, OW, KH, KW)` where each "
    "`(KH, KW)` slice along `(OH, OW)` is one kernel-sized window.\n"
    "\n"
    "**The stride tuple.** With input strides `(s_b, s_ic, s_h, s_w)`:\n"
    "\n"
    "```\n"
    "x.as_strided(\n"
    "    size=(B, IC, OH, OW, KH, KW),\n"
    "    stride=(s_b, s_ic, s_h, s_w, s_h, s_w),\n"
    ")\n"
    "```\n"
    "\n"
    "The trailing pair `(s_h, s_w)` walks *within* a window. The middle pair "
    "`(s_h, s_w)` walks *between* windows. **Same strides, different "
    "semantics** — this is the subtle teaching point of ARENA's 2-D conv: the "
    "OUTPUT spatial dims share strides with the KERNEL spatial dims, because "
    "both index into the same input rows/columns.\n"
    "\n"
    "**Equivalence.** Contract via "
    "`einops.einsum(x_windows, weight, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')` "
    "and the result equals `F.conv2d(x, weight)` to fp tolerance.\n"
    "\n"
    "**No data is copied** — `as_strided` only constructs a view header."
)

RECAP_CONVT_AXIS_SWAP = (
    "## ConvTranspose2d kernel axis swap — quick refresher\n"
    "\n"
    "`nn.ConvTranspose2d(in_channels=IC, out_channels=OC, kernel_size=(KH, KW)).weight` "
    "has shape **`(IC, OC, KH, KW)`** — input channels first, output channels "
    "second. This is the **opposite** of `nn.Conv2d`, which uses "
    "`(OC, IC, KH, KW)`.\n"
    "\n"
    "**Why the swap.** ConvTranspose is conv's *adjoint*. Whereas a forward "
    "conv contracts input channels into output channels (one filter per OC), "
    "the adjoint contracts output channels back into input channels — so the "
    "natural axis order flips. PyTorch's storage layout follows that "
    "mathematical role: the first axis indexes the *contraction-input* of the "
    "operation.\n"
    "\n"
    "**The trap.** Building a ConvTranspose by copying a Conv2d's `(OC, IC, "
    "KH, KW)` weight tensor in won't shape-check the way you expect. You "
    "either need to `.transpose(0, 1)` the weight or you'll get a "
    "`size mismatch` error.\n"
    "\n"
    "**Quick check.** `nn.Conv2d(3, 16, 5).weight.shape == (16, 3, 5, 5)`; "
    "`nn.ConvTranspose2d(3, 16, 5).weight.shape == (3, 16, 5, 5)`. Memorize "
    "the asymmetry — it's a common interview question for the same reason."
)

RECAP_CONVT_AS_FLIPPED_CONV = (
    "## ConvTranspose as flipped-kernel padded-conv — quick refresher\n"
    "\n"
    "`nn.ConvTranspose2d` looks unfamiliar but is just a regular convolution "
    "in disguise. For stride 1, no output_padding, kernel size `K`:\n"
    "\n"
    "```\n"
    "ConvTranspose2d(x, w)  ==  Conv2d(\n"
    "    F.pad(x, (K-1, K-1, K-1, K-1)),     # add K-1 zero rows/cols on every side\n"
    "    w.flip(-1).flip(-2).transpose(0, 1) # kernel-flip + axis-swap\n"
    ")\n"
    "```\n"
    "\n"
    "**The three transforms:**\n"
    "1. **Pad input by `K-1`** on every spatial side. This is why transpose-conv "
    "*expands* spatial dims — the output is `H_in + K - 1` (stride 1 case).\n"
    "2. **Flip the kernel** along both spatial axes (`flip(-1)` then `flip(-2)`). "
    "This is the adjoint of cross-correlation.\n"
    "3. **Swap channel axes** of the kernel (`transpose(0, 1)`) so its layout "
    "matches what a regular conv expects.\n"
    "\n"
    "**Why this matters.** Many CNN papers describe \"upsampling conv\" or "
    "\"deconvolution\"; once you internalize the flipped-padded equivalence, "
    "all of the mystery dissolves and you can reason about its output shape "
    "with the same `(H + 2P - K) // S + 1` formula you already know.\n"
    "\n"
    "(For stride > 1 there's an additional step of inserting `S - 1` zeros "
    "between each input pixel before padding — out of scope for this drill, "
    "we cover the stride-1 case.)"
)

RECAP_RELU_MAX = (
    "## ReLU as elementwise max — quick refresher\n"
    "\n"
    "`ReLU(x) = max(x, 0)` — applied **elementwise** to every entry of the "
    "input tensor. In PyTorch:\n"
    "\n"
    "```\n"
    "y = t.maximum(x, t.tensor(0.0))      # the canonical form\n"
    "y = x.clamp(min=0)                   # equivalent\n"
    "y = F.relu(x)                        # library form\n"
    "```\n"
    "\n"
    "**Read the math:** Negative entries become 0; non-negative entries pass "
    "through unchanged. The function is piecewise-linear with a kink at "
    "`x = 0`.\n"
    "\n"
    "**Derivative.** `dReLU/dx = 1 if x > 0 else 0`. At `x = 0` the derivative "
    "is undefined — there's a **sub-gradient** anywhere in `[0, 1]` and the "
    "convention depends on which PyTorch op you use:\n"
    "\n"
    "- `t.maximum(x, t.tensor(0.0))` → grad **0.5** at `x = 0` (the symmetric "
    "average — `maximum` distributes equally on ties).\n"
    "- `F.relu(x)` → grad **0** at `x = 0` (the smallest sub-gradient, "
    "PyTorch's `relu` convention).\n"
    "\n"
    "Both are valid sub-gradients; both work for training. The asymmetry "
    "matters only at the measure-zero point `x = 0`.\n"
    "\n"
    "**Why the simple max.** ReLU is cheap (one comparison, one branch), it "
    "doesn't saturate for large positive inputs (unlike sigmoid/tanh), and "
    "its gradient is exactly 1 in the active region — three properties that "
    "make it the default activation for hidden layers since 2010."
)

RECAP_NO_RELU_FINAL = (
    "## No ReLU on the final classifier layer — quick refresher\n"
    "\n"
    "A classification CNN ends with a linear layer that produces **logits**: "
    "real numbers per class. The next step is `F.cross_entropy`, which "
    "internally applies `LogSoftmax` to the logits before computing the loss.\n"
    "\n"
    "**Why no ReLU on the final layer.** ReLU clips negatives to 0. Once "
    "applied to logits:\n"
    "\n"
    "- All previously negative logits become 0.\n"
    "- `softmax([0, 0, ..., 0, big_positive]) → [near_uniform, ..., dominated_by_big]`.\n"
    "- Worse: if **every** logit is negative pre-ReLU, ReLU produces an "
    "all-zero vector → `softmax([0, ..., 0])` is the uniform distribution → "
    "every prediction is `1 / num_classes` → no learning signal differentiated "
    "across classes.\n"
    "\n"
    "**Diagnosing it.** Symptoms of a stray final-layer ReLU:\n"
    "- Training accuracy plateaus at `1 / num_classes`.\n"
    "- The model's logits are non-negative *everywhere* (a histogram never "
    "shows negatives).\n"
    "- Loss never drops below `log(num_classes)`.\n"
    "\n"
    "**The fix.** Strip the final ReLU. The pattern is "
    "`Linear → ReLU → ... → Linear → ReLU → Linear` (no activation on the "
    "very last linear). For *binary* classification with `BCEWithLogitsLoss` "
    "the same rule applies: feed raw logits in, never ReLU'd."
)


SPECS = [
    # ═══════════════════════════════════════════════════════════════════════
    # conv-channel-sum (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-channel-sum",
        "subtopic": "CNN: Channel-axis sum semantics",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_CHANNEL_SUM,
        "exercise_index": 1,
        "exercise_title": "verify conv2d contracts the IC axis",
        "slug": "verify-conv2d-contracts-the-ic-axis",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["conv2d", "einsum", "channel-sum", "contraction"],
        "kcs": ["conv-ic-contraction", "conv-per-oc-filter"],
        "lo": (
            "Analyze the per-output-channel sum semantics of conv2d by "
            "decomposing `F.conv2d` into a sum of `IC` single-channel "
            "convolutions and verifying numerical equivalence."
        ),
        "prompt_body": (
            "Implement `ex1_conv2d_by_ic_sum(x, weight)`. Given input "
            "`x: (B, IC, H, W)` and kernel `weight: (OC, IC, KH, KW)`, compute "
            "the same result as `F.conv2d(x, weight)` (stride 1, no padding) "
            "but **only by looping over `IC` and summing single-channel "
            "convolutions**.\n\n"
            "**Algorithm:**\n"
            "1. Allocate an output tensor `y` of the right shape "
            "(`(B, OC, H-KH+1, W-KW+1)`).\n"
            "2. For each `ic` in `range(IC)`:\n"
            "   - Slice `x[:, ic:ic+1, :, :]` — one input channel kept as a "
            "size-1 axis: `(B, 1, H, W)`.\n"
            "   - Slice `weight[:, ic:ic+1, :, :]` — one input-channel "
            "kernel tap kept as a size-1 axis: `(OC, 1, KH, KW)`.\n"
            "   - Run `F.conv2d` on this single-channel pair and **add** the "
            "result into `y`.\n"
            "3. Return `y`.\n\n"
            "**The point of the drill.** Decomposing the IC contraction "
            "manually makes it obvious that conv2d is *per-OC, summed across "
            "IC* — the IC axis is contracted, not preserved.\n\n"
            "The test compares your output to `F.conv2d(x, weight)` to fp "
            "tolerance."
        ),
        "stub": (
            "def ex1_conv2d_by_ic_sum(x: Tensor, weight: Tensor) -> Tensor:\n"
            '    """conv2d, but reconstructed as a sum over input channels."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# Small smoke test: B=1, IC=3 (RGB-like), OC=2, K=3.\n"
            "B, IC, H, W = 1, 3, 8, 8\n"
            "OC, KH, KW = 2, 3, 3\n"
            "x = t.randn(B, IC, H, W, generator=rng)\n"
            "weight = t.randn(OC, IC, KH, KW, generator=rng)\n"
            "y_ours = ex1_conv2d_by_ic_sum(x, weight)\n"
            "y_ref  = F.conv2d(x, weight)\n"
            "assert y_ours.shape == y_ref.shape == (B, OC, H - KH + 1, W - KW + 1), (\n"
            "    f'shape mismatch: ours={tuple(y_ours.shape)} ref={tuple(y_ref.shape)}'\n"
            ")\n"
            "assert y_ours.dtype == t.float32\n"
            "assert t.allclose(y_ours, y_ref, atol=1e-4), 'IC-sum decomposition must equal F.conv2d'\n"
            "\n"
            "# Larger spec — confirms contraction holds for many IC.\n"
            "B, IC, H, W = 2, 7, 12, 12\n"
            "OC, KH, KW = 4, 5, 5\n"
            "x2 = t.randn(B, IC, H, W, generator=rng)\n"
            "w2 = t.randn(OC, IC, KH, KW, generator=rng)\n"
            "y2 = ex1_conv2d_by_ic_sum(x2, w2)\n"
            "y2_ref = F.conv2d(x2, w2)\n"
            "assert t.allclose(y2, y2_ref, atol=1e-4)\n"
            "\n"
            "# Sanity probe: zero out one IC slot of the kernel and confirm the\n"
            "# output drops by exactly that channel's contribution.\n"
            "w_masked = w2.clone()\n"
            "w_masked[:, 0, :, :] = 0.0  # kill ic=0\n"
            "y_masked = ex1_conv2d_by_ic_sum(x2, w_masked)\n"
            "y_only0  = F.conv2d(x2[:, 0:1], w2[:, 0:1])\n"
            "assert t.allclose(y_masked + y_only0, y2_ref, atol=1e-4), (\n"
            "    'killing ic=0 in the kernel must subtract exactly that channel\\'s contribution'\n"
            ")\n"
            "\n"
            "# IC=1 edge case — should be a no-op channel sum (one iteration only).\n"
            "x1 = t.randn(1, 1, 6, 6, generator=rng)\n"
            "w1 = t.randn(3, 1, 2, 2, generator=rng)\n"
            "out1 = ex1_conv2d_by_ic_sum(x1, w1)\n"
            "assert t.allclose(out1, F.conv2d(x1, w1), atol=1e-5)"
        ),
        "solution_body": (
            "def ex1_conv2d_by_ic_sum(x: Tensor, weight: Tensor) -> Tensor:\n"
            "    from torch.nn import functional as F\n"
            "    B, IC, H, W = x.shape\n"
            "    OC, _, KH, KW = weight.shape\n"
            "    y = t.zeros(B, OC, H - KH + 1, W - KW + 1, dtype=x.dtype)\n"
            "    for ic in range(IC):\n"
            "        y = y + F.conv2d(x[:, ic:ic+1], weight[:, ic:ic+1])\n"
            "    return y"
        ),
        "solution_notes": (
            "**Why slicing with `ic:ic+1` (not `[ic]`).** Keeping a size-1 "
            "axis preserves the 4-D layout `(B, 1, H, W)` that `F.conv2d` "
            "expects. Using `[ic]` would collapse to `(B, H, W)` and crash.\n\n"
            "**Why `y = y + ...` (not `y +=`).** `+=` would be an in-place "
            "op on a freshly allocated zero tensor — fine here, but writing "
            "the non-in-place form makes the accumulation obvious and "
            "autograd-safe if you ever want to backprop through this.\n\n"
            "**The bigger picture.** This is exactly what the einsum form "
            "compiles to: `'b ic h w, oc ic kh kw -> b oc h_out w_out'` says "
            "\"sum over `ic` (and `kh, kw`).\" The loop makes the contraction "
            "explicit; the einsum makes it fast."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-kernel-shape (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-kernel-shape",
        "subtopic": "CNN: Kernel shape (OC, IC, KH, KW)",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_KERNEL_SHAPE,
        "exercise_index": 1,
        "exercise_title": "introspect a conv2d weight tensor's axes",
        "slug": "introspect-a-conv2d-weight-tensors-axes",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["conv2d", "weight-shape", "introspection", "param-count"],
        "kcs": ["conv-kernel-axis-order", "conv-param-count-from-shape"],
        "lo": (
            "Apply the `(OC, IC, KH, KW)` weight-shape convention to extract "
            "the four axes of a `nn.Conv2d` weight tensor and compute the "
            "exact parameter count (weights only, no bias)."
        ),
        "prompt_body": (
            "Implement `ex1_conv2d_weight_facts(conv)`. Given an instantiated "
            "`nn.Conv2d` module, return a dict with the following keys:\n\n"
            "- `'out_channels'` (int) — `OC`, first axis of `weight`.\n"
            "- `'in_channels'`  (int) — `IC`, second axis.\n"
            "- `'kernel_height'` (int) — `KH`, third axis.\n"
            "- `'kernel_width'`  (int) — `KW`, fourth axis.\n"
            "- `'n_weight_params'` (int) — total scalars in `weight` "
            "(`OC * IC * KH * KW`).\n\n"
            "**Hint.** Read `conv.weight.shape` and unpack the four axes. "
            "Do NOT trust `conv.in_channels` / `conv.out_channels` — derive "
            "everything from `weight.shape` so the drill targets the layout, "
            "not the module's stored attributes.\n\n"
            "The test instantiates several `nn.Conv2d` modules and confirms "
            "your extraction matches the constructor args."
        ),
        "stub": (
            "def ex1_conv2d_weight_facts(conv) -> dict:\n"
            '    """Return {out_channels, in_channels, kernel_height, kernel_width, n_weight_params}."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "def _check(ic, oc, k, expect_params):\n"
            "    conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k)\n"
            "    facts = ex1_conv2d_weight_facts(conv)\n"
            "    kh, kw = (k, k) if isinstance(k, int) else k\n"
            "    expected = {\n"
            "        'out_channels': oc,\n"
            "        'in_channels': ic,\n"
            "        'kernel_height': kh,\n"
            "        'kernel_width':  kw,\n"
            "        'n_weight_params': expect_params,\n"
            "    }\n"
            "    assert facts == expected, f'mismatch for ic={ic} oc={oc} k={k}:\\n  got      {facts}\\n  expected {expected}'\n"
            "\n"
            "# Standard 3x3 from 3 (RGB) → 16 OC.\n"
            "_check(3, 16, 3, expect_params=16 * 3 * 3 * 3)\n"
            "# 1x1 conv (channel mixer).\n"
            "_check(64, 128, 1, expect_params=128 * 64 * 1 * 1)\n"
            "# Non-square kernel.\n"
            "_check(8, 4, (5, 3), expect_params=4 * 8 * 5 * 3)\n"
            "# Big channel count — typical mid-resnet conv.\n"
            "_check(256, 512, 3, expect_params=512 * 256 * 9)\n"
            "\n"
            "# Spot-check by re-flattening the weight tensor.\n"
            "conv_spot = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)\n"
            "facts = ex1_conv2d_weight_facts(conv_spot)\n"
            "assert facts['n_weight_params'] == conv_spot.weight.numel(), (\n"
            "    f\"n_weight_params should equal weight.numel(): {facts['n_weight_params']} vs {conv_spot.weight.numel()}\"\n"
            ")\n"
            "\n"
            "# Cross-axis swap detection: confirm we did NOT accidentally read\n"
            "# the IC-first ConvTranspose layout. A Conv2d's weight.shape[0]\n"
            "# must be OC.\n"
            "conv_check = nn.Conv2d(7, 13, 3)\n"
            "assert ex1_conv2d_weight_facts(conv_check)['out_channels'] == 13\n"
            "assert ex1_conv2d_weight_facts(conv_check)['in_channels']  == 7\n"
            "print('  layout confirmed: (OC, IC, KH, KW) — NOT (IC, OC, ...)')"
        ),
        "solution_body": (
            "def ex1_conv2d_weight_facts(conv) -> dict:\n"
            "    OC, IC, KH, KW = conv.weight.shape\n"
            "    return {\n"
            "        'out_channels':    int(OC),\n"
            "        'in_channels':     int(IC),\n"
            "        'kernel_height':   int(KH),\n"
            "        'kernel_width':    int(KW),\n"
            "        'n_weight_params': int(OC * IC * KH * KW),\n"
            "    }"
        ),
        "solution_notes": (
            "**Why `int(...)` casts.** `conv.weight.shape` is a "
            "`torch.Size`, whose entries are plain `int` already in modern "
            "PyTorch — but on older builds they can be `torch.SymInt` (under "
            "compile/dynamic shapes). Explicit `int()` guarantees JSON-"
            "serializable plain ints regardless of source.\n\n"
            "**Why NOT use `conv.in_channels`.** The drill targets the "
            "*layout convention* — i.e., the fact that `weight.shape[0] == "
            "OC`. Reading `conv.in_channels` would bypass that and miss the "
            "point. The exception (out of scope for this drill) is "
            "`nn.ConvTranspose2d`, where the same `.in_channels` attribute "
            "exists but `weight.shape[0] == IC` (the swapped layout).\n\n"
            "**Param-count consequence.** A typical ResNet-block conv 256→512 "
            "kernel 3×3 is 1.18M params. Multiply by the dozens of such "
            "layers in a deep model and you see why CNNs are mostly *kernel "
            "weights* — far more than the input data itself."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-stride-downsample (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-stride-downsample",
        "subtopic": "CNN: Stride downsample arithmetic",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_STRIDE_DOWNSAMPLE,
        "exercise_index": 1,
        "exercise_title": "predict strided conv output length",
        "slug": "predict-strided-conv-output-length",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["stride", "downsample", "output-shape", "off-by-one"],
        "kcs": ["stride-floor-div-formula", "stride-leading-window-plus-one"],
        "lo": (
            "Apply `H_out = (H_in - K) // S + 1` to predict the output "
            "spatial length of a strided 1-D convolution and verify against "
            "`F.conv1d` for stride > 1."
        ),
        "prompt_body": (
            "Implement `ex1_strided_conv_outlen(h_in, k, s)`. Return the "
            "output length of a stride-`s` 1-D convolution with kernel size "
            "`k` and no padding.\n\n"
            "**Formula.**\n"
            "```\n"
            "h_out = (h_in - k) // s + 1\n"
            "```\n\n"
            "**Hint.** Use Python integer arithmetic — no tensors. The "
            "`+ 1` accounts for the *leading* window starting at index 0; "
            "the floor division counts how many additional stride-`s` "
            "positions fit before the trailing edge.\n\n"
            "**Common off-by-one.** `h_in // s` is the naive 'downsample by "
            "s' answer; it is wrong by exactly 1 (it forgets the leading "
            "window). The test deliberately exercises this trap.\n\n"
            "After your computation, the test cross-checks against a real "
            "`F.conv1d` call with random weights, so any off-by-one breaks "
            "the assertion immediately."
        ),
        "stub": (
            "def ex1_strided_conv_outlen(h_in: int, k: int, s: int) -> int:\n"
            '    """Output length of a stride-s conv with kernel k, no padding."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "# Direct value checks.\n"
            "assert ex1_strided_conv_outlen(10, 3, 1) == 8,  '10-3+1 = 8 (stride 1)'\n"
            "assert ex1_strided_conv_outlen(10, 3, 2) == 4,  '(10-3)//2 + 1 = 4 (not 5)'\n"
            "assert ex1_strided_conv_outlen(32, 3, 2) == 15, '(32-3)//2 + 1 = 15 (NOT 16!)'\n"
            "assert ex1_strided_conv_outlen(32, 5, 2) == 14, '(32-5)//2 + 1 = 14'\n"
            "assert ex1_strided_conv_outlen(7,  7, 1) == 1,  'kernel == input → exactly 1 window'\n"
            "assert ex1_strided_conv_outlen(9,  3, 3) == 3,  'stride == kernel: non-overlapping → 3'\n"
            "\n"
            "# Off-by-one trap — verify we did NOT just return h_in // s.\n"
            "wrong = 32 // 2  # would give 16\n"
            "right = ex1_strided_conv_outlen(32, 3, 2)\n"
            "assert right != wrong, 'off-by-one: should be 15, not h_in // s = 16'\n"
            "\n"
            "# Stride-1 special case: h_out = h_in - k + 1.\n"
            "for h in [8, 16, 33, 64]:\n"
            "    for k in [1, 3, 5]:\n"
            "        assert ex1_strided_conv_outlen(h, k, 1) == h - k + 1\n"
            "\n"
            "# Cross-check against actual F.conv1d for many (h, k, s) combos.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "cases = [(20, 3, 2), (32, 5, 3), (50, 7, 4), (100, 1, 5), (17, 4, 1), (11, 11, 1)]\n"
            "for h_in, k, s in cases:\n"
            "    x = t.randn(1, 1, h_in, generator=rng)\n"
            "    w = t.randn(1, 1, k, generator=rng)\n"
            "    actual = F.conv1d(x, w, stride=s).shape[-1]\n"
            "    predicted = ex1_strided_conv_outlen(h_in, k, s)\n"
            "    assert predicted == actual, (\n"
            "        f'h={h_in} k={k} s={s}: predicted {predicted}, actual {actual}'\n"
            "    )\n"
            "\n"
            "# Fractional trailing window must be DROPPED (floor, not round).\n"
            "# h=33, k=3, s=2 → (33-3)/2 + 1 = 16.0 exactly — fine.\n"
            "# h=33, k=3, s=3 → (33-3)/3 + 1 = 11.0 — fine.\n"
            "# h=34, k=3, s=3 → (34-3)/3 + 1 = 11.333 → floor = 11.\n"
            "assert ex1_strided_conv_outlen(34, 3, 3) == 11, 'partial trailing window dropped'"
        ),
        "solution_body": (
            "def ex1_strided_conv_outlen(h_in: int, k: int, s: int) -> int:\n"
            "    return (h_in - k) // s + 1"
        ),
        "solution_notes": (
            "**Why the `+ 1` matters.** The number of valid window *positions* "
            "is `floor((h_in - k) / s) + 1`. Think of it as: the leading window "
            "starts at index 0 (that's 1 position); the floored quotient counts "
            "how many additional stride-`s` positions fit before the right edge "
            "would push the kernel out of bounds.\n\n"
            "**Floor vs round.** PyTorch uses *floor*: a partial trailing "
            "window is dropped silently. Some libraries (Caffe, older "
            "Theano) used *ceil*. If you ever port code between frameworks, "
            "this is the first off-by-one to suspect.\n\n"
            "**Same-padding rescue.** With padding `p = (k - 1) // 2` (for "
            "odd `k`) and stride `s`, you get `h_out = ceil(h_in / s)` "
            "(approximately) — the canonical 'downsample by s' shape that "
            "ResNet stride-2 layers rely on. Without padding, the off-by-one "
            "bites every time.\n\n"
            "**Generalization to 2-D.** Apply the same formula independently "
            "to height and width: `(H, W) → ((H-KH)//SH + 1, (W-KW)//SW + 1)`."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-windowing-2d (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-windowing-2d",
        "subtopic": "CNN: 2-D conv windowing",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_WINDOWING_2D,
        "exercise_index": 1,
        "exercise_title": "build the 2-D conv window view via as_strided",
        "slug": "build-the-2d-conv-window-view-via-as-strided",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["as_strided", "windowing-2d", "view", "conv2d-equivalence", "visualization"],
        "kcs": ["windowing-2d-stride-pattern", "windowing-2d-output-shape"],
        "lo": (
            "Apply `as_strided` along **both** spatial axes to build the "
            "`(B, IC, OH, OW, KH, KW)` window view of a 2-D input for "
            "stride-1 convolution, and verify einsum-with-kernel matches "
            "`F.conv2d`."
        ),
        "prompt_body": (
            "Implement `ex1_conv2d_windows(x, KH, KW)`. Given input "
            "`x: (B, IC, H, W)` and kernel sizes `KH, KW`, return the strided "
            "window view of shape `(B, IC, OH, OW, KH, KW)` where "
            "`OH = H - KH + 1`, `OW = W - KW + 1`, and each `(KH, KW)` slice "
            "along the new `(OH, OW)` axes is one stride-1 window of `x`.\n\n"
            "**The trick.** Read `x.stride()` to get `(s_b, s_ic, s_h, s_w)`, "
            "then call:\n\n"
            "```\n"
            "x.as_strided(\n"
            "    size=(B, IC, OH, OW, KH, KW),\n"
            "    stride=(s_b, s_ic, s_h, s_w, s_h, s_w),\n"
            ")\n"
            "```\n\n"
            "**The teaching point.** The middle `(s_h, s_w)` advances "
            "*between* windows; the trailing `(s_h, s_w)` advances *within* "
            "a window. Same stride values, different semantic roles. "
            "Adjacent windows in `OH` overlap by `KH - 1` rows; in `OW` by "
            "`KW - 1` cols.\n\n"
            "**Constraints.** No copy — your returned tensor must share "
            "storage with `x` (the test confirms with `.data_ptr()`).\n\n"
            "After your view, the test contracts against a random kernel via "
            "`einops.einsum(..., 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')` "
            "and compares to `F.conv2d`.\n\n"
            "The visualization plots one input-channel feature map alongside "
            "a sample window so you can see the windowing geometrically."
        ),
        "stub": (
            "def ex1_conv2d_windows(x: Tensor, KH: int, KW: int) -> Tensor:\n"
            '    """Return (B, IC, OH, OW, KH, KW) window view of x for stride-1 conv2d."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "# --- Shape + no-copy check ---\n"
            "rng = t.Generator().manual_seed(0)\n"
            "x = t.arange(1.0, 1 + 1 * 1 * 6 * 6).reshape(1, 1, 6, 6).contiguous()\n"
            "KH, KW = 3, 3\n"
            "win = ex1_conv2d_windows(x, KH, KW)\n"
            "OH, OW = 6 - KH + 1, 6 - KW + 1\n"
            "assert win.shape == (1, 1, OH, OW, KH, KW), (\n"
            "    f'expected (1,1,{OH},{OW},{KH},{KW}), got {tuple(win.shape)}'\n"
            ")\n"
            "assert win.dtype == x.dtype\n"
            "assert win.data_ptr() == x.data_ptr(), 'must be a view (share storage with x)'\n"
            "\n"
            "# --- Value check at a few (oh, ow) positions ---\n"
            "for oh in range(OH):\n"
            "    for ow in range(OW):\n"
            "        ref = x[0, 0, oh:oh+KH, ow:ow+KW]\n"
            "        got = win[0, 0, oh, ow]\n"
            "        assert t.allclose(got, ref), f'window ({oh},{ow}) mismatch:\\n{got}\\nvs\\n{ref}'\n"
            "\n"
            "# --- Equivalence with F.conv2d on a multi-channel input ---\n"
            "B, IC, H, W, OC = 2, 3, 14, 16, 4\n"
            "KH2, KW2 = 5, 3\n"
            "x2 = t.randn(B, IC, H, W, generator=rng)\n"
            "weight = t.randn(OC, IC, KH2, KW2, generator=rng)\n"
            "win2 = ex1_conv2d_windows(x2, KH2, KW2)\n"
            "assert win2.shape == (B, IC, H - KH2 + 1, W - KW2 + 1, KH2, KW2)\n"
            "y_manual = einops.einsum(\n"
            "    win2, weight,\n"
            "    'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow',\n"
            ")\n"
            "y_native = F.conv2d(x2, weight)\n"
            "assert t.allclose(y_manual, y_native, atol=1e-4), (\n"
            "    'einsum(windows, weight) must equal F.conv2d to fp tolerance'\n"
            ")\n"
            "\n"
            "# --- Edge: KH == H, KW == W → single window of size (H, W) ---\n"
            "x3 = t.arange(25.0).reshape(1, 1, 5, 5).contiguous()\n"
            "win3 = ex1_conv2d_windows(x3, 5, 5)\n"
            "assert win3.shape == (1, 1, 1, 1, 5, 5)\n"
            "assert t.allclose(win3[0, 0, 0, 0], x3[0, 0]), 'single-window value must equal x'\n"
            "\n"
            "# --- Visualization: input feature map + one window highlighted ---\n"
            "import matplotlib.pyplot as plt\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "vis_x = x2[0, 0]                                          # (H, W) of batch 0, ic 0\n"
            "vis_win = win2[0, 0, 4, 6]                                # window at (oh=4, ow=6)\n"
            "axes[0].imshow(vis_x.numpy(), cmap='viridis')\n"
            "import matplotlib.patches as patches\n"
            "rect = patches.Rectangle((6 - 0.5, 4 - 0.5), KW2, KH2,\n"
            "                         linewidth=2, edgecolor='red', facecolor='none')\n"
            "axes[0].add_patch(rect)\n"
            "axes[0].set_title(f'input ic=0 with one ({KH2}x{KW2}) window highlighted')\n"
            "axes[1].imshow(vis_win.numpy(), cmap='viridis')\n"
            "axes[1].set_title('that window — view, not copy')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex1_conv2d_windows(x: Tensor, KH: int, KW: int) -> Tensor:\n"
            "    B, IC, H, W = x.shape\n"
            "    OH = H - KH + 1\n"
            "    OW = W - KW + 1\n"
            "    s_b, s_ic, s_h, s_w = x.stride()\n"
            "    return x.as_strided(\n"
            "        size=(B, IC, OH, OW, KH, KW),\n"
            "        stride=(s_b, s_ic, s_h, s_w, s_h, s_w),\n"
            "    )"
        ),
        "solution_notes": (
            "**Why `(s_h, s_w)` appears twice.** The new `OH` axis means "
            "'which window' along the height — advancing by 1 in `OH` moves "
            "the window down by 1 input row, which is `s_h` elements in "
            "storage. The new `KH` axis means 'position within a window' — "
            "also `s_h`. Same stride, different roles. Likewise for "
            "`OW` and `KW`.\n\n"
            "**The shape size.** A `(B, IC, OH, OW, KH, KW)` tensor *looks* "
            "like it should occupy `B * IC * OH * OW * KH * KW` elements of "
            "memory — but it doesn't, because `as_strided` doesn't copy. The "
            "windows alias each other; total storage stays at `B * IC * H * W`. "
            "This is why ARENA's 'from-scratch conv' is fast.\n\n"
            "**For strided conv (stride > 1).** Multiply the OH/OW strides by "
            "the conv stride: `stride=(s_b, s_ic, s_h * SH, s_w * SW, s_h, s_w)`. "
            "Compute `OH = (H - KH) // SH + 1`, etc. The KH/KW pair is "
            "unchanged because we always read every position within a window.\n\n"
            "**For padding.** Pre-pad the input with zeros (see the "
            "`conv-padding-zero` drill), then window the padded tensor. "
            "Composing the two drills gives the full ARENA conv2d."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ═══════════════════════════════════════════════════════════════════════
    # convT-kernel-axis-swap (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "convT-kernel-axis-swap",
        "subtopic": "CNN: ConvT kernel axis swap",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_CONVT_AXIS_SWAP,
        "exercise_index": 1,
        "exercise_title": "compare Conv2d and ConvTranspose2d weight shapes",
        "slug": "compare-conv2d-and-convtranspose2d-weight-shapes",
        "bloom_level": "Analyze",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["ConvTranspose2d", "weight-shape", "axis-swap", "introspection"],
        "kcs": ["convT-weight-axis-order", "conv-vs-convT-layout"],
        "lo": (
            "Analyze the axis-order asymmetry between `nn.Conv2d` "
            "`(OC, IC, KH, KW)` and `nn.ConvTranspose2d` `(IC, OC, KH, KW)` "
            "weight tensors by extracting each module's `weight.shape` and "
            "labeling each axis."
        ),
        "prompt_body": (
            "Implement `ex1_conv_layout_compare(ic, oc, k)`. Construct both a "
            "`nn.Conv2d(ic, oc, k)` and a `nn.ConvTranspose2d(ic, oc, k)` "
            "with matching constructor args, then return a dict:\n\n"
            "```python\n"
            "{\n"
            "  'conv_weight_shape':        tuple(...),   # from nn.Conv2d\n"
            "  'convT_weight_shape':       tuple(...),   # from nn.ConvTranspose2d\n"
            "  'conv_axis0_role':          'OC' or 'IC',\n"
            "  'convT_axis0_role':         'OC' or 'IC',\n"
            "  'axes_swapped':             True | False,\n"
            "}\n"
            "```\n\n"
            "**The point of the drill.** Both modules take the same "
            "`(in_channels, out_channels, kernel_size)` constructor args, "
            "yet produce weight tensors whose **first two axes are swapped**. "
            "Read both `.weight.shape` tuples directly and label each "
            "module's axis-0 role.\n\n"
            "**Hint.** `nn.Conv2d(ic, oc, k).weight.shape == (oc, ic, k, k)` "
            "(OC first). `nn.ConvTranspose2d(ic, oc, k).weight.shape == "
            "(ic, oc, k, k)` (IC first). So:\n"
            "- `conv_axis0_role = 'OC'`\n"
            "- `convT_axis0_role = 'IC'`\n"
            "- `axes_swapped = True` (always, unless `ic == oc`)."
        ),
        "stub": (
            "def ex1_conv_layout_compare(ic: int, oc: int, k: int) -> dict:\n"
            '    """Return shape + axis-0 role for nn.Conv2d and nn.ConvTranspose2d."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "# Canonical case: ic != oc so the swap is visible.\n"
            "facts = ex1_conv_layout_compare(3, 16, 5)\n"
            "assert facts['conv_weight_shape']  == (16, 3, 5, 5), f'conv shape wrong: {facts[\"conv_weight_shape\"]}'\n"
            "assert facts['convT_weight_shape'] == (3, 16, 5, 5), f'convT shape wrong: {facts[\"convT_weight_shape\"]}'\n"
            "assert facts['conv_axis0_role']  == 'OC', 'Conv2d axis-0 is OC'\n"
            "assert facts['convT_axis0_role'] == 'IC', 'ConvTranspose2d axis-0 is IC'\n"
            "assert facts['axes_swapped'] is True\n"
            "\n"
            "# Different (ic, oc) pair — also swapped.\n"
            "facts2 = ex1_conv_layout_compare(64, 128, 3)\n"
            "assert facts2['conv_weight_shape']  == (128, 64, 3, 3)\n"
            "assert facts2['convT_weight_shape'] == (64, 128, 3, 3)\n"
            "assert facts2['axes_swapped'] is True\n"
            "\n"
            "# Sanity: when ic == oc, axes_swapped is still True in the\n"
            "# *semantic* sense (the role of axis-0 still differs), but the\n"
            "# observable shape tuples are equal. We define the field by ROLE,\n"
            "# not by shape equality.\n"
            "facts3 = ex1_conv_layout_compare(8, 8, 3)\n"
            "assert facts3['conv_weight_shape']  == (8, 8, 3, 3)\n"
            "assert facts3['convT_weight_shape'] == (8, 8, 3, 3)\n"
            "assert facts3['conv_axis0_role']  == 'OC'\n"
            "assert facts3['convT_axis0_role'] == 'IC'\n"
            "assert facts3['axes_swapped'] is True, 'role still differs even when shapes coincide'\n"
            "\n"
            "# Direct module spot-check — no relying on facts dict.\n"
            "c  = nn.Conv2d(7, 13, 3)\n"
            "ct = nn.ConvTranspose2d(7, 13, 3)\n"
            "assert c.weight.shape  == (13, 7, 3, 3)\n"
            "assert ct.weight.shape == (7, 13, 3, 3)\n"
            "print('  Conv2d:          weight.shape[0] == OC (=', c.weight.shape[0], ')')\n"
            "print('  ConvTranspose2d: weight.shape[0] == IC (=', ct.weight.shape[0], ')')"
        ),
        "solution_body": (
            "def ex1_conv_layout_compare(ic: int, oc: int, k: int) -> dict:\n"
            "    from torch import nn\n"
            "    conv  = nn.Conv2d(ic, oc, k)\n"
            "    convT = nn.ConvTranspose2d(ic, oc, k)\n"
            "    return {\n"
            "        'conv_weight_shape':  tuple(conv.weight.shape),\n"
            "        'convT_weight_shape': tuple(convT.weight.shape),\n"
            "        'conv_axis0_role':    'OC',\n"
            "        'convT_axis0_role':   'IC',\n"
            "        'axes_swapped':       True,\n"
            "    }"
        ),
        "solution_notes": (
            "**Why the swap exists.** ConvTranspose is the *adjoint* of "
            "convolution. The forward conv contracts IC into OC; its adjoint "
            "contracts OC back into IC. PyTorch's storage convention is "
            "*first axis = the operation's contraction-input* — so Conv2d's "
            "axis 0 is OC (the filter index) and ConvTranspose2d's axis 0 is "
            "IC (since transposing the operation also transposes the role of "
            "the leading axis).\n\n"
            "**The trap in practice.** Loading a Conv2d's `state_dict` into a "
            "ConvTranspose2d's slot raises a `size mismatch` error (unless "
            "`ic == oc`). The fix is `weight = saved_weight.transpose(0, 1)`. "
            "Some checkpoint-converter tools do this automatically — but the "
            "asymmetry is the most-cited PyTorch interview gotcha for a "
            "reason.\n\n"
            "**Why we still call it `axes_swapped=True` when `ic == oc`.** "
            "The two tensors have *identical* observable shapes when "
            "`ic == oc`, but their **semantic axis roles still differ**. If "
            "you treated them as interchangeable and later changed the "
            "channel count, the model would silently break — so the role "
            "label is the load-bearing fact, not the shape tuple."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # convT-as-flipped-padded-conv (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "convT-as-flipped-padded-conv",
        "subtopic": "CNN: ConvT as flipped padded conv",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_CONVT_AS_FLIPPED_CONV,
        "exercise_index": 1,
        "exercise_title": "rebuild ConvTranspose2d as flipped padded Conv2d (stride 1)",
        "slug": "rebuild-convtranspose2d-as-flipped-padded-conv2d",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["ConvTranspose2d", "flip-kernel", "padding", "adjoint"],
        "kcs": ["convT-padded-conv-equivalence", "convT-kernel-flip-rule"],
        "lo": (
            "Analyze the equivalence between stride-1 `F.conv_transpose2d` "
            "and a regular `F.conv2d` on a `K-1`-padded input with a "
            "flipped-and-axis-swapped kernel."
        ),
        "prompt_body": (
            "Implement `ex1_convT_as_padded_conv(x, weight)`. Given input "
            "`x: (B, IC, H, W)` and ConvTranspose2d weight "
            "`weight: (IC, OC, K, K)` (square kernel for simplicity), "
            "reproduce `F.conv_transpose2d(x, weight)` (stride 1, no "
            "padding, no output_padding) by:\n\n"
            "1. **Padding `x` by `K - 1` on every spatial side** with zeros "
            "(use `F.pad(x, (K-1, K-1, K-1, K-1))`).\n"
            "2. **Flipping the kernel** along both spatial axes "
            "(`weight.flip(-1).flip(-2)`).\n"
            "3. **Swapping the kernel's channel axes** so it becomes "
            "`(OC, IC, K, K)` (`.transpose(0, 1)`).\n"
            "4. Running `F.conv2d` on the padded input with the "
            "flipped+swapped kernel.\n\n"
            "**Return** the resulting tensor; it must equal "
            "`F.conv_transpose2d(x, weight)` to fp tolerance.\n\n"
            "**The point of the drill.** ConvTranspose is just regular conv "
            "in disguise. Once you internalize the three transforms (pad, "
            "flip, swap), all of its mysteries dissolve.\n\n"
            "Hint on shape: the output is `(B, OC, H + K - 1, W + K - 1)` "
            "(stride 1, no extra padding). Transpose-conv *expands* spatial "
            "dims — that's why it's the canonical upsampling op."
        ),
        "stub": (
            "def ex1_convT_as_padded_conv(x: Tensor, weight: Tensor) -> Tensor:\n"
            '    """Reproduce F.conv_transpose2d using F.conv2d + pad + flip."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# Smoke test: small dims.\n"
            "B, IC, OC, H, W, K = 1, 2, 3, 5, 5, 3\n"
            "x = t.randn(B, IC, H, W, generator=rng)\n"
            "# ConvTranspose2d weight layout is (IC, OC, K, K).\n"
            "w = t.randn(IC, OC, K, K, generator=rng)\n"
            "y_ours = ex1_convT_as_padded_conv(x, w)\n"
            "y_ref  = F.conv_transpose2d(x, w)\n"
            "assert y_ours.shape == y_ref.shape == (B, OC, H + K - 1, W + K - 1), (\n"
            "    f'shape mismatch: ours={tuple(y_ours.shape)} ref={tuple(y_ref.shape)}'\n"
            ")\n"
            "assert y_ours.dtype == t.float32\n"
            "assert t.allclose(y_ours, y_ref, atol=1e-4), 'padded-flipped-conv must equal conv_transpose2d'\n"
            "\n"
            "# Larger test with multiple IC/OC and non-trivial kernel.\n"
            "B, IC, OC, H, W, K = 2, 4, 6, 8, 12, 5\n"
            "x2 = t.randn(B, IC, H, W, generator=rng)\n"
            "w2 = t.randn(IC, OC, K, K, generator=rng)\n"
            "y2 = ex1_convT_as_padded_conv(x2, w2)\n"
            "y2_ref = F.conv_transpose2d(x2, w2)\n"
            "assert y2.shape == (B, OC, H + K - 1, W + K - 1)\n"
            "assert t.allclose(y2, y2_ref, atol=1e-4)\n"
            "\n"
            "# K=1 edge case: pad amount 0, no flip effect → essentially a 1x1 conv.\n"
            "x3 = t.randn(1, 3, 4, 4, generator=rng)\n"
            "w3 = t.randn(3, 5, 1, 1, generator=rng)\n"
            "y3 = ex1_convT_as_padded_conv(x3, w3)\n"
            "y3_ref = F.conv_transpose2d(x3, w3)\n"
            "assert y3.shape == (1, 5, 4, 4)\n"
            "assert t.allclose(y3, y3_ref, atol=1e-4), 'K=1 case must still match'\n"
            "\n"
            "# Forgetting the flip is the most common bug — verify our solution\n"
            "# really does flip by checking that NOT flipping gives a DIFFERENT result.\n"
            "K = 3\n"
            "x4 = t.randn(1, 2, 4, 4, generator=rng)\n"
            "w4 = t.randn(2, 3, K, K, generator=rng)\n"
            "x4_pad = F.pad(x4, (K - 1,) * 4)\n"
            "y_no_flip = F.conv2d(x4_pad, w4.transpose(0, 1))           # flip omitted\n"
            "y_correct = ex1_convT_as_padded_conv(x4, w4)\n"
            "assert not t.allclose(y_no_flip, y_correct, atol=1e-4), (\n"
            "    'omitting the flip must produce a DIFFERENT result — '\n"
            "    'if these are equal, your weight is rotationally symmetric by accident'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_convT_as_padded_conv(x: Tensor, weight: Tensor) -> Tensor:\n"
            "    from torch.nn import functional as F\n"
            "    K = weight.shape[-1]\n"
            "    x_pad = F.pad(x, (K - 1, K - 1, K - 1, K - 1))           # (B, IC, H+2(K-1), W+2(K-1))\n"
            "    w_flipped = weight.flip(-1).flip(-2)                     # spatial flip\n"
            "    w_swapped = w_flipped.transpose(0, 1)                    # (OC, IC, K, K)\n"
            "    return F.conv2d(x_pad, w_swapped)"
        ),
        "solution_notes": (
            "**Why pad by `K - 1`.** A regular `(K, K)` conv on input of "
            "size `(H, W)` produces output `(H - K + 1, W - K + 1)`. To get "
            "transpose-conv's *expanded* output `(H + K - 1, W + K - 1)`, we "
            "need an effective input of `(H + 2(K-1), W + 2(K-1))` — i.e., "
            "`K-1` padding on every side.\n\n"
            "**Why flip both spatial axes.** The mathematical adjoint of "
            "cross-correlation (what PyTorch calls 'conv') is correlation "
            "with the flipped kernel. The flip swaps the role of the kernel "
            "indices `(kh, kw) ↔ (K-1-kh, K-1-kw)`, which is exactly what "
            "the adjoint operation requires when you derive it from the "
            "summation form.\n\n"
            "**Why swap channel axes.** ConvTranspose2d stores `(IC, OC, K, K)`, "
            "but `F.conv2d` expects `(OC, IC, K, K)`. The `.transpose(0, 1)` "
            "is purely a layout reformat — no flipping involved, just an "
            "axis relabel. Forgetting this raises a `size mismatch`.\n\n"
            "**Stride > 1 generalization.** For `stride = S` you ALSO need to "
            "insert `S - 1` zero rows/columns between every input pixel "
            "*before* padding by `K - 1`. That's where the upsampling factor "
            "comes from. We restrict to stride 1 here to keep the drill on "
            "the three-transforms story; the stride extension is a strictly "
            "harder follow-up."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # relu-elementwise-max (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "relu-elementwise-max",
        "subtopic": "CNN: ReLU as elementwise max",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_RELU_MAX,
        "exercise_index": 1,
        "exercise_title": "implement ReLU + verify the derivative jump at 0",
        "slug": "implement-relu-and-verify-derivative-jump",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["relu", "elementwise", "derivative", "autograd"],
        "kcs": ["relu-max-with-zero", "relu-derivative-at-zero"],
        "lo": (
            "Apply `ReLU(x) = max(x, 0)` elementwise via `t.maximum`, then "
            "use autograd to verify the derivative is 1 for x > 0, 0 for "
            "x < 0, and 0.5 at x = 0 (the symmetric sub-gradient of "
            "`t.maximum` on ties)."
        ),
        "prompt_body": (
            "Implement `ex1_relu_and_grad(x)`. Given a 1-D tensor "
            "`x: (N,)` with `requires_grad=True` already set, compute "
            "`y = ReLU(x)` (using `t.maximum`, NOT `F.relu`), then "
            "use `t.autograd.grad` to compute `dy_sum/dx` where "
            "`y_sum = y.sum()`. Return `(y, dy_dx)` — each shape `(N,)`.\n\n"
            "**Required implementation.** Use `t.maximum(x, t.tensor(0.0))` "
            "for the forward pass — this drill targets the canonical "
            "elementwise-max form. Equivalent options (`x.clamp(min=0)`, "
            "`F.relu(x)`) are forbidden so the drill exercises the "
            "*definition*, not the library shortcut.\n\n"
            "**For the gradient:**\n"
            "```\n"
            "(grad,) = t.autograd.grad(y.sum(), x)\n"
            "```\n"
            "This computes `d(sum(y))/dx = dy/dx` (since summing then "
            "differentiating w.r.t. each component yields a vector "
            "`(dy_0/dx_0, dy_1/dx_1, ...)` — exactly the per-element "
            "derivative).\n\n"
            "**The verification.** The test confirms:\n"
            "- For positive inputs, grad is 1.\n"
            "- For negative inputs, grad is 0.\n"
            "- At exactly x=0, grad is **0.5** (the symmetric sub-gradient "
            "that `t.maximum` produces on ties — different from `F.relu`, "
            "which uses 0). Both are valid sub-gradients in the closed "
            "interval `[0, 1]`."
        ),
        "stub": (
            "def ex1_relu_and_grad(x: Tensor):\n"
            '    """Returns (y, dy_dx) where y = ReLU(x) via t.maximum."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Inputs spanning negatives, zero, positives, and a few exact zeros.\n"
            "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0, 0.0, -1.0, 3.0], requires_grad=True)\n"
            "y, dy_dx = ex1_relu_and_grad(x)\n"
            "\n"
            "# --- Forward pass ---\n"
            "expected_y = t.tensor([0.0, 0.0, 0.0, 0.5, 2.0, 0.0, 0.0, 3.0])\n"
            "assert y.shape == x.shape, f'y shape: {tuple(y.shape)}'\n"
            "assert y.dtype == t.float32\n"
            "assert t.allclose(y.detach(), expected_y, atol=1e-6), f'forward wrong:\\n{y.detach()}\\nvs\\n{expected_y}'\n"
            "\n"
            "# --- Gradient ---\n"
            "# Positive → 1; negative → 0; exact 0 → 0.5 (t.maximum sub-gradient on ties).\n"
            "expected_grad = t.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0])\n"
            "assert dy_dx.shape == x.shape\n"
            "assert t.allclose(dy_dx, expected_grad, atol=1e-6), (\n"
            "    f'grad wrong:\\n{dy_dx}\\nvs\\n{expected_grad}\\n'\n"
            "    f'(if you got 0 at x=0, you used F.relu instead of t.maximum)'\n"
            ")\n"
            "\n"
            "# --- Cross-check against F.relu (forward only — grads differ at x=0) ---\n"
            "from torch.nn import functional as F\n"
            "x2 = t.linspace(-3, 3, 41, requires_grad=True)\n"
            "y2, g2 = ex1_relu_and_grad(x2)\n"
            "y2_ref = F.relu(x2.detach())\n"
            "assert t.allclose(y2.detach(), y2_ref, atol=1e-6), 'forward must match F.relu'\n"
            "# Pure positives in (0, 3] have grad 1; pure negatives in [-3, 0) have grad 0.\n"
            "is_strictly_positive = x2.detach() > 0\n"
            "is_strictly_negative = x2.detach() < 0\n"
            "assert t.allclose(g2[is_strictly_positive], t.ones(is_strictly_positive.sum()), atol=1e-6)\n"
            "assert t.allclose(g2[is_strictly_negative], t.zeros(is_strictly_negative.sum()), atol=1e-6)\n"
            "\n"
            "# --- Verify the implementation used t.maximum, not e.g. simple multiplication ---\n"
            "# (We can't easily inspect AST, but we CAN check behavior at NaN:\n"
            "#  t.maximum(nan, 0.0) = nan; t.where(x>0, x, 0) = 0. The drill spec\n"
            "#  says use t.maximum, so nan input must produce nan output.)\n"
            "x_nan = t.tensor([float('nan'), 1.0], requires_grad=True)\n"
            "y_nan, _ = ex1_relu_and_grad(x_nan)\n"
            "assert t.isnan(y_nan[0]).item(), 't.maximum(nan, 0) is nan — this confirms the canonical form'\n"
            "assert y_nan[1].item() == 1.0\n"
            "\n"
            "# --- Visualization: y(x) and dy/dx side by side ---\n"
            "import matplotlib.pyplot as plt\n"
            "x_plot = t.linspace(-3, 3, 121, requires_grad=True)\n"
            "y_plot, g_plot = ex1_relu_and_grad(x_plot)\n"
            "fig, ax = plt.subplots(1, 2, figsize=(9, 4))\n"
            "ax[0].plot(x_plot.detach().numpy(), y_plot.detach().numpy(), 'b-', linewidth=2)\n"
            "ax[0].axvline(0, color='grey', linestyle='--', alpha=0.5)\n"
            "ax[0].axhline(0, color='grey', linestyle='--', alpha=0.5)\n"
            "ax[0].set_title('ReLU(x) = max(x, 0)')\n"
            "ax[0].set_xlabel('x'); ax[0].set_ylabel('y'); ax[0].grid(True, alpha=0.3)\n"
            "ax[1].plot(x_plot.detach().numpy(), g_plot.numpy(), 'r-', linewidth=2)\n"
            "ax[1].axvline(0, color='grey', linestyle='--', alpha=0.5)\n"
            "ax[1].set_title('dReLU/dx (jump at 0)')\n"
            "ax[1].set_xlabel('x'); ax[1].set_ylabel('dy/dx'); ax[1].grid(True, alpha=0.3)\n"
            "ax[1].set_ylim(-0.2, 1.2)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex1_relu_and_grad(x: Tensor):\n"
            "    y = t.maximum(x, t.tensor(0.0))\n"
            "    (dy_dx,) = t.autograd.grad(y.sum(), x)\n"
            "    return y, dy_dx"
        ),
        "solution_notes": (
            "**Why `t.maximum` and not `t.max`.** `t.max(x, t.tensor(0.0))` "
            "is the same elementwise op, but `t.max(x)` (single arg) is the "
            "*reduction* — it returns a scalar (the global max), which is "
            "not what we want. `t.maximum(a, b)` is unambiguously "
            "elementwise.\n\n"
            "**The derivative jump at 0.** Mathematically the derivative "
            "is undefined at `x = 0` (left-derivative is 0, right-derivative "
            "is 1). The op you use decides the sub-gradient convention:\n\n"
            "- `t.maximum(x, 0)` → grad `0.5` at ties (symmetric average — "
            "this is what THIS drill exercises).\n"
            "- `F.relu(x)` → grad `0` at ties (smallest sub-gradient).\n\n"
            "Both choices live inside the valid sub-gradient interval `[0, 1]`. "
            "In practice the discrepancy is harmless because exact zeros are "
            "measure-zero events for random init.\n\n"
            "**Why we test with NaN.** A common 'optimization' is to write "
            "ReLU as `x * (x > 0)`, which produces 0 at NaN inputs (since "
            "`nan > 0` is `False`). The canonical `t.maximum` form propagates "
            "NaN, which is the correct IEEE-754 behavior and what `F.relu` "
            "does too. So the NaN test is implicitly an 'are you using the "
            "right form' check.\n\n"
            "**Why bother teaching this.** Most CNN bugs come from the layers "
            "between ReLUs (BN, conv layout). But a stray ReLU on the wrong "
            "tensor — final-classifier output, attention logits, gates — "
            "silently kills the negatives and breaks training. The "
            "`no-relu-on-final-layer` drill goes deeper into that failure."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ═══════════════════════════════════════════════════════════════════════
    # no-relu-on-final-layer (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "no-relu-on-final-layer",
        "subtopic": "CNN: No-ReLU on final layer",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_NO_RELU_FINAL,
        "exercise_index": 1,
        "exercise_title": "diagnose and strip a stray ReLU on the classifier head",
        "slug": "diagnose-and-strip-stray-relu-on-classifier-head",
        "bloom_level": "Evaluate",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["relu", "classifier", "logits", "debugging"],
        "kcs": ["final-layer-no-activation", "logits-vs-probs-pipeline"],
        "lo": (
            "Evaluate a broken CNN-classifier whose final layer has a stray "
            "ReLU; diagnose the failure mode (no negative logits → biased "
            "softmax) and return a fixed module that drops the final ReLU."
        ),
        "prompt_body": (
            "You are given a small classifier `BrokenClassifier` (instantiated "
            "in the test cell) whose architecture is:\n"
            "```\n"
            "Linear(in_features=8, out_features=16) → ReLU →\n"
            "Linear(in_features=16, out_features=4) → ReLU  ← STRAY!\n"
            "```\n"
            "The final ReLU clips all negative logits to 0. With random init, "
            "many logits end up exactly 0; downstream `F.cross_entropy` then "
            "computes near-uniform probabilities → loss never drops below "
            "`log(4) ≈ 1.386`.\n\n"
            "Implement `ex1_fix_classifier(broken)`. Given the broken module, "
            "**return a new `nn.Module`** that:\n\n"
            "1. **Reuses the broken model's weight tensors** (you may copy "
            "`broken.fc1.weight.data` / `broken.fc1.bias.data` etc., or "
            "reuse the modules directly — your call). Do NOT re-initialize.\n"
            "2. **Drops the final ReLU.** The forward pass must be "
            "`fc1 → ReLU → fc2` — no activation after `fc2`.\n"
            "3. Has the same input/output shape contract as `broken`.\n\n"
            "**Hint.** The simplest fix is a custom `nn.Module` that holds "
            "references to `broken.fc1` and `broken.fc2` and applies "
            "`F.relu` only between them. Then return an instance of that "
            "class.\n\n"
            "The test confirms:\n"
            "- The fixed model produces **at least some negative outputs** "
            "for random input (proving the final ReLU is gone).\n"
            "- Cross-entropy loss on a random-label batch is strictly LOWER "
            "than the broken model's loss after a few SGD steps (proving "
            "the model can actually learn now)."
        ),
        "stub": (
            "def ex1_fix_classifier(broken):\n"
            '    """Return a new nn.Module with the final ReLU removed."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "from torch.nn import functional as F\n"
            "\n"
            "# The deliberately-broken classifier — final layer has a stray ReLU.\n"
            "class BrokenClassifier(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc1 = nn.Linear(8, 16)\n"
            "        self.fc2 = nn.Linear(16, 4)\n"
            "\n"
            "    def forward(self, x):\n"
            "        h = F.relu(self.fc1(x))\n"
            "        logits = F.relu(self.fc2(h))   # ← BUG: stray ReLU on final layer\n"
            "        return logits\n"
            "\n"
            "t.manual_seed(0)\n"
            "broken = BrokenClassifier()\n"
            "fixed  = ex1_fix_classifier(broken)\n"
            "\n"
            "# --- Shape contract preserved ---\n"
            "x = t.randn(32, 8)\n"
            "out_broken = broken(x)\n"
            "out_fixed  = fixed(x)\n"
            "assert out_broken.shape == out_fixed.shape == (32, 4), (\n"
            "    f'shapes: broken={tuple(out_broken.shape)} fixed={tuple(out_fixed.shape)}'\n"
            ")\n"
            "\n"
            "# --- Diagnose: broken model has NO negative logits anywhere ---\n"
            "assert (out_broken >= 0).all(), 'broken model: final ReLU should clip every logit to >= 0'\n"
            "\n"
            "# --- Fix: must produce some negative logits on random input ---\n"
            "has_negative = (out_fixed < 0).any().item()\n"
            "assert has_negative, 'fixed model must produce at least one negative logit (no final ReLU)'\n"
            "frac_neg = (out_fixed < 0).float().mean().item()\n"
            "print(f'  fraction negative in fixed-model logits: {frac_neg:.3f}')\n"
            "\n"
            "# --- Weights were preserved (not re-init'd) ---\n"
            "# Either fixed reused broken's modules directly, or copied the parameter values.\n"
            "# We detect re-init by checking that fc2(x_pre_relu) matches in both models.\n"
            "with t.no_grad():\n"
            "    h_broken = F.relu(broken.fc1(x))\n"
            "    h_fixed  = F.relu(broken.fc1(x))   # same fc1\n"
            "    # The fixed model's pre-final-ReLU output == broken's fc2(h) result.\n"
            "    # Since broken applies ReLU AFTER fc2, the fixed model's output equals\n"
            "    # broken's pre-ReLU output, which we can compute manually:\n"
            "    pre_relu = broken.fc2(h_broken)\n"
            "assert t.allclose(out_fixed, pre_relu, atol=1e-5), (\n"
            "    'fixed output must equal broken.fc2(broken.fc1(x).relu()) — same weights, no final ReLU'\n"
            ")\n"
            "\n"
            "# --- Training behaviour: fixed model achieves lower CE loss after a few SGD steps ---\n"
            "# We need a LEARNABLE task — random labels won't separate the models because\n"
            "# neither model can learn random noise. Use a deterministic linear rule:\n"
            "# label = argmax(W_true @ x). The fixed model (which can produce negative\n"
            "# logits) can learn this; the broken model cannot.\n"
            "rng_data = t.Generator().manual_seed(123)\n"
            "W_true = t.randn(4, 8, generator=rng_data)            # ground-truth linear rule\n"
            "X_train = t.randn(512, 8, generator=rng_data)\n"
            "y_train = (X_train @ W_true.T).argmax(dim=1)          # deterministic labels\n"
            "\n"
            "def _ce_after_n_steps(model, n_steps=200, batch_size=64):\n"
            "    opt = t.optim.SGD(model.parameters(), lr=0.1)\n"
            "    rng = t.Generator().manual_seed(42)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        idx = t.randint(0, X_train.shape[0], (batch_size,), generator=rng)\n"
            "        xb = X_train[idx]\n"
            "        yb = y_train[idx]\n"
            "        logits = model(xb)\n"
            "        loss = F.cross_entropy(logits, yb)\n"
            "        opt.zero_grad()\n"
            "        loss.backward()\n"
            "        opt.step()\n"
            "        losses.append(loss.item())\n"
            "    return losses\n"
            "\n"
            "t.manual_seed(0)\n"
            "broken_train = BrokenClassifier()\n"
            "fixed_train  = ex1_fix_classifier(broken_train)\n"
            "broken_losses = _ce_after_n_steps(broken_train)\n"
            "fixed_losses  = _ce_after_n_steps(fixed_train)\n"
            "broken_final = sum(broken_losses[-20:]) / 20\n"
            "fixed_final  = sum(fixed_losses[-20:])  / 20\n"
            "print(f'  broken model final-20 mean CE loss: {broken_final:.4f}')\n"
            "print(f'  fixed  model final-20 mean CE loss: {fixed_final:.4f}')\n"
            "assert fixed_final < broken_final - 0.1, (\n"
            "    f'fixed model should train to MUCH LOWER CE loss ({fixed_final:.4f}) '\n"
            "    f'than broken ({broken_final:.4f}) on a learnable linear task'\n"
            ")\n"
            "# Specifically: broken plateaus near log(4) ≈ 1.386; fixed should beat it.\n"
            "import math\n"
            "uniform_loss = math.log(4)\n"
            "print(f'  uniform-distribution baseline: log(4) = {uniform_loss:.4f}')\n"
            "assert fixed_final < uniform_loss - 0.1, 'fixed model should beat the log(K) plateau'"
        ),
        "solution_body": (
            "def ex1_fix_classifier(broken):\n"
            "    from torch import nn\n"
            "    from torch.nn import functional as F\n"
            "\n"
            "    class FixedClassifier(nn.Module):\n"
            "        def __init__(self, fc1, fc2):\n"
            "            super().__init__()\n"
            "            self.fc1 = fc1   # REUSE — no re-init\n"
            "            self.fc2 = fc2\n"
            "\n"
            "        def forward(self, x):\n"
            "            h = F.relu(self.fc1(x))\n"
            "            return self.fc2(h)        # NO final ReLU — return raw logits\n"
            "\n"
            "    return FixedClassifier(broken.fc1, broken.fc2)"
        ),
        "solution_notes": (
            "**Why the broken model plateaus at `log(K)`.** Cross-entropy "
            "loss for a uniform prediction over `K` classes is exactly "
            "`log(K)` (Shannon entropy of the uniform distribution). When "
            "every logit is non-negative and many are zero, softmax produces "
            "a near-uniform distribution → loss `≈ log(4) ≈ 1.386` for "
            "`K=4`. The model literally cannot learn to discriminate, "
            "because gradients through the final ReLU are zero for half its "
            "inputs.\n\n"
            "**Why reuse the modules.** Re-initializing `fc1`/`fc2` would "
            "give the fixed model a head start over `broken_train` purely "
            "from the new init — masking the real cause. Reusing means both "
            "models start from identical weights; any difference at "
            "training-end comes from the architectural change only.\n\n"
            "**The architectural rule.** For classification, the canonical "
            "pattern is:\n"
            "```\n"
            "[Linear → activation] x N   →   Linear   →   loss(logits, labels)\n"
            "```\n"
            "The final `Linear` outputs **logits** (any real number, "
            "possibly negative). `F.cross_entropy` does the `LogSoftmax + "
            "NLLLoss` fused internally. Applying ReLU before "
            "`cross_entropy` is the most common 'why isn't my model "
            "learning' bug after wrong learning rate.\n\n"
            "**Symmetric case for regression.** A regression network "
            "outputting only non-negative values (with ReLU on the final "
            "layer) can never produce negative predictions — useful for "
            "things like predicting counts, but a silent bug if your target "
            "actually has negatives."
        ),
    },
]


def main():
    written = []
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        written.append(rel)
        print(f"wrote {rel}")
    print(f"\nTotal: {len(written)} notebooks")


if __name__ == "__main__":
    main()
