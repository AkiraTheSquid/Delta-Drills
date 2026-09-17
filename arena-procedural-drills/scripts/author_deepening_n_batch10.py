#!/usr/bin/env python3
"""Author 8 ex2 deepening drills for ARENA chap-0 CNN atoms.

Each ex2 probes a DISTINCT facet from the existing ex1 — different cognitive
operation, different surface context. ONE LO + ONE Bloom + <=2 KCs per drill.

Atom layout (8 atoms × 1 ex2 each):
  conv-channel-sum                  — ex2: verify per-OC linearity by OC zero-out
  conv-kernel-shape                 — ex2: construct a (OC, IC, KH, KW) weight from spec
  conv-stride-downsample            — ex2: find P that makes stride-S halve cleanly
  conv-windowing-2d                 — ex2: extend windowing to stride-S
  no-relu-on-final-layer            — ex2: detect stray final-ReLU from output stats
  avgpool-reduce                    — ex2: global avgpool via einops.reduce collapse
  batchnorm-affine-params           — ex2: recover gamma/beta from x_hat + y
  block-group-stack                 — ex2: introspect group, count shape-changers

Verification re-runs each spec's solution against its test_body inside the
build venv (torch 2.12.0+cpu) before any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_DEEP = "prereqs_cnn_deep"
TOPIC_EXTRAS = "prereqs_cnn_extras"

# ─────────────────────────────────────────────────────────────────────────────
# Recaps — short refreshers tuned to each ex2's facet.
# ─────────────────────────────────────────────────────────────────────────────

RECAP_CHANNEL_SUM_EX2 = (
    "## Conv2d per-OC linearity — quick refresher\n"
    "\n"
    "Convolution is linear in the kernel. Concretely, output channel `oc` is "
    "computed from `weight[oc]` alone — independent of every other output "
    "channel slot:\n"
    "\n"
    "```\n"
    "y[:, oc, :, :] = sum_ic conv2d_single(x[:, ic, :, :], weight[oc, ic, :, :])\n"
    "```\n"
    "\n"
    "**The consequence.** Zeroing out `weight[oc, :, :, :]` for some specific "
    "`oc` makes that output channel identically zero everywhere, while every "
    "other output channel is bit-exactly unchanged. The 16 filters of a "
    "Conv2d(IC, 16, K) are 16 fully independent linear maps stacked along the "
    "OC axis — they share input but not weights.\n"
    "\n"
    "**Why this matters.** It's the mathematical justification for "
    "per-channel pruning, filter visualization, and the channel-wise sparsity "
    "tricks that ResNet-family lottery-ticket papers exploit. Each `weight[oc]` "
    "is its own independent filter; the OC axis is a STACK, not a contraction."
)

RECAP_KERNEL_SHAPE_EX2 = (
    "## Construct a Conv2d weight from spec — quick refresher\n"
    "\n"
    "A `nn.Conv2d(IC, OC, (KH, KW))` weight tensor has shape "
    "**`(OC, IC, KH, KW)`** — output channels first. To build a weight tensor "
    "with custom contents that drops into a Conv2d slot, you must:\n"
    "\n"
    "1. Get the axis order right: `(OC, IC, KH, KW)`.\n"
    "2. Use floating-point dtype (`torch.float32` by default) — Conv2d won't "
    "accept int tensors as weights.\n"
    "3. Total scalars = `OC * IC * KH * KW`.\n"
    "\n"
    "**Common spec.** Constructors take `(in_channels, out_channels, "
    "kernel_size)` — note the order is IC-then-OC for the constructor, but "
    "the resulting `weight` tensor is OC-first. Don't conflate these two "
    "argument orderings.\n"
    "\n"
    "**Quick sanity probe.** After construction, verify "
    "`weight.shape == (OC, IC, KH, KW)` AND "
    "`nn.Conv2d(IC, OC, (KH, KW)).weight.shape == weight.shape` — both must "
    "agree exactly."
)

RECAP_STRIDE_PADDING_EX2 = (
    "## Same-padding for clean stride-S halving — quick refresher\n"
    "\n"
    "Default `Conv2d(stride=S, K)` (no padding) gives `H_out = (H_in - K) // S "
    "+ 1` — usually 1 short of the 'clean halve' answer `H_in // S`. The fix "
    "is padding `P` on each side, which generalizes the formula to:\n"
    "\n"
    "```\n"
    "H_out = (H_in + 2*P - K) // S + 1\n"
    "```\n"
    "\n"
    "**The 'clean half' goal.** For `S = 2` you want `H_out == H_in // 2`. "
    "Plug in and solve: `H_in // 2 = (H_in + 2P - K) // 2 + 1`, which gives "
    "`P = (K - 1) // 2` for odd `K` (and `H_in` even).\n"
    "\n"
    "**The general 'same-pad for stride-S' rule.** For odd kernel size `K`:\n"
    "\n"
    "```\n"
    "P = (K - 1) // 2\n"
    "```\n"
    "\n"
    "This works for `K = 3 → P = 1`, `K = 5 → P = 2`, `K = 7 → P = 3`. With "
    "this padding, every stride-`S` conv divides the input length cleanly by "
    "`S` (assuming the input is itself divisible by `S`).\n"
    "\n"
    "**Why this matters.** ResNet, U-Net, and every modern CNN family uses "
    "this exact pairing — odd kernel + `(K-1)//2` padding + stride-2 — to get "
    "predictable 2× downsampling per stage. The off-by-one without padding "
    "would compound across 4-5 stages and produce ugly non-power-of-2 sizes."
)

RECAP_WINDOWING_STRIDE_EX2 = (
    "## Strided 2-D windowing via `as_strided` — quick refresher\n"
    "\n"
    "Generalizing the stride-1 case: given `x: (B, IC, H, W)`, kernel "
    "`(KH, KW)`, and conv stride `(SH, SW)`, the strided window view is:\n"
    "\n"
    "```\n"
    "x.as_strided(\n"
    "    size=(B, IC, OH, OW, KH, KW),\n"
    "    stride=(s_b, s_ic, s_h * SH, s_w * SW, s_h, s_w),\n"
    ")\n"
    "```\n"
    "\n"
    "where `OH = (H - KH) // SH + 1` and `OW = (W - KW) // SW + 1`.\n"
    "\n"
    "**The only change from stride-1.** The MIDDLE pair `(s_h, s_w)` — which "
    "walks *between* windows — gets multiplied by `(SH, SW)`. The trailing "
    "pair `(s_h, s_w)` — which walks *within* a window — is unchanged "
    "(we always read every pixel inside a window).\n"
    "\n"
    "**Why.** Adjacent windows along the new `OH` axis used to be 1 input-row "
    "apart (stride 1). Now they're `SH` input-rows apart, so each step in `OH` "
    "moves `s_h * SH` storage positions. Same for width.\n"
    "\n"
    "**Equivalence.** Contract via `einops.einsum(..., 'b ic oh ow kh kw, oc "
    "ic kh kw -> b oc oh ow')` and the result equals "
    "`F.conv2d(x, weight, stride=(SH, SW))` to fp tolerance."
)

RECAP_NO_RELU_DETECT_EX2 = (
    "## Detecting a stray final ReLU from output stats — quick refresher\n"
    "\n"
    "A correctly-built classifier produces **logits**: real numbers, "
    "approximately mean-zero, with a substantial fraction NEGATIVE on random "
    "input. If you find a model whose output is *never* negative across many "
    "random inputs, that's the smoking gun for an accidental ReLU on the "
    "final layer (or `nn.Softplus`, `nn.Sigmoid`, etc.).\n"
    "\n"
    "**The detection rule.**\n"
    "\n"
    "```\n"
    "model.eval()\n"
    "with t.no_grad():\n"
    "    x = t.randn(batch, in_features)        # standard-normal input\n"
    "    y = model(x)\n"
    "    fraction_neg = (y < 0).float().mean().item()\n"
    "    suspicious = fraction_neg < 1e-3       # essentially never negative\n"
    "```\n"
    "\n"
    "**Why standard-normal input.** Half of the input entries are negative; "
    "after one linear layer the pre-activation distribution is still roughly "
    "zero-mean; after a ReLU it's strictly non-negative. Subsequent linear "
    "layers shift this around but a well-initialized network should produce "
    "outputs with substantial negative mass unless the final activation "
    "clips them.\n"
    "\n"
    "**The threshold.** Pure float comparisons can produce a few stray "
    "negative zeros even after ReLU, so we use `< 1e-3` not `== 0` for "
    "robustness. A real un-clipped classifier will have ~30-50% negative "
    "logits on random input — the gap is enormous."
)

RECAP_AVGPOOL_GLOBAL_EX2 = (
    "## Global average pooling — quick refresher\n"
    "\n"
    "Global avg-pool collapses the ENTIRE spatial extent into one scalar per "
    "(batch, channel). The einops form is the cleanest:\n"
    "\n"
    "```\n"
    "y = einops.reduce(x, 'b c h w -> b c', 'mean')\n"
    "```\n"
    "\n"
    "Input `(B, C, H, W)` → output `(B, C)`. The `h` and `w` axes are absent "
    "from the right side — that means they're reduced (averaged out).\n"
    "\n"
    "**Why ResNet uses this.** The last conv block of ResNet outputs "
    "`(B, 512, 7, 7)` (for 224×224 input). Before the classifier `Linear(512, "
    "num_classes)` can consume it, the spatial dims must collapse. Global "
    "avg-pool is the canonical way:\n"
    "\n"
    "```\n"
    "(B, 512, 7, 7)  →  mean over (h, w)  →  (B, 512)  →  Linear  →  (B, num_classes)\n"
    "```\n"
    "\n"
    "**Equivalence.** `nn.AdaptiveAvgPool2d((1, 1))(x).squeeze(-1).squeeze(-1)` "
    "produces an identical tensor. The einops form is more transparent — "
    "you SEE the reduction in the pattern string.\n"
    "\n"
    "**Contrast with local AvgPool2d.** Local pool keeps spatial structure "
    "(just at lower resolution); global pool destroys spatial structure "
    "entirely. Both use the same `mean` reduction — only the pattern string "
    "(factor pattern vs. straight reduce) differs."
)

RECAP_BN_RECOVER_EX2 = (
    "## Recovering BN's gamma/beta from before-after pairs — quick refresher\n"
    "\n"
    "BatchNorm's affine step is `y[:, c] = gamma[c] * x_hat[:, c] + beta[c]` "
    "per channel. Given access to both `x_hat` and `y` (the inputs and outputs "
    "of the affine step), you can recover `(gamma, beta)` per channel via "
    "simple linear regression — but for the BN case there's a much cleaner "
    "trick.\n"
    "\n"
    "**Two-point recovery.** Pick any TWO entries within the same channel "
    "where `x_hat` has distinct values:\n"
    "\n"
    "```\n"
    "y0 = gamma * x_hat0 + beta\n"
    "y1 = gamma * x_hat1 + beta\n"
    "\n"
    "gamma = (y1 - y0) / (x_hat1 - x_hat0)        # slope\n"
    "beta  = y0 - gamma * x_hat0                  # y-intercept\n"
    "```\n"
    "\n"
    "**Per-channel form.** Because the affine is channel-independent, you "
    "can do all `C` channels in parallel:\n"
    "\n"
    "```\n"
    "# Flatten everything per channel, then pick two distinct (x_hat, y) pairs.\n"
    "x_hat_flat = einops.rearrange(x_hat, 'b c h w -> c (b h w)')\n"
    "y_flat     = einops.rearrange(y,     'b c h w -> c (b h w)')\n"
    "gamma = (y_flat[:, 1] - y_flat[:, 0]) / (x_hat_flat[:, 1] - x_hat_flat[:, 0])\n"
    "beta  = y_flat[:, 0] - gamma * x_hat_flat[:, 0]\n"
    "```\n"
    "\n"
    "**Why this works.** The affine map `x → gamma * x + beta` is an "
    "**affine line** in (x_hat, y) space — slope `gamma`, intercept `beta`. "
    "Two points determine a line, so two `(x_hat, y)` pairs determine "
    "`(gamma, beta)` exactly."
)

RECAP_BLOCK_GROUP_INTROSPECT_EX2 = (
    "## Introspecting a BlockGroup's shape-change pattern — quick refresher\n"
    "\n"
    "A correctly-built ResNet BlockGroup has exactly **ONE** shape-changing "
    "block — block 0 — followed by `n_blocks - 1` identity-shaped blocks "
    "(in_feats == out_feats AND first_stride == 1). Verifying this invariant "
    "is the cleanest way to detect a mis-stacked group.\n"
    "\n"
    "**The introspection idiom.** Given a `nn.Sequential` of `ResBlock`s, "
    "walk children and tabulate:\n"
    "\n"
    "```\n"
    "shape_changers = [\n"
    "    i for i, b in enumerate(group)\n"
    "    if b.in_feats != b.out_feats or b.first_stride != 1\n"
    "]\n"
    "```\n"
    "\n"
    "Then assert `shape_changers == [0]` (only block 0 changes shape) AND "
    "`group[0].out_feats == group[-1].out_feats` (output width consistent).\n"
    "\n"
    "**Why this matters.** If somebody slips a downsample into block 3, the "
    "residual skip-connection breaks (input/output shapes no longer match) "
    "and the model silently underperforms or crashes at forward time. The "
    "structural check catches it before any data flows.\n"
    "\n"
    "**Generalization.** The same one-shape-change invariant holds for every "
    "BlockGroup variant — Bottleneck-ResNet, ResNeXt, DenseNet's transition "
    "blocks. As long as a 'group' is the unit of width/resolution change, "
    "exactly one block at the top of the group does the changing."
)


# ─────────────────────────────────────────────────────────────────────────────
# Exercise specs.
# ─────────────────────────────────────────────────────────────────────────────

SPECS = [
    # ═══════════════════════════════════════════════════════════════════════
    # conv-channel-sum (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-channel-sum",
        "subtopic": "CNN: Channel-axis sum semantics",
        "topic_folder": TOPIC_DEEP,
        "atom_recap_md": RECAP_CHANNEL_SUM_EX2,
        "exercise_index": 2,
        "exercise_title": "verify per-OC linearity by zeroing one output filter",
        "slug": "verify-per-oc-linearity-by-zeroing-one-filter",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["conv2d", "per-channel", "linearity", "filter-independence"],
        "kcs": ["conv-per-oc-independence", "conv-filter-stacking"],
        "lo": (
            "Analyze the per-output-channel independence of conv2d by zeroing "
            "the kernel slice for one OC slot and verifying that the matching "
            "output channel is identically zero while all other OC slots are "
            "bit-exactly unchanged."
        ),
        "prompt_body": (
            "Implement `ex2_zero_one_oc(x, weight, target_oc)`. Given input "
            "`x: (B, IC, H, W)`, kernel `weight: (OC, IC, KH, KW)`, and an "
            "integer `target_oc` in `[0, OC)`, return a **tuple** "
            "`(y_full, y_zeroed)` where:\n\n"
            "- `y_full = F.conv2d(x, weight)` — the reference output.\n"
            "- `y_zeroed = F.conv2d(x, w_zeroed)` where `w_zeroed` is a copy "
            "of `weight` with **only** `weight[target_oc, :, :, :]` set to "
            "zero (every other OC slot unchanged).\n\n"
            "**The teaching point.** The OC axis is a STACK of independent "
            "filters. Zeroing `weight[target_oc]` must:\n"
            "1. Set `y_zeroed[:, target_oc, :, :]` to all zeros (no kernel "
            "contribution for that filter).\n"
            "2. Leave EVERY OTHER `y_zeroed[:, other_oc, :, :]` bit-exactly "
            "equal to `y_full[:, other_oc, :, :]` — `target_oc`'s weights are "
            "independent of every other filter.\n\n"
            "**Hint.** Use `weight.clone()` to make a mutable copy, then "
            "in-place zero `w_zeroed[target_oc] = 0`. Do NOT modify the "
            "original `weight` tensor — the test confirms by re-running "
            "`F.conv2d(x, weight)` after your call.\n\n"
            "The test then walks every OC slot and asserts the bit-exact "
            "match for non-target slots and the all-zero invariant for the "
            "target."
        ),
        "stub": (
            "def ex2_zero_one_oc(x: Tensor, weight: Tensor, target_oc: int):\n"
            '    """Return (y_full, y_zeroed) where weight[target_oc] is zeroed in the second."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "rng = t.Generator().manual_seed(0)\n"
            "B, IC, H, W = 2, 3, 10, 10\n"
            "OC, KH, KW = 5, 3, 3\n"
            "x = t.randn(B, IC, H, W, generator=rng)\n"
            "weight = t.randn(OC, IC, KH, KW, generator=rng)\n"
            "weight_snapshot = weight.clone()\n"
            "\n"
            "# Run for target_oc = 2.\n"
            "target = 2\n"
            "y_full, y_zeroed = ex2_zero_one_oc(x, weight, target)\n"
            "\n"
            "# --- Shape + dtype contract ---\n"
            "OH, OW = H - KH + 1, W - KW + 1\n"
            "assert y_full.shape == (B, OC, OH, OW)\n"
            "assert y_zeroed.shape == (B, OC, OH, OW)\n"
            "assert y_full.dtype == t.float32\n"
            "\n"
            "# --- y_full must equal direct F.conv2d (sanity) ---\n"
            "assert t.allclose(y_full, F.conv2d(x, weight), atol=1e-6)\n"
            "\n"
            "# --- target OC slot of y_zeroed must be all zeros ---\n"
            "target_slice = y_zeroed[:, target, :, :]\n"
            "assert t.allclose(target_slice, t.zeros_like(target_slice), atol=1e-6), (\n"
            "    f'OC slot {target} of y_zeroed should be all zeros (kernel zeroed), '\n"
            "    f'got max-abs {target_slice.abs().max().item():.4e}'\n"
            ")\n"
            "\n"
            "# --- Every OTHER OC slot must equal y_full exactly ---\n"
            "for oc in range(OC):\n"
            "    if oc == target:\n"
            "        continue\n"
            "    diff = (y_zeroed[:, oc] - y_full[:, oc]).abs().max().item()\n"
            "    assert diff == 0.0, (\n"
            "        f'OC slot {oc} differs by {diff:.4e} after zeroing OC {target} — '\n"
            "        f'per-OC independence violated'\n"
            "    )\n"
            "\n"
            "# --- Original weight must NOT have been mutated ---\n"
            "assert t.allclose(weight, weight_snapshot, atol=0.0), (\n"
            "    'ex2_zero_one_oc must not mutate the input weight tensor'\n"
            ")\n"
            "\n"
            "# --- Different target_oc values all work ---\n"
            "for tgt in [0, 1, 4]:\n"
            "    _, y_z = ex2_zero_one_oc(x, weight, tgt)\n"
            "    assert t.allclose(y_z[:, tgt], t.zeros_like(y_z[:, tgt]), atol=1e-6)\n"
            "    for oc in range(OC):\n"
            "        if oc != tgt:\n"
            "            assert (y_z[:, oc] - y_full[:, oc]).abs().max().item() == 0.0, (\n"
            "                f'tgt={tgt}: OC {oc} mutated unexpectedly'\n"
            "            )\n"
            "\n"
            "# --- Edge case: OC=1 → zeroing the only filter gives all-zero output ---\n"
            "w1 = t.randn(1, 3, 3, 3, generator=rng)\n"
            "x1 = t.randn(1, 3, 6, 6, generator=rng)\n"
            "yf, yz = ex2_zero_one_oc(x1, w1, 0)\n"
            "assert t.allclose(yz, t.zeros_like(yz), atol=1e-6), 'OC=1: zeroed kernel → zeroed output'"
        ),
        "solution_body": (
            "def ex2_zero_one_oc(x: Tensor, weight: Tensor, target_oc: int):\n"
            "    from torch.nn import functional as F\n"
            "    y_full = F.conv2d(x, weight)\n"
            "    w_zeroed = weight.clone()\n"
            "    w_zeroed[target_oc] = 0.0\n"
            "    y_zeroed = F.conv2d(x, w_zeroed)\n"
            "    return y_full, y_zeroed"
        ),
        "solution_notes": (
            "**Why `weight.clone()` and not `.detach()`.** `.clone()` makes a "
            "true memory-independent copy; `.detach()` shares storage. "
            "Mutating a `.detach()`-ed tensor would mutate the original "
            "`weight` — the test catches this via the `weight_snapshot` "
            "assertion.\n\n"
            "**Why the equality is BIT-EXACT.** Conv2d's output for each OC "
            "is `sum_ic conv2d_single(x[:, ic], weight[oc, ic, :, :])`. "
            "Different `oc` slots NEVER share weight data, so changing "
            "`weight[target_oc]` cannot change any other slot's output — not "
            "even by floating-point noise. The test uses `diff == 0.0` "
            "exactly, not `allclose`.\n\n"
            "**The deeper invariant.** This is the structural justification "
            "for filter pruning: you can zero out any subset of OC slots in "
            "a trained network and the OTHER channels are bit-identical, so "
            "downstream layers see the same data on those channels. (The "
            "pruning literature uses this to motivate single-filter ablations "
            "as causal: any change in downstream loss is attributable to the "
            "pruned filter alone.)\n\n"
            "**Contrast with IC.** Zeroing `weight[:, target_ic, :, :, :]` "
            "(all OCs at one IC) is a DIFFERENT operation — it kills input "
            "channel `target_ic`'s contribution to *every* output channel. "
            "OC is a stack; IC is a sum."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-kernel-shape (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-kernel-shape",
        "subtopic": "CNN: Kernel shape (OC, IC, KH, KW)",
        "topic_folder": TOPIC_DEEP,
        "atom_recap_md": RECAP_KERNEL_SHAPE_EX2,
        "exercise_index": 2,
        "exercise_title": "construct a Conv2d weight tensor from a spec dict",
        "slug": "construct-a-conv2d-weight-tensor-from-spec",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["conv2d", "weight-shape", "construction", "axis-order"],
        "kcs": ["conv-kernel-axis-order", "conv-weight-construction"],
        "lo": (
            "Apply the `(OC, IC, KH, KW)` weight-shape convention to build a "
            "float32 weight tensor that matches the shape of "
            "`nn.Conv2d(in_channels=IC, out_channels=OC, kernel_size=(KH, KW)).weight`."
        ),
        "prompt_body": (
            "Implement `ex2_build_conv_weight(spec)`. Given a dict like:\n\n"
            "```python\n"
            "spec = {\n"
            "    'in_channels':  3,\n"
            "    'out_channels': 16,\n"
            "    'kernel_height': 5,\n"
            "    'kernel_width':  3,\n"
            "    'fill_value':    0.25,    # every weight entry set to this\n"
            "}\n"
            "```\n\n"
            "Return a `torch.Tensor` of dtype `float32` and shape "
            "**`(OC, IC, KH, KW)`** filled with `spec['fill_value']`.\n\n"
            "**The catch.** The constructor args use `(in_channels, "
            "out_channels, kernel_size)` order, but the resulting weight "
            "tensor is OC-first. Don't write a (IC, OC, KH, KW) tensor — "
            "that's the ConvTranspose2d layout and the shape-check against "
            "`nn.Conv2d(...).weight.shape` will fail.\n\n"
            "**Use `t.full(...)` or `t.empty(...).fill_(...)`** — anything "
            "that produces a fp32 tensor of the right shape. Don't use "
            "`t.tensor([[...]])` literal — too tedious for arbitrary OC, IC.\n\n"
            "The test instantiates a matching `nn.Conv2d` and confirms "
            "`weight.shape == module.weight.shape` exactly (the layout check), "
            "and `weight.numel() == OC * IC * KH * KW` (param count)."
        ),
        "stub": (
            "def ex2_build_conv_weight(spec: dict) -> Tensor:\n"
            '    """Build a float32 (OC, IC, KH, KW) weight tensor from spec dict."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "# --- Canonical case: 3 IC → 16 OC, 5x3 kernel ---\n"
            "spec = {'in_channels': 3, 'out_channels': 16, 'kernel_height': 5, 'kernel_width': 3, 'fill_value': 0.25}\n"
            "w = ex2_build_conv_weight(spec)\n"
            "assert isinstance(w, Tensor), f'must return a Tensor, got {type(w).__name__}'\n"
            "assert w.dtype == t.float32, f'dtype must be float32, got {w.dtype}'\n"
            "assert w.shape == (16, 3, 5, 3), f'shape wrong: expected (16,3,5,3), got {tuple(w.shape)}'\n"
            "assert t.allclose(w, t.full((16, 3, 5, 3), 0.25), atol=1e-6)\n"
            "\n"
            "# --- Layout MUST match the real Conv2d (this is the load-bearing assertion) ---\n"
            "module = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 3))\n"
            "assert w.shape == module.weight.shape, (\n"
            "    f'weight shape {tuple(w.shape)} must equal Conv2d.weight.shape {tuple(module.weight.shape)} '\n"
            "    f'— if you built (IC, OC, KH, KW) you have the ConvTranspose2d layout'\n"
            ")\n"
            "\n"
            "# --- Tensor must be assignable as a Conv2d weight ---\n"
            "with t.no_grad():\n"
            "    module.weight.copy_(w)\n"
            "assert t.allclose(module.weight, w, atol=1e-6)\n"
            "\n"
            "# --- Square kernel ---\n"
            "spec2 = {'in_channels': 64, 'out_channels': 128, 'kernel_height': 3, 'kernel_width': 3, 'fill_value': -1.0}\n"
            "w2 = ex2_build_conv_weight(spec2)\n"
            "assert w2.shape == (128, 64, 3, 3)\n"
            "assert w2.numel() == 128 * 64 * 9\n"
            "assert (w2 == -1.0).all()\n"
            "\n"
            "# --- 1x1 conv (channel mixer) ---\n"
            "spec3 = {'in_channels': 256, 'out_channels': 64, 'kernel_height': 1, 'kernel_width': 1, 'fill_value': 0.0}\n"
            "w3 = ex2_build_conv_weight(spec3)\n"
            "assert w3.shape == (64, 256, 1, 1)\n"
            "assert (w3 == 0.0).all()\n"
            "\n"
            "# --- Non-square: KH != KW ---\n"
            "spec4 = {'in_channels': 8, 'out_channels': 4, 'kernel_height': 7, 'kernel_width': 1, 'fill_value': 2.5}\n"
            "w4 = ex2_build_conv_weight(spec4)\n"
            "assert w4.shape == (4, 8, 7, 1), f'KH != KW case: expected (4,8,7,1), got {tuple(w4.shape)}'\n"
            "module4 = nn.Conv2d(8, 4, (7, 1))\n"
            "assert w4.shape == module4.weight.shape\n"
            "\n"
            "# --- Distinguish from ConvTranspose2d layout (IC, OC, KH, KW) ---\n"
            "ct_module = nn.ConvTranspose2d(3, 16, (5, 3))\n"
            "assert w.shape != ct_module.weight.shape, (\n"
            "    f'your weight matches ConvTranspose2d layout {tuple(ct_module.weight.shape)} — '\n"
            "    f'should match Conv2d layout {tuple(module.weight.shape)}'\n"
            ")"
        ),
        "solution_body": (
            "def ex2_build_conv_weight(spec: dict) -> Tensor:\n"
            "    OC = spec['out_channels']\n"
            "    IC = spec['in_channels']\n"
            "    KH = spec['kernel_height']\n"
            "    KW = spec['kernel_width']\n"
            "    return t.full((OC, IC, KH, KW), float(spec['fill_value']), dtype=t.float32)"
        ),
        "solution_notes": (
            "**Why `t.full(..., dtype=t.float32)`.** `t.full((shape,), 0)` "
            "would default to `int64` for an integer fill value, which "
            "`nn.Conv2d` won't accept. Explicit `dtype=t.float32` defends "
            "against int fill values too.\n\n"
            "**The OC-first rule, in plain English.** `weight[oc, ic, kh, kw]` "
            "is 'the `(kh, kw)` tap of input-channel `ic` for output-channel "
            "`oc`'. Reading axis 0 first as OC matches how you'd describe "
            "'pick filter 5 of 16, then pick its red-channel tap at position "
            "(2, 1)'.\n\n"
            "**Why the constructor order is different.** `nn.Conv2d(in, out, "
            "k)` follows the linear-algebra convention `y = Wx` where `W` is "
            "described as 'how to map FROM input TO output' — in→out order. "
            "But once you HAVE the weight tensor, indexing it by filter "
            "(`weight[oc]`) is more useful than indexing by input channel, so "
            "PyTorch's storage layout puts OC first. Two different "
            "conventions; both serve their use case.\n\n"
            "**ConvTranspose2d gotcha.** Loading a Conv2d's weight into a "
            "ConvTranspose2d's slot (without transposing axis 0 and 1) "
            "raises `size mismatch`. Conv2d = `(OC, IC, KH, KW)`; "
            "ConvTranspose2d = `(IC, OC, KH, KW)`. The test deliberately "
            "asserts they're DIFFERENT to catch the mix-up."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-stride-downsample (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-stride-downsample",
        "subtopic": "CNN: Stride downsample arithmetic",
        "topic_folder": TOPIC_DEEP,
        "atom_recap_md": RECAP_STRIDE_PADDING_EX2,
        "exercise_index": 2,
        "exercise_title": "find padding that gives a clean stride-S halve",
        "slug": "find-padding-for-clean-stride-halve",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["stride", "padding", "same-padding", "downsample", "off-by-one"],
        "kcs": ["same-pad-formula-odd-kernel", "stride-pad-conv-shape"],
        "lo": (
            "Analyze the `(H + 2P - K) // S + 1` shape formula to derive the "
            "padding `P = (K - 1) // 2` that makes an odd-kernel stride-S "
            "conv output equal `H_in // S` exactly."
        ),
        "prompt_body": (
            "Implement `ex2_same_pad_for_halve(k)`. Given an odd kernel size "
            "`k`, return the integer padding `P` such that a "
            "`Conv2d(..., kernel_size=k, stride=2, padding=P)` applied to an "
            "input of even spatial size `H_in` produces output of size "
            "exactly `H_in // 2`.\n\n"
            "**Derivation.** The general output formula is:\n\n"
            "```\n"
            "H_out = (H_in + 2*P - K) // S + 1\n"
            "```\n\n"
            "For `S = 2`, set `H_out = H_in // 2`. With odd `K` and even "
            "`H_in`, the closed-form padding is:\n\n"
            "```\n"
            "P = (K - 1) // 2\n"
            "```\n\n"
            "**Constraint.** Your implementation must validate that `k` is "
            "ODD — raise `ValueError(\"kernel must be odd\")` if `k` is even, "
            "since the closed form breaks down for even kernels (you'd need "
            "asymmetric padding).\n\n"
            "**Hint.** The formula is genuinely just `(k - 1) // 2`. The work "
            "is in convincing yourself by working through `k = 3, 5, 7`: each "
            "should produce `P = 1, 2, 3` respectively, and the test verifies "
            "the *actual* conv output shape equals `H_in // 2` for several "
            "`H_in` values.\n\n"
            "The test cross-checks the predicted padding against real "
            "`nn.Conv2d` output shapes for multiple `(K, H_in)` combos."
        ),
        "stub": (
            "def ex2_same_pad_for_halve(k: int) -> int:\n"
            '    """Padding P that makes stride-2 conv (odd K) halve the input cleanly."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "# --- Direct values ---\n"
            "assert ex2_same_pad_for_halve(1) == 0, 'K=1: no padding needed'\n"
            "assert ex2_same_pad_for_halve(3) == 1, 'K=3: P=1 (the canonical ResNet kernel)'\n"
            "assert ex2_same_pad_for_halve(5) == 2, 'K=5: P=2'\n"
            "assert ex2_same_pad_for_halve(7) == 3, 'K=7: P=3 (ResNet stem)'\n"
            "assert ex2_same_pad_for_halve(9) == 4, 'K=9: P=4'\n"
            "\n"
            "# --- Even-K must raise ---\n"
            "for bad_k in [2, 4, 6, 8]:\n"
            "    try:\n"
            "        ex2_same_pad_for_halve(bad_k)\n"
            "    except ValueError as e:\n"
            "        assert 'odd' in str(e).lower(), f'wrong error message: {e}'\n"
            "    else:\n"
            "        raise AssertionError(f'k={bad_k} should raise ValueError')\n"
            "\n"
            "# --- Cross-check against real nn.Conv2d shapes ---\n"
            "for k in [1, 3, 5, 7]:\n"
            "    p = ex2_same_pad_for_halve(k)\n"
            "    conv = nn.Conv2d(3, 8, kernel_size=k, stride=2, padding=p)\n"
            "    for h_in in [16, 32, 64, 128]:\n"
            "        x = t.zeros(1, 3, h_in, h_in)\n"
            "        out_shape = conv(x).shape[-1]\n"
            "        assert out_shape == h_in // 2, (\n"
            "            f'k={k} p={p} h_in={h_in}: expected {h_in//2}, got {out_shape}'\n"
            "        )\n"
            "\n"
            "# --- Verify by the formula directly ---\n"
            "for k in [3, 5, 7, 11]:\n"
            "    p = ex2_same_pad_for_halve(k)\n"
            "    for h_in in [16, 32, 50, 100]:\n"
            "        formula_out = (h_in + 2 * p - k) // 2 + 1\n"
            "        assert formula_out == h_in // 2, (\n"
            "            f'formula breaks at k={k} p={p} h_in={h_in}: got {formula_out}, want {h_in//2}'\n"
            "        )\n"
            "\n"
            "# --- Contrast: WITHOUT padding (P=0), stride-2 conv is OFF by 1 ---\n"
            "for k in [3, 5]:\n"
            "    naive_p = 0\n"
            "    conv_nopad = nn.Conv2d(3, 8, kernel_size=k, stride=2, padding=naive_p)\n"
            "    out = conv_nopad(t.zeros(1, 3, 32, 32)).shape[-1]\n"
            "    # Without padding: (32 - k) // 2 + 1 = 15 (K=3) or 14 (K=5).\n"
            "    # WITH our padding: would be 16 (32 // 2).\n"
            "    assert out < 16, f'k={k} no-pad: should be < 16 (off-by-one), got {out}'"
        ),
        "solution_body": (
            "def ex2_same_pad_for_halve(k: int) -> int:\n"
            "    if k % 2 == 0:\n"
            "        raise ValueError(f'kernel must be odd, got k={k}')\n"
            "    return (k - 1) // 2"
        ),
        "solution_notes": (
            "**Why `(k - 1) // 2`.** Plug into the shape formula:\n"
            "```\n"
            "H_out = (H_in + 2*((k-1)//2) - k) // 2 + 1\n"
            "      = (H_in + (k-1) - k) // 2 + 1     # for odd k, 2 * (k-1)/2 = k-1\n"
            "      = (H_in - 1) // 2 + 1\n"
            "      = H_in // 2                       # for even H_in\n"
            "```\n"
            "The off-by-one is absorbed by the `+ 1` from the leading-window "
            "term; padding adds exactly enough on each side to make the math "
            "work.\n\n"
            "**Why the odd-K constraint.** For even K, `(k - 1) / 2` is not "
            "an integer. You'd need asymmetric padding (`k // 2` on one side, "
            "`k // 2 - 1` on the other) — but `nn.Conv2d(padding=...)` only "
            "accepts symmetric padding. Even kernels require `F.pad` first, "
            "then `Conv2d(padding=0)`. The constraint sidesteps this.\n\n"
            "**Generalizing to stride S.** The same derivation gives "
            "`P = (k - 1) // 2` for ANY odd k and ANY stride S, provided "
            "`H_in % S == 0`. The padding formula is independent of stride — "
            "stride determines the output spacing, padding determines the "
            "fence-post offset.\n\n"
            "**Real-world use.** Every modern CNN's stride-2 downsample layer "
            "uses this padding. Look at any ResNet block's `nn.Conv2d` "
            "constructor and you'll see `kernel_size=3, stride=2, padding=1` "
            "— that's this drill's answer applied."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-windowing-2d (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-windowing-2d",
        "subtopic": "CNN: 2-D conv windowing",
        "topic_folder": TOPIC_DEEP,
        "atom_recap_md": RECAP_WINDOWING_STRIDE_EX2,
        "exercise_index": 2,
        "exercise_title": "extend the conv2d window view to stride S",
        "slug": "extend-conv2d-windowing-to-stride-s",
        "bloom_level": "Apply",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["as_strided", "windowing-2d", "stride", "conv2d-strided"],
        "kcs": ["windowing-stride-multiplier", "strided-conv-output-shape"],
        "lo": (
            "Apply `as_strided` with stride-multiplied between-window "
            "advances to build a `(B, IC, OH, OW, KH, KW)` window view of a "
            "2-D input for arbitrary conv stride `(SH, SW)`, and verify "
            "einsum-with-kernel matches `F.conv2d(stride=...)`."
        ),
        "prompt_body": (
            "Implement `ex2_conv2d_windows_strided(x, KH, KW, SH, SW)`. "
            "Given `x: (B, IC, H, W)`, kernel sizes `(KH, KW)`, and conv "
            "strides `(SH, SW)`, return the strided window view of shape "
            "`(B, IC, OH, OW, KH, KW)` where:\n\n"
            "- `OH = (H - KH) // SH + 1`\n"
            "- `OW = (W - KW) // SW + 1`\n"
            "- Each `(KH, KW)` slice along `(OH, OW)` is one kernel-sized "
            "window of `x` at conv stride `(SH, SW)`.\n\n"
            "**The trick (generalized from stride-1).** Read "
            "`x.stride()` = `(s_b, s_ic, s_h, s_w)`, then:\n\n"
            "```\n"
            "x.as_strided(\n"
            "    size=(B, IC, OH, OW, KH, KW),\n"
            "    stride=(s_b, s_ic, s_h * SH, s_w * SW, s_h, s_w),\n"
            ")\n"
            "```\n\n"
            "**The only change from stride-1.** The MIDDLE pair `(s_h * SH, "
            "s_w * SW)` — between-window step — gets multiplied by the conv "
            "stride. The TRAILING pair `(s_h, s_w)` — within-window step — "
            "is unchanged.\n\n"
            "**Why.** Adjacent windows in `OH` are now `SH` input rows apart "
            "(not 1). So moving 1 along `OH` moves `SH` rows in storage, "
            "i.e., `s_h * SH` elements. The within-window axes always read "
            "every position, so they keep `s_h` and `s_w`.\n\n"
            "**Constraints.** No copy — return must share storage with `x`.\n\n"
            "The test contracts against a random kernel via `einops.einsum` "
            "and compares to `F.conv2d(x, weight, stride=(SH, SW))` for "
            "multiple `(SH, SW)` combos."
        ),
        "stub": (
            "def ex2_conv2d_windows_strided(x: Tensor, KH: int, KW: int, SH: int, SW: int) -> Tensor:\n"
            '    """Return (B, IC, OH, OW, KH, KW) strided window view for stride-(SH, SW) conv2d."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# --- Stride-1 case must reduce to the simple view ---\n"
            "x = t.arange(1.0, 1 + 1 * 1 * 8 * 8).reshape(1, 1, 8, 8).contiguous()\n"
            "KH, KW = 3, 3\n"
            "win11 = ex2_conv2d_windows_strided(x, KH, KW, 1, 1)\n"
            "OH = (8 - KH) // 1 + 1\n"
            "OW = (8 - KW) // 1 + 1\n"
            "assert win11.shape == (1, 1, OH, OW, KH, KW)\n"
            "assert win11.data_ptr() == x.data_ptr(), 'must be a view (share storage with x)'\n"
            "# Value spot-check at (0, 0).\n"
            "assert t.allclose(win11[0, 0, 0, 0], x[0, 0, :KH, :KW])\n"
            "\n"
            "# --- Stride-2 case: OH = (8-3)//2 + 1 = 3 ---\n"
            "win22 = ex2_conv2d_windows_strided(x, KH, KW, 2, 2)\n"
            "OH2 = (8 - KH) // 2 + 1\n"
            "OW2 = (8 - KW) // 2 + 1\n"
            "assert win22.shape == (1, 1, OH2, OW2, KH, KW), f'got {tuple(win22.shape)}'\n"
            "assert win22.data_ptr() == x.data_ptr()\n"
            "# Adjacent windows in OH should be 2 rows apart in x.\n"
            "for oh in range(OH2):\n"
            "    for ow in range(OW2):\n"
            "        ref = x[0, 0, oh*2:oh*2+KH, ow*2:ow*2+KW]\n"
            "        assert t.allclose(win22[0, 0, oh, ow], ref), (\n"
            "            f'stride-2 window ({oh},{ow}) mismatch'\n"
            "        )\n"
            "\n"
            "# --- Equivalence with F.conv2d for multiple stride combos ---\n"
            "B, IC, H, W, OC = 2, 3, 16, 16, 4\n"
            "KH2, KW2 = 3, 3\n"
            "x2 = t.randn(B, IC, H, W, generator=rng)\n"
            "weight = t.randn(OC, IC, KH2, KW2, generator=rng)\n"
            "for SH, SW in [(1, 1), (2, 2), (3, 3), (2, 1), (1, 2), (2, 3)]:\n"
            "    win2 = ex2_conv2d_windows_strided(x2, KH2, KW2, SH, SW)\n"
            "    OH_exp = (H - KH2) // SH + 1\n"
            "    OW_exp = (W - KW2) // SW + 1\n"
            "    assert win2.shape == (B, IC, OH_exp, OW_exp, KH2, KW2), (\n"
            "        f'(SH,SW)=({SH},{SW}): shape {tuple(win2.shape)} vs expected ({B},{IC},{OH_exp},{OW_exp},{KH2},{KW2})'\n"
            "    )\n"
            "    y_manual = einops.einsum(\n"
            "        win2, weight,\n"
            "        'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow',\n"
            "    )\n"
            "    y_native = F.conv2d(x2, weight, stride=(SH, SW))\n"
            "    assert y_manual.shape == y_native.shape, (\n"
            "        f'(SH,SW)=({SH},{SW}): einsum shape mismatch'\n"
            "    )\n"
            "    assert t.allclose(y_manual, y_native, atol=1e-4), (\n"
            "        f'(SH,SW)=({SH},{SW}): einsum(windows, weight) must equal F.conv2d(stride=...)'\n"
            "    )\n"
            "\n"
            "# --- Edge: SH = SW = H = W = K (single non-overlapping window) ---\n"
            "x3 = t.arange(16.0).reshape(1, 1, 4, 4).contiguous()\n"
            "win3 = ex2_conv2d_windows_strided(x3, 4, 4, 4, 4)\n"
            "assert win3.shape == (1, 1, 1, 1, 4, 4)\n"
            "assert t.allclose(win3[0, 0, 0, 0], x3[0, 0])\n"
            "\n"
            "# --- Edge: stride larger than 1 but kernel exactly == H/W: still 1 window ---\n"
            "x4 = t.randn(1, 1, 5, 5, generator=rng)\n"
            "win4 = ex2_conv2d_windows_strided(x4, 5, 5, 3, 3)\n"
            "assert win4.shape == (1, 1, 1, 1, 5, 5)"
        ),
        "solution_body": (
            "def ex2_conv2d_windows_strided(x: Tensor, KH: int, KW: int, SH: int, SW: int) -> Tensor:\n"
            "    B, IC, H, W = x.shape\n"
            "    OH = (H - KH) // SH + 1\n"
            "    OW = (W - KW) // SW + 1\n"
            "    s_b, s_ic, s_h, s_w = x.stride()\n"
            "    return x.as_strided(\n"
            "        size=(B, IC, OH, OW, KH, KW),\n"
            "        stride=(s_b, s_ic, s_h * SH, s_w * SW, s_h, s_w),\n"
            "    )"
        ),
        "solution_notes": (
            "**The single load-bearing change.** Compare to the stride-1 "
            "version: only the middle pair `(s_h, s_w)` becomes "
            "`(s_h * SH, s_w * SW)`. Everything else is identical. The "
            "trailing pair stays `(s_h, s_w)` because we always step ONE row "
            "down within a window, regardless of conv stride.\n\n"
            "**Why two pairs of `(s_h, s_w)` exist at all.** The window view "
            "introduces TWO new axes for height (`OH` for between-window, "
            "`KH` for within-window) and TWO for width. Each new axis needs "
            "its own stride. The same input row participates in both 'I am "
            "the start of window N' and 'I am the M-th row of window N-1' — "
            "so the stride values overlap but the semantic roles don't.\n\n"
            "**Connecting to the output shape.** `OH = (H - KH) // SH + 1` "
            "is the same formula from `conv-stride-downsample`. Window count "
            "= valid kernel positions along the axis. The leading window "
            "starts at index 0; subsequent windows step by `SH`; the last "
            "must fit entirely (floor division).\n\n"
            "**For padding too.** Compose with the `conv-padding-zero` drill: "
            "pre-pad `x` by `F.pad`, then window the padded tensor with this "
            "strided view. That's the full ARENA conv2d implementation in two "
            "composable atoms."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # no-relu-on-final-layer (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "no-relu-on-final-layer",
        "subtopic": "CNN: No-ReLU on final layer",
        "topic_folder": TOPIC_DEEP,
        "atom_recap_md": RECAP_NO_RELU_DETECT_EX2,
        "exercise_index": 2,
        "exercise_title": "detect a stray final ReLU from output statistics",
        "slug": "detect-stray-final-relu-from-output-stats",
        "bloom_level": "Evaluate",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["relu", "diagnose", "logits", "output-stats"],
        "kcs": ["final-relu-detection-via-negativity", "logits-mean-zero-prior"],
        "lo": (
            "Evaluate whether an arbitrary classifier has a stray final-layer "
            "activation by sampling its outputs on standard-normal input and "
            "checking the fraction of negative logits against a threshold."
        ),
        "prompt_body": (
            "Implement `ex2_has_final_relu(model, in_features, batch=256)`. "
            "Given a classifier `model` (an `nn.Module`) and its expected "
            "`in_features`, return `True` if the model has a stray "
            "non-negative-clipping activation (ReLU, Softplus, Sigmoid, etc.) "
            "on its final layer; `False` otherwise.\n\n"
            "**Detection rule.**\n\n"
            "1. Put the model in `eval()` mode.\n"
            "2. Inside `with t.no_grad():`, sample `x = t.randn(batch, "
            "in_features)`.\n"
            "3. Compute `y = model(x)`.\n"
            "4. Compute `fraction_negative = (y < 0).float().mean().item()`.\n"
            "5. Return `True` if `fraction_negative < 1e-3` (essentially "
            "never negative → must have a clipping activation); `False` "
            "otherwise.\n\n"
            "**Why the threshold is `< 1e-3` not `== 0`.** Float arithmetic "
            "can produce stray `-0.0` values even after a `ReLU` (rare but "
            "possible). The `1e-3` threshold lets a few stragglers slip "
            "through while still catching the qualitative all-non-negative "
            "case. A real un-clipped classifier on random input will have "
            "~30-50% negative entries — the gap is huge.\n\n"
            "**Why standard-normal input.** It's the canonical 'unbiased' "
            "test distribution. With Xavier/He-initialized layers, "
            "intermediate activations stay roughly zero-mean unit-variance, "
            "and unclipped final outputs are also zero-mean — so half should "
            "be negative.\n\n"
            "The test runs your detector against a known-broken model "
            "(final ReLU present) and a known-good one (no final activation) "
            "and confirms it returns `True` and `False` respectively."
        ),
        "stub": (
            "def ex2_has_final_relu(model, in_features: int, batch: int = 256) -> bool:\n"
            '    """Return True iff model output appears to be clipped at zero (stray final activation)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "from torch.nn import functional as F\n"
            "\n"
            "# --- Known-broken: final ReLU present ---\n"
            "class BrokenClassifier(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc1 = nn.Linear(8, 16)\n"
            "        self.fc2 = nn.Linear(16, 4)\n"
            "    def forward(self, x):\n"
            "        h = F.relu(self.fc1(x))\n"
            "        return F.relu(self.fc2(h))   # stray\n"
            "\n"
            "# --- Known-good: no final activation ---\n"
            "class GoodClassifier(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc1 = nn.Linear(8, 16)\n"
            "        self.fc2 = nn.Linear(16, 4)\n"
            "    def forward(self, x):\n"
            "        h = F.relu(self.fc1(x))\n"
            "        return self.fc2(h)\n"
            "\n"
            "t.manual_seed(0)\n"
            "broken = BrokenClassifier()\n"
            "good   = GoodClassifier()\n"
            "\n"
            "# --- Broken → True ---\n"
            "result_broken = ex2_has_final_relu(broken, in_features=8)\n"
            "assert result_broken is True, f'broken classifier: detector should return True, got {result_broken}'\n"
            "\n"
            "# --- Good → False ---\n"
            "result_good = ex2_has_final_relu(good, in_features=8)\n"
            "assert result_good is False, f'good classifier: detector should return False, got {result_good}'\n"
            "\n"
            "# --- Detector must put model in eval mode (or at least not crash on dropout) ---\n"
            "class GoodWithDropout(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc1 = nn.Linear(8, 16)\n"
            "        self.drop = nn.Dropout(0.5)\n"
            "        self.fc2 = nn.Linear(16, 4)\n"
            "    def forward(self, x):\n"
            "        return self.fc2(self.drop(F.relu(self.fc1(x))))\n"
            "\n"
            "t.manual_seed(0)\n"
            "good_drop = GoodWithDropout()\n"
            "assert ex2_has_final_relu(good_drop, in_features=8) is False, (\n"
            "    'detector should still return False for a good classifier with dropout'\n"
            ")\n"
            "\n"
            "# --- Sigmoid on final layer also detected (output in (0, 1) → all >= 0) ---\n"
            "class SigmoidClassifier(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc1 = nn.Linear(8, 16)\n"
            "        self.fc2 = nn.Linear(16, 4)\n"
            "    def forward(self, x):\n"
            "        return t.sigmoid(self.fc2(F.relu(self.fc1(x))))\n"
            "\n"
            "t.manual_seed(0)\n"
            "sig = SigmoidClassifier()\n"
            "assert ex2_has_final_relu(sig, in_features=8) is True, (\n"
            "    'detector should return True for a final-sigmoid classifier '\n"
            "    '(output in (0,1) is also entirely non-negative)'\n"
            ")\n"
            "\n"
            "# --- Detector must use no_grad (we test by checking it doesn't break params) ---\n"
            "t.manual_seed(0)\n"
            "good2 = GoodClassifier()\n"
            "for p in good2.parameters():\n"
            "    assert p.grad is None\n"
            "_ = ex2_has_final_relu(good2, in_features=8)\n"
            "# Grads should STILL be None — the detector must not have called .backward().\n"
            "for p in good2.parameters():\n"
            "    assert p.grad is None, 'detector should not produce gradients'"
        ),
        "solution_body": (
            "def ex2_has_final_relu(model, in_features: int, batch: int = 256) -> bool:\n"
            "    was_training = model.training\n"
            "    model.eval()\n"
            "    try:\n"
            "        with t.no_grad():\n"
            "            x = t.randn(batch, in_features)\n"
            "            y = model(x)\n"
            "            fraction_negative = (y < 0).float().mean().item()\n"
            "    finally:\n"
            "        if was_training:\n"
            "            model.train()\n"
            "    return fraction_negative < 1e-3"
        ),
        "solution_notes": (
            "**Why preserve the training/eval state.** The detector should "
            "be a non-invasive probe — calling it shouldn't permanently flip "
            "a model to eval mode. Restoring `was_training` keeps the model's "
            "external state identical to before the call.\n\n"
            "**Why `model.eval()` matters.** Dropout and BatchNorm behave "
            "stochastically in train mode. Eval-mode dropout is the identity; "
            "eval-mode BN uses running stats. Without eval, BatchNorm with "
            "small-batch random input can produce wild outputs that confuse "
            "the detector.\n\n"
            "**The `with t.no_grad():` block.** Two reasons:\n"
            "1. **Speed.** No grad graph means faster forward pass and less "
            "memory.\n"
            "2. **Correctness contract.** The test verifies parameter "
            "gradients are still `None` after the detector runs — proves we "
            "didn't accidentally build a backward graph.\n\n"
            "**Threshold robustness.** Why `1e-3` not `0`?\n"
            "- After `t.maximum(x, t.tensor(0.0))` exactly-zero entries CAN "
            "happen but they're not strictly negative; `(y < 0)` is False "
            "for them.\n"
            "- `F.relu(-0.0)` returns `0.0` (positive zero), so no negatives.\n"
            "- Stray rounding through subsequent linear layers can in "
            "principle produce a `-eps`, but `1e-3` is plenty of margin while "
            "still rejecting the 30-50% expected from an unclipped model.\n\n"
            "**Limitations.** This detector catches NON-NEGATIVE-CLIPPING "
            "activations (ReLU, Sigmoid, Softplus, GELU). It would MISS a "
            "stray Tanh (output in `(-1, 1)` — still has negatives) or a "
            "Softmax (output sums to 1 but each entry is non-negative — "
            "would be flagged). For a more complete check, also test the "
            "OUTPUT SUM (Softmax → sums to 1) and MAX (Sigmoid → max <= 1)."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # avgpool-reduce (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "avgpool-reduce",
        "subtopic": "CNN: AvgPool as reduce",
        "topic_folder": TOPIC_EXTRAS,
        "atom_recap_md": RECAP_AVGPOOL_GLOBAL_EX2,
        "exercise_index": 2,
        "exercise_title": "build global avgpool via einops.reduce",
        "slug": "build-global-avgpool-via-einops-reduce",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["avgpool", "global-pool", "einops-reduce", "resnet-head"],
        "kcs": ["global-avgpool-collapse", "adaptive-pool-equivalence"],
        "lo": (
            "Apply `einops.reduce` with a full spatial collapse pattern to "
            "implement global average pooling — turning a `(B, C, H, W)` "
            "feature map into a `(B, C)` tensor — and verify against "
            "`nn.AdaptiveAvgPool2d`."
        ),
        "prompt_body": (
            "Implement `ex2_global_avgpool(x)`. Given input `x: (B, C, H, W)`, "
            "return a `(B, C)` tensor whose entries are the **mean** of each "
            "channel's entire spatial extent.\n\n"
            "**Use einops.reduce with a spatial-collapse pattern.**\n\n"
            "```\n"
            "einops.reduce(x, 'b c h w -> b c', 'mean')\n"
            "```\n\n"
            "The `h` and `w` axes are ABSENT from the right side — meaning "
            "they're reduced away. Compared to the LOCAL avgpool pattern "
            "`'b c (h p1) (w p2) -> b c h w'`, here there's no factoring "
            "because we want to collapse the ENTIRE spatial extent, not "
            "block-by-block.\n\n"
            "**Why this is its own atom.** ResNet's classifier head is "
            "literally `global_avgpool → Linear → loss`. Turning "
            "`(B, 512, 7, 7)` into `(B, 512)` is the bottleneck — once you "
            "have it, the linear layer can produce class logits.\n\n"
            "**Shape contract.** Input `(B, C, H, W)`; output `(B, C)`. The "
            "test deliberately uses non-square `(H, W)` to make sure you're "
            "not accidentally assuming `H == W`.\n\n"
            "The test cross-checks against `nn.AdaptiveAvgPool2d((1, 1))(x)"
            ".squeeze(-1).squeeze(-1)` and confirms identical output."
        ),
        "stub": (
            "def ex2_global_avgpool(x: Tensor) -> Tensor:\n"
            '    """Global average pool: (B, C, H, W) -> (B, C)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# --- Hand-checkable case ---\n"
            "x = t.tensor([[\n"
            "    [[1.0, 2.0], [3.0, 4.0]],     # channel 0: mean = 2.5\n"
            "    [[10.0, 20.0], [30.0, 40.0]], # channel 1: mean = 25.0\n"
            "]])  # shape (1, 2, 2, 2)\n"
            "y = ex2_global_avgpool(x)\n"
            "assert y.shape == (1, 2), f'expected (1, 2), got {tuple(y.shape)}'\n"
            "assert y.dtype == t.float32\n"
            "expected = t.tensor([[2.5, 25.0]])\n"
            "assert t.allclose(y, expected, atol=1e-6), f'value mismatch: {y} vs {expected}'\n"
            "\n"
            "# --- Cross-check against AdaptiveAvgPool2d ---\n"
            "pool = nn.AdaptiveAvgPool2d((1, 1))\n"
            "for B, C, H, W in [(2, 3, 8, 8), (1, 64, 7, 7), (4, 16, 14, 14), (1, 512, 7, 7)]:\n"
            "    xr = t.randn(B, C, H, W, generator=rng)\n"
            "    yr = ex2_global_avgpool(xr)\n"
            "    yref = pool(xr).squeeze(-1).squeeze(-1)\n"
            "    assert yr.shape == yref.shape == (B, C), f'({B},{C},{H},{W}): shape mismatch'\n"
            "    assert t.allclose(yr, yref, atol=1e-5), f'({B},{C},{H},{W}): value mismatch'\n"
            "\n"
            "# --- Non-square spatial dims ---\n"
            "x_ns = t.randn(2, 4, 7, 3, generator=rng)\n"
            "y_ns = ex2_global_avgpool(x_ns)\n"
            "assert y_ns.shape == (2, 4)\n"
            "# Manual check: each output entry is the mean over 7*3 = 21 elements.\n"
            "for b in range(2):\n"
            "    for c in range(4):\n"
            "        assert t.allclose(y_ns[b, c], x_ns[b, c].mean(), atol=1e-5)\n"
            "\n"
            "# --- Constant input → constant output (mean preserved) ---\n"
            "x_c = t.full((2, 3, 6, 6), 7.5)\n"
            "y_c = ex2_global_avgpool(x_c)\n"
            "assert y_c.shape == (2, 3)\n"
            "assert t.allclose(y_c, t.full((2, 3), 7.5), atol=1e-6)\n"
            "\n"
            "# --- H = W = 1 edge case (already 'collapsed') ---\n"
            "x_one = t.randn(1, 5, 1, 1, generator=rng)\n"
            "y_one = ex2_global_avgpool(x_one)\n"
            "assert y_one.shape == (1, 5)\n"
            "assert t.allclose(y_one, x_one.squeeze(-1).squeeze(-1), atol=1e-6)\n"
            "\n"
            "# --- Composability: pool then Linear (ResNet head pattern) ---\n"
            "feat = t.randn(8, 64, 7, 7, generator=rng)\n"
            "pooled = ex2_global_avgpool(feat)            # (8, 64)\n"
            "head = nn.Linear(64, 10)\n"
            "logits = head(pooled)\n"
            "assert logits.shape == (8, 10), 'global pool output must feed nn.Linear directly'"
        ),
        "solution_body": (
            "def ex2_global_avgpool(x: Tensor) -> Tensor:\n"
            "    return einops.reduce(x, 'b c h w -> b c', 'mean')"
        ),
        "solution_notes": (
            "**Why the einops pattern is so terse.** `'b c h w -> b c'` "
            "says: 'keep b and c, drop h and w'. The `'mean'` argument says "
            "how to drop them. One line, no kernel-size or stride to compute "
            "— and it generalizes to N-D pooling by adding more axes.\n\n"
            "**Equivalent rewrites.**\n"
            "- `x.mean(dim=(-2, -1))` — same result, less self-documenting.\n"
            "- `nn.AdaptiveAvgPool2d((1, 1))(x).squeeze(-1).squeeze(-1)` — "
            "PyTorch's idiom; works but has the awkward `.squeeze(-1)` "
            "boilerplate to remove the size-1 spatial dims.\n"
            "- `einops.reduce(x, 'b c h w -> b c ()', 'mean').squeeze(-1)` — "
            "intermediate form; uses anonymous-axis syntax `()` to keep "
            "spatial dims as size-1, then squeezes. The bare pattern (no "
            "spatial axes in output) is cleaner.\n\n"
            "**Why ResNet uses global avgpool.** It collapses spatial extent "
            "into a single per-channel summary statistic, which makes the "
            "model **invariant to spatial position** of features. A cat in "
            "the top-left and a cat in the bottom-right produce the same "
            "global-pool output (modulo translation-equivariance of the "
            "conv layers above). This is a desirable inductive bias for "
            "classification.\n\n"
            "**Contrast with flatten + Linear.** The pre-ResNet pattern was "
            "`flatten(x)` → `Linear(C*H*W, num_classes)`. For a 512-channel "
            "7x7 input, that's `25088 * 1000 ~= 25M` params just for the "
            "head. Global pool reduces this to `512 * 1000 = 512K` — 50× "
            "smaller and immune to spatial reshuffling."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # batchnorm-affine-params (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "batchnorm-affine-params",
        "subtopic": "CNN: BatchNorm affine params",
        "topic_folder": TOPIC_EXTRAS,
        "atom_recap_md": RECAP_BN_RECOVER_EX2,
        "exercise_index": 2,
        "exercise_title": "recover BatchNorm gamma and beta from before-after pairs",
        "slug": "recover-batchnorm-gamma-beta-from-pairs",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["batchnorm", "affine", "inverse", "two-point-recovery"],
        "kcs": ["bn-affine-line-recovery", "bn-per-channel-independence"],
        "lo": (
            "Analyze BatchNorm's per-channel affine map `y = gamma * x_hat "
            "+ beta` by recovering `gamma` and `beta` from two distinct "
            "`(x_hat, y)` pairs per channel — verifying the affine line is "
            "fully determined by two points."
        ),
        "prompt_body": (
            "Implement `ex2_recover_bn_params(x_hat, y)`. Given:\n\n"
            "- `x_hat: (B, C, H, W)` — pre-affine normalized input.\n"
            "- `y: (B, C, H, W)` — post-affine output (you know `y = gamma * "
            "x_hat + beta` was applied per channel).\n\n"
            "Return `(gamma, beta)` — each a 1-D tensor of length `C` — "
            "recovered from the data.\n\n"
            "**Two-point recovery, per channel.**\n\n"
            "For each channel `c`, pick two **distinct** entries within that "
            "channel where `x_hat` has different values. Let them be "
            "`(x0, y0)` and `(x1, y1)`. Then:\n\n"
            "```\n"
            "gamma[c] = (y1 - y0) / (x1 - x0)        # slope of the affine line\n"
            "beta[c]  = y0 - gamma[c] * x0           # y-intercept\n"
            "```\n\n"
            "**The cleanest vectorized form.** Flatten each channel and take "
            "the first two entries (you can assume the test inputs have "
            "distinct `x_hat[:, c, 0, 0]` and `x_hat[:, c, 0, 1]` for every "
            "`c`):\n\n"
            "```\n"
            "x_hat_flat = einops.rearrange(x_hat, 'b c h w -> c (b h w)')\n"
            "y_flat     = einops.rearrange(y,     'b c h w -> c (b h w)')\n"
            "gamma = (y_flat[:, 1] - y_flat[:, 0]) / (x_hat_flat[:, 1] - x_hat_flat[:, 0])\n"
            "beta  = y_flat[:, 0] - gamma * x_hat_flat[:, 0]\n"
            "```\n\n"
            "**Why this works.** An affine map in 1-D is a LINE; two distinct "
            "points uniquely determine a line's slope and intercept. Since "
            "BN's affine is channel-INDEPENDENT, you can do all `C` channels "
            "in parallel (no cross-channel coupling).\n\n"
            "The test runs you against a known `(gamma, beta)` and confirms "
            "your recovered values match to fp tolerance."
        ),
        "stub": (
            "def ex2_recover_bn_params(x_hat: Tensor, y: Tensor):\n"
            '    """Return (gamma, beta) recovered from (x_hat, y) per channel."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rng = t.Generator().manual_seed(0)\n"
            "\n"
            "# --- Known gamma/beta, generate y, recover ---\n"
            "B, C, H, W = 4, 5, 6, 6\n"
            "x_hat = t.randn(B, C, H, W, generator=rng)\n"
            "gamma_true = t.tensor([1.0, 2.5, -0.5, 0.0001, 7.3])\n"
            "beta_true  = t.tensor([0.0, -1.0, 3.14, 0.0, -2.7])\n"
            "y = gamma_true.view(1, -1, 1, 1) * x_hat + beta_true.view(1, -1, 1, 1)\n"
            "\n"
            "gamma_rec, beta_rec = ex2_recover_bn_params(x_hat, y)\n"
            "\n"
            "# --- Shape contract ---\n"
            "assert gamma_rec.shape == (C,), f'gamma shape: {tuple(gamma_rec.shape)}'\n"
            "assert beta_rec.shape  == (C,), f'beta shape: {tuple(beta_rec.shape)}'\n"
            "\n"
            "# --- Value recovery ---\n"
            "assert t.allclose(gamma_rec, gamma_true, atol=1e-5), (\n"
            "    f'gamma mismatch:\\n  recovered {gamma_rec}\\n  expected  {gamma_true}'\n"
            ")\n"
            "assert t.allclose(beta_rec, beta_true, atol=1e-5), (\n"
            "    f'beta mismatch:\\n  recovered {beta_rec}\\n  expected  {beta_true}'\n"
            ")\n"
            "\n"
            "# --- Round-trip: applying recovered params reconstructs y ---\n"
            "y_recon = gamma_rec.view(1, -1, 1, 1) * x_hat + beta_rec.view(1, -1, 1, 1)\n"
            "assert t.allclose(y_recon, y, atol=1e-5), 'gamma_rec, beta_rec must reconstruct y'\n"
            "\n"
            "# --- Identity case: y = x_hat → gamma = ones, beta = zeros ---\n"
            "x2 = t.randn(2, 4, 3, 3, generator=rng)\n"
            "y2 = x2.clone()\n"
            "g2, b2 = ex2_recover_bn_params(x2, y2)\n"
            "assert t.allclose(g2, t.ones(4), atol=1e-5), f'identity gamma should be 1: got {g2}'\n"
            "assert t.allclose(b2, t.zeros(4), atol=1e-5), f'identity beta should be 0: got {b2}'\n"
            "\n"
            "# --- Pure shift: gamma = 1, beta = something ---\n"
            "x3 = t.randn(1, 3, 4, 4, generator=rng)\n"
            "shift = t.tensor([5.0, -10.0, 0.5])\n"
            "y3 = x3 + shift.view(1, -1, 1, 1)\n"
            "g3, b3 = ex2_recover_bn_params(x3, y3)\n"
            "assert t.allclose(g3, t.ones(3), atol=1e-5)\n"
            "assert t.allclose(b3, shift, atol=1e-5)\n"
            "\n"
            "# --- Pure scale: gamma = something, beta = 0 ---\n"
            "x4 = t.randn(1, 3, 4, 4, generator=rng)\n"
            "scale = t.tensor([2.0, -1.5, 0.25])\n"
            "y4 = scale.view(1, -1, 1, 1) * x4\n"
            "g4, b4 = ex2_recover_bn_params(x4, y4)\n"
            "assert t.allclose(g4, scale, atol=1e-5)\n"
            "assert t.allclose(b4, t.zeros(3), atol=1e-5)\n"
            "\n"
            "# --- Large C (parallel recovery across many channels) ---\n"
            "C_big = 32\n"
            "x5 = t.randn(2, C_big, 5, 5, generator=rng)\n"
            "g5_true = t.randn(C_big, generator=rng)\n"
            "b5_true = t.randn(C_big, generator=rng)\n"
            "y5 = g5_true.view(1, -1, 1, 1) * x5 + b5_true.view(1, -1, 1, 1)\n"
            "g5, b5 = ex2_recover_bn_params(x5, y5)\n"
            "assert t.allclose(g5, g5_true, atol=1e-5)\n"
            "assert t.allclose(b5, b5_true, atol=1e-5)"
        ),
        "solution_body": (
            "def ex2_recover_bn_params(x_hat: Tensor, y: Tensor):\n"
            "    x_flat = einops.rearrange(x_hat, 'b c h w -> c (b h w)')\n"
            "    y_flat = einops.rearrange(y,     'b c h w -> c (b h w)')\n"
            "    # Two distinct points per channel — first two entries of the flat axis.\n"
            "    x0 = x_flat[:, 0]\n"
            "    x1 = x_flat[:, 1]\n"
            "    y0 = y_flat[:, 0]\n"
            "    y1 = y_flat[:, 1]\n"
            "    gamma = (y1 - y0) / (x1 - x0)\n"
            "    beta  = y0 - gamma * x0\n"
            "    return gamma, beta"
        ),
        "solution_notes": (
            "**Why two points suffice.** An affine map `y = gamma * x + "
            "beta` has exactly TWO free parameters (slope and intercept). "
            "Two distinct `(x, y)` pairs give two equations in those two "
            "unknowns — uniquely solvable. Any THIRD point must lie on the "
            "recovered line (a hidden invariant the test could check).\n\n"
            "**Why the rearrange pattern.** `'b c h w -> c (b h w)'` "
            "flattens the (B, H, W) axes per channel while keeping C as "
            "the leading axis. The result `(C, B*H*W)` lets us index "
            "`[:, 0]` and `[:, 1]` to get two entries per channel "
            "simultaneously — all `C` recoveries happen as a vectorized "
            "computation.\n\n"
            "**A more robust alternative** (out of scope here) would be "
            "least-squares regression per channel: fit `(gamma, beta)` to "
            "ALL entries in that channel via "
            "`gamma, beta = lstsq([x, 1], y)`. Two-point is faster but "
            "amplifies noise if `x0` and `x1` happen to be very close. For "
            "exact arithmetic (our test case), both produce identical "
            "results.\n\n"
            "**Channel independence is what makes this work.** If BN had "
            "cross-channel coupling (like `y[c] = gamma[c, c'] * x[c'] + "
            "beta[c]`), you couldn't recover params channel-by-channel — "
            "you'd need joint regression over all `C` channels. The strict "
            "per-channel affine is what enables the trivial recovery.\n\n"
            "**The connection back to BN's forward.** This drill exercises "
            "INVERTING the affine. The forward `gamma * x_hat + beta` and "
            "this inverse are duals — together they show that the affine "
            "stage is a bijection (modulo `gamma = 0` channels, which "
            "collapse to a constant)."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # block-group-stack (ex2)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "block-group-stack",
        "subtopic": "CNN: BlockGroup stack",
        "topic_folder": TOPIC_EXTRAS,
        "atom_recap_md": RECAP_BLOCK_GROUP_INTROSPECT_EX2,
        "exercise_index": 2,
        "exercise_title": "introspect a BlockGroup and verify the shape-change invariant",
        "slug": "introspect-blockgroup-shape-change-invariant",
        "bloom_level": "Evaluate",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["resnet", "block-group", "introspection", "invariant"],
        "kcs": ["block-group-one-shape-changer", "block-group-output-width-consistency"],
        "lo": (
            "Evaluate a `nn.Sequential` of `ResBlock`s against the canonical "
            "BlockGroup invariant (exactly one shape-changing block at index "
            "0, all subsequent blocks identity-shaped) by walking children "
            "and returning a structured audit."
        ),
        "prompt_body": (
            "A toy `ResBlock` is provided (same as the ex1 drill):\n\n"
            "```\n"
            "class ResBlock(nn.Module):\n"
            "    def __init__(self, in_feats, out_feats, first_stride=1):\n"
            "        super().__init__()\n"
            "        self.in_feats     = in_feats\n"
            "        self.out_feats    = out_feats\n"
            "        self.first_stride = first_stride\n"
            "        self.proj = nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride)\n"
            "    def forward(self, x):\n"
            "        return self.proj(x)\n"
            "```\n\n"
            "Implement `ex2_audit_block_group(group)`. Given a `nn.Sequential` "
            "of `ResBlock`s, return a dict:\n\n"
            "```python\n"
            "{\n"
            "  'n_blocks':              int,                       # total blocks in the group\n"
            "  'shape_changers':        List[int],                 # indices of blocks where in_feats != out_feats OR first_stride != 1\n"
            "  'is_canonical':          bool,                      # True iff shape_changers == [0] (and group is non-empty)\n"
            "  'output_width':          int,                       # group[-1].out_feats\n"
            "  'output_width_consistent': bool,                    # True iff EVERY block has out_feats == output_width\n"
            "}\n"
            "```\n\n"
            "**The canonical invariant** is: exactly ONE shape-changing "
            "block, and it's at index 0. Every subsequent block has "
            "`in_feats == out_feats` AND `first_stride == 1`. The "
            "introspection walks `group` (which iterates its children) and "
            "tabulates the facts.\n\n"
            "**Edge cases.**\n"
            "- `len(group) == 1` → `shape_changers` is `[0]` if that block "
            "is a shape-changer, else `[]`. `is_canonical` is True iff "
            "`shape_changers == [0]`.\n"
            "- Empty `nn.Sequential` → `n_blocks == 0`, `shape_changers == "
            "[]`, `is_canonical == False`, `output_width_consistent == True` "
            "(vacuous).\n\n"
            "The test instantiates correctly-built groups, deliberately-"
            "miswired ones (downsample in the middle, mismatched widths), "
            "and confirms your audit catches each case."
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
            "def ex2_audit_block_group(group) -> dict:\n"
            '    """Return {n_blocks, shape_changers, is_canonical, output_width, output_width_consistent}."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "\n"
            "# --- Canonical group: first block changes, rest identity ---\n"
            "good = nn.Sequential(\n"
            "    ResBlock(64, 128, first_stride=2),\n"
            "    ResBlock(128, 128, first_stride=1),\n"
            "    ResBlock(128, 128, first_stride=1),\n"
            "    ResBlock(128, 128, first_stride=1),\n"
            ")\n"
            "audit = ex2_audit_block_group(good)\n"
            "assert audit['n_blocks'] == 4\n"
            "assert audit['shape_changers'] == [0], f'expected [0], got {audit[\"shape_changers\"]}'\n"
            "assert audit['is_canonical'] is True\n"
            "assert audit['output_width'] == 128\n"
            "assert audit['output_width_consistent'] is True\n"
            "\n"
            "# --- Stage-0-style group: no shape change at all ---\n"
            "stage0 = nn.Sequential(\n"
            "    ResBlock(64, 64, first_stride=1),\n"
            "    ResBlock(64, 64, first_stride=1),\n"
            "    ResBlock(64, 64, first_stride=1),\n"
            ")\n"
            "audit0 = ex2_audit_block_group(stage0)\n"
            "assert audit0['n_blocks'] == 3\n"
            "assert audit0['shape_changers'] == [], 'no block changes shape'\n"
            "assert audit0['is_canonical'] is False, 'canonical = exactly [0]; empty is non-canonical'\n"
            "assert audit0['output_width'] == 64\n"
            "assert audit0['output_width_consistent'] is True\n"
            "\n"
            "# --- Single-block group: just the shape-changer ---\n"
            "single = nn.Sequential(ResBlock(32, 64, first_stride=2))\n"
            "audit_single = ex2_audit_block_group(single)\n"
            "assert audit_single['n_blocks'] == 1\n"
            "assert audit_single['shape_changers'] == [0]\n"
            "assert audit_single['is_canonical'] is True\n"
            "assert audit_single['output_width'] == 64\n"
            "assert audit_single['output_width_consistent'] is True\n"
            "\n"
            "# --- BROKEN: downsample in the middle ---\n"
            "broken_mid = nn.Sequential(\n"
            "    ResBlock(64, 128, first_stride=2),\n"
            "    ResBlock(128, 128, first_stride=1),\n"
            "    ResBlock(128, 128, first_stride=2),    # ← stride > 1 in middle, bug\n"
            "    ResBlock(128, 128, first_stride=1),\n"
            ")\n"
            "audit_bm = ex2_audit_block_group(broken_mid)\n"
            "assert audit_bm['shape_changers'] == [0, 2], f'should catch both: {audit_bm[\"shape_changers\"]}'\n"
            "assert audit_bm['is_canonical'] is False, 'two shape-changers → non-canonical'\n"
            "assert audit_bm['output_width_consistent'] is True, '128 throughout'\n"
            "\n"
            "# --- BROKEN: mismatched widths ---\n"
            "broken_widths = nn.Sequential(\n"
            "    ResBlock(64, 128, first_stride=2),\n"
            "    ResBlock(128, 256, first_stride=1),   # ← widens in middle, bug\n"
            "    ResBlock(256, 256, first_stride=1),\n"
            ")\n"
            "audit_bw = ex2_audit_block_group(broken_widths)\n"
            "assert audit_bw['shape_changers'] == [0, 1]\n"
            "assert audit_bw['is_canonical'] is False\n"
            "assert audit_bw['output_width'] == 256\n"
            "assert audit_bw['output_width_consistent'] is False, 'block 0 outputs 128, group output is 256'\n"
            "\n"
            "# --- Empty Sequential ---\n"
            "empty = nn.Sequential()\n"
            "audit_empty = ex2_audit_block_group(empty)\n"
            "assert audit_empty['n_blocks'] == 0\n"
            "assert audit_empty['shape_changers'] == []\n"
            "assert audit_empty['is_canonical'] is False, 'empty cannot be canonical'\n"
            "assert audit_empty['output_width_consistent'] is True, 'vacuously true'"
        ),
        "solution_body": (
            "def ex2_audit_block_group(group) -> dict:\n"
            "    blocks = list(group)\n"
            "    n = len(blocks)\n"
            "    shape_changers = [\n"
            "        i for i, b in enumerate(blocks)\n"
            "        if b.in_feats != b.out_feats or b.first_stride != 1\n"
            "    ]\n"
            "    if n == 0:\n"
            "        return {\n"
            "            'n_blocks': 0,\n"
            "            'shape_changers': [],\n"
            "            'is_canonical': False,\n"
            "            'output_width': 0,\n"
            "            'output_width_consistent': True,\n"
            "        }\n"
            "    output_width = blocks[-1].out_feats\n"
            "    consistent = all(b.out_feats == output_width for b in blocks)\n"
            "    return {\n"
            "        'n_blocks': n,\n"
            "        'shape_changers': shape_changers,\n"
            "        'is_canonical': shape_changers == [0],\n"
            "        'output_width': output_width,\n"
            "        'output_width_consistent': consistent,\n"
            "    }"
        ),
        "solution_notes": (
            "**Why `is_canonical` requires `[0]` exactly.** Two failure "
            "modes both produce non-canonical groups:\n"
            "1. `shape_changers == []` — no block changes shape; the group "
            "isn't doing anything structural.\n"
            "2. `shape_changers == [0, 2, ...]` — multiple shape-changers; "
            "the group has been mis-stacked.\n\n"
            "Both should fail the canonical check; the `== [0]` predicate "
            "captures both cases in one line.\n\n"
            "**Why `output_width_consistent` is its own field.** A group can "
            "have `is_canonical == True` AND `output_width_consistent == "
            "True` simultaneously (the well-formed case), but they're "
            "*independently* useful diagnostics. Some bugs only break "
            "consistency (mismatched widths through the middle) without "
            "adding a second shape-changer.\n\n"
            "**Why `list(group)` instead of `for b in group`.** "
            "`nn.Sequential` is iterable but its `__len__` is well-defined; "
            "`list(group)` makes the iteration and indexing explicit. For "
            "`enumerate(blocks)` and `blocks[-1]`, the list form is "
            "clearer than working with the underlying iterator.\n\n"
            "**Empty-group edge case.** `n_blocks == 0` returns "
            "`is_canonical = False` (an empty group is trivially "
            "non-canonical — it can't have a shape-changer at index 0). "
            "`output_width_consistent = True` is vacuously true (no blocks "
            "to disagree). `output_width = 0` is a sentinel; the test "
            "doesn't rely on a specific value here, just that the call "
            "doesn't crash."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Verifier — re-runs each spec's solution against its test_body in-process.
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"[deepening_n_batch10] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_n_batch10] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_n_batch10] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
