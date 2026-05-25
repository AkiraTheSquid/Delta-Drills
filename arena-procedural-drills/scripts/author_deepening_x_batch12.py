#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 12, group X).

Atoms (6 numerical-modules + 2 optimizer-internals):
    - encoder-decoder-symmetric    (ex2: detect a BROKEN symmetric layout — mismatched stride/upsample)
    - kaiming-uniform-init         (ex2: contrast fan_in vs fan_out mode on a Conv2d weight)
    - loss-item-scalar-extract     (ex2: .item() on non-scalar raises; .mean().item() works)
    - rearrange-as-sequential-layer(ex2: NHWC patch-flatten via Rearrange — patchify ordering vs channel-major)
    - sqrt-eps-stabilize           (ex2: eps INSIDE sqrt vs OUTSIDE — only inside is stable at var=0)
    - stride-zero-broadcast        (ex2: detect zero-stride view, call .contiguous(), show stride+storage change)
    - clip-grad-norm-pre-step      (ex2: reimplement clip_grad_norm_ from scratch with global L2)
    - ema-second-moment            (ex2: derive Adam's per-coord adaptive step-scale = 1 / (sqrt(v) + eps))

Each ex2 hits a DISTINCT deepening facet from ex1. ONE LO + ONE Bloom + <=2 KCs.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_NUM = "prereqs_numerical_modules"
TOPIC_OPT = "prereqs_optimizer_internals"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_ENCDEC_BREAK = (
    "## Encoder-decoder symmetry — what BREAKS it\n"
    "\n"
    "Ex1 BUILT a symmetric autoencoder where each encoder pool was mirrored by "
    "a decoder upsample. The deepening move is to ANALYZE a given config and "
    "report whether end-to-end shape is preserved — and if not, where the "
    "asymmetry lives.\n"
    "\n"
    "**Per-stage shape arithmetic.** For each encoder stage with stride `s`, "
    "the spatial size divides by `s` (integer division). For each decoder "
    "stage with upsample factor `u`, the spatial size multiplies by `u`. End-"
    "to-end shape is preserved iff the product of encoder strides == product "
    "of decoder upsamples AND the input H/W is divisible by the encoder "
    "stride-product.\n"
    "\n"
    "```python\n"
    "enc_div = 1\n"
    "for s in encoder_strides:\n"
    "    enc_div *= s\n"
    "dec_mul = 1\n"
    "for u in decoder_upsamples:\n"
    "    dec_mul *= u\n"
    "# Symmetric iff enc_div == dec_mul AND H % enc_div == 0.\n"
    "```\n"
    "\n"
    "**Why divisibility matters even when products match.** A stride-2 conv "
    "on a 7×7 input rounds down to 3×3. Upsampling 3×3 by 2 gives 6×6 — NOT "
    "the original 7×7. The product matches but the intermediate truncation "
    "destroys the round-trip. This is the silent bug: shapes look 'symmetric' "
    "but the autoencoder output is off-by-one."
)

RECAP_KAIMING_FAN = (
    "## Kaiming uniform — fan_in vs fan_out\n"
    "\n"
    "Ex1 initialized a Linear weight with bound `1/sqrt(fan_in)`. The "
    "deepening move is `fan_out` mode and a CONVOLUTIONAL weight where "
    "fan_in ≠ fan_out.\n"
    "\n"
    "For a `Conv2d(in_ch, out_ch, kernel=k)` weight of shape `(out_ch, "
    "in_ch, k, k)`:\n"
    "- `fan_in  = in_ch  * k * k` — receptive field × input channels\n"
    "- `fan_out = out_ch * k * k` — receptive field × output channels\n"
    "\n"
    "```python\n"
    "# Kaiming-uniform bound = gain * sqrt(3 / fan)\n"
    "# For relu nonlinearity, gain = sqrt(2).\n"
    "# So bound = sqrt(2) * sqrt(3 / fan) = sqrt(6 / fan).\n"
    "bound_in  = math.sqrt(6.0 / fan_in)\n"
    "bound_out = math.sqrt(6.0 / fan_out)\n"
    "```\n"
    "\n"
    "**Why two modes exist.** `fan_in` preserves the variance of activations "
    "on the FORWARD pass; `fan_out` preserves it on the BACKWARD pass. For a "
    "Conv expanding 3→64 channels, `fan_in=27` and `fan_out=576` produce "
    "DIFFERENT bounds — `fan_out` mode shrinks weights ~4.6× more.\n"
    "\n"
    "**Default in PyTorch.** `nn.init.kaiming_uniform_` defaults to "
    "`mode='fan_in', nonlinearity='leaky_relu', a=sqrt(5)` for legacy "
    "compatibility. Most modern code passes `nonlinearity='relu'` explicitly."
)

RECAP_LOSS_ITEM_RAISES = (
    "## `.item()` requires a 0-d (scalar) tensor\n"
    "\n"
    "Ex1 used `.item()` on a scalar loss. The deepening move is the FAILURE "
    "mode: `.item()` on a tensor with `numel() > 1` raises `RuntimeError`.\n"
    "\n"
    "```python\n"
    ">>> t.tensor([1.0, 2.0]).item()\n"
    "RuntimeError: a Tensor with 2 elements cannot be converted to Scalar\n"
    "```\n"
    "\n"
    "**The standard fix.** Reduce to a scalar first — `.mean().item()`, "
    "`.sum().item()`, `.max().item()`. Each picks a SCALAR aggregate; "
    "`.item()` then extracts the Python float.\n"
    "\n"
    "**Why this matters for logging.** A per-sample loss tensor `(B,)` is "
    "what `F.cross_entropy(reduction='none')` returns. Trying to log it "
    "directly with `.item()` crashes — you'd want `.mean().item()` (average "
    "over the batch) or iterate. The error message is helpful but only if "
    "you read it."
)

RECAP_REARRANGE_PATCHIFY = (
    "## Rearrange for patchify — `(b c (h ph) (w pw)) -> b (h w) (ph pw c)`\n"
    "\n"
    "Ex1 used `Rearrange` as a flatten-to-(B, C·H·W) layer inside "
    "`nn.Sequential`. The deepening move is the CANONICAL ViT/transformer "
    "patchify rearrange — split an image into non-overlapping patches and "
    "flatten each patch.\n"
    "\n"
    "```python\n"
    "from einops.layers.torch import Rearrange\n"
    "patchify = Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=2, pw=2)\n"
    "# (B, 3, 8, 8) -> (B, 16, 12)   # 16 patches of (2·2·3) = 12 features\n"
    "```\n"
    "\n"
    "**Why the inner ordering `(ph pw c)` matters.** Inside each patch, "
    "einops flattens in left-to-right order. `(ph pw c)` walks pixel-by-"
    "pixel then channel-by-channel — NHWC ordering. Reordering to "
    "`(c ph pw)` gives channel-major, which is what PyTorch tensors use "
    "natively but is NOT what a HuggingFace ViT expects. Wrong ordering "
    "leads to silent feature-mismatch at the linear projection.\n"
    "\n"
    "**Why this composes inside `nn.Sequential`.** Same trick as ex1 — "
    "`Rearrange` is an `nn.Module`, so it slots between conv stages "
    "without a custom forward. The next `Linear(patch_dim, embed_dim)` "
    "sees `(B, num_patches, patch_dim)` directly."
)

RECAP_SQRT_EPS_PLACEMENT = (
    "## Where to place `eps` — INSIDE sqrt vs OUTSIDE\n"
    "\n"
    "Ex1 used `sqrt(var + eps)` and showed it survives a zero-variance "
    "channel. The deepening move is the CONTRAST: what about `sqrt(var) + "
    "eps`? Same eps, just outside the sqrt.\n"
    "\n"
    "```python\n"
    "# INSIDE  — stable at var=0:\n"
    "sigma_in  = (var + eps).sqrt()   # sqrt(eps) ≈ 3.16e-3 when eps=1e-5\n"
    "# OUTSIDE — UNSTABLE at var=0:\n"
    "sigma_out = var.sqrt() + eps     # sqrt(0) + eps = eps ≈ 1e-5\n"
    "```\n"
    "\n"
    "**Both are finite at var=0** — neither divides by zero — but they "
    "give VERY different normalizers. With `eps=1e-5` and a constant "
    "channel (var=0, x - mean = 0):\n"
    "- inside:  `0 / 3.16e-3 = 0` — clean.\n"
    "- outside: `0 / 1e-5 = 0` — also clean here, but...\n"
    "\n"
    "**The OUTSIDE placement breaks the gradient.** `d/dvar sqrt(var)` is "
    "`1/(2*sqrt(var))` — infinite at `var=0`. Backprop through "
    "`sqrt(var) + eps` produces a non-finite gradient even though the "
    "forward value is finite. INSIDE placement keeps the derivative bounded "
    "because `d/dvar sqrt(var+eps) = 1/(2*sqrt(var+eps))` is at most "
    "`1/(2*sqrt(eps))`, never infinite.\n"
    "\n"
    "**This is why PyTorch's BatchNorm/LayerNorm use INSIDE placement.** "
    "It's not about NaN in the forward — it's about NaN in the BACKWARD."
)

RECAP_STRIDE_CONTIGUOUS = (
    "## Zero-stride detection → `.contiguous()` materialization\n"
    "\n"
    "Ex1 distinguished `.expand()` (zero-stride view) from `.repeat()` (true "
    "copy). The deepening move: given an ARBITRARY tensor, detect whether "
    "any axis has stride 0, then call `.contiguous()` and verify the "
    "materialization happened.\n"
    "\n"
    "```python\n"
    "x = t.arange(3).expand(4, 3)        # shape (4, 3), stride (0, 1)\n"
    "has_broadcast = any(s == 0 for s in x.stride())   # True\n"
    "y = x.contiguous()                  # forces a copy\n"
    "# y has stride (3, 1) and y.storage().nbytes() == 4*3 * elem_size\n"
    "all(s != 0 for s in y.stride())   # True\n"
    "x.data_ptr() != y.data_ptr()       # True — different storage\n"
    "```\n"
    "\n"
    "**Why `.contiguous()` after a broadcast.** Many kernels (e.g. "
    "`view`, MKL/cuDNN-backed ops) require contiguous input. A zero-stride "
    "broadcast LOOKS like a tensor of the right shape but is actually a "
    "pinned 1-D buffer being indexed. `.contiguous()` is the canonical "
    "fix — it allocates fresh storage, copies the broadcasted values, and "
    "returns a stride-`(M, 1)` tensor.\n"
    "\n"
    "**Storage size as a diagnostic.** Pre-contiguous, `x.storage().nbytes()` "
    "reflects only the SOURCE 1-D buffer (3 elements). Post-contiguous, it "
    "reflects the full materialized shape (12 elements). Comparing these is "
    "the cheap way to confirm 'yes, the broadcast was actually copied'."
)

RECAP_CLIP_FROM_SCRATCH = (
    "## Reimplement `clip_grad_norm_` from scratch\n"
    "\n"
    "Ex1 called `torch.nn.utils.clip_grad_norm_`. The deepening move is to "
    "WRITE the same function from scratch over a list of parameters with "
    "`.grad` attributes. Knowing the internals lets you debug clipping bugs "
    "and write per-group variants.\n"
    "\n"
    "Algorithm (matches PyTorch's reference):\n"
    "```python\n"
    "# 1. Collect grads (skip params with .grad is None).\n"
    "grads = [p.grad for p in params if p.grad is not None]\n"
    "if not grads:\n"
    "    return 0.0\n"
    "# 2. Compute global L2 norm = sqrt(sum_i ||g_i||^2).\n"
    "total = t.sqrt(sum(g.detach().pow(2).sum() for g in grads))\n"
    "# 3. Compute scale; ONLY apply if total > max_norm.\n"
    "if total > max_norm:\n"
    "    scale = max_norm / (total + 1e-6)\n"
    "    for g in grads:\n"
    "        g.mul_(scale)\n"
    "return total.item()\n"
    "```\n"
    "\n"
    "**Return PRE-clip norm.** Same contract as the library function — the "
    "value returned is what the norm WAS, not what it became. This is what "
    "you log to monitor training stability.\n"
    "\n"
    "**Skip `None` grads.** Some params (frozen layers, params with no path "
    "to the loss) never have a grad. Including `None` in the sum would "
    "crash. Skipping them gives the same answer as PyTorch.\n"
    "\n"
    "**In-place `g.mul_(scale)`.** Don't allocate; the optimizer reads "
    "`.grad` by reference after this returns."
)

RECAP_EMA_ADAPTIVE_SCALE = (
    "## Adam adaptive step scale — `1 / (sqrt(v) + eps)`\n"
    "\n"
    "Ex1 maintained the EMA second-moment buffer `v`. The deepening move is "
    "the next downstream step: derive the PER-COORDINATE step-scale that "
    "the Adam denominator produces.\n"
    "\n"
    "```python\n"
    "# After computing v_t = beta2*v + (1-beta2)*g^2:\n"
    "step_scale = 1.0 / (v.sqrt() + eps)\n"
    "# Coordinate-wise adaptive lr multiplier.\n"
    "# Big |g| history → big v → SMALL step_scale.\n"
    "# Small |g| history → small v → LARGE step_scale.\n"
    "```\n"
    "\n"
    "**Why eps is OUTSIDE the sqrt here (unlike BatchNorm).** Adam's eps "
    "primarily prevents divide-by-zero when v=0 (early steps before any "
    "grad accumulation). The PyTorch reference (and the original Kingma & "
    "Ba paper) uses `sqrt(v) + eps`. This is a deliberate departure from "
    "the BatchNorm convention — see the paper's appendix for the "
    "mean-square-error analysis.\n"
    "\n"
    "**Operational meaning.** A coordinate with v=100 gets step_scale ≈ "
    "0.1 — gradients are LARGE on this axis, so Adam takes SMALL steps. A "
    "coordinate with v=0.01 gets step_scale ≈ 10 — gradients are small on "
    "this axis, so Adam takes LARGE steps. That's adaptive per-parameter "
    "lr — the whole point of Adam over SGD.\n"
    "\n"
    "**Watch out for v=0.** If v hasn't accumulated yet (or all observed "
    "g were 0), `step_scale = 1/eps` is huge. With eps=1e-8, that's a 1e8 "
    "lr multiplier — explosive. This is why Adam normally has a "
    "bias-correction step BEFORE the denominator AND why eps is usually "
    "set non-trivially (1e-7 or 1e-8, not 1e-30)."
)


# ---------------------------------------------------------------------------
# SPEC 1 — encoder-decoder-symmetric ex2
# ---------------------------------------------------------------------------

SPEC_ENCDEC_BREAK = {
    "atom_id": "encoder-decoder-symmetric",
    "subtopic": "CNN: Encoder-decoder symmetric layout",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_ENCDEC_BREAK,
    "exercise_index": 2,
    "exercise_title": "diagnose a broken encoder-decoder config and report the asymmetry",
    "slug": "diagnose-broken-encoder-decoder-asymmetry",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["encoder-decoder", "symmetry", "shape-arithmetic", "diagnostic"],
    "kcs": [
        "stride-product-must-equal-upsample-product",
        "input-divisible-by-stride-product",
    ],
    "lo": (
        "Analyze an (encoder_strides, decoder_upsamples, input_size) config "
        "and return a diagnostic dict reporting whether end-to-end shape is "
        "preserved — and if not, whether the failure is a product mismatch, "
        "a divisibility violation, or both."
    ),
    "prompt_body": (
        "Implement `ex2_diagnose_symmetry(encoder_strides, decoder_upsamples, "
        "input_hw)`. The deepening variant of ex1.\n\n"
        "Inputs:\n"
        "- `encoder_strides`: `list[int]`, e.g. `[2, 2]` (two pool stages).\n"
        "- `decoder_upsamples`: `list[int]`, e.g. `[2, 2]`.\n"
        "- `input_hw`: `int`, the H = W of a square input.\n\n"
        "Return a dict with EXACTLY these keys:\n\n"
        "- `'enc_div'`: `int`, product of `encoder_strides` (1 if list is empty).\n"
        "- `'dec_mul'`: `int`, product of `decoder_upsamples` (1 if list is empty).\n"
        "- `'product_matches'`: `bool`, `enc_div == dec_mul`.\n"
        "- `'input_divisible'`: `bool`, `input_hw % enc_div == 0`.\n"
        "- `'predicted_output_hw'`: `int`, the spatial size you'd get if you "
        "ran this config: `(input_hw // enc_div) * dec_mul` — floor-division "
        "on encoder, multiplication on decoder.\n"
        "- `'shape_preserved'`: `bool`, `predicted_output_hw == input_hw`. "
        "True iff BOTH `product_matches` and `input_divisible`.\n"
        "- `'failure_reason'`: `str | None`. `None` if `shape_preserved` is "
        "True. Otherwise one of `'product_mismatch'`, `'not_divisible'`, "
        "or `'both'` (when product matches but divisibility fails AND there's "
        "a product mismatch, return `'both'`).\n\n"
        "Do not raise on empty lists — treat empty as product 1 (identity)."
    ),
    "stub": (
        "def ex2_diagnose_symmetry(encoder_strides: list, decoder_upsamples: list, input_hw: int) -> dict:\n"
        '    """Return a shape-symmetry diagnostic dict for the given config."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Symmetric, divisible: shape preserved ===\n"
        "d = ex2_diagnose_symmetry([2, 2], [2, 2], 8)\n"
        "assert d['enc_div'] == 4 and d['dec_mul'] == 4\n"
        "assert d['product_matches'] is True\n"
        "assert d['input_divisible'] is True\n"
        "assert d['predicted_output_hw'] == 8\n"
        "assert d['shape_preserved'] is True\n"
        "assert d['failure_reason'] is None\n"
        "\n"
        "# === Product mismatch (more encoder than decoder) ===\n"
        "d = ex2_diagnose_symmetry([2, 2, 2], [2, 2], 32)\n"
        "assert d['enc_div'] == 8 and d['dec_mul'] == 4\n"
        "assert d['product_matches'] is False\n"
        "assert d['input_divisible'] is True   # 32 % 8 == 0\n"
        "# Predicted: 32 // 8 = 4; 4 * 4 = 16. NOT 32.\n"
        "assert d['predicted_output_hw'] == 16\n"
        "assert d['shape_preserved'] is False\n"
        "assert d['failure_reason'] == 'product_mismatch'\n"
        "\n"
        "# === Divisibility violation (product matches but H not divisible) ===\n"
        "d = ex2_diagnose_symmetry([2, 2], [2, 2], 7)\n"
        "assert d['enc_div'] == 4 and d['dec_mul'] == 4\n"
        "assert d['product_matches'] is True\n"
        "assert d['input_divisible'] is False   # 7 % 4 != 0\n"
        "# Predicted: 7 // 4 = 1; 1 * 4 = 4. NOT 7.\n"
        "assert d['predicted_output_hw'] == 4\n"
        "assert d['shape_preserved'] is False\n"
        "assert d['failure_reason'] == 'not_divisible'\n"
        "\n"
        "# === Both failures at once ===\n"
        "d = ex2_diagnose_symmetry([2, 2, 2], [2, 2], 9)\n"
        "assert d['enc_div'] == 8 and d['dec_mul'] == 4\n"
        "assert d['product_matches'] is False\n"
        "assert d['input_divisible'] is False   # 9 % 8 != 0\n"
        "assert d['shape_preserved'] is False\n"
        "assert d['failure_reason'] == 'both'\n"
        "\n"
        "# === Empty encoder + empty decoder = identity ===\n"
        "d = ex2_diagnose_symmetry([], [], 16)\n"
        "assert d['enc_div'] == 1 and d['dec_mul'] == 1\n"
        "assert d['shape_preserved'] is True\n"
        "assert d['failure_reason'] is None\n"
        "assert d['predicted_output_hw'] == 16\n"
        "\n"
        "# === Empty encoder, non-empty decoder = product mismatch ===\n"
        "d = ex2_diagnose_symmetry([], [2], 16)\n"
        "assert d['enc_div'] == 1 and d['dec_mul'] == 2\n"
        "assert d['product_matches'] is False\n"
        "assert d['input_divisible'] is True   # 16 % 1 == 0\n"
        "assert d['shape_preserved'] is False\n"
        "assert d['failure_reason'] == 'product_mismatch'\n"
        "\n"
        "# === Three-stage symmetric with divisible H ===\n"
        "d = ex2_diagnose_symmetry([2, 2, 2], [2, 2, 2], 64)\n"
        "assert d['enc_div'] == 8 and d['dec_mul'] == 8\n"
        "assert d['shape_preserved'] is True\n"
        "assert d['failure_reason'] is None\n"
        "\n"
        "# === Non-uniform strides ===\n"
        "d = ex2_diagnose_symmetry([2, 4], [4, 2], 32)\n"
        "assert d['enc_div'] == 8 and d['dec_mul'] == 8\n"
        "assert d['shape_preserved'] is True\n"
        "\n"
        "# === All returned keys exactly ===\n"
        "expected_keys = {'enc_div', 'dec_mul', 'product_matches', 'input_divisible',\n"
        "                 'predicted_output_hw', 'shape_preserved', 'failure_reason'}\n"
        "assert set(ex2_diagnose_symmetry([2], [2], 4).keys()) == expected_keys"
    ),
    "solution_body": (
        "def ex2_diagnose_symmetry(encoder_strides, decoder_upsamples, input_hw):\n"
        "    enc_div = 1\n"
        "    for s in encoder_strides:\n"
        "        enc_div *= s\n"
        "    dec_mul = 1\n"
        "    for u in decoder_upsamples:\n"
        "        dec_mul *= u\n"
        "    product_matches = (enc_div == dec_mul)\n"
        "    input_divisible = (input_hw % enc_div == 0)\n"
        "    predicted = (input_hw // enc_div) * dec_mul\n"
        "    shape_preserved = product_matches and input_divisible\n"
        "    if shape_preserved:\n"
        "        failure_reason = None\n"
        "    elif not product_matches and not input_divisible:\n"
        "        failure_reason = 'both'\n"
        "    elif not product_matches:\n"
        "        failure_reason = 'product_mismatch'\n"
        "    else:\n"
        "        failure_reason = 'not_divisible'\n"
        "    return {\n"
        "        'enc_div': enc_div,\n"
        "        'dec_mul': dec_mul,\n"
        "        'product_matches': product_matches,\n"
        "        'input_divisible': input_divisible,\n"
        "        'predicted_output_hw': predicted,\n"
        "        'shape_preserved': shape_preserved,\n"
        "        'failure_reason': failure_reason,\n"
        "    }"
    ),
    "solution_notes": (
        "**Why predicted uses `//` not `/`.** Conv with stride truncates — "
        "a 7×7 input with stride-2 gives 3×3, not 3.5×3.5. The "
        "`predicted_output_hw` formula has to mirror that truncation; "
        "otherwise the diagnostic disagrees with real model behavior.\n\n"
        "**`'both'` is a distinct category.** A config with BOTH a product "
        "mismatch AND a non-divisible input fails in two ways. Reporting "
        "just one would let the user fix it (e.g. balance the upsamples) "
        "and then hit the second error in the next iteration. `'both'` is "
        "the 'fix both before re-running' signal.\n\n"
        "**Empty list → product 1.** The neutral element of multiplication. "
        "No encoder/decoder stages == identity transformation. The check "
        "still works because `input_hw % 1 == 0` always."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — kaiming-uniform-init ex2
# ---------------------------------------------------------------------------

SPEC_KAIMING_FAN = {
    "atom_id": "kaiming-uniform-init",
    "subtopic": "Init: Kaiming uniform",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_KAIMING_FAN,
    "exercise_index": 2,
    "exercise_title": "Kaiming-uniform Conv2d init in fan_in vs fan_out mode",
    "slug": "kaiming-uniform-conv2d-fan-in-vs-fan-out",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["kaiming", "conv2d", "fan_in", "fan_out", "init"],
    "kcs": [
        "conv-weight-fan-in-vs-fan-out-formula",
        "kaiming-uniform-relu-bound",
    ],
    "lo": (
        "Apply the Kaiming-uniform bound `sqrt(6/fan)` to a Conv2d weight "
        "in BOTH `'fan_in'` and `'fan_out'` mode, returning the two "
        "initialized weight tensors and their empirical max-abs values for "
        "comparison."
    ),
    "prompt_body": (
        "Implement `ex2_kaiming_conv_two_modes(in_ch, out_ch, kernel)`. "
        "Initialize a Conv2d weight in BOTH modes and report the bounds.\n\n"
        "Use `nn.init.kaiming_uniform_(weight, mode=..., nonlinearity='relu')`.\n\n"
        "Return a dict with EXACTLY these keys:\n\n"
        "- `'fan_in'`: `int`, `in_ch * kernel * kernel`.\n"
        "- `'fan_out'`: `int`, `out_ch * kernel * kernel`.\n"
        "- `'bound_fan_in'`: `float`, `sqrt(6.0 / fan_in)` (the theoretical "
        "uniform bound for `nonlinearity='relu'`, i.e. gain=sqrt(2)).\n"
        "- `'bound_fan_out'`: `float`, `sqrt(6.0 / fan_out)`.\n"
        "- `'weight_fan_in'`: `torch.Tensor`, the Conv2d weight of shape "
        "`(out_ch, in_ch, kernel, kernel)` initialized with `mode='fan_in'`.\n"
        "- `'weight_fan_out'`: `torch.Tensor`, same shape, "
        "`mode='fan_out'`.\n"
        "- `'empirical_max_in'`: `float`, `weight_fan_in.abs().max().item()`.\n"
        "- `'empirical_max_out'`: `float`, "
        "`weight_fan_out.abs().max().item()`.\n\n"
        "Constraints:\n"
        "- Build each weight as an empty `(out_ch, in_ch, kernel, kernel)` "
        "tensor — do NOT construct a full `nn.Conv2d` module.\n"
        "- Seed `t.manual_seed(0)` BEFORE the first init and AGAIN before "
        "the second so the two are directly comparable.\n"
        "- The empirical max-abs MUST be `<= bound` (uniform is bounded)."
    ),
    "stub": (
        "def ex2_kaiming_conv_two_modes(in_ch: int, out_ch: int, kernel: int) -> dict:\n"
        '    """Init a Conv2d weight in fan_in mode + fan_out mode, return bounds + tensors."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "\n"
        "# === Expanding conv: 3 -> 64 channels, 3x3 kernel ===\n"
        "d = ex2_kaiming_conv_two_modes(in_ch=3, out_ch=64, kernel=3)\n"
        "assert d['fan_in'] == 27, f'fan_in wrong: {d[\"fan_in\"]}'\n"
        "assert d['fan_out'] == 576, f'fan_out wrong: {d[\"fan_out\"]}'\n"
        "assert math.isclose(d['bound_fan_in'], math.sqrt(6.0 / 27), rel_tol=1e-6)\n"
        "assert math.isclose(d['bound_fan_out'], math.sqrt(6.0 / 576), rel_tol=1e-6)\n"
        "# fan_in mode produces LARGER bound than fan_out mode for an expanding conv.\n"
        "assert d['bound_fan_in'] > d['bound_fan_out'], (\n"
        "    f'expanding conv should have bound_fan_in > bound_fan_out; got '\n"
        "    f'{d[\"bound_fan_in\"]} vs {d[\"bound_fan_out\"]}'\n"
        ")\n"
        "\n"
        "# === Shape of both weights ===\n"
        "assert d['weight_fan_in'].shape == (64, 3, 3, 3)\n"
        "assert d['weight_fan_out'].shape == (64, 3, 3, 3)\n"
        "\n"
        "# === Empirical max-abs is within the theoretical bound ===\n"
        "assert d['empirical_max_in'] <= d['bound_fan_in'] + 1e-6, (\n"
        "    f'fan_in empirical max-abs {d[\"empirical_max_in\"]} exceeds bound '\n"
        "    f'{d[\"bound_fan_in\"]} — kaiming_uniform_ should be bounded'\n"
        ")\n"
        "assert d['empirical_max_out'] <= d['bound_fan_out'] + 1e-6\n"
        "\n"
        "# === Empirical max is close to (not way below) the bound — large fan-in is statistically dense ===\n"
        "# 64*27 = 1728 samples; for U(-b, b) the expected max-abs approaches b as N grows.\n"
        "assert d['empirical_max_in'] >= 0.7 * d['bound_fan_in'], (\n"
        "    f'fan_in empirical max-abs {d[\"empirical_max_in\"]} suspiciously low '\n"
        "    f'vs bound {d[\"bound_fan_in\"]} — check the init was actually applied'\n"
        ")\n"
        "assert d['empirical_max_out'] >= 0.7 * d['bound_fan_out']\n"
        "\n"
        "# === Square conv (in_ch == out_ch): two bounds equal ===\n"
        "d = ex2_kaiming_conv_two_modes(in_ch=32, out_ch=32, kernel=3)\n"
        "assert d['fan_in'] == d['fan_out'] == 288\n"
        "assert math.isclose(d['bound_fan_in'], d['bound_fan_out'], rel_tol=1e-6)\n"
        "\n"
        "# === Contracting conv: 128 -> 8 (encoder->classifier head) ===\n"
        "d = ex2_kaiming_conv_two_modes(in_ch=128, out_ch=8, kernel=1)\n"
        "assert d['fan_in'] == 128 and d['fan_out'] == 8\n"
        "# Contracting → fan_out mode is LARGER bound.\n"
        "assert d['bound_fan_out'] > d['bound_fan_in']\n"
        "\n"
        "# === Returned tensors are floating point ===\n"
        "assert d['weight_fan_in'].dtype == t.float32 or d['weight_fan_in'].dtype == t.float64\n"
        "\n"
        "# === All keys present ===\n"
        "expected_keys = {'fan_in', 'fan_out', 'bound_fan_in', 'bound_fan_out',\n"
        "                 'weight_fan_in', 'weight_fan_out',\n"
        "                 'empirical_max_in', 'empirical_max_out'}\n"
        "assert set(d.keys()) == expected_keys, f'keys wrong: {set(d.keys())}'"
    ),
    "solution_body": (
        "def ex2_kaiming_conv_two_modes(in_ch, out_ch, kernel):\n"
        "    import math\n"
        "    fan_in = in_ch * kernel * kernel\n"
        "    fan_out = out_ch * kernel * kernel\n"
        "    bound_in = math.sqrt(6.0 / fan_in)\n"
        "    bound_out = math.sqrt(6.0 / fan_out)\n"
        "    shape = (out_ch, in_ch, kernel, kernel)\n"
        "\n"
        "    t.manual_seed(0)\n"
        "    w_in = t.empty(*shape)\n"
        "    t.nn.init.kaiming_uniform_(w_in, mode='fan_in', nonlinearity='relu')\n"
        "\n"
        "    t.manual_seed(0)\n"
        "    w_out = t.empty(*shape)\n"
        "    t.nn.init.kaiming_uniform_(w_out, mode='fan_out', nonlinearity='relu')\n"
        "\n"
        "    return {\n"
        "        'fan_in': fan_in,\n"
        "        'fan_out': fan_out,\n"
        "        'bound_fan_in': bound_in,\n"
        "        'bound_fan_out': bound_out,\n"
        "        'weight_fan_in': w_in,\n"
        "        'weight_fan_out': w_out,\n"
        "        'empirical_max_in': w_in.abs().max().item(),\n"
        "        'empirical_max_out': w_out.abs().max().item(),\n"
        "    }"
    ),
    "solution_notes": (
        "**`nonlinearity='relu'` ⇒ gain=sqrt(2).** PyTorch's gain table "
        "(`torch.nn.init.calculate_gain`) gives sqrt(2) for relu. The "
        "uniform bound is `gain * sqrt(3/fan)` = `sqrt(2) * sqrt(3/fan)` "
        "= `sqrt(6/fan)`. Pass `nonlinearity='relu'` explicitly; the "
        "default `'leaky_relu'` with `a=sqrt(5)` is the legacy behavior "
        "for backward compatibility with old PyTorch defaults.\n\n"
        "**Same seed for both modes.** Without resetting the seed, the "
        "fan_out init would consume different random numbers than fan_in, "
        "and the comparison would be confounded by the RNG state. "
        "`t.manual_seed(0)` before each init guarantees the only "
        "difference is the SCALE (the bound), not the underlying "
        "samples.\n\n"
        "**Why expanding convs use fan_in mode by default.** Forward-pass "
        "variance preservation. For a Conv 3→64, fan_in=27 weights "
        "contribute to each output activation; the variance of that sum "
        "needs the weight variance scaled by `2/fan_in`. fan_out preserves "
        "the BACKWARD pass instead — useful when the bottleneck is "
        "gradient flow, not forward signal."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — loss-item-scalar-extract ex2
# ---------------------------------------------------------------------------

SPEC_LOSS_ITEM_RAISES = {
    "atom_id": "loss-item-scalar-extract",
    "subtopic": "PyTorch: loss.item() scalar extract",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_LOSS_ITEM_RAISES,
    "exercise_index": 2,
    "exercise_title": "diagnose why .item() raises on per-sample loss + fix with .mean().item()",
    "slug": "diagnose-item-on-non-scalar-and-fix-with-mean-item",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["item", "scalar", "RuntimeError", "reduction"],
    "kcs": [
        "item-requires-zero-d-tensor",
        "reduce-before-item",
    ],
    "lo": (
        "Analyze the failure mode of `.item()` on a multi-element tensor "
        "(per-sample loss) by catching the `RuntimeError` and demonstrate "
        "the canonical fix — `.mean().item()` and `.sum().item()` — return "
        "scalar Python floats."
    ),
    "prompt_body": (
        "Implement `ex2_item_failure_and_fix(per_sample_loss)`. The "
        "deepening variant of ex1.\n\n"
        "Inputs:\n"
        "- `per_sample_loss`: a 1-D `torch.Tensor` of shape `(B,)` — e.g. "
        "what `F.cross_entropy(reduction='none')` would return.\n\n"
        "Return a dict with EXACTLY these keys:\n\n"
        "- `'numel'`: `int`, `per_sample_loss.numel()`.\n"
        "- `'item_raised'`: `bool`, `True` if `.item()` on the input raises "
        "`RuntimeError`, else `False`. (Catch the exception — do NOT let it "
        "propagate.)\n"
        "- `'item_error_msg'`: `str | None`. If `.item()` raised, the str "
        "message of the exception. Else `None`.\n"
        "- `'mean_scalar'`: `float`, `per_sample_loss.mean().item()`. "
        "Always works.\n"
        "- `'sum_scalar'`: `float`, `per_sample_loss.sum().item()`. Always "
        "works.\n"
        "- `'mean_scalar_type'`: `type`, `type(mean_scalar)` — must be "
        "Python `float`, not `torch.Tensor`.\n\n"
        "Behavior on a 0-d input:\n"
        "- If `per_sample_loss.numel() == 1`, `.item()` does NOT raise. "
        "Set `'item_raised'=False` and `'item_error_msg'=None`. "
        "`mean_scalar` and `sum_scalar` still compute (each will equal the "
        "single value).\n\n"
        "Constraints:\n"
        "- Catch ONLY `RuntimeError` from `.item()` — don't catch all "
        "exceptions.\n"
        "- Do not mutate the input tensor."
    ),
    "stub": (
        "def ex2_item_failure_and_fix(per_sample_loss: Tensor) -> dict:\n"
        '    """Demonstrate .item() failure on non-scalar tensor + the .mean().item() fix."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Multi-element 1-D loss tensor (B=4): .item() must raise ===\n"
        "loss = t.tensor([0.5, 1.0, 2.0, 0.5])\n"
        "d = ex2_item_failure_and_fix(loss)\n"
        "assert d['numel'] == 4\n"
        "assert d['item_raised'] is True, f'item must raise on numel>1, got item_raised={d[\"item_raised\"]}'\n"
        "assert isinstance(d['item_error_msg'], str)\n"
        "assert 'Scalar' in d['item_error_msg'] or 'scalar' in d['item_error_msg'] or '4' in d['item_error_msg'], (\n"
        "    f'error message should mention scalar or the element count; got {d[\"item_error_msg\"]!r}'\n"
        ")\n"
        "# === The .mean().item() and .sum().item() fixes return real Python floats ===\n"
        "assert d['mean_scalar_type'] is float, f'mean_scalar must be Python float, got {d[\"mean_scalar_type\"]}'\n"
        "import math\n"
        "assert math.isclose(d['mean_scalar'], 1.0, rel_tol=1e-6)\n"
        "assert math.isclose(d['sum_scalar'], 4.0, rel_tol=1e-6)\n"
        "\n"
        "# === 0-d tensor (loss.mean() output already): .item() does NOT raise ===\n"
        "scalar = t.tensor(7.5)\n"
        "d = ex2_item_failure_and_fix(scalar)\n"
        "assert d['numel'] == 1\n"
        "assert d['item_raised'] is False, f'item should succeed on 0-d, got item_raised={d[\"item_raised\"]}'\n"
        "assert d['item_error_msg'] is None\n"
        "assert math.isclose(d['mean_scalar'], 7.5, rel_tol=1e-6)\n"
        "assert math.isclose(d['sum_scalar'], 7.5, rel_tol=1e-6)\n"
        "\n"
        "# === Single-element 1-D tensor: numel==1, item() also succeeds ===\n"
        "single = t.tensor([3.14])\n"
        "d = ex2_item_failure_and_fix(single)\n"
        "assert d['numel'] == 1\n"
        "assert d['item_raised'] is False, 'item must succeed when numel==1, regardless of rank'\n"
        "assert d['item_error_msg'] is None\n"
        "assert math.isclose(d['mean_scalar'], 3.14, rel_tol=1e-5)\n"
        "\n"
        "# === Larger batch ===\n"
        "loss = t.arange(10).float()  # [0, 1, ..., 9], sum=45, mean=4.5\n"
        "d = ex2_item_failure_and_fix(loss)\n"
        "assert d['numel'] == 10\n"
        "assert d['item_raised'] is True\n"
        "assert math.isclose(d['mean_scalar'], 4.5, rel_tol=1e-6)\n"
        "assert math.isclose(d['sum_scalar'], 45.0, rel_tol=1e-6)\n"
        "\n"
        "# === Input is not mutated ===\n"
        "loss = t.tensor([1.0, 2.0, 3.0])\n"
        "loss_clone = loss.clone()\n"
        "_ = ex2_item_failure_and_fix(loss)\n"
        "assert t.equal(loss, loss_clone), 'input tensor must not be mutated'\n"
        "\n"
        "# === Higher-rank multi-element tensor also raises on .item() ===\n"
        "loss = t.ones(2, 3)  # numel=6\n"
        "d = ex2_item_failure_and_fix(loss)\n"
        "assert d['numel'] == 6\n"
        "assert d['item_raised'] is True\n"
        "assert math.isclose(d['mean_scalar'], 1.0, rel_tol=1e-6)\n"
        "assert math.isclose(d['sum_scalar'], 6.0, rel_tol=1e-6)\n"
        "\n"
        "# === sum_scalar is Python float, not a 0-d tensor ===\n"
        "assert type(d['sum_scalar']) is float, f'sum_scalar must be Python float, got {type(d[\"sum_scalar\"]).__name__}'\n"
        "\n"
        "# === All keys ===\n"
        "expected_keys = {'numel', 'item_raised', 'item_error_msg', 'mean_scalar',\n"
        "                 'sum_scalar', 'mean_scalar_type'}\n"
        "assert set(d.keys()) == expected_keys, f'keys wrong: {set(d.keys())}'"
    ),
    "solution_body": (
        "def ex2_item_failure_and_fix(per_sample_loss):\n"
        "    numel = per_sample_loss.numel()\n"
        "    item_raised = False\n"
        "    item_error_msg = None\n"
        "    try:\n"
        "        _ = per_sample_loss.item()\n"
        "    except RuntimeError as e:\n"
        "        item_raised = True\n"
        "        item_error_msg = str(e)\n"
        "    mean_scalar = per_sample_loss.mean().item()\n"
        "    sum_scalar = per_sample_loss.sum().item()\n"
        "    return {\n"
        "        'numel': numel,\n"
        "        'item_raised': item_raised,\n"
        "        'item_error_msg': item_error_msg,\n"
        "        'mean_scalar': mean_scalar,\n"
        "        'sum_scalar': sum_scalar,\n"
        "        'mean_scalar_type': type(mean_scalar),\n"
        "    }"
    ),
    "solution_notes": (
        "**`.item()` on numel==1 succeeds REGARDLESS of rank.** A "
        "`(1, 1, 1)` tensor with one element calls `.item()` fine. The "
        "check inside PyTorch is `numel() == 1`, not `ndim == 0`. This "
        "matters when a batch happens to be size 1 — your `.item()` calls "
        "won't crash, but they will crash the moment batch grows.\n\n"
        "**Catch `RuntimeError`, not `Exception`.** PyTorch raises "
        "`RuntimeError` specifically. Catching a broader exception class "
        "would swallow unrelated bugs (e.g. an `AttributeError` from a "
        "typo). Narrow except clauses are how you keep error handling "
        "focused.\n\n"
        "**`type(x) is float` for the Python-float check.** "
        "`isinstance(x, float)` is also fine here — both `True` and "
        "`False` for `float` and not `Tensor`. The distinction matters "
        "more when a subclass is involved; here either works."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — rearrange-as-sequential-layer ex2
# ---------------------------------------------------------------------------

SPEC_REARRANGE_PATCHIFY = {
    "atom_id": "rearrange-as-sequential-layer",
    "subtopic": "Einops: Rearrange as nn.Sequential layer",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_REARRANGE_PATCHIFY,
    "exercise_index": 2,
    "exercise_title": "Rearrange-based patchify layer inside an nn.Sequential",
    "slug": "rearrange-patchify-layer-inside-sequential",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["rearrange", "patchify", "vit", "sequential"],
    "kcs": [
        "patchify-rearrange-pattern",
        "rearrange-as-sequential-module",
    ],
    "lo": (
        "Apply the einops patchify pattern "
        "`'b c (h ph) (w pw) -> b (h w) (ph pw c)'` as an "
        "`einops.layers.torch.Rearrange` layer inside an `nn.Sequential` "
        "that maps `(B, 3, H, W)` to `(B, num_patches, embed_dim)`."
    ),
    "prompt_body": (
        "Implement `ex2_patchify_sequential(in_channels, height, width, "
        "patch_size, embed_dim)`. Build a ViT-style patch embedder using "
        "ONLY `nn.Sequential` + `einops.layers.torch.Rearrange` + "
        "`nn.Linear` — no custom forward.\n\n"
        "Inputs (all positional ok):\n"
        "- `in_channels`: e.g. 3.\n"
        "- `height, width`: input spatial size; both must be divisible by "
        "`patch_size`.\n"
        "- `patch_size`: side length of a square patch.\n"
        "- `embed_dim`: output dimensionality per patch.\n\n"
        "Pipeline (in order):\n"
        "1. `Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', "
        "ph=patch_size, pw=patch_size)` — split into non-overlapping "
        "patches and flatten EACH patch in NHWC order. Output shape: "
        "`(B, num_patches, patch_dim)` where `num_patches = (height // "
        "patch_size) * (width // patch_size)` and `patch_dim = patch_size "
        "* patch_size * in_channels`.\n"
        "2. `nn.Linear(patch_dim, embed_dim)` — project each patch to "
        "`embed_dim`.\n\n"
        "Constraints:\n"
        "- Return `nn.Sequential`, not a custom Module.\n"
        "- The model must accept `(B, in_channels, height, width)` and "
        "return `(B, num_patches, embed_dim)`.\n"
        "- DO NOT add extra layers (no LayerNorm, no positional encoding) "
        "— this drill is about the Rearrange-as-layer pattern, not the "
        "full ViT recipe."
    ),
    "stub": (
        "def ex2_patchify_sequential(in_channels: int, height: int, width: int,\n"
        "                             patch_size: int, embed_dim: int):\n"
        '    """Return an nn.Sequential that patchifies (B,C,H,W) -> (B, n_patches, embed_dim)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from einops.layers.torch import Rearrange\n"
        "\n"
        "# === Basic: 3x8x8 with 2x2 patches, embed_dim=12 ===\n"
        "model = ex2_patchify_sequential(in_channels=3, height=8, width=8,\n"
        "                                 patch_size=2, embed_dim=12)\n"
        "\n"
        "# === Must be an nn.Sequential (not a wrapper Module) ===\n"
        "assert isinstance(model, t.nn.Sequential), (\n"
        "    f'expected nn.Sequential, got {type(model).__name__} — the point is to '\n"
        "    'compose Rearrange + Linear with no custom forward.'\n"
        ")\n"
        "\n"
        "# === Children: exactly Rearrange then Linear ===\n"
        "kids = list(model.children())\n"
        "assert len(kids) == 2, f'pipeline should be exactly 2 layers, got {len(kids)}: {kids}'\n"
        "assert isinstance(kids[0], Rearrange), f'first layer must be Rearrange, got {type(kids[0]).__name__}'\n"
        "assert isinstance(kids[1], t.nn.Linear), f'second layer must be Linear, got {type(kids[1]).__name__}'\n"
        "\n"
        "# === Linear has the correct shape (patch_dim -> embed_dim) ===\n"
        "patch_dim = 2 * 2 * 3   # 12\n"
        "num_patches = (8 // 2) * (8 // 2)   # 16\n"
        "assert kids[1].in_features == patch_dim, f'Linear in_features must be {patch_dim}, got {kids[1].in_features}'\n"
        "assert kids[1].out_features == 12, f'Linear out_features must be embed_dim=12, got {kids[1].out_features}'\n"
        "\n"
        "# === Forward shape is (B, num_patches, embed_dim) ===\n"
        "x = t.randn(4, 3, 8, 8)\n"
        "y = model(x)\n"
        "assert y.shape == (4, num_patches, 12), f'expected (4, 16, 12), got {tuple(y.shape)}'\n"
        "\n"
        "# === Larger config: 3x16x32 with 4x4 patches, embed_dim=64 ===\n"
        "model = ex2_patchify_sequential(in_channels=3, height=16, width=32,\n"
        "                                 patch_size=4, embed_dim=64)\n"
        "x = t.randn(2, 3, 16, 32)\n"
        "y = model(x)\n"
        "# num_patches = (16/4) * (32/4) = 4 * 8 = 32\n"
        "assert y.shape == (2, 32, 64), f'expected (2, 32, 64), got {tuple(y.shape)}'\n"
        "\n"
        "# === Patchify ordering: NHWC inside the patch (ph pw c) ===\n"
        "# Build a synthetic input where patch (0,0) has a known pattern.\n"
        "# We construct so the first patch is [[[r0,r1],[r2,r3]], [[g0,g1],[g2,g3]], [[b0,b1],[b2,b3]]].\n"
        "# After 'b c (h ph) (w pw) -> b (h w) (ph pw c)' the first patch row is:\n"
        "#   [r0, g0, b0, r1, g1, b1, r2, g2, b2, r3, g3, b3] (NHWC inside the patch)\n"
        "# We can verify by checking that the patchify output (no Linear) follows this layout.\n"
        "pat = Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=2, pw=2)\n"
        "x = t.zeros(1, 3, 2, 2)\n"
        "x[0, 0, 0, 0] = 100   # red top-left\n"
        "x[0, 1, 0, 0] = 200   # green top-left\n"
        "x[0, 2, 0, 0] = 300   # blue top-left\n"
        "out = pat(x)\n"
        "# Shape: (1, 1, 12) — one patch with 12 features.\n"
        "assert out.shape == (1, 1, 12)\n"
        "# First three features = [r0, g0, b0] = [100, 200, 300].\n"
        "assert out[0, 0, 0].item() == 100, f'expected r0=100 at position 0, got {out[0,0,0].item()}'\n"
        "assert out[0, 0, 1].item() == 200, f'expected g0=200 at position 1, got {out[0,0,1].item()}'\n"
        "assert out[0, 0, 2].item() == 300, f'expected b0=300 at position 2, got {out[0,0,2].item()}'\n"
        "\n"
        "# === Single-patch degenerate case: patch_size == height == width ===\n"
        "model = ex2_patchify_sequential(in_channels=3, height=4, width=4,\n"
        "                                 patch_size=4, embed_dim=10)\n"
        "x = t.randn(1, 3, 4, 4)\n"
        "y = model(x)\n"
        "assert y.shape == (1, 1, 10), f'one-patch case wrong: {tuple(y.shape)}'\n"
        "\n"
        "# === No custom forward — model.forward is the inherited Sequential.forward ===\n"
        "assert type(model).forward is t.nn.Sequential.forward, 'must not override forward'"
    ),
    "solution_body": (
        "def ex2_patchify_sequential(in_channels, height, width, patch_size, embed_dim):\n"
        "    from einops.layers.torch import Rearrange\n"
        "    patch_dim = patch_size * patch_size * in_channels\n"
        "    return t.nn.Sequential(\n"
        "        Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)',\n"
        "                  ph=patch_size, pw=patch_size),\n"
        "        t.nn.Linear(patch_dim, embed_dim),\n"
        "    )"
    ),
    "solution_notes": (
        "**Why `(ph pw c)` and not `(c ph pw)`.** ViT and most transformer "
        "literatures flatten each patch in NHWC order — pixel-by-pixel, "
        "channel-last. PyTorch tensors are NCHW natively, so this "
        "rearrange does the channel-last conversion as part of the "
        "flatten. Reordering to `(c ph pw)` would give channel-major; "
        "downstream features would still be patch_dim long but their "
        "meaning is permuted.\n\n"
        "**The Rearrange-as-Module trick.** "
        "`einops.layers.torch.Rearrange` is an `nn.Module`, so it slots "
        "into `nn.Sequential` directly. Without it, you'd need a custom "
        "Module class just to call `einops.rearrange` inside `forward` — "
        "kills the no-boilerplate point.\n\n"
        "**`patch_dim = patch_size * patch_size * in_channels`.** Each "
        "patch is `ph × pw × c` floats. Always derive it from the inputs; "
        "hardcoding `768` or `512` (standard ViT-Base / ViT-Small values) "
        "is the trap that breaks the moment you change `in_channels` for "
        "grayscale or hyperspectral."
    ),
    "extra_imports": ["from einops.layers.torch import Rearrange"],
}


# ---------------------------------------------------------------------------
# SPEC 5 — sqrt-eps-stabilize ex2
# ---------------------------------------------------------------------------

SPEC_SQRT_EPS_PLACEMENT = {
    "atom_id": "sqrt-eps-stabilize",
    "subtopic": "Numerical: sqrt-eps stabilization",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_SQRT_EPS_PLACEMENT,
    "exercise_index": 2,
    "exercise_title": "contrast sqrt(var+eps) vs sqrt(var)+eps — gradient stability at var=0",
    "slug": "contrast-eps-inside-vs-outside-sqrt-gradient-stability",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["sqrt", "eps", "gradient", "batchnorm", "stability"],
    "kcs": [
        "eps-inside-vs-outside-sqrt-forward",
        "eps-inside-bounds-the-gradient",
    ],
    "lo": (
        "Analyze the two `eps`-placement variants — `sqrt(var + eps)` vs "
        "`sqrt(var) + eps` — by computing forward values and "
        "backward-pass gradients at `var=0`, and confirm the INSIDE "
        "placement bounds `d/dvar` while the OUTSIDE placement diverges."
    ),
    "prompt_body": (
        "Implement `ex2_compare_eps_placement(var, eps)`. The deepening "
        "variant of ex1.\n\n"
        "Inputs:\n"
        "- `var`: a 1-D `torch.Tensor` of variances (can include 0). "
        "Must be `float`, `requires_grad=False` going in (this function "
        "will build its own requires-grad variants for the gradient pass).\n"
        "- `eps`: `float`.\n\n"
        "Return a dict with EXACTLY these keys:\n\n"
        "- `'sigma_inside'`: `torch.Tensor`, `(var + eps).sqrt()` — same "
        "shape as `var`, no_grad.\n"
        "- `'sigma_outside'`: `torch.Tensor`, `var.sqrt() + eps`.\n"
        "- `'sigma_inside_finite'`: `bool`, `torch.isfinite(sigma_inside).all().item()`.\n"
        "- `'sigma_outside_finite'`: `bool`, "
        "`torch.isfinite(sigma_outside).all().item()`.\n"
        "- `'grad_inside'`: `torch.Tensor`, the gradient `d sigma_inside / "
        "d var` at the given `var` values — computed by autograd on the "
        "inside expression with `.sum().backward()`.\n"
        "- `'grad_outside'`: `torch.Tensor`, same for the outside "
        "expression. NOTE: at `var=0` this will be `inf` because the "
        "derivative of `sqrt(var)` at 0 is unbounded.\n"
        "- `'grad_inside_finite'`: `bool`.\n"
        "- `'grad_outside_finite'`: `bool`.\n"
        "- `'inside_grad_upper_bound'`: `float`, "
        "`1.0 / (2 * sqrt(eps))` — the analytical max of `d sqrt(var + "
        "eps) / d var`, achieved at `var=0`.\n\n"
        "Constraints:\n"
        "- Use a FRESH `var.clone().detach().requires_grad_(True)` for "
        "each gradient computation so the two graphs don't share state.\n"
        "- Always return shapes matching the input."
    ),
    "stub": (
        "def ex2_compare_eps_placement(var: Tensor, eps: float) -> dict:\n"
        '    """Compare eps INSIDE vs OUTSIDE sqrt — forward + backward at var=0."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "\n"
        "# === Case 1: var contains a zero ===\n"
        "var = t.tensor([0.0, 0.25, 1.0, 4.0])\n"
        "eps = 1e-5\n"
        "d = ex2_compare_eps_placement(var, eps)\n"
        "\n"
        "# Forward values — both finite, but differ at var=0.\n"
        "assert d['sigma_inside_finite'] is True\n"
        "assert d['sigma_outside_finite'] is True\n"
        "assert math.isclose(d['sigma_inside'][0].item(), math.sqrt(eps), rel_tol=1e-6), (\n"
        "    f'inside sqrt at var=0 should be sqrt(eps)={math.sqrt(eps)}, got {d[\"sigma_inside\"][0].item()}'\n"
        ")\n"
        "assert math.isclose(d['sigma_outside'][0].item(), eps, rel_tol=1e-6), (\n"
        "    f'outside sqrt at var=0 should be 0 + eps = {eps}, got {d[\"sigma_outside\"][0].item()}'\n"
        ")\n"
        "# At var>0 the two are very close (just shifted by ~eps).\n"
        "for i in [1, 2, 3]:\n"
        "    assert abs(d['sigma_inside'][i].item() - d['sigma_outside'][i].item()) < 1e-2, (\n"
        "        f'at var>0 the two placements should be close; got '\n"
        "        f'inside={d[\"sigma_inside\"][i].item()}, outside={d[\"sigma_outside\"][i].item()}'\n"
        "    )\n"
        "\n"
        "# === Gradient at var=0: inside is BOUNDED, outside is INF ===\n"
        "assert d['grad_inside_finite'] is True, (\n"
        "    f'gradient of sqrt(var+eps) must be finite at var=0; got grad_inside={d[\"grad_inside\"]}'\n"
        ")\n"
        "assert d['grad_outside_finite'] is False, (\n"
        "    f'gradient of sqrt(var)+eps must be NON-finite at var=0; got grad_outside={d[\"grad_outside\"]}'\n"
        ")\n"
        "# Specifically, grad_outside[0] is inf or nan.\n"
        "assert not t.isfinite(d['grad_outside'][0]).item(), (\n"
        "    f'd sqrt(var)/d var at var=0 must be inf; got {d[\"grad_outside\"][0].item()}'\n"
        ")\n"
        "\n"
        "# === Analytical upper bound on inside gradient: 1/(2*sqrt(eps)) ===\n"
        "expected_bound = 1.0 / (2 * math.sqrt(eps))\n"
        "assert math.isclose(d['inside_grad_upper_bound'], expected_bound, rel_tol=1e-6)\n"
        "# Empirical grad_inside at var=0 should equal the bound (it's the max).\n"
        "assert math.isclose(d['grad_inside'][0].item(), expected_bound, rel_tol=1e-4), (\n"
        "    f'grad_inside at var=0 should equal analytical bound {expected_bound}, got {d[\"grad_inside\"][0].item()}'\n"
        ")\n"
        "# Empirical grad_inside at var>0 should be LESS THAN the bound.\n"
        "for i in [1, 2, 3]:\n"
        "    g = d['grad_inside'][i].item()\n"
        "    assert g < expected_bound, f'grad at var={var[i].item()} ({g}) should be < bound ({expected_bound})'\n"
        "\n"
        "# === Case 2: var all > 0 — both gradients finite ===\n"
        "var = t.tensor([0.1, 0.5, 1.0, 4.0])\n"
        "d = ex2_compare_eps_placement(var, eps=1e-5)\n"
        "assert d['grad_inside_finite'] is True\n"
        "assert d['grad_outside_finite'] is True\n"
        "# And the gradients are very close (the eps placement matters only near var=0).\n"
        "for i in range(4):\n"
        "    gi = d['grad_inside'][i].item()\n"
        "    go = d['grad_outside'][i].item()\n"
        "    assert abs(gi - go) < 1e-3, f'far from var=0 the two grads should agree; got {gi} vs {go}'\n"
        "\n"
        "# === Case 3: shapes match input ===\n"
        "var = t.tensor([0.0, 0.0, 0.0])\n"
        "d = ex2_compare_eps_placement(var, eps=1e-4)\n"
        "assert d['sigma_inside'].shape == var.shape\n"
        "assert d['grad_inside'].shape == var.shape\n"
        "assert d['sigma_outside'].shape == var.shape\n"
        "assert d['grad_outside'].shape == var.shape\n"
        "# All three var=0 entries blow up on outside.\n"
        "assert d['grad_outside_finite'] is False\n"
        "\n"
        "# === All keys ===\n"
        "expected_keys = {'sigma_inside', 'sigma_outside',\n"
        "                 'sigma_inside_finite', 'sigma_outside_finite',\n"
        "                 'grad_inside', 'grad_outside',\n"
        "                 'grad_inside_finite', 'grad_outside_finite',\n"
        "                 'inside_grad_upper_bound'}\n"
        "assert set(d.keys()) == expected_keys, f'keys wrong: {set(d.keys())}'"
    ),
    "solution_body": (
        "def ex2_compare_eps_placement(var, eps):\n"
        "    import math\n"
        "    # Forward (no grad needed here).\n"
        "    with t.no_grad():\n"
        "        sigma_inside = (var + eps).sqrt()\n"
        "        sigma_outside = var.sqrt() + eps\n"
        "\n"
        "    # Backward for inside.\n"
        "    var_in = var.clone().detach().requires_grad_(True)\n"
        "    ((var_in + eps).sqrt()).sum().backward()\n"
        "    grad_inside = var_in.grad.detach().clone()\n"
        "\n"
        "    # Backward for outside.\n"
        "    var_out = var.clone().detach().requires_grad_(True)\n"
        "    (var_out.sqrt() + eps).sum().backward()\n"
        "    grad_outside = var_out.grad.detach().clone()\n"
        "\n"
        "    return {\n"
        "        'sigma_inside': sigma_inside,\n"
        "        'sigma_outside': sigma_outside,\n"
        "        'sigma_inside_finite': bool(t.isfinite(sigma_inside).all().item()),\n"
        "        'sigma_outside_finite': bool(t.isfinite(sigma_outside).all().item()),\n"
        "        'grad_inside': grad_inside,\n"
        "        'grad_outside': grad_outside,\n"
        "        'grad_inside_finite': bool(t.isfinite(grad_inside).all().item()),\n"
        "        'grad_outside_finite': bool(t.isfinite(grad_outside).all().item()),\n"
        "        'inside_grad_upper_bound': 1.0 / (2 * math.sqrt(eps)),\n"
        "    }"
    ),
    "solution_notes": (
        "**The forward is a red herring.** Both placements give finite "
        "forward values at `var=0` (`sqrt(eps)` vs `eps`). The "
        "stability argument is about the BACKWARD pass. "
        "`d/dvar sqrt(var) = 1/(2*sqrt(var))` blows up at 0; "
        "`d/dvar sqrt(var+eps) = 1/(2*sqrt(var+eps))` is bounded by "
        "`1/(2*sqrt(eps))`.\n\n"
        "**Fresh `requires_grad_(True)` per pass.** Reusing the same "
        "tensor across two `.backward()` calls accumulates gradients "
        "into the SAME `.grad` field — you'd see the sum of both, not "
        "either one cleanly. `clone().detach().requires_grad_(True)` is "
        "the standard 'new graph leaf' pattern.\n\n"
        "**Why BatchNorm/LayerNorm chose INSIDE.** Even in float32, "
        "early-training activations can produce variance very close to "
        "zero on collapsed channels. INSIDE placement means a max "
        "gradient of `1/(2*sqrt(1e-5)) ≈ 158` — large but FINITE. "
        "OUTSIDE placement would produce inf grads that propagate up the "
        "network and NaN-out the optimizer. The PyTorch convention is "
        "explicit in `aten/src/ATen/native/Normalization.cpp`."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — stride-zero-broadcast ex2
# ---------------------------------------------------------------------------

SPEC_STRIDE_CONTIGUOUS = {
    "atom_id": "stride-zero-broadcast",
    "subtopic": "PyTorch: Zero-stride broadcasting",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_STRIDE_CONTIGUOUS,
    "exercise_index": 2,
    "exercise_title": "detect zero-stride view, materialize via .contiguous(), report stride + storage delta",
    "slug": "detect-zero-stride-and-materialize-via-contiguous",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["stride", "contiguous", "broadcast", "storage"],
    "kcs": [
        "detect-zero-stride-from-stride-tuple",
        "contiguous-allocates-fresh-storage",
    ],
    "lo": (
        "Apply `.stride()` inspection to detect a zero-stride broadcast "
        "view, then call `.contiguous()` and report the stride change + "
        "storage size delta that confirm materialization."
    ),
    "prompt_body": (
        "Implement `ex2_materialize_broadcast(x)`. The deepening variant "
        "of ex1.\n\n"
        "Inputs:\n"
        "- `x`: an arbitrary `torch.Tensor`. May or may not have a "
        "zero-stride axis.\n\n"
        "Return a dict with EXACTLY these keys:\n\n"
        "- `'has_zero_stride'`: `bool`, `True` iff any axis of `x.stride()` "
        "is `0`.\n"
        "- `'stride_before'`: `tuple[int, ...]`, `tuple(x.stride())`.\n"
        "- `'storage_nbytes_before'`: `int`, "
        "`x.untyped_storage().nbytes()` — the size of the underlying "
        "storage buffer in bytes (note: this is the source buffer, may be "
        "smaller than `x.numel() * elem_size` when broadcast).\n"
        "- `'y'`: `torch.Tensor`, `x.contiguous()`. If `x` was already "
        "contiguous, `y` is `x` itself (PyTorch's contract — "
        "`x.contiguous()` returns `x` when already contiguous). Otherwise "
        "`y` is a fresh copy.\n"
        "- `'stride_after'`: `tuple[int, ...]`, `tuple(y.stride())`.\n"
        "- `'storage_nbytes_after'`: `int`, "
        "`y.untyped_storage().nbytes()`.\n"
        "- `'is_contiguous_before'`: `bool`, `x.is_contiguous()`.\n"
        "- `'is_contiguous_after'`: `bool`, `y.is_contiguous()` — must be "
        "`True` for any `y`.\n"
        "- `'storage_shared'`: `bool`, "
        "`x.data_ptr() == y.data_ptr()` — `True` iff `y` is the same "
        "storage as `x` (i.e. no copy was needed).\n\n"
        "Constraints:\n"
        "- Do not mutate `x`.\n"
        "- `stride_after` must contain NO zero strides for a tensor with "
        "more than one element."
    ),
    "stub": (
        "def ex2_materialize_broadcast(x: Tensor) -> dict:\n"
        '    """Detect zero-stride, call .contiguous(), report stride + storage deltas."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Case 1: expand creates zero-stride view ===\n"
        "src = t.arange(3, dtype=t.float32)\n"
        "x = src.unsqueeze(0).expand(4, 3)   # shape (4, 3), stride (0, 1)\n"
        "d = ex2_materialize_broadcast(x)\n"
        "assert d['has_zero_stride'] is True\n"
        "assert d['stride_before'] == (0, 1), f'expand stride wrong: {d[\"stride_before\"]}'\n"
        "assert d['is_contiguous_before'] is False, 'expanded view should not be contiguous'\n"
        "# After .contiguous(): no zero stride, fully contiguous.\n"
        "assert d['stride_after'] == (3, 1), f'contiguous stride wrong: {d[\"stride_after\"]}'\n"
        "assert d['is_contiguous_after'] is True\n"
        "# Storage NBYTES grew: from 3 floats (12 bytes) to 12 floats (48 bytes).\n"
        "assert d['storage_nbytes_after'] > d['storage_nbytes_before'], (\n"
        "    f'.contiguous should grow storage; before={d[\"storage_nbytes_before\"]}, '\n"
        "    f'after={d[\"storage_nbytes_after\"]}'\n"
        ")\n"
        "assert d['storage_nbytes_after'] == 4 * 12, f'expected 48 bytes after, got {d[\"storage_nbytes_after\"]}'\n"
        "assert d['storage_shared'] is False, 'expand + contiguous must allocate fresh storage'\n"
        "assert t.equal(d['y'], x), 'materialized values must equal the broadcasted view'\n"
        "\n"
        "# === Case 2: already-contiguous tensor — no copy, no zero stride ===\n"
        "x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "d = ex2_materialize_broadcast(x)\n"
        "assert d['has_zero_stride'] is False\n"
        "assert d['stride_before'] == (4, 1)\n"
        "assert d['is_contiguous_before'] is True\n"
        "assert d['is_contiguous_after'] is True\n"
        "# Already contiguous → .contiguous() returns self.\n"
        "assert d['storage_shared'] is True, 'already-contiguous .contiguous() should NOT copy'\n"
        "assert d['storage_nbytes_before'] == d['storage_nbytes_after']\n"
        "assert d['stride_after'] == d['stride_before']\n"
        "\n"
        "# === Case 3: non-contiguous but no zero-stride (transpose) ===\n"
        "x = t.arange(12, dtype=t.float32).reshape(3, 4).transpose(0, 1)   # shape (4, 3), stride (1, 4)\n"
        "d = ex2_materialize_broadcast(x)\n"
        "assert d['has_zero_stride'] is False   # transpose doesn't introduce zero stride\n"
        "assert d['is_contiguous_before'] is False\n"
        "assert d['stride_before'] == (1, 4)\n"
        "assert d['stride_after'] == (3, 1)\n"
        "assert d['is_contiguous_after'] is True\n"
        "# Storage SIZE may or may not change (could stay equal — both have 12 elements).\n"
        "# But the data_ptr DOES change — fresh allocation for the contiguous copy.\n"
        "assert d['storage_shared'] is False, 'transpose + contiguous must allocate fresh storage'\n"
        "\n"
        "# === Case 4: multi-axis broadcast — zero strides on multiple axes ===\n"
        "x = t.tensor(7.0).expand(2, 3, 4)   # stride (0, 0, 0)\n"
        "d = ex2_materialize_broadcast(x)\n"
        "assert d['has_zero_stride'] is True\n"
        "assert d['stride_before'] == (0, 0, 0), f'fully-broadcast stride wrong: {d[\"stride_before\"]}'\n"
        "assert d['stride_after'] == (12, 4, 1), f'contiguous stride wrong: {d[\"stride_after\"]}'\n"
        "assert d['storage_shared'] is False\n"
        "# y should be 24 copies of 7.0.\n"
        "assert t.equal(d['y'], t.full((2, 3, 4), 7.0))\n"
        "\n"
        "# === Case 5: input not mutated ===\n"
        "src = t.arange(3, dtype=t.float32)\n"
        "x = src.expand(2, 3)\n"
        "x_stride_before = x.stride()\n"
        "_ = ex2_materialize_broadcast(x)\n"
        "assert x.stride() == x_stride_before, 'input must not be mutated'\n"
        "\n"
        "# === All keys ===\n"
        "expected_keys = {'has_zero_stride', 'stride_before', 'storage_nbytes_before',\n"
        "                 'y', 'stride_after', 'storage_nbytes_after',\n"
        "                 'is_contiguous_before', 'is_contiguous_after',\n"
        "                 'storage_shared'}\n"
        "assert set(d.keys()) == expected_keys, f'keys wrong: {set(d.keys())}'"
    ),
    "solution_body": (
        "def ex2_materialize_broadcast(x):\n"
        "    stride_before = tuple(x.stride())\n"
        "    has_zero_stride = any(s == 0 for s in stride_before)\n"
        "    storage_nbytes_before = x.untyped_storage().nbytes()\n"
        "    is_contig_before = x.is_contiguous()\n"
        "\n"
        "    y = x.contiguous()\n"
        "\n"
        "    stride_after = tuple(y.stride())\n"
        "    storage_nbytes_after = y.untyped_storage().nbytes()\n"
        "    is_contig_after = y.is_contiguous()\n"
        "    storage_shared = (x.data_ptr() == y.data_ptr())\n"
        "\n"
        "    return {\n"
        "        'has_zero_stride': has_zero_stride,\n"
        "        'stride_before': stride_before,\n"
        "        'storage_nbytes_before': storage_nbytes_before,\n"
        "        'y': y,\n"
        "        'stride_after': stride_after,\n"
        "        'storage_nbytes_after': storage_nbytes_after,\n"
        "        'is_contiguous_before': is_contig_before,\n"
        "        'is_contiguous_after': is_contig_after,\n"
        "        'storage_shared': storage_shared,\n"
        "    }"
    ),
    "solution_notes": (
        "**`x.contiguous()` returns `x` when already contiguous.** This "
        "is documented PyTorch behavior — no-op cost, no allocation. The "
        "`storage_shared` check via `data_ptr()` lets you detect "
        "whether a copy actually happened, which is what matters for "
        "memory budgeting.\n\n"
        "**`untyped_storage().nbytes()` over `numel() * elem_size`.** "
        "For a broadcast view, `numel()` reports the broadcasted shape "
        "(e.g. 12 for a `(4, 3)` expanded view) but storage is only the "
        "source buffer (3 elements). `untyped_storage().nbytes()` "
        "reports the TRUE allocation. Comparing before/after gives the "
        "real memory delta.\n\n"
        "**Transpose is non-contiguous but stride-positive.** It "
        "permutes the strides, doesn't zero them. So "
        "`has_zero_stride=False` while `is_contiguous=False`. The two "
        "diagnostics are independent and `.contiguous()` fixes BOTH (it "
        "always returns a stride-(M·N, N, 1)-style tensor)."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — clip-grad-norm-pre-step ex2
# ---------------------------------------------------------------------------

SPEC_CLIP_FROM_SCRATCH = {
    "atom_id": "clip-grad-norm-pre-step",
    "subtopic": "Optimizer: clip_grad_norm pre-step",
    "topic_folder": TOPIC_OPT,
    "atom_recap_md": RECAP_CLIP_FROM_SCRATCH,
    "exercise_index": 2,
    "exercise_title": "reimplement clip_grad_norm_ from scratch and match torch's reference",
    "slug": "reimplement-clip-grad-norm-from-scratch",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["clip-grad", "manual", "global-l2-norm", "in-place"],
    "kcs": [
        "global-l2-norm-from-grad-list",
        "in-place-rescale-when-above-threshold",
    ],
    "lo": (
        "Apply the global-L2-norm clipping algorithm by reimplementing "
        "`torch.nn.utils.clip_grad_norm_` from scratch — collect grads, "
        "compute the global norm, rescale in-place only if total exceeds "
        "`max_norm`, return the pre-clip norm."
    ),
    "prompt_body": (
        "Implement `ex2_clip_grad_norm_manual(params, max_norm)`. The "
        "from-scratch deepening of ex1's library-call variant.\n\n"
        "Algorithm (matches PyTorch's reference):\n"
        "1. Collect `grads = [p.grad for p in params if p.grad is not "
        "None]`.\n"
        "2. If `grads` is empty: return `0.0` (no params have grads).\n"
        "3. Compute the GLOBAL L2 norm: `total = sqrt(sum_i (g_i ** "
        "2).sum())` — concat all grads' squared sums, then sqrt.\n"
        "4. If `total > max_norm`: rescale every grad in-place via "
        "`g.mul_(max_norm / (total + 1e-6))`. The `+ 1e-6` matches "
        "PyTorch's reference (avoids /0 when total is tiny).\n"
        "5. If `total <= max_norm`: do NOT touch the grads.\n"
        "6. Return `total.item()` (pre-clip norm as a Python float).\n\n"
        "Inputs:\n"
        "- `params`: iterable of `nn.Parameter` or any tensors with a "
        "`.grad` attribute.\n"
        "- `max_norm`: `float`.\n\n"
        "Output: `float` — pre-clip global norm.\n\n"
        "Constraints:\n"
        "- Do NOT call `torch.nn.utils.clip_grad_norm_` or "
        "`torch.nn.utils.clip_grad_value_` — write the math.\n"
        "- Use `.detach()` when reading grads for the norm computation "
        "(don't build a graph through the clip).\n"
        "- The in-place rescale must mutate the original `.grad` tensors "
        "(downstream optimizer reads them by reference)."
    ),
    "stub": (
        "def ex2_clip_grad_norm_manual(params, max_norm: float) -> float:\n"
        '    """Reimplement clip_grad_norm_ over a global L2 norm; return pre-clip norm."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "import torch.nn.utils as nn_utils\n"
        "\n"
        "# === Case 1: norm > max_norm, all grads rescaled by max_norm/total ===\n"
        "p1 = t.nn.Parameter(t.zeros(2))\n"
        "p2 = t.nn.Parameter(t.zeros(2))\n"
        "p1.grad = t.tensor([3.0, 4.0])     # contributes 25 to sumsq\n"
        "p2.grad = t.tensor([0.0, 0.0])     # contributes 0\n"
        "# Global norm = sqrt(25) = 5.\n"
        "total = ex2_clip_grad_norm_manual([p1, p2], max_norm=1.0)\n"
        "assert isinstance(total, float), f'must return Python float, got {type(total).__name__}'\n"
        "assert math.isclose(total, 5.0, rel_tol=1e-5), f'pre-clip norm should be 5.0, got {total}'\n"
        "# Scale = 1.0 / (5.0 + 1e-6) ≈ 0.2. So p1.grad ≈ [0.6, 0.8].\n"
        "assert t.allclose(p1.grad, t.tensor([0.6, 0.8]), atol=1e-3), (\n"
        "    f'p1.grad should be rescaled to [0.6, 0.8], got {p1.grad}'\n"
        ")\n"
        "assert t.allclose(p2.grad, t.tensor([0.0, 0.0]), atol=1e-6)\n"
        "\n"
        "# === Case 2: norm < max_norm, grads UNCHANGED ===\n"
        "p = t.nn.Parameter(t.zeros(2))\n"
        "p.grad = t.tensor([0.3, 0.4])      # norm = 0.5\n"
        "total = ex2_clip_grad_norm_manual([p], max_norm=1.0)\n"
        "assert math.isclose(total, 0.5, rel_tol=1e-5)\n"
        "# No clipping: grad unchanged.\n"
        "assert t.equal(p.grad, t.tensor([0.3, 0.4])), (\n"
        "    f'no clipping should occur when norm < max_norm; got {p.grad}'\n"
        ")\n"
        "\n"
        "# === Case 3: matches PyTorch's clip_grad_norm_ exactly ===\n"
        "t.manual_seed(42)\n"
        "p1 = t.nn.Parameter(t.randn(3, 4))\n"
        "p2 = t.nn.Parameter(t.randn(5))\n"
        "p1.grad = t.randn(3, 4) * 10\n"
        "p2.grad = t.randn(5) * 10\n"
        "# Reference: clone the grads and run torch's clip.\n"
        "p1_ref = t.nn.Parameter(p1.detach().clone())\n"
        "p2_ref = t.nn.Parameter(p2.detach().clone())\n"
        "p1_ref.grad = p1.grad.clone()\n"
        "p2_ref.grad = p2.grad.clone()\n"
        "ref_norm = nn_utils.clip_grad_norm_([p1_ref, p2_ref], max_norm=1.5).item()\n"
        "# Now run our implementation.\n"
        "our_norm = ex2_clip_grad_norm_manual([p1, p2], max_norm=1.5)\n"
        "assert math.isclose(our_norm, ref_norm, rel_tol=1e-4), (\n"
        "    f'our pre-clip norm {our_norm} != torch ref {ref_norm}'\n"
        ")\n"
        "assert t.allclose(p1.grad, p1_ref.grad, atol=1e-4), (\n"
        "    f'rescaled p1.grad mismatch:\\nours={p1.grad}\\nref ={p1_ref.grad}'\n"
        ")\n"
        "assert t.allclose(p2.grad, p2_ref.grad, atol=1e-4)\n"
        "\n"
        "# === Case 4: None grads skipped silently ===\n"
        "p1 = t.nn.Parameter(t.zeros(2))\n"
        "p2 = t.nn.Parameter(t.zeros(2))\n"
        "p1.grad = t.tensor([3.0, 4.0])\n"
        "p2.grad = None                       # frozen / never participated in loss\n"
        "total = ex2_clip_grad_norm_manual([p1, p2], max_norm=1.0)\n"
        "assert math.isclose(total, 5.0, rel_tol=1e-5)\n"
        "# Only p1.grad was rescaled. p2.grad still None.\n"
        "assert p2.grad is None\n"
        "\n"
        "# === Case 5: all-None grads → return 0.0 ===\n"
        "p1 = t.nn.Parameter(t.zeros(2))\n"
        "p2 = t.nn.Parameter(t.zeros(2))\n"
        "p1.grad = None\n"
        "p2.grad = None\n"
        "total = ex2_clip_grad_norm_manual([p1, p2], max_norm=1.0)\n"
        "assert total == 0.0, f'all-None grads should give 0.0, got {total}'\n"
        "\n"
        "# === Case 6: in-place mutation (same data_ptr before and after) ===\n"
        "p = t.nn.Parameter(t.zeros(2))\n"
        "p.grad = t.tensor([10.0, 0.0])   # norm 10\n"
        "ptr_before = p.grad.data_ptr()\n"
        "ex2_clip_grad_norm_manual([p], max_norm=1.0)\n"
        "assert p.grad.data_ptr() == ptr_before, (\n"
        "    'must rescale in-place; allocating new grad tensor breaks optimizer reference'\n"
        ")\n"
        "\n"
        "# === Case 7: direction preserved (scalar rescale only) ===\n"
        "p = t.nn.Parameter(t.zeros(3))\n"
        "p.grad = t.tensor([6.0, 8.0, 0.0])   # norm 10\n"
        "ex2_clip_grad_norm_manual([p], max_norm=2.0)\n"
        "# Direction must be preserved.\n"
        "actual_norm = p.grad.norm().item()\n"
        "assert math.isclose(actual_norm, 2.0, rel_tol=1e-3), (\n"
        "    f'post-clip norm should be 2.0, got {actual_norm}'\n"
        ")\n"
        "# Ratio check: orig 6:8:0 → scaled 1.2:1.6:0.\n"
        "assert t.allclose(p.grad, t.tensor([1.2, 1.6, 0.0]), atol=1e-3)"
    ),
    "solution_body": (
        "def ex2_clip_grad_norm_manual(params, max_norm):\n"
        "    grads = [p.grad for p in params if p.grad is not None]\n"
        "    if not grads:\n"
        "        return 0.0\n"
        "    total_sq = sum((g.detach() ** 2).sum() for g in grads)\n"
        "    total = total_sq.sqrt()\n"
        "    if total.item() > max_norm:\n"
        "        scale = max_norm / (total + 1e-6)\n"
        "        for g in grads:\n"
        "            g.mul_(scale)\n"
        "    return total.item()"
    ),
    "solution_notes": (
        "**Global norm, not per-tensor norms.** The clip is on the "
        "CONCATENATED gradient vector. `sqrt(sum_i ||g_i||^2)` is the L2 "
        "norm of the flattened concat — equivalent to `sqrt(sumsq across "
        "ALL params)`. Per-tensor clipping (norm each tensor "
        "independently to max_norm) is a DIFFERENT algorithm and gives "
        "different results.\n\n"
        "**`+ 1e-6` in the denominator.** Matches PyTorch's reference "
        "implementation. Guards against div-by-zero when total is "
        "subnormal (rare in practice, but the library guards it so we do "
        "too — this is also what makes our output bit-identical to "
        "theirs).\n\n"
        "**`.mul_(scale)` not `g *= scale`.** Both mutate in place, but "
        "`g.mul_(scale)` is the canonical PyTorch idiom and is what the "
        "reference does. Avoids the trap where `g = g * scale` "
        "REBINDS g to a new tensor (and the optimizer still holds a "
        "reference to the OLD one)."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — ema-second-moment ex2
# ---------------------------------------------------------------------------

SPEC_EMA_ADAPTIVE_SCALE = {
    "atom_id": "ema-second-moment",
    "subtopic": "Optimizer: Adam EMA second moment",
    "topic_folder": TOPIC_OPT,
    "atom_recap_md": RECAP_EMA_ADAPTIVE_SCALE,
    "exercise_index": 2,
    "exercise_title": "derive Adam's per-coordinate adaptive step-scale from the v buffer",
    "slug": "adam-adaptive-step-scale-from-v-buffer",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["adam", "adaptive-lr", "step-scale", "second-moment"],
    "kcs": [
        "step-scale-equals-one-over-sqrt-v-plus-eps",
        "big-v-small-scale-small-v-big-scale",
    ],
    "lo": (
        "Apply the Adam denominator formula `step_scale = 1 / (sqrt(v) + "
        "eps)` to a v-buffer tensor and verify the per-coordinate "
        "adaptive-lr signature — big-grad-history coords get small "
        "scales, small-grad-history coords get large scales."
    ),
    "prompt_body": (
        "Implement `ex2_adam_step_scale(v, eps)`. The deepening variant "
        "of ex1.\n\n"
        "Inputs:\n"
        "- `v`: `torch.Tensor`, the Adam second-moment buffer "
        "(elementwise, non-negative). May contain zeros (early-step "
        "buffer).\n"
        "- `eps`: `float`, the Adam epsilon (typical 1e-8).\n\n"
        "Compute:\n\n"
        "`step_scale = 1.0 / (v.sqrt() + eps)`\n\n"
        "Return a dict with EXACTLY these keys:\n\n"
        "- `'step_scale'`: `torch.Tensor`, same shape as `v`.\n"
        "- `'step_scale_at_v_zero'`: `float`, `1.0 / eps` — the maximum "
        "achievable step-scale (when v=0).\n"
        "- `'max_step_scale'`: `float`, "
        "`step_scale.max().item()`. Must be `<= step_scale_at_v_zero + "
        "1e-6`.\n"
        "- `'min_step_scale'`: `float`, `step_scale.min().item()`.\n"
        "- `'v_at_max_scale_idx'`: `int`, the FLAT index of the v entry "
        "that produced the LARGEST step_scale (use "
        "`v.argmin().item()` — smallest v → largest scale).\n"
        "- `'v_at_min_scale_idx'`: `int`, "
        "`v.argmax().item()` — largest v → smallest scale.\n\n"
        "Constraints:\n"
        "- Use `v.sqrt()` not `t.sqrt(v)` (equivalent, but consistent "
        "with ARENA's tensor-method style).\n"
        "- Use Python `/` and `+` — they dispatch to elementwise ops on "
        "tensors.\n"
        "- Do not mutate `v`."
    ),
    "stub": (
        "def ex2_adam_step_scale(v: Tensor, eps: float) -> dict:\n"
        '    """Compute Adam adaptive step-scale 1/(sqrt(v)+eps) and per-coord stats."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "\n"
        "# === Hand-computed: v = [0.0, 1.0, 100.0], eps=0.01 ===\n"
        "v = t.tensor([0.0, 1.0, 100.0])\n"
        "eps = 0.01\n"
        "d = ex2_adam_step_scale(v, eps)\n"
        "# step_scale = 1 / (sqrt(v) + eps):\n"
        "#   v=0 -> 1/(0 + 0.01) = 100\n"
        "#   v=1 -> 1/(1 + 0.01) ≈ 0.9901\n"
        "#   v=100 -> 1/(10 + 0.01) ≈ 0.0999\n"
        "assert d['step_scale'].shape == v.shape\n"
        "assert math.isclose(d['step_scale'][0].item(), 100.0, rel_tol=1e-5)\n"
        "assert math.isclose(d['step_scale'][1].item(), 1.0 / 1.01, rel_tol=1e-5)\n"
        "assert math.isclose(d['step_scale'][2].item(), 1.0 / 10.01, rel_tol=1e-5)\n"
        "\n"
        "# === The big-v-small-scale, small-v-big-scale signature ===\n"
        "assert d['max_step_scale'] == d['step_scale'][0].item()    # at v=0\n"
        "assert d['min_step_scale'] == d['step_scale'][2].item()    # at v=100\n"
        "assert d['v_at_max_scale_idx'] == 0     # smallest v\n"
        "assert d['v_at_min_scale_idx'] == 2     # largest v\n"
        "\n"
        "# === The v=0 ceiling is exactly 1/eps ===\n"
        "assert math.isclose(d['step_scale_at_v_zero'], 1.0 / 0.01, rel_tol=1e-5)\n"
        "assert d['max_step_scale'] <= d['step_scale_at_v_zero'] + 1e-6\n"
        "\n"
        "# === Realistic Adam config: eps=1e-8 ===\n"
        "v = t.tensor([1e-12, 1e-4, 1.0, 1e4])   # spanning 16 orders of magnitude\n"
        "eps = 1e-8\n"
        "d = ex2_adam_step_scale(v, eps)\n"
        "# v=1e-12 gives sqrt=1e-6 → step_scale ≈ 1/(1e-6 + 1e-8) ≈ 9.9e5\n"
        "# v=1e4   gives sqrt=100  → step_scale ≈ 1/100.00000001 ≈ 0.01\n"
        "assert d['step_scale'][3].item() < d['step_scale'][2].item() < d['step_scale'][1].item() < d['step_scale'][0].item(), (\n"
        "    f'step_scale must be monotone decreasing in v; got {d[\"step_scale\"]}'\n"
        ")\n"
        "assert d['v_at_max_scale_idx'] == 0\n"
        "assert d['v_at_min_scale_idx'] == 3\n"
        "\n"
        "# === Higher-rank v works (Adam's v shares the param's shape) ===\n"
        "v = t.tensor([[0.0, 0.25], [1.0, 4.0]])    # shape (2, 2)\n"
        "d = ex2_adam_step_scale(v, eps=1e-3)\n"
        "assert d['step_scale'].shape == (2, 2)\n"
        "# Hand-check (0, 0): 1/(0 + 1e-3) = 1000\n"
        "assert math.isclose(d['step_scale'][0, 0].item(), 1000.0, rel_tol=1e-3)\n"
        "# Hand-check (1, 1): 1/(2 + 1e-3) ≈ 0.4998\n"
        "assert math.isclose(d['step_scale'][1, 1].item(), 1.0 / 2.001, rel_tol=1e-4)\n"
        "# argmin/argmax over the FLAT layout:\n"
        "assert d['v_at_max_scale_idx'] == 0   # flat index 0 = v=0.0\n"
        "assert d['v_at_min_scale_idx'] == 3   # flat index 3 = v=4.0\n"
        "\n"
        "# === All entries non-negative ===\n"
        "v = t.tensor([0.0, 0.5, 1.0, 2.0, 10.0])\n"
        "d = ex2_adam_step_scale(v, eps=1e-8)\n"
        "assert (d['step_scale'] > 0).all(), 'step_scale must be strictly positive'\n"
        "\n"
        "# === Input v is not mutated ===\n"
        "v = t.tensor([1.0, 4.0, 9.0])\n"
        "v_clone = v.clone()\n"
        "_ = ex2_adam_step_scale(v, eps=1e-8)\n"
        "assert t.equal(v, v_clone), 'must not mutate v buffer'\n"
        "\n"
        "# === All keys ===\n"
        "expected_keys = {'step_scale', 'step_scale_at_v_zero', 'max_step_scale',\n"
        "                 'min_step_scale', 'v_at_max_scale_idx', 'v_at_min_scale_idx'}\n"
        "assert set(d.keys()) == expected_keys, f'keys wrong: {set(d.keys())}'"
    ),
    "solution_body": (
        "def ex2_adam_step_scale(v, eps):\n"
        "    step_scale = 1.0 / (v.sqrt() + eps)\n"
        "    return {\n"
        "        'step_scale': step_scale,\n"
        "        'step_scale_at_v_zero': 1.0 / eps,\n"
        "        'max_step_scale': step_scale.max().item(),\n"
        "        'min_step_scale': step_scale.min().item(),\n"
        "        'v_at_max_scale_idx': v.argmin().item(),\n"
        "        'v_at_min_scale_idx': v.argmax().item(),\n"
        "    }"
    ),
    "solution_notes": (
        "**`v.argmin()` finds the MAX-scale coordinate.** Because "
        "`step_scale` is strictly decreasing in `v` (when v >= 0), the "
        "smallest v gives the largest scale. We can find the argmax of "
        "`step_scale` directly OR the argmin of `v` — same index. "
        "Doing it via `v.argmin()` avoids needing the step_scale tensor "
        "for the lookup.\n\n"
        "**Why eps is OUTSIDE the sqrt for Adam.** Unlike BatchNorm "
        "(where eps INSIDE the sqrt bounds the backward gradient), Adam's "
        "eps is a forward-pass division floor. The Kingma & Ba paper has "
        "it outside; PyTorch follows. With eps=1e-8 and v=0, step_scale "
        "= 1e8 — huge but finite.\n\n"
        "**The 1e8 step-scale ceiling is the early-training trap.** "
        "Before grad accumulation, v is near zero everywhere. Adam's "
        "denominator is tiny → step_scale is huge → updates can be "
        "explosive. The standard fix is BIAS CORRECTION (divide v by "
        "`1 - beta2^t`, which is small at t=1 so v_hat is INFLATED to "
        "the steady-state magnitude). That correction is a separate "
        "atom — this drill focuses just on the scale formula."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_ENCDEC_BREAK,
    SPEC_KAIMING_FAN,
    SPEC_LOSS_ITEM_RAISES,
    SPEC_REARRANGE_PATCHIFY,
    SPEC_SQRT_EPS_PLACEMENT,
    SPEC_STRIDE_CONTIGUOUS,
    SPEC_CLIP_FROM_SCRATCH,
    SPEC_EMA_ADAPTIVE_SCALE,
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
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
            "nn": nn,
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
    print(f"[deepening_x_batch12] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_x_batch12] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_x_batch12] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
