#!/usr/bin/env python3
"""Author 8 deepening ex2 notebooks for single-drill ARENA atoms (batch D).

Each spec adds ONE new ex2 that probes a DISTINCT facet from the existing ex1
in the same folder. PS4 framing — one LO, one Bloom, max 2 KCs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ============================================================== 1: unbind-tuple-unpack
# ex1: two-level destructure on rays (B,2,3) -> ox,oy,oz,dx,dy,dz
# ex2: split QKV from packed-attention tensor (3, B, H, S, D) -> Q,K,V each (B,H,S,D)
SPEC_UNBIND = {
    "atom_id": "unbind-tuple-unpack",
    "subtopic": "PyTorch: Unbind tuple-unpack",
    "topic_folder": "prereqs_einops_advanced",
    "atom_recap_md": (
        "## unbind tuple-unpack — quick refresher\n"
        "\n"
        "`x.unbind(dim=k)` returns a **Python tuple** of `x.shape[k]` view-tensors "
        "with axis `k` removed. Tuple-unpacking the result gives you named slices "
        "without any indexing arithmetic:\n"
        "\n"
        "```python\n"
        "q, k, v = qkv_stack.unbind(dim=0)\n"
        "```\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 destructured rays `(B,2,3)` into "
        "`ox/oy/oz/dx/dy/dz` (a two-level decomposition on a small leading axis). "
        "ex2 destructures the canonical QKV-stack-leading-axis pattern from a "
        "packed attention projection — same single `unbind` op, different "
        "semantic axis (a length-3 *triple-of-tensors* leading axis, not a "
        "geometric `xyz` axis)."
    ),
    "exercise_index": 2,
    "exercise_title": "split packed QKV stack into named Q, K, V tensors",
    "slug": "split-packed-qkv-stack-into-named-q-k-v",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["attention", "qkv", "unbind-dim-0", "named-destructure"],
    "kcs": ["unbind-returns-python-tuple", "unbind-leading-axis-destructure"],
    "lo": (
        "Apply `unbind(dim=0)` to a packed `(3, B, H, S, D)` QKV-stack tensor "
        "from a fused attention projection, producing three named "
        "`(B, H, S, D)` tensors `q, k, v` ready for per-tensor downstream ops."
    ),
    "prompt_body": (
        "Implement `ex2_split_qkv(qkv)`.\n\n"
        "A fused attention projection produces a single tensor `qkv` of shape "
        "`(3, B, H, S, D)` — the leading length-3 axis stacks Q, K, V (in that "
        "order). Split it into three named tensors with `unbind`.\n\n"
        "**Rules.**\n"
        "1. Use `qkv.unbind(dim=0)` and tuple-unpack the result.\n"
        "2. Return a dict `{'q': ..., 'k': ..., 'v': ...}` so the caller can "
        "name-address each tensor.\n"
        "3. Each value must have shape `(B, H, S, D)`. No reshape / index "
        "arithmetic — destructure only.\n"
        "4. Verify the views still share storage with `qkv` (they're views, not "
        "copies). The test asserts this via `data_ptr()`.\n\n"
        "Inputs:\n"
        "- `qkv`: `(3, B, H, S, D)` float tensor.\n\n"
        "Output: dict with keys `'q'`, `'k'`, `'v'`, each a `(B, H, S, D)` view."
    ),
    "stub": (
        "def ex2_split_qkv(qkv: Tensor) -> dict:\n"
        '    """Unbind a (3, B, H, S, D) packed QKV stack into named Q/K/V views."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "B, H, S, D = 2, 4, 5, 8\n"
        "# Build a deterministic packed QKV so we can identify slices by value.\n"
        "qkv = t.stack([\n"
        "    t.full((B, H, S, D), 1.0),   # Q := all 1.0\n"
        "    t.full((B, H, S, D), 2.0),   # K := all 2.0\n"
        "    t.full((B, H, S, D), 3.0),   # V := all 3.0\n"
        "], dim=0)\n"
        "assert qkv.shape == (3, B, H, S, D)\n"
        "\n"
        "out = ex2_split_qkv(qkv)\n"
        "assert isinstance(out, dict), f'expected dict, got {type(out).__name__}'\n"
        "assert set(out.keys()) == {'q', 'k', 'v'}, f'wrong keys: {set(out.keys())}'\n"
        "\n"
        "for name, expected_val in [('q', 1.0), ('k', 2.0), ('v', 3.0)]:\n"
        "    arr = out[name]\n"
        "    assert arr.shape == (B, H, S, D), f'{name}: shape {tuple(arr.shape)} != {(B,H,S,D)}'\n"
        "    assert arr.dtype == t.float32, f'{name}: dtype {arr.dtype}'\n"
        "    assert t.all(arr == expected_val), f'{name}: not all {expected_val}'\n"
        "\n"
        "# Views, not copies — storage shared with the source.\n"
        "assert out['q'].data_ptr() == qkv.data_ptr(), 'q must be a view of qkv (data_ptr mismatch)'\n"
        "# k and v live at later offsets in the same storage.\n"
        "assert out['k'].storage().data_ptr() == qkv.storage().data_ptr(), 'k must share storage with qkv'\n"
        "assert out['v'].storage().data_ptr() == qkv.storage().data_ptr(), 'v must share storage with qkv'\n"
        "\n"
        "# Random-data smoke test — slice 1 of unbind must equal qkv[1].\n"
        "rng = t.Generator().manual_seed(0)\n"
        "qkv2 = t.randn(3, 1, 2, 3, 4, generator=rng)\n"
        "out2 = ex2_split_qkv(qkv2)\n"
        "assert t.equal(out2['q'], qkv2[0]), 'q must equal qkv[0]'\n"
        "assert t.equal(out2['k'], qkv2[1]), 'k must equal qkv[1]'\n"
        "assert t.equal(out2['v'], qkv2[2]), 'v must equal qkv[2]'"
    ),
    "solution_body": (
        "def ex2_split_qkv(qkv: Tensor) -> dict:\n"
        "    q, k, v = qkv.unbind(dim=0)\n"
        "    return {'q': q, 'k': k, 'v': v}"
    ),
    "solution_notes": (
        "**Why `unbind(dim=0)` and not `qkv[0], qkv[1], qkv[2]`.** Both work, "
        "but `unbind` makes the destructure intent explicit at the call-site "
        "(\"peel the leading axis into a tuple\") and composes cleanly with "
        "Python tuple-unpack. The indexed form looks like three independent "
        "operations even though they're one decomposition.\n\n"
        "**Why this is the canonical QKV pattern.** Fused attention "
        "projections (e.g. nanoGPT, ViT, transformer encoder blocks) produce a "
        "single `(3*D, B, ...)` projection then `view(3, D, B, ...)` and "
        "unbind. The unbind step is the natural \"untie the three tensors\" "
        "boundary.\n\n"
        "**Difference from ex1.** ex1 was a two-level unbind on a `(B,2,3)` "
        "ray tensor (inner-axis-and-then-last-axis decomposition over a "
        "geometric `xyz` axis). ex2 is a single-level unbind on the LEADING "
        "length-3 axis of a packed-projection tensor — same op, different "
        "semantic axis position."
    ),
    "extra_imports": [],
}


# ============================================================== 2: einops-rearrange-flatten
# ex1: flatten 'b c h w -> b (c h w)' for CNN→Linear; ex2: UNFLATTEN inverse 'b (c h w) -> b c h w'
SPEC_FLATTEN = {
    "atom_id": "einops-rearrange-flatten",
    "subtopic": "Einops: Rearrange-as-flatten",
    "topic_folder": "prereqs_einops_advanced",
    "atom_recap_md": (
        "## einops.rearrange — flatten and unflatten\n"
        "\n"
        "Parenthesised axes in einops can do *composition* (flatten) **or** "
        "*decomposition* (unflatten), depending on which side of the arrow they "
        "appear on. To unflatten you must supply the named sizes:\n"
        "\n"
        "```python\n"
        "x = rearrange(flat, 'b (c h w) -> b c h w', c=C, h=H, w=W)\n"
        "```\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 did the **forward** flatten "
        "`'b c h w -> b (c h w)'` (CNN feature map → linear-head input). ex2 "
        "does the **inverse** un-flatten `'b (c h w) -> b c h w'` — the move "
        "you need going from a linear layer's flat output BACK to a spatial "
        "feature map (e.g. the first stage of a decoder / generator)."
    ),
    "exercise_index": 2,
    "exercise_title": "unflatten a linear-projected vector back to a CNN feature map",
    "slug": "unflatten-linear-projection-to-cnn-feature-map",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["unflatten", "decoder", "rearrange", "axis-decomposition"],
    "kcs": ["rearrange-axis-decomposition-via-parens", "rearrange-named-sizes-required"],
    "lo": (
        "Apply `einops.rearrange` with the `'b (c h w) -> b c h w'` pattern "
        "to decompose a 2-D `(B, C*H*W)` linear-projection output back into a "
        "4-D `(B, C, H, W)` CNN feature map, supplying the named axis sizes."
    ),
    "prompt_body": (
        "Implement `ex2_unflatten_to_feature_map(flat, C, H, W)`.\n\n"
        "A `Linear` layer produced a 2-D batch of vectors `flat` of shape "
        "`(B, C*H*W)`. Reshape it into a 4-D feature map `(B, C, H, W)` using "
        "the einops decomposition pattern.\n\n"
        "**Rules.**\n"
        "1. Use `einops.rearrange` with the `'b (c h w) -> b c h w'` pattern.\n"
        "2. You **must** pass `c=C, h=H, w=W` as keyword args — einops can't "
        "infer how to split the composite axis without them.\n"
        "3. The mapping must preserve the row-major ordering: the leftmost "
        "axis in the parenthesis (`c`) varies SLOWEST, the rightmost (`w`) "
        "varies FASTEST. Equivalently, `flat[b, c*H*W + h*W + w] == out[b, c, h, w]`.\n\n"
        "Inputs:\n"
        "- `flat`: `(B, C*H*W)` float tensor.\n"
        "- `C`, `H`, `W`: int axis sizes.\n\n"
        "Output: `(B, C, H, W)` float tensor."
    ),
    "stub": (
        "def ex2_unflatten_to_feature_map(flat: Tensor, C: int, H: int, W: int) -> Tensor:\n"
        '    """Decompose (B, C*H*W) → (B, C, H, W) via einops rearrange."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Build a deterministic flat vector then unflatten.\n"
        "B, C, H, W = 2, 3, 4, 5\n"
        "flat = t.arange(B * C * H * W, dtype=t.float32).reshape(B, C * H * W)\n"
        "out = ex2_unflatten_to_feature_map(flat, C, H, W)\n"
        "assert out.shape == (B, C, H, W), f'shape {tuple(out.shape)} != {(B,C,H,W)}'\n"
        "assert out.dtype == t.float32, f'dtype {out.dtype}'\n"
        "\n"
        "# Verify row-major mapping cell-by-cell.\n"
        "for b in range(B):\n"
        "    for c in range(C):\n"
        "        for h in range(H):\n"
        "            for w in range(W):\n"
        "                expected = flat[b, c * H * W + h * W + w].item()\n"
        "                got = out[b, c, h, w].item()\n"
        "                assert got == expected, f'cell ({b},{c},{h},{w}): got {got}, expected {expected}'\n"
        "\n"
        "# Round-trip identity: unflatten then flatten back returns the original.\n"
        "round_trip = einops.rearrange(out, 'b c h w -> b (c h w)')\n"
        "assert t.equal(round_trip, flat), 'round-trip unflatten→flatten failed'\n"
        "\n"
        "# Different shape — non-square decoder use-case (B=1, project to 256x8x8).\n"
        "rng = t.Generator().manual_seed(1)\n"
        "z = t.randn(1, 256 * 8 * 8, generator=rng)\n"
        "feat = ex2_unflatten_to_feature_map(z, 256, 8, 8)\n"
        "assert feat.shape == (1, 256, 8, 8), f'decoder-shape: {tuple(feat.shape)}'\n"
        "# A different size assignment of the same flat input must produce a different shape.\n"
        "feat2 = ex2_unflatten_to_feature_map(z, 64, 16, 16)\n"
        "assert feat2.shape == (1, 64, 16, 16), 'must respect supplied named sizes'"
    ),
    "solution_body": (
        "def ex2_unflatten_to_feature_map(flat: Tensor, C: int, H: int, W: int) -> Tensor:\n"
        "    return einops.rearrange(flat, 'b (c h w) -> b c h w', c=C, h=H, w=W)"
    ),
    "solution_notes": (
        "**Why einops needs the named sizes.** When you flatten "
        "`'b c h w -> b (c h w)'` einops can infer the composite size from "
        "the input. Going the OTHER way, einops can't know how to split a "
        "single composite axis into three — `60 = 3·4·5` is one of many "
        "factorisations. You must supply `c=C, h=H, w=W` to disambiguate.\n\n"
        "**Mapping order matters.** The leftmost axis in the parenthesis "
        "varies slowest. `'b (c h w) -> b c h w'` and `'b (w h c) -> b c h w'` "
        "produce DIFFERENT outputs from the same input — the latter is a "
        "transposed feature map. einops won't catch the bug; the row-major "
        "convention is on you.\n\n"
        "**Difference from ex1.** ex1 was the forward flatten "
        "`'b c h w -> b (c h w)'` used to feed a CNN feature map into a "
        "`Linear` classifier head. ex2 is the inverse — the move you make in "
        "a decoder when going from a `Linear(latent → C·H·W)` projection BACK "
        "to a spatial feature map. Same op family, opposite direction."
    ),
    "extra_imports": [],
}


# ============================================================== 3: einops-reduce-min
# ex1: per-channel spatial floor 'b c h w -> b c' with 'min'
# ex2: min-pool via 'b c (h p1) (w p2) -> b c h w' with reduction='min'
SPEC_REDUCE_MIN = {
    "atom_id": "einops-reduce-min",
    "subtopic": "Einops: Reduce with min",
    "topic_folder": "prereqs_tensor_utils",
    "atom_recap_md": (
        "## einops.reduce — patch-pool patterns\n"
        "\n"
        "`einops.reduce` is the toolkit's most expressive op: it can replicate "
        "ALL of `mean/max/min/sum` AND any window-pool by adding "
        "parenthesised composites on the LHS that get reduced:\n"
        "\n"
        "```python\n"
        "out = reduce(x, 'b c (h p1) (w p2) -> b c h w', 'min', p1=2, p2=2)\n"
        "```\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 used the simple `'b c h w -> b c'` "
        "shape (collapse ALL spatial axes to a single per-channel floor). ex2 "
        "uses the much more interesting **patch-pool** shape — collapse "
        "`p1×p2` sub-blocks within `h, w` but KEEP a coarser spatial grid. "
        "Same `'min'` op, structurally richer pattern."
    ),
    "exercise_index": 2,
    "exercise_title": "2x2 min-pool a feature map via einops patch reduction",
    "slug": "2x2-min-pool-feature-map-via-patch-reduction",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["min-pool", "patch-pool", "reduce", "downsample"],
    "kcs": ["reduce-min-op", "reduce-patch-pattern-via-parens"],
    "lo": (
        "Apply `einops.reduce(..., 'min')` with the patch pattern "
        "`'b c (h p1) (w p2) -> b c h w'` to perform 2x2 min-pooling on a "
        "`(B, C, H, W)` feature map, producing a `(B, C, H/2, W/2)` "
        "downsampled tensor."
    ),
    "prompt_body": (
        "Implement `ex2_min_pool_2x2(x)`.\n\n"
        "Given a `(B, C, H, W)` float tensor where `H` and `W` are both "
        "**even**, perform 2x2 min-pooling — partition the spatial grid into "
        "non-overlapping `2×2` patches and take the min within each.\n\n"
        "**Rules.**\n"
        "1. Use `einops.reduce` with the patch pattern "
        "`'b c (h p1) (w p2) -> b c h w'` and `reduction='min'`.\n"
        "2. Pass `p1=2, p2=2` as keyword args.\n"
        "3. The output shape must be `(B, C, H//2, W//2)`. Each output cell "
        "`(h, w)` is the minimum of the four input cells `(2h..2h+1, 2w..2w+1)`.\n"
        "4. Do NOT use `F.max_pool2d` with a negated input — the point is to "
        "express the pool directly via the patch pattern.\n\n"
        "Inputs:\n"
        "- `x`: `(B, C, H, W)` float tensor, `H, W` even.\n\n"
        "Output: `(B, C, H//2, W//2)` float tensor of per-patch minima."
    ),
    "stub": (
        "def ex2_min_pool_2x2(x: Tensor) -> Tensor:\n"
        '    """2x2 min-pool via einops patch reduction."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Hand-crafted (1, 1, 4, 4) input — the patch mins are easy to read off.\n"
        "x = t.tensor([[[[ 4.,  2.,  9.,  7.],\n"
        "                [ 1.,  3.,  8.,  6.],\n"
        "                [12., 10.,  0., -1.],\n"
        "                [11., 13., -2., -3.]]]])\n"
        "# 2x2 patches:  TL min(4,2,1,3)=1   TR min(9,7,8,6)=6\n"
        "#               BL min(12,10,11,13)=10  BR min(0,-1,-2,-3)=-3\n"
        "expected = t.tensor([[[[ 1.,  6.],\n"
        "                       [10., -3.]]]])\n"
        "out = ex2_min_pool_2x2(x)\n"
        "assert out.shape == (1, 1, 2, 2), f'shape {tuple(out.shape)} != (1,1,2,2)'\n"
        "assert t.equal(out, expected), f'wrong pooled values:\\n{out}\\nvs\\n{expected}'\n"
        "\n"
        "# Larger batched test — verify against a direct loop reference.\n"
        "B, C, H, W = 2, 3, 6, 8\n"
        "rng = t.Generator().manual_seed(7)\n"
        "x_big = t.randn(B, C, H, W, generator=rng)\n"
        "out_big = ex2_min_pool_2x2(x_big)\n"
        "assert out_big.shape == (B, C, H // 2, W // 2)\n"
        "for b in range(B):\n"
        "    for c in range(C):\n"
        "        for h in range(H // 2):\n"
        "            for w in range(W // 2):\n"
        "                patch = x_big[b, c, 2*h:2*h+2, 2*w:2*w+2]\n"
        "                ref = patch.min().item()\n"
        "                got = out_big[b, c, h, w].item()\n"
        "                assert abs(got - ref) < 1e-6, f'cell ({b},{c},{h},{w}): got {got}, ref {ref}'\n"
        "\n"
        "# Sanity vs negated max-pool — must agree.\n"
        "import torch.nn.functional as F\n"
        "ref = -F.max_pool2d(-x_big, kernel_size=2, stride=2)\n"
        "assert t.allclose(out_big, ref, atol=1e-6), 'must equal -max_pool2d(-x)'"
    ),
    "solution_body": (
        "def ex2_min_pool_2x2(x: Tensor) -> Tensor:\n"
        "    return einops.reduce(x, 'b c (h p1) (w p2) -> b c h w', 'min', p1=2, p2=2)"
    ),
    "solution_notes": (
        "**Why this pattern works.** Each parenthesised `(h p1)` says: \"the "
        "input has an `h*p1`-long axis; treat it as a grid of `h` patches of "
        "size `p1`\". Any axis present on the LHS but ABSENT from the RHS "
        "gets reduced. So `p1` and `p2` are reduced (collapsed into the "
        "specified `reduction='min'`), while `h` and `w` survive as the "
        "coarser output spatial axes.\n\n"
        "**Why `'min'` is unusual.** PyTorch ships `F.max_pool2d` and "
        "`F.avg_pool2d` but no `min_pool2d` — the standard trick is "
        "`-max_pool2d(-x)`. einops's `reduce` makes the direct expression "
        "possible *without* sign-flipping, which matters when `x` has been "
        "passed through a layer that's sensitive to the sign (e.g. an "
        "intermediate that's clamped to non-negative).\n\n"
        "**Difference from ex1.** ex1 did the *total* spatial floor "
        "`'b c h w -> b c'` — collapse everything to a single value per "
        "channel. ex2 keeps a coarser spatial grid intact, which is what real "
        "min-pool / max-pool layers do."
    ),
    "extra_imports": [],
}


# ============================================================== 4: inf-masking
# ex1: causal attention mask; ex2: pad-mask attention (mask PAD tokens in keys)
SPEC_INF_MASK = {
    "atom_id": "inf-masking",
    "subtopic": "Numpy: Inf-fill masking trick",
    "topic_folder": "prereqs_einops_advanced",
    "atom_recap_md": (
        "## inf-fill masking before softmax — quick refresher\n"
        "\n"
        "Replacing forbidden positions with `-inf` before softmax sends their "
        "weight to exactly zero (since `exp(-inf) = 0`) while leaving the "
        "unmasked positions to renormalise to one. The trick is shape-agnostic:\n"
        "\n"
        "```python\n"
        "masked = scores.masked_fill(forbidden, float('-inf'))\n"
        "weights = masked.softmax(dim=-1)\n"
        "```\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 applied a **causal** mask — a "
        "lower-triangular pattern that forbids future positions, the same for "
        "every batch element. ex2 applies a **pad** mask — a *per-batch* "
        "boolean over the key axis that forbids PAD tokens. Same `-inf` "
        "trick, different mask topology (one is fixed per-sequence-length, "
        "the other varies per-batch-element)."
    ),
    "exercise_index": 2,
    "exercise_title": "pad-mask attention — drop PAD tokens from the keys",
    "slug": "pad-mask-attention-drop-pad-tokens-from-keys",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["pad-mask", "attention", "masked_fill", "per-batch-mask"],
    "kcs": ["masked-fill-neg-inf-before-softmax", "pad-mask-broadcast-on-keys"],
    "lo": (
        "Apply `masked_fill(-inf)` with a per-batch `(B, S_k)` pad mask "
        "broadcast across the query axis to a `(B, S_q, S_k)` attention "
        "scores tensor, producing softmax weights that place zero mass on "
        "PAD-token keys."
    ),
    "prompt_body": (
        "Implement `ex2_pad_mask_attention(scores, pad_mask)`.\n\n"
        "You have raw attention scores `scores` of shape `(B, S_q, S_k)` and "
        "a key-side pad mask `pad_mask` of shape `(B, S_k)`, where "
        "`pad_mask[b, j] = True` iff key position `j` in batch element `b` is "
        "a PAD token that must NOT receive attention.\n\n"
        "**Rules.**\n"
        "1. Broadcast `pad_mask` to `(B, 1, S_k)` so it applies to every "
        "query row but is independent for each batch element.\n"
        "2. Use `masked_fill` with `float('-inf')` to set forbidden positions.\n"
        "3. Softmax along `dim=-1` (the key axis).\n"
        "4. After softmax, the weight on any masked key must be EXACTLY 0 "
        "(not just small). Each row over un-masked keys must sum to 1.0.\n\n"
        "Inputs:\n"
        "- `scores`: `(B, S_q, S_k)` float tensor.\n"
        "- `pad_mask`: `(B, S_k)` bool tensor, `True` = PAD = forbidden.\n\n"
        "Output: `(B, S_q, S_k)` softmax weights with zero mass on PAD keys.\n\n"
        "The visualization renders the post-softmax attention weights as a "
        "heatmap per batch element so you can see the masked columns "
        "disappear."
    ),
    "stub": (
        "def ex2_pad_mask_attention(scores: Tensor, pad_mask: Tensor) -> Tensor:\n"
        '    """Pad-masked softmax: zero attention weight on PAD-token keys."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Hand-built scores so the masked weights are predictable.\n"
        "scores = t.tensor([\n"
        "    # batch 0: last two keys are PAD\n"
        "    [[0.0, 0.0, 5.0, 5.0],\n"
        "     [1.0, 1.0, 0.0, 0.0],\n"
        "     [2.0, 0.0, 0.0, 0.0]],\n"
        "    # batch 1: only key 0 is valid (extreme)\n"
        "    [[3.0, 0.0, 0.0, 0.0],\n"
        "     [4.0, 9.9, 9.9, 9.9],\n"
        "     [0.0, 9.9, 9.9, 9.9]],\n"
        "])\n"
        "pad_mask = t.tensor([\n"
        "    [False, False, True,  True],\n"
        "    [False, True,  True,  True],\n"
        "])\n"
        "w = ex2_pad_mask_attention(scores, pad_mask)\n"
        "assert w.shape == scores.shape\n"
        "assert w.dtype == t.float32\n"
        "\n"
        "# Masked positions must be exactly zero.\n"
        "assert t.all(w[0, :, 2:] == 0), f'b=0 PAD columns not zero:\\n{w[0]}'\n"
        "assert t.all(w[1, :, 1:] == 0), f'b=1 PAD columns not zero:\\n{w[1]}'\n"
        "\n"
        "# Rows sum to 1 — only over un-masked keys, but softmax renormalises.\n"
        "row_sums = w.sum(dim=-1)\n"
        "assert t.allclose(row_sums, t.ones_like(row_sums), atol=1e-6), f'row sums:\\n{row_sums}'\n"
        "\n"
        "# Batch 1: only key 0 is valid → its weight must be 1.0 for every query.\n"
        "assert t.allclose(w[1, :, 0], t.ones(3), atol=1e-6), f'b=1 single-valid-key not 1.0: {w[1, :, 0]}'\n"
        "\n"
        "# Batch 0 row 0: scores [0, 0] over un-masked keys → uniform [0.5, 0.5].\n"
        "assert t.allclose(w[0, 0, :2], t.tensor([0.5, 0.5]), atol=1e-6), f'uniform-over-valid wrong: {w[0,0,:2]}'\n"
        "\n"
        "# --- Heatmap visualization ---\n"
        "fig, axes = plt.subplots(1, 2, figsize=(8, 3))\n"
        "for b in range(2):\n"
        "    axes[b].imshow(w[b].numpy(), cmap='magma', vmin=0, vmax=1, aspect='auto')\n"
        "    axes[b].set_title(f'batch {b} — PAD cols ' + str([j for j, m in enumerate(pad_mask[b].tolist()) if m]))\n"
        "    axes[b].set_xlabel('key index')\n"
        "    axes[b].set_ylabel('query index')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_pad_mask_attention(scores: Tensor, pad_mask: Tensor) -> Tensor:\n"
        "    # (B, S_k) -> (B, 1, S_k) so it broadcasts across queries.\n"
        "    masked = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))\n"
        "    return masked.softmax(dim=-1)"
    ),
    "solution_notes": (
        "**Why broadcast on `dim=1`.** The pad mask only knows about KEYS "
        "(which sequence positions are PAD), not queries. The forbidden set "
        "is the same for every query but varies per batch element — so the "
        "shape must broadcast `(B, S_k) -> (B, 1, S_k) -> (B, S_q, S_k)`.\n\n"
        "**Why `-inf` and not a large negative.** `exp(-1e9) ≈ 0` will work "
        "for normal-precision softmax, but `exp(-inf) == 0` exactly. The "
        "test checks `== 0` (not `< 1e-9`) precisely to catch the "
        "`large-negative-but-not-inf` mistake.\n\n"
        "**Difference from ex1.** ex1's causal mask was a `(T, T)` "
        "lower-triangular pattern that depended only on positions (same "
        "across batch). ex2's pad mask is `(B, S_k)` — every batch element "
        "has its OWN PAD pattern, so the mask must broadcast through the "
        "batch axis correctly. The `unsqueeze(1)` on the query axis (not the "
        "batch axis) is the crucial detail."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 5: singular-matrix-mask-trick
# ex1: detect singular slices via det, overwrite with I, solve; ex2: block-diagonal
# partial-singular — split into BLOCK sub-batch, detect per-block, solve only valid blocks.
SPEC_SINGULAR = {
    "atom_id": "singular-matrix-mask-trick",
    "subtopic": "Numpy: Singular matrix mask trick",
    "topic_folder": "prereqs_geometry_cnn",
    "atom_recap_md": (
        "## singular-matrix mask trick — quick refresher\n"
        "\n"
        "When a batched `linalg.solve` would otherwise crash on singular "
        "slices, the standard rescue is:\n"
        "\n"
        "1. Detect singular slices: `is_sing = la.det(A).abs() < eps`.\n"
        "2. Overwrite those slices with the identity (any safe invertible "
        "matrix would do): `A_safe[is_sing] = t.eye(N)`.\n"
        "3. Solve everywhere; return `(x, is_valid)` so the caller can mask "
        "the bogus rows downstream.\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 ran the trick over a flat batch of "
        "`(B, N, N)` matrices (whole-slice singular). ex2 extends it to a "
        "**block-diagonal** structure: the matrix decomposes into two "
        "`(N/2, N/2)` blocks, and singularity is detected **per block** "
        "(some blocks invertible, others not). The trick generalises one "
        "level — same det/identity-replace pattern, applied to a "
        "block-batched view."
    ),
    "exercise_index": 2,
    "exercise_title": "block-diagonal partial-singular solve — detect per-block",
    "slug": "block-diagonal-partial-singular-solve-per-block",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["block-diagonal", "per-block-singular", "linalg-solve", "partial-rescue"],
    "kcs": ["singular-detect-via-det", "singular-overwrite-identity"],
    "lo": (
        "Apply the singular-matrix-mask trick at block granularity: given a "
        "block-diagonal batch `(B, 2*N, 2*N)`, split into a `(B, 2, N, N)` "
        "block view, detect singular blocks via `la.det`, identity-replace "
        "them, and solve."
    ),
    "prompt_body": (
        "Implement `ex2_block_diag_solve(A, b, N, eps=1e-8)`.\n\n"
        "Each `A[i]` is a `(2*N, 2*N)` BLOCK-DIAGONAL matrix with two "
        "`(N, N)` blocks: `A[i] = block_diag(A[i, :N, :N], A[i, N:, N:])`. "
        "Some blocks may be singular but not necessarily both.\n\n"
        "**Steps.**\n"
        "1. Extract the two diagonal blocks into a tensor of shape "
        "`(B, 2, N, N)` (block index 0 is top-left, block index 1 is "
        "bottom-right).\n"
        "2. Extract the matching segments of `b` into `(B, 2, N)`.\n"
        "3. Compute `dets = la.det(blocks)` — shape `(B, 2)`.\n"
        "4. Build `is_valid = dets.abs() >= eps` — shape `(B, 2)`.\n"
        "5. Identity-replace invalid blocks: `blocks_safe[~is_valid] = eye(N)` "
        "(use fancy indexing).\n"
        "6. `x = la.solve(blocks_safe, b_blocks)` — shape `(B, 2, N)`.\n"
        "7. Return `(x, is_valid)` — the per-block solution and per-block "
        "validity mask. The caller is expected to zero or ignore the "
        "`~is_valid` slices.\n\n"
        "Inputs:\n"
        "- `A`: `(B, 2*N, 2*N)` block-diagonal float tensor.\n"
        "- `b`: `(B, 2*N)` float tensor.\n"
        "- `N`: block size (int).\n"
        "- `eps`: singularity threshold.\n\n"
        "Output: `(x, is_valid)` — `x` is `(B, 2, N)`, `is_valid` is "
        "`(B, 2)` bool."
    ),
    "stub": (
        "def ex2_block_diag_solve(A: Tensor, b: Tensor, N: int, eps: float = 1e-8):\n"
        '    """Per-block singular-mask solve on a 2-block-diagonal batch."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.linalg as la\n"
        "N = 2\n"
        "# Build B=3 block-diag matrices with varied singularity per block.\n"
        "A = t.zeros(3, 4, 4)\n"
        "# entry 0: both blocks invertible\n"
        "A[0, :2, :2] = t.tensor([[1., 0.], [0., 2.]])\n"
        "A[0, 2:, 2:] = t.tensor([[3., 0.], [0., 4.]])\n"
        "# entry 1: first block invertible, second singular (rank 1)\n"
        "A[1, :2, :2] = t.tensor([[5., 0.], [0., 6.]])\n"
        "A[1, 2:, 2:] = t.tensor([[1., 1.], [1., 1.]])\n"
        "# entry 2: first block singular (zero), second invertible\n"
        "A[2, :2, :2] = t.zeros(2, 2)\n"
        "A[2, 2:, 2:] = t.tensor([[7., 0.], [0., 8.]])\n"
        "b = t.ones(3, 4)\n"
        "\n"
        "x, is_valid = ex2_block_diag_solve(A, b, N)\n"
        "assert x.shape == (3, 2, 2), f'x shape {tuple(x.shape)} != (3,2,2)'\n"
        "assert is_valid.shape == (3, 2), f'is_valid shape {tuple(is_valid.shape)} != (3,2)'\n"
        "assert is_valid.dtype == t.bool, f'is_valid dtype {is_valid.dtype}'\n"
        "\n"
        "# Validity mask correctness.\n"
        "expected_valid = t.tensor([\n"
        "    [True,  True],   # both blocks invertible\n"
        "    [True,  False],  # second block singular\n"
        "    [False, True],   # first block singular\n"
        "])\n"
        "assert t.equal(is_valid, expected_valid), f'is_valid:\\n{is_valid}\\nvs\\n{expected_valid}'\n"
        "\n"
        "# Solution values on valid blocks must match a direct solve.\n"
        "# entry 0 block 0: diag(1,2)·x = [1,1] → x=[1, 0.5]\n"
        "assert t.allclose(x[0, 0], t.tensor([1.0, 0.5]), atol=1e-5)\n"
        "# entry 0 block 1: diag(3,4)·x = [1,1] → x=[1/3, 1/4]\n"
        "assert t.allclose(x[0, 1], t.tensor([1/3, 1/4]), atol=1e-5)\n"
        "# entry 1 block 0: diag(5,6)·x = [1,1] → x=[1/5, 1/6]\n"
        "assert t.allclose(x[1, 0], t.tensor([1/5, 1/6]), atol=1e-5)\n"
        "# entry 2 block 1: diag(7,8)·x = [1,1] → x=[1/7, 1/8]\n"
        "assert t.allclose(x[2, 1], t.tensor([1/7, 1/8]), atol=1e-5)\n"
        "\n"
        "# Did not crash on singular blocks.\n"
        "assert t.isfinite(x).all(), f'NaN/Inf leaked from singular block:\\n{x}'\n"
        "\n"
        "# Non-trivial off-diagonal-free block. Larger N=3 smoke test.\n"
        "N2 = 3\n"
        "rng = t.Generator().manual_seed(42)\n"
        "A2 = t.zeros(2, 6, 6)\n"
        "for i in range(2):\n"
        "    A2[i, :3, :3] = t.eye(3) * (i + 1)  # always invertible\n"
        "    A2[i, 3:, 3:] = t.eye(3) * (i + 1) * 2\n"
        "b2 = t.randn(2, 6, generator=rng)\n"
        "x2, v2 = ex2_block_diag_solve(A2, b2, N2)\n"
        "assert x2.shape == (2, 2, 3)\n"
        "assert t.all(v2), 'all blocks invertible — every is_valid must be True'"
    ),
    "solution_body": (
        "def ex2_block_diag_solve(A: Tensor, b: Tensor, N: int, eps: float = 1e-8):\n"
        "    import torch.linalg as la\n"
        "    # (B, 2*N, 2*N) -> stack the two diagonal blocks -> (B, 2, N, N)\n"
        "    blocks = t.stack([A[:, :N, :N], A[:, N:, N:]], dim=1)\n"
        "    b_blocks = t.stack([b[:, :N], b[:, N:]], dim=1)\n"
        "    # Per-block singular detection.\n"
        "    dets = la.det(blocks)             # (B, 2)\n"
        "    is_valid = dets.abs() >= eps      # (B, 2)\n"
        "    # Identity-replace the invalid blocks.\n"
        "    blocks_safe = blocks.clone()\n"
        "    eye = t.eye(N, dtype=blocks.dtype)\n"
        "    blocks_safe[~is_valid] = eye\n"
        "    # Solve everywhere; caller masks bogus slices via is_valid.\n"
        "    x = la.solve(blocks_safe, b_blocks)\n"
        "    return x, is_valid"
    ),
    "solution_notes": (
        "**Why per-block-level detection is the right granularity.** A "
        "block-diagonal matrix's full determinant is the product of its "
        "block determinants — so if EITHER block is singular the full "
        "`la.det` flags the whole slice as singular and ex1's "
        "whole-slice mask would throw away the good block too. Detecting "
        "per-block recovers more information without losing crash safety.\n\n"
        "**Why `blocks_safe = blocks.clone()`.** Fancy-indexing assignment "
        "`blocks_safe[~is_valid] = eye` writes into the source storage, "
        "so cloning before mutating preserves the caller's `A`.\n\n"
        "**Difference from ex1.** ex1 ran the det-singular-detect / "
        "identity-replace / solve trick over a flat batch of `(B, N, N)` "
        "slices — one validity flag per matrix. ex2 does the SAME trick but "
        "after first decomposing each matrix into its block-diagonal pieces, "
        "producing one validity flag per BLOCK. The structural pattern "
        "(detect → identity-replace → solve → return mask) is preserved; "
        "what changes is the unit of detection."
    ),
    "extra_imports": [],
}


# ============================================================== 6: conv-windowing-1d
# ex1: as_strided to build the view; ex2: Tensor.unfold API equivalent
SPEC_CONV1D = {
    "atom_id": "conv-windowing-1d",
    "subtopic": "CNN: 1-D conv windowing",
    "topic_folder": "prereqs_geometry_cnn",
    "atom_recap_md": (
        "## 1-D conv windowing — two ways to build the window view\n"
        "\n"
        "To express a 1-D conv as an einsum, you need a window view "
        "`(B, IC, OW, KW)` of the input. There are two natural APIs:\n"
        "\n"
        "1. **`as_strided`** — explicit shape + stride arithmetic.\n"
        "2. **`Tensor.unfold(dim, size, step)`** — high-level sliding-window "
        "API that returns the same view with sane defaults.\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 built the window view manually via "
        "`as_strided` (compute output width, compute strides). ex2 uses the "
        "**`Tensor.unfold`** API instead — same view, much less arithmetic, "
        "and verifies that the resulting einsum still matches `F.conv1d`. "
        "Knowing both lets you choose: `as_strided` for stride !=1 / "
        "non-trivial layouts, `unfold` for the common case."
    ),
    "exercise_index": 2,
    "exercise_title": "1-D conv windowing via Tensor.unfold (the high-level API)",
    "slug": "1d-conv-windowing-via-tensor-unfold",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["unfold", "sliding-window", "conv1d", "einsum"],
    "kcs": ["unfold-sliding-window-api", "windowing-output-width"],
    "lo": (
        "Apply `Tensor.unfold(dim, size, step)` to build the `(B, IC, OW, KW)` "
        "stride-1 window view of a 1-D input, then verify that contracting it "
        "against a kernel via einsum matches `F.conv1d`."
    ),
    "prompt_body": (
        "Implement `ex2_conv1d_unfold(x, KW)`.\n\n"
        "Given a `(B, IC, W)` float tensor `x` and a kernel width `KW`, "
        "return the stride-1 window view of shape `(B, IC, OW, KW)` where "
        "`OW = W - KW + 1`. Use `Tensor.unfold` rather than `as_strided`.\n\n"
        "**Rules.**\n"
        "1. Call `x.unfold(dimension=-1, size=KW, step=1)` — that's the "
        "high-level API.\n"
        "2. Output shape must be exactly `(B, IC, OW, KW)`.\n"
        "3. Verify (in the test) that contracting against a kernel via "
        "`einsum('b i o k, c i k -> b c o', windows, weight)` reproduces "
        "`F.conv1d(x, weight)`.\n\n"
        "Inputs:\n"
        "- `x`: `(B, IC, W)` float tensor with `W >= KW`.\n"
        "- `KW`: int kernel width.\n\n"
        "Output: `(B, IC, OW, KW)` window view (a view, not a copy)."
    ),
    "stub": (
        "def ex2_conv1d_unfold(x: Tensor, KW: int) -> Tensor:\n"
        '    """Build (B, IC, OW, KW) stride-1 window view via Tensor.unfold."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "# Small deterministic input — easy to read off the windows.\n"
        "x = t.arange(20, dtype=t.float32).reshape(1, 2, 10)\n"
        "# row 0: 0..9, row 1: 10..19\n"
        "KW = 3\n"
        "w = ex2_conv1d_unfold(x, KW)\n"
        "assert w.shape == (1, 2, 8, 3), f'shape {tuple(w.shape)} != (1,2,8,3)'\n"
        "# Window 0 of channel 0 is x[0,0,0:3] = [0,1,2].\n"
        "assert t.equal(w[0, 0, 0], t.tensor([0., 1., 2.]))\n"
        "# Window 4 of channel 0 is x[0,0,4:7] = [4,5,6].\n"
        "assert t.equal(w[0, 0, 4], t.tensor([4., 5., 6.]))\n"
        "# Window 7 of channel 1 is x[0,1,7:10] = [17,18,19].\n"
        "assert t.equal(w[0, 1, 7], t.tensor([17., 18., 19.]))\n"
        "# Storage is shared — unfold returns a view.\n"
        "assert w.storage().data_ptr() == x.storage().data_ptr(), 'unfold should return a view'\n"
        "\n"
        "# Verify equivalence to F.conv1d via einsum.\n"
        "B, IC, W, OC = 2, 3, 12, 4\n"
        "KW = 5\n"
        "rng = t.Generator().manual_seed(0)\n"
        "x_big = t.randn(B, IC, W, generator=rng)\n"
        "weight = t.randn(OC, IC, KW, generator=rng)\n"
        "windows = ex2_conv1d_unfold(x_big, KW)\n"
        "assert windows.shape == (B, IC, W - KW + 1, KW)\n"
        "# einsum contraction: 'b i o k, c i k -> b c o'\n"
        "via_einsum = t.einsum('b i o k, c i k -> b c o', windows, weight)\n"
        "via_conv = F.conv1d(x_big, weight)\n"
        "assert via_einsum.shape == via_conv.shape, f'{via_einsum.shape} vs {via_conv.shape}'\n"
        "assert t.allclose(via_einsum, via_conv, atol=1e-5), f'unfold+einsum != conv1d (max diff {(via_einsum-via_conv).abs().max().item()})'"
    ),
    "solution_body": (
        "def ex2_conv1d_unfold(x: Tensor, KW: int) -> Tensor:\n"
        "    return x.unfold(dimension=-1, size=KW, step=1)"
    ),
    "solution_notes": (
        "**Why `Tensor.unfold` exists.** `as_strided` is powerful but easy "
        "to get wrong — one bad stride and you read off the end of storage "
        "(silent corruption). `unfold` is the safer high-level cousin that "
        "computes strides for you. Same output, no arithmetic.\n\n"
        "**Watch the new axis position.** `x.unfold(dimension=-1, size=KW, "
        "step=1)` *appends* the `KW` axis at the end, so a `(B, IC, W)` "
        "input becomes `(B, IC, OW, KW)` — exactly the layout you need for "
        "the einsum `'b i o k, c i k -> b c o'`. No transpose needed.\n\n"
        "**When to still use `as_strided`.** `unfold` only supports a single "
        "dimension at a time and forces `step=1` for full-coverage views. "
        "For multi-axis windowing (2-D / 3-D convs) or for non-standard "
        "strides, you still drop to `as_strided`. ex1 showed the explicit "
        "form; ex2 shows the ergonomic shortcut."
    ),
    "extra_imports": [],
}


# ============================================================== 7: convT-kernel-axis-swap
# ex1: introspect weight.shape; ex2: CONSTRUCT a ConvT2d weight FROM a Conv2d weight
SPEC_CONVT_AXIS = {
    "atom_id": "convT-kernel-axis-swap",
    "subtopic": "CNN: ConvT kernel axis swap",
    "topic_folder": "prereqs_cnn_deep",
    "atom_recap_md": (
        "## ConvT kernel axis swap — quick refresher\n"
        "\n"
        "`nn.Conv2d.weight` has shape `(OC, IC, KH, KW)`.\n"
        "`nn.ConvTranspose2d.weight` has shape `(IC, OC, KH, KW)`.\n"
        "\n"
        "The first two axes are SWAPPED relative to one another — this is the "
        "root of every \"my weights don't fit\" bug when porting kernels "
        "between conv and convT.\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 *introspected* the two layouts "
        "(read each weight.shape, label what axis 0 means). ex2 *constructs* "
        "— given a Conv2d weight tensor, build the equivalent ConvT2d weight "
        "tensor by performing the axis-0/axis-1 swap, and verify the "
        "constructed tensor loads cleanly into a `nn.ConvTranspose2d` with "
        "the right `in_channels` / `out_channels`."
    ),
    "exercise_index": 2,
    "exercise_title": "construct a ConvT2d weight from a Conv2d weight via axis swap",
    "slug": "construct-convT2d-weight-from-conv2d-weight",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["axis-swap", "weight-construction", "convT", "transpose"],
    "kcs": ["convT-weight-axis-order", "conv-vs-convT-layout"],
    "lo": (
        "Apply the Conv2d→ConvT2d weight axis swap (`transpose(0, 1)` on the "
        "`(OC, IC, KH, KW)` weight) to construct a valid ConvT2d weight "
        "tensor and load it into a `nn.ConvTranspose2d` whose "
        "in_channels/out_channels match the original Conv2d's IC/OC."
    ),
    "prompt_body": (
        "Implement `ex2_conv_to_convT_weight(conv_weight)`.\n\n"
        "Given a Conv2d weight tensor `conv_weight` of shape "
        "`(OC, IC, KH, KW)`, return the equivalent ConvT2d weight tensor of "
        "shape `(IC, OC, KH, KW)` (axes 0 and 1 swapped, spatial axes "
        "untouched).\n\n"
        "**Rules.**\n"
        "1. Use `.transpose(0, 1).contiguous()` — `contiguous()` matters "
        "because `nn.ConvTranspose2d` will fail to accept a non-contiguous "
        "weight via `.weight.data.copy_(...)` if shapes don't match.\n"
        "2. Spatial axes (axis 2, 3) MUST be unchanged — this is not a "
        "kernel-flip operation; only the channel axes swap.\n"
        "3. The returned tensor must load into a `nn.ConvTranspose2d(IC, "
        "OC, kernel_size=(KH, KW))` whose `weight.shape` matches.\n\n"
        "Inputs:\n"
        "- `conv_weight`: `(OC, IC, KH, KW)` float tensor (e.g. from a "
        "`nn.Conv2d.weight`).\n\n"
        "Output: `(IC, OC, KH, KW)` contiguous float tensor."
    ),
    "stub": (
        "def ex2_conv_to_convT_weight(conv_weight: Tensor) -> Tensor:\n"
        '    """Swap (OC, IC) -> (IC, OC) to convert Conv2d weight to ConvT2d weight."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "OC, IC, KH, KW = 4, 2, 3, 5\n"
        "rng = t.Generator().manual_seed(0)\n"
        "conv_w = t.randn(OC, IC, KH, KW, generator=rng)\n"
        "ct_w = ex2_conv_to_convT_weight(conv_w)\n"
        "assert ct_w.shape == (IC, OC, KH, KW), f'shape {tuple(ct_w.shape)} != {(IC,OC,KH,KW)}'\n"
        "assert ct_w.dtype == conv_w.dtype, f'dtype changed: {ct_w.dtype}'\n"
        "assert ct_w.is_contiguous(), 'output must be contiguous'\n"
        "\n"
        "# Cell-wise verification — swapped channel pair, identical spatial slice.\n"
        "for oc in range(OC):\n"
        "    for ic in range(IC):\n"
        "        assert t.equal(ct_w[ic, oc], conv_w[oc, ic]), (\n"
        "            f'spatial slice (ic={ic}, oc={oc}) mismatch'\n"
        "        )\n"
        "\n"
        "# Round-trip — swap back must equal the original.\n"
        "round_trip = ex2_conv_to_convT_weight(ct_w)\n"
        "assert round_trip.shape == conv_w.shape\n"
        "assert t.equal(round_trip, conv_w), 'swap is its own inverse'\n"
        "\n"
        "# Load into a real nn.ConvTranspose2d to confirm the shape is accepted.\n"
        "ct = nn.ConvTranspose2d(in_channels=IC, out_channels=OC, kernel_size=(KH, KW), bias=False)\n"
        "assert ct.weight.shape == ct_w.shape, (\n"
        "    f'nn.ConvTranspose2d expects {tuple(ct.weight.shape)}, our constructed weight is {tuple(ct_w.shape)}'\n"
        ")\n"
        "with t.no_grad():\n"
        "    ct.weight.data.copy_(ct_w)\n"
        "assert t.equal(ct.weight, ct_w), 'copy_ should make the layer hold our exact weight'\n"
        "\n"
        "# Sanity — the constructed convT forward runs without shape errors.\n"
        "y = t.randn(1, IC, 4, 6, generator=rng)\n"
        "out = ct(y)\n"
        "# OH = (4 - 1) * 1 - 0 + (KH - 1) + 1 = 6;  OW = 6 + (KW-1) = 10\n"
        "assert out.shape == (1, OC, 6, 10), f'unexpected forward shape {tuple(out.shape)}'"
    ),
    "solution_body": (
        "def ex2_conv_to_convT_weight(conv_weight: Tensor) -> Tensor:\n"
        "    return conv_weight.transpose(0, 1).contiguous()"
    ),
    "solution_notes": (
        "**Why `transpose(0, 1)` and nothing else.** The ONLY layout "
        "difference between Conv2d's `(OC, IC, KH, KW)` and "
        "ConvTranspose2d's `(IC, OC, KH, KW)` is the first two axes — the "
        "spatial axes are identical. So a single channel-axis transpose "
        "is the whole conversion.\n\n"
        "**Why `.contiguous()`.** `.transpose` returns a non-contiguous view "
        "(same storage, swapped strides). Many downstream consumers — "
        "including `nn.Parameter` storage round-trips and CUDA kernels — "
        "expect contiguous tensors. Forcing contiguity here makes the "
        "constructed tensor a drop-in replacement.\n\n"
        "**Why this is NOT enough to *compute* a transposed conv.** The "
        "axis-swap alone gives you the right LAYOUT, but a true "
        "ConvT-via-Conv2d equivalence ALSO needs a spatial flip on KH/KW "
        "and a `K-1` zero pad on the input (see the "
        "`convT-as-flipped-padded-conv` atom). ex2 here is only about the "
        "layout transformation; flipping is a separate skill.\n\n"
        "**Difference from ex1.** ex1 *read* the two layouts — given "
        "modules, report their weight shapes and label axis 0. ex2 "
        "*produces* — given one layout, construct the other. Recognising "
        "the difference vs. effecting the difference are two different "
        "cognitive operations."
    ),
    "extra_imports": [],
}


# ============================================================== 8: convT-as-flipped-padded-conv
# ex1: rebuild ConvT from Conv (general); ex2: verify equivalence on a SMALL hand-checked numeric example
SPEC_CONVT_FLIP = {
    "atom_id": "convT-as-flipped-padded-conv",
    "subtopic": "CNN: ConvT as flipped padded conv",
    "topic_folder": "prereqs_cnn_deep",
    "atom_recap_md": (
        "## ConvT-as-flipped-padded-conv — quick refresher\n"
        "\n"
        "A stride-1 `F.conv_transpose2d(x, w)` is exactly equivalent to "
        "`F.conv2d(pad(x, K-1), flip_swap(w))`, where:\n"
        "\n"
        "- `pad(x, K-1)` zero-pads the spatial axes by `K-1` on every side.\n"
        "- `flip_swap(w) = w.flip([2, 3]).transpose(0, 1).contiguous()` "
        "spatially flips the kernel and swaps the channel axes.\n"
        "\n"
        "**This drill (ex2) vs ex1.** ex1 implemented the equivalence in "
        "full generality (large random `x` / `w`, tolerance-checked against "
        "`F.conv_transpose2d`). ex2 reduces the question to a single "
        "**hand-checked** numeric example — a `2×2` input and a `2×2` "
        "kernel whose expected `3×3` output can be computed by hand. The "
        "exercise forces you to confront the per-cell arithmetic and "
        "verify *byte-exact* equality (no `atol`)."
    ),
    "exercise_index": 2,
    "exercise_title": "hand-checked 2x2 input / 2x2 kernel ConvT equivalence",
    "slug": "hand-checked-2x2-convT-equivalence",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["hand-check", "byte-exact", "convT", "flip-and-pad"],
    "kcs": ["convT-padded-conv-equivalence", "convT-kernel-flip-rule"],
    "lo": (
        "Analyze the stride-1 ConvT-as-flipped-padded-conv equivalence on a "
        "hand-computable `2×2` input / `2×2` identity-diagonal kernel, "
        "demonstrating byte-exact agreement between the rebuilt conv2d path "
        "and the canonical ConvT2d output."
    ),
    "prompt_body": (
        "Implement `ex2_hand_check_convT_equivalence()`.\n\n"
        "You will build a tiny example and verify the equivalence is exact "
        "(no tolerance). The setup is fixed:\n\n"
        "- Input `x` of shape `(1, 1, 2, 2)` with values `[[1., 2.], [3., 4.]]`.\n"
        "- Weight `w` of shape `(1, 1, 2, 2)` (ConvT2d layout `(IC, OC, KH, KW)`) "
        "with values `[[1., 0.], [0., 1.]]` — a diagonal 2x2 kernel.\n\n"
        "The expected stride-1 `F.conv_transpose2d(x, w)` output (computable by "
        "hand from the four overlapping kernel placements) is the `(1, 1, 3, 3)` "
        "tensor:\n\n"
        "```\n"
        "[[1., 2., 0.],\n"
        " [3., 5., 2.],\n"
        " [0., 3., 4.]]\n"
        "```\n\n"
        "**Steps.**\n"
        "1. Build `x` and `w` (the fixed values above).\n"
        "2. Compute `convT_out = F.conv_transpose2d(x, w)`.\n"
        "3. Compute the equivalent via conv2d: pad `x` by `K-1=1` on every "
        "spatial side, flip `w` spatially via `w.flip([2, 3])`, swap channel "
        "axes via `.transpose(0, 1).contiguous()`, then run "
        "`F.conv2d(x_pad, w_flipped_swapped)`.\n"
        "4. Verify both equal the hand-computed expected matrix above EXACTLY "
        "(use `t.equal`, not `t.allclose`).\n"
        "5. Return a dict `{'convT': convT_out, 'conv_equiv': conv_out, "
        "'expected': expected}`.\n\n"
        "The function takes no arguments — everything is fixed for the "
        "hand-check.\n\n"
        "The visualization renders the three matrices side by side as "
        "heatmaps so you can eyeball the equivalence."
    ),
    "stub": (
        "def ex2_hand_check_convT_equivalence() -> dict:\n"
        '    """Hand-checked 2x2/2x2 ConvT equivalence — returns the three matrices."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "result = ex2_hand_check_convT_equivalence()\n"
        "assert isinstance(result, dict), f'expected dict, got {type(result).__name__}'\n"
        "assert set(result.keys()) >= {'convT', 'conv_equiv', 'expected'}, f'keys {set(result.keys())}'\n"
        "\n"
        "convT_out = result['convT']\n"
        "conv_equiv = result['conv_equiv']\n"
        "expected = result['expected']\n"
        "\n"
        "# All three are (1, 1, 3, 3) float32.\n"
        "for name, arr in [('convT', convT_out), ('conv_equiv', conv_equiv), ('expected', expected)]:\n"
        "    assert arr.shape == (1, 1, 3, 3), f'{name} shape {tuple(arr.shape)} != (1,1,3,3)'\n"
        "    assert arr.dtype == t.float32, f'{name} dtype {arr.dtype}'\n"
        "\n"
        "# Hand-computed reference must be the canonical matrix.\n"
        "ref = t.tensor([[[[1., 2., 0.],\n"
        "                   [3., 5., 2.],\n"
        "                   [0., 3., 4.]]]])\n"
        "assert t.equal(expected, ref), f'expected matrix wrong:\\n{expected}'\n"
        "\n"
        "# Byte-exact agreement on this tiny example — no atol needed.\n"
        "assert t.equal(convT_out, expected), f'F.conv_transpose2d output not byte-exact:\\n{convT_out}'\n"
        "assert t.equal(conv_equiv, expected), f'conv-equivalent output not byte-exact:\\n{conv_equiv}'\n"
        "assert t.equal(convT_out, conv_equiv), 'convT and conv_equiv must agree byte-exactly'\n"
        "\n"
        "# --- Side-by-side heatmap ---\n"
        "fig, axes = plt.subplots(1, 3, figsize=(9, 3))\n"
        "titles = ['expected (hand-computed)', 'F.conv_transpose2d', 'F.conv2d(pad·flip·swap)']\n"
        "mats = [expected, convT_out, conv_equiv]\n"
        "for ax, title, m in zip(axes, titles, mats):\n"
        "    im = ax.imshow(m.squeeze().numpy(), cmap='viridis')\n"
        "    ax.set_title(title)\n"
        "    for i in range(3):\n"
        "        for j in range(3):\n"
        "            ax.text(j, i, f'{m[0,0,i,j].item():.0f}', ha='center', va='center', color='white')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_hand_check_convT_equivalence() -> dict:\n"
        "    import torch.nn.functional as F\n"
        "    x = t.tensor([[[[1., 2.], [3., 4.]]]])\n"
        "    w = t.tensor([[[[1., 0.], [0., 1.]]]])   # ConvT layout (IC=1, OC=1, KH=2, KW=2)\n"
        "    convT_out = F.conv_transpose2d(x, w)\n"
        "    # Equivalent path: pad K-1=1, flip kernel spatially, swap channel axes.\n"
        "    x_pad = F.pad(x, (1, 1, 1, 1))\n"
        "    w_flipped = w.flip([2, 3]).transpose(0, 1).contiguous()\n"
        "    conv_equiv = F.conv2d(x_pad, w_flipped)\n"
        "    expected = t.tensor([[[[1., 2., 0.],\n"
        "                            [3., 5., 2.],\n"
        "                            [0., 3., 4.]]]])\n"
        "    return {'convT': convT_out, 'conv_equiv': conv_equiv, 'expected': expected}"
    ),
    "solution_notes": (
        "**Why this example is fully hand-checkable.** With the diagonal "
        "kernel `[[1, 0], [0, 1]]`, the ConvT places one copy of the "
        "`(1, 2, 3, 4)` input at each of its four valid overlap positions "
        "in the `3×3` output grid — the centre cell receives two "
        "contributions (`1 + 4 = 5`), the off-diagonal cells one. No "
        "floating-point error to absorb, so `t.equal` (byte-exact) replaces "
        "`t.allclose`.\n\n"
        "**Why `t.equal` matters.** Tolerance-based assertions can hide "
        "small layout / sign bugs that only show up at larger scale. The "
        "hand-check exposes the equivalence at machine-integer precision — "
        "if the byte-exact equality breaks on this tiny case, the larger "
        "tolerance-checked equivalence in ex1 is almost certainly broken "
        "too, just hidden by the tolerance.\n\n"
        "**Difference from ex1.** ex1 reproduced the equivalence on random "
        "`(B, IC, H, W)` inputs with random kernels and verified via "
        "`t.allclose` — the *generality* check. ex2 reduces to one fixed "
        "input that can be computed by hand and verifies via `t.equal` — "
        "the *precision* check. Both are needed: random for coverage, "
        "hand-checked for catching layout bugs that randomness might "
        "average out."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


SPECS = [
    SPEC_UNBIND,
    SPEC_FLATTEN,
    SPEC_REDUCE_MIN,
    SPEC_INF_MASK,
    SPEC_SINGULAR,
    SPEC_CONV1D,
    SPEC_CONVT_AXIS,
    SPEC_CONVT_FLIP,
]


for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
