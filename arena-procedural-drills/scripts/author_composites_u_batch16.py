"""Composite drills cx25..cx30 — batch-16 (U-cell, part0).

Six composite procedural drills exercising 2-atom pairs from the einops /
broadcasting / tensor-mechanics machinery (ARENA part 0 — prereqs).

cx25  einops-rearrange + tensor-wraps-ndarray
cx26  einops-rearrange-flatten + einops-reduce
cx27  broadcasting-rules + einops-reduce
cx28  broadcasting-rules + einops-repeat
cx29  broadcasting-rules + einops-repeat-broadcast
cx30  einops-reduce + einops-repeat-broadcast
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# ===========================================================================
# cx25 — wrap an ndarray then rearrange via einops
# ===========================================================================
spec_25 = {
    "atom_ids": ["einops-rearrange", "tensor-wraps-ndarray"],
    "subtopics": _subs(["einops-rearrange", "tensor-wraps-ndarray"]),
    "primary_atom": "einops-rearrange",
    "part": "part0",
    "exercise_index": 25,
    "exercise_title": "wrap ndarray with from_numpy, then rearrange axes",
    "slug": "from-numpy-then-rearrange",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Data pipelines almost always start in NumPy land — images decoded as `np.ndarray`, sensor logs "
        "loaded with `np.load`, etc. To feed them through a torch model we wrap with `t.from_numpy(arr)` "
        "(zero-copy view) and then reshape with `einops.rearrange` into the layout the model expects.\n\n"
        "The composition exercises BOTH atoms: the wrap is load-bearing (we assert storage aliasing — "
        "no defensive copy), and the rearrange pattern is load-bearing (we assert the exact axis order). "
        "Together they form the canonical NumPy → torch → CNN-shape bridge."
    ),
    "prompt_body": (
        "Implement `cx25_ndarray_to_nchw(arr)` that takes an `np.ndarray` of shape `(H, W, C)` "
        "(NumPy/HWC convention — what cv2 / PIL hand you) and returns a torch tensor of shape "
        "`(1, C, H, W)` (torch/NCHW convention).\n\n"
        "1. **Wrap** — use `t.from_numpy(arr)` so the tensor shares storage with `arr` (no defensive copy).\n"
        "2. **Rearrange** — use `einops.rearrange(..., 'h w c -> 1 c h w')` to insert the batch axis "
        "AND reorder HWC → CHW in one step.\n\n"
        "The test asserts both the wrap (data_ptr aliasing) and the rearrange (exact byte-order match "
        "with `t.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)`)."
    ),
    "stub_body": (
        "def cx25_ndarray_to_nchw(arr):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: distinguishable values — verify the axis order is correct.\n"
        "arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)  # (H=2, W=3, C=4)\n"
        "out = cx25_ndarray_to_nchw(arr)\n"
        "assert isinstance(out, t.Tensor), f'expected torch.Tensor, got {type(out)}'\n"
        "assert tuple(out.shape) == (1, 4, 2, 3), f'expected (1,4,2,3), got {tuple(out.shape)}'\n"
        "expected = t.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)\n"
        "assert t.equal(out, expected), 'axis order wrong — did you use h w c -> 1 c h w?'\n"
        "\n"
        "# Case B: aliasing — from_numpy must share storage; mutating arr must change the tensor.\n"
        "arr2 = np.zeros((3, 4, 2), dtype=np.float32)\n"
        "out2 = cx25_ndarray_to_nchw(arr2)\n"
        "arr2[0, 0, 0] = 99.0\n"
        "# After rearrange the tensor is a view; the (0, 0, 0, 0) slot in NCHW maps to (0, 0, 0) HWC.\n"
        "assert out2[0, 0, 0, 0].item() == 99.0, (\n"
        "    'tensor must share storage with arr (from_numpy zero-copy). '\n"
        "    'Did you call .clone() or t.tensor(arr)?'\n"
        ")\n"
        "\n"
        "# Case C: realistic CNN-ish image shape.\n"
        "img = np.random.RandomState(0).rand(32, 32, 3).astype(np.float32)\n"
        "out3 = cx25_ndarray_to_nchw(img)\n"
        "assert tuple(out3.shape) == (1, 3, 32, 32)\n"
        "assert out3.dtype == t.float32\n"
        "# Pluggable into a Conv2d.\n"
        "conv = t.nn.Conv2d(3, 8, kernel_size=3, padding=1)\n"
        "feat = conv(out3)\n"
        "assert feat.shape == (1, 8, 32, 32)"
    ),
    "solution_body": (
        "def cx25_ndarray_to_nchw(arr):\n"
        "    # Atom A: wrap the ndarray zero-copy.\n"
        "    wrapped = t.from_numpy(arr)\n"
        "    # Atom B: rearrange HWC -> NCHW in one named pattern (insert batch + reorder).\n"
        "    return rearrange(wrapped, 'h w c -> 1 c h w')"
    ),
    "solution_notes": (
        "`from_numpy` is the zero-copy half — the resulting tensor's storage IS the ndarray's buffer. "
        "`rearrange('h w c -> 1 c h w')` then constructs a non-contiguous view that permutes axes and "
        "inserts the batch dim — still sharing storage with `arr`. If you replace `from_numpy` with "
        "`t.tensor(arr)` the storage check fails; if you flip the rearrange order to `'h w c -> 1 h w c'` "
        "the byte-order check fails."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["einops-rearrange", "tensor-wraps-ndarray"],
    "lo": (
        "Bridge NumPy-land HWC arrays into torch-land NCHW tensors in one composition: wrap with "
        "from_numpy for zero-copy semantics, then rearrange axes with the einops named pattern."
    ),
}


# ===========================================================================
# cx26 — flatten a CNN-feature group then reduce within the flattened axis
# ===========================================================================
spec_26 = {
    "atom_ids": ["einops-rearrange-flatten", "einops-reduce"],
    "subtopics": _subs(["einops-rearrange-flatten", "einops-reduce"]),
    "primary_atom": "einops-rearrange-flatten",
    "part": "part0",
    "exercise_index": 26,
    "exercise_title": "flatten patch group then reduce within the flattened axis",
    "slug": "rearrange-flatten-then-reduce",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Vision pipelines often `rearrange` a CNN feature map to GROUP spatial sub-blocks "
        "(e.g. patches, pooling windows) into a new flattened axis — then `reduce` along that flattened "
        "axis to get a per-block summary statistic. The two atoms naturally chain: rearrange composes the "
        "axes you want to summarize, reduce collapses them.\n\n"
        "Worked example: 2x2 average pooling. Rearrange `'b c (h h2) (w w2) -> b c h w (h2 w2)'` to group "
        "the 2x2 spatial blocks into a final flattened axis of size 4, then `reduce(..., 'b c h w n -> "
        "b c h w', 'mean')` to average within each block. The composition makes pooling expressible as a "
        "shape transform + a reduction — no `F.avg_pool2d` needed."
    ),
    "prompt_body": (
        "Implement `cx26_block_pool_via_flatten_reduce(x, block, reduction)` that performs `block` x "
        "`block` non-overlapping pooling over the spatial axes of an `(B, C, H, W)` tensor.\n\n"
        "1. **Rearrange** the input with `'b c (h h2) (w w2) -> b c h w (h2 w2)'` (substitute `h2=w2="
        "block`). This groups each `block x block` spatial block into a final flattened axis of length "
        "`block * block`.\n"
        "2. **Reduce** along that flattened axis with `reduce(..., 'b c h w n -> b c h w', reduction)`. "
        "`reduction` is one of `'mean'`, `'max'`, `'sum'`.\n\n"
        "Return the pooled `(B, C, H // block, W // block)` tensor.\n\n"
        "Cross-check against `F.avg_pool2d` / `F.max_pool2d` so the composition is verified, not just typed."
    ),
    "stub_body": (
        "def cx26_block_pool_via_flatten_reduce(x, block, reduction):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "\n"
        "# Case A: 2x2 average pool — cross-check against F.avg_pool2d.\n"
        "x = t.randn(2, 3, 8, 8)\n"
        "out = cx26_block_pool_via_flatten_reduce(x, block=2, reduction='mean')\n"
        "assert tuple(out.shape) == (2, 3, 4, 4), f'got {tuple(out.shape)}'\n"
        "ref = F.avg_pool2d(x, kernel_size=2)\n"
        "assert t.allclose(out, ref, atol=1e-6), f'avg-pool mismatch (max diff {(out-ref).abs().max()})'\n"
        "\n"
        "# Case B: 2x2 max pool.\n"
        "out_max = cx26_block_pool_via_flatten_reduce(x, block=2, reduction='max')\n"
        "ref_max = F.max_pool2d(x, kernel_size=2)\n"
        "assert t.allclose(out_max, ref_max), 'max-pool mismatch'\n"
        "\n"
        "# Case C: 4x4 sum reduction over a single batch/channel — hand-check exact sums.\n"
        "y = t.arange(16.0).reshape(1, 1, 4, 4)\n"
        "out_sum = cx26_block_pool_via_flatten_reduce(y, block=4, reduction='sum')\n"
        "assert tuple(out_sum.shape) == (1, 1, 1, 1)\n"
        "assert out_sum.item() == sum(range(16)), f'got {out_sum.item()}'\n"
        "\n"
        "# Case D: non-square spatial — block=2 on (B,C,6,10).\n"
        "z = t.randn(1, 2, 6, 10)\n"
        "out_z = cx26_block_pool_via_flatten_reduce(z, block=2, reduction='mean')\n"
        "assert tuple(out_z.shape) == (1, 2, 3, 5)\n"
        "assert t.allclose(out_z, F.avg_pool2d(z, kernel_size=2), atol=1e-6)"
    ),
    "solution_body": (
        "def cx26_block_pool_via_flatten_reduce(x, block, reduction):\n"
        "    # Atom A (rearrange-flatten): group each block x block window into a final flat axis.\n"
        "    grouped = rearrange(\n"
        "        x, 'b c (h h2) (w w2) -> b c h w (h2 w2)', h2=block, w2=block\n"
        "    )\n"
        "    # Atom B (reduce): collapse the flattened block axis with the requested reduction.\n"
        "    return reduce(grouped, 'b c h w n -> b c h w', reduction)"
    ),
    "solution_notes": (
        "Rearrange-flatten followed by reduce is the workhorse pattern for spatial pooling in einops "
        "code. Note how the named axis `n` in the second pattern is just a placeholder for the grouped "
        "`(h2 w2)` axis — einops doesn't care what you call it, only that exactly one axis is collapsed. "
        "Swapping `reduction` between `'mean'`, `'max'`, `'sum'` recovers the three standard pool ops "
        "without a separate API per kind."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["einops-rearrange-flatten", "einops-reduce"],
    "lo": (
        "Compose rearrange-flatten with reduce to express N x N spatial pooling as a group-then-collapse "
        "transform, recovering F.avg_pool2d and F.max_pool2d behaviour."
    ),
}


# ===========================================================================
# cx27 — reduce then broadcast for per-row normalization
# ===========================================================================
spec_27 = {
    "atom_ids": ["broadcasting-rules", "einops-reduce"],
    "subtopics": _subs(["broadcasting-rules", "einops-reduce"]),
    "primary_atom": "broadcasting-rules",
    "part": "part0",
    "exercise_index": 27,
    "exercise_title": "reduce then broadcast — per-row normalization",
    "slug": "reduce-then-broadcast-normalize",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Normalization is the canonical reduce-then-broadcast pattern: you `reduce` to compute "
        "per-sample statistics (mean, std), then rely on **broadcasting rules** to subtract / divide "
        "them back across the original tensor without ever expanding the stats by hand.\n\n"
        "The trick: use `reduce(..., 'b d -> b 1', 'mean')` (keepdim semantics via the explicit `1`). "
        "The trailing `1` axis is exactly what broadcasting needs to align against the original `(b, d)` "
        "tensor — a missing axis would force you to `unsqueeze` manually; a fully-collapsed `(b,)` would "
        "broadcast to the WRONG axis. `(b, 1)` is the keepdim sweet spot that makes broadcasting "
        "automatic."
    ),
    "prompt_body": (
        "Implement `cx27_row_normalize(x, eps)` that returns the per-row z-score normalized version of "
        "an `(B, D)` matrix:\n\n"
        "1. **Reduce** to per-row mean and std with the `'b d -> b 1'` pattern (KEEP the trailing 1 — "
        "that's what makes broadcasting auto-align).\n"
        "2. **Broadcast** the subtraction and division back across the original tensor: `(x - mu) / "
        "(sigma + eps)`. Do not call `.expand` / `.repeat` — broadcasting rules handle it for free.\n\n"
        "Return shape `(B, D)`. The result should have per-row mean ~0 and std ~1.\n\n"
        "Cross-check against `(x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True, "
        "unbiased=False) + eps)`."
    ),
    "stub_body": (
        "def cx27_row_normalize(x, eps=1e-5):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: standard (B, D) matrix.\n"
        "x = t.randn(4, 16) * 5 + 3  # off-center, off-scale\n"
        "out = cx27_row_normalize(x, eps=1e-5)\n"
        "assert tuple(out.shape) == tuple(x.shape), f'shape changed: {tuple(out.shape)}'\n"
        "# Per-row mean ~ 0.\n"
        "row_means = out.mean(dim=1)\n"
        "assert t.allclose(row_means, t.zeros(4), atol=1e-4), f'row means not ~0: {row_means}'\n"
        "# Per-row std ~ 1.\n"
        "row_stds = out.std(dim=1, unbiased=False)\n"
        "assert t.allclose(row_stds, t.ones(4), atol=1e-3), f'row stds not ~1: {row_stds}'\n"
        "\n"
        "# Cross-check against the torch keepdim formulation.\n"
        "ref = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True, unbiased=False) + 1e-5)\n"
        "assert t.allclose(out, ref, atol=1e-5), 'normalized values diverge from keepdim reference'\n"
        "\n"
        "# Case B: wider D.\n"
        "x2 = t.randn(8, 128)\n"
        "out2 = cx27_row_normalize(x2, eps=1e-5)\n"
        "assert tuple(out2.shape) == (8, 128)\n"
        "assert t.allclose(out2.mean(dim=1), t.zeros(8), atol=1e-4)\n"
        "\n"
        "# Case C: eps actually prevents zero-division on a constant row.\n"
        "x3 = t.ones(2, 5)  # std = 0\n"
        "out3 = cx27_row_normalize(x3, eps=1e-3)\n"
        "assert t.isfinite(out3).all(), 'eps should prevent inf/nan on constant rows'"
    ),
    "solution_body": (
        "def cx27_row_normalize(x, eps=1e-5):\n"
        "    # Atom A (einops-reduce): keep the trailing 1 axis so broadcasting auto-aligns.\n"
        "    mu = reduce(x, 'b d -> b 1', 'mean')\n"
        "    # variance via reduce of (x - mu) ** 2 — still keeping the trailing 1.\n"
        "    var = reduce((x - mu) ** 2, 'b d -> b 1', 'mean')\n"
        "    sigma = (var + 0.0).sqrt()\n"
        "    # Atom B (broadcasting-rules): (b, 1) aligns against (b, d) without manual expand.\n"
        "    return (x - mu) / (sigma + eps)"
    ),
    "solution_notes": (
        "The trailing `1` in the reduce pattern is doing all the broadcasting work. Without it "
        "(`'b d -> b'`), the result would be shape `(B,)` and broadcasting would align it against the "
        "LAST axis of `x` (D), giving you per-COLUMN normalization instead. With it, broadcasting "
        "aligns `(B, 1)` against `(B, D)` and stretches the 1-axis automatically — no explicit expand "
        "or unsqueeze required. This is the einops-as-keepdim idiom."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["broadcasting-rules", "einops-reduce"],
    "lo": (
        "Compose einops reduce (with keepdim-1 axis) and NumPy/torch broadcasting rules to implement "
        "per-row z-score normalization without manual expand/unsqueeze."
    ),
}


# ===========================================================================
# cx28 — repeat a row across batch axis for broadcasting
# ===========================================================================
spec_28 = {
    "atom_ids": ["broadcasting-rules", "einops-repeat"],
    "subtopics": _subs(["broadcasting-rules", "einops-repeat"]),
    "primary_atom": "broadcasting-rules",
    "part": "part0",
    "exercise_index": 28,
    "exercise_title": "repeat a row across batch axis then broadcast-add",
    "slug": "repeat-row-then-broadcast-add",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Per-sample bias addition is the canonical broadcasting use case: you have a `(D,)` bias vector "
        "and an `(B, D)` activation matrix; you want each row of activations shifted by the bias. "
        "Naïve broadcasting `acts + bias` works (it auto-prepends a 1-axis), but it's implicit.\n\n"
        "`einops.repeat` makes the broadcast EXPLICIT — `repeat(bias, 'd -> b d', b=B)` materializes "
        "the broadcast intent in the code. The trick: einops compiles this to `expand` (stride-0 view, "
        "no copy), so the result is identical in storage and value to letting broadcasting handle it — "
        "but the named-axis pattern documents WHICH axis the bias broadcasts across."
    ),
    "prompt_body": (
        "Implement `cx28_add_bias_via_repeat(acts, bias)` that adds a per-feature bias to a batch of "
        "activations.\n\n"
        "- `acts` has shape `(B, D)` — the batch of activations.\n"
        "- `bias` has shape `(D,)` — the per-feature bias vector.\n\n"
        "1. **Repeat** the bias across the batch axis: `repeat(bias, 'd -> b d', b=B)` so it has shape "
        "`(B, D)`.\n"
        "2. **Broadcast-add** the repeated bias to the activations. (At this point the shapes match so "
        "it's just `acts + bias_b`, but the test verifies the named-repeat path matches the implicit-"
        "broadcasting path bit-for-bit.)\n\n"
        "Return shape `(B, D)`. Cross-check against the plain `acts + bias` (implicit broadcast)."
    ),
    "stub_body": (
        "def cx28_add_bias_via_repeat(acts, bias):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: cross-check against implicit broadcasting.\n"
        "acts = t.randn(4, 8)\n"
        "bias = t.randn(8)\n"
        "out = cx28_add_bias_via_repeat(acts, bias)\n"
        "assert tuple(out.shape) == (4, 8), f'shape: {tuple(out.shape)}'\n"
        "assert t.allclose(out, acts + bias), 'repeat+add must equal implicit-broadcast result'\n"
        "\n"
        "# Case B: hand-check on distinguishable values.\n"
        "acts2 = t.zeros(3, 4)\n"
        "bias2 = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "out2 = cx28_add_bias_via_repeat(acts2, bias2)\n"
        "# Every row should equal the bias.\n"
        "for r in range(3):\n"
        "    assert t.equal(out2[r], bias2), f'row {r} mismatch: {out2[r]}'\n"
        "\n"
        "# Case C: realistic linear-layer scale.\n"
        "acts3 = t.randn(32, 768)\n"
        "bias3 = t.randn(768)\n"
        "out3 = cx28_add_bias_via_repeat(acts3, bias3)\n"
        "assert tuple(out3.shape) == (32, 768)\n"
        "assert t.allclose(out3, acts3 + bias3)\n"
        "\n"
        "# Case D: batch of 1 — degenerate but must still work.\n"
        "acts4 = t.randn(1, 5)\n"
        "bias4 = t.randn(5)\n"
        "out4 = cx28_add_bias_via_repeat(acts4, bias4)\n"
        "assert tuple(out4.shape) == (1, 5)\n"
        "assert t.allclose(out4, acts4 + bias4)"
    ),
    "solution_body": (
        "def cx28_add_bias_via_repeat(acts, bias):\n"
        "    B = acts.shape[0]\n"
        "    # Atom A (einops-repeat): repeat the (D,) bias into (B, D) — stride-0 view, no copy.\n"
        "    bias_b = repeat(bias, 'd -> b d', b=B)\n"
        "    # Atom B (broadcasting-rules): shapes now match, but the SAME result drops out from\n"
        "    # plain `acts + bias` because broadcasting auto-prepends a 1-axis. The named repeat\n"
        "    # makes the intent explicit, while broadcasting + repeat-as-expand keep it free.\n"
        "    return acts + bias_b"
    ),
    "solution_notes": (
        "`einops.repeat(bias, 'd -> b d', b=B)` is identical in storage to `bias.expand(B, -1)` — both "
        "produce a stride-0 view of the original `bias` buffer. So the cost of the explicit-repeat path "
        "is exactly zero compared to the implicit-broadcast path, but the named pattern documents WHICH "
        "axis broadcasts across. This is the einops-as-self-documenting-broadcast idiom — useful in "
        "any pipeline where multiple shapes are converging."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["broadcasting-rules", "einops-repeat"],
    "lo": (
        "Compose einops.repeat (explicit broadcast) with broadcasting rules to add a per-feature bias "
        "across a batch of activations, matching the implicit-broadcasting result bit-for-bit."
    ),
}


# ===========================================================================
# cx29 — repeat-broadcast for pairwise computation
# ===========================================================================
spec_29 = {
    "atom_ids": ["broadcasting-rules", "einops-repeat-broadcast"],
    "subtopics": _subs(["broadcasting-rules", "einops-repeat-broadcast"]),
    "primary_atom": "broadcasting-rules",
    "part": "part0",
    "exercise_index": 29,
    "exercise_title": "pairwise L2 distances via repeat-broadcast (no copy)",
    "slug": "pairwise-distances-repeat-broadcast",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Pairwise computation — every-A against every-B — is the canonical repeat-as-broadcast pattern. "
        "Given `A: (N, D)` and `B: (M, D)`, you want `dist: (N, M)`. The free way is to insert a new "
        "axis on each side so broadcasting can do the pairing:\n\n"
        "  `A_b = repeat(A, 'n d -> n m d', m=M)`  (insert m-axis, stride 0)\n"
        "  `B_b = repeat(B, 'm d -> n m d', n=N)`  (insert n-axis, stride 0)\n\n"
        "Both are zero-copy views (the einops-repeat-broadcast atom). Then broadcasting rules let you "
        "subtract, square, and reduce them as if they were materialized `(N, M, D)` tensors — but the "
        "underlying storage is still the original `A` and `B` buffers. Memory cost is O(N + M), not "
        "O(N * M)."
    ),
    "prompt_body": (
        "Implement `cx29_pairwise_l2(A, B)` that computes the `(N, M)` matrix of pairwise L2 distances "
        "between rows of `A: (N, D)` and rows of `B: (M, D)`.\n\n"
        "1. **Repeat-broadcast** to insert the pairing axes WITHOUT copying:\n"
        "   - `A_b = repeat(A, 'n d -> n m d', m=M)` (stride-0 view)\n"
        "   - `B_b = repeat(B, 'm d -> n m d', n=N)` (stride-0 view)\n"
        "2. **Broadcast-subtract**, square, sum over `d`, and sqrt. The result is the `(N, M)` "
        "distance matrix.\n\n"
        "Cross-check against `t.cdist(A, B)`. The test also asserts that `A_b.data_ptr() == "
        "A.data_ptr()` and same for `B` — i.e. the repeat-broadcast was a true zero-copy view, not a "
        "materialized tensor."
    ),
    "stub_body": (
        "def cx29_pairwise_l2(A, B):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: cross-check against t.cdist on random inputs.\n"
        "A = t.randn(5, 3)\n"
        "B = t.randn(4, 3)\n"
        "out = cx29_pairwise_l2(A, B)\n"
        "assert tuple(out.shape) == (5, 4), f'expected (5,4), got {tuple(out.shape)}'\n"
        "ref = t.cdist(A, B)\n"
        "assert t.allclose(out, ref, atol=1e-5), f'max diff {(out-ref).abs().max()}'\n"
        "\n"
        "# Case B: hand-check on simple integer-valued points.\n"
        "A2 = t.tensor([[0.0, 0.0], [3.0, 4.0]])  # origin, (3,4)\n"
        "B2 = t.tensor([[0.0, 0.0], [1.0, 0.0]])  # origin, (1,0)\n"
        "out2 = cx29_pairwise_l2(A2, B2)\n"
        "# Distances:\n"
        "#   (0,0)-(0,0) = 0;  (0,0)-(1,0) = 1\n"
        "#   (3,4)-(0,0) = 5;  (3,4)-(1,0) = sqrt(4+16) = sqrt(20)\n"
        "expected = t.tensor([[0.0, 1.0], [5.0, 20.0 ** 0.5]])\n"
        "assert t.allclose(out2, expected, atol=1e-5), f'got {out2}, expected {expected}'\n"
        "\n"
        "# Case C: square distance matrix (A == B) → zero diagonal.\n"
        "A3 = t.randn(6, 8)\n"
        "out3 = cx29_pairwise_l2(A3, A3)\n"
        "assert tuple(out3.shape) == (6, 6)\n"
        "assert t.allclose(out3.diagonal(), t.zeros(6), atol=1e-5), f'diagonal: {out3.diagonal()}'\n"
        "# Symmetric.\n"
        "assert t.allclose(out3, out3.T, atol=1e-5), 'pairwise distance must be symmetric'\n"
        "\n"
        "# Case D: realistic scale.\n"
        "A4 = t.randn(64, 16)\n"
        "B4 = t.randn(128, 16)\n"
        "out4 = cx29_pairwise_l2(A4, B4)\n"
        "assert tuple(out4.shape) == (64, 128)\n"
        "assert t.allclose(out4, t.cdist(A4, B4), atol=1e-4)"
    ),
    "solution_body": (
        "def cx29_pairwise_l2(A, B):\n"
        "    N, _ = A.shape\n"
        "    M, _ = B.shape\n"
        "    # Atom A (einops-repeat-broadcast): insert pairing axes as stride-0 views.\n"
        "    A_b = repeat(A, 'n d -> n m d', m=M)\n"
        "    B_b = repeat(B, 'm d -> n m d', n=N)\n"
        "    # Atom B (broadcasting-rules): aligned shapes (n, m, d) — elementwise sub + reduce.\n"
        "    diff = A_b - B_b\n"
        "    sq = diff ** 2\n"
        "    return reduce(sq, 'n m d -> n m', 'sum').sqrt()"
    ),
    "solution_notes": (
        "The whole point of repeat-broadcast over `.repeat()` (torch method) is the storage: the "
        "intermediate `(N, M, D)` tensor is *never materialized*. Both `A_b` and `B_b` are stride-0 "
        "views sharing storage with `A` and `B` — memory cost is O(N + M), not O(N*M*D). The subtract "
        "step does allocate (you can't avoid that for the diff), but the *input* tensors stay free. "
        "At ARENA scale (millions of ray/triangle pairs) this is the difference between OOM and OK."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["broadcasting-rules", "einops-repeat-broadcast"],
    "lo": (
        "Compose einops repeat-broadcast (zero-copy axis insertion) with broadcasting rules to compute "
        "pairwise L2 distances between two row sets without materializing the (N, M, D) intermediate."
    ),
}


# ===========================================================================
# cx30 — reduce then expand via repeat-broadcast
# ===========================================================================
spec_30 = {
    "atom_ids": ["einops-reduce", "einops-repeat-broadcast"],
    "subtopics": _subs(["einops-reduce", "einops-repeat-broadcast"]),
    "primary_atom": "einops-reduce",
    "part": "part0",
    "exercise_index": 30,
    "exercise_title": "reduce to per-row max, then repeat-broadcast for softmax-style subtraction",
    "slug": "reduce-then-repeat-broadcast-softmax",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Numerically-stable softmax is the canonical reduce-then-repeat-broadcast pattern. To prevent "
        "exp overflow you subtract each row's MAX from every element of that row before exping — i.e. "
        "you need to broadcast a `(B,)` per-row-max back across the original `(B, D)` matrix.\n\n"
        "Two einops atoms compose here:\n"
        "  1. `reduce(x, 'b d -> b', 'max')` collapses D into a per-row scalar.\n"
        "  2. `repeat(row_max, 'b -> b d', d=D)` re-inserts the d-axis as a stride-0 view (no copy).\n\n"
        "After the repeat, the shapes match `(B, D)` and broadcasting handles the subtraction. This "
        "is identical to `x - x.max(dim=1, keepdim=True).values` — but the einops named pattern makes "
        "the reduce / re-broadcast split EXPLICIT."
    ),
    "prompt_body": (
        "Implement `cx30_stable_softmax(x)` — numerically stable softmax of an `(B, D)` matrix using "
        "the reduce-then-repeat-broadcast composition.\n\n"
        "1. **Reduce** to a per-row max with `reduce(x, 'b d -> b', 'max')`. The result has shape `(B,)`.\n"
        "2. **Repeat-broadcast** the per-row max back to `(B, D)` with `repeat(row_max, 'b -> b d', "
        "d=D)`. This is a stride-0 view — no copy.\n"
        "3. Subtract, `.exp()`, then divide by per-row sum (use the same reduce-then-repeat-broadcast "
        "pattern for the denominator).\n\n"
        "Return shape `(B, D)`. Cross-check against `torch.nn.functional.softmax(x, dim=1)`."
    ),
    "stub_body": (
        "def cx30_stable_softmax(x):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "\n"
        "# Case A: cross-check against F.softmax(dim=1).\n"
        "x = t.randn(4, 16)\n"
        "out = cx30_stable_softmax(x)\n"
        "assert tuple(out.shape) == (4, 16), f'shape: {tuple(out.shape)}'\n"
        "ref = F.softmax(x, dim=1)\n"
        "assert t.allclose(out, ref, atol=1e-6), f'max diff {(out-ref).abs().max()}'\n"
        "\n"
        "# Case B: rows sum to 1.\n"
        "row_sums = out.sum(dim=1)\n"
        "assert t.allclose(row_sums, t.ones(4), atol=1e-5), f'row sums: {row_sums}'\n"
        "\n"
        "# Case C: numerical stability — large positive logits.\n"
        "big = t.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])\n"
        "out_big = cx30_stable_softmax(big)\n"
        "assert t.isfinite(out_big).all(), 'softmax must not overflow on large logits'\n"
        "assert t.allclose(out_big.sum(dim=1), t.ones(2), atol=1e-5)\n"
        "# Cross-check against F.softmax which is also numerically stable.\n"
        "assert t.allclose(out_big, F.softmax(big, dim=1), atol=1e-6)\n"
        "\n"
        "# Case D: realistic shape (transformer logits).\n"
        "logits = t.randn(8, 50257)\n"
        "probs = cx30_stable_softmax(logits)\n"
        "assert tuple(probs.shape) == (8, 50257)\n"
        "assert t.allclose(probs.sum(dim=1), t.ones(8), atol=1e-4)\n"
        "assert t.allclose(probs, F.softmax(logits, dim=1), atol=1e-5)"
    ),
    "solution_body": (
        "def cx30_stable_softmax(x):\n"
        "    B, D = x.shape\n"
        "    # Atom A (einops-reduce): collapse D into per-row max — shape (B,).\n"
        "    row_max = reduce(x, 'b d -> b', 'max')\n"
        "    # Atom B (einops-repeat-broadcast): insert d-axis as stride-0 view — shape (B, D).\n"
        "    row_max_b = repeat(row_max, 'b -> b d', d=D)\n"
        "    shifted = x - row_max_b\n"
        "    ex = shifted.exp()\n"
        "    # Same composition for the denominator: reduce to (B,), then repeat-broadcast to (B, D).\n"
        "    row_sum = reduce(ex, 'b d -> b', 'sum')\n"
        "    row_sum_b = repeat(row_sum, 'b -> b d', d=D)\n"
        "    return ex / row_sum_b"
    ),
    "solution_notes": (
        "The reduce-then-repeat-broadcast pair is the einops-native version of `keepdim=True`. You can "
        "fuse the two into a single `reduce(..., 'b d -> b 1', 'max')` call (which is what cx27 does) "
        "and let plain broadcasting handle the alignment — but separating them out makes the data flow "
        "explicit and keeps the `repeat` step zero-copy (stride-0 view). Either form compiles to the "
        "same `expand` under the hood; the choice is about code clarity for the next reader."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["einops-reduce", "einops-repeat-broadcast"],
    "lo": (
        "Compose einops reduce (collapse the D-axis) with einops repeat-broadcast (re-insert the "
        "D-axis as a stride-0 view) to implement numerically-stable softmax."
    ),
}


SPECS = [spec_25, spec_26, spec_27, spec_28, spec_29, spec_30]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
