"""Composite drills cx7..cx12 — batch-16 (R-cell, part0).

Six composite procedural drills pairing einops.reduce with five sister
einops atoms (plus the sum/broadcast duality back-fn pair). Each composite
forces the learner to wire reduce together with another atom in one fn.

cx7   einops-reduce + broadcasting-rules
cx8   einops-reduce + einops-repeat
cx9   einops-reduce + einops-rearrange-flatten
cx10  einops-reduce + einops-repeat-broadcast
cx11  einops-reduce + sum-and-broadcast-duality
cx12  einops-rearrange + einops-reduce
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
# cx7 — reduce along an axis, then broadcast result back to original shape
# ===========================================================================
spec_7 = {
    "atom_ids": ["einops-reduce", "broadcasting-rules"],
    "subtopics": _subs(["einops-reduce", "broadcasting-rules"]),
    "primary_atom": "einops-reduce",
    "part": "part0",
    "exercise_index": 7,
    "exercise_title": "channel-centered image via reduce + right-align broadcast",
    "slug": "reduce-then-broadcast-back",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Centering a feature map along an axis is a two-step move that exercises both atoms in "
        "one expression. First `einops.reduce(x, 'b c h w -> b 1 h w', 'mean')` collapses the channel "
        "axis to a singleton — the keepdim-style reduce that returns the mean with the reduced axis "
        "still present as a size-1 slot. Then `x - mean` triggers NumPy/PyTorch broadcasting: shapes "
        "`(B, C, H, W)` and `(B, 1, H, W)` right-align, the size-1 axis is stretched, and the result "
        "lands at `x.shape`. Atom 1 controls WHAT shape comes out of the reduction; atom 2 explains "
        "WHY subtracting the size-1 tensor doesn't error and lands at the right shape."
    ),
    "prompt_body": (
        "Implement `cx7_center_per_channel_then_broadcast(x)` that takes a 4-D feature map of shape "
        "`(B, C, H, W)` and returns a tensor of the SAME shape, where each `(B, H, W)` slot has been "
        "centered against its per-batch per-channel mean over the spatial axes.\n\n"
        "Two atoms, one expression:\n\n"
        "1. **Reduce with keepdim semantics** — use `einops.reduce(x, 'b c h w -> b c 1 1', 'mean')` "
        "(NOT `x.mean(dim=(-2,-1))`). The point is to keep the reduced axes as size-1 slots so the "
        "broadcast back works without a manual `unsqueeze`.\n"
        "2. **Right-align broadcast** — return `x - mean`. The shapes are `(B, C, H, W)` vs `(B, C, 1, 1)`; "
        "broadcasting rules right-align them, every pair is either equal or has a 1, so the result is "
        "`(B, C, H, W)`. Do not call `.expand` or `.repeat`.\n\n"
        "Also assert (inside your fn) that `mean.shape == (B, C, 1, 1)` to make atom 1 visible."
    ),
    "stub_body": (
        "def cx7_center_per_channel_then_broadcast(x):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "x = t.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()\n"
        "out = cx7_center_per_channel_then_broadcast(x)\n"
        "assert out.shape == x.shape, f'shape mismatch: {out.shape} vs {x.shape}'\n"
        "# Each per-(b,c) spatial slice must now have mean 0.\n"
        "slice_means = out.mean(dim=(-2, -1))\n"
        "assert t.allclose(slice_means, t.zeros_like(slice_means), atol=1e-5), slice_means\n"
        "# Numerical cross-check: equivalent to x - x.mean(dim=(-2,-1), keepdim=True).\n"
        "ref = x - x.mean(dim=(-2, -1), keepdim=True)\n"
        "assert t.allclose(out, ref), 'differs from keepdim mean subtraction'\n"
        "\n"
        "# Case B: non-uniform values + different shape.\n"
        "x2 = t.randn(4, 2, 6, 7)\n"
        "out2 = cx7_center_per_channel_then_broadcast(x2)\n"
        "assert out2.shape == x2.shape\n"
        "assert t.allclose(out2.mean(dim=(-2, -1)), t.zeros(4, 2), atol=1e-5)\n"
        "\n"
        "# Case C: implementation must use einops.reduce (sanity: a single-element batch still works).\n"
        "x3 = t.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])  # (1,1,2,3)\n"
        "out3 = cx7_center_per_channel_then_broadcast(x3)\n"
        "assert out3.shape == (1, 1, 2, 3)\n"
        "assert t.allclose(out3.mean(), t.tensor(0.0), atol=1e-5)"
    ),
    "solution_body": (
        "def cx7_center_per_channel_then_broadcast(x):\n"
        "    # Atom 1: reduce H,W to size-1 slots (keepdim-style via einops pattern).\n"
        "    mean = reduce(x, 'b c h w -> b c 1 1', 'mean')\n"
        "    B, C, _, _ = x.shape\n"
        "    assert mean.shape == (B, C, 1, 1), mean.shape\n"
        "    # Atom 2: broadcasting rules — (B,C,H,W) vs (B,C,1,1) right-aligns; size-1 axes stretch.\n"
        "    return x - mean"
    ),
    "solution_notes": (
        "Critically, the pattern `'b c h w -> b c 1 1'` is the einops way of asking for keepdim=True. "
        "Without the size-1 placeholders, you'd get `(B, C)` and the subtraction would need an explicit "
        "`unsqueeze(-1).unsqueeze(-1)` — losing the composition."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["reduce-pick-aggregator", "predict-broadcast-shape"],
    "lo": (
        "Compose einops.reduce with keepdim-style size-1 placeholders and right-align broadcasting "
        "rules to center a 4-D feature map per (batch, channel) without explicit unsqueeze."
    ),
}


# ===========================================================================
# cx8 — reduce then re-tile across a NEW axis with repeat
# ===========================================================================
spec_8 = {
    "atom_ids": ["einops-reduce", "einops-repeat"],
    "subtopics": _subs(["einops-reduce", "einops-repeat"]),
    "primary_atom": "einops-reduce",
    "part": "part0",
    "exercise_index": 8,
    "exercise_title": "per-batch mean then re-tile across a new heads axis",
    "slug": "reduce-then-repeat-tile",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Reduce and repeat are inverses on the named-axis side: `reduce` drops a labelled axis, "
        "`repeat` introduces a new labelled axis bound by a kwarg. Composing them in one fn — "
        "`reduce` collapses the existing C axis to a scalar-per-batch, then `repeat` materialises a "
        "fresh `heads` axis of size H so every head sees the same per-batch summary — is the standard "
        "ARENA move when you need a broadcasted query for each head/sample/copy.\n\n"
        "This is NOT the same as broadcasting back into the original C axis. `repeat` actually "
        "introduces a NEW named output dim, with `heads=H` as a kwarg binding."
    ),
    "prompt_body": (
        "Implement `cx8_per_batch_mean_per_head(x, heads)` that takes a tensor of shape `(B, C)` and "
        "an `int heads` and returns a tensor of shape `(B, heads)` where every head row is a copy of "
        "the per-batch mean over C.\n\n"
        "Two atoms in one expression:\n\n"
        "1. **Reduce** the channel axis with `einops.reduce(x, 'b c -> b', 'mean')` — note the axis "
        "is fully removed (no size-1 slot here, because the next step adds a *different* named axis).\n"
        "2. **Repeat** the per-batch scalar across a fresh `heads` axis with `einops.repeat(per_batch, "
        "'b -> b heads', heads=heads)`. The new axis is bound by kwarg, not derived from the input."
    ),
    "stub_body": (
        "def cx8_per_batch_mean_per_head(x, heads):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "x = t.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])  # (B=2, C=3)\n"
        "out = cx8_per_batch_mean_per_head(x, heads=4)\n"
        "assert out.shape == (2, 4), f'expected (2,4), got {out.shape}'\n"
        "# Every head row of a given batch must equal the per-batch mean.\n"
        "expected_per_batch = t.tensor([3.0, 4.0])\n"
        "for h in range(4):\n"
        "    assert t.allclose(out[:, h], expected_per_batch), out\n"
        "\n"
        "# Case B: random + heads=1 (degenerate but legal).\n"
        "x2 = t.randn(5, 7)\n"
        "out2 = cx8_per_batch_mean_per_head(x2, heads=1)\n"
        "assert out2.shape == (5, 1)\n"
        "assert t.allclose(out2[:, 0], x2.mean(dim=1))\n"
        "\n"
        "# Case C: heads=8 — verify stride-0 / repeat semantics (all heads identical).\n"
        "x3 = t.randn(3, 4)\n"
        "out3 = cx8_per_batch_mean_per_head(x3, heads=8)\n"
        "assert out3.shape == (3, 8)\n"
        "assert t.allclose(out3.std(dim=1), t.zeros(3), atol=1e-6), 'heads axis should be uniform'"
    ),
    "solution_body": (
        "def cx8_per_batch_mean_per_head(x, heads):\n"
        "    # Atom 1: reduce — drop the C axis entirely (b c -> b).\n"
        "    per_batch = reduce(x, 'b c -> b', 'mean')\n"
        "    # Atom 2: repeat — introduce a NEW named axis 'heads', bound by kwarg.\n"
        "    return repeat(per_batch, 'b -> b heads', heads=heads)"
    ),
    "solution_notes": (
        "The reduce drops a labelled axis; the repeat introduces one. They are not the same axis — that "
        "would be no-op identity. The kwarg `heads=heads` is what binds the new axis size."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["reduce-pick-aggregator", "repeat-add-axis"],
    "lo": (
        "Compose einops.reduce (drop axis) with einops.repeat (introduce new kwarg-bound axis) to "
        "produce a per-batch summary tiled across a fresh heads dimension."
    ),
}


# ===========================================================================
# cx9 — flatten-by-group via einops, then reduce within each group
# ===========================================================================
spec_9 = {
    "atom_ids": ["einops-reduce", "einops-rearrange-flatten"],
    "subtopics": _subs(["einops-reduce", "einops-rearrange-flatten"]),
    "primary_atom": "einops-reduce",
    "part": "part0",
    "exercise_index": 9,
    "exercise_title": "group-flatten then reduce within each group — avg-pool style",
    "slug": "flatten-by-group-then-reduce",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's average-pool / patch-summary moves want to (a) RESHAPE so the pool window lives on a "
        "single named axis, then (b) REDUCE over that axis. `einops.rearrange` with parenthesised axis "
        "composition is the flatten step: `'b c (h1 h2) (w1 w2) -> b c (h1 w1) (h2 w2)'` regroups the "
        "spatial axes so every pool window is contiguous. Then `einops.reduce` over the inner "
        "`(h2 w2)` axis is the actual pooling op.\n\n"
        "Composing them tests both atoms: you must (i) write a rearrange pattern that uses "
        "axis-composition via parens (atom: rearrange-flatten) and (ii) pick the right reduce "
        "aggregator + pattern (atom: reduce)."
    ),
    "prompt_body": (
        "Implement `cx9_pool_2x2_via_einops(x)` that takes a 4-D feature map of shape `(B, C, H, W)` "
        "with `H % 2 == 0` and `W % 2 == 0`, and returns a `(B, C, H//2, W//2)` average-pooled tensor.\n\n"
        "Two atoms, one composition:\n\n"
        "1. **Rearrange-flatten** — use `einops.rearrange(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1 "
        "(h2 w2)', h2=2, w2=2)`. Note the trailing `(h2 w2)` — axis composition via parens collapses "
        "the 2×2 pool window onto a single inner axis of length 4. Bind h2/w2 by kwarg.\n"
        "2. **Reduce** — call `einops.reduce(..., 'b c h1 w1 win -> b c h1 w1', 'mean')` to collapse "
        "the new `win` axis. Renaming `(h2 w2) -> win` happens in the rearrange step, so you may use "
        "any inner-axis name in the reduce.\n\n"
        "Equivalent to `F.avg_pool2d(x, kernel_size=2)`, but the point is to do it with einops."
    ),
    "stub_body": (
        "def cx9_pool_2x2_via_einops(x):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "x = t.arange(1 * 2 * 4 * 6).reshape(1, 2, 4, 6).float()\n"
        "out = cx9_pool_2x2_via_einops(x)\n"
        "assert out.shape == (1, 2, 2, 3), f'expected (1,2,2,3), got {out.shape}'\n"
        "ref = F.avg_pool2d(x, kernel_size=2)\n"
        "assert t.allclose(out, ref), f'differs from avg_pool2d: {out} vs {ref}'\n"
        "\n"
        "# Case B: larger random batch.\n"
        "x2 = t.randn(3, 4, 8, 8)\n"
        "out2 = cx9_pool_2x2_via_einops(x2)\n"
        "assert out2.shape == (3, 4, 4, 4)\n"
        "assert t.allclose(out2, F.avg_pool2d(x2, kernel_size=2))\n"
        "\n"
        "# Case C: smallest legal — H=W=2 → output (B, C, 1, 1).\n"
        "x3 = t.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])\n"
        "out3 = cx9_pool_2x2_via_einops(x3)\n"
        "assert out3.shape == (1, 1, 1, 1)\n"
        "assert t.allclose(out3, t.tensor([[[[2.5]]]]))"
    ),
    "solution_body": (
        "def cx9_pool_2x2_via_einops(x):\n"
        "    # Atom A: rearrange-flatten — axis-composition via parens regroups the 2x2 pool window\n"
        "    # onto a single trailing 'win' axis of length 4.\n"
        "    regrouped = rearrange(\n"
        "        x,\n"
        "        'b c (h1 h2) (w1 w2) -> b c h1 w1 (h2 w2)',\n"
        "        h2=2, w2=2,\n"
        "    )\n"
        "    # Atom B: reduce — collapse the 'win' axis with 'mean'.\n"
        "    return reduce(regrouped, 'b c h1 w1 win -> b c h1 w1', 'mean')"
    ),
    "solution_notes": (
        "The parenthesised pattern is the magic — `(h1 h2)` means 'this axis is the COMPOSITION of two "
        "logical axes h1 and h2'. einops needs `h2=2` as a kwarg to know how to factor the existing H "
        "into h1 × h2. Once the window lives on its own axis, reduce is just one more line."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["rearrange-axis-composition-via-parens", "reduce-pick-aggregator"],
    "lo": (
        "Compose einops.rearrange (axis composition via parens) with einops.reduce ('mean') to "
        "implement 2x2 average pooling without any explicit reshape or strided op."
    ),
}


# ===========================================================================
# cx10 — reduce then expand via repeat-as-broadcast (zero-stride pairing)
# ===========================================================================
spec_10 = {
    "atom_ids": ["einops-reduce", "einops-repeat-broadcast"],
    "subtopics": _subs(["einops-reduce", "einops-repeat-broadcast"]),
    "primary_atom": "einops-reduce",
    "part": "part0",
    "exercise_index": 10,
    "exercise_title": "row mean then pair every row-mean with every column via repeat-broadcast",
    "slug": "reduce-then-repeat-broadcast",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`einops.repeat` has two distinct uses: introducing a new named axis (kwarg-bound), and "
        "broadcasting an existing tensor against another by inserting a zero-stride axis. ARENA's "
        "ray-tracing pair-every-with-every drill is the canonical repeat-as-broadcast example.\n\n"
        "Here we compose it with reduce. First reduce a 2-D `(R, C)` matrix to its per-row mean of "
        "shape `(R,)`. Then use `repeat` to broadcast the row-means against every column index, "
        "yielding a `(R, N)` 'pair-every-with-every' tensor where row r, column n holds the r-th row "
        "mean. This is repeat-as-broadcast: no copy semantics — einops materialises a zero-stride view."
    ),
    "prompt_body": (
        "Implement `cx10_row_mean_paired_with_n_cols(x, n)` that takes a `(R, C)` matrix and an int "
        "`n` and returns a `(R, n)` tensor where every column holds the per-row mean of `x`.\n\n"
        "Two atoms in one fn:\n\n"
        "1. **Reduce** — collapse the column axis with `einops.reduce(x, 'r c -> r', 'mean')`.\n"
        "2. **Repeat-as-broadcast** — use `einops.repeat(row_means, 'r -> r n', n=n)` to pair every "
        "row-mean with every output column. This is the same pattern as ARENA's "
        "ray-with-triangle pairing: insert a new axis whose size is bound by kwarg, and every slot "
        "along the new axis holds the same row-mean (zero-stride broadcast semantics).\n\n"
        "Sanity check inside the fn: `assert row_means.shape == (R,)` so atom 1 is visible."
    ),
    "stub_body": (
        "def cx10_row_mean_paired_with_n_cols(x, n):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "x = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # (R=3, C=3)\n"
        "out = cx10_row_mean_paired_with_n_cols(x, n=4)\n"
        "assert out.shape == (3, 4), f'expected (3,4), got {out.shape}'\n"
        "# Per-row means: 2, 5, 8 — must appear in every column.\n"
        "expected_means = t.tensor([2.0, 5.0, 8.0])\n"
        "for col in range(4):\n"
        "    assert t.allclose(out[:, col], expected_means), out\n"
        "\n"
        "# Case B: random R x C, n=1.\n"
        "x2 = t.randn(7, 5)\n"
        "out2 = cx10_row_mean_paired_with_n_cols(x2, n=1)\n"
        "assert out2.shape == (7, 1)\n"
        "assert t.allclose(out2[:, 0], x2.mean(dim=1))\n"
        "\n"
        "# Case C: uniform-along-new-axis invariant (std along n is zero).\n"
        "x3 = t.randn(4, 6)\n"
        "out3 = cx10_row_mean_paired_with_n_cols(x3, n=10)\n"
        "assert out3.shape == (4, 10)\n"
        "assert t.allclose(out3.std(dim=1), t.zeros(4), atol=1e-6), 'new axis must be uniform'"
    ),
    "solution_body": (
        "def cx10_row_mean_paired_with_n_cols(x, n):\n"
        "    # Atom 1: reduce — drop the C axis to get per-row scalars.\n"
        "    row_means = reduce(x, 'r c -> r', 'mean')\n"
        "    assert row_means.shape == (x.shape[0],), row_means.shape\n"
        "    # Atom 2: repeat-as-broadcast — insert a new 'n' axis bound by kwarg; every slot is\n"
        "    # the same row_mean. einops models this as a zero-stride broadcast under the hood.\n"
        "    return repeat(row_means, 'r -> r n', n=n)"
    ),
    "solution_notes": (
        "Repeat is not always 'copy data' — when used to pair every X with every Y, it's a "
        "broadcast/expand under the hood. einops's contract is on shape, not memory: the output looks "
        "like every column holds the row-mean, and the framework picks the most efficient impl."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["reduce-pick-aggregator", "repeat-inserts-zero-stride-axis"],
    "lo": (
        "Compose einops.reduce ('mean' over an axis) with einops.repeat (zero-stride broadcast along "
        "a new kwarg-bound axis) to pair every row-summary with every output column."
    ),
}


# ===========================================================================
# cx11 — sum/broadcast duality realised via einops.reduce 'sum'
# ===========================================================================
spec_11 = {
    "atom_ids": ["einops-reduce", "sum-and-broadcast-duality"],
    "subtopics": _subs(["einops-reduce", "sum-and-broadcast-duality"]),
    "primary_atom": "einops-reduce",
    "part": "part0",
    "exercise_index": 11,
    "exercise_title": "sum-broadcast duality via einops.reduce 'sum' as the back-fn",
    "slug": "sum-broadcast-duality-via-reduce",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Forward `sum(x, dim)` is a reduce; its backward expands the upstream grad. Forward "
        "`broadcast_to(x, shape)` is an expand; its backward sums out the expanded axes. The two are "
        "exact adjoints — the sum/broadcast duality.\n\n"
        "The trick: you don't need a hand-rolled `unbroadcast` to write `broadcast_back`. "
        "`einops.reduce` with `'sum'` IS the unbroadcast. If forward expanded `(R,)` to `(R, C)` "
        "(broadcast_to introduces or stretches axes), then `broadcast_back` is "
        "`reduce(grad_out, 'r c -> r', 'sum')`. And if forward summed `(R, C) -> (R,)`, then "
        "`sum_back` re-broadcasts via `repeat(grad_out, 'r -> r c', c=C)`. The reduce atom and the "
        "duality atom collapse into the same einops vocabulary."
    ),
    "prompt_body": (
        "Implement TWO functions wired together, then sanity-check they round-trip:\n\n"
        "1. `cx11_sum_forward(x)` — forward pass: collapse the column axis of `(R, C)` to `(R,)` "
        "using `einops.reduce(x, 'r c -> r', 'sum')`. Return the result.\n"
        "2. `cx11_broadcast_back(grad_out, x)` — backward of a `(R,) -> (R, C)` broadcast "
        "(equivalently, the gradient w.r.t. the size-1 / absent axis when forward broadcast). "
        "Implement it as `einops.reduce(grad_out, 'r c -> r', 'sum')` — exact same call pattern as "
        "the sum forward. The duality says: backward of broadcast IS reduce-with-sum.\n\n"
        "Round-trip invariant: if forward was `out = x.sum(dim=1)` with grad_out `ones_like(out)`, "
        "then `broadcast_back(ones_like(x), x)` must equal `ones_like(out)` (each column contributes "
        "1 unit of grad → the row total is C ones, but the GRAD for the original `(R,)` shape — i.e. "
        "the broadcast input — sums those C ones back into a scalar per row). "
        "More precisely, calling `broadcast_back(t.ones_like(x), x)` returns a `(R,)` tensor of all-C."
    ),
    "stub_body": (
        "def cx11_sum_forward(x):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx11_broadcast_back(grad_out, x):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "x = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (R=2, C=3)\n"
        "# Forward sum.\n"
        "out = cx11_sum_forward(x)\n"
        "assert out.shape == (2,), out.shape\n"
        "assert t.allclose(out, t.tensor([6.0, 15.0]))\n"
        "\n"
        "# Backward of broadcast: ones_like(x) -> (R,) of all C.\n"
        "g = cx11_broadcast_back(t.ones_like(x), x)\n"
        "assert g.shape == (2,), g.shape\n"
        "assert t.allclose(g, t.tensor([3.0, 3.0]))\n"
        "\n"
        "# Duality cross-check via autograd: y = x.sum(dim=1), then dy/dx is ones_like(x);\n"
        "# the broadcast_back of dy/dx must be of all-C (the sum-down of a uniform broadcast).\n"
        "x2 = t.randn(4, 5, requires_grad=True)\n"
        "y2 = x2.sum(dim=1)\n"
        "y2.backward(t.ones_like(y2))\n"
        "g2 = cx11_broadcast_back(x2.grad, x2)\n"
        "assert g2.shape == (4,)\n"
        "assert t.allclose(g2, t.full((4,), 5.0))\n"
        "\n"
        "# And the forward pass itself uses einops.reduce: cross-check vs torch sum.\n"
        "x3 = t.randn(3, 7)\n"
        "assert t.allclose(cx11_sum_forward(x3), x3.sum(dim=1))"
    ),
    "solution_body": (
        "def cx11_sum_forward(x):\n"
        "    # Atom: einops.reduce — collapse C with 'sum'.\n"
        "    return reduce(x, 'r c -> r', 'sum')\n"
        "\n"
        "def cx11_broadcast_back(grad_out, x):\n"
        "    # Atom: sum/broadcast duality — backward of broadcast IS reduce-with-'sum'.\n"
        "    # Same einops call pattern as the forward sum; that IS the duality.\n"
        "    return reduce(grad_out, 'r c -> r', 'sum')"
    ),
    "solution_notes": (
        "The point of the composite is to make the duality concrete: when both forward sum and the "
        "backward of broadcast share the SAME einops.reduce('sum') call, the adjoint relationship "
        "is no longer an abstract claim — it's the same line of code."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["reduce-pick-aggregator", "sum-and-broadcast-duality"],
    "lo": (
        "Show that the backward of broadcast is exactly einops.reduce('sum') by writing forward-sum "
        "and broadcast-back with the same call signature."
    ),
}


# ===========================================================================
# cx12 — rearrange dims for reduce convenience, then reduce
# ===========================================================================
spec_12 = {
    "atom_ids": ["einops-rearrange", "einops-reduce"],
    "subtopics": _subs(["einops-rearrange", "einops-reduce"]),
    "primary_atom": "einops-rearrange",
    "part": "part0",
    "exercise_index": 12,
    "exercise_title": "rearrange to put reducible axes last, then reduce them",
    "slug": "rearrange-then-reduce",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Real-world tensors don't always come pre-arranged in the order you want to reduce over. "
        "`einops.rearrange` is the pure axis-reorder primitive — no shape change, just a permutation "
        "of named axes. Once the axes you want to collapse are in a convenient position, "
        "`einops.reduce` with a per-axis pattern is trivial.\n\n"
        "Composing them means: write a rearrange pattern that ONLY permutes axes (no flattening, no "
        "size change), then write a reduce pattern that drops the right ones. The composition is the "
        "ARENA-standard pre-process-then-reduce idiom."
    ),
    "prompt_body": (
        "Implement `cx12_rearrange_then_reduce_max(x)` that takes a 4-D tensor of shape "
        "`(B, H, W, C)` (channels-last) and returns a `(B, C)` tensor of the per-(batch, channel) "
        "MAX over spatial H, W axes.\n\n"
        "Two-step composition:\n\n"
        "1. **Rearrange** the channels-last layout to channels-first via "
        "`einops.rearrange(x, 'b h w c -> b c h w')`. This is the identity-up-to-permutation atom — "
        "no axis flattening, no size change, just a permutation.\n"
        "2. **Reduce** the spatial axes with `einops.reduce(permuted, 'b c h w -> b c', 'max')`. "
        "Per-(batch, channel) max over H,W.\n\n"
        "Assert inside the fn that `permuted.shape == (B, C, H, W)` so atom 1 is visible. The point "
        "is NOT to write `x.permute(0, 3, 1, 2).amax(dim=(-2, -1))` — the point is to write both "
        "einops calls and verify each is type-pure (rearrange = pure axis reorder; reduce = pure axis "
        "drop)."
    ),
    "stub_body": (
        "def cx12_rearrange_then_reduce_max(x):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "x = t.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()  # (B=2, H=3, W=4, C=5)\n"
        "out = cx12_rearrange_then_reduce_max(x)\n"
        "assert out.shape == (2, 5), f'expected (2,5), got {out.shape}'\n"
        "# Cross-check vs torch: amax over H,W of the channels-first view.\n"
        "ref = x.permute(0, 3, 1, 2).amax(dim=(-2, -1))\n"
        "assert t.allclose(out, ref), f'mismatch: {out} vs {ref}'\n"
        "\n"
        "# Case B: random + larger spatial.\n"
        "x2 = t.randn(4, 8, 8, 3)\n"
        "out2 = cx12_rearrange_then_reduce_max(x2)\n"
        "assert out2.shape == (4, 3)\n"
        "assert t.allclose(out2, x2.permute(0, 3, 1, 2).amax(dim=(-2, -1)))\n"
        "\n"
        "# Case C: degenerate H=W=1 — max equals the single value per (B, C).\n"
        "x3 = t.randn(2, 1, 1, 6)\n"
        "out3 = cx12_rearrange_then_reduce_max(x3)\n"
        "assert out3.shape == (2, 6)\n"
        "assert t.allclose(out3, x3.squeeze(1).squeeze(1))"
    ),
    "solution_body": (
        "def cx12_rearrange_then_reduce_max(x):\n"
        "    B, H, W, C = x.shape\n"
        "    # Atom 1: rearrange — pure axis permutation, no shape change beyond the reorder.\n"
        "    permuted = rearrange(x, 'b h w c -> b c h w')\n"
        "    assert permuted.shape == (B, C, H, W), permuted.shape\n"
        "    # Atom 2: reduce — drop the spatial axes with 'max'.\n"
        "    return reduce(permuted, 'b c h w -> b c', 'max')"
    ),
    "solution_notes": (
        "The split is intentional: each einops call is type-pure. The rearrange does ONLY axis "
        "reordering (no parens-flatten, no kwarg-bound new axis); the reduce does ONLY axis dropping "
        "(no axis composition, no new axes). Composing them gives you the pre-process-then-reduce "
        "idiom without leaving einops vocabulary."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["rearrange-identity-pattern", "reduce-pick-aggregator"],
    "lo": (
        "Compose einops.rearrange (pure permutation) with einops.reduce ('max' drop) to compute a "
        "channels-last spatial-max in pure einops vocabulary."
    ),
}


for spec in [spec_7, spec_8, spec_9, spec_10, spec_11, spec_12]:
    out = emit_composite(spec)
    print(f"wrote {out}")
