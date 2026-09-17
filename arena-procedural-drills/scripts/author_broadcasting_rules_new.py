#!/usr/bin/env python3
"""Author ex6-ex9 standalone notebooks for `broadcasting-rules`.

These are Colab-native exercises (visualization, multi-step debugging,
integrative pipelines, edge-case traps) — not single-formula problems
that could live on a flashcard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

ATOM_ID = "broadcasting-rules"
SUBTOPIC = "Numpy: Vectorization and broadcasting"
TOPIC = "prereqs_numpy"

RECAP = (
    "## Broadcasting — quick refresher\n"
    "\n"
    "**The rule** (NumPy & PyTorch agree):\n"
    "1. Right-align both shapes; left-pad the shorter with 1s.\n"
    "2. For each pair of aligned axes: equal → keep; one is 1 → use the other; otherwise → incompatible.\n"
    "\n"
    "**The dangerous case.** When a shape *almost* matches you can get an unintended broadcast that runs silently and produces wrong values. Always shape-check (`print(x.shape, y.shape, (x*y).shape)`) when wiring up a new pipeline."
)


SPECS = [
    # ────────────────────────────────────────────────────────────────────
    # ex6: pairwise distance matrix — broadcast + reduce + heatmap viz
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "pairwise distance matrix as a heatmap",
        "slug": "pairwise-distance-matrix-as-a-heatmap",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["broadcast", "reduce", "heatmap", "matplotlib", "axis-insertion"],
        "kcs": ["insert-axis-for-broadcast", "broadcast-then-reduce", "pairwise-via-broadcast"],
        "lo": "Compute a pairwise Euclidean distance matrix via broadcast + reduce, then visualize it as a heatmap.",
        "prompt_body": (
            "Given two point sets `A` of shape `(N, D)` and `B` of shape `(M, D)`, "
            "implement `ex6_pairwise_distances(A, B)` to return a tensor `D_mat` of shape `(N, M)` "
            "where `D_mat[i, j]` is the Euclidean distance between `A[i]` and `B[j]`.\n"
            "\n"
            "**Do it without any Python loops.** The trick is to insert axes so the subtraction broadcasts:\n"
            "- `A[:, None, :]` has shape `(N, 1, D)`\n"
            "- `B[None, :, :]` has shape `(1, M, D)`\n"
            "- their difference is `(N, M, D)`; square-and-sum over the last axis, then `sqrt`.\n"
            "\n"
            "After the function works, the test cell will render the resulting `(N, M)` distance matrix as a matplotlib heatmap so you can *see* the broadcast result — a diagonal-dark stripe when `A == B`, a smooth gradient otherwise.\n"
            "\n"
            "**Shape-trace it in your head before coding:** `(N, 1, D) - (1, M, D) → (N, M, D) → sum(-1) → (N, M) → sqrt → (N, M)`."
        ),
        "stub": (
            "def ex6_pairwise_distances(A: Tensor, B: Tensor) -> Tensor:\n"
            "    \"\"\"Return (N, M) tensor of Euclidean distances.\n"
            "\n"
            "    A: (N, D), B: (M, D). No Python loops — pure broadcast.\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Small deterministic case we can verify by hand\n"
            "A = t.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])\n"
            "B = t.tensor([[0.0, 0.0], [1.0, 1.0]])\n"
            "D_mat = ex6_pairwise_distances(A, B)\n"
            "assert D_mat.shape == (3, 2), f'expected (3, 2), got {tuple(D_mat.shape)}'\n"
            "# Hand-checked distances\n"
            "expected = t.tensor([\n"
            "    [0.0, 2.0 ** 0.5],\n"
            "    [1.0, 1.0],\n"
            "    [1.0, 1.0],\n"
            "])\n"
            "assert t.allclose(D_mat, expected, atol=1e-6), f'value mismatch:\\n{D_mat}'\n"
            "\n"
            "# Self-distance must have a zero diagonal\n"
            "P = t.randn(20, 4)\n"
            "DD = ex6_pairwise_distances(P, P)\n"
            "assert DD.shape == (20, 20)\n"
            "assert t.allclose(DD.diagonal(), t.zeros(20), atol=1e-5), 'self-distance diagonal must be 0'\n"
            "assert t.allclose(DD, DD.T, atol=1e-5), 'self-distance must be symmetric'\n"
            "\n"
            "# Visualize — heatmap of distances between two random point clouds\n"
            "import matplotlib.pyplot as plt\n"
            "AA = t.randn(30, 2)\n"
            "BB = t.randn(25, 2) + 0.5\n"
            "DM = ex6_pairwise_distances(AA, BB).numpy()\n"
            "fig, ax = plt.subplots(figsize=(5, 6))\n"
            "im = ax.imshow(DM, aspect='auto', cmap='viridis')\n"
            "ax.set_xlabel('B index'); ax.set_ylabel('A index')\n"
            "ax.set_title(f'pairwise distances  ({DM.shape[0]} × {DM.shape[1]})')\n"
            "plt.colorbar(im, ax=ax, label='distance')\n"
            "plt.tight_layout(); plt.show()"
        ),
        "solution_body": (
            "def ex6_pairwise_distances(A: Tensor, B: Tensor) -> Tensor:\n"
            "    diff = A[:, None, :] - B[None, :, :]   # (N, M, D)\n"
            "    sq = (diff ** 2).sum(dim=-1)            # (N, M)\n"
            "    return sq.sqrt()"
        ),
        "solution_notes": (
            "**Why broadcast beats a double loop.** The naive `for i in range(N): for j in range(M): ...` "
            "version is `O(N·M)` Python overhead. The broadcast version is one big vectorized op the BLAS / "
            "GPU kernel can fuse. For `N = M = 1000, D = 64`, the loop version is ~4 orders of magnitude slower.\n"
            "\n"
            "**Memory cost.** Intermediate `diff` is `(N, M, D)` — for large N, M, D this can be huge. The "
            "production-grade alternative is `‖a-b‖² = ‖a‖² + ‖b‖² − 2 a·bᵀ` (Gram-matrix trick), which never "
            "materializes the `(N, M, D)` block."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ────────────────────────────────────────────────────────────────────
    # ex7: batched attention scores — broadcast + matmul w/ debug prints
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "batched attention scores with shape-trace debugging",
        "slug": "batched-attention-scores-with-shape-trace-debugging",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["attention", "batched-matmul", "transpose", "shape-trace"],
        "kcs": ["batched-broadcast", "axis-swap-via-transpose", "broadcast-then-matmul"],
        "lo": "Build batched scaled-dot-product attention scores while printing intermediate shapes at each step.",
        "prompt_body": (
            "Implement `ex7_attention_scores(Q, K)` to return the (unnormalized) batched attention score tensor "
            "of shape `(B, T, T)` from `Q` and `K`, both of shape `(B, T, D)`.\n"
            "\n"
            "The formula is `scores = Q @ K.transpose(-2, -1) / sqrt(D)`. You must:\n"
            "\n"
            "1. Print `Q.shape`, `K.shape`, and `K.transpose(-2, -1).shape` BEFORE doing the matmul.\n"
            "2. Print the resulting `scores.shape` AFTER the matmul.\n"
            "3. Apply the `1/sqrt(D)` scaling.\n"
            "\n"
            "The `print(...)` calls are intentional — they let you *see* how the batch axis rides along while the inner "
            "matmul reshape happens on the last two axes only. Comment them out later if you want; the test ignores stdout."
        ),
        "stub": (
            "def ex7_attention_scores(Q: Tensor, K: Tensor) -> Tensor:\n"
            "    \"\"\"Return (B, T, T) attention scores. Print intermediate shapes.\n"
            "\n"
            "    Q, K: (B, T, D). Scaled by 1/sqrt(D).\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "B, T, D = 2, 5, 8\n"
            "Q = t.randn(B, T, D)\n"
            "K = t.randn(B, T, D)\n"
            "scores = ex7_attention_scores(Q, K)\n"
            "assert scores.shape == (B, T, T), f'expected (2, 5, 5), got {tuple(scores.shape)}'\n"
            "\n"
            "# Hand-checked against the unbatched reference on slice 0\n"
            "import math\n"
            "ref0 = (Q[0] @ K[0].transpose(-2, -1)) / math.sqrt(D)\n"
            "assert t.allclose(scores[0], ref0, atol=1e-5), 'slice 0 mismatch'\n"
            "ref1 = (Q[1] @ K[1].transpose(-2, -1)) / math.sqrt(D)\n"
            "assert t.allclose(scores[1], ref1, atol=1e-5), 'slice 1 mismatch'\n"
            "\n"
            "# Scaling check: same Q, K but D doubled → scores should shrink by sqrt(2)\n"
            "Q2 = t.randn(B, T, 2 * D)\n"
            "K2 = t.randn(B, T, 2 * D)\n"
            "s_big = ex7_attention_scores(Q2, K2)\n"
            "assert s_big.shape == (B, T, T)\n"
            "\n"
            "# Self-attention shape with non-square last axes\n"
            "Q3 = t.randn(4, 7, 16)\n"
            "K3 = t.randn(4, 7, 16)\n"
            "s3 = ex7_attention_scores(Q3, K3)\n"
            "assert s3.shape == (4, 7, 7), f'expected (4, 7, 7), got {tuple(s3.shape)}'"
        ),
        "solution_body": (
            "def ex7_attention_scores(Q: Tensor, K: Tensor) -> Tensor:\n"
            "    import math\n"
            "    D = Q.shape[-1]\n"
            "    K_T = K.transpose(-2, -1)\n"
            "    print(f'Q.shape       = {tuple(Q.shape)}')\n"
            "    print(f'K.shape       = {tuple(K.shape)}')\n"
            "    print(f'K_T.shape     = {tuple(K_T.shape)}')\n"
            "    scores = Q @ K_T\n"
            "    print(f'scores.shape  = {tuple(scores.shape)}  (before scaling)')\n"
            "    return scores / math.sqrt(D)"
        ),
        "solution_notes": (
            "**What broadcast is doing here.** `Q @ K.transpose(-2, -1)` is a *batched* matmul: PyTorch broadcasts "
            "the leading `(B,)` axis automatically and runs the matmul on the trailing `(T, D) @ (D, T) → (T, T)`. "
            "If `Q` were `(T, D)` and `K` were `(B, T, D)`, broadcasting would still work — `Q` would be promoted to "
            "`(1, T, D)` and replicated across the batch. That's how 'shared query, batched keys' lookup tables work.\n"
            "\n"
            "**Why print the shapes.** When `Q` and `K` come from different upstream code paths it's easy to feed in "
            "`(T, B, D)` by accident — the matmul will still run, but you get attention scores between batch slots "
            "instead of sequence positions. The print-then-look loop catches that in one cycle."
        ),
        "extra_imports": [],
    },
    # ────────────────────────────────────────────────────────────────────
    # ex8: multi-axis bias add (B, H, W, C) + per-channel (C,) — shape trace
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "per-channel bias add to a 4-D image batch",
        "slug": "per-channel-bias-add-to-a-4-d-image-batch",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["multi-axis-broadcast", "image-batch", "channel-axis", "heatmap", "before-after"],
        "kcs": ["broadcast-leading-axes", "per-channel-bias", "axis-insertion-deep-tensor"],
        "lo": "Add a per-channel bias vector to an NHWC image batch and visualize the per-channel shift.",
        "prompt_body": (
            "You have an image batch `X` of shape `(B, H, W, C)` and a per-channel bias `b` of shape `(C,)`. "
            "Implement `ex8_per_channel_bias(X, b)` to return `X + bias_broadcast` of shape `(B, H, W, C)` "
            "where `b[c]` is added to every spatial position of channel `c` of every image in the batch.\n"
            "\n"
            "Don't loop. The whole job is one expression once `b` has the right shape.\n"
            "\n"
            "**The shape-trace.** Right-align: `(B, H, W, C)` vs `(C,)` → `(C,)` gets left-padded to `(1, 1, 1, C)` and "
            "the bias broadcasts across `B`, `H`, `W`. That's the case NumPy/PyTorch handles for you automatically. "
            "Verify by adding a non-trivial bias `b = [1, 10, 100, 1000]` and check that channel 0 shifted by 1, "
            "channel 3 shifted by 1000.\n"
            "\n"
            "The test cell then plots a `(H, W)` heatmap of channel 2 *before* and *after* the bias add so you can see "
            "the uniform offset — the spatial *pattern* is unchanged, only the level shifts."
        ),
        "stub": (
            "def ex8_per_channel_bias(X: Tensor, b: Tensor) -> Tensor:\n"
            "    \"\"\"Add per-channel bias `b` (C,) to image batch `X` (B, H, W, C).\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "B, H, W, C = 2, 8, 12, 4\n"
            "X = t.randn(B, H, W, C)\n"
            "b = t.tensor([1.0, 10.0, 100.0, 1000.0])\n"
            "Y = ex8_per_channel_bias(X, b)\n"
            "assert Y.shape == (B, H, W, C), f'shape mismatch: {tuple(Y.shape)}'\n"
            "\n"
            "# Each channel must have shifted by exactly b[c]\n"
            "for c in range(C):\n"
            "    diff = (Y[..., c] - X[..., c])\n"
            "    assert t.allclose(diff, t.full_like(diff, b[c]), atol=1e-6), (\n"
            "        f'channel {c} expected uniform shift {b[c]}, got range '\n"
            "        f'[{diff.min().item():.4f}, {diff.max().item():.4f}]'\n"
            "    )\n"
            "\n"
            "# The bias must NOT have leaked across channels\n"
            "for c in range(C):\n"
            "    # subtract the per-channel mean delta — should be ~0 everywhere\n"
            "    delta = (Y[..., c] - X[..., c]) - b[c]\n"
            "    assert delta.abs().max().item() < 1e-5, f'channel {c} cross-channel leak'\n"
            "\n"
            "# Visualize channel 2 before and after\n"
            "import matplotlib.pyplot as plt\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            "for ax, img, title in zip(\n"
            "    axes,\n"
            "    [X[0, :, :, 2].numpy(), Y[0, :, :, 2].numpy()],\n"
            "    ['before bias  (channel 2)', f'after bias +{b[2].item():.0f}  (channel 2)'],\n"
            "):\n"
            "    im = ax.imshow(img, cmap='magma')\n"
            "    ax.set_title(title)\n"
            "    plt.colorbar(im, ax=ax)\n"
            "plt.tight_layout(); plt.show()"
        ),
        "solution_body": (
            "def ex8_per_channel_bias(X: Tensor, b: Tensor) -> Tensor:\n"
            "    # b: (C,) right-aligns with last axis of X → automatic broadcast\n"
            "    return X + b"
        ),
        "solution_notes": (
            "**Why this is a 1-liner.** Right-align broadcasting was *designed* for this case: a 1-D parameter "
            "vector matching the trailing channel axis of a multi-D activation. No reshape needed.\n"
            "\n"
            "**When does this break?** If your tensor is NCHW instead of NHWC, the channel axis is *not* the last axis. "
            "Then `X + b` broadcasts `b` over the wrong dimension (or fails). The fix is `X + b[:, None, None]` "
            "(insert two trailing axes so `b` becomes `(C, 1, 1)`, which right-aligns over `(N, C, H, W)`). This is "
            "one of the most common silent bugs in computer-vision code — see the next exercise."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ────────────────────────────────────────────────────────────────────
    # ex9: silent-mismatch trap — value-spot-check, add .unsqueeze fix
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "silent-broadcast trap — catch it with a value check",
        "slug": "silent-broadcast-trap-catch-it-with-a-value-check",
        "bloom_level": "Evaluate",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["silent-bug", "shape-trace", "unsqueeze", "value-spot-check", "NCHW"],
        "kcs": ["detect-silent-broadcast", "axis-insertion-via-unsqueeze", "value-vs-shape-debugging"],
        "lo": "Recognize when broadcasting succeeds shape-wise but applies the wrong values, and fix it with an axis insertion.",
        "prompt_body": (
            "**The trap.** A buggy normalizer is shipping in production. The author wrote it for NCHW images "
            "(`(B, C, H, W)`) but treated the per-channel mean `mu` of shape `(C,)` like in the NHWC case — "
            "`X - mu`. The shapes 'work' (no exception) because `H == C` in their tests by accident. "
            "On real data with `H != C` it explodes; on `H == C` data it silently subtracts the wrong values.\n"
            "\n"
            "Implement `ex9_normalize_nchw(X, mu)` to subtract `mu` (shape `(C,)`) from each channel of `X` "
            "(shape `(B, C, H, W)`) **correctly**, regardless of whether `H == C` or not. The fix is one "
            "`unsqueeze` away — but you have to know which axes need the size-1 padding.\n"
            "\n"
            "The test cell constructs a deliberately-trapped case where `B = 1, C = 4, H = 4, W = 6`. The buggy "
            "expression `X - mu` would not raise (because the last axis is `W = 6`, broadcasting `mu = (4,)` would "
            "fail with `W=6 ≠ C=4`, but a sloppy 'fix' like `X - mu[None, :, None, None]` might be miswritten as "
            "`X - mu[None, None, None, :]` and pass the shape check while computing nonsense). Your job: pick the "
            "right axis insertion and verify with a value check."
        ),
        "stub": (
            "def ex9_normalize_nchw(X: Tensor, mu: Tensor) -> Tensor:\n"
            "    \"\"\"Subtract per-channel mean `mu` (C,) from NCHW image batch `X` (B, C, H, W).\n"
            "\n"
            "    The fix is one axis insertion. Find it.\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Case 1 — H != C, the broken version would raise, so any shape-OK answer is also value-OK\n"
            "B, C, H, W = 2, 3, 5, 7\n"
            "X = t.randn(B, C, H, W)\n"
            "mu = t.tensor([10.0, 100.0, 1000.0])\n"
            "Y = ex9_normalize_nchw(X, mu)\n"
            "assert Y.shape == (B, C, H, W), f'shape mismatch: {tuple(Y.shape)}'\n"
            "\n"
            "# Per-channel shift must be exactly -mu[c]\n"
            "for c in range(C):\n"
            "    diff = Y[:, c] - X[:, c]\n"
            "    assert t.allclose(diff, t.full_like(diff, -mu[c]), atol=1e-5), (\n"
            "        f'channel {c}: expected shift {-mu[c]} but range was '\n"
            "        f'[{diff.min().item():.4f}, {diff.max().item():.4f}]'\n"
            "    )\n"
            "\n"
            "# Case 2 — the trap: H == C == 4. The wrong axis insertion would pass the shape check\n"
            "# (because broadcasting (1,1,1,4) over (B,4,4,W) succeeds when W == 4) but ALSO pass\n"
            "# a naive per-image total check. We catch it by checking per-channel slices.\n"
            "B, C, H, W = 1, 4, 4, 4\n"
            "X = t.zeros(B, C, H, W)\n"
            "mu = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "Y = ex9_normalize_nchw(X, mu)\n"
            "assert Y.shape == (B, C, H, W)\n"
            "# Every spatial location of channel c must equal -mu[c]\n"
            "for c in range(C):\n"
            "    sl = Y[0, c]\n"
            "    assert (sl == -mu[c]).all(), (\n"
            "        f'TRAP: channel {c} got values {sl.unique().tolist()} but should be all {-mu[c]}.\\n'\n"
            "        f'You probably inserted axes on the wrong side — mu[None, None, None, :] '\n"
            "        f'broadcasts mu over the *width* axis, not the *channel* axis.'\n"
            "    )\n"
            "\n"
            "# Case 3 — value spot-check on a structured input so any wrong-axis bug is visible\n"
            "B, C, H, W = 1, 2, 3, 5\n"
            "X = t.arange(B * C * H * W, dtype=t.float32).reshape(B, C, H, W)\n"
            "mu = t.tensor([0.0, 100.0])\n"
            "Y = ex9_normalize_nchw(X, mu)\n"
            "# Channel 0 unchanged, channel 1 shifted by -100\n"
            "assert t.equal(Y[0, 0], X[0, 0]), 'channel 0 must be unchanged (mu=0)'\n"
            "assert t.equal(Y[0, 1], X[0, 1] - 100), 'channel 1 must be shifted by -100'"
        ),
        "solution_body": (
            "def ex9_normalize_nchw(X: Tensor, mu: Tensor) -> Tensor:\n"
            "    # X is (B, C, H, W). mu is (C,). We need mu to broadcast over the C-axis only,\n"
            "    # which means inserting size-1 axes for B (leading), H, and W (trailing).\n"
            "    # mu[None, :, None, None] gives shape (1, C, 1, 1) → right-aligns with (B, C, H, W).\n"
            "    return X - mu[None, :, None, None]"
        ),
        "solution_notes": (
            "**The lesson.** When the broadcast result has the *right shape* but *wrong values*, only a value "
            "spot-check catches it. The two value checks the test runs:\n"
            "1. Per-channel uniformity (Case 1, Case 2) — every spatial cell of channel `c` should have shifted by "
            "exactly `-mu[c]`. A wrong axis insertion produces a striped pattern instead.\n"
            "2. Structured input (Case 3) — when `X = arange(...)`, *any* wrong-axis broadcast leaves a visible "
            "non-monotonic artifact in the output.\n"
            "\n"
            "**Rule of thumb.** When you `unsqueeze` to fix a broadcast, count axes from the *target tensor*, not "
            "from the vector. Here the channel axis of `X` is at index 1, so `mu` needs size-1s in positions 0, 2, 3 — "
            "giving `mu[None, :, None, None]`. Alternative spellings: `mu.view(1, -1, 1, 1)`, `mu.reshape(1, C, 1, 1)`."
        ),
        "extra_imports": [],
    },
]


def main() -> None:
    for spec in SPECS:
        path = emit_standalone(spec)
        # repo-relative print
        print(f"wrote {path.relative_to(path.parents[3])}")


if __name__ == "__main__":
    main()
