#!/usr/bin/env python3
"""Author ex6-ex9 standalone notebooks for `boolean-mask-identity-replace`.

These are Colab-native exercises (visualization, multi-step debugging,
integrative ML pipelines, edge-case exploration) — not single-formula
problems that could live on a flashcard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

ATOM_ID = "boolean-mask-identity-replace"
SUBTOPIC = "Numpy: Indexing and selection"
TOPIC = "prereqs_numpy"

RECAP = (
    "## Mask & substitute — quick refresher\n"
    "\n"
    "**Build a mask.** Any comparison returns a `dtype=bool` tensor of the same shape: `x < 0`, `x.abs() < eps`, `(x > 0) & (x < 1)`. Combine with `&`, `|`, `~`.\n"
    "\n"
    "**Write through a mask.** `y[mask] = value` modifies in place. Scalars broadcast; tensor values must match the shape of `y[mask]` after broadcasting. Always `clone()` first if the function must not mutate its input.\n"
    "\n"
    "**The dangerous case.** When a mask is the *wrong* shape, indexing can silently collapse axes or pick the wrong cells. Always check `mask.sum()` and `mask.shape` before trusting the result."
)


SPECS = [
    # ────────────────────────────────────────────────────────────────────
    # ex6: causal attention mask — build, apply, visualize
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "causal attention mask — build, apply, visualize",
        "slug": "causal-attention-mask-build-apply-visualize",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["causal-mask", "attention", "lower-triangular", "imshow", "-inf-fill"],
        "kcs": ["mask-from-comparison", "masked-fill-neg-inf", "broadcast-mask-over-batch"],
        "lo": "Construct a causal attention mask via index comparison, apply it to scores with -inf, and visualize both.",
        "prompt_body": (
            "Implement `ex6_causal_mask_apply(scores)` to (1) build a causal (lower-triangular) boolean mask "
            "for sequence length `T = scores.shape[-1]`, then (2) return a copy of `scores` with the masked-out "
            "(strictly upper-triangular) positions set to `-inf`.\n"
            "\n"
            "Build the mask via index comparison (not `torch.tril` — we want you to see the broadcast):\n"
            "- `idx = torch.arange(T)`  → shape `(T,)`\n"
            "- `mask = idx[None, :] <= idx[:, None]`  → shape `(T, T)`, lower-triangular, True on/below the diagonal\n"
            "\n"
            "Then `scores.masked_fill(~mask, float('-inf'))` (or `scores[~mask] = -inf` on a clone — but masked_fill is cleaner because it broadcasts to the leading batch axes for free).\n"
            "\n"
            "After the function passes, the test cell plots two heatmaps side-by-side: the boolean mask, and a softmax of the masked scores — you should see a clean lower-triangular structure in both."
        ),
        "stub": (
            "def ex6_causal_mask_apply(scores: Tensor) -> Tensor:\n"
            "    \"\"\"Apply causal mask to attention scores of shape (..., T, T).\n"
            "\n"
            "    Returns a tensor of the same shape with upper-triangular positions set to -inf.\n"
            "    Build the mask via arange + broadcast comparison; do NOT use torch.tril.\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Unbatched\n"
            "T = 5\n"
            "scores = t.randn(T, T)\n"
            "masked = ex6_causal_mask_apply(scores)\n"
            "assert masked.shape == (T, T)\n"
            "for i in range(T):\n"
            "    for j in range(T):\n"
            "        if j > i:\n"
            "            assert masked[i, j].item() == float('-inf'), f'expected -inf at ({i},{j}) got {masked[i, j].item()}'\n"
            "        else:\n"
            "            assert masked[i, j].item() == scores[i, j].item(), f'value changed at ({i},{j})'\n"
            "\n"
            "# Batched — mask broadcasts over leading axes\n"
            "B, H, T = 2, 3, 4\n"
            "batched = t.randn(B, H, T, T)\n"
            "out = ex6_causal_mask_apply(batched)\n"
            "assert out.shape == (B, H, T, T)\n"
            "# upper-triangular positions in EVERY batch slice must be -inf\n"
            "for b in range(B):\n"
            "    for h in range(H):\n"
            "        for i in range(T):\n"
            "            for j in range(i + 1, T):\n"
            "                assert out[b, h, i, j].item() == float('-inf')\n"
            "        # diagonal + below preserved\n"
            "        for i in range(T):\n"
            "            for j in range(i + 1):\n"
            "                assert out[b, h, i, j].item() == batched[b, h, i, j].item()\n"
            "\n"
            "# Softmax check: after masking, each row's softmax must sum to 1 and be zero on the upper-tri\n"
            "T = 6\n"
            "raw = t.randn(T, T)\n"
            "m = ex6_causal_mask_apply(raw)\n"
            "probs = m.softmax(dim=-1)\n"
            "assert t.allclose(probs.sum(dim=-1), t.ones(T), atol=1e-5), 'softmax rows must sum to 1'\n"
            "for i in range(T):\n"
            "    for j in range(i + 1, T):\n"
            "        assert probs[i, j].item() == 0.0, f'masked position ({i},{j}) should be 0 after softmax'\n"
            "\n"
            "# Visualize: build the mask + plot the softmax of masked scores\n"
            "import matplotlib.pyplot as plt\n"
            "T_vis = 10\n"
            "raw_vis = t.randn(T_vis, T_vis)\n"
            "idx = t.arange(T_vis)\n"
            "mask_vis = (idx[None, :] <= idx[:, None])\n"
            "probs_vis = ex6_causal_mask_apply(raw_vis).softmax(dim=-1)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9, 4))\n"
            "axes[0].imshow(mask_vis.numpy(), cmap='gray_r')\n"
            "axes[0].set_title('causal mask  (True = keep)')\n"
            "im = axes[1].imshow(probs_vis.numpy(), cmap='viridis')\n"
            "axes[1].set_title('softmax(masked scores)')\n"
            "plt.colorbar(im, ax=axes[1])\n"
            "for ax in axes:\n"
            "    ax.set_xlabel('key position'); ax.set_ylabel('query position')\n"
            "plt.tight_layout(); plt.show()"
        ),
        "solution_body": (
            "def ex6_causal_mask_apply(scores: Tensor) -> Tensor:\n"
            "    T = scores.shape[-1]\n"
            "    idx = t.arange(T, device=scores.device)\n"
            "    keep = idx[None, :] <= idx[:, None]   # (T, T), True on/below diagonal\n"
            "    return scores.masked_fill(~keep, float('-inf'))"
        ),
        "solution_notes": (
            "**Why mask BEFORE softmax.** `softmax(-inf) = 0` exactly, so the masked positions get zero probability "
            "*and* the remaining unmasked positions still sum to 1 (the `exp(-inf) = 0` terms drop out of the denominator). "
            "If you instead masked AFTER softmax (multiplying by the bool mask), each row would no longer sum to 1.\n"
            "\n"
            "**Why arange-broadcast and not `torch.tril(torch.ones(...))`.** Both work, but the arange-compare form "
            "generalizes — swap `<=` for `< k` and you get a *banded* mask, swap `idx` for token-position IDs and you "
            "get a per-token-aware mask (useful when packing multiple sequences into one row)."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ────────────────────────────────────────────────────────────────────
    # ex7: padded-sequence mean — multi-step debug
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "padded-sequence mean ignoring pad positions",
        "slug": "padded-sequence-mean-ignoring-pad-positions",
        "bloom_level": "Apply",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["padding-mask", "masked-mean", "broadcast-mask", "axis-reduce", "transformer-pooling"],
        "kcs": ["broadcast-mask-over-feature-axis", "masked-sum-then-normalize", "guard-against-zero-length"],
        "lo": "Compute per-sequence mean pooling over padded (B, T, D) inputs using a padding mask.",
        "prompt_body": (
            "Given a padded batch `X` of shape `(B, T, D)` and a padding mask `pad_mask` of shape `(B, T)` where "
            "`True` marks **real (non-pad)** positions, implement `ex7_masked_mean(X, pad_mask)` to return a tensor "
            "of shape `(B, D)` where each row is the *mean of the real positions only* — pad positions must not "
            "contribute, and the divisor must be the per-sequence real-length, not `T`.\n"
            "\n"
            "**Multi-step debug.** Before returning, `print(...)` four things in this order so you can introspect the pipeline:\n"
            "1. `pad_mask.shape` and `pad_mask.sum(dim=1)` — per-sequence real-length\n"
            "2. The expanded-mask shape after `pad_mask.unsqueeze(-1)` — should be `(B, T, 1)`\n"
            "3. The masked-sum shape — `(B, D)`\n"
            "4. The divisor — `pad_mask.sum(dim=1, keepdim=True).clamp(min=1).float()` — shape `(B, 1)`, with the "
            "`clamp(min=1)` guard so an all-pad row doesn't divide by zero.\n"
            "\n"
            "Verify by hand: for a single sequence of 3 real tokens and 2 pad tokens, the output should equal "
            "the unmasked mean of the first 3 rows."
        ),
        "stub": (
            "def ex7_masked_mean(X: Tensor, pad_mask: Tensor) -> Tensor:\n"
            "    \"\"\"Per-sequence mean of real positions in a padded (B, T, D) batch.\n"
            "\n"
            "    X        : (B, T, D)\n"
            "    pad_mask : (B, T)  True = real token, False = pad\n"
            "    returns  : (B, D)  per-sequence mean over real positions only\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-checked case — 1 sequence, 5 timesteps, 3 real + 2 pad, D = 2\n"
            "X = t.tensor([[\n"
            "    [1.0, 10.0],\n"
            "    [2.0, 20.0],\n"
            "    [3.0, 30.0],\n"
            "    [99.0, 99.0],  # pad — must NOT contribute\n"
            "    [99.0, 99.0],  # pad\n"
            "]])\n"
            "pad_mask = t.tensor([[True, True, True, False, False]])\n"
            "out = ex7_masked_mean(X, pad_mask)\n"
            "assert out.shape == (1, 2), f'expected (1, 2), got {tuple(out.shape)}'\n"
            "expected = t.tensor([[2.0, 20.0]])  # (1+2+3)/3, (10+20+30)/3\n"
            "assert t.allclose(out, expected, atol=1e-6), f'expected {expected.tolist()}, got {out.tolist()}'\n"
            "\n"
            "# Variable-length batch\n"
            "B, T, D = 3, 4, 2\n"
            "X2 = t.tensor([\n"
            "    [[1., 1.], [1., 1.], [0., 0.], [0., 0.]],  # 2 real, mean = [1, 1]\n"
            "    [[2., 4.], [4., 8.], [6., 12.], [0., 0.]],  # 3 real, mean = [4, 8]\n"
            "    [[5., 0.], [0., 0.], [0., 0.], [0., 0.]],  # 1 real, mean = [5, 0]\n"
            "])\n"
            "pad_mask2 = t.tensor([\n"
            "    [True, True, False, False],\n"
            "    [True, True, True, False],\n"
            "    [True, False, False, False],\n"
            "])\n"
            "out2 = ex7_masked_mean(X2, pad_mask2)\n"
            "exp2 = t.tensor([[1., 1.], [4., 8.], [5., 0.]])\n"
            "assert t.allclose(out2, exp2, atol=1e-6), f'expected {exp2}, got {out2}'\n"
            "\n"
            "# Edge case — all-pad sequence must not NaN. Mean should be 0 (divisor clamped to 1)\n"
            "X3 = t.tensor([[[5., 5.], [5., 5.]]])\n"
            "pad3 = t.tensor([[False, False]])\n"
            "out3 = ex7_masked_mean(X3, pad3)\n"
            "assert out3.shape == (1, 2)\n"
            "assert t.isfinite(out3).all(), f'all-pad sequence produced non-finite output: {out3}'\n"
            "assert t.allclose(out3, t.zeros(1, 2), atol=1e-6), f'all-pad must yield zeros, got {out3}'"
        ),
        "solution_body": (
            "def ex7_masked_mean(X: Tensor, pad_mask: Tensor) -> Tensor:\n"
            "    # 1. Per-sequence real-length\n"
            "    lengths = pad_mask.sum(dim=1)              # (B,)\n"
            "    print(f'pad_mask.shape = {tuple(pad_mask.shape)}, lengths = {lengths.tolist()}')\n"
            "\n"
            "    # 2. Expand mask to (B, T, 1) so it broadcasts over D\n"
            "    m = pad_mask.unsqueeze(-1).to(X.dtype)     # (B, T, 1)\n"
            "    print(f'expanded mask shape = {tuple(m.shape)}')\n"
            "\n"
            "    # 3. Masked sum — pad positions contribute 0\n"
            "    s = (X * m).sum(dim=1)                     # (B, D)\n"
            "    print(f'masked sum shape    = {tuple(s.shape)}')\n"
            "\n"
            "    # 4. Divisor with all-pad guard\n"
            "    div = lengths.clamp(min=1).unsqueeze(-1).to(X.dtype)   # (B, 1)\n"
            "    print(f'divisor shape       = {tuple(div.shape)}, divisor = {div.squeeze(-1).tolist()}')\n"
            "    return s / div"
        ),
        "solution_notes": (
            "**Why `unsqueeze(-1)` and not `unsqueeze(1)`.** Right-align: `(B, T, D)` vs `(B, T, 1)` works (broadcast "
            "the trailing 1 over D). `(B, T)` alone would right-align as `(_, B, T)` and fail. The mental model is "
            "'add a feature axis to the mask so it lines up with the feature axis of the data'.\n"
            "\n"
            "**Why clamp the divisor.** An all-pad row has `lengths[i] = 0`. Dividing by 0 gives NaN, which then poisons "
            "every downstream operation. `clamp(min=1)` returns 0/1 = 0 for that row, which is a sensible default — "
            "the upstream loss should mask out the all-pad row anyway, but defending in depth is cheap.\n"
            "\n"
            "**Real-world use.** This is exactly how BERT-style sentence-mean pooling, HuggingFace `mean_pooling`, "
            "and many sequence-level classifiers compute their pooled representation."
        ),
        "extra_imports": [],
    },
    # ────────────────────────────────────────────────────────────────────
    # ex8: outlier removal with multi-criterion mask — scatter viz
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "outlier removal with combined boolean masks",
        "slug": "outlier-removal-with-combined-boolean-masks",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["mask-combine", "bitwise-and", "bitwise-or", "scatter-plot", "outlier"],
        "kcs": ["combine-masks-with-&-and-|", "negate-mask-with-~", "boolean-row-selection"],
        "lo": "Combine 2-3 boolean masks via &/|/~ to select a subset of rows, and visualize the keep/drop split.",
        "prompt_body": (
            "Given a `(N, 2)` array of 2-D points `pts` and an `(N,)` quality score `score`, implement "
            "`ex8_keep_clean_points(pts, score)` to return three things:\n"
            "\n"
            "- `kept` — the subset of `pts` that satisfy ALL of:\n"
            "  1. inside the unit box: `-1 ≤ x ≤ 1` AND `-1 ≤ y ≤ 1`\n"
            "  2. `score > 0.5`\n"
            "  3. NOT (both coordinates near zero): NOT (`abs(x) < 0.05` AND `abs(y) < 0.05`)  ← rejects degenerate origin cluster\n"
            "- `kept_mask` — the `(N,)` bool mask used to select `kept`\n"
            "- `dropped_mask` — the negation, `~kept_mask`\n"
            "\n"
            "Return as a tuple `(kept, kept_mask, dropped_mask)`. Build each criterion as its own bool tensor first, "
            "then combine with `&` and `~`. The test cell will use the masks to color points kept vs dropped on a scatter "
            "plot — you should see only the 'good' annulus survive."
        ),
        "stub": (
            "def ex8_keep_clean_points(pts: Tensor, score: Tensor):\n"
            "    \"\"\"Filter (N, 2) points by combined criteria.\n"
            "\n"
            "    Returns (kept, kept_mask, dropped_mask).\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-crafted set of 8 points covering each criterion\n"
            "pts = t.tensor([\n"
            "    [0.0, 0.0],     # 0  origin → drop (criterion 3)\n"
            "    [0.5, 0.5],     # 1  inside, score>0.5, not origin → keep\n"
            "    [2.0, 0.0],     # 2  outside unit box → drop (criterion 1)\n"
            "    [-0.5, 0.5],    # 3  inside, score>0.5, not origin → keep\n"
            "    [0.5, -0.5],    # 4  inside, score=0.3 (below thresh) → drop (criterion 2)\n"
            "    [0.01, 0.01],   # 5  near origin → drop (criterion 3)\n"
            "    [0.9, 0.9],     # 6  inside, score>0.5, not origin → keep\n"
            "    [-1.5, 0.0],    # 7  outside unit box → drop\n"
            "])\n"
            "score = t.tensor([0.9, 0.9, 0.9, 0.9, 0.3, 0.9, 0.9, 0.9])\n"
            "kept, km, dm = ex8_keep_clean_points(pts, score)\n"
            "expected_keep_idx = t.tensor([False, True, False, True, False, False, True, False])\n"
            "assert t.equal(km, expected_keep_idx), f'kept_mask mismatch: {km.tolist()}'\n"
            "assert t.equal(dm, ~expected_keep_idx), 'dropped_mask must be negation of kept_mask'\n"
            "assert kept.shape == (3, 2), f'expected 3 kept points, got {tuple(kept.shape)}'\n"
            "assert t.equal(kept, pts[expected_keep_idx]), 'kept rows must equal pts[kept_mask]'\n"
            "\n"
            "# Cardinality\n"
            "assert km.sum().item() == 3\n"
            "assert dm.sum().item() == 5\n"
            "assert (km & dm).sum().item() == 0, 'kept and dropped must be disjoint'\n"
            "assert (km | dm).sum().item() == km.numel(), 'kept | dropped must cover all rows'\n"
            "\n"
            "# Visualize on a larger random cloud\n"
            "import matplotlib.pyplot as plt\n"
            "t.manual_seed(42)\n"
            "N = 400\n"
            "pts_big = t.empty(N, 2).uniform_(-1.5, 1.5)\n"
            "score_big = t.rand(N)\n"
            "kp, kmb, dmb = ex8_keep_clean_points(pts_big, score_big)\n"
            "fig, ax = plt.subplots(figsize=(6, 6))\n"
            "ax.scatter(pts_big[dmb, 0].numpy(), pts_big[dmb, 1].numpy(),\n"
            "           c='lightgray', s=12, label=f'dropped ({dmb.sum().item()})')\n"
            "ax.scatter(pts_big[kmb, 0].numpy(), pts_big[kmb, 1].numpy(),\n"
            "           c='tab:blue', s=18, label=f'kept ({kmb.sum().item()})')\n"
            "ax.axhline(0, lw=0.3); ax.axvline(0, lw=0.3)\n"
            "ax.set_aspect('equal'); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)\n"
            "ax.set_title('outlier removal — kept vs dropped')\n"
            "ax.legend(loc='upper right')\n"
            "plt.tight_layout(); plt.show()"
        ),
        "solution_body": (
            "def ex8_keep_clean_points(pts: Tensor, score: Tensor):\n"
            "    x, y = pts[:, 0], pts[:, 1]\n"
            "    in_box       = (x >= -1) & (x <= 1) & (y >= -1) & (y <= 1)\n"
            "    high_score   = score > 0.5\n"
            "    near_origin  = (x.abs() < 0.05) & (y.abs() < 0.05)\n"
            "    kept_mask    = in_box & high_score & (~near_origin)\n"
            "    dropped_mask = ~kept_mask\n"
            "    return pts[kept_mask], kept_mask, dropped_mask"
        ),
        "solution_notes": (
            "**`&` vs `and`.** Always use `&` / `|` / `~` for bool tensors. Python `and` / `or` short-circuit on a "
            "scalar bool, so on a tensor they raise `RuntimeError: Boolean value of Tensor with more than one element "
            "is ambiguous`. The bitwise operators are elementwise.\n"
            "\n"
            "**Parens matter.** `x >= -1 & x <= 1` parses as `x >= (-1 & x) <= 1` because `&` binds tighter than `>=`. "
            "Always parenthesize each comparison: `(x >= -1) & (x <= 1)`.\n"
            "\n"
            "**`pts[mask]` vs `pts[mask, :]`.** When `mask` is 1-D, both are equivalent and produce a 2-D result. "
            "When the mask matches multiple axes (e.g. 2-D mask on 2-D tensor), the result is 1-D and you've collapsed "
            "the geometry — that's a common surprise."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ────────────────────────────────────────────────────────────────────
    # ex9: NMS baseline — iterative mask update, integrative
    # ────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "non-max suppression via iterative mask update",
        "slug": "non-max-suppression-via-iterative-mask-update",
        "bloom_level": "Analyze",
        "difficulty_num": 5,
        "difficulty_dots": "🔴🔴🔴🔴🔴",
        "keywords": ["nms", "iterative-mask", "suppression", "object-detection", "integrative"],
        "kcs": ["mask-update-in-a-loop", "argmax-of-masked-vector", "mask-from-row-of-matrix"],
        "lo": "Implement a baseline non-max suppression by iteratively selecting the highest-scoring survivor and masking out its neighbours.",
        "prompt_body": (
            "Implement `ex9_nms(scores, overlap, iou_threshold)`, a baseline NMS that takes:\n"
            "- `scores`: `(N,)` confidence scores\n"
            "- `overlap`: `(N, N)` symmetric matrix where `overlap[i, j]` is the IoU between box `i` and box `j`. Diagonal is 1.\n"
            "- `iou_threshold`: float\n"
            "\n"
            "Return an `(N,)` boolean tensor `kept` where `kept[i] = True` iff box `i` survives suppression.\n"
            "\n"
            "**Algorithm** (you must use boolean-mask updates — no `argsort`, no `for box in sorted_boxes` Python list):\n"
            "\n"
            "```\n"
            "alive = torch.ones(N, dtype=torch.bool)\n"
            "kept  = torch.zeros(N, dtype=torch.bool)\n"
            "while alive.any():\n"
            "    # 1. pick the alive box with highest score\n"
            "    masked_scores = scores.clone()\n"
            "    masked_scores[~alive] = -float('inf')\n"
            "    i = int(masked_scores.argmax())\n"
            "    kept[i]  = True\n"
            "    alive[i] = False\n"
            "    # 2. suppress every alive box that overlaps too much with i\n"
            "    suppress  = (overlap[i] > iou_threshold) & alive\n"
            "    alive[suppress] = False\n"
            "return kept\n"
            "```\n"
            "\n"
            "Implement that. The integrative test will give you a small set of partially-overlapping boxes with "
            "hand-checkable survivors."
        ),
        "stub": (
            "def ex9_nms(scores: Tensor, overlap: Tensor, iou_threshold: float) -> Tensor:\n"
            "    \"\"\"Baseline non-max suppression via iterative boolean-mask updates.\n"
            "\n"
            "    Returns a (N,) bool tensor: True for surviving boxes.\n"
            "    \"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Case 1 — hand-crafted: 4 boxes. 0 and 1 overlap heavily; 2 and 3 are isolated.\n"
            "# scores: 0=0.9 (highest), 1=0.8 (overlaps 0), 2=0.7 (alone), 3=0.6 (alone)\n"
            "scores = t.tensor([0.9, 0.8, 0.7, 0.6])\n"
            "overlap = t.tensor([\n"
            "    [1.0, 0.8, 0.0, 0.0],\n"
            "    [0.8, 1.0, 0.0, 0.0],\n"
            "    [0.0, 0.0, 1.0, 0.1],\n"
            "    [0.0, 0.0, 0.1, 1.0],\n"
            "])\n"
            "kept = ex9_nms(scores, overlap, iou_threshold=0.5)\n"
            "assert kept.dtype == t.bool, f'expected bool dtype, got {kept.dtype}'\n"
            "assert kept.shape == (4,), f'expected (4,), got {tuple(kept.shape)}'\n"
            "# Expected: keep 0 (highest, suppresses 1), keep 2 (isolated), keep 3 (isolated)\n"
            "expected = t.tensor([True, False, True, True])\n"
            "assert t.equal(kept, expected), f'expected {expected.tolist()}, got {kept.tolist()}'\n"
            "\n"
            "# Case 2 — chain suppression: 0 → 1 → 2 all overlap each other\n"
            "scores2 = t.tensor([0.5, 0.9, 0.7])\n"
            "overlap2 = t.tensor([\n"
            "    [1.0, 0.8, 0.6],\n"
            "    [0.8, 1.0, 0.7],\n"
            "    [0.6, 0.7, 1.0],\n"
            "])\n"
            "kept2 = ex9_nms(scores2, overlap2, iou_threshold=0.5)\n"
            "# Highest is 1 → suppresses 0 and 2. Only 1 survives.\n"
            "assert t.equal(kept2, t.tensor([False, True, False])), f'chain suppression failed: {kept2.tolist()}'\n"
            "\n"
            "# Case 3 — no overlap at all → everything survives\n"
            "N = 6\n"
            "scores3 = t.rand(N)\n"
            "overlap3 = t.eye(N)\n"
            "kept3 = ex9_nms(scores3, overlap3, iou_threshold=0.5)\n"
            "assert kept3.all().item(), f'no-overlap case should keep everything, got {kept3.tolist()}'\n"
            "\n"
            "# Case 4 — threshold sensitivity. With threshold=0.9, less suppression → more survivors.\n"
            "scores4 = t.tensor([0.9, 0.8, 0.7, 0.6])\n"
            "overlap4 = t.tensor([\n"
            "    [1.0, 0.6, 0.0, 0.0],\n"
            "    [0.6, 1.0, 0.0, 0.0],\n"
            "    [0.0, 0.0, 1.0, 0.1],\n"
            "    [0.0, 0.0, 0.1, 1.0],\n"
            "])\n"
            "low_thresh = ex9_nms(scores4, overlap4, iou_threshold=0.5)\n"
            "high_thresh = ex9_nms(scores4, overlap4, iou_threshold=0.9)\n"
            "# At 0.5, the 0.6 overlap suppresses box 1; at 0.9 it doesn't.\n"
            "assert low_thresh.sum().item() == 3 and not low_thresh[1].item()\n"
            "assert high_thresh.sum().item() == 4 and high_thresh.all().item()"
        ),
        "solution_body": (
            "def ex9_nms(scores: Tensor, overlap: Tensor, iou_threshold: float) -> Tensor:\n"
            "    N = scores.shape[0]\n"
            "    alive = t.ones(N, dtype=t.bool)\n"
            "    kept = t.zeros(N, dtype=t.bool)\n"
            "    while alive.any().item():\n"
            "        masked_scores = scores.clone()\n"
            "        masked_scores[~alive] = float('-inf')\n"
            "        i = int(masked_scores.argmax())\n"
            "        kept[i] = True\n"
            "        alive[i] = False\n"
            "        suppress = (overlap[i] > iou_threshold) & alive\n"
            "        alive[suppress] = False\n"
            "    return kept"
        ),
        "solution_notes": (
            "**Three boolean-mask moves combined.**\n"
            "1. *Mask-then-argmax* — `scores[~alive] = -inf; argmax(scores)` picks the highest survivor in one shot.\n"
            "2. *Mask-from-a-row* — `(overlap[i] > thresh)` gives an `(N,)` bool flagging everything that overlaps box `i`. "
            "Combine with `& alive` so we don't 're-suppress' already-dead boxes (idempotent, but tidy).\n"
            "3. *Mask write-back* — `alive[suppress] = False` flips an arbitrary subset in one indexed assign.\n"
            "\n"
            "**Why this isn't the production NMS.** Real implementations sort once up front and iterate the sorted list, "
            "which avoids re-scanning `scores` each loop. They also operate on (x1, y1, x2, y2) boxes and compute IoU "
            "on the fly. The all-boolean version here is pedagogically clean: one mask per logical role."
        ),
        "extra_imports": [],
    },
]


def main() -> None:
    for spec in SPECS:
        path = emit_standalone(spec)
        print(f"wrote {path.relative_to(path.parents[3])}")


if __name__ == "__main__":
    main()
