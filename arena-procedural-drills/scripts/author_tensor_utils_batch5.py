#!/usr/bin/env python3
"""Author batch-5: 8 standalone single-exercise drills for misc tensor + training-utility atoms.

Each atom gets ONE exercise (ex1) under `prereqs_tensor_utils/<atom>/01-<slug>.ipynb`.

Atoms:
    detach-clone-snapshot
    cuda-empty-cache
    einops-reduce-min
    vector-normalize-keepdim
    where-clip-negative
    index-by-tensor
    topk-predictions
    matvec

Run:
    python arena-procedural-drills/scripts/author_tensor_utils_batch5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone  # noqa: E402

TOPIC = "prereqs_tensor_utils"


# =============================================================================
# detach-clone-snapshot
# =============================================================================
DETACH_RECAP = (
    "## detach + clone — graph-free deep copy refresher\n"
    "\n"
    "`x.detach()` returns a NEW tensor that shares storage with `x` but is "
    "detached from the autograd graph (its `requires_grad` is `False`). "
    "`x.clone()` returns a NEW tensor with its OWN storage but stays inside "
    "the graph (gradients still flow back through the clone op).\n"
    "\n"
    "Combine them — `x.detach().clone()` — to take a **graph-free deep "
    "copy**. The result has its own storage AND is severed from the graph: "
    "perfect for snapshotting hidden states across optimizer steps, logging "
    "activations without holding the graph, or stashing a target for a "
    "self-distillation step.\n"
    "\n"
    "**Why both.** `detach()` alone aliases the source — writing to the "
    "snapshot mutates the original. `clone()` alone keeps the autograd "
    "graph alive — you'd leak memory across training steps. Only the pair "
    "gives you the snapshot semantics you actually want."
)

DETACH_SPEC = {
    "atom_id": "detach-clone-snapshot",
    "subtopic": "PyTorch: detach + clone snapshot",
    "topic_folder": TOPIC,
    "atom_recap_md": DETACH_RECAP,
    "exercise_index": 1,
    "exercise_title": "snapshot a hidden state across an optimizer step",
    "slug": "snapshot-hidden-state",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["detach", "clone", "snapshot", "autograd"],
    "kcs": ["detach-strips-graph", "clone-copies-storage"],
    "lo": (
        "Apply `.detach().clone()` to snapshot a tensor so the snapshot has "
        "its own storage and is severed from the autograd graph."
    ),
    "prompt_body": (
        "Implement `ex1_snapshot(x)`. Given a leaf tensor `x` with "
        "`requires_grad=True`, return a snapshot that:\n\n"
        "1. Has its OWN storage (mutating the snapshot must not affect `x`).\n"
        "2. Has `requires_grad == False` (it is severed from the graph).\n"
        "3. Has the same values and shape as `x`.\n\n"
        "Use `x.detach().clone()`. Order matters for readability — "
        "`detach` first severs the graph, `clone` second copies storage. "
        "(The opposite order works too, but `detach().clone()` is the "
        "idiomatic snapshot.)\n\n"
        "Input: any float tensor.\n"
        "Output: a graph-free deep copy."
    ),
    "stub": (
        "def ex1_snapshot(x: Tensor) -> Tensor:\n"
        '    """Return a graph-free deep copy of x."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "x = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "snap = ex1_snapshot(x)\n"
        "# 1. Values match.\n"
        "assert t.allclose(snap, x.detach()), f'values mismatch: {snap} vs {x}'\n"
        "# 2. Graph-free.\n"
        "assert snap.requires_grad is False, 'snapshot must have requires_grad=False'\n"
        "assert snap.grad_fn is None, f'snapshot must have no grad_fn, got {snap.grad_fn}'\n"
        "# 3. Separate storage — mutating snap must not touch x.\n"
        "assert snap.data_ptr() != x.data_ptr(), 'snapshot must NOT share storage with x'\n"
        "snap_copy_for_mut = snap.clone()\n"
        "snap_copy_for_mut[0] = 999.0\n"
        "# x must remain unchanged (we never wrote to snap itself in-place, but we still verify).\n"
        "assert x[0].item() == 1.0, f'source x was mutated: {x}'\n"
        "\n"
        "# Realistic test — snapshot mid-graph then keep training.\n"
        "w = t.randn(4, requires_grad=True)\n"
        "y = (w * 2.0).sum()\n"
        "y_snap = ex1_snapshot(w * 2.0)  # snapshot something INSIDE the graph\n"
        "assert y_snap.requires_grad is False\n"
        "assert y_snap.grad_fn is None\n"
        "# Backprop through the LIVE side still works.\n"
        "y.backward()\n"
        "assert w.grad is not None and t.allclose(w.grad, t.full((4,), 2.0))\n"
        "\n"
        "# Bigger shape.\n"
        "big = t.randn(8, 16, requires_grad=True)\n"
        "snap_big = ex1_snapshot(big)\n"
        "assert snap_big.shape == (8, 16)\n"
        "assert snap_big.requires_grad is False\n"
        "assert snap_big.data_ptr() != big.data_ptr()"
    ),
    "solution_body": (
        "def ex1_snapshot(x: Tensor) -> Tensor:\n"
        "    return x.detach().clone()"
    ),
    "solution_notes": (
        "**Why `.detach()` first.** Reading `detach().clone()` left-to-right "
        "is 'sever the graph, then copy storage' — which matches what you "
        "want semantically. `clone().detach()` is identical at runtime but "
        "reads backwards.\n\n"
        "**Why not `.data`.** `x.data` looks tempting but is a footgun: it "
        "aliases storage AND silently violates autograd's view tracking. "
        "Prefer `detach()` in modern code.\n\n"
        "**Common use cases.** Logging hidden states without keeping the "
        "graph alive; building a teacher target in self-distillation; "
        "stashing the last-step parameters for a EMA / Polyak average."
    ),
    "extra_imports": [],
}


# =============================================================================
# cuda-empty-cache
# =============================================================================
EMPTY_CACHE_RECAP = (
    "## torch.cuda.empty_cache — what it actually does\n"
    "\n"
    "PyTorch's CUDA allocator keeps a **block cache** of GPU memory that was "
    "allocated and then freed by Python but NOT returned to the driver. "
    "Future allocations reuse cached blocks (fast). `nvidia-smi` reports the "
    "cached memory as 'used' even though no live tensor occupies it.\n"
    "\n"
    "`torch.cuda.empty_cache()` releases ALL cached blocks back to the CUDA "
    "driver. After the call, `nvidia-smi` drops — but no live tensor is "
    "affected. The call does NOT free live tensors. It does NOT speed up "
    "training. It does NOT reduce peak memory. Its ONLY purpose is to make "
    "GPU memory visible to OTHER processes (or to `nvidia-smi`).\n"
    "\n"
    "**Mocking for CPU.** This drill is CPU-friendly: we replace "
    "`torch.cuda.empty_cache` with a mock so the function runs everywhere "
    "and we can assert it was called the expected number of times."
)

EMPTY_CACHE_SPEC = {
    "atom_id": "cuda-empty-cache",
    "subtopic": "PyTorch: torch.cuda.empty_cache",
    "topic_folder": TOPIC,
    "atom_recap_md": EMPTY_CACHE_RECAP,
    "exercise_index": 1,
    "exercise_title": "periodic cache release in an eval loop",
    "slug": "periodic-cache-release",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["cuda", "memory", "cache", "mock"],
    "kcs": ["empty-cache-semantics", "empty-cache-no-live-tensor-impact"],
    "lo": (
        "Apply `torch.cuda.empty_cache()` at a controlled cadence inside an "
        "eval loop, verifying via mock that the call count matches the "
        "intended cadence and that no live tensor is destroyed."
    ),
    "prompt_body": (
        "Implement `ex1_eval_loop_with_cache_release(batches, release_every)`. "
        "The standard 'free cached blocks every K batches' eval pattern:\n\n"
        "1. `batches` is a list of tensors (each is one eval batch).\n"
        "2. For each batch: compute a per-batch sum (`b.sum()`) and append "
        "the scalar to a running list of results.\n"
        "3. After every `release_every` batches (1-indexed: so if "
        "`release_every=3`, call after batch 3, 6, 9, …) call "
        "`torch.cuda.empty_cache()`.\n"
        "4. Return the list of per-batch sums (as a 1-D float tensor).\n\n"
        "The test patches `torch.cuda.empty_cache` with a mock so it works "
        "on CPU and counts call frequency.\n\n"
        "Input: `batches: list[Tensor]`, `release_every: int >= 1`.\n"
        "Output: 1-D float tensor of length `len(batches)`."
    ),
    "stub": (
        "def ex1_eval_loop_with_cache_release(batches, release_every: int) -> Tensor:\n"
        '    """Sum each batch and call torch.cuda.empty_cache every release_every batches."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from unittest.mock import patch\n"
        "\n"
        "# Build 7 batches of varied sizes.\n"
        "batches = [t.ones(3, 4), t.full((2, 2), 2.0), t.full((5,), 3.0),\n"
        "           t.ones(1), t.full((4,), 0.5), t.zeros(3, 3), t.full((2, 2), -1.0)]\n"
        "expected_sums = t.tensor([b.sum().item() for b in batches])\n"
        "\n"
        "with patch('torch.cuda.empty_cache') as mock_release:\n"
        "    out = ex1_eval_loop_with_cache_release(batches, release_every=3)\n"
        "    # Released after batch 3 and after batch 6 → 2 calls.\n"
        "    assert mock_release.call_count == 2, (\n"
        "        f'expected 2 empty_cache calls (after batch 3 and 6), got {mock_release.call_count}'\n"
        "    )\n"
        "\n"
        "assert out.shape == (7,), f'expected (7,), got {tuple(out.shape)}'\n"
        "assert t.allclose(out, expected_sums), f'sums wrong: {out} vs {expected_sums}'\n"
        "\n"
        "# release_every=1 → call after every batch.\n"
        "with patch('torch.cuda.empty_cache') as mock_every:\n"
        "    _ = ex1_eval_loop_with_cache_release(batches, release_every=1)\n"
        "    assert mock_every.call_count == 7, f'expected 7 calls, got {mock_every.call_count}'\n"
        "\n"
        "# release_every larger than batch count → 0 calls.\n"
        "with patch('torch.cuda.empty_cache') as mock_none:\n"
        "    _ = ex1_eval_loop_with_cache_release(batches, release_every=100)\n"
        "    assert mock_none.call_count == 0, f'expected 0 calls, got {mock_none.call_count}'\n"
        "\n"
        "# Live-tensor invariant — the input batches must not be mutated or freed.\n"
        "for i, b in enumerate(batches):\n"
        "    assert b.numel() > 0, f'batch {i} was destroyed by the loop'"
    ),
    "solution_body": (
        "def ex1_eval_loop_with_cache_release(batches, release_every: int) -> Tensor:\n"
        "    sums = []\n"
        "    for i, b in enumerate(batches, start=1):\n"
        "        sums.append(b.sum())\n"
        "        if i % release_every == 0:\n"
        "            t.cuda.empty_cache()\n"
        "    return t.stack(sums)"
    ),
    "solution_notes": (
        "**`empty_cache` does NOT free live tensors.** The test's "
        "'live-tensor invariant' check is the whole point — even after the "
        "cache is released, every batch in the input list is still a valid "
        "tensor with all its data.\n\n"
        "**When to actually call it.** Three legitimate reasons: (1) handing "
        "the GPU back to another process; (2) before measuring memory with "
        "`nvidia-smi`; (3) before a known-spiky allocation that needs a "
        "fresh contiguous block. NEVER as a 'magic fix' for OOM — if you "
        "are OOM the live tensors are too big, not the cache.\n\n"
        "**Why mock for testing.** The function is pure side-effect on a "
        "global allocator. Mocking lets us verify call cadence without "
        "needing real CUDA hardware in CI."
    ),
    "extra_imports": [],
}


# =============================================================================
# einops-reduce-min
# =============================================================================
REDUCE_MIN_RECAP = (
    "## einops.reduce with 'min' — quick refresher\n"
    "\n"
    "`reduce(x, '<in> -> <out>', 'min')` collapses the axes dropped on the "
    "right side, taking the elementwise **minimum** over each collapsed "
    "group. Same syntax as `'mean'` / `'sum'` / `'max'` — only the op "
    "changes.\n"
    "\n"
    "Common uses: per-channel **min** for a channel-floor normalization, "
    "per-row **min** to identify the worst feature in a batch, or "
    "windowed-min for a max-pool-style erosion. Composable with `(...)` "
    "decomposition: `reduce(x, 'b (h h2) (w w2) -> b h w', 'min', h2=2, "
    "w2=2)` is a 2×2 min-pool.\n"
    "\n"
    "**Min vs max symmetry.** Every property of `'max'` reductions has a "
    "twin: `min(-x) == -max(x)`, min-pooling is dilation's dual, and "
    "argmin gates the same flow patterns as argmax."
)

REDUCE_MIN_SPEC = {
    "atom_id": "einops-reduce-min",
    "subtopic": "Einops: Reduce with min",
    "topic_folder": TOPIC,
    "atom_recap_md": REDUCE_MIN_RECAP,
    "exercise_index": 1,
    "exercise_title": "per-channel min floor for a feature map",
    "slug": "per-channel-min-floor",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["min", "reduce", "channel-floor", "broadcasting"],
    "kcs": ["reduce-min-op", "reduce-axis-collapse"],
    "lo": (
        "Apply `einops.reduce(..., 'min')` to compute a per-channel "
        "spatial minimum across `(B, C, H, W)`, returning the resulting "
        "`(B, C)` floor that can be subtracted to zero-floor each map."
    ),
    "prompt_body": (
        "Implement `ex1_channel_min_floor(x)`. Given an activation tensor "
        "shaped `(B, C, H, W)`:\n\n"
        "1. Use `einops.reduce` with the `'min'` op to collapse the "
        "spatial axes `H` and `W`, leaving `(B, C)`.\n"
        "2. The result is the per-sample, per-channel **floor** — the "
        "smallest value across the spatial map.\n\n"
        "Input: `(B, C, H, W)` float tensor.\n"
        "Output: `(B, C)` float tensor containing the spatial-min of each "
        "channel.\n\n"
        "You must use `einops.reduce` (NOT `x.min()` / `x.amin()`)."
    ),
    "stub": (
        "def ex1_channel_min_floor(x: Tensor) -> Tensor:\n"
        '    """Return per-sample, per-channel spatial min via einops.reduce."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Hand-built sample: 1 batch, 2 channels, 2x2 spatial.\n"
        "x = t.tensor([[\n"
        "    [[3.0, 1.0], [4.0, 2.0]],   # channel 0 min = 1.0\n"
        "    [[-5.0, 0.0], [7.0, 6.0]],  # channel 1 min = -5.0\n"
        "]])\n"
        "out = ex1_channel_min_floor(x)\n"
        "assert out.shape == (1, 2), f'expected (1,2), got {tuple(out.shape)}'\n"
        "assert t.allclose(out, t.tensor([[1.0, -5.0]])), f'got {out}'\n"
        "\n"
        "# Bigger random tensor — match reference torch.amin.\n"
        "rng = t.Generator().manual_seed(7)\n"
        "big = t.randn(4, 8, 16, 16, generator=rng)\n"
        "out_big = ex1_channel_min_floor(big)\n"
        "ref = big.amin(dim=(-2, -1))  # reference using built-in\n"
        "assert out_big.shape == (4, 8)\n"
        "assert t.allclose(out_big, ref, atol=1e-6), 'einops.reduce min must match amin'\n"
        "\n"
        "# Edge: single-pixel spatial — min equals the value.\n"
        "tiny = t.tensor([[[[0.5]], [[-0.5]]]])  # (1,2,1,1)\n"
        "out_tiny = ex1_channel_min_floor(tiny)\n"
        "assert t.allclose(out_tiny, t.tensor([[0.5, -0.5]]))"
    ),
    "solution_body": (
        "def ex1_channel_min_floor(x: Tensor) -> Tensor:\n"
        "    return reduce(x, 'b c h w -> b c', 'min')"
    ),
    "solution_notes": (
        "**The pattern.** `'b c h w -> b c'` drops `h` and `w`, so `min` "
        "is taken across all H*W spatial positions of each (batch, channel) "
        "pair. This is the same shape contract as `x.amin(dim=(-2, -1))` "
        "but reads more declaratively — you can see which axes survive.\n\n"
        "**Use case — floor subtraction.** `floor = ex1_channel_min_floor(x)`; "
        "`normalized = x - floor[..., None, None]` zero-floors each "
        "channel of each sample so the smallest value becomes 0. This is "
        "how some attention-vis pipelines stabilize displays.\n\n"
        "**Why not `x.amin(dim=(-2, -1))`.** Both work. `einops.reduce` "
        "wins when the pattern is part of a larger pipeline whose other "
        "steps are already `rearrange` / `repeat` — keeping a uniform "
        "vocabulary makes the code readable."
    ),
    "extra_imports": [],
}


# =============================================================================
# vector-normalize-keepdim
# =============================================================================
NORMALIZE_RECAP = (
    "## vector normalize with keepdim — quick refresher\n"
    "\n"
    "Unit-normalizing along the last axis is `x / x.norm(dim=-1, "
    "keepdim=True)`. The `keepdim=True` is the load-bearing part: without "
    "it the norm tensor loses the last axis (`(B,)` instead of `(B, 1)`) "
    "and the divide either fails or broadcasts the wrong way.\n"
    "\n"
    "**Why keepdim.** Broadcast rules align trailing dims. If `x` is "
    "`(B, D)` and the divisor is `(B,)`, broadcasting tries to align "
    "`(B,)` with the LAST axis of `x` (size `D`) — wrong. With "
    "`keepdim=True` the divisor is `(B, 1)`, which aligns with the "
    "trailing `D` and broadcasts across it.\n"
    "\n"
    "Use it for cosine-similarity inputs, contrastive embeddings, RoPE "
    "rotation arrays, or any 'put each row on the unit sphere' move."
)

NORMALIZE_SPEC = {
    "atom_id": "vector-normalize-keepdim",
    "subtopic": "PyTorch: vector normalize keepdim",
    "topic_folder": TOPIC,
    "atom_recap_md": NORMALIZE_RECAP,
    "exercise_index": 1,
    "exercise_title": "row-wise L2 normalize a batch of embeddings",
    "slug": "row-wise-l2-normalize",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["normalize", "keepdim", "broadcasting", "embeddings"],
    "kcs": ["norm-with-keepdim", "broadcast-divide"],
    "lo": (
        "Apply `x.norm(dim=-1, keepdim=True)` to L2-normalize every row of "
        "a batch of embeddings so each row lies on the unit sphere."
    ),
    "prompt_body": (
        "Implement `ex1_row_normalize(x)`. Given an embedding batch "
        "shaped `(B, D)`:\n\n"
        "1. Compute the per-row L2 norm with `x.norm(dim=-1, keepdim=True)` "
        "→ shape `(B, 1)`.\n"
        "2. Divide `x` by that norm; broadcasting expands `(B, 1)` over "
        "the `D` trailing dimension.\n"
        "3. Return the normalized tensor, same shape as `x`.\n\n"
        "Edge case: if a row is the zero vector, `0 / 0` produces `nan`. "
        "The test does NOT pass zero rows — but be aware that real pipelines "
        "add a small `eps` (e.g. `x.norm(...).clamp(min=1e-12)`) to handle "
        "the case.\n\n"
        "Input: `(B, D)` float tensor.\n"
        "Output: `(B, D)` float tensor; every row has L2 norm `≈ 1.0`."
    ),
    "stub": (
        "def ex1_row_normalize(x: Tensor) -> Tensor:\n"
        '    """L2-normalize each row of x with keepdim broadcasting."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "x = t.tensor([\n"
        "    [3.0, 4.0],         # norm 5, normalized = (0.6, 0.8)\n"
        "    [0.0, 1.0],         # already unit\n"
        "    [-1.0, 0.0],        # already unit\n"
        "    [2.0, 2.0, 1.0],    # different D — handled separately below\n"
        "][:3])\n"
        "out = ex1_row_normalize(x)\n"
        "assert out.shape == x.shape\n"
        "expected = t.tensor([\n"
        "    [0.6, 0.8],\n"
        "    [0.0, 1.0],\n"
        "    [-1.0, 0.0],\n"
        "])\n"
        "assert t.allclose(out, expected, atol=1e-6), f'got {out}'\n"
        "# Each row has unit norm.\n"
        "row_norms = out.norm(dim=-1)\n"
        "assert t.allclose(row_norms, t.ones(3), atol=1e-6), f'row norms {row_norms}'\n"
        "\n"
        "# Larger random batch — every row must end with L2 norm ~1.\n"
        "rng = t.Generator().manual_seed(2)\n"
        "big = t.randn(64, 128, generator=rng)\n"
        "big_out = ex1_row_normalize(big)\n"
        "assert big_out.shape == (64, 128)\n"
        "assert t.allclose(big_out.norm(dim=-1), t.ones(64), atol=1e-5)\n"
        "\n"
        "# Cosine similarity between row 0 and itself must be 1.0 after normalize.\n"
        "v = t.tensor([[1.0, 2.0, 3.0, 4.0]])\n"
        "v_n = ex1_row_normalize(v)\n"
        "cos = (v_n * v_n).sum(dim=-1)\n"
        "assert t.allclose(cos, t.tensor([1.0]), atol=1e-6)"
    ),
    "solution_body": (
        "def ex1_row_normalize(x: Tensor) -> Tensor:\n"
        "    return x / x.norm(dim=-1, keepdim=True)"
    ),
    "solution_notes": (
        "**`keepdim=True` is the only interesting line.** Drop it and you "
        "get a shape error (or worse, a silent wrong-axis broadcast). The "
        "rule: when you want the reduction result to **broadcast back over "
        "the reduced axis**, keep the dim.\n\n"
        "**Why not `F.normalize`.** `torch.nn.functional.normalize(x, "
        "dim=-1)` does exactly this AND adds an `eps` for numerical safety. "
        "In production prefer it. The manual version is the drill because "
        "it makes the `keepdim` mechanic visible.\n\n"
        "**Cosine-similarity setup.** After row-normalize, "
        "`a @ b.transpose(0, 1)` gives the full pairwise cosine matrix. "
        "This is the entire setup for contrastive losses (CLIP, SimCLR)."
    ),
    "extra_imports": [],
}


# =============================================================================
# where-clip-negative
# =============================================================================
WHERE_RECAP = (
    "## torch.where for conditional selection — quick refresher\n"
    "\n"
    "`torch.where(cond, a, b)` returns a tensor whose entries come from "
    "`a` where `cond` is true and from `b` where it's false. All three "
    "tensors broadcast against each other.\n"
    "\n"
    "**ReLU via where.** `torch.where(x < 0, torch.zeros_like(x), x)` is "
    "exactly ReLU — clip negatives to zero, pass positives through. The "
    "same pattern with `x > 6` and `6.0` gives ReLU6; with custom values "
    "it gives clamped activations.\n"
    "\n"
    "**Why prefer `where` over masked-assign.** `out = x.clone(); out[x<0] "
    "= 0` works but breaks autograd in subtle ways (in-place on a clone "
    "still has detach-y semantics). `torch.where` is differentiable and "
    "composable — gradients flow through both branches based on `cond`."
)

WHERE_SPEC = {
    "atom_id": "where-clip-negative",
    "subtopic": "PyTorch: where to clip negative",
    "topic_folder": TOPIC,
    "atom_recap_md": WHERE_RECAP,
    "exercise_index": 1,
    "exercise_title": "relu via torch.where",
    "slug": "relu-via-where",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["where", "relu", "conditional", "broadcasting"],
    "kcs": ["where-tertiary-op", "where-zeros-like-pattern"],
    "lo": (
        "Apply `torch.where(x < 0, zeros, x)` to implement ReLU as a "
        "branching selection over the input tensor."
    ),
    "prompt_body": (
        "Implement `ex1_relu_via_where(x)`. ReLU is defined as `max(0, "
        "x)` elementwise. Express it as a `torch.where` call:\n\n"
        "1. Build the condition `x < 0` (a boolean tensor with the same "
        "shape as `x`).\n"
        "2. Use `torch.where(cond, zeros_like(x), x)`: where `cond` is "
        "true (i.e. `x` is negative), pick `0.0`; otherwise keep `x`.\n"
        "3. Return the result.\n\n"
        "Input: any float tensor.\n"
        "Output: same shape as input; negatives are zeroed, non-negatives "
        "pass through.\n\n"
        "You must use `torch.where` (NOT `x.clamp(min=0)` / `F.relu`)."
    ),
    "stub": (
        "def ex1_relu_via_where(x: Tensor) -> Tensor:\n"
        '    """ReLU implemented via torch.where(x < 0, 0, x)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "x = t.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])\n"
        "out = ex1_relu_via_where(x)\n"
        "expected = t.tensor([0.0, 0.0, 0.0, 1.0, 2.0])\n"
        "assert t.allclose(out, expected), f'got {out}'\n"
        "assert out.shape == x.shape\n"
        "\n"
        "# 2-D input — must broadcast cleanly.\n"
        "x2 = t.tensor([[-1.0, 0.0, 0.5], [3.0, -3.0, 2.0]])\n"
        "out2 = ex1_relu_via_where(x2)\n"
        "expected2 = t.tensor([[0.0, 0.0, 0.5], [3.0, 0.0, 2.0]])\n"
        "assert t.allclose(out2, expected2)\n"
        "\n"
        "# Big random tensor — must equal torch.relu reference.\n"
        "rng = t.Generator().manual_seed(13)\n"
        "big = t.randn(32, 64, generator=rng)\n"
        "big_out = ex1_relu_via_where(big)\n"
        "assert t.allclose(big_out, t.relu(big)), 'must match torch.relu'\n"
        "# All outputs non-negative.\n"
        "assert (big_out >= 0).all().item(), 'ReLU output must be non-negative'\n"
        "\n"
        "# Gradient flows through positive entries only.\n"
        "z = t.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)\n"
        "ex1_relu_via_where(z).sum().backward()\n"
        "expected_grad = t.tensor([0.0, 1.0, 0.0, 1.0])\n"
        "assert t.allclose(z.grad, expected_grad), f'grad wrong: {z.grad}'"
    ),
    "solution_body": (
        "def ex1_relu_via_where(x: Tensor) -> Tensor:\n"
        "    return t.where(x < 0, t.zeros_like(x), x)"
    ),
    "solution_notes": (
        "**Argument order.** `torch.where(cond, a, b)` is 'cond ? a : b' "
        "— think C ternary. Here `cond = x < 0` returns true for "
        "negatives, so the `a` branch (`zeros_like(x)`) handles negatives "
        "and the `b` branch (`x`) handles non-negatives.\n\n"
        "**Why `zeros_like(x)` not just `0`.** `torch.where` requires the "
        "two value branches to have a common dtype with `x`. Using "
        "`zeros_like(x)` is the safest: it matches `dtype`, `device`, and "
        "shape. A scalar `0` also works on modern PyTorch but is more "
        "fragile across dtypes.\n\n"
        "**Generalization.** Swap the condition and constants to build "
        "any clipped activation: `where(x > 6, 6.0, x)` after ReLU = "
        "ReLU6; `where(x < lo, lo, where(x > hi, hi, x))` = hard-clamp."
    ),
    "extra_imports": [],
}


# =============================================================================
# index-by-tensor
# =============================================================================
INDEX_RECAP = (
    "## advanced indexing: index a tensor BY a tensor — quick refresher\n"
    "\n"
    "Slice indexing: `x[1:4]` produces a view, shape derived from the "
    "slice. Tensor indexing: `x[idx]` where `idx` is a LongTensor "
    "produces a **gather**, and the output shape becomes `idx.shape + "
    "x.shape[1:]`.\n"
    "\n"
    "**Key rule.** When you index a `(N, D)` tensor with a `(K,)` index, "
    "you get `(K, D)`. When you index with a `(K, L)` index, you get "
    "`(K, L, D)`. The index tensor's shape **becomes** the leading "
    "shape of the output — you're broadcasting the gather over an "
    "arbitrary index layout.\n"
    "\n"
    "Use it for: embedding lookups (`embed_table[token_ids]`), gathering "
    "minibatch examples by id, scattering predictions back to "
    "original-data order."
)

INDEX_SPEC = {
    "atom_id": "index-by-tensor",
    "subtopic": "PyTorch: index by tensor",
    "topic_folder": TOPIC,
    "atom_recap_md": INDEX_RECAP,
    "exercise_index": 1,
    "exercise_title": "embedding lookup with 2d index tensor",
    "slug": "embedding-lookup-2d-index",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["advanced-indexing", "gather", "embedding", "lookup"],
    "kcs": ["index-by-long-tensor", "advanced-indexing-shape-rule"],
    "lo": (
        "Apply `embed[idx]` advanced indexing where `idx` is a 2-D "
        "LongTensor of token ids, returning the corresponding "
        "`(B, T, D)` embedding lookup."
    ),
    "prompt_body": (
        "Implement `ex1_embedding_lookup(embed, idx)`. The canonical "
        "embedding-table lookup:\n\n"
        "1. `embed` is a `(V, D)` embedding table — `V` is vocab size, "
        "`D` is embedding dim.\n"
        "2. `idx` is a `(B, T)` LongTensor of token ids, each in "
        "`[0, V)`.\n"
        "3. Use advanced indexing `embed[idx]` to produce a `(B, T, D)` "
        "tensor where `out[b, t] == embed[idx[b, t]]`.\n\n"
        "The index tensor's shape becomes the LEADING shape of the "
        "output. Do not loop — use the single-expression `embed[idx]` "
        "form.\n\n"
        "Input: `embed: (V, D) Tensor`, `idx: (B, T) LongTensor`.\n"
        "Output: `(B, T, D) Tensor`."
    ),
    "stub": (
        "def ex1_embedding_lookup(embed: Tensor, idx: Tensor) -> Tensor:\n"
        '    """Look up idx in embed: out[b,t] = embed[idx[b,t]]."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Tiny vocab, dim 3.\n"
        "embed = t.tensor([\n"
        "    [0.0, 0.0, 0.0],   # token 0\n"
        "    [1.0, 1.0, 1.0],   # token 1\n"
        "    [2.0, 2.0, 2.0],   # token 2\n"
        "    [3.0, 3.0, 3.0],   # token 3\n"
        "])  # (4, 3)\n"
        "idx = t.tensor([\n"
        "    [0, 1, 2],\n"
        "    [3, 0, 1],\n"
        "], dtype=t.long)  # (2, 3)\n"
        "out = ex1_embedding_lookup(embed, idx)\n"
        "assert out.shape == (2, 3, 3), f'expected (2,3,3), got {tuple(out.shape)}'\n"
        "expected = t.tensor([\n"
        "    [[0.0,0.0,0.0], [1.0,1.0,1.0], [2.0,2.0,2.0]],\n"
        "    [[3.0,3.0,3.0], [0.0,0.0,0.0], [1.0,1.0,1.0]],\n"
        "])\n"
        "assert t.allclose(out, expected), f'got {out}'\n"
        "\n"
        "# Larger realistic shape.\n"
        "V, D, B, T = 100, 16, 4, 12\n"
        "rng = t.Generator().manual_seed(5)\n"
        "big_embed = t.randn(V, D, generator=rng)\n"
        "big_idx = t.randint(0, V, (B, T), generator=rng)\n"
        "big_out = ex1_embedding_lookup(big_embed, big_idx)\n"
        "assert big_out.shape == (B, T, D)\n"
        "# Spot check: out[0, 0] must equal embed[idx[0, 0]].\n"
        "assert t.allclose(big_out[0, 0], big_embed[big_idx[0, 0].item()])\n"
        "assert t.allclose(big_out[3, 7], big_embed[big_idx[3, 7].item()])\n"
        "\n"
        "# 1-D index works too (different leading shape).\n"
        "idx_1d = t.tensor([2, 0, 3, 1, 2], dtype=t.long)\n"
        "out_1d = ex1_embedding_lookup(embed, idx_1d)\n"
        "assert out_1d.shape == (5, 3)\n"
        "assert t.allclose(out_1d[0], embed[2])\n"
        "assert t.allclose(out_1d[3], embed[1])"
    ),
    "solution_body": (
        "def ex1_embedding_lookup(embed: Tensor, idx: Tensor) -> Tensor:\n"
        "    return embed[idx]"
    ),
    "solution_notes": (
        "**The whole skill is the one-liner.** Once you've internalized "
        "the shape rule — `idx.shape + embed.shape[1:]` — the syntax is "
        "trivial. The drill is teaching the rule, not the syntax.\n\n"
        "**Why this is NOT a slice.** Slices `embed[1:4]` produce views "
        "with no copy. Advanced indexing with a tensor produces a NEW "
        "tensor — gradients flow through the gather op, and the result "
        "does not alias `embed`'s storage.\n\n"
        "**Why this is the foundation of `nn.Embedding`.** `nn.Embedding` "
        "is essentially a learnable `(V, D)` parameter and an "
        "`embed[idx]` forward. No magic — just advanced indexing with "
        "the right grads."
    ),
    "extra_imports": [],
}


# =============================================================================
# topk-predictions
# =============================================================================
TOPK_RECAP = (
    "## logits.topk for top-k accuracy — quick refresher\n"
    "\n"
    "`logits.topk(k, dim=-1)` returns a named tuple `(values, indices)` "
    "where `values` are the `k` largest entries along `dim` and "
    "`indices` are their positions. For classification logits "
    "`(B, num_classes)`, `topk(5, dim=-1).indices` gives the `(B, 5)` "
    "tensor of the model's top-5 predicted class ids per sample.\n"
    "\n"
    "**Top-k accuracy.** A prediction is 'top-5 correct' if the true "
    "label appears anywhere in the top-5 predicted ids: "
    "`(topk_indices == label.unsqueeze(-1)).any(dim=-1)`. Top-1 "
    "accuracy is the special case `k=1`.\n"
    "\n"
    "ImageNet and many other benchmarks report top-1 AND top-5 because "
    "the long tail of classes is genuinely ambiguous — getting it in "
    "the top 5 is a useful signal even when the argmax is wrong."
)

TOPK_SPEC = {
    "atom_id": "topk-predictions",
    "subtopic": "Eval: topk predictions",
    "topic_folder": TOPIC,
    "atom_recap_md": TOPK_RECAP,
    "exercise_index": 1,
    "exercise_title": "top-5 accuracy on a batch of logits",
    "slug": "top5-accuracy",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["topk", "accuracy", "eval", "classification"],
    "kcs": ["topk-returns-values-and-indices", "topk-membership-test"],
    "lo": (
        "Apply `logits.topk(k=5, dim=-1).indices` then `any` over the "
        "k axis to compute top-5 classification accuracy from a batch "
        "of logits and ground-truth labels."
    ),
    "prompt_body": (
        "Implement `ex1_top5_accuracy(logits, labels)`. The standard "
        "ImageNet-style top-5 evaluation:\n\n"
        "1. `logits` has shape `(B, num_classes)`.\n"
        "2. `labels` has shape `(B,)` and dtype `long`.\n"
        "3. Use `logits.topk(k=5, dim=-1)` — call the result `topk_out`. "
        "Use `topk_out.indices` (shape `(B, 5)`) for the predicted "
        "class ids.\n"
        "4. A sample is correct if its true label appears in its top-5 "
        "indices. Use broadcasting: `(topk_out.indices == "
        "labels.unsqueeze(-1)).any(dim=-1)`.\n"
        "5. Return the mean of that boolean tensor (as a float scalar "
        "tensor) — the top-5 accuracy in `[0, 1]`.\n\n"
        "Input: `logits: (B, C)`, `labels: (B,) long`.\n"
        "Output: scalar tensor, top-5 accuracy."
    ),
    "stub": (
        "def ex1_top5_accuracy(logits: Tensor, labels: Tensor) -> Tensor:\n"
        '    """Return top-5 accuracy as a scalar float tensor in [0, 1]."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# 4 samples, 10 classes. Hand-build logits so we know the top-5.\n"
        "logits = t.tensor([\n"
        "    # sample 0: top-5 indices are [9, 8, 7, 6, 5] (logits descending)\n"
        "    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],\n"
        "    # sample 1: same logits — top-5 [9, 8, 7, 6, 5]\n"
        "    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],\n"
        "    # sample 2: reversed — top-5 [0, 1, 2, 3, 4]\n"
        "    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],\n"
        "    # sample 3: same reversed — top-5 [0, 1, 2, 3, 4]\n"
        "    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],\n"
        "])\n"
        "labels = t.tensor([9, 0, 0, 9], dtype=t.long)\n"
        "# Sample 0 label 9 IS in [9,8,7,6,5] → correct\n"
        "# Sample 1 label 0 is NOT in [9,8,7,6,5] → wrong\n"
        "# Sample 2 label 0 IS in [0,1,2,3,4] → correct\n"
        "# Sample 3 label 9 is NOT in [0,1,2,3,4] → wrong\n"
        "# accuracy = 2/4 = 0.5\n"
        "acc = ex1_top5_accuracy(logits, labels)\n"
        "assert isinstance(acc, Tensor), f'expected Tensor, got {type(acc)}'\n"
        "assert acc.shape == () or acc.numel() == 1, f'expected scalar, got {acc.shape}'\n"
        "assert abs(acc.item() - 0.5) < 1e-6, f'expected 0.5, got {acc.item()}'\n"
        "\n"
        "# All-correct case.\n"
        "labels_all = t.tensor([9, 7, 0, 2], dtype=t.long)\n"
        "# Sample 0 label 9 ✓, Sample 1 label 7 ✓, Sample 2 label 0 ✓, Sample 3 label 2 ✓\n"
        "acc_all = ex1_top5_accuracy(logits, labels_all)\n"
        "assert abs(acc_all.item() - 1.0) < 1e-6, f'expected 1.0, got {acc_all.item()}'\n"
        "\n"
        "# Realistic-shape smoke test.\n"
        "rng = t.Generator().manual_seed(11)\n"
        "big_logits = t.randn(32, 1000, generator=rng)\n"
        "big_labels = t.randint(0, 1000, (32,), generator=rng)\n"
        "big_acc = ex1_top5_accuracy(big_logits, big_labels)\n"
        "assert 0.0 <= big_acc.item() <= 1.0\n"
        "# Compare against a reference implementation.\n"
        "ref_top5 = big_logits.topk(5, dim=-1).indices\n"
        "ref_acc = (ref_top5 == big_labels.unsqueeze(-1)).any(dim=-1).float().mean()\n"
        "assert abs(big_acc.item() - ref_acc.item()) < 1e-6"
    ),
    "solution_body": (
        "def ex1_top5_accuracy(logits: Tensor, labels: Tensor) -> Tensor:\n"
        "    topk_out = logits.topk(k=5, dim=-1)\n"
        "    correct = (topk_out.indices == labels.unsqueeze(-1)).any(dim=-1)\n"
        "    return correct.float().mean()"
    ),
    "solution_notes": (
        "**The named-tuple return.** `logits.topk(5, dim=-1)` returns "
        "`torch.return_types.topk(values=..., indices=...)`. Always "
        "access with `.values` / `.indices`, never positional indexing "
        "— the named form is self-documenting.\n\n"
        "**`labels.unsqueeze(-1)` is the broadcast trick.** Labels are "
        "`(B,)` and top-k indices are `(B, 5)`. Adding the trailing "
        "size-1 axis to labels (`(B, 1)`) lets them broadcast across "
        "the k=5 dimension for the equality test.\n\n"
        "**Why `.float().mean()`.** The `correct` tensor is `bool`. "
        "`mean` on a bool tensor is undefined in many PyTorch versions; "
        "cast to float first so the mean is the fraction of correct "
        "predictions."
    ),
    "extra_imports": [],
}


# =============================================================================
# matvec
# =============================================================================
MATVEC_RECAP = (
    "## matrix-vector product — quick refresher\n"
    "\n"
    "`(M, N) @ (N,)` returns a **vector** of shape `(M,)`. This is the "
    "honest matrix-vector product — no batching, no broadcasting, just "
    "linear algebra.\n"
    "\n"
    "**Vector vs column-matrix.** `(M, N) @ (N, 1)` returns a "
    "**matrix** of shape `(M, 1)`. Two extra characters, totally "
    "different output rank. A `(M, 1)` matrix is a 2-D tensor — you "
    "need an extra `.squeeze(-1)` (or `[:, 0]`) to get back to a "
    "`(M,)` vector.\n"
    "\n"
    "Use the `(M,)` form when the result feeds a 1-D operation "
    "(softmax over classes, a 1-D loss); use the `(M, 1)` form when "
    "you want a column-vector for concatenation with other matrices."
)

MATVEC_SPEC = {
    "atom_id": "matvec",
    "subtopic": "PyTorch: matrix-vector product",
    "topic_folder": TOPIC,
    "atom_recap_md": MATVEC_RECAP,
    "exercise_index": 1,
    "exercise_title": "linear layer forward as matvec",
    "slug": "linear-layer-forward-matvec",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["matvec", "linear-layer", "shape-discipline", "matmul"],
    "kcs": ["matvec-output-is-1d", "vector-vs-column-distinction"],
    "lo": (
        "Apply `W @ x` where `W: (M, N)` and `x: (N,)` to compute a "
        "single linear-layer forward as a true `(M,)` vector (not a "
        "`(M, 1)` column matrix)."
    ),
    "prompt_body": (
        "Implement `ex1_linear_forward(W, x, b)`. A single-sample "
        "linear layer forward:\n\n"
        "1. `W` has shape `(M, N)` — the weight matrix.\n"
        "2. `x` has shape `(N,)` — the input feature vector.\n"
        "3. `b` has shape `(M,)` — the bias vector.\n"
        "4. Return `W @ x + b` — must be shape `(M,)`, NOT `(M, 1)`.\n\n"
        "Do NOT use `x.unsqueeze(-1)` to make it a column. Pass the "
        "1-D `x` directly to `@` — PyTorch returns a 1-D result.\n\n"
        "Input: `W: (M, N)`, `x: (N,)`, `b: (M,)`.\n"
        "Output: `(M,)` float tensor."
    ),
    "stub": (
        "def ex1_linear_forward(W: Tensor, x: Tensor, b: Tensor) -> Tensor:\n"
        '    """Compute W @ x + b for vector x; output must be a 1-D vector."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Hand-built: W maps 3 inputs to 2 outputs.\n"
        "W = t.tensor([\n"
        "    [1.0, 2.0, 3.0],\n"
        "    [4.0, 5.0, 6.0],\n"
        "])  # (2, 3)\n"
        "x = t.tensor([1.0, 1.0, 1.0])  # (3,)\n"
        "b = t.tensor([10.0, 20.0])     # (2,)\n"
        "# W @ x = [1+2+3, 4+5+6] = [6, 15]; + b = [16, 35]\n"
        "out = ex1_linear_forward(W, x, b)\n"
        "assert out.ndim == 1, f'output must be 1-D vector, got ndim={out.ndim} shape={tuple(out.shape)}'\n"
        "assert out.shape == (2,), f'expected (2,), got {tuple(out.shape)}'\n"
        "assert t.allclose(out, t.tensor([16.0, 35.0])), f'got {out}'\n"
        "\n"
        "# Larger realistic shape.\n"
        "M, N = 64, 128\n"
        "rng = t.Generator().manual_seed(17)\n"
        "Wb = t.randn(M, N, generator=rng)\n"
        "xb = t.randn(N, generator=rng)\n"
        "bb = t.randn(M, generator=rng)\n"
        "out_b = ex1_linear_forward(Wb, xb, bb)\n"
        "assert out_b.ndim == 1, f'shape contract violated: ndim={out_b.ndim}'\n"
        "assert out_b.shape == (M,)\n"
        "# Reference via F.linear (which expects (M,N) weights and produces (M,) for 1-D input).\n"
        "import torch.nn.functional as F_\n"
        "ref = F_.linear(xb, Wb, bb)\n"
        "assert t.allclose(out_b, ref, atol=1e-5), 'must match F.linear'\n"
        "\n"
        "# CRITICAL distinction — column-matrix form has a different shape.\n"
        "x_col = xb.unsqueeze(-1)  # (N, 1)\n"
        "col_result = Wb @ x_col   # (M, 1) — this is the WRONG output rank\n"
        "assert col_result.ndim == 2 and col_result.shape == (M, 1), 'sanity: column form returns matrix'\n"
        "assert col_result.shape != out_b.shape, 'matvec output must NOT be a column matrix'"
    ),
    "solution_body": (
        "def ex1_linear_forward(W: Tensor, x: Tensor, b: Tensor) -> Tensor:\n"
        "    return W @ x + b"
    ),
    "solution_notes": (
        "**The 1-D rule.** When the right-hand operand of `@` is 1-D, "
        "PyTorch treats it as a vector and returns a 1-D result. When "
        "you `unsqueeze(-1)` to make it `(N, 1)`, you've promoted it to "
        "a column matrix and the result is `(M, 1)` — same numbers, "
        "different rank.\n\n"
        "**Why rank matters downstream.** Many ops (softmax, cross-"
        "entropy, BatchNorm in 1-D mode) expect a specific rank. "
        "Carrying around `(M, 1)` when you meant `(M,)` causes silent "
        "broadcasting bugs that surface much later. Get the rank right "
        "at the matvec step.\n\n"
        "**Batched version.** For `(B, M, N) @ (B, N) -> (B, M)`, use "
        "`torch.einsum('bij,bj->bi', W, x)` or `(W @ x.unsqueeze(-1))."
        "squeeze(-1)`. Plain `@` also works thanks to batched matmul "
        "rules, but einsum makes the contract explicit."
    ),
    "extra_imports": [],
}


# =============================================================================
# Drive it
# =============================================================================
SPECS = [
    DETACH_SPEC,
    EMPTY_CACHE_SPEC,
    REDUCE_MIN_SPEC,
    NORMALIZE_SPEC,
    WHERE_SPEC,
    INDEX_SPEC,
    TOPK_SPEC,
    MATVEC_SPEC,
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
