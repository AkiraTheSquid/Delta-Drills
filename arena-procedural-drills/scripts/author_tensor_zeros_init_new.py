#!/usr/bin/env python3
"""Author Colab-native standalones (ex6-ex8) for atom `tensor-zeros-init`.

Each exercise exercises something flashcards cannot deliver: a visualization,
a multi-step debug pipeline, or an integrative ML-adjacent buffer pattern.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

ATOM_ID = "tensor-zeros-init"
SUBTOPIC = "Numpy: Core array literacy"
TOPIC = "prereqs_numpy"

RECAP = (
    "## torch zero-init — quick refresher\n"
    "\n"
    "**The allocate-then-scatter pattern.** Pre-allocate a buffer of the right "
    "`(shape, dtype, device)` with `t.zeros(...)`, then write per-element results "
    "into it via indexed assignment or `index_add_` / `scatter_add_`. This is faster "
    "and clearer than `list.append` + `t.stack`, and it's the canonical move for "
    "histograms, confusion matrices, depth buffers, and any per-ray accumulator.\n"
    "\n"
    "**Dtype matters.** Default is `float32`. Index buffers MUST be `t.long`. "
    "Counters should be `t.long` (or `t.int64`). Use `t.zeros_like(x)` when you "
    "want a fresh buffer that mirrors `x.shape + x.dtype + x.device` exactly."
)


SPECS = [
    # --------------------------------------------------------------- ex6
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "histogram via scatter into a zeros buffer",
        "slug": "histogram-via-scatter-into-zeros-buffer",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["histogram", "scatter-add", "visualization", "bincount"],
        "kcs": ["zeros-1d-shape", "zeros-dtype-control", "zeros-allocate-then-fill"],
        "lo": (
            "Allocate a `(n_bins,)` integer zero buffer and accumulate counts "
            "into it via `index_add_`; then plot the histogram with matplotlib."
        ),
        "prompt_body": (
            "Implement `ex6_histogram(samples, n_bins)`. Build a 1-D histogram "
            "the manual way:\n\n"
            "1. Allocate a `(n_bins,)` zero counter buffer with `dtype=t.long`.\n"
            "2. For each value in `samples` (which are already integer bin "
            "indices in `[0, n_bins)`), increment the matching counter by 1. "
            "Use `counts.index_add_(0, samples, t.ones_like(samples))` so the "
            "scatter happens in one shot.\n"
            "3. Return the counts tensor.\n\n"
            "Inputs:\n"
            "- `samples`: 1-D `t.long` tensor, values in `[0, n_bins)`.\n"
            "- `n_bins`: int.\n\n"
            "Output: `(n_bins,)` `t.long` tensor whose sum equals `len(samples)`.\n\n"
            "After the test passes, the visualization cell below the solution "
            "draws the histogram as a matplotlib bar chart so you can see the "
            "distribution your scatter produced."
        ),
        "stub": (
            "def ex6_histogram(samples: Tensor, n_bins: int) -> Tensor:\n"
            '    """Allocate (n_bins,) long zeros, scatter-add 1 per sample, return counts."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "samples = t.tensor([0, 0, 0, 2, 2, 5, 5, 5, 5, 9], dtype=t.long)\n"
            "counts = ex6_histogram(samples, n_bins=10)\n"
            "assert counts.shape == (10,), f'expected (10,), got {tuple(counts.shape)}'\n"
            "assert counts.dtype == t.long, f'expected dtype long, got {counts.dtype}'\n"
            "expected = t.tensor([3, 0, 2, 0, 0, 4, 0, 0, 0, 1], dtype=t.long)\n"
            "assert t.equal(counts, expected), f'value mismatch:\\n{counts}\\nvs\\n{expected}'\n"
            "assert counts.sum().item() == len(samples), 'sum of counts must equal len(samples)'\n"
            "# Edge case — empty samples must yield an all-zero buffer (not error).\n"
            "empty_counts = ex6_histogram(t.zeros(0, dtype=t.long), n_bins=4)\n"
            "assert empty_counts.shape == (4,) and empty_counts.sum().item() == 0, 'empty samples must yield all-zero counts'\n"
            "\n"
            "# --- Visualization (only runs if the assertions above passed) ---\n"
            "rng = t.Generator().manual_seed(42)\n"
            "big_samples = t.randint(0, 20, (500,), generator=rng)\n"
            "big_counts = ex6_histogram(big_samples, n_bins=20)\n"
            "fig, ax = plt.subplots(figsize=(8, 3))\n"
            "ax.bar(range(20), big_counts.tolist(), color='steelblue', edgecolor='black')\n"
            "ax.set_xlabel('bin index')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title(f'ex6 histogram — 500 samples into 20 bins (sum={big_counts.sum().item()})')\n"
            "ax.set_xticks(range(20))\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex6_histogram(samples: Tensor, n_bins: int) -> Tensor:\n"
            "    counts = t.zeros(n_bins, dtype=t.long)\n"
            "    counts.index_add_(0, samples, t.ones_like(samples))\n"
            "    return counts"
        ),
        "solution_notes": (
            "**Why `index_add_` and not a Python `for` loop?** `index_add_` runs "
            "the scatter as a single fused op — no Python overhead per element. "
            "For 500 samples it's noticeable; for 5M samples (one frame of a Ray "
            "Tracing accumulator) it's the difference between 50ms and 30s.\n\n"
            "**Why `dtype=t.long` for the counter?** Counts are integers. A "
            "`float32` counter quietly loses precision once counts exceed ~16M "
            "(the float32 mantissa runs out). `t.long` (int64) handles up to "
            "9.2e18 counts.\n\n"
            "**Alternative one-liner.** `t.bincount(samples, minlength=n_bins)` "
            "does the same thing and is even more idiomatic — but this exercise "
            "drills the allocate-then-scatter pattern explicitly so you see the "
            "machinery `bincount` hides."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # --------------------------------------------------------------- ex7
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "confusion matrix from (pred, true) pairs",
        "slug": "confusion-matrix-from-pred-true-pairs",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["confusion-matrix", "scatter-add", "visualization", "classification"],
        "kcs": ["zeros-multi-axis-shape", "zeros-dtype-control", "zeros-allocate-then-fill"],
        "lo": (
            "Allocate a `(C, C)` zero buffer and accumulate `(pred, true)` pairs "
            "into it via 2-D scatter; visualize the matrix as a heatmap."
        ),
        "prompt_body": (
            "Implement `ex7_confusion_matrix(preds, trues, n_classes)`. The "
            "canonical classification-debug pattern:\n\n"
            "1. Allocate a `(n_classes, n_classes)` zero matrix with `dtype=t.long`. "
            "Convention: row = predicted class, column = true class.\n"
            "2. For each `(p, y)` pair, increment `cm[p, y]` by 1.\n"
            "3. Return the matrix.\n\n"
            "Trick: a 2-D scatter is most cleanly done by flattening. Compute "
            "`flat_idx = preds * n_classes + trues`, allocate a flat `(n_classes "
            "* n_classes,)` buffer, scatter-add 1 per index, then `.view(n_classes, "
            "n_classes)`. This trains the allocate-then-reshape idiom you'll use "
            "again for occupancy grids and voxel volumes.\n\n"
            "Inputs:\n"
            "- `preds`: 1-D `t.long`, values in `[0, n_classes)`.\n"
            "- `trues`: 1-D `t.long`, same shape as `preds`.\n"
            "- `n_classes`: int.\n\n"
            "Output: `(n_classes, n_classes)` `t.long` tensor. Diagonal entries "
            "are the correct-prediction counts.\n\n"
            "The visualization below the solution renders the matrix as a "
            "matplotlib heatmap so you can read off the misclassification "
            "patterns visually."
        ),
        "stub": (
            "def ex7_confusion_matrix(preds: Tensor, trues: Tensor, n_classes: int) -> Tensor:\n"
            '    """Allocate (C, C) long zeros; scatter (pred, true) counts; return matrix."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# 3 classes, 8 predictions. 5 correct (diag), 3 wrong.\n"
            "preds = t.tensor([0, 1, 2, 0, 1, 2, 1, 2], dtype=t.long)\n"
            "trues = t.tensor([0, 1, 2, 0, 0, 1, 2, 0], dtype=t.long)\n"
            "cm = ex7_confusion_matrix(preds, trues, n_classes=3)\n"
            "assert cm.shape == (3, 3), f'expected (3,3), got {tuple(cm.shape)}'\n"
            "assert cm.dtype == t.long, f'expected dtype long, got {cm.dtype}'\n"
            "# Manual count: rows=pred, cols=true.\n"
            "#   pred=0,true=0: 2  pred=1,true=1: 1  pred=2,true=2: 1\n"
            "#   pred=1,true=0: 1  pred=1,true=2: 1  pred=2,true=0: 1  pred=2,true=1: 1\n"
            "expected = t.tensor([\n"
            "    [2, 0, 0],\n"
            "    [1, 1, 1],\n"
            "    [1, 1, 1],\n"
            "], dtype=t.long)\n"
            "assert t.equal(cm, expected), f'value mismatch:\\n{cm}\\nvs\\n{expected}'\n"
            "assert cm.sum().item() == len(preds), 'matrix total must equal n_samples'\n"
            "assert cm.diag().sum().item() == 4, f'expected 4 correct on diagonal, got {cm.diag().sum().item()}'\n"
            "\n"
            "# --- Heatmap visualization ---\n"
            "rng = t.Generator().manual_seed(7)\n"
            "n_classes = 5\n"
            "big_trues = t.randint(0, n_classes, (300,), generator=rng)\n"
            "# Simulate a noisy classifier: 70% correct, 30% random.\n"
            "noise_mask = t.rand(300, generator=rng) < 0.3\n"
            "big_preds = t.where(noise_mask, t.randint(0, n_classes, (300,), generator=rng), big_trues)\n"
            "big_cm = ex7_confusion_matrix(big_preds, big_trues, n_classes=n_classes)\n"
            "fig, ax = plt.subplots(figsize=(5, 4))\n"
            "im = ax.imshow(big_cm.numpy(), cmap='Blues')\n"
            "ax.set_xlabel('true class')\n"
            "ax.set_ylabel('predicted class')\n"
            "ax.set_title(f'ex7 confusion matrix (300 samples, ~70% acc)')\n"
            "ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))\n"
            "for i in range(n_classes):\n"
            "    for j in range(n_classes):\n"
            "        ax.text(j, i, str(big_cm[i, j].item()), ha='center', va='center',\n"
            "                color='white' if big_cm[i, j].item() > big_cm.max().item() / 2 else 'black')\n"
            "plt.colorbar(im, ax=ax)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex7_confusion_matrix(preds: Tensor, trues: Tensor, n_classes: int) -> Tensor:\n"
            "    flat_idx = preds * n_classes + trues\n"
            "    flat = t.zeros(n_classes * n_classes, dtype=t.long)\n"
            "    flat.index_add_(0, flat_idx, t.ones_like(flat_idx))\n"
            "    return flat.view(n_classes, n_classes)"
        ),
        "solution_notes": (
            "**The 2-D-via-flat trick.** PyTorch's `index_add_` only takes 1-D "
            "indices into a 1-D output, so a 2-D scatter is done by linearising "
            "`(row, col) → row * n_cols + col`, scattering into a flat buffer, "
            "then reshaping back. This is the SAME pattern used to splat "
            "fragments into a 2-D framebuffer in Ray Tracing.\n\n"
            "**Why diagonal sum = accuracy * N.** Every `(pred==true)` pair "
            "lands on the diagonal. `cm.diag().sum() / cm.sum()` is the "
            "accuracy.\n\n"
            "**The integrative load.** Three KCs at once: multi-axis allocation, "
            "long-dtype counter, and indexed scatter — Lohr et al. ITiCSE 2025 "
            "shows 3-KC exercises drop to ~40% solvability. Expect to look "
            "things up."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # --------------------------------------------------------------- ex8
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "z-buffer painter with per-step debug",
        "slug": "z-buffer-painter-with-per-step-debug",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["z-buffer", "depth-test", "ray-tracing", "multi-step-debug"],
        "kcs": ["zeros-1d-shape", "zeros-like-mirrors-input", "zeros-allocate-then-fill"],
        "lo": (
            "Combine `t.zeros` + `t.zeros_like` + indexed conditional write to "
            "build a per-pixel depth buffer; return the buffer history for debug."
        ),
        "prompt_body": (
            "Implement `ex8_zbuffer(num_pixels, objects)`. A miniature Ray "
            "Tracing z-buffer that you can step through:\n\n"
            "1. Allocate a `(num_pixels,)` float zero buffer named `z_buf`, then "
            "fill it with `+inf` (no object yet → infinitely far).\n"
            "2. Allocate a `(num_pixels,)` `t.long` zero buffer named `obj_id` "
            "(which object owns each pixel; 0 means 'none').\n"
            "3. For each `(name, pixel_idxs, depths)` in `objects`, perform a "
            "depth test: where `depths < z_buf[pixel_idxs]`, overwrite both "
            "`z_buf[pixel_idxs]` and `obj_id[pixel_idxs]`.\n"
            "4. Record a `(z_buf.clone(), obj_id.clone())` snapshot after each "
            "object so the caller can replay the painter.\n\n"
            "Inputs:\n"
            "- `num_pixels`: int.\n"
            "- `objects`: list of `(name: str, pixel_idxs: long Tensor (K,), depths: float Tensor (K,))`. "
            "`name` is for printing only; objects are numbered 1, 2, ... in input order.\n\n"
            "Output: a dict with keys `'z_buf'` (final), `'obj_id'` (final), and "
            "`'history'` (list of `(name, z_buf_clone, obj_id_clone)` after each step). "
            "Also print the per-step pixel-ownership count so the caller sees the "
            "painter evolve.\n\n"
            "> ⚠️ **Integrative exercise.** Combines 3 KCs (shape, zeros_like, "
            "indexed conditional write) plus a debug-introspection loop. Expect "
            "a step up vs Exercises 1-5."
        ),
        "stub": (
            "def ex8_zbuffer(num_pixels: int, objects: list) -> dict:\n"
            '    """Per-pixel depth buffer with per-object snapshots."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# 5 pixels. Object 1 covers pixels [0, 1, 2] at depth 5.0.\n"
            "# Object 2 covers pixels [1, 3] at depth 3.0 (closer at pixel 1 — overwrites).\n"
            "# Object 3 covers pixel [4] at depth 8.0.\n"
            "objects = [\n"
            "    ('back-wall', t.tensor([0, 1, 2], dtype=t.long), t.tensor([5.0, 5.0, 5.0])),\n"
            "    ('cube',      t.tensor([1, 3],     dtype=t.long), t.tensor([3.0, 3.0])),\n"
            "    ('far-light', t.tensor([4],         dtype=t.long), t.tensor([8.0])),\n"
            "]\n"
            "result = ex8_zbuffer(5, objects)\n"
            "assert set(result.keys()) >= {'z_buf', 'obj_id', 'history'}, f'missing keys: {result.keys()}'\n"
            "z = result['z_buf']\n"
            "oid = result['obj_id']\n"
            "assert z.shape == (5,) and z.dtype == t.float32, f'z_buf shape/dtype wrong: {z.shape}/{z.dtype}'\n"
            "assert oid.shape == (5,) and oid.dtype == t.long, f'obj_id shape/dtype wrong: {oid.shape}/{oid.dtype}'\n"
            "expected_z = t.tensor([5.0, 3.0, 5.0, 3.0, 8.0])\n"
            "expected_oid = t.tensor([1, 2, 1, 2, 3], dtype=t.long)\n"
            "assert t.allclose(z, expected_z), f'z_buf mismatch: {z} vs {expected_z}'\n"
            "assert t.equal(oid, expected_oid), f'obj_id mismatch: {oid} vs {expected_oid}'\n"
            "# History snapshots — must be deep copies, not aliases of the live buffer.\n"
            "assert len(result['history']) == 3, f'expected 3 history snapshots, got {len(result[\"history\"])}'\n"
            "name0, z0, o0 = result['history'][0]\n"
            "assert name0 == 'back-wall'\n"
            "# After step 0 only pixels 0,1,2 should be owned (== 1).\n"
            "assert t.equal(o0, t.tensor([1, 1, 1, 0, 0], dtype=t.long)), f'snapshot 0 obj_id wrong: {o0}'\n"
            "# Snapshot must be a clone — mutating live buffer mustn't change history.\n"
            "oid[0] = 99\n"
            "assert result['history'][0][2][0].item() == 1, 'history snapshot must be cloned, not aliased'\n"
            "\n"
            "# --- Visualization: replay the painter across snapshots ---\n"
            "fig, axes = plt.subplots(1, 3, figsize=(10, 2.4))\n"
            "for ax, (name, z_snap, o_snap) in zip(axes, result['history']):\n"
            "    im = ax.imshow(o_snap.numpy().reshape(1, -1), cmap='tab10', vmin=0, vmax=9, aspect='auto')\n"
            "    ax.set_title(f'after {name}')\n"
            "    ax.set_yticks([])\n"
            "    ax.set_xticks(range(5))\n"
            "    for i, val in enumerate(o_snap.tolist()):\n"
            "        ax.text(i, 0, str(val), ha='center', va='center', color='white', fontsize=12)\n"
            "fig.suptitle('ex8 z-buffer ownership over time (0 = empty)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex8_zbuffer(num_pixels: int, objects: list) -> dict:\n"
            "    z_buf = t.zeros(num_pixels)\n"
            "    z_buf.fill_(float('inf'))\n"
            "    obj_id = t.zeros(num_pixels, dtype=t.long)\n"
            "    history = []\n"
            "    for i, (name, pixel_idxs, depths) in enumerate(objects, start=1):\n"
            "        closer = depths < z_buf[pixel_idxs]\n"
            "        winning_pixels = pixel_idxs[closer]\n"
            "        winning_depths = depths[closer]\n"
            "        z_buf[winning_pixels] = winning_depths\n"
            "        obj_id[winning_pixels] = i\n"
            "        owned = (obj_id != 0).sum().item()\n"
            "        print(f'  step {i} ({name}): {owned}/{num_pixels} pixels owned')\n"
            "        history.append((name, z_buf.clone(), obj_id.clone()))\n"
            "    return {'z_buf': z_buf, 'obj_id': obj_id, 'history': history}"
        ),
        "solution_notes": (
            "**The depth-test pattern.** This is the integer-arithmetic heart "
            "of a Ray Tracing renderer: every closest-hit query is a per-pixel "
            "depth test against a sentinel-initialised buffer. Sentinel = `+inf` "
            "(so any real hit wins). Owner-id starts at 0 (none).\n\n"
            "**Why clone the history.** `z_buf` and `obj_id` are mutated in "
            "place by subsequent steps. If you store the live references "
            "instead of clones, every snapshot will end up identical to the "
            "final state — a classic alias bug. The test's mutation-after-the-"
            "fact assertion catches it.\n\n"
            "**Why the per-step print.** Multi-step pipelines fail silently in "
            "the middle. Logging `owned / num_pixels` after each object turns "
            "the loop into a self-narrating debug trace — essential when you "
            "later replace static `objects` with a real scene."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
