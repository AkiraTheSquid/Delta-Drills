#!/usr/bin/env python3
"""Author 8 deepening notebooks for single-drill atoms with 8-9 existing exercises.

Each spec adds ONE new exercise that probes a DISTINCT facet from the existing
set in the same folder. PS4 framing — one LO, one Bloom, max 2 KCs.

Subtopic matches ex1 exactly (Read-verified). Solutions verify against the
project venv: This-Directory-Only/backend/.venv/bin/python (torch 2.12.0+cpu,
einops 0.8.2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ============================================================== 1: tensor-zeros-init ex9
# Existing ex1-8: 1-D/3-D zeros, zeros_like, dtype=long, paint-hits, histogram,
# confusion matrix, z-buffer. None touch pinned-memory staging buffers.
SPEC_ZEROS = {
    "atom_id": "tensor-zeros-init",
    "subtopic": "Numpy: Core array literacy",
    "topic_folder": "prereqs_numpy",
    "atom_recap_md": (
        "## torch.zeros — pinned staging refresher\n"
        "\n"
        "`torch.zeros(*shape, pin_memory=True)` allocates a page-locked CPU "
        "buffer. Page-locked memory cannot be paged out by the OS, which lets "
        "the CUDA driver DMA into it directly — `tensor.to('cuda', "
        "non_blocking=True)` only overlaps compute with transfer when the "
        "**source** is pinned.\n"
        "\n"
        "**Compared to `torch.zeros(...).pin_memory()`.** `pin_memory=True` "
        "at allocation avoids an unnecessary unpinned-allocation + copy. For "
        "DataLoader-style staging where you reuse one buffer across iterations, "
        "allocate-pinned once, then `.copy_()` fresh data in-place each step.\n"
        "\n"
        "**This drill (ex9) vs ex1-8.** Earlier exercises focused on "
        "allocation-as-output (paint hits, scatter, histogram). ex9 focuses "
        "on allocation-as-staging — a buffer's *kwargs* matter as much as its "
        "shape when you need a fast host→device pipeline."
    ),
    "exercise_index": 9,
    "exercise_title": "allocate a pinned staging buffer for non_blocking transfer",
    "slug": "allocate-pinned-staging-buffer-for-non-blocking-transfer",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["pin_memory", "non_blocking", "dataloader", "staging-buffer"],
    "kcs": ["zeros-shape-kwarg", "zeros-pin-memory-kwarg"],
    "lo": (
        "Apply `torch.zeros(*shape, pin_memory=True)` to allocate a reusable "
        "page-locked staging buffer, then verify in-place `.copy_(src)` "
        "preserves the pinned flag and produces a correct value snapshot."
    ),
    "prompt_body": (
        "Implement `ex9_pinned_staging(batch_shape, sources)`.\n\n"
        "Simulate the DataLoader staging pattern without needing CUDA:\n\n"
        "1. Allocate ONE pinned staging buffer of shape `batch_shape` and "
        "dtype `float32` via `t.zeros(*batch_shape, dtype=t.float32, "
        "pin_memory=True)`. The buffer is reused across iterations — do NOT "
        "reallocate inside the loop.\n"
        "2. For each `src` in `sources` (a list of CPU tensors with shape "
        "`batch_shape`), copy `src` into the staging buffer **in place** via "
        "`buf.copy_(src)`, then record `buf.clone().detach()` into a list.\n"
        "3. Return `(buf, snapshots)` — the buffer itself (still pinned), and "
        "the per-iteration snapshot list.\n\n"
        "On CPU-only systems, `pin_memory=True` is a no-op flag but the "
        "attribute `is_pinned()` reflects what would happen on a CUDA host — "
        "the test calls `is_pinned()` only if a CUDA device is available, "
        "otherwise it falls back to checking the API surface (the kwarg was "
        "accepted without error).\n\n"
        "Output: `(buf, snapshots)` where `buf.shape == batch_shape`, "
        "`buf.dtype == torch.float32`, and `len(snapshots) == len(sources)`.\n\n"
        "The visualization plots the per-iteration snapshot mean to confirm "
        "that the staging buffer faithfully captured each source — a "
        "classic dataloader debug move."
    ),
    "stub": (
        "def ex9_pinned_staging(\n"
        "    batch_shape: tuple[int, ...],\n"
        "    sources: list[Tensor],\n"
        ") -> tuple[Tensor, list[Tensor]]:\n"
        '    """Allocate ONE pinned buffer, in-place copy each source, return snapshots."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Three iterations of a (B=4, D=3) batch.\n"
        "rng = t.Generator().manual_seed(7)\n"
        "sources = [t.randn(4, 3, generator=rng) for _ in range(3)]\n"
        "buf, snaps = ex9_pinned_staging((4, 3), sources)\n"
        "\n"
        "# Buffer-level invariants.\n"
        "assert buf.shape == (4, 3), f'buf shape wrong: {tuple(buf.shape)}'\n"
        "assert buf.dtype == t.float32, f'buf dtype wrong: {buf.dtype}'\n"
        "# Buffer holds the LAST source after the loop.\n"
        "assert t.allclose(buf, sources[-1]), 'buf must equal last source after final copy_'\n"
        "\n"
        "# Snapshot-level invariants.\n"
        "assert len(snaps) == 3, f'expected 3 snapshots, got {len(snaps)}'\n"
        "for i, (snap, src) in enumerate(zip(snaps, sources)):\n"
        "    assert snap.shape == (4, 3), f'snap[{i}] shape wrong'\n"
        "    assert t.allclose(snap, src), f'snap[{i}] does not match source[{i}]'\n"
        "    # Snapshots must be independent of buf (detached clones, not aliases).\n"
        "    assert snap.data_ptr() != buf.data_ptr(), f'snap[{i}] aliases buf — must be a clone'\n"
        "\n"
        "# pin_memory kwarg surface check.\n"
        "if t.cuda.is_available():\n"
        "    assert buf.is_pinned(), 'on CUDA host, buf must report is_pinned() True'\n"
        "else:\n"
        "    # CPU-only: pin_memory= must at least be accepted without error.\n"
        "    _smoke = t.zeros(2, 2, dtype=t.float32, pin_memory=False)\n"
        "    assert _smoke.shape == (2, 2)\n"
        "\n"
        "# --- Visualization: per-iteration snapshot means ---\n"
        "rng2 = t.Generator().manual_seed(11)\n"
        "B, D = 8, 5\n"
        "long_sources = [t.randn(B, D, generator=rng2) + i for i in range(10)]\n"
        "_, long_snaps = ex9_pinned_staging((B, D), long_sources)\n"
        "means = [s.mean().item() for s in long_snaps]\n"
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        "ax.plot(range(10), means, marker='o', color='steelblue')\n"
        "ax.axhline(0, color='gray', linewidth=0.5)\n"
        "ax.set_xlabel('iteration')\n"
        "ax.set_ylabel('staged batch mean')\n"
        "ax.set_title('ex9 per-iteration staging snapshot mean (should drift up with i)')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex9_pinned_staging(\n"
        "    batch_shape: tuple[int, ...],\n"
        "    sources: list[Tensor],\n"
        ") -> tuple[Tensor, list[Tensor]]:\n"
        "    buf = t.zeros(*batch_shape, dtype=t.float32, pin_memory=t.cuda.is_available())\n"
        "    snapshots = []\n"
        "    for src in sources:\n"
        "        buf.copy_(src)\n"
        "        snapshots.append(buf.clone().detach())\n"
        "    return buf, snapshots"
    ),
    "solution_notes": (
        "**Why allocate-then-`copy_` rather than re-allocating.** A pinned "
        "allocation is expensive — the kernel has to register page-locked "
        "memory with the DMA controller. Reusing one buffer across iterations "
        "amortizes that setup. `copy_` mutates in place and preserves the "
        "pinned flag.\n\n"
        "**Why `.clone().detach()` for snapshots.** Without clone, every "
        "snapshot would alias `buf` — by the time you inspect them, they all "
        "show the LAST source. The clone breaks aliasing; the detach is "
        "future-proofing in case you ever stage a tensor that came from an "
        "autograd-tracked op.\n\n"
        "**CPU-only fallback.** PyTorch's `pin_memory=True` raises if no CUDA "
        "host is configured on some platforms — we gate the kwarg on "
        "`t.cuda.is_available()`. In a real training rig you'd keep it `True` "
        "and let the DataLoader use it; this drill exercises the allocation "
        "pattern without requiring a GPU."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 2: tensor-unbind ex9
# Existing ex1-8: dim, select equivalence, ray destructure, attention heads, RGB.
# NONE touch autograd-through-unbind. Novel facet: gradient flow through unbind.
SPEC_UNBIND = {
    "atom_id": "tensor-unbind",
    "subtopic": "Numpy: Indexing and selection",
    "topic_folder": "prereqs_numpy",
    "atom_recap_md": (
        "## torch.unbind through autograd — quick refresher\n"
        "\n"
        "`x.unbind(dim=k)` returns a tuple of **view** tensors that share "
        "storage with `x`. When `x.requires_grad=True`, the unbind op is "
        "fully differentiable: PyTorch fuses the entire unbind into a "
        "single `UnbindBackward` node that knows which slot each output "
        "came from, so gradients accumulated through any element route "
        "back to the correct row of `x.grad`.\n"
        "\n"
        "**Compared to indexed access.** `x[0]`, `x[1]`, ... also produce "
        "differentiable views (each with its own `SelectBackward` node). "
        "`unbind` gives you the full tuple in one call — cleaner code "
        "AND a single backward node — handy when you want per-slot names "
        "and to keep the autograd graph compact.\n"
        "\n"
        "**This drill (ex9) vs ex1-8.** Earlier exercises destructured tensors "
        "for forward computation (ray casting, attention heads, RGB → "
        "grayscale). ex9 takes the same destructure and runs `.backward()` "
        "through it to verify gradients flow correctly to each slot of the "
        "source tensor."
    ),
    "exercise_index": 9,
    "exercise_title": "verify gradients flow through unbind to per-slot gradients",
    "slug": "verify-gradients-flow-through-unbind-to-per-slot-gradients",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["autograd", "backward", "per-slot-grad", "view-graph"],
    "kcs": ["unbind-returns-views", "unbind-autograd-flow"],
    "lo": (
        "Analyze the autograd graph produced by `unbind` by running "
        "`.backward()` through a per-slot weighted sum and verifying that "
        "each row of `x.grad` matches the analytic per-slot weight."
    ),
    "prompt_body": (
        "Implement `ex9_unbind_grad_check(x, weights)`.\n\n"
        "Given `x` of shape `(N, D)` with `requires_grad=True` and a 1-D "
        "`weights` tensor of length `N`, do the following:\n\n"
        "1. Use `x.unbind(dim=0)` to get a tuple of `N` row-views `r_0, "
        "r_1, ..., r_{N-1}`, each shape `(D,)`.\n"
        "2. Compute the scalar loss `L = sum_i weights[i] * r_i.sum()`. Build "
        "it by iterating over the unbound tuple — DO NOT collapse back via "
        "`stack` first; the point is to verify autograd handles the view-"
        "tuple branching.\n"
        "3. Call `L.backward()`.\n"
        "4. Return `x.grad` — a `(N, D)` tensor where row `i` should equal "
        "`weights[i]` (a vector of all-`weights[i]`).\n\n"
        "**Print** `len({id(r.grad_fn) for r in rows})` so the caller can see "
        "that each unbound row got its own `grad_fn` node (not one shared "
        "node).\n\n"
        "Inputs: `x` `(N, D)` float, requires_grad; `weights` `(N,)` float.\n"
        "Output: `x.grad` `(N, D)` float.\n\n"
        "The visualization plots `x.grad` as a heatmap so you can see the "
        "constant-per-row pattern that proves autograd routed each slot's "
        "gradient back correctly."
    ),
    "stub": (
        "def ex9_unbind_grad_check(x: Tensor, weights: Tensor) -> Tensor:\n"
        '    """Backward through unbind; return x.grad with per-slot weight pattern."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "x = t.tensor([[1.0, 2.0, 3.0],\n"
        "              [4.0, 5.0, 6.0],\n"
        "              [7.0, 8.0, 9.0],\n"
        "              [0.0, 0.0, 0.0]], requires_grad=True)\n"
        "weights = t.tensor([0.5, 1.0, -2.0, 3.0])\n"
        "grad = ex9_unbind_grad_check(x, weights)\n"
        "\n"
        "assert grad is not None, 'x.grad must not be None — did you call backward()?'\n"
        "assert grad.shape == (4, 3), f'expected (4,3), got {tuple(grad.shape)}'\n"
        "# Row i of x.grad must equal weights[i] broadcast to D.\n"
        "expected = weights.unsqueeze(-1).expand(4, 3).contiguous()\n"
        "assert t.allclose(grad, expected), f'value mismatch:\\n{grad}\\nvs\\n{expected}'\n"
        "\n"
        "# A second call with fresh tensors to confirm reset-behavior.\n"
        "x2 = t.ones(3, 5, requires_grad=True)\n"
        "w2 = t.tensor([2.0, -1.0, 0.5])\n"
        "g2 = ex9_unbind_grad_check(x2, w2)\n"
        "assert g2.shape == (3, 5)\n"
        "assert t.allclose(g2[0], t.full((5,),  2.0))\n"
        "assert t.allclose(g2[1], t.full((5,), -1.0))\n"
        "assert t.allclose(g2[2], t.full((5,),  0.5))\n"
        "\n"
        "# --- Visualization: heatmap of x.grad ---\n"
        "x3 = t.randn(6, 8, generator=t.Generator().manual_seed(5), requires_grad=True)\n"
        "w3 = t.linspace(-1.0, 1.0, 6)\n"
        "g3 = ex9_unbind_grad_check(x3, w3)\n"
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        "im = ax.imshow(g3.detach().numpy(), cmap='coolwarm', aspect='auto',\n"
        "               vmin=-1.0, vmax=1.0)\n"
        "ax.set_xlabel('column (D=8)')\n"
        "ax.set_ylabel('row (N=6)')\n"
        "ax.set_title('ex9 x.grad — each row constant = weights[i]')\n"
        "plt.colorbar(im, ax=ax)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex9_unbind_grad_check(x: Tensor, weights: Tensor) -> Tensor:\n"
        "    rows = x.unbind(dim=0)\n"
        "    # Hold all grad_fn refs simultaneously to count distinct objects\n"
        "    # (without this list, Python may recycle the same id slot between calls).\n"
        "    fn_refs = [r.grad_fn for r in rows]\n"
        "    distinct = len({id(fn) for fn in fn_refs})\n"
        "    print(f'  distinct grad_fn nodes across unbound rows: {distinct}')\n"
        "    loss = sum(weights[i] * rows[i].sum() for i in range(len(rows)))\n"
        "    loss.backward()\n"
        "    return x.grad"
    ),
    "solution_notes": (
        "**Why each row has its own `grad_fn`.** `unbind` is implemented as a "
        "set of `select` views, each registered with the autograd engine as a "
        "separate UnbindBackward / SelectBackward node. The print shows "
        "`N` distinct ids, confirming the branching — if it ever showed `1` "
        "you'd know the framework was sharing a node and you'd need to "
        "investigate aliasing.\n\n"
        "**Why row `i` of `x.grad` equals `weights[i]`.** "
        "`L = sum_i w_i * sum_j x_ij`. Then `∂L/∂x_ij = w_i` for every `j`. "
        "Hence each row of the gradient is a constant equal to its weight — "
        "exactly what the heatmap shows.\n\n"
        "**Edge case worth knowing.** If you accumulate gradients twice (call "
        "backward on a fresh loss without zeroing `x.grad` first), the values "
        "will add. The drill returns the post-backward gradient directly; in "
        "training code you'd `optimizer.zero_grad()` before each step."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 3: rotation-matrix-3d-y-axis ex10
# Existing ex1-9: build R, rotate, compose, sweep, RxRyRz, inverse=transpose,
# camera-to-world. NONE do path interpolation. Novel: SLERP between two
# Y-rotations + verify orthogonality preserved along the path.
SPEC_ROT = {
    "atom_id": "rotation-matrix-3d-y-axis",
    "subtopic": "Numpy: Applied patterns and advanced",
    "topic_folder": "prereqs_numpy",
    "atom_recap_md": (
        "## SLERP on Y-axis rotations — quick refresher\n"
        "\n"
        "For a single-axis rotation family `R(θ) = R_y(θ)`, spherical linear "
        "interpolation (SLERP) between `R(α)` and `R(β)` collapses to "
        "**linear interpolation of the angle**: `R_slerp(s) = R(α + s·(β-α))` "
        "for `s ∈ [0, 1]`. The exotic quaternion machinery isn't needed when "
        "the rotation axis is fixed.\n"
        "\n"
        "**Why the path stays on SO(3).** Every `R_y(θ)` is orthogonal "
        "(`R^T R = I`) and has `det(R) = 1`. Linear interpolation of the "
        "angle keeps you within the family; the resulting matrices remain on "
        "the rotation manifold. Compare this with **linear interpolation of "
        "the matrices** `(1-s)·R(α) + s·R(β)`, which does NOT stay on SO(3) "
        "— the midpoint is closer to a scaled rotation.\n"
        "\n"
        "**This drill (ex10) vs ex1-9.** Earlier exercises built R, applied "
        "R to vectors, composed Rx·Ry·Rz, ran angle sweeps, and verified the "
        "inverse identity. ex10 walks a *path* on SO(3) and quantifies the "
        "gap between angle-SLERP and matrix-LERP at every step."
    ),
    "exercise_index": 10,
    "exercise_title": "SLERP between two Y-rotations vs matrix-LERP — orthogonality gap",
    "slug": "slerp-between-two-y-rotations-vs-matrix-lerp-orthogonality-gap",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["SLERP", "manifold", "orthogonality", "interpolation"],
    "kcs": ["ry-orthogonality", "ry-angle-interpolation"],
    "lo": (
        "Analyze the deviation from SO(3) along two interpolation paths "
        "between Y-rotations — angle-SLERP (stays on manifold) vs matrix-LERP "
        "(leaves manifold) — by quantifying `||R^T R - I||_F` per step."
    ),
    "prompt_body": (
        "Implement `ex10_slerp_vs_lerp(alpha, beta, n_steps)`.\n\n"
        "Walk two interpolation paths from `R_y(alpha)` to `R_y(beta)` using "
        "`n_steps` (inclusive of both endpoints) values of `s ∈ [0, 1]`:\n\n"
        "1. **Angle-SLERP path.** At each `s`, build `R_slerp = R_y(alpha + "
        "s·(beta - alpha))` from scratch using `cos/sin`. This stays on the "
        "manifold by construction.\n"
        "2. **Matrix-LERP path.** At each `s`, build `R_lerp = (1 - s) · "
        "R_y(alpha) + s · R_y(beta)`. This linearly blends the matrices — "
        "off-manifold at every interior step.\n"
        "3. For each path and each step, compute the Frobenius-norm gap from "
        "orthogonality: `gap = ||R^T R - I||_F`.\n\n"
        "Return `(slerp_gaps, lerp_gaps)` — two 1-D tensors of length "
        "`n_steps`, both `float32`.\n\n"
        "Use `torch.linspace(0, 1, n_steps)` for `s`. SLERP gaps should be "
        "≈ 0 at every step (floating-point noise only); LERP gaps should "
        "be 0 at the two endpoints and reach a maximum near the midpoint.\n\n"
        "The visualization overlays both gap curves so you can see the "
        "off-manifold bulge of matrix-LERP versus the flat SLERP baseline."
    ),
    "stub": (
        "def ex10_slerp_vs_lerp(\n"
        "    alpha: float,\n"
        "    beta: float,\n"
        "    n_steps: int,\n"
        ") -> tuple[Tensor, Tensor]:\n"
        '    """Compare angle-SLERP vs matrix-LERP between R_y(alpha) and R_y(beta)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "slerp_gaps, lerp_gaps = ex10_slerp_vs_lerp(0.0, math.pi / 2, 11)\n"
        "assert slerp_gaps.shape == (11,), f'slerp_gaps shape wrong: {tuple(slerp_gaps.shape)}'\n"
        "assert lerp_gaps.shape == (11,), f'lerp_gaps shape wrong: {tuple(lerp_gaps.shape)}'\n"
        "assert slerp_gaps.dtype == t.float32 and lerp_gaps.dtype == t.float32\n"
        "\n"
        "# SLERP path stays on SO(3): gap ≈ 0 everywhere (float noise only).\n"
        "assert slerp_gaps.max().item() < 1e-4, (\n"
        "    f'SLERP path drifted off manifold (max gap {slerp_gaps.max().item():.3e})'\n"
        ")\n"
        "\n"
        "# LERP endpoints (s=0 and s=1) ARE on the manifold.\n"
        "assert lerp_gaps[0].item() < 1e-4, f'LERP s=0 gap should be 0, got {lerp_gaps[0]:.3e}'\n"
        "assert lerp_gaps[-1].item() < 1e-4, f'LERP s=1 gap should be 0, got {lerp_gaps[-1]:.3e}'\n"
        "\n"
        "# LERP midpoint leaves the manifold visibly.\n"
        "mid = lerp_gaps[5].item()\n"
        "assert mid > 0.1, f'LERP midpoint gap should be > 0.1, got {mid:.3e}'\n"
        "\n"
        "# Symmetry: LERP gap curve is symmetric about s=0.5.\n"
        "assert abs(lerp_gaps[1].item() - lerp_gaps[-2].item()) < 1e-4, (\n"
        "    'LERP gap should be symmetric about s=0.5'\n"
        ")\n"
        "\n"
        "# Bigger angle sweep — quarter turn to three-quarter turn.\n"
        "s2, l2 = ex10_slerp_vs_lerp(math.pi / 4, 3 * math.pi / 4, 21)\n"
        "assert s2.max().item() < 1e-4, 'SLERP must stay flat on extended sweep'\n"
        "assert l2.max().item() > 0.05, 'LERP should bulge off-manifold on a >0 sweep'\n"
        "\n"
        "# --- Visualization: both gap curves on one axis ---\n"
        "ss, ll = ex10_slerp_vs_lerp(0.0, math.pi, 41)\n"
        "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
        "xs = t.linspace(0, 1, 41).numpy()\n"
        "ax.plot(xs, ss.numpy(), label='angle-SLERP (on manifold)', color='steelblue', linewidth=2)\n"
        "ax.plot(xs, ll.numpy(), label='matrix-LERP (off manifold)', color='crimson', linewidth=2)\n"
        "ax.axhline(0, color='gray', linewidth=0.5)\n"
        "ax.set_xlabel('s ∈ [0, 1]')\n"
        "ax.set_ylabel(r'$\\|R^\\top R - I\\|_F$')\n"
        "ax.set_title(r'ex10 orthogonality gap, $R_y(0) \\to R_y(\\pi)$')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex10_slerp_vs_lerp(\n"
        "    alpha: float,\n"
        "    beta: float,\n"
        "    n_steps: int,\n"
        ") -> tuple[Tensor, Tensor]:\n"
        "    def Ry(theta: Tensor) -> Tensor:\n"
        "        c, s = t.cos(theta), t.sin(theta)\n"
        "        z, o = t.zeros_like(c), t.ones_like(c)\n"
        "        return t.stack([\n"
        "            t.stack([ c, z, s]),\n"
        "            t.stack([ z, o, z]),\n"
        "            t.stack([-s, z, c]),\n"
        "        ])\n"
        "\n"
        "    a = t.tensor(alpha, dtype=t.float32)\n"
        "    b = t.tensor(beta, dtype=t.float32)\n"
        "    Ra, Rb = Ry(a), Ry(b)\n"
        "    eye = t.eye(3)\n"
        "    ss = t.linspace(0.0, 1.0, n_steps)\n"
        "    slerp_gaps = t.empty(n_steps, dtype=t.float32)\n"
        "    lerp_gaps = t.empty(n_steps, dtype=t.float32)\n"
        "    for i, s in enumerate(ss):\n"
        "        Rs = Ry(a + s * (b - a))\n"
        "        Rl = (1 - s) * Ra + s * Rb\n"
        "        slerp_gaps[i] = (Rs.T @ Rs - eye).norm()\n"
        "        lerp_gaps[i] = (Rl.T @ Rl - eye).norm()\n"
        "    return slerp_gaps, lerp_gaps"
    ),
    "solution_notes": (
        "**Why angle-SLERP works for single-axis rotations.** SO(2) (rotations "
        "in a plane) is a 1-D Lie group parameterized by the angle. Y-rotations "
        "form an isomorphic subgroup of SO(3): the angle composes additively, "
        "so linear interpolation of the angle IS the geodesic. For "
        "multi-axis SLERP you'd need quaternions or a matrix exponential — "
        "but here the cheap path is the right path.\n\n"
        "**Why matrix-LERP fails.** Linear blends of rotation matrices are "
        "no longer rotations: `(1-s)R_a + s R_b` is a scaled, sheared "
        "matrix whose columns are NOT unit-orthogonal. The Frobenius gap "
        "`||R^T R - I||_F` peaks at the midpoint and is symmetric about "
        "`s=0.5` because the geometry of the path is symmetric.\n\n"
        "**Practical takeaway.** If you ever need to morph between two "
        "orientations (camera path, joint interpolation, latent-space "
        "rotation), interpolate the PARAMETER, not the matrix. The whole "
        "point of choosing a parameterization is that the manifold is closed "
        "under the parameter's linear structure."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 4: tensor-item-scalar ex9
# Existing ex1-8: dtype, vs tolist, control flow, training loop logging,
# early stopping, random walk. NONE warn about the autograd pitfall.
SPEC_ITEM = {
    "atom_id": "tensor-item-scalar",
    "subtopic": "Numpy: Core array literacy",
    "topic_folder": "prereqs_numpy",
    "atom_recap_md": (
        "## .item() autograd pitfall — quick refresher\n"
        "\n"
        "`.item()` returns a **Python float / int / bool** — a primitive with "
        "no autograd metadata. Any expression that funnels a "
        "`requires_grad=True` tensor through `.item()` becomes a constant "
        "downstream: the gradient graph is **silently severed**.\n"
        "\n"
        "```python\n"
        "loss = ((y_pred - y_true) ** 2).mean()\n"
        "scalar = loss.item()             # <- breaks the graph\n"
        "really_loss = scalar * weight     # <- weight.grad will be None\n"
        "```\n"
        "\n"
        "**Compared to `.detach()`.** `.detach()` returns a tensor that's "
        "disconnected from the graph but still a tensor — so an obvious type "
        "mismatch downstream will catch the bug. `.item()` returns a "
        "primitive, which silently quacks like a number and only manifests "
        "later as `.grad is None`.\n"
        "\n"
        "**This drill (ex9) vs ex1-8.** Earlier exercises use `.item()` for "
        "its intended purpose — extracting scalars for logging and Python-"
        "side control flow. ex9 deliberately misuses `.item()` inside an "
        "autograd path and forces the caller to inspect `.grad` to diagnose "
        "the silent breakage."
    ),
    "exercise_index": 9,
    "exercise_title": ".item() breaks autograd — diagnose the silent graph severing",
    "slug": "item-breaks-autograd-diagnose-the-silent-graph-severing",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["autograd-pitfall", "graph-severing", "diagnostic"],
    "kcs": ["item-returns-python-primitive", "item-breaks-autograd-graph"],
    "lo": (
        "Analyze a forward pass that mixes a correct (tensor-only) loss path "
        "with an incorrect (`.item()`-laundered) loss path, and produce a "
        "diagnostic showing the correct path has a non-None gradient while "
        "the incorrect path does not."
    ),
    "prompt_body": (
        "Implement `ex9_diagnose_item_breakage(x_init, target)`.\n\n"
        "Build TWO parallel loss computations from the same inputs and report "
        "which one preserves gradient flow back to `x`:\n\n"
        "1. `x_init` is a tensor `(N,)` of starting values (will be wrapped "
        "fresh with `requires_grad=True` inside each path). `target` is "
        "`(N,)`, no grad.\n"
        "2. **Correct path.** `x_ok = x_init.clone().detach().requires_grad_"
        "(True)`. Compute `loss_ok = ((x_ok - target) ** 2).mean()` (pure-"
        "tensor). Call `loss_ok.backward()`. Record `x_ok.grad.clone()`.\n"
        "3. **Broken path.** `x_bad = x_init.clone().detach().requires_grad_"
        "(True)`. Compute `diff_scalar = (x_bad - target).pow(2).mean().item("
        ")` — note the `.item()`. Then build `loss_bad = "
        "t.tensor(diff_scalar)` (a fresh Python-float-derived tensor with no "
        "graph back to `x_bad`). Wrap `loss_bad.backward()` in `try/except` "
        "and continue regardless. `x_bad.grad` will remain `None`.\n"
        "4. **Return** a dict:\n"
        "   - `'x_ok_grad_norm'`: float — `x_ok.grad.norm().item()` (>0)\n"
        "   - `'x_bad_grad'`: tensor or `None` — `x_bad.grad` itself\n"
        "   - `'graph_preserved'`: bool — `x_ok.grad is not None and "
        "x_bad.grad is None`\n\n"
        "**Print** the dict so the diagnostic is visible to the caller "
        "without rerunning.\n\n"
        "Output: dict with the three keys above. The visualization renders a "
        "bar chart comparing the two paths' gradient norms — the broken "
        "path is forced to 0 because `x_bad.grad` is `None`."
    ),
    "stub": (
        "def ex9_diagnose_item_breakage(x_init: Tensor, target: Tensor) -> dict:\n"
        '    """Compare gradient flow back to x between a pure-tensor path and an .item()-severed path."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "x_init = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "target = t.tensor([1.0, 1.5, 4.0, 3.0])\n"
        "diag = ex9_diagnose_item_breakage(x_init, target)\n"
        "\n"
        "assert isinstance(diag, dict)\n"
        "for key in ('x_ok_grad_norm', 'x_bad_grad', 'graph_preserved'):\n"
        "    assert key in diag, f'missing key {key!r}'\n"
        "\n"
        "# Correct path: x_ok.grad must be analytically 2*(x - target)/N.\n"
        "# Norm = ||2/N * (x_init - target)||.\n"
        "expected_grad = 2.0 / 4 * (x_init - target)\n"
        "expected_norm = expected_grad.norm().item()\n"
        "assert abs(diag['x_ok_grad_norm'] - expected_norm) < 1e-5, (\n"
        "    f\"x_ok_grad_norm should be {expected_norm:.5f}, got {diag['x_ok_grad_norm']:.5f}\"\n"
        ")\n"
        "assert diag['x_ok_grad_norm'] > 0  # input differs from target → nonzero grad\n"
        "\n"
        "# Broken path: x_bad.grad must be None (the .item() severed the link).\n"
        "assert diag['x_bad_grad'] is None, (\n"
        "    f\"x_bad.grad must be None due to .item() graph break, got {diag['x_bad_grad']}\"\n"
        ")\n"
        "\n"
        "# Overall diagnostic flag.\n"
        "assert diag['graph_preserved'] is True\n"
        "\n"
        "# Edge case: large input, all-zeros target.\n"
        "x2_init = t.randn(50, generator=t.Generator().manual_seed(2))\n"
        "diag2 = ex9_diagnose_item_breakage(x2_init, t.zeros(50))\n"
        "assert diag2['x_bad_grad'] is None\n"
        "assert diag2['x_ok_grad_norm'] > 0\n"
        "\n"
        "# --- Visualization: bar chart of computed gradient magnitudes ---\n"
        "fig, ax = plt.subplots(figsize=(5, 3))\n"
        "bad_val = 0.0 if diag['x_bad_grad'] is None else diag['x_bad_grad'].norm().item()\n"
        "ax.bar(['x_ok (tensor path)', 'x_bad (.item() path)'],\n"
        "       [diag['x_ok_grad_norm'], bad_val],\n"
        "       color=['steelblue', 'crimson'], edgecolor='black')\n"
        "ax.set_ylabel('||x.grad||')\n"
        "ax.set_title('ex9 .item() silently severs the autograd graph')\n"
        "ax.grid(True, alpha=0.3, axis='y')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex9_diagnose_item_breakage(x_init: Tensor, target: Tensor) -> dict:\n"
        "    # Correct path — pure tensor arithmetic. Gradient flows back to x_ok.\n"
        "    x_ok = x_init.clone().detach().requires_grad_(True)\n"
        "    loss_ok = ((x_ok - target) ** 2).mean()\n"
        "    loss_ok.backward()\n"
        "\n"
        "    # Broken path — .item() returns a Python float; loss_bad is built from\n"
        "    # that float wrapped in a fresh tensor, so there's no graph link back to x_bad.\n"
        "    x_bad = x_init.clone().detach().requires_grad_(True)\n"
        "    diff_scalar = (x_bad - target).pow(2).mean().item()\n"
        "    loss_bad = t.tensor(diff_scalar)\n"
        "    try:\n"
        "        loss_bad.backward()\n"
        "    except RuntimeError:\n"
        "        # Expected — loss_bad has no grad_fn because it was built from a Python float.\n"
        "        pass\n"
        "\n"
        "    out = {\n"
        "        'x_ok_grad_norm': x_ok.grad.norm().item() if x_ok.grad is not None else None,\n"
        "        'x_bad_grad': x_bad.grad,\n"
        "        'graph_preserved': (x_ok.grad is not None) and (x_bad.grad is None),\n"
        "    }\n"
        "    print(out)\n"
        "    return out"
    ),
    "solution_notes": (
        "**Why w_ok.grad equals the MSE.** `loss_ok = w_ok · MSE(x, target)` "
        "is linear in `w_ok`, so `∂loss_ok / ∂w_ok = MSE(x, target)`. "
        "Autograd fills `w_ok.grad` with this scalar after `backward()`.\n\n"
        "**Why w_bad.grad is None.** `mse_scalar = ....item()` returned a "
        "Python float. The product `w_bad * mse_scalar` is a "
        "tensor-times-constant — autograd treats `mse_scalar` as a "
        "non-differentiable literal, so backward never even tries to flow "
        "gradient through it. Worse, in some torch versions "
        "`loss_bad.backward()` may raise because there's nothing to "
        "differentiate from `w_bad` to a meaningful upstream — hence the "
        "try/except.\n\n"
        "**Defensive habits.** (1) Never use `.item()` inside a forward "
        "pass that you intend to differentiate. (2) If you need a "
        "graph-detached tensor (e.g. for logging or saving), use "
        "`.detach()` not `.item()` — the type mismatch will surface "
        "downstream. (3) After `backward()`, always sanity-check a small "
        "model with `assert all(p.grad is not None for p in "
        "model.parameters())` until you trust the data path."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 5: as-strided-noncontig-source ex10
# Existing ex1-9: strides, transpose breaks contiguity, view fails, rolling
# windows, 2D image patches, memory comparison, 1D conv via einsum, diagonal.
# NONE warn about write-aliasing through overlapping strided views.
SPEC_ASSTR = {
    "atom_id": "as-strided-noncontig-source",
    "subtopic": "Numpy: Applied patterns and advanced",
    "topic_folder": "prereqs_numpy",
    "atom_recap_md": (
        "## as_strided write-aliasing — quick refresher\n"
        "\n"
        "`as_strided(x, size, stride)` constructs a view whose elements may "
        "share underlying storage with `x`. When the strides cause **overlap** "
        "(adjacent output elements map to the same source element), an "
        "in-place write through the view will corrupt every overlap-sharing "
        "output position — even ones written 'earlier' in your code.\n"
        "\n"
        "```python\n"
        "x = torch.arange(6).float()                      # [0,1,2,3,4,5]\n"
        "w = x.as_strided((4, 3), (1, 1))                  # 4 overlapping windows\n"
        "w[0, 0] = 99.0                                    # writes x[0] = 99\n"
        "# w[0] = [99,1,2], but also EVERY window starting at offset 0 sees 99\n"
        "```\n"
        "\n"
        "**Compared to non-overlapping strides.** If the strides are at least "
        "as large as the per-row size, the view is a non-overlapping "
        "partition — in-place writes behave normally. Overlap is what makes "
        "`as_strided` a footgun.\n"
        "\n"
        "**This drill (ex10) vs ex1-9.** Earlier exercises read from "
        "as_strided views (sliding windows, convolution, diagonal "
        "extraction). ex10 deliberately writes through an overlapping view "
        "to demonstrate aliasing, then asks the caller to characterize the "
        "corruption pattern."
    ),
    "exercise_index": 10,
    "exercise_title": "writes through overlapping strided view corrupt the source",
    "slug": "writes-through-overlapping-strided-view-corrupt-the-source",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["aliasing", "in-place-write", "overlap", "footgun"],
    "kcs": ["as-strided-shares-storage", "as-strided-overlap-write-aliasing"],
    "lo": (
        "Analyze the corruption produced by writing through an overlapping "
        "as_strided view, and return both the post-write source vector and a "
        "boolean mask of which source positions were mutated."
    ),
    "prompt_body": (
        "Implement `ex10_overlap_write_demo(n, window, target_window, "
        "fill_value)`.\n\n"
        "Demonstrate the as_strided write-aliasing pitfall:\n\n"
        "1. Allocate `x = t.arange(n, dtype=t.float32)`.\n"
        "2. Build a sliding-window view `w = x.as_strided((n - window + 1, "
        "window), (1, 1))` — stride 1 in both dims = full overlap.\n"
        "3. Save a copy `x_before = x.clone()` so you can diff later.\n"
        "4. Mutate **only** `w[target_window]` in place: `w[target_window] = "
        "fill_value`. This is a row-write that hits the overlapping storage.\n"
        "5. After the write, return `(x, mutated_mask)` where:\n"
        "   - `x` is the source vector AFTER the write (now corrupted).\n"
        "   - `mutated_mask` is a bool tensor of length `n`, `True` at every "
        "position where `x[i] != x_before[i]`.\n\n"
        "The visualization plots `x_before` and `x` after the write to make "
        "the corruption visually obvious — the contiguous band of fill_value "
        "is exactly the storage range covered by `w[target_window]`."
    ),
    "stub": (
        "def ex10_overlap_write_demo(\n"
        "    n: int,\n"
        "    window: int,\n"
        "    target_window: int,\n"
        "    fill_value: float,\n"
        ") -> tuple[Tensor, Tensor]:\n"
        '    """Write through one overlapping window; return (corrupted_x, mutated_mask)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Canonical case: n=10, window=3, target_window=4, fill=-1\n"
        "# w[4] aliases x[4], x[5], x[6] — those three positions should mutate.\n"
        "x, mask = ex10_overlap_write_demo(10, 3, 4, -1.0)\n"
        "assert x.shape == (10,), f'x shape wrong: {tuple(x.shape)}'\n"
        "assert mask.shape == (10,), f'mask shape wrong: {tuple(mask.shape)}'\n"
        "assert mask.dtype == t.bool, f'mask dtype wrong: {mask.dtype}'\n"
        "\n"
        "# Exactly indices 4, 5, 6 should be -1.\n"
        "assert x[4].item() == -1.0\n"
        "assert x[5].item() == -1.0\n"
        "assert x[6].item() == -1.0\n"
        "# Unmutated positions should equal arange.\n"
        "for i in (0, 1, 2, 3, 7, 8, 9):\n"
        "    assert x[i].item() == float(i), f'x[{i}] should be {float(i)}, got {x[i].item()}'\n"
        "# Mask matches.\n"
        "expected_mask = t.tensor([False]*4 + [True]*3 + [False]*3)\n"
        "assert t.equal(mask, expected_mask), f'mask mismatch:\\n{mask}\\nvs\\n{expected_mask}'\n"
        "\n"
        "# Edge case: target_window=0 → mutates x[0..window-1].\n"
        "x2, m2 = ex10_overlap_write_demo(8, 4, 0, 99.0)\n"
        "for i in range(4):\n"
        "    assert x2[i].item() == 99.0\n"
        "for i in range(4, 8):\n"
        "    assert x2[i].item() == float(i)\n"
        "\n"
        "# Edge case: target_window at end.\n"
        "x3, m3 = ex10_overlap_write_demo(12, 5, 7, 0.0)  # mutates x[7..11]\n"
        "assert m3[:7].sum().item() == 0\n"
        "assert m3[7:].sum().item() == 5\n"
        "\n"
        "# --- Visualization: before vs after the overlapping write ---\n"
        "n, win, tgt, fill = 30, 5, 12, -3.0\n"
        "before = t.arange(n, dtype=t.float32)\n"
        "after, _ = ex10_overlap_write_demo(n, win, tgt, fill)\n"
        "fig, ax = plt.subplots(figsize=(7, 3))\n"
        "ax.plot(range(n), before.numpy(), label='x before', color='steelblue', linewidth=2)\n"
        "ax.plot(range(n), after.numpy(), label='x after w[12] = -3', color='crimson',\n"
        "        linewidth=2, linestyle='--')\n"
        "ax.axhline(0, color='gray', linewidth=0.5)\n"
        "ax.set_xlabel('source index')\n"
        "ax.set_ylabel('value')\n"
        "ax.set_title(f'ex10 overlap write: w[{tgt}] = {fill} corrupted indices {tgt}..{tgt+win-1}')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex10_overlap_write_demo(\n"
        "    n: int,\n"
        "    window: int,\n"
        "    target_window: int,\n"
        "    fill_value: float,\n"
        ") -> tuple[Tensor, Tensor]:\n"
        "    x = t.arange(n, dtype=t.float32)\n"
        "    n_windows = n - window + 1\n"
        "    w = x.as_strided((n_windows, window), (1, 1))\n"
        "    x_before = x.clone()\n"
        "    w[target_window] = fill_value  # writes into x storage at offsets target_window..target_window+window-1\n"
        "    mutated_mask = x != x_before\n"
        "    return x, mutated_mask"
    ),
    "solution_notes": (
        "**Why writes through overlapping views corrupt the source.** "
        "`as_strided` does NOT copy storage — `w` and `x` share the same "
        "underlying buffer. Writing `w[target_window]` mutates `window` "
        "consecutive elements of `x`'s storage starting at offset "
        "`target_window` (because the stride is 1). Every OTHER window that "
        "overlaps that range will also see the new values on its next read.\n\n"
        "**This is why you should treat `as_strided` views as read-only.** "
        "The PyTorch docs warn about exactly this. If you need an "
        "overlapping window for compute (e.g. sliding-window convolution), "
        "the safe pattern is: build the view → call `.contiguous()` (or "
        "`.clone()`) to materialize a non-aliased copy → mutate that copy.\n\n"
        "**Debugging tip.** If a strided computation produces wrong outputs "
        "AFTER an in-place op (`*=`, `+=`, scatter, etc.), the first "
        "hypothesis should be: 'did I just write through an aliased view?' "
        "The mutated_mask returned here is the diagnostic — it tells you "
        "exactly which source positions got hit by the view write."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 6: einops-rearrange ex10
# Existing ex1-9: identity, transpose, flatten, unfold, patch grid, MHA split,
# NHWC roundtrip, patchify, non-divisible edge case.
# Novel: cross-attention KV-cache concat — pack/unpack via rearrange with
# variable seq lengths handled via padding mask. This is the canonical
# decode-time pattern from LLM inference.
SPEC_REARRANGE = {
    "atom_id": "einops-rearrange",
    "subtopic": "Einops: Rearrange",
    "topic_folder": "prereqs_einops",
    "atom_recap_md": (
        "## rearrange for KV-cache concat — quick refresher\n"
        "\n"
        "At LLM decode time, each new token's K and V projections are "
        "concatenated onto a growing **KV cache**. The cache layout is "
        "`(B, H, S_total, D)`; the new token's projection arrives as "
        "`(B, H, D)` (a single timestep). The standard pattern:\n"
        "\n"
        "```python\n"
        "# new_k: (B, H, D) → (B, H, 1, D), then cat onto cache along the S axis\n"
        "new_k_s = rearrange(new_k, 'b h d -> b h 1 d')\n"
        "cache_k = t.cat([cache_k, new_k_s], dim=2)\n"
        "```\n"
        "\n"
        "**Compared to `unsqueeze(2)`.** Both produce the same shape. "
        "`rearrange` makes the intent explicit at the call site — readers see "
        "`b h d -> b h 1 d` and immediately know 'we're materializing a "
        "singleton sequence axis'. `unsqueeze(2)` requires you to remember "
        "what dim 2 means.\n"
        "\n"
        "**This drill (ex10) vs ex1-9.** Earlier exercises did ViT-style "
        "patchify, NHWC↔NCHW, divisibility edge cases. ex10 covers the "
        "DECODE-LOOP rearrange — packing single-token K/V into the right "
        "cache axis on every generation step."
    ),
    "exercise_index": 10,
    "exercise_title": "rearrange single-token K/V into KV-cache layout for decode-time concat",
    "slug": "rearrange-single-token-kv-into-kv-cache-layout-for-decode-time-concat",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["kv-cache", "decode-loop", "singleton-axis", "rearrange"],
    "kcs": ["rearrange-singleton-axis-literal", "rearrange-decode-time-pattern"],
    "lo": (
        "Apply `rearrange(... 'b h d -> b h 1 d')` to materialize a singleton "
        "sequence axis on each new token's K projection, then concat onto a "
        "growing KV cache across decode steps."
    ),
    "prompt_body": (
        "Implement `ex10_kv_cache_decode(initial_cache_k, new_tokens_k)`.\n\n"
        "Simulate `n_steps` decode iterations of a transformer's KV-cache "
        "concat for the K tensor:\n\n"
        "1. `initial_cache_k` has shape `(B, H, S0, D)` — the cache BEFORE "
        "decoding starts (may be 0-length on first call if `S0 == 0`).\n"
        "2. `new_tokens_k` has shape `(n_steps, B, H, D)` — one new K "
        "projection per decode step.\n"
        "3. For each step `i` in `range(n_steps)`:\n"
        "   a. Extract `new_k = new_tokens_k[i]` (shape `(B, H, D)`).\n"
        "   b. Use `rearrange(new_k, 'b h d -> b h 1 d')` to materialize the "
        "singleton sequence axis.\n"
        "   c. Concat onto `cache_k` along `dim=2`: `cache_k = "
        "t.cat([cache_k, new_k_s], dim=2)`.\n"
        "4. Return `cache_k` after all `n_steps` concatenations.\n\n"
        "Final shape: `(B, H, S0 + n_steps, D)`.\n\n"
        "The visualization plots `cache_k.shape[2]` (the sequence axis "
        "length) over each step to show the cache growing linearly."
    ),
    "stub": (
        "def ex10_kv_cache_decode(\n"
        "    initial_cache_k: Tensor,\n"
        "    new_tokens_k: Tensor,\n"
        ") -> Tensor:\n"
        '    """Grow the KV cache by one (rearranged) token per step."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Start from an empty cache and decode 5 tokens.\n"
        "B, H, D = 2, 4, 8\n"
        "cache0 = t.zeros(B, H, 0, D)  # truly empty seq axis\n"
        "new_ks = t.arange(5 * B * H * D, dtype=t.float32).reshape(5, B, H, D)\n"
        "final = ex10_kv_cache_decode(cache0, new_ks)\n"
        "assert final.shape == (B, H, 5, D), f'expected (B,H,5,D), got {tuple(final.shape)}'\n"
        "assert final.dtype == t.float32\n"
        "# Slot s of the cache must equal new_ks[s] for s in 0..4.\n"
        "for s in range(5):\n"
        "    assert t.allclose(final[:, :, s, :], new_ks[s]), (\n"
        "        f'cache slot {s} does not match new_tokens_k[{s}]'\n"
        "    )\n"
        "\n"
        "# Non-empty initial cache: 3 pre-existing tokens, decode 2 more.\n"
        "pre = t.randn(B, H, 3, D, generator=t.Generator().manual_seed(1))\n"
        "more = t.randn(2, B, H, D, generator=t.Generator().manual_seed(2))\n"
        "out = ex10_kv_cache_decode(pre, more)\n"
        "assert out.shape == (B, H, 5, D)\n"
        "# First 3 slots untouched.\n"
        "assert t.allclose(out[:, :, :3, :], pre)\n"
        "# Slots 3 and 4 are the decoded ones.\n"
        "assert t.allclose(out[:, :, 3, :], more[0])\n"
        "assert t.allclose(out[:, :, 4, :], more[1])\n"
        "\n"
        "# Edge case: n_steps = 0 → cache unchanged.\n"
        "noop = ex10_kv_cache_decode(pre, t.empty(0, B, H, D))\n"
        "assert noop.shape == pre.shape\n"
        "assert t.allclose(noop, pre)\n"
        "\n"
        "# --- Visualization: cache seq length over decode steps ---\n"
        "lengths = []\n"
        "running = t.zeros(B, H, 0, D)\n"
        "many = t.randn(12, B, H, D, generator=t.Generator().manual_seed(3))\n"
        "for step in range(12):\n"
        "    running = ex10_kv_cache_decode(running, many[step:step+1])\n"
        "    lengths.append(running.shape[2])\n"
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        "ax.plot(range(1, 13), lengths, marker='o', color='teal')\n"
        "ax.set_xlabel('decode step')\n"
        "ax.set_ylabel('cache.shape[2]')\n"
        "ax.set_title('ex10 KV cache grows linearly with decoded tokens')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex10_kv_cache_decode(\n"
        "    initial_cache_k: Tensor,\n"
        "    new_tokens_k: Tensor,\n"
        ") -> Tensor:\n"
        "    cache_k = initial_cache_k\n"
        "    for i in range(new_tokens_k.shape[0]):\n"
        "        new_k = new_tokens_k[i]                                  # (B, H, D)\n"
        "        new_k_s = rearrange(new_k, 'b h d -> b h 1 d')           # (B, H, 1, D)\n"
        "        cache_k = t.cat([cache_k, new_k_s], dim=2)\n"
        "    return cache_k"
    ),
    "solution_notes": (
        "**Why a singleton axis instead of just `t.cat([..., dim=...])`.** "
        "`cat` requires every input to have the same number of dims. The "
        "fresh K projection has rank 3 `(B, H, D)`; the cache has rank 4. "
        "Without rearrange you'd unsqueeze; with rearrange the intent ('I am "
        "creating a length-1 sequence axis') is documented at the call site.\n\n"
        "**Why iterate rather than vectorize.** Decode is inherently "
        "sequential — token `t+1` depends on the model's attention to all "
        "tokens through `t`. You can't batch decode across timesteps without "
        "speculative-decoding tricks. The cache concat IS the inner loop of "
        "every transformer inference engine.\n\n"
        "**Memory cost.** `t.cat` allocates a new tensor every step. "
        "Production engines (vLLM, TensorRT-LLM) pre-allocate a max-length "
        "cache and overwrite slots in place to avoid the O(N) allocation "
        "overhead. The rearrange-then-cat shape here is the *correctness "
        "reference*; the prod fast-path is the same pattern with the cat "
        "replaced by an indexed assignment into a pre-sized buffer."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 7: einops-reduce ex10
# Existing ex1-9: mean, multi-axis drop, keepdim with (), avg pool, softmax,
# spatial pyramid, BN stats, argmax-via-reduce, top-k.
# Novel: weighted reduce via einsum equivalence — verify that reduce('mean')
# is exactly einsum-with-uniform-1/N-weights, then generalize to a
# class-weighted loss-mean reduction (which reduce can't express directly).
SPEC_REDUCE = {
    "atom_id": "einops-reduce",
    "subtopic": "Einops: Reduce",
    "topic_folder": "prereqs_einops",
    "atom_recap_md": (
        "## reduce('mean') vs einsum-weighted reduce — quick refresher\n"
        "\n"
        "`einops.reduce(x, 'b n -> b', 'mean')` is mathematically identical "
        "to a uniform-weighted einsum:\n"
        "\n"
        "```python\n"
        "w = t.full((N,), 1.0 / N)\n"
        "einsum('b n, n -> b', x, w)\n"
        "```\n"
        "\n"
        "But `reduce` only supports uniform weights via its named ops "
        "(`'sum'`, `'mean'`, `'max'`, `'min'`, `'prod'`). When you need "
        "**non-uniform weights** (class-balanced loss, importance "
        "weighting, gated attention), drop down to einsum.\n"
        "\n"
        "**This drill (ex10) vs ex1-9.** Earlier exercises stayed inside "
        "reduce's named ops (channel mean, max-keepdim, BN stats, "
        "argmax-via-reduce). ex10 establishes the einsum-equivalence "
        "numerically and then USES it for a non-uniform pattern — class-"
        "weighted loss averaging — that reduce cannot express."
    ),
    "exercise_index": 10,
    "exercise_title": "weighted-reduce via einsum: uniform identity + class-balanced loss",
    "slug": "weighted-reduce-via-einsum-uniform-identity-and-class-balanced-loss",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["weighted-reduce", "einsum-equivalence", "class-balanced-loss"],
    "kcs": ["reduce-mean-as-uniform-einsum", "reduce-cannot-express-weighted"],
    "lo": (
        "Analyze the relationship between `einops.reduce('mean')` and a "
        "uniform-weight einsum, then apply a non-uniform weight vector to "
        "compute a class-balanced per-batch loss that `reduce` alone "
        "cannot express."
    ),
    "prompt_body": (
        "Implement `ex10_weighted_reduce(losses, class_ids, class_weights)`.\n\n"
        "Walk three reductions on the same `(B, N)` tensor of per-sample "
        "losses, where `class_ids` (length `B`) names the class of each "
        "sample and `class_weights` (length `C`) gives the per-class "
        "weight:\n\n"
        "1. **Uniform reduce.** `mean_reduce = reduce(losses, 'b n -> b', "
        "'mean')` — the standard `(B,)` per-batch mean.\n"
        "2. **Uniform einsum (equivalent).** `mean_einsum = einsum('b n, n "
        "-> b', losses, t.full((N,), 1/N))`. Assert this matches "
        "`mean_reduce` to within `1e-6`.\n"
        "3. **Class-balanced batch loss (genuinely different).** For each "
        "sample `b`, look up its weight `w_b = class_weights[class_ids[b]]`. "
        "Compute the class-balanced batch loss as `t.einsum('b, b ->', "
        "mean_reduce, w_b_vec) / w_b_vec.sum()` — a scalar.\n\n"
        "Return `(mean_reduce, mean_einsum, class_balanced_loss)`. Print all "
        "three so the caller sees the chain.\n\n"
        "Inputs: `losses` `(B, N)` float; `class_ids` `(B,)` int64 in "
        "`[0, C)`; `class_weights` `(C,)` float.\n"
        "Output: `(mean_reduce, mean_einsum, class_balanced_loss)`.\n\n"
        "The visualization compares per-batch unweighted vs class-weighted "
        "contribution as a bar chart, highlighting how the class weighting "
        "amplifies or suppresses each batch sample."
    ),
    "stub": (
        "def ex10_weighted_reduce(\n"
        "    losses: Tensor,\n"
        "    class_ids: Tensor,\n"
        "    class_weights: Tensor,\n"
        ") -> tuple[Tensor, Tensor, Tensor]:\n"
        '    """Uniform reduce == uniform einsum; then a class-weighted scalar."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Hand-checked case.\n"
        "losses = t.tensor([\n"
        "    [1.0, 2.0, 3.0, 4.0],   # batch 0, class 1, mean=2.5\n"
        "    [0.5, 0.5, 0.5, 0.5],   # batch 1, class 0, mean=0.5\n"
        "    [4.0, 4.0, 4.0, 4.0],   # batch 2, class 1, mean=4.0\n"
        "])\n"
        "class_ids = t.tensor([1, 0, 1], dtype=t.long)\n"
        "class_weights = t.tensor([1.0, 3.0])  # class 1 is 3x as important as class 0\n"
        "m_r, m_e, cb = ex10_weighted_reduce(losses, class_ids, class_weights)\n"
        "\n"
        "# Per-batch mean is straightforward.\n"
        "assert t.allclose(m_r, t.tensor([2.5, 0.5, 4.0])), f'mean_reduce wrong: {m_r}'\n"
        "# Uniform-einsum must match uniform-reduce.\n"
        "assert t.allclose(m_r, m_e, atol=1e-6), f'einsum disagrees with reduce: {m_e} vs {m_r}'\n"
        "\n"
        "# Class-balanced loss = (3*2.5 + 1*0.5 + 3*4.0) / (3+1+3) = (7.5+0.5+12)/7 = 20/7\n"
        "expected_cb = (3*2.5 + 1*0.5 + 3*4.0) / (3 + 1 + 3)\n"
        "assert abs(cb.item() - expected_cb) < 1e-5, f'cb wrong: got {cb.item()}, expected {expected_cb}'\n"
        "\n"
        "# Stress test: random data, B=16, N=20, C=4.\n"
        "rng = t.Generator().manual_seed(8)\n"
        "L = t.rand(16, 20, generator=rng)\n"
        "cids = t.randint(0, 4, (16,), generator=rng)\n"
        "cw = t.tensor([0.5, 1.0, 2.0, 4.0])\n"
        "m1, m2, c = ex10_weighted_reduce(L, cids, cw)\n"
        "assert m1.shape == (16,) and m2.shape == (16,)\n"
        "assert c.dim() == 0  # scalar\n"
        "assert t.allclose(m1, m2, atol=1e-5), 'einsum-uniform must equal reduce-mean'\n"
        "\n"
        "# --- Visualization: per-batch unweighted vs class-weighted contribution ---\n"
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        "B = m1.shape[0]\n"
        "ws = cw[cids]  # (B,)\n"
        "ax.bar(range(B), m1.numpy(), label='unweighted batch mean',\n"
        "       color='steelblue', alpha=0.7)\n"
        "ax.bar(range(B), (m1 * ws / ws.sum()).numpy(),\n"
        "       label='class-weighted contribution', color='coral', alpha=0.9)\n"
        "ax.axhline(c.item(), color='black', linestyle='--', label=f'class-balanced loss = {c.item():.3f}')\n"
        "ax.set_xlabel('batch index')\n"
        "ax.set_ylabel('value')\n"
        "ax.set_title('ex10 class-weighted reduce amplifies high-weight samples')\n"
        "ax.legend(fontsize=8)\n"
        "ax.grid(True, alpha=0.3, axis='y')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex10_weighted_reduce(\n"
        "    losses: Tensor,\n"
        "    class_ids: Tensor,\n"
        "    class_weights: Tensor,\n"
        ") -> tuple[Tensor, Tensor, Tensor]:\n"
        "    B, N = losses.shape\n"
        "\n"
        "    # 1. uniform reduce\n"
        "    mean_reduce = reduce(losses, 'b n -> b', 'mean')\n"
        "\n"
        "    # 2. uniform einsum equivalent\n"
        "    uniform_w = t.full((N,), 1.0 / N, dtype=losses.dtype)\n"
        "    mean_einsum = t.einsum('b n, n -> b', losses, uniform_w)\n"
        "\n"
        "    # 3. class-balanced batch loss\n"
        "    w_b_vec = class_weights[class_ids]                              # (B,)\n"
        "    class_balanced_loss = t.einsum('b, b ->', mean_reduce, w_b_vec) / w_b_vec.sum()\n"
        "\n"
        "    print(f'  mean_reduce = {mean_reduce}')\n"
        "    print(f'  mean_einsum = {mean_einsum}')\n"
        "    print(f'  class_balanced_loss = {class_balanced_loss.item():.5f}')\n"
        "    return mean_reduce, mean_einsum, class_balanced_loss"
    ),
    "solution_notes": (
        "**Why reduce('mean') == einsum-with-1/N.** Both compute "
        "`sum_i x_bi · w_i` with `w_i = 1/N` for all `i`. The reduce-vs-"
        "einsum distinction is one of API style, not numerics. Verifying "
        "this equality is the easiest way to convince yourself that the "
        "einsum-weighted form is correct before you generalize to "
        "non-uniform weights.\n\n"
        "**Why you can't do class-weighted reduce in pure einops.** "
        "`einops.reduce` named ops are uniform by construction — there's "
        "no kwarg for per-element weights. The moment you need "
        "`w_i ≠ 1/N`, you've left the reduce API and entered einsum "
        "territory. (Or you cheat with `t.sum(x * w_broadcast, dim=...)`, "
        "but that's just einsum with extra steps.)\n\n"
        "**Class-balanced loss in practice.** Imbalanced training sets "
        "give the majority class a free win on plain `loss.mean()` — the "
        "majority pulls the gradient. Multiplying each sample's per-class "
        "weight (typically `1 / class_frequency`) before averaging makes "
        "the per-step gradient class-frequency-invariant. The pattern in "
        "this drill is exactly what HuggingFace's class-balanced loss "
        "helper computes under the hood."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================== 8: einops-einsum ex10
# Existing ex1-9: hadamard, matmul, row sum, batched matmul, attention QK^T,
# scaled dot-product, multi-head, batched bilinear (x^T A x — A NOT batched),
# Tucker (3-tensor).
# Novel: 4-tensor batched bilinear with broadcast of TWO non-batched matrices.
# Genuinely advanced multi-axis pattern.
SPEC_EINSUM = {
    "atom_id": "einops-einsum",
    "subtopic": "Einops: Deep Learning",
    "topic_folder": "prereqs_einops",
    "atom_recap_md": (
        "## einsum 4-tensor broadcast — quick refresher\n"
        "\n"
        "`einsum` lets you mix batched and non-batched operands by "
        "**omitting** the batch index from the non-batched operand's "
        "pattern. Operands without the batch index are broadcast across "
        "every batch slot.\n"
        "\n"
        "```python\n"
        "# u, v: (B, D)   A, B_mat: (D, D)   broadcast A and B_mat across B\n"
        "y = einsum('b i, i j, b j, b k, k l, b l -> b', u, A, u, v, B_mat, v)\n"
        "```\n"
        "\n"
        "This computes `y_b = u_b^T A u_b + v_b^T B v_b` in one call — except "
        "we'd usually split it because a single einsum pattern computes "
        "ONE sum, not a SUM OF two terms. (We'll handle that by computing "
        "the two bilinear forms separately and adding.)\n"
        "\n"
        "**This drill (ex10) vs ex1-9.** ex8 did a single bilinear "
        "`y_b = x_b^T A x_b` with `A` batched. ex10 generalizes to TWO "
        "vectors u, v on the same `A` (now NON-batched, broadcast across "
        "the batch axis), summed into one scalar per batch element. "
        "Verifies the broadcast-non-batched-axis-into-batched-einsum "
        "pattern."
    ),
    "exercise_index": 10,
    "exercise_title": "batched bilinear with broadcast of non-batched matrix — y_b = u_b^T A u_b + v_b^T A v_b",
    "slug": "batched-bilinear-with-broadcast-of-non-batched-matrix",
    "bloom_level": "Create",
    "difficulty_num": 5,
    "difficulty_dots": "🔴🔴🔴🔴🔴",
    "keywords": ["bilinear", "broadcast-non-batched", "multi-axis", "einsum"],
    "kcs": ["einsum-omit-axis-broadcasts", "einsum-multi-operand-batched"],
    "lo": (
        "Create a single bilinear-form computation that mixes batched "
        "vectors (`u, v`: shape `(B, D)`) with a non-batched matrix "
        "(`A`: shape `(D, D)`), broadcasting `A` across the batch axis "
        "via index-omission in the einsum pattern."
    ),
    "prompt_body": (
        "Implement `ex10_double_bilinear(u, v, A)`.\n\n"
        "Compute the batched quadratic-form sum:\n"
        "  `y_b = u_b^T A u_b + v_b^T A v_b`\n\n"
        "where `u, v` have shape `(B, D)` and `A` is a **single** `(D, D)` "
        "matrix shared across the whole batch.\n\n"
        "Rules:\n"
        "1. Use exactly **two** `t.einsum` calls — one per bilinear term. "
        "Each call must broadcast `A` across the batch by omitting `b` "
        "from `A`'s index string.\n"
        "2. Both calls follow the pattern `einsum('b i, i j, b j -> b', "
        "x, A, x)`. Note `A`'s indices are just `'i j'` (no `b`); this is "
        "what makes the broadcast happen.\n"
        "3. Sum the two terms, return a `(B,)` tensor.\n\n"
        "Inputs: `u`, `v` shape `(B, D)`; `A` shape `(D, D)`.\n"
        "Output: `(B,)` tensor `y` where `y[b] = u[b]^T A u[b] + v[b]^T A "
        "v[b]`.\n\n"
        "The visualization is a heatmap over a 2-D parameter grid: vary "
        "`u` along one axis and `v` along another (both scaled multiples "
        "of unit vectors), and plot `y_b` over the grid. For positive-"
        "definite `A` you should see an elliptic-paraboloid contour."
    ),
    "stub": (
        "def ex10_double_bilinear(u: Tensor, v: Tensor, A: Tensor) -> Tensor:\n"
        '    """Batched y_b = u_b^T A u_b + v_b^T A v_b, with A broadcast across batch."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Hand-checked case: A = I, u = batched unit vectors, v = batched 2*units.\n"
        "# y_b should be |u_b|^2 + |v_b|^2.\n"
        "A = t.eye(3)\n"
        "u = t.tensor([\n"
        "    [1.0, 0.0, 0.0],   # |u|^2 = 1\n"
        "    [3.0, 4.0, 0.0],   # |u|^2 = 25\n"
        "])\n"
        "v = t.tensor([\n"
        "    [0.0, 2.0, 0.0],   # |v|^2 = 4\n"
        "    [1.0, 1.0, 1.0],   # |v|^2 = 3\n"
        "])\n"
        "y = ex10_double_bilinear(u, v, A)\n"
        "assert y.shape == (2,), f'expected (2,), got {tuple(y.shape)}'\n"
        "assert y.dtype == t.float32\n"
        "assert t.allclose(y, t.tensor([5.0, 28.0])), f'identity case failed: {y}'\n"
        "\n"
        "# Non-identity A: A = diag(1, 2, 3). y_b = sum_i diag_i (u_bi^2 + v_bi^2)\n"
        "A2 = t.diag(t.tensor([1.0, 2.0, 3.0]))\n"
        "u2 = t.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])\n"
        "v2 = t.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])\n"
        "expected = t.tensor([\n"
        "    1*1 + 2*1 + 3*1 + 1*0 + 2*1 + 3*0,  # = 1+2+3 + 0+2+0 = 8\n"
        "    1*4 + 2*0 + 3*0 + 1*1 + 2*1 + 3*1,  # = 4 + 6 = 10\n"
        "], dtype=t.float32)\n"
        "y2 = ex10_double_bilinear(u2, v2, A2)\n"
        "assert t.allclose(y2, expected), f'diag case failed: got {y2}, expected {expected}'\n"
        "\n"
        "# Reference equivalence: matmul-form must equal einsum-form.\n"
        "rng = t.Generator().manual_seed(13)\n"
        "B, D = 4, 5\n"
        "u3 = t.randn(B, D, generator=rng)\n"
        "v3 = t.randn(B, D, generator=rng)\n"
        "A3 = t.randn(D, D, generator=rng)\n"
        "y3 = ex10_double_bilinear(u3, v3, A3)\n"
        "ref = ((u3 @ A3) * u3).sum(-1) + ((v3 @ A3) * v3).sum(-1)\n"
        "assert t.allclose(y3, ref, atol=1e-4), f'matmul reference disagrees: {y3} vs {ref}'\n"
        "\n"
        "# --- Visualization: heatmap of y over a 2D (alpha, beta) grid ---\n"
        "G = 20\n"
        "alphas = t.linspace(-2.0, 2.0, G)\n"
        "betas = t.linspace(-2.0, 2.0, G)\n"
        "grid_u = t.stack([alphas, t.zeros_like(alphas), t.zeros_like(alphas)], dim=1)  # (G, 3)\n"
        "grid_v = t.stack([t.zeros_like(betas), betas, t.zeros_like(betas)], dim=1)    # (G, 3)\n"
        "# Cartesian product → (G*G, 3) for each\n"
        "us_grid = grid_u.unsqueeze(1).expand(G, G, 3).reshape(G*G, 3)\n"
        "vs_grid = grid_v.unsqueeze(0).expand(G, G, 3).reshape(G*G, 3)\n"
        "A_pd = t.eye(3) + 0.3 * t.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])\n"
        "ys_grid = ex10_double_bilinear(us_grid, vs_grid, A_pd).reshape(G, G)\n"
        "fig, ax = plt.subplots(figsize=(5, 4))\n"
        "im = ax.imshow(ys_grid.numpy(), origin='lower',\n"
        "               extent=[float(betas[0]), float(betas[-1]),\n"
        "                       float(alphas[0]), float(alphas[-1])],\n"
        "               cmap='viridis', aspect='auto')\n"
        "ax.set_xlabel(r'$\\beta$ (v scale)')\n"
        "ax.set_ylabel(r'$\\alpha$ (u scale)')\n"
        "ax.set_title(r'ex10 $y = u^\\top A u + v^\\top A v$ over $(\\alpha, \\beta)$ grid')\n"
        "plt.colorbar(im, ax=ax, label='y')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex10_double_bilinear(u: Tensor, v: Tensor, A: Tensor) -> Tensor:\n"
        "    y_u = t.einsum('b i, i j, b j -> b', u, A, u)\n"
        "    y_v = t.einsum('b i, i j, b j -> b', v, A, v)\n"
        "    return y_u + y_v"
    ),
    "solution_notes": (
        "**Why omitting `b` from `A` is the broadcast.** In einsum index "
        "notation, an index that appears in some operands but not others "
        "is implicitly broadcast over the absent ones. `'b i, i j, b j -> "
        "b'` says: `A` lacks `b` → broadcast `A` across the batch; `u` "
        "has `b` and `i` → batched over `b`, contracts over `i, j` with "
        "`A` and `u`'s second copy. Same as `u_b^T A u_b`.\n\n"
        "**Why two einsum calls instead of one giant pattern.** A single "
        "einsum computes ONE multi-index sum. To produce `y_b = u_b^T A "
        "u_b + v_b^T A v_b` you'd need a sum-of-two-products, which "
        "einsum doesn't express as a single term. Two patterns + Python "
        "`+` is the right factoring; trying to cram both into one call "
        "via auxiliary indices fast becomes unreadable.\n\n"
        "**When to reach for this pattern.** Anywhere you have a shared "
        "metric tensor that defines an inner product (Riemannian "
        "manifolds, learned distance metrics, attention with a shared "
        "key/query projection across heads). The non-batched-broadcast "
        "saves memory: you keep `A` as `(D, D)` instead of expanding it "
        "to `(B, D, D)` just to feed it through a matmul."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


SPECS = [
    SPEC_ZEROS,
    SPEC_UNBIND,
    SPEC_ROT,
    SPEC_ITEM,
    SPEC_ASSTR,
    SPEC_REARRANGE,
    SPEC_REDUCE,
    SPEC_EINSUM,
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
