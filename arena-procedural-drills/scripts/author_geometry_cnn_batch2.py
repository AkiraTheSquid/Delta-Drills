#!/usr/bin/env python3
"""Author 8 standalone Colab drills for ray-tracing + CNN-shape prereq atoms.

Targets atoms used heavily across ARENA chapter 0 parts 1 (ray tracing) and 2
(CNNs). Each notebook is brand-new (no prior split parents) under the new
`prereqs_geometry_cnn` topic folder.

Atom layout (8 exercises across 7 atoms):
  ray-parametric-form           — ex1, ex2  (2 exercises)
  triangle-barycentric          — ex1       (1)
  linalg-solve-batched          — ex1       (1)
  singular-matrix-mask-trick    — ex1       (1)
  conv-output-shape             — ex1       (1)
  conv-padding-zero             — ex1       (1)
  conv-windowing-1d             — ex1       (1)

Constraints (per Doughty ACE 2024 + Maier 2021):
  - One LO + one Bloom per exercise.
  - <= 2 concurrent KCs per exercise.
  - Solution body runs cleanly under test_body.
  - Conv drills stay focused on SHAPE arithmetic / padding mechanics
    (not the full conv operation — that belongs to as-strided-windowing).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_geometry_cnn"

# ─────────────────────────────────────────────────────────────────────────────
# Recap snippets — one per atom; reused across that atom's exercises.
# ─────────────────────────────────────────────────────────────────────────────

RECAP_RAY_PARAM = (
    "## Ray parametric form — quick refresher\n"
    "\n"
    "A ray in 3-D is specified by an **origin** `O` (a point) and a **direction** "
    "`D` (a vector). Every point on the ray is\n"
    "\n"
    "```\n"
    "R(u) = O + u * D,  u >= 0\n"
    "```\n"
    "\n"
    "**Anatomy:**\n"
    "- `u = 0` returns the origin.\n"
    "- `u > 0` walks forward along `D`.\n"
    "- `u < 0` would walk *backward* — by convention, real rays restrict `u >= 0`.\n"
    "\n"
    "**Storage convention used in ARENA.** A ray is a `(2, 3)` tensor: row 0 is "
    "`O`, row 1 is `D`. A batch of `B` rays is a `(B, 2, 3)` tensor. To evaluate "
    "`B` rays at a single scalar `u`, you compute `origin + u * direction` with "
    "ordinary broadcasting; to evaluate one ray at `M` parameter values, you "
    "broadcast `(M,)` against `(3,)`.\n"
    "\n"
    "**Why `D` is not required to be unit-length.** If `||D|| != 1` then `u` is "
    "not a metric distance — it's a parameter along `D`. ARENA leaves `D` "
    "un-normalized everywhere, so `u` is dimensionless."
)

RECAP_TRIANGLE_BARY = (
    "## Triangle barycentric coordinates — quick refresher\n"
    "\n"
    "Any point `P` in the plane of triangle `ABC` can be written\n"
    "\n"
    "```\n"
    "P = A + u * (B - A) + v * (C - A)\n"
    "```\n"
    "\n"
    "where `(u, v)` are the **barycentric coordinates** of `P` w.r.t. the edge "
    "basis `(B - A, C - A)`. `P` lies *inside* the triangle iff\n"
    "\n"
    "```\n"
    "u >= 0  AND  v >= 0  AND  u + v <= 1\n"
    "```\n"
    "\n"
    "(Equality on any bound puts `P` on an edge or vertex.) The third coordinate "
    "`w = 1 - u - v` is implied; together `(w, u, v)` are the conventional "
    "weights on `(A, B, C)`.\n"
    "\n"
    "**Why ARENA's ray-triangle intersection cares.** Plugging the ray "
    "`R(s) = O + s * D` into the triangle's plane equation gives a 3x3 linear "
    "system whose solution is exactly `(s, u, v)`. Once you have it, the "
    "intersection test is the three inequalities above."
)

RECAP_LINALG_SOLVE = (
    "## `t.linalg.solve` — batched form\n"
    "\n"
    "`t.linalg.solve(A, b)` solves `A x = b` for `x`. It accepts **arbitrary "
    "leading batch dimensions** as long as the last two axes of `A` are square:\n"
    "\n"
    "- `A: (..., n, n)` and `b: (..., n)` → `x: (..., n)`\n"
    "- `A: (..., n, n)` and `b: (..., n, k)` → `x: (..., n, k)` (k right-hand sides)\n"
    "\n"
    "Internally it factors each `(n, n)` slice once and substitutes — vastly "
    "faster than a Python loop over `t.linalg.inv` + matmul, and numerically "
    "better-behaved.\n"
    "\n"
    "**Failure mode.** If any leading slice's `A` is singular (or even very "
    "close to it), the call raises `LinAlgError`. The standard workaround is "
    "the *singular-matrix-mask trick*: detect singular slices via "
    "`t.linalg.det(A).abs() < eps`, overwrite them with the identity so the "
    "solve succeeds, then mask their results out of the final answer."
)

RECAP_SINGULAR_MASK = (
    "## Singular-matrix mask trick — quick refresher\n"
    "\n"
    "`t.linalg.solve(A, b)` raises if **any** slice of `A` is singular. In a "
    "batched setting that's often unacceptable — a single bad slice should not "
    "kill the entire solve. The standard workaround:\n"
    "\n"
    "1. **Detect** singular slices: `dets = t.linalg.det(A); is_singular = dets.abs() < eps`.\n"
    "2. **Mask in** the identity matrix at those slices: `A[is_singular] = t.eye(n)`. "
    "Now `solve` succeeds everywhere (the identity slices return `b` unchanged).\n"
    "3. **Mask out** the spurious results from the final boolean predicate: "
    "`valid = predicate & ~is_singular`.\n"
    "\n"
    "**Why it works.** The identity overwrite is purely cosmetic — we never "
    "trust those entries, we just need solve to not crash. The mask in step 3 "
    "removes them from the answer.\n"
    "\n"
    "**Watch out.** `A[is_singular] = t.eye(n)` relies on broadcasting: the "
    "`(n, n)` identity broadcasts to every selected slice. Confirm shapes "
    "before using; for shapes other than the last two dims you may need "
    "`A[is_singular] = t.eye(n).expand_as(A[is_singular])`."
)

RECAP_CONV_OUTSHAPE = (
    "## Conv output shape — the formula\n"
    "\n"
    "For a 2-D convolution with input `(B, IC, H, W)`, kernel `(OC, IC, KH, KW)`, "
    "**stride** `(SH, SW)`, **padding** `(PH, PW)`, and **dilation** `(DH, DW)`:\n"
    "\n"
    "```\n"
    "H_out = floor( (H + 2*PH - DH*(KH - 1) - 1) / SH ) + 1\n"
    "W_out = floor( (W + 2*PW - DW*(KW - 1) - 1) / SW ) + 1\n"
    "```\n"
    "\n"
    "For the common case `dilation=1`, this simplifies to\n"
    "\n"
    "```\n"
    "H_out = floor( (H + 2*PH - KH) / SH ) + 1\n"
    "W_out = floor( (W + 2*PW - KW) / SW ) + 1\n"
    "```\n"
    "\n"
    "**The shape rule.** Output is `(B, OC, H_out, W_out)` — the batch and "
    "kernel-output-channels axes pass through unchanged; `IC` is *contracted "
    "away* by the convolution; `H` and `W` shrink per the formula.\n"
    "\n"
    "**Special cases worth memorizing:**\n"
    "- `padding = (KH-1)//2` with stride 1, odd KH → `H_out = H` (\"same\" padding).\n"
    "- `padding = 0`, stride 1 → `H_out = H - KH + 1` (the minimal no-pad form ARENA's `conv1d_minimal` uses).\n"
    "- `stride = KH`, padding 0 → `H_out = H // KH` (non-overlapping tiles)."
)

RECAP_CONV_PAD = (
    "## Zero-padding a conv input — quick refresher\n"
    "\n"
    "Convolution shrinks spatial dimensions. To preserve them (or to give the "
    "kernel something to multiply against at the boundary), we **pad** the "
    "input with zeros before convolving.\n"
    "\n"
    "**1-D form.** Given `x: (B, IC, W)` and pad amounts `left, right`:\n"
    "```\n"
    "x_padded.shape == (B, IC, left + W + right)\n"
    "x_padded[:, :, left : left + W] == x\n"
    "x_padded[:, :, :left]            == 0\n"
    "x_padded[:, :, left + W:]        == 0\n"
    "```\n"
    "\n"
    "**Implementation idioms.**\n"
    "- Manual: `x_padded = x.new_full((B, IC, left + W + right), 0.0); x_padded[:, :, left:left+W] = x`.\n"
    "- Functional: `F.pad(x, (left, right))` — last-axis-first ordering, careful.\n"
    "\n"
    "**Why zero specifically.** Zero is the **additive identity** for the "
    "kernel's dot product — padded cells contribute nothing to the output. "
    "For *max-pool*, by contrast, padding must be `-inf` (the additive identity "
    "of max), not zero, or padded cells will spuriously win the max."
)

RECAP_CONV_WIN_1D = (
    "## 1-D conv windowing via `as_strided` — quick refresher\n"
    "\n"
    "A 1-D convolution can be expressed as **two separate steps**:\n"
    "\n"
    "1. **Window** the input into a strided view of shape "
    "`(B, IC, OW, KW)` — each `(KW,)` slice along the new `OW` axis is one "
    "kernel-sized window of the original input. The windows overlap.\n"
    "2. **Einsum** the window against the kernel: "
    "`einops.einsum(x_strided, weight, 'b ic ow kw, oc ic kw -> b oc ow')`.\n"
    "\n"
    "The windowing step is the load-bearing trick. For input strides "
    "`(s_b, s_ic, s_w)` and a window count `OW = W - KW + 1` (stride-1 case):\n"
    "\n"
    "```\n"
    "x_strided = x.as_strided(\n"
    "    size=(B, IC, OW, KW),\n"
    "    stride=(s_b, s_ic, s_w, s_w),  # last two strides are EQUAL: s_w\n"
    ")\n"
    "```\n"
    "\n"
    "The trailing `s_w` on the new `OW` axis means \"advance by one element of "
    "the original W axis when you move to the next window\" — i.e., adjacent "
    "windows overlap by `KW - 1` elements. The trailing `s_w` on `KW` walks "
    "*within* a window. **No data is copied** — `x_strided` is a view into "
    "the same storage as `x`.\n"
    "\n"
    "**Equivalence check.** The result must agree with `F.conv1d(x, weight)` "
    "to floating-point tolerance."
)


SPECS = [
    # ═══════════════════════════════════════════════════════════════════════
    # ray-parametric-form  (2 exercises)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "ray-parametric-form",
        "subtopic": "Geometry: Ray parametric form",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_RAY_PARAM,
        "exercise_index": 1,
        "exercise_title": "evaluate one ray at many parameter values",
        "slug": "evaluate-one-ray-at-many-parameter-values",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["ray", "parametric", "broadcasting", "visualization"],
        "kcs": ["ray-eval-broadcast-u", "ray-origin-direction-storage"],
        "lo": (
            "Apply the parametric ray equation `R(u) = O + u * D` to evaluate "
            "a single ray at a 1-D batch of parameter values, returning a "
            "tensor of points along the ray."
        ),
        "prompt_body": (
            "Implement `ex1_eval_ray(ray, us)`.\n\n"
            "- `ray` has shape `(2, 3)` — row 0 is the origin `O`, row 1 is the "
            "direction `D`.\n"
            "- `us` has shape `(M,)` — the parameter values to evaluate at.\n"
            "- Return shape `(M, 3)` where row `m` is `O + us[m] * D`.\n\n"
            "**Hint.** Unpack the ray with one of the patterns you've seen "
            "(`ray[0], ray[1]` or `ray.unbind(dim=0)`). Then broadcast `us` "
            "against `D`: `us` is `(M,)`, `D` is `(3,)`, so reshape `us` to "
            "`(M, 1)` before multiplying so it lines up against the trailing 3.\n\n"
            "The visualization plots the projected (x, z) trajectory of the "
            "ray points so you can verify the line goes through the origin and "
            "extends along `D`."
        ),
        "stub": (
            "def ex1_eval_ray(ray: Tensor, us: Tensor) -> Tensor:\n"
            '    """Evaluate R(u) = O + u*D at each u in `us`. Returns (M, 3)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Ray from (1, 0, 0) in direction (0, 0, 1) — walks along +z.\n"
            "ray = t.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])\n"
            "us = t.tensor([0.0, 1.0, 2.0, 3.0])\n"
            "out = ex1_eval_ray(ray, us)\n"
            "expected = t.tensor([\n"
            "    [1.0, 0.0, 0.0],\n"
            "    [1.0, 0.0, 1.0],\n"
            "    [1.0, 0.0, 2.0],\n"
            "    [1.0, 0.0, 3.0],\n"
            "])\n"
            "assert out.shape == (4, 3), f'expected (4,3), got {tuple(out.shape)}'\n"
            "assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "assert t.allclose(out, expected, atol=1e-6), f'value mismatch:\\n{out}\\nvs\\n{expected}'\n"
            "\n"
            "# u=0 must return the origin.\n"
            "ray2 = t.tensor([[2.0, -1.0, 4.0], [3.0, 0.5, -2.0]])\n"
            "out2 = ex1_eval_ray(ray2, t.tensor([0.0]))\n"
            "assert t.allclose(out2, ray2[0:1], atol=1e-6), 'R(0) must equal O'\n"
            "\n"
            "# Non-unit direction is allowed (parameter is dimensionless, not metric).\n"
            "out3 = ex1_eval_ray(ray2, t.tensor([1.0]))\n"
            "assert t.allclose(out3[0], ray2[0] + ray2[1], atol=1e-6), 'R(1) = O + D'\n"
            "\n"
            "# Large sweep — value test\n"
            "ray4 = t.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])\n"
            "us4 = t.linspace(-2, 2, 21)\n"
            "out4 = ex1_eval_ray(ray4, us4)\n"
            "assert out4.shape == (21, 3)\n"
            "assert t.allclose(out4[:, 0], us4, atol=1e-6), 'x should equal u for D=(1,1,0)'\n"
            "assert t.allclose(out4[:, 1], us4, atol=1e-6), 'y should equal u for D=(1,1,0)'\n"
            "assert t.allclose(out4[:, 2], t.zeros(21), atol=1e-6), 'z stays 0'\n"
            "\n"
            "# --- Visualization: the ray as a line of points in (x, z) ---\n"
            "vis_ray = t.tensor([[1.0, 0.0, -1.0], [0.5, 0.0, 0.8]])\n"
            "vis_us = t.linspace(0, 4, 30)\n"
            "vis_pts = ex1_eval_ray(vis_ray, vis_us)\n"
            "fig, ax = plt.subplots(figsize=(5, 5))\n"
            "ax.scatter(vis_pts[:, 0].numpy(), vis_pts[:, 2].numpy(),\n"
            "           c=vis_us.numpy(), cmap='viridis', s=20)\n"
            "ax.scatter([vis_ray[0, 0].item()], [vis_ray[0, 2].item()],\n"
            "           c='red', s=80, marker='*', label='origin O')\n"
            "ax.set_xlabel('x')\n"
            "ax.set_ylabel('z')\n"
            "ax.set_title('ex1 ray points (color = u)')\n"
            "ax.set_aspect('equal')\n"
            "ax.grid(True, alpha=0.3)\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex1_eval_ray(ray: Tensor, us: Tensor) -> Tensor:\n"
            "    O, D = ray[0], ray[1]              # each (3,)\n"
            "    return O + us.unsqueeze(-1) * D    # (M,1) * (3,) -> (M,3)"
        ),
        "solution_notes": (
            "**Why `unsqueeze(-1)`.** Bare `us * D` would try to broadcast "
            "`(M,)` against `(3,)`, which only works when `M == 3` — a "
            "silent bug. Adding a trailing size-1 axis (`(M, 1)`) lines up "
            "explicitly against the trailing 3 of `D`.\n\n"
            "**Why `O + ...` works without unsqueezing `O`.** Broadcasting "
            "lines up trailing dims: `(M, 3) + (3,) → (M, 3)`. `O` is "
            "automatically replicated across the leading `M` axis.\n\n"
            "**Equivalent forms.** `O + t.outer(us, D)` also works and is "
            "arguably clearer about the rank-1 outer-product structure. Both "
            "are O(M*3) flops; pick whichever reads better in context."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ───────────────────────────────────────────────────────────────────────
    {
        "atom_id": "ray-parametric-form",
        "subtopic": "Geometry: Ray parametric form",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_RAY_PARAM,
        "exercise_index": 2,
        "exercise_title": "evaluate a batch of rays at one parameter",
        "slug": "evaluate-a-batch-of-rays-at-one-parameter",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["ray", "batched", "parametric", "visualization"],
        "kcs": ["ray-eval-broadcast-batch", "ray-origin-direction-storage"],
        "lo": (
            "Apply the parametric ray equation `R(u) = O + u * D` across a "
            "batch of rays at a single scalar parameter `u`, returning one "
            "endpoint per ray."
        ),
        "prompt_body": (
            "Implement `ex2_eval_ray_batch(rays, u)`.\n\n"
            "- `rays` has shape `(B, 2, 3)` — row 0 of each `(2, 3)` block is "
            "`O_b`, row 1 is `D_b`.\n"
            "- `u` is a Python `float`.\n"
            "- Return shape `(B, 3)`: the point `O_b + u * D_b` for each ray.\n\n"
            "**Hint.** Slice off origins and directions with `rays[:, 0]` and "
            "`rays[:, 1]` (each `(B, 3)`). Then the arithmetic is plain "
            "broadcasting — no `unsqueeze` needed because `u` is scalar.\n\n"
            "The visualization plots all `B` endpoints in the X-Z plane next "
            "to their origins, drawing the connecting segments so you can see "
            "the fan of rays."
        ),
        "stub": (
            "def ex2_eval_ray_batch(rays: Tensor, u: float) -> Tensor:\n"
            '    """Evaluate each ray at the same scalar u. Returns (B, 3)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rays = t.tensor([\n"
            "    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # ray 0: from origin along +x\n"
            "    [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],   # ray 1: from (1,1,0) along +y\n"
            "    [[2.0, 0.0, 1.0], [0.0, 0.0, -1.0]],  # ray 2: from (2,0,1) along -z\n"
            "])\n"
            "out = ex2_eval_ray_batch(rays, 2.0)\n"
            "expected = t.tensor([\n"
            "    [2.0, 0.0, 0.0],\n"
            "    [1.0, 3.0, 0.0],\n"
            "    [2.0, 0.0, -1.0],\n"
            "])\n"
            "assert out.shape == (3, 3), f'expected (3,3), got {tuple(out.shape)}'\n"
            "assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "assert t.allclose(out, expected, atol=1e-6), f'value mismatch:\\n{out}\\nvs\\n{expected}'\n"
            "\n"
            "# u=0 → all origins.\n"
            "out0 = ex2_eval_ray_batch(rays, 0.0)\n"
            "assert t.allclose(out0, rays[:, 0], atol=1e-6), 'u=0 must return origins'\n"
            "\n"
            "# u=1 → O + D.\n"
            "out1 = ex2_eval_ray_batch(rays, 1.0)\n"
            "assert t.allclose(out1, rays[:, 0] + rays[:, 1], atol=1e-6), 'u=1 must return O+D'\n"
            "\n"
            "# Larger fan of rays for the viz.\n"
            "rng = t.Generator().manual_seed(7)\n"
            "B = 40\n"
            "angles = t.linspace(0, t.pi, B)\n"
            "origins = t.zeros(B, 3)\n"
            "directions = t.stack([t.cos(angles), t.zeros(B), t.sin(angles)], dim=1)\n"
            "fan_rays = t.stack([origins, directions], dim=1)\n"
            "out_fan = ex2_eval_ray_batch(fan_rays, 1.5)\n"
            "assert out_fan.shape == (B, 3)\n"
            "# Endpoints must lie on a circle of radius 1.5 (since |D|=1 here, u=1.5).\n"
            "radii = (out_fan[:, 0]**2 + out_fan[:, 2]**2).sqrt()\n"
            "assert t.allclose(radii, t.full((B,), 1.5), atol=1e-5), 'fan endpoints should be radius 1.5'\n"
            "\n"
            "# --- Visualization: ray fan in (x, z) plane ---\n"
            "fig, ax = plt.subplots(figsize=(5, 5))\n"
            "for b in range(B):\n"
            "    ax.plot(\n"
            "        [origins[b, 0].item(), out_fan[b, 0].item()],\n"
            "        [origins[b, 2].item(), out_fan[b, 2].item()],\n"
            "        c='steelblue', alpha=0.4, linewidth=0.8,\n"
            "    )\n"
            "ax.scatter(out_fan[:, 0].numpy(), out_fan[:, 2].numpy(), c='crimson', s=15, label='R(1.5)')\n"
            "ax.scatter([0], [0], c='black', s=50, marker='*', label='shared origin')\n"
            "ax.set_xlabel('x')\n"
            "ax.set_ylabel('z')\n"
            "ax.set_title(f'ex2 fan of {B} rays evaluated at u=1.5')\n"
            "ax.set_aspect('equal')\n"
            "ax.grid(True, alpha=0.3)\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex2_eval_ray_batch(rays: Tensor, u: float) -> Tensor:\n"
            "    O = rays[:, 0]   # (B, 3)\n"
            "    D = rays[:, 1]   # (B, 3)\n"
            "    return O + u * D"
        ),
        "solution_notes": (
            "**Why no `unsqueeze` here.** `u` is a Python scalar — multiplying "
            "a tensor by it is rank-preserving (`(B, 3) * scalar → (B, 3)`). "
            "Compare with ex1, where the parameter sweep was a `(M,)` tensor "
            "and we had to add an axis to align.\n\n"
            "**Slicing vs unbind.** `rays[:, 0]` and `rays[:, 1]` create views "
            "(no copy). `O, D = rays.unbind(dim=1)` is equivalent and arguably "
            "cleaner — it makes the two-row decomposition explicit. Both "
            "compile to the same arithmetic.\n\n"
            "**Generalizing.** To allow per-ray parameters (an `(B,)` tensor "
            "of `u` values instead of one scalar), you'd write "
            "`O + us.unsqueeze(-1) * D` — the same `(B, 1) * (B, 3)` trick "
            "from ex1, just with the batch axis playing the role of `M`."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ═══════════════════════════════════════════════════════════════════════
    # triangle-barycentric (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "triangle-barycentric",
        "subtopic": "Geometry: Barycentric coords",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_TRIANGLE_BARY,
        "exercise_index": 1,
        "exercise_title": "point-in-triangle test from (u, v)",
        "slug": "point-in-triangle-test-from-uv",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["triangle", "barycentric", "predicate", "visualization"],
        "kcs": ["barycentric-inside-predicate", "barycentric-edge-basis"],
        "lo": (
            "Apply the barycentric inside-triangle predicate "
            "`u >= 0 & v >= 0 & u + v <= 1` to a batch of `(u, v)` coordinate "
            "pairs and return one boolean per point."
        ),
        "prompt_body": (
            "Implement `ex1_inside_triangle(uvs)`. Given barycentric coordinates "
            "`uvs` of shape `(N, 2)` (column 0 is `u`, column 1 is `v`), return "
            "a boolean tensor of shape `(N,)` that is `True` exactly when the "
            "point `P = A + u*(B-A) + v*(C-A)` lies inside (or on the boundary "
            "of) the triangle `ABC`.\n\n"
            "**Hint.** Three predicates ANDed together: `u >= 0`, `v >= 0`, "
            "`u + v <= 1`. No matrix solve, no projection — this drill is "
            "purely the inside test in barycentric space.\n\n"
            "The visualization scatters the input points colored by inside/"
            "outside and overlays the canonical triangle `(0,0), (1,0), (0,1)` "
            "in `(u, v)` space so you can see the predicate boundary."
        ),
        "stub": (
            "def ex1_inside_triangle(uvs: Tensor) -> Tensor:\n"
            '    """True where (u, v) is inside the unit barycentric triangle."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Canonical points: centroid (1/3, 1/3) inside, a vertex (1, 0)\n"
            "# on the boundary, (0.6, 0.6) outside (u+v=1.2>1), (-0.1, 0.5) outside (u<0).\n"
            "uvs = t.tensor([\n"
            "    [1/3, 1/3],   # centroid → inside\n"
            "    [0.0, 0.0],   # vertex A → inside (boundary)\n"
            "    [1.0, 0.0],   # vertex B → inside (boundary)\n"
            "    [0.0, 1.0],   # vertex C → inside (boundary)\n"
            "    [0.5, 0.5],   # midpoint of BC → inside (boundary, u+v=1)\n"
            "    [0.6, 0.6],   # u+v=1.2 > 1 → outside\n"
            "    [-0.1, 0.5],  # u < 0 → outside\n"
            "    [0.5, -0.1],  # v < 0 → outside\n"
            "    [2.0, 2.0],   # way outside\n"
            "])\n"
            "out = ex1_inside_triangle(uvs)\n"
            "expected = t.tensor([True, True, True, True, True, False, False, False, False])\n"
            "assert out.shape == (9,), f'expected (9,), got {tuple(out.shape)}'\n"
            "assert out.dtype == t.bool, f'expected bool, got {out.dtype}'\n"
            "assert t.equal(out, expected), f'predicate wrong:\\n{out}\\nvs\\n{expected}'\n"
            "\n"
            "# Independent ground truth: matplotlib's path.contains_points uses ray-casting,\n"
            "# a completely different algorithm than the (u, v) inequality predicate.\n"
            "# Open-boundary points are excluded by contains_points, so we keep interior\n"
            "# points only (>= 1e-6 inside every edge) to avoid boundary-tie noise.\n"
            "from matplotlib.path import Path as _MplPath\n"
            "rng = t.Generator().manual_seed(13)\n"
            "rand_uvs = t.rand(500, 2, generator=rng) * 2 - 0.5   # range [-0.5, 1.5]\n"
            "interior_mask = (rand_uvs[:, 0] > 1e-6) & (rand_uvs[:, 1] > 1e-6) & (rand_uvs[:, 0] + rand_uvs[:, 1] < 1 - 1e-6)\n"
            "exterior_mask = (rand_uvs[:, 0] < -1e-6) | (rand_uvs[:, 1] < -1e-6) | (rand_uvs[:, 0] + rand_uvs[:, 1] > 1 + 1e-6)\n"
            "safe = interior_mask | exterior_mask  # drop near-boundary to avoid algorithm-tie noise\n"
            "rand_out = ex1_inside_triangle(rand_uvs)\n"
            "tri_path = _MplPath([(0, 0), (1, 0), (0, 1)])\n"
            "mpl_inside = t.tensor(tri_path.contains_points(rand_uvs.numpy()))\n"
            "assert t.equal(rand_out[safe], mpl_inside[safe]), 'must match matplotlib ray-cast ground truth'\n"
            "\n"
            "# --- Visualization: inside vs outside in (u, v) space ---\n"
            "fig, ax = plt.subplots(figsize=(5, 5))\n"
            "inside = rand_out.numpy()\n"
            "ax.scatter(rand_uvs[inside, 0].numpy(), rand_uvs[inside, 1].numpy(),\n"
            "           c='seagreen', s=12, alpha=0.6, label='inside')\n"
            "ax.scatter(rand_uvs[~inside, 0].numpy(), rand_uvs[~inside, 1].numpy(),\n"
            "           c='lightgrey', s=12, alpha=0.6, label='outside')\n"
            "# Draw the canonical triangle boundary.\n"
            "tri = [[0, 0], [1, 0], [0, 1], [0, 0]]\n"
            "tri_x = [p[0] for p in tri]; tri_y = [p[1] for p in tri]\n"
            "ax.plot(tri_x, tri_y, 'k-', linewidth=2)\n"
            "ax.set_xlabel('u')\n"
            "ax.set_ylabel('v')\n"
            "ax.set_title('ex1 inside-triangle predicate in (u, v) space')\n"
            "ax.set_aspect('equal')\n"
            "ax.legend(loc='upper right')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex1_inside_triangle(uvs: Tensor) -> Tensor:\n"
            "    u = uvs[:, 0]\n"
            "    v = uvs[:, 1]\n"
            "    return (u >= 0) & (v >= 0) & (u + v <= 1)"
        ),
        "solution_notes": (
            "**Why three predicates, not four.** A 2-simplex has three edges "
            "(u=0, v=0, u+v=1). Each edge contributes one inequality; the "
            "interior is the intersection. The third coordinate `w = 1-u-v` "
            "is redundant — the constraint `w >= 0` is exactly `u + v <= 1`.\n\n"
            "**Boundary handling.** Using `>=` and `<=` (not strict) makes "
            "edge/vertex points count as inside. This matches ARENA's "
            "`triangle_ray_intersects` convention; some renderers use strict "
            "inequalities to avoid double-counting shared edges.\n\n"
            "**This drill skips the solve.** The full ray-triangle intersection "
            "(Möller-Trumbore) first solves a 3x3 system for `(s, u, v)`, then "
            "applies this predicate plus `s >= 0`. Splitting the predicate from "
            "the solve lets each subskill be drilled independently."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ═══════════════════════════════════════════════════════════════════════
    # linalg-solve-batched (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "linalg-solve-batched",
        "subtopic": "PyTorch: Batched linalg.solve",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_LINALG_SOLVE,
        "exercise_index": 1,
        "exercise_title": "solve a batch of 2x2 systems",
        "slug": "solve-a-batch-of-2x2-systems",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["linalg", "solve", "batched", "shape"],
        "kcs": ["linalg-solve-leading-batch", "linalg-solve-shape-contract"],
        "lo": (
            "Apply `t.linalg.solve` with leading batch dimensions to solve "
            "`K` independent 2x2 systems in a single call, returning a "
            "`(K, 2)` stack of solutions."
        ),
        "prompt_body": (
            "Implement `ex1_batched_solve_2x2(A, b)`.\n\n"
            "- `A` has shape `(K, 2, 2)` — `K` square coefficient matrices.\n"
            "- `b` has shape `(K, 2)` — `K` right-hand-side vectors.\n"
            "- Return shape `(K, 2)`: the `k`th row is the solution `x_k` "
            "to `A[k] @ x_k == b[k]`.\n\n"
            "**Hint.** This is *exactly* what `t.linalg.solve` does — pass "
            "`A` and `b` as-is. No loops, no per-slice indexing. The output "
            "shape mirrors `b`.\n\n"
            "Assume all matrices are non-singular (the singular-matrix-mask "
            "trick is a separate drill).\n\n"
            "After solving, the test verifies by recomputing `A @ x` and "
            "comparing against `b` to floating-point tolerance."
        ),
        "stub": (
            "def ex1_batched_solve_2x2(A: Tensor, b: Tensor) -> Tensor:\n"
            '    """Solve A_k x_k = b_k for k = 0..K-1 in one shot."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-picked: two 2x2 systems with known solutions.\n"
            "A = t.tensor([\n"
            "    [[1.0, 0.0],\n"
            "     [0.0, 1.0]],   # identity → x = b\n"
            "    [[2.0, 1.0],\n"
            "     [1.0, 3.0]],   # det = 5\n"
            "])\n"
            "b = t.tensor([\n"
            "    [3.0, -1.0],\n"
            "    [5.0,  5.0],\n"
            "])\n"
            "x = ex1_batched_solve_2x2(A, b)\n"
            "assert x.shape == (2, 2), f'expected (2,2), got {tuple(x.shape)}'\n"
            "assert x.dtype == t.float32, f'expected float32, got {x.dtype}'\n"
            "# System 0: identity, solution is b itself.\n"
            "assert t.allclose(x[0], t.tensor([3.0, -1.0]), atol=1e-5)\n"
            "# System 1: 2*x0 + x1 = 5; x0 + 3*x1 = 5 → x0=2, x1=1.\n"
            "assert t.allclose(x[1], t.tensor([2.0, 1.0]), atol=1e-5), f'x1 wrong: {x[1]}'\n"
            "# Reconstruction must match b.\n"
            "recon = t.einsum('kij,kj->ki', A, x)\n"
            "assert t.allclose(recon, b, atol=1e-5), f'A@x != b:\\n{recon}\\nvs\\n{b}'\n"
            "\n"
            "# Random K=64 batch of non-singular 2x2 systems.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "K = 64\n"
            "# Build A as I + small perturbation so dets stay well away from 0.\n"
            "A_big = t.eye(2).expand(K, 2, 2) + 0.3 * t.randn(K, 2, 2, generator=rng)\n"
            "b_big = t.randn(K, 2, generator=rng)\n"
            "x_big = ex1_batched_solve_2x2(A_big, b_big)\n"
            "assert x_big.shape == (K, 2)\n"
            "recon_big = t.einsum('kij,kj->ki', A_big, x_big)\n"
            "assert t.allclose(recon_big, b_big, atol=1e-4), 'A @ x must reconstruct b across the batch'\n"
            "\n"
            "# Sanity: each row matches the unbatched single-system solve.\n"
            "for k in [0, 17, 63]:\n"
            "    single = t.linalg.solve(A_big[k], b_big[k])\n"
            "    assert t.allclose(x_big[k], single, atol=1e-5), f'row {k} disagrees with single-solve'"
        ),
        "solution_body": (
            "def ex1_batched_solve_2x2(A: Tensor, b: Tensor) -> Tensor:\n"
            "    return t.linalg.solve(A, b)"
        ),
        "solution_notes": (
            "**The shape contract.** `t.linalg.solve(A, b)` with "
            "`A: (..., n, n)` and `b: (..., n)` returns `x: (..., n)`. The "
            "leading dims of `A` and `b` must agree (or be broadcastable); "
            "the last two of `A` must be square and equal to `b`'s last dim.\n\n"
            "**Why not loop.** A Python loop over `K` would re-dispatch to "
            "BLAS once per system, paying per-call overhead. The batched call "
            "fuses the LU factorizations into a single C-level loop — for "
            "`K=64` already 10x+ faster, and the gap widens with larger `K`.\n\n"
            "**Multiple right-hand sides.** If `b` has shape `(..., n, m)` "
            "instead of `(..., n)`, you get one solution column per "
            "`m`-slice — same factorization, different substitutions. Useful "
            "for problems like \"compute the inverse\" (set `b = I`).\n\n"
            "**Singular slices crash the whole call.** That's the failure mode "
            "the `singular-matrix-mask-trick` drill addresses."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # singular-matrix-mask-trick (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "singular-matrix-mask-trick",
        "subtopic": "Numpy: Singular matrix mask trick",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_SINGULAR_MASK,
        "exercise_index": 1,
        "exercise_title": "solve a batch with some singular matrices",
        "slug": "solve-a-batch-with-some-singular-matrices",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["singular", "linalg", "mask", "boolean-index"],
        "kcs": ["singular-detect-via-det", "singular-overwrite-identity"],
        "lo": (
            "Apply the singular-matrix-mask trick: detect singular slices via "
            "`t.linalg.det`, overwrite them with the identity, run the solve "
            "without crashing, and return per-slice `(solution, is_valid)` "
            "where `is_valid` flags non-singular slices."
        ),
        "prompt_body": (
            "Implement `ex1_solve_with_singular_mask(A, b, eps=1e-8)`.\n\n"
            "- `A` has shape `(K, n, n)`. **Some slices may be singular.**\n"
            "- `b` has shape `(K, n)`.\n"
            "- `eps` is the singular-detection threshold on `|det|`.\n\n"
            "Return `(x, is_valid)`:\n"
            "- `x: (K, n)` — the solution at each slice. For singular slices "
            "this entry is undefined (we just need the solve not to crash).\n"
            "- `is_valid: (K,) bool` — `True` where the original `A[k]` was "
            "non-singular.\n\n"
            "**Algorithm.**\n"
            "1. `dets = t.linalg.det(A)` then `is_singular = dets.abs() < eps`.\n"
            "2. **Clone** `A` (don't mutate the input!) and overwrite singular "
            "slices with `t.eye(n)`.\n"
            "3. `x = t.linalg.solve(A_safe, b)`.\n"
            "4. Return `x`, `~is_singular`.\n\n"
            "The grader explicitly constructs a batch with two known singular "
            "matrices and verifies you mask them correctly while still solving "
            "the well-conditioned ones."
        ),
        "stub": (
            "def ex1_solve_with_singular_mask(A: Tensor, b: Tensor, eps: float = 1e-8):\n"
            '    """Returns (x, is_valid). Singular slices flagged in is_valid=False."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "n = 2\n"
            "# Build batch: 2 good 2x2 systems + 2 singular ones (det = 0).\n"
            "A = t.stack([\n"
            "    t.tensor([[1.0, 0.0], [0.0, 1.0]]),     # identity, det=1, x=b\n"
            "    t.tensor([[2.0, 1.0], [1.0, 3.0]]),     # det=5, well-conditioned\n"
            "    t.tensor([[1.0, 2.0], [2.0, 4.0]]),     # SINGULAR (rows are 1:2 proportional)\n"
            "    t.tensor([[0.0, 0.0], [3.0, 1.0]]),     # SINGULAR (first row all zero)\n"
            "])\n"
            "b = t.tensor([\n"
            "    [3.0, -1.0],\n"
            "    [5.0,  5.0],\n"
            "    [9.0, 18.0],\n"
            "    [4.0,  2.0],\n"
            "])\n"
            "A_before = A.clone()  # to confirm we didn't mutate input\n"
            "x, is_valid = ex1_solve_with_singular_mask(A, b)\n"
            "\n"
            "# Did NOT mutate input.\n"
            "assert t.equal(A, A_before), 'must not mutate the input A in place'\n"
            "\n"
            "# Shapes / dtypes.\n"
            "assert x.shape == (4, 2), f'x: expected (4,2), got {tuple(x.shape)}'\n"
            "assert is_valid.shape == (4,), f'is_valid: expected (4,), got {tuple(is_valid.shape)}'\n"
            "assert is_valid.dtype == t.bool\n"
            "\n"
            "# Validity mask: slices 0 and 1 valid, 2 and 3 singular.\n"
            "expected_valid = t.tensor([True, True, False, False])\n"
            "assert t.equal(is_valid, expected_valid), f'valid mask wrong: {is_valid} vs {expected_valid}'\n"
            "\n"
            "# Good slices must hit the right answer.\n"
            "assert t.allclose(x[0], t.tensor([3.0, -1.0]), atol=1e-5)\n"
            "assert t.allclose(x[1], t.tensor([2.0, 1.0]),  atol=1e-5)\n"
            "\n"
            "# Solve must finish without raising — already proven by reaching here.\n"
            "print('  singular-batch solve completed without crashing')\n"
            "\n"
            "# Random large batch: inject a known number of singular slices.\n"
            "rng = t.Generator().manual_seed(2)\n"
            "K = 50\n"
            "A_big = t.eye(2).expand(K, 2, 2) + 0.2 * t.randn(K, 2, 2, generator=rng)\n"
            "A_big = A_big.clone()  # break the expand-storage so we can mutate slices\n"
            "# Force slices [3, 17, 42] to be singular (rank 1).\n"
            "singular_idx = [3, 17, 42]\n"
            "for k in singular_idx:\n"
            "    A_big[k] = t.tensor([[1.0, 2.0], [2.0, 4.0]])\n"
            "b_big = t.randn(K, 2, generator=rng)\n"
            "x_big, valid_big = ex1_solve_with_singular_mask(A_big, b_big)\n"
            "assert x_big.shape == (K, 2)\n"
            "# Exactly the three injected slices should be flagged invalid.\n"
            "invalid_idx = (~valid_big).nonzero(as_tuple=True)[0].tolist()\n"
            "assert sorted(invalid_idx) == sorted(singular_idx), (\n"
            "    f'invalid indices mismatch: {sorted(invalid_idx)} vs {sorted(singular_idx)}'\n"
            ")\n"
            "# Reconstruction OK on valid slices.\n"
            "recon = t.einsum('kij,kj->ki', A_big[valid_big], x_big[valid_big])\n"
            "assert t.allclose(recon, b_big[valid_big], atol=1e-4), (\n"
            "    'reconstruction failed on the non-singular slices'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_solve_with_singular_mask(A: Tensor, b: Tensor, eps: float = 1e-8):\n"
            "    K, n, _ = A.shape\n"
            "    dets = t.linalg.det(A)\n"
            "    is_singular = dets.abs() < eps\n"
            "    A_safe = A.clone()                # don't mutate caller's tensor\n"
            "    A_safe[is_singular] = t.eye(n)    # broadcast identity into singular slices\n"
            "    x = t.linalg.solve(A_safe, b)\n"
            "    return x, ~is_singular"
        ),
        "solution_notes": (
            "**Why clone.** `A[is_singular] = ...` writes through the original "
            "tensor's storage. If the caller passed a view of something larger "
            "(e.g. one part of a bigger model state), in-place mutation is a "
            "silent bug. `A.clone()` allocates fresh storage so the rewrite is "
            "scoped to our local copy.\n\n"
            "**Why identity specifically.** Any non-singular matrix would let "
            "`solve` succeed, but the identity has the cleanest semantics — "
            "the spurious 'solution' it produces is just `b[k]` unchanged, so "
            "if you ever forget to mask it out, downstream `NaN`s won't "
            "propagate from those slices. (You'll still get *wrong* answers, "
            "but they're finite and easy to debug.)\n\n"
            "**The det threshold.** `1e-8` is a typical default for `float32`; "
            "for `float64` use `1e-12`. A more principled approach uses the "
            "condition number (`t.linalg.cond`), but `|det| < eps` is the "
            "ARENA convention and fast.\n\n"
            "**Caveat for n > 2.** `t.eye(n)` is `(n, n)` and broadcasts "
            "cleanly into `A[is_singular]` (which has shape `(M, n, n)` for "
            "`M` singular slices). The broadcast rule is the same one that "
            "lets `A[is_singular] = 0.0` zero a batch of matrices."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-output-shape (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-output-shape",
        "subtopic": "CNN: Conv output shape",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_CONV_OUTSHAPE,
        "exercise_index": 1,
        "exercise_title": "compute conv2d output shape analytically",
        "slug": "compute-conv2d-output-shape-analytically",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["conv", "shape", "formula", "analytical"],
        "kcs": ["conv-output-shape-formula", "conv-shape-batch-pass-through"],
        "lo": (
            "Apply the 2-D convolution output-shape formula "
            "`H_out = floor((H + 2P - K) / S) + 1` (and analogously for `W`) "
            "to predict the output shape `(B, OC, H_out, W_out)` from input "
            "shape, kernel size, stride, and padding."
        ),
        "prompt_body": (
            "Implement `ex1_conv2d_outshape(input_shape, out_channels, kernel_size, "
            "stride, padding)`.\n\n"
            "- `input_shape` is a tuple `(B, IC, H, W)`.\n"
            "- `kernel_size`, `stride`, `padding` are each a 2-tuple "
            "`(h_val, w_val)`.\n"
            "- Return a tuple `(B, out_channels, H_out, W_out)`.\n\n"
            "**Formula (dilation=1).**\n"
            "```\n"
            "H_out = (H + 2*PH - KH) // SH + 1\n"
            "W_out = (W + 2*PW - KW) // SW + 1\n"
            "```\n\n"
            "**Hint.** Use Python integer arithmetic — no tensors needed. The "
            "batch axis and `out_channels` axis pass through unchanged; the "
            "input channels axis `IC` is *contracted away* (does not appear "
            "in the output shape).\n\n"
            "After your computation, the test creates an empty `nn.Conv2d` "
            "with the same hyperparams, runs the same input through it, and "
            "confirms your predicted shape matches the actual tensor's shape."
        ),
        "stub": (
            "def ex1_conv2d_outshape(\n"
            "    input_shape,\n"
            "    out_channels,\n"
            "    kernel_size,\n"
            "    stride,\n"
            "    padding,\n"
            "):\n"
            '    """Return the (B, OC, H_out, W_out) shape Conv2d would produce."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "def _check(input_shape, oc, k, s, p):\n"
            "    predicted = ex1_conv2d_outshape(input_shape, oc, k, s, p)\n"
            "    # Truth: actually run an empty Conv2d and read the shape.\n"
            "    conv = nn.Conv2d(\n"
            "        in_channels=input_shape[1], out_channels=oc,\n"
            "        kernel_size=k, stride=s, padding=p,\n"
            "    )\n"
            "    x = t.zeros(*input_shape)\n"
            "    actual = tuple(conv(x).shape)\n"
            "    assert tuple(predicted) == actual, (\n"
            "        f'shape mismatch for {input_shape} oc={oc} k={k} s={s} p={p}:\\n'\n"
            "        f'  predicted {tuple(predicted)}\\n  actual    {actual}'\n"
            "    )\n"
            "\n"
            "# No-pad, stride-1: H_out = H - K + 1.\n"
            "_check((1, 3, 32, 32), 16, (3, 3), (1, 1), (0, 0))\n"
            "# Same-padding for stride-1 odd kernel: H_out = H.\n"
            "_check((1, 3, 32, 32), 16, (3, 3), (1, 1), (1, 1))\n"
            "# Stride-2: halves the spatial axes (with same-pad).\n"
            "_check((4, 8, 64, 64), 32, (3, 3), (2, 2), (1, 1))\n"
            "# Non-square everything.\n"
            "_check((2, 1, 28, 40), 4, (5, 3), (2, 1), (2, 1))\n"
            "# Stride==kernel, no pad → non-overlapping tiles.\n"
            "_check((1, 3, 24, 24), 6, (4, 4), (4, 4), (0, 0))\n"
            "# Big batch, big channels.\n"
            "_check((8, 64, 56, 56), 128, (1, 1), (1, 1), (0, 0))\n"
            "# Padding that adds more than the kernel removes (output > input).\n"
            "_check((1, 3, 8, 8), 1, (3, 3), (1, 1), (2, 2))\n"
            "# Stride > 1, asymmetric kernel + padding.\n"
            "_check((1, 1, 17, 19), 1, (4, 2), (3, 2), (1, 0))\n"
            "\n"
            "# Direct value spot-check: H=10, K=3, S=2, P=1 → (10+2-3)//2 + 1 = 5.\n"
            "spot = ex1_conv2d_outshape((1, 1, 10, 10), 1, (3, 3), (2, 2), (1, 1))\n"
            "assert spot == (1, 1, 5, 5), f'expected (1,1,5,5), got {spot}'"
        ),
        "solution_body": (
            "def ex1_conv2d_outshape(input_shape, out_channels, kernel_size, stride, padding):\n"
            "    B, IC, H, W = input_shape\n"
            "    KH, KW = kernel_size\n"
            "    SH, SW = stride\n"
            "    PH, PW = padding\n"
            "    H_out = (H + 2 * PH - KH) // SH + 1\n"
            "    W_out = (W + 2 * PW - KW) // SW + 1\n"
            "    return (B, out_channels, H_out, W_out)"
        ),
        "solution_notes": (
            "**Floor division is essential.** `(H + 2*PH - KH) / SH` may be "
            "fractional — convolution drops any partial window at the right "
            "edge. Use `//` (integer floor) to match PyTorch's behavior.\n\n"
            "**Why `+ 1`.** With stride `S` and effective input length "
            "`L = H + 2*PH`, the number of valid kernel positions is "
            "`floor((L - K) / S) + 1`. The `+1` counts the *first* position "
            "(window starting at 0); the floored quotient counts the "
            "additional positions reachable by stepping `S` at a time.\n\n"
            "**Pass-through axes.** `B` (batch) and `OC` (output channels) "
            "appear unchanged in the output. `IC` (input channels) **does "
            "not appear** — it's contracted by the dot product with the "
            "kernel's `IC` axis.\n\n"
            "**Dilation generalization.** Replace `KH` with the *effective* "
            "kernel size `DH * (KH - 1) + 1` for dilation `DH > 1`. The drill "
            "fixes dilation=1 to keep the formula clean."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-padding-zero (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-padding-zero",
        "subtopic": "CNN: Conv zero padding",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_CONV_PAD,
        "exercise_index": 1,
        "exercise_title": "build a zero-padded 1-D input by slice assignment",
        "slug": "build-a-zero-padded-1d-input-by-slice-assignment",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["padding", "slice-assign", "boundary"],
        "kcs": ["pad-allocate-zero-buffer", "pad-slice-assign-interior"],
        "lo": (
            "Apply the manual zero-padding pattern: allocate a zero-filled "
            "destination tensor sized to fit the padded input, then copy the "
            "original into the interior via slice assignment."
        ),
        "prompt_body": (
            "Implement `ex1_pad1d_zeros(x, left, right)`.\n\n"
            "- `x` has shape `(B, IC, W)`.\n"
            "- `left`, `right` are non-negative ints.\n"
            "- Return shape `(B, IC, left + W + right)`. Entries before column "
            "`left` and after column `left + W` must be exactly zero; entries "
            "in the interior must equal the corresponding columns of `x`.\n\n"
            "**Required approach.** Build the output by hand (so the mechanics "
            "are visible), not via `F.pad`:\n\n"
            "1. Allocate a zero buffer with `x.new_zeros(B, IC, left + W + right)` "
            "(this respects `x`'s dtype and device).\n"
            "2. Use slice assignment to copy `x` into columns "
            "`[left : left + W]` of the buffer.\n\n"
            "Edge cases the test exercises:\n"
            "- `left == 0` and `right == 0` (no-op, output should equal input).\n"
            "- `left > 0`, `right == 0` (front-only pad).\n"
            "- `left == 0`, `right > 0` (back-only pad).\n"
            "- Non-trivial values in `x` (no accidental zeroing of interior)."
        ),
        "stub": (
            "def ex1_pad1d_zeros(x: Tensor, left: int, right: int) -> Tensor:\n"
            '    """Return x padded with `left` and `right` zeros along the last axis."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Basic: pad (1, 2, 3) → length 5+1+2 = 8.\n"
            "x = t.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])  # (1, 1, 5)\n"
            "out = ex1_pad1d_zeros(x, 1, 2)\n"
            "expected = t.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]]])\n"
            "assert out.shape == (1, 1, 8), f'expected (1,1,8), got {tuple(out.shape)}'\n"
            "assert out.dtype == x.dtype\n"
            "assert t.allclose(out, expected, atol=1e-6), f'values wrong:\\n{out}\\nvs\\n{expected}'\n"
            "\n"
            "# Boundary cells must be exactly zero (not just close to zero).\n"
            "assert (out[..., :1] == 0).all(), 'left boundary must be exact 0'\n"
            "assert (out[..., -2:] == 0).all(), 'right boundary must be exact 0'\n"
            "\n"
            "# No-op: left=right=0.\n"
            "noop = ex1_pad1d_zeros(x, 0, 0)\n"
            "assert noop.shape == x.shape\n"
            "assert t.allclose(noop, x), 'pad(0, 0) must equal x'\n"
            "\n"
            "# Front-only.\n"
            "front = ex1_pad1d_zeros(x, 3, 0)\n"
            "assert front.shape == (1, 1, 8)\n"
            "assert (front[..., :3] == 0).all(), 'front 3 must be 0'\n"
            "assert t.allclose(front[..., 3:], x), 'remainder must match x'\n"
            "\n"
            "# Back-only.\n"
            "back = ex1_pad1d_zeros(x, 0, 4)\n"
            "assert back.shape == (1, 1, 9)\n"
            "assert (back[..., 5:] == 0).all(), 'back 4 must be 0'\n"
            "assert t.allclose(back[..., :5], x), 'front must match x'\n"
            "\n"
            "# Multi-channel, batched: must not flatten across B / IC.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "x2 = t.randn(2, 3, 4, generator=rng)\n"
            "out2 = ex1_pad1d_zeros(x2, 2, 1)\n"
            "assert out2.shape == (2, 3, 7)\n"
            "# Interior matches.\n"
            "assert t.allclose(out2[:, :, 2:6], x2, atol=1e-6), 'interior must equal x2'\n"
            "# Boundaries zero in every (b, c) slice.\n"
            "assert (out2[:, :, :2] == 0).all(), 'left zeros must hold across batch+channels'\n"
            "assert (out2[:, :, 6:] == 0).all(), 'right zeros must hold across batch+channels'\n"
            "\n"
            "# Cross-check against F.pad on a random tensor (must agree exactly).\n"
            "from torch.nn import functional as F\n"
            "x3 = t.randn(1, 2, 7, generator=rng)\n"
            "left, right = 3, 5\n"
            "ours = ex1_pad1d_zeros(x3, left, right)\n"
            "ref  = F.pad(x3, (left, right), value=0.0)\n"
            "assert t.allclose(ours, ref), 'must agree with F.pad(value=0.0)'\n"
            "\n"
            "# Conv-readiness: passing the padded tensor through F.conv1d with no\n"
            "# extra padding must equal F.conv1d(x, weight, padding=(left, right))\n"
            "# when left == right.\n"
            "k = 3\n"
            "weight = t.randn(2, 2, k, generator=rng)\n"
            "pad = 2\n"
            "padded = ex1_pad1d_zeros(x3, pad, pad)\n"
            "y_manual = F.conv1d(padded, weight)\n"
            "y_native = F.conv1d(x3, weight, padding=pad)\n"
            "assert t.allclose(y_manual, y_native, atol=1e-5), 'pre-padded conv should equal native padded conv'"
        ),
        "solution_body": (
            "def ex1_pad1d_zeros(x: Tensor, left: int, right: int) -> Tensor:\n"
            "    B, IC, W = x.shape\n"
            "    out = x.new_zeros(B, IC, left + W + right)\n"
            "    out[..., left : left + W] = x\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why `new_zeros` instead of `t.zeros`.** `x.new_zeros(...)` "
            "inherits `x`'s dtype and device automatically. `t.zeros(...)` "
            "defaults to `float32` on CPU regardless of `x`, which silently "
            "promotes/demotes when you assign into it.\n\n"
            "**Slice assignment vs `F.pad`.** Both produce identical output. "
            "Slice assignment is more explicit (the slice indices `[left : "
            "left + W]` show *where* the original lives in the padded tensor). "
            "`F.pad(x, (left, right))` is more idiomatic in PyTorch code — its "
            "argument is in *reverse* axis order (last-axis pad comes first), "
            "which is a common bug source.\n\n"
            "**Why this matters for conv.** ARENA's `conv1d`-from-scratch "
            "pre-pads the input rather than threading `padding` through the "
            "windowing math — the as_strided windowing formula becomes "
            "`ow = (padded_w - kw) // stride + 1` with no padding term. This "
            "is the canonical 'pad first, then window' pattern.\n\n"
            "**Pad value for maxpool is not zero.** For `max_pool` you'd want "
            "`x.new_full((B, IC, total_w), -t.inf)` — zero would let padded "
            "cells win the max for negative-valued inputs."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # conv-windowing-1d (1)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "conv-windowing-1d",
        "subtopic": "CNN: 1-D conv windowing",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_CONV_WIN_1D,
        "exercise_index": 1,
        "exercise_title": "build the 1-D conv window view via as_strided",
        "slug": "build-the-1d-conv-window-view-via-as-strided",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["as_strided", "windowing", "view", "conv1d-equivalence"],
        "kcs": ["windowing-stride-pattern", "windowing-output-width"],
        "lo": (
            "Apply `as_strided` to build the `(B, IC, OW, KW)` window view of "
            "a 1-D input for stride-1 convolution, then verify that contracting "
            "it against a kernel via einsum equals `F.conv1d`."
        ),
        "prompt_body": (
            "Implement `ex1_conv1d_windows(x, KW)`. Given input "
            "`x` of shape `(B, IC, W)` and kernel width `KW`, return the "
            "strided window view of shape `(B, IC, OW, KW)` where `OW = "
            "W - KW + 1` and each `(KW,)` slice along the new `OW` axis is "
            "one stride-1 window of `x`.\n\n"
            "**The trick.** Read `x.stride()` to get `(s_b, s_ic, s_w)`, then "
            "call:\n"
            "```\n"
            "x.as_strided(\n"
            "    size=(B, IC, OW, KW),\n"
            "    stride=(s_b, s_ic, s_w, s_w),\n"
            ")\n"
            "```\n"
            "The trailing pair `(s_w, s_w)` is the key — same stride on the "
            "`OW` axis as on the `KW` axis means adjacent windows are offset "
            "by 1 element of the original `W` axis (so they overlap by "
            "`KW - 1`).\n\n"
            "**Constraints.** No copy — your returned tensor must share "
            "storage with `x` (the test confirms this).\n\n"
            "The verification cell contracts your window view against a random "
            "kernel via `einops.einsum(..., 'b ic ow kw, oc ic kw -> b oc ow')` "
            "and compares against `F.conv1d`."
        ),
        "stub": (
            "def ex1_conv1d_windows(x: Tensor, KW: int) -> Tensor:\n"
            '    """Return (B, IC, OW, KW) window view of x for stride-1 conv1d."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.nn import functional as F\n"
            "\n"
            "# --- Shape & no-copy check ---\n"
            "rng = t.Generator().manual_seed(0)\n"
            "x = t.arange(1.0, 11.0).reshape(1, 1, 10).contiguous()\n"
            "KW = 3\n"
            "win = ex1_conv1d_windows(x, KW)\n"
            "OW = 10 - KW + 1\n"
            "assert win.shape == (1, 1, OW, KW), f'expected (1,1,{OW},{KW}), got {tuple(win.shape)}'\n"
            "assert win.dtype == x.dtype\n"
            "# No copy — must share storage.\n"
            "assert win.data_ptr() == x.data_ptr(), 'windows must be a view (share storage with x)'\n"
            "\n"
            "# --- Value check ---\n"
            "# First window = [1, 2, 3], next = [2, 3, 4], etc.\n"
            "for k in range(OW):\n"
            "    assert t.allclose(win[0, 0, k], x[0, 0, k:k+KW]), (\n"
            "        f'window {k}: expected {x[0,0,k:k+KW]}, got {win[0,0,k]}'\n"
            "    )\n"
            "\n"
            "# --- Equivalence with F.conv1d on a multi-channel input ---\n"
            "B, IC, W, OC = 2, 3, 12, 4\n"
            "KW2 = 5\n"
            "x2 = t.randn(B, IC, W, generator=rng)\n"
            "weight = t.randn(OC, IC, KW2, generator=rng)\n"
            "win2 = ex1_conv1d_windows(x2, KW2)\n"
            "assert win2.shape == (B, IC, W - KW2 + 1, KW2)\n"
            "y_manual = einops.einsum(win2, weight, 'b ic ow kw, oc ic kw -> b oc ow')\n"
            "y_native = F.conv1d(x2, weight)  # default stride=1, padding=0\n"
            "assert t.allclose(y_manual, y_native, atol=1e-4), (\n"
            "    'einsum(windows, weight) must equal F.conv1d(x, weight) to fp tolerance'\n"
            ")\n"
            "\n"
            "# --- Edge: KW == W → single window of size W ---\n"
            "x3 = t.arange(1.0, 6.0).reshape(1, 1, 5).contiguous()\n"
            "win3 = ex1_conv1d_windows(x3, 5)\n"
            "assert win3.shape == (1, 1, 1, 5), f'edge case shape: {tuple(win3.shape)}'\n"
            "assert t.allclose(win3[0, 0, 0], x3[0, 0]), 'single-window value must equal x'\n"
            "\n"
            "# --- Edge: KW == 1 → OW == W and each window is a length-1 slice ---\n"
            "win4 = ex1_conv1d_windows(x3, 1)\n"
            "assert win4.shape == (1, 1, 5, 1)\n"
            "assert t.allclose(win4.squeeze(-1), x3), 'KW=1 windows squeezed should equal x'"
        ),
        "solution_body": (
            "def ex1_conv1d_windows(x: Tensor, KW: int) -> Tensor:\n"
            "    B, IC, W = x.shape\n"
            "    OW = W - KW + 1\n"
            "    s_b, s_ic, s_w = x.stride()\n"
            "    return x.as_strided(\n"
            "        size=(B, IC, OW, KW),\n"
            "        stride=(s_b, s_ic, s_w, s_w),\n"
            "    )"
        ),
        "solution_notes": (
            "**Reading the stride tuple.** A tensor's `.stride()` returns the "
            "number of *elements* (not bytes) you advance through storage when "
            "you increment each axis by 1. For a contiguous `(B, IC, W)`, that "
            "is `(IC*W, W, 1)` — but you don't need to know this; just read "
            "`x.stride()` and reuse the values.\n\n"
            "**Why `(s_w, s_w)` on the trailing pair.** The new `OW` axis "
            "means 'window index' — stepping by 1 in `OW` must move the "
            "window by 1 element of the original `W`, so its stride is `s_w`. "
            "The `KW` axis means 'position within a window' — stepping by 1 "
            "in `KW` also moves 1 element of `W`, so its stride is also `s_w`. "
            "Same stride; different semantics.\n\n"
            "**Common pitfall.** If `x` was itself created by striding "
            "(e.g. a previous as_strided view), `s_w` may not be `1`. The "
            "ARENA code explicitly warns about this — never hardcode "
            "`stride=(s_b, s_ic, 1, 1)`. Always read from `x.stride()`.\n\n"
            "**No data is copied.** `as_strided` constructs a view header "
            "pointing into the same storage. That's why this is fast: the "
            "expensive operation is the einsum that follows, not the windowing.\n\n"
            "**Generalizing.** Add a `stride` parameter by multiplying the `OW` "
            "stride: `stride=(s_b, s_ic, s_w * conv_stride, s_w)`. Add padding "
            "by first calling the `conv-padding-zero` drill on `x`, then "
            "windowing the padded version. That's the full ARENA 1-D conv "
            "in three composable pieces."
        ),
    },
]


def main():
    written = []
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        written.append(rel)
        print(f"wrote {rel}")
    print(f"\nTotal: {len(written)} notebooks")


if __name__ == "__main__":
    main()
