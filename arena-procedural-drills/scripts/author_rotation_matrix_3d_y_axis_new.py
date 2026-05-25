#!/usr/bin/env python3
"""Author 4 new Colab-native exercises (ex6-ex9) for atom `rotation-matrix-3d-y-axis`.

These require visualization and integrative pipelines that flashcards can't deliver:
  ex6 — Animated rotation sweep: rotate points at 0/30/60/90/180°, 5 subplots
  ex7 — Compose Rx · Ry · Rz, rotate a cube's 8 vertices, 3-D scatter before/after
  ex8 — Inverse rotation = transpose: numerical sweep + error magnitudes
  ex9 — Camera-to-world transform: rays + pose → world frame, step-by-step shape debug
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

ATOM_ID = "rotation-matrix-3d-y-axis"
SUBTOPIC = "Numpy: Applied patterns and advanced"
TOPIC = "prereqs_numpy"

RECAP = (
    "## Y-axis rotation — quick refresher\n"
    "\n"
    "**The matrix.** Right-hand rotation by `θ` about Y:\n"
    "```\n"
    "R_y(θ) = [[ cos θ,  0,  sin θ],\n"
    "          [ 0,      1,  0    ],\n"
    "          [-sin θ,  0,  cos θ]]\n"
    "```\n"
    "Anything along Y stays put (middle row `[0,1,0]`); the X-Z plane rotates.\n"
    "\n"
    "**Acting on data.** Column-vector form: `v' = R @ v`. Batch of row-vectors "
    "`(N, 3)`: `points' = points @ R.T`. Composition: `R(α) @ R(β) = R(α + β)` for "
    "single-axis rotations; multi-axis rotations don't commute.\n"
    "\n"
    "**Numerical truth.** Rotation matrices are orthogonal: `R @ R.T = I` and "
    "`R.inverse() == R.T`. Floating-point composition accumulates ~1e-7 error per matmul."
)

SPECS = [
    # ─────────────────────────────────────────────────────────────────────────
    # ex6 — Animated rotation sweep: 5 angles, 5 subplots
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "rotation sweep — visualize 5 angles",
        "slug": "rotation-sweep-visualize-5-angles",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["rotation", "batch", "visualization", "matplotlib", "sweep"],
        "kcs": ["rotation-matrix-y-construct", "rotate-batch-of-points"],
        "lo": "Apply Y-rotation across a sweep of angles to a single point cloud and "
              "render each rotated snapshot as a subplot to inspect the trajectory visually.",
        "prompt_body": (
            "Implement `ex6_rotation_sweep(points, angles)`. Given a `(N, 3)` batch of "
            "points and a 1-D tensor of `M` scalar angles (in radians), return a stacked "
            "tensor of shape `(M, N, 3)` where row `m` is `points` rotated by `angles[m]` "
            "around the Y axis.\n"
            "\n"
            "**Hint.** Build `R_y(θ_m)` once per angle in a Python loop, then "
            "`points @ R.T`. Stack the per-angle results with `t.stack(..., dim=0)`. "
            "(There's also a fully vectorized version using `t.stack` of the rotation "
            "matrices first; the loop version is fine for this exercise.)\n"
            "\n"
            "The test verifies values at the canonical angles 0, π/2, π, then renders 5 "
            "X-Z-plane subplots showing the rotated point cloud at each angle so you can "
            "see the trajectory."
        ),
        "stub": (
            "def ex6_rotation_sweep(points: Tensor, angles: Tensor) -> Tensor:\n"
            "    \"\"\"Rotate (N, 3) `points` by each scalar in `angles`. Returns (M, N, 3).\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "# A simple 4-point square in the X-Z plane (Y=0).\n"
            "points = t.tensor([\n"
            "    [ 1.0, 0.0,  0.0],\n"
            "    [ 0.0, 0.0,  1.0],\n"
            "    [-1.0, 0.0,  0.0],\n"
            "    [ 0.0, 0.0, -1.0],\n"
            "])\n"
            "angles = t.tensor([0.0, math.pi / 6, math.pi / 3, math.pi / 2, math.pi])\n"
            "out = ex6_rotation_sweep(points, angles)\n"
            "\n"
            "# Shape check.\n"
            "assert out.shape == (5, 4, 3), f'expected (5, 4, 3), got {tuple(out.shape)}'\n"
            "\n"
            "# Angle 0 must be the identity.\n"
            "assert t.allclose(out[0], points, atol=1e-6), 'angle 0 should leave points unchanged'\n"
            "\n"
            "# Angle π should negate the X-Z components, leave Y alone.\n"
            "expected_pi = points.clone()\n"
            "expected_pi[:, 0] = -points[:, 0]\n"
            "expected_pi[:, 2] = -points[:, 2]\n"
            "assert t.allclose(out[4], expected_pi, atol=1e-5), 'angle π should negate X and Z'\n"
            "\n"
            "# Angle π/2: +X → -Z, +Z → +X.\n"
            "p_x_axis = out[3, 0]\n"
            "assert t.allclose(p_x_axis, t.tensor([0.0, 0.0, -1.0]), atol=1e-6), \\\n"
            "    f'+X at π/2 should be (0,0,-1), got {p_x_axis}'\n"
            "p_z_axis = out[3, 1]\n"
            "assert t.allclose(p_z_axis, t.tensor([1.0, 0.0, 0.0]), atol=1e-6), \\\n"
            "    f'+Z at π/2 should be (1,0,0), got {p_z_axis}'\n"
            "\n"
            "# Length preservation across the whole sweep.\n"
            "norms = out.pow(2).sum(dim=-1).sqrt()\n"
            "assert t.allclose(norms, t.ones(5, 4), atol=1e-5), 'all rotated points should have unit norm'\n"
            "\n"
            "# Visualize: 5 subplots showing the X-Z projection at each angle.\n"
            "fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))\n"
            "for i, ax in enumerate(axes):\n"
            "    pts = out[i]\n"
            "    ax.scatter(pts[:, 0].numpy(), pts[:, 2].numpy(), s=60, c=range(4), cmap='viridis')\n"
            "    ax.scatter(points[:, 0].numpy(), points[:, 2].numpy(), s=30, c='lightgray', marker='x')\n"
            "    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)\n"
            "    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)\n"
            "    ax.set_title(f'θ = {angles[i].item():.2f} rad')\n"
            "    ax.set_xlabel('X'); ax.set_ylabel('Z')\n"
            "fig.suptitle('Y-axis rotation sweep (gray X = original)')\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
            "print(f'output shape: {tuple(out.shape)}')\n"
            "print(f'norms per angle: {norms.mean(dim=-1).tolist()} (all should be ~1.0)')"
        ),
        "solution_body": (
            "def ex6_rotation_sweep(points: Tensor, angles: Tensor) -> Tensor:\n"
            "    snapshots = []\n"
            "    for theta in angles:\n"
            "        c, s = t.cos(theta).item(), t.sin(theta).item()\n"
            "        R = t.tensor([\n"
            "            [c,   0.0, s  ],\n"
            "            [0.0, 1.0, 0.0],\n"
            "            [-s,  0.0, c  ],\n"
            "        ])\n"
            "        snapshots.append(points @ R.T)\n"
            "    return t.stack(snapshots, dim=0)"
        ),
        "solution_notes": (
            "**Loop vs vectorized.** The loop version is clearest for first contact. For "
            "performance you can build a `(M, 3, 3)` stack of rotation matrices in one go "
            "with `t.stack` of `(M,)` cos/sin tensors, then do a single batched matmul "
            "`points @ Rs.transpose(-1, -2)` (broadcast against the `M` axis). The math is "
            "identical; only the wall-clock differs.\n"
            "\n"
            "**Where you'll use this.** Generating training-time augmentations (random "
            "rotations as a `(M, ...)` axis), orbiting a camera around a fixed scene, "
            "rendering a turntable GIF of a 3-D model. Anywhere you need *the same data, "
            "different viewing angles*."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex7 — Compose Rx · Ry · Rz on a cube; 3-D scatter before/after
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "compose Rx · Ry · Rz on a cube + 3-D scatter",
        "slug": "compose-rx-ry-rz-on-cube-3d-scatter",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["multi-axis-rotation", "composition", "cube", "3d-scatter", "integrative"],
        "kcs": ["rotation-matrix-y-construct", "rotation-composition-multi-axis", "rotate-batch-of-points"],
        "lo": "Compose three single-axis rotations into a multi-axis transform and apply it "
              "to a cube's 8 vertices, then plot the original and rotated cubes in 3-D.",
        "prompt_body": (
            "Implement `ex7_compose_xyz(points, ax, ay, az)`. Given a `(N, 3)` batch of "
            "points and three scalar angles, build the rotation matrices `R_x(ax)`, "
            "`R_y(ay)`, `R_z(az)` and return the rotated points `points @ (R_x @ R_y @ R_z).T`.\n"
            "\n"
            "**The three matrices** (right-hand rule):\n"
            "```\n"
            "R_x(θ) = [[1, 0, 0], [0, c, -s], [0, s, c]]\n"
            "R_y(θ) = [[c, 0, s], [0, 1, 0], [-s, 0, c]]\n"
            "R_z(θ) = [[c, -s, 0], [s, c, 0], [0, 0, 1]]\n"
            "```\n"
            "where `c = cos θ`, `s = sin θ` for each axis's angle.\n"
            "\n"
            "Apply order matters: `R_x @ R_y @ R_z` is **not** equal to "
            "`R_z @ R_y @ R_x`. Use the order given.\n"
            "\n"
            "The test (1) checks all-zero angles act as identity, (2) checks distances "
            "from origin are preserved, (3) checks a hand-computed single-axis case, "
            "then plots the original cube vertices vs the rotated cube in a 3-D scatter."
        ),
        "stub": (
            "def ex7_compose_xyz(points: Tensor, ax: float, ay: float, az: float) -> Tensor:\n"
            "    \"\"\"Apply R_x(ax) @ R_y(ay) @ R_z(az) to (N, 3) points. Returns (N, 3).\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "# 8 cube vertices at the corners of [-1, 1]^3.\n"
            "cube = t.tensor([\n"
            "    [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0],\n"
            "    [-1.0,  1.0, -1.0], [ 1.0,  1.0, -1.0],\n"
            "    [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0],\n"
            "    [-1.0,  1.0,  1.0], [ 1.0,  1.0,  1.0],\n"
            "])\n"
            "\n"
            "# All zeros → identity.\n"
            "out0 = ex7_compose_xyz(cube, 0.0, 0.0, 0.0)\n"
            "assert out0.shape == (8, 3), f'expected (8,3), got {tuple(out0.shape)}'\n"
            "assert t.allclose(out0, cube, atol=1e-6), 'zero angles should not move points'\n"
            "\n"
            "# Distance preservation: all vertices stay at sqrt(3) from origin.\n"
            "out = ex7_compose_xyz(cube, 0.4, -0.7, 1.1)\n"
            "norms = out.pow(2).sum(dim=-1).sqrt()\n"
            "assert t.allclose(norms, t.full((8,), math.sqrt(3.0)), atol=1e-5), \\\n"
            "    f'cube vertices should stay at sqrt(3) from origin, got {norms}'\n"
            "\n"
            "# Single-axis check: only Y rotation by π/2.\n"
            "out_y = ex7_compose_xyz(cube, 0.0, math.pi / 2, 0.0)\n"
            "# R_y(π/2): X → -Z, Z → +X, Y unchanged.\n"
            "# So vertex (1, 1, 1) → (1, 1, -1)? Let's compute: c=0, s=1.\n"
            "# R_y @ [1,1,1] = [c*1 + s*1, 1, -s*1 + c*1] = [1, 1, -1].\n"
            "vert_111 = out_y[7]\n"
            "assert t.allclose(vert_111, t.tensor([1.0, 1.0, -1.0]), atol=1e-6), \\\n"
            "    f'(1,1,1) under R_y(π/2) should be (1,1,-1), got {vert_111}'\n"
            "\n"
            "# Non-commutativity smoke check.\n"
            "out_xyz = ex7_compose_xyz(cube, 0.5, 0.5, 0.5)\n"
            "# Compare against a manual zyx ordering to confirm we built xyz, not zyx.\n"
            "ax, ay, az = 0.5, 0.5, 0.5\n"
            "cx, sx = math.cos(ax), math.sin(ax)\n"
            "cy, sy = math.cos(ay), math.sin(ay)\n"
            "cz, sz = math.cos(az), math.sin(az)\n"
            "Rx = t.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=t.float32)\n"
            "Ry = t.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=t.float32)\n"
            "Rz = t.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=t.float32)\n"
            "ref_xyz = cube @ (Rx @ Ry @ Rz).T\n"
            "assert t.allclose(out_xyz, ref_xyz, atol=1e-5), \\\n"
            "    f'composition order mismatch:\\n{out_xyz}\\nvs\\n{ref_xyz}'\n"
            "\n"
            "# 3-D scatter: original cube (gray) vs rotated cube (color).\n"
            "from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection)\n"
            "fig = plt.figure(figsize=(6, 5.5))\n"
            "ax_ = fig.add_subplot(111, projection='3d')\n"
            "ax_.scatter(cube[:, 0], cube[:, 1], cube[:, 2], c='lightgray', s=60, label='original')\n"
            "ax_.scatter(out[:, 0], out[:, 1], out[:, 2], c=range(8), cmap='plasma', s=80, label='rotated')\n"
            "for i in range(8):\n"
            "    ax_.plot([cube[i, 0], out[i, 0]],\n"
            "             [cube[i, 1], out[i, 1]],\n"
            "             [cube[i, 2], out[i, 2]], 'k--', alpha=0.3, linewidth=0.7)\n"
            "ax_.set_title('Cube before (gray) and after R_x · R_y · R_z rotation')\n"
            "ax_.set_xlabel('X'); ax_.set_ylabel('Y'); ax_.set_zlabel('Z')\n"
            "ax_.legend(loc='upper left')\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(f'rotated cube shape: {tuple(out.shape)}')\n"
            "print(f'norm preservation: max deviation = {(norms - math.sqrt(3.0)).abs().max().item():.2e}')"
        ),
        "solution_body": (
            "def ex7_compose_xyz(points: Tensor, ax: float, ay: float, az: float) -> Tensor:\n"
            "    import math\n"
            "    cx, sx = math.cos(ax), math.sin(ax)\n"
            "    cy, sy = math.cos(ay), math.sin(ay)\n"
            "    cz, sz = math.cos(az), math.sin(az)\n"
            "    Rx = t.tensor([\n"
            "        [1.0, 0.0, 0.0],\n"
            "        [0.0, cx,  -sx],\n"
            "        [0.0, sx,  cx ],\n"
            "    ])\n"
            "    Ry = t.tensor([\n"
            "        [cy,  0.0, sy ],\n"
            "        [0.0, 1.0, 0.0],\n"
            "        [-sy, 0.0, cy ],\n"
            "    ])\n"
            "    Rz = t.tensor([\n"
            "        [cz,  -sz, 0.0],\n"
            "        [sz,   cz, 0.0],\n"
            "        [0.0, 0.0, 1.0],\n"
            "    ])\n"
            "    R = Rx @ Ry @ Rz\n"
            "    return points @ R.T"
        ),
        "solution_notes": (
            "**Order matters.** `R_x @ R_y @ R_z` reads right-to-left when acting on a "
            "column vector: first rotate about Z, then about Y, then about X. Swapping "
            "the order produces a different orientation for any non-trivial angle "
            "combination. Pick one convention and stick to it across your whole codebase.\n"
            "\n"
            "**Why we can't just \"add angles\".** Single-axis rotations commute "
            "(`R_y(α) @ R_y(β) = R_y(α+β)`), but multi-axis don't. This is why "
            "orientation in 3-D needs three numbers (Euler angles, axis-angle, quaternion) "
            "with an agreed-upon convention — there's no scalar shortcut."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex8 — Inverse rotation = transpose: sweep + error magnitudes
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "inverse rotation = transpose — numerical sweep",
        "slug": "inverse-rotation-equals-transpose-numerical-sweep",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["orthogonality", "inverse", "transpose", "numerical-error", "sweep"],
        "kcs": ["rotation-matrix-y-construct", "rotation-orthogonality"],
        "lo": "Verify the orthogonality identity `R(θ)⁻¹ = R(-θ) = R(θ).T` numerically "
              "across an angle sweep and quantify floating-point deviation.",
        "prompt_body": (
            "Implement `ex8_inverse_error_sweep(angles)`. Given a 1-D tensor of `M` angles, "
            "return a `dict` with three `(M,)` tensors:\n"
            "\n"
            "```\n"
            "{\n"
            "  'err_inv_vs_negtheta': |R(θ)⁻¹  -  R(-θ)|.max() per angle,\n"
            "  'err_inv_vs_transpose': |R(θ)⁻¹  -  R(θ).T|.max() per angle,\n"
            "  'err_RRT_minus_I':       |R(θ) @ R(θ).T  -  I|.max() per angle,\n"
            "}\n"
            "```\n"
            "\n"
            "Build `R(θ)`, compute its inverse with `t.linalg.inv`, then compare against "
            "(a) `R(-θ)` built from scratch, (b) `R(θ).T`, and (c) the identity-preservation "
            "check `R @ R.T == I`. Use `.max()` to collapse the 3×3 absolute-difference "
            "matrix to a scalar per angle.\n"
            "\n"
            "The test verifies all three error tensors are at floating-point noise level "
            "(< 1e-5) and prints a comparison plot of error magnitude vs angle."
        ),
        "stub": (
            "def ex8_inverse_error_sweep(angles: Tensor) -> dict:\n"
            "    \"\"\"Return a dict of (M,) per-angle max-abs error tensors.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "angles = t.linspace(-math.pi, math.pi, 21)\n"
            "out = ex8_inverse_error_sweep(angles)\n"
            "\n"
            "# Structure.\n"
            "expected_keys = {'err_inv_vs_negtheta', 'err_inv_vs_transpose', 'err_RRT_minus_I'}\n"
            "assert set(out.keys()) == expected_keys, f'expected {expected_keys}, got {set(out.keys())}'\n"
            "\n"
            "for k in expected_keys:\n"
            "    assert isinstance(out[k], Tensor), f'{k} must be a Tensor, got {type(out[k])}'\n"
            "    assert out[k].shape == (21,), f'{k} should be (21,), got {tuple(out[k].shape)}'\n"
            "\n"
            "# All errors should be at floating-point noise level for a well-conditioned R.\n"
            "for k in expected_keys:\n"
            "    max_err = out[k].max().item()\n"
            "    assert max_err < 1e-4, f'{k}: max error {max_err} too large — orthogonality should hold'\n"
            "\n"
            "# Sanity: transpose-vs-inv should be even smaller than inv-vs-negtheta, since\n"
            "# transpose is exact and -theta involves a fresh cos/sin pair.\n"
            "median_transpose = out['err_inv_vs_transpose'].median().item()\n"
            "assert median_transpose < 1e-5, f'transpose ought to be near-exact: got {median_transpose}'\n"
            "\n"
            "# Visualize error magnitude vs angle.\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.semilogy(angles.numpy(), out['err_inv_vs_negtheta'].numpy(),\n"
            "            'o-', label='|R⁻¹ - R(-θ)|', alpha=0.8)\n"
            "ax.semilogy(angles.numpy(), out['err_inv_vs_transpose'].numpy(),\n"
            "            's-', label='|R⁻¹ - R.T|', alpha=0.8)\n"
            "ax.semilogy(angles.numpy(), out['err_RRT_minus_I'].numpy(),\n"
            "            '^-', label='|R @ R.T - I|', alpha=0.8)\n"
            "ax.set_xlabel('θ (rad)'); ax.set_ylabel('max abs error (log scale)')\n"
            "ax.set_title('Orthogonality identities hold at float32 precision')\n"
            "ax.grid(True, which='both', alpha=0.3)\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(f'max error across all checks: {max(v.max().item() for v in out.values()):.2e}')\n"
            "print(f'median |R⁻¹ - R.T| = {median_transpose:.2e} (effectively zero — they ARE equal)')"
        ),
        "solution_body": (
            "def ex8_inverse_error_sweep(angles: Tensor) -> dict:\n"
            "    err_inv_neg, err_inv_T, err_RRT = [], [], []\n"
            "    I = t.eye(3)\n"
            "    for theta in angles:\n"
            "        c, s = t.cos(theta).item(), t.sin(theta).item()\n"
            "        R = t.tensor([\n"
            "            [c,   0.0, s  ],\n"
            "            [0.0, 1.0, 0.0],\n"
            "            [-s,  0.0, c  ],\n"
            "        ])\n"
            "        cn, sn = t.cos(-theta).item(), t.sin(-theta).item()\n"
            "        R_neg = t.tensor([\n"
            "            [cn,   0.0, sn ],\n"
            "            [0.0,  1.0, 0.0],\n"
            "            [-sn,  0.0, cn ],\n"
            "        ])\n"
            "        R_inv = t.linalg.inv(R)\n"
            "        err_inv_neg.append((R_inv - R_neg).abs().max())\n"
            "        err_inv_T.append((R_inv - R.T).abs().max())\n"
            "        err_RRT.append((R @ R.T - I).abs().max())\n"
            "    return {\n"
            "        'err_inv_vs_negtheta': t.stack(err_inv_neg),\n"
            "        'err_inv_vs_transpose': t.stack(err_inv_T),\n"
            "        'err_RRT_minus_I': t.stack(err_RRT),\n"
            "    }"
        ),
        "solution_notes": (
            "**Why transpose = inverse for rotation matrices.** Rotation matrices are "
            "*orthogonal* — their rows (and columns) form an orthonormal basis. The defining "
            "property is `R @ R.T = I`, which by definition means `R.T = R⁻¹`. Computing an "
            "inverse via `t.linalg.inv` solves a linear system; computing a transpose is "
            "free. For rotations, always use `.T`.\n"
            "\n"
            "**Why the errors aren't exactly zero.** `t.linalg.inv` uses LU decomposition "
            "and accumulates roundoff. The transpose route is exact (it's just a stride "
            "swap), so `|R⁻¹ - R.T|` ≈ 1e-7 not 0 — that's the inv path's error, not "
            "the transpose's."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ex9 — Camera-to-world transform: rays + pose → world frame
    # ─────────────────────────────────────────────────────────────────────────
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 9,
        "exercise_title": "camera-to-world transform (rays + pose)",
        "slug": "camera-to-world-transform-rays-and-pose",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["camera-pose", "world-frame", "ray-transform", "integrative", "shape-debug"],
        "kcs": ["rotation-matrix-y-construct", "rotate-batch-of-points", "rigid-body-transform"],
        "lo": "Integrate batched rotation, broadcast addition of a translation, and the "
              "camera-vs-world distinction for rays vs points, with step-by-step shape "
              "debugging.",
        "prompt_body": (
            "Implement `ex9_cam_to_world(ray_dirs_cam, ray_origins_cam, R_cw, t_cw)` for "
            "a NeRF-style ray transform.\n"
            "\n"
            "**Inputs.**\n"
            "- `ray_dirs_cam`: `(B, 3)` ray directions in camera frame (unit length, no translation).\n"
            "- `ray_origins_cam`: `(B, 3)` ray origins in camera frame (a point — gets translated).\n"
            "- `R_cw`: `(3, 3)` rotation from camera to world.\n"
            "- `t_cw`: `(3,)` translation of camera origin in world coords.\n"
            "\n"
            "**Outputs.** A `dict`:\n"
            "```\n"
            "{\n"
            "  'dirs_world':    (B, 3) — directions: rotate only, NO translation\n"
            "  'origins_world': (B, 3) — points: rotate AND translate (add t_cw)\n"
            "}\n"
            "```\n"
            "\n"
            "**Key insight.** Vectors (directions) only rotate; points (origins) rotate + "
            "translate. Mixing this up is the most common bug in any 3-D pipeline.\n"
            "\n"
            "**Pipeline (build in order, print at each step):**\n"
            "1. Print `ray_dirs_cam.shape`, `ray_origins_cam.shape`, `R_cw.shape`, `t_cw.shape`.\n"
            "2. Compute `dirs_world = ray_dirs_cam @ R_cw.T` — print its shape.\n"
            "3. Compute `origins_world = ray_origins_cam @ R_cw.T + t_cw` (broadcast). Print shape.\n"
            "4. Return both in the dict.\n"
            "\n"
            "> ⚠️ **Integrative.** Three concepts: batch rotation, point-vs-vector distinction, "
            "broadcast addition. Print shapes between every step — don't trust a one-liner."
        ),
        "stub": (
            "def ex9_cam_to_world(ray_dirs_cam: Tensor, ray_origins_cam: Tensor,\n"
            "                     R_cw: Tensor, t_cw: Tensor) -> dict:\n"
            "    \"\"\"Transform camera-frame rays into world frame. Returns dict with 'dirs_world' and 'origins_world'.\"\"\"\n"
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "B = 6\n"
            "t.manual_seed(11)\n"
            "ray_dirs_cam = t.nn.functional.normalize(t.randn(B, 3), dim=-1)\n"
            "ray_origins_cam = t.zeros(B, 3)  # camera-frame rays start at the camera origin\n"
            "\n"
            "# Camera rotated π/4 about Y, translated to (5, 0, 2) in world coords.\n"
            "theta = t.tensor(math.pi / 4)\n"
            "c, s = t.cos(theta).item(), t.sin(theta).item()\n"
            "R_cw = t.tensor([\n"
            "    [c,   0.0, s  ],\n"
            "    [0.0, 1.0, 0.0],\n"
            "    [-s,  0.0, c  ],\n"
            "])\n"
            "t_cw = t.tensor([5.0, 0.0, 2.0])\n"
            "\n"
            "out = ex9_cam_to_world(ray_dirs_cam, ray_origins_cam, R_cw, t_cw)\n"
            "\n"
            "# Structure.\n"
            "assert set(out.keys()) == {'dirs_world', 'origins_world'}, \\\n"
            "    f'expected dirs_world+origins_world, got {sorted(out.keys())}'\n"
            "\n"
            "# Shapes.\n"
            "assert out['dirs_world'].shape == (B, 3), f'dirs shape {out[\"dirs_world\"].shape}'\n"
            "assert out['origins_world'].shape == (B, 3), f'origins shape {out[\"origins_world\"].shape}'\n"
            "\n"
            "# Origins (all zero in camera frame) must all map to exactly t_cw in world frame.\n"
            "for i in range(B):\n"
            "    assert t.allclose(out['origins_world'][i], t_cw, atol=1e-6), \\\n"
            "        f'origin {i}: expected {t_cw}, got {out[\"origins_world\"][i]}'\n"
            "\n"
            "# Direction lengths must be preserved (rotation is orthogonal).\n"
            "norms_cam = ray_dirs_cam.pow(2).sum(dim=-1).sqrt()\n"
            "norms_world = out['dirs_world'].pow(2).sum(dim=-1).sqrt()\n"
            "assert t.allclose(norms_cam, norms_world, atol=1e-6), \\\n"
            "    'direction lengths should be preserved through rotation'\n"
            "\n"
            "# Critical: directions must NOT have t_cw added to them.\n"
            "dirs_with_translation = ray_dirs_cam @ R_cw.T + t_cw\n"
            "assert not t.allclose(out['dirs_world'], dirs_with_translation, atol=1e-3), \\\n"
            "    'BUG: directions should rotate only — you accidentally added translation!'\n"
            "\n"
            "# Hand-check: a camera-frame ray pointing along -Z (canonical \"forward\")\n"
            "# under R_cw(π/4) about Y should be (-sin(π/4), 0, -cos(π/4)) in world frame.\n"
            "fwd_cam = t.tensor([[0.0, 0.0, -1.0]])\n"
            "fwd_world = ex9_cam_to_world(fwd_cam, t.zeros(1, 3), R_cw, t_cw)['dirs_world'][0]\n"
            "expected_fwd = t.tensor([-math.sin(math.pi / 4), 0.0, -math.cos(math.pi / 4)])\n"
            "assert t.allclose(fwd_world, expected_fwd, atol=1e-6), \\\n"
            "    f'forward ray rotation mismatch: {fwd_world} vs {expected_fwd}'\n"
            "\n"
            "# Step-by-step shape debug printout.\n"
            "print('=== shapes pipeline ===')\n"
            "print(f'  ray_dirs_cam:    {tuple(ray_dirs_cam.shape)}')\n"
            "print(f'  ray_origins_cam: {tuple(ray_origins_cam.shape)}')\n"
            "print(f'  R_cw:            {tuple(R_cw.shape)}')\n"
            "print(f'  t_cw:            {tuple(t_cw.shape)}')\n"
            "print(f'  -> dirs_world:    {tuple(out[\"dirs_world\"].shape)}  (rotated, no translate)')\n"
            "print(f'  -> origins_world: {tuple(out[\"origins_world\"].shape)}  (rotated + translated)')\n"
            "print()\n"
            "print(f'all origins land at t_cw = {t_cw.tolist()}: True')\n"
            "print(f'direction norms preserved: {t.allclose(norms_cam, norms_world)}')"
        ),
        "solution_body": (
            "def ex9_cam_to_world(ray_dirs_cam: Tensor, ray_origins_cam: Tensor,\n"
            "                     R_cw: Tensor, t_cw: Tensor) -> dict:\n"
            "    # Directions: rotate only.  (B, 3) @ (3, 3).T -> (B, 3)\n"
            "    dirs_world = ray_dirs_cam @ R_cw.T\n"
            "    # Origins: rotate, then add the camera translation in world coords.\n"
            "    # t_cw is (3,); broadcasts over the B axis.\n"
            "    origins_world = ray_origins_cam @ R_cw.T + t_cw\n"
            "    return {'dirs_world': dirs_world, 'origins_world': origins_world}"
        ),
        "solution_notes": (
            "**The vector-vs-point distinction.** A direction has no position — it's the "
            "difference of two points. Translating it would be a category error: "
            "`(p₁ + t) - (p₂ + t) = p₁ - p₂`, the translation cancels. So directions get "
            "rotation only. Points (origins) carry position and get the full rigid "
            "transform `R @ p + t`.\n"
            "\n"
            "**Where this lives in real code.** Every NeRF/3DGS/SfM pipeline has this exact "
            "function near the top of its ray-generation step. Get it wrong and your scene "
            "appears at the right orientation but completely the wrong location — a "
            "famously hard-to-debug failure mode because *most* test cases (single "
            "translation, single rotation) still pass.\n"
            "\n"
            "**Homogeneous-coordinates alternative.** You can pack rotation+translation "
            "into a `(4, 4)` matrix and represent points as `(x, y, z, 1)` vs directions as "
            "`(x, y, z, 0)` — the trailing 0 zeros out the translation column automatically. "
            "Same math, less code, but harder to debug shape-wise."
        ),
        "extra_imports": [],
    },
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        print(f"wrote {path.relative_to(path.parents[4])}")
