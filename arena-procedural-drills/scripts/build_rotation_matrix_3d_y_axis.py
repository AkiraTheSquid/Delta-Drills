#!/usr/bin/env python3
"""Build the rotation-matrix-3d-y-axis procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis.ipynb`
(Phase 4.12). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `rotation-matrix-3d-y-axis` is the canonical "rotate stuff around the
vertical axis" operation in 3-D — used in Ray Tracing for camera orbits,
object turntables, and Möller-Trumbore basis alignment. Bridges to bank
topic `Numpy` via the explicit token rule (`rotation-matrix`) in
atom_readiness.js, reporting to subtopic `Numpy: Applied patterns and advanced`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis.ipynb"


def verify_solutions(specs: list[dict]) -> None:
    try:
        import torch as t
        import numpy as np
        from torch import Tensor
    except ImportError as e:
        raise SystemExit(
            f"[build verify] missing runtime dep: {e}\n"
            f"  pip install torch numpy\n"
            f"  refusing to write notebook with unverified solutions."
        )

    base_ns = {"t": t, "np": np, "Tensor": Tensor}
    failures: list[str] = []
    for spec in specs:
        ns = dict(base_ns)
        try:
            exec(spec["solution_body"], ns)
        except Exception as e:
            failures.append(f"{spec['id']} ({spec['title']}) — solution_body did not compile: {e!r}")
            continue
        test_src = f"def _test_{spec['id']}():\n    {spec['test_body']}"
        try:
            exec(test_src, ns)
        except Exception as e:
            failures.append(f"{spec['id']} ({spec['title']}) — test_body did not compile: {e!r}")
            continue
        try:
            ns[f"_test_{spec['id']}"]()
        except Exception as e:
            failures.append(f"{spec['id']} ({spec['title']}) — assertion failed: {e!r}")
            continue

    if failures:
        print("[build verify] FAIL — refusing to emit notebook.", file=sys.stderr)
        for line in failures:
            print(f"  ✗ {line}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[build verify] OK — {len(specs)} canonical solutions pass their tests.")


ATOM_ID = "rotation-matrix-3d-y-axis"
SUBTOPIC = "Numpy: Applied patterns and advanced"
TITLE = "rotation matrix (3-D, Y-axis) — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "cos-sin-tensor-scalars",
        "kind": "component-skill",
        "description": "Compute `cos(θ)` and `sin(θ)` as 0-D tensors using `torch.cos` / `torch.sin`. The atomic primitive for any rotation.",
    },
    {
        "id": "rotation-matrix-y-construct",
        "kind": "component-skill",
        "description": "Assemble the 3×3 Y-axis rotation matrix `R_y(θ) = [[c, 0, s], [0, 1, 0], [-s, 0, c]]` from scalar `c = cos(θ)`, `s = sin(θ)`. The Y row stays the identity row because Y is the rotation axis.",
    },
    {
        "id": "rotation-applied-to-vector",
        "kind": "component-skill",
        "description": "Apply a rotation matrix to a 3-vector via `R @ v`. Result shape `(3,)`. Geometric check: rotating `(1, 0, 0)` by π/2 about Y gives `(0, 0, -1)` under right-hand convention.",
    },
    {
        "id": "rotation-composes-on-axis",
        "kind": "component-skill",
        "description": "Verify that `R_y(α) @ R_y(β) ≈ R_y(α + β)` (rotations about the same axis form a 1-parameter group). The structural invariant that lets you cache rotation matrices and compose them.",
    },
    {
        "id": "rotate-batch-of-points",
        "kind": "integrative-skill",
        "description": "Rotate an entire `(N, 3)` batch of points around the Y axis by `θ` in one matrix multiply. Combines matrix construction, batched matmul shape arithmetic, and right-hand sign convention — the canonical 'rotate a camera or object' operation.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "compute cos(θ) and sin(θ) as tensors",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["torch-cos", "torch-sin", "scalar-tensor"],
        "kcs": ["cos-sin-tensor-scalars"],
        "lo": "Recall how to compute `cos(θ)` and `sin(θ)` as scalar tensors.",
        "prompt_body": (
            "Implement `ex1_cos_sin(theta)`. Given a 0-D tensor `theta` (an angle in "
            "radians), return a tuple `(cos_t, sin_t)` of two 0-D tensors.\n\n"
            "Use `torch.cos` and `torch.sin`. Both inputs and outputs are torch "
            "tensors — don't convert to Python floats."
        ),
        "stub": (
            "def ex1_cos_sin(theta: Tensor) -> tuple:\n"
            '    """Return (cos(theta), sin(theta)) as 0-D tensors."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "    theta = t.tensor(math.pi / 2)\n"
            "    c, s = ex1_cos_sin(theta)\n"
            "    assert c.dim() == 0 and s.dim() == 0, 'both outputs must be 0-D'\n"
            "    assert t.allclose(c, t.tensor(0.0), atol=1e-6), f'cos(pi/2) should be 0, got {c.item()}'\n"
            "    assert t.allclose(s, t.tensor(1.0), atol=1e-6), f'sin(pi/2) should be 1, got {s.item()}'\n"
            "\n"
            "    theta0 = t.tensor(0.0)\n"
            "    c0, s0 = ex1_cos_sin(theta0)\n"
            "    assert t.allclose(c0, t.tensor(1.0)) and t.allclose(s0, t.tensor(0.0)), 'cos(0)=1, sin(0)=0'"
        ),
        "solution_body": (
            "def ex1_cos_sin(theta: Tensor) -> tuple:\n"
            "    return t.cos(theta), t.sin(theta)"
        ),
        "solution_notes": (
            "**Why tensor-typed cos/sin?** If you do `math.cos(theta.item())` you "
            "drop the autograd graph and the device — fine for a one-off, broken "
            "if `theta` is a learnable parameter or lives on GPU. Always prefer "
            "`torch.cos` / `torch.sin` when the input is a tensor."
        ),
    },
    {
        "id": "ex2",
        "title": "build the 3×3 Y-axis rotation matrix",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["rotation-matrix", "y-axis", "tensor-stack"],
        "kcs": ["rotation-matrix-y-construct"],
        "lo": "Apply the Y-axis rotation matrix formula to assemble a 3×3 tensor from `cos(θ)` and `sin(θ)`.",
        "prompt_body": (
            "Implement `ex2_rotation_y(theta)` to return the 3×3 Y-axis rotation "
            "matrix as a `(3, 3)` float tensor:\n\n"
            "```\n"
            "R_y(θ) = [[ cos θ,  0,  sin θ],\n"
            "          [ 0,      1,  0    ],\n"
            "          [-sin θ,  0,  cos θ]]\n"
            "```\n\n"
            "The middle row is `[0, 1, 0]` because the Y axis is the rotation axis "
            "— a point with only a Y component is fixed by the rotation.\n\n"
            "Hint: build the matrix via `t.tensor([[c, 0, s], ...])` after extracting "
            "`c.item()` / `s.item()`, OR use `t.stack` for autograd-safety. Either is "
            "fine for this exercise."
        ),
        "stub": (
            "def ex2_rotation_y(theta: Tensor) -> Tensor:\n"
            '    """3×3 Y-axis rotation matrix at angle theta."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "    R = ex2_rotation_y(t.tensor(0.0))\n"
            "    assert R.shape == (3, 3), f'shape mismatch: {R.shape}'\n"
            "    assert t.allclose(R, t.eye(3), atol=1e-6), 'R_y(0) should be identity'\n"
            "\n"
            "    R90 = ex2_rotation_y(t.tensor(math.pi / 2))\n"
            "    expected = t.tensor([\n"
            "        [0.0, 0.0, 1.0],\n"
            "        [0.0, 1.0, 0.0],\n"
            "        [-1.0, 0.0, 0.0],\n"
            "    ])\n"
            "    assert t.allclose(R90, expected, atol=1e-6), f'R_y(pi/2) mismatch:\\n{R90}'\n"
            "    # The middle row MUST be [0, 1, 0] regardless of theta.\n"
            "    assert t.allclose(R90[1], t.tensor([0.0, 1.0, 0.0])), 'Y row should always be [0, 1, 0]'"
        ),
        "solution_body": (
            "def ex2_rotation_y(theta: Tensor) -> Tensor:\n"
            "    c = t.cos(theta)\n"
            "    s = t.sin(theta)\n"
            "    return t.tensor([\n"
            "        [c.item(), 0.0,  s.item()],\n"
            "        [0.0,      1.0,  0.0],\n"
            "        [-s.item(), 0.0, c.item()],\n"
            "    ])"
        ),
        "solution_notes": (
            "**Sign convention.** The matrix above is the standard right-hand rule "
            "rotation: looking *down* the +Y axis (from above), the rotation goes "
            "counter-clockwise. Some texts (and graphics APIs like DirectX) use "
            "the left-hand variant with the signs flipped — `[[c, 0, -s], ..., [s, 0, c]]`. "
            "ARENA uses the right-hand convention.\n\n"
            "**Autograd-safe alternative.** If `theta` is a learnable parameter, "
            "use `torch.stack` to avoid breaking the graph:\n"
            "```python\n"
            "row0 = t.stack([c, t.zeros_like(c), s])\n"
            "row1 = t.tensor([0.0, 1.0, 0.0])\n"
            "row2 = t.stack([-s, t.zeros_like(c), c])\n"
            "return t.stack([row0, row1, row2])\n"
            "```"
        ),
    },
    {
        "id": "ex3",
        "title": "rotate a single 3-vector",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["matvec", "right-hand-rule", "geometric-check"],
        "kcs": ["rotation-applied-to-vector"],
        "lo": "Apply a rotation matrix to a 3-vector via matrix-vector multiplication.",
        "prompt_body": (
            "Implement `ex3_rotate_vector(v, theta)` to rotate the 3-vector `v` "
            "by angle `theta` about the Y axis. Output shape `(3,)`.\n\n"
            "Use `R_y(θ) @ v` where you build `R_y` from Exercise 2.\n\n"
            "Geometric check (right-hand rule, looking down +Y):\n"
            "- `(1, 0, 0)` rotated by π/2 → `(0, 0, -1)` (X goes to -Z).\n"
            "- `(0, 0, 1)` rotated by π/2 → `(1, 0, 0)` (+Z goes to +X).\n"
            "- Any `(0, y, 0)` stays fixed."
        ),
        "stub": (
            "def ex3_rotate_vector(v: Tensor, theta: Tensor) -> Tensor:\n"
            '    """Rotate v (shape (3,)) about Y by theta. Returns (3,)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "    half_pi = t.tensor(math.pi / 2)\n"
            "    # +X axis rotates to -Z under right-hand rule.\n"
            "    out = ex3_rotate_vector(t.tensor([1.0, 0.0, 0.0]), half_pi)\n"
            "    assert out.shape == (3,), f'shape mismatch: {out.shape}'\n"
            "    assert t.allclose(out, t.tensor([0.0, 0.0, -1.0]), atol=1e-6), f'(1,0,0) @ π/2 should be (0,0,-1), got {out.tolist()}'\n"
            "    # +Z rotates to +X.\n"
            "    out2 = ex3_rotate_vector(t.tensor([0.0, 0.0, 1.0]), half_pi)\n"
            "    assert t.allclose(out2, t.tensor([1.0, 0.0, 0.0]), atol=1e-6), f'(0,0,1) @ π/2 should be (1,0,0), got {out2.tolist()}'\n"
            "    # Y-component fixed.\n"
            "    out3 = ex3_rotate_vector(t.tensor([0.0, 7.0, 0.0]), half_pi)\n"
            "    assert t.allclose(out3, t.tensor([0.0, 7.0, 0.0]), atol=1e-6), 'pure Y vector must be fixed by Y rotation'\n"
            "    # theta = 0 is identity.\n"
            "    out4 = ex3_rotate_vector(t.tensor([1.0, 2.0, 3.0]), t.tensor(0.0))\n"
            "    assert t.allclose(out4, t.tensor([1.0, 2.0, 3.0]), atol=1e-6), 'theta=0 must be identity'"
        ),
        "solution_body": (
            "def ex3_rotate_vector(v: Tensor, theta: Tensor) -> Tensor:\n"
            "    c = t.cos(theta).item()\n"
            "    s = t.sin(theta).item()\n"
            "    R = t.tensor([\n"
            "        [c, 0.0, s],\n"
            "        [0.0, 1.0, 0.0],\n"
            "        [-s, 0.0, c],\n"
            "    ])\n"
            "    return R @ v"
        ),
        "solution_notes": (
            "**Why `R @ v` not `v @ R`?** Conventional rotation matrices act on "
            "*column* vectors: `v_rotated = R @ v_column`. If you store vectors "
            "as rows (Numpy / PyTorch default), you'd write `v_rotated = v_row @ R.T`. "
            "Same math — just transpose the matrix to match your vector layout.\n\n"
            "**Length preservation.** Rotation matrices are orthogonal (R @ R.T = I), "
            "so `||R @ v|| == ||v||` — a great sanity check during debugging. If "
            "your output norm changes, you almost certainly have a sign error or "
            "an unnormalised axis."
        ),
    },
    {
        "id": "ex4",
        "title": "verify rotation composition R(α)·R(β) = R(α+β)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["composition", "group-property", "matmul-check"],
        "kcs": ["rotation-composes-on-axis"],
        "lo": "Apply matrix multiplication to demonstrate the group property of rotations about a single axis.",
        "prompt_body": (
            "Implement `ex4_compose_rotations(alpha, beta)` to compute the product "
            "`R_y(α) @ R_y(β)` and return it as a `(3, 3)` tensor.\n\n"
            "The test then independently checks that the returned matrix equals "
            "`R_y(α + β)`. This is the 1-parameter group property: rotations about "
            "the same axis add their angles.\n\n"
            "Use your Y rotation matrix construction from Exercise 2."
        ),
        "stub": (
            "def ex4_compose_rotations(alpha: Tensor, beta: Tensor) -> Tensor:\n"
            '    """Return R_y(alpha) @ R_y(beta). Should equal R_y(alpha + beta)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "    def R_y(theta):\n"
            "        c, s = math.cos(theta), math.sin(theta)\n"
            "        return t.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])\n"
            "\n"
            "    alpha = t.tensor(0.3)\n"
            "    beta = t.tensor(1.2)\n"
            "    product = ex4_compose_rotations(alpha, beta)\n"
            "    assert product.shape == (3, 3), f'expected (3,3), got {product.shape}'\n"
            "    # Independent ground truth: R(α + β).\n"
            "    expected = R_y(alpha.item() + beta.item())\n"
            "    assert t.allclose(product, expected, atol=1e-5), f'composition broken:\\n{product}\\nvs\\n{expected}'\n"
            "\n"
            "    # Symmetric case: α + (-α) = 0 → identity.\n"
            "    out_zero = ex4_compose_rotations(t.tensor(0.7), t.tensor(-0.7))\n"
            "    assert t.allclose(out_zero, t.eye(3), atol=1e-5), 'R(α) @ R(-α) must be identity'"
        ),
        "solution_body": (
            "def ex4_compose_rotations(alpha: Tensor, beta: Tensor) -> Tensor:\n"
            "    def R_y(theta):\n"
            "        c = t.cos(theta).item()\n"
            "        s = t.sin(theta).item()\n"
            "        return t.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])\n"
            "    return R_y(alpha) @ R_y(beta)"
        ),
        "solution_notes": (
            "**Why this only works on the same axis.** `R_y(α) @ R_y(β) = R_y(α+β)` "
            "because rotations about the same axis commute. But `R_y(α) @ R_x(β) ≠ "
            "R_x(β) @ R_y(α)` in general — that's the whole point of 3-D orientation "
            "being non-commutative. Don't try to compose Euler angles by adding!\n\n"
            "**Use of the group property.** Want to pre-compute 60 rotations 6° "
            "apart for a turntable? Compute `R_y(6°)` once and matrix-multiply it "
            "into a running accumulator. Cheaper than rebuilding from scratch each "
            "frame."
        ),
    },
    {
        "id": "ex5",
        "title": "rotate a batch of points around Y",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["batch-rotation", "matmul-shape", "ray-tracing", "multi-kc"],
        "kcs": [
            "rotation-matrix-y-construct",
            "rotation-applied-to-vector",
            "rotate-batch-of-points",
        ],
        "lo": "Synthesize matrix construction + matmul-shape arithmetic to rotate a (N, 3) batch of points around the Y axis in one operation.",
        "prompt_body": (
            "Implement `ex5_rotate_batch(points, theta)`. Given a `(N, 3)` batch of "
            "3-D points and a scalar angle `theta`, return a `(N, 3)` batch of "
            "rotated points.\n\n"
            "Strategy: build `R_y(θ)` of shape `(3, 3)` once, then multiply with "
            "the batch. Two equivalent ways:\n"
            "- `points @ R.T` — points are row vectors; transpose R to match.\n"
            "- `(R @ points.T).T` — explicit column-vector form.\n\n"
            "Pick whichever you find clearer. Both produce identical results.\n\n"
            "> ⚠️ **Integrative exercise.** Combines 3 KCs (matrix construction, "
            "matvec → batched matmul shape arithmetic, batch rotation). Empirical "
            "work (Lohr et al. ITiCSE 2025) shows 3-concept exercises drop to "
            "~40% solvability — expect a step up vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_rotate_batch(points: Tensor, theta: Tensor) -> Tensor:\n"
            '    """Rotate a (N, 3) batch of points around Y by theta. Returns (N, 3)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "    half_pi = t.tensor(math.pi / 2)\n"
            "    points = t.tensor([\n"
            "        [1.0, 0.0, 0.0],   # +X axis\n"
            "        [0.0, 0.0, 1.0],   # +Z axis\n"
            "        [0.0, 5.0, 0.0],   # pure Y (must be fixed)\n"
            "        [1.0, 2.0, 0.0],   # XY corner\n"
            "    ])\n"
            "    out = ex5_rotate_batch(points, half_pi)\n"
            "    assert out.shape == (4, 3), f'expected (4,3), got {out.shape}'\n"
            "    expected = t.tensor([\n"
            "        [0.0, 0.0, -1.0],  # X → -Z\n"
            "        [1.0, 0.0, 0.0],   # +Z → +X\n"
            "        [0.0, 5.0, 0.0],   # Y fixed\n"
            "        [0.0, 2.0, -1.0],  # X→-Z, Y fixed\n"
            "    ])\n"
            "    assert t.allclose(out, expected, atol=1e-6), f'value mismatch:\\n{out}\\nvs\\n{expected}'\n"
            "\n"
            "    # Length-preserving check on a random batch.\n"
            "    t.manual_seed(42)\n"
            "    rnd = t.randn(8, 3)\n"
            "    norms_before = rnd.pow(2).sum(dim=1).sqrt()\n"
            "    norms_after = ex5_rotate_batch(rnd, t.tensor(0.7)).pow(2).sum(dim=1).sqrt()\n"
            "    assert t.allclose(norms_before, norms_after, atol=1e-5), 'rotation must preserve per-row norms'\n"
            "\n"
            "    # theta=0 identity.\n"
            "    assert t.allclose(ex5_rotate_batch(points, t.tensor(0.0)), points, atol=1e-6)"
        ),
        "solution_body": (
            "def ex5_rotate_batch(points: Tensor, theta: Tensor) -> Tensor:\n"
            "    c = t.cos(theta).item()\n"
            "    s = t.sin(theta).item()\n"
            "    R = t.tensor([\n"
            "        [c, 0.0, s],\n"
            "        [0.0, 1.0, 0.0],\n"
            "        [-s, 0.0, c],\n"
            "    ])\n"
            "    return points @ R.T"
        ),
        "solution_notes": (
            "**Why `points @ R.T`?** PyTorch stores batches as `(N, D)` — rows are "
            "samples. The rotation formula `v' = R @ v` assumes `v` is a *column* "
            "vector. Two ways to reconcile:\n"
            "- Stack as rows, transpose R: `points_row @ R.T` gives `(N, 3)` directly.\n"
            "- Stack as cols, then transpose result: `(R @ points.T).T` — same math.\n\n"
            "Both compile to the same matmul; pick by readability. `@ R.T` is the "
            "idiomatic PyTorch form because it preserves the `(N, D)` row-major "
            "layout.\n\n"
            "**Where you'll use this.** Camera orbiting, object turntables, "
            "constructing per-vertex normals after rotating a mesh, generating "
            "rotated test data for invariance checks. Anywhere a scene needs to "
            "look at the model from a different angle."
        ),
    },
]


# ----- cell builders ----------------------------------------------------


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "\n".join(lines),
    }


def exercise_header_md(spec: dict, index: int) -> dict:
    keywords_str = ", ".join(spec["keywords"])
    kcs_str = ", ".join(f"`{k}`" for k in spec["kcs"])
    return md(
        f"### Exercise {index} — {spec['title']}",
        "",
        "> ```yaml",
        f"> Difficulty: {spec['difficulty_dots']}",
        f"> Bloom level: {spec['bloom_level']}",
        f"> LO: {spec['lo']}",
        f"> Keywords: {keywords_str}",
        f"> ```",
        "",
        f"**KCs targeted:** {kcs_str}",
        "",
        spec["prompt_body"],
    )


def exercise_code(spec: dict) -> dict:
    return code(
        spec["stub"],
        "",
        "",
        f"def _test_{spec['id']}():",
        f"    {spec['test_body']}",
        f"    _dd_passed.add('{spec['id']}')",
        f'    print("{spec["id"]} ✓")',
        "",
        f"_test_{spec['id']}()",
    )


def exercise_solution_md(spec: dict) -> dict:
    lines = [
        "<details><summary>Solution</summary>",
        "",
        "```python",
        spec["solution_body"],
        "```",
    ]
    if spec["solution_notes"]:
        lines.extend(["", spec["solution_notes"]])
    lines.append("</details>")
    return md(*lines)


# ----- assemble notebook ------------------------------------------------

cells = []

cells.append(md(
    f"# {TITLE}",
    "",
    "> Procedural drill from [Delta Drills](https://delta-drills.vercel.app).",
    f"> Atom: `{ATOM_ID}`. When a test cell passes, your progress is reported back to your account.",
    "",
    "**What you'll practice.** Five rotation-matrix patterns that ramp from `cos/sin` → matrix assembly → rotate-a-vector → composition law → rotate-a-batch. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
    "",
    "**Per-exercise structure** (Doughty et al. ACE 2024 — `[Bloom level] + [LO] + [Keywords] + [KCs]`):",
    "Each exercise begins with a yaml block stating its Bloom cognitive level, learning objective, keywords, and the knowledge components (KCs) it targets. This makes the cognitive demand explicit instead of buried.",
))

cells.append(md("## Setup"))

cells.append(code(
    "import math",
    "import numpy as np",
    "import torch as t",
    "from torch import Tensor",
    "",
    "t.manual_seed(0)",
    "np.random.seed(0)",
))

cells.append(md(
    "## Connect to Delta Drills",
    "",
    f"Paste your Delta Drills auth token below so this drill can report progress on the `{SUBTOPIC}` subtopic.",
    "You can copy the token from your Delta Drills account page.",
    "",
    f"This drill exercises the **atom `{ATOM_ID}`**, which bridges to the bank subtopic `{SUBTOPIC}` for EWMA state. Completing all 5 exercises triggers a single `arena-rating` beacon at the end of the notebook.",
))

cells.append(code(
    "# === Delta Drills auth ===",
    'DD_TOKEN = ""  # paste your token here, then run this cell',
    f'DD_ATOM_ID = "{ATOM_ID}"',
    f'DD_SUBTOPIC = "{SUBTOPIC}"',
    'DD_BACKEND_URL = "https://delta-drills-backend.fly.dev"',
    "",
    "# Track which exercises passed in this session.",
    "_dd_passed = set()",
))

cells.append(md(
    "## Y-axis rotation — quick refresher",
    "",
    "**The matrix.** For a right-hand rotation by `θ` about the Y axis:",
    "```",
    "R_y(θ) = [[ cos θ,  0,  sin θ],",
    "          [ 0,      1,  0    ],",
    "          [-sin θ,  0,  cos θ]]",
    "```",
    "",
    "**Why the middle row is `[0, 1, 0]`.** The Y axis is the rotation axis — anything along Y stays where it is. The X-Z plane is what gets rotated.",
    "",
    "**Right-hand convention.** Looking down the +Y axis, the rotation goes counter-clockwise: +X → -Z, +Z → +X.",
    "",
    "**Acting on vectors.**",
    "- Column-vector: `v' = R @ v` (input shape `(3,)`).",
    "- Batch of row-vectors: `points' = points @ R.T` (input shape `(N, 3)`).",
    "",
    "**Composition.** `R_y(α) @ R_y(β) = R_y(α + β)` — rotations about a single axis add angles.",
))

for i, spec in enumerate(EXERCISE_SPECS, start=1):
    cells.append(exercise_header_md(spec, i))
    cells.append(exercise_code(spec))
    cells.append(exercise_solution_md(spec))

cells.append(md(
    "## Done",
    "",
    "Run the cell below to report your progress to Delta Drills. The beacon fires only if all 5 exercises passed.",
))

cells.append(code(
    "# === Delta Drills completion beacon ===",
    "import urllib.request as _dd_req, json as _dd_json",
    "",
    "_DD_REQUIRED = {'ex1', 'ex2', 'ex3', 'ex4', 'ex5'}",
    "",
    "def _dd_feedback_level(num_passed: int) -> str:",
    '    """Map exercise-pass count → arena-rating feedback enum."""',
    "    if num_passed == 5: return 'not_much'",
    "    if num_passed >= 3: return 'somewhat'",
    "    return 'a_lot'",
    "",
    "def report_completion():",
    "    missing = _DD_REQUIRED - _dd_passed",
    "    if missing:",
    '        print(f"[Delta Drills] {len(missing)} exercises still failing: {sorted(missing)}.")',
    '        print("[Delta Drills] not reporting until all 5 pass.")',
    "        return",
    "    if not DD_TOKEN:",
    "        print('[Delta Drills] DD_TOKEN is empty — completion not reported.')",
    "        return",
    "    body = _dd_json.dumps({",
    "        'exercise_title': f'procedural-drill:{DD_ATOM_ID}',",
    "        'subtopics': [DD_SUBTOPIC],",
    "        'feedback': _dd_feedback_level(len(_dd_passed)),",
    "        'correct': True,",
    "    }).encode('utf-8')",
    "    req = _dd_req.Request(",
    "        f'{DD_BACKEND_URL}/api/practice/arena-rating',",
    "        data=body,",
    "        headers={",
    "            'Content-Type': 'application/json',",
    "            'Authorization': f'Bearer {DD_TOKEN}',",
    "        },",
    "        method='POST',",
    "    )",
    "    try:",
    "        with _dd_req.urlopen(req, timeout=5) as r:",
    "            resp = _dd_json.loads(r.read())",
    "        print(f'[Delta Drills] reported {DD_ATOM_ID} (subtopic={DD_SUBTOPIC!r})')",
    "        print(f'[Delta Drills] EWMA updated: {resp}')",
    "    except Exception as e:",
    "        print(f'[Delta Drills] beacon failed: {e}')",
    "",
    "report_completion()",
))

exercises_metadata = [
    {
        "id": spec["id"],
        "title": spec["title"],
        "bloom_level": spec["bloom_level"],
        "difficulty": spec["difficulty_num"],
        "keywords": spec["keywords"],
        "kcs": spec["kcs"],
        "lo": spec["lo"],
    }
    for spec in EXERCISE_SPECS
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "delta_drills": {
            "atom_id": ATOM_ID,
            "subtopic": SUBTOPIC,
            "drill_kind": "procedural",
            "template_version": TEMPLATE_VERSION,
            "kc_decomposition": KC_DECOMPOSITION,
            "exercises": exercises_metadata,
            "integration_risks": {
                "ex5": "3-KC integrative exercise — at Tutor-Kai solvability cliff. Track student pass rate; do not treat ex5 failure alone as evidence of atom non-mastery.",
            },
            "prompting_pattern": "Doughty et al. ACE 2024 (LO + Bloom + Keywords per exercise)",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for c in nb["cells"]:
    if isinstance(c["source"], str):
        c["source"] = [line + "\n" for line in c["source"].split("\n")]
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")

for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"

verify_solutions(EXERCISE_SPECS)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
print(f"  atom_id: {ATOM_ID}")
print(f"  bridges to: {SUBTOPIC}")
print(f"  template: {TEMPLATE_VERSION}")
print(f"  KCs: {len(KC_DECOMPOSITION)}  ({sum(1 for k in KC_DECOMPOSITION if k['kind'] == 'component-skill')} component, {sum(1 for k in KC_DECOMPOSITION if k['kind'] == 'integrative-skill')} integrative)")
print(f"  exercises: {len(exercises_metadata)}")
for ex in exercises_metadata:
    print(f"    {ex['id']}: {ex['title']} | Bloom={ex['bloom_level']} | diff={ex['difficulty']} | KCs={ex['kcs']}")
