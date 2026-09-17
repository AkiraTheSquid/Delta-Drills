"""Composite drills cx13..cx18 — batch-17 (X-cell, part1 — ARENA ray tracing).

Six composite procedural drills exercising 2-atom pairs from the ray-tracing
machinery (ARENA part 1 — prereqs). Each composite chains `tensor-unbind` or
`unbind-tuple-unpack` against a partner atom to mirror real ARENA part-1 code.

cx13  tensor-unbind        + ray-parametric-form     — unbind O, D from (NR, 2, 3) rays
cx14  tensor-unbind        + linalg-solve-batched    — unbind then solve per-ray
cx15  tensor-unbind        + stack-vs-cat            — unbind then stack along new axis
cx16  tensor-unbind        + unbind-tuple-unpack     — chain unbind then tuple-unpack
cx17  tensor-unbind        + triangle-barycentric    — extract (s, u, v) from solve result
cx18  unbind-tuple-unpack  + ray-parametric-form     — unpack (O, D) from stacked ray tensor
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
# cx13 — unbind (O, D) from a (NR, 2, 3) ray tensor, then build ray(t) = O + tD
# ===========================================================================
spec_13 = {
    "atom_ids": ["tensor-unbind", "ray-parametric-form"],
    "subtopics": _subs(["tensor-unbind", "ray-parametric-form"]),
    "primary_atom": "tensor-unbind",
    "part": "part1",
    "exercise_index": 13,
    "exercise_title": "unbind O,D from rays of shape (NR, 2, 3) and evaluate ray(t)",
    "slug": "unbind-rays-then-parametric-eval",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "In ARENA part 1, the canonical ray layout is `rays: (NR, 2, 3)` — each ray is a stacked "
        "`(origin, direction)` pair along axis 1. To evaluate the ray's parametric form `P(t) = O + t * D` "
        "we first need to *separate* the origin and direction tensors. `tensor-unbind` along axis 1 does "
        "this in one named call, returning a tuple `(O, D)` where each has shape `(NR, 3)`.\n\n"
        "The composition is `unbind` → `ray-parametric-form`. `unbind(rays, dim=1)` splits along the "
        "stacked-pair axis (no copy — each output is a stride view), and the parametric eval is just "
        "`O + t * D` with `t` broadcasting against the per-ray axis. The unbind is load-bearing: a wrong "
        "`dim` argument (e.g. `dim=0`) would give you `NR` tensors of shape `(2, 3)` instead of two "
        "tensors of shape `(NR, 3)`, and the parametric eval would silently give nonsense."
    ),
    "prompt_body": (
        "Implement `cx13_ray_at_t(rays, t_scalar)` that takes a ray batch of shape `(NR, 2, 3)` (axis 1 "
        "is the stacked `(origin, direction)` pair) and a scalar `t_scalar: float`, and returns the "
        "per-ray point `P(t) = O + t_scalar * D` of shape `(NR, 3)`.\n\n"
        "1. **Unbind** along axis 1 to separate origin and direction: `O, D = t.unbind(rays, dim=1)`. "
        "Both `O` and `D` have shape `(NR, 3)`.\n"
        "2. **Apply the parametric form** `O + t_scalar * D`. Broadcasting against the scalar is "
        "automatic.\n\n"
        "Return shape `(NR, 3)`. The test verifies `O` and `D` were taken from the correct axis (axis 1, "
        "not axis 0) and that the parametric eval matches a hand-computed reference."
    ),
    "stub_body": (
        "def cx13_ray_at_t(rays, t_scalar):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built rays — origins at the column index, directions all unit-x.\n"
        "# rays[i, 0] = origin_i = (i, 0, 0); rays[i, 1] = direction_i = (1, 0, 0).\n"
        "NR = 5\n"
        "origins = t.stack([t.tensor([float(i), 0.0, 0.0]) for i in range(NR)])  # (NR, 3)\n"
        "directions = t.stack([t.tensor([1.0, 0.0, 0.0]) for _ in range(NR)])    # (NR, 3)\n"
        "rays = t.stack([origins, directions], dim=1)  # (NR, 2, 3)\n"
        "assert tuple(rays.shape) == (NR, 2, 3)\n"
        "\n"
        "# At t=2: P_i = (i + 2, 0, 0).\n"
        "out = cx13_ray_at_t(rays, 2.0)\n"
        "assert tuple(out.shape) == (NR, 3), f'expected (NR,3), got {tuple(out.shape)}'\n"
        "expected = t.stack([t.tensor([float(i) + 2.0, 0.0, 0.0]) for i in range(NR)])\n"
        "assert t.allclose(out, expected), f'parametric eval wrong: {out}\\nexpected: {expected}'\n"
        "\n"
        "# Case B: t=0 should return origins exactly.\n"
        "out0 = cx13_ray_at_t(rays, 0.0)\n"
        "assert t.allclose(out0, origins), 'at t=0 result must equal origins'\n"
        "\n"
        "# Case C: random rays — cross-check against manual indexing.\n"
        "rays2 = t.randn(7, 2, 3)\n"
        "out2 = cx13_ray_at_t(rays2, 1.5)\n"
        "ref = rays2[:, 0, :] + 1.5 * rays2[:, 1, :]\n"
        "assert t.allclose(out2, ref), 'unbind axis is wrong — did you use dim=1?'\n"
        "\n"
        "# Case D: negative t (ray going backward).\n"
        "out_neg = cx13_ray_at_t(rays2, -0.5)\n"
        "ref_neg = rays2[:, 0, :] - 0.5 * rays2[:, 1, :]\n"
        "assert t.allclose(out_neg, ref_neg)"
    ),
    "solution_body": (
        "def cx13_ray_at_t(rays, t_scalar):\n"
        "    # Atom A (tensor-unbind): split the stacked (origin, direction) pair along axis 1.\n"
        "    O, D = t.unbind(rays, dim=1)\n"
        "    # Atom B (ray-parametric-form): P(t) = O + t * D.\n"
        "    return O + t_scalar * D"
    ),
    "solution_notes": (
        "`t.unbind(rays, dim=1)` returns a tuple of length 2 (the size of axis 1), each of shape "
        "`(NR, 3)`. Tuple-unpacking via `O, D = ...` is idiomatic. Common bug: using `dim=0` gives you "
        "`NR` tensors of shape `(2, 3)` (one per ray) — wrong axis. Another bug: using "
        "`rays[:, 0]` / `rays[:, 1]` works too but is less explicit about WHY we're splitting; unbind "
        "documents the intent (\"this axis enumerates the stacked components\")."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["tensor-unbind", "ray-parametric-form"],
    "lo": (
        "Compose tensor-unbind (split a stacked-pair axis) with the ray parametric form O + tD to "
        "evaluate a batch of rays at a scalar parameter."
    ),
}


# ===========================================================================
# cx14 — unbind the per-ray LHS / RHS, then solve the per-ray 3x3 system
# ===========================================================================
spec_14 = {
    "atom_ids": ["tensor-unbind", "linalg-solve-batched"],
    "subtopics": _subs(["tensor-unbind", "linalg-solve-batched"]),
    "primary_atom": "tensor-unbind",
    "part": "part1",
    "exercise_index": 14,
    "exercise_title": "unbind per-ray LHS / RHS columns then linalg.solve in batch",
    "slug": "unbind-then-batched-linalg-solve",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Ray-triangle intersection in ARENA part 1 reduces to a 3x3 linear system per ray:\n"
        "  `[-D | (B - A) | (C - A)] · [s, u, v]^T = (O - A)`\n\n"
        "When you're packing per-ray inputs, it's natural to STACK the three column vectors into a "
        "single tensor `cols: (NR, 3, 3)` along axis 2 (each column is one of `-D`, `B-A`, `C-A`). To "
        "feed this to `torch.linalg.solve` you don't actually need to split it — `solve` takes the full "
        "matrix. But the RHS often arrives stacked too (e.g. multiple offsets per ray), and `unbind` is "
        "the canonical way to peel off the slice you want.\n\n"
        "This composition pairs `tensor-unbind` (peel a per-ray RHS off a `(NR, K, 3)` block) with "
        "`linalg-solve-batched` (solve all NR systems in one batched call). The batched solve is the "
        "load-bearing atom — looping over rays would work but is 50-100x slower."
    ),
    "prompt_body": (
        "Implement `cx14_solve_per_ray(mats, rhs_stack, which)` that solves a batch of 3x3 linear "
        "systems, one per ray.\n\n"
        "- `mats: (NR, 3, 3)` — per-ray coefficient matrices.\n"
        "- `rhs_stack: (NR, K, 3)` — a stack of K per-ray RHS vectors. Axis 1 enumerates which RHS "
        "  variant.\n"
        "- `which: int` — index in `[0, K)` selecting which RHS variant to solve against.\n\n"
        "1. **Unbind** `rhs_stack` along axis 1 to get a tuple of K tensors each of shape `(NR, 3)`. "
        "Select element `which` from that tuple — this is your per-ray RHS.\n"
        "2. **Batched solve** with `torch.linalg.solve(mats, rhs)` — the trailing 3-axis of `rhs` is "
        "treated as the column vector and broadcasting handles the batch axis. Returns shape `(NR, 3)`.\n\n"
        "Cross-check against a Python for-loop calling `torch.linalg.solve` on each ray individually."
    ),
    "stub_body": (
        "def cx14_solve_per_ray(mats, rhs_stack, which):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: small, hand-built — invertible diagonals, known solutions.\n"
        "NR, K = 4, 3\n"
        "# Per-ray identity-scaled matrices: mats[i] = (i+1) * I_3.\n"
        "mats = t.stack([(i + 1.0) * t.eye(3) for i in range(NR)])  # (NR, 3, 3)\n"
        "# RHS stack: K=3 variants. Variant 0 = ones; variant 1 = [1,2,3]; variant 2 = [4,5,6].\n"
        "v0 = t.ones(NR, 3)\n"
        "v1 = t.tensor([[1.0, 2.0, 3.0]]).expand(NR, 3).contiguous()\n"
        "v2 = t.tensor([[4.0, 5.0, 6.0]]).expand(NR, 3).contiguous()\n"
        "rhs_stack = t.stack([v0, v1, v2], dim=1)  # (NR, 3, 3) — K=3 along axis 1\n"
        "assert tuple(rhs_stack.shape) == (NR, K, 3)\n"
        "\n"
        "# Pick variant 1 ([1,2,3]). Solution per ray = rhs / (i+1).\n"
        "out = cx14_solve_per_ray(mats, rhs_stack, which=1)\n"
        "assert tuple(out.shape) == (NR, 3)\n"
        "expected = t.stack([t.tensor([1.0, 2.0, 3.0]) / (i + 1.0) for i in range(NR)])\n"
        "assert t.allclose(out, expected, atol=1e-5), f'got {out}, expected {expected}'\n"
        "\n"
        "# Case B: variant 0 (ones) — solution per ray = 1/(i+1) * ones.\n"
        "out0 = cx14_solve_per_ray(mats, rhs_stack, which=0)\n"
        "expected0 = t.stack([t.full((3,), 1.0 / (i + 1.0)) for i in range(NR)])\n"
        "assert t.allclose(out0, expected0, atol=1e-5)\n"
        "\n"
        "# Case C: random invertible mats — cross-check against per-ray loop.\n"
        "t.manual_seed(42)\n"
        "mats2 = t.randn(6, 3, 3) + 4.0 * t.eye(3)  # well-conditioned\n"
        "rhs2 = t.randn(6, 2, 3)\n"
        "out2 = cx14_solve_per_ray(mats2, rhs2, which=0)\n"
        "ref2 = t.stack([t.linalg.solve(mats2[i], rhs2[i, 0]) for i in range(6)])\n"
        "assert t.allclose(out2, ref2, atol=1e-4), f'batched solve diverged from per-ray loop'"
    ),
    "solution_body": (
        "def cx14_solve_per_ray(mats, rhs_stack, which):\n"
        "    # Atom A (tensor-unbind): split the K-axis (axis 1) into a tuple of K tensors\n"
        "    # each of shape (NR, 3).\n"
        "    rhs_variants = t.unbind(rhs_stack, dim=1)\n"
        "    rhs = rhs_variants[which]  # (NR, 3)\n"
        "    # Atom B (linalg-solve-batched): solve all NR 3x3 systems in one call.\n"
        "    return t.linalg.solve(mats, rhs)"
    ),
    "solution_notes": (
        "`t.unbind(rhs_stack, dim=1)` returns a Python tuple — indexing `[which]` is a constant-time "
        "pick of one stride-view. No copy until `linalg.solve` actually consumes it. For "
        "`linalg.solve(mats, rhs)` where `mats: (NR, 3, 3)` and `rhs: (NR, 3)`, PyTorch treats `rhs` as "
        "a per-ray column vector (the trailing 3 is the system dimension). If you accidentally pass the "
        "full `rhs_stack` (without unbinding), the solver tries to solve K right-hand-sides per ray and "
        "you get back `(NR, 3, K)` (or worse, a shape error). Unbind is the explicit-selection step."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["tensor-unbind", "linalg-solve-batched"],
    "lo": (
        "Compose tensor-unbind (split a stacked-RHS axis) with torch.linalg.solve's batched 3x3 "
        "interface to solve NR per-ray linear systems in one call."
    ),
}


# ===========================================================================
# cx15 — unbind components, optionally transform each, then re-stack along a NEW axis
# ===========================================================================
spec_15 = {
    "atom_ids": ["tensor-unbind", "stack-vs-cat"],
    "subtopics": _subs(["tensor-unbind", "stack-vs-cat"]),
    "primary_atom": "tensor-unbind",
    "part": "part1",
    "exercise_index": 15,
    "exercise_title": "unbind components then re-stack along a new axis (stack vs cat)",
    "slug": "unbind-then-stack-new-axis",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A frequent ARENA pattern is the round-trip: take a packed tensor, *unbind* it along one axis "
        "to apply per-component transformations, then *stack* the results back into a new packed tensor. "
        "The composition exercises the **stack vs cat** distinction sharply: `stack` inserts a NEW axis "
        "(grows ndim by 1), while `cat` glues along an EXISTING axis (preserves ndim).\n\n"
        "Use case: given `rays: (NR, 2, 3)`, you want to scale the direction component (axis 1, index 1) "
        "by `dir_scale` but leave origin unchanged, then reassemble. The natural sequence is\n"
        "  1. `O, D = t.unbind(rays, dim=1)` — both `(NR, 3)`.\n"
        "  2. `D_scaled = D * dir_scale`.\n"
        "  3. `t.stack([O, D_scaled], dim=1)` — reinserts the size-2 axis at position 1, recovering "
        "`(NR, 2, 3)`.\n\n"
        "Using `t.cat` here would be a bug: `cat([O, D_scaled], dim=1)` produces `(NR, 6)` (concatenated "
        "along the 3-axis), losing the structural split."
    ),
    "prompt_body": (
        "Implement `cx15_scale_direction(rays, dir_scale)` that takes `rays: (NR, 2, 3)` (axis 1 = "
        "[origin, direction]) and a scalar `dir_scale`, scales each ray's direction vector by "
        "`dir_scale` (leaves origins unchanged), and returns the rebuilt `(NR, 2, 3)` tensor.\n\n"
        "1. **Unbind** along axis 1: `O, D = t.unbind(rays, dim=1)`.\n"
        "2. Compute `D_scaled = D * dir_scale`.\n"
        "3. **Stack** along a NEW axis at position 1: `t.stack([O, D_scaled], dim=1)`. The result "
        "shape must equal `rays.shape` exactly — `(NR, 2, 3)`.\n\n"
        "Common bug: `t.cat([O, D_scaled], dim=1)` gives `(NR, 6)` — wrong shape, wrong structure. The "
        "test catches that explicitly."
    ),
    "stub_body": (
        "def cx15_scale_direction(rays, dir_scale):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built rays, scale by 3.\n"
        "NR = 4\n"
        "O = t.stack([t.tensor([float(i), 0.0, 0.0]) for i in range(NR)])\n"
        "D = t.stack([t.tensor([0.0, 1.0, 0.0]) for _ in range(NR)])\n"
        "rays = t.stack([O, D], dim=1)\n"
        "assert tuple(rays.shape) == (NR, 2, 3)\n"
        "\n"
        "out = cx15_scale_direction(rays, 3.0)\n"
        "assert tuple(out.shape) == (NR, 2, 3), (\n"
        "    f'shape changed: {tuple(out.shape)}. '\n"
        "    'Did you use cat instead of stack? cat would give (NR, 6).'\n"
        ")\n"
        "# Origins unchanged.\n"
        "assert t.allclose(out[:, 0, :], O), 'origins should be unchanged'\n"
        "# Directions scaled by 3.\n"
        "assert t.allclose(out[:, 1, :], D * 3.0), 'directions should be 3x'\n"
        "\n"
        "# Case B: dir_scale = 0 — directions zeroed, origins intact.\n"
        "out0 = cx15_scale_direction(rays, 0.0)\n"
        "assert t.allclose(out0[:, 0, :], O)\n"
        "assert t.allclose(out0[:, 1, :], t.zeros_like(D))\n"
        "\n"
        "# Case C: random rays, negative scale.\n"
        "rays2 = t.randn(7, 2, 3)\n"
        "out2 = cx15_scale_direction(rays2, -1.0)\n"
        "assert tuple(out2.shape) == (7, 2, 3)\n"
        "assert t.allclose(out2[:, 0, :], rays2[:, 0, :])\n"
        "assert t.allclose(out2[:, 1, :], -rays2[:, 1, :])\n"
        "\n"
        "# Case D: stress — explicit cross-check via manual axis index.\n"
        "expected = rays2.clone()\n"
        "expected[:, 1, :] = expected[:, 1, :] * 2.5\n"
        "out3 = cx15_scale_direction(rays2, 2.5)\n"
        "assert t.allclose(out3, expected)"
    ),
    "solution_body": (
        "def cx15_scale_direction(rays, dir_scale):\n"
        "    # Atom A (tensor-unbind): peel apart the (origin, direction) pair along axis 1.\n"
        "    O, D = t.unbind(rays, dim=1)\n"
        "    D_scaled = D * dir_scale\n"
        "    # Atom B (stack-vs-cat): stack INSERTS a new axis of size 2 at position 1.\n"
        "    # cat would concatenate along an existing axis -> wrong shape.\n"
        "    return t.stack([O, D_scaled], dim=1)"
    ),
    "solution_notes": (
        "The unbind/stack pair is the structural inverse: `t.stack(t.unbind(x, dim=d), dim=d)` returns "
        "a tensor with the same shape AND values as `x`. That property is what makes this round-trip "
        "safe for transformations that don't change per-component shape. If you reach for `t.cat` "
        "instead, the axis-1 dimension disappears (it's glued into axis 0 or axis 2 depending on dim) — "
        "the shape signal alone catches the bug. `stack` introduces a new axis; `cat` merges along an "
        "existing one. Memorize that one-liner."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["tensor-unbind", "stack-vs-cat"],
    "lo": (
        "Compose tensor-unbind with t.stack (along a new axis) to round-trip a packed-pair tensor "
        "through a per-component transformation, distinguishing stack from cat."
    ),
}


# ===========================================================================
# cx16 — chain unbind with tuple-unpack across two axes
# ===========================================================================
spec_16 = {
    "atom_ids": ["tensor-unbind", "unbind-tuple-unpack"],
    "subtopics": _subs(["tensor-unbind", "unbind-tuple-unpack"]),
    "primary_atom": "tensor-unbind",
    "part": "part1",
    "exercise_index": 16,
    "exercise_title": "chained unbind + tuple-unpack: extract three triangle vertices",
    "slug": "chained-unbind-tuple-unpack-triangle-verts",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA part 1 represents triangle batches as `triangles: (NT, 3, 3)` — axis 1 enumerates the "
        "three vertices `(A, B, C)`, axis 2 is the spatial 3-vector. The natural way to pull out the "
        "three vertices is\n"
        "  `A, B, C = t.unbind(triangles, dim=1)`\n"
        "exercising both atoms in one line: `tensor-unbind` to split, `unbind-tuple-unpack` to bind "
        "the three returned tensors to named variables simultaneously.\n\n"
        "Why this matters: the Python tuple-unpack idiom asserts at runtime that the LHS arity matches "
        "the RHS tuple length. If you wrote `A, B = t.unbind(triangles, dim=1)` you'd get `ValueError: "
        "too many values to unpack (expected 2)` — a structural assertion for free. That's why ARENA "
        "code uses unbind+tuple-unpack instead of three separate `triangles[:, i]` indexing calls."
    ),
    "prompt_body": (
        "Implement `cx16_triangle_normals(triangles)` that computes per-triangle unit-normal vectors "
        "via the chained unbind/tuple-unpack pattern.\n\n"
        "Input: `triangles: (NT, 3, 3)` — axis 1 enumerates vertices `(A, B, C)`, axis 2 is the 3-vector "
        "coordinates.\n\n"
        "1. **Unbind + tuple-unpack** the three vertices in ONE statement: "
        "`A, B, C = t.unbind(triangles, dim=1)`. Each is `(NT, 3)`.\n"
        "2. Compute the edge vectors `e1 = B - A` and `e2 = C - A`, each `(NT, 3)`.\n"
        "3. Compute the cross product `n = t.cross(e1, e2, dim=-1)` (the unnormalized normal).\n"
        "4. Return the L2-normalized normal: divide by `t.linalg.vector_norm(n, dim=-1, keepdim=True)`. "
        "Result shape `(NT, 3)`.\n\n"
        "The test asserts that the three unbound tensors map to vertices A, B, C in order — i.e. the "
        "tuple-unpack arity is correct and ordering matches axis 1."
    ),
    "stub_body": (
        "def cx16_triangle_normals(triangles):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built triangles in the z=0 plane — normals must be ±z.\n"
        "# Triangle 0: A=(0,0,0) B=(1,0,0) C=(0,1,0) — normal +z.\n"
        "# Triangle 1: A=(0,0,0) B=(0,1,0) C=(1,0,0) — reversed winding -> normal -z.\n"
        "tri0 = t.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])\n"
        "tri1 = t.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])\n"
        "triangles = t.stack([tri0, tri1])  # (2, 3, 3)\n"
        "out = cx16_triangle_normals(triangles)\n"
        "assert tuple(out.shape) == (2, 3), f'expected (2,3), got {tuple(out.shape)}'\n"
        "assert t.allclose(out[0], t.tensor([0.0, 0.0, 1.0]), atol=1e-6), f'tri0 normal: {out[0]}'\n"
        "assert t.allclose(out[1], t.tensor([0.0, 0.0, -1.0]), atol=1e-6), f'tri1 normal: {out[1]}'\n"
        "\n"
        "# Case B: random triangles — verify the normal is perpendicular to each edge.\n"
        "t.manual_seed(0)\n"
        "tris = t.randn(8, 3, 3)\n"
        "normals = cx16_triangle_normals(tris)\n"
        "assert tuple(normals.shape) == (8, 3)\n"
        "# Unit length.\n"
        "norms = t.linalg.vector_norm(normals, dim=-1)\n"
        "assert t.allclose(norms, t.ones(8), atol=1e-5), f'normals not unit length: {norms}'\n"
        "# Perpendicular to e1 = B - A and e2 = C - A.\n"
        "e1 = tris[:, 1, :] - tris[:, 0, :]\n"
        "e2 = tris[:, 2, :] - tris[:, 0, :]\n"
        "dot1 = (normals * e1).sum(dim=-1)\n"
        "dot2 = (normals * e2).sum(dim=-1)\n"
        "assert t.allclose(dot1, t.zeros(8), atol=1e-4), f'normal not perp to e1: {dot1}'\n"
        "assert t.allclose(dot2, t.zeros(8), atol=1e-4), f'normal not perp to e2: {dot2}'\n"
        "\n"
        "# Case C: structural — ensure the tuple-unpack ordering is A,B,C and not (e.g.) C,B,A.\n"
        "# Build a triangle where swapping vertices changes the normal sign.\n"
        "tri_canonical = t.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])\n"
        "n_canonical = cx16_triangle_normals(tri_canonical)\n"
        "assert t.allclose(n_canonical[0], t.tensor([0.0, 0.0, 1.0]), atol=1e-6), (\n"
        "    'normal sign wrong — did you unbind A,B,C in the wrong order? '\n"
        "    f'got {n_canonical[0]}'\n"
        ")"
    ),
    "solution_body": (
        "def cx16_triangle_normals(triangles):\n"
        "    # Atoms A+B combined (tensor-unbind + unbind-tuple-unpack): one statement, three\n"
        "    # named bindings. The tuple-unpack arity (3) asserts at runtime that axis 1 has size 3.\n"
        "    A, B, C = t.unbind(triangles, dim=1)\n"
        "    e1 = B - A\n"
        "    e2 = C - A\n"
        "    n = t.cross(e1, e2, dim=-1)\n"
        "    return n / t.linalg.vector_norm(n, dim=-1, keepdim=True)"
    ),
    "solution_notes": (
        "The single-line `A, B, C = t.unbind(triangles, dim=1)` is the canonical ARENA idiom. It does "
        "TWO things at once: (1) splits the size-3 vertex axis into a 3-tuple of `(NT, 3)` tensors, and "
        "(2) binds each element to a meaningfully-named variable. The structural assertion is free — "
        "Python raises `ValueError` if the tuple length doesn't match the LHS arity. Compare with "
        "`triangles[:, 0]; triangles[:, 1]; triangles[:, 2]`: three separate indexing calls, no name "
        "binding, no arity assertion. The unbind+unpack form is tighter and self-documenting."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["tensor-unbind", "unbind-tuple-unpack"],
    "lo": (
        "Compose tensor-unbind with Python tuple-unpacking to extract three named triangle vertices in "
        "one statement, exercising the arity-check property as a structural assertion."
    ),
}


# ===========================================================================
# cx17 — extract (s, u, v) from the per-ray linalg.solve result via unbind
# ===========================================================================
spec_17 = {
    "atom_ids": ["tensor-unbind", "triangle-barycentric"],
    "subtopics": _subs(["tensor-unbind", "triangle-barycentric"]),
    "primary_atom": "tensor-unbind",
    "part": "part1",
    "exercise_index": 17,
    "exercise_title": "unbind (s,u,v) from a (NR, 3) solve result and classify intersections",
    "slug": "unbind-suv-then-barycentric-test",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's ray-triangle intersection solves a 3x3 linear system per ray and gets back a `(NR, 3)` "
        "tensor where the three components are `(s, u, v)`: `s` is the parametric distance along the "
        "ray, and `(u, v)` are the **barycentric coordinates** locating the intersection inside the "
        "triangle.\n\n"
        "The valid-intersection test combines barycentric facts:\n"
        "  - `u >= 0`\n"
        "  - `v >= 0`\n"
        "  - `u + v <= 1`     (this is the triangle-barycentric atom — the simplex constraint)\n"
        "  - `s >= 0`         (ray facing forward, not behind the camera)\n\n"
        "To express this cleanly you `unbind` the `(NR, 3)` solve result along axis 1 to get three "
        "`(NR,)` tensors `s, u, v`, then combine the four boolean constraints. The named-vector unpack "
        "is what makes the test readable — without it you'd be writing `solve_result[:, 1] >= 0` four "
        "times."
    ),
    "prompt_body": (
        "Implement `cx17_classify_hits(solve_result)` where `solve_result: (NR, 3)` packs the per-ray "
        "`(s, u, v)` solution from `torch.linalg.solve`.\n\n"
        "1. **Unbind** along axis 1 to get three `(NR,)` tensors: `s, u, v = t.unbind(solve_result, "
        "dim=1)`.\n"
        "2. Apply the **barycentric** + ray-forward validity test elementwise:\n"
        "   - `u >= 0`\n"
        "   - `v >= 0`\n"
        "   - `u + v <= 1`\n"
        "   - `s >= 0`\n"
        "3. Combine all four with `&` (boolean AND).\n\n"
        "Return a boolean `(NR,)` tensor — `True` for valid intersections, `False` otherwise."
    ),
    "stub_body": (
        "def cx17_classify_hits(solve_result):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built (s, u, v) triples spanning all rejection cases.\n"
        "# Layout: [s, u, v] per ray.\n"
        "solve_result = t.tensor([\n"
        "    [1.0,  0.25, 0.25],   # 0: valid (interior)\n"
        "    [0.0,  0.0,  0.0 ],   # 1: valid (vertex A; boundary)\n"
        "    [2.0,  0.5,  0.5 ],   # 2: valid (edge BC, u+v == 1)\n"
        "    [-0.5, 0.3,  0.3 ],   # 3: invalid (s < 0, behind camera)\n"
        "    [1.0,  -0.1, 0.3 ],   # 4: invalid (u < 0)\n"
        "    [1.0,  0.3,  -0.1],   # 5: invalid (v < 0)\n"
        "    [1.0,  0.7,  0.7 ],   # 6: invalid (u + v > 1)\n"
        "    [3.0,  0.2,  0.5 ],   # 7: valid\n"
        "])\n"
        "out = cx17_classify_hits(solve_result)\n"
        "assert out.dtype == t.bool, f'expected bool tensor, got dtype {out.dtype}'\n"
        "assert tuple(out.shape) == (8,), f'expected (8,), got {tuple(out.shape)}'\n"
        "expected = t.tensor([True, True, True, False, False, False, False, True])\n"
        "assert t.equal(out, expected), f'classification mismatch: {out} vs {expected}'\n"
        "\n"
        "# Case B: large random batch — at least cross-check against a reference computation.\n"
        "t.manual_seed(7)\n"
        "sr = t.randn(64, 3)\n"
        "out2 = cx17_classify_hits(sr)\n"
        "s_ref = sr[:, 0]\n"
        "u_ref = sr[:, 1]\n"
        "v_ref = sr[:, 2]\n"
        "ref = (u_ref >= 0) & (v_ref >= 0) & ((u_ref + v_ref) <= 1) & (s_ref >= 0)\n"
        "assert t.equal(out2, ref), 'unbind ordering wrong — must be (s, u, v) along axis 1'\n"
        "\n"
        "# Case C: empty batch — degenerate but mustn't crash.\n"
        "out3 = cx17_classify_hits(t.empty(0, 3))\n"
        "assert tuple(out3.shape) == (0,)\n"
        "assert out3.dtype == t.bool"
    ),
    "solution_body": (
        "def cx17_classify_hits(solve_result):\n"
        "    # Atom A (tensor-unbind): extract the three named components from the (NR, 3) result.\n"
        "    s, u, v = t.unbind(solve_result, dim=1)\n"
        "    # Atom B (triangle-barycentric): the inside-triangle test is u>=0, v>=0, u+v<=1;\n"
        "    # plus s>=0 ensures the hit is along the forward ray direction.\n"
        "    return (u >= 0) & (v >= 0) & ((u + v) <= 1) & (s >= 0)"
    ),
    "solution_notes": (
        "The barycentric simplex constraint `u >= 0, v >= 0, u + v <= 1` describes the interior + "
        "boundary of the standard 2-simplex. Adding `s >= 0` filters out hits BEHIND the camera (the "
        "ray equation `O + s*D` with negative `s` points the wrong way). Unbind makes the four "
        "constraints readable as four short boolean ops; alternative `solve_result[:, 0] >= 0` style "
        "code obscures which component is which. Note the edge case `u + v == 1` (lying on edge BC) is "
        "INCLUDED — `<=` not `<`. Many bugs come from flipping that to strict inequality."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["tensor-unbind", "triangle-barycentric"],
    "lo": (
        "Compose tensor-unbind with the triangle barycentric inside-test (u>=0, v>=0, u+v<=1) and "
        "the ray-forward constraint (s>=0) to classify a batch of ray-triangle intersections."
    ),
}


# ===========================================================================
# cx18 — tuple-unpack (O, D) from a stacked ray tensor, then parametric eval
# ===========================================================================
spec_18 = {
    "atom_ids": ["unbind-tuple-unpack", "ray-parametric-form"],
    "subtopics": _subs(["unbind-tuple-unpack", "ray-parametric-form"]),
    "primary_atom": "unbind-tuple-unpack",
    "part": "part1",
    "exercise_index": 18,
    "exercise_title": "tuple-unpack (O, D) from stacked rays then evaluate at a per-ray t vector",
    "slug": "tuple-unpack-rays-then-vector-t-eval",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Where cx13 used a scalar `t`, the real ARENA intersection code evaluates rays at a "
        "PER-RAY parameter — typically `s: (NR,)` returned by the batched linalg.solve. The composition "
        "is the same shape (unbind → parametric eval) but with broadcasting subtlety.\n\n"
        "Given `rays: (NR, 2, 3)` and `t_vec: (NR,)`, the parametric form `P_i = O_i + t_vec_i * D_i` "
        "requires the scalar `t_vec_i` to broadcast against the 3-vector `D_i`. Standard broadcasting "
        "DOES NOT do this directly: `t_vec: (NR,) * D: (NR, 3)` aligns the trailing axes WRONG — `(NR,)` "
        "tries to align against the 3-axis. The fix is to add a trailing 1-axis: `t_vec[:, None]` (or "
        "`rearrange(t_vec, 'nr -> nr 1')`), giving `(NR, 1)` which broadcasts against `(NR, 3)` "
        "correctly.\n\n"
        "Atom focus: `unbind-tuple-unpack` is the named-binding step; `ray-parametric-form` is the "
        "broadcasting-aware evaluation. Both load-bearing."
    ),
    "prompt_body": (
        "Implement `cx18_ray_at_per_ray_t(rays, t_vec)` where:\n\n"
        "- `rays: (NR, 2, 3)` — axis 1 = [origin, direction]\n"
        "- `t_vec: (NR,)` — a per-ray scalar parameter\n\n"
        "Return `P: (NR, 3)` with `P_i = O_i + t_vec_i * D_i`.\n\n"
        "1. **Tuple-unpack via unbind**: `O, D = t.unbind(rays, dim=1)` — the arity-2 unpack asserts "
        "that axis 1 has size 2.\n"
        "2. **Broadcasting fix-up**: `t_vec[:, None]` to reshape `(NR,)` → `(NR, 1)` so it broadcasts "
        "against `D: (NR, 3)` along the correct axis. (Plain `t_vec * D` is a bug — wrong axis "
        "alignment.)\n"
        "3. **Parametric form**: `O + t_vec[:, None] * D`.\n\n"
        "Return shape `(NR, 3)`."
    ),
    "stub_body": (
        "def cx18_ray_at_per_ray_t(rays, t_vec):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built rays at origin, all directions = +x; per-ray t = [0, 1, 2, 3].\n"
        "# Expected: P_i = (i, 0, 0).\n"
        "NR = 4\n"
        "O = t.zeros(NR, 3)\n"
        "D = t.zeros(NR, 3); D[:, 0] = 1.0\n"
        "rays = t.stack([O, D], dim=1)\n"
        "t_vec = t.tensor([0.0, 1.0, 2.0, 3.0])\n"
        "out = cx18_ray_at_per_ray_t(rays, t_vec)\n"
        "assert tuple(out.shape) == (NR, 3), f'expected (NR,3), got {tuple(out.shape)}'\n"
        "expected = t.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])\n"
        "assert t.allclose(out, expected), f'got {out}\\nexpected {expected}'\n"
        "\n"
        "# Case B: random rays + random t_vec — cross-check against an explicit per-ray loop.\n"
        "t.manual_seed(11)\n"
        "rays2 = t.randn(7, 2, 3)\n"
        "tv2 = t.randn(7)\n"
        "out2 = cx18_ray_at_per_ray_t(rays2, tv2)\n"
        "ref2 = t.stack([rays2[i, 0] + tv2[i] * rays2[i, 1] for i in range(7)])\n"
        "assert t.allclose(out2, ref2, atol=1e-5), (\n"
        "    'broadcast mismatch — did you forget t_vec[:, None]? '\n"
        "    'Plain t_vec * D aligns t_vec against the 3-axis (wrong).'\n"
        ")\n"
        "\n"
        "# Case C: t_vec of all zeros — must reduce to origins exactly.\n"
        "out0 = cx18_ray_at_per_ray_t(rays2, t.zeros(7))\n"
        "assert t.allclose(out0, rays2[:, 0, :])\n"
        "\n"
        "# Case D: t_vec of all ones — must reduce to O + D.\n"
        "out1 = cx18_ray_at_per_ray_t(rays2, t.ones(7))\n"
        "ref1 = rays2[:, 0, :] + rays2[:, 1, :]\n"
        "assert t.allclose(out1, ref1)"
    ),
    "solution_body": (
        "def cx18_ray_at_per_ray_t(rays, t_vec):\n"
        "    # Atom A (unbind-tuple-unpack): named bindings for O and D. The arity-2 LHS asserts\n"
        "    # at runtime that axis 1 has size 2.\n"
        "    O, D = t.unbind(rays, dim=1)\n"
        "    # Atom B (ray-parametric-form): need t_vec[:, None] so (NR, 1) broadcasts against\n"
        "    # D: (NR, 3) on the correct axis. Plain t_vec * D would align (NR,) against the\n"
        "    # 3-axis — wrong.\n"
        "    return O + t_vec[:, None] * D"
    ),
    "solution_notes": (
        "The `[:, None]` (or equivalently `.unsqueeze(-1)` / `rearrange(t_vec, 'nr -> nr 1')`) is the "
        "load-bearing broadcasting fix. Without it, PyTorch tries to align `(NR,)` against the trailing "
        "axis of `D: (NR, 3)` — i.e. the 3-axis — and you get either a shape error (if NR != 3) or, "
        "worse, *no* error and a silently-wrong result (if NR happens to equal 3). Always reshape your "
        "per-row scalar to `(NR, 1)` before broadcasting against a `(NR, D)` tensor. This is the same "
        "pattern as keepdim=True in cx27 but in the OPPOSITE direction (re-inserting an axis instead "
        "of preserving one)."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["unbind-tuple-unpack", "ray-parametric-form"],
    "lo": (
        "Compose unbind-tuple-unpack (named O/D binding from a stacked ray tensor) with the parametric "
        "form P = O + tD, handling per-ray-t broadcasting by inserting a trailing 1-axis."
    ),
}


SPECS = [spec_13, spec_14, spec_15, spec_16, spec_17, spec_18]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
