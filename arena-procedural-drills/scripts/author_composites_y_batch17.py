"""Composite drills cx19..cx24 — batch-17 (Y-cell, part1).

Six composite procedural drills exercising 2-atom pairs from the ARENA part-1
ray-tracing barycentric / triangle machinery.

cx19  triangle-barycentric + linalg-solve-batched
cx20  triangle-barycentric + ray-parametric-form
cx21  triangle-barycentric + stack-vs-cat
cx22  triangle-barycentric + tensor-unbind
cx23  triangle-barycentric + unbind-tuple-unpack
cx24  triangle-barycentric + boolean-mask-combine
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
# cx19 — barycentric intersection test via batched solve
# ===========================================================================
spec_19 = {
    "atom_ids": ["triangle-barycentric", "linalg-solve-batched"],
    "subtopics": _subs(["triangle-barycentric", "linalg-solve-batched"]),
    "primary_atom": "triangle-barycentric",
    "part": "part1",
    "exercise_index": 19,
    "exercise_title": "barycentric ray-triangle intersection via batched linalg.solve",
    "slug": "barycentric-via-batched-solve",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's ray-triangle intersection rewrites the geometric question 'does this ray hit this "
        "triangle?' as a 3x3 LINEAR SYSTEM in the unknowns `(s, u, v)`:\n\n"
        "  `O + s D = A + u (B - A) + v (C - A)`\n\n"
        "Rearranged: `[-D | (B-A) | (C-A)] [s; u; v] = O - A`. To run it over a batch of (ray, triangle) "
        "pairs you stack the LHS into a `(N, 3, 3)` tensor and the RHS into `(N, 3)`, then call "
        "`torch.linalg.solve(M, y)` once — vectorized, no Python loop.\n\n"
        "The composition makes both atoms load-bearing: the barycentric formulation produces the system, "
        "and the batched solve is what makes it tractable over millions of pairs."
    ),
    "prompt_body": (
        "Implement `cx19_intersect_batched(rays, triangle)` that, for a batch of `N` rays and a single "
        "triangle, returns the `(N, 3)` tensor of `(s, u, v)` solutions to the barycentric system.\n\n"
        "Inputs:\n"
        "- `rays`: shape `(N, 2, 3)` — `rays[i, 0]` is origin `O_i`, `rays[i, 1]` is direction `D_i`.\n"
        "- `triangle`: shape `(3, 3)` — rows are vertices `A`, `B`, `C`.\n\n"
        "1. **Barycentric setup** — build the 3x3 system per ray:\n"
        "   - Columns of `M_i`: `[-D_i, B-A, C-A]`.\n"
        "   - RHS `y_i = O_i - A`.\n"
        "2. **Batched solve** — call `t.linalg.solve(M, y)` on the stacked `(N, 3, 3)` and `(N, 3)` "
        "tensors. Return the `(N, 3)` `(s, u, v)` matrix.\n\n"
        "The test cross-checks each row against a per-ray `t.linalg.solve` call (no batching) and "
        "verifies that plugging `(s, u, v)` back into the parametric equations recovers the same 3D "
        "point on both sides."
    ),
    "stub_body": (
        "def cx19_intersect_batched(rays, triangle):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: small N, hand-built rays + triangle.\n"
        "A = t.tensor([0.0, 0.0, 0.0])\n"
        "B = t.tensor([1.0, 0.0, 0.0])\n"
        "C = t.tensor([0.0, 1.0, 0.0])\n"
        "tri = t.stack([A, B, C], dim=0)  # (3, 3)\n"
        "# Two rays from z=1 shooting toward -z.\n"
        "O = t.tensor([[0.25, 0.25, 1.0], [2.0, 2.0, 1.0]])\n"
        "D = t.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])\n"
        "rays = t.stack([O, D], dim=1)  # (2, 2, 3)\n"
        "out = cx19_intersect_batched(rays, tri)\n"
        "assert tuple(out.shape) == (2, 3), f'expected (2,3), got {tuple(out.shape)}'\n"
        "# Ray 0 hits at (0.25, 0.25, 0) => s=1, u=0.25, v=0.25.\n"
        "assert t.allclose(out[0], t.tensor([1.0, 0.25, 0.25]), atol=1e-5), f'ray0 sol: {out[0]}'\n"
        "# Ray 1 also lands on the plane at s=1 but outside the triangle (u=2, v=2).\n"
        "assert t.allclose(out[1], t.tensor([1.0, 2.0, 2.0]), atol=1e-5), f'ray1 sol: {out[1]}'\n"
        "\n"
        "# Case B: cross-check against per-ray solve on a random batch.\n"
        "t.manual_seed(7)\n"
        "N = 32\n"
        "rays_b = t.randn(N, 2, 3)\n"
        "tri_b = t.randn(3, 3)\n"
        "out_b = cx19_intersect_batched(rays_b, tri_b)\n"
        "assert tuple(out_b.shape) == (N, 3)\n"
        "A_b = tri_b[0]\n"
        "for i in range(N):\n"
        "    O_i, D_i = rays_b[i, 0], rays_b[i, 1]\n"
        "    M_i = t.stack([-D_i, tri_b[1] - A_b, tri_b[2] - A_b], dim=1)\n"
        "    y_i = O_i - A_b\n"
        "    ref = t.linalg.solve(M_i, y_i)\n"
        "    assert t.allclose(out_b[i], ref, atol=1e-4), f'row {i} mismatch: {out_b[i]} vs {ref}'\n"
        "\n"
        "# Case C: plug (s, u, v) back into both sides — must match in 3D.\n"
        "s, u, v = out_b[0, 0], out_b[0, 1], out_b[0, 2]\n"
        "lhs = rays_b[0, 0] + s * rays_b[0, 1]\n"
        "rhs = tri_b[0] + u * (tri_b[1] - tri_b[0]) + v * (tri_b[2] - tri_b[0])\n"
        "assert t.allclose(lhs, rhs, atol=1e-4), f'parametric check failed: {lhs} vs {rhs}'"
    ),
    "solution_body": (
        "def cx19_intersect_batched(rays, triangle):\n"
        "    A, B, C = triangle[0], triangle[1], triangle[2]\n"
        "    O = rays[..., 0, :]  # (N, 3)\n"
        "    D = rays[..., 1, :]  # (N, 3)\n"
        "    N = O.shape[0]\n"
        "    # Atom A (triangle-barycentric): build the per-ray 3x3 system.\n"
        "    # Columns are [-D, B-A, C-A]; stack along last dim so M has shape (N, 3, 3).\n"
        "    col0 = -D                                       # (N, 3)\n"
        "    col1 = (B - A).unsqueeze(0).expand(N, -1)       # (N, 3)\n"
        "    col2 = (C - A).unsqueeze(0).expand(N, -1)       # (N, 3)\n"
        "    M = t.stack([col0, col1, col2], dim=-1)         # (N, 3, 3)\n"
        "    y = O - A                                       # (N, 3)\n"
        "    # Atom B (linalg-solve-batched): one vectorised solve over the leading batch dim.\n"
        "    return t.linalg.solve(M, y)                     # (N, 3)"
    ),
    "solution_notes": (
        "Batched `t.linalg.solve` is what makes ARENA's million-ray intersection tractable — the per-ray "
        "Python loop in the test is here only as ground truth. The barycentric setup is the load-bearing "
        "piece: each column of `M` matches one term of the parametric equation, in exactly the order the "
        "unknowns `(s, u, v)` are stacked."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["triangle-barycentric", "linalg-solve-batched"],
    "lo": (
        "Compose the barycentric 3x3 linear-system formulation with batched torch.linalg.solve to "
        "intersect a batch of rays against a triangle in a single vectorised call."
    ),
}


# ===========================================================================
# cx20 — derive (s, u, v) from ray-parametric form
# ===========================================================================
spec_20 = {
    "atom_ids": ["triangle-barycentric", "ray-parametric-form"],
    "subtopics": _subs(["triangle-barycentric", "ray-parametric-form"]),
    "primary_atom": "triangle-barycentric",
    "part": "part1",
    "exercise_index": 20,
    "exercise_title": "derive (s, u, v) from ray-parametric + triangle-plane equality",
    "slug": "suv-from-ray-parametric-form",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The ray-parametric form says any point on a ray is `P(s) = O + s D`. The barycentric form says "
        "any point on a triangle is `Q(u, v) = A + u (B - A) + v (C - A)`. Setting them EQUAL and "
        "rearranging is what gives you the 3x3 linear system in `(s, u, v)`.\n\n"
        "This drill exercises the DERIVATION explicitly: given `(s, u, v)`, you must reconstruct the 3D "
        "intersection point TWO ways (once via the ray, once via the triangle) and assert they agree. "
        "That's the contract the linear-system formulation is enforcing under the hood."
    ),
    "prompt_body": (
        "Implement `cx20_suv_round_trip(O, D, A, B, C, s, u, v)` that returns the 3D point computed BOTH "
        "ways from a known `(s, u, v)` solution:\n\n"
        "- `ray_point = O + s * D`  (ray-parametric form)\n"
        "- `tri_point = A + u * (B - A) + v * (C - A)`  (barycentric form)\n\n"
        "Return a tuple `(ray_point, tri_point)`. Each is a `(3,)` tensor.\n\n"
        "The test asserts `t.allclose(ray_point, tri_point)` for a valid `(s, u, v)` solution and that "
        "the two diverge when `(s, u, v)` is perturbed."
    ),
    "stub_body": (
        "def cx20_suv_round_trip(O, D, A, B, C, s, u, v):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built triangle in the z=0 plane, ray from z=1 down.\n"
        "A = t.tensor([0.0, 0.0, 0.0])\n"
        "B = t.tensor([1.0, 0.0, 0.0])\n"
        "C = t.tensor([0.0, 1.0, 0.0])\n"
        "O = t.tensor([0.3, 0.4, 1.0])\n"
        "D = t.tensor([0.0, 0.0, -1.0])\n"
        "# Known solution: ray reaches z=0 at s=1, hits (0.3, 0.4, 0) => u=0.3, v=0.4.\n"
        "rp, tp = cx20_suv_round_trip(O, D, A, B, C, 1.0, 0.3, 0.4)\n"
        "assert tuple(rp.shape) == (3,) and tuple(tp.shape) == (3,)\n"
        "assert t.allclose(rp, t.tensor([0.3, 0.4, 0.0]), atol=1e-6)\n"
        "assert t.allclose(tp, t.tensor([0.3, 0.4, 0.0]), atol=1e-6)\n"
        "assert t.allclose(rp, tp, atol=1e-6), 'ray and triangle reconstructions must agree'\n"
        "\n"
        "# Case B: random valid solution — solve the system, then round-trip.\n"
        "t.manual_seed(11)\n"
        "A2 = t.randn(3); B2 = t.randn(3); C2 = t.randn(3)\n"
        "O2 = t.randn(3); D2 = t.randn(3)\n"
        "M = t.stack([-D2, B2 - A2, C2 - A2], dim=1)\n"
        "y = O2 - A2\n"
        "sol = t.linalg.solve(M, y)\n"
        "s2, u2, v2 = sol[0].item(), sol[1].item(), sol[2].item()\n"
        "rp2, tp2 = cx20_suv_round_trip(O2, D2, A2, B2, C2, s2, u2, v2)\n"
        "assert t.allclose(rp2, tp2, atol=1e-4), f'rp={rp2} tp={tp2}'\n"
        "\n"
        "# Case C: a perturbed (s, u, v) must diverge.\n"
        "rp3, tp3 = cx20_suv_round_trip(O2, D2, A2, B2, C2, s2 + 0.5, u2, v2)\n"
        "assert not t.allclose(rp3, tp3, atol=1e-2), 'perturbed s should break the round-trip'"
    ),
    "solution_body": (
        "def cx20_suv_round_trip(O, D, A, B, C, s, u, v):\n"
        "    # Atom A (ray-parametric-form): P(s) = O + s D.\n"
        "    ray_point = O + s * D\n"
        "    # Atom B (triangle-barycentric): Q(u, v) = A + u (B - A) + v (C - A).\n"
        "    tri_point = A + u * (B - A) + v * (C - A)\n"
        "    return ray_point, tri_point"
    ),
    "solution_notes": (
        "This drill makes the geometric equality that DEFINES the 3x3 system load-bearing: the linear "
        "solver succeeds iff `O + s D = A + u (B - A) + v (C - A)` holds in 3D. Anywhere `(s, u, v)` "
        "fails to round-trip both sides, the intersection is fictitious."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["triangle-barycentric", "ray-parametric-form"],
    "lo": (
        "Reconstruct the 3D intersection point from a barycentric (s, u, v) solution two ways — via the "
        "ray's parametric form and via the triangle's barycentric form — and verify they agree."
    ),
}


# ===========================================================================
# cx21 — stack columns of (B-A, C-A, -D) for the 3x3 system
# ===========================================================================
spec_21 = {
    "atom_ids": ["triangle-barycentric", "stack-vs-cat"],
    "subtopics": _subs(["triangle-barycentric", "stack-vs-cat"]),
    "primary_atom": "triangle-barycentric",
    "part": "part1",
    "exercise_index": 21,
    "exercise_title": "build the 3x3 barycentric matrix via t.stack along the column axis",
    "slug": "stack-columns-for-barycentric-system",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The barycentric 3x3 system has the shape `[-D | (B-A) | (C-A)] [s; u; v] = O - A`. Each column "
        "is one `(3,)` vector, and the matrix is built by GLUING those three vectors along a NEW axis "
        "(the column axis).\n\n"
        "This is exactly the `torch.stack` vs `torch.cat` distinction:\n"
        "  - `t.stack([v1, v2, v3], dim=1)` → `(3, 3)` with each `v_i` as a COLUMN (new axis at dim 1).\n"
        "  - `t.cat([v1, v2, v3], dim=0)` → `(9,)` — concatenated along an EXISTING axis (wrong shape).\n\n"
        "Picking the wrong one is the most common bug in the ARENA implementation of this. The drill "
        "forces the correct choice."
    ),
    "prompt_body": (
        "Implement `cx21_barycentric_matrix(D, A, B, C)` that returns the 3x3 matrix whose columns are "
        "`[-D, B - A, C - A]`.\n\n"
        "- All inputs are `(3,)` tensors.\n"
        "- Output shape: `(3, 3)`. Column 0 is `-D`, column 1 is `B - A`, column 2 is `C - A`.\n\n"
        "Use `torch.stack` (NOT `torch.cat`) along `dim=1` so the three vectors become columns. The "
        "test asserts shape `(3, 3)` and that each column equals the expected vector — and also that "
        "`M @ t.tensor([s, u, v]) == O - A` for a known solution (i.e. the matrix is correctly assembled "
        "for the barycentric system)."
    ),
    "stub_body": (
        "def cx21_barycentric_matrix(D, A, B, C):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: each column lands in the right slot.\n"
        "D = t.tensor([0.0, 0.0, -1.0])\n"
        "A = t.tensor([0.0, 0.0, 0.0])\n"
        "B = t.tensor([1.0, 0.0, 0.0])\n"
        "C = t.tensor([0.0, 1.0, 0.0])\n"
        "M = cx21_barycentric_matrix(D, A, B, C)\n"
        "assert tuple(M.shape) == (3, 3), f'expected (3,3), got {tuple(M.shape)}'\n"
        "assert t.equal(M[:, 0], -D), f'col0: {M[:, 0]}'\n"
        "assert t.equal(M[:, 1], B - A), f'col1: {M[:, 1]}'\n"
        "assert t.equal(M[:, 2], C - A), f'col2: {M[:, 2]}'\n"
        "\n"
        "# Case B: assembling the right system — a known (s, u, v) round-trips.\n"
        "O = t.tensor([0.25, 0.25, 1.0])\n"
        "rhs = O - A\n"
        "sol = t.linalg.solve(M, rhs)\n"
        "assert t.allclose(sol, t.tensor([1.0, 0.25, 0.25]), atol=1e-5), f'sol: {sol}'\n"
        "\n"
        "# Case C: random — cross-check against a manually-constructed matrix.\n"
        "t.manual_seed(3)\n"
        "for _ in range(5):\n"
        "    D2, A2, B2, C2 = t.randn(3), t.randn(3), t.randn(3), t.randn(3)\n"
        "    M2 = cx21_barycentric_matrix(D2, A2, B2, C2)\n"
        "    assert tuple(M2.shape) == (3, 3)\n"
        "    ref = t.zeros(3, 3)\n"
        "    ref[:, 0] = -D2\n"
        "    ref[:, 1] = B2 - A2\n"
        "    ref[:, 2] = C2 - A2\n"
        "    assert t.allclose(M2, ref, atol=1e-6), f'mismatch:\\n{M2}\\nvs\\n{ref}'\n"
        "\n"
        "# Case D: catch the cat-instead-of-stack bug — wrong-shape output should fail tuple(...) test.\n"
        "assert M2.numel() == 9, 'matrix should have 9 elements (3x3) — did you use cat instead of stack?'"
    ),
    "solution_body": (
        "def cx21_barycentric_matrix(D, A, B, C):\n"
        "    # Atom A (triangle-barycentric): columns are [-D, B - A, C - A].\n"
        "    col0 = -D\n"
        "    col1 = B - A\n"
        "    col2 = C - A\n"
        "    # Atom B (stack-vs-cat): STACK along a NEW dim=1 to make each (3,) a column.\n"
        "    # Using cat would concat along an existing axis -> (9,), wrong shape.\n"
        "    return t.stack([col0, col1, col2], dim=1)"
    ),
    "solution_notes": (
        "`t.stack(..., dim=1)` creates a new axis at position 1, turning three `(3,)` vectors into a "
        "`(3, 3)` matrix with each vector as a column. `t.cat(..., dim=0)` would concatenate them along "
        "the existing axis 0 and give `(9,)` — a flat vector, not a matrix. The barycentric system needs "
        "the new-axis variant; this is the canonical stack-vs-cat call."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["triangle-barycentric", "stack-vs-cat"],
    "lo": (
        "Assemble the 3x3 barycentric LHS matrix by stacking the column vectors [-D, B-A, C-A] along a "
        "NEW axis — picking torch.stack over torch.cat by deriving the target shape."
    ),
}


# ===========================================================================
# cx22 — unbind solve result into (s, u, v)
# ===========================================================================
spec_22 = {
    "atom_ids": ["triangle-barycentric", "tensor-unbind"],
    "subtopics": _subs(["triangle-barycentric", "tensor-unbind"]),
    "primary_atom": "triangle-barycentric",
    "part": "part1",
    "exercise_index": 22,
    "exercise_title": "unbind a batched (N, 3) solve result into three (N,) tensors (s, u, v)",
    "slug": "unbind-solve-result-suv",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "After solving the barycentric 3x3 system in batch — `sol = t.linalg.solve(M, y)` with shape "
        "`(N, 3)` — you want the three SCALAR-PER-RAY component tensors `s`, `u`, `v` separately so "
        "you can build the inside-test mask: `(u >= 0) & (v >= 0) & (u + v <= 1)`.\n\n"
        "`tensor.unbind(dim=-1)` does this in one call: it returns a tuple of `dim`-removed tensors, "
        "each `(N,)`. Compared to `sol[:, 0], sol[:, 1], sol[:, 2]` it's a single named call and reads "
        "as 'split this axis into its constituent components' — which is the load-bearing semantic."
    ),
    "prompt_body": (
        "Implement `cx22_unbind_suv(sol)` that takes the `(N, 3)` solve output and returns three `(N,)` "
        "tensors `(s, u, v)`.\n\n"
        "Use `sol.unbind(dim=-1)` (or `t.unbind(sol, dim=-1)`) and return the resulting tuple.\n\n"
        "The test checks: tuple length 3, each component shape `(N,)`, exact-equality with "
        "`sol[:, 0]`, `sol[:, 1]`, `sol[:, 2]`, and that the components share storage with `sol` (unbind "
        "produces views, not copies)."
    ),
    "stub_body": (
        "def cx22_unbind_suv(sol):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built solve output.\n"
        "sol = t.tensor([\n"
        "    [1.0, 0.25, 0.25],\n"
        "    [2.0, 0.10, 0.80],\n"
        "    [0.5, 0.40, 0.30],\n"
        "])\n"
        "out = cx22_unbind_suv(sol)\n"
        "assert isinstance(out, tuple), f'expected tuple, got {type(out)}'\n"
        "assert len(out) == 3, f'expected 3 components, got {len(out)}'\n"
        "s, u, v = out\n"
        "assert tuple(s.shape) == (3,) and tuple(u.shape) == (3,) and tuple(v.shape) == (3,)\n"
        "assert t.equal(s, t.tensor([1.0, 2.0, 0.5]))\n"
        "assert t.equal(u, t.tensor([0.25, 0.10, 0.40]))\n"
        "assert t.equal(v, t.tensor([0.25, 0.80, 0.30]))\n"
        "\n"
        "# Case B: each component must match the slice along that column.\n"
        "t.manual_seed(5)\n"
        "sol2 = t.randn(64, 3)\n"
        "s2, u2, v2 = cx22_unbind_suv(sol2)\n"
        "assert t.equal(s2, sol2[:, 0])\n"
        "assert t.equal(u2, sol2[:, 1])\n"
        "assert t.equal(v2, sol2[:, 2])\n"
        "\n"
        "# Case C: storage aliasing — unbind returns views.\n"
        "assert s2.data_ptr() == sol2.data_ptr() or s2._base is not None, (\n"
        "    'unbind should return views sharing storage with sol'\n"
        ")"
    ),
    "solution_body": (
        "def cx22_unbind_suv(sol):\n"
        "    # Atom A (triangle-barycentric): the last axis of sol is (s, u, v).\n"
        "    # Atom B (tensor-unbind): split that axis into three (N,) views in one call.\n"
        "    return sol.unbind(dim=-1)"
    ),
    "solution_notes": (
        "`unbind(dim=-1)` is the named alternative to indexing each column manually — for a 3-component "
        "barycentric result it's three lines compressed to one, and the returned tensors are views "
        "(stride magic, not copies). This composes nicely with the inside-test in cx24 where each "
        "component flows into its own predicate."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 1,
    "kcs": ["triangle-barycentric", "tensor-unbind"],
    "lo": (
        "Use tensor.unbind to destructure a (N, 3) barycentric solve result into three (N,) "
        "components (s, u, v) in a single call."
    ),
}


# ===========================================================================
# cx23 — tuple-unpack the solve result
# ===========================================================================
spec_23 = {
    "atom_ids": ["triangle-barycentric", "unbind-tuple-unpack"],
    "subtopics": _subs(["triangle-barycentric", "unbind-tuple-unpack"]),
    "primary_atom": "triangle-barycentric",
    "part": "part1",
    "exercise_index": 23,
    "exercise_title": "tuple-unpack the barycentric solve into s, u, v in a single line",
    "slug": "tuple-unpack-solve-suv",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA uses the *unbind + tuple-unpack* idiom on the last axis of a stacked tensor — e.g. "
        "`ox, oy, oz, dx, dy, dz = rays.flatten(start_dim=-2).unbind(dim=-1)` to destructure rays into "
        "six scalar-per-ray streams in one line.\n\n"
        "Here we apply the same idiom to the barycentric solve result. Given `sol: (N, 3)`, the line "
        "`s, u, v = sol.unbind(dim=-1)` produces three named tensors in one Python statement — no "
        "indexing, no intermediate variable. The test verifies the unpack lands in the right order "
        "(`s` is the ray-parameter, `u` and `v` are the barycentric coords)."
    ),
    "prompt_body": (
        "Implement `cx23_unpack_suv(sol)` that uses the `unbind + tuple-unpack` idiom to destructure "
        "the `(N, 3)` solve result into three `(N,)` tensors in a single statement, then returns them "
        "as `(s, u, v)`.\n\n"
        "Required style: write the destructure as `s, u, v = sol.unbind(dim=-1)` (or the equivalent "
        "`t.unbind(sol, dim=-1)`). DO NOT use slice indexing — the drill is about the named idiom.\n\n"
        "The test checks the returned tuple matches `sol`'s columns AND that the order is `(s, u, v)` "
        "by plugging it into the round-trip equality from cx20."
    ),
    "stub_body": (
        "def cx23_unpack_suv(sol):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: column 0 is s, column 1 is u, column 2 is v.\n"
        "sol = t.tensor([\n"
        "    [1.0, 0.25, 0.25],\n"
        "    [0.5, 0.10, 0.40],\n"
        "])\n"
        "s, u, v = cx23_unpack_suv(sol)\n"
        "assert t.equal(s, sol[:, 0])\n"
        "assert t.equal(u, sol[:, 1])\n"
        "assert t.equal(v, sol[:, 2])\n"
        "assert tuple(s.shape) == (2,) and tuple(u.shape) == (2,) and tuple(v.shape) == (2,)\n"
        "\n"
        "# Case B: round-trip via cx20-style reconstruction.\n"
        "A = t.tensor([0.0, 0.0, 0.0])\n"
        "B = t.tensor([1.0, 0.0, 0.0])\n"
        "C = t.tensor([0.0, 1.0, 0.0])\n"
        "O = t.tensor([0.25, 0.25, 1.0])\n"
        "D = t.tensor([0.0, 0.0, -1.0])\n"
        "# Known solution s=1, u=0.25, v=0.25.\n"
        "sol2 = t.tensor([[1.0, 0.25, 0.25]])\n"
        "s2, u2, v2 = cx23_unpack_suv(sol2)\n"
        "ray_point = O + s2[0] * D\n"
        "tri_point = A + u2[0] * (B - A) + v2[0] * (C - A)\n"
        "assert t.allclose(ray_point, tri_point, atol=1e-6), (\n"
        "    f'unpack order wrong: ray={ray_point} vs tri={tri_point}'\n"
        ")\n"
        "\n"
        "# Case C: scale up.\n"
        "t.manual_seed(9)\n"
        "sol3 = t.randn(128, 3)\n"
        "s3, u3, v3 = cx23_unpack_suv(sol3)\n"
        "assert t.equal(s3, sol3[:, 0])\n"
        "assert t.equal(u3, sol3[:, 1])\n"
        "assert t.equal(v3, sol3[:, 2])"
    ),
    "solution_body": (
        "def cx23_unpack_suv(sol):\n"
        "    # Atom A (triangle-barycentric): the last axis of sol holds (s, u, v) in that order.\n"
        "    # Atom B (unbind-tuple-unpack): single-line destructure of the last axis.\n"
        "    s, u, v = sol.unbind(dim=-1)\n"
        "    return s, u, v"
    ),
    "solution_notes": (
        "Same mechanic as cx22, but the LOAD-BEARING part is the tuple-unpack on the LHS. In ARENA "
        "code the longer form `ox, oy, oz, dx, dy, dz = rays.flatten(start_dim=-2).unbind(dim=-1)` "
        "destructures a `(N, 2, 3)` ray tensor into six `(N,)` streams in one line — same idiom, "
        "wider unpack."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 1,
    "kcs": ["triangle-barycentric", "unbind-tuple-unpack"],
    "lo": (
        "Apply the unbind + tuple-unpack idiom to destructure a (N, 3) barycentric solve result into "
        "three named (N,) tensors (s, u, v) in a single statement."
    ),
}


# ===========================================================================
# cx24 — combine (u>=0) & (v>=0) & (u+v<=1) intersection mask
# ===========================================================================
spec_24 = {
    "atom_ids": ["triangle-barycentric", "boolean-mask-combine"],
    "subtopics": _subs(["triangle-barycentric", "boolean-mask-combine"]),
    "primary_atom": "triangle-barycentric",
    "part": "part1",
    "exercise_index": 24,
    "exercise_title": "combine (u>=0) & (v>=0) & (u+v<=1) into the inside-triangle mask",
    "slug": "boolean-mask-combine-inside-triangle",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Once you have `(u, v)` from the barycentric solve, the inside-the-triangle test is THREE "
        "predicates AND-ed together: `u >= 0`, `v >= 0`, `u + v <= 1`. Each predicate is a `(N,)` boolean "
        "tensor; the combined mask is their elementwise AND.\n\n"
        "This is the canonical `boolean-mask-combine` pattern: build the predicates separately (so each "
        "is named and debuggable), then `&` them into a single mask. NEVER `|` here — every predicate "
        "must hold simultaneously. The pattern composes cleanly with `triangle-barycentric` because the "
        "three predicates are exactly the geometric definition of 'point lies inside the triangle in "
        "barycentric coordinates'."
    ),
    "prompt_body": (
        "Implement `cx24_inside_triangle(u, v)` that takes two `(N,)` tensors of barycentric coordinates "
        "and returns a `(N,)` boolean mask: `True` where the corresponding `(u, v)` falls inside the "
        "triangle.\n\n"
        "Inside-the-triangle: ALL THREE of `u >= 0`, `v >= 0`, `u + v <= 1` must hold.\n\n"
        "Build the predicates separately (so each is named) and combine with `&` (NOT `|`). Return the "
        "combined boolean mask.\n\n"
        "The test covers:\n"
        "- hand-built `(u, v)` pairs with known inside/outside status,\n"
        "- the corners of the unit triangle (boundary => True for `>=` / `<=`),\n"
        "- a random batch cross-checked against `(u >= 0) & (v >= 0) & (u + v <= 1)`."
    ),
    "stub_body": (
        "def cx24_inside_triangle(u, v):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built points.\n"
        "u = t.tensor([0.25,  0.0,  1.0, -0.1, 0.6, 0.5])\n"
        "v = t.tensor([0.25,  1.0,  0.0,  0.5, 0.6, 0.4])\n"
        "# Inside? (0.25, 0.25) yes; (0, 1) yes (corner); (1, 0) yes (corner);\n"
        "# (-0.1, 0.5) no (u<0); (0.6, 0.6) no (u+v=1.2>1); (0.5, 0.4) yes.\n"
        "expected = t.tensor([True, True, True, False, False, True])\n"
        "mask = cx24_inside_triangle(u, v)\n"
        "assert mask.dtype == t.bool, f'expected bool, got {mask.dtype}'\n"
        "assert tuple(mask.shape) == tuple(u.shape)\n"
        "assert t.equal(mask, expected), f'got {mask}, expected {expected}'\n"
        "\n"
        "# Case B: random batch — cross-check against the canonical AND.\n"
        "t.manual_seed(13)\n"
        "u2 = t.randn(1024) * 0.6\n"
        "v2 = t.randn(1024) * 0.6\n"
        "mask2 = cx24_inside_triangle(u2, v2)\n"
        "ref = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)\n"
        "assert t.equal(mask2, ref), 'mask must match (u>=0) & (v>=0) & (u+v<=1)'\n"
        "\n"
        "# Case C: catch the OR-instead-of-AND bug.\n"
        "# All-True if you used | because each predicate is True for a different subset.\n"
        "u3 = t.tensor([-1.0, 0.5,  2.0])\n"
        "v3 = t.tensor([ 0.5, -1.0, 2.0])  # none of these are inside the triangle\n"
        "mask3 = cx24_inside_triangle(u3, v3)\n"
        "assert not mask3.any(), f'all three points are outside; mask: {mask3}'"
    ),
    "solution_body": (
        "def cx24_inside_triangle(u, v):\n"
        "    # Atom A (triangle-barycentric): a point is inside iff u >= 0, v >= 0, u + v <= 1.\n"
        "    p_u = u >= 0\n"
        "    p_v = v >= 0\n"
        "    p_sum = (u + v) <= 1\n"
        "    # Atom B (boolean-mask-combine): elementwise AND across the three predicates.\n"
        "    return p_u & p_v & p_sum"
    ),
    "solution_notes": (
        "Building the three predicates as named tensors first (rather than one big chained expression) "
        "makes each easy to inspect at debug time — you can `mask.any()` / `mask.sum()` per predicate "
        "to see which constraint is firing. The combining operator is `&` (elementwise AND): `|` would "
        "give the OR of the three regions, which is the WHOLE plane minus the third-quadrant + the "
        "u+v>1 wedge — emphatically not the triangle interior."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["triangle-barycentric", "boolean-mask-combine"],
    "lo": (
        "Combine the three barycentric predicates (u >= 0, v >= 0, u + v <= 1) into a single boolean "
        "mask via elementwise AND — the inside-the-triangle test."
    ),
}


SPECS = [spec_19, spec_20, spec_21, spec_22, spec_23, spec_24]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
