"""Composite drills cx1..cx6 — batch-17 (V-cell, part1 ray tracing).

Six composite procedural drills exercising 2-atom pairs from the ray-tracing
machinery (ARENA part 1 — ray-triangle intersection prereqs).

cx1  ray-parametric-form + stack-vs-cat
cx2  linalg-solve-batched + stack-vs-cat
cx3  linalg-solve-batched + ray-parametric-form
cx4  linalg-solve-batched + singular-matrix-mask-trick
cx5  ray-parametric-form + singular-matrix-mask-trick
cx6  einops-repeat + ray-parametric-form
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
# cx1 — build (NR, 3, 3) linear system by stacking columns
# ===========================================================================
spec_1 = {
    "atom_ids": ["ray-parametric-form", "stack-vs-cat"],
    "subtopics": _subs(["ray-parametric-form", "stack-vs-cat"]),
    "primary_atom": "ray-parametric-form",
    "part": "part1",
    "exercise_index": 1,
    "exercise_title": "build (NR, 3, 3) ray/triangle linear system by stacking columns",
    "slug": "build-ray-triangle-linsys-via-stack",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Ray-triangle intersection reduces to the 3x3 linear system\n\n"
        "  `[-D | B-A | C-A] @ [u, v, w] = O - A`\n\n"
        "where `O, D` are the ray origin/direction and `A, B, C` are the triangle vertices. The "
        "left-hand-side matrix is built from THREE COLUMN VECTORS — each of shape `(3,)` per ray — that "
        "need to be assembled into a `(3, 3)` matrix. Across `NR` rays we want a `(NR, 3, 3)` batch.\n\n"
        "This is the `stack-vs-cat` question in its purest form. We have three `(NR, 3)` column tensors "
        "and need a `(NR, 3, 3)` output. Same rank as the inputs means... NO — we need to INSERT a new "
        "axis of size 3 (one per column), so `t.stack(..., dim=-1)` is the right call. `cat(dim=-1)` "
        "would give `(NR, 9)`, wrong rank, silent shape bug.\n\n"
        "The `ray-parametric-form` atom is exercised in the column construction itself: `-D` IS the "
        "direction column, scaled by `-1` — i.e. the parametric form `R(u) = O + u*D` rearranged so `D` "
        "appears on the LHS as a coefficient of the `u` unknown."
    ),
    "prompt_body": (
        "Implement `cx1_build_linsys(rays, triangle)` that builds the `(NR, 3, 3)` linear-system matrix "
        "for ray-triangle intersection.\n\n"
        "- `rays` has shape `(NR, 2, 3)` — `rays[r, 0]` is origin `O_r`, `rays[r, 1]` is direction `D_r`.\n"
        "- `triangle` has shape `(3, 3)` — rows are vertices `A, B, C`.\n\n"
        "1. **Construct three column vectors**, each of shape `(NR, 3)`:\n"
        "   - `col0 = -D` (negated ray direction; this is the `ray-parametric-form` half — the LHS "
        "coefficient of the ray-parameter unknown).\n"
        "   - `col1 = B - A` (constant across rays — broadcast it to `(NR, 3)`).\n"
        "   - `col2 = C - A` (same).\n"
        "2. **Stack** the three columns into a `(NR, 3, 3)` matrix. Use `t.stack(..., dim=-1)` — that "
        "INSERTS a new trailing axis of size 3 (one per column). Do NOT use `cat` (which would extend an "
        "existing axis and give the wrong rank).\n\n"
        "Return the `(NR, 3, 3)` LHS matrix. The downstream solve `mat @ [u,v,w] = O - A` uses this exact "
        "shape contract."
    ),
    "stub_body": (
        "def cx1_build_linsys(rays, triangle):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: NR=2, hand-picked triangle and rays.\n"
        "rays = t.tensor([\n"
        "    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # origin=0, +x direction\n"
        "    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],   # origin=(0,1,0), +z\n"
        "])\n"
        "triangle = t.tensor([\n"
        "    [1.0, 0.0, 0.0],  # A\n"
        "    [1.0, 1.0, 0.0],  # B\n"
        "    [1.0, 0.0, 1.0],  # C\n"
        "])\n"
        "out = cx1_build_linsys(rays, triangle)\n"
        "assert tuple(out.shape) == (2, 3, 3), f'expected (2,3,3), got {tuple(out.shape)}'\n"
        "\n"
        "# Column 0 = -D for each ray.\n"
        "assert t.allclose(out[0, :, 0], t.tensor([-1.0, 0.0, 0.0]))\n"
        "assert t.allclose(out[1, :, 0], t.tensor([0.0, 0.0, -1.0]))\n"
        "# Column 1 = B - A = (0,1,0); same for every ray.\n"
        "assert t.allclose(out[0, :, 1], t.tensor([0.0, 1.0, 0.0]))\n"
        "assert t.allclose(out[1, :, 1], t.tensor([0.0, 1.0, 0.0]))\n"
        "# Column 2 = C - A = (0,0,1); same for every ray.\n"
        "assert t.allclose(out[0, :, 2], t.tensor([0.0, 0.0, 1.0]))\n"
        "assert t.allclose(out[1, :, 2], t.tensor([0.0, 0.0, 1.0]))\n"
        "\n"
        "# Case B: random scale — verify the column structure holds.\n"
        "NR = 16\n"
        "rng = t.Generator().manual_seed(0)\n"
        "rays_r = t.randn(NR, 2, 3, generator=rng)\n"
        "tri_r = t.randn(3, 3, generator=rng)\n"
        "mat = cx1_build_linsys(rays_r, tri_r)\n"
        "assert tuple(mat.shape) == (NR, 3, 3)\n"
        "# Column 0 must equal -D for every ray.\n"
        "assert t.allclose(mat[:, :, 0], -rays_r[:, 1])\n"
        "# Columns 1 and 2 must be the constant (B-A) and (C-A) broadcast across NR.\n"
        "for r in range(NR):\n"
        "    assert t.allclose(mat[r, :, 1], tri_r[1] - tri_r[0])\n"
        "    assert t.allclose(mat[r, :, 2], tri_r[2] - tri_r[0])\n"
        "\n"
        "# Case C: shape sanity — stacking, not catting.\n"
        "# If you used cat(dim=-1) you'd get (NR, 9). Catch that bug.\n"
        "assert mat.ndim == 3, 'expected rank 3 (NR,3,3) — did you use cat instead of stack?'\n"
        "assert mat.shape[-1] == 3, 'last axis should be size 3 (3 columns), not 9'"
    ),
    "solution_body": (
        "def cx1_build_linsys(rays, triangle):\n"
        "    NR = rays.shape[0]\n"
        "    # ray-parametric-form: the LHS coefficient of u in R(u)=O+u*D is -D after moving to LHS.\n"
        "    D = rays[:, 1]                  # (NR, 3)\n"
        "    A, B, C = triangle[0], triangle[1], triangle[2]\n"
        "    col0 = -D                       # (NR, 3)\n"
        "    # Broadcast the constant column vectors across NR rays.\n"
        "    col1 = (B - A).expand(NR, 3)    # (NR, 3)\n"
        "    col2 = (C - A).expand(NR, 3)    # (NR, 3)\n"
        "    # stack-vs-cat: we need a NEW axis (3 columns) -> stack(dim=-1), not cat.\n"
        "    return t.stack([col0, col1, col2], dim=-1)  # (NR, 3, 3)"
    ),
    "solution_notes": (
        "The `stack(dim=-1)` is doing the load-bearing work: each of `col0`, `col1`, `col2` has shape "
        "`(NR, 3)`; we want them to become columns 0, 1, 2 of an `(NR, 3, 3)` matrix. Inserting a NEW "
        "trailing axis is the textbook `stack` case. `cat(dim=-1)` would concatenate along the existing "
        "trailing axis and give `(NR, 9)` — wrong rank, fails the downstream solve.\n\n"
        "Note that `-D` is the `ray-parametric-form` atom rearranged: starting from `R(u) = O + u*D` and "
        "asking `R(u) ∈ triangle`, you end up with `-u*D + v*(B-A) + w*(C-A) = O - A`, so the matrix has "
        "`-D` as its first column."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ray-parametric-form", "stack-vs-cat"],
    "lo": (
        "Compose the parametric-ray LHS construction with the stack-vs-cat dispatch to assemble the "
        "(NR, 3, 3) ray-triangle linear system from three (NR, 3) column vectors."
    ),
}


# ===========================================================================
# cx2 — batched solve over stacked matrices
# ===========================================================================
spec_2 = {
    "atom_ids": ["linalg-solve-batched", "stack-vs-cat"],
    "subtopics": _subs(["linalg-solve-batched", "stack-vs-cat"]),
    "primary_atom": "linalg-solve-batched",
    "part": "part1",
    "exercise_index": 2,
    "exercise_title": "stack columns into batched matrix, then linalg.solve in one shot",
    "slug": "stack-then-batched-solve",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The batched-solve atom expects shape `(K, n, n)` for `A` and `(K, n)` for `b`. But upstream you "
        "often hold the column vectors of each `A_k` SEPARATELY — three `(K, n)` tensors that need to be "
        "fused into one `(K, n, n)` tensor before the solve.\n\n"
        "`stack-vs-cat` answers the dispatch question: same rank as inputs would need `cat`, but here we "
        "need an EXTRA axis (one slot per column), so it's `t.stack(cols, dim=-1)`. The trailing-axis "
        "insertion lines up exactly with `linalg.solve`'s `(K, n, n)` shape contract — last axis indexes "
        "the columns, second-to-last indexes the row entries, first axis is the batch.\n\n"
        "This composition is the meat of the ARENA Day 1 ray-triangle pipeline: build columns from "
        "geometry, stack to a batched matrix, batched solve, done — no Python loops."
    ),
    "prompt_body": (
        "Implement `cx2_solve_from_columns(cols, b)` that solves `K` independent `n x n` linear systems "
        "given the columns of each `A_k` separately.\n\n"
        "- `cols` is a list of `n` tensors, each of shape `(K, n)` — `cols[j]` is column `j` of each "
        "system's matrix.\n"
        "- `b` has shape `(K, n)` — the right-hand sides.\n\n"
        "1. **Stack** the columns into `A` of shape `(K, n, n)` with `t.stack(cols, dim=-1)`. (Last-axis "
        "insertion: each column becomes a slot along the new trailing axis.) Do NOT use `cat` — same-rank "
        "concatenation would produce `(K, n*n)`, not `(K, n, n)`.\n"
        "2. **Batched solve** with `t.linalg.solve(A, b)` — one call, no loop.\n\n"
        "Return the `(K, n)` solution tensor."
    ),
    "stub_body": (
        "def cx2_solve_from_columns(cols, b):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: K=2 hand-picked 2x2 systems.\n"
        "# System 0: A = [[1,0],[0,1]], b = [3,-1] -> x = [3,-1]\n"
        "# System 1: A = [[2,1],[1,3]], b = [5, 5] -> x = [2, 1]\n"
        "col0 = t.tensor([[1.0, 0.0], [2.0, 1.0]])  # K=2, n=2: col 0 of each system\n"
        "col1 = t.tensor([[0.0, 1.0], [1.0, 3.0]])  # col 1 of each system\n"
        "b = t.tensor([[3.0, -1.0], [5.0, 5.0]])\n"
        "x = cx2_solve_from_columns([col0, col1], b)\n"
        "assert tuple(x.shape) == (2, 2), f'expected (2,2), got {tuple(x.shape)}'\n"
        "assert t.allclose(x[0], t.tensor([3.0, -1.0]), atol=1e-5)\n"
        "assert t.allclose(x[1], t.tensor([2.0, 1.0]), atol=1e-5)\n"
        "\n"
        "# Case B: K=8 random 3x3 systems — cross-check by recomputing A @ x == b.\n"
        "rng = t.Generator().manual_seed(1)\n"
        "K, n = 8, 3\n"
        "# Build well-conditioned A as I + small perturbation.\n"
        "I3 = t.eye(3).unsqueeze(0).expand(K, n, n).clone()\n"
        "A_full = I3 + 0.1 * t.randn(K, n, n, generator=rng)\n"
        "b_r = t.randn(K, n, generator=rng)\n"
        "cols_r = [A_full[:, :, j] for j in range(n)]\n"
        "x_r = cx2_solve_from_columns(cols_r, b_r)\n"
        "assert tuple(x_r.shape) == (K, n)\n"
        "recon = t.einsum('kij,kj->ki', A_full, x_r)\n"
        "assert t.allclose(recon, b_r, atol=1e-4), f'A@x != b, max diff {(recon-b_r).abs().max()}'\n"
        "\n"
        "# Case C: ray-triangle-style 3-column build, then solve.\n"
        "# Verify the composition matches a manual stack + solve.\n"
        "K2, n2 = 4, 3\n"
        "rng2 = t.Generator().manual_seed(2)\n"
        "c0 = t.randn(K2, n2, generator=rng2)\n"
        "c1 = t.randn(K2, n2, generator=rng2)\n"
        "c2 = t.randn(K2, n2, generator=rng2)\n"
        "# Add a diagonal boost to keep matrices well-conditioned.\n"
        "boost = t.stack([c0, c1, c2], dim=-1) + 2.0 * t.eye(n2)\n"
        "c0b = boost[:, :, 0]; c1b = boost[:, :, 1]; c2b = boost[:, :, 2]\n"
        "b2 = t.randn(K2, n2, generator=rng2)\n"
        "x2 = cx2_solve_from_columns([c0b, c1b, c2b], b2)\n"
        "ref = t.linalg.solve(t.stack([c0b, c1b, c2b], dim=-1), b2)\n"
        "assert t.allclose(x2, ref, atol=1e-5)"
    ),
    "solution_body": (
        "def cx2_solve_from_columns(cols, b):\n"
        "    # stack-vs-cat: NEW trailing axis (one slot per column) -> stack(dim=-1).\n"
        "    A = t.stack(cols, dim=-1)        # (K, n, n)\n"
        "    # linalg-solve-batched: single C-level batched call, no Python loop.\n"
        "    return t.linalg.solve(A, b)      # (K, n)"
    ),
    "solution_notes": (
        "Two atoms, two lines. The `stack(dim=-1)` choice is what makes this a `stack-vs-cat` exercise — "
        "`cat(dim=-1)` would silently produce shape `(K, n*n)` which would then either crash `linalg.solve` "
        "(rank mismatch) or, worse, succeed-with-broadcasting in a way that gives garbage answers.\n\n"
        "Once the shapes are right, `t.linalg.solve(A, b)` fuses K LU factorizations into one BLAS call. "
        "The leading batch dim is just a convention — solve treats it as 'do this n x n problem K times', "
        "no loop required."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["linalg-solve-batched", "stack-vs-cat"],
    "lo": (
        "Compose stack-as-new-axis with batched linalg.solve: assemble per-system column tensors into "
        "(K, n, n) via stack(dim=-1), then dispatch a single batched solve over the batch."
    ),
}


# ===========================================================================
# cx3 — solve t in P = O + tD for ray-plane intersection
# ===========================================================================
spec_3 = {
    "atom_ids": ["linalg-solve-batched", "ray-parametric-form"],
    "subtopics": _subs(["linalg-solve-batched", "ray-parametric-form"]),
    "primary_atom": "linalg-solve-batched",
    "part": "part1",
    "exercise_index": 3,
    "exercise_title": "solve t in P=O+tD for ray-plane intersection, then evaluate P",
    "slug": "solve-ray-plane-intersection",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Ray-plane intersection takes both atoms in one turn. The plane is `n · (P - Q) = 0` (normal `n`, "
        "point on plane `Q`); the ray is `P(t) = O + t*D` (the `ray-parametric-form` atom). Substituting:\n\n"
        "  `n · (O + t*D - Q) = 0`  ⇒  `t = (n · (Q - O)) / (n · D)`\n\n"
        "That's a scalar 1x1 linear solve per ray. For a batch of `NR` rays, that's a `(NR, 1, 1)` "
        "coefficient matrix and a `(NR, 1)` RHS — the canonical `linalg-solve-batched` shape contract. "
        "After solving, we plug `t` back into the parametric form to recover the 3D intersection point "
        "`P = O + t*D` — that's the second use of `ray-parametric-form`.\n\n"
        "The whole pipeline is two atom-calls deep: solve for `t`, then evaluate `P(t)`. No loops."
    ),
    "prompt_body": (
        "Implement `cx3_ray_plane_intersect(rays, plane_normal, plane_point)` that computes the "
        "intersection points of `NR` rays with a single plane.\n\n"
        "- `rays` has shape `(NR, 2, 3)`: `rays[r, 0]` is origin `O_r`, `rays[r, 1]` is direction `D_r`.\n"
        "- `plane_normal` has shape `(3,)`: the plane normal `n`.\n"
        "- `plane_point` has shape `(3,)`: a point `Q` on the plane.\n\n"
        "Return `(t_vals, points)`:\n"
        "- `t_vals: (NR,)` — the ray parameter at intersection.\n"
        "- `points: (NR, 3)` — the 3D intersection points `O + t*D`.\n\n"
        "**Algorithm.**\n"
        "1. Build a `(NR, 1, 1)` coefficient matrix `A[r] = [[n · D_r]]` and a `(NR, 1)` RHS "
        "`b[r] = [n · (Q - O_r)]`.\n"
        "2. `t.linalg.solve(A, b)` → `(NR, 1)`. Squeeze to `(NR,)`.\n"
        "3. Evaluate `P = O + t * D` per ray (the `ray-parametric-form` atom) — broadcast `t: (NR, 1)` "
        "against `D: (NR, 3)`.\n\n"
        "Assume no rays are parallel to the plane (no near-zero `n · D` — the singular case is a "
        "separate drill)."
    ),
    "stub_body": (
        "def cx3_ray_plane_intersect(rays, plane_normal, plane_point):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: plane z=0 (normal=+z, point=origin), rays heading down.\n"
        "rays = t.tensor([\n"
        "    [[0.0, 0.0, 5.0], [0.0, 0.0, -1.0]],   # straight down from (0,0,5) -> hits z=0 at t=5\n"
        "    [[1.0, 1.0, 2.0], [0.0, 0.0, -2.0]],   # at (1,1,2), direction -2z -> t=1, P=(1,1,0)\n"
        "])\n"
        "n_plane = t.tensor([0.0, 0.0, 1.0])\n"
        "q_plane = t.tensor([0.0, 0.0, 0.0])\n"
        "t_vals, pts = cx3_ray_plane_intersect(rays, n_plane, q_plane)\n"
        "assert tuple(t_vals.shape) == (2,), f'expected (2,), got {tuple(t_vals.shape)}'\n"
        "assert tuple(pts.shape) == (2, 3), f'expected (2,3), got {tuple(pts.shape)}'\n"
        "assert t.allclose(t_vals, t.tensor([5.0, 1.0]), atol=1e-5), f't_vals: {t_vals}'\n"
        "expected_pts = t.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])\n"
        "assert t.allclose(pts, expected_pts, atol=1e-5), f'pts: {pts}'\n"
        "\n"
        "# Case B: tilted plane — cross-check by verifying every point lies on the plane.\n"
        "rng = t.Generator().manual_seed(3)\n"
        "NR = 32\n"
        "rays_r = t.randn(NR, 2, 3, generator=rng)\n"
        "# Make sure directions are not orthogonal to plane normal (no parallel rays).\n"
        "n_r = t.tensor([1.0, 1.0, 1.0])\n"
        "q_r = t.tensor([0.0, 0.0, 0.0])\n"
        "# Bias directions toward the normal so n.D is comfortably nonzero.\n"
        "rays_r[:, 1] = rays_r[:, 1] + n_r\n"
        "t_vals_r, pts_r = cx3_ray_plane_intersect(rays_r, n_r, q_r)\n"
        "assert tuple(t_vals_r.shape) == (NR,)\n"
        "assert tuple(pts_r.shape) == (NR, 3)\n"
        "# Every intersection point must satisfy n . (P - Q) ~= 0.\n"
        "residual = (pts_r - q_r) @ n_r\n"
        "assert t.allclose(residual, t.zeros(NR), atol=1e-4), f'residual: {residual.abs().max()}'\n"
        "# And P must equal O + t*D per ray.\n"
        "O = rays_r[:, 0]\n"
        "D = rays_r[:, 1]\n"
        "assert t.allclose(pts_r, O + t_vals_r.unsqueeze(-1) * D, atol=1e-4)"
    ),
    "solution_body": (
        "def cx3_ray_plane_intersect(rays, plane_normal, plane_point):\n"
        "    O = rays[:, 0]              # (NR, 3)\n"
        "    D = rays[:, 1]              # (NR, 3)\n"
        "    NR = O.shape[0]\n"
        "    # Build (NR, 1, 1) coefficient: n . D per ray.\n"
        "    nd = (D * plane_normal).sum(dim=-1)             # (NR,)\n"
        "    A = nd.reshape(NR, 1, 1)\n"
        "    # Build (NR, 1) RHS: n . (Q - O) per ray.\n"
        "    rhs = ((plane_point - O) * plane_normal).sum(dim=-1).reshape(NR, 1)\n"
        "    # linalg-solve-batched over (NR, 1, 1) systems.\n"
        "    t_solved = t.linalg.solve(A, rhs).squeeze(-1)   # (NR,)\n"
        "    # ray-parametric-form: P = O + t * D, broadcast t:(NR,1) against D:(NR,3).\n"
        "    points = O + t_solved.unsqueeze(-1) * D\n"
        "    return t_solved, points"
    ),
    "solution_notes": (
        "The 1x1 batched solve looks like overkill for a scalar division, and it IS — you could just "
        "write `t = rhs / nd`. But the point of this drill is the SHAPE DISCIPLINE: batched solve's "
        "`(K, n, n)` / `(K, n)` contract scales seamlessly to the ray-triangle 3x3 case (the next "
        "drill), so practicing the shape arithmetic on the trivial 1x1 case is good muscle memory.\n\n"
        "After solving, plugging `t` back into `O + t*D` exercises `ray-parametric-form` a second time. "
        "The `.unsqueeze(-1)` is the same broadcast trick from the standalone drill: `t: (NR,1)` vs "
        "`D: (NR,3)` lines up trailing dims so each ray's scalar parameter scales its own 3-vector "
        "direction."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["linalg-solve-batched", "ray-parametric-form"],
    "lo": (
        "Compose batched linalg.solve (1x1 systems) with the parametric ray equation to compute "
        "ray-plane intersections in pure batched tensor ops — no Python loops."
    ),
}


# ===========================================================================
# cx4 — patch singular dets with eye(3) before solve
# ===========================================================================
spec_4 = {
    "atom_ids": ["linalg-solve-batched", "singular-matrix-mask-trick"],
    "subtopics": _subs(["linalg-solve-batched", "singular-matrix-mask-trick"]),
    "primary_atom": "linalg-solve-batched",
    "part": "part1",
    "exercise_index": 4,
    "exercise_title": "patch singular slices with eye(n) before batched solve",
    "slug": "patch-singular-then-batched-solve",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`t.linalg.solve(A, b)` crashes the ENTIRE batched call if even one slice is singular — there's "
        "no per-slice error mode. The `singular-matrix-mask-trick` is the standard workaround: detect "
        "singular slices via `|det| < eps`, overwrite them with the identity (so the solve succeeds on "
        "garbage data), then mask the spurious outputs afterwards.\n\n"
        "The composition is THE protective shell around batched solve:\n\n"
        "1. `dets = t.linalg.det(A)` → `(K,)`\n"
        "2. `is_singular = dets.abs() < eps` → `(K,)` bool\n"
        "3. `A_safe = A.clone()`; `A_safe[is_singular] = t.eye(n)` (broadcasts I across the singular "
        "slices)\n"
        "4. `x = t.linalg.solve(A_safe, b)` — never crashes now\n"
        "5. Return `(x, ~is_singular)` so the caller knows which rows are real.\n\n"
        "This is the production pattern for any batched solve over data of unknown quality."
    ),
    "prompt_body": (
        "Implement `cx4_robust_batched_solve(A, b, eps=1e-8)` that runs a batched solve over `(K, n, n)` "
        "systems where SOME slices may be singular, without crashing.\n\n"
        "Inputs:\n"
        "- `A`: `(K, n, n)` — coefficient matrices (some may be singular).\n"
        "- `b`: `(K, n)` — right-hand sides.\n"
        "- `eps`: float — detection threshold on `|det|`.\n\n"
        "Returns `(x, is_valid)`:\n"
        "- `x: (K, n)` — solutions. Entries at singular slices are garbage (don't trust them).\n"
        "- `is_valid: (K,) bool` — `True` where the original `A[k]` was non-singular.\n\n"
        "**Steps:**\n"
        "1. `dets = t.linalg.det(A)`, `is_singular = dets.abs() < eps`.\n"
        "2. `A_safe = A.clone()` — do NOT mutate the caller's tensor. Then "
        "`A_safe[is_singular] = t.eye(n)` to overwrite singular slices.\n"
        "3. `x = t.linalg.solve(A_safe, b)` — single batched call, no loop.\n"
        "4. Return `x, ~is_singular`."
    ),
    "stub_body": (
        "def cx4_robust_batched_solve(A, b, eps=1e-8):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-picked 3x3 batch with one singular slice in the middle.\n"
        "A = t.stack([\n"
        "    t.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),   # det=8, x = b/2\n"
        "    t.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]),   # SINGULAR (row1 = 2*row0)\n"
        "    t.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),   # identity, x = b\n"
        "])\n"
        "b = t.tensor([\n"
        "    [4.0, 6.0, 8.0],\n"
        "    [1.0, 2.0, 3.0],\n"
        "    [-1.0, 0.5, 2.0],\n"
        "])\n"
        "A_before = A.clone()\n"
        "x, is_valid = cx4_robust_batched_solve(A, b)\n"
        "\n"
        "# Did NOT mutate input.\n"
        "assert t.equal(A, A_before), 'must not mutate the input A in place'\n"
        "# Shapes / dtypes.\n"
        "assert tuple(x.shape) == (3, 3), f'x shape: {tuple(x.shape)}'\n"
        "assert tuple(is_valid.shape) == (3,)\n"
        "assert is_valid.dtype == t.bool\n"
        "# Mask correctness.\n"
        "assert is_valid.tolist() == [True, False, True], f'is_valid: {is_valid.tolist()}'\n"
        "# Non-singular solutions are correct.\n"
        "assert t.allclose(x[0], t.tensor([2.0, 3.0, 4.0]), atol=1e-5), f'x[0]: {x[0]}'\n"
        "assert t.allclose(x[2], t.tensor([-1.0, 0.5, 2.0]), atol=1e-5), f'x[2]: {x[2]}'\n"
        "# x[1] is undefined, but must be finite (no NaN propagation).\n"
        "assert t.isfinite(x[1]).all(), f'x[1] must be finite (identity gives finite garbage): {x[1]}'\n"
        "\n"
        "# Case B: all singular — confirm no crash, all is_valid=False.\n"
        "A_all_sing = t.zeros(3, 2, 2)\n"
        "b_s = t.randn(3, 2)\n"
        "x_s, valid_s = cx4_robust_batched_solve(A_all_sing, b_s)\n"
        "assert tuple(x_s.shape) == (3, 2)\n"
        "assert not valid_s.any(), 'all should be invalid'\n"
        "assert t.isfinite(x_s).all(), 'no NaN/inf even when all singular'\n"
        "\n"
        "# Case C: all non-singular — cross-check against plain batched solve.\n"
        "rng = t.Generator().manual_seed(4)\n"
        "K, n = 16, 3\n"
        "A_good = t.eye(n).unsqueeze(0).expand(K, n, n) + 0.5 * t.randn(K, n, n, generator=rng)\n"
        "b_g = t.randn(K, n, generator=rng)\n"
        "x_g, valid_g = cx4_robust_batched_solve(A_good, b_g)\n"
        "assert valid_g.all(), 'all should be valid (well-conditioned)'\n"
        "assert t.allclose(x_g, t.linalg.solve(A_good, b_g), atol=1e-4)"
    ),
    "solution_body": (
        "def cx4_robust_batched_solve(A, b, eps=1e-8):\n"
        "    K, n, _ = A.shape\n"
        "    # singular-matrix-mask-trick: detect via determinant.\n"
        "    dets = t.linalg.det(A)\n"
        "    is_singular = dets.abs() < eps\n"
        "    # Clone so we don't mutate the caller's tensor.\n"
        "    A_safe = A.clone()\n"
        "    # Overwrite singular slices with the identity — broadcast (n,n) eye into the\n"
        "    # boolean-indexed (M, n, n) slice.\n"
        "    A_safe[is_singular] = t.eye(n)\n"
        "    # linalg-solve-batched: never crashes now, but garbage at singular slices.\n"
        "    x = t.linalg.solve(A_safe, b)\n"
        "    return x, ~is_singular"
    ),
    "solution_notes": (
        "The two atoms are tightly coupled in real ARENA code — you basically never use raw "
        "`linalg.solve` on data of unknown provenance. The mask trick is the standard armor.\n\n"
        "Why identity specifically: any non-singular matrix would let the solve succeed, but `eye(n)` "
        "produces solutions of `x = b` at those slices — i.e. FINITE garbage. If a downstream consumer "
        "ever forgets to mask via `is_valid`, they'll see weird answers but never `NaN` propagation, "
        "which makes the bug debuggable instead of silent.\n\n"
        "The det threshold (1e-8) is appropriate for float32; for float64 use 1e-12. A more principled "
        "approach uses condition number (`t.linalg.cond`) but `|det| < eps` is the ARENA convention "
        "and is much faster."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["linalg-solve-batched", "singular-matrix-mask-trick"],
    "lo": (
        "Compose the singular-matrix-mask trick with batched linalg.solve to run a robust batched solve "
        "that never crashes on singular slices, returning a validity mask for downstream consumers."
    ),
}


# ===========================================================================
# cx5 — guard against parallel rays (det=0) via identity-replacement
# ===========================================================================
spec_5 = {
    "atom_ids": ["ray-parametric-form", "singular-matrix-mask-trick"],
    "subtopics": _subs(["ray-parametric-form", "singular-matrix-mask-trick"]),
    "primary_atom": "ray-parametric-form",
    "part": "part1",
    "exercise_index": 5,
    "exercise_title": "guard against rays parallel to plane via identity replacement",
    "slug": "guard-parallel-rays-via-identity",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A ray `P(t) = O + t*D` (the parametric form) misses a plane entirely when `D` is parallel to "
        "the plane — `n · D == 0`. In a batched ray-plane intersection that's the SAME failure mode as "
        "a singular coefficient matrix: the 1x1 system `[n·D] * t = n·(Q-O)` has determinant zero.\n\n"
        "The `singular-matrix-mask-trick` works identically at the 1x1 scale: detect singular slices, "
        "replace them with the identity (here just `1.0`), let the solve succeed, then expose validity "
        "via a boolean mask so the caller never trusts the garbage `t` value for parallel rays.\n\n"
        "Composition flow:\n"
        "1. Compute `nd = n · D` per ray (the load-bearing scalar of `ray-parametric-form` at the plane).\n"
        "2. Build a `(NR, 1, 1)` matrix; detect `|nd| < eps`.\n"
        "3. Overwrite singular slots with `eye(1)`, solve, then mask out garbage solutions.\n"
        "4. Evaluate `P = O + t*D` again — but only the masked-valid rows are real intersections."
    ),
    "prompt_body": (
        "Implement `cx5_safe_ray_plane(rays, n_plane, q_plane, eps=1e-8)` — a ray-plane intersection "
        "that gracefully handles rays parallel to the plane.\n\n"
        "Inputs:\n"
        "- `rays`: `(NR, 2, 3)` — `rays[r, 0]` is origin, `rays[r, 1]` is direction.\n"
        "- `n_plane`: `(3,)` — plane normal.\n"
        "- `q_plane`: `(3,)` — point on plane.\n"
        "- `eps`: detection threshold on `|n · D|`.\n\n"
        "Returns `(t_vals, points, hit)`:\n"
        "- `t_vals: (NR,)` — solved parameters (garbage where `hit=False`).\n"
        "- `points: (NR, 3)` — `O + t*D` (garbage where `hit=False`, but always FINITE).\n"
        "- `hit: (NR,) bool` — `True` where the ray actually intersects (i.e. `|n·D| >= eps`).\n\n"
        "**Algorithm:**\n"
        "1. Compute `nd[r] = n_plane · D_r`, shape `(NR,)`.\n"
        "2. `is_parallel = nd.abs() < eps`.\n"
        "3. Build coefficient `A: (NR, 1, 1)` from `nd`; CLONE then overwrite singular slots with `1.0`.\n"
        "4. Build RHS `b: (NR, 1)` from `n_plane · (Q - O)`.\n"
        "5. `t_solved = solve(A_safe, b).squeeze(-1)` → `(NR,)`.\n"
        "6. `points = O + t_solved.unsqueeze(-1) * D` (the parametric-form atom again).\n"
        "7. Return `t_solved, points, ~is_parallel`."
    ),
    "stub_body": (
        "def cx5_safe_ray_plane(rays, n_plane, q_plane, eps=1e-8):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: mix of perpendicular (hits) and parallel (misses) rays vs plane z=0.\n"
        "rays = t.tensor([\n"
        "    [[0.0, 0.0, 5.0], [0.0, 0.0, -1.0]],   # hits z=0 at t=5\n"
        "    [[2.0, 3.0, 1.0], [1.0, 0.0, 0.0]],   # PARALLEL — D is along +x, n.D=0\n"
        "    [[1.0, 1.0, 2.0], [0.5, 0.5, -1.0]],   # hits z=0 at t=2\n"
        "    [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],   # PARALLEL, n.D=0\n"
        "])\n"
        "n_p = t.tensor([0.0, 0.0, 1.0])\n"
        "q_p = t.tensor([0.0, 0.0, 0.0])\n"
        "t_vals, pts, hit = cx5_safe_ray_plane(rays, n_p, q_p)\n"
        "\n"
        "assert tuple(t_vals.shape) == (4,), f't_vals shape: {tuple(t_vals.shape)}'\n"
        "assert tuple(pts.shape) == (4, 3)\n"
        "assert tuple(hit.shape) == (4,)\n"
        "assert hit.dtype == t.bool\n"
        "assert hit.tolist() == [True, False, True, False], f'hit: {hit.tolist()}'\n"
        "\n"
        "# Real hits have correct t and points.\n"
        "assert t.allclose(t_vals[0], t.tensor(5.0), atol=1e-5)\n"
        "assert t.allclose(t_vals[2], t.tensor(2.0), atol=1e-5)\n"
        "assert t.allclose(pts[0], t.tensor([0.0, 0.0, 0.0]), atol=1e-5)\n"
        "assert t.allclose(pts[2], t.tensor([2.0, 2.0, 0.0]), atol=1e-5)\n"
        "\n"
        "# Parallel rays — garbage but finite.\n"
        "assert t.isfinite(t_vals).all(), 'parallel rays must produce finite garbage (not NaN/inf)'\n"
        "assert t.isfinite(pts).all()\n"
        "\n"
        "# Case B: all parallel — no crash, all hit=False.\n"
        "rays_p = t.tensor([\n"
        "    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],\n"
        "    [[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]],\n"
        "])\n"
        "t_p, pts_p, hit_p = cx5_safe_ray_plane(rays_p, n_p, q_p)\n"
        "assert not hit_p.any()\n"
        "assert t.isfinite(t_p).all()\n"
        "assert t.isfinite(pts_p).all()\n"
        "\n"
        "# Case C: all non-parallel — verify intersections lie on plane.\n"
        "rng = t.Generator().manual_seed(5)\n"
        "NR = 24\n"
        "rays_r = t.randn(NR, 2, 3, generator=rng)\n"
        "n_r = t.tensor([0.0, 1.0, 0.0])\n"
        "# Boost D toward +y so n.D is nonzero.\n"
        "rays_r[:, 1, 1] = rays_r[:, 1, 1] + 5.0\n"
        "t_r, pts_r, hit_r = cx5_safe_ray_plane(rays_r, n_r, q_p)\n"
        "assert hit_r.all(), 'all should hit'\n"
        "residual = (pts_r - q_p) @ n_r\n"
        "assert t.allclose(residual, t.zeros(NR), atol=1e-3)"
    ),
    "solution_body": (
        "def cx5_safe_ray_plane(rays, n_plane, q_plane, eps=1e-8):\n"
        "    O = rays[:, 0]              # (NR, 3)\n"
        "    D = rays[:, 1]              # (NR, 3)\n"
        "    NR = O.shape[0]\n"
        "    # n . D per ray — zero iff D is parallel to the plane.\n"
        "    nd = (D * n_plane).sum(dim=-1)                  # (NR,)\n"
        "    is_parallel = nd.abs() < eps\n"
        "    # Build coefficient (NR, 1, 1) and apply singular-matrix-mask-trick.\n"
        "    A = nd.reshape(NR, 1, 1).clone()\n"
        "    A[is_parallel] = t.eye(1)                       # broadcast (1,1) into masked slots\n"
        "    # RHS.\n"
        "    rhs = ((q_plane - O) * n_plane).sum(dim=-1).reshape(NR, 1)\n"
        "    # linalg.solve — safe now.\n"
        "    t_solved = t.linalg.solve(A, rhs).squeeze(-1)   # (NR,)\n"
        "    # ray-parametric-form: P = O + t * D.\n"
        "    points = O + t_solved.unsqueeze(-1) * D\n"
        "    return t_solved, points, ~is_parallel"
    ),
    "solution_notes": (
        "Parallel rays are the geometric face of singular coefficient matrices in ray-plane "
        "intersection. The mask trick at the 1x1 scale just replaces `n·D ≈ 0` with `1.0` — the solve "
        "succeeds and produces the harmless 'solution' `t = n·(Q-O)`, which downstream consumers IGNORE "
        "because `hit=False`.\n\n"
        "Why this is the right armor: a naive `t.linalg.solve` on the original `(NR, 1, 1)` batch would "
        "either crash (PyTorch raises LinAlgError on the whole batch as soon as one slice is singular) "
        "or, if you switched to per-ray division `t = rhs / nd`, you'd get `inf` or `nan` for parallel "
        "rays and corrupt every downstream broadcast that touches them. The mask + identity-replace "
        "keeps EVERYTHING finite, and the validity flag is exposed to the caller.\n\n"
        "Note that `ray-parametric-form` appears twice in the solution: implicitly in `n · D` (where `D` "
        "is the parametric direction vector), and explicitly in `P = O + t*D` at the end."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ray-parametric-form", "singular-matrix-mask-trick"],
    "lo": (
        "Compose the parametric ray equation with the singular-matrix-mask trick to detect "
        "rays-parallel-to-plane via |n·D|<eps, identity-replace the singular slot, and return both the "
        "(possibly-garbage) intersection AND a per-ray validity flag."
    ),
}


# ===========================================================================
# cx6 — broadcast a triangle across NR rays via einops.repeat
# ===========================================================================
spec_6 = {
    "atom_ids": ["einops-repeat", "ray-parametric-form"],
    "subtopics": _subs(["einops-repeat", "ray-parametric-form"]),
    "primary_atom": "einops-repeat",
    "part": "part1",
    "exercise_index": 6,
    "exercise_title": "broadcast a triangle across NR rays via einops.repeat, then evaluate rays at u",
    "slug": "repeat-triangle-then-eval-rays",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "When you intersect `NR` rays against a SINGLE triangle, the triangle vertices `(A, B, C)` have "
        "shape `(3, 3)` — but to align them with `(NR, 3)` ray data they need to be repeated across the "
        "ray axis. `einops.repeat(triangle, 'v d -> nr v d', nr=NR)` is the explicit broadcast — a "
        "stride-0 view, no copy, with the named axis documenting WHICH dim is broadcasting.\n\n"
        "Combined with `ray-parametric-form` `R(u) = O + u*D`, this composition lets you compute, for a "
        "batch of rays and a batch of parameter values, the 3D points along each ray AND have the "
        "triangle data pre-aligned for the downstream intersection test — all in pure tensor ops, no "
        "Python loops over rays.\n\n"
        "The point: `einops.repeat` is the explicit-broadcast tool — it's what you reach for when "
        "implicit broadcasting would be ambiguous (which axis broadcasts where?). Here the triangle "
        "needs to be aligned against the ray axis, and the named pattern makes that explicit."
    ),
    "prompt_body": (
        "Implement `cx6_eval_rays_with_triangle(rays, us, triangle)` that:\n\n"
        "1. Evaluates each ray at its corresponding parameter (`ray-parametric-form`).\n"
        "2. Broadcasts the single triangle across all rays (`einops.repeat`).\n\n"
        "Inputs:\n"
        "- `rays`: `(NR, 2, 3)` — `rays[r, 0]` is origin, `rays[r, 1]` is direction.\n"
        "- `us`: `(NR,)` — one parameter value per ray.\n"
        "- `triangle`: `(3, 3)` — three vertices of a single triangle, rows = vertices.\n\n"
        "Returns `(points, tri_broadcast)`:\n"
        "- `points: (NR, 3)` — `O_r + us[r] * D_r` (the parametric form, one point per ray).\n"
        "- `tri_broadcast: (NR, 3, 3)` — the same triangle, broadcast across `NR` rays via "
        "`repeat(triangle, 'v d -> nr v d', nr=NR)`.\n\n"
        "The test asserts that `tri_broadcast` is a true zero-copy view (shares storage with `triangle`) "
        "and that `points` matches the manual parametric evaluation."
    ),
    "stub_body": (
        "def cx6_eval_rays_with_triangle(rays, us, triangle):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: NR=2, hand-picked rays and triangle.\n"
        "rays = t.tensor([\n"
        "    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # walks +x\n"
        "    [[0.0, 0.0, 5.0], [0.0, 0.0, -1.0]],  # walks -z from (0,0,5)\n"
        "])\n"
        "us = t.tensor([2.0, 3.0])\n"
        "triangle = t.tensor([\n"
        "    [1.0, 0.0, 0.0],\n"
        "    [1.0, 1.0, 0.0],\n"
        "    [1.0, 0.0, 1.0],\n"
        "])\n"
        "points, tri_b = cx6_eval_rays_with_triangle(rays, us, triangle)\n"
        "\n"
        "# points: parametric evaluation.\n"
        "assert tuple(points.shape) == (2, 3), f'points shape: {tuple(points.shape)}'\n"
        "expected_pts = t.tensor([\n"
        "    [2.0, 0.0, 0.0],   # (0,0,0) + 2*(1,0,0)\n"
        "    [0.0, 0.0, 2.0],   # (0,0,5) + 3*(0,0,-1) = (0,0,2)\n"
        "])\n"
        "assert t.allclose(points, expected_pts, atol=1e-5), f'points: {points}'\n"
        "\n"
        "# tri_b: same triangle broadcast across rays.\n"
        "assert tuple(tri_b.shape) == (2, 3, 3), f'tri_b shape: {tuple(tri_b.shape)}'\n"
        "assert t.equal(tri_b[0], triangle), 'tri_b[0] should equal triangle'\n"
        "assert t.equal(tri_b[1], triangle), 'tri_b[1] should equal triangle'\n"
        "\n"
        "# Storage-sharing check — repeat is a stride-0 view, not a copy.\n"
        "assert tri_b.data_ptr() == triangle.data_ptr(), (\n"
        "    'einops.repeat should produce a stride-0 view sharing storage with `triangle`. '\n"
        "    'Did you accidentally .clone() or .contiguous()?'\n"
        ")\n"
        "\n"
        "# Case B: random NR=32 — cross-check against manual broadcast.\n"
        "rng = t.Generator().manual_seed(6)\n"
        "NR = 32\n"
        "rays_r = t.randn(NR, 2, 3, generator=rng)\n"
        "us_r = t.randn(NR, generator=rng)\n"
        "tri_r = t.randn(3, 3, generator=rng)\n"
        "pts_r, trb_r = cx6_eval_rays_with_triangle(rays_r, us_r, tri_r)\n"
        "assert tuple(pts_r.shape) == (NR, 3)\n"
        "assert tuple(trb_r.shape) == (NR, 3, 3)\n"
        "# Manual ref for points: O + u * D per ray.\n"
        "O = rays_r[:, 0]\n"
        "D = rays_r[:, 1]\n"
        "expected = O + us_r.unsqueeze(-1) * D\n"
        "assert t.allclose(pts_r, expected, atol=1e-5)\n"
        "# Manual ref for triangle broadcast.\n"
        "for r in range(NR):\n"
        "    assert t.equal(trb_r[r], tri_r), f'tri_b[{r}] differs from triangle'\n"
        "# Storage check holds at scale.\n"
        "assert trb_r.data_ptr() == tri_r.data_ptr()"
    ),
    "solution_body": (
        "def cx6_eval_rays_with_triangle(rays, us, triangle):\n"
        "    NR = rays.shape[0]\n"
        "    O = rays[:, 0]                  # (NR, 3)\n"
        "    D = rays[:, 1]                  # (NR, 3)\n"
        "    # ray-parametric-form: R(u) = O + u * D, with us broadcast over the 3-axis.\n"
        "    points = O + us.unsqueeze(-1) * D\n"
        "    # einops-repeat: explicit broadcast of the single triangle across the NR axis.\n"
        "    # Stride-0 view — no copy, storage shared with `triangle`.\n"
        "    tri_broadcast = repeat(triangle, 'v d -> nr v d', nr=NR)\n"
        "    return points, tri_broadcast"
    ),
    "solution_notes": (
        "Two cheap operations, both load-bearing. `us.unsqueeze(-1) * D` is the textbook broadcast for "
        "the parametric form — it lines up `(NR, 1)` against `(NR, 3)` so each ray's scalar parameter "
        "scales its own 3-vector direction. `repeat(triangle, 'v d -> nr v d', nr=NR)` is the explicit "
        "version of `triangle.expand(NR, 3, 3)` — same stride-0 storage, but the named pattern makes "
        "intent obvious to the next reader.\n\n"
        "This composition is the per-ray data prep step right before a ray-triangle intersection: rays "
        "evaluated, triangle broadcast — then downstream code can `stack([-D, B-A, C-A], dim=-1)` to "
        "build the `(NR, 3, 3)` LHS matrix from cx1, and pass it to the batched solve from cx3."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["einops-repeat", "ray-parametric-form"],
    "lo": (
        "Compose einops.repeat (zero-copy axis insertion) with the parametric ray equation to align a "
        "single triangle against a batch of NR rays and evaluate each ray at its parameter — both in "
        "pure tensor ops with no copies."
    ),
}


SPECS = [spec_1, spec_2, spec_3, spec_4, spec_5, spec_6]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
