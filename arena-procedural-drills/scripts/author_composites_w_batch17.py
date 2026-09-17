"""Composite drills cx7..cx12 — batch-17 (W-cell, part1, ARENA ray tracing).

Six composite procedural drills exercising 2-atom pairs from the einops /
linear-algebra / stack-vs-cat machinery in the ARENA ray-tracing context
(part1 — ray-triangle intersection prereqs).

cx7   einops-repeat + einops-repeat-broadcast       outer-product (NR,NT) two-repeats
cx8   einops-repeat + stack-vs-cat                  repeat then stack a per-ray system
cx9   einops-repeat-broadcast + stack-vs-cat        repeat-broadcast then cat along axis
cx10  einops-repeat + linalg-solve-batched          broadcast triangle then batched solve
cx11  einops-repeat-broadcast + linalg-solve-batched repeat-broadcast LHS into solve
cx12  einops-repeat + singular-matrix-mask-trick    broadcast eye(3) fallback over NR
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
# cx7 — outer-product (NR, NT) broadcast via two repeats
# ===========================================================================
spec_7 = {
    "atom_ids": ["einops-repeat", "einops-repeat-broadcast"],
    "subtopics": _subs(["einops-repeat", "einops-repeat-broadcast"]),
    "primary_atom": "einops-repeat-broadcast",
    "part": "part1",
    "exercise_index": 7,
    "exercise_title": "every-ray every-triangle (NR, NT, 3) via two repeats",
    "slug": "outer-product-nr-nt-via-two-repeats",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA ray-tracing constantly builds `(NR, NT, ...)` tensors — one slot per (ray, triangle) "
        "pair. The free way is two `einops.repeat` calls, both compiling to stride-0 views:\n\n"
        "  `rays_b = repeat(rays, 'r p d -> r t p d', t=NT)`     (insert t-axis)\n"
        "  `tris_b = repeat(tris, 't v d -> r t v d', r=NR)`     (insert r-axis)\n\n"
        "Both atoms are flavours of repeat: the first is plain `einops-repeat` adding a single axis, "
        "the second is `einops-repeat-broadcast` — same syntax, but the intent is to pair every-with-every. "
        "Together they're the canonical ARENA pattern for ray-triangle intersection batching."
    ),
    "prompt_body": (
        "Implement `cx7_pair_rays_with_triangles(rays, triangles)` that builds the every-ray every-triangle "
        "broadcast pair.\n\n"
        "- `rays` has shape `(NR, 2, 3)` — origin and direction stacked along axis 1.\n"
        "- `triangles` has shape `(NT, 3, 3)` — three vertices A,B,C stacked along axis 1.\n\n"
        "1. **Repeat** rays into `(NR, NT, 2, 3)`: `repeat(rays, 'r p d -> r t p d', t=NT)`.\n"
        "2. **Repeat-broadcast** triangles into `(NR, NT, 3, 3)`: `repeat(tris, 't v d -> r t v d', r=NR)`.\n\n"
        "Return the tuple `(rays_b, tris_b)`. Both must be stride-0 views (zero-copy) — the test asserts "
        "`data_ptr` aliasing back to the original tensors."
    ),
    "stub_body": (
        "def cx7_pair_rays_with_triangles(rays, triangles):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "NR, NT = 5, 4\n"
        "rays = t.randn(NR, 2, 3)\n"
        "triangles = t.randn(NT, 3, 3)\n"
        "rays_b, tris_b = cx7_pair_rays_with_triangles(rays, triangles)\n"
        "assert tuple(rays_b.shape) == (NR, NT, 2, 3), f'rays_b: {tuple(rays_b.shape)}'\n"
        "assert tuple(tris_b.shape) == (NR, NT, 3, 3), f'tris_b: {tuple(tris_b.shape)}'\n"
        "# Zero-copy: storage aliases the originals.\n"
        "assert rays_b.data_ptr() == rays.data_ptr(), 'rays_b must be a stride-0 view of rays'\n"
        "assert tris_b.data_ptr() == triangles.data_ptr(), 'tris_b must be a stride-0 view of triangles'\n"
        "# Values: every (r, t) slot of rays_b equals rays[r]; every (r, t) of tris_b equals triangles[t].\n"
        "for r in range(NR):\n"
        "    for tri in range(NT):\n"
        "        assert t.equal(rays_b[r, tri], rays[r]), f'rays_b[{r},{tri}] != rays[{r}]'\n"
        "        assert t.equal(tris_b[r, tri], triangles[tri]), f'tris_b[{r},{tri}] != triangles[{tri}]'\n"
        "\n"
        "# Realistic ARENA scale.\n"
        "rays2 = t.randn(200, 2, 3)\n"
        "tris2 = t.randn(50, 3, 3)\n"
        "rb2, tb2 = cx7_pair_rays_with_triangles(rays2, tris2)\n"
        "assert tuple(rb2.shape) == (200, 50, 2, 3)\n"
        "assert tuple(tb2.shape) == (200, 50, 3, 3)\n"
        "assert rb2.data_ptr() == rays2.data_ptr()\n"
        "assert tb2.data_ptr() == tris2.data_ptr()"
    ),
    "solution_body": (
        "def cx7_pair_rays_with_triangles(rays, triangles):\n"
        "    NR = rays.shape[0]\n"
        "    NT = triangles.shape[0]\n"
        "    # Atom A (einops-repeat): insert the t-axis on rays — stride-0 view.\n"
        "    rays_b = repeat(rays, 'r p d -> r t p d', t=NT)\n"
        "    # Atom B (einops-repeat-broadcast): insert the r-axis on triangles — stride-0 view.\n"
        "    tris_b = repeat(triangles, 't v d -> r t v d', r=NR)\n"
        "    return rays_b, tris_b"
    ),
    "solution_notes": (
        "Both repeats compile to `expand` (stride-0 views) — memory stays O(NR + NT), not O(NR*NT). "
        "If you reach for `.repeat()` (the torch method, not einops), you allocate a full materialized "
        "(NR, NT, ...) tensor and lose the storage-aliasing property. The einops version is the only "
        "form that survives ARENA-scale (>>1M ray-tri pairs)."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["einops-repeat", "einops-repeat-broadcast"],
    "lo": (
        "Compose two einops repeats to build the every-ray every-triangle (NR, NT, ...) broadcast pair "
        "as stride-0 views, the canonical ARENA ray-tracing pairing pattern."
    ),
}


# ===========================================================================
# cx8 — repeat then stack a per-ray system (origins, directions)
# ===========================================================================
spec_8 = {
    "atom_ids": ["einops-repeat", "stack-vs-cat"],
    "subtopics": _subs(["einops-repeat", "stack-vs-cat"]),
    "primary_atom": "stack-vs-cat",
    "part": "part1",
    "exercise_index": 8,
    "exercise_title": "repeat a constant origin then stack with per-ray directions",
    "slug": "repeat-origin-then-stack-with-dirs",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Many ARENA helpers expect rays in `(NR, 2, 3)` packed form — `[origin, direction]` stacked "
        "along a new middle axis. But the input data often comes split: a single shared origin "
        "`(3,)` plus a `(NR, 3)` bundle of directions.\n\n"
        "The composition is:\n"
        "  1. `einops-repeat` broadcasts the `(3,)` origin to `(NR, 3)` — stride-0 view.\n"
        "  2. `stack-vs-cat` picks `torch.stack` (NOT cat!) because we need to INSERT a new axis of "
        "size 2, not extend an existing one.\n\n"
        "Pick wrong (cat instead of stack) and you'd get `(NR, 6)` — same total elements, wrong shape. "
        "The atom recap is: stack inserts an axis, cat extends one."
    ),
    "prompt_body": (
        "Implement `cx8_pack_rays(origin, directions)` — combine a single shared `origin: (3,)` with "
        "a per-ray `directions: (NR, 3)` into a packed rays tensor of shape `(NR, 2, 3)`.\n\n"
        "1. **Repeat** the origin across the ray axis: `repeat(origin, 'd -> r d', r=NR)`. Result "
        "shape `(NR, 3)`.\n"
        "2. **Stack vs cat**: the target shape is `(NR, 2, 3)` — you need a NEW axis of size 2 between "
        "`NR` and `3`. That's `torch.stack([..., ...], dim=1)`, not `torch.cat` (which would give `(NR, 6)`).\n\n"
        "Return shape `(NR, 2, 3)`. The test asserts `rays[:, 0] == origin_broadcast` and "
        "`rays[:, 1] == directions`."
    ),
    "stub_body": (
        "def cx8_pack_rays(origin, directions):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: cross-check against the manual reference.\n"
        "origin = t.tensor([1.0, 2.0, 3.0])\n"
        "directions = t.randn(5, 3)\n"
        "rays = cx8_pack_rays(origin, directions)\n"
        "assert tuple(rays.shape) == (5, 2, 3), f'shape: {tuple(rays.shape)}'\n"
        "# Slot 0 of every ray is the shared origin.\n"
        "for r in range(5):\n"
        "    assert t.equal(rays[r, 0], origin), f'ray {r} origin mismatch: {rays[r, 0]}'\n"
        "    assert t.equal(rays[r, 1], directions[r]), f'ray {r} direction mismatch: {rays[r, 1]}'\n"
        "\n"
        "# Case B: NEGATIVE check — cat would produce (NR, 6), not (NR, 2, 3).\n"
        "# Make sure the student did NOT just cat along dim=-1.\n"
        "assert rays.ndim == 3, f'expected 3-D (NR,2,3), got {rays.ndim}-D — did you use cat?'\n"
        "\n"
        "# Case C: realistic scale.\n"
        "origin2 = t.tensor([0.0, 0.0, 0.0])\n"
        "dirs2 = t.randn(200, 3)\n"
        "rays2 = cx8_pack_rays(origin2, dirs2)\n"
        "assert tuple(rays2.shape) == (200, 2, 3)\n"
        "assert t.allclose(rays2[:, 0], t.zeros(200, 3))\n"
        "assert t.equal(rays2[:, 1], dirs2)\n"
        "\n"
        "# Case D: hand-check on small tensor.\n"
        "o3 = t.tensor([7.0, 8.0, 9.0])\n"
        "d3 = t.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])\n"
        "out3 = cx8_pack_rays(o3, d3)\n"
        "expected = t.tensor([[[7.0, 8.0, 9.0], [1.0, 0.0, 0.0]],\n"
        "                     [[7.0, 8.0, 9.0], [0.0, 1.0, 0.0]]])\n"
        "assert t.equal(out3, expected), f'got {out3}'"
    ),
    "solution_body": (
        "def cx8_pack_rays(origin, directions):\n"
        "    NR = directions.shape[0]\n"
        "    # Atom A (einops-repeat): broadcast the shared origin to per-ray (NR, 3) — stride-0 view.\n"
        "    origins = repeat(origin, 'd -> r d', r=NR)\n"
        "    # Atom B (stack-vs-cat): we need a NEW axis of size 2 — that's stack (NOT cat).\n"
        "    #   stack inserts an axis, cat extends one.\n"
        "    return t.stack([origins, directions], dim=1)"
    ),
    "solution_notes": (
        "The stack-vs-cat picker rule: count what you have vs what you want.\n"
        "  - origins: (NR, 3); directions: (NR, 3). Two same-shape tensors.\n"
        "  - target: (NR, 2, 3) — a NEW axis of size 2.\n"
        "→ stack along dim=1.\n\n"
        "If the target had been (NR, 6) — same axes but extended — that's cat. The mnemonic: "
        "len(inputs) becomes the NEW axis size for stack, but is summed into an EXISTING axis for cat."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["einops-repeat", "stack-vs-cat"],
    "lo": (
        "Compose einops repeat (broadcast a scalar origin per-ray) with torch.stack (insert a new "
        "size-2 axis) to assemble the (NR, 2, 3) packed rays tensor that ARENA helpers expect."
    ),
}


# ===========================================================================
# cx9 — repeat-broadcast then cat along axis (assemble (NR, NT, 9) bundle)
# ===========================================================================
spec_9 = {
    "atom_ids": ["einops-repeat-broadcast", "stack-vs-cat"],
    "subtopics": _subs(["einops-repeat-broadcast", "stack-vs-cat"]),
    "primary_atom": "stack-vs-cat",
    "part": "part1",
    "exercise_index": 9,
    "exercise_title": "repeat-broadcast triangle vertices then cat along the d-axis",
    "slug": "repeat-broadcast-then-cat-vertices",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Some ray-triangle solvers want the three triangle vertices in a single concatenated 9-vector "
        "per (ray, triangle) slot — `(NR, NT, 9)` where the last axis is `[Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz]`.\n\n"
        "Build it two ways atoms compose:\n"
        "  1. `einops-repeat-broadcast` inserts the ray-axis into each vertex `(NT, 3) -> (NR, NT, 3)`.\n"
        "  2. `stack-vs-cat` picks `torch.cat(..., dim=-1)` — we're EXTENDING an existing 3-axis to 9, "
        "NOT inserting a new one. Picking stack here would give the wrong shape `(NR, NT, 3, 3)`.\n\n"
        "The rule from cx8 reverses here: same total axes in the target → cat, not stack."
    ),
    "prompt_body": (
        "Implement `cx9_broadcast_and_concat_vertices(triangles, NR)` that takes triangle vertices "
        "`triangles: (NT, 3, 3)` (axis 1 = vertex A/B/C, axis 2 = xyz) and the number of rays `NR`, "
        "and returns a `(NR, NT, 9)` tensor where each slot is the concatenated 9-vector `[A, B, C]`.\n\n"
        "1. **Split** triangles into A, B, C — three `(NT, 3)` tensors.\n"
        "2. **Repeat-broadcast** each into `(NR, NT, 3)` with `repeat(..., 't d -> r t d', r=NR)`. "
        "These are stride-0 views.\n"
        "3. **Stack vs cat**: target last axis size 9 = 3 + 3 + 3 — that's EXTENDING the existing "
        "d-axis. Use `torch.cat([..., ..., ...], dim=-1)` (NOT stack).\n\n"
        "Return shape `(NR, NT, 9)`. The test asserts the slicing `[..., :3] == A`, `[..., 3:6] == B`, "
        "`[..., 6:] == C` for every (r, t)."
    ),
    "stub_body": (
        "def cx9_broadcast_and_concat_vertices(triangles, NR):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "NT = 4\n"
        "triangles = t.randn(NT, 3, 3)  # (NT, vertex_abc, xyz)\n"
        "NR = 5\n"
        "out = cx9_broadcast_and_concat_vertices(triangles, NR)\n"
        "assert tuple(out.shape) == (NR, NT, 9), f'shape: {tuple(out.shape)}'\n"
        "# Slot layout check.\n"
        "for r in range(NR):\n"
        "    for ti in range(NT):\n"
        "        assert t.equal(out[r, ti, :3], triangles[ti, 0]), 'first 3 must be vertex A'\n"
        "        assert t.equal(out[r, ti, 3:6], triangles[ti, 1]), 'middle 3 must be vertex B'\n"
        "        assert t.equal(out[r, ti, 6:], triangles[ti, 2]), 'last 3 must be vertex C'\n"
        "\n"
        "# Case B: hand-check on simple tensor.\n"
        "tri2 = t.tensor([[[1.0, 2.0, 3.0],\n"
        "                  [4.0, 5.0, 6.0],\n"
        "                  [7.0, 8.0, 9.0]]])  # (NT=1, V=3, D=3)\n"
        "out2 = cx9_broadcast_and_concat_vertices(tri2, NR=2)\n"
        "expected_row = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])\n"
        "assert tuple(out2.shape) == (2, 1, 9)\n"
        "assert t.equal(out2[0, 0], expected_row)\n"
        "assert t.equal(out2[1, 0], expected_row)\n"
        "\n"
        "# Case C: NEGATIVE — stack would give (NR, NT, 3, 3), not (NR, NT, 9).\n"
        "assert out.ndim == 3, f'expected 3-D (NR,NT,9), got {out.ndim}-D — did you use stack?'\n"
        "\n"
        "# Case D: realistic scale.\n"
        "tri3 = t.randn(50, 3, 3)\n"
        "out3 = cx9_broadcast_and_concat_vertices(tri3, NR=100)\n"
        "assert tuple(out3.shape) == (100, 50, 9)"
    ),
    "solution_body": (
        "def cx9_broadcast_and_concat_vertices(triangles, NR):\n"
        "    A = triangles[:, 0]  # (NT, 3)\n"
        "    B = triangles[:, 1]\n"
        "    C = triangles[:, 2]\n"
        "    # Atom A (einops-repeat-broadcast): insert ray-axis as stride-0 view on each vertex.\n"
        "    A_b = repeat(A, 't d -> r t d', r=NR)\n"
        "    B_b = repeat(B, 't d -> r t d', r=NR)\n"
        "    C_b = repeat(C, 't d -> r t d', r=NR)\n"
        "    # Atom B (stack-vs-cat): we want the LAST axis extended from 3 to 9 — that's cat, not stack.\n"
        "    return t.cat([A_b, B_b, C_b], dim=-1)"
    ),
    "solution_notes": (
        "stack-vs-cat decision tree: are you ADDING an axis or GROWING one?\n"
        "  - Target has more axes than each input  → stack (inserts axis).\n"
        "  - Target has same axes as each input    → cat (extends an axis).\n\n"
        "Here each `A_b`/`B_b`/`C_b` is (NR, NT, 3) and the target is (NR, NT, 9) — same axes, "
        "9 = 3+3+3 → cat along the last axis. stack would have given (NR, NT, 3, 3)."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["einops-repeat-broadcast", "stack-vs-cat"],
    "lo": (
        "Compose einops repeat-broadcast (insert ray axis as stride-0 view) with torch.cat (extend the "
        "last axis from 3 to 9) to bundle three triangle vertices into a per-pair 9-vector."
    ),
}


# ===========================================================================
# cx10 — broadcast triangle vertices then batched solve
# ===========================================================================
spec_10 = {
    "atom_ids": ["einops-repeat", "linalg-solve-batched"],
    "subtopics": _subs(["einops-repeat", "linalg-solve-batched"]),
    "primary_atom": "linalg-solve-batched",
    "part": "part1",
    "exercise_index": 10,
    "exercise_title": "broadcast a single triangle across rays, then batched solve",
    "slug": "broadcast-triangle-then-batched-solve",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Single-triangle ray intersection over NR rays is a great warm-up for the full batched solver. "
        "The setup: one triangle (A, B, C), many rays. For each ray you solve a 3x3 system "
        "`[-D, B-A, C-A] @ [s, u, v]^T = O - A`. Vectorize across NR rays with `torch.linalg.solve`.\n\n"
        "Atoms compose like this:\n"
        "  1. `einops-repeat` broadcasts the single triangle's `(B-A)` and `(C-A)` edge vectors from "
        "`(3,)` to `(NR, 3)` so the LHS can be built per-ray.\n"
        "  2. `linalg-solve-batched` runs `torch.linalg.solve(LHS, RHS)` on the resulting `(NR, 3, 3)` "
        "and `(NR, 3)` stacks in one shot — the batch axis is leading.\n\n"
        "The reason we don't just write a Python loop: solve-batched dispatches to a fused LAPACK kernel."
    ),
    "prompt_body": (
        "Implement `cx10_intersect_rays_one_triangle(rays, A, B, C)` — for each of `NR` rays, "
        "solve the ray-triangle system for `(s, u, v)`. Ray `i` is "
        "`rays[i] = [origin_i, direction_i]` of shape `(2, 3)`.\n\n"
        "The system per-ray is\n"
        "  `[-D_i | B-A | C-A] @ [s_i, u_i, v_i]^T = O_i - A`.\n\n"
        "1. Compute edge vectors `e1 = B - A` and `e2 = C - A` — each `(3,)`.\n"
        "2. **Repeat** them to `(NR, 3)`: `repeat(e1, 'd -> r d', r=NR)` and similarly for `e2`. "
        "Stride-0 views — no copies.\n"
        "3. Negate the per-ray directions, stack `[-D, e1_b, e2_b]` along a new axis to form the "
        "`(NR, 3, 3)` LHS (column order matters: column 0 = -D, column 1 = e1, column 2 = e2). "
        "Hint: stack along `dim=-1` so columns line up.\n"
        "4. Compute the RHS `O - A` of shape `(NR, 3)`.\n"
        "5. **linalg-solve-batched**: `t.linalg.solve(LHS, RHS)` returns `(NR, 3)` — the per-ray "
        "`(s, u, v)`.\n\n"
        "Cross-check by reconstructing: for each ray, `O + s*D` should equal `A + u*e1 + v*e2`."
    ),
    "stub_body": (
        "def cx10_intersect_rays_one_triangle(rays, A, B, C):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built rays that all hit the same triangle.\n"
        "A = t.tensor([0.0, 0.0, 0.0])\n"
        "B = t.tensor([1.0, 0.0, 0.0])\n"
        "C = t.tensor([0.0, 1.0, 0.0])\n"
        "# Three rays, all from z=1 pointing down (-z), aimed at known triangle points.\n"
        "origins = t.tensor([[0.25, 0.25, 1.0], [0.5, 0.1, 1.0], [0.1, 0.5, 1.0]])\n"
        "dirs = t.tensor([[0.0, 0.0, -1.0]] * 3)\n"
        "rays = t.stack([origins, dirs], dim=1)  # (3, 2, 3)\n"
        "out = cx10_intersect_rays_one_triangle(rays, A, B, C)\n"
        "assert tuple(out.shape) == (3, 3), f'shape: {tuple(out.shape)}'\n"
        "s, u, v = out[:, 0], out[:, 1], out[:, 2]\n"
        "# All rays travel a distance of 1 along -z.\n"
        "assert t.allclose(s, t.ones(3), atol=1e-5), f's: {s}'\n"
        "# (u, v) should equal the (x, y) of each origin since triangle is at origin with edges along x and y.\n"
        "assert t.allclose(u, origins[:, 0], atol=1e-5), f'u: {u}'\n"
        "assert t.allclose(v, origins[:, 1], atol=1e-5), f'v: {v}'\n"
        "\n"
        "# Case B: reconstruction round-trip — random rays + non-degenerate triangle.\n"
        "t.manual_seed(42)\n"
        "A2 = t.randn(3)\n"
        "B2 = A2 + t.tensor([1.0, 0.0, 0.0])\n"
        "C2 = A2 + t.tensor([0.0, 1.0, 0.0])\n"
        "rays2 = t.randn(20, 2, 3)\n"
        "# Make rays point somewhere consistent.\n"
        "rays2[:, 1] = t.tensor([0.0, 0.0, -1.0])\n"
        "out2 = cx10_intersect_rays_one_triangle(rays2, A2, B2, C2)\n"
        "assert tuple(out2.shape) == (20, 3)\n"
        "s2, u2, v2 = out2[:, 0:1], out2[:, 1:2], out2[:, 2:3]\n"
        "lhs = rays2[:, 0] + s2 * rays2[:, 1]  # O + s*D\n"
        "e1_2 = B2 - A2\n"
        "e2_2 = C2 - A2\n"
        "rhs = A2 + u2 * e1_2 + v2 * e2_2\n"
        "assert t.allclose(lhs, rhs, atol=1e-4), f'round-trip failed: max diff {(lhs-rhs).abs().max()}'"
    ),
    "solution_body": (
        "def cx10_intersect_rays_one_triangle(rays, A, B, C):\n"
        "    NR = rays.shape[0]\n"
        "    O = rays[:, 0]   # (NR, 3)\n"
        "    D = rays[:, 1]   # (NR, 3)\n"
        "    e1 = B - A       # (3,)\n"
        "    e2 = C - A       # (3,)\n"
        "    # Atom A (einops-repeat): broadcast (3,) edges to (NR, 3) — stride-0 views.\n"
        "    e1_b = repeat(e1, 'd -> r d', r=NR)\n"
        "    e2_b = repeat(e2, 'd -> r d', r=NR)\n"
        "    # Build LHS: columns [-D, e1_b, e2_b] — stack along last dim so each column is a 3-vec.\n"
        "    LHS = t.stack([-D, e1_b, e2_b], dim=-1)  # (NR, 3, 3)\n"
        "    RHS = O - A                                # (NR, 3)\n"
        "    # Atom B (linalg-solve-batched): solve all NR systems in one fused call.\n"
        "    return t.linalg.solve(LHS, RHS)"
    ),
    "solution_notes": (
        "`torch.linalg.solve(A, b)` accepts a leading batch axis: `A` shape `(*, n, n)`, `b` shape "
        "`(*, n)`. Under the hood it dispatches to a batched LAPACK gesv, which is dramatically faster "
        "than a Python `for` loop over single solves. The einops repeats are zero-copy, so the dominant "
        "cost is the solve itself — exactly what you want."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["einops-repeat", "linalg-solve-batched"],
    "lo": (
        "Compose einops repeat (broadcast a constant triangle to NR copies) with torch.linalg.solve "
        "batched (single fused LAPACK call across the ray axis) to solve NR ray-triangle systems at once."
    ),
}


# ===========================================================================
# cx11 — repeat-broadcast LHS into batched solve over (NR, NT)
# ===========================================================================
spec_11 = {
    "atom_ids": ["einops-repeat-broadcast", "linalg-solve-batched"],
    "subtopics": _subs(["einops-repeat-broadcast", "linalg-solve-batched"]),
    "primary_atom": "linalg-solve-batched",
    "part": "part1",
    "exercise_index": 11,
    "exercise_title": "every-ray every-triangle solve via repeat-broadcast LHS",
    "slug": "repeat-broadcast-lhs-into-batched-solve",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Generalize cx10 from one triangle to many: solve every (ray, triangle) pair in one batched "
        "call. The hard step is building the `(NR, NT, 3, 3)` LHS without materializing redundant copies.\n\n"
        "Atoms compose:\n"
        "  1. `einops-repeat-broadcast` inserts the t-axis on per-ray data and the r-axis on per-tri "
        "data, both as stride-0 views.\n"
        "  2. `linalg-solve-batched` accepts ANY leading batch shape — `(*, n, n)` works for a "
        "`(NR, NT, 3, 3)` LHS just as well as a `(NR, 3, 3)`. The batch dimensions can be plural.\n\n"
        "Result: `(NR, NT, 3)` of `(s, u, v)` for every pair. The solve broadcasts the rays across "
        "triangles and vice-versa for free because repeat-broadcast already aligned the shapes."
    ),
    "prompt_body": (
        "Implement `cx11_intersect_all_pairs(rays, triangles)` — for every (ray, triangle) pair, "
        "solve the 3x3 system and return `(NR, NT, 3)` of `(s, u, v)`.\n\n"
        "- `rays`:      `(NR, 2, 3)`  — `[origin, direction]` per ray.\n"
        "- `triangles`: `(NT, 3, 3)` — three vertices A/B/C per triangle.\n\n"
        "Steps:\n"
        "1. Split rays into `O: (NR, 3)`, `D: (NR, 3)`. Split triangles into `A, B, C: (NT, 3)` each. "
        "Compute `e1 = B - A`, `e2 = C - A` — `(NT, 3)`.\n"
        "2. **Repeat-broadcast** to the (NR, NT, 3) shape, all stride-0 views:\n"
        "   - `O_b = repeat(O, 'r d -> r t d', t=NT)`\n"
        "   - `D_b = repeat(D, 'r d -> r t d', t=NT)`\n"
        "   - `A_b = repeat(A, 't d -> r t d', r=NR)`\n"
        "   - `e1_b = repeat(e1, 't d -> r t d', r=NR)`, `e2_b = repeat(e2, 't d -> r t d', r=NR)`\n"
        "3. Build LHS `(NR, NT, 3, 3)` by stacking `[-D_b, e1_b, e2_b]` as COLUMNS (stack along `dim=-1`).\n"
        "4. Build RHS `(NR, NT, 3)` = `O_b - A_b`.\n"
        "5. **linalg-solve-batched**: `t.linalg.solve(LHS, RHS)` — the leading `(NR, NT)` is the "
        "batch shape; LAPACK handles all NR*NT solves in one shot.\n\n"
        "Return shape `(NR, NT, 3)`. Cross-check against the per-pair Python-loop reference (the test "
        "computes both and asserts allclose)."
    ),
    "stub_body": (
        "def cx11_intersect_all_pairs(rays, triangles):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: small case — cross-check against a Python-loop reference using cx10's logic.\n"
        "t.manual_seed(0)\n"
        "NR, NT = 4, 3\n"
        "rays = t.randn(NR, 2, 3)\n"
        "# Force directions to point along -z so all solves are well-posed.\n"
        "rays[:, 1] = t.tensor([0.1, 0.2, -1.0])\n"
        "tri_centers = t.randn(NT, 3) * 0.1\n"
        "triangles = t.stack([\n"
        "    tri_centers,\n"
        "    tri_centers + t.tensor([1.0, 0.0, 0.0]),\n"
        "    tri_centers + t.tensor([0.0, 1.0, 0.0]),\n"
        "], dim=1)  # (NT, 3, 3)\n"
        "out = cx11_intersect_all_pairs(rays, triangles)\n"
        "assert tuple(out.shape) == (NR, NT, 3), f'shape: {tuple(out.shape)}'\n"
        "\n"
        "# Reference loop.\n"
        "ref = t.zeros(NR, NT, 3)\n"
        "for ri in range(NR):\n"
        "    O = rays[ri, 0]; D = rays[ri, 1]\n"
        "    for ti in range(NT):\n"
        "        A = triangles[ti, 0]; B = triangles[ti, 1]; C = triangles[ti, 2]\n"
        "        LHS = t.stack([-D, B - A, C - A], dim=-1)\n"
        "        RHS = O - A\n"
        "        ref[ri, ti] = t.linalg.solve(LHS, RHS)\n"
        "assert t.allclose(out, ref, atol=1e-4), f'max diff {(out - ref).abs().max()}'\n"
        "\n"
        "# Case B: reconstruction round-trip on one slot.\n"
        "ri, ti = 2, 1\n"
        "s, u, v = out[ri, ti]\n"
        "O = rays[ri, 0]; D = rays[ri, 1]\n"
        "A = triangles[ti, 0]; e1 = triangles[ti, 1] - A; e2 = triangles[ti, 2] - A\n"
        "lhs_pt = O + s * D\n"
        "rhs_pt = A + u * e1 + v * e2\n"
        "assert t.allclose(lhs_pt, rhs_pt, atol=1e-4), f'point mismatch: {lhs_pt} vs {rhs_pt}'\n"
        "\n"
        "# Case C: realistic ARENA-ish scale.\n"
        "rays3 = t.randn(30, 2, 3)\n"
        "rays3[:, 1] = t.tensor([0.0, 0.0, -1.0])\n"
        "tri3 = t.randn(20, 3, 3)\n"
        "out3 = cx11_intersect_all_pairs(rays3, tri3)\n"
        "assert tuple(out3.shape) == (30, 20, 3)"
    ),
    "solution_body": (
        "def cx11_intersect_all_pairs(rays, triangles):\n"
        "    NR = rays.shape[0]\n"
        "    NT = triangles.shape[0]\n"
        "    O = rays[:, 0]    # (NR, 3)\n"
        "    D = rays[:, 1]    # (NR, 3)\n"
        "    A = triangles[:, 0]  # (NT, 3)\n"
        "    e1 = triangles[:, 1] - A  # (NT, 3)\n"
        "    e2 = triangles[:, 2] - A  # (NT, 3)\n"
        "    # Atom A (einops-repeat-broadcast): align everything to (NR, NT, 3) as stride-0 views.\n"
        "    O_b  = repeat(O,  'r d -> r t d', t=NT)\n"
        "    D_b  = repeat(D,  'r d -> r t d', t=NT)\n"
        "    A_b  = repeat(A,  't d -> r t d', r=NR)\n"
        "    e1_b = repeat(e1, 't d -> r t d', r=NR)\n"
        "    e2_b = repeat(e2, 't d -> r t d', r=NR)\n"
        "    # Stack columns [-D, e1, e2] to form a (NR, NT, 3, 3) LHS.\n"
        "    LHS = t.stack([-D_b, e1_b, e2_b], dim=-1)\n"
        "    RHS = O_b - A_b\n"
        "    # Atom B (linalg-solve-batched): the leading (NR, NT) is the batch shape.\n"
        "    return t.linalg.solve(LHS, RHS)"
    ),
    "solution_notes": (
        "`torch.linalg.solve` accepts arbitrary batch shapes — `(NR, NT, 3, 3)` is fine, the function "
        "broadcasts/loops internally. Because all the einops repeats are stride-0 views, the input "
        "footprint stays O(NR + NT), not O(NR*NT). The fused LAPACK call is the only meaningful "
        "allocation. This is exactly the form ARENA uses for its `raytrace_mesh` kernel."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["einops-repeat-broadcast", "linalg-solve-batched"],
    "lo": (
        "Compose einops repeat-broadcast (align ray + triangle data to (NR, NT, ...) stride-0 views) "
        "with batched torch.linalg.solve to compute every-ray every-triangle (s, u, v) in one fused call."
    ),
}


# ===========================================================================
# cx12 — broadcast eye(3) fallback over NR via repeat + singular-matrix mask
# ===========================================================================
spec_12 = {
    "atom_ids": ["einops-repeat", "singular-matrix-mask-trick"],
    "subtopics": _subs(["einops-repeat", "singular-matrix-mask-trick"]),
    "primary_atom": "singular-matrix-mask-trick",
    "part": "part1",
    "exercise_index": 12,
    "exercise_title": "swap singular LHS for a broadcast eye(3) before solve",
    "slug": "broadcast-eye3-fallback-for-singular",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Real ARENA ray batches contain some degenerate (ray, triangle) pairs whose LHS is singular — "
        "`linalg.solve` would raise. The standard trick is to MASK them out BEFORE the solve: detect "
        "the singular rows via `|det| < eps` and overwrite their LHS with `eye(3)` (always solvable). "
        "After the solve you reset those slots to a sentinel (inf or NaN) so downstream filters skip them.\n\n"
        "Atoms compose:\n"
        "  1. `einops-repeat` broadcasts a single `eye(3)` into `(NR, 3, 3)` — stride-0 view, no copy.\n"
        "  2. `singular-matrix-mask-trick`: compute the per-row det, build a singular-mask, "
        "`torch.where(mask, eye_b, LHS)` to substitute, run `linalg.solve`, then overwrite the masked "
        "rows of the result with `inf`.\n\n"
        "The composition produces a robust solver that never throws on singular slots."
    ),
    "prompt_body": (
        "Implement `cx12_safe_batched_solve(LHS, RHS, eps=1e-8)` — a batched solve that gracefully "
        "handles singular matrices.\n\n"
        "- `LHS` shape `(NR, 3, 3)`, `RHS` shape `(NR, 3)`.\n"
        "- A row is 'singular' if `|det(LHS[i])| < eps`.\n\n"
        "Steps:\n"
        "1. Compute the per-row det `(NR,)` and build the mask `singular = |det| < eps`.\n"
        "2. **Repeat** an `eye(3)` across the NR axis: `eye_b = repeat(t.eye(3), 'a b -> r a b', r=NR)`. "
        "Stride-0 view, no copy.\n"
        "3. **Mask-substitute** LHS: where `singular[i]` is True, swap `LHS[i]` for `eye_b[i]`. Use "
        "`torch.where(singular[:, None, None], eye_b, LHS)` to broadcast the mask across the trailing dims.\n"
        "4. Run `t.linalg.solve(LHS_safe, RHS)` — gives shape `(NR, 3)`.\n"
        "5. Overwrite the singular slots of the result with `+inf`: `out[singular] = float('inf')`.\n\n"
        "Return the cleaned `(NR, 3)` result."
    ),
    "stub_body": (
        "def cx12_safe_batched_solve(LHS, RHS, eps=1e-8):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: NO singular matrices — output must match the plain batched solve.\n"
        "t.manual_seed(0)\n"
        "NR = 5\n"
        "LHS_good = t.randn(NR, 3, 3)\n"
        "# Make sure they're well-conditioned: add a strong diagonal.\n"
        "LHS_good += 3.0 * t.eye(3)\n"
        "RHS = t.randn(NR, 3)\n"
        "out = cx12_safe_batched_solve(LHS_good, RHS)\n"
        "ref = t.linalg.solve(LHS_good, RHS)\n"
        "assert tuple(out.shape) == (NR, 3), f'shape: {tuple(out.shape)}'\n"
        "assert t.allclose(out, ref, atol=1e-5), 'no-singular path must equal plain solve'\n"
        "\n"
        "# Case B: ONE singular matrix in the middle — that slot must be inf, others fine.\n"
        "LHS_mix = LHS_good.clone()\n"
        "# Make slot 2 singular (rank deficient — copy row 0 into row 1).\n"
        "LHS_mix[2, 1] = LHS_mix[2, 0]\n"
        "out_mix = cx12_safe_batched_solve(LHS_mix, RHS)\n"
        "assert tuple(out_mix.shape) == (NR, 3)\n"
        "# Slot 2 should be inf.\n"
        "assert t.isinf(out_mix[2]).all(), f'singular slot must be inf, got {out_mix[2]}'\n"
        "# Other slots match the plain solve.\n"
        "for i in [0, 1, 3, 4]:\n"
        "    plain_i = t.linalg.solve(LHS_mix[i], RHS[i])\n"
        "    assert t.allclose(out_mix[i], plain_i, atol=1e-5), f'slot {i} diverges'\n"
        "\n"
        "# Case C: ALL singular — every slot inf, no exception.\n"
        "LHS_bad = t.zeros(3, 3, 3)\n"
        "RHS_bad = t.randn(3, 3)\n"
        "out_bad = cx12_safe_batched_solve(LHS_bad, RHS_bad)\n"
        "assert tuple(out_bad.shape) == (3, 3)\n"
        "assert t.isinf(out_bad).all(), 'all-singular result must be all-inf'\n"
        "\n"
        "# Case D: realistic mixed batch.\n"
        "NR2 = 20\n"
        "L2 = t.randn(NR2, 3, 3) + 3.0 * t.eye(3)\n"
        "# Sprinkle some singular slots.\n"
        "for idx in [3, 7, 15]:\n"
        "    L2[idx] = t.zeros(3, 3)\n"
        "R2 = t.randn(NR2, 3)\n"
        "out2 = cx12_safe_batched_solve(L2, R2)\n"
        "assert tuple(out2.shape) == (NR2, 3)\n"
        "for idx in [3, 7, 15]:\n"
        "    assert t.isinf(out2[idx]).all(), f'slot {idx} must be inf'\n"
        "# Non-singular slot stays finite.\n"
        "assert t.isfinite(out2[0]).all()"
    ),
    "solution_body": (
        "def cx12_safe_batched_solve(LHS, RHS, eps=1e-8):\n"
        "    NR = LHS.shape[0]\n"
        "    # Detect singular rows via |det|.\n"
        "    dets = t.linalg.det(LHS)            # (NR,)\n"
        "    singular = dets.abs() < eps         # (NR,) bool\n"
        "    # Atom A (einops-repeat): broadcast a single eye(3) to (NR, 3, 3) — stride-0 view.\n"
        "    eye_b = repeat(t.eye(3, dtype=LHS.dtype), 'a b -> r a b', r=NR)\n"
        "    # Atom B (singular-matrix-mask-trick): swap singular LHS rows for eye(3) BEFORE solving.\n"
        "    LHS_safe = t.where(singular[:, None, None], eye_b, LHS)\n"
        "    out = t.linalg.solve(LHS_safe, RHS)\n"
        "    # Sentinel the masked slots so downstream filters can drop them.\n"
        "    out[singular] = float('inf')\n"
        "    return out"
    ),
    "solution_notes": (
        "The eye(3)-substitution trick is the standard ARENA pattern for never-throw batched solves. "
        "Why eye(3)? Because `solve(eye, b) = b` always, regardless of `b` — so the substituted slots "
        "run through the kernel without raising, and we then overwrite them with `inf` as a sentinel. "
        "The einops repeat keeps the substitution cheap: the `(NR, 3, 3)` eye is a stride-0 view of a "
        "single 3x3 buffer, so memory cost is O(9), not O(9*NR)."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["einops-repeat", "singular-matrix-mask-trick"],
    "lo": (
        "Compose einops repeat (broadcast a single eye(3) over the batch) with the singular-matrix "
        "mask trick (where-substitute then sentinel-replace) to build a never-throw batched solver."
    ),
}


SPECS = [spec_7, spec_8, spec_9, spec_10, spec_11, spec_12]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
