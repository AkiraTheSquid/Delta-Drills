"""Composite drills cx25..cx30 — batch-17 (Z-cell, part1).

Six composite procedural drills exercising 2-atom pairs from the ARENA ray-
tracing masking machinery (ARENA part 1 — ray tracing, masking section).
The shared anchor atom is `boolean-mask-combine`; each cx pairs it with a
neighbour atom that lives in the same ARENA ray-tracing dataflow.

cx25  boolean-mask-combine + einops-repeat
cx26  boolean-mask-combine + einops-repeat-broadcast
cx27  boolean-mask-combine + linalg-solve-batched
cx28  boolean-mask-combine + ray-parametric-form
cx29  boolean-mask-combine + singular-matrix-mask-trick
cx30  boolean-mask-combine + stack-vs-cat
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
# cx25 — combine a per-ray mask with a per-triangle mask via repeat
# ===========================================================================
spec_25 = {
    "atom_ids": ["boolean-mask-combine", "einops-repeat"],
    "subtopics": _subs(["boolean-mask-combine", "einops-repeat"]),
    "primary_atom": "boolean-mask-combine",
    "part": "part1",
    "exercise_index": 25,
    "exercise_title": "combine per-ray and per-triangle masks across a repeated dim",
    "slug": "combine-masks-across-repeated-dim",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "In ARENA ray-tracing we end up with **two predicates that live on different axes**:\n"
        "- `ray_ok` — shape `(NR,)` — \"this ray is in-bounds / not degenerate\".\n"
        "- `tri_ok` — shape `(NT,)` — \"this triangle is front-facing\".\n\n"
        "To AND them into a per-(ray, triangle) mask of shape `(NR, NT)` we need to **align the axes**. "
        "The pattern: `einops.repeat` the per-ray mask along a new `NT` axis (and the per-triangle mask "
        "along a new `NR` axis), then boolean-AND the two `(NR, NT)` masks elementwise.\n\n"
        "Why `repeat` and not raw broadcasting? `repeat` makes the intent explicit and the resulting "
        "tensors have the matching final shape — convenient when downstream code asserts on `mask.shape`.\n\n"
        "**Anatomy.** `repeat(ray_ok, 'r -> r t', t=NT)` materializes the ray mask across triangles; "
        "`repeat(tri_ok, 't -> r t', r=NR)` does the symmetric thing. Combine with `&`."
    ),
    "prompt_body": (
        "Implement `cx25_combine_ray_tri_masks(ray_ok, tri_ok)`.\n\n"
        "- `ray_ok`: boolean tensor of shape `(NR,)`.\n"
        "- `tri_ok`: boolean tensor of shape `(NT,)`.\n"
        "- Return: boolean tensor of shape `(NR, NT)` where `out[r, t] == ray_ok[r] & tri_ok[t]`.\n\n"
        "1. **Repeat** — use `einops.repeat` to lift `ray_ok` to shape `(NR, NT)` and `tri_ok` to "
        "shape `(NR, NT)`.\n"
        "2. **Boolean combine** — AND the two `(NR, NT)` masks.\n\n"
        "The test fuzzes random `(NR, NT)` shapes and cross-checks against the naive outer-AND "
        "`ray_ok[:, None] & tri_ok[None, :]`."
    ),
    "stub_body": (
        "def cx25_combine_ray_tri_masks(ray_ok, tri_ok):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: tiny hand-built example.\n"
        "ray_ok = t.tensor([True, False, True, True])\n"
        "tri_ok = t.tensor([True, True, False])\n"
        "out = cx25_combine_ray_tri_masks(ray_ok, tri_ok)\n"
        "assert out.dtype == t.bool, f'expected bool, got {out.dtype}'\n"
        "assert tuple(out.shape) == (4, 3), f'expected (4,3), got {tuple(out.shape)}'\n"
        "expected = ray_ok[:, None] & tri_ok[None, :]\n"
        "assert t.equal(out, expected), 'mask mismatch — did you AND across the right axes?'\n"
        "\n"
        "# Case B: edge — all True.\n"
        "ray_ok = t.ones(5, dtype=t.bool)\n"
        "tri_ok = t.ones(7, dtype=t.bool)\n"
        "out = cx25_combine_ray_tri_masks(ray_ok, tri_ok)\n"
        "assert tuple(out.shape) == (5, 7)\n"
        "assert out.all().item()\n"
        "\n"
        "# Case C: edge — one side all False.\n"
        "ray_ok = t.zeros(3, dtype=t.bool)\n"
        "tri_ok = t.tensor([True, False, True, True])\n"
        "out = cx25_combine_ray_tri_masks(ray_ok, tri_ok)\n"
        "assert tuple(out.shape) == (3, 4)\n"
        "assert not out.any().item()\n"
        "\n"
        "# Case D: fuzz against the naive outer-AND reference.\n"
        "rng = t.Generator().manual_seed(17)\n"
        "for NR, NT in [(2, 3), (10, 4), (1, 8), (8, 1), (16, 16)]:\n"
        "    ro = (t.rand(NR, generator=rng) > 0.4)\n"
        "    to_ = (t.rand(NT, generator=rng) > 0.4)\n"
        "    out = cx25_combine_ray_tri_masks(ro, to_)\n"
        "    assert tuple(out.shape) == (NR, NT)\n"
        "    assert t.equal(out, ro[:, None] & to_[None, :])"
    ),
    "solution_body": (
        "def cx25_combine_ray_tri_masks(ray_ok, tri_ok):\n"
        "    NR = ray_ok.shape[0]\n"
        "    NT = tri_ok.shape[0]\n"
        "    # Atom A (einops-repeat): lift each per-axis mask onto the joint (NR, NT) grid.\n"
        "    ray_grid = repeat(ray_ok, 'r -> r t', t=NT)\n"
        "    tri_grid = repeat(tri_ok, 't -> r t', r=NR)\n"
        "    # Atom B (boolean-mask-combine): elementwise AND on the matching shape.\n"
        "    return ray_grid & tri_grid"
    ),
    "solution_notes": (
        "`einops.repeat` here is doing the same job as `ray_ok[:, None].expand(NR, NT)` — the result "
        "is logically broadcast across the new axis. The explicit `repeat` makes the joint axis "
        "intent self-documenting, which is a win in long ARENA ray-tracing pipelines where many "
        "`(NR, NT)` tensors need to line up axis-for-axis."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["boolean-mask-combine", "einops-repeat"],
    "lo": (
        "Compose einops repeat (lift per-axis masks onto a joint grid) with boolean mask combine "
        "(elementwise AND) to construct the (NR, NT) per-ray-per-triangle predicate."
    ),
}


# ===========================================================================
# cx26 — combine masks via repeat-broadcast (every-ray-with-every-triangle)
# ===========================================================================
spec_26 = {
    "atom_ids": ["boolean-mask-combine", "einops-repeat-broadcast"],
    "subtopics": _subs(["boolean-mask-combine", "einops-repeat-broadcast"]),
    "primary_atom": "boolean-mask-combine",
    "part": "part1",
    "exercise_index": 26,
    "exercise_title": "combine masks via repeat-broadcast pairing — every ray vs every triangle",
    "slug": "combine-masks-via-repeat-broadcast",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Same problem shape as cx25 but a different mechanic. `einops.repeat` with an axis name that "
        "matches the SOURCE shape produces a **stride-0 view** — no copy, no materialized tensor — "
        "just a broadcast view sized for the joint `(NR, NT)` grid. This is the einops-native version "
        "of `tensor.expand(...)`.\n\n"
        "**The pairing pattern.** When you want every ray to be paired with every triangle, you "
        "`repeat` *both* per-axis tensors onto the joint grid with stride-0 views, then `&` them. The "
        "result is a real boolean tensor — but the inputs to `&` were free.\n\n"
        "**Anatomy.**\n"
        "- `repeat(ray_ok, 'r -> r t', t=NT)` — `(NR,)` lifted to `(NR, NT)` via stride-0 broadcast on `t`.\n"
        "- `repeat(tri_ok, 't -> r t', r=NR)` — `(NT,)` lifted to `(NR, NT)` via stride-0 broadcast on `r`.\n"
        "- `&` — materializes the boolean grid, ONE allocation total.\n\n"
        "**Why care.** In ARENA the rays-vs-triangles cross product is the inner loop. A stride-0 lift "
        "keeps memory traffic linear in `NR + NT`, not `NR * NT`, up until the final AND."
    ),
    "prompt_body": (
        "Implement `cx26_pair_masks_broadcast(ray_ok, tri_ok)` — same I/O contract as cx25, but THIS "
        "drill exercises the broadcast/no-copy property of einops repeat.\n\n"
        "- `ray_ok`: boolean tensor of shape `(NR,)`.\n"
        "- `tri_ok`: boolean tensor of shape `(NT,)`.\n"
        "- Return: boolean tensor of shape `(NR, NT)` matching `ray_ok[:, None] & tri_ok[None, :]`.\n\n"
        "1. **Repeat-broadcast** — use `einops.repeat` to lift both masks. Pre-AND, the two views "
        "must be stride-0 along their broadcast axis (no copy).\n"
        "2. **Boolean combine** — `&` to materialize the joint grid.\n\n"
        "The test verifies the values AND probes the stride-0 property of the intermediate broadcasts."
    ),
    "stub_body": (
        "def cx26_pair_masks_broadcast(ray_ok, tri_ok):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx26_lift_only(ray_ok, tri_ok):\n"
        "    \"\"\"Return the two pre-AND broadcast views, for stride-0 inspection.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built example, check the AND matches the outer.\n"
        "ray_ok = t.tensor([True, True, False])\n"
        "tri_ok = t.tensor([False, True, True, True])\n"
        "out = cx26_pair_masks_broadcast(ray_ok, tri_ok)\n"
        "assert out.dtype == t.bool\n"
        "assert tuple(out.shape) == (3, 4)\n"
        "assert t.equal(out, ray_ok[:, None] & tri_ok[None, :])\n"
        "\n"
        "# Case B: stride-0 inspection on the intermediate lifts.\n"
        "ray_ok = t.tensor([True, False, True, True, False])\n"
        "tri_ok = t.tensor([True, True, False])\n"
        "ray_grid, tri_grid = cx26_lift_only(ray_ok, tri_ok)\n"
        "assert tuple(ray_grid.shape) == (5, 3)\n"
        "assert tuple(tri_grid.shape) == (5, 3)\n"
        "# stride-0 along the broadcast axis = repeat is a view, not a copy.\n"
        "assert ray_grid.stride()[1] == 0, f'ray_grid stride along t should be 0 (broadcast view), got {ray_grid.stride()}'\n"
        "assert tri_grid.stride()[0] == 0, f'tri_grid stride along r should be 0 (broadcast view), got {tri_grid.stride()}'\n"
        "\n"
        "# Case C: fuzz vs reference.\n"
        "rng = t.Generator().manual_seed(26)\n"
        "for NR, NT in [(1, 5), (12, 7), (3, 3), (32, 1)]:\n"
        "    ro = (t.rand(NR, generator=rng) > 0.5)\n"
        "    to_ = (t.rand(NT, generator=rng) > 0.5)\n"
        "    out = cx26_pair_masks_broadcast(ro, to_)\n"
        "    assert tuple(out.shape) == (NR, NT)\n"
        "    assert t.equal(out, ro[:, None] & to_[None, :])"
    ),
    "solution_body": (
        "def cx26_lift_only(ray_ok, tri_ok):\n"
        "    NR = ray_ok.shape[0]\n"
        "    NT = tri_ok.shape[0]\n"
        "    # einops repeat with a name-only target axis = stride-0 broadcast view, no copy.\n"
        "    ray_grid = repeat(ray_ok, 'r -> r t', t=NT)\n"
        "    tri_grid = repeat(tri_ok, 't -> r t', r=NR)\n"
        "    return ray_grid, tri_grid\n"
        "\n"
        "def cx26_pair_masks_broadcast(ray_ok, tri_ok):\n"
        "    ray_grid, tri_grid = cx26_lift_only(ray_ok, tri_ok)\n"
        "    # Atom (boolean-mask-combine): ONE materializing op on the joint shape.\n"
        "    return ray_grid & tri_grid"
    ),
    "solution_notes": (
        "Note both helper and combine return the same answer as cx25, but the intermediate tensors "
        "are stride-0 views — proving the repeat was a broadcast, not a tile. The `& ` is the only "
        "place memory is allocated for the full `(NR, NT)` grid."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["boolean-mask-combine", "einops-repeat-broadcast"],
    "lo": (
        "Compose einops repeat-broadcast (stride-0 lifts onto the joint NR x NT grid) with "
        "boolean mask combine (single materializing AND) to build the every-ray-with-every-triangle "
        "predicate without copying the inputs."
    ),
}


# ===========================================================================
# cx27 — mask the linalg.solve survivors (drop non-finite / out-of-range)
# ===========================================================================
spec_27 = {
    "atom_ids": ["boolean-mask-combine", "linalg-solve-batched"],
    "subtopics": _subs(["boolean-mask-combine", "linalg-solve-batched"]),
    "primary_atom": "boolean-mask-combine",
    "part": "part1",
    "exercise_index": 27,
    "exercise_title": "mask the linalg.solve survivors — finite AND in-range",
    "slug": "mask-solve-survivors",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's triangle-intersection pipeline solves a batched `(B, 3, 3)` linear system to recover "
        "`(u, v, w)` barycentric coordinates per (ray, triangle) candidate. Two failure modes:\n"
        "- **Non-finite outputs** — even if the system isn't exactly singular, near-singular slices "
        "can return inf/nan. We mask those out with `t.isfinite(x).all(dim=-1)`.\n"
        "- **Out-of-range outputs** — even a clean solve can return `u, v` outside `[0, 1]`, which "
        "means the intersection is outside the triangle. We mask those out with `(0 <= u) & (u <= 1) "
        "& (0 <= v) & (v <= 1)`.\n\n"
        "The composition: do the batched solve first, then AND the two predicates into a single "
        "`(B,)` boolean — the rays that survive both checks.\n\n"
        "**Anatomy.**\n"
        "1. `x = t.linalg.solve(A, b)` — solve every slice at once, shape `(B, 3)`.\n"
        "2. `finite = t.isfinite(x).all(dim=-1)` — per-slice finiteness check.\n"
        "3. `in_range = (x >= 0).all(dim=-1) & (x <= 1).all(dim=-1)` — per-slice range check.\n"
        "4. `valid = finite & in_range` — boolean-AND combine.\n\n"
        "The `& ` is doing real work: it collapses two `(B,)` predicates into a single mask the "
        "caller can use to index back into the original batch."
    ),
    "prompt_body": (
        "Implement `cx27_solve_and_mask(A, b)`.\n\n"
        "- `A`: float tensor of shape `(B, N, N)`. May include ill-conditioned slices, but no slice "
        "is exactly singular (`det == 0`) — that's the job of cx29.\n"
        "- `b`: float tensor of shape `(B, N)`.\n\n"
        "Return `(x, valid)`:\n"
        "- `x`: solve result, shape `(B, N)`. NaNs / infs in the raw output are allowed; do NOT "
        "post-process them.\n"
        "- `valid`: boolean tensor of shape `(B,)` where `valid[i]` is True iff `x[i]` is entirely "
        "finite AND all of its entries are in `[0, 1]`.\n\n"
        "1. **Batched solve** — one call to `t.linalg.solve(A, b)`. No loops.\n"
        "2. **Mask combine** — AND `finite` and `in_range` into `valid`."
    ),
    "stub_body": (
        "def cx27_solve_and_mask(A, b):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: clean 2x2 batch, all slices solvable AND in-range.\n"
        "A = t.tensor([\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "    [[2.0, 0.0], [0.0, 2.0]],\n"
        "])\n"
        "b = t.tensor([[0.3, 0.4], [0.5, 0.5]])  # solutions: [0.3, 0.4] and [0.25, 0.25].\n"
        "x, valid = cx27_solve_and_mask(A, b)\n"
        "assert tuple(x.shape) == (2, 2)\n"
        "assert tuple(valid.shape) == (2,)\n"
        "assert valid.dtype == t.bool\n"
        "assert valid.all().item(), f'expected both valid, got {valid}'\n"
        "assert t.allclose(x, t.tensor([[0.3, 0.4], [0.25, 0.25]]), atol=1e-5)\n"
        "\n"
        "# Case B: out-of-range slice — solve succeeds but answer is outside [0, 1].\n"
        "A = t.tensor([\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "])\n"
        "b = t.tensor([[0.3, 0.4], [2.5, -1.0]])  # 2nd slice is way out of [0,1].\n"
        "x, valid = cx27_solve_and_mask(A, b)\n"
        "assert valid.tolist() == [True, False], f'expected [True, False], got {valid.tolist()}'\n"
        "\n"
        "# Case C: ill-conditioned (very large cond number) but representable — solve succeeds but\n"
        "# the result is huge in magnitude, so it should fail the in-range mask.\n"
        "A = t.tensor([\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "    [[1.0, 1.0], [1.0, 1.0001]],  # nearly rank-1 but not exactly singular.\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "])\n"
        "b = t.tensor([[0.2, 0.3], [1.0, 1.0], [0.5, 0.5]])\n"
        "x, valid = cx27_solve_and_mask(A, b)\n"
        "assert tuple(valid.shape) == (3,)\n"
        "# Slices 0 and 2 are clean identity solves — must be valid.\n"
        "assert valid[0].item() and valid[2].item()\n"
        "# Slice 1's solve (b = [1,1] against a near-rank-1 A whose b is in the column space) is\n"
        "# [1, 0] or [0, 1] depending on numerical luck — either way, in [0, 1], so valid.\n"
        "# What we really test here is that the solve did not crash on an ill-conditioned slice.\n"
        "# x[0] and x[2] must equal their b's exactly (identity A).\n"
        "assert t.allclose(x[0], b[0])\n"
        "assert t.allclose(x[2], b[2])\n"
        "\n"
        "# Case D: 3x3, larger batch — fuzz vs reference using identity A's.\n"
        "B = 8\n"
        "A = t.eye(3).expand(B, 3, 3).contiguous()\n"
        "b = t.rand(B, 3)  # solutions == b. Some will be in [0,1] (almost all here).\n"
        "x, valid = cx27_solve_and_mask(A, b)\n"
        "assert tuple(x.shape) == (B, 3)\n"
        "assert t.allclose(x, b, atol=1e-5)\n"
        "expected_valid = t.isfinite(b).all(dim=-1) & (b >= 0).all(dim=-1) & (b <= 1).all(dim=-1)\n"
        "assert t.equal(valid, expected_valid)"
    ),
    "solution_body": (
        "def cx27_solve_and_mask(A, b):\n"
        "    # Atom A (linalg-solve-batched): one call, no loops.\n"
        "    x = t.linalg.solve(A, b)\n"
        "    # Per-slice predicates over the last (N) axis.\n"
        "    finite = t.isfinite(x).all(dim=-1)\n"
        "    in_range = (x >= 0).all(dim=-1) & (x <= 1).all(dim=-1)\n"
        "    # Atom B (boolean-mask-combine): AND the two (B,) predicates.\n"
        "    valid = finite & in_range\n"
        "    return x, valid"
    ),
    "solution_notes": (
        "The composition is small but load-bearing: in real ARENA code `valid` is what selects the "
        "rays that actually hit a triangle. Splitting into `finite` + `in_range` makes the failure "
        "modes auditable — you can log how many slices each predicate vetoed."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["boolean-mask-combine", "linalg-solve-batched"],
    "lo": (
        "Compose batched linalg.solve (one solve over the whole (B, N, N) batch) with boolean mask "
        "combine (AND finite-ness with in-range) to produce the per-slice survivor predicate."
    ),
}


# ===========================================================================
# cx28 — combine ray-form constraint (t >= 0) with barycentric constraints
# ===========================================================================
spec_28 = {
    "atom_ids": ["boolean-mask-combine", "ray-parametric-form"],
    "subtopics": _subs(["boolean-mask-combine", "ray-parametric-form"]),
    "primary_atom": "boolean-mask-combine",
    "part": "part1",
    "exercise_index": 28,
    "exercise_title": "combine valid-ray mask (t >= 0) with barycentric constraints",
    "slug": "combine-valid-ray-mask-with-bary",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's `triangle_ray_intersects` decomposes the per-(ray, triangle) intersection test into "
        "two physical predicates:\n"
        "1. **Ray-form constraint** — the intersection's `t` parameter (along `R(t) = O + t*D`) must "
        "be `>= 0`. A negative `t` means the triangle is *behind* the ray's origin, which by "
        "convention doesn't count as an intersection.\n"
        "2. **Barycentric constraints** — the intersection's `(u, v)` inside the triangle must "
        "satisfy `u >= 0 AND v >= 0 AND u + v <= 1`. Otherwise the line-plane intersection point is "
        "outside the triangle's three edges.\n\n"
        "The composition: compute `t`, `u`, `v` per (ray, triangle), build each predicate as a "
        "boolean tensor of the SAME shape, then AND them all together.\n\n"
        "**Anatomy.**\n"
        "- `ray_form_ok = t_param >= 0`  (atom: ray-parametric-form — interpreting `t` per the ray "
        "equation)\n"
        "- `bary_ok = (u >= 0) & (v >= 0) & (u + v <= 1)`\n"
        "- `hits = ray_form_ok & bary_ok`  (atom: boolean-mask-combine)\n\n"
        "**Why care.** Forgetting the `t >= 0` mask is the canonical ARENA debugging session: rays "
        "appear to \"hit\" triangles that are behind them. Combining it with the barycentric mask "
        "via `&` is the fix."
    ),
    "prompt_body": (
        "Implement `cx28_combine_ray_bary(t_param, u, v)`.\n\n"
        "- `t_param`: float tensor of any shape `S` — the ray-parameter at the line-plane intersection. "
        "Per the ray parametric form `R(t) = O + t*D`, this must be `>= 0` for a real hit.\n"
        "- `u`, `v`: float tensors of shape `S` — the barycentric coords inside the triangle. The "
        "third barycentric `w = 1 - u - v` is implicit.\n\n"
        "Return a boolean tensor of shape `S` where each entry is True iff:\n"
        "- `t_param >= 0` (ray-form constraint), AND\n"
        "- `0 <= u`, `0 <= v`, `u + v <= 1` (barycentric inside-triangle constraints).\n\n"
        "1. **Ray-form mask** — `t_param >= 0`.\n"
        "2. **Barycentric mask** — three predicates AND'd together.\n"
        "3. **Combine** — AND the two masks.\n\n"
        "All comparisons use `>=` and `<=` (closed boundaries — points on the triangle's edge count "
        "as hits, ARENA convention)."
    ),
    "stub_body": (
        "def cx28_combine_ray_bary(t_param, u, v):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built single-element tensors covering all four corners of the truth table.\n"
        "# row 0: t<0, bary ok    -> False (ray behind origin).\n"
        "# row 1: t>=0, bary ok   -> True.\n"
        "# row 2: t>=0, u<0       -> False (outside triangle).\n"
        "# row 3: t>=0, u+v>1     -> False.\n"
        "t_param = t.tensor([-0.5, 1.0, 1.0, 1.0])\n"
        "u = t.tensor([0.3, 0.3, -0.1, 0.6])\n"
        "v = t.tensor([0.3, 0.3, 0.5, 0.6])\n"
        "out = cx28_combine_ray_bary(t_param, u, v)\n"
        "assert out.dtype == t.bool, f'expected bool, got {out.dtype}'\n"
        "assert tuple(out.shape) == (4,)\n"
        "assert out.tolist() == [False, True, False, False], f'got {out.tolist()}'\n"
        "\n"
        "# Case B: closed-boundary convention — exactly-on-edge counts as hit.\n"
        "t_param = t.tensor([0.0, 1.0, 1.0, 1.0])\n"
        "u = t.tensor([0.0, 0.0, 1.0, 0.5])\n"
        "v = t.tensor([0.0, 1.0, 0.0, 0.5])\n"
        "out = cx28_combine_ray_bary(t_param, u, v)\n"
        "assert out.tolist() == [True, True, True, True], f'closed-boundary mismatch — got {out.tolist()}'\n"
        "\n"
        "# Case C: 2-D shape — every (NR, NT) entry independently masked.\n"
        "rng = t.Generator().manual_seed(28)\n"
        "NR, NT = 6, 5\n"
        "t_param = t.randn(NR, NT, generator=rng)\n"
        "u = t.randn(NR, NT, generator=rng)\n"
        "v = t.randn(NR, NT, generator=rng)\n"
        "out = cx28_combine_ray_bary(t_param, u, v)\n"
        "assert tuple(out.shape) == (NR, NT)\n"
        "expected = (t_param >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n"
        "assert t.equal(out, expected)\n"
        "\n"
        "# Case D: validate the ARENA failure mode this fixes — drop t_param mask, expect mismatch.\n"
        "t_param = t.tensor([-2.0])  # behind origin.\n"
        "u = t.tensor([0.3])\n"
        "v = t.tensor([0.3])\n"
        "out = cx28_combine_ray_bary(t_param, u, v)\n"
        "wrong = (u >= 0) & (v >= 0) & (u + v <= 1)  # ARENA bug: missing ray-form check.\n"
        "assert out.item() is False\n"
        "assert wrong.item() is True\n"
        "assert out.item() != wrong.item(), 'cx28 must combine BOTH masks — bug if t_param >= 0 is dropped'"
    ),
    "solution_body": (
        "def cx28_combine_ray_bary(t_param, u, v):\n"
        "    # Atom A (ray-parametric-form): t < 0 means the triangle is behind the ray origin.\n"
        "    ray_form_ok = t_param >= 0\n"
        "    # Barycentric inside-triangle predicate (three constraints AND'd).\n"
        "    bary_ok = (u >= 0) & (v >= 0) & (u + v <= 1)\n"
        "    # Atom B (boolean-mask-combine): AND the two masks elementwise.\n"
        "    return ray_form_ok & bary_ok"
    ),
    "solution_notes": (
        "The `t_param >= 0` half is the ARENA-flavored expression of the ray parametric form — "
        "saying \"only points reachable by R(t>=0) count\". Forgetting it is one of the canonical "
        "ARENA bugs (rays seem to hit things behind them). The barycentric AND is the inside-"
        "triangle test. The final AND is what `boolean-mask-combine` is for."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["boolean-mask-combine", "ray-parametric-form"],
    "lo": (
        "Compose the ray parametric form (t >= 0 = in front of origin) with boolean mask combine "
        "(AND with the barycentric inside-triangle predicate) to express the full ray-triangle "
        "intersection test."
    ),
}


# ===========================================================================
# cx29 — combine "valid solve" with "non-singular det" via the mask-in trick
# ===========================================================================
spec_29 = {
    "atom_ids": ["boolean-mask-combine", "singular-matrix-mask-trick"],
    "subtopics": _subs(["boolean-mask-combine", "singular-matrix-mask-trick"]),
    "primary_atom": "boolean-mask-combine",
    "part": "part1",
    "exercise_index": 29,
    "exercise_title": "combine non-singular det mask with in-range solve mask",
    "slug": "combine-nonsingular-and-solve-survivors",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's `raytrace_triangle` has to solve `(B, 3, 3)` systems where SOME slices are exactly "
        "singular (parallel ray, degenerate triangle). `t.linalg.solve` raises if ANY slice is "
        "singular — so we use the **singular-matrix-mask trick**:\n"
        "1. Compute `dets = t.linalg.det(A)`.\n"
        "2. Build `is_singular = dets.abs() < eps` — boolean of shape `(B,)`.\n"
        "3. **Mask in** the identity matrix at singular slices: `A_safe = A.clone(); "
        "A_safe[is_singular] = t.eye(N)`. Now `solve` succeeds everywhere.\n"
        "4. Run the solve on `A_safe`. The values at singular slices are garbage.\n"
        "5. **Mask out** those garbage values with `valid = (~is_singular) & in_range`.\n\n"
        "The composition: this drill exercises both the **mask-in** half (overwrite singular A's "
        "with the identity) and the **mask-out** half (boolean-AND of `~is_singular` with the "
        "downstream `in_range` predicate).\n\n"
        "**Anatomy.**\n"
        "- `nonsingular = ~is_singular` — the per-slice gating mask.\n"
        "- `in_range = (x >= 0).all(dim=-1) & (x <= 1).all(dim=-1)` — barycentric range check.\n"
        "- `valid = nonsingular & in_range` — boolean-AND the two masks."
    ),
    "prompt_body": (
        "Implement `cx29_safe_solve(A, b, eps=1e-8)`.\n\n"
        "- `A`: float tensor of shape `(B, N, N)`. May include slices with `det(A) == 0` (parallel "
        "rays / degenerate triangles).\n"
        "- `b`: float tensor of shape `(B, N)`.\n"
        "- `eps`: tolerance for the singular test.\n\n"
        "Return `(x, valid)`:\n"
        "- `x`: solve result for the masked-in `A`, shape `(B, N)`. At singular slices the values "
        "are arbitrary (we mask them out, not zero them out — keeps the downstream code branch-free).\n"
        "- `valid`: boolean tensor of shape `(B,)`, True iff the slice is non-singular AND `x[i]` "
        "is entirely in `[0, 1]`.\n\n"
        "1. **Mask-in trick** — compute `is_singular = t.linalg.det(A).abs() < eps`, then overwrite "
        "the singular slices of a COPY of `A` with `t.eye(N)`. Do NOT mutate the caller's `A`.\n"
        "2. **Solve** the masked-in system.\n"
        "3. **Mask combine** — AND `~is_singular` with the in-range predicate."
    ),
    "stub_body": (
        "def cx29_safe_solve(A, b, eps=1e-8):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: clean 2x2 batch, all non-singular, all in-range.\n"
        "A = t.tensor([\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "    [[2.0, 0.0], [0.0, 2.0]],\n"
        "])\n"
        "b = t.tensor([[0.3, 0.4], [0.5, 0.5]])\n"
        "x, valid = cx29_safe_solve(A, b)\n"
        "assert tuple(x.shape) == (2, 2)\n"
        "assert tuple(valid.shape) == (2,)\n"
        "assert valid.dtype == t.bool\n"
        "assert valid.tolist() == [True, True]\n"
        "assert t.allclose(x, t.tensor([[0.3, 0.4], [0.25, 0.25]]), atol=1e-5)\n"
        "\n"
        "# Case B: one slice is EXACTLY singular (det == 0). The mask-in trick must prevent a crash.\n"
        "A = t.tensor([\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "    [[1.0, 1.0], [1.0, 1.0]],  # rank-1, det = 0.\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "])\n"
        "b = t.tensor([[0.2, 0.3], [99.0, 99.0], [0.5, 0.5]])\n"
        "# This MUST NOT raise. If it does, the singular-matrix-mask trick wasn't applied.\n"
        "x, valid = cx29_safe_solve(A, b)\n"
        "assert tuple(x.shape) == (3, 2)\n"
        "assert valid.tolist() == [True, False, True], f'got {valid.tolist()}'\n"
        "# At non-singular slices, x must equal b (identity A's).\n"
        "assert t.allclose(x[0], b[0])\n"
        "assert t.allclose(x[2], b[2])\n"
        "\n"
        "# Case C: caller's A must not be mutated by the mask-in step.\n"
        "A_orig = t.tensor([\n"
        "    [[1.0, 1.0], [1.0, 1.0]],  # singular.\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "])\n"
        "A = A_orig.clone()\n"
        "b = t.tensor([[1.0, 1.0], [0.5, 0.5]])\n"
        "x, valid = cx29_safe_solve(A, b)\n"
        "assert t.equal(A, A_orig), 'cx29 must not mutate the caller A — clone before masking-in'\n"
        "\n"
        "# Case D: out-of-range solve at a non-singular slice still gets masked out.\n"
        "A = t.tensor([\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "    [[1.0, 0.0], [0.0, 1.0]],\n"
        "])\n"
        "b = t.tensor([[0.3, 0.4], [2.5, -0.5]])\n"
        "x, valid = cx29_safe_solve(A, b)\n"
        "assert valid.tolist() == [True, False]"
    ),
    "solution_body": (
        "def cx29_safe_solve(A, b, eps=1e-8):\n"
        "    B, N, _ = A.shape\n"
        "    # Atom A (singular-matrix-mask-trick): detect singular slices and mask in the identity\n"
        "    # on a COPY so the caller's A is preserved.\n"
        "    dets = t.linalg.det(A)\n"
        "    is_singular = dets.abs() < eps\n"
        "    A_safe = A.clone()\n"
        "    A_safe[is_singular] = t.eye(N, dtype=A.dtype, device=A.device)\n"
        "    x = t.linalg.solve(A_safe, b)\n"
        "    # In-range predicate over the last axis.\n"
        "    in_range = (x >= 0).all(dim=-1) & (x <= 1).all(dim=-1)\n"
        "    # Atom B (boolean-mask-combine): AND the (~singular) mask with the in-range predicate.\n"
        "    valid = (~is_singular) & in_range\n"
        "    return x, valid"
    ),
    "solution_notes": (
        "The `A.clone()` is critical: without it, the caller's tensor gets a bunch of identity rows "
        "stamped over its singular slices — a silent state corruption bug. The `~is_singular & "
        "in_range` AND is the canonical ARENA shape: one mask says \"the math was valid\", the "
        "other says \"the result is physically meaningful\"."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["boolean-mask-combine", "singular-matrix-mask-trick"],
    "lo": (
        "Compose the singular-matrix mask-in trick (overwrite singular slices with identity so the "
        "batched solve does not crash) with boolean mask combine (AND of ~singular and in-range) to "
        "produce a per-slice survivor predicate."
    ),
}


# ===========================================================================
# cx30 — stack per-predicate masks along a new axis, then AND-reduce across it
# ===========================================================================
spec_30 = {
    "atom_ids": ["boolean-mask-combine", "stack-vs-cat"],
    "subtopics": _subs(["boolean-mask-combine", "stack-vs-cat"]),
    "primary_atom": "boolean-mask-combine",
    "part": "part1",
    "exercise_index": 30,
    "exercise_title": "stack per-predicate masks, then AND-reduce across the new axis",
    "slug": "stack-masks-then-combine",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "When you have **K predicates** that all live on the same shape `S`, two patterns get the "
        "joint AND:\n"
        "- **Pairwise chain** — `m0 & m1 & m2 & ...`. Fine for K=2, gets unwieldy at K=5.\n"
        "- **Stack-and-reduce** — `t.stack([m0, m1, m2, ...], dim=0).all(dim=0)`. Scales to any K, "
        "and the stacked tensor is itself a useful debug artifact (`(K, *S)` lets you see WHICH "
        "predicate vetoed which slice).\n\n"
        "The key choice is `stack` vs `cat`:\n"
        "- `t.stack(tensors, dim=0)` — INSERTS a new axis of length `K`. Output shape `(K, *S)`. "
        "Use this when each tensor is one \"slot\".\n"
        "- `t.cat(tensors, dim=0)` — CONCATENATES along an existing axis. Output shape "
        "`(K*S[0], *S[1:])`. Use this when you're growing an existing axis.\n\n"
        "For the K-mask AND, you want `stack` — you want a NEW axis to reduce over, not a longer "
        "version of the first axis.\n\n"
        "**Anatomy.**\n"
        "- `stacked = t.stack(masks, dim=0)`  → `(K, *S)` boolean tensor.\n"
        "- `combined = stacked.all(dim=0)`    → `(*S,)` boolean tensor — atom: boolean-mask-combine.\n"
        "\nReducing with `.all` is mathematically equivalent to chaining `&`, but the stack form "
        "is K-agnostic and lets you inspect the per-predicate breakdown for free."
    ),
    "prompt_body": (
        "Implement two functions.\n\n"
        "1. `cx30_stack_masks(masks)` — take a Python list of K boolean tensors, each shape `S`. "
        "Return a `(K, *S)` boolean tensor via `t.stack`. (Why `stack` and not `cat`: stack inserts "
        "a NEW axis of length K; cat would concatenate along an existing axis and lose the per-"
        "predicate identity.)\n"
        "2. `cx30_and_reduce(masks)` — same input. Return a single shape-`S` boolean tensor that is "
        "True iff every mask is True at that position. Implementation: stack with `cx30_stack_masks`, "
        "then `.all(dim=0)`.\n\n"
        "The two-step decomposition exercises both atoms cleanly: `stack` constructs the joint "
        "tensor, `boolean-mask-combine` (via `.all`) collapses it."
    ),
    "stub_body": (
        "def cx30_stack_masks(masks):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx30_and_reduce(masks):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: K=3, S=(4,). Stack must produce (3, 4); AND-reduce must produce (4,).\n"
        "m0 = t.tensor([True, True, True, False])\n"
        "m1 = t.tensor([True, True, False, True])\n"
        "m2 = t.tensor([True, False, True, True])\n"
        "stacked = cx30_stack_masks([m0, m1, m2])\n"
        "assert stacked.dtype == t.bool\n"
        "assert tuple(stacked.shape) == (3, 4), f'expected (3,4), got {tuple(stacked.shape)}'\n"
        "# Per-predicate identity must be preserved.\n"
        "assert t.equal(stacked[0], m0)\n"
        "assert t.equal(stacked[1], m1)\n"
        "assert t.equal(stacked[2], m2)\n"
        "combined = cx30_and_reduce([m0, m1, m2])\n"
        "assert combined.dtype == t.bool\n"
        "assert tuple(combined.shape) == (4,)\n"
        "assert combined.tolist() == [True, False, False, False]\n"
        "\n"
        "# Case B: 2-D shape, K=2. Verify stack yields (K, H, W).\n"
        "rng = t.Generator().manual_seed(30)\n"
        "H, W = 5, 6\n"
        "m0 = t.rand(H, W, generator=rng) > 0.3\n"
        "m1 = t.rand(H, W, generator=rng) > 0.3\n"
        "stacked = cx30_stack_masks([m0, m1])\n"
        "assert tuple(stacked.shape) == (2, H, W)\n"
        "combined = cx30_and_reduce([m0, m1])\n"
        "assert tuple(combined.shape) == (H, W)\n"
        "assert t.equal(combined, m0 & m1)\n"
        "\n"
        "# Case C: K=5 ARENA-shaped predicates over a (NR, NT)=(8, 7) grid — the canonical use case.\n"
        "rng = t.Generator().manual_seed(31)\n"
        "NR, NT = 8, 7\n"
        "masks = [t.rand(NR, NT, generator=rng) > 0.2 for _ in range(5)]\n"
        "stacked = cx30_stack_masks(masks)\n"
        "assert tuple(stacked.shape) == (5, NR, NT)\n"
        "combined = cx30_and_reduce(masks)\n"
        "assert tuple(combined.shape) == (NR, NT)\n"
        "# Cross-check against the pairwise chain.\n"
        "ref = masks[0]\n"
        "for m in masks[1:]:\n"
        "    ref = ref & m\n"
        "assert t.equal(combined, ref)\n"
        "\n"
        "# Case D: stack-vs-cat sanity — the result of stack on K shape-(4,) tensors has shape (K, 4),\n"
        "# NOT (K*4,). If a candidate accidentally used cat we'd see (12,) here.\n"
        "m0 = t.tensor([True, False, True, True])\n"
        "m1 = t.tensor([False, True, True, True])\n"
        "m2 = t.tensor([True, True, False, True])\n"
        "stacked = cx30_stack_masks([m0, m1, m2])\n"
        "assert tuple(stacked.shape) == (3, 4), (\n"
        "    f'expected (3, 4) from stack — did you accidentally use cat? Got {tuple(stacked.shape)}'\n"
        ")"
    ),
    "solution_body": (
        "def cx30_stack_masks(masks):\n"
        "    # Atom A (stack-vs-cat): t.stack INSERTS a new dim 0 of length K = len(masks).\n"
        "    # cat would have concatenated along an existing dim and lost the per-predicate axis.\n"
        "    return t.stack(masks, dim=0)\n"
        "\n"
        "def cx30_and_reduce(masks):\n"
        "    stacked = cx30_stack_masks(masks)\n"
        "    # Atom B (boolean-mask-combine): collapse the new K-axis with logical-AND.\n"
        "    # Equivalent to chaining `m0 & m1 & ... & m_{K-1}` but K-agnostic.\n"
        "    return stacked.all(dim=0)"
    ),
    "solution_notes": (
        "Why stack instead of cat: each mask is its own predicate — we want them on SEPARATE slots "
        "of a new axis so the per-predicate breakdown is recoverable (`stacked[i]` is predicate i). "
        "Cat would have concatenated along an existing axis and the per-predicate identity would be "
        "lost. The `.all(dim=0)` is the same operation as `m0 & m1 & ... & m_{K-1}` but K-agnostic "
        "and amenable to debugging — print `stacked.all(dim=(1, 2))` to see how many entries each "
        "predicate kept."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["boolean-mask-combine", "stack-vs-cat"],
    "lo": (
        "Compose torch.stack (insert a new K-axis to hold each predicate as its own slot) with "
        "boolean mask combine (collapse the new axis with .all) to AND together K shape-S masks "
        "in a K-agnostic, debuggable form."
    ),
}


SPECS = [spec_25, spec_26, spec_27, spec_28, spec_29, spec_30]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
