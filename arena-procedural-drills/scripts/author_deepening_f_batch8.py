#!/usr/bin/env python3
"""Batch 8 (deepening f) — author 1 new standalone Colab drill for each of 8 atoms.

Each new exercise targets a DISTINCT facet not covered by the existing notebooks
in that atom's folder. Subtopic key is copied verbatim from the corresponding
ex1 notebook's `delta_drills.subtopic` metadata field.

Atom → folder → new exercise index:
  ray-parametric-form              prereqs_geometry_cnn        ex3
  einsum-contraction               prereqs_einops_advanced     ex3
  einops-repeat-broadcast          prereqs_einops_advanced     ex3
  training-step-cycle              prereqs_training_loop       ex4
  contiguous-layout                prereqs_tensor_mechanics    ex4
  as-strided-windowing             prereqs_tensor_mechanics    ex4
  boolean-mask-identity-replace    prereqs_numpy               ex10
  broadcasting-rules               prereqs_numpy               ex10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# =====================================================================
# Atom recaps (kept short; flashcards carry full theory).
# =====================================================================

RECAP_RAY = (
    "## Ray parametric form — quick refresher\n"
    "\n"
    "A ray is `R(u) = O + u * D` where `O` is the origin and `D` is the "
    "direction. For `u >= 0` the equation traces the ray forward; `u < 0` "
    "is behind the origin. To **reflect** a ray off a plane with normal "
    "`n`, you keep the origin at the hit point and replace the direction "
    "with `D - 2 * (D · n) * n` (the component along `n` flips sign)."
)

RECAP_EINSUM = (
    "## einsum index contraction — quick refresher\n"
    "\n"
    "In `einsum('...->...', *tensors)`, an index that appears on the input "
    "side but NOT on the output side is **contracted** (summed). An index "
    "that appears on both sides is preserved as a free axis. Repeated "
    "inputs of the same letter across different operands cause that letter "
    "to be **matched then contracted** — this is how multi-tensor products "
    "(trilinear interp, factor models) collapse cleanly to a single line."
)

RECAP_EREPEAT = (
    "## einops.repeat as broadcast — quick refresher\n"
    "\n"
    "`einops.repeat(x, 'a b -> a n b', n=N)` inserts a NEW axis of length "
    "`N` *without* allocating `N` copies — internally the new axis is a "
    "stride-0 view. This lets you 'pair every X with every Y' (broadcast "
    "to `(NX, NY, ...)`) without quadratic memory. The downstream "
    "elementwise op materialises the result lazily."
)

RECAP_TRAIN = (
    "## Training-step cycle — quick refresher\n"
    "\n"
    "The canonical step is forward → loss → backward → step → zero_grad. "
    "`backward()` ACCUMULATES into `param.grad`, so failing to call "
    "`zero_grad()` makes consecutive batches add their grads. "
    "**Gradient accumulation** exploits that behaviour deliberately: you "
    "call `backward()` on `N` micro-batches and only fire `step()` + "
    "`zero_grad()` after the N-th, simulating an N×-larger effective batch."
)

RECAP_CONTIG = (
    "## Contiguous layout — quick refresher\n"
    "\n"
    "A tensor is contiguous when its strides match the row-major formula "
    "`stride[i] = prod(shape[i+1:])`. Operations like `.transpose()`, "
    "`.permute()`, and `.t()` produce *views* with rearranged strides — "
    "they are NOT contiguous, so `view()` will fail. Slicing along the "
    "outer dim keeps contiguity; slicing/striding an inner dim does not."
)

RECAP_STRIDED = (
    "## as_strided windowing — quick refresher\n"
    "\n"
    "`x.as_strided(size, stride)` returns a view into `x.storage()` with "
    "*manually specified* shape and stride tuples — no bounds checking. "
    "For a 2-D image `(H, W)`, a `KxK` patch grid is "
    "`size=(H-K+1, W-K+1, K, K)` and "
    "`stride=(W, 1, W, 1)` — outer strides walk the patch origin, inner "
    "strides walk inside one patch. This is the canonical Conv-input "
    "im2col primitive."
)

RECAP_BOOLMASK = (
    "## Boolean mask + identity replace — quick refresher\n"
    "\n"
    "A boolean mask `m` of the same shape (or broadcastable to) `x` "
    "selects elements where `m == True`. Indexed assignment "
    "`x[m] = value` replaces those positions. The 'identity replace' "
    "pattern: for a batch of matrices, replace the singular / NaN ones "
    "with the identity matrix so downstream `solve` / `inverse` calls "
    "succeed without short-circuiting."
)

RECAP_BCAST = (
    "## Broadcasting rules — quick refresher\n"
    "\n"
    "Two shapes broadcast right-aligned: from the trailing axis backward, "
    "each pair of dims must be equal, OR one of them must be 1, OR one of "
    "them must be missing. Multi-axis broadcasting (e.g. `(B, 1, T, T)` "
    "with `(B, H, T, T)`) is what lets a single padding mask apply to "
    "every attention head without copying."
)


# =====================================================================
# Spec list — one new exercise per atom, distinct facet.
# =====================================================================

SPECS = [
    # ----------------------------------------------------------------- ray ex3
    {
        "atom_id": "ray-parametric-form",
        "subtopic": "Geometry: Ray parametric form",
        "topic_folder": "prereqs_geometry_cnn",
        "atom_recap_md": RECAP_RAY,
        "exercise_index": 3,
        "exercise_title": "reflect rays off a ground plane and trace the bounce",
        "slug": "reflect-rays-off-a-ground-plane-and-trace-the-bounce",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["ray", "reflection", "two-segment", "ground-plane", "integrative"],
        "kcs": ["ray-eval-broadcast-batch", "ray-origin-direction-storage"],
        "lo": (
            "Compose the parametric ray equation with a planar-reflection "
            "update to trace a batch of rays through one bounce off the "
            "y=0 plane, returning both the hit-point batch and the "
            "post-bounce ray batch."
        ),
        "prompt_body": (
            "Existing ex1 / ex2 evaluate a single ray equation. This one "
            "*composes* the equation with a reflection update — a "
            "two-stage pipeline.\n\n"
            "Implement `ex3_bounce_off_ground(rays)`:\n\n"
            "1. `rays` has shape `(B, 2, 3)` — `rays[:, 0]` are origins "
            "`O`, `rays[:, 1]` are directions `D`. All `D` have `D.y < 0` "
            "(pointing down). All `O` have `O.y > 0` (above the ground).\n"
            "2. Solve for the hit parameter `u_hit = -O.y / D.y` (one "
            "scalar per ray).\n"
            "3. Hit point `H = O + u_hit * D` via the parametric form — "
            "every `H.y` must be ~0.\n"
            "4. Reflect `D` across the ground-plane normal `n = (0,1,0)`: "
            "`D_reflected = D - 2 * (D · n) * n`. Because `n=(0,1,0)`, "
            "this simply flips the y component.\n"
            "5. Return a new `(B, 2, 3)` `rays_after` tensor whose origin "
            "row is `H` and direction row is `D_reflected`.\n\n"
            "Output dtype must be `float32`. After the bounce, every "
            "`D_reflected.y` must be positive (heading up)."
        ),
        "stub": (
            "def ex3_bounce_off_ground(rays: Tensor) -> Tensor:\n"
            '    """Return (B,2,3) post-bounce rays after reflecting off y=0."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Two hand-checked rays.\n"
            "rays = t.tensor([\n"
            "    # ray 0 — origin (0, 2, 0), straight down → hits (0,0,0), reflects to (0,1,0)\n"
            "    [[0.0, 2.0, 0.0], [0.0, -1.0, 0.0]],\n"
            "    # ray 1 — origin (0, 3, 0), direction (1, -1, 0) → hits (3,0,0), reflects to (1,1,0)\n"
            "    [[0.0, 3.0, 0.0], [1.0, -1.0, 0.0]],\n"
            "])\n"
            "out = ex3_bounce_off_ground(rays)\n"
            "assert out.shape == (2, 2, 3), f'expected (2,2,3), got {tuple(out.shape)}'\n"
            "assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "expected = t.tensor([\n"
            "    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],\n"
            "    [[3.0, 0.0, 0.0], [1.0, 1.0, 0.0]],\n"
            "])\n"
            "assert t.allclose(out, expected, atol=1e-5), f'value mismatch:\\n{out}\\nvs\\n{expected}'\n"
            "# Every hit must be on the ground.\n"
            "assert t.allclose(out[:, 0, 1], t.zeros(2), atol=1e-5), 'hit-point y must be 0'\n"
            "# Every reflected direction must be heading up.\n"
            "assert (out[:, 1, 1] > 0).all(), 'reflected direction must have positive y'\n"
            "\n"
            "# Batch smoke test on a fan of rays + bounce visualization.\n"
            "rng = t.Generator().manual_seed(7)\n"
            "B = 40\n"
            "origins = t.stack([t.linspace(-3, 3, B), t.full((B,), 2.5), t.zeros(B)], dim=1)\n"
            "directions = t.stack([\n"
            "    0.4 * t.randn(B, generator=rng),\n"
            "    t.full((B,), -1.0),\n"
            "    t.zeros(B),\n"
            "], dim=1)\n"
            "big_rays = t.stack([origins, directions], dim=1)\n"
            "big_out = ex3_bounce_off_ground(big_rays)\n"
            "hits = big_out[:, 0]\n"
            "reflected = big_out[:, 1]\n"
            "assert t.allclose(hits[:, 1], t.zeros(B), atol=1e-5)\n"
            "assert (reflected[:, 1] > 0).all()\n"
            "\n"
            "# --- Bounce visualization (X-Y projection) ---\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "for i in range(B):\n"
            "    O = origins[i].numpy(); H = hits[i].numpy()\n"
            "    R = (hits[i] + 1.5 * reflected[i]).numpy()  # trace 1.5 units after bounce\n"
            "    ax.plot([O[0], H[0]], [O[1], H[1]], color='tab:blue', alpha=0.5, linewidth=0.7)\n"
            "    ax.plot([H[0], R[0]], [H[1], R[1]], color='tab:orange', alpha=0.5, linewidth=0.7)\n"
            "ax.axhline(0, color='k', linewidth=1)\n"
            "ax.set_xlabel('x'); ax.set_ylabel('y')\n"
            "ax.set_title(f'ex3 ray bounce off y=0  (blue=incoming, orange=reflected, B={B})')\n"
            "ax.set_aspect('equal')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex3_bounce_off_ground(rays: Tensor) -> Tensor:\n"
            "    O = rays[:, 0]\n"
            "    D = rays[:, 1]\n"
            "    u_hit = -O[:, 1] / D[:, 1]                     # (B,)\n"
            "    H = O + u_hit.unsqueeze(-1) * D                # (B, 3)\n"
            "    n = t.tensor([0.0, 1.0, 0.0])                  # ground normal\n"
            "    D_reflected = D - 2 * (D @ n).unsqueeze(-1) * n\n"
            "    return t.stack([H, D_reflected], dim=1).to(t.float32)"
        ),
        "solution_notes": (
            "**Two stages of the parametric form.** First we *consume* the "
            "ray equation to find the hit point (`H = O + u_hit * D`); "
            "then we *re-emit* a new ray rooted at the hit point with a "
            "transformed direction. The atom is the equation; the drill "
            "is the composition.\n\n"
            "**Why the formula `D - 2 (D·n) n` works.** Decompose `D` "
            "into `D_parallel + D_perp` where `D_parallel = (D·n) n` "
            "(along the normal) and `D_perp` is in the plane. Reflection "
            "negates the parallel component → `D - 2 (D·n) n`. For our "
            "axis-aligned normal `(0,1,0)`, this collapses to flipping "
            "the y component, but the general form is what real "
            "raytracers use for arbitrary plane normals.\n\n"
            "**`u_hit` shape gotcha.** `O[:,1]` and `D[:,1]` are both "
            "`(B,)`, so `u_hit` is `(B,)`. To multiply against `D` of "
            "shape `(B,3)` you need an `unsqueeze(-1)` — exactly the "
            "same broadcast pattern ex1 and ex2 used."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ----------------------------------------------------------------- einsum ex3
    {
        "atom_id": "einsum-contraction",
        "subtopic": "Einsum: Index contraction semantics",
        "topic_folder": "prereqs_einops_advanced",
        "atom_recap_md": RECAP_EINSUM,
        "exercise_index": 3,
        "exercise_title": "trilinear interpolation as a three-tensor einsum",
        "slug": "trilinear-interpolation-as-a-three-tensor-einsum",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["trilinear", "multi-tensor-einsum", "factorised-weights", "nerf-volumetric"],
        "kcs": ["einsum-repeated-index-sums", "einsum-batch-axis-passthrough"],
        "lo": (
            "Compose a three-operand einsum that contracts three separate "
            "1-D weight axes simultaneously against a 3-D voxel grid, "
            "implementing factorised trilinear interpolation in one line."
        ),
        "prompt_body": (
            "ex1 and ex2 covered single-axis contraction (`'bij,i->bj'`-"
            "style). This drill is a *three-operand* contraction — "
            "factorised trilinear interpolation, the kind used in "
            "tri-plane NeRFs and volumetric grids.\n\n"
            "Implement `ex3_trilinear(vol, wx, wy, wz)`:\n\n"
            "1. `vol` is a voxel grid `(D, H, W)` — values to interpolate.\n"
            "2. `wx`, `wy`, `wz` are 1-D weight vectors of length `W`, "
            "`H`, `D` respectively, each summing to 1.\n"
            "3. Compute the scalar `sum_{d,h,w} vol[d,h,w] * wz[d] * "
            "wy[h] * wx[w]` using a SINGLE `t.einsum` call.\n"
            "4. The einsum pattern should treat `d`, `h`, `w` as repeated "
            "indices contracted across `vol` and the three weight "
            "vectors.\n\n"
            "Hint: the pattern looks like `'dhw,w,h,d->'` (all four "
            "indices contract).\n\n"
            "Output: scalar `float32` tensor (shape `()`)."
        ),
        "stub": (
            "def ex3_trilinear(vol: Tensor, wx: Tensor, wy: Tensor, wz: Tensor) -> Tensor:\n"
            '    """Single-einsum trilinear contraction. Returns a 0-d float32 tensor."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-checked case: all weights uniform → result is the mean.\n"
            "vol = t.arange(2 * 3 * 4, dtype=t.float32).reshape(2, 3, 4)  # (D=2, H=3, W=4)\n"
            "wx = t.full((4,), 1/4)\n"
            "wy = t.full((3,), 1/3)\n"
            "wz = t.full((2,), 1/2)\n"
            "out = ex3_trilinear(vol, wx, wy, wz)\n"
            "assert out.shape == (), f'expected scalar, got {tuple(out.shape)}'\n"
            "assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "assert t.allclose(out, vol.mean()), f'uniform weights → mean, got {out.item()} vs {vol.mean().item()}'\n"
            "\n"
            "# Hand-checked case: one-hot weights pick a single voxel.\n"
            "wz = t.tensor([0.0, 1.0])  # pick d=1\n"
            "wy = t.tensor([0.0, 1.0, 0.0])  # pick h=1\n"
            "wx = t.tensor([0.0, 0.0, 1.0, 0.0])  # pick w=2\n"
            "picked = ex3_trilinear(vol, wx, wy, wz)\n"
            "assert t.allclose(picked, vol[1, 1, 2]), f'one-hot pick wrong: {picked.item()} vs {vol[1,1,2].item()}'\n"
            "\n"
            "# Linearity check: trilinear is linear in vol.\n"
            "rng = t.Generator().manual_seed(2)\n"
            "vol_a = t.randn(3, 4, 5, generator=rng)\n"
            "vol_b = t.randn(3, 4, 5, generator=rng)\n"
            "wx = t.softmax(t.randn(5, generator=rng), dim=0)\n"
            "wy = t.softmax(t.randn(4, generator=rng), dim=0)\n"
            "wz = t.softmax(t.randn(3, generator=rng), dim=0)\n"
            "lhs = ex3_trilinear(vol_a + 2.5 * vol_b, wx, wy, wz)\n"
            "rhs = ex3_trilinear(vol_a, wx, wy, wz) + 2.5 * ex3_trilinear(vol_b, wx, wy, wz)\n"
            "assert t.allclose(lhs, rhs, atol=1e-4), f'linearity broken: {lhs.item()} vs {rhs.item()}'\n"
            "\n"
            "# Equivalence vs an explicit triple loop (small grid only).\n"
            "vol_s = t.randn(2, 3, 2, generator=rng)\n"
            "wx_s = t.softmax(t.randn(2, generator=rng), dim=0)\n"
            "wy_s = t.softmax(t.randn(3, generator=rng), dim=0)\n"
            "wz_s = t.softmax(t.randn(2, generator=rng), dim=0)\n"
            "expected_loop = 0.0\n"
            "for d in range(2):\n"
            "    for h in range(3):\n"
            "        for w in range(2):\n"
            "            expected_loop += vol_s[d,h,w].item() * wz_s[d].item() * wy_s[h].item() * wx_s[w].item()\n"
            "got = ex3_trilinear(vol_s, wx_s, wy_s, wz_s)\n"
            "assert abs(got.item() - expected_loop) < 1e-4, f'einsum vs loop mismatch: {got.item()} vs {expected_loop}'\n"
            "\n"
            "# --- Visualization: scan a single weight axis, plot interp ---\n"
            "vol_v = t.linspace(0, 1, 16).reshape(2, 2, 4)\n"
            "n_steps = 9\n"
            "vals = []\n"
            "for alpha in t.linspace(0, 1, n_steps):\n"
            "    wx_v = t.tensor([alpha.item(), 0, 1 - alpha.item(), 0])\n"
            "    wy_v = t.tensor([0.5, 0.5])\n"
            "    wz_v = t.tensor([0.5, 0.5])\n"
            "    vals.append(ex3_trilinear(vol_v, wx_v, wy_v, wz_v).item())\n"
            "fig, ax = plt.subplots(figsize=(5, 3))\n"
            "ax.plot(t.linspace(0, 1, n_steps).numpy(), vals, marker='o')\n"
            "ax.set_xlabel('alpha (wx between voxel-0 and voxel-2)')\n"
            "ax.set_ylabel('interpolated value')\n"
            "ax.set_title('ex3 trilinear interp is linear in each weight')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex3_trilinear(vol: Tensor, wx: Tensor, wy: Tensor, wz: Tensor) -> Tensor:\n"
            "    return t.einsum('dhw,w,h,d->', vol, wx, wy, wz).to(t.float32)"
        ),
        "solution_notes": (
            "**Multi-operand contraction in one symbol per axis.** Each "
            "letter `d`, `h`, `w` appears in `vol` AND in one weight "
            "vector AND is absent from the output → it gets matched "
            "across operands then summed. einsum handles the "
            "broadcast-and-reduce in one shot; you don't need three "
            "separate `*` and `.sum()` calls.\n\n"
            "**Why this generalises to NeRFs.** Tri-plane and "
            "factorised-grid models store the volume as outer products "
            "of low-rank axis tensors. Sampling is exactly this "
            "contraction — `einsum` makes it 10× more readable than "
            "nested `unsqueeze` + `*` + `sum`.\n\n"
            "**`'dhw,w,h,d->'` vs `'dhw,d,h,w->'`.** Order of operands "
            "doesn't matter as long as each letter shows up in exactly "
            "one of the weight inputs. We listed `wx, wy, wz` to match "
            "spatial reading order; einsum reorders internally."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ----------------------------------------------------------------- erepeat ex3
    {
        "atom_id": "einops-repeat-broadcast",
        "subtopic": "Einops: Repeat-as-broadcast",
        "topic_folder": "prereqs_einops_advanced",
        "atom_recap_md": RECAP_EREPEAT,
        "exercise_index": 3,
        "exercise_title": "few-shot prototype broadcast for cosine-similarity classifier",
        "slug": "few-shot-prototype-broadcast-for-cosine-similarity-classifier",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["few-shot", "prototype", "cosine-similarity", "metric-learning", "integrative"],
        "kcs": ["repeat-inserts-zero-stride-axis", "repeat-pair-every-with-every"],
        "lo": (
            "Compose two `einops.repeat` broadcasts (queries vs prototypes) "
            "to produce an `(N, C)` cosine-similarity logit matrix without "
            "materialising the `(N, C, D)` intermediate."
        ),
        "prompt_body": (
            "ex1 paired rays × triangles. ex2 broadcast a per-token bias. "
            "This one is the *metric-learning* facet: pair every query "
            "with every class prototype to get a cosine-similarity logit "
            "matrix — the prototypical-net forward pass.\n\n"
            "Implement `ex3_proto_logits(queries, prototypes)`:\n\n"
            "1. `queries` has shape `(N, D)` — N query embeddings.\n"
            "2. `prototypes` has shape `(C, D)` — one mean-embedding per "
            "class.\n"
            "3. Use `einops.repeat` to expand:\n"
            "   - `queries`  → `(N, C, D)` with `'n d -> n c d'`, `c=C`\n"
            "   - `prototypes` → `(N, C, D)` with `'c d -> n c d'`, `n=N`\n"
            "4. Compute cosine similarity element-wise along `D`:\n"
            "   `cos(a, b) = (a · b) / (||a|| ||b||)`\n"
            "5. Return the `(N, C)` similarity matrix.\n\n"
            "Output dtype: `float32`. Values in `[-1, 1]`. The dot "
            "product collapses the `D` axis — the broadcast machinery "
            "supplies the `(N, C)` shape."
        ),
        "stub": (
            "def ex3_proto_logits(queries: Tensor, prototypes: Tensor) -> Tensor:\n"
            '    """Return (N, C) cosine-similarity logits via einops.repeat broadcast."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-checked case: orthonormal-ish prototypes.\n"
            "queries = t.tensor([\n"
            "    [1.0, 0.0, 0.0],   # exactly the x-axis\n"
            "    [0.0, 1.0, 0.0],   # exactly the y-axis\n"
            "    [0.7071, 0.7071, 0.0],  # 45° between x and y\n"
            "])\n"
            "prototypes = t.tensor([\n"
            "    [1.0, 0.0, 0.0],   # class 0: x-axis\n"
            "    [0.0, 1.0, 0.0],   # class 1: y-axis\n"
            "    [0.0, 0.0, 1.0],   # class 2: z-axis\n"
            "])\n"
            "logits = ex3_proto_logits(queries, prototypes)\n"
            "assert logits.shape == (3, 3), f'expected (3,3), got {tuple(logits.shape)}'\n"
            "assert logits.dtype == t.float32, f'expected float32, got {logits.dtype}'\n"
            "# Diagonal: query == prototype → cos = 1 for rows 0 and 1.\n"
            "assert abs(logits[0, 0].item() - 1.0) < 1e-4\n"
            "assert abs(logits[1, 1].item() - 1.0) < 1e-4\n"
            "# Off-diagonal of orthogonal pairs: 0.\n"
            "assert abs(logits[0, 1].item()) < 1e-4\n"
            "assert abs(logits[0, 2].item()) < 1e-4\n"
            "# 45° query gives 1/sqrt(2) to both class 0 and class 1, 0 to class 2.\n"
            "assert abs(logits[2, 0].item() - 0.7071) < 1e-3\n"
            "assert abs(logits[2, 1].item() - 0.7071) < 1e-3\n"
            "assert abs(logits[2, 2].item()) < 1e-3\n"
            "# Argmax-picks-correct on diagonal.\n"
            "assert logits.argmax(dim=1).tolist()[:2] == [0, 1]\n"
            "\n"
            "# Bounds check on random embeddings.\n"
            "rng = t.Generator().manual_seed(42)\n"
            "Q = t.randn(20, 64, generator=rng)\n"
            "P = t.randn(5, 64, generator=rng)\n"
            "L = ex3_proto_logits(Q, P)\n"
            "assert L.shape == (20, 5)\n"
            "assert L.min().item() >= -1.0 - 1e-5\n"
            "assert L.max().item() <=  1.0 + 1e-5\n"
            "\n"
            "# Symmetric: ex3(q, p)[i, j] == ex3(p, q)[j, i].\n"
            "L_t = ex3_proto_logits(P, Q)\n"
            "assert t.allclose(L, L_t.T, atol=1e-5)\n"
            "\n"
            "# --- Visualization: heatmap of class logits for a synthetic few-shot run ---\n"
            "n_classes = 4\n"
            "centres = t.randn(n_classes, 8, generator=rng)\n"
            "# 12 queries: 3 near each class centre + light noise.\n"
            "query_rows = []\n"
            "for c in range(n_classes):\n"
            "    for _ in range(3):\n"
            "        query_rows.append(centres[c] + 0.3 * t.randn(8, generator=rng))\n"
            "queries_v = t.stack(query_rows)  # (12, 8)\n"
            "L_v = ex3_proto_logits(queries_v, centres)\n"
            "fig, ax = plt.subplots(figsize=(4, 5))\n"
            "im = ax.imshow(L_v.numpy(), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')\n"
            "ax.set_xlabel('class prototype')\n"
            "ax.set_ylabel('query index')\n"
            "ax.set_title('ex3 cosine-similarity logits')\n"
            "plt.colorbar(im, ax=ax, label='cosine sim')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex3_proto_logits(queries: Tensor, prototypes: Tensor) -> Tensor:\n"
            "    N, D = queries.shape\n"
            "    C, _ = prototypes.shape\n"
            "    q = repeat(queries,    'n d -> n c d', c=C)   # (N, C, D), stride-0 along c\n"
            "    p = repeat(prototypes, 'c d -> n c d', n=N)   # (N, C, D), stride-0 along n\n"
            "    dots = (q * p).sum(dim=-1)                    # (N, C)\n"
            "    q_norm = q.norm(dim=-1)                       # (N, C)\n"
            "    p_norm = p.norm(dim=-1)                       # (N, C)\n"
            "    return (dots / (q_norm * p_norm + 1e-12)).to(t.float32)"
        ),
        "solution_notes": (
            "**Two `repeat`s collapse to one shape.** Each query needs a "
            "copy per class; each prototype needs a copy per query. "
            "`einops.repeat` inserts the missing axis as a stride-0 view, "
            "so the `(N, C, D)` intermediate is virtual — no `N*C*D` "
            "memory cost until the elementwise op fires.\n\n"
            "**Why we don't pre-normalise.** You CAN normalise `queries` "
            "and `prototypes` to unit length first and skip the division, "
            "and that's what real ProtoNet code does. Here we compute "
            "the full formula to make the broadcast pattern visible — "
            "the norms are taken AFTER the repeat, on the shared "
            "`(N, C, D)` shape, so they too rely on the broadcast.\n\n"
            "**Contrast with ex1.** ex1 used `repeat` to pair rays × "
            "triangles in a geometry context. This drill is the same "
            "broadcast pattern in a metric-learning context — the atom "
            "(zero-stride pairing) generalises across domains."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ----------------------------------------------------------------- training ex4
    {
        "atom_id": "training-step-cycle",
        "subtopic": "PyTorch: Training step cycle",
        "topic_folder": "prereqs_training_loop",
        "atom_recap_md": RECAP_TRAIN,
        "exercise_index": 4,
        "exercise_title": "modify the cycle for N-step gradient accumulation",
        "slug": "modify-the-cycle-for-n-step-gradient-accumulation",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["gradient-accumulation", "micro-batch", "effective-batch-size", "modify-the-cycle"],
        "kcs": ["training-step-five-call-order", "training-step-zero-grad-resets-accumulation"],
        "lo": (
            "Modify the canonical 5-call training step into an N-micro-"
            "batch gradient-accumulation variant that defers `step()` "
            "and `zero_grad()` until every Nth micro-batch."
        ),
        "prompt_body": (
            "ex1–ex3 explored the *correct order* and *ordering bugs* of "
            "the canonical 5-call cycle. This one *deliberately modifies* "
            "the cycle to exploit grad accumulation — the standard trick "
            "for fitting a large effective batch on a small device.\n\n"
            "Implement `ex4_accumulate(model, x_batches, y_batches, "
            "optimizer, loss_fn, accum_steps)`:\n\n"
            "1. `x_batches` and `y_batches` are length-`M` lists of micro-"
            "batch tensors, where `M % accum_steps == 0`.\n"
            "2. For each micro-batch `i`:\n"
            "   - forward + loss + backward (accumulate grads into "
            "`param.grad`)\n"
            "   - only call `optimizer.step()` + `optimizer.zero_grad()` "
            "when `(i + 1) % accum_steps == 0`.\n"
            "3. Divide each micro-batch loss by `accum_steps` BEFORE "
            "backward so the accumulated grad equals the average of the "
            "micro-batch grads (same magnitude as a single big batch).\n"
            "4. Return a list of the loss values *before* the division "
            "(one per micro-batch), so the caller can plot loss per "
            "micro-step.\n\n"
            "The test verifies that the parameter trajectory after "
            "`accum_steps=2` on `M=4` micro-batches matches a single-"
            "batch baseline run on the concatenated data — i.e. that "
            "the modification is mathematically equivalent."
        ),
        "stub": (
            "def ex4_accumulate(model, x_batches, y_batches, optimizer,\n"
            "                   loss_fn, accum_steps: int) -> list:\n"
            '    """Gradient-accumulated training loop. Returns per-micro-batch losses."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "def make_model():\n"
            "    rng = t.Generator().manual_seed(123)\n"
            "    m = nn.Linear(3, 1, bias=False)\n"
            "    with t.no_grad():\n"
            "        m.weight.copy_(t.randn(1, 3, generator=rng))\n"
            "    return m\n"
            "\n"
            "rng = t.Generator().manual_seed(0)\n"
            "X = t.randn(8, 3, generator=rng)\n"
            "y = t.randn(8, 1, generator=rng)\n"
            "x_micro = list(X.split(2))   # 4 micro-batches of size 2\n"
            "y_micro = list(y.split(2))\n"
            "\n"
            "# Baseline: ONE big step on the concatenated data, lr / accum.\n"
            "baseline = make_model()\n"
            "opt_b = t.optim.SGD(baseline.parameters(), lr=0.1)\n"
            "loss_fn = nn.MSELoss()\n"
            "opt_b.zero_grad()\n"
            "loss_b = loss_fn(baseline(X), y)\n"
            "loss_b.backward()\n"
            "opt_b.step()\n"
            "ref_w = baseline.weight.detach().clone()\n"
            "\n"
            "# Test: 4-micro-batch run with accum_steps=4.\n"
            "model = make_model()\n"
            "opt = t.optim.SGD(model.parameters(), lr=0.1)\n"
            "# Pre-zero so the first backward starts clean (typical usage).\n"
            "opt.zero_grad()\n"
            "losses = ex4_accumulate(model, x_micro, y_micro, opt, loss_fn, accum_steps=4)\n"
            "assert isinstance(losses, list), f'expected list, got {type(losses).__name__}'\n"
            "assert len(losses) == 4, f'expected 4 losses, got {len(losses)}'\n"
            "got_w = model.weight.detach().clone()\n"
            "assert t.allclose(got_w, ref_w, atol=1e-5), (\n"
            "    f'parameter mismatch after accumulation:\\n'\n"
            "    f'  got:      {got_w}\\n'\n"
            "    f'  baseline: {ref_w}\\n'\n"
            "    'gradient accumulation must equal a single big-batch step.'\n"
            ")\n"
            "\n"
            "# Sanity: accum_steps=1 == standard per-micro-batch SGD trajectory.\n"
            "rng2 = t.Generator().manual_seed(0)\n"
            "X2 = t.randn(6, 3, generator=rng2)\n"
            "y2 = t.randn(6, 1, generator=rng2)\n"
            "x_m2 = list(X2.split(2)); y_m2 = list(y2.split(2))\n"
            "model_a = make_model(); opt_a = t.optim.SGD(model_a.parameters(), lr=0.05)\n"
            "opt_a.zero_grad()\n"
            "ex4_accumulate(model_a, x_m2, y_m2, opt_a, loss_fn, accum_steps=1)\n"
            "model_b = make_model(); opt_b2 = t.optim.SGD(model_b.parameters(), lr=0.05)\n"
            "for xi, yi in zip(x_m2, y_m2):\n"
            "    opt_b2.zero_grad()\n"
            "    L = loss_fn(model_b(xi), yi)\n"
            "    L.backward()\n"
            "    opt_b2.step()\n"
            "assert t.allclose(model_a.weight, model_b.weight, atol=1e-5), 'accum_steps=1 must match per-batch SGD'\n"
            "\n"
            "# --- Visualization: loss curve over micro-batches (real run) ---\n"
            "rng3 = t.Generator().manual_seed(5)\n"
            "n_micro = 16\n"
            "Xv = t.randn(n_micro * 4, 3, generator=rng3)\n"
            "true_w = t.tensor([[1.0, -2.0, 0.5]])\n"
            "yv = Xv @ true_w.T + 0.1 * t.randn(n_micro * 4, 1, generator=rng3)\n"
            "model_v = nn.Linear(3, 1, bias=False)\n"
            "opt_v = t.optim.SGD(model_v.parameters(), lr=0.05)\n"
            "opt_v.zero_grad()\n"
            "loss_curve = ex4_accumulate(model_v, list(Xv.split(4)), list(yv.split(4)),\n"
            "                            opt_v, loss_fn, accum_steps=4)\n"
            "fig, ax = plt.subplots(figsize=(6, 3))\n"
            "ax.plot(loss_curve, marker='o')\n"
            "for k in range(4, n_micro + 1, 4):\n"
            "    ax.axvline(k - 1, color='red', alpha=0.3, linestyle='--')\n"
            "ax.set_xlabel('micro-batch index')\n"
            "ax.set_ylabel('MSE loss (pre-division)')\n"
            "ax.set_title('ex4 grad-accum: red dashes = optimizer step boundaries')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex4_accumulate(model, x_batches, y_batches, optimizer,\n"
            "                   loss_fn, accum_steps: int) -> list:\n"
            "    losses = []\n"
            "    for i, (xb, yb) in enumerate(zip(x_batches, y_batches)):\n"
            "        # 1. forward\n"
            "        pred = model(xb)\n"
            "        # 2. loss\n"
            "        loss = loss_fn(pred, yb)\n"
            "        losses.append(loss.item())\n"
            "        # 3. backward (scaled so accumulated grad == averaged grad)\n"
            "        (loss / accum_steps).backward()\n"
            "        # 4. step + 5. zero_grad — only on the Nth micro-batch\n"
            "        if (i + 1) % accum_steps == 0:\n"
            "            optimizer.step()\n"
            "            optimizer.zero_grad()\n"
            "    return losses"
        ),
        "solution_notes": (
            "**Why divide loss by `accum_steps`.** `MSELoss` already "
            "averages WITHIN one micro-batch, so accumulating N "
            "micro-batches without rescaling gives a grad that's `N×` "
            "too big. Dividing the loss by `N` before each backward "
            "makes the accumulated grad equal to the grad you'd get "
            "from one big batch of size `N × B_micro`.\n\n"
            "**Why the test pre-zeroes once outside the helper.** The "
            "helper only zeroes AFTER the Nth backward. Without an "
            "initial `zero_grad()` the very first backward would "
            "accumulate into whatever was already in `param.grad` (zero "
            "for a fresh model, but garbage if you reuse the model "
            "across calls).\n\n"
            "**Step-before-backward bugs (ex3) get sneakier here.** "
            "With accumulation, calling `step()` every micro-batch (not "
            "every Nth) trains correctly but with `accum_steps×` "
            "effective LR — easy to miss because the loss still goes "
            "down. The mathematical-equivalence assertion catches it."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ----------------------------------------------------------------- contig ex4
    {
        "atom_id": "contiguous-layout",
        "subtopic": "PyTorch: Contiguous layout",
        "topic_folder": "prereqs_tensor_mechanics",
        "atom_recap_md": RECAP_CONTIG,
        "exercise_index": 4,
        "exercise_title": "predict which slice operations break contiguity",
        "slug": "predict-which-slice-operations-break-contiguity",
        "bloom_level": "Evaluate",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["slicing", "view-breakage", "contig-prediction", "outer-vs-inner-axis"],
        "kcs": ["contiguous-stride-formula", "is-contiguous-check"],
        "lo": (
            "Evaluate a list of slice expressions on a contiguous base "
            "tensor and predict which ones return a still-contiguous "
            "view (outer-axis slices, full-axis slices) vs which break "
            "it (strided inner-axis slices, transpose-then-slice)."
        ),
        "prompt_body": (
            "ex1 derived strides of contiguous tensors. ex3 classified "
            "contiguity from a `(shape, stride)` tuple. This drill "
            "*evaluates* a concrete slice expression and predicts its "
            "contiguity *without* executing it on a real tensor — the "
            "test then runs the slice and confirms.\n\n"
            "Implement `ex4_predict_contiguous(shape, slice_spec)`:\n\n"
            "1. `shape` is a tuple of ints — the base tensor is a "
            "contiguous `t.arange(prod(shape)).reshape(shape)`.\n"
            "2. `slice_spec` is a tuple of Python `slice` objects, one "
            "per dim. Each is `slice(start, stop, step)` where any "
            "field may be `None`.\n"
            "3. Predict whether the resulting view is contiguous and "
            "return `True` or `False`.\n\n"
            "Rules to encode:\n"
            "- If the slice on dim `i` has `step != 1`, contiguity "
            "breaks (unless it shrinks the dim to length 1 or 0).\n"
            "- All inner dims (dim > leading-trimmed prefix) must be "
            "full-length slices (`slice(None, None, None)`).\n"
            "- The leading prefix may have arbitrary `start:stop` with "
            "`step=1`, since outer-axis slicing keeps contig.\n\n"
            "Edge cases: any axis sliced to length 0 or 1 effectively "
            "removes its contribution — `t.is_contiguous()` returns "
            "True. You may simply trust `t.is_contiguous()` on the "
            "concrete tensor as the ground truth for those edge cases "
            "(see the solution)."
        ),
        "stub": (
            "def ex4_predict_contiguous(shape: tuple, slice_spec: tuple) -> bool:\n"
            '    """Return True iff applying slice_spec to a contiguous arange(*shape) yields a contiguous view."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "def _ground_truth(shape, spec):\n"
            "    base = t.arange(math.prod(shape)).reshape(shape)\n"
            "    return base[spec].is_contiguous()\n"
            "\n"
            "CASES = [\n"
            "    # (shape, slice_spec, label)\n"
            "    ((4, 5),    (slice(0, 2),         slice(None)),       'outer prefix → contig'),\n"
            "    ((4, 5),    (slice(None),         slice(0, 3)),       'inner partial → NOT contig'),\n"
            "    ((4, 5),    (slice(None),         slice(None, None, 2)),  'inner step!=1 → NOT contig'),\n"
            "    ((4, 5),    (slice(None, None, 2),slice(None)),       'outer step!=1 → NOT contig'),\n"
            "    ((3, 4, 5), (slice(1, 3),         slice(None),        slice(None)), 'leading partial → contig'),\n"
            "    ((3, 4, 5), (slice(None),         slice(0, 2),        slice(None)), 'middle partial → NOT contig'),\n"
            "    ((3, 4, 5), (slice(None),         slice(None),        slice(0, 4)), 'inner partial → NOT contig'),\n"
            "    ((3, 4, 5), (slice(None),         slice(None),        slice(None)), 'full slice → contig'),\n"
            "    ((3, 4, 5), (slice(0, 1),         slice(None),        slice(None)), 'leading len-1 → contig'),\n"
            "]\n"
            "for shape, spec, label in CASES:\n"
            "    pred = ex4_predict_contiguous(shape, spec)\n"
            "    truth = _ground_truth(shape, spec)\n"
            "    assert pred == truth, f'mismatch on `{label}`: predicted {pred}, truth {truth}'\n"
            "\n"
            "# Property fuzz on a 4-D shape.\n"
            "rng = t.Generator().manual_seed(1)\n"
            "shape4 = (2, 3, 4, 5)\n"
            "for _ in range(30):\n"
            "    spec = []\n"
            "    for d in shape4:\n"
            "        choice = t.randint(0, 4, (1,), generator=rng).item()\n"
            "        if choice == 0:\n"
            "            spec.append(slice(None))\n"
            "        elif choice == 1:\n"
            "            spec.append(slice(None, None, 2))\n"
            "        elif choice == 2:\n"
            "            spec.append(slice(0, max(d - 1, 1)))\n"
            "        else:\n"
            "            spec.append(slice(1, d))\n"
            "    spec = tuple(spec)\n"
            "    assert ex4_predict_contiguous(shape4, spec) == _ground_truth(shape4, spec), (\n"
            "        f'fuzz fail on spec={spec}'\n"
            "    )\n"
            "\n"
            "# --- Visualization: small grid of contig vs non-contig slices ---\n"
            "grid_specs = [\n"
            "    (slice(None), slice(None)),\n"
            "    (slice(0, 2), slice(None)),\n"
            "    (slice(None), slice(0, 3)),\n"
            "    (slice(None, None, 2), slice(None)),\n"
            "]\n"
            "preds = [ex4_predict_contiguous((4, 5), s) for s in grid_specs]\n"
            "fig, axes = plt.subplots(1, len(grid_specs), figsize=(11, 2.5))\n"
            "base = t.arange(20).reshape(4, 5)\n"
            "for ax, spec, p in zip(axes, grid_specs, preds):\n"
            "    sliced = base[spec]\n"
            "    ax.imshow(sliced.numpy(), cmap='Blues')\n"
            "    edge = 'green' if p else 'red'\n"
            "    for s in ax.spines.values():\n"
            "        s.set_edgecolor(edge); s.set_linewidth(3)\n"
            "    ax.set_title(f'{spec}\\n→ contig={p}')\n"
            "    ax.set_xticks([]); ax.set_yticks([])\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex4_predict_contiguous(shape: tuple, slice_spec: tuple) -> bool:\n"
            "    # Row-major contig invariant after slicing:\n"
            "    #   For each dim i with result-length > 1, EVERY dim j > i must be\n"
            "    #   full-extent with step 1 (or trivial). Step != 1 on any length>1 dim\n"
            "    #   also breaks contig.\n"
            "    info = []\n"
            "    for d, s in zip(shape, slice_spec):\n"
            "        start, stop, step = s.indices(d)\n"
            "        if step > 0:\n"
            "            length = max(0, (stop - start + step - 1) // step)\n"
            "        else:\n"
            "            length = max(0, (start - stop - step - 1) // (-step))\n"
            "        full = (start == 0 and stop == d and step == 1)\n"
            "        info.append((length, full, step))\n"
            "    # Step != 1 on a non-trivial dim → never contig.\n"
            "    for length, full, step in info:\n"
            "        if step != 1 and length > 1:\n"
            "            return False\n"
            "    # Collect the indices of non-trivial, non-full dims.\n"
            "    nontriv_nonfull = [i for i, (length, full, _) in enumerate(info)\n"
            "                       if length > 1 and not full]\n"
            "    if len(nontriv_nonfull) > 1:\n"
            "        return False\n"
            "    if len(nontriv_nonfull) == 1:\n"
            "        k = nontriv_nonfull[0]\n"
            "        for j in range(k):\n"
            "            # Any earlier dim with length > 1 (full or not) wrecks contig:\n"
            "            # its stride won't match the shrunken inner product.\n"
            "            if info[j][0] > 1:\n"
            "                return False\n"
            "    return True"
        ),
        "solution_notes": (
            "**Outer partial = OK, inner partial = breaks.** Row-major "
            "storage means slicing the leading axis is just a pointer "
            "bump — the underlying bytes for the chosen rows are still "
            "contiguous. Slicing an INNER axis leaves gaps in the "
            "underlying buffer (you skip elements that belong to the "
            "rows you didn't drop), so strides no longer match the "
            "row-major formula.\n\n"
            "**`step != 1` always breaks** (unless the resulting axis "
            "has length ≤ 1, in which case the dim is degenerate and "
            "PyTorch reports it as contiguous trivially).\n\n"
            "**Why this is a one-line lookup in real code:** "
            "`tensor.is_contiguous()`. But predicting it from the "
            "slice spec alone — without instantiating anything — is "
            "the skill you need for **shape-checking decorators**, "
            "torch.fx tracing, and writing fused kernels."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ----------------------------------------------------------------- as_strided ex4
    {
        "atom_id": "as-strided-windowing",
        "subtopic": "PyTorch: as_strided windowing",
        "topic_folder": "prereqs_tensor_mechanics",
        "atom_recap_md": RECAP_STRIDED,
        "exercise_index": 4,
        "exercise_title": "2-D image patch grid via as_strided (im2col primitive)",
        "slug": "2d-image-patch-grid-via-as-strided",
        "bloom_level": "Apply",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["im2col", "patch-extraction", "2d-windowing", "conv-prep"],
        "kcs": ["as-strided-window-size", "as-strided-window-stride"],
        "lo": (
            "Apply `t.as_strided` along TWO spatial axes simultaneously "
            "to extract every KxK patch from an `(H, W)` image into an "
            "`(H-K+1, W-K+1, K, K)` view, the standard im2col primitive "
            "behind conv layers."
        ),
        "prompt_body": (
            "ex1–ex3 covered 1-D sliding windows (compute size/stride, "
            "channelled extension, step>1). This one extends to a TRUE "
            "2-D window pass — the im2col primitive behind `Conv2d`.\n\n"
            "Implement `ex4_image_patches(img, K)`:\n\n"
            "1. `img` has shape `(H, W)` — a single-channel image, "
            "contiguous.\n"
            "2. Extract all `KxK` patches (stride 1 in both spatial dims) "
            "as a single view of shape `(H-K+1, W-K+1, K, K)`.\n"
            "3. Use ONE `t.as_strided` call. The size tuple is "
            "`(H-K+1, W-K+1, K, K)`. The stride tuple — based on the "
            "fact that `img.stride()` is `(W, 1)` for a contiguous "
            "image — should be `(W, 1, W, 1)`:\n"
            "   - outer two strides step the patch *origin* across the "
            "image one row / one column at a time\n"
            "   - inner two strides step *within* a patch one row / one "
            "column at a time\n\n"
            "4. Return the `(H-K+1, W-K+1, K, K)` view.\n\n"
            "The test verifies one hand-picked patch, the global "
            "min/max, and a flatten-then-matmul use as a depthwise "
            "convolution sanity check."
        ),
        "stub": (
            "def ex4_image_patches(img: Tensor, K: int) -> Tensor:\n"
            '    """Return (H-K+1, W-K+1, K, K) view of all KxK patches."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "img = t.arange(20, dtype=t.float32).reshape(4, 5)\n"
            "patches = ex4_image_patches(img, K=2)\n"
            "assert patches.shape == (3, 4, 2, 2), f'expected (3,4,2,2), got {tuple(patches.shape)}'\n"
            "assert patches.dtype == t.float32\n"
            "# Hand-checked patch [0, 0]: the top-left 2x2 corner.\n"
            "expected_00 = t.tensor([[0.0, 1.0], [5.0, 6.0]])\n"
            "assert t.allclose(patches[0, 0], expected_00), f'top-left patch wrong:\\n{patches[0,0]}'\n"
            "# Hand-checked patch [2, 3]: the bottom-right 2x2 corner.\n"
            "expected_23 = t.tensor([[13.0, 14.0], [18.0, 19.0]])\n"
            "assert t.allclose(patches[2, 3], expected_23), f'bottom-right patch wrong:\\n{patches[2,3]}'\n"
            "\n"
            "# Patch [1, 2]: starts at img[1, 2].\n"
            "expected_12 = t.tensor([[7.0, 8.0], [12.0, 13.0]])\n"
            "assert t.allclose(patches[1, 2], expected_12)\n"
            "\n"
            "# Larger K.\n"
            "img2 = t.arange(36, dtype=t.float32).reshape(6, 6)\n"
            "patches2 = ex4_image_patches(img2, K=3)\n"
            "assert patches2.shape == (4, 4, 3, 3)\n"
            "# Center-most patch at [1, 1] starts at img2[1, 1].\n"
            "assert t.allclose(patches2[1, 1, 0], img2[1, 1:4])\n"
            "assert t.allclose(patches2[1, 1, 2], img2[3, 1:4])\n"
            "\n"
            "# Storage-sharing sanity: it's a view, not a copy.\n"
            "img3 = t.arange(16, dtype=t.float32).reshape(4, 4).clone()\n"
            "p3 = ex4_image_patches(img3, K=2)\n"
            "img3[0, 0] = -999.0\n"
            "assert p3[0, 0, 0, 0].item() == -999.0, 'expected a view, got a copy'\n"
            "\n"
            "# Use as a depthwise sum-conv: sum over each patch should equal a hand-rolled box filter.\n"
            "rng = t.Generator().manual_seed(11)\n"
            "imrand = t.randn(8, 8, generator=rng)\n"
            "P = ex4_image_patches(imrand, K=3)\n"
            "sums = P.sum(dim=(-2, -1))\n"
            "# Compare to nested Python loop.\n"
            "expected_sums = t.empty(6, 6)\n"
            "for i in range(6):\n"
            "    for j in range(6):\n"
            "        expected_sums[i, j] = imrand[i:i+3, j:j+3].sum()\n"
            "assert t.allclose(sums, expected_sums, atol=1e-5), 'patch sums must match loop reference'\n"
            "\n"
            "# --- Visualization: box-blur the image via patch-mean ---\n"
            "H_v, W_v = 32, 32\n"
            "ys = t.linspace(-1, 1, H_v).unsqueeze(1).expand(H_v, W_v)\n"
            "xs = t.linspace(-1, 1, W_v).unsqueeze(0).expand(H_v, W_v)\n"
            "img_v = t.exp(-(xs ** 2 + ys ** 2) / 0.05) + 0.3 * t.randn(H_v, W_v, generator=rng)\n"
            "blurred = ex4_image_patches(img_v, K=5).mean(dim=(-2, -1))\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))\n"
            "ax1.imshow(img_v.numpy(), cmap='magma'); ax1.set_title('input (noisy)')\n"
            "ax2.imshow(blurred.numpy(), cmap='magma'); ax2.set_title('ex4 5x5 box blur via patches')\n"
            "for a in (ax1, ax2): a.axis('off')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex4_image_patches(img: Tensor, K: int) -> Tensor:\n"
            "    H, W = img.shape\n"
            "    out_H, out_W = H - K + 1, W - K + 1\n"
            "    sH, sW = img.stride()                              # (W, 1) for contiguous\n"
            "    return img.as_strided(\n"
            "        size=(out_H, out_W, K, K),\n"
            "        stride=(sH, sW, sH, sW),\n"
            "    )"
        ),
        "solution_notes": (
            "**The stride pattern is *the same tuple twice*.** Outer "
            "two axes step the patch origin (`sH`, `sW`); inner two "
            "axes step inside a patch (`sH`, `sW`). They use the same "
            "underlying strides because both motions live in the same "
            "2-D storage. This is why `as_strided` shines: a 1-line "
            "view that replaces a 4-deep nested loop.\n\n"
            "**Why this is the im2col primitive.** A 2-D convolution "
            "is `patches @ kernel.flatten()` after this view + a "
            "`.reshape(out_H * out_W, K * K)`. Real Conv2d uses a "
            "fused kernel, but `Conv2d.forward` is morally this "
            "expression. ARENA's `conv2d_minimal` builds exactly this "
            "view; you just wrote it.\n\n"
            "**Out-of-bounds reads are the silent killer.** `as_strided` "
            "does not check that your stride tuple stays inside "
            "`img.storage()` — if you put `K > min(H, W)` here you "
            "get a corrupt view that reads past the buffer end. Always "
            "guard with `K <= min(H, W)` in production."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ----------------------------------------------------------------- boolmask ex10
    {
        "atom_id": "boolean-mask-identity-replace",
        "subtopic": "Numpy: Indexing and selection",
        "topic_folder": "prereqs_numpy",
        "atom_recap_md": RECAP_BOOLMASK,
        "exercise_index": 10,
        "exercise_title": "stabilize a normalizing-flow Jacobian batch with identity replace",
        "slug": "stabilize-a-normalizing-flow-jacobian-batch-with-identity-replace",
        "bloom_level": "Create",
        "difficulty_num": 5,
        "difficulty_dots": "🔴🔴🔴🔴🔴",
        "keywords": ["normalizing-flow", "jacobian", "log-det", "numerical-stability", "integrative", "ml-adjacent"],
        "kcs": ["mask-from-condition", "mask-update-in-a-loop"],
        "lo": (
            "Compose three boolean masks (NaN, near-singular, sign-flip) "
            "into a single 'bad-Jacobian' indicator and substitute the "
            "identity matrix into the bad slots so a downstream log-"
            "determinant computation stays finite."
        ),
        "prompt_body": (
            "Existing ex1–ex9 covered mask-from-comparison, masked "
            "assignment, identity substitution, NMS, attention masks, "
            "padding-mean, outlier removal. This drill is the "
            "*normalizing-flow* facet: a single end-to-end pipeline "
            "that combines THREE bad-condition masks (NaN, near-"
            "singular det, sign-flipped det) and identity-replaces the "
            "bad Jacobians so `logdet` stays finite.\n\n"
            "Implement `ex10_stabilize_jacobian_logdet(jacobians, eps)`:\n\n"
            "1. `jacobians` has shape `(B, D, D)` — one Jacobian per "
            "batch element. Some may contain NaN, may be near-singular "
            "(`|det| < eps`), or may have flipped sign "
            "(`det < 0` indicates an orientation reversal we don't "
            "trust for this flow).\n"
            "2. Build three masks of shape `(B,)`:\n"
            "   - `nan_mask`: any NaN in the matrix\n"
            "   - `singular_mask`: `|det| < eps`\n"
            "   - `flip_mask`: `det < 0`\n"
            "3. Combine via `bad = nan_mask | singular_mask | flip_mask`.\n"
            "4. Replace the bad Jacobians with the `(D, D)` identity "
            "matrix (det = 1, logdet = 0).\n"
            "5. Return `(stabilized, logdet, bad)` — the cleaned "
            "`(B, D, D)` tensor, the `(B,)` `logdet`, and the `(B,)` "
            "bool mask of slots that were replaced.\n\n"
            "Output dtypes: `stabilized` float32, `logdet` float32, "
            "`bad` bool. The bad slots must have `logdet == 0`."
        ),
        "stub": (
            "def ex10_stabilize_jacobian_logdet(jacobians: Tensor, eps: float = 1e-6):\n"
            '    """Return (stabilized, logdet, bad). Replaces bad Jacobians with I."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Construct a controlled batch.\n"
            "good = t.tensor([[2.0, 0.0], [0.0, 3.0]])              # det=6\n"
            "near_singular = t.tensor([[1e-9, 0.0], [0.0, 1.0]])    # det≈0\n"
            "sign_flip = t.tensor([[0.0, 1.0], [1.0, 0.0]])         # det=-1\n"
            "nan_mat = t.tensor([[float('nan'), 0.0], [0.0, 1.0]])\n"
            "rot = t.tensor([[0.6, -0.8], [0.8,  0.6]])              # det=1\n"
            "jacs = t.stack([good, near_singular, sign_flip, nan_mat, rot]).to(t.float32)\n"
            "\n"
            "stabilized, logdet, bad = ex10_stabilize_jacobian_logdet(jacs, eps=1e-4)\n"
            "assert stabilized.shape == jacs.shape\n"
            "assert stabilized.dtype == t.float32\n"
            "assert logdet.shape == (5,)\n"
            "assert bad.dtype == t.bool\n"
            "assert bad.shape == (5,)\n"
            "# Slots 1, 2, 3 should be flagged bad; slots 0 and 4 are clean.\n"
            "assert bad.tolist() == [False, True, True, True, False], f'bad mask wrong: {bad.tolist()}'\n"
            "# Bad slots got identity (det=1, logdet=0).\n"
            "assert t.allclose(stabilized[1], t.eye(2))\n"
            "assert t.allclose(stabilized[2], t.eye(2))\n"
            "assert t.allclose(stabilized[3], t.eye(2))\n"
            "assert abs(logdet[1].item()) < 1e-5\n"
            "assert abs(logdet[2].item()) < 1e-5\n"
            "assert abs(logdet[3].item()) < 1e-5\n"
            "# Good slots are untouched.\n"
            "assert t.allclose(stabilized[0], good)\n"
            "assert t.allclose(stabilized[4], rot)\n"
            "# Good logdets are correct.\n"
            "import math\n"
            "assert abs(logdet[0].item() - math.log(6.0)) < 1e-4\n"
            "assert abs(logdet[4].item() - 0.0) < 1e-3   # det(rot)=1, logdet=0\n"
            "# No NaN anywhere in logdet.\n"
            "assert t.isfinite(logdet).all(), f'logdet contains non-finite: {logdet}'\n"
            "\n"
            "# Scale up: 200 random Jacobians, inject 10% bad.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "B = 200\n"
            "big = t.eye(3).unsqueeze(0).expand(B, 3, 3).clone()\n"
            "noise = 0.3 * t.randn(B, 3, 3, generator=rng)\n"
            "big = big + noise\n"
            "bad_idx = t.randperm(B, generator=rng)[:20]\n"
            "big[bad_idx[:7]] = float('nan')             # NaN injection\n"
            "big[bad_idx[7:14]] = 1e-12                   # near-singular\n"
            "big[bad_idx[14:]] = -big[bad_idx[14:]]       # flip sign by negation (det flips sign for odd D)\n"
            "\n"
            "stab, ld, bd = ex10_stabilize_jacobian_logdet(big, eps=1e-4)\n"
            "assert t.isfinite(ld).all(), 'no NaN allowed in logdet'\n"
            "# Every bad slot has logdet 0.\n"
            "assert t.allclose(ld[bd], t.zeros(bd.sum().item()), atol=1e-5)\n"
            "# At least the 20 injected bad slots are flagged (we may also catch some natural-near-singular ones).\n"
            "for i in bad_idx.tolist():\n"
            "    assert bd[i].item(), f'injected-bad slot {i} not flagged'\n"
            "\n"
            "# --- Visualization: bar chart of logdet, flagged bad slots highlighted ---\n"
            "fig, ax = plt.subplots(figsize=(8, 3))\n"
            "colors = ['tab:red' if b else 'tab:blue' for b in bd.tolist()]\n"
            "ax.bar(range(B), ld.numpy(), color=colors, edgecolor='none')\n"
            "ax.set_xlabel('batch index'); ax.set_ylabel('logdet (red = replaced with I)')\n"
            "ax.set_title(f'ex10 stabilised logdet — {bd.sum().item()}/{B} replaced')\n"
            "ax.axhline(0, color='k', linewidth=0.5)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex10_stabilize_jacobian_logdet(jacobians: Tensor, eps: float = 1e-6):\n"
            "    B, D, _ = jacobians.shape\n"
            "    # Three diagnostic masks of shape (B,).\n"
            "    nan_mask = t.isnan(jacobians).any(dim=(-2, -1))\n"
            "    # det of NaN matrices is NaN; mask them off first to compute det safely.\n"
            "    safe_for_det = jacobians.clone()\n"
            "    safe_for_det[nan_mask] = t.eye(D)\n"
            "    dets = t.linalg.det(safe_for_det)\n"
            "    singular_mask = dets.abs() < eps\n"
            "    flip_mask     = dets < 0\n"
            "    bad = nan_mask | singular_mask | flip_mask\n"
            "    # Identity-replace.\n"
            "    stabilized = jacobians.clone()\n"
            "    stabilized[bad] = t.eye(D)\n"
            "    # logdet of the cleaned batch — guaranteed finite for det > 0.\n"
            "    logdet = t.log(t.linalg.det(stabilized).clamp(min=eps))\n"
            "    return stabilized.to(t.float32), logdet.to(t.float32), bad"
        ),
        "solution_notes": (
            "**Compose multiple masks with `|`.** Each diagnostic "
            "(NaN, near-singular, flipped) is a single bool tensor; "
            "stacking them with bitwise-or gives one combined "
            "predicate that drives a single indexed assignment "
            "`stabilized[bad] = I`. This pattern is everywhere in "
            "production ML — guarding linalg ops, masking dead "
            "neurons, gating MoE routes.\n\n"
            "**Why we clone before det.** `t.linalg.det` of a matrix "
            "with NaN entries returns NaN, which propagates through "
            "`<` comparisons in surprising ways. Substituting `I` for "
            "NaN slots BEFORE det makes the singular/flip masks "
            "clean — even though those slots get re-substituted with "
            "`I` again afterwards.\n\n"
            "**Why log-det not det.** Normalizing flows accumulate "
            "log-Jacobian-determinants across many layers. Working in "
            "log-space avoids the under/overflow that comes from "
            "multiplying many `det`s; identity-replacement guarantees "
            "`log(det) = 0` for the slots we couldn't trust, so the "
            "training loss stays finite."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # ----------------------------------------------------------------- bcast ex10
    {
        "atom_id": "broadcasting-rules",
        "subtopic": "Numpy: Vectorization and broadcasting",
        "topic_folder": "prereqs_numpy",
        "atom_recap_md": RECAP_BCAST,
        "exercise_index": 10,
        "exercise_title": "end-to-end scaled-dot-product attention via 4-axis broadcasting",
        "slug": "end-to-end-scaled-dot-product-attention-via-4-axis-broadcasting",
        "bloom_level": "Create",
        "difficulty_num": 5,
        "difficulty_dots": "🔴🔴🔴🔴🔴",
        "keywords": ["attention", "multi-head", "padding-mask", "scaled-dot-product", "integrative", "ml-adjacent"],
        "kcs": ["predict-broadcast-shape", "axis-insertion-via-unsqueeze"],
        "lo": (
            "Combine batch-axis, head-axis, and padding-mask broadcasting "
            "in a single end-to-end scaled-dot-product attention forward "
            "pass, materialising no intermediates beyond what broadcasting "
            "supplies."
        ),
        "prompt_body": (
            "Existing ex1–ex9 covered the broadcasting rule, "
            "row/column broadcast, axis insertion, outer product, "
            "pairwise distance, attention SCORES (ex7), per-channel "
            "bias, and a silent-broadcast trap. This drill is the "
            "*end-to-end attention* integration: compute the FULL "
            "scaled-dot-product attention output (not just the "
            "scores), and broadcast a per-sample padding mask across "
            "all heads.\n\n"
            "Implement `ex10_attention(Q, K, V, pad_mask)`:\n\n"
            "1. `Q`, `K`, `V` all have shape `(B, H, T, D)` — batch, "
            "heads, sequence, head-dim.\n"
            "2. `pad_mask` has shape `(B, T)` — `True` where the token "
            "is a real token, `False` where it is padding.\n"
            "3. Compute attention scores: `S = Q @ K.transpose(-1, -2) "
            "/ sqrt(D)` (shape `(B, H, T, T)`).\n"
            "4. Broadcast the padding mask **from `(B, T)` to "
            "`(B, 1, 1, T)`** so it applies across all heads and all "
            "query positions; set padded KEY positions to `-inf`.\n"
            "5. Softmax over the last axis to get attention weights "
            "`(B, H, T, T)`.\n"
            "6. Weighted sum: `out = weights @ V` (shape `(B, H, T, D)`).\n"
            "7. Return `(out, weights)` — both `float32`.\n\n"
            "Key broadcasting moves:\n"
            "- `(B, T)` → `(B, 1, 1, T)` via two `unsqueeze`s (head + "
            "query-position) — broadcast across `H` and the query "
            "axis.\n"
            "- Padded-row safety: softmax over an all-`-inf` row "
            "produces NaN; the test tolerates that for fully-padded "
            "rows but rejects NaN in real rows."
        ),
        "stub": (
            "def ex10_attention(Q: Tensor, K: Tensor, V: Tensor, pad_mask: Tensor):\n"
            '    """Return (out, weights). Padding mask broadcast across heads + query positions."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "B, H, T, D = 2, 4, 5, 8\n"
            "rng = t.Generator().manual_seed(7)\n"
            "Q = t.randn(B, H, T, D, generator=rng)\n"
            "K = t.randn(B, H, T, D, generator=rng)\n"
            "V = t.randn(B, H, T, D, generator=rng)\n"
            "# Pad mask: batch 0 has last 2 positions padded; batch 1 is fully unpadded.\n"
            "pad_mask = t.tensor([\n"
            "    [True, True, True, False, False],\n"
            "    [True, True, True, True,  True ],\n"
            "])\n"
            "out, w = ex10_attention(Q, K, V, pad_mask)\n"
            "assert out.shape == (B, H, T, D), f'out shape: {tuple(out.shape)}'\n"
            "assert w.shape == (B, H, T, T), f'weights shape: {tuple(w.shape)}'\n"
            "assert out.dtype == t.float32\n"
            "assert w.dtype == t.float32\n"
            "\n"
            "# Real tokens: weights along last axis sum to 1.\n"
            "real_row_sums = w[0, :, :, :].sum(dim=-1)\n"
            "assert t.allclose(real_row_sums, t.ones_like(real_row_sums), atol=1e-4)\n"
            "real_row_sums_b1 = w[1, :, :, :].sum(dim=-1)\n"
            "assert t.allclose(real_row_sums_b1, t.ones_like(real_row_sums_b1), atol=1e-4)\n"
            "\n"
            "# Padded KEY positions get zero attention weight in batch 0.\n"
            "assert t.allclose(w[0, :, :, 3], t.zeros(H, T), atol=1e-5), 'padded key 3 must get 0 weight'\n"
            "assert t.allclose(w[0, :, :, 4], t.zeros(H, T), atol=1e-5), 'padded key 4 must get 0 weight'\n"
            "# No padding in batch 1 → all weights strictly positive.\n"
            "assert (w[1] > 0).all().item(), 'unpadded batch should have positive weights everywhere'\n"
            "\n"
            "# Cross-check vs torch.nn.functional reference.\n"
            "import torch.nn.functional as F\n"
            "attn_mask = ~pad_mask.unsqueeze(1).unsqueeze(1).expand(B, H, T, T)\n"
            "ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=~attn_mask)\n"
            "assert t.allclose(out, ref, atol=1e-4), 'output must match torch.nn.functional reference'\n"
            "\n"
            "# Larger random shape — no NaN anywhere when all tokens are real.\n"
            "pm_all_real = t.ones(B, T, dtype=t.bool)\n"
            "out2, w2 = ex10_attention(Q, K, V, pm_all_real)\n"
            "assert t.isfinite(out2).all() and t.isfinite(w2).all()\n"
            "\n"
            "# --- Visualization: attention heatmap, head 0 of batch 0 (with padding) ---\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))\n"
            "axes[0].imshow(w[0, 0].detach().numpy(), cmap='viridis', vmin=0, vmax=1)\n"
            "axes[0].set_title('batch 0 head 0\\n(keys 3,4 padded)')\n"
            "axes[0].set_xlabel('key idx'); axes[0].set_ylabel('query idx')\n"
            "axes[1].imshow(w[1, 0].detach().numpy(), cmap='viridis', vmin=0, vmax=1)\n"
            "axes[1].set_title('batch 1 head 0\\n(no padding)')\n"
            "axes[1].set_xlabel('key idx'); axes[1].set_ylabel('query idx')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex10_attention(Q: Tensor, K: Tensor, V: Tensor, pad_mask: Tensor):\n"
            "    import math\n"
            "    B, H, T, D = Q.shape\n"
            "    scores = Q @ K.transpose(-1, -2) / math.sqrt(D)         # (B, H, T, T)\n"
            "    # (B, T) -> (B, 1, 1, T) so it broadcasts over heads + query axis.\n"
            "    key_mask = pad_mask.unsqueeze(1).unsqueeze(1)            # (B, 1, 1, T)\n"
            "    scores = scores.masked_fill(~key_mask, float('-inf'))\n"
            "    weights = t.softmax(scores, dim=-1)                      # (B, H, T, T)\n"
            "    out = weights @ V                                        # (B, H, T, D)\n"
            "    return out.to(t.float32), weights.to(t.float32)"
        ),
        "solution_notes": (
            "**Two `unsqueeze`s map `(B, T)` onto `(B, H, T, T)`.** "
            "Position 1 (head axis) and position 2 (query axis) are "
            "inserted as size-1 dims; right-align broadcasting then "
            "expands both to their full size. One `pad_mask` of "
            "`B * T` bools controls `B * H * T * T` attention "
            "entries — broadcasting buys you a `H * T` reduction.\n\n"
            "**Why we mask the KEY axis, not the query axis.** A "
            "padded query position is a row of the attention matrix "
            "that we'll throw away downstream (its output is "
            "irrelevant). A padded KEY position must NEVER contribute "
            "to ANY query's output, so we zero it on the column axis. "
            "Masking columns to `-inf` before softmax → softmax sends "
            "them to 0.\n\n"
            "**Contrast with ex7.** ex7 computed batched scores with "
            "shape-trace debug. This drill adds the V multiply, the "
            "mask, and the softmax — the whole attention block — and "
            "verifies against `torch.nn.functional.scaled_dot_product_"
            "attention`. That's the broadcasting payoff: 6 lines of "
            "torch is the entire attention layer."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
