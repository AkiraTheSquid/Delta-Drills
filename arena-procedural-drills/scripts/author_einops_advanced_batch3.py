#!/usr/bin/env python3
"""Author 8 standalone Colab drills for advanced einops/einsum + masking atoms.

Targets ARENA chap-0 atoms that recur heavily in ray-tracing (part 1) and
CNNs (part 2) and are *distinct* from the einops basics already covered by
batch-1:

  einsum-contraction         — ex1, ex2  (INDEX rules, not the pattern library)
  einops-repeat-broadcast    — ex1, ex2  (zero-stride broadcasting trick)
  boolean-mask-combine       — ex1       (multi-criterion AND/OR/NOT)
  inf-masking                — ex1       (attention -inf fill for softmax)
  unbind-tuple-unpack        — ex1       (Pythonic destructure of axis)
  einops-rearrange-flatten   — ex1       (`'b h w c -> b (h w c)'`)

Constraints (per Doughty ACE 2024 + Maier 2021):
  - One LO + one Bloom per exercise.
  - <= 2 concurrent KCs per exercise.
  - Solution body runs cleanly under test_body (torch 2.12.0+cpu, einops 0.8.2).
  - These drills break up ARENA composites — smaller skills, not ARENA-scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_einops_advanced"

# ─────────────────────────────────────────────────────────────────────────────
# Recap snippets — one per atom; reused across that atom's exercises.
# ─────────────────────────────────────────────────────────────────────────────

RECAP_EINSUM_CONTRACTION = (
    "## Einsum index rules — quick refresher\n"
    "\n"
    "`einsum` is governed by **two simple index rules**:\n"
    "\n"
    "1. **An index that appears on BOTH sides of `->`** is preserved (carried "
    "through unchanged on every operand and the output). Think of it as a "
    "broadcasted/batch axis.\n"
    "2. **An index that appears on the INPUT but NOT on the output** is "
    "*summed-contracted*: einsum multiplies aligned entries and sums them away.\n"
    "\n"
    "**Repeated index on the same operand.** If an index appears twice on one "
    "operand (e.g. `'i i -> i'`), it pulls the **diagonal**. If `'i i ->'` with "
    "nothing on the right, it pulls the diagonal *and* sums it — that's the trace.\n"
    "\n"
    "**Worked examples of the rules in action:**\n"
    "- `'i j, j k -> i k'`: `j` repeats across operands → sum-contracted (matmul).\n"
    "- `'i j -> i'`: `j` dropped from rhs → row sum.\n"
    "- `'i j -> '`: all indices dropped → grand sum (scalar).\n"
    "- `'i j, i j -> i j'`: nothing dropped → elementwise product (Hadamard).\n"
    "- `'i, j -> i j'`: nothing repeated, both kept → outer product.\n"
    "\n"
    "**Why this matters.** Once you internalise these two rules, you can read "
    "*any* einsum pattern from left to right and predict the output without "
    "running it. That's the whole pedagogical payoff."
)

RECAP_EINOPS_REPEAT_BROADCAST = (
    "## Einops repeat as broadcast — quick refresher\n"
    "\n"
    "`einops.repeat(x, '... -> ... new', new=N)` is **not** a memcpy when used "
    "for the pure broadcast pattern. Under the hood, einops compiles it to a "
    "`torch.Tensor.expand` (or numpy stride trick) that sets the **stride of the "
    "new axis to zero** — every position along the new axis points at the same "
    "underlying storage cell.\n"
    "\n"
    "**Three names for the same trick:**\n"
    "- `einops.repeat(x, 'b d -> b n d', n=N)`\n"
    "- `x.unsqueeze(1).expand(-1, N, -1)`\n"
    "- `x[:, None, :].broadcast_to((x.shape[0], N, x.shape[1]))`\n"
    "\n"
    "All three produce a view with `stride=0` on the inserted axis. No memory "
    "is allocated for the duplicates — they're a single value read N times.\n"
    "\n"
    "**When this is the right call.** Anywhere you need to *pair every X with "
    "every Y* (e.g. every ray with every triangle in ARENA's ray tracer), reach "
    "for `einops.repeat` — it produces the expanded view in O(1) memory.\n"
    "\n"
    "**Compared to `repeat` that actually copies.** "
    "`einops.repeat(x, 'b d -> b (n d)', n=N)` *does* materialise the copy "
    "because the output shape collapses the repeat axis into another. Only "
    "patterns that **insert** a new axis (and leave it un-grouped) stay at "
    "stride 0."
)

RECAP_BOOL_MASK_COMBINE = (
    "## Combining boolean masks — quick refresher\n"
    "\n"
    "Multi-criterion selection is the bread-and-butter of vectorised filtering. "
    "PyTorch / NumPy give you three logical operators on `bool` tensors:\n"
    "\n"
    "- `&` — elementwise AND\n"
    "- `|` — elementwise OR\n"
    "- `~` — elementwise NOT (logical, on `bool` tensors)\n"
    "\n"
    "All three are **elementwise** and broadcast normally. They produce a "
    "fresh `bool` tensor; the operands are not consumed.\n"
    "\n"
    "**Parenthesise every comparison.** `&` / `|` bind tighter than `<` / `>` / "
    "`==` in Python, so `x > 0 & x < 10` parses as `x > (0 & x) < 10` — a "
    "silent disaster. Always write `(x > 0) & (x < 10)`.\n"
    "\n"
    "**Watch the dtype.** The operands must be `bool`. If you have `0/1` "
    "integer flags, convert with `.bool()` first or you'll get bitwise math on "
    "integers (`& == bitwise-and`, not logical-and).\n"
    "\n"
    "**Canonical ARENA pattern.** The ray-triangle inside test ANDs five "
    "predicates: `(s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular`. "
    "Five comparisons → five bool tensors → ANDed down to one final mask."
)

RECAP_INF_MASKING = (
    "## `masked_fill` with -inf — quick refresher\n"
    "\n"
    "The canonical attention-masking trick:\n"
    "\n"
    "```python\n"
    "scores = scores.masked_fill(mask, float('-inf'))\n"
    "weights = scores.softmax(dim=-1)\n"
    "```\n"
    "\n"
    "**Why -inf and not 0.** Softmax computes `exp(s) / sum(exp(s))`. If you "
    "zero out a masked position *before* softmax, `exp(0) = 1` and it still "
    "gets weight `1 / (1 + sum_others)` — a non-trivial contribution. If you "
    "fill with `-inf`, `exp(-inf) = 0` exactly, so the position contributes "
    "zero to both numerator and denominator — fully suppressed.\n"
    "\n"
    "**Numerical sanity.** PyTorch's softmax does the log-sum-exp trick "
    "internally, so even with `-inf` in the input, the output is well-defined "
    "(unless an entire row is masked — then you get `nan` from `0/0`).\n"
    "\n"
    "**In-place vs out-of-place.** `masked_fill_` (trailing underscore) "
    "mutates; `masked_fill` returns a new tensor. Both take a `bool` mask "
    "with `True` = \"fill this position\". Mismatched dtype on the mask "
    "raises.\n"
    "\n"
    "**Broadcasting.** The mask broadcasts against the scores tensor. A "
    "`(T, T)` causal mask applied to `(B, H, T, T)` scores is the standard "
    "transformer pattern — the mask is replicated across batch and heads "
    "with zero memory cost."
)

RECAP_UNBIND_TUPLE = (
    "## Unbind for tuple destructure — quick refresher\n"
    "\n"
    "`x.unbind(dim=k)` returns a **Python tuple** of `x.shape[k]` view-tensors, "
    "each with axis `k` removed. The Pythonic move is to *destructure* that "
    "tuple directly into named variables:\n"
    "\n"
    "```python\n"
    "origin, direction = rays.unbind(dim=1)         # rays: (B, 2, 3)\n"
    "ox, oy, oz        = origin.unbind(dim=-1)      # origin: (B, 3)\n"
    "```\n"
    "\n"
    "**Why this beats indexing.** `rays[:, 0]` and `rays[:, 1]` also work, but "
    "they hide the *count* and the *labels*. Destructuring into named "
    "variables tells the reader \"this axis has exactly 2 elements, here are "
    "their meanings\". Mis-spell a name → instant `NameError`; mis-index → "
    "silent semantic bug.\n"
    "\n"
    "**It's a tuple, not a list.** That means it works with `*args` "
    "splat (`func(*rays.unbind(dim=0))`) and with sequence unpacking — but "
    "*not* with list mutation. If you need to modify the slices before "
    "restacking, wrap in `list(...)` first.\n"
    "\n"
    "**Views, not copies.** Writes through the destructured tensors alias the "
    "source (just like slicing). Read-only consumers don't care, but if you "
    "intend to mutate, call `.clone()` on each slice."
)

RECAP_REARRANGE_FLATTEN = (
    "## Einops rearrange-as-flatten — quick refresher\n"
    "\n"
    "The most common flatten pattern in computer vision:\n"
    "\n"
    "```python\n"
    "rearrange(x, 'b h w c -> b (h w c)')\n"
    "rearrange(x, 'b c h w -> b (c h w)')\n"
    "```\n"
    "\n"
    "Parentheses *group* the named axes into a single composite axis. The "
    "order inside the parens is the **inner-loop-fastest** order of the "
    "flatten — change `(h w c)` to `(c h w)` and the resulting 1-D layout is "
    "completely different (even though the shape is the same).\n"
    "\n"
    "**Why this is preferred over `.view(b, -1)`:**\n"
    "1. **Self-documenting** — the axis names make the flatten order obvious.\n"
    "2. **Order-explicit** — `view`/`flatten` use *whichever order the memory "
    "happens to be in*. If `x` is non-contiguous, `view` errors; `rearrange` "
    "transparently calls `.contiguous()` for you.\n"
    "3. **Compositional** — `'b c h w -> b (c h w)'` is one of dozens of "
    "patterns you can mix and match without learning a separate API for "
    "each (`flatten`, `view`, `reshape`, `permute+view`, ...).\n"
    "\n"
    "**Canonical CNN use.** Right before the final `Linear` head, `'b c h w "
    "-> b (c h w)'` collapses the spatial+channel axes into one feature axis. "
    "ARENA's `nn.Flatten` is exactly this pattern."
)


SPECS = [
    # ═══════════════════════════════════════════════════════════════════════
    # einsum-contraction (2 exercises)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "einsum-contraction",
        "subtopic": "Einsum: Index contraction semantics",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EINSUM_CONTRACTION,
        "exercise_index": 1,
        "exercise_title": "predict + verify which indices get summed",
        "slug": "predict-and-verify-which-indices-get-summed",
        "bloom_level": "Analyze",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["index-rules", "sum-contraction", "prediction"],
        "kcs": ["einsum-repeated-index-sums", "einsum-missing-rhs-reduces"],
        "lo": (
            "Analyse an einsum pattern string by listing which indices are "
            "preserved (appear on both sides) versus contracted (appear on "
            "input but not output), then verify your prediction against the "
            "actual output shape."
        ),
        "prompt_body": (
            "Implement `ex1_predict_contraction(pattern, input_shapes)`.\n\n"
            "Given:\n"
            "- `pattern`: an einops einsum pattern string like "
            "`'i j, j k -> i k'`.\n"
            "- `input_shapes`: a list of tuples, one per operand, giving each "
            "operand's shape.\n\n"
            "Return a `dict` with three keys:\n"
            "- `'preserved'`: sorted list of indices appearing on both sides "
            "of `->`.\n"
            "- `'contracted'`: sorted list of indices appearing on input but "
            "missing from output.\n"
            "- `'output_shape'`: the tuple einsum should produce, computed "
            "purely from the index rules (no torch call).\n\n"
            "**The rules you must apply:**\n"
            "- An index on input + output → preserved (size = the matching "
            "operand axis).\n"
            "- An index on input only → contracted (summed away).\n"
            "- All operand axes labelled with the same index must have the "
            "same size.\n\n"
            "The test cell calls `einops.einsum` on random tensors and "
            "verifies your predicted `output_shape` matches the real result."
        ),
        "stub": (
            "def ex1_predict_contraction(pattern: str, input_shapes: list) -> dict:\n"
            '    """Return {\'preserved\': [...], \'contracted\': [...], \'output_shape\': (...)}."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# --- Case A: standard matmul ---\n"
            "out = ex1_predict_contraction('i j, j k -> i k', [(3, 4), (4, 5)])\n"
            "assert out['preserved']  == ['i', 'k'], f\"preserved: {out['preserved']}\"\n"
            "assert out['contracted'] == ['j'],      f\"contracted: {out['contracted']}\"\n"
            "assert out['output_shape'] == (3, 5),   f\"shape: {out['output_shape']}\"\n"
            "real = einops.einsum(t.randn(3, 4), t.randn(4, 5), 'i j, j k -> i k')\n"
            "assert tuple(real.shape) == out['output_shape']\n"
            "\n"
            "# --- Case B: row sum (j dropped from rhs) ---\n"
            "out = ex1_predict_contraction('i j -> i', [(2, 7)])\n"
            "assert out['preserved']  == ['i']\n"
            "assert out['contracted'] == ['j']\n"
            "assert out['output_shape'] == (2,)\n"
            "real = einops.einsum(t.randn(2, 7), 'i j -> i')\n"
            "assert tuple(real.shape) == out['output_shape']\n"
            "\n"
            "# --- Case C: batched matmul (b preserved through both) ---\n"
            "out = ex1_predict_contraction('b i k, b k j -> b i j', [(2, 3, 4), (2, 4, 5)])\n"
            "assert out['preserved']  == ['b', 'i', 'j']\n"
            "assert out['contracted'] == ['k']\n"
            "assert out['output_shape'] == (2, 3, 5)\n"
            "real = einops.einsum(t.randn(2, 3, 4), t.randn(2, 4, 5), 'b i k, b k j -> b i j')\n"
            "assert tuple(real.shape) == out['output_shape']\n"
            "\n"
            "# --- Case D: outer product (nothing repeated, both kept) ---\n"
            "out = ex1_predict_contraction('i, j -> i j', [(3,), (5,)])\n"
            "assert out['preserved']  == ['i', 'j']\n"
            "assert out['contracted'] == []\n"
            "assert out['output_shape'] == (3, 5)\n"
            "real = einops.einsum(t.randn(3), t.randn(5), 'i, j -> i j')\n"
            "assert tuple(real.shape) == out['output_shape']\n"
            "\n"
            "# --- Case E: hadamard (nothing dropped → no contraction) ---\n"
            "out = ex1_predict_contraction('i j, i j -> i j', [(4, 6), (4, 6)])\n"
            "assert out['contracted'] == []\n"
            "assert out['output_shape'] == (4, 6)\n"
            "\n"
            "print('all 5 cases predicted correctly')"
        ),
        "solution_body": (
            "def ex1_predict_contraction(pattern: str, input_shapes: list) -> dict:\n"
            "    lhs, rhs = pattern.split('->')\n"
            "    rhs_idx = rhs.strip().split()\n"
            "    operand_idx = [op.strip().split() for op in lhs.split(',')]\n"
            "    # All input indices (across operands), deduped, sorted.\n"
            "    all_input = set()\n"
            "    for ops in operand_idx:\n"
            "        all_input.update(ops)\n"
            "    preserved = sorted(i for i in all_input if i in rhs_idx)\n"
            "    contracted = sorted(i for i in all_input if i not in rhs_idx)\n"
            "    # Map each named index to its size (look up in the first operand that has it).\n"
            "    sizes = {}\n"
            "    for ops, shape in zip(operand_idx, input_shapes):\n"
            "        for name, n in zip(ops, shape):\n"
            "            sizes.setdefault(name, n)\n"
            "    output_shape = tuple(sizes[i] for i in rhs_idx)\n"
            "    return {'preserved': preserved, 'contracted': contracted, 'output_shape': output_shape}"
        ),
        "solution_notes": (
            "**Why this is an Analyse-level task.** You're not running einsum "
            "and reading the shape off — you're *predicting* the shape from "
            "the pattern string alone. That forces you to internalise the two "
            "rules: shared-with-rhs survives, missing-from-rhs gets summed.\n\n"
            "**The size-map trick.** When the same name appears on multiple "
            "operands (`'i j, j k -> i k'`), all occurrences must have the "
            "same size — einsum will error otherwise. `sizes.setdefault` "
            "records the first occurrence; you could optionally assert that "
            "later occurrences match for an even stricter checker.\n\n"
            "**What about repeated index on one operand?** `'i i -> i'` is "
            "the diagonal trick; einops handles it but this simple predictor "
            "doesn't. The standard ARENA patterns never use that form, so it's "
            "out of scope here."
        ),
    },
    {
        "atom_id": "einsum-contraction",
        "subtopic": "Einsum: Index contraction semantics",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EINSUM_CONTRACTION,
        "exercise_index": 2,
        "exercise_title": "lambertian dot product via index contraction",
        "slug": "lambertian-dot-product-via-index-contraction",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["dot-product", "lambertian", "ray-tracing", "lighting"],
        "kcs": ["einsum-repeated-index-sums", "einsum-batch-axis-passthrough"],
        "lo": (
            "Apply the einsum index rules to compute a batched dot product "
            "between per-triangle surface normals and a single light direction, "
            "using a pattern where one index is shared (and contracted) and "
            "one is batch (and preserved)."
        ),
        "prompt_body": (
            "Implement `ex2_lambertian_intensity(normals, light)` — the dot "
            "product that drives ARENA's Lambertian shading bonus exercise:\n\n"
            "- `normals` has shape `(NT, 3)` — one unit-length normal per "
            "triangle.\n"
            "- `light` has shape `(3,)` — a single light direction.\n"
            "- Return `(NT,)` — the dot product of each normal with the light.\n\n"
            "**Constraint.** You must use `einops.einsum` (not `@`, not "
            "`(normals * light).sum(-1)`). The whole point of the drill is "
            "naming the indices.\n\n"
            "**Build the pattern from the rules.** You need `nt` to survive "
            "(every triangle gets its own output) and `dims` to be summed "
            "(that's the dot product). Pattern: "
            "`'nt dims, dims -> nt'` — `dims` is shared across operands and "
            "missing from rhs → contracted. `nt` is on the first operand only "
            "and on rhs → preserved.\n\n"
            "Don't clamp the output here (the ARENA Lambertian step does "
            "`max(0, ...)` separately to suppress backward-facing normals)."
        ),
        "stub": (
            "def ex2_lambertian_intensity(normals: Tensor, light: Tensor) -> Tensor:\n"
            '    """Per-triangle dot product n_t . light via einops.einsum."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-build 3 normals + an overhead light → known intensities.\n"
            "normals = t.tensor([\n"
            "    [0.0, 1.0, 0.0],    # straight up — faces the light fully\n"
            "    [0.0, -1.0, 0.0],   # straight down — back-faces the light\n"
            "    [1.0, 0.0, 0.0],    # sideways — orthogonal to light\n"
            "])\n"
            "light = t.tensor([0.0, 1.0, 0.0])\n"
            "out = ex2_lambertian_intensity(normals, light)\n"
            "assert out.shape == (3,), f'expected (3,), got {tuple(out.shape)}'\n"
            "expected = t.tensor([1.0, -1.0, 0.0])\n"
            "assert t.allclose(out, expected, atol=1e-6), f'value mismatch:\\n{out}\\nvs\\n{expected}'\n"
            "\n"
            "# Ground truth via raw torch dot product (no einsum) — must agree.\n"
            "rng = t.Generator().manual_seed(2)\n"
            "big_normals = t.randn(50, 3, generator=rng)\n"
            "big_normals = big_normals / big_normals.norm(dim=1, keepdim=True)\n"
            "big_light   = t.tensor([0.3, 0.8, 0.5])\n"
            "out_big = ex2_lambertian_intensity(big_normals, big_light)\n"
            "ground = (big_normals * big_light).sum(dim=-1)\n"
            "assert out_big.shape == (50,)\n"
            "assert t.allclose(out_big, ground, atol=1e-5), 'einsum dot must match (n*l).sum(-1)'\n"
            "\n"
            "# All intensities for unit normals + unit light live in [-1, 1].\n"
            "unit_light = big_light / big_light.norm()\n"
            "ints = ex2_lambertian_intensity(big_normals, unit_light)\n"
            "assert ints.min() >= -1.0 - 1e-5 and ints.max() <= 1.0 + 1e-5, 'unit dot must be in [-1,1]'\n"
            "\n"
            "print('lambertian intensities computed via einsum index contraction')"
        ),
        "solution_body": (
            "def ex2_lambertian_intensity(normals: Tensor, light: Tensor) -> Tensor:\n"
            "    return einops.einsum(normals, light, 'nt dims, dims -> nt')"
        ),
        "solution_notes": (
            "**Reading the pattern through the rules.** "
            "`'nt dims, dims -> nt'`:\n"
            "- `dims` appears on both operands but NOT on rhs → contracted "
            "(summed). That's the dot product.\n"
            "- `nt` appears on the first operand and on rhs → preserved. "
            "Each triangle gets its own scalar.\n"
            "- `light` has no `nt` axis but lives in the same `dims` space — "
            "einsum broadcasts it against every triangle. No `repeat` needed.\n\n"
            "**Why we don't write `nt dims, nt dims -> nt`.** You *could* "
            "first `repeat` the light to `(NT, 3)` and then contract — that's "
            "the explicit form. But einsum is happy to broadcast missing "
            "batch axes for free. The shorter pattern is idiomatic.\n\n"
            "**Where ARENA uses this exactly.** "
            "`einops.einsum(normals, light.to(device), 'nt dims, dims -> nt')` "
            "is the Lambertian step in `raytrace_mesh_lighting` (0_1_9). The "
            "downstream `t.where(intensity > 0, intensity, 0)` is the "
            "back-face clip — distinct atom, not this one."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # einops-repeat-broadcast (2 exercises)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "einops-repeat-broadcast",
        "subtopic": "Einops: Repeat-as-broadcast",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EINOPS_REPEAT_BROADCAST,
        "exercise_index": 1,
        "exercise_title": "every-ray-with-every-triangle pairing without copy",
        "slug": "every-ray-with-every-triangle-pairing-without-copy",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["broadcast", "pair-every-with-every", "stride-zero", "ray-tracing"],
        "kcs": ["repeat-inserts-zero-stride-axis", "repeat-pair-every-with-every"],
        "lo": (
            "Apply `einops.repeat` to broadcast a `(NT, 3, 3)` triangle batch "
            "against an `(NR, 2, 3)` ray batch so every (ray, triangle) pair "
            "becomes one slot of an `(NR, NT, ...)` tensor — without "
            "materialising the copies."
        ),
        "prompt_body": (
            "Implement `ex1_pair_rays_with_triangles(rays, triangles)`.\n\n"
            "- `rays` has shape `(NR, 2, 3)` — `NR` rays, each `(origin, "
            "direction)` of length 3.\n"
            "- `triangles` has shape `(NT, 3, 3)` — `NT` triangles, each "
            "`(A, B, C)` of length 3.\n\n"
            "Return a tuple `(rays_b, tris_b)` where:\n"
            "- `rays_b.shape == (NR, NT, 2, 3)` — every ray paired against "
            "every triangle.\n"
            "- `tris_b.shape == (NR, NT, 3, 3)` — every triangle paired "
            "against every ray.\n\n"
            "**Constraint — no copies.** Use `einops.repeat` to insert the "
            "new axis. After your call, the returned tensors must share "
            "storage with the original inputs (the test asserts "
            "`data_ptr()` equality). That's only true if the new axis has "
            "stride 0; using `.repeat()` (the torch method, which copies) "
            "or any pattern that *groups* the new axis (`(NR NT)`) will "
            "fail the storage check.\n\n"
            "**Patterns to use:**\n"
            "- `einops.repeat(rays, 'nr p d -> nr nt p d', nt=NT)`\n"
            "- `einops.repeat(triangles, 'nt p d -> nr nt p d', nr=NR)`"
        ),
        "stub": (
            "def ex1_pair_rays_with_triangles(\n"
            "    rays: Tensor, triangles: Tensor\n"
            ") -> tuple[Tensor, Tensor]:\n"
            '    """Broadcast every ray with every triangle. No copy."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "NR, NT = 4, 5\n"
            "rays      = t.randn(NR, 2, 3)\n"
            "triangles = t.randn(NT, 3, 3)\n"
            "\n"
            "rays_b, tris_b = ex1_pair_rays_with_triangles(rays, triangles)\n"
            "assert rays_b.shape == (NR, NT, 2, 3), f'rays_b: {tuple(rays_b.shape)}'\n"
            "assert tris_b.shape == (NR, NT, 3, 3), f'tris_b: {tuple(tris_b.shape)}'\n"
            "\n"
            "# --- No copy check ---\n"
            "assert rays_b.data_ptr() == rays.data_ptr(), (\n"
            "    'rays_b must be a stride-0 view of rays (no memory copy). '\n"
            "    'Did you accidentally call .repeat()/.contiguous()/.clone()?'\n"
            ")\n"
            "assert tris_b.data_ptr() == triangles.data_ptr(), (\n"
            "    'tris_b must be a stride-0 view of triangles'\n"
            ")\n"
            "\n"
            "# --- Value check: every slice along the broadcast axis is identical ---\n"
            "for r in range(NR):\n"
            "    for tri in range(NT):\n"
            "        assert t.equal(rays_b[r, tri], rays[r]), f'rays mis-broadcast at ({r},{tri})'\n"
            "        assert t.equal(tris_b[r, tri], triangles[tri]), f'tris mis-broadcast at ({r},{tri})'\n"
            "\n"
            "# --- Stride-0 check on the inserted axis ---\n"
            "# rays_b inserted axis is `nt` (position 1); tris_b inserted axis is `nr` (position 0).\n"
            "assert rays_b.stride()[1] == 0, f'rays_b axis-1 stride must be 0, got {rays_b.stride()[1]}'\n"
            "assert tris_b.stride()[0] == 0, f'tris_b axis-0 stride must be 0, got {tris_b.stride()[0]}'\n"
            "\n"
            "# --- Smoke test at realistic ARENA scale (no allocation blowup) ---\n"
            "big_rays = t.randn(2000, 2, 3)\n"
            "big_tris = t.randn(100, 3, 3)\n"
            "br, bt = ex1_pair_rays_with_triangles(big_rays, big_tris)\n"
            "assert br.shape == (2000, 100, 2, 3) and bt.shape == (2000, 100, 3, 3)\n"
            "assert br.data_ptr() == big_rays.data_ptr() and bt.data_ptr() == big_tris.data_ptr()\n"
            "print('paired 2000 x 100 = 200,000 (ray, tri) slots with zero copies')"
        ),
        "solution_body": (
            "def ex1_pair_rays_with_triangles(rays: Tensor, triangles: Tensor) -> tuple[Tensor, Tensor]:\n"
            "    NR = rays.shape[0]\n"
            "    NT = triangles.shape[0]\n"
            "    rays_b = einops.repeat(rays, 'nr p d -> nr nt p d', nt=NT)\n"
            "    tris_b = einops.repeat(triangles, 'nt p d -> nr nt p d', nr=NR)\n"
            "    return rays_b, tris_b"
        ),
        "solution_notes": (
            "**Storage check is the whole point.** You can produce the right "
            "shape with `rays.repeat(1, NT, 1, 1)` (torch's `.repeat()`) or "
            "with `.expand().contiguous()` — but both **materialise** the "
            "200,000-row tensor, which at scale (ARENA's mesh exercise hits "
            "millions of pairs) explodes memory. The `data_ptr()` assertion "
            "catches both mistakes immediately.\n\n"
            "**Why einops can do this.** `'nr p d -> nr nt p d'` declares "
            "that `nt` is a NEW axis (not present on input), and asks for "
            "size `nt=NT`. Internally einops compiles this to "
            "`x.unsqueeze(1).expand(-1, NT, -1, -1)` — `expand` is the "
            "stride-0 trick.\n\n"
            "**Contrast with the materialising case.** "
            "`einops.repeat(rays, 'nr p d -> (nr k) p d', k=NT)` would also "
            "produce `(NR*NT, 2, 3)`-ish output, but because the repeat axis "
            "is *grouped* with `nr`, the output can't be a view — einops "
            "*does* memcpy. Rule of thumb: a fresh, ungrouped repeat axis is "
            "free; any grouping forces a copy."
        ),
    },
    {
        "atom_id": "einops-repeat-broadcast",
        "subtopic": "Einops: Repeat-as-broadcast",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_EINOPS_REPEAT_BROADCAST,
        "exercise_index": 2,
        "exercise_title": "per-token positional bias broadcast across batch",
        "slug": "per-token-positional-bias-broadcast-across-batch",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["broadcast", "positional-bias", "attention", "transformer"],
        "kcs": ["repeat-inserts-zero-stride-axis", "repeat-vs-torch-method-repeat"],
        "lo": (
            "Apply `einops.repeat` to broadcast a `(T,)` per-position scalar "
            "bias across an `(B, H, T)` batch axis tensor without copying — "
            "the canonical transformer per-token bias pattern."
        ),
        "prompt_body": (
            "Implement `ex2_broadcast_bias(bias, scores)`.\n\n"
            "- `bias` has shape `(T,)` — one scalar per sequence position.\n"
            "- `scores` has shape `(B, H, T)` — per-head per-token attention "
            "logits.\n\n"
            "Return `bias_b` with shape `(B, H, T)` — `bias` broadcast across "
            "the batch and head axes so it can be *added* to `scores`.\n\n"
            "**Constraint — no copy.** Use `einops.repeat` to insert the "
            "leading `b` and `h` axes. The returned tensor must share storage "
            "with `bias` (the test asserts `data_ptr()` equality and "
            "stride-0 on the inserted axes).\n\n"
            "Pattern: `einops.repeat(bias, 't -> b h t', b=B, h=H)`.\n\n"
            "**Don't return `scores + bias_b`** — just return `bias_b`. The "
            "drill is about producing the broadcast view, not the addition."
        ),
        "stub": (
            "def ex2_broadcast_bias(bias: Tensor, scores: Tensor) -> Tensor:\n"
            '    """Broadcast (T,) bias to scores.shape (B, H, T). No copy."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "B, H, T = 2, 4, 6\n"
            "bias   = t.arange(T, dtype=t.float32)\n"
            "scores = t.randn(B, H, T)\n"
            "\n"
            "bias_b = ex2_broadcast_bias(bias, scores)\n"
            "assert bias_b.shape == (B, H, T), f'expected ({B},{H},{T}), got {tuple(bias_b.shape)}'\n"
            "\n"
            "# --- No copy check ---\n"
            "assert bias_b.data_ptr() == bias.data_ptr(), (\n"
            "    'bias_b must share storage with bias (no memcpy). '\n"
            "    'Did you accidentally call .repeat() or .clone()?'\n"
            ")\n"
            "\n"
            "# --- Stride-0 on inserted axes (b=0, h=1) ---\n"
            "assert bias_b.stride()[0] == 0, f'axis-0 (b) stride must be 0, got {bias_b.stride()[0]}'\n"
            "assert bias_b.stride()[1] == 0, f'axis-1 (h) stride must be 0, got {bias_b.stride()[1]}'\n"
            "\n"
            "# --- Value check: every (b, h, :) slice equals bias ---\n"
            "for b_i in range(B):\n"
            "    for h_i in range(H):\n"
            "        assert t.equal(bias_b[b_i, h_i], bias), f'bias mis-broadcast at ({b_i},{h_i})'\n"
            "\n"
            "# --- Operational check: addable to scores → produces expected per-token bias ---\n"
            "biased = scores + bias_b\n"
            "expected = scores + bias.view(1, 1, T)\n"
            "assert t.allclose(biased, expected, atol=1e-6), 'addition with broadcast bias must match manual reshape+add'\n"
            "\n"
            "# --- Edge: T=1 (degenerate broadcast — still must be a view) ---\n"
            "tiny_bias = t.tensor([3.14])\n"
            "tiny_scores = t.randn(2, 3, 1)\n"
            "tiny_b = ex2_broadcast_bias(tiny_bias, tiny_scores)\n"
            "assert tiny_b.shape == (2, 3, 1)\n"
            "assert tiny_b.data_ptr() == tiny_bias.data_ptr()\n"
            "print('positional bias broadcast as a stride-0 view')"
        ),
        "solution_body": (
            "def ex2_broadcast_bias(bias: Tensor, scores: Tensor) -> Tensor:\n"
            "    B, H, T = scores.shape\n"
            "    return einops.repeat(bias, 't -> b h t', b=B, h=H)"
        ),
        "solution_notes": (
            "**Why this matters more than it looks.** In a real transformer "
            "the bias broadcast happens at every layer, every step — if it "
            "materialised, you'd pay `B * H * T` memory per layer (which "
            "for `B=8, H=32, T=2048` is half a million floats). Keeping it "
            "as a stride-0 view costs `T` floats total.\n\n"
            "**`einops.repeat` vs `torch.Tensor.repeat`.** Same name, "
            "opposite semantics! `bias.repeat(B, H, 1)` *does* memcpy — "
            "`torch.Tensor.repeat` is the materialising version. "
            "`einops.repeat` is the broadcast-when-possible version. Confusing, "
            "but the storage assertion in the test catches it instantly.\n\n"
            "**Equivalent torch idiom.** `bias.view(1, 1, T).expand(B, H, T)` "
            "is the same view, written in raw PyTorch. einops is preferred "
            "because the pattern string `'t -> b h t'` documents the "
            "dimensional intent."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # boolean-mask-combine (1 exercise)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "boolean-mask-combine",
        "subtopic": "Numpy: Boolean mask combine",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_BOOL_MASK_COMBINE,
        "exercise_index": 1,
        "exercise_title": "five-predicate ray-triangle inside test",
        "slug": "five-predicate-ray-triangle-inside-test",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["and", "or", "not", "ray-tracing", "inside-test"],
        "kcs": ["mask-bitwise-and-or", "mask-parenthesize-comparisons"],
        "lo": (
            "Apply elementwise `&`, `|`, `~` on bool tensors to combine five "
            "comparison-derived predicates into the single 'point inside "
            "triangle and matrix not singular' mask used by ARENA's batched "
            "ray-triangle intersection."
        ),
        "prompt_body": (
            "Implement `ex1_inside_test(s, u, v, is_singular)`.\n\n"
            "Given the per-(ray, triangle) outputs of `t.linalg.solve` on "
            "ARENA's Moller-Trumbore matrix:\n"
            "- `s`: `(NR, NT)` ray parameter (signed distance along the ray).\n"
            "- `u, v`: `(NR, NT)` barycentric coords.\n"
            "- `is_singular`: `(NR, NT)` bool — True where the 3x3 was "
            "singular and the solve output is garbage.\n\n"
            "Return a `(NR, NT)` bool mask that is `True` iff:\n"
            "1. `s >= 0` (intersection in front of the ray origin)\n"
            "2. `u >= 0`\n"
            "3. `v >= 0`\n"
            "4. `u + v <= 1` (inside the triangle's barycentric region)\n"
            "5. NOT `is_singular`\n\n"
            "**Critical syntax point.** Each comparison MUST be "
            "parenthesised — `&` binds tighter than `>=` in Python, so "
            "`s >= 0 & u >= 0` silently parses as "
            "`s >= (0 & u) >= 0`. Write `(s >= 0) & (u >= 0) & ...`."
        ),
        "stub": (
            "def ex1_inside_test(s: Tensor, u: Tensor, v: Tensor, is_singular: Tensor) -> Tensor:\n"
            '    """Combine 5 predicates → (NR, NT) bool intersect mask."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-build 6 (ray, tri) slots — one for each failure mode + one all-True.\n"
            "s   = t.tensor([[ 0.5, -0.1,  0.5,  0.5,  0.5,  0.5]])\n"
            "u   = t.tensor([[ 0.3,  0.3, -0.1,  0.3,  0.6,  0.3]])\n"
            "v   = t.tensor([[ 0.3,  0.3,  0.3,  -0.1, 0.6,  0.3]])\n"
            "sng = t.tensor([[False, False, False, False, False, True]])\n"
            "out = ex1_inside_test(s, u, v, sng)\n"
            "# Slot 0: all pass → True. Slots 1-5: each fails one predicate.\n"
            "expected = t.tensor([[True, False, False, False, False, False]])\n"
            "assert out.dtype == t.bool, f'expected bool, got {out.dtype}'\n"
            "assert out.shape == (1, 6), f'expected (1,6), got {tuple(out.shape)}'\n"
            "assert t.equal(out, expected), f'mismatch:\\n  got      {out}\\n  expected {expected}'\n"
            "\n"
            "# --- Random batch — must agree with the long-form (((... & ...) & ...) & ...) ---\n"
            "rng = t.Generator().manual_seed(11)\n"
            "NR, NT = 7, 9\n"
            "s2 = t.randn(NR, NT, generator=rng)\n"
            "u2 = t.randn(NR, NT, generator=rng) * 0.6\n"
            "v2 = t.randn(NR, NT, generator=rng) * 0.6\n"
            "sng2 = t.randn(NR, NT, generator=rng) > 1.5  # ~6% True\n"
            "ground = (s2 >= 0) & (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1) & (~sng2)\n"
            "assert t.equal(ex1_inside_test(s2, u2, v2, sng2), ground), 'random-batch mask mismatch'\n"
            "\n"
            "# --- Edge: all singular → all False regardless of other predicates ---\n"
            "all_sng = t.ones(NR, NT, dtype=t.bool)\n"
            "assert not ex1_inside_test(s2, u2, v2, all_sng).any().item(), 'all-singular must yield all-False'\n"
            "\n"
            "# --- Edge: trivially passing case (s=u=v=0, not singular) → boundary hit, expect True ---\n"
            "zero = t.zeros(2, 2)\n"
            "ok = t.zeros(2, 2, dtype=t.bool)\n"
            "assert ex1_inside_test(zero, zero, zero, ok).all().item(), '(s=u=v=0, ~singular) lies on the boundary → must be True'\n"
            "print(f'random-batch: {ground.sum().item()}/{NR*NT} slots passed all 5 predicates')"
        ),
        "solution_body": (
            "def ex1_inside_test(s: Tensor, u: Tensor, v: Tensor, is_singular: Tensor) -> Tensor:\n"
            "    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (~is_singular)"
        ),
        "solution_notes": (
            "**Order of operations.** `&` chains left-to-right; "
            "`a & b & c & d & e` is `((((a & b) & c) & d) & e)`. Python "
            "doesn't short-circuit elementwise — every predicate is fully "
            "evaluated, then the bitwise AND ladder collapses them.\n\n"
            "**Why parenthesise every comparison.** `&` and `|` are bitwise "
            "operators in Python, with precedence *higher* than `<`, `>`, "
            "`==`. So `s >= 0 & u >= 0` parses as `s >= (0 & u) >= 0` — which "
            "is at best wrong and at worst silently runs on int tensors. The "
            "parens around each comparison are mandatory.\n\n"
            "**Why `~` not `not`.** `not` calls `__bool__()` which only works "
            "on a 0-D tensor (and raises on a >0-D one). `~` is the "
            "elementwise bitwise NOT, which on `bool` tensors is the "
            "logical NOT.\n\n"
            "**Performance note.** Each `&` allocates a new bool tensor. "
            "If memory is tight on long predicate chains, build incrementally: "
            "`mask = s >= 0; mask &= u >= 0; ...` (`&=` is in-place)."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # inf-masking (1 exercise)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "inf-masking",
        "subtopic": "Numpy: Inf-fill masking trick",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_INF_MASKING,
        "exercise_index": 1,
        "exercise_title": "causal attention masking with -inf + softmax + viz",
        "slug": "causal-attention-masking-with-inf-and-softmax-viz",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["attention", "causal-mask", "softmax", "visualization"],
        "kcs": ["masked-fill-neg-inf-before-softmax", "softmax-zeros-masked-positions"],
        "lo": (
            "Apply `masked_fill` with `-inf` to a scores tensor before softmax "
            "so the masked positions receive zero weight in the output "
            "distribution, then visualise before-and-after attention heatmaps "
            "to confirm the masked half is fully suppressed."
        ),
        "prompt_body": (
            "Implement `ex1_causal_softmax(scores)`.\n\n"
            "- `scores` has shape `(T, T)` — pre-softmax attention logits.\n"
            "- Build a causal mask: position `i` may attend to positions "
            "`0..i` but not `i+1..T-1`. (True above the diagonal = blocked.)\n"
            "- Fill blocked positions with `-inf`, then softmax along the "
            "**last axis** (the key axis).\n"
            "- Return the `(T, T)` attention weights.\n\n"
            "**Why -inf and not 0.** `softmax(0) = exp(0) / sum_exp = "
            "1 / (1 + sum_others)` — masked positions still get a tiny "
            "non-zero weight. `softmax(-inf) = exp(-inf) / sum_exp = "
            "0 / sum_exp = 0` exactly. Verifying this is the whole point of "
            "the exercise.\n\n"
            "The visualization renders the *before-mask* and *after-mask* "
            "softmax side by side so you can see the upper triangle go from "
            "non-zero to flat zero."
        ),
        "stub": (
            "def ex1_causal_softmax(scores: Tensor) -> Tensor:\n"
            '    """Causal mask + softmax. Returns (T, T) row-normalised weights."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "T = 5\n"
            "scores = t.randn(T, T, generator=t.Generator().manual_seed(0))\n"
            "weights = ex1_causal_softmax(scores)\n"
            "\n"
            "assert weights.shape == (T, T)\n"
            "assert weights.dtype == t.float32\n"
            "\n"
            "# Each row must sum to 1.\n"
            "row_sums = weights.sum(dim=-1)\n"
            "assert t.allclose(row_sums, t.ones(T), atol=1e-6), f'rows must sum to 1, got {row_sums}'\n"
            "\n"
            "# Upper triangle (mask=True) must be EXACTLY 0 — that's the -inf payoff.\n"
            "mask = t.triu(t.ones(T, T, dtype=t.bool), diagonal=1)\n"
            "masked_vals = weights[mask]\n"
            "assert (masked_vals == 0).all().item(), (\n"
            "    'upper triangle must be exactly 0 (proves -inf was used). '\n"
            "    f'found non-zero values: {masked_vals[masked_vals != 0]}. '\n"
            "    'Did you mask with 0 instead of -inf?'\n"
            ")\n"
            "\n"
            "# Lower triangle + diagonal must be > 0 (softmax never zeros out non-masked positions).\n"
            "assert (weights[~mask] > 0).all().item(), 'non-masked entries must be strictly positive'\n"
            "\n"
            "# Position 0 attends only to itself → weight 1.\n"
            "assert abs(weights[0, 0].item() - 1.0) < 1e-6, f'row 0 must be [1, 0, 0, 0, 0], got {weights[0]}'\n"
            "\n"
            "# Ground truth via PyTorch primitives.\n"
            "expected = scores.masked_fill(mask, float('-inf')).softmax(dim=-1)\n"
            "assert t.allclose(weights, expected, atol=1e-6), 'must match canonical masked_fill + softmax'\n"
            "\n"
            "# --- Visualization: before-mask vs after-mask softmax ---\n"
            "T_viz = 8\n"
            "scores_viz = t.randn(T_viz, T_viz, generator=t.Generator().manual_seed(1))\n"
            "before = scores_viz.softmax(dim=-1)               # no mask\n"
            "after  = ex1_causal_softmax(scores_viz)           # with -inf mask\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))\n"
            "im1 = ax1.imshow(before.numpy(), cmap='viridis', vmin=0, vmax=before.max().item())\n"
            "ax1.set_title('softmax WITHOUT mask\\n(upper triangle has weight)')\n"
            "ax1.set_xlabel('key position'); ax1.set_ylabel('query position')\n"
            "plt.colorbar(im1, ax=ax1, fraction=0.046)\n"
            "im2 = ax2.imshow(after.numpy(), cmap='viridis', vmin=0, vmax=after.max().item())\n"
            "ax2.set_title('softmax WITH -inf mask\\n(upper triangle is EXACTLY 0)')\n"
            "ax2.set_xlabel('key position'); ax2.set_ylabel('query position')\n"
            "plt.colorbar(im2, ax=ax2, fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex1_causal_softmax(scores: Tensor) -> Tensor:\n"
            "    T = scores.shape[-1]\n"
            "    mask = t.triu(t.ones(T, T, dtype=t.bool), diagonal=1)\n"
            "    return scores.masked_fill(mask, float('-inf')).softmax(dim=-1)"
        ),
        "solution_notes": (
            "**`t.triu(..., diagonal=1)`.** Returns the strict upper triangle "
            "(excluding the diagonal). For causal attention, the diagonal "
            "*must* be allowed (a token can attend to itself), so use "
            "`diagonal=1` not `diagonal=0`.\n\n"
            "**Why `masked_fill` not `scores[mask] = -inf`.** Same result, but "
            "`masked_fill` is functional (returns a new tensor — no surprise "
            "mutation of the caller's `scores`) and is the idiomatic torch "
            "API for this exact pattern.\n\n"
            "**Row-of-all-masked → nan.** If an entire row is masked (no "
            "valid keys at all), softmax does `0/0` and you get `nan`. In "
            "practice the diagonal is always unmasked for causal attention, "
            "so the issue doesn't arise — but for arbitrary masks, add a "
            "sentinel: ensure at least one True per row before applying.\n\n"
            "**Numerical stability.** PyTorch's `softmax` does log-sum-exp "
            "internally: it subtracts `max(scores, dim=-1)` before exping. "
            "If your row has `-inf`s, the subtraction yields `-inf` minus a "
            "finite number = `-inf`, so `exp(...) = 0` exactly. The math "
            "stays clean — no `inf - inf = nan` traps."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # unbind-tuple-unpack (1 exercise)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "unbind-tuple-unpack",
        "subtopic": "PyTorch: Unbind tuple-unpack",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_UNBIND_TUPLE,
        "exercise_index": 1,
        "exercise_title": "ARENA-style two-level destructure of rays into (ox, oy, oz, dx, dy, dz)",
        "slug": "two-level-destructure-rays-to-named-components",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["unbind", "destructure", "ray-tracing", "named-components"],
        "kcs": ["unbind-returns-python-tuple", "unbind-two-level-destructure"],
        "lo": (
            "Apply two levels of `unbind` + tuple unpacking to a `(B, 2, 3)` "
            "ray batch, producing six named `(B,)` tensors `(ox, oy, oz, dx, "
            "dy, dz)` ready for analytic per-axis arithmetic."
        ),
        "prompt_body": (
            "Implement `ex1_destructure_rays(rays)`.\n\n"
            "- `rays` has shape `(B, 2, 3)` — row 0 of each `(2, 3)` block is "
            "the origin, row 1 is the direction.\n"
            "- Return a `dict` with six keys: `'ox'`, `'oy'`, `'oz'`, `'dx'`, "
            "`'dy'`, `'dz'` — each a `(B,)` tensor.\n\n"
            "**Required idiom (not an option).**\n"
            "1. First level: `origin, direction = rays.unbind(dim=1)` — "
            "tuple destructure of the size-2 axis.\n"
            "2. Second level: `ox, oy, oz = origin.unbind(dim=-1)` and "
            "`dx, dy, dz = direction.unbind(dim=-1)` — tuple destructure of "
            "the size-3 axis.\n\n"
            "**Forbidden alternatives.** You may NOT use `rays[:, 0, 0]` / "
            "`rays[:, 1, 2]` / `select` / index arithmetic. The drill is "
            "about exercising `unbind` *as a destructure*, not as a slicer. "
            "(The test does not enforce this — it's pedagogical discipline. "
            "Look at the solution if you're unsure.)\n\n"
            "**Why this is worth a whole drill.** Two-level unbind + named "
            "destructure is the ARENA ray-tracing idiom that makes the "
            "Moller-Trumbore solver readable. Without it, the next line is "
            "`t_hit = -rays[:, 0, 1] / rays[:, 1, 1]` — write-only code. "
            "With it: `t_hit = -oy / dy`."
        ),
        "stub": (
            "def ex1_destructure_rays(rays: Tensor) -> dict:\n"
            '    """Two-level unbind → {ox, oy, oz, dx, dy, dz} each (B,)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rays = t.tensor([\n"
            "    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],     # ray 0\n"
            "    [[10., 20., 30.], [40., 50., 60.]],     # ray 1\n"
            "    [[-1., -2., -3.], [-4., -5., -6.]],     # ray 2\n"
            "])\n"
            "out = ex1_destructure_rays(rays)\n"
            "\n"
            "expected_keys = {'ox', 'oy', 'oz', 'dx', 'dy', 'dz'}\n"
            "assert set(out.keys()) == expected_keys, f'keys: {set(out.keys())} vs {expected_keys}'\n"
            "\n"
            "for k, v in out.items():\n"
            "    assert v.shape == (3,), f'{k!r} shape: {tuple(v.shape)} (expected (3,))'\n"
            "    assert v.dtype == t.float32, f'{k!r} dtype: {v.dtype}'\n"
            "\n"
            "# Value checks - rays[i, 0] = (ox[i], oy[i], oz[i]); rays[i, 1] = (dx[i], dy[i], dz[i]).\n"
            "assert t.equal(out['ox'], t.tensor([1.0, 10., -1.]))\n"
            "assert t.equal(out['oy'], t.tensor([2.0, 20., -2.]))\n"
            "assert t.equal(out['oz'], t.tensor([3.0, 30., -3.]))\n"
            "assert t.equal(out['dx'], t.tensor([4.0, 40., -4.]))\n"
            "assert t.equal(out['dy'], t.tensor([5.0, 50., -5.]))\n"
            "assert t.equal(out['dz'], t.tensor([6.0, 60., -6.]))\n"
            "\n"
            "# Views, not copies — writes through the destructured tensors should be invisible to the caller's `rays` here\n"
            "# because the test rebuilds `rays`, but the storage check confirms `unbind` returned a view.\n"
            "assert out['oy'].data_ptr() != 0  # has storage\n"
            "# Verify oy is actually a view of rays (data_ptr falls within rays' storage range).\n"
            "rays_start = rays.data_ptr()\n"
            "rays_end   = rays_start + rays.numel() * rays.element_size()\n"
            "assert rays_start <= out['oy'].data_ptr() < rays_end, 'oy must be a view into rays storage'\n"
            "\n"
            "# Bigger batch — operational use: compute t_hit = -oy / dy for the ground-plane intersection.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "big = t.randn(50, 2, 3, generator=rng)\n"
            "d = ex1_destructure_rays(big)\n"
            "t_hit = -d['oy'] / d['dy']\n"
            "assert t_hit.shape == (50,), f'derived t_hit shape: {tuple(t_hit.shape)}'\n"
            "# Verify against direct indexing.\n"
            "t_hit_ref = -big[:, 0, 1] / big[:, 1, 1]\n"
            "assert t.allclose(t_hit, t_hit_ref, equal_nan=True), '-oy/dy must match -rays[:,0,1]/rays[:,1,1]'\n"
            "print('two-level destructure round-trip verified on 50 random rays')"
        ),
        "solution_body": (
            "def ex1_destructure_rays(rays: Tensor) -> dict:\n"
            "    origin, direction = rays.unbind(dim=1)\n"
            "    ox, oy, oz = origin.unbind(dim=-1)\n"
            "    dx, dy, dz = direction.unbind(dim=-1)\n"
            "    return {'ox': ox, 'oy': oy, 'oz': oz, 'dx': dx, 'dy': dy, 'dz': dz}"
        ),
        "solution_notes": (
            "**The two `unbind` calls do different jobs.** "
            "`rays.unbind(dim=1)` peels the 2-axis (origin vs direction). "
            "`origin.unbind(dim=-1)` peels the 3-axis (x vs y vs z "
            "components). Both *destructure* an axis of known small length "
            "into named tensors — no explicit loop, no indexing.\n\n"
            "**Why named over indexed.** Compare the next line of an "
            "actual ARENA solver:\n"
            "```python\n"
            "# Named (this exercise):\n"
            "t_hit = -oy / dy\n"
            "# Indexed (no destructure):\n"
            "t_hit = -rays[:, 0, 1] / rays[:, 1, 1]\n"
            "```\n"
            "The indexed form is correct but read-only — any reviewer has to "
            "stop and decode `[:, 0, 1]`. Named is self-documenting.\n\n"
            "**Six aliases, one storage.** All six returned tensors are views "
            "into the same underlying storage as `rays`. Cheap to create, "
            "instantly garbage-collected when the dict goes out of scope. The "
            "storage-range assertion in the test verifies this — the address "
            "of `oy` lies strictly inside the byte range owned by `rays`."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # einops-rearrange-flatten (1 exercise)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "atom_id": "einops-rearrange-flatten",
        "subtopic": "Einops: Rearrange-as-flatten",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_REARRANGE_FLATTEN,
        "exercise_index": 1,
        "exercise_title": "CNN flatten before the Linear head — channel + spatial collapse",
        "slug": "cnn-flatten-before-linear-head",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["flatten", "cnn", "linear-head", "axis-composition"],
        "kcs": ["rearrange-axis-composition-via-parens", "rearrange-flatten-preserves-batch"],
        "lo": (
            "Apply `einops.rearrange` with the `'b c h w -> b (c h w)'` "
            "pattern to flatten a 4-D CNN feature map into a 2-D `(B, "
            "features)` matrix suitable for a `Linear` layer, while "
            "preserving the batch axis."
        ),
        "prompt_body": (
            "Implement `ex1_flatten_cnn_features(x)`.\n\n"
            "- `x` has shape `(B, C, H, W)` — the canonical PyTorch CNN "
            "feature map (batch, channels, height, width).\n"
            "- Return shape `(B, C*H*W)` — the per-sample feature vector, "
            "batch axis preserved.\n\n"
            "**Constraint.** Use exactly `rearrange(x, 'b c h w -> b (c h "
            "w)')`. No `view`, no `flatten`, no `reshape`. The pattern "
            "string is load-bearing — the test asserts the flatten order "
            "matches `(c h w)`, which is `nn.Flatten`'s default and what "
            "`x.flatten(start_dim=1)` produces.\n\n"
            "**Why this specific pattern.** ARENA's `SimpleMLP` and "
            "`ConvNet` both compose `Flatten() → Linear(...)`. The `Linear` "
            "layer expects a 2-D `(B, features)` input; this rearrange is "
            "the bridge."
        ),
        "stub": (
            "def ex1_flatten_cnn_features(x: Tensor) -> Tensor:\n"
            '    """Flatten (B, C, H, W) -> (B, C*H*W) preserving batch axis."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-build to verify the flatten order is (c h w).\n"
            "# Use distinguishable values so we can verify the ordering, not just the shape.\n"
            "x = t.arange(24.0).reshape(1, 2, 3, 4)  # (B=1, C=2, H=3, W=4)\n"
            "out = ex1_flatten_cnn_features(x)\n"
            "assert out.shape == (1, 24), f'expected (1, 24), got {tuple(out.shape)}'\n"
            "# rearrange 'b c h w -> b (c h w)' must produce the same byte order as x.flatten(1).\n"
            "expected = x.flatten(start_dim=1)\n"
            "assert t.equal(out, expected), (\n"
            "    f'flatten order mismatch — did you write (h w c) or (w h c) instead of (c h w)?\\n'\n"
            "    f'got:      {out[0]}\\n'\n"
            "    f'expected: {expected[0]}'\n"
            ")\n"
            "\n"
            "# --- Realistic CNN-ish shape ---\n"
            "rng = t.Generator().manual_seed(0)\n"
            "B, C, H, W = 8, 32, 7, 7\n"
            "feat = t.randn(B, C, H, W, generator=rng)\n"
            "flat = ex1_flatten_cnn_features(feat)\n"
            "assert flat.shape == (B, C * H * W), f'expected ({B}, {C*H*W}), got {tuple(flat.shape)}'\n"
            "assert t.equal(flat, feat.flatten(start_dim=1)), 'must match nn.Flatten default behavior'\n"
            "\n"
            "# --- Pluggable into a real Linear layer ---\n"
            "lin = t.nn.Linear(C * H * W, 10)\n"
            "logits = lin(flat)\n"
            "assert logits.shape == (B, 10), 'output of Linear(flat) must be (B, num_classes)'\n"
            "\n"
            "# --- Edge: B=1 (single sample) ---\n"
            "single = t.randn(1, 4, 5, 5)\n"
            "fs = ex1_flatten_cnn_features(single)\n"
            "assert fs.shape == (1, 100)\n"
            "\n"
            "# --- Edge: C=H=W=1 (degenerate scalar-per-sample) ---\n"
            "scalar = t.randn(3, 1, 1, 1)\n"
            "ss = ex1_flatten_cnn_features(scalar)\n"
            "assert ss.shape == (3, 1)\n"
            "assert t.equal(ss, scalar.view(3, 1))\n"
            "print('flatten matches nn.Flatten on CNN-shaped, single-sample, and scalar inputs')"
        ),
        "solution_body": (
            "def ex1_flatten_cnn_features(x: Tensor) -> Tensor:\n"
            "    return rearrange(x, 'b c h w -> b (c h w)')"
        ),
        "solution_notes": (
            "**Why `(c h w)` and not `(h w c)`.** PyTorch's default tensor "
            "layout is channels-first (`B, C, H, W`). The byte order in "
            "memory is `c` outer, `w` inner. `(c h w)` matches that — it's "
            "free, no copy needed (einops doesn't even call `.contiguous()` "
            "in this case). `(h w c)` would force a permute + copy.\n\n"
            "**Equivalent torch idioms.**\n"
            "- `x.flatten(start_dim=1)` — most concise.\n"
            "- `x.view(x.shape[0], -1)` — fastest (requires contiguous).\n"
            "- `nn.Flatten()(x)` — the Module form ARENA uses inside `Sequential`.\n\n"
            "All four produce the identical tensor; the test checks against "
            "`x.flatten(start_dim=1)` as the ground truth.\n\n"
            "**Why we prefer the rearrange form even though it's the longest.** "
            "Reading code: `'b c h w -> b (c h w)'` tells you the axes by "
            "name, the order they're flattened in, and what survives. "
            "`x.view(B, -1)` tells you nothing about which axes get "
            "collapsed or in what order. In a CNN pipeline with multiple "
            "reshape steps (patchify, unpatchify, channel-shuffle, ...), "
            "named patterns prevent silent bugs."
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
