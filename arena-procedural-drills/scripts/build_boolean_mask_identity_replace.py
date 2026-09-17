#!/usr/bin/env python3
"""Build the boolean-mask-identity-replace procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace.ipynb`
(Phase 4.11). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `boolean-mask-identity-replace` covers the Ray Tracing pattern of
detecting degenerate matrices (e.g. singular A in `Ax=b` for ray-triangle
intersection) and replacing them with identity + zero RHS so the whole
batch can still be solved — degenerate slots come out as the zero vector
instead of crashing the solve. Bridges to bank topic `Numpy` via
the explicit token rule (`boolean-mask`) in atom_readiness.js, reporting
to subtopic `Numpy: Indexing and selection`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace.ipynb"


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


ATOM_ID = "boolean-mask-identity-replace"
SUBTOPIC = "Numpy: Indexing and selection"
TITLE = "boolean mask & identity replace — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "mask-from-condition",
        "kind": "component-skill",
        "description": "Build a boolean mask tensor from a comparison: `mask = x < 0`. Result has the same shape as `x` and dtype `torch.bool`.",
    },
    {
        "id": "boolean-indexed-write",
        "kind": "component-skill",
        "description": "Assign a scalar via boolean indexing: `x[mask] = 0.0`. Modifies in place; the mask must match the shape of the indexed slice.",
    },
    {
        "id": "row-mask-broadcast-assign",
        "kind": "component-skill",
        "description": "Use a 1-D boolean mask of shape `(B,)` on a 2-D tensor of shape `(B, N)`: `x[row_mask] = row_value`. The right-hand side broadcasts across all flagged rows.",
    },
    {
        "id": "identity-substitute-singular-batched",
        "kind": "component-skill",
        "description": "Replace singular submatrices in a batched matrix tensor `(B, N, N)` with the identity matrix, using a 1-D bool mask of shape `(B,)`. Cleans up the system so a batched solve can complete.",
    },
    {
        "id": "safe-batched-solve",
        "kind": "integrative-skill",
        "description": "Combine determinant-based singularity detection, identity substitution in `A`, zero substitution in `b`, and `torch.linalg.solve` to compute a per-batch linear-system solution where degenerate slots return as the zero vector instead of NaN / error. The canonical 'safe batched solve' Ray Tracing pattern.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "build a boolean mask from a comparison",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["bool-tensor", "elementwise-compare", "dtype-bool"],
        "kcs": ["mask-from-condition"],
        "lo": "Recall that comparison ops on tensors produce a boolean mask of the same shape.",
        "prompt_body": (
            "Implement `ex1_negative_mask(x)` to return a boolean tensor marking "
            "the negative entries of `x`. Output dtype must be `torch.bool`.\n\n"
            "Use the elementwise comparison `x < 0`."
        ),
        "stub": (
            "def ex1_negative_mask(x: Tensor) -> Tensor:\n"
            '    """Return a bool tensor marking entries of x that are strictly negative."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([1.0, -2.0, 3.0, -4.0, 0.0])\n"
            "    m = ex1_negative_mask(x)\n"
            "    assert m.shape == x.shape, f'shape mismatch: {m.shape} vs {x.shape}'\n"
            "    assert m.dtype == t.bool, f'expected dtype bool, got {m.dtype}'\n"
            "    expected = t.tensor([False, True, False, True, False])\n"
            "    assert t.equal(m, expected), f'mask mismatch: {m.tolist()}'\n"
            "    # 0.0 must NOT be flagged — '<' is strict.\n"
            "    assert m[4].item() is False, '0.0 is not strictly negative'"
        ),
        "solution_body": (
            "def ex1_negative_mask(x: Tensor) -> Tensor:\n"
            "    return x < 0"
        ),
        "solution_notes": (
            "**Strict vs non-strict.** `x < 0` excludes `0.0`; `x <= 0` would "
            "include it. Choose based on what you mean: if you're detecting "
            "degenerate / non-positive values that need fixing, you usually want "
            "`<=`; if you're separating sign, `<` keeps zero in the positive bin."
        ),
    },
    {
        "id": "ex2",
        "title": "replace negatives with zero (non-destructive)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["masked-assign", "clone", "in-place-vs-functional"],
        "kcs": ["boolean-indexed-write"],
        "lo": "Apply boolean-indexed scalar assignment to clamp negative values to zero — without mutating the input.",
        "prompt_body": (
            "Implement `ex2_clamp_zero(x)` to return a new tensor where every "
            "negative entry of `x` has been replaced with `0.0`. The input "
            "tensor `x` must NOT be modified.\n\n"
            "Strategy:\n"
            "1. `y = x.clone()` to get a fresh tensor.\n"
            "2. `y[y < 0] = 0.0` — boolean-indexed scalar write.\n"
            "3. Return `y`."
        ),
        "stub": (
            "def ex2_clamp_zero(x: Tensor) -> Tensor:\n"
            '    """Return a copy of x with negatives replaced by 0. Does NOT mutate x."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([1.0, -2.0, 3.0, -4.0, 0.0])\n"
            "    x_orig = x.clone()  # snapshot to verify non-mutation\n"
            "    y = ex2_clamp_zero(x)\n"
            "    assert y.shape == x.shape\n"
            "    expected = t.tensor([1.0, 0.0, 3.0, 0.0, 0.0])\n"
            "    assert t.allclose(y, expected), f'value mismatch: {y.tolist()}'\n"
            "    # Critical — input must be untouched.\n"
            "    assert t.allclose(x, x_orig), 'input tensor was mutated — must clone first'"
        ),
        "solution_body": (
            "def ex2_clamp_zero(x: Tensor) -> Tensor:\n"
            "    y = x.clone()\n"
            "    y[y < 0] = 0.0\n"
            "    return y"
        ),
        "solution_notes": (
            "**Why clone first?** Boolean-indexed assignment (`y[mask] = value`) "
            "is *in-place* on `y`. If you skipped `.clone()`, you'd be writing "
            "into the caller's tensor — a silent bug that surfaces when the "
            "caller expects their input intact.\n\n"
            "**Alternative.** `torch.where(x < 0, t.zeros_like(x), x)` is the "
            "functional form — no clone, no in-place. Slightly slower in some "
            "cases but never mutates."
        ),
    },
    {
        "id": "ex3",
        "title": "zero out flagged rows of a 2-D tensor",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["row-mask", "broadcast-assign", "batch-zero"],
        "kcs": ["row-mask-broadcast-assign"],
        "lo": "Apply a 1-D bool mask on the first axis of a 2-D tensor to zero out whole rows in one assignment.",
        "prompt_body": (
            "Implement `ex3_zero_flagged_rows(b, mask)`. Given a `(B, N)` tensor "
            "`b` and a 1-D bool mask of shape `(B,)`, return a copy of `b` where "
            "every row flagged `True` has been set to all zeros. Unflagged rows "
            "stay unchanged.\n\n"
            "Strategy: `out = b.clone(); out[mask] = 0.0` — the scalar `0.0` "
            "broadcasts to fill every selected row.\n\n"
            "Input `b` must NOT be mutated."
        ),
        "stub": (
            "def ex3_zero_flagged_rows(b: Tensor, mask: Tensor) -> Tensor:\n"
            '    """Zero out rows of b where mask is True. Does NOT mutate b."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "b = t.tensor([\n"
            "        [1.0, 2.0, 3.0],\n"
            "        [4.0, 5.0, 6.0],\n"
            "        [7.0, 8.0, 9.0],\n"
            "        [10.0, 11.0, 12.0],\n"
            "    ])\n"
            "    b_orig = b.clone()\n"
            "    mask = t.tensor([False, True, False, True])\n"
            "    out = ex3_zero_flagged_rows(b, mask)\n"
            "    assert out.shape == b.shape, f'shape mismatch: {out.shape} vs {b.shape}'\n"
            "    expected = t.tensor([\n"
            "        [1.0, 2.0, 3.0],\n"
            "        [0.0, 0.0, 0.0],\n"
            "        [7.0, 8.0, 9.0],\n"
            "        [0.0, 0.0, 0.0],\n"
            "    ])\n"
            "    assert t.allclose(out, expected), f'value mismatch:\\n{out}'\n"
            "    assert t.allclose(b, b_orig), 'input tensor was mutated'\n"
            "    # All-False mask should leave b unchanged.\n"
            "    no_mask = t.zeros(4, dtype=t.bool)\n"
            "    assert t.allclose(ex3_zero_flagged_rows(b, no_mask), b), 'all-False mask should be no-op'"
        ),
        "solution_body": (
            "def ex3_zero_flagged_rows(b: Tensor, mask: Tensor) -> Tensor:\n"
            "    out = b.clone()\n"
            "    out[mask] = 0.0\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why this works.** Indexing `out[mask]` with a 1-D bool mask of "
            "shape `(B,)` against a `(B, N)` tensor selects `K = mask.sum()` "
            "rows, giving a `(K, N)` view. Writing scalar `0.0` broadcasts "
            "across that view to fill every selected row with zeros."
        ),
    },
    {
        "id": "ex4",
        "title": "substitute identity for flagged matrices",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["batched-matrix", "identity-eye", "singular-cleanup"],
        "kcs": ["identity-substitute-singular-batched"],
        "lo": "Apply a bool mask to a batched matrix tensor to overwrite flagged submatrices with the identity matrix.",
        "prompt_body": (
            "Implement `ex4_identity_substitute(A, singular_mask)`. Given a "
            "`(B, N, N)` batched matrix tensor `A` and a `(B,)` bool mask "
            "flagging singular ones, return a copy of `A` where every flagged "
            "submatrix has been replaced with `torch.eye(N)`.\n\n"
            "Hint: `out[singular_mask] = t.eye(N, dtype=A.dtype)`. The `(N, N)` "
            "identity broadcasts across the selected `(K, N, N)` slice.\n\n"
            "Input `A` must NOT be mutated."
        ),
        "stub": (
            "def ex4_identity_substitute(A: Tensor, singular_mask: Tensor) -> Tensor:\n"
            '    """Replace flagged (B, N, N) submatrices with identity. No mutation."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "A = t.tensor([\n"
            "        [[1.0, 2.0], [3.0, 4.0]],\n"
            "        [[0.0, 0.0], [0.0, 0.0]],   # singular\n"
            "        [[5.0, 6.0], [7.0, 8.0]],\n"
            "        [[2.0, 1.0], [4.0, 2.0]],   # singular (rank 1)\n"
            "    ])\n"
            "    A_orig = A.clone()\n"
            "    mask = t.tensor([False, True, False, True])\n"
            "    out = ex4_identity_substitute(A, mask)\n"
            "    assert out.shape == A.shape, f'shape mismatch: {out.shape} vs {A.shape}'\n"
            "    # Non-flagged kept untouched.\n"
            "    assert t.allclose(out[0], A[0]), 'row 0 should be unchanged'\n"
            "    assert t.allclose(out[2], A[2]), 'row 2 should be unchanged'\n"
            "    # Flagged replaced with identity.\n"
            "    I2 = t.eye(2)\n"
            "    assert t.allclose(out[1], I2), 'row 1 should be eye(2)'\n"
            "    assert t.allclose(out[3], I2), 'row 3 should be eye(2)'\n"
            "    # Non-mutation.\n"
            "    assert t.allclose(A, A_orig), 'input was mutated'"
        ),
        "solution_body": (
            "def ex4_identity_substitute(A: Tensor, singular_mask: Tensor) -> Tensor:\n"
            "    out = A.clone()\n"
            "    N = A.shape[-1]\n"
            "    out[singular_mask] = t.eye(N, dtype=A.dtype)\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why this matters.** Calling `torch.linalg.solve(A, b)` on a "
            "batched `A` blows up the entire call if even one slice is singular. "
            "Substituting identity (and a matching zero in `b` — Exercise 5) "
            "lets the solve complete; the originally-singular slots come out "
            "as the zero vector, which downstream code can detect and handle "
            "separately (e.g. 'ray missed the triangle, color = background')."
        ),
    },
    {
        "id": "ex5",
        "title": "safe batched solve (singular → zero solution)",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["batched-solve", "det", "ray-triangle", "multi-kc"],
        "kcs": [
            "mask-from-condition",
            "identity-substitute-singular-batched",
            "row-mask-broadcast-assign",
            "safe-batched-solve",
        ],
        "lo": "Synthesize singularity detection, identity substitution in `A`, zero substitution in `b`, and a batched solve to produce a per-slot solution where degenerate slots come out as zero vectors.",
        "prompt_body": (
            "Implement `ex5_safe_solve(A, b, eps=1e-6)`. Solve a batched linear "
            "system `A @ x = b` where some `A` slices may be singular.\n\n"
            "Inputs:\n"
            "- `A`: `(B, N, N)` float tensor.\n"
            "- `b`: `(B, N)` float tensor.\n"
            "- `eps`: singularity threshold on `|det(A_i)|`.\n\n"
            "Algorithm:\n"
            "1. Compute `dets = torch.det(A)` → shape `(B,)`.\n"
            "2. `singular_mask = dets.abs() < eps` — flags degenerate slots.\n"
            "3. `A_safe = A.clone(); A_safe[singular_mask] = eye(N, dtype=A.dtype)`.\n"
            "4. `b_safe = b.clone(); b_safe[singular_mask] = 0.0`.\n"
            "5. Solve `torch.linalg.solve(A_safe, b_safe.unsqueeze(-1)).squeeze(-1)`.\n"
            "6. Return result of shape `(B, N)`. Degenerate slots come out as zeros.\n\n"
            "Neither `A` nor `b` should be mutated.\n\n"
            "> ⚠️ **Integrative exercise.** Combines 4 KCs (mask-from-condition, "
            "identity-substitute-batched, row-mask-broadcast-assign, safe-batched-solve). "
            "Empirical work (Lohr et al. ITiCSE 2025) shows 3+ concept exercises "
            "drop to ~40% solvability — expect a clear step up vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_safe_solve(A: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:\n"
            '    """Batched solve with singular-A substitution. Degenerate slots → zero vector."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Two well-conditioned systems and one singular.\n"
            "    A = t.tensor([\n"
            "        [[2.0, 0.0], [0.0, 1.0]],   # solve: x = b / [2, 1]\n"
            "        [[0.0, 0.0], [0.0, 0.0]],   # singular — det = 0\n"
            "        [[1.0, 1.0], [0.0, 1.0]],   # upper triangular, det = 1\n"
            "    ])\n"
            "    b = t.tensor([\n"
            "        [4.0, 3.0],   # → [2.0, 3.0]\n"
            "        [5.0, 6.0],   # singular slot — must come out [0, 0]\n"
            "        [3.0, 2.0],   # → [1.0, 2.0]\n"
            "    ])\n"
            "    A_orig = A.clone(); b_orig = b.clone()\n"
            "    out = ex5_safe_solve(A, b)\n"
            "    assert out.shape == b.shape, f'expected {b.shape}, got {out.shape}'\n"
            "    # Well-conditioned slots: actual linear-system solutions.\n"
            "    assert t.allclose(out[0], t.tensor([2.0, 3.0]), atol=1e-5), f'slot 0 wrong: {out[0]}'\n"
            "    assert t.allclose(out[2], t.tensor([1.0, 2.0]), atol=1e-5), f'slot 2 wrong: {out[2]}'\n"
            "    # Singular slot: zero vector (not NaN, not error).\n"
            "    assert t.allclose(out[1], t.zeros(2)), f'singular slot must be zero, got {out[1]}'\n"
            "    assert not t.isnan(out).any(), 'output must contain no NaN'\n"
            "    # Non-mutation.\n"
            "    assert t.allclose(A, A_orig), 'A was mutated'\n"
            "    assert t.allclose(b, b_orig), 'b was mutated'\n"
            "    # No-singular case — output must match a plain solve.\n"
            "    A2 = t.tensor([[[3.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]])\n"
            "    b2 = t.tensor([[6.0, 4.0], [5.0, 7.0]])\n"
            "    out2 = ex5_safe_solve(A2, b2)\n"
            "    expected2 = t.linalg.solve(A2, b2.unsqueeze(-1)).squeeze(-1)\n"
            "    assert t.allclose(out2, expected2, atol=1e-5), 'no-singular case must match plain solve'"
        ),
        "solution_body": (
            "def ex5_safe_solve(A: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:\n"
            "    dets = t.det(A)\n"
            "    singular = dets.abs() < eps\n"
            "    A_safe = A.clone()\n"
            "    b_safe = b.clone()\n"
            "    N = A.shape[-1]\n"
            "    A_safe[singular] = t.eye(N, dtype=A.dtype)\n"
            "    b_safe[singular] = 0.0\n"
            "    sol = t.linalg.solve(A_safe, b_safe.unsqueeze(-1)).squeeze(-1)\n"
            "    return sol"
        ),
        "solution_notes": (
            "**Why this matters in Ray Tracing.** Ray-triangle intersection "
            "boils down to a 3×3 linear system per ray (the Möller-Trumbore "
            "matrix). Rays parallel to a triangle's plane give a singular "
            "matrix — and if you call `torch.linalg.solve` on the batch, "
            "ONE singular system crashes the WHOLE call. Substituting identity "
            "in `A` and zero in `b` lets the batched solve complete; the "
            "originally-singular slots come out as zeros, which the caller "
            "treats as 'no intersection'.\n\n"
            "**Why solve on `b.unsqueeze(-1)` not `b` directly?** `torch.linalg.solve(A, b)` "
            "accepts both `(B, N)` and `(B, N, K)` for `b`. The `.unsqueeze(-1)` "
            "form is explicit and consistent across PyTorch versions; we squeeze "
            "the trailing 1 to get back to `(B, N)`.\n\n"
            "**Why `eps = 1e-6` and not `1e-12`?** Determinants of `N×N` matrices "
            "scale with the magnitude of entries — a near-singular but numerically "
            "small `det` can mean a poorly-conditioned, not actually-singular, "
            "matrix. `1e-6` is a conservative threshold for float32 work; tune "
            "based on your expected input magnitudes."
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
    "**What you'll practice.** Five mask-and-substitute patterns that ramp from `x < 0` → clamp → row-zero → identity-substitute → safe batched solve. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
    "",
    "**Per-exercise structure** (Doughty et al. ACE 2024 — `[Bloom level] + [LO] + [Keywords] + [KCs]`):",
    "Each exercise begins with a yaml block stating its Bloom cognitive level, learning objective, keywords, and the knowledge components (KCs) it targets. This makes the cognitive demand explicit instead of buried.",
))

cells.append(md("## Setup"))

cells.append(code(
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
    "## Mask & substitute — quick refresher",
    "",
    "**Build a mask.** Any comparison returns a `dtype=bool` tensor with the same shape as its input: `x < 0`, `x.abs() < eps`, `(x > 0) & (x < 1)`.",
    "",
    "**Write through a mask.** `y[mask] = value` modifies in place. If `value` is a scalar, it broadcasts over the masked region. If `value` is a tensor, its shape must match the shape of `y[mask]` after broadcasting.",
    "",
    "**Always clone first** if you want a non-mutating function — `y = x.clone(); y[mask] = 0; return y`. Otherwise the caller's input gets clobbered.",
    "",
    "**Identity substitute.** `A[singular_mask] = torch.eye(N)` replaces flagged `(N, N)` submatrices with the identity — the standard cleanup before a batched `linalg.solve` so one degenerate slice doesn't crash the whole batch.",
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
                "ex5": "4-KC integrative exercise — past Tutor-Kai solvability cliff. Track student pass rate; do not treat ex5 failure alone as evidence of atom non-mastery.",
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
