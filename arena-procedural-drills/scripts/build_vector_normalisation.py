#!/usr/bin/env python3
"""Build the vector-normalisation procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/vector-normalisation.ipynb`
— the first non-einops drill (Phase 4.5). Mirrors the v0.2 template
established by `build_einops_rearrange.py`. Inherits the verify_solutions
gate.

Atom `vector-normalisation` lives at the heart of Ray Tracing (surface
normals must be unit length) but the skill itself is pure NumPy/PyTorch.
Bridges to the bank topic `Numpy` via the explicit token rule in
atom_readiness.js.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/vector-normalisation.ipynb"


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


ATOM_ID = "vector-normalisation"
SUBTOPIC = "Numpy: Applied patterns and advanced"
TITLE = "vector normalisation — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "l2-norm-compute",
        "kind": "component-skill",
        "description": "Compute the Euclidean (L2) norm of a 1-D tensor. Verifies the basic `.norm()` or `pow(2).sum().sqrt()` call.",
    },
    {
        "id": "unit-vector-divide",
        "kind": "component-skill",
        "description": "Divide a vector by its scalar norm to produce a unit vector. Verifies the division shape and that the result has norm 1.",
    },
    {
        "id": "norm-along-axis-keepdim",
        "kind": "component-skill",
        "description": "Compute the norm along a specific axis with `keepdim=True` so the divisor broadcasts back against the input. The shape-preservation trick that lets you normalize a batch in one line.",
    },
    {
        "id": "per-row-batch-normalize",
        "kind": "component-skill",
        "description": "Normalize every row of a 2-D tensor to unit length using the axis-keepdim pattern. The canonical 'normalize a batch of surface normals' operation.",
    },
    {
        "id": "safe-zero-norm-normalize",
        "kind": "integrative-skill",
        "description": "Combine axis-keepdim normalize with an `eps` floor so zero-norm vectors don't produce NaN. The integrative KC — relies on all four component KCs above plus numerical-stability instinct.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "L2 norm of a 1-D vector",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["l2-norm", "scalar-output", "euclidean"],
        "kcs": ["l2-norm-compute"],
        "lo": "Recall how to compute the Euclidean norm of a 1-D tensor.",
        "prompt_body": (
            "Implement `ex1_l2_norm(v)` to compute the Euclidean (L2) norm of a 1-D "
            "tensor: `sqrt(sum(v_i^2))`. Output is a 0-D scalar tensor.\n\n"
            "Use `torch.linalg.norm`, `torch.norm`, or `v.pow(2).sum().sqrt()` — any "
            "of the three is fine."
        ),
        "stub": (
            "def ex1_l2_norm(v: Tensor) -> Tensor:\n"
            '    """L2 norm of a 1-D tensor. Returns a 0-D scalar."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "v = t.tensor([3.0, 4.0])                             # known norm = 5\n"
            "    n = ex1_l2_norm(v)\n"
            "    assert n.dim() == 0, f'expected scalar (0-D), got shape {tuple(n.shape)}'\n"
            "    assert t.allclose(n, t.tensor(5.0)), f'expected 5.0, got {n.item()}'\n"
            "\n"
            "    v2 = t.tensor([1.0, 0.0, 0.0])                   # unit-x\n"
            "    assert t.allclose(ex1_l2_norm(v2), t.tensor(1.0)), 'unit-x norm should be 1'"
        ),
        "solution_body": (
            "def ex1_l2_norm(v: Tensor) -> Tensor:\n"
            "    return v.pow(2).sum().sqrt()"
        ),
        "solution_notes": (
            "**Three equivalent calls:** `v.norm()`, `torch.linalg.norm(v)`, "
            "`v.pow(2).sum().sqrt()`. The first two are concise; the third makes "
            "the formula explicit and is what `torch.norm` does under the hood."
        ),
    },
    {
        "id": "ex2",
        "title": "normalize a 1-D vector to unit length",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["unit-vector", "scalar-division", "normalization"],
        "kcs": ["unit-vector-divide"],
        "lo": "Apply scalar division by the norm to produce a unit vector from a 1-D tensor.",
        "prompt_body": (
            "Implement `ex2_unit_vector(v)` to return `v / ||v||`. Output shape "
            "matches the input shape, and the result must have norm 1.\n\n"
            "Don't use `torch.nn.functional.normalize` — write the division explicitly. "
            "The point is to see the shape arithmetic (scalar division)."
        ),
        "stub": (
            "def ex2_unit_vector(v: Tensor) -> Tensor:\n"
            '    """Return v / ||v||. Output shape matches v; result has norm 1."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "v = t.tensor([3.0, 4.0])\n"
            "    u = ex2_unit_vector(v)\n"
            "    assert u.shape == v.shape, f'shape mismatch: {u.shape} vs {v.shape}'\n"
            "    assert t.allclose(u, t.tensor([0.6, 0.8])), f'expected [0.6, 0.8], got {u.tolist()}'\n"
            "    assert t.allclose(u.norm(), t.tensor(1.0)), 'unit-vector norm should be 1.0'"
        ),
        "solution_body": (
            "def ex2_unit_vector(v: Tensor) -> Tensor:\n"
            "    return v / v.pow(2).sum().sqrt()"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex3",
        "title": "norm along an axis with keepdim",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["axis-aware", "keepdim", "broadcast-prep"],
        "kcs": ["norm-along-axis-keepdim"],
        "lo": "Apply `dim=` and `keepdim=True` to compute per-row norms shaped for broadcast.",
        "prompt_body": (
            "Implement `ex3_row_norms_keepdim(x)` to compute the L2 norm of every "
            "row of a 2-D tensor, **keeping** the reduced axis as size 1.\n\n"
            "Input shape: `(N, D)`. Output shape: `(N, 1)` — NOT `(N,)`. The "
            "size-1 axis lets the result broadcast back against the input.\n\n"
            "Use `keepdim=True` (or `(...,)` axis bookkeeping by hand)."
        ),
        "stub": (
            "def ex3_row_norms_keepdim(x: Tensor) -> Tensor:\n"
            '    """Per-row L2 norm with keepdim. (N, D) → (N, 1)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([[3.0, 4.0], [0.0, 5.0], [1.0, 2.0]])  # norms: 5, 5, sqrt(5)\n"
            "    n = ex3_row_norms_keepdim(x)\n"
            "    assert n.shape == (3, 1), f'expected (3,1), got {tuple(n.shape)}'\n"
            "    expected = t.tensor([[5.0], [5.0], [(5.0)**0.5]])\n"
            "    assert t.allclose(n, expected), f'value mismatch: {n.tolist()}'\n"
            "    # Critical broadcast check: (3, D) / (3, 1) must work without error.\n"
            "    broadcast_ok = x / n\n"
            "    assert broadcast_ok.shape == x.shape, 'keepdim should let result broadcast against input'"
        ),
        "solution_body": (
            "def ex3_row_norms_keepdim(x: Tensor) -> Tensor:\n"
            "    return x.pow(2).sum(dim=1, keepdim=True).sqrt()"
        ),
        "solution_notes": (
            "**Why `keepdim=True`?** Without it the result is shape `(N,)`, which "
            "does NOT broadcast back against `(N, D)` for division — PyTorch would "
            "right-align `(N,)` against `(N, D)`'s last dim and complain. With "
            "`keepdim=True` you get `(N, 1)` which broadcasts cleanly."
        ),
    },
    {
        "id": "ex4",
        "title": "normalize every row of a batch",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["batch-normalize", "surface-normals", "ray-tracing"],
        "kcs": ["per-row-batch-normalize"],
        "lo": "Apply per-row normalization to a batch of vectors using axis-keepdim broadcast.",
        "prompt_body": (
            "Implement `ex4_batch_unit_vectors(x)` to L2-normalize every row of a "
            "batch.\n\n"
            "Input shape: `(N, D)`. Output shape: `(N, D)`. Every row of the "
            "output should be a unit vector.\n\n"
            "This is the canonical operation for converting a batch of raw surface "
            "normals into the unit normals needed for shading. Equivalent to "
            "`torch.nn.functional.normalize(x, dim=1)` — write it explicitly using "
            "the keepdim norm from Exercise 3."
        ),
        "stub": (
            "def ex4_batch_unit_vectors(x: Tensor) -> Tensor:\n"
            '    """Per-row L2-normalize. (N, D) → (N, D), every row has norm 1."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn.functional as F\n"
            "    x = t.tensor([[3.0, 4.0], [0.0, 5.0], [1.0, 2.0], [-1.0, -1.0]])\n"
            "    u = ex4_batch_unit_vectors(x)\n"
            "    assert u.shape == x.shape, f'shape mismatch: {u.shape} vs {x.shape}'\n"
            "    row_norms = u.pow(2).sum(dim=1).sqrt()\n"
            "    assert t.allclose(row_norms, t.ones(4), atol=1e-6), f'rows should have norm 1, got {row_norms.tolist()}'\n"
            "    assert t.allclose(u, F.normalize(x, dim=1), atol=1e-6), 'should match F.normalize(x, dim=1)'"
        ),
        "solution_body": (
            "def ex4_batch_unit_vectors(x: Tensor) -> Tensor:\n"
            "    return x / x.pow(2).sum(dim=1, keepdim=True).sqrt()"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex5",
        "title": "safe normalize with epsilon (zero-norm guard)",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["numerical-stability", "epsilon", "nan-guard", "multi-kc"],
        "kcs": [
            "norm-along-axis-keepdim",
            "per-row-batch-normalize",
            "safe-zero-norm-normalize",
        ],
        "lo": "Synthesize axis-keepdim norm + per-row divide + epsilon floor to normalize safely even when some rows are zero vectors.",
        "prompt_body": (
            "Implement `ex5_safe_normalize(x, eps=1e-12)` to L2-normalize every row "
            "of `x` **without producing NaN** for zero-norm rows.\n\n"
            "Input shape: `(N, D)`. Output shape: `(N, D)`. Use `eps` as the floor "
            "for the divisor: `divisor = max(norm, eps)`. Zero-norm rows then come "
            "out as zero rows (not NaN, not inf).\n\n"
            "Strategy: `torch.clamp(norm, min=eps)` is the clean way. Don't add "
            "`eps` inside the sqrt — that quietly inflates non-zero norms.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3+ KCs in one expression; "
            "empirical work (Lohr et al. ITiCSE 2025) shows 3-concept LLM-generated "
            "exercises drop from ~94% to ~40% solvability. Expect a step in "
            "difficulty here vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_safe_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:\n"
            '    """Per-row L2 normalize with eps floor on the divisor.\n'
            "\n"
            "    Zero-norm rows return as zero rows (no NaN).\n"
            '    """\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([\n"
            "        [3.0, 4.0],     # norm 5\n"
            "        [0.0, 0.0],     # zero vector — must NOT produce NaN\n"
            "        [1.0, 0.0],     # already unit\n"
            "        [-2.0, 0.0],    # norm 2\n"
            "    ])\n"
            "    u = ex5_safe_normalize(x)\n"
            "    assert u.shape == x.shape, f'shape mismatch: {u.shape}'\n"
            "    assert not t.isnan(u).any(), 'output contains NaN — eps floor not applied'\n"
            "    assert not t.isinf(u).any(), 'output contains inf — eps floor not applied'\n"
            "    # Non-zero rows must be properly normalized.\n"
            "    assert t.allclose(u[0], t.tensor([0.6, 0.8]), atol=1e-6), 'row 0 should be [0.6, 0.8]'\n"
            "    assert t.allclose(u[1], t.zeros(2)), 'zero-input row should be zero-output row'\n"
            "    assert t.allclose(u[2], t.tensor([1.0, 0.0])), 'unit-x row should stay unit-x'\n"
            "    assert t.allclose(u[3], t.tensor([-1.0, 0.0])), 'row 3 should be [-1, 0]'"
        ),
        "solution_body": (
            "def ex5_safe_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:\n"
            "    norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()\n"
            "    return x / t.clamp(norm, min=eps)"
        ),
        "solution_notes": (
            "**Why `clamp` and not `norm + eps`?**\n"
            "- `norm + eps`: the divisor is always at least `eps`, so a vector of "
            "norm 1.0 gets divided by `1.0 + 1e-12` — a tiny but real bias.\n"
            "- `clamp(norm, min=eps)`: the divisor is `norm` for any norm ≥ eps and "
            "`eps` only for zero-norm rows. No bias for well-conditioned vectors.\n\n"
            "**Why eps=1e-12 in float32?** float32 machine epsilon is ~1.19e-7. Any "
            "non-zero norm we care about is many orders of magnitude above 1e-12, "
            "so `clamp` only kicks in for genuine zeros. Lower than ~1e-20 risks "
            "the divisor itself underflowing."
        ),
    },
]


# ----- cell builders (same as einops drills) ----------------------------


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
    "**What you'll practice.** Five vector-normalization patterns that ramp from L2 norm → unit vector → axis-aware keepdim → batch normalize → safe normalize with epsilon floor. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "## Vector normalisation — quick refresher",
    "",
    "**L2 norm:** `||v|| = sqrt(sum_i v_i^2)`. In PyTorch:",
    "- `v.norm()` / `torch.linalg.norm(v)` — for any tensor.",
    "- `v.pow(2).sum().sqrt()` — same thing, explicit.",
    "",
    "**Per-row normalize:** `x / x.pow(2).sum(dim=1, keepdim=True).sqrt()`.",
    "The `keepdim=True` is critical — it keeps the reduced axis as size 1 so the divisor broadcasts back.",
    "",
    "**Safe normalize:** wrap the divisor in `torch.clamp(norm, min=eps)` to avoid NaN on zero-norm rows.",
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
