#!/usr/bin/env python3
"""Build the tensor-unbind procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/tensor-unbind.ipynb`
(Phase 4.10). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `tensor-unbind` is the canonical 'peel a stacked axis into a tuple of
slices' op. In Ray Tracing it splits a `(N, 2, 3)` rays tensor into
`(origin, direction)` along dim=1; in many ARENA exercises it replaces a
clumsy chain of `x[:, 0]`-style picks. Bridges to bank topic `Numpy` via
the explicit token rule (`tensor-unbind`) in atom_readiness.js, reporting
to subtopic `Numpy: Indexing and selection`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/tensor-unbind.ipynb"


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


ATOM_ID = "tensor-unbind"
SUBTOPIC = "Numpy: Indexing and selection"
TITLE = "tensor unbind — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "unbind-default-dim",
        "kind": "component-skill",
        "description": "Call `torch.unbind(x)` (default `dim=0`). Returns a tuple of `x.shape[0]` tensors, each with one fewer dim than `x`.",
    },
    {
        "id": "unbind-explicit-dim",
        "kind": "component-skill",
        "description": "Call `torch.unbind(x, dim=k)` to peel along an arbitrary axis. Returns `x.shape[k]` slices, each missing axis `k`.",
    },
    {
        "id": "unbind-equiv-to-slice",
        "kind": "component-skill",
        "description": "Recognize that `unbind(x, dim=k)[i]` equals `x.select(k, i)` — i.e. the same data, just packaged as a tuple. No copy; views of the original storage.",
    },
    {
        "id": "unbind-tuple-destructure",
        "kind": "component-skill",
        "description": "Use Python tuple unpacking to destructure unbind output: `a, b, c = x.unbind(0)`. The idiomatic way to extract a fixed small number of named slices.",
    },
    {
        "id": "unbind-ray-decomposition",
        "kind": "integrative-skill",
        "description": "Decompose a `(N, 2, 3)` rays tensor into `origin, direction` along `dim=1`, then evaluate `origin + t * direction` at a given parameter. The canonical Ray Tracing use of unbind.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "unbind along default dim 0",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["torch-unbind", "default-dim", "tuple-of-slices"],
        "kcs": ["unbind-default-dim"],
        "lo": "Recall that `torch.unbind(x)` peels along `dim=0` into a tuple of `x.shape[0]` slices.",
        "prompt_body": (
            "Implement `ex1_unbind_rows(x)` to return a tuple of 1-D tensors — "
            "one for each row of the 2-D input `x`. Use `torch.unbind` with the "
            "default `dim=0`.\n\n"
            "Input shape: `(R, C)`. Output: a tuple of `R` tensors, each of "
            "shape `(C,)`."
        ),
        "stub": (
            "def ex1_unbind_rows(x: Tensor) -> tuple:\n"
            '    """Unbind a 2-D tensor along dim 0 → tuple of rows."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([[1.0, 2.0, 3.0],\n"
            "                  [4.0, 5.0, 6.0],\n"
            "                  [7.0, 8.0, 9.0]])\n"
            "    out = ex1_unbind_rows(x)\n"
            "    assert isinstance(out, tuple), f'expected tuple, got {type(out).__name__}'\n"
            "    assert len(out) == 3, f'expected 3 rows, got {len(out)}'\n"
            "    for i, row in enumerate(out):\n"
            "        assert row.shape == (3,), f'row {i} shape {row.shape}, expected (3,)'\n"
            "        assert t.allclose(row, x[i]), f'row {i} mismatch'"
        ),
        "solution_body": (
            "def ex1_unbind_rows(x: Tensor) -> tuple:\n"
            "    return t.unbind(x)"
        ),
        "solution_notes": (
            "**Default `dim=0`.** `t.unbind(x)` is shorthand for `t.unbind(x, dim=0)`. "
            "The result is a tuple — NOT a list, NOT a tensor — because unbind "
            "always returns a Python sequence of views over storage, never a "
            "single tensor."
        ),
    },
    {
        "id": "ex2",
        "title": "unbind along an explicit axis",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["dim-arg", "column-unbind", "axis-aware"],
        "kcs": ["unbind-explicit-dim"],
        "lo": "Apply the `dim=` argument to unbind along a non-default axis.",
        "prompt_body": (
            "Implement `ex2_unbind_columns(x)` to return a tuple of column tensors "
            "from a 2-D input.\n\n"
            "Input shape: `(R, C)`. Output: a tuple of `C` tensors, each of "
            "shape `(R,)`. Use `dim=1`."
        ),
        "stub": (
            "def ex2_unbind_columns(x: Tensor) -> tuple:\n"
            '    """Unbind a 2-D tensor along dim 1 → tuple of columns."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([[1.0, 2.0, 3.0],\n"
            "                  [4.0, 5.0, 6.0]])\n"
            "    out = ex2_unbind_columns(x)\n"
            "    assert isinstance(out, tuple), f'expected tuple, got {type(out).__name__}'\n"
            "    assert len(out) == 3, f'expected 3 columns, got {len(out)}'\n"
            "    for j, col in enumerate(out):\n"
            "        assert col.shape == (2,), f'col {j} shape {col.shape}, expected (2,)'\n"
            "        assert t.allclose(col, x[:, j]), f'col {j} mismatch'"
        ),
        "solution_body": (
            "def ex2_unbind_columns(x: Tensor) -> tuple:\n"
            "    return t.unbind(x, dim=1)"
        ),
        "solution_notes": (
            "**`dim=1` on a `(R, C)` tensor** peels along the column axis, "
            "yielding `C` tensors each of shape `(R,)`. The shape drops the "
            "unbound axis — `(R, C)` → `C` × `(R,)`."
        ),
    },
    {
        "id": "ex3",
        "title": "unbind equivalence with `.select`",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["select", "equivalence", "view-semantics"],
        "kcs": ["unbind-equiv-to-slice"],
        "lo": "Apply the equivalence `unbind(x, dim=k)[i] == x.select(k, i)` to pick a single slice via `select`.",
        "prompt_body": (
            "Implement `ex3_select_via_unbind(x, dim, i)`: return the `i`-th slice "
            "along `dim`, using only `torch.unbind` and Python tuple indexing.\n\n"
            "Functionally equivalent to `x.select(dim, i)` — the point is to make "
            "the equivalence concrete.\n\n"
            "Inputs: any-shape tensor `x`, integer `dim`, integer `i` in `[0, x.shape[dim])`."
        ),
        "stub": (
            "def ex3_select_via_unbind(x: Tensor, dim: int, i: int) -> Tensor:\n"
            '    """Return the i-th slice along `dim`, implemented via unbind."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.randn(4, 5, 6)\n"
            "    assert t.allclose(ex3_select_via_unbind(x, 0, 2), x.select(0, 2))\n"
            "    assert t.allclose(ex3_select_via_unbind(x, 1, 0), x.select(1, 0))\n"
            "    assert t.allclose(ex3_select_via_unbind(x, 2, 5), x.select(2, 5))\n"
            "    # Output shape must drop the selected axis.\n"
            "    out = ex3_select_via_unbind(x, 1, 3)\n"
            "    assert out.shape == (4, 6), f'expected (4,6), got {tuple(out.shape)}'"
        ),
        "solution_body": (
            "def ex3_select_via_unbind(x: Tensor, dim: int, i: int) -> Tensor:\n"
            "    return t.unbind(x, dim=dim)[i]"
        ),
        "solution_notes": (
            "**Why this matters.** `unbind` is a *view* op — each output tensor "
            "shares storage with `x`. It's not slower than `select` for picking "
            "one slice; the cost is the Python tuple construction. Prefer `select` "
            "when you want ONE slice; prefer `unbind` when you want ALL slices "
            "(so you don't loop with `select` N times)."
        ),
    },
    {
        "id": "ex4",
        "title": "destructure a fixed number of slices",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["destructure", "xyz", "fixed-arity"],
        "kcs": ["unbind-tuple-destructure"],
        "lo": "Apply tuple unpacking to give names to the components of a small-axis unbind.",
        "prompt_body": (
            "Implement `ex4_xyz_components(v)`: given a 1-D 3-vector `v`, return "
            "`x + y + z` (the sum of its three components), but extract them "
            "via tuple unpacking from `v.unbind()`.\n\n"
            "Example: `v = tensor([1.0, 2.0, 3.0])` → `x, y, z = v.unbind()` → "
            "return `x + y + z` = 6.0 (as a 0-D tensor).\n\n"
            "The point is the idiom `x, y, z = v.unbind()` — replaces the clunky "
            "`v[0], v[1], v[2]` form when you want named scalar slices."
        ),
        "stub": (
            "def ex4_xyz_components(v: Tensor) -> Tensor:\n"
            '    """Destructure a 3-vector via unbind, return x + y + z."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "v = t.tensor([1.0, 2.0, 3.0])\n"
            "    out = ex4_xyz_components(v)\n"
            "    assert out.dim() == 0, f'expected scalar (0-D), got shape {tuple(out.shape)}'\n"
            "    assert t.allclose(out, t.tensor(6.0)), f'expected 6.0, got {out.item()}'\n"
            "\n"
            "    v2 = t.tensor([-1.0, 0.5, 2.5])\n"
            "    assert t.allclose(ex4_xyz_components(v2), t.tensor(2.0))"
        ),
        "solution_body": (
            "def ex4_xyz_components(v: Tensor) -> Tensor:\n"
            "    x, y, z = v.unbind()\n"
            "    return x + y + z"
        ),
        "solution_notes": (
            "**Why named destructure helps.** Adding three components is trivial "
            "either way, but consider Phong shading: `ka * Ia + kd * Id * dot(N, L) + "
            "ks * Is * dot(R, V)^n`. Having `N, L, R, V = vectors.unbind()` early "
            "makes the formula readable; threading `vectors[0], vectors[1], ...` "
            "through obscures the math."
        ),
    },
    {
        "id": "ex5",
        "title": "decompose rays, evaluate at parameter t",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["ray-tracing", "origin-direction", "param-evaluate", "multi-kc"],
        "kcs": [
            "unbind-explicit-dim",
            "unbind-tuple-destructure",
            "unbind-ray-decomposition",
        ],
        "lo": "Synthesize axis-explicit unbind + tuple destructure to split a rays tensor and evaluate the parametric ray equation.",
        "prompt_body": (
            "Implement `ex5_evaluate_rays(rays, t_param)`. The canonical Ray Tracing "
            "use of `unbind`:\n\n"
            "Input `rays`: shape `(N, 2, 3)`. Each ray is a `(2, 3)` block where "
            "row 0 is the origin and row 1 is the direction (in 3-D).\n\n"
            "Output: `(N, 3)` — for each ray, return `origin + t_param * direction`.\n\n"
            "Implementation: use `rays.unbind(dim=1)` to peel out origin and "
            "direction as `(N, 3)` tensors, then return the parametric evaluation.\n\n"
            "> ⚠️ **Integrative exercise.** Combines 3 KCs (explicit dim, tuple "
            "destructure, ray decomposition). Empirical work (Lohr et al. ITiCSE "
            "2025) shows 3-concept exercises drop to ~40% solvability — expect "
            "a step up vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_evaluate_rays(rays: Tensor, t_param: float) -> Tensor:\n"
            '    """Given rays of shape (N, 2, 3) where rays[i] = [origin, direction],\n'
            "    return origin + t_param * direction for each ray. Output: (N, 3).\n"
            '    """\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rays = t.tensor([\n"
            "        # ray 0 — at origin, direction +x\n"
            "        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],\n"
            "        # ray 1 — at (1, 2, 3), direction +y\n"
            "        [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]],\n"
            "        # ray 2 — at (0, 0, 1), direction (1, 1, 0)\n"
            "        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],\n"
            "    ])\n"
            "    out = ex5_evaluate_rays(rays, t_param=2.0)\n"
            "    assert out.shape == (3, 3), f'expected (3,3), got {tuple(out.shape)}'\n"
            "    expected = t.tensor([\n"
            "        [2.0, 0.0, 0.0],   # 0 + 2*x = (2, 0, 0)\n"
            "        [1.0, 4.0, 3.0],   # (1,2,3) + 2*(0,1,0) = (1, 4, 3)\n"
            "        [2.0, 2.0, 1.0],   # (0,0,1) + 2*(1,1,0) = (2, 2, 1)\n"
            "    ])\n"
            "    assert t.allclose(out, expected), f'value mismatch:\\n{out}\\nvs\\n{expected}'\n"
            "    # t=0 should return origins unchanged.\n"
            "    out0 = ex5_evaluate_rays(rays, t_param=0.0)\n"
            "    assert t.allclose(out0, rays[:, 0, :]), 't=0 must return origins'"
        ),
        "solution_body": (
            "def ex5_evaluate_rays(rays: Tensor, t_param: float) -> Tensor:\n"
            "    origin, direction = rays.unbind(dim=1)\n"
            "    return origin + t_param * direction"
        ),
        "solution_notes": (
            "**The parametric ray equation: `r(t) = o + t·d`.** This is the most "
            "common Ray Tracing computation — given an array of rays and a hit "
            "parameter, find the world-space hit point. `unbind(dim=1)` is the "
            "cleanest split: `origin` and `direction` come out as named `(N, 3)` "
            "tensors with no shape arithmetic.\n\n"
            "**Compare:** `rays[:, 0, :] + t * rays[:, 1, :]` works and is "
            "equivalent, but the unbind version reads like math. Prefer it when "
            "your variables have physical meaning."
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
    "**What you'll practice.** Five unbind patterns that ramp from default-dim → explicit-dim → equivalence-with-`select` → tuple-destructure → ray-equation evaluation. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "## Unbind — quick refresher",
    "",
    "**What it does.** `torch.unbind(x, dim=k)` returns a tuple of `x.shape[k]` tensors, each with axis `k` removed. The result is a *Python tuple* (not a tensor) of *views* (no copy).",
    "",
    "**Default dim is 0.** `x.unbind()` peels along axis 0; `x.unbind(dim=1)` peels along axis 1.",
    "",
    "**Idiomatic destructure.** `origin, direction = rays.unbind(dim=1)` is the canonical way to split a `(N, 2, 3)` rays tensor into two `(N, 3)` named components.",
    "",
    "**Equivalence.** `x.unbind(dim=k)[i]` == `x.select(k, i)`. Prefer `select` for picking ONE slice; prefer `unbind` when you want ALL slices.",
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
