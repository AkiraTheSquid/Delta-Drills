#!/usr/bin/env python3
"""Build the broadcasting-rules procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/broadcasting-rules.ipynb`
(Phase 4.7). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `broadcasting-rules` is the prereq under almost every Numpy/PyTorch
operation — get the shape arithmetic wrong and everything downstream
silently produces the wrong tensor. Bridges to the bank topic `Numpy` via
the explicit token rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/broadcasting-rules.ipynb"


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


ATOM_ID = "broadcasting-rules"
SUBTOPIC = "Numpy: Vectorization and broadcasting"
TITLE = "broadcasting rules — procedural drill"
TEMPLATE_VERSION = "v0.2"

KC_DECOMPOSITION = [
    {
        "id": "predict-broadcast-shape",
        "kind": "component-skill",
        "description": "Given two shapes, predict the broadcasted result shape (or detect incompatibility). The literal mechanical rule that all broadcasting depends on.",
    },
    {
        "id": "broadcast-row-vector",
        "kind": "component-skill",
        "description": "Add/multiply a 1-D row vector of shape `(D,)` against a 2-D matrix of shape `(N, D)`. The most common broadcasting case in practice.",
    },
    {
        "id": "broadcast-column-vector",
        "kind": "component-skill",
        "description": "Add/multiply a 2-D column vector of shape `(N, 1)` against a 2-D matrix of shape `(N, D)`. Requires explicit `unsqueeze(1)` or `[:, None]` to keep the trailing axis as size 1.",
    },
    {
        "id": "broadcast-via-unsqueeze",
        "kind": "component-skill",
        "description": "Use `unsqueeze` / `[:, None]` to insert a size-1 axis exactly where broadcasting needs it. The fix for almost every 'shape mismatch' error in deep-learning code.",
    },
    {
        "id": "broadcast-outer-product",
        "kind": "integrative-skill",
        "description": "Combine column-vector broadcast + row-vector broadcast to produce an outer product `u_i * v_j` of shape `(N, M)` from two 1-D inputs. The integrative KC — relies on all four component KCs.",
    },
]

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "predict the broadcast shape",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["shape-rule", "right-align", "incompatibility"],
        "kcs": ["predict-broadcast-shape"],
        "lo": "Recall and apply the right-align broadcasting rule on two shape tuples.",
        "prompt_body": (
            "Implement `ex1_broadcast_shape(shape_a, shape_b)` to return the "
            "shape that would result from broadcasting two tensors of the given "
            "shapes, OR raise `ValueError` if they're incompatible.\n\n"
            "**The rule** (NumPy & PyTorch agree):\n"
            "1. Right-align both shapes, left-pad the shorter with 1s.\n"
            "2. For each pair `(a, b)` of aligned axes: keep if equal; if exactly "
            "one is 1, use the other; otherwise → incompatible.\n\n"
            "Return the result as a tuple of ints.\n\n"
            "**Examples:**\n"
            "- `(3, 4)` and `(4,)` → `(3, 4)`\n"
            "- `(2, 1, 3)` and `(5, 3)` → `(2, 5, 3)`\n"
            "- `(3, 4)` and `(3,)` → `ValueError` (right-align mismatch on last axis)"
        ),
        "stub": (
            "def ex1_broadcast_shape(shape_a, shape_b):\n"
            '    """Return broadcasted shape (tuple) or raise ValueError if incompatible."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Compatible cases\n"
            "    assert ex1_broadcast_shape((3, 4), (4,)) == (3, 4)\n"
            "    assert ex1_broadcast_shape((3, 4), (1, 4)) == (3, 4)\n"
            "    assert ex1_broadcast_shape((3, 1), (4,)) == (3, 4)\n"
            "    assert ex1_broadcast_shape((2, 1, 3), (5, 3)) == (2, 5, 3)\n"
            "    assert ex1_broadcast_shape((1,), (5, 6, 7)) == (5, 6, 7)\n"
            "    assert ex1_broadcast_shape((), (5,)) == (5,)\n"
            "\n"
            "    # Incompatible cases — must raise ValueError\n"
            "    raised = False\n"
            "    try:\n"
            "        ex1_broadcast_shape((3, 4), (3,))   # right-align: (3,4) vs (1,3) → mismatch at axis -1\n"
            "    except ValueError:\n"
            "        raised = True\n"
            "    assert raised, 'should have raised ValueError for incompatible shapes (3,4) vs (3,)'\n"
            "\n"
            "    raised = False\n"
            "    try:\n"
            "        ex1_broadcast_shape((2, 3), (4, 3))\n"
            "    except ValueError:\n"
            "        raised = True\n"
            "    assert raised, 'should have raised ValueError for (2,3) vs (4,3)'"
        ),
        "solution_body": (
            "def ex1_broadcast_shape(shape_a, shape_b):\n"
            "    a = list(shape_a)\n"
            "    b = list(shape_b)\n"
            "    n = max(len(a), len(b))\n"
            "    a = [1] * (n - len(a)) + a\n"
            "    b = [1] * (n - len(b)) + b\n"
            "    out = []\n"
            "    for ai, bi in zip(a, b):\n"
            "        if ai == bi:\n"
            "            out.append(ai)\n"
            "        elif ai == 1:\n"
            "            out.append(bi)\n"
            "        elif bi == 1:\n"
            "            out.append(ai)\n"
            "        else:\n"
            "            raise ValueError(f'incompatible axes: {ai} vs {bi}')\n"
            "    return tuple(out)"
        ),
        "solution_notes": (
            "**Why right-align?** Trailing axes correspond to the fastest-varying "
            "memory layout. Aligning shapes from the right means a `(D,)` vector "
            "broadcasts across rows of an `(N, D)` matrix, which is the natural "
            "'one weight per feature' case. Left-align would have given you "
            "'one weight per sample' instead — a different, much less common need."
        ),
    },
    {
        "id": "ex2",
        "title": "row-vector broadcast (add bias to a batch)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["row-broadcast", "bias", "per-feature"],
        "kcs": ["broadcast-row-vector"],
        "lo": "Apply 1-D broadcast across the leading axis of a 2-D batch (per-feature bias).",
        "prompt_body": (
            "Implement `ex2_add_bias(x, b)` to add a per-feature bias to a batch.\n\n"
            "Input shapes: `x` is `(N, D)`, `b` is `(D,)`. Output shape: `(N, D)`. "
            "Every row of the output should equal `x[n] + b`.\n\n"
            "This is just `x + b` — but the point is to recognize that right-align "
            "broadcasting handles it for you. No `unsqueeze`, no manual broadcast — "
            "the rule does the work."
        ),
        "stub": (
            "def ex2_add_bias(x: Tensor, b: Tensor) -> Tensor:\n"
            '    """Add per-feature bias. (N, D) + (D,) → (N, D)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3).reshape(2, 3).float()       # (2, 3)\n"
            "    b = t.tensor([10.0, 20.0, 30.0])                # (3,)\n"
            "    y = ex2_add_bias(x, b)\n"
            "    assert y.shape == (2, 3), f'expected (2,3), got {tuple(y.shape)}'\n"
            "    expected = t.tensor([[10.0, 21.0, 32.0], [13.0, 24.0, 35.0]])\n"
            "    assert t.equal(y, expected), f'value mismatch: {y.tolist()}'"
        ),
        "solution_body": (
            "def ex2_add_bias(x: Tensor, b: Tensor) -> Tensor:\n"
            "    return x + b"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex3",
        "title": "column-vector broadcast (per-sample scale)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["column-broadcast", "per-sample", "unsqueeze"],
        "kcs": ["broadcast-column-vector", "broadcast-via-unsqueeze"],
        "lo": "Apply `unsqueeze(1)` / `[:, None]` to make a 1-D per-sample weight broadcast across the feature axis of a 2-D batch.",
        "prompt_body": (
            "Implement `ex3_scale_rows(x, w)` to multiply each row of a batch by a "
            "per-sample scalar.\n\n"
            "Input shapes: `x` is `(N, D)`, `w` is `(N,)`. Output shape: `(N, D)`. "
            "Every row of the output should equal `x[n] * w[n]`.\n\n"
            "**Watch out:** `x * w` does NOT work — right-align broadcasting tries "
            "to match `(N,)` against `(N, D)`'s last axis `D`. You need to reshape "
            "`w` to `(N, 1)` first via `w.unsqueeze(1)` or `w[:, None]`."
        ),
        "stub": (
            "def ex3_scale_rows(x: Tensor, w: Tensor) -> Tensor:\n"
            '    """Per-sample scale. (N, D) * (N,) → (N, D) via unsqueeze."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3).reshape(2, 3).float()        # (2, 3)\n"
            "    w = t.tensor([2.0, 10.0])                        # (2,)\n"
            "    y = ex3_scale_rows(x, w)\n"
            "    assert y.shape == (2, 3), f'expected (2,3), got {tuple(y.shape)}'\n"
            "    expected = t.tensor([[0.0, 2.0, 4.0], [30.0, 40.0, 50.0]])\n"
            "    assert t.equal(y, expected), f'value mismatch: {y.tolist()}'"
        ),
        "solution_body": (
            "def ex3_scale_rows(x: Tensor, w: Tensor) -> Tensor:\n"
            "    return x * w.unsqueeze(1)"
        ),
        "solution_notes": (
            "**Why does `x * w` fail with N≠D?** Right-align takes `(N,)` and "
            "tries to broadcast it against `(N, D)`'s last axis (`D`). If `N == D` "
            "it accidentally works (and gives the WRONG result — it scales by "
            "feature instead of by sample). With `N ≠ D` you get a shape error. "
            "`unsqueeze(1)` makes `w` into `(N, 1)` so it right-aligns as "
            "`(N, 1)` vs `(N, D)` → `(N, D)`. Always be explicit about which axis "
            "you're broadcasting along."
        ),
    },
    {
        "id": "ex4",
        "title": "insert a missing axis where broadcast fails",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["unsqueeze", "axis-insertion", "shape-debug"],
        "kcs": ["broadcast-via-unsqueeze"],
        "lo": "Apply targeted axis insertion (`unsqueeze`) to make a 3-D + 1-D broadcast work along the desired axis.",
        "prompt_body": (
            "Implement `ex4_scale_channel(x, w)` to scale each feature channel of "
            "a feature-map batch by a per-channel weight.\n\n"
            "Input shapes: `x` is `(B, C, H, W)`, `w` is `(C,)`. Output shape: "
            "`(B, C, H, W)`. Each `out[b, c, :, :] == x[b, c, :, :] * w[c]`.\n\n"
            "Right-align would try to broadcast `(C,)` against the trailing `W` "
            "axis — wrong. Reshape `w` to insert size-1 axes where they need to "
            "be so the broadcast targets the channel axis instead.\n\n"
            "**Hint:** the right shape for `w` to broadcast against `(B, C, H, W)` "
            "along the channel axis is `(1, C, 1, 1)`. Use `w.reshape(...)` or "
            "chained `unsqueeze` calls — both are fine."
        ),
        "stub": (
            "def ex4_scale_channel(x: Tensor, w: Tensor) -> Tensor:\n"
            '    """Per-channel scale. (B, C, H, W) * (C,) → (B, C, H, W) via axis insertion."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.ones(2, 3, 4, 5)\n"
            "    w = t.tensor([1.0, 2.0, 3.0])  # (C=3,)\n"
            "    y = ex4_scale_channel(x, w)\n"
            "    assert y.shape == (2, 3, 4, 5), f'expected (2,3,4,5), got {tuple(y.shape)}'\n"
            "    # Channel 0 should be 1.0 everywhere, channel 1 should be 2.0, channel 2 should be 3.0.\n"
            "    assert t.allclose(y[:, 0, :, :], t.ones(2, 4, 5)), 'channel 0 not scaled by w[0]=1'\n"
            "    assert t.allclose(y[:, 1, :, :], 2.0 * t.ones(2, 4, 5)), 'channel 1 not scaled by w[1]=2'\n"
            "    assert t.allclose(y[:, 2, :, :], 3.0 * t.ones(2, 4, 5)), 'channel 2 not scaled by w[2]=3'"
        ),
        "solution_body": (
            "def ex4_scale_channel(x: Tensor, w: Tensor) -> Tensor:\n"
            "    return x * w.reshape(1, -1, 1, 1)"
        ),
        "solution_notes": (
            "**Three equivalent forms** for inserting axes:\n"
            "- `w.reshape(1, -1, 1, 1)` — explicit; `-1` infers C.\n"
            "- `w[None, :, None, None]` — slice-syntax `None` is shorthand for `unsqueeze`.\n"
            "- `w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)` — chained, error-prone in higher dims.\n\n"
            "Pick whichever reads clearest at the call site. `reshape` is usually "
            "the most explicit for >2 axis insertions."
        ),
    },
    {
        "id": "ex5",
        "title": "outer product via column×row broadcast",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["outer-product", "column-times-row", "integration", "multi-kc"],
        "kcs": [
            "broadcast-row-vector",
            "broadcast-column-vector",
            "broadcast-outer-product",
        ],
        "lo": "Synthesize column-vector broadcast + row-vector broadcast to compute the outer product of two 1-D tensors without `torch.outer`.",
        "prompt_body": (
            "Implement `ex5_outer(u, v)` to compute the outer product of two 1-D "
            "tensors.\n\n"
            "Input shapes: `u` is `(N,)`, `v` is `(M,)`. Output shape: `(N, M)`. "
            "Each `out[i, j] == u[i] * v[j]`.\n\n"
            "**Use broadcasting only** — no `torch.outer`, no `einsum`, no `unsqueeze` "
            "+ matmul. Strategy: reshape `u` to a column `(N, 1)` and `v` to a row "
            "`(1, M)`, then multiply. The right-align rule produces `(N, M)`.\n\n"
            "Equivalent to `torch.outer(u, v)`.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3+ KCs (column-broadcast, "
            "row-broadcast, axis insertion) in one expression; empirical work (Lohr "
            "et al. ITiCSE 2025) shows 3-concept LLM-generated exercises drop from "
            "~94% to ~40% solvability. Expect a step in difficulty here vs "
            "Exercises 1-4."
        ),
        "stub": (
            "def ex5_outer(u: Tensor, v: Tensor) -> Tensor:\n"
            '    """Outer product via broadcasting. (N,) * (M,) → (N, M)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "u = t.tensor([1.0, 2.0, 3.0])     # (3,)\n"
            "    v = t.tensor([10.0, 20.0])         # (2,)\n"
            "    M = ex5_outer(u, v)\n"
            "    assert M.shape == (3, 2), f'expected (3,2), got {tuple(M.shape)}'\n"
            "    expected = t.tensor([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]])\n"
            "    assert t.equal(M, expected), f'value mismatch: {M.tolist()}'\n"
            "    # Also compare to torch.outer ground truth.\n"
            "    assert t.equal(M, t.outer(u, v)), 'differs from t.outer(u, v)'"
        ),
        "solution_body": (
            "def ex5_outer(u: Tensor, v: Tensor) -> Tensor:\n"
            "    return u.unsqueeze(1) * v.unsqueeze(0)"
        ),
        "solution_notes": (
            "**Reading the pattern.**\n"
            "- `u.unsqueeze(1)` → shape `(N, 1)` (column vector).\n"
            "- `v.unsqueeze(0)` → shape `(1, M)` (row vector).\n"
            "- `(N, 1) * (1, M)` right-aligns as `(N, 1) * (1, M)` → broadcasts to "
            "`(N, M)`.\n\n"
            "Each broadcast tile fills a row (from `v`) or a column (from `u`); "
            "their elementwise product gives `u[i] * v[j]` at every position.\n\n"
            "**Equivalent forms:** `u[:, None] * v[None, :]` (same thing, slice "
            "syntax); `torch.einsum('i,j->ij', u, v)`; `torch.outer(u, v)`. The "
            "broadcasting form is worth knowing because it generalizes to higher "
            "ranks (e.g. batched outer product) without changing the mental model."
        ),
    },
]


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


cells = []

cells.append(md(
    f"# {TITLE}",
    "",
    "> Procedural drill from [Delta Drills](https://delta-drills.vercel.app).",
    f"> Atom: `{ATOM_ID}`. When a test cell passes, your progress is reported back to your account.",
    "",
    "**What you'll practice.** Five broadcasting patterns that ramp from predicting the result shape → row-vector broadcast → column-vector broadcast → targeted axis insertion → outer product via broadcast. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "## Broadcasting — quick refresher",
    "",
    "**The rule** (NumPy & PyTorch agree):",
    "1. Right-align both shapes; left-pad the shorter with 1s.",
    "2. For each pair of aligned axes: equal → keep; one is 1 → use the other; otherwise → incompatible.",
    "",
    "**Three patterns you reach for constantly:**",
    "- **Row broadcast** — `(N, D) + (D,)` works automatically. Adds a per-feature bias.",
    "- **Column broadcast** — `(N, D) * w` where `w` is `(N,)` fails. Reshape `w` to `(N, 1)` first.",
    "- **Axis insertion** — `unsqueeze` / `[:, None]` / `reshape` are all valid ways to insert a size-1 axis where broadcasting needs it.",
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
