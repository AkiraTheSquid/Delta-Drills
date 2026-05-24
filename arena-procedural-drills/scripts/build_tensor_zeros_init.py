#!/usr/bin/env python3
"""Build the tensor-zeros-init procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/tensor-zeros-init.ipynb`
(Phase 4.9). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `tensor-zeros-init` is the most-linked Ray Tracing prereq (8 problem
links): every per-ray computation needs an output buffer of the right
shape/dtype/device allocated up front. Bridges to bank topic `Numpy` via
the explicit token rule (`tensor-zeros`) in atom_readiness.js, and lands
on subtopic `Numpy: Core array literacy` for EWMA reporting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/tensor-zeros-init.ipynb"


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


ATOM_ID = "tensor-zeros-init"
SUBTOPIC = "Numpy: Core array literacy"
TITLE = "tensor zeros init — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "zeros-1d-shape",
        "kind": "component-skill",
        "description": "Allocate a 1-D zero tensor with `torch.zeros(n)`. Verifies the basic call returns shape `(n,)` filled with floating zeros.",
    },
    {
        "id": "zeros-multi-axis-shape",
        "kind": "component-skill",
        "description": "Allocate a multi-axis zero tensor with positional args `torch.zeros(b, h, w)` (or tuple form). Verifies the shape construction.",
    },
    {
        "id": "zeros-like-mirrors-input",
        "kind": "component-skill",
        "description": "Use `torch.zeros_like(x)` to mirror the input's shape AND dtype AND device. The canonical 'same shape, fresh zero buffer' move for accumulators.",
    },
    {
        "id": "zeros-dtype-control",
        "kind": "component-skill",
        "description": "Override the default dtype with `dtype=torch.long` (or `torch.bool`, etc.). Index buffers MUST be integer dtype — float zeros silently fail when used in `gather` / `index_select`.",
    },
    {
        "id": "zeros-allocate-then-fill",
        "kind": "integrative-skill",
        "description": "Allocate a typed zero buffer of the right shape and then scatter values into it via indexed assignment. The integrative KC — combines shape control, dtype default, and indexed-write. The canonical per-ray output-buffer pattern in Ray Tracing.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "allocate a 1-D zero vector",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["torch-zeros", "shape", "dtype-default"],
        "kcs": ["zeros-1d-shape"],
        "lo": "Recall the basic `torch.zeros(n)` allocation call.",
        "prompt_body": (
            "Implement `ex1_zeros_1d(n)` to return a 1-D tensor of `n` floating-point "
            "zeros. Default dtype is `torch.float32`.\n\n"
            "Use `torch.zeros(n)`."
        ),
        "stub": (
            "def ex1_zeros_1d(n: int) -> Tensor:\n"
            '    """Return a 1-D tensor of n floating zeros."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "out = ex1_zeros_1d(5)\n"
            "    assert out.shape == (5,), f'expected (5,), got {tuple(out.shape)}'\n"
            "    assert out.dtype == t.float32, f'default dtype should be float32, got {out.dtype}'\n"
            "    assert t.all(out == 0), f'expected all zeros, got {out.tolist()}'\n"
            "    assert ex1_zeros_1d(0).shape == (0,), 'n=0 must still return a 0-element tensor (not error)'"
        ),
        "solution_body": (
            "def ex1_zeros_1d(n: int) -> Tensor:\n"
            "    return t.zeros(n)"
        ),
        "solution_notes": (
            "**Default dtype is float32.** `torch.zeros(5)` is shorthand for "
            "`torch.zeros(5, dtype=torch.float32)`. To get integers you must "
            "pass `dtype=` explicitly — see Exercise 4."
        ),
    },
    {
        "id": "ex2",
        "title": "allocate a 3-D zero tensor",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["multi-axis", "shape-args", "batched-buffer"],
        "kcs": ["zeros-multi-axis-shape"],
        "lo": "Apply positional shape arguments to allocate a multi-axis zero tensor.",
        "prompt_body": (
            "Implement `ex2_zeros_3d(b, h, w)` to return a `(b, h, w)` tensor of "
            "floating zeros — the kind of buffer you'd allocate for a batch of "
            "rendered images.\n\n"
            "Either `torch.zeros(b, h, w)` (positional) or `torch.zeros((b, h, w))` "
            "(tuple) works."
        ),
        "stub": (
            "def ex2_zeros_3d(b: int, h: int, w: int) -> Tensor:\n"
            '    """Return a (b, h, w) zero tensor."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "out = ex2_zeros_3d(2, 3, 4)\n"
            "    assert out.shape == (2, 3, 4), f'expected (2,3,4), got {tuple(out.shape)}'\n"
            "    assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "    assert t.all(out == 0), 'expected all zeros'\n"
            "    # Singleton dims must also work.\n"
            "    assert ex2_zeros_3d(1, 1, 1).shape == (1, 1, 1), 'singleton dims must be preserved'"
        ),
        "solution_body": (
            "def ex2_zeros_3d(b: int, h: int, w: int) -> Tensor:\n"
            "    return t.zeros(b, h, w)"
        ),
        "solution_notes": (
            "**Positional vs tuple.** Both `t.zeros(b, h, w)` and `t.zeros((b, h, w))` "
            "produce the same tensor. The positional form is idiomatic when shape "
            "is known at write-time; the tuple form is useful when you have a shape "
            "computed dynamically (e.g. `t.zeros(x.shape)`)."
        ),
    },
    {
        "id": "ex3",
        "title": "zeros_like — mirror an input's shape and dtype",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["zeros-like", "shape-mirror", "dtype-mirror"],
        "kcs": ["zeros-like-mirrors-input"],
        "lo": "Apply `torch.zeros_like` to allocate a fresh zero buffer matching the input's shape AND dtype.",
        "prompt_body": (
            "Implement `ex3_zeros_like(x)` to return a fresh zero tensor with the "
            "same shape, dtype, and device as `x`.\n\n"
            "Use `torch.zeros_like(x)`. Critically: the result must NOT be a view "
            "or alias of `x` — writing to the output must not change `x`."
        ),
        "stub": (
            "def ex3_zeros_like(x: Tensor) -> Tensor:\n"
            '    """Return a fresh zero buffer mirroring x.shape and x.dtype."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x_int = t.tensor([[1, 2, 3], [4, 5, 6]], dtype=t.int64)\n"
            "    out = ex3_zeros_like(x_int)\n"
            "    assert out.shape == x_int.shape, f'shape mismatch: {out.shape} vs {x_int.shape}'\n"
            "    assert out.dtype == t.int64, f'dtype must be mirrored: got {out.dtype}'\n"
            "    assert t.all(out == 0), 'must be all zeros'\n"
            "    # Aliasing check — writing to out must not mutate x.\n"
            "    out[0, 0] = 99\n"
            "    assert x_int[0, 0].item() == 1, 'zeros_like must be a FRESH tensor, not a view'\n"
            "\n"
            "    x_float = t.randn(4, 5)\n"
            "    out_f = ex3_zeros_like(x_float)\n"
            "    assert out_f.shape == x_float.shape\n"
            "    assert out_f.dtype == t.float32, 'float32 input → float32 output'"
        ),
        "solution_body": (
            "def ex3_zeros_like(x: Tensor) -> Tensor:\n"
            "    return t.zeros_like(x)"
        ),
        "solution_notes": (
            "**Why prefer `zeros_like(x)` over `zeros(x.shape)`?**\n"
            "- `zeros(x.shape)` only copies the shape — the dtype reverts to float32 "
            "and the device reverts to CPU. If `x` is a `int64` GPU tensor, your "
            "buffer ends up float32 on CPU — silent breakage the moment you try to "
            "use it as indices or do an op against `x`.\n"
            "- `zeros_like(x)` mirrors shape + dtype + device. Always the right call "
            "when allocating a per-input accumulator."
        ),
    },
    {
        "id": "ex4",
        "title": "integer index buffer with dtype=long",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["dtype-long", "index-buffer", "gather-ready"],
        "kcs": ["zeros-dtype-control"],
        "lo": "Apply `dtype=torch.long` to allocate an integer buffer suitable for use as indices.",
        "prompt_body": (
            "Implement `ex4_index_buffer(n)` to return a 1-D zero tensor of length "
            "`n` with dtype `torch.long` (= int64). Integer dtype is required because "
            "PyTorch's `gather` / `index_select` / advanced indexing reject float "
            "indices.\n\n"
            "Use `torch.zeros(n, dtype=torch.long)`."
        ),
        "stub": (
            "def ex4_index_buffer(n: int) -> Tensor:\n"
            '    """Return a zero index buffer of length n, dtype=long."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "idx = ex4_index_buffer(4)\n"
            "    assert idx.shape == (4,), f'expected (4,), got {tuple(idx.shape)}'\n"
            "    assert idx.dtype == t.long, f'expected dtype long (int64), got {idx.dtype}'\n"
            "    assert t.all(idx == 0), 'expected all zeros'\n"
            "    # Critical functional check — must be usable as indices into another tensor.\n"
            "    source = t.tensor([10.0, 20.0, 30.0])\n"
            "    gathered = source[idx]  # all zeros → picks element 0 four times.\n"
            "    assert t.allclose(gathered, t.tensor([10.0, 10.0, 10.0, 10.0])), 'index buffer must work for advanced indexing'"
        ),
        "solution_body": (
            "def ex4_index_buffer(n: int) -> Tensor:\n"
            "    return t.zeros(n, dtype=t.long)"
        ),
        "solution_notes": (
            "**Why `long` and not `int`?** PyTorch's advanced-indexing path requires "
            "`int64` (`torch.long`). `torch.int32` works for some ops but fails on "
            "`gather` and on CPU advanced indexing — a common confusing footgun. "
            "Default to `long` for any tensor that will hold indices."
        ),
    },
    {
        "id": "ex5",
        "title": "allocate output buffer, then paint hits",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["accumulator", "indexed-assign", "ray-tracing", "multi-kc"],
        "kcs": [
            "zeros-multi-axis-shape",
            "zeros-dtype-control",
            "zeros-allocate-then-fill",
        ],
        "lo": "Synthesize shape arg + dtype default + indexed assignment to scatter per-ray hit colors into an output buffer.",
        "prompt_body": (
            "Implement `ex5_paint_hits(num_rays, hit_indices, hit_colors)`. The "
            "canonical Ray Tracing output-buffer pattern:\n\n"
            "1. Allocate a `(num_rays, 3)` zero buffer (float32 by default — perfect "
            "for RGB colors in `[0, 1]`).\n"
            "2. For each `k`, write `hit_colors[k]` into row `hit_indices[k]`.\n"
            "3. Rays not in `hit_indices` stay `[0, 0, 0]` (black — no hit).\n\n"
            "Inputs:\n"
            "- `num_rays`: int.\n"
            "- `hit_indices`: 1-D long tensor, shape `(K,)`, values in `[0, num_rays)`.\n"
            "- `hit_colors`: 2-D float tensor, shape `(K, 3)`.\n\n"
            "Output: `(num_rays, 3)` float32 tensor.\n\n"
            "Hint: `out[hit_indices] = hit_colors` does the scatter in one shot.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3 KCs (shape allocation, "
            "default dtype, indexed assignment); empirical work (Lohr et al. ITiCSE "
            "2025) shows 3-concept LLM-generated exercises drop from ~94% to ~40% "
            "solvability. Expect a step up vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_paint_hits(num_rays: int, hit_indices: Tensor, hit_colors: Tensor) -> Tensor:\n"
            '    """Allocate (num_rays, 3) zero buffer; write hit_colors at hit_indices."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "hit_indices = t.tensor([0, 2, 3], dtype=t.long)\n"
            "    hit_colors = t.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])\n"
            "    out = ex5_paint_hits(5, hit_indices, hit_colors)\n"
            "    assert out.shape == (5, 3), f'expected (5, 3), got {tuple(out.shape)}'\n"
            "    assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "    expected = t.tensor([\n"
            "        [1.0, 0.0, 0.0],  # ray 0 — red hit\n"
            "        [0.0, 0.0, 0.0],  # ray 1 — no hit, stays zero\n"
            "        [0.0, 1.0, 0.0],  # ray 2 — green hit\n"
            "        [0.0, 0.0, 1.0],  # ray 3 — blue hit\n"
            "        [0.0, 0.0, 0.0],  # ray 4 — no hit, stays zero\n"
            "    ])\n"
            "    assert t.allclose(out, expected), f'value mismatch:\\n{out}\\nvs\\n{expected}'\n"
            "    # Edge case — no hits at all.\n"
            "    empty_idx = t.zeros(0, dtype=t.long)\n"
            "    empty_col = t.zeros(0, 3)\n"
            "    out_empty = ex5_paint_hits(3, empty_idx, empty_col)\n"
            "    assert out_empty.shape == (3, 3) and t.all(out_empty == 0), 'no-hits case must return all-zero buffer'"
        ),
        "solution_body": (
            "def ex5_paint_hits(num_rays: int, hit_indices: Tensor, hit_colors: Tensor) -> Tensor:\n"
            "    out = t.zeros(num_rays, 3)\n"
            "    out[hit_indices] = hit_colors\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why this pattern matters.** Every per-ray Ray Tracing computation "
            "uses this shape: allocate a `(num_rays, ...)` output buffer with the "
            "right dtype, compute the mask of which rays did something, scatter "
            "the per-hit values back in. Rays that don't hit anything keep the "
            "default fill (zero / -inf / NaN sentinel depending on the use).\n\n"
            "**Why the indexed-assign works.** `out[hit_indices] = hit_colors` "
            "uses advanced indexing: PyTorch evaluates `hit_indices` as a list "
            "of row positions and writes the matching row from `hit_colors` into "
            "each. Requires `hit_indices.dtype == long` — see Exercise 4."
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
    "**What you'll practice.** Five allocation patterns that ramp from `torch.zeros(n)` → multi-axis shape → `zeros_like` → dtype-long index buffer → allocate-then-scatter for the canonical Ray Tracing per-ray output-buffer pattern. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "## Tensor allocation — quick refresher",
    "",
    "**The four shapes of `zeros`:**",
    "- `t.zeros(n)` — 1-D, shape `(n,)`, default `float32`.",
    "- `t.zeros(b, h, w)` — multi-axis positional args.",
    "- `t.zeros_like(x)` — mirror `x.shape` + `x.dtype` + `x.device`.",
    "- `t.zeros(n, dtype=t.long)` — override dtype for index buffers.",
    "",
    "**The accumulator pattern.** Allocate the right-shaped zero buffer first; scatter per-element results into it via indexed assignment. Cleaner and faster than `append`-and-stack.",
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
