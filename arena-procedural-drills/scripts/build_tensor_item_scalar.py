#!/usr/bin/env python3
"""Build the tensor-item-scalar procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/tensor-item-scalar.ipynb`
(Phase 4.13). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `tensor-item-scalar` covers `.item()` — the tensor-to-Python-scalar
extraction op. In Ray Tracing it's how you read out a hit-parameter t or
a hit-count for printing / control flow; in any training loop it's how you
get the loss value out for logging. Bridges to bank topic `Numpy` via the
explicit token rule (`tensor-item`) in atom_readiness.js, reporting to
subtopic `Numpy: Core array literacy`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/tensor-item-scalar.ipynb"


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


ATOM_ID = "tensor-item-scalar"
SUBTOPIC = "Numpy: Core array literacy"
TITLE = "tensor .item() — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "item-from-zero-dim",
        "kind": "component-skill",
        "description": "Call `.item()` on a 0-D tensor to extract a Python scalar of the matching native type. The atomic call.",
    },
    {
        "id": "item-from-one-elem-1d",
        "kind": "component-skill",
        "description": "Recognize that `.item()` also works on any single-element tensor regardless of shape — `tensor([7]).item()`, `tensor([[3]]).item()`, etc.",
    },
    {
        "id": "item-dtype-preservation",
        "kind": "component-skill",
        "description": "Understand that `.item()` returns the matching Python type: `float32` / `float64` → `float`, `int64` / `bool` → `int` / `bool`. The Python value mirrors the tensor's dtype family.",
    },
    {
        "id": "item-vs-tolist-vs-many",
        "kind": "component-skill",
        "description": "Pick the right scalar-extraction op: `.item()` for exactly one element, `.tolist()` for arbitrary shape → nested Python lists. Calling `.item()` on a multi-element tensor raises — handle that.",
    },
    {
        "id": "item-for-python-control-flow",
        "kind": "integrative-skill",
        "description": "Use `.item()` to bridge from tensor space to Python space for control flow: compute a tensor condition / reduction, then `.item()` so the result can drive `if` statements, `range(...)`, or a Python-level count returned to the caller.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "extract a Python float from a 0-D tensor",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["item", "0-D-tensor", "python-float"],
        "kcs": ["item-from-zero-dim"],
        "lo": "Recall how to extract a Python scalar from a 0-D tensor.",
        "prompt_body": (
            "Implement `ex1_scalar_to_float(x)` to return the Python `float` "
            "value of a 0-D float tensor.\n\n"
            "Use `x.item()`. The return type must be a plain Python `float`, NOT "
            "a tensor."
        ),
        "stub": (
            "def ex1_scalar_to_float(x: Tensor) -> float:\n"
            '    """Extract a Python float from a 0-D float tensor."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor(3.5)\n"
            "    out = ex1_scalar_to_float(x)\n"
            "    assert isinstance(out, float), f'expected float, got {type(out).__name__}'\n"
            "    assert out == 3.5, f'expected 3.5, got {out}'\n"
            "    # Negative.\n"
            "    assert ex1_scalar_to_float(t.tensor(-2.0)) == -2.0\n"
            "    # Note: 0-D from a reduction.\n"
            "    assert ex1_scalar_to_float(t.tensor([1.0, 2.0, 3.0]).sum()) == 6.0"
        ),
        "solution_body": (
            "def ex1_scalar_to_float(x: Tensor) -> float:\n"
            "    return x.item()"
        ),
        "solution_notes": (
            "**Why `.item()` and not `float(x)`?** Both work for 0-D float tensors. "
            "But `.item()` is the official PyTorch API, gives a clear error on "
            "wrong-shape input, and works uniformly across float / int / bool. "
            "`float(x)` is older Python coercion machinery — it works here but "
            "doesn't generalise. Default to `.item()`."
        ),
    },
    {
        "id": "ex2",
        "title": "extract from a single-element 1-D tensor",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["1-elem", "any-shape", "scalar-tensor"],
        "kcs": ["item-from-one-elem-1d"],
        "lo": "Apply `.item()` to a single-element tensor of arbitrary shape.",
        "prompt_body": (
            "Implement `ex2_one_elem_extract(x)`. `x` is a tensor containing exactly "
            "one element, but it might have any shape — `(1,)`, `(1, 1)`, `(1, 1, 1)`. "
            "Return the Python scalar value.\n\n"
            "`.item()` works on any single-element tensor, no matter how many size-1 "
            "axes wrap it."
        ),
        "stub": (
            "def ex2_one_elem_extract(x: Tensor) -> float:\n"
            '    """Extract the Python scalar from a single-element tensor."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "assert ex2_one_elem_extract(t.tensor([7.0])) == 7.0\n"
            "    assert ex2_one_elem_extract(t.tensor([[7.0]])) == 7.0\n"
            "    assert ex2_one_elem_extract(t.tensor([[[7.0]]])) == 7.0\n"
            "    # Result must be a Python float, not a tensor.\n"
            "    out = ex2_one_elem_extract(t.tensor([[2.5]]))\n"
            "    assert isinstance(out, float), f'expected float, got {type(out).__name__}'"
        ),
        "solution_body": (
            "def ex2_one_elem_extract(x: Tensor) -> float:\n"
            "    return x.item()"
        ),
        "solution_notes": (
            "**Why `.item()` doesn't care about wrapper axes.** Internally PyTorch "
            "checks `x.numel() == 1`, not `x.dim() == 0`. So any tensor with exactly "
            "one element — regardless of how many `(1, 1, 1, …)` wrappers — works. "
            "This is handy when you've kept axes around via `keepdim=True` and the "
            "reduction collapsed to a single scalar."
        ),
    },
    {
        "id": "ex3",
        "title": ".item() preserves dtype family",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["dtype-mapping", "int-vs-float", "bool"],
        "kcs": ["item-dtype-preservation"],
        "lo": "Apply `.item()` across dtypes and recognize that the Python return type follows the tensor's dtype family.",
        "prompt_body": (
            "Implement `ex3_describe_scalar(x)`. Given a single-element tensor of "
            "any dtype, return a tuple `(value, python_type_name)` where:\n"
            "- `value = x.item()`\n"
            "- `python_type_name` is the result of `type(value).__name__` — e.g. "
            "`'int'`, `'float'`, `'bool'`.\n\n"
            "Float dtypes (`float32`, `float64`) produce Python `float`. Integer "
            "dtypes (`int64`, `int32`) produce Python `int`. `bool` produces `bool`."
        ),
        "stub": (
            "def ex3_describe_scalar(x: Tensor) -> tuple:\n"
            '    """Return (x.item(), type name string)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "v, name = ex3_describe_scalar(t.tensor(3.5))\n"
            "    assert v == 3.5 and name == 'float', f'float32: got ({v!r}, {name!r})'\n"
            "    v, name = ex3_describe_scalar(t.tensor(7, dtype=t.long))\n"
            "    assert v == 7 and name == 'int', f'long: got ({v!r}, {name!r})'\n"
            "    v, name = ex3_describe_scalar(t.tensor(True))\n"
            "    assert v is True and name == 'bool', f'bool: got ({v!r}, {name!r})'\n"
            "    # double precision still maps to Python float.\n"
            "    v, name = ex3_describe_scalar(t.tensor(2.0, dtype=t.float64))\n"
            "    assert v == 2.0 and name == 'float', f'float64: got ({v!r}, {name!r})'"
        ),
        "solution_body": (
            "def ex3_describe_scalar(x: Tensor) -> tuple:\n"
            "    v = x.item()\n"
            "    return v, type(v).__name__"
        ),
        "solution_notes": (
            "**Practical implication.** If you `.item()` an `int64` tensor and pass "
            "it to a function that expects `float`, you'll silently get integer "
            "division behavior in Python 2-style edge cases (today rare, but the "
            "`/` vs `//` distinction still matters). Always check what dtype you "
            "started with before consuming the scalar.\n\n"
            "**Numerical precision.** `float32` tensors lose precision when "
            "`.item()`'d to Python `float` (which is 64-bit) — but the *value* "
            "was already truncated in the tensor. `.item()` doesn't help or hurt."
        ),
    },
    {
        "id": "ex4",
        "title": ".item() vs .tolist() — pick the right tool",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["tolist", "shape-sensitive", "error-handling"],
        "kcs": ["item-vs-tolist-vs-many"],
        "lo": "Apply `.item()` for single-element tensors and `.tolist()` for multi-element ones — and detect when the wrong one would error.",
        "prompt_body": (
            "Implement `ex4_extract_safely(x)`. Given a tensor of unknown size, "
            "return:\n"
            "- A Python scalar (via `.item()`) if `x` has exactly one element.\n"
            "- A nested Python list (via `.tolist()`) otherwise.\n\n"
            "Use `x.numel()` to check the size. Don't blindly call `.item()` — "
            "it raises `RuntimeError` on multi-element tensors."
        ),
        "stub": (
            "def ex4_extract_safely(x: Tensor):\n"
            '    """Scalar for 1-elem tensors, nested list otherwise."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Scalar case.\n"
            "    out = ex4_extract_safely(t.tensor([42.0]))\n"
            "    assert isinstance(out, float) and out == 42.0, f'1-elem should give float, got {out!r}'\n"
            "    # Multi-element 1-D.\n"
            "    out = ex4_extract_safely(t.tensor([1.0, 2.0, 3.0]))\n"
            "    assert out == [1.0, 2.0, 3.0], f'1-D list mismatch: {out!r}'\n"
            "    # 2-D → nested list.\n"
            "    out = ex4_extract_safely(t.tensor([[1.0, 2.0], [3.0, 4.0]]))\n"
            "    assert out == [[1.0, 2.0], [3.0, 4.0]], f'2-D list mismatch: {out!r}'\n"
            "    # 0-D — single element.\n"
            "    assert ex4_extract_safely(t.tensor(5.0)) == 5.0"
        ),
        "solution_body": (
            "def ex4_extract_safely(x: Tensor):\n"
            "    if x.numel() == 1:\n"
            "        return x.item()\n"
            "    return x.tolist()"
        ),
        "solution_notes": (
            "**The error you avoid.** `t.tensor([1.0, 2.0]).item()` raises "
            "`RuntimeError: a Tensor with 2 elements cannot be converted to "
            "Scalar`. Branching on `numel()` keeps your utility code working "
            "across shapes.\n\n"
            "**When to use `.tolist()`.** For dumping a small tensor to JSON, "
            "logging structured data, or hand-comparing values during debugging. "
            "For large tensors it's wasteful — prefer summary stats."
        ),
    },
    {
        "id": "ex5",
        "title": "tensor → Python control flow",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["control-flow", "reduce-then-item", "logging-loop", "multi-kc"],
        "kcs": [
            "item-from-zero-dim",
            "item-dtype-preservation",
            "item-for-python-control-flow",
        ],
        "lo": "Synthesize tensor reduction + `.item()` to bridge tensor space to Python space, returning a Python-typed result usable in `if` / `range` / logging.",
        "prompt_body": (
            "Implement `ex5_count_above(x, threshold)`. The canonical 'tensor → "
            "Python scalar for control flow' pattern.\n\n"
            "Given a 2-D tensor `x` of shape `(B, D)` and a scalar `threshold`, "
            "count how many rows have L2 norm strictly greater than `threshold`. "
            "Return a plain Python `int` (NOT a 0-D tensor).\n\n"
            "Steps:\n"
            "1. Compute per-row L2 norms — shape `(B,)`.\n"
            "2. Build a bool mask `norms > threshold`.\n"
            "3. Sum the mask to get a 0-D `int64` tensor.\n"
            "4. `.item()` to get a Python `int`.\n\n"
            "Why a Python int? So callers can use the count in `if count > 0:` "
            "without `.item()` boilerplate, or pass it to `range(...)`.\n\n"
            "> ⚠️ **Integrative exercise.** Combines 3 KCs (0-D-extract, dtype "
            "preservation, tensor→Python control flow bridge). Empirical work "
            "(Lohr et al. ITiCSE 2025) shows 3-concept exercises drop to ~40% "
            "solvability — expect a step up vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_count_above(x: Tensor, threshold: float) -> int:\n"
            '    """Count rows with L2 norm > threshold, return a Python int."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([\n"
            "        [3.0, 4.0],   # norm 5\n"
            "        [0.0, 0.0],   # norm 0\n"
            "        [1.0, 0.0],   # norm 1\n"
            "        [6.0, 8.0],   # norm 10\n"
            "    ])\n"
            "    # threshold = 0.5: rows 0, 2, 3 qualify (norms 5, 1, 10).\n"
            "    count = ex5_count_above(x, threshold=0.5)\n"
            "    assert isinstance(count, int), f'expected int, got {type(count).__name__}'\n"
            "    assert count == 3, f'expected 3, got {count}'\n"
            "\n"
            "    # threshold = 4.0: rows 0, 3 qualify (norms 5, 10).\n"
            "    assert ex5_count_above(x, threshold=4.0) == 2\n"
            "\n"
            "    # threshold = 100.0: no rows qualify.\n"
            "    zero_count = ex5_count_above(x, threshold=100.0)\n"
            "    assert isinstance(zero_count, int) and zero_count == 0\n"
            "\n"
            "    # Must be a Python int — usable in range().\n"
            "    consumed = list(range(ex5_count_above(x, threshold=0.5)))\n"
            "    assert consumed == [0, 1, 2], 'result must be usable in range()'"
        ),
        "solution_body": (
            "def ex5_count_above(x: Tensor, threshold: float) -> int:\n"
            "    norms = x.pow(2).sum(dim=1).sqrt()\n"
            "    mask = norms > threshold\n"
            "    return mask.sum().item()"
        ),
        "solution_notes": (
            "**Why a Python int and not a 0-D tensor?** Callers that use the "
            "result as a loop bound, condition, or array length need a Python "
            "scalar. PyTorch is fine with most operator overloads, but `range(t.tensor(5))` "
            "raises in some versions, and a 0-D tensor stored in a dict key won't "
            "hash. `.item()` is the unambiguous bridge.\n\n"
            "**Cost.** `.item()` is a host-device sync if `x` is on GPU — every "
            "call blocks the kernel queue. Inside a tight training loop you should "
            "minimise `.item()` calls (cache the loss tensor, `.item()` only when "
            "logging) but for one-off control flow it's fine."
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
    "**What you'll practice.** Five `.item()` patterns that ramp from 0-D extract → single-elem extract → dtype preservation → `.item()` vs `.tolist()` → tensor → Python control flow. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "## `.item()` — quick refresher",
    "",
    "**What it does.** Extracts a Python scalar (float / int / bool) from a tensor that has exactly one element. Bridges tensor space to Python space.",
    "",
    "**Requirements.** `x.numel() == 1`. Shape doesn't matter — `(1,)`, `(1, 1)`, `()` all work as long as there's exactly one element. Multi-element tensors raise `RuntimeError`.",
    "",
    "**Dtype mapping:**",
    "- `float32` / `float64` → Python `float`",
    "- `int32` / `int64` → Python `int`",
    "- `bool` → Python `bool`",
    "",
    "**When to use:** logging losses, control-flow conditions, returning counts to callers, dict / set keys (tensors aren't hashable).",
    "",
    "**When NOT to use:** inside tight inner loops on GPU — every `.item()` is a host-device sync that blocks the kernel queue.",
    "",
    "**For more elements:** `.tolist()` returns a (nested) Python list of any shape.",
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
