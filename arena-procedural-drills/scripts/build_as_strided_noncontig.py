#!/usr/bin/env python3
"""Build the as-strided-noncontig-source procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source.ipynb`
(Phase 4.8). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `as-strided-noncontig-source` is the foundational pitfall drill for
tensor memory layout — get `.view()` vs `.reshape()` vs `.contiguous()`
wrong and you'll spend hours debugging silently-wrong results. Bridges to
the bank topic `Numpy` via the explicit token rule (`stride`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source.ipynb"


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


ATOM_ID = "as-strided-noncontig-source"
SUBTOPIC = "Numpy: Applied patterns and advanced"
TITLE = "non-contiguous tensors and as_strided — procedural drill"
TEMPLATE_VERSION = "v0.2"

KC_DECOMPOSITION = [
    {
        "id": "strides-anatomy",
        "kind": "component-skill",
        "description": "Read `.stride()` output for a contiguous N-D tensor and explain what each entry means (elements per step along that axis).",
    },
    {
        "id": "transpose-makes-noncontig",
        "kind": "component-skill",
        "description": "Recognize that `.T` / `.transpose()` returns a view that shares storage with the source but is no longer contiguous — `is_contiguous()` returns False.",
    },
    {
        "id": "view-requires-contig",
        "kind": "component-skill",
        "description": "Recall that `.view()` requires the source to be contiguous; calling it on a transposed view raises `RuntimeError`.",
    },
    {
        "id": "contiguous-fixes-view",
        "kind": "component-skill",
        "description": "Apply `.contiguous()` to copy a non-contiguous view into a fresh contiguous layout so `.view()` works again. Same as calling `.reshape()`, which does this automatically when needed.",
    },
    {
        "id": "as-strided-rolling-window",
        "kind": "integrative-skill",
        "description": "Combine stride arithmetic with `torch.Tensor.as_strided` to build a sliding-window view over a 1-D tensor in O(1) memory. The integrative KC — relies on understanding strides anatomy and contiguity guarantees.",
    },
]

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "read the strides of a contiguous tensor",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["strides", "memory-layout", "contiguous"],
        "kcs": ["strides-anatomy"],
        "lo": "Recall what `.stride()` returns for a contiguous 2-D tensor.",
        "prompt_body": (
            "Implement `ex1_get_strides(x)` to return the strides of `x` as a "
            "Python tuple of ints.\n\n"
            "For a contiguous `(H, W)` float tensor, the strides are `(W, 1)` — "
            "moving one step along axis 0 skips `W` elements (a full row), moving "
            "one step along axis 1 skips 1 element (one column).\n\n"
            "Just return `tuple(x.stride())`."
        ),
        "stub": (
            "def ex1_get_strides(x: Tensor) -> tuple:\n"
            '    """Return x.stride() as a tuple of ints."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.zeros(3, 5)\n"
            "    s = ex1_get_strides(x)\n"
            "    assert isinstance(s, tuple), f'expected tuple, got {type(s)}'\n"
            "    assert s == (5, 1), f'expected (5, 1) for (3, 5) contiguous tensor, got {s}'\n"
            "\n"
            "    y = t.zeros(4, 7, 9)\n"
            "    assert ex1_get_strides(y) == (63, 9, 1), f'expected (63, 9, 1), got {ex1_get_strides(y)}'"
        ),
        "solution_body": (
            "def ex1_get_strides(x: Tensor) -> tuple:\n"
            "    return tuple(x.stride())"
        ),
        "solution_notes": (
            "**How to read the result.** For a contiguous tensor of shape "
            "`(d0, d1, ..., dn)`, the stride is `(d1*d2*...*dn, d2*...*dn, ..., dn, 1)`. "
            "It's just the product of all later dimensions — a recipe for converting "
            "an N-D index into a flat memory offset."
        ),
    },
    {
        "id": "ex2",
        "title": "transpose breaks contiguity",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["transpose", "non-contiguous", "view-vs-copy"],
        "kcs": ["transpose-makes-noncontig"],
        "lo": "Apply `is_contiguous()` to show that `.T` returns a non-contiguous view.",
        "prompt_body": (
            "Implement `ex2_transpose_contig_flags(x)` to return a tuple `(orig_contig, "
            "transposed_contig)` — the contiguity flag of the original tensor and "
            "of its transpose.\n\n"
            "For any rectangular (non-square) contiguous matrix, the transpose IS a "
            "view (shares storage), but its strides are reversed — `(1, W)` instead "
            "of `(W, 1)` — so it's no longer contiguous in row-major order."
        ),
        "stub": (
            "def ex2_transpose_contig_flags(x: Tensor) -> tuple[bool, bool]:\n"
            '    """Return (x.is_contiguous(), x.T.is_contiguous())."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12).reshape(3, 4).float()\n"
            "    orig, t_contig = ex2_transpose_contig_flags(x)\n"
            "    assert orig is True, f'fresh reshaped tensor should be contiguous, got {orig}'\n"
            "    assert t_contig is False, f'transpose of (3,4) should be non-contiguous, got {t_contig}'\n"
            "    # Sanity: storage is shared (the transpose is a view, not a copy).\n"
            "    assert x.data_ptr() == x.T.data_ptr(), 'x and x.T should share storage'"
        ),
        "solution_body": (
            "def ex2_transpose_contig_flags(x: Tensor) -> tuple:\n"
            "    return (x.is_contiguous(), x.T.is_contiguous())"
        ),
        "solution_notes": (
            "**Why transpose is a view.** `.T` doesn't move data — it just swaps "
            "the strides. For `(3, 4)` with strides `(4, 1)`, the transpose has "
            "shape `(4, 3)` and strides `(1, 4)` over the SAME storage buffer. "
            "The result is logically transposed but the memory pattern no longer "
            "matches row-major contiguous layout, hence `is_contiguous() → False`."
        ),
    },
    {
        "id": "ex3",
        "title": ".view() raises on non-contiguous input",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["view", "runtime-error", "contiguity-requirement"],
        "kcs": ["view-requires-contig"],
        "lo": "Apply a try/except to detect that `.view()` raises on a non-contiguous tensor.",
        "prompt_body": (
            "Implement `ex3_view_fails_on_transpose(x)` to return `True` if calling "
            "`.view(-1)` on `x.T` raises `RuntimeError`, `False` if it does not.\n\n"
            "`.view()` requires the source to be contiguous. The transpose of a "
            "rectangular matrix is a view but not contiguous → `.view()` complains "
            "rather than silently moving data.\n\n"
            "Use a `try/except RuntimeError` block."
        ),
        "stub": (
            "def ex3_view_fails_on_transpose(x: Tensor) -> bool:\n"
            '    """Return True if x.T.view(-1) raises RuntimeError, else False."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12).reshape(3, 4).float()\n"
            "    assert ex3_view_fails_on_transpose(x) is True, '.view(-1) on x.T should raise on non-contiguous'\n"
            "    # Sanity: the original tensor is contiguous so .view(-1) on it works.\n"
            "    assert x.view(-1).shape == (12,), 'view on a contiguous tensor should work'"
        ),
        "solution_body": (
            "def ex3_view_fails_on_transpose(x: Tensor) -> bool:\n"
            "    try:\n"
            "        x.T.view(-1)\n"
            "        return False\n"
            "    except RuntimeError:\n"
            "        return True"
        ),
        "solution_notes": (
            "**Why `.view()` is strict.** It's the zero-copy reshape. It must "
            "succeed only when the new shape is compatible with the existing "
            "stride pattern — which for arbitrary reshapes means the source must "
            "be contiguous. If you want 'reshape if possible, copy if needed' use "
            "`.reshape()` instead — it falls back to a copy silently."
        ),
    },
    {
        "id": "ex4",
        "title": "fix .view() with .contiguous()",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["contiguous-copy", "view-fix", "row-major"],
        "kcs": ["contiguous-fixes-view"],
        "lo": "Apply `.contiguous()` before `.view()` to copy a non-contiguous view into a fresh row-major layout.",
        "prompt_body": (
            "Implement `ex4_flatten_transpose(x)` to flatten `x.T` to a 1-D tensor "
            "in row-major (C) order, the **transpose-then-flatten** layout.\n\n"
            "Input shape: `(H, W)`. Output shape: `(H * W,)`. Order: read `x.T` "
            "row by row, top to bottom (so the result is `[x[0,0], x[1,0], "
            "x[2,0], ..., x[0,1], x[1,1], ...]` — column-major over the original).\n\n"
            "**Strategy:** call `.contiguous()` on `x.T` first (materializes a "
            "fresh contiguous tensor with the transposed layout), then `.view(-1)`. "
            "Equivalent: `x.T.reshape(-1)` (PyTorch's reshape does the copy for "
            "you when needed). Both are accepted."
        ),
        "stub": (
            "def ex4_flatten_transpose(x: Tensor) -> Tensor:\n"
            '    """Flatten x.T in row-major order. (H, W) → (H*W,)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12).reshape(3, 4).float()\n"
            "    y = ex4_flatten_transpose(x)\n"
            "    assert y.shape == (12,), f'expected (12,), got {tuple(y.shape)}'\n"
            "    # Expected order: column-major over the original = [x[0,0],x[1,0],x[2,0], x[0,1],x[1,1],x[2,1], ...]\n"
            "    expected = t.tensor([0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.])\n"
            "    assert t.equal(y, expected), f'value mismatch: {y.tolist()} vs {expected.tolist()}'"
        ),
        "solution_body": (
            "def ex4_flatten_transpose(x: Tensor) -> Tensor:\n"
            "    return x.T.contiguous().view(-1)"
        ),
        "solution_notes": (
            "**`.contiguous()` semantics.** If the tensor is already contiguous, "
            "it's a no-op (returns `self`). Otherwise it allocates a fresh storage "
            "buffer and copies values into row-major order. After that any view "
            "operation is legal.\n\n"
            "**Pythonic shortcut:** `x.T.reshape(-1)`. `.reshape()` tries to "
            "produce a view (zero copy) and falls back to copy if it can't. Most "
            "code uses `.reshape()` and forgets about `.contiguous()` entirely — "
            "knowing the distinction is mostly defensive."
        ),
    },
    {
        "id": "ex5",
        "title": "rolling window via as_strided (zero-copy)",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["as-strided", "sliding-window", "integration", "multi-kc"],
        "kcs": [
            "strides-anatomy",
            "as-strided-rolling-window",
        ],
        "lo": "Synthesize stride arithmetic with `as_strided` to build an O(1)-memory sliding-window view of a 1-D tensor.",
        "prompt_body": (
            "Implement `ex5_rolling_window(x, w)` to build a 2-D view where row `i` "
            "is the length-`w` window starting at position `i` of `x`.\n\n"
            "Input shape: `(N,)` (1-D). Output shape: `(N - w + 1, w)`. Each "
            "`out[i, j] == x[i + j]`.\n\n"
            "Use `x.as_strided(size, stride)` directly — no Python loop, no `.clone()`, "
            "no `torch.cat`. The result must share storage with `x` (it's a view).\n\n"
            "**Stride hint:** moving one step along the output's row axis = moving "
            "one step along the source. Moving one step along the output's column "
            "axis = also moving one step along the source. So both output strides "
            "equal `x.stride(0)`.\n\n"
            "> ⚠️ **Integrative exercise.** This combines stride mechanics + memory "
            "aliasing + the as_strided API surface — empirical work (Lohr et al. "
            "ITiCSE 2025) shows 3-concept LLM-generated exercises drop from ~94% "
            "to ~40% solvability. Expect a step in difficulty here vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_rolling_window(x: Tensor, w: int) -> Tensor:\n"
            '    """Sliding window over a 1-D tensor. (N,) → (N-w+1, w) view."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(7).float()                 # [0, 1, 2, 3, 4, 5, 6]\n"
            "    y = ex5_rolling_window(x, w=3)\n"
            "    assert y.shape == (5, 3), f'expected (5,3), got {tuple(y.shape)}'\n"
            "    expected = t.tensor([\n"
            "        [0., 1., 2.],\n"
            "        [1., 2., 3.],\n"
            "        [2., 3., 4.],\n"
            "        [3., 4., 5.],\n"
            "        [4., 5., 6.],\n"
            "    ])\n"
            "    assert t.equal(y, expected), f'value mismatch:\\n{y}'\n"
            "    # Must be a view, not a copy — same storage pointer.\n"
            "    assert y.data_ptr() == x.data_ptr(), 'rolling-window result should share storage with x (zero-copy view)'"
        ),
        "solution_body": (
            "def ex5_rolling_window(x: Tensor, w: int) -> Tensor:\n"
            "    n = x.shape[0]\n"
            "    s = x.stride(0)\n"
            "    return x.as_strided(size=(n - w + 1, w), stride=(s, s))"
        ),
        "solution_notes": (
            "**Reading the stride pair.**\n"
            "- Output stride for axis 0 = `s` (step from row `i` to row `i+1` = "
            "advance one element in source).\n"
            "- Output stride for axis 1 = `s` (step from column `j` to column "
            "`j+1` within a row = also advance one element).\n\n"
            "Both row and column advance one source-element. The 2-D view "
            "**overlaps** itself — every source element appears in multiple rows. "
            "That's the magic: O(1) memory, no copy.\n\n"
            "**Danger zone.** `as_strided` does NOT check bounds. If you pass a "
            "stride/size that walks off the end of storage, you get undefined "
            "behavior — silent garbage, or a segfault. Use the higher-level "
            "`torch.nn.functional.unfold` / `torch.tensor.unfold(dim, size, step)` "
            "for sliding windows in production code; this exercise is to "
            "understand what those calls do under the hood."
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
    "**What you'll practice.** Five patterns around tensor memory layout that ramp from reading strides → recognizing transpose breaks contiguity → seeing `.view()` fail on non-contig → fixing it with `.contiguous()` → building a zero-copy sliding-window view via `as_strided`. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "## Strides and contiguity — quick refresher",
    "",
    "**Stride** = number of elements to skip in storage to advance one step along that axis.",
    "- A contiguous `(H, W)` tensor has stride `(W, 1)`.",
    "- A contiguous `(B, C, H, W)` tensor has stride `(C*H*W, H*W, W, 1)`.",
    "",
    "**Contiguity** = the strides match the row-major layout of the current shape.",
    "- `.T` swaps strides but not data → the result is a view but not contiguous.",
    "- `.view()` requires contiguous input — it never copies.",
    "- `.reshape()` makes a view if possible, copies if not.",
    "- `.contiguous()` forces a row-major copy if the tensor isn't already contiguous.",
    "",
    "**`as_strided(size, stride)`** is the lowest-level view constructor — you provide the exact shape and stride pair. Bypasses all safety checks; trust the values you pass.",
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
                "ex5": "as_strided is an unsafe primitive; the integrative exercise tests stride mechanics without bounds-checking. Track student pass rate carefully — a failure may reflect API unfamiliarity rather than non-mastery of the underlying memory model.",
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
