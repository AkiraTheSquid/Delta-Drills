#!/usr/bin/env python3
"""Build the einops-einsum procedural drill notebook.

Generates `arena-procedural-drills/prereqs_einops/einops-einsum.ipynb` —
the fourth atom-keyed procedural drill (Phase 4.4). Mirrors the v0.2
template established by `build_einops_rearrange.py`.

Inherits build-time `verify_solutions()` gate — every canonical solution
must pass its in-notebook test or the build aborts before write.

Re-run this whenever you want to regenerate the notebook from the canonical
source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_einops/einops-einsum.ipynb"


# ----- build-time solution verification ---------------------------------

def verify_solutions(specs: list[dict]) -> None:
    try:
        import torch as t
        import numpy as np
        from torch import Tensor
        import einops
        from einops import einsum
    except ImportError as e:
        raise SystemExit(
            f"[build verify] missing runtime dep: {e}\n"
            f"  pip install torch numpy einops  # required for build-time solution verification\n"
            f"  refusing to write notebook with unverified solutions."
        )

    base_ns = {
        "t": t, "np": np, "Tensor": Tensor,
        "einops": einops, "einsum": einsum,
    }
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


ATOM_ID = "einops-einsum"
SUBTOPIC = "Einops: Deep Learning"
TITLE = "einops.einsum — procedural drill"
TEMPLATE_VERSION = "v0.2"

# NOTE on subtopic choice: the question bank stores einsum problems under
# topic="Einops" with one of {Rearrange, Reduce, Repeat, Deep Learning}.
# Einsum problems live in "Deep Learning" (it's the catch-all for the
# einsum/contraction patterns). Confirmed against
# This-Directory-Only/csv files of problems/einops_problems.csv.

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "einsum-elementwise",
        "kind": "component-skill",
        "description": "Multiply two tensors elementwise with `'i j, i j -> i j'` — every index appears on both sides of the comma and on the output. No contraction, no broadcast.",
    },
    {
        "id": "einsum-matmul-contraction",
        "kind": "component-skill",
        "description": "Contract a single repeated index between two operands and drop it from the output — `'i k, k j -> i j'` is matrix multiply. Repeated input index that does NOT appear on output = sum-reduction.",
    },
    {
        "id": "einsum-reduce-via-omit",
        "kind": "component-skill",
        "description": "Drop an axis from the output to reduce over it. `'i j -> i'` is row-sum, `'i j ->'` is total sum. No second operand needed.",
    },
    {
        "id": "einsum-batched",
        "kind": "component-skill",
        "description": "Carry a non-contracted index through both operands and the output — `'b i k, b k j -> b i j'` is batched matmul. The `b` axis is preserved (broadcast), `k` is contracted.",
    },
    {
        "id": "einsum-attention-scores",
        "kind": "integrative-skill",
        "description": "Combine batching + reduction-via-omission + the matmul pattern in a single contraction — the attention QK^T case. The integrative KC — relies on fluency in the four component KCs above.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "elementwise product (Hadamard)",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["elementwise", "hadamard", "no-contraction"],
        "kcs": ["einsum-elementwise"],
        "lo": "Recall that an einsum pattern with every index on every side performs elementwise multiplication.",
        "prompt_body": (
            "Implement `ex1_hadamard(x, y)` to compute the elementwise product of two "
            "2-D tensors of the same shape.\n\n"
            "Input shapes: both `(i, j)`. Output shape: `(i, j)`.\n\n"
            "Use `einops.einsum(...)` — not `x * y`. The point is to write the "
            "pattern. Every index that appears on the output must also appear in "
            "every input — no contraction, no broadcast."
        ),
        "stub": (
            "def ex1_hadamard(x: Tensor, y: Tensor) -> Tensor:\n"
            '    """Elementwise product of two (i, j) tensors via einsum."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12).reshape(3, 4).float()\n"
            "    y = t.arange(12, 24).reshape(3, 4).float()\n"
            "    z = ex1_hadamard(x, y)\n"
            "    assert z.shape == (3, 4), f'expected (3,4), got {z.shape}'\n"
            "    assert t.equal(z, x * y), 'values differ from x * y'"
        ),
        "solution_body": (
            "def ex1_hadamard(x: Tensor, y: Tensor) -> Tensor:\n"
            "    return einsum(x, y, 'i j, i j -> i j')"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex2",
        "title": "matrix multiplication (single-index contraction)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["matmul", "contraction", "shared-index"],
        "kcs": ["einsum-matmul-contraction"],
        "lo": "Apply the einsum convention to perform matrix multiplication by contracting the shared inner index.",
        "prompt_body": (
            "Implement `ex2_matmul(x, y)` for the standard matrix product.\n\n"
            "Input shapes: `x` is `(i, k)`, `y` is `(k, j)`. Output shape: `(i, j)`.\n\n"
            "The inner index `k` appears in both inputs but **not** on the output — "
            "that's einsum's signal to sum-contract over it. Equivalent to `x @ y`."
        ),
        "stub": (
            "def ex2_matmul(x: Tensor, y: Tensor) -> Tensor:\n"
            '    """Matrix multiply (i, k) @ (k, j) → (i, j) via einsum."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3).reshape(2, 3).float()\n"
            "    y = t.arange(3 * 4).reshape(3, 4).float()\n"
            "    z = ex2_matmul(x, y)\n"
            "    assert z.shape == (2, 4), f'expected (2,4), got {z.shape}'\n"
            "    assert t.allclose(z, x @ y), 'values differ from x @ y'"
        ),
        "solution_body": (
            "def ex2_matmul(x: Tensor, y: Tensor) -> Tensor:\n"
            "    return einsum(x, y, 'i k, k j -> i j')"
        ),
        "solution_notes": (
            "**Why does `k` disappear?** The einsum rule: every axis name that "
            "appears on input but **not** on output is sum-reduced. `k` appears in "
            "both `x` and `y` (so values get multiplied pointwise along that axis), "
            "and the absence on the output side triggers the sum. Net result: "
            "`z[i, j] = sum_k x[i, k] * y[k, j]`."
        ),
    },
    {
        "id": "ex3",
        "title": "row sum (omit-to-reduce)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["reduction", "omit-axis", "single-operand"],
        "kcs": ["einsum-reduce-via-omit"],
        "lo": "Apply the omit-to-reduce convention with a single operand to compute a row sum.",
        "prompt_body": (
            "Implement `ex3_row_sum(x)` to sum each row of a 2-D tensor.\n\n"
            "Input shape: `(i, j)`. Output shape: `(i,)`.\n\n"
            "Use `einops.einsum` with a **single** operand. The axis you want to "
            "reduce just gets dropped from the output side of the pattern — there's "
            "no `sum` keyword.\n\n"
            "Equivalent to `x.sum(dim=1)`."
        ),
        "stub": (
            "def ex3_row_sum(x: Tensor) -> Tensor:\n"
            '    """Sum each row. (i, j) → (i,) via einsum (single operand)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(3 * 4).reshape(3, 4).float()\n"
            "    z = ex3_row_sum(x)\n"
            "    assert z.shape == (3,), f'expected (3,), got {z.shape}'\n"
            "    assert t.allclose(z, x.sum(dim=1)), 'values differ from x.sum(dim=1)'"
        ),
        "solution_body": (
            "def ex3_row_sum(x: Tensor) -> Tensor:\n"
            "    return einsum(x, 'i j -> i')"
        ),
        "solution_notes": (
            "**Single-operand einsum** is just the reduction case of the general "
            "convention: any axis name on the input that's missing from the output "
            "is summed. `'i j -> i'` sums over `j`. `'i j ->'` (empty output) would "
            "sum both axes and return a scalar."
        ),
    },
    {
        "id": "ex4",
        "title": "batched matrix multiply (preserve a batch axis)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["batched", "matmul", "preserve-axis"],
        "kcs": ["einsum-batched"],
        "lo": "Apply einsum to batched matmul by carrying a non-contracted axis through both operands and the output.",
        "prompt_body": (
            "Implement `ex4_batched_matmul(x, y)` for a batched matmul.\n\n"
            "Input shapes: `x` is `(b, i, k)`, `y` is `(b, k, j)`. Output shape: "
            "`(b, i, j)`.\n\n"
            "The `b` axis appears in both inputs **and** on the output — that's how "
            "einsum carries it through. The `k` axis still contracts. Equivalent to "
            "`torch.bmm(x, y)` or `x @ y` (PyTorch broadcasts the last two dims)."
        ),
        "stub": (
            "def ex4_batched_matmul(x: Tensor, y: Tensor) -> Tensor:\n"
            '    """Batched matmul (b, i, k) @ (b, k, j) → (b, i, j) via einsum."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3 * 4).reshape(2, 3, 4).float()\n"
            "    y = t.arange(2 * 4 * 5).reshape(2, 4, 5).float()\n"
            "    z = ex4_batched_matmul(x, y)\n"
            "    assert z.shape == (2, 3, 5), f'expected (2,3,5), got {z.shape}'\n"
            "    assert t.allclose(z, x @ y), 'values differ from x @ y'"
        ),
        "solution_body": (
            "def ex4_batched_matmul(x: Tensor, y: Tensor) -> Tensor:\n"
            "    return einsum(x, y, 'b i k, b k j -> b i j')"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex5",
        "title": "attention scores QK^T (batched + reduce + matmul-like)",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["attention", "qkt", "integration", "multi-kc"],
        "kcs": [
            "einsum-matmul-contraction",
            "einsum-batched",
            "einsum-attention-scores",
        ],
        "lo": "Synthesize batched einsum with index-contraction to produce attention scores (QK^T) without reshaping.",
        "prompt_body": (
            "Implement `ex5_attention_scores(q, k)` to compute pre-softmax attention "
            "scores.\n\n"
            "Input shapes: `q` is `(b, q_len, d)`, `k` is `(b, k_len, d)`. Output "
            "shape: `(b, q_len, k_len)`. Each `out[b, i, j]` is the inner product "
            "`sum_d q[b, i, d] * k[b, j, d]`.\n\n"
            "This is QK^T from a Transformer attention head, batched over `b`. "
            "Notice **you do not transpose** `k` — einsum handles the index "
            "alignment for you. `d` appears on both inputs and not on the output, "
            "so it's contracted.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3+ KCs in one pattern; "
            "empirical work (Lohr et al. ITiCSE 2025) shows 3-concept LLM-generated "
            "exercises drop from ~94% to ~40% solvability. Expect a step in "
            "difficulty here vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_attention_scores(q: Tensor, k: Tensor) -> Tensor:\n"
            '    """Attention scores. (b, q_len, d) @ (b, k_len, d)^T → (b, q_len, k_len).\n'
            "\n"
            "    Each out[b, i, j] = sum_d q[b, i, d] * k[b, j, d].\n"
            '    """\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "b, q_len, k_len, d = 2, 3, 5, 4\n"
            "    q = t.randn(b, q_len, d)\n"
            "    k = t.randn(b, k_len, d)\n"
            "    out = ex5_attention_scores(q, k)\n"
            "    assert out.shape == (b, q_len, k_len), f'expected ({b},{q_len},{k_len}), got {out.shape}'\n"
            "    # Ground truth via explicit batched matmul with manual transpose.\n"
            "    expected = q @ k.transpose(-2, -1)\n"
            "    assert t.allclose(out, expected, atol=1e-5), 'values differ from q @ k.transpose(-2,-1)'"
        ),
        "solution_body": (
            "def ex5_attention_scores(q: Tensor, k: Tensor) -> Tensor:\n"
            "    return einsum(q, k, 'b q d, b k d -> b q k')"
        ),
        "solution_notes": (
            "**Reading the pattern.**\n"
            "- `b` appears in both inputs and on the output → carried through (batch).\n"
            "- `q` only appears in the first input and on the output → preserved.\n"
            "- `k` only appears in the second input and on the output → preserved.\n"
            "- `d` appears in both inputs but **not** on the output → contracted "
            "(this is the dot product).\n\n"
            "**Why this is the integrative case.** You're simultaneously batching "
            "(KC #4), preserving two independent non-contracted axes from different "
            "operands (extends KC #2 from `i,k → k,j` to a non-square layout), and "
            "letting omitting `d` do the reduction (KC #3). No transpose, no "
            "rearrange, no reshape — the pattern string carries the full intent."
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
    "**What you'll practice.** Five `einops.einsum` patterns that ramp from elementwise product → matrix multiply → omit-to-reduce → batched matmul → attention QK^T. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
    "",
    "**Per-exercise structure** (Doughty et al. ACE 2024 — `[Bloom level] + [LO] + [Keywords] + [KCs]`):",
    "Each exercise begins with a yaml block stating its Bloom cognitive level, learning objective, keywords, and the knowledge components (KCs) it targets. This makes the cognitive demand explicit instead of buried.",
))

cells.append(md("## Setup"))

cells.append(code(
    "import numpy as np",
    "import torch as t",
    "from torch import Tensor",
    "import einops",
    "from einops import einsum",
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
    "## einops.einsum — quick refresher",
    "",
    "`einsum(*tensors, pattern)` performs sum-contraction over named indices:",
    "1. **Elementwise** — `'i j, i j -> i j'` multiplies pointwise (no reduction).",
    "2. **Matmul** — `'i k, k j -> i j'` contracts the shared `k` (sum-reduce).",
    "3. **Single-operand reduce** — `'i j -> i'` sums over `j` (no second tensor needed).",
    "4. **Batched** — `'b i k, b k j -> b i j'` carries `b` through, contracts `k`.",
    "",
    "**The two rules:**",
    "- An index that appears on input AND output → preserved (broadcast-like).",
    "- An index that appears on input but NOT on output → sum-contracted.",
))

# Emit per-exercise: [header md, code stub+test, solution md]
for i, spec in enumerate(EXERCISE_SPECS, start=1):
    cells.append(exercise_header_md(spec, i))
    cells.append(exercise_code(spec))
    cells.append(exercise_solution_md(spec))

# Completion beacon
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
    "    if num_passed == 5: return 'not_much'   # 5/5 → felt easy",
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

# ----- assemble notebook -----

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

# Build-time gate: every canonical solution must pass its in-notebook
# assertions before we emit the notebook. SystemExit(1) on failure.
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
