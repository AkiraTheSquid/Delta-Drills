#!/usr/bin/env python3
"""Build the einops-reduce procedural drill notebook.

Generates `arena-procedural-drills/prereqs_einops/einops-reduce.ipynb` —
the second atom-keyed procedural drill (Phase 4.2). Mirrors the v0.2
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
OUT = REPO / "arena-procedural-drills/prereqs_einops/einops-reduce.ipynb"


# ----- build-time solution verification ---------------------------------
# Run every canonical solution against its in-notebook test BEFORE writing
# the notebook. Same exec/assert pattern as scripts/validate_arena_solutions.py
# from Phase 2f-ii (#95) — applied at build time instead of harvest time.

def verify_solutions(specs: list[dict]) -> None:
    try:
        import torch as t
        import numpy as np
        from torch import Tensor
        import torch.nn.functional as F
        import einops
        from einops import reduce
    except ImportError as e:
        raise SystemExit(
            f"[build verify] missing runtime dep: {e}\n"
            f"  pip install torch numpy einops  # required for build-time solution verification\n"
            f"  refusing to write notebook with unverified solutions."
        )

    base_ns = {
        "t": t, "np": np, "Tensor": Tensor, "F": F,
        "einops": einops, "reduce": reduce,
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


ATOM_ID = "einops-reduce"
SUBTOPIC = "Einops: Reduce"
TITLE = "einops.reduce — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "reduce-pick-aggregator",
        "kind": "component-skill",
        "description": "Collapse a single named axis using the right reduction string ('mean' / 'sum' / 'max' / 'min'). Verifies the basic `reduce(x, pattern, op)` call shape.",
    },
    {
        "id": "reduce-multi-axis",
        "kind": "component-skill",
        "description": "Drop multiple named axes in one call (the global-pool case). Requires understanding that any name on the left that does not appear on the right is reduced over.",
    },
    {
        "id": "reduce-keepdim-broadcast",
        "kind": "component-skill",
        "description": "Use `()` on the output side to keep a size-1 placeholder where an axis was collapsed, so the result broadcasts back against the input.",
    },
    {
        "id": "reduce-with-decomposition",
        "kind": "component-skill",
        "description": "Combine `(h h2)` axis decomposition on the left with reduction over the inner factor — the canonical pooling pattern.",
    },
    {
        "id": "reduce-normalize-pattern",
        "kind": "integrative-skill",
        "description": "Combine reduction + keepdim + broadcast arithmetic to normalize a tensor along an axis (subtract-max, divide-by-sum, etc.). The integrative KC — relies on fluency in the four component KCs above.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "channel mean",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["aggregator", "single-axis-drop", "mean"],
        "kcs": ["reduce-pick-aggregator"],
        "lo": "Recall the `reduce(x, pattern, op)` call shape for collapsing a single named axis.",
        "prompt_body": (
            "Implement `ex1_channel_mean(x)` to average a `(h, w, c)` image over its "
            "channel axis. Output shape: `(h, w)`.\n\n"
            "Use `einops.reduce(...)` with the `'mean'` reduction — not `x.mean(dim=-1)`. "
            "The point is to write the pattern."
        ),
        "stub": (
            "def ex1_channel_mean(x: Tensor) -> Tensor:\n"
            '    """Reduce `x` of shape (h, w, c) to (h, w) by averaging channels."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(4 * 5 * 3).reshape(4, 5, 3).float()\n"
            "    y = ex1_channel_mean(x)\n"
            "    assert y.shape == (4, 5), f'expected (4,5), got {y.shape}'\n"
            "    assert t.allclose(y, x.mean(dim=-1)), 'values differ from x.mean(dim=-1)'"
        ),
        "solution_body": (
            "def ex1_channel_mean(x: Tensor) -> Tensor:\n"
            "    return reduce(x, 'h w c -> h w', 'mean')"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex2",
        "title": "per-image global mean (multi-axis drop)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["multi-axis-drop", "global-pool", "mean"],
        "kcs": ["reduce-multi-axis"],
        "lo": "Apply the reduce pattern to collapse three axes in a single call (global-mean per batch item).",
        "prompt_body": (
            "Implement `ex2_global_mean(x)` to compute one scalar per image: the mean "
            "of all channel × spatial values.\n\n"
            "Input shape: `(b, c, h, w)`. Output shape: `(b,)`.\n\n"
            "Any axis name that appears on the left but **not** on the right is reduced "
            "over — so you drop `c`, `h`, and `w` at once."
        ),
        "stub": (
            "def ex2_global_mean(x: Tensor) -> Tensor:\n"
            '    """Reduce (b, c, h, w) → (b,) by averaging over c, h, w."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()\n"
            "    y = ex2_global_mean(x)\n"
            "    assert y.shape == (2,), f'expected (2,), got {y.shape}'\n"
            "    assert t.allclose(y, x.mean(dim=(1, 2, 3))), 'values differ from x.mean(dim=(1,2,3))'"
        ),
        "solution_body": (
            "def ex2_global_mean(x: Tensor) -> Tensor:\n"
            "    return reduce(x, 'b c h w -> b', 'mean')"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex3",
        "title": "per-image spatial max (keepdim with ())",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["keepdim", "placeholder-axis", "max"],
        "kcs": ["reduce-keepdim-broadcast"],
        "lo": "Apply `()` on the output side to preserve a collapsed axis as size-1 for downstream broadcasting.",
        "prompt_body": (
            "Implement `ex3_spatial_max(x)` to compute the per-(batch, channel) "
            "spatial maximum, keeping the H and W axes as size-1 placeholders so "
            "the result broadcasts back against `x`.\n\n"
            "Input shape: `(b, c, h, w)`. Output shape: `(b, c, 1, 1)`.\n\n"
            "Use `()` on the right side of the pattern wherever you want a size-1 "
            "axis to remain instead of being dropped."
        ),
        "stub": (
            "def ex3_spatial_max(x: Tensor) -> Tensor:\n"
            '    """Reduce (b, c, h, w) → (b, c, 1, 1) by taking max over h, w."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()\n"
            "    y = ex3_spatial_max(x)\n"
            "    assert y.shape == (2, 3, 1, 1), f'expected (2,3,1,1), got {y.shape}'\n"
            "    assert t.allclose(y, x.amax(dim=(2, 3), keepdim=True)), 'values differ from amax(keepdim=True)'\n"
            "    # Should broadcast — (x - y) must produce no shape error and have x's shape.\n"
            "    diff = x - y\n"
            "    assert diff.shape == x.shape, f'broadcast failed: diff shape {diff.shape}'"
        ),
        "solution_body": (
            "def ex3_spatial_max(x: Tensor) -> Tensor:\n"
            "    return reduce(x, 'b c h w -> b c () ()', 'max')"
        ),
        "solution_notes": (
            "**Why `()`?** Without it the pattern would be `'b c h w -> b c'` and the "
            "result would have shape `(b, c)` — same values, but it would not broadcast "
            "back against `(b, c, h, w)` because the trailing axes are missing. `()` is "
            "the einops equivalent of `keepdim=True`."
        ),
    },
    {
        "id": "ex4",
        "title": "2×2 average pool (axis decomposition + reduce)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["pooling", "decomposition", "kwarg-binding"],
        "kcs": ["reduce-with-decomposition"],
        "lo": "Apply axis decomposition `(h h2)` on the input side combined with reduction over the inner factor — the canonical average-pool pattern.",
        "prompt_body": (
            "Implement `ex4_avg_pool_2x2(x)` to 2×2-average-pool a BCHW tensor.\n\n"
            "Input shape: `(b, c, H, W)` where `H` and `W` are even. Output shape: "
            "`(b, c, H/2, W/2)`.\n\n"
            "Decompose `H` into `(h h2)` and `W` into `(w w2)` on the **left** side. "
            "Pass `h2=2, w2=2` as kwargs. On the **right** side keep only `h` and `w` "
            "— the `h2` and `w2` axes get reduced over.\n\n"
            "Equivalent to `torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)`."
        ),
        "stub": (
            "def ex4_avg_pool_2x2(x: Tensor) -> Tensor:\n"
            '    """2×2 average pool. (b, c, H, W) → (b, c, H/2, W/2)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4).float()\n"
            "    y = ex4_avg_pool_2x2(x)\n"
            "    assert y.shape == (2, 3, 2, 2), f'expected (2,3,2,2), got {y.shape}'\n"
            "    expected = F.avg_pool2d(x, kernel_size=2, stride=2)\n"
            "    assert t.allclose(y, expected), 'values differ from F.avg_pool2d(kernel=2, stride=2)'"
        ),
        "solution_body": (
            "def ex4_avg_pool_2x2(x: Tensor) -> Tensor:\n"
            "    return reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)"
        ),
        "solution_notes": (
            "**Why pass `h2=` and `w2=`?** When you decompose with `(h h2)`, einops "
            "needs to know one of the two sizes — the other is inferred from `H`. "
            "Naming the inner factor `h2` and binding it via kwarg fixes the pool "
            "window size."
        ),
    },
    {
        "id": "ex5",
        "title": "row-wise softmax stabilization (reduce + keepdim + broadcast)",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["softmax-stabilize", "broadcast-subtract", "integration", "multi-kc"],
        "kcs": [
            "reduce-pick-aggregator",
            "reduce-keepdim-broadcast",
            "reduce-normalize-pattern",
        ],
        "lo": "Synthesize reduce + keepdim + broadcast subtraction to perform numerically-stable softmax preprocessing.",
        "prompt_body": (
            "Implement `ex5_softmax_stabilize(x)` to subtract the per-row maximum from "
            "every element. This is the standard pre-softmax stabilization step that "
            "keeps `exp(x)` from overflowing.\n\n"
            "Input shape: `(b, n)`. Output shape: `(b, n)`. After your transform, every "
            "row's maximum should be exactly `0`.\n\n"
            "Use `einops.reduce` with `()` on the row axis so the per-row max keeps a "
            "size-1 placeholder, then broadcast-subtract.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3+ KCs in one expression; "
            "empirical work (Lohr et al. ITiCSE 2025) shows 3-concept LLM-generated "
            "exercises drop from ~94% to ~40% solvability. Expect a step in difficulty "
            "here vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_softmax_stabilize(x: Tensor) -> Tensor:\n"
            '    """Subtract per-row max from x. (b, n) → (b, n).\n'
            "\n"
            "    After this transform, every row of the result has max == 0.\n"
            '    """\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([[1.0, 3.0, 2.0, 5.0], [10.0, 7.0, 8.0, 9.0], [-1.0, -3.0, 0.0, -2.0]])\n"
            "    y = ex5_softmax_stabilize(x)\n"
            "    assert y.shape == x.shape, f'shape mismatch: {y.shape} vs {x.shape}'\n"
            "    row_max = y.amax(dim=1)\n"
            "    assert t.allclose(row_max, t.zeros(3)), f'row maxes should all be 0, got {row_max}'\n"
            "    # Softmax-invariant: softmax(x) == softmax(x - row_max)\n"
            "    assert t.allclose(x.softmax(dim=1), y.softmax(dim=1)), 'softmax-invariance broken'"
        ),
        "solution_body": (
            "def ex5_softmax_stabilize(x: Tensor) -> Tensor:\n"
            "    row_max = reduce(x, 'b n -> b ()', 'max')\n"
            "    return x - row_max"
        ),
        "solution_notes": (
            "**Reading the pattern.**\n"
            "- `'b n -> b ()'` reduces over `n` and keeps a size-1 placeholder, giving "
            "`row_max` shape `(b, 1)`.\n"
            "- `x - row_max` then broadcasts the per-row max across all columns.\n\n"
            "**Why this matters.** `exp(x_i)` for large positive `x_i` overflows. "
            "Subtracting the per-row max is mathematically a no-op for softmax (numerator "
            "and denominator both scale by `exp(-row_max)`) but keeps every exponent "
            "≤ 0, so `exp` stays in `(0, 1]`."
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
    "**What you'll practice.** Five `einops.reduce` patterns that ramp from single-axis mean → multi-axis global pool → keepdim broadcast → decomposed average pool → softmax stabilization. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
    "",
    "**Per-exercise structure** (Doughty et al. ACE 2024 — `[Bloom level] + [LO] + [Keywords] + [KCs]`):",
    "Each exercise begins with a yaml block stating its Bloom cognitive level, learning objective, keywords, and the knowledge components (KCs) it targets. This makes the cognitive demand explicit instead of buried.",
))

cells.append(md("## Setup"))

cells.append(code(
    "import numpy as np",
    "import torch as t",
    "from torch import Tensor",
    "import torch.nn.functional as F",
    "import einops",
    "from einops import reduce",
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
    "## einops.reduce — quick refresher",
    "",
    "`reduce(tensor, pattern, reduction, **axes_lengths)` collapses named axes:",
    "1. **Single-axis drop** — `'h w c -> h w'` with `'mean'` averages channels.",
    "2. **Multi-axis drop** — `'b c h w -> b'` reduces three axes at once.",
    "3. **Keepdim placeholder** — `'b c h w -> b c () ()'` keeps size-1 axes for broadcasting.",
    "4. **Decompose-then-reduce** — `'b c (h h2) (w w2) -> b c h w'` with `h2=2, w2=2` does 2×2 pooling.",
    "",
    "Reduction strings: `'mean'`, `'sum'`, `'max'`, `'min'`, `'prod'`, `'any'`, `'all'`, or a callable.",
    "Any axis that appears on the left but not on the right is reduced over.",
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
