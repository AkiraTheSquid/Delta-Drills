#!/usr/bin/env python3
"""Build the einops-rearrange procedural drill notebook.

Generates `arena-procedural-drills/prereqs_einops/einops-rearrange.ipynb` —
the first atom-keyed procedural drill (Phase 4). Defines the template for
all future procedural drills.

v0.2 — adopts the Doughty et al. (ACE 2024) per-exercise prompt structure
(LO + Bloom level + Keywords + KCs in each exercise header) and emits a
machine-readable `metadata.delta_drills.exercises[]` array so the
orchestrator can do variety control + scaffold selection downstream.

Re-run this whenever you want to regenerate the notebook from the canonical
source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_einops/einops-rearrange.ipynb"


# ----- build-time solution verification ---------------------------------
# Run every canonical solution against its in-notebook test BEFORE writing
# the notebook. If any exercise fails (shape mismatch, value mismatch,
# ImportError, syntax error in stub/solution drift) the build aborts.
# Same exec/assert pattern as scripts/validate_arena_solutions.py from
# Phase 2f-ii (#95) — applied at build time instead of harvest time.

def verify_solutions(specs: list[dict]) -> None:
    try:
        import torch as t
        import numpy as np
        from torch import Tensor
        import einops
        from einops import rearrange
    except ImportError as e:
        raise SystemExit(
            f"[build verify] missing runtime dep: {e}\n"
            f"  pip install torch numpy einops  # required for build-time solution verification\n"
            f"  refusing to write notebook with unverified solutions."
        )

    base_ns = {
        "t": t, "np": np, "Tensor": Tensor,
        "einops": einops, "rearrange": rearrange,
    }
    failures: list[str] = []
    for spec in specs:
        ns = dict(base_ns)
        # 1. exec the canonical solution to define the function under test
        try:
            exec(spec["solution_body"], ns)
        except Exception as e:
            failures.append(f"{spec['id']} ({spec['title']}) — solution_body did not compile: {e!r}")
            continue
        # 2. build a test function mirroring exercise_code() in the notebook
        test_src = f"def _test_{spec['id']}():\n    {spec['test_body']}"
        try:
            exec(test_src, ns)
        except Exception as e:
            failures.append(f"{spec['id']} ({spec['title']}) — test_body did not compile: {e!r}")
            continue
        # 3. call the test against the canonical solution
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

ATOM_ID = "einops-rearrange"
SUBTOPIC = "Einops: Rearrange"
TITLE = "einops.rearrange — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------
# The atom `einops-rearrange` decomposes into 5 component-skill KCs in the
# KLI sense. Each exercise targets one or more KCs.

KC_DECOMPOSITION = [
    {
        "id": "rearrange-identity-pattern",
        "kind": "component-skill",
        "description": "Write a rearrange pattern that names every axis on both sides without reordering or transforming them. Verifies syntactic understanding of the pattern string.",
    },
    {
        "id": "rearrange-axis-swap",
        "kind": "component-skill",
        "description": "Use rearrange to permute axes (the transpose case). Requires understanding that axis identifiers carry through, not positions.",
    },
    {
        "id": "rearrange-axis-composition",
        "kind": "component-skill",
        "description": "Use `(a b c)` on the output side to flatten N axes into one, in row-major order. Requires understanding the composition stride convention.",
    },
    {
        "id": "rearrange-axis-decomposition",
        "kind": "component-skill",
        "description": "Use `(a b)` on the input side with a kwarg-bound size to split one axis into two named axes. Requires understanding why one side of the pair must be size-bound.",
    },
    {
        "id": "rearrange-combined-patterns",
        "kind": "integrative-skill",
        "description": "Combine decomposition + reordering + composition in one pattern (the ViT patch-embedding case). The integrative KC — relies on fluency in the four component KCs above.",
    },
]

# ----- exercise specs ---------------------------------------------------
# Each spec is the canonical source of truth for one notebook exercise.
# Doughty et al. ACE 2024 found explicit LO + Bloom level + Keywords in
# the prompt cuts LO-misalignment from ~12% to ~4.8%. Emitting those into
# the per-exercise markdown header — and into metadata.delta_drills.
# exercises[] for the orchestrator — is the minimum viable adoption.

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "identity rearrange",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["identity-pattern", "axis-naming"],
        "kcs": ["rearrange-identity-pattern"],
        "lo": "Recall the pattern syntax for an identity rearrange of a 2-D tensor.",
        "prompt_body": (
            "Implement `ex1_identity(x)` so it returns `x` rearranged by a pattern "
            "that leaves the layout unchanged. The input is a 2-D tensor of shape "
            "`(b, c)`.\n\nUse `einops.rearrange` (not `.clone()` or `.contiguous()`) "
            "— the point is to write the pattern."
        ),
        "stub": (
            "def ex1_identity(x: Tensor) -> Tensor:\n"
            '    """Rearrange `x` of shape (b, c) to the same shape (b, c)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12).reshape(3, 4)\n"
            "    y = ex1_identity(x)\n"
            "    assert y.shape == x.shape, f'shape mismatch: {y.shape} vs {x.shape}'\n"
            "    assert t.equal(y, x), 'values differ'"
        ),
        "solution_body": (
            "def ex1_identity(x: Tensor) -> Tensor:\n"
            "    return rearrange(x, 'b c -> b c')"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex2",
        "title": "axis swap (transpose)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["transpose", "axis-renaming", "permutation"],
        "kcs": ["rearrange-axis-swap"],
        "lo": "Apply the rearrange pattern syntax to perform a 2-D transpose.",
        "prompt_body": (
            "Implement `ex2_swap(x)` to swap the two axes of a 2-D tensor. Input "
            "shape `(rows, cols)`, output shape `(cols, rows)`.\n\nThis is "
            "equivalent to `x.T` — but write it as a rearrange pattern."
        ),
        "stub": (
            "def ex2_swap(x: Tensor) -> Tensor:\n"
            '    """Rearrange `x` of shape (rows, cols) to shape (cols, rows)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12).reshape(3, 4)\n"
            "    y = ex2_swap(x)\n"
            "    assert y.shape == (4, 3), f'expected (4,3), got {y.shape}'\n"
            "    assert t.equal(y, x.T), 'values differ from x.T'"
        ),
        "solution_body": (
            "def ex2_swap(x: Tensor) -> Tensor:\n"
            "    return rearrange(x, 'rows cols -> cols rows')"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex3",
        "title": "image flatten (axis composition)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["composition", "flatten", "row-major"],
        "kcs": ["rearrange-axis-composition"],
        "lo": "Apply axis composition `(c h w)` on the output side to flatten a 4-D tensor's trailing axes in row-major order.",
        "prompt_body": (
            "Implement `ex3_flatten(x)` to flatten a batch of CHW images into a "
            "batch of feature vectors.\n\n"
            "Input shape: `(b, c, h, w)`. Output shape: `(b, c * h * w)`.\n\n"
            "Use a **composed** axis on the right side: `(c h w)` collapses three "
            "named axes into one. Row-major order — channel varies slowest, width "
            "varies fastest."
        ),
        "stub": (
            "def ex3_flatten(x: Tensor) -> Tensor:\n"
            '    """Rearrange (b, c, h, w) → (b, c*h*w) by composing the trailing 3 axes."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()\n"
            "    y = ex3_flatten(x)\n"
            "    assert y.shape == (2, 60), f'expected (2,60), got {y.shape}'\n"
            "    # Row-major flatten should match torch.reshape exactly.\n"
            "    assert t.equal(y, x.reshape(2, 60)), 'values differ from torch.reshape'"
        ),
        "solution_body": (
            "def ex3_flatten(x: Tensor) -> Tensor:\n"
            "    return rearrange(x, 'b c h w -> b (c h w)')"
        ),
        "solution_notes": (
            "**Why row-major?** `einops` composes axes in the order written. "
            "`(c h w)` means the stride pattern is `(c × h × w, h × w, w, 1)` — "
            "equivalent to `torch.reshape` on a contiguous tensor."
        ),
    },
    {
        "id": "ex4",
        "title": "batch unfold (axis decomposition)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["decomposition", "kwarg-binding", "micro-batching"],
        "kcs": ["rearrange-axis-decomposition"],
        "lo": "Apply axis decomposition `(a b)` on the input side with a kwarg-bound size to split one axis into two.",
        "prompt_body": (
            "Implement `ex4_unfold(x, micro_batch_size)` to split the leading "
            "batch dimension into a `(num_micro, micro_batch_size)` pair.\n\n"
            "Input shape: `(B, c)` where `B = num_micro × micro_batch_size`. "
            "Output shape: `(num_micro, micro_batch_size, c)`.\n\n"
            "This is **decomposition** — one axis on the left becomes a "
            "parenthesized pair, with one side bound via a keyword argument. "
            "You'll need to pass `micro_batch_size` into `rearrange` as a named "
            "axis length."
        ),
        "stub": (
            "def ex4_unfold(x: Tensor, micro_batch_size: int) -> Tensor:\n"
            '    """Rearrange (B, c) → (num_micro, micro_batch_size, c).\n'
            "\n"
            "    Assumes B is divisible by micro_batch_size.\n"
            '    """\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(12 * 5).reshape(12, 5).float()\n"
            "    y = ex4_unfold(x, micro_batch_size=4)\n"
            "    assert y.shape == (3, 4, 5), f'expected (3,4,5), got {y.shape}'\n"
            "    # The first micro-batch should be the first 4 rows of x.\n"
            "    assert t.equal(y[0], x[:4]), 'micro-batch 0 does not match x[:4]'\n"
            "    assert t.equal(y[1], x[4:8]), 'micro-batch 1 does not match x[4:8]'"
        ),
        "solution_body": (
            "def ex4_unfold(x: Tensor, micro_batch_size: int) -> Tensor:\n"
            "    return rearrange(x, '(num_micro mb) c -> num_micro mb c', mb=micro_batch_size)"
        ),
        "solution_notes": (
            "**Why pass `mb=`?** When you decompose an axis with `(a b)`, einops "
            "needs to know one of the two sizes — the other is inferred from the "
            "total length. The naming on left and right just needs to be consistent."
        ),
    },
    {
        "id": "ex5",
        "title": "patch grid (ViT-style patch embedding prep)",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["patchify", "vit", "integration", "multi-kc"],
        "kcs": [
            "rearrange-axis-decomposition",
            "rearrange-axis-composition",
            "rearrange-combined-patterns",
        ],
        "lo": "Synthesize axis decomposition, reordering, and composition into a single rearrange pattern that produces a ViT patch-embedding layout.",
        "prompt_body": (
            "Implement `ex5_patchify(x, patch_size)` to break a batch of images "
            "into a flat sequence of patches.\n\n"
            "Input shape: `(b, c, H, W)` where `H` and `W` are both divisible by "
            "`patch_size`. Output shape: "
            "`(b, num_patches, patch_size * patch_size * c)` where "
            "`num_patches = (H // patch_size) * (W // patch_size)`.\n\n"
            "This combines **decomposition** (split each spatial axis into "
            "`(h p1)` and `(w p2)`), **reordering** (move patch dims after grid "
            "dims), and **composition** (flatten grid into a sequence and pixels "
            "into a feature vector). It's the operation at the start of a Vision "
            "Transformer.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3+ KCs in one pattern; "
            "empirical work (Lohr et al. ITiCSE 2025) shows 3-concept LLM-generated "
            "exercises drop from ~94% to ~40% solvability. Expect a step in "
            "difficulty here vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_patchify(x: Tensor, patch_size: int) -> Tensor:\n"
            '    """Rearrange (b, c, H, W) → (b, num_patches, patch_size*patch_size*c).\n'
            "\n"
            "    Patch order: row-major over the (h, w) patch grid.\n"
            "    Pixel order inside a patch: row-major over (p1, p2), then channel.\n"
            '    """\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "b, c, H, W, p = 2, 3, 8, 8, 4\n"
            "    x = t.arange(b * c * H * W).reshape(b, c, H, W).float()\n"
            "    y = ex5_patchify(x, patch_size=p)\n"
            "    num_patches = (H // p) * (W // p)\n"
            "    feat = p * p * c\n"
            "    assert y.shape == (b, num_patches, feat), f'expected ({b},{num_patches},{feat}), got {y.shape}'\n"
            "\n"
            "    # Round-trip: the first patch of the first image should be the top-left\n"
            "    # p×p block across all c channels, flattened in (p1, p2, c) order.\n"
            "    top_left = x[0, :, :p, :p]                      # (c, p, p)\n"
            "    expected_patch0 = rearrange(top_left, 'c p1 p2 -> (p1 p2 c)')\n"
            "    assert t.equal(y[0, 0], expected_patch0), 'first patch does not match top-left block'"
        ),
        "solution_body": (
            "def ex5_patchify(x: Tensor, patch_size: int) -> Tensor:\n"
            "    return rearrange(\n"
            "        x,\n"
            "        'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',\n"
            "        p1=patch_size, p2=patch_size,\n"
            "    )"
        ),
        "solution_notes": (
            "**Reading the pattern.**\n"
            "- `(h p1)` and `(w p2)` decompose H and W into (grid, patch) factor "
            "pairs. Pass `p1=` and `p2=` so the grid sizes can be inferred.\n"
            "- `(h w)` on the right composes the grid into the sequence axis.\n"
            "- `(p1 p2 c)` composes patch pixels and channels into the per-token "
            "feature vector. The order `(p1 p2 c)` matters — it determines the "
            "layout the downstream Linear layer sees. ViT papers usually write "
            "`(p1 p2 c)` so adjacent pixels are adjacent in the feature dim."
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
    """Doughty-style structured header for the exercise.

    The yaml block makes LO + Bloom + Keywords + KCs visible to the
    learner up-front (matches the published prompt template that beats
    human-authored MCQs on LO-alignment) and machine-readable for any
    downstream tooling that scrapes the notebook.
    """
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
    """Stub function + test cell. _dd_passed.add(<ex_id>) on success."""
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
    "**What you'll practice.** Five `einops.rearrange` patterns that ramp from identity → axis swap → composition → decomposition → patching. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "from einops import rearrange",
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
    "## einops.rearrange — quick refresher",
    "",
    "`rearrange(tensor, pattern, **axes_lengths)` does three things with one pattern:",
    "1. **Reorder axes** — `'h w -> w h'` is a transpose.",
    "2. **Compose axes** — `'h w c -> (h w) c'` flattens spatial dims into one.",
    "3. **Decompose axes** — `'(b1 b2) c -> b1 b2 c'` splits one axis into two (requires `b1=` or `b2=`).",
    "",
    "Identifiers on the right side must match identifiers on the left — every axis is named, every axis is accounted for. No transposing semantics beyond what the pattern says.",
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

# Public-facing metadata that downstream tools (orchestrator, scaffolding
# ladder generator, KC tracker) consume.
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
                # Per Lohr et al. ITiCSE 2025: 3-KC LLM-generated exercises
                # drop from ~94% to ~40% solvability. Flag ex5 explicitly
                # so the orchestrator can downweight its mastery signal
                # until empirical pass-rate data is available.
                "ex5": "3-KC integrative exercise — at Tutor-Kai solvability cliff. Track student pass rate; do not treat ex5 failure alone as evidence of atom non-mastery.",
            },
            "prompting_pattern": "Doughty et al. ACE 2024 (LO + Bloom + Keywords per exercise)",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Each cell needs `source` as list-of-lines for nbformat strict.
for c in nb["cells"]:
    if isinstance(c["source"], str):
        c["source"] = [line + "\n" for line in c["source"].split("\n")]
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")

# Stable per-cell ids so nbformat doesn't warn.
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
