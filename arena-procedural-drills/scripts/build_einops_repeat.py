#!/usr/bin/env python3
"""Build the einops-repeat procedural drill notebook.

Generates `arena-procedural-drills/prereqs_einops/einops-repeat.ipynb` —
the third atom-keyed procedural drill (Phase 4.3). Mirrors the v0.2
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
OUT = REPO / "arena-procedural-drills/prereqs_einops/einops-repeat.ipynb"


# ----- build-time solution verification ---------------------------------

def verify_solutions(specs: list[dict]) -> None:
    try:
        import torch as t
        import numpy as np
        from torch import Tensor
        import torch.nn.functional as F
        import einops
        from einops import repeat
    except ImportError as e:
        raise SystemExit(
            f"[build verify] missing runtime dep: {e}\n"
            f"  pip install torch numpy einops  # required for build-time solution verification\n"
            f"  refusing to write notebook with unverified solutions."
        )

    base_ns = {
        "t": t, "np": np, "Tensor": Tensor, "F": F,
        "einops": einops, "repeat": repeat,
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


ATOM_ID = "einops-repeat"
SUBTOPIC = "Einops: Repeat"
TITLE = "einops.repeat — procedural drill"
TEMPLATE_VERSION = "v0.2"

# ----- knowledge-component decomposition --------------------------------

KC_DECOMPOSITION = [
    {
        "id": "repeat-add-axis",
        "kind": "component-skill",
        "description": "Introduce a brand-new named axis on the output side with a kwarg-bound size. The basic broadcast case.",
    },
    {
        "id": "repeat-stretch-via-composition",
        "kind": "component-skill",
        "description": "Use `(h r)` composition on the output side with `r=` to stretch an existing axis — every source row appears `r` times consecutively (nearest-neighbor stretch).",
    },
    {
        "id": "repeat-tile-via-composition",
        "kind": "component-skill",
        "description": "Use `(r h)` composition on the output side with `r=` to tile — the full original sequence appears `r` times. Mirror-image of the stretch pattern; order of factors inside the parentheses matters.",
    },
    {
        "id": "repeat-match-broadcast-shape",
        "kind": "component-skill",
        "description": "Use repeat to materialize a tensor at a downstream broadcast-target shape (e.g. add a trailing feature dim of size `d` to per-token weights).",
    },
    {
        "id": "repeat-nearest-upsample",
        "kind": "integrative-skill",
        "description": "Combine decomposition + stretch + composition to perform 2-D nearest-neighbor upsampling. The integrative KC — relies on fluency in the four component KCs above.",
    },
]

# ----- exercise specs ---------------------------------------------------

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "broadcast across batch",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["add-axis", "broadcast", "kwarg-binding"],
        "kcs": ["repeat-add-axis"],
        "lo": "Recall the `repeat(x, pattern, **size)` call shape for introducing a new named output axis.",
        "prompt_body": (
            "Implement `ex1_broadcast_batch(x, b)` to materialize a single `(c, h, w)` "
            "image into a batch of `b` identical copies.\n\n"
            "Input shape: `(c, h, w)`. Output shape: `(b, c, h, w)`.\n\n"
            "Use `einops.repeat` with the new axis bound by kwarg — not `x.unsqueeze(0)"
            ".expand(...)` or `torch.stack`. The point is to write the pattern."
        ),
        "stub": (
            "def ex1_broadcast_batch(x: Tensor, b: int) -> Tensor:\n"
            '    """Repeat `x` of shape (c, h, w) into shape (b, c, h, w)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(3 * 4 * 5).reshape(3, 4, 5).float()\n"
            "    y = ex1_broadcast_batch(x, b=2)\n"
            "    assert y.shape == (2, 3, 4, 5), f'expected (2,3,4,5), got {y.shape}'\n"
            "    assert t.equal(y[0], x), 'batch 0 does not match x'\n"
            "    assert t.equal(y[1], x), 'batch 1 does not match x'"
        ),
        "solution_body": (
            "def ex1_broadcast_batch(x: Tensor, b: int) -> Tensor:\n"
            "    return repeat(x, 'c h w -> b c h w', b=b)"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex2",
        "title": "per-token weight → per-feature weight",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["match-shape", "trailing-axis", "attention-mask"],
        "kcs": ["repeat-match-broadcast-shape"],
        "lo": "Apply repeat to add a trailing feature axis that lets a per-token weight broadcast against a per-token-per-feature tensor.",
        "prompt_body": (
            "Implement `ex2_per_token_to_per_feature(w, d)` so each per-token weight "
            "is materialized across `d` feature columns.\n\n"
            "Input shape: `(b, t)`. Output shape: `(b, t, d)`. Every "
            "`y[b, t, :]` should equal `w[b, t]`.\n\n"
            "This is the shape you need when you want to scale a feature tensor of "
            "shape `(b, t, d)` by a per-token weight."
        ),
        "stub": (
            "def ex2_per_token_to_per_feature(w: Tensor, d: int) -> Tensor:\n"
            '    """Repeat (b, t) → (b, t, d) by replicating w across the feature dim."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "w = t.tensor([[0.1, 0.5, 0.9, 0.0], [0.2, 0.3, 0.4, 0.5]])  # (b=2, t=4)\n"
            "    y = ex2_per_token_to_per_feature(w, d=3)\n"
            "    assert y.shape == (2, 4, 3), f'expected (2,4,3), got {y.shape}'\n"
            "    # Every feature column should match the source per-token weight.\n"
            "    for di in range(3):\n"
            "        assert t.allclose(y[:, :, di], w), f'feature column {di} does not match w'"
        ),
        "solution_body": (
            "def ex2_per_token_to_per_feature(w: Tensor, d: int) -> Tensor:\n"
            "    return repeat(w, 'b t -> b t d', d=d)"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex3",
        "title": "vertical stretch (row-stretch via composition)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["stretch", "composition", "factor-order"],
        "kcs": ["repeat-stretch-via-composition"],
        "lo": "Apply `(h r)` composition on the output side to stretch an axis — every source row appears `r` times consecutively (nearest-neighbor stretch).",
        "prompt_body": (
            "Implement `ex3_stretch_vertical(x, r)` to vertically stretch a 2-D image: "
            "each row of the input should appear `r` times consecutively in the output.\n\n"
            "Input shape: `(h, w)`. Output shape: `(h*r, w)`. Row pattern: "
            "`[x[0], x[0], ..., x[1], x[1], ..., x[2], ...]` with each input row "
            "appearing `r` times.\n\n"
            "Compose `(h r)` on the **output** side — order matters. The inner factor "
            "`r` varies fastest, so positions `0..r-1` come from source row 0."
        ),
        "stub": (
            "def ex3_stretch_vertical(x: Tensor, r: int) -> Tensor:\n"
            '    """Repeat (h, w) → (h*r, w), each source row appears r times in a block."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(3 * 4).reshape(3, 4).float()\n"
            "    y = ex3_stretch_vertical(x, r=2)\n"
            "    assert y.shape == (6, 4), f'expected (6,4), got {y.shape}'\n"
            "    # Source row i should appear at output rows i*r .. i*r+r-1.\n"
            "    assert t.equal(y[0], x[0]) and t.equal(y[1], x[0]), 'block 0 does not match x[0]'\n"
            "    assert t.equal(y[2], x[1]) and t.equal(y[3], x[1]), 'block 1 does not match x[1]'\n"
            "    assert t.equal(y[4], x[2]) and t.equal(y[5], x[2]), 'block 2 does not match x[2]'"
        ),
        "solution_body": (
            "def ex3_stretch_vertical(x: Tensor, r: int) -> Tensor:\n"
            "    return repeat(x, 'h w -> (h r) w', r=r)"
        ),
        "solution_notes": (
            "**Why `(h r)` and not `(r h)`?** `(h r)` says the output's leading axis "
            "is composed of `h` outer blocks of size `r` each — so source row 0 fills "
            "positions `0..r-1`, source row 1 fills positions `r..2r-1`, etc. This is "
            "the **stretch** pattern (nearest-neighbor upsample). Swap the factor order "
            "to get **tiling** instead — see Exercise 4."
        ),
    },
    {
        "id": "ex4",
        "title": "horizontal tile (sequence-tile via composition)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["tile", "composition", "factor-order"],
        "kcs": ["repeat-tile-via-composition"],
        "lo": "Apply `(r w)` composition on the output side to tile an axis — the full original sequence appears `r` times in a row.",
        "prompt_body": (
            "Implement `ex4_tile_horizontal(x, r)` to horizontally tile a 2-D image: "
            "each row of the output is `r` concatenated copies of the same input row.\n\n"
            "Input shape: `(h, w)`. Output shape: `(h, r*w)`. Each row of the output "
            "is `[x[i], x[i], ..., x[i]]` (r times concatenated).\n\n"
            "Compose `(r w)` on the **output** side — the inner factor `w` varies "
            "fastest, so positions `0..w-1` of every tile match the original row."
        ),
        "stub": (
            "def ex4_tile_horizontal(x: Tensor, r: int) -> Tensor:\n"
            '    """Repeat (h, w) → (h, r*w), each row is r copies of x[i] concatenated."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(3 * 4).reshape(3, 4).float()\n"
            "    y = ex4_tile_horizontal(x, r=2)\n"
            "    assert y.shape == (3, 8), f'expected (3,8), got {y.shape}'\n"
            "    # Each row of y should be [x[i], x[i]] concatenated.\n"
            "    for i in range(3):\n"
            "        assert t.equal(y[i, :4], x[i]), f'tile 0 of row {i} does not match x[{i}]'\n"
            "        assert t.equal(y[i, 4:], x[i]), f'tile 1 of row {i} does not match x[{i}]'"
        ),
        "solution_body": (
            "def ex4_tile_horizontal(x: Tensor, r: int) -> Tensor:\n"
            "    return repeat(x, 'h w -> h (r w)', r=r)"
        ),
        "solution_notes": (
            "**Stretch vs tile — factor order matters.**\n"
            "- `(r w)` → outer factor is `r`, so the full `w`-long row appears `r` "
            "times consecutively → **tiling**.\n"
            "- `(w r)` → outer factor is `w`, inner factor `r` → each source column "
            "appears `r` times before moving on → **stretching**.\n\n"
            "Same factor names, opposite layouts. Always think about which factor "
            "varies fastest in the output's flat memory layout."
        ),
    },
    {
        "id": "ex5",
        "title": "2×2 nearest-neighbor upsample (decompose + stretch + compose)",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["upsample", "nearest-neighbor", "integration", "multi-kc"],
        "kcs": [
            "repeat-add-axis",
            "repeat-stretch-via-composition",
            "repeat-nearest-upsample",
        ],
        "lo": "Synthesize new-axis introduction with axis composition to perform 2-D nearest-neighbor upsampling.",
        "prompt_body": (
            "Implement `ex5_upsample_2x2(x)` to nearest-neighbor upsample a batch of "
            "feature maps by 2× in both spatial dimensions.\n\n"
            "Input shape: `(b, c, h, w)`. Output shape: `(b, c, 2h, 2w)`. Each input "
            "pixel `x[..., i, j]` should appear as a `2×2` block at output positions "
            "`y[..., 2i:2i+2, 2j:2j+2]`.\n\n"
            "Introduce two new axes `p1, p2` of size 2, then compose them with the "
            "spatial axes so each pixel stretches into a 2×2 block.\n\n"
            "Equivalent to `torch.nn.functional.interpolate(x, scale_factor=2, "
            "mode='nearest')`.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3+ KCs in one pattern; "
            "empirical work (Lohr et al. ITiCSE 2025) shows 3-concept LLM-generated "
            "exercises drop from ~94% to ~40% solvability. Expect a step in difficulty "
            "here vs Exercises 1-4."
        ),
        "stub": (
            "def ex5_upsample_2x2(x: Tensor) -> Tensor:\n"
            '    """Nearest-neighbor 2x2 upsample. (b, c, h, w) → (b, c, 2h, 2w).\n'
            "\n"
            "    Each input pixel becomes a 2x2 block of identical values.\n"
            '    """\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.arange(1 * 1 * 2 * 2).reshape(1, 1, 2, 2).float()\n"
            "    y = ex5_upsample_2x2(x)\n"
            "    assert y.shape == (1, 1, 4, 4), f'expected (1,1,4,4), got {y.shape}'\n"
            "    expected = F.interpolate(x, scale_factor=2, mode='nearest')\n"
            "    assert t.equal(y, expected), 'values differ from F.interpolate(scale=2, mode=nearest)'\n"
            "\n"
            "    # Also test a larger random tensor.\n"
            "    x2 = t.randn(2, 3, 5, 7)\n"
            "    y2 = ex5_upsample_2x2(x2)\n"
            "    assert y2.shape == (2, 3, 10, 14), f'expected (2,3,10,14), got {y2.shape}'\n"
            "    assert t.allclose(y2, F.interpolate(x2, scale_factor=2, mode='nearest')), 'random-input mismatch'"
        ),
        "solution_body": (
            "def ex5_upsample_2x2(x: Tensor) -> Tensor:\n"
            "    return repeat(\n"
            "        x,\n"
            "        'b c h w -> b c (h p1) (w p2)',\n"
            "        p1=2, p2=2,\n"
            "    )"
        ),
        "solution_notes": (
            "**Reading the pattern.**\n"
            "- `(h p1)` on the output side: source row `i` fills output rows "
            "`i*p1 .. i*p1+p1-1`. Same shape recipe as Exercise 3, but applied to "
            "an axis already present in the input.\n"
            "- `(w p2)` does the same horizontally.\n"
            "- `b` and `c` pass through untouched.\n\n"
            "**Stretch vs tile here.** `(h p1)` stretches — every source pixel fills a "
            "contiguous 2×2 patch. If you wrote `(p1 h)` instead you'd get the entire "
            "row tiled twice vertically, which is **not** nearest-neighbor upsampling."
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
    "**What you'll practice.** Five `einops.repeat` patterns that ramp from new-axis broadcast → per-token-to-per-feature → vertical stretch → horizontal tile → 2×2 nearest-neighbor upsample. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "from einops import repeat",
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
    "## einops.repeat — quick refresher",
    "",
    "`repeat(tensor, pattern, **axes_lengths)` introduces new axes or stretches existing ones:",
    "1. **New axis** — `'c h w -> b c h w'` with `b=4` broadcasts across a new batch dim.",
    "2. **Trailing axis** — `'b t -> b t d'` with `d=64` materializes a per-token weight at per-feature width.",
    "3. **Stretch (nearest-neighbor)** — `'h w -> (h r) w'` with `r=2` makes each row appear twice in a block.",
    "4. **Tile** — `'h w -> h (r w)'` with `r=2` concatenates two copies of every row.",
    "",
    "Difference between **stretch** `(h r)` and **tile** `(r h)`: the factor written first varies slower. `(h r)` puts source row 0 at positions `0..r-1`; `(r h)` puts source row 0 at positions `0, h, 2h, ...`.",
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
