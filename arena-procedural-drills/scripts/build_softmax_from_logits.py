#!/usr/bin/env python3
"""Build the softmax-from-logits procedural drill notebook.

Generates `arena-procedural-drills/prereqs_numpy/softmax-from-logits.ipynb`
(Phase 4.6). Mirrors the v0.2 template. Inherits the verify_solutions gate.

Atom `softmax-from-logits` is the foundational numerical-stability drill —
you can't build classification or attention without getting this right.
Bridges to the bank topic `Numpy` via the explicit token rule in
atom_readiness.js.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "arena-procedural-drills/prereqs_numpy/softmax-from-logits.ipynb"


def verify_solutions(specs: list[dict]) -> None:
    try:
        import torch as t
        import numpy as np
        from torch import Tensor
        import torch.nn.functional as F
    except ImportError as e:
        raise SystemExit(
            f"[build verify] missing runtime dep: {e}\n"
            f"  pip install torch numpy\n"
            f"  refusing to write notebook with unverified solutions."
        )

    base_ns = {"t": t, "np": np, "Tensor": Tensor, "F": F}
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


ATOM_ID = "softmax-from-logits"
SUBTOPIC = "Numpy: Applied patterns and advanced"
TITLE = "softmax from logits — procedural drill"
TEMPLATE_VERSION = "v0.2"

KC_DECOMPOSITION = [
    {
        "id": "softmax-naive-formula",
        "kind": "component-skill",
        "description": "Compute softmax via the literal definition `exp(x) / exp(x).sum()`. Verifies the elementwise exp and the scalar division.",
    },
    {
        "id": "softmax-subtract-max",
        "kind": "component-skill",
        "description": "Subtract the per-batch max from logits BEFORE applying exp, so the largest exponent is 0 (no overflow). The standard numerical-stability trick.",
    },
    {
        "id": "softmax-axis-aware",
        "kind": "component-skill",
        "description": "Compute softmax along a specific axis of a batched logits tensor using `keepdim=True` on the max and the sum so they broadcast back.",
    },
    {
        "id": "softmax-logsumexp",
        "kind": "component-skill",
        "description": "Compute log-softmax via `x - logsumexp(x)` without ever materializing exp/sum. Required for cross-entropy of large vocabularies where numerical precision matters.",
    },
    {
        "id": "softmax-cross-entropy-stable",
        "kind": "integrative-skill",
        "description": "Combine subtract-max + log-sum-exp + class-index lookup to produce numerically-stable per-sample cross-entropy directly from logits. The integrative KC — relies on all four component KCs.",
    },
]

EXERCISE_SPECS = [
    {
        "id": "ex1",
        "title": "naive softmax (1-D logits)",
        "bloom_level": "Remember",
        "difficulty_dots": "⚪⚪⚪⚪⚪",
        "difficulty_num": 1,
        "keywords": ["softmax", "exp", "naive-formula"],
        "kcs": ["softmax-naive-formula"],
        "lo": "Recall the softmax formula `exp(x) / sum(exp(x))` and apply it to a 1-D tensor.",
        "prompt_body": (
            "Implement `ex1_softmax_naive(logits)` using the literal softmax formula:\n\n"
            "`softmax(x)_i = exp(x_i) / sum_j exp(x_j)`.\n\n"
            "Input: 1-D tensor of small values (so overflow doesn't matter yet). "
            "Output: same shape, every entry in [0, 1], sums to 1.\n\n"
            "Don't use `torch.softmax` — write the formula directly. We'll deal with "
            "overflow in the next exercise."
        ),
        "stub": (
            "def ex1_softmax_naive(logits: Tensor) -> Tensor:\n"
            '    """Naive 1-D softmax. exp(x) / sum(exp(x))."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "logits = t.tensor([1.0, 2.0, 3.0, -1.0])\n"
            "    p = ex1_softmax_naive(logits)\n"
            "    assert p.shape == logits.shape, f'shape mismatch: {p.shape}'\n"
            "    assert (p >= 0).all() and (p <= 1).all(), 'probabilities must lie in [0, 1]'\n"
            "    assert t.allclose(p.sum(), t.tensor(1.0)), f'probabilities should sum to 1, got {p.sum().item()}'\n"
            "    assert t.allclose(p, t.softmax(logits, dim=0)), 'values differ from t.softmax'"
        ),
        "solution_body": (
            "def ex1_softmax_naive(logits: Tensor) -> Tensor:\n"
            "    e = logits.exp()\n"
            "    return e / e.sum()"
        ),
        "solution_notes": "",
    },
    {
        "id": "ex2",
        "title": "stable softmax (subtract-max trick)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴⚪⚪⚪⚪",
        "difficulty_num": 2,
        "keywords": ["stable-softmax", "subtract-max", "overflow-guard"],
        "kcs": ["softmax-subtract-max"],
        "lo": "Apply the subtract-max trick to make softmax numerically stable for large logits.",
        "prompt_body": (
            "Implement `ex2_softmax_stable(logits)` so that even logits of magnitude "
            "1000 don't produce `inf` or `nan`.\n\n"
            "Subtract the maximum logit before exponentiating. Mathematically a no-op "
            "(softmax is shift-invariant), but it keeps every exponent ≤ 0, so `exp` "
            "stays in `(0, 1]` instead of overflowing.\n\n"
            "Input: 1-D tensor. Output: same shape, sums to 1, no `inf`/`nan` even "
            "for extreme inputs."
        ),
        "stub": (
            "def ex2_softmax_stable(logits: Tensor) -> Tensor:\n"
            '    """Stable 1-D softmax. exp(x - max(x)) / sum(exp(x - max(x)))."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Normal case still correct\n"
            "    logits = t.tensor([1.0, 2.0, 3.0, -1.0])\n"
            "    p = ex2_softmax_stable(logits)\n"
            "    assert t.allclose(p, t.softmax(logits, dim=0)), 'normal-case mismatch'\n"
            "\n"
            "    # Overflow stress test — naive softmax would NaN here.\n"
            "    huge = t.tensor([1000.0, 1001.0, 999.0])\n"
            "    ph = ex2_softmax_stable(huge)\n"
            "    assert not t.isnan(ph).any() and not t.isinf(ph).any(), 'output contains NaN/Inf — subtract-max not applied'\n"
            "    assert t.allclose(ph.sum(), t.tensor(1.0)), f'sum should be 1, got {ph.sum().item()}'\n"
            "    assert t.allclose(ph, t.softmax(huge, dim=0)), 'overflow-case mismatch'"
        ),
        "solution_body": (
            "def ex2_softmax_stable(logits: Tensor) -> Tensor:\n"
            "    shifted = logits - logits.max()\n"
            "    e = shifted.exp()\n"
            "    return e / e.sum()"
        ),
        "solution_notes": (
            "**Why this works.** `softmax(x + c) = softmax(x)` for any constant `c`. "
            "Setting `c = -max(x)` shifts the largest input to 0 — its `exp` becomes "
            "1 (the maximum possible non-overflowing value), and every other `exp` is "
            "in `(0, 1)`. No overflow, no precision loss in the dominant term."
        ),
    },
    {
        "id": "ex3",
        "title": "row-wise softmax (axis-aware with keepdim)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "difficulty_num": 3,
        "keywords": ["axis-aware", "keepdim", "batched-softmax"],
        "kcs": ["softmax-axis-aware"],
        "lo": "Apply softmax to each row of a 2-D logits tensor using `keepdim=True` for the max and the sum.",
        "prompt_body": (
            "Implement `ex3_softmax_rows(logits)` to apply softmax independently to "
            "every row of a 2-D tensor.\n\n"
            "Input shape: `(N, C)` — batch of `N` rows, each over `C` classes. "
            "Output shape: `(N, C)`. Every row of the output should be a probability "
            "distribution (sums to 1).\n\n"
            "Use the subtract-max + keepdim pattern. `max(dim=1, keepdim=True)` "
            "returns a NamedTuple `(values, indices)` — you want `.values`. Same for "
            "the sum: keep its axis with `keepdim=True` so it broadcasts back."
        ),
        "stub": (
            "def ex3_softmax_rows(logits: Tensor) -> Tensor:\n"
            '    """Row-wise stable softmax. (N, C) → (N, C), each row sums to 1."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "logits = t.tensor([\n"
            "        [1.0, 2.0, 3.0],\n"
            "        [-2.0, 0.0, 1.0],\n"
            "        [1000.0, 999.0, 1001.0],   # overflow stress\n"
            "    ])\n"
            "    p = ex3_softmax_rows(logits)\n"
            "    assert p.shape == logits.shape, f'shape mismatch: {p.shape}'\n"
            "    row_sums = p.sum(dim=1)\n"
            "    assert t.allclose(row_sums, t.ones(3)), f'each row should sum to 1, got {row_sums.tolist()}'\n"
            "    assert not t.isnan(p).any() and not t.isinf(p).any(), 'NaN/Inf in output — subtract-max missing'\n"
            "    assert t.allclose(p, t.softmax(logits, dim=1)), 'values differ from t.softmax(logits, dim=1)'"
        ),
        "solution_body": (
            "def ex3_softmax_rows(logits: Tensor) -> Tensor:\n"
            "    row_max = logits.max(dim=1, keepdim=True).values\n"
            "    e = (logits - row_max).exp()\n"
            "    return e / e.sum(dim=1, keepdim=True)"
        ),
        "solution_notes": (
            "**`keepdim=True` is critical twice here.** Without it on `max`, you'd "
            "get a 1-D row of per-batch maxes that wouldn't broadcast against the "
            "2-D logits. Same for the sum in the divisor. With `keepdim=True` both "
            "stay 2-D `(N, 1)` and broadcast cleanly against `(N, C)`."
        ),
    },
    {
        "id": "ex4",
        "title": "log-softmax via logsumexp (no exp materialized)",
        "bloom_level": "Apply",
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "difficulty_num": 3,
        "keywords": ["log-softmax", "logsumexp", "cross-entropy-prep"],
        "kcs": ["softmax-logsumexp"],
        "lo": "Apply the identity `log_softmax(x) = x - logsumexp(x)` to compute log-probabilities directly.",
        "prompt_body": (
            "Implement `ex4_log_softmax_rows(logits)` to return `log(softmax(logits))` "
            "row-wise.\n\n"
            "Input shape: `(N, C)`. Output shape: `(N, C)`. Every row's `exp` of the "
            "output should sum to 1 (i.e. they're valid log-probabilities).\n\n"
            "Don't compute softmax and then take the log — that loses precision for "
            "near-zero probabilities. Use the identity:\n\n"
            "`log_softmax(x) = x - logsumexp(x)`\n\n"
            "where `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`. PyTorch ships "
            "`torch.logsumexp(x, dim=..., keepdim=True)` — use it.\n\n"
            "Equivalent to `torch.log_softmax(logits, dim=1)`."
        ),
        "stub": (
            "def ex4_log_softmax_rows(logits: Tensor) -> Tensor:\n"
            '    """Log-softmax along dim=1. (N, C) → (N, C)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "logits = t.tensor([\n"
            "        [1.0, 2.0, 3.0, -1.0],\n"
            "        [1000.0, 1001.0, 1002.0, 999.0],\n"
            "    ])\n"
            "    lp = ex4_log_softmax_rows(logits)\n"
            "    assert lp.shape == logits.shape, f'shape mismatch: {lp.shape}'\n"
            "    # exp of log-probs should sum to 1 per row.\n"
            "    assert t.allclose(lp.exp().sum(dim=1), t.ones(2), atol=1e-5), 'rows of exp(log_softmax) should sum to 1'\n"
            "    assert not t.isnan(lp).any() and not t.isinf(lp).any(), 'NaN/Inf in output'\n"
            "    assert t.allclose(lp, t.log_softmax(logits, dim=1), atol=1e-5), 'values differ from t.log_softmax'"
        ),
        "solution_body": (
            "def ex4_log_softmax_rows(logits: Tensor) -> Tensor:\n"
            "    return logits - t.logsumexp(logits, dim=1, keepdim=True)"
        ),
        "solution_notes": (
            "**Why not `softmax(x).log()`?** When the true probability of class i is "
            "1e-30, `softmax(x)[i]` underflows to 0 and `log(0) = -inf`. Computing "
            "`log_softmax` directly avoids the underflow because the subtraction "
            "stays in the log domain.\n\n"
            "**Why `logsumexp` is stable.** It's secretly the subtract-max trick "
            "again: `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`. The exp "
            "inside is on shifted-down values so it never overflows."
        ),
    },
    {
        "id": "ex5",
        "title": "stable per-sample cross-entropy from logits",
        "bloom_level": "Create",
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "difficulty_num": 4,
        "keywords": ["cross-entropy", "logsumexp", "integration", "multi-kc"],
        "kcs": [
            "softmax-subtract-max",
            "softmax-logsumexp",
            "softmax-cross-entropy-stable",
        ],
        "lo": "Synthesize stable log-softmax + class-index lookup to compute per-sample cross-entropy directly from raw logits.",
        "prompt_body": (
            "Implement `ex5_cross_entropy_per_sample(logits, targets)` to compute "
            "per-sample cross-entropy from raw logits and integer class targets.\n\n"
            "Input shapes: `logits` is `(N, C)`, `targets` is `(N,)` of class "
            "indices in `[0, C)`. Output shape: `(N,)` — one loss per sample.\n\n"
            "The math: `loss_n = -log_softmax(logits_n)[targets_n]`. Use the "
            "logsumexp identity from Exercise 4 — never materialize `exp` or "
            "intermediate probabilities.\n\n"
            "Two well-known ways to do the class-index lookup:\n"
            "- `logits.gather(1, targets.unsqueeze(1)).squeeze(1)`\n"
            "- `logits[torch.arange(N), targets]` (integer-array indexing)\n\n"
            "Equivalent to `torch.nn.functional.cross_entropy(logits, targets, "
            "reduction='none')`.\n\n"
            "> ⚠️ **Integrative exercise.** This combines 3+ KCs (subtract-max, "
            "logsumexp, integer indexing) in one expression; empirical work (Lohr et "
            "al. ITiCSE 2025) shows 3-concept LLM-generated exercises drop from "
            "~94% to ~40% solvability. Expect a step in difficulty here vs "
            "Exercises 1-4."
        ),
        "stub": (
            "def ex5_cross_entropy_per_sample(logits: Tensor, targets: Tensor) -> Tensor:\n"
            '    """Per-sample cross-entropy from raw logits. (N, C), (N,) → (N,)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "logits = t.tensor([\n"
            "        [1.0, 2.0, 3.0, -1.0],\n"
            "        [-2.0, 0.0, 1.0, 0.5],\n"
            "        [1000.0, 1001.0, 999.0, 998.0],   # overflow stress\n"
            "    ])\n"
            "    targets = t.tensor([2, 1, 0])\n"
            "    loss = ex5_cross_entropy_per_sample(logits, targets)\n"
            "    assert loss.shape == (3,), f'expected (3,), got {tuple(loss.shape)}'\n"
            "    assert not t.isnan(loss).any() and not t.isinf(loss).any(), 'NaN/Inf in output'\n"
            "    expected = F.cross_entropy(logits, targets, reduction='none')\n"
            "    assert t.allclose(loss, expected, atol=1e-5), f'values differ from F.cross_entropy:\\n  got      {loss}\\n  expected {expected}'"
        ),
        "solution_body": (
            "def ex5_cross_entropy_per_sample(logits: Tensor, targets: Tensor) -> Tensor:\n"
            "    log_probs = logits - t.logsumexp(logits, dim=1, keepdim=True)\n"
            "    N = logits.shape[0]\n"
            "    return -log_probs[t.arange(N), targets]"
        ),
        "solution_notes": (
            "**Reading the pattern.**\n"
            "- `logits - logsumexp(logits, dim=1, keepdim=True)` → log-probabilities, "
            "stable for any logit magnitude.\n"
            "- `log_probs[arange(N), targets]` → integer-array indexing picks "
            "`log_probs[n, targets[n]]` for every `n`.\n"
            "- Negate → cross-entropy (we minimize `-log p(correct class)`).\n\n"
            "**Equivalent gather form:** "
            "`-log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)`. Functionally "
            "identical; `gather` is preferred when `targets` has more dims (e.g. "
            "language-modeling next-token prediction with shape `(B, T)`)."
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
    "**What you'll practice.** Five softmax patterns that ramp from naive 1-D softmax → subtract-max stable softmax → row-wise softmax → log-softmax via logsumexp → stable per-sample cross-entropy. Read the docstring, fill the function body, run the test cell. The solution sits in the collapsed `<details>` block below each exercise.",
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
    "## Softmax — quick refresher",
    "",
    "**Naive:** `softmax(x)_i = exp(x_i) / sum_j exp(x_j)`. Overflows whenever any `x_i` is large.",
    "",
    "**Stable:** `softmax(x) = softmax(x - max(x))`. Same values, but every exponent is ≤ 0 so `exp` stays in `(0, 1]`.",
    "",
    "**Log-softmax:** `log_softmax(x) = x - logsumexp(x)`. Avoids materializing `exp(x)` so it stays precise for tiny probabilities.",
    "",
    "**Cross-entropy from logits:** `loss_n = -log_softmax(logits_n)[target_n]`. The canonical classification objective — never go through `softmax` first.",
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
