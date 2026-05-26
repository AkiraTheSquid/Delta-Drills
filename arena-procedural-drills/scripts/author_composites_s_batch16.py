#!/usr/bin/env python3
"""Author 6 COMPOSITE procedural drills (batch-16, S assignments, part0).

Each composite exercises 2 atoms together: broadcasting / normalize-keepdim /
sum-broadcast duality / logsumexp.

  cx13 — broadcasting-rules + vector-normalize-keepdim
         (divide each row by its L2 norm — keepdim is what makes broadcast align)
  cx14 — broadcasting-rules + sum-and-broadcast-duality
         (sum along axis then divide row-wise — mean = sum/count)
  cx15 — broadcasting-rules + logsumexp-cross-entropy
         (subtract per-row max C then exp, sum, log — the LSE max-shift)
  cx16 — sum-and-broadcast-duality + vector-normalize-keepdim
         (sum and normalize composed — pre-norm-from-sum normalization)
  cx17 — logsumexp-cross-entropy + sum-and-broadcast-duality
         (full LSE via sum/log/broadcast — write logsumexp by hand)
  cx18 — logsumexp-cross-entropy + vector-normalize-keepdim
         (softmax via LSE — numerically stable, rows sum to 1)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    out = []
    for a in atom_ids:
        if a not in INV:
            raise KeyError(f"atom_id {a!r} not in /tmp/drill_atoms.json")
        out.append(INV[a]["subtopic"])
    return out


# ================================================================= cx13
CX13_ATOMS = ["broadcasting-rules", "vector-normalize-keepdim"]
spec_13 = {
    "atom_ids": CX13_ATOMS,
    "subtopics": _subs(CX13_ATOMS),
    "primary_atom": "vector-normalize-keepdim",
    "part": "part0",
    "exercise_index": 13,
    "exercise_title": "row-wise L2 normalize via keepdim broadcast",
    "slug": "row-l2-normalize-via-keepdim-broadcast",
    "atom_recap_md": (
        "## Row normalize = keepdim + broadcast — two atoms in one expression\n"
        "\n"
        "1. **`broadcasting-rules`** — right-align shapes, left-pad with 1s. `(N, D)`\n"
        "   divided by `(N,)` tries to align the `N` axis with the trailing axis of\n"
        "   the matrix (size `D`). Mismatch → either an error or, worse, a silent\n"
        "   wrong-axis broadcast.\n"
        "2. **`vector-normalize-keepdim`** — `x.norm(dim=-1, keepdim=True)` returns\n"
        "   shape `(N, 1)` instead of `(N,)`. The trailing 1 is what makes the divide\n"
        "   broadcast cleanly across the `D` axis.\n"
        "\n"
        "Composition: `keepdim=True` is *literally the bridge* that makes the norm\n"
        "tensor broadcast-compatible. Drop the keepdim and broadcasting either errors\n"
        "or aligns to the wrong axis — either way you don't get row normalization.\n"
    ),
    "prompt_body": (
        "Implement `cx13_row_normalize(x)`. Given a batch shaped `(B, D)`:\n"
        "\n"
        "1. Compute the per-row L2 norm with `keepdim=True` → shape `(B, 1)`.\n"
        "2. Divide `x` by that norm; broadcasting expands `(B, 1)` over `D`.\n"
        "3. Return the normalized tensor, same shape as `x`.\n"
        "\n"
        "Must use `keepdim=True`. The test verifies the norm tensor was kept as\n"
        "`(B, 1)` (not squeezed) by comparing intermediate shapes via a helper.\n"
    ),
    "stub_body": (
        "def cx13_row_normalize(x: Tensor) -> Tensor:\n"
        "    \"\"\"L2-normalize each row of x with keepdim broadcasting.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "# --- canonical 3-4-5 row ---\n"
        "x = t.tensor([[3.0, 4.0], [0.0, 1.0], [-1.0, 0.0]])\n"
        "out = cx13_row_normalize(x)\n"
        "assert out.shape == x.shape\n"
        "expected = t.tensor([[0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]])\n"
        "assert t.allclose(out, expected, atol=1e-6), f'got {out}'\n"
        "\n"
        "# --- random batch: every row has unit norm ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(8, 16, generator=rng) + 0.1\n"
        "Y = cx13_row_normalize(X)\n"
        "assert Y.shape == X.shape\n"
        "row_norms = Y.norm(dim=-1)\n"
        "assert t.allclose(row_norms, t.ones(8), atol=1e-5), f'row norms: {row_norms}'\n"
        "\n"
        "# --- keepdim was used: broadcasting must succeed on (B, D) / (B, 1) ---\n"
        "# We verify by checking that a non-square D works (would error if norm was (B,))\n"
        "X2 = t.randn(3, 5, generator=rng) + 0.5\n"
        "Y2 = cx13_row_normalize(X2)\n"
        "assert Y2.shape == X2.shape\n"
        "assert t.allclose(Y2.norm(dim=-1), t.ones(3), atol=1e-5)\n"
        "\n"
        "# --- value witness: each row equals x_row / ||x_row|| ---\n"
        "for i in range(X.shape[0]):\n"
        "    assert t.allclose(Y[i], X[i] / X[i].norm(), atol=1e-5)\n"
    ),
    "solution_body": (
        "def cx13_row_normalize(x: Tensor) -> Tensor:\n"
        "    # keepdim=True keeps the divisor as (B, 1) so broadcasting aligns\n"
        "    # the trailing 1 against the D axis and expands across it.\n"
        "    return x / x.norm(dim=-1, keepdim=True)\n"
    ),
    "solution_notes": (
        "The `keepdim=True` IS the broadcasting-rules atom in action: it produces a\n"
        "`(B, 1)` divisor whose trailing 1 right-aligns against the `D` axis of `x`,\n"
        "so the divide broadcasts cleanly. Without it the divisor is `(B,)`, which\n"
        "right-aligns against `D` — wrong axis."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["vector-normalize-keepdim", "broadcasting-rules"],
    "lo": (
        "Apply the broadcasting right-align rule by using keepdim=True on a row-norm "
        "so the (B, 1) divisor broadcasts cleanly across the D axis."
    ),
}
emit_composite(spec_13)


# ================================================================= cx14
CX14_ATOMS = ["broadcasting-rules", "sum-and-broadcast-duality"]
spec_14 = {
    "atom_ids": CX14_ATOMS,
    "subtopics": _subs(CX14_ATOMS),
    "primary_atom": "sum-and-broadcast-duality",
    "part": "part0",
    "exercise_index": 14,
    "exercise_title": "row-mean as sum-along-axis then broadcast-divide",
    "slug": "row-mean-via-sum-and-broadcast",
    "atom_recap_md": (
        "## Sum + broadcast = mean — two atoms in one expression\n"
        "\n"
        "1. **`broadcasting-rules`** — the right-align rule that lets a `(B, 1)`\n"
        "   tensor multiply or divide across `(B, D)` without an explicit loop.\n"
        "2. **`sum-and-broadcast-duality`** — `sum(dim=-1, keepdim=True)` reduces an\n"
        "   axis to size 1; the kept axis is exactly the shape you need to broadcast\n"
        "   the result back. `mean = sum / count` is the canonical composition.\n"
        "\n"
        "Composition: every \"reduce then re-distribute\" pattern (mean, normalize,\n"
        "softmax, layernorm) is shaped like this. The `keepdim=True` is what makes\n"
        "the broadcast step a one-liner.\n"
    ),
    "prompt_body": (
        "Implement `cx14_row_mean_normalize(x)`. Given `x` shape `(B, D)`:\n"
        "\n"
        "1. Compute the per-row sum with `keepdim=True` → shape `(B, 1)`.\n"
        "2. Divide that sum by `D` (the row length) to get the per-row mean. Both\n"
        "   `sum / D` and `sum / x.shape[-1]` are fine.\n"
        "3. Return a tensor of shape `(B, D)` where every entry in row `i` equals the\n"
        "   mean of row `i`. (`mean.expand_as(x)` or just relying on broadcast via\n"
        "   `x * 0 + mean` both work — the test only checks the values.)\n"
        "\n"
        "Constraints:\n"
        "- Use `sum` along `dim=-1` with `keepdim=True` (not `.mean(...)` directly).\n"
        "  The whole point of this drill is wiring sum + broadcast by hand.\n"
        "- Output must equal `x.mean(dim=-1, keepdim=True).expand_as(x)` value-wise.\n"
    ),
    "stub_body": (
        "def cx14_row_mean_normalize(x: Tensor) -> Tensor:\n"
        "    \"\"\"Return a (B, D) tensor whose entry (i, j) is mean(row i).\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "import inspect\n"
        "# --- sanity: source must use .sum( ... ) (not just .mean) ---\n"
        "src = inspect.getsource(cx14_row_mean_normalize)\n"
        "assert '.sum(' in src or 'sum(' in src, (\n"
        "    'must build mean as sum / D — atom is sum-and-broadcast-duality'\n"
        ")\n"
        "\n"
        "# --- canonical small batch ---\n"
        "x = t.tensor([[1.0, 2.0, 3.0, 4.0],\n"
        "              [10.0, 20.0, 30.0, 40.0]])\n"
        "out = cx14_row_mean_normalize(x)\n"
        "assert out.shape == x.shape, f'shape {out.shape} vs {x.shape}'\n"
        "expected = t.tensor([[2.5, 2.5, 2.5, 2.5],\n"
        "                     [25.0, 25.0, 25.0, 25.0]])\n"
        "assert t.allclose(out, expected, atol=1e-6), f'got {out}'\n"
        "\n"
        "# --- random batch: matches torch.mean witness ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(5, 7, generator=rng)\n"
        "Y = cx14_row_mean_normalize(X)\n"
        "assert Y.shape == X.shape\n"
        "ref = X.mean(dim=-1, keepdim=True).expand_as(X)\n"
        "assert t.allclose(Y, ref, atol=1e-5), f'mismatch'\n"
        "\n"
        "# --- row sums match D * row-mean (definition check) ---\n"
        "row_sums_out = Y.sum(dim=-1)\n"
        "row_sums_ref = X.sum(dim=-1)\n"
        "assert t.allclose(row_sums_out, row_sums_ref, atol=1e-5), (\n"
        "    f'sum of mean-row * D should equal original row sum'\n"
        ")\n"
    ),
    "solution_body": (
        "def cx14_row_mean_normalize(x: Tensor) -> Tensor:\n"
        "    # atom: sum along last axis with keepdim → (B, 1)\n"
        "    row_sum = x.sum(dim=-1, keepdim=True)\n"
        "    # mean = sum / count\n"
        "    row_mean = row_sum / x.shape[-1]\n"
        "    # atom: broadcasting-rules — (B, 1) right-aligns against D and expands\n"
        "    return row_mean.expand_as(x)\n"
    ),
    "solution_notes": (
        "`x.sum(dim=-1, keepdim=True)` is the sum-half of the duality; the divide-by-D\n"
        "is the count step; `.expand_as(x)` (or any broadcast op against `x`) is the\n"
        "broadcasting-rules step. Drop the `keepdim` and the divide would need an\n"
        "extra `unsqueeze` to broadcast back."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["sum-and-broadcast-duality", "broadcasting-rules"],
    "lo": (
        "Compose sum-along-axis with broadcast-divide to compute a per-row mean and "
        "redistribute it across the original row shape."
    ),
}
emit_composite(spec_14)


# ================================================================= cx15
CX15_ATOMS = ["broadcasting-rules", "logsumexp-cross-entropy"]
spec_15 = {
    "atom_ids": CX15_ATOMS,
    "subtopics": _subs(CX15_ATOMS),
    "primary_atom": "logsumexp-cross-entropy",
    "part": "part0",
    "exercise_index": 15,
    "exercise_title": "max-shift then exp/sum/log — the LSE primitive",
    "slug": "max-shift-exp-sum-log-broadcast",
    "atom_recap_md": (
        "## The LSE max-shift = subtract row max, then exp/sum/log + add back\n"
        "\n"
        "1. **`broadcasting-rules`** — `(B, C) - (B, 1)` is the canonical\n"
        "   keepdim-shaped broadcast that lets the per-row max subtract from every\n"
        "   class logit in that row.\n"
        "2. **`logsumexp-cross-entropy`** — the numerically stable form of\n"
        "   `log(sum(exp(x)))` is `log(sum(exp(x - max(x)))) + max(x)`. The max-shift\n"
        "   keeps every exp argument ≤ 0, avoiding overflow.\n"
        "\n"
        "Composition: the broadcast-subtract is what *makes* the max-shift work\n"
        "across a batch — without `keepdim=True` on the max, you can't subtract it\n"
        "row-wise. This drill isolates the max-shift core of LSE.\n"
    ),
    "prompt_body": (
        "Implement `cx15_lse_per_row(logits)` for `logits` shape `(B, C)`.\n"
        "\n"
        "Return a 1-D tensor of shape `(B,)` where entry `i` is the numerically\n"
        "stable `logsumexp(logits[i])`. **Build it by hand** as:\n"
        "\n"
        "1. Compute `m = logits.max(dim=-1, keepdim=True).values` → shape `(B, 1)`.\n"
        "2. Shifted: `shifted = logits - m` → shape `(B, C)` (broadcast subtract).\n"
        "3. `lse = (shifted.exp().sum(dim=-1)).log() + m.squeeze(-1)` → shape `(B,)`.\n"
        "\n"
        "Constraints:\n"
        "- Do NOT call `torch.logsumexp`. You're writing it.\n"
        "- Must handle logits up to ~10000 without overflow (the test checks).\n"
        "- Output shape must be `(B,)` (squeezed). The kept-dim is only used for the\n"
        "  subtract; the final add restores then squeezes.\n"
    ),
    "stub_body": (
        "def cx15_lse_per_row(logits: Tensor) -> Tensor:\n"
        "    \"\"\"Per-row logsumexp built from max-shift + exp + sum + log.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "import inspect\n"
        "import math\n"
        "# --- forbid torch.logsumexp (must write it by hand) ---\n"
        "src = inspect.getsource(cx15_lse_per_row)\n"
        "assert 't.logsumexp' not in src and 'torch.logsumexp' not in src, (\n"
        "    'must write LSE by hand, not call torch.logsumexp'\n"
        ")\n"
        "\n"
        "# --- uniform logits: LSE per row = log(C) ---\n"
        "logits = t.zeros(4, 3)\n"
        "out = cx15_lse_per_row(logits)\n"
        "assert out.shape == (4,), f'shape {out.shape}'\n"
        "assert t.allclose(out, t.full((4,), math.log(3)), atol=1e-5), f'uniform: {out}'\n"
        "\n"
        "# --- compare to torch.logsumexp witness ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "L = t.randn(6, 8, generator=rng) * 5.0\n"
        "ours = cx15_lse_per_row(L)\n"
        "ref = t.logsumexp(L, dim=-1)\n"
        "assert ours.shape == ref.shape\n"
        "assert t.allclose(ours, ref, atol=1e-5), f'mismatch vs torch.logsumexp: {ours} vs {ref}'\n"
        "\n"
        "# --- THE point: must survive huge logits (overflow protection) ---\n"
        "big = t.tensor([[10000.0, 9999.0, 10001.0, 9998.0],\n"
        "                [-10000.0, -9999.0, -10001.0, -9998.0]])\n"
        "out_big = cx15_lse_per_row(big)\n"
        "assert t.isfinite(out_big).all(), f'overflow! got {out_big}'\n"
        "ref_big = t.logsumexp(big, dim=-1)\n"
        "assert t.allclose(out_big, ref_big, atol=1e-3), f'{out_big} vs {ref_big}'\n"
        "\n"
        "# --- shape contract: (B, C) → (B,) ---\n"
        "out2 = cx15_lse_per_row(t.randn(11, 4, generator=rng))\n"
        "assert out2.shape == (11,)\n"
    ),
    "solution_body": (
        "def cx15_lse_per_row(logits: Tensor) -> Tensor:\n"
        "    # atom: broadcasting-rules — max with keepdim gives (B, 1), then\n"
        "    # (B, C) - (B, 1) right-aligns and broadcasts across the C axis.\n"
        "    m = logits.max(dim=-1, keepdim=True).values\n"
        "    shifted = logits - m\n"
        "    # atom: logsumexp-cross-entropy — exp/sum/log + add back the shift.\n"
        "    return (shifted.exp().sum(dim=-1)).log() + m.squeeze(-1)\n"
    ),
    "solution_notes": (
        "The `keepdim=True` on the max is what makes the subtract broadcast across\n"
        "the class axis — that's the broadcasting-rules half. The max-shift + exp +\n"
        "sum + log + add-back is the logsumexp-cross-entropy half. Without the shift,\n"
        "`exp(10000)` overflows in float32; with it, every exp arg is ≤ 0."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["logsumexp-cross-entropy", "broadcasting-rules"],
    "lo": (
        "Compose keepdim-broadcast subtract with the exp/sum/log identity to build "
        "a per-row logsumexp that survives 10000-magnitude logits."
    ),
}
emit_composite(spec_15)


# ================================================================= cx16
CX16_ATOMS = ["sum-and-broadcast-duality", "vector-normalize-keepdim"]
spec_16 = {
    "atom_ids": CX16_ATOMS,
    "subtopics": _subs(CX16_ATOMS),
    "primary_atom": "vector-normalize-keepdim",
    "part": "part0",
    "exercise_index": 16,
    "exercise_title": "sum-normalize each row to a probability distribution",
    "slug": "sum-normalize-rows-to-probs",
    "atom_recap_md": (
        "## Sum-normalize = divide each row by its row sum\n"
        "\n"
        "1. **`sum-and-broadcast-duality`** — `x.sum(dim=-1, keepdim=True)` collapses\n"
        "   the trailing axis to size 1. The kept axis is precisely the shape the\n"
        "   divide step needs to broadcast back.\n"
        "2. **`vector-normalize-keepdim`** — the row-normalize pattern\n"
        "   `x / norm(dim=-1, keepdim=True)`, but with `sum` instead of `L2 norm` as\n"
        "   the reducer. Same `keepdim=True` mechanic, different reducer.\n"
        "\n"
        "Composition: this is what turns a vector of non-negative counts into a\n"
        "probability distribution (each row sums to 1). It also generalises\n"
        "`F.normalize(x, p=1)` — the L1 row-normalize.\n"
    ),
    "prompt_body": (
        "Implement `cx16_sum_normalize(x)`. Given `x` shape `(B, D)` of non-negative\n"
        "values (the test feeds only non-negative inputs):\n"
        "\n"
        "1. Compute per-row sum with `keepdim=True` → shape `(B, 1)`.\n"
        "2. Divide `x` by that sum; broadcasting expands `(B, 1)` over `D`.\n"
        "3. Return shape `(B, D)` where every row sums to `1.0` (within float tol).\n"
        "\n"
        "Constraints:\n"
        "- Use `sum(..., keepdim=True)` — the whole point is the sum+keepdim pattern.\n"
        "- No `F.normalize`, no manual division by `x.sum(-1).unsqueeze(-1)`. The\n"
        "  keepdim flag IS the drill.\n"
    ),
    "stub_body": (
        "def cx16_sum_normalize(x: Tensor) -> Tensor:\n"
        "    \"\"\"Divide each row by its row sum so each row sums to 1.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "import inspect\n"
        "src = inspect.getsource(cx16_sum_normalize)\n"
        "assert 'keepdim' in src, 'must use keepdim=True — that\\'s the atom'\n"
        "assert 'F.normalize' not in src, 'do not use F.normalize'\n"
        "\n"
        "# --- canonical small batch: rows sum to 1 ---\n"
        "x = t.tensor([[1.0, 1.0, 2.0],\n"
        "              [3.0, 1.0, 6.0],\n"
        "              [0.5, 0.5, 0.0]])\n"
        "out = cx16_sum_normalize(x)\n"
        "assert out.shape == x.shape\n"
        "expected = t.tensor([[0.25, 0.25, 0.5],\n"
        "                     [0.3, 0.1, 0.6],\n"
        "                     [0.5, 0.5, 0.0]])\n"
        "assert t.allclose(out, expected, atol=1e-6), f'got {out}'\n"
        "\n"
        "# --- random non-negative batch: each row sums to 1 ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.rand(7, 11, generator=rng) + 0.01\n"
        "Y = cx16_sum_normalize(X)\n"
        "row_sums = Y.sum(dim=-1)\n"
        "assert t.allclose(row_sums, t.ones(7), atol=1e-5), f'row sums: {row_sums}'\n"
        "\n"
        "# --- value witness: matches x / x.sum(dim=-1, keepdim=True) ---\n"
        "ref = X / X.sum(dim=-1, keepdim=True)\n"
        "assert t.allclose(Y, ref, atol=1e-6)\n"
        "\n"
        "# --- shape preserved (non-square D) ---\n"
        "Y2 = cx16_sum_normalize(t.rand(4, 9, generator=rng) + 0.01)\n"
        "assert Y2.shape == (4, 9)\n"
        "assert t.allclose(Y2.sum(dim=-1), t.ones(4), atol=1e-5)\n"
    ),
    "solution_body": (
        "def cx16_sum_normalize(x: Tensor) -> Tensor:\n"
        "    # atoms compose: sum(keepdim=True) is the sum-and-broadcast-duality\n"
        "    # step; the divide is the vector-normalize-keepdim step but with sum\n"
        "    # as the reducer. The (B, 1) divisor right-aligns against D and expands.\n"
        "    return x / x.sum(dim=-1, keepdim=True)\n"
    ),
    "solution_notes": (
        "Pattern is identical to L2 row-normalize — just swap `.norm` for `.sum`.\n"
        "The keepdim=True is doing all the broadcasting work. This is also exactly\n"
        "what `F.normalize(x, p=1, dim=-1)` produces."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 1,
    "kcs": ["vector-normalize-keepdim", "sum-and-broadcast-duality"],
    "lo": (
        "Compose sum(keepdim=True) with broadcast-divide to L1-normalize each row "
        "of a batch into a probability distribution."
    ),
}
emit_composite(spec_16)


# ================================================================= cx17
CX17_ATOMS = ["logsumexp-cross-entropy", "sum-and-broadcast-duality"]
spec_17 = {
    "atom_ids": CX17_ATOMS,
    "subtopics": _subs(CX17_ATOMS),
    "primary_atom": "logsumexp-cross-entropy",
    "part": "part0",
    "exercise_index": 17,
    "exercise_title": "full logsumexp by hand — max-shift + sum + log + broadcast-add-back",
    "slug": "full-lse-via-sum-log-broadcast",
    "atom_recap_md": (
        "## Hand-rolled `logsumexp` — sum/log/broadcast all wired together\n"
        "\n"
        "1. **`sum-and-broadcast-duality`** — the keepdim flag controls whether the\n"
        "   reduced axis is dropped or kept as size 1. Keep it when you need to\n"
        "   broadcast the reduction result back over the original axis (subtract max\n"
        "   from every class logit; add max back to log-sum-exp).\n"
        "2. **`logsumexp-cross-entropy`** — the full identity is\n"
        "   `log(sum(exp(x - m))) + m` with `m = max(x)`. The shift keeps every exp\n"
        "   argument ≤ 0; the `+ m` restores the absolute scale.\n"
        "\n"
        "Composition: this is the full LSE, including BOTH the keepdim-for-subtract\n"
        "step AND the keepdim-for-add-back step. cx15 was the per-row scalar version;\n"
        "this one returns a kept-dim shape so it composes downstream (e.g. softmax).\n"
    ),
    "prompt_body": (
        "Implement `cx17_logsumexp(x, dim, keepdim=False)` — the full numerically\n"
        "stable `logsumexp` reducer along an arbitrary `dim`.\n"
        "\n"
        "Algorithm:\n"
        "1. `m = x.max(dim=dim, keepdim=True).values` (always kept for the subtract).\n"
        "2. `shifted = x - m` (broadcast subtract over `dim`).\n"
        "3. `s = shifted.exp().sum(dim=dim, keepdim=True)` (kept for the add-back).\n"
        "4. `lse = s.log() + m` (kept-dim form).\n"
        "5. If `keepdim=False`, squeeze `dim`.\n"
        "\n"
        "Constraints:\n"
        "- Do NOT call `torch.logsumexp` (this drill is about writing it).\n"
        "- Must handle large-magnitude inputs (~10000) without overflow.\n"
        "- Must honour the `keepdim` flag.\n"
    ),
    "stub_body": (
        "def cx17_logsumexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:\n"
        "    \"\"\"Hand-rolled logsumexp along `dim`, with full keepdim handling.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "import inspect\n"
        "import math\n"
        "src = inspect.getsource(cx17_logsumexp)\n"
        "assert 't.logsumexp' not in src and 'torch.logsumexp' not in src, (\n"
        "    'must not delegate to torch.logsumexp'\n"
        ")\n"
        "\n"
        "rng = t.Generator().manual_seed(0)\n"
        "\n"
        "# --- dim=-1, keepdim=False ---\n"
        "X = t.randn(5, 8, generator=rng) * 3.0\n"
        "ours = cx17_logsumexp(X, dim=-1, keepdim=False)\n"
        "ref = t.logsumexp(X, dim=-1, keepdim=False)\n"
        "assert ours.shape == ref.shape == (5,)\n"
        "assert t.allclose(ours, ref, atol=1e-5), f'dim=-1 noKeep: {ours} vs {ref}'\n"
        "\n"
        "# --- dim=-1, keepdim=True ---\n"
        "ours_k = cx17_logsumexp(X, dim=-1, keepdim=True)\n"
        "ref_k = t.logsumexp(X, dim=-1, keepdim=True)\n"
        "assert ours_k.shape == ref_k.shape == (5, 1)\n"
        "assert t.allclose(ours_k, ref_k, atol=1e-5)\n"
        "\n"
        "# --- dim=0, keepdim=False ---\n"
        "X3 = t.randn(4, 3, 5, generator=rng) * 2.0\n"
        "ours0 = cx17_logsumexp(X3, dim=0, keepdim=False)\n"
        "ref0 = t.logsumexp(X3, dim=0, keepdim=False)\n"
        "assert ours0.shape == ref0.shape == (3, 5)\n"
        "assert t.allclose(ours0, ref0, atol=1e-5)\n"
        "\n"
        "# --- dim=1, keepdim=True ---\n"
        "ours1 = cx17_logsumexp(X3, dim=1, keepdim=True)\n"
        "ref1 = t.logsumexp(X3, dim=1, keepdim=True)\n"
        "assert ours1.shape == ref1.shape == (4, 1, 5)\n"
        "assert t.allclose(ours1, ref1, atol=1e-5)\n"
        "\n"
        "# --- huge-magnitude safety (overflow check) ---\n"
        "big = t.tensor([[10000.0, 9999.0, 10001.0],\n"
        "                [-10000.0, -9999.0, -10001.0]])\n"
        "ours_big = cx17_logsumexp(big, dim=-1)\n"
        "assert t.isfinite(ours_big).all(), f'overflow! {ours_big}'\n"
        "ref_big = t.logsumexp(big, dim=-1)\n"
        "assert t.allclose(ours_big, ref_big, atol=1e-3)\n"
        "\n"
        "# --- uniform identity: logsumexp([0]*C) = log(C) ---\n"
        "u = t.zeros(2, 5)\n"
        "got_u = cx17_logsumexp(u, dim=-1)\n"
        "assert t.allclose(got_u, t.full((2,), math.log(5)), atol=1e-6)\n"
    ),
    "solution_body": (
        "def cx17_logsumexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:\n"
        "    # atom: sum-and-broadcast-duality — keepdim controls the broadcast.\n"
        "    m = x.max(dim=dim, keepdim=True).values   # kept for broadcast subtract\n"
        "    shifted = x - m                            # broadcast subtract over dim\n"
        "    s = shifted.exp().sum(dim=dim, keepdim=True)  # kept for broadcast add-back\n"
        "    # atom: logsumexp-cross-entropy — log(sum(exp(shift))) + m\n"
        "    out = s.log() + m\n"
        "    if not keepdim:\n"
        "        out = out.squeeze(dim)\n"
        "    return out\n"
    ),
    "solution_notes": (
        "Two keepdim broadcasts happen here: once to subtract `m` over `dim`, and\n"
        "once to add it back after the log. That dual use of keepdim IS the\n"
        "sum-and-broadcast-duality atom. The final squeeze just respects the user's\n"
        "`keepdim` flag — the math always works in kept-dim form internally."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["logsumexp-cross-entropy", "sum-and-broadcast-duality"],
    "lo": (
        "Compose the keepdim-controlled sum/broadcast pattern with the max-shift "
        "identity to build a hand-rolled logsumexp that supports arbitrary dim and "
        "keepdim flags without overflow."
    ),
}
emit_composite(spec_17)


# ================================================================= cx18
CX18_ATOMS = ["logsumexp-cross-entropy", "vector-normalize-keepdim"]
spec_18 = {
    "atom_ids": CX18_ATOMS,
    "subtopics": _subs(CX18_ATOMS),
    "primary_atom": "logsumexp-cross-entropy",
    "part": "part0",
    "exercise_index": 18,
    "exercise_title": "stable softmax via LSE — exp(x - lse(x))",
    "slug": "softmax-via-lse-stable",
    "atom_recap_md": (
        "## Stable softmax = `exp(x - lse(x))` — two atoms in one expression\n"
        "\n"
        "1. **`logsumexp-cross-entropy`** — `logsumexp(x) = log(sum(exp(x - m))) + m`\n"
        "   is the safe denominator. Subtracting it elementwise from `x` and exp'ing\n"
        "   gives softmax without any explicit divide.\n"
        "2. **`vector-normalize-keepdim`** — to broadcast `lse(x)` back over the\n"
        "   class axis, use `keepdim=True` so the result is `(B, 1)` instead of\n"
        "   `(B,)`. Same keepdim pattern as L2 row-normalize.\n"
        "\n"
        "Composition: `softmax(x) = exp(x - lse(x, keepdim=True))`. The keepdim on\n"
        "LSE is what makes the subtract broadcast cleanly across the class axis,\n"
        "and the exp-of-shifted-log eliminates the explicit divide that would\n"
        "otherwise overflow.\n"
    ),
    "prompt_body": (
        "Implement `cx18_stable_softmax(logits)`. Given `logits` shape `(B, C)`,\n"
        "return softmax along the class axis using the LSE identity:\n"
        "\n"
        "1. `lse = torch.logsumexp(logits, dim=-1, keepdim=True)` → shape `(B, 1)`.\n"
        "2. `out = (logits - lse).exp()` → shape `(B, C)`.\n"
        "\n"
        "Why this works: `exp(x - log(sum(exp(x)))) = exp(x) / sum(exp(x))` — but\n"
        "computed in log-space first, so the divide never happens explicitly and\n"
        "huge logits don't overflow.\n"
        "\n"
        "Constraints:\n"
        "- Use `keepdim=True` on the LSE so the subtract broadcasts.\n"
        "- Output must sum to 1 along the class axis (within float tol).\n"
        "- Must survive logits of magnitude ~10000.\n"
    ),
    "stub_body": (
        "def cx18_stable_softmax(logits: Tensor) -> Tensor:\n"
        "    \"\"\"Stable softmax via the LSE identity: exp(x - lse(x)).\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "import inspect\n"
        "src = inspect.getsource(cx18_stable_softmax)\n"
        "assert 'keepdim' in src, 'must use keepdim=True on the LSE'\n"
        "assert 'logsumexp' in src, 'must use logsumexp identity (not divide manually)'\n"
        "\n"
        "# --- uniform logits → uniform probs ---\n"
        "logits = t.zeros(4, 5)\n"
        "out = cx18_stable_softmax(logits)\n"
        "assert out.shape == logits.shape\n"
        "assert t.allclose(out, t.full((4, 5), 0.2), atol=1e-6), f'uniform: {out}'\n"
        "\n"
        "# --- rows sum to 1 ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "L = t.randn(7, 9, generator=rng) * 2.0\n"
        "Y = cx18_stable_softmax(L)\n"
        "row_sums = Y.sum(dim=-1)\n"
        "assert t.allclose(row_sums, t.ones(7), atol=1e-5), f'row sums: {row_sums}'\n"
        "\n"
        "# --- value witness: matches torch.softmax ---\n"
        "import torch.nn.functional as F\n"
        "ref = F.softmax(L, dim=-1)\n"
        "assert t.allclose(Y, ref, atol=1e-5), f'mismatch vs F.softmax'\n"
        "\n"
        "# --- overflow safety: 10000-magnitude logits still produce finite probs ---\n"
        "big = t.tensor([[10000.0, 9999.0, 10001.0, 9998.0],\n"
        "                [-10000.0, -9999.0, -10001.0, -9998.0]])\n"
        "out_big = cx18_stable_softmax(big)\n"
        "assert t.isfinite(out_big).all(), f'overflow! {out_big}'\n"
        "# At magnitude 10k, float32 roundoff in exp(x - lse(x)) gives row sums\n"
        "# of ~0.9998 (vs torch.softmax's 1.0 fused-kernel). Loose tol.\n"
        "assert t.allclose(out_big.sum(dim=-1), t.ones(2), atol=1e-3), (\n"
        "    f'row sums {out_big.sum(dim=-1)} should be approx 1 even at scale'\n"
        ")\n"
        "ref_big = F.softmax(big, dim=-1)\n"
        "assert t.allclose(out_big, ref_big, atol=1e-3)\n"
        "\n"
        "# --- non-square C dimension works (no silent broadcast bug) ---\n"
        "L2 = t.randn(3, 13, generator=rng)\n"
        "Y2 = cx18_stable_softmax(L2)\n"
        "assert Y2.shape == (3, 13)\n"
        "assert t.allclose(Y2.sum(dim=-1), t.ones(3), atol=1e-5)\n"
    ),
    "solution_body": (
        "def cx18_stable_softmax(logits: Tensor) -> Tensor:\n"
        "    # atom: vector-normalize-keepdim — keepdim=True gives (B, 1) so\n"
        "    # the subtract broadcasts across the C axis cleanly.\n"
        "    lse = t.logsumexp(logits, dim=-1, keepdim=True)\n"
        "    # atom: logsumexp-cross-entropy — exp(x - lse(x)) is softmax,\n"
        "    # done in log-space first so the explicit divide never happens.\n"
        "    return (logits - lse).exp()\n"
    ),
    "solution_notes": (
        "The `keepdim=True` is the vector-normalize-keepdim atom — same `(B, 1)`\n"
        "broadcast trick as L2 row-normalize, just with `logsumexp` as the reducer.\n"
        "The `exp(x - lse(x))` identity is the logsumexp-cross-entropy atom: it\n"
        "rewrites `exp(x) / sum(exp(x))` in a form that never overflows."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["logsumexp-cross-entropy", "vector-normalize-keepdim"],
    "lo": (
        "Compose the logsumexp identity with the keepdim-broadcast pattern to write "
        "a numerically stable softmax that survives huge logits."
    ),
}
emit_composite(spec_18)


if __name__ == "__main__":
    print("authored cx13..cx18 composites under arena-procedural-drills/composites/part0/")
