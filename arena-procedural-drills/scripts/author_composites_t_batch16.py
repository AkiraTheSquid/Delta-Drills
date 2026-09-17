"""Composite drills cx19..cx24 — batch-16 (T-cell, part0: indexing/gather).

Six composite procedural drills exercising 2-atom pairs from the indexing /
view / einops machinery (ARENA part 0). Each composite forces the learner to
apply both atoms together in ONE function.

cx19  arange-fancy-index-cross-entropy + index-by-tensor
cx20  index-by-tensor + slice-view-mutation
cx21  index-by-tensor + tensor-item-scalar
cx22  slice-view-mutation + tensor-wraps-ndarray
cx23  arange-fancy-index-cross-entropy + einops-rearrange
cx24  logsumexp-cross-entropy + einops-rearrange
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# ===========================================================================
# cx19 — per-sample target logits via logits[arange(B), target]
# ===========================================================================
spec_19 = {
    "atom_ids": ["arange-fancy-index-cross-entropy", "index-by-tensor"],
    "subtopics": _subs(["arange-fancy-index-cross-entropy", "index-by-tensor"]),
    "primary_atom": "arange-fancy-index-cross-entropy",
    "part": "part0",
    "exercise_index": 19,
    "exercise_title": "pick per-sample target logits via logits[arange(B), target]",
    "slug": "arange-fancy-index-per-sample-target",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`index-by-tensor` is the general atom: you index a tensor with another tensor and "
        "PyTorch returns the gathered values shaped by the index. `arange(B)` paired with a "
        "`target` of shape `(B,)` is the specific cross-entropy use case — `logits[arange(B), "
        "target]` returns a 1-D `(B,)` slice picking ONE column per row.\n\n"
        "The composition exercises both atoms together: arange-fancy-index is the IDIOM, "
        "index-by-tensor is the MECHANISM. Get the mechanism wrong (e.g. `logits[:, target]`) "
        "and you get a (B, B) broadcasting result instead of the per-sample (B,) vector you "
        "wanted."
    ),
    "prompt_body": (
        "Implement `cx19_pick_target_logits(logits, target)` that:\n\n"
        "- Takes `logits` of shape `(B, C)` and `target` of shape `(B,)` (dtype long).\n"
        "- Returns the per-sample target logit of shape `(B,)` via `logits[arange(B), target]`.\n\n"
        "The output must NOT be `(B, B)` — that would mean you wrote `logits[:, target]` "
        "(broadcasting) instead of the arange-paired form."
    ),
    "stub_body": (
        "def cx19_pick_target_logits(logits, target):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Trivial case: B=3, target picks one column per row.\n"
        "logits = t.tensor([\n"
        "    [10.0, 20.0, 30.0],\n"
        "    [40.0, 50.0, 60.0],\n"
        "    [70.0, 80.0, 90.0],\n"
        "])\n"
        "target = t.tensor([0, 1, 2])\n"
        "got = cx19_pick_target_logits(logits, target)\n"
        "assert got.shape == (3,), f'shape: {got.shape}'\n"
        "assert t.allclose(got, t.tensor([10.0, 50.0, 90.0])), f'value: {got}'\n"
        "\n"
        "# Constant-column target → must NOT broadcast.\n"
        "target_const = t.tensor([1, 1, 1])\n"
        "got_const = cx19_pick_target_logits(logits, target_const)\n"
        "assert got_const.shape == (3,), (\n"
        "    f'output must be (3,) not (3,3); did you write logits[:, target]? Got {got_const.shape}'\n"
        ")\n"
        "assert t.allclose(got_const, t.tensor([20.0, 50.0, 80.0])), got_const\n"
        "\n"
        "# Compare against the BROADCASTING wrong answer — they must differ in shape.\n"
        "broadcast_wrong = logits[:, target_const]\n"
        "assert broadcast_wrong.shape == (3, 3), 'sanity: logits[:, target] does broadcast'\n"
        "assert got_const.shape != broadcast_wrong.shape\n"
        "\n"
        "# Larger random batch — witness via slow Python loop.\n"
        "rng = t.Generator().manual_seed(0)\n"
        "big_logits = t.randn(32, 10, generator=rng)\n"
        "big_target = t.randint(0, 10, (32,), generator=rng)\n"
        "got_big = cx19_pick_target_logits(big_logits, big_target)\n"
        "assert got_big.shape == (32,)\n"
        "expected = t.tensor([big_logits[i, big_target[i].item()].item() for i in range(32)])\n"
        "assert t.allclose(got_big, expected), 'value mismatch on (32,10)'\n"
        "\n"
        "# Composes cleanly into CE: lse - picked = per-sample cross-entropy.\n"
        "lse = t.logsumexp(logits, dim=-1)\n"
        "picked = cx19_pick_target_logits(logits, target)\n"
        "per_sample_ce = lse - picked\n"
        "assert per_sample_ce.shape == (3,)"
    ),
    "solution_body": (
        "def cx19_pick_target_logits(logits, target):\n"
        "    # arange-fancy-index idiom: index dim-0 by arange(B), dim-1 by target.\n"
        "    # Both are 1-D tensors of length B → output is 1-D length B (NOT broadcast).\n"
        "    B = logits.shape[0]\n"
        "    return logits[t.arange(B), target]"
    ),
    "solution_notes": (
        "The arange-pairing is what suppresses broadcasting. `logits[:, target]` would broadcast "
        "the (B,)-shaped target across the (B,) row axis, producing (B, B). Pairing target with "
        "`arange(B)` (also length B) tells PyTorch you want a 1-D gather, not a 2-D cross-product."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["arange-fancy-index-cross-entropy", "index-by-tensor"],
    "lo": (
        "Use the arange-fancy-index idiom to gather per-sample target logits from a (B,C) "
        "tensor in shape (B,), without triggering broadcasting."
    ),
}


# ===========================================================================
# cx20 — index-by-tensor selects rows; mutate them in place via slice-view
# ===========================================================================
spec_20 = {
    "atom_ids": ["index-by-tensor", "slice-view-mutation"],
    "subtopics": _subs(["index-by-tensor", "slice-view-mutation"]),
    "primary_atom": "index-by-tensor",
    "part": "part0",
    "exercise_index": 20,
    "exercise_title": "mutate selected rows in place via index-by-tensor write",
    "slug": "index-by-tensor-row-mutation",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`index-by-tensor` is normally a READ — `mat[idx]` returns gathered values. But PyTorch "
        "lets you also WRITE through that same expression: `mat[idx] = value` mutates the "
        "selected rows in place, sharing storage with the original tensor. That's the same "
        "underlying mechanic as slice-view-mutation (assigning to a view modifies the source) — "
        "just with a tensor index instead of a slice.\n\n"
        "Composing these two atoms: select rows by an index tensor, write to them, and verify "
        "(a) the storage pointer didn't move, (b) the selected rows are now zero, and (c) the "
        "non-selected rows are untouched."
    ),
    "prompt_body": (
        "Implement `cx20_zero_rows_inplace(mat, idx)` that:\n\n"
        "- Takes `mat` of shape `(N, D)` and `idx` of shape `(K,)` long (the row indices to zero).\n"
        "- Zeros the selected rows IN PLACE via index-by-tensor write: `mat[idx] = 0.0` (or "
        "equivalent slice-view assignment).\n"
        "- Returns `mat` (the same tensor object — must not reallocate).\n\n"
        "The unchanged rows must be bit-identical, and `mat.data_ptr()` must not move."
    ),
    "stub_body": (
        "def cx20_zero_rows_inplace(mat, idx):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "mat = t.arange(20.0).reshape(5, 4) + 1.0\n"
        "# rows: [1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]\n"
        "original_ptr = mat.data_ptr()\n"
        "before_others = mat[t.tensor([0, 2, 4])].clone()\n"
        "idx = t.tensor([1, 3], dtype=t.long)\n"
        "out = cx20_zero_rows_inplace(mat, idx)\n"
        "\n"
        "# Identity + storage preserved.\n"
        "assert out is mat, 'must return the same tensor object (in-place)'\n"
        "assert out.data_ptr() == original_ptr, 'must not reallocate'\n"
        "\n"
        "# Selected rows now zero.\n"
        "assert t.allclose(mat[1], t.zeros(4)), f'row 1 not zeroed: {mat[1]}'\n"
        "assert t.allclose(mat[3], t.zeros(4)), f'row 3 not zeroed: {mat[3]}'\n"
        "\n"
        "# Non-selected rows untouched.\n"
        "after_others = mat[t.tensor([0, 2, 4])]\n"
        "assert t.allclose(before_others, after_others), 'non-selected rows mutated'\n"
        "\n"
        "# Larger random case.\n"
        "rng = t.Generator().manual_seed(11)\n"
        "big = t.randn(20, 8, generator=rng)\n"
        "snap = big.clone()\n"
        "kill = t.tensor([0, 5, 7, 19], dtype=t.long)\n"
        "keep_mask = t.ones(20, dtype=t.bool); keep_mask[kill] = False\n"
        "cx20_zero_rows_inplace(big, kill)\n"
        "assert t.allclose(big[kill], t.zeros(4, 8))\n"
        "assert t.allclose(big[keep_mask], snap[keep_mask]), 'kept rows must be untouched'\n"
        "\n"
        "# Empty idx is a no-op.\n"
        "small = t.ones(3, 2)\n"
        "ptr = small.data_ptr()\n"
        "cx20_zero_rows_inplace(small, t.tensor([], dtype=t.long))\n"
        "assert t.allclose(small, t.ones(3, 2)), 'empty idx must leave mat untouched'\n"
        "assert small.data_ptr() == ptr"
    ),
    "solution_body": (
        "def cx20_zero_rows_inplace(mat, idx):\n"
        "    # Index-by-tensor on the LHS: PyTorch resolves mat[idx] to a slice-view of the\n"
        "    # selected rows, then broadcasts the RHS scalar zero across them. Storage is\n"
        "    # shared with mat — this is the same mechanic as slice-view mutation.\n"
        "    mat[idx] = 0.0\n"
        "    return mat"
    ),
    "solution_notes": (
        "Index-by-tensor works on both sides of `=`. On the right it READS (gather); on the left "
        "it WRITES (scatter). The write path shares storage with `mat` — there's no copy and no "
        "reallocation, which is why `data_ptr()` is unchanged. Don't reach for `index_copy_` or "
        "`scatter_` unless you need their specific semantics — plain `mat[idx] = value` is the "
        "idiom for this composition."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["index-by-tensor", "slice-view-mutation"],
    "lo": (
        "Combine index-by-tensor with slice-view mutation to zero a set of rows in place "
        "without reallocating storage."
    ),
}


# ===========================================================================
# cx21 — index a scalar position then unwrap to Python int
# ===========================================================================
spec_21 = {
    "atom_ids": ["index-by-tensor", "tensor-item-scalar"],
    "subtopics": _subs(["index-by-tensor", "tensor-item-scalar"]),
    "primary_atom": "index-by-tensor",
    "part": "part0",
    "exercise_index": 21,
    "exercise_title": "index a scalar position then unwrap via .item() to Python int",
    "slug": "index-by-tensor-then-item",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Indexing a tensor with a tensor index returns a TENSOR — even if you only asked for one "
        "element. `arr[t.tensor(2)]` gives you a 0-D tensor, not a Python scalar. To use that "
        "value as a normal Python int/float (e.g. as an index into a Python list, a dict key, or "
        "an argument to `range`), you must call `.item()` to unwrap it.\n\n"
        "Composing these two atoms: gather a scalar via index-by-tensor, then unwrap to a Python "
        "scalar via `.item()`. The composition exercises both the gather mechanic AND the "
        "tensor↔Python boundary."
    ),
    "prompt_body": (
        "Implement `cx21_lookup_int(arr, idx)` that:\n\n"
        "- Takes `arr` (1-D long tensor) and `idx` (a 0-D long tensor, e.g. `t.tensor(3)`).\n"
        "- Indexes `arr` by `idx` to get a 0-D tensor.\n"
        "- Unwraps it to a **Python `int`** via `.item()` and returns the int.\n\n"
        "The return type must be `int`, NOT a tensor — that's the whole point of the "
        "tensor-item-scalar atom."
    ),
    "stub_body": (
        "def cx21_lookup_int(arr, idx):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "arr = t.tensor([10, 20, 30, 40, 50], dtype=t.long)\n"
        "\n"
        "# Basic lookup.\n"
        "got = cx21_lookup_int(arr, t.tensor(2))\n"
        "assert isinstance(got, int), f'expected int, got {type(got).__name__}'\n"
        "assert got == 30, f'expected 30, got {got}'\n"
        "\n"
        "# Edge: first and last.\n"
        "assert cx21_lookup_int(arr, t.tensor(0)) == 10\n"
        "assert cx21_lookup_int(arr, t.tensor(4)) == 50\n"
        "\n"
        "# The result must be USABLE as a Python int — pass it to range(), dict key, etc.\n"
        "k = cx21_lookup_int(arr, t.tensor(1))\n"
        "assert list(range(k)) == list(range(20)), 'return must be usable in range()'\n"
        "d = {k: 'ok'}\n"
        "assert d[20] == 'ok', 'return must be hashable as a plain int'\n"
        "\n"
        "# Result is NOT a tensor — calling .item() on it would be a TypeError.\n"
        "result = cx21_lookup_int(arr, t.tensor(3))\n"
        "raised = False\n"
        "try: result.item()\n"
        "except AttributeError: raised = True\n"
        "assert raised, 'result must be a bare int, not a tensor'\n"
        "\n"
        "# Compose with a follow-up index: feed the int back as a python-side index.\n"
        "table = ['a', 'b', 'c', 'd', 'e', 'f', 'g']\n"
        "# arr=[10,20,30,40,50]; arr[idx=2] = 30 — too large for table — so use small arr.\n"
        "small = t.tensor([0, 2, 4, 6], dtype=t.long)\n"
        "i = cx21_lookup_int(small, t.tensor(1))\n"
        "assert table[i] == 'c', f'expected table[2]==\"c\", got table[{i}]={table[i]!r}'"
    ),
    "solution_body": (
        "def cx21_lookup_int(arr, idx):\n"
        "    # index-by-tensor: arr[idx] gives a 0-D tensor (NOT a Python int).\n"
        "    scalar_tensor = arr[idx]\n"
        "    # tensor-item-scalar: .item() unwraps a 0-D tensor to a Python scalar.\n"
        "    return scalar_tensor.item()"
    ),
    "solution_notes": (
        "The tensor↔Python boundary is real: tensors are first-class GPU/dtype-aware objects, "
        "Python ints are not. `.item()` is the explicit crossing — it forces a CPU sync (so use "
        "sparingly in hot paths) but is required for any operation that needs a real Python int "
        "(list indexing, dict keys, range, `if x == 3:`). Forgetting `.item()` is the #1 cause "
        "of 'expected int, got Tensor' bugs."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 1,
    "kcs": ["index-by-tensor", "tensor-item-scalar"],
    "lo": (
        "Gather a scalar element from a 1-D tensor and cross the tensor→Python boundary "
        "with .item() so the result is usable as a native Python int."
    ),
}


# ===========================================================================
# cx22 — view mutation propagates to underlying ndarray storage
# ===========================================================================
spec_22 = {
    "atom_ids": ["slice-view-mutation", "tensor-wraps-ndarray"],
    "subtopics": _subs(["slice-view-mutation", "tensor-wraps-ndarray"]),
    "primary_atom": "tensor-wraps-ndarray",
    "part": "part0",
    "exercise_index": 22,
    "exercise_title": "slice-view mutation on a from_numpy tensor propagates to the ndarray",
    "slug": "from-numpy-slice-view-aliasing",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`t.from_numpy(arr)` wraps an ndarray WITHOUT copying — the tensor and the array share "
        "the same underlying storage. Slice-view mutation says that assigning to `tensor[...] = "
        "value` mutates the source storage, not just a local copy. Compose them and you get: "
        "writing to a slice of a from_numpy tensor mutates the ORIGINAL ndarray, visible from "
        "the numpy side. The reverse is also true: mutating `arr` shows up in the tensor.\n\n"
        "Contrast with `t.tensor(arr)` which COPIES — mutations don't propagate. The composite "
        "drill verifies both directions of the from_numpy aliasing AND that `t.tensor` is "
        "isolated."
    ),
    "prompt_body": (
        "Implement `cx22_aliasing_writes(arr)` that:\n\n"
        "1. Build `wrapped = t.from_numpy(arr)` (shares storage) and `copied = t.tensor(arr)` "
        "(independent copy).\n"
        "2. Write `wrapped[0] = 999.0` — a slice-view mutation that must propagate to `arr` "
        "(through the from_numpy aliasing).\n"
        "3. Write `arr[1] = 777.0` — a numpy-side mutation that must propagate back into "
        "`wrapped` (but NOT into `copied`).\n"
        "4. Return a dict reporting `arr_after`, `wrapped_after`, `copied_after`, "
        "`wrapped_shares_storage`, `copied_shares_storage`.\n\n"
        "The contract: from_numpy aliases (mutations propagate both ways), t.tensor does not."
    ),
    "stub_body": (
        "def cx22_aliasing_writes(arr):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "arr = np.arange(5, dtype=np.float32)\n"
        "# arr = [0, 1, 2, 3, 4] — function will mutate arr[0]→999, arr[1]→777.\n"
        "report = cx22_aliasing_writes(arr)\n"
        "\n"
        "assert isinstance(report, dict)\n"
        "for k in ('arr_after','wrapped_after','copied_after',\n"
        "         'wrapped_shares_storage','copied_shares_storage'):\n"
        "    assert k in report, f'missing key {k!r}'\n"
        "\n"
        "# arr itself was mutated twice: arr[0]=999 (via wrapped), arr[1]=777 (direct numpy).\n"
        "assert report['arr_after'][0] == 999.0, f'arr[0] should be 999.0, got {report[\"arr_after\"][0]}'\n"
        "assert report['arr_after'][1] == 777.0, f'arr[1] should be 777.0, got {report[\"arr_after\"][1]}'\n"
        "\n"
        "# wrapped sees BOTH writes — slice-view through aliasing storage.\n"
        "wrapped_after = report['wrapped_after']\n"
        "assert float(wrapped_after[0]) == 999.0\n"
        "assert float(wrapped_after[1]) == 777.0\n"
        "assert report['wrapped_shares_storage'] is True, 'from_numpy MUST share storage'\n"
        "\n"
        "# copied saw NEITHER write — it was an independent snapshot at t.tensor(arr) time.\n"
        "copied_after = report['copied_after']\n"
        "assert float(copied_after[0]) == 0.0, 'copied[0] must be 0.0 (snapshot before mutation)'\n"
        "assert float(copied_after[1]) == 1.0, 'copied[1] must be 1.0 (snapshot before mutation)'\n"
        "assert report['copied_shares_storage'] is False, 't.tensor MUST NOT share storage'\n"
        "\n"
        "# Confirm arr itself was actually mutated (not just claimed).\n"
        "assert arr[0] == 999.0\n"
        "assert arr[1] == 777.0"
    ),
    "solution_body": (
        "def cx22_aliasing_writes(arr):\n"
        "    # tensor-wraps-ndarray: from_numpy shares storage; t.tensor copies.\n"
        "    wrapped = t.from_numpy(arr)\n"
        "    copied = t.tensor(arr)\n"
        "\n"
        "    # slice-view mutation through the alias: wrapped[0]=999 hits arr's storage.\n"
        "    wrapped[0] = 999.0\n"
        "    # And the reverse: numpy-side mutation is visible in wrapped (same storage).\n"
        "    arr[1] = 777.0\n"
        "\n"
        "    # Storage-sharing detection: from_numpy shares; t.tensor does not.\n"
        "    wrapped_shares = wrapped.data_ptr() == arr.__array_interface__['data'][0]\n"
        "    copied_shares = copied.data_ptr() == arr.__array_interface__['data'][0]\n"
        "    return {\n"
        "        'arr_after': arr.copy(),\n"
        "        'wrapped_after': wrapped.clone(),\n"
        "        'copied_after': copied.clone(),\n"
        "        'wrapped_shares_storage': bool(wrapped_shares),\n"
        "        'copied_shares_storage': bool(copied_shares),\n"
        "    }"
    ),
    "solution_notes": (
        "The two atoms collapse into one mental model: from_numpy creates an ALIAS, t.tensor "
        "creates a SNAPSHOT. Slice-view writes (`wrapped[0]=999`) just exercise that alias — "
        "they're not special, they're the normal in-place semantics of a view. If you wanted the "
        "tensor to be independent, you'd write `t.tensor(arr)` (or `t.from_numpy(arr).clone()`). "
        "Don't use `from_numpy` if the source ndarray might mutate under you."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["slice-view-mutation", "tensor-wraps-ndarray"],
    "lo": (
        "Show that slice-view mutation on a from_numpy tensor propagates to the underlying "
        "ndarray storage, while t.tensor produces an isolated copy."
    ),
}


# ===========================================================================
# cx23 — fancy-index along a dim of a rearranged tensor
# ===========================================================================
spec_23 = {
    "atom_ids": ["arange-fancy-index-cross-entropy", "einops-rearrange"],
    "subtopics": _subs(["arange-fancy-index-cross-entropy", "einops-rearrange"]),
    "primary_atom": "arange-fancy-index-cross-entropy",
    "part": "part0",
    "exercise_index": 23,
    "exercise_title": "rearrange then fancy-index — pick per-sample logit on (C, B) form",
    "slug": "rearrange-then-arange-fancy-index",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA code sometimes stores logits as `(C, B)` (class-first) and sometimes as `(B, C)` "
        "(batch-first). The arange-fancy-index idiom is written for batch-first: `logits[arange"
        "(B), target]`. If your incoming tensor is class-first, you `rearrange(logits, 'c b -> "
        "b c')` first and THEN apply the idiom.\n\n"
        "Composing these two atoms exercises a real ARENA bug-magnet: indexing the wrong axis. "
        "If you forget the rearrange and write `logits[arange(B), target]` on a (C, B) tensor, "
        "you'll index the CLASS axis with arange(B) — silently wrong (or shape-error if "
        "B != C)."
    ),
    "prompt_body": (
        "Implement `cx23_pick_after_rearrange(logits_cb, target)` that:\n\n"
        "- Takes `logits_cb` of shape `(C, B)` (class-first) and `target` of shape `(B,)`.\n"
        "- Uses `einops.rearrange` to swap to batch-first `(B, C)`.\n"
        "- Picks per-sample target logits via `logits_bc[arange(B), target]`, returning shape "
        "`(B,)`.\n\n"
        "The output must match what you'd get by applying the same pick to the batch-first "
        "form directly. No transpose tricks (`.T`) — use `rearrange` explicitly, that's the atom "
        "being exercised."
    ),
    "stub_body": (
        "def cx23_pick_after_rearrange(logits_cb, target):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Trivial: C=3, B=4. logits_cb[c, b] = c*10 + b.\n"
        "C, B = 3, 4\n"
        "logits_cb = t.tensor([[c*10.0 + b for b in range(B)] for c in range(C)])\n"
        "# Equivalent batch-first form: logits_bc[b, c] = c*10 + b.\n"
        "target = t.tensor([0, 1, 2, 1])\n"
        "got = cx23_pick_after_rearrange(logits_cb, target)\n"
        "assert got.shape == (B,), f'shape: {got.shape}'\n"
        "# Per-sample target logit = target[b]*10 + b.\n"
        "expected = t.tensor([0*10.0 + 0, 1*10.0 + 1, 2*10.0 + 2, 1*10.0 + 3])\n"
        "assert t.allclose(got, expected), f'got {got}, expected {expected}'\n"
        "\n"
        "# Cross-check vs the batch-first reference pick.\n"
        "logits_bc = rearrange(logits_cb, 'c b -> b c')\n"
        "ref = logits_bc[t.arange(B), target]\n"
        "assert t.allclose(got, ref)\n"
        "\n"
        "# Larger random case where C != B (so axis confusion would shape-error).\n"
        "rng = t.Generator().manual_seed(2)\n"
        "C2, B2 = 7, 32\n"
        "logits_cb2 = t.randn(C2, B2, generator=rng)\n"
        "target2 = t.randint(0, C2, (B2,), generator=rng)\n"
        "got2 = cx23_pick_after_rearrange(logits_cb2, target2)\n"
        "assert got2.shape == (B2,)\n"
        "logits_bc2 = rearrange(logits_cb2, 'c b -> b c')\n"
        "ref2 = logits_bc2[t.arange(B2), target2]\n"
        "assert t.allclose(got2, ref2)\n"
        "\n"
        "# Sanity: dtype preserved.\n"
        "logits_d = t.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=t.float64)  # (C=2, B=2)\n"
        "target_d = t.tensor([0, 1])\n"
        "out_d = cx23_pick_after_rearrange(logits_d, target_d)\n"
        "assert out_d.dtype == t.float64"
    ),
    "solution_body": (
        "def cx23_pick_after_rearrange(logits_cb, target):\n"
        "    # einops-rearrange: swap class-first to batch-first.\n"
        "    logits_bc = rearrange(logits_cb, 'c b -> b c')\n"
        "    # arange-fancy-index on the canonical (B, C) shape.\n"
        "    B = logits_bc.shape[0]\n"
        "    return logits_bc[t.arange(B), target]"
    ),
    "solution_notes": (
        "The two-step pattern — `rearrange` to a canonical shape, then apply the canonical idiom "
        "— is everywhere in ARENA. `einops.rearrange` is preferred over `.T` / `.permute` because "
        "the axis names document the intent: the next reader can SEE that the function expects "
        "batch-first downstream. If `C == B` (e.g. self-attention with C=seq_len, B=batch=C), "
        "forgetting the rearrange would still type-check but be silently wrong — that's why the "
        "test uses non-square shapes for the larger case."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["arange-fancy-index-cross-entropy", "einops-rearrange"],
    "lo": (
        "Compose einops.rearrange with the arange-fancy-index idiom to pick per-sample target "
        "logits from a class-first (C, B) tensor."
    ),
}


# ===========================================================================
# cx24 — rearrange a 1-D logit row to add an explicit broadcast axis before LSE
# ===========================================================================
spec_24 = {
    "atom_ids": ["logsumexp-cross-entropy", "einops-rearrange"],
    "subtopics": _subs(["logsumexp-cross-entropy", "einops-rearrange"]),
    "primary_atom": "logsumexp-cross-entropy",
    "part": "part0",
    "exercise_index": 24,
    "exercise_title": "rearrange row to (1, C) for stable per-sample logsumexp CE",
    "slug": "rearrange-then-logsumexp-ce",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Stable cross-entropy is `mean(logsumexp(logits, dim=-1) - logits[arange(B), target])`. "
        "`logsumexp` along the class axis subtracts the row-max internally and avoids the "
        "`exp(big-number)` overflow that the naive `-log softmax` form has. But it needs a "
        "well-formed class axis to reduce over.\n\n"
        "If your input is a single 1-D logit row of shape `(C,)` (no batch axis yet), the cross-"
        "entropy machinery expects `(B, C)` so the `dim=-1` reduction is unambiguous and the "
        "`arange(B)` target-index still works. `einops.rearrange(row, 'c -> 1 c')` is the "
        "explicit, self-documenting way to add that broadcast axis — vs `row.unsqueeze(0)` "
        "which works but hides the intent."
    ),
    "prompt_body": (
        "Implement `cx24_ce_single_row(logits_row, target_class)` that:\n\n"
        "- Takes `logits_row` of shape `(C,)` (a single example's class logits) and "
        "`target_class` — a Python int OR a 0-D tensor — the correct class index.\n"
        "- Uses `einops.rearrange(logits_row, 'c -> 1 c')` to lift to a (1, C) batch-of-one.\n"
        "- Computes stable cross-entropy via `logsumexp(logits, dim=-1) - logits[arange(B), "
        "target]`. The result is a scalar 0-D tensor (the per-sample CE; mean of one sample is "
        "itself).\n\n"
        "Must be numerically stable — `logits_row = [1000.0, 999.0, 998.0]` with target=0 must "
        "NOT overflow."
    ),
    "stub_body": (
        "def cx24_ce_single_row(logits_row, target_class):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import math\n"
        "import torch.nn.functional as F\n"
        "\n"
        "# Uniform 3-class → CE = log(3) for any target.\n"
        "loss = cx24_ce_single_row(t.zeros(3), 1)\n"
        "assert loss.shape == (), f'expected scalar, got {loss.shape}'\n"
        "assert abs(loss.item() - math.log(3)) < 1e-5, f'expected log(3), got {loss.item()}'\n"
        "\n"
        "# Match torch.nn.functional.cross_entropy on a typical row.\n"
        "row = t.tensor([2.0, 1.0, 0.1, -0.5])\n"
        "for tgt in range(4):\n"
        "    ours = cx24_ce_single_row(row, tgt)\n"
        "    ref = F.cross_entropy(row.unsqueeze(0), t.tensor([tgt]))\n"
        "    assert abs(ours.item() - ref.item()) < 1e-5, (\n"
        "        f'target={tgt}: ours={ours.item()}, ref={ref.item()}'\n"
        "    )\n"
        "\n"
        "# Stability stress: logits at scale 1000 must not overflow.\n"
        "big_row = t.tensor([1000.0, 999.0, 998.0])\n"
        "big_loss = cx24_ce_single_row(big_row, 0)\n"
        "assert t.isfinite(big_loss).item(), (\n"
        "    f'huge logits must not produce inf/nan; got {big_loss.item()} — '\n"
        "    'are you using logsumexp, or did you call exp(logits) directly?'\n"
        ")\n"
        "expected = math.log(1 + math.exp(-1) + math.exp(-2))\n"
        "assert abs(big_loss.item() - expected) < 1e-4, f'big-logit loss wrong: {big_loss.item()}'\n"
        "\n"
        "# Sanity: naive exp(big_row) would overflow at this scale.\n"
        "assert not t.isfinite(t.exp(big_row)).all().item(), (\n"
        "    'sanity: exp(1000) should be inf — stability test is meaningful'\n"
        ")\n"
        "\n"
        "# Confident-correct → near-zero loss.\n"
        "confident = t.tensor([100.0, 0.0, 0.0])\n"
        "loss_conf = cx24_ce_single_row(confident, 0)\n"
        "assert loss_conf.item() < 1e-5, f'confident-correct should be ~0, got {loss_conf.item()}'\n"
        "\n"
        "# target_class accepts a 0-D tensor too.\n"
        "loss_t = cx24_ce_single_row(row, t.tensor(2))\n"
        "ref_t = F.cross_entropy(row.unsqueeze(0), t.tensor([2]))\n"
        "assert abs(loss_t.item() - ref_t.item()) < 1e-5"
    ),
    "solution_body": (
        "def cx24_ce_single_row(logits_row, target_class):\n"
        "    # einops-rearrange: lift 1-D row to a (1, C) batch-of-one.\n"
        "    # Self-documenting: the next reader sees we've added a batch axis on purpose.\n"
        "    logits = rearrange(logits_row, 'c -> 1 c')\n"
        "\n"
        "    # Coerce target to a (1,)-shape long tensor regardless of whether int or 0-D.\n"
        "    if isinstance(target_class, int):\n"
        "        target = t.tensor([target_class], dtype=t.long)\n"
        "    else:\n"
        "        target = target_class.view(1).long()\n"
        "\n"
        "    # logsumexp-cross-entropy: stable CE via logsumexp - arange-fancy-index.\n"
        "    B = logits.shape[0]  # == 1\n"
        "    lse = t.logsumexp(logits, dim=-1)\n"
        "    picked = logits[t.arange(B), target]\n"
        "    per_sample = lse - picked\n"
        "    return per_sample.mean()  # mean of 1 == itself, scalar 0-D."
    ),
    "solution_notes": (
        "Two reasons rearrange beats unsqueeze here: (1) the einops string `'c -> 1 c'` reads as "
        "'this is a class-axis row being given a batch axis' — no axis-counting required; (2) if "
        "you later refactor to handle multiple rows, you can change to `'b c -> b c'` (identity) "
        "without rewriting the rest of the function. Stability comes from `logsumexp`, NOT from "
        "the rearrange — but the rearrange is what makes `dim=-1` and `arange(B)` agree on what "
        "the batch axis IS."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["logsumexp-cross-entropy", "einops-rearrange"],
    "lo": (
        "Combine einops.rearrange with the logsumexp-cross-entropy idiom to compute a "
        "numerically stable per-sample cross-entropy from a single 1-D logit row."
    ),
}


SPECS = [spec_19, spec_20, spec_21, spec_22, spec_23, spec_24]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
