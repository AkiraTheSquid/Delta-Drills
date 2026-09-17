#!/usr/bin/env python3
"""Author 8 deepening ex2 notebooks for high-frequency atoms.

Each spec adds ONE new exercise that probes a DISTINCT facet from the
existing ex1 in the same folder. PS4 framing (one LO, one Bloom, max 2 KCs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ============================================================== 1: slice-view-mutation
SPEC_SLICE_VIEW = {
    "atom_id": "slice-view-mutation",
    "subtopic": "PyTorch: Slice view mutation",
    "topic_folder": "prereqs_tensor_mechanics",
    "atom_recap_md": (
        "## Slice view mutation — quick refresher\n"
        "\n"
        "Slicing returns a **view** sharing storage with the source. Writes through "
        "the view alias the source — and crucially, `data_ptr()` lets you *prove* "
        "the aliasing from the outside:\n"
        "\n"
        "```python\n"
        "row = x[2]\n"
        "assert row.data_ptr() == x.data_ptr() + 2 * x.stride(0) * x.element_size()\n"
        "```\n"
        "\n"
        "The previous drill (ex1) zeroed the **diagonal** via `mat.diagonal()[:] = 0`. "
        "This drill zeroes a **row** via the more direct `mat[i, :] = val` syntax and "
        "uses `data_ptr()` to assert the source storage was mutated, not replaced."
    ),
    "exercise_index": 2,
    "exercise_title": "zero a row in-place and prove storage aliasing via data_ptr",
    "slug": "zero-row-inplace-prove-data-ptr-aliasing",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["slice", "view", "in-place", "row", "data_ptr"],
    "kcs": ["slice-returns-view", "view-writes-alias-source"],
    "lo": (
        "Analyze the storage-aliasing property of slice-views by mutating a single "
        "row of a matrix in place via `mat[i, :] = val` and verifying `data_ptr()` "
        "is unchanged across the write."
    ),
    "prompt_body": (
        "Implement `ex2_zero_row_inplace(mat, i, value)`.\n\n"
        "Given a `(R, C)` float tensor `mat`, a row index `i`, and a scalar `value`, "
        "set every entry of row `i` to `value` **in place** using the "
        "`mat[i, :] = value` slice-assignment syntax. Then return `(mat, did_alias)` "
        "where `did_alias` is a bool: `True` iff `mat.data_ptr()` was unchanged across "
        "the write (proving the write went to the original storage, not a fresh "
        "allocation).\n\n"
        "**Rules.**\n"
        "1. No reassignment of `mat` — work in place.\n"
        "2. Capture `mat.data_ptr()` BEFORE the write, then again AFTER, and set "
        "`did_alias = (ptr_before == ptr_after)`.\n"
        "3. Other rows must be untouched.\n\n"
        "Inputs:\n"
        "- `mat`: `(R, C)` float tensor.\n"
        "- `i`: int row index.\n"
        "- `value`: Python float (gets broadcast across the row).\n\n"
        "Output: `(mat, did_alias)` — the same tensor object plus the aliasing flag."
    ),
    "stub": (
        "def ex2_zero_row_inplace(mat: Tensor, i: int, value: float):\n"
        '    """Zero row i in place via mat[i, :] = value; return (mat, did_alias)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "mat = t.arange(12, dtype=t.float32).reshape(3, 4) + 1.0\n"
        "# mat = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]\n"
        "ptr_before = mat.data_ptr()\n"
        "out, did_alias = ex2_zero_row_inplace(mat, 1, 0.0)\n"
        "\n"
        "# Returned the SAME object — in-place semantics.\n"
        "assert out is mat, 'must return the same tensor object'\n"
        "assert out.data_ptr() == ptr_before, 'storage must not be reallocated'\n"
        "assert did_alias is True, f'did_alias must be True (writing through a slice-view aliases), got {did_alias!r}'\n"
        "\n"
        "# Row 1 zeroed; rows 0 and 2 untouched.\n"
        "assert t.equal(mat[1], t.zeros(4)), f'row 1 not zeroed: {mat[1]}'\n"
        "assert t.equal(mat[0], t.tensor([1., 2., 3., 4.])), f'row 0 changed: {mat[0]}'\n"
        "assert t.equal(mat[2], t.tensor([9., 10., 11., 12.])), f'row 2 changed: {mat[2]}'\n"
        "\n"
        "# Non-zero value test.\n"
        "mat2 = t.zeros(4, 3)\n"
        "out2, alias2 = ex2_zero_row_inplace(mat2, 2, 7.5)\n"
        "assert alias2 is True\n"
        "assert t.equal(mat2[2], t.full((3,), 7.5)), f'row 2 should be 7.5 fill, got {mat2[2]}'\n"
        "assert t.equal(mat2[0], t.zeros(3)) and t.equal(mat2[1], t.zeros(3)) and t.equal(mat2[3], t.zeros(3)), 'other rows must remain zero'\n"
        "\n"
        "# Demonstrate the dual proof — an EXTERNAL view of the same row also sees the change.\n"
        "mat3 = t.ones(2, 5)\n"
        "external_view = mat3[0]            # also a view, shares storage\n"
        "ex2_zero_row_inplace(mat3, 0, -1.0)\n"
        "assert t.equal(external_view, t.full((5,), -1.0)), (\n"
        "    f'external view should see the mutation (storage is shared), got {external_view}'\n"
        ")"
    ),
    "solution_body": (
        "def ex2_zero_row_inplace(mat: Tensor, i: int, value: float):\n"
        "    ptr_before = mat.data_ptr()\n"
        "    mat[i, :] = value\n"
        "    ptr_after = mat.data_ptr()\n"
        "    did_alias = (ptr_before == ptr_after)\n"
        "    return mat, did_alias"
    ),
    "solution_notes": (
        "**Why `data_ptr()` is the cleanest aliasing proof.** Two tensors share "
        "storage iff their `data_ptr()` values are equal (for the same offset). "
        "Capturing it before and after a write proves the mutation went to the "
        "original allocation — it's how PyTorch's own tests verify in-place ops.\n\n"
        "**Difference from ex1.** ex1 used `mat.diagonal()[:] = 0` — an indexed "
        "view of a non-contiguous axis. ex2 uses the more common `mat[i, :] = val` "
        "row-slice form and adds the explicit `data_ptr()` check, which catches "
        "the subtle bug where someone writes `mat = mat.clone(); mat[i] = val` "
        "and breaks aliasing without noticing."
    ),
    "extra_imports": [],
}


# ============================================================== 2: boolean-mask-combine
SPEC_BOOL_MASK = {
    "atom_id": "boolean-mask-combine",
    "subtopic": "Numpy: Boolean mask combine",
    "topic_folder": "prereqs_einops_advanced",
    "atom_recap_md": (
        "## Combining boolean masks — quick refresher\n"
        "\n"
        "Three logical operators on bool tensors: `&`, `|`, `~`. All elementwise, "
        "all broadcast normally. PyTorch also exposes **XOR** via `^` — true iff "
        "exactly one operand is true.\n"
        "\n"
        "```python\n"
        "a ^ b   # symmetric difference\n"
        "~(a | b)  # neither\n"
        "a & ~b    # a but not b\n"
        "```\n"
        "\n"
        "The previous drill (ex1) ANDed five inside-test predicates into a single "
        "mask. This drill exercises the **OR / NOT / XOR** half of the algebra: "
        "combining several outlier-detection masks via OR (any outlier), AND-NOT "
        "(outlier on metric A but normal on B), and XOR (disagreement between two "
        "detectors)."
    ),
    "exercise_index": 2,
    "exercise_title": "outlier-mask algebra via OR, AND-NOT, and XOR",
    "slug": "outlier-mask-algebra-or-and-not-xor",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["or", "not", "xor", "outlier", "set-algebra"],
    "kcs": ["mask-bitwise-and-or", "mask-parenthesize-comparisons"],
    "lo": (
        "Apply elementwise `|`, `~`, and `^` to combine three per-row outlier "
        "predicates into (any-outlier, A-only-outlier, A-XOR-B disagreement) masks."
    ),
    "prompt_body": (
        "Implement `ex2_outlier_masks(x, threshold)`.\n\n"
        "Given a 1-D tensor `x` of length `N` and a scalar `threshold`, build three "
        "boolean masks and return them as a tuple `(any_out, a_only, disagree)`:\n\n"
        "1. **Detector A — magnitude:** `mask_a = x.abs() > threshold`.\n"
        "2. **Detector B — sign-flip:** `mask_b = x < 0`.\n"
        "3. **Detector C — non-finite:** `mask_c = ~t.isfinite(x)`.\n"
        "\n"
        "Combine:\n"
        "- `any_out  = mask_a | mask_b | mask_c`  (flagged by at least one detector)\n"
        "- `a_only   = mask_a & ~mask_b & ~mask_c`  (flagged by A but not B and not C)\n"
        "- `disagree = mask_a ^ mask_b`  (A and B disagree — XOR)\n"
        "\n"
        "Each output must be `dtype=bool` and shape `(N,)`.\n\n"
        "**Critical:** parenthesize every comparison. `&`, `|`, `^` bind tighter "
        "than `<`, `>`, `==` — `x > 0 | x < 1` parses as `x > (0 | x) < 1`."
    ),
    "stub": (
        "def ex2_outlier_masks(x: Tensor, threshold: float):\n"
        '    """Return (any_out, a_only, disagree) bool masks."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Construct a vector hitting each combination explicitly.\n"
        "x = t.tensor([\n"
        "    0.5,      # neither A (|x|<2) nor B (x>=0) nor C (finite) — clean\n"
        "    3.0,      # A only (|x|>2, x>=0, finite)\n"
        "    -0.1,     # B only\n"
        "    -5.0,     # A and B\n"
        "    float('nan'),  # C only (and A? |nan|>2 is False; B? nan<0 is False) — pure C\n"
        "    float('inf'),  # C and A (inf>2 is True), not B\n"
        "])\n"
        "any_out, a_only, disagree = ex2_outlier_masks(x, threshold=2.0)\n"
        "\n"
        "assert any_out.dtype == t.bool and a_only.dtype == t.bool and disagree.dtype == t.bool\n"
        "assert any_out.shape == (6,) and a_only.shape == (6,) and disagree.shape == (6,)\n"
        "\n"
        "expected_any      = t.tensor([False, True,  True,  True,  True,  True])\n"
        "expected_a_only   = t.tensor([False, True,  False, False, False, False])\n"
        "# disagree = A XOR B  →  exactly one is True\n"
        "#  idx 0: A=F B=F → F\n"
        "#  idx 1: A=T B=F → T\n"
        "#  idx 2: A=F B=T → T\n"
        "#  idx 3: A=T B=T → F\n"
        "#  idx 4: A=F B=F → F (nan>2 and nan<0 both False)\n"
        "#  idx 5: A=T B=F → T (inf>2 True, inf<0 False)\n"
        "expected_disagree = t.tensor([False, True,  True,  False, False, True])\n"
        "\n"
        "assert t.equal(any_out, expected_any), f'any_out: got {any_out}, want {expected_any}'\n"
        "assert t.equal(a_only, expected_a_only), f'a_only: got {a_only}, want {expected_a_only}'\n"
        "assert t.equal(disagree, expected_disagree), f'disagree: got {disagree}, want {expected_disagree}'\n"
        "\n"
        "# Random sanity vs longhand.\n"
        "rng = t.Generator().manual_seed(7)\n"
        "y = t.randn(200, generator=rng) * 3\n"
        "ao, aol, dis = ex2_outlier_masks(y, threshold=2.5)\n"
        "ma = y.abs() > 2.5\n"
        "mb = y < 0\n"
        "mc = ~t.isfinite(y)\n"
        "assert t.equal(ao, ma | mb | mc)\n"
        "assert t.equal(aol, ma & ~mb & ~mc)\n"
        "assert t.equal(dis, ma ^ mb)"
    ),
    "solution_body": (
        "def ex2_outlier_masks(x: Tensor, threshold: float):\n"
        "    mask_a = x.abs() > threshold\n"
        "    mask_b = x < 0\n"
        "    mask_c = ~t.isfinite(x)\n"
        "    any_out = mask_a | mask_b | mask_c\n"
        "    a_only = mask_a & ~mask_b & ~mask_c\n"
        "    disagree = mask_a ^ mask_b\n"
        "    return any_out, a_only, disagree"
    ),
    "solution_notes": (
        "**XOR `^` is the disagreement detector.** When you have two independent "
        "outlier signals, `a ^ b` highlights the rows where they disagree — these "
        "are the cases worth manually reviewing.\n\n"
        "**`isfinite` is the canonical NaN/Inf check.** `x != x` works for NaN "
        "only; `~t.isfinite(x)` catches both NaN and ±inf in one shot.\n\n"
        "**Difference from ex1.** ex1 built `mask & mask & mask & ...` — an "
        "intersection of *constraints*. ex2 builds `mask | mask | mask` (union) "
        "and `mask ^ mask` (symmetric difference), exercising the disjunctive "
        "half of the boolean algebra."
    ),
    "extra_imports": [],
}


# ============================================================== 3: relu-elementwise-max
SPEC_RELU = {
    "atom_id": "relu-elementwise-max",
    "subtopic": "CNN: ReLU as elementwise max",
    "topic_folder": "prereqs_cnn_deep",
    "atom_recap_md": (
        "## ReLU as elementwise max — in-place edition\n"
        "\n"
        "Three forward forms compute the same values; their storage semantics differ:\n"
        "\n"
        "```python\n"
        "y = t.maximum(x, t.tensor(0.0))   # OUT-OF-PLACE — y is a NEW tensor\n"
        "y = F.relu(x)                      # OUT-OF-PLACE — same\n"
        "x.relu_()                          # IN-PLACE — mutates x; returns x\n"
        "```\n"
        "\n"
        "The previous drill (ex1) used `t.maximum` and inspected the sub-gradient at "
        "`x=0`. This drill targets the **in-place form `x.relu_()`** — proving via "
        "`data_ptr()` that the input storage is reused, and understanding why that "
        "saves memory but breaks autograd for non-leaf tensors that need their input "
        "for backward."
    ),
    "exercise_index": 2,
    "exercise_title": "in-place relu_ and verify storage identity",
    "slug": "in-place-relu-verify-storage-identity",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["relu", "in-place", "data_ptr", "storage"],
    "kcs": ["relu-max-with-zero", "relu-in-place-aliases-input"],
    "lo": (
        "Analyze the in-place ReLU operator `Tensor.relu_()` by applying it to a "
        "tensor and verifying via `data_ptr()` that the returned tensor shares "
        "storage with the input."
    ),
    "prompt_body": (
        "Implement `ex2_relu_inplace(x)` that applies ReLU **in place** and returns "
        "`(out, same_storage)` where `same_storage` is a bool: `True` iff the "
        "returned tensor's `data_ptr()` equals the input's BEFORE-call `data_ptr()`.\n\n"
        "**Required call.** Use `x.relu_()` (the trailing underscore is PyTorch's "
        "convention for in-place ops). NOT `F.relu(x)`, NOT `t.maximum(x, 0)` — "
        "those allocate new tensors and would yield `same_storage = False`.\n\n"
        "**Required no_grad context.** Wrap the in-place call in `with t.no_grad():` "
        "so the in-place mutation doesn't pollute the autograd graph (this is also "
        "what PyTorch's own activation modules do under the `inplace=True` flag).\n\n"
        "Input: `x` — float tensor of any shape (NOT necessarily a leaf with "
        "`requires_grad`).\n"
        "Output: `(out, same_storage)` — `out is x` and `same_storage == True`.\n\n"
        "After return, `x` itself has had negatives replaced with 0."
    ),
    "stub": (
        "def ex2_relu_inplace(x: Tensor):\n"
        '    """Apply ReLU in place via x.relu_(); return (out, same_storage)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])\n"
        "ptr_before = x.data_ptr()\n"
        "out, same_storage = ex2_relu_inplace(x)\n"
        "\n"
        "# Identity and aliasing.\n"
        "assert out is x, 'in-place op must return the SAME object'\n"
        "assert out.data_ptr() == ptr_before, 'data_ptr changed → storage was not reused'\n"
        "assert same_storage is True, f'same_storage must be True, got {same_storage!r}'\n"
        "\n"
        "# Forward values.\n"
        "expected = t.tensor([0.0, 0.0, 0.0, 0.5, 2.0])\n"
        "assert t.equal(x, expected), f'forward wrong: {x} vs {expected}'\n"
        "\n"
        "# Larger random shape — confirm storage is reused for any shape.\n"
        "rng = t.Generator().manual_seed(0)\n"
        "y = t.randn(4, 5, generator=rng)\n"
        "y_ref_relu = y.clamp(min=0).clone()\n"
        "ptr_y = y.data_ptr()\n"
        "out_y, ok_y = ex2_relu_inplace(y)\n"
        "assert ok_y is True\n"
        "assert out_y.data_ptr() == ptr_y\n"
        "assert t.allclose(y, y_ref_relu), f'larger-tensor forward wrong'\n"
        "\n"
        "# Anti-cheat: ensure they used .relu_(), not e.g. clamp_().\n"
        "# We can't introspect the call directly, but we can verify the mutation\n"
        "# matches relu (not, say, abs) for negatives.\n"
        "z = t.tensor([-3.0, -1.0, 4.0])\n"
        "out_z, _ = ex2_relu_inplace(z)\n"
        "assert t.equal(z, t.tensor([0.0, 0.0, 4.0])), 'mutation must match relu semantics'\n"
        "# After in-place ReLU, all entries are non-negative.\n"
        "assert (z >= 0).all().item(), 'all entries must be >= 0 post-ReLU'"
    ),
    "solution_body": (
        "def ex2_relu_inplace(x: Tensor):\n"
        "    ptr_before = x.data_ptr()\n"
        "    with t.no_grad():\n"
        "        out = x.relu_()\n"
        "    same_storage = (out.data_ptr() == ptr_before)\n"
        "    return out, same_storage"
    ),
    "solution_notes": (
        "**The underscore convention.** PyTorch suffixes in-place ops with `_`. "
        "`relu_`, `add_`, `mul_`, `clamp_` all mutate `self` and return `self`. "
        "The non-underscore counterpart (`relu`, `add`, etc.) returns a fresh "
        "tensor — `data_ptr()` differs.\n\n"
        "**When the memory savings are worth it.** Activation modules expose "
        "`inplace=True` (e.g. `nn.ReLU(inplace=True)`) precisely because for "
        "deep networks the activation tensors are large and the gradient w.r.t. "
        "the pre-activation can be recovered from the post-activation alone "
        "(zero stays zero, positive stays positive). For most other ops, "
        "in-place saves memory but breaks autograd because the original tensor "
        "is needed for backward.\n\n"
        "**Difference from ex1.** ex1 used the **out-of-place** `t.maximum` and "
        "verified the autograd sub-gradient at `x=0`. ex2 uses the **in-place** "
        "`relu_` and verifies storage identity via `data_ptr()`."
    ),
    "extra_imports": [],
}


# ============================================================== 4: module-extra-repr
SPEC_EXTRA_REPR = {
    "atom_id": "module-extra-repr",
    "subtopic": "PyTorch: Module __repr__",
    "topic_folder": "prereqs_pytorch_modules",
    "atom_recap_md": (
        "## `extra_repr` — nested module edition\n"
        "\n"
        "When a Module has child modules registered as attributes, `print(parent)` "
        "auto-indents and prints each child's `__repr__` on its own line — this is "
        "PyTorch's tree-pretty-printer:\n"
        "\n"
        "```\n"
        "MyMLP(\n"
        "  hidden_dim=4\n"
        "  (linear): MyReprLinear(in_features=3, out_features=4, bias=True)\n"
        "  (act): ReLU()\n"
        ")\n"
        "```\n"
        "\n"
        "The previous drill (ex1) implemented `extra_repr` on a **flat** Linear-style "
        "module. This drill exercises the same operator on a **nested** parent that "
        "owns a child Linear — confirming the parent's `extra_repr` appears at the "
        "TOP and the child's full repr appears INDENTED under it."
    ),
    "exercise_index": 2,
    "exercise_title": "nested module repr — parent extra_repr above an indented child",
    "slug": "nested-module-repr-parent-and-indented-child",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["extra_repr", "nested-module", "child-indentation", "tree-print"],
    "kcs": ["extra-repr-returns-string", "extra-repr-shows-up-in-nested-print"],
    "lo": (
        "Apply `extra_repr` to a parent Module that owns a registered child Linear, "
        "and verify the child's full repr appears INDENTED beneath the parent's "
        "extra_repr in the default tree print."
    ),
    "prompt_body": (
        "Implement `ex2_build_mlp(in_features, hidden_dim, out_features)`. Build a "
        "two-layer MLP-shaped parent module that stores `hidden_dim` as the ONLY "
        "thing surfaced via `extra_repr`, and owns a registered child `nn.Linear`:\n\n"
        "```python\n"
        "class MyMLP(nn.Module):\n"
        "    def __init__(self, in_features, hidden_dim, out_features):\n"
        "        super().__init__()\n"
        "        self.hidden_dim = hidden_dim\n"
        "        self.fc1 = nn.Linear(in_features, hidden_dim)\n"
        "        self.fc2 = nn.Linear(hidden_dim, out_features)\n"
        "    def extra_repr(self):\n"
        "        return f'hidden_dim={self.hidden_dim}'\n"
        "    def forward(self, x):\n"
        "        return self.fc2(t.relu(self.fc1(x)))\n"
        "```\n"
        "\n"
        "Then return an instance from `ex2_build_mlp(...)`.\n\n"
        "**Why this is the deepening facet.** The test verifies BOTH that "
        "`extra_repr` contains `hidden_dim=...` AND that `repr(model)` contains the "
        "child Linear's `(fc1): Linear(...)` line with INDENTATION (the leading "
        "two-space indent that PyTorch's `__repr__` inserts under a parent). The "
        "child contribution comes for free from `nn.Module.__repr__` recursion — "
        "the drill confirms `extra_repr` doesn't break that recursion."
    ),
    "stub": (
        "def ex2_build_mlp(in_features: int, hidden_dim: int, out_features: int):\n"
        '    """Return a MyMLP instance: extra_repr surfaces hidden_dim; owns fc1, fc2."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "model = ex2_build_mlp(in_features=3, hidden_dim=8, out_features=2)\n"
        "import torch.nn as nn\n"
        "\n"
        "assert isinstance(model, nn.Module), 'must be an nn.Module'\n"
        "\n"
        "# extra_repr surfaces hidden_dim only.\n"
        "extra = model.extra_repr()\n"
        "assert isinstance(extra, str), f'extra_repr must return str, got {type(extra)}'\n"
        "assert 'hidden_dim=8' in extra, f'extra_repr must contain hidden_dim=8, got {extra!r}'\n"
        "\n"
        "# Repr contains both the parent extra_repr AND each child registered as a sub-module.\n"
        "r = repr(model)\n"
        "assert 'hidden_dim=8' in r, f'parent extra_repr must appear in repr, got:\\n{r}'\n"
        "assert '(fc1):' in r, f'child fc1 must appear under parent, got:\\n{r}'\n"
        "assert '(fc2):' in r, f'child fc2 must appear under parent, got:\\n{r}'\n"
        "assert 'Linear' in r, 'child Linear class name must appear'\n"
        "\n"
        "# Critical: child repr must be INDENTED (two-space leading per PyTorch's tree printer).\n"
        "lines = r.split('\\n')\n"
        "fc1_line = next(L for L in lines if '(fc1):' in L)\n"
        "assert fc1_line.startswith('  '), (\n"
        "    f'child line must be indented (PyTorch nested tree print), got: {fc1_line!r}'\n"
        ")\n"
        "\n"
        "# Forward must still work end-to-end (the module is a real MLP, not just a repr container).\n"
        "x = t.randn(4, 3)\n"
        "y = model(x)\n"
        "assert y.shape == (4, 2), f'forward shape: expected (4,2), got {tuple(y.shape)}'\n"
        "\n"
        "# Parameters must be registered (the child Linears should each contribute 2 tensors).\n"
        "params = list(model.parameters())\n"
        "assert len(params) == 4, f'expected 4 params (W1, b1, W2, b2), got {len(params)}'"
    ),
    "solution_body": (
        "def ex2_build_mlp(in_features: int, hidden_dim: int, out_features: int):\n"
        "    import torch.nn as nn\n"
        "    class MyMLP(nn.Module):\n"
        "        def __init__(self, in_features, hidden_dim, out_features):\n"
        "            super().__init__()\n"
        "            self.hidden_dim = hidden_dim\n"
        "            self.fc1 = nn.Linear(in_features, hidden_dim)\n"
        "            self.fc2 = nn.Linear(hidden_dim, out_features)\n"
        "        def extra_repr(self):\n"
        "            return f'hidden_dim={self.hidden_dim}'\n"
        "        def forward(self, x):\n"
        "            return self.fc2(t.relu(self.fc1(x)))\n"
        "    return MyMLP(in_features, hidden_dim, out_features)"
    ),
    "solution_notes": (
        "**The recursion is free — but `extra_repr` MUST return a string.** If "
        "you accidentally return a tensor or `None`, the default `nn.Module."
        "__repr__` will raise during printing. Always: `return f'k=v, ...'`.\n\n"
        "**Indentation comes from `nn.Module.__repr__`.** It calls `_addindent` "
        "on each child's repr and prefixes `(name): ` — you do NOT need to "
        "format the child yourself. Your `extra_repr` lives only on its own "
        "line; the children fall through to PyTorch's tree printer.\n\n"
        "**Difference from ex1.** ex1 verified the FLAT case (`bias=True/False` "
        "as a boolean, not a tensor). ex2 verifies the NESTED case — that "
        "`extra_repr` plays nicely with registered child modules in the "
        "recursive print."
    ),
    "extra_imports": [],
}


# ============================================================== 5: dataloader-batching
SPEC_DATALOADER = {
    "atom_id": "dataloader-batching",
    "subtopic": "PyTorch: DataLoader batching",
    "topic_folder": "prereqs_optimizer_internals",
    "atom_recap_md": (
        "## DataLoader — `drop_last` semantics\n"
        "\n"
        "If `len(dataset) % batch_size != 0`, the final batch is **smaller** than "
        "the others. The `drop_last` kwarg controls what to do with it:\n"
        "\n"
        "- `drop_last=False` (default): keep the partial batch. `len(loader) = "
        "ceil(N / B)`, last batch has `N % B` items.\n"
        "- `drop_last=True`: discard it. `len(loader) = N // B`, every batch is "
        "exactly `B` items.\n"
        "\n"
        "When does `drop_last=True` matter? **BatchNorm with very small final "
        "batches** (variance estimate is unstable on tiny batches), and **DDP** "
        "(uneven batch sizes across ranks cause hang). Otherwise keep the partial "
        "batch — losing a few samples per epoch is wasteful.\n"
        "\n"
        "The previous drill (ex1) wrapped a TensorDataset and iterated batches with "
        "the shuffle conventions. This drill targets **`drop_last`** specifically — "
        "counting batches in both modes and verifying batch-size invariants."
    ),
    "exercise_index": 2,
    "exercise_title": "drop_last semantics — count batches and verify size invariants",
    "slug": "drop-last-semantics-count-batches-and-verify-size-invariants",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dataloader", "drop-last", "partial-batch", "batch-count"],
    "kcs": ["dataloader-wraps-dataset", "dataloader-drop-last-discards-partial"],
    "lo": (
        "Apply `DataLoader(..., drop_last=True/False)` to a dataset whose size "
        "does not divide the batch size evenly, and count the resulting batches "
        "to confirm the partial-batch handling rule."
    ),
    "prompt_body": (
        "Implement `ex2_drop_last_counts(N, batch_size)`. Build a "
        "`TensorDataset(t.arange(N).float())` and wrap it TWICE — once with "
        "`drop_last=True`, once with `drop_last=False`. Iterate each loader fully, "
        "collecting the batch sizes into a list. Return:\n\n"
        "```python\n"
        "{\n"
        "  'with_drop':    [b0_size, b1_size, ...],  # drop_last=True\n"
        "  'without_drop': [b0_size, b1_size, ...],  # drop_last=False\n"
        "}\n"
        "```\n"
        "\n"
        "Set `shuffle=False` on both loaders so the iteration order is deterministic.\n\n"
        "**The verification rules** (test will check these):\n"
        "1. `len(with_drop) == N // batch_size`  (integer floor).\n"
        "2. Every entry of `with_drop` equals `batch_size`.\n"
        "3. `len(without_drop) == math.ceil(N / batch_size)`.\n"
        "4. All entries of `without_drop` except possibly the LAST equal `batch_size`.\n"
        "5. The last entry of `without_drop` equals `N % batch_size` if `N % batch_size != 0`, else `batch_size`.\n"
    ),
    "stub": (
        "def ex2_drop_last_counts(N: int, batch_size: int) -> dict:\n"
        '    """Return {with_drop: [...], without_drop: [...]} batch-size lists."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "from torch.utils.data import TensorDataset, DataLoader\n"
        "\n"
        "# Case 1 — non-dividing: 10 samples, batch_size 3.\n"
        "result = ex2_drop_last_counts(N=10, batch_size=3)\n"
        "assert set(result.keys()) == {'with_drop', 'without_drop'}, f'bad keys: {result.keys()}'\n"
        "wd, nd = result['with_drop'], result['without_drop']\n"
        "assert wd == [3, 3, 3], f'with_drop expected [3,3,3], got {wd}'\n"
        "assert nd == [3, 3, 3, 1], f'without_drop expected [3,3,3,1], got {nd}'\n"
        "\n"
        "# Case 2 — clean division: 8 samples, batch_size 4.\n"
        "r2 = ex2_drop_last_counts(N=8, batch_size=4)\n"
        "assert r2['with_drop'] == [4, 4], f'with_drop expected [4,4], got {r2[chr(39)+\"with_drop\"+chr(39)]}'\n"
        "assert r2['without_drop'] == [4, 4], f'without_drop expected [4,4], got {r2[chr(39)+\"without_drop\"+chr(39)]}'\n"
        "\n"
        "# Case 3 — single partial batch only: 3 samples, batch_size 5.\n"
        "r3 = ex2_drop_last_counts(N=3, batch_size=5)\n"
        "assert r3['with_drop'] == [], f'with_drop expected [], got {r3[chr(39)+\"with_drop\"+chr(39)]}'\n"
        "assert r3['without_drop'] == [3], f'without_drop expected [3], got {r3[chr(39)+\"without_drop\"+chr(39)]}'\n"
        "\n"
        "# Case 4 — confirm counts vs the closed-form rules on a bigger setting.\n"
        "N, B = 127, 16\n"
        "r4 = ex2_drop_last_counts(N=N, batch_size=B)\n"
        "assert len(r4['with_drop']) == N // B, f'with_drop count: expected {N//B}, got {len(r4[chr(39)+\"with_drop\"+chr(39)])}'\n"
        "assert len(r4['without_drop']) == math.ceil(N / B), f'without_drop count: expected {math.ceil(N/B)}, got {len(r4[chr(39)+\"without_drop\"+chr(39)])}'\n"
        "assert all(sz == B for sz in r4['with_drop']), 'with_drop entries must all equal batch_size'\n"
        "assert all(sz == B for sz in r4['without_drop'][:-1]), 'without_drop non-final entries must equal batch_size'\n"
        "tail = r4['without_drop'][-1]\n"
        "expected_tail = N % B if N % B != 0 else B\n"
        "assert tail == expected_tail, f'tail batch wrong: expected {expected_tail}, got {tail}'"
    ),
    "solution_body": (
        "def ex2_drop_last_counts(N: int, batch_size: int) -> dict:\n"
        "    from torch.utils.data import TensorDataset, DataLoader\n"
        "    ds = TensorDataset(t.arange(N).float())\n"
        "    with_drop = [b[0].shape[0] for b in DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)]\n"
        "    without_drop = [b[0].shape[0] for b in DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)]\n"
        "    return {'with_drop': with_drop, 'without_drop': without_drop}"
    ),
    "solution_notes": (
        "**When `drop_last=True` is mandatory.** Synchronous distributed training "
        "(DDP) hangs if different ranks see different batch counts. Setting "
        "`drop_last=True` aligns them. BatchNorm modules with `track_running_stats="
        "True` also misbehave on tiny final batches because the running variance "
        "estimate goes haywire on N=1 or N=2.\n\n"
        "**When it's wasteful.** Validation/test loops should set "
        "`drop_last=False` — you want every sample in your metrics.\n\n"
        "**Difference from ex1.** ex1 set up train (shuffle=True) and test "
        "(shuffle=False) loaders with default `drop_last`. ex2 zeroes in on the "
        "`drop_last` knob and verifies the floor vs ceil batch-count formula."
    ),
    "extra_imports": [],
}


# ============================================================== 6: optimizer-init-params-list
SPEC_OPT_INIT = {
    "atom_id": "optimizer-init-params-list",
    "subtopic": "PyTorch: Optimizer init",
    "topic_folder": "prereqs_training_loop",
    "atom_recap_md": (
        "## Param-groups — list-of-dicts edition\n"
        "\n"
        "Real PyTorch optimizers don't just take a flat list of params — they take "
        "a list of **param-group dicts**, each with its own hyperparameters:\n"
        "\n"
        "```python\n"
        "opt = torch.optim.SGD([\n"
        "    {'params': model.backbone.parameters(), 'lr': 1e-4},\n"
        "    {'params': model.head.parameters(),     'lr': 1e-2},\n"
        "], momentum=0.9)\n"
        "```\n"
        "\n"
        "Each dict must contain `'params'` (an iterable). Per-group hyperparameters "
        "(`lr`, `weight_decay`, etc.) override the optimizer-level default; missing "
        "ones fall back to the constructor kwargs. Internally the optimizer "
        "materializes each group's `params` into a list (same rule as ex1).\n"
        "\n"
        "The previous drill (ex1) materialized a single generator into a list. "
        "This drill exercises the **per-group-list-of-dicts** layout — building a "
        "two-group optimizer that applies different learning rates to different "
        "parameter subsets."
    ),
    "exercise_index": 2,
    "exercise_title": "param-groups list-of-dicts with different per-group learning rates",
    "slug": "param-groups-list-of-dicts-different-lrs",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["param-groups", "list-of-dicts", "per-group-lr", "hand-rolled-optimizer"],
    "kcs": ["optimizer-init-list-vs-generator", "optimizer-param-groups-list-of-dicts"],
    "lo": (
        "Apply the param-groups pattern (list-of-dicts, each carrying its own `lr`) "
        "by building a hand-rolled SGD optimizer whose `.step()` uses the per-group "
        "learning rate instead of a single optimizer-wide value."
    ),
    "prompt_body": (
        "Implement `Ex2GroupOptimizer` — a hand-rolled SGD optimizer that accepts a "
        "list-of-dicts param-group spec.\n\n"
        "**`__init__(self, param_groups)`:**\n"
        "1. `param_groups` is a list of dicts, each shaped `{'params': iterable, 'lr': float}`.\n"
        "2. Materialize each group's `'params'` into a list (same rule as ex1, but per-group).\n"
        "3. Store `self.param_groups` as a list of dicts; each dict must have a `'params'` LIST and an `'lr'` float.\n\n"
        "**`@t.no_grad()` `.step(self)`:**\n"
        "- For each group `g` in `self.param_groups`, iterate `g['params']` and apply "
        "`p.data -= g['lr'] * p.grad` for every param with non-None `.grad`.\n"
        "- Each group uses its OWN `'lr'` — that's the whole point.\n\n"
        "**`.zero_grad(self)`:**\n"
        "- For every param in every group, set `p.grad = None`.\n\n"
        "The test sets up two groups (backbone with `lr=0.01`, head with `lr=1.0`) "
        "and verifies the head's parameters move 100x further than the backbone's "
        "for the same gradient magnitude."
    ),
    "stub": (
        "class Ex2GroupOptimizer:\n"
        '    """Hand-rolled SGD with per-group learning rates."""\n'
        "\n"
        "    def __init__(self, param_groups):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    @t.no_grad()\n"
        "    def step(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def zero_grad(self):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build two tiny linear layers — call one 'backbone', one 'head'.\n"
        "backbone = nn.Linear(2, 2, bias=False)\n"
        "head = nn.Linear(2, 1, bias=False)\n"
        "with t.no_grad():\n"
        "    backbone.weight.fill_(0.0)\n"
        "    head.weight.fill_(0.0)\n"
        "\n"
        "# Build optimizer with two param groups, very different lrs.\n"
        "opt = Ex2GroupOptimizer([\n"
        "    {'params': backbone.parameters(), 'lr': 0.01},\n"
        "    {'params': head.parameters(),     'lr': 1.0},\n"
        "])\n"
        "\n"
        "# Each group's params must be a LIST (materialized).\n"
        "assert isinstance(opt.param_groups, list), 'param_groups must be a list'\n"
        "assert len(opt.param_groups) == 2, f'expected 2 groups, got {len(opt.param_groups)}'\n"
        "for g in opt.param_groups:\n"
        "    assert isinstance(g, dict)\n"
        "    assert 'params' in g and 'lr' in g\n"
        "    assert isinstance(g['params'], list), f'group params must be a list (materialized), got {type(g[chr(39)+\"params\"+chr(39)])}'\n"
        "\n"
        "# Assign identical gradients to backbone and head weights.\n"
        "backbone.weight.grad = t.ones_like(backbone.weight)\n"
        "head.weight.grad = t.ones_like(head.weight)\n"
        "\n"
        "opt.step()\n"
        "\n"
        "# Backbone moved by -0.01, head moved by -1.0 — exact per-group lr behavior.\n"
        "assert t.allclose(backbone.weight, t.full_like(backbone.weight, -0.01), atol=1e-7), (\n"
        "    f'backbone should be -0.01, got {backbone.weight}'\n"
        ")\n"
        "assert t.allclose(head.weight, t.full_like(head.weight, -1.0), atol=1e-7), (\n"
        "    f'head should be -1.0, got {head.weight}'\n"
        ")\n"
        "\n"
        "# zero_grad clears every param across every group.\n"
        "opt.zero_grad()\n"
        "for g in opt.param_groups:\n"
        "    for p in g['params']:\n"
        "        assert p.grad is None, f'zero_grad must set grad=None across groups, got {p.grad}'\n"
        "\n"
        "# Pass a generator into one group — must still survive multiple .step() calls.\n"
        "m = nn.Linear(3, 1, bias=False)\n"
        "with t.no_grad(): m.weight.fill_(0.0)\n"
        "import types\n"
        "g_in = m.parameters()\n"
        "assert isinstance(g_in, types.GeneratorType), 'precondition'\n"
        "opt2 = Ex2GroupOptimizer([{'params': g_in, 'lr': 0.5}])\n"
        "m.weight.grad = t.ones_like(m.weight)\n"
        "opt2.step()\n"
        "m.weight.grad = t.ones_like(m.weight)\n"
        "opt2.step()  # second call MUST also work — group's params must have been materialized.\n"
        "assert t.allclose(m.weight, t.full_like(m.weight, -1.0), atol=1e-7), (\n"
        "    f'after two steps with lr=0.5, weight should be -1.0, got {m.weight}'\n"
        ")"
    ),
    "solution_body": (
        "class Ex2GroupOptimizer:\n"
        "    def __init__(self, param_groups):\n"
        "        self.param_groups = []\n"
        "        for g in param_groups:\n"
        "            self.param_groups.append({\n"
        "                'params': list(g['params']),\n"
        "                'lr': g['lr'],\n"
        "            })\n"
        "\n"
        "    @t.no_grad()\n"
        "    def step(self):\n"
        "        for g in self.param_groups:\n"
        "            lr = g['lr']\n"
        "            for p in g['params']:\n"
        "                if p.grad is not None:\n"
        "                    p.data -= lr * p.grad\n"
        "\n"
        "    def zero_grad(self):\n"
        "        for g in self.param_groups:\n"
        "            for p in g['params']:\n"
        "                p.grad = None"
    ),
    "solution_notes": (
        "**Why list-of-dicts instead of a flat list.** Fine-tuning regimes "
        "almost always want a smaller learning rate on the pre-trained backbone "
        "and a larger one on the freshly initialized head. Param groups make "
        "this a 5-line setup instead of two separate optimizers (which would "
        "complicate scheduler logic and checkpoint serialization).\n\n"
        "**Per-group materialization.** The same generator-exhaustion bug from "
        "ex1 applies to EACH group independently — if you write `g['params']` "
        "without wrapping in `list(...)`, the first `.step()` consumes the "
        "generator and subsequent steps silently skip that group.\n\n"
        "**Difference from ex1.** ex1 handled the **single-list** case. ex2 "
        "extends to the **list-of-dicts** case with heterogeneous per-group "
        "hyperparameters — the canonical fine-tuning pattern."
    ),
    "extra_imports": [],
}


# ============================================================== 7: zero-grad-set-none
SPEC_ZERO_GRAD = {
    "atom_id": "zero-grad-set-none",
    "subtopic": "PyTorch: zero_grad",
    "topic_folder": "prereqs_training_loop",
    "atom_recap_md": (
        "## zero_grad — when does it go in the loop?\n"
        "\n"
        "The canonical training step has FOUR statements in a fixed order:\n"
        "\n"
        "```python\n"
        "for xb, yb in loader:\n"
        "    loss = loss_fn(model(xb), yb)\n"
        "    loss.backward()       # 1. compute gradients\n"
        "    optimizer.step()      # 2. apply update\n"
        "    optimizer.zero_grad() # 3. wipe grads for the NEXT step\n"
        "```\n"
        "\n"
        "Equivalently, `zero_grad()` can be the FIRST statement of the NEXT "
        "iteration — but it MUST sit between `step` of iteration N and `backward` "
        "of iteration N+1. Putting it BETWEEN `backward` and `step` of the same "
        "iteration is the **classic silent bug**: gradients are wiped before the "
        "optimizer can read them, so `.step()` becomes a no-op and the model "
        "never learns.\n"
        "\n"
        "The previous drill (ex1) implemented the *body* of `zero_grad` "
        "(`p.grad = None` for each param). This drill targets the **ORDERING** — "
        "given a buggy training loop, identify which placement is correct and fix it."
    ),
    "exercise_index": 2,
    "exercise_title": "diagnose a training loop where zero_grad runs BEFORE step",
    "slug": "diagnose-training-loop-zero-grad-before-step",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["training-loop", "zero-grad", "ordering", "silent-bug", "diagnose"],
    "kcs": ["zero-grad-set-to-none-semantics", "zero-grad-must-follow-step"],
    "lo": (
        "Analyze the relative ordering of `backward`, `step`, and `zero_grad` in a "
        "training loop and fix a variant in which `zero_grad` was placed BEFORE "
        "`step` so the optimizer would have nothing to read."
    ),
    "prompt_body": (
        "Below is a buggy training step. The author put `optimizer.zero_grad()` "
        "between `backward()` and `step()` — so the gradients computed by `backward` "
        "are wiped to None BEFORE `step` runs, and the parameters never get updated. "
        "The loss never decreases.\n\n"
        "Implement `ex2_fixed_step(model, opt, x, y, loss_fn)` — ONE call to the "
        "fixed training step. The fix is to set `p.grad = None` AFTER `step()`, not "
        "between `backward()` and `step()`. The function returns the post-step loss "
        "value (the float scalar from before the update — i.e. the value of `loss` "
        "that the gradients were computed against).\n\n"
        "**Required order:**\n"
        "1. `pred = model(x)`\n"
        "2. `loss = loss_fn(pred, y)`\n"
        "3. `loss.backward()`\n"
        "4. `opt.step()`\n"
        "5. for `p in model.parameters(): p.grad = None`  ← AFTER step\n"
        "6. return `loss.item()`\n\n"
        "**Why the bug is silent.** No exception, no warning. The model's loss "
        "just stays flat across epochs. Detection requires logging the loss "
        "trajectory — if it's literally constant when it should be falling, this "
        "is the first place to look.\n\n"
        "The test runs the fixed step many times on a quadratic loss and confirms "
        "the loss decreases monotonically — then runs the buggy version (provided "
        "below) and confirms the loss is constant."
    ),
    "stub": (
        "def ex2_buggy_step(model, opt, x, y, loss_fn):\n"
        "    # This is the BUG — zero_grad runs BEFORE step.\n"
        "    pred = model(x)\n"
        "    loss = loss_fn(pred, y)\n"
        "    loss.backward()\n"
        "    for p in model.parameters():\n"
        "        p.grad = None         # <-- wipes grads too early\n"
        "    opt.step()                  # <-- no-op: nothing to read\n"
        "    return loss.item()\n"
        "\n"
        "\n"
        "def ex2_fixed_step(model, opt, x, y, loss_fn) -> float:\n"
        '    """Fixed training step: backward → step → zero_grad → return loss."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "\n"
        "# Setup — fit a tiny linear model on a 1-D regression task.\n"
        "t.manual_seed(0)\n"
        "x = t.linspace(-1, 1, 32).unsqueeze(1)\n"
        "y = 3 * x + 1   # ground-truth line\n"
        "\n"
        "# Buggy run — loss should NOT decrease.\n"
        "model_buggy = nn.Linear(1, 1)\n"
        "opt_buggy = t.optim.SGD(model_buggy.parameters(), lr=0.1)\n"
        "losses_buggy = [ex2_buggy_step(model_buggy, opt_buggy, x, y, F.mse_loss) for _ in range(20)]\n"
        "assert losses_buggy[-1] >= losses_buggy[0] - 1e-6, (\n"
        "    f'buggy loop should NOT improve (zero_grad wipes grads before step); '\n"
        "    f'got {losses_buggy[0]:.4f} → {losses_buggy[-1]:.4f}'\n"
        ")\n"
        "\n"
        "# Fixed run — loss must decrease monotonically (strictly across the 20-step horizon).\n"
        "model_fixed = nn.Linear(1, 1)\n"
        "opt_fixed = t.optim.SGD(model_fixed.parameters(), lr=0.1)\n"
        "losses_fixed = [ex2_fixed_step(model_fixed, opt_fixed, x, y, F.mse_loss) for _ in range(20)]\n"
        "assert losses_fixed[-1] < losses_fixed[0] * 0.5, (\n"
        "    f'fixed loop should at least halve the loss in 20 steps; '\n"
        "    f'got {losses_fixed[0]:.4f} → {losses_fixed[-1]:.4f}'\n"
        ")\n"
        "# After the loop, grads must be None (zero_grad ran AFTER each step).\n"
        "for p in model_fixed.parameters():\n"
        "    assert p.grad is None, f'after the loop, grads must be None (post-step zero_grad), got {p.grad}'\n"
        "\n"
        "# Returned value matches the loss BEFORE the step.\n"
        "model_check = nn.Linear(1, 1)\n"
        "opt_check = t.optim.SGD(model_check.parameters(), lr=0.1)\n"
        "manual_pred = model_check(x)\n"
        "manual_loss = F.mse_loss(manual_pred, y).item()\n"
        "returned = ex2_fixed_step(model_check, opt_check, x, y, F.mse_loss)\n"
        "assert abs(returned - manual_loss) < 1e-6, (\n"
        "    f'returned loss should be the pre-step loss; got {returned:.6f}, want {manual_loss:.6f}'\n"
        ")"
    ),
    "solution_body": (
        "def ex2_buggy_step(model, opt, x, y, loss_fn):\n"
        "    # The bug — kept as a reference foil.\n"
        "    pred = model(x)\n"
        "    loss = loss_fn(pred, y)\n"
        "    loss.backward()\n"
        "    for p in model.parameters():\n"
        "        p.grad = None\n"
        "    opt.step()\n"
        "    return loss.item()\n"
        "\n"
        "\n"
        "def ex2_fixed_step(model, opt, x, y, loss_fn) -> float:\n"
        "    pred = model(x)\n"
        "    loss = loss_fn(pred, y)\n"
        "    loss.backward()\n"
        "    opt.step()\n"
        "    for p in model.parameters():\n"
        "        p.grad = None\n"
        "    return loss.item()"
    ),
    "solution_notes": (
        "**The standard tripwire.** Every PyTorch tutorial places `zero_grad` "
        "at one of two correct spots: (a) the LAST line of the training step "
        "(after `step`), or (b) the FIRST line of the next iteration (before "
        "`backward`). Anywhere ELSE is wrong. The 'between backward and step' "
        "placement looks innocent — same set of three function names, just "
        "reordered — but it silently neuters the optimizer.\n\n"
        "**Why it's silent.** `step` reads `param.grad`; if it's `None`, it "
        "does nothing for that param (no exception). With `set_to_none=True` "
        "this is the explicit contract — `None` means 'no gradient yet, skip.'\n\n"
        "**Difference from ex1.** ex1 implemented the BODY of `zero_grad` "
        "(set each `.grad` to `None`). ex2 targets the ORDERING — diagnosing "
        "and fixing a training loop where `zero_grad` ran at the wrong moment."
    ),
    "extra_imports": [],
}


# ============================================================== 8: param-grad-access
SPEC_PARAM_GRAD = {
    "atom_id": "param-grad-access",
    "subtopic": "PyTorch: param.grad access",
    "topic_folder": "prereqs_backprop",
    "atom_recap_md": (
        "## param.grad — gradient norm logging\n"
        "\n"
        "Per-parameter gradient norms are the first-line training diagnostic. They "
        "let you spot:\n"
        "\n"
        "- **vanishing grads** (norm → 0 in deep layers — model not learning)\n"
        "- **exploding grads** (norm → ∞ — need clipping / smaller lr)\n"
        "- **dead layers** (grad is `None` — that subgraph wasn't reached)\n"
        "\n"
        "```python\n"
        "for name, p in model.named_parameters():\n"
        "    if p.grad is None:\n"
        "        print(f'{name}: NO GRAD')\n"
        "        continue\n"
        "    print(f'{name}: {p.grad.norm(p=2).item():.4e}')\n"
        "```\n"
        "\n"
        "The previous drill (ex1) applied the canonical SGD step with the None-grad "
        "skip rule. This drill targets the **diagnostic** half: computing the L2 "
        "norm of each parameter's gradient (skipping None) and returning a dict "
        "keyed by param NAME — exactly the data structure logged to TensorBoard."
    ),
    "exercise_index": 2,
    "exercise_title": "compute per-parameter gradient L2 norms with None-grad skip",
    "slug": "per-parameter-grad-l2-norm-with-none-skip",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["grad-norm", "named-parameters", "diagnostic", "logging"],
    "kcs": ["param-grad-access", "param-grad-none-guard-skip"],
    "lo": (
        "Apply the `named_parameters()` + `.grad.norm()` pattern to build a "
        "diagnostic dict mapping parameter name → L2 grad norm, omitting params "
        "whose `.grad` is `None`."
    ),
    "prompt_body": (
        "Implement `ex2_grad_norms(model)`. Return a `dict[str, float]` mapping "
        "each parameter's NAME (from `model.named_parameters()`) to "
        "`p.grad.norm(p=2).item()`. Parameters whose `.grad is None` must be "
        "**OMITTED** from the dict (not present as None, not present as 0.0 — "
        "absent entirely).\n\n"
        "**Required pattern.**\n"
        "```python\n"
        "out = {}\n"
        "for name, p in model.named_parameters():\n"
        "    if p.grad is None:\n"
        "        continue\n"
        "    out[name] = p.grad.norm(p=2).item()\n"
        "return out\n"
        "```\n"
        "\n"
        "Inputs:\n"
        "- `model`: any `nn.Module` whose `.named_parameters()` yields `(str, Parameter)` pairs.\n"
        "\n"
        "Output:\n"
        "- `dict[str, float]` — names are stable across calls (PyTorch names them "
        "by attribute path, e.g. `'fc1.weight'`, `'fc1.bias'`).\n\n"
        "**Why per-parameter not flat-concat.** A flat concat (`.norm()` on every "
        "grad joined together) hides per-layer behavior. The per-param dict is "
        "what lets you spot the dead layer."
    ),
    "stub": (
        "def ex2_grad_norms(model) -> dict:\n"
        '    """{param-name: L2 grad norm} dict, skipping params whose .grad is None."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a tiny model with named parameters spanning a Linear + a free Parameter.\n"
        "class Mini(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(3, 2)\n"
        "        self.bias_extra = nn.Parameter(t.zeros(2))\n"
        "    def forward(self, x):\n"
        "        return self.fc(x) + self.bias_extra\n"
        "\n"
        "model = Mini()\n"
        "x = t.tensor([[1.0, 1.0, 1.0]])\n"
        "y = t.tensor([[3.0, 5.0]])\n"
        "loss = ((model(x) - y) ** 2).sum()\n"
        "loss.backward()\n"
        "\n"
        "norms = ex2_grad_norms(model)\n"
        "assert isinstance(norms, dict), f'must return dict, got {type(norms)}'\n"
        "# Every param participated → all three names must be in the dict.\n"
        "assert set(norms.keys()) == {'fc.weight', 'fc.bias', 'bias_extra'}, (\n"
        "    f'expected keys {{fc.weight, fc.bias, bias_extra}}, got {set(norms.keys())}'\n"
        ")\n"
        "# Values are floats and match the manual computation.\n"
        "for name, p in model.named_parameters():\n"
        "    expected = p.grad.norm(p=2).item()\n"
        "    got = norms[name]\n"
        "    assert isinstance(got, float), f'{name}: value must be Python float, got {type(got)}'\n"
        "    assert abs(got - expected) < 1e-6, f'{name}: got {got}, expected {expected}'\n"
        "\n"
        "# All norms strictly positive (every param had a non-trivial gradient here).\n"
        "for name, n in norms.items():\n"
        "    assert n > 0, f'{name}: norm should be > 0 after non-trivial backward, got {n}'\n"
        "\n"
        "# --- None-grad skip behavior ---\n"
        "# Fresh model, NO backward called — every param has grad=None.\n"
        "fresh = Mini()\n"
        "empty = ex2_grad_norms(fresh)\n"
        "assert empty == {}, f'fresh model with no backward must yield empty dict, got {empty}'\n"
        "\n"
        "# Mixed — manually populate ONE param's grad, leave the others None.\n"
        "mixed = Mini()\n"
        "mixed.fc.weight.grad = t.ones_like(mixed.fc.weight)  # norm = sqrt(6)\n"
        "got = ex2_grad_norms(mixed)\n"
        "assert set(got.keys()) == {'fc.weight'}, f'only fc.weight should appear; got {set(got.keys())}'\n"
        "assert abs(got['fc.weight'] - (6.0 ** 0.5)) < 1e-5, f'norm of all-ones (3,2) tensor is sqrt(6); got {got[chr(39)+\"fc.weight\"+chr(39)]}'"
    ),
    "solution_body": (
        "def ex2_grad_norms(model) -> dict:\n"
        "    out = {}\n"
        "    for name, p in model.named_parameters():\n"
        "        if p.grad is None:\n"
        "            continue\n"
        "        out[name] = p.grad.norm(p=2).item()\n"
        "    return out"
    ),
    "solution_notes": (
        "**Why `named_parameters()` not `parameters()`.** The names give you "
        "the path through the module tree (`fc1.weight`, `block.0.attn.qkv`, "
        "etc.) — required for logging to TensorBoard / Weights & Biases per-"
        "parameter charts. `parameters()` yields tensors only, with no name "
        "attached.\n\n"
        "**The skip vs the zero.** Omitting a param entirely (vs storing `0.0`) "
        "is the right call here: it tells the consumer 'this param didn't get "
        "a gradient', not 'this param got a gradient of magnitude exactly zero' "
        "— two semantically different conditions that look identical in a bar "
        "chart if you collapse them.\n\n"
        "**Difference from ex1.** ex1 PERFORMED the SGD step (`p.data -= lr * "
        "p.grad`) with the None-grad skip. ex2 only READS the gradient (`p.grad"
        ".norm()`) with the same skip rule — the diagnostic counterpart of the "
        "update step."
    ),
    "extra_imports": [],
}


SPECS = [
    SPEC_SLICE_VIEW,
    SPEC_BOOL_MASK,
    SPEC_RELU,
    SPEC_EXTRA_REPR,
    SPEC_DATALOADER,
    SPEC_OPT_INIT,
    SPEC_ZERO_GRAD,
    SPEC_PARAM_GRAD,
]


for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
