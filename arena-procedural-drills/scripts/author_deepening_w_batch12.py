#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 12, group W).

Atoms (6 prereqs_misc_cleanup + 2 prereqs_numerical_modules):
    - leaf-tensor-condition     (ex2: 3-way classify leaf-trainable / non-leaf / leaf-frozen)
    - linspace-out-param        (ex2: contrast linspace(out=) with .copy_(linspace(...)))
    - optimizer-repr-string     (ex2: multi-param-group __repr__ matching PyTorch's format)
    - rmul-scalar-tensor-mix    (ex2: __sub__/__rsub__ asymmetry — rsub flips order)
    - tensor-reshape-view       (ex2: post-transpose .view() raises, .reshape() works)
    - trainer-subclass-extend   (ex2: 3-level MRO chain — each subclass extends via super())
    - conditional-hparam-branch (ex2: conditional Dropout — p>0 → nn.Dropout, p==0 → identity branch)
    - device-consistent-construct (ex2: helper that infers device+dtype from an existing param)

Each ex2 hits a DISTINCT facet from ex1. ONE LO + ONE Bloom + <=2 KCs per drill.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_MISC = "prereqs_misc_cleanup"
TOPIC_NUM = "prereqs_numerical_modules"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_LEAF_THREEWAY = (
    "## Leaf condition — the full 3-way classification\n"
    "\n"
    "Ex1 split tensors into LEAF vs NON-LEAF using `grad_fn is None`. Real "
    "autograd graphs have a THIRD category that matters: leaves WITHOUT "
    "`requires_grad`. They satisfy `grad_fn is None` but `.grad` never fills.\n"
    "\n"
    "| Category               | `grad_fn`     | `requires_grad` | `.grad` after backward |\n"
    "|------------------------|---------------|-----------------|------------------------|\n"
    "| leaf-trainable         | `None`        | `True`          | populated              |\n"
    "| non-leaf (interior)    | not-`None`    | `True`          | None (unless retained) |\n"
    "| leaf-frozen (constant) | `None`        | `False`         | None — no grad needed  |\n"
    "\n"
    "The same `(grad_fn, requires_grad)` pair determines all three. ARENA's "
    "MiniTensor uses the same 2-field condition (`recipe`, `requires_grad`).\n"
    "\n"
    "**Why the third category matters.** Input data `x` (from a dataloader) "
    "and frozen-backbone params during fine-tuning both fall into the "
    "leaf-frozen bucket. If you accidentally set `requires_grad=True` on the "
    "data, the graph balloons and autograd keeps activations alive for "
    "backprop you never call."
)

RECAP_LINSPACE_OUT_VS_COPY = (
    "## `linspace(out=)` vs `.copy_(linspace(...))` — same data, different alloc\n"
    "\n"
    "Ex1 used `linspace(out=buf)` to fill a pre-allocated buffer. The "
    "deepening move contrasts that against the obvious alternative — "
    "`buf.copy_(t.linspace(...))` — and shows they produce the SAME numeric "
    "result but DIFFERENT allocation behaviour:\n"
    "\n"
    "```python\n"
    "# Option A: out= fills buf directly, no temp.\n"
    "t.linspace(0., 1., 100, out=buf)\n"
    "\n"
    "# Option B: linspace returns a new tensor, then copy_ overwrites buf.\n"
    "# Thread dtype=buf.dtype so the temp doesn't get cast on copy.\n"
    "buf.copy_(t.linspace(0., 1., 100, dtype=buf.dtype))\n"
    "```\n"
    "\n"
    "**Both preserve `buf.data_ptr()`.** `out=` writes in place; `copy_` "
    "also writes in place into the existing storage. So downstream pointers "
    "to `buf` stay valid in either case.\n"
    "\n"
    "**Option A allocates 0 extra tensors. Option B allocates 1 (the "
    "linspace return value), then discards it.** In a tight loop, the temp "
    "is GC-able but you pay allocator churn. `out=` is the zero-temp idiom."
)

RECAP_OPT_REPR_MULTIGROUP = (
    "## Multi-param-group `__repr__` — PyTorch's actual format\n"
    "\n"
    "Ex1 built a single-group repr. Real `torch.optim.SGD` supports MULTIPLE "
    "param groups (different lr/momentum per group — common for finetuning "
    "with head-vs-backbone lrs). The repr format PyTorch uses:\n"
    "\n"
    "```\n"
    "SGD (\n"
    "Parameter Group 0\n"
    "    lr: 0.001\n"
    "    momentum: 0.9\n"
    "Parameter Group 1\n"
    "    lr: 0.0001\n"
    "    momentum: 0.0\n"
    ")\n"
    "```\n"
    "\n"
    "**Keys are sorted alphabetically within each group.** PyTorch enforces "
    "this for stable diffs across versions.\n"
    "\n"
    "**Header = class name + space + `(`.** Footer = `)` on its own line. "
    "Each `Parameter Group N` header is followed by `    key: value` lines "
    "(4-space indent).\n"
    "\n"
    "**Why this matters.** When a finetune script prints `optim`, the "
    "instructor needs to see `lr=0.001` for the head and `lr=0.0001` for "
    "the backbone in ONE glance — a flat single-group repr hides the bug."
)

RECAP_RSUB_ASYMMETRY = (
    "## `__sub__` / `__rsub__` — the asymmetry that catches everyone\n"
    "\n"
    "Ex1 implemented `__mul__` / `__rmul__` — symmetric because multiplication "
    "commutes (`2 * t == t * 2`). Subtraction does NOT commute, and that's "
    "where `__rsub__` gets misimplemented:\n"
    "\n"
    "```python\n"
    "t.__sub__(other)   # returns self - other  (self on the left)\n"
    "t.__rsub__(other)  # returns other - self  (self on the RIGHT — flipped!)\n"
    "```\n"
    "\n"
    "**Why Python calls `__rsub__`.** When you write `5 - my_tensor`, Python "
    "first tries `(5).__sub__(my_tensor)` — `int` doesn't know about your "
    "tensor class, returns `NotImplemented`. Python then tries the REFLECTED "
    "method: `my_tensor.__rsub__(5)`. The convention is that `__rsub__` "
    "must compute `other - self`, NOT `self - other`.\n"
    "\n"
    "**The trap.** Naive implementations write `return self.value - other` "
    "for `__rsub__`. That makes `5 - t` equal `t - 5` — exactly backwards.\n"
    "\n"
    "ARENA's manual-autograd Tensor wrapper hits this trap in chap-0 because "
    "MiniTensor needs to mimic torch.Tensor's full op set."
)

RECAP_VIEW_AFTER_TRANSPOSE = (
    "## `view()` requires contiguous — `reshape()` falls back to a copy\n"
    "\n"
    "Ex1 picked between `view` and `reshape` based on whether the input was "
    "already contiguous. The deepening drill exercises the COMMON failure: "
    "after `.transpose()`, the tensor has the right logical shape but the "
    "underlying storage is in the wrong order.\n"
    "\n"
    "```python\n"
    "x = t.arange(24).reshape(2, 3, 4)\n"
    "y = x.transpose(0, 2)        # shape (4, 3, 2) but NOT contiguous\n"
    "y.view(24)                    # RuntimeError: view size is not compatible\n"
    "y.reshape(24)                 # OK — copies under the hood\n"
    "y.contiguous().view(24)       # OK — explicit copy then view\n"
    "```\n"
    "\n"
    "**Why `view` raises.** `view` requires that the requested shape can be "
    "satisfied by re-striding the EXISTING storage. After transpose, "
    "elements that are logically adjacent are not physically adjacent — "
    "no stride choice works. PyTorch refuses to silently copy and forces "
    "you to either call `.contiguous()` first or switch to `.reshape()`.\n"
    "\n"
    "**`reshape` decides for you.** If a `view` is possible, `reshape` "
    "returns one (no copy, same `data_ptr()`). If not, it falls back to "
    "`.contiguous().view(...)` (one copy, new `data_ptr()`). This is the "
    "safe default — use `view` only when you've already proven contiguity."
)

RECAP_TRAINER_MRO_CHAIN = (
    "## 3-level subclass chain — `super()._step` MRO walking\n"
    "\n"
    "Ex1 had ONE subclass extending ONE base. The deepening move stacks a "
    "THIRD level: `BaseTrainer ← LoggingTrainer ← FrozenLoggingTrainer`. "
    "Each level adds its own metric by calling `super()._step(batch)` first "
    "and then mutating the returned dict.\n"
    "\n"
    "```python\n"
    "class BaseTrainer:\n"
    "    def _step(self, batch):\n"
    "        return {'loss': float(batch[0].abs().sum())}\n"
    "\n"
    "class LoggingTrainer(BaseTrainer):\n"
    "    def _step(self, batch):\n"
    "        d = super()._step(batch)\n"
    "        d['input_mag'] = float(batch[0].abs().mean())\n"
    "        return d\n"
    "\n"
    "class FrozenLoggingTrainer(LoggingTrainer):\n"
    "    def _step(self, batch):\n"
    "        d = super()._step(batch)  # walks MRO → LoggingTrainer._step\n"
    "        d['grad_norm'] = 0.0       # head is frozen this run\n"
    "        return d\n"
    "```\n"
    "\n"
    "**Why MRO matters.** `super()._step(batch)` does NOT mean "
    "`BaseTrainer._step(batch)`. Python walks the method resolution order "
    "(`type(self).__mro__`) and calls the NEXT class in the chain. From "
    "`FrozenLoggingTrainer.__mro__`, the next class is `LoggingTrainer`, "
    "which itself defers to `BaseTrainer` via its own `super()`. The chain "
    "composes — each level's metric ends up in the final dict."
)

RECAP_COND_DROPOUT = (
    "## Conditional Dropout branch — `p > 0` builds a real module, `p == 0` skips\n"
    "\n"
    "Ex1 gated `bias=True/False` on `nn.Linear`. The deepening move handles "
    "the same conditional pattern for a regularization submodule: when "
    "`dropout > 0`, build an `nn.Dropout(p=dropout)` and apply it; when "
    "`dropout == 0`, skip the module entirely (don't even register it).\n"
    "\n"
    "```python\n"
    "class Block(nn.Module):\n"
    "    def __init__(self, d_in, d_out, dropout: float):\n"
    "        super().__init__()\n"
    "        self.linear = nn.Linear(d_in, d_out)\n"
    "        if dropout > 0:\n"
    "            self.drop = nn.Dropout(p=dropout)\n"
    "        # else: no self.drop attribute at all\n"
    "\n"
    "    def forward(self, x):\n"
    "        x = self.linear(x)\n"
    "        if hasattr(self, 'drop'):\n"
    "            x = self.drop(x)\n"
    "        return x\n"
    "```\n"
    "\n"
    "**Why `hasattr` over `if self.drop is not None`.** Setting "
    "`self.drop = None` works in plain Python but registers `None` against "
    "the `nn.Module` child machinery in some versions and surprises "
    "`named_modules()`. Either skip the attribute entirely (cleanest) or "
    "use `nn.Identity()` as a no-op placeholder — both keep the forward "
    "path branch-free at the cost of one extra module slot.\n"
    "\n"
    "**Same `nn.Linear` either way.** Dropout has zero parameters, so "
    "`len(list(model.parameters()))` is identical regardless of branch — "
    "the only difference is whether `model.modules()` includes a Dropout."
)

RECAP_DEVICE_INFER = (
    "## Inferring device + dtype from an existing param — `next(parameters())`\n"
    "\n"
    "Ex1 read `device` + `dtype` directly off the input tensor `x`. The "
    "deepening move handles the case where you need to allocate scratch "
    "BEFORE you have an input — e.g. inside `__init__` after the module "
    "has been `.to(device)`-moved. The canonical idiom:\n"
    "\n"
    "```python\n"
    "def _ref_param(self):\n"
    "    return next(self.parameters())  # first registered Parameter\n"
    "\n"
    "def make_scratch(self, shape):\n"
    "    ref = self._ref_param()\n"
    "    return t.zeros(shape, device=ref.device, dtype=ref.dtype)\n"
    "```\n"
    "\n"
    "**Why `next(self.parameters())` is the standard trick.** All registered "
    "Parameters live on the same device after `.to(device)` — picking the "
    "first one is enough. This is what `torch.nn.Transformer` and HF's "
    "modeling code use internally when they need to materialize a mask or "
    "positional buffer without an input tensor in hand.\n"
    "\n"
    "**Failure mode if the module has no parameters.** `next()` raises "
    "`StopIteration`. Guard with a `try/except` or `try` `next(self.buffers())` "
    "as fallback. The drill explicitly tests both the happy path AND the "
    "no-params guard."
)


# ---------------------------------------------------------------------------
# SPEC 1 — leaf-tensor-condition ex2
# ---------------------------------------------------------------------------

SPEC_LEAF = {
    "atom_id": "leaf-tensor-condition",
    "subtopic": "Backprop: leaf tensor condition",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_LEAF_THREEWAY,
    "exercise_index": 2,
    "exercise_title": "classify tensors three ways: leaf-trainable, non-leaf, and leaf-frozen",
    "slug": "classify-tensors-three-ways-leaf-trainable-nonleaf-leaf-frozen",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["leaf", "requires-grad", "grad-fn", "autograd"],
    "kcs": [
        "leaf-iff-no-grad-fn",
        "trainable-iff-requires-grad",
    ],
    "lo": (
        "Analyze each tensor against the (grad_fn, requires_grad) pair to "
        "classify it as 'leaf-trainable', 'non-leaf', or 'leaf-frozen' — the "
        "full three-way split PyTorch's autograd uses."
    ),
    "prompt_body": (
        "Implement `ex2_classify_three_ways(tensors)`. Take a list of "
        "`Tensor`s and return a list of strings, one per tensor:\n\n"
        "- `'leaf-trainable'` if `grad_fn is None` AND `requires_grad` is True\n"
        "- `'non-leaf'` if `grad_fn is not None`\n"
        "- `'leaf-frozen'` if `grad_fn is None` AND `requires_grad` is False\n\n"
        "Inputs:\n"
        "- `tensors`: `list[Tensor]`.\n\n"
        "Output: `list[str]` of the same length, drawn from the three "
        "labels above.\n\n"
        "Do NOT use `.is_leaf` — implement from `grad_fn` + `requires_grad` "
        "from first principles. (`.is_leaf` returns True for BOTH "
        "leaf-trainable and leaf-frozen — it can't distinguish them on its "
        "own, which is why the 3-way split needs `requires_grad`.)"
    ),
    "stub": (
        "def ex2_classify_three_ways(tensors: list) -> list:\n"
        '    """Return one of {leaf-trainable, non-leaf, leaf-frozen} per tensor."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Build a graph that has all three categories ===\n"
        "a = t.tensor([1.0, 2.0, 3.0], requires_grad=True)   # leaf-trainable\n"
        "b = t.tensor([4.0, 5.0, 6.0], requires_grad=True)   # leaf-trainable\n"
        "c = t.tensor([7.0, 8.0, 9.0], requires_grad=False)  # leaf-frozen\n"
        "data = t.tensor([0.1, 0.2, 0.3])                    # leaf-frozen (default requires_grad=False)\n"
        "y = a * b           # non-leaf\n"
        "z = y + c           # non-leaf (op on grad-tracked tensor stays in graph)\n"
        "w = data * 2.0      # leaf-frozen — op on grad-free tensors does NOT track\n"
        "\n"
        "out = ex2_classify_three_ways([a, b, c, data, y, z, w])\n"
        "expected = ['leaf-trainable', 'leaf-trainable', 'leaf-frozen', 'leaf-frozen', 'non-leaf', 'non-leaf', 'leaf-frozen']\n"
        "assert out == expected, f'expected {expected}, got {out}'\n"
        "\n"
        "# === Empty input ===\n"
        "assert ex2_classify_three_ways([]) == []\n"
        "\n"
        "# === Single leaf-trainable ===\n"
        "x = t.randn(3, requires_grad=True)\n"
        "assert ex2_classify_three_ways([x]) == ['leaf-trainable']\n"
        "\n"
        "# === Single leaf-frozen ===\n"
        "x = t.randn(3)\n"
        "assert ex2_classify_three_ways([x]) == ['leaf-frozen']\n"
        "\n"
        "# === Single non-leaf ===\n"
        "x = t.randn(3, requires_grad=True)\n"
        "y = x.relu()\n"
        "assert ex2_classify_three_ways([y]) == ['non-leaf']\n"
        "\n"
        "# === detach() of a non-leaf becomes leaf-frozen (no grad_fn, no requires_grad) ===\n"
        "a = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "y = a * 2\n"
        "y_det = y.detach()\n"
        "assert ex2_classify_three_ways([y_det]) == ['leaf-frozen'], (\n"
        "    f'detached interior should be leaf-frozen; got {ex2_classify_three_ways([y_det])}'\n"
        ")\n"
        "\n"
        "# === Return type sanity ===\n"
        "result = ex2_classify_three_ways([a])\n"
        "assert isinstance(result, list) and all(isinstance(s, str) for s in result)"
    ),
    "solution_body": (
        "def ex2_classify_three_ways(tensors):\n"
        "    out = []\n"
        "    for x in tensors:\n"
        "        if x.grad_fn is not None:\n"
        "            out.append('non-leaf')\n"
        "        elif x.requires_grad:\n"
        "            out.append('leaf-trainable')\n"
        "        else:\n"
        "            out.append('leaf-frozen')\n"
        "    return out"
    ),
    "solution_notes": (
        "**Order of checks matters.** `grad_fn is not None` MUST come first. "
        "A non-leaf that happens to also have `requires_grad=True` (almost "
        "all of them) would otherwise be miscategorized as leaf-trainable. "
        "The non-leaf classification is the most specific condition.\n\n"
        "**Why `.is_leaf` alone can't do this.** PyTorch's `.is_leaf` returns "
        "True for BOTH leaf-trainable and leaf-frozen tensors. It's a 2-way "
        "split (leaf vs interior), not a 3-way one. The 3-way classification "
        "needs the additional `requires_grad` bit to disambiguate the two "
        "leaf flavors.\n\n"
        "**`detach()` produces a leaf-frozen.** `y.detach()` returns a new "
        "tensor that shares storage but has `grad_fn=None` and "
        "`requires_grad=False` — exactly the leaf-frozen condition. This is "
        "the standard idiom for 'freeze gradient flow through this tensor'."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — linspace-out-param ex2
# ---------------------------------------------------------------------------

SPEC_LINSPACE = {
    "atom_id": "linspace-out-param",
    "subtopic": "PyTorch: linspace out= param",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_LINSPACE_OUT_VS_COPY,
    "exercise_index": 2,
    "exercise_title": "contrast linspace(out=) with .copy_(linspace(...)) — same data, different alloc",
    "slug": "contrast-linspace-out-with-copy-linspace",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["linspace", "out", "copy_", "data_ptr", "allocation"],
    "kcs": [
        "out-vs-copy-equivalence",
        "data-ptr-preserved-in-place",
    ],
    "lo": (
        "Analyze whether `linspace(out=buf)` and `buf.copy_(linspace(...))` "
        "produce identical numeric results and identical `data_ptr()` for "
        "`buf`, demonstrating both are in-place writes that preserve the "
        "buffer identity."
    ),
    "prompt_body": (
        "Implement `ex2_fill_two_ways(buf_a, buf_b, start, end)`. Both "
        "buffers have the same shape — a 1-D float tensor. You will fill "
        "EACH using a different idiom and return a dict reporting whether "
        "(a) the resulting data is equal, (b) each buffer's `data_ptr()` "
        "was preserved.\n\n"
        "Steps:\n\n"
        "1. Record the original `data_ptr()` of each buffer: `ptr_a_before "
        "= buf_a.data_ptr()`, `ptr_b_before = buf_b.data_ptr()`.\n"
        "2. Fill `buf_a` using `t.linspace(start, end, buf_a.numel(), "
        "out=buf_a)` — the `out=` idiom.\n"
        "3. Fill `buf_b` using `buf_b.copy_(t.linspace(start, end, "
        "buf_b.numel(), dtype=buf_b.dtype))` — the temp-then-copy idiom. "
        "Threading `dtype=buf_b.dtype` makes the temp match the buffer "
        "so the copy is bit-exact (no float32→float64 round-trip).\n"
        "4. Return a dict:\n"
        "   - `'equal'`: True iff `t.equal(buf_a, buf_b)`.\n"
        "   - `'ptr_a_preserved'`: True iff `buf_a.data_ptr() == "
        "ptr_a_before`.\n"
        "   - `'ptr_b_preserved'`: True iff `buf_b.data_ptr() == "
        "ptr_b_before`.\n\n"
        "Both buffers MUST be mutated in place (caller will inspect them "
        "afterward)."
    ),
    "stub": (
        "def ex2_fill_two_ways(buf_a, buf_b, start: float, end: float) -> dict:\n"
        '    """Fill buf_a via linspace(out=), buf_b via .copy_(linspace(...)). Return audit dict."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Basic correctness ===\n"
        "buf_a = t.empty(100)\n"
        "buf_b = t.empty(100)\n"
        "ptr_a0, ptr_b0 = buf_a.data_ptr(), buf_b.data_ptr()\n"
        "result = ex2_fill_two_ways(buf_a, buf_b, 0.0, 1.0)\n"
        "assert isinstance(result, dict)\n"
        "assert result['equal'] is True, f'data should be equal across idioms; got {result}'\n"
        "assert result['ptr_a_preserved'] is True, f'out= must preserve buf_a ptr; got {result}'\n"
        "assert result['ptr_b_preserved'] is True, f'copy_ must preserve buf_b ptr; got {result}'\n"
        "assert buf_a.data_ptr() == ptr_a0, 'buf_a moved'\n"
        "assert buf_b.data_ptr() == ptr_b0, 'buf_b moved'\n"
        "\n"
        "# === Buffer values match the linspace they would have got ===\n"
        "expected = t.linspace(0.0, 1.0, 100)\n"
        "assert t.equal(buf_a, expected), 'buf_a contents != expected linspace'\n"
        "assert t.equal(buf_b, expected), 'buf_b contents != expected linspace'\n"
        "\n"
        "# === Non-default range ===\n"
        "buf_a = t.empty(50)\n"
        "buf_b = t.empty(50)\n"
        "result = ex2_fill_two_ways(buf_a, buf_b, -3.0, 7.0)\n"
        "assert result == {'equal': True, 'ptr_a_preserved': True, 'ptr_b_preserved': True}\n"
        "expected = t.linspace(-3.0, 7.0, 50)\n"
        "assert t.allclose(buf_a, expected) and t.allclose(buf_b, expected)\n"
        "\n"
        "# === Float64 buffers (matches their own dtype) ===\n"
        "buf_a = t.empty(20, dtype=t.float64)\n"
        "buf_b = t.empty(20, dtype=t.float64)\n"
        "result = ex2_fill_two_ways(buf_a, buf_b, 0.0, 1.0)\n"
        "assert result['equal'] is True\n"
        "assert buf_a.dtype == t.float64 and buf_b.dtype == t.float64\n"
        "\n"
        "# === Tiny buffer (n=2 — endpoints only) ===\n"
        "buf_a = t.empty(2)\n"
        "buf_b = t.empty(2)\n"
        "result = ex2_fill_two_ways(buf_a, buf_b, 5.0, 10.0)\n"
        "assert result['equal'] is True\n"
        "assert t.allclose(buf_a, t.tensor([5.0, 10.0]))\n"
        "\n"
        "# === Length-1 buffer (linspace returns just start) ===\n"
        "buf_a = t.empty(1)\n"
        "buf_b = t.empty(1)\n"
        "result = ex2_fill_two_ways(buf_a, buf_b, 4.2, 9.9)\n"
        "assert result['equal'] is True\n"
        "assert buf_a.item() == t.linspace(4.2, 9.9, 1).item()\n"
        "\n"
        "# === Pre-existing content overwritten (not added/merged) ===\n"
        "buf_a = t.full((10,), 99.0)\n"
        "buf_b = t.full((10,), 99.0)\n"
        "ex2_fill_two_ways(buf_a, buf_b, 0.0, 1.0)\n"
        "expected = t.linspace(0.0, 1.0, 10)\n"
        "assert t.equal(buf_a, expected) and t.equal(buf_b, expected), 'must overwrite, not merge'"
    ),
    "solution_body": (
        "def ex2_fill_two_ways(buf_a, buf_b, start, end):\n"
        "    ptr_a_before = buf_a.data_ptr()\n"
        "    ptr_b_before = buf_b.data_ptr()\n"
        "    # Idiom A: out= writes directly into buf_a, no temp.\n"
        "    t.linspace(start, end, buf_a.numel(), out=buf_a)\n"
        "    # Idiom B: linspace returns a fresh tensor (dtype-matched), copy_ overwrites buf_b in place.\n"
        "    buf_b.copy_(t.linspace(start, end, buf_b.numel(), dtype=buf_b.dtype))\n"
        "    return {\n"
        "        'equal': bool(t.equal(buf_a, buf_b)),\n"
        "        'ptr_a_preserved': buf_a.data_ptr() == ptr_a_before,\n"
        "        'ptr_b_preserved': buf_b.data_ptr() == ptr_b_before,\n"
        "    }"
    ),
    "solution_notes": (
        "**Both idioms preserve `data_ptr()`.** `out=` writes into the "
        "existing storage. `.copy_()` does the same — it overwrites the "
        "destination's storage with the source's values, NOT replacing the "
        "storage. So any caller that captured `buf.data_ptr()` before the "
        "fill still has a valid pointer afterward.\n\n"
        "**The difference is allocator pressure.** `out=` allocates zero "
        "extra tensors. `.copy_(linspace(...))` allocates ONE fresh tensor "
        "(the linspace return value), copies it into `buf_b`, then "
        "discards it. In a tight loop you pay allocator + GC churn.\n\n"
        "**`t.equal` over `t.allclose` for this assertion.** `t.equal` is "
        "exact-element equality. Since both idioms ultimately call the SAME "
        "linspace kernel against the same dtype, the bits are identical — "
        "exact equality is the strongest statement we can make."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — optimizer-repr-string ex2
# ---------------------------------------------------------------------------

SPEC_OPT_REPR = {
    "atom_id": "optimizer-repr-string",
    "subtopic": "Optimizer: __repr__ string",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_OPT_REPR_MULTIGROUP,
    "exercise_index": 2,
    "exercise_title": "multi-param-group __repr__ matching PyTorch's actual format",
    "slug": "multi-param-group-repr-pytorch-format",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["repr", "param-groups", "optimizer", "format"],
    "kcs": [
        "param-group-header-format",
        "alphabetical-key-ordering",
    ],
    "lo": (
        "Apply PyTorch's optimizer-repr format — header `<Name> (`, one "
        "`Parameter Group <i>` block per group with alphabetically-sorted "
        "`    key: value` lines, footer `)` — to a hand-rolled SGD that "
        "stores its hparams as a list of group dicts."
    ),
    "prompt_body": (
        "Implement `ex2_format_multigroup_sgd(param_groups)`. Given a list "
        "of param-group dicts, return PyTorch's canonical multi-group repr "
        "string.\n\n"
        "Inputs:\n"
        "- `param_groups`: `list[dict]`. Each dict contains hparam keys "
        "(e.g. `'lr'`, `'momentum'`, `'weight_decay'`, ...). Each dict MAY "
        "include a `'params'` key — IGNORE it for the repr (PyTorch does "
        "too).\n\n"
        "Output: single `str` with this EXACT shape (note: trailing `)` on "
        "its own line, NO trailing newline):\n\n"
        "```\n"
        "SGD (\n"
        "Parameter Group 0\n"
        "    lr: 0.001\n"
        "    momentum: 0.9\n"
        "Parameter Group 1\n"
        "    lr: 0.0001\n"
        "    momentum: 0.0\n"
        "    weight_decay: 0.0005\n"
        ")\n"
        "```\n"
        "\n"
        "Rules:\n"
        "1. Header line: literal `'SGD ('`.\n"
        "2. Per group: `'Parameter Group <i>'` line (zero-indexed), then "
        "one `'    <key>: <value>'` line per hparam, with keys sorted "
        "ALPHABETICALLY, excluding `'params'`.\n"
        "3. Values rendered with `str(value)` — float `0.001` becomes "
        "`'0.001'`, int `64` becomes `'64'`, etc.\n"
        "4. Footer line: literal `')'`.\n"
        "5. Lines joined with `'\\n'`; NO trailing newline at the end."
    ),
    "stub": (
        "def ex2_format_multigroup_sgd(param_groups: list) -> str:\n"
        '    """Return the multi-param-group repr string for hand-rolled SGD."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Single-group case (degenerate) ===\n"
        "groups = [{'lr': 0.001, 'momentum': 0.9}]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "expected = 'SGD (\\nParameter Group 0\\n    lr: 0.001\\n    momentum: 0.9\\n)'\n"
        "assert out == expected, f'single-group mismatch.\\nGot:\\n{out!r}\\nExpected:\\n{expected!r}'\n"
        "\n"
        "# === Two-group case (the headline test) ===\n"
        "groups = [\n"
        "    {'lr': 0.001, 'momentum': 0.9},\n"
        "    {'lr': 0.0001, 'momentum': 0.0, 'weight_decay': 0.0005},\n"
        "]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "expected = (\n"
        "    'SGD (\\n'\n"
        "    'Parameter Group 0\\n'\n"
        "    '    lr: 0.001\\n'\n"
        "    '    momentum: 0.9\\n'\n"
        "    'Parameter Group 1\\n'\n"
        "    '    lr: 0.0001\\n'\n"
        "    '    momentum: 0.0\\n'\n"
        "    '    weight_decay: 0.0005\\n'\n"
        "    ')'\n"
        ")\n"
        "assert out == expected, f'two-group mismatch.\\nGot:\\n{out!r}\\nExpected:\\n{expected!r}'\n"
        "\n"
        "# === Alphabetical key ordering enforced ===\n"
        "# Insertion order is z, a, m → output must be a, m, z.\n"
        "groups = [{'zeta': 1, 'alpha': 2, 'momentum': 3}]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "lines = out.split('\\n')\n"
        "# Lines 2, 3, 4 are the hparam lines (after 'SGD (' and 'Parameter Group 0').\n"
        "assert lines[2].lstrip().startswith('alpha:'), f'keys must sort alphabetically; got line[2]={lines[2]!r}'\n"
        "assert lines[3].lstrip().startswith('momentum:')\n"
        "assert lines[4].lstrip().startswith('zeta:')\n"
        "\n"
        "# === 'params' key is excluded ===\n"
        "groups = [{'lr': 0.01, 'params': ['fake_param_list'], 'momentum': 0.9}]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "assert 'params:' not in out, f\"'params' key must be excluded; got {out!r}\"\n"
        "assert 'fake_param_list' not in out, f'param values must not leak into repr; got {out!r}'\n"
        "assert 'lr: 0.01' in out and 'momentum: 0.9' in out\n"
        "\n"
        "# === Three groups (group counter increments correctly) ===\n"
        "groups = [{'lr': 1.0}, {'lr': 2.0}, {'lr': 3.0}]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "assert 'Parameter Group 0' in out\n"
        "assert 'Parameter Group 1' in out\n"
        "assert 'Parameter Group 2' in out\n"
        "# And in the right order:\n"
        "p0 = out.index('Parameter Group 0')\n"
        "p1 = out.index('Parameter Group 1')\n"
        "p2 = out.index('Parameter Group 2')\n"
        "assert p0 < p1 < p2, 'group headers must appear in order'\n"
        "\n"
        "# === No trailing newline ===\n"
        "groups = [{'lr': 0.001}]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "assert not out.endswith('\\n'), f'must not end with newline; got {out!r}'\n"
        "assert out.endswith(')'), f'must end with bare close-paren; got {out!r}'\n"
        "\n"
        "# === Integer hparams render correctly ===\n"
        "groups = [{'lr': 0.001, 'batch_size': 64}]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "assert '    batch_size: 64' in out\n"
        "assert '    lr: 0.001' in out\n"
        "# Alphabetical: batch_size before lr.\n"
        "assert out.index('batch_size:') < out.index('lr:'), 'batch_size comes before lr alphabetically'\n"
        "\n"
        "# === Empty group dict (no hparams) — just header + footer line ===\n"
        "groups = [{}]\n"
        "out = ex2_format_multigroup_sgd(groups)\n"
        "assert out == 'SGD (\\nParameter Group 0\\n)', f'empty group mismatch: {out!r}'"
    ),
    "solution_body": (
        "def ex2_format_multigroup_sgd(param_groups):\n"
        "    lines = ['SGD (']\n"
        "    for i, group in enumerate(param_groups):\n"
        "        lines.append(f'Parameter Group {i}')\n"
        "        keys = sorted(k for k in group.keys() if k != 'params')\n"
        "        for k in keys:\n"
        "            lines.append(f'    {k}: {group[k]}')\n"
        "    lines.append(')')\n"
        "    return '\\n'.join(lines)"
    ),
    "solution_notes": (
        "**Sort keys per group, not globally.** Different groups can have "
        "different hparam sets (e.g. group 1 has `weight_decay`, group 0 "
        "doesn't). Sort each group's keys independently.\n\n"
        "**Exclude `'params'` before sorting.** The params list is one or "
        "more `nn.Parameter` objects — printing them dumps tensor reprs "
        "into the optimizer repr and makes it useless. PyTorch's own "
        "`Optimizer.__repr__` does this exclusion at the source.\n\n"
        "**`'\\n'.join(lines)` not `'\\n'.join(lines) + '\\n'`.** PyTorch's "
        "actual repr has no trailing newline — that's a `print()` choice, "
        "not part of the string. Adding one breaks string-equality tests."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — rmul-scalar-tensor-mix ex2
# ---------------------------------------------------------------------------

SPEC_RSUB = {
    "atom_id": "rmul-scalar-tensor-mix",
    "subtopic": "PyTorch: __rmul__ scalar/tensor mix",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_RSUB_ASYMMETRY,
    "exercise_index": 2,
    "exercise_title": "implement __sub__ and __rsub__ so 5 - my_t evaluates to 5 - my_t.value (not my_t.value - 5)",
    "slug": "implement-sub-and-rsub-with-correct-operand-order",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dunder", "rsub", "operand-order", "asymmetry"],
    "kcs": [
        "rsub-flips-operand-order",
        "wrapper-returns-same-class",
    ],
    "lo": (
        "Apply Python's reflected-op convention to a wrapper class such "
        "that `self - other` calls `__sub__(self - other)` and `other - "
        "self` calls `__rsub__(other - self)` — preserving operand order "
        "in both directions and returning the same wrapper class."
    ),
    "prompt_body": (
        "Implement the class `ScalarBox` below. It wraps a single float "
        "and supports subtraction with both `int`/`float` and other "
        "`ScalarBox` instances. The KEY test: `5 - box(3)` must equal "
        "`box(2)` — NOT `box(-2)`.\n\n"
        "Requirements:\n\n"
        "1. `__init__(self, value)` — store as `self.value = float(value)`.\n"
        "2. `__sub__(self, other)` — `self - other`. If `other` is a "
        "`ScalarBox`, use `other.value`; if a number, use it directly. "
        "Returns a new `ScalarBox(self.value - other_v)`.\n"
        "3. `__rsub__(self, other)` — Python calls this when LEFT operand "
        "doesn't know about ScalarBox. Returns `ScalarBox(other - "
        "self.value)` — note the operand order is `other - self`, NOT "
        "`self - other`.\n"
        "4. `__eq__(self, other)` — provided for the tests; compare "
        "`self.value` to either `other.value` or `other` directly (with "
        "tolerance 1e-9).\n"
        "5. `__repr__` — returns `f'ScalarBox({self.value})'`.\n\n"
        "Constraints:\n"
        "- Both `__sub__` and `__rsub__` must return a new `ScalarBox`, "
        "not a plain float.\n"
        "- `ScalarBox(3) - ScalarBox(1)` (two boxes) goes through "
        "`__sub__`, not `__rsub__`."
    ),
    "stub": (
        "class ScalarBox:\n"
        "    def __init__(self, value):\n"
        "        self.value = float(value)\n"
        "\n"
        "    def __sub__(self, other):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def __rsub__(self, other):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def __eq__(self, other):\n"
        "        ov = other.value if isinstance(other, ScalarBox) else other\n"
        "        return abs(self.value - float(ov)) < 1e-9\n"
        "\n"
        "    def __hash__(self):\n"
        "        return hash(self.value)\n"
        "\n"
        "    def __repr__(self):\n"
        "        return f'ScalarBox({self.value})'\n"
        "\n"
        "def ex2_scalarbox():\n"
        "    return ScalarBox"
    ),
    "test_body": (
        "Cls = ex2_scalarbox()\n"
        "\n"
        "# === box - scalar (normal __sub__) ===\n"
        "result = Cls(10) - 3\n"
        "assert isinstance(result, Cls), f'must return ScalarBox; got {type(result).__name__}'\n"
        "assert result == Cls(7), f'10 - 3 should be 7, got {result}'\n"
        "\n"
        "# === scalar - box (the headline __rsub__ test) ===\n"
        "result = 5 - Cls(3)\n"
        "assert isinstance(result, Cls), f'5 - Cls(3) must return ScalarBox; got {type(result).__name__}'\n"
        "assert result == Cls(2), f'5 - Cls(3) should be Cls(2), NOT Cls(-2); got {result}'\n"
        "\n"
        "# === The asymmetry: box - scalar != scalar - box ===\n"
        "left = Cls(10) - 4    # 10 - 4 = 6\n"
        "right = 4 - Cls(10)   # 4 - 10 = -6\n"
        "assert left == Cls(6), f'box - scalar wrong: {left}'\n"
        "assert right == Cls(-6), f'scalar - box wrong: {right}'\n"
        "assert left != right, 'asymmetry must hold: box-scalar != scalar-box'\n"
        "\n"
        "# === box - box uses __sub__ (NOT __rsub__) ===\n"
        "result = Cls(7) - Cls(2)\n"
        "assert isinstance(result, Cls)\n"
        "assert result == Cls(5), f'box - box wrong: {result}'\n"
        "\n"
        "# === Float scalar on the left ===\n"
        "result = 1.5 - Cls(0.5)\n"
        "assert result == Cls(1.0), f'1.5 - Cls(0.5) should be Cls(1.0), got {result}'\n"
        "\n"
        "# === Negative results work ===\n"
        "result = 2 - Cls(5)\n"
        "assert result == Cls(-3), f'2 - Cls(5) should be Cls(-3), got {result}'\n"
        "\n"
        "# === Zero results work ===\n"
        "result = 7 - Cls(7)\n"
        "assert result == Cls(0), f'7 - Cls(7) should be Cls(0), got {result}'\n"
        "\n"
        "# === Chain: scalar - box - scalar ===\n"
        "# 10 - Cls(3) = Cls(7) ; Cls(7) - 2 = Cls(5)\n"
        "result = 10 - Cls(3) - 2\n"
        "assert result == Cls(5), f'chain wrong: {result}'\n"
        "\n"
        "# === Chain: scalar - box - box ===\n"
        "# 10 - Cls(3) = Cls(7) ; Cls(7) - Cls(2) = Cls(5)\n"
        "result = 10 - Cls(3) - Cls(2)\n"
        "assert result == Cls(5), f'mixed chain wrong: {result}'\n"
        "\n"
        "# === Repr ===\n"
        "assert repr(Cls(3.0)) == 'ScalarBox(3.0)'"
    ),
    "solution_body": (
        "class ScalarBox:\n"
        "    def __init__(self, value):\n"
        "        self.value = float(value)\n"
        "\n"
        "    def __sub__(self, other):\n"
        "        other_v = other.value if isinstance(other, ScalarBox) else other\n"
        "        return ScalarBox(self.value - other_v)\n"
        "\n"
        "    def __rsub__(self, other):\n"
        "        # other is on the LEFT — operand order is other - self, NOT self - other.\n"
        "        other_v = other.value if isinstance(other, ScalarBox) else other\n"
        "        return ScalarBox(other_v - self.value)\n"
        "\n"
        "    def __eq__(self, other):\n"
        "        ov = other.value if isinstance(other, ScalarBox) else other\n"
        "        return abs(self.value - float(ov)) < 1e-9\n"
        "\n"
        "    def __hash__(self):\n"
        "        return hash(self.value)\n"
        "\n"
        "    def __repr__(self):\n"
        "        return f'ScalarBox({self.value})'\n"
        "\n"
        "def ex2_scalarbox():\n"
        "    return ScalarBox"
    ),
    "solution_notes": (
        "**The one-line bug.** `__rsub__` returning `self.value - other` "
        "(symmetric with `__sub__`) is the universal mistake. It makes "
        "`5 - box(3)` equal `box(-2)`. The reflected method MUST flip the "
        "operand order: `other - self`.\n\n"
        "**Why Python's convention works this way.** Reflected methods are "
        "called as a FALLBACK when the LEFT operand returned `NotImplemented`. "
        "Python's logic: 'try `left.__op__(right)`, then `right.__rop__(left)`. "
        "From `__rop__`'s perspective, `self` is on the right and `other` is "
        "on the left — so it must compute `other OP self`, not `self OP other`.\n\n"
        "**Same trap for `__rtruediv__`, `__rmod__`, `__rpow__`.** Any "
        "non-commutative operator has this asymmetry. Only `+` and `*` "
        "(and bitwise `&`, `|`, `^`) commute, so their reflected versions "
        "are symmetric — that's why `__rmul__` from ex1 was easy."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — tensor-reshape-view ex2
# ---------------------------------------------------------------------------

SPEC_VIEW = {
    "atom_id": "tensor-reshape-view",
    "subtopic": "PyTorch: reshape vs view",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_VIEW_AFTER_TRANSPOSE,
    "exercise_index": 2,
    "exercise_title": "post-transpose contiguity trap — view raises, reshape works",
    "slug": "post-transpose-contiguity-trap-view-vs-reshape",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["view", "reshape", "transpose", "contiguous", "stride"],
    "kcs": [
        "view-raises-on-noncontig",
        "reshape-copies-when-needed",
    ],
    "lo": (
        "Analyze whether `.view()` raises after a `.transpose()` while "
        "`.reshape()` succeeds, and explain the result via `data_ptr()` "
        "comparison — `reshape` copies after a non-contiguous transpose."
    ),
    "prompt_body": (
        "Implement `ex2_post_transpose_audit(x)`. Takes a contiguous tensor "
        "`x` of shape `(A, B, C)`. Performs `y = x.transpose(0, 2)` "
        "(shape `(C, B, A)`, NOT contiguous in general) and then attempts "
        "BOTH `y.view(C*B*A)` and `y.reshape(C*B*A)`.\n\n"
        "Return a dict with these keys:\n\n"
        "- `'y_contiguous'`: bool — `y.is_contiguous()` immediately after "
        "transpose.\n"
        "- `'view_raised'`: bool — True iff `y.view(C*B*A)` raises "
        "`RuntimeError`. False if it succeeded.\n"
        "- `'reshape_succeeded'`: bool — True iff `y.reshape(C*B*A)` "
        "returned a tensor without raising.\n"
        "- `'reshape_data_ptr_same'`: bool — True iff the reshape's "
        "`data_ptr()` equals `y.data_ptr()` (i.e. no copy). False if a "
        "copy was made.\n"
        "- `'reshape_values_correct'`: bool — True iff the reshape's "
        "flattened content equals `y.flatten()`.\n\n"
        "Constraints:\n"
        "- Catch `RuntimeError` from the `.view()` attempt — do NOT let "
        "it propagate.\n"
        "- All five keys must be present in the returned dict."
    ),
    "stub": (
        "def ex2_post_transpose_audit(x) -> dict:\n"
        '    """Audit view-vs-reshape behaviour after transpose."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Standard non-contiguous case (the headline) ===\n"
        "x = t.arange(24).reshape(2, 3, 4).float()\n"
        "report = ex2_post_transpose_audit(x)\n"
        "assert isinstance(report, dict)\n"
        "for k in ('y_contiguous', 'view_raised', 'reshape_succeeded', 'reshape_data_ptr_same', 'reshape_values_correct'):\n"
        "    assert k in report, f'missing key {k}: {report}'\n"
        "\n"
        "assert report['y_contiguous'] is False, f'transposed tensor should NOT be contiguous; got {report}'\n"
        "assert report['view_raised'] is True, f'view(numel) on non-contig must raise; got {report}'\n"
        "assert report['reshape_succeeded'] is True, f'reshape must succeed; got {report}'\n"
        "assert report['reshape_data_ptr_same'] is False, (\n"
        "    f'after non-contig transpose, reshape must COPY (different data_ptr); got {report}'\n"
        ")\n"
        "assert report['reshape_values_correct'] is True, f'reshape values mismatch; got {report}'\n"
        "\n"
        "# === Already-contiguous corner case: transpose dims of size 1 keeps contig. ===\n"
        "# A tensor with shape (1, 3, 1) transposed (0, 2) is still (1, 3, 1) and contiguous.\n"
        "x = t.arange(3).reshape(1, 3, 1).float()\n"
        "report = ex2_post_transpose_audit(x)\n"
        "# In this case y IS contiguous, so view should work and reshape should NOT copy.\n"
        "assert report['y_contiguous'] is True, f'(1,3,1) transposed should stay contig; got {report}'\n"
        "assert report['view_raised'] is False, f'view on contiguous must succeed; got {report}'\n"
        "assert report['reshape_succeeded'] is True\n"
        "assert report['reshape_data_ptr_same'] is True, (\n"
        "    f'reshape on contig should NOT copy (same data_ptr); got {report}'\n"
        ")\n"
        "assert report['reshape_values_correct'] is True\n"
        "\n"
        "# === Larger non-contig case ===\n"
        "x = t.arange(60).reshape(3, 4, 5).float()\n"
        "report = ex2_post_transpose_audit(x)\n"
        "assert report['y_contiguous'] is False\n"
        "assert report['view_raised'] is True\n"
        "assert report['reshape_succeeded'] is True\n"
        "assert report['reshape_values_correct'] is True\n"
        "\n"
        "# === Values check: after transpose(0,2), flatten of y should match the\n"
        "# reshape that the function produced. ===\n"
        "x = t.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])  # (2,2,2)\n"
        "report = ex2_post_transpose_audit(x)\n"
        "assert report['reshape_values_correct'] is True, (\n"
        "    f'reshape contents must match y.flatten(); got {report}'\n"
        ")\n"
        "\n"
        "# === Sanity: view_raised + view succeeded are mutually exclusive ===\n"
        "for x_test in [t.arange(24).reshape(2,3,4).float(),\n"
        "               t.arange(3).reshape(1,3,1).float()]:\n"
        "    r = ex2_post_transpose_audit(x_test)\n"
        "    # If view raised, that means view did NOT succeed; vice versa.\n"
        "    assert isinstance(r['view_raised'], bool)\n"
        "    assert isinstance(r['reshape_succeeded'], bool)\n"
        "\n"
        "# === reshape_succeeded is always True for these inputs (reshape never fails on valid numel) ===\n"
        "assert ex2_post_transpose_audit(t.arange(24).reshape(2,3,4).float())['reshape_succeeded'] is True\n"
        "assert ex2_post_transpose_audit(t.arange(3).reshape(1,3,1).float())['reshape_succeeded'] is True"
    ),
    "solution_body": (
        "def ex2_post_transpose_audit(x):\n"
        "    A, B, C = x.shape\n"
        "    y = x.transpose(0, 2)  # (C, B, A) — usually non-contiguous\n"
        "    n = C * B * A\n"
        "    report = {'y_contiguous': bool(y.is_contiguous())}\n"
        "\n"
        "    # Attempt the .view() — catch the RuntimeError it raises on non-contig.\n"
        "    try:\n"
        "        _ = y.view(n)\n"
        "        report['view_raised'] = False\n"
        "    except RuntimeError:\n"
        "        report['view_raised'] = True\n"
        "\n"
        "    # .reshape() never raises for the same numel; may or may not copy.\n"
        "    try:\n"
        "        r = y.reshape(n)\n"
        "        report['reshape_succeeded'] = True\n"
        "        report['reshape_data_ptr_same'] = (r.data_ptr() == y.data_ptr())\n"
        "        report['reshape_values_correct'] = bool(t.equal(r, y.flatten()))\n"
        "    except RuntimeError:\n"
        "        report['reshape_succeeded'] = False\n"
        "        report['reshape_data_ptr_same'] = False\n"
        "        report['reshape_values_correct'] = False\n"
        "    return report"
    ),
    "solution_notes": (
        "**The asymmetry is by design.** `view` PROMISES no copy — that's "
        "its contract. When a no-copy view isn't possible (non-contig "
        "storage that the new shape can't be re-strided over), it raises "
        "rather than silently copy and break the promise. `reshape` makes "
        "no such promise — it falls back to `.contiguous().view(...)` "
        "when needed.\n\n"
        "**`data_ptr()` is the copy oracle.** Same `data_ptr()` → no copy "
        "(view-like). Different `data_ptr()` → copy was made (new storage). "
        "After a non-contig transpose, `reshape` always copies; the new "
        "data_ptr is your evidence.\n\n"
        "**The `(1, 3, 1)` corner case isn't pedantry.** Tensors with "
        "singleton dimensions are common (broadcasting, channels-first "
        "with batch=1). Their strides interact with transpose in ways "
        "that preserve contiguity in some axes — the audit helper exposes "
        "this so the user sees that contig is a runtime property, not a "
        "shape-derivable one."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — trainer-subclass-extend ex2
# ---------------------------------------------------------------------------

SPEC_TRAINER = {
    "atom_id": "trainer-subclass-extend",
    "subtopic": "Trainer: subclass extend pattern",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_TRAINER_MRO_CHAIN,
    "exercise_index": 2,
    "exercise_title": "three-level trainer chain — each subclass extends _step via super() and adds one metric",
    "slug": "three-level-trainer-chain-mro-super-delegation",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["mro", "super", "subclass", "trainer", "chain"],
    "kcs": [
        "super-walks-mro-not-base",
        "each-level-extends-dict",
    ],
    "lo": (
        "Apply `super()._step(batch)` at each of three subclass levels so "
        "Python's MRO walks the chain Base → Logging → FrozenLogging and "
        "the final dict accumulates the metrics added by every level."
    ),
    "prompt_body": (
        "`BaseTrainer` is provided in the stub. Its `_step(batch)` returns "
        "`{'loss': float(batch[0].abs().sum())}`. Build TWO further "
        "subclass levels:\n\n"
        "1. `LoggingTrainer(BaseTrainer)`:\n"
        "   - Override `_step(self, batch)`:\n"
        "     a. Call `super()._step(batch)` and capture the dict `d`.\n"
        "     b. Add `d['input_mag'] = float(batch[0].abs().mean())`.\n"
        "     c. Return `d`.\n"
        "2. `FrozenLoggingTrainer(LoggingTrainer)`:\n"
        "   - Override `_step(self, batch)`:\n"
        "     a. Call `super()._step(batch)` (walks MRO to "
        "`LoggingTrainer._step`, which itself defers to "
        "`BaseTrainer._step`).\n"
        "     b. Add `d['grad_norm'] = 0.0` (head is frozen, no grad).\n"
        "     c. Return `d`.\n\n"
        "Return both classes from `ex2_trainer_chain()` as a 2-tuple "
        "`(LoggingTrainer, FrozenLoggingTrainer)`.\n\n"
        "Constraints:\n"
        "- Each subclass MUST call `super()._step(batch)` — do not "
        "re-implement the base body.\n"
        "- `FrozenLoggingTrainer._step` MUST NOT call "
        "`LoggingTrainer._step` directly by name — use `super()`.\n"
        "- The final dict from `FrozenLoggingTrainer._step` must contain "
        "all three keys: `'loss'`, `'input_mag'`, `'grad_norm'`."
    ),
    "stub": (
        "class BaseTrainer:\n"
        "    def _step(self, batch):\n"
        "        return {'loss': float(batch[0].abs().sum())}\n"
        "\n"
        "# LoggingTrainer and FrozenLoggingTrainer go here.\n"
        "\n"
        "def ex2_trainer_chain():\n"
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "Logging, Frozen = ex2_trainer_chain()\n"
        "\n"
        "# === Class hierarchy ===\n"
        "assert issubclass(Logging, BaseTrainer), 'LoggingTrainer must inherit from BaseTrainer'\n"
        "assert issubclass(Frozen, Logging), 'FrozenLoggingTrainer must inherit from LoggingTrainer'\n"
        "assert issubclass(Frozen, BaseTrainer), 'FrozenLoggingTrainer transitively inherits from BaseTrainer'\n"
        "\n"
        "# === LoggingTrainer adds input_mag ===\n"
        "x = t.tensor([[-3.0, -1.0, 1.0, 3.0]])  # abs sum = 8, abs mean = 2.0\n"
        "batch = (x,)\n"
        "d = Logging()._step(batch)\n"
        "assert set(d.keys()) == {'loss', 'input_mag'}, f'LoggingTrainer keys wrong: {d.keys()}'\n"
        "assert abs(d['loss'] - 8.0) < 1e-6, f'loss wrong: {d}'\n"
        "assert abs(d['input_mag'] - 2.0) < 1e-6, f'input_mag wrong: {d}'\n"
        "\n"
        "# === FrozenLoggingTrainer adds grad_norm AND preserves the two earlier keys ===\n"
        "d = Frozen()._step(batch)\n"
        "assert set(d.keys()) == {'loss', 'input_mag', 'grad_norm'}, f'FrozenLoggingTrainer keys wrong: {d.keys()}'\n"
        "assert abs(d['loss'] - 8.0) < 1e-6\n"
        "assert abs(d['input_mag'] - 2.0) < 1e-6\n"
        "assert d['grad_norm'] == 0.0\n"
        "\n"
        "# === MRO check: super() in FrozenLoggingTrainer goes to LoggingTrainer ===\n"
        "mro_names = [c.__name__ for c in Frozen.__mro__]\n"
        "assert mro_names[:3] == [Frozen.__name__, Logging.__name__, 'BaseTrainer'], (\n"
        "    f'MRO must be Frozen -> Logging -> BaseTrainer, got {mro_names}'\n"
        ")\n"
        "\n"
        "# === Different batch shapes still work ===\n"
        "x = t.tensor([[1.0, -1.0]])  # abs sum=2, abs mean=1\n"
        "d = Frozen()._step((x,))\n"
        "assert abs(d['loss'] - 2.0) < 1e-6 and abs(d['input_mag'] - 1.0) < 1e-6\n"
        "\n"
        "# === Pure BaseTrainer still works (no extra keys) ===\n"
        "d = BaseTrainer()._step((t.tensor([[1.0, 1.0]]),))\n"
        "assert set(d.keys()) == {'loss'}\n"
        "\n"
        "# === LoggingTrainer instance is NOT a FrozenLoggingTrainer ===\n"
        "assert not isinstance(Logging(), Frozen), 'Logging instance must not be Frozen'\n"
        "assert isinstance(Frozen(), Logging), 'Frozen instance IS a Logging instance'"
    ),
    "solution_body": (
        "class LoggingTrainer(BaseTrainer):\n"
        "    def _step(self, batch):\n"
        "        d = super()._step(batch)\n"
        "        d['input_mag'] = float(batch[0].abs().mean())\n"
        "        return d\n"
        "\n"
        "class FrozenLoggingTrainer(LoggingTrainer):\n"
        "    def _step(self, batch):\n"
        "        d = super()._step(batch)\n"
        "        d['grad_norm'] = 0.0\n"
        "        return d\n"
        "\n"
        "def ex2_trainer_chain():\n"
        "    return (LoggingTrainer, FrozenLoggingTrainer)"
    ),
    "solution_notes": (
        "**`super()` walks the MRO, not the static base.** From "
        "`FrozenLoggingTrainer._step`, `super()._step(batch)` does NOT mean "
        "`BaseTrainer._step(batch)`. It means 'the next class in "
        "`type(self).__mro__` after `FrozenLoggingTrainer`' — which is "
        "`LoggingTrainer`. The chain composes automatically.\n\n"
        "**Why explicit naming breaks the chain.** If "
        "`FrozenLoggingTrainer._step` called `LoggingTrainer._step(self, "
        "batch)` directly, it would still work for this exact hierarchy. "
        "But it breaks under multiple inheritance — a sibling subclass "
        "wouldn't get a chance to run. `super()` is the future-proof form.\n\n"
        "**Same dict mutated in place.** Each level adds its key into "
        "the dict returned by `super()._step`. No copying. The base "
        "creates the dict; each subclass extends it. By the time control "
        "returns to the caller, all three keys are present."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — conditional-hparam-branch ex2
# ---------------------------------------------------------------------------

SPEC_COND_DROPOUT = {
    "atom_id": "conditional-hparam-branch",
    "subtopic": "PyTorch: Conditional hparam branch",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_COND_DROPOUT,
    "exercise_index": 2,
    "exercise_title": "conditional Dropout submodule — only register nn.Dropout when p > 0",
    "slug": "conditional-dropout-submodule-only-when-p-gt-zero",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dropout", "conditional", "submodule", "hparam"],
    "kcs": [
        "conditional-submodule-registration",
        "hasattr-gated-forward",
    ],
    "lo": (
        "Apply the conditional-submodule pattern: register `nn.Dropout(p)` "
        "as `self.drop` only when `p > 0`, and gate the forward call on "
        "`hasattr(self, 'drop')` so the `p == 0` path skips the module "
        "entirely (no `nn.Identity` placeholder)."
    ),
    "prompt_body": (
        "Implement `ex2_make_block(d_in, d_out, dropout)` — returns a "
        "`nn.Module` subclass instance.\n\n"
        "Behavior:\n\n"
        "1. The class (define it inside the helper or at module scope — "
        "your choice) is `nn.Module`-based.\n"
        "2. In `__init__`:\n"
        "   - Call `super().__init__()`.\n"
        "   - Register `self.linear = nn.Linear(d_in, d_out)`.\n"
        "   - If `dropout > 0`: register `self.drop = nn.Dropout(p="
        "dropout)`.\n"
        "   - If `dropout == 0`: do NOT register `self.drop` at all (no "
        "`nn.Identity`, no `self.drop = None`).\n"
        "3. In `forward(x)`:\n"
        "   - `x = self.linear(x)`.\n"
        "   - If `hasattr(self, 'drop')`: `x = self.drop(x)`.\n"
        "   - Return `x`.\n\n"
        "Constraints:\n"
        "- `dropout` is a float in `[0.0, 1.0)`.\n"
        "- For `dropout == 0`, `dict(self.named_modules())` must NOT "
        "contain a `'drop'` key.\n"
        "- For `dropout > 0`, it MUST contain a `'drop'` key whose value "
        "is an `nn.Dropout` instance.\n"
        "- The Linear's parameter count must be unchanged regardless of "
        "the dropout branch."
    ),
    "stub": (
        "def ex2_make_block(d_in: int, d_out: int, dropout: float):\n"
        '    """Build a Linear+optional-Dropout block. Dropout registered only when p > 0."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# === dropout == 0: no drop submodule at all ===\n"
        "block = ex2_make_block(4, 6, dropout=0.0)\n"
        "assert isinstance(block, nn.Module)\n"
        "named = dict(block.named_modules())\n"
        "assert 'drop' not in named, f'p=0 must NOT register drop; got modules {list(named.keys())}'\n"
        "assert 'linear' in named, f'linear must be registered; got modules {list(named.keys())}'\n"
        "assert not hasattr(block, 'drop'), 'with p=0, hasattr(block, drop) must be False'\n"
        "\n"
        "# === dropout > 0: drop submodule IS registered ===\n"
        "block = ex2_make_block(4, 6, dropout=0.5)\n"
        "assert hasattr(block, 'drop'), 'with p>0, block.drop must exist'\n"
        "assert isinstance(block.drop, nn.Dropout)\n"
        "assert abs(block.drop.p - 0.5) < 1e-9\n"
        "named = dict(block.named_modules())\n"
        "assert 'drop' in named and isinstance(named['drop'], nn.Dropout)\n"
        "\n"
        "# === Parameter count is the SAME either way (Dropout has 0 params) ===\n"
        "n_params_no_drop = sum(p.numel() for p in ex2_make_block(4, 6, 0.0).parameters())\n"
        "n_params_with_drop = sum(p.numel() for p in ex2_make_block(4, 6, 0.5).parameters())\n"
        "assert n_params_no_drop == n_params_with_drop, (\n"
        "    f'Dropout has no params; counts must match. got {n_params_no_drop} vs {n_params_with_drop}'\n"
        ")\n"
        "# 4*6 + 6 = 30\n"
        "assert n_params_no_drop == 30\n"
        "\n"
        "# === Forward works in both branches (eval mode, no randomness) ===\n"
        "block = ex2_make_block(4, 6, dropout=0.0)\n"
        "block.eval()\n"
        "x = t.randn(3, 4)\n"
        "y = block(x)\n"
        "assert y.shape == (3, 6), f'forward output shape wrong: {y.shape}'\n"
        "\n"
        "block = ex2_make_block(4, 6, dropout=0.5)\n"
        "block.eval()  # dropout in eval mode is a no-op — deterministic\n"
        "y = block(x)\n"
        "assert y.shape == (3, 6)\n"
        "# In eval mode, dropout=0.5 should give the same output as dropout=0.0\n"
        "# (when both blocks have the same Linear weights — they won't here\n"
        "# because of separate random init, so just check the shape).\n"
        "\n"
        "# === Train mode: dropout > 0 actually drops in train() ===\n"
        "t.manual_seed(42)\n"
        "block = ex2_make_block(4, 6, dropout=0.9)\n"
        "block.train()\n"
        "x = t.ones(100, 4)\n"
        "y = block(x)\n"
        "n_zeros = (y == 0).sum().item()\n"
        "assert n_zeros > 0, f'with p=0.9 in train mode, some outputs should be zeroed; got n_zeros={n_zeros}'\n"
        "\n"
        "# === Train mode: dropout == 0 has zero zeros (with non-zero linear bias) ===\n"
        "block = ex2_make_block(4, 6, dropout=0.0)\n"
        "block.train()\n"
        "# Set the linear weights to all-1 and bias to all-1 so the output is deterministic non-zero.\n"
        "with t.no_grad():\n"
        "    block.linear.weight.fill_(1.0)\n"
        "    block.linear.bias.fill_(1.0)\n"
        "y = block(t.ones(10, 4))\n"
        "# Each output element should be 4*1 + 1 = 5.\n"
        "assert (y == 5.0).all(), f'p=0 must not zero anything; got y={y}'\n"
        "\n"
        "# === Linear submodule unchanged across branches ===\n"
        "for p in [0.0, 0.1, 0.5, 0.9]:\n"
        "    block = ex2_make_block(8, 16, dropout=p)\n"
        "    assert isinstance(block.linear, nn.Linear)\n"
        "    assert block.linear.in_features == 8 and block.linear.out_features == 16"
    ),
    "solution_body": (
        "import torch.nn as nn\n"
        "\n"
        "class _Block(nn.Module):\n"
        "    def __init__(self, d_in, d_out, dropout):\n"
        "        super().__init__()\n"
        "        self.linear = nn.Linear(d_in, d_out)\n"
        "        if dropout > 0:\n"
        "            self.drop = nn.Dropout(p=dropout)\n"
        "        # else: do nothing — no attribute set at all.\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = self.linear(x)\n"
        "        if hasattr(self, 'drop'):\n"
        "            x = self.drop(x)\n"
        "        return x\n"
        "\n"
        "def ex2_make_block(d_in, d_out, dropout):\n"
        "    return _Block(d_in, d_out, dropout)"
    ),
    "solution_notes": (
        "**`if dropout > 0` is the registration gate.** `nn.Module.__setattr__` "
        "auto-registers any `nn.Module` value into `self._modules`. If you "
        "never assign `self.drop`, no registration happens — "
        "`named_modules()` doesn't see it, `state_dict()` doesn't have an "
        "entry, and `forward()` skips it via `hasattr`.\n\n"
        "**Why not `self.drop = nn.Identity()`.** `nn.Identity()` ALSO "
        "registers (as a child module). Functionally equivalent at "
        "forward-time, but `named_modules()` then includes an extra entry "
        "that downstream introspection code (e.g. logging hooks, hardware "
        "compilers) has to learn to ignore. The drill picks the "
        "no-registration form as the strictest conditional pattern.\n\n"
        "**Parameter count is invariant.** `nn.Dropout` has zero `Parameter`s "
        "— only a `p` config attribute. Whether you include it or not, "
        "`model.parameters()` returns the same list. This is what makes "
        "the conditional safe to toggle mid-sweep."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — device-consistent-construct ex2
# ---------------------------------------------------------------------------

SPEC_DEVICE_INFER = {
    "atom_id": "device-consistent-construct",
    "subtopic": "PyTorch: Device-consistent tensor construction",
    "topic_folder": TOPIC_NUM,
    "atom_recap_md": RECAP_DEVICE_INFER,
    "exercise_index": 2,
    "exercise_title": "infer device + dtype from an existing param via next(self.parameters())",
    "slug": "infer-device-dtype-from-next-self-parameters",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["device", "dtype", "parameters", "next", "scratch"],
    "kcs": [
        "next-self-parameters-ref",
        "fallback-when-no-parameters",
    ],
    "lo": (
        "Apply the `next(self.parameters())` idiom to infer the module's "
        "current device + dtype without an input tensor in hand, and "
        "gracefully fall back when the module has no registered "
        "parameters."
    ),
    "prompt_body": (
        "Implement `ex2_make_scratch_module()`. Returns an `nn.Module` "
        "subclass instance with these methods:\n\n"
        "1. `__init__(self)`:\n"
        "   - `super().__init__()`.\n"
        "   - Register `self.linear = nn.Linear(8, 4)` so the module has "
        "at least one Parameter.\n"
        "2. `make_scratch(self, shape)`:\n"
        "   - Get a reference param via `next(self.parameters())`. If "
        "`StopIteration` (no params), fall back to `torch.empty(0).device` "
        "(CPU) and `torch.empty(0).dtype` (float32).\n"
        "   - Allocate and return `t.zeros(shape, device=ref.device, "
        "dtype=ref.dtype)`.\n\n"
        "Constraints:\n"
        "- DO NOT hardcode `device='cpu'` or `dtype=t.float32`. Always go "
        "through `next(self.parameters())` (or the fallback path) so the "
        "scratch tracks the module's CURRENT location after `.to(...)`.\n"
        "- For the no-params fallback, the caller's tests will TEMPORARILY "
        "blank out the module's params — your method must handle that.\n"
        "- Return value: a tensor with `shape == shape` (tuple-equal), "
        "`device == ref.device`, `dtype == ref.dtype`."
    ),
    "stub": (
        "def ex2_make_scratch_module():\n"
        '    """Return a Module whose make_scratch(shape) infers device+dtype from its first param."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# === Default fresh module — params are on CPU float32 ===\n"
        "mod = ex2_make_scratch_module()\n"
        "assert isinstance(mod, nn.Module)\n"
        "assert hasattr(mod, 'make_scratch'), 'make_scratch method missing'\n"
        "assert any(True for _ in mod.parameters()), 'module must have at least one parameter'\n"
        "\n"
        "buf = mod.make_scratch((3, 5))\n"
        "assert isinstance(buf, t.Tensor)\n"
        "assert tuple(buf.shape) == (3, 5), f'shape wrong: {buf.shape}'\n"
        "assert buf.device.type == 'cpu', f'fresh module on CPU; buf must be CPU; got {buf.device}'\n"
        "assert buf.dtype == t.float32, f'default param dtype is float32; got {buf.dtype}'\n"
        "assert (buf == 0).all(), 'scratch must be zero-filled'\n"
        "\n"
        "# === After .to(dtype=float64), scratch follows the param's dtype ===\n"
        "mod = ex2_make_scratch_module()\n"
        "mod = mod.to(dtype=t.float64)\n"
        "buf = mod.make_scratch((2, 2))\n"
        "assert buf.dtype == t.float64, f'after to(float64), scratch must be float64; got {buf.dtype}'\n"
        "assert buf.device.type == 'cpu'\n"
        "\n"
        "# === After .to(dtype=float16), scratch follows again ===\n"
        "mod = ex2_make_scratch_module()\n"
        "mod = mod.to(dtype=t.float16)\n"
        "buf = mod.make_scratch((4,))\n"
        "assert buf.dtype == t.float16, f'after to(float16), scratch must be float16; got {buf.dtype}'\n"
        "\n"
        "# === Shape from a tuple, list, and torch.Size all work ===\n"
        "mod = ex2_make_scratch_module()\n"
        "for shape in [(2, 3), [4, 5], t.Size([6, 7])]:\n"
        "    buf = mod.make_scratch(shape)\n"
        "    assert tuple(buf.shape) == tuple(shape), f'shape arg {shape} mishandled: {buf.shape}'\n"
        "\n"
        "# === Empty shape → 0-D tensor ===\n"
        "mod = ex2_make_scratch_module()\n"
        "buf = mod.make_scratch(())\n"
        "assert buf.dim() == 0\n"
        "assert buf.item() == 0.0\n"
        "\n"
        "# === No-params fallback — strip the module's parameters and confirm fallback fires ===\n"
        "mod = ex2_make_scratch_module()\n"
        "# Remove the linear so the module has zero registered params.\n"
        "del mod.linear  # nn.Module __delattr__ removes from _modules and _parameters\n"
        "assert not any(True for _ in mod.parameters()), 'module should have no params after deletion'\n"
        "buf = mod.make_scratch((3,))\n"
        "assert isinstance(buf, t.Tensor)\n"
        "assert tuple(buf.shape) == (3,)\n"
        "assert buf.device.type == 'cpu', f'fallback must be CPU; got {buf.device}'\n"
        "assert buf.dtype == t.float32, f'fallback must be float32; got {buf.dtype}'\n"
        "\n"
        "# === After multiple .to() ops the scratch stays consistent with current state ===\n"
        "mod = ex2_make_scratch_module().to(dtype=t.float64).to(dtype=t.float32)\n"
        "buf = mod.make_scratch((2,))\n"
        "assert buf.dtype == t.float32, f'after double .to() the final dtype should win; got {buf.dtype}'"
    ),
    "solution_body": (
        "import torch.nn as nn\n"
        "\n"
        "class _ScratchModule(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.linear = nn.Linear(8, 4)\n"
        "\n"
        "    def make_scratch(self, shape):\n"
        "        try:\n"
        "            ref = next(self.parameters())\n"
        "            device, dtype = ref.device, ref.dtype\n"
        "        except StopIteration:\n"
        "            fallback = t.empty(0)\n"
        "            device, dtype = fallback.device, fallback.dtype\n"
        "        return t.zeros(shape, device=device, dtype=dtype)\n"
        "\n"
        "def ex2_make_scratch_module():\n"
        "    return _ScratchModule()"
    ),
    "solution_notes": (
        "**`next(self.parameters())` is the canonical 'where is this "
        "module?' query.** All registered Parameters share the same device "
        "after `.to(device)` (PyTorch's `Module.to` walks all params). "
        "Picking the first one is sufficient — the rest agree.\n\n"
        "**`StopIteration` fallback is real.** A module can have zero "
        "registered parameters (e.g. a pure-activation block, a custom "
        "Module wrapping a functional). Without the `try/except`, "
        "`next(self.parameters())` raises and your `make_scratch` is "
        "useless on those modules. Hugging Face's "
        "`PreTrainedModel._device` does the same fallback.\n\n"
        "**Don't ALSO check `self.buffers()`.** Buffers (registered via "
        "`register_buffer`) live on the same device as parameters after "
        "`.to(...)`, but a buffer with `dtype=t.long` (e.g. position ids) "
        "would give your scratch the wrong dtype. Parameters-first, with "
        "an empty-tensor fallback, is the right precedence."
    ),
    "extra_imports": [],
}


SPECS = [
    SPEC_LEAF,
    SPEC_LINSPACE,
    SPEC_OPT_REPR,
    SPEC_RSUB,
    SPEC_VIEW,
    SPEC_TRAINER,
    SPEC_COND_DROPOUT,
    SPEC_DEVICE_INFER,
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
    from torch import Tensor
    import einops
    from einops import rearrange, reduce, repeat

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"

        ns = {
            "t": t,
            "np": np,
            "nn": nn,
            "Tensor": Tensor,
            "einops": einops,
            "rearrange": rearrange,
            "reduce": reduce,
            "repeat": repeat,
            "_dd_passed": set(),
            "__name__": "__main__",
        }
        t.manual_seed(0)
        np.random.seed(0)

        try:
            exec(spec["stub"], ns)
        except Exception:
            pass

        try:
            exec(spec["solution_body"], ns)
            exec(spec["test_body"], ns)
        except Exception as e:
            failed.append((tag, repr(e), traceback.format_exc()))
            continue
        passed += 1
        print(f"  [verify] {tag}: ok")

    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, err, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(err)
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[deepening_w_batch12] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_w_batch12] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_w_batch12] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
