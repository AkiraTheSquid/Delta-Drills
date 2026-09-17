#!/usr/bin/env python3
"""Author 8 standalone Colab drills for final-cleanup misc atoms (batch-7).

Atoms covered (each drill = ONE LO + ONE Bloom level, max 2 KCs):

  any-reduce-axis              — 1 drill (ex1)  prereqs_misc_cleanup
  leaf-tensor-condition        — 1 drill (ex1)  prereqs_misc_cleanup
  rmul-scalar-tensor-mix       — 1 drill (ex1)  prereqs_misc_cleanup
  optimizer-repr-string        — 1 drill (ex1)  prereqs_misc_cleanup
  functional-module-wrap       — 1 drill (ex1)  prereqs_misc_cleanup
  tensor-reshape-view          — 1 drill (ex1)  prereqs_misc_cleanup
  linspace-out-param           — 1 drill (ex1)  prereqs_misc_cleanup
  trainer-subclass-extend      — 1 drill (ex1)  prereqs_misc_cleanup

These are smaller constituent skills folded out of ARENA chap-0 composite
exercises so each drill targets a single procedural KC.

Each spec is verified by re-running its solution against its test_body
inside the build venv (torch 2.12.0+cpu) before emission.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ---------------------------------------------------------------------------
# Per-atom recap blocks.
# ---------------------------------------------------------------------------

RECAP_ANY_REDUCE_AXIS = (
    "## Numpy: `any()` reduce along axis — quick refresher\n"
    "\n"
    "`mask.any(dim=k)` collapses axis `k` of a bool tensor by OR-ing every "
    "element along it. Result has the same shape as `mask` with axis `k` "
    "removed:\n"
    "\n"
    "```python\n"
    "mask = torch.tensor([[False, False, True],\n"
    "                     [False, False, False],\n"
    "                     [True,  True,  False]])    # shape (3, 3)\n"
    "mask.any(dim=1)   # tensor([ True, False,  True])   shape (3,)\n"
    "mask.any(dim=0)   # tensor([ True,  True,  True])   shape (3,)\n"
    "```\n"
    "\n"
    "**Read it as 'at least one True along axis k'.** Row 0 has a True → "
    "True. Row 1 is all False → False. Row 2 has a True → True. "
    "`dim=0` collapses ROWS (reduces down the columns); `dim=1` "
    "collapses COLUMNS (reduces across each row).\n"
    "\n"
    "**`numpy.any(arr, axis=k)` is the equivalent.** Same semantics, "
    "different keyword name (`axis=` vs `dim=`).\n"
    "\n"
    "**Use `keepdim=True` to preserve rank.** "
    "`mask.any(dim=1, keepdim=True)` returns `(3, 1)` instead of `(3,)` — "
    "lets you broadcast the reduce result back against the original.\n"
    "\n"
    "**`.all(dim=k)` is the AND-cousin.** True iff every element along "
    "axis `k` is True. Use it for 'no failures' checks; use `.any()` "
    "for 'at least one hit' checks."
)

RECAP_LEAF_TENSOR_CONDITION = (
    "## Backprop: leaf-tensor condition — quick refresher\n"
    "\n"
    "Autograd distinguishes LEAF tensors from INTERIOR tensors. A leaf "
    "is the BOUNDARY between user-supplied data and computed data:\n"
    "\n"
    "```python\n"
    "x = torch.randn(3, requires_grad=True)   # LEAF: user created\n"
    "y = x * 2                                 # INTERIOR: y.grad_fn = MulBackward\n"
    "z = y.sum()                               # INTERIOR\n"
    "x.is_leaf, y.is_leaf, z.is_leaf          # (True, False, False)\n"
    "```\n"
    "\n"
    "**The exact condition (PyTorch):** a tensor is a leaf iff it was "
    "NOT produced by an autograd-tracked operation. Equivalently, "
    "`t.grad_fn is None`. Every tensor you `torch.tensor(...)`, every "
    "`nn.Parameter`, every `torch.randn(...)` is a leaf.\n"
    "\n"
    "**For ARENA's manual-autograd MiniTensor:** the same condition "
    "translates directly. The wrapper stores a `.recipe` field that "
    "records the parent op + parents; `is_leaf` is just:\n"
    "\n"
    "```python\n"
    "@property\n"
    "def is_leaf(self):\n"
    "    return self.recipe is None\n"
    "```\n"
    "\n"
    "**Why this matters for `.backward()`.** Autograd accumulates "
    "gradients into `.grad` ONLY for leaf tensors that have "
    "`requires_grad=True`. Interior tensors compute gradients during "
    "the backward sweep but discard them after passing them to their "
    "parents — unless you call `.retain_grad()` to override that.\n"
    "\n"
    "**The trap.** `x = torch.randn(3, requires_grad=True).to('cuda')` "
    "is NOT a leaf — the `.to(...)` is an autograd op, so the result "
    "has a `grad_fn` and `.grad` will silently stay `None` after "
    "backward. Move-then-set-requires-grad is the safe order."
)

RECAP_RMUL_SCALAR_TENSOR_MIX = (
    "## PyTorch: `__rmul__` scalar/tensor mix — quick refresher\n"
    "\n"
    "When you write `3 * tensor`, Python first tries `int.__mul__(3, "
    "tensor)`. `int` doesn't know what a `Tensor` is, so it returns "
    "`NotImplemented`. Python then calls `tensor.__rmul__(3)` — the "
    "*reflected* multiply. That's how scalar-on-the-left works:\n"
    "\n"
    "```python\n"
    "class MiniTensor:\n"
    "    def __mul__(self, other):    return self._mul(other)   # tensor * x\n"
    "    def __rmul__(self, other):   return self._mul(other)   # x * tensor (x doesn't know us)\n"
    "```\n"
    "\n"
    "**Both directions MUST exist.** `tensor * 3` dispatches to "
    "`__mul__`. `3 * tensor` dispatches to `__rmul__`. If you only "
    "implement `__mul__`, the scalar-on-the-left form raises "
    "`TypeError`.\n"
    "\n"
    "**For multiplication, the two are usually symmetric.** "
    "`a * b == b * a` for scalar-tensor mixes, so `__rmul__` can "
    "simply delegate to `__mul__`. For NON-commutative ops "
    "(`__matmul__` / `__rmatmul__`, `__sub__` / `__rsub__`), the "
    "reflected version must SWAP the operand order: "
    "`__rsub__(self, other)` returns `other - self`, not `self - other`.\n"
    "\n"
    "**Why ARENA's MiniTensor needs both.** Tests will write expressions "
    "like `2 * x` (scalar literal first) and `x * 2` (tensor first) "
    "interchangeably. The framework cannot assume one ordering. "
    "Implement both to make every test pass.\n"
    "\n"
    "**The same applies to `__add__`/`__radd__`, `__sub__`/`__rsub__`, "
    "`__truediv__`/`__rtruediv__`, etc.** All the binary numeric "
    "dunders come in `__op__` + `__rop__` pairs."
)

RECAP_OPTIMIZER_REPR_STRING = (
    "## Optimizer: `__repr__` string — quick refresher\n"
    "\n"
    "Implementing `__repr__` gives your optimizer a debug-friendly "
    "string in tracebacks, REPL printouts, and log lines. The "
    "convention is `ClassName(arg1=value1, arg2=value2)` — looks "
    "like the constructor that would re-create it:\n"
    "\n"
    "```python\n"
    "class SGD:\n"
    "    def __init__(self, params, lr, momentum=0.0):\n"
    "        self.params = list(params)\n"
    "        self.lr = lr\n"
    "        self.momentum = momentum\n"
    "\n"
    "    def __repr__(self):\n"
    "        return f'SGD(lr={self.lr}, momentum={self.momentum})'\n"
    "\n"
    ">>> opt = SGD(model.parameters(), lr=1e-3, momentum=0.9)\n"
    ">>> opt\n"
    "SGD(lr=0.001, momentum=0.9)\n"
    "```\n"
    "\n"
    "**`__repr__` vs `__str__`.** `__repr__` is for developers — "
    "unambiguous, ideally `eval`-able. `__str__` is for end users — "
    "pretty-printed. When `__str__` is not defined, `str(x)` falls back "
    "to `__repr__`. For optimizers, only `__repr__` is needed.\n"
    "\n"
    "**Why include hyperparameters, not the full param list.** The "
    "param tensors are bulky and uninformative in a debug print. "
    "Hyperparameters (`lr`, `momentum`, `weight_decay`) are what you "
    "actually want to see when checking that the optimizer was "
    "constructed correctly.\n"
    "\n"
    "**PyTorch's own optimizer repr.** `torch.optim.SGD` prints a "
    "multi-line repr showing each param group's hparams. The pattern "
    "is the same — show config, hide bulk data.\n"
    "\n"
    "**Use f-strings, not `%` formatting.** Modern, readable, and "
    "evaluates `self.lr` etc. inline."
)

RECAP_FUNCTIONAL_MODULE_WRAP = (
    "## PyTorch: functional vs Module wrap — quick refresher\n"
    "\n"
    "Every common neural-net building block exists in TWO forms in "
    "PyTorch: a stateless functional version (`torch.nn.functional`) "
    "and a stateful Module wrapper (`torch.nn`):\n"
    "\n"
    "```python\n"
    "import torch.nn.functional as F\n"
    "import torch.nn as nn\n"
    "\n"
    "# Functional — call once, no state, no params.\n"
    "y = F.relu(x)\n"
    "y = F.linear(x, weight, bias)\n"
    "y = F.dropout(x, p=0.5, training=True)\n"
    "\n"
    "# Module — instantiate, owns params/buffers, integrates with nn.Module.\n"
    "y = nn.ReLU()(x)\n"
    "y = nn.Linear(in_features=10, out_features=4)(x)\n"
    "y = nn.Dropout(p=0.5)(x)\n"
    "```\n"
    "\n"
    "**When to use `F.`** Stateless ops (`relu`, `softmax`, `gelu`, "
    "`cross_entropy`, `pad`). No params. No training/eval mode "
    "switching. You just need the function applied.\n"
    "\n"
    "**When to use `nn.`** When you need:\n"
    "- Parameters tracked by `model.parameters()` (e.g. `nn.Linear`).\n"
    "- Train/eval mode switching (`nn.Dropout`, `nn.BatchNorm2d`).\n"
    "- Composition with `nn.Sequential`.\n"
    "- `state_dict` serialization.\n"
    "\n"
    "**The Module is a thin wrapper over the functional.** "
    "`nn.ReLU().forward(x)` is literally `F.relu(x)`. The wrapper just "
    "adds `nn.Module` registration so it shows up in printouts and "
    "state dicts.\n"
    "\n"
    "**The 'dropout trap'.** `F.dropout(x, p=0.5)` ALWAYS applies "
    "dropout — there's no implicit train/eval mode. You must pass "
    "`training=self.training` or call from inside an `nn.Dropout` "
    "module which handles the flag for you."
)

RECAP_TENSOR_RESHAPE_VIEW = (
    "## PyTorch: `.reshape()` vs `.view()` — quick refresher\n"
    "\n"
    "Both change shape without changing data. They differ on what "
    "happens when the source is NOT contiguous in memory:\n"
    "\n"
    "```python\n"
    "x = torch.arange(12).reshape(3, 4)\n"
    "x.is_contiguous()         # True\n"
    "x.view(4, 3)              # OK — same storage, new strides\n"
    "x.reshape(4, 3)           # OK — same storage, new strides\n"
    "\n"
    "y = x.T                    # transpose; y.is_contiguous() == False\n"
    "y.view(12)                 # RuntimeError: view requires contiguous\n"
    "y.reshape(12)              # OK — silently copies into a new contiguous tensor\n"
    "```\n"
    "\n"
    "**`.view()` is strict.** Requires the source to be contiguous. "
    "Returns a view (shares storage). Cheap (O(1)). Raises "
    "`RuntimeError` if the source isn't contiguous.\n"
    "\n"
    "**`.reshape()` is forgiving.** First tries to return a view "
    "(O(1)). If that's not possible (non-contiguous source), it "
    "silently makes a contiguous copy (O(n)). Always succeeds for "
    "shape-compatible targets.\n"
    "\n"
    "**Decision tree.**\n"
    "- Need to GUARANTEE no copy → `.view()` (and call "
    "`.contiguous()` first if you're not sure).\n"
    "- Don't care about the copy → `.reshape()` (default in most "
    "code).\n"
    "- Want explicit copy → `.reshape().clone()` or "
    "`.contiguous().view()`.\n"
    "\n"
    "**Why ARENA's stride exercises insist on `.view()`.** The "
    "exercise is teaching stride math — accidentally copying defeats "
    "the lesson. In application code, `.reshape()` is the safer "
    "default."
)

RECAP_LINSPACE_OUT_PARAM = (
    "## PyTorch: `linspace(out=)` param — quick refresher\n"
    "\n"
    "Most tensor-creation ops in PyTorch accept an `out=` kwarg that "
    "writes the result into a PRE-ALLOCATED tensor instead of "
    "allocating a fresh one:\n"
    "\n"
    "```python\n"
    "pre = torch.empty(11)             # allocate once, reuse forever\n"
    "torch.linspace(0, 1, 11, out=pre)  # fills pre in place; returns pre\n"
    "```\n"
    "\n"
    "**Why this exists.** Inner loops that need a fresh buffer every "
    "iteration would otherwise pay the cost of allocation + "
    "deallocation N times. With `out=`, the loop allocates once and "
    "writes-through every iteration:\n"
    "\n"
    "```python\n"
    "buf = torch.empty(N)\n"
    "for t_max in schedule:\n"
    "    torch.linspace(0, t_max, N, out=buf)   # zero-alloc\n"
    "    do_something_with(buf)\n"
    "```\n"
    "\n"
    "**Contract.** The `out` tensor must have the right shape and "
    "dtype (or PyTorch will resize it, which defeats the purpose). "
    "The function modifies `out` in place AND returns it — so you "
    "can chain: `y = torch.linspace(0, 1, 11, out=buf).pow(2)`.\n"
    "\n"
    "**Same kwarg on every creation op.** `torch.zeros(... out=)`, "
    "`torch.arange(..., out=)`, `torch.randn(..., out=)`, "
    "`torch.empty_like(..., out=)`. Same semantics: in-place write, "
    "must be pre-sized.\n"
    "\n"
    "**Don't confuse with `_inplace` (`fill_`, `zero_`).** Those are "
    "method-on-an-existing-tensor (`x.fill_(0)`). `out=` is a kwarg "
    "to a free function. Both achieve the same effect (modify in "
    "place) via different APIs."
)

RECAP_TRAINER_SUBCLASS_EXTEND = (
    "## Trainer: subclass-extend pattern — quick refresher\n"
    "\n"
    "When you want to ADD behavior to a base trainer's `_step` "
    "(per-batch hook) without rewriting it, override + `super()`:\n"
    "\n"
    "```python\n"
    "class BaseTrainer:\n"
    "    def _step(self, batch):\n"
    "        x, y = batch\n"
    "        out = self.model(x)\n"
    "        loss = self.loss_fn(out, y)\n"
    "        loss.backward()\n"
    "        self.optimizer.step()\n"
    "        self.optimizer.zero_grad()\n"
    "        return {'loss': loss.item()}\n"
    "\n"
    "class MyTrainer(BaseTrainer):\n"
    "    def _step(self, batch):\n"
    "        out = super()._step(batch)        # run the base step first\n"
    "        out['my_extra_metric'] = self._compute_extra(batch)  # add to it\n"
    "        return out\n"
    "```\n"
    "\n"
    "**Three rules of the extend pattern.**\n"
    "1. Call `super()._step(batch)` FIRST so the base does its work.\n"
    "2. ADD to the result (don't replace it) — extend the dict.\n"
    "3. RETURN the extended result — callers expect the same shape "
    "the base returned, plus your additions.\n"
    "\n"
    "**Why not write a new `_step` from scratch.** Duplicating the "
    "base body works, but every fix to `BaseTrainer._step` would "
    "need to be re-applied to your subclass. The `super()._step()` "
    "call inherits ALL future fixes for free.\n"
    "\n"
    "**The 'before / after' variations.**\n"
    "- BEFORE: `self._pre_hook(); out = super()._step(batch); return out` — run a hook, then delegate.\n"
    "- AFTER:  `out = super()._step(batch); self._post_hook(out); return out` — delegate, then a hook.\n"
    "- AROUND: do both — the most common shape for logging.\n"
    "\n"
    "**Compose, don't replace.** If you find yourself NOT calling "
    "`super()._step()` in an override, you're not extending — "
    "you're replacing. Make a new class instead so the inheritance "
    "tells the right story."
)


# ---------------------------------------------------------------------------
# Specs.
# ---------------------------------------------------------------------------

SPECS = [

    # =========================================================
    # any-reduce-axis — ex1
    # =========================================================
    {
        "atom_id": "any-reduce-axis",
        "subtopic": "Numpy: any() reduce along axis",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_ANY_REDUCE_AXIS,
        "exercise_index": 1,
        "exercise_title": "row-wise any() to flag rows containing any True",
        "slug": "row-wise-any-to-flag-rows-containing-any-true",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["any", "reduce", "axis", "bool-mask"],
        "kcs": [
            "any-reduce-axis-collapse",
            "axis-direction-convention",
        ],
        "lo": (
            "Apply `.any(dim=k)` to collapse one axis of a 2-D bool "
            "tensor, distinguishing `dim=0` (per-column) from `dim=1` "
            "(per-row) collapses."
        ),
        "prompt_body": (
            "Implement `ex1_row_has_true(mask)`. Take an `(N, M)` "
            "boolean tensor and return an `(N,)` boolean tensor where "
            "entry `i` is `True` iff row `i` of `mask` contains AT "
            "LEAST ONE `True`.\n\n"
            "Inputs:\n"
            "- `mask`: `(N, M)` bool tensor.\n\n"
            "Output: `(N,)` bool tensor.\n\n"
            "Constraints:\n"
            "- Use `.any(dim=...)` — do NOT write a Python loop.\n"
            "- Output dtype must be `torch.bool`.\n"
            "- Output shape must be `(N,)` (NOT `(N, 1)`); do not pass "
            "`keepdim=True`."
        ),
        "stub": (
            "def ex1_row_has_true(mask: Tensor) -> Tensor:\n"
            '    """(N, M) bool -> (N,) bool: True iff row has at least one True."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Hand-traced reference ===\n"
            "mask = t.tensor([\n"
            "    [False, False, True ],   # row 0: has True\n"
            "    [False, False, False],   # row 1: all False\n"
            "    [True,  True,  False],   # row 2: has True\n"
            "    [False, True,  False],   # row 3: has True\n"
            "])\n"
            "out = ex1_row_has_true(mask)\n"
            "expected = t.tensor([True, False, True, True])\n"
            "assert out.dtype == t.bool, f'expected bool, got {out.dtype}'\n"
            "assert out.shape == (4,), f'expected shape (4,), got {tuple(out.shape)}'\n"
            "assert t.equal(out, expected), f'expected {expected}, got {out}'\n"
            "\n"
            "# === All-True input -> all True ===\n"
            "all_true = t.ones(5, 3, dtype=t.bool)\n"
            "assert t.equal(ex1_row_has_true(all_true), t.ones(5, dtype=t.bool))\n"
            "\n"
            "# === All-False input -> all False ===\n"
            "all_false = t.zeros(5, 3, dtype=t.bool)\n"
            "assert t.equal(ex1_row_has_true(all_false), t.zeros(5, dtype=t.bool))\n"
            "\n"
            "# === Single-column input -> just that column's values ===\n"
            "col = t.tensor([[True], [False], [True]])\n"
            "assert t.equal(ex1_row_has_true(col), t.tensor([True, False, True]))\n"
            "\n"
            "# === The axis convention check ===\n"
            "# If a student used dim=0 by mistake on a (3, 5) input, they would\n"
            "# get a length-5 vector. The output length must equal N (the first dim).\n"
            "rect = t.tensor([\n"
            "    [True,  False, False, False, False],   # row 0: has True\n"
            "    [False, False, False, False, False],   # row 1: all False\n"
            "    [False, False, False, False, True ],   # row 2: has True\n"
            "])\n"
            "out_rect = ex1_row_has_true(rect)\n"
            "assert out_rect.shape == (3,), (\n"
            "    f'output should have length N=3 (first dim), got {tuple(out_rect.shape)} '\n"
            "    '— did you use dim=0 by mistake?'\n"
            ")\n"
            "assert t.equal(out_rect, t.tensor([True, False, True]))\n"
            "\n"
            "# === Larger random check vs reference Python loop ===\n"
            "rng = t.Generator().manual_seed(0)\n"
            "big = t.randint(0, 2, (32, 64), generator=rng).bool()\n"
            "ref = t.tensor([row.any().item() for row in big])\n"
            "assert t.equal(ex1_row_has_true(big), ref)"
        ),
        "solution_body": (
            "def ex1_row_has_true(mask):\n"
            "    return mask.any(dim=1)"
        ),
        "solution_notes": (
            "**One line, one axis.** `mask.any(dim=1)` collapses axis "
            "1 (the columns) and returns one boolean per row.\n\n"
            "**Why not `dim=0`.** `dim=0` collapses ROWS, giving you "
            "ONE boolean per column. That's the opposite question — "
            "'does this column contain any True'.\n\n"
            "**Why not `dim=-1`.** It works (last axis == columns for "
            "2-D), but `dim=1` is more explicit when you know the "
            "tensor is 2-D. Use `dim=-1` for code that works across "
            "arbitrary ranks.\n\n"
            "**Generalizes.** `.all()`, `.sum()`, `.max()`, `.min()`, "
            "`.mean()` all share the same `dim=` API — same collapse "
            "rules, different reduction op."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # leaf-tensor-condition — ex1
    # =========================================================
    {
        "atom_id": "leaf-tensor-condition",
        "subtopic": "Backprop: leaf tensor condition",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_LEAF_TENSOR_CONDITION,
        "exercise_index": 1,
        "exercise_title": "classify tensors as leaf vs interior via the grad_fn condition",
        "slug": "classify-tensors-as-leaf-vs-interior-via-the-grad-fn-condition",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["leaf", "is-leaf", "grad-fn", "autograd"],
        "kcs": [
            "leaf-iff-no-grad-fn",
            "user-created-vs-computed-distinction",
        ],
        "lo": (
            "Apply the leaf-tensor condition (`grad_fn is None`) to "
            "classify each tensor in a small graph as leaf vs interior, "
            "matching PyTorch's `.is_leaf` semantics."
        ),
        "prompt_body": (
            "Implement `ex1_classify_leaf(tensors)`. Take a list of "
            "`Tensor`s and return a list of bools, where entry `i` is "
            "`True` iff `tensors[i]` is a LEAF — i.e. it was not "
            "produced by an autograd-tracked operation.\n\n"
            "Inputs:\n"
            "- `tensors`: `list[Tensor]`.\n\n"
            "Output: `list[bool]` of the same length.\n\n"
            "Definition (matches PyTorch):\n"
            "- A tensor is a leaf iff `tensor.grad_fn is None`.\n"
            "- Do NOT use `.is_leaf` directly — implement the check "
            "from first principles (`grad_fn`).\n\n"
            "This drill exercises the same condition ARENA's "
            "MiniTensor wrapper uses (`is_leaf = (self.recipe is "
            "None)`) — the manual-autograd analogue of `grad_fn`."
        ),
        "stub": (
            "def ex1_classify_leaf(tensors: list) -> list:\n"
            '    """Return list[bool]: True iff the tensor is a leaf (no grad_fn)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === User-created tensors are leaves ===\n"
            "a = t.tensor([1.0, 2.0])\n"
            "b = t.randn(3, requires_grad=True)\n"
            "c = t.zeros(4)\n"
            "assert ex1_classify_leaf([a, b, c]) == [True, True, True]\n"
            "\n"
            "# === Computed tensors are NOT leaves ===\n"
            "x = t.randn(3, requires_grad=True)\n"
            "y = x * 2          # MulBackward\n"
            "z = y.sum()        # SumBackward\n"
            "w = x + x          # AddBackward\n"
            "# x itself is a leaf, y/z/w are not.\n"
            "assert ex1_classify_leaf([x, y, z, w]) == [True, False, False, False]\n"
            "\n"
            "# === A no-op clone of a requires_grad=True tensor is NOT a leaf ===\n"
            "p = t.randn(2, requires_grad=True)\n"
            "p_clone = p.clone()    # CloneBackward\n"
            "assert ex1_classify_leaf([p, p_clone]) == [True, False]\n"
            "\n"
            "# === An nn.Parameter is a leaf ===\n"
            "param = t.nn.Parameter(t.randn(3))\n"
            "assert ex1_classify_leaf([param]) == [True]\n"
            "\n"
            "# === Detached tensors are leaves ===\n"
            "x = t.randn(3, requires_grad=True)\n"
            "x_detached = x.detach()\n"
            "# x is a leaf (user-created), x_detached is also a leaf (detach severs grad_fn).\n"
            "assert ex1_classify_leaf([x, x_detached]) == [True, True]\n"
            "\n"
            "# === Mixed graph: classify all in one call ===\n"
            "a = t.tensor([1.0], requires_grad=True)\n"
            "b = t.tensor([2.0], requires_grad=True)\n"
            "c = a + b            # AddBackward\n"
            "d = c * a            # MulBackward\n"
            "e = t.tensor([3.0])  # leaf, no requires_grad\n"
            "expected = [True, True, False, False, True]\n"
            "assert ex1_classify_leaf([a, b, c, d, e]) == expected\n"
            "\n"
            "# === Cross-check against PyTorch's .is_leaf attribute ===\n"
            "many = [\n"
            "    t.randn(2),\n"
            "    t.randn(2, requires_grad=True),\n"
            "    t.randn(2, requires_grad=True) * 3,\n"
            "    t.randn(2) + 1.0,\n"
            "    t.nn.Parameter(t.randn(2)),\n"
            "]\n"
            "ours = ex1_classify_leaf(many)\n"
            "torch_ref = [x.is_leaf for x in many]\n"
            "assert ours == torch_ref, f'mismatch with PyTorch: ours={ours} ref={torch_ref}'"
        ),
        "solution_body": (
            "def ex1_classify_leaf(tensors):\n"
            "    return [t.grad_fn is None for t in tensors]"
        ),
        "solution_notes": (
            "**The whole condition is `grad_fn is None`.** Every "
            "autograd-tracked op (`+`, `*`, `clone`, `to`, etc.) "
            "attaches a `grad_fn` to the result. A tensor without one "
            "wasn't produced by any tracked op — it's at the boundary "
            "of the graph.\n\n"
            "**Why `requires_grad` is NOT the leaf check.** Both "
            "`a = t.tensor([1.0])` (no grad) and `b = t.tensor([1.0], "
            "requires_grad=True)` are leaves. `requires_grad` "
            "determines whether autograd accumulates grad INTO `.grad` "
            "on backward; it doesn't change the leaf/interior "
            "classification.\n\n"
            "**For ARENA's MiniTensor.** Replace `t.grad_fn` with "
            "`t.recipe` and the function becomes the manual-autograd "
            "version: `return [tensor.recipe is None for tensor in "
            "tensors]`. Same semantics — the recipe records WHERE the "
            "tensor came from, so `recipe is None` means user-created."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # rmul-scalar-tensor-mix — ex1
    # =========================================================
    {
        "atom_id": "rmul-scalar-tensor-mix",
        "subtopic": "PyTorch: __rmul__ scalar/tensor mix",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_RMUL_SCALAR_TENSOR_MIX,
        "exercise_index": 1,
        "exercise_title": "implement __mul__ and __rmul__ so both 2*t and t*2 work",
        "slug": "implement-mul-and-rmul-so-both-2-times-t-and-t-times-2-work",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["dunder", "rmul", "reflected-op", "scalar-tensor"],
        "kcs": [
            "rmul-implements-reflected-multiply",
            "mul-rmul-symmetry-for-commutative-op",
        ],
        "lo": (
            "Apply the `__mul__` / `__rmul__` dunder pair to a custom "
            "tensor wrapper so scalar*tensor and tensor*scalar both "
            "dispatch correctly, exploiting the commutativity of "
            "multiplication."
        ),
        "prompt_body": (
            "Complete the `Wrapper` class below by implementing "
            "`__mul__` and `__rmul__` so that both `2 * w` (scalar on "
            "the left) and `w * 2` (scalar on the right) return a "
            "new `Wrapper` containing `2 * w.data`.\n\n"
            "Inputs:\n"
            "- `self.data` is a `torch.Tensor` stored on the wrapper.\n"
            "- The other operand is a Python scalar (int or float).\n\n"
            "Output: a NEW `Wrapper` whose `.data` is the elementwise "
            "product. Do not mutate `self`.\n\n"
            "Constraints:\n"
            "- BOTH `__mul__` and `__rmul__` must be implemented.\n"
            "- `__rmul__` may delegate to `__mul__` (multiplication "
            "is commutative for scalar/tensor mixes).\n"
            "- Return a `Wrapper`, NOT a raw `Tensor`."
        ),
        "stub": (
            "class Wrapper:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n"
            "\n"
            "    def __mul__(self, other):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def __rmul__(self, other):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def __repr__(self):\n"
            "        return f'Wrapper({self.data.tolist()})'"
        ),
        "test_body": (
            "# === tensor * scalar (left-multiply) ===\n"
            "w = Wrapper(t.tensor([1.0, 2.0, 3.0]))\n"
            "out = w * 2\n"
            "assert isinstance(out, Wrapper), f'expected Wrapper, got {type(out).__name__}'\n"
            "assert t.allclose(out.data, t.tensor([2.0, 4.0, 6.0])), f'got {out.data}'\n"
            "\n"
            "# === scalar * tensor (right-multiply, dispatches to __rmul__) ===\n"
            "out = 2 * w\n"
            "assert isinstance(out, Wrapper), f'expected Wrapper from scalar*w, got {type(out).__name__}'\n"
            "assert t.allclose(out.data, t.tensor([2.0, 4.0, 6.0])), f'got {out.data}'\n"
            "\n"
            "# === Both directions produce the same result (commutativity) ===\n"
            "for scalar in [-1.0, 0.0, 0.5, 3, 7.5]:\n"
            "    left = scalar * w\n"
            "    right = w * scalar\n"
            "    assert t.allclose(left.data, right.data), (\n"
            "        f'asymmetric for scalar={scalar}: left={left.data} right={right.data}'\n"
            "    )\n"
            "    assert isinstance(left, Wrapper)\n"
            "    assert isinstance(right, Wrapper)\n"
            "\n"
            "# === Does NOT mutate self ===\n"
            "original = t.tensor([1.0, 2.0, 3.0])\n"
            "w = Wrapper(original.clone())\n"
            "_ = 5 * w\n"
            "assert t.allclose(w.data, original), 'rmul should not mutate self'\n"
            "_ = w * 5\n"
            "assert t.allclose(w.data, original), 'mul should not mutate self'\n"
            "\n"
            "# === Confirm __rmul__ is actually being called (proves the dispatch path) ===\n"
            "# Python only calls __rmul__ when the left operand's __mul__ returned NotImplemented\n"
            "# or doesn't know how to handle the right operand. int.__mul__(2, Wrapper) returns\n"
            "# NotImplemented, so Python falls back to Wrapper.__rmul__(2). If __rmul__ were\n"
            "# missing, 2 * w would raise TypeError.\n"
            "w = Wrapper(t.tensor([10.0]))\n"
            "try:\n"
            "    result = 3 * w\n"
            "except TypeError as e:\n"
            "    raise AssertionError(\n"
            "        f'scalar * Wrapper raised TypeError — did you forget __rmul__? ({e})'\n"
            "    ) from None\n"
            "assert t.allclose(result.data, t.tensor([30.0]))\n"
            "\n"
            "# === Larger tensor smoke test ===\n"
            "big = t.randn(64, generator=t.Generator().manual_seed(0))\n"
            "w = Wrapper(big.clone())\n"
            "out_left = 1.5 * w\n"
            "out_right = w * 1.5\n"
            "assert t.allclose(out_left.data, out_right.data)\n"
            "assert t.allclose(out_left.data, big * 1.5)"
        ),
        "solution_body": (
            "class Wrapper:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n"
            "\n"
            "    def __mul__(self, other):\n"
            "        return Wrapper(self.data * other)\n"
            "\n"
            "    def __rmul__(self, other):\n"
            "        # Multiplication is commutative for scalar/tensor mixes, so we can\n"
            "        # safely delegate to __mul__. For NON-commutative ops (like __sub__),\n"
            "        # __rsub__ would need to compute other - self.data instead.\n"
            "        return self.__mul__(other)\n"
            "\n"
            "    def __repr__(self):\n"
            "        return f'Wrapper({self.data.tolist()})'"
        ),
        "solution_notes": (
            "**`__rmul__` makes scalar-on-the-left work.** Python "
            "tries `int.__mul__(3, wrapper)` first; `int` returns "
            "`NotImplemented`; Python then calls "
            "`wrapper.__rmul__(3)`. Without `__rmul__`, `3 * wrapper` "
            "raises `TypeError`.\n\n"
            "**Delegation is fine for commutative ops.** "
            "`self.__mul__(other)` works because `a * b == b * a` for "
            "scalar/tensor multiplication. For `__rsub__`, you'd need "
            "`return Wrapper(other - self.data)` — the operand swap "
            "is essential.\n\n"
            "**The same pair applies to `__add__`/`__radd__`, "
            "`__truediv__`/`__rtruediv__`, `__matmul__`/`__rmatmul__`, "
            "etc.** Every binary numeric dunder has a reflected "
            "counterpart. ARENA's MiniTensor implements ALL of them.\n\n"
            "**Return a new `Wrapper`, never a raw `Tensor`.** "
            "Otherwise chains like `(2 * w) * 3` break — the "
            "intermediate would lose its wrapper identity."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # optimizer-repr-string — ex1
    # =========================================================
    {
        "atom_id": "optimizer-repr-string",
        "subtopic": "Optimizer: __repr__ string",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_OPTIMIZER_REPR_STRING,
        "exercise_index": 1,
        "exercise_title": "give a hand-rolled SGD a debug-friendly __repr__",
        "slug": "give-a-hand-rolled-sgd-a-debug-friendly-repr",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["repr", "dunder", "optimizer", "debug"],
        "kcs": [
            "repr-returns-constructor-like-string",
            "repr-includes-hparams-excludes-bulk",
        ],
        "lo": (
            "Apply `__repr__` to a custom optimizer class so its "
            "string form lists hyperparameters in a "
            "ClassName(arg=value, ...) shape and `repr(opt)` returns "
            "the same."
        ),
        "prompt_body": (
            "Complete the `SGD` class below by implementing "
            "`__repr__` so it returns the string:\n\n"
            "    'SGD(lr=<lr>, momentum=<momentum>)'\n\n"
            "where `<lr>` and `<momentum>` are the current attribute "
            "values, formatted using their default `repr()` (i.e. "
            "f-string `{self.lr}` and `{self.momentum}`).\n\n"
            "Inputs / state:\n"
            "- `self.lr`: float\n"
            "- `self.momentum`: float\n"
            "- `self.params`: list (the actual parameter tensors — do "
            "NOT include them in the repr)\n\n"
            "Constraints:\n"
            "- The output is a single line.\n"
            "- The format is exactly `SGD(lr=<lr>, momentum=<momentum>)` "
            "with one space after the comma (matches Python's default "
            "f-string formatting).\n"
            "- `repr(opt)` and `str(opt)` must both produce the same "
            "string (when `__str__` is not defined, it falls back to "
            "`__repr__`)."
        ),
        "stub": (
            "class SGD:\n"
            "    def __init__(self, params, lr, momentum=0.0):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "        self.momentum = momentum\n"
            "\n"
            "    def __repr__(self):\n"
            "        raise NotImplementedError()"
        ),
        "test_body": (
            "# === Basic case ===\n"
            "params = [t.zeros(10), t.zeros(20)]\n"
            "opt = SGD(params, lr=0.001, momentum=0.9)\n"
            "r = repr(opt)\n"
            "assert isinstance(r, str), f'__repr__ must return str, got {type(r).__name__}'\n"
            "assert r == 'SGD(lr=0.001, momentum=0.9)', f'expected exact format, got {r!r}'\n"
            "\n"
            "# === str() falls back to __repr__ when __str__ is undefined ===\n"
            "assert str(opt) == repr(opt), 'str(opt) and repr(opt) must agree'\n"
            "\n"
            "# === Different values ===\n"
            "opt2 = SGD([t.zeros(1)], lr=0.1, momentum=0.0)\n"
            "assert repr(opt2) == 'SGD(lr=0.1, momentum=0.0)', f'got {repr(opt2)!r}'\n"
            "\n"
            "# === Repr does NOT include the params list (would be huge) ===\n"
            "big_opt = SGD([t.randn(10_000)], lr=1e-4, momentum=0.99)\n"
            "r = repr(big_opt)\n"
            "assert 'tensor' not in r.lower(), (\n"
            "    f'repr should NOT include the param tensors, got {r!r}'\n"
            ")\n"
            "assert len(r) < 100, f'repr is way too long, got {len(r)} chars: {r!r}'\n"
            "\n"
            "# === Class name is in the repr (so tracebacks identify the type) ===\n"
            "assert r.startswith('SGD('), f'repr should start with class name, got {r!r}'\n"
            "\n"
            "# === Mutating an attribute changes the repr live ===\n"
            "opt = SGD([t.zeros(3)], lr=1e-3, momentum=0.5)\n"
            "r1 = repr(opt)\n"
            "opt.lr = 1e-5\n"
            "r2 = repr(opt)\n"
            "assert r1 != r2, 'repr should reflect current attribute values'\n"
            "assert 'lr=1e-05' in r2, f'expected updated lr to show, got {r2!r}'\n"
            "\n"
            "# === Both hparams appear ===\n"
            "opt = SGD([t.zeros(1)], lr=0.5, momentum=0.5)\n"
            "r = repr(opt)\n"
            "assert 'lr=' in r and 'momentum=' in r, f'both hparams must appear, got {r!r}'"
        ),
        "solution_body": (
            "class SGD:\n"
            "    def __init__(self, params, lr, momentum=0.0):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "        self.momentum = momentum\n"
            "\n"
            "    def __repr__(self):\n"
            "        return f'SGD(lr={self.lr}, momentum={self.momentum})'"
        ),
        "solution_notes": (
            "**Constructor-mirror format.** "
            "`SGD(lr=0.001, momentum=0.9)` reads like the call that "
            "would re-create the object — ideal for debugging and "
            "logging. PyTorch's own `torch.optim.Optimizer.__repr__` "
            "follows the same pattern (just multi-line because of "
            "param groups).\n\n"
            "**Hparams in, params out.** The `self.params` list "
            "could be megabytes for a real model. Hyperparameters are "
            "what you actually want to see in a print statement; "
            "weights belong in `state_dict`, not `repr`.\n\n"
            "**`__str__` falls back to `__repr__`.** When `__str__` "
            "isn't defined, `str(x)` returns `repr(x)`. That's why "
            "the test asserts `str(opt) == repr(opt)` — you got both "
            "for free.\n\n"
            "**f-string formatting matches Python's defaults.** "
            "`f'{self.lr}'` formats a float as Python's default "
            "(e.g. `0.001`, `1e-05`). Don't try to special-case "
            "scientific vs decimal; let Python decide."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # functional-module-wrap — ex1
    # =========================================================
    {
        "atom_id": "functional-module-wrap",
        "subtopic": "PyTorch: functional module wrap",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_FUNCTIONAL_MODULE_WRAP,
        "exercise_index": 1,
        "exercise_title": "wrap F.relu in an nn.Module to make it composable",
        "slug": "wrap-f-relu-in-an-nn-module-to-make-it-composable",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["functional", "module", "relu", "compose"],
        "kcs": [
            "functional-module-equivalence",
            "module-wrap-stateless-fn",
        ],
        "lo": (
            "Apply the functional-to-Module wrap pattern: implement a "
            "stateless `nn.Module` whose `forward` simply calls "
            "`F.relu`, matching `nn.ReLU` semantically."
        ),
        "prompt_body": (
            "Implement `MyReLU(nn.Module)` whose `forward(x)` returns "
            "`F.relu(x)`. This is exactly what `nn.ReLU` does "
            "internally — wrap a stateless functional in an "
            "`nn.Module` so it composes with `nn.Sequential` and "
            "shows up in `state_dict`/`model.modules()` printouts.\n\n"
            "Constraints:\n"
            "- Subclass `nn.Module`.\n"
            "- Call `super().__init__()` in your `__init__`.\n"
            "- `forward(x)` must return `F.relu(x)` — do NOT "
            "instantiate `nn.ReLU` and delegate (defeats the point).\n"
            "- No parameters — the module is stateless.\n\n"
            "Output: an `nn.Module` subclass that behaves identically "
            "to `nn.ReLU` on every input."
        ),
        "stub": (
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "class MyReLU(nn.Module):\n"
            "    def __init__(self):\n"
            "        raise NotImplementedError()\n"
            "\n"
            "    def forward(self, x):\n"
            "        raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "# === Identical to nn.ReLU on positive / negative / zero ===\n"
            "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])\n"
            "my = MyReLU()\n"
            "torch_ref = nn.ReLU()\n"
            "assert t.allclose(my(x), torch_ref(x)), (\n"
            "    f'MyReLU output should match nn.ReLU. got {my(x)}, ref {torch_ref(x)}'\n"
            ")\n"
            "\n"
            "# === Identical to F.relu directly ===\n"
            "assert t.allclose(my(x), F.relu(x))\n"
            "\n"
            "# === Subclass check ===\n"
            "assert isinstance(my, nn.Module), 'MyReLU must subclass nn.Module'\n"
            "\n"
            "# === No parameters (stateless) ===\n"
            "params = list(my.parameters())\n"
            "assert params == [], f'MyReLU should have no params, got {params}'\n"
            "buffers = list(my.buffers())\n"
            "assert buffers == [], f'MyReLU should have no buffers, got {buffers}'\n"
            "\n"
            "# === Composes inside nn.Sequential ===\n"
            "net = nn.Sequential(\n"
            "    nn.Linear(4, 3),\n"
            "    MyReLU(),\n"
            "    nn.Linear(3, 2),\n"
            ")\n"
            "x = t.randn(8, 4)\n"
            "out = net(x)\n"
            "assert out.shape == (8, 2)\n"
            "# Equivalent with stock nn.ReLU.\n"
            "t.manual_seed(42)\n"
            "net1 = nn.Sequential(nn.Linear(4, 3), MyReLU(), nn.Linear(3, 2))\n"
            "t.manual_seed(42)\n"
            "net2 = nn.Sequential(nn.Linear(4, 3), nn.ReLU(),  nn.Linear(3, 2))\n"
            "assert t.allclose(net1(x), net2(x), atol=1e-6), (\n"
            "    'composed network with MyReLU must match composed network with nn.ReLU'\n"
            ")\n"
            "\n"
            "# === Gradient flows through correctly (autograd preserved) ===\n"
            "x = t.randn(5, requires_grad=True)\n"
            "y = MyReLU()(x).sum()\n"
            "y.backward()\n"
            "# Gradient of relu(x).sum() w.r.t. x is 1 where x>0, 0 elsewhere.\n"
            "expected_grad = (x.detach() > 0).float()\n"
            "assert t.allclose(x.grad, expected_grad), (\n"
            "    f'gradient mismatch: expected {expected_grad}, got {x.grad}'\n"
            ")\n"
            "\n"
            "# === Large input smoke test ===\n"
            "big = t.randn(1024, 64, generator=t.Generator().manual_seed(0))\n"
            "assert t.allclose(MyReLU()(big), F.relu(big))\n"
            "\n"
            "# === forward must use F.relu, not nn.ReLU delegation ===\n"
            "# Quick proxy check: MyReLU should have no child modules.\n"
            "children = list(my.children())\n"
            "assert children == [], (\n"
            "    f'MyReLU should not delegate to nn.ReLU; expected no children, got {children}'\n"
            ")"
        ),
        "solution_body": (
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "class MyReLU(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "\n"
            "    def forward(self, x):\n"
            "        return F.relu(x)"
        ),
        "solution_notes": (
            "**Three lines, full Module integration.** "
            "`super().__init__()` registers the module with the "
            "`nn.Module` machinery (children, parameters, "
            "state_dict). `forward` delegates to the functional.\n\n"
            "**Why a class for a stateless function.** The class form "
            "buys you `nn.Sequential` composition, automatic mode "
            "propagation (train/eval), and visibility in `model` "
            "printouts. The functional form is a flat function — "
            "great for one-off use inside a custom `forward`.\n\n"
            "**`nn.ReLU`'s actual implementation** is essentially what "
            "you just wrote: `super().__init__()` + "
            "`forward = F.relu`. (It also accepts an `inplace=True` "
            "flag that calls `F.relu_` instead.) Every other "
            "stateless `nn.X` (`nn.GELU`, `nn.Softmax`, `nn.Sigmoid`) "
            "is the same pattern.\n\n"
            "**Gradient comes for free.** Because `F.relu` is "
            "autograd-tracked, wrapping it in a Module doesn't break "
            "the backward pass. You didn't have to write any backward "
            "logic — that's the value of staying inside the "
            "functional layer."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # tensor-reshape-view — ex1
    # =========================================================
    {
        "atom_id": "tensor-reshape-view",
        "subtopic": "PyTorch: reshape vs view",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_TENSOR_RESHAPE_VIEW,
        "exercise_index": 1,
        "exercise_title": "choose reshape vs view based on whether the input is contiguous",
        "slug": "choose-reshape-vs-view-based-on-whether-input-is-contiguous",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["reshape", "view", "contiguous", "stride"],
        "kcs": [
            "view-requires-contiguous",
            "reshape-falls-back-to-copy",
        ],
        "lo": (
            "Apply the reshape-vs-view decision rule: use `.view()` "
            "when you require a no-copy reshape (and want a "
            "`RuntimeError` if that's impossible), use `.reshape()` "
            "when a copy is acceptable as a fallback."
        ),
        "prompt_body": (
            "Implement TWO functions that demonstrate the difference "
            "between `.view()` and `.reshape()`:\n\n"
            "**1. `ex1_strict_view(x, shape)`** — return "
            "`x.view(*shape)`. This MUST raise `RuntimeError` when "
            "`x` is not contiguous and the target shape requires a "
            "memory reorder. Do NOT call `.contiguous()` first.\n\n"
            "**2. `ex1_safe_reshape(x, shape)`** — return "
            "`x.reshape(*shape)`. This works on contiguous AND "
            "non-contiguous inputs (silently copies when needed).\n\n"
            "Inputs:\n"
            "- `x`: a `Tensor`.\n"
            "- `shape`: a tuple of ints (the target shape).\n\n"
            "Output: a `Tensor` of the target shape.\n\n"
            "Note: `x.view(*shape)` unpacks the tuple into "
            "positional ints (the standard PyTorch idiom). Same for "
            "`x.reshape(*shape)`."
        ),
        "stub": (
            "def ex1_strict_view(x: Tensor, shape: tuple) -> Tensor:\n"
            '    """Return x.view(*shape). Raises on non-contiguous."""\n'
            "    raise NotImplementedError()\n"
            "\n"
            "def ex1_safe_reshape(x: Tensor, shape: tuple) -> Tensor:\n"
            '    """Return x.reshape(*shape). Always succeeds (may copy)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === On contiguous input, both work and return EQUAL data ===\n"
            "x = t.arange(12).reshape(3, 4)\n"
            "assert x.is_contiguous()\n"
            "v = ex1_strict_view(x, (4, 3))\n"
            "r = ex1_safe_reshape(x, (4, 3))\n"
            "assert v.shape == (4, 3)\n"
            "assert r.shape == (4, 3)\n"
            "assert t.equal(v, r), 'on contiguous input, view and reshape produce equal results'\n"
            "\n"
            "# === On contiguous input, view returns a VIEW (shares storage) ===\n"
            "x = t.arange(12).reshape(3, 4)\n"
            "v = ex1_strict_view(x, (12,))\n"
            "assert v.data_ptr() == x.data_ptr(), 'view should share storage with source'\n"
            "\n"
            "# === On NON-contiguous input, view RAISES ===\n"
            "y = x.T   # transpose → non-contig\n"
            "assert not y.is_contiguous()\n"
            "try:\n"
            "    ex1_strict_view(y, (12,))\n"
            "except RuntimeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('view on non-contig should raise RuntimeError')\n"
            "\n"
            "# === On NON-contiguous input, reshape silently copies and succeeds ===\n"
            "r = ex1_safe_reshape(y, (12,))\n"
            "assert r.shape == (12,), f'expected (12,), got {tuple(r.shape)}'\n"
            "# Reshape on non-contig may share or copy; the CONTRACT is just that it works.\n"
            "# The data must equal y flattened in row-major.\n"
            "expected = y.contiguous().flatten()\n"
            "assert t.equal(r, expected), f'reshape result mismatch: got {r} expected {expected}'\n"
            "\n"
            "# === 1-D source can be reshaped back into a 1-D target of any compatible length ===\n"
            "x_flat = t.tensor([7.0])\n"
            "v_sc = ex1_strict_view(x_flat, (1,))\n"
            "assert v_sc.shape == (1,), f'expected (1,), got {tuple(v_sc.shape)}'\n"
            "assert v_sc.item() == 7.0\n"
            "r_sc = ex1_safe_reshape(x_flat, (1, 1, 1))\n"
            "assert r_sc.shape == (1, 1, 1)\n"
            "assert r_sc.item() == 7.0\n"
            "\n"
            "# === reshape can collapse / expand any compatible shape ===\n"
            "x = t.arange(24)\n"
            "for shp in [(24,), (4, 6), (2, 3, 4), (1, 24, 1)]:\n"
            "    r = ex1_safe_reshape(x, shp)\n"
            "    assert r.shape == shp, f'expected {shp}, got {tuple(r.shape)}'\n"
            "    assert t.equal(r.flatten(), x), f'data lost reshaping to {shp}'\n"
            "\n"
            "# === Confirm view did NOT call contiguous internally ===\n"
            "# If a student wrote `x.contiguous().view(*shape)`, the non-contig test above\n"
            "# would have silently succeeded instead of raising. Re-test to be sure.\n"
            "x = t.arange(6).reshape(2, 3)\n"
            "y = x.T   # (3, 2) non-contig\n"
            "raised = False\n"
            "try:\n"
            "    _ = ex1_strict_view(y, (6,))\n"
            "except RuntimeError:\n"
            "    raised = True\n"
            "assert raised, (\n"
            "    'ex1_strict_view must NOT call .contiguous() — '\n"
            "    'it must raise on non-contig input'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_strict_view(x, shape):\n"
            "    return x.view(*shape)\n"
            "\n"
            "def ex1_safe_reshape(x, shape):\n"
            "    return x.reshape(*shape)"
        ),
        "solution_notes": (
            "**Two one-liners, two different contracts.** "
            "`view` is strict, no-copy, and shares storage; "
            "`reshape` is permissive and copies when needed. The "
            "code is trivial — the LEARNING is which one to reach "
            "for at each call site.\n\n"
            "**Why ARENA insists on `.view()` for stride exercises.** "
            "The lesson is stride arithmetic. If `.reshape()` "
            "silently copies, you've lost the stride property the "
            "test was checking. `.view()` raises so you notice.\n\n"
            "**Application code defaults to `.reshape()`.** "
            "Performance-sensitive inner loops use `.view()` (with "
            "`.contiguous()` upstream to guarantee it works). Most "
            "code just wants the shape change and doesn't care about "
            "the copy.\n\n"
            "**`.view()` is a bit faster when it works** — no branch, "
            "no contiguity check on the copy path. But the gap is "
            "measured in nanoseconds; correctness matters more."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # linspace-out-param — ex1
    # =========================================================
    {
        "atom_id": "linspace-out-param",
        "subtopic": "PyTorch: linspace out= param",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_LINSPACE_OUT_PARAM,
        "exercise_index": 1,
        "exercise_title": "use linspace(out=) to fill a pre-allocated buffer in place",
        "slug": "use-linspace-out-to-fill-a-pre-allocated-buffer-in-place",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["linspace", "out", "in-place", "pre-allocation"],
        "kcs": [
            "out-kwarg-fills-in-place",
            "out-tensor-aliasing",
        ],
        "lo": (
            "Apply `torch.linspace(..., out=buf)` to fill a "
            "pre-allocated buffer in place across a zero-alloc inner "
            "loop, exploiting that `out=` shares storage with the "
            "buffer."
        ),
        "prompt_body": (
            "Implement `ex1_fill_schedule(buf, t_maxes)`. The "
            "zero-alloc time-grid pattern used in diffusion samplers "
            "and physics solvers:\n\n"
            "1. `buf` is a 1-D float tensor with `N = buf.numel()` "
            "elements. It is THE buffer — every call must write into "
            "it in place, never allocate a new tensor.\n"
            "2. `t_maxes` is a list of floats; for each `t_max`, call "
            "`torch.linspace(0.0, t_max, N, out=buf)` to fill `buf` "
            "with the schedule `[0, t_max/(N-1), 2*t_max/(N-1), ..., "
            "t_max]`.\n"
            "3. After each fill, append `buf.sum().item()` to a "
            "results list (so the test can verify the right value "
            "lived in `buf` at the right moment).\n"
            "4. Return the results list.\n\n"
            "Constraints:\n"
            "- MUST use `out=buf` — do not allocate a new tensor per "
            "iteration.\n"
            "- The returned tensor from `torch.linspace(..., out=buf)` "
            "is the SAME object as `buf` (aliased) — the test "
            "verifies this.\n\n"
            "Inputs:\n"
            "- `buf`: 1-D `Tensor` of any size.\n"
            "- `t_maxes`: `list[float]`.\n\n"
            "Output: `list[float]` (one per `t_max`)."
        ),
        "stub": (
            "def ex1_fill_schedule(buf: Tensor, t_maxes: list) -> list:\n"
            '    """Fill buf in place for each t_max via linspace(out=buf)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Hand-traced: linspace(0, 2, 3, out=buf) -> [0, 1, 2], sum=3 ===\n"
            "buf = t.empty(3)\n"
            "out = ex1_fill_schedule(buf, [2.0])\n"
            "assert out == [3.0], f'expected [3.0], got {out}'\n"
            "assert t.allclose(buf, t.tensor([0.0, 1.0, 2.0])), f'buf state wrong: {buf}'\n"
            "\n"
            "# === buf object identity preserved across calls ===\n"
            "buf = t.empty(5)\n"
            "buf_id = id(buf)\n"
            "buf_ptr = buf.data_ptr()\n"
            "out = ex1_fill_schedule(buf, [1.0, 2.0, 3.0])\n"
            "assert id(buf) == buf_id, 'buf reference changed — did you reassign instead of writing in place?'\n"
            "assert buf.data_ptr() == buf_ptr, 'buf storage changed — out= should NOT reallocate'\n"
            "# After three calls, buf should hold the LAST schedule (0..3, 5 points).\n"
            "expected_last = t.linspace(0.0, 3.0, 5)\n"
            "assert t.allclose(buf, expected_last), f'buf should hold last schedule, got {buf}'\n"
            "\n"
            "# === Per-iter sums verified ===\n"
            "buf = t.empty(11)\n"
            "out = ex1_fill_schedule(buf, [1.0, 5.0])\n"
            "# linspace(0, 1, 11).sum() = 5.5; linspace(0, 5, 11).sum() = 27.5\n"
            "expected_sums = [5.5, 27.5]\n"
            "for got, want in zip(out, expected_sums):\n"
            "    assert abs(got - want) < 1e-5, f'sum mismatch: got {got}, expected {want}'\n"
            "\n"
            "# === No allocation per iter: count storage objects via _typed_storage ===\n"
            "# After many calls, buf must still hold the same underlying storage.\n"
            "buf = t.empty(7)\n"
            "storages = []\n"
            "for _ in range(100):\n"
            "    ex1_fill_schedule(buf, [1.0])\n"
            "    storages.append(buf.data_ptr())\n"
            "assert len(set(storages)) == 1, (\n"
            "    f'storage changed across iterations ({len(set(storages))} distinct ptrs) — '\n"
            "    f'are you allocating a new tensor instead of using out=buf?'\n"
            ")\n"
            "\n"
            "# === Empty schedule list -> empty output, buf untouched ===\n"
            "buf = t.tensor([99.0, 98.0, 97.0])\n"
            "out = ex1_fill_schedule(buf, [])\n"
            "assert out == []\n"
            "assert t.allclose(buf, t.tensor([99.0, 98.0, 97.0])), 'buf should be untouched for empty input'\n"
            "\n"
            "# === buf size determines linspace N (not a separate arg) ===\n"
            "buf3 = t.empty(3)\n"
            "buf10 = t.empty(10)\n"
            "out3 = ex1_fill_schedule(buf3, [1.0])\n"
            "out10 = ex1_fill_schedule(buf10, [1.0])\n"
            "# linspace(0, 1, 3).sum() = 0 + 0.5 + 1 = 1.5\n"
            "# linspace(0, 1, 10).sum() = 5.0\n"
            "assert abs(out3[0] - 1.5) < 1e-5, f'expected 1.5, got {out3[0]}'\n"
            "assert abs(out10[0] - 5.0) < 1e-5, f'expected 5.0, got {out10[0]}'"
        ),
        "solution_body": (
            "def ex1_fill_schedule(buf, t_maxes):\n"
            "    N = buf.numel()\n"
            "    results = []\n"
            "    for t_max in t_maxes:\n"
            "        t.linspace(0.0, t_max, N, out=buf)\n"
            "        results.append(buf.sum().item())\n"
            "    return results"
        ),
        "solution_notes": (
            "**`out=buf` is the key.** The function fills `buf` in "
            "place AND returns it. We discard the return value and "
            "just read `buf.sum()` — same data, same storage.\n\n"
            "**Why this matters for inner loops.** Diffusion samplers, "
            "Runge-Kutta solvers, and ODE integrators reuse a small "
            "time-grid buffer thousands of times per call. Without "
            "`out=`, each iteration allocates → fills → "
            "deallocates a fresh `N`-element tensor. With `out=`, the "
            "loop allocates ONCE.\n\n"
            "**Same pattern, many ops.** `torch.zeros(out=)`, "
            "`torch.arange(out=)`, `torch.randn(out=)`, "
            "`torch.matmul(a, b, out=)`. Every op that takes "
            "`out=` follows the same in-place + return-aliased "
            "contract.\n\n"
            "**Pre-sized buffer is the contract.** `buf` must be "
            "1-D, the right size, and the right dtype, or PyTorch "
            "will silently resize it (defeating the no-alloc goal). "
            "In production code, allocate `buf` once at setup time "
            "and reuse it forever."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # trainer-subclass-extend — ex1
    # =========================================================
    {
        "atom_id": "trainer-subclass-extend",
        "subtopic": "Trainer: subclass extend pattern",
        "topic_folder": "prereqs_misc_cleanup",
        "atom_recap_md": RECAP_TRAINER_SUBCLASS_EXTEND,
        "exercise_index": 1,
        "exercise_title": "subclass a trainer and extend _step via super() delegation",
        "slug": "subclass-a-trainer-and-extend-step-via-super-delegation",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["subclass", "super", "hook", "trainer"],
        "kcs": [
            "super-step-delegate-then-extend",
            "preserve-base-return-shape",
        ],
        "lo": (
            "Apply the subclass-extend pattern: override "
            "`BaseTrainer._step` in a subclass that calls "
            "`super()._step(batch)` first and then ADDS a new metric "
            "to the returned dict."
        ),
        "prompt_body": (
            "`BaseTrainer` is provided in the stub. It has a "
            "`_step(batch)` method that runs the base training step "
            "and returns a dict like `{'loss': <float>}`.\n\n"
            "Implement `LoggingTrainer(BaseTrainer)`. In its "
            "`_step(batch)`:\n"
            "1. Call `super()._step(batch)` FIRST. Capture the result "
            "(a dict).\n"
            "2. Compute `extra = batch[0].abs().mean().item()` — the "
            "mean absolute value of the input tensor. This is a fake "
            "'input magnitude' metric used to demonstrate the extend "
            "pattern.\n"
            "3. Add `'input_mag': extra` to the dict.\n"
            "4. Return the EXTENDED dict — must contain BOTH the "
            "base's `'loss'` key AND your new `'input_mag'` key.\n\n"
            "Constraints:\n"
            "- MUST call `super()._step(batch)` — do not duplicate "
            "the base body.\n"
            "- MUST return the same dict object (or a dict with the "
            "same `loss` value) — base callers expect the loss key "
            "intact.\n"
            "- Do NOT add `__init__`; inherit it from `BaseTrainer`."
        ),
        "stub": (
            "import torch.nn as nn\n"
            "\n"
            "class BaseTrainer:\n"
            "    def __init__(self, model, lr=1e-2):\n"
            "        self.model = model\n"
            "        self.opt = t.optim.SGD(model.parameters(), lr=lr)\n"
            "        self.loss_fn = nn.MSELoss()\n"
            "\n"
            "    def _step(self, batch):\n"
            "        x, y = batch\n"
            "        pred = self.model(x)\n"
            "        loss = self.loss_fn(pred, y)\n"
            "        self.opt.zero_grad()\n"
            "        loss.backward()\n"
            "        self.opt.step()\n"
            "        return {'loss': loss.item()}\n"
            "\n"
            "class LoggingTrainer(BaseTrainer):\n"
            "    def _step(self, batch):\n"
            "        raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "\n"
            "# === Subclass extends base: result has BOTH keys ===\n"
            "t.manual_seed(0)\n"
            "model = nn.Linear(3, 2)\n"
            "trainer = LoggingTrainer(model)\n"
            "x = t.randn(4, 3)\n"
            "y = t.randn(4, 2)\n"
            "out = trainer._step((x, y))\n"
            "assert isinstance(out, dict), f'must return dict, got {type(out).__name__}'\n"
            "assert 'loss' in out, f'must keep base loss key, got keys {list(out.keys())}'\n"
            "assert 'input_mag' in out, f'must add input_mag key, got keys {list(out.keys())}'\n"
            "\n"
            "# === input_mag matches the formula ===\n"
            "expected_mag = x.abs().mean().item()\n"
            "assert abs(out['input_mag'] - expected_mag) < 1e-6, (\n"
            "    f'input_mag wrong: got {out[\"input_mag\"]}, expected {expected_mag}'\n"
            ")\n"
            "\n"
            "# === Base step still ran (params moved) ===\n"
            "t.manual_seed(0)\n"
            "model = nn.Linear(3, 2)\n"
            "snapshot = [p.detach().clone() for p in model.parameters()]\n"
            "trainer = LoggingTrainer(model)\n"
            "trainer._step((x, y))\n"
            "moved = any(\n"
            "    not t.allclose(p1, p2) for p1, p2 in zip(model.parameters(), snapshot)\n"
            ")\n"
            "assert moved, 'super()._step should have triggered an optimizer step that moved params'\n"
            "\n"
            "# === Subclass extension matches base loss key (sanity vs running base directly) ===\n"
            "t.manual_seed(0)\n"
            "model_a = nn.Linear(3, 2)\n"
            "base = BaseTrainer(model_a)\n"
            "out_base = base._step((x, y))\n"
            "\n"
            "t.manual_seed(0)\n"
            "model_b = nn.Linear(3, 2)\n"
            "ext = LoggingTrainer(model_b)\n"
            "out_ext = ext._step((x, y))\n"
            "assert abs(out_base['loss'] - out_ext['loss']) < 1e-6, (\n"
            "    f'extended loss should match base loss: base={out_base[\"loss\"]} ext={out_ext[\"loss\"]}'\n"
            ")\n"
            "\n"
            "# === Inheritance chain: LoggingTrainer IS a BaseTrainer ===\n"
            "assert isinstance(trainer, BaseTrainer), 'LoggingTrainer must subclass BaseTrainer'\n"
            "\n"
            "# === Did NOT duplicate base body: super() must be called.\n"
            "# Proxy check: monkey-patch BaseTrainer._step to track calls, then verify.\n"
            "calls = []\n"
            "orig_step = BaseTrainer._step\n"
            "def tracked_step(self, batch):\n"
            "    calls.append(1)\n"
            "    return orig_step(self, batch)\n"
            "BaseTrainer._step = tracked_step\n"
            "try:\n"
            "    t.manual_seed(0)\n"
            "    trainer = LoggingTrainer(nn.Linear(3, 2))\n"
            "    trainer._step((x, y))\n"
            "    assert calls == [1], f'BaseTrainer._step should have been called exactly once, got {len(calls)} calls'\n"
            "finally:\n"
            "    BaseTrainer._step = orig_step\n"
            "\n"
            "# === Multi-step works (state preserved across calls via inherited __init__) ===\n"
            "t.manual_seed(0)\n"
            "trainer = LoggingTrainer(nn.Linear(3, 2))\n"
            "losses = []\n"
            "for _ in range(5):\n"
            "    out = trainer._step((x, y))\n"
            "    losses.append(out['loss'])\n"
            "# Loss should decrease on this trivial supervised toy.\n"
            "assert losses[-1] < losses[0], f'expected loss to decrease across 5 steps, got {losses}'"
        ),
        "solution_body": (
            "import torch.nn as nn\n"
            "\n"
            "class BaseTrainer:\n"
            "    def __init__(self, model, lr=1e-2):\n"
            "        self.model = model\n"
            "        self.opt = t.optim.SGD(model.parameters(), lr=lr)\n"
            "        self.loss_fn = nn.MSELoss()\n"
            "\n"
            "    def _step(self, batch):\n"
            "        x, y = batch\n"
            "        pred = self.model(x)\n"
            "        loss = self.loss_fn(pred, y)\n"
            "        self.opt.zero_grad()\n"
            "        loss.backward()\n"
            "        self.opt.step()\n"
            "        return {'loss': loss.item()}\n"
            "\n"
            "class LoggingTrainer(BaseTrainer):\n"
            "    def _step(self, batch):\n"
            "        # 1. Delegate to the base step (runs the actual training step,\n"
            "        #    returns {'loss': <float>}).\n"
            "        out = super()._step(batch)\n"
            "        # 2. Extend with an extra metric.\n"
            "        out['input_mag'] = batch[0].abs().mean().item()\n"
            "        # 3. Return the extended dict (still contains base's 'loss').\n"
            "        return out"
        ),
        "solution_notes": (
            "**Three lines, full extend pattern.** "
            "`super()._step(batch)` does the heavy lifting. The "
            "subclass only adds what's new — the input-magnitude "
            "metric — and returns the extended dict.\n\n"
            "**Why not write a fresh `_step` from scratch.** You'd "
            "duplicate the optimizer step, the zero_grad, the "
            "backward — and any future fix to "
            "`BaseTrainer._step` (e.g. adding gradient clipping) "
            "would need to be re-applied to your subclass. With "
            "`super()._step()`, the fix is inherited for free.\n\n"
            "**The 'before/after/around' variants.** "
            "BEFORE: `self._pre(); out = super()._step(batch); "
            "return out`. AFTER: `out = super()._step(batch); "
            "self._post(out); return out`. AROUND: do both — most "
            "common for logging hooks.\n\n"
            "**Inherit `__init__` for free.** Because "
            "`LoggingTrainer` doesn't define its own `__init__`, it "
            "uses `BaseTrainer.__init__`. If you needed extra "
            "subclass-specific state, you'd write "
            "`def __init__(self, model, lr=1e-2, log_dir=None): "
            "super().__init__(model, lr); self.log_dir = log_dir`."
        ),
        "extra_imports": [],
    },

]


# ---------------------------------------------------------------------------
# Verify each solution against its test body in-process.
# ---------------------------------------------------------------------------

def _verify_spec(spec):
    """Compile a tiny module from solution + test, run it, raise on failure."""
    atom_id = spec["atom_id"]
    ex_idx = spec["exercise_index"]
    src_lines = [
        "import numpy as np",
        "import torch as t",
        "from torch import Tensor",
        "import einops",
        "from einops import rearrange, reduce, repeat",
        "",
        "t.manual_seed(0)",
        "np.random.seed(0)",
        "",
    ]
    for extra in spec.get("extra_imports", []) or []:
        src_lines.append(extra)
    src_lines.append("")
    src_lines.append(spec["solution_body"])
    src_lines.append("")
    src_lines.append(spec["test_body"])
    src = "\n".join(src_lines)
    ns = {}
    try:
        exec(compile(src, f"<verify {atom_id} ex{ex_idx}>", "exec"), ns)
    except Exception:
        print(f"\n--- VERIFICATION FAILED for {atom_id} ex{ex_idx} ---", file=sys.stderr)
        traceback.print_exc()
        print("--- source ---", file=sys.stderr)
        for i, line in enumerate(src.splitlines(), 1):
            print(f"{i:4d}  {line}", file=sys.stderr)
        raise


def main():
    for spec in SPECS:
        print(f"verifying {spec['atom_id']} ex{spec['exercise_index']} ...", flush=True)
        _verify_spec(spec)
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()
