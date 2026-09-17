#!/usr/bin/env python3
"""Author 8 ex3 deepening drills (batch 14, group E — prereqs_autograd_pt3).

Atoms (8 in prereqs_autograd_pt3) — each gets a DISTINCT third facet:
    - box-array-to-tensor-with-recipe  (ex3: build parents dict by argnum — bookkeeping step)
    - coerce-float-arg-to-array        (ex3: coerce KWARGS values, leave control-flag kwargs)
    - get-children-callable-param      (ex3: Module.__call__ delegates to forward — callable facet)
    - grad-accumulate-on-leaf          (ex3: gradient accumulation across micro-batches, no zero between)
    - inplace-op-unsafe-warning        (ex3: detach() — peel Recipe off so subsequent in-place is safe)
    - parameter-subclass-of-tensor     (ex3: freeze(module, prefix) — mutate requires_grad on matching Parameters)
    - sum-and-broadcast-duality        (ex3: mean_back — mean is sum/n; backward = sum_back / n)
    - unbox-args-tensor-to-array       (ex3: unbox KWARGS dict — different container than args/nested-list)

Each ex3 is distinct from ex1 + ex2 along the atom's load-bearing axis.
ONE LO + ONE Bloom + <=2 KCs each. The MiniTensor / Recipe shared preamble is
emitted automatically by `emit_standalone`.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_pt3"

# Shared MiniTensor + Recipe preamble — appended to each notebook's setup cell
# via `extra_imports`. Matches the preamble used by every ex1/ex2 drill in
# this folder, so ex3 drills can reference MiniTensor / Recipe / grad_tracking_enabled
# without re-declaring them inside the stub or solution.
AUTOGRAD_PREAMBLE = (
    "# === manual autograd primitives — shared across all drills in this folder ===\n"
    "from dataclasses import dataclass, field\n"
    "from typing import Any, Callable, Optional\n"
    "\n"
    "grad_tracking_enabled = True\n"
    "\n"
    "@dataclass\n"
    "class Recipe:\n"
    "    func: Optional[Callable] = None\n"
    "    args: tuple = ()\n"
    "    kwargs: dict = field(default_factory=dict)\n"
    "    parents: dict = field(default_factory=dict)\n"
    "\n"
    "class MiniTensor:\n"
    "    \"\"\"Minimal Tensor wrapper for the ARENA-style manual-autograd drills.\n"
    "    Wraps a raw `torch.Tensor` in `.array`. Carries optional `.recipe`,\n"
    "    `.requires_grad`, and `.grad` (the accumulated gradient at leaves).\"\"\"\n"
    "    def __init__(self, array, requires_grad: bool = False, recipe=None):\n"
    "        self.array = array\n"
    "        self.requires_grad = requires_grad\n"
    "        self.recipe = recipe\n"
    "        self.grad = None\n"
    "    def __repr__(self):\n"
    "        return f'MiniTensor({self.array!r}, requires_grad={self.requires_grad})'"
)


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_PARENTS_DICT = (
    "## `build_parents` — the bookkeeping step that lets backward find the inputs\n"
    "\n"
    "Ex1 boxed the raw output + recipe. Ex2 composed the whole wrapper. The "
    "third facet of the same atom is the parents-dict construction — the "
    "bookkeeping that lets the reverse pass know which input occupied which "
    "argument slot:\n"
    "\n"
    "```python\n"
    "parents = {\n"
    "    idx: a for idx, a in enumerate(args)\n"
    "    if isinstance(a, MiniTensor) and a.requires_grad\n"
    "}\n"
    "```\n"
    "\n"
    "**Why a dict keyed by argnum.** Each back fn for `op(x, y)` needs to "
    "know which gradient goes to x and which goes to y — they're routed by "
    "ARGUMENT INDEX. `parents[0]` is the first MiniTensor input, `parents[1]` "
    "the second, etc. Skipping non-MiniTensor args means indices in `parents` "
    "are NOT contiguous: `op(t1, 3.0, t2)` produces `parents = {0: t1, 2: t2}`.\n"
    "\n"
    "**Why filter on `requires_grad`.** A MiniTensor with `requires_grad=False` "
    "is a frozen input — gradient flow stops there. Including it in `parents` "
    "would waste a back-fn dispatch and corrupt the leaf-set the reverse pass "
    "uses to know when to stop. Filtering at build time keeps the graph "
    "minimal.\n"
    "\n"
    "**Why skip when `requires_grad` is False.** No-grad inputs don't need "
    "to be revisited on backward. Filtering keeps the parents dict small "
    "and the reverse pass fast — especially important when one of the inputs "
    "is a giant constant (e.g. positional embeddings) that the user "
    "explicitly froze."
)

RECAP_COERCE_KWARGS = (
    "## Coerce **kwargs** values — same scalar rule, different container\n"
    "\n"
    "Ex1 coerced a single positional arg. Ex2 coerced the args tuple. The "
    "third facet: the SAME promote-scalar-to-tensor rule applied across the "
    "kwargs dict. PyTorch ops commonly take numeric hyperparameters as "
    "kwargs — `t.clamp(x, min=0.0, max=1.0)`, `t.pow(x, exponent=2.0)` — and "
    "the wrapper must coerce these the same way it coerces positional "
    "scalars so the Recipe stores tensors everywhere.\n"
    "\n"
    "```python\n"
    "coerce_kwargs({'min': 0.0, 'max': 1.0, 'keepdim': True, 'dim': 2})\n"
    "  → {'min': tensor(0.0), 'max': tensor(1.0), 'keepdim': True, 'dim': 2}\n"
    "```\n"
    "\n"
    "**Control-flag kwargs stay raw.** `keepdim`, `dim`, `out` are NOT "
    "numeric scalars in the math sense — they're flags / axis indices. The "
    "raw torch op interprets `dim` as a Python int specifically; passing "
    "`tensor(2)` raises. We get this for free by reusing the int/float-only "
    "rule from ex1 — but the trap is `dim`: a Python int. The simplest "
    "out: coerce ONLY `float` values in kwargs, leave `int` / `bool` alone.\n"
    "\n"
    "**Why this matters.** Without kwarg coercion, the Recipe stores "
    "`kwargs={'min': 0.0}` — a Python float. The reverse pass for clamp "
    "needs `min` as a tensor (to compare against `x.array` with masking). "
    "Heterogeneous storage forces N tensor-vs-float checks across every "
    "back fn — same complaint we had against not coercing positional "
    "args."
)

RECAP_MODULE_CALL = (
    "## `__call__` — the callable half of get_children / parameters / Module\n"
    "\n"
    "Ex1 walked `__dict__` to yield child tensors. Ex2 added recursion + "
    "dotted names. The atom name is `get-children-callable-param` — the "
    "third load-bearing facet is the CALLABLE half: `nn.Module.__call__` "
    "routes through `forward`. That's why `model(x)` works without you "
    "ever writing `__call__` in your subclass.\n"
    "\n"
    "```python\n"
    "class Module:\n"
    "    def __call__(self, *args, **kwargs):\n"
    "        return self.forward(*args, **kwargs)\n"
    "\n"
    "    def forward(self, *args, **kwargs):\n"
    "        raise NotImplementedError(\n"
    "            f'{type(self).__name__} must implement forward'\n"
    "        )\n"
    "```\n"
    "\n"
    "**Why route through `forward`, not just rename.** The full PyTorch "
    "`nn.Module.__call__` runs forward AND fires pre/post hooks, training "
    "vs eval branching, and grad-mode bookkeeping AROUND the forward call. "
    "Even our minimal version leaves the hook slot open — subclasses can "
    "override `__call__` to add behavior without touching `forward`'s "
    "subclass-overridden body.\n"
    "\n"
    "**Why `forward` must raise NotImplementedError.** A subclass that "
    "forgets to define `forward` would silently return None on `model(x)`. "
    "Raising at the base level forces the subclass to be explicit about its "
    "compute. Same rationale as `abc.abstractmethod` — fail at the right "
    "abstraction level."
)

RECAP_GRAD_ACCUM_MICROBATCH = (
    "## Gradient accumulation across micro-batches — accumulate on PURPOSE\n"
    "\n"
    "Ex1 implemented the per-touch accumulate. Ex2 added `zero_grad` to "
    "clear between training steps. The third facet flips the script: "
    "INTENTIONAL multi-step accumulation across MICRO-BATCHES, where the "
    "absence of zero_grad is the feature, not the bug.\n"
    "\n"
    "```python\n"
    "# Effective batch_size = micro_bs * n_microbatches, with constant memory.\n"
    "for microbatch in batches:\n"
    "    loss = compute_loss(model, microbatch) / n_microbatches\n"
    "    backward(loss)        # adds to p.grad for each param\n"
    "# After the loop, p.grad is the average over the full effective batch.\n"
    "optimizer_step(model, lr)\n"
    "zero_grad(model.parameters())  # clear ONLY after the optimizer step\n"
    "```\n"
    "\n"
    "**Why divide each micro-loss by `n_microbatches`.** Without the "
    "division, summing N micro-losses gives N x the desired loss — the "
    "gradient is N x too large. Dividing inside the loop is mathematically "
    "equivalent to averaging the full batch's loss, but you only ever hold "
    "one micro-batch's activations in memory.\n"
    "\n"
    "**Why no zero_grad between micro-steps.** That's the whole point — "
    "the accumulation IS the simulated big-batch gradient. Calling "
    "zero_grad between micro-batches would defeat the purpose and only "
    "the last micro-batch's gradient would inform the update.\n"
    "\n"
    "**Why this is its own pattern.** Gradient accumulation is how you "
    "train 64-batch-size models on a GPU that only fits 8. The pattern is "
    "load-bearing for every LLM training run since GPT-2 — and it's "
    "fundamentally the same `accumulate_grad` from ex1, just deliberately "
    "called across multiple forward/backward passes before zeroing."
)

RECAP_DETACH = (
    "## `detach()` — peel the Recipe off so in-place becomes safe\n"
    "\n"
    "Ex1 made in-place ops refuse when `.recipe is not None`. Ex2 added a "
    "context-manager escape hatch. The third facet is the surgical escape: "
    "`detach()` returns a NEW MiniTensor sharing the SAME `.array` but "
    "with `recipe=None` and `requires_grad=False`. It says 'I know what "
    "I'm doing — disconnect this tensor from the graph here'.\n"
    "\n"
    "```python\n"
    "def detach(x: MiniTensor) -> MiniTensor:\n"
    "    # New wrapper, same underlying storage, no recipe.\n"
    "    return MiniTensor(x.array, requires_grad=False, recipe=None)\n"
    "```\n"
    "\n"
    "**Same array — but different wrapper.** `detach()` is NOT a copy of "
    "the data. The new MiniTensor's `.array` IS the same `torch.Tensor` "
    "object as `x.array`. So mutating the detached one's `.array` ALSO "
    "mutates the original's `.array` — but the original's recipe is no "
    "longer hooked into the new one's identity, so the in-place guard on "
    "the detached wrapper passes.\n"
    "\n"
    "**Why this is dangerous AND useful.** Dangerous: you've severed the "
    "graph at this point. The reverse pass won't propagate through the "
    "detached node. Useful: in inference, in stop-gradient operations "
    "(BatchNorm running stats, target networks in RL), and at boundaries "
    "where you genuinely want autograd to ignore a subgraph. PyTorch's "
    "`tensor.detach()` is the exact same idiom.\n"
    "\n"
    "**Contrast with the context manager from ex2.** `inplace_unsafe()` "
    "disables the guard globally. `detach()` disables it for ONE specific "
    "tensor by replacing its wrapper. Local vs global. Both legitimate; "
    "different ergonomic trade-offs."
)

RECAP_FREEZE_PARAMETERS = (
    "## `freeze(module, prefix)` — mutate `requires_grad` on selected Parameters\n"
    "\n"
    "Ex1 defined `Parameter` as a typed subclass. Ex2 filtered for trainable "
    "params via `isinstance(_, Parameter)`. The third facet is the "
    "operational use of the type tag: walk parameters() and FLIP "
    "`requires_grad` based on a name prefix — the standard frozen-backbone "
    "pattern for fine-tuning.\n"
    "\n"
    "```python\n"
    "def freeze(module, prefix):\n"
    "    \"\"\"Set requires_grad=False on every Parameter whose dotted name starts with prefix.\"\"\"\n"
    "    for name, p in module.parameters():\n"
    "        if name.startswith(prefix):\n"
    "            p.requires_grad = False\n"
    "```\n"
    "\n"
    "**Why the type tag is what matters, not requires_grad.** A Parameter "
    "with `requires_grad=False` is STILL a Parameter — `trainable_params` "
    "still yields it (ex2's load-bearing invariant). What changes is the "
    "gradient-flow gate: the wrapper's any-rg-input check excludes "
    "rg=False parents, so no gradient ever reaches the frozen param.\n"
    "\n"
    "**Why mutate in place, not return a new module.** Modules carry "
    "state — buffers, configs, registered hooks. Cloning the whole module "
    "to flip one flag would be wasteful and break any external reference "
    "to the module. PyTorch's standard idiom is exactly the same: "
    "`for p in model.encoder.parameters(): p.requires_grad = False`.\n"
    "\n"
    "**The dotted prefix matches naming from ex2.** `freeze(model, 'fc1.')` "
    "freezes everything under `fc1` — `fc1.weight`, `fc1.bias`. "
    "`freeze(model, 'fc1.weight')` freezes ONLY the weight. The prefix is "
    "a string match, not a glob, so the trailing `.` matters."
)

RECAP_MEAN_BACK = (
    "## `mean_back` — mean is `sum / n`, so its backward is `sum_back / n`\n"
    "\n"
    "Ex1 wrote `sum_back` (broadcast back across the reduced axis) and "
    "`broadcast_back` (sum out expanded axes) as duals. Ex2 verified the "
    "adjoint identity for sum. The third facet derives a NEW back fn from "
    "ex1's sum_back: mean is just sum divided by the reduction count, so "
    "its backward is the same broadcast-back divided by the same count.\n"
    "\n"
    "```python\n"
    "# Forward: out = x.mean(dim=k) == x.sum(dim=k) / x.shape[k]\n"
    "# Backward:\n"
    "#   d/dx [sum(x, k) / N] = sum_back(grad_out, x, k) / N\n"
    "def mean_back(grad_out, out, x, dim, keepdim=False):\n"
    "    n = x.shape[dim]\n"
    "    return sum_back(grad_out, out, x, dim, keepdim) / n\n"
    "```\n"
    "\n"
    "**Why this works without re-deriving from scratch.** The chain rule "
    "applied to a scalar multiple is dead simple: `d(c*f)/dx = c * df/dx`. "
    "Mean is `(1/N) * sum`. So its gradient is `(1/N) * grad-of-sum`. "
    "We're not approximating — this is the exact derivative.\n"
    "\n"
    "**The reduction count is `x.shape[dim]`, not `out.numel()`.** "
    "For a multi-axis mean `x.mean(dim=(0,1))`, the count is "
    "`x.shape[0] * x.shape[1]`. The single-axis case used here is just "
    "`x.shape[dim]` — the size of the axis we collapsed.\n"
    "\n"
    "**Why this is the duality showing up in derived form.** Mean is a "
    "linear map with a scalar normalization. Sum is a linear map without "
    "it. The duality (`sum_back` broadcasts) transfers through the scalar "
    "untouched — `mean_back` is the same broadcast, scaled. It's how "
    "every loss function with `.mean()` gets its gradient: same machinery "
    "as `sum_back`, just divided."
)

RECAP_UNBOX_KWARGS = (
    "## `unbox_kwargs` — unboxing the dict-shaped container\n"
    "\n"
    "Ex1 unboxed the positional args tuple. Ex2 recursed into nested "
    "list/tuple containers. The third facet completes the picture: "
    "kwargs are a `dict[str, value]` — a different container that also "
    "needs unboxing because nothing prevents a user from passing a "
    "MiniTensor as a kwarg.\n"
    "\n"
    "```python\n"
    "def unbox_kwargs(kwargs: dict) -> dict:\n"
    "    return {\n"
    "        k: v.array if isinstance(v, MiniTensor) else v\n"
    "        for k, v in kwargs.items()\n"
    "    }\n"
    "```\n"
    "\n"
    "**Why this is non-trivial despite the one-liner.** Many ops take "
    "tensors as kwargs: `t.where(cond, x, y)` (cond as a kwarg in custom "
    "wrappers), `t.scatter(input, dim, index, src)` (`src` as kwarg). "
    "Without kwarg unboxing, the raw torch fn receives a MiniTensor and "
    "crashes with `AttributeError: 'MiniTensor' has no attribute ...`.\n"
    "\n"
    "**Why a dict-comprehension preserves key order.** Python 3.7+ "
    "guarantees dict insertion order. Iterating `kwargs.items()` and "
    "rebuilding via comprehension preserves it. Important when the raw "
    "fn relies on kwarg ORDER for its repr or for downstream caches.\n"
    "\n"
    "**Dual of ex1.** Same `isinstance(_, MiniTensor)` test, same `.array` "
    "swap. The only difference is the container: tuple vs dict. Together "
    "with the positional unbox + the nested-container unbox from ex2, "
    "this covers every shape user code can throw at the wrapper."
)


# ---------------------------------------------------------------------------
# SPEC 1 — box-array-to-tensor-with-recipe ex3
# ---------------------------------------------------------------------------

SPEC_BOX = {
    "atom_id": "box-array-to-tensor-with-recipe",
    "subtopic": "Backprop: Box array as Tensor + recipe",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_PARENTS_DICT,
    "exercise_index": 3,
    "exercise_title": "build_parents: argnum → MiniTensor dict, filtered by requires_grad",
    "slug": "build-parents-by-argnum-filter-requires-grad",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["parents", "argnum", "build", "requires-grad", "bookkeeping"],
    "kcs": [
        "parents-dict-by-argidx",
        "filter-on-requires-grad",
    ],
    "lo": (
        "Apply the parents-dict construction step of wrap_forward_fn: walk "
        "positional args once and yield a dict keyed by argument index, "
        "containing only the MiniTensors whose requires_grad is True."
    ),
    "prompt_body": (
        "Implement `ex3_build_parents(args)`. The wrapper has already "
        "completed unbox/coerce; this is the bookkeeping pass that records "
        "WHICH input occupied WHICH argument slot, so the reverse pass can "
        "route gradients back by argnum.\n\n"
        "Inputs:\n"
        "- `args`: tuple of positional inputs (mixed `MiniTensor`, raw "
        "`torch.Tensor`, Python scalars, etc.).\n\n"
        "Output: a `dict[int, MiniTensor]` where:\n\n"
        "- Keys are the original positional indices (0-based) — NOT "
        "renumbered.\n"
        "- Values are the MiniTensor instances themselves (identity, not "
        "copies).\n"
        "- Include an arg ONLY when `isinstance(a, MiniTensor)` AND "
        "`a.requires_grad` is True.\n"
        "- Non-MiniTensor args and rg=False MiniTensors are SKIPPED — "
        "their index does NOT appear in the dict.\n\n"
        "Examples:\n\n"
        "```\n"
        "build_parents((t_rg, 3.0, t_rg2))   → {0: t_rg, 2: t_rg2}\n"
        "build_parents((t_no, t_rg))         → {1: t_rg}\n"
        "build_parents(())                   → {}\n"
        "build_parents((3.0, 4.0))           → {}\n"
        "```\n\n"
        "Constraints:\n"
        "- Indices in the output are NOT contiguous when scalars / "
        "rg=False inputs intersperse — that's by design.\n"
        "- Values must be the same Python object (not a copy)."
    ),
    "stub": (
        "def ex3_build_parents(args: tuple) -> dict:\n"
        '    """Argnum → MiniTensor dict, filtered to requires_grad=True only."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === Empty input ===\n"
        "    assert ex3_build_parents(()) == {}\n"
        "\n"
        "    # === All non-MiniTensor → empty parents ===\n"
        "    assert ex3_build_parents((1, 2.0, 'x')) == {}\n"
        "    assert ex3_build_parents((t.tensor([1.0]), 2)) == {}, (\n"
        "        'raw torch.Tensor is NOT a MiniTensor — must be skipped'\n"
        "    )\n"
        "\n"
        "    # === Single MiniTensor with rg=True ===\n"
        "    m = MiniTensor(t.tensor([1.0, 2.0]), requires_grad=True)\n"
        "    assert ex3_build_parents((m,)) == {0: m}\n"
        "    parents = ex3_build_parents((m,))\n"
        "    assert parents[0] is m, 'value must BE the same MiniTensor object'\n"
        "\n"
        "    # === MiniTensor with rg=False is filtered out ===\n"
        "    m_no = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "    assert ex3_build_parents((m_no,)) == {}, (\n"
        "        'rg=False MiniTensor must be filtered out'\n"
        "    )\n"
        "\n"
        "    # === Sparse indices: scalar at position 1, MiniTensors at 0 and 2 ===\n"
        "    m1 = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "    m2 = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "    parents = ex3_build_parents((m1, 3.0, m2))\n"
        "    assert parents == {0: m1, 2: m2}, (\n"
        "        f'indices must reflect ORIGINAL positions, not be renumbered; got {parents}'\n"
        "    )\n"
        "    # KEY: index 1 must NOT appear.\n"
        "    assert 1 not in parents, 'scalar arg position must not be in parents'\n"
        "\n"
        "    # === Mixed rg=True and rg=False MiniTensors ===\n"
        "    m_no2 = MiniTensor(t.tensor([7.0]), requires_grad=False)\n"
        "    parents = ex3_build_parents((m1, m_no2, m2))\n"
        "    assert parents == {0: m1, 2: m2}, (\n"
        "        f'rg=False MiniTensor at idx 1 must be filtered; got {parents}'\n"
        "    )\n"
        "\n"
        "    # === Length matters: with 5 args and only one MiniTensor at idx 3 ===\n"
        "    parents = ex3_build_parents((1, 2.0, 'x', m1, None))\n"
        "    assert parents == {3: m1}, f'idx must be 3 (original); got {parents}'\n"
        "\n"
        "    # === Two rg=True at adjacent positions ===\n"
        "    parents = ex3_build_parents((m1, m2))\n"
        "    assert parents == {0: m1, 1: m2}\n"
        "\n"
        "    # === Identity (values are NOT copies) ===\n"
        "    parents = ex3_build_parents((m1, m2))\n"
        "    assert parents[0] is m1\n"
        "    assert parents[1] is m2\n"
        "\n"
        "    # === Return type ===\n"
        "    parents = ex3_build_parents((m1,))\n"
        "    assert isinstance(parents, dict), (\n"
        "        f'must return dict, got {type(parents).__name__}'\n"
        "    )\n"
        "    # All keys are ints, all values are MiniTensors with rg=True.\n"
        "    for k, v in parents.items():\n"
        "        assert isinstance(k, int)\n"
        "        assert isinstance(v, MiniTensor) and v.requires_grad is True\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_build_parents(args):\n"
        "    return {\n"
        "        idx: a for idx, a in enumerate(args)\n"
        "        if isinstance(a, MiniTensor) and a.requires_grad\n"
        "    }"
    ),
    "solution_notes": (
        "**Indices are ORIGINAL, not renumbered.** `build_parents((m, 3.0, m2))` "
        "must return `{0: m, 2: m2}`, NOT `{0: m, 1: m2}`. The argnum is what "
        "back fns dispatch on — `multiply_back0(grad, ...)` knows it's working "
        "on argument 0, `multiply_back1` on argument 1. Renumbering would "
        "send gradients to the wrong slots.\n\n"
        "**Dict comprehension over `enumerate(args)` is canonical.** "
        "Single-pass, O(n), no temp lists, indices and elements paired up "
        "naturally. The two-condition filter (`isinstance` AND `requires_grad`) "
        "is short-circuit — `requires_grad` is only accessed if the type "
        "check passed, so we never `AttributeError` on a non-MiniTensor.\n\n"
        "**Why this is a third distinct facet.** Ex1 boxed the output + "
        "recipe. Ex2 composed the wrapper. Ex3 is the bookkeeping primitive "
        "that ex2 calls internally — same scan as `unbox_args`, different "
        "filter and different output shape. The wrapper actually combines "
        "all three: unbox, build_parents, then box-with-recipe."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 2 — coerce-float-arg-to-array ex3
# ---------------------------------------------------------------------------

SPEC_COERCE = {
    "atom_id": "coerce-float-arg-to-array",
    "subtopic": "Backprop: Coerce float arg to array",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_COERCE_KWARGS,
    "exercise_index": 3,
    "exercise_title": "coerce_kwargs: promote float kwargs to tensors, leave control flags raw",
    "slug": "coerce-kwargs-float-only-leave-int-and-bool",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["kwargs", "coerce", "float-only", "control-flag", "wrap-forward"],
    "kcs": [
        "kwargs-coerce-float-only",
        "preserve-int-and-bool-flags",
    ],
    "lo": (
        "Apply the float-only coercion rule across a kwargs dict: promote "
        "any float-valued entry to a 0-D tensor while leaving int / bool "
        "control flags raw so axis-indices and keepdim flags continue to "
        "be Python primitives the raw torch fn can use."
    ),
    "prompt_body": (
        "Implement `ex3_coerce_kwargs(kwargs)`. Return a NEW dict where "
        "every entry whose value is a Python `float` (NOT int, NOT bool) "
        "has been replaced with `t.tensor(float(value))`. Everything else "
        "is identity pass-through.\n\n"
        "Rules:\n\n"
        "- `float` value → `t.tensor(float(v))` (0-D float32 tensor).\n"
        "- `int` value → pass-through (control-flag use: `dim`, `step`, "
        "`groups`).\n"
        "- `bool` value → pass-through (control-flag use: `keepdim`, "
        "`unbiased`).\n"
        "- `torch.Tensor` value → pass-through unchanged (identity).\n"
        "- `MiniTensor` value → pass-through (unbox happens later, in a "
        "different stage).\n"
        "- All other types (`tuple`, `None`, `str`, ...) → pass-through.\n"
        "- Keys preserved exactly. Dict insertion order preserved (Python "
        "3.7+ semantics).\n\n"
        "Examples:\n\n"
        "```\n"
        "coerce_kwargs({'min': 0.0, 'max': 1.0, 'keepdim': True})\n"
        "  → {'min': tensor(0.0), 'max': tensor(1.0), 'keepdim': True}\n"
        "\n"
        "coerce_kwargs({'dim': 1, 'step': 2})\n"
        "  → {'dim': 1, 'step': 2}    # ints stay (axis indices)\n"
        "\n"
        "coerce_kwargs({})\n"
        "  → {}\n"
        "```\n\n"
        "Why float-only here (and NOT int): kwargs like `dim` and `step` "
        "are axis indices — passing a 0-D tensor where the raw torch op "
        "expects a Python int raises a `TypeError` deep in the C++ layer. "
        "Floats, by contrast, are almost always numeric parameters (`min`, "
        "`max`, `eps`, `p` for dropout) and should be tensors so the "
        "Recipe stores tensors everywhere."
    ),
    "stub": (
        "def ex3_coerce_kwargs(kwargs: dict) -> dict:\n"
        '    """Coerce float values in kwargs to 0-D tensors; leave int/bool/others alone."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === Empty kwargs ===\n"
        "    assert ex3_coerce_kwargs({}) == {}\n"
        "\n"
        "    # === Single float coerced ===\n"
        "    out = ex3_coerce_kwargs({'min': 0.5})\n"
        "    assert set(out.keys()) == {'min'}\n"
        "    assert isinstance(out['min'], t.Tensor)\n"
        "    assert out['min'].ndim == 0\n"
        "    assert out['min'].dtype == t.float32\n"
        "    assert out['min'].item() == 0.5\n"
        "\n"
        "    # === Int is preserved (axis index use case) ===\n"
        "    out = ex3_coerce_kwargs({'dim': 1})\n"
        "    assert out == {'dim': 1}, f'int must NOT be coerced (axis index): {out}'\n"
        "    assert isinstance(out['dim'], int)\n"
        "    assert not isinstance(out['dim'], t.Tensor)\n"
        "\n"
        "    # === Bool is preserved (keepdim flag use case) ===\n"
        "    out = ex3_coerce_kwargs({'keepdim': True, 'unbiased': False})\n"
        "    assert out['keepdim'] is True\n"
        "    assert out['unbiased'] is False\n"
        "\n"
        "    # === Mixed: float + int + bool ===\n"
        "    out = ex3_coerce_kwargs({'min': 0.0, 'max': 1.0, 'dim': 1, 'keepdim': True})\n"
        "    assert isinstance(out['min'], t.Tensor) and out['min'].item() == 0.0\n"
        "    assert isinstance(out['max'], t.Tensor) and out['max'].item() == 1.0\n"
        "    assert out['dim'] == 1 and isinstance(out['dim'], int)\n"
        "    assert out['keepdim'] is True\n"
        "\n"
        "    # === The subclass-of-int trap: True/False values must NOT coerce ===\n"
        "    # bool is a subclass of int in Python, but they're separate types\n"
        "    # for our purposes — control flags stay raw.\n"
        "    out = ex3_coerce_kwargs({'flag1': True, 'flag2': False})\n"
        "    assert out['flag1'] is True, f'bool kwarg must pass through, got {out[\"flag1\"]}'\n"
        "    assert out['flag2'] is False\n"
        "\n"
        "    # === torch.Tensor value is pass-through (identity) ===\n"
        "    raw = t.tensor([1.0, 2.0])\n"
        "    out = ex3_coerce_kwargs({'tensor_arg': raw})\n"
        "    assert out['tensor_arg'] is raw, 'torch.Tensor identity preserved'\n"
        "\n"
        "    # === MiniTensor value is pass-through (unbox happens later) ===\n"
        "    m = MiniTensor(t.tensor([1.0]))\n"
        "    out = ex3_coerce_kwargs({'src': m})\n"
        "    assert out['src'] is m, 'MiniTensor identity preserved at coerce stage'\n"
        "\n"
        "    # === Other types pass through ===\n"
        "    out = ex3_coerce_kwargs({'name': 'foo', 'shape': (3, 4), 'opt': None})\n"
        "    assert out == {'name': 'foo', 'shape': (3, 4), 'opt': None}\n"
        "\n"
        "    # === Dict insertion order preserved ===\n"
        "    # Build a kwargs with deliberate key ordering.\n"
        "    inp = {}\n"
        "    inp['z'] = 1.0\n"
        "    inp['a'] = 2.0\n"
        "    inp['m'] = True\n"
        "    out = ex3_coerce_kwargs(inp)\n"
        "    assert list(out.keys()) == ['z', 'a', 'm'], (\n"
        "        f'insertion order must be preserved, got {list(out.keys())}'\n"
        "    )\n"
        "\n"
        "    # === Return type is dict, not generator / list-of-tuples ===\n"
        "    out = ex3_coerce_kwargs({'x': 1.0})\n"
        "    assert isinstance(out, dict)\n"
        "\n"
        "    # === Original dict is NOT mutated (new dict returned) ===\n"
        "    inp = {'min': 0.0}\n"
        "    out = ex3_coerce_kwargs(inp)\n"
        "    assert inp == {'min': 0.0}, f'input dict must not be mutated; got {inp}'\n"
        "    assert inp['min'] == 0.0  # still a python float\n"
        "\n"
        "    # === Real-world use: pre-coerced kwargs flow through t.clamp ===\n"
        "    raw_x = t.tensor([-1.0, 0.5, 2.0])\n"
        "    coerced = ex3_coerce_kwargs({'min': 0.0, 'max': 1.0})\n"
        "    # Raw torch.clamp accepts tensors as min/max kwargs in modern PyTorch.\n"
        "    out_tensor = t.clamp(raw_x, **coerced)\n"
        "    assert t.allclose(out_tensor, t.tensor([0.0, 0.5, 1.0]))\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_coerce_kwargs(kwargs):\n"
        "    out = {}\n"
        "    for k, v in kwargs.items():\n"
        "        # bool is a subclass of int; both are NOT coerced.\n"
        "        # Only Python float gets promoted to a 0-D tensor.\n"
        "        if isinstance(v, float) and not isinstance(v, bool):\n"
        "            out[k] = t.tensor(float(v))\n"
        "        else:\n"
        "            out[k] = v\n"
        "    return out"
    ),
    "solution_notes": (
        "**Float-only coercion is the design choice.** Most kwargs that "
        "are floats are numeric parameters (`min`, `max`, `eps`, dropout "
        "`p`) that the back fn wants as tensors. Ints are almost always "
        "axis indices (`dim`, `step`, `groups`) — raw torch ops type-check "
        "these as Python ints. Coercing ints would break those ops.\n\n"
        "**`isinstance(v, float) and not isinstance(v, bool)`.** Belt-and-"
        "suspenders: `bool` is a subclass of `int`, not of `float` in "
        "Python — so `isinstance(True, float)` is already False, and the "
        "second clause is technically redundant. We keep it for clarity "
        "and to match ex1's defensive style. The same belt-and-suspenders "
        "appears in PyTorch's own type-check helpers.\n\n"
        "**New dict, not in-place mutation.** `coerce_kwargs` is in the "
        "wrapper hot path; callers reuse the original `kwargs` dict for "
        "Recipe.kwargs storage (the un-coerced version). Mutating in "
        "place would corrupt the Recipe.\n\n"
        "**This complements ex2's `coerce_args`.** Same numeric promotion, "
        "different container. The wrapper calls `coerce_args(args)` and "
        "`coerce_kwargs(kwargs)` in sequence before invoking the raw "
        "forward fn."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 3 — get-children-callable-param ex3
# ---------------------------------------------------------------------------

SPEC_GET_CHILDREN = {
    "atom_id": "get-children-callable-param",
    "subtopic": "Backprop: get_children callable param",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MODULE_CALL,
    "exercise_index": 3,
    "exercise_title": "Module.__call__ delegates to forward — and forward must raise NotImplementedError",
    "slug": "module-call-delegates-forward-raises-notimplemented",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["module", "call", "forward", "callable", "notimplemented"],
    "kcs": [
        "module-call-delegates-forward",
        "forward-abstract-raise",
    ],
    "lo": (
        "Apply the nn.Module callable convention: define `__call__` on the "
        "base class to delegate to `forward`, and define `forward` itself "
        "to raise `NotImplementedError` so subclasses are forced to be "
        "explicit about their compute."
    ),
    "prompt_body": (
        "Implement the `Module` base class with TWO methods:\n\n"
        "1. `__call__(self, *args, **kwargs)`:\n"
        "   - Delegates to `self.forward(*args, **kwargs)` and returns "
        "the result.\n"
        "   - Pass through all positional + keyword args verbatim.\n\n"
        "2. `forward(self, *args, **kwargs)`:\n"
        "   - Raises `NotImplementedError` with a message that includes "
        "`type(self).__name__` so the error reads something like "
        "`'MyLayer must implement forward'`.\n\n"
        "Return the `Module` class itself from `ex3_module_class()` so "
        "the tests can subclass it.\n\n"
        "Constraints:\n"
        "- A subclass that defines `forward` MUST be callable via "
        "`instance(...)` and return whatever its `forward` returns.\n"
        "- A subclass that does NOT define `forward` MUST raise "
        "`NotImplementedError` when called.\n"
        "- The error message MUST contain the subclass's class name "
        "(not the literal string 'Module')."
    ),
    "stub": (
        "def ex3_module_class():\n"
        '    """Return a Module class whose __call__ delegates to forward."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    Module = ex3_module_class()\n"
        "\n"
        "    # === Class has __call__ and forward ===\n"
        "    assert callable(Module), 'Module class itself must be callable for instantiation'\n"
        "    inst = Module()\n"
        "    assert hasattr(inst, '__call__'), 'instance must define __call__'\n"
        "    assert hasattr(inst, 'forward'), 'instance must define forward'\n"
        "\n"
        "    # === Subclass with forward → callable, returns forward's result ===\n"
        "    class AddOne(Module):\n"
        "        def forward(self, x):\n"
        "            return x + 1\n"
        "\n"
        "    layer = AddOne()\n"
        "    assert layer(t.tensor([1.0, 2.0])).tolist() == [2.0, 3.0], (\n"
        "        'instance call must delegate to forward and return its result'\n"
        "    )\n"
        "\n"
        "    # === args + kwargs both pass through ===\n"
        "    class Linear(Module):\n"
        "        def forward(self, x, bias=0.0):\n"
        "            return x + bias\n"
        "\n"
        "    lin = Linear()\n"
        "    out = lin(t.tensor([1.0, 2.0]), bias=10.0)\n"
        "    assert out.tolist() == [11.0, 12.0], (\n"
        "        f'kwargs must pass through __call__ to forward; got {out}'\n"
        "    )\n"
        "\n"
        "    # === Multiple positional args ===\n"
        "    class TwoArg(Module):\n"
        "        def forward(self, a, b):\n"
        "            return a * b\n"
        "    assert TwoArg()(t.tensor(3.0), t.tensor(4.0)).item() == 12.0\n"
        "\n"
        "    # === Subclass WITHOUT forward → calling raises NotImplementedError ===\n"
        "    class NoForward(Module):\n"
        "        pass\n"
        "\n"
        "    raised = False\n"
        "    msg = ''\n"
        "    try:\n"
        "        NoForward()(t.tensor([1.0]))\n"
        "    except NotImplementedError as e:\n"
        "        raised = True\n"
        "        msg = str(e)\n"
        "    assert raised, (\n"
        "        'subclass that does not define forward must raise NotImplementedError'\n"
        "    )\n"
        "    assert 'NoForward' in msg, (\n"
        "        f'error message must include subclass name, got: {msg!r}'\n"
        "    )\n"
        "    # The base class name 'Module' is too generic — must be the SUBCLASS name.\n"
        "    # (NoForward is the subclass; 'Module' is the base.)\n"
        "\n"
        "    # === Even raw Module().__call__() raises (since base forward raises) ===\n"
        "    raised = False\n"
        "    try:\n"
        "        Module()(t.tensor([1.0]))\n"
        "    except NotImplementedError:\n"
        "        raised = True\n"
        "    assert raised, 'base Module called directly must also raise'\n"
        "\n"
        "    # === Hook-friendly: subclass can override __call__ and STILL chain to base ===\n"
        "    # (This is the load-bearing reason for the __call__-around-forward split.)\n"
        "    class WithLogger(Module):\n"
        "        def __init__(self):\n"
        "            self.calls = 0\n"
        "        def forward(self, x):\n"
        "            return x * 2\n"
        "        def __call__(self, x):\n"
        "            self.calls += 1\n"
        "            return super().__call__(x)\n"
        "\n"
        "    w = WithLogger()\n"
        "    out = w(t.tensor([5.0]))\n"
        "    assert out.item() == 10.0\n"
        "    assert w.calls == 1, 'subclass __call__ logging should fire'\n"
        "    w(t.tensor([1.0]))\n"
        "    assert w.calls == 2\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_module_class():\n"
        "    class Module:\n"
        "        def __call__(self, *args, **kwargs):\n"
        "            return self.forward(*args, **kwargs)\n"
        "\n"
        "        def forward(self, *args, **kwargs):\n"
        "            raise NotImplementedError(\n"
        "                f'{type(self).__name__} must implement forward'\n"
        "            )\n"
        "    return Module"
    ),
    "solution_notes": (
        "**Why split `__call__` from `forward`.** Two reasons. (a) The "
        "user writes `forward` in subclasses; calling code uses `model(x)`, "
        "which routes through `__call__`. That gives a HOOK POINT: the base "
        "class (or a subclass) can wrap `__call__` to add logging, "
        "pre/post hooks, training-vs-eval branching — without touching "
        "subclass-overridden `forward`. (b) It matches the conventional "
        "PyTorch API surface.\n\n"
        "**`type(self).__name__`, NOT `self.__class__.__name__`.** Both "
        "work, but `type(self)` is the canonical way to get the runtime "
        "type — works correctly even when `__class__` has been monkey-"
        "patched. Same result for normal subclasses; safer in metaclass "
        "territory.\n\n"
        "**Why raise instead of returning silently.** A subclass that "
        "forgot `forward` returning None silently would propagate "
        "garbage through the model. Raising at the base level forces the "
        "subclass author to be explicit — fail loud at the right "
        "abstraction level. Same pattern as `abc.abstractmethod`.\n\n"
        "**This is the third facet of the atom.** Ex1 walked `__dict__` "
        "for children (the 'get-children' half). Ex2 recursed for "
        "`parameters()`. Ex3 covers the 'callable' half — the atom name "
        "is literally `get-children-callable-param`, and `__call__` "
        "delegation is what makes a Module a callable parameterized "
        "function instead of just a container."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 4 — grad-accumulate-on-leaf ex3
# ---------------------------------------------------------------------------

SPEC_GRAD_ACCUM = {
    "atom_id": "grad-accumulate-on-leaf",
    "subtopic": "Backprop: Grad accumulate on leaf",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_GRAD_ACCUM_MICROBATCH,
    "exercise_index": 3,
    "exercise_title": "gradient accumulation across micro-batches — divide loss by n, zero only after step",
    "slug": "gradient-accumulation-microbatch-divide-loss-by-n",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["microbatch", "accumulation", "effective-batch", "loss-scale", "training-loop"],
    "kcs": [
        "intentional-accumulation-across-microbatches",
        "loss-divide-by-microbatch-count",
    ],
    "lo": (
        "Apply the micro-batch gradient-accumulation pattern: divide each "
        "micro-batch's loss by the micro-batch count, accumulate gradients "
        "across micro-batches WITHOUT zero_grad in between, and zero only "
        "after the (simulated) optimizer step."
    ),
    "prompt_body": (
        "Implement `ex3_train_one_effective_batch(param, microbatch_grads, lr)`. "
        "Simulate one optimizer step over an EFFECTIVE batch that's split "
        "into multiple micro-batches.\n\n"
        "Inputs:\n"
        "- `param`: a `MiniTensor` with `requires_grad=True`, starting with "
        "`param.grad = None`. This is the parameter to update.\n"
        "- `microbatch_grads`: a list of `torch.Tensor`s — one gradient "
        "per micro-batch, all the same shape as `param.array`. Each "
        "tensor is what a backward pass over that micro-batch would have "
        "produced (NOT yet divided).\n"
        "- `lr`: float, learning rate.\n\n"
        "Behavior:\n\n"
        "1. **For each micro-batch gradient `g`** in `microbatch_grads`:\n"
        "   - Compute `g_scaled = g / len(microbatch_grads)` — emulates "
        "dividing the per-microbatch loss by N (the canonical gradient-"
        "accumulation pattern).\n"
        "   - Accumulate into `param.grad` using the ex1 pattern: set on "
        "first touch (when `param.grad is None`), add otherwise — and ALWAYS "
        "rebind (`param.grad = param.grad + g_scaled`), NOT in-place `+=`.\n"
        "   - DO NOT zero `param.grad` between micro-batches — the "
        "accumulation IS the simulated big-batch gradient.\n"
        "2. **After the loop, simulate one optimizer step**:\n"
        "   - `param.array = param.array - lr * param.grad`.\n"
        "3. **Then zero the grad** (set `param.grad = None`) — this is "
        "the convention from ex2.\n"
        "4. **Return** the updated `param` (same MiniTensor object).\n\n"
        "Constraints:\n"
        "- Accumulation MUST use rebinding `+`, not in-place `+=` (so "
        "external references to the old grad tensor stay intact).\n"
        "- `param.grad` MUST be `None` after the function returns "
        "(post-step zero).\n"
        "- `param.array` MUST reflect the AVERAGE gradient applied with "
        "`lr`, not the sum."
    ),
    "stub": (
        "def ex3_train_one_effective_batch(param: MiniTensor, microbatch_grads: list, lr: float):\n"
        '    """One optimizer step over an effective batch split into N micro-batches."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === Single micro-batch (degenerate case — equivalent to one normal step) ===\n"
        "    p = MiniTensor(t.tensor([10.0, 10.0]), requires_grad=True)\n"
        "    g1 = t.tensor([2.0, 4.0])\n"
        "    result = ex3_train_one_effective_batch(p, [g1], lr=0.1)\n"
        "    assert result is p, 'must return same MiniTensor object'\n"
        "    # 1 micro-batch: scaled grad = g1 / 1 = g1; param -= 0.1 * g1 → [9.8, 9.6]\n"
        "    assert t.allclose(p.array, t.tensor([9.8, 9.6])), (\n"
        "        f'single-microbatch update wrong: {p.array}'\n"
        "    )\n"
        "    assert p.grad is None, f'grad must be zeroed after step; got {p.grad}'\n"
        "\n"
        "    # === Four equal micro-batches — should equal one big step with the average gradient ===\n"
        "    p = MiniTensor(t.tensor([10.0]), requires_grad=True)\n"
        "    grads = [t.tensor([4.0]), t.tensor([4.0]), t.tensor([4.0]), t.tensor([4.0])]\n"
        "    ex3_train_one_effective_batch(p, grads, lr=0.1)\n"
        "    # Each grad / 4 = 1.0, accumulated 4 times = 4.0; lr*4.0 = 0.4 → 10.0 - 0.4 = 9.6\n"
        "    assert t.allclose(p.array, t.tensor([9.6])), (\n"
        "        f'4x identical grads should give a 4.0 average; got param={p.array}'\n"
        "    )\n"
        "    assert p.grad is None\n"
        "\n"
        "    # === Non-uniform micro-batches: gradient is the AVERAGE, not the sum ===\n"
        "    p = MiniTensor(t.tensor([0.0]), requires_grad=True)\n"
        "    grads = [t.tensor([10.0]), t.tensor([20.0])]   # average = 15\n"
        "    ex3_train_one_effective_batch(p, grads, lr=1.0)\n"
        "    # accumulated = 10/2 + 20/2 = 15; param = 0 - 1.0*15 = -15\n"
        "    assert t.allclose(p.array, t.tensor([-15.0])), (\n"
        "        f'should apply AVERAGE gradient, got {p.array}'\n"
        "    )\n"
        "\n"
        "    # === Equivalence: micro-batched matches one big batch with the averaged grad ===\n"
        "    # Reference: one-shot step with avg gradient.\n"
        "    p_ref = MiniTensor(t.tensor([5.0, 5.0]), requires_grad=True)\n"
        "    g_a, g_b, g_c = t.tensor([1.0, 0.0]), t.tensor([0.0, 1.0]), t.tensor([1.0, 1.0])\n"
        "    avg = (g_a + g_b + g_c) / 3\n"
        "    p_ref.array = p_ref.array - 0.5 * avg\n"
        "\n"
        "    p = MiniTensor(t.tensor([5.0, 5.0]), requires_grad=True)\n"
        "    ex3_train_one_effective_batch(p, [g_a, g_b, g_c], lr=0.5)\n"
        "    assert t.allclose(p.array, p_ref.array, atol=1e-7), (\n"
        "        f'microbatched != big-batch equivalent: {p.array} vs {p_ref.array}'\n"
        "    )\n"
        "\n"
        "    # === Rebinding semantics: no in-place += during accumulation ===\n"
        "    # If we hold a reference to the grad mid-loop, it must NOT change underneath us.\n"
        "    # We can't easily intercept mid-loop, so instead we verify final-grad-rebinding\n"
        "    # by checking that p.grad has been set to None (which can't happen if the\n"
        "    # earlier grads were in-place mutating the same buffer — see ex1).\n"
        "    p = MiniTensor(t.tensor([0.0]), requires_grad=True)\n"
        "    ex3_train_one_effective_batch(p, [t.tensor([1.0]), t.tensor([2.0])], lr=0.0)\n"
        "    # lr=0 → param unchanged but step still runs; grad still zeroed at end.\n"
        "    assert p.array.item() == 0.0\n"
        "    assert p.grad is None, 'zero_grad after step is mandatory'\n"
        "\n"
        "    # === Sequential calls: second call works fine after first cleared grad ===\n"
        "    p = MiniTensor(t.tensor([10.0]), requires_grad=True)\n"
        "    ex3_train_one_effective_batch(p, [t.tensor([2.0]), t.tensor([2.0])], lr=0.1)\n"
        "    after_first = p.array.clone()\n"
        "    ex3_train_one_effective_batch(p, [t.tensor([4.0]), t.tensor([4.0])], lr=0.1)\n"
        "    # First call: avg grad 2.0, step 0.2 → 9.8\n"
        "    # Second call: avg grad 4.0, step 0.4 → 9.4\n"
        "    assert t.allclose(after_first, t.tensor([9.8]))\n"
        "    assert t.allclose(p.array, t.tensor([9.4])), (\n"
        "        f'consecutive accumulation steps must not leak grads: {p.array}'\n"
        "    )\n"
        "\n"
        "    # === Two micro-batches with zero-sum gradients → no parameter update ===\n"
        "    p = MiniTensor(t.tensor([100.0]), requires_grad=True)\n"
        "    ex3_train_one_effective_batch(p, [t.tensor([1.0]), t.tensor([-1.0])], lr=10.0)\n"
        "    # avg = 0 → param unchanged.\n"
        "    assert t.allclose(p.array, t.tensor([100.0])), (\n"
        "        f'zero-sum micro-grads must not move param: {p.array}'\n"
        "    )\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_train_one_effective_batch(param, microbatch_grads, lr):\n"
        "    n = len(microbatch_grads)\n"
        "    # Accumulate per-microbatch gradients (each scaled by 1/n) into param.grad.\n"
        "    # Mirrors the canonical pattern: divide micro-loss by n inside the loop;\n"
        "    # NO zero_grad between micro-batches — the accumulation IS the big-batch grad.\n"
        "    for g in microbatch_grads:\n"
        "        g_scaled = g / n\n"
        "        if param.grad is None:\n"
        "            param.grad = g_scaled\n"
        "        else:\n"
        "            # Rebind, not in-place — external references stay valid.\n"
        "            param.grad = param.grad + g_scaled\n"
        "    # Simulated optimizer step: param.array -= lr * param.grad (leaf in-place is safe).\n"
        "    param.array = param.array - lr * param.grad\n"
        "    # Zero the grad NOW (after the step), not between micro-batches.\n"
        "    param.grad = None\n"
        "    return param"
    ),
    "solution_notes": (
        "**The division-by-N is the load-bearing detail.** Without it, "
        "accumulating N micro-batches gives an N-times-too-large gradient — "
        "your effective lr is silently 4x or 16x what you set. Some "
        "papers report this bug as 'the model trained fine but used the "
        "wrong lr'.\n\n"
        "**Why zero_grad goes AFTER the optimizer step, not between "
        "micro-batches.** Between micro-batches we WANT the gradients to "
        "stack — that's the whole pattern. Only after the step has "
        "consumed the accumulated gradient do we clear it for the next "
        "effective batch.\n\n"
        "**Rebind, never `+=`.** Ex1 covered this for safety against "
        "external references; here it matters for the same reason — "
        "if a logging hook snapshots `param.grad` mid-loop, in-place "
        "mutation would change the snapshot retroactively.\n\n"
        "**Why this is the third facet.** Ex1 was the per-touch primitive. "
        "Ex2 was the inter-step boundary (`zero_grad`). Ex3 is the "
        "intra-step intentional accumulation — a deliberate use of the "
        "ex1 mechanism that ex2's discipline (no leaked grads between "
        "STEPS) doesn't preclude. The three exercises together cover "
        "the full lifecycle of `param.grad`."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 5 — inplace-op-unsafe-warning ex3
# ---------------------------------------------------------------------------

SPEC_INPLACE = {
    "atom_id": "inplace-op-unsafe-warning",
    "subtopic": "Backprop: In-place op unsafe warning",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_DETACH,
    "exercise_index": 3,
    "exercise_title": "detach: new wrapper sharing array, no recipe, so in-place becomes safe",
    "slug": "detach-shares-array-no-recipe-safe-inplace",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["detach", "stop-gradient", "shared-storage", "graph-disconnect"],
    "kcs": [
        "detach-clears-recipe",
        "shared-storage-different-wrapper",
    ],
    "lo": (
        "Apply the detach pattern: return a new MiniTensor that shares the "
        "underlying .array (same object) but has recipe=None and "
        "requires_grad=False, so the in-place guard from ex1 no longer "
        "fires on the detached wrapper."
    ),
    "prompt_body": (
        "Implement two functions:\n\n"
        "1. **`ex3_detach(x: MiniTensor) -> MiniTensor`** — return a NEW "
        "`MiniTensor` such that:\n"
        "   - `result.array is x.array` (SAME torch.Tensor object — no "
        "copy).\n"
        "   - `result.recipe is None` (the recipe-chain ends here).\n"
        "   - `result.requires_grad is False` (no gradient flow through "
        "this point).\n"
        "   - `result is not x` (a fresh MiniTensor wrapper).\n\n"
        "2. **`ex3_add_inplace_safe(x: MiniTensor, y: MiniTensor) -> "
        "MiniTensor`** — same guard as ex1: if `x.recipe is not None`, "
        "raise `RuntimeError` with 'in-place' in the message; otherwise "
        "`x.array += y.array` and return `x`.\n\n"
        "Why both: the test scenario is the standard 'I have an "
        "intermediate, want to mutate it explicitly'. The fix is "
        "`detach()` to peel the recipe off, then the in-place guard "
        "from ex1 passes.\n\n"
        "Constraints:\n"
        "- `detach()` must NOT clone the underlying tensor. Identity "
        "of `.array` is critical for storage-sharing semantics.\n"
        "- Mutating the detached wrapper's `.array` MUST mutate the "
        "original's `.array` too (they share storage).\n"
        "- The original tensor's `.recipe` and `.requires_grad` MUST "
        "be unchanged."
    ),
    "stub": (
        "def ex3_detach(x: MiniTensor) -> MiniTensor:\n"
        '    """Return a fresh wrapper sharing x.array, with recipe=None, requires_grad=False."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def ex3_add_inplace_safe(x: MiniTensor, y: MiniTensor) -> MiniTensor:\n"
        '    """Same in-place guard as ex1: refuse when x.recipe is not None."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === detach: returns a new wrapper, same .array ===\n"
        "    raw = t.tensor([1.0, 2.0, 3.0])\n"
        "    x = MiniTensor(raw, requires_grad=True)\n"
        "    x.recipe = Recipe(t.add, (raw,), {}, {0: x})\n"
        "\n"
        "    detached = ex3_detach(x)\n"
        "    assert isinstance(detached, MiniTensor)\n"
        "    assert detached is not x, 'detach must return a new wrapper, not the same object'\n"
        "    assert detached.array is x.array, (\n"
        "        f'detached.array must BE x.array (same torch.Tensor object), got identity mismatch'\n"
        "    )\n"
        "    assert detached.recipe is None, (\n"
        "        f'detached.recipe must be None, got {detached.recipe!r}'\n"
        "    )\n"
        "    assert detached.requires_grad is False, (\n"
        "        f'detached.requires_grad must be False, got {detached.requires_grad}'\n"
        "    )\n"
        "\n"
        "    # === Original's recipe and requires_grad are UNCHANGED ===\n"
        "    assert x.recipe is not None, 'detach must not mutate x.recipe'\n"
        "    assert x.requires_grad is True, 'detach must not mutate x.requires_grad'\n"
        "\n"
        "    # === Storage IS shared: mutating detached.array also changes x.array ===\n"
        "    detached.array[0] = 99.0\n"
        "    assert x.array[0].item() == 99.0, (\n"
        "        'detached and original must share storage — mutation on one visible on other'\n"
        "    )\n"
        "    # restore for further tests\n"
        "    detached.array[0] = 1.0\n"
        "\n"
        "    # === The use case: in-place add on the original REFUSES (recipe is set) ===\n"
        "    raised = False\n"
        "    try:\n"
        "        ex3_add_inplace_safe(x, MiniTensor(t.tensor([10.0, 10.0, 10.0])))\n"
        "    except RuntimeError as e:\n"
        "        raised = True\n"
        "        msg = str(e).lower()\n"
        "        assert 'in-place' in msg or 'inplace' in msg or 'in place' in msg, (\n"
        "            f'error must mention in-place, got: {e}'\n"
        "        )\n"
        "    assert raised, 'in-place on x (recipe-carrying) must refuse'\n"
        "\n"
        "    # === Now via detach: in-place on the detached wrapper SUCCEEDS ===\n"
        "    detached = ex3_detach(x)\n"
        "    y = MiniTensor(t.tensor([10.0, 10.0, 10.0]))\n"
        "    result = ex3_add_inplace_safe(detached, y)\n"
        "    assert result is detached\n"
        "    assert t.allclose(detached.array, t.tensor([11.0, 12.0, 13.0])), (\n"
        "        f'in-place add via detach failed: {detached.array}'\n"
        "    )\n"
        "    # And critically, x.array IS detached.array → it ALSO changed (shared storage).\n"
        "    assert t.allclose(x.array, t.tensor([11.0, 12.0, 13.0])), (\n"
        "        'x.array also reflects mutation (shared storage with detached)'\n"
        "    )\n"
        "\n"
        "    # === Leaf (no recipe) detach is a no-op semantically but still creates a new wrapper ===\n"
        "    leaf = MiniTensor(t.tensor([1.0, 2.0]), requires_grad=True)\n"
        "    assert leaf.recipe is None\n"
        "    d = ex3_detach(leaf)\n"
        "    assert d is not leaf\n"
        "    assert d.array is leaf.array\n"
        "    assert d.recipe is None\n"
        "    assert d.requires_grad is False\n"
        "    # Leaf's requires_grad stays True after detach (we never mutated the original).\n"
        "    assert leaf.requires_grad is True\n"
        "\n"
        "    # === Double detach is idempotent (still works) ===\n"
        "    dd = ex3_detach(d)\n"
        "    assert dd.array is leaf.array  # transitively shared\n"
        "    assert dd.recipe is None\n"
        "    assert dd.requires_grad is False\n"
        "\n"
        "    # === Detach has no requires_grad flag — always False ===\n"
        "    # Even when input was requires_grad=False, detach still returns rg=False.\n"
        "    no_rg = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "    d = ex3_detach(no_rg)\n"
        "    assert d.requires_grad is False\n"
        "\n"
        "    # === The error message specifically mentions in-place ===\n"
        "    bad = MiniTensor(t.tensor([1.0]))\n"
        "    bad.recipe = Recipe(t.add, (), {}, {})\n"
        "    raised = False\n"
        "    try:\n"
        "        ex3_add_inplace_safe(bad, MiniTensor(t.tensor([2.0])))\n"
        "    except RuntimeError:\n"
        "        raised = True\n"
        "    assert raised\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_detach(x):\n"
        "    # New wrapper, SAME underlying torch.Tensor (no copy).\n"
        "    # recipe=None severs the graph at this point; requires_grad=False\n"
        "    # tells downstream wrap_forward_fn 'don't track gradients through me'.\n"
        "    return MiniTensor(x.array, requires_grad=False, recipe=None)\n"
        "\n"
        "\n"
        "def ex3_add_inplace_safe(x, y):\n"
        "    if x.recipe is not None:\n"
        "        raise RuntimeError(\n"
        "            'in-place op forbidden on a Tensor with a recipe — '\n"
        "            'would corrupt cached values on the graph; '\n"
        "            'use detach() first if you really mean it'\n"
        "        )\n"
        "    x.array += y.array\n"
        "    return x"
    ),
    "solution_notes": (
        "**Storage sharing is the whole point — and the danger.** "
        "`detach()` returns a wrapper whose `.array IS x.array` (same "
        "Python object). Mutating the detached wrapper's `.array` mutates "
        "the original's `.array` too. The guard from ex1 only inspects "
        "the WRAPPER'S `.recipe`, not the underlying storage — so by "
        "swapping wrappers we bypass the guard while still mutating the "
        "same memory. The user has explicitly chosen to take ownership "
        "of the consequence.\n\n"
        "**When this is correct vs. when it's a bug.** Correct: in `with "
        "torch.no_grad():` blocks, in stop-gradient ops (BN running stats, "
        "target networks), at clearly-marked subgraph boundaries. Buggy: "
        "anywhere the user 'just wanted the .array' and didn't realize "
        "detaching also disconnects future ops from the graph.\n\n"
        "**Detach vs. clone+detach.** `clone()` copies storage; `detach()` "
        "shares storage. `x.detach().clone()` is the common idiom for 'I "
        "want both: a detached AND a separate-storage tensor I can mutate "
        "freely without affecting x'. The drill keeps detach as the "
        "minimal primitive — clone is a separate concern.\n\n"
        "**Contrast with ex2's context manager.** `inplace_unsafe()` "
        "disables the guard globally for a code block. `detach()` "
        "disables it surgically by replacing the wrapper. Same goal, "
        "different ergonomic granularity. PyTorch ships both."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 6 — parameter-subclass-of-tensor ex3
# ---------------------------------------------------------------------------

SPEC_PARAMETER = {
    "atom_id": "parameter-subclass-of-tensor",
    "subtopic": "Backprop: Parameter subclasses Tensor",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_FREEZE_PARAMETERS,
    "exercise_index": 3,
    "exercise_title": "freeze(module, prefix): flip requires_grad=False on every Parameter whose name starts with prefix",
    "slug": "freeze-module-prefix-flip-requires-grad-on-parameters",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["freeze", "fine-tune", "prefix", "requires-grad", "parameter"],
    "kcs": [
        "freeze-by-name-prefix",
        "type-tag-stable-under-rg-flip",
    ],
    "lo": (
        "Apply the freeze-by-prefix pattern: walk a module's parameters() "
        "and set requires_grad=False on every Parameter whose dotted name "
        "starts with a given prefix, leaving the Parameter type tag "
        "(and thus trainable_params filtering) intact."
    ),
    "prompt_body": (
        "We've given you `Parameter` (subclass of MiniTensor with "
        "`requires_grad=True` default), a `Module` base class, and a "
        "`parameters(self)` generator that yields `(dotted_name, "
        "Parameter)` for every Parameter on the module (recursively).\n\n"
        "Implement `ex3_freeze(module, prefix)`. Walk `module.parameters()` "
        "and set `p.requires_grad = False` on every Parameter whose "
        "yielded `dotted_name` starts with `prefix` (use `str.startswith`).\n\n"
        "Behavior:\n\n"
        "- Mutate Parameters in place (no return value needed; return "
        "`None`).\n"
        "- `prefix` is a literal string match via `startswith`, NOT a "
        "regex or glob.\n"
        "- Parameters whose name does NOT start with `prefix` are "
        "untouched.\n"
        "- After the call, the frozen Parameters are STILL Parameters "
        "(type unchanged) — they just have `requires_grad=False`.\n\n"
        "Examples:\n\n"
        "```\n"
        "model = MLPWithEncoder()  # has 'encoder.fc1.weight', 'encoder.fc1.bias',\n"
        "                          #      'head.weight', 'head.bias'\n"
        "freeze(model, 'encoder.')\n"
        "  → 'encoder.fc1.weight' and 'encoder.fc1.bias' now rg=False\n"
        "  → 'head.weight' and 'head.bias' still rg=True\n"
        "\n"
        "freeze(model, 'encoder.fc1.weight')\n"
        "  → only that specific weight is frozen; bias stays trainable\n"
        "```\n\n"
        "Constraints:\n"
        "- Use `for name, p in module.parameters(): ...` — do NOT "
        "re-implement the walk.\n"
        "- Do NOT change `p.array` or `p.recipe` — only flip "
        "`requires_grad`.\n"
        "- `parameters()` STILL yields the frozen params after freezing "
        "(they're still Parameters; only the flag changed)."
    ),
    "stub": (
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "\n"
        "\n"
        "class Module:\n"
        "    def parameters(self):\n"
        "        \"\"\"DFS yield of (dotted_name, Parameter) leaves.\"\"\"\n"
        "        for name, val in self.__dict__.items():\n"
        "            if isinstance(val, Parameter):\n"
        "                yield name, val\n"
        "            elif isinstance(val, Module):\n"
        "                for sub_name, sub_val in val.parameters():\n"
        "                    yield f'{name}.{sub_name}', sub_val\n"
        "\n"
        "\n"
        "def ex3_freeze(module, prefix: str) -> None:\n"
        '    """Set requires_grad=False on every Parameter whose dotted name starts with prefix."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === Single-level model: freeze one specific param ===\n"
        "    class Linear(Module):\n"
        "        def __init__(self):\n"
        "            self.weight = Parameter(t.randn(4, 3))\n"
        "            self.bias = Parameter(t.zeros(4))\n"
        "\n"
        "    lin = Linear()\n"
        "    assert lin.weight.requires_grad is True\n"
        "    assert lin.bias.requires_grad is True\n"
        "\n"
        "    ex3_freeze(lin, 'weight')\n"
        "    assert lin.weight.requires_grad is False, 'weight must be frozen'\n"
        "    assert lin.bias.requires_grad is True, (\n"
        "        f'bias must stay trainable; rg={lin.bias.requires_grad}'\n"
        "    )\n"
        "\n"
        "    # === Two-level model: freeze a subtree by prefix ===\n"
        "    class MLPWithEncoder(Module):\n"
        "        def __init__(self):\n"
        "            self.encoder = Linear()\n"
        "            self.head = Linear()\n"
        "\n"
        "    model = MLPWithEncoder()\n"
        "    # All four params should start trainable.\n"
        "    for name, p in model.parameters():\n"
        "        assert p.requires_grad is True, f'{name} should start trainable'\n"
        "\n"
        "    ex3_freeze(model, 'encoder.')\n"
        "    flags = {name: p.requires_grad for name, p in model.parameters()}\n"
        "    assert flags['encoder.weight'] is False\n"
        "    assert flags['encoder.bias'] is False\n"
        "    assert flags['head.weight'] is True\n"
        "    assert flags['head.bias'] is True\n"
        "\n"
        "    # === Frozen Parameters are STILL Parameters (type unchanged) ===\n"
        "    assert isinstance(model.encoder.weight, Parameter), (\n"
        "        'freeze must not change the type — only the flag'\n"
        "    )\n"
        "    assert isinstance(model.encoder.bias, Parameter)\n"
        "\n"
        "    # === Frozen Parameters are still yielded by parameters() ===\n"
        "    names_after = [n for n, _ in model.parameters()]\n"
        "    assert names_after == ['encoder.weight', 'encoder.bias',\n"
        "                           'head.weight', 'head.bias'], (\n"
        "        f'parameters() must still yield all four; got {names_after}'\n"
        "    )\n"
        "\n"
        "    # === Specific-param freeze ===\n"
        "    model2 = MLPWithEncoder()\n"
        "    ex3_freeze(model2, 'encoder.weight')\n"
        "    # Only encoder.weight should be frozen.\n"
        "    flags = {name: p.requires_grad for name, p in model2.parameters()}\n"
        "    assert flags['encoder.weight'] is False\n"
        "    assert flags['encoder.bias'] is True, (\n"
        "        'specific-param freeze must not affect siblings'\n"
        "    )\n"
        "    assert flags['head.weight'] is True\n"
        "    assert flags['head.bias'] is True\n"
        "\n"
        "    # === Empty prefix freezes EVERYTHING (startswith('') is always True) ===\n"
        "    model3 = MLPWithEncoder()\n"
        "    ex3_freeze(model3, '')\n"
        "    for name, p in model3.parameters():\n"
        "        assert p.requires_grad is False, f'{name} should be frozen'\n"
        "\n"
        "    # === Non-matching prefix is a no-op ===\n"
        "    model4 = MLPWithEncoder()\n"
        "    ex3_freeze(model4, 'no_such_prefix.')\n"
        "    for name, p in model4.parameters():\n"
        "        assert p.requires_grad is True, f'{name} should still be trainable'\n"
        "\n"
        "    # === Three-level nesting works ===\n"
        "    class Block(Module):\n"
        "        def __init__(self):\n"
        "            self.inner = Linear()\n"
        "\n"
        "    class Net(Module):\n"
        "        def __init__(self):\n"
        "            self.block = Block()\n"
        "            self.classifier = Linear()\n"
        "\n"
        "    net = Net()\n"
        "    ex3_freeze(net, 'block.')\n"
        "    flags = {n: p.requires_grad for n, p in net.parameters()}\n"
        "    assert flags['block.inner.weight'] is False\n"
        "    assert flags['block.inner.bias'] is False\n"
        "    assert flags['classifier.weight'] is True\n"
        "    assert flags['classifier.bias'] is True\n"
        "\n"
        "    # === Return value is None ===\n"
        "    model5 = MLPWithEncoder()\n"
        "    ret = ex3_freeze(model5, 'head.')\n"
        "    assert ret is None, f'freeze should return None, got {ret!r}'\n"
        "\n"
        "    # === .array is untouched ===\n"
        "    model6 = MLPWithEncoder()\n"
        "    arr_before = model6.encoder.weight.array.clone()\n"
        "    ex3_freeze(model6, 'encoder.')\n"
        "    assert t.allclose(model6.encoder.weight.array, arr_before), (\n"
        "        '.array must be untouched — only requires_grad flips'\n"
        "    )\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_freeze(module, prefix):\n"
        "    for name, p in module.parameters():\n"
        "        if name.startswith(prefix):\n"
        "            p.requires_grad = False"
    ),
    "solution_notes": (
        "**`startswith` is the canonical prefix test.** It returns True "
        "for an empty prefix (a useful 'freeze all' shortcut) and False "
        "for non-prefixes. No glob/regex needed — we WANT the strict "
        "literal match, since prefixes ARE the natural hierarchy "
        "encoded by dotted names.\n\n"
        "**Why mutate, not return a new module.** Modules are stateful — "
        "they carry buffers, configs, registered hooks. Cloning to flip "
        "one flag wastes memory and breaks any external reference. "
        "PyTorch's idiom is exactly this in-place mutation: "
        "`for p in model.encoder.parameters(): p.requires_grad = False`.\n\n"
        "**Why the type-tag persists.** A frozen Parameter is STILL a "
        "Parameter. The type identifies 'this is trainable state, "
        "possibly frozen', distinct from 'this is a buffer (running mean)' "
        "or 'this is an incidental tensor'. `trainable_params` from ex2 "
        "filters by TYPE, not by `requires_grad` — frozen Parameters still "
        "show up in state_dict / checkpointing, just don't receive gradient "
        "updates.\n\n"
        "**Why this is the third facet.** Ex1 defined the type. Ex2 used "
        "the type as a filter. Ex3 toggles the orthogonal `requires_grad` "
        "flag on instances found via that filter — showing the type-tag "
        "and the gradient-flag are independent axes. Same Parameter, "
        "different temporary state."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 7 — sum-and-broadcast-duality ex3
# ---------------------------------------------------------------------------

SPEC_SUM_BROADCAST = {
    "atom_id": "sum-and-broadcast-duality",
    "subtopic": "Backprop: sum/broadcast duality",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MEAN_BACK,
    "exercise_index": 3,
    "exercise_title": "mean_back: derive from sum_back by dividing by the reduction count",
    "slug": "mean-back-derive-from-sum-back-divide-by-reduction-count",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["mean", "back-fn", "sum-back", "scalar-multiple", "chain-rule"],
    "kcs": [
        "mean-is-scaled-sum",
        "scalar-factor-passes-through-backward",
    ],
    "lo": (
        "Apply the chain rule's scalar-multiple invariance to derive "
        "mean_back from sum_back: since mean(x, dim) = sum(x, dim) / N, "
        "the backward is the same broadcast-back divided by the same N."
    ),
    "prompt_body": (
        "Implement two functions, building on ex1:\n\n"
        "**1. `ex3_sum_back(grad_out, out, x, dim, keepdim=False)`** — "
        "same as ex1. Re-insert axis via `unsqueeze` if `keepdim=False`, "
        "then `.expand_as(x).clone()`.\n\n"
        "**2. `ex3_mean_back(grad_out, out, x, dim, keepdim=False)`** — "
        "backward for `out = x.mean(dim=dim, keepdim=keepdim)`. Reuse "
        "`ex3_sum_back` and divide the result by `x.shape[dim]` (the "
        "reduction count):\n\n"
        "```python\n"
        "def mean_back(grad_out, out, x, dim, keepdim=False):\n"
        "    n = x.shape[dim]\n"
        "    return sum_back(grad_out, out, x, dim, keepdim) / n\n"
        "```\n\n"
        "Inputs (both functions):\n"
        "- `grad_out`: `torch.Tensor`, gradient flowing in from the next "
        "node — shape matches `out`.\n"
        "- `out`: `torch.Tensor`, the FORWARD output (kept for ABI "
        "consistency with PyTorch back-fns — `mean_back` doesn't actually "
        "use it).\n"
        "- `x`: `torch.Tensor`, the input the forward op reduced over.\n"
        "- `dim`: int, axis index.\n"
        "- `keepdim`: bool, whether the forward kept the axis as size 1.\n\n"
        "Constraints:\n"
        "- `mean_back` MUST reuse `sum_back` (don't reimplement the "
        "broadcasting from scratch).\n"
        "- Output shape must equal `x.shape`.\n"
        "- Output values must match `torch.autograd` on the equivalent "
        "forward."
    ),
    "stub": (
        "def ex3_sum_back(grad_out, out, x, dim: int, keepdim: bool = False):\n"
        '    """Backward of x.sum(dim, keepdim). Reused by mean_back."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def ex3_mean_back(grad_out, out, x, dim: int, keepdim: bool = False):\n"
        '    """Backward of x.mean(dim, keepdim) — sum_back / reduction_count."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === sum_back sanity (precondition for mean_back) ===\n"
        "    x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "    g = ex3_sum_back(t.ones(3), x.sum(dim=1), x, dim=1, keepdim=False)\n"
        "    assert g.shape == (3, 4)\n"
        "    assert t.allclose(g, t.ones(3, 4))\n"
        "\n"
        "    # === mean_back: dim=1 over (3, 4), uniform grad_out ===\n"
        "    x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "    out = x.mean(dim=1)  # shape (3,)\n"
        "    g_in = t.tensor([1.0, 1.0, 1.0])\n"
        "    g = ex3_mean_back(g_in, out, x, dim=1, keepdim=False)\n"
        "    assert g.shape == x.shape, f'shape: {g.shape}'\n"
        "    # mean over a 4-element axis means each input contributes 1/4 — so the\n"
        "    # gradient of a uniform-1 grad_out is uniform 1/4.\n"
        "    expected = t.full((3, 4), 0.25)\n"
        "    assert t.allclose(g, expected), (\n"
        "        f'mean_back uniform grad_out=1 → all 1/4; got {g}'\n"
        "    )\n"
        "\n"
        "    # === mean_back vs torch.autograd ===\n"
        "    x_ref = t.arange(12, dtype=t.float32).reshape(3, 4).clone().detach().requires_grad_(True)\n"
        "    y = x_ref.mean(dim=1).sum()\n"
        "    y.backward()\n"
        "    g_ours = ex3_mean_back(t.ones(3), x_ref.detach().mean(dim=1), x_ref.detach(),\n"
        "                           dim=1, keepdim=False)\n"
        "    assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "        f'mean_back disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        "    )\n"
        "\n"
        "    # === mean_back: dim=0 (different reduction count) ===\n"
        "    x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "    out = x.mean(dim=0)  # shape (4,), reduction size = 3\n"
        "    g_in = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "    g = ex3_mean_back(g_in, out, x, dim=0, keepdim=False)\n"
        "    assert g.shape == (3, 4)\n"
        "    # Each row should be g_in / 3.\n"
        "    expected_row = g_in / 3.0\n"
        "    for row in g:\n"
        "        assert t.allclose(row, expected_row), f'row mismatch: {row} vs {expected_row}'\n"
        "\n"
        "    # === mean_back keepdim=True ===\n"
        "    x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "    out = x.mean(dim=1, keepdim=True)  # shape (3, 1)\n"
        "    g_in = t.tensor([[1.0], [2.0], [3.0]])\n"
        "    g = ex3_mean_back(g_in, out, x, dim=1, keepdim=True)\n"
        "    assert g.shape == (3, 4)\n"
        "    # Each row should be g_in[i] / 4 broadcast to length 4.\n"
        "    assert t.allclose(g[0], t.full((4,), 0.25))\n"
        "    assert t.allclose(g[1], t.full((4,), 0.5))\n"
        "    assert t.allclose(g[2], t.full((4,), 0.75))\n"
        "\n"
        "    # === Larger 3-D tensor ===\n"
        "    x = t.randn(2, 3, 5)\n"
        "    out_ref = x.mean(dim=2)\n"
        "    g_in = t.randn(2, 3)\n"
        "    g = ex3_mean_back(g_in, out_ref, x, dim=2, keepdim=False)\n"
        "    assert g.shape == (2, 3, 5)\n"
        "    # Vs autograd:\n"
        "    x_ref = x.clone().detach().requires_grad_(True)\n"
        "    (x_ref.mean(dim=2) * g_in).sum().backward()\n"
        "    assert t.allclose(g, x_ref.grad, atol=1e-6), (\n"
        "        f'3-D mean_back disagrees with autograd; max diff '\n"
        "        f'{(g - x_ref.grad).abs().max().item()}'\n"
        "    )\n"
        "\n"
        "    # === Adjoint identity holds for mean (analogue of sum's adjoint) ===\n"
        "    # <mean(x, dim), y> should equal <x, mean_back(y, x, dim) > / N? No —\n"
        "    # the correct identity is: <Ax, y> == <x, A^T y> where A=mean = sum/N\n"
        "    # so A^T = sum_back / N = mean_back. Thus <mean(x, dim), y> == <x, mean_back(y, ..., dim)>.\n"
        "    t.manual_seed(0)\n"
        "    x = t.randn(3, 4)\n"
        "    Ax = x.mean(dim=1)\n"
        "    y = t.randn(3)\n"
        "    lhs = (Ax * y).sum()\n"
        "    rhs = (x * ex3_mean_back(y, Ax, x, dim=1, keepdim=False)).sum()\n"
        "    assert t.allclose(lhs, rhs, atol=1e-5), (\n"
        "        f'mean adjoint identity: lhs={lhs.item()}, rhs={rhs.item()}'\n"
        "    )\n"
        "\n"
        "    # === The derived form: mean_back IS sum_back divided by N ===\n"
        "    # Same call on same inputs, verify the explicit relationship.\n"
        "    x = t.randn(3, 4)\n"
        "    g_in = t.randn(3)\n"
        "    out = x.mean(dim=1)\n"
        "    g_mean = ex3_mean_back(g_in, out, x, dim=1, keepdim=False)\n"
        "    g_sum = ex3_sum_back(g_in, x.sum(dim=1), x, dim=1, keepdim=False)\n"
        "    assert t.allclose(g_mean, g_sum / x.shape[1], atol=1e-6), (\n"
        "        'mean_back must equal sum_back / N exactly'\n"
        "    )\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_sum_back(grad_out, out, x, dim, keepdim=False):\n"
        "    if not keepdim:\n"
        "        grad_out = grad_out.unsqueeze(dim)\n"
        "    return grad_out.expand_as(x).clone()\n"
        "\n"
        "\n"
        "def ex3_mean_back(grad_out, out, x, dim, keepdim=False):\n"
        "    # Mean = Sum / N → its derivative is sum_back / N (chain rule on a scalar mult).\n"
        "    n = x.shape[dim]\n"
        "    return ex3_sum_back(grad_out, out, x, dim, keepdim) / n"
    ),
    "solution_notes": (
        "**Mean is sum scaled.** `x.mean(dim=k) = x.sum(dim=k) / x.shape[k]`. "
        "Apply the chain rule: `d/dx[c * f(x)] = c * df/dx`. The scalar "
        "`1/N` passes through the linear backward unchanged — so "
        "`mean_back = sum_back / N`. No new derivation needed; the "
        "duality from ex1 transfers via scalar multiplication.\n\n"
        "**Why `x.shape[dim]`, not `out.numel()`.** For a single-axis "
        "mean, these are equal. For a multi-axis mean (`x.mean(dim=(0,1))`), "
        "the count is the product of the reduced axes' sizes. The "
        "drill keeps to single-axis for clarity — multi-axis is a "
        "straightforward extension once the principle is in.\n\n"
        "**Why `out` is in the signature but unused.** PyTorch's back-fn "
        "ABI passes `out` (the forward output) for back-fns that need it "
        "(e.g. `sigmoid_back` reuses `sigmoid(x)` from forward). `mean_back` "
        "doesn't need it, but keeping the parameter in the signature lets "
        "the dispatch table store all back-fns with one ABI.\n\n"
        "**The adjoint identity still holds.** Same `<Ax, y> = <x, A^T y>` "
        "as ex2 — just with `A = mean(dim)` and `A^T = mean_back`. "
        "Verifying this is the gold-standard test for any back-fn; the "
        "scalar `1/N` factor doesn't change the identity, only the "
        "magnitude on both sides.\n\n"
        "**Where this shows up in real models.** Every cross-entropy "
        "loss that averages over a batch uses `mean_back` on the reverse "
        "pass. Every BatchNorm running-mean update is a `mean` forward "
        "whose backward (if it weren't intentionally detached) would be "
        "`mean_back`. Same machinery as `sum_back`, just divided."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


# ---------------------------------------------------------------------------
# SPEC 8 — unbox-args-tensor-to-array ex3
# ---------------------------------------------------------------------------

SPEC_UNBOX = {
    "atom_id": "unbox-args-tensor-to-array",
    "subtopic": "Backprop: Unbox Tensor args to array",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_UNBOX_KWARGS,
    "exercise_index": 3,
    "exercise_title": "unbox_kwargs: dict-comprehension unboxes MiniTensor values, preserves keys + order",
    "slug": "unbox-kwargs-dict-preserves-keys-and-order",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["unbox", "kwargs", "dict", "wrap-forward", "container"],
    "kcs": [
        "unbox-kwargs-dict-comprehension",
        "preserve-keys-and-insertion-order",
    ],
    "lo": (
        "Apply the unboxing rule across a kwargs dict: replace each "
        "MiniTensor-valued entry with its .array, leaving keys and "
        "insertion order intact and all non-MiniTensor values "
        "pass-through."
    ),
    "prompt_body": (
        "Implement `ex3_unbox_kwargs(kwargs)`. Same unbox rule as ex1 "
        "(positional args) and ex2 (nested lists/tuples), but for a "
        "DICT container.\n\n"
        "Behavior:\n\n"
        "- For each `(k, v)` in `kwargs.items()`:\n"
        "  - If `isinstance(v, MiniTensor)`: replace `v` with `v.array`.\n"
        "  - Otherwise: keep `v` unchanged (identity pass-through).\n"
        "- Keys are preserved exactly.\n"
        "- Insertion order is preserved (Python 3.7+ dict semantics).\n"
        "- Return type is `dict` (NOT `collections.OrderedDict`, NOT a "
        "generator).\n"
        "- Original `kwargs` dict is NOT mutated — return a NEW dict.\n\n"
        "Examples:\n\n"
        "```\n"
        "unbox_kwargs({'src': m1, 'dim': 1})    → {'src': m1.array, 'dim': 1}\n"
        "unbox_kwargs({'min': 0.0, 'max': 1.0}) → {'min': 0.0, 'max': 1.0}   # pass-through\n"
        "unbox_kwargs({})                        → {}\n"
        "```\n\n"
        "Constraints:\n"
        "- Use `isinstance(v, MiniTensor)` — NOT duck-typing on `.array` "
        "(would catch numpy ndarrays which have an `.array` protocol).\n"
        "- Raw `torch.Tensor` values MUST pass through (not MiniTensor → "
        "unchanged).\n"
        "- The unboxed `.array` value MUST be identity-equal to the "
        "original (`result[k] is v.array`)."
    ),
    "stub": (
        "def ex3_unbox_kwargs(kwargs: dict) -> dict:\n"
        '    """Replace MiniTensor values with .array; pass-through everything else."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === Empty dict ===\n"
        "    assert ex3_unbox_kwargs({}) == {}\n"
        "\n"
        "    # === All non-MiniTensor values pass through ===\n"
        "    assert ex3_unbox_kwargs({'dim': 1, 'keepdim': True}) == {'dim': 1, 'keepdim': True}\n"
        "    assert ex3_unbox_kwargs({'min': 0.0, 'max': 1.0}) == {'min': 0.0, 'max': 1.0}\n"
        "\n"
        "    # === Single MiniTensor unwrapped ===\n"
        "    raw = t.tensor([1.0, 2.0, 3.0])\n"
        "    m = MiniTensor(raw)\n"
        "    result = ex3_unbox_kwargs({'src': m})\n"
        "    assert isinstance(result, dict)\n"
        "    assert set(result.keys()) == {'src'}\n"
        "    assert result['src'] is raw, (\n"
        "        'unboxed value must BE the same torch.Tensor object (identity)'\n"
        "    )\n"
        "\n"
        "    # === Mixed: MiniTensor + Python scalars ===\n"
        "    result = ex3_unbox_kwargs({'src': m, 'dim': 1, 'keepdim': True})\n"
        "    assert result['src'] is raw\n"
        "    assert result['dim'] == 1\n"
        "    assert result['keepdim'] is True\n"
        "\n"
        "    # === Raw torch.Tensor values pass through (NOT MiniTensor → not unboxed) ===\n"
        "    raw_passthrough = t.tensor([9.0])\n"
        "    result = ex3_unbox_kwargs({'tensor_arg': raw_passthrough})\n"
        "    assert result['tensor_arg'] is raw_passthrough, (\n"
        "        'raw torch.Tensor must pass through (only MiniTensor unboxes)'\n"
        "    )\n"
        "\n"
        "    # === Numpy ndarray pass-through (don't be fooled by .array protocol) ===\n"
        "    arr = np.array([1.0, 2.0, 3.0])\n"
        "    result = ex3_unbox_kwargs({'arr': arr})\n"
        "    assert result['arr'] is arr, (\n"
        "        'np.ndarray must pass through — isinstance(MiniTensor) is False '\n"
        "        '(do not duck-type on .array)'\n"
        "    )\n"
        "\n"
        "    # === Insertion order preserved ===\n"
        "    inp = {}\n"
        "    inp['z'] = m\n"
        "    inp['a'] = 1.0\n"
        "    inp['m'] = True\n"
        "    result = ex3_unbox_kwargs(inp)\n"
        "    assert list(result.keys()) == ['z', 'a', 'm'], (\n"
        "        f'insertion order must be preserved, got {list(result.keys())}'\n"
        "    )\n"
        "\n"
        "    # === Original kwargs NOT mutated ===\n"
        "    inp = {'src': m, 'dim': 1}\n"
        "    result = ex3_unbox_kwargs(inp)\n"
        "    assert inp == {'src': m, 'dim': 1}, (\n"
        "        f'input dict must not be mutated; got {inp}'\n"
        "    )\n"
        "    assert inp['src'] is m, 'value identity preserved in input'\n"
        "\n"
        "    # === Several MiniTensors in one dict ===\n"
        "    m2 = MiniTensor(t.tensor([4.0]))\n"
        "    m3 = MiniTensor(t.tensor([5.0]))\n"
        "    result = ex3_unbox_kwargs({'a': m, 'b': m2, 'c': m3})\n"
        "    assert result['a'] is m.array\n"
        "    assert result['b'] is m2.array\n"
        "    assert result['c'] is m3.array\n"
        "\n"
        "    # === Other non-MiniTensor types pass through (None, str, tuple, list) ===\n"
        "    result = ex3_unbox_kwargs({'opt': None, 'name': 'x', 'shape': (3, 4), 'lst': [1, 2]})\n"
        "    assert result == {'opt': None, 'name': 'x', 'shape': (3, 4), 'lst': [1, 2]}\n"
        "\n"
        "    # === Return is dict (not generator, not OrderedDict-required, not list of tuples) ===\n"
        "    result = ex3_unbox_kwargs({'x': 1})\n"
        "    assert type(result) is dict, (\n"
        "        f'must return a plain dict, got {type(result).__name__}'\n"
        "    )\n"
        "\n"
        "    # === The use case end-to-end: kwargs-as-raw-tensor flows into a torch op ===\n"
        "    # t.add accepts an 'alpha' scalar kwarg — but for a MiniTensor src kwarg case,\n"
        "    # something like t.scatter takes 'src' as a kwarg-positional.\n"
        "    x = t.tensor([0.0, 0.0, 0.0])\n"
        "    m_src = MiniTensor(t.tensor([1.0, 2.0, 3.0]))\n"
        "    kw = ex3_unbox_kwargs({'src': m_src})\n"
        "    # Simulate the raw op consuming kw — just identity here.\n"
        "    assert kw['src'] is m_src.array\n"
        "    print('ex3 ✓')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_unbox_kwargs(kwargs):\n"
        "    return {\n"
        "        k: v.array if isinstance(v, MiniTensor) else v\n"
        "        for k, v in kwargs.items()\n"
        "    }"
    ),
    "solution_notes": (
        "**Dict comprehension is the canonical one-liner.** Same shape "
        "as ex1's tuple comprehension — `isinstance(v, MiniTensor)` gate, "
        "`.array` swap on hit, identity pass-through on miss. The "
        "difference is just the container: dict vs tuple.\n\n"
        "**Why a new dict, not in-place mutation.** Callers retain the "
        "original `kwargs` dict for `Recipe.kwargs` storage — the Recipe "
        "wants the ORIGINAL types (MiniTensor) for graph-traversal "
        "purposes, while the raw forward fn needs the unboxed versions. "
        "Mutating in place would corrupt the Recipe.\n\n"
        "**`isinstance` over duck-typing.** Numpy ndarrays expose an "
        "`.array` interface protocol — duck-typing on `.array` would "
        "incorrectly unbox them. `isinstance(v, MiniTensor)` is precise: "
        "only our wrapper class gets the unboxing treatment.\n\n"
        "**Why this is the third facet of the atom.** Ex1 covered the "
        "tuple container (positional args). Ex2 covered nested "
        "list/tuple containers (cat / stack signatures). Ex3 covers the "
        "dict container (kwargs). Together they handle every Python "
        "container shape the wrapper layer encounters — and the rule is "
        "the same scalar check applied to each shape's natural traversal."
    ),
    "extra_imports": [AUTOGRAD_PREAMBLE],
}


SPECS = [
    SPEC_BOX,
    SPEC_COERCE,
    SPEC_GET_CHILDREN,
    SPEC_GRAD_ACCUM,
    SPEC_INPLACE,
    SPEC_PARAMETER,
    SPEC_SUM_BROADCAST,
    SPEC_UNBOX,
]


# ---------------------------------------------------------------------------
# Verifier — exec the solution + test_body against the same shared preamble
# the emitted notebooks use, so we catch authoring bugs before writing files.
# ---------------------------------------------------------------------------

PREAMBLE = """
import numpy as np
import torch as t
from torch import Tensor

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

grad_tracking_enabled = True

@dataclass
class Recipe:
    func: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)

class MiniTensor:
    def __init__(self, array, requires_grad: bool = False, recipe=None):
        self.array = array
        self.requires_grad = requires_grad
        self.recipe = recipe
        self.grad = None
    def __repr__(self):
        return f'MiniTensor({self.array!r}, requires_grad={self.requires_grad})'
"""


def _verify_all(specs):
    passed = 0
    failed = []
    for spec in specs:
        tag = f"{spec['atom_id']}/ex{spec['exercise_index']}"
        ns = {"__name__": "__main__"}
        t_seed_setup = "t.manual_seed(0)\nnp.random.seed(0)\n"
        try:
            exec(compile(PREAMBLE, f"<preamble:{tag}>", "exec"), ns)
            exec(compile(t_seed_setup, f"<seed:{tag}>", "exec"), ns)
            # Exec the stub first so any class/function definitions outside
            # the solution body (e.g. Parameter base class, Module base) land
            # in the namespace. The stub's `raise NotImplementedError` will
            # be SHADOWED when the solution body redefines the same names.
            try:
                exec(compile(spec["stub"], f"<stub:{tag}>", "exec"), ns)
            except Exception:
                pass
            exec(compile(spec["solution_body"], f"<solution:{tag}>", "exec"), ns)
            exec(compile(spec["test_body"], f"<test:{tag}>", "exec"), ns)
        except Exception:
            failed.append((tag, traceback.format_exc()))
            continue
        passed += 1
        print(f"  [verify] {tag}: ok")

    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[deepening_e_batch14] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_e_batch14] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_e_batch14] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
