#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA part-4 custom-Tensor capstone atoms.

Eight single-exercise standalones under ``prereqs_custom_tensor/``:

  * linear-affine-on-custom-tensor   — ex1  (Backprop: Linear affine on custom Tensor)
  * kaiming-uniform-sf-init           — ex1  (Init: Kaiming uniform SF init)
  * parameter-wrap-around-tensor      — ex1  (Backprop: Parameter wrap around Tensor)
  * module-base-class-custom          — ex1  (Backprop: Module base class custom)
  * logsumexp-cross-entropy           — ex1  (Loss: logsumexp cross-entropy)
  * arange-fancy-index-cross-entropy  — ex1  (Loss: arange fancy-index cross-entropy)
  * sgd-vanilla-from-scratch          — ex1  (Optimizer: SGD vanilla from scratch)
  * grads-dict-accumulate-parents     — ex1  (Backprop: grads dict accumulate parents)

These cover the ARENA part-4 capstone where the student stacks their hand-built
``Tensor`` / ``Recipe`` / ``BACK_FUNCS`` primitives into a working
``Linear`` + ``Module`` + cross-entropy + SGD training loop. Each drill is ONE
smaller skill — built on top of (not duplicating) batch-2 / batch-3 / batch-4
atoms.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_custom_tensor"


# ---------------------------------------------------------------- shared preamble

# Shared preamble — defines a MiniTensor wrapper + Recipe + grad_tracking_enabled
# matching the conventions used by batches 3 / 4. Drills can reference these
# names directly. For drills that need actual autograd wiring (linear affine,
# grads-dict accumulate) we DON'T inject BACK_FUNCS or wrap_forward_fn here; the
# drill body builds whatever forward/backward fragment it needs on top of the
# MiniTensor scaffolding.
_CUSTOM_TENSOR_PREAMBLE = (
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


# ---------------------------------------------------------------- recaps

RECAP_LINEAR_AFFINE = (
    "## Linear affine on custom Tensor — quick refresher\n"
    "\n"
    "A `Linear` layer is just the affine map `out = input @ weight + bias`. When "
    "you build it on YOUR hand-written `Tensor` (rather than `torch.nn.Linear`), "
    "TWO things change:\n"
    "\n"
    "1. **`@` and `+` must be wrapped ops.** Each produces a new `MiniTensor` "
    "with a `Recipe`, so the reverse pass can find the parents (`input`, "
    "`weight`, `bias`) and call the right `_back` fns.\n"
    "2. **`weight` and `bias` are `Parameter` (Tensor subclass).** They're "
    "leaves with `requires_grad=True`, so the wrapped `@` / `+` propagate grad "
    "through them via the OR rule.\n"
    "\n"
    "```python\n"
    "class Linear:\n"
    "    def __init__(self, in_f, out_f):\n"
    "        self.weight = Parameter(t.empty(in_f, out_f))   # init separately\n"
    "        self.bias   = Parameter(t.zeros(out_f))\n"
    "    def forward(self, x: MiniTensor) -> MiniTensor:\n"
    "        return mm(x, self.weight) + self.bias            # broadcast over batch\n"
    "```\n"
    "\n"
    "Output shape: `(B, in_f) @ (in_f, out_f) + (out_f,)` → `(B, out_f)`. The "
    "bias broadcasts over the batch axis — bias gradient is the broadcast-sum."
)

RECAP_KAIMING_SF = (
    "## Kaiming uniform with `sf = 1/sqrt(fan_in)` — quick refresher\n"
    "\n"
    "ARENA's `Linear.__init__` uses the **simplified Kaiming uniform** form:\n"
    "\n"
    "```\n"
    "sf = 1 / sqrt(fan_in)\n"
    "weight ~ Uniform(-sf, +sf)\n"
    "```\n"
    "\n"
    "where `fan_in` is the number of input units (`weight.shape[0]` for a "
    "`(in_features, out_features)` weight, or `in_channels * kernel_h * "
    "kernel_w` for a conv).\n"
    "\n"
    "**vs the `sqrt(6/fan_in)` form.** The more general Kaiming uniform is "
    "`U(-sqrt(6/fan_in), +sqrt(6/fan_in))` — chosen so `Var(w) == 2/fan_in` "
    "(activation-preserving for ReLU). The `1/sqrt(fan_in)` form is what "
    "PyTorch's `nn.Linear` actually ships, and it's what ARENA uses. They are "
    "DIFFERENT scales — don't conflate.\n"
    "\n"
    "Sampling recipe:\n"
    "```python\n"
    "sf = fan_in ** -0.5\n"
    "weight = (t.rand(in_f, out_f) * 2 - 1) * sf   # uniform on (-sf, +sf)\n"
    "```\n"
    "\n"
    "Sanity: `weight.std()` ≈ `sf / sqrt(3)` (population std of `U(-sf, sf)`)."
)

RECAP_PARAMETER_WRAP = (
    "## Parameter as wrapper-around-Tensor — quick refresher (anti-pattern)\n"
    "\n"
    "A tempting but WRONG `Parameter` design is to use **composition** — store "
    "the wrapped tensor as an attribute:\n"
    "\n"
    "```python\n"
    "# anti-pattern: NOT how nn.Parameter is built\n"
    "class WrapParam:\n"
    "    def __init__(self, tensor):\n"
    "        self.tensor = tensor             # composition, not inheritance\n"
    "```\n"
    "\n"
    "Why this looks fine at first: `WrapParam(t.zeros(3)).tensor` is the raw "
    "tensor, every op you want to do still works on `.tensor` directly.\n"
    "\n"
    "Why it's actually broken: **`isinstance(p, MiniTensor)` returns `False`**. "
    "Every helper in the autograd layer — `build_parents`, `unbox_args`, "
    "`get_children`, `parameters()` — filters by `isinstance(_, MiniTensor)`. A "
    "wrapped-Parameter is silently skipped by all of them, so:\n"
    "\n"
    "- The optimizer never sees it.\n"
    "- The reverse pass never accumulates grad on it.\n"
    "- The compute graph treats it like a plain Python object — invisible.\n"
    "\n"
    "The right design is inheritance: `class Parameter(MiniTensor): ...`. The "
    "drill below makes you reproduce the broken-by-composition version and "
    "*observe* the silent failure."
)

RECAP_MODULE_BASE = (
    "## `Module` base class from scratch — quick refresher\n"
    "\n"
    "A minimal `nn.Module` clone needs THREE things:\n"
    "\n"
    "1. **`__init__`** — initialize an internal store for parameters / "
    "submodules. Don't rely on the subclass to call `super().__init__()` and "
    "*still* work if it doesn't — but the canonical pattern is to require it.\n"
    "2. **`__setattr__`** — intercept attribute assignment. When the value is "
    "a `Parameter` (or `Module`), register it in the parameters / submodules "
    "store. Plain attributes (ints, lists, etc.) bypass.\n"
    "3. **`parameters()` walker** — yields every `Parameter`, walking recursively "
    "into submodules via depth-first traversal.\n"
    "\n"
    "```python\n"
    "class Module:\n"
    "    def __init__(self):\n"
    "        self._parameters = {}\n"
    "        self._modules = {}\n"
    "    def __setattr__(self, name, value):\n"
    "        if isinstance(value, Parameter):\n"
    "            self._parameters[name] = value\n"
    "        elif isinstance(value, Module):\n"
    "            self._modules[name] = value\n"
    "        object.__setattr__(self, name, value)\n"
    "    def parameters(self):\n"
    "        yield from self._parameters.values()\n"
    "        for m in self._modules.values():\n"
    "            yield from m.parameters()\n"
    "    def forward(self, *args, **kwargs):\n"
    "        raise NotImplementedError\n"
    "```\n"
    "\n"
    "**Why `__setattr__`.** Auto-registration is what makes `self.weight = "
    "Parameter(...)` Just Work. Without it the user has to call "
    "`self.register_parameter('weight', w)` explicitly."
)

RECAP_LOGSUMEXP_CE = (
    "## Cross-entropy via logsumexp — quick refresher\n"
    "\n"
    "The naive cross-entropy formula\n"
    "\n"
    "```\n"
    "loss[i] = -log(softmax(logits[i])[target[i]])\n"
    "        = -log( exp(logits[i, target[i]]) / sum_k exp(logits[i, k]) )\n"
    "```\n"
    "\n"
    "is numerically dangerous: `exp(logits)` overflows for logits > ~88 in "
    "float32. The stable rewrite uses `logsumexp`:\n"
    "\n"
    "```\n"
    "loss[i] = logsumexp(logits[i]) - logits[i, target[i]]\n"
    "```\n"
    "\n"
    "where `logsumexp(x) = log(sum_k exp(x_k - max(x))) + max(x)`. The "
    "`-max(x)` shift keeps every exp argument ≤ 0, so no overflow.\n"
    "\n"
    "`torch.logsumexp(logits, dim=-1)` ships this for you. Use it.\n"
    "\n"
    "Identity check: with `logits = [0, 0, 0]` the naive softmax is `[1/3, "
    "1/3, 1/3]` and the loss for any target is `log(3) ≈ 1.0986`. The "
    "logsumexp form gives `log(3) - 0 = 1.0986` — same answer, no overflow."
)

RECAP_ARANGE_FANCY = (
    "## `logits[arange(B), labels]` fancy-index — quick refresher\n"
    "\n"
    "Per-sample target-logit extraction shows up in EVERY classification loss. "
    "You have `logits` of shape `(B, C)` and `labels` of shape `(B,)`. You want "
    "a `(B,)` vector where position `i` holds `logits[i, labels[i]]`.\n"
    "\n"
    "**The wrong way (slow, breaks autograd).** Python loop:\n"
    "```python\n"
    "picked = t.stack([logits[i, labels[i]] for i in range(B)])\n"
    "```\n"
    "\n"
    "**The right way (vectorized).** Use advanced indexing with `arange`:\n"
    "```python\n"
    "picked = logits[t.arange(B), labels]    # shape (B,)\n"
    "```\n"
    "\n"
    "Mechanics: when you index with TWO 1-D tensors of the same length, "
    "PyTorch pairs them positionally. `arange(B) = [0,1,2,...,B-1]` and "
    "`labels = [l0, l1, ...]` zip to `[(0,l0), (1,l1), ...]`, picking one "
    "element per row.\n"
    "\n"
    "**Why `arange`, not `slice(None)`.** A plain `logits[:, labels]` would "
    "broadcast — `(B, B)` output — not `(B,)`. The `arange` makes the row "
    "axis advance in lockstep with the column axis."
)

RECAP_SGD_VANILLA = (
    "## Vanilla SGD from scratch — quick refresher\n"
    "\n"
    "The simplest possible optimizer. Given a list of parameters and a learning "
    "rate, one step of SGD is:\n"
    "\n"
    "```\n"
    "for p in params:\n"
    "    p <- p - lr * p.grad\n"
    "    p.grad <- None    # or zero — zero_grad-style\n"
    "```\n"
    "\n"
    "Two things to get right:\n"
    "\n"
    "1. **In-place mutation of `p.array`.** `p.array -= lr * p.grad` is "
    "preferred over `p.array = p.array - lr * p.grad` because the same tensor "
    "object stays alive — any external reference (state dicts, "
    "checkpointers) keeps pointing to the live weights.\n"
    "2. **Zero the grad after stepping.** Otherwise next backward call "
    "ACCUMULATES — you'd see two steps' worth of gradient on the next update. "
    "Setting `p.grad = None` is the cheapest reset; `p.grad.zero_()` also "
    "works.\n"
    "\n"
    "No momentum, no weight decay, no Nesterov — that's all in `SGD`'s richer "
    "siblings. This is the 5-line baseline."
)

RECAP_GRADS_DICT = (
    "## `grads` dict — accumulate into parents — quick refresher\n"
    "\n"
    "The reverse pass keeps a single `dict[MiniTensor, torch.Tensor]` mapping "
    "each node to its accumulated gradient. When the dispatcher computes a "
    "new contribution for a parent, it has to ADD to (not overwrite) the "
    "existing entry — a parent may receive contributions from multiple "
    "children:\n"
    "\n"
    "```python\n"
    "grads = {end_node: end_grad}\n"
    "for node in topo_order_reversed:\n"
    "    out_grad = grads[node]\n"
    "    for argnum, parent in node.recipe.parents.items():\n"
    "        back_fn = BACK_FUNCS.get(node.recipe.func, argnum)\n"
    "        contribution = back_fn(out_grad, node.array, *node.recipe.args)\n"
    "        grads[parent] = grads.get(parent, 0) + contribution   # ← THE accumulation\n"
    "```\n"
    "\n"
    "**Why `.get(parent, 0) + contribution`.** A parent on first visit isn't "
    "in `grads` yet — `get(parent, 0)` seeds with the additive identity. On "
    "subsequent visits the existing accumulator is the running sum. `+` is "
    "non-mutating, returning a fresh tensor.\n"
    "\n"
    "**Common bug.** Writing `grads[parent] = contribution` (overwrite) — "
    "looks fine on linear graphs but silently drops contributions when a "
    "node has two children. `y = x + x` makes `x` a parent of `y` at argnum 0 "
    "AND argnum 1; both contributions must be summed."
)


# ---------------------------------------------------------------- spec helper

def _spec(
    *,
    atom_id: str,
    subtopic: str,
    recap: str,
    ex_idx: int,
    ex_title: str,
    slug: str,
    bloom: str,
    difficulty_num: int,
    keywords: list[str],
    kcs: list[str],
    lo: str,
    prompt_body: str,
    stub: str,
    test_body: str,
    solution_body: str,
    solution_notes: str = "",
    extra_imports: list[str] | None = None,
) -> dict:
    dots = ("🔴" * difficulty_num) + ("⚪" * (5 - difficulty_num))
    merged_imports = [_CUSTOM_TENSOR_PREAMBLE] + list(extra_imports or [])
    return {
        "atom_id": atom_id,
        "subtopic": subtopic,
        "topic_folder": TOPIC,
        "atom_recap_md": recap,
        "exercise_index": ex_idx,
        "exercise_title": ex_title,
        "slug": slug,
        "bloom_level": bloom,
        "difficulty_num": difficulty_num,
        "difficulty_dots": dots,
        "keywords": keywords,
        "kcs": kcs,
        "lo": lo,
        "prompt_body": prompt_body,
        "stub": stub,
        "test_body": test_body,
        "solution_body": solution_body,
        "solution_notes": solution_notes,
        "extra_imports": merged_imports,
    }


# =========================================================================
# atom: linear-affine-on-custom-tensor
# =========================================================================

SPEC_LINEAR_AFFINE = _spec(
    atom_id="linear-affine-on-custom-tensor",
    subtopic="Backprop: Linear affine on custom Tensor",
    recap=RECAP_LINEAR_AFFINE,
    ex_idx=1,
    ex_title="forward pass of Linear over hand-written Tensor wrappers",
    slug="linear-affine-forward-over-custom-tensor",
    bloom="Apply",
    difficulty_num=3,
    keywords=["linear", "affine", "matmul", "bias-broadcast", "custom-tensor"],
    kcs=["linear-affine-on-custom-tensor", "box-array-to-tensor-with-recipe"],
    lo=(
        "Apply the `out = input @ weight + bias` affine map over the custom "
        "MiniTensor wrappers, producing a result MiniTensor whose Recipe "
        "names matmul as its forward fn and lists both Parameter inputs as "
        "parents."
    ),
    prompt_body=(
        "Implement `linear_forward(x, weight, bias)` — the forward pass of "
        "an ARENA-style `Linear` layer over MiniTensors. The drill is about "
        "wiring the affine map correctly on YOUR wrapper class; the autograd "
        "you'd want underneath is mocked by a simple Recipe attached to the "
        "matmul output.\n\n"
        "Inputs:\n"
        "- `x`:      `MiniTensor` of shape `(B, in_features)` — the batched input.\n"
        "- `weight`: `MiniTensor` of shape `(in_features, out_features)` — a "
        "Parameter.\n"
        "- `bias`:   `MiniTensor` of shape `(out_features,)` — a Parameter.\n\n"
        "Behavior:\n"
        "1. Compute `mm_arr = x.array @ weight.array`, shape `(B, out_features)`.\n"
        "2. Wrap as `mm = MiniTensor(mm_arr, requires_grad=(x.requires_grad or "
        "weight.requires_grad))` and attach `mm.recipe = Recipe(func=t.matmul, "
        "args=(x.array, weight.array), kwargs={}, parents={0: x, 1: weight})`.\n"
        "3. Compute `out_arr = mm.array + bias.array` (bias broadcasts over "
        "the batch axis).\n"
        "4. Wrap as `out = MiniTensor(out_arr, requires_grad=(mm.requires_grad "
        "or bias.requires_grad))` and attach `out.recipe = Recipe(func=t.add, "
        "args=(mm.array, bias.array), kwargs={}, parents={0: mm, 1: bias})`.\n"
        "5. Return `out`.\n\n"
        "Why the two-level Recipe. A real ARENA pipeline replaces the "
        "Recipe-construction lines with `mm = wrapped_matmul(x, weight)` and "
        "`out = wrapped_add(mm, bias)` — `wrap_forward_fn` builds the same "
        "Recipe. We construct manually here so the drill stays focused on the "
        "affine-map mechanics."
    ),
    stub=(
        "def linear_forward(x: MiniTensor, weight: MiniTensor, bias: MiniTensor) -> MiniTensor:\n"
        '    """Compute out = x @ weight + bias as a MiniTensor with chained Recipes."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- shape correctness ---\n"
        "B, in_f, out_f = 4, 3, 5\n"
        "x = MiniTensor(t.randn(B, in_f), requires_grad=False)\n"
        "weight = MiniTensor(t.randn(in_f, out_f), requires_grad=True)\n"
        "bias = MiniTensor(t.zeros(out_f), requires_grad=True)\n"
        "out = linear_forward(x, weight, bias)\n"
        "assert isinstance(out, MiniTensor), f'expected MiniTensor, got {type(out).__name__}'\n"
        "assert out.array.shape == (B, out_f), f'shape mismatch: {out.array.shape}'\n"
        "\n"
        "# --- numerical correctness ---\n"
        "expected = x.array @ weight.array + bias.array\n"
        "assert t.allclose(out.array, expected, atol=1e-5), 'value mismatch on affine map'\n"
        "\n"
        "# --- requires_grad OR-propagates from weight/bias ---\n"
        "assert out.requires_grad is True, (\n"
        "    'output requires_grad must be True when any Parameter input '\n"
        "    'requires grad (OR-propagation)'\n"
        ")\n"
        "\n"
        "# --- bias broadcasts (not stacked) ---\n"
        "bias_nonzero = MiniTensor(t.tensor([10.0, 20.0, 30.0, 40.0, 50.0]), requires_grad=True)\n"
        "x_zero = MiniTensor(t.zeros(B, in_f))\n"
        "out_zero = linear_forward(x_zero, weight, bias_nonzero)\n"
        "assert t.allclose(out_zero.array, bias_nonzero.array.unsqueeze(0).expand(B, out_f)), (\n"
        "    'zero input + nonzero bias must broadcast the bias across the batch axis'\n"
        ")\n"
        "\n"
        "# --- Recipe is chained: out.recipe.func is t.add, mm.recipe.func is t.matmul ---\n"
        "assert out.recipe is not None and out.recipe.func is t.add, (\n"
        "    f'out.recipe.func must be t.add (the final op), got {None if out.recipe is None else out.recipe.func}'\n"
        ")\n"
        "assert set(out.recipe.parents.keys()) == {0, 1}, f'out parents argidx wrong: {out.recipe.parents}'\n"
        "assert out.recipe.parents[1] is bias, 'arg-1 of the add must be the bias Parameter'\n"
        "mm = out.recipe.parents[0]\n"
        "assert isinstance(mm, MiniTensor) and mm.recipe is not None, 'arg-0 must be a MiniTensor with a Recipe'\n"
        "assert mm.recipe.func is t.matmul, f'mm.recipe.func must be t.matmul, got {mm.recipe.func}'\n"
        "assert mm.recipe.parents[0] is x and mm.recipe.parents[1] is weight, (\n"
        "    'mm Recipe must name x as arg-0 and weight as arg-1'\n"
        ")\n"
        "\n"
        "# --- larger smoke test: confirm shapes for a non-trivial batch ---\n"
        "x_big = MiniTensor(t.randn(32, 64))\n"
        "w_big = MiniTensor(t.randn(64, 10), requires_grad=True)\n"
        "b_big = MiniTensor(t.randn(10), requires_grad=True)\n"
        "y_big = linear_forward(x_big, w_big, b_big)\n"
        "assert y_big.array.shape == (32, 10), f'big shape: {y_big.array.shape}'\n"
        "assert t.allclose(y_big.array, x_big.array @ w_big.array + b_big.array, atol=1e-4)"
    ),
    solution_body=(
        "def linear_forward(x: MiniTensor, weight: MiniTensor, bias: MiniTensor) -> MiniTensor:\n"
        "    # step 1: matmul\n"
        "    mm_arr = x.array @ weight.array\n"
        "    mm = MiniTensor(\n"
        "        mm_arr,\n"
        "        requires_grad=(x.requires_grad or weight.requires_grad),\n"
        "    )\n"
        "    mm.recipe = Recipe(\n"
        "        func=t.matmul,\n"
        "        args=(x.array, weight.array),\n"
        "        kwargs={},\n"
        "        parents={0: x, 1: weight},\n"
        "    )\n"
        "    # step 2: bias add (broadcasts over batch axis)\n"
        "    out_arr = mm.array + bias.array\n"
        "    out = MiniTensor(\n"
        "        out_arr,\n"
        "        requires_grad=(mm.requires_grad or bias.requires_grad),\n"
        "    )\n"
        "    out.recipe = Recipe(\n"
        "        func=t.add,\n"
        "        args=(mm.array, bias.array),\n"
        "        kwargs={},\n"
        "        parents={0: mm, 1: bias},\n"
        "    )\n"
        "    return out"
    ),
    solution_notes=(
        "**Two ops, two Recipes.** The Recipe chain is `x -> mm -> out`. The "
        "intermediate `mm` is its own MiniTensor (so the reverse pass can "
        "find weight as `mm.recipe.parents[1]`). Collapsing into one Recipe "
        "would conflate matmul-back and add-back into a single dispatch — "
        "exactly what `wrap_forward_fn` avoids.\n\n"
        "**Bias broadcasts in the forward.** Shape `(B, out_f) + (out_f,)` "
        "uses the standard right-aligned broadcast rules. The corresponding "
        "`add_back1` for the bias path has to `unbroadcast` over the batch "
        "axis — sum out the leading `B` dim — to recover the `(out_f,)` "
        "grad shape. That's why the bias-grad code is `grad_out.sum(dim=0)`.\n\n"
        "**Why this is the capstone shape.** A working `linear_forward` plus "
        "matmul-back + add-back gives you a fully end-to-end-trainable Linear "
        "layer on your hand-built autograd. From here, stacking layers, ReLU, "
        "cross-entropy, and SGD assembles a complete MNIST training loop "
        "without touching `torch.autograd` once."
    ),
)


# =========================================================================
# atom: kaiming-uniform-sf-init
# =========================================================================

SPEC_KAIMING_SF = _spec(
    atom_id="kaiming-uniform-sf-init",
    subtopic="Init: Kaiming uniform SF init",
    recap=RECAP_KAIMING_SF,
    ex_idx=1,
    ex_title="initialize weight as Uniform(-sf, +sf) with sf = 1/sqrt(fan_in)",
    slug="kaiming-uniform-sf-1-over-sqrt-fan-in",
    bloom="Apply",
    difficulty_num=2,
    keywords=["kaiming", "uniform", "init", "fan-in", "scale-factor"],
    kcs=["kaiming-uniform-sf-init", "rand-uniform-shift-scale"],
    lo=(
        "Apply the ARENA Kaiming-uniform recipe `sf = 1/sqrt(fan_in)`, "
        "`weight ~ Uniform(-sf, +sf)`, producing a tensor of the requested "
        "shape with the correct empirical spread."
    ),
    prompt_body=(
        "Implement `kaiming_uniform_sf(in_features, out_features, generator)`. "
        "The Linear-layer weight initializer ARENA uses (and PyTorch's "
        "`nn.Linear` default):\n\n"
        "1. `fan_in = in_features` (number of input units feeding each output "
        "neuron).\n"
        "2. `sf = 1 / sqrt(fan_in)` (the scale factor).\n"
        "3. Sample `(in_features, out_features)` floats uniformly on "
        "`(-sf, +sf)` using the provided `torch.Generator`.\n"
        "4. Return as a `torch.Tensor` (not a MiniTensor — wrapping into "
        "`Parameter` is a separate atom).\n\n"
        "**Distinct from the `sqrt(6/fan_in)` form.** Don't conflate. ARENA "
        "uses `1/sqrt(fan_in)`; the deeper-theory form is `sqrt(6/fan_in)` "
        "(activation-preserving for ReLU under stronger assumptions). Both "
        "are real, but the drill is specifically the ARENA / PyTorch default.\n\n"
        "Hint: `t.rand(shape, generator=g)` is uniform on `[0, 1)`. To get "
        "`(-sf, +sf)`, do `(t.rand(shape, generator=g) * 2 - 1) * sf`.\n\n"
        "Output: `torch.Tensor` of shape `(in_features, out_features)`."
    ),
    stub=(
        "def kaiming_uniform_sf(\n"
        "    in_features: int, out_features: int, generator: t.Generator\n"
        ") -> Tensor:\n"
        '    """Sample weight ~ Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import math\n"
        "# --- shape ---\n"
        "g = t.Generator().manual_seed(0)\n"
        "w = kaiming_uniform_sf(3, 5, g)\n"
        "assert isinstance(w, t.Tensor), f'expected torch.Tensor, got {type(w).__name__}'\n"
        "assert w.shape == (3, 5), f'shape: {w.shape}'\n"
        "\n"
        "# --- bounds: |w| <= sf for every element ---\n"
        "sf = 1.0 / math.sqrt(3)\n"
        "assert w.abs().max().item() <= sf + 1e-6, (\n"
        "    f'all entries must lie in (-sf, +sf); max |w| = {w.abs().max().item()} '\n"
        "    f'vs sf = {sf:.4f}'\n"
        ")\n"
        "\n"
        "# --- large-sample empirical check: std ~= sf / sqrt(3) ---\n"
        "g2 = t.Generator().manual_seed(1)\n"
        "fan_in = 100\n"
        "w_big = kaiming_uniform_sf(fan_in, 50_000, g2)  # 5M samples\n"
        "sf_big = 1.0 / math.sqrt(fan_in)\n"
        "assert w_big.abs().max().item() <= sf_big + 1e-6\n"
        "expected_std = sf_big / math.sqrt(3.0)\n"
        "empirical_std = w_big.std().item()\n"
        "rel_err = abs(empirical_std - expected_std) / expected_std\n"
        "assert rel_err < 0.02, (\n"
        "    f'empirical std {empirical_std:.4f} too far from expected '\n"
        "    f'{expected_std:.4f} (rel err {rel_err:.4f}); init may use wrong sf'\n"
        ")\n"
        "\n"
        "# --- mean ~= 0 (uniform on symmetric interval is zero-mean) ---\n"
        "assert abs(w_big.mean().item()) < 1e-3, (\n"
        "    f'sample mean must be near 0, got {w_big.mean().item():.4f}'\n"
        ")\n"
        "\n"
        "# --- generator is honored: same seed → same tensor ---\n"
        "g_a = t.Generator().manual_seed(42)\n"
        "g_b = t.Generator().manual_seed(42)\n"
        "w_a = kaiming_uniform_sf(4, 7, g_a)\n"
        "w_b = kaiming_uniform_sf(4, 7, g_b)\n"
        "assert t.allclose(w_a, w_b), 'same seed must produce the same weight tensor'\n"
        "\n"
        "# --- DIFFERENT seed → different tensor (probabilistic sanity check) ---\n"
        "g_c = t.Generator().manual_seed(99)\n"
        "w_c = kaiming_uniform_sf(4, 7, g_c)\n"
        "assert not t.allclose(w_a, w_c), 'different seed should produce different tensor'\n"
        "\n"
        "# --- scale shrinks with fan_in: sf for fan_in=400 is half of sf for fan_in=100 ---\n"
        "g_big1 = t.Generator().manual_seed(7)\n"
        "g_big2 = t.Generator().manual_seed(7)\n"
        "w_100 = kaiming_uniform_sf(100, 1000, g_big1)\n"
        "w_400 = kaiming_uniform_sf(400, 1000, g_big2)\n"
        "ratio = w_400.abs().max().item() / w_100.abs().max().item()\n"
        "# sf_400 / sf_100 = sqrt(100/400) = 0.5\n"
        "assert 0.35 < ratio < 0.65, (\n"
        "    f'sf must scale as 1/sqrt(fan_in); got max-abs ratio {ratio:.3f}, expected ~0.5'\n"
        ")"
    ),
    solution_body=(
        "def kaiming_uniform_sf(\n"
        "    in_features: int, out_features: int, generator: t.Generator\n"
        ") -> Tensor:\n"
        "    # sf = 1 / sqrt(fan_in)  — ARENA / PyTorch nn.Linear default\n"
        "    sf = in_features ** -0.5\n"
        "    # Uniform(-sf, +sf) = (Uniform(0, 1) * 2 - 1) * sf\n"
        "    raw = t.rand(in_features, out_features, generator=generator)\n"
        "    return (raw * 2 - 1) * sf"
    ),
    solution_notes=(
        "**The two Kaiming forms are NOT the same.** ARENA / PyTorch use "
        "`U(-1/sqrt(fan_in), +1/sqrt(fan_in))`. The 'activation-preserving' "
        "form is `U(-sqrt(6/fan_in), +sqrt(6/fan_in))` — chosen so "
        "`Var(w) = 2/fan_in`. With the ARENA form, "
        "`Var(w) = sf^2 / 3 = 1/(3 * fan_in)` — smaller by a factor of 6. "
        "Both initialize networks that train; they're calibrated for "
        "different objective functions.\n\n"
        "**Why the empirical std ≈ sf / sqrt(3).** The population std of "
        "`Uniform(-a, +a)` is `a / sqrt(3)` (variance = `a^2 / 3`). With "
        "`a = sf`, that's `sf / sqrt(3)`. The test asserts this to within 2% "
        "relative error using 5M samples — sufficient to catch the wrong-form "
        "init (which would land at `sf * sqrt(2)`).\n\n"
        "**Generator threading.** Passing a `torch.Generator` instead of "
        "calling `t.rand()` globally lets the caller seed reproducibly — "
        "essential for testable inits, deterministic experiments, and "
        "regression tests that compare two networks' weights."
    ),
)


# =========================================================================
# atom: parameter-wrap-around-tensor (the anti-pattern; drill that it fails)
# =========================================================================

SPEC_PARAMETER_WRAP = _spec(
    atom_id="parameter-wrap-around-tensor",
    subtopic="Backprop: Parameter wrap around Tensor",
    recap=RECAP_PARAMETER_WRAP,
    ex_idx=1,
    ex_title="observe the composition-Parameter anti-pattern breaks isinstance",
    slug="parameter-composition-anti-pattern-observe-breakage",
    bloom="Analyze",
    difficulty_num=3,
    keywords=["parameter", "composition", "anti-pattern", "isinstance", "is-a-vs-has-a"],
    kcs=["parameter-wrap-around-tensor", "parameter-subclass-of-tensor"],
    lo=(
        "Analyze the composition-Parameter (HAS-A) design vs the "
        "subclass-Parameter (IS-A) design by building both, then asserting "
        "that only the IS-A form survives the `isinstance(_, MiniTensor)` "
        "filter that every autograd helper depends on."
    ),
    prompt_body=(
        "Implement TWO Parameter designs and a comparison helper, so the "
        "test cell can demonstrate the silent-failure mode of the wrong one.\n\n"
        "**1. `WrapParam(tensor)`** — the COMPOSITION design (anti-pattern).\n"
        "   ```python\n"
        "   class WrapParam:\n"
        "       def __init__(self, tensor):\n"
        "           self.tensor = tensor   # stored as an attribute\n"
        "           self.requires_grad = True\n"
        "   ```\n"
        "   Does NOT inherit from `MiniTensor`.\n\n"
        "**2. `IsAParam(MiniTensor)`** — the SUBCLASS design (correct).\n"
        "   ```python\n"
        "   class IsAParam(MiniTensor):\n"
        "       def __init__(self, array):\n"
        "           super().__init__(array, requires_grad=True)\n"
        "   ```\n\n"
        "**3. `collect_params_via_isinstance(things)`** — the helper every "
        "autograd layer uses. Returns the subset of `things` that pass "
        "`isinstance(_, MiniTensor)`:\n"
        "   ```python\n"
        "   def collect_params_via_isinstance(things):\n"
        "       return [x for x in things if isinstance(x, MiniTensor)]\n"
        "   ```\n\n"
        "The point of the drill: when `collect_params_via_isinstance` is given "
        "a mixed bag containing both a `WrapParam` and an `IsAParam`, the "
        "WrapParam is **silently dropped**. The test cell asserts this — "
        "you're observing the bug, not fixing it.\n\n"
        "Why this matters: every helper in your autograd layer "
        "(`build_parents`, `unbox_args`, `parameters()` walker) uses exactly "
        "this isinstance gate. Composition-Parameters become invisible "
        "trainable state. Use IS-A."
    ),
    stub=(
        "class WrapParam:\n"
        '    """Composition (HAS-A) Parameter — stores the tensor as an attribute. WRONG design."""\n'
        "    def __init__(self, tensor):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "\n"
        "class IsAParam(MiniTensor):\n"
        '    """Subclass (IS-A) Parameter — inherits from MiniTensor. CORRECT design."""\n'
        "    def __init__(self, array):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "\n"
        "def collect_params_via_isinstance(things: list) -> list:\n"
        '    """Filter to MiniTensor instances — the canonical autograd-layer helper."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- WrapParam stores its tensor as `.tensor`, NOT as `.array` ---\n"
        "raw = t.tensor([1.0, 2.0, 3.0])\n"
        "wp = WrapParam(raw)\n"
        "assert wp.tensor is raw, 'WrapParam should expose the raw tensor as `.tensor`'\n"
        "assert wp.requires_grad is True, 'WrapParam default should be requires_grad=True'\n"
        "\n"
        "# --- WrapParam is NOT a MiniTensor — that's the whole problem ---\n"
        "assert not isinstance(wp, MiniTensor), (\n"
        "    'WrapParam (composition) must NOT be a MiniTensor subclass — '\n"
        "    'that IS the anti-pattern under study'\n"
        ")\n"
        "\n"
        "# --- IsAParam IS a MiniTensor (subclass) ---\n"
        "ip = IsAParam(t.tensor([4.0, 5.0, 6.0]))\n"
        "assert isinstance(ip, MiniTensor), 'IsAParam (subclass) MUST be a MiniTensor'\n"
        "assert isinstance(ip, IsAParam)\n"
        "assert ip.requires_grad is True, 'IsAParam default should be requires_grad=True'\n"
        "assert ip.array is not None and t.equal(ip.array, t.tensor([4.0, 5.0, 6.0]))\n"
        "\n"
        "# --- the helper picks up MiniTensors only ---\n"
        "x = MiniTensor(t.zeros(3))\n"
        "got = collect_params_via_isinstance([wp, ip, x, 3.14, 'noise'])\n"
        "assert ip in got, 'IsAParam must survive the isinstance filter'\n"
        "assert x in got, 'plain MiniTensor must survive the isinstance filter'\n"
        "assert wp not in got, (\n"
        "    'WrapParam must be silently DROPPED — composition Parameters are '\n"
        "    'invisible to the autograd layer. This is the bug.'\n"
        ")\n"
        "\n"
        "# --- the load-bearing observation: only one of the two designs survives ---\n"
        "wp_count = sum(1 for x in got if isinstance(x, WrapParam))\n"
        "ip_count = sum(1 for x in got if isinstance(x, IsAParam))\n"
        "assert wp_count == 0, f'WrapParams collected (should be zero): {wp_count}'\n"
        "assert ip_count >= 1, f'IsAParams collected (should be >=1): {ip_count}'\n"
        "\n"
        "# --- the silent-bug demo: build a fake `parameters()` walker over a list ---\n"
        "all_things = [WrapParam(t.tensor([10.0])), WrapParam(t.tensor([20.0])),\n"
        "              IsAParam(t.tensor([30.0]))]\n"
        "trainable = collect_params_via_isinstance(all_things)\n"
        "assert len(trainable) == 1, (\n"
        "    f'expected 1 trainable param (only the IsAParam), got {len(trainable)} — '\n"
        "    f'two WrapParams were silently dropped'\n"
        ")\n"
        "# A real training loop iterating these would update ONE weight, not three,\n"
        "# and the user would never get an error message — the bug is invisible.\n"
        "\n"
        "# --- the fix: convert WrapParams to IsAParams. Sanity check the converted set. ---\n"
        "converted = [IsAParam(wp.tensor) if isinstance(wp, WrapParam) else wp for wp in all_things]\n"
        "trainable_after = collect_params_via_isinstance(converted)\n"
        "assert len(trainable_after) == 3, (\n"
        "    'after IS-A conversion all 3 must survive the filter — proves the '\n"
        "    'silent loss in the previous step was caused by HAS-A, nothing else'\n"
        ")"
    ),
    solution_body=(
        "class WrapParam:\n"
        '    """Composition (HAS-A) Parameter — the WRONG design."""\n'
        "    def __init__(self, tensor):\n"
        "        self.tensor = tensor\n"
        "        self.requires_grad = True\n"
        "\n"
        "\n"
        "class IsAParam(MiniTensor):\n"
        '    """Subclass (IS-A) Parameter — the RIGHT design."""\n'
        "    def __init__(self, array):\n"
        "        super().__init__(array, requires_grad=True)\n"
        "\n"
        "\n"
        "def collect_params_via_isinstance(things: list) -> list:\n"
        "    return [x for x in things if isinstance(x, MiniTensor)]"
    ),
    solution_notes=(
        "**The composition-vs-inheritance choice is load-bearing.** It's "
        "tempting to start with `WrapParam(tensor)` because composition 'feels "
        "safer' — no inheritance, no MRO surprises. But the entire autograd "
        "scaffolding consists of helpers that say `if isinstance(x, "
        "MiniTensor): ...`. A composition-Parameter fails that gate; it "
        "becomes invisible to `build_parents`, `unbox_args`, `get_children`, "
        "and `parameters()`.\n\n"
        "**Why this fails silently.** No exception is raised. No warning. The "
        "training loop runs, the loss decreases (because *some* parameters "
        "are still updating), and a fraction of the model's weights stay "
        "frozen at their init values. Debugging this means inspecting "
        "`list(model.parameters())` and noticing it's mysteriously short.\n\n"
        "**PyTorch's actual design.** `torch.nn.Parameter(torch.Tensor)` — "
        "subclassing, exactly the `IsAParam` form. The minimal class body in "
        "PyTorch consists of a `__new__` override (to handle the "
        "`requires_grad=True` default) and a `__deepcopy__` for state-dict "
        "copying. Nothing else. The IS-A relationship IS the design."
    ),
)


# =========================================================================
# atom: module-base-class-custom
# =========================================================================

SPEC_MODULE_BASE = _spec(
    atom_id="module-base-class-custom",
    subtopic="Backprop: Module base class custom",
    recap=RECAP_MODULE_BASE,
    ex_idx=1,
    ex_title="build Module: __setattr__ registers params, parameters() walks recursively",
    slug="module-base-setattr-register-and-parameters-walker",
    bloom="Apply",
    difficulty_num=3,
    keywords=["module", "setattr", "parameters", "recursive-walker", "submodule"],
    kcs=["module-base-class-custom", "parameter-subclass-of-tensor"],
    lo=(
        "Apply the `__setattr__`-as-registrar pattern to build a minimal "
        "`Module` base class whose `parameters()` walks all directly-assigned "
        "Parameters plus those of any submodule, transitively."
    ),
    prompt_body=(
        "Implement `Module` and a sample `Parameter` subclass so the test "
        "cell can build a tiny 2-layer model and verify all parameters are "
        "discoverable.\n\n"
        "**1. `Parameter(MiniTensor)`** — already covered in batch-3 / batch-4. "
        "Subclass `MiniTensor`, default `requires_grad=True`. Just re-define "
        "it here so the drill is self-contained.\n\n"
        "**2. `Module` base class.** Required surface:\n"
        "   - `__init__(self)` — initialize `self._parameters = {}` and "
        "`self._modules = {}` BEFORE any other attribute is set. Use "
        "`object.__setattr__` for this bootstrap so the custom `__setattr__` "
        "doesn't recurse on the registry dicts themselves.\n"
        "   - `__setattr__(self, name, value)` — when `value` is a "
        "`Parameter`, register it in `self._parameters[name]`. When `value` "
        "is a `Module`, register in `self._modules[name]`. Then ALWAYS call "
        "`object.__setattr__(self, name, value)` so the attribute is also "
        "accessible via normal `.name` lookup.\n"
        "   - `parameters(self)` — generator that yields:\n"
        "     1. Every direct `Parameter` (`self._parameters.values()`).\n"
        "     2. Every parameter of every submodule "
        "(`m.parameters()` for `m` in `self._modules.values()`).\n"
        "   - `forward(self, *args, **kwargs)` — abstract; raise "
        "`NotImplementedError()`.\n\n"
        "**Why `__setattr__` not `register_parameter`.** The intercept lets "
        "users write `self.weight = Parameter(...)` and have it Just Work. "
        "Without it, every layer would need explicit `register_parameter` "
        "calls.\n\n"
        "**Bootstrap order matters.** `object.__setattr__(self, "
        "'_parameters', {})` in `__init__` is crucial. If you wrote "
        "`self._parameters = {}`, your custom `__setattr__` would fire and "
        "try to read `self._parameters` to register the value — but "
        "`self._parameters` doesn't exist yet. Stack overflow."
    ),
    stub=(
        "class Parameter(MiniTensor):\n"
        '    """Trainable leaf — MiniTensor subclass, requires_grad=True default."""\n'
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "\n"
        "\n"
        "class Module:\n"
        '    """Minimal nn.Module clone — registers Parameters / Modules via __setattr__."""\n'
        "    def __init__(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def __setattr__(self, name, value):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def parameters(self):\n"
        '        """Yield every Parameter in self, walking submodules recursively."""\n'
        "        raise NotImplementedError()\n"
        "\n"
        "    def forward(self, *args, **kwargs):\n"
        "        raise NotImplementedError()"
    ),
    test_body=(
        "# --- single-layer module: direct Parameter is registered ---\n"
        "class TinyLayer(Module):\n"
        "    def __init__(self, in_f, out_f):\n"
        "        super().__init__()\n"
        "        self.weight = Parameter(t.zeros(in_f, out_f))\n"
        "        self.bias = Parameter(t.zeros(out_f))\n"
        "        self.in_f = in_f  # plain attribute — must NOT register\n"
        "\n"
        "layer = TinyLayer(3, 5)\n"
        "params = list(layer.parameters())\n"
        "assert len(params) == 2, f'TinyLayer must expose 2 Parameters, got {len(params)}'\n"
        "assert layer.weight in params and layer.bias in params, 'weight and bias both expected'\n"
        "# Plain attributes (ints, etc.) must not show up as parameters.\n"
        "assert 3 not in params, 'plain int attribute must not be registered as a Parameter'\n"
        "\n"
        "# --- attribute access still works (object.__setattr__ also stores the attr) ---\n"
        "assert isinstance(layer.weight, Parameter), 'self.weight must be retrievable normally'\n"
        "assert layer.in_f == 3, 'plain attributes must be retrievable normally'\n"
        "\n"
        "# --- nested module: parameters() walks submodules recursively ---\n"
        "class Net(Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.layer1 = TinyLayer(4, 8)\n"
        "        self.layer2 = TinyLayer(8, 2)\n"
        "        self.standalone = Parameter(t.zeros(7))\n"
        "\n"
        "net = Net()\n"
        "all_params = list(net.parameters())\n"
        "# 2 from layer1 (w, b) + 2 from layer2 (w, b) + 1 standalone = 5\n"
        "assert len(all_params) == 5, (\n"
        "    f'Net must expose 5 Parameters (2 + 2 + 1), got {len(all_params)} — '\n"
        "    'recursive walker must include submodule parameters'\n"
        ")\n"
        "assert net.standalone in all_params\n"
        "assert net.layer1.weight in all_params\n"
        "assert net.layer2.bias in all_params\n"
        "\n"
        "# --- self._modules tracks the submodules dict ---\n"
        "assert 'layer1' in net._modules and 'layer2' in net._modules, (\n"
        "    f'submodules must be registered by name: {list(net._modules.keys())}'\n"
        ")\n"
        "assert net._modules['layer1'] is net.layer1\n"
        "\n"
        "# --- forward is abstract on the base class ---\n"
        "try:\n"
        "    Module().forward()\n"
        "    raise AssertionError('Module.forward() must raise NotImplementedError on the base class')\n"
        "except NotImplementedError:\n"
        "    pass\n"
        "\n"
        "# --- reassigning a Parameter to the same name replaces the entry ---\n"
        "old_w = layer.weight\n"
        "new_w = Parameter(t.ones(3, 5))\n"
        "layer.weight = new_w\n"
        "params_after = list(layer.parameters())\n"
        "assert new_w in params_after, 'reassigned Parameter must appear in parameters()'\n"
        "assert old_w not in params_after, 'old Parameter must be evicted by reassignment'\n"
        "\n"
        "# --- non-Parameter, non-Module assignment must not pollute the registries ---\n"
        "layer.note = 'hello'\n"
        "assert 'note' not in layer._parameters\n"
        "assert 'note' not in layer._modules\n"
        "assert layer.note == 'hello'"
    ),
    solution_body=(
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "\n"
        "\n"
        "class Module:\n"
        "    def __init__(self):\n"
        "        # bootstrap: bypass our own __setattr__ to install the registries\n"
        "        object.__setattr__(self, '_parameters', {})\n"
        "        object.__setattr__(self, '_modules', {})\n"
        "\n"
        "    def __setattr__(self, name, value):\n"
        "        if isinstance(value, Parameter):\n"
        "            self._parameters[name] = value\n"
        "            # remove any prior submodule slot with the same name\n"
        "            self._modules.pop(name, None)\n"
        "        elif isinstance(value, Module):\n"
        "            self._modules[name] = value\n"
        "            self._parameters.pop(name, None)\n"
        "        else:\n"
        "            self._parameters.pop(name, None)\n"
        "            self._modules.pop(name, None)\n"
        "        # always also expose the attr through normal lookup\n"
        "        object.__setattr__(self, name, value)\n"
        "\n"
        "    def parameters(self):\n"
        "        for p in self._parameters.values():\n"
        "            yield p\n"
        "        for m in self._modules.values():\n"
        "            yield from m.parameters()\n"
        "\n"
        "    def forward(self, *args, **kwargs):\n"
        "        raise NotImplementedError()"
    ),
    solution_notes=(
        "**Why `object.__setattr__` in `__init__`.** During `__init__`, the "
        "registries don't exist yet. If you write `self._parameters = {}` the "
        "custom `__setattr__` fires and tries to inspect `self._parameters` — "
        "infinite recursion (or AttributeError, depending on order). "
        "`object.__setattr__(self, '_parameters', {})` bypasses the override "
        "and installs the dict directly.\n\n"
        "**Why both registries are updated on every assignment.** If a user "
        "writes `self.layer = SomeSubmodule()` and then later "
        "`self.layer = Parameter(...)`, the previous `_modules['layer']` "
        "must be evicted — otherwise `parameters()` would yield the old "
        "submodule's params even though `self.layer` no longer points to it. "
        "The `.pop(name, None)` calls handle this defensively.\n\n"
        "**Recursive walk via `yield from`.** A non-recursive `parameters()` "
        "(only yielding `_parameters.values()`) would miss everything in "
        "submodules — a 2-layer MLP's `Linear` weights would be invisible. "
        "`yield from m.parameters()` makes the walker depth-first and "
        "transitively complete."
    ),
)


# =========================================================================
# atom: logsumexp-cross-entropy
# =========================================================================

SPEC_LOGSUMEXP_CE = _spec(
    atom_id="logsumexp-cross-entropy",
    subtopic="Loss: logsumexp cross-entropy",
    recap=RECAP_LOGSUMEXP_CE,
    ex_idx=1,
    ex_title="numerically stable cross-entropy via logsumexp",
    slug="cross-entropy-via-logsumexp-stable",
    bloom="Apply",
    difficulty_num=3,
    keywords=["cross-entropy", "logsumexp", "numerical-stability", "softmax", "overflow"],
    kcs=["logsumexp-cross-entropy", "arange-fancy-index-cross-entropy"],
    lo=(
        "Apply the logsumexp identity to compute cross-entropy as "
        "`logsumexp(logits) - logits[arange(B), target]`, avoiding the "
        "softmax overflow that the naive formulation triggers for large "
        "logits."
    ),
    prompt_body=(
        "Implement `cross_entropy_logsumexp(logits, target)`. Per-sample, "
        "the loss is\n\n"
        "```\n"
        "loss[i] = logsumexp(logits[i]) - logits[i, target[i]]\n"
        "```\n\n"
        "Return the mean loss across the batch — a 0-D `torch.Tensor`.\n\n"
        "Inputs:\n"
        "- `logits`: shape `(B, C)`, float (any magnitude — the function "
        "MUST handle logits up to ~10000 without overflow).\n"
        "- `target`: shape `(B,)`, integer class indices in `[0, C)`.\n\n"
        "Output: scalar tensor (`shape == ()`), mean cross-entropy.\n\n"
        "**Use `torch.logsumexp(logits, dim=-1)`** for the first term. It "
        "internally subtracts the per-row max before exp, so the formula is "
        "stable even when logits are huge.\n\n"
        "**For the second term**, you need `logits[arange(B), target]` — the "
        "advanced-indexing pattern for picking out one entry per row. (A "
        "separate drill covers that pattern in isolation; here we use it.)\n\n"
        "**Compare with the naive version.** The naive formula "
        "`-log(softmax(logits)[target])` is mathematically identical but "
        "overflows for `logits[i, k] > ~88` in float32 — `exp(89) > 3e38 > "
        "float32_max`. The test cell stresses this case explicitly.\n\n"
        "Do NOT call `torch.nn.functional.cross_entropy` — write the "
        "formula directly."
    ),
    stub=(
        "def cross_entropy_logsumexp(logits: Tensor, target: Tensor) -> Tensor:\n"
        '    """Compute mean cross-entropy via logsumexp(logits) - logits[arange(B), target]."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import math\n"
        "# --- baseline: uniform logits → log(C) for any target ---\n"
        "logits = t.zeros(4, 3)  # uniform → softmax = 1/3 → -log(1/3) = log(3)\n"
        "target = t.tensor([0, 1, 2, 0])\n"
        "loss = cross_entropy_logsumexp(logits, target)\n"
        "assert loss.shape == (), f'expected scalar, got shape {loss.shape}'\n"
        "assert abs(loss.item() - math.log(3)) < 1e-5, (\n"
        "    f'uniform logits → log(C)={math.log(3):.4f}, got {loss.item():.4f}'\n"
        ")\n"
        "\n"
        "# --- compare against torch.nn.functional.cross_entropy (the witness) ---\n"
        "import torch.nn.functional as F\n"
        "logits2 = t.tensor([[2.0, 1.0, 0.1], [0.5, -1.0, 3.0]])\n"
        "target2 = t.tensor([0, 2])\n"
        "loss2 = cross_entropy_logsumexp(logits2, target2)\n"
        "ref2 = F.cross_entropy(logits2, target2)\n"
        "assert abs(loss2.item() - ref2.item()) < 1e-5, (\n"
        "    f'mismatch vs F.cross_entropy: ours={loss2.item():.6f}, ref={ref2.item():.6f}'\n"
        ")\n"
        "\n"
        "# --- THE STABILITY TEST: logits = 1000 must NOT overflow ---\n"
        "big_logits = t.tensor([[1000.0, 999.0, 998.0], [0.0, 500.0, 0.0]])\n"
        "big_target = t.tensor([0, 1])\n"
        "big_loss = cross_entropy_logsumexp(big_logits, big_target)\n"
        "assert t.isfinite(big_loss).item(), (\n"
        "    f'huge logits must not produce inf/nan; got {big_loss.item()} — '\n"
        "    'are you using logsumexp, or did you call exp(logits) directly?'\n"
        ")\n"
        "# The expected loss for row 0 (target 0): logsumexp([1000,999,998]) - 1000\n"
        "# = log(exp(0) + exp(-1) + exp(-2)) ~ 0.4076.\n"
        "# Row 1 (target 1): logsumexp([0,500,0]) - 500 ~ log(2*exp(-500) + 1) ~ 0 -> 0.\n"
        "expected_row0 = math.log(1 + math.exp(-1) + math.exp(-2))\n"
        "expected = (expected_row0 + 0.0) / 2\n"
        "assert abs(big_loss.item() - expected) < 1e-4, (\n"
        "    f'huge-logit loss wrong: got {big_loss.item()}, expected {expected:.4f}'\n"
        ")\n"
        "\n"
        "# --- confirm the naive version WOULD overflow at this scale (sanity for the test) ---\n"
        "naive_overflowed = not t.isfinite(t.exp(big_logits)).all().item()\n"
        "assert naive_overflowed, (\n"
        "    'sanity: t.exp(big_logits) should overflow at scale 1000 — '\n"
        "    'if this assertion fails, the test stress case is too weak'\n"
        ")\n"
        "\n"
        "# --- batch-mean semantics (not sum) ---\n"
        "lo = t.zeros(10, 5)\n"
        "ta = t.zeros(10, dtype=t.long)\n"
        "loss_mean = cross_entropy_logsumexp(lo, ta)\n"
        "assert abs(loss_mean.item() - math.log(5)) < 1e-5, (\n"
        "    f'must return MEAN (not sum); 10 samples × log(5) summed would be {10*math.log(5):.4f}, '\n"
        "    f'got {loss_mean.item():.4f}'\n"
        ")\n"
        "\n"
        "# --- target on the correct class with very high logit → near-zero loss ---\n"
        "confident = t.tensor([[100.0, 0.0, 0.0]])  # logits favor class 0 strongly\n"
        "loss_confident = cross_entropy_logsumexp(confident, t.tensor([0]))\n"
        "assert loss_confident.item() < 1e-5, (\n"
        "    f'confident-correct logits should yield ~0 loss, got {loss_confident.item()}'\n"
        ")"
    ),
    solution_body=(
        "def cross_entropy_logsumexp(logits: Tensor, target: Tensor) -> Tensor:\n"
        "    # logsumexp over the class axis — numerically stable\n"
        "    lse = t.logsumexp(logits, dim=-1)                # shape (B,)\n"
        "    # pick out per-sample target logit via arange-fancy-index\n"
        "    B = logits.shape[0]\n"
        "    picked = logits[t.arange(B), target]              # shape (B,)\n"
        "    # per-sample loss, then batch mean\n"
        "    per_sample = lse - picked\n"
        "    return per_sample.mean()"
    ),
    solution_notes=(
        "**Why logsumexp survives where naive softmax dies.** "
        "`torch.logsumexp(x)` computes `log(sum(exp(x - max(x)))) + max(x)`. "
        "The max-shift keeps every exp argument ≤ 0, so the largest term is "
        "`exp(0) = 1`. The naive `softmax(x) = exp(x) / sum(exp(x))` "
        "doesn't shift — and `exp(1000)` is `inf` in any float type.\n\n"
        "**The cross-entropy identity.** For per-sample loss:\n"
        "```\n"
        "loss = -log(softmax(x)[target])\n"
        "     = -log(exp(x[target]) / sum_k exp(x[k]))\n"
        "     = -x[target] + log(sum_k exp(x[k]))\n"
        "     = logsumexp(x) - x[target]\n"
        "```\n"
        "The substitution is exact; the only difference is numerical.\n\n"
        "**Mean vs sum.** `F.cross_entropy` defaults to `reduction='mean'`, "
        "which is what most training loops want (loss scale doesn't depend "
        "on batch size). `sum` is occasionally used when gradient "
        "accumulation across mini-batches needs to be unbiased."
    ),
)


# =========================================================================
# atom: arange-fancy-index-cross-entropy
# =========================================================================

SPEC_ARANGE_FANCY = _spec(
    atom_id="arange-fancy-index-cross-entropy",
    subtopic="Loss: arange fancy-index cross-entropy",
    recap=RECAP_ARANGE_FANCY,
    ex_idx=1,
    ex_title="pick per-sample target logits via logits[arange(B), target]",
    slug="arange-fancy-index-per-sample-target-logits",
    bloom="Apply",
    difficulty_num=2,
    keywords=["arange", "fancy-indexing", "advanced-indexing", "per-sample", "target-logit"],
    kcs=["arange-fancy-index-cross-entropy", "logsumexp-cross-entropy"],
    lo=(
        "Apply NumPy/PyTorch advanced indexing with `arange(B)` paired with "
        "a `(B,)` index tensor to extract one column per row, producing a "
        "`(B,)` vector of per-sample target logits."
    ),
    prompt_body=(
        "Implement `pick_target_logits(logits, target)`. Returns a `(B,)` "
        "tensor where position `i` is `logits[i, target[i]]`.\n\n"
        "Inputs:\n"
        "- `logits`: shape `(B, C)`, float.\n"
        "- `target`: shape `(B,)`, integer class indices in `[0, C)`.\n\n"
        "Output: shape `(B,)`, same dtype as `logits`.\n\n"
        "**Vectorized one-liner:**\n"
        "```python\n"
        "logits[t.arange(B), target]\n"
        "```\n\n"
        "Mechanics: when you index a 2-D tensor with TWO 1-D tensors of the "
        "same length, PyTorch pairs them positionally. `arange(B) = [0, 1, "
        "..., B-1]` and `target = [t0, t1, ...]` zip into `[(0, t0), "
        "(1, t1), ...]`. One element per row.\n\n"
        "**Forbidden:** Python for-loops. The drill is specifically the "
        "vectorized pattern. A loop over `range(B)` would be `O(B)` Python "
        "overhead and would break grad accumulation (it materializes one "
        "scalar at a time, with separate Recipes).\n\n"
        "**Why `arange`, not `:`.** `logits[:, target]` broadcasts to "
        "`(B, B)` — every row indexed by every target — not what we want. "
        "The `arange` advances the row axis in lockstep with the column axis."
    ),
    stub=(
        "def pick_target_logits(logits: Tensor, target: Tensor) -> Tensor:\n"
        '    """Return logits[arange(B), target] — per-sample target logits, shape (B,)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- the trivial case: B=3, target picks one column per row ---\n"
        "logits = t.tensor([\n"
        "    [10.0, 20.0, 30.0],\n"
        "    [40.0, 50.0, 60.0],\n"
        "    [70.0, 80.0, 90.0],\n"
        "])\n"
        "target = t.tensor([0, 1, 2])\n"
        "got = pick_target_logits(logits, target)\n"
        "assert got.shape == (3,), f'shape: {got.shape}'\n"
        "assert t.allclose(got, t.tensor([10.0, 50.0, 90.0])), f'value: {got}'\n"
        "\n"
        "# --- target on the same column for every row → constant slice ---\n"
        "target_const = t.tensor([1, 1, 1])\n"
        "got_const = pick_target_logits(logits, target_const)\n"
        "assert t.allclose(got_const, t.tensor([20.0, 50.0, 80.0])), (\n"
        "    f'target=1 for all rows must pick column 1: {got_const}'\n"
        ")\n"
        "\n"
        "# --- DO NOT broadcast — output is (B,), NOT (B, B) ---\n"
        "assert got_const.shape == (3,), (\n"
        "    f'output must be (3,) not (3,3); did you write logits[:, target]? '\n"
        "    f'Got {got_const.shape}'\n"
        ")\n"
        "# Explicit comparison against the wrong (broadcasting) result.\n"
        "broadcast_wrong = logits[:, target_const]\n"
        "assert broadcast_wrong.shape == (3, 3), 'sanity: logits[:, target] does broadcast'\n"
        "assert got_const.shape != broadcast_wrong.shape, (\n"
        "    'your result must NOT match the broadcasting version'\n"
        ")\n"
        "\n"
        "# --- larger batch: random logits, target picks one per row ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "big_logits = t.randn(32, 10, generator=rng)\n"
        "big_target = t.randint(0, 10, (32,), generator=rng)\n"
        "got_big = pick_target_logits(big_logits, big_target)\n"
        "assert got_big.shape == (32,)\n"
        "# Witness via a Python loop (slow but obviously correct).\n"
        "expected_big = t.tensor([big_logits[i, big_target[i].item()].item() for i in range(32)])\n"
        "assert t.allclose(got_big, expected_big), 'value mismatch on (32, 10) batch'\n"
        "\n"
        "# --- dtype preserved ---\n"
        "logits_d = t.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=t.float64)\n"
        "target_d = t.tensor([0, 1])\n"
        "out_d = pick_target_logits(logits_d, target_d)\n"
        "assert out_d.dtype == t.float64, f'dtype must be preserved, got {out_d.dtype}'\n"
        "\n"
        "# --- composes with cross-entropy: lse - picked is per-sample CE loss ---\n"
        "lse = t.logsumexp(logits, dim=-1)\n"
        "picked = pick_target_logits(logits, target)\n"
        "per_sample_ce = lse - picked\n"
        "assert per_sample_ce.shape == (3,), 'pick_target_logits must compose into CE without reshape'\n"
        "# Sanity: row 0 target 0, logits [10,20,30] → lse ~= 30+log(1+e^-10+e^-20) ~ 30, picked=10 → ~20\n"
        "assert per_sample_ce[0].item() > 19.0 and per_sample_ce[0].item() < 21.0, (\n"
        "    f'CE row 0 sanity: {per_sample_ce[0].item()}'\n"
        ")"
    ),
    solution_body=(
        "def pick_target_logits(logits: Tensor, target: Tensor) -> Tensor:\n"
        "    B = logits.shape[0]\n"
        "    # advanced indexing: arange row axis paired with target column axis\n"
        "    return logits[t.arange(B), target]"
    ),
    solution_notes=(
        "**Two 1-D index tensors = positional pairing.** This is the key "
        "rule of NumPy / PyTorch advanced indexing: when you index with "
        "multiple 1-D integer tensors of the same length, they get zipped "
        "into coordinate tuples. `logits[[0,1,2], [t0,t1,t2]]` is `[logits"
        "[0,t0], logits[1,t1], logits[2,t2]]`. The `arange(B)` is just a "
        "compact way to spell `[0, 1, ..., B-1]`.\n\n"
        "**Why this matters for autograd.** Advanced indexing is "
        "differentiable: the gradient w.r.t. `logits` is a sparse tensor "
        "that scatters the upstream `(B,)` gradient back to the original "
        "`(B, C)` positions (zeros everywhere except `(i, target[i])`). A "
        "Python for-loop would materialize each scalar separately, breaking "
        "this clean gradient path.\n\n"
        "**Common alternative — `gather`.** `logits.gather(dim=1, "
        "index=target.unsqueeze(-1)).squeeze(-1)` does the same thing. "
        "Slightly more verbose but generalizes cleanly to higher-rank "
        "tensors. For the `(B, C)` classification case, `arange` indexing "
        "is the idiomatic move."
    ),
)


# =========================================================================
# atom: sgd-vanilla-from-scratch
# =========================================================================

SPEC_SGD_VANILLA = _spec(
    atom_id="sgd-vanilla-from-scratch",
    subtopic="Optimizer: SGD vanilla from scratch",
    recap=RECAP_SGD_VANILLA,
    ex_idx=1,
    ex_title="single-step SGD: in-place update + zero the grad",
    slug="sgd-vanilla-step-and-zero-grad",
    bloom="Apply",
    difficulty_num=2,
    keywords=["sgd", "optimizer", "in-place-update", "zero-grad", "vanilla"],
    kcs=["sgd-vanilla-from-scratch", "grad-accumulate-on-leaf"],
    lo=(
        "Apply the vanilla-SGD update rule (`p -= lr * p.grad`) in place "
        "across a parameter list, then zero each parameter's grad so the "
        "next backward call doesn't double-accumulate."
    ),
    prompt_body=(
        "Implement `sgd_step(params, lr)`. One pass of vanilla SGD over a "
        "list of `MiniTensor` parameters:\n\n"
        "1. For each `p in params`:\n"
        "   - Skip if `p.grad is None` (a parameter that didn't participate "
        "in the last forward — its grad slot is empty; updating with `None` "
        "would crash).\n"
        "   - Otherwise: **in-place** update `p.array -= lr * p.grad`. "
        "Use `p.array -=` (or `.sub_`), NOT `p.array = p.array - ...` — "
        "the test asserts that the underlying tensor object is preserved.\n"
        "2. After the update, set `p.grad = None` so the next backward "
        "call starts from a clean slate.\n\n"
        "**Returns** `None`. Mutates `params` in place.\n\n"
        "**Why in-place.** The same `p.array` tensor object stays alive — "
        "any external reference (state dicts, checkpointers, the "
        "`build_parents` dict you handed to the autograd dispatcher) keeps "
        "pointing to the live weights. Re-binding `p.array` to a new tensor "
        "would break those references.\n\n"
        "**Why `p.grad = None` (and not `.zero_()`).** Both work, but "
        "`None` is cheaper (no memory write, just dropping the reference) "
        "and is what PyTorch recommends since 1.7. The next backward call "
        "checks `if p.grad is None: p.grad = first_contribution`, which is "
        "exactly the leaf-accumulate pattern from a prior atom.\n\n"
        "No momentum, no weight decay, no Nesterov — just the 5-line "
        "baseline that ARENA's training-loop drill assembles."
    ),
    stub=(
        "def sgd_step(params: list, lr: float) -> None:\n"
        '    """One SGD step: p.array -= lr * p.grad in place, then p.grad = None."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- update direction is correct: p -= lr * p.grad ---\n"
        "p1 = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "p1.grad = t.tensor([0.1, 0.2, 0.3])\n"
        "p2 = MiniTensor(t.tensor([10.0, 20.0]), requires_grad=True)\n"
        "p2.grad = t.tensor([1.0, 2.0])\n"
        "\n"
        "p1_array_id = id(p1.array)\n"
        "p2_array_id = id(p2.array)\n"
        "\n"
        "sgd_step([p1, p2], lr=0.1)\n"
        "\n"
        "# value check\n"
        "assert t.allclose(p1.array, t.tensor([0.99, 1.98, 2.97]), atol=1e-6), (\n"
        "    f'p1 update wrong: {p1.array}'\n"
        ")\n"
        "assert t.allclose(p2.array, t.tensor([9.9, 19.8]), atol=1e-6), (\n"
        "    f'p2 update wrong: {p2.array}'\n"
        ")\n"
        "\n"
        "# --- grad cleared after update ---\n"
        "assert p1.grad is None, 'p1.grad must be None after sgd_step'\n"
        "assert p2.grad is None, 'p2.grad must be None after sgd_step'\n"
        "\n"
        "# --- in-place update preserves the tensor object identity ---\n"
        "assert id(p1.array) == p1_array_id, (\n"
        "    'p1.array must be the SAME tensor object after sgd_step — '\n"
        "    'did you use `p.array = p.array - lr * p.grad`? Use `-=` or `.sub_`.'\n"
        ")\n"
        "assert id(p2.array) == p2_array_id, 'p2.array object identity broken'\n"
        "\n"
        "# --- learning rate scales: lr=0 produces no change ---\n"
        "p3 = MiniTensor(t.tensor([5.0, 5.0]), requires_grad=True)\n"
        "p3.grad = t.tensor([1.0, 1.0])\n"
        "before = p3.array.clone()\n"
        "sgd_step([p3], lr=0.0)\n"
        "assert t.allclose(p3.array, before), 'lr=0 must leave params unchanged'\n"
        "assert p3.grad is None, 'lr=0 must still clear the grad (zero-grad semantics)'\n"
        "\n"
        "# --- parameters with grad=None are SKIPPED (not crashed on) ---\n"
        "p4 = MiniTensor(t.tensor([7.0, 7.0]), requires_grad=True)\n"
        "p4.grad = None  # didn't participate in last forward\n"
        "p5 = MiniTensor(t.tensor([100.0]), requires_grad=True)\n"
        "p5.grad = t.tensor([10.0])\n"
        "before_p4 = p4.array.clone()\n"
        "sgd_step([p4, p5], lr=0.5)\n"
        "assert t.allclose(p4.array, before_p4), 'p4 had grad=None — must be left alone'\n"
        "assert t.allclose(p5.array, t.tensor([95.0])), 'p5 update wrong'\n"
        "assert p4.grad is None and p5.grad is None\n"
        "\n"
        "# --- empty params list is a no-op ---\n"
        "sgd_step([], lr=1.0)  # must not crash\n"
        "\n"
        "# --- two successive steps converge a tiny quadratic ---\n"
        "# Minimize (w - 5)^2; gradient is 2*(w - 5); start at w=0.\n"
        "w = MiniTensor(t.tensor([0.0]), requires_grad=True)\n"
        "for step in range(100):\n"
        "    w.grad = 2 * (w.array - 5)\n"
        "    sgd_step([w], lr=0.1)\n"
        "assert abs(w.array.item() - 5.0) < 1e-3, (\n"
        "    f'convergence test failed: w should approach 5, got {w.array.item()}'\n"
        ")"
    ),
    solution_body=(
        "def sgd_step(params: list, lr: float) -> None:\n"
        "    for p in params:\n"
        "        if p.grad is None:\n"
        "            continue\n"
        "        # in-place: keep p.array as the SAME tensor object\n"
        "        p.array -= lr * p.grad\n"
        "        # zero the grad so the next backward starts fresh\n"
        "        p.grad = None"
    ),
    solution_notes=(
        "**Why in-place mutation is non-negotiable.** A real training loop "
        "looks like:\n"
        "```python\n"
        "params = list(model.parameters())\n"
        "optimizer = SGD(params, lr=0.01)\n"
        "for batch in loader:\n"
        "    loss = model(batch).backward()\n"
        "    optimizer.step()\n"
        "```\n"
        "`params` was captured once. If `step()` re-binds `p.array = new`, "
        "the `params` list still points to the new arrays (via `p.array`), "
        "but anything else that held a reference to the OLD tensor object "
        "(state dict, checkpoint writer, an `nn.utils.parametrize` wrapper) "
        "is now stale. In-place `-=` keeps every external reference live.\n\n"
        "**Why `p.grad = None` instead of `p.grad.zero_()`.** Both clear "
        "the grad, but `None` is faster (no tensor allocation persisted, no "
        "memory write), and signals to the next backward call that it "
        "should *create* a fresh grad tensor rather than overwriting. "
        "PyTorch's `optimizer.zero_grad(set_to_none=True)` is now the "
        "default since 2.0 for exactly this reason.\n\n"
        "**Convergence sanity.** The included quadratic test "
        "`min (w-5)^2` is the minimal proof that the step rule is correct "
        "(direction AND scale). After 100 steps at lr=0.1, w should be "
        "within 1e-3 of 5 — the geometric decay rate is `|1 - 2*lr|^k = "
        "0.8^100 ≈ 2e-10`, more than enough."
    ),
)


# =========================================================================
# atom: grads-dict-accumulate-parents
# =========================================================================

SPEC_GRADS_DICT = _spec(
    atom_id="grads-dict-accumulate-parents",
    subtopic="Backprop: grads dict accumulate parents",
    recap=RECAP_GRADS_DICT,
    ex_idx=1,
    ex_title="accumulate per-parent contributions in the reverse-pass grads dict",
    slug="grads-dict-accumulate-via-get-default-zero",
    bloom="Apply",
    difficulty_num=3,
    keywords=["grads-dict", "accumulate", "parents", "get-default", "reverse-pass"],
    kcs=["grads-dict-accumulate-parents", "grad-accumulate-on-leaf"],
    lo=(
        "Apply the `grads[parent] = grads.get(parent, 0) + contribution` "
        "pattern across a parents-list, correctly summing contributions when "
        "the same parent appears more than once."
    ),
    prompt_body=(
        "Implement `accumulate_into_grads(grads, contributions)`. Mutates "
        "the `grads` dict in place:\n\n"
        "- `grads`: `dict[MiniTensor, torch.Tensor]` — the reverse-pass "
        "accumulator. May start empty or may already have entries.\n"
        "- `contributions`: `list[tuple[MiniTensor, torch.Tensor]]` — a list "
        "of `(parent_node, new_gradient)` pairs the dispatcher just "
        "computed for one node's outgoing edges.\n\n"
        "For each `(parent, g)` in `contributions`:\n"
        "```\n"
        "grads[parent] = grads.get(parent, 0) + g\n"
        "```\n\n"
        "Two rules — both critical:\n\n"
        "**1. Use `.get(parent, 0)`, not `grads[parent]`.** A parent visited "
        "for the first time isn't in the dict yet. `grads[parent]` would "
        "raise `KeyError`. `grads.get(parent, 0)` seeds with the additive "
        "identity 0, which broadcasts correctly with any-shape tensor.\n\n"
        "**2. Use `+`, not `+=` or overwrite.** `+=` mutates the existing "
        "tensor (dangerous if the caller holds a reference). Overwriting "
        "`grads[parent] = g` drops earlier contributions — exactly the bug "
        "that breaks `y = x + x` reverse pass (where `x` is parent twice "
        "and both contributions must sum).\n\n"
        "Returns `None`. The caller iterates `grads` after this finishes.\n\n"
        "**Why this is the load-bearing line of the reverse pass.** "
        "Topological-sort the graph, walk it in reverse, and for each node "
        "call this function with the dispatcher's per-arg contributions. "
        "When the walk reaches a leaf, `grads[leaf]` already holds the "
        "fully-summed total derivative — ready to copy into `leaf.grad`."
    ),
    stub=(
        "def accumulate_into_grads(grads: dict, contributions: list) -> None:\n"
        '    """Sum each (parent, g) contribution into grads[parent] using get-default-zero."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- single contribution to an empty grads dict ---\n"
        "p1 = MiniTensor(t.zeros(3))\n"
        "p2 = MiniTensor(t.zeros(3))\n"
        "grads = {}\n"
        "g1 = t.tensor([1.0, 2.0, 3.0])\n"
        "ret = accumulate_into_grads(grads, [(p1, g1)])\n"
        "assert ret is None, 'should return None (mutates in place)'\n"
        "assert p1 in grads, 'p1 must be added to grads'\n"
        "assert t.allclose(grads[p1], g1), f'grads[p1] = {grads[p1]}'\n"
        "\n"
        "# --- two parents, both new ---\n"
        "grads = {}\n"
        "accumulate_into_grads(grads, [(p1, t.tensor([1.0, 1.0, 1.0])),\n"
        "                              (p2, t.tensor([5.0, 5.0, 5.0]))])\n"
        "assert len(grads) == 2\n"
        "assert t.allclose(grads[p1], t.tensor([1.0, 1.0, 1.0]))\n"
        "assert t.allclose(grads[p2], t.tensor([5.0, 5.0, 5.0]))\n"
        "\n"
        "# --- THE CRITICAL TEST: same parent appears TWICE → contributions sum ---\n"
        "grads = {}\n"
        "accumulate_into_grads(grads, [(p1, t.tensor([1.0, 2.0, 3.0])),\n"
        "                              (p1, t.tensor([10.0, 20.0, 30.0]))])\n"
        "assert t.allclose(grads[p1], t.tensor([11.0, 22.0, 33.0])), (\n"
        "    f'same-parent contributions must SUM, got {grads[p1]} — '\n"
        "    'did you overwrite instead of accumulating?'\n"
        ")\n"
        "\n"
        "# --- contributions accumulate ACROSS calls (parent already in grads) ---\n"
        "grads = {p1: t.tensor([100.0, 100.0, 100.0])}\n"
        "accumulate_into_grads(grads, [(p1, t.tensor([1.0, 2.0, 3.0]))])\n"
        "assert t.allclose(grads[p1], t.tensor([101.0, 102.0, 103.0])), (\n"
        "    f'pre-existing grads entry must be ADDED to, got {grads[p1]}'\n"
        ")\n"
        "\n"
        "# --- empty contributions list is a no-op ---\n"
        "grads = {p1: t.tensor([7.0, 7.0, 7.0])}\n"
        "accumulate_into_grads(grads, [])\n"
        "assert t.allclose(grads[p1], t.tensor([7.0, 7.0, 7.0])), 'empty list must not modify grads'\n"
        "\n"
        "# --- `+` not `+=`: the original grad tensor must NOT be mutated ---\n"
        "original = t.tensor([5.0, 5.0])\n"
        "grads = {p1: original}\n"
        "original_id = id(original)\n"
        "accumulate_into_grads(grads, [(p1, t.tensor([1.0, 1.0]))])\n"
        "# After the +, grads[p1] should be a NEW tensor; the original must be untouched.\n"
        "assert t.allclose(original, t.tensor([5.0, 5.0])), (\n"
        "    f'the previous grad tensor must NOT be mutated in place, got {original}; '\n"
        "    'did you use `+=` instead of `+`?'\n"
        ")\n"
        "assert id(grads[p1]) != original_id, 'grads[p1] must be re-bound to a new tensor'\n"
        "assert t.allclose(grads[p1], t.tensor([6.0, 6.0])), f'sum wrong: {grads[p1]}'\n"
        "\n"
        "# --- the y = x + x scenario: parent x appears twice in one node's contributions ---\n"
        "# Reverse pass for `y = x + x` produces contributions [(x, grad_out), (x, grad_out)]\n"
        "# — argnum 0 and argnum 1 both feed back to x. Both must sum into grads[x].\n"
        "x = MiniTensor(t.tensor([1.0, 1.0]), requires_grad=True)\n"
        "grad_out = t.tensor([10.0, 10.0])\n"
        "grads = {}\n"
        "accumulate_into_grads(grads, [(x, grad_out), (x, grad_out)])\n"
        "assert t.allclose(grads[x], t.tensor([20.0, 20.0])), (\n"
        "    f'y = x + x reverse-pass case: grads[x] must equal 2 * grad_out, got {grads[x]}'\n"
        ")"
    ),
    solution_body=(
        "def accumulate_into_grads(grads: dict, contributions: list) -> None:\n"
        "    for parent, g in contributions:\n"
        "        # .get(parent, 0) seeds first-touch with additive identity;\n"
        "        # `+` produces a fresh tensor (no in-place mutation of any prior grad).\n"
        "        grads[parent] = grads.get(parent, 0) + g"
    ),
    solution_notes=(
        "**`get(parent, 0)` vs `grads[parent]`.** The bare lookup raises "
        "`KeyError` on first touch. The `.get(parent, 0)` default seeds "
        "the accumulator with the integer 0 — which then broadcasts "
        "correctly with the tensor `g` (producing a fresh tensor of the "
        "right shape, dtype, and device). Subsequent calls find an actual "
        "tensor in the dict and add to it.\n\n"
        "**`+` vs `+=`.** `tensor_a + tensor_b` allocates a new tensor; "
        "neither operand is mutated. `tensor_a += tensor_b` mutates "
        "`tensor_a` in place, which is faster but breaks any caller that "
        "holds a reference to the old grad tensor. The reverse-pass "
        "dispatcher often hands grad tensors around — mutating them in "
        "place corrupts other computations.\n\n"
        "**Why `y = x + x` is the canonical stress case.** Both arg-0 and "
        "arg-1 of the add point back to the same parent `x`. The "
        "dispatcher hands the accumulator two contributions for the same "
        "parent in a single call. An overwriting implementation would keep "
        "only the last contribution — `dL/dx` would be off by a factor of "
        "2. The test asserts the sum explicitly.\n\n"
        "**Composes with leaf-accumulate.** At the end of the reverse "
        "walk, the dispatcher copies `grads[leaf]` into `leaf.grad` via "
        "the `accumulate_grad` helper from the previous atom. The `grads` "
        "dict is the transient working memory; `leaf.grad` is the "
        "persistent output."
    ),
)


# ---------------------------------------------------------------- assembly

ALL_SPECS = [
    SPEC_LINEAR_AFFINE,
    SPEC_KAIMING_SF,
    SPEC_PARAMETER_WRAP,
    SPEC_MODULE_BASE,
    SPEC_LOGSUMEXP_CE,
    SPEC_ARANGE_FANCY,
    SPEC_SGD_VANILLA,
    SPEC_GRADS_DICT,
]


# ---------------------------------------------------------------------------
# Verifier — exec the SHARED PREAMBLE + each solution + each test body in a
# fresh namespace before emitting. Catches author drift at build time.
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    from torch import Tensor

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"
        ns = {
            "t": t,
            "np": np,
            "Tensor": Tensor,
            "_dd_passed": set(),
            "__name__": "__main__",
        }
        t.manual_seed(0)
        np.random.seed(0)

        try:
            exec(_CUSTOM_TENSOR_PREAMBLE, ns)
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
    print(f"[custom-tensor batch6] Verifying {len(ALL_SPECS)} specs against torch backend...")
    _verify_all(ALL_SPECS)

    print(f"\n[custom-tensor batch6] All verified — emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[custom-tensor batch6] {len(ALL_SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
