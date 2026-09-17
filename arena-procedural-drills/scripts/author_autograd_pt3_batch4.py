#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA manual-autograd PART 3 atoms —
the WRAPPER MECHANICS + VALUE-SEMANTICS layer of the tiny-autograd engine.

Eight single-exercise standalones, under ``prereqs_autograd_pt3/``:

  * box-array-to-tensor-with-recipe    — ex1
  * unbox-args-tensor-to-array         — ex1
  * get-children-callable-param        — ex1
  * coerce-float-arg-to-array          — ex1
  * inplace-op-unsafe-warning          — ex1
  * parameter-subclass-of-tensor       — ex1
  * grad-accumulate-on-leaf            — ex1
  * sum-and-broadcast-duality          — ex1

These are the WRAPPER plumbing + value-semantics that the part-1 backward
fns (`prereqs_autograd_internals/`, `prereqs_backprop/`) compose with.

Where part-1 covered the math (chain rule, arg-position, kwargs, Recipe,
parents, grad toggle, requires_grad, unbroadcast), this batch covers:

- **box/unbox**       — the two halves of wrap_forward_fn (raw <-> Tensor)
- **get_children**    — nn.Module-style param iteration
- **float coercion**  — `multiply(t, 3.0)` → wrap the 3.0 before fwd
- **in-place guard**  — refuse mutation when a Recipe exists (graph safety)
- **Parameter**       — Tensor subclass that defaults to requires_grad=True
- **leaf grad accum** — `leaf.grad = (leaf.grad or 0) + g` at backward time
- **sum/broadcast**   — duality: forward sum(axis) ↔ backward broadcast(axis)

Tests use plain ``torch.Tensor`` for shape/value math; we never call
``torch.autograd`` on the hand-written ops.

Self-verifier execs the SHARED PREAMBLE + each spec's solution + test_body
in a fresh namespace before emitting, so author drift is caught at build
time, not at student-run time.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_pt3"


# ---------------------------------------------------------------- atom recaps

RECAP_BOX = (
    "## Box array → Tensor (with Recipe) — quick refresher\n"
    "\n"
    "The **second half** of `wrap_forward_fn` takes a raw `torch.Tensor` "
    "(the output of `fwd_fn(*raw, **kw)`) and **boxes** it back into a "
    "`Tensor` wrapper, attaching a freshly-constructed `Recipe` so the "
    "reverse pass can find the parent edges later:\n"
    "\n"
    "```python\n"
    "out = Tensor(out_raw, requires_grad=requires_grad)\n"
    "if requires_grad:\n"
    "    out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
    "```\n"
    "\n"
    "Two rules:\n"
    "- **Always attach `out.recipe` when `requires_grad` is True.** Without "
    "  it, the reverse pass hits a non-leaf with no Recipe → KeyError in "
    "  `BACK_FUNCS.get(...)`.\n"
    "- **Skip the Recipe when `requires_grad` is False.** Saves the graph "
    "  bookkeeping during inference / no_grad blocks. Leaves are also "
    "  recipe-less (`.recipe is None`).\n"
    "\n"
    "The Recipe holds the *raw* args (already unboxed by the first half of "
    "the wrapper). The Tensor wrapper carries `.array` (the raw underlying "
    "tensor) and `.requires_grad` (propagated by the toggle/any-input gate)."
)

RECAP_UNBOX = (
    "## Unbox Tensor args → raw arrays — quick refresher\n"
    "\n"
    "The **first half** of `wrap_forward_fn` strips wrappers off positional "
    "args so the underlying `fwd_fn` (e.g. `torch.log`) sees raw "
    "`torch.Tensor` (or scalars / shape tuples) — it has no idea our "
    "`Tensor` class exists:\n"
    "\n"
    "```python\n"
    "raw_args = tuple(\n"
    "    a.array if isinstance(a, Tensor) else a\n"
    "    for a in args\n"
    ")\n"
    "out_raw = fwd_fn(*raw_args, **kwargs)\n"
    "```\n"
    "\n"
    "Two rules:\n"
    "- **`isinstance(a, Tensor)` is the gate.** Anything else (int, float, "
    "  tuple, ndarray) passes through untouched.\n"
    "- **Read `.array`, never copy.** The raw tensor stays the same object "
    "  — the Recipe later stores these same raw tensors for replay; "
    "  cloning would burn memory and break identity invariants.\n"
    "\n"
    "Dual of `build_parents`: where `build_parents` *keeps* the wrappers "
    "(filtered, keyed by argnum), `unbox_args` *replaces* them with their "
    "`.array`. Same `isinstance` check, opposite transform."
)

RECAP_GET_CHILDREN = (
    "## get_children — nn.Module-style param iteration — quick refresher\n"
    "\n"
    "`nn.Module` walks its **submodules** + **parameters** so the optimizer "
    "can find every trainable tensor. The minimal pattern: scan `__dict__` "
    "for any attribute that is a `Tensor` (or another `Module`) and yield "
    "`(name, value)`:\n"
    "\n"
    "```python\n"
    "def get_children(self):\n"
    "    for name, val in self.__dict__.items():\n"
    "        if isinstance(val, (Tensor, Module)):\n"
    "            yield name, val\n"
    "```\n"
    "\n"
    "Three rules:\n"
    "- **`yield`, don't return a list.** Callers usually want to iterate "
    "  once; generators are zero-allocation and compose with `for` cleanly.\n"
    "- **Skip non-Tensor / non-Module attributes.** A `Linear` layer also "
    "  stores `in_features: int`, `out_features: int` — those are config, "
    "  not trainable state, and must not show up in the optimizer's param "
    "  list.\n"
    "- **Recurse via submodules' `get_children` (NOT here).** This helper "
    "  is the LOCAL step; `parameters()` is the recursive walk built on top "
    "  of it. Stay shallow here.\n"
    "\n"
    "Convention: `name` is the attribute name (e.g. `\"weight\"`), used "
    "later for state-dict keys and debug output."
)

RECAP_FLOAT_COERCE = (
    "## Coerce float arg → tensor (boxed) — quick refresher\n"
    "\n"
    "Some forward ops take a `float` constant: `multiply(t, 3.0)`, "
    "`add(t, 1.0)`, `pow(t, 2.0)`. The raw `torch` op accepts it fine "
    "(scalar broadcast), but our autograd wrapper must **coerce the float "
    "to a 0-D tensor** BEFORE running the forward so that:\n"
    "\n"
    "1. The Recipe stores a `torch.Tensor` (not a Python float) at that "
    "argidx — downstream backward fns can do tensor math on it without a "
    "type-check.\n"
    "2. The shape-of-output computation is consistent — `t.tensor(3.0)` "
    "broadcasts the same as the Python literal.\n"
    "\n"
    "```python\n"
    "def coerce_to_tensor(arg):\n"
    "    if isinstance(arg, (int, float)):\n"
    "        return t.tensor(float(arg))\n"
    "    return arg\n"
    "```\n"
    "\n"
    "Note: the coerced value is a **leaf** with `requires_grad=False` — "
    "constants are not parents of the output, so they're skipped by "
    "`build_parents`. The coercion is purely about type uniformity for "
    "the forward call and Recipe storage."
)

RECAP_INPLACE_WARN = (
    "## In-place op unsafe — refuse on Recipe-carrying Tensor — quick refresher\n"
    "\n"
    "In-place ops (`x.add_(y)`, `x.mul_(2)`) overwrite the array in place. "
    "If `x` is a non-leaf — i.e. `x.recipe is not None` — its underlying "
    "array is **also** stored in some other node's Recipe as a `parent` "
    "or `arg`. Mutating it CORRUPTS the graph: a downstream back fn will "
    "compute the wrong local gradient because the cached input it reads "
    "has been silently replaced.\n"
    "\n"
    "Canonical guard — raise (or warn) before doing the mutation:\n"
    "\n"
    "```python\n"
    "def add_inplace(x: Tensor, y: Tensor) -> Tensor:\n"
    "    if x.recipe is not None:\n"
    "        raise RuntimeError(\n"
    "            'in-place op forbidden on a Tensor with a recipe — '\n"
    "            'it would corrupt cached values on the graph'\n"
    "        )\n"
    "    x.array += y.array\n"
    "    return x\n"
    "```\n"
    "\n"
    "Leaves (`.recipe is None`) are safe to mutate — they're not cached as "
    "intermediate values anywhere. Hence the simple rule: **in-place is "
    "OK iff `.recipe is None`**. PyTorch's `RuntimeError: a leaf Variable "
    "that requires grad is being used in an in-place operation` is the "
    "same idea (slightly different scope)."
)

RECAP_PARAMETER = (
    "## Parameter subclasses Tensor — quick refresher\n"
    "\n"
    "`nn.Parameter` is a `Tensor` subclass whose only difference is the "
    "default value of `requires_grad` — it's `True`, not `False`:\n"
    "\n"
    "```python\n"
    "class Parameter(Tensor):\n"
    "    def __init__(self, array, requires_grad: bool = True):\n"
    "        super().__init__(array, requires_grad=requires_grad)\n"
    "```\n"
    "\n"
    "Two rules:\n"
    "- **Subclass, don't compose.** A `Parameter` IS-A `Tensor`. Every op "
    "  that takes `Tensor` accepts a `Parameter` automatically — no "
    "  conversion needed.\n"
    "- **`isinstance(p, Tensor)` returns True for Parameters.** Critical "
    "  because `build_parents` and `unbox_args` both `isinstance(a, "
    "  Tensor)` — Parameters would be silently dropped if they weren't "
    "  Tensor subclasses.\n"
    "\n"
    "The signaling value of the class itself: `isinstance(x, Parameter)` "
    "is how `nn.Module.parameters()` distinguishes the trainable state "
    "from the (Tensor-typed) buffers / activations. Same array data — "
    "different role, encoded purely through the class."
)

RECAP_GRAD_ACCUMULATE = (
    "## Grad accumulate on leaf — quick refresher\n"
    "\n"
    "When the reverse pass reaches a **leaf** (`.recipe is None`, "
    "`.requires_grad is True`), it must **accumulate** the incoming "
    "gradient into `leaf.grad`. Two cases:\n"
    "\n"
    "- **First time we touch this leaf** (`leaf.grad is None`): set "
    "  `leaf.grad = g`.\n"
    "- **Second+ time** (some other path through the graph already "
    "  reached it): `leaf.grad = leaf.grad + g`.\n"
    "\n"
    "Canonical form:\n"
    "```python\n"
    "def accumulate_grad(leaf: Tensor, g: Tensor) -> None:\n"
    "    if leaf.grad is None:\n"
    "        leaf.grad = g\n"
    "    else:\n"
    "        leaf.grad = leaf.grad + g\n"
    "```\n"
    "\n"
    "Why ACCUMULATE and not overwrite: a single Tensor can appear "
    "multiple times in the graph (e.g. `y = w * w` — `w` is a parent of "
    "`y` twice). Each path through the graph contributes its own "
    "`dL/dw`; the *total* derivative is the sum. Overwriting would keep "
    "only the last-visited path's contribution.\n"
    "\n"
    "This is why PyTorch's `.backward()` accumulates into `.grad` and why "
    "you must `optimizer.zero_grad()` between training steps."
)

RECAP_SUM_BROADCAST = (
    "## sum/broadcast duality — quick refresher\n"
    "\n"
    "Reduction and replication are **dual** under backprop. If the forward "
    "pass collapses an axis, the backward pass restores it; if the forward "
    "pass replicates an axis, the backward pass sums it back:\n"
    "\n"
    "| forward op            | backward op                       |\n"
    "|-----------------------|-----------------------------------|\n"
    "| `out = x.sum(dim=k)`  | `grad_x = grad.unsqueeze(k).expand_as(x)` |\n"
    "| `out = x.broadcast_to(big_shape)` | `grad_x = grad.sum_to(x.shape)` |\n"
    "\n"
    "The math: `sum` is a linear map (matrix of all 1s along the axis). "
    "Its transpose is broadcast (matrix of all 1s the other way). "
    "Backprop sends gradients through the *transpose* of the forward "
    "linear map — so `sum` and `broadcast` swap roles.\n"
    "\n"
    "Concretely for `sum_back(grad_out, out, x, *, dim, keepdim=False)`:\n"
    "- If `keepdim=False` (axis was DROPPED), re-insert the axis: "
    "  `grad_out = grad_out.unsqueeze(dim)`.\n"
    "- Then broadcast back to `x.shape`: `grad_x = grad_out.expand_as(x)`.\n"
    "\n"
    "Symmetric: `broadcast_back(grad_out, out, x)` calls `unbroadcast(grad_"
    "out, x)` — sums out the axes that got expanded. Same operation seen "
    "from the other side of the duality."
)


# ---------------------------------------------------------------- spec helper

# Shared autograd-internals preamble that every drill in this file gets injected
# into its standalone notebook via _emit_standalone's extra_imports hook.
# Mirrors the preamble in author_autograd_internals_batch3.py so part-3 drills
# can name MiniTensor / Recipe / grad_tracking_enabled out of the box.
_AUTOGRAD_PREAMBLE = (
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
    # Every drill in this batch needs the shared preamble (MiniTensor, Recipe,
    # grad_tracking_enabled). Append any caller-supplied extras afterward.
    merged_imports = [_AUTOGRAD_PREAMBLE] + list(extra_imports or [])
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
# atom: box-array-to-tensor-with-recipe  (1 exercise)
# =========================================================================

SPEC_BOX = _spec(
    atom_id="box-array-to-tensor-with-recipe",
    subtopic="Backprop: Box array as Tensor + recipe",
    recap=RECAP_BOX,
    ex_idx=1,
    ex_title="box raw output into MiniTensor + attach Recipe when grad-tracked",
    slug="box-raw-output-into-tensor-and-attach-recipe",
    bloom="Apply",
    difficulty_num=3,
    keywords=["box", "recipe", "wrap-forward", "requires-grad", "leaf"],
    kcs=["box-array-to-tensor-with-recipe", "recipe-dataclass"],
    lo=(
        "Apply the boxing half of wrap_forward_fn: wrap a raw output array "
        "in a MiniTensor, set requires_grad from the input gate, and "
        "attach a Recipe only when grad-tracked."
    ),
    prompt_body=(
        "Implement `box_with_recipe(out_raw, fwd_fn, raw_args, kwargs, "
        "parents, requires_grad)`. It is the SECOND half of "
        "`wrap_forward_fn` — given the already-computed raw output and "
        "all the bookkeeping the first half collected, produce the "
        "boxed `MiniTensor` ready to return to the caller.\n\n"
        "Two rules:\n\n"
        "**1. Always box.** Construct `out = MiniTensor(out_raw, "
        "requires_grad=requires_grad)`. The caller hands you the bool "
        "already computed by the requires-grad gate — don't recompute.\n\n"
        "**2. Attach a Recipe IFF `requires_grad` is True.** When False, "
        "leave `out.recipe = None`. Two reasons: (a) inference / no_grad "
        "shouldn't pay the Recipe construction cost, (b) `out.recipe is "
        "None` is the leaf/no-graph signal the reverse pass uses to stop "
        "traversing.\n\n"
        "The Recipe always takes the 4-tuple `(fwd_fn, raw_args, kwargs, "
        "parents)` in that order.\n\n"
        "Inputs:\n"
        "- `out_raw: torch.Tensor` — raw output of `fwd_fn(*raw_args, "
        "**kwargs)`.\n"
        "- `fwd_fn: Callable` — the raw forward fn (e.g. `torch.log`).\n"
        "- `raw_args: tuple` — already unboxed positional args.\n"
        "- `kwargs: dict` — original keyword args.\n"
        "- `parents: dict[int, MiniTensor]` — argnum → input MiniTensor.\n"
        "- `requires_grad: bool` — precomputed gate result.\n\n"
        "Returns: a `MiniTensor`."
    ),
    stub=(
        "def box_with_recipe(\n"
        "    out_raw,\n"
        "    fwd_fn,\n"
        "    raw_args: tuple,\n"
        "    kwargs: dict,\n"
        "    parents: dict,\n"
        "    requires_grad: bool,\n"
        ") -> MiniTensor:\n"
        '    """Wrap raw output into a MiniTensor and attach Recipe iff requires_grad."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- requires_grad=True: Recipe attached with correct 4-field shape ---\n"
        "raw_out = t.tensor([1.0, 2.0])\n"
        "x = MiniTensor(t.tensor([0.0, t.e]))  # t.e is already a Python float\n"
        "parents = {0: x}\n"
        "out = box_with_recipe(raw_out, t.log, (x.array,), {}, parents, True)\n"
        "\n"
        "assert isinstance(out, MiniTensor), 'output must be a MiniTensor'\n"
        "assert out.array is raw_out, 'box must store the SAME raw tensor (identity, not copy)'\n"
        "assert out.requires_grad is True, 'requires_grad must propagate from arg'\n"
        "assert out.recipe is not None, 'requires_grad=True → Recipe must be attached'\n"
        "assert out.recipe.func is t.log, f'recipe.func wrong: {out.recipe.func}'\n"
        "assert out.recipe.args == (x.array,), f'recipe.args wrong: {out.recipe.args}'\n"
        "assert out.recipe.kwargs == {}, f'recipe.kwargs wrong: {out.recipe.kwargs}'\n"
        "assert out.recipe.parents == {0: x}, f'recipe.parents wrong: {out.recipe.parents}'\n"
        "\n"
        "# --- requires_grad=False: NO Recipe attached ---\n"
        "out2 = box_with_recipe(raw_out, t.log, (x.array,), {}, parents, False)\n"
        "assert isinstance(out2, MiniTensor)\n"
        "assert out2.array is raw_out\n"
        "assert out2.requires_grad is False\n"
        "assert out2.recipe is None, (\n"
        "    'requires_grad=False → Recipe must be None '\n"
        "    '(no graph bookkeeping during inference)'\n"
        ")\n"
        "\n"
        "# --- kwargs preserved on Recipe ---\n"
        "raw = t.ones(3, 4).sum(dim=1)\n"
        "x2 = MiniTensor(t.ones(3, 4))\n"
        "out3 = box_with_recipe(raw, t.sum, (x2.array,), {'dim': 1}, {0: x2}, True)\n"
        "assert out3.recipe.kwargs == {'dim': 1}, (\n"
        "    f'kwargs lost on Recipe: {out3.recipe.kwargs}'\n"
        ")\n"
        "\n"
        "# --- empty parents (e.g. const-only call) still gets Recipe when rg=True ---\n"
        "out4 = box_with_recipe(t.tensor(3.0), t.add, (1.0, 2.0), {}, {}, True)\n"
        "assert out4.recipe is not None\n"
        "assert out4.recipe.parents == {}\n"
        "\n"
        "# --- Recipe field order: positional construction matches dataclass ---\n"
        "from dataclasses import fields\n"
        "names = [f.name for f in fields(Recipe)]\n"
        "assert names == ['func', 'args', 'kwargs', 'parents'], (\n"
        "    'Recipe schema drift would break box_with_recipe'\n"
        ")"
    ),
    solution_body=(
        "def box_with_recipe(\n"
        "    out_raw,\n"
        "    fwd_fn,\n"
        "    raw_args: tuple,\n"
        "    kwargs: dict,\n"
        "    parents: dict,\n"
        "    requires_grad: bool,\n"
        ") -> MiniTensor:\n"
        "    # Always box — even when not grad-tracked, callers expect a MiniTensor.\n"
        "    out = MiniTensor(out_raw, requires_grad=requires_grad)\n"
        "    # Attach Recipe ONLY when grad-tracking — saves bookkeeping in no_grad.\n"
        "    if requires_grad:\n"
        "        out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "    return out"
    ),
    solution_notes=(
        "**Why the conditional Recipe.** During inference, gradient "
        "information is not needed — every node would carry a Recipe that "
        "nothing ever reads. Skipping the construction saves both "
        "allocation and reference-keeping on the parents (so they can be "
        "garbage-collected sooner).\n\n"
        "**`out.array is raw_out` (identity).** Boxing wraps; it does NOT "
        "copy. The Tensor wrapper is a thin shell — the same underlying "
        "raw tensor flows through forward and reverse passes. This is "
        "critical for cached-value reuse in backward fns (the `out` "
        "parameter of `sigmoid_back`, etc.).\n\n"
        "**Recipe field order matters.** Recipe is constructed positionally "
        "`Recipe(fwd_fn, raw_args, kwargs, parents)`. If you accidentally "
        "build it as `Recipe(fwd_fn, kwargs, raw_args, parents)` because "
        "your IDE auto-completed wrong, the reverse pass replays "
        "`fwd_fn(*kwargs, **raw_args)` and crashes. The Recipe schema is "
        "a fixed 4-tuple — every wrapper in the codebase reads it the "
        "same way."
    ),
)


# =========================================================================
# atom: unbox-args-tensor-to-array  (1 exercise)
# =========================================================================

SPEC_UNBOX = _spec(
    atom_id="unbox-args-tensor-to-array",
    subtopic="Backprop: Unbox Tensor args to array",
    recap=RECAP_UNBOX,
    ex_idx=1,
    ex_title="unbox MiniTensor positional args to raw arrays, pass-through non-Tensors",
    slug="unbox-tensor-args-to-raw-arrays",
    bloom="Apply",
    difficulty_num=2,
    keywords=["unbox", "wrap-forward", "isinstance", "raw-array"],
    kcs=["unbox-args-tensor-to-array", "parents-dict-by-argidx"],
    lo=(
        "Apply the unboxing half of wrap_forward_fn: replace each "
        "MiniTensor arg with its `.array`, leave non-Tensors untouched, "
        "preserve order."
    ),
    prompt_body=(
        "Implement `unbox_args(args)`. Given a tuple of positional inputs "
        "(some `MiniTensor`, some Python scalars / ndarray / shape "
        "tuples), return a NEW tuple where every `MiniTensor` has been "
        "replaced by its `.array`, in the same positions:\n\n"
        "```\n"
        "unbox_args((t1, 3.0, t2))   == (t1.array, 3.0, t2.array)\n"
        "unbox_args((5, t1, 'x'))    == (5, t1.array, 'x')\n"
        "unbox_args(())              == ()\n"
        "```\n\n"
        "This is the FIRST half of `wrap_forward_fn`. The underlying raw "
        "fn (e.g. `torch.log`) doesn't know our `MiniTensor` class "
        "exists — it expects plain `torch.Tensor`. The unbox step bridges "
        "the wrapper-world to the raw-world before the call.\n\n"
        "Two rules:\n\n"
        "**1. `isinstance(a, MiniTensor)` is the gate.** Anything else "
        "passes through unchanged. Don't duck-type on `.array` — random "
        "objects can have an `.array` attribute (numpy ndarrays do, via "
        "the array protocol).\n\n"
        "**2. Read `.array` directly — don't copy.** The raw tensor IS "
        "the same object. The Recipe will store it; copying would burn "
        "memory and break identity invariants used by `is`-checks later.\n\n"
        "Canonical one-liner: `tuple(a.array if isinstance(a, MiniTensor) "
        "else a for a in args)`."
    ),
    stub=(
        "def unbox_args(args: tuple) -> tuple:\n"
        '    """Replace each MiniTensor in args with its .array; preserve order + non-Tensors."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- empty + all-non-Tensor pass-through ---\n"
        "assert unbox_args(()) == ()\n"
        "assert unbox_args((1, 2.0, 'x', (3, 4))) == (1, 2.0, 'x', (3, 4))\n"
        "\n"
        "# --- single MiniTensor unwrapped ---\n"
        "raw1 = t.tensor([1.0, 2.0])\n"
        "t1 = MiniTensor(raw1)\n"
        "result = unbox_args((t1,))\n"
        "assert isinstance(result, tuple), 'must return a tuple, not a list'\n"
        "assert len(result) == 1\n"
        "assert result[0] is raw1, 'must store the SAME raw tensor (identity, not copy)'\n"
        "\n"
        "# --- two MiniTensors at consecutive positions ---\n"
        "raw2 = t.tensor([3.0, 4.0])\n"
        "t2 = MiniTensor(raw2)\n"
        "result = unbox_args((t1, t2))\n"
        "assert result == (raw1, raw2)\n"
        "assert result[0] is raw1 and result[1] is raw2, 'identity preserved'\n"
        "\n"
        "# --- mixed: Tensor then float ---\n"
        "result = unbox_args((t1, 3.0))\n"
        "assert result == (raw1, 3.0)\n"
        "\n"
        "# --- mixed: float then Tensor (order preserved, NOT collapsed) ---\n"
        "result = unbox_args((3.0, t1))\n"
        "assert result == (3.0, raw1), (\n"
        "    f'order must be preserved (not collapsed), got {result}'\n"
        ")\n"
        "\n"
        "# --- mixed batch: int, Tensor, tuple, Tensor ---\n"
        "result = unbox_args((5, t1, (1, 2, 3), t2))\n"
        "assert result == (5, raw1, (1, 2, 3), raw2)\n"
        "\n"
        "# --- raw torch.Tensor MUST pass through (not MiniTensor → unchanged) ---\n"
        "raw_passthrough = t.tensor([9.0])\n"
        "result = unbox_args((raw_passthrough, t1))\n"
        "assert result[0] is raw_passthrough, (\n"
        "    'raw torch.Tensor must pass through untouched (only MiniTensor gets unboxed)'\n"
        ")\n"
        "assert result[1] is raw1\n"
        "\n"
        "# --- length always matches input length ---\n"
        "for inp in [(t1,), (t1, t2), (1, 2, 3, t1, 4), ()]:\n"
        "    assert len(unbox_args(inp)) == len(inp), (\n"
        "        f'unbox_args dropped/added entries: input len {len(inp)}, output {unbox_args(inp)}'\n"
        "    )\n"
        "\n"
        "# --- the unboxed value is a torch.Tensor, not a MiniTensor ---\n"
        "result = unbox_args((t1,))\n"
        "assert isinstance(result[0], t.Tensor), 'unboxed value must be torch.Tensor'\n"
        "assert not isinstance(result[0], MiniTensor), (\n"
        "    'unboxed value must NOT still be a MiniTensor'\n"
        ")"
    ),
    solution_body=(
        "def unbox_args(args: tuple) -> tuple:\n"
        "    return tuple(\n"
        "        a.array if isinstance(a, MiniTensor) else a\n"
        "        for a in args\n"
        "    )"
    ),
    solution_notes=(
        "**Why `isinstance(a, MiniTensor)` and not `hasattr(a, 'array')`.** "
        "Duck-typing on `.array` would catch random objects that happen "
        "to expose `.array` — `numpy.ndarray` literally has an `.array` "
        "interface protocol. `isinstance` is precise: we want the wrapper "
        "class, not anything array-shaped.\n\n"
        "**Returns a tuple, not a generator.** A generator would only "
        "iterate once — `fwd_fn(*raw_args, **kwargs)` would consume it "
        "but Recipe construction (which reuses `raw_args`) would see an "
        "exhausted iterator. Tuple is canonical: immutable, re-iterable, "
        "cheap.\n\n"
        "**Dual of `build_parents`.** Both walk `args` with the same "
        "`isinstance` check. `unbox_args` REPLACES each MiniTensor with "
        "its `.array`. `build_parents` KEEPS each MiniTensor (filtered, "
        "keyed by argnum). Together they're the two outputs of the same "
        "scan — sometimes written as one combined helper that returns "
        "`(raw_args, parents)` in a single pass."
    ),
)


# =========================================================================
# atom: get-children-callable-param  (1 exercise)
# =========================================================================

SPEC_GET_CHILDREN = _spec(
    atom_id="get-children-callable-param",
    subtopic="Backprop: get_children callable param",
    recap=RECAP_GET_CHILDREN,
    ex_idx=1,
    ex_title="get_children yields (name, value) for Tensor-valued attributes",
    slug="get-children-yields-tensor-valued-attributes",
    bloom="Apply",
    difficulty_num=3,
    keywords=["get_children", "nn.Module", "parameters", "yield", "isinstance"],
    kcs=["get-children-callable-param", "parameter-subclass-of-tensor"],
    lo=(
        "Apply the nn.Module-style get_children pattern: scan __dict__ "
        "for MiniTensor-valued attributes and yield (name, tensor) pairs, "
        "skipping configuration (ints, strings, layer-shape ints, etc.)."
    ),
    prompt_body=(
        "We've given you a tiny `Module` base class. Implement "
        "`get_children(self)` as a **generator** that yields `(name, "
        "value)` for every attribute on `self` whose value `isinstance(_, "
        "MiniTensor)`:\n\n"
        "```\n"
        "class Linear(Module):\n"
        "    def __init__(self):\n"
        "        self.weight = MiniTensor(t.randn(4, 3), requires_grad=True)\n"
        "        self.bias   = MiniTensor(t.zeros(4),    requires_grad=True)\n"
        "        self.in_features = 3   # config: NOT a child\n"
        "        self.out_features = 4  # config: NOT a child\n"
        "\n"
        "list(Linear().get_children())\n"
        "  → [('weight', <MiniTensor ...>), ('bias', <MiniTensor ...>)]\n"
        "```\n\n"
        "Three rules:\n\n"
        "**1. `yield`, do not `return` a list.** Generators are zero-"
        "allocation and compose with `for` cleanly. The caller usually "
        "wants `for name, child in m.get_children(): ...`.\n\n"
        "**2. Skip non-MiniTensor attributes.** A `Linear` layer also "
        "stores `in_features: int`, `out_features: int` — those are "
        "config, not trainable state, and must NOT show up. Use "
        "`isinstance(val, MiniTensor)`.\n\n"
        "**3. Stay shallow (don't recurse).** This helper is the LOCAL "
        "step; recursive walks (`parameters()`) are built on top of it.\n\n"
        "Scan `self.__dict__.items()` — that's where Python keeps "
        "instance attributes, in insertion order."
    ),
    stub=(
        "class Module:\n"
        '    """Tiny nn.Module stand-in. Subclasses set MiniTensor-valued attrs."""\n'
        "    def get_children(self):\n"
        '        """Yield (name, value) for every MiniTensor attribute on self."""\n'
        "        raise NotImplementedError()"
    ),
    test_body=(
        "import inspect\n"
        "\n"
        "# --- empty module: yields nothing ---\n"
        "m_empty = Module()\n"
        "assert list(m_empty.get_children()) == [], (\n"
        "    'empty module must yield nothing'\n"
        ")\n"
        "\n"
        "# --- it must be a generator (or at least an iterable, not a list) ---\n"
        "class L1(Module):\n"
        "    def __init__(self):\n"
        "        self.w = MiniTensor(t.randn(2, 3), requires_grad=True)\n"
        "        self.b = MiniTensor(t.zeros(2),    requires_grad=True)\n"
        "        self.in_features = 3\n"
        "        self.out_features = 2\n"
        "\n"
        "m = L1()\n"
        "children = list(m.get_children())\n"
        "names = [n for n, _ in children]\n"
        "values = [v for _, v in children]\n"
        "assert names == ['w', 'b'], (\n"
        "    f'should yield only MiniTensor attrs in insertion order, got {names}'\n"
        ")\n"
        "assert values[0] is m.w, 'value must BE the same object (identity, not copy)'\n"
        "assert values[1] is m.b\n"
        "\n"
        "# --- config attrs MUST be skipped ---\n"
        "names_set = set(names)\n"
        "assert 'in_features' not in names_set, (\n"
        "    'int config attrs must not appear in get_children output'\n"
        ")\n"
        "assert 'out_features' not in names_set\n"
        "\n"
        "# --- mixed-type attrs: only MiniTensor instances are yielded ---\n"
        "class Mixed(Module):\n"
        "    def __init__(self):\n"
        "        self.name = 'mlp'           # str → skip\n"
        "        self.dropout_p = 0.1        # float → skip\n"
        "        self.shape = (3, 4)         # tuple → skip\n"
        "        self.W = MiniTensor(t.randn(4, 3))   # MiniTensor → KEEP\n"
        "        self.scratch = t.zeros(4)   # raw torch.Tensor → skip\n"
        "        self.b = MiniTensor(t.zeros(4))      # MiniTensor → KEEP\n"
        "\n"
        "mx = Mixed()\n"
        "names = [n for n, _ in mx.get_children()]\n"
        "assert names == ['W', 'b'], (\n"
        "    f'must yield only MiniTensor attrs, in insertion order, got {names}'\n"
        ")\n"
        "\n"
        "# --- raw torch.Tensor must be SKIPPED (only MiniTensor counts as a child) ---\n"
        "assert 'scratch' not in set(names), (\n"
        "    'raw torch.Tensor must be skipped (only MiniTensor counts as a child)'\n"
        ")\n"
        "\n"
        "# --- Parameter (Tensor subclass) is also picked up, since isinstance(p, MiniTensor) holds ---\n"
        "class Param(MiniTensor):\n"
        "    def __init__(self, array):\n"
        "        super().__init__(array, requires_grad=True)\n"
        "\n"
        "class WithParam(Module):\n"
        "    def __init__(self):\n"
        "        self.p = Param(t.randn(3))\n"
        "        self.q = MiniTensor(t.randn(3))\n"
        "\n"
        "wp = WithParam()\n"
        "names = [n for n, _ in wp.get_children()]\n"
        "assert names == ['p', 'q'], (\n"
        "    f'Parameter subclass of MiniTensor must be included, got {names}'\n"
        ")\n"
        "\n"
        "# --- get_children itself behaves like a generator: returns an iterator object ---\n"
        "iterator = m.get_children()\n"
        "assert iter(iterator) is iterator or inspect.isgenerator(iterator), (\n"
        "    'get_children should be a generator / iterator, not a list — '\n"
        "    'zero-allocation for callers that just want `for ... in`'\n"
        "    f' (got {type(iterator).__name__})'\n"
        ")"
    ),
    solution_body=(
        "class Module:\n"
        "    def get_children(self):\n"
        "        for name, val in self.__dict__.items():\n"
        "            if isinstance(val, MiniTensor):\n"
        "                yield name, val"
    ),
    solution_notes=(
        "**Why `__dict__.items()` (not `dir(self)`).** `dir` includes "
        "inherited class attributes, methods, dunder fields — all junk "
        "for this use case. `__dict__` is exactly the instance attributes "
        "the user set in `__init__`, in insertion order (Python 3.7+).\n\n"
        "**Why a generator.** A caller that just wants to iterate "
        "(`for name, child in m.get_children(): ...`) pays zero allocation "
        "cost — no intermediate list. If a caller wants a list, they can "
        "always `list(m.get_children())` explicitly.\n\n"
        "**Parameter subclassing matters here.** `isinstance(p, "
        "MiniTensor)` is True for any `MiniTensor` subclass (including "
        "`Parameter`). If `Parameter` were composition-not-inheritance, "
        "`get_children` would silently skip every trainable param — which "
        "is exactly the reason for the IS-A relationship in the part-3 "
        "`parameter-subclass-of-tensor` atom.\n\n"
        "**Recursion lives elsewhere.** A full `parameters(recurse=True)` "
        "would walk children of children. Keep this primitive shallow — "
        "single responsibility, easy to test, easy to compose."
    ),
)


# =========================================================================
# atom: coerce-float-arg-to-array  (1 exercise)
# =========================================================================

SPEC_COERCE = _spec(
    atom_id="coerce-float-arg-to-array",
    subtopic="Backprop: Coerce float arg to array",
    recap=RECAP_FLOAT_COERCE,
    ex_idx=1,
    ex_title="coerce_to_array: wrap int/float as 0-D tensor, pass-through others",
    slug="coerce-float-or-int-arg-to-0d-tensor",
    bloom="Apply",
    difficulty_num=2,
    keywords=["coerce", "scalar-promotion", "0-d-tensor", "wrap-forward"],
    kcs=["coerce-float-arg-to-array", "unbox-args-tensor-to-array"],
    lo=(
        "Apply the float/int coercion step at the wrap_forward_fn entry: "
        "promote Python scalars to 0-D tensors so the Recipe and "
        "downstream back fns see uniform tensor types."
    ),
    prompt_body=(
        "Implement `coerce_to_array(arg)`. The autograd wrapper sees "
        "calls like `multiply(t, 3.0)` where the second positional arg "
        "is a Python `float`. Before doing anything else, we promote it "
        "to a 0-D `torch.Tensor` so:\n\n"
        "1. The Recipe stores a `torch.Tensor` (not a `float`) at that "
        "argidx — back fns can do tensor math on it without a type-check.\n"
        "2. Forward-call broadcasting is consistent — `t.tensor(3.0)` "
        "broadcasts the same as the Python literal.\n\n"
        "Rules:\n\n"
        "- **`int` or `float` → `torch.tensor(float(arg))`.** Always "
        "promote to `float32` (the default for `torch.tensor(0.0)`). Even "
        "for `int` input — most downstream ops want float math anyway "
        "(`multiply`, `divide`, `pow`).\n"
        "- **`torch.Tensor` → pass-through.** No copy. The whole point is "
        "that already-tensor args are left alone.\n"
        "- **Anything else (tuple, str, None, ...) → pass-through.** "
        "Shape-tuples for `reshape` / `view` are legitimate non-tensor "
        "args and must not be wrapped.\n\n"
        "Signature: `coerce_to_array(arg) -> any`. Result: tensor for "
        "scalars, original for everything else.\n\n"
        "Note: do NOT use `isinstance(arg, (int, float, bool))` — bool is "
        "a subclass of int in Python, and we DON'T want to coerce `True`/"
        "`False` to a tensor (they're metadata flags like `keepdim`). "
        "Stick to `(int, float)` and explicitly check NOT `bool`."
    ),
    stub=(
        "def coerce_to_array(arg):\n"
        '    """Promote int/float to 0-D tensor; pass through tensors and other types."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- float coerced to 0-D float tensor ---\n"
        "out = coerce_to_array(3.0)\n"
        "assert isinstance(out, t.Tensor), f'float must coerce to Tensor, got {type(out)}'\n"
        "assert out.ndim == 0, f'must be 0-D (scalar), got ndim={out.ndim}'\n"
        "assert out.dtype == t.float32, f'must be float32, got {out.dtype}'\n"
        "assert out.item() == 3.0\n"
        "\n"
        "# --- int coerced to 0-D float tensor (NOT int tensor) ---\n"
        "out = coerce_to_array(5)\n"
        "assert isinstance(out, t.Tensor)\n"
        "assert out.ndim == 0\n"
        "assert out.dtype == t.float32, (\n"
        "    f'int must coerce to float (not int) — downstream ops want float math, '\n"
        "    f'got {out.dtype}'\n"
        ")\n"
        "assert out.item() == 5.0\n"
        "\n"
        "# --- existing tensor passes through unchanged (identity) ---\n"
        "raw = t.tensor([1.0, 2.0, 3.0])\n"
        "out = coerce_to_array(raw)\n"
        "assert out is raw, 'tensor input must pass through (no copy)'\n"
        "\n"
        "# --- shape tuple (e.g. reshape arg) passes through ---\n"
        "out = coerce_to_array((3, 4))\n"
        "assert out == (3, 4)\n"
        "assert not isinstance(out, t.Tensor)\n"
        "\n"
        "# --- string / None / list pass through ---\n"
        "assert coerce_to_array('x') == 'x'\n"
        "assert coerce_to_array(None) is None\n"
        "assert coerce_to_array([1, 2]) == [1, 2]\n"
        "\n"
        "# --- bool must NOT be coerced (subclass-of-int trap) ---\n"
        "# True/False are commonly used as keepdim flags etc. — must pass through.\n"
        "out = coerce_to_array(True)\n"
        "assert out is True, (\n"
        "    'bool must pass through (NOT be coerced) — '\n"
        "    'bool is a subclass of int in Python; avoid (int, float) catching it'\n"
        ")\n"
        "assert coerce_to_array(False) is False\n"
        "\n"
        "# --- the coerced tensor broadcasts the same way the float would ---\n"
        "# this is the operational reason for coercion: forward call semantics unchanged.\n"
        "raw = t.tensor([1.0, 2.0, 3.0])\n"
        "scalar = coerce_to_array(2.0)\n"
        "assert t.allclose(raw * scalar, t.tensor([2.0, 4.0, 6.0])), (\n"
        "    'coerced scalar must broadcast like the float literal'\n"
        ")\n"
        "\n"
        "# --- negative float ---\n"
        "out = coerce_to_array(-1.5)\n"
        "assert out.item() == -1.5"
    ),
    solution_body=(
        "def coerce_to_array(arg):\n"
        "    # bool is a subclass of int — must check for it FIRST and pass through,\n"
        "    # otherwise the (int, float) branch below would coerce True/False to tensors.\n"
        "    if isinstance(arg, bool):\n"
        "        return arg\n"
        "    if isinstance(arg, (int, float)):\n"
        "        return t.tensor(float(arg))\n"
        "    return arg"
    ),
    solution_notes=(
        "**Why coerce at the wrapper entry, not in each back fn.** Doing "
        "it once in the wrapper means every downstream back fn can assume "
        "tensor args. Pushing the check into each back fn would mean N "
        "copies of the same scalar guard — and the Recipe would store "
        "heterogenous types, breaking introspection tools.\n\n"
        "**Why promote `int` to `float` and not `int64`.** Most "
        "elementwise ops we'll backprop through (`multiply`, `divide`, "
        "`pow`, `add`) want floating-point math. `multiply_back0(grad, "
        "out, x, scalar)` does `grad * scalar` — if `scalar` is an "
        "int-tensor, PyTorch's type promotion can give surprising "
        "results. Keeping scalars as float32 sidesteps this.\n\n"
        "**The bool trap.** `isinstance(True, int)` is `True` in Python "
        "— `bool` is literally a subclass of `int`. If you write "
        "`isinstance(arg, (int, float))` without an earlier bool guard, "
        "`coerce_to_array(True)` returns `tensor(1.0)` — but the caller "
        "passed `True` as a `keepdim` kwarg, not a numeric scalar. Now "
        "`sum(x, keepdim=tensor(1.0))` raises a confusing TypeError far "
        "from the cause."
    ),
)


# =========================================================================
# atom: inplace-op-unsafe-warning  (1 exercise)
# =========================================================================

SPEC_INPLACE = _spec(
    atom_id="inplace-op-unsafe-warning",
    subtopic="Backprop: In-place op unsafe warning",
    recap=RECAP_INPLACE_WARN,
    ex_idx=1,
    ex_title="add_inplace_safe: refuse when .recipe is not None, mutate otherwise",
    slug="add-inplace-safe-refuse-when-recipe-attached",
    bloom="Apply",
    difficulty_num=3,
    keywords=["in-place", "graph-safety", "recipe", "guard", "mutation"],
    kcs=["inplace-op-unsafe-warning", "recipe-dataclass"],
    lo=(
        "Apply the in-place safety guard: refuse to mutate any Tensor "
        "whose Recipe is non-None (would corrupt cached values in the "
        "compute graph), allow it only for leaves."
    ),
    prompt_body=(
        "Implement `add_inplace_safe(x: MiniTensor, y: MiniTensor)` — an "
        "in-place `x += y` that **refuses** to mutate `x` when `x` is "
        "part of a compute graph.\n\n"
        "Rule:\n\n"
        "- **If `x.recipe is not None`:** raise `RuntimeError(...)`. The "
        "error message must mention 'in-place' (lower-case, no hyphen — "
        "the test grep is case-insensitive but unambiguous).\n"
        "- **Otherwise (`x.recipe is None`):** mutate `x.array` in place "
        "(`x.array += y.array`) and return `x`.\n\n"
        "**Why the guard exists.** An in-place mutation overwrites "
        "`x.array`. If `x` is non-leaf, its array is also stored in "
        "other nodes' Recipes as a parent / arg. The next time the "
        "reverse pass visits one of those nodes, the back fn reads the "
        "*mutated* value instead of the original — silent correctness "
        "bug.\n\n"
        "Leaves (`.recipe is None`) are safe: they are not cached as "
        "intermediates anywhere. Parameter updates in the optimizer "
        "(`param.array -= lr * param.grad`) are exactly this case — "
        "leaf, no Recipe, mutate freely.\n\n"
        "Signature: `add_inplace_safe(x: MiniTensor, y: MiniTensor) -> "
        "MiniTensor`. The returned MiniTensor must be the SAME object as "
        "`x` (in-place semantics)."
    ),
    stub=(
        "def add_inplace_safe(x: MiniTensor, y: MiniTensor) -> MiniTensor:\n"
        '    """In-place x += y; refuse with RuntimeError if x has a Recipe."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- leaf + leaf: in-place mutation allowed, returns same object ---\n"
        "x = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=False)\n"
        "y = MiniTensor(t.tensor([10.0, 20.0, 30.0]))\n"
        "assert x.recipe is None, 'precondition'\n"
        "result = add_inplace_safe(x, y)\n"
        "\n"
        "assert result is x, 'in-place must return the SAME MiniTensor object'\n"
        "assert t.allclose(x.array, t.tensor([11.0, 22.0, 33.0])), (\n"
        "    f'leaf in-place add failed: {x.array}'\n"
        ")\n"
        "\n"
        "# --- y untouched (only x mutates) ---\n"
        "assert t.allclose(y.array, t.tensor([10.0, 20.0, 30.0])), 'y must not change'\n"
        "\n"
        "# --- non-leaf (x.recipe is not None) → must raise RuntimeError ---\n"
        "x2 = MiniTensor(t.tensor([1.0, 2.0]), requires_grad=True)\n"
        "x2.recipe = Recipe(t.add, (t.zeros(2), t.zeros(2)), {}, {})\n"
        "assert x2.recipe is not None, 'precondition'\n"
        "\n"
        "raised = False\n"
        "try:\n"
        "    add_inplace_safe(x2, y)\n"
        "except RuntimeError as e:\n"
        "    raised = True\n"
        "    msg = str(e).lower()\n"
        "    assert 'in-place' in msg or 'inplace' in msg or 'in place' in msg, (\n"
        "        f'RuntimeError message must mention in-place, got: {e}'\n"
        "    )\n"
        "assert raised, 'add_inplace_safe must raise RuntimeError when x.recipe is set'\n"
        "\n"
        "# --- and CRITICALLY: x2.array must be unchanged after the refusal ---\n"
        "assert t.allclose(x2.array, t.tensor([1.0, 2.0])), (\n"
        "    f'on refusal, x.array must NOT have been mutated, got {x2.array}'\n"
        ")\n"
        "\n"
        "# --- requires_grad=True but recipe=None (leaf param) is ALLOWED ---\n"
        "# This is the optimizer case: param.array -= lr * param.grad on a leaf.\n"
        "p = MiniTensor(t.tensor([5.0, 5.0]), requires_grad=True)\n"
        "assert p.recipe is None\n"
        "g = MiniTensor(t.tensor([1.0, 1.0]))\n"
        "add_inplace_safe(p, g)\n"
        "assert t.allclose(p.array, t.tensor([6.0, 6.0])), (\n"
        "    f'leaf param with requires_grad=True but recipe=None must be mutable, got {p.array}'\n"
        ")\n"
        "\n"
        "# --- the guard is on RECIPE, not on requires_grad ---\n"
        "# Confirming: a recipe-carrying tensor with requires_grad=False is STILL refused.\n"
        "x3 = MiniTensor(t.zeros(2), requires_grad=False)\n"
        "x3.recipe = Recipe(t.add, (), {}, {})\n"
        "raised2 = False\n"
        "try:\n"
        "    add_inplace_safe(x3, y)\n"
        "except RuntimeError:\n"
        "    raised2 = True\n"
        "assert raised2, (\n"
        "    'recipe-not-None must always refuse, regardless of requires_grad'\n"
        ")"
    ),
    solution_body=(
        "def add_inplace_safe(x: MiniTensor, y: MiniTensor) -> MiniTensor:\n"
        "    # Guard: any Tensor whose `.array` may be cached in some downstream\n"
        "    # node's Recipe (i.e. any non-leaf) is unsafe to mutate. The signal\n"
        "    # is `x.recipe is not None` — leaves never carry a Recipe.\n"
        "    if x.recipe is not None:\n"
        "        raise RuntimeError(\n"
        "            'in-place op forbidden on a Tensor with a recipe — '\n"
        "            'would corrupt cached values on the graph'\n"
        "        )\n"
        "    # Leaf path: safe to mutate.\n"
        "    x.array += y.array\n"
        "    return x"
    ),
    solution_notes=(
        "**The guard is on `.recipe`, not on `requires_grad`.** A leaf "
        "parameter has `requires_grad=True` AND `recipe is None` — and "
        "the optimizer mutates it in place every step. Conversely, a "
        "non-leaf intermediate could in theory have `requires_grad=False` "
        "but still be cached in a Recipe (rare, but possible if the "
        "input graph mixed grad-tracked and non-tracked subtrees). The "
        "correctness condition is purely about whether the array is "
        "cached, which the Recipe presence tracks exactly.\n\n"
        "**Why raise, not warn.** A silent warning lets the bug propagate "
        "— the user sees slightly-wrong gradients, blames the model, "
        "and spends days debugging. Raising forces the user to either "
        "(a) realize the in-place was a mistake and replace it with "
        "out-of-place, or (b) call `.detach()` first to peel the Recipe "
        "off explicitly and own the choice.\n\n"
        "**PyTorch's actual message.** `RuntimeError: a leaf Variable "
        "that requires grad is being used in an in-place operation.` The "
        "scope is slightly different (PyTorch tracks leaf-with-rg "
        "specifically), but the pattern is identical: refuse the "
        "mutation, force the user to be explicit."
    ),
)


# =========================================================================
# atom: parameter-subclass-of-tensor  (1 exercise)
# =========================================================================

SPEC_PARAMETER = _spec(
    atom_id="parameter-subclass-of-tensor",
    subtopic="Backprop: Parameter subclasses Tensor",
    recap=RECAP_PARAMETER,
    ex_idx=1,
    ex_title="Parameter subclasses MiniTensor with requires_grad=True default",
    slug="parameter-subclass-of-tensor-requires-grad-default",
    bloom="Apply",
    difficulty_num=2,
    keywords=["parameter", "subclass", "requires-grad", "default", "is-a"],
    kcs=["parameter-subclass-of-tensor", "get-children-callable-param"],
    lo=(
        "Apply the Parameter IS-A Tensor pattern: subclass MiniTensor "
        "with requires_grad=True as the default, preserving isinstance "
        "compatibility so optimizer/get_children find it."
    ),
    prompt_body=(
        "Define `class Parameter(MiniTensor)` — a tiny subclass whose "
        "ONLY behavioral difference is the default value of "
        "`requires_grad`:\n\n"
        "```python\n"
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "```\n\n"
        "Rules:\n\n"
        "**1. Subclass, don't compose.** `Parameter` IS-A `MiniTensor`. "
        "Every op accepting a `MiniTensor` accepts a `Parameter` "
        "automatically — `multiply(p, t)` works without any conversion.\n\n"
        "**2. `requires_grad=True` is the default.** Trainable params "
        "ALWAYS need grad. If the user omits the kwarg, the default "
        "True kicks in.\n\n"
        "**3. Allow override.** `Parameter(arr, requires_grad=False)` "
        "must work — used for frozen layers / fine-tuning.\n\n"
        "**4. `isinstance(p, MiniTensor)` is True for any `Parameter`.** "
        "This is the LOAD-BEARING property: `build_parents`, `unbox_"
        "args`, `get_children` all use `isinstance(_, MiniTensor)` as "
        "their gate. If `Parameter` were a separate class (composition "
        "not inheritance), every trainable param would be silently "
        "skipped by those helpers — the whole autograd layer would "
        "ignore your model's parameters.\n\n"
        "**5. The class itself signals role.** `isinstance(x, "
        "Parameter)` is how a future `parameters()` walker would "
        "distinguish 'trainable state' from 'intermediate Tensor'. Same "
        "data, different role — encoded purely through the class."
    ),
    stub=(
        "class Parameter(MiniTensor):\n"
        '    """Subclass MiniTensor with requires_grad=True as the default."""\n'
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        raise NotImplementedError()\n"
    ),
    test_body=(
        "# --- Parameter is a subclass of MiniTensor ---\n"
        "assert issubclass(Parameter, MiniTensor), (\n"
        "    'Parameter must subclass MiniTensor, not compose with it'\n"
        ")\n"
        "\n"
        "# --- default requires_grad=True ---\n"
        "p = Parameter(t.zeros(3))\n"
        "assert p.requires_grad is True, (\n"
        "    'Parameter default must be requires_grad=True, '\n"
        "    f'got {p.requires_grad}'\n"
        ")\n"
        "assert isinstance(p, MiniTensor), 'Parameter instance must be a MiniTensor too'\n"
        "assert isinstance(p, Parameter)\n"
        "\n"
        "# --- override to False (frozen layer) ---\n"
        "p_frozen = Parameter(t.zeros(3), requires_grad=False)\n"
        "assert p_frozen.requires_grad is False, (\n"
        "    f'override requires_grad=False must work, got {p_frozen.requires_grad}'\n"
        ")\n"
        "\n"
        "# --- .array stored as-is ---\n"
        "raw = t.tensor([1.0, 2.0, 3.0])\n"
        "p2 = Parameter(raw)\n"
        "assert p2.array is raw, 'Parameter must store the raw tensor (identity, not copy)'\n"
        "\n"
        "# --- .recipe is None at construction (leaves carry no Recipe) ---\n"
        "assert p2.recipe is None, 'fresh Parameter is a leaf — no Recipe'\n"
        "\n"
        "# --- .grad starts as None (will be set by accumulate_grad later) ---\n"
        "assert p2.grad is None, 'fresh Parameter grad must start as None'\n"
        "\n"
        "# --- Parameter passes the isinstance(_, MiniTensor) gate used by helpers ---\n"
        "# This is the load-bearing test: every wrapper helper filters by isinstance(_, MiniTensor),\n"
        "# so Parameters must pass that test or they get silently skipped.\n"
        "def _build_parents(args):\n"
        "    return {idx: a for idx, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
        "\n"
        "x = MiniTensor(t.zeros(3))\n"
        "parents = _build_parents((x, p))\n"
        "assert parents == {0: x, 1: p}, (\n"
        "    'Parameter must be picked up by isinstance(_, MiniTensor) — '\n"
        "    f'got {parents}'\n"
        ")\n"
        "\n"
        "# --- the role signal: isinstance(x, Parameter) distinguishes from a plain MiniTensor ---\n"
        "assert isinstance(p, Parameter)\n"
        "assert not isinstance(x, Parameter), (\n"
        "    'plain MiniTensor must NOT pass isinstance(_, Parameter) — '\n"
        "    'subclassing must not pollute the parent class'\n"
        ")"
    ),
    solution_body=(
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)"
    ),
    solution_notes=(
        "**Why a subclass with only a default change.** It buys two "
        "things at near-zero cost:\n"
        "1. **Default value.** `Parameter(t.zeros(3))` is the common case "
        "(trainable weight); kwargs default of True saves the user from "
        "writing `requires_grad=True` on every line of `__init__`.\n"
        "2. **Type-as-tag.** `isinstance(x, Parameter)` is the only "
        "reliable way for `parameters()` to distinguish trainable state "
        "from incidental tensors. Same `.array`, same `.recipe`, same "
        "`.requires_grad` semantics — just a typing marker.\n\n"
        "**Why NOT composition.** A composition design (`class "
        "Parameter: def __init__(self, t): self.tensor = t`) would mean "
        "`isinstance(p, MiniTensor)` is False. Every helper in the "
        "wrapper layer (`build_parents`, `unbox_args`, `get_children`) "
        "filters by `isinstance(_, MiniTensor)` — and would silently "
        "skip every parameter. The autograd layer would ignore your "
        "model's weights — silent, devastating bug.\n\n"
        "**PyTorch's actual design.** `torch.nn.Parameter(torch.Tensor)` "
        "— exactly this pattern. The class body in PyTorch is similarly "
        "minimal: just an override of `__new__` to handle the "
        "`requires_grad=True` default and a `__deepcopy__` for state-"
        "dict-friendly copying. The IS-A relationship is the load-"
        "bearing design choice."
    ),
)


# =========================================================================
# atom: grad-accumulate-on-leaf  (1 exercise)
# =========================================================================

SPEC_ACCUMULATE = _spec(
    atom_id="grad-accumulate-on-leaf",
    subtopic="Backprop: Grad accumulate on leaf",
    recap=RECAP_GRAD_ACCUMULATE,
    ex_idx=1,
    ex_title="accumulate_grad: leaf.grad = (leaf.grad or 0) + g",
    slug="accumulate-grad-on-leaf-set-or-add",
    bloom="Apply",
    difficulty_num=3,
    keywords=["accumulate", "grad", "leaf", "first-touch", "zero-grad"],
    kcs=["grad-accumulate-on-leaf", "parameter-subclass-of-tensor"],
    lo=(
        "Apply the leaf-grad accumulation pattern: set leaf.grad on "
        "first visit, add to it on subsequent visits — the standard "
        "branching pattern that requires .zero_grad() between training "
        "steps."
    ),
    prompt_body=(
        "Implement `accumulate_grad(leaf: MiniTensor, g: torch.Tensor) "
        "-> None`. The reverse pass calls this every time it reaches a "
        "leaf (`leaf.requires_grad is True`, `leaf.recipe is None`). "
        "Two cases:\n\n"
        "**1. First time we touch this leaf** (`leaf.grad is None`):\n"
        "   Set `leaf.grad = g`.\n\n"
        "**2. Second+ time we touch it** (`leaf.grad is not None`):\n"
        "   Set `leaf.grad = leaf.grad + g`.\n\n"
        "Notes:\n\n"
        "- **Mutates `leaf.grad`; returns `None`.** (Caller doesn't need "
        "the value.)\n"
        "- **Use `+`, not `+=`.** Re-bind `leaf.grad` to a new tensor "
        "rather than mutating the existing grad tensor in place. Some "
        "tests check that an externally-held reference to the old grad "
        "tensor is NOT mutated — important for the `optimizer.step()` "
        "case where the grad tensor is read elsewhere.\n"
        "- **Shape: `g.shape == leaf.array.shape`** (caller's "
        "responsibility — earlier `unbroadcast` step handles this). "
        "Trust it.\n\n"
        "**Why accumulate.** A single Tensor can appear multiple times "
        "in the compute graph — e.g. `y = w * w` makes `w` a parent of "
        "`y` twice (argnum 0 AND argnum 1). The total derivative `dL/dw "
        "= dL/dy * 2w` is the SUM of contributions from each path. "
        "Overwriting would keep only the last-visited path's "
        "contribution.\n\n"
        "This is why PyTorch's `.backward()` accumulates into `.grad` "
        "and why training loops must call `optimizer.zero_grad()` (or "
        "manually set `p.grad = None`) between steps — otherwise "
        "gradients from previous steps stay around and corrupt the next "
        "update."
    ),
    stub=(
        "def accumulate_grad(leaf: MiniTensor, g) -> None:\n"
        '    """Set leaf.grad = g if None else leaf.grad + g. Mutates leaf; no return."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- first-touch: leaf.grad is None → set to g ---\n"
        "leaf = MiniTensor(t.zeros(3), requires_grad=True)\n"
        "assert leaf.grad is None, 'precondition'\n"
        "g = t.tensor([1.0, 2.0, 3.0])\n"
        "ret = accumulate_grad(leaf, g)\n"
        "\n"
        "assert ret is None, 'accumulate_grad must return None (mutates in place)'\n"
        "assert leaf.grad is not None\n"
        "assert t.allclose(leaf.grad, t.tensor([1.0, 2.0, 3.0])), (\n"
        "    f'first-touch must set leaf.grad = g, got {leaf.grad}'\n"
        ")\n"
        "\n"
        "# --- second-touch: leaf.grad is not None → ADD g ---\n"
        "g2 = t.tensor([10.0, 20.0, 30.0])\n"
        "accumulate_grad(leaf, g2)\n"
        "assert t.allclose(leaf.grad, t.tensor([11.0, 22.0, 33.0])), (\n"
        "    f'second-touch must ADD g, got {leaf.grad}'\n"
        ")\n"
        "\n"
        "# --- many touches: sum-like accumulation across an iteration ---\n"
        "leaf2 = MiniTensor(t.zeros(4), requires_grad=True)\n"
        "for i in range(5):\n"
        "    accumulate_grad(leaf2, t.ones(4))\n"
        "assert t.allclose(leaf2.grad, t.full((4,), 5.0)), (\n"
        "    f'5x ones should accumulate to 5: got {leaf2.grad}'\n"
        ")\n"
        "\n"
        "# --- rebind semantics: external reference to OLD grad is not mutated ---\n"
        "# This is the critical safety property — the optimizer's reference to\n"
        "# leaf.grad must NOT silently change underneath it.\n"
        "leaf3 = MiniTensor(t.zeros(3), requires_grad=True)\n"
        "accumulate_grad(leaf3, t.tensor([1.0, 2.0, 3.0]))\n"
        "old_ref = leaf3.grad\n"
        "old_ref_clone = old_ref.clone()\n"
        "accumulate_grad(leaf3, t.tensor([10.0, 20.0, 30.0]))\n"
        "assert t.allclose(old_ref, old_ref_clone), (\n"
        "    'externally-held reference to leaf.grad must NOT be mutated in place — '\n"
        '    \"use `leaf.grad = leaf.grad + g`, NOT `leaf.grad += g`\"\n'
        ")\n"
        "assert leaf3.grad is not old_ref, (\n"
        "    'leaf.grad must REBIND to a new tensor, not mutate the old one'\n"
        ")\n"
        "\n"
        "# --- the canonical `y = w * w` use case ---\n"
        "# w appears at argnum 0 AND argnum 1 of multiply → reverse visits w twice.\n"
        "# total dL/dw = dL/dy * 2w; accumulating from both paths gives the right total.\n"
        "w = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "# Path 1: contribution from arg-0 of multiply (= grad_out * w = 1 * 3 = 3)\n"
        "accumulate_grad(w, t.tensor([3.0]))\n"
        "# Path 2: contribution from arg-1 of multiply (= grad_out * w = 1 * 3 = 3)\n"
        "accumulate_grad(w, t.tensor([3.0]))\n"
        "# Total should be 6 — same as torch.autograd would give for d(w^2)/dw = 2w = 6\n"
        "assert t.allclose(w.grad, t.tensor([6.0])), (\n"
        "    f'y = w * w → dL/dw should be 2w = 6 via 2-path accumulation, got {w.grad}'\n"
        ")\n"
        "# Cross-check against torch.autograd for confidence.\n"
        "w_ref = t.tensor([3.0], requires_grad=True)\n"
        "y = w_ref * w_ref\n"
        "y.sum().backward()\n"
        "assert t.allclose(w.grad, w_ref.grad), (\n"
        "    f'accumulation must match torch.autograd: ours={w.grad}, ref={w_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def accumulate_grad(leaf: MiniTensor, g) -> None:\n"
        "    if leaf.grad is None:\n"
        "        # First-touch: set directly (no allocation for an initial zero).\n"
        "        leaf.grad = g\n"
        "    else:\n"
        "        # Subsequent touches: REBIND (NOT in-place +=).\n"
        "        # Rebinding leaves any externally-held reference to the old grad\n"
        "        # untouched — important if the optimizer is holding the grad.\n"
        "        leaf.grad = leaf.grad + g"
    ),
    solution_notes=(
        "**Why `+`, not `+=`.** `leaf.grad += g` mutates the existing "
        "grad tensor in place. If anything else holds a reference to "
        "that tensor (e.g. `optimizer.step()` snapshotted `p.grad` "
        "before the accumulate ran), the snapshot silently changes "
        "underneath. Using `+` rebinds `leaf.grad` to a fresh tensor, "
        "leaving the old one alone.\n\n"
        "**Why first-touch sets, not adds-to-zeros.** Skips an "
        "unnecessary `t.zeros_like(g)` allocation on every leaf's first "
        "visit. Across a model with thousands of parameters, the "
        "allocation cost matters.\n\n"
        "**Why `zero_grad()` exists.** Because `accumulate_grad` ALWAYS "
        "adds (never overwrites), the previous step's gradient stays in "
        "`leaf.grad` forever unless explicitly cleared. This is why "
        "PyTorch's training loop has the canonical `optimizer.zero_"
        "grad()` line — it sets every `p.grad = None` (or zeros it), so "
        "the next backward starts from a clean state.\n\n"
        "**The shared-parent case is the whole reason for this design.** "
        "`y = w * w` has `w` as parent twice. `y = a * b` then `z = y + "
        "b` has `b` as parent of two different intermediates. Real "
        "models share parameters across layers (e.g. weight-tied "
        "embedding and unembedding). Each path contributes a separate "
        "gradient; the total derivative is the sum."
    ),
)


# =========================================================================
# atom: sum-and-broadcast-duality  (1 exercise)
# =========================================================================

SPEC_SUM_BROADCAST = _spec(
    atom_id="sum-and-broadcast-duality",
    subtopic="Backprop: sum/broadcast duality",
    recap=RECAP_SUM_BROADCAST,
    ex_idx=1,
    ex_title="sum_back and broadcast_back as dual ops",
    slug="sum-back-and-broadcast-back-dual-ops",
    bloom="Apply",
    difficulty_num=4,
    keywords=["sum", "broadcast", "duality", "back-fn", "keepdim", "unsqueeze"],
    kcs=["sum-and-broadcast-duality", "unbroadcast-pattern"],
    lo=(
        "Apply the sum/broadcast duality by writing sum_back (re-insert "
        "axis + expand) and broadcast_back (sum out expanded axes), "
        "demonstrating they are transposes of one another."
    ),
    prompt_body=(
        "Implement TWO dual back fns:\n\n"
        "**1. `sum_back(grad_out, out, x, dim, keepdim=False)`** — "
        "backward for `out = x.sum(dim=dim, keepdim=keepdim)`.\n"
        "   - If `keepdim=False`, `out` lost the axis at `dim`; "
        "re-insert it: `grad_out = grad_out.unsqueeze(dim)`.\n"
        "   - Then broadcast to `x.shape`: "
        "`grad_x = grad_out.expand_as(x).clone()`.\n"
        "   - (The `.clone()` matters — `expand_as` produces a view "
        "with stride 0, and downstream `+=` accumulations on views "
        "with overlapping memory misbehave. `.clone()` materializes a "
        "fresh contiguous tensor.)\n\n"
        "**2. `broadcast_back(grad_out, out, x)`** — backward for "
        "`out = x.broadcast_to(out.shape)`. This is the `unbroadcast` "
        "pattern: sum out the axes that were expanded so `grad_x.shape "
        "== x.shape`.\n"
        "   - Step A: while `grad_out.ndim > x.ndim`: `grad_out = "
        "grad_out.sum(dim=0)`.\n"
        "   - Step B: for each axis `i` in `x.shape` where `x.shape[i] "
        "== 1` and `grad_out.shape[i] != 1`: `grad_out = grad_out.sum("
        "dim=i, keepdim=True)`.\n\n"
        "**Why this is the duality.** `sum` is a linear map (a matrix "
        "of ones along the summed axis). Its transpose is `broadcast` "
        "(ones the other way). Backprop sends gradients through the "
        "TRANSPOSE of the forward linear map. So `sum_back` ≈ "
        "broadcast, and `broadcast_back` ≈ sum. Same physical op, dual "
        "roles.\n\n"
        "Inputs are plain `torch.Tensor`. No autograd. Return tensors "
        "with the right shape per back-fn convention."
    ),
    stub=(
        "def sum_back(grad_out: Tensor, out: Tensor, x: Tensor, dim: int, keepdim: bool = False) -> Tensor:\n"
        '    """Backward of x.sum(dim, keepdim). Broadcast grad_out to x.shape."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def broadcast_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Backward of x.broadcast_to(out.shape). Sum out expanded axes."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- sum_back, keepdim=False, dim=1 ---\n"
        "x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "out = x.sum(dim=1)  # shape (3,)\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "g = sum_back(grad_out, out, x, dim=1, keepdim=False)\n"
        "assert g.shape == x.shape, f'sum_back shape: {g.shape}'\n"
        "assert t.allclose(g, t.ones(3, 4)), f'sum_back value (all ones): {g}'\n"
        "\n"
        "# --- sum_back, keepdim=False, dim=0 ---\n"
        "out = x.sum(dim=0)  # shape (4,)\n"
        "grad_out = t.tensor([2.0, 3.0, 5.0, 7.0])\n"
        "g = sum_back(grad_out, out, x, dim=0, keepdim=False)\n"
        "assert g.shape == (3, 4)\n"
        "# each row should be a copy of grad_out (broadcast back across the summed dim)\n"
        "assert t.allclose(g[0], grad_out)\n"
        "assert t.allclose(g[1], grad_out)\n"
        "assert t.allclose(g[2], grad_out)\n"
        "\n"
        "# --- sum_back, keepdim=True ---\n"
        "out = x.sum(dim=1, keepdim=True)  # shape (3, 1)\n"
        "grad_out = t.tensor([[1.0], [2.0], [3.0]])\n"
        "g = sum_back(grad_out, out, x, dim=1, keepdim=True)\n"
        "assert g.shape == (3, 4)\n"
        "# each row should be filled with that row's grad value\n"
        "assert t.allclose(g[0], t.full((4,), 1.0))\n"
        "assert t.allclose(g[1], t.full((4,), 2.0))\n"
        "assert t.allclose(g[2], t.full((4,), 3.0))\n"
        "\n"
        "# --- sum_back vs torch.autograd witness ---\n"
        "# Make x_ref a true leaf with requires_grad — reshape produces a view, not a leaf,\n"
        "# so we clone+detach+requires_grad_ to get a fresh leaf with the right shape.\n"
        "x_ref = t.arange(12, dtype=t.float32).reshape(3, 4).clone().detach().requires_grad_(True)\n"
        "y = x_ref.sum(dim=1).sum()\n"
        "y.backward()\n"
        "g_ours = sum_back(t.ones(3), x_ref.detach().sum(dim=1), x_ref.detach(), dim=1, keepdim=False)\n"
        "assert t.allclose(g_ours, x_ref.grad, atol=1e-6), (\n"
        "    f'sum_back disagrees with autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")\n"
        "\n"
        "# --- broadcast_back: dual case A (leading axes added) ---\n"
        "# forward: x.broadcast_to((2, 3, 4)) — leading dim 2 added\n"
        "x_orig = t.zeros(3, 4)\n"
        "grad_out = t.ones(2, 3, 4)\n"
        "g = broadcast_back(grad_out, t.zeros(2, 3, 4), x_orig)\n"
        "assert g.shape == (3, 4), f'broadcast_back leading-axes shape: {g.shape}'\n"
        "assert t.allclose(g, t.full((3, 4), 2.0)), f'sum-out failed: {g}'\n"
        "\n"
        "# --- broadcast_back: dual case B (size-1 axis expanded) ---\n"
        "# forward: x.shape=(1,4) broadcast to (3,4)\n"
        "x_orig = t.zeros(1, 4)\n"
        "grad_out = t.ones(3, 4)\n"
        "g = broadcast_back(grad_out, t.zeros(3, 4), x_orig)\n"
        "assert g.shape == (1, 4), f'broadcast_back size-1 shape: {g.shape}'\n"
        "assert t.allclose(g, t.full((1, 4), 3.0))\n"
        "\n"
        "# --- broadcast_back: combined A + B ---\n"
        "x_orig = t.zeros(1, 4)\n"
        "grad_out = t.ones(2, 3, 4)\n"
        "g = broadcast_back(grad_out, t.zeros(2, 3, 4), x_orig)\n"
        "assert g.shape == (1, 4)\n"
        "assert t.allclose(g, t.full((1, 4), 6.0)), f'A+B value (2*3=6): {g}'\n"
        "\n"
        "# --- DUALITY check: sum_back ∘ broadcast_back relationship ---\n"
        "# If we sum, then sum-back, the result is a constant copy across the summed axis.\n"
        "# If we then sum that axis again, we should get back grad_out * shape[dim].\n"
        "x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "grad_out = t.tensor([1.0, 2.0, 3.0])\n"
        "g_back = sum_back(grad_out, x.sum(dim=1), x, dim=1, keepdim=False)  # (3, 4)\n"
        "# Reduce the broadcast-back axis: should recover 4 * grad_out\n"
        "assert t.allclose(g_back.sum(dim=1), 4 * grad_out), (\n"
        "    f'duality round-trip wrong: {g_back.sum(dim=1)} vs {4 * grad_out}'\n"
        ")"
    ),
    solution_body=(
        "def sum_back(grad_out: Tensor, out: Tensor, x: Tensor, dim: int, keepdim: bool = False) -> Tensor:\n"
        "    # If keepdim=False, the forward sum DROPPED the axis at `dim`.\n"
        "    # Re-insert it (size 1) so we can expand cleanly back to x.shape.\n"
        "    if not keepdim:\n"
        "        grad_out = grad_out.unsqueeze(dim)\n"
        "    # Broadcast back to x.shape. .clone() materializes a contiguous tensor\n"
        "    # (expand_as gives a stride-0 view that misbehaves under accumulation).\n"
        "    return grad_out.expand_as(x).clone()\n"
        "\n"
        "\n"
        "def broadcast_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # Step A: peel leading axes that broadcasting added.\n"
        "    while grad_out.ndim > x.ndim:\n"
        "        grad_out = grad_out.sum(dim=0)\n"
        "    # Step B: collapse size-1 axes that were expanded; keepdim preserves shape match.\n"
        "    for i, size in enumerate(x.shape):\n"
        "        if size == 1 and grad_out.shape[i] != 1:\n"
        "            grad_out = grad_out.sum(dim=i, keepdim=True)\n"
        "    return grad_out"
    ),
    solution_notes=(
        "**Why `unsqueeze + expand_as` and not just `expand_as`.** "
        "`expand_as` requires the source ndim to match the target ndim "
        "(or be smaller with appropriate trailing alignment). When "
        "`keepdim=False`, `grad_out` has one fewer dim than `x` at "
        "position `dim`. The `unsqueeze(dim)` puts the missing size-1 "
        "axis back so `expand_as` can do its work.\n\n"
        "**Why `.clone()` after `expand_as`.** `expand_as` returns a "
        "VIEW with stride 0 along the expanded axis — every position in "
        "that axis points to the SAME memory cell. Doing `grad += "
        "something` on that view writes to the same cell N times, "
        "accumulating wrong. `.clone()` materializes a fresh contiguous "
        "tensor where each position has its own storage. Subtle bug — "
        "easy to miss until accumulation tests fail.\n\n"
        "**The transpose intuition.** Think of `sum(dim=k)` as left-"
        "multiplying by a row vector of ones. Its transpose is a column "
        "vector of ones, which left-multiplied broadcasts. Backprop "
        "always sends gradients through the transpose of the forward "
        "linear op. So:\n"
        "- forward = sum → backward = broadcast (insert + expand)\n"
        "- forward = broadcast → backward = sum (collapse axes)\n\n"
        "**Where this shows up in real models.** Every `(B, *)` mean "
        "/ sum reduction in a loss function uses `sum_back` on the "
        "reverse pass. Every bias add (`out = x + bias` where "
        "`bias.shape == (C,)` and `x.shape == (B, C)`) broadcasts the "
        "bias forward and needs `broadcast_back` to compute "
        "`dL/dbias`. The duality is what makes broadcasting transparent "
        "for the model author."
    ),
)


# =========================================================================
# emit
# =========================================================================

ALL_SPECS = [
    SPEC_BOX,
    SPEC_UNBOX,
    SPEC_GET_CHILDREN,
    SPEC_COERCE,
    SPEC_INPLACE,
    SPEC_PARAMETER,
    SPEC_ACCUMULATE,
    SPEC_SUM_BROADCAST,
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
            # The shared preamble defines MiniTensor / Recipe / grad_tracking_enabled
            # — every spec's solution + test_body assumes these are in scope.
            exec(_AUTOGRAD_PREAMBLE, ns)
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
    print(f"[autograd-pt3 batch4] Verifying {len(ALL_SPECS)} specs against torch backend...")
    _verify_all(ALL_SPECS)

    print(f"\n[autograd-pt3 batch4] All verified — emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[autograd-pt3 batch4] {len(ALL_SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
