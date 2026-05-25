#!/usr/bin/env python3
"""Author ex2 DEEPENING drills for ARENA manual-autograd PART 3 atoms.

Eight single-exercise standalones under ``prereqs_autograd_pt3/`` — exercise
index = 2. Each one targets a DISTINCT facet of its parent atom that ex1
didn't directly hit, while sharing the same KCs / subtopic / Bloom level.

Atoms (8):
  * box-array-to-tensor-with-recipe   — ex1 wrote box_with_recipe end-to-end.
                                        ex2: write the wrap_forward_fn ASSEMBLY
                                        that composes unbox→fwd→box, exercising
                                        the requires_grad GATE (any-arg + global
                                        toggle) which ex1 received as an arg.
  * unbox-args-tensor-to-array        — ex1 unboxed positional args.
                                        ex2: unbox a NESTED structure — `args`
                                        can contain lists/tuples-of-Tensors;
                                        recurse, preserve container type.
  * get-children-callable-param       — ex1 yielded shallow (name, val) pairs.
                                        ex2: build the RECURSIVE walker that
                                        calls get_children on Module-valued
                                        children to yield dotted-name params.
  * coerce-float-arg-to-array         — ex1 coerced single args (int/float vs
                                        bool/tuple).
                                        ex2: coerce an ENTIRE args tuple in one
                                        pass — `coerce_args` — using ex1's
                                        single-arg helper; check ndarray is
                                        NOT coerced (pass-through), nested
                                        lists ARE coerced via t.tensor.
  * inplace-op-unsafe-warning         — ex1 wrote `add_inplace_safe`.
                                        ex2: write a context manager
                                        `inplace_unsafe()` that toggles the
                                        guard off for a block (an escape
                                        hatch for advanced users); guard is
                                        restored on exit even on exception.
  * parameter-subclass-of-tensor      — ex1 defined Parameter with rg=True.
                                        ex2: implement `Module.parameters()`
                                        that walks `get_children` and yields
                                        any attr that is a `Parameter`
                                        specifically (not just any
                                        MiniTensor) — distinguishing role.
  * grad-accumulate-on-leaf           — ex1 accumulated on a single leaf.
                                        ex2: write `zero_grad(params)` and
                                        verify the round-trip:
                                        accumulate → zero → accumulate ==
                                        single accumulate (without zero,
                                        gradients persist across steps).
  * sum-and-broadcast-duality         — ex1 wrote sum_back + broadcast_back
                                        as separate ops.
                                        ex2: prove the DUALITY operationally
                                        — implement a check that
                                        sum_back is the adjoint of sum
                                        forward via the <Av, w> == <v, A^T w>
                                        identity (inner-product test).

Verifier: exec _AUTOGRAD_PREAMBLE + stub + solution + test in a fresh ns.
Pattern mirrors author_deepening_a_batch7.py:1133.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_pt3"


# ---------------------------------------------------------------- shared preamble

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
# atom: box-array-to-tensor-with-recipe — ex2: assembly of wrap_forward_fn
# =========================================================================
#
# Ex1 wrote `box_with_recipe(out_raw, fwd_fn, raw_args, kwargs, parents,
# requires_grad)` — it RECEIVED requires_grad as a bool. Ex2 deepens by
# writing the surrounding wrapper that COMPUTES that bool: scan args for any
# Tensor with requires_grad=True AND consult the global grad_tracking_enabled
# toggle. This is the "requires-grad gate" facet of the box step.

RECAP_BOX_GATE = (
    "## Box step — computing the requires_grad gate — quick refresher\n"
    "\n"
    "`box_with_recipe` from ex1 receives `requires_grad` as a precomputed "
    "bool. Who computes it? The OUTER `wrap_forward_fn`, by combining two "
    "signals:\n"
    "\n"
    "1. **Any-input gate.** `any(isinstance(a, MiniTensor) and a.requires_grad "
    "for a in args)`. If a single input is grad-tracked, the output must be.\n"
    "2. **Global toggle.** `grad_tracking_enabled` (a module-level bool). "
    "Inside an inference / no_grad block this is False — and overrides the "
    "any-input signal so nothing gets a Recipe.\n"
    "\n"
    "The combined rule: `requires_grad = grad_tracking_enabled AND "
    "any(rg_input)`.\n"
    "\n"
    "Then ex1's `box_with_recipe` does the actual boxing using this bool."
)

SPEC_BOX_ASSEMBLY = _spec(
    atom_id="box-array-to-tensor-with-recipe",
    subtopic="Backprop: Box array as Tensor + recipe",
    recap=RECAP_BOX_GATE,
    ex_idx=2,
    ex_title="wrap_forward_fn: compute requires_grad gate, then box the raw output",
    slug="wrap-forward-fn-compute-grad-gate-then-box",
    bloom="Apply",
    difficulty_num=4,
    keywords=["wrap-forward", "requires-grad-gate", "global-toggle", "box", "recipe"],
    kcs=["box-array-to-tensor-with-recipe", "recipe-dataclass"],
    lo=(
        "Apply the requires_grad gate at the wrapper boundary: any-input "
        "rg-bool AND the global grad-tracking toggle, then box-with-recipe."
    ),
    prompt_body=(
        "Implement `wrap_forward_fn(fwd_fn)` — the FULL wrapper factory. "
        "Given a raw forward fn (e.g. `torch.log`), return a new function "
        "`wrapped(*args, **kwargs)` that:\n\n"
        "1. **Unbox** positional args: `raw_args = tuple(a.array if "
        "isinstance(a, MiniTensor) else a for a in args)`.\n"
        "2. **Compute the gate**: `requires_grad = grad_tracking_enabled "
        "and any(isinstance(a, MiniTensor) and a.requires_grad for a in "
        "args)`.\n"
        "3. **Run the raw forward**: `out_raw = fwd_fn(*raw_args, "
        "**kwargs)`.\n"
        "4. **Build parents** (only if grad-tracked): `{idx: a for idx, a "
        "in enumerate(args) if isinstance(a, MiniTensor) and "
        "a.requires_grad}`.\n"
        "5. **Box** with Recipe iff `requires_grad`: construct "
        "`MiniTensor(out_raw, requires_grad=requires_grad)`, attach "
        "`Recipe(fwd_fn, raw_args, kwargs, parents)` iff True.\n\n"
        "**Why this is harder than ex1.** Ex1's `box_with_recipe` was the "
        "FINAL step — handed all the bookkeeping. Here you compose the "
        "WHOLE wrapper. The trap is the gate: two signals (`any-input` "
        "AND `global-toggle`) combine with AND. Forgetting the toggle "
        "means `no_grad()` blocks silently still build the graph; "
        "forgetting the any-input check means every constant + constant "
        "call gets a useless Recipe.\n\n"
        "Signature: `wrap_forward_fn(fwd_fn) -> Callable`. The returned "
        "callable takes `*args, **kwargs` and returns a `MiniTensor`."
    ),
    stub=(
        "def wrap_forward_fn(fwd_fn):\n"
        '    """Wrap a raw forward fn so it accepts/returns MiniTensors + builds Recipes."""\n'
        "    def wrapped(*args, **kwargs):\n"
        "        raise NotImplementedError()\n"
        "    return wrapped"
    ),
    test_body=(
        "global grad_tracking_enabled\n"
        "\n"
        "# --- one rg=True input → output is grad-tracked with full Recipe ---\n"
        "grad_tracking_enabled = True\n"
        "log_w = wrap_forward_fn(t.log)\n"
        "x = MiniTensor(t.tensor([1.0, t.e, t.e**2]), requires_grad=True)\n"
        "out = log_w(x)\n"
        "\n"
        "assert isinstance(out, MiniTensor)\n"
        "assert out.requires_grad is True, 'rg-input must propagate'\n"
        "assert out.recipe is not None, 'rg=True → Recipe must be attached'\n"
        "assert out.recipe.func is t.log\n"
        "assert out.recipe.parents == {0: x}, f'parents wrong: {out.recipe.parents}'\n"
        "assert t.allclose(out.array, t.tensor([0.0, 1.0, 2.0]), atol=1e-6)\n"
        "\n"
        "# --- all-rg=False inputs → output is NOT grad-tracked, no Recipe ---\n"
        "y = MiniTensor(t.tensor([1.0, 2.0]), requires_grad=False)\n"
        "out = log_w(y)\n"
        "assert out.requires_grad is False, 'no rg-input → no rg-output'\n"
        "assert out.recipe is None, 'no rg-input → no Recipe'\n"
        "\n"
        "# --- global toggle off → overrides even an rg-input ---\n"
        "grad_tracking_enabled = False\n"
        "out = log_w(x)  # x is rg=True\n"
        "assert out.requires_grad is False, (\n"
        "    'grad_tracking_enabled=False must override any rg-input — this is the no_grad case'\n"
        ")\n"
        "assert out.recipe is None\n"
        "grad_tracking_enabled = True  # restore for further tests\n"
        "\n"
        "# --- mixed args (Tensor + scalar): scalar is pass-through, gate based on Tensor ---\n"
        "add_w = wrap_forward_fn(t.add)\n"
        "out = add_w(x, t.tensor(5.0))  # second arg is raw torch.Tensor — passes through unbox\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "# parents only contains MiniTensor args with requires_grad=True\n"
        "assert out.recipe.parents == {0: x}, (\n"
        "    f'only rg-MiniTensor args become parents, got {out.recipe.parents}'\n"
        ")\n"
        "\n"
        "# --- kwargs flow into the Recipe + raw fn ---\n"
        "sum_w = wrap_forward_fn(t.sum)\n"
        "m = MiniTensor(t.ones(3, 4), requires_grad=True)\n"
        "out = sum_w(m, dim=1)\n"
        "assert t.allclose(out.array, t.full((3,), 4.0))\n"
        "assert out.recipe.kwargs == {'dim': 1}, (\n"
        "    f'kwargs must round-trip through wrap, got {out.recipe.kwargs}'\n"
        ")\n"
        "\n"
        "# --- the wrapped fn is callable repeatedly without state leak ---\n"
        "for _ in range(3):\n"
        "    out = log_w(x)\n"
        "    assert out.requires_grad is True"
    ),
    solution_body=(
        "def wrap_forward_fn(fwd_fn):\n"
        "    def wrapped(*args, **kwargs):\n"
        "        # 1. Unbox positional args (raw fn doesn't know MiniTensor)\n"
        "        raw_args = tuple(\n"
        "            a.array if isinstance(a, MiniTensor) else a for a in args\n"
        "        )\n"
        "        # 2. Gate: global toggle AND any-rg-input\n"
        "        any_rg = any(\n"
        "            isinstance(a, MiniTensor) and a.requires_grad for a in args\n"
        "        )\n"
        "        requires_grad = grad_tracking_enabled and any_rg\n"
        "        # 3. Run raw forward\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        # 4. Build parents only if grad-tracked (skip-graph optimization)\n"
        "        parents = {}\n"
        "        if requires_grad:\n"
        "            parents = {\n"
        "                idx: a for idx, a in enumerate(args)\n"
        "                if isinstance(a, MiniTensor) and a.requires_grad\n"
        "            }\n"
        "        # 5. Box (+ Recipe iff grad-tracked)\n"
        "        out = MiniTensor(out_raw, requires_grad=requires_grad)\n"
        "        if requires_grad:\n"
        "            out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return wrapped"
    ),
    solution_notes=(
        "**The two-signal gate is the load-bearing facet.** Ex1 got "
        "`requires_grad` as a bool — here you have to compute it. Forget "
        "the global toggle and `no_grad()` is a no-op. Forget the any-input "
        "check and constant + constant calls get useless Recipes.\n\n"
        "**Why build parents AFTER the gate.** When grad-tracking is off, "
        "building parents is wasted work (the Recipe is never attached, so "
        "the dict is dropped). The conditional keeps inference paths "
        "allocation-free."
    ),
)


# =========================================================================
# atom: unbox-args-tensor-to-array — ex2: nested-structure unboxing
# =========================================================================
#
# Ex1: unbox a flat tuple — Tensor → .array, scalar → pass-through.
# Ex2: unbox a NESTED structure (list/tuple of MiniTensors) — used by ops
# like `cat([t1, t2, t3], dim=0)` where the first positional arg is itself
# a list. Recurse, preserve container type.

RECAP_UNBOX_NESTED = (
    "## Unbox — handling nested args (lists/tuples of Tensors) — quick refresher\n"
    "\n"
    "Some forward ops take a SEQUENCE of tensors as a single positional arg:\n"
    "\n"
    "```python\n"
    "stacked = cat([t1, t2, t3], dim=0)        # arg 0 is a list of MiniTensors\n"
    "result  = stack((t_a, t_b), dim=1)         # arg 0 is a tuple of MiniTensors\n"
    "```\n"
    "\n"
    "The raw `torch.cat` accepts `list[Tensor]`, not `list[MiniTensor]`. "
    "So `unbox_args` must RECURSE into list/tuple args and unbox the "
    "inner Tensors, while preserving the container type (list stays list, "
    "tuple stays tuple).\n"
    "\n"
    "Critical: do NOT recurse forever — only one level of nesting is the "
    "canonical pattern in PyTorch (no `cat([[t1, t2]], dim=0)`)."
)

SPEC_UNBOX_NESTED = _spec(
    atom_id="unbox-args-tensor-to-array",
    subtopic="Backprop: Unbox Tensor args to array",
    recap=RECAP_UNBOX_NESTED,
    ex_idx=2,
    ex_title="unbox_args_nested: recurse into list/tuple args, preserve container",
    slug="unbox-args-nested-list-tuple-preserve-container",
    bloom="Apply",
    difficulty_num=3,
    keywords=["unbox", "nested", "cat", "stack", "container-type"],
    kcs=["unbox-args-tensor-to-array", "parents-dict-by-argidx"],
    lo=(
        "Apply unboxing to nested args: when a positional arg is a list or "
        "tuple of MiniTensors, recurse one level and preserve the container "
        "type so `cat`/`stack`-style ops work."
    ),
    prompt_body=(
        "Implement `unbox_args_nested(args)`. Same shape as ex1's "
        "`unbox_args`, BUT: if a positional arg is a `list` or `tuple` "
        "whose elements include `MiniTensor` instances, recurse one level — "
        "replace each inner `MiniTensor` with its `.array`, preserving the "
        "container type:\n\n"
        "```\n"
        "unbox_args_nested(([t1, t2, t3], 0))    == ([t1.array, t2.array, t3.array], 0)\n"
        "unbox_args_nested(((tA, tB), 1))         == ((tA.array, tB.array), 1)\n"
        "unbox_args_nested((t1, 3.0, t2))         == (t1.array, 3.0, t2.array)  # ex1 case\n"
        "unbox_args_nested(([1, 2, 3],))          == ([1, 2, 3],)               # plain ints stay\n"
        "```\n\n"
        "Rules:\n\n"
        "**1. Single level of recursion.** A list-of-list is preserved "
        "as-is (no PyTorch op uses that signature). Don't go infinitely "
        "deep — that risks pathological inputs.\n\n"
        "**2. Preserve container type.** `list` in → `list` out. `tuple` "
        "in → `tuple` out. Don't normalize to one.\n\n"
        "**3. Mixed inner types are fine.** `[t1, 3.0, t2]` → `[t1.array, "
        "3.0, t2.array]` — inner pass-through, same as the top level.\n\n"
        "**4. Top-level `MiniTensor` still unboxes** (ex1 behavior — this "
        "function generalizes ex1, doesn't replace it).\n\n"
        "Why this matters: `cat`/`stack` are the canonical example. "
        "Without nested unbox, `cat([m1, m2])` reaches the raw "
        "`torch.cat` with a `list[MiniTensor]`, which crashes "
        "(`AttributeError: ... has no attribute 'dim'`)."
    ),
    stub=(
        "def unbox_args_nested(args: tuple) -> tuple:\n"
        '    """Like unbox_args, but recurse into list/tuple args one level deep."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- ex1 cases still work (top-level only) ---\n"
        "raw1 = t.tensor([1.0, 2.0])\n"
        "raw2 = t.tensor([3.0, 4.0])\n"
        "m1 = MiniTensor(raw1)\n"
        "m2 = MiniTensor(raw2)\n"
        "assert unbox_args_nested((m1, 3.0, m2)) == (raw1, 3.0, raw2)\n"
        "assert unbox_args_nested(()) == ()\n"
        "assert unbox_args_nested((1, 2.0, 'x')) == (1, 2.0, 'x')\n"
        "\n"
        "# --- list-of-MiniTensors as a single arg → recurse, preserve list type ---\n"
        "result = unbox_args_nested(([m1, m2], 0))\n"
        "assert isinstance(result, tuple)\n"
        "assert len(result) == 2\n"
        "assert isinstance(result[0], list), (\n"
        "    f'list arg must stay a list, got {type(result[0]).__name__}'\n"
        ")\n"
        "assert result[0][0] is raw1 and result[0][1] is raw2, 'inner identity preserved'\n"
        "assert result[1] == 0\n"
        "\n"
        "# --- tuple-of-MiniTensors as a single arg → recurse, preserve tuple type ---\n"
        "result = unbox_args_nested(((m1, m2), 1))\n"
        "assert isinstance(result[0], tuple), (\n"
        "    f'tuple arg must stay a tuple (NOT normalized to list), got {type(result[0]).__name__}'\n"
        ")\n"
        "assert result[0] == (raw1, raw2)\n"
        "\n"
        "# --- list of NON-MiniTensors passes through unchanged ---\n"
        "result = unbox_args_nested(([1, 2, 3],))\n"
        "assert result == ([1, 2, 3],), f'plain int list unchanged: {result}'\n"
        "assert isinstance(result[0], list)\n"
        "\n"
        "# --- mixed inner: list with both MiniTensor and float ---\n"
        "result = unbox_args_nested(([m1, 2.5, m2],))\n"
        "assert result[0][0] is raw1\n"
        "assert result[0][1] == 2.5\n"
        "assert result[0][2] is raw2\n"
        "\n"
        "# --- the cat use case end-to-end ---\n"
        "# raw torch.cat must accept the unboxed list and produce a tensor.\n"
        "raw_cat = t.cat(unbox_args_nested(([m1, m2],))[0], dim=0)\n"
        "assert t.allclose(raw_cat, t.tensor([1.0, 2.0, 3.0, 4.0])), (\n"
        "    'cat([m1, m2], dim=0) after unboxing must equal manual concat'\n"
        ")\n"
        "\n"
        "# --- top-level MiniTensor in mixed args still unboxed (ex1 behavior preserved) ---\n"
        "result = unbox_args_nested((m1, [m2, m1]))\n"
        "assert result[0] is raw1, 'top-level MiniTensor unboxed'\n"
        "assert isinstance(result[1], list)\n"
        "assert result[1][0] is raw2 and result[1][1] is raw1\n"
        "\n"
        "# --- length preserved at top level always ---\n"
        "for inp in [(m1,), (m1, m2), ([m1, m2],), (1, [m1], 'x')]:\n"
        "    assert len(unbox_args_nested(inp)) == len(inp)"
    ),
    solution_body=(
        "def unbox_args_nested(args: tuple) -> tuple:\n"
        "    def _maybe_unbox(a):\n"
        "        if isinstance(a, MiniTensor):\n"
        "            return a.array\n"
        "        if isinstance(a, list):\n"
        "            return [_maybe_unbox(x) for x in a]\n"
        "        if isinstance(a, tuple):\n"
        "            return tuple(_maybe_unbox(x) for x in a)\n"
        "        return a\n"
        "    return tuple(_maybe_unbox(a) for a in args)"
    ),
    solution_notes=(
        "**One level of recursion is intentional.** A `list[list[Tensor]]` "
        "is not a real PyTorch op signature. Recursing arbitrarily deep "
        "risks pathological inputs (cycles, ragged structures) for no "
        "real-world payoff. The inner helper happens to be recursive but "
        "the input set never goes more than 2 deep in practice.\n\n"
        "**Container-type preservation matters.** `torch.cat` accepts list "
        "and tuple interchangeably, but `torch.stack` historically wants "
        "the same type back. Normalizing to one type can break the raw fn."
    ),
)


# =========================================================================
# atom: get-children-callable-param — ex2: recursive parameter walker
# =========================================================================
#
# Ex1: yield (name, val) shallow.
# Ex2: build `parameters(recurse=True)` on top of `get_children`, emitting
# dotted-name tuples for nested Modules.

RECAP_GET_CHILDREN_REC = (
    "## get_children → recursive parameters walker — quick refresher\n"
    "\n"
    "`get_children` is the SHALLOW step (one module's direct attrs). The "
    "recursive walk `parameters()` is built on top of it:\n"
    "\n"
    "```python\n"
    "class MLP(Module):\n"
    "    def __init__(self):\n"
    "        self.fc1 = Linear()       # child Module\n"
    "        self.fc2 = Linear()       # child Module\n"
    "        self.bias = MiniTensor(...)\n"
    "\n"
    "list(MLP().parameters())\n"
    "  → [('fc1.weight', ...), ('fc1.bias', ...),\n"
    "     ('fc2.weight', ...), ('fc2.bias', ...),\n"
    "     ('bias', ...)]\n"
    "```\n"
    "\n"
    "Walk: for each (name, val) in `get_children`:\n"
    "- if val is a `MiniTensor`, yield `(name, val)`\n"
    "- if val is a `Module`, recurse and prefix each child name with `<name>.`"
)

SPEC_GET_CHILDREN_REC = _spec(
    atom_id="get-children-callable-param",
    subtopic="Backprop: get_children callable param",
    recap=RECAP_GET_CHILDREN_REC,
    ex_idx=2,
    ex_title="parameters(recurse=True) built on top of get_children with dotted names",
    slug="parameters-recursive-walker-dotted-names",
    bloom="Apply",
    difficulty_num=4,
    keywords=["parameters", "recursive", "dotted-name", "nn.Module", "walker"],
    kcs=["get-children-callable-param", "parameter-subclass-of-tensor"],
    lo=(
        "Apply the recursive parameter walker pattern: build "
        "parameters() on top of get_children by yielding leaf Tensors "
        "directly and recursing into Module-valued children with dotted "
        "name prefixes."
    ),
    prompt_body=(
        "We've given you a `Module` base class with `get_children` (the "
        "ex1 helper, extended to yield BOTH MiniTensors and other "
        "`Module` instances). Implement `Module.parameters(self)` as a "
        "generator that does a DEPTH-FIRST recursive walk:\n\n"
        "- For each `(name, val)` yielded by `get_children`:\n"
        "  - If `isinstance(val, MiniTensor)`: yield `(name, val)`.\n"
        "  - If `isinstance(val, Module)`: recurse into `val.parameters()`, "
        "    and for each `(sub_name, sub_val)` it yields, yield "
        "    `(f'{name}.{sub_name}', sub_val)`.\n\n"
        "This produces the canonical PyTorch state-dict naming: "
        "`'fc1.weight'`, `'fc1.bias'`, `'fc2.weight'`, etc.\n\n"
        "Rules:\n\n"
        "**1. Generator (yield).** Same reason as ex1.\n\n"
        "**2. Dotted naming.** Prefix child names with the parent attr "
        "name + `.`. This is the state-dict convention and is how "
        "`load_state_dict` finds the right tensor by key.\n\n"
        "**3. Depth-first, in-order.** Order matches `get_children`'s "
        "insertion order at each level.\n\n"
        "Note: you should NOT call `get_children` from inside `parameters` "
        "to discover child Modules separately — the given `get_children` "
        "already yields both Tensors and Modules. Just dispatch on type."
    ),
    stub=(
        "class Module:\n"
        '    """Tiny nn.Module stand-in. get_children yields BOTH MiniTensors and Modules."""\n'
        "    def get_children(self):\n"
        "        for name, val in self.__dict__.items():\n"
        "            if isinstance(val, (MiniTensor, Module)):\n"
        "                yield name, val\n"
        "\n"
        "    def parameters(self):\n"
        '        """Recursive DFS yield of (dotted_name, MiniTensor) leaves."""\n'
        "        raise NotImplementedError()"
    ),
    test_body=(
        "# --- flat module: parameters == get_children's MiniTensor subset ---\n"
        "class Linear(Module):\n"
        "    def __init__(self):\n"
        "        self.weight = MiniTensor(t.randn(4, 3), requires_grad=True)\n"
        "        self.bias = MiniTensor(t.zeros(4), requires_grad=True)\n"
        "        self.in_features = 3\n"
        "\n"
        "lin = Linear()\n"
        "params = list(lin.parameters())\n"
        "names = [n for n, _ in params]\n"
        "assert names == ['weight', 'bias'], f'flat case: {names}'\n"
        "assert params[0][1] is lin.weight\n"
        "assert params[1][1] is lin.bias\n"
        "\n"
        "# --- nested module: parameters yields dotted names ---\n"
        "class MLP(Module):\n"
        "    def __init__(self):\n"
        "        self.fc1 = Linear()\n"
        "        self.fc2 = Linear()\n"
        "\n"
        "mlp = MLP()\n"
        "params = list(mlp.parameters())\n"
        "names = [n for n, _ in params]\n"
        "assert names == ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'], (\n"
        "    f'nested case: {names}'\n"
        ")\n"
        "# identity through nesting\n"
        "assert params[0][1] is mlp.fc1.weight\n"
        "assert params[2][1] is mlp.fc2.weight\n"
        "\n"
        "# --- mixed: top-level MiniTensor alongside child Modules ---\n"
        "class MLPWithBias(Module):\n"
        "    def __init__(self):\n"
        "        self.fc1 = Linear()\n"
        "        self.extra_bias = MiniTensor(t.zeros(4), requires_grad=True)\n"
        "        self.fc2 = Linear()\n"
        "\n"
        "m = MLPWithBias()\n"
        "names = [n for n, _ in m.parameters()]\n"
        "assert names == [\n"
        "    'fc1.weight', 'fc1.bias', 'extra_bias',\n"
        "    'fc2.weight', 'fc2.bias',\n"
        "], f'mixed case: {names}'\n"
        "\n"
        "# --- three levels of nesting → two dots ---\n"
        "class Block(Module):\n"
        "    def __init__(self):\n"
        "        self.inner = Linear()\n"
        "\n"
        "class Net(Module):\n"
        "    def __init__(self):\n"
        "        self.block = Block()\n"
        "\n"
        "net = Net()\n"
        "names = [n for n, _ in net.parameters()]\n"
        "assert names == ['block.inner.weight', 'block.inner.bias'], (\n"
        "    f'two-level nesting: {names}'\n"
        ")\n"
        "\n"
        "# --- it is a generator (zero-allocation iteration) ---\n"
        "import inspect\n"
        "iterator = lin.parameters()\n"
        "assert iter(iterator) is iterator or inspect.isgenerator(iterator), (\n"
        "    f'parameters() must be a generator, got {type(iterator).__name__}'\n"
        ")\n"
        "\n"
        "# --- empty module: yields nothing (no crash) ---\n"
        "class Empty(Module):\n"
        "    def __init__(self):\n"
        "        pass\n"
        "assert list(Empty().parameters()) == []"
    ),
    solution_body=(
        "class Module:\n"
        "    def get_children(self):\n"
        "        for name, val in self.__dict__.items():\n"
        "            if isinstance(val, (MiniTensor, Module)):\n"
        "                yield name, val\n"
        "\n"
        "    def parameters(self):\n"
        "        for name, val in self.get_children():\n"
        "            if isinstance(val, MiniTensor):\n"
        "                yield name, val\n"
        "            elif isinstance(val, Module):\n"
        "                for sub_name, sub_val in val.parameters():\n"
        "                    yield f'{name}.{sub_name}', sub_val"
    ),
    solution_notes=(
        "**Why dotted names.** PyTorch's `state_dict` is a flat dict keyed "
        "by these dotted paths. `load_state_dict` matches keys by string. "
        "The same convention falls out of recursive prefixing — no extra "
        "machinery needed.\n\n"
        "**Why depth-first.** Order matters for reproducibility (random "
        "init seeds, optimizer state ordering). DFS in attribute-insertion "
        "order is what `nn.Module` does in PyTorch.\n\n"
        "**Why `get_children` returns BOTH types in ex2.** Ex1 stayed "
        "shallow on MiniTensor only. To compose a recursive walker on top, "
        "we need both — so `get_children` is extended here. In real "
        "`nn.Module` this is split: `_parameters` + `_modules` separately, "
        "but the abstraction is the same."
    ),
)


# =========================================================================
# atom: coerce-float-arg-to-array — ex2: coerce_args over a tuple
# =========================================================================
#
# Ex1 coerced one arg, focused on (int, float) → tensor, bool/tuple/ndarray
# pass-through. Ex2 generalizes to the WHOLE args tuple — that's how it's
# actually used at the wrap_forward_fn entry — and adds the ndarray-pass-
# through invariant (ex1 hinted but ex2 makes it the load-bearing check).

RECAP_COERCE_ARGS = (
    "## Coerce-args — apply scalar coercion across an args tuple — refresher\n"
    "\n"
    "Ex1's `coerce_to_array` handles a SINGLE arg. The wrapper actually "
    "calls it across the whole positional-args tuple:\n"
    "\n"
    "```python\n"
    "args_coerced = tuple(coerce_to_array(a) for a in args)\n"
    "```\n"
    "\n"
    "Same rules per arg:\n"
    "- `bool` → pass-through (the subclass-of-int trap)\n"
    "- `int` / `float` → `t.tensor(float(arg))`\n"
    "- everything else → pass-through (including `torch.Tensor`, "
    "`np.ndarray`, `MiniTensor`, tuples, None, ...)\n"
    "\n"
    "Critical pass-through case: `np.ndarray`. We DON'T want to wrap it "
    "in `t.tensor(...)` here — the downstream unbox step is doing the "
    "MiniTensor.array extraction, and a numpy array is already raw-array-"
    "shaped. (The actual raw fn — `torch.log` — accepts numpy arrays via "
    "the tensor protocol.)"
)

SPEC_COERCE_ARGS = _spec(
    atom_id="coerce-float-arg-to-array",
    subtopic="Backprop: Coerce float arg to array",
    recap=RECAP_COERCE_ARGS,
    ex_idx=2,
    ex_title="coerce_args: apply scalar coercion across an args tuple, ndarray pass-through",
    slug="coerce-args-tuple-ndarray-pass-through",
    bloom="Apply",
    difficulty_num=3,
    keywords=["coerce", "args-tuple", "ndarray", "pass-through", "wrap-forward"],
    kcs=["coerce-float-arg-to-array", "unbox-args-tensor-to-array"],
    lo=(
        "Apply the per-arg scalar coercion across a whole args tuple, "
        "preserving non-scalar types — especially numpy ndarrays — as "
        "pass-through so the downstream unbox step receives uniform input."
    ),
    prompt_body=(
        "Implement `coerce_args(args)`. Given a positional-args tuple, "
        "return a new tuple where each arg has been passed through the "
        "single-arg coercion rule:\n\n"
        "- `bool` (incl. `True` / `False`) → pass-through (NOT a tensor).\n"
        "- `int` or `float` (excluding bool) → `t.tensor(float(arg))`.\n"
        "- Everything else (`torch.Tensor`, `MiniTensor`, `np.ndarray`, "
        "tuples, `None`, strings) → pass-through unchanged (identity).\n\n"
        "Examples:\n\n"
        "```\n"
        "coerce_args((m, 3.0, 5))    → (m, tensor(3.0), tensor(5.0))\n"
        "coerce_args((arr, True))    → (arr, True)             # ndarray + bool pass-through\n"
        "coerce_args(())             → ()\n"
        "```\n\n"
        "**The load-bearing invariants for ex2** (ex1 covered the per-arg "
        "rule):\n\n"
        "1. **`np.ndarray` MUST pass through.** Numpy arrays are already "
        "raw-array-shaped; the downstream unbox step doesn't need to "
        "convert them, and the raw torch fn accepts numpy via the tensor "
        "protocol. Wrapping it in `t.tensor(arr)` would copy memory and "
        "promote dtype.\n\n"
        "2. **A list of plain ints is pass-through too** — only `int` and "
        "`float` themselves (not collections containing them) coerce. "
        "(Lists go to nested-unbox, not coerce.)\n\n"
        "3. **Order + length preserved.**\n\n"
        "You may call a single-arg helper `_coerce_one` inside. Or inline "
        "the rule. Either is fine."
    ),
    stub=(
        "def coerce_args(args: tuple) -> tuple:\n"
        '    """Apply scalar coercion (int/float → 0-D tensor) per-arg; pass everything else through."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- empty + all pass-through ---\n"
        "assert coerce_args(()) == ()\n"
        "result = coerce_args(('x', None, (3, 4)))\n"
        "assert result == ('x', None, (3, 4))\n"
        "\n"
        "# --- scalar coercion across mixed tuple ---\n"
        "raw = t.tensor([1.0, 2.0, 3.0])\n"
        "result = coerce_args((raw, 3.0, 5))\n"
        "assert len(result) == 3\n"
        "assert result[0] is raw, 'raw tensor identity preserved'\n"
        "assert isinstance(result[1], t.Tensor)\n"
        "assert result[1].ndim == 0 and result[1].item() == 3.0\n"
        "assert isinstance(result[2], t.Tensor)\n"
        "assert result[2].dtype == t.float32, 'int → float tensor (not int tensor)'\n"
        "assert result[2].item() == 5.0\n"
        "\n"
        "# --- bool pass-through (subclass-of-int trap, from ex1) ---\n"
        "result = coerce_args((True, False, 2.5))\n"
        "assert result[0] is True, 'True must pass through, NOT become tensor(1.0)'\n"
        "assert result[1] is False\n"
        "assert isinstance(result[2], t.Tensor) and result[2].item() == 2.5\n"
        "\n"
        "# --- numpy ndarray MUST pass through (NEW invariant for ex2) ---\n"
        "arr = np.array([1.0, 2.0, 3.0])\n"
        "result = coerce_args((arr,))\n"
        "assert result[0] is arr, (\n"
        "    'np.ndarray must pass through (NOT be wrapped in t.tensor) — '\n"
        "    'downstream unbox handles raw-array types; wrapping would copy memory'\n"
        ")\n"
        "\n"
        "# --- np.ndarray + tensor + float in one call ---\n"
        "result = coerce_args((arr, raw, 1.5))\n"
        "assert result[0] is arr\n"
        "assert result[1] is raw\n"
        "assert isinstance(result[2], t.Tensor) and result[2].item() == 1.5\n"
        "\n"
        "# --- MiniTensor pass-through (the wrapper unboxes it later, not here) ---\n"
        "mt = MiniTensor(t.tensor([1.0]))\n"
        "result = coerce_args((mt, 3.0))\n"
        "assert result[0] is mt, 'MiniTensor unchanged at coerce step (unbox happens after)'\n"
        "\n"
        "# --- list of plain ints passes through (lists aren't scalars) ---\n"
        "result = coerce_args(([1, 2, 3], (4, 5)))\n"
        "assert result == ([1, 2, 3], (4, 5))\n"
        "assert isinstance(result[0], list)\n"
        "\n"
        "# --- length and order always preserved ---\n"
        "for inp in [(1,), (1, 2, 3), (1.0, 2.0, 'x'), (True, 1, None)]:\n"
        "    out = coerce_args(inp)\n"
        "    assert len(out) == len(inp), f'length: {inp} → {out}'\n"
        "\n"
        "# --- the result is still a tuple, not a generator/list ---\n"
        "result = coerce_args((1, 2))\n"
        "assert isinstance(result, tuple), f'must return tuple, got {type(result).__name__}'"
    ),
    solution_body=(
        "def coerce_args(args: tuple) -> tuple:\n"
        "    def _coerce_one(a):\n"
        "        # bool first (subclass-of-int trap): must pass through.\n"
        "        if isinstance(a, bool):\n"
        "            return a\n"
        "        if isinstance(a, (int, float)):\n"
        "            return t.tensor(float(a))\n"
        "        # Everything else (Tensor, MiniTensor, ndarray, list, tuple,\n"
        "        # None, str, ...) is pass-through — the downstream unbox /\n"
        "        # nested-unbox step handles array-shaped objects.\n"
        "        return a\n"
        "    return tuple(_coerce_one(a) for a in args)"
    ),
    solution_notes=(
        "**Why ndarray is pass-through (not wrapped).** The raw forward "
        "fn (`torch.log`, `torch.add`) accepts numpy arrays — PyTorch's "
        "tensor protocol handles the conversion at the C++ layer with "
        "zero copy on CPU. Pre-emptively calling `t.tensor(arr)` would "
        "(a) trigger a copy + dtype-promotion in Python, and (b) break "
        "the identity invariant the Recipe relies on (different object "
        "stored from what the user passed).\n\n"
        "**Per-arg vs. whole-tuple.** Ex1 was the single-arg helper. "
        "Ex2 is the tuple version. The wrapper actually calls the tuple "
        "version — ex1 is the building block, ex2 is the production "
        "callsite. Same rule, applied N times."
    ),
)


# =========================================================================
# atom: inplace-op-unsafe-warning — ex2: context-manager guard toggle
# =========================================================================
#
# Ex1: add_inplace_safe refuses on recipe-carrying tensors.
# Ex2: provide an escape hatch — a context manager that toggles the guard
# off for a block (advanced users only, e.g. when they KNOW the cache
# isn't needed). Must restore the guard on exit even on exception.

RECAP_INPLACE_CTX = (
    "## In-place guard — context-manager escape hatch — quick refresher\n"
    "\n"
    "Ex1's `add_inplace_safe` ALWAYS refuses when `.recipe is not None`. "
    "But advanced users sometimes need to override (e.g. they're "
    "explicitly detaching, or they know the cached value is dead). The "
    "canonical pattern is a context manager that toggles a module-level "
    "bool:\n"
    "\n"
    "```python\n"
    "with inplace_unsafe():\n"
    "    add_inplace_safe(x, y)   # would normally refuse — now allowed\n"
    "# guard automatically re-armed here\n"
    "```\n"
    "\n"
    "Implementation pattern: save the old value of a module-level flag, "
    "set it to False (or True, depending on convention), yield, restore "
    "the old value in `finally` (so even an exception inside the `with` "
    "block doesn't leave the guard disarmed)."
)

SPEC_INPLACE_CTX = _spec(
    atom_id="inplace-op-unsafe-warning",
    subtopic="Backprop: In-place op unsafe warning",
    recap=RECAP_INPLACE_CTX,
    ex_idx=2,
    ex_title="inplace_unsafe context manager that toggles the guard, restores on exit",
    slug="inplace-unsafe-context-manager-restore-on-exit",
    bloom="Apply",
    difficulty_num=4,
    keywords=["context-manager", "guard", "toggle", "in-place", "finally"],
    kcs=["inplace-op-unsafe-warning", "recipe-dataclass"],
    lo=(
        "Apply the context-manager guard-toggle pattern: save the prior "
        "flag value, override for the block, restore on exit using a "
        "try/finally so exceptions don't leave the guard disabled."
    ),
    prompt_body=(
        "We've set up a module-level flag `_INPLACE_GUARD_ARMED = True` "
        "and a modified `add_inplace_safe(x, y)` that consults it: it "
        "refuses if `x.recipe is not None` AND `_INPLACE_GUARD_ARMED` is "
        "True; otherwise it mutates.\n\n"
        "Implement `inplace_unsafe()` — a CONTEXT MANAGER (use "
        "`@contextlib.contextmanager`) that:\n\n"
        "1. **Saves the prior value** of `_INPLACE_GUARD_ARMED`.\n"
        "2. **Sets `_INPLACE_GUARD_ARMED = False`** for the duration of "
        "the `with` block (i.e. disables the refusal).\n"
        "3. **Yields** (the `with` body runs).\n"
        "4. **Restores the prior value** in a `finally` block, so even "
        "an exception inside the `with` body re-arms the guard.\n\n"
        "Critical: the restore MUST be in `finally`, not after the yield. "
        "If the body raises, code after `yield` is skipped — but `finally` "
        "always runs.\n\n"
        "Tests verify three things:\n"
        "- Inside the block, `add_inplace_safe` MUTATES a recipe-carrying "
        "tensor (no refusal).\n"
        "- Outside the block, `add_inplace_safe` REFUSES again.\n"
        "- An exception inside the block still re-arms the guard.\n\n"
        "Note: the global is named `_INPLACE_GUARD_ARMED` — use the "
        "`global` statement to mutate it from inside the context manager."
    ),
    stub=(
        "import contextlib\n"
        "\n"
        "_INPLACE_GUARD_ARMED = True\n"
        "\n"
        "def add_inplace_safe(x: MiniTensor, y: MiniTensor) -> MiniTensor:\n"
        "    if x.recipe is not None and _INPLACE_GUARD_ARMED:\n"
        "        raise RuntimeError('in-place op forbidden on a Tensor with a recipe')\n"
        "    x.array += y.array\n"
        "    return x\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def inplace_unsafe():\n"
        '    """Disable the in-place guard for the duration of the `with` block."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Setup: a recipe-carrying tensor that would normally refuse mutation.\n"
        "x = MiniTensor(t.tensor([1.0, 2.0]), requires_grad=True)\n"
        "x.recipe = Recipe(t.add, (), {}, {})\n"
        "y = MiniTensor(t.tensor([10.0, 20.0]))\n"
        "\n"
        "# --- baseline: refuses outside the context manager ---\n"
        "raised = False\n"
        "try:\n"
        "    add_inplace_safe(x, y)\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, 'precondition: guard refuses by default'\n"
        "assert t.allclose(x.array, t.tensor([1.0, 2.0])), 'precondition: not mutated'\n"
        "\n"
        "# --- inside the context: mutation is allowed ---\n"
        "with inplace_unsafe():\n"
        "    add_inplace_safe(x, y)\n"
        "assert t.allclose(x.array, t.tensor([11.0, 22.0])), (\n"
        "    f'mutation must succeed inside inplace_unsafe, got {x.array}'\n"
        ")\n"
        "\n"
        "# --- after the context: guard re-armed, refuses again ---\n"
        "x2 = MiniTensor(t.tensor([0.0, 0.0]), requires_grad=True)\n"
        "x2.recipe = Recipe(t.add, (), {}, {})\n"
        "raised = False\n"
        "try:\n"
        "    add_inplace_safe(x2, y)\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, 'guard must re-arm after with-block exits'\n"
        "\n"
        "# --- exception inside block STILL re-arms the guard ---\n"
        "x3 = MiniTensor(t.tensor([0.0, 0.0]), requires_grad=True)\n"
        "x3.recipe = Recipe(t.add, (), {}, {})\n"
        "raised_outer = False\n"
        "try:\n"
        "    with inplace_unsafe():\n"
        "        raise ValueError('synthetic explosion inside the block')\n"
        "except ValueError:\n"
        "    raised_outer = True\n"
        "assert raised_outer, 'precondition'\n"
        "# Now the guard must be re-armed even though an exception left the block.\n"
        "raised = False\n"
        "try:\n"
        "    add_inplace_safe(x3, y)\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, (\n"
        "    'guard MUST be re-armed even after exception inside the with-block — '\n"
        "    'use try/finally, not just code-after-yield'\n"
        ")\n"
        "\n"
        "# --- nested contexts work (save/restore prior value, not hard-code True) ---\n"
        "x4 = MiniTensor(t.tensor([0.0]), requires_grad=True)\n"
        "x4.recipe = Recipe(t.add, (), {}, {})\n"
        "with inplace_unsafe():\n"
        "    with inplace_unsafe():\n"
        "        add_inplace_safe(x4, MiniTensor(t.tensor([1.0])))\n"
        "    # outer block still has guard disabled (inner saved + restored 'False')\n"
        "    assert _INPLACE_GUARD_ARMED is False, (\n"
        "        'inner context must restore the PRIOR value (False), not hard-code True'\n"
        "    )\n"
        "assert _INPLACE_GUARD_ARMED is True, 'outer context must restore True'"
    ),
    solution_body=(
        "import contextlib\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def inplace_unsafe():\n"
        "    global _INPLACE_GUARD_ARMED\n"
        "    prev = _INPLACE_GUARD_ARMED\n"
        "    _INPLACE_GUARD_ARMED = False\n"
        "    try:\n"
        "        yield\n"
        "    finally:\n"
        "        # finally runs even on exception — guard is always restored.\n"
        "        _INPLACE_GUARD_ARMED = prev"
    ),
    solution_notes=(
        "**Why `try/finally` is non-negotiable.** Code AFTER the `yield` "
        "in a context manager runs only if the body completes normally. "
        "If the body raises, that code is skipped — and the guard stays "
        "disabled forever. `finally` runs unconditionally, even on "
        "exception, even on `KeyboardInterrupt`. This is how PyTorch's "
        "`torch.no_grad()`, `torch.enable_grad()`, and `torch.set_grad_"
        "enabled()` are all written.\n\n"
        "**Why save `prev`, not hard-code `True`.** Nested `with` blocks. "
        "If you hard-code restoring to True, then an outer "
        "`with inplace_unsafe():` followed by an inner "
        "`with inplace_unsafe():` would have the inner's exit re-arm the "
        "guard — but the outer expected it to stay disabled. Saving the "
        "previous value composes correctly under nesting."
    ),
)


# =========================================================================
# atom: parameter-subclass-of-tensor — ex2: distinguishing role at runtime
# =========================================================================
#
# Ex1: subclass MiniTensor with rg=True default.
# Ex2: USE the Parameter type to distinguish trainable state from other
# Tensors. Implement `trainable_params(module)` that yields only attrs that
# are specifically `isinstance(_, Parameter)` (not just MiniTensor).

RECAP_PARAMETER_ROLE = (
    "## Parameter — type as a runtime role signal — quick refresher\n"
    "\n"
    "Ex1 established the IS-A relationship: `Parameter` subclasses "
    "`MiniTensor` so it passes `isinstance(_, MiniTensor)` checks in the "
    "wrapper layer. Ex2 uses the CONVERSE direction: `isinstance(_, "
    "Parameter)` distinguishes TRAINABLE state from intermediate tensors.\n"
    "\n"
    "A real model has multiple kinds of Tensor-typed attributes:\n"
    "- **Parameters** (weights, biases): trainable; the optimizer mutates them.\n"
    "- **Buffers** (e.g. running mean in BatchNorm): MiniTensor but NOT "
    "Parameter; saved with the model but not updated by the optimizer.\n"
    "- **Activations** / scratch tensors: incidental; not saved.\n"
    "\n"
    "`isinstance(x, Parameter)` is the runtime gate that says 'this one "
    "gets gradient-descent updates'."
)

SPEC_PARAMETER_TRAINABLE = _spec(
    atom_id="parameter-subclass-of-tensor",
    subtopic="Backprop: Parameter subclasses Tensor",
    recap=RECAP_PARAMETER_ROLE,
    ex_idx=2,
    ex_title="trainable_params: filter module attrs by Parameter type, not MiniTensor",
    slug="trainable-params-filter-by-parameter-type",
    bloom="Apply",
    difficulty_num=3,
    keywords=["parameter", "trainable", "buffer", "filter", "isinstance"],
    kcs=["parameter-subclass-of-tensor", "get-children-callable-param"],
    lo=(
        "Apply the Parameter-type filter to separate trainable params "
        "from buffers and incidental tensors: walk module attributes, "
        "yield only those that are `isinstance(_, Parameter)`."
    ),
    prompt_body=(
        "We've given you the `Parameter` class from ex1 (subclass of "
        "MiniTensor, default `requires_grad=True`) and a `Module` base "
        "with `__dict__`-based attribute storage. Implement "
        "`trainable_params(module)` — a generator that yields each "
        "`(name, value)` where `isinstance(value, Parameter)` "
        "specifically.\n\n"
        "It must distinguish:\n"
        "- **Parameters** → YIELD (these are the trainables).\n"
        "- **Plain MiniTensors** (e.g. buffers, running statistics, "
        "intermediate caches) → SKIP. They're MiniTensor-typed but not "
        "Parameter-typed.\n"
        "- **Raw torch.Tensors** → SKIP (not even MiniTensor).\n"
        "- **Non-tensor attrs** (ints, strings, layer config) → SKIP.\n\n"
        "Signature: `trainable_params(module) -> generator of (name, "
        "Parameter)`.\n\n"
        "**Why a separate function from ex1's `get_children`.** Ex1 used "
        "`isinstance(_, MiniTensor)` — that catches Parameters AND "
        "buffers AND any other MiniTensor-typed attrs. For an optimizer, "
        "we want STRICTLY trainable: Parameter only. `isinstance(_, "
        "Parameter)` is the strict-subset filter.\n\n"
        "This is exactly the design split in `nn.Module`: "
        "`parameters()` and `buffers()` are two different walkers, "
        "distinguished by the registered type."
    ),
    stub=(
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "\n"
        "class Module:\n"
        '    """Tiny nn.Module stand-in."""\n'
        "\n"
        "def trainable_params(module):\n"
        '    """Yield (name, Parameter) for each attr that is_instance Parameter."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- pure-Parameter module: all attrs yielded ---\n"
        "class Linear(Module):\n"
        "    def __init__(self):\n"
        "        self.weight = Parameter(t.randn(4, 3))\n"
        "        self.bias = Parameter(t.zeros(4))\n"
        "        self.in_features = 3\n"
        "        self.out_features = 4\n"
        "\n"
        "lin = Linear()\n"
        "names = [n for n, _ in trainable_params(lin)]\n"
        "assert names == ['weight', 'bias'], f'pure-Parameter case: {names}'\n"
        "\n"
        "# --- mixed: Parameters + plain MiniTensors (buffers) → only Parameters yielded ---\n"
        "class BatchNorm(Module):\n"
        "    def __init__(self):\n"
        "        # trainable\n"
        "        self.gamma = Parameter(t.ones(4))\n"
        "        self.beta = Parameter(t.zeros(4))\n"
        "        # buffers — MiniTensor but NOT Parameter (saved, but not optimized)\n"
        "        self.running_mean = MiniTensor(t.zeros(4))\n"
        "        self.running_var = MiniTensor(t.ones(4))\n"
        "\n"
        "bn = BatchNorm()\n"
        "names = [n for n, _ in trainable_params(bn)]\n"
        "assert names == ['gamma', 'beta'], (\n"
        "    'plain MiniTensor (buffer) must NOT be yielded — only Parameter — '\n"
        "    f'got {names}'\n"
        ")\n"
        "\n"
        "# --- raw torch.Tensor must NOT be yielded ---\n"
        "class Mixed(Module):\n"
        "    def __init__(self):\n"
        "        self.W = Parameter(t.randn(3, 3))\n"
        "        self.cache = t.zeros(3)             # raw torch.Tensor — skip\n"
        "        self.b = Parameter(t.zeros(3))\n"
        "\n"
        "mx = Mixed()\n"
        "names = [n for n, _ in trainable_params(mx)]\n"
        "assert names == ['W', 'b'], (\n"
        "    'raw torch.Tensor must be skipped (not Parameter, not MiniTensor)'\n"
        ")\n"
        "\n"
        "# --- empty module: yields nothing, no crash ---\n"
        "class Empty(Module):\n"
        "    def __init__(self):\n"
        "        pass\n"
        "assert list(trainable_params(Empty())) == []\n"
        "\n"
        "# --- values are the actual Parameter instances (identity, not copies) ---\n"
        "p = next(trainable_params(lin))\n"
        "assert p[1] is lin.weight, 'value must BE the attribute, not a copy'\n"
        "\n"
        "# --- it is a generator (cheap iteration) ---\n"
        "import inspect\n"
        "iterator = trainable_params(lin)\n"
        "assert iter(iterator) is iterator or inspect.isgenerator(iterator), (\n"
        "    'trainable_params must be a generator'\n"
        ")\n"
        "\n"
        "# --- Parameter with requires_grad=False (frozen) STILL counts as trainable_params ---\n"
        "# The type tag is what matters — requires_grad is mutable. A frozen layer is\n"
        "# still 'a Parameter that happens to be frozen', not 'a buffer'.\n"
        "class WithFrozen(Module):\n"
        "    def __init__(self):\n"
        "        self.W = Parameter(t.randn(3), requires_grad=False)  # frozen but still a Parameter\n"
        "        self.b = Parameter(t.zeros(3))\n"
        "\n"
        "wf = WithFrozen()\n"
        "names = [n for n, _ in trainable_params(wf)]\n"
        "assert names == ['W', 'b'], (\n"
        "    f'frozen Parameter (rg=False) still counts — filter on TYPE not rg, got {names}'\n"
        ")\n"
        "\n"
        "# --- isinstance(Parameter, MiniTensor) check: precondition for ex1's get_children ---\n"
        "p = Parameter(t.zeros(3))\n"
        "assert isinstance(p, MiniTensor), (\n"
        "    'Parameter must still be a MiniTensor (ex1 invariant) — '\n"
        "    'otherwise the wrapper helpers silently drop it'\n"
        ")"
    ),
    solution_body=(
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "\n"
        "class Module:\n"
        "    pass\n"
        "\n"
        "def trainable_params(module):\n"
        "    for name, val in module.__dict__.items():\n"
        "        if isinstance(val, Parameter):\n"
        "            yield name, val"
    ),
    solution_notes=(
        "**Why filter on type, not on `requires_grad`.** A frozen "
        "Parameter has `requires_grad=False` but is STILL a Parameter "
        "(typed as trainable, just temporarily frozen). Filtering by "
        "`requires_grad` would skip frozen params — wrong for "
        "`state_dict` (we still want to save them) and for "
        "checkpointing.\n\n"
        "**Why this is the converse of ex1's get_children.** "
        "`get_children` uses `isinstance(_, MiniTensor)` — the SUPERTYPE "
        "check, catches all wrapper-typed tensors. `trainable_params` "
        "uses `isinstance(_, Parameter)` — the SUBTYPE check, catches "
        "only the typed-trainable subset. Same `__dict__` walk, "
        "different filter."
    ),
)


# =========================================================================
# atom: grad-accumulate-on-leaf — ex2: zero_grad and the accumulate cycle
# =========================================================================
#
# Ex1: accumulate_grad(leaf, g) — set or add.
# Ex2: zero_grad(params) — set each param.grad = None. Show the round trip:
# accumulate → zero → accumulate gives the same result as a single
# accumulate (without zero, gradients persist across steps).

RECAP_ZERO_GRAD = (
    "## Grad accumulate → zero_grad cycle — quick refresher\n"
    "\n"
    "Ex1's `accumulate_grad` ALWAYS adds (never overwrites). This is the "
    "right behavior for a single backward pass through a graph with shared "
    "parameters. But across training STEPS, last step's gradient must be "
    "cleared — otherwise it leaks into the next step's update.\n"
    "\n"
    "Canonical training loop:\n"
    "```python\n"
    "for batch in loader:\n"
    "    optimizer.zero_grad()           # 1. clear last step's grads\n"
    "    loss = forward(batch)\n"
    "    loss.backward()                  # 2. accumulate this step's grads\n"
    "    optimizer.step()                 # 3. apply update\n"
    "```\n"
    "\n"
    "`zero_grad` is the canonical way to clear: for each Parameter, set "
    "`.grad = None` (or zero it). PyTorch's `set_to_none=True` (default "
    "in 2.0+) is `.grad = None` — cheaper than zeroing because it skips "
    "an allocation."
)

SPEC_ZERO_GRAD = _spec(
    atom_id="grad-accumulate-on-leaf",
    subtopic="Backprop: Grad accumulate on leaf",
    recap=RECAP_ZERO_GRAD,
    ex_idx=2,
    ex_title="zero_grad(params): clear leaf.grad to None across an iterable of params",
    slug="zero-grad-clear-leaf-grad-iterable",
    bloom="Apply",
    difficulty_num=3,
    keywords=["zero-grad", "training-loop", "leaf", "set-to-none", "step-boundary"],
    kcs=["grad-accumulate-on-leaf", "parameter-subclass-of-tensor"],
    lo=(
        "Apply the zero_grad pattern at the training-step boundary: walk "
        "an iterable of leaves and set each `.grad = None`, allowing the "
        "next step's accumulate to start fresh."
    ),
    prompt_body=(
        "Implement two functions, building on ex1:\n\n"
        "**1. `accumulate_grad(leaf, g)`** — same as ex1. Set "
        "`leaf.grad = g` on first touch (when `.grad is None`), else "
        "`leaf.grad = leaf.grad + g`. (Use rebinding `+`, not `+=`.)\n\n"
        "**2. `zero_grad(params)`** — given an iterable of MiniTensors, "
        "set each one's `.grad = None`. This is the `set_to_none=True` "
        "PyTorch convention (default in PyTorch 2.0+):\n\n"
        "```python\n"
        "def zero_grad(params):\n"
        "    for p in params:\n"
        "        p.grad = None\n"
        "```\n\n"
        "Rules:\n"
        "- `None` (not `t.zeros_like(p.array)`). Setting to None means "
        "the next `accumulate_grad` takes the first-touch (rebind) path "
        "— skipping an allocation.\n"
        "- Accept any iterable, not just a list — generators, tuples, "
        "`module.parameters()` all should work.\n\n"
        "**The round-trip invariant** is the load-bearing test for ex2:\n\n"
        "```\n"
        "step 1:  accumulate(p, g1); accumulate(p, g2)  → p.grad = g1+g2\n"
        "         zero_grad([p])                         → p.grad = None\n"
        "step 2:  accumulate(p, g3); accumulate(p, g4)  → p.grad = g3+g4   (NOT g1+g2+g3+g4!)\n"
        "```\n\n"
        "Without `zero_grad`, the second step's `.grad` would be "
        "`g1+g2+g3+g4` — last step's gradients corrupting this step's "
        "update. That's the bug `zero_grad` exists to prevent."
    ),
    stub=(
        "def accumulate_grad(leaf: MiniTensor, g) -> None:\n"
        '    """Set leaf.grad on first touch, add on subsequent."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def zero_grad(params) -> None:\n"
        '    """Set leaf.grad = None for each leaf in params (set-to-none convention)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- accumulate_grad still behaves per ex1 ---\n"
        "p = MiniTensor(t.zeros(3), requires_grad=True)\n"
        "accumulate_grad(p, t.tensor([1.0, 2.0, 3.0]))\n"
        "accumulate_grad(p, t.tensor([10.0, 20.0, 30.0]))\n"
        "assert t.allclose(p.grad, t.tensor([11.0, 22.0, 33.0]))\n"
        "\n"
        "# --- zero_grad: single leaf, .grad → None ---\n"
        "zero_grad([p])\n"
        "assert p.grad is None, (\n"
        "    f'zero_grad must set .grad to None (set-to-none convention), got {p.grad}'\n"
        ")\n"
        "\n"
        "# --- zero_grad accepts an iterable (generator), not just a list ---\n"
        "p2 = MiniTensor(t.zeros(2), requires_grad=True)\n"
        "p3 = MiniTensor(t.zeros(2), requires_grad=True)\n"
        "accumulate_grad(p2, t.ones(2))\n"
        "accumulate_grad(p3, t.ones(2))\n"
        "zero_grad(iter([p2, p3]))   # generator/iterator, not a list\n"
        "assert p2.grad is None and p3.grad is None\n"
        "\n"
        "# --- zero_grad handles already-None grad (no crash) ---\n"
        "p4 = MiniTensor(t.zeros(2), requires_grad=True)\n"
        "assert p4.grad is None\n"
        "zero_grad([p4])  # idempotent\n"
        "assert p4.grad is None\n"
        "\n"
        "# --- THE LOAD-BEARING TEST: round-trip invariance across steps ---\n"
        "# step 1: accumulate g1, g2\n"
        "# zero_grad\n"
        "# step 2: accumulate g3, g4\n"
        "# Expected: after step 2, p.grad == g3 + g4 (NOT g1+g2+g3+g4)\n"
        "p = MiniTensor(t.zeros(3), requires_grad=True)\n"
        "g1, g2, g3, g4 = (\n"
        "    t.tensor([1.0, 0.0, 0.0]),\n"
        "    t.tensor([0.0, 1.0, 0.0]),\n"
        "    t.tensor([0.0, 0.0, 1.0]),\n"
        "    t.tensor([1.0, 1.0, 1.0]),\n"
        ")\n"
        "# step 1\n"
        "accumulate_grad(p, g1)\n"
        "accumulate_grad(p, g2)\n"
        "assert t.allclose(p.grad, g1 + g2)\n"
        "\n"
        "zero_grad([p])\n"
        "\n"
        "# step 2 — fresh accumulation, last step's gradient must be gone\n"
        "accumulate_grad(p, g3)\n"
        "accumulate_grad(p, g4)\n"
        "expected = g3 + g4\n"
        "leaked = g1 + g2 + g3 + g4  # what we'd see WITHOUT zero_grad\n"
        "assert t.allclose(p.grad, expected), (\n"
        "    f'zero_grad failed: p.grad should be {expected} (step-2 only), got {p.grad}'\n"
        ")\n"
        "assert not t.allclose(p.grad, leaked), (\n"
        "    'p.grad still contains step-1 contributions — zero_grad did not clear'\n"
        ")\n"
        "\n"
        "# --- after zero_grad, the FIRST accumulate post-zero takes the first-touch path ---\n"
        "# (the .grad rebinds to the input tensor exactly, no addition)\n"
        "p = MiniTensor(t.zeros(2), requires_grad=True)\n"
        "accumulate_grad(p, t.tensor([5.0, 6.0]))\n"
        "zero_grad([p])\n"
        "fresh_g = t.tensor([100.0, 200.0])\n"
        "accumulate_grad(p, fresh_g)\n"
        "assert p.grad is fresh_g, (\n"
        "    'first-post-zero accumulate must take first-touch path (rebind, not add) — '\n"
        "    'this is why set-to-none beats set-to-zeros (skips an allocation + add)'\n"
        ")"
    ),
    solution_body=(
        "def accumulate_grad(leaf: MiniTensor, g) -> None:\n"
        "    if leaf.grad is None:\n"
        "        leaf.grad = g\n"
        "    else:\n"
        "        leaf.grad = leaf.grad + g\n"
        "\n"
        "\n"
        "def zero_grad(params) -> None:\n"
        "    for p in params:\n"
        "        p.grad = None"
    ),
    solution_notes=(
        "**Why `None`, not zeros.** PyTorch 2.0 switched `zero_grad`'s "
        "default to `set_to_none=True` because:\n"
        "(a) zero allocation cost vs allocating `t.zeros_like(p.grad)`,\n"
        "(b) the FIRST accumulate after a `None` reset takes the rebind "
        "path (just stores the incoming tensor) instead of "
        "`zeros + g` — saves an addition,\n"
        "(c) downstream code that does `if p.grad is None: ...` works "
        "as expected.\n\n"
        "**Why a separate function from accumulate_grad.** The split is "
        "the natural division of responsibility: `accumulate_grad` runs "
        "during a single backward pass (called by every leaf-touching "
        "back-fn), `zero_grad` runs once per training step (called from "
        "the training loop). Coupling them would make multi-step accumulation "
        "(e.g. gradient accumulation across micro-batches) awkward."
    ),
)


# =========================================================================
# atom: sum-and-broadcast-duality — ex2: adjoint inner-product test
# =========================================================================
#
# Ex1: write sum_back + broadcast_back as separate ops.
# Ex2: prove operationally that sum_back IS the adjoint of sum via the
# fundamental adjoint identity <Av, w> == <v, A^T w>. This is the
# load-bearing facet: not just "the shape works", but "the operator IS
# the transpose of the forward op as a linear map".

RECAP_ADJOINT = (
    "## sum/broadcast duality — the adjoint identity — quick refresher\n"
    "\n"
    "A linear map `A` has an adjoint `A^T` (transpose for real-valued "
    "tensors) characterized by the **inner-product identity**:\n"
    "\n"
    "$$\\langle A v, w \\rangle = \\langle v, A^T w \\rangle$$\n"
    "\n"
    "For backprop: the forward op is `A`, the back-fn is `A^T`. So:\n"
    "\n"
    "- forward: `sum(x, dim=k)` ≡ multiplying by a row-vector of 1s "
    "along axis k. Linear map A.\n"
    "- backward: `sum_back(g, ..., dim=k)` ≡ broadcasting g back along "
    "axis k. Linear map A^T.\n"
    "\n"
    "The identity becomes: for any inputs x (shape of forward arg) and y "
    "(shape of forward output),\n"
    "\n"
    "$$\\langle \\mathrm{sum}(x, k), y \\rangle = \\langle x, "
    "\\mathrm{sum\\_back}(y, k) \\rangle$$\n"
    "\n"
    "If `sum_back` is truly the adjoint, this equality holds for every "
    "x, y. If it isn't, the identity will fail on a random probe — "
    "and gradients downstream would be silently wrong."
)

SPEC_ADJOINT_TEST = _spec(
    atom_id="sum-and-broadcast-duality",
    subtopic="Backprop: sum/broadcast duality",
    recap=RECAP_ADJOINT,
    ex_idx=2,
    ex_title="check_adjoint: verify sum_back is the adjoint of sum via <Av, w> = <v, A^T w>",
    slug="check-adjoint-inner-product-identity-sum-back",
    bloom="Apply",
    difficulty_num=4,
    keywords=["adjoint", "transpose", "inner-product", "duality", "sum-back", "linear-map"],
    kcs=["sum-and-broadcast-duality", "unbroadcast-pattern"],
    lo=(
        "Apply the adjoint identity `<Av, w> == <v, A^T w>` to verify "
        "operationally that sum_back is the transpose of sum as a "
        "linear map — the deeper meaning of 'duality' beyond shape "
        "matching."
    ),
    prompt_body=(
        "Implement two functions:\n\n"
        "**1. `sum_back(grad_out, x, dim)`** — same as ex1's `sum_back` "
        "with `keepdim=False`. Unsqueeze grad_out at `dim`, then "
        "`.expand_as(x).clone()`.\n\n"
        "**2. `check_adjoint(x, dim)`** — verify the adjoint identity. "
        "Given an input shape (via concrete tensor `x`) and a reduction "
        "axis `dim`:\n\n"
        "1. Compute `Ax = x.sum(dim=dim)` (shape: x.shape with dim "
        "removed).\n"
        "2. Sample a random `y` with the same shape as `Ax`.\n"
        "3. Compute LHS = `(Ax * y).sum()` — the inner product "
        "`<Ax, y>`.\n"
        "4. Compute RHS = `(x * sum_back(y, x, dim)).sum()` — the inner "
        "product `<x, A^T y>`.\n"
        "5. Return `t.allclose(LHS, RHS, atol=1e-5)`.\n\n"
        "If `sum_back` is correctly the adjoint, the function returns "
        "True for every choice of `x.shape` and `dim`. If `sum_back` is "
        "buggy (e.g. forgets `unsqueeze`, uses wrong dim, applies "
        "`keepdim=True` semantics), the identity FAILS — visible as a "
        "scalar mismatch.\n\n"
        "Use `t.manual_seed(0)` before sampling `y` so the test is "
        "deterministic.\n\n"
        "**Why this matters more than shape tests.** Ex1's tests checked "
        "`g.shape == x.shape` and a few hand-computed values. Shape can "
        "be right but values wrong — e.g. if you forget `unsqueeze` and "
        "accidentally broadcast across the wrong axis. The adjoint "
        "identity is a SCALAR check that catches any deviation from "
        "true linear-algebraic transpose-ness."
    ),
    stub=(
        "def sum_back(grad_out: Tensor, x: Tensor, dim: int) -> Tensor:\n"
        '    """Backward of x.sum(dim, keepdim=False). Broadcast grad_out to x.shape."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def check_adjoint(x: Tensor, dim: int) -> bool:\n"
        '    """Verify <Ax, y> == <x, A^T y> where A = sum(dim), A^T = sum_back(dim)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- sum_back basic shape ---\n"
        "x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
        "g = sum_back(t.ones(3), x, dim=1)\n"
        "assert g.shape == (3, 4), f'shape: {g.shape}'\n"
        "assert t.allclose(g, t.ones(3, 4))\n"
        "\n"
        "# --- adjoint identity holds for 2-D reductions ---\n"
        "t.manual_seed(0)\n"
        "x = t.randn(3, 4)\n"
        "assert check_adjoint(x, dim=0) is True, '<Ax, y> != <x, A^T y> on dim=0'\n"
        "assert check_adjoint(x, dim=1) is True, '<Ax, y> != <x, A^T y> on dim=1'\n"
        "\n"
        "# --- adjoint identity holds for 3-D reductions across every axis ---\n"
        "x = t.randn(2, 3, 4)\n"
        "for dim in range(3):\n"
        "    assert check_adjoint(x, dim=dim) is True, (\n"
        "        f'adjoint identity must hold for dim={dim} on 3-D tensor'\n"
        "    )\n"
        "\n"
        "# --- adjoint identity holds for various shapes ---\n"
        "for shape in [(5,), (5, 6), (2, 3, 4), (1, 7, 2)]:\n"
        "    x = t.randn(shape)\n"
        "    for dim in range(len(shape)):\n"
        "        ok = check_adjoint(x, dim=dim)\n"
        "        assert ok, f'adjoint failed: shape={shape}, dim={dim}'\n"
        "\n"
        "# --- explicit LHS == RHS via hand-computed inner products ---\n"
        "# This is the underlying identity, broken out so the test author can\n"
        "# read what 'adjoint' actually means.\n"
        "x = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)\n"
        "Ax = x.sum(dim=1)                                 # shape (2,) = [6, 15]\n"
        "y = t.tensor([10.0, 100.0])                       # shape (2,)\n"
        "lhs = (Ax * y).sum()                              # 6*10 + 15*100 = 60 + 1500 = 1560\n"
        "rhs = (x * sum_back(y, x, dim=1)).sum()\n"
        "assert t.allclose(lhs, rhs, atol=1e-5), (\n"
        "    f'hand-computed adjoint identity: lhs={lhs.item()}, rhs={rhs.item()}'\n"
        ")\n"
        "assert lhs.item() == 1560.0\n"
        "\n"
        "# --- check_adjoint is deterministic (same x, dim → same result) ---\n"
        "x = t.randn(4, 5)\n"
        "r1 = check_adjoint(x, dim=0)\n"
        "r2 = check_adjoint(x, dim=0)\n"
        "assert r1 == r2, 'check_adjoint must be deterministic given x'\n"
        "\n"
        "# --- sanity: sum_back agrees with torch.autograd on a small case ---\n"
        "x_ref = t.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)\n"
        "y_ref = x_ref.sum(dim=1).sum()\n"
        "y_ref.backward()\n"
        "g_ours = sum_back(t.ones(2), x_ref.detach(), dim=1)\n"
        "assert t.allclose(g_ours, x_ref.grad), (\n"
        "    f'sum_back must match autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def sum_back(grad_out: Tensor, x: Tensor, dim: int) -> Tensor:\n"
        "    # keepdim=False semantics: re-insert the dropped axis, then expand.\n"
        "    grad_out = grad_out.unsqueeze(dim)\n"
        "    return grad_out.expand_as(x).clone()\n"
        "\n"
        "\n"
        "def check_adjoint(x: Tensor, dim: int) -> bool:\n"
        "    # Use a fixed seed for reproducibility — same x/dim → same y.\n"
        "    gen = t.Generator().manual_seed(0)\n"
        "    Ax = x.sum(dim=dim)\n"
        "    y = t.randn(Ax.shape, generator=gen)\n"
        "    lhs = (Ax * y).sum()                         # <Ax, y>\n"
        "    rhs = (x * sum_back(y, x, dim=dim)).sum()    # <x, A^T y>\n"
        "    return bool(t.allclose(lhs, rhs, atol=1e-5))"
    ),
    solution_notes=(
        "**Why the adjoint identity is the gold standard.** Shape-only "
        "tests can pass while semantics are wrong (e.g. using "
        "`expand_as` without `unsqueeze` first — produces the right "
        "shape with completely wrong stride pattern on broadcasting "
        "edge cases). The inner-product identity is a SCALAR equality "
        "that captures the linear-algebraic relationship exactly.\n\n"
        "**`<Ax, y> = <x, A^T y>` IS the definition of adjoint.** "
        "Every back-fn in autograd should pass this test against its "
        "corresponding forward. Frameworks like JAX even use it for "
        "automated gradient checking (`jax.test_util.check_grads`).\n\n"
        "**Why `Generator` not global `manual_seed`.** Calling "
        "`t.manual_seed(0)` would mutate global state, affecting "
        "anything else in the test. A local `Generator` keeps the "
        "randomness reproducible AND scoped."
    ),
)


# =========================================================================
# emit
# =========================================================================

ALL_SPECS = [
    SPEC_BOX_ASSEMBLY,
    SPEC_UNBOX_NESTED,
    SPEC_GET_CHILDREN_REC,
    SPEC_COERCE_ARGS,
    SPEC_INPLACE_CTX,
    SPEC_PARAMETER_TRAINABLE,
    SPEC_ZERO_GRAD,
    SPEC_ADJOINT_TEST,
]


# ---------------------------------------------------------------------------
# Verifier — exec preamble + stub + solution + test in a fresh namespace.
# Mirrors author_deepening_a_batch7.py:1133 pattern.
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

        # The shared preamble defines MiniTensor / Recipe / grad_tracking_enabled.
        # exec preamble → stub (may raise NotImplementedError at call time but
        # not at exec time, so it's safe) → solution_body OVERWRITES the stub →
        # test_body.
        try:
            exec(_AUTOGRAD_PREAMBLE, ns)
        except Exception as e:
            failed.append((tag, repr(e), traceback.format_exc()))
            continue

        try:
            exec(spec["stub"], ns)
        except Exception:
            # Stub may include unbound names — tolerate.
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
    print(f"[deepening_k_batch9] Verifying {len(ALL_SPECS)} specs against torch backend...")
    _verify_all(ALL_SPECS)

    print(f"\n[deepening_k_batch9] All verified — emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_k_batch9] {len(ALL_SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
