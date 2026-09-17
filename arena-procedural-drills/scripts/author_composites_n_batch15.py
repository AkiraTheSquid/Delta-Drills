"""Composite drills cx13..cx18 — batch-15 N-series.

Each spec exercises 2-3 atoms together. Emits notebooks to
arena-procedural-drills/composites/part4/<NNN>-<slug>.ipynb.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# ---------------------------------------------------------------------------
# cx13 — wrap a kwargs-taking op (sum dim/keepdim); kwargs stored in recipe
# atoms: kwargs-pass-through-recipe, wrap-forward-fn-generic
# ---------------------------------------------------------------------------
spec_13 = {
    "atom_ids": ["kwargs-pass-through-recipe", "wrap-forward-fn-generic"],
    "subtopics": _subs(["kwargs-pass-through-recipe", "wrap-forward-fn-generic"]),
    "primary_atom": "wrap-forward-fn-generic",
    "part": "part4",
    "exercise_index": 13,
    "exercise_title": "wrap_forward_fn threads kwargs into call AND Recipe",
    "slug": "wrap-forward-fn-with-kwargs-into-recipe",
    "atom_recap_md": (
        "## Composing wrap_forward_fn with kwargs-pass-through Recipe\n\n"
        "`wrap_forward_fn` is the factory that converts a raw numerical fn into a `MiniTensor`-aware "
        "wrapper. The unbox/call/box skeleton is the same for every op, but ops with **keyword args** "
        "(`sum(x, dim=1, keepdim=True)`) force TWO places where the kwargs must thread:\n\n"
        "1. into the **forward call** — otherwise `sum` reduces the wrong axis.\n"
        "2. onto the **Recipe** — otherwise the reverse pass has no way to call `back_fn(..., **recipe.kwargs)` "
        "with the same kwargs.\n\n"
        "This composite exercises both atoms in one wrapper: write `wrap_forward_fn(fwd_fn)` so it (a) unboxes "
        "MiniTensor inputs, (b) calls `fwd_fn(*raw_args, **kwargs)` with kwargs threaded, and (c) attaches "
        "`Recipe(func, raw_args, kwargs, parents)` carrying the SAME kwargs dict the call used.\n\n"
        "The forward result is observable. The Recipe-kwargs storage is INVISIBLE until the reverse pass runs — "
        "which is exactly when the bug bites. The test checks both directly."
    ),
    "prompt_body": (
        "Write `cx13_wrap_forward_fn(fwd_fn)` that returns a closure `tensor_func(*args, **kwargs)` which:\n\n"
        "1. **Unboxes** every `MiniTensor` arg to its `.array`; passes non-Tensor args through unchanged.\n"
        "2. Calls `fwd_fn(*raw_args, **kwargs)` — kwargs MUST reach the call so `sum(x, dim=1)` reduces the right axis.\n"
        "3. Builds `out = MiniTensor(out_raw)` and attaches `out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)` "
        "where `parents = {idx: a for idx, a in enumerate(args) if isinstance(a, MiniTensor)}`. The Recipe must store "
        "the SAME kwargs dict the call used.\n\n"
        "Use `t.sum` as the witness op (it takes `dim` and `keepdim`). Don't touch the differentiability gate — assume "
        "every op is differentiable for this drill."
    ),
    "stub_body": (
        "from dataclasses import dataclass, field\n"
        "from typing import Callable, Optional\n\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Optional[Callable] = None\n"
        "    args: tuple = ()\n"
        "    kwargs: dict = field(default_factory=dict)\n"
        "    parents: dict = field(default_factory=dict)\n\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad: bool = False, recipe=None):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = recipe\n\n"
        "def cx13_wrap_forward_fn(fwd_fn):\n"
        "    \"\"\"Return tensor_func that threads kwargs into call AND Recipe.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "wrapped_sum = cx13_wrap_forward_fn(t.sum)\n"
        "x = MiniTensor(t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\n"
        "# (a) forward call gets kwargs (dim=1)\n"
        "out = wrapped_sum(x, dim=1)\n"
        "assert isinstance(out, MiniTensor), 'output must be a MiniTensor'\n"
        "assert t.allclose(out.array, t.tensor([6.0, 15.0])), (\n"
        "    f'forward dim=1 was ignored: {out.array} (expected [6, 15])'\n"
        ")\n\n"
        "# (b) Recipe carries the SAME kwargs the call used\n"
        "assert out.recipe is not None, 'Recipe was never attached'\n"
        "assert out.recipe.func is t.sum\n"
        "assert out.recipe.kwargs == {'dim': 1}, (\n"
        "    f'Recipe.kwargs missing or wrong: {out.recipe.kwargs}'\n"
        ")\n"
        "assert 0 in out.recipe.parents and out.recipe.parents[0] is x\n\n"
        "# (c) Recipe.args are RAW unboxed tensors\n"
        "assert isinstance(out.recipe.args[0], t.Tensor)\n"
        "assert not isinstance(out.recipe.args[0], MiniTensor), 'args must be unboxed'\n\n"
        "# (d) two kwargs thread through together\n"
        "out2 = wrapped_sum(x, dim=1, keepdim=True)\n"
        "assert out2.array.shape == (2, 1), f'keepdim ignored: {out2.array.shape}'\n"
        "assert out2.recipe.kwargs == {'dim': 1, 'keepdim': True}\n\n"
        "# (e) no-kwargs case stores empty dict (NOT None)\n"
        "wrapped_log = cx13_wrap_forward_fn(t.log)\n"
        "y = MiniTensor(t.tensor([1.0, t.e, t.e * t.e]))\n"
        "out3 = wrapped_log(y)\n"
        "assert t.allclose(out3.array, t.tensor([0.0, 1.0, 2.0]), atol=1e-5)\n"
        "assert out3.recipe.kwargs == {}, (\n"
        "    f'empty kwargs must be {{}}, got {out3.recipe.kwargs!r}'\n"
        ")\n\n"
        "# (f) the wrapper is a fresh closure per call\n"
        "wrapped_log2 = cx13_wrap_forward_fn(t.log)\n"
        "assert wrapped_log2 is not wrapped_log, 'each wrap returns a new closure'\n\n"
        "# (g) non-Tensor args pass through unchanged\n"
        "wrapped_clamp = cx13_wrap_forward_fn(t.clamp)\n"
        "z = MiniTensor(t.tensor([-1.0, 0.5, 2.0]))\n"
        "out4 = wrapped_clamp(z, min=0.0, max=1.0)\n"
        "assert t.allclose(out4.array, t.tensor([0.0, 0.5, 1.0]))\n"
        "assert out4.recipe.kwargs == {'min': 0.0, 'max': 1.0}"
    ),
    "solution_body": (
        "from dataclasses import dataclass, field\n"
        "from typing import Callable, Optional\n\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Optional[Callable] = None\n"
        "    args: tuple = ()\n"
        "    kwargs: dict = field(default_factory=dict)\n"
        "    parents: dict = field(default_factory=dict)\n\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad=False, recipe=None):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = recipe\n\n"
        "def cx13_wrap_forward_fn(fwd_fn):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = tuple(\n"
        "            a.array if isinstance(a, MiniTensor) else a for a in args\n"
        "        )\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        parents = {\n"
        "            i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)\n"
        "        }\n"
        "        out = MiniTensor(out_raw)\n"
        "        out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    "solution_notes": (
        "The two places kwargs flow — `fwd_fn(*raw_args, **kwargs)` AND `Recipe(..., kwargs, ...)` — are independent: "
        "forgetting either is a silent bug that only surfaces on the reverse pass."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["wrap-forward-fn-generic", "kwargs-pass-through-recipe", "recipe-dataclass"],
    "lo": "Compose the wrap_forward_fn unbox/call/box skeleton with the kwargs pass-through pattern so a single wrapper handles kwargs-taking ops end-to-end.",
}
emit_composite(spec_13)


# ---------------------------------------------------------------------------
# cx14 — back fn replays original kwargs via **recipe.kwargs splat
# atoms: kwargs-pass-through-recipe, backward-fn-signature
# ---------------------------------------------------------------------------
spec_14 = {
    "atom_ids": ["kwargs-pass-through-recipe", "backward-fn-signature"],
    "subtopics": _subs(["kwargs-pass-through-recipe", "backward-fn-signature"]),
    "primary_atom": "backward-fn-signature",
    "part": "part4",
    "exercise_index": 14,
    "exercise_title": "back fn replays kwargs via **recipe.kwargs",
    "slug": "back-fn-replays-recipe-kwargs",
    "atom_recap_md": (
        "## Composing the canonical back-fn signature with kwargs replay\n\n"
        "ARENA back fns share one signature: `back_fn(grad_out, out, *args, **kwargs) -> grad_in`. "
        "For an op like `sum(x, dim=1)`, the back fn `sum_back` MUST know which `dim` was reduced — "
        "otherwise it can't broadcast the upstream gradient back to the input shape.\n\n"
        "The Recipe stores the kwargs the forward call used. The reverse-pass dispatcher splats them in:\n\n"
        "```python\n"
        "grad_in = back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)\n"
        "```\n\n"
        "This composite has you write `sum_back` to the canonical signature AND wire it into a dispatcher that "
        "replays the kwargs from the Recipe. The point is that the back-fn body NEVER hardcodes `dim` — it reads "
        "it from the kwargs the Recipe carried."
    ),
    "prompt_body": (
        "Implement two pieces:\n\n"
        "**1. `sum_back(grad_out, out, x, dim=None, keepdim=False)`** — the canonical back fn for `t.sum`. "
        "Given `out = x.sum(dim=dim, keepdim=keepdim)`, return `dL/dx` shaped like `x` by broadcasting `grad_out` "
        "back over the reduced axis. Two cases:\n"
        "  - `dim is None` (full reduction): `grad_out` is a 0-D tensor; broadcast it to `x.shape`.\n"
        "  - `dim is int`: re-insert the reduced axis via `unsqueeze(dim)` (if not `keepdim`), then `expand_as(x)`.\n\n"
        "**2. `cx14_dispatch_back(node, grad_out)`** — given a `MiniTensor` `node` whose `recipe` came from a `sum` call, "
        "look up the right back fn and call it as `back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)`. "
        "Return the resulting grad.\n\n"
        "The dispatcher must `**recipe.kwargs`-splat. It must NOT inspect or hardcode any specific kwarg."
    ),
    "stub_body": (
        "from dataclasses import dataclass, field\n"
        "from typing import Callable, Optional\n\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Optional[Callable] = None\n"
        "    args: tuple = ()\n"
        "    kwargs: dict = field(default_factory=dict)\n"
        "    parents: dict = field(default_factory=dict)\n\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad=False, recipe=None):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = recipe\n\n"
        "def sum_back(grad_out, out, x, dim=None, keepdim=False):\n"
        "    \"\"\"Canonical signature; broadcast grad_out back to x.shape.\"\"\"\n"
        "    raise NotImplementedError\n\n"
        "# Registry the dispatcher reads from.\n"
        "BACK_FUNCS = {(t.sum, 0): sum_back}\n\n"
        "def cx14_dispatch_back(node, grad_out):\n"
        "    \"\"\"Dispatch the back fn for the recipe and replay kwargs via **splat.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Build a node as if produced by a wrap_forward_fn(t.sum) call.\n"
        "x_raw = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n"
        "out_raw = t.sum(x_raw, dim=1)\n"
        "node = MiniTensor(out_raw)\n"
        "node.recipe = Recipe(t.sum, (x_raw,), {'dim': 1}, {0: 'fake-parent'})\n\n"
        "# (a) dispatch replays dim=1 → grad spreads back along axis 1\n"
        "grad_out = t.tensor([1.0, 1.0])\n"
        "g = cx14_dispatch_back(node, grad_out)\n"
        "assert g.shape == x_raw.shape, (\n"
        "    f'sum_back failed to broadcast back to x shape: got {g.shape}'\n"
        ")\n"
        "assert t.allclose(g, t.ones_like(x_raw)), f'expected ones, got {g}'\n\n"
        "# (b) kwargs is the ONLY source of truth — flip dim=0 and the shape spread flips too.\n"
        "out_raw0 = t.sum(x_raw, dim=0)\n"
        "node0 = MiniTensor(out_raw0)\n"
        "node0.recipe = Recipe(t.sum, (x_raw,), {'dim': 0}, {0: 'fake'})\n"
        "g0 = cx14_dispatch_back(node0, t.tensor([1.0, 1.0, 1.0]))\n"
        "assert g0.shape == x_raw.shape\n"
        "assert t.allclose(g0, t.ones_like(x_raw))\n\n"
        "# (c) keepdim flows through unchanged via the **splat (different inserted shape).\n"
        "out_kd = t.sum(x_raw, dim=1, keepdim=True)   # shape (2,1)\n"
        "node_kd = MiniTensor(out_kd)\n"
        "node_kd.recipe = Recipe(t.sum, (x_raw,), {'dim': 1, 'keepdim': True}, {0: 'fake'})\n"
        "g_kd = cx14_dispatch_back(node_kd, t.tensor([[1.0], [1.0]]))\n"
        "assert g_kd.shape == x_raw.shape, f'keepdim branch shape: {g_kd.shape}'\n"
        "assert t.allclose(g_kd, t.ones_like(x_raw))\n\n"
        "# (d) full reduction (dim=None) — grad_out is 0-D, broadcast to x.shape.\n"
        "out_full = t.sum(x_raw)\n"
        "node_full = MiniTensor(out_full)\n"
        "node_full.recipe = Recipe(t.sum, (x_raw,), {}, {0: 'fake'})\n"
        "g_full = cx14_dispatch_back(node_full, t.tensor(1.0))\n"
        "assert g_full.shape == x_raw.shape\n"
        "assert t.allclose(g_full, t.ones_like(x_raw))\n\n"
        "# (e) the back fn agrees with torch.autograd on a non-unit grad_out.\n"
        "x_ref = t.tensor([[2.0, -3.0], [1.0, 4.0]], requires_grad=True)\n"
        "y = x_ref.sum(dim=1)\n"
        "y.backward(t.tensor([5.0, 7.0]))\n"
        "ours = sum_back(t.tensor([5.0, 7.0]), x_ref.detach().sum(dim=1), x_ref.detach(), dim=1)\n"
        "assert t.allclose(ours, x_ref.grad), f'disagree with autograd: {ours} vs {x_ref.grad}'"
    ),
    "solution_body": (
        "def sum_back(grad_out, out, x, dim=None, keepdim=False):\n"
        "    # Canonical signature; broadcast grad_out back across the reduced axis.\n"
        "    if dim is None:\n"
        "        # Full reduction: grad_out is 0-D; broadcast to x.shape.\n"
        "        return grad_out.expand(x.shape).clone()\n"
        "    g = grad_out if keepdim else grad_out.unsqueeze(dim)\n"
        "    return g.expand_as(x).clone()\n\n"
        "BACK_FUNCS = {(t.sum, 0): sum_back}\n\n"
        "def cx14_dispatch_back(node, grad_out):\n"
        "    # Look up by recipe.func; argnum 0 (single-arg op).\n"
        "    back_fn = BACK_FUNCS[(node.recipe.func, 0)]\n"
        "    # **splat — kwargs come from the Recipe, never hardcoded.\n"
        "    return back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)"
    ),
    "solution_notes": (
        "The dispatcher line `back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)` is the ONE "
        "place kwargs flow from forward into reverse. Storing them on the Recipe (cx13) and splatting them here "
        "(cx14) is what keeps `sum_back` op-agnostic — same call shape regardless of op."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["backward-fn-signature", "kwargs-pass-through-recipe", "chain-rule-elementwise"],
    "lo": "Write a back fn that consumes recipe.kwargs via **splat so the reverse pass replays the forward op's keyword args without hardcoding them.",
}
emit_composite(spec_14)


# ---------------------------------------------------------------------------
# cx15 — BackwardFuncLookup keyed by (fwd_fn, argnum); miss raises KeyError
# atoms: backward-func-lookup, arg-position-back-functions
# ---------------------------------------------------------------------------
spec_15 = {
    "atom_ids": ["backward-func-lookup", "arg-position-back-functions"],
    "subtopics": _subs(["backward-func-lookup", "arg-position-back-functions"]),
    "primary_atom": "backward-func-lookup",
    "part": "part4",
    "exercise_index": 15,
    "exercise_title": "BackwardFuncLookup with asymmetric div_back0 / div_back1",
    "slug": "backward-lookup-with-arg-position-div-back",
    "atom_recap_md": (
        "## Composing the registry with arg-position back fns\n\n"
        "`BackwardFuncLookup` is the dispatcher — a dict keyed by `(forward_fn, arg_position) -> back_fn`. "
        "The arg-position part is non-trivial: asymmetric binary ops like `divide` have DIFFERENT back fns "
        "for arg-0 and arg-1:\n\n"
        "- `div_back0(grad_out, out, x, y) = grad_out / y`             (d(x/y)/dx = 1/y)\n"
        "- `div_back1(grad_out, out, x, y) = grad_out * (-x / y**2)`   (d(x/y)/dy = -x/y²)\n\n"
        "Both register into the SAME lookup under the same `t.divide` key — distinguished only by argnum. "
        "A missing `(fn, argnum)` key raises `KeyError` with a diagnostic message so the user knows which "
        "registration they forgot."
    ),
    "prompt_body": (
        "Implement two pieces:\n\n"
        "**1. `BackwardFuncLookup`** — a dict-backed registry with two methods:\n"
        "  - `add_back_func(fwd, argnum, back_fn)` stores under key `(fwd, argnum)`.\n"
        "  - `get_back_func(fwd, argnum)` returns the stored fn, raising `KeyError` with a message that mentions "
        "  both the forward fn AND the argnum on a miss.\n\n"
        "**2. `div_back0` and `div_back1`** — the asymmetric per-arg back fns for `out = x / y` (no broadcasting; "
        "assume `x.shape == y.shape`). Then register BOTH into a `BackwardFuncLookup` under `t.divide` at argnum 0 "
        "and argnum 1.\n\n"
        "The test exercises register + retrieve + miss + dispatcher-style call."
    ),
    "stub_body": (
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        raise NotImplementedError\n\n"
        "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
        "        raise NotImplementedError\n\n"
        "    def get_back_func(self, forward_fn, arg_position):\n"
        "        raise NotImplementedError\n\n"
        "def div_back0(grad_out, out, x, y):\n"
        "    raise NotImplementedError\n\n"
        "def div_back1(grad_out, out, x, y):\n"
        "    raise NotImplementedError\n\n"
        "def cx15_build_registry():\n"
        "    \"\"\"Return a BackwardFuncLookup with div_back0/div_back1 registered under t.divide.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "BF = cx15_build_registry()\n"
        "assert isinstance(BF, BackwardFuncLookup)\n\n"
        "# (a) both argnums resolve to DIFFERENT fns\n"
        "f0 = BF.get_back_func(t.divide, 0)\n"
        "f1 = BF.get_back_func(t.divide, 1)\n"
        "assert f0 is not f1, 'asymmetric op: argnum 0 and 1 must map to different fns'\n\n"
        "# (b) the math is right\n"
        "x = t.tensor([6.0, 10.0]); y = t.tensor([2.0, 5.0])\n"
        "out = x / y\n"
        "g0 = f0(t.ones(2), out, x, y)\n"
        "g1 = f1(t.ones(2), out, x, y)\n"
        "assert t.allclose(g0, 1 / y), f'div_back0 wrong: {g0} vs {1/y}'\n"
        "assert t.allclose(g1, -x / y**2), f'div_back1 wrong: {g1} vs {-x/y**2}'\n\n"
        "# (c) miss raises KeyError with diagnostic mentioning fn AND argnum\n"
        "try:\n"
        "    BF.get_back_func(t.sin, 0)\n"
        "except KeyError as e:\n"
        "    msg = str(e)\n"
        "    assert 'sin' in msg or 'torch' in msg, f'message missing fn: {msg!r}'\n"
        "    assert '0' in msg, f'message missing argnum: {msg!r}'\n"
        "else:\n"
        "    raise AssertionError('missing fn must raise KeyError')\n\n"
        "# (d) right fn / wrong argnum also raises\n"
        "try:\n"
        "    BF.get_back_func(t.divide, 7)\n"
        "except KeyError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('missing argnum must raise KeyError')\n\n"
        "# (e) dispatcher-style call: agree with torch.autograd\n"
        "x_ref = t.tensor([3.0, 8.0], requires_grad=True)\n"
        "y_ref = t.tensor([2.0, 4.0], requires_grad=True)\n"
        "z = (x_ref / y_ref).sum()\n"
        "z.backward()\n"
        "out_c = x_ref.detach() / y_ref.detach()\n"
        "g0_ours = BF.get_back_func(t.divide, 0)(t.ones(2), out_c, x_ref.detach(), y_ref.detach())\n"
        "g1_ours = BF.get_back_func(t.divide, 1)(t.ones(2), out_c, x_ref.detach(), y_ref.detach())\n"
        "assert t.allclose(g0_ours, x_ref.grad, atol=1e-6)\n"
        "assert t.allclose(g1_ours, y_ref.grad, atol=1e-6)\n\n"
        "# (f) two registries are independent\n"
        "BF2 = BackwardFuncLookup()\n"
        "try:\n"
        "    BF2.get_back_func(t.divide, 0)\n"
        "except KeyError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('separate instances must not share storage')"
    ),
    "solution_body": (
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self.back_funcs = {}\n\n"
        "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
        "        self.back_funcs[(forward_fn, arg_position)] = back_fn\n\n"
        "    def get_back_func(self, forward_fn, arg_position):\n"
        "        key = (forward_fn, arg_position)\n"
        "        if key not in self.back_funcs:\n"
        "            raise KeyError(\n"
        "                f'No back_fn for ({forward_fn!r}, argnum={arg_position}). '\n"
        "                f'Did you forget add_back_func(fn, {arg_position}, ...)?'\n"
        "            )\n"
        "        return self.back_funcs[key]\n\n"
        "def div_back0(grad_out, out, x, y):\n"
        "    return grad_out / y\n\n"
        "def div_back1(grad_out, out, x, y):\n"
        "    return grad_out * (-x / (y * y))\n\n"
        "def cx15_build_registry():\n"
        "    bf = BackwardFuncLookup()\n"
        "    bf.add_back_func(t.divide, 0, div_back0)\n"
        "    bf.add_back_func(t.divide, 1, div_back1)\n"
        "    return bf"
    ),
    "solution_notes": (
        "The 2-tuple `(fn, argnum)` is the whole reason asymmetric ops register cleanly under one key prefix — "
        "nested dicts would force an extra `.get()` step and the diagnostic message gets noisier. Symmetric ops "
        "(add, multiply) still register both argnums even though the bodies are mirror images."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["backward-func-lookup", "arg-position-back-functions", "register-back-fn-after-wrap"],
    "lo": "Compose a BackwardFuncLookup with per-arg-position back fns so an asymmetric binary op (divide) registers two distinct back fns under one forward-fn key, and a missing registration surfaces as a diagnostic KeyError.",
}
emit_composite(spec_15)


# ---------------------------------------------------------------------------
# cx16 — register mul_back0/mul_back1 under multiply at argnums 0 and 1
# atoms: backward-func-lookup, multiply-back, arg-position-back-functions
# ---------------------------------------------------------------------------
spec_16 = {
    "atom_ids": ["backward-func-lookup", "multiply-back", "arg-position-back-functions"],
    "subtopics": _subs(["backward-func-lookup", "multiply-back", "arg-position-back-functions"]),
    "primary_atom": "multiply-back",
    "part": "part4",
    "exercise_index": 16,
    "exercise_title": "register multiply_back0 / multiply_back1 in the lookup",
    "slug": "multiply-back-registered-in-lookup",
    "atom_recap_md": (
        "## Composing multiply_back (with unbroadcast) into the BackwardFuncLookup\n\n"
        "`multiply` is mathematically symmetric (`x*y = y*x`), but the manual autograd still registers TWO back fns "
        "— `multiply_back0` for arg-0 and `multiply_back1` for arg-1 — because the dispatcher doesn't know any op "
        "is symmetric. It just looks up `(fn, argnum)`.\n\n"
        "Both bodies follow the SAME pattern: local derivative * grad_out, then `unbroadcast(...)` to collapse any "
        "broadcast axes back to the parent's shape. Then both register into the `BackwardFuncLookup` under "
        "`t.multiply` at argnum 0 and 1.\n\n"
        "This composite has you wire the full registration end-to-end: write the back fns, register them, and "
        "then dispatch by `(t.multiply, argnum)` to compute grads."
    ),
    "prompt_body": (
        "Implement:\n\n"
        "**1. `BackwardFuncLookup`** with `add_back_func` and `get_back_func` (raises `KeyError` on miss).\n\n"
        "**2. `multiply_back0(grad_out, out, x, y)`** — returns `unbroadcast(grad_out * y, x)`, shaped like `x`.\n\n"
        "**3. `multiply_back1(grad_out, out, x, y)`** — returns `unbroadcast(grad_out * x, y)`, shaped like `y`.\n\n"
        "**4. `cx16_build_lookup()`** — returns a populated `BackwardFuncLookup` with both back fns registered "
        "under `t.multiply` at argnum 0 and 1 (use `add_back_func`, not direct dict access).\n\n"
        "The `unbroadcast(grad, original)` helper is provided. The test exercises lookup + dispatch + value + "
        "broadcast collapse."
    ),
    "stub_body": (
        "def unbroadcast(grad, original):\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n\n"
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        raise NotImplementedError\n"
        "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
        "        raise NotImplementedError\n"
        "    def get_back_func(self, forward_fn, arg_position):\n"
        "        raise NotImplementedError\n\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    raise NotImplementedError\n\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    raise NotImplementedError\n\n"
        "def cx16_build_lookup():\n"
        "    \"\"\"Return a BackwardFuncLookup with both multiply_back fns registered under t.multiply.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "BF = cx16_build_lookup()\n"
        "assert isinstance(BF, BackwardFuncLookup)\n\n"
        "# (a) both argnums resolve and the registry is keyed by t.multiply\n"
        "f0 = BF.get_back_func(t.multiply, 0)\n"
        "f1 = BF.get_back_func(t.multiply, 1)\n"
        "assert f0 is multiply_back0\n"
        "assert f1 is multiply_back1\n"
        "assert f0 is not f1\n\n"
        "# (b) same-shape multiply: grads are simply y and x scaled by grad_out\n"
        "x = t.tensor([2.0, 3.0, 4.0]); y = t.tensor([5.0, 6.0, 7.0])\n"
        "out = x * y\n"
        "g0 = f0(t.ones(3), out, x, y)\n"
        "g1 = f1(t.ones(3), out, x, y)\n"
        "assert g0.shape == x.shape and g1.shape == y.shape\n"
        "assert t.allclose(g0, y) and t.allclose(g1, x)\n\n"
        "# (c) BROADCAST: x shape (1,4) * y shape (3,4) -> out (3,4); grad must collapse back\n"
        "x_b = t.tensor([[1.0, 2.0, 3.0, 4.0]])\n"
        "y_b = t.tensor([[5.0, 6.0, 7.0, 8.0],\n"
        "                 [9.0, 10.0, 11.0, 12.0],\n"
        "                 [13.0, 14.0, 15.0, 16.0]])\n"
        "out_b = x_b * y_b\n"
        "grad_b = t.ones(3, 4)\n"
        "g0_b = f0(grad_b, out_b, x_b, y_b)\n"
        "g1_b = f1(grad_b, out_b, x_b, y_b)\n"
        "assert g0_b.shape == x_b.shape, (\n"
        "    f'multiply_back0 forgot unbroadcast: shape {g0_b.shape} != {x_b.shape}'\n"
        ")\n"
        "assert g1_b.shape == y_b.shape\n"
        "expected_g0 = (grad_b * y_b).sum(dim=0, keepdim=True)\n"
        "assert t.allclose(g0_b, expected_g0)\n"
        "assert t.allclose(g1_b, grad_b * x_b.expand_as(y_b))\n\n"
        "# (d) cross-check the broadcast case vs torch.autograd\n"
        "xr = x_b.clone().requires_grad_(True)\n"
        "yr = y_b.clone().requires_grad_(True)\n"
        "(xr * yr).sum().backward()\n"
        "g0_v = f0(t.ones(3, 4), xr.detach() * yr.detach(), xr.detach(), yr.detach())\n"
        "g1_v = f1(t.ones(3, 4), xr.detach() * yr.detach(), xr.detach(), yr.detach())\n"
        "assert t.allclose(g0_v, xr.grad), f'disagree x: {g0_v} vs {xr.grad}'\n"
        "assert t.allclose(g1_v, yr.grad), f'disagree y: {g1_v} vs {yr.grad}'\n\n"
        "# (e) dispatcher-style: argnum comes from a parents dict like the real reverse pass would build\n"
        "parents = {0: x, 1: y}\n"
        "grads = {idx: BF.get_back_func(t.multiply, idx)(t.ones(3), out, x, y) for idx in parents}\n"
        "assert t.allclose(grads[0], y) and t.allclose(grads[1], x)"
    ),
    "solution_body": (
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self.back_funcs = {}\n"
        "    def add_back_func(self, forward_fn, arg_position, back_fn):\n"
        "        self.back_funcs[(forward_fn, arg_position)] = back_fn\n"
        "    def get_back_func(self, forward_fn, arg_position):\n"
        "        key = (forward_fn, arg_position)\n"
        "        if key not in self.back_funcs:\n"
        "            raise KeyError(\n"
        "                f'No back_fn for ({forward_fn!r}, argnum={arg_position}).'\n"
        "            )\n"
        "        return self.back_funcs[key]\n\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    if not isinstance(y, t.Tensor):\n"
        "        y = t.tensor(y)\n"
        "    return unbroadcast(grad_out * y, x)\n\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    if not isinstance(x, t.Tensor):\n"
        "        x = t.tensor(x)\n"
        "    return unbroadcast(grad_out * x, y)\n\n"
        "def cx16_build_lookup():\n"
        "    bf = BackwardFuncLookup()\n"
        "    bf.add_back_func(t.multiply, 0, multiply_back0)\n"
        "    bf.add_back_func(t.multiply, 1, multiply_back1)\n"
        "    return bf"
    ),
    "solution_notes": (
        "Symmetric ops still register twice — mirror bodies, separate (fn, argnum) keys. The dispatcher is the "
        "audience: it looks up by argnum regardless of the op's mathematical symmetry."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["backward-func-lookup", "multiply-back", "arg-position-back-functions"],
    "lo": "Wire multiply_back0 and multiply_back1 (with unbroadcast) into a BackwardFuncLookup under t.multiply so the reverse pass can dispatch by (fn, argnum) to either arg's gradient.",
}
emit_composite(spec_16)


# ---------------------------------------------------------------------------
# cx17 — is_differentiable=False ops skip recipe build even when grad enabled
# atoms: is-differentiable-flag, grad-tracking-global-toggle
# ---------------------------------------------------------------------------
spec_17 = {
    "atom_ids": ["is-differentiable-flag", "grad-tracking-global-toggle"],
    "subtopics": _subs(["is-differentiable-flag", "grad-tracking-global-toggle"]),
    "primary_atom": "is-differentiable-flag",
    "part": "part4",
    "exercise_index": 17,
    "exercise_title": "three-gate requires_grad: per-op flag short-circuits even when toggle is on",
    "slug": "is-differentiable-vs-global-toggle",
    "atom_recap_md": (
        "## Composing the global grad-tracking toggle with the per-op is_differentiable flag\n\n"
        "`requires_grad` on an output is the AND of three gates:\n\n"
        "```\n"
        "requires_grad = grad_tracking_enabled   # gate 1: global (runtime)\n"
        "            AND is_differentiable        # gate 2: per-op (closure)\n"
        "            AND any(input.requires_grad) # gate 3: inputs tracked\n"
        "```\n\n"
        "Gate 1 changes at RUNTIME (`set_grad_tracking(False)` from a `NoGrad` ctx). Gate 2 is captured ONCE at "
        "wrap-time and is sticky for the lifetime of that wrapper. The two are independent — turning the global "
        "toggle on does NOT make a non-differentiable op suddenly differentiable.\n\n"
        "This composite has you implement both gates together and prove the independence with a state-table test."
    ),
    "prompt_body": (
        "Implement THREE pieces:\n\n"
        "**1. `set_grad_tracking(enabled)`** — write the module-level `grad_tracking_enabled` global "
        "(`globals()['grad_tracking_enabled'] = enabled`).\n\n"
        "**2. `make_check_requires_grad(is_differentiable)`** — factory that returns a `check(args) -> bool` closing "
        "over `is_differentiable`. `check` reads `grad_tracking_enabled` from the module globals FRESH each call and "
        "ANDs all three gates.\n\n"
        "**3. `cx17_wrap(fwd_fn, is_differentiable=True)`** — a tiny wrapper that uses the factory: returns a "
        "`tensor_func(*args)` that boxes the result in a `MiniTensor` and only builds a `Recipe` when the three-gate "
        "check returns True.\n\n"
        "The test runs the full 2x2 truth table over `(toggle on/off, is_diff True/False)` to prove the gates "
        "are independent and BOTH must be True for a Recipe to appear."
    ),
    "stub_body": (
        "from dataclasses import dataclass, field\n"
        "from typing import Callable, Optional\n\n"
        "grad_tracking_enabled = True\n\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Optional[Callable] = None\n"
        "    args: tuple = ()\n"
        "    kwargs: dict = field(default_factory=dict)\n"
        "    parents: dict = field(default_factory=dict)\n\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad=False, recipe=None):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = recipe\n\n"
        "def set_grad_tracking(enabled: bool):\n"
        "    raise NotImplementedError\n\n"
        "def make_check_requires_grad(is_differentiable: bool):\n"
        "    raise NotImplementedError\n\n"
        "def cx17_wrap(fwd_fn, is_differentiable=True):\n"
        "    \"\"\"Wrapper that builds a Recipe only when all three gates pass.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Reset the toggle in case a previous cell left it off.\n"
        "set_grad_tracking(True)\n"
        "assert globals()['grad_tracking_enabled'] is True\n\n"
        "add_diff    = cx17_wrap(t.add, is_differentiable=True)\n"
        "eq_nondiff  = cx17_wrap(t.eq, is_differentiable=False)\n\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([1.0, 0.0, 3.0]), requires_grad=True)\n\n"
        "# (i) toggle=True, is_diff=True, tracked inputs → requires_grad True, recipe attached\n"
        "out = add_diff(a, b)\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "assert out.recipe.func is t.add\n\n"
        "# (ii) toggle=True, is_diff=False, tracked inputs → False, NO recipe\n"
        "out = eq_nondiff(a, b)\n"
        "assert out.requires_grad is False, 'is_differentiable=False must short-circuit'\n"
        "assert out.recipe is None, 'non-diff op must NOT build Recipe even with toggle on'\n\n"
        "# (iii) toggle=False, is_diff=True, tracked inputs → False, NO recipe\n"
        "set_grad_tracking(False)\n"
        "out = add_diff(a, b)\n"
        "assert out.requires_grad is False, 'global toggle off must override is_differentiable=True'\n"
        "assert out.recipe is None\n\n"
        "# (iv) toggle=False, is_diff=False, tracked inputs → False (definitely)\n"
        "out = eq_nondiff(a, b)\n"
        "assert out.requires_grad is False and out.recipe is None\n\n"
        "# RESTORE toggle\n"
        "set_grad_tracking(True)\n\n"
        "# (v) per-op flag is STICKY across calls (closure capture works)\n"
        "# After many calls, eq_nondiff still says False — there is no way to flip it without re-wrapping.\n"
        "for _ in range(5):\n"
        "    assert eq_nondiff(a, b).requires_grad is False\n\n"
        "# (vi) global toggle FLIPS add_diff back and forth at runtime (no re-wrap needed)\n"
        "set_grad_tracking(False)\n"
        "assert add_diff(a, b).requires_grad is False\n"
        "set_grad_tracking(True)\n"
        "assert add_diff(a, b).requires_grad is True\n\n"
        "# (vii) gate 3: untracked inputs always short-circuit\n"
        "u = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "v = MiniTensor(t.tensor([2.0]), requires_grad=False)\n"
        "out = add_diff(u, v)\n"
        "assert out.requires_grad is False and out.recipe is None\n\n"
        "# (viii) make_check_requires_grad doesn't cross-contaminate between factory calls\n"
        "c_diff = make_check_requires_grad(True)\n"
        "c_nondiff = make_check_requires_grad(False)\n"
        "assert c_diff((a,)) is True\n"
        "assert c_nondiff((a,)) is False\n"
        "assert c_diff((a,)) is True, 'second factory call leaked into first closure'"
    ),
    "solution_body": (
        "def set_grad_tracking(enabled: bool):\n"
        "    globals()['grad_tracking_enabled'] = enabled\n\n"
        "def make_check_requires_grad(is_differentiable: bool):\n"
        "    def check(args):\n"
        "        return (\n"
        "            globals()['grad_tracking_enabled']\n"
        "            and is_differentiable\n"
        "            and any(\n"
        "                isinstance(a, MiniTensor) and a.requires_grad for a in args\n"
        "            )\n"
        "        )\n"
        "    return check\n\n"
        "def cx17_wrap(fwd_fn, is_differentiable=True):\n"
        "    check = make_check_requires_grad(is_differentiable)\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = tuple(\n"
        "            a.array if isinstance(a, MiniTensor) else a for a in args\n"
        "        )\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        rg = check(args)\n"
        "        out = MiniTensor(out_raw, requires_grad=rg)\n"
        "        if rg:\n"
        "            parents = {\n"
        "                i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)\n"
        "            }\n"
        "            out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    "solution_notes": (
        "The independence point is the load-bearing one: turning the global toggle on does NOT re-enable a "
        "non-differentiable op. Gate 2 lives in the closure; you cannot flip it without re-wrapping."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["is-differentiable-flag", "grad-tracking-global-toggle", "requires-grad-propagation"],
    "lo": "Compose the runtime grad-tracking global toggle with the per-op is_differentiable closure flag so the wrapper short-circuits Recipe construction whenever EITHER gate is off, independent of the other.",
}
emit_composite(spec_17)


# ---------------------------------------------------------------------------
# cx18 — wrap a non-diff op (argmax); result is a leaf with no recipe
# atoms: non-diff-fn-wrap, is-differentiable-flag
# ---------------------------------------------------------------------------
spec_18 = {
    "atom_ids": ["non-diff-fn-wrap", "is-differentiable-flag"],
    "subtopics": _subs(["non-diff-fn-wrap", "is-differentiable-flag"]),
    "primary_atom": "non-diff-fn-wrap",
    "part": "part4",
    "exercise_index": 18,
    "exercise_title": "wrap argmax with is_differentiable=False; output is a graph leaf",
    "slug": "non-diff-argmax-leaf-output",
    "atom_recap_md": (
        "## Composing non-diff-fn-wrap with the is_differentiable flag\n\n"
        "Some forward ops have no useful gradient — `t.argmax` returns int64 indices, `t.eq` returns bools. "
        "We want to call them on `MiniTensor`s without:\n"
        "  - setting `requires_grad=True` on the output (no gradient could flow back).\n"
        "  - attaching a `Recipe` (the reverse pass would try to recurse past it and either crash or waste work).\n\n"
        "The fix is the `is_differentiable=False` kwarg on `wrap_forward_fn`, captured in the wrapper's closure. "
        "When the flag is False, the wrapper:\n"
        "  1. Still computes the forward result (we DO want the value).\n"
        "  2. Forces `requires_grad=False` and `recipe=None` on the output.\n"
        "  3. Makes the output behave like a graph leaf — backprop's sorted-graph walk stops there naturally."
    ),
    "prompt_body": (
        "Implement `cx18_wrap_forward_fn(fwd_fn, is_differentiable=True)`. Behavior:\n\n"
        "1. Unbox each `MiniTensor` arg to its `.array`; pass non-Tensors through.\n"
        "2. Call `fwd_fn(*raw_args, **kwargs)`.\n"
        "3. Three-gate AND: `requires_grad = grad_tracking_enabled AND is_differentiable AND any(input is tracked MiniTensor)`. "
        "Read `grad_tracking_enabled` from `globals()` FRESH each call.\n"
        "4. Build `out = MiniTensor(out_raw, requires_grad=rg)`. Only when `rg` is True, attach `out.recipe = Recipe(...)`. "
        "Otherwise leave `recipe=None`.\n\n"
        "Then wrap `t.argmax` with `is_differentiable=False` and prove the output is a graph leaf."
    ),
    "stub_body": (
        "from dataclasses import dataclass, field\n"
        "from typing import Callable, Optional\n\n"
        "grad_tracking_enabled = True\n\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Optional[Callable] = None\n"
        "    args: tuple = ()\n"
        "    kwargs: dict = field(default_factory=dict)\n"
        "    parents: dict = field(default_factory=dict)\n\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad=False, recipe=None):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = recipe\n\n"
        "def cx18_wrap_forward_fn(fwd_fn, is_differentiable=True):\n"
        "    raise NotImplementedError\n\n"
        "def cx18_is_leaf(node):\n"
        "    \"\"\"True iff this MiniTensor terminates the backward graph (recipe is None).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "globals()['grad_tracking_enabled'] = True\n\n"
        "add    = cx18_wrap_forward_fn(t.add)\n"
        "argmax = cx18_wrap_forward_fn(t.argmax, is_differentiable=False)\n"
        "eq     = cx18_wrap_forward_fn(t.eq,     is_differentiable=False)\n\n"
        "x = MiniTensor(t.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), requires_grad=True)\n\n"
        "# (a) diff op: requires_grad propagates, Recipe is built\n"
        "out_add = add(x, y)\n"
        "assert out_add.requires_grad is True\n"
        "assert out_add.recipe is not None and out_add.recipe.func is t.add\n"
        "assert cx18_is_leaf(out_add) is False, 'differentiable add must NOT be a leaf'\n\n"
        "# (b) argmax: forward value still correct, but output is a graph leaf\n"
        "out_ax = argmax(x)\n"
        "assert out_ax.array.item() == 5, f'forward value wrong: {out_ax.array}'\n"
        "assert out_ax.requires_grad is False, 'non-diff must force requires_grad=False'\n"
        "assert out_ax.recipe is None, 'non-diff must NOT build a Recipe'\n"
        "assert cx18_is_leaf(out_ax) is True, 'argmax output must be a graph leaf'\n\n"
        "# (c) argmax dtype is int64 — proves we really called torch.argmax, not a stand-in\n"
        "assert out_ax.array.dtype in (t.int64, t.long), f'argmax should return int64: {out_ax.array.dtype}'\n\n"
        "# (d) eq: bool dtype, still a leaf\n"
        "out_eq = eq(x, y)\n"
        "assert out_eq.array.dtype == t.bool\n"
        "assert out_eq.requires_grad is False and out_eq.recipe is None\n"
        "assert cx18_is_leaf(out_eq) is True\n\n"
        "# (e) is_differentiable is STICKY (closure capture). Many calls, still leaf.\n"
        "for _ in range(3):\n"
        "    assert cx18_is_leaf(argmax(x)) is True\n\n"
        "# (f) global toggle off → add ALSO becomes a leaf (gate 1 fails),\n"
        "#     but argmax stays a leaf for an INDEPENDENT reason (gate 2)\n"
        "globals()['grad_tracking_enabled'] = False\n"
        "try:\n"
        "    assert cx18_is_leaf(add(x, y)) is True, 'toggle-off makes diff op a leaf too'\n"
        "    assert cx18_is_leaf(argmax(x)) is True, 'non-diff still a leaf'\n"
        "finally:\n"
        "    globals()['grad_tracking_enabled'] = True\n\n"
        "# (g) chained call: add(argmax(x).float(), x) — but argmax's output stays a leaf\n"
        "# i.e. backprop would stop at out_ax even if a downstream op tracks through it.\n"
        "out_ax2 = argmax(x)\n"
        "assert out_ax2.recipe is None  # graph terminates here, period."
    ),
    "solution_body": (
        "def cx18_wrap_forward_fn(fwd_fn, is_differentiable=True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw_args = tuple(\n"
        "            a.array if isinstance(a, MiniTensor) else a for a in args\n"
        "        )\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        rg = (\n"
        "            globals()['grad_tracking_enabled']\n"
        "            and is_differentiable\n"
        "            and any(\n"
        "                isinstance(a, MiniTensor) and a.requires_grad for a in args\n"
        "            )\n"
        "        )\n"
        "        out = MiniTensor(out_raw, requires_grad=rg)\n"
        "        if rg:\n"
        "            parents = {\n"
        "                i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)\n"
        "            }\n"
        "            out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func\n\n"
        "def cx18_is_leaf(node):\n"
        "    return node.recipe is None"
    ),
    "solution_notes": (
        "`recipe=None` is what makes the node terminal. The sorted-graph walk in `backprop` checks `if node.recipe "
        "is None: continue` — that's the exact line that protects the reverse pass from trying to differentiate "
        "argmax. The is_differentiable closure is the input contract; recipe=None is the output contract."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["non-diff-fn-wrap", "is-differentiable-flag", "requires-grad-propagation"],
    "lo": "Compose the is_differentiable closure flag with the non-diff wrap path so wrapping argmax (or eq) produces a MiniTensor with requires_grad=False AND recipe=None, terminating the backward graph at that node.",
}
emit_composite(spec_18)


print("emitted cx13..cx18")
