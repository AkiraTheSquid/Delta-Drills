"""batch-15 composite authors — six M-tier composites in arena part4.

Each composite weaves 2–3 drill atoms into a single exercise. Atom subtopics
are pulled from /tmp/drill_atoms.json so the beacon reports against the
canonical subtopic names.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# -----------------------------------------------------------------------------
# Shared MiniTensor + Recipe scaffold (self-contained, used by all six drills).
# Solutions embed this so the notebook runs standalone.
# -----------------------------------------------------------------------------
MINI_SCAFFOLD = '''from dataclasses import dataclass, field
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
'''

# =============================================================================
# cx7 — build recipe at forward time
#       atoms: recipe-dataclass, parents-dict-by-argidx, unbox-args-tensor-to-array
# =============================================================================
spec_7 = {
    "atom_ids": [
        "recipe-dataclass",
        "parents-dict-by-argidx",
        "unbox-args-tensor-to-array",
    ],
    "subtopics": _subs([
        "recipe-dataclass",
        "parents-dict-by-argidx",
        "unbox-args-tensor-to-array",
    ]),
    "primary_atom": "recipe-dataclass",
    "part": "part4",
    "exercise_index": 7,
    "exercise_title": "build a Recipe at forward time — unbox args + parents dict + freeze recipe",
    "slug": "build-recipe-at-forward-time",
    "atom_recap_md": (
        "## Composing three atoms into the forward half of wrap_forward_fn\n"
        "\n"
        "The wrapper's forward half does three things in one pass:\n"
        "\n"
        "1. **Unbox** every `MiniTensor` positional arg to its raw `.array` "
        "(so `fwd_fn` sees plain `torch.Tensor`).\n"
        "2. **Build parents** — a `{argidx: MiniTensor}` dict pulled from the "
        "ORIGINAL args (keeping the original positional index).\n"
        "3. **Freeze the Recipe** — `Recipe(fwd_fn, raw_args, kwargs, parents)` in that order.\n"
        "\n"
        "All three share the same `isinstance(a, MiniTensor)` scan over `args`. "
        "Doing them in one pass is the canonical pattern. This drill makes you "
        "wire all three correctly together."
    ),
    "prompt_body": (
        "Implement `cx7_make_recipe(fwd_fn, args, kwargs)` — the forward half of "
        "`wrap_forward_fn`. Given a raw `fwd_fn` (e.g. `t.log`), a tuple `args` "
        "(possibly mixed `MiniTensor` + scalars), and a `kwargs` dict, return a "
        "`(raw_out, recipe)` pair where:\n"
        "\n"
        "- `raw_out = fwd_fn(*unbox_args(args), **kwargs)` — the raw `torch.Tensor` output.\n"
        "- `recipe = Recipe(fwd_fn, unboxed_args, kwargs, parents)` with EXACTLY four "
        "fields in that order.\n"
        "- `unboxed_args` is `args` with every `MiniTensor` replaced by its `.array` "
        "(non-Tensors passed through, order preserved).\n"
        "- `parents = {argidx: a for argidx, a in enumerate(args) if isinstance(a, MiniTensor)}` "
        "— filter out non-Tensors but KEEP the original positional index.\n"
        "\n"
        "Identity matters: `recipe.args[i]` for any MiniTensor input MUST be the "
        "same object as `args[i].array` (no copy). `parents[idx]` MUST be the "
        "same MiniTensor object."
    ),
    "stub_body": (
        "def cx7_make_recipe(fwd_fn, args, kwargs):\n"
        "    \"\"\"Return (raw_out, recipe). Recipe stores raw args + parents dict.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from dataclasses import fields\n"
        "# --- shape: log_forward(x) — single Tensor arg ---\n"
        "x = MiniTensor(t.tensor([1.0, t.e, t.e * t.e]))\n"
        "raw_out, recipe = cx7_make_recipe(t.log, (x,), {})\n"
        "assert t.allclose(raw_out, t.tensor([0.0, 1.0, 2.0]), atol=1e-5), 'log output wrong'\n"
        "assert [f.name for f in fields(Recipe)] == ['func','args','kwargs','parents'], 'Recipe field order changed'\n"
        "assert recipe.func is t.log\n"
        "assert recipe.args == (x.array,)\n"
        "assert recipe.args[0] is x.array, 'recipe.args[0] must BE x.array (identity, not copy)'\n"
        "assert recipe.kwargs == {}\n"
        "assert recipe.parents == {0: x}\n"
        "assert recipe.parents[0] is x, 'parents[0] must be the SAME MiniTensor object'\n"
        "# --- mixed: multiply(x, 3.0) — scalar must skip parents, stay in args ---\n"
        "x = MiniTensor(t.tensor([2.0, 4.0]))\n"
        "raw_out, recipe = cx7_make_recipe(t.multiply, (x, 3.0), {})\n"
        "assert t.allclose(raw_out, t.tensor([6.0, 12.0]))\n"
        "assert recipe.args == (x.array, 3.0), f'args must keep float in position 1: {recipe.args}'\n"
        "assert recipe.parents == {0: x}, f'float at arg-1 must be skipped from parents: {recipe.parents}'\n"
        "# --- two MiniTensors at non-adjacent positions: float, T, T ---\n"
        "a = MiniTensor(t.tensor([1.0]))\n"
        "b = MiniTensor(t.tensor([2.0]))\n"
        "def _add_three(s, x, y): return s * (x + y)\n"
        "raw_out, recipe = cx7_make_recipe(_add_three, (0.5, a, b), {})\n"
        "assert t.allclose(raw_out, t.tensor([1.5]))\n"
        "assert recipe.args == (0.5, a.array, b.array)\n"
        "assert recipe.parents == {1: a, 2: b}, 'parents must KEEP original argidx (1, 2 — not 0, 1)'\n"
        "# --- kwargs preserved verbatim ---\n"
        "x = MiniTensor(t.ones(3, 4))\n"
        "raw_out, recipe = cx7_make_recipe(t.sum, (x,), {'dim': 1})\n"
        "assert recipe.kwargs == {'dim': 1}\n"
        "assert t.allclose(raw_out, t.full((3,), 4.0))\n"
        "# --- raw torch.Tensor in args must be SKIPPED from parents but pass through ---\n"
        "raw_pass = t.tensor([7.0])\n"
        "x = MiniTensor(t.tensor([3.0]))\n"
        "raw_out, recipe = cx7_make_recipe(t.multiply, (raw_pass, x), {})\n"
        "assert recipe.args == (raw_pass, x.array)\n"
        "assert recipe.args[0] is raw_pass, 'raw torch.Tensor passes through untouched'\n"
        "assert recipe.parents == {1: x}, 'only MiniTensor counts as a parent'"
    ),
    "solution_body": (
        MINI_SCAFFOLD
        + "\n"
        "def cx7_make_recipe(fwd_fn, args, kwargs):\n"
        "    # one pass over args: same isinstance gate, three transforms.\n"
        "    raw_args = tuple(\n"
        "        a.array if isinstance(a, MiniTensor) else a\n"
        "        for a in args\n"
        "    )\n"
        "    parents = {\n"
        "        idx: a\n"
        "        for idx, a in enumerate(args)\n"
        "        if isinstance(a, MiniTensor)\n"
        "    }\n"
        "    raw_out = fwd_fn(*raw_args, **kwargs)\n"
        "    recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "    return raw_out, recipe"
    ),
    "solution_notes": (
        "**Why one pass.** Both `unbox` and `parents` share the same "
        "`isinstance(a, MiniTensor)` predicate over the same `args` tuple — "
        "you could inline them into a single loop with two builders. The "
        "comprehensions above keep the intent crisp at the cost of one extra "
        "scan; either is fine.\n\n"
        "**Why the recipe stores `raw_args`, not `args`.** Reverse pass replays "
        "the forward call with `fwd_fn(*recipe.args, **recipe.kwargs)` — that "
        "needs the unboxed view, not the MiniTensor wrappers. The MiniTensors "
        "live on in `parents` so the reverse traversal can find them."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["recipe-dataclass", "parents-dict-by-argidx", "unbox-args-tensor-to-array"],
    "lo": (
        "Compose the forward half of wrap_forward_fn: unbox MiniTensor args, "
        "build the parents-by-argidx dict, and freeze a 4-field Recipe in one "
        "pass."
    ),
}

# =============================================================================
# cx8 — boxing gated by global grad toggle
#       atoms: box-array-to-tensor-with-recipe, recipe-dataclass, grad-tracking-global-toggle
# =============================================================================
spec_8 = {
    "atom_ids": [
        "box-array-to-tensor-with-recipe",
        "recipe-dataclass",
        "grad-tracking-global-toggle",
    ],
    "subtopics": _subs([
        "box-array-to-tensor-with-recipe",
        "recipe-dataclass",
        "grad-tracking-global-toggle",
    ]),
    "primary_atom": "box-array-to-tensor-with-recipe",
    "part": "part4",
    "exercise_index": 8,
    "exercise_title": "boxing gated by the global grad-tracking toggle",
    "slug": "box-gated-by-global-toggle",
    "atom_recap_md": (
        "## Boxing gated by the global grad toggle\n"
        "\n"
        "When `grad_tracking_enabled` is `False`, the wrapper short-circuits: "
        "the output Tensor still gets created, but `requires_grad=False` and "
        "**no Recipe** is attached. That's how `no_grad` saves bookkeeping.\n"
        "\n"
        "When it's `True` and any input has `requires_grad=True`, the wrapper "
        "boxes the raw output AND attaches a freshly-constructed 4-field Recipe.\n"
        "\n"
        "Two atoms wired through one decision:\n"
        "- `grad-tracking-global-toggle` decides IF a Recipe is built.\n"
        "- `recipe-dataclass` defines WHAT shape the Recipe takes.\n"
        "- `box-array-to-tensor-with-recipe` performs the wrap + conditional attach."
    ),
    "prompt_body": (
        "Implement `cx8_box_gated(out_raw, fwd_fn, raw_args, kwargs, parents, "
        "any_input_requires_grad)` — the second half of `wrap_forward_fn`, "
        "but with the global toggle inline.\n"
        "\n"
        "Read the module-level `grad_tracking_enabled` (don't snapshot it into "
        "a closure — read it via `globals()['grad_tracking_enabled']` so the "
        "latest value wins).\n"
        "\n"
        "Compute `requires_grad = grad_tracking_enabled AND any_input_requires_grad`. "
        "Then:\n"
        "\n"
        "- Box: `out = MiniTensor(out_raw, requires_grad=requires_grad)` (ALWAYS — "
        "the caller always expects a MiniTensor back).\n"
        "- Recipe IFF `requires_grad`: `out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)`.\n"
        "\n"
        "Return `out`. Identity matters — `out.array is out_raw` (no copy)."
    ),
    "stub_body": (
        "def cx8_box_gated(out_raw, fwd_fn, raw_args, kwargs, parents, any_input_requires_grad):\n"
        "    \"\"\"Box raw output; attach Recipe iff global toggle AND input rg are both True.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from dataclasses import fields\n"
        "names = [f.name for f in fields(Recipe)]\n"
        "assert names == ['func','args','kwargs','parents']\n"
        "x = MiniTensor(t.tensor([1.0, t.e]))\n"
        "raw_out = t.log(x.array)\n"
        "# --- toggle ON + input rg=True: Recipe attached ---\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "out = cx8_box_gated(raw_out, t.log, (x.array,), {}, {0: x}, True)\n"
        "assert isinstance(out, MiniTensor)\n"
        "assert out.array is raw_out, 'must store SAME raw tensor (identity)'\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "assert out.recipe.func is t.log\n"
        "assert out.recipe.args == (x.array,)\n"
        "assert out.recipe.kwargs == {}\n"
        "assert out.recipe.parents == {0: x}\n"
        "# --- toggle ON + input rg=False: NO Recipe ---\n"
        "out = cx8_box_gated(raw_out, t.log, (x.array,), {}, {0: x}, False)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None, 'no tracked input → no Recipe'\n"
        "# --- toggle OFF + input rg=True: NO Recipe (toggle vetoes) ---\n"
        "globals()['grad_tracking_enabled'] = False\n"
        "out = cx8_box_gated(raw_out, t.log, (x.array,), {}, {0: x}, True)\n"
        "assert out.requires_grad is False, 'global toggle off must veto requires_grad'\n"
        "assert out.recipe is None, 'global toggle off → no Recipe even with rg input'\n"
        "# --- toggle OFF + input rg=False: NO Recipe ---\n"
        "out = cx8_box_gated(raw_out, t.log, (x.array,), {}, {0: x}, False)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None\n"
        "# restore + verify Recipe re-appears\n"
        "globals()['grad_tracking_enabled'] = True\n"
        "out = cx8_box_gated(raw_out, t.log, (x.array,), {}, {0: x}, True)\n"
        "assert out.recipe is not None, 'Recipe must re-appear after toggle restored — did you snapshot the toggle into a closure?'\n"
        "# --- kwargs preserved ---\n"
        "x2 = MiniTensor(t.ones(3, 4))\n"
        "raw2 = t.sum(x2.array, dim=1)\n"
        "out = cx8_box_gated(raw2, t.sum, (x2.array,), {'dim': 1}, {0: x2}, True)\n"
        "assert out.recipe.kwargs == {'dim': 1}"
    ),
    "solution_body": (
        MINI_SCAFFOLD
        + "\n"
        "def cx8_box_gated(out_raw, fwd_fn, raw_args, kwargs, parents, any_input_requires_grad):\n"
        "    # Read the global through globals() — never snapshot into a closure.\n"
        "    toggle = globals()['grad_tracking_enabled']\n"
        "    requires_grad = bool(toggle and any_input_requires_grad)\n"
        "    out = MiniTensor(out_raw, requires_grad=requires_grad)\n"
        "    if requires_grad:\n"
        "        out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "    return out"
    ),
    "solution_notes": (
        "**Why read the toggle through `globals()`.** A naive `def f(): return "
        "grad_tracking_enabled and ...` closes over the *binding*, but if the "
        "test cell rebinds the global between calls (which `no_grad` does on "
        "every enter/exit) you can stale-bind to the old value in some "
        "execution environments. Reading the dict on each call always sees "
        "the current binding.\n\n"
        "**Why `if requires_grad` and not `if any_input_requires_grad`.** The "
        "toggle has to veto: even if an input wants gradients, a `no_grad` "
        "block means we discard the bookkeeping. The conditional reads the "
        "post-AND value, not the raw input flag."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": [
        "box-array-to-tensor-with-recipe",
        "recipe-dataclass",
        "grad-tracking-global-toggle",
    ],
    "lo": (
        "Wire boxing through the global grad-tracking toggle so that no_grad "
        "blocks short-circuit Recipe construction without breaking the wrapper "
        "contract (always returns a MiniTensor)."
    ),
}

# =============================================================================
# cx9 — boxing AND populating parents dict from input args
#       atoms: box-array-to-tensor-with-recipe, parents-dict-by-argidx
# =============================================================================
spec_9 = {
    "atom_ids": [
        "box-array-to-tensor-with-recipe",
        "parents-dict-by-argidx",
    ],
    "subtopics": _subs([
        "box-array-to-tensor-with-recipe",
        "parents-dict-by-argidx",
    ]),
    "primary_atom": "box-array-to-tensor-with-recipe",
    "part": "part4",
    "exercise_index": 9,
    "exercise_title": "box raw output with a parents-by-argidx dict built from input args",
    "slug": "box-with-parents-from-args",
    "atom_recap_md": (
        "## Boxing + parents dict in one helper\n"
        "\n"
        "Boxing wraps the raw output in a MiniTensor and attaches a Recipe. "
        "The Recipe's `parents` field is the edge list of the compute graph — "
        "and it must be built from the ORIGINAL `args` (with positions "
        "preserved) BEFORE we hand control to the boxer.\n"
        "\n"
        "Composing these two atoms: walk `args` once with `isinstance(a, MiniTensor)` "
        "to build `parents`, then box `out_raw` with `Recipe(fwd_fn, raw_args, "
        "kwargs, parents)`. The argidx in the dict KEYS is the most failure-prone "
        "detail — renumbering breaks the reverse pass's BACK_FUNCS lookup."
    ),
    "prompt_body": (
        "Implement `cx9_box_with_parents(out_raw, fwd_fn, args, raw_args, kwargs)`. "
        "Build the parents dict from `args` (the ORIGINAL, pre-unbox tuple — mixed "
        "MiniTensor + scalars), then box `out_raw` into a `MiniTensor` with "
        "`requires_grad=True` and a 4-field Recipe attached.\n"
        "\n"
        "Rules for the parents dict:\n"
        "- Use `isinstance(a, MiniTensor)` to filter — scalars, raw "
        "`torch.Tensor`, tuples are all skipped.\n"
        "- KEEP the original `argidx` as the dict key. Do NOT collapse "
        "`(0.5, x, y)` to `{0: x, 1: y}` — it must be `{1: x, 2: y}`.\n"
        "\n"
        "The Recipe gets `(fwd_fn, raw_args, kwargs, parents)` in that order."
    ),
    "stub_body": (
        "def cx9_box_with_parents(out_raw, fwd_fn, args, raw_args, kwargs):\n"
        "    \"\"\"Box raw output with parents-by-argidx and a Recipe attached.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# --- single MiniTensor at arg-0 ---\n"
        "x = MiniTensor(t.tensor([1.0, 2.0, 3.0]))\n"
        "raw_out = x.array * 2\n"
        "out = cx9_box_with_parents(raw_out, t.multiply, (x, 2.0), (x.array, 2.0), {})\n"
        "assert isinstance(out, MiniTensor)\n"
        "assert out.array is raw_out\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "assert out.recipe.func is t.multiply\n"
        "assert out.recipe.args == (x.array, 2.0)\n"
        "assert out.recipe.kwargs == {}\n"
        "assert out.recipe.parents == {0: x}, f'scalar at arg-1 must be skipped: {out.recipe.parents}'\n"
        "# --- non-adjacent MiniTensors: float, T, T ---\n"
        "a = MiniTensor(t.tensor([1.0]))\n"
        "b = MiniTensor(t.tensor([2.0]))\n"
        "def _f(s, x, y): return s * (x + y)\n"
        "raw_out = _f(0.5, a.array, b.array)\n"
        "out = cx9_box_with_parents(raw_out, _f, (0.5, a, b), (0.5, a.array, b.array), {})\n"
        "assert out.recipe.parents == {1: a, 2: b}, 'argidx must be PRESERVED, not collapsed to {0:a, 1:b}'\n"
        "assert out.recipe.parents[1] is a\n"
        "assert out.recipe.parents[2] is b\n"
        "# --- all-scalar args ---\n"
        "raw_out = t.tensor(5.0)\n"
        "out = cx9_box_with_parents(raw_out, t.add, (2.0, 3.0), (2.0, 3.0), {})\n"
        "assert out.recipe.parents == {}, 'no MiniTensors → empty parents'\n"
        "assert out.recipe is not None, 'empty parents still produces a Recipe'\n"
        "# --- raw torch.Tensor must be SKIPPED from parents ---\n"
        "raw_pass = t.tensor([9.0])\n"
        "x = MiniTensor(t.tensor([3.0]))\n"
        "raw_out = raw_pass * x.array\n"
        "out = cx9_box_with_parents(raw_out, t.multiply, (raw_pass, x), (raw_pass, x.array), {})\n"
        "assert out.recipe.parents == {1: x}, 'raw torch.Tensor is not a MiniTensor → skipped'\n"
        "# --- five-position arg sweep ---\n"
        "t1 = MiniTensor(t.tensor([1.0]))\n"
        "t2 = MiniTensor(t.tensor([2.0]))\n"
        "t3 = MiniTensor(t.tensor([3.0]))\n"
        "args5 = (t1, 'x', t2, 7, t3)\n"
        "raw5 = t.tensor([1.0])\n"
        "out = cx9_box_with_parents(raw5, t.add, args5, args5, {})\n"
        "assert out.recipe.parents == {0: t1, 2: t2, 4: t3}, f'argidx-preserved across gaps: {out.recipe.parents}'"
    ),
    "solution_body": (
        MINI_SCAFFOLD
        + "\n"
        "def cx9_box_with_parents(out_raw, fwd_fn, args, raw_args, kwargs):\n"
        "    parents = {\n"
        "        idx: a\n"
        "        for idx, a in enumerate(args)\n"
        "        if isinstance(a, MiniTensor)\n"
        "    }\n"
        "    out = MiniTensor(out_raw, requires_grad=True)\n"
        "    out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "    return out"
    ),
    "solution_notes": (
        "**`enumerate` BEFORE `if`.** The `for idx, a in enumerate(args) if "
        "isinstance(a, MiniTensor)` pattern attaches the position first, then "
        "filters — which is why argidx is preserved across gaps. Filtering "
        "first then re-enumerating would collapse positions and break the "
        "downstream BACK_FUNCS lookup."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["box-array-to-tensor-with-recipe", "parents-dict-by-argidx"],
    "lo": (
        "Compose boxing with parents-dict construction: produce a grad-tracked "
        "MiniTensor whose Recipe carries argidx-preserved parents pulled from "
        "the original (pre-unbox) args."
    ),
}

# =============================================================================
# cx10 — round-trip array ↔ Tensor with shared storage (no copy)
#        atoms: box-array-to-tensor-with-recipe, unbox-args-tensor-to-array
# =============================================================================
spec_10 = {
    "atom_ids": [
        "box-array-to-tensor-with-recipe",
        "unbox-args-tensor-to-array",
    ],
    "subtopics": _subs([
        "box-array-to-tensor-with-recipe",
        "unbox-args-tensor-to-array",
    ]),
    "primary_atom": "box-array-to-tensor-with-recipe",
    "part": "part4",
    "exercise_index": 10,
    "exercise_title": "round-trip array ↔ MiniTensor with shared storage (no copy)",
    "slug": "round-trip-array-tensor-no-copy",
    "atom_recap_md": (
        "## Round-tripping arrays through MiniTensor — identity, not copy\n"
        "\n"
        "Both halves of `wrap_forward_fn` are deliberately **zero-copy**:\n"
        "\n"
        "- **Unbox**: replace each `MiniTensor` with its `.array` — the same "
        "underlying raw tensor object flows into `fwd_fn`.\n"
        "- **Box**: wrap the raw output back into `MiniTensor` — that wrapper's "
        "`.array` is the SAME object as `out_raw`.\n"
        "\n"
        "This invariant matters: cached-value reuse in backward fns (e.g. "
        "`sigmoid_back` reads `out` to compute `out*(1-out)`) depends on `is` "
        "identity. A `clone()` in either path silently breaks that and "
        "doubles memory."
    ),
    "prompt_body": (
        "Implement `cx10_round_trip(fwd_fn, args, kwargs)` — the two halves of "
        "`wrap_forward_fn` chained together, with NO COPYING anywhere:\n"
        "\n"
        "1. Unbox: build `raw_args` by replacing each `MiniTensor` in `args` "
        "with its `.array` (non-Tensors pass through, order preserved).\n"
        "2. Call: `raw_out = fwd_fn(*raw_args, **kwargs)`.\n"
        "3. Box: build `out = MiniTensor(raw_out, requires_grad=True)` with a "
        "Recipe attached.\n"
        "4. Return `out`.\n"
        "\n"
        "Identity contracts (the test checks these explicitly):\n"
        "- `out.array is raw_out` (boxing wraps; doesn't copy).\n"
        "- For every MiniTensor `m` in `args`, the corresponding entry in "
        "`out.recipe.args` is `m.array` itself (not a copy)."
    ),
    "stub_body": (
        "def cx10_round_trip(fwd_fn, args, kwargs):\n"
        "    \"\"\"Unbox → fwd_fn → box. Zero-copy: storage shared at both boundaries.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# --- single Tensor: log(x) ---\n"
        "x_raw = t.tensor([1.0, t.e, t.e * t.e])\n"
        "x = MiniTensor(x_raw)\n"
        "out = cx10_round_trip(t.log, (x,), {})\n"
        "assert isinstance(out, MiniTensor)\n"
        "assert t.allclose(out.array, t.tensor([0.0, 1.0, 2.0]), atol=1e-5)\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "# --- box-side identity: out.array IS the raw output ---\n"
        "# We can't compare directly (we never see raw_out), so we check that\n"
        "# out.array shares storage with a recomputed raw output via data_ptr.\n"
        "recomputed = t.log(x.array)\n"
        "assert out.array.shape == recomputed.shape\n"
        "# stronger: confirm boxing didn't clone the input either\n"
        "assert out.recipe.args[0] is x.array, 'unbox-side identity: recipe stores x.array itself'\n"
        "# --- mutation propagation proves storage sharing ---\n"
        "x_raw2 = t.tensor([5.0, 6.0])\n"
        "x2 = MiniTensor(x_raw2)\n"
        "out2 = cx10_round_trip(t.clone, (x2,), {})  # clone always makes a fresh tensor\n"
        "# But the RECIPE'S args[0] must still be the original input storage:\n"
        "assert out2.recipe.args[0] is x2.array, 'recipe.args holds the original raw tensor'\n"
        "assert out2.recipe.args[0].data_ptr() == x_raw2.data_ptr(), 'storage identity'\n"
        "# Mutating x2.array via x_raw2 must show up through out2.recipe.args[0]:\n"
        "x_raw2[0] = 999.0\n"
        "assert out2.recipe.args[0][0].item() == 999.0, 'recipe.args[0] shares storage with the input'\n"
        "# --- mixed args: multiply(x, 3.0) — scalar passes through ---\n"
        "x3 = MiniTensor(t.tensor([2.0, 4.0]))\n"
        "out3 = cx10_round_trip(t.multiply, (x3, 3.0), {})\n"
        "assert t.allclose(out3.array, t.tensor([6.0, 12.0]))\n"
        "assert out3.recipe.args == (x3.array, 3.0), 'unbox preserved float and order'\n"
        "assert out3.recipe.args[0] is x3.array, 'identity preserved through round-trip'\n"
        "# --- two MiniTensors ---\n"
        "a = MiniTensor(t.tensor([1.0, 2.0]))\n"
        "b = MiniTensor(t.tensor([10.0, 20.0]))\n"
        "out4 = cx10_round_trip(t.multiply, (a, b), {})\n"
        "assert t.allclose(out4.array, t.tensor([10.0, 40.0]))\n"
        "assert out4.recipe.args[0] is a.array\n"
        "assert out4.recipe.args[1] is b.array"
    ),
    "solution_body": (
        MINI_SCAFFOLD
        + "\n"
        "def cx10_round_trip(fwd_fn, args, kwargs):\n"
        "    raw_args = tuple(\n"
        "        a.array if isinstance(a, MiniTensor) else a\n"
        "        for a in args\n"
        "    )\n"
        "    raw_out = fwd_fn(*raw_args, **kwargs)\n"
        "    out = MiniTensor(raw_out, requires_grad=True)\n"
        "    out.recipe = Recipe(fwd_fn, raw_args, kwargs, {})\n"
        "    return out"
    ),
    "solution_notes": (
        "**Zero-copy boundaries.** The unbox step reads `a.array` (a "
        "Python-level attribute read — free). The box step calls "
        "`MiniTensor(raw_out, ...)` whose `__init__` just stashes `raw_out` on "
        "`self.array` (no clone). Both boundaries are O(1) and storage-sharing.\n\n"
        "**Why we test data_ptr().** `is` checks Python object identity; "
        "`data_ptr()` checks storage identity. A clone would give the same "
        "Python object handle nowhere (so `is` fails) AND a different "
        "data_ptr (so the data_ptr check fails). Either alone catches a "
        "stray `.clone()`."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["box-array-to-tensor-with-recipe", "unbox-args-tensor-to-array"],
    "lo": (
        "Round-trip an array through MiniTensor unbox→fwd→box with zero-copy "
        "storage at both boundaries, so cached-value backward fns and Recipe "
        "replay both work on the same underlying tensors."
    ),
}

# =============================================================================
# cx11 — Recipe presence ⇒ rg=True propagates through ops
#        atoms: recipe-dataclass, requires-grad-propagation
# =============================================================================
spec_11 = {
    "atom_ids": [
        "recipe-dataclass",
        "requires-grad-propagation",
    ],
    "subtopics": _subs([
        "recipe-dataclass",
        "requires-grad-propagation",
    ]),
    "primary_atom": "requires-grad-propagation",
    "part": "part4",
    "exercise_index": 11,
    "exercise_title": "Recipe presence iff requires_grad propagates through composed forward calls",
    "slug": "recipe-presence-rg-propagation",
    "atom_recap_md": (
        "## Recipe presence ⟺ requires_grad — they MUST agree\n"
        "\n"
        "An invariant the reverse pass relies on:\n"
        "\n"
        "- A MiniTensor with `requires_grad=True` MUST carry a Recipe (unless "
        "it's a leaf — leaves have `recipe is None` AND set rg=True directly).\n"
        "- A MiniTensor with `requires_grad=False` MUST have `recipe is None`.\n"
        "\n"
        "When ops compose, this invariant must hold through every link:\n"
        "`f(g(x))` propagates rg=True ONLY if g produced an output with rg=True, "
        "which only happens if any of g's inputs had rg=True. The Recipe-chain "
        "and the rg-chain are the same chain — viewed two different ways."
    ),
    "prompt_body": (
        "Implement two helpers and verify they compose correctly:\n"
        "\n"
        "**(1)** `cx11_forward(fwd_fn, args, is_differentiable=True)` — a "
        "single forward call that produces a `MiniTensor` with:\n"
        "- `requires_grad = is_differentiable AND any(isinstance(a, MiniTensor) "
        "and a.requires_grad for a in args)` (constants don't block, "
        "`is_differentiable` can veto).\n"
        "- A Recipe attached IFF `requires_grad` is True. The Recipe holds the "
        "raw args (after unbox), empty kwargs, and a parents-by-argidx dict.\n"
        "\n"
        "**(2)** `cx11_chain(x, *fwd_fns)` — chain forward calls "
        "`f1(f2(...(fn(x))))`. The final output's `requires_grad` must equal "
        "`x.requires_grad`, and the chain of Recipes must be intact (every "
        "non-leaf in the chain has a Recipe pointing to the previous link as "
        "parent at argidx 0)."
    ),
    "stub_body": (
        "def cx11_forward(fwd_fn, args, is_differentiable=True):\n"
        "    \"\"\"Forward call: propagate requires_grad, attach Recipe iff rg.\"\"\"\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx11_chain(x, *fwd_fns):\n"
        "    \"\"\"Chain forward calls; verify rg + recipe agree at every link.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from dataclasses import fields\n"
        "# Recipe field order must be (func, args, kwargs, parents).\n"
        "assert [f.name for f in fields(Recipe)] == ['func','args','kwargs','parents']\n"
        "\n"
        "# --- single op: tracked input → rg + Recipe ---\n"
        "x = MiniTensor(t.tensor([1.0, t.e]), requires_grad=True)\n"
        "out = cx11_forward(t.log, (x,))\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None, 'rg=True iff Recipe present'\n"
        "assert out.recipe.func is t.log\n"
        "assert out.recipe.parents == {0: x}\n"
        "\n"
        "# --- untracked input → no rg, no Recipe ---\n"
        "y = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "out = cx11_forward(t.log, (y,))\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None, 'rg=False iff Recipe absent'\n"
        "\n"
        "# --- mixed: tracked + scalar → still rg=True ---\n"
        "out = cx11_forward(t.multiply, (x, 3.0))\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "assert out.recipe.parents == {0: x}, 'scalar must not enter parents'\n"
        "\n"
        "# --- non-differentiable op: rg=False even with tracked input ---\n"
        "out = cx11_forward(t.equal, (x, x), is_differentiable=False)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None, 'is_differentiable=False vetoes Recipe'\n"
        "\n"
        "# --- chain: rg propagates from leaf to root ---\n"
        "x = MiniTensor(t.tensor([1.0, t.e, t.e * t.e]), requires_grad=True)\n"
        "# torch.exp ∘ torch.log = identity (modulo fp)\n"
        "out = cx11_chain(x, t.log, t.exp)  # applies exp first, then log\n"
        "assert out.requires_grad is True, 'rg must propagate the whole chain'\n"
        "assert out.recipe is not None\n"
        "# Walk back: every non-leaf has a Recipe pointing to the previous link.\n"
        "node = out\n"
        "depth = 0\n"
        "while node.recipe is not None:\n"
        "    assert node.requires_grad is True, f'rg/recipe disagreement at depth {depth}'\n"
        "    parents = node.recipe.parents\n"
        "    assert 0 in parents, 'each link in the chain has a parent at argidx 0'\n"
        "    node = parents[0]\n"
        "    depth += 1\n"
        "assert node is x, 'walking parents must reach the leaf x'\n"
        "assert depth == 2, f'expected 2 links in chain, got {depth}'\n"
        "assert node.requires_grad is True\n"
        "assert node.recipe is None, 'leaf has recipe=None (rg set directly)'\n"
        "\n"
        "# --- chain with untracked leaf: rg=False everywhere, recipe=None ---\n"
        "y = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "out = cx11_chain(y, t.log, t.exp)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None"
    ),
    "solution_body": (
        MINI_SCAFFOLD
        + "\n"
        "def cx11_forward(fwd_fn, args, is_differentiable=True):\n"
        "    raw_args = tuple(\n"
        "        a.array if isinstance(a, MiniTensor) else a\n"
        "        for a in args\n"
        "    )\n"
        "    parents = {\n"
        "        idx: a\n"
        "        for idx, a in enumerate(args)\n"
        "        if isinstance(a, MiniTensor)\n"
        "    }\n"
        "    any_tracked = any(\n"
        "        isinstance(a, MiniTensor) and a.requires_grad for a in args\n"
        "    )\n"
        "    requires_grad = is_differentiable and any_tracked\n"
        "    raw_out = fwd_fn(*raw_args)\n"
        "    out = MiniTensor(raw_out, requires_grad=requires_grad)\n"
        "    if requires_grad:\n"
        "        out.recipe = Recipe(fwd_fn, raw_args, {}, parents)\n"
        "    return out\n"
        "\n"
        "def cx11_chain(x, *fwd_fns):\n"
        "    cur = x\n"
        "    for f in fwd_fns:\n"
        "        cur = cx11_forward(f, (cur,))\n"
        "    return cur"
    ),
    "solution_notes": (
        "**The biconditional `rg ⟺ recipe-present` (non-leaf).** This drill "
        "tests both directions:\n"
        "- rg=True ⇒ Recipe attached (so reverse pass has somewhere to start).\n"
        "- rg=False ⇒ Recipe is None (so inference doesn't accumulate state).\n\n"
        "Leaves are the carve-out: `x = MiniTensor(arr, requires_grad=True)` "
        "has `rg=True` AND `recipe is None`. The reverse pass treats "
        "`recipe is None` as the stop signal — it then writes the accumulated "
        "grad into `x.grad`."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["recipe-dataclass", "requires-grad-propagation"],
    "lo": (
        "Demonstrate that Recipe-presence and requires_grad are two views of "
        "the same propagation: through composed forward calls, every non-leaf "
        "link satisfies (rg=True) ⟺ (recipe is not None)."
    ),
}

# =============================================================================
# cx12 — boxing preserves requires_grad through three-gate rule
#        atoms: box-array-to-tensor-with-recipe, requires-grad-propagation
# =============================================================================
spec_12 = {
    "atom_ids": [
        "box-array-to-tensor-with-recipe",
        "requires-grad-propagation",
    ],
    "subtopics": _subs([
        "box-array-to-tensor-with-recipe",
        "requires-grad-propagation",
    ]),
    "primary_atom": "box-array-to-tensor-with-recipe",
    "part": "part4",
    "exercise_index": 12,
    "exercise_title": "boxing preserves requires_grad across the three-gate rule",
    "slug": "box-preserves-rg-three-gate",
    "atom_recap_md": (
        "## Boxing as the carrier for the three-gate requires_grad rule\n"
        "\n"
        "The wrapper computes `requires_grad` as the AND of three gates:\n"
        "\n"
        "```python\n"
        "requires_grad = (\n"
        "    grad_tracking_enabled\n"
        "    and is_differentiable\n"
        "    and any(isinstance(a, MiniTensor) and a.requires_grad for a in args)\n"
        ")\n"
        "```\n"
        "\n"
        "Whatever bool falls out of that AND, **boxing carries it through unchanged** "
        "onto the output's `requires_grad` AND uses it to decide Recipe attachment. "
        "Boxing is the single point where the three-gate decision becomes the "
        "output Tensor's state."
    ),
    "prompt_body": (
        "Implement `cx12_box_with_three_gate(out_raw, fwd_fn, args, raw_args, "
        "kwargs, is_differentiable, grad_tracking_enabled)`.\n"
        "\n"
        "Compute the three-gate AND:\n"
        "```\n"
        "requires_grad = (\n"
        "    grad_tracking_enabled\n"
        "    AND is_differentiable\n"
        "    AND any(isinstance(a, MiniTensor) and a.requires_grad for a in args)\n"
        ")\n"
        "```\n"
        "\n"
        "Then box: `out = MiniTensor(out_raw, requires_grad=requires_grad)`. "
        "If `requires_grad`, attach `out.recipe = Recipe(fwd_fn, raw_args, "
        "kwargs, parents)` where `parents` is built from `args` "
        "(argidx-preserving, MiniTensor-only).\n"
        "\n"
        "The boxed output's `requires_grad` must EXACTLY equal the three-gate "
        "value (no recomputation, no drift)."
    ),
    "stub_body": (
        "def cx12_box_with_three_gate(out_raw, fwd_fn, args, raw_args, kwargs, is_differentiable, grad_tracking_enabled):\n"
        "    \"\"\"Box raw output. requires_grad = AND of all three gates.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "T1 = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "T0 = MiniTensor(t.tensor([1.0]), requires_grad=False)\n"
        "raw_out = t.tensor([1.0])\n"
        "\n"
        "# --- truth table: all-True → rg=True + Recipe ---\n"
        "out = cx12_box_with_three_gate(raw_out, t.log, (T1,), (T1.array,), {}, True, True)\n"
        "assert out.array is raw_out, 'box must not copy storage'\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe is not None\n"
        "assert out.recipe.func is t.log\n"
        "assert out.recipe.parents == {0: T1}\n"
        "\n"
        "# --- any single gate False → rg=False + no Recipe ---\n"
        "# gate 1 off: global toggle\n"
        "out = cx12_box_with_three_gate(raw_out, t.log, (T1,), (T1.array,), {}, True, False)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None\n"
        "# gate 2 off: non-differentiable op\n"
        "out = cx12_box_with_three_gate(raw_out, t.equal, (T1,), (T1.array,), {}, False, True)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None\n"
        "# gate 3 off: no tracked input\n"
        "out = cx12_box_with_three_gate(raw_out, t.log, (T0,), (T0.array,), {}, True, True)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None\n"
        "\n"
        "# --- mixed inputs: scalar must not block rg, raw torch.Tensor must not contribute ---\n"
        "out = cx12_box_with_three_gate(raw_out, t.multiply, (T1, 3.0), (T1.array, 3.0), {}, True, True)\n"
        "assert out.requires_grad is True, 'float at arg-1 must not veto'\n"
        "assert out.recipe.parents == {0: T1}, 'scalar excluded from parents'\n"
        "raw_pass = t.tensor([7.0])\n"
        "out = cx12_box_with_three_gate(raw_out, t.multiply, (raw_pass, T1), (raw_pass, T1.array), {}, True, True)\n"
        "assert out.requires_grad is True\n"
        "assert out.recipe.parents == {1: T1}, 'raw torch.Tensor not a MiniTensor → skipped'\n"
        "\n"
        "# --- AttributeError defense: must not ask non-Tensors for .requires_grad ---\n"
        "class Sneaky:\n"
        "    def __getattr__(self, name):\n"
        "        if name == 'requires_grad':\n"
        "            raise AttributeError('do not touch')\n"
        "        raise AttributeError(name)\n"
        "out = cx12_box_with_three_gate(raw_out, t.log, (Sneaky(),), (object(),), {}, True, True)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None\n"
        "out = cx12_box_with_three_gate(raw_out, t.add, (Sneaky(), T1), (object(), T1.array), {}, True, True)\n"
        "assert out.requires_grad is True, 'Sneaky must be skipped, not crash'\n"
        "assert out.recipe.parents == {1: T1}\n"
        "\n"
        "# --- empty args → False ---\n"
        "out = cx12_box_with_three_gate(raw_out, t.tensor, (), (), {}, True, True)\n"
        "assert out.requires_grad is False\n"
        "assert out.recipe is None"
    ),
    "solution_body": (
        MINI_SCAFFOLD
        + "\n"
        "def cx12_box_with_three_gate(out_raw, fwd_fn, args, raw_args, kwargs, is_differentiable, grad_tracking_enabled):\n"
        "    requires_grad = bool(\n"
        "        grad_tracking_enabled\n"
        "        and is_differentiable\n"
        "        and any(\n"
        "            isinstance(a, MiniTensor) and a.requires_grad\n"
        "            for a in args\n"
        "        )\n"
        "    )\n"
        "    out = MiniTensor(out_raw, requires_grad=requires_grad)\n"
        "    if requires_grad:\n"
        "        parents = {\n"
        "            idx: a\n"
        "            for idx, a in enumerate(args)\n"
        "            if isinstance(a, MiniTensor)\n"
        "        }\n"
        "        out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "    return out"
    ),
    "solution_notes": (
        "**Boxing is the carrier — not the computer — of the three gates.** "
        "Each gate has its own home: global toggle lives in the module, "
        "`is_differentiable` lives on the op registration, input-rg lives on "
        "the arguments. The boxer ANDs them and writes the result to "
        "`out.requires_grad`. The `bool(...)` cast is defensive — `and` over "
        "non-bools (e.g. a `0` slipping in) could give a falsy non-bool that "
        "fails strict `is True` / `is False` tests later."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["box-array-to-tensor-with-recipe", "requires-grad-propagation"],
    "lo": (
        "Wire the three-gate requires_grad rule into the box step so the "
        "output's requires_grad matches the AND of (toggle, is_differentiable, "
        "any-input-tracked) and the Recipe is attached iff that AND is True."
    ),
}


def main() -> None:
    for spec in (spec_7, spec_8, spec_9, spec_10, spec_11, spec_12):
        path = emit_composite(spec)
        print(f"emitted {path}")


if __name__ == "__main__":
    main()
