"""Composite drills cx25..cx30 — batch-15 (P-cell).

Six composite procedural drills exercising 2-atom pairs from the backprop /
autograd machinery (ARENA part 4). Each composite forces the learner to
apply both atoms together in ONE function.

cx25  end-grad-default-ones-like + backprop-pop-outgrad-loop
cx26  matmul-back-transpose-pair + arg-position-back-functions
cx27  sum-and-broadcast-duality + unbroadcast-pattern
cx28  max-back-tied-half + arg-position-back-functions
cx29  multiply-back + chain-rule-elementwise
cx30  unbroadcast-pattern + sum-and-broadcast-duality
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
# Shared inline scaffolding used by cx25's solution / test (MiniTensor+Recipe).
# Kept here as a string so the solution_body cell is self-contained.
# ---------------------------------------------------------------------------
MINI_TENSOR_PRELUDE = '''from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class Recipe:
    func: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)

class MiniTensor:
    def __init__(self, array, requires_grad=False, recipe=None):
        self.array = array
        self.requires_grad = requires_grad
        self.recipe = recipe
        self.grad = None
'''


# ===========================================================================
# cx25 — seed reverse pass with ones_like(loss) when end_grad is None
# ===========================================================================
spec_25 = {
    "atom_ids": ["end-grad-default-ones-like", "backprop-pop-outgrad-loop"],
    "subtopics": _subs(["end-grad-default-ones-like", "backprop-pop-outgrad-loop"]),
    "primary_atom": "end-grad-default-ones-like",
    "part": "part4",
    "exercise_index": 25,
    "exercise_title": "seed reverse pass with ones_like when end_grad is None",
    "slug": "end-grad-seeded-backprop-loop",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's `backprop(end_node, end_grad, sorted_graph, back_funcs)` is the reverse-pass driver. "
        "Before the loop can start, you need a seed gradient for `end_node`. If the caller passed "
        "`end_grad=None`, the convention is `torch.ones_like(end_node.array)` — interpreted as "
        "`d(end_node.sum())/d(end_node)`. Once seeded, the loop pops each node's accumulated grad, "
        "dispatches the per-arg back_fn, and accumulates into each parent's slot.\n\n"
        "Composing them in one function exercises BOTH atoms: you must default the seed AND drive the "
        "reverse loop, with leaves landing in `.grad` and intermediates living in a scratch dict."
    ),
    "prompt_body": (
        "Implement `cx25_backprop(end_node, end_grad, sorted_graph, back_funcs)` that:\n\n"
        "1. **Seeds** the reverse pass — if `end_grad is None`, default to `t.ones_like(end_node.array)`. "
        "Otherwise unbox `end_grad.array` (asserting shape matches `end_node.array`).\n"
        "2. **Drives** the reverse loop — for each node in `sorted_graph`, pop the accumulated grad, "
        "dispatch `back_funcs[(recipe.func, argnum)]` for each parent in `recipe.parents`, and accumulate "
        "into the parent's slot. Leaves (no recipe) get `.grad` populated/accumulated."
    ),
    "stub_body": (
        "def cx25_backprop(end_node, end_grad, sorted_graph, back_funcs):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from dataclasses import dataclass, field\n"
        "from typing import Any, Callable, Optional\n"
        "\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Optional[Callable] = None\n"
        "    args: tuple = ()\n"
        "    kwargs: dict = field(default_factory=dict)\n"
        "    parents: dict = field(default_factory=dict)\n"
        "\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad=False, recipe=None):\n"
        "        self.array = array\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = recipe\n"
        "        self.grad = None\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y): return grad_out * y\n"
        "def multiply_back1(grad_out, out, x, y): return grad_out * x\n"
        "BF = {(t.multiply, 0): multiply_back0, (t.multiply, 1): multiply_back1}\n"
        "\n"
        "# Case A: end_grad=None on a non-scalar end node → ones_like default.\n"
        "x = MiniTensor(t.tensor([2.0, 3.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([5.0, 7.0]), requires_grad=True)\n"
        "out_arr = x.array * y.array\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y})\n"
        "cx25_backprop(out, None, [out, x, y], BF)\n"
        "assert t.allclose(x.grad, y.array), f'expected y, got {x.grad}'\n"
        "assert t.allclose(y.grad, x.array), f'expected x, got {y.grad}'\n"
        "\n"
        "# Case B: explicit end_grad — used directly as seed.\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 4.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([1.0, 1.0, 1.0]), requires_grad=True)\n"
        "out_arr = a.array * b.array\n"
        "out = MiniTensor(out_arr, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.multiply, args=(a.array, b.array), kwargs={}, parents={0: a, 1: b})\n"
        "seed = MiniTensor(t.tensor([0.5, 0.5, 0.5]))\n"
        "cx25_backprop(out, seed, [out, a, b], BF)\n"
        "assert t.allclose(a.grad, t.tensor([0.5, 0.5, 0.5])), f'explicit-seed wrong: {a.grad}'\n"
        "\n"
        "# Case C: shape-mismatched seed must raise AssertionError.\n"
        "wrong = MiniTensor(t.zeros(2, 5))\n"
        "fresh = MiniTensor(t.tensor([1.0, 2.0, 4.0]), requires_grad=True)\n"
        "fresh_out = MiniTensor(fresh.array * 1.0, requires_grad=True)\n"
        "fresh_out.recipe = Recipe(func=t.multiply, args=(fresh.array, t.tensor([1.0,1.0,1.0])), kwargs={},\n"
        "                          parents={0: fresh})\n"
        "raised = False\n"
        "try:\n"
        "    cx25_backprop(fresh_out, wrong, [fresh_out, fresh], BF)\n"
        "except AssertionError:\n"
        "    raised = True\n"
        "assert raised, 'shape-mismatched end_grad should raise AssertionError'"
    ),
    "solution_body": (
        "def cx25_backprop(end_node, end_grad, sorted_graph, back_funcs):\n"
        "    # Atom A: seed default ← ones_like(end_node.array) when end_grad is None.\n"
        "    if end_grad is None:\n"
        "        seed = t.ones_like(end_node.array)\n"
        "    else:\n"
        "        assert end_grad.array.shape == end_node.array.shape, (\n"
        "            f'end_grad shape {tuple(end_grad.array.shape)} mismatches '\n"
        "            f'end_node shape {tuple(end_node.array.shape)}'\n"
        "        )\n"
        "        seed = end_grad.array\n"
        "    # Atom B: reverse-pass loop — pop accumulated grad, dispatch per parent.\n"
        "    grads = {id(end_node): seed}\n"
        "    for node in sorted_graph:\n"
        "        nid = id(node)\n"
        "        if nid not in grads:\n"
        "            continue\n"
        "        grad_out = grads.pop(nid)\n"
        "        if node.recipe is None:\n"
        "            if node.grad is None:\n"
        "                node.grad = grad_out\n"
        "            else:\n"
        "                node.grad = node.grad + grad_out\n"
        "            continue\n"
        "        for argnum, parent in node.recipe.parents.items():\n"
        "            back_fn = back_funcs[(node.recipe.func, argnum)]\n"
        "            gp = back_fn(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)\n"
        "            pid = id(parent)\n"
        "            grads[pid] = grads.get(pid, 0) + gp"
    ),
    "solution_notes": (
        "The seed step and the loop step are inseparable in real ARENA code — the loop can't start "
        "without a seed in `grads`, and the seed has no purpose without the loop. `ones_like` is the "
        "correct default because the implicit reduction for a non-scalar end_node is `.sum()`, whose "
        "Jacobian is exactly ones."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["end-grad-default-ones-like", "backprop-pop-outgrad-loop"],
    "lo": (
        "Compose the seed-default rule with the reverse-pass driver loop to run a complete "
        "backprop from a non-scalar end node when no explicit end_grad is supplied."
    ),
}


# ===========================================================================
# cx26 — matmul_back0/1 with the arg-position split
# ===========================================================================
spec_26 = {
    "atom_ids": ["matmul-back-transpose-pair", "arg-position-back-functions"],
    "subtopics": _subs(["matmul-back-transpose-pair", "arg-position-back-functions"]),
    "primary_atom": "matmul-back-transpose-pair",
    "part": "part4",
    "exercise_index": 26,
    "exercise_title": "matmul_back0 vs matmul_back1 — transpose pair, mirror argnums",
    "slug": "matmul-back-argnum-split",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA registers back-fns by `(func, argnum)`. For asymmetric ops like `@` (matmul), the back-fn "
        "is genuinely different per arg position. `matmul_back0` returns `dL/dx` for `x @ y`; "
        "`matmul_back1` returns `dL/dy`. They share grad_out but transpose the OTHER operand: "
        "`grad_out @ y.T` vs `x.T @ grad_out`. The arg-position-back-functions atom is what registers "
        "these distinct entries; the matmul-back-transpose-pair atom is the specific (x, y) ↔ (y.T, x.T) "
        "mirror that makes both shapes work out."
    ),
    "prompt_body": (
        "Implement `cx26_matmul_back(grad_out, out, x, y, argnum)` that returns:\n\n"
        "- if `argnum == 0`: `dL/dx` = `grad_out @ y.T`\n"
        "- if `argnum == 1`: `dL/dy` = `x.T @ grad_out`\n"
        "- otherwise: raise `ValueError`\n\n"
        "One function, two formulas, dispatched by argnum — the same pattern ARENA uses inside its "
        "`back_funcs` registry."
    ),
    "stub_body": (
        "def cx26_matmul_back(grad_out, out, x, y, argnum):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "x = t.randn(4, 3)\n"
        "y = t.randn(3, 5)\n"
        "out = x @ y\n"
        "grad_out = t.randn(4, 5)\n"
        "\n"
        "gx = cx26_matmul_back(grad_out, out, x, y, argnum=0)\n"
        "gy = cx26_matmul_back(grad_out, out, x, y, argnum=1)\n"
        "assert gx.shape == x.shape, f'gx shape: {gx.shape} vs {x.shape}'\n"
        "assert gy.shape == y.shape, f'gy shape: {gy.shape} vs {y.shape}'\n"
        "assert t.allclose(gx, grad_out @ y.T), 'gx formula wrong'\n"
        "assert t.allclose(gy, x.T @ grad_out), 'gy formula wrong'\n"
        "\n"
        "# Cross-check against torch autograd.\n"
        "xa = x.clone().requires_grad_(True)\n"
        "ya = y.clone().requires_grad_(True)\n"
        "(xa @ ya).backward(grad_out)\n"
        "assert t.allclose(gx, xa.grad), f'gx vs autograd mismatch'\n"
        "assert t.allclose(gy, ya.grad), f'gy vs autograd mismatch'\n"
        "\n"
        "# Bad argnum must raise.\n"
        "raised = False\n"
        "try: cx26_matmul_back(grad_out, out, x, y, argnum=2)\n"
        "except ValueError: raised = True\n"
        "assert raised, 'argnum=2 should raise ValueError'"
    ),
    "solution_body": (
        "def cx26_matmul_back(grad_out, out, x, y, argnum):\n"
        "    # Arg-position dispatch: same op, different formula per argnum.\n"
        "    if argnum == 0:\n"
        "        # dL/dx: (m,k) @ (k,n) shape works as grad_out @ y.T → (m,n).\n"
        "        return grad_out @ y.T\n"
        "    if argnum == 1:\n"
        "        # dL/dy: x.T @ grad_out → (k,m) @ (m,n) = (k,n) = y.shape.\n"
        "        return x.T @ grad_out\n"
        "    raise ValueError(f'matmul has args (x, y); argnum must be 0 or 1, got {argnum}')"
    ),
    "solution_notes": (
        "The transpose pair `(y.T, x.T)` is what makes both shapes line up. If you forgot the transpose, "
        "`grad_out @ y` would be a shape error, not a numerical one — the type system catches it. The "
        "argnum split exists because matmul is not commutative, so the back-fn genuinely needs two "
        "different code paths."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["matmul-back-transpose-pair", "arg-position-back-functions"],
    "lo": (
        "Dispatch matmul back-fns by arg position, returning the correct transpose-pair "
        "formula for each operand."
    ),
}


# ===========================================================================
# cx27 — sum_back and broadcast_back are duals
# ===========================================================================
spec_27 = {
    "atom_ids": ["sum-and-broadcast-duality", "unbroadcast-pattern"],
    "subtopics": _subs(["sum-and-broadcast-duality", "unbroadcast-pattern"]),
    "primary_atom": "sum-and-broadcast-duality",
    "part": "part4",
    "exercise_index": 27,
    "exercise_title": "sum_back broadcasts; unbroadcast_back collapses — dual ops",
    "slug": "sum-back-broadcast-duality",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Forward `sum(x, dim)` reduces — backward must broadcast the upstream grad back to `x.shape`. "
        "Forward `broadcast_to(x, shape)` expands — backward must SUM the upstream grad back down to "
        "`x.shape`. They are exact duals: one inserts a size-1 axis and expands, the other peels leading "
        "axes and sums out size-1 axes (the unbroadcast pattern).\n\n"
        "Composing them means: take a tensor `x`, run forward `sum(dim, keepdim=False)`, then verify "
        "that the gradient round-trip — sum_back to expand the upstream grad back to `x.shape`, AND "
        "unbroadcast to collapse a broadcast back to `x.shape` — agree on the same shape and act as "
        "exact inverses on size-1 axes."
    ),
    "prompt_body": (
        "Implement `cx27_round_trip(x, dim)` that returns a dict with three tensors:\n\n"
        "- `'forward_sum'` — `x.sum(dim=dim, keepdim=False)`\n"
        "- `'sum_back_grad'` — the gradient w.r.t. `x` of `forward_sum.sum()` (i.e. seed `grad_out = "
        "ones_like(forward_sum)`, then sum_back to `x.shape`). Use unsqueeze+expand+clone.\n"
        "- `'unbroadcast_grad'` — call `unbroadcast(t.ones_like(x), x)` (no-op since shapes match) AND "
        "call `unbroadcast(t.ones(x.shape[0], *([1] * (x.ndim - 1))).expand_as(x).clone(), x.sum(dim=dim, "
        "keepdim=True).expand_as(x).clone() * 0 + x.sum(dim=dim, keepdim=True))` — the contract is that "
        "after the round trip, `sum_back_grad` and an `unbroadcast` of `forward_sum.unsqueeze(dim)."
        "expand_as(x).clone()` (i.e. broadcasting back) match. Concretely: define `unbroadcast` inline, "
        "then assert that `sum_back(ones_like(forward_sum), dim, x.shape)` equals "
        "`unbroadcast(ones_like(x), x)` — both should be `ones_like(x)`.\n\n"
        "Simplification: the test just asks for `sum_back_grad` and `unbroadcast_grad` to both equal "
        "`t.ones_like(x)` when the seed is `ones_like(forward_sum)`. You must define `unbroadcast` "
        "AND `sum_back` inline — that's the composition."
    ),
    "stub_body": (
        "def cx27_round_trip(x, dim):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: 2-D sum over dim=1, keepdim=False.\n"
        "x = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n"
        "out = cx27_round_trip(x, dim=1)\n"
        "assert set(out.keys()) >= {'forward_sum', 'sum_back_grad', 'unbroadcast_grad'}, out.keys()\n"
        "assert out['forward_sum'].shape == (2,), out['forward_sum'].shape\n"
        "assert t.allclose(out['forward_sum'], t.tensor([6.0, 15.0]))\n"
        "assert out['sum_back_grad'].shape == x.shape, out['sum_back_grad'].shape\n"
        "assert t.allclose(out['sum_back_grad'], t.ones_like(x)), out['sum_back_grad']\n"
        "assert out['unbroadcast_grad'].shape == x.shape, out['unbroadcast_grad'].shape\n"
        "assert t.allclose(out['unbroadcast_grad'], t.ones_like(x)), out['unbroadcast_grad']\n"
        "# Duality: sum_back of ones-seed equals unbroadcast of broadcast-back of ones-seed.\n"
        "assert t.allclose(out['sum_back_grad'], out['unbroadcast_grad'])\n"
        "\n"
        "# Case B: 3-D sum over dim=0 — leading-axis case.\n"
        "x2 = t.randn(4, 3, 5)\n"
        "out2 = cx27_round_trip(x2, dim=0)\n"
        "assert out2['forward_sum'].shape == (3, 5)\n"
        "assert out2['sum_back_grad'].shape == x2.shape\n"
        "assert t.allclose(out2['sum_back_grad'], t.ones_like(x2))\n"
        "assert t.allclose(out2['sum_back_grad'], out2['unbroadcast_grad'])"
    ),
    "solution_body": (
        "def cx27_round_trip(x, dim):\n"
        "    # Inline unbroadcast: peel leading + collapse size-1 axes.\n"
        "    def unbroadcast(grad, original):\n"
        "        while grad.ndim > original.ndim:\n"
        "            grad = grad.sum(dim=0)\n"
        "        for i, size in enumerate(original.shape):\n"
        "            if size == 1 and grad.shape[i] != 1:\n"
        "                grad = grad.sum(dim=i, keepdim=True)\n"
        "        return grad\n"
        "\n"
        "    # Inline sum_back: re-insert the dropped axis, then expand back to x.shape.\n"
        "    def sum_back(grad_out, out, x_ref, dim, keepdim=False):\n"
        "        if not keepdim:\n"
        "            grad_out = grad_out.unsqueeze(dim)\n"
        "        return grad_out.expand_as(x_ref).clone()\n"
        "\n"
        "    forward_sum = x.sum(dim=dim, keepdim=False)\n"
        "    seed = t.ones_like(forward_sum)\n"
        "    sum_back_grad = sum_back(seed, forward_sum, x, dim, keepdim=False)\n"
        "    # Broadcast-back path: pretend forward was broadcast_to(x.shape) — unbroadcast collapses it.\n"
        "    broadcast_back = sum_back_grad.clone()  # already at x.shape via sum_back\n"
        "    unbroadcast_grad = unbroadcast(broadcast_back, x)\n"
        "    return {\n"
        "        'forward_sum': forward_sum,\n"
        "        'sum_back_grad': sum_back_grad,\n"
        "        'unbroadcast_grad': unbroadcast_grad,\n"
        "    }"
    ),
    "solution_notes": (
        "The two atoms are duals: `sum_back` inserts a size-1 axis and expands; `unbroadcast` collapses "
        "leading + size-1 axes by summing. On a uniform-ones seed they agree exactly. The deeper "
        "invariant: `sum_back ∘ broadcast == id` on the size-1-axis Jacobian — they are inverse adjoints."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["sum-and-broadcast-duality", "unbroadcast-pattern"],
    "lo": (
        "Show that sum_back (expand) and unbroadcast_back (collapse) are dual ops by round-tripping "
        "a uniform-ones gradient through both and verifying they land on the same x.shape."
    ),
}


# ===========================================================================
# cx28 — maximum_back with tied half-mass, dispatched by argnum
# ===========================================================================
spec_28 = {
    "atom_ids": ["max-back-tied-half", "arg-position-back-functions"],
    "subtopics": _subs(["max-back-tied-half", "arg-position-back-functions"]),
    "primary_atom": "max-back-tied-half",
    "part": "part4",
    "exercise_index": 28,
    "exercise_title": "maximum_back with 50/50 tie-splitting, argnum 0 vs 1 mirror",
    "slug": "maximum-back-tied-argnum-split",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`maximum(x, y)` is elementwise — the gradient lands on whichever input was larger at each "
        "position. At ties (x == y) the convention is to split the gradient mass 50/50 so total mass is "
        "conserved. Like matmul, maximum needs TWO back-fns: argnum 0 gives `dL/dx`, argnum 1 gives "
        "`dL/dy`. The masks mirror: `(x > y) + 0.5 * (x == y)` for arg-0; `(x < y) + 0.5 * (x == y)` for "
        "arg-1. They must sum to 1 everywhere — that's the mass-conservation invariant."
    ),
    "prompt_body": (
        "Implement `cx28_maximum_back(grad_out, out, x, y, argnum)` that returns:\n\n"
        "- if `argnum == 0`: `grad_out * ((x > y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype))`\n"
        "- if `argnum == 1`: `grad_out * ((x < y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype))`\n"
        "- otherwise: raise `ValueError`\n\n"
        "Verify that the masks for argnum 0 and 1 sum to `ones_like(grad_out)` at every position — "
        "that's the mass-conservation check that catches a missing 0.5 factor."
    ),
    "stub_body": (
        "def cx28_maximum_back(grad_out, out, x, y, argnum):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Construct overlap + ties.\n"
        "x = t.tensor([1.0, 5.0, 3.0, 7.0, 7.0])\n"
        "y = t.tensor([4.0, 2.0, 3.0, 7.0, 6.0])\n"
        "out = t.maximum(x, y)\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0, 1.0, 1.0])\n"
        "\n"
        "gx = cx28_maximum_back(grad_out, out, x, y, argnum=0)\n"
        "gy = cx28_maximum_back(grad_out, out, x, y, argnum=1)\n"
        "\n"
        "# Position-by-position: x<y, x>y, tie, tie, x>y.\n"
        "expected_gx = t.tensor([0.0, 1.0, 0.5, 0.5, 1.0])\n"
        "expected_gy = t.tensor([1.0, 0.0, 0.5, 0.5, 0.0])\n"
        "assert t.allclose(gx, expected_gx), f'gx wrong: got {gx}, expected {expected_gx}'\n"
        "assert t.allclose(gy, expected_gy), f'gy wrong: got {gy}, expected {expected_gy}'\n"
        "\n"
        "# Mass conservation: gx + gy == grad_out everywhere.\n"
        "assert t.allclose(gx + gy, grad_out), f'masses do not sum to grad_out: {gx + gy}'\n"
        "\n"
        "# argnum dispatch — bad argnum must raise ValueError.\n"
        "raised = False\n"
        "try: cx28_maximum_back(grad_out, out, x, y, argnum=2)\n"
        "except ValueError: raised = True\n"
        "assert raised, 'bad argnum should raise ValueError'\n"
        "\n"
        "# Cross-check with autograd at a non-tied case.\n"
        "xa = t.tensor([1.0, 5.0, 8.0]).requires_grad_(True)\n"
        "ya = t.tensor([4.0, 2.0, 3.0]).requires_grad_(True)\n"
        "t.maximum(xa, ya).sum().backward()\n"
        "g0 = cx28_maximum_back(t.ones(3), None, xa.detach(), ya.detach(), argnum=0)\n"
        "g1 = cx28_maximum_back(t.ones(3), None, xa.detach(), ya.detach(), argnum=1)\n"
        "assert t.allclose(g0, xa.grad), f'argnum=0 vs autograd: {g0} vs {xa.grad}'\n"
        "assert t.allclose(g1, ya.grad), f'argnum=1 vs autograd: {g1} vs {ya.grad}'"
    ),
    "solution_body": (
        "def cx28_maximum_back(grad_out, out, x, y, argnum):\n"
        "    if argnum == 0:\n"
        "        # x wins where x > y; half-share at ties so mass is conserved.\n"
        "        mask = (x > y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype)\n"
        "        return grad_out * mask\n"
        "    if argnum == 1:\n"
        "        # mirror split: y wins where x < y; same half-share at ties.\n"
        "        mask = (x < y).to(grad_out.dtype) + 0.5 * (x == y).to(grad_out.dtype)\n"
        "        return grad_out * mask\n"
        "    raise ValueError(f'maximum has args (x, y); argnum must be 0 or 1, got {argnum}')"
    ),
    "solution_notes": (
        "Mass conservation is the smoking-gun test: `gx + gy == grad_out`. If you skipped the 0.5 at "
        "ties (using strict `<` and `>` for both), tied positions would receive zero gradient and you'd "
        "lose mass. If you used non-strict `<=` and `>=`, tied positions would receive double mass. "
        "The 50/50 split is the unique convention that preserves the total."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["max-back-tied-half", "arg-position-back-functions"],
    "lo": (
        "Implement maximum_back with the tied half-mass convention and dispatch correctly by "
        "argnum so total gradient mass is conserved across both back-fns."
    ),
}


# ===========================================================================
# cx29 — multiply_back applied as the elementwise chain rule
# ===========================================================================
spec_29 = {
    "atom_ids": ["multiply-back", "chain-rule-elementwise"],
    "subtopics": _subs(["multiply-back", "chain-rule-elementwise"]),
    "primary_atom": "multiply-back",
    "part": "part4",
    "exercise_index": 29,
    "exercise_title": "multiply_back as elementwise chain rule — dL/dx = grad_out * y",
    "slug": "multiply-back-elementwise-chain",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The elementwise chain rule says: for `out = f(x, y)`, `dL/dx = grad_out * (∂out/∂x)`, all "
        "elementwise. For `f = multiply`, `∂out/∂x = y` and `∂out/∂y = x`, giving the multiply-back "
        "formulas `dL/dx = grad_out * y` and `dL/dy = grad_out * x`. So `multiply_back` is the simplest "
        "concrete instance of the elementwise chain rule applied to a binary op — no special derivative, "
        "the OTHER operand IS the derivative.\n\n"
        "Composing them: implement both `multiply_back0` and `multiply_back1` and verify against the "
        "elementwise chain-rule pattern (grad_out * local-derivative) and against torch's autograd."
    ),
    "prompt_body": (
        "Implement `cx29_multiply_back(grad_out, out, x, y, argnum)` that:\n\n"
        "- if `argnum == 0`: returns `grad_out * y` (chain rule with local derivative `y`)\n"
        "- if `argnum == 1`: returns `grad_out * x` (chain rule with local derivative `x`)\n\n"
        "Same shapes as `x` / `y` respectively. No broadcasting in this drill — that's covered by cx30. "
        "Cross-check against `torch.autograd` so the elementwise chain-rule formula is verified, not "
        "just typed."
    ),
    "stub_body": (
        "def cx29_multiply_back(grad_out, out, x, y, argnum):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y = t.tensor([5.0, 7.0, 11.0])\n"
        "out = x * y\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "\n"
        "gx = cx29_multiply_back(grad_out, out, x, y, argnum=0)\n"
        "gy = cx29_multiply_back(grad_out, out, x, y, argnum=1)\n"
        "assert t.allclose(gx, y), f'multiply_back0 with ones-grad_out should equal y; got {gx}'\n"
        "assert t.allclose(gy, x), f'multiply_back1 with ones-grad_out should equal x; got {gy}'\n"
        "\n"
        "# Non-unit grad_out — true chain rule.\n"
        "grad_out2 = t.tensor([0.5, 2.0, 3.0])\n"
        "gx2 = cx29_multiply_back(grad_out2, out, x, y, argnum=0)\n"
        "gy2 = cx29_multiply_back(grad_out2, out, x, y, argnum=1)\n"
        "assert t.allclose(gx2, grad_out2 * y), f'chain rule arg0 wrong'\n"
        "assert t.allclose(gy2, grad_out2 * x), f'chain rule arg1 wrong'\n"
        "\n"
        "# Cross-check against autograd.\n"
        "xa = x.clone().requires_grad_(True)\n"
        "ya = y.clone().requires_grad_(True)\n"
        "(xa * ya).backward(grad_out2)\n"
        "assert t.allclose(gx2, xa.grad)\n"
        "assert t.allclose(gy2, ya.grad)"
    ),
    "solution_body": (
        "def cx29_multiply_back(grad_out, out, x, y, argnum):\n"
        "    # Elementwise chain rule: dL/dz = grad_out * (∂out/∂z).\n"
        "    # For multiply: ∂(x*y)/∂x = y, and ∂(x*y)/∂y = x.\n"
        "    # The 'OTHER operand IS the derivative' — that's why multiply_back is so clean.\n"
        "    if argnum == 0:\n"
        "        return grad_out * y\n"
        "    if argnum == 1:\n"
        "        return grad_out * x\n"
        "    raise ValueError(f'multiply has args (x, y); argnum must be 0 or 1, got {argnum}')"
    ),
    "solution_notes": (
        "The elementwise chain rule is the simplest case of backward: no transpose, no broadcasting "
        "(this drill skips broadcasting deliberately), just `grad_out * local-derivative`. For multiply "
        "the local derivative is the OTHER operand — that's why these two atoms collapse into a "
        "one-line formula per argnum. The same pattern generalizes to `div_back0 = grad_out / y`, "
        "`pow_back0 = grad_out * y * x**(y-1)`, etc."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["multiply-back", "chain-rule-elementwise"],
    "lo": (
        "Apply the elementwise chain rule to derive multiply_back0 and multiply_back1 in a single "
        "argnum-dispatched function, verified against torch autograd."
    ),
}


# ===========================================================================
# cx30 — unbroadcast collapses what broadcast_to expanded
# ===========================================================================
spec_30 = {
    "atom_ids": ["unbroadcast-pattern", "sum-and-broadcast-duality"],
    "subtopics": _subs(["unbroadcast-pattern", "sum-and-broadcast-duality"]),
    "primary_atom": "unbroadcast-pattern",
    "part": "part4",
    "exercise_index": 30,
    "exercise_title": "unbroadcast inverts broadcast_to — sum out the axes that were expanded",
    "slug": "unbroadcast-inverts-broadcast",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`np.broadcast_to(x, shape)` (forward) expands `x` into a larger shape by prepending 1-axes "
        "and/or stretching size-1 axes. `unbroadcast(grad, original)` (backward) does the exact inverse: "
        "sum over the prepended leading axes (they didn't exist in `original`), then sum out the size-1 "
        "axes (with `keepdim=True` so the shape matches). The two are exact duals — broadcasting in the "
        "forward pass is exactly summing in the backward pass.\n\n"
        "This is the sum-and-broadcast-duality atom at its purest: same axes, opposite operations. The "
        "composite drill verifies that `unbroadcast(broadcast_to(x, big_shape), x).shape == x.shape` and "
        "that the gradient values are the count-correct sums."
    ),
    "prompt_body": (
        "Implement `cx30_unbroadcast_after_broadcast(x, big_shape)` that:\n\n"
        "1. Computes `expanded = t.broadcast_to(x, big_shape).clone()` — the forward broadcast.\n"
        "2. Calls an inline `unbroadcast(grad, original)` to collapse `t.ones(big_shape)` back to "
        "`x.shape`. Use the unbroadcast pattern: peel leading axes with `sum(dim=0)`; then collapse "
        "size-1 axes with `sum(dim=i, keepdim=True)`.\n"
        "3. Returns `{'expanded_shape': expanded.shape, 'collapsed_grad': <unbroadcast result>}`.\n\n"
        "The collapsed grad must have shape `x.shape` and values equal to the number of broadcast "
        "copies at each position — that's the count-of-summed-positions, which is the sum-broadcast "
        "duality at work."
    ),
    "stub_body": (
        "def cx30_unbroadcast_after_broadcast(x, big_shape):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: 1-D (3,) broadcast to (4, 3) — leading axis added.\n"
        "x = t.tensor([1.0, 2.0, 3.0])\n"
        "out = cx30_unbroadcast_after_broadcast(x, big_shape=(4, 3))\n"
        "assert tuple(out['expanded_shape']) == (4, 3), out['expanded_shape']\n"
        "assert out['collapsed_grad'].shape == x.shape, out['collapsed_grad'].shape\n"
        "# Each x position got summed over the leading 4 axis: ones(4,3).sum(dim=0) → [4,4,4].\n"
        "assert t.allclose(out['collapsed_grad'], t.tensor([4.0, 4.0, 4.0]))\n"
        "\n"
        "# Case B: (1, 3) broadcast to (5, 3) — size-1 axis expanded.\n"
        "x2 = t.tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)\n"
        "out2 = cx30_unbroadcast_after_broadcast(x2, big_shape=(5, 3))\n"
        "assert out2['collapsed_grad'].shape == x2.shape  # must KEEP the size-1 dim.\n"
        "assert t.allclose(out2['collapsed_grad'], t.tensor([[5.0, 5.0, 5.0]]))\n"
        "\n"
        "# Case C: scalar broadcast to (2, 3) — both peel-leading-axes paths fire.\n"
        "x3 = t.tensor(7.0)  # 0-D\n"
        "out3 = cx30_unbroadcast_after_broadcast(x3, big_shape=(2, 3))\n"
        "assert out3['collapsed_grad'].shape == x3.shape\n"
        "assert t.allclose(out3['collapsed_grad'], t.tensor(6.0))  # 2*3=6 positions summed.\n"
        "\n"
        "# Case D: (3, 1, 5) broadcast to (3, 4, 5) — middle size-1 axis expanded.\n"
        "x4 = t.randn(3, 1, 5)\n"
        "out4 = cx30_unbroadcast_after_broadcast(x4, big_shape=(3, 4, 5))\n"
        "assert out4['collapsed_grad'].shape == x4.shape\n"
        "# Sum-of-ones across the expanded dim-1 (size 4) → each entry is 4.\n"
        "assert t.allclose(out4['collapsed_grad'], t.full_like(x4, 4.0))"
    ),
    "solution_body": (
        "def cx30_unbroadcast_after_broadcast(x, big_shape):\n"
        "    # Forward: expand x into big_shape (zero-copy view; clone for cleanliness).\n"
        "    expanded = t.broadcast_to(x, big_shape).clone()\n"
        "\n"
        "    # Backward dual: unbroadcast collapses the axes broadcast_to added/expanded.\n"
        "    def unbroadcast(grad, original):\n"
        "        # Step 1: peel leading axes that broadcasting prepended.\n"
        "        while grad.ndim > original.ndim:\n"
        "            grad = grad.sum(dim=0)\n"
        "        # Step 2: collapse size-1 axes that were stretched, KEEPING shape.\n"
        "        for i, size in enumerate(original.shape):\n"
        "            if size == 1 and grad.shape[i] != 1:\n"
        "                grad = grad.sum(dim=i, keepdim=True)\n"
        "        return grad\n"
        "\n"
        "    grad_in = t.ones(big_shape, dtype=x.dtype if x.is_floating_point() else t.float32)\n"
        "    collapsed_grad = unbroadcast(grad_in, x)\n"
        "    return {'expanded_shape': tuple(expanded.shape), 'collapsed_grad': collapsed_grad}"
    ),
    "solution_notes": (
        "The sum-broadcast duality is the deepest invariant of this drill: forward `broadcast_to` "
        "EXPANDS, backward `unbroadcast` SUMS, and they cancel on size-1 axes. The values in "
        "`collapsed_grad` count how many broadcast copies summed into each original position — that's "
        "literally the size of the expanded axis. If you used `keepdim=False` in step 2 you'd drop the "
        "size-1 axes and the shape assert would fail."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["unbroadcast-pattern", "sum-and-broadcast-duality"],
    "lo": (
        "Show that np.broadcast_to (forward) and unbroadcast (backward) are dual ops by collapsing "
        "a uniform-ones gradient back to the original tensor's shape with count-correct values."
    ),
}


SPECS = [spec_25, spec_26, spec_27, spec_28, spec_29, spec_30]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
