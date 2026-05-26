#!/usr/bin/env python3
"""Author 6 COMPOSITE procedural drills for batch-15 (group O — backprop driver cluster).

Composites (cx19..cx24) — exercise 2-3 ARENA-style autograd atoms together:

  cx19  dfs-three-set-toposort + cycle-detection-temp-set
        — DFS toposort over a DAG with temp/perm/visiting trio; raises on cycle.

  cx20  backprop-pop-outgrad-loop + dispatch-back-fn-from-recipe
        — main reverse-pass loop: pop next out-grad, dispatch back_fn from recipe.

  cx21  dispatch-back-fn-from-recipe + parents-dict-by-argidx
        — dispatch back_fn over each (argnum, parent) pair in recipe.parents dict.

  cx22  dfs-three-set-toposort + sorted-computational-graph
        — build sorted graph from a MiniTensor by DFS over recipe.parents, reversed.

  cx23  grad-accumulate-on-leaf + backprop-pop-outgrad-loop
        — accumulate grads on leaves at end of reverse pass (rebind, not in-place).

  cx24  back-fn-call-with-recipe-args + kwargs-pass-through-recipe
        — call back_fn with positional from parents + **recipe.kwargs (dim, keepdim).

Single composite exercise per drill; the test exercises EACH constituent atom.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite

# ---------------------------------------------------------------------------
# Inventory — atom_id -> subtopic
# ---------------------------------------------------------------------------

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# ---------------------------------------------------------------------------
# cx19 — dfs-three-set-toposort + cycle-detection-temp-set
# ---------------------------------------------------------------------------

RECAP_19 = (
    "## Composing DFS toposort with temp-set cycle detection\n"
    "\n"
    "The two atoms are TWO RESPONSIBILITIES of one three-colour DFS walk:\n"
    "\n"
    "- **`dfs-three-set-toposort`** — produce a deps-first ordering of every\n"
    "  reachable node (root LAST). The output is the `result` list.\n"
    "- **`cycle-detection-temp-set`** — the `temp` (gray) set on the recursion\n"
    "  stack: re-entering it means we walked a back-edge → cycle.\n"
    "\n"
    "```python\n"
    "def topo_sort_with_cycle_check(root, get_children):\n"
    "    perm = set()     # finished subtrees (cycle-detection-temp-set: NOT a cycle)\n"
    "    temp = set()     # currently-on-stack (cycle-detection-temp-set: IS a cycle)\n"
    "    result = []\n"
    "    def visit(node):\n"
    "        nid = id(node)\n"
    "        if nid in perm: return                # legal shared descendant\n"
    "        if nid in temp:                        # back-edge → CYCLE\n"
    "            raise ValueError(f'cycle at {node!r}')\n"
    "        temp.add(nid)\n"
    "        for child in get_children(node):\n"
    "            visit(child)\n"
    "        temp.remove(nid)                       # pop off recursion stack\n"
    "        perm.add(nid)\n"
    "        result.append(node)                    # deps-first append\n"
    "    visit(root)\n"
    "    return result\n"
    "```\n"
    "\n"
    "**Why both atoms compose into one function.** A DAG-only toposort needs\n"
    "BOTH outputs: a valid ordering AND a guarantee it ran on a DAG. Drop\n"
    "the temp-set check and you'd silently produce a partial list on a\n"
    "cyclic input. Drop the perm-set and you'd flag every diamond DAG as a\n"
    "false cycle. Two color sets, two purposes — both load-bearing.\n"
    "\n"
    "**Result order: deps-first, root LAST.** This is the order a FORWARD\n"
    "pass would use (leaves first, then their consumers). The reverse pass\n"
    "wants the OPPOSITE — `[::-1]` on the output flips it. That's the next\n"
    "composite (cx22)."
)

SPEC_19 = {
    "atom_ids": ["dfs-three-set-toposort", "cycle-detection-temp-set"],
    "subtopics": _subs(["dfs-three-set-toposort", "cycle-detection-temp-set"]),
    "primary_atom": "dfs-three-set-toposort",
    "part": "part4",
    "exercise_index": 19,
    "exercise_title": "DFS toposort with cycle detection via temp/perm/visiting trio",
    "slug": "dfs-toposort-with-temp-set-cycle-detection",
    "atom_recap_md": RECAP_19,
    "prompt_body": (
        "Implement `cx19_topo_sort_with_cycle_check(root, get_children)` — a "
        "three-colour DFS that returns descendants of `root` in deps-first "
        "order (root LAST) AND raises `ValueError` on any cycle.\n\n"
        "**Both atoms in one function.** This is the canonical ARENA "
        "`topological_sort` — every later composite (cx20 backprop loop, cx22 "
        "sorted-graph builder) uses it.\n\n"
        "**Contract.**\n"
        "- Returns `list` of nodes reachable from `root` via `get_children`.\n"
        "- Every node appears AFTER all of its transitive children — deps-first.\n"
        "- `root` is the LAST element.\n"
        "- Each reachable node appears EXACTLY once (diamond DAGs collapse).\n"
        "- Raises `ValueError` on a cycle (self-loop, two-node, deep-graph).\n\n"
        "**Algorithm** — three colours, both atoms together:\n\n"
        "```python\n"
        "def visit(node):\n"
        "    nid = id(node)\n"
        "    if nid in perm: return                  # already finished\n"
        "    if nid in temp: raise ValueError(...)   # cycle (temp-set atom)\n"
        "    temp.add(nid)\n"
        "    for child in get_children(node):\n"
        "        visit(child)\n"
        "    temp.remove(nid)                        # pop off stack\n"
        "    perm.add(nid)\n"
        "    result.append(node)                     # deps-first (toposort atom)\n"
        "```\n\n"
        "Use `id(node)` as the set key — safe for any object identity.\n\n"
        "**Failure modes the test catches.**\n"
        "1. Forgetting `temp.remove(nid)` → siblings sharing a leaf falsely flag as cycle.\n"
        "2. Forgetting `perm.add(nid)` → diamond DAGs falsely flag as cycle.\n"
        "3. Appending BEFORE recursing into children → wrong order (root would be first)."
    ),
    "stub_body": (
        "def cx19_topo_sort_with_cycle_check(root, get_children):\n"
        '    """Three-colour DFS: deps-first toposort, raise ValueError on cycle."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- helper graph node ---\n"
        "class N:\n"
        "    def __init__(self, name, *children):\n"
        "        self.name = name\n"
        "        self.children = list(children)\n"
        "    def __repr__(self): return f'N({self.name})'\n"
        "\n"
        "def get_children(n): return n.children\n"
        "\n"
        "# === linear chain a -> b -> c — deps-first, root last ===\n"
        "c = N('c'); b = N('b', c); a = N('a', b)\n"
        "order = cx19_topo_sort_with_cycle_check(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names == ['c', 'b', 'a'], f'linear: {names}'\n"
        "\n"
        "# === diamond DAG — d appears ONCE despite two paths ===\n"
        "d = N('d'); b = N('b', d); c = N('c', d); a = N('a', b, c)\n"
        "order = cx19_topo_sort_with_cycle_check(a, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names.count('d') == 1, f'd must appear once: {names}'\n"
        "assert names[-1] == 'a', f'root LAST: {names}'\n"
        "assert names.index('d') < names.index('b') < names.index('a')\n"
        "assert names.index('d') < names.index('c') < names.index('a')\n"
        "\n"
        "# === self-loop → ValueError (cycle-detection-temp-set in action) ===\n"
        "s = N('s'); s.children = [s]\n"
        "raised = False\n"
        "try: cx19_topo_sort_with_cycle_check(s, get_children)\n"
        "except ValueError: raised = True\n"
        "assert raised, 'self-loop must raise ValueError'\n"
        "\n"
        "# === two-node cycle → ValueError ===\n"
        "x = N('x'); y = N('y')\n"
        "x.children = [y]; y.children = [x]\n"
        "raised = False\n"
        "try: cx19_topo_sort_with_cycle_check(x, get_children)\n"
        "except ValueError: raised = True\n"
        "assert raised, 'two-node cycle must raise ValueError'\n"
        "\n"
        "# === mid-graph cycle (p -> q -> r -> q) → ValueError ===\n"
        "p = N('p'); q = N('q'); r = N('r')\n"
        "p.children = [q]; q.children = [r]; r.children = [q]\n"
        "raised = False\n"
        "try: cx19_topo_sort_with_cycle_check(p, get_children)\n"
        "except ValueError: raised = True\n"
        "assert raised, 'mid-graph cycle must raise'\n"
        "\n"
        "# === Two siblings share a leaf — NOT a cycle (regression: temp.remove) ===\n"
        "leaf = N('leaf'); xx = N('x', leaf); yy = N('y', leaf)\n"
        "root = N('root', xx, yy)\n"
        "order = cx19_topo_sort_with_cycle_check(root, get_children)\n"
        "names = [n.name for n in order]\n"
        "assert names.count('leaf') == 1, 'shared leaf must appear ONCE'\n"
        "assert names[-1] == 'root', 'root LAST despite shared descendant'\n"
        "\n"
        "# === singleton — single node, no children ===\n"
        "lonely = N('lonely')\n"
        "order = cx19_topo_sort_with_cycle_check(lonely, get_children)\n"
        "assert order == [lonely], f'singleton: {order}'"
    ),
    "solution_body": (
        "def cx19_topo_sort_with_cycle_check(root, get_children):\n"
        "    result = []\n"
        "    perm = set()   # fully processed (NOT a cycle if re-visited)\n"
        "    temp = set()   # currently on DFS stack (IS a cycle if re-visited)\n"
        "\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm:\n"
        "            return                          # legal shared descendant\n"
        "        if nid in temp:\n"
        "            raise ValueError(f'cycle at {node!r}')\n"
        "        temp.add(nid)\n"
        "        for child in get_children(node):\n"
        "            visit(child)\n"
        "        temp.remove(nid)                    # pop off recursion stack\n"
        "        perm.add(nid)\n"
        "        result.append(node)                 # deps-first append\n"
        "\n"
        "    visit(root)\n"
        "    return result"
    ),
    "solution_notes": (
        "**Two color sets, two atoms, two failure modes.** `perm` is the "
        "`cycle-detection-temp-set` insight that 'finished subtree' is not "
        "the same as 'currently in-flight'. `temp` is the back-edge detector. "
        "Drop either and the algorithm breaks on either diamond DAGs (false "
        "cycle) or shared-leaf siblings (false cycle from forgotten remove).\n\n"
        "**`result.append(node)` is the `dfs-three-set-toposort` insight.** "
        "The append happens AFTER all children have been visited — that's "
        "what makes it deps-first. Appending before recursing would give you "
        "root-first order (which is what the reverse pass eventually wants, "
        "but the standard idiom builds deps-first and reverses)."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["dfs-three-set-toposort", "cycle-detection-temp-set"],
    "lo": (
        "Apply a three-colour DFS that simultaneously produces a deps-first "
        "topological ordering AND raises ValueError on any back-edge, by "
        "maintaining `temp` (in-flight) and `perm` (finished) sets keyed by "
        "id(node) and appending each node to the result only after its "
        "subtree has been fully visited."
    ),
}


# ---------------------------------------------------------------------------
# cx20 — backprop-pop-outgrad-loop + dispatch-back-fn-from-recipe
# ---------------------------------------------------------------------------

RECAP_20 = (
    "## Composing the pop-outgrad loop with recipe-based dispatch\n"
    "\n"
    "The reverse-pass driver is two atoms wired into one tight loop:\n"
    "\n"
    "- **`backprop-pop-outgrad-loop`** — the OUTER walk: iterate the sorted\n"
    "  graph (end-node first), pop each node's accumulated grad out of the\n"
    "  `grads` dict, route to either leaf-write or parent-dispatch.\n"
    "- **`dispatch-back-fn-from-recipe`** — the INNER step: for each parent,\n"
    "  look up `back_funcs[(recipe.func, argnum)]` and call it.\n"
    "\n"
    "```python\n"
    "def backprop(end_node, end_grad, sorted_graph, back_funcs):\n"
    "    grads = {id(end_node): end_grad}\n"
    "    for node in sorted_graph:                       # pop-outgrad loop\n"
    "        if id(node) not in grads: continue\n"
    "        grad_out = grads.pop(id(node))              # POP, don't peek\n"
    "        if node.recipe is None:                     # leaf — write .grad\n"
    "            node.grad = grad_out if node.grad is None else node.grad + grad_out\n"
    "            continue\n"
    "        for argnum, parent in node.recipe.parents.items():    # dispatch loop\n"
    "            back_fn = back_funcs[(node.recipe.func, argnum)]  # (atom 2)\n"
    "            gp = back_fn(grad_out, node.array,\n"
    "                         *node.recipe.args, **node.recipe.kwargs)\n"
    "            grads[id(parent)] = grads.get(id(parent), 0) + gp\n"
    "```\n"
    "\n"
    "**Why these atoms must compose.** The pop-outgrad loop without dispatch\n"
    "has nothing to call (it knows WHEN to step but not WHAT to invoke).\n"
    "Dispatch without the loop has no driving traversal (it knows WHAT to\n"
    "call but not in what ORDER). Together they form the complete reverse-\n"
    "pass driver — every later autograd extension (gradient checkpointing,\n"
    "double-backward, mixed precision) is a small modification of THIS loop."
)

SPEC_20 = {
    "atom_ids": ["backprop-pop-outgrad-loop", "dispatch-back-fn-from-recipe"],
    "subtopics": _subs(["backprop-pop-outgrad-loop", "dispatch-back-fn-from-recipe"]),
    "primary_atom": "backprop-pop-outgrad-loop",
    "part": "part4",
    "exercise_index": 20,
    "exercise_title": "reverse-pass driver: pop next out-grad, dispatch back_fn from recipe",
    "slug": "backprop-pop-and-dispatch-loop",
    "atom_recap_md": RECAP_20,
    "prompt_body": (
        "Implement `cx20_backprop(end_node, end_grad, sorted_graph, back_funcs)` "
        "— the full reverse-pass driver. Two atoms compose:\n\n"
        "1. **OUTER walk** (`backprop-pop-outgrad-loop`) — iterate "
        "`sorted_graph` (end-node FIRST, leaves last), pop each node's "
        "accumulated grad from `grads`, route to leaf-write or dispatch.\n"
        "2. **INNER dispatch** (`dispatch-back-fn-from-recipe`) — for each "
        "`(argnum, parent)` in `node.recipe.parents.items()`, look up "
        "`back_funcs[(node.recipe.func, argnum)]` and call it with "
        "`(grad_out, node.array, *recipe.args, **recipe.kwargs)`.\n\n"
        "**Inputs.**\n"
        "- `end_node` — MiniTensor at which to start the reverse pass.\n"
        "- `end_grad` — `torch.Tensor` with `dL/d(end_node)`. Usually `t.ones_like`.\n"
        "- `sorted_graph` — `list[MiniTensor]` in reverse-topological order.\n"
        "- `back_funcs` — `dict[(forward_fn, argnum), back_fn]` registry.\n\n"
        "**Three invariants the test enforces.**\n"
        "1. **POP, don't peek.** `grads.pop(id(node))` — once popped, the "
        "node's grad is gone from the dict.\n"
        "2. **ACCUMULATE with `+`**, never overwrite — diamond DAGs route "
        "grad through the same parent twice.\n"
        "3. **Leaves write `.grad`; non-leaves stay in `grads`.** Leaf = "
        "`node.recipe is None`. For leaves with `.grad` already set, "
        "accumulate (don't overwrite).\n\n"
        "Return `None`. Mutate `.grad` on each leaf in place."
    ),
    "stub_body": (
        "def cx20_backprop(end_node, end_grad, sorted_graph, back_funcs):\n"
        '    """Reverse-pass driver: pop-outgrad loop + recipe-based dispatch."""\n'
        "    raise NotImplementedError()"
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
        "        self.array = array; self.requires_grad = requires_grad\n"
        "        self.recipe = recipe; self.grad = None\n"
        "\n"
        "# back_fns (raw torch)\n"
        "def log_back(grad_out, out, x): return grad_out / x\n"
        "def mul_back0(grad_out, out, x, y): return grad_out * y\n"
        "def mul_back1(grad_out, out, x, y): return grad_out * x\n"
        "BF = {(t.log, 0): log_back, (t.multiply, 0): mul_back0, (t.multiply, 1): mul_back1}\n"
        "\n"
        "# === TEST 1: x*y — both leaves get correct grads ===\n"
        "x = MiniTensor(t.tensor([2.0, 3.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([5.0, 7.0]), requires_grad=True)\n"
        "out = MiniTensor(x.array * y.array, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y})\n"
        "cx20_backprop(out, t.ones(2), [out, x, y], BF)\n"
        "assert t.allclose(x.grad, y.array), f'd(xy)/dx=y; got {x.grad}'\n"
        "assert t.allclose(y.grad, x.array), f'd(xy)/dy=x; got {y.grad}'\n"
        "\n"
        "# === TEST 2: diamond accumulation z * z → d/dz = 2z ===\n"
        "z = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "out = MiniTensor(z.array * z.array, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.multiply, args=(z.array, z.array), kwargs={}, parents={0: z, 1: z})\n"
        "cx20_backprop(out, t.ones(1), [out, z], BF)\n"
        "assert t.allclose(z.grad, t.tensor([6.0])), f'd(z^2)/dz=2z=6; got {z.grad}'\n"
        "\n"
        "# === TEST 3: chain log(b) then a * c — three-node graph ===\n"
        "import math\n"
        "a = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([math.e]), requires_grad=True)\n"
        "c = MiniTensor(t.log(b.array), requires_grad=True)\n"
        "c.recipe = Recipe(func=t.log, args=(b.array,), kwargs={}, parents={0: b})\n"
        "out = MiniTensor(a.array * c.array, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.multiply, args=(a.array, c.array), kwargs={}, parents={0: a, 1: c})\n"
        "cx20_backprop(out, t.ones(1), [out, a, c, b], BF)\n"
        "assert t.allclose(a.grad, t.tensor([1.0]), atol=1e-5), f'a.grad={a.grad}'\n"
        "assert t.allclose(b.grad, t.tensor([2.0 / math.e]), atol=1e-5), f'b.grad={b.grad}'\n"
        "\n"
        "# === TEST 4: leaf .grad accumulates across calls (does not overwrite) ===\n"
        "a = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "a.grad = t.tensor([10.0])\n"
        "out = MiniTensor(t.log(a.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "cx20_backprop(out, t.ones(1), [out, a], BF)\n"
        "assert t.allclose(a.grad, t.tensor([11.0])), f'must accumulate (10+1=11), got {a.grad}'\n"
        "\n"
        "# === TEST 5: dispatch via (recipe.func, argnum) — asymmetric mul_back0 vs mul_back1 ===\n"
        "# Different x, y values → confirm mul_back0 picked y (not x) and vice versa\n"
        "x = MiniTensor(t.tensor([4.0]), requires_grad=True)\n"
        "y = MiniTensor(t.tensor([11.0]), requires_grad=True)\n"
        "out = MiniTensor(x.array * y.array, requires_grad=True)\n"
        "out.recipe = Recipe(func=t.multiply, args=(x.array, y.array), kwargs={}, parents={0: x, 1: y})\n"
        "cx20_backprop(out, t.ones(1), [out, x, y], BF)\n"
        "assert t.allclose(x.grad, t.tensor([11.0])), f'mul_back0 must pick y=11; got {x.grad}'\n"
        "assert t.allclose(y.grad, t.tensor([4.0])), f'mul_back1 must pick x=4; got {y.grad}'"
    ),
    "solution_body": (
        "def cx20_backprop(end_node, end_grad, sorted_graph, back_funcs):\n"
        "    grads = {id(end_node): end_grad}\n"
        "    for node in sorted_graph:\n"
        "        nid = id(node)\n"
        "        if nid not in grads:\n"
        "            continue                                # not reached this pass\n"
        "        grad_out = grads.pop(nid)                   # POP, don't peek\n"
        "        if node.recipe is None:                     # leaf — write .grad\n"
        "            if node.grad is None:\n"
        "                node.grad = grad_out\n"
        "            else:\n"
        "                node.grad = node.grad + grad_out     # rebind, not in-place\n"
        "            continue\n"
        "        # non-leaf: dispatch each parent\n"
        "        for argnum, parent in node.recipe.parents.items():\n"
        "            back_fn = back_funcs[(node.recipe.func, argnum)]      # dispatch atom\n"
        "            grad_parent = back_fn(\n"
        "                grad_out,\n"
        "                node.array,\n"
        "                *node.recipe.args,\n"
        "                **node.recipe.kwargs,\n"
        "            )\n"
        "            pid = id(parent)\n"
        "            grads[pid] = grads.get(pid, 0) + grad_parent          # accumulate\n"
        "    return None"
    ),
    "solution_notes": (
        "**Two atoms, one loop body.** The outer `for node in sorted_graph` "
        "is the `backprop-pop-outgrad-loop` skeleton; the inner `for argnum, "
        "parent in ...` is the `dispatch-back-fn-from-recipe` step. They're "
        "ALWAYS used together — the dispatch makes no sense without the "
        "driving traversal.\n\n"
        "**Why `grads.pop` and `grads.get(pid, 0) + gp`.** Pop frees the "
        "popped node's grad (no leaks, surfaces double-consumption bugs). "
        "`.get(pid, 0) + gp` handles BOTH (a) first-time touch of a parent "
        "(seed with 0) and (b) diamond accumulation (add to existing entry).\n\n"
        "**Leaves vs non-leaves split.** `node.recipe is None` is the leaf "
        "marker; leaves get `.grad` written (rebind, NOT `+=`), non-leaves "
        "stay in the scratch `grads` dict. Splitting the storage means the "
        "scratch dict can be GC'd at function exit while leaf `.grad`s "
        "persist for the optimizer."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["backprop-pop-outgrad-loop", "dispatch-back-fn-from-recipe"],
    "lo": (
        "Apply the reverse-pass driver pattern: iterate a reverse-topologically "
        "sorted graph, pop each node's accumulated grad, dispatch the per-arg "
        "back_fn via `(recipe.func, argnum)` lookup, and accumulate the result "
        "into each parent's slot — writing `.grad` on leaves and routing "
        "intermediate grads through the scratch dict."
    ),
}


# ---------------------------------------------------------------------------
# cx21 — dispatch-back-fn-from-recipe + parents-dict-by-argidx
# ---------------------------------------------------------------------------

RECAP_21 = (
    "## Composing recipe dispatch with the parents-dict-by-argidx walk\n"
    "\n"
    "When the reverse pass needs the gradients for a non-leaf node's inputs:\n"
    "\n"
    "- **`parents-dict-by-argidx`** — `recipe.parents` is the\n"
    "  `{argnum: parent_tensor}` dict (non-Tensor args already filtered, "
    "  ORIGINAL argnum preserved as the key).\n"
    "- **`dispatch-back-fn-from-recipe`** — for each `(argnum, parent)` "
    "  pair, look up `back_funcs[(recipe.func, argnum)]`.\n"
    "\n"
    "```python\n"
    "def dispatch_all(node, back_funcs):\n"
    "    triples = []\n"
    "    for argnum, parent in node.recipe.parents.items():   # by-argidx walk\n"
    "        back_fn = back_funcs[(node.recipe.func, argnum)] # argnum-keyed dispatch\n"
    "        triples.append((argnum, parent, back_fn))\n"
    "    return triples\n"
    "```\n"
    "\n"
    "**Both atoms are joined at the argnum.** The parents-dict's KEY is the\n"
    "argnum that the dispatcher's LOOKUP needs. If the parents-dict\n"
    "renumbered (e.g. collapsed `{0: t, 2: u}` to `{0: t, 1: u}`), the\n"
    "dispatcher would look up `(fn, 1)` for the second parent and get the\n"
    "WRONG back_fn — that's why parents-dict-by-argidx is strict about\n"
    "preserving the original index.\n"
    "\n"
    "**Asymmetric ops are the test.** `multiply` registers `(mul, 0)` and\n"
    "`(mul, 1)` with the SAME body. `divide` registers `(div, 0)` and `(div,\n"
    "1)` with DIFFERENT bodies (`grad/y` vs `-grad*x/y**2`). If parents are\n"
    "indexed wrong, divide gets the wrong derivative."
)

SPEC_21 = {
    "atom_ids": ["dispatch-back-fn-from-recipe", "parents-dict-by-argidx"],
    "subtopics": _subs(["dispatch-back-fn-from-recipe", "parents-dict-by-argidx"]),
    "primary_atom": "dispatch-back-fn-from-recipe",
    "part": "part4",
    "exercise_index": 21,
    "exercise_title": "dispatch back_fn over each (argnum, parent) in parents dict",
    "slug": "dispatch-all-via-parents-dict-argidx",
    "atom_recap_md": RECAP_21,
    "prompt_body": (
        "Implement `cx21_dispatch_all(args, raw_func, back_funcs)` — given the "
        "RAW positional args at forward-call time (a mix of MiniTensors and "
        "non-Tensors), the forward function, and the back_funcs registry, "
        "return a `list[(argnum, parent, back_fn)]` ready for the back_fn "
        "call site. Two atoms compose:\n\n"
        "**Step 1 — `parents-dict-by-argidx`.** Build a parents dict using "
        "`{idx: a for idx, a in enumerate(args) if isinstance(a, MiniTensor)}`. "
        "MUST keep the ORIGINAL argidx — DO NOT renumber.\n\n"
        "**Step 2 — `dispatch-back-fn-from-recipe`.** For each `(argnum, "
        "parent)` in that dict, look up `back_funcs[(raw_func, argnum)]` and "
        "append `(argnum, parent, back_fn)` to the result list.\n\n"
        "**Test surface.**\n"
        "- `multiply(t, 3.0)` → parents `{0: t}`, dispatched to `(mul, 0)`.\n"
        "- `multiply(3.0, t)` → parents `{1: t}` (argnum stays 1!), "
        "dispatched to `(mul, 1)` NOT `(mul, 0)`.\n"
        "- `divide(x, y)` (asymmetric) → both `(div, 0)` and `(div, 1)` get "
        "the right body.\n"
        "- Missing `(raw_func, argnum)` → propagate `KeyError`.\n\n"
        "**A `MiniTensor` class is provided in the test.**"
    ),
    "stub_body": (
        "def cx21_dispatch_all(args, raw_func, back_funcs) -> list:\n"
        '    """Build parents dict by argidx, dispatch back_fn for each parent."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "class MiniTensor:\n"
        "    def __init__(self, array, requires_grad=False):\n"
        "        self.array = array; self.requires_grad = requires_grad\n"
        "\n"
        "def log_back(grad_out, out, x): return grad_out / x\n"
        "def mul_back0(grad_out, out, x, y): return grad_out * y\n"
        "def mul_back1(grad_out, out, x, y): return grad_out * x\n"
        "def div_back0(grad_out, out, x, y): return grad_out / y\n"
        "def div_back1(grad_out, out, x, y): return -grad_out * x / (y * y)\n"
        "\n"
        "BF = {\n"
        "    (t.log, 0): log_back,\n"
        "    (t.multiply, 0): mul_back0, (t.multiply, 1): mul_back1,\n"
        "    (t.divide, 0): div_back0, (t.divide, 1): div_back1,\n"
        "}\n"
        "\n"
        "# === single parent (log) ===\n"
        "b = MiniTensor(t.tensor([2.0]))\n"
        "triples = cx21_dispatch_all((b,), t.log, BF)\n"
        "assert len(triples) == 1\n"
        "argnum, parent, back_fn = triples[0]\n"
        "assert argnum == 0 and parent is b and back_fn is log_back\n"
        "\n"
        "# === two parents (multiply) — both back_fns from symmetric op ===\n"
        "x = MiniTensor(t.tensor([2.0])); y = MiniTensor(t.tensor([3.0]))\n"
        "triples = cx21_dispatch_all((x, y), t.multiply, BF)\n"
        "assert len(triples) == 2\n"
        "by_argnum = {a: (p, f) for a, p, f in triples}\n"
        "assert by_argnum[0] == (x, mul_back0)\n"
        "assert by_argnum[1] == (y, mul_back1)\n"
        "\n"
        "# === non-Tensor at arg-0 — argnum MUST STAY 1, not collapse to 0 ===\n"
        "a = MiniTensor(t.tensor([7.0]))\n"
        "triples = cx21_dispatch_all((3.0, a), t.multiply, BF)\n"
        "assert len(triples) == 1\n"
        "argnum, parent, back_fn = triples[0]\n"
        "assert argnum == 1, f'argnum must remain 1 (not collapse to 0); got {argnum}'\n"
        "assert parent is a\n"
        "assert back_fn is mul_back1, 'must dispatch to mul_back1 (NOT mul_back0)'\n"
        "\n"
        "# === asymmetric divide — both back_fns must differ ===\n"
        "p, q = MiniTensor(t.tensor([6.0])), MiniTensor(t.tensor([2.0]))\n"
        "triples = cx21_dispatch_all((p, q), t.divide, BF)\n"
        "by_argnum = {a: (par, f) for a, par, f in triples}\n"
        "assert by_argnum[0] == (p, div_back0)\n"
        "assert by_argnum[1] == (q, div_back1)\n"
        "assert by_argnum[0][2] is not by_argnum[1][2], 'div_back0 != div_back1'\n"
        "\n"
        "# === confirm asymmetric math by running the back_fns ===\n"
        "grad_p = by_argnum[0][2](t.ones(1), p.array / q.array, p.array, q.array)\n"
        "grad_q = by_argnum[1][2](t.ones(1), p.array / q.array, p.array, q.array)\n"
        "assert t.allclose(grad_p, t.tensor([0.5])), f'd(p/q)/dp=1/q=0.5; got {grad_p}'\n"
        "assert t.allclose(grad_q, t.tensor([-1.5])), f'd(p/q)/dq=-p/q^2=-1.5; got {grad_q}'\n"
        "\n"
        "# === non-contiguous parents (5, t, (1,2), u) ===\n"
        "u = MiniTensor(t.tensor([1.0]))\n"
        "triples = cx21_dispatch_all((5, x, (1, 2), u), t.multiply, BF)\n"
        "assert len(triples) == 2\n"
        "argnums = sorted(tr[0] for tr in triples)\n"
        "assert argnums == [1, 3], f'expected argnums [1, 3]; got {argnums}'\n"
        "\n"
        "# === missing registration → KeyError ===\n"
        "raised = False\n"
        "try: cx21_dispatch_all((x,), t.sin, BF)\n"
        "except KeyError: raised = True\n"
        "assert raised, 'unregistered (fn, argnum) must propagate KeyError'\n"
        "\n"
        "# === all non-Tensors → empty list ===\n"
        "triples = cx21_dispatch_all((1.0, 2.0, 'x'), t.multiply, BF)\n"
        "assert triples == [], f'all non-Tensors -> empty list, got {triples}'"
    ),
    "solution_body": (
        "def cx21_dispatch_all(args, raw_func, back_funcs):\n"
        "    # Step 1: parents-dict-by-argidx — preserve ORIGINAL idx, skip non-Tensors.\n"
        "    parents = {\n"
        "        idx: a for idx, a in enumerate(args) if isinstance(a, MiniTensor)\n"
        "    }\n"
        "    # Step 2: dispatch-back-fn-from-recipe — (raw_func, argnum) key per parent.\n"
        "    results = []\n"
        "    for argnum, parent in parents.items():\n"
        "        back_fn = back_funcs[(raw_func, argnum)]\n"
        "        results.append((argnum, parent, back_fn))\n"
        "    return results"
    ),
    "solution_notes": (
        "**`enumerate` BEFORE `isinstance` filter — load-bearing.** Doing it "
        "the other way (filter first, then enumerate) re-numbers the "
        "survivors. `(3.0, t)` would yield `{0: t}` instead of `{1: t}` — "
        "and the dispatcher would look up `(mul, 0)` instead of `(mul, 1)`. "
        "For symmetric ops you'd never notice; for divide you'd get the "
        "WRONG derivative.\n\n"
        "**The argnum threads through three layers.** It's the KEY in "
        "`parents-dict-by-argidx`, the LOOKUP component in "
        "`dispatch-back-fn-from-recipe`, and the SELECTOR that picks the "
        "right back_fn body. Renumber it anywhere and the chain breaks.\n\n"
        "**KeyError propagates intentionally.** The caller (the reverse-pass "
        "driver) is the right place to decorate the error with context "
        "('node at depth N in graph X'). The dispatcher just signals 'no "
        "back_fn registered for this (fn, argnum)'."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["dispatch-back-fn-from-recipe", "parents-dict-by-argidx"],
    "lo": (
        "Apply the joint parents-dict-by-argidx + back_fn-dispatch pattern: "
        "build `{argnum: Tensor}` preserving original positional indices, "
        "then look up `back_funcs[(forward_fn, argnum)]` per entry, returning "
        "the `(argnum, parent, back_fn)` triples the reverse-pass driver uses."
    ),
}


# ---------------------------------------------------------------------------
# cx22 — dfs-three-set-toposort + sorted-computational-graph
# ---------------------------------------------------------------------------

RECAP_22 = (
    "## Composing DFS toposort with the sorted-computational-graph wrapper\n"
    "\n"
    "Two atoms, one helper:\n"
    "\n"
    "- **`dfs-three-set-toposort`** — generic DFS that returns descendants\n"
    "  of a node in deps-first order (root LAST).\n"
    "- **`sorted-computational-graph`** — wrap that helper for MiniTensors:\n"
    "  walk via `recipe.parents.values()`, then REVERSE so the end-node\n"
    "  comes FIRST (the order the reverse pass wants).\n"
    "\n"
    "```python\n"
    "def sorted_computational_graph(tensor):\n"
    "    def get_parents(t_):\n"
    "        if t_.recipe is None: return []      # leaves: no parents\n"
    "        return list(t_.recipe.parents.values())\n"
    "    return dfs_topo_sort(tensor, get_parents)[::-1]\n"
    "```\n"
    "\n"
    "**Why the two-layer split.** The DFS topo-sort is generic — reusable\n"
    "for any kind of DAG (forward graph, computation graph, build-system\n"
    "dependency graph). The MiniTensor-specific bit (`recipe.parents` is the\n"
    "parent set; leaves have no recipe) is isolated in the `get_parents`\n"
    "closure. Same DFS skeleton works for both forward and reverse\n"
    "traversals.\n"
    "\n"
    "**The `[::-1]` is the orientation flip.** `dfs_topo_sort` produces\n"
    "deps-first (root LAST). Reverse-pass wants end-node FIRST so it can\n"
    "seed `grads = {id(end): end_grad}` and walk outward. Reversing the\n"
    "deps-first list is the cheapest way to flip orientation without writing\n"
    "a second sort."
)

SPEC_22 = {
    "atom_ids": ["dfs-three-set-toposort", "sorted-computational-graph"],
    "subtopics": _subs(["dfs-three-set-toposort", "sorted-computational-graph"]),
    "primary_atom": "sorted-computational-graph",
    "part": "part4",
    "exercise_index": 22,
    "exercise_title": "build sorted graph from a MiniTensor — DFS over recipe.parents, reversed",
    "slug": "sorted-computational-graph-via-dfs-toposort",
    "atom_recap_md": RECAP_22,
    "prompt_body": (
        "Implement TWO helpers that compose:\n\n"
        "**1. `cx22_topo_sort(root, get_children)`** — generic three-colour "
        "DFS topological sort. Returns descendants of `root` such that `root` "
        "is LAST (deps-first). Raises `ValueError` on a cycle.\n\n"
        "**2. `cx22_sorted_graph(tensor)`** — wrap the above for MiniTensors:\n"
        "- Define `get_parents(t_)`: return `[]` if `t_.recipe is None`, else "
        "`list(t_.recipe.parents.values())`.\n"
        "- Call `cx22_topo_sort(tensor, get_parents)` and REVERSE the result.\n\n"
        "**Contract for `cx22_sorted_graph(tensor)`.**\n"
        "- `result[0] is tensor` — the end node comes FIRST.\n"
        "- `result[-1].recipe is None` — last is some leaf.\n"
        "- Each unique node in the graph appears exactly once.\n"
        "- Reverse-iteration order: every node appears BEFORE all of its "
        "parents (so a parent's accumulator is fully summed by the time the "
        "loop reaches it).\n\n"
        "**Recipe / MiniTensor are provided in the test cell.**"
    ),
    "stub_body": (
        "def cx22_topo_sort(root, get_children):\n"
        '    """Three-colour DFS topo sort, deps-first (root LAST)."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "def cx22_sorted_graph(tensor):\n"
        '    """Reverse-topological order of MiniTensor graph (end node FIRST)."""\n'
        "    raise NotImplementedError()"
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
        "        self.array = array; self.requires_grad = requires_grad\n"
        "        self.recipe = recipe; self.grad = None\n"
        "\n"
        "# === cx22_topo_sort: generic DFS sanity (linear chain) ===\n"
        "class N:\n"
        "    def __init__(self, name, *children):\n"
        "        self.name = name; self.children = list(children)\n"
        "def get_ch(n): return n.children\n"
        "c = N('c'); b = N('b', c); a = N('a', b)\n"
        "order = cx22_topo_sort(a, get_ch)\n"
        "assert [n.name for n in order] == ['c', 'b', 'a'], 'deps-first, root LAST'\n"
        "\n"
        "# === cx22_topo_sort: cycle detection ===\n"
        "x = N('x'); y = N('y')\n"
        "x.children = [y]; y.children = [x]\n"
        "raised = False\n"
        "try: cx22_topo_sort(x, get_ch)\n"
        "except ValueError: raised = True\n"
        "assert raised, 'cycle must raise ValueError'\n"
        "\n"
        "# === cx22_sorted_graph: diamond compute graph ===\n"
        "# leaves: a, b, c; d = a*b; e = log(c); f = d*e; g = log(f).\n"
        "a = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "b = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "c = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "d = MiniTensor(a.array * b.array, requires_grad=True)\n"
        "d.recipe = Recipe(func=t.multiply, args=(a.array, b.array), kwargs={}, parents={0: a, 1: b})\n"
        "e = MiniTensor(t.log(c.array), requires_grad=True)\n"
        "e.recipe = Recipe(func=t.log, args=(c.array,), kwargs={}, parents={0: c})\n"
        "f = MiniTensor(d.array * e.array, requires_grad=True)\n"
        "f.recipe = Recipe(func=t.multiply, args=(d.array, e.array), kwargs={}, parents={0: d, 1: e})\n"
        "g = MiniTensor(t.log(f.array), requires_grad=True)\n"
        "g.recipe = Recipe(func=t.log, args=(f.array,), kwargs={}, parents={0: f})\n"
        "\n"
        "order = cx22_sorted_graph(g)\n"
        "\n"
        "# === end node FIRST ===\n"
        "assert order[0] is g, f'first must be end node g; got {order[0]}'\n"
        "\n"
        "# === all 7 unique nodes present ===\n"
        "assert len(order) == 7, f'expected 7 nodes, got {len(order)}'\n"
        "ids = {id(n) for n in order}\n"
        "assert ids == {id(a), id(b), id(c), id(d), id(e), id(f), id(g)}\n"
        "\n"
        "# === every node appears BEFORE its parents (reverse-topo invariant) ===\n"
        "pos = {id(n): i for i, n in enumerate(order)}\n"
        "assert pos[id(g)] < pos[id(f)], 'g before f'\n"
        "assert pos[id(f)] < pos[id(d)], 'f before d'\n"
        "assert pos[id(f)] < pos[id(e)], 'f before e'\n"
        "assert pos[id(d)] < pos[id(a)], 'd before a'\n"
        "assert pos[id(d)] < pos[id(b)], 'd before b'\n"
        "assert pos[id(e)] < pos[id(c)], 'e before c'\n"
        "\n"
        "# === singleton (just a leaf) ===\n"
        "lonely = MiniTensor(t.tensor([5.0]), requires_grad=True)\n"
        "assert cx22_sorted_graph(lonely) == [lonely], 'singleton'\n"
        "\n"
        "# === composition with cx22_topo_sort: the sorted_graph wrapper IS\n"
        "# `topo_sort(tensor, get_parents)[::-1]` — confirm by manual call\n"
        "def get_parents_fn(t_):\n"
        "    if t_.recipe is None: return []\n"
        "    return list(t_.recipe.parents.values())\n"
        "manual = cx22_topo_sort(g, get_parents_fn)[::-1]\n"
        "assert [id(n) for n in manual] == [id(n) for n in cx22_sorted_graph(g)], (\n"
        "    'cx22_sorted_graph must be cx22_topo_sort(...)[::-1]')"
    ),
    "solution_body": (
        "def cx22_topo_sort(root, get_children):\n"
        "    result = []\n"
        "    perm = set(); temp = set()\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm: return\n"
        "        if nid in temp:\n"
        "            raise ValueError(f'cycle at {node!r}')\n"
        "        temp.add(nid)\n"
        "        for child in get_children(node):\n"
        "            visit(child)\n"
        "        temp.remove(nid); perm.add(nid)\n"
        "        result.append(node)\n"
        "    visit(root)\n"
        "    return result\n"
        "\n"
        "def cx22_sorted_graph(tensor):\n"
        "    def get_parents(t_):\n"
        "        if t_.recipe is None: return []\n"
        "        return list(t_.recipe.parents.values())\n"
        "    return cx22_topo_sort(tensor, get_parents)[::-1]"
    ),
    "solution_notes": (
        "**The two atoms split GENERIC sort from DOMAIN walk.** "
        "`dfs-three-set-toposort` knows nothing about MiniTensors — it just "
        "needs a `get_children` callable. `sorted-computational-graph` "
        "supplies that callable for the MiniTensor recipe graph. Same DFS "
        "engine, different traversal rule.\n\n"
        "**`[::-1]` flips deps-first into end-first.** The generic sort "
        "appends each node AFTER visiting all its children — deps-first, "
        "root last. The reverse pass wants the OPPOSITE — start at the loss "
        "node, walk to the leaves. Reversing is O(N) and keeps both "
        "orderings derivable from one helper.\n\n"
        "**`get_parents` on leaves returns `[]`.** A leaf has `recipe is "
        "None` — no upstream tensors to walk to. Returning `[]` is the DFS "
        "termination signal; the visit appends the leaf and the recursion "
        "unwinds."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["dfs-three-set-toposort", "sorted-computational-graph"],
    "lo": (
        "Apply the generic-DFS + domain-walk composition: write a reusable "
        "three-colour topological sort, then wrap it for the MiniTensor "
        "recipe graph by supplying a `get_parents` closure and reversing the "
        "deps-first output to get end-node-first reverse-pass order."
    ),
}


# ---------------------------------------------------------------------------
# cx23 — grad-accumulate-on-leaf + backprop-pop-outgrad-loop
# ---------------------------------------------------------------------------

RECAP_23 = (
    "## Composing leaf-grad accumulation with the reverse-pass driver\n"
    "\n"
    "When the reverse pass reaches a node, it splits two ways:\n"
    "\n"
    "- **non-leaf** (`recipe is not None`) — dispatch to back_fns, route the\n"
    "  parent grads through the scratch `grads` dict.\n"
    "- **leaf** (`recipe is None`) — write the accumulated grad to "
    "  `node.grad` for the optimizer to consume.\n"
    "\n"
    "The LEAF branch is where `grad-accumulate-on-leaf` lives:\n"
    "\n"
    "```python\n"
    "for node in sorted_graph:                        # pop-outgrad loop\n"
    "    if id(node) not in grads: continue\n"
    "    grad_out = grads.pop(id(node))\n"
    "    if node.recipe is None:                      # LEAF — accumulate\n"
    "        if node.grad is None:\n"
    "            node.grad = grad_out                 # first touch\n"
    "        else:\n"
    "            node.grad = node.grad + grad_out     # rebind, NOT +=\n"
    "        continue\n"
    "    # ... non-leaf dispatch ...\n"
    "```\n"
    "\n"
    "**Critical: rebind, not in-place.** `node.grad = node.grad + g` "
    "creates a new tensor and rebinds. `node.grad += g` mutates the\n"
    "EXISTING grad tensor — and if `optimizer.step()` is holding a\n"
    "reference to that tensor, the snapshot silently changes under it.\n"
    "Rebinding leaves external references alone.\n"
    "\n"
    "**Why accumulate, not overwrite.** A single leaf can be a parent of\n"
    "multiple consumers (e.g. `y = w * w` makes `w` a parent of `y` TWICE\n"
    "via argnums 0 AND 1; weight-tied embedding/unembedding makes a tensor\n"
    "a parent of two separate downstream ops). Each path's contribution is\n"
    "summed into the leaf's grad."
)

SPEC_23 = {
    "atom_ids": ["grad-accumulate-on-leaf", "backprop-pop-outgrad-loop"],
    "subtopics": _subs(["grad-accumulate-on-leaf", "backprop-pop-outgrad-loop"]),
    "primary_atom": "grad-accumulate-on-leaf",
    "part": "part4",
    "exercise_index": 23,
    "exercise_title": "reverse-pass driver with leaf-grad accumulation (rebind, not +=)",
    "slug": "backprop-leaf-grad-accumulation",
    "atom_recap_md": RECAP_23,
    "prompt_body": (
        "Implement `cx23_backprop(end_node, end_grad, sorted_graph, "
        "back_funcs)` — the reverse-pass driver, focused on the LEAF "
        "accumulation step. Two atoms compose:\n\n"
        "**OUTER** (`backprop-pop-outgrad-loop`) — pop `grad_out` from the "
        "scratch `grads` dict for each node in `sorted_graph`.\n\n"
        "**LEAF branch** (`grad-accumulate-on-leaf`) — when `node.recipe "
        "is None`, write to `node.grad`:\n"
        "- If `node.grad is None`: set `node.grad = grad_out` (first touch).\n"
        "- Else: `node.grad = node.grad + grad_out` — **REBIND** with `+`, "
        "NOT in-place `+=`.\n\n"
        "For non-leaf nodes, dispatch each parent via `back_funcs[(recipe.func, "
        "argnum)]` and accumulate into `grads`.\n\n"
        "**The rebind-vs-in-place test is load-bearing.** The test holds an "
        "external reference to a leaf's grad BEFORE a second backward call. "
        "After the second call, the OLD reference MUST be unchanged (rebind "
        "semantics) — the leaf MUST hold a NEW tensor object.\n\n"
        "**The `y = w * w` accumulation test** confirms the multi-path case: "
        "`w` is a parent at argnum 0 AND argnum 1, so two contributions sum "
        "into `w.grad` → final value = `2w`.\n\n"
        "Return `None`."
    ),
    "stub_body": (
        "def cx23_backprop(end_node, end_grad, sorted_graph, back_funcs):\n"
        '    """Reverse-pass driver, focus: leaf-grad accumulation (rebind, not in-place)."""\n'
        "    raise NotImplementedError()"
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
        "        self.array = array; self.requires_grad = requires_grad\n"
        "        self.recipe = recipe; self.grad = None\n"
        "\n"
        "def log_back(grad_out, out, x): return grad_out / x\n"
        "def mul_back0(grad_out, out, x, y): return grad_out * y\n"
        "def mul_back1(grad_out, out, x, y): return grad_out * x\n"
        "BF = {(t.log, 0): log_back, (t.multiply, 0): mul_back0, (t.multiply, 1): mul_back1}\n"
        "\n"
        "# === TEST 1: y = w * w → w is parent at argnum 0 AND 1 → grad = 2w ===\n"
        "w = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "y = MiniTensor(w.array * w.array, requires_grad=True)\n"
        "y.recipe = Recipe(func=t.multiply, args=(w.array, w.array), kwargs={}, parents={0: w, 1: w})\n"
        "cx23_backprop(y, t.ones(1), [y, w], BF)\n"
        "assert t.allclose(w.grad, t.tensor([6.0])), f'd(w^2)/dw=2w=6; got {w.grad}'\n"
        "\n"
        "# === TEST 2: first-touch sets directly (grad was None) ===\n"
        "a = MiniTensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "assert a.grad is None\n"
        "out = MiniTensor(t.log(a.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(a.array,), kwargs={}, parents={0: a})\n"
        "cx23_backprop(out, t.ones(3), [out, a], BF)\n"
        "assert t.allclose(a.grad, 1 / a.array), f'first-touch log: {a.grad}'\n"
        "\n"
        "# === TEST 3: second-touch accumulates (leaf.grad already set) ===\n"
        "b = MiniTensor(t.tensor([2.0]), requires_grad=True)\n"
        "b.grad = t.tensor([100.0])\n"
        "out = MiniTensor(t.log(b.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(b.array,), kwargs={}, parents={0: b})\n"
        "cx23_backprop(out, t.ones(1), [out, b], BF)\n"
        "# expected: 100 + 1/b = 100 + 0.5 = 100.5\n"
        "assert t.allclose(b.grad, t.tensor([100.5])), f'accumulate, not overwrite: {b.grad}'\n"
        "\n"
        "# === TEST 4: REBIND, NOT in-place — load-bearing for optimizer compat ===\n"
        "c = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
        "c.grad = t.tensor([5.0])\n"
        "old_ref = c.grad\n"
        "old_clone = old_ref.clone()\n"
        "out = MiniTensor(t.log(c.array), requires_grad=True)\n"
        "out.recipe = Recipe(func=t.log, args=(c.array,), kwargs={}, parents={0: c})\n"
        "cx23_backprop(out, t.ones(1), [out, c], BF)\n"
        "assert c.grad is not old_ref, 'leaf.grad must REBIND to a new tensor (not +=)'\n"
        "assert t.allclose(old_ref, old_clone), 'external reference must NOT be mutated'\n"
        "\n"
        "# === TEST 5: cross-check against torch.autograd ===\n"
        "w = MiniTensor(t.tensor([3.0]), requires_grad=True)\n"
        "y = MiniTensor(w.array * w.array, requires_grad=True)\n"
        "y.recipe = Recipe(func=t.multiply, args=(w.array, w.array), kwargs={}, parents={0: w, 1: w})\n"
        "cx23_backprop(y, t.ones(1), [y, w], BF)\n"
        "w_ref = t.tensor([3.0], requires_grad=True)\n"
        "y_ref = w_ref * w_ref\n"
        "y_ref.sum().backward()\n"
        "assert t.allclose(w.grad, w_ref.grad), f'must match torch.autograd: ours={w.grad}, ref={w_ref.grad}'"
    ),
    "solution_body": (
        "def cx23_backprop(end_node, end_grad, sorted_graph, back_funcs):\n"
        "    grads = {id(end_node): end_grad}\n"
        "    for node in sorted_graph:\n"
        "        nid = id(node)\n"
        "        if nid not in grads:\n"
        "            continue\n"
        "        grad_out = grads.pop(nid)\n"
        "        if node.recipe is None:\n"
        "            # LEAF — accumulate; REBIND with `+`, not in-place `+=`.\n"
        "            if node.grad is None:\n"
        "                node.grad = grad_out                  # first touch\n"
        "            else:\n"
        "                node.grad = node.grad + grad_out      # rebind!\n"
        "            continue\n"
        "        # non-leaf: dispatch every parent\n"
        "        for argnum, parent in node.recipe.parents.items():\n"
        "            back_fn = back_funcs[(node.recipe.func, argnum)]\n"
        "            gp = back_fn(grad_out, node.array,\n"
        "                         *node.recipe.args, **node.recipe.kwargs)\n"
        "            pid = id(parent)\n"
        "            grads[pid] = grads.get(pid, 0) + gp\n"
        "    return None"
    ),
    "solution_notes": (
        "**`+` rebinds, `+=` mutates.** The optimizer's "
        "`p.data -= lr * p.grad` line typically reads `p.grad` mid-step. "
        "If our backward MUTATES the grad tensor in place via `+=`, that "
        "live reference silently changes value. Rebinding produces a fresh "
        "tensor; the optimizer's snapshot is safe.\n\n"
        "**First-touch sets directly to skip an allocation.** "
        "`node.grad = grad_out` (no `+ 0`) avoids an unnecessary "
        "`t.zeros_like(grad_out)` and an add on the first call per leaf. "
        "Across a model with K parameters, that's K saved allocations per "
        "step.\n\n"
        "**Accumulation is why `zero_grad()` exists.** Because `cx23_backprop` "
        "ALWAYS adds (never overwrites), the previous step's grads stay "
        "around forever unless explicitly cleared. PyTorch's "
        "`optimizer.zero_grad()` exists precisely to clear `.grad` between "
        "training steps."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["grad-accumulate-on-leaf", "backprop-pop-outgrad-loop"],
    "lo": (
        "Apply the reverse-pass driver pattern with proper leaf-grad "
        "accumulation: pop each node's grad from the scratch dict, and for "
        "leaf nodes either set `.grad` on first touch or REBIND "
        "`node.grad = node.grad + grad_out` on subsequent touches so any "
        "externally-held reference to the old grad tensor remains "
        "unmutated."
    ),
}


# ---------------------------------------------------------------------------
# cx24 — back-fn-call-with-recipe-args + kwargs-pass-through-recipe
# ---------------------------------------------------------------------------

RECAP_24 = (
    "## Composing the back_fn call site with kwargs pass-through\n"
    "\n"
    "Two atoms join at the back_fn invocation:\n"
    "\n"
    "- **`kwargs-pass-through-recipe`** — the FORWARD wrapper threads kwargs "
    "  (e.g. `dim`, `keepdim`) into BOTH the forward call AND the Recipe.\n"
    "- **`back-fn-call-with-recipe-args`** — the REVERSE call site splats "
    "  those same kwargs back out: `back_fn(grad_out, node.array, "
    "  *node.recipe.args, **node.recipe.kwargs)`.\n"
    "\n"
    "```python\n"
    "# FORWARD (kwargs-pass-through-recipe):\n"
    "def wrap_forward_fn(fwd):\n"
    "    def tensor_func(*args, **kwargs):\n"
    "        raw = tuple(a.array if isinstance(a, MiniTensor) else a for a in args)\n"
    "        out_raw = fwd(*raw, **kwargs)                       # (1) into call\n"
    "        out = MiniTensor(out_raw)\n"
    "        parents = {i: a for i, a in enumerate(args) if isinstance(a, MiniTensor)}\n"
    "        out.recipe = Recipe(fwd, raw, kwargs, parents)      # (2) into Recipe\n"
    "        return out\n"
    "    return tensor_func\n"
    "\n"
    "# REVERSE (back-fn-call-with-recipe-args):\n"
    "def call_back_fn(back_fn, grad_out, node):\n"
    "    return back_fn(\n"
    "        grad_out,\n"
    "        node.array,                       # raw torch.Tensor, NOT the MiniTensor\n"
    "        *node.recipe.args,                # positional from forward\n"
    "        **node.recipe.kwargs,             # kwargs from forward — same dim/keepdim\n"
    "    )\n"
    "```\n"
    "\n"
    "**The kwargs join forward and reverse.** Without (1), the forward "
    "output is wrong (`sum(x, dim=1)` would reduce over the default axis). "
    "Without (2), the Recipe loses the `dim` — so at reverse time, "
    "`sum_back` has no idea which axis to broadcast back along, and shape "
    "errors blow up. Both halves are load-bearing; this composite tests "
    "them as ONE round-trip."
)

SPEC_24 = {
    "atom_ids": ["back-fn-call-with-recipe-args", "kwargs-pass-through-recipe"],
    "subtopics": _subs(["back-fn-call-with-recipe-args", "kwargs-pass-through-recipe"]),
    "primary_atom": "back-fn-call-with-recipe-args",
    "part": "part4",
    "exercise_index": 24,
    "exercise_title": "wrap_forward + call_back_fn round-trip — kwargs survive both halves",
    "slug": "wrap-forward-and-call-back-with-kwargs",
    "atom_recap_md": RECAP_24,
    "prompt_body": (
        "Implement TWO halves of the kwargs round-trip:\n\n"
        "**1. `cx24_wrap_forward_fn(fwd_fn)`** — closure that returns "
        "`tensor_func(*args, **kwargs)` which:\n"
        "  a. Unboxes each MiniTensor's `.array`; passes non-MiniTensors through.\n"
        "  b. Calls `fwd_fn(*raw_args, **kwargs)` — kwargs MUST reach the call.\n"
        "  c. Boxes the result in `MiniTensor(out_raw)` and attaches "
        "`out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)` — kwargs "
        "MUST be stored.\n"
        "  d. `parents = {idx: a for idx, a in enumerate(args) if "
        "isinstance(a, MiniTensor)}`.\n\n"
        "**2. `cx24_call_back_fn(back_fn, grad_out, node)`** — canonical "
        "back_fn call:\n"
        "  ```python\n"
        "  return back_fn(\n"
        "      grad_out,\n"
        "      node.array,             # raw torch.Tensor, NOT the wrapper\n"
        "      *node.recipe.args,      # positional from forward\n"
        "      **node.recipe.kwargs,   # kwargs from forward\n"
        "  )\n"
        "  ```\n\n"
        "**Test round-trip.** Wrap `t.sum`, call it with `dim=1, keepdim=True`. "
        "Confirm:\n"
        "- forward output has shape `(N, 1)` (proves kwargs reached the call).\n"
        "- `out.recipe.kwargs == {'dim': 1, 'keepdim': True}` (proves they "
        "were stored).\n"
        "- `cx24_call_back_fn` invokes the back_fn with those same kwargs "
        "(proves the reverse side reads them).\n\n"
        "**Three common bugs the test catches.**\n"
        "1. Forgetting `**kwargs` in step 2 → forward call uses wrong defaults.\n"
        "2. Forgetting `kwargs` in step 3 → Recipe has empty kwargs.\n"
        "3. Passing `node` instead of `node.array` to back_fn → back_fn sees "
        "a MiniTensor wrapper, not a raw torch.Tensor.\n\n"
        "**A `MiniTensor` and `Recipe` are provided in the test cell.**"
    ),
    "stub_body": (
        "def cx24_wrap_forward_fn(fwd_fn):\n"
        '    """Return tensor_func that boxes/unboxes and stores kwargs on Recipe."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "def cx24_call_back_fn(back_fn, grad_out, node):\n"
        '    """Invoke back_fn(grad_out, node.array, *recipe.args, **recipe.kwargs)."""\n'
        "    raise NotImplementedError()"
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
        "        self.array = array; self.requires_grad = requires_grad\n"
        "        self.recipe = recipe; self.grad = None\n"
        "\n"
        "# === FORWARD half: kwargs reach BOTH the call and the Recipe ===\n"
        "wrapped_sum = cx24_wrap_forward_fn(t.sum)\n"
        "x = MiniTensor(t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n"
        "out = wrapped_sum(x, dim=1)\n"
        "assert isinstance(out, MiniTensor), 'forward must return MiniTensor'\n"
        "assert t.allclose(out.array, t.tensor([6.0, 15.0])), f'sum(dim=1) wrong: {out.array}'\n"
        "assert out.recipe is not None, 'Recipe must be attached'\n"
        "assert out.recipe.func is t.sum, 'Recipe.func wrong'\n"
        "assert out.recipe.kwargs == {'dim': 1}, f'Recipe.kwargs missing dim: {out.recipe.kwargs}'\n"
        "assert 0 in out.recipe.parents and out.recipe.parents[0] is x\n"
        "\n"
        "# === SECOND kwarg threads through (keepdim) ===\n"
        "out2 = wrapped_sum(x, dim=1, keepdim=True)\n"
        "assert out2.array.shape == (2, 1), f'keepdim ignored: shape={out2.array.shape}'\n"
        "assert out2.recipe.kwargs == {'dim': 1, 'keepdim': True}, out2.recipe.kwargs\n"
        "\n"
        "# === empty-kwargs case still works ===\n"
        "wrapped_log = cx24_wrap_forward_fn(t.log)\n"
        "y = MiniTensor(t.tensor([1.0, t.e, t.e ** 2]))\n"
        "out3 = wrapped_log(y)\n"
        "assert t.allclose(out3.array, t.tensor([0.0, 1.0, 2.0]), atol=1e-5)\n"
        "assert out3.recipe.kwargs == {}, f'empty kwargs case: {out3.recipe.kwargs}'\n"
        "\n"
        "# === args on Recipe are RAW torch.Tensors (unboxed) ===\n"
        "assert isinstance(out.recipe.args[0], t.Tensor)\n"
        "assert not isinstance(out.recipe.args[0], MiniTensor), 'args must be unboxed'\n"
        "\n"
        "# === REVERSE half: back_fn receives kwargs from Recipe ===\n"
        "received = {}\n"
        "def sum_back(grad_out, out_, x_, dim=None, keepdim=False):\n"
        "    received['grad_out_shape'] = tuple(grad_out.shape)\n"
        "    received['out_type'] = type(out_)\n"
        "    received['x_type'] = type(x_)\n"
        "    received['dim'] = dim\n"
        "    received['keepdim'] = keepdim\n"
        "    if dim is None: return grad_out * t.ones_like(x_)\n"
        "    if not keepdim: grad_out = grad_out.unsqueeze(dim)\n"
        "    return grad_out.expand_as(x_)\n"
        "\n"
        "grad_out = t.tensor([1.0, 1.0])\n"
        "result = cx24_call_back_fn(sum_back, grad_out, out)   # out has kwargs={'dim': 1}\n"
        "assert received['dim'] == 1, f'kwargs not threaded: dim={received[\"dim\"]}'\n"
        "assert received['keepdim'] is False, 'keepdim default reaches back_fn'\n"
        "assert received['out_type'] is t.Tensor, (\n"
        "    f'back_fn must get raw torch.Tensor (node.array), got {received[\"out_type\"]}')\n"
        "assert received['x_type'] is t.Tensor, 'recipe.args[0] should be raw'\n"
        "assert result.shape == x.array.shape, f'broadcast back via kwargs: {result.shape}'\n"
        "\n"
        "# === both kwargs reach back_fn (dim AND keepdim) ===\n"
        "received.clear()\n"
        "cx24_call_back_fn(sum_back, t.ones(2, 1), out2)        # out2 has dim=1, keepdim=True\n"
        "assert received['dim'] == 1, received\n"
        "assert received['keepdim'] is True, received\n"
        "\n"
        "# === arbitrary kwarg (scale) round-trips correctly ===\n"
        "def fake_op(x, *, scale=1.0): return x * scale\n"
        "wrapped_fake = cx24_wrap_forward_fn(fake_op)\n"
        "z = MiniTensor(t.tensor([2.0, 3.0]))\n"
        "out_fake = wrapped_fake(z, scale=4.0)\n"
        "assert t.allclose(out_fake.array, t.tensor([8.0, 12.0])), out_fake.array\n"
        "assert out_fake.recipe.kwargs == {'scale': 4.0}\n"
        "spy = {}\n"
        "def fake_back(grad_out, out_, x_, *, scale=1.0):\n"
        "    spy['scale'] = scale\n"
        "    return grad_out * scale\n"
        "g = cx24_call_back_fn(fake_back, t.ones(2), out_fake)\n"
        "assert spy['scale'] == 4.0, f'scale must reach back_fn: {spy}'\n"
        "assert t.allclose(g, t.tensor([4.0, 4.0])), g"
    ),
    "solution_body": (
        "def cx24_wrap_forward_fn(fwd_fn):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        # Unbox MiniTensors → raw torch.Tensors; pass-through non-MiniTensors.\n"
        "        raw_args = tuple(\n"
        "            a.array if isinstance(a, MiniTensor) else a for a in args\n"
        "        )\n"
        "        # (1) forward MUST receive kwargs\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        # parents-dict-by-argidx (composed elsewhere) — keep original idx.\n"
        "        parents = {\n"
        "            idx: a for idx, a in enumerate(args) if isinstance(a, MiniTensor)\n"
        "        }\n"
        "        out = MiniTensor(out_raw)\n"
        "        # (2) kwargs MUST also be stored on the Recipe — for the reverse call.\n"
        "        out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func\n"
        "\n"
        "def cx24_call_back_fn(back_fn, grad_out, node):\n"
        "    # Canonical four-channel invocation — note BOTH splats.\n"
        "    return back_fn(\n"
        "        grad_out,\n"
        "        node.array,                     # raw torch.Tensor, NOT MiniTensor\n"
        "        *node.recipe.args,              # positional from forward\n"
        "        **node.recipe.kwargs,           # kwargs from forward (dim, keepdim, ...)\n"
        "    )"
    ),
    "solution_notes": (
        "**Two atoms, one round-trip.** `kwargs-pass-through-recipe` is the "
        "FORWARD half — kwargs reach (a) the actual call and (b) the stored "
        "Recipe. `back-fn-call-with-recipe-args` is the REVERSE half — "
        "kwargs are splatted out of the Recipe and into the back_fn. Drop "
        "either half and the round-trip breaks; only the COMPOSITION yields "
        "a working `sum`/`mean`/`max` autograd op.\n\n"
        "**Three splats matter.** `*raw_args` unboxes the positional inputs; "
        "`*node.recipe.args` re-splats them on the way back; "
        "`**node.recipe.kwargs` re-splats the keyword args. Drop any of "
        "the three and a real autograd implementation fails on shape "
        "mismatches deep inside the back_fn.\n\n"
        "**`node.array`, not `node`.** Back_fns operate on raw torch tensors "
        "so they can do tensor math without unboxing in each body. Passing "
        "the MiniTensor wrapper would force every back_fn body to start with "
        "`out.array` — defeating the whole point of the unboxing wrapper."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["back-fn-call-with-recipe-args", "kwargs-pass-through-recipe"],
    "lo": (
        "Apply the kwargs round-trip: in the forward wrapper, thread `**kwargs` "
        "into both the forward call and the stored Recipe; in the reverse "
        "call site, splat `*node.recipe.args` and `**node.recipe.kwargs` "
        "into the back_fn alongside `grad_out` and `node.array` so the "
        "back_fn replays the op with the same keyword arguments the forward "
        "used."
    ),
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    for spec in (SPEC_19, SPEC_20, SPEC_21, SPEC_22, SPEC_23, SPEC_24):
        path = emit_composite(spec)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
