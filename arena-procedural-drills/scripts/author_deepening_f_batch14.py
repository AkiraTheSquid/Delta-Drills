#!/usr/bin/env python3
"""Author 8 ex3 deepening drills (batch 14, group F — backprop + driver cluster).

Atoms:
  prereqs_backprop/
    - buffer-copy_-inplace  (ex3: contrast copy_() with mul_/add_ EMA recipe)
    - param-grad-access      (ex3: classify grads NaN/zero/finite — diagnostic histogram)
  prereqs_backprop_driver/
    - back-fn-call-with-recipe-args (ex3: validate grad_out.shape == node.array.shape pre-call)
    - backprop-pop-outgrad-loop     (ex3: instrument with per-node max|grad| reverse-trace)
    - cycle-detection-temp-set      (ex3: enumerate ALL cycles, not just first)
    - dfs-three-set-toposort        (ex3: deterministic toposort with child sort-key tiebreaker)
    - dispatch-back-fn-from-recipe  (ex3: fallback to (fn, None) wildcard when (fn, argnum) missing)
    - grad-expressed-in-out         (ex3: softplus_back via cached out — third activation)

Each ex3 = DISTINCT third facet from ex1/ex2. ONE LO + ONE Bloom + <=2 KCs per drill.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_BP = "prereqs_backprop"
TOPIC_BPD = "prereqs_backprop_driver"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_COPY_VS_MULADD = (
    "## `copy_()` EMA vs in-place `mul_(...).add_(..., alpha=m)` recipe — same buffer, two routes\n"
    "\n"
    "Ex1 wrote the EMA update through `running_mean.copy_(new_value)` where\n"
    "`new_value = (1 - m) * running_mean + m * batch_mean` was built as a\n"
    "fresh tensor first. The deepening move is to do the SAME math WITHOUT\n"
    "ever allocating that intermediate — purely in-place:\n"
    "\n"
    "```python\n"
    "# Route A (ex1): allocate a temp, then copy_ overwrites the buffer.\n"
    "running_mean.copy_((1 - m) * running_mean + m * batch_mean)\n"
    "\n"
    "# Route B (ex3): two chained in-place ops, zero temps.\n"
    "running_mean.mul_(1 - m).add_(batch_mean, alpha=m)\n"
    "```\n"
    "\n"
    "**Both preserve `data_ptr()`.** `copy_`, `mul_`, and `add_` all write\n"
    "into the existing storage. So registered-buffer links survive either\n"
    "route — `id(bn.running_mean)` is the same before and after.\n"
    "\n"
    "**Route B allocates zero extra tensors.** Route A allocates two\n"
    "intermediates (`(1-m)*running_mean` and `m*batch_mean`) plus a third for\n"
    "the sum — all GC-able but allocator-churning in a tight loop.\n"
    "\n"
    "**`add_(other, alpha=m)` is the standard idiom.** `tensor.add_(other,\n"
    "alpha=k)` computes `tensor += k * other` in place without materializing\n"
    "`k * other`. This is what `torch.optim.SGD.step()` uses internally for\n"
    "the momentum buffer."
)

RECAP_GRAD_HEALTH_CLASSIFY = (
    "## Grad health classification — three buckets per param\n"
    "\n"
    "Ex1 applied the SGD step (`p.data -= lr * p.grad`). Ex2 collected per-\n"
    "param L2 norms. The deepening move is a DIAGNOSTIC view: for each\n"
    "parameter, ask `is this grad healthy?` and bucket it into one of three\n"
    "categories:\n"
    "\n"
    "| Bucket   | Condition                                       | Meaning                                |\n"
    "|----------|--------------------------------------------------|----------------------------------------|\n"
    "| `'nan'`  | `t.isnan(p.grad).any()` or `t.isinf(p.grad).any()` | numerical blow-up — abort the step    |\n"
    "| `'zero'` | all-finite AND `p.grad.abs().max() == 0`         | dead neuron / disconnected layer       |\n"
    "| `'ok'`   | all-finite AND any nonzero element               | healthy — training can proceed         |\n"
    "\n"
    "Params with `.grad is None` are SKIPPED — they didn't participate in\n"
    "the forward pass, so there's nothing to diagnose.\n"
    "\n"
    "**Why classify rather than norm.** A grad of L2-norm `1e-12` and a grad\n"
    "of `nan` are both 'small' under the norm view but have completely\n"
    "different operational meanings. The three-bucket classification gives\n"
    "you an unambiguous trigger surface — `'nan'` → skip step + maybe halve\n"
    "lr; `'zero'` → check upstream layer connectivity; `'ok'` → proceed.\n"
    "\n"
    "**Order of checks matters.** Test `nan/inf` FIRST. A tensor of all-NaN\n"
    "satisfies `abs().max() == nan != 0`, so the zero-check is meaningless\n"
    "until NaN is ruled out."
)

RECAP_PRECALL_SHAPE = (
    "## Pre-call shape validation — fail loudly before the back_fn runs\n"
    "\n"
    "Ex1 invoked the back_fn with the canonical four-arg shape; ex2 recorded\n"
    "what each channel received. The deepening move is to ASSERT shape\n"
    "compatibility BEFORE the call. If `grad_out.shape != node.array.shape`,\n"
    "we know the dispatcher already broke the chain rule — raise a precise\n"
    "error naming both shapes instead of letting the back_fn produce a\n"
    "cryptic broadcast/elementwise failure deep inside its math.\n"
    "\n"
    "```python\n"
    "def call_back_fn_validated(back_fn, grad_out, node):\n"
    "    if grad_out.shape != node.array.shape:\n"
    "        raise ValueError(\n"
    "            f'grad_out.shape {tuple(grad_out.shape)} != node.array.shape '\n"
    "            f'{tuple(node.array.shape)} — upstream dispatcher bug'\n"
    "        )\n"
    "    return back_fn(grad_out, node.array,\n"
    "                   *node.recipe.args, **node.recipe.kwargs)\n"
    "```\n"
    "\n"
    "**Why it's worth a guard.** In a real autograd, a `grad_out`/`out`\n"
    "shape mismatch typically surfaces as `RuntimeError: The size of tensor\n"
    "a (5) must match the size of tensor b (3) at non-singleton dimension\n"
    "1` inside someone else's elementwise op — six call-frames deep, no\n"
    "context about which node failed. A pre-call assert pinpoints the\n"
    "dispatcher as the culprit immediately.\n"
    "\n"
    "**`tuple(shape)` for the message.** `torch.Size([3, 5])` reprs as\n"
    "`torch.Size([3, 5])` which is verbose. Tupling first gives `(3, 5)`."
)

RECAP_REVERSE_MAXABS_TRACE = (
    "## Per-node max|grad| trace — vanishing/exploding diagnostic\n"
    "\n"
    "Ex1 ran the reverse-pass driver; ex2 counted per-leaf accumulations.\n"
    "The deepening move tracks the L∞ MAGNITUDE of each `grad_out` as it\n"
    "flows back — a fingerprint of vanishing-gradient (values → 0 deep in\n"
    "the graph) or exploding-gradient (values → ∞) pathologies.\n"
    "\n"
    "```python\n"
    "def backprop_traced(end_node, end_grad, sorted_graph, back_funcs):\n"
    "    grads = {id(end_node): end_grad}\n"
    "    trace = []  # [(node_id, max_abs_grad_out), ...] in pop order\n"
    "    for node in sorted_graph:\n"
    "        if id(node) not in grads: continue\n"
    "        grad_out = grads.pop(id(node))\n"
    "        trace.append((id(node), float(grad_out.abs().max())))\n"
    "        # ... rest of driver as ex1 ...\n"
    "    return trace\n"
    "```\n"
    "\n"
    "**Why max|.| not norm.** L∞ catches a SINGLE explosive element — one\n"
    "rogue activation that's about to overflow on the next forward pass.\n"
    "L2 averages it out. For health monitoring, the worst element is the\n"
    "right signal.\n"
    "\n"
    "**Trace order matches reverse-pass order.** The first entry is\n"
    "`end_node`; the last entries are leaves. A monotone-decreasing trace\n"
    "is the vanishing signature; an increasing one is exploding. Most real\n"
    "graphs show a noisy mix — but a clean monotone pattern over 50+ layers\n"
    "is what RNN tutorials famously visualize."
)

RECAP_ENUMERATE_ALL_CYCLES = (
    "## Enumerate ALL cycles — multi-SCC view, not just the first\n"
    "\n"
    "Ex1 returned `True` on any back-edge; ex2 returned ONE cycle path.\n"
    "The deepening move is to find EVERY distinct cycle reachable from\n"
    "`root`. A graph with two disjoint cycles `(a→b→a)` and `(c→d→e→c)`\n"
    "should produce a list of TWO paths.\n"
    "\n"
    "```python\n"
    "def all_cycles(root, get_children):\n"
    "    cycles = []           # accumulator across the whole walk\n"
    "    on_stack = set()\n"
    "    perm = set()\n"
    "    path = []\n"
    "    def visit(node):\n"
    "        nid = id(node)\n"
    "        if nid in perm: return\n"
    "        if nid in on_stack:\n"
    "            i = next(j for j, n in enumerate(path) if id(n) == nid)\n"
    "            cycles.append(path[i:] + [node])\n"
    "            return         # do NOT halt — keep exploring other branches\n"
    "        on_stack.add(nid); path.append(node)\n"
    "        for child in get_children(node):\n"
    "            visit(child)\n"
    "        on_stack.discard(nid); path.pop()\n"
    "        perm.add(nid)\n"
    "    visit(root)\n"
    "    return cycles\n"
    "```\n"
    "\n"
    "**Critical change from ex2: don't return on back-edge.** Ex2 returned\n"
    "the first cycle and aborted. To enumerate ALL cycles, the back-edge\n"
    "case must RECORD and CONTINUE — explore siblings of the current node.\n"
    "Forgetting this is the universal bug.\n"
    "\n"
    "**Dedup is the user's problem.** Two paths that traverse the same\n"
    "cycle starting from different nodes are technically distinct here.\n"
    "The drill returns paths as-discovered; canonicalization (rotate to\n"
    "smallest-id-first) is left to the caller — out of scope for the\n"
    "core enumeration loop."
)

RECAP_DETERMINISTIC_TOPOSORT = (
    "## Deterministic toposort — sort children by key for reproducible output\n"
    "\n"
    "Ex1's recursive toposort and ex2's iterative variant both produce a\n"
    "valid order — but a different valid order if `get_children` returns\n"
    "children in different iteration orders. The deepening move enforces\n"
    "DETERMINISM by sorting children by a caller-supplied key before\n"
    "recursing.\n"
    "\n"
    "```python\n"
    "def topological_sort_keyed(root, get_children, key):\n"
    "    perm, temp = set(), set()\n"
    "    result = []\n"
    "    def visit(node):\n"
    "        nid = id(node)\n"
    "        if nid in perm: return\n"
    "        if nid in temp: raise ValueError('cycle')\n"
    "        temp.add(nid)\n"
    "        # Sort children BEFORE recursing → output is now key-deterministic.\n"
    "        for child in sorted(get_children(node), key=key):\n"
    "            visit(child)\n"
    "        temp.remove(nid)\n"
    "        perm.add(nid)\n"
    "        result.append(node)\n"
    "    visit(root)\n"
    "    return result\n"
    "```\n"
    "\n"
    "**Why determinism matters.** Two CI runs over the same graph should\n"
    "yield the same toposort. Without a tiebreaker, dict-iteration order\n"
    "(Python 3.7+ preserves insertion order but the INSERTION order can\n"
    "still vary by user code) leaks into the result, making cache\n"
    "invalidation and test diffs noisy.\n"
    "\n"
    "**Deps-first invariant is unchanged.** Sorting children only reorders\n"
    "SIBLINGS at each level. Every node still appears AFTER all of its\n"
    "(transitive) children — the toposort guarantee survives the\n"
    "tiebreaker.\n"
    "\n"
    "**`key=key`, not `key=str`.** Hardcoding `str` would force nodes to\n"
    "have a string representation; the caller-passed key gives the user\n"
    "control (e.g. by node name, by topo-index, by registration order)."
)

RECAP_WILDCARD_DISPATCH = (
    "## Wildcard fallback — `(fn, None)` matches any argnum\n"
    "\n"
    "Ex1 looked up `(recipe.func, argnum)`; ex2 wrapped the lookup with a\n"
    "friendly KeyError. The deepening move handles a real registry pattern:\n"
    "some back_fns are SYMMETRIC across all argnums (e.g. `add_back` for\n"
    "`x + y` returns the same `grad_out` regardless of argnum). The\n"
    "registry stores them as `(fn, None)` and the dispatcher falls back to\n"
    "this wildcard when no exact match is found.\n"
    "\n"
    "```python\n"
    "back_funcs = {\n"
    "    (multiply, 0): mul_back_0,   # specific — needs y\n"
    "    (multiply, 1): mul_back_1,   # specific — needs x\n"
    "    (add, None):   add_back,     # WILDCARD — same for any argnum\n"
    "}\n"
    "\n"
    "def lookup(fn, argnum):\n"
    "    if (fn, argnum) in back_funcs:\n"
    "        return back_funcs[(fn, argnum)]   # exact match wins\n"
    "    if (fn, None) in back_funcs:\n"
    "        return back_funcs[(fn, None)]     # wildcard fallback\n"
    "    raise KeyError(f'No back_fn for ({fn.__name__}, {argnum})')\n"
    "```\n"
    "\n"
    "**Precedence: exact before wildcard.** If both `(add, 0)` and\n"
    "`(add, None)` are registered, `argnum=0` MUST resolve to the exact\n"
    "match. The wildcard only fires when no specific entry exists.\n"
    "\n"
    "**Why this is the production pattern.** Hand-registering `(fn, 0)`,\n"
    "`(fn, 1)`, ..., `(fn, k)` for every k-ary symmetric op is bloat. The\n"
    "wildcard lets the registry stay sparse: one entry per op family, with\n"
    "argnum-specific overrides only where the math differs."
)

RECAP_SOFTPLUS_BACK = (
    "## softplus_back via cached out — third activation, same pattern\n"
    "\n"
    "Ex1 wrote `sigmoid_back = grad_out * out * (1 - out)`. Ex2 wrote\n"
    "`tanh_back = grad_out * (1 - out**2)`. The deepening move applies the\n"
    "SAME 'grad expressed in `out`' template to a THIRD activation —\n"
    "`softplus(x) = log(1 + exp(x))` — to confirm you've internalized the\n"
    "pattern rather than memorized two specific formulas.\n"
    "\n"
    "**Math.**\n"
    "```\n"
    "out = softplus(x) = log(1 + exp(x))\n"
    "\n"
    "d/dx softplus(x) = exp(x) / (1 + exp(x))\n"
    "                 = sigmoid(x)\n"
    "                 = 1 - exp(-out)        ← expressed in out\n"
    "```\n"
    "\n"
    "The last step uses `exp(-out) = exp(-log(1+exp(x))) = 1/(1+exp(x))`,\n"
    "so `1 - exp(-out) = exp(x)/(1+exp(x)) = sigmoid(x)`.\n"
    "\n"
    "**Chain rule.**\n"
    "```\n"
    "dL/dx = grad_out * (1 - t.exp(-out))\n"
    "```\n"
    "\n"
    "**Why this transfer test matters.** Sigmoid and tanh have algebraically\n"
    "obvious closed forms in `out`. Softplus needs ONE rewrite to express\n"
    "the derivative in `out` — and that rewrite (`sigmoid(x) = 1 -\n"
    "exp(-softplus(x))`) is the test of whether you understood why caching\n"
    "`out` is even useful. The drill explicitly forbids calling `t.sigmoid`\n"
    "or recomputing `softplus` from `x`."
)


# ---------------------------------------------------------------------------
# SPEC 1 — buffer-copy_-inplace  ex3
# ---------------------------------------------------------------------------

SPEC_BUFFER_COPY = {
    "atom_id": "buffer-copy_-inplace",
    "subtopic": "PyTorch: in-place buffer copy",
    "topic_folder": TOPIC_BP,
    "atom_recap_md": RECAP_COPY_VS_MULADD,
    "exercise_index": 3,
    "exercise_title": "in-place EMA via mul_(1-m).add_(other, alpha=m) — zero-temp recipe",
    "slug": "in-place-ema-via-mul-add-zero-temp-recipe",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["mul_", "add_", "ema", "in-place", "zero-temp"],
    "kcs": [
        "in-place-mul-add-recipe",
        "data-ptr-preserved-in-place",
    ],
    "lo": (
        "Apply the chained `mul_(1-m).add_(other, alpha=m)` in-place EMA "
        "recipe to update a registered BatchNorm-style buffer without "
        "allocating any intermediate tensors, preserving the buffer's "
        "`data_ptr()` and `id()` identity."
    ),
    "prompt_body": (
        "Implement `ex3_ema_inplace(running_mean, batch_mean, momentum)`. "
        "Update `running_mean` IN PLACE to "
        "`(1 - momentum) * running_mean + momentum * batch_mean` using "
        "EXACTLY two chained in-place ops on `running_mean` — no temporary "
        "tensor allocation, no `.copy_(...)` call.\n\n"
        "Required recipe:\n\n"
        "```python\n"
        "running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)\n"
        "```\n\n"
        "Inputs:\n"
        "- `running_mean`: `Tensor`. The buffer to mutate.\n"
        "- `batch_mean`: `Tensor` of the same shape as `running_mean`.\n"
        "- `momentum`: `float` in `[0.0, 1.0]`.\n\n"
        "Return value: `None`. The function mutates `running_mean` in place. "
        "The test asserts both `id(running_mean)` and `running_mean.data_ptr()` "
        "are preserved across the call — proof the registered-buffer link "
        "survives.\n\n"
        "Constraints:\n"
        "- DO NOT use `.copy_(...)` (ex1 already covered that route).\n"
        "- DO NOT create a fresh tensor via `1 - m * running_mean + ...` "
        "and assign — that breaks the zero-temp invariant.\n"
        "- DO NOT use `t.lerp_` either — the assertion is that you can hand-"
        "build the EMA from `mul_` and `add_`."
    ),
    "stub": (
        "def ex3_ema_inplace(running_mean: Tensor, batch_mean: Tensor, momentum: float) -> None:\n"
        '    """Zero-temp in-place EMA: running_mean.mul_(1-m).add_(batch_mean, alpha=m)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # === Identity preservation: id() AND data_ptr() unchanged ===\n"
        "    rm = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "    bm = t.tensor([5.0, 6.0, 7.0, 8.0])\n"
        "    id_before = id(rm)\n"
        "    ptr_before = rm.data_ptr()\n"
        "    out = ex3_ema_inplace(rm, bm, 0.1)\n"
        "    assert out is None, 'function must return None (mutates in place)'\n"
        "    assert id(rm) == id_before, f'id changed — wrong route; expected {id_before}, got {id(rm)}'\n"
        "    assert rm.data_ptr() == ptr_before, f'data_ptr changed — buffer reallocated; expected {ptr_before}, got {rm.data_ptr()}'\n"
        "\n"
        "    # === Numeric correctness: (1-m)*rm + m*bm ===\n"
        "    expected = 0.9 * t.tensor([1.0, 2.0, 3.0, 4.0]) + 0.1 * t.tensor([5.0, 6.0, 7.0, 8.0])\n"
        "    assert t.allclose(rm, expected, atol=1e-7), f'numeric mismatch: got {rm}, expected {expected}'\n"
        "\n"
        "    # === momentum=0 → buffer unchanged ===\n"
        "    rm = t.tensor([1.0, 2.0])\n"
        "    bm = t.tensor([99.0, 99.0])\n"
        "    ex3_ema_inplace(rm, bm, 0.0)\n"
        "    assert t.allclose(rm, t.tensor([1.0, 2.0])), f'momentum=0 should leave rm unchanged; got {rm}'\n"
        "\n"
        "    # === momentum=1 → buffer becomes batch_mean ===\n"
        "    rm = t.tensor([1.0, 2.0])\n"
        "    bm = t.tensor([5.0, 6.0])\n"
        "    ex3_ema_inplace(rm, bm, 1.0)\n"
        "    assert t.allclose(rm, t.tensor([5.0, 6.0])), f'momentum=1 should make rm == bm; got {rm}'\n"
        "\n"
        "    # === Works on a real registered buffer ===\n"
        "    import torch.nn as nn\n"
        "    class TinyBN(nn.Module):\n"
        "        def __init__(self, n):\n"
        "            super().__init__()\n"
        "            self.register_buffer('running_mean', t.zeros(n))\n"
        "    bn = TinyBN(3)\n"
        "    original_id = id(bn.running_mean)\n"
        "    original_ptr = bn.running_mean.data_ptr()\n"
        "    ex3_ema_inplace(bn.running_mean, t.tensor([10.0, 20.0, 30.0]), 0.5)\n"
        "    assert id(bn.running_mean) == original_id, 'registered buffer id broken'\n"
        "    assert bn.running_mean.data_ptr() == original_ptr, 'registered buffer storage broken'\n"
        "    assert t.allclose(bn.running_mean, t.tensor([5.0, 10.0, 15.0])), bn.running_mean\n"
        "\n"
        "    # === Repeated calls — successive EMA steps converge toward batch_mean ===\n"
        "    rm = t.zeros(2)\n"
        "    bm = t.tensor([1.0, 1.0])\n"
        "    ptr_initial = rm.data_ptr()\n"
        "    for _ in range(100):\n"
        "        ex3_ema_inplace(rm, bm, 0.1)\n"
        "    assert rm.data_ptr() == ptr_initial, 'data_ptr changed across iterations'\n"
        "    # 100 iters of EMA with m=0.1 toward [1, 1] should be very close.\n"
        "    assert t.allclose(rm, t.ones(2), atol=1e-3), f'EMA failed to converge: {rm}'\n"
        "\n"
        "    # === Shape: 2-D buffer (matches BatchNorm2d affine running stats) ===\n"
        "    rm = t.zeros(3, 4)\n"
        "    bm = t.ones(3, 4)\n"
        "    ex3_ema_inplace(rm, bm, 0.25)\n"
        "    assert t.allclose(rm, 0.25 * t.ones(3, 4)), rm\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_ema_inplace(running_mean, batch_mean, momentum):\n"
        "    # Zero-temp in-place EMA. mul_ scales running_mean by (1-m) in place;\n"
        "    # add_ with alpha=m adds m*batch_mean without materializing the product.\n"
        "    running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)"
    ),
    "solution_notes": (
        "**`add_(other, alpha=k)` avoids the multiply temp.** Without the "
        "`alpha=` kwarg you'd write `running_mean.add_(momentum * "
        "batch_mean)` — which allocates `momentum * batch_mean` first. The "
        "`alpha=` form does the multiply-and-add as a single fused kernel.\n\n"
        "**Chain order matters.** `mul_(1-m).add_(other, alpha=m)` computes "
        "`((1-m) * rm) + m * other`. The reverse — `add_` first then `mul_` — "
        "would compute `(rm + other) * (1-m)`, a completely different "
        "expression.\n\n"
        "**Why this isn't just style.** In a training loop with N modules "
        "and K BatchNorm layers each, the EMA update runs K*N times per "
        "step. Replacing three temps with zero per call adds up to "
        "measurable allocator savings on a hot path."
    ),
    "extra_imports": ["import torch.nn as nn"],
}


# ---------------------------------------------------------------------------
# SPEC 2 — param-grad-access  ex3
# ---------------------------------------------------------------------------

SPEC_PARAM_GRAD = {
    "atom_id": "param-grad-access",
    "subtopic": "PyTorch: param.grad access",
    "topic_folder": TOPIC_BP,
    "atom_recap_md": RECAP_GRAD_HEALTH_CLASSIFY,
    "exercise_index": 3,
    "exercise_title": "classify each grad as nan, zero, or ok — three-bucket health diagnostic",
    "slug": "classify-grad-health-nan-zero-ok",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["nan", "inf", "zero-grad", "diagnostic", "health"],
    "kcs": [
        "param-grad-none-guard-skip",
        "nan-before-zero-check-order",
    ],
    "lo": (
        "Analyze each parameter's `.grad` tensor and classify it as "
        "`'nan'` (any NaN/Inf), `'zero'` (all finite + all zero), or "
        "`'ok'` (all finite + at least one nonzero), skipping params "
        "whose `.grad is None`."
    ),
    "prompt_body": (
        "Implement `ex3_classify_grad_health(model)`. Return a "
        "`dict[str, str]` mapping each parameter NAME (from "
        "`model.named_parameters()`) to one of three labels:\n\n"
        "- `'nan'` — if `t.isnan(p.grad).any()` OR `t.isinf(p.grad).any()`. "
        "This check MUST come first (a tensor of all-NaN reads as nonzero "
        "to the zero-check, so order matters).\n"
        "- `'zero'` — all finite AND `p.grad.abs().max().item() == 0.0`.\n"
        "- `'ok'` — all finite AND any element nonzero.\n\n"
        "Parameters whose `.grad is None` MUST be OMITTED from the dict "
        "(not present as `'none'`, just absent). This matches the "
        "ex2 convention.\n\n"
        "Required ordering:\n"
        "1. None-guard: skip if `p.grad is None`.\n"
        "2. NaN/Inf check FIRST.\n"
        "3. Zero check second.\n"
        "4. Default to `'ok'`.\n\n"
        "Return type: `dict[str, str]`. The test feeds a model with a "
        "mix of all four cases (none, nan, zero, ok) and asserts exact "
        "labels per parameter."
    ),
    "stub": (
        "def ex3_classify_grad_health(model) -> dict:\n"
        '    """Return {param-name: nan/zero/ok}, skipping params with .grad is None."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    import torch.nn as nn\n"
        "\n"
        "    # Build a model with four named params so we can hand-set each .grad.\n"
        "    class FourParams(nn.Module):\n"
        "        def __init__(self):\n"
        "            super().__init__()\n"
        "            self.p_none = nn.Parameter(t.zeros(3))\n"
        "            self.p_nan  = nn.Parameter(t.zeros(3))\n"
        "            self.p_zero = nn.Parameter(t.zeros(3))\n"
        "            self.p_ok   = nn.Parameter(t.zeros(3))\n"
        "\n"
        "    model = FourParams()\n"
        "    # leave model.p_none.grad as None\n"
        "    model.p_nan.grad  = t.tensor([1.0, float('nan'), 3.0])\n"
        "    model.p_zero.grad = t.zeros(3)\n"
        "    model.p_ok.grad   = t.tensor([0.1, 0.2, 0.3])\n"
        "\n"
        "    out = ex3_classify_grad_health(model)\n"
        "    assert isinstance(out, dict), f'must return dict, got {type(out).__name__}'\n"
        "    assert 'p_none' not in out, f'p_none has grad=None; must be OMITTED, got {out}'\n"
        "    assert out.get('p_nan') == 'nan', f'p_nan should be nan; got {out}'\n"
        "    assert out.get('p_zero') == 'zero', f'p_zero should be zero; got {out}'\n"
        "    assert out.get('p_ok') == 'ok', f'p_ok should be ok; got {out}'\n"
        "    assert len(out) == 3, f'should have 3 entries (p_none omitted); got {out}'\n"
        "\n"
        "    # === Inf is classified as 'nan' (same bucket — numerical blow-up) ===\n"
        "    model.p_ok.grad = t.tensor([1.0, float('inf'), 2.0])\n"
        "    out = ex3_classify_grad_health(model)\n"
        "    assert out['p_ok'] == 'nan', f'inf grad should bucket to nan; got {out}'\n"
        "\n"
        "    # === Negative inf also bucketed as 'nan' ===\n"
        "    model.p_ok.grad = t.tensor([1.0, float('-inf'), 2.0])\n"
        "    out = ex3_classify_grad_health(model)\n"
        "    assert out['p_ok'] == 'nan', f'-inf grad should bucket to nan; got {out}'\n"
        "\n"
        "    # === All-NaN tensor: zero-check would erroneously hit, so order is the test ===\n"
        "    model.p_ok.grad = t.tensor([float('nan'), float('nan'), float('nan')])\n"
        "    out = ex3_classify_grad_health(model)\n"
        "    assert out['p_ok'] == 'nan', f'all-NaN must classify as nan (order test); got {out}'\n"
        "\n"
        "    # === A grad with a single nonzero element is 'ok', not 'zero' ===\n"
        "    model.p_ok.grad = t.tensor([0.0, 0.0, 1e-8])\n"
        "    out = ex3_classify_grad_health(model)\n"
        "    assert out['p_ok'] == 'ok', f'tiny nonzero element is still ok; got {out}'\n"
        "\n"
        "    # === Empty model (no params) returns empty dict ===\n"
        "    class Empty(nn.Module):\n"
        "        pass\n"
        "    assert ex3_classify_grad_health(Empty()) == {}, 'empty model should yield empty dict'\n"
        "\n"
        "    # === Nested model: dotted names from named_parameters preserved ===\n"
        "    class Nested(nn.Module):\n"
        "        def __init__(self):\n"
        "            super().__init__()\n"
        "            self.fc = nn.Linear(2, 2)\n"
        "    nested = Nested()\n"
        "    nested.fc.weight.grad = t.tensor([[1.0, 2.0], [3.0, 4.0]])\n"
        "    nested.fc.bias.grad = t.zeros(2)\n"
        "    out = ex3_classify_grad_health(nested)\n"
        "    assert out == {'fc.weight': 'ok', 'fc.bias': 'zero'}, out\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_classify_grad_health(model):\n"
        "    out = {}\n"
        "    for name, p in model.named_parameters():\n"
        "        if p.grad is None:\n"
        "            continue\n"
        "        # ORDER MATTERS: nan/inf first — an all-NaN tensor would otherwise\n"
        "        # fail the zero-check ambiguously (nan != 0 but also nan != nonzero).\n"
        "        if t.isnan(p.grad).any().item() or t.isinf(p.grad).any().item():\n"
        "            out[name] = 'nan'\n"
        "        elif p.grad.abs().max().item() == 0.0:\n"
        "            out[name] = 'zero'\n"
        "        else:\n"
        "            out[name] = 'ok'\n"
        "    return out"
    ),
    "solution_notes": (
        "**`.any().item()` is the safe coerce.** `t.isnan(x).any()` returns "
        "a 0-D tensor that's truthy in Python — but mixing tensor-bools "
        "with `or`/`and` triggers DeprecationWarnings on some torch "
        "versions. Coercing to Python bool with `.item()` is explicit.\n\n"
        "**Why combine nan + inf into one bucket.** Both indicate "
        "numerical pathology in the same way for an optimizer — the step "
        "should be skipped, the learning rate possibly cut, the run "
        "possibly aborted. Two separate buckets would just add a needless "
        "branch in caller code.\n\n"
        "**`abs().max() == 0` over `(p.grad == 0).all()`.** Both work for "
        "finite tensors, but `abs().max()` is one reduction kernel vs an "
        "equality-broadcast + reduce. Marginal but consistent with the "
        "PyTorch internals style."
    ),
    "extra_imports": ["import torch.nn as nn"],
}


# ---------------------------------------------------------------------------
# SPEC 3 — back-fn-call-with-recipe-args  ex3
# ---------------------------------------------------------------------------

SPEC_BACK_FN_CALL = {
    "atom_id": "back-fn-call-with-recipe-args",
    "subtopic": "Backprop: back fn call with recipe args",
    "topic_folder": TOPIC_BPD,
    "atom_recap_md": RECAP_PRECALL_SHAPE,
    "exercise_index": 3,
    "exercise_title": "validate grad_out.shape == node.array.shape before invoking back_fn",
    "slug": "validate-grad-out-shape-before-back-fn-call",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["validation", "shape-check", "guard", "back-fn-call"],
    "kcs": [
        "back-fn-call-with-recipe-args",
        "precall-shape-guard",
    ],
    "lo": (
        "Apply a pre-call shape guard that raises a `ValueError` naming "
        "both `grad_out.shape` and `node.array.shape` before invoking the "
        "back_fn — pinpointing the dispatcher rather than letting an "
        "elementwise broadcast error surface from deep inside the back_fn."
    ),
    "prompt_body": (
        "Implement `ex3_call_back_fn_validated(back_fn, grad_out, node)`. "
        "Same canonical four-arg invocation as ex1, plus a pre-call shape "
        "guard:\n\n"
        "1. If `grad_out.shape != node.array.shape`, raise "
        "`ValueError('grad_out.shape <A> != node.array.shape <B> — upstream "
        "dispatcher bug')` where `<A>` is `tuple(grad_out.shape)` and "
        "`<B>` is `tuple(node.array.shape)`.\n"
        "2. Otherwise call `back_fn(grad_out, node.array, "
        "*node.recipe.args, **node.recipe.kwargs)` and return its result.\n\n"
        "Shapes are the ONLY check — DON'T also validate `recipe.args` or "
        "`recipe.kwargs` (out of scope).\n\n"
        "Why `tuple(shape)` and not raw `torch.Size`. `torch.Size([3, 5])` "
        "reprs as `torch.Size([3, 5])` — verbose. The test asserts both "
        "tuples appear in the message text.\n\n"
        "Node has the standard MiniTensor shape:\n"
        "- `node.array` — a `torch.Tensor` (the cached forward output).\n"
        "- `node.recipe.args` — a tuple of raw tensors.\n"
        "- `node.recipe.kwargs` — a dict.\n\n"
        "The test provides a `Node` namespace class for you."
    ),
    "stub": (
        "def ex3_call_back_fn_validated(back_fn, grad_out, node):\n"
        '    """Pre-call shape guard, then canonical back_fn invocation."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    from types import SimpleNamespace\n"
        "\n"
        "    def make_node(array, args=(), kwargs=None):\n"
        "        return SimpleNamespace(\n"
        "            array=array,\n"
        "            recipe=SimpleNamespace(args=args, kwargs=kwargs or {}),\n"
        "        )\n"
        "\n"
        "    # === Happy path: shapes match, back_fn runs ===\n"
        "    out = t.tensor([1.0, 2.0, 3.0])\n"
        "    grad_out = t.tensor([0.5, 0.5, 0.5])\n"
        "    x = t.tensor([0.0, 1.0, 2.0])\n"
        "    node = make_node(array=out, args=(x,))\n"
        "    def my_back(g_out, out_, x_):\n"
        "        return g_out * x_\n"
        "    result = ex3_call_back_fn_validated(my_back, grad_out, node)\n"
        "    assert t.allclose(result, t.tensor([0.0, 0.5, 1.0])), result\n"
        "\n"
        "    # === Mismatch raises with both shapes in the message ===\n"
        "    out = t.tensor([1.0, 2.0, 3.0])\n"
        "    grad_out = t.tensor([0.5, 0.5])  # wrong shape\n"
        "    node = make_node(array=out)\n"
        "    try:\n"
        "        ex3_call_back_fn_validated(my_back, grad_out, node)\n"
        "        assert False, 'should have raised ValueError'\n"
        "    except ValueError as e:\n"
        "        msg = str(e)\n"
        "        assert '(2,)' in msg or '(2)' in msg, f'grad_out shape (2,) missing from msg: {msg!r}'\n"
        "        assert '(3,)' in msg or '(3)' in msg, f'node.array shape (3,) missing from msg: {msg!r}'\n"
        "        assert 'dispatcher' in msg.lower(), f'should hint at dispatcher bug; got {msg!r}'\n"
        "\n"
        "    # === Multi-dim mismatch ===\n"
        "    out = t.zeros(3, 4)\n"
        "    grad_out = t.zeros(3, 5)\n"
        "    node = make_node(array=out)\n"
        "    try:\n"
        "        ex3_call_back_fn_validated(my_back, grad_out, node)\n"
        "        assert False\n"
        "    except ValueError as e:\n"
        "        msg = str(e)\n"
        "        assert '(3, 5)' in msg, f'grad_out (3,5) should appear in msg: {msg!r}'\n"
        "        assert '(3, 4)' in msg, f'node.array (3,4) should appear in msg: {msg!r}'\n"
        "\n"
        "    # === kwargs pass-through verified on happy path ===\n"
        "    out = t.tensor([1.0, 2.0, 3.0])\n"
        "    grad_out = t.tensor([1.0, 1.0, 1.0])\n"
        "    node = make_node(array=out, args=(t.zeros(3),), kwargs={'dim': 0, 'keepdim': False})\n"
        "    received = {}\n"
        "    def back_recording(g_out, out_, x_, dim=None, keepdim=None):\n"
        "        received['dim'] = dim\n"
        "        received['keepdim'] = keepdim\n"
        "        return g_out\n"
        "    ex3_call_back_fn_validated(back_recording, grad_out, node)\n"
        "    assert received == {'dim': 0, 'keepdim': False}, received\n"
        "\n"
        "    # === 0-D tensor — same shape is fine ===\n"
        "    out = t.tensor(2.0)\n"
        "    grad_out = t.tensor(1.0)\n"
        "    node = make_node(array=out)\n"
        "    def back_passthrough(g_out, out_):\n"
        "        return g_out\n"
        "    r = ex3_call_back_fn_validated(back_passthrough, grad_out, node)\n"
        "    assert r.item() == 1.0\n"
        "\n"
        "    # === 0-D vs 1-D is a mismatch ===\n"
        "    out = t.tensor(2.0)               # shape ()\n"
        "    grad_out = t.tensor([1.0])        # shape (1,)\n"
        "    node = make_node(array=out)\n"
        "    try:\n"
        "        ex3_call_back_fn_validated(back_passthrough, grad_out, node)\n"
        "        assert False\n"
        "    except ValueError:\n"
        "        pass\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_call_back_fn_validated(back_fn, grad_out, node):\n"
        "    if grad_out.shape != node.array.shape:\n"
        "        raise ValueError(\n"
        "            f'grad_out.shape {tuple(grad_out.shape)} != node.array.shape '\n"
        "            f'{tuple(node.array.shape)} \\u2014 upstream dispatcher bug'\n"
        "        )\n"
        "    return back_fn(\n"
        "        grad_out,\n"
        "        node.array,\n"
        "        *node.recipe.args,\n"
        "        **node.recipe.kwargs,\n"
        "    )"
    ),
    "solution_notes": (
        "**`tuple(shape)` is the readable form.** `torch.Size([3, 5])` "
        "reprs as `torch.Size([3, 5])` — verbose for an error message. "
        "Tupling first yields `(3, 5)` which matches Python literal "
        "convention.\n\n"
        "**Why this guard is dispatcher-level, not back_fn-level.** The "
        "back_fn itself is op-specific (sigmoid_back, sum_back, ...). "
        "Shape compatibility between `grad_out` and `out` is INVARIANT "
        "across all back_fns — chain rule requires it. Putting the check "
        "at the dispatcher avoids duplicating it 30 times in every back_fn.\n\n"
        "**Don't validate `args` or `kwargs` shapes here.** Each back_fn "
        "knows what shapes its specific args should be (`multiply_back0` "
        "needs `y` of broadcastable shape, etc.). Pushing those checks up "
        "to the dispatcher would require a per-fn validation table — "
        "more code than just letting the back_fn raise its own error."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — backprop-pop-outgrad-loop  ex3
# ---------------------------------------------------------------------------

SPEC_BACKPROP_LOOP = {
    "atom_id": "backprop-pop-outgrad-loop",
    "subtopic": "Backprop: backprop pop-outgrad loop",
    "topic_folder": TOPIC_BPD,
    "atom_recap_md": RECAP_REVERSE_MAXABS_TRACE,
    "exercise_index": 3,
    "exercise_title": "instrument reverse-pass with per-node max|grad_out| trace",
    "slug": "instrument-reverse-pass-with-maxabs-trace",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["trace", "max-abs", "vanishing-gradient", "exploding-gradient", "diagnostic"],
    "kcs": [
        "backprop-pop-outgrad-loop",
        "linf-grad-trace-per-node",
    ],
    "lo": (
        "Analyze the reverse pass by recording the L∞ magnitude "
        "(`grad_out.abs().max().item()`) for every node as it's popped, "
        "producing a trace list that fingerprints vanishing/exploding "
        "gradient pathologies."
    ),
    "prompt_body": (
        "Implement `ex3_backprop_traced(end_node, end_grad, sorted_graph, "
        "back_funcs)` — the same reverse-pass driver as ex1, but also "
        "return a `trace` list capturing the L∞ magnitude of each "
        "node's popped `grad_out`.\n\n"
        "Signature:\n\n"
        "```python\n"
        "def ex3_backprop_traced(end_node, end_grad, sorted_graph, back_funcs):\n"
        "    ...\n"
        "    return trace  # list[tuple[int, float]]\n"
        "```\n\n"
        "Semantics:\n"
        "- `trace` is a list of `(id(node), float(grad_out.abs().max()))` "
        "entries, appended IMMEDIATELY AFTER popping `grad_out` from the "
        "accumulator and BEFORE dispatching to back_fns.\n"
        "- Only nodes that ACTUALLY get popped (i.e. have a grad routed to "
        "them) appear in the trace. Skipped nodes (no entry in `grads`) "
        "are absent.\n"
        "- Order: matches the reverse-pass walk through `sorted_graph` — "
        "`end_node` typically first, leaves last.\n\n"
        "Same three reverse-pass invariants as ex1:\n"
        "1. POP (don't peek) the entry in `grads`.\n"
        "2. ACCUMULATE (don't overwrite) — `+=` per parent.\n"
        "3. LEAVES write `.grad`; non-leaves stay in the `grads` dict.\n\n"
        "Mutate `.grad` on leaves in place; return only `trace`."
    ),
    "stub": (
        "def ex3_backprop_traced(end_node, end_grad, sorted_graph, back_funcs) -> list:\n"
        '    """Reverse-pass driver + per-node max|grad_out| trace."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    from types import SimpleNamespace\n"
        "\n"
        "    # === MiniTensor scaffold ===\n"
        "    def leaf(value):\n"
        "        return SimpleNamespace(array=t.tensor(value), recipe=None, grad=None)\n"
        "    def interior(value, func, parents, args=(), kwargs=None):\n"
        "        return SimpleNamespace(\n"
        "            array=t.tensor(value),\n"
        "            recipe=SimpleNamespace(func=func, parents=parents, args=args, kwargs=kwargs or {}),\n"
        "            grad=None,\n"
        "        )\n"
        "\n"
        "    # === Chain: x -> y = x*2 -> z = y*3.  d(z)/d(x) = 6 ===\n"
        "    x = leaf([1.0, 2.0, 3.0])\n"
        "    y = interior([2.0, 4.0, 6.0], func='mul', parents={0: x}, args=(x.array, 2.0))\n"
        "    z = interior([6.0, 12.0, 18.0], func='mul', parents={0: y}, args=(y.array, 3.0))\n"
        "\n"
        "    def mul_back(grad_out, out_, x_, c):\n"
        "        return grad_out * c\n"
        "\n"
        "    back_funcs = {('mul', 0): mul_back}\n"
        "    sorted_graph = [z, y, x]   # reverse-topological\n"
        "    end_grad = t.ones(3)\n"
        "\n"
        "    trace = ex3_backprop_traced(z, end_grad, sorted_graph, back_funcs)\n"
        "\n"
        "    # === Trace shape: list of (id, float) ===\n"
        "    assert isinstance(trace, list), f'trace must be list; got {type(trace).__name__}'\n"
        "    assert all(isinstance(e, tuple) and len(e) == 2 for e in trace), trace\n"
        "    assert all(isinstance(e[0], int) and isinstance(e[1], float) for e in trace), trace\n"
        "\n"
        "    # === Three entries, one per popped node ===\n"
        "    assert len(trace) == 3, f'expected 3 trace entries, got {len(trace)}: {trace}'\n"
        "\n"
        "    # === Order matches reverse walk ===\n"
        "    ids = [e[0] for e in trace]\n"
        "    assert ids == [id(z), id(y), id(x)], f'trace order wrong: {ids}'\n"
        "\n"
        "    # === Magnitudes: z gets end_grad (max=1), y gets 3*end_grad (max=3), x gets 6 (max=6) ===\n"
        "    mags = [e[1] for e in trace]\n"
        "    assert abs(mags[0] - 1.0) < 1e-6, f'z mag should be 1.0; got {mags[0]}'\n"
        "    assert abs(mags[1] - 3.0) < 1e-6, f'y mag should be 3.0; got {mags[1]}'\n"
        "    assert abs(mags[2] - 6.0) < 1e-6, f'x mag should be 6.0; got {mags[2]}'\n"
        "\n"
        "    # === Leaf .grad written correctly ===\n"
        "    assert x.grad is not None, 'leaf .grad must be populated'\n"
        "    assert t.allclose(x.grad, t.tensor([6.0, 6.0, 6.0])), x.grad\n"
        "\n"
        "    # === Diamond DAG: out = x*x, accumulation via both parents ===\n"
        "    x = leaf([2.0])\n"
        "    sq = interior([4.0], func='mul', parents={0: x, 1: x}, args=(x.array, x.array))\n"
        "    def mul_back_0(grad_out, out_, a, b):\n"
        "        return grad_out * b\n"
        "    def mul_back_1(grad_out, out_, a, b):\n"
        "        return grad_out * a\n"
        "    back_funcs = {('mul', 0): mul_back_0, ('mul', 1): mul_back_1}\n"
        "    trace = ex3_backprop_traced(sq, t.ones(1), [sq, x], back_funcs)\n"
        "    assert len(trace) == 2, f'expected 2 trace entries; got {len(trace)}'\n"
        "    assert trace[0][0] == id(sq) and trace[1][0] == id(x)\n"
        "    # x's accumulated grad = 2 + 2 = 4 → its popped max is 4.0\n"
        "    assert abs(trace[1][1] - 4.0) < 1e-6, f'x mag should be 4.0; got {trace[1][1]}'\n"
        "    assert t.allclose(x.grad, t.tensor([4.0])), x.grad\n"
        "\n"
        "    # === Decreasing trace = vanishing signature ===\n"
        "    # Chain of x4 — start with grad 1.0, accumulate scaling of 0.5 each step\n"
        "    a = leaf([1.0])\n"
        "    b = interior([0.5], func='scale', parents={0: a}, args=(a.array,))\n"
        "    c = interior([0.25], func='scale', parents={0: b}, args=(b.array,))\n"
        "    d = interior([0.125], func='scale', parents={0: c}, args=(c.array,))\n"
        "    def scale_back(grad_out, out_, x_):\n"
        "        return grad_out * 0.5\n"
        "    back_funcs = {('scale', 0): scale_back}\n"
        "    trace = ex3_backprop_traced(d, t.ones(1), [d, c, b, a], back_funcs)\n"
        "    # mags should be 1.0, 0.5, 0.25, 0.125 — monotone-decreasing = vanishing.\n"
        "    expected = [1.0, 0.5, 0.25, 0.125]\n"
        "    for i, (got_id, got_mag) in enumerate(trace):\n"
        "        assert abs(got_mag - expected[i]) < 1e-6, f'mag {i}: expected {expected[i]}, got {got_mag}'\n"
        "    # And it's strictly monotone — the vanishing fingerprint.\n"
        "    mags = [e[1] for e in trace]\n"
        "    assert all(mags[i] > mags[i+1] for i in range(len(mags)-1)), f'vanishing: not monotone: {mags}'\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_backprop_traced(end_node, end_grad, sorted_graph, back_funcs):\n"
        "    grads = {id(end_node): end_grad}\n"
        "    trace = []\n"
        "    for node in sorted_graph:\n"
        "        if id(node) not in grads:\n"
        "            continue\n"
        "        grad_out = grads.pop(id(node))\n"
        "        # Record L\\u221E magnitude AFTER pop, BEFORE dispatch — this is what\n"
        "        # was routed to this node, including any diamond-DAG accumulation.\n"
        "        trace.append((id(node), float(grad_out.abs().max().item())))\n"
        "        if node.recipe is None:\n"
        "            node.grad = grad_out if node.grad is None else node.grad + grad_out\n"
        "            continue\n"
        "        for argnum, parent in node.recipe.parents.items():\n"
        "            bf = back_funcs[(node.recipe.func, argnum)]\n"
        "            gp = bf(grad_out, node.array, *node.recipe.args, **node.recipe.kwargs)\n"
        "            pid = id(parent)\n"
        "            grads[pid] = grads.get(pid, 0) + gp\n"
        "    return trace"
    ),
    "solution_notes": (
        "**Record AFTER pop, BEFORE dispatch.** Recording before pop would "
        "miss the accumulation step (diamond DAG: parent gets contributions "
        "from multiple children; the magnitude isn't final until the pop). "
        "Recording after dispatch would conflate the parent's incoming "
        "grad with the post-back_fn grad, which is a DIFFERENT quantity.\n\n"
        "**L∞ catches the worst element.** A grad of shape `(1024,)` with "
        "1023 zeros and one `1e10` is exploding in real terms; L2 averages "
        "it to `1e10/sqrt(1024) ≈ 3.1e8` which understates the danger. "
        "Max-abs is the right alarm signal.\n\n"
        "**`.item()` for the float cast.** The trace must be JSON-serializable "
        "for downstream logging. Python floats (not 0-D tensors) achieve that."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — cycle-detection-temp-set  ex3
# ---------------------------------------------------------------------------

SPEC_CYCLE_DETECT = {
    "atom_id": "cycle-detection-temp-set",
    "subtopic": "Backprop: cycle detection via temp set",
    "topic_folder": TOPIC_BPD,
    "atom_recap_md": RECAP_ENUMERATE_ALL_CYCLES,
    "exercise_index": 3,
    "exercise_title": "enumerate ALL cycles reachable from root, not just the first",
    "slug": "enumerate-all-cycles-not-just-first",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["cycle-enumeration", "all-cycles", "back-edge", "dfs", "diagnostic"],
    "kcs": [
        "cycle-detection-temp-set",
        "continue-after-back-edge",
    ],
    "lo": (
        "Apply the temp-set DFS cycle-detection pattern to enumerate "
        "EVERY cycle reachable from `root` (recording each back-edge "
        "path and CONTINUING the walk), rather than returning on the "
        "first one found."
    ),
    "prompt_body": (
        "Implement `ex3_all_cycles(root, get_children)`. Same input "
        "contract as ex1/ex2: a `root` node and a `get_children` callable "
        "returning the node's direct successors.\n\n"
        "Return: `list[list[node]]`. Each inner list is one cycle, with "
        "the back-edge target REPEATED at the end (so `[a, b, c, a]` "
        "closes the cycle). Empty list `[]` if no cycles exist.\n\n"
        "**Critical difference from ex2.** Ex2 returned the first cycle "
        "and aborted the walk. Ex3 must RECORD the cycle and CONTINUE — "
        "explore the rest of the graph so disjoint cycles are also found.\n\n"
        "Algorithm:\n\n"
        "```python\n"
        "def ex3_all_cycles(root, get_children):\n"
        "    cycles = []\n"
        "    on_stack = set()\n"
        "    perm = set()\n"
        "    path = []\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm: return\n"
        "        if nid in on_stack:\n"
        "            i = next(j for j, n in enumerate(path) if id(n) == nid)\n"
        "            cycles.append(path[i:] + [node])\n"
        "            return  # do NOT halt the walk\n"
        "        on_stack.add(nid); path.append(node)\n"
        "        for child in get_children(node):\n"
        "            visit(child)\n"
        "        on_stack.discard(nid); path.pop()\n"
        "        perm.add(nid)\n"
        "    visit(root)\n"
        "    return cycles\n"
        "```\n\n"
        "Use `id(node)` for set membership. The test nodes are dataclass-like "
        "and don't override `__hash__`/`__eq__` in a way that conflicts.\n\n"
        "Do NOT canonicalize / dedup paths — return them in the order "
        "discovered."
    ),
    "stub": (
        "def ex3_all_cycles(root, get_children) -> list:\n"
        '    """Return list of all cycle paths reachable from root; [] if none."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    # Node with `children: list` and `name: str` for readable diffs.\n"
        "    class N:\n"
        "        def __init__(self, name):\n"
        "            self.name = name\n"
        "            self.children = []\n"
        "        def __repr__(self):\n"
        "            return f'N({self.name})'\n"
        "\n"
        "    def kids(n):\n"
        "        return n.children\n"
        "\n"
        "    # === Pure DAG — no cycles ===\n"
        "    a, b, c, d = N('a'), N('b'), N('c'), N('d')\n"
        "    a.children = [b, c]\n"
        "    b.children = [d]\n"
        "    c.children = [d]\n"
        "    assert ex3_all_cycles(a, kids) == [], 'diamond DAG has no cycles'\n"
        "\n"
        "    # === Single cycle ===\n"
        "    a, b, c = N('a'), N('b'), N('c')\n"
        "    a.children = [b]\n"
        "    b.children = [c]\n"
        "    c.children = [a]\n"
        "    cycles = ex3_all_cycles(a, kids)\n"
        "    assert len(cycles) == 1, f'expected 1 cycle; got {len(cycles)}: {cycles}'\n"
        "    names = [n.name for n in cycles[0]]\n"
        "    assert names == ['a', 'b', 'c', 'a'], f'cycle path wrong: {names}'\n"
        "\n"
        "    # === Self-loop ===\n"
        "    a = N('a')\n"
        "    a.children = [a]\n"
        "    cycles = ex3_all_cycles(a, kids)\n"
        "    assert len(cycles) == 1\n"
        "    assert [n.name for n in cycles[0]] == ['a', 'a']\n"
        "\n"
        "    # === Two disjoint cycles (the headline test) ===\n"
        "    # Root r has two children r->a and r->c. a forms cycle a->b->a, c forms cycle c->d->e->c.\n"
        "    r, a, b, c, d, e = N('r'), N('a'), N('b'), N('c'), N('d'), N('e')\n"
        "    r.children = [a, c]\n"
        "    a.children = [b]\n"
        "    b.children = [a]      # cycle 1: a-b-a\n"
        "    c.children = [d]\n"
        "    d.children = [e]\n"
        "    e.children = [c]      # cycle 2: c-d-e-c\n"
        "    cycles = ex3_all_cycles(r, kids)\n"
        "    assert len(cycles) == 2, f'expected 2 disjoint cycles; got {len(cycles)}: {cycles}'\n"
        "    sigs = sorted([''.join(n.name for n in cyc) for cyc in cycles])\n"
        "    assert sigs == ['aba', 'cdec'], f'cycle signatures wrong: {sigs}'\n"
        "\n"
        "    # === Nested: cycle inside a cycle (two back-edges) ===\n"
        "    # a -> b -> c -> b (inner cycle), c -> a (outer cycle)\n"
        "    a, b, c = N('a'), N('b'), N('c')\n"
        "    a.children = [b]\n"
        "    b.children = [c]\n"
        "    c.children = [b, a]   # two back-edges from c\n"
        "    cycles = ex3_all_cycles(a, kids)\n"
        "    assert len(cycles) == 2, f'expected 2 cycles; got {len(cycles)}: {cycles}'\n"
        "    sigs = sorted([''.join(n.name for n in cyc) for cyc in cycles])\n"
        "    # c.children = [b, a]: visit() at c first recurses into b (back-edge → 'bcb'),\n"
        "    # then recurses into a (back-edge → 'abca'). Both cycles found.\n"
        "    assert sigs == ['abca', 'bcb'], f'expected cycles abca + bcb; got {sigs}'\n"
        "\n"
        "    # === The CONTINUE-not-RETURN check: after finding cycle 1, the walk must explore further. ===\n"
        "    # If implementer returns on first back-edge, only 1 cycle will be found here.\n"
        "    r, x, y, z = N('r'), N('x'), N('y'), N('z')\n"
        "    r.children = [x, z]\n"
        "    x.children = [y]\n"
        "    y.children = [x]    # cycle x-y-x (found first)\n"
        "    z.children = [z]    # cycle z-z (found ONLY if walk continues)\n"
        "    cycles = ex3_all_cycles(r, kids)\n"
        "    assert len(cycles) == 2, (\n"
        "        f'must continue walking after finding first cycle; got {len(cycles)}'\n"
        "    )\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_all_cycles(root, get_children):\n"
        "    cycles = []\n"
        "    on_stack = set()\n"
        "    perm = set()\n"
        "    path = []\n"
        "\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm:\n"
        "            return\n"
        "        if nid in on_stack:\n"
        "            i = next(j for j, n in enumerate(path) if id(n) == nid)\n"
        "            cycles.append(path[i:] + [node])\n"
        "            return  # CONTINUE the outer walk — do not halt\n"
        "        on_stack.add(nid)\n"
        "        path.append(node)\n"
        "        for child in get_children(node):\n"
        "            visit(child)\n"
        "        on_stack.discard(nid)\n"
        "        path.pop()\n"
        "        perm.add(nid)\n"
        "\n"
        "    visit(root)\n"
        "    return cycles"
    ),
    "solution_notes": (
        "**The one-line bug ex2-vs-ex3.** Ex2's `find_cycle` returned the "
        "found path immediately. Ex3 must APPEND and KEEP WALKING — the "
        "back-edge case records the cycle but lets siblings get explored. "
        "Forgetting this is the universal mistake when going from "
        "find-one to find-all.\n\n"
        "**`on_stack.discard` + `path.pop` MUST both run on the way up.** "
        "Sibling-subtree contamination otherwise: leftover `on_stack` "
        "entries cause false positives on subtrees that share ancestors.\n\n"
        "**Don't dedup — leave that to the caller.** Two paths that "
        "traverse the same cycle starting from different nodes are "
        "technically distinct under this algorithm. Canonicalization "
        "(rotate to smallest-id-first, etc.) is a separate concern from "
        "discovery."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — dfs-three-set-toposort  ex3
# ---------------------------------------------------------------------------

SPEC_DFS_TOPOSORT = {
    "atom_id": "dfs-three-set-toposort",
    "subtopic": "Backprop: DFS three-set toposort",
    "topic_folder": TOPIC_BPD,
    "atom_recap_md": RECAP_DETERMINISTIC_TOPOSORT,
    "exercise_index": 3,
    "exercise_title": "deterministic toposort — sort children by key for reproducible output",
    "slug": "deterministic-toposort-with-child-sort-key",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["toposort", "deterministic", "tiebreaker", "sort-key", "reproducibility"],
    "kcs": [
        "dfs-three-set-toposort",
        "child-key-tiebreaker",
    ],
    "lo": (
        "Apply the three-color recursive toposort with a caller-supplied "
        "`key` callable that orders children at every level — producing "
        "an output that is deterministic under arbitrary `get_children` "
        "iteration order while preserving the deps-first invariant."
    ),
    "prompt_body": (
        "Implement `ex3_topological_sort_keyed(root, get_children, key)`. "
        "Same contract as ex1's recursive `topological_sort`, but with a "
        "tiebreaker:\n\n"
        "- `root`: the start node.\n"
        "- `get_children(node) -> list` returning the node's direct successors.\n"
        "- `key`: a callable `node -> comparable`. Children are sorted "
        "by this key BEFORE recursing.\n\n"
        "Output: `list[node]` in deps-FIRST order (root LAST). Same one-"
        "appearance-per-node and cycle invariants as ex1:\n"
        "- `ValueError('cycle')` on a back-edge.\n"
        "- Each reachable node appears EXACTLY once (diamond DAGs OK).\n"
        "- `root` is the LAST element of the returned list.\n\n"
        "**Determinism is the headline.** Calling the function twice on "
        "the same graph with the same `key` MUST return identical output, "
        "regardless of `get_children` iteration order.\n\n"
        "Use `id(node)` for set keys (temp + perm sets, as in ex1/ex2). "
        "Children sort uses the caller-supplied `key`, NOT `id`.\n\n"
        "Algorithm sketch:\n\n"
        "```python\n"
        "def ex3_topological_sort_keyed(root, get_children, key):\n"
        "    perm, temp = set(), set()\n"
        "    result = []\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm: return\n"
        "        if nid in temp: raise ValueError('cycle')\n"
        "        temp.add(nid)\n"
        "        for child in sorted(get_children(node), key=key):\n"
        "            visit(child)\n"
        "        temp.remove(nid); perm.add(nid)\n"
        "        result.append(node)\n"
        "    visit(root)\n"
        "    return result\n"
        "```"
    ),
    "stub": (
        "def ex3_topological_sort_keyed(root, get_children, key) -> list:\n"
        '    """Deterministic three-color toposort; children sorted by `key` at each level."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    class N:\n"
        "        def __init__(self, name):\n"
        "            self.name = name\n"
        "            self.children = []\n"
        "        def __repr__(self):\n"
        "            return f'N({self.name})'\n"
        "    def kids(n):\n"
        "        return n.children\n"
        "    def by_name(n):\n"
        "        return n.name\n"
        "\n"
        "    # === Linear chain: a -> b -> c.  deps-first → [c, b, a] ===\n"
        "    a, b, c = N('a'), N('b'), N('c')\n"
        "    a.children = [b]\n"
        "    b.children = [c]\n"
        "    out = ex3_topological_sort_keyed(a, kids, by_name)\n"
        "    assert [n.name for n in out] == ['c', 'b', 'a'], [n.name for n in out]\n"
        "    assert out[-1] is a, 'root must be LAST'\n"
        "\n"
        "    # === Determinism: child order in get_children should NOT affect output. ===\n"
        "    a, b, c, d = N('a'), N('b'), N('c'), N('d')\n"
        "    a.children = [b, c]\n"
        "    b.children = [d]\n"
        "    c.children = [d]\n"
        "    out1 = ex3_topological_sort_keyed(a, kids, by_name)\n"
        "    # Reverse child order in a — should still yield the same sorted output.\n"
        "    a.children = [c, b]\n"
        "    out2 = ex3_topological_sort_keyed(a, kids, by_name)\n"
        "    assert [n.name for n in out1] == [n.name for n in out2], (\n"
        "        f'output not deterministic across child orderings: {[n.name for n in out1]} vs {[n.name for n in out2]}'\n"
        "    )\n"
        "\n"
        "    # === Diamond DAG: each node appears EXACTLY once ===\n"
        "    assert len(out1) == 4 and len({id(n) for n in out1}) == 4\n"
        "    # And root is last:\n"
        "    assert out1[-1] is a\n"
        "    # And d appears before b and c (deps-first):\n"
        "    names = [n.name for n in out1]\n"
        "    assert names.index('d') < names.index('b'), names\n"
        "    assert names.index('d') < names.index('c'), names\n"
        "    # Sibling ordering: among b vs c, alphabetical (key=by_name) → b before c.\n"
        "    assert names.index('b') < names.index('c'), f'sibling order should be alphabetical: {names}'\n"
        "\n"
        "    # === Cycle raises ValueError ===\n"
        "    a, b = N('a'), N('b')\n"
        "    a.children = [b]\n"
        "    b.children = [a]\n"
        "    try:\n"
        "        ex3_topological_sort_keyed(a, kids, by_name)\n"
        "        assert False, 'cycle should raise'\n"
        "    except ValueError:\n"
        "        pass\n"
        "\n"
        "    # === Numeric key — non-string sort works ===\n"
        "    n1, n2, n3 = N('a'), N('b'), N('c')\n"
        "    n1.children = [n2, n3]\n"
        "    n2.priority = 5\n"
        "    n3.priority = 1\n"
        "    n1.priority = 0\n"
        "    by_priority = lambda nd: nd.priority\n"
        "    out = ex3_topological_sort_keyed(n1, kids, by_priority)\n"
        "    # Children sorted by priority: n3 (1) then n2 (5).  So order [n3, n2, n1].\n"
        "    assert [n.name for n in out] == ['c', 'b', 'a'], [n.name for n in out]\n"
        "\n"
        "    # === Caller can override default ordering. ===\n"
        "    # Same graph as the diamond, but key=lambda n: -ord(n.name[0]) reverses sibling order.\n"
        "    a, b, c, d = N('a'), N('b'), N('c'), N('d')\n"
        "    a.children = [b, c]\n"
        "    b.children = [d]\n"
        "    c.children = [d]\n"
        "    reverse_alpha = lambda n: -ord(n.name[0])\n"
        "    out = ex3_topological_sort_keyed(a, kids, reverse_alpha)\n"
        "    names = [n.name for n in out]\n"
        "    # Now sibling order is c before b (reversed):\n"
        "    assert names.index('c') < names.index('b'), names\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_topological_sort_keyed(root, get_children, key):\n"
        "    perm, temp = set(), set()\n"
        "    result = []\n"
        "\n"
        "    def visit(node):\n"
        "        nid = id(node)\n"
        "        if nid in perm:\n"
        "            return\n"
        "        if nid in temp:\n"
        "            raise ValueError('cycle')\n"
        "        temp.add(nid)\n"
        "        # Sort children BEFORE recursing — deterministic sibling order.\n"
        "        for child in sorted(get_children(node), key=key):\n"
        "            visit(child)\n"
        "        temp.remove(nid)\n"
        "        perm.add(nid)\n"
        "        result.append(node)\n"
        "\n"
        "    visit(root)\n"
        "    return result"
    ),
    "solution_notes": (
        "**Sort children, not the result.** Sorting the result list would "
        "destroy the toposort invariant (children must come before parents). "
        "Sorting siblings AT EACH LEVEL only reorders nodes that are "
        "topologically incomparable — the deps-first guarantee survives.\n\n"
        "**`sorted(..., key=key)` is stable.** Python's sort is stable, so "
        "ties in `key(node)` preserve the order from `get_children`. The "
        "test uses unique keys to avoid this corner.\n\n"
        "**`id(node)` for sets, `key(node)` for the order.** Identity vs "
        "ordering are different concerns. Sets dedup by identity (so "
        "diamond DAGs don't double-visit); the `key` callable only "
        "determines sibling traversal order."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — dispatch-back-fn-from-recipe  ex3
# ---------------------------------------------------------------------------

SPEC_DISPATCH = {
    "atom_id": "dispatch-back-fn-from-recipe",
    "subtopic": "Backprop: dispatch back fn from recipe",
    "topic_folder": TOPIC_BPD,
    "atom_recap_md": RECAP_WILDCARD_DISPATCH,
    "exercise_index": 3,
    "exercise_title": "wildcard (fn, None) fallback when (fn, argnum) is unregistered",
    "slug": "wildcard-fn-none-fallback-when-argnum-missing",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dispatch", "wildcard", "fallback", "back-funcs", "precedence"],
    "kcs": [
        "dispatch-back-fn-from-recipe",
        "exact-before-wildcard-precedence",
    ],
    "lo": (
        "Apply a two-step lookup that first probes `(recipe.func, argnum)` "
        "for an exact back_fn, then falls back to `(recipe.func, None)` as "
        "a wildcard — enforcing exact-match precedence over wildcard, "
        "raising `KeyError` only when neither key is registered."
    ),
    "prompt_body": (
        "Implement `ex3_dispatch_with_wildcard(node, back_funcs)` — same "
        "return shape as ex1's `dispatch_back_fns` (`list[(argnum, "
        "parent, back_fn)]`), but with a two-step lookup per parent:\n\n"
        "For each `(argnum, parent)` in `node.recipe.parents`:\n\n"
        "1. **Exact match.** If `(node.recipe.func, argnum) in back_funcs`, "
        "use that back_fn.\n"
        "2. **Wildcard fallback.** Otherwise, if `(node.recipe.func, None) "
        "in back_funcs`, use that back_fn.\n"
        "3. **Neither.** Raise `KeyError(f'No back_fn for ({fn_name}, "
        "{argnum})')` where `fn_name = getattr(node.recipe.func, "
        "'__name__', repr(node.recipe.func))`.\n\n"
        "**Precedence is mandatory.** If both `(fn, 0)` and `(fn, None)` "
        "exist, `argnum=0` MUST resolve to the EXACT match. The wildcard "
        "is a fallback, not an override.\n\n"
        "**Use case.** Symmetric ops like `add(x, y)` have the SAME back_fn "
        "for argnum=0 and argnum=1: `add_back(grad_out, out) = grad_out` "
        "regardless of which parent. Registering it once as `(add, None)` "
        "instead of duplicating for every argnum keeps the registry "
        "sparse.\n\n"
        "Assume `node.recipe is not None` (the caller filters leaves)."
    ),
    "stub": (
        "def ex3_dispatch_with_wildcard(node, back_funcs) -> list:\n"
        '    """Dispatch with (fn, argnum) exact match → (fn, None) wildcard fallback."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    from types import SimpleNamespace\n"
        "\n"
        "    def make_node(func, parents):\n"
        "        return SimpleNamespace(\n"
        "            recipe=SimpleNamespace(func=func, parents=parents)\n"
        "        )\n"
        "\n"
        "    def add_back_wildcard(grad_out, out_, *args, **kw):\n"
        "        return grad_out\n"
        "    def mul_back_0(grad_out, out_, x, y):\n"
        "        return grad_out * y\n"
        "    def mul_back_1(grad_out, out_, x, y):\n"
        "        return grad_out * x\n"
        "\n"
        "    # === Wildcard-only registration: (add, None) matches all argnums ===\n"
        "    back_funcs = {('add', None): add_back_wildcard}\n"
        "    x_parent = SimpleNamespace(name='x')\n"
        "    y_parent = SimpleNamespace(name='y')\n"
        "    node = make_node('add', {0: x_parent, 1: y_parent})\n"
        "    out = ex3_dispatch_with_wildcard(node, back_funcs)\n"
        "    assert len(out) == 2, f'expected 2 triples; got {len(out)}'\n"
        "    # Both should map to the wildcard fn.\n"
        "    assert all(bf is add_back_wildcard for (_, _, bf) in out), out\n"
        "    # argnums preserved:\n"
        "    argnums = [e[0] for e in out]\n"
        "    assert sorted(argnums) == [0, 1]\n"
        "\n"
        "    # === Exact-match registration: (mul, 0) and (mul, 1) both specific ===\n"
        "    back_funcs = {('mul', 0): mul_back_0, ('mul', 1): mul_back_1}\n"
        "    node = make_node('mul', {0: x_parent, 1: y_parent})\n"
        "    out = ex3_dispatch_with_wildcard(node, back_funcs)\n"
        "    triples = {an: bf for (an, _, bf) in out}\n"
        "    assert triples[0] is mul_back_0\n"
        "    assert triples[1] is mul_back_1\n"
        "\n"
        "    # === The headline test: exact match wins over wildcard ===\n"
        "    def override_back_0(grad_out, out_, *args, **kw):\n"
        "        return grad_out * 999.0\n"
        "    back_funcs = {\n"
        "        ('add', None): add_back_wildcard,    # wildcard\n"
        "        ('add', 0):    override_back_0,      # exact override for argnum=0\n"
        "    }\n"
        "    node = make_node('add', {0: x_parent, 1: y_parent})\n"
        "    out = ex3_dispatch_with_wildcard(node, back_funcs)\n"
        "    triples = {an: bf for (an, _, bf) in out}\n"
        "    assert triples[0] is override_back_0, f'exact match (add,0) must win over (add,None); got {triples[0]}'\n"
        "    assert triples[1] is add_back_wildcard, f'argnum=1 has no exact match, must fall back to wildcard; got {triples[1]}'\n"
        "\n"
        "    # === Neither exact nor wildcard → KeyError with op + argnum named ===\n"
        "    back_funcs = {('mul', 0): mul_back_0}   # nothing for argnum=1, no wildcard\n"
        "    node = make_node('mul', {0: x_parent, 1: y_parent})\n"
        "    try:\n"
        "        ex3_dispatch_with_wildcard(node, back_funcs)\n"
        "        assert False, 'should have raised KeyError'\n"
        "    except KeyError as e:\n"
        "        msg = str(e)\n"
        "        assert 'mul' in msg, f'KeyError must name fn; got {msg}'\n"
        "        assert '1' in msg, f'KeyError must name argnum; got {msg}'\n"
        "\n"
        "    # === Triple shape: (argnum, parent, back_fn) ===\n"
        "    back_funcs = {('add', None): add_back_wildcard}\n"
        "    node = make_node('add', {0: x_parent})\n"
        "    out = ex3_dispatch_with_wildcard(node, back_funcs)\n"
        "    assert len(out) == 1\n"
        "    argnum, parent, back_fn = out[0]\n"
        "    assert argnum == 0\n"
        "    assert parent is x_parent\n"
        "    assert back_fn is add_back_wildcard\n"
        "\n"
        "    # === Real callable for fn (with __name__) — fn_name extraction works ===\n"
        "    def my_op():\n"
        "        pass\n"
        "    back_funcs = {(my_op, 0): mul_back_0}\n"
        "    node = make_node(my_op, {1: y_parent})    # argnum=1 unregistered, no wildcard\n"
        "    try:\n"
        "        ex3_dispatch_with_wildcard(node, back_funcs)\n"
        "        assert False\n"
        "    except KeyError as e:\n"
        "        msg = str(e)\n"
        "        assert 'my_op' in msg, f'__name__ must surface; got {msg}'\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_dispatch_with_wildcard(node, back_funcs):\n"
        "    results = []\n"
        "    fn = node.recipe.func\n"
        "    for argnum, parent in node.recipe.parents.items():\n"
        "        if (fn, argnum) in back_funcs:\n"
        "            back_fn = back_funcs[(fn, argnum)]            # exact wins\n"
        "        elif (fn, None) in back_funcs:\n"
        "            back_fn = back_funcs[(fn, None)]              # wildcard fallback\n"
        "        else:\n"
        "            fn_name = getattr(fn, '__name__', repr(fn))\n"
        "            raise KeyError(\n"
        "                f'No back_fn for ({fn_name}, {argnum})'\n"
        "            )\n"
        "        results.append((argnum, parent, back_fn))\n"
        "    return results"
    ),
    "solution_notes": (
        "**`in back_funcs` is the right probe.** `back_funcs.get((fn, "
        "argnum))` and checking for `None` would conflate 'unregistered' "
        "with 'registered as None'. Membership check is unambiguous.\n\n"
        "**Order matters: exact before wildcard.** An `if/elif` chain "
        "enforces this naturally — once `(fn, argnum)` is found, the "
        "wildcard branch never executes. Reversing the order would make "
        "every wildcard registration shadow every exact one — a bug.\n\n"
        "**Why `getattr(fn, '__name__', repr(fn))` in the error.** Some "
        "registry keys are strings (e.g. `'add'` in our tests), not "
        "callables. `repr('add')` yields `\"'add'\"`. `getattr` returns "
        "the string directly when no `__name__` attribute exists (well, "
        "strings DO have `__name__`? — actually no, strings don't have "
        "`__name__`, so this falls through to `repr(fn)`). For real "
        "callables, `fn.__name__` is what users recognize."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — grad-expressed-in-out  ex3
# ---------------------------------------------------------------------------

SPEC_GRAD_IN_OUT = {
    "atom_id": "grad-expressed-in-out",
    "subtopic": "Backprop: grad expressed in out",
    "topic_folder": TOPIC_BPD,
    "atom_recap_md": RECAP_SOFTPLUS_BACK,
    "exercise_index": 3,
    "exercise_title": "softplus_back via cached out — third activation, no recompute from x",
    "slug": "softplus-back-via-cached-out-no-recompute",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["softplus", "cached-out", "elementwise", "no-recompute", "transfer"],
    "kcs": [
        "grad-expressed-in-out",
        "back-fn-uses-cached-out",
    ],
    "lo": (
        "Apply the 'grad expressed in `out`' pattern to softplus by "
        "writing `softplus_back(grad_out, out, x) = grad_out * (1 - "
        "t.exp(-out))`, reusing the cached forward output rather than "
        "recomputing `softplus(x)` or invoking `t.sigmoid(x)`."
    ),
    "prompt_body": (
        "Implement `ex3_softplus_back(grad_out, out, x)` using ONLY the "
        "cached `out` — no `t.sigmoid(x)`, no `t.softplus(x)`, no `t.exp(x)` "
        "on the raw input.\n\n"
        "**Math.** `out = softplus(x) = log(1 + exp(x))`. The derivative "
        "is `sigmoid(x)`, which equals `1 - exp(-out)`:\n\n"
        "```\n"
        "exp(-out) = exp(-log(1+exp(x))) = 1 / (1+exp(x))\n"
        "1 - exp(-out) = exp(x) / (1+exp(x)) = sigmoid(x)\n"
        "```\n\n"
        "So by the chain rule:\n\n"
        "```\n"
        "dL/dx = grad_out * (1 - t.exp(-out))\n"
        "```\n\n"
        "**The drill's constraint.** The function MUST be expressible in "
        "ONE line that uses `out` and not `x`. The test will pass a "
        "deliberately WRONG `x` (zeros, or unrelated values) to catch any "
        "attempt to recompute from `x`.\n\n"
        "Inputs:\n"
        "- `grad_out`: `Tensor`, dL/d(out), same shape as `out`.\n"
        "- `out`: `Tensor`, the cached `softplus(x_real)` from forward.\n"
        "- `x`: `Tensor`, the original input — passed for signature "
        "compatibility but UNUSED.\n\n"
        "Output: `Tensor` of the same shape, `dL/dx`."
    ),
    "stub": (
        "def ex3_softplus_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """dL/dx for softplus, via cached out: grad_out * (1 - exp(-out))."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _test_ex3():\n"
        "    import torch.nn.functional as F\n"
        "\n"
        "    # === Reference path: autograd-derived sigmoid(x) ===\n"
        "    x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)\n"
        "    out = F.softplus(x)\n"
        "    ref_grad = t.sigmoid(x).detach()    # d/dx softplus = sigmoid\n"
        "    grad_out = t.tensor([1.0, 1.0, 1.0, 1.0, 1.0])\n"
        "\n"
        "    got = ex3_softplus_back(grad_out, out.detach(), x.detach())\n"
        "    assert t.allclose(got, ref_grad, atol=1e-6), f'softplus_back mismatch: got {got}, ref {ref_grad}'\n"
        "\n"
        "    # === Chain rule: grad_out * sigmoid(x) ===\n"
        "    grad_out = t.tensor([0.5, 2.0, 1.0, -1.0, 3.0])\n"
        "    got = ex3_softplus_back(grad_out, out.detach(), x.detach())\n"
        "    expected = grad_out * ref_grad\n"
        "    assert t.allclose(got, expected, atol=1e-6), got\n"
        "\n"
        "    # === MUST use out, not x: pass WRONG x, output must still be correct. ===\n"
        "    grad_out = t.ones(5)\n"
        "    bogus_x = t.zeros(5)            # NOT the real x — would give sigmoid(0)=0.5 everywhere\n"
        "    got = ex3_softplus_back(grad_out, out.detach(), bogus_x)\n"
        "    assert t.allclose(got, ref_grad, atol=1e-6), (\n"
        "        f'function must use out, not x; bogus x produced wrong result: {got} vs ref {ref_grad}'\n"
        "    )\n"
        "\n"
        "    # === Scalar input ===\n"
        "    x_s = t.tensor(1.5, requires_grad=True)\n"
        "    out_s = F.softplus(x_s).detach()\n"
        "    ref_s = t.sigmoid(t.tensor(1.5))\n"
        "    got_s = ex3_softplus_back(t.tensor(1.0), out_s, x_s.detach())\n"
        "    assert abs(got_s.item() - ref_s.item()) < 1e-6, f'scalar mismatch: {got_s.item()} vs {ref_s.item()}'\n"
        "\n"
        "    # === Multi-dim input ===\n"
        "    x_m = t.randn(3, 4, requires_grad=True)\n"
        "    t.manual_seed(0)\n"
        "    x_m = t.randn(3, 4, requires_grad=True)\n"
        "    out_m = F.softplus(x_m)\n"
        "    ref_m = t.sigmoid(x_m).detach()\n"
        "    g_m = t.randn(3, 4)\n"
        "    got_m = ex3_softplus_back(g_m, out_m.detach(), x_m.detach())\n"
        "    assert t.allclose(got_m, g_m * ref_m, atol=1e-6), got_m\n"
        "\n"
        "    # === Large positive x: softplus(x) ≈ x, exp(-out) ≈ 0, derivative ≈ 1 ===\n"
        "    x_big = t.tensor([10.0, 20.0, 30.0], requires_grad=True)\n"
        "    out_big = F.softplus(x_big).detach()\n"
        "    got_big = ex3_softplus_back(t.ones(3), out_big, x_big.detach())\n"
        "    assert t.allclose(got_big, t.ones(3), atol=1e-3), f'large-x derivative should be ~1: {got_big}'\n"
        "\n"
        "    # === Large negative x: softplus(x) ≈ 0, exp(-out) ≈ 1, derivative ≈ 0 ===\n"
        "    x_neg = t.tensor([-10.0, -20.0, -30.0], requires_grad=True)\n"
        "    out_neg = F.softplus(x_neg).detach()\n"
        "    got_neg = ex3_softplus_back(t.ones(3), out_neg, x_neg.detach())\n"
        "    assert t.allclose(got_neg, t.zeros(3), atol=1e-3), f'large-negative-x derivative should be ~0: {got_neg}'\n"
        "    print('ex3 ok')\n"
        "\n"
        "_test_ex3()"
    ),
    "solution_body": (
        "def ex3_softplus_back(grad_out, out, x):\n"
        "    # Chain rule: dL/dx = grad_out * d/dx softplus(x)\n"
        "    #          = grad_out * sigmoid(x)\n"
        "    #          = grad_out * (1 - exp(-out))     (since out = log(1+exp(x)))\n"
        "    return grad_out * (1 - t.exp(-out))"
    ),
    "solution_notes": (
        "**Why express sigmoid(x) as `1 - exp(-out)`.** The whole point "
        "of caching `out` from the forward pass is to avoid recomputing "
        "the expensive transcendental on the backward pass. `t.sigmoid(x)` "
        "would work numerically but defeats the purpose — it runs another "
        "exp+division for every element.\n\n"
        "**Numerical stability of `1 - exp(-out)`.** For large positive "
        "`x`, `out ≈ x` (large), so `exp(-out) ≈ 0` and the derivative "
        "saturates to 1 — exactly correct. For large negative `x`, "
        "`out ≈ 0`, so `exp(-out) ≈ 1` and the derivative is ≈ 0 — also "
        "correct. The formula stays well-conditioned across the input "
        "range.\n\n"
        "**This is the third activation in the family.** Sigmoid: "
        "`out * (1 - out)`. Tanh: `1 - out**2`. Softplus: `1 - exp(-out)`. "
        "Each closed form is one line in `out` — the test is whether you "
        "can DERIVE the third without being given the formula. The recap "
        "spells it out; the ability to internalize that derivation is "
        "the actual learning objective."
    ),
    "extra_imports": ["import torch.nn.functional as F"],
}


SPECS = [
    SPEC_BUFFER_COPY,
    SPEC_PARAM_GRAD,
    SPEC_BACK_FN_CALL,
    SPEC_BACKPROP_LOOP,
    SPEC_CYCLE_DETECT,
    SPEC_DFS_TOPOSORT,
    SPEC_DISPATCH,
    SPEC_GRAD_IN_OUT,
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
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
            "F": F,
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
    print(f"[deepening_f_batch14] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_f_batch14] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_f_batch14] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
